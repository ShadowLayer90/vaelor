"""Enumerate this machine's network interfaces, and say how - or why not.

The inventory used to have exactly one way to answer this question: run
``ip -j address show`` and parse its JSON. Every failure of that one command -
binary absent, non-zero exit, unparseable output - produced the same value, an
empty list, and the interface downstream printed *"No network interfaces were
reported"*.

**On a machine that is serving that sentence over HTTP, the sentence is never
true.** It was being shown on a workstation answering on ``enp193s0`` at
192.168.4.58. An empty list is not the absence of interfaces; it is the absence
of an answer, and the two were indistinguishable in the payload.

The likely cause on that machine is in this repository, not on it.
``deploy/systemd/vaelor-control-plane.service`` sets::

    RestrictAddressFamilies=AF_UNIX AF_INET AF_INET6

``ip`` enumerates through **AF_NETLINK**, which is not in that list, so the
seccomp filter fails its ``socket()`` call and it exits with *"Cannot open
netlink socket: Address family not supported by protocol"* and no stdout.
``vaelor-hardware-bridge.service`` already adds ``AF_NETLINK`` for this reason;
the control plane does not. It fits every symptom, including the one the
screenshot volunteers: the connectivity probe in the same card still works,
because it uses AF_INET.

Two things follow, and this module does both:

1. **Never depend on one privileged path for a fact the kernel publishes in
   plain files.** ``/sys/class/net`` needs no netlink and no privilege, and
   carries name, state, MTU, MAC, and every byte counter the card shows.
   Addresses come from an ``AF_INET`` ioctl and ``/proc/net/if_inet6``, neither
   of which is netlink either.
2. **Say which source answered, and why the better one did not.** A degraded
   answer that cannot be recognised as degraded is worse than no answer.
"""

from __future__ import annotations

import json
import socket
import struct
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional

#: ``SIOCGIFADDR`` and ``SIOCGIFNETMASK``. These are the ioctls ``ifconfig``
#: used before ``ip`` existed, and they answer over an ordinary AF_INET socket.
_SIOCGIFADDR = 0x8915
_SIOCGIFNETMASK = 0x891B

#: The same bound the previous implementation applied. A machine with hundreds
#: of container veth pairs must not turn one inventory call into an unbounded
#: payload.
MAX_INTERFACES = 32

#: Reported in ``source`` so a reader can tell a full answer from a partial one.
SOURCE_IP = "ip"
SOURCE_SYSFS = "sysfs"
SOURCE_NONE = ""


def _text(path: Path) -> str:
    try:
        return path.read_text(encoding="utf-8", errors="replace").strip()
    except (OSError, ValueError):
        return ""


def _counter(path: Path) -> int:
    try:
        return int(_text(path))
    except (TypeError, ValueError):
        return 0


def _prefix_from_netmask(mask: str) -> int:
    try:
        return bin(struct.unpack("!I", socket.inet_aton(mask))[0]).count("1")
    except (OSError, struct.error, ValueError):
        return 0


def _ipv4_address(name: str) -> List[Dict[str, Any]]:
    """The interface's IPv4 address, over a socket family that is permitted.

    ``fcntl`` does not exist off Linux, and an interface with no address raises
    ``OSError``. Both mean "no address to report", which is a fact about this
    interface and not a failure of the enumeration.
    """
    try:
        import fcntl
    except ImportError:
        return []
    request = struct.pack("256s", name.encode("utf-8")[:15])
    try:
        with socket.socket(socket.AF_INET, socket.SOCK_DGRAM) as probe:
            address = socket.inet_ntoa(
                fcntl.ioctl(probe.fileno(), _SIOCGIFADDR, request)[20:24]
            )
            mask = socket.inet_ntoa(
                fcntl.ioctl(probe.fileno(), _SIOCGIFNETMASK, request)[20:24]
            )
    except (OSError, ValueError, struct.error):
        return []
    return [{"family": "inet", "address": address,
             "prefix": _prefix_from_netmask(mask)}]


def _ipv6_addresses(proc_root: Path) -> Dict[str, List[Dict[str, Any]]]:
    """IPv6 addresses per interface, from the file the kernel already writes.

    ``/proc/net/if_inet6`` is one line per address: 32 hex characters, the
    interface index, the prefix length in hex, the scope, the flags, and the
    device name.
    """
    found: Dict[str, List[Dict[str, Any]]] = {}
    for line in _text(proc_root / "net/if_inet6").splitlines():
        fields = line.split()
        if len(fields) < 6 or len(fields[0]) != 32:
            continue
        grouped = ":".join(fields[0][index:index + 4] for index in range(0, 32, 4))
        try:
            address = socket.inet_ntop(
                socket.AF_INET6, socket.inet_pton(socket.AF_INET6, grouped)
            )
            prefix = int(fields[2], 16)
        except (OSError, ValueError):
            continue
        found.setdefault(fields[5], []).append(
            {"family": "inet6", "address": address, "prefix": prefix}
        )
    return found


def _from_ip_json(payload: Any, net_class: Path) -> List[Dict[str, Any]]:
    interfaces = []
    for item in list(payload)[:MAX_INTERFACES]:
        if not isinstance(item, dict):
            continue
        name = str(item.get("ifname", ""))
        stats = net_class / name / "statistics"
        interfaces.append({
            "name": name,
            "state": str(item.get("operstate", "UNKNOWN")).lower(),
            "mtu": item.get("mtu"),
            "mac": item.get("address", ""),
            "addresses": [
                {"family": info.get("family"), "address": info.get("local"),
                 "prefix": info.get("prefixlen")}
                for info in list(item.get("addr_info", []))[:16]
            ],
            "rx_bytes": _counter(stats / "rx_bytes"),
            "tx_bytes": _counter(stats / "tx_bytes"),
            "rx_errors": _counter(stats / "rx_errors"),
            "tx_errors": _counter(stats / "tx_errors"),
        })
    return interfaces


def _from_sysfs(net_class: Path, proc_root: Path) -> List[Dict[str, Any]]:
    try:
        entries = sorted(net_class.iterdir())
    except (OSError, ValueError):
        return []
    ipv6 = _ipv6_addresses(proc_root)
    interfaces = []
    for entry in entries[:MAX_INTERFACES]:
        name = entry.name
        stats = entry / "statistics"
        interfaces.append({
            "name": name,
            "state": (_text(entry / "operstate") or "unknown").lower(),
            "mtu": _counter(entry / "mtu") or None,
            "mac": _text(entry / "address"),
            "addresses": _ipv4_address(name) + ipv6.get(name, []),
            "rx_bytes": _counter(stats / "rx_bytes"),
            "tx_bytes": _counter(stats / "tx_bytes"),
            "rx_errors": _counter(stats / "rx_errors"),
            "tx_errors": _counter(stats / "tx_errors"),
        })
    return interfaces


def _failure_detail(result: Any, found: str) -> str:
    """Why ``ip`` did not answer, in the words it used where it gave any."""
    if not found:
        return "iproute2 is not installed, so `ip` could not be run"
    stderr = " ".join(str(getattr(result, "stderr", "") or "").split())[:200]
    if stderr:
        return "`ip` failed: {}".format(stderr)
    return "`ip` returned nothing that could be read as an interface list"


def interface_inventory(
    command: Callable[[List[str]], Any],
    finder: Callable[[str], Optional[str]],
    *,
    net_class: Path = Path("/sys/class/net"),
    proc_root: Path = Path("/proc"),
) -> Dict[str, Any]:
    """Interfaces and routes, with the provenance of both.

    ``collected`` is the key that matters. It is the difference between "this
    machine has no interfaces" - which is false on anything answering a request
    - and "Vaelor could not read them", and it is the distinction the payload
    could not previously express at all.
    """
    routes: List[Any] = []
    interfaces: List[Dict[str, Any]] = []
    source = SOURCE_NONE
    detail = ""
    address = None
    found = finder("ip")
    if found:
        address = command([found, "-j", "address", "show"])
        try:
            payload = json.loads(getattr(address, "stdout", "") or "")
        except (json.JSONDecodeError, TypeError, ValueError):
            payload = None
        if isinstance(payload, list) and payload:
            interfaces = _from_ip_json(payload, net_class)
            source = SOURCE_IP
            try:
                routes = json.loads(
                    getattr(command([found, "-j", "route", "show"]), "stdout", "") or ""
                )[:MAX_INTERFACES]
            except (json.JSONDecodeError, TypeError, ValueError):
                routes = []
    if not interfaces:
        # The fallback is not a lesser answer for the card's purposes: name,
        # state, addresses, and counters are every field it renders.
        why = _failure_detail(address, found or "")
        interfaces = _from_sysfs(net_class, proc_root)
        if interfaces:
            source = SOURCE_SYSFS
            detail = (
                "Interface details were read from the kernel's own files "
                "because {}.".format(why)
            )
        else:
            detail = (
                "Vaelor could not read this machine's network interfaces: "
                "{}, and {} could not be listed either. This is a reporting "
                "failure, not an absence of networking."
            ).format(why, net_class)
    return {
        "interfaces": interfaces,
        "routes": routes,
        "collected": bool(interfaces),
        "source": source,
        "detail": detail,
    }
