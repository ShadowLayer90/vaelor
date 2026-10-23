"""The one answer to "which names and addresses does the console certificate carry" (VD-212).

Before VD-212 two places derived it two ways: the installer's ``_cert_san_list``
(every global IPv4, the hostname, its FQDN, localhost, 127.0.0.1) and
:func:`vaelor.host_desktop_tls.certificate_identities` (the routed address,
``hostname.<search domain>``, the resolver's canonical names). This module is
their union, plus ``hostname.local`` (mDNS), and is the only place either is
read for the household authority (LESSONS 6: three readers once derived the
controller's certificate three ways).

Everything here is read from the kernel and the resolver by the root authority,
never from a file the control plane can write: a name added to the certificate
is a name the household root vouches for.

The household root is name-constrained to the private LAN (owner, VD-212 build
answers), so a reading can hold names the root may not sign. Those are dropped
and **reported by name** (LESSONS 1: a certificate quietly missing the address
an owner types is a warning nobody can explain).

**Container and VM plumbing is not an address the console is reached on**
(VD-133 1.5.5 #1). Docker's bridges (``docker0``, ``docker_gwbridge``,
``br-<12 hex>``), their ``veth`` pairs and libvirt's ``virbr*`` carry gateway
addresses (172.17.0.1, ...) that answer only from this machine's own containers.
Reading them as identity put them in the certificate, and because Docker makes
and removes them on its own schedule - ``docker swarm init`` at controller
setup creates ``docker_gwbridge``, an app deploy creates a ``br-*`` - every one
changed the identity and re-issued the console certificate under every open
tab (LESSONS 6: the reading answered "which addresses does this machine hold"
when the question was "which addresses is the console reached on"). An
address is left out only when :func:`vaelor.network_interfaces.is_host_plumbing`
says so - a plumbing NAME on a SOFTWARE link, the one rule every reader of
"which interface is plumbing" shares - so a physical NIC, an owner's own ``br0``
bridging the LAN, a bond, a VLAN or a WireGuard link always stays. IPv6
addresses on plumbing are left out too, wherever they are named.
"""

from __future__ import annotations

import ipaddress
import re
import socket
import subprocess
from dataclasses import dataclass, field
from typing import Callable, Iterable, List, Sequence, Tuple

from . import host_desktop_tls, tls_public_names
from .cluster_link import RFC1918_NETWORKS
from .network_interfaces import is_host_plumbing

#: The private LAN the root may vouch for (owner, VD-212: "No, private LAN
#: only" - Tailscale's 100.64.0.0/10 is deliberately absent).
#: The RFC 1918 ranges come from their one home, cluster_link (LESSONS 6).
PRIVATE_LAN_NETWORKS = RFC1918_NETWORKS + tuple(ipaddress.ip_network(value) for value in (
    "127.0.0.0/8",
    "fc00::/7",
    "::1/128",
))
#: Local DNS names the root may vouch for, each with every name below it. The
#: machine's bare hostname at creation is added to these in the root itself.
PRIVATE_DNS_SUFFIXES = ("local", "lan", "home.arpa", "internal", "localhost")

_LABEL = re.compile(r"[a-z0-9](?:[a-z0-9-]{0,61}[a-z0-9])?")
_IP_TIMEOUT_SECONDS = 5

Runner = Callable[..., "subprocess.CompletedProcess[str]"]


@dataclass(frozen=True)
class IdentityReading:
    """One reading of the machine's names and addresses.

    ``complete`` is False when a source could not be read. An incomplete
    reading still yields a certificate, but it never counts as evidence that
    the identity changed (LESSONS 8: the observer's failure is not the
    machine's state).
    """

    names: Tuple[str, ...]
    addresses: Tuple[str, ...]
    complete: bool = True
    notes: Tuple[str, ...] = field(default=())


def bare_hostname(hostname: str | None = None) -> str:
    """The hostname's first label, lower-cased, or "" when it is not a DNS label."""
    value = (socket.gethostname() if hostname is None else hostname) or ""
    label = value.strip().rstrip(".").split(".", 1)[0].lower()
    return label if _LABEL.fullmatch(label) and label != "localhost" else ""


#: One global address as ``ip -o addr`` lists it: (interface, family, address, prefix).
InterfaceAddress = Tuple[str, str, str, int]


def interface_addresses(run: Runner = subprocess.run) -> Tuple[List[InterfaceAddress], str]:
    """Every global-scope address with its interface and prefix length.

    Returns (rows, failure); ``failure`` is "" when the reading worked. The one
    parser of ``ip -o addr`` here: the identity reading and the console
    address the trust steps name both read it.
    """
    try:
        result = run(
            ["ip", "-o", "addr", "show", "scope", "global"],
            capture_output=True, text=True, timeout=_IP_TIMEOUT_SECONDS,
            check=False,
        )
    except (OSError, subprocess.SubprocessError) as error:
        return [], "the interface addresses could not be read ({})".format(
            type(error).__name__)
    if result.returncode != 0:
        return [], "`ip addr` exited {}".format(result.returncode)
    rows: List[InterfaceAddress] = []
    for line in result.stdout.splitlines():
        fields = line.split()
        family = "inet" if "inet" in fields else "inet6" if "inet6" in fields else ""
        position = fields.index(family) + 1 if family else 0
        if not family or position >= len(fields):
            continue
        address, _, prefix = fields[position].partition("/")
        rows.append((fields[1] if len(fields) > 1 else "", family, address,
                     int(prefix) if prefix.isdigit() else 128))
    return rows, ""


def global_ipv4_addresses(
    run: Runner = subprocess.run,
    host_internal: Callable[[str], bool] = is_host_plumbing,
) -> Tuple[List[str], List[str], str]:
    """Every global-scope IPv4 address, split by whether the console is reached on it.

    Returns (addresses, plumbing_addresses, failure). ``failure`` is "" when
    the reading worked; an empty first list with no failure is a machine with
    no global address. The second list is every address, IPv4 and IPv6, on
    container and VM plumbing (:func:`vaelor.network_interfaces.is_host_plumbing`):
    never identity, and :func:`read_identity` also drops them when another
    source names them (adversarial review F8). IPv6 addresses are otherwise not
    taken from here: a temporary SLAAC address rotates, and as identity it
    would re-issue the certificate on its own schedule.
    """
    rows, failure = interface_addresses(run)
    if failure:
        return [], [], failure
    found: List[str] = []
    internal: List[str] = []
    for interface, family, address, _prefix in rows:
        if host_internal(interface):
            internal.append(address)
        elif family == "inet":
            found.append(address)
    return found, internal, ""


def read_identity(
    *,
    certificate_identities: Callable[[], Tuple[List[str], List[str]]] = (
        host_desktop_tls.certificate_identities),
    global_ipv4: Callable[[], Tuple[List[str], List[str], str]] = global_ipv4_addresses,
    hostname: Callable[[], str] = socket.gethostname,
) -> IdentityReading:
    """Read every name and address an owner could type to reach this machine.

    Sorted and de-duplicated, so two readings of an unchanged machine compare
    equal whatever order the sources answered in. An address on container or VM
    plumbing is left out whichever source named it - the resolver can return
    every local address for the machine's own name (nss-myhostname).
    """
    names, addresses = certificate_identities()
    notes: List[str] = []
    complete = True
    extra, internal, failure = global_ipv4()
    if failure:
        complete = False
        notes.append(failure)
    label = bare_hostname(hostname())
    candidate_names = list(names) + ([label, label + ".local"] if label else [])
    candidate_names.append("localhost")
    plumbing = set(_unique_addresses(internal))
    return IdentityReading(
        names=_unique_names(candidate_names),
        addresses=tuple(
            a for a in _unique_addresses(list(addresses) + list(extra) + ["127.0.0.1"])
            if a not in plumbing),
        complete=complete,
        notes=tuple(notes),
    )


def _unique_names(values: Iterable[str]) -> Tuple[str, ...]:
    cleaned = set()
    for value in values:
        name = str(value or "").strip().rstrip(".").lower()
        if name and all(_LABEL.fullmatch(part) for part in name.split(".")):
            cleaned.add(name)
    return tuple(sorted(cleaned))


def _unique_addresses(values: Iterable[str]) -> Tuple[str, ...]:
    parsed = set()
    for value in values:
        try:
            parsed.add(ipaddress.ip_address(str(value or "").split("%", 1)[0]))
        except ValueError:
            continue
    return tuple(str(item) for item in sorted(parsed, key=lambda a: (a.version, a)))


def dns_name_permitted(name: str, permitted: Sequence[str]) -> bool:
    """True when ``name`` is one of ``permitted`` or a name below one."""
    lowered = name.lower().rstrip(".")
    for subtree in permitted:
        base = subtree.lower().strip(".")
        if lowered == base or lowered.endswith("." + base):
            return True
    return False


def address_permitted(address: str, networks: Sequence) -> bool:
    parsed = ipaddress.ip_address(address)
    return any(parsed.version == net.version and parsed in net for net in networks)


def filter_identity(
    reading: IdentityReading,
    permitted_dns: Sequence[str],
    permitted_networks: Sequence = PRIVATE_LAN_NETWORKS,
) -> Tuple[Tuple[str, ...], Tuple[str, ...], Tuple[str, ...]]:
    """Split a reading into what the root may sign and what it may not.

    Returns (names, addresses, dropped); every dropped entry names the value
    and why, so the caller can report it rather than shrink silently.
    """
    names: List[str] = []
    dropped: List[str] = []
    for name in reading.names:
        refusal = ""
        if "." not in name and name not in PRIVATE_DNS_SUFFIXES:
            # A bare name that is a public TLD is refused even when an older
            # root's constraints would admit it (VD-212 review: `dev`).
            refusal = tls_public_names.bare_label_refusal(name)
        if not refusal and not dns_name_permitted(name, permitted_dns):
            refusal = "not a local name this authority may vouch for"
        if refusal:
            dropped.append("{} ({})".format(name, refusal))
        else:
            names.append(name)
    addresses = tuple(
        a for a in reading.addresses if address_permitted(a, permitted_networks))
    dropped += ["{} (not a private LAN address)".format(a)
                for a in reading.addresses if a not in addresses]
    return tuple(names), addresses, tuple(dropped)
