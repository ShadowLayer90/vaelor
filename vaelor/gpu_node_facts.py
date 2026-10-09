"""Facts read off a GPU node over the transport, each a parser plus a thin read.

`gpu_pool_runtime` owns the docker, vLLM and Ray invocations; this module owns
the questions those invocations need answered ABOUT the box they will run on.
Two live here, and both exist for the same reason: they DIFFER between
appliances and may never be assumed.

* **The numeric render/video GIDs** (`resolve_group_ids`). A container image
  carries no host group database, so ``--group-add render`` cannot resolve
  inside it; the grant must be numeric, and the number is the host's.
* **The cluster network interface** (`interface_for_address`). RCCL and Gloo
  bind a NIC by name, and the two Strix Halo appliances name their wired links
  differently - one ``enp…`` (a PCI-addressed NIC), one ``enx…`` (a MAC-named
  one). NCCL will accept a prefix; Gloo wants the exact name, so no single
  operator-typed value can be right for both, and a deploy that pushed one to
  every node bound the wrong link on half of them. The node is asked instead -
  but NOT for its default route (VD-125). A default route is a lab-shaped
  guess: a VPN or tunnel (``tailscale0``, ``wg0``), a docker/libvirt bridge or
  a second uplink can hold it, and binding that link makes the collective hang
  rather than fail. The variation-proof fact is the NIC that carries THE
  ADDRESS THE CLUSTER REACHES THE NODE ON - the longest-prefix connected route
  containing it, the default route excluded.

The cluster interface is captured into a node's inventory at enrolment
(`ssh_transport.probe`), and the deploy prefers that record to reading the node
- but it is the THIRD answer, not the first. An operator's per-node pin and
then a fleet-wide pin outrank it, and the record itself is trusted only while
the address it was captured against is still the address the deploy advertises
for the node; a live read follows whenever none of those three holds, which is
a record written before the fact existed AND a record whose address has since
moved. `cluster_interfaces`, below, is where that order lives.
The GIDs are NOT captured there: the deploy resolves them per node, up front.
Nor is a box with no NIC on the cluster address refused at join - enrolment is
generic (CPU-only Swarm nodes join the Swarm too), so `probe` records an empty
name with the reason why, the fleet card surfaces it, and the refusal is the GPU
DEPLOY's: made up front and by node name, before any image, weights or unit.

**The captured fact is keyed on the SSH CREDENTIAL's host, and it says so.**
`ssh_transport._cluster_address` can only know the address this control plane
manages the node on; the deploy asks `advertise_address` for the address the
cluster will use. On a dual-homed box those are the same address, and on one
managed over a separate management NIC they are not. The deploy already handles
the difference correctly - it compares the two and falls back to a live read
against the address the cluster will actually use - so nothing binds a
management link. What was left is the REPORT: the fleet card showed a bare NIC
name, so an operator reading "enp1s0" could not tell it was the management NIC
and would be told nothing about the link the collective would really bind. The
fix chosen is to label the fact with its address rather than to guess a second
one: the transport cannot know the cluster's advertised address (it is the
deploy's input, not the node's), and inventing one would be the "lab-shaped
guess" this module exists to refuse. So the record carries ``address`` beside
``name``, the card renders "enp1s0 (192.0.2.11)", and a mismatch with the
address the operator knows the cluster uses is visible on the machine itself.

Every rule here is a pure function over text, so the whole surface is testable
without a node, and each read is one allowlisted ``cat`` through
`ssh_transport.SshTransport.run`. `interface_name` is the ONE home of the
interface-name rule: the runtime validates the value it is handed by calling
it rather than keeping a second copy that could drift.
"""

from __future__ import annotations

import ipaddress
import re
from pathlib import Path
from typing import Any, Callable, Dict, Iterable, List, Optional, Tuple

#: The host groups that own the accelerator character devices a GPU container
#: must be granted. Names here, numbers on the node - see `parse_group_ids`.
GPU_DEVICE_GROUPS = ("render", "video")

#: A NIC's physical medium, as the captured fact spells it. Two real answers and
#: an honest third: ``wireless`` when the kernel's own ``DEVTYPE=wlan`` is on the
#: link, ``wired`` when no such signal is present AND the link reports a real
#: Ethernet speed, and ``unknown`` otherwise - a link that is down, or a uevent
#: and a speed neither of which could be read - never a bare default either way.
MEDIUM_WIRED = "wired"
MEDIUM_WIRELESS = "wireless"
MEDIUM_UNKNOWN = "unknown"

#: The kernel's positive wireless signal on a netdev's own ``uevent``. A real
#: Wi-Fi NIC's 802.11 bitrate is NOT in sysfs ``speed`` (that file is an
#: Ethernet MII notion), so the medium is read off THIS line rather than guessed
#: from the speed number - and its ABSENCE is, on its own, not a wired verdict.
_UEVENT_WIRELESS = "DEVTYPE=wlan"

#: The columns `interface_for_address` reads out of a route row:
#: ``Iface Destination Gateway Flags RefCnt Use Metric Mask MTU Window IRTT``.
#: A row shorter than the mask column cannot be a route line (the header
#: included), so it is skipped rather than parsed.
_ROUTE_IFACE_COLUMN = 0
_ROUTE_DESTINATION_COLUMN = 1
_ROUTE_GATEWAY_COLUMN = 2
_ROUTE_FLAGS_COLUMN = 3
_ROUTE_METRIC_COLUMN = 6
_ROUTE_MASK_COLUMN = 7

#: ``RTF_GATEWAY`` from the kernel's ``route.h``: this row is reached THROUGH a
#: next hop rather than being on the link. Both the gateway word and this flag
#: are checked, because either one alone can be the truthful column.
_RTF_GATEWAY = 0x0002

#: A ``/proc/net/route`` hex word: the kernel writes these columns as exactly
#: eight hex digits, so anything else is the header row or junk.
_ROUTE_WORD = re.compile(r"[0-9A-Fa-f]{8}")


def interface_name(value: str) -> str:
    """The NIC name RCCL/Gloo bind to, validated to a Linux interface shape.

    RCCL picks its own route otherwise, which on a multi-NIC appliance can be
    the wrong link; pinning it to the private cluster NIC is the documented fix.
    Fifteen characters is the kernel's own ceiling (``IFNAMSIZ`` less its NUL),
    so a longer value never named a real link.
    """
    interface = str(value).strip()
    if not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9._-]{0,14}", interface):
        raise ValueError("Choose a valid private-network interface name.")
    return interface


def parse_group_ids(
    text: str, names: Iterable[str] = GPU_DEVICE_GROUPS
) -> List[int]:
    """The numeric GIDs of ``names`` read out of an ``/etc/group`` dump, or raise.

    A container image has no host group database, so ``--group-add render`` fails
    with "no matching entries in group file"; the GIDs must be numeric and differ
    per node (990/44 on the current boxes, never assumed). Pure, so it is testable
    without a transport, and it refuses rather than start a blind container.
    """
    wanted = list(names)
    found: Dict[str, int] = {}
    for line in str(text).splitlines():
        fields = line.split(":")
        if len(fields) < 3:
            continue
        if fields[0] in wanted and fields[2].strip().isdigit():
            found.setdefault(fields[0], int(fields[2].strip()))
    unresolved = [name for name in wanted if name not in found]
    if unresolved:
        raise ValueError(
            "Could not resolve the {} group(s) to numeric GIDs on this node, so "
            "the vLLM container cannot be granted the accelerator "
            "devices.".format(", ".join(unresolved))
        )
    return [found[name] for name in wanted]


def _route_word(field: str) -> Optional[int]:
    """One ``/proc/net/route`` address column as an integer, or None.

    The kernel writes these columns in the host's byte order - LITTLE-endian on
    every appliance Vaelor runs on - so 10.20.30.0 is spelled ``001E140A`` and
    a /24 mask ``00FFFFFF``. Nothing here un-swaps them: the address being
    matched is packed the same way in `interface_for_address`, and a mask, an
    AND and a popcount all mean the same thing in either order. Returns None
    for anything that is not exactly eight hex digits, which is how the header
    row and any junk line drop out - the width is the kernel's own, so a shorter
    "word" was never one of these columns.
    """
    text = str(field).strip()
    if not _ROUTE_WORD.fullmatch(text):
        return None
    return int(text, 16)


def interface_for_address(route_text: str, address: str) -> str:
    """The NIC that carries ``address`` (`connected_route`, the name alone)."""
    return connected_route(route_text, address)[0]


def connected_route(route_text: str, address: str) -> Tuple[str, int]:
    """``(NIC, prefix length)`` of the link carrying ``address``, off ``/proc/net/route``.

    The prefix rides with the name because the cluster link (VD-162) needs the
    NETWORK an address sits on, and that is this same row's mask: a second
    parser for it could disagree with this one about which link carries an
    address.

    This is the cluster interface (VD-125), and it is deliberately NOT the
    default route: a VPN or tunnel (``tailscale0``, ``wg0``), a docker or
    libvirt bridge, or a second uplink can hold the default route on a box that
    reaches the cluster over an entirely different NIC, and RCCL/Gloo bound to
    that link hang instead of failing. What is true on every variation is that
    the address the cluster talks to the node on sits on exactly one link, and
    the kernel's own table says which: the connected route whose
    ``Destination`` equals ``address`` masked by its ``Mask``.

    Only CONNECTED routes are candidates: a row is one when its ``Gateway`` word
    is zero AND its ``Flags`` lack ``RTF_GATEWAY``, which is what "the address
    sits on this link" means. Skipping the default route alone is not enough -
    a VPN split tunnel or a static route can put the CLUSTER's own /24 on
    ``wg0`` via a next hop, and at a lower metric it ties on prefix and wins,
    so the collective binds the tunnel and hangs. Among the connected rows the
    LONGEST mask wins - a /24 on the cluster NIC beats a /16 someone left on a
    second link - and an exact tie is broken by the lowest ``Metric``, the
    kernel's own preference. Pure, and it refuses rather than hand back a guess.

    ``/proc/net/route`` is the MAIN table only: a link reachable solely through
    policy routing or another table is invisible to this fact, and such a node
    answers "no interface" rather than a wrong one.
    """
    try:
        wanted = int.from_bytes(
            ipaddress.IPv4Address(str(address).strip()).packed, "little"
        )
    except (ipaddress.AddressValueError, ValueError) as error:
        raise ValueError(
            "A node's cluster address must be an IPv4 address; got "
            "{!r}.".format(address)
        ) from error

    best: Optional[Tuple[int, int, str]] = None
    for line in str(route_text).splitlines():
        fields = line.split()
        if len(fields) <= _ROUTE_MASK_COLUMN:
            continue
        destination = _route_word(fields[_ROUTE_DESTINATION_COLUMN])
        mask = _route_word(fields[_ROUTE_MASK_COLUMN])
        gateway = _route_word(fields[_ROUTE_GATEWAY_COLUMN])
        if destination is None or mask is None or gateway is None:
            continue
        try:
            # The one DECIMAL column of the four; the rest are hex. `Flags` is
            # hex but narrower than a route word, so it is read on its own.
            metric = int(fields[_ROUTE_METRIC_COLUMN])
            flags = int(fields[_ROUTE_FLAGS_COLUMN], 16)
        except ValueError:
            continue
        if destination == 0:
            continue
        if gateway or flags & _RTF_GATEWAY:
            continue
        if wanted & mask != destination:
            continue
        candidate = (bin(mask).count("1"), -metric, fields[_ROUTE_IFACE_COLUMN])
        if best is None or candidate[:2] > best[:2]:
            best = candidate
    if best is None:
        raise ValueError(
            "No network interface on the node carries the cluster address {}; "
            "refresh the node or pin its cluster interface.".format(address)
        )
    return interface_name(best[2]), best[0]


def resolve_group_ids(
    transport, names: Iterable[str] = GPU_DEVICE_GROUPS
) -> List[int]:
    """The node's numeric render/video GIDs, read from ``/etc/group``.

    Resolved per node, never assumed: the GIDs differ between boxes and a name
    is meaningless in the container. ``cat`` is allowlisted and needs no sudo.
    """
    return parse_group_ids(transport.run(["cat", "/etc/group"]), names)


def resolve_interface(transport, address: str) -> str:
    """The NIC carrying ``address`` on this node, read off its routing table.

    One allowlisted, unprivileged ``cat`` of the kernel's own routing table -
    no ``ip``/``route`` binary is on the transport's first-argv allowlist, and
    ``/proc/net/route`` is the same information without widening it. The name
    goes through `interface_name` before it can reach a container's
    ``NCCL_SOCKET_IFNAME``, so a node answering with something unexpected fails
    here rather than inside a started server. This is the LAST of the deploy's
    four answers: it runs when neither operator pin is set and the enrolled
    record is missing or was captured against a different address (see this
    module's docstring), never in preference to a fact that still holds.
    """
    return interface_for_address(
        transport.run(["cat", "/proc/net/route"]), address
    )


def link_speed_path(name: str) -> str:
    """The sysfs file holding ``name``'s link speed, name validated FIRST.

    `interface_name` is the ONE home of the interface-name rule, and calling it
    here is what makes a path-injection impossible: a value carrying a ``..`` or
    a slash is refused before it is ever interpolated into a path, so no read
    built on this can be pointed anywhere but at one real link's own files. The
    two boundaries that carry the ``cat`` (the SSH transport and the root
    bridge) pin the SAME shape independently; this is the client-side spelling.
    """
    return "/sys/class/net/{}/speed".format(interface_name(name))


def link_uevent_path(name: str) -> str:
    """The sysfs ``uevent`` for ``name``, its name validated exactly as above."""
    return "/sys/class/net/{}/uevent".format(interface_name(name))


def parse_link_speed(text: str) -> Tuple[Optional[int], str]:
    """``/sys/class/net/<name>/speed`` as Mb/s, or ``(None, reason)``.

    The file holds a positive integer for a link that is up and negotiated,
    ``-1`` for one that is down or whose rate the driver will not report, and on
    some drivers an ``EINVAL`` the caller has already turned into an empty read.
    Every non-positive or unreadable answer is UNKNOWN with a short reason,
    never a fabricated number: a link rate shown as a fact must have been read.
    """
    raw = str(text).strip()
    if not raw:
        return None, (
            "This link's speed could not be read from sysfs, so its rate is "
            "unknown."
        )
    try:
        value = int(raw)
    except ValueError:
        return None, (
            "This link's sysfs speed was not a number, so its rate is unknown."
        )
    if value <= 0:
        return None, (
            "This link reports no negotiated speed (it is down, or the driver "
            "publishes none), so its rate is unknown."
        )
    return value, ""


def parse_link_medium(
    uevent_text: str, speed_mbps: Optional[int]
) -> Tuple[str, str]:
    """``(medium, reason)`` from a NIC's ``uevent`` and its parsed speed.

    ``DEVTYPE=wlan`` on the link's own ``uevent`` is the kernel's positive
    wireless signal and settles the answer whatever the speed file says. With no
    such signal the medium is called ``wired`` only when there is a POSITIVE
    wired signal - a link reporting a real Ethernet speed - so a down link, or a
    ``uevent`` that could not be read, is ``unknown`` with a reason rather than
    a wired guess. Wired is never the bare default (VD-125's honest-degradation
    rule): the misleading static "2.5G / Wi-Fi" label this fact replaces was
    exactly such a guess.
    """
    for line in str(uevent_text).splitlines():
        if line.strip() == _UEVENT_WIRELESS:
            return MEDIUM_WIRELESS, ""
    if isinstance(speed_mbps, int) and speed_mbps > 0:
        return MEDIUM_WIRED, ""
    return MEDIUM_UNKNOWN, (
        "No wireless signal was found and no Ethernet link speed confirmed a "
        "wired connection, so this link's medium is unknown."
    )


def link_facts(speed_text: str, uevent_text: str) -> Dict[str, Any]:
    """The link fields the cluster-interface fact carries, from two sysfs reads.

    A pure function over the two file contents, so the whole speed/medium
    surface is testable without a node. The shape is uniform - the four keys are
    always present - and honest: an unreadable value is ``None``/``unknown`` with
    a companion ``*_reason`` saying why, and the reason is empty when the value
    was measured. Each reader (`ssh_transport` over the transport for a worker,
    the controller over its own ``/sys``) supplies the text; this owns what the
    text means, so both machines describe a link the same way.
    """
    speed_mbps, speed_reason = parse_link_speed(speed_text)
    medium, medium_reason = parse_link_medium(uevent_text, speed_mbps)
    return {
        "speed_mbps": speed_mbps,
        "speed_reason": speed_reason,
        "medium": medium,
        "medium_reason": medium_reason,
    }


def _read_sys_text(path: str) -> str:
    """One world-readable sysfs file as text, or ``""`` if it cannot be read.

    An ``OSError`` (a missing file, a link that has since vanished) becomes an
    empty read, which `link_facts` reports as an honest unknown - the controller
    must never crash discovery over a NIC file, exactly as the transport read is
    ``optional``.
    """
    try:
        return Path(path).read_text(encoding="utf-8", errors="replace")
    except OSError:
        return ""


def read_local_link(
    name: str, read_text: Optional[Callable[[str], str]] = None
) -> Dict[str, Any]:
    """This machine's OWN link fields for ``name``, read off local ``/sys``.

    The controller reads its own sysfs directly - the files are world-readable
    (``0444``) and the machine is right here - exactly as it reads its own
    routing table rather than taking a transport round trip. ``read_text`` is
    injectable so the read is testable without a real ``/sys``; by default it
    reads the validated sysfs path. Kept here, in the LAST home of the interface
    fact rules, so the controller adds no second copy of them to a module
    already at its line ceiling (VD-125).
    """
    reader = read_text or _read_sys_text
    return link_facts(
        reader(link_speed_path(name)), reader(link_uevent_path(name))
    )


def _read_proc_route() -> Optional[str]:
    """This machine's own ``/proc/net/route``, or ``None`` if it cannot be read.

    World-readable, so no privilege and no transport round trip is needed - the
    controller reads its own routing table exactly as it reads its own sysfs. An
    ``OSError`` (the file absent, e.g. off a non-Linux dev box) is ``None``, which
    the caller turns into an honest reason rather than a crash.
    """
    try:
        return Path("/proc/net/route").read_text(encoding="utf-8", errors="replace")
    except OSError:
        return None


def local_cluster_interface(
    address: str,
    read_route: Optional[Callable[[], Optional[str]]] = None,
    read_text: Optional[Callable[[str], str]] = None,
) -> Dict[str, Any]:
    """This machine's OWN cluster NIC for ``address``, in the probe's shape.

    The head's copy of the node fact `ssh_transport.probe` captures over SSH:
    the same routing-table rule (`interface_for_address`) applied to THIS
    machine's own ``/proc/net/route``, then the link's speed and medium off its
    own world-readable sysfs (`read_local_link`) - so the controller and an
    enrolled worker describe a link identically, from one home. Read live rather
    than stored because, unlike an enrolled node, this machine is right here.

    Honest and never guessed, exactly as the probe's is: an unreadable routing
    table or an address on no connected subnet leaves ``name`` empty and says
    why in ``reason``, and the GPU deploy is what turns that into a refusal, by
    node, before anything is started. ``read_route``/``read_text`` are injectable
    so the whole surface is testable without a real ``/proc`` or ``/sys``.
    """
    route = (read_route or _read_proc_route)()
    if route is None:
        return {
            "name": "", "address": address,
            "reason": (
                "This controller's own routing table could not be read, so the "
                "interface carrying its cluster address is unknown."
            ),
        }
    try:
        name = interface_for_address(route, address)
    except ValueError as error:
        return {"name": "", "address": address, "reason": str(error)}
    return {
        "name": name, "address": address,
        **read_local_link(name, read_text=read_text),
    }


def enrolled_interface(node: Dict[str, str], address: str) -> str:
    """The node's captured cluster NIC, when it still describes ``address``.

    Absent, empty (the probe recorded a reason instead of a guess) or captured
    against a different address, this answers "" and the caller reads the node
    live. The name is re-validated because it comes out of a stored record, not
    out of the parser that wrote it - and a stored name that fails the rule is
    refused BY NODE, exactly as the live read is, so the operator is told whose
    record to refresh rather than only that some name was unusable.
    """
    fact = (node.get("inventory") or {}).get("cluster_interface")
    if not isinstance(fact, dict):
        return ""
    name = str(fact.get("name", "") or "").strip()
    if not name or str(fact.get("address", "") or "").strip() != address:
        return ""
    try:
        return interface_name(name)
    except ValueError as error:
        raise ValueError("{}: {}".format(node["name"], error)) from error


def cluster_interfaces(
    participants, transports, node_pins, pinned, *,
    advertise_address, resolve,
) -> Tuple[Dict[str, str], Dict[str, str]]:
    """Every participant's cluster NIC, and where each answer came from.

    There is no fleet-wide answer to default to. RCCL/Gloo take an interface
    NAME, and the two Strix Halo appliances name their wired links differently
    (``enp…`` on one, ``enx…`` on the other); NCCL would accept a prefix but
    Gloo wants the exact name, so pushing one value to every node bound a wrong
    or absent link on the others.

    VD-125 settles where the answer comes from, in this order:

    1. ``payload["interfaces"][node_id]`` - a per-node operator pin.
    2. ``payload["interface"]`` - a fleet-wide operator pin.
    3. The node's own enrolled fact, ``inventory["cluster_interface"]``,
       accepted only when the address it was captured against is still the
       address this deploy advertises for the node. A node that moved subnets
       since enrolment has a stale name, and binding it would hang the
       collective rather than fail it.
    4. A live read (``resolve``), whenever none of the three above holds - a
       record enrolled before the fact existed, and equally one whose address
       has since moved.

    Sources are returned beside the names so the deployment record says which of
    the four each node's NIC came from, one rung per name: ``pinned-node``,
    ``pinned-fleet``, ``enrolled``, ``resolved``. The two pins are told apart
    because they are different operator acts and lead to different repairs - a
    per-node pin was typed for THIS machine, while a fleet pin was inherited by
    it and is the one that used to bind a wrong link on every box it was not
    typed for. A record that called both "pinned" could not answer which.
    ``advertise_address`` and ``resolve`` are passed in rather than imported:
    the deploy owns how a node's address is validated and how a transport is
    reached, and this owns only the order.
    """
    interfaces: Dict[str, str] = {}
    sources: Dict[str, str] = {}
    for node in participants:
        node_id = node["id"]
        pin = node_pins.get(node_id)
        if pin or pinned:
            interfaces[node_id] = pin or pinned
            sources[node_id] = "pinned-node" if pin else "pinned-fleet"
            continue
        # `advertise_address` refuses a host it cannot read as a private IPv4,
        # and its sentence is about an address, not about a machine. Prefix the
        # node exactly as the reads below do, so every refusal out of this loop
        # answers "which machine" first.
        try:
            address = advertise_address(node["host"])
        except ValueError as error:
            raise ValueError("{}: {}".format(node["name"], error)) from error
        enrolled = enrolled_interface(node, address)
        if enrolled:
            interfaces[node_id] = enrolled
            sources[node_id] = "enrolled"
            continue
        try:
            interfaces[node_id] = resolve(transports[node_id], address)
        except ValueError as error:
            raise ValueError("{}: {}".format(node["name"], error)) from error
        sources[node_id] = "resolved"
    return interfaces, sources
