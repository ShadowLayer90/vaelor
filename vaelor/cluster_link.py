"""The cluster link: which of this machine's own network links carries a split model.

VD-162. When one model is split across machines (the capacity intent), Ray and
the RCCL/Gloo collectives exchange activations between them. Which link that
traffic rides was, until now, whatever link carried the address the machine was
enrolled on. The lab pair has a second, much faster link between the two boxes
(Thunderbolt / USB4), and using it meant editing the launch by hand.

**The setting is a choice among THIS machine's own links - nothing is scanned.**
Discovery lists the links the kernel already has (``/sys/class/net``), with the
address and network each carries (``/proc/net/fib_trie`` and
``/proc/net/route``). No address is probed, no peer is searched for, and no AI
server is discovered (VD-137): the owner picks a link, and the machines that
take part in a split must each already hold an address on that link's network.

**What is derived from the choice, and where.** `split_bindings` answers, for
every machine in a split, the address it holds INSIDE the chosen link's network
and the interface carrying that address. Those two values are what the launch
renders as the Ray node address, ``VLLM_HOST_IP``, ``NCCL_SOCKET_IFNAME`` and
``GLOO_SOCKET_IFNAME`` (`gpu_pool_runtime`). A machine with no address on that
network refuses the deploy by name, before an image, weights or a unit.

**Two checks stand between a name and a privileged launch.** The name must pass
the one interface-name rule (`gpu_node_facts.interface_name`) AND be one of the
links discovery found on this machine. A stored name that no longer matches a
link is reported as such and refuses a split; it is never passed on.

**No link chosen** answers the enrolled address and the interface carrying
it (`gpu_node_facts.cluster_interfaces`): the split rides the machines'
Ethernet, fenced by socket owner (owner 2026-10-01, ACC-187, `gpu_ray_plane`).
Replicated serving runs no collective and never reads this module.

**Limits, stated rather than hidden.** IPv4 only: a link that carries only an
IPv6 address is listed as having no IPv4 address and cannot be chosen. And the
link is identified on every machine by its NETWORK, so two different links
that were given the same small network (two point-to-point /30s numbered
alike) cannot be told apart - each machine must have exactly one link on the
chosen network.

**A read that failed says so.** "This link has no address" and "this
machine's address tables could not be read" are different facts, and only the
first is a statement about the link.

Pure over text wherever a rule lives, so the whole surface is testable without
a node: the readers hand text in, and the rules never open a file.
"""

from __future__ import annotations

import ipaddress
import os
from typing import Any, Callable, Dict, Iterable, List, Mapping, Optional, Tuple

from . import gpu_node_facts
from .cluster_placement import CONTROLLER_PLACEMENT_ID

NET_ROOT = "/sys/class/net"
ROUTE_PATH = "/proc/net/route"
FIB_TRIE_PATH = "/proc/net/fib_trie"

#: The cluster-store key the choice is kept under. The cluster store is the one
#: state both the control plane (which records the choice) and the workload
#: executor (which deploys with it) already open.
SETTING_KEY = "cluster_link"

#: The exact ``confirm`` value a change to the choice must carry.
SET_CONFIRMATION = "set-cluster-link"

#: A link's kind. Three of the four are the medium words `gpu_node_facts`
#: already owns for the enrolled cluster interface, taken from there so the
#: two facts cannot spell a medium differently; Thunderbolt is the one kind
#: the medium rule has no word for, because it is read off the driver.
KIND_WIRED = gpu_node_facts.MEDIUM_WIRED
KIND_WIRELESS = gpu_node_facts.MEDIUM_WIRELESS
KIND_THUNDERBOLT = "thunderbolt"
KIND_UNKNOWN = gpu_node_facts.MEDIUM_UNKNOWN

#: The kernel driver behind a Thunderbolt / USB4 network link. Read off the
#: link's own ``device/driver``, never guessed from its name.
_THUNDERBOLT_DRIVERS = frozenset({"thunderbolt-net", "thunderbolt_net"})

#: Where a split deployment's interface answer came from, beside the four
#: `gpu_node_facts.cluster_interfaces` already records.
SOURCE_CLUSTER_LINK = "cluster-link"

#: The address ranges set aside for a private network (RFC 1918). Python's
#: ``is_private`` is wider than this - it also answers yes for the ranges
#: reserved for documentation and for benchmarking - and an address from one of
#: those on a real link is a mistake to say, not a network to bind (review).
_LAN_NETWORKS = tuple(
    ipaddress.ip_network(network)
    for network in ("10.0.0.0/8", "172.16.0.0/12", "192.168.0.0/16")
)


def lan_address(value: Any) -> bool:
    """Whether ``value`` is an IPv4 address in a private-network range."""
    try:
        address = ipaddress.IPv4Address(str(value))
    except (ipaddress.AddressValueError, ValueError):
        return False
    return any(address in network for network in _LAN_NETWORKS)


# --- text rules -----------------------------------------------------------

def local_ipv4_addresses(fib_trie_text: Any) -> List[str]:
    """The IPv4 addresses a machine holds, from its ``/proc/net/fib_trie``.

    The kernel lists every address of its own as a ``/32 host LOCAL`` leaf
    under the address line above it. The file repeats each table, so answers
    are de-duplicated in first-seen order. Loopback addresses are left out: no
    other machine can reach them.
    """
    found: List[str] = []
    last = ""
    for line in str(fib_trie_text or "").splitlines():
        text = line.strip()
        if text.startswith("|--"):
            last = text[3:].strip()
            continue
        if text.startswith("/32 host LOCAL") and last:
            try:
                address = ipaddress.IPv4Address(last)
            except (ipaddress.AddressValueError, ValueError):
                continue
            if not address.is_loopback and str(address) not in found:
                found.append(str(address))
    return found


def node_links(route_text: Any, fib_trie_text: Any) -> List[Dict[str, str]]:
    """``[{name, address, network}]``: each address a machine holds and its link.

    The interface and the prefix both come from the one connected-route rule
    (`gpu_node_facts.connected_route`), so this agrees with the enrolled
    cluster-interface fact about which link carries an address. An address on
    no connected network (a bare /32) names no network and is left out.
    """
    links: List[Dict[str, str]] = []
    for address in local_ipv4_addresses(fib_trie_text):
        try:
            name, prefix = gpu_node_facts.connected_route(str(route_text or ""), address)
        except ValueError:
            continue
        network = ipaddress.ip_network("{}/{}".format(address, prefix), strict=False)
        links.append({"name": name, "address": address, "network": str(network)})
    return links


def node_links_if_read(route_text: Any, fib_trie_text: Any) -> Optional[List[Dict[str, str]]]:
    """`node_links`, or ``None`` when either table came back empty.

    Both files always hold something on a running machine (a header line, the
    loopback leaf), so an empty one is a read that failed. ``None`` keeps that
    apart from ``[]``, which is a machine whose tables were read and hold no
    usable address.
    """
    if not str(route_text or "").strip() or not str(fib_trie_text or "").strip():
        return None
    return node_links(route_text, fib_trie_text)


def binding_in_network(
    links: Iterable[Mapping[str, Any]], network: str,
) -> Optional[Tuple[str, str]]:
    """``(address, interface)`` a machine holds inside ``network``, or ``None``."""
    wanted = ipaddress.ip_network(network)
    for link in links:
        try:
            address = ipaddress.IPv4Address(str(link.get("address", "")))
            name = gpu_node_facts.interface_name(str(link.get("name", "")))
        except (ipaddress.AddressValueError, ValueError):
            continue
        if address in wanted and lan_address(address):
            return str(address), name
    return None


# --- this machine's own links --------------------------------------------

def _read_text(path: str) -> str:
    try:
        with open(path, encoding="utf-8", errors="replace") as handle:
            return handle.read(65536)
    except OSError:
        return ""


def _list_names() -> List[str]:
    try:
        return sorted(os.listdir(NET_ROOT))
    except OSError:
        return []


def _driver_of(name: str) -> str:
    """The driver behind a link, or ``""`` for a link with no device (virtual)."""
    device = "{}/{}/device".format(NET_ROOT, name)
    if not os.path.exists(device):
        return ""
    try:
        return os.path.basename(os.path.realpath(device + "/driver")) or "unknown"
    except OSError:
        return "unknown"


def _unusable_reason(kind: str, address: str, tables_read: bool) -> str:
    if not tables_read:
        return (
            "This machine's address tables could not be read, so whether this "
            "link has an address is not known."
        )
    if not address:
        return (
            "This link has no IPv4 address, so other machines cannot reach "
            "this one over it."
        )
    if kind == KIND_WIRELESS:
        return (
            "This is a Wi-Fi link. A split model needs a wired or Thunderbolt "
            "link between the machines."
        )
    if not lan_address(address):
        return (
            "This link's address is not a private network address, so it is "
            "not offered for traffic between your own machines."
        )
    return ""


def local_node_links(
    read_text: Optional[Callable[[str], str]] = None,
) -> Optional[List[Dict[str, str]]]:
    """This machine's own addresses and links, or ``None`` if they could not be read."""
    reader = read_text or _read_text
    return node_links_if_read(reader(ROUTE_PATH), reader(FIB_TRIE_PATH))


def discover_links(
    *, list_names: Callable[[], List[str]] = _list_names,
    driver_of: Callable[[str], str] = _driver_of,
    read_text: Callable[[str], str] = _read_text,
) -> List[Dict[str, Any]]:
    """This machine's own network links, each with whether it can be the cluster link.

    Only links with a device behind them: loopback, container bridges and
    virtual pairs have none and are not links between machines. Every value is
    read - the kind from the driver and the kernel's own wireless mark, the
    speed and state from the link's files - and an unread one is ``None`` or
    ``unknown``, never a default. ``usable`` is false with a plain ``reason``
    for a link that cannot carry a split.
    """
    held = local_node_links(read_text)
    tables_read = held is not None
    carried: Dict[str, Dict[str, str]] = {}
    for link in held or []:
        carried.setdefault(link["name"], link)
    found: List[Dict[str, Any]] = []
    for raw in list_names():
        try:
            name = gpu_node_facts.interface_name(raw)
        except ValueError:
            continue
        driver = driver_of(name)
        if not driver:
            continue
        facts = gpu_node_facts.read_local_link(name, read_text=read_text)
        if facts["medium"] == gpu_node_facts.MEDIUM_WIRELESS:
            kind = KIND_WIRELESS
        elif driver in _THUNDERBOLT_DRIVERS:
            kind = KIND_THUNDERBOLT
        elif facts["medium"] == gpu_node_facts.MEDIUM_WIRED:
            kind = KIND_WIRED
        else:
            kind = KIND_UNKNOWN
        mtu_text = read_text("{}/{}/mtu".format(NET_ROOT, name)).strip()
        holds = carried.get(name, {})
        address = holds.get("address", "")
        reason = _unusable_reason(kind, address, tables_read)
        found.append({
            "name": name,
            "kind": kind,
            "state": read_text("{}/{}/operstate".format(NET_ROOT, name)).strip() or "unknown",
            "speed_mbps": facts["speed_mbps"],
            "mtu": int(mtu_text) if mtu_text.isdigit() else None,
            "address": address,
            "network": holds.get("network", ""),
            "usable": not reason,
            "reason": reason,
        })
    return found


def require_link(name: Any, links: Iterable[Mapping[str, Any]]) -> Dict[str, Any]:
    """The discovered link called ``name``, or a plain refusal.

    Shape first, then membership, then whether it can carry a split - so a
    value that names no link on this machine never travels further.
    """
    wanted = gpu_node_facts.interface_name(str(name or ""))
    for link in links:
        if link.get("name") == wanted:
            if not link.get("usable"):
                raise ValueError("{} cannot be the cluster link. {}".format(
                    wanted, link.get("reason", "")
                ).strip())
            return dict(link)
    raise ValueError(
        "{} is not one of this machine's network links. Choose one of the "
        "links listed for this machine.".format(wanted)
    )


# --- the stored choice ----------------------------------------------------

class ClusterLinkSetting:
    """The owner's choice, kept in the cluster store under :data:`SETTING_KEY`."""

    def __init__(self, store: Any):
        self.store = store

    def interface(self) -> str:
        value = self.store.state_value(SETTING_KEY)
        if not isinstance(value, dict):
            return ""
        return str(value.get("interface", "") or "")

    def choose(self, name: Any, links: Iterable[Mapping[str, Any]]) -> Dict[str, Any]:
        link = require_link(name, links)
        self.store.put_state_value(SETTING_KEY, {"interface": link["name"]})
        return link

    def clear(self) -> None:
        self.store.delete_state_value(SETTING_KEY)


def chosen_link(
    setting: ClusterLinkSetting, links: Iterable[Mapping[str, Any]],
) -> Optional[Dict[str, Any]]:
    """The chosen link as it is NOW, or ``None`` when none is chosen.

    ``valid`` is false, with the reason, when the stored name no longer matches
    a usable link on this machine - a link unplugged, renamed or left without
    an address since it was chosen. That state is shown and refuses a split;
    the stored name is not silently dropped or replaced.
    """
    name = setting.interface()
    if not name:
        return None
    try:
        return {**require_link(name, links), "valid": True}
    except ValueError as error:
        return {
            "name": name, "valid": False, "usable": False,
            "reason": str(error), "address": "", "network": "",
        }


def link_for_split(store: Any) -> Optional[Dict[str, Any]]:
    """The link a split deploy must use now: a usable one, ``None``, or a refusal."""
    link = chosen_link(ClusterLinkSetting(store), discover_links())
    if link is None:
        return None
    if not link["valid"]:
        raise ValueError(
            "The cluster link chosen in Cluster > Setup cannot be used right "
            "now. {} Choose a link again, or clear the choice.".format(link["reason"])
        )
    return link


# --- the split launch -----------------------------------------------------

def read_node_tables(transport: Any) -> Optional[List[Dict[str, str]]]:
    """One machine's addresses and links, over the channel it was enrolled on.

    ``None`` when either table came back empty: a read that failed is not a
    machine with no addresses.
    """
    return node_links_if_read(
        transport.run(["cat", ROUTE_PATH]), transport.run(["cat", FIB_TRIE_PATH])
    )


def _tables_of(node: Mapping[str, Any], read: Callable[[], Any]) -> List[Dict[str, str]]:
    """One machine's tables for a deploy, or a refusal that names the machine."""
    try:
        tables = read()
    except (OSError, RuntimeError) as error:
        raise ValueError(
            "{}'s network addresses could not be read ({}), so Vaelor cannot "
            "tell which link it is on.".format(node["name"], str(error)[:160])
        ) from error
    if tables is None:
        raise ValueError(
            "{}'s network addresses could not be read, so Vaelor cannot tell "
            "which link it is on. Recheck the machine, then try again.".format(
                node["name"]
            )
        )
    return list(tables)


#: The refusal of interface pins beside the cluster link. The link is chosen
#: per split (VD-167), so the advice is about this split, not Cluster > Setup.
PINS_WITH_LINK = (
    "This deployment names network interfaces itself, and this split is set "
    "to ride the cluster link. Choose Ethernet for this split, or clear the "
    "interface pin."
)


def refuse_pins_with_link(link: Any, node_pins: Any, pinned: Any) -> None:
    """Refuse a deploy that pins interfaces while the split rides a cluster link.

    Two owners of one answer: the pins say which interface each machine binds,
    and so does the link. Neither silently wins; the deploy is refused before
    anything is written, and says which of the two to drop.
    """
    if link is not None and (node_pins or pinned):
        raise ValueError(PINS_WITH_LINK)


def split_bindings(
    participants, transports, node_pins, pinned, *,
    advertise_address, resolve, link: Optional[Mapping[str, Any]],
    read_tables: Callable[[Any], Any] = read_node_tables,
) -> Tuple[Dict[str, str], Dict[str, str], Dict[str, str]]:
    """``(addresses, interfaces, sources)`` for every machine in a split.

    With no link chosen this is `gpu_node_facts.cluster_interfaces` beside each
    machine's enrolled address - the behaviour before this setting existed.
    With one, each machine's address and interface are the ones it holds
    inside the chosen link's network (the same-network rule), the controller's
    from the link itself and a worker's from its own tables. Interface pins in
    the deploy and a chosen link are two owners of one answer, so a deploy
    carrying both is refused rather than one silently winning.
    """
    if link is None:
        interfaces, sources = gpu_node_facts.cluster_interfaces(
            participants, transports, node_pins, pinned,
            advertise_address=advertise_address, resolve=resolve,
        )
        addresses = {}
        for node in participants:
            try:
                addresses[node["id"]] = advertise_address(node["host"])
            except ValueError as error:
                raise ValueError("{}: {}".format(node["name"], error)) from error
        return addresses, interfaces, sources
    refuse_pins_with_link(link, node_pins, pinned)
    network = str(link["network"])
    addresses: Dict[str, str] = {}
    interfaces: Dict[str, str] = {}
    for node in participants:
        node_id = node["id"]
        if node_id == CONTROLLER_PLACEMENT_ID:
            binding: Optional[Tuple[str, str]] = (
                str(link["address"]), gpu_node_facts.interface_name(str(link["name"]))
            )
        else:
            binding = binding_in_network(
                _tables_of(node, lambda: read_tables(transports[node_id])), network
            )
        if binding is None:
            raise ValueError(
                "{} has no address on the cluster link's network ({}). Connect "
                "it to that link and give it an address there, or clear the "
                "cluster link in Cluster > Setup.".format(node["name"], network)
            )
        if binding[0] in addresses.values():
            raise ValueError(
                "{} holds the same address on the cluster link as another "
                "machine in this deployment ({}). Each machine needs its own "
                "address on that link.".format(node["name"], binding[0])
            )
        addresses[node_id], interfaces[node_id] = binding
    sources = {node_id: SOURCE_CLUSTER_LINK for node_id in interfaces}
    return addresses, interfaces, sources


def confirm_recorded_bindings(
    participants, transports, addresses: Mapping[str, str],
    sources: Mapping[str, str], *,
    read_tables: Callable[[Any], Any], local_tables: Callable[[], Any],
) -> None:
    """Before a Load: each machine still holds the link address it was deployed on.

    A Load re-serves a split on the binding its record wrote, with no look at
    the current choice. That is right while the link is as it was, and a hang
    when it is not: a collective told to bind an address the machine no longer
    holds waits instead of failing (review S8). So each machine that was bound
    through the cluster link is asked, first, whether it still holds that
    address - and one that does not refuses the Load by name, before any unit.

    Only bindings that came from the cluster link are checked. An enrolled
    address is the address Vaelor reaches the machine ON, so reaching it at
    all is the proof, and a record written before this setting has none to
    check.
    """
    for node in participants:
        node_id = node["id"]
        recorded = str(addresses.get(node_id, "") or "")
        if not recorded or sources.get(node_id) != SOURCE_CLUSTER_LINK:
            continue
        if node_id == CONTROLLER_PLACEMENT_ID:
            tables = _tables_of(node, local_tables)
        else:
            tables = _tables_of(node, lambda: read_tables(transports[node_id]))
        if not any(str(item.get("address", "")) == recorded for item in tables):
            raise ValueError(
                "{} no longer holds the address this deployment was deployed "
                "on ({}). Its cluster link may be unplugged or may have been "
                "given a new address. Remove the deployment and deploy it "
                "again so it binds the addresses the machines hold now.".format(
                    node["name"], recorded
                )
            )
