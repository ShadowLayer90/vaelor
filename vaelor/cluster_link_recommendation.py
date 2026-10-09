"""Which link a tensor-parallel split should ride: the fastest one the machines share.

Owner decisions, 2026-10-01: a split may use either link, the normal Ethernet
connection each machine is reached on or the owner-chosen cluster link
(`cluster_link`, VD-162). A tensor-parallel split talks on every layer, so the
form RECOMMENDS the faster link for it - by measured speed, never by name: if
the Ethernet is faster than the Thunderbolt cable, the Ethernet is recommended.
A pipeline split gets no recommendation; it measured the same on 1 GbE and on
Thunderbolt.

**How a link's speed is judged.**

* A link counts only if EVERY machine of the split is on it: the Ethernet
  always (it is how each machine is reached); the cluster link only when each
  worker holds an address on its network.
* Each end's figure is its EFFECTIVE speed where Vaelor knows one
  (:data:`KNOWN_EFFECTIVE_MBPS`), else the speed the link NEGOTIATED, labelled
  as negotiated. Thunderbolt networking negotiates 40 Gb/s but carried about
  9 Gb/s each way between the lab machines, a limit of the ``thunderbolt-net``
  driver, so 9 Gb/s is what it is compared on.
* A link is as fast as its slowest end. A direct cable between two machines
  (a /30 or /31 network) has one speed, so the controller's reading of its end
  is the link's; on a wider network a worker's end is not read and is unknown.
* An end whose speed is unknown makes the comparison impossible, and two links
  within :data:`TIE_RATIO` of each other are a tie: either way nothing is
  recommended, and the sentence says why. No winner is invented.

Pure over the facts handed in, so every case is testable without a machine.
"""

from __future__ import annotations

import ipaddress
from typing import Any, Dict, Iterable, List, Mapping, Optional

from .cluster_link import KIND_THUNDERBOLT, refuse_pins_with_link
from .cluster_placement import CONTROLLER_PLACEMENT_ID

#: The two links a split can ride, by the key the form sends.
LINK_ETHERNET = "ethernet"
LINK_CLUSTER = "cluster-link"
SPLIT_LINKS = (LINK_ETHERNET, LINK_CLUSTER)

#: The fit and deploy field carrying the owner's link choice for a split.
SPLIT_LINK_FIELD = "split_link"

#: The refusal when the form asks for the cluster link and none is chosen.
NO_CLUSTER_LINK = (
    "No cluster link is chosen in Cluster > Setup, so this split cannot ride "
    "one. Choose a link there, or carry the split over Ethernet."
)

#: What a link of a kind actually carries where that is known to differ from
#: what it negotiates (Mb/s, each direction). Measured 2026-09-30 between the
#: two lab machines: ``thunderbolt-net`` at 40 Gb/s negotiated moved about
#: 9 Gb/s each way, the driver's own ceiling.
KNOWN_EFFECTIVE_MBPS: Dict[str, int] = {KIND_THUNDERBOLT: 9000}

#: Two links within 10% of each other are not told apart.
TIE_RATIO = 1.10

#: The widest network a direct cable between two machines is given.
_DIRECT_PREFIX = 30


def _gbps(mbps: int) -> str:
    if mbps >= 1000:
        return "{:g} Gb/s".format(round(mbps / 1000.0, 1))
    return "{} Mb/s".format(mbps)


def end_figure(kind: Any, speed_mbps: Any) -> Optional[Dict[str, Any]]:
    """One end's comparable speed, or ``None`` when it is not known.

    ``{"mbps", "basis", "negotiated"}``: ``basis`` is ``"effective"`` when a
    known ceiling applies and is below what was negotiated, else ``"negotiated"``.
    """
    negotiated = None
    if isinstance(speed_mbps, int) and not isinstance(speed_mbps, bool) and speed_mbps > 0:
        negotiated = speed_mbps
    ceiling = KNOWN_EFFECTIVE_MBPS.get(str(kind or ""))
    if ceiling is not None and (negotiated is None or ceiling < negotiated):
        return {"mbps": ceiling, "basis": "effective", "negotiated": negotiated}
    if negotiated is None:
        return None
    return {"mbps": negotiated, "basis": "negotiated", "negotiated": negotiated}


def judge_link(key: str, label: str, ends: Iterable[Mapping[str, Any]]) -> Dict[str, Any]:
    """A link's speed: its slowest end, or unknown naming the end that is.

    Each end is ``{"node", "kind", "speed_mbps"}``.
    """
    slowest: Optional[Dict[str, Any]] = None
    for end in ends:
        figure = end_figure(end.get("kind"), end.get("speed_mbps"))
        if figure is None:
            return {"key": key, "label": label, "mbps": None,
                    "unknown_on": str(end.get("node") or "")}
        if slowest is None or figure["mbps"] < slowest["mbps"]:
            slowest = {**figure, "node": str(end.get("node") or "")}
    if slowest is None:
        return {"key": key, "label": label, "mbps": None, "unknown_on": ""}
    return {"key": key, "label": label, **slowest}


def _phrase(link: Mapping[str, Any]) -> str:
    """``Thunderbolt link: 9 Gb/s effective (40 Gb/s negotiated)`` and the like."""
    if link["basis"] == "effective":
        text = "{}: {} effective".format(link["label"], _gbps(link["mbps"]))
        if link.get("negotiated"):
            text += " ({} negotiated)".format(_gbps(link["negotiated"]))
        return text
    return "{}: {} negotiated".format(link["label"], _gbps(link["mbps"]))


def recommend(links: List[Dict[str, Any]]) -> Dict[str, Any]:
    """The link to recommend for a tensor-parallel split, and the plain reason.

    ``links`` are :func:`judge_link` answers for the links every machine is on.
    """
    if not links:
        return {"recommended": None, "links": [], "reason": (
            "No link reaches every machine of the split, so none is recommended.")}
    if len(links) == 1:
        return {"recommended": links[0]["key"], "links": links, "reason": (
            "{} is the only link every machine of the split is on.".format(links[0]["label"]))}
    unknown = [link for link in links if link["mbps"] is None]
    if unknown:
        first = unknown[0]
        where = " on {}".format(first["unknown_on"]) if first.get("unknown_on") else ""
        return {"recommended": None, "links": links, "reason": (
            "The speed of the {}{} is not known, so Vaelor cannot say which link "
            "is faster and recommends neither.".format(first["label"], where))}
    ranked = sorted(links, key=lambda link: link["mbps"], reverse=True)
    best, runner = ranked[0], ranked[1]
    if best["mbps"] < runner["mbps"] * TIE_RATIO:
        return {"recommended": None, "links": links, "reason": (
            "{} and {} are about as fast as each other, so neither is "
            "recommended.".format(_phrase(best), _phrase(runner)))}
    slow_end = " on {}".format(runner["node"]) if runner.get("node") else ""
    return {"recommended": best["key"], "links": links, "reason": (
        "Recommended for a tensor-parallel split, which talks on every layer: "
        "{} vs {}{}.".format(_phrase(best), _phrase(runner), slow_end))}


def _direct(network: Any) -> bool:
    try:
        return ipaddress.ip_network(str(network), strict=False).prefixlen >= _DIRECT_PREFIX
    except ValueError:
        return False


def _on_network(links: Any, network: Any) -> bool:
    return any(isinstance(link, Mapping) and link.get("network") == network
               for link in (links or []))


def split_links(
    nodes: Iterable[Mapping[str, Any]], *, controller_ethernet: Mapping[str, Any],
    chosen: Optional[Mapping[str, Any]],
) -> List[Dict[str, Any]]:
    """The links every machine of a split is on, each judged.

    ``nodes`` carry ``node_id``, ``name`` and, for a worker, the stored
    ``inventory`` with its ``cluster_interface`` (``medium``, ``speed_mbps``)
    and ``links``. ``controller_ethernet`` is the controller's enrolled
    interface read live, in the same shape; ``chosen`` is the cluster link as
    `cluster_link` reports it (``kind``, ``speed_mbps``, ``network``), or ``None``.
    """
    ethernet_ends: List[Dict[str, Any]] = []
    cluster_ends: List[Dict[str, Any]] = []
    everyone_on_link = chosen is not None
    direct = chosen is not None and _direct(chosen.get("network"))
    for node in nodes:
        name = str(node.get("name") or node.get("node_id") or "")
        if node.get("node_id") == CONTROLLER_PLACEMENT_ID:
            enrolled = controller_ethernet
        else:
            enrolled = (node.get("inventory") or {}).get("cluster_interface") or {}
        ethernet_ends.append({"node": name, "kind": enrolled.get("medium"),
                              "speed_mbps": enrolled.get("speed_mbps")})
        if chosen is None:
            continue
        if node.get("node_id") == CONTROLLER_PLACEMENT_ID:
            cluster_ends.append({"node": name, "kind": chosen.get("kind"),
                                 "speed_mbps": chosen.get("speed_mbps")})
        elif not _on_network((node.get("inventory") or {}).get("links"), chosen.get("network")):
            everyone_on_link = False
        elif direct:
            # A direct cable has the one speed both of its ends negotiated.
            cluster_ends.append({"node": name, "kind": chosen.get("kind"),
                                 "speed_mbps": chosen.get("speed_mbps")})
        else:
            # On a wider network this worker's own end was not read.
            cluster_ends.append({"node": name, "kind": "", "speed_mbps": None})
    judged = [judge_link(LINK_ETHERNET, "Ethernet", ethernet_ends)]
    if everyone_on_link:
        label = "Thunderbolt link" if chosen.get("kind") == KIND_THUNDERBOLT else "Cluster link"
        judged.append(judge_link(
            LINK_CLUSTER, "{} ({})".format(label, chosen.get("name")), cluster_ends))
    return judged


def recommend_for_split(decision: Mapping[str, Any], store: Any) -> Dict[str, Any]:
    """The recommendation for a fit's split, from what this controller holds.

    The controller's own enrolled interface and chosen link are read live; a
    worker's come from its stored probe. Any read that fails leaves that fact
    unknown, which the recommendation then says rather than guessing.
    """
    from . import cluster_link, gpu_node_facts

    nodes = []
    for view in (decision.get("placement") or {}).get("nodes") or []:
        node = {"node_id": view.get("node_id"), "name": view.get("name")}
        if view.get("node_id") != CONTROLLER_PLACEMENT_ID:
            try:
                node["inventory"] = (store.get_node(str(view.get("node_id"))) or {}).get("inventory") or {}
            except (AttributeError, KeyError, TypeError, OSError, ValueError):
                node["inventory"] = {}
        nodes.append(node)
    try:
        address = str((store.controller() or {}).get("advertise_address") or "")
        ethernet = gpu_node_facts.local_cluster_interface(address) if address else {}
    except (AttributeError, KeyError, TypeError, OSError, ValueError):
        ethernet = {}
    try:
        chosen = cluster_link.chosen_link(
            cluster_link.ClusterLinkSetting(store), cluster_link.discover_links())
    except (AttributeError, KeyError, TypeError, OSError, ValueError):
        chosen = None
    if chosen is not None and not chosen.get("valid"):
        chosen = None
    return recommend(split_links(nodes, controller_ethernet=ethernet or {}, chosen=chosen))


def parse_split_link(value: Any) -> Optional[str]:
    """The link the owner chose for a split, or ``None`` when none was sent.

    ``None`` keeps the behaviour from before the form offered the choice: the
    link chosen in Cluster > Setup, or Ethernet when there is none.
    """
    if value is None or value == "":
        return None
    if value not in SPLIT_LINKS:
        raise ValueError(
            "A split rides either 'ethernet' or 'cluster-link'; '{}' is neither.".format(
                str(value)[:40]))
    return str(value)


def link_for_choice(choice: Optional[str], read_link: Any) -> Optional[Mapping[str, Any]]:
    """The cluster link a split binds for the owner's ``choice``, or ``None`` for Ethernet.

    ``read_link`` is `cluster_link.link_for_split` (or the deploy's own
    reader): it raises when a chosen link is gone. Ethernet never reads it, so
    a gone link refuses only a split that asked for it; asking for the cluster
    link with none chosen is refused rather than quietly riding Ethernet.
    """
    if choice == LINK_ETHERNET:
        return None
    link = read_link()
    if choice == LINK_CLUSTER and link is None:
        raise ValueError(NO_CLUSTER_LINK)
    return link


def split_link_for(choice: Optional[str], read_link: Any, node_pins: Any, pinned: Any
                   ) -> Optional[Mapping[str, Any]]:
    """The link a split binds, refusing interface pins beside a cluster link.

    The ONE derivation the fit preview and the deploy share, so a preview
    cannot say "fits" for a split the deploy then refuses.
    """
    link = link_for_choice(choice, read_link)
    refuse_pins_with_link(link, node_pins, pinned)
    return link
