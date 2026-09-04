"""One cluster, one processor architecture — decided by what was discovered.

Per **VD-031** a controller manages workers of its own architecture class only:
Pi joins Pi, Z2 joins Z2, never mixed. Docker Swarm will schedule an arm64-only
image onto an amd64 node and fail at container *start*, so the UI reports a
successful deploy for a workload that can never run. Refusing the join removes
that failure mode at the root instead of defending against it downstream.

Two rules this module exists to hold:

* **The architecture is observed, never declared.** Every value here comes from
  `uname -m` read over the enrollment SSH session, or from this appliance's own
  ``platform.machine()``. Nothing consults the machine class, and nothing trusts
  a field a joining node could simply assert (VD-005).
* **There is no per-architecture branch.** The gate is one equality between two
  discovered values. Adding ``if arch == "arm64"`` anywhere in shared clustering
  code is the failure VD-004 names; there is nothing here for such a branch to
  select between.
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional


#: ``uname -m`` and Docker platform spellings that mean the same silicon. The
#: canonical form on the right is Docker's ``node.platform.arch`` vocabulary,
#: which is what a Swarm constraint and a container manifest both speak.
ARCHITECTURE_CLASSES = {
    "aarch64": "arm64",
    "arm64": "arm64",
    "armv8b": "arm64",
    "armv8l": "arm64",
    "amd64": "amd64",
    "x86-64": "amd64",
    "x86_64": "amd64",
}

#: Prose for an operator. A refusal that says "incompatible node" cannot be
#: acted on; one that says "this controller is 64-bit ARM and that node is
#: x86-64" can.
ARCHITECTURE_LABELS = {
    "arm64": "64-bit ARM (arm64)",
    "amd64": "x86-64 (amd64)",
}

#: The classes a Vaelor cluster runs at all. A 32-bit or unrecognised host is
#: not a different cluster, it is not a cluster worker.
CLUSTER_ARCHITECTURES = frozenset(ARCHITECTURE_CLASSES.values())

UNSUPPORTED_ARCHITECTURE = (
    "A Vaelor cluster worker must be 64-bit ARM or x86-64."
)


def architecture_class(value: Any) -> str:
    """Canonical class for a discovered architecture string.

    Returns ``""`` for anything unrecognised — including the empty string a
    failed probe leaves behind. An unknown architecture is never quietly
    treated as matching, because "we could not tell" and "it matches" are the
    two answers this module must never confuse.
    """
    return ARCHITECTURE_CLASSES.get(str(value or "").strip().lower(), "")


def architecture_label(value: Any) -> str:
    """Human name for a discovered architecture, falling back to the raw value."""
    normalized = architecture_class(value)
    if normalized:
        return ARCHITECTURE_LABELS[normalized]
    raw = str(value or "").strip()
    return raw or "an unknown architecture"


def inventory_architecture(record: Optional[Dict[str, Any]]) -> str:
    """The architecture class of an enrolled node, from its probed inventory."""
    inventory = (record or {}).get("inventory") or {}
    return architecture_class(inventory.get("architecture"))


def controller_architecture(hardware: Optional[Dict[str, Any]]) -> str:
    """The architecture class of this appliance, from its own hardware probe.

    ``hardware`` is the `copilot_setup.hardware_inventory` payload, whose
    ``architecture`` is `platform.machine()`. It is a reading, not a
    declaration, and it is deliberately not the machine class: a workstation
    driver is selected by SMBIOS presence, which says nothing about silicon.
    """
    return architecture_class((hardware or {}).get("architecture"))


def _describe(where: str) -> str:
    label = str(where or "").strip()
    return f"The node at {label}" if label else "That node"


def join_compatibility(
    controller: Any, worker: Any, *, where: str = ""
) -> Dict[str, Any]:
    """Whether a worker may join a controller, and — when not — why.

    ``where`` is the worker's address or name, so the refusal names the machine
    the operator has to go and look at.
    """
    controller_class = architecture_class(controller)
    worker_class = architecture_class(worker)
    result = {
        "compatible": False,
        "reason": "",
        "controller_architecture": controller_class,
        "worker_architecture": worker_class,
    }
    if not worker_class:
        result["reason"] = (
            f"{_describe(where)} reports {architecture_label(worker)}. "
            f"{UNSUPPORTED_ARCHITECTURE}"
        )
        return result
    if not controller_class:
        # Failing closed. Admitting a worker because the controller could not
        # read its own silicon is how a mixed cluster gets formed by accident.
        result["reason"] = (
            "This controller could not determine its own processor "
            "architecture, so it cannot confirm that a worker matches it."
        )
        return result
    if controller_class != worker_class:
        result["reason"] = (
            "This controller is {controller}; {node} is {worker}. A Vaelor "
            "cluster keeps one processor architecture, so enrol this node "
            "into a cluster whose controller is {worker} instead.".format(
                controller=ARCHITECTURE_LABELS[controller_class],
                node=_describe(where).lower(),
                worker=ARCHITECTURE_LABELS[worker_class],
            )
        )
        return result
    result["compatible"] = True
    return result


#: What a controller is entitled to *do* about one enrolled node, given the two
#: readings it managed to take. Only `MISMATCH` authorises removal (VD-033).
MATCHES = "matches"
UNREADABLE_NODE = "unreadable-node"
UNREADABLE_CONTROLLER = "unreadable-controller"
MISMATCH = "mismatch"

#: The one disposition that removal may act on. Named, so that a caller has to
#: say which state it is destroying rather than inverting a boolean.
REMOVABLE_DISPOSITIONS = frozenset({MISMATCH})


def eviction_disposition(
    controller: Any, node: Dict[str, Any]
) -> Dict[str, Any]:
    """Whether an enrolled node's architecture warrants removal, and why.

    ``join_compatibility`` answers a single boolean, and that is right for a
    join: *anything other than a confirmed match* is refused, so "the other
    architecture" and "this controller could not read one of the two values"
    can share an answer. Removal cannot share it. It destroys state, so it
    needs the three cases kept apart:

    * **mismatch** — both values were read and they differ. This is evidence,
      and the only case that authorises removal.
    * **unreadable node** — the node's probed architecture is missing or is not
      one Vaelor clusters. *Unknown is not mismatch.* Report it, leave it.
    * **unreadable controller** — this appliance could not read its own
      silicon, so it can judge nothing. Nothing is removed.

    The reading is the *stored, probed* inventory, exactly as the join gate and
    the fleet view use it. Nothing here consults the machine class, and nothing
    trusts a field the node could assert. A node that is simply unreachable
    right now still carries whatever was last discovered about it, and that
    discovery — not its silence — is what this answers on.
    """
    controller_class = architecture_class(controller)
    node_class = inventory_architecture(node)
    raw = (node.get("inventory") or {}).get("architecture")
    where = _describe(str(node.get("host") or node.get("name") or ""))
    result = {
        "disposition": MISMATCH,
        "removable": False,
        "controller_architecture": controller_class,
        "node_architecture": node_class,
        "label": architecture_label(raw),
        "reason": "",
    }
    if not controller_class:
        result["disposition"] = UNREADABLE_CONTROLLER
        result["reason"] = (
            "This controller could not determine its own processor "
            f"architecture, so it cannot tell whether {where.lower()} matches "
            "it. Nothing is removed on a reading this appliance never took."
        )
        return result
    if not node_class:
        result["disposition"] = UNREADABLE_NODE
        result["reason"] = (
            f"{where} reports {architecture_label(raw)}, which this controller "
            "does not recognise. An architecture that could not be read is not "
            "a mismatch, so this node is reported and left enrolled."
        )
        return result
    if node_class == controller_class:
        result["disposition"] = MATCHES
        return result
    result["removable"] = True
    result["reason"] = (
        "This controller is {controller}; {node} is {worker}. A Vaelor cluster "
        "keeps one processor architecture, so this node is drained and removed "
        "from the fleet.".format(
            controller=ARCHITECTURE_LABELS[controller_class],
            node=where.lower(),
            worker=ARCHITECTURE_LABELS[node_class],
        )
    )
    return result


def mismatched_nodes(
    controller: Any, nodes: List[Dict[str, Any]]
) -> List[Dict[str, Any]]:
    """The enrolled nodes a controller is entitled to drain and remove.

    Deliberately not "every node that is not a match" — that set also contains
    the nodes nothing could be read about, and removing those would turn a
    failed probe into a destroyed enrollment.
    """
    return [
        {
            "node_id": str(node.get("id", "")),
            "name": str(node.get("name", "")),
            "host": str(node.get("host", "")),
            "joined": bool((node.get("labels") or {}).get("swarm_node_id")),
            **verdict,
        }
        for node, verdict in (
            (node, eviction_disposition(controller, node)) for node in nodes
        )
        if verdict["disposition"] in REMOVABLE_DISPOSITIONS
    ]


def node_architecture_fact(
    controller: Any, node: Dict[str, Any]
) -> Dict[str, Any]:
    """The architecture fact published for one enrolled node.

    Every enrolled node carries this, matching or not. Refusing new joins does
    nothing about a node enrolled before the rule existed, and a fleet view that
    showed such a node as ordinary would be the same lie by a different route.

    ``disposition`` and ``removable`` ride along because "we could not read it"
    and "it is the wrong one" look identical through ``matches_controller``
    alone, and under VD-033 they have opposite consequences.
    """
    verdict = join_compatibility(
        controller,
        (node.get("inventory") or {}).get("architecture"),
        where=str(node.get("host") or node.get("name") or ""),
    )
    disposition = eviction_disposition(controller, node)
    return {
        "class": verdict["worker_architecture"],
        "label": architecture_label(
            (node.get("inventory") or {}).get("architecture")
        ),
        "matches_controller": verdict["compatible"],
        "reason": verdict["reason"],
        "disposition": disposition["disposition"],
        "removable": disposition["removable"],
        "removal_reason": disposition["reason"],
    }


def fleet_architecture(
    controller: Any, nodes: List[Dict[str, Any]]
) -> Dict[str, Any]:
    """Controller architecture plus every enrolled node that does not share it."""
    controller_class = architecture_class(controller)
    conflicts = [
        {
            "node_id": str(node.get("id", "")),
            "name": str(node.get("name", "")),
            "host": str(node.get("host", "")),
            "architecture": fact["class"],
            "label": fact["label"],
            "reason": fact["reason"],
            # A conflict list that only said "does not match" would invite its
            # reader to remove everything on it, and a node whose probe failed
            # is on it too. Which of the two each entry is travels with it.
            "disposition": fact["disposition"],
            "removable": fact["removable"],
            "removal_reason": fact["removal_reason"],
        }
        for node, fact in (
            (node, node_architecture_fact(controller, node)) for node in nodes
        )
        if not fact["matches_controller"]
    ]
    return {
        "controller": controller_class,
        "label": (
            ARCHITECTURE_LABELS[controller_class]
            if controller_class
            else "an architecture this appliance could not read"
        ),
        "homogeneous": not conflicts,
        "conflicts": conflicts,
        "note": (
            "A cluster runs one processor architecture. Workers are admitted "
            "on what the controller observes over SSH, not on the operating "
            "system they are running."
        ),
    }
