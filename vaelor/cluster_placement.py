"""Shared placement facts for the local controller and enrolled nodes."""

from __future__ import annotations

from typing import Any, Dict, Iterable, Set


CONTROLLER_PLACEMENT_ID = "controller"

#: How the head controller names itself as a placement target. Held here so a
#: consumer that builds controller facts (`cluster_capacity`) imports the one
#: name rather than re-spelling it - keeping the sentence out of a third module,
#: which the duplicate-sentence guard counts as a fresh instance of one idea in
#: two places. `cluster_operations._placement_node` keeps its own copy for now,
#: so the recorded pair is unchanged.
CONTROLLER_PLACEMENT_NAME = "This Vaelor controller"


def add_controller_placement(
    summary: Dict[str, Any], hardware: Dict[str, Any]
) -> Dict[str, Any]:
    """Expose the local manager through the bounded placement contract."""
    controller = summary.setdefault("controller", {})
    runtime = summary.get("runtime", {})
    initialized = bool(
        controller.get("initialized")
        and runtime.get("initialized")
        and runtime.get("control_available")
    )
    node_id = str(controller.get("cluster_id") or runtime.get("node_id") or "")
    live = next(
        (
            node for node in runtime.get("nodes", [])
            if str(node.get("id", "")) == node_id
        ),
        None,
    )
    memory = int(hardware.get("memory_total_bytes", 0) or 0)
    storage = int(hardware.get("storage_free_bytes", 0) or 0)
    eligible = initialized and memory >= 1024 ** 3 and storage >= 2 * 1024 ** 3
    reason = ""
    if not initialized:
        reason = "Initialize this controller before placing cluster workloads."
    elif memory < 1024 ** 3:
        reason = "The controller needs at least 1 GB of physical memory."
    elif storage < 2 * 1024 ** 3:
        reason = "Free at least 2 GB on the controller before placing a model."
    controller["placement"] = {
        "id": CONTROLLER_PLACEMENT_ID,
        "name": CONTROLLER_PLACEMENT_NAME,
        "host": str(
            controller.get("advertise_address")
            or controller.get("candidate_address")
            or "127.0.0.1"
        ),
        "port": 0,
        "role": "head-controller",
        "state": "ready" if eligible else "unavailable",
        "runtime_state": "ready" if eligible else "unavailable",
        "runtime": live or ({
            "status": "Ready",
            "availability": "Active",
            "manager_status": "Leader",
        } if initialized else None),
        "host_key_fingerprint": "",
        "labels": {"swarm_node_id": node_id} if node_id else {},
        "inventory": {
            "architecture": hardware.get("architecture", "unknown"),
            # #113: `cpu_count` is PHYSICAL CORES on every path; `cpu_threads`
            # carries the logical/thread count as a separate field. Do not treat
            # threads as cores. `pooled_operations` sizes llama `--threads` from
            # `cpu_threads`, and the fleet UI shows `cpu_count` as "cores", so
            # the two must not be the same number on an SMT part. Enrolled nodes
            # read physical cores from CPU topology (ssh_transport); the
            # controller does the same from its own `cpu_cores`.
            "cpu_count": hardware.get("cpu_cores", 0),
            "cpu_threads": hardware.get("cpu_threads") or hardware.get("cpu_cores", 0),
            "memory_bytes": memory,
            "root_free_bytes": storage,
            "os": "This appliance",
            "docker": bool(runtime.get("available")),
            "reachable": True,
        },
        "eligible": eligible,
        "reason": reason,
    }
    # VD-125: carry the controller's OWN cluster link into the placement node the
    # serve form renders. `cluster_manager.summary` reads it live off this
    # machine and lands it on `controller["inventory"]["cluster_interface"]`;
    # this placement builds a fresh inventory from the hardware probe, so without
    # copying the fact across the serve form would read "link speed unknown" for
    # the controller while the fleet card had the real link. Left absent when the
    # summary attached none (a not-yet-initialised controller), so the honest
    # fallback stands rather than a guessed link.
    cluster_interface = (controller.get("inventory") or {}).get("cluster_interface")
    if isinstance(cluster_interface, dict):
        controller["placement"]["inventory"]["cluster_interface"] = cluster_interface
    return summary


def departed_node_ids(store: Any, node_ids: Iterable[Any]) -> Set[str]:
    """The ids in ``node_ids`` that name neither this controller nor any
    enrolled machine - machines that have left the fleet (review R2).

    Read from the cluster store at the moment of asking, the one owner of
    "is this machine in the fleet". A store that cannot be listed answers the
    empty set: when nothing can be proved gone, nothing is exempted.
    """
    listing = getattr(store, "list_nodes", None)
    if not callable(listing):
        return set()
    try:
        enrolled = {str(node.get("id", "")) for node in listing() or []}
    except Exception:  # noqa: BLE001 - unreadable is "cannot tell"
        return set()
    return {
        str(node_id) for node_id in node_ids
        if str(node_id) and str(node_id) != CONTROLLER_PLACEMENT_ID
        and str(node_id) not in enrolled
    }


def placement_node(manager, summary: Dict[str, Any], node_id: Any):
    """Resolve a controller placement or an enrolled node record."""
    if str(node_id) == CONTROLLER_PLACEMENT_ID:
        target = summary.get("controller", {}).get("placement")
        return target if target and target.get("eligible") else None
    return manager.store.get_node(str(node_id))


#: G5 (VD-194, extending VD-179): the Assistant is never placed on a worker.
#: A worker holds no Assistant runtime or model; the controller's serves the
#: cluster. Said to whoever asked for one.
ASSISTANT_ON_WORKER_REFUSED = (
    "The Assistant runs on this controller only; a cluster worker holds no Assistant of its own."
)
#: The payload fields a request could use to name a machine.
_PLACEMENT_FIELDS = ("node_id", "node_ids", "placement", "target_node")


def refuse_assistant_placement(payload: object) -> None:
    """Refuse an Assistant deploy or install that names any machine but this controller."""
    if not isinstance(payload, dict):
        return
    for name in _PLACEMENT_FIELDS:
        value = payload.get(name)
        named = value if isinstance(value, (list, tuple)) else ([value] if value not in (None, "") else [])
        if any(str(item) != CONTROLLER_PLACEMENT_ID for item in named):
            raise ValueError(ASSISTANT_ON_WORKER_REFUSED)
