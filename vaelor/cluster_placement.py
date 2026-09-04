"""Shared placement facts for the local controller and enrolled nodes."""

from __future__ import annotations

from typing import Any, Dict


CONTROLLER_PLACEMENT_ID = "controller"


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
        "name": "This Vaelor controller",
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
    return summary


def placement_node(manager, summary: Dict[str, Any], node_id: Any):
    """Resolve a controller placement or an enrolled node record."""
    if str(node_id) == CONTROLLER_PLACEMENT_ID:
        target = summary.get("controller", {}).get("placement")
        return target if target and target.get("eligible") else None
    return manager.store.get_node(str(node_id))
