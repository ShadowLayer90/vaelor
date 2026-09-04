"""Approval plans for controller initialization and worker membership changes."""

from __future__ import annotations

from typing import Any, Dict

from .cluster_network import private_controller_ipv4
from .cluster_plan_contract import ClusterPlanContext, ClusterPlanError


def plan_initialize(
    action: str, body: Dict[str, Any], context: ClusterPlanContext
) -> Dict[str, Any]:
    try:
        advertise_address = private_controller_ipv4(
            body.get("advertise_address", "")
        )
    except ValueError as error:
        raise ClusterPlanError(
            "cluster_controller_address", str(error)
        ) from error
    return {
        # "Pi" is a board name, and this plan is read on x86 workstations too;
        # "this node" is the machine-class-neutral term the rest of the cluster
        # copy already uses ("make this node the head controller"), so it stays
        # correct on every machine instead of asserting a Raspberry Pi.
        "title": "Initialize this node as the head controller",
        "steps": [
            "Enable Docker Swarm manager mode on this Vaelor node.",
            f"Advertise {advertise_address} on TCP 2377 for worker joins.",
            "Keep workload traffic inside Docker's encrypted node control plane.",
        ],
        "impact": "Existing standalone Compose apps remain standalone and are not migrated.",
        "approval_required": True,
    }


def plan_join_node(
    action: str, body: Dict[str, Any], context: ClusterPlanContext
) -> Dict[str, Any]:
    node = context.enrolled_node(body.get("node_id", ""), "Enroll a node first.")
    return {
        "title": f"Join {node['name']} as a worker",
        "steps": [
            "Re-check the pinned SSH host key and hardware inventory.",
            "Install the OS Docker package if Docker is missing.",
            "Join with a one-time Swarm worker token, then rotate that token.",
            "Apply Vaelor node labels for architecture and memory-aware placement.",
        ],
        "impact": "The worker will accept cluster workloads from this head controller.",
        "approval_required": True,
    }


def plan_node_availability(
    action: str, body: Dict[str, Any], context: ClusterPlanContext
) -> Dict[str, Any]:
    node = context.enrolled_node(body.get("node_id", ""), "Enroll a node first.")
    draining = action == "drain-node"
    return {
        "title": (
            f"Drain {node['name']} for maintenance"
            if draining else f"Return {node['name']} to service"
        ),
        "steps": (
            [
                "Stop scheduling new cluster tasks on this worker.",
                "Let Swarm move replicated services to other eligible nodes.",
                "Keep SSH enrollment and cluster membership intact.",
            ]
            if draining else
            [
                "Mark this worker active in Docker Swarm.",
                "Allow new services and replacement tasks to use its resources.",
                "Keep existing node labels and placement rules.",
            ]
        ),
        "impact": (
            "Single-replica services without another eligible worker may stop while draining."
            if draining else
            "New cluster work may begin using this node immediately."
        ),
        "approval_required": True,
    }


def plan_evict_mismatched(
    action: str, body: Dict[str, Any], context: ClusterPlanContext
) -> Dict[str, Any]:
    """Plan the removal of nodes whose architecture is not the controller's.

    Built from the summary's own architecture reading, so the plan can only
    ever name nodes the controller has already classified as a *confirmed*
    mismatch. A node whose architecture could not be read never reaches here,
    and the operation re-checks the classification before it removes anything —
    the plan is what the operator reads, not what authorises the removal.
    """
    fleet = context.summary.get("architecture") or {}
    targets = [
        conflict for conflict in fleet.get("conflicts", [])
        if conflict.get("removable")
    ]
    requested = str(body.get("node_id", "")).strip()
    if requested:
        targets = [
            target for target in targets
            if str(target.get("node_id", "")) == requested
        ]
    if not targets:
        raise ClusterPlanError(
            "cluster_architecture_match",
            "No enrolled node is a confirmed architecture mismatch.",
            status=404,
        )
    named = ", ".join(
        f"{target.get('name') or target.get('host')} ({target.get('label')})"
        for target in targets
    )
    return {
        "title": (
            f"Drain and remove {len(targets)} mismatched worker"
            + ("s" if len(targets) != 1 else "")
        ),
        "steps": [
            f"This controller is {fleet.get('label', 'unknown')}; {named}.",
            "Stop scheduling on the mismatched worker and move its tasks off.",
            "Ask it to leave Swarm, then delete its manager record.",
            "Remove its fleet inventory and its encrypted SSH credential.",
        ],
        "impact": (
            "A Vaelor cluster keeps one processor architecture, and work "
            "placed on a mismatched worker cannot start. Its containers stop "
            "and it leaves the fleet. Nodes whose architecture could not be "
            "read are reported, not removed."
        ),
        "approval_required": True,
    }


def plan_remove_node(
    action: str, body: Dict[str, Any], context: ClusterPlanContext
) -> Dict[str, Any]:
    node = context.enrolled_node(
        body.get("node_id", ""), "The worker was not found."
    )
    joined = bool(node.get("labels", {}).get("swarm_node_id"))
    return {
        "title": f"Remove {node['name']} from this fleet",
        "steps": (
            [
                "Drain the worker so no new tasks are placed there.",
                "Ask the worker to leave Swarm cleanly.",
                "Remove its manager record, inventory, and encrypted SSH credential.",
            ]
            if joined else
            [
                "Remove the unjoined worker inventory.",
                "Delete its encrypted SSH credential.",
            ]
        ),
        "impact": (
            "Services without another eligible replica may be interrupted."
            if joined else
            "No cluster workloads are running on this enrollment."
        ),
        "approval_required": True,
    }
