"""Approval plans for controller initialization and worker membership changes."""

from __future__ import annotations

from typing import Any, Dict

from .cluster_network import private_controller_ipv4
from .cluster_node_removal import (
    FORCED_REMOVAL_ACK,
    LEFT_ON_MACHINE,
    LEFT_RUNNING_ON_DEAD_MACHINE,
    deployments_using_node,
    in_use_reason,
    machine_name,
    telemetry_provisioned,
)
from .cluster_plan_contract import ClusterPlanContext, ClusterPlanError
from .gpu_memory_pool import pool_left_note
from .cluster_service_state import STATEFUL_DRAIN_ACK

#: The refusal both removal plans give for an id that names no enrolled node.
_WORKER_NOT_FOUND = "The worker was not found."


def _drain_data_loss_apps(node_id: str, context: ClusterPlanContext) -> list:
    """Stateful apps pinned to this node that have no recovery backup — the ones
    a drain would take offline with no way to migrate their data (item 2). Read
    from the same facts the drain gate enforces so the plan states plainly what
    the operation will require."""
    constraint = f"node.labels.vaelor.node_id=={node_id}"
    store = context.callbacks.get("cluster_backups")
    names = []
    for service in context.summary.get("runtime", {}).get("services", []):
        name = str(service.get("name", ""))
        if not name.startswith("vaelor-app-"):
            continue
        try:
            details = context.manager.driver.service_details(name)
        except (AttributeError, OSError, ValueError):
            continue
        if constraint not in [str(value) for value in details.get("constraints", [])]:
            continue
        if not any(
            item.get("type") == "volume" for item in details.get("mounts", [])
        ):
            continue
        try:
            if not store.list(service_name=name, limit=1):
                names.append(name)
        except (AttributeError, OSError, ValueError):
            continue
    return names


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
    """The join plan: every step `ClusterOperations.join_node` and
    `DockerSwarmDriver.join_worker` take, and only the checks they make (ACC-124).

    It used to promise a re-check of the host key and hardware (the join reads
    the STORED inventory from enrolment or the last Recheck) and left out the
    two destructive steps the driver takes when the machine is still in a
    cluster: a forced ``docker swarm leave`` and the removal of that host's old
    entries from this controller's node list.
    """
    node = context.enrolled_node(body.get("node_id", ""), "Enroll a node first.")
    inventory = node.get("inventory") or {}
    steps = [
        "Connect over SSH, refusing the connection if the machine's host key "
        "is not the one pinned at enrolment.",
        "Check the hardware recorded at enrolment or at the last Recheck "
        "(processor architecture, memory, free disk); it is not re-read, so "
        "Recheck first if the machine has changed.",
    ]
    if not inventory.get("docker"):
        steps.append(
            "Install Docker from the operating system's packages, because it "
            "was not found at the last check."
        )
    steps += [
        "If the machine is still a member of any cluster - an earlier join to "
        "this one, or a different cluster - force it to leave that cluster "
        "first. Anything that cluster runs on the machine stops.",
        "Join with a one-time worker token, then rotate that token.",
        "If it was forced out of a cluster, remove the old entries this "
        "controller holds under the machine's hostname.",
        "Apply Vaelor node labels for architecture and memory-aware placement.",
    ]
    return {
        "title": f"Join {node['name']} as a worker",
        "steps": steps,
        "impact": (
            "The worker will accept cluster workloads from this head "
            "controller. A machine that belongs to another cluster is taken "
            "out of it."
        ),
        "approval_required": True,
    }


def plan_node_availability(
    action: str, body: Dict[str, Any], context: ClusterPlanContext
) -> Dict[str, Any]:
    node = context.enrolled_node(body.get("node_id", ""), "Enroll a node first.")
    draining = action == "drain-node"
    at_risk = _drain_data_loss_apps(node["id"], context) if draining else []
    plan = {
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
    if at_risk:
        # Data loss stated plainly (item 2): a stateful app pinned here keeps its
        # data on this machine and cannot be migrated. Draining it with no backup
        # loses that data, so the plan says so and carries the typed ack the
        # operation requires.
        plan["impact"] = (
            "This worker runs a pinned app that keeps its data on it and has "
            "no recovery backup ({}). Draining takes that app offline and its "
            "data cannot be migrated. Back it up first, or acknowledge the data "
            "loss to proceed.".format(", ".join(at_risk))
        )
        plan["data_loss_ack"] = STATEFUL_DRAIN_ACK
    return plan


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


#: A joined worker whose stored cluster entry will not decode (review round 2):
#: it is asked to leave from the machine itself, and the controller's own list
#: may keep a "down" entry for it, which the job's message names.
UNREAD_ENTRY_STEP = (
    "Its cluster entry on this controller could not be read, so ask the worker to "
    "leave the cluster from the machine itself. This controller's cluster list may "
    "then show it as down until that entry is removed there."
)


def plan_remove_node(
    action: str, body: Dict[str, Any], context: ClusterPlanContext
) -> Dict[str, Any]:
    """The removal plan, stating exactly what `cluster_node_removal` does (ACC-117).

    Refused outright while a model deployment still names the machine - the
    same check the job repeats - and otherwise listing the telemetry agent's
    removal, the swarm steps, the deletion, and what stays on the machine.
    """
    node = context.enrolled_node(
        body.get("node_id", ""), _WORKER_NOT_FOUND
    )
    store = context.manager.store
    in_use = deployments_using_node(store.list_pooled_deployments(), node["id"])
    if in_use:
        raise ClusterPlanError(
            "cluster_node_in_use", in_use_reason(machine_name(node), in_use),
            status=409,
        )
    swarm_id = node.get("labels", {}).get("swarm_node_id")
    joined = bool(swarm_id) or node.get("state") == "joined"
    steps = []
    if telemetry_provisioned(store, node["id"]):
        steps.append(
            "Remove Vaelor's telemetry agent from the machine and revoke its "
            "reporting key. If the machine cannot be reached, stop here and "
            "change nothing."
        )
    if joined and swarm_id:
        steps += [
            "Drain the worker so no new tasks are placed there.",
            "Ask the worker to leave the cluster cleanly, then delete its "
            "entry on this controller.",
        ]
    elif joined:
        steps.append(UNREAD_ENTRY_STEP)
    steps.append(
        "Delete its stored inventory and encrypted SSH credential. Vaelor "
        "cannot reach the machine after this."
    )
    return {
        "title": f"Remove {node['name']} from this fleet",
        "steps": steps,
        "impact": (
            ("Services without another eligible replica may be interrupted. "
             if joined else "Nothing in the cluster runs on this enrolment. ")
            + LEFT_ON_MACHINE
            # VD-161: said BEFORE the removal, while the console can still
            # take the setting off for the owner.
            + pool_left_note(node.get("inventory"))
        ),
        "approval_required": True,
    }


def plan_force_remove_node(
    action: str, body: Dict[str, Any], context: ClusterPlanContext
) -> Dict[str, Any]:
    """The FORCED removal plan (owner decision 2026-09-28), for a dead machine.

    Offered only after a plain removal was refused because the machine is in
    use or cannot be reached. It names every deployment that loses the machine
    and what may still be running on it, and it carries the typed
    acknowledgement the job requires. `cluster_node_removal` repeats every
    check when the job runs: a machine it CAN reach is refused.
    """
    node = context.enrolled_node(
        body.get("node_id", ""), _WORKER_NOT_FOUND
    )
    store = context.manager.store
    machine = machine_name(node)
    losing = deployments_using_node(store.list_pooled_deployments(), node["id"])
    steps = [
        "Check that {} cannot be reached. If it can, stop: remove its "
        "deployments normally instead.".format(machine) if losing else
        "Try to reach {} one last time for each step below; carry on "
        "without it where it does not answer.".format(machine),
    ]
    if losing:
        steps.append(
            "Mark {} degraded: {} lose{} this machine's part and must be "
            "removed and deployed again.".format(
                ", ".join(losing), "they" if len(losing) > 1 else "it",
                "" if len(losing) > 1 else "s",
            )
        )
    if node.get("labels", {}).get("swarm_node_id"):
        steps.append(
            "Force it out of the cluster and delete its entry on this controller."
        )
    if telemetry_provisioned(store, node["id"]):
        steps.append(
            "Try to remove its telemetry agent; if it does not answer, leave "
            "the agent there and revoke its reporting key."
        )
    steps.append("Delete its stored inventory and encrypted SSH credential.")
    return {
        "title": f"Remove {node['name']} by force",
        "steps": steps,
        "impact": LEFT_RUNNING_ON_DEAD_MACHINE,
        "approval_required": True,
        "data_loss_ack": FORCED_REMOVAL_ACK,
        "ack_prompt": (
            "A forced removal cannot be undone, and anything still running on "
            "{} is left there. Type {} to confirm.".format(
                machine, FORCED_REMOVAL_ACK
            )
        ),
    }
