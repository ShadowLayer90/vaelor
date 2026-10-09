"""Removing a vLLM GPU deployment: stop, leave, delete the credential, drop the row.

Extracted from `gpu_pool_operations.GpuPoolOperations.remove` (that module was at
999 lines) so the forced-worker-removal fix of review round 3 could land. The
method is now a one-line delegate; the order and the refusals are unchanged.

**A machine that has left the fleet cannot block the removal (review R1/R2).**
After an owner removed a dead worker by force, removing the deployment it
served failed with "Cluster node was not found." - AFTER the controller's units
were stopped and BEFORE `leave_if_owned`, so AI Chat was never restored. The
exemption is derived from the cluster store itself (`departed_node_ids`: an id
that is neither this controller nor an enrolled node), never from the
``lost_nodes`` marker on the record: that marker is display only, and the mode
watch's teardown, an unload's write or a load's write can each put a stale copy
of the units back over it.

Runs in the executor (the job) and, for a preview-side instance, the control
plane; the mode switch is only ever the executor's.
"""

from __future__ import annotations

import logging
from typing import Any, Dict, List, Mapping

from .cluster_gpu_sizing import VERDICT_DISTRIBUTED, VERDICT_REPLICATED
from .cluster_placement import departed_node_ids
from .gpu_pool_replicas import stop_balancer
from .gpu_pool_units import (
    VLLM_ENGINE, describe_left_behind, plane_left, record_mode, remove_blockers,
    serving_units, unstopped,
)
from .gpu_ray_plane import RayPlane

LOGGER = logging.getLogger(__name__)

#: What `remove` raises when a unit it derived would not stop. The record is
#: KEPT: dropping it would delete the only product-side handle on a container
#: still holding a node's GPU, which is the defect the derivation exists to
#: close. The sentence names every unit and node, so the operator can clear it
#: by hand or simply remove again once the node answers.
UNITS_NOT_STOPPED = (
    "{} The deployment record was kept; clear that on the machine named, or "
    "remove this deployment again to retry."
)


def remove_gpu_deployment(ops: Any, payload: Dict[str, Any]) -> Dict[str, Any]:
    """Stop a GPU deployment's units and drop its record.

    Accepts ANY vLLM record - healthy, failed, or the ``deploying`` row a
    deploy that died with its executor left behind - because the units to
    stop are derived from the record (`stop_units_for`), never read off a
    list only some records carried. A rollback is best-effort against a
    cluster that has already misbehaved, so it can leave a vLLM container
    holding a node's GPU; refusing to remove the record that names it left
    the operator with no product-side way to clear it. The pull oneshots a
    failed deploy recorded are stopped too.

    **The GPU marker gates every state, because the two tiers share one
    table.** `pooled_operations` writes its own ``failed`` record with no
    ``units`` at all, so relaxing the gate for "failed and unit-less" handed
    this method a CPU llama.cpp deployment to resolve GPU transports for and
    delete. Every vLLM record - the ``deploying`` one included - is stamped
    with ``engine``, so a GPU deploy that failed before its first unit is
    still removable without any relaxation at all.

    **The order is stop, leave, delete the credential, drop the record
    (D9).** `leave` restores the ``ai-chat`` lease, and the broker's
    ``delete`` cascades a credential's purpose assignments - so deleting
    first would clear the lease out from under the restore and leave a window
    with AI Chat pointing at nothing. Stopping the units first is what makes
    the restore true rather than optimistic: the 30 s GPU failure-watch
    relaunches llama.cpp on the memory those units were holding. A unit
    that would NOT stop is therefore a refusal (:data:`UNITS_NOT_STOPPED`)
    with the record kept, not a row quietly dropped over a running server -
    unless its node has left the fleet, which nothing can ever stop again.

    **Nothing between the stops and the leave may raise.** A pull oneshot
    whose node cannot be resolved or reached is logged and passed over, so
    the ``ai-chat`` restore always runs once the units are down.

    **The record is dropped under the GPU serving lock** (VD-129). The
    mode watch reads a replicated row, probes its replicas - up to five
    seconds each, outside the lock, because a probe is not a thing to
    hold the lock across - and writes the liveness back under the lock
    after re-reading the row (`gpu_cluster_mode_watch._observe_replicas`).
    Dropping the row under the same lock is what makes that re-read
    decisive: a removal cannot land between the re-read and the write, so
    a removed row is never resurrected as a healthy "0 of 2 serving" that
    refuses the next deploy of the same name. The stops above stay
    outside it, as VD-127's third round chose.
    """
    if payload.get("confirm") != "remove-gpu-inference":
        raise ValueError("Confirm the reviewed GPU deployment removal.")
    name = str(payload.get("name", "")).strip().lower()
    deployment = ops.store.get_pooled_deployment(name)
    if deployment is None:
        raise ValueError("The GPU deployment was not found.")
    units = deployment.get("units", {}) or {}
    if units.get("engine") != VLLM_ENGINE:
        raise ValueError("That deployment is not a vLLM GPU deployment.")
    departed = departed_node_ids(ops.store, deployment.get("node_ids") or [])
    not_stopped = ops.stop_units_for(deployment)
    # Which of those block the removal is the record's mode's rule
    # (`gpu_pool_units.remove_blockers`): every one for a distributed
    # record, the controller's alone for a replicated one, whose worker
    # replica on a node that left or was unplugged must not hold the row
    # hostage (VD-129) - and never one on a node that left the fleet.
    blockers = remove_blockers(deployment, not_stopped, departed)
    if blockers:
        raise RuntimeError(
            UNITS_NOT_STOPPED.format(describe_left_behind(blockers))
        )
    for entry in not_stopped:
        LOGGER.warning(
            "Removing %s with %s still to be stopped on %s: %s", name,
            entry.get("unit"), entry.get("node"), entry.get("error"),
        )
    for entry in units.get("pulls", []) or []:
        node_id = str((entry or {}).get("node_id", ""))
        unit = str((entry or {}).get("unit", ""))
        if not node_id or not unit or node_id in departed:
            continue
        try:
            ops._pull.stop_quietly(
                ops._transport(ops.joined_node(node_id, include_credential=True)),
                unit,
            )
        except Exception as error:  # noqa: BLE001 - the restore must still run
            LOGGER.warning(
                "Removing %s: could not stop pull %s on %s: %s",
                name, unit, node_id, error,
            )
    restored = (
        ops.mode_switch.leave_if_owned(name)
        if ops.mode_switch is not None else {}
    )
    credential_id = str(deployment.get("credential_id", ""))
    if credential_id:
        ops.broker.delete(credential_id)
    with ops._serving_lock():
        ops.store.delete_pooled_deployment(name)
    return {
        "name": name,
        "removed": True,
        "model_cache_retained": True,
        "runtime_retained": True,
        "mode": restored,
        "unstopped": not_stopped,
    }


# --- the stops `GpuPoolOperations` makes -------------------------------------
#
# Moved here beside `remove`, their main caller, when the split's Ray plane
# took `gpu_pool_operations` past its ceiling; the operations keep
# `_stop_each` and `stop_units_for` as the names every caller uses.

def stop_placements(ops: Any, placements: List[Any]) -> List[Dict[str, str]]:
    """Stop ``(transport, node, unit)`` placements, last started first.

    The ONE stop loop: the deploy's rollback hands it what it started, and
    `stop_derived_units` hands it what a record derives. Best-effort by
    necessity - it runs because something already went wrong, often the
    node itself - but a swallowed failure here leaves a vLLM container
    holding a node's whole GPU with nothing naming it. Each failure is
    logged with the node and the unit AND returned as `unstopped`, so the
    caller can put those two facts on the record rather than in a log
    nobody reads until the GPU is found full. A split's Ray plane
    (`gpu_ray_plane.RayPlane`) is cleared too; one that could not be (its
    slice would not stop, or its firewall would not go) is returned the same
    way, so `remove` keeps the record naming it (ACC-187 review 2).
    """
    failures: List[Dict[str, str]] = []
    for transport, node, unit in reversed(placements):
        if isinstance(unit, RayPlane):  # never raises; a failure is NAMED (SC5)
            failure = ops.runtime.clear_ray_plane(transport, unit.deployment)
            if failure:
                failures.append(plane_left(
                    node.get("id", ""), node.get("name", node.get("id")), failure))
            continue
        try:
            ops.runtime.stop_unit(transport, unit)
        except Exception as error:  # noqa: BLE001 - never mask the cause
            name = node.get("name", node.get("id"))
            LOGGER.warning("Could not stop %s on %s: %s", unit, name, error)
            failures.append(unstopped(node.get("id", ""), name, unit, error))
    return failures


def stop_derived_units(ops: Any, deployment: Mapping[str, Any]) -> List[Dict[str, str]]:
    """Stop every serving unit ``deployment`` DERIVES, each on its own node.

    The stop both `remove` and the mode watch's abandoned pass make, and
    the only one either makes: the units come from
    `gpu_pool_units.serving_units` - the record's name and node list - never
    from a list stored on the record, so a ``deploying`` row a dead deploy
    left behind is stopped exactly as a healthy one is. Each unit goes
    through the transport of the node that holds it (the bridge for this
    controller, SSH for a worker), one transport per machine. The runtime's
    `stop_unit` treats a unit systemd never loaded as a no-op, so deriving a
    worker the deploy never reached costs one read and no false failure.

    Never raises. A node that cannot be resolved (removed from the cluster
    since) or reached is a failure NAMED with its unit and returned: the
    watch writes the list on the row, and `remove` refuses to drop a row
    that still names something running - every one of them for a
    distributed record, the controller's alone for a replicated one
    (`gpu_pool_units.remove_blockers`).

    A split's Ray plane is cleared on every machine, after its units
    (ACC-163). A replicated record's balancer (VD-129) is stopped here too,
    after its replicas; a mode-less record gets both, because what is not
    there costs nothing to stop and a record written before the word existed
    can then orphan nothing.
    """
    placements: List[Any] = []
    failures: List[Dict[str, str]] = []
    # Listed FIRST so the reversed stop clears the plane after its units.
    planes = [] if record_mode(deployment) == VERDICT_REPLICATED else [
        (str(node_id), RayPlane(str(deployment.get("name", "") or "")))
        for node_id in (deployment.get("node_ids") or []) if str(node_id)
    ]
    reached: Dict[str, Any] = {}
    for node_id, unit in planes + serving_units(deployment):
        try:
            if node_id not in reached:
                node = ops.joined_node(node_id, include_credential=True)
                reached[node_id] = (ops._transport(node), node)
            placements.append((*reached[node_id], unit))
        except Exception as error:  # noqa: BLE001 - name it, keep going
            LOGGER.warning("Could not reach %s to stop %s: %s", node_id, unit, error)
            if not isinstance(unit, RayPlane):
                failures.append(unstopped(node_id, node_id, unit, error))
    failures += ops._stop_each(placements)
    if record_mode(deployment) != VERDICT_DISTRIBUTED:
        failures += stop_balancer(ops.balancer, str(deployment.get("name", "") or ""))
    return failures
