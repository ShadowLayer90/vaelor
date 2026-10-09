"""Manual LOAD / UNLOAD of a cluster vLLM deployment (G3a).

Unloading stops a healthy vLLM deployment's serving units to give a node's
GPU (and this controller's unified memory) back, WITHOUT deleting the
deployment: its ``pooled_deployments`` row, its credential-broker profile and
its cached weights are all kept, and its state becomes ``unloaded``. Loading
re-serves an ``unloaded`` deployment warm - the weights are already on disk
(the model cache) and the compile cache persists, so the replicas or the
sharded server come back in seconds rather than after a fresh pull.

**Why this is not `remove`, and not `deploy`.** `gpu_pool_operations.remove`
tears everything down - it stops the units, `leave`s Mode B (restoring AI Chat
to llama.cpp), deletes the credential and drops the row. Unload does the FIRST
of those and none of the rest: the deployment is paused, not gone. `deploy`
builds a new deployment from a fit plan and a weights pull, entering the mode
switch; load rebuilds an EXISTING one from its stored record, reusing the
deploy start path MINUS the pull and MINUS `enter` (Mode B is already held).

**This lives in its own module because the serving-core modules are at the
1,000-line ceiling** (`gpu_pool_operations` and `gpu_pool_runtime` both sit at
or one line under it). The two entry points are functions over a
`GpuPoolOperations` instance - the same "shared behavior in a focused module"
shape `gpu_pool_replicas.rotate_cluster_serving_key` uses - and
`cluster_operations.ClusterOperations.unload_gpu_inference`/`load_gpu_inference`
wire them, exactly as the GPU deploy and remove are wired.

**The ai-chat lease is parked honestly on unload (design C3).** A
controller-led deployment holds the ``ai-chat`` lease on its cluster
credential (the mode switch's `repoint` put it there). While the deployment is
unloaded that endpoint is dead, so pointing AI Chat at it would give a
misleading 502. Unload therefore parks the lease exactly as
`ClusterModeSwitch.enter._degrade_ai_chat` does when clustering first takes
over - onto the Assistant's on-device connection when the box has one, else
cleared - so AI Chat degrades to an honest "model not loaded / unavailable"
state rather than hanging. The mode file stays Mode B, so the GPU
failure-watch never relaunches llama.cpp into the freed aperture, and the mode
reconcile's ``unloaded`` branch converges nothing. Load re-points AI Chat back
at the (unchanged) cluster credential once the deployment is healthy again.

**Only while AI Chat follows the cluster (VD-210).** A connection the owner
chose in AI Chat is neither parked by an unload nor taken back by a Load
(`gpu_cluster_ai_chat`), so the lease no longer says which unload this was:
the unload records its cause in the mode file instead, where the LLM Server's
door reads it.
"""

from __future__ import annotations

import logging
from typing import Any, Dict, List

from .cluster_gpu_sizing import VERDICT_DISTRIBUTED, VERDICT_REPLICATED
from .cluster_link import confirm_recorded_bindings
from .cluster_llm_deploy import refuse_out_of_service_nodes
from .cluster_placement import departed_node_ids
from .gpu_cluster_ai_chat import record_unload, resume_after_load
from .gpu_cluster_mode import DEPLOYING_STATE, HEALTHY_STATE, UNLOADED_STATE
from .gpu_idle_watch import idle_timeout_from_payload, is_deployment_idle_now
from .gpu_pool_replicas import _reput_cluster_key, stop_balancer
#: What each unit a Load installs was rendered from (W4-D1).
from . import gpu_render_ledger as render_ledger
#: A row the watch failed is Loaded through its own recovery (W4-D7).
from .gpu_pool_recover import FAILED_STATE, LOAD_IT_AGAIN, load_failed_deployment
from .model_thinking import thinking_from_record
#: A Load renders the serving options the record carries, or - for a record
#: from before they existed - the ones its repo's profile implies.
from .gpu_pool_refit import WEIGHT_FIELD, options_for_load_checked
from .vllm_images import download_line
from .vllm_serve_options import CONFIG_READ_FIELD
from .gpu_pool_units import (
    RAY_PLANE_KIND, VLLM_ENGINE, deployment_name, describe_unstopped, record_mode,
    remove_blockers,
)
from .gpu_serving_target import (
    CLUSTER_INFERENCE_PURPOSE, MODE_CLUSTER,
)

LOGGER = logging.getLogger(__name__)

#: The confirm token an unload request must carry, mirroring `remove`'s.
UNLOAD_CONFIRM = "unload-gpu-inference"
#: The confirm token a load request must carry.
LOAD_CONFIRM = "load-gpu-inference"

#: Refused when a unit an unload derived would not stop, and it is one whose
#: node keeps the deployment (`gpu_pool_units.remove_blockers`): the deployment
#: is LEFT serving rather than half-recorded as unloaded, exactly as `remove`
#: keeps the row when a controller unit will not stop.
UNITS_NOT_STOPPED = (
    "Could not stop {}. The deployment was left serving; stop it on that node "
    "or unload this deployment again to retry."
)

#: Refused when unload is asked of a deployment that is not serving. Only a
#: healthy deployment has units to stop; an already-unloaded or failed one has
#: none, and re-running unload over it would be a no-op dressed as an action.
NOT_HEALTHY_TO_UNLOAD = (
    "This GPU deployment reads '{}', not healthy, so nothing is serving that "
    "unload could reclaim."
)

#: Refused when unload is asked of a deployment the watch (or its deploy)
#: failed: nothing of it serves, and Load is what starts it again (W4-D7).
FAILED_TO_UNLOAD = (
    "This GPU deployment reads 'failed', so nothing of it is serving to unload. "
    + LOAD_IT_AGAIN
)

#: Refused when load is asked of a deployment that is not in the unloaded rest
#: state (a healthy one is answered as an idempotent success above this).
NOT_UNLOADED_TO_LOAD = (
    "This GPU deployment reads '{}', not unloaded, so there is nothing paused "
    "to load; unload it first to pause it without removing it."
)

#: Refused when a distributed record predates G3a and so never recorded the
#: launch settings a warm reload needs (the fraction vLLM was launched at and
#: the context the KV cache was sized for). Such a deployment cannot be
#: rebuilt from its record alone; a fresh deploy is the honest path.
PREDATES_LOAD = (
    "This GPU deployment was created before load support and did not record "
    "its launch settings, so it cannot be loaded warm; remove it and deploy "
    "it again."
)

#: Refused when a record names a machine that has left the fleet and carries
#: no plain reason of its own (review R6).
DEPARTED_TO_LOAD = (
    "This GPU deployment uses a machine that is no longer in the fleet ({}), "
    "so it cannot be loaded; remove it and deploy it again."
)

#: Refused when a record names no machine at all (review A5): a Load has
#: nowhere to serve it, and the recovery used to fail on a bare IndexError.
NO_MACHINES_TO_LOAD = (
    "This GPU deployment's record names no machine to load it on, so it cannot "
    "be loaded; remove it and deploy it again."
)

#: The one message for "no such deployment", worded for this pair of ops.
NOT_FOUND = "No GPU deployment by that name was found."

#: The one message for "not a vLLM record", worded for this pair of ops.
NOT_VLLM = "That deployment is not a vLLM GPU serving deployment."


def _switch_owns(ops: Any, name: str) -> bool:
    """Whether the mode switch currently holds Mode B for THIS deployment.

    The gate for touching the ``ai-chat`` lease at all: only a controller-led
    (Mode B) deployment ever moved that lease onto its cluster credential, so
    only for one does parking or restoring it mean anything. A worker-led
    deployment left the mode file reading Mode A and AI Chat untouched, and a
    control-plane instance of the operations has no switch (D12).
    """
    switch = getattr(ops, "mode_switch", None)
    if switch is None:
        return False
    state = switch.state()
    return state.mode == MODE_CLUSTER and state.deployment_name == name


def _park_ai_chat(ops: Any, name: str, *, idle: bool) -> None:
    """Record the unload's cause, and move ``ai-chat`` off a manually unloaded cluster.

    The switch's own `_degrade_ai_chat` clears the lease (the parking `enter`
    does when clustering first takes over), so AI Chat asks for a connection
    rather than timing out against the dead cluster port - and only while AI
    Chat is on the cluster: the owner's own connection is never taken
    (`gpu_cluster_ai_chat.record_unload`, VD-210). An idle unload parks
    nothing (BL-1). Only when this deployment holds Mode B.
    """
    if _switch_owns(ops, name):
        record_unload(ops.mode_switch, name, idle=idle)


def _stop_lan_proxy(ops: Any, name: str, *, units_stopped: bool = True) -> None:
    """Move the LLM Server's door off the now-dead serving port (FIX 1, VD-159).

    A proxy left up in front of a stopped upstream answers the LAN with an
    nginx 502, so the door is moved at once: onto the wake responder after an
    IDLE unload (ACC-058), and onto the door that answers "not loaded" for
    itself in every other case - the same step the reconcile's unloaded pass
    makes (`ClusterModeSwitch.converge_unloaded_doors`), taken here so the
    port is not closed until that pass comes round. A door that cannot be
    moved is stopped, as before. Only when this deployment holds Mode B, and
    best effort.

    Called twice by an unload. First with ``units_stopped`` False, the
    moment the row reads unloaded and BEFORE a unit is stopped, so no request
    meets a 502 while the units go down. That move is the LLM Server's door
    and nothing else: never the wake door, because the lease still names the
    cluster and a wake would load the model the unload is stopping; and
    never the balancer, which fronts replicas that are still up and stay up
    if the unload cannot stop them (re-review R1). Then again once the units
    are down, when the balancer stops and an idle unload gets its wake door.
    """
    switch = getattr(ops, "mode_switch", None)
    if switch is None or not _switch_owns(ops, name):
        return
    try:
        switch.converge_unloaded_doors(
            allow_wake=units_stopped, stop_balancer=units_stopped,
        )
        return
    except Exception as error:  # noqa: BLE001 - fall back to the plain stop
        LOGGER.warning(
            "Could not move the LLM Server's door while unloading %r: %s",
            name, error,
        )
    try:
        switch.llm_proxy.stop()
    except Exception as error:  # noqa: BLE001 - the reconcile keeps it down too
        LOGGER.warning(
            "Could not stop the LLM Server proxy while unloading %r: %s",
            name, error,
        )


def _serve_lan_proxy(ops: Any, name: str) -> None:
    """Point the LLM Server's door at the serving cluster at once (review S5).

    The mirror of :func:`_stop_lan_proxy`, for a Load that just finished and
    for an unload that had to put its row back to healthy: the model is
    serving, so the door must not go on answering "not loaded" until the next
    reconcile pass. The balancer is converged from the row as it now is, too
    (re-review R1): the replicas are serving, so their front must be up.
    Only when this deployment holds Mode B, and best effort - the reconcile
    converges both regardless.
    """
    switch = getattr(ops, "mode_switch", None)
    if switch is None or not _switch_owns(ops, name):
        return
    try:
        switch.converge_serving_door(ops.store.get_pooled_deployment(name))
    except Exception as error:  # noqa: BLE001 - the reconcile converges it anyway
        LOGGER.warning(
            "Could not point the LLM Server's door at %r at once: %s", name, error,
        )


def _restore_ai_chat(ops: Any, name: str, credential_id: str) -> bool:
    """Point ``ai-chat`` back at the reloaded cluster credential (design 3).

    Mirrors the reconcile's healthy pass: when this deployment holds Mode B,
    the lease is re-activated on its (unchanged) cluster credential so AI Chat
    comes back on the reloaded model at once rather than after the next 30 s
    reconcile. Best effort - a healthy load is never failed over a lease move,
    which the reconcile would repair regardless.
    """
    if not credential_id or not _switch_owns(ops, name):
        return False
    try:
        # VD-210: and only while AI Chat follows the cluster.
        return resume_after_load(ops.mode_switch, name, credential_id)
    except Exception as error:  # noqa: BLE001 - the reconcile re-points anyway
        LOGGER.warning(
            "AI Chat could not be pointed back at the reloaded cluster '%s': %s",
            name, error,
        )
        return False


def unload_deployment(ops: Any, payload: Dict[str, Any]) -> Dict[str, Any]:
    """Stop a healthy vLLM deployment's units, keeping its record and credential.

    Confirm-gated, engine-gated (a non-vLLM row is refused, like `remove`), and
    state-gated: only a ``healthy`` deployment is unloadable. The unit stops run
    OUTSIDE the serving lock (VD-127); a unit whose node keeps the deployment
    (`remove_blockers`) that will not stop is a refusal with the deployment left
    stopped, never a row quietly marked unloaded over a running server. The row
    is flipped to ``unloaded`` UNDER the lock BEFORE the stops (FIX 2), so a
    reconcile tick in the stop window converges the LAN doors down instead of
    restarting the balancer or false-failing Mode B; a blocker reverts the row
    to ``healthy``. On a manual unload the ``ai-chat`` lease is parked (design
    C3); an idle-triggered unload instead HOLDS it on the unloaded cluster so a
    later send wakes the model (BL-1). The LLM Server's door is moved off
    the stopped model either way (FIX 1, VD-159); everything else - units
    metadata, model id, credential id, endpoint - is kept for `load`.
    """
    if (payload or {}).get("confirm") != UNLOAD_CONFIRM:
        raise ValueError("Confirm the reviewed GPU deployment unload.")
    name = deployment_name(str((payload or {}).get("name", "")))
    deployment = ops.store.get_pooled_deployment(name)
    if deployment is None:
        raise ValueError(NOT_FOUND)
    units = deployment.get("units", {}) or {}
    if units.get("engine") != VLLM_ENGINE:
        raise ValueError(NOT_VLLM)
    state = str(deployment.get("state", ""))
    if state == FAILED_STATE:
        raise ValueError(FAILED_TO_UNLOAD)
    if state != HEALTHY_STATE:
        raise ValueError(NOT_HEALTHY_TO_UNLOAD.format(state))

    model_id = str(deployment.get("model_id", "") or "")
    node_ids = [str(node_id) for node_id in (deployment.get("node_ids") or [])]
    endpoint = str(deployment.get("endpoint", "") or "")
    credential_id = str(deployment.get("credential_id", "") or "")
    # G3b: an idle-triggered unload the mode reconcile enqueued carries
    # ``reason: "idle"``, and differs from a manual unload in two ways below - a
    # fresh idle re-check before the stops (CN-1), and holding the
    # ``ai-chat`` lease on the unloaded cluster rather than parking it (BL-1).
    idle_unload = str((payload or {}).get("reason", "") or "") == "idle"

    def _write(row_state):
        with ops._serving_lock():
            ops.store.put_pooled_deployment(
                name=name, state=row_state, model_id=model_id,
                node_ids=node_ids, units=units, endpoint=endpoint,
                credential_id=credential_id,
            )

    # CN-1: the enqueue widened the gap between reading idle and acting on it,
    # so an idle unload re-reads idleness FRESHLY, off the serving lock, the
    # instant before it stops anything, and no-ops if a request has landed
    # since - leaving the row healthy for the reconcile to keep serving. The
    # scrape needs no lock: user requests reach vLLM directly and never take
    # GPU_SERVING_LOCK, so holding it could not stop a late request anyway;
    # only the row-state flip below does, and it keeps the lock. A manual
    # unload keeps its unconditional stop.
    if idle_unload:
        still_idle = is_deployment_idle_now(ops, deployment)
        if not still_idle:
            return {
                "name": name,
                "unloaded": False,
                "state": HEALTHY_STATE,
                "reason": "not-idle",
            }
    # FIX 2: flip the row to ``unloaded`` UNDER the lock BEFORE stopping the
    # units, so a 30 s reconcile tick landing in the sub-second stop window
    # reads ``unloaded`` (`_unloaded_pass`, converge-down) rather than
    # ``healthy`` - which would restart the balancer, or on a slow stop
    # false-fail Mode B after four REPLICA_DOWN passes. The stops stay
    # outside the lock (VD-127).
    _write(UNLOADED_STATE)
    # Review S5: the door leaves the model NOW, before the first unit is
    # stopped, so no request meets a 502 while they go down.
    with ops._serving_lock():
        _stop_lan_proxy(ops, name, units_stopped=False)
    not_stopped = ops.stop_units_for(deployment)
    # Every unit stopped but a split's firewall would not clear: the model IS
    # unloaded, so the row says so and carries the note (review 3, SC1); the
    # next Load fences again, and `remove` refuses until the plane is clear.
    planes_left = [entry for entry in not_stopped if entry.get("kind") == RAY_PLANE_KIND]
    not_stopped = [entry for entry in not_stopped if entry.get("kind") != RAY_PLANE_KIND]
    if planes_left:
        units["plane_cleanup"] = planes_left
        _write(UNLOADED_STATE)
    blockers = remove_blockers(
        deployment, not_stopped, departed_node_ids(ops.store, node_ids)
    )
    if blockers:
        # A unit whose node keeps the deployment would not stop: revert to a
        # consistent serving row and refuse, the way `remove` keeps the row.
        # The healthy reconcile pass then reconverges the LAN doors up. A
        # clean-up note belongs to an unloaded row only (review 4).
        units.pop("plane_cleanup", None)
        _write(HEALTHY_STATE)
        with ops._serving_lock():
            _serve_lan_proxy(ops, name)
        raise RuntimeError(UNITS_NOT_STOPPED.format(describe_unstopped(blockers)))
    for entry in not_stopped:
        LOGGER.warning(
            "Unloading %s with %s still to be stopped on %s: %s", name,
            entry.get("unit"), entry.get("node"), entry.get("error"),
        )

    # A successful unload only: park the ai-chat lease (design C3) and take
    # the LLM Server LAN proxy down (FIX 1), both under the lock. The row is
    # already ``unloaded`` from the flip above.
    with ops._serving_lock():
        # BL-1: a manual unload parks ai-chat onto the Assistant; an idle
        # unload holds it on the unloaded cluster credential so the next send
        # wakes it, rather than silently downgrading to the on-device model.
        _park_ai_chat(ops, name, idle=idle_unload)
        _stop_lan_proxy(ops, name)
    return {
        "name": name,
        "unloaded": True,
        "state": UNLOADED_STATE,
        "freed_nodes": node_ids,
        "model_cache_retained": True,
        "credential_retained": True,
        "unstopped": not_stopped,
    }


def load_deployment(
    ops: Any, payload: Dict[str, Any], progress=None, placement_status=None,
) -> Dict[str, Any]:
    """Re-serve an unloaded vLLM deployment warm, from its stored record.

    Confirm-gated and engine-gated. A load of an already-``healthy`` row is an
    idempotent no-op success; any state but ``unloaded`` or ``healthy`` is
    refused. The start path is the deploy's, MINUS the weights pull (cached) and
    MINUS the mode switch's `enter` (Mode B is already held): the image is
    ensured and the render/video GIDs re-read per node, then the replicated or
    the distributed serving body re-runs from the record's own settings and
    waits on the endpoint. The credential is reused, not re-minted - the row is
    written ``deploying`` while loading (a window the mode reconcile leaves alone
    because `cluster.gpu.load` counts as a deploy in flight, B2), then ``healthy``
    - and AI Chat is repointed at it. A failure stops what it started and returns
    the row to ``unloaded`` so it can be retried.
    """
    if (payload or {}).get("confirm") != LOAD_CONFIRM:
        raise ValueError("Confirm the reviewed GPU deployment load.")
    name = deployment_name(str((payload or {}).get("name", "")))
    deployment = ops.store.get_pooled_deployment(name)
    if deployment is None:
        raise ValueError(NOT_FOUND)
    units = dict(deployment.get("units", {}) or {})
    if units.get("engine") != VLLM_ENGINE:
        raise ValueError(NOT_VLLM)
    state = str(deployment.get("state", ""))
    if state == HEALTHY_STATE:
        return {
            "name": name, "loaded": True, "state": HEALTHY_STATE,
            "already_healthy": True,
        }
    if state == FAILED_STATE:
        # W4-D7: the recovery for a row the watch failed (`gpu_pool_recover`).
        return load_failed_deployment(ops, deployment, progress, placement_status)
    if state != UNLOADED_STATE:
        raise ValueError(NOT_UNLOADED_TO_LOAD.format(state))

    report = progress or (lambda _percent, _message: None)
    participants = load_participants(ops, deployment, placement_status)
    participant_ids = [node["id"] for node in participants]
    credential_id = str(deployment.get("credential_id", "") or "")
    mode = record_mode(deployment)
    replicated = mode == VERDICT_REPLICATED
    repo = str(units.get("repo", "") or deployment.get("model_id", "") or "")
    revision = units.get("revision")
    port = int(units.get("port", 0) or 0)
    gpu_memory_utilization = float(units.get("gpu_memory_utilization", 0) or 0)
    max_model_len = int(units.get("max_model_len", 0) or 0)
    # The thinking default the record carries; a record from before the
    # setting existed carries none and is served with thinking OFF from this
    # Load on (owner decision 2026-09-29) - never the old think-by-default.
    thinking = thinking_from_record(units)
    options, config_read, cache_note = load_options(ops, deployment, participants)
    transports = {node["id"]: ops._transport(node) for node in participants}
    confirm_load_bindings(ops, deployment, participants, transports)

    with ops._serving_lock():
        ops.store.put_pooled_deployment(
            name=name, state=DEPLOYING_STATE, model_id=repo,
            node_ids=participant_ids, units=units,
            endpoint=str(deployment.get("endpoint", "") or ""),
            credential_id=credential_id,
        )
    started: List[Any] = []
    try:
        group_ids: Dict[str, List[int]] = {}
        for index, node in enumerate(participants):
            report(
                10 + int(25 * index / len(participants)),
                f"Readying the cached vLLM image on {node['name']} to load",
            )
            ops.runtime.ensure_image(
                transports[node["id"]], image=options.vllm_image,
                on_download=lambda profile, node=node: report(
                    12, download_line(profile, node["name"]),
                ),
            )
            group_ids[node["id"]] = ops.runtime.resolve_group_ids(
                transports[node["id"]]
            )
        with render_ledger.recording() as renders:
            if replicated:
                new_units, endpoint = _load_replicated(
                    ops, name, participants, transports, group_ids, repo, revision,
                    port, units, gpu_memory_utilization, max_model_len, payload,
                    report, started, credential_id, thinking, options.as_record(),
                )
            else:
                new_units, endpoint = _load_distributed(
                    ops, name, participants, participant_ids, transports, group_ids,
                    repo, revision, port, units, gpu_memory_utilization,
                    max_model_len, payload, report, started, thinking,
                    options.as_record(), credential_id,
                )
        # What every unit was rendered from, for the next upgrade (W4-D1).
        new_units[render_ledger.RENDERS_FIELD] = render_ledger.entries(renders, transports)
        # What the record said about its model's layout and weight size is
        # carried to the loaded record, as the scale-to-zero window is below:
        # the serve rebuild does not carry them, and a later Load reads both.
        if CONFIG_READ_FIELD in units:
            new_units[CONFIG_READ_FIELD] = config_read
        if WEIGHT_FIELD in units:
            new_units[WEIGHT_FIELD] = units[WEIGHT_FIELD]
        # G3b (BL-1): the serve rebuild does not carry idle_timeout - it is
        # not a serve launch arg but the scale-to-zero policy the record keeps
        # - so re-stamp it from the kept (unloaded) record the SAME way deploy
        # stamps it after its own serve, preserving 0/absent as off. Without
        # this the woken model loses its window and never auto-unloads again.
        idle_timeout = idle_timeout_from_payload(units)
        if idle_timeout:
            new_units["idle_timeout"] = idle_timeout
        with ops._serving_lock():
            ops.store.put_pooled_deployment(
                name=name, state=HEALTHY_STATE, model_id=repo,
                node_ids=participant_ids, units=new_units, endpoint=endpoint,
                credential_id=credential_id,
            )
            restored = _restore_ai_chat(ops, name, credential_id)
            # Review S5: and the LLM Server's door with it, in the same step.
            _serve_lan_proxy(ops, name)
    except Exception:
        not_stopped = ops._stop_each(started)
        if replicated:
            not_stopped += stop_balancer(ops.balancer, name)
        with ops._serving_lock():
            ops.store.put_pooled_deployment(
                name=name, state=UNLOADED_STATE, model_id=repo,
                node_ids=participant_ids, units=units,
                endpoint=str(deployment.get("endpoint", "") or ""),
                credential_id=credential_id,
            )
        raise

    report(100, "The vLLM deployment is loaded and serving behind the gateway")
    return {
        "name": name,
        "loaded": True,
        "state": HEALTHY_STATE,
        "mode": mode,
        "node_ids": participant_ids,
        "endpoint": endpoint,
        "credential_id": credential_id,
        "ai_chat_restored": restored,
        # Why the unit was served as deployed although the model's layout
        # has since been read ("" otherwise).
        "cache_note": cache_note,
    }


def load_participants(
    ops: Any, deployment: Dict[str, Any], placement_status=None,
) -> List[Dict[str, Any]]:
    """The machines a Load of this record would serve on, or the Load's refusal.

    Every check a Load makes of the record and the fleet before it builds a
    transport, in the Load's own words: a machine that has left the fleet, a
    machine drained out of service, a split recorded before Load existed. The
    upgrade's refresh (`gpu_pool_refresh`) asks the SAME checks of a serving
    record before it stops anything, so a refresh never stops a model that the
    Load it is about to run would refuse.
    """
    units = deployment.get("units", {}) or {}
    node_ids = [
        str(node_id) for node_id in (deployment.get("node_ids") or [])
        if str(node_id)
    ]
    if not node_ids:
        raise ValueError(NO_MACHINES_TO_LOAD)
    # Review R6: a machine that has left the fleet (a forced removal) cannot
    # be loaded onto; say so with the record's own reason instead of the bare
    # "Cluster node was not found." the resolver below would raise.
    departed = departed_node_ids(ops.store, node_ids)
    if departed:
        raise ValueError(
            str(units.get("degraded_reason", "") or "")
            or DEPARTED_TO_LOAD.format(", ".join(sorted(departed)))
        )
    # The record wrote its nodes lead-first (`gpu_pool_units.lead_first`), and
    # `serving_units` trusts that order, so the participants are rebuilt in it.
    participants = [
        ops.joined_node(node_id, include_credential=True) for node_id in node_ids
    ]
    # Review R6: the same drain rule every model deploy follows - a load puts
    # model servers on each machine exactly as a deploy does.
    if placement_status is not None:
        refuse_out_of_service_nodes(participants, placement_status())
    # A distributed record from before G3a never recorded these, and cannot be
    # rebuilt warm - refuse BEFORE building transports or writing ``deploying``.
    if record_mode(deployment) != VERDICT_REPLICATED and (
        float(units.get("gpu_memory_utilization", 0) or 0) <= 0
        or int(units.get("max_model_len", 0) or 0) <= 0
    ):
        raise ValueError(PREDATES_LOAD)
    return participants


def load_options(ops: Any, deployment: Dict[str, Any], participants: List[Dict[str, Any]]):
    """``(options, config_read, note)`` a Load of this record serves with, or its refusal.

    The serving options the record carries (or its repo implies), re-checked
    before anything is written, with the image the units will run. A record
    that says its model's config.json could not be read when it was deployed
    gains the cache settings the file now decides - keeping every choice its
    owner made - only if the fit still places it. Reads the record and the
    machines' pool sizes only, so the upgrade's refresh asks it of a serving
    record before it stops anything.
    """
    units = deployment.get("units", {}) or {}
    return options_for_load_checked(
        ops, units, str(units.get("repo", "") or deployment.get("model_id", "") or ""),
        units.get("revision"), participants=participants,
        replicated=record_mode(deployment) == VERDICT_REPLICATED,
        fraction=float(units.get("gpu_memory_utilization", 0) or 0),
        context=int(units.get("max_model_len", 0) or 0),
    )


def confirm_load_bindings(
    ops: Any, deployment: Dict[str, Any], participants: List[Dict[str, Any]],
    transports: Dict[str, Any],
) -> None:
    """A split's machines still hold the addresses the record was deployed on.

    VD-162: each machine bound through the cluster link is asked whether it
    still holds its address (review S8); one that does not refuses the Load
    by name, before the record is written as loading and before any image is
    readied, rather than starting a collective that would hang. A replicated
    record binds no link and has nothing to confirm.
    """
    if record_mode(deployment) == VERDICT_REPLICATED:
        return
    units = deployment.get("units", {}) or {}
    confirm_recorded_bindings(
        participants, transports, dict(units.get("addresses") or {}),
        dict(units.get("interface_sources") or {}),
        read_tables=ops.runtime.read_link_tables,
        local_tables=ops.runtime.read_local_link_tables,
    )


def _load_replicated(
    ops, name, participants, transports, group_ids, repo, revision, port, units,
    gpu_memory_utilization, max_model_len, payload, report, started,
    credential_id, thinking=False, vllm_options=None,
):
    """Re-serve the throughput intent (VD-129) from the record.

    Reuses `ReplicatedServing.serve` exactly as the deploy does - so it MINTS a
    fresh cluster key and re-renders every gate and the balancer with it - then
    re-keys the KEPT cluster credential in place to that key
    (`gpu_pool_replicas._reput_cluster_key`), so AI Chat and the balancer stay
    in step without the credential id ever changing. The internal key is never
    revealed, exactly as a fresh deploy's is not.
    """
    replica_port = int(units.get("replica_port", 0) or 0)
    new_units, endpoint, api_key = ops._replicas.serve(
        name=name, participants=participants, transports=transports,
        group_ids=group_ids, repo=repo, revision=revision, port=port,
        replica_port=replica_port,
        gpu_memory_utilization=gpu_memory_utilization,
        max_model_len=max_model_len, payload=payload, report=report,
        started=started, thinking_default=thinking, vllm_options=vllm_options,
    )
    if credential_id:
        lease = ops.broker.resolve(credential_id, CLUSTER_INFERENCE_PURPOSE)
        _reput_cluster_key(ops.broker, credential_id, lease, api_key)
    return new_units, endpoint


def _load_distributed(
    ops, name, participants, participant_ids, transports, group_ids, repo,
    revision, port, units, gpu_memory_utilization, max_model_len, payload,
    report, started, thinking=False, vllm_options=None, credential_id="",
):
    """Re-serve the capacity intent from the record: one sharded server over Ray.

    Reuses `GpuPoolOperations._serve_distributed` with the parallel degrees
    and the per-node cluster interfaces the record wrote - the collective is
    rebuilt on the identical topology it was deployed with, with no live NIC
    re-read. The record's API bind host is NOT reused: every lead binds
    loopback now (ACC-162), so a worker-led record an older build bound to
    every interface comes back behind its keyed gate, and the kept
    credential is re-keyed in place to the gate's new key.
    """
    placement = {
        "mode": VERDICT_DISTRIBUTED,
        "parallelism": str(units.get("parallelism", "") or ""),
        "tensor_parallel_size": int(units.get("tensor_parallel_size", 1) or 1),
        "pipeline_parallel_size": int(units.get("pipeline_parallel_size", 1) or 1),
        "nodes": participants,
    }
    # The addresses the record was deployed on (VD-162), so a Load rebuilds
    # the collective on the same link; absent on a record written before
    # them, which re-serves on the enrolled address. `load_deployment`
    # has already confirmed each machine still holds its address.
    addresses = dict(units.get("addresses") or {})
    interface_sources = dict(units.get("interface_sources") or {})
    new_units, endpoint, api_key = ops._serve_distributed(
        name=name, participants=participants, transports=transports,
        group_ids=group_ids, interfaces=dict(units.get("interfaces") or {}),
        interface_sources=interface_sources,
        addresses=addresses,
        repo=repo, revision=revision, port=port,
        gpu_memory_utilization=gpu_memory_utilization,
        max_model_len=max_model_len, placement=placement, payload=payload,
        report=report, started=started, thinking_default=thinking,
        vllm_options=vllm_options,
    )
    if api_key and credential_id:
        lease = ops.broker.resolve(credential_id, CLUSTER_INFERENCE_PURPOSE)
        _reput_cluster_key(ops.broker, credential_id, lease, api_key)
    return new_units, endpoint
