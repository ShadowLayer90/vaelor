"""One pass of the GPU serving mode watch: the record, the job fact, the switch.

Housed out of `cluster_operations` for the 1,000-line ceiling `CLAUDE.md` sets,
and a coherent job on its own. :meth:`vaelor.gpu_cluster_mode.ClusterModeSwitch.reconcile`
owns every decision the pass makes but can read neither the cluster store nor
the executor's job loop, and holds no transport to a node; this is the one place
those facts are fetched and handed over, and the one place the switch's verdict
is written back onto the deployment record.

* **The record** for the deployment the mode file names, from the cluster
  store. Unreadable is not "gone": a store locked at boot keeps the current mode
  for one more pass rather than tearing a working cluster down on a transient
  error - the worst possible reading of a locked database.
* **The job fact** - whether a ``cluster.llm.deploy`` job is executing in this
  process - supplied by the executor service, which runs the job on its loop and
  this pass on a thread beside it. It is what tells a deploy in flight from one
  that died with the executor (VD-127, B3); the switch's docstring records why a
  timer was the wrong stand-in for it.
* **The stop.** A record the switch leaves over may still be running its units
  - the row a dead deploy left reads ``deploying`` and lists none of them; a
  healthy row whose credential an operator deleted is SERVING on them. The pool
  operations derive them from the record (`gpu_pool_units`) and stop each on
  its node; the switch runs that stop inside every leave it makes over a
  record, BEFORE it restores the Mode A lease, so llama.cpp is never relaunched
  beside a vLLM server still loading or still serving. What would not stop
  comes back by node and unit - to the row, and to the switch, which does not
  restore the lease at all when one of them is on the controller.
* **The write-back.** The row the switch left over is marked ``failed`` here
  with a note that is TRUE: why the cluster was torn down (one sentence per
  reason the switch can answer), and then either that everything the deploy
  started was stopped and removal clears the record, or exactly what is still
  running and where - and, when that is on the controller, that AI Chat was
  left unassigned rather than put beside it. The first note said removal would
  clear what the deploy started while `remove` read a list the row never had -
  a promise about a code path nothing had executed.
* **The replicas' health (VD-129).** For a healthy ``replicated`` row the pass
  probes each replica's ``/health`` over HTTP first - no transport, through
  `gpu_pool_replicas.ReplicaHealth` - and writes ``units.replicas[].alive``
  and ``units.replicas[].reachable`` on the row, so the console reads "k of N
  serving" from the first and says which of the rest answered unhealthy and
  at which nothing answered at all (a node that left the cluster) from the
  second. Zero of N for four consecutive passes marks the row ``failed``
  naming every replica's last state BEFORE the switch is asked, so the switch
  answers ``no-healthy-deployment`` and tears the cluster down through the
  same stop-then-leave every other leave takes; below that the record stays
  healthy. The probes run OUTSIDE the serving lock (up to five seconds each,
  and the lock is not held across a socket); the write-back re-reads the row
  under the lock and writes only if it is still the healthy row it probed,
  because `remove` drops the row under the same lock and a write over a row
  that was removed while the probes ran would resurrect it.
"""

from __future__ import annotations

import logging
from typing import Any, Callable, Dict, List, Mapping, Optional

from .gpu_cluster_mode import (
    CLUSTER_CREDENTIAL_MISSING, DEPLOY_ABANDONED, MODE_CLUSTER, NO_HEALTHY_DEPLOYMENT,
    ClusterModeSwitch,
)
from .gpu_idle_watch import observe_idle
from .gpu_pool_balancer import BALANCER_OTHER_RELEASE
from .gpu_pool_recover import (
    CONNECTION_DELETED_FIELD, FAILED_STATE, LOAD_IT_AGAIN, load_offered,
)
from .gpu_pool_serving_health import SplitHealth
from .gpu_pool_units import (
    RAY_PLANE_KIND, controller_unstopped, describe_unstopped, is_replicated, unstopped,
)

LOGGER = logging.getLogger(__name__)


def executor_mode_switch() -> ClusterModeSwitch:
    """The production switch, wired for the workload executor and only for it.

    Housed beside the pass the executor runs it through, so the switch module
    itself holds no production wiring. Everything it holds is a stateless
    adapter over a socket the executor may open - the credential broker, the
    root hardware bridge (which serialises GPU starts and stops under its own
    lock, so a second client object is not a second supervisor), and the two
    JSON records - so this builds them fresh rather than reaching into the
    executor's own lazily-cached seams. The lock is the switch module's
    :data:`~vaelor.gpu_cluster_mode.GPU_SERVING_LOCK`, which is what makes
    this switch and the executor's GPU failure-watch mutually exclusive (D5).

    Imported inside the call so that a control plane, a Pi, or a test importing
    this module never opens a bridge socket by importing it.
    """
    from .credential_broker import CredentialBrokerClient
    from .gpu_pool_replicas import default_balancer
    from .gpu_rocm_supervisor import GpuBridgeLauncher, GpuRocmSupervisor
    from .hardware_bridge import HardwareBridgeClient
    from .llm_server_proxy import LlmServerProxyController

    return ClusterModeSwitch(
        CredentialBrokerClient(),
        GpuRocmSupervisor(GpuBridgeLauncher(HardwareBridgeClient())),
        LlmServerProxyController(HardwareBridgeClient()),
        balancer=default_balancer(),
    )

#: The pass's own answer when the cluster store could not be read: the mode is
#: kept, nothing is decided, and the next pass asks again.
STORE_UNAVAILABLE = "store-unavailable"

#: Why the row was torn down: one sentence per reason the switch leaves on, and
#: the reasons this pass writes a row for - a verdict absent from this table
#: (in flight, healthy, Mode A) rewrites nothing. Keyed on the switch's own
#: reason words so a fourth leave would have to be named here to reach a row.
LEAVE_REASON_NOTES = {
    DEPLOY_ABANDONED: (
        "The workload executor stopped while this deploy was running."
    ),
    NO_HEALTHY_DEPLOYMENT: (
        "This deployment was no longer healthy while the appliance still read "
        "as clustering on it, so the cluster was torn down."
    ),
    CLUSTER_CREDENTIAL_MISSING: (
        "The cluster's connection was deleted from the credential broker "
        "while it was serving AI Chat, so the cluster was torn down."
    ),
}

#: What follows the reason when every unit the record derives was stopped (or
#: was never there). Removal then clears the RECORD - there is nothing left on
#: the nodes for it to clear. Said when the row cannot be Loaded again
#: (`gpu_pool_recover.load_offered` says no): a deploy that died before it
#: served, or a cluster whose connection was deleted.
_STOPPED = (
    " Vaelor stopped what the deploy had started and returned the appliance "
    "to AI Chat on llama.cpp."
)
_REMOVE_IT = " Remove this deployment to clear its record, then deploy again."
STOPPED_OUTCOME = _STOPPED + _REMOVE_IT

#: The same outcome for a row Load can start again - one that served, whether
#: it then stopped answering or a Load of it died with the executor: its record
#: is whole, so Load starts it again from it (W4-D7, `gpu_pool_recover`);
#: Remove clears it instead. Chosen by `load_offered`, the answer the console
#: offers Load from, never by the leave's reason (review A1).
STOPPED_LOADABLE_OUTCOME = _STOPPED + " " + LOAD_IT_AGAIN

#: What a row the watch fails says while the teardown in the same pass is
#: still stopping its units (review A8): the stop runs over each machine's
#: transport and can take tens of seconds, and the row used to read only its
#: health sentence for that long, with no way out named. `_mark_left`
#: replaces it with what actually happened once the stop is done.
TEARDOWN_UNDERWAY = (
    " Vaelor is stopping what the deploy had started and returning the "
    "appliance to AI Chat on llama.cpp."
)

#: What follows instead when a stop FAILED: the unit and the node, verbatim,
#: because "may still be running" without a name sends the operator to guess at
#: which machine's GPU is full. Removal retries the same stop.
UNSTOPPED_OUTCOME = (
    " The appliance was returned to AI Chat on llama.cpp, but {} could not be "
    "stopped and may still be running. Stop it on that node, or remove this "
    "deployment to retry, then deploy again."
)

#: What follows when the unit that would not stop is on the CONTROLLER: AI
#: Chat was left unassigned, because restoring it would have relaunched
#: llama.cpp beside a vLLM that may still hold this GPU (the one exception to
#: "always restore", `gpu_cluster_mode.CONTROLLER_UNIT_UNSTOPPED`). Chosen by
#: the same `controller_unstopped` the switch decided on, so the note cannot
#: claim a restore the switch did not make. It names the controller's units;
#: the raw list beside it carries every node's.
CONTROLLER_UNSTOPPED_OUTCOME = (
    " AI Chat was left unassigned because {} could not be stopped and may "
    "still hold this controller's GPU; stop it by hand, then remove this "
    "deployment to retry and choose an AI Chat connection."
)

#: Said beside the outcome when a split's firewall and slice could not be
#: cleared on a machine (ACC-187 review 4): nothing runs there, and Remove
#: retries the clean-up.
PLANE_LEFT_OUTCOME = (
    " The split's firewall and slice could not be cleared on {}; remove this "
    "deployment to retry the clean-up."
)

def leave_note(reason: str, not_stopped: List[Dict[str, str]], loadable: bool) -> str:
    """The row's ``units["failure"]`` for a leave: why, then what happened on the nodes.

    Composed from one sentence per reason and one per outcome, so three
    reasons times three outcomes is six sentences rather than nine, and a
    change to what "stopped" means is made once. ``loadable`` is
    `gpu_pool_recover.load_offered` of the row being written (:func:`_mark_left`):
    whether to point at Load or at Remove is the console's own answer, so the
    note and the button cannot disagree (review A1, LESSONS 6).
    """
    why = LEAVE_REASON_NOTES[reason]
    planes = [entry for entry in not_stopped if entry.get("kind") == RAY_PLANE_KIND]
    units = [entry for entry in not_stopped if entry.get("kind") != RAY_PLANE_KIND]
    # A split firewall left behind holds no GPU: it is said as a clean-up
    # note beside the outcome, never as a unit still running (review 4).
    cleanup = PLANE_LEFT_OUTCOME.format(", ".join(
        str(entry.get("node") or entry.get("node_id") or "a machine") for entry in planes
    )) if planes else ""
    held = controller_unstopped(units)
    if held:
        return why + CONTROLLER_UNSTOPPED_OUTCOME.format(describe_unstopped(held)) + cleanup
    if units:
        return why + UNSTOPPED_OUTCOME.format(describe_unstopped(units)) + cleanup
    return why + (STOPPED_LOADABLE_OUTCOME if loadable else STOPPED_OUTCOME) + cleanup


def teardown_underway_note(deployment: Mapping[str, Any]) -> str:
    """What follows a failing row's health sentence while the switch tears it down.

    The same way out the finished note will name, chosen the same way
    (`load_offered` of the failed row), so the first reading and the last
    agree on Load or Remove.
    """
    failed = {**deployment, "state": FAILED_STATE}
    after = " " + LOAD_IT_AGAIN if load_offered(failed) else _REMOVE_IT
    return " " + LEAVE_REASON_NOTES[NO_HEALTHY_DEPLOYMENT] + TEARDOWN_UNDERWAY + after


#: Where the in-progress teardown note starts inside a stored ``failure``.
_UNDERWAY_START = " " + LEAVE_REASON_NOTES[NO_HEALTHY_DEPLOYMENT] + TEARDOWN_UNDERWAY


def _without_underway(failure: str) -> str:
    """``failure`` without a trailing in-progress teardown note (round 1, A8).

    A pass that wrote the note and died before `_mark_left` (the executor
    killed mid-stop) leaves it on the stored row; the next teardown's finished
    note replaces it rather than following it, so the row never says both "is
    stopping" and "stopped", nor names its way out twice.
    """
    cut = failure.find(_UNDERWAY_START)
    return failure[:cut] if cut >= 0 else failure


def reconcile_gpu_cluster_mode(
    switch: Any,
    store: Any,
    deploy_job_active: Callable[[], bool],
    stop_units: Callable[[Mapping[str, Any]], List[Dict[str, str]]],
    replica_health: Any = None,
    ops: Any = None,
    job_store: Any = None,
) -> Dict[str, Any]:
    """The 30 s mode watch's whole pass, non-fatal by construction.

    Without a switch (the control plane's `ClusterOperations` instance, D12) it
    is a no-op; an unreadable store answers :data:`STORE_UNAVAILABLE` without
    consulting the switch; and `reconcile` itself never raises. ``stop_units``
    is `GpuPoolOperations.stop_units_for`: given the record, it stops what the
    record derives and returns what would not stop. ``replica_health`` is
    `GpuPoolOperations.replica_health` (a `gpu_pool_replicas.ReplicaHealth`),
    the reading of a healthy replicated row's replicas that is written back
    on the row before the switch decides; ``None`` reads nothing, which is
    what a caller with no replicated deployments loses nothing by. The
    verdicts in :data:`LEAVE_REASON_NOTES` are the ones this layer acts on,
    because they are the ones that leave a row behind.
    """
    if switch is None:
        return {}
    state = switch.state()
    deployment: Optional[Mapping[str, Any]] = None
    if state.deployment_name:
        try:
            deployment = store.get_pooled_deployment(state.deployment_name)
        except Exception:  # noqa: BLE001 - a store locked at boot is not proof
            return {
                "mode": state.mode, "reconciled": False,
                "reason": STORE_UNAVAILABLE,
            }
    # Review A8: a row failed here is torn down by the switch in this same
    # pass when it is the one Mode B holds, so its first reading says so.
    underway = (
        teardown_underway_note(deployment)
        if deployment is not None and state.mode == MODE_CLUSTER
        and not state.deploy_in_flight else ""
    )
    if replica_health is not None and deployment is not None:
        deployment = _observe_replicas(switch, store, deployment, replica_health, underway)
    # ACC-055: a split row is checked after deploy too - the mode's own row
    # here, before the switch decides, and every other split row below.
    split = SplitHealth.for_operations(ops) if ops is not None else None
    if split is not None and deployment is not None:
        deployment = _observe_split(switch, store, deployment, split, underway)
    not_stopped: List[Dict[str, str]] = []

    def _stop_what_it_started() -> List[Dict[str, str]]:
        # Only with a record: the units are derived from it, and a claim whose
        # row is already gone (a switch-less remove deleted it) derives
        # nothing. Whatever the stop reports - or raises - lands in the list
        # the row is written from, so the note below cannot say "stopped"
        # about something this pass did not stop; the same list is answered
        # to the switch, which reads it to decide whether the lease may be
        # restored at all.
        if deployment is not None:
            try:
                not_stopped.extend(stop_units(deployment))
            except Exception as error:  # noqa: BLE001 - the row must say so
                not_stopped.append(unstopped("", "", "", error))
        return not_stopped

    result = switch.reconcile(
        deployment, deploy_job_active=deploy_job_active,
        stop_deploy_units=_stop_what_it_started,
    )
    reason = str(result.get("reason", ""))
    if reason in LEAVE_REASON_NOTES and deployment is not None:
        _mark_left(store, deployment, reason, not_stopped)
    if "balancer_converged" in result and deployment is not None:
        _note_balancer_release(switch, store, deployment)
    # G3b scale-to-zero: after the switch pass has released the serving lock,
    # a healthy Mode-B verdict gets one idle poll; a busy or non-serving one
    # forgets its idle progress. Enqueues (never runs) an unload, so this stays
    # a short off-lock pass. A no-op without the executor collaborators.
    observe_idle(switch, ops, store, result, job_store)
    if split is not None:
        _observe_other_splits(switch, store, split, state.deployment_name)
    return result


#: What a split row the mode file does not name gets after its failure note:
#: no switch owns it, so nothing is torn down for the owner.
UNOWNED_SPLIT_OUTCOME = (
    " Remove this deployment to stop the parts that are still running."
)


def _observe_split(
    switch: Any, store: Any, deployment: Mapping[str, Any], split: Any,
    note_after: str = "",
) -> Optional[Mapping[str, Any]]:
    """Write a split row's health reading; fail it after too many bad checks.

    The split twin of :func:`_observe_replicas` (ACC-055), written the same
    way: the reads run outside the switch's lock, and the write re-reads the
    row under it and lands only on the row that was read. ``note_after``
    follows the failure on the stored row only (`_write_observation`).
    """
    observed = split.observe(deployment)
    if observed is None:
        return deployment
    units = {**(deployment.get("units") or {}), "health": observed["health"]}
    _note_degraded(units, deployment.get("units") or {}, observed.get("degraded_reason", ""))
    state = str(deployment.get("state", ""))
    if observed["failure"]:
        units["failure"] = observed["failure"]
        state = FAILED_STATE
        LOGGER.warning(
            "The GPU cluster deployment '%s' split across machines stopped "
            "serving: %s", deployment.get("name", ""), observed["failure"],
        )
    return _write_observation(switch, store, deployment, state, units, note_after)


def _note_degraded(
    units: Dict[str, Any], previous: Mapping[str, Any], reason: str,
) -> None:
    """Write (or clear) the split watch's own ``degraded_reason`` (review S1).

    The field is the fleet's degraded display. A forced removal of an
    unreachable machine owns it (``lost_nodes`` beside it), and the watch never
    overwrites or clears that note; otherwise the watch clears only the note
    it wrote itself, remembered as ``health.degraded``.
    """
    if previous.get("lost_nodes"):
        return
    earlier = previous.get("health") if isinstance(previous.get("health"), Mapping) else {}
    if reason:
        units["degraded_reason"] = reason
        units["health"] = {**units["health"], "degraded": reason}
    elif earlier.get("degraded") and units.get("degraded_reason") == earlier.get("degraded"):
        units.pop("degraded_reason", None)


def _observe_other_splits(
    switch: Any, store: Any, split: Any, mode_name: str,
) -> None:
    """Check every healthy split row the mode file does not name (ACC-055).

    A split the controller does not take part in never enters the mode file,
    so the pass above never reads it. A failed one is only marked - with a note
    that its parts may still run - because no switch owns it to tear down.
    Never raises.
    """
    try:
        rows = list(store.list_pooled_deployments())
    except Exception as error:  # noqa: BLE001 - the next pass reads again
        LOGGER.warning("The split deployments could not be listed: %s", error)
        return
    for row in rows:
        if str(row.get("name", "")) == str(mode_name or ""):
            continue
        try:
            # One write, whole from the first reading (review A8).
            _observe_split(switch, store, row, split, UNOWNED_SPLIT_OUTCOME)
        except Exception as error:  # noqa: BLE001 - one row must not stop the rest
            LOGGER.warning(
                "The split deployment '%s' could not be checked: %s",
                row.get("name", ""), error,
            )


def _observe_replicas(
    switch: Any, store: Any, deployment: Mapping[str, Any], replica_health: Any,
    note_after: str = "",
) -> Optional[Mapping[str, Any]]:
    """Write a replicated row's replica liveness, failing it at 0 of N for long.

    The record handed back is what the switch decides on: the same row with
    ``units.replicas[].alive`` and ``.reachable`` refreshed, or - after
    :data:`~vaelor.gpu_pool_replicas.REPLICA_DOWN_PASSES` passes with no
    replica answering - the row rewritten ``failed`` with every replica's
    last state as its ``failure``, so the switch's ``no-healthy-deployment``
    leave follows and `_mark_left` appends why the cluster was torn down. A
    row that could not be rewritten keeps the in-memory reading for this
    pass and is logged; nothing is decided on a store that will not answer.

    The probes run outside the switch's lock; the write does not. It re-reads
    the row under the lock first and writes only over the healthy replicated
    row it probed - a row `remove` dropped meanwhile (it drops under the same
    lock) is answered as ``None``, no such record, and a row whose state
    changed is answered as it now is. Without the re-read a probe that took
    long enough for a removal to complete put the row back, healthy, and the
    next deploy of that name was refused as a duplicate.
    """
    observed = replica_health.observe(deployment)
    if observed is None:
        return deployment
    units = {**(deployment.get("units") or {}), "replicas": observed["replicas"]}
    state = str(deployment.get("state", ""))
    if observed["failure"]:
        units["failure"] = observed["failure"]
        state = FAILED_STATE
        LOGGER.warning(
            "The GPU cluster deployment '%s' has no replica answering: %s",
            deployment.get("name", ""), observed["failure"],
        )
    return _write_observation(switch, store, deployment, state, units, note_after)


def _note_balancer_release(
    switch: Any, store: Any, deployment: Mapping[str, Any],
) -> None:
    """Write on a healthy replicated row whether its balancer is on another release.

    The balancer controller knows, from the relaunch it made, when the root
    bridge renders another release's config (`BalancerController.release_skewed`,
    2026-09-30 review S1); that fact lives in the executor's memory, and the
    console reads the row. Written only when it changes, through the same
    locked re-read as every other watch reading. A controller that cannot say
    reads as "not skewed".
    """
    if not is_replicated(deployment):
        return
    name = str(deployment.get("name", ""))
    asker = getattr(getattr(switch, "balancer", None), "release_skewed", None)
    skewed = bool(asker(name)) if callable(asker) else False
    units = dict(deployment.get("units") or {})
    if bool(units.get(BALANCER_OTHER_RELEASE)) == skewed:
        return
    if skewed:
        units[BALANCER_OTHER_RELEASE] = True
    else:
        units.pop(BALANCER_OTHER_RELEASE, None)
    _write_observation(switch, store, deployment, str(deployment.get("state", "")), units)


def _write_observation(
    switch: Any, store: Any, deployment: Mapping[str, Any], state: str,
    units: Mapping[str, Any], note_after: str = "",
) -> Optional[Mapping[str, Any]]:
    """Write one watch reading onto the row it was read from, under the lock.

    Re-reads the row under the switch's lock and writes only over a row still
    in the state that was read; answers what is there now otherwise (``None``
    for a row `remove` dropped meanwhile). ``note_after`` follows a failure on
    the STORED row only, so a reader sees the way out from the first reading;
    the row answered back carries the bare failure, which `_mark_left` puts
    in front of the finished teardown note (review A8).
    """
    name = str(deployment.get("name", ""))
    with switch.watch_lock:
        try:
            current = store.get_pooled_deployment(name)
        except Exception as error:  # noqa: BLE001 - the reading stands for this pass
            LOGGER.warning(
                "The GPU cluster deployment '%s' could not be re-read before "
                "its health reading was written: %s", name, error,
            )
            return deployment
        # Gone, or no longer the healthy row that was probed: nothing to
        # write, and the switch decides on what is there now.
        if current is None or current.get("state") != deployment.get("state"):
            return current
        row = {**current, "state": state, "units": dict(units)}
        stored = dict(units)
        # Only a row this reading fails (round 1: keyed on a ``failure``
        # field, a serving row that carried an old one gained it every pass).
        if note_after and state == FAILED_STATE and stored.get("failure"):
            stored["failure"] = str(stored["failure"]) + note_after
        try:
            store.put_pooled_deployment(
                name=name, state=state,
                model_id=str(row.get("model_id", "") or ""),
                node_ids=list(row.get("node_ids") or []), units=stored,
                endpoint=str(row.get("endpoint", "") or ""),
                credential_id=str(row.get("credential_id", "") or ""),
            )
        except Exception as error:  # noqa: BLE001 - the reading stands for this pass
            LOGGER.warning(
                "The GPU cluster deployment '%s' health reading could not be "
                "written: %s", name, error,
            )
    return row


def _mark_left(
    store: Any,
    deployment: Mapping[str, Any],
    reason: str,
    not_stopped: List[Dict[str, str]],
) -> None:
    """Turn the row the switch left over into an honest ``failed`` one.

    The note is chosen by why the switch left and by what actually happened on
    the nodes, and the raw list rides beside it under ``unstopped`` so a reader
    that wants the fields (node id, unit, error) has them without parsing a
    sentence. A note the row already carries stays in front of it: a failed
    row's ``failure`` is the deploy's own reason for failing, and the teardown
    is what happened next, not a replacement for it. Best effort, and logged
    rather than raised: Mode A has already been restored by the time this
    runs, and a row that could not be rewritten is a stale label, not a
    stranded appliance.
    """
    units = dict(deployment.get("units") or {})
    if reason == CLUSTER_CREDENTIAL_MISSING:
        # The broker said the connection is gone: Load would refuse it
        # (`gpu_pool_recover.CONNECTION_GONE`), so the row must not offer it.
        units[CONNECTION_DELETED_FIELD] = True
    loadable = load_offered({**deployment, "state": FAILED_STATE, "units": units})
    note = leave_note(reason, not_stopped, loadable)
    previous = _without_underway(str(units.get("failure", "") or ""))
    if previous:
        note = previous + " " + note
    try:
        store.put_pooled_deployment(
            name=str(deployment.get("name", "")),
            state=FAILED_STATE,
            model_id=str(deployment.get("model_id", "") or ""),
            node_ids=list(deployment.get("node_ids") or []),
            units={
                **units,
                "failure": note,
                "unstopped": list(not_stopped),
            },
            endpoint=str(deployment.get("endpoint", "") or ""),
            credential_id=str(deployment.get("credential_id", "") or ""),
        )
    except Exception as error:  # noqa: BLE001 - the mode is already restored
        LOGGER.warning(
            "The GPU cluster deployment '%s' the mode switch left over could "
            "not be marked failed: %s", deployment.get("name", ""), error,
        )
