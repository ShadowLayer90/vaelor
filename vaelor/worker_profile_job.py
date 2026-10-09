"""The ``cluster.node.profile`` job: queue it, run it, and show it (VD-194 P2).

Runs in the WORKLOAD EXECUTOR, one job at a time, like every cluster job
(`cluster_jobs.execute_cluster_job`). Queued by this controller at three
moments and nowhere else:

* at join - the join job queues it as its last step, so the profile (with the
  telemetry agent, owner answer 1) is laid down right after;
* at Recheck - when the reading Recheck takes shows something the profile
  owns that differs (`cluster_worker_profile`);
* by the 15-minute check - for a joined worker that answers and differs, at
  most :data:`SWEEP_TRIES` times for the same expected profile, so a change
  that cannot be made is not retried every quarter hour (design §3a.6). A
  Recheck always queues, whatever the count.

**The job never writes to a worker that does not answer** (design §3a.5): it
asks first, and a worker that is silent gets a failed, unreachable check
beside its last reading - the card reads "Behind, not reachable" - and
nothing else. The 15-minute check queues the job again once it answers.

**Nothing is reported as done unless it was read back.** The job re-reads the
worker after changing it and again after writing the marker; only a reading
that matches is stored as current. Every outcome is a sentence the backend
owns (VD-173), carried on the job row and on the card.
"""

from __future__ import annotations

import logging
import secrets
import time
from typing import Any, Callable, Dict, List, Mapping, Optional

from . import worker_profile as wp
from . import worker_profile_amd_smi as amd_smi
from . import worker_profile_apply as runner
from .cluster_job_confirmations import PROFILE_REMOVAL_CONFIRM, confirmation_for
from .cluster_node_removal import answers_now, deployments_using_node, failure_words, machine_name
from .cluster_store import NODE_NOT_FOUND
from .gpu_cluster_mode import UNLOADED_STATE
from .job_vocabulary import ACTIVE_JOB_STATES, CLUSTER_NODE_PROFILE_JOB, TERMINAL_JOB_STATES
from .ssh_transport import SshTransport, channel_drops, machine_failures
from .worker_profile_probe import read_profile
from .worker_profile_state import appliance_present, appliance_reading

LOGGER = logging.getLogger(__name__)

#: Why a job was queued, as its ledger row says. Nobody clicked anything for
#: the first and the last; the middle is the administrator's Recheck.
JOINED_REASON, RECHECK_REASON, SWEEP_REASON = "joined", "recheck", "15-minute check"
SYSTEM_ACTOR = "system"

#: How many failed tries the 15-minute check queues for one expected profile
#: before it waits for a Recheck or a newer release (design §3a.6).
SWEEP_TRIES = 3


#: The sentence a job reaching an internal error ends with. Fixed words; the
#: cause is in the executor's log with its traceback (LESSONS 1 / 24).
INTERNAL_FAILURE = "The update stopped on an internal error (logged); nothing more was changed."
UNREACHABLE = ("{} did not answer, so nothing was changed. The 15-minute check queues the update "
               "again once it answers.")
APPLIANCE_UNREAD = ("Whether the full appliance is on {} could not be read, so nothing was changed: "
                    "its folders must not be changed under it.")
READING_UNREAD = "{}'s software could not be read, so nothing was changed."
#: A failure the transport or the stored sign-in reported, in `failure_words`' words.
FAILED = "Update failed: {}."


# -- queueing ---------------------------------------------------------------

def _node_jobs(job_store, node_id: str, limit: int = 500) -> List[Dict[str, Any]]:
    """This worker's profile jobs, newest first, read by type so no other job crowds them out (review S6)."""
    return [job for job in job_store.list_of_type(CLUSTER_NODE_PROFILE_JOB, limit=limit)
            if str((job.get("payload") or {}).get("node_id")) == str(node_id)]


def latest_profile_job(job_store, node_id: str) -> Optional[Dict[str, Any]]:
    jobs = _node_jobs(job_store, node_id)
    return jobs[0] if jobs else None


def failed_tries(job_store, node_id: str, expected_digest: str) -> int:
    """Finished jobs for this expected profile that did not bring the worker to it.

    Counted by the digest the job was QUEUED for (its payload), so every way a
    job can end short of the profile counts - an internal error, a refused
    sign-in, a reading that could not be taken, a held or deferred change -
    not only the failures that got far enough to record a digest (review S6).
    A job that applied or found the worker current ends the count.
    """
    count = 0
    for job in _node_jobs(job_store, node_id):
        result = job.get("result") or {}
        if result.get("outcome") in runner.SUCCESSES:
            break
        if job.get("state") not in TERMINAL_JOB_STATES:
            continue
        if (job.get("payload") or {}).get("expected_digest", expected_digest) == expected_digest:
            count += 1
    return count


def queue_profile_job(job_store, node_id: str, reason: str, *, expected_digest: str = "",
                      bounded: bool = False) -> Optional[Dict[str, Any]]:
    """Queue one apply for ``node_id`` unless one is already queued or running.

    ``bounded`` is the 15-minute check's limit: no new job once
    :data:`SWEEP_TRIES` have failed for ``expected_digest``. Never raises:
    a store that cannot be written is logged, and the caller's own work stands.
    """
    try:
        for job in job_store.unfinished_of_type(CLUSTER_NODE_PROFILE_JOB):
            if str((job.get("payload") or {}).get("node_id")) == str(node_id):
                return job
        if bounded and expected_digest and failed_tries(job_store, node_id, expected_digest) >= SWEEP_TRIES:
            return None
        return job_store.create(CLUSTER_NODE_PROFILE_JOB, SYSTEM_ACTOR, {
            "node_id": str(node_id), "action": "apply", "reason": reason,
            "expected_digest": str(expected_digest or ""),
            "confirm": confirmation_for(CLUSTER_NODE_PROFILE_JOB),
        })
    except Exception:  # noqa: BLE001 - logged with its traceback; the caller's work stands
        LOGGER.exception("could not queue the worker-profile update of node %s", node_id)
        return None


# -- what the card shows about the job ---------------------------------------------

QUEUED_NOTE = "An update of this machine's worker software is queued."
RUNNING_NOTE = "This machine's worker software is being updated now."


def card_job(job: Optional[Mapping[str, Any]]) -> Optional[Dict[str, Any]]:
    """The newest job as the card's state derivation reads it: queued, running, failed or done."""
    if not job:
        return None
    state = str(job.get("state") or "")
    if state == "queued":
        phase = "queued"
    elif state in ACTIVE_JOB_STATES:
        phase = "running"
    elif state == "failed":
        phase = "failed"
    else:
        phase = "done"
    return {"state": phase, "message": str(job.get("message") or ""),
            "at": int(job.get("updated_at") or 0) // 1000}


def update_note(job: Optional[Mapping[str, Any]], age: Callable[[float], str]) -> str:
    """The card's line about the newest update, in the backend's words."""
    if not job:
        return ""
    if job["state"] == "queued":
        return QUEUED_NOTE
    if job["state"] == "running":
        return RUNNING_NOTE
    when = age(job["at"]) if job.get("at") else ""
    when = when[len("checked "):] if when.startswith("checked ") else when
    return "Last update{}: {}".format(" ({})".format(when) if when else "", job["message"])


# -- running ----------------------------------------------------------------

def _mint(store, node_id: str) -> Dict[str, Callable[..., Any]]:
    """A fresh ingest key for this agent, with its previous hash kept to put back."""
    import hashlib

    previous = next((str(digest) for stored, digest in store.ingest_key_hashes() if stored == node_id), "")

    def mint() -> str:
        key = "vnk_{}".format(secrets.token_urlsafe(32))
        store.set_ingest_key_hash(node_id, hashlib.sha256(key.encode("utf-8")).hexdigest())
        return key

    return {"mint": mint, "restore": lambda: store.set_ingest_key_hash(node_id, previous),
            "provisioned": bool(previous)}


def _record_attempt(store, node_id: str, *, reason: str, unreachable: bool) -> None:
    """A failed check beside the last reading, which is kept (LESSONS 8)."""
    try:
        record = dict(store.node_profile(node_id))
        record.pop("unreadable", None)
        record["attempt"] = {"at": int(time.time()), "ok": False, "unreachable": unreachable, "reason": reason}
        store.set_node_profile(node_id, record)
    except Exception:  # noqa: BLE001 - logged; the job's own outcome is still said
        LOGGER.exception("could not record the failed check of node %s", node_id)


def _store_reading(store, node_id: str, reading: Dict[str, Any], amd: Optional[Mapping[str, Any]]) -> None:
    now = int(time.time())
    record = {"checked_at": now, "reading": reading, "attempt": {"at": now, "ok": True}}
    if amd:
        record["amd_smi"] = dict(amd)
    store.set_node_profile(node_id, record)


def _facts(reading: Mapping[str, Any], stored: Mapping[str, Any]) -> Dict[str, Any]:
    """The probe's facts, with the GPU-sampler decision the control plane made at its last
    reading (it reads the worker's own telemetry, which this process cannot)."""
    facts = dict(reading.get("facts") or {})
    previous = (stored.get("reading") or {}).get("facts") if isinstance(stored.get("reading"), dict) else {}
    if isinstance(previous, dict) and "gpu_sampler" in previous:
        facts["gpu_sampler"] = previous["gpu_sampler"]
    return facts


def _labels() -> Dict[str, str]:
    return {component.id: component.label for component in wp.MANIFEST}


def run_profile_job(job: Dict[str, Any], operations, job_store, checkpoint) -> Dict[str, Any]:
    """Run one ``cluster.node.profile`` job and finish its row."""
    payload = dict(job.get("payload") or {})
    if payload.get("action") == "remove":
        return _finish(job_store, job, remove_job(payload, operations))
    if payload.get("confirm") != confirmation_for(CLUSTER_NODE_PROFILE_JOB):
        raise ValueError("Confirm the worker profile update.")
    try:
        result = apply_job(payload, operations, checkpoint)
    except ValueError:
        raise
    except Exception:  # noqa: BLE001 - a programming error is logged as one (LESSONS 1)
        LOGGER.exception("internal error updating the worker profile of node %s", payload.get("node_id"))
        result = {"outcome": "failed-internal", "message": INTERNAL_FAILURE, "ok": False}
    return _finish(job_store, job, result)


def _finish(job_store, job, result: Dict[str, Any]) -> Dict[str, Any]:
    state = "completed" if result.get("ok") else "failed"
    return job_store.finish(job["id"], state=state, message=result["message"], result=result)


def apply_job(payload: Mapping[str, Any], operations, checkpoint) -> Dict[str, Any]:
    """Measure, change what differs, read back, and say what happened."""
    from .cluster_worker_profile import controller_ca_source, controller_inputs
    from .worker_telemetry_runtime import default_telegraf_resolver

    store = operations.store
    node_id = str(payload.get("node_id") or "")
    node = store.get_node(node_id, include_credential=True)
    if node is None:
        raise ValueError(NODE_NOT_FOUND)
    machine = machine_name(node)
    labels = _labels()
    checkpoint(10, "Checking that {} answers".format(machine), "starting")
    try:
        profile = operations.broker.resolve(node["credential_id"], "cluster-node")
    except machine_failures() as error:
        return {"outcome": "refused", "ok": False, "message": FAILED.format(failure_words(error))}
    transport = SshTransport(profile, timeout=120)
    if not answers_now(lambda: transport):
        _record_attempt(store, node_id, reason="{} did not answer".format(machine), unreachable=True)
        return {"outcome": "deferred", "ok": True, "message": UNREACHABLE.format(machine)}
    checkpoint(20, "Reading what {} holds".format(machine), "starting")
    stored = store.node_profile(node_id)
    try:
        reading = read_profile(transport, node_id)
    except machine_failures() as error:
        _record_attempt(store, node_id, reason=failure_words(error), unreachable=isinstance(error, channel_drops()))
        return {"outcome": runner.CONNECTION_LOST, "ok": False, "message": FAILED.format(
            failure_words(error))}
    if not reading.get("ok"):
        return {"outcome": runner.INCOMPLETE, "ok": False, "message": READING_UNREAD.format(machine)}
    reading["facts"] = _facts(reading, stored)
    inventory = node.get("inventory") or {}
    architecture = str(inventory.get("architecture", ""))
    inputs = controller_inputs(store, architecture)
    expected = wp.node_expected(inventory, inputs, reading["facts"])
    appliance = appliance_reading(reading.get("appliance"))
    if appliance["reading"] == "unread":
        return {"outcome": runner.INCOMPLETE, "ok": False, "expected_digest": expected["digest"],
                "message": APPLIANCE_UNREAD.format(machine)}
    present = appliance_present(appliance)
    keys = _mint(store, node_id)
    previous_amd = stored.get("amd_smi") if isinstance(stored.get("amd_smi"), dict) else None
    planned = runner.plan(expected, reading, present, amd_smi_record=previous_amd,
                          key_provisioned=keys["provisioned"])
    base = {"expected_digest": expected["digest"], "planned": planned, "node_id": node_id}
    if planned["refused"]:
        return {**base, "outcome": runner.REFUSED, "ok": False, "message": runner.outcome_message(
            runner.REFUSED, machine=machine, labels=labels, applied=[], planned=planned)}
    held = _held(store, node_id, planned)
    # A held change keeps the marker back too: the worker does not hold the profile yet.
    apply_ids = [component for component in planned["apply"]
                 if component not in held and not (held and component == "marker")]
    if not apply_ids:
        outcome = runner.HELD if held else (runner.INCOMPLETE if planned["unread"] else runner.CURRENT)
        _store_reading(store, node_id, reading, previous_amd)
        return {**base, "outcome": outcome, "ok": outcome in runner.SUCCESSES or outcome == runner.HELD,
                "message": runner.outcome_message(outcome, machine=machine, labels=labels, applied=[],
                                                  planned={**planned, "disruptive": held})}
    applier = runner.ProfileApplier(transport, runner.Inputs(
        node_id=node_id, cluster_id=str(store.controller().get("cluster_id") or ""),
        architecture=architecture, ingest_url=inputs["ingest_url"], controller_ca_source=controller_ca_source(),
        telegraf_artifact=default_telegraf_resolver, telegraf_binary_sha256=inputs["telegraf_binary_sha256"],
        mint_key=keys["mint"], restore_key=keys["restore"], expected_digest=expected["digest"],
        previous_amd_smi=previous_amd, facts=reading["facts"]), reading)
    checkpoint(40, "Updating the worker software on {}".format(machine), "starting")
    try:
        applied = applier.apply(apply_ids)
    except runner.ApplyFailed as failure:
        if failure.outcome == runner.CONNECTION_LOST:
            # Read again when it answers (design §3); any other failure is the
            # job's own and is said on its row, not as a failed check.
            _record_attempt(store, node_id, reason="{} stopped answering".format(machine), unreachable=True)
        return {**base, "outcome": failure.outcome, "ok": False, "failed_component": failure.component,
                "restore": failure.undo, "message": runner.outcome_message(
                    failure.outcome, machine=machine, labels=labels, applied=[], failed=failure.component,
                    reason=str(failure), undo=failure.undo)}
    amd = applier.amd_smi_record or previous_amd
    checkpoint(80, "Reading {} back".format(machine), "starting")
    return _read_back(store, transport, node_id, expected, present, amd, applier, applied,
                      {**base, "applied": applied, "amd_smi": amd}, machine, labels,
                      reading["facts"].get("gpu_sampler"))


def _read_back(store, transport, node_id, expected, present, amd, applier, applied, base,
               machine, labels, sampler) -> Dict[str, Any]:
    """Read the worker again; write the marker only if everything else now matches."""
    try:
        reading = read_profile(transport, node_id)
    except machine_failures() as error:
        _record_attempt(store, node_id, reason=failure_words(error), unreachable=isinstance(error, channel_drops()))
        return {**base, "outcome": runner.CONNECTION_LOST, "ok": False, "message": runner.outcome_message(
            runner.CONNECTION_LOST, machine=machine, labels=labels, applied=applied, failed="marker")}
    reading["facts"] = {**(reading.get("facts") or {}), "gpu_sampler": sampler}
    after = runner.plan(expected, reading, present, amd_smi_record=amd, key_provisioned=True)
    still = [component for component in after["apply"] if component != "marker"] + after["unread"]
    if still:
        _store_reading(store, node_id, reading, amd)
        return {**base, "outcome": runner.INCOMPLETE, "ok": False, "message": (
            "Not updated in full: {} still differ after the change, so the profile was not marked as "
            "applied.".format(", ".join(labels.get(component, component) for component in still)))}
    try:
        applier.write_marker()
        final = read_profile(transport, node_id)
    except runner.ApplyFailed as failure:
        return {**base, "outcome": failure.outcome, "ok": False, "failed_component": "marker",
                "message": runner.outcome_message(failure.outcome, machine=machine, labels=labels, applied=applied,
                                                  failed="marker", reason=str(failure), undo=failure.undo)}
    except machine_failures() as error:
        _record_attempt(store, node_id, reason=failure_words(error), unreachable=isinstance(error, channel_drops()))
        return {**base, "outcome": runner.CONNECTION_LOST, "ok": False, "message": runner.outcome_message(
            runner.CONNECTION_LOST, machine=machine, labels=labels, applied=applied, failed="marker")}
    final["facts"] = reading["facts"]
    _store_reading(store, node_id, final, amd)
    return {**base, "outcome": runner.APPLIED, "ok": True, "message": runner.outcome_message(
        runner.APPLIED, machine=machine, labels=labels, applied=applied, planned=base.get("planned"))}


def _held(store, node_id: str, planned: Mapping[str, Any]) -> List[str]:
    """Disruptive changes held while a loaded deployment names this worker (design §3a.4)."""
    if not planned.get("disruptive"):
        return []
    # Every deployment but an unloaded one counts as loaded: a hold that errs
    # waits longer, never interrupts serving.
    loaded = [deployment for deployment in store.list_pooled_deployments()
              if str(deployment.get("state")) != UNLOADED_STATE]
    return list(planned["disruptive"]) if deployments_using_node(loaded, node_id) else []


# -- removal -----------------------------------------------------------------

def take_profile_off(store, node: Mapping[str, Any], transport) -> List[str]:
    """Take the profile's pieces off a worker that leaves (design §5 default).

    The telemetry agent, the sampler and their files, the sensor-module
    setting, the amd-smi preferences and holds (the packages stay, and AMD's
    source and key go only if the profile added them), and the marker last.
    Folders, Docker, images and model files stay. Raises the transport's error.
    """
    from .worker_telemetry_runtime import WorkerTelemetryRuntime

    WorkerTelemetryRuntime().uninstall(transport)
    store.set_ingest_key_hash(node["id"], "")
    removed = [component.path for component in wp.MANIFEST if component.id in (
        "telegraf", "emitter", "controller-ca", "telegraf-config", "telegraf-unit", "sampler", "sampler-unit")]
    record = store.node_profile(node["id"])
    removed += amd_smi.remove(transport, record.get("amd_smi") if isinstance(record, dict) else None)
    for path in (wp.WMI_MODULE_CONF_PATH, wp.MARKER_PATH):
        transport.run(["rm", "-f", path], sudo=True)
        removed.append(path)
    return removed


def remove_job(payload: Mapping[str, Any], operations) -> Dict[str, Any]:
    """The ``remove`` action: the worker stays enrolled, its profile comes off."""
    if payload.get("confirm") != PROFILE_REMOVAL_CONFIRM:
        raise ValueError("Confirm taking the worker profile off this machine.")
    store = operations.store
    node = store.get_node(str(payload.get("node_id") or ""), include_credential=True)
    if node is None:
        raise ValueError(NODE_NOT_FOUND)
    try:
        transport = SshTransport(operations.broker.resolve(node["credential_id"], "cluster-node"), timeout=120)
        removed = take_profile_off(store, node, transport)
    except machine_failures() as error:
        return {"outcome": "failed", "ok": False, "message": "The worker profile could not be taken off: {}."
                .format(failure_words(error))}
    return {"outcome": "removed", "ok": True, "removed": removed, "message": (
        "The worker profile was taken off {}. Its folders, Docker, images and model files stay; AMD's "
        "amd-smi packages stay installed and are no longer held.".format(machine_name(node)))}
