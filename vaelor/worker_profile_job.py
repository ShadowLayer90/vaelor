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

**A worker Vaelor was uninstalled on is left alone** (owner 2026-10-21,
VD-133 1.5.5 #16): the reading this job takes decides, by the positive
evidence `worker_profile_state.uninstall_evidence` names, and nothing is
written there unless the job was queued by the owner's Re-add (the join).

**Nothing is reported as done unless it was read back.** The job re-reads the
worker after changing it and again after writing the marker; only a reading
that matches is stored as current. Every outcome is a sentence the backend
owns (VD-173), carried on the job row and on the card.
"""

from __future__ import annotations

import logging
import secrets
import time
from dataclasses import dataclass, field, replace
from typing import Any, Callable, Dict, List, Mapping, Optional, Set, Tuple

from . import worker_profile as wp
from . import worker_profile_amd_smi as amd_smi
from . import worker_profile_apply as runner
from .cluster_job_confirmations import PROFILE_REMOVAL_CONFIRM, confirmation_for
from .cluster_node_removal import answers_now, deployments_using_node, failure_words, machine_name
from .cluster_store import NODE_NOT_FOUND
from .cluster_worker_telemetry import AUTOMATIC_REPAIR_ACTOR
from .gpu_cluster_mode import UNLOADED_STATE
from .job_vocabulary import ACTIVE_JOB_STATES, CLUSTER_NODE_PROFILE_JOB, TERMINAL_JOB_STATES
from .ssh_transport import SshTransport, channel_drops, machine_failures
from .worker_profile_probe import read_profile
from .worker_profile_state import (
    PROFILE_TAKEN_OFF, UNINSTALL_RECORD, appliance_present, appliance_reading, hold_uninstalled, uninstalled,
)

LOGGER = logging.getLogger(__name__)

#: Why a job was queued, as its ledger row says. Nobody clicked anything for
#: the first and the last; the middle is the administrator's Recheck.
JOINED_REASON, RECHECK_REASON, SWEEP_REASON = "joined", "recheck", "15-minute check"
SYSTEM_ACTOR = "system"
#: The outcome of a ``remove`` that took the profile off: the remove action's
#: one success, as `worker_profile_apply.SUCCESSES` is the apply action's
#: (`operation_resolution` reads both, VD-133 1.5.5 #4).
REMOVED = "removed"

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
#: The outcome of an automatic update that found Vaelor uninstalled on the
#: worker (owner 2026-10-21, VD-133 1.5.5 #16): nothing is written there until
#: the owner chooses Re-add (the join, which queues :data:`JOINED_REASON`) or Remove.
UNINSTALLED = "uninstalled"
UNINSTALLED_LEFT = ("Vaelor was uninstalled on {}, so nothing was put back. Re-add it to the cluster, "
                    "or Remove it.")
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


#: The words a pending automatic update ends with when the owner's Re-add takes its place.
SUPERSEDED_BY_READD = "Replaced by the update the owner's Re-add queued for this machine."


def queue_profile_job(job_store, node_id: str, reason: str, *, expected_digest: str = "",
                      bounded: bool = False, audit: Optional["AuditTrail"] = None,
                      summary: str = "") -> Optional[Dict[str, Any]]:
    """Queue one apply for ``node_id`` unless one is already queued or running.

    ``bounded`` is the 15-minute check's limit: no new job once
    :data:`SWEEP_TRIES` have failed for ``expected_digest``. A pending update
    absorbs a new one, except that the owner's Re-add (:data:`JOINED_REASON`)
    supersedes a queued automatic one and queues behind a running one: that
    update would end "uninstalled" and the owner would press Re-add again
    (owner 2026-10-21 follow-up list). ``audit`` is the control plane's
    trail: an update it queues is audited (ACC-121), with ``summary`` as its
    words. Never raises: a store that cannot be written is logged, and the
    caller's own work stands.
    """
    try:
        pending = [job for job in job_store.unfinished_of_type(CLUSTER_NODE_PROFILE_JOB)
                   if str((job.get("payload") or {}).get("node_id")) == str(node_id)]
        for job in pending:
            if reason != JOINED_REASON or (job.get("payload") or {}).get("reason") == JOINED_REASON:
                return _store_answered(audit, node_id, job)
        for job in pending:
            if job.get("state") == "queued":
                job_store.finish(job["id"], state="superseded", message=SUPERSEDED_BY_READD)
        if bounded and expected_digest and failed_tries(job_store, node_id, expected_digest) >= SWEEP_TRIES:
            return _store_answered(audit, node_id, None)
        queued = job_store.create(CLUSTER_NODE_PROFILE_JOB, SYSTEM_ACTOR, {
            "node_id": str(node_id), "action": "apply", "reason": reason,
            "expected_digest": str(expected_digest or ""),
            "confirm": confirmation_for(CLUSTER_NODE_PROFILE_JOB),
        })
    except Exception:  # noqa: BLE001 - logged with its traceback; the caller's work stands
        LOGGER.exception("could not queue the worker-profile update of node %s", node_id)
        queued = None
    if reason != JOINED_REASON and not (queued or {}).get("deduplicated"):
        # The owner's Re-add is audited by the route that queued the join.
        audit_profile_queued(audit, node_id, queued, reason, summary)
    return queued


# -- the audit rows of automatic changes (owner 2026-10-21 follow-up list, ACC-121) --

#: The audit actions of the two changes this controller makes to a worker on
#: its own. Written by the control plane only: the executor, which runs the
#: jobs, cannot open the audit trail (LESSONS 13).
PROFILE_AUDIT_ACTION = "cluster.node.profile"
DRAIN_AUDIT_ACTION = "cluster.node.drain"
#: On a stored uninstalled verdict: its drain has its audit row.
DRAIN_AUDITED = "drain_audited"
PROFILE_AUDIT_WORDS = ("This worker's software differs from what this controller lays down (found by the "
                       "{}), so Vaelor queued an update of it.")
PROFILE_UNQUEUED_WORDS = "Vaelor could not queue an update of this worker's software (logged)."
DRAIN_AUDIT_WORDS = {
    True: ("Vaelor was uninstalled on this worker, so this controller queued its drain in the cluster: "
           "no new work is placed there until it is re-added."),
    False: "Vaelor was uninstalled on this worker, and its drain in the cluster could not be queued (logged).",
}


@dataclass
class AuditTrail:
    """Where, and as whom, a change this controller makes to a worker is audited.

    ``actor`` is :data:`AUTOMATIC_REPAIR_ACTOR` for the 15-minute check and
    the telemetry repair, and the administrator (with their address) for a
    change their Recheck queued, as ACC-121's Recheck rows are (Adopted,
    2026-10-21). ``failing`` is the control plane's own memory of which
    (action, node) is failing: a failure is one row when it starts, not one a
    pass, and the next success clears it - `_audited_telemetry_repair`'s rule.
    """

    security: Any
    actor: str = AUTOMATIC_REPAIR_ACTOR
    remote_addr: str = ""
    failing: Set[Tuple[str, str]] = field(default_factory=set)


def _store_answered(audit: Optional[AuditTrail], node_id: str, job: Optional[Dict[str, Any]]):
    """The queue answered without a new job: a failure it had, has ended."""
    if audit is not None:
        audit.failing.discard((PROFILE_AUDIT_ACTION, str(node_id)))
    return job


def _audit_row(audit: AuditTrail, action: str, node_id: str, ok: bool, details: Dict[str, Any],
               transition: bool = True) -> bool:
    """Write one row; with ``transition``, a failure only on its way into failing. Whether it is recorded.

    A drain passes ``transition=False``: it is never retried, and its stored
    ``drain_audited`` already makes it one row per uninstall, so a failing
    set kept in memory would only swallow the row of a LATER uninstall (review 2).
    """
    key = (action, str(node_id))
    if ok:
        audit.failing.discard(key)
    elif transition and key in audit.failing:
        return True
    security = audit.security
    try:
        security.audit(audit.actor, action, "queued" if ok else "failure", target=str(node_id),
                       remote_addr=audit.remote_addr, details=details)
    except Exception:  # noqa: BLE001 - the change stands; the missing row is logged (ACC-121)
        LOGGER.warning("could not record the automatic change %s of node %s", action, node_id)
        return False
    if not ok and transition:
        audit.failing.add(key)
    return True


def audit_profile_queued(audit: Optional[AuditTrail], node_id: str, queued: Optional[Mapping[str, Any]],
                         reason: str, summary: str = "") -> None:
    """One row for an update just queued (``queued``), or that could not be (``None``)."""
    if audit is None:
        return
    details = {"reason": reason, "job": str((queued or {}).get("id") or "")}
    details["summary"] = (summary or PROFILE_AUDIT_WORDS.format(reason)) if queued else PROFILE_UNQUEUED_WORDS
    _audit_row(audit, PROFILE_AUDIT_ACTION, node_id, bool(queued), details)


def audit_drain_once(audit: Optional[AuditTrail], node_id: str, verdict: Mapping[str, Any],
                     queued_now: bool = False) -> Dict[str, Any]:
    """The verdict, with its drain audited if it was not yet - whichever process queued that drain.

    Called by the control plane wherever it writes a worker's record, so a
    drain the executor queued is audited at the controller's next read
    (LESSONS 13: the executor cannot write the audit trail on the box), under
    the automatic actor; ``queued_now`` is a drain this very call queued,
    which is the caller's.
    """
    held = dict(verdict)
    if audit is None or "drain_job" not in held or held.get(DRAIN_AUDITED):
        return held
    if not queued_now:
        audit = replace(audit, actor=AUTOMATIC_REPAIR_ACTOR, remote_addr="")
    queued = bool(held["drain_job"])
    if _audit_row(audit, DRAIN_AUDIT_ACTION, node_id, queued,
                  {"job": str(held["drain_job"]), "summary": DRAIN_AUDIT_WORDS[queued]}, transition=False):
        held[DRAIN_AUDITED] = True
    return held


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


def _store_reading(store, node_id: str, reading: Dict[str, Any], amd: Optional[Mapping[str, Any]],
                   held: Optional[Mapping[str, Any]] = None, since: Optional[float] = None) -> None:
    """Store a reading as current; ``held`` is an uninstalled verdict it keeps (owner 2026-10-21).

    ``since`` is when this job started (round 2 M1): a verdict a 15-minute
    check or Recheck set while the job ran is the newer reading of the
    machine, and is kept rather than overwritten.
    """
    if not held and since is not None:
        try:
            current = uninstalled(store.node_profile(node_id))
        except Exception:  # noqa: BLE001 - logged; this job's own reading stands
            LOGGER.exception("could not read the software record of node %s before storing", node_id)
            current = None
        if current and float(current.get("at") or 0) >= float(since):
            held = current
    now = int(time.time())
    record: Dict[str, Any] = {"checked_at": now, "reading": reading, "attempt": {"at": now, "ok": True}}
    if amd:
        record["amd_smi"] = dict(amd)
    if held:
        record["uninstalled"] = dict(held)
    store.set_node_profile(node_id, record)


def _uninstalled_gate(store, transport, node_id: str, stored: Mapping[str, Any], reading: Dict[str, Any],
                      reason: str, machine: str, job_store=None) -> Optional[Dict[str, Any]]:
    """Leave an uninstalled worker alone unless the owner chose Re-add (owner 2026-10-21).

    Every automatic update - the 15-minute check, a Recheck, the telemetry
    repair - reaches the worker through this job, so this read, taken now on
    the machine, is the one that decides (LESSONS 8: positive evidence only).
    The owner's Re-add is the join, which queues :data:`JOINED_REASON`: then
    the uninstall record comes off the machine and the verdict off the store,
    and the update goes ahead. Raises the transport's error.
    """
    held = hold_uninstalled(stored, {"reading": reading}, time.time()).get("uninstalled")
    if not held:
        return None
    if reason != JOINED_REASON:
        held = drain_when_new(stored, held, job_store, node_id)
        _store_reading(store, node_id, reading, stored.get("amd_smi") if isinstance(
            stored.get("amd_smi"), dict) else None, held=held)
        return {"outcome": UNINSTALLED, "ok": True, "node_id": node_id,
                "message": UNINSTALLED_LEFT.format(machine)}
    if (reading.get("uninstalled") or {}).get("present") is True:
        transport.run(["rm", "-f", wp.UNINSTALLED_PATH], sudo=True)
    record = dict(stored)
    record.pop("uninstalled", None)
    store.set_node_profile(node_id, record)
    return None


# -- the drain of a worker Vaelor was uninstalled on (owner-adopted 2026-10-21) --

#: Why the drain was queued, on its payload; the job is the ordinary reviewed
#: availability change (`ClusterOperations.set_node_availability`), so a
#: pinned stateful app with no backup still refuses it rather than losing data.
DRAIN_REASON = "vaelor-uninstalled"
DRAIN_NOTES = {
    "queued": "This controller is draining it in the cluster, so the cluster places no new work there.",
    "done": "Drained in the cluster: the cluster places no new work there until it is re-added.",
    "failed": "It could not be drained in the cluster, so the cluster may still place work there: {}",
    "unqueued": ("It could not be drained in the cluster: the drain could not be queued (logged). Drain it "
                 "on its card."),
    "unread": "Whether it was drained in the cluster could not be read (logged).",
}


def drain_when_new(previous: Mapping[str, Any], held: Mapping[str, Any], job_store, node_id: str) -> Dict[str, Any]:
    """Queue the cluster drain the first time a worker reads as uninstalled; the verdict, noting the job.

    Controller-side only: nothing is written to the worker. Called where the
    verdict is set (the 15-minute check and Recheck in the control plane, the
    apply job in the executor); a verdict that already stood is returned as it was.
    """
    verdict = dict(held)
    if uninstalled(previous) is not None or job_store is None or "drain_job" in verdict:
        return verdict
    if verdict.get("evidence") != UNINSTALL_RECORD:
        # Round 2 M2 (decided): only the record a worker's own uninstall
        # leaves drains it. A missing marker is also what the installer's
        # --leave-worker-role shows for a moment, and nothing would undo a
        # drain queued then; the card still says the machine was wiped.
        return verdict
    try:
        job = job_store.create("cluster.node.availability", SYSTEM_ACTOR, {
            "confirm": "set-worker-drain", "availability": "drain", "node_id": str(node_id),
            "reason": DRAIN_REASON})
        verdict["drain_job"] = str(job.get("id") or "")
    except Exception:  # noqa: BLE001 - said on the card, the cause logged
        LOGGER.exception("could not queue the drain of node %s, which Vaelor was uninstalled on", node_id)
        verdict["drain_job"] = ""
    return verdict


def readd_since(job_store, node_id: str, since: float) -> bool:
    """Whether the owner's Re-add update for this worker is unfinished, or finished at or after ``since``.

    Round 2 M1: a check that probed the machine before Re-add removed the
    uninstall record, and read the store after Re-add cleared the verdict,
    would otherwise set a fresh verdict - and drain - on a machine just re-added.
    """
    for job in _node_jobs(job_store, node_id):
        if (job.get("payload") or {}).get("reason") != JOINED_REASON:
            continue
        if job.get("state") not in TERMINAL_JOB_STATES:
            return True
        if int(job.get("updated_at") or 0) / 1000 >= float(since):
            return True
    return False


def drain_note(verdict: Optional[Mapping[str, Any]], job_store) -> str:
    """The card's line about that drain, in the job's own words when it failed; never claims a drain."""
    if not verdict or "drain_job" not in verdict:
        return ""
    if not verdict["drain_job"]:
        return DRAIN_NOTES["unqueued"]
    try:
        job = job_store.get(verdict["drain_job"]) if job_store is not None else None
    except Exception:  # noqa: BLE001 - said, logged
        LOGGER.exception("could not read the drain job of an uninstalled worker")
        job = None
    if not job:
        return DRAIN_NOTES["unread"]
    state = str(job.get("state") or "")
    if state in ("completed", "healthy"):
        return DRAIN_NOTES["done"]
    if state in TERMINAL_JOB_STATES:
        # Failed, refused, cancelled or superseded: not drained, and said so.
        return DRAIN_NOTES["failed"].format(str(job.get("message") or "the drain did not finish"))
    return DRAIN_NOTES["queued"]


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
        result = apply_job(payload, operations, checkpoint, job_store=job_store)
    except ValueError:
        raise
    except Exception:  # noqa: BLE001 - a programming error is logged as one (LESSONS 1)
        LOGGER.exception("internal error updating the worker profile of node %s", payload.get("node_id"))
        result = {"outcome": "failed-internal", "message": INTERNAL_FAILURE, "ok": False}
    return _finish(job_store, job, result)


def _finish(job_store, job, result: Dict[str, Any]) -> Dict[str, Any]:
    state = "completed" if result.get("ok") else "failed"
    return job_store.finish(job["id"], state=state, message=result["message"], result=result)


def apply_job(payload: Mapping[str, Any], operations, checkpoint, job_store=None) -> Dict[str, Any]:
    """Measure, change what differs, read back, and say what happened."""
    from .cluster_worker_profile import controller_ca_source, controller_inputs
    from .worker_telemetry_runtime import default_telegraf_resolver

    started = int(time.time())
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
    try:
        left = _uninstalled_gate(store, transport, node_id, stored, reading, str(payload.get("reason") or ""),
                                 machine, job_store)
    except machine_failures() as error:
        _record_attempt(store, node_id, reason=failure_words(error), unreachable=isinstance(error, channel_drops()))
        return {"outcome": runner.CONNECTION_LOST, "ok": False, "message": FAILED.format(failure_words(error))}
    if left is not None:
        return left
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
        _store_reading(store, node_id, reading, previous_amd, since=started)
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
                      reading["facts"].get("gpu_sampler"), started)


def _read_back(store, transport, node_id, expected, present, amd, applier, applied, base,
               machine, labels, sampler, started: float) -> Dict[str, Any]:
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
        _store_reading(store, node_id, reading, amd, since=started)
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
    _store_reading(store, node_id, final, amd, since=started)
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
    for path in (wp.WMI_MODULE_CONF_PATH, wp.UNINSTALLED_PATH, wp.MARKER_PATH):
        transport.run(["rm", "-f", path], sudo=True)
        removed.append(path)
    # This controller removed the marker itself, so its absence is never read
    # as the machine having been wiped (owner 2026-10-21, VD-133 1.5.5 #16).
    record = dict(store.node_profile(node["id"]) or {})
    record[PROFILE_TAKEN_OFF] = True
    store.set_node_profile(node["id"], record)
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
    return {"outcome": REMOVED, "ok": True, "removed": removed, "message": (
        "The worker profile was taken off {}. Its folders, Docker, images and model files stay; AMD's "
        "amd-smi packages stay installed and are no longer held.".format(machine_name(node)))}
