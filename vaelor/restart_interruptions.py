"""What a restart would cut off, read before the owner confirms one.

VD-133 1.5.5 #5. In the 1.5.4 cold install the owner restarted the controller
to apply a new GPU memory pool size while a worker software update was
running. The update was cut off and then showed as failed, and the restart
dialog had said nothing about it.

**The rule is the one the restart itself applies.** When the workload
executor starts again, `JobStore.recover_interrupted` marks every job in
`job_vocabulary.ACTIVE_JOB_STATES` failed ("Interrupted when the workload
service restarted"). Those jobs, and only those, are what a restart of the
controller interrupts, so this module reads that same set from the same store
(LESSONS 6: one answer to one question). A queued job is not listed: it keeps
its place and runs once the controller is back.

An update being applied is listed as well. Its download is a job, but the
install that follows runs outside the job store until it writes its result;
`upgrade_progress.progress` already says whether an update is still working,
and that reading is reused here, not a second one.

**A worker's restart cuts off no job.** The executor runs one job at a time
(`executor_service.main`), and the restart of a worker is itself a job, so it
starts only after the running one ends - and is refused if it waited longer
than `gpu_memory_pool_nodes.RESTART_QUEUE_LIMIT_SECONDS`. What runs now is
still listed, as work the restart waits behind.

Agent tasks are not listed: an interrupted task is put back in the queue
(`AgentTaskStore.recover_stale`), not failed.

**Who sees what** (LESSONS 24). An administrator sees every job. Anyone else
sees their own jobs in full and only a count of the others, the scoping
`GET /jobs` applies. A store that cannot be read says so; it is never "nothing
is running" (LESSONS 8).
"""

from __future__ import annotations

import hashlib
from typing import Any, Callable, Dict, List, Optional, Tuple

from .gpu_memory_pool_nodes import RESTART_QUEUE_LIMIT_SECONDS
from .job_vocabulary import ACTIVE_JOB_STATES

#: The job that downloads and checks an update (`appliance_upgrade`).
UPGRADE_JOB = "appliance.upgrade"

#: ``effect``: what the restart does to what is listed.
EFFECT_INTERRUPTS = "interrupts"
EFFECT_WAITS = "waits"

UNREADABLE = (
    "Vaelor could not read which jobs are running, so it cannot say what this "
    "restart would interrupt. Check Activity before restarting."
)


def _item(job: Dict[str, Any]) -> Dict[str, Any]:
    return {
        "id": str(job.get("id", "")),
        "type": str(job.get("type", "")),
        "message": str(job.get("message", "") or "")[:300],
        "progress": job.get("progress") if isinstance(job.get("progress"), int) else None,
    }


#: What an update status that could not be read is listed as (review r3).
UPDATE_UNKNOWN = {
    "id": "update-status-unknown", "type": UPGRADE_JOB, "progress": None,
    "message": "Whether a Vaelor update is being applied could not be read; one may be cut off.",
}

#: What a non-administrator is told about an update being applied (review L2):
#: that one is, never whose or to which version.
GENERIC_UPDATE_ID = "update"


def _update_item(
    upgrade: Optional[Dict[str, Any]], listed: List[Dict[str, Any]], *,
    administrator: bool, boot_time: Optional[int],
) -> Tuple[Optional[Dict[str, Any]], str]:
    """The update still being applied, or a note that one was cut off by the last boot.

    Returns ``(item, note)``. No item when the update's own job is already
    listed. An update begun before this machine's current boot whose apply
    never wrote a result was stopped by that boot (review L3): it is not
    something a restart now would cut off, so it is a note, not an item.
    """
    if not isinstance(upgrade, dict) or not upgrade.get("active"):
        return None, ""
    job_id = str(upgrade.get("job_id", ""))
    if any(item["id"] == job_id for item in listed):
        return None, ""
    started = upgrade.get("started_at")
    if isinstance(boot_time, int) and isinstance(started, int) and 0 < started < boot_time:
        return None, (
            "A Vaelor update started before this machine last restarted and did "
            "not finish; that restart stopped it. The update panel says what it reached."
        )
    target = str(upgrade.get("to_version", "") or "") if administrator else ""
    return {
        "id": job_id if administrator and job_id else GENERIC_UPDATE_ID,
        "type": UPGRADE_JOB,
        "message": (
            "The update to Vaelor {} is still being applied.".format(target) if target
            else "A Vaelor update is still being applied."
        ),
        "progress": None,
    }, ""


def others_digest(ids: List[str]) -> str:
    """An opaque name for a set of other accounts' job ids (review L1).

    The dialog shows an operator only how many jobs other accounts run; it
    sends this back so the boundary can tell "the same jobs" from "as many
    different ones" without the ids ever leaving the server.
    """
    if not ids:
        return ""
    return hashlib.sha256(",".join(sorted(ids)).encode("utf-8")).hexdigest()[:24]


def read_boot_time(path: str = "/proc/stat") -> Optional[int]:
    """This machine's boot time (epoch seconds) from the kernel, or ``None``."""
    try:
        with open(path, encoding="ascii", errors="replace") as handle:
            for line in handle:
                if line.startswith("btime "):
                    value = line.split()[1:2]
                    return int(value[0]) if value and value[0].isdigit() else None
    except OSError:
        return None
    return None


def restart_impact(
    job_store: Any, *, controller: bool, actor: Optional[str],
    upgrade: Callable[[], Optional[Dict[str, Any]]] = lambda: None,
    boot_time: Callable[[], Optional[int]] = read_boot_time,
) -> Dict[str, Any]:
    """What restarting a machine would interrupt, or wait behind, for ``actor``.

    ``controller`` is True for this controller (the Power menu's Reboot and
    Shut down, and the pool card's restart of this controller) and False for
    a worker. ``actor`` is ``None`` for an administrator, who sees every job;
    otherwise only that account's jobs are described and the rest counted,
    with :func:`others_digest` naming them. ``upgrade`` reads the update's
    progress (`upgrade_progress.progress`).
    """
    effect = EFFECT_INTERRUPTS if controller else EFFECT_WAITS
    answer: Dict[str, Any] = {
        "readable": True, "reason": "", "effect": effect, "items": [],
        "others": 0, "others_digest": "",
        "wait_limit_seconds": None if controller else RESTART_QUEUE_LIMIT_SECONDS,
    }
    if job_store is None:
        return dict(answer, readable=False, reason=UNREADABLE)
    try:
        running = [job for job in job_store.records(None) if job.get("state") in ACTIVE_JOB_STATES]
    except Exception:  # noqa: BLE001 - an unread store is said, never "nothing runs"
        return dict(answer, readable=False, reason=UNREADABLE)
    mine = [job for job in running if actor is None or job.get("actor") == actor]
    items = [_item(job) for job in mine]
    other_ids = [str(job.get("id", "")) for job in running if job not in mine]
    answer["others"] = len(other_ids)
    answer["others_digest"] = others_digest(other_ids)
    if controller:
        try:
            update, note = _update_item(
                upgrade(), [_item(job) for job in running],
                administrator=actor is None, boot_time=boot_time(),
            )
        except Exception:  # noqa: BLE001 - the jobs above still stand; say the gap
            # Review r3: an unknown is an item the owner must tick and the
            # route requires, never a silently empty list.
            update, note = dict(UPDATE_UNKNOWN), ""
        answer["reason"] = note
        if update is not None:
            items.append(update)
    answer["items"] = items
    return answer


def unacknowledged(impact: Dict[str, Any], body: Any) -> bool:
    """Whether a confirmed restart leaves out work it would interrupt.

    ``body`` is the power request: ``acknowledged`` lists the ids the dialog
    showed and the owner ticked, ``acknowledged_others`` the digest of the
    other accounts' jobs it counted (review L1: a count alone let one job
    finish and a different one start unseen). A job that started after the
    dialog read, or a caller that never read at all, is refused here rather
    than cut off - the dialog's list is only as current as the moment it was
    read (LESSONS 18: the check belongs at the boundary, not only on the
    screen). One of the owner's own jobs that finished meanwhile is no reason
    to refuse; a changed set of other accounts' jobs is, because it cannot be
    told apart from new work.
    """
    if impact.get("effect") != EFFECT_INTERRUPTS:
        return False
    record = body if isinstance(body, dict) else {}
    named = record.get("acknowledged")
    seen = {str(value) for value in named} if isinstance(named, list) else set()
    if any(item["id"] not in seen for item in impact.get("items") or []):
        return True
    digest = str(impact.get("others_digest") or "")
    return bool(digest) and record.get("acknowledged_others") != digest
