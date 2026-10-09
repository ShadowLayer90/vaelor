"""What a schedule or an alert rule is doing right now, in the owner's words.

Split out of :mod:`vaelor.automations` so the store keeps storage and this
module keeps the one derivation every screen reads (LESSONS 6: one owner per
question). The Routines cards, the History strip and the per-agent chips all
showed a green "enabled"/"watching" pill that came straight from the ``enabled``
column, so a schedule whose every run failed, a one-time schedule that had
already fired, and an alert rule on a machine that had stopped reporting all
looked healthy (ACC-086, ACC-135, ACC-139). Each status here is derived from
what was measured - the latest run's recorded outcome, when a rule last read a
value, what the last delivery attempt said - and a state the code cannot vouch
for is never reported as healthy.

Every status is ``{"state", "label", "detail"}``: ``state`` is the stable word a
client maps to a tone, ``label`` the short pill text, ``detail`` one plain
sentence (empty when there is nothing to add).
"""

from __future__ import annotations

import math
from typing import Any, Dict, Mapping, Optional

from .telemetry_store import (  # noqa: F401 - the window, re-exported
    NOT_REPORTING_LABEL, REPORTING_WINDOW_SECONDS, is_reporting,
)

#: Recorded run states that mean the scheduled run did not happen or did not
#: finish. ``blocked`` is written when the owner lost the right to run it and
#: ``failed`` when the task could not be created or the run itself failed.
FAILED_RUN_STATES = frozenset({"failed", "blocked"})

#: The scheduler polls every few seconds, so a rule whose machine is reporting
#: re-reads its value on every pass. A value older than the shared reporting
#: window plus a pass or two means the machine went quiet.
TRIGGER_READING_GRACE_SECONDS = 15

#: Plain words for a recorded run's state (runs mirror their agent task).
RUN_STATE_WORDS = {
    "queued": "started",
    "triage": "being prepared",
    "ready": "waiting to run",
    "running": "running",
    "needs_approval": "waiting for approval",
    "completed": "finished",
    "archived": "finished",
    "failed": "failed",
    "blocked": "was blocked",
    "cancelled": "was cancelled",
}


def _status(state: str, label: str, detail: str = "") -> Dict[str, str]:
    return {"state": state, "label": label, "detail": detail}


def plain_reason(message: Any) -> str:
    """A stored reason without its leading machine code (``code: sentence``).

    Errors are stored as ``owner_not_authorized: the account ...`` or
    ``task_creation: ...`` so logs stay greppable; the owner reads the sentence.
    """
    text = " ".join(str(message or "").split())
    head, separator, tail = text.partition(": ")
    if separator and head and all(part.isalnum() for part in head.split("_")):
        return tail[:1].upper() + tail[1:]
    return text


def next_slot_after(scheduled: float, interval: float, now: float) -> float:
    """The first slot of this interval schedule strictly after ``now``.

    A schedule that was paused, or a control plane that was down, has missed
    slots. Advancing one interval per run from the missed slot fired one
    catch-up run per poll until it caught up - a burst on re-enable (ACC-139).
    Skipping every missed slot runs the schedule once and resumes its cadence.
    """
    candidate = float(scheduled) + float(interval)
    if candidate > now:
        return candidate
    missed = math.floor((now - float(scheduled)) / float(interval)) + 1
    return float(scheduled) + float(interval) * missed


def schedule_status(automation: Mapping[str, Any], last_run: Optional[Mapping[str, Any]],
                    now: float) -> Dict[str, str]:
    """The one status a schedule shows.

    ``failed_once`` (a one-time schedule whose only run failed or was blocked),
    ``finished``, ``paused``, ``failing``, ``due`` or ``scheduled``. A failed
    run outranks "finished": a one-time schedule that never did its job must
    not read as a grey, completed one.
    """
    failed = bool(last_run) and str(last_run.get("state", "")) in FAILED_RUN_STATES
    reason = (plain_reason(last_run.get("message")) if last_run else "") or "No reason was recorded."
    finished_once = automation.get("kind") == "once" and automation.get("next_run_at") is None
    if finished_once and failed:
        return _status(
            "failed_once", "Ran once and failed",
            "Its one run did not complete, and a one-time schedule does not run "
            "again: " + reason,
        )
    if finished_once:
        return _status(
            "finished", "Finished",
            "This one-time schedule has already run, so it will not run again.",
        )
    if not automation.get("enabled"):
        return _status("paused", "Paused", "Paused - it will not run until you enable it.")
    if failed:
        return _status("failing", "Last run failed", "The last run did not complete: " + reason)
    next_run = automation.get("next_run_at")
    if next_run is not None and float(next_run) <= now:
        return _status("due", "Due now", "Due now - it will start within a few seconds.")
    return _status("scheduled", "Scheduled")


def last_run_view(last_run: Optional[Mapping[str, Any]]) -> Optional[Dict[str, Any]]:
    """The latest recorded run in plain words, or ``None`` when it never ran."""
    if not last_run:
        return None
    state = str(last_run.get("state", ""))
    return {
        "state": state,
        "state_label": RUN_STATE_WORDS.get(state, "status unknown"),
        "message": plain_reason(last_run.get("message")),
        "at": last_run.get("created_at"),
        "task_id": last_run.get("task_id"),
    }


def _delivery_failed(summary: str) -> bool:
    return "failed (" in summary or summary.startswith("delivery_error")


def trigger_status(trigger: Mapping[str, Any], now: float) -> Dict[str, str]:
    """The one status an alert rule shows, derived from what it last measured.

    ``watching`` needs a reading taken within the shared reporting window; an
    enabled rule without one says it is waiting (never read) or that its machine
    stopped reporting. A recorded evaluation error, then a failed delivery, are
    shown ahead of "watching" because both mean an alert would not reach you.
    """
    if not trigger.get("enabled"):
        return _status("paused", "Paused", "Paused - it will not fire until you enable it.")
    error = plain_reason(trigger.get("last_error"))
    if error:
        return _status("failing", "Not working", error)
    delivery = str(trigger.get("last_delivery") or "")
    if delivery and _delivery_failed(delivery):
        return _status(
            "delivery_failing", "Delivery failed",
            "The last alert could not be delivered: " + plain_reason(delivery),
        )
    read_at = trigger.get("last_value_at")
    if read_at is None:
        return _status(
            "waiting", "Waiting for a reading",
            "No reading of this signal has arrived from this machine yet.",
        )
    # The engine reads on a cadence, so the reading's age is judged after the
    # grace a single evaluation pass takes, on the one reporting window.
    if not is_reporting(max(0.0, now - float(read_at) - TRIGGER_READING_GRACE_SECONDS)):
        return _status(
            "not_reporting", NOT_REPORTING_LABEL,
            "This machine has stopped reporting this signal, so the rule cannot fire.",
        )
    return _status("watching", "Watching")
