"""What the controller last did with each worker's telemetry posts (ACC-126).

The keyed ingest route refuses a post whose line timestamps sit more than
``INGEST_CLOCK_SKEW_SECONDS`` from the controller's clock, and that refusal is
right: accepting it would backfill or future-date the history. What was wrong
is that nothing remembered it. The route answered ``400`` to the worker and
logged nothing, the Fleet screen had no way to say "this worker's clock is
wrong", and the self-heal reconcile - which judges liveness from the newest
stored row - saw "no samples" and reinstalled a healthy agent every window,
for ever. A refusal nobody records is an absence the observer manufactured
(LESSONS 8).

So this module is the one record of it, per node, in this control-plane
process:

* ``last_accepted_at`` - the CONTROLLER-clock time the last post from the node
  was accepted and stored. Unlike a stored row's own timestamp (the worker's
  clock), this cannot be skewed by the worker, so it is the honest "is this
  worker posting" signal.
* ``last_refused_at`` / ``refused_reason`` - the controller-clock time of the
  newest clock-skew refusal, and ``CLOCK_SKEW_REFUSAL`` while the node's newest
  post was refused for it (``""`` once a post is accepted again).
* ``clock_offset_seconds`` - worker clock minus controller clock, measured from
  the newest line timestamp of the newest post (positive: the worker is ahead).

It is in-memory and thread-safe, the same pattern the ingest route already
uses for its per-node rate budget: the ingest route and the cluster reconcile
run in the one control-plane process, so a module-level record under a lock is
shared by both without a store. A restart forgets it, and every reader treats
a missing entry as "nothing received since the control plane started", never
as a healthy default.

The clock warning is logged here too, rate-limited per node, naming the node id
and the offset - never the ingest key, and never the body.
"""

from __future__ import annotations

import logging
import threading
import time
from typing import Any, Dict, Optional

from .telemetry_ingest import INGEST_CLOCK_SKEW_SECONDS
from .telemetry_store import is_reporting

LOGGER = logging.getLogger(__name__)

#: The ``refused_reason`` recorded while a node's newest post was refused
#: because its timestamps sat outside the accepted window.
CLOCK_SKEW_REFUSAL = "clock-skew"

#: The ``refused_reason`` recorded while a node's newest post was refused for
#: resending a buffered backlog older than the window. The clock is fine; the
#: agent is stuck retrying, and restarting it drops the backlog.
BACKLOG_REFUSAL = "stale-backlog"

#: How recent a refusal must be to still describe the worker, in seconds. A
#: refused agent retries every few seconds, so a refusal older than this means
#: the worker has stopped posting at all: it then reads as not reporting, and a
#: hung agent is repaired like any other silent one, rather than being described
#: for ever by its last refusal.
REFUSAL_CURRENT_SECONDS = 120

#: An accepted post whose clock is at least this far off the controller's is
#: still warned about: the worker's stored rows are dated by its own clock, so a
#: drift this size already makes it look silent to anything reading row times,
#: well before the refusal threshold is reached.
CLOCK_DRIFT_WARN_SECONDS = 30

#: The shortest gap between two clock warnings for the same node, in seconds.
#: A worker posts every second; one line per node per five minutes is enough
#: to find in the journal without drowning it.
CLOCK_WARNING_INTERVAL_SECONDS = 300

_LOCK = threading.Lock()
_STATUS: Dict[str, Dict[str, Any]] = {}
#: Keyed by (node, warning kind), so a backlog warning cannot silence a clock
#: warning for the same node, or the reverse.
_LAST_WARNED: Dict[tuple, float] = {}


def _blank() -> Dict[str, Any]:
    return {
        "last_accepted_at": None,
        "last_refused_at": None,
        "refused_reason": "",
        "clock_offset_seconds": None,
        # Readings the bounds table discarded from this node's posts (VD-147):
        # how many since the control plane started, which fields the last
        # time, and when.
        "implausible_dropped": 0,
        "implausible_fields": [],
        "implausible_at": None,
    }


def _warn_about_clock(
    node_id: str, offset: Optional[float], refused: bool, now: float
) -> None:
    """One rate-limited WARNING naming the node and its clock offset."""
    if offset is None:
        return
    if not refused and abs(offset) < CLOCK_DRIFT_WARN_SECONDS:
        return
    if not _warning_due(node_id, "clock", now):
        return
    if refused:
        LOGGER.warning(
            "Telemetry from node %s is being refused: its clock is %+.1f s off "
            "the controller's (worker minus controller), beyond the %s s window. "
            "Enable NTP on that worker.",
            node_id, offset, INGEST_CLOCK_SKEW_SECONDS,
        )
    else:
        LOGGER.warning(
            "Telemetry from node %s is accepted but its clock is %+.1f s off the "
            "controller's (worker minus controller); posts are refused beyond "
            "%s s. Enable NTP on that worker.",
            node_id, offset, INGEST_CLOCK_SKEW_SECONDS,
        )


def _warning_due(node_id: str, kind: str, now: float) -> bool:
    """True, and the clock started, when this node's ``kind`` warning may log."""
    with _LOCK:
        last = _LAST_WARNED.get((node_id, kind))
        if last is not None and now - last < CLOCK_WARNING_INTERVAL_SECONDS:
            return False
        _LAST_WARNED[(node_id, kind)] = now
        return True


def record_backlog_refused(node_id: str, now: Optional[float] = None) -> None:
    """Record that a post was refused for resending readings older than the window."""
    moment = time.time() if now is None else now
    with _LOCK:
        entry = _STATUS.setdefault(node_id, _blank())
        entry["last_refused_at"] = moment
        entry["refused_reason"] = BACKLOG_REFUSAL
    if _warning_due(node_id, "backlog", moment):
        LOGGER.warning(
            "Telemetry from node %s is being refused: its agent is resending "
            "readings older than the %s s window (a backlog held while this "
            "controller was unreachable). Restarting the agent clears it.",
            node_id, INGEST_CLOCK_SKEW_SECONDS,
        )


def refusal_in_force(status: Dict[str, Any], now: Optional[float] = None) -> str:
    """The refusal that still describes a node, or ``""`` - the one rule.

    A refusal counts only while it is the node's NEWEST outcome (no post was
    accepted after it) and it is recent (:data:`REFUSAL_CURRENT_SECONDS`). Read
    by the history route's ``ingest`` object and by the self-heal reconcile, so
    the screen and the repair cannot disagree about whether a worker is being
    refused or has simply gone quiet.
    """
    moment = time.time() if now is None else now
    reason = str((status or {}).get("refused_reason") or "")
    refused_at = (status or {}).get("last_refused_at")
    accepted_at = (status or {}).get("last_accepted_at")
    if not reason or refused_at is None:
        return ""
    if accepted_at is not None and float(accepted_at) > float(refused_at):
        return ""
    if moment - float(refused_at) > REFUSAL_CURRENT_SECONDS:
        return ""
    return reason


def record_accepted(
    node_id: str, clock_offset_seconds: Optional[float], now: Optional[float] = None
) -> None:
    """Record that a post from ``node_id`` was validated and stored."""
    moment = time.time() if now is None else now
    with _LOCK:
        entry = _STATUS.setdefault(node_id, _blank())
        entry["last_accepted_at"] = moment
        entry["refused_reason"] = ""
        entry["clock_offset_seconds"] = clock_offset_seconds
    _warn_about_clock(node_id, clock_offset_seconds, False, moment)


def record_implausible(node_id: str, fields: Any, now: Optional[float] = None) -> None:
    """Record that readings in a post from ``node_id`` were discarded as implausible.

    The post itself was accepted; only the out-of-range values were dropped
    (`telemetry_bounds`). Counted here so the screen can say "an implausible
    reading from this machine was discarded" instead of showing a silent gap.
    """
    discarded = [str(name) for name in (fields or ())]
    names = sorted(set(discarded))
    if not names:
        return
    moment = time.time() if now is None else now
    with _LOCK:
        entry = _STATUS.setdefault(node_id, _blank())
        # Every discard counts, a field discarded on two lines twice (pass-3 review).
        entry["implausible_dropped"] = int(entry.get("implausible_dropped") or 0) + len(discarded)
        entry["implausible_fields"] = names
        entry["implausible_at"] = moment


def record_clock_refused(
    node_id: str, clock_offset_seconds: Optional[float], now: Optional[float] = None
) -> None:
    """Record that a post from ``node_id`` was refused for its timestamps."""
    moment = time.time() if now is None else now
    with _LOCK:
        entry = _STATUS.setdefault(node_id, _blank())
        entry["last_refused_at"] = moment
        entry["refused_reason"] = CLOCK_SKEW_REFUSAL
        entry["clock_offset_seconds"] = clock_offset_seconds
    _warn_about_clock(node_id, clock_offset_seconds, True, moment)


def ingest_status(node_id: str) -> Dict[str, Any]:
    """A copy of ``node_id``'s record; every value empty when none is held.

    Keys: ``last_accepted_at``, ``last_refused_at`` (controller-clock epoch
    seconds or None), ``refused_reason`` (``CLOCK_SKEW_REFUSAL`` or ``""``) and
    ``clock_offset_seconds`` (worker minus controller, or None).
    """
    with _LOCK:
        entry = _STATUS.get(node_id)
        return dict(entry) if entry is not None else _blank()


def forget(node_id: str) -> None:
    """Drop everything held for ``node_id`` (a node removed from the fleet)."""
    with _LOCK:
        _STATUS.pop(node_id, None)
        # Warnings are keyed (node, kind) since the backlog warning got its
        # own clock (review nit): drop every kind for this node.
        for key in [key for key in _LAST_WARNED if key[0] == node_id]:
            _LAST_WARNED.pop(key, None)


def _count(amount: int, unit: str) -> str:
    return "{} {}{}".format(amount, unit, "" if amount == 1 else "s")


def clock_gap_words(seconds: float) -> str:
    """An offset's size in owner words: "40 seconds", "3 minutes", "2 hours"."""
    gap = abs(float(seconds))
    if gap < 90:
        return _count(int(round(gap)), "second")
    if gap < 90 * 60:
        return _count(int(round(gap / 60)), "minute")
    if gap < 48 * 3600:
        return _count(int(round(gap / 3600)), "hour")
    return _count(int(round(gap / 86400)), "day")


def _clock_phrase(offset: float) -> str:
    direction = "ahead of" if offset > 0 else "behind"
    return "This worker's clock is {} {} the controller's".format(
        clock_gap_words(offset), direction
    )


def _reason(clock_refused: bool, offset: Optional[float]) -> str:
    """The plain-English sentence for the Fleet card, or ``""``."""
    if clock_refused:
        if offset is None:
            return (
                "This worker's readings are being refused because their "
                "timestamps are too far from the controller's clock. Turn on "
                "network time (NTP) on the worker."
            )
        return (
            "{}, so its readings are being refused. Turn on network time (NTP) "
            "on the worker."
        ).format(_clock_phrase(offset))
    if offset is not None and abs(offset) >= CLOCK_DRIFT_WARN_SECONDS:
        return (
            "{}. Its readings are still accepted, but they will be refused once "
            "the gap passes {}. Turn on network time (NTP) on the worker."
        ).format(_clock_phrase(offset), clock_gap_words(INGEST_CLOCK_SKEW_SECONDS))
    return ""


def _epoch(value: Optional[float]) -> Optional[int]:
    return None if value is None else int(value)


def describe_ingest(node_id: str, now: Optional[float] = None) -> Dict[str, Any]:
    """The history route's ``ingest`` object for one worker (contract 1).

    ``received_at`` / ``received_age_seconds`` are when the controller last
    ACCEPTED a post from the node, on the controller's clock - null after a
    control-plane restart until the next post. ``clock_refused`` is whether the
    newest post was refused for its timestamps, and ``clock_refused_at`` when
    (null otherwise). ``reason`` is a plain sentence when the clock is refused
    or drifting toward refusal, else ``""``.
    """
    moment = time.time() if now is None else now
    status = ingest_status(node_id)
    accepted = status["last_accepted_at"]
    offset = status["clock_offset_seconds"]
    in_force = refusal_in_force(status, moment)
    refused = in_force == CLOCK_SKEW_REFUSAL
    backlog = in_force == BACKLOG_REFUSAL
    return {
        "received_at": _epoch(accepted),
        "received_age_seconds": (
            None if accepted is None else max(0, int(moment - accepted))
        ),
        "clock_offset_seconds": None if offset is None else round(float(offset), 1),
        "clock_refused": refused,
        "clock_refused_at": _epoch(status["last_refused_at"]) if refused else None,
        "backlog_refused": backlog,
        "reason": _BACKLOG_REASON if backlog else _reason(refused, offset),
    }


def reporting_reading(received: Any, sampled: Any) -> Optional[float]:
    """Which of two readings dates a worker's last report: the controller's.

    ``received`` is from the controller's clock (a receipt time or its age),
    ``sampled`` from the worker's own (a stored row's time or its age). The
    receipt wins whenever there is one, so a skewed worker clock can neither
    make a posting worker look silent nor a silent one look fresh; the row is
    read only when nothing has been received since the controller started.
    """
    for value in (received, sampled):
        if isinstance(value, (int, float)) and not isinstance(value, bool):
            return float(value)
    return None


def node_is_reporting(sample_age_seconds: Any, ingest: Optional[Dict[str, Any]]) -> bool:
    """THE verdict on "is this node reporting?" (ACC-083, ACC-122, ACC-126, ACC-128).

    ``ingest`` is :func:`describe_ingest`'s object for a worker (``None`` for
    the controller, whose rows are written in-process). A worker whose newest
    post was refused for its clock is not reporting; otherwise the age asked of
    ``telemetry_store.is_reporting`` - the one window - is the controller's
    receive age when known, else the stored sample's age.
    """
    ingest = ingest or {}
    if ingest.get("clock_refused"):
        return False
    return is_reporting(reporting_reading(ingest.get("received_age_seconds"), sample_age_seconds))


def worker_is_reporting(node_id: str, sample_age_seconds: Any, now: Optional[float] = None) -> bool:
    """:func:`node_is_reporting` for a worker, read against its live ingest record."""
    return node_is_reporting(sample_age_seconds, describe_ingest(node_id, now))


_BACKLOG_REASON = (
    "This worker's telemetry agent is stuck resending readings it held while "
    "it could not reach the controller, and they are too old to accept, so its "
    "new readings are refused too. Vaelor restarts the agent at its next "
    "automatic repair; Recheck does it now. Its clock is fine."
)
