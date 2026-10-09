"""One machine's history: the window the question names, and what happened in it.

**VD-205 item 3.** ``metrics.history`` read the controller whatever machine the
question named, so "what happened to the worker last night" was answered from
the controller's own rows. Every history read here names its machine: the
controller when the question names none, the named worker otherwise, and a
role that fits several machines is asked about rather than guessed.

**Review S1.** A named window ("last night", "at 3am", "on Monday", "before
the reboot") is read with a local start and an end bound
(`assistant_time_windows`), and the answer says which window it read.

"What happened" adds that machine's events in the window: serving starts,
stops, restarts and model loads (the usage ledger's own records), telemetry
gaps (in the rows themselves, and the ingest refusal record), reinstalls (the
telemetry reconcile audit) and fired alerts.
"""

from __future__ import annotations

import logging
import re
import time
from datetime import datetime
from typing import Any, Callable, Dict, List, Mapping, NamedTuple, Optional

from .assistant_time_windows import MACHINE_OWNER, Window, window_or_reason
from .telemetry_ingest_status import BACKLOG_REFUSAL, CLOCK_SKEW_REFUSAL
from .telemetry_store import MAX_HISTORY_WINDOW_SECONDS
from .telemetry_trend import span

#: The source a "what happened" answer about this controller is recorded under.
HISTORY_SOURCE = "built-in-history"

#: How a sentence names this controller when no worker is meant.
CONTROLLER_NAME = "this controller"

LOGGER = logging.getLogger(__name__)

#: The step the usage ledger is read at for engine restarts.
RESTART_STEP_SECONDS = 300

#: Two buckets missing in a row is a gap worth naming.
GAP_BUCKETS = 2

#: Words that ask what happened, not only what a reading did. Each branch is
#: covered, and deleted in turn, by `tests/test_assistant_pattern_branches.py`
#: ("any events" was dropped there: "events" already matched every question it
#: did, so it was a branch no deletion could notice).
EVENT_WORDS_PATTERN = re.compile(
    r"\b(?:what happened|anything happen|what went on|events|go wrong|went wrong)\b")


def asks_what_happened(message: str) -> bool:
    """Whether the question asks what happened, not only what a reading did."""
    return EVENT_WORDS_PATTERN.search(str(message or "").lower()) is not None


def _moment(value: Any) -> Optional[float]:
    if isinstance(value, (int, float)) and not isinstance(value, bool):
        return float(value)
    text = str(value or "").strip()
    if not text:
        return None
    try:
        return datetime.fromisoformat(text.replace("Z", "+00:00")).timestamp()
    except ValueError:
        return None


def _read(ranged: Callable[..., Any], since: int, node: Optional[str], until: int) -> Any:
    """One ranged read; the end bound is passed only when there is one."""
    extra = {"until_seconds": until} if until > 0 else {}
    positional = (since, None, node) if node else (since,)
    try:
        # The controller: no node argument, the shape every caller passes.
        return ranged(*positional, **extra)
    except TypeError:
        if not extra:
            raise
        # A reader with no end bound: the caller drops the rows past the end.
        LOGGER.warning("the telemetry reader takes no end bound; filtering the window here")
        return ranged(*positional)


#: The window a worker's history is read over when the question names none
#: (adversarial review should-fix 3): a worker has no recent count read to
#: fall back on, so an unnamed window is the last hour, said as such.
DEFAULT_WORKER_WINDOW = "1h"
DEFAULT_WORKER_WINDOW_WORDS = "over the last hour"

#: Why a machine's history could not be read because no reader is wired.
NOT_WIRED = (
    "This controller has no time-ranged telemetry reader wired in, so the "
    "history of {name} cannot be read; another machine's history is not used "
    "in its place."
)


#: Why a machine's history read failed below the store (review round 4).
UNREAD = "The telemetry history of {name} could not be read."


def machine_history(
    message: str, history_range: Any, *, node: Optional[str] = None, now: Optional[float] = None,
    boot_time: Any = None, window_text: Optional[str] = None, boot_owner: str = MACHINE_OWNER,
    name: str = CONTROLLER_NAME,
) -> Optional[Dict[str, Any]]:
    """The machine's retained rows over the window the question names.

    ``None`` only when the question names no window and the machine is the
    controller: the caller then keeps its recent count read. Every other
    case returns a record - the rows, or ``available: False`` with the reason:
    a named window that cannot be read (``refused``, review B-3), or no ranged
    reader wired (``unwired``) - and never the controller's rows for a worker.
    """
    from .assistant_machine_tools import _parse_window
    from .telemetry_store import TelemetryStoreError

    clock = time.time() if now is None else float(now)
    window, refusal = window_or_reason(message, clock, boot_time, boot_owner)
    if refusal:
        return {"available": False, "refused": True, "reason": refusal, "samples": []}
    if window is None:
        text = window_text or (DEFAULT_WORKER_WINDOW if node else None)
        seconds = _parse_window(text) if text else None
        if seconds is None:
            return None
        window = Window(clock - seconds, clock, "" if window_text else DEFAULT_WORKER_WINDOW_WORDS)
    if history_range is None:
        if node is None:
            # This controller keeps its recent count read, whose span the
            # answer states; only a worker is refused rather than misread.
            return None
        return {"available": False, "unwired": True, "samples": [],
                "reason": NOT_WIRED.format(name=name)}
    since = max(1, int(clock - window.start))
    until = max(0, int(clock - window.end))
    try:
        result = _read(history_range, since, node, until) or {}
    except TelemetryStoreError as error:
        return {"available": False, "reason": str(error)[:500], "samples": [], "window_words": window.words}
    except OSError:
        # Review round 4: a reader that failed at the OS raised out of the
        # answer. Logged with its traceback; said as unread, never as empty.
        LOGGER.warning("the telemetry history of %s could not be read", name, exc_info=True)
        return {"available": False, "reason": UNREAD.format(name=name), "samples": [],
                "window_words": window.words}
    buckets = [row for row in result.get("buckets") or [] if isinstance(row, Mapping)
               and (_moment(row.get("time")) is None or _moment(row.get("time")) <= window.end + 1)]
    return {
        "available": True,
        "reason": "" if buckets else (
            "The telemetry store is readable but holds no samples in that window."),
        "window_seconds": int(result.get("window_seconds", since)),
        "bucket_seconds": int(result.get("bucket_seconds", 0) or 0),
        "samples": buckets, "buckets": buckets,
        "window_words": window.words, "window_start": window.start, "window_end": window.end,
        "node": node,
    }


def peak_line(samples: List[Mapping[str, Any]], field: str, label: str, unit: str, lowest: bool = False) -> str:
    """The high (or low) of one reading and when it happened, local time."""
    best = None
    for row in samples:
        value = row.get(field)
        if isinstance(value, (int, float)) and not isinstance(value, bool):
            if best is None or (value < best[0] if lowest else value > best[0]):
                best = (float(value), row.get("time"))
    if best is None:
        return ""
    moment = _moment(best[1])
    when = " at {}".format(datetime.fromtimestamp(moment).astimezone().strftime("%H:%M on %Y-%m-%d")) \
        if moment is not None else ""
    return "The {} {} was {:.1f}{}{}.".format("lowest" if lowest else "highest", label, best[0], unit, when)


def gaps(samples: List[Mapping[str, Any]], bucket_seconds: int) -> List[str]:
    """Stretches with no telemetry inside the window, from the rows themselves."""
    if bucket_seconds <= 0:
        return []
    moments = sorted(moment for moment in (_moment(row.get("time")) for row in samples) if moment is not None)
    found = []
    for earlier, later in zip(moments, moments[1:]):
        if later - earlier > (GAP_BUCKETS + 1) * bucket_seconds:
            found.append("no telemetry arrived between {} and {}".format(
                datetime.fromtimestamp(earlier).astimezone().strftime("%H:%M"),
                datetime.fromtimestamp(later).astimezone().strftime("%H:%M")))
    return found


class Events(NamedTuple):
    """What happened in a window, and which records were - and were not - read."""

    lines: List[str]
    read: List[str]
    unread: List[str]


#: How each event record is named when the answer says what it read.
_SOURCE_NAMES = {
    "model_usage": "the model serving record",
    "audit": "the telemetry repair record",
    "alerts": "the alert history",
    "ingest": "the telemetry intake record (its newest refusal and discard since the control plane started)",
}

#: What a refused post is said as, by the refusal recorded (`telemetry_ingest_status`).
#: A refusal followed by an accepted post keeps its time but not its kind.
_INGEST_REFUSALS = {
    CLOCK_SKEW_REFUSAL: "this controller refused {}'s telemetry because its clock was too far from the "
                        "controller's",
    BACKLOG_REFUSAL: "this controller refused {}'s telemetry because it was resending readings older "
                     "than the accepted window",
}
_INGEST_REFUSED = "this controller refused a telemetry post from {}"


def _ingest_events(ingest_status: Callable[[str], Mapping[str, Any]], node: str,
                   start: float, end: float, name: str) -> List[tuple]:
    """A worker's newest ingest refusal and discarded reading in the window (FOLLOWUP VD205-c).

    The record `describe_ingest` reads is in memory and keeps only the newest
    of each, so the source name says so; an older one is not claimed absent.
    """
    status = ingest_status(node)
    found = []
    refused = _moment(status.get("last_refused_at"))
    if refused is not None and start <= refused <= end:
        found.append((refused, _INGEST_REFUSALS.get(
            str(status.get("refused_reason") or ""), _INGEST_REFUSED).format(name)))
    discarded = _moment(status.get("implausible_at"))
    if discarded is not None and start <= discarded <= end:
        found.append((discarded, "an implausible reading from {} ({}) was discarded".format(
            name, ", ".join(status.get("implausible_fields") or []) or "a field")))
    return found


def machine_events(sources: Mapping[str, Any], node: str, start: float, end: float,
                   name: str = "") -> Events:
    """What happened to one machine in a window, from the records that hold it.

    ``sources``: ``model_usage`` (the usage ledger: serving starts, stops,
    engine restarts and model changes), ``audit_between`` (the audit trail
    read by time, for the telemetry reconcile's reinstalls) and
    ``automations`` (the alert store, read by time; a controller rule has
    ``node == ""``) and ``ingest_status`` (a worker's newest refused or
    discarded post, FOLLOWUP VD205-c). **A record that is not wired or fails to read is named as
    not read** (adversarial review B-2, LESSONS 8): "nothing happened" may
    only be claimed of the records actually read.
    """
    from .cluster_placement import CONTROLLER_PLACEMENT_ID

    found: List[tuple] = []
    read: List[str] = []
    unread: List[str] = []
    ledger = sources.get("model_usage")
    if ledger is None:
        unread.append("model_usage")
    else:
        try:
            for row in ledger.replica_transitions(int(start), int(end)):
                at = float(row.get("at") or 0)
                if str(row.get("node_id") or "") == node and start <= at <= end:
                    found.append((at, "{} serving {}".format(
                        "started" if row.get("kind") == "started" else "stopped",
                        row.get("identity") or "a model")))
            model = None
            for row in ledger.replica_buckets_stepped(int(start), int(end), RESTART_STEP_SECONDS):
                at = _moment(row.get("bucket"))
                if str(row.get("node_id") or "") != node or at is None:
                    continue
                if (row.get("reset_ticks") or 0) > 0:
                    found.append((at, "its model engine restarted"))
                if row.get("model") and model is not None and row.get("model") != model:
                    found.append((at, "it loaded {}".format(row.get("model"))))
                model = row.get("model") or model
            read.append("model_usage")
        except Exception:  # noqa: BLE001 - one reader's failure is logged and named, never fatal
            LOGGER.exception("could not read serving transitions for a machine history")
            unread.append("model_usage")
    audit = sources.get("audit_between")
    if audit is None:
        unread.append("audit")
    else:
        try:
            from .cluster_worker_telemetry import RECONCILE_AUDIT_ACTION, RECONCILE_CHANGES

            for row in audit(start, end, RECONCILE_AUDIT_ACTION, node):
                at = _moment(row.get("created_at"))
                if at is not None and start <= at <= end:
                    found.append((at, RECONCILE_CHANGES.get(str(row.get("result")),
                                                            "Vaelor repaired the telemetry agent")))
            read.append("audit")
        except Exception:  # noqa: BLE001
            LOGGER.exception("could not read the audit trail for a machine history")
            unread.append("audit")
    store = sources.get("automations")
    if store is None:
        unread.append("alerts")
    else:
        try:
            for row in store.trigger_runs_between(start, end):
                owner = str(row.get("node") or "") or CONTROLLER_PLACEMENT_ID
                at = _moment(row.get("created_at"))
                if owner == node and at is not None and start <= at <= end:
                    found.append((at, "the alert '{}' fired {}".format(
                        row.get("name") or row.get("source") or "rule",
                        _alert_reading(row.get("source"), row.get("value")))))
            read.append("alerts")
        except Exception:  # noqa: BLE001
            LOGGER.exception("could not read fired alerts for a machine history")
            unread.append("alerts")
    ingest = sources.get("ingest_status")
    if node != CONTROLLER_PLACEMENT_ID and ingest is None:
        unread.append("ingest")
    elif node != CONTROLLER_PLACEMENT_ID:
        # The controller writes its own rows in-process; only a worker posts.
        try:
            found.extend(_ingest_events(ingest, node, start, end, name or "this worker"))
            read.append("ingest")
        except Exception:  # noqa: BLE001
            LOGGER.exception("could not read the telemetry intake record for a machine history")
            unread.append("ingest")
    return Events(
        ["{}: {}".format(datetime.fromtimestamp(at).astimezone().strftime("%H:%M"), text)
         for at, text in sorted(found)],
        [_SOURCE_NAMES[key] for key in read], [_SOURCE_NAMES[key] for key in unread])


def _alert_reading(source: Any, value: Any) -> str:
    """What the reading was when an alert fired, in its unit.

    "fired at 91.0" read as a time (adversarial review, round 2); the value is
    the reading that crossed the rule's threshold.
    """
    from .telemetry_trend import TRENDED_READINGS

    try:
        number = float(value)
    except (TypeError, ValueError):
        return "(its reading was not recorded)"
    units = {reading.field: reading.unit for reading in TRENDED_READINGS}
    units.setdefault("storage_percent", "%")
    unit = units.get(str(source or ""), "")
    return "when the reading was {:.1f}{}".format(number, " " + unit if unit == "°C" else unit)


def events_sentence(events: Events, gap_lines: List[str]) -> str:
    """The events line, claiming nothing of a record that was not read."""
    happened = events.lines + gap_lines
    if happened:
        line = "Events in that window: {}.".format("; ".join(happened))
    elif events.read:
        line = "Nothing is recorded for that window in {} or in the telemetry itself.".format(
            _listed(events.read))
    else:
        line = "No event record could be read for that window."
    if events.unread:
        line += " Not read: {}, so events there are not known.".format(_listed(events.unread))
    return line


def _listed(items: List[str]) -> str:
    return items[0] if len(items) == 1 else "{} and {}".format(", ".join(items[:-1]), items[-1])


class HistoryRead(NamedTuple):
    """What a past-time question reads: the rows, the words that frame them, or a question back."""

    history: Optional[Dict[str, Any]]
    lead: List[str]
    ask: str = ""
    node: Optional[str] = None
    name: str = ""


#: Words that make a past question about the cluster's every machine
#: (review round 3, N-S1): each machine's window is read in turn.
#: "cluster" with or without "the", any case, and "fleet" (final review).
_EVERY_MACHINE_HISTORY = ("cluster", "fleet", "every machine's", "all machines'")
#: Console features that have no telemetry history of their own; their past is
#: never answered from this controller's rows (review round 3, N-S1).
_FEATURE_WORDS = ("ai chat", "llm server", "llm-server", "assistant")


def history_scope(message: str, appliance: Mapping[str, Any]) -> Any:
    """``"feature"``, the machines to read in turn, or ``None`` for one machine."""
    from .assistant_machine_names import EVERY_MACHINE_WORDS, Machine, resolve_machines
    from .phrase_match import mentions

    lower = str(message or "").lower()
    directory = [Machine(**item) if isinstance(item, Mapping) else item
                 for item in appliance.get("machines") or []]
    resolution = resolve_machines(message, directory) if directory else None
    if mentions(lower, _FEATURE_WORDS) and not (resolution and resolution.machines):
        return "feature"
    workers = [machine for machine in directory if machine.role == "worker"]
    if workers and (mentions(lower, EVERY_MACHINE_WORDS + _EVERY_MACHINE_HISTORY)
                    or (resolution is not None and resolution.every)):
        return directory
    return None


def history_for_question(message: str, facts: Mapping[str, Any], appliance: Mapping[str, Any],
                         window_text: Optional[str] = None, machine: Any = None) -> HistoryRead:
    """The machine and window a past-time question names, read once.

    Never answers about a worker from the controller's rows: a worker whose
    history cannot be read is said as not read. ``machine`` reads that one
    machine, for a question about every machine (`history_scope`).
    """
    from .assistant_fact_access import usable_fact
    from .assistant_machine_names import (
        Machine, Resolution, inventory_names, no_machine_named, resolve_machines, single, unknown_bare_name,
        unknown_machine,
    )

    directory = [Machine(**item) if isinstance(item, Mapping) else item
                 for item in appliance.get("machines") or []]
    resolution = (Resolution([machine]) if machine is not None
                  else resolve_machines(message, directory) if directory else None)
    if resolution is not None and resolution.ambiguous:
        return HistoryRead(None, [], ask=resolution.ambiguous)
    # Review round 2 (S4): "What happened on the Z9 last night?" named a
    # machine this cluster does not have and was answered from this
    # controller's rows. In a cluster, a name that matches nothing is asked about.
    brief = appliance.get("machine_brief") if isinstance(appliance.get("machine_brief"), Mapping) else {}
    not_machines = inventory_names(facts, str(brief.get("text") or ""))
    # Review round 4 (R3-B1): a bare "on Z9" or "Zeus's" is asked about too.
    missing = ((unknown_machine(message, directory, not_machines)
                or unknown_bare_name(message, directory, not_machines))
               if any(machine.role == "worker" for machine in directory)
               and resolution is not None and not resolution.machines else "")
    if missing:
        return HistoryRead(None, [], ask=no_machine_named(missing, directory))
    machine = single(resolution) if resolution is not None else None
    node = machine.node if machine is not None and machine.role == "worker" else None
    name = machine.name if node else ""
    telemetry = usable_fact(facts, "system.telemetry") or {}
    # Review B-4: this controller's boot time is never a worker's. A worker's
    # last start is not read, and "before the reboot" on it is said as such.
    owner = name or CONTROLLER_NAME
    history = machine_history(
        message, appliance.get("telemetry_history_range"), node=node,
        boot_time=None if node else telemetry.get("boot_time"), window_text=window_text,
        boot_owner=owner[:1].upper() + owner[1:], name=owner)
    lead = []
    if history is not None and history.get("window_words"):
        # The window is measured on this controller's clock, whichever
        # machine's rows are read (adversarial review, round 2).
        lead.append("Reading {} telemetry {} (this controller's local time).".format(
            "{}'s".format(name) if name else "this controller's", history["window_words"]))
    elif name:
        lead.append("Reading {}'s telemetry.".format(name))
    return HistoryRead(history, lead, node=node, name=name)


#: Moved from `assistant_answer_scope` at its line ceiling (VD-205 live check, review round 2).
def reads_a_recent_window(
    history: Dict[str, Any], samples: List[Any], held: bool
) -> bool:
    """Whether this read is a recent slice of a larger store (VD-097 / #224).

    The closing span sentence must not call a partial read "everything this
    appliance has retained" when older rows exist, nor claim more is retained
    when the read already holds all there is. Two read shapes reach here and the
    signal differs:

    * **count-based** (`metrics_history` without a window) returns ``requested``,
      and getting a full page - ``len(samples) >= requested`` - means older rows
      were left behind.
    * a count read **cut to the tool-result size limit** carries ``cut``:
      the store held the rows it asked for and the oldest were left out, so
      more is retained however few came back (VD-205 review B1).
    * **windowed** (`_metrics_trend`) returns ``window_seconds`` and no
      ``requested``. Older rows exist beyond the window only when the window is
      shorter than the store's retention *and* the buckets actually reach back
      to the window's **far edge**. The threshold is near-full for a reason a
      half-window one got wrong: a store aged 0.7 of the asked window covers 70%
      of it and has nothing older, yet `>= 0.5 * window` called it a slice of a
      larger store and told the owner "more is retained" - false on a fresh
      reboot. Buckets are capped at 168, so a genuinely full window measures
      ≈ window minus one bucket (~0.6% short); ``0.9`` clears that and still
      rejects the aged-0.7 store, which measures well under it. A store younger
      than the window measures far below ``0.9`` and is correctly not told more
      exists.
    """
    if not held:
        return False
    window_seconds = history.get("window_seconds")
    if isinstance(window_seconds, int) and window_seconds > 0:
        if window_seconds >= MAX_HISTORY_WINDOW_SECONDS:
            return False
        measured = span(samples)
        return (
            measured.seconds is not None
            and measured.seconds >= 0.9 * window_seconds
        )
    if history.get("cut") is True:
        return True
    requested = history.get("requested")
    return (
        isinstance(requested, int) and requested > 0
        and len(samples) >= requested
    )


#: The disclosure a windowed answer owes, because its figures are not readings.
#:
#: `_metrics_trend` downsamples a span to at most 168 buckets, each the **mean**
#: of the samples in it. Stating an endpoint ("from 40°C to 47.8°C") off a mean
#: and calling it a reading is the same class of overstatement this module was
#: written to stop, one layer in: the min/max spike guard in
#: `telemetry_trend.describe` runs over the means too, so a bucket can average
#: out a real peak. The count path holds raw rows and says "retained samples"
#: unchanged; only the windowed path appends this, and it names the resolution
#: and the peak a bucket can hide. (No direction word here - the module never
#: writes one outside a computed movement, and the guard scans for it.)
AVERAGED_BUCKETS = (
    "These figures are {}-second averaged buckets, not individual readings: "
    "they show the shape of the change across the window, and a brief spike "
    "between two buckets can be averaged away and not appear here."
)


def windowed_evidence(
    samples: List[Any], history: Dict[str, Any]
) -> Dict[str, str]:
    """The evidence entry for a windowed read, naming what the rows are.

    `trend_evidence` says "retained telemetry samples", which is true of the
    count path's raw rows and misleading of a windowed read's downsampled means.
    This states the resolution instead, so the evidence panel and the answer's
    own caveat agree about what was read.
    """
    bucket = history.get("bucket_seconds")
    return {
        "source": "metrics.history",
        "summary": "Read {} downsampled telemetry buckets, each a {}-second "
                   "average, covering the requested window.".format(
                       span(samples).count, int(bucket) if bucket else 0),
    }
