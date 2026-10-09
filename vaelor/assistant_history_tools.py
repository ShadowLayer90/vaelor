"""`metrics.history`: retained telemetry, for one machine, by count or by window.

Split out of `assistant_machine_tools` at the line ceiling (adversarial review
should-fix 8). A named machine (``node``) is resolved before anything is read
(review B-5): it is read over its own rows, over the last hour when no window
is given, and an unknown name is refused - the controller's rows are never
returned for another machine.
"""

from __future__ import annotations

from typing import Any, Dict, Optional

from .cluster_placement import CONTROLLER_PLACEMENT_NAME
from .telemetry_store import TelemetryStoreError

#: The window a named machine is read over when the caller gives none.
NAMED_MACHINE_WINDOW = "1h"


def _machine_label(callbacks: Dict[str, Any], node: Optional[str]) -> str:
    """The Fleet screen's name for ``node`` (``None`` is this controller)."""
    if not node:
        return CONTROLLER_PLACEMENT_NAME
    from .assistant_machine_names import machine_directory

    return next((machine.name for machine in machine_directory(callbacks) if machine.node == node), node)


#: How much of a store failure's explanation reaches the model.
#:
#: It was 200, and the live reason for a failed retention start is longer than
#: that. The owner saw `...so it did not start and it is unknown whet` - cut
#: mid-word, and the clause cut away was the only actionable one, "the hardware
#: bridge may not have bound its socket". A cap is still wanted, because a
#: driver's error text can run to kilobytes and every character is prompt.
#: Raising it is half the repair; the store layer also now leads its sentences
#: with the part an operator can act on.
MAX_REASON_CHARACTERS = 500


def _readable_reason(text: str) -> str:
    """Collapse whitespace and trim to the cap on a word boundary."""
    collapsed = " ".join(str(text).split())
    if len(collapsed) <= MAX_REASON_CHARACTERS:
        return collapsed
    cut = collapsed[:MAX_REASON_CHARACTERS]
    boundary = cut.rfind(" ")
    if boundary > 0:
        cut = cut[:boundary]
    return cut + " ..."


#: The duration units `metrics.history`'s `window` argument accepts, in seconds.
#: Seconds through days; nothing coarser, because the store keeps seven days and
#: `telemetry_history_range` clamps a longer window to that anyway.
_WINDOW_UNITS = {
    "s": 1, "sec": 1, "secs": 1, "second": 1, "seconds": 1,
    "m": 60, "min": 60, "mins": 60, "minute": 60, "minutes": 60,
    "h": 3600, "hr": 3600, "hrs": 3600, "hour": 3600, "hours": 3600,
    "d": 86400, "day": 86400, "days": 86400,
}


def _parse_window(value: Any) -> Optional[int]:
    """Seconds for a window like ``"24h"`` or ``"7d"``, or None if unreadable.

    A number and a unit, in that order: ``90m``, ``24h``, ``7d``. The unit is
    required - a bare ``24`` could be seconds, minutes or hours and guessing is
    how a "last 24 hours" question becomes a 24-second read. Returns None rather
    than a default so the caller can reject a malformed window with a reason
    instead of silently answering a different question than the one asked.
    """
    if not isinstance(value, str):
        return None
    text = value.strip().lower().replace(" ", "")
    if not text:
        return None
    digits = 0
    while digits < len(text) and text[digits].isdigit():
        digits += 1
    if digits == 0 or digits == len(text):
        return None
    unit = _WINDOW_UNITS.get(text[digits:])
    if unit is None:
        return None
    amount = int(text[:digits])
    if amount <= 0:
        return None
    return amount * unit


def _cap_bytes() -> int:
    from .assistant_tools import MAX_RESULT_BYTES  # imported late: it imports this module

    return MAX_RESULT_BYTES


def _size(payload: Dict[str, Any]) -> int:
    from .assistant_tools import result_bytes

    return result_bytes(payload)


#: Row keys a retained sample keeps besides its numeric readings.
_SAMPLE_KEYS = ("time", "node")


def _readings_only(sample: Any) -> Any:
    """``sample`` with its numbers, time and node, and none of its descriptive text.

    VD-205: a stored Z2 row is about 2.9 KB and more than half of it is text
    that is the same in every row - sensor sources, the thermal-norms note, the
    ECC summary - so thirty raw rows ran past the tool-result cap. The trend
    reads numbers only (`telemetry_trend._series`), and a windowed bucket
    already carries exactly these fields (InfluxDB's ``MEAN(*)`` skips strings),
    so both reads now hand the model the same shape.
    """
    if not isinstance(sample, dict):
        return sample
    return {key: value for key, value in sample.items()
            if key in _SAMPLE_KEYS or (isinstance(value, (int, float)) and not isinstance(value, bool))}


#: How far below the measured fit a coarser re-read aims, so the extra bucket
#: InfluxDB's epoch-aligned grouping can add still fits.
_REREAD_MARGIN = 0.9

#: Most coarser re-reads one windowed read makes; the first nearly always fits.
_REREADS = 4


def _metrics_trend(callbacks: Dict[str, Any], window: str, node: Any = None) -> Dict[str, Any]:
    """The ranged branch of `metrics.history`: a downsampled trend over a window.

    Kept beside `metrics_history` and sharing its honest-failure discipline: a
    malformed window is rejected with a reason, a missing callback is named a
    wiring fault rather than an appliance setting, and a store that is off or
    unreadable returns the store layer's own sentence rather than crashing.
    """
    seconds = _parse_window(window)
    if seconds is None:
        return {
            "available": False,
            "reason": (
                "The requested window {!r} is not a duration I can read. Use a "
                "number and a unit, for example '24h' for the last day or '7d' "
                "for the last week."
            ).format(window),
            "samples": [],
        }
    ranged = callbacks.get("telemetry_history_range")
    if ranged is None:
        return {
            "available": False,
            "reason": (
                "This control plane has no time-ranged telemetry callback wired "
                "into it, so a window cannot be read. That is a wiring fault "
                "rather than an appliance setting."
            ),
            "samples": [],
        }
    from .assistant_machine_names import node_for_name

    machine = node_for_name(callbacks, node)
    if machine is not None and not machine:
        return {"available": False, "samples": [], "machine": None, "reason": (
            "No machine in this cluster is named {!r}, so no history was read; "
            "another machine's history is not used in its place.".format(node))}
    label = _machine_label(callbacks, machine)

    def read(points: Optional[int]) -> Dict[str, Any]:
        # No node named: the store reads THIS controller's rows only, never the
        # enrolled workers' rows averaged in beside them (ACC-082). A named
        # worker reads its own rows (VD-205 item 3).
        if machine:
            return _trend_payload(ranged(seconds, points, machine), seconds, label)
        return _trend_payload(ranged(seconds) if points is None else ranged(seconds, points),
                              seconds, label)

    try:
        payload = read(None)
        # VD-205: up to 168 bucket means of every numeric field ran past the
        # tool-result cap on the Z2. Re-read the SAME window in fewer, coarser
        # buckets sized from the measured bucket cost, so the whole window is
        # still covered; never the newest buckets passed off as the window.
        points = len(payload["buckets"])
        for _ in range(_REREADS):
            if _size(payload) <= _cap_bytes() or points <= 1:
                break
            empty = _size({**payload, "samples": [], "buckets": []})
            per_bucket = (_size(payload) - empty) / max(1, len(payload["buckets"]))
            fit = int((_cap_bytes() - empty) / per_bucket * _REREAD_MARGIN)
            points = max(1, min(fit, points - 1))
            payload = read(points)
    except TelemetryStoreError as error:
        return {
            "available": False,
            "reason": _readable_reason(error),
            "samples": [],
        }
    return payload


def _trend_payload(result: Dict[str, Any], seconds: int, label: str) -> Dict[str, Any]:
    """What `metrics.history` returns for one ranged read of the store."""
    buckets = list(result.get("buckets") or [])
    window_seconds = int(result.get("window_seconds", seconds))
    bucket_seconds = int(result.get("bucket_seconds", 0))
    return {
        "available": True,
        # Whose rows these are, by the name the Fleet screen shows (review B-5).
        "machine": label,
        # An empty result from a working store is the case that must say so, for
        # the reason the count path says it: silence reads to a model as "no
        # record of last night", the complaint VD-095 was raised about.
        "reason": "" if buckets else (
            "The telemetry store is readable but holds no samples in that "
            "window yet. Retention records from the moment it starts, so a "
            "window reaching before that holds nothing to compare against."
        ),
        "window_seconds": window_seconds,
        "bucket_seconds": bucket_seconds,
        # Exposed under `samples` as well as `buckets` so the same trend
        # machinery that reads the count path reads a window unchanged; each
        # bucket is a downsampled sample, oldest to newest, carrying the same
        # field names a raw sample does.
        "samples": buckets,
        "buckets": buckets,
        "note": (
            "Each row is a {}-second bucket, the mean of the samples in it, "
            "ordered oldest to newest over the last {} seconds. Buckets are "
            "downsampled: they show the shape of a trend, not every reading."
        ).format(bucket_seconds, window_seconds),
    }


def metrics_history(callbacks: Dict[str, Any], arguments: Dict[str, Any]) -> Dict[str, Any]:
    """Recent telemetry samples, so a trend can be read instead of one instant.

    Two ways to ask. Without `window`, the newest `limit` rows by count (the
    original behaviour, 1..120). With `window` - a duration like `"24h"` or
    `"7d"` - a downsampled trend over that span of *time*, so "the last 24 hours"
    or "the last 7 days" is answered by the store rather than by whatever the
    last 120 one-second samples happen to cover.
    """
    window = arguments.get("window")
    node = arguments.get("node")
    if isinstance(node, str) and node.strip():
        # Review B-5 (LESSONS 6): a named machine was ignored unless a window
        # was set, and the count read below is the controller's own rows. The
        # machine is resolved first; with no window it is read over the last
        # hour, and an unknown name is refused.
        if not (isinstance(window, str) and window.strip()):
            window = NAMED_MACHINE_WINDOW
        return _metrics_trend(callbacks, window, node)
    if isinstance(window, str) and window.strip():
        return _metrics_trend(callbacks, window)
    history = callbacks.get("telemetry_history")
    if history is None:
        return {
            "available": False,
            # This branch means no history callback was wired into this control
            # plane at all - a build or wiring fault, not an appliance setting.
            # It used to say "history is not retained on this appliance", word
            # for word the sentence VD-095 was raised about, so a future wiring
            # break would have reproduced the identical symptom and cost the
            # same diagnosis a second time.
            "reason": (
                "This control plane has no telemetry history callback wired "
                "into it, so no history can be read. That is a wiring fault "
                "rather than an appliance setting."
            ),
            "samples": [],
        }
    limit = arguments.get("limit", 30)
    if isinstance(limit, bool) or not isinstance(limit, int) or not 1 <= limit <= 120:
        limit = 30
    try:
        samples = history(limit)
    except TelemetryStoreError as error:
        # One domain error, not a tuple of built-ins. The tuple that used to be
        # here - (AttributeError, OSError, TypeError, ValueError) - could not
        # catch `InfluxDBClientError`, which inherits straight from `Exception`,
        # so the single failure this path actually produced was the one it could
        # not report and this `reason` field was unreachable (VD-095 defect 4).
        return {
            "available": False,
            "reason": _readable_reason(error),
            "samples": [],
        }
    if not isinstance(samples, list):
        # Coercing this to `[]` used to make it share the "readable but holds no
        # samples yet" sentence below, which is false about a type fault: the
        # store may be full. Say what actually happened.
        return {
            "available": False,
            "reason": (
                "The telemetry history callback returned {} rather than a list "
                "of samples. That is a fault in this control plane, not a "
                "statement about what the appliance recorded."
            ).format(type(samples).__name__),
            "samples": [],
        }
    samples = [_readings_only(sample) for sample in samples[-limit:]]
    payload = {
        "available": True,
        # An empty result from a working store is the case that has to say so.
        # Silence here reads to a model as "the appliance has no record of last
        # night", which is the complaint VD-095 was raised about - and it would
        # arrive from a *successful* start, so nothing else in the payload
        # contradicts it.
        "reason": "" if samples else (
            "The telemetry store is readable but holds no samples yet. "
            "Retention records from the moment it starts, so a recent restart "
            "leaves nothing to compare against until it has been running."
        ),
        "requested": limit,
        "returned": len(samples),
        "samples": samples,
        "note": _COUNT_NOTE,
    }
    return _fit_newest(payload)


_COUNT_NOTE = (
    "Samples are ordered oldest to newest and carry numeric readings only. A "
    "single reading cannot distinguish a spike from a sustained load; these can."
)


def _fit_newest(payload: Dict[str, Any]) -> Dict[str, Any]:
    """``payload`` with its oldest samples dropped until it fits the tool-result cap.

    VD-205: on the Z2 the default read was refused whole ("larger than the
    safe output limit"), so a past-time question had no history at all. With
    readings only, thirty Z2 rows fit; a machine with far more sensors gets
    the newest rows that fit and a note that says how many, so the span it
    covers is never read as the one asked for.
    """
    samples = payload["samples"]
    if _size(payload) <= _cap_bytes():
        return payload
    budget = _cap_bytes() - _size({**payload, "samples": [], "returned": len(samples),
                                   "note": _cut_note(len(samples), payload["requested"])})
    kept = 0
    for sample in reversed(samples):
        cost = _size(sample) + 1  # and its comma
        if cost > budget:
            break
        budget -= cost
        kept += 1
    kept_samples = samples[len(samples) - kept:]
    # `cut` says older rows were read and left out, so the span sentence never
    # calls these "everything this appliance has retained" (review B1).
    payload = {**payload, "samples": kept_samples, "returned": kept, "cut": True,
               "note": _cut_note(kept, payload["requested"])}
    if not kept_samples:
        payload["reason"] = (
            "One retained sample is larger than the size limit for a single "
            "tool result, so none could be returned."
        )
    return payload


def _cut_note(kept: int, requested: int) -> str:
    return "{} Only the newest {} of the {} samples asked for fit the size limit for one tool result.".format(
        _COUNT_NOTE, kept, requested)
