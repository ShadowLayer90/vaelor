"""The arithmetic of serving figures: pooled rates, ratios, coverage, speed bands.

VD-147 (slice S3). The Performance dashboard draws, per machine and for the
cluster, how fast a model decodes, how much it produces, how much of each
prompt came from cache. Every one of those is derived from the per-replica
buckets the usage ledger keeps (`model_usage_replica`), and every derivation
is one of the few functions here. The serving poller builds its live card
figures from the same functions, so the card and the dashboard cannot come to
disagree about the same interval (LESSONS 6).

**Pure: no I/O, no clock, no store.** Rows in, numbers (or ``None``) out.

The rules that matter:

* **A cluster figure is pooled, never a mean of means.** Two machines that
  decoded 900 and 100 intervals are weighted 9 to 1, not 1 to 1.
* **Nothing measured is ``None``, never 0.** A bucket in which nothing was
  decoded has no decode speed; zero would read as a stalled engine.
* **A rate is taken only over a span that was really observed.** The ledger's
  baselines survive a missed scrape and a control-plane restart, which is right
  for usage and wrong for a rate: the first read after a two-minute gap carries
  two minutes of tokens, and dividing them by one tick draws a spike that never
  happened. :func:`rate_span_bound` and :func:`span_is_rateable` decide.
* **Speed bands are bucket bands.** vLLM reports per-request speed as a
  histogram with fixed edges; :func:`histogram_band` reports the band a
  percentile falls in, by nearest rank, and never interpolates inside it - an
  interpolated midpoint would contradict the exact mean beside it.
"""

from __future__ import annotations

import math
from typing import Any, Iterable, List, Mapping, Optional, Sequence, Tuple

from .serving_metrics import SERVING_FETCH_TIMEOUT_SECONDS, SERVING_POLL_SECONDS

#: How many measured tick periods a counter increase may span and still be
#: turned into a rate. Two, so one missed tick is tolerated: the increase then
#: covers two periods and is divided by the two periods it really spans.
RATE_SPAN_FACTOR = 2.0

#: The longest span ever turned into a rate, whatever the tick period, in
#: seconds. The bound is per tick because a tick is not a fixed length: the
#: poller wakes every ``SERVING_POLL_SECONDS`` and then reads each replica in
#: turn, each read allowed a few seconds (``SERVING_FETCH_TIMEOUT_SECONDS`` on
#: loopback, longer through a worker's SSH session), so a slow but healthy
#: tick is legitimately longer than the nominal period and a fixed bound would
#: call it a gap. This cap stops a pathologically slow tick licensing an
#: arbitrarily long span: six nominal periods.
MAX_RATE_SPAN_CAP_SECONDS = 6 * SERVING_POLL_SECONDS

#: The shortest tick period the bound is computed from: a period measured as
#: (nearly) zero must not make every span a gap.
_MIN_TICK_PERIOD_SECONDS = SERVING_POLL_SECONDS / 2

#: Below this share of a bucket actually read, a SUM for the bucket is missing
#: too much of a machine to be shown as a total: it is withheld (``None``)
#: rather than drawn low. A pooled rate or a maximum over the same bucket is
#: still correct over the part that was read, so those are flagged partial
#: instead.
MIN_BUCKET_COVERAGE = 0.8

#: What a bucket is, as the dashboard will say it. This module owns the words.
BUCKET_MEASURED = "measured"
BUCKET_PARTIAL = "partial"
BUCKET_RESTARTED = "restarted"
BUCKET_GAP = "gap"
BUCKET_MODEL_CHANGED = "model_changed"
BUCKET_NOT_READ = "not_read"
#: The bucket-state vocabulary as one literal set, for its wire table.
BUCKET_STATES = frozenset({"measured", "partial", "restarted", "gap", "model_changed", "not_read"})

#: The owner sentence for each bucket state that is not a plain measurement.
BUCKET_REASONS = {
    BUCKET_RESTARTED: "The engine restarted in this interval.",
    BUCKET_GAP: (
        "Vaelor could not read this machine for part of this time; what it did "
        "then is counted in usage but not charted as a speed."
    ),
    BUCKET_MODEL_CHANGED: "The model changed during this interval.",
    BUCKET_NOT_READ: "Vaelor could not read this machine in this interval.",
}

#: Why two sets of speed bands cannot be combined.
BANDS_CHANGED = "vLLM's speed bands changed during this interval."


def _number(value: Any) -> Optional[float]:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        return None
    try:
        number = float(value)
    except (OverflowError, ValueError):
        return None
    return number if math.isfinite(number) else None


def rate_span_bound(last_tick_period: Any = None) -> float:
    """The longest span, in seconds, an increase may cover and still be a rate.

    ``last_tick_period`` is the poller's own measured time between the starts
    of its two most recent ticks (monotonic clock), or ``None`` before it has
    two. The bound is :data:`RATE_SPAN_FACTOR` times that period, capped at
    :data:`MAX_RATE_SPAN_CAP_SECONDS`; with no measured period yet it is the
    nominal poll period plus one read's timeout.
    """
    period = _number(last_tick_period)
    if period is None or period <= 0:
        period = SERVING_POLL_SECONDS + SERVING_FETCH_TIMEOUT_SECONDS
    period = max(period, _MIN_TICK_PERIOD_SECONDS)
    return min(RATE_SPAN_FACTOR * period, MAX_RATE_SPAN_CAP_SECONDS)


def span_is_rateable(span_seconds: Any, last_tick_period: Any = None) -> bool:
    """Whether an increase observed over ``span_seconds`` may feed a rate or ratio.

    The span is the time since the replica's stored baseline was read. A span
    past the bound is a gap (a blocked scrape, a control plane that was down);
    a span **below zero** is a wall clock that stepped backwards - also a gap,
    because no honest duration can be put under the increase. Zero is not
    rateable either: nothing can be divided by it.
    """
    span = _number(span_seconds)
    if span is None or span <= 0:
        return False
    return span <= rate_span_bound(last_tick_period)


#: How far the wall clock may run ahead of the monotonic clock over one span
#: before the span counts as a suspend (pass-3 review SC-E): five seconds, or
#: half the span when that is more - an NTP slew is a fraction of a second,
#: a suspend is the whole time the machine slept.
SUSPEND_TOLERANCE_SECONDS = 5.0
SUSPEND_TOLERANCE_SHARE = 0.5

#: How many ticks apart two stamped reads of a replica may be and still give a
#: rate: the next tick, or the one after (one missed read tolerated, and the
#: increase is divided by the time it really spans).
MAX_TICKS_APART = 2


def stamped_span(
    prior: Mapping[str, Any], reading: Mapping[str, Any],
) -> Optional[Tuple[float, bool]]:
    """``(seconds, rateable)`` from two poller stamps, or ``None`` when they cannot be compared.

    Two readings carry a stamp (``session``, ``tick``, ``monotonic``) from the
    poller that read them (`serving_metrics_poller.TickClock.stamp`). Only
    stamps from the SAME session can be compared - a monotonic reading means
    nothing to another process - and then the span is measured on that
    monotonic clock, which no wall-clock step moves (review S-7). The span is
    rateable when the reads are one or :data:`MAX_TICKS_APART` ticks apart and
    it is no longer than :data:`MAX_RATE_SPAN_CAP_SECONDS`: a tick is judged by
    its number, not by how long its reads took, so a slow but healthy tick -
    replicas read one after another through SSH, each near its timeout - is
    never mistaken for a gap (review S-8), while a stalled poller is.
    """
    keys = ("session", "tick", "monotonic")
    if any(prior.get(key) is None or reading.get(key) is None for key in keys):
        return None
    if prior["session"] != reading["session"]:
        return None
    before, after = _number(prior["monotonic"]), _number(reading["monotonic"])
    ticks_apart = reading["tick"] - prior["tick"] if isinstance(reading["tick"], int) and isinstance(prior["tick"], int) else None
    if before is None or after is None or ticks_apart is None:
        return None
    span = after - before
    rateable = 0 < span <= MAX_RATE_SPAN_CAP_SECONDS and 1 <= ticks_apart <= MAX_TICKS_APART
    # The monotonic clock stops while the machine is suspended (pass-2 review
    # SC-E): the wall clock then moves far more than the monotonic span, and
    # the increase covers time the monotonic span does not. That is a gap.
    wall_before, wall_after = _number(prior.get("read_at")), _number(reading.get("read_at"))
    if wall_before is not None and wall_after is not None and \
            (wall_after - wall_before) - span > max(SUSPEND_TOLERANCE_SECONDS, SUSPEND_TOLERANCE_SHARE * span):
        rateable = False
    return span, rateable


def pooled_rate(rows: Iterable[Mapping[str, Any]], numerator: str, seconds: str) -> Optional[float]:
    """Σ ``numerator`` ÷ Σ ``seconds`` over the rows that carry both; else ``None``.

    Weighted by what each row actually measured: a row with no seconds
    contributes nothing, and no rows with seconds is no rate at all.
    """
    total = elapsed = 0.0
    for row in rows:
        amount, span = _number(row.get(numerator)), _number(row.get(seconds))
        if amount is None or span is None or span <= 0:
            continue
        total += amount
        elapsed += span
    return total / elapsed if elapsed > 0 else None


def pooled_ratio(rows: Iterable[Mapping[str, Any]], numerator: str, denominator: str) -> Optional[float]:
    """Σ ``numerator`` ÷ Σ ``denominator`` over the rows that carry both; else ``None``.

    The denominator is a count (prompt tokens looked up), not a duration, but
    the pooling rule is the rate's: a row is used only with both halves, so a
    ratio is never built from a numerator and a denominator of different spans.
    """
    return pooled_rate(rows, numerator, denominator)


def maximum(rows: Iterable[Mapping[str, Any]], field: str) -> Optional[float]:
    """The largest reading of ``field`` across the rows, or ``None`` when none has one."""
    values = [value for value in (_number(row.get(field)) for row in rows) if value is not None]
    return max(values) if values else None


def coverage(rows: Iterable[Mapping[str, Any]]) -> Optional[float]:
    """The share of the attempted time that was really read: Σ covered ÷ Σ attempted.

    ``None`` when nothing was attempted (the replica was not a target then),
    which is a different answer from 0.0 (it was a target and was never read).
    """
    attempted = covered = 0.0
    for row in rows:
        attempted += _number(row.get("attempted_seconds")) or 0.0
        covered += _number(row.get("covered_seconds")) or 0.0
    if attempted <= 0:
        return None
    return min(1.0, covered / attempted)


def bucket_state(row: Optional[Mapping[str, Any]]) -> Tuple[str, str]:
    """``(state, reason)`` for one replica bucket row.

    In order: a bucket that saw two models is :data:`BUCKET_MODEL_CHANGED`;
    one with no covered time is a restart, a gap or simply not read, by what
    its ticks were; one read for less than :data:`MIN_BUCKET_COVERAGE` of the
    time is partial; anything else is measured. ``None`` (no row) is not read.
    """
    if not row:
        return BUCKET_NOT_READ, BUCKET_REASONS[BUCKET_NOT_READ]
    if (_number(row.get("models_seen")) or 1) > 1:
        return BUCKET_MODEL_CHANGED, BUCKET_REASONS[BUCKET_MODEL_CHANGED]
    covered = _number(row.get("covered_seconds")) or 0.0
    if covered <= 0:
        for field, state in (("reset_ticks", BUCKET_RESTARTED), ("gap_ticks", BUCKET_GAP)):
            if (_number(row.get(field)) or 0) > 0:
                return state, BUCKET_REASONS[state]
        return BUCKET_NOT_READ, BUCKET_REASONS[BUCKET_NOT_READ]
    share = coverage([row])
    if share is not None and share < MIN_BUCKET_COVERAGE:
        return BUCKET_PARTIAL, ""
    return BUCKET_MEASURED, ""


def bin_increases(
    previous: Optional[Sequence[float]], current: Sequence[float],
) -> Optional[List[int]]:
    """Per-bin increases between two cumulative histogram readings.

    ``previous`` and ``current`` are the cumulative ``le`` counts in edge order
    with the ``+Inf`` total last. The result has the same length: the number of
    observations that landed in each bin ``(edge[i-1], edge[i]]`` during the
    interval. ``None`` when there is no earlier reading, the shapes differ (the
    edges changed), or any count went backwards (the engine restarted): in each
    case the interval's distribution is not known.
    """
    if previous is None or len(previous) != len(current):
        return None
    grown = [float(now) - float(before) for before, now in zip(previous, current)]
    if any(step < 0 for step in grown):
        return None
    bins = [grown[0]] + [grown[index] - grown[index - 1] for index in range(1, len(grown))]
    if any(count < -1e-6 for count in bins):
        return None
    return [max(0, int(round(count))) for count in bins]


def merge_bins(
    histograms: Iterable[Tuple[Any, Mapping[int, int]]],
) -> Tuple[Optional[Any], Optional[dict]]:
    """``(edge_set, bins)`` of several histograms pooled, or ``(None, None)``.

    Each item is ``(edge_set, {bin index: count})``. Histograms are pooled only
    when they share ONE edge set: bins of two different edge lists are counts
    of different things, and adding them would invent a distribution. Two sets
    (across machines, across time, or across a vLLM upgrade) give no pooled
    band at all - the caller says :data:`BANDS_CHANGED` or shows each machine's
    own.
    """
    merged: dict = {}
    seen = None
    for edge_set, bins in histograms:
        if not bins:
            continue
        if seen is None:
            seen = edge_set
        elif edge_set != seen:
            return None, None
        for index, count in bins.items():
            merged[int(index)] = merged.get(int(index), 0) + int(count)
    if seen is None:
        return None, None
    return seen, merged


def histogram_band(
    bins: Mapping[int, int], edges: Sequence[float], quantile: float,
) -> Optional[Tuple[float, Optional[float]]]:
    """The ``(lower, upper)`` edges, in seconds, of the bin holding a quantile.

    ``bins`` maps a bin index to the observations in it; bin ``i`` is
    ``(edges[i-1], edges[i]]`` and the bin after the last edge is the overflow
    (``upper`` is ``None``: slower than the largest edge). Nearest rank: the
    observation at position ``ceil(quantile * total)``, and the bin it sits in
    is the answer - never a point inside it. ``None`` when there are no
    observations.
    """
    total = sum(max(0, int(count)) for count in bins.values())
    if total <= 0 or not 0 < quantile <= 1:
        return None
    rank = max(1, math.ceil(quantile * total))
    running = 0
    for index in range(len(edges) + 1):
        running += max(0, int(bins.get(index, 0)))
        if running >= rank:
            lower = float(edges[index - 1]) if index > 0 else 0.0
            upper = float(edges[index]) if index < len(edges) else None
            return lower, upper
    return None


def band_tokens_per_second(
    band: Optional[Tuple[float, Optional[float]]],
) -> Optional[Tuple[Optional[float], Optional[float]]]:
    """A seconds-per-token band as a ``(slowest, fastest)`` tokens-per-second band.

    The reciprocal flips the ends: the bin's upper time edge is its slowest
    speed. An open overflow bin has no slowest speed (``None``); a bin starting
    at zero seconds has no fastest one.
    """
    if band is None:
        return None
    lower, upper = band
    slowest = None if upper is None or upper <= 0 else 1.0 / upper
    fastest = None if lower <= 0 else 1.0 / lower
    return slowest, fastest
