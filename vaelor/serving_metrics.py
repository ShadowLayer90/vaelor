"""Parse the llama.cpp serving engine's Prometheus ``/metrics`` (Phase E′).

The GPU AI-Chat engine (the ROCmFPX ``llama-server``) exposes a Prometheus text
endpoint on its loopback port once ``LLAMA_ARG_ENDPOINT_METRICS=1`` is set
(:data:`vaelor.gpu_rocmfpx_service.LLAMA_METRICS_ENDPOINT_ENV`, already wired).
This module turns that text into the handful of live gauges the Performance tab
can honestly show, and nothing else.

**It is pure and does no I/O.** The scrape (the HTTP GET, the poll loop, the
store write) lives in :mod:`vaelor.serving_metrics_poller`; this is only the
text→dict step, so it is fully unit-testable against a captured sample with no
socket, no engine and no database.

**What llama.cpp actually emits, and what it does not.** Captured live on the Z2,
the fork's ``/metrics`` carries the ``llamacpp:`` family below — a decode rate, a
prefill rate, the running/waiting queue depth, and per-decode counters. It does
**not** emit per-request TTFT/TPOT histograms, a KV-cache-usage ratio, prefix
cache hit rate, or goodput: those are vLLM-only, and this module's vLLM parser below now scrapes them. So this collects the gauges the
engine gives and never invents the ones it does not — an absent line is an
**absent key**, never a reassuring ``0`` (:mod:`vaelor.performance_snapshot`
keeps the vLLM-only signals honest-degraded on the strength of that).

The honesty rule runs all the way down: a value that is not a finite number
(``NaN``, ``+Inf``, ``-Inf`` — all legal in Prometheus text) is skipped rather
than stored as a sentinel, and malformed input yields an empty dict rather than
raising into the poll loop that reads a possibly-truncated body.
"""

from __future__ import annotations

import math
import re
from typing import Any, Dict, List, Optional, Tuple

#: How often the serving pollers scrape, in seconds. The serving gauges are a
#: "what is happening right now" read, not the per-second machine telemetry the
#: DataLogger keeps, so ten seconds is plenty and keeps the load on the engine
#: and the store trivial. It lives in this pure module (the pollers re-export
#: it) so the bucket arithmetic can read it without importing the pollers.
SERVING_POLL_SECONDS = 10.0

#: How long to wait on the loopback ``/metrics`` GET before giving up. Short: the
#: endpoint is on this machine's loopback, so a read that does not answer quickly
#: is a stalled or dead engine, and the poller should skip this tick rather than
#: hold the thread on a slow socket.
SERVING_FETCH_TIMEOUT_SECONDS = 3.0

#: The metric-name prefix every line this engine emits carries. Matched exactly
#: so a stray non-``llamacpp:`` exporter sharing the endpoint cannot smuggle a
#: value into a normalized key.
_PREFIX = "llamacpp:"

#: The ``llamacpp:`` gauges that matter, mapped to the normalized keys the
#: snapshot and the store use. A dict rather than a word-set tuple on purpose:
#: the normalized names are this module's contract with
#: :mod:`vaelor.performance_snapshot`, kept in one place so the two cannot drift.
#:
#: * ``predicted_tokens_seconds`` — the decode rate (output tokens/s), the number
#:   a user feels as "how fast is it typing".
#: * ``prompt_tokens_seconds`` — the prefill rate (input tokens/s).
#: * ``requests_processing`` — requests the engine is actively decoding.
#: * ``requests_deferred`` — requests queued and waiting for a slot; running
#:   high while decode is slow is the queue-bound tell.
#: * ``n_busy_slots_per_decode`` — how many slots the engine batched per decode
#:   step, a batching-efficiency read.
#:
#: The counters (``*_total``, ``n_decode_total``) are deliberately left out: they
#: are monotonic totals that mean nothing without two samples and a rate, and the
#: engine already exposes the rates above as gauges.
_GAUGES: Dict[str, str] = {
    "predicted_tokens_seconds": "decode_tokens_per_second",
    "prompt_tokens_seconds": "prompt_tokens_per_second",
    "requests_processing": "requests_processing",
    "requests_deferred": "requests_deferred",
    "n_busy_slots_per_decode": "busy_slots_per_decode",
}

#: One Prometheus sample line: ``<metric>[{labels}] <value> [timestamp]``. Only
#: the metric name and the value are captured; optional label braces are matched
#: and dropped (this engine emits none today, but a labelled line must not defeat
#: the parse), and any trailing timestamp is ignored. Anchored at the start so a
#: ``#`` comment or blank line simply does not match.
_SAMPLE = re.compile(
    r"^(?P<name>[A-Za-z_:][A-Za-z0-9_:]*)(?:\{[^}]*\})?\s+(?P<value>\S+)"
)


def parse_llamacpp_metrics(text: str) -> Dict[str, float]:
    """The normalized live gauges from a llama.cpp ``/metrics`` body.

    Returns a dict keyed by the normalized names in :data:`_GAUGES`, carrying
    only the gauges the body actually reported as finite numbers. A metric the
    body omits is **absent** from the result (never ``0``); a ``#`` comment, a
    blank line, a non-``llamacpp:`` metric, and a non-finite value are all
    skipped; and any input that is not the expected text — ``None``, HTML, a
    truncated body — yields an empty dict rather than raising, because the caller
    is a poll loop that must not die on a bad scrape.
    """
    result: Dict[str, float] = {}
    if not isinstance(text, str):
        return result
    for line in text.splitlines():
        stripped = line.strip()
        if not stripped or stripped.startswith("#"):
            continue
        match = _SAMPLE.match(stripped)
        if match is None:
            continue
        name = match.group("name")
        if not name.startswith(_PREFIX):
            continue
        key = _GAUGES.get(name[len(_PREFIX):])
        if key is None:
            continue
        try:
            value = float(match.group("value"))
        except (TypeError, ValueError):
            # A malformed value on one line must not lose the rest of the body.
            continue
        if not math.isfinite(value):
            # NaN / +Inf / -Inf are legal Prometheus values and real readings of
            # "no measurement"; a sentinel stored as a number would read back as
            # a real rate, so the line is dropped instead.
            continue
        result[key] = value
    return result


def counter_increase(previous: float, current: float) -> "tuple[float, bool]":
    """``(increase, reset)`` of a cumulative counter between two readings.

    The one reading of a Prometheus counter's movement, shared by the serving
    card's rates (`serving_metrics_poller`) and the usage ledger
    (`model_usage`). A counter lower than before means the engine restarted
    and counts from zero again, so the whole new reading is what it counted
    since (Prometheus's own ``increase``); ``reset`` says so, and each caller
    states what it does with a restart.
    """
    if current < previous:
        return current, True
    return current - previous, False


#: The two ``llamacpp:`` counters that say how much the model did, for the usage
#: ledger (:mod:`vaelor.model_usage`). The card above reads the rate GAUGES and
#: leaves these out; usage needs the cumulative counts, which two readings turn
#: into what was done in between. llama.cpp keeps no request counter.
#: ``tokens_predicted_seconds_total`` is the time the slots spent writing those
#: tokens; the serving poller turns the two into the per-answer writing speed a
#: window is judged on (`generation_health`). The usage ledger reads only the
#: token counts and ignores it.
_LLAMACPP_COUNTERS = frozenset({
    "prompt_tokens_total", "tokens_predicted_total", "tokens_predicted_seconds_total",
    # The time spent reading prompts, beside the tokens read: kept per replica
    # bucket for the dashboard (VD-147). The usage ledger ignores it.
    "prompt_seconds_total",
})


def parse_llamacpp_counters(text: str) -> Dict[str, float]:
    """The cumulative token counters from a llama.cpp ``/metrics`` body.

    Keyed by the counter's own name without the prefix
    (``prompt_tokens_total``, ``tokens_predicted_total``). The same honesty rule
    as :func:`parse_llamacpp_metrics`: an omitted or non-finite counter is an
    absent key, and a body that is not the expected text yields ``{}``.
    """
    result: Dict[str, float] = {}
    if not isinstance(text, str):
        return result
    for line in text.splitlines():
        match = _SAMPLE.match(line.strip())
        if match is None or not match.group("name").startswith(_PREFIX):
            continue
        name = match.group("name")[len(_PREFIX):]
        if name not in _LLAMACPP_COUNTERS:
            continue
        try:
            value = float(match.group("value"))
        except (TypeError, ValueError):
            continue
        if math.isfinite(value):
            result[name] = value
    return result


#: The ``vllm:`` families the serving card and per-model RED read, mapped to the
#: normalized keys the vLLM serving poll aggregates. The gauges are read as they
#: stand; the ``*_total`` counters feed the decode/prefill RATES the poller
#: derives from two reads (a total here means nothing without a second sample and
#: a time base), and each histogram ``_sum``/``_count`` pair feeds a TTFT/TPOT
#: average. One dict so the exact spellings, which a vLLM upgrade may re-prefix,
#: sit in a single place the way the llama.cpp gauge map above does.
#:
#: These deep families ARE emitted by the vLLM serving engine; the drift the
#: card hit was a V0->V1 rename this map now tracks: the KV-cache gauge is
#: ``kv_cache_usage_perc`` (was ``gpu_cache_usage_perc``) and the per-token
#: latency histogram is ``inter_token_latency_seconds`` (was
#: ``time_per_output_token_seconds``). Added alongside them are the prefix-cache
#: hit/query counters, the preemption counter, and the end-to-end latency
#: histogram whose ``_sum``/``_count`` and cumulative buckets back the goodput
#: ratio (see :data:`GOODPUT_BUDGET_SECONDS`).
_VLLM_SERIES: Dict[str, str] = {
    "vllm:num_requests_running": "requests_processing",
    "vllm:num_requests_waiting": "requests_deferred",
    "vllm:kv_cache_usage_perc": "kv_cache_fraction",
    "vllm:generation_tokens_total": "generation_tokens_total",
    "vllm:prompt_tokens_total": "prompt_tokens_total",
    "vllm:request_success_total": "request_success_total",
    "vllm:time_to_first_token_seconds_sum": "ttft_seconds_sum",
    "vllm:time_to_first_token_seconds_count": "ttft_seconds_count",
    "vllm:inter_token_latency_seconds_sum": "tpot_seconds_sum",
    "vllm:inter_token_latency_seconds_count": "tpot_seconds_count",
    "vllm:prefix_cache_hits_total": "prefix_cache_hits_total",
    "vllm:prefix_cache_queries_total": "prefix_cache_queries_total",
    "vllm:num_preemptions_total": "preemptions_total",
    "vllm:e2e_request_latency_seconds_sum": "e2e_seconds_sum",
    "vllm:e2e_request_latency_seconds_count": "e2e_total_count",
}

#: The label vLLM stamps on every per-model series. Found by substring, not regex,
#: so the per-model split costs nothing the aggregate sum does not already pay.
_MODEL_LABEL = 'model_name="'

#: The families that are a FRACTION or a ratio, not a count. Summing these
#: across a body's per-model series - or across replicas in the poller - is
#: meaningless (two replicas at 0.25 KV-cache are not 0.50 full), so they are
#: aggregated by MAX: the busiest contributor is the honest headline, and a
#: mean would hide one hot replica behind an idle one. Everything else
#: (running, waiting, token and success counts, and the histogram sum/count
#: pair a pooled average is later built from) is additive and keeps summing.
_MAX_KEYS = frozenset({"kv_cache_fraction"})


def _model_name(series: str) -> str:
    """The ``model_name`` label carried by one Prometheus series, or ``""``."""
    start = series.find(_MODEL_LABEL)
    if start == -1:
        return ""
    start += len(_MODEL_LABEL)
    end = series.find('"', start)
    return series[start:end] if end != -1 else ""


def _accumulate(store: Dict[str, float], key: str, value: float) -> None:
    """Fold one series value into ``store`` - MAX for a fraction, else a sum."""
    if key in _MAX_KEYS:
        prior = store.get(key)
        store[key] = value if prior is None else max(prior, value)
    else:
        store[key] = store.get(key, 0.0) + value

#: The end-to-end request latency below which a request counts toward goodput,
#: in seconds. The share of requests finishing within this budget is the
#: goodput ratio; it is a single documented constant so the parser's bucket
#: selection and the poller's reported budget cannot drift apart.
GOODPUT_BUDGET_SECONDS = 30.0

#: The cumulative-histogram series whose per-bucket ``le`` edge the goodput
#: count is read from. Matched on the bare series name, its ``le`` label read
#: separately.
_E2E_BUCKET_SERIES = "vllm:e2e_request_latency_seconds_bucket"

#: The time-to-first-token histogram's buckets, kept (summed over every label
#: set) under ``ttft_bucket:<edge seconds>`` so the poller can turn each
#: cumulative count into what changed during a tick (`generation_health`).
_TTFT_BUCKET_SERIES = "vllm:time_to_first_token_seconds_bucket"
TTFT_BUCKET_KEY_PREFIX = "ttft_bucket:"

#: The ``finished_reason`` label values that count as an error for per-model
#: RED. A completion that stopped or hit its length cap is a success; one that
#: was aborted or errored is not.
_ERROR_FINISH_REASONS = frozenset({"abort", "error"})

#: The label prefixes read out of a series by substring, the same cheap way
#: :data:`_MODEL_LABEL` is read.
_FINISH_LABEL = 'finished_reason="'
_LE_LABEL = 'le="'


def _finished_reason(series: str) -> str:
    """The ``finished_reason`` label carried by one series, or ``\"\"``."""
    start = series.find(_FINISH_LABEL)
    if start == -1:
        return ""
    start += len(_FINISH_LABEL)
    end = series.find('"', start)
    return series[start:end] if end != -1 else ""


def _bucket_le(series: str) -> 'float | None':
    """The finite ``le`` bucket edge of one histogram series, or ``None``.

    ``le=\"+Inf\"`` (the overflow bucket) and a missing or unparseable label
    both return ``None``, so the goodput selection ignores the total bucket and
    never trusts a malformed edge.
    """
    start = series.find(_LE_LABEL)
    if start == -1:
        return None
    start += len(_LE_LABEL)
    end = series.find('"', start)
    if end == -1:
        return None
    try:
        edge = float(series[start:end])
    except (TypeError, ValueError):
        return None
    return edge if math.isfinite(edge) else None


def _fold_goodput_buckets(
    aggregate: Dict[str, float],
    models: Dict[str, Dict[str, float]],
    bucket_best: Dict[str, Any],
) -> None:
    """Write the chosen within-budget bucket count into aggregate and per model.

    ``bucket_best`` maps a scope (a ``model_name`` label, or ``\"\"`` for an
    unlabelled series) to the ``(edge, count)`` of the smallest ``le`` bucket at
    or above the budget. The per-model counts sum into the aggregate, so the
    aggregate goodput is the whole-engine share; a scope with no qualifying
    bucket leaves no key, keeping the honesty rule.
    """
    if not bucket_best:
        return
    total = 0.0
    for scope, (edge, count) in bucket_best.items():
        total += count
        if scope:
            models.setdefault(scope, {})["e2e_within_budget_count"] = count
    aggregate["e2e_within_budget_count"] = total


def parse_vllm_serving(text: str) -> Dict[str, Any]:
    """The vLLM ``/metrics`` families the serving card and per-model RED need.

    Returns ``{"aggregate": {...}, "models": {name: {...}}}``. ``aggregate`` SUMS
    each family over every labelled series (all models, all finish reasons), which
    is the whole-engine reading the card shows; ``models`` keeps the same
    normalized keys split by the ``model_name`` label, which is what per-model RED
    reads. Additive families (running/waiting, the token and success counts, the
    histogram ``_sum``/``_count`` pair) are SUMMED across labelled series;
    ``kv_cache_fraction`` is a ratio, so it is taken as the MAX across them (see
    :data:`_MAX_KEYS`), never summed into an impossible fraction. Both carry only
    the families the body reported as finite numbers: a family the body omits is
    an absent key here, never a fabricated ``0`` - the same honesty rule
    :func:`parse_llamacpp_metrics` keeps. Any input that is not
    the expected text yields empty maps rather than raising, because the caller is
    a poll loop that must survive a truncated or wrong body.

    It does NO rate arithmetic. The ``*_total`` counters come back as their raw
    cumulative values; the decode and prefill tokens-per-second the card shows are
    a delta the poller computes across two of these reads, not a number this pure
    parse could honestly return from one body.

    Beyond the queue and token counters it also captures, per family and per
    model: the prefix-cache hit/query counters, the preemption counter, the
    end-to-end latency ``_sum``/``_count``, and ``e2e_within_budget_count`` - the
    cumulative count in the smallest e2e bucket whose ``le`` edge is at or above
    :data:`GOODPUT_BUDGET_SECONDS`. ``request_success_total`` still sums every
    finish reason, and ``request_errors_total`` sums only the aborted and errored
    ones, so per-model RED has both a request and an error count.
    """
    aggregate: Dict[str, float] = {}
    models: Dict[str, Dict[str, float]] = {}
    if not isinstance(text, str):
        return {"aggregate": aggregate, "models": models}
    #: scope (model_name or "") -> (le_edge, cumulative_count) of the smallest
    #: e2e bucket at or above the budget, resolved into goodput after the loop.
    bucket_best: Dict[str, Any] = {}
    for raw in text.splitlines():
        line = raw.strip()
        if not line or line.startswith("#"):
            continue
        fields = line.split()
        if len(fields) < 2:
            continue
        series = fields[0]
        metric = series.split("{", 1)[0]
        try:
            value = float(fields[-1])
        except (TypeError, ValueError):
            continue
        if not math.isfinite(value):
            continue
        if metric == _TTFT_BUCKET_SERIES:
            edge = _bucket_le(series)
            if edge is not None:
                _accumulate(aggregate, "%s%r" % (TTFT_BUCKET_KEY_PREFIX, edge), value)
            continue
        if metric == _E2E_BUCKET_SERIES:
            edge = _bucket_le(series)
            if edge is None or edge < GOODPUT_BUDGET_SECONDS:
                continue
            scope = _model_name(series)
            prior = bucket_best.get(scope)
            if prior is None or edge < prior[0]:
                bucket_best[scope] = (edge, value)
            continue
        key = _VLLM_SERIES.get(metric)
        if key is None:
            continue
        _accumulate(aggregate, key, value)
        model = _model_name(series)
        if model:
            _accumulate(models.setdefault(model, {}), key, value)
        if key == "request_success_total" and _finished_reason(series) in _ERROR_FINISH_REASONS:
            _accumulate(aggregate, "request_errors_total", value)
            if model:
                _accumulate(models.setdefault(model, {}), "request_errors_total", value)
    _fold_goodput_buckets(aggregate, models, bucket_best)
    return {"aggregate": aggregate, "models": models}


# --- The per-replica performance reading (VD-147, slice S3) ------------------
#: The two vLLM histograms that can say how fast requests decode. The
#: per-request one (``request_time_per_output_token_seconds``: one observation
#: per finished request, its mean seconds per token) is what the dashboard's
#: speed bands are built from; the inter-token one is the fallback for a build
#: that does not emit it, and the tooltip then says "per token, not per
#: request". The numbers are the ``family`` the ledger stores.
FAMILY_PER_REQUEST = 1
FAMILY_INTER_TOKEN = 2
_HISTOGRAM_SERIES = {
    "vllm:request_time_per_output_token_seconds_bucket": FAMILY_PER_REQUEST,
    "vllm:inter_token_latency_seconds_bucket": FAMILY_INTER_TOKEN,
}

#: The histogram of how many tokens each finished request produced. Its
#: ``le="1.0"`` bucket counts the requests that wrote 0 or 1 token - and vLLM
#: observes exactly those with a time-per-token of ZERO (``decode_time/(n-1)``
#: with nothing to divide by), so they land in the fastest speed bin and read
#: as 100 tok/s or more. They are counted here so the ledger can take them
#: back out of that bin (S0 finding, 2026-09-29, read from the image's source).
_SHORT_REPLY_SERIES = "vllm:request_generation_tokens_bucket"
_SHORT_REPLY_EDGE = 1.0

#: Prompt tokens by where they came from. ``vllm:prompt_tokens_total`` COUNTS
#: reused tokens (measured: a prompt sent twice added 48 to it, 32 of them from
#: cache), so it cannot be the "computed" layer of a cache chart; this family
#: can: ``local_compute`` is prefill that ran, ``local_cache_hit`` is prefill
#: that was skipped.
_PROMPT_SOURCE_SERIES = "vllm:prompt_tokens_by_source_total"
_SOURCE_LABEL = 'source="'

#: When the engine process created its token counter: an exact restart marker.
#: A counter going backwards also shows a restart, but only once the new
#: process has counted less than the old one had; this changes at once.
_CREATED_SERIES = "vllm:generation_tokens_created"


def _label(series: str, prefix: str) -> str:
    """The value of one label in a Prometheus series name, or ``""``."""
    start = series.find(prefix)
    if start == -1:
        return ""
    start += len(prefix)
    end = series.find('"', start)
    return series[start:end] if end != -1 else ""


def _raw_le(series: str) -> Optional[float]:
    """A histogram series' ``le`` edge, ``+Inf`` as ``math.inf``; ``None`` if unreadable."""
    raw = _label(series, _LE_LABEL)
    if not raw:
        return None
    try:
        edge = float(raw)
    except ValueError:
        return None
    return None if math.isnan(edge) else edge


def parse_vllm_performance(text: str) -> Dict[str, Any]:
    """What one vLLM ``/metrics`` body says about speed bands, cache sources and restarts.

    Returns::

        {"histograms": {family: {"edges": [finite edges, ascending],
                                  "cumulative": [count per edge, then the +Inf total]}},
         "short_replies": cumulative count of requests that wrote 0 or 1 token, or None,
         "prompt_sources": {"local_compute": n, "local_cache_hit": n, ...},
         "created": epoch seconds the engine created its token counter, or None}

    Every count is summed over the body's label sets (every model), as
    :func:`parse_vllm_serving` sums its families. A family the body does not
    carry is absent from ``histograms``; a histogram with no ``+Inf`` line is
    dropped rather than given an invented total. Cumulative counts are returned
    as they stand: what changed between two readings is the ledger's to work
    out, against the baseline it keeps. Anything that is not the expected text
    yields the empty shape, never an exception.
    """
    buckets: Dict[int, Dict[float, float]] = {}
    sources: Dict[str, float] = {}
    short: Optional[float] = None
    created: Optional[float] = None
    if isinstance(text, str):
        for raw in text.splitlines():
            line = raw.strip()
            if not line or line.startswith("#"):
                continue
            fields = line.split()
            if len(fields) < 2:
                continue
            series = fields[0]
            metric = series.split("{", 1)[0]
            try:
                value = float(fields[-1])
            except (TypeError, ValueError):
                continue
            if not math.isfinite(value):
                continue
            family = _HISTOGRAM_SERIES.get(metric)
            if family is not None:
                edge = _raw_le(series)
                if edge is not None:
                    counts = buckets.setdefault(family, {})
                    counts[edge] = counts.get(edge, 0.0) + value
            elif metric == _SHORT_REPLY_SERIES:
                if _raw_le(series) == _SHORT_REPLY_EDGE:
                    short = (short or 0.0) + value
            elif metric == _PROMPT_SOURCE_SERIES:
                source = _label(series, _SOURCE_LABEL)
                if source:
                    sources[source] = sources.get(source, 0.0) + value
            elif metric == _CREATED_SERIES:
                created = value if created is None else max(created, value)
    histograms: Dict[int, Dict[str, List[float]]] = {}
    for family, counts in buckets.items():
        if math.inf not in counts:
            continue
        edges = sorted(edge for edge in counts if edge != math.inf)
        histograms[family] = {
            "edges": edges,
            "cumulative": [counts[edge] for edge in edges] + [counts[math.inf]],
        }
    return {
        "histograms": histograms, "short_replies": short,
        "prompt_sources": sources, "created": created,
    }


def histogram_in_use(performance: Dict[str, Any]) -> Optional[Tuple[int, Dict[str, List[float]]]]:
    """``(family, histogram)`` the speed bands are built from, or ``None``.

    The per-request family when the body carries it, otherwise the inter-token
    fallback - never both, so a bucket holds one kind of band.
    """
    histograms = performance.get("histograms") or {}
    for family in (FAMILY_PER_REQUEST, FAMILY_INTER_TOKEN):
        if family in histograms:
            return family, histograms[family]
    return None
