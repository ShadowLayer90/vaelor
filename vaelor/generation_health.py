"""How fast the model starts and writes an answer: the basis the speed verdict is judged on.

A chat answer's whole-request time is mostly how LONG the answer is: the
cluster's Qwen3-8B writes about 13 tokens a second per stream, so a 400-token
answer takes half a minute however healthy the model is, and a whole-request
p95 judged against a 2-second budget read every chat answer as "over budget"
(owner finding, 2026-09-29). So the speed verdict is judged on the two things a
person waiting on an answer actually feels, both measured by the model itself:

* **Time to first word** - how long after the request the first token came out
  (queueing plus reading the prompt). vLLM keeps it as the
  ``vllm:time_to_first_token_seconds`` histogram; judged as the p95 against
  :data:`TTFT_P95_BUDGET_MS`.
* **Speed while answering** - tokens per second per answer while it is being
  written, the inter-token intervals over the seconds they took (vLLM's
  ``vllm:inter_token_latency_seconds``; llama.cpp's
  ``tokens_predicted_total`` over ``tokens_predicted_seconds_total``), pooled
  across every answer and replica in the window, never a mean of means. Judged
  against :data:`DECODE_FLOOR_TOKENS_PER_SECOND`.

Whole-request times per way in stay on the card as information
(:mod:`vaelor.request_health`), never judged.

**Measured per window, by the pollers, never estimated.** Each tick of a
serving poller (:mod:`vaelor.serving_metrics_poller`) writes, beside the live
gauges, what changed during that tick: how many answers started and in which
time-to-first-word bucket (cumulative, as vLLM keeps them), and how many tokens
were written in how many seconds. A replica whose counters went backwards (a
restart) sits that tick out. :func:`serving_window_totals` sums those fields
over the window asked for, so the verdict covers exactly the window the card
shows, including answers that finished long before the newest sample.

**A percentile from buckets is a band, never interpolated** (VD-147 rule 3):
the p95 is reported as "between two bucket edges". The budget is set ON one of
vLLM 0.22.1's bucket edges (2.5 s; the edges are read from
``vllm/v1/metrics/loggers.py`` at that tag), so "is the p95 within the budget"
is answered exactly - the share of answers that began within 2.5 s is a count,
not an estimate - rather than left undecidable whenever the p95 falls inside the
1-to-2.5 s bucket that a 2-second budget would straddle.

**Nothing is judged without a basis.** Each figure carries one of
:mod:`vaelor.request_health`'s state words - ``measured``, ``not-known`` (the
store could not be read, or nothing was read in the window) or ``not-timed``
(the engine does not keep this timer: llama.cpp has no time-to-first-token
histogram) - and a plain sentence; a window with no answer in it is measured
and judged nothing.
"""

from __future__ import annotations

import math
from typing import Any, Dict, List, Mapping, Optional, Tuple

from .gpu_pool_units import LLAMACPP_ENGINE, VLLM_ENGINE
from .request_health import DOOR_MEASURED, DOOR_NOT_KNOWN, DOOR_NOT_TIMED

#: The time-to-first-word p95 a healthy model stays within, in milliseconds.
#: 2.5 s is a vLLM 0.22.1 histogram edge, so the verdict is exact (see the
#: module docstring); it covers queueing and reading the prompt, not the answer.
TTFT_P95_BUDGET_MS = 2500.0

#: The share of answers that must start within the budget: the p95.
TTFT_BUDGET_SHARE = 0.95

#: The slowest per-answer writing speed that still reads as healthy, in tokens a
#: second. A quick reader takes in about 300 words a minute - 5 words, or about
#: 6.5 tokens, a second - so an answer written at 8 or more stays ahead of the
#: person reading it; below that they wait on every line. The cluster's
#: Qwen3-8B in BF16 measured about 12.9 on 2026-09-29.
DECODE_FLOOR_TOKENS_PER_SECOND = 8.0

#: The per-tick fields the pollers write and this module sums - their one owner.
TTFT_COUNT_FIELD = "ttft_window_count"
TTFT_SECONDS_FIELD = "ttft_window_seconds"
TTFT_BUCKET_PREFIX = "ttft_window_le_ms_"
DECODE_TOKENS_FIELD = "decode_window_tokens"
DECODE_SECONDS_FIELD = "decode_window_seconds"
_WINDOW_PREFIXES = ("ttft_window_", "decode_window_")

#: The plain sentences, one per reason a figure has no verdict.
TTFT_NOT_TIMED = (
    "The model serving here does not keep a timer for when its first word came "
    "out (llama.cpp has none), so time to first word is not measured."
)
TIMINGS_NOT_READ = (
    "The model's own timings were not read in this window, so its speed is "
    "not known"
)
TIMINGS_UNREADABLE = (
    "The model's own timings could not be read from the telemetry store, so "
    "its speed is not known."
)
TTFT_NOT_REPORTED = (
    "The serving engine's readings in this window carry no time-to-first-word "
    "figure, so time to first word is not known."
)
NO_ANSWER_STARTED = "No answer started in this window, so there is no time to first word to judge."

#: Which serving engine a window was read from, as the snapshot knows it from
#: the fleet's own record (never inferred from which fields happen to be
#: missing, LESSONS 8): the cluster's vLLM, or this machine's llama.cpp.
ENGINE_VLLM = VLLM_ENGINE
ENGINE_LLAMACPP = LLAMACPP_ENGINE
NO_ANSWER_WRITTEN = "No answer was being written in this window, so there is no writing speed to judge."


def ttft_bucket_field(edge_seconds: float) -> str:
    """The per-tick field one cumulative time-to-first-token bucket is written under."""
    return "{}{}".format(TTFT_BUCKET_PREFIX, int(round(float(edge_seconds) * 1000)))


def _finite(value: Any) -> Optional[float]:
    if isinstance(value, bool) or value is None:
        return None
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    return number if math.isfinite(number) else None


def window_fields(values: Mapping[str, Any]) -> Dict[str, float]:
    """The per-tick window fields in ``values``, as floats (one field type in the store)."""
    fields: Dict[str, float] = {}
    for key, value in values.items():
        number = _finite(value)
        if number is not None and str(key).startswith(_WINDOW_PREFIXES):
            fields[str(key)] = number
    return fields


def serving_window_totals(
    database: Any, seconds: int, measurement: str,
) -> Optional[Dict[str, float]]:
    """Every window field summed over the last ``seconds``; ``None`` when unreadable.

    ``{}`` is a store that was read and holds no window field in the span -
    nothing was being served and read - which is not the same as a store that
    could not be read (``None``); the verdict says which.
    """
    if database is None:
        return None
    try:
        if hasattr(database, "is_ready") and not database.is_ready():
            return None
        span = max(1, int(seconds))
        rows = database.get_trend(measurement, span, span, function="sum")
    except Exception:  # noqa: BLE001 - an unreadable store is "not known", never zero
        return None
    totals: Dict[str, float] = {}
    for row in rows or ():
        if not isinstance(row, Mapping):
            continue
        for key, value in window_fields(row).items():
            totals[key] = totals.get(key, 0.0) + value
    return totals


def _buckets(totals: Mapping[str, float]) -> List[Tuple[float, float]]:
    """``(edge seconds, cumulative count)`` for every bucket in ``totals``, ascending."""
    edges: List[Tuple[float, float]] = []
    for key, value in totals.items():
        if not key.startswith(TTFT_BUCKET_PREFIX):
            continue
        try:
            edge_ms = int(key[len(TTFT_BUCKET_PREFIX):])
        except ValueError:
            continue
        edges.append((edge_ms / 1000.0, float(value)))
    return sorted(edges)


def _ttft(
    totals: Optional[Mapping[str, float]], not_read: str, engine: Optional[str],
) -> Dict[str, Any]:
    budget_s = TTFT_P95_BUDGET_MS / 1000.0
    base: Dict[str, Any] = {
        "budget_ms": TTFT_P95_BUDGET_MS, "budget_share": TTFT_BUDGET_SHARE,
        "answers": None, "p95_band_ms": None, "share_within_budget": None,
        "within_budget": None,
    }
    if totals is None:
        return {**base, "state": DOOR_NOT_KNOWN, "detail": TIMINGS_UNREADABLE}
    count = totals.get(TTFT_COUNT_FIELD)
    if count is None:
        # "Not timed" is a fact about the ENGINE, so it comes from the engine
        # the record says is serving - a vLLM tick that lost its histogram is
        # "not known", never llama.cpp's sentence. With no cluster the engine
        # is only inferred to be llama.cpp, so the sentence also needs
        # llama.cpp's own writing-speed figures in the window: a box with no
        # local model has none and must not be said to run llama.cpp.
        wrote = totals.get(DECODE_SECONDS_FIELD) is not None
        if engine == ENGINE_LLAMACPP and wrote and totals.get(DECODE_TOKENS_FIELD) is not None:
            return {**base, "state": DOOR_NOT_TIMED, "detail": TTFT_NOT_TIMED}
        if wrote:
            return {**base, "state": DOOR_NOT_KNOWN, "detail": TTFT_NOT_REPORTED}
        return {**base, "state": DOOR_NOT_KNOWN, "detail": not_read}
    answers = int(round(count))
    if answers <= 0:
        return {**base, "state": DOOR_MEASURED, "answers": 0, "detail": NO_ANSWER_STARTED}
    rank = max(1, math.ceil(TTFT_BUDGET_SHARE * answers))
    buckets = _buckets(totals)
    lower, upper = 0.0, None
    for edge, cumulative in buckets:
        if cumulative >= rank:
            upper = edge
            break
        lower = edge
    at_budget = next((cumulative for edge, cumulative in buckets if edge == budget_s), None)
    if at_budget is not None:
        share = min(1.0, at_budget / answers)
        within: Optional[bool] = at_budget >= rank
    else:
        share = None
        within = True if upper is not None and upper <= budget_s else (
            False if lower >= budget_s else None
        )
    band = [round(lower * 1000.0), None if upper is None else round(upper * 1000.0)]
    return {
        **base, "state": DOOR_MEASURED, "answers": answers, "p95_band_ms": band,
        "share_within_budget": None if share is None else round(share, 4),
        "within_budget": within, "detail": "",
    }


def _decode(totals: Optional[Mapping[str, float]], not_read: str) -> Dict[str, Any]:
    base: Dict[str, Any] = {
        "floor_tokens_per_second": DECODE_FLOOR_TOKENS_PER_SECOND,
        "tokens_per_second": None, "within_budget": None,
    }
    if totals is None:
        return {**base, "state": DOOR_NOT_KNOWN, "detail": TIMINGS_UNREADABLE}
    seconds = totals.get(DECODE_SECONDS_FIELD)
    tokens = totals.get(DECODE_TOKENS_FIELD)
    if seconds is None or tokens is None:
        return {**base, "state": DOOR_NOT_KNOWN, "detail": not_read}
    if seconds <= 0 or tokens <= 0:
        return {**base, "state": DOOR_MEASURED, "detail": NO_ANSWER_WRITTEN}
    speed = tokens / seconds
    return {
        **base, "state": DOOR_MEASURED, "tokens_per_second": round(speed, 1),
        "within_budget": speed >= DECODE_FLOOR_TOKENS_PER_SECOND, "detail": "",
    }


def generation_health(
    totals: Optional[Mapping[str, float]], *, serving_reason: str = "",
    engine: Optional[str] = None,
) -> Dict[str, Any]:
    """The speed verdict for one window, from the model's own timings only.

    ``totals`` is :func:`serving_window_totals`'s answer (``None`` when the
    store could not be read); ``serving_reason`` is the serving section's own
    sentence for why the engine is not being read (unloaded, not serving),
    added when nothing was read in the window; ``engine`` is
    :data:`ENGINE_VLLM`, :data:`ENGINE_LLAMACPP` or ``None`` when the record
    could not say. ``within_budget`` is ``False``
    when either judged figure is outside its budget, ``True`` when every judged
    figure is inside, and ``None`` when nothing could be judged.
    """
    not_read = TIMINGS_NOT_READ + "." + (" " + serving_reason if serving_reason else "")
    ttft = _ttft(totals, not_read, engine)
    decode = _decode(totals, not_read)
    verdicts = [part["within_budget"] for part in (ttft, decode) if part["within_budget"] is not None]
    return {
        "basis": "model-timings",
        "ttft": ttft,
        "decode": decode,
        "within_budget": (all(verdicts) if verdicts else None),
        "judged": [name for name, part in (("ttft", ttft), ("decode", decode))
                   if part["within_budget"] is not None],
    }



def seconds_text(milliseconds: Optional[float]) -> str:
    """``2.5 s`` / ``250 ms`` for a band edge, ``more than the largest bucket`` for none."""
    if milliseconds is None:
        return "longer than the model's largest timer bucket"
    if milliseconds < 1000:
        return "%d ms" % round(milliseconds)
    return ("%.1f s" % (milliseconds / 1000.0)).replace(".0 s", " s")


def ttft_sentence(ttft: Mapping[str, Any]) -> str:
    """How the answers in the window started, in plain words, from the measured figures.

    The count is the ENGINE's - every answer whose first word the model timed,
    whichever way its request came in (ACC-147: 37 of 39 had gone straight to
    the balancer) - so it is said as the model's own and never credited to one
    door, and as answers STARTED, since first words are what it counts.
    """
    share = ttft.get("share_within_budget")
    band = ttft.get("p95_band_ms") or [None, None]
    budget = seconds_text(ttft.get("budget_ms"))
    answered = "the model started %d answer%s" % (
        ttft.get("answers") or 0, "" if ttft.get("answers") == 1 else "s",
    )
    if share is not None:
        return "%s; %s%% began within %s" % (answered, _percent(share), budget)
    within = "%s; 95%% began within %s" % (answered, seconds_text(band[1]))
    if not band[0]:
        # The first bucket starts at 0 ms, and every answer took longer than
        # that: the band has no lower edge worth saying.
        return within
    return "%s, and some took longer than %s" % (within, seconds_text(band[0]))


def decode_sentence(decode: Mapping[str, Any]) -> str:
    """The pooled writing speed: tokens over the seconds spent writing them, every answer together."""
    return "answers were written at %s tokens a second on average while writing" % _number(
        decode.get("tokens_per_second"))


def _percent(share: float) -> str:
    value = share * 100.0
    return ("%.1f" % value).rstrip("0").rstrip(".")


def _number(value: Any) -> str:
    number = _finite(value)
    if number is None:
        return "?"
    return ("%.1f" % number).rstrip("0").rstrip(".")

