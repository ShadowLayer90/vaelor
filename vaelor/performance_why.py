"""The Performance tab's one-line "why": the most degraded signal, named from real readings.

Split out of `performance_snapshot` (near the 1,000-line ceiling) when the
speed verdict moved from whole-request time to the model's own timings (owner
decision, 2026-09-29): a chat answer's whole-request time is mostly how long
the answer is, so judging it against a 2-second budget read every answer as
"over budget". The verdict is now the model's time to first word and its
writing speed (:mod:`vaelor.generation_health`); whole-request times are named
in the detail as information, never as the problem.

It never invents a bottleneck: a signal that was not measured is said to be
not measured, with the reason, and the attribution names a pegged GPU or a hot
CPU only when a node reported one.
"""

from __future__ import annotations

import math
from typing import Any, Dict, List, Mapping, Optional, Sequence

from .generation_health import (
    DECODE_FLOOR_TOKENS_PER_SECOND, TTFT_BUDGET_SHARE, TTFT_P95_BUDGET_MS,
    decode_sentence, seconds_text, ttft_sentence,
)
from .request_health import (
    doors_with_traffic, failure_breakdown, join_nouns, no_traffic_headline,
    retry_sentence, unmeasured_sentences,
)

#: An error rate at or above this is called out on its own, independent of how
#: it compares to the baseline: a tenth of requests failing is a problem even if
#: it has been failing all along.
ERROR_RATE_ATTENTION = 0.05

#: A utilisation percentage at or above this reads as "this resource is the
#: ceiling" for the saturation verdict and the latency attribution.
SATURATION_PERCENT = 85.0

#: A GPU this busy while answers are slow points at a compute-bound serving
#: path; below GPU_IDLE_PERCENT with a hot CPU points at prefill on the CPU.
GPU_PEGGED_PERCENT = 90.0
GPU_IDLE_PERCENT = 50.0

#: The smallest weight a speed problem carries against a failure rate, so a
#: breached budget is not outranked by a trickle of errors (the old latency
#: verdict's "rose" floor, kept).
_SPEED_SEVERITY_FLOOR = 0.25


def _finite(value: Any) -> Optional[float]:
    if isinstance(value, bool) or value is None:
        return None
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    return number if math.isfinite(number) else None


def _pegged_node(nodes_use: Sequence[Mapping[str, Any]]) -> Optional[str]:
    """The name of a node whose GPU is pegged, if any - the compute-bound tell."""
    for node in nodes_use:
        gpu = node.get("use", {}).get("gpu_busy_percent")
        if gpu is not None and gpu >= GPU_PEGGED_PERCENT:
            return node.get("name")
    return None


def _cpu_bound_node(nodes_use: Sequence[Mapping[str, Any]]) -> Optional[str]:
    """A node with a hot CPU and an idle GPU - the prefill-on-CPU tell."""
    for node in nodes_use:
        cpu = node.get("use", {}).get("cpu_percent")
        gpu = node.get("use", {}).get("gpu_busy_percent")
        if cpu is not None and cpu >= SATURATION_PERCENT and (gpu is None or gpu < GPU_IDLE_PERCENT):
            return node.get("name")
    return None


def _serving_pressure_note(serving_metrics: Mapping[str, Any]) -> str:
    """A factual clause about the serving queue/decode, from collected values only.

    A non-zero waiting queue is the queue-bound tell, and the decode rate says
    how fast the engine is emitting tokens right now. Built only from numbers
    the engine reported - an absent gauge contributes nothing - so the clause
    is empty unless there is something real to say.
    """
    waiting = _finite(serving_metrics.get("requests_deferred"))
    running = _finite(serving_metrics.get("requests_processing"))
    decode = _finite(serving_metrics.get("decode_tokens_per_second"))
    if waiting is not None and waiting >= 1:
        tail = "" if decode is None else f" while each stream decodes at {round(decode)} tok/s"
        running_text = "" if running is None else f" behind {round(running)} running"
        return (
            f"The serving engine has {round(waiting)} request(s) waiting{running_text}"
            f"{tail} — the queue is backing up."
        )
    if decode is not None and running is not None and running >= 1:
        return f"The serving engine is decoding at {round(decode)} tok/s per stream with {round(running)} request(s) running."
    return ""


def _engine_only_why(
    serving_metrics: Mapping[str, Any], recent: Mapping[str, Any],
) -> Optional[Dict[str, Any]]:
    """The "why" when the engine is working but no request was recorded (ACC-050).

    Every door records a request when it FINISHES, so a long stream is not in
    any log until it ends. An empty window beside an engine reporting requests
    running or waiting is therefore not "no traffic". ``None`` when the live
    gauges show nothing in flight (or were not collected).
    """
    running = _finite(serving_metrics.get("requests_processing"))
    waiting = _finite(serving_metrics.get("requests_deferred"))
    if (running or 0) < 1 and (waiting or 0) < 1:
        return None
    return {
        "signal": "engine_only",
        "headline": (
            f"The serving engine is working ({round(running or 0)} running, "
            f"{round(waiting or 0)} waiting), but no finished request was recorded "
            "in this window, so there is no request latency to judge."
        ),
        "detail": " ".join(part for part in (
            "A request is recorded when it finishes, so requests still in flight "
            "are not in these figures yet.",
            retry_sentence(recent),
            unmeasured_sentences(recent, include_untimed=True),
            _serving_pressure_note(serving_metrics),
        ) if part),
    }


#: Said when the engine answered requests no recorded way in carried (ACC-148).
NO_DOOR_CARRIED = (
    "no recorded way in carried these requests - they reached the model some "
    "other way, such as straight to its balancer"
)


def _within_budget(generation: Mapping[str, Any]) -> str:
    """"Within budget: " and each judged figure of the model's, the one spelling."""
    judged = []
    if (generation.get("ttft") or {}).get("within_budget") is True:
        judged.append(ttft_sentence(generation["ttft"]))
    if (generation.get("decode") or {}).get("within_budget") is True:
        judged.append(decode_sentence(generation["decode"]))
    return "Within budget: " + ", and ".join(judged)


def _engine_judged_why(
    generation: Mapping[str, Any], nodes_use: Sequence[Mapping[str, Any]],
    recent: Mapping[str, Any],
) -> Optional[Dict[str, Any]]:
    """The verdict on the engine's own figures when no door recorded a request (ACC-148).

    The model timed answers in the window, but none came through a way in that
    records them, so "no request was recorded" would hide the engine's own
    first-word and writing-speed figures. ``None`` when the engine started no
    answer, or its figures could not be judged.
    """
    ttft = generation.get("ttft") or {}
    if not (ttft.get("answers") or 0) or generation.get("within_budget") is None:
        return None
    left_out = " ".join(part for part in (
        retry_sentence(recent), unmeasured_sentences(recent, include_untimed=False),
    ) if part)
    problem = _speed_problem(generation)
    if problem is not None:
        return {
            "signal": "latency",
            "headline": _capital("; ".join(problem["clauses"])) + f" - {_attribution(nodes_use)}",
            "detail": " ".join(part for part in (
                _capital(NO_DOOR_CARRIED) + ".", _missing_parts(generation), left_out,
            ) if part),
        }
    return {
        "signal": "within_budget",
        "headline": _within_budget(generation) + "; " + NO_DOOR_CARRIED + ".",
        "detail": " ".join(part for part in (_missing_parts(generation), left_out) if part),
    }


def _duration_text(milliseconds: Any) -> str:
    value = _finite(milliseconds) or 0.0
    if value < 1000:
        return "%d ms" % round(value)
    return ("%.1f s" % (value / 1000.0)).replace(".0 s", " s")


def whole_request_sentence(recent: Mapping[str, Any], through: str) -> str:
    """The whole-request percentiles, said as information and why they are not judged."""
    latency = recent.get("latency_ms") or {}
    if latency.get("p95") is None:
        return ""
    return (
        f"For information, whole requests through {through} took "
        f"{_duration_text(latency.get('p50'))} typically and "
        f"{_duration_text(latency.get('p95'))} at the 95th percentile; that is "
        "mostly how long each answer was, so it is not judged."
    )


def _attribution(nodes_use: Sequence[Mapping[str, Any]]) -> str:
    pegged = _pegged_node(nodes_use)
    cpu_bound = _cpu_bound_node(nodes_use)
    if pegged:
        return f"GPU/compute-bound on {pegged}, whose GPU is pegged."
    if cpu_bound:
        return f"CPU/prefill-bound on {cpu_bound}, whose CPU is hot while its GPU is idle."
    return (
        "No node's GPU or CPU is pegged, so the cause is not visible in the "
        "utilisation signals collected so far."
    )


def _speed_problem(generation: Mapping[str, Any]) -> Optional[Dict[str, Any]]:
    """``{severity, clauses}`` for each judged figure outside its budget, or ``None``."""
    ttft = generation.get("ttft") or {}
    decode = generation.get("decode") or {}
    clauses: List[str] = []
    severity = 0.0
    if ttft.get("within_budget") is False:
        share = _finite(ttft.get("share_within_budget"))
        clauses.append(
            "answers are slow to start: " + ttft_sentence(ttft)
            + ", where the budget is %d%% within %s" % (
                round(TTFT_BUDGET_SHARE * 100), seconds_text(TTFT_P95_BUDGET_MS),
            )
        )
        shortfall = (TTFT_BUDGET_SHARE - share) / TTFT_BUDGET_SHARE if share is not None else 0.25
        severity = max(severity, _SPEED_SEVERITY_FLOOR + max(0.0, shortfall))
    if decode.get("within_budget") is False:
        speed = _finite(decode.get("tokens_per_second")) or 0.0
        clauses.append(
            decode_sentence(decode) + ", under the floor of %g tokens a second"
            % DECODE_FLOOR_TOKENS_PER_SECOND
        )
        shortfall = (DECODE_FLOOR_TOKENS_PER_SECOND - speed) / DECODE_FLOOR_TOKENS_PER_SECOND
        severity = max(severity, _SPEED_SEVERITY_FLOOR + max(0.0, shortfall))
    if not clauses:
        return None
    return {"severity": severity, "clauses": clauses}


def _missing_parts(generation: Mapping[str, Any]) -> str:
    """The reasons for each timing that was not judged, each said once."""
    sentences: List[str] = []
    for part in ("ttft", "decode"):
        detail = str((generation.get(part) or {}).get("detail") or "")
        if detail and detail not in sentences:
            sentences.append(detail)
    return " ".join(sentences)


def _capital(text: str) -> str:
    return text[:1].upper() + text[1:]


def diagnose(
    recent: Mapping[str, Any],
    baseline: Mapping[str, Any],
    nodes_use: Sequence[Mapping[str, Any]],
    serving: Optional[Mapping[str, Any]] = None,
) -> Dict[str, Any]:
    """The one-line "why" - from real signals only.

    Failures are diffed against the baseline window; speed is judged on the
    model's own time to first word and writing speed in the recent window
    (``recent["generation"]``, from :func:`vaelor.generation_health.generation_health`),
    and the most degraded signal is named with a bounded attribution. When no
    timing could be judged the answer says so and why, rather than "within
    budget". Whole-request percentiles appear in the detail as information.
    """
    serving_metrics = (
        serving.get("metrics") or {}
        if isinstance(serving, Mapping) and serving.get("collected")
        else {}
    )
    if not recent.get("traffic"):
        engine_judged = _engine_judged_why(recent.get("generation") or {}, nodes_use, recent)
        if engine_judged is not None:
            return engine_judged
        engine_only = _engine_only_why(serving_metrics, recent)
        if engine_only is not None:
            return engine_only
        if recent.get("retry_later"):
            return {
                "signal": "retry_later",
                "headline": retry_sentence(recent),
                "detail": unmeasured_sentences(recent, include_untimed=True),
            }
        return {
            "signal": "no_traffic",
            "headline": no_traffic_headline(recent),
            "detail": unmeasured_sentences(recent, include_untimed=True),
        }
    candidates: List[tuple[float, Dict[str, Any]]] = []
    through = join_nouns(doors_with_traffic(recent)) or "every way in"
    left_out = " ".join(part for part in (
        retry_sentence(recent), unmeasured_sentences(recent, include_untimed=False),
    ) if part)
    information = whole_request_sentence(recent, through)

    error_rate = recent.get("error_rate") or 0.0
    baseline_error = baseline.get("error_rate") or 0.0
    if error_rate >= ERROR_RATE_ATTENTION or error_rate > baseline_error:
        rose = error_rate > baseline_error
        breakdown = failure_breakdown(recent)
        candidates.append((
            error_rate,
            {
                "signal": "errors",
                "headline": (
                    f"Failures rising: {round(error_rate * 100, 1)}% of requests through "
                    f"{through} are failing"
                    + (f", up from {round(baseline_error * 100, 1)}%." if rose else ".")
                ),
                "detail": " ".join(part for part in (
                    f"{recent.get('failures', 0)} of {recent.get('requests', 0)} requests "
                    "failed with a server error"
                    + (f" ({breakdown})." if breakdown else "."),
                    left_out,
                ) if part),
            },
        ))

    generation = recent.get("generation") or {}
    problem = _speed_problem(generation)
    if problem is not None:
        headline = _capital("; ".join(problem["clauses"])) + f" - {_attribution(nodes_use)}"
        detail = " ".join(part for part in (
            _serving_pressure_note(serving_metrics), _missing_parts(generation),
            information, left_out,
        ) if part)
        candidates.append((problem["severity"], {"signal": "latency", "headline": headline, "detail": detail}))

    if candidates:
        candidates.sort(key=lambda item: item[0], reverse=True)
        return candidates[0][1]
    if generation.get("within_budget") is None:
        return {
            "signal": "not_measured",
            "headline": (
                f"Requests came in through {through}, but the model's own timings "
                "could not be judged in this window, so there is no speed verdict."
            ),
            "detail": " ".join(part for part in (
                _missing_parts(generation), information, left_out,
            ) if part),
        }
    return {
        "signal": "within_budget",
        # The model's figures are the engine's, across all its traffic
        # (ACC-147), so no door is named ON them; the ways in the recorded
        # requests used are said in a sentence of their own (VD-148).
        "headline": (
            _within_budget(generation) + "; errors are low. "
            f"Recorded requests came in through {through}."
        ),
        "detail": " ".join(part for part in (
            _missing_parts(generation), information, left_out,
        ) if part),
    }
