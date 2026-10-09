"""Request health for every way a request reaches the model: rate, errors, latency.

The Performance card's request figures used to be read from the inference
gateway's own log alone, so a request an app sent through the LLM Server - which
never passes through any Vaelor process - left the card reading "No recent
traffic" while the model's own counters showed it served. Every door that
records its requests one by one now feeds one view:

* **the inference gateway** (``/inference/v1``), from its request log
  (:mod:`vaelor.inference_metrics`, source ``gateway``);
* **the LLM Server**, from its gate's access log (:mod:`vaelor.llm_gate_usage`),
  one line per admitted request with the gate's own time taken;
* **AI Chat**, from the same request log as the gateway under its own source
  (``ai-chat``) - but only while a GPU cluster serves it: an answer from this
  machine's own model is not timed one by one, and its row says so.

The door ids are :mod:`vaelor.usage_rollup`'s source words, so a request is
filed under the door that recorded it and nowhere else. "Door" is this module's
word; nothing the owner reads says it.

**Percentiles come only from measured per-request times.** Each door's p50/p95/
p99 are the nearest-rank percentiles of that door's own requests, so a slow door
is never hidden inside a fast total. The combined figure is the nearest-rank
percentile of every door's
requests POOLED - each request counted once, at the one door it came through -
never an average of the doors' percentiles, which is not a percentile of
anything. Each door's time is an end-to-end request time as that door saw it:
the gateway's proxied round trip, the gate's ``$request_time`` (first request
byte to last response byte), and AI Chat's call to the model until the whole
answer was read.

**Whole-request time is information, not a verdict** (owner decision,
2026-09-29). It is mostly how long the answer is - a long answer from a model
writing 13 tokens a second takes half a minute however healthy it is - so no
door and no total is judged on it. The speed verdict is judged on the model's
own time to first word and writing speed (:mod:`vaelor.generation_health`),
which the Performance snapshot sets on the combined block; every row here
carries ``latency_role: "information"`` and no verdict of its own.

**An LLM Server 503 is a request to try again, not a served request.** It is
what the model behind the LLM Server answers while it is loading (llama.cpp's
"Loading model" on this machine) or waking from being unloaded (a cluster's
wake door, VD-145), in a few milliseconds, so it is counted apart
(``retry_later``) and left out of the latency percentiles and the failure rate.
The sentence says both, because the gate's log cannot tell which it was.

**A door that cannot be read is never a zero.** Its row carries a state other
than :data:`DOOR_MEASURED` with the reason in plain words, and its counts are
``None`` unless requests it did record are in the span.
"""

from __future__ import annotations

import math
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence

from .assistant_console_places import INFERENCE_GATEWAY, LLM_SERVER
from .inference_metrics import detail_coverage, is_client_error, is_failure
from .usage_rollup import SOURCE_AI_CHAT, SOURCE_GATEWAY, SOURCE_LLM_SERVER, SOURCE_UNATTRIBUTED

#: What a whole-request percentile is for: information only. The verdict's
#: basis is `generation_health`'s; this word says so on every row.
LATENCY_INFORMATION = "information"

#: The doors, in the order the card lists them. A request recorded before its
#: store kept a door is filed as unattributed and listed only when present.
DOOR_ORDER = (SOURCE_GATEWAY, SOURCE_LLM_SERVER, SOURCE_AI_CHAT)

#: What a door's row is called on the card.
DOOR_LABELS = {
    SOURCE_GATEWAY: INFERENCE_GATEWAY,
    SOURCE_LLM_SERVER: LLM_SERVER,
    SOURCE_AI_CHAT: "AI Chat",
    SOURCE_UNATTRIBUTED: "Way in not recorded",
}
#: What the unattributed row is, said on the row itself (ACC-168): the label
#: alone read as a door that failed to record something.
UNATTRIBUTED_DETAIL = (
    "These requests were recorded before Vaelor kept which way each one came "
    "in, so they cannot be listed under one. They are counted in the totals above."
)
#: What a door is called inside a sentence.
_NOUNS = {
    SOURCE_GATEWAY: "the inference gateway",
    SOURCE_LLM_SERVER: "the LLM Server",
    SOURCE_AI_CHAT: DOOR_LABELS[SOURCE_AI_CHAT],
    SOURCE_UNATTRIBUTED: "requests from before Vaelor recorded where each came in",
}

#: Whether a door's requests are in the figures: measured (read and counting),
#: not known (its record could not be read or is not being written), off (it is
#: switched off), or not timed (its requests are not timed one by one right
#: now). The one owner of these words.
DOOR_STATES = ("measured", "not-known", "off", "not-timed")
DOOR_MEASURED, DOOR_NOT_KNOWN, DOOR_OFF, DOOR_NOT_TIMED = DOOR_STATES

#: For a record this reading was not handed (an older wiring or a test double;
#: production hands every reader the same sources).
DOOR_NOT_WIRED = "The record of these requests was not available to this reading, so they are not known here."
AI_CHAT_NOT_TIMED = (
    "AI Chat's answers are timed one by one only while a GPU cluster serves "
    "them, so AI Chat's requests right now are not in these figures."
)
LLM_SERVER_OFF = "The LLM Server is off, so no request can reach the model through it."
#: The LLM Server's usage-log states, said about requests and their times.
LLM_SERVER_NOT_LOGGING = (
    "The LLM Server is not recording its usage log right now, or Vaelor could not "
    "ask whether it is, so its requests and how long they took are not known."
)
LLM_SERVER_NOT_READING = (
    "Vaelor has not read the LLM Server's usage log recently, so its newest "
    "requests and how long they took may be missing."
)
LLM_SERVER_UNREADABLE = (
    "Vaelor could not read the LLM Server's usage log, so its requests and how "
    "long they took are not known."
)
#: The sentence a door carries instead when requests it recorded ARE in the
#: span, so its state never contradicts the counts beside it.
_WITH_ROWS = {
    DOOR_OFF: "Switched off now; the requests shown arrived before that.",
    DOOR_NOT_TIMED: "Not timed one by one right now; the requests shown were timed earlier.",
    DOOR_NOT_KNOWN: "Its record could not be read just now, so there may be more requests than shown.",
}
TOKENS_EXCLUDE_LLM_SERVER = (
    "Tokens are counted at the inference gateway and AI Chat only; the LLM "
    "Server's gate does not see them. Each model's own count is under Model usage."
)

#: What the LLM Server answers while the cluster model wakes: try again later.
RETRY_STATUS = 503

#: The span a request card's own window is called in a sentence.
THIS_WINDOW = "in this window"


def percentile(values: Sequence[float], fraction: float) -> Optional[float]:
    """The nearest-rank percentile of ``values``, or ``None`` when empty.

    Nearest-rank (not interpolated) because the inputs are measured request
    times and an interpolated p99 between two real samples invents a latency
    nobody observed. ``fraction`` is 0..1; ``percentile(xs, 0.95)`` is the p95.
    """
    ordered = sorted(float(value) for value in values)
    if not ordered:
        return None
    rank = max(1, math.ceil(fraction * len(ordered)))
    return ordered[min(rank, len(ordered)) - 1]


def door_of(row: Mapping[str, Any]) -> str:
    """The door a request row was recorded at; unattributed when it names none."""
    door = str(row.get("door") or "")
    return door if door in DOOR_LABELS else SOURCE_UNATTRIBUTED


def is_retry_later(row: Mapping[str, Any]) -> bool:
    """An LLM Server 503: the app was asked to try again, nothing was served."""
    try:
        status = int(row.get("status") or 0)
    except (TypeError, ValueError):
        return False
    return door_of(row) == SOURCE_LLM_SERVER and status == RETRY_STATUS


def _latency(rows: Sequence[Mapping[str, Any]]) -> Optional[Dict[str, float]]:
    durations = [float(row.get("duration_ms", 0) or 0) for row in rows]
    if not durations:
        return None
    return {
        "p50": round(percentile(durations, 0.50) or 0.0, 1),
        "p95": round(percentile(durations, 0.95) or 0.0, 1),
        "p99": round(percentile(durations, 0.99) or 0.0, 1),
        "max": round(max(durations), 1),
    }


#: No row is judged on its whole-request time (see the module docstring); the
#: keys stay, empty, so every reader of the old shape keeps reading it.
_NOT_JUDGED = {"within_budget": None, "over_budget_by_ms": None, "latency_role": LATENCY_INFORMATION}


def request_red(
    rows: Sequence[Mapping[str, Any]],
    window_seconds: int,
) -> Dict[str, Any]:
    """RED for one window of recorded requests, latency stated as percentiles.

    ``rows`` are raw request records (``status``, ``duration_ms``, token
    counts where the door sees them). With none, ``traffic`` is ``False`` and
    every latency and rate is ``None`` - there is no honest 0 ms for a window
    nothing happened in.

    ``failures`` counts server-side failures only (5xx); a request the model
    refused as malformed or too long is the caller's error and is counted in
    ``client_errors`` instead (ACC-056).
    """
    count = len(rows)
    seconds = max(1, int(window_seconds))
    if count == 0:
        return {
            **_NOT_JUDGED, "traffic": False, "requests": 0, "failures": 0, "client_errors": 0,
            "requests_per_min": 0.0, "error_rate": None, "latency_ms": None,
            "tokens_per_sec": None, "prompt_tokens": 0, "completion_tokens": 0,
            "note": "No request was recorded in this window.",
        }
    failures = sum(1 for row in rows if is_failure(row.get("status")))
    prompt_tokens = sum(max(0, int(row.get("prompt_tokens", 0) or 0)) for row in rows)
    completion_tokens = sum(max(0, int(row.get("completion_tokens", 0) or 0)) for row in rows)
    latency = _latency(rows)
    return {
        **_NOT_JUDGED, "traffic": True, "requests": count, "failures": failures,
        "client_errors": sum(1 for row in rows if is_client_error(row.get("status"))),
        "requests_per_min": round(count / (seconds / 60.0), 2),
        "error_rate": round(failures / count, 4),
        "latency_ms": latency,
        "tokens_per_sec": round((prompt_tokens + completion_tokens) / seconds, 2),
        "prompt_tokens": prompt_tokens, "completion_tokens": completion_tokens,
        "note": "",
    }


def _door_row(
    door: str, rows: Sequence[Mapping[str, Any]], retries: int, seconds: int,
    state: Mapping[str, Any], coverage: Optional[Mapping[str, Any]],
) -> Dict[str, Any]:
    """One door's figures (whole-request time as information), or ``None`` counts when it could not be read."""
    word = str(state.get("state") or DOOR_MEASURED)
    recorded = bool(rows) or retries > 0
    known = word == DOOR_MEASURED or recorded
    detail = str(state.get("detail") or "")
    if recorded and word in _WITH_ROWS:
        detail = _WITH_ROWS[word]
    elif door == SOURCE_UNATTRIBUTED and not detail:
        detail = UNATTRIBUTED_DETAIL
    count = len(rows)
    failures = sum(1 for row in rows if is_failure(row.get("status")))
    latency = _latency(rows)
    return {
        "door": door,
        "label": DOOR_LABELS[door],
        "state": word,
        "detail": detail,
        "traffic": bool(rows),
        "requests": count if known else None,
        "failures": failures if known else None,
        "client_errors": sum(1 for row in rows if is_client_error(row.get("status"))) if known else None,
        "retry_later": retries if known else None,
        "error_rate": round(failures / count, 4) if count else None,
        "requests_per_min": round(count / (max(1, seconds) / 60.0), 2) if known else None,
        "latency_ms": latency,
        **_NOT_JUDGED,
        "coverage": dict(coverage) if coverage else None,
    }


def request_health(
    rows: Sequence[Mapping[str, Any]],
    window_seconds: int,
    *,
    door_states: Optional[Mapping[str, Mapping[str, Any]]] = None,
    coverage: Optional[Mapping[str, Mapping[str, Any]]] = None,
) -> Dict[str, Any]:
    """The combined RED of every door plus one row per door.

    ``rows`` carry a ``door``; ``door_states`` maps a door to ``{state,
    detail}`` (a door with no entry is measured); ``coverage`` maps a door to
    what its per-request detail covers. The combined figures pool every door's
    served requests (see the module docstring). ``unmeasured`` names the doors
    whose traffic is not known, and ``retry_later`` the LLM Server's 503
    answers. No row carries a verdict: the snapshot judges the model's timings.
    """
    states = dict(door_states or {})
    served: List[Mapping[str, Any]] = []
    by_door: Dict[str, List[Mapping[str, Any]]] = {}
    retries: Dict[str, int] = {}
    for row in rows:
        door = door_of(row)
        by_door.setdefault(door, [])
        if is_retry_later(row):
            retries[door] = retries.get(door, 0) + 1
            continue
        by_door[door].append(row)
        served.append(row)
    doors = list(DOOR_ORDER) + ([SOURCE_UNATTRIBUTED] if SOURCE_UNATTRIBUTED in by_door else [])
    seconds = max(1, int(window_seconds))
    door_rows = [
        _door_row(door, by_door.get(door, []), retries.get(door, 0), seconds,
                  states.get(door) or {}, (coverage or {}).get(door))
        for door in doors
    ]
    block = request_red(served, seconds)
    block["doors"] = door_rows
    block["latency_basis"] = "pooled-per-request"
    block["unmeasured"] = [row["door"] for row in door_rows if row["state"] == DOOR_NOT_KNOWN]
    block["retry_later"] = sum(retries.values())
    block["tokens_note"] = TOKENS_EXCLUDE_LLM_SERVER if by_door.get(SOURCE_LLM_SERVER) else ""
    if not block["traffic"]:
        block["note"] = quiet_clause(block, THIS_WINDOW) + "."
    return block


def door_coverage(
    retained_since: Mapping[str, Optional[float]], *, window_start: float, lifetime: bool,
    kept: Optional[Mapping[str, int]] = None,
) -> Dict[str, Dict[str, Any]]:
    """What each door's per-request detail covers (ACC-047), keyed by door.

    ``kept`` is how many requests that door's store keeps one by one, where it
    differs from the gateway's.
    """
    return {
        door: detail_coverage(
            since, window_start=window_start, lifetime=lifetime, kept=(kept or {}).get(door),
        )
        for door, since in retained_since.items()
    }


def join_nouns(doors: Iterable[str]) -> str:
    """``the LLM Server`` / ``the inference gateway and the LLM Server`` / ``a, b and c``."""
    nouns = [_NOUNS[door] for door in doors if door in _NOUNS]
    if len(nouns) <= 1:
        return "".join(nouns)
    return ", ".join(nouns[:-1]) + " and " + nouns[-1]


def doors_with_traffic(block: Mapping[str, Any]) -> List[str]:
    return [row["door"] for row in block.get("doors") or () if row.get("traffic")]


def measured_doors(block: Mapping[str, Any]) -> List[str]:
    return [row["door"] for row in block.get("doors") or () if row.get("state") == DOOR_MEASURED]



def quiet_clause(block: Mapping[str, Any], span: str) -> str:
    """"No request was recorded {span} through ..." - naming what was, and was not, read."""
    measured = measured_doors(block)
    unknown = block.get("unmeasured") or []
    if not measured:
        clause = "No request's record could be read " + span
    else:
        clause = "No request was recorded " + span + " through " + join_nouns(measured)
    if unknown and measured:
        clause += ", and the requests through " + join_nouns(unknown) + " are not known"
    return clause


def unmeasured_sentences(block: Mapping[str, Any], *, include_untimed: bool) -> str:
    """The reasons for every door whose traffic is not in the figures.

    A door that cannot be read always; one that is off or not timed only when
    ``include_untimed`` (the no-traffic answer, where it explains a silence).
    """
    wanted = {DOOR_NOT_KNOWN} | ({DOOR_OFF, DOOR_NOT_TIMED} if include_untimed else set())
    sentences: List[str] = []
    for row in block.get("doors") or ():
        detail = row.get("detail")
        if row.get("state") in wanted and detail and not row.get("traffic") and detail not in sentences:
            sentences.append(detail)  # one reason shared by several doors is said once
    return " ".join(sentences)


def no_traffic_headline(block: Mapping[str, Any]) -> str:
    """The "nothing to diagnose" sentence, naming what was read and what was not."""
    return quiet_clause(block, THIS_WINDOW) + ", so there is no latency to diagnose."


def retry_sentence(block: Mapping[str, Any]) -> str:
    """The LLM Server's 503 answers in the span, said as what they are."""
    count = int(block.get("retry_later") or 0)
    if not count:
        return ""
    return (
        f"{count} request{'s' if count != 1 else ''} through the LLM Server "
        f"{'were' if count != 1 else 'was'} answered 503 (try again later): the model "
        "was loading or waking. They are left out of the latency and failure figures."
    )


def failure_breakdown(block: Mapping[str, Any]) -> str:
    """``the LLM Server: 2 of 5`` for each door with traffic, when more than one had any."""
    rows = [row for row in block.get("doors") or () if row.get("traffic")]
    if len(rows) < 2:
        return ""
    return "; ".join(
        "%s: %d of %d" % (_NOUNS[row["door"]], row["failures"] or 0, row["requests"] or 0)
        for row in rows
    )
