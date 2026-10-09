"""The Performance dashboard's words, ranges and now layer (VD-147 S4, spec §1, §3, §6).

This module owns the dashboard's vocabulary: the freshness words and their
labels (:func:`freshness_projection` is their only producer - the browser holds
no threshold and no word), the value-state words, the serving-routing words,
and the step table. It also assembles the NOW layer, the part of the dashboard
that refreshes every ten seconds: the engine's state, each machine's freshness,
the hottest-GPU and busiest-CPU tiles, and every panel's freshness. The range
layer (tiles T1/T2/T3/T5, the panels, events) is `performance_dashboard_range`.

Everything here is a pure function of the inputs `performance_dashboard_source`
gathers, so the §6 state matrix is tested without a store. Rules held
throughout (§3.4):

* base units only - the frontend formats every number;
* a node's ``key`` is internal and never rendered; its ``name`` always is;
* a missing value is ``None`` with a state and a reason, never a zero;
* reasons are the owner constants, referenced, never retyped.
"""

from __future__ import annotations

from typing import Any, Dict, List, Mapping, Optional, Sequence, Tuple

from .gpu_vendor_status import HEALTHY_CODES, STATUS_FIELD, known_code, status_sentence
from .performance_serving import (
    CLUSTER_LOADING, CLUSTER_NONE, CLUSTER_SERVING, CLUSTER_UNKNOWN, CLUSTER_UNLOADED,
    PLACEMENT_SPLIT,
)
from .platforms.gpu_temperature import gpu_temperature_block
from .serving_metrics import SERVING_POLL_SECONDS
from .telemetry_ingest_status import node_is_reporting
from .telemetry_store import DEFAULT_RETENTION_DAYS, NOT_REPORTING_LABEL, REPORTING_WINDOW_SECONDS

# --------------------------------------------------------------------------
# Ranges and steps (§2.3)
# --------------------------------------------------------------------------

#: Every range the dashboard offers, with its one step. A step is a multiple of
#: the serving grain (20 s) and, from six hours up, of the roll-up's minute; no
#: range has more than :data:`DASHBOARD_MAX_BUCKETS` buckets.
RANGE_STEPS: Tuple[Tuple[str, int, int], ...] = (
    ("15m", 15 * 60, 20),
    ("1h", 3600, 20),
    ("6h", 6 * 3600, 120),
    ("24h", 24 * 3600, 360),
    ("3d", 3 * 86400, 1080),
    ("7d", 7 * 86400, 2520),
)

#: The range the tab opens on (owner decision D4).
DEFAULT_RANGE = "1h"

#: The most buckets one dashboard series carries. Distinct from the telemetry
#: store's ``MAX_HISTORY_BUCKETS`` (168), which exists because a small model
#: reads that payload; the two answer different questions and are not aligned.
DASHBOARD_MAX_BUCKETS = 240


def offered_ranges(retention_days: int = DEFAULT_RETENTION_DAYS) -> Dict[str, Tuple[int, int]]:
    """``{range: (seconds, step)}`` for every range the store's retention can fill.

    Tied to the one retention constant, so a change of the retention period
    changes the offered ranges and nothing here hard-codes seven days.
    """
    limit = int(retention_days) * 86400
    return {key: (seconds, step) for key, seconds, step in RANGE_STEPS if seconds <= limit}


#: The sentence for a range the dashboard does not offer (a 400).
RANGE_NOT_OFFERED = "The dashboard shows the last {offered}; pick one of those."


def range_refused(retention_days: int = DEFAULT_RETENTION_DAYS) -> str:
    return RANGE_NOT_OFFERED.format(offered=", ".join(offered_ranges(retention_days)))


# --------------------------------------------------------------------------
# Freshness: the one producer of the words (§3.4, review S7)
# --------------------------------------------------------------------------

FRESH_LIVE = "live"
FRESH_PARTIAL = "partial"
FRESH_STALE = "stale"
FRESH_NOT_REPORTING = "not_reporting"
FRESH_UNLOADED = "unloaded"
FRESH_LOADING = "loading"
FRESH_NOT_SERVING = "not_serving"
FRESH_UNKNOWN = "unknown"
FRESH_NOT_COLLECTED = "not_collected"

#: The label a screen shows for each freshness word.
FRESHNESS_LABELS: Dict[str, str] = {
    FRESH_LIVE: "Live",
    FRESH_PARTIAL: "Partly live",
    FRESH_STALE: "Not live",
    FRESH_NOT_REPORTING: NOT_REPORTING_LABEL,
    FRESH_UNLOADED: "Unloaded",
    FRESH_LOADING: "Loading",
    FRESH_NOT_SERVING: "Not serving",
    FRESH_UNKNOWN: "Unknown",
    FRESH_NOT_COLLECTED: "Not collected",
}


#: The freshness vocabulary as written down for its wire table
#: (`tests/test_wire_vocabularies`); held equal to the labels' keys below.
FRESHNESS_WORDS = frozenset({
    "live", "partial", "stale", "not_reporting", "unloaded", "loading",
    "not_serving", "unknown", "not_collected",
})


#: The colour family a screen gives each freshness word (the frontend's
#: ``statusTones``: neutral, info, success, warning, danger).
TONE_NEUTRAL = "neutral"
TONE_INFO = "info"
TONE_SUCCESS = "success"
TONE_WARNING = "warning"
TONE_DANGER = "danger"
#: The tone vocabulary, written down for its wire table: the payload sends
#: these and the frontend's ``statusTones`` colours exactly these.
STATUS_TONES = frozenset({"neutral", "info", "success", "warning", "danger"})
FRESHNESS_TONES: Dict[str, str] = {
    FRESH_LIVE: TONE_SUCCESS, FRESH_PARTIAL: TONE_WARNING, FRESH_STALE: TONE_WARNING,
    FRESH_NOT_REPORTING: TONE_DANGER, FRESH_UNLOADED: TONE_NEUTRAL, FRESH_LOADING: TONE_INFO,
    FRESH_NOT_SERVING: TONE_NEUTRAL, FRESH_UNKNOWN: TONE_WARNING, FRESH_NOT_COLLECTED: TONE_NEUTRAL,
}

#: How a screen dates a machine that stopped reporting; ``{time}`` is filled in
#: the viewer's own clock (pass-3 review: the browser holds no word).
NOT_REPORTING_SINCE = NOT_REPORTING_LABEL + " since {time}"


def freshness_projection(state: str, **detail: Any) -> Dict[str, Any]:
    """``{state, label, tone, ...detail}``: the ONLY place a freshness word meets its label."""
    if state not in FRESHNESS_LABELS:
        raise ValueError("unknown freshness word")
    return {"state": state, "label": FRESHNESS_LABELS[state], "tone": FRESHNESS_TONES[state], **detail}


# --------------------------------------------------------------------------
# Value states (§3.4)
# --------------------------------------------------------------------------

VALUE_MEASURED = "measured"
VALUE_PARTIAL = "partial"
VALUE_IDLE = "idle"
VALUE_NOT_REPORTED = "not_reported"
VALUE_NOT_MEASURED = "not_measured"
VALUE_NOT_REPORTING = "not_reporting"
VALUE_NOT_SERVING = "not_serving"
VALUE_UNKNOWN = "unknown"
#: The value-state vocabulary, written down for its wire table.
VALUE_STATES = frozenset({
    "measured", "partial", "idle", "not_reported", "not_measured", "not_reporting",
    "not_serving", "unknown",
})

# --------------------------------------------------------------------------
# Serving layout and routing (owner, 2026-09-30)
# --------------------------------------------------------------------------

#: How requests reach a deployment's copies. A replicated deployment keeps each
#: conversation on one replica (sticky routing), so its machines can be busy
#: unevenly; a split model has one engine and no routing between copies.
ROUTING_STICKY = "sticky"
ROUTING_ONE_ENGINE = "one-engine"
ROUTING_LOCAL = "single-machine"
#: The routing vocabulary, written down for its wire table.
ROUTING_WORDS = frozenset({"sticky", "one-engine", "single-machine"})

#: What the status strip says about each layout. The split sentence is the
#: serving block's own (`performance_serving.serving_scope_note`).
#: One line at 1920 px: the strip's sentence is room the first chart row
#: needs above the fold (frontend/scripts/dashboard-layout-contract.mjs).
STICKY_ROUTING_NOTE = (
    "Each conversation stays on one machine, so load can differ; cluster "
    "figures add or pool the machines."
)

#: The status strip's badge for each engine state.
BADGE_SERVING_MANY = "Serving on {count} machines"
BADGE_SERVING_PARTLY = "Serving: {read} of {total} machines read"
BADGE_SERVING_ONE_MODEL_SPLIT = "Serving one model split across {count} machines"
BADGE_SERVING_ONE_MODEL_SPLIT_UNCOUNTED = "Serving one model split across machines"
BADGE_SERVING_SINGLE = "Serving (single machine)"
BADGE_UNLOADED = "Unloaded"
BADGE_LOADING = "Loading"
BADGE_NOT_SERVING = FRESHNESS_LABELS[FRESH_NOT_SERVING]
BADGE_UNKNOWN = "Unknown"
BADGE_NO_LIVE_READING = "No live reading"

ENGINE_STATE_SERVING = "serving"
ENGINE_STATE_STALE = "stale"

#: A node's host readings when it has no row in the now layer's window and no
#: receipt to date it by.
NODE_NEVER_REPORTED = "Vaelor has no reading from this machine yet."


def _number(value: Any) -> Optional[float]:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        return None
    try:
        number = float(value)
    except (OverflowError, ValueError):
        return None
    return number if number == number and abs(number) != float("inf") else None


def engine_block(serving: Mapping[str, Any]) -> Dict[str, Any]:
    """The status strip's engine: kind, mode, state, placement, routing, badge and sentences.

    ``serving`` is `performance_serving.serving_section`'s block - the one owner
    of "is the engine live, stale, unloaded, loading, none or unknown".
    """
    cluster = str(serving.get("cluster_state") or CLUSTER_NONE)
    collected, stale = bool(serving.get("collected")), bool(serving.get("stale"))
    metrics = serving.get("metrics") if isinstance(serving.get("metrics"), Mapping) else {}
    placement = str(serving.get("placement") or "")
    if cluster == CLUSTER_UNKNOWN:
        kind, state = None, CLUSTER_UNKNOWN
    elif cluster in (CLUSTER_SERVING, CLUSTER_LOADING, CLUSTER_UNLOADED):
        kind = "vllm"
        state = cluster if cluster != CLUSTER_SERVING else (ENGINE_STATE_STALE if stale else ENGINE_STATE_SERVING)
    elif collected or stale:
        kind, state, placement = "llama.cpp", ENGINE_STATE_STALE if stale else ENGINE_STATE_SERVING, "single"
    else:
        kind, state = None, CLUSTER_NONE
    routing = {"replicated": ROUTING_STICKY, PLACEMENT_SPLIT: ROUTING_ONE_ENGINE, "single": ROUTING_LOCAL}.get(placement, "")
    total, read = _number(metrics.get("replicas_total")), _number(metrics.get("replicas_read"))
    machines = _number(serving.get("machines"))
    return {
        "kind": kind, "mode": "cluster" if kind == "vllm" else "single" if kind else None,
        "state": state, "cause": str(serving.get("cause") or ""),
        "placement": placement or None, "routing": routing,
        "routing_note": STICKY_ROUTING_NOTE if routing == ROUTING_STICKY else "",
        "badge": _badge(state, kind, placement, total, read, machines),
        "reason": str(serving.get("reason") or ""),
        "coverage_note": str(serving.get("coverage_note") or ""),
        "scope_note": str(serving.get("scope_note") or ""),
    }


def _badge(state: str, kind: Optional[str], placement: str, total: Optional[float], read: Optional[float],
           machines: Optional[float] = None) -> str:
    if state == CLUSTER_UNKNOWN:
        return BADGE_UNKNOWN
    if state == CLUSTER_UNLOADED:
        return BADGE_UNLOADED
    if state == CLUSTER_LOADING:
        return BADGE_LOADING
    if state == ENGINE_STATE_STALE:
        return BADGE_NO_LIVE_READING
    if state != ENGINE_STATE_SERVING:
        return BADGE_NOT_SERVING
    if kind == "llama.cpp":
        return BADGE_SERVING_SINGLE
    if placement == PLACEMENT_SPLIT:
        # A split model is ONE engine, so one replica: the machines it spans
        # are the record's count (pass-3 review: "split across 1 machines").
        if machines and machines >= 2:
            return BADGE_SERVING_ONE_MODEL_SPLIT.format(count=int(machines))
        return BADGE_SERVING_ONE_MODEL_SPLIT_UNCOUNTED
    if total and read is not None and read < total:
        return BADGE_SERVING_PARTLY.format(read=int(read), total=int(total))
    return BADGE_SERVING_MANY.format(count=int(total or 1))


# --------------------------------------------------------------------------
# Nodes and the now-layer tiles
# --------------------------------------------------------------------------


def node_freshness(node: Mapping[str, Any]) -> Dict[str, Any]:
    """A machine's freshness: its row's age judged by the one reporting rule.

    ``node`` carries ``row_age_seconds`` (the bounded now-layer read), ``ingest``
    (`describe_ingest`, ``None`` for the controller) and ``since`` (the newest
    row or receipt, epoch seconds, when the bounded read had none).
    """
    age = _number(node.get("row_age_seconds"))
    ingest = node.get("ingest") if isinstance(node.get("ingest"), Mapping) else None
    received = _number((ingest or {}).get("received_age_seconds"))
    basis = "receipt" if received is not None else "row"
    reporting = node_is_reporting(age, dict(ingest) if ingest else None) if (age is not None or received is not None) else False
    if reporting:
        return freshness_projection(FRESH_LIVE, age_seconds=received if received is not None else age, basis=basis, reason="")
    reason = str((ingest or {}).get("reason") or "")
    since = node.get("since")
    if since is None and not reason:
        reason = NODE_NEVER_REPORTED
    template = {"since_template": NOT_REPORTING_SINCE} if since is not None else {}
    return freshness_projection(FRESH_NOT_REPORTING, age_seconds=age, basis=basis, since=since, reason=reason,
                                **template)


def _gpu_status(row: Mapping[str, Any], name: str) -> Tuple[Optional[int], str]:
    code = known_code(row.get(STATUS_FIELD))
    if code is None:
        return None, ""
    return code, "" if code in HEALTHY_CODES else status_sentence(code, name)


def now_nodes(nodes: Sequence[Mapping[str, Any]]) -> List[Dict[str, Any]]:
    """Each machine for the strip and the legends: key, name, role, slot, freshness, GPU status."""
    shaped = []
    for node in nodes:
        row = node.get("row") if isinstance(node.get("row"), Mapping) else {}
        code, reason = _gpu_status(row, str(node.get("name") or ""))
        shaped.append({
            "key": str(node.get("key") or ""), "name": str(node.get("name") or ""),
            "role": str(node.get("role") or ""), "slot": node.get("slot"),
            "freshness": node_freshness(node),
            "clock_note": str(((node.get("ingest") or {}) if isinstance(node.get("ingest"), Mapping) else {}).get("reason") or ""),
            "gpu_status": code, "gpu_status_reason": reason,
        })
    return shaped


#: Why a now-layer tile has no value at all.
NO_MACHINE_REPORTING = "No machine is reporting right now, so there is no current reading."
NO_GPU_TEMPERATURE_READ = "No machine reported a GPU temperature."


def hottest_gpu_tile(nodes: Sequence[Mapping[str, Any]], fresh: Mapping[str, str]) -> Dict[str, Any]:
    """T4: the hottest GPU now, its own sensor and its own bands; every machine beneath."""
    rows = []
    for node in nodes:
        if fresh.get(str(node.get("key"))) != FRESH_LIVE:
            rows.append({"key": node.get("key"), "name": node.get("name"), "state": VALUE_NOT_REPORTING,
                         "value": None, "sensor": None, "bands": None, "note": ""})
            continue
        reading = dict(node.get("row") or {})
        if node.get("temperature_label") and "gpu_temperature_label" not in reading:
            reading["gpu_temperature_label"] = node.get("temperature_label")
        block = gpu_temperature_block(
            reading, node.get("limits"), node.get("policy") or None,
            class_known=node.get("class_known", True) is not False,
        )
        code, reason = _gpu_status(reading, str(node.get("name") or ""))
        state = VALUE_MEASURED if block["value_c"] is not None else VALUE_NOT_MEASURED
        rows.append({"key": node.get("key"), "name": node.get("name"), "state": state,
                     "value": block["value_c"], "sensor": block["sensor"], "bands": block["bands"],
                     "note": block["note"] or (reason if state == VALUE_NOT_MEASURED else "")})
    measured = [row for row in rows if row["value"] is not None]
    hottest = max(measured, key=lambda row: row["value"]) if measured else None
    if hottest is None:
        reason = NO_MACHINE_REPORTING if all(row["state"] == VALUE_NOT_REPORTING for row in rows) else NO_GPU_TEMPERATURE_READ
        return {"state": VALUE_NOT_MEASURED, "unit": "celsius", "cluster": None, "nodes": rows, "reason": reason}
    return {
        "state": VALUE_MEASURED, "unit": "celsius", "reason": "",
        "cluster": {"value": hottest["value"], "name": hottest["name"], "sensor": hottest["sensor"],
                    "bands": hottest["bands"], "note": hottest["note"]},
        "nodes": rows,
    }


def busiest_cpu_tile(nodes: Sequence[Mapping[str, Any]], fresh: Mapping[str, str]) -> Dict[str, Any]:
    """T6: the busiest host processor now, named (owner decision D6)."""
    rows = []
    for node in nodes:
        live = fresh.get(str(node.get("key"))) == FRESH_LIVE
        value = _number((node.get("row") or {}).get("cpu_percent")) if live else None
        state = VALUE_MEASURED if value is not None else VALUE_NOT_REPORTING if not live else VALUE_NOT_MEASURED
        rows.append({"key": node.get("key"), "name": node.get("name"), "state": state, "value": value})
    measured = [row for row in rows if row["value"] is not None]
    if not measured:
        return {"state": VALUE_NOT_MEASURED, "unit": "percent", "cluster": None, "nodes": rows,
                "reason": NO_MACHINE_REPORTING}
    busiest = max(measured, key=lambda row: row["value"])
    return {"state": VALUE_MEASURED, "unit": "percent", "reason": "",
            "cluster": {"value": busiest["value"], "name": busiest["name"]}, "nodes": rows}


# --------------------------------------------------------------------------
# Panel freshness
# --------------------------------------------------------------------------

SERVING_PANELS = ("output_throughput", "prompt_tokens", "decode_speed", "kv_cache")
HOST_PANELS = ("gpu_temperature", "gpu_power", "host_cpu_memory", "gpu_busy",
               "package_power", "gtt_used", "gpu_edge_temperature")
#: Serving panels llama.cpp does not feed at all.
LLAMACPP_NOT_COLLECTED = ("kv_cache",)

_ENGINE_TO_FRESH = {
    ENGINE_STATE_SERVING: FRESH_LIVE, ENGINE_STATE_STALE: FRESH_STALE,
    CLUSTER_UNLOADED: FRESH_UNLOADED, CLUSTER_LOADING: FRESH_LOADING,
    CLUSTER_NONE: FRESH_NOT_SERVING, CLUSTER_UNKNOWN: FRESH_UNKNOWN,
}


def panels_freshness(engine: Mapping[str, Any], node_states: Sequence[str]) -> Dict[str, Dict[str, Any]]:
    """Every panel's freshness: serving panels follow the engine, host panels the machines."""
    serving_word = _ENGINE_TO_FRESH.get(str(engine.get("state")), FRESH_UNKNOWN)
    if serving_word == FRESH_LIVE and engine.get("coverage_note"):
        serving_word = FRESH_PARTIAL
    live = sum(1 for state in node_states if state == FRESH_LIVE)
    host_word = FRESH_LIVE if node_states and live == len(node_states) else FRESH_PARTIAL if live else FRESH_NOT_REPORTING
    shaped = {}
    for panel in SERVING_PANELS:
        word = FRESH_NOT_COLLECTED if engine.get("kind") == "llama.cpp" and panel in LLAMACPP_NOT_COLLECTED else serving_word
        shaped[panel] = freshness_projection(word)
    for panel in HOST_PANELS:
        shaped[panel] = freshness_projection(host_word)
    return shaped


def build_now(inputs: Mapping[str, Any]) -> Dict[str, Any]:
    """The now layer (§3.3) from `performance_dashboard_source.gather_now`'s inputs."""
    engine = engine_block(inputs.get("serving") or {})
    nodes = list(inputs.get("nodes") or [])
    shaped = now_nodes(nodes)
    fresh = {node["key"]: node["freshness"]["state"] for node in shaped}
    # Imported here: the words module reads this one's vocabulary.
    from .performance_dashboard_words import BADGE_TONES, now_tile_words, annotate

    engine["badge_tone"] = BADGE_TONES.get(str(engine.get("state")), TONE_NEUTRAL)
    if engine.get("state") == ENGINE_STATE_SERVING and engine.get("coverage_note"):
        engine["badge_tone"] = TONE_WARNING
    tiles = {"hottest_gpu": hottest_gpu_tile(nodes, fresh), "busiest_cpu": busiest_cpu_tile(nodes, fresh)}
    now_tile_words(tiles)
    return annotate({
        "generated_at": inputs.get("now"),
        "refresh_seconds": SERVING_POLL_SECONDS,
        "reporting_window_seconds": REPORTING_WINDOW_SECONDS,
        "engine": engine,
        "nodes": shaped,
        "tiles": tiles,
        "panels_freshness": panels_freshness(engine, [node["freshness"]["state"] for node in shaped]),
    })
