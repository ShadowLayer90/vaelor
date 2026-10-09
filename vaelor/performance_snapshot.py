"""Assemble the cluster Performance snapshot — the "why is this slow?" surface.

This is §6b's MVP, and it is deliberately *intelligence-driven*: it does not
dump raw matrices, it answers one question in one sentence and then backs it
with the signals it used. Everything here is built from telemetry Vaelor
**already emits** - the request logs of every door a request can come in by
(:mod:`vaelor.request_health`) and the per-node utilisation history - so
nothing below is a scrape that does not yet exist.

Three honesty rules run through the whole module, and they are the point of it:

* **Speed is judged on the model's own timings, never a bare average.** The
  verdict is time to first word (a p95 against a stated budget, from the
  model's bucketed timer) and writing speed per answer against a floor
  (:mod:`vaelor.generation_health`). Whole-request p50/p95/p99 per way in stay
  as information - straight from the raw ``duration_ms`` rows, never averaged -
  because a long answer takes long to finish however healthy the model is.
* **A value that was not measured is never fabricated.** Zero requests in the
  window yields no latency at all (``None``), not a reassuring ``0 ms``. A node
  that is not reporting telemetry yields no USE, not ``0%``.
* **The serving engine's live gauges are collected; the deep per-request ones
  are honest-degraded, not guessed.** Phase E′ scrapes the GPU llama.cpp
  engine's Prometheus ``/metrics`` on the controller, so the decode/prefill rate
  and the running/waiting queue depth are now real readings in the ``serving``
  section — but only while the engine is serving, and an absent gauge is an
  absent value, never a ``0``. The deeper per-request signals - TTFT/TPOT,
  KV-cache utilisation, prefix-cache hit rate, preemptions, goodput and per-model
  RED - are all emitted by the vLLM serving engine, so once GPU clustering serves
  on vLLM the controller's scrape feeds them into the ``serving`` section too. On
  the single-node llama.cpp engine, which emits none of them, they are honestly
  named as "not collected" with the serving path that unlocks them, never a number.

The request RED covers **every door Vaelor measures** - the inference gateway,
the LLM Server and AI Chat - with one row per door and a combined figure
(``requests``), not per deployment: no door's record has a model column. The
combined percentiles pool the doors' measured per-request times; they are never
an average of per-door percentiles (:mod:`vaelor.request_health`).
"""

from __future__ import annotations

import time
from typing import Any, Dict, Mapping, Optional, Sequence

from .health_evaluation import evaluate_health
from .platforms.gpu_temperature import gpu_temperature_block
from .model_usage import deployment_usage
from .generation_health import (
    ENGINE_LLAMACPP, ENGINE_VLLM, TTFT_P95_BUDGET_MS, generation_health,
)
#: The one-line "why" lives in `performance_why` (split out at the line
#: ceiling); its thresholds are re-exported here for the readers that import
#: them from this module.
from .performance_why import (  # noqa: F401 - re-exported
    ERROR_RATE_ATTENTION, GPU_IDLE_PERCENT, GPU_PEGGED_PERCENT, SATURATION_PERCENT,
    diagnose,
)
from .request_health import (
    door_coverage, quiet_clause, request_health, request_red,
)
#: The serving section, its cluster words and its reason sentences live in
#: `performance_serving` (split out at the line ceiling, VD-147 S1a); they are
#: re-exported here for the readers that import them from this module.
from .performance_serving import (  # noqa: F401 - re-exported
    CLUSTER_LOADING, CLUSTER_NONE, CLUSTER_SERVING, CLUSTER_UNKNOWN, CLUSTER_UNLOADED,
    SERVING_CLUSTER_NOT_SCRAPED, SERVING_FUTURE_TOLERANCE_SECONDS, SERVING_LOADING,
    SERVING_NOT_COLLECTED, SERVING_RECORD_UNREADABLE, SERVING_SCALED_TO_ZERO,
    SERVING_STALE_AFTER_SECONDS,
    SERVING_UNLOADED, SERVING_UNLOADED_BY_HAND, UNLOAD_IDLE, UNLOAD_MANUAL,
    _finite, _human_window, serving_section, uncollected_signals,
)
from .inference_metrics import detail_coverage
from .usage_rollup import SOURCE_GATEWAY


#: The USE signals read off a node's latest telemetry, and the stored field each
#: comes from. Utilisation/saturation percentages plus the unified-memory (GTT)
#: footprint and the two temperatures — the ceiling-finding set, not everything.
_USE_FIELDS: tuple[tuple[str, str], ...] = (
    ("cpu_percent", "cpu_percent"),
    ("gpu_busy_percent", "gpu_busy_percent"),
    ("memory_percent", "memory_percent"),
    ("npu_activity_percent", "npu_activity_percent"),
    ("gpu_gtt_used_bytes", "gpu_gtt_used_bytes"),
    ("gpu_gtt_total_bytes", "gpu_gtt_total_bytes"),
    ("cpu_temperature_c", "cpu_temperature"),
    # The GPU's two temperature readings, each under its own name: edge (the
    # hwmon sensor) and graphics engine (amd-smi). Which one a screen shows is
    # `gpu_temperature_reading`'s choice, carried in the node's
    # ``gpu_temperature`` block with the sensor named (VD-147).
    ("gpu_temperature_c", "gpu_temperature_c"),
    ("gpu_gfx_temperature_c", "gpu_gfx_temperature_c"),
)

#: The three utilisation signals that can be a saturation ceiling, mapped to the
#: noun the verdict uses. Temperature and GTT are reported but do not by
#: themselves name a throughput ceiling.
_CEILING_SIGNALS = (
    ("gpu", "gpu_busy_percent"),
    ("cpu", "cpu_percent"),
    ("memory", "memory_percent"),
)


def node_use(
    node: Mapping[str, Any],
    thermal_policy: Optional[Mapping[str, Any]] = None,
) -> Dict[str, Any]:
    """One fleet node's USE block: its utilisation, a ceiling verdict, a health band.

    ``node`` carries ``id``/``name``/``role`` and ``metrics`` — the node's latest
    telemetry, or ``None``/empty when it is not reporting. A non-reporting node is
    marked ``reporting: False`` with no USE numbers rather than a wall of zeros,
    reusing the E2c "not reporting" idea.

    A node may carry its OWN ``thermal_policy`` (its machine class's bands) and
    ``gpu_temperature_limits`` (what its GPU reports for itself); they are used
    in place of the ``thermal_policy`` argument, which remains the fallback for
    a node that carries neither. A worker is never judged on the controller's
    bands (VD-147).
    """
    metrics = node.get("metrics") or {}
    label = node.get("gpu_temperature_label")
    if label and "gpu_temperature_label" not in metrics:
        # A worker's row carries numbers only; its sensor's name is on record.
        metrics = {**metrics, "gpu_temperature_label": label}
    own_policy = node.get("thermal_policy")
    policy = own_policy if isinstance(own_policy, Mapping) else thermal_policy
    limits = node.get("gpu_temperature_limits")
    use = {name: _finite(metrics.get(field)) for name, field in _USE_FIELDS}
    reporting = any(value is not None for value in use.values())
    ceiling_name: Optional[str] = None
    ceiling_value: Optional[float] = None
    for name, field in _CEILING_SIGNALS:
        value = use.get(field)
        if value is None:
            continue
        if ceiling_value is None or value > ceiling_value:
            ceiling_name, ceiling_value = name, value
    if not reporting:
        verdict = "Not reporting telemetry, so its utilisation cannot be read."
    elif ceiling_value is None:
        verdict = "Reporting, but no utilisation signal to judge saturation."
    elif ceiling_value >= SATURATION_PERCENT:
        verdict = f"{ceiling_name} is the ceiling at {round(ceiling_value)}%."
    else:
        verdict = f"Headroom on every resource; busiest is {ceiling_name} at {round(ceiling_value)}%."
    return {
        "id": str(node.get("id", "")),
        "name": str(node.get("name", "") or node.get("id", "")),
        "role": str(node.get("role", "")),
        "reporting": reporting,
        "last_sample_at": node.get("last_sample_at"),
        "last_sample_age_seconds": node.get("last_sample_age_seconds"),
        "use": use,
        "saturation": {
            "ceiling": ceiling_name if reporting else None,
            "value": round(ceiling_value, 1) if ceiling_value is not None else None,
            "saturated": bool(ceiling_value is not None and ceiling_value >= SATURATION_PERCENT),
            "verdict": verdict,
        },
        # The one GPU temperature a screen shows for this node, with the
        # sensor it came from and this node's own bands.
        "gpu_temperature": gpu_temperature_block(
            metrics, limits, policy or None, class_known=node.get("machine_class_known", True) is not False,
        ),
        "health": _node_health(metrics, policy, limits, node.get("machine_class_known", True) is not False),
    }


def _node_health(
    metrics: Mapping[str, Any], thermal_policy: Optional[Mapping[str, Any]],
    gpu_limits: Optional[Mapping[str, Any]] = None, class_known: bool = True,
) -> Dict[str, Any]:
    """A node's health band from its telemetry, or an honest not-evaluated note.

    Reuses ``evaluate_health`` so the bands and the "only claim what was
    measured" rule are the same ones Home and the Assistant answer with; a node
    with no readings produces ``checked: []``, which the caller renders as "not
    evaluated" rather than a green light.
    """
    evaluated = evaluate_health(dict(metrics or {}), dict(thermal_policy or {}), gpu_limits, class_known)
    return {
        "status": evaluated["status"],
        "reasons": evaluated["reasons"],
        "checked": evaluated["checked"],
        "notes": evaluated["notes"],
    }



#: The one wording for each Phoenix trace state, so the "on"/"starting"/"off"
#: copy reads the same everywhere it is rendered. Kept beside the other honest
#: "not collected" copy rather than spelled in the frontend.
_TRACES_RUNNING = "Per-request traces are captured in Phoenix."
_TRACES_STARTING = (
    "Phoenix is enabled but its collector is not up yet — traces will appear once "
    "it finishes starting."
)
_TRACES_OFF = (
    "Per-request traces are off — deploy Phoenix (Arize) to capture a span per "
    "inference request."
)


def traces_section(traces: Optional[Mapping[str, Any]]) -> Dict[str, Any]:
    """The Phoenix trace-collector block: enabled/running plus an honest reason.

    ``traces`` is the raw ``{enabled, running, otlp_endpoint, ui_port, ui_host}`` the
    ``phoenix_status`` callback returns (or ``None`` when tracing is unwired), and
    this normalizes it into the typed section the tab renders. ``collected`` is
    ``running`` — the affordance ("view request traces") shows only when the
    collector is actually up, and the honest-degrade copy shows otherwise, never a
    dead link. No value is fabricated: an absent block reads as off.
    """
    enabled = bool(traces.get("enabled")) if isinstance(traces, Mapping) else False
    running = bool(traces.get("running")) if isinstance(traces, Mapping) else False
    endpoint = str(traces.get("otlp_endpoint") or "") if isinstance(traces, Mapping) else ""
    ui_port = int(traces.get("ui_port") or 0) if isinstance(traces, Mapping) else 0
    ui_host = str(traces.get("ui_host") or "") if isinstance(traces, Mapping) else ""
    if running:
        reason = _TRACES_RUNNING
    elif enabled:
        reason = _TRACES_STARTING
    else:
        reason = _TRACES_OFF
    return {
        "enabled": enabled,
        "running": running,
        "collected": running,
        "otlp_endpoint": endpoint,
        "ui_port": ui_port,
        # Where the UI answers (W6-6), as the status said; "" when it did not.
        "ui_host": ui_host,
        "reason": reason,
    }


#: The snapshot keys renamed when the request card came to cover every door,
#: old name to new, served under both names for one release (Beta 2).
DEPRECATED_KEYS = {
    "gateway": "requests",
    "baseline_gateway": "baseline_requests",
    "gateway_detail": "requests.doors[].coverage",
    "gateway.scope": "requests.latency_basis",
}


def build_snapshot(
    *,
    window_seconds: int,
    recent_rows: Sequence[Mapping[str, Any]],
    baseline_rows: Sequence[Mapping[str, Any]],
    nodes: Sequence[Mapping[str, Any]],
    overall_health: Mapping[str, Any],
    thermal_policy: Optional[Mapping[str, Any]] = None,
    generated_at_ms: int = 0,
    serving: Optional[Mapping[str, Any]] = None,
    serving_window: Optional[Mapping[str, float]] = None,
    traces: Optional[Mapping[str, Any]] = None,
    deployments: Optional[Sequence[Mapping[str, Any]]] = None,
    detail_retained_since: Optional[Mapping[str, Optional[float]]] = None,
    detail_kept: Optional[Mapping[str, int]] = None,
    lifetime_rows: Sequence[Mapping[str, Any]] = (),
    lifetime_seconds: float = 0.0,
    now: Optional[float] = None,
    cluster: Optional[Mapping[str, Any]] = None,
    door_states: Optional[Mapping[str, Mapping[str, Any]]] = None,
) -> Dict[str, Any]:
    """The whole typed Performance snapshot, assembled from already-fetched data.

    Pure: every input is a plain structure the route fetched from the telemetry
    callbacks and the doors' request stores, so this is unit-testable without a
    Flask app, a database, or a live cluster. The shape is the seed of the
    future agent-parseable snapshot — typed and summarised, one sentence of
    diagnosis plus the signals behind it, not a raw matrix.

    Request rows carry the ``door`` they were recorded at; ``door_states``
    says, per door, whether its requests could be read (a door with no entry
    is measured); ``detail_retained_since`` says, per door, from when its
    per-request detail is complete (ACC-047). ``serving_window`` is the model's
    own timings summed over the window (`generation_health.serving_window_totals`;
    ``None`` when they could not be read), which the speed verdict is judged on.
    """
    clock = now if now is not None else (generated_at_ms / 1000.0 if generated_at_ms else time.time())
    retained = dict(detail_retained_since or {})

    def health(rows: Sequence[Mapping[str, Any]], seconds: int, lifetime: bool) -> Dict[str, Any]:
        coverage = door_coverage(
            retained, window_start=clock - int(window_seconds), lifetime=lifetime, kept=detail_kept,
        )
        return request_health(rows, seconds, door_states=door_states, coverage=coverage)

    recent = health(recent_rows, window_seconds, False)
    baseline = request_red(baseline_rows, window_seconds)
    nodes_use = [node_use(node, thermal_policy) for node in nodes]
    serving_block = serving_section(serving, clock, cluster)
    # The speed verdict (owner decision 2026-09-29): the model's own time to
    # first word and writing speed over this window, never whole-request time.
    # The serving section's own reason is added when nothing was read.
    # Which engine served comes from the fleet's record: a vLLM deployment on
    # record is the cluster's vLLM, none is this machine's llama.cpp, and an
    # unreadable record says neither.
    cluster_word = serving_block.get("cluster_state")
    engine = (
        ENGINE_VLLM if cluster_word in (CLUSTER_SERVING, CLUSTER_UNLOADED, CLUSTER_LOADING)
        else ENGINE_LLAMACPP if cluster_word == CLUSTER_NONE else None
    )
    generation = generation_health(
        serving_window,
        serving_reason="" if serving_block.get("collected") else str(serving_block.get("reason") or ""),
        engine=engine,
    )
    recent["generation"] = generation
    recent["budget_ms"] = TTFT_P95_BUDGET_MS
    recent["budget_metric"] = "ttft-p95"
    recent["within_budget"] = generation["within_budget"]
    # The request card's value: the windowed reading, or - only when NO door
    # recorded a request in the selected window - a lifetime fallback so the
    # card is not blank beside the lifetime per-deployment totals. The WHY
    # banner below stays windowed: it is diagnosed from ``recent``, never from
    # this fallback, so it still answers "right now".
    request_panel = dict(recent)
    request_panel["window_scope"] = "window"
    # Only a window that was truly quiet falls back: never while a door could
    # not be read (its traffic is not known, so the window is not "idle") nor
    # while the LLM Server was answering "try again later".
    quiet = not recent["traffic"] and not recent["unmeasured"] and not recent["retry_later"]
    if quiet and lifetime_rows:
        lifetime = health(lifetime_rows, max(1, int(lifetime_seconds)), True)
        if lifetime["traffic"]:
            lifetime["window_scope"] = "lifetime"
            # No verdict on lifetime data (ACC-052): a verdict is a claim about
            # how the model is doing now, and these requests may be days old.
            # The percentiles stay, stated with how old the newest is. The
            # model's timings are this WINDOW's, so a lifetime panel carries
            # none rather than figures from a different span.
            lifetime["budget_ms"] = TTFT_P95_BUDGET_MS
            lifetime["budget_metric"] = "ttft-p95"
            lifetime["within_budget"] = None
            newest = max(float(row.get("created_at", 0) or 0) for row in lifetime_rows)
            age = max(0.0, clock - newest) if newest > 0 else None
            lifetime["last_request_at"] = newest if newest > 0 else None
            lifetime["last_request_age_seconds"] = None if age is None else round(age)
            lifetime["note"] = (
                quiet_clause(recent, "in the last " + _human_window(window_seconds))
                + ("" if age is None else "; the most recent was " + _human_window(age) + " ago")
                + ". Showing lifetime totals since metering began, too old to judge "
                "the model's speed on."
            )
            request_panel = lifetime
    baseline_summary = {
        "traffic": baseline["traffic"],
        "requests": baseline["requests"],
        "error_rate": baseline["error_rate"],
        "p95_ms": (baseline["latency_ms"] or {}).get("p95") if baseline["latency_ms"] else None,
    }
    gateway_detail = next(
        (row["coverage"] for row in request_panel["doors"] if row["door"] == SOURCE_GATEWAY and row["coverage"]),
        detail_coverage(None, window_start=clock, lifetime=False),
    )
    return {
        "window_seconds": int(window_seconds),
        "baseline_window_seconds": int(window_seconds),
        "generated_at": int(generated_at_ms),
        # Every door's requests, combined and per door; each door row carries
        # what its per-request detail covers (ACC-047).
        "requests": request_panel,
        "baseline_requests": baseline_summary,
        # DEPRECATED aliases of the keys above, kept for one release so a reader
        # of the old shape keeps working (see ``deprecated_keys``). ``gateway``
        # now covers every door, and its ``scope`` says so.
        "gateway": {**request_panel, "scope": "all-doors"},
        "baseline_gateway": baseline_summary,
        "gateway_detail": gateway_detail,
        "deprecated_keys": dict(DEPRECATED_KEYS),
        "nodes": nodes_use,
        "serving": serving_block,
        "traces": traces_section(traces),
        # How much each deployment's MODEL did, from its own counters, and
        # whether that ledger could be read at all (ACC-044/045).
        "deployments": deployment_usage(deployments),
        "deployments_readable": deployments is not None,
        "health": {
            "status": overall_health.get("status", "offline"),
            "reasons": list(overall_health.get("reasons", [])),
            "checked": list(overall_health.get("checked", [])),
        },
        "why": diagnose(recent, baseline, nodes_use, serving_block),
        "uncollected": uncollected_signals(serving_block),
    }
