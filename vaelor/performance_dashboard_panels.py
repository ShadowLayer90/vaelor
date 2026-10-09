"""The Performance dashboard's range layer, assembled (VD-147 S4, spec §1.2, §3.2, §6).

`build_range` puts together what `performance_dashboard_range` folds from the
usage ledger (serving panels, tiles T1/T2/T3, events) and what the host buckets
hold (the GPU temperature, GPU power, host CPU and memory and GPU busy panels,
tile T5, and Advanced's three panels). One step for every panel in a response,
so a crosshair lines up across charts.

A panel whose source is absent is present with no series, a state and the
owner's reason - never omitted and never zero-filled. Which state, when the
serving panels have nothing to draw, is the engine's own word (`engine_block`).

What history means is decided by the history itself (pass-3 review S4-B4):
whether a stretch was llama.cpp or vLLM is each ledger row's own ``engine``,
never the engine serving now. Every title, label and tone the screen shows is
added here from `performance_dashboard_words`.
"""

from __future__ import annotations

from typing import Any, Dict, List, Mapping, Optional, Sequence, Tuple

from .performance_dashboard import (
    ENGINE_STATE_STALE, VALUE_IDLE, VALUE_MEASURED, VALUE_NOT_MEASURED, VALUE_NOT_REPORTING,
    VALUE_NOT_SERVING, VALUE_UNKNOWN, engine_block,
)
from .performance_dashboard_range import (
    AGGREGATE_NAMES, HOST_NOT_IN_RANGE, LAYER_CACHED, LAYER_COMPUTED, LAYER_PROCESSED, LLAMACPP_AT_COMPLETION,
    LLAMACPP_NO_BANDS, LLAMACPP_NO_CACHE_FIGURES, LLAMACPP_NO_KV, NO_SERVING_IN_RANGE, REMOVED_MACHINE,
    SERVES_NOTHING_SINGLE, SHORT_REPLIES_INCLUDED, UNDER_COVERED_NAMED, WHOLE_MODEL_KEY,
    avg_gpu_power_tile, cluster_series, current_model, events, field_samples, fold_rows, llama_only,
    measures_gpu_power, series_key, serving_names, serving_series, serving_tiles, speed_bands, stopped_buckets,
    tile_scope,
)
from .performance_dashboard_words import (
    OFF_UNTIL_CHOSEN, PANEL_SCOPES, PANEL_TITLES, TILE_TITLES, annotate, gpu_temperature_title, labelled_bands,
    range_labels, tile_caption,
)
from .performance_serving import CLUSTER_LOADING, CLUSTER_NONE, CLUSTER_UNKNOWN, CLUSTER_UNLOADED
from .platforms.gpu_temperature import (
    SENSOR_GRAPHICS_ENGINE, gpu_temperature_block, gpu_temperature_reading,
)
from .serving_buckets import MIN_BUCKET_COVERAGE

#: The unit each panel is drawn in (base units; the frontend formats).
_UNITS = {
    "output_throughput": "tokens_per_second", "prompt_tokens": "tokens_per_second",
    "decode_speed": "tokens_per_second", "kv_cache": "ratio", "gpu_temperature": "celsius",
    "gpu_power": "watts", "host_cpu_memory": "percent", "gpu_busy": "percent",
    "package_power": "watts", "gtt_used": "bytes", "gpu_edge_temperature": "celsius",
}

#: The serving state a panel takes when the engine had nothing to draw.
_ENGINE_TO_VALUE = {
    CLUSTER_UNKNOWN: VALUE_UNKNOWN, CLUSTER_NONE: VALUE_NOT_SERVING, CLUSTER_UNLOADED: VALUE_IDLE,
    CLUSTER_LOADING: VALUE_IDLE, ENGINE_STATE_STALE: VALUE_IDLE,
}

#: A machine's two lines in the host CPU and memory panel.
CPU_LINE = "{name} CPU"
MEMORY_LINE = "{name} memory"


def _number(value: Any) -> Optional[float]:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        return None
    number = float(value)
    return number if number == number and abs(number) != float("inf") else None


def _panel(name: str, series: List[Dict[str, Any]], state: str = "", reason: str = "", **extra: Any) -> Dict[str, Any]:
    drawn = any(value is not None for line in series for value in line.get("values") or [])
    return {"title": PANEL_TITLES.get(name, ""), "scope": PANEL_SCOPES.get(name, ""),
            "state": state or (VALUE_MEASURED if drawn else VALUE_IDLE),
            "unit": _UNITS[name], "reason": reason, "series": series, **extra}


def _serving_panels(inputs: Mapping[str, Any], engine: Mapping[str, Any], names: Mapping[str, str],
                    folded: Mapping[str, Any], start: int, step: int, count: int) -> Dict[str, Dict[str, Any]]:
    rows = list(inputs.get("replica_rows") or [])
    llamacpp = llama_only(rows)
    any_llamacpp = any(row.get("engine") == "llama.cpp" for row in rows)
    split = WHOLE_MODEL_KEY in folded
    if not folded:
        state = _ENGINE_TO_VALUE.get(str(engine.get("state")), VALUE_IDLE)
        reason = str(engine.get("reason") or "") or NO_SERVING_IN_RANGE
        return {name: _panel(name, [], state=state, reason=reason)
                for name in ("output_throughput", "prompt_tokens", "decode_speed", "kv_cache")}
    histograms = [] if llamacpp else list(inputs.get("histograms") or [])
    # Each machine's own bands, from its own replicas' histograms: two machines
    # on different edge sets keep theirs (pass-3 review), only the pooled
    # cluster line can have none in a bucket where the sets differ.
    replica_key = {str(row.get("replica")): series_key(row) for row in rows}
    node_bands = {key: speed_bands([row for row in histograms if replica_key.get(str(row.get("replica"))) == key],
                                   start, step, count) for key in folded}
    bands = speed_bands(histograms, start, step, count) if histograms else []
    short = any((row.get("band_includes_short_replies") or 0) for row in rows)
    idle_workers = []
    if llamacpp:
        idle_workers = [{"key": node["key"], "name": node["name"], "kind": "node", "values": [],
                         "state": VALUE_NOT_SERVING, "reason": SERVES_NOTHING_SINGLE}
                        for node in inputs.get("nodes") or []
                        if node.get("role") != "controller" and node.get("key") not in folded]

    quiet = stopped_buckets(inputs.get("transitions") or [], start, step, count)

    def lines(metric: str, how: Optional[str]) -> List[Dict[str, Any]]:
        own = serving_series(folded, names, count, metric, quiet)
        if metric == "decode" and histograms:
            for line in own:
                if any(band is not None for band in node_bands.get(line["key"], [])):
                    line["bands"] = node_bands[line["key"]]
        if split or how is None or len(folded) < 2:
            return own + idle_workers
        return [cluster_series(folded, count, metric, how, names)] + own + idle_workers

    decode = lines("decode", "pooled")
    clustered = bool(decode) and decode[0]["key"] == "cluster"
    if clustered:
        decode[0]["bands"] = bands
    own_bands = any(line.get("bands") for line in decode)
    panels = {
        "output_throughput": _panel("output_throughput", lines("throughput", "sum"),
                                    caption=LLAMACPP_AT_COMPLETION if any_llamacpp else ""),
        "decode_speed": _panel("decode_speed", decode, bands=None if clustered or own_bands else bands,
                               bands_note=LLAMACPP_NO_BANDS if llamacpp else (SHORT_REPLIES_INCLUDED if short else "")),
        "kv_cache": (_panel("kv_cache", [], state=VALUE_NOT_MEASURED, reason=LLAMACPP_NO_KV) if llamacpp
                     else _panel("kv_cache", lines("kv", "max"))),
    }
    computed = serving_series(folded, names, count, "computed", quiet)
    cached = [] if llamacpp else serving_series(folded, names, count, "cached", quiet)
    cached_sum, cached_why = _sum_lines(cached, count)
    computed_sum, computed_why = _sum_lines(computed, count)
    layers = [
        {"key": "cached", "name": LAYER_CACHED, "kind": "layer",
         "values": [] if llamacpp else cached_sum, "bucket_reasons": [] if llamacpp else cached_why,
         "state": VALUE_NOT_MEASURED if llamacpp else VALUE_MEASURED,
         "reason": LLAMACPP_NO_CACHE_FIGURES if llamacpp else ""},
        {"key": "computed", "name": LAYER_PROCESSED if llamacpp else LAYER_COMPUTED, "kind": "layer",
         "values": computed_sum, "bucket_reasons": computed_why, "state": VALUE_MEASURED, "reason": ""},
    ]
    panels["prompt_tokens"] = _panel("prompt_tokens", layers, by_node={"cached": cached, "computed": computed})
    if llamacpp:
        # llama.cpp reports no reuse, so the panel is not a comparison (S0).
        panels["prompt_tokens"]["title"] = LAYER_PROCESSED
    return panels


def _sum_lines(series: Sequence[Mapping[str, Any]], count: int) -> Tuple[List[Optional[float]], List[str]]:
    """A layer's "all machines" line and its reasons: null where an expected machine is under-covered, named."""
    totals: List[Optional[float]] = []
    reasons: List[str] = []
    for index in range(count):
        parts, short = [], []
        for line in series:
            share = (line.get("coverage") or [None] * count)[index]
            value = line["values"][index]
            if share is not None and share < MIN_BUCKET_COVERAGE:
                short.append(str(line.get("name") or "") or REMOVED_MACHINE)
            if share is not None:
                parts.append(value)
        totals.append(None if short or not parts or any(part is None for part in parts) else sum(parts))
        # A total left out because a machine was read too little says which one (pass-4 renders).
        reasons.append(UNDER_COVERED_NAMED.format(names=", ".join(short)) if short else "")
    return totals, reasons


def _host_line(host: Mapping[int, Mapping[str, Any]], field: str, start: int, step: int, count: int) -> Dict[str, Any]:
    values, coverage = [], []
    for index in range(count):
        row = host.get(start + index * step) or {}
        values.append(_number(row.get(field)))
        coverage.append(min(1.0, field_samples(row, field) / step) if row else None)
    return {"values": values, "coverage": coverage}


def _history_nodes(inputs: Mapping[str, Any], host: Mapping[str, Any]) -> List[Dict[str, Any]]:
    """The fleet, then every machine with history in the range that has left it.

    A removed machine keeps its lines under a name that is not its id, in the
    colour it had (its slot is held until its history ages out).
    """
    nodes = [dict(node) for node in inputs.get("nodes") or []]
    fleet = {str(node.get("key")) for node in nodes}
    held = inputs.get("held_slots") or {}
    for key in host:
        if key not in fleet:
            nodes.append({"key": key, "name": REMOVED_MACHINE, "role": "", "slot": held.get(key), "row": {},
                          "limits": {}, "policy": {}, "class_known": True})
    return nodes


def _host_panels(inputs: Mapping[str, Any], start: int, step: int, count: int) -> Dict[str, Dict[str, Any]]:
    names = ("gpu_temperature", "gpu_power", "host_cpu_memory", "gpu_busy",
             "package_power", "gtt_used", "gpu_edge_temperature")
    error = str(inputs.get("host_reason") or "")
    if inputs.get("host") is None:
        shaped = {name: _panel(name, [], state=VALUE_UNKNOWN, reason=error) for name in names}
        shaped["gpu_temperature"]["title"] = gpu_temperature_title([])
        return shaped
    host = inputs.get("host") or {}
    panels: Dict[str, List[Dict[str, Any]]] = {name: [] for name in names}
    band_sets, sensors = [], []
    for node in _history_nodes(inputs, host):
        key, name = str(node.get("key")), str(node.get("name") or "")
        buckets = host.get(key)
        if not buckets:
            absent = {"key": key, "name": name, "kind": "node", "values": [], "state": VALUE_NOT_REPORTING,
                      "reason": HOST_NOT_IN_RANGE}
            for panel in names:
                panels[panel].append(dict(absent))
            continue
        base = {"key": key, "name": name, "kind": "node", "slot": node.get("slot"), "state": VALUE_MEASURED, "reason": ""}
        reading = dict(node.get("row") or {})
        if node.get("temperature_label"):
            reading.setdefault("gpu_temperature_label", node.get("temperature_label"))
        _value, sensor = gpu_temperature_reading(reading)
        if sensor is None:
            sensor = SENSOR_GRAPHICS_ENGINE if any(r.get("gpu_gfx_temperature_c") is not None for r in buckets.values()) else None
        field = "gpu_gfx_temperature_c" if sensor == SENSOR_GRAPHICS_ENGINE else "gpu_temperature_c"
        block = gpu_temperature_block(reading, node.get("limits"), node.get("policy") or None,
                                      class_known=node.get("class_known", True) is not False)
        band_sets.append(block["bands"])
        sensors.append(sensor)
        panels["gpu_temperature"].append({**base, **_host_line(buckets, field, start, step, count), "sensor": sensor})
        measures, why = measures_gpu_power(node)
        if measures:
            panels["gpu_power"].append({**base, **_host_line(buckets, "gpu_power_watts", start, step, count)})
        else:
            # A machine whose vendor tool says it cannot read GPU power has no
            # line and says why (pass-3 review), as T5 does.
            panels["gpu_power"].append({**base, "values": [], "state": VALUE_NOT_MEASURED, "reason": why})
        panels["gpu_busy"].append({**base, **_host_line(buckets, "gpu_busy_percent", start, step, count)})
        panels["host_cpu_memory"].append({**base, "key": key + ":cpu", "name": CPU_LINE.format(name=name), "metric": "cpu",
                                          **_host_line(buckets, "cpu_percent", start, step, count)})
        panels["host_cpu_memory"].append({**base, "key": key + ":memory", "name": MEMORY_LINE.format(name=name),
                                          "metric": "memory", **_host_line(buckets, "memory_percent", start, step, count)})
        package = "cpu_package_power_watts" if node.get("role") == "controller" else "gpu_socket_power_watts"
        panels["package_power"].append({**base, **_host_line(buckets, package, start, step, count)})
        panels["gtt_used"].append({**base, **_host_line(buckets, "gpu_gtt_used_bytes", start, step, count)})
        panels["gpu_edge_temperature"].append({**base, **_host_line(buckets, "gpu_temperature_c", start, step, count)})
    shaped = {name: _panel(name, lines, reason=error) for name, lines in panels.items()}
    common = band_sets[0] if band_sets and all(bands == band_sets[0] for bands in band_sets) else None
    shaped["gpu_temperature"]["title"] = gpu_temperature_title(sensors)
    shaped["gpu_temperature"]["thresholds"] = labelled_bands(common)
    shaped["gpu_temperature"]["bands_by_node"] = None if common is not None else [labelled_bands(b) for b in band_sets]
    power_lines = [line for line in panels["gpu_power"] if line.get("values")]
    if len(power_lines) > 1:
        total = _host_total(power_lines, count)
        total["off_label"] = OFF_UNTIL_CHOSEN.format(name=total["name"])
        shaped["gpu_power"]["series"] = [total] + shaped["gpu_power"]["series"]
        shaped["gpu_power"]["hidden_by_default"] = ["cluster"]
    return shaped


def _host_total(lines: Sequence[Mapping[str, Any]], count: int) -> Dict[str, Any]:
    values, partial, reasons = [], [], []
    for index in range(count):
        short = [line for line in lines if (line["coverage"][index] or 0.0) < MIN_BUCKET_COVERAGE]
        parts = [line["values"][index] for line in lines]
        partial.append(bool(short))
        reasons.append(UNDER_COVERED_NAMED.format(names=", ".join(str(line.get("name") or "") for line in short))
                       if short else "")
        values.append(None if short or any(part is None for part in parts) else sum(parts))
    return {"key": "cluster", "name": AGGREGATE_NAMES["sum"], "kind": "aggregate", "aggregation": "sum",
            "values": values, "partial": partial, "bucket_reasons": reasons, "state": VALUE_MEASURED, "reason": ""}


def _tile_words(tiles: Dict[str, Dict[str, Any]], scope: str, since: Optional[int]) -> None:
    """Each range tile's title and caption; a serving tile's caption may carry ``{since}``."""
    for key, tile in tiles.items():
        tile["title"] = TILE_TITLES[key]
        if key == "avg_gpu_power":
            tile["caption"] = tile_caption(key, str(tile.get("scope") or ""))
            continue
        lead = scope
        if key == "slowest_machine" and tile.get("cluster"):
            lead = "{} · {}".format(tile["cluster"].get("name") or "", scope)
        extra = str(tile.get("caption") or tile.get("note") or "")
        tile["caption"] = tile_caption(key, "{}. {}".format(lead, extra) if extra else lead)
        tile["scope_since"] = since


def build_range(inputs: Mapping[str, Any]) -> Dict[str, Any]:
    """The range layer (§3.2) from `performance_dashboard_source.gather_range`'s inputs."""
    start, step, count = int(inputs["start"]), int(inputs["step"]), int(inputs["count"])
    engine = engine_block(inputs.get("serving") or {})
    rows = list(inputs.get("replica_rows") or [])
    node_names = {str(node.get("key")): str(node.get("name") or "") for node in inputs.get("nodes") or []}
    names = serving_names(rows, node_names)
    model = current_model(rows)
    current = [row for row in rows if model and row.get("model") == model["model"]]
    folded = fold_rows(rows, start, step, count)
    scope, since = tile_scope(rows, model, int(inputs["range_seconds"]))
    host = inputs.get("host")
    tiles = serving_tiles(current, names, engine, scope)
    tiles["avg_gpu_power"] = (
        avg_gpu_power_tile(host, inputs.get("nodes") or [], start, step, count, int(inputs["range_seconds"]))
        if host is not None else {"state": VALUE_UNKNOWN, "unit": "watts", "cluster": None, "nodes": [],
                                  "excluded": [], "reason": str(inputs.get("host_reason") or ""), "scope": ""}
    )
    _tile_words(tiles, scope, since)
    panels = _serving_panels(inputs, engine, names, folded, start, step, count)
    panels.update(_host_panels(inputs, start, step, count))
    return annotate({
        "range": inputs.get("range"), "range_seconds": int(inputs["range_seconds"]),
        "start": start, "step": step, "count": count, "generated_at": inputs.get("now"),
        "current_model": model,
        "engine": {"kind": engine["kind"], "placement": engine["placement"], "routing": engine["routing"],
                   "scope_note": engine["scope_note"]},
        "events": events(rows, start, step, count, names, inputs.get("transitions") or []),
        "tiles": tiles, "panels": panels,
        "host_source": inputs.get("host_source") or "", "host_note": str(inputs.get("host_note") or ""),
        **range_labels(inputs.get("range")),
    })
