"""The Performance dashboard's words for what it measures, sent with the payload (VD-147 S4/S6).

Every word that says something ABOUT the readings - a panel's or tile's title,
a state's label and tone, a threshold mark's label, a range's name, a unit's
suffix, a line that is off until chosen - comes from here (pass-3 review), so
a rename is one change in one place and the screen cannot describe a reading
in words the backend did not choose. The screen's own controls and landmarks
("Range", "Reload", "Show as table", region names) are the frontend's chrome
and are not sent; the frontend guard `dashboardHoldsNoWord.test.ts` holds the
state, freshness and routing words out of the dashboard's modules.

:func:`annotate` is the last pass over a finished layer: every value-state
block gets its ``state_label`` and ``tone``. A freshness block already carries
its own (``performance_dashboard.freshness_projection``), and the engine block
is left alone - its badge has its own tone.
"""

from __future__ import annotations

from typing import Any, Dict, List, Mapping, Optional

from .performance_dashboard import (
    ENGINE_STATE_SERVING, ENGINE_STATE_STALE, FRESH_NOT_SERVING, FRESH_UNKNOWN, FRESHNESS_LABELS,
    TONE_DANGER, TONE_INFO, TONE_NEUTRAL, TONE_SUCCESS, TONE_WARNING, VALUE_IDLE, VALUE_MEASURED, VALUE_NOT_MEASURED, VALUE_NOT_REPORTED, VALUE_NOT_REPORTING,
    VALUE_NOT_SERVING, VALUE_PARTIAL, VALUE_UNKNOWN, offered_ranges,
)
from .answer_evidence import GPU_POWER
from .health_evaluation import GPU_SENSOR_NOUNS
from .performance_serving import CLUSTER_LOADING, CLUSTER_NONE, CLUSTER_UNKNOWN, CLUSTER_UNLOADED
from .platforms.gpu_temperature import SENSOR_EDGE
from .telemetry_store import NOT_REPORTING_LABEL

#: A value state's label and tone.
VALUE_LABELS: Dict[str, str] = {
    VALUE_MEASURED: "Measured",
    VALUE_PARTIAL: "Partly measured",
    VALUE_IDLE: "Nothing to show",
    VALUE_NOT_REPORTED: "Not reported",
    VALUE_NOT_MEASURED: "Not measured",
    VALUE_NOT_REPORTING: NOT_REPORTING_LABEL,
    VALUE_NOT_SERVING: FRESHNESS_LABELS[FRESH_NOT_SERVING],
    VALUE_UNKNOWN: FRESHNESS_LABELS[FRESH_UNKNOWN],
}
VALUE_TONES: Dict[str, str] = {
    VALUE_MEASURED: TONE_SUCCESS, VALUE_PARTIAL: TONE_WARNING, VALUE_IDLE: TONE_NEUTRAL,
    VALUE_NOT_REPORTED: TONE_NEUTRAL, VALUE_NOT_MEASURED: TONE_NEUTRAL,
    VALUE_NOT_REPORTING: TONE_DANGER, VALUE_NOT_SERVING: TONE_NEUTRAL, VALUE_UNKNOWN: TONE_WARNING,
}

#: The status strip badge's tone for each engine state.
BADGE_TONES: Dict[str, str] = {
    ENGINE_STATE_SERVING: TONE_SUCCESS, ENGINE_STATE_STALE: TONE_WARNING,
    CLUSTER_UNLOADED: TONE_NEUTRAL, CLUSTER_LOADING: TONE_INFO,
    CLUSTER_NONE: TONE_NEUTRAL, CLUSTER_UNKNOWN: TONE_WARNING,
}

#: Each panel's short title, as the Performance boards print it (VD-200). What
#: the reading is of - a stream, the graphics engine, the whole chip - moved to
#: its scope line below, so a shorter title drops no fact.
PANEL_TITLES: Dict[str, str] = {
    "output_throughput": "Output throughput",
    "prompt_tokens": "Prompt tokens",
    "decode_speed": "Decode speed",
    "kv_cache": "Peak KV cache usage",
    "gpu_power": GPU_POWER,
    "host_cpu_memory": "Host CPU and memory",
    "gpu_busy": "GPU busy",
    "package_power": "Package power",
    "gtt_used": "Unified GPU memory in use",
    "gpu_edge_temperature": GPU_SENSOR_NOUNS[SENSOR_EDGE],
}
#: GPU temperature's title keeps its sensor (LESSONS 5: a temperature is named
#: by the sensor that read it), so it is longer than the board's.
GPU_TEMPERATURE_TITLE = "GPU temperature ({sensor})"
GPU_TEMPERATURE_TITLE_MIXED = "GPU temperature (per machine's sensor)"

#: Each panel's scope line under its title (the boards' "Per stream · tok/s"):
#: what the lines measure. The screen adds the unit; a panel with no scope here
#: shows the range instead.
PANEL_SCOPES: Dict[str, str] = {
    "output_throughput": "All copies together",
    "prompt_tokens": "Reused from cache vs computed",
    "decode_speed": "Per stream",
    "kv_cache": "Share of the cache in use",
    "gpu_power": "Graphics engine",
    "host_cpu_memory": "CPU solid, memory dashed",
    "gpu_busy": "Share of the graphics processor in use",
    "package_power": "Whole chip",
    "gtt_used": "GTT",
}

#: Each tile's title (spec §1.1), short as the boards print it.
TILE_TITLES: Dict[str, str] = {
    "decode_speed_mean": PANEL_TITLES["decode_speed"],
    "slowest_machine": "Slowest machine",
    "prefix_cache_hits": "Prefix cache hits",
    "hottest_gpu": "Hottest GPU",
    "avg_gpu_power": "Avg GPU power",
    "busiest_cpu": "Busiest host CPU",
}

#: What a tile's short title leaves out, said first in its caption.
TILE_QUALIFIERS: Dict[str, str] = {
    "decode_speed_mean": "Per stream, mean",
    "avg_gpu_power": PANEL_SCOPES["gpu_power"],
}

#: What each offered range is called: the chooser's options and the captions' lead.
RANGE_LABELS: Dict[str, str] = {
    "15m": "Last 15 minutes", "1h": "Last 1 hour", "6h": "Last 6 hours",
    "24h": "Last 24 hours", "3d": "Last 3 days", "7d": "Last 7 days",
}

#: The suffix the screen writes after a number in each base unit.
UNIT_SUFFIXES: Dict[str, str] = {
    "tokens_per_second": "tok/s", "celsius": "°C", "watts": "W", "ratio": "%", "percent": "%", "bytes": "",
}

#: A temperature mark's words.
MARK_LABEL = "{level} {value} °C"

#: The now-layer tiles' captions.
HOTTEST_CAPTION = "{name}, {sensor}, now"

#: A line that is off until chosen (the GPU power panel's cluster total): the
#: toggle says so rather than looking struck out (pass-4 review).
OFF_UNTIL_CHOSEN = "{name} (off - select to show)"
BUSIEST_CAPTION = "{name}, now"
NOW_CAPTION = "now"


def sensor_words(sensor: Optional[str]) -> str:
    """A sensor as a phrase: "graphics engine sensor", "edge sensor", "unlabelled sensor" (never "sensor sensor")."""
    text = str(sensor or "")
    return text if not text or text.endswith("sensor") else text + " sensor"


def gpu_temperature_title(sensors: List[Optional[str]]) -> str:
    """The panel's title: one sensor named, or "per machine's sensor" when they differ."""
    named = {sensor for sensor in sensors if sensor}
    if len(named) == 1:
        return GPU_TEMPERATURE_TITLE.format(sensor=sensor_words(named.pop()))
    return GPU_TEMPERATURE_TITLE_MIXED


def tile_caption(key: str, caption: str) -> str:
    """A tile's caption, led by what its short title leaves out ("Per stream, mean · ...")."""
    qualifier = TILE_QUALIFIERS.get(key, "")
    return " · ".join(part for part in (qualifier, caption) if part)


def _degrees(value: float) -> str:
    return ("%.1f" % value).rstrip("0").rstrip(".")


def labelled_bands(bands: Optional[Mapping[str, Any]]) -> Optional[Dict[str, Any]]:
    """A band set with the words for its marks: ``marks`` [{value, label}] and ``labels``."""
    if not isinstance(bands, Mapping):
        return None
    shaped = dict(bands)
    marks, labels = [], {}
    for level, key in (("warning", "warning_c"), ("critical", "critical_c")):
        value = bands.get(key)
        if isinstance(value, (int, float)) and not isinstance(value, bool):
            labels[level] = MARK_LABEL.format(level=level, value=_degrees(float(value)))
            marks.append({"value": float(value), "label": labels[level]})
    shaped["marks"], shaped["labels"] = marks, labels
    return shaped


def range_labels(range_key: Optional[str]) -> Dict[str, Any]:
    """The range layer's words: the chooser's options, the unit suffixes, and this range's lead."""
    options = [{"value": key, "label": RANGE_LABELS.get(key, key)} for key in offered_ranges()]
    return {
        "caption_lead": RANGE_LABELS.get(str(range_key or ""), ""),
        "labels": {"range_options": options, "units": dict(UNIT_SUFFIXES)},
    }


def now_tile_words(tiles: Dict[str, Dict[str, Any]]) -> None:
    """The now layer's two tiles: titles, captions and their marks' words, in place."""
    for key, tile in tiles.items():
        tile["title"] = TILE_TITLES[key]
        cluster = tile.get("cluster") or {}
        if key == "hottest_gpu":
            if cluster:
                cluster["bands"] = labelled_bands(cluster.get("bands"))
                tile["caption"] = HOTTEST_CAPTION.format(name=cluster.get("name") or "",
                                                         sensor=sensor_words(cluster.get("sensor")))
            for node in tile.get("nodes") or []:
                node["bands"] = labelled_bands(node.get("bands"))
        elif cluster:
            tile["caption"] = BUSIEST_CAPTION.format(name=cluster.get("name") or "")
        tile.setdefault("caption", NOW_CAPTION)


def annotate(value: Any, parent: str = "") -> Any:
    """Give every value-state block its label and tone, in place; returns ``value``."""
    if isinstance(value, dict):
        state = value.get("state")
        if parent != "engine" and "label" not in value and state in VALUE_LABELS:
            value.setdefault("state_label", VALUE_LABELS[state])
            value.setdefault("tone", VALUE_TONES[state])
        for key, item in value.items():
            if key != "engine":
                annotate(item, key)
    elif isinstance(value, list):
        for item in value:
            annotate(item, parent)
    return value
