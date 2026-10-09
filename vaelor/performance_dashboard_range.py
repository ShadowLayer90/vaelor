"""The Performance dashboard's range layer: serving panels, host panels, tiles, events (VD-147 S4).

Pure functions of what `performance_dashboard_source.gather_range` read:

* the usage ledger's per-replica 20 s buckets and speed-band rows
  (`model_usage.ModelUsageLedger.replica_buckets` / ``replica_histograms``),
  folded into the range's step;
* the host buckets `telemetry_reader.TelemetryReader.host_buckets` returned.

Aggregation follows spec §1.5 and is `serving_buckets`' wherever it can be:
a SUM across machines (output throughput, prompt tokens, GPU power) is
``None`` in a bucket where an expected machine was read for less than
`serving_buckets.MIN_BUCKET_COVERAGE` of it, flagged partial; a POOLED rate or a
maximum is computed over what was read and flagged partial. A machine's own
line is drawn over its covered time with its coverage beside it.

A model split across machines (``distributed``) has ONE engine: its serving
panels draw a single "Whole model (read on …)" line and no per-machine split.
A replicated deployment routes each conversation to one machine (sticky
routing), so its machines' lines differ and the slowest-machine tile names the
one that decoded too little to compare.
"""

from __future__ import annotations

from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence, Tuple

from .gpu_vendor_status import CODES_WITH_READINGS, STATUS_FIELD, known_code, status_sentence
from .performance_dashboard import VALUE_IDLE, VALUE_MEASURED, VALUE_NOT_MEASURED, VALUE_PARTIAL
from .performance_serving import PLACEMENT_SPLIT
from .serving_buckets import (
    BANDS_CHANGED, MIN_BUCKET_COVERAGE, band_tokens_per_second, bucket_state, histogram_band, merge_bins,
)
from .model_usage import ENGINE_VLLM
from .performance_marker_lines import engine_near, line_name
from .serving_metrics import FAMILY_PER_REQUEST
from .telemetry_reader import SAMPLES_KEY

#: A machine must have decoded at least this many inter-token intervals in the
#: range to count for "Slowest machine": about 15 s of decoding at this
#: hardware's ceiling, so one short reply cannot make a machine "slowest".
MIN_T2_INTERVALS = 200

#: The key of the one serving line a split model draws. Internal, never rendered.
WHOLE_MODEL_KEY = "whole-model"

#: The owner sentences this layer adds. Every other reason is an owner's constant.
WHOLE_MODEL_NAME = "Whole model (read on {lead})"
WHOLE_MODEL_NAME_UNNAMED = "Whole model"
#: Said once for every machine left out of "Slowest machine" (ACC-206): the
#: tile still lists their speeds, so the sentence says they are not ranked,
#: never that they have no speed.
DECODED_TOO_LITTLE = "Not ranked, too little decoded in this range: {names}."
ONLY_ONE_MACHINE_SERVING = "Only one machine is serving; this equals the mean."
ONLY_ONE_ENGINE = "Only one engine serves this model; this equals the mean."
NOTHING_DECODED = "Nothing was decoded in this range, so there is no speed to show."
NO_PROMPTS_LOOKED_UP = "No prompts were looked up in this range."
LLAMACPP_NO_CACHE_FIGURES = "llama.cpp's metrics do not report prompt-cache reuse."
LLAMACPP_NO_KV = "This llama.cpp build does not report KV-cache occupancy."
LLAMACPP_NO_BANDS = "llama.cpp reports no per-request distribution."
LLAMACPP_AT_COMPLETION = "llama.cpp adds a reply's tokens when it finishes."
NO_SERVING_IN_RANGE = "Nothing was served in this range."
NO_GPU_POWER_MEASURED = "No machine measured its GPU's power in this range."
GPU_POWER_TOTAL_OF = "average total of {names}"
#: The same scope when no machine is left out (W5 review, fold slack): the list
#: beneath the tile names every machine, so the scope counts them instead of
#: running four long names into six lines above the first chart row.
GPU_POWER_TOTAL_OF_BOTH = "average total of both machines"
GPU_POWER_TOTAL_OF_ALL = "average total of all {count} machines"
GPU_POWER_COVERED = "{covered} of {range} with every measuring machine reporting"
HOST_NOT_IN_RANGE = "Vaelor holds no reading from this machine in this range."
#: A machine the average GPU power tile leaves out, named (ACC-207): a tile
#: shows several machines, so "this machine" there names none of them.
GPU_POWER_NOT_IN_RANGE = "{name} reported no GPU power in this range."
#: A serving bucket with no row while the replica was, as far as the record
#: shows, still a target: a slow tick, a locked ledger or an unreadable record.
#: It is a gap in what was recorded, never "stopped serving" (pass-4 review B1).
NO_RECORD_IN_INTERVAL = "Vaelor recorded nothing for this machine in this interval."
SERVES_NOTHING_SINGLE = "Serves nothing in single-machine mode."
SHORT_REPLIES_INCLUDED = "includes very short replies"
#: The two layers of the prompt-tokens panel. Cached tokens skip prefill but
#: still occupy KV memory and attention, so they are not called "free".
LAYER_CACHED = "Reused from cache (no prefill compute)"
LAYER_COMPUTED = "Computed (prefill)"
#: llama.cpp's one layer (S0): its prompt counter excludes nothing it could
#: call reused, so the layer says what was processed, not what was computed.
LAYER_PROCESSED = "Prompt tokens processed"

#: The cluster line's name for each way it combines the machines.
AGGREGATE_NAMES = {"pooled": "Cluster (pooled)", "sum": "Cluster total", "max": "Cluster (highest)"}
#: Why a cluster total is missing in an interval: the machine it would leave out.
UNDER_COVERED_NAMED = "{names} read for too little of this interval to add up."
#: Why T5 has no figure: the machines that were never all covered together.
GPU_POWER_NEVER_COVERED = "{names} did not report for enough of any one interval to total their power."

#: A removed machine's name in history: its id is never shown.
REMOVED_MACHINE = "A removed machine"

#: The scope a range tile states: the model, and the span - or, when the range
#: holds another model, when the current one began (spec §1.6).
TILE_SCOPE = "{model} · over the last {span}"
TILE_SCOPE_NO_MODEL = "over the last {span}"
TILE_SCOPE_SINCE = "{model} · since {{since}}"

EVENT_RESTART = "Engine restarted"
EVENT_STARTED = "Started serving"
EVENT_STOPPED = "Stopped serving"
EVENT_MODEL = "Now serving {model}"
EVENT_TO_CLUSTER = "Switched to cluster serving"
EVENT_TO_SINGLE = "Switched to single-machine serving"
#: Every marker names what it happened on (ACC-200): a machine, or the whole
#: model a split serves. Two machines stopping together are two events, each
#: named, never one label drawn twice.
EVENT_ON = "{event} ({machine})"
EVENT_WHOLE_MODEL = "whole model"

#: The per-bucket reason for a figure llama.cpp does not report.
_NOT_FROM_LLAMACPP = {"kv": LLAMACPP_NO_KV, "cached": LLAMACPP_NO_CACHE_FIGURES}

_SUM_COLUMNS = (
    "attempted_seconds", "covered_seconds", "reset_ticks", "gap_ticks",
    "prompt_tokens", "completion_tokens", "requests", "itl_seconds", "itl_intervals",
    "prefix_hits", "prefix_queries", "preemptions", "prompt_computed", "prompt_cached", "prompt_seconds",
)
_MAX_COLUMNS = ("kv_max", "running_max", "waiting_max", "models_seen", "band_includes_short_replies")


def _number(value: Any) -> Optional[float]:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        return None
    try:
        number = float(value)
    except (OverflowError, ValueError):
        return None
    return number if number == number and abs(number) != float("inf") else None


def _ratio(top: Optional[float], bottom: Optional[float]) -> Optional[float]:
    return top / bottom if top is not None and bottom and bottom > 0 else None


#: Removed machines in a list are counted, never repeated (review 2).
REMOVED_MACHINES = "{count} removed machines"


def names_words(names: Sequence[str]) -> str:
    """Names as a sentence lists them: "A", "A and B", "A, B and 2 removed machines"."""
    named = [name for name in names if name and name != REMOVED_MACHINE]
    removed = sum(1 for name in names if name == REMOVED_MACHINE)
    if removed:
        named.append(REMOVED_MACHINES.format(count=removed) if removed > 1
                     else REMOVED_MACHINE if not named else REMOVED_MACHINE[:1].lower() + REMOVED_MACHINE[1:])
    return named[0] if len(named) == 1 else "{} and {}".format(", ".join(named[:-1]), named[-1]) if named else ""


def duration_words(seconds: float) -> str:
    """A span as the screen reads it: "57 min", "5 h 40 min", "3 days"."""
    seconds = int(max(0, seconds))
    if seconds >= 86400 and seconds % 86400 == 0:
        days = seconds // 86400
        return "{} day{}".format(days, "" if days == 1 else "s")
    if seconds < 60:
        return "{} s".format(seconds)
    hours, minutes = divmod(seconds // 60, 60)
    if hours and minutes:
        return "{} h {} min".format(hours, minutes)
    if hours:
        return "{} h".format(hours)
    return "{} min".format(minutes)


# --------------------------------------------------------------------------
# Serving rows folded into the range's buckets
# --------------------------------------------------------------------------


def series_key(row: Mapping[str, Any]) -> str:
    """Which serving line a ledger row, or a recorded start or stop, belongs to.

    Its machine, or the whole split model. One rule for both: a marker is
    named from the line's own buckets (B9), so the two must key alike - they
    were once two functions with the same body (LESSONS 6).
    """
    return WHOLE_MODEL_KEY if row.get("placement") == PLACEMENT_SPLIT else str(row.get("node_id") or "")


def fold_rows(
    rows: Iterable[Mapping[str, Any]], start: int, step: int, count: int,
) -> Dict[str, Dict[int, Dict[str, Any]]]:
    """``{series key: {bucket index: merged row}}`` for the ledger rows in the range."""
    folded: Dict[str, Dict[int, Dict[str, Any]]] = {}
    for row in rows:
        index = (int(row.get("bucket") or 0) - start) // step
        if not 0 <= index < count:
            continue
        slot = folded.setdefault(series_key(row), {})
        merged = slot.get(index)
        if merged is None:
            slot[index] = dict(row)
            slot[index]["_models"] = {str(row.get("model") or "")}
            slot[index]["_engines"] = {str(row.get("engine") or "")}
            continue
        for column in _SUM_COLUMNS:
            a, b = _number(merged.get(column)), _number(row.get(column))
            merged[column] = None if a is None and b is None else (a or 0.0) + (b or 0.0)
        for column in _MAX_COLUMNS:
            values = [value for value in (_number(merged.get(column)), _number(row.get(column))) if value is not None]
            merged[column] = max(values) if values else None
        merged["_models"].add(str(row.get("model") or ""))
        merged["_engines"].add(str(row.get("engine") or ""))
        merged["models_seen"] = max(int(merged.get("models_seen") or 1), len(merged["_models"]))
        merged["node_name"] = row.get("node_name") or merged.get("node_name")
    return folded


def _coverage(row: Optional[Mapping[str, Any]]) -> Optional[float]:
    if not row:
        return None
    attempted = _number(row.get("attempted_seconds")) or 0.0
    return None if attempted <= 0 else min(1.0, (_number(row.get("covered_seconds")) or 0.0) / attempted)


def _node_value(row: Optional[Mapping[str, Any]], metric: str) -> Optional[float]:
    """One machine's figure in one bucket; ``None`` for a bucket with nothing to divide."""
    if not row or bucket_state(row)[0] not in ("measured", "partial"):
        return None
    covered = _number(row.get("covered_seconds"))
    if metric == "throughput":
        return _ratio(_number(row.get("completion_tokens")), covered)
    if metric == "decode":
        return _ratio(_number(row.get("itl_intervals")), _number(row.get("itl_seconds")))
    if metric == "cached":
        cached = _number(row.get("prompt_cached"))
        return _ratio(cached if cached is not None else _number(row.get("prefix_hits")), covered)
    if metric == "computed":
        computed = _number(row.get("prompt_computed"))
        if computed is None:
            queries, hits = _number(row.get("prefix_queries")), _number(row.get("prefix_hits"))
            computed = queries - hits if queries is not None and hits is not None else _number(row.get("prompt_tokens"))
        return _ratio(computed, covered)
    if metric == "kv":
        return _number(row.get("kv_max"))
    raise ValueError(metric)


def _under_covered(rows: Mapping[int, Mapping[str, Any]], index: int) -> bool:
    share = _coverage(rows.get(index))
    return share is not None and share < MIN_BUCKET_COVERAGE


def serving_series(
    folded: Mapping[str, Mapping[int, Mapping[str, Any]]], names: Mapping[str, str], count: int, metric: str,
    not_serving: Optional[Mapping[str, Iterable[int]]] = None,
) -> List[Dict[str, Any]]:
    """Each machine's (or the whole model's) line for one serving metric.

    A bucket with no row says nothing when the record shows the replica was
    not a target then (``not_serving``, from `stopped_buckets`) or when it is
    the bucket still filling; otherwise it is :data:`NO_RECORD_IN_INTERVAL`.
    A llama.cpp bucket of a metric llama.cpp does not report says so, bucket by
    bucket, so a range that switched engines is judged where it switched.
    """
    series = []
    for key, rows in folded.items():
        values = [_node_value(rows.get(index), metric) for index in range(count)]
        coverage = [_coverage(rows.get(index)) for index in range(count)]
        quiet = set((not_serving or {}).get(key) or ())
        states = [_bucket_reason(rows.get(index), metric, index in quiet or index == count - 1)
                  for index in range(count)]
        series.append({
            "key": key, "name": names.get(key, ""), "kind": "aggregate" if key == WHOLE_MODEL_KEY else "node",
            "values": values, "coverage": coverage,
            "state": VALUE_MEASURED if any(value is not None for value in values) else VALUE_IDLE,
            "reason": "", "bucket_reasons": [reason for _state, reason in states],
        })
    return series


def _bucket_reason(row: Optional[Mapping[str, Any]], metric: str, quiet: bool) -> Tuple[str, str]:
    if row is None:
        return ("", "") if quiet else ("", NO_RECORD_IN_INTERVAL)
    if row.get("engine") == "llama.cpp" and metric in _NOT_FROM_LLAMACPP:
        return ("", _NOT_FROM_LLAMACPP[metric])
    return bucket_state(row)


def stopped_buckets(
    transitions: Sequence[Mapping[str, Any]], start: int, step: int, count: int,
) -> Dict[str, set]:
    """``{series key: {bucket index}}``: the buckets a replica was recorded as not serving.

    A bucket counts when the replica's last recorded start or stop at its
    beginning was a stop and it did not start again inside it. With no record
    at all nothing is claimed: the bucket is a gap, not a stop.
    """
    by_key: Dict[str, List[Tuple[float, str]]] = {}
    for row in transitions:
        by_key.setdefault(series_key(row), []).append((float(row.get("at") or 0.0), str(row.get("kind") or "")))
    quiet: Dict[str, set] = {}
    for key, marks in by_key.items():
        marks.sort()
        for index in range(count):
            begin, end = start + index * step, start + (index + 1) * step
            before = [kind for moment, kind in marks if moment <= begin]
            restarted = any(kind == "started" for moment, kind in marks if begin < moment < end)
            if before and before[-1] == "stopped" and not restarted:
                quiet.setdefault(key, set()).add(index)
    return quiet


def cluster_series(
    folded: Mapping[str, Mapping[int, Mapping[str, Any]]], count: int, metric: str, how: str,
    names: Optional[Mapping[str, str]] = None,
) -> Dict[str, Any]:
    """The cluster line: ``sum`` is null on under-coverage; ``pooled``/``max`` are flagged partial."""
    values: List[Optional[float]] = []
    partial: List[bool] = []
    reasons: List[str] = []
    for index in range(count):
        expected = {key: rows for key, rows in folded.items() if index in rows and _coverage(rows[index]) is not None}
        short_keys = [key for key, rows in expected.items() if _under_covered(rows, index)]
        short = bool(short_keys)
        partial.append(short)
        reasons.append(UNDER_COVERED_NAMED.format(names=", ".join((names or {}).get(key, "") or REMOVED_MACHINE
                                                                  for key in short_keys))
                       if short and how == "sum" else "")
        if not expected:
            values.append(None)
            continue
        if how == "sum":
            parts = [_node_value(rows.get(index), metric) for rows in expected.values()]
            values.append(None if short or any(part is None for part in parts) else sum(parts))
        elif how == "pooled":
            decoded = [rows for rows in expected.values() if _node_value(rows.get(index), "decode") is not None]
            intervals = sum(_number(rows[index].get("itl_intervals")) or 0.0 for rows in decoded)
            seconds = sum(_number(rows[index].get("itl_seconds")) or 0.0 for rows in decoded)
            values.append(_ratio(intervals, seconds))
        else:
            parts = [part for part in (_node_value(rows.get(index), metric) for rows in expected.values())
                     if part is not None]
            values.append(max(parts) if parts else None)
    return {"key": "cluster", "name": AGGREGATE_NAMES[how], "kind": "aggregate", "aggregation": how,
            "values": values, "partial": partial, "bucket_reasons": reasons,
            "state": VALUE_MEASURED if any(value is not None for value in values) else VALUE_IDLE, "reason": ""}


def speed_bands(histograms: Iterable[Mapping[str, Any]], start: int, step: int, count: int) -> List[Optional[Dict[str, Any]]]:
    """Per bucket, the typical and slowest-10 % request speed bands, in tokens per second (D3)."""
    grouped: Dict[int, List[Tuple[Any, Mapping[int, int]]]] = {}
    edges_of: Dict[Any, Sequence[float]] = {}
    for row in histograms:
        if row.get("family") != FAMILY_PER_REQUEST:
            continue
        index = (int(row.get("bucket") or 0) - start) // step
        if 0 <= index < count:
            grouped.setdefault(index, []).append((row.get("edge_set_id"), row.get("bins") or {}))
            edges_of[row.get("edge_set_id")] = row.get("edges") or []
    bands: List[Optional[Dict[str, Any]]] = []
    for index in range(count):
        edge_set, bins = merge_bins(grouped.get(index, []))
        if edge_set is None:
            bands.append({"changed": BANDS_CHANGED} if len(grouped.get(index, [])) > 1 else None)
            continue
        edges = edges_of[edge_set]
        bands.append({
            "p50": band_tokens_per_second(histogram_band(bins, edges, 0.5)),
            "p90": band_tokens_per_second(histogram_band(bins, edges, 0.9)),
        })
    return bands


# --------------------------------------------------------------------------
# Tiles T1, T2, T3 (serving) and T5 (host)
# --------------------------------------------------------------------------


def _pooled(rows: Iterable[Mapping[str, Any]], top: str, bottom: str) -> Tuple[Optional[float], float]:
    total = seconds = 0.0
    for row in rows:
        a, b = _number(row.get(top)), _number(row.get(bottom))
        if a is None or b is None or b <= 0 or bucket_state(row)[0] not in ("measured", "partial"):
            continue
        total += a
        seconds += b
    return (total / seconds if seconds > 0 else None), total


def serving_tiles(
    rows: Sequence[Mapping[str, Any]], names: Mapping[str, str], engine: Mapping[str, Any], scope: str,
) -> Dict[str, Dict[str, Any]]:
    """T1 decode speed per stream, T2 slowest machine, T3 prefix-cache hits (current model only)."""
    by_key: Dict[str, List[Mapping[str, Any]]] = {}
    for row in rows:
        by_key.setdefault(series_key(row), []).append(row)
    # Judged by the rows' own engine, not the one serving now (pass-3 review
    # S4-B4): vLLM history keeps its figures after a switch to llama.cpp.
    llamacpp = llama_only(rows)
    mean, _ = _pooled(rows, "itl_intervals", "itl_seconds")
    per_node = {key: _pooled(node_rows, "itl_intervals", "itl_seconds") for key, node_rows in by_key.items()}
    t1 = {
        "state": VALUE_MEASURED if mean is not None else VALUE_IDLE, "unit": "tokens_per_second",
        "aggregation": "pooled", "scope": scope,
        "cluster": {"value": mean, "partial": False},
        "nodes": [{"key": key, "name": names.get(key, ""), "value": value[0],
                   "state": VALUE_MEASURED if value[0] is not None else VALUE_IDLE} for key, value in per_node.items()],
        "reason": "" if mean is not None else NOTHING_DECODED,
        "caption": LLAMACPP_AT_COMPLETION if llamacpp else "",
    }
    qualifying = {key: value for key, value in per_node.items() if value[0] is not None and value[1] >= MIN_T2_INTERVALS}
    left_out = [names.get(key, "") or REMOVED_MACHINE for key in per_node if key not in qualifying]
    excluded = [DECODED_TOO_LITTLE.format(names=names_words(left_out))] if left_out else []
    if qualifying:
        slow_key = min(qualifying, key=lambda key: qualifying[key][0])
        note = ONLY_ONE_ENGINE if WHOLE_MODEL_KEY in by_key else ONLY_ONE_MACHINE_SERVING if len(by_key) == 1 else ""
        t2 = {"state": VALUE_MEASURED, "unit": "tokens_per_second", "scope": scope, "note": note,
              "cluster": {"value": qualifying[slow_key][0], "name": names.get(slow_key, "")},
              "nodes": t1["nodes"], "excluded": excluded, "reason": ""}
    else:
        # The one sentence is the reason, not also an exclusion: the screen
        # joins both, and said it once per machine and then again (ACC-206).
        t2 = {"state": VALUE_IDLE, "unit": "tokens_per_second", "scope": scope, "note": "",
              "cluster": None, "nodes": t1["nodes"], "excluded": [],
              "reason": excluded[0] if excluded else NOTHING_DECODED}
    if llamacpp:
        t3 = {"state": VALUE_NOT_MEASURED, "unit": "ratio", "cluster": None, "nodes": [], "scope": scope,
              "reason": LLAMACPP_NO_CACHE_FIGURES}
    else:
        ratio, _hits = _pooled(rows, "prefix_hits", "prefix_queries")
        t3 = {
            "state": VALUE_MEASURED if ratio is not None else VALUE_IDLE, "unit": "ratio", "scope": scope,
            "cluster": {"value": ratio, "partial": False},
            "nodes": [{"key": key, "name": names.get(key, ""),
                       "value": _pooled(node_rows, "prefix_hits", "prefix_queries")[0]} for key, node_rows in by_key.items()],
            "reason": "" if ratio is not None else NO_PROMPTS_LOOKED_UP,
        }
    return {"decode_speed_mean": t1, "slowest_machine": t2, "prefix_cache_hits": t3}


def measures_gpu_power(node: Mapping[str, Any]) -> Tuple[bool, str]:
    """Whether a machine measures its GPU's power, and the code's sentence when it does not."""
    code = known_code((node.get("row") or {}).get(STATUS_FIELD))
    if code is None or code in CODES_WITH_READINGS:
        return True, ""
    return False, status_sentence(code, str(node.get("name") or ""))


def avg_gpu_power_tile(
    host: Mapping[str, Mapping[int, Mapping[str, Any]]], nodes: Sequence[Mapping[str, Any]],
    start: int, step: int, count: int, range_seconds: int,
) -> Dict[str, Any]:
    """T5 (owner decision D2): the mean, over buckets where every measuring machine is covered, of their summed mean power."""
    measuring, excluded = [], []
    for node in nodes:
        ok, reason = measures_gpu_power(node)
        series = host.get(str(node.get("key")), {})
        if ok and any(_number(row.get("gpu_power_watts")) is not None for row in series.values()):
            measuring.append(node)
        else:
            excluded.append({"name": node.get("name"),
                             "reason": reason or GPU_POWER_NOT_IN_RANGE.format(name=node.get("name") or REMOVED_MACHINE)})
    if not measuring:
        return {"state": VALUE_NOT_MEASURED, "unit": "watts", "aggregation": "mean_of_bucket_sums",
                "cluster": None, "nodes": [], "excluded": excluded, "reason": NO_GPU_POWER_MEASURED, "scope": ""}
    totals, covered = [], 0
    short_of: Dict[str, int] = {}
    per_node: Dict[str, List[float]] = {str(node.get("key")): [] for node in measuring}
    for index in range(count):
        moment = start + index * step
        parts = []
        for node in measuring:
            row = host.get(str(node.get("key")), {}).get(moment)
            value = _number((row or {}).get("gpu_power_watts"))
            if value is not None:
                per_node[str(node.get("key"))].append(value)
            if value is None or field_samples(row, "gpu_power_watts") < MIN_BUCKET_COVERAGE * step:
                short_of[str(node.get("key"))] = short_of.get(str(node.get("key")), 0) + 1
                parts = None
                break
            parts.append(value)
        if parts is not None:
            totals.append(sum(parts))
            covered += 1
    names = ", ".join(str(node.get("name")) for node in measuring)
    if excluded or len(measuring) == 1:
        total_of = GPU_POWER_TOTAL_OF.format(names=names)
    else:
        total_of = GPU_POWER_TOTAL_OF_BOTH if len(measuring) == 2 else GPU_POWER_TOTAL_OF_ALL.format(count=len(measuring))
    scope = "{}, {}".format(
        total_of,
        GPU_POWER_COVERED.format(covered=duration_words(covered * step), range=duration_words(range_seconds)),
    )
    return {
        "state": VALUE_MEASURED if totals else VALUE_PARTIAL, "unit": "watts",
        "aggregation": "mean_of_bucket_sums", "scope": scope,
        "cluster": {"value": sum(totals) / len(totals) if totals else None,
                    "covered_seconds": covered * step, "partial": covered < count},
        "nodes": [{"key": str(node.get("key")), "name": node.get("name"),
                   "value": (sum(per_node[str(node.get("key"))]) / len(per_node[str(node.get("key"))]))
                   if per_node[str(node.get("key"))] else None} for node in measuring],
        "excluded": excluded,
        "reason": "" if totals else GPU_POWER_NEVER_COVERED.format(names=", ".join(
            str(node.get("name")) for node in measuring if str(node.get("key")) in short_of) or names),
    }


# --------------------------------------------------------------------------
# Events and the current model
# --------------------------------------------------------------------------


def events(
    rows: Sequence[Mapping[str, Any]], start: int, step: int, count: int, names: Mapping[str, str],
    transitions: Sequence[Mapping[str, Any]] = (),
) -> List[Dict[str, Any]]:
    """Markers from the ledger's records only (§1.6): starts, stops, restarts, model and engine changes.

    "Started serving" and "Stopped serving" are the poller's recorded
    transitions (a replica joining or leaving its targets), never inferred
    from buckets without a row: a slow tick, a locked ledger or an unreadable
    deployment record leaves such buckets while the replica served on
    (pass-4 review B1).
    """
    found: List[Dict[str, Any]] = []
    end = start + step * count
    # The name each machine's rows recorded; a row written without one never
    # erases a name an earlier row kept (review 2).
    recorded = {str(row.get("node_id") or ""): str(row.get("node_name")) for row in transitions if row.get("node_name")}

    folded = fold_rows(rows, start, step, count)

    def event(moment: int, kind: str, key: str, what: str) -> Dict[str, Any]:
        machine = (EVENT_WHOLE_MODEL if key == WHOLE_MODEL_KEY
                   else names.get(key) or recorded.get(key) or REMOVED_MACHINE)
        # B9: a line is named by what serves on it - one machine can carry
        # llama.cpp and its share of a vLLM split. An engine change is itself
        # about the engine, so it keeps the machine alone.
        engine = "" if kind == "engine" else engine_near(folded.get(key, {}), (moment - start) // step, kind=kind)
        line = line_name(machine, engine, whole=key == WHOLE_MODEL_KEY)
        return {"t": moment, "kind": kind, "node": key, "machine": machine, "line": line,
                "label": EVENT_ON.format(event=what, machine=line)}

    # Each replica's own state is tracked, and a line's start or stop is an
    # event only when "some replica of this line serves" changes (ACC-200 and
    # its review): the machines of a split each record their replica leaving,
    # and a switch to the cluster writes vLLM's start before llama.cpp's stop
    # on the same machine, which serves throughout.
    # A line none of whose replicas has a record yet is taken at the first
    # one's word: a stop then means the line was serving until it.
    known: Dict[str, Dict[str, bool]] = {}
    for row in sorted(transitions, key=lambda item: float(item.get("at") or 0.0)):
        moment, kind, key = float(row.get("at") or 0.0), str(row.get("kind") or ""), series_key(row)
        replicas = known.setdefault(key, {})
        was = any(replicas.values()) if replicas else kind == "stopped"
        replicas[str(row.get("replica") or "")] = kind == "started"
        if was != any(replicas.values()) and start <= moment < end:
            found.append(event(int(moment), kind, key, EVENT_STARTED if kind == "started" else EVENT_STOPPED))
    # Restarts, model and engine changes are read from the range's own buckets,
    # so rows the database already grouped by step and raw rows folded here
    # give the same markers (pass-4 review): a change is dated to the bucket
    # it happened in, the one where the new name first appears.
    for key, buckets in folded.items():
        model = engine = None
        for index in sorted(buckets):
            bucket = buckets[index]
            moment = start + index * step
            if (_number(bucket.get("reset_ticks")) or 0) > 0:
                found.append(event(moment, "restart", key, EVENT_RESTART))
            model, changed = _next_name(model, bucket.get("_models"))
            if changed:
                found.append(event(moment, "model", key, EVENT_MODEL.format(model=model)))
            engine, changed = _next_name(engine, bucket.get("_engines"))
            if changed:
                found.append(event(moment, "engine", key, EVENT_TO_CLUSTER if engine == ENGINE_VLLM else EVENT_TO_SINGLE))
    return sorted(found, key=lambda event: (event["t"], event["kind"]))


def _next_name(current: Optional[str], seen: Any) -> Tuple[Optional[str], bool]:
    """The name a bucket leaves in force, and whether it changed there.

    An empty name (a row written before the name was known) changes nothing;
    in a bucket holding two names, the new one is the one that is not current.
    """
    names = sorted(name for name in (seen or ()) if name)
    if not names:
        return current, False
    fresh = [name for name in names if name != current]
    if current is None:
        return (names[0] if len(names) == 1 else fresh[-1]), False
    if not fresh:
        return current, False
    return fresh[-1], True


def llama_only(rows: Iterable[Mapping[str, Any]]) -> bool:
    """Whether every row in a set was served by llama.cpp (and there is at least one)."""
    engines = {str(row.get("engine") or "") for row in rows}
    return engines == {"llama.cpp"}


def field_samples(row: Optional[Mapping[str, Any]], field: str) -> float:
    """How many samples a host bucket holds for ONE field (pass-3 review: per-field coverage).

    ``<field>_n`` when the reader counted it; otherwise the bucket's CPU sample
    count, which is what every field had before the per-field counts.
    """
    if not row:
        return 0.0
    own = _number(row.get(field + "_n"))
    return own if own is not None else (_number(row.get(SAMPLES_KEY)) or 0.0)


def tile_scope(rows: Sequence[Mapping[str, Any]], model: Optional[Mapping[str, Any]], range_seconds: int) -> Tuple[str, Optional[int]]:
    """``(scope, since)``: the tiles' scope, with ``{since}`` when the range holds another model."""
    span = duration_words(range_seconds)
    if not model or not model.get("model"):
        return TILE_SCOPE_NO_MODEL.format(span=span), None
    short = str(model["model"]).rsplit("/", 1)[-1]
    if any(row.get("model") and row.get("model") != model["model"] for row in rows):
        return TILE_SCOPE_SINCE.format(model=short), int(model["since"])
    return TILE_SCOPE.format(model=short, span=span), None


def current_model(rows: Sequence[Mapping[str, Any]]) -> Optional[Dict[str, Any]]:
    """The newest model served in the range, its engine, and when its trailing run began."""
    ordered = sorted(rows, key=lambda row: int(row.get("bucket") or 0))
    if not ordered:
        return None
    newest = ordered[-1]
    since = int(newest.get("bucket") or 0)
    for row in reversed(ordered):
        if row.get("model") != newest.get("model"):
            break
        since = int(row.get("bucket") or 0)
    return {"model": newest.get("model") or "", "engine": newest.get("engine") or "", "since": since}


def serving_names(rows: Iterable[Mapping[str, Any]], names: Mapping[str, str]) -> Dict[str, str]:
    """The name each serving line is drawn under: the machine's, or the whole split model's."""
    shaped = dict(names)
    for row in rows:
        if series_key(row) == WHOLE_MODEL_KEY:
            lead = str(row.get("node_name") or "")
            shaped[WHOLE_MODEL_KEY] = WHOLE_MODEL_NAME.format(lead=lead) if lead else WHOLE_MODEL_NAME_UNNAMED
        elif row.get("node_id") and not shaped.get(str(row.get("node_id"))):
            shaped[str(row["node_id"])] = str(row.get("node_name") or "")
    return shaped
