"""What the Performance dashboard reads, gathered once per request (VD-147 S4).

The two layers read the SAME owners the Performance tab's snapshot does, so a
number cannot differ between the dashboard and the Diagnostics under it:

* the engine: `performance_snapshot_source._cluster_state` (the fleet's record,
  Mode-B gated) and `performance_serving.serving_section` over the newest
  serving sample;
* the machines: the controller, and every enrolled worker with its own class
  bands, GPU limits and sensor label (`performance_snapshot_source.worker_nodes`);
* freshness: the now layer's ONE bounded read (`telemetry_reader`), judged by
  `telemetry_ingest_status.node_is_reporting` with the controller's receive age
  first; when the bounded read has no row for a worker, the ingest record's
  receipt dates it, and only when that is absent too is one unbounded
  single-node read made;
* serving history: the usage ledger's per-replica buckets and speed bands;
* host history: `telemetry_reader.TelemetryReader.host_buckets`.

Every read that fails becomes that part's honest state; none fails the request.
"""

from __future__ import annotations

import sqlite3
import time
from typing import Any, Dict, List, Mapping, Optional

from .cluster_placement import CONTROLLER_PLACEMENT_ID, CONTROLLER_PLACEMENT_NAME
from .gpu_cluster_mode_state import ClusterModeStore
from .performance_dashboard import offered_ranges
from .performance_dashboard_range import REMOVED_MACHINE
from .performance_serving import serving_section
from .performance_snapshot_source import (
    _cluster_state, _controller_gpu_limits, _thermal_policy, worker_nodes,
)
from .telemetry_ingest_status import describe_ingest
from .telemetry_reader import ROLLUP_FROM_SECONDS, TelemetryReaderError

#: Why the host panels are empty when the control plane has no dashboard reader.
READER_NOT_WIRED = (
    "This control plane has no dashboard reader for the telemetry store, so "
    "host history cannot be drawn. That is a wiring fault, not an appliance setting."
)

__all__ = ["READER_NOT_WIRED", "REMOVED_MACHINE", "gather_now", "gather_range"]


def _machines(callbacks: Mapping[str, Any]) -> List[Dict[str, Any]]:
    """The controller and every enrolled worker, with their own bands, limits and sensor label."""
    controller = {
        "key": CONTROLLER_PLACEMENT_ID, "name": CONTROLLER_PLACEMENT_NAME, "role": "controller",
        "policy": _thermal_policy(dict(callbacks)), "limits": _controller_gpu_limits(dict(callbacks)),
        "class_known": True, "temperature_label": "",
    }
    workers = [{
        "key": worker["id"], "name": worker["name"], "role": "worker",
        "policy": worker.get("thermal_policy") or {}, "limits": worker.get("gpu_temperature_limits") or {},
        "class_known": worker.get("machine_class_known", True), "temperature_label": worker.get("gpu_temperature_label", ""),
    } for worker in worker_nodes(dict(callbacks))]
    return [controller] + workers


def _slots(callbacks: Mapping[str, Any], machines: List[Dict[str, Any]], now: float) -> None:
    store = callbacks.get("chart_slots")
    try:
        assigned = store.assign([machine["key"] for machine in machines], now) if store is not None else {}
    except (OSError, sqlite3.Error, AttributeError):
        assigned = {}
    for index, machine in enumerate(machines):
        machine["slot"] = assigned.get(machine["key"], index + 1)


def _serving(callbacks: Mapping[str, Any], now: float) -> Dict[str, Any]:
    mode_state = ClusterModeStore().read()  # fails safe to Mode A
    cluster = _cluster_state(dict(callbacks), mode_state)
    latest = callbacks.get("serving_metrics_latest")
    try:
        sample = latest() if latest is not None else None
    except (OSError, RuntimeError, TypeError, ValueError):
        sample = None
    return serving_section(sample, now, cluster)


def gather_now(callbacks: Mapping[str, Any], now: Optional[float] = None) -> Dict[str, Any]:
    """The now layer's inputs: the engine, and every machine's newest row and freshness facts."""
    moment = time.time() if now is None else float(now)
    machines = _machines(callbacks)
    _slots(callbacks, machines, moment)
    reader = callbacks.get("telemetry_reader")
    rows: Dict[str, Dict[str, Any]] = {}
    reader_ok = reader is not None
    if reader is not None:
        try:
            rows = reader.latest_per_node()
        except TelemetryReaderError:
            reader_ok = False
    for machine in machines:
        key = machine["key"]
        machine["ingest"] = describe_ingest(key, moment) if machine["role"] == "worker" else None
        row = rows.get(key)
        if row is None and not reader_ok and key == CONTROLLER_PLACEMENT_ID:
            # Retention off or unreadable: the controller's live sample still
            # answers "now" (§6, the retention row).
            current = callbacks.get("current_data")
            try:
                live = dict(current() or {}) if current is not None else {}
            except (OSError, RuntimeError, TypeError, ValueError):
                live = {}
            row = {**live, "time": moment} if live else None
        machine["row"] = row
        machine["row_age_seconds"] = None if row is None or row.get("time") is None else max(0.0, moment - float(row["time"]))
        machine["since"] = None
        if row is None:
            received = (machine["ingest"] or {}).get("received_at")
            if received is not None:
                machine["since"] = received
            elif reader_ok:
                try:
                    older = reader.latest_for(key)
                except (TelemetryReaderError, ValueError):
                    older = None
                machine["since"] = None if older is None else older.get("time")
    return {"now": moment, "serving": _serving(callbacks, moment), "nodes": machines}


def _bounds(range_key: str, now: float) -> Dict[str, int]:
    seconds, step = offered_ranges()[range_key]
    count = seconds // step
    end = (int(now) // step) * step + step
    return {"range_seconds": seconds, "step": step, "count": count, "start": end - count * step}


def gather_range(
    callbacks: Mapping[str, Any], range_key: str, now: Optional[float] = None,
    now_inputs: Optional[Mapping[str, Any]] = None,
) -> Dict[str, Any]:
    """The range layer's inputs for one offered range (the caller has checked it).

    ``now_inputs`` is the now layer's gathered reads when the caller already
    has them (the routes share one set per window); otherwise they are read.
    """
    moment = time.time() if now is None else float(now)
    bounds = _bounds(range_key, moment)
    start, end = bounds["start"], bounds["start"] + bounds["count"] * bounds["step"]
    now_inputs = now_inputs if now_inputs is not None else gather_now(callbacks, moment)
    ledger = callbacks.get("model_usage")
    try:
        # Grouped by the range's step in SQL: eight replicas over seven days
        # folded in Python took seconds (pass-3 review).
        replica_rows = ledger.replica_buckets_stepped(start, end, bounds["step"]) if ledger is not None else []
        histograms = ledger.replica_histograms_stepped(start, end, bounds["step"]) if ledger is not None else []
        transitions = ledger.replica_transitions(start, end) if ledger is not None else []
    except (OSError, sqlite3.Error, AttributeError):
        replica_rows, histograms, transitions = [], [], []
    known = {machine["key"] for machine in now_inputs["nodes"]}
    for row in replica_rows:
        if row.get("node_id") not in known and not row.get("node_name"):
            row["node_name"] = REMOVED_MACHINE
    reader = callbacks.get("telemetry_reader")
    ready = _call(callbacks.get("telemetry_rollup_ready"), False)
    host, source, reason, note = None, "", READER_NOT_WIRED, ""
    if reader is not None:
        try:
            # Until the roll-up is created and backfilled, a long range reads
            # the raw rows (pass-3 review S4-B2): never a range of false gaps.
            host, source = reader.host_buckets(start, end, bounds["step"], rollup_ready=bool(ready))
            reason = ""
        except TelemetryReaderError as error:
            host, reason = None, str(error)
    if not ready and bounds["range_seconds"] >= ROLLUP_FROM_SECONDS:
        note = str(_call(callbacks.get("telemetry_rollup_note"), "") or "")
    return {
        **bounds, "range": range_key, "now": moment, "serving": now_inputs["serving"],
        "nodes": now_inputs["nodes"], "replica_rows": replica_rows, "histograms": histograms,
        "transitions": transitions,
        "host": host, "host_source": source, "host_reason": reason, "host_note": note,
        "held_slots": _held(callbacks),
    }


def _call(callback: Any, fallback: Any) -> Any:
    """A wiring callback's answer, or ``fallback`` when it is absent or fails."""
    if callback is None:
        return fallback
    try:
        return callback()
    except (OSError, RuntimeError, TypeError, ValueError):
        return fallback


def _held(callbacks: Mapping[str, Any]) -> Dict[str, int]:
    """Every chart slot still held, a removed machine's included (its history keeps its colour)."""
    store = callbacks.get("chart_slots")
    try:
        return dict(store.held()) if store is not None else {}
    except (OSError, sqlite3.Error, AttributeError):
        return {}
