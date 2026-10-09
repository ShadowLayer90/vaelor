"""Per-node telemetry *history* for the Fleet machine-detail (Phase E1/E2).

`GET /api/v2/telemetry/history?window=<24h|7d|...>&node=<id>` returns a
downsampled short-history trend for ONE machine, shaped for the
machine-detail's live+trend metrics view.
It is REUSE, not reinvention:

* the window is parsed by the same `_parse_window` the Assistant's windowed
  history uses ("24h", "7d", …), so the two cannot read a duration differently;
* the trend itself comes from the EXISTING ``telemetry_history_range``
  control-plane callback, which wraps ``telemetry_store.history_range`` — the
  same store, the same seven-day / 168-bucket downsampling, and the identical
  four retention states (off / starting / failed / running). This route never
  opens InfluxDB and never re-derives a bound the store already owns;
* the retention state is the very ``describe_retention`` dict
  ``GET /telemetry/retention`` serves, read through the ``telemetry_retention``
  callback, so "off"/"starting"/"failed"/"running" read here exactly as they do
  there.

**One machine per response, and the response says which.** The store holds the
controller's own rows and every enrolled worker's ingested rows in the same
measurement. With no ``?node`` the response is the CONTROLLER's series only
(``scope: "controller"``) - never the mean of every machine, which is what it
silently was once workers began reporting (ACC-082). ``?node=<worker>`` reads
that worker's series (``scope: "node"``).

**The current reading is a raw sample, not a bucket.** ``latest`` is the
node's newest stored row, keyed like ``series``; the newest *bucket* of a 24h
window is the mean of a partly elapsed stretch of minutes and is not the
reading now. ``last_sample_at`` / ``last_sample_age_seconds`` date that row.

**A worker's ingest is reported beside its data.** ``ingest`` (worker scope
only, null for the controller) says when the controller last accepted a post
from the worker on the controller's own clock, the worker's measured clock
offset, and - when its posts are being refused for a wrong clock - a plain
sentence saying so (ACC-126). A stored row is dated by the worker's clock, so
without this a drifting clock reads as a silent worker.

**A missing reading is never fabricated.** A metric no bucket measured comes
back as ``null``-valued points (or, when retention is off/starting/failed, empty
series with the reason), so the client draws the "unmeasured" empty state rather
than a flat line at zero.
"""

from __future__ import annotations

import math
from typing import Any, Dict, List, Optional

from flask import request

from .api_common import ApiContext, payload as _payload
from .assistant_machine_tools import _parse_window
from .cluster_placement import CONTROLLER_PLACEMENT_ID, CONTROLLER_PLACEMENT_NAME
from .gpu_vendor_status import HEALTHY_CODES, STATUS_FIELD, known_code, status_sentence
from .platforms.gpu_telemetry import gpu_power_sensor
from .platforms.gpu_temperature import gpu_temperature_reading
from .telemetry_bounds import implausible_note
from .telemetry_ingest_status import describe_ingest, ingest_status, node_is_reporting
from .telemetry_store import (
    MAX_HISTORY_BUCKETS,
    REPORTING_WINDOW_SECONDS,
    TelemetryStoreError,
)

#: The window used when the client sends none, or one this route cannot read. A
#: malformed value falls back here rather than being rejected, because this is a
#: dashboard read with a sensible default, not a typed query the caller must get
#: exactly right. `history_range` still clamps a too-long window to the store's
#: retention period, so the response reports the span actually used.
DEFAULT_WINDOW = "24h"
DEFAULT_WINDOW_SECONDS = 24 * 60 * 60

#: Each client-facing series and the stored telemetry field it reads. A trend
#: bucket carries the flattened field names the data logger writes — `get_trend`
#: strips the aggregate prefix, so a bucket's `cpu_temperature` is the same key a
#: raw sample has — which is why a series is one influx field mapped once here.
#: The machine-detail needs load, memory, GPU activity, unified (GTT) memory and
#: both temperatures; anything a machine never measured is simply absent from the
#: buckets and surfaces as null points.
_SERIES_FIELDS: tuple[tuple[str, str], ...] = (
    ("processor_load", "cpu_percent"),
    ("memory_percent", "memory_percent"),
    ("gpu_busy_percent", "gpu_busy_percent"),
    ("gpu_gtt_used_bytes", "gpu_gtt_used_bytes"),
    ("gpu_gtt_total_bytes", "gpu_gtt_total_bytes"),
    ("cpu_temperature_c", "cpu_temperature"),
    # Two GPU temperatures, each its own series: edge (hwmon) and graphics
    # engine (amd-smi). ``latest.gpu_temperature_sensor`` names which one the
    # owner of that choice picked for this node's newest row (VD-147).
    ("gpu_temperature_c", "gpu_temperature_c"),
    ("gpu_gfx_temperature_c", "gpu_gfx_temperature_c"),
    # The GPU's own power: the graphics engine's on an integrated part (the
    # controller's amd-smi call, a worker's sampler), never the package figure
    # (`gpu_socket_power_watts`), which is the whole SoC (ACC-204, VD-040).
    ("gpu_power_watts", "gpu_power_watts"),
)


def _finite(value: Any) -> Optional[float]:
    """A stored reading as a finite float, or ``None`` for anything that is not.

    A bucket may omit a field entirely (nothing measured it in that window) or
    carry a non-finite aggregate; both are holes, and a hole is ``None`` — never
    a substituted zero, which on a trend line is indistinguishable from a sensor
    pinned at zero.
    """
    if isinstance(value, bool) or value is None:
        return None
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    return number if math.isfinite(number) else None


def _series(buckets: List[Dict[str, Any]]) -> Dict[str, Dict[str, Any]]:
    """One ``{points: [{t, v}]}`` per metric, oldest to newest.

    Every series shares the one bucket timeline `history_range` returned, so a
    metric this machine does not measure has a point for each bucket with a
    ``null`` value rather than a shorter, misaligned array.
    """
    return {
        name: {
            "points": [
                {"t": row.get("time"), "v": _finite(row.get(field))}
                for row in buckets
            ]
        }
        for name, field in _SERIES_FIELDS
    }


def _empty_series() -> Dict[str, Dict[str, Any]]:
    """The series shape with no points, for a store that is off or unreadable."""
    return {name: {"points": []} for name, _ in _SERIES_FIELDS}


def _latest(row: Any, node_name: str = "", implausible: int = 0) -> Dict[str, Any]:
    """The newest RAW sample as ``{t, values}``, keyed like ``series``.

    ``row`` is the store's newest stored row for the node (fields as written,
    ``time`` in epoch seconds) or ``None`` when nothing is stored. A field the
    node never measured is ``None``, never zero; ``t`` is ``None`` when there is
    no row at all.

    Two sentences ride with it for the machine's card (review S-15), each ``""``
    when there is nothing to say: why its GPU readings are missing, from the
    row's own status code, and how many readings were discarded as physically
    implausible.
    """
    source = row if isinstance(row, dict) else {}
    stamp = _finite(source.get("time"))
    return {
        "t": None if stamp is None else int(stamp),
        "values": {name: _finite(source.get(field)) for name, field in _SERIES_FIELDS},
        # "graphics engine", "edge", or None when this row has no GPU temperature.
        "gpu_temperature_sensor": gpu_temperature_reading(source)[1],
        # What measured the power, by the telemetry owner's word (ACC-204).
        "gpu_power_sensor": gpu_power_sensor(source),
        "gpu_readings_note": _gpu_readings_note(source, node_name),
        "implausible_note": implausible_note(implausible, node_name),
    }


def _gpu_readings_note(row: Dict[str, Any], node_name: str) -> str:
    """Why a machine's GPU readings are missing, in the status table's words, or ``""``."""
    if not row:
        return ""
    code = known_code(row.get(STATUS_FIELD))
    if code is None:
        # No status at all: an agent from before these readings - said only on
        # a machine whose row shows a GPU, never on one that has none.
        has_gpu = any(_finite(row.get(field)) is not None for field in ("gpu_busy_percent", "gpu_temperature_c"))
        return status_sentence(None, node_name) if has_gpu and STATUS_FIELD not in row else ""
    return "" if code in HEALTHY_CODES else status_sentence(code, node_name)


def _node_name(callbacks: Dict[str, Any], node_id: str) -> str:
    """What a machine is called on screen: the controller's placement name, a worker's enrolled name."""
    if node_id == CONTROLLER_PLACEMENT_ID:
        return CONTROLLER_PLACEMENT_NAME
    manager = callbacks.get("cluster_manager")
    try:
        return next((str(record.get("name") or "") for record in manager.store.list_nodes()
                     if str(record.get("id")) == node_id), "")
    except (AttributeError, OSError, TypeError, ValueError):
        return ""


def _implausible_count(callbacks: Dict[str, Any], scope: str, node_id: str) -> int:
    """How many readings the bounds table dropped for this machine since the control plane started."""
    try:
        if scope == "node":
            return int(ingest_status(node_id).get("implausible_dropped") or 0)
        counter = callbacks.get("telemetry_implausible")
        return int(counter() or 0) if counter is not None else 0
    except (TypeError, ValueError, AttributeError):
        return 0


def _ingest(scope: str, node: str) -> Optional[Dict[str, Any]]:
    """The worker's ingest status for a node-scoped read; None for the controller.

    The controller's own rows are written in-process by its data logger, not
    posted, so there is no ingest to describe for it.
    """
    return describe_ingest(node) if scope == "node" else None


def _retention(callbacks: Dict[str, Any]) -> Optional[Dict[str, Any]]:
    """The configured-vs-applied retention state, or ``None`` when unwired.

    Read through the same ``telemetry_retention`` callback ``/telemetry/
    retention`` serves, so the state and its wording are `describe_retention`'s,
    not a second copy that could drift from it.
    """
    status = callbacks.get("telemetry_retention")
    if status is None:
        return None
    try:
        return status()
    except TelemetryStoreError:
        return None


def _known_node(callbacks: Dict[str, Any], node_id: str) -> bool:
    """Whether ``node_id`` names this controller or an enrolled worker.

    The controller is always known; a worker is known when the cluster store
    lists it. An unwired or unreadable store leaves only the controller known,
    so an unknown ``?node`` degrades to an honest empty payload rather than a
    500 or a filter against a node that does not exist.
    """
    if node_id == CONTROLLER_PLACEMENT_ID:
        return True
    manager = callbacks.get("cluster_manager")
    if manager is None:
        return False
    try:
        return any(
            str(record.get("id")) == node_id for record in manager.store.list_nodes()
        )
    except (AttributeError, OSError, TypeError, ValueError):
        return False


def register_telemetry_history_routes(context: ApiContext) -> None:
    blueprint = context.blueprint
    callbacks = context.callbacks
    require_auth = context.require_auth

    @blueprint.get("/telemetry/history")
    @require_auth("viewer")
    def telemetry_history():
        requested = request.args.get("window", DEFAULT_WINDOW) or DEFAULT_WINDOW
        seconds = _parse_window(requested)
        if seconds is None:
            seconds = DEFAULT_WINDOW_SECONDS
        retention = _retention(callbacks)
        # No `?node` is the controller: a read filtered to its own rows (tagged
        # `node=controller`, plus the untagged rows written before the tag
        # existed), never every node's rows averaged together (ACC-082). `?node`
        # names the controller or an enrolled worker; the id is validated
        # against the cluster store so an unknown one degrades honestly rather
        # than filtering against a node that does not exist.
        requested_node = request.args.get("node", "") or ""
        resolved_node = CONTROLLER_PLACEMENT_ID
        scope = "controller"
        if requested_node:
            if not _known_node(callbacks, requested_node):
                return _payload(_unavailable(
                    scope="node", node=requested_node, retention=retention,
                    requested=requested, seconds=seconds,
                    reason=(
                        "No cluster node has that id, so there is no per-node "
                        "telemetry series to read for it."
                    ),
                ))
            resolved_node = requested_node
            scope = (
                "controller" if requested_node == CONTROLLER_PLACEMENT_ID else "node"
            )
        ingest = _ingest(scope, resolved_node)
        base = {
            "scope": scope,
            "node": resolved_node,
            "requested_window": requested,
            "retention": retention,
            # Read from the ingest record, not the store, so a worker whose
            # posts are refused is explained even while its series is empty.
            "ingest": ingest,
            # The server's one verdict on freshness (ACC-128), on the
            # controller's receive clock first (ACC-126). Without a store read
            # only the receipt can say; the success branch adds the row's age.
            "reporting": node_is_reporting(None, ingest),
            "reporting_window_seconds": REPORTING_WINDOW_SECONDS,
        }
        ranged = callbacks.get("telemetry_history_range")
        if ranged is None:
            # No time-ranged callback wired in at all: a build or wiring fault,
            # named as one rather than as an appliance setting.
            return _payload({
                **base,
                "available": False,
                "reason": (
                    "This control plane has no time-ranged telemetry callback "
                    "wired into it, so history cannot be read. That is a wiring "
                    "fault rather than an appliance setting."
                ),
                "window_seconds": seconds,
                "bucket_seconds": 0,
                "sample_interval_seconds": 0,
                "function": "mean",
                "series": _empty_series(),
                "latest": _latest(None),
                "last_sample_at": None,
                "last_sample_age_seconds": None,
            })
        try:
            # The node is always named, the controller included, so no read
            # here can fall through to an every-node average.
            result = ranged(seconds, MAX_HISTORY_BUCKETS, resolved_node)
        except TelemetryStoreError as error:
            # Retention off / still starting / failed / on-but-unreadable: the
            # store's own sentence, with the retention state beside it, and empty
            # series rather than a fabricated flat line.
            return _payload({
                **base,
                "available": False,
                "reason": str(error),
                "window_seconds": seconds,
                "bucket_seconds": 0,
                "sample_interval_seconds": 0,
                "function": "mean",
                "series": _empty_series(),
                "latest": _latest(None),
                "last_sample_at": None,
                "last_sample_age_seconds": None,
            })
        buckets = [row for row in (result.get("buckets") or []) if isinstance(row, dict)]
        bucket_seconds = int(result.get("bucket_seconds", 0) or 0)
        return _payload({
            **base,
            "available": True,
            # An empty result from a working store must say so: silence reads as
            # "no record", the complaint VD-095 exists to remove.
            "reason": "" if buckets else (
                "The telemetry store is readable but holds no samples in this "
                "window yet. Retention records from the moment it starts, so a "
                "window reaching before that holds nothing to show."
            ),
            "window_seconds": int(result.get("window_seconds", seconds)),
            "bucket_seconds": bucket_seconds,
            # The resolution actually returned, so the client can say "10-minute
            # buckets over the last 24h" rather than guessing from point spacing.
            "sample_interval_seconds": bucket_seconds,
            "function": result.get("function", "mean"),
            "series": _series(buckets),
            # The node's newest raw sample, the current reading as measured
            # (WF3): the newest bucket above is a mean, not "now".
            "latest": _latest(
                result.get("latest"), _node_name(callbacks, resolved_node),
                _implausible_count(callbacks, scope, resolved_node),
            ),
            # When this node last reported and how long ago (E2c's "not reporting
            # since X"), dated by the row's own stamp; None from a store that
            # never held a row for it.
            "last_sample_at": result.get("last_sample_at"),
            "last_sample_age_seconds": result.get("last_sample_age_seconds"),
            # The same verdict, now with the stored row's age to fall back on.
            "reporting": node_is_reporting(result.get("last_sample_age_seconds"), ingest),
        })


def _unavailable(*, scope, node, retention, requested, seconds, reason):
    """The available:false payload for an unknown node, shaped like the others."""
    return {
        "scope": scope,
        "node": node,
        "requested_window": requested,
        "retention": retention,
        # An unknown node was never enrolled, so it has no ingest to describe.
        "ingest": None,
        "available": False,
        "reason": reason,
        "window_seconds": seconds,
        "bucket_seconds": 0,
        "sample_interval_seconds": 0,
        "function": "mean",
        "series": _empty_series(),
        "latest": _latest(None),
        "last_sample_at": None,
        "last_sample_age_seconds": None,
    }
