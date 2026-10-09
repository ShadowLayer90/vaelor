"""The GPU readings the Assistant gives for each machine, from the telemetry owner (ACC-201).

The controller's own adapter is read in `assistant_machine_tools.gpu_status`
from its live sample. A cluster's workers are not on this machine's bus: their
GPU readings are the rows their sampler hands the telemetry emitter, which the
controller's telemetry store already holds (`telemetry_ingest.ACCEPTED_FIELDS`).
This reads each enrolled worker's newest row through the same
``telemetry_history_range`` and the same freshness verdict the Performance tab
uses (`telemetry_ingest_status.worker_is_reporting`), so the Assistant cannot
call a reading current that the tab calls stale.

**Power is named by what measured it.** On an integrated part the GPU's power
is the graphics engine's own (``amd-smi apu_average_gfx_power``); the package
figure (``gpu_socket_power_watts``) is the whole SoC and is never given as the
GPU's (VD-040). A discrete card's hwmon figure is the board's.
"""

from __future__ import annotations

from typing import Any, Dict, List, Mapping, Optional

from .gpu_vendor_status import CODES_WITH_READINGS, STATUS_FIELD, known_code, status_sentence
from .platforms.gpu_telemetry import POWER_SENSOR_BOARD, gpu_power_sensor
from .platforms.gpu_temperature import gpu_temperature_reading
from .telemetry_ingest_status import worker_is_reporting
from .telemetry_store import MAX_HISTORY_BUCKETS, TelemetryStoreError


#: How far back the newest worker row is looked for. Freshness is judged by
#: the reporting verdict, not by this window.
WORKER_WINDOW_SECONDS = 600

NOT_REPORTING = "The GPU on {name} has no current reading: it is not reporting its telemetry now."
NO_POWER_READING = "The GPU on {name} gave no power reading."
#: The controller's own store could not be read (off, failed, not wired): said
#: as the controller's, never as the worker's silence (review of ACC-201).
STORE_UNREADABLE = (
    "This controller could not read its telemetry store, so the workers' GPU "
    "readings are not known right now."
)
#: One worker's row could not be read while the others' could: that worker is named.
WORKER_UNREADABLE = "This controller could not read the GPU readings of {name} from its telemetry store."
#: A worker enrolled without a name is never named by its internal id.
UNNAMED_WORKER = "a worker with no name"


def _number(value: Any) -> Optional[float]:
    return float(value) if isinstance(value, (int, float)) and not isinstance(value, bool) else None


def power_sensor(record: Mapping[str, Any], live: Optional[Mapping[str, Any]]) -> Optional[str]:
    """The name of what measured a controller adapter's power, or None with no power."""
    if live is not None:
        return gpu_power_sensor(dict(live))
    return POWER_SENSOR_BOARD if _number(record.get("power_watts")) is not None else None


def _workers(callbacks: Mapping[str, Any]) -> List[Dict[str, Any]]:
    from .performance_snapshot_source import worker_nodes

    return worker_nodes(dict(callbacks))


def worker_gpu_readings(callbacks: Mapping[str, Any]) -> List[Dict[str, Any]]:
    """Each enrolled worker's GPU power and temperature now, with sensors and reasons."""
    ranged = callbacks.get("telemetry_history_range")
    machines: List[Dict[str, Any]] = []
    for worker in _workers(callbacks):
        named = str(worker.get("name") or "")
        name = named if named and named != str(worker.get("id")) else UNNAMED_WORKER
        if ranged is None:
            machines.append({"name": name, "reporting": None, "store_failed": True})
            continue
        try:
            result = ranged(WORKER_WINDOW_SECONDS, MAX_HISTORY_BUCKETS, worker["id"]) or {}
        except (TelemetryStoreError, OSError, TypeError, ValueError):
            machines.append({"name": name, "reporting": None, "store_failed": True})
            continue
        latest, age = result.get("latest"), result.get("last_sample_age_seconds")
        if not isinstance(latest, Mapping) or not worker_is_reporting(str(worker["id"]), age):
            machines.append({"name": name, "reporting": False, "reason": NOT_REPORTING.format(name=name)})
            continue
        power = _number(latest.get("gpu_power_watts"))
        code = known_code(latest.get(STATUS_FIELD))
        temperature, sensor = gpu_temperature_reading(latest, edge_label=worker.get("gpu_temperature_label"))
        machines.append({
            "name": name, "reporting": True,
            "power_watts": power, "power_sensor": gpu_power_sensor(dict(latest)),
            "power_reason": "" if power is not None else (
                status_sentence(code, name) if code is None or code not in CODES_WITH_READINGS
                else NO_POWER_READING.format(name=name)),
            "temperature_c": temperature, "temperature_sensor": sensor,
            # VD-205 item 5: the worker's GPU use and GPU memory (GTT), from
            # the same row, so a GPU answer covers what the worker is holding.
            "busy_percent": _number(latest.get("gpu_busy_percent")),
            "gtt_used_bytes": _number(latest.get("gpu_gtt_used_bytes")),
            "gtt_total_bytes": _number(latest.get("gpu_gtt_total_bytes")),
        })
    # The shared sentence only when no worker's row could be read: one failed
    # read among good ones is that worker's, named (review 2 of ACC-201).
    every_read_failed = bool(machines) and all(machine.get("store_failed") for machine in machines)
    for machine in machines:
        if machine.pop("store_failed", False):
            machine["reason"] = STORE_UNREADABLE if every_read_failed else WORKER_UNREADABLE.format(name=machine["name"])
    return machines
