"""The GPU snapshot of the on-demand serving profile: what is read, and what it is called.

Split out of :mod:`vaelor.serving_profiler` (VD-147, slice S1b) at the line
ceiling, and corrected in the move. The snapshot used to take the first field
whose name looked right, and that produced two labels that did not match what
was measured (LESSONS 5):

* **Power.** ``amd-smi monitor`` prints one ``power_usage`` figure. On an AMD
  integrated GPU that figure is the whole chip's package power (measured on
  both Strix Halo boxes: 26 W and 25 W, equal to ``socket_power``, while the
  graphics engine itself drew 0.44 W and 3.56 W). It was shown as the GPU's
  power. The graphics engine's own draw is ``apu_average_gfx_power`` in
  ``amd-smi metric``, the same field the telemetry reads
  (:func:`vaelor.platforms.graphics_software.gpu_gfx_power_watts`, VD-040), so
  the snapshot now reads it there and says which kind of power each figure is.
* **Temperature.** The first of ``edge_temperature``, ``hotspot_temperature``,
  ``gpu_temperature``, ``temperature`` or ``temp`` found anywhere in the output
  was shown as "GPU temp"; the last two match any sensor at all. The snapshot
  now reads named fields only and carries the sensor with the value.

Every figure is absent, never zero, when the tool did not print it.
"""

from __future__ import annotations

import json
from typing import Any, Dict, List, Optional, Sequence

from .platforms.gpu_temperature import (
    SENSOR_EDGE, SENSOR_GRAPHICS_ENGINE, SENSOR_HOTSPOT,
)
from .platforms.graphics_software import (
    FREQUENCY_UNITS, PERCENT_UNITS, POWER_UNITS, TEMPERATURE_UNITS, _amd_smi_numbers,
    gpu_gfx_power_watts, gpu_gfx_temperature_c, integrated_amd_gpu,
    json_after_banner, socket_power_watts,
)

#: What kind of power a snapshot's ``power_watts`` is (``power_kind``). The
#: screen words its label from this, so package power is never shown under a
#: GPU heading.
POWER_GRAPHICS_ENGINE = "graphics-engine"
POWER_GPU = "gpu"
POWER_UNIDENTIFIED = "unidentified"


def discovered_integrated_gpu() -> Optional[bool]:
    """Whether discovery finds an AMD integrated GPU on this machine.

    ``None`` when no accelerator is discovered at all, which is "cannot tell"
    and not "no": the snapshot then marks ``monitor``'s power figure as
    unidentified rather than calling it the GPU's or the package's.
    """
    from .platforms.accelerators import discover_accelerators

    records = discover_accelerators()
    if not records:
        return None
    # Any discovered record, not the first (pass-2 review SC-H): a display
    # adapter enumerated first must not hide the integrated GPU behind it.
    return any(integrated_amd_gpu(record) for record in records)


def amd_smi_command(*, amd_smi_path: str) -> List[str]:
    """A one-shot ``amd-smi monitor`` snapshot as JSON, covering GFX use and the
    per-process GPU share.

    ``monitor`` (not ``metric``) is the form the grounding proved carries the
    PROCESS INFO - the serving process's GTT and compute share - alongside the
    GFX clock and occupancy. ``--json`` gives the machine form
    :func:`parse_amd_smi` reads; the flags request power (``-p``), temperature
    (``-t``), GFX use (``-u``), memory (``-m``) and the process list (``-q``).
    The parser searches by field name and tolerates a shape mismatch by leaving
    a field absent, so a build whose flags differ degrades honestly.
    """
    return [amd_smi_path, "monitor", "--json", "-p", "-t", "-u", "-m", "-q"]


def amd_smi_metric_command(*, amd_smi_path: str) -> List[str]:
    """The ``amd-smi metric`` read that carries the graphics engine's own figures.

    The same command the telemetry runs. ``monitor`` has no graphics-engine
    power or temperature on an integrated part, so the snapshot takes those two
    from here.
    """
    return [amd_smi_path, "metric", "--json"]


#: amd-smi field-name candidates, most-specific first. amd-smi's JSON keys vary a
#: little across versions, so each signal is searched for by a small set of
#: synonyms rather than a fixed path; the first that resolves to a number wins,
#: and none resolving leaves the field absent (never zero).
_AMD_MONITOR_POWER_KEYS = ("power_usage", "socket_power", "average_socket_power")
_AMD_GFX_CLOCK_KEYS = ("gfx_clock", "gfxclk", "gfx_clk", "sclk")
_AMD_GFX_ACTIVITY_KEYS = ("gfx_activity", "gfx_usage", "graphics_usage", "gfx", "gpu_usage")
_AMD_GTT_KEYS = ("gtt_mem", "gtt_memory", "gtt_used", "used_gtt", "gtt")
_AMD_PID_KEYS = ("pid", "process_id")
_AMD_PROC_NAME_KEYS = ("name", "process_name", "proc_name")
_AMD_PROC_CU_KEYS = ("cu_occupancy", "compute_unit", "gfx", "cu_percent")

#: The temperature fields the snapshot will read, each with the sensor it is.
#: Named fields only: a bare ``temperature`` could be any sensor in the output.
_EDGE_TEMPERATURE_KEYS = ("edge_temperature", "edge")
_HOTSPOT_TEMPERATURE_KEYS = ("hotspot_temperature",)


def parse_amd_smi(
    stdout: Any, serving_pid: Optional[int] = None, *,
    metric_stdout: Any = None, integrated: Optional[bool] = None,
) -> Dict[str, Any]:
    """Structured GPU fields from an ``amd-smi monitor --json`` snapshot.

    ``metric_stdout`` is the ``amd-smi metric --json`` output taken beside it,
    or ``None`` when that read did not run or failed. ``integrated`` says
    whether discovery found an AMD integrated GPU: ``True``, ``False``, or
    ``None`` when it could not tell.

    Power, in this order: the graphics engine's own draw from ``metric``
    (``power_kind`` :data:`POWER_GRAPHICS_ENGINE`); else, on a part known NOT to
    be integrated, ``monitor``'s figure, which is the card's own
    (:data:`POWER_GPU`); else, on a part that could not be identified,
    ``monitor``'s figure marked :data:`POWER_UNIDENTIFIED`. On an integrated
    part ``monitor``'s figure is the whole chip and is never ``power_watts``:
    it is reported as ``package_power_watts``, as is ``metric``'s socket power.

    Temperature, in this order, always with its sensor
    (``gpu_temperature_sensor``): graphics engine, then edge, from ``metric``;
    then edge from ``monitor``; then ``monitor``'s hotspot, only on a part known
    not to be integrated (an integrated GPU has no junction sensor, and what
    ``monitor`` prints under that name there is not one).
    """
    data = json_after_banner(stdout) if isinstance(stdout, str) else None
    metric = json_after_banner(metric_stdout) if isinstance(metric_stdout, str) else None
    if data is None and metric is None:
        return {}

    snapshot: Dict[str, Any] = {}
    gfx_power = gpu_gfx_power_watts(metric) if metric is not None else None
    monitor_power = _first_number(_find_field(data, _AMD_MONITOR_POWER_KEYS), POWER_UNITS)
    package_power = socket_power_watts(metric) if metric is not None else None
    if gfx_power is not None:
        snapshot["power_watts"] = round(gfx_power, 2)
        snapshot["power_kind"] = POWER_GRAPHICS_ENGINE
    elif monitor_power is not None and integrated is not True:
        snapshot["power_watts"] = round(monitor_power, 2)
        snapshot["power_kind"] = POWER_GPU if integrated is False else POWER_UNIDENTIFIED
    if package_power is None and integrated is True:
        package_power = monitor_power
    if package_power is not None:
        snapshot["package_power_watts"] = round(package_power, 2)

    for value, sensor in _temperature_candidates(data, metric, integrated):
        if value is not None:
            snapshot["gpu_temperature_c"] = round(value, 2)
            snapshot["gpu_temperature_sensor"] = sensor
            break

    _set_number(snapshot, "gfx_clock_mhz", _find_field(data, _AMD_GFX_CLOCK_KEYS), FREQUENCY_UNITS)
    _set_number(snapshot, "gfx_activity_percent", _find_field(data, _AMD_GFX_ACTIVITY_KEYS), PERCENT_UNITS)

    process = _find_serving_process(data, serving_pid)
    if process is not None:
        snapshot["process"] = process
    return snapshot


def _temperature_candidates(data: Any, metric: Any, integrated: Optional[bool]):
    """``(value, sensor)`` in the order the snapshot prefers them."""
    if metric is not None:
        yield gpu_gfx_temperature_c(metric), SENSOR_GRAPHICS_ENGINE
        # A bare ``edge`` key is the edge temperature only inside the
        # document's temperature block (review S-25): ``fan.edge`` is not one.
        yield _first_number(_find_field(_temperature_block(metric), _EDGE_TEMPERATURE_KEYS), TEMPERATURE_UNITS), SENSOR_EDGE
    yield _first_number(_find_field(data, _EDGE_TEMPERATURE_KEYS[:1]), TEMPERATURE_UNITS), SENSOR_EDGE
    if integrated is False:
        yield _first_number(_find_field(data, _HOTSPOT_TEMPERATURE_KEYS), TEMPERATURE_UNITS), SENSOR_HOTSPOT


def _temperature_block(document: Any) -> Any:
    """The ``temperature`` section(s) of a ``metric`` document, wherever they nest."""
    found: List[Any] = []

    def walk(node: Any) -> None:
        if isinstance(node, dict):
            for key, value in node.items():
                if str(key).lower() == "temperature" and isinstance(value, dict):
                    found.append(value)
                else:
                    walk(value)
        elif isinstance(node, list):
            for item in node:
                walk(item)

    walk(document)
    return found


def _find_serving_process(data: Any, serving_pid: Optional[int]) -> Optional[Dict[str, Any]]:
    """The serving PID's own row from the snapshot's process list, if present."""
    if not isinstance(serving_pid, int) or serving_pid <= 0:
        return None
    for entry in _iter_process_entries(data):
        pid = _first_number(_find_field(entry, _AMD_PID_KEYS))
        if pid is None or int(pid) != int(serving_pid):
            continue
        row: Dict[str, Any] = {"pid": int(serving_pid), "raw": entry}
        name = _find_field(entry, _AMD_PROC_NAME_KEYS)
        if isinstance(name, str) and name.strip():
            row["name"] = name.strip()
        _set_number(row, "gtt", _find_field(entry, _AMD_GTT_KEYS))
        _set_number(row, "compute_percent", _find_field(entry, _AMD_PROC_CU_KEYS))
        return row
    return None


def _iter_process_entries(data: Any):
    """Yield the dict entries of the snapshot's process list, wherever it nests.

    amd-smi puts the process list under a key whose name carries "process"; the
    list holds one dict per process, sometimes wrapped one level deeper (a
    ``{"process_info": {...}}`` shape). Both are yielded flat so the PID match does
    not depend on the exact nesting.
    """
    if isinstance(data, dict):
        for key, value in data.items():
            if "process" in str(key).lower() and isinstance(value, list):
                for item in value:
                    if isinstance(item, dict):
                        # Unwrap a single-key ``{"process_info": {...}}`` wrapper.
                        inner = item.get("process_info") if len(item) == 1 else None
                        yield inner if isinstance(inner, dict) else item
            else:
                yield from _iter_process_entries(value)
    elif isinstance(data, list):
        for item in data:
            yield from _iter_process_entries(item)


def _find_field(obj: Any, names: Sequence[str]) -> Any:
    """The first value under any of ``names`` (case-insensitive), searched deep.

    amd-smi's ``--json`` nests differently across versions, so a signal is found
    by field name anywhere in the tree rather than by a fixed path. The synonyms
    are tried in order so the most specific key wins over a broader one that also
    matches.
    """
    wanted = [name.lower() for name in names]
    for target in wanted:
        found = _find_one(obj, target)
        if found is not None:
            return found
    return None


def _find_one(obj: Any, target: str) -> Any:
    if isinstance(obj, dict):
        for key, value in obj.items():
            if str(key).lower() == target:
                return value
        for value in obj.values():
            found = _find_one(value, target)
            if found is not None:
                return found
    elif isinstance(obj, list):
        for item in obj:
            found = _find_one(item, target)
            if found is not None:
                return found
    return None


def _first_number(raw: Any, units: Optional[Dict[str, float]] = None) -> Optional[float]:
    """Unwrap an amd-smi reading to one number, or ``None``.

    ``--json`` wraps a measurement as ``{"value": 8, "unit": "W"}`` and may nest a
    scalar in a list; the literal ``"N/A"`` amd-smi prints for an unpublished field
    is not a number, so it becomes ``None`` (absent) rather than zero. With
    ``units`` (the quantity's table in `graphics_software`) the unit is read and
    scaled the way every other amd-smi reader does (review S-24): a value in a
    unit the table does not hold is absent, never taken as printed.
    """
    if units is not None:
        numbers = _amd_smi_numbers(raw, units)
        if numbers or _wrapped(raw):
            return numbers[0] if numbers else None
    items = raw if isinstance(raw, list) else [raw]
    for item in items:
        value = item.get("value") if isinstance(item, dict) else item
        if isinstance(value, bool):
            continue
        if isinstance(value, (int, float)):
            return float(value)
        if isinstance(value, str):
            try:
                return float(value.strip())
            except ValueError:
                continue
    return None


def _wrapped(raw: Any) -> bool:
    """Whether a reading is in the unit-carrying ``{"value": ...}`` form."""
    items = raw if isinstance(raw, list) else [raw]
    return any(isinstance(item, dict) and "value" in item for item in items)


def _set_number(target: Dict[str, Any], key: str, raw: Any, units: Optional[Dict[str, float]] = None) -> None:
    """Record ``key`` only when ``raw`` resolves to a real number.

    The absent-not-zero rule in one place: a field the snapshot did not carry is
    left OUT of the result, so the frontend renders an honest gap for it rather
    than a fabricated value.
    """
    number = _first_number(raw, units)
    if number is not None:
        target[key] = round(number, 2)
