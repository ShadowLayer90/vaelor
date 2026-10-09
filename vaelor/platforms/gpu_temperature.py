"""One answer to "what is this GPU's temperature, and which sensor said so?"

VD-147 (owner decision D5). A GPU on these machines has two temperature
readings and they are not the same thing:

* the **graphics engine** temperature, which ``amd-smi`` reports as
  ``apu_temperature_gfx`` and the telemetry stores as ``gpu_gfx_temperature_c``;
* the **edge** temperature, the kernel's hwmon ``temp1_input`` (label ``edge``),
  stored as ``gpu_temperature_c``.

Every screen used to show the edge sensor under a label that named no sensor,
and the profiler showed whichever field it found first (LESSONS 5: a label
that does not match what was measured). :func:`gpu_temperature_reading` is the one place that
chooses: the graphics engine when it was read, otherwise edge, otherwise
nothing - and it always hands back the sensor's name with the value, so the
sentence on the screen can say which it is.

Standard library only: nothing here reaches a subprocess or a third party.
"""

from __future__ import annotations

import math
from typing import Any, Dict, Mapping, Optional, Tuple

from ..telemetry_bounds import plausible_temperature_limit

#: The sensor names, as the owner reads them. This module owns the words; a
#: payload carries one of them beside every GPU temperature.
SENSOR_GRAPHICS_ENGINE = "graphics engine"
SENSOR_EDGE = "edge"
#: A sensor the driver did not label (review S-20): not called "edge" - that is
#: amdgpu's name for one particular sensor, and another driver's unlabelled
#: temp1 may be anything.
SENSOR_UNLABELLED = "unlabelled sensor"
#: A discrete card's junction sensor. Only the on-demand profile reads it, and
#: only on a part known not to be integrated (an integrated GPU has none).
SENSOR_HOTSPOT = "hotspot"

#: The stored telemetry field behind each sensor. ``gpu_temperature_c`` keeps
#: its name for compatibility; it has always been the edge sensor.
GPU_GFX_TEMPERATURE_KEY = "gpu_gfx_temperature_c"
GPU_EDGE_TEMPERATURE_KEY = "gpu_temperature_c"

#: The key that carries the hwmon sensor's own label beside its reading.
GPU_TEMPERATURE_LABEL_KEY = "gpu_temperature_label"

#: Driver labels that have an owner word. Any other label is kept as the
#: driver wrote it (review S-20): a sensor is named as its driver names it.
_LABEL_WORDS = {"edge": SENSOR_EDGE, "junction": SENSOR_HOTSPOT, "hotspot": SENSOR_HOTSPOT}
#: The longest sensor name kept from a driver label.
_LABEL_MAX = 32


def sensor_from_label(label: Any) -> str:
    """The sensor name for a hwmon ``temp1_label``: the owner word, else the label itself.

    No label at all is :data:`SENSOR_UNLABELLED`, never a guessed ``edge``
    (review S-20). A label is lower-cased, stripped of
    anything unprintable and cut to :data:`_LABEL_MAX` characters: it is data a
    driver wrote, shown on a screen.
    """
    cleaned = "".join(ch for ch in str(label or "") if ch.isprintable()).strip().lower()[:_LABEL_MAX]
    if not cleaned:
        return SENSOR_UNLABELLED
    return _LABEL_WORDS.get(cleaned, cleaned)


def _reading(value: Any) -> Optional[float]:
    """A finite number, or ``None``. A boolean is not a temperature."""
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        return None
    try:
        number = float(value)
    except (OverflowError, ValueError):
        return None
    return number if math.isfinite(number) else None


def gpu_temperature_reading(
    readings: Optional[Mapping[str, Any]], edge_label: Any = None,
) -> Tuple[Optional[float], Optional[str]]:
    """``(degrees Celsius, sensor)`` for a node's telemetry row, or ``(None, None)``.

    ``readings`` is a node's newest telemetry (the controller's live sample or a
    worker's stored row). The graphics-engine reading wins when it is present;
    the hwmon sensor is the fallback, named by the driver's label - the row's
    own :data:`GPU_TEMPERATURE_LABEL_KEY`, else ``edge_label`` (a worker's
    enrolment record), else ``edge``. A row with neither reading has no GPU
    temperature, which is an absence and never a zero.
    """
    row = readings if isinstance(readings, Mapping) else {}
    value = _reading(row.get(GPU_GFX_TEMPERATURE_KEY))
    if value is not None:
        return value, SENSOR_GRAPHICS_ENGINE
    value = _reading(row.get(GPU_EDGE_TEMPERATURE_KEY))
    if value is not None:
        return value, sensor_from_label(row.get(GPU_TEMPERATURE_LABEL_KEY) or edge_label)
    return None, None


#: Where a node's GPU temperature bands came from. This module owns the words.
BAND_SOURCE_GPU = "gpu"
BAND_SOURCE_MACHINE_CLASS = "machine class"

#: What the screen says when neither the GPU nor its machine's class gives a
#: limit: there are no marks to draw and nothing to judge the reading against.
NO_GPU_TEMPERATURE_LIMIT = "No temperature limit is reported for this GPU."

#: What the screen says when the GPU reports no limit and Vaelor does not know
#: what class of machine this is: no bands are drawn and the reading is not
#: judged - never a generic class's numbers presented as this machine's
#: (review S-19).
UNKNOWN_CLASS_NO_BANDS = (
    "No GPU limit is reported and Vaelor does not know what kind of machine "
    "this is, so this temperature is shown but not judged."
)

#: What the screen says when the bands are the machine class's, not the GPU's.
MACHINE_CLASS_BANDS_NOTE = (
    "No GPU limit reported; using this machine's processor limits."
)


def gpu_temperature_bands(
    sensor: Optional[str],
    limits: Optional[Mapping[str, Any]] = None,
    machine_policy: Optional[Mapping[str, Any]] = None,
) -> Optional[Dict[str, Any]]:
    """The warning and critical marks for ONE node's GPU reading, or ``None``.

    Bands are per node and per sensor (owner decision D5):

    1. the limits the GPU reports for **that sensor** (``limits`` is
       ``{sensor: {warning_c, critical_c}}``, from hwmon or the sampler), with
       ``source`` :data:`BAND_SOURCE_GPU`. A graphics-engine reading is never
       judged against edge limits, or the reverse;
    2. otherwise **that node's own** machine-class bands (``machine_policy`` is
       `platforms.base.thermal_policy` for the node's class - never another
       node's), with ``source`` :data:`BAND_SOURCE_MACHINE_CLASS`;
    3. otherwise ``None``: no marks, and the reading is not judged.

    A mark that is missing is ``None`` inside the result, never a default.
    """
    if not sensor:
        return None
    own = limits.get(sensor) if isinstance(limits, Mapping) else None
    if isinstance(own, Mapping):
        warning, critical = _reading(own.get("warning_c")), _reading(own.get("critical_c"))
        if warning is not None or critical is not None:
            return {"warning_c": warning, "critical_c": critical, "source": BAND_SOURCE_GPU}
    if isinstance(machine_policy, Mapping):
        warning = _reading(machine_policy.get("warning_c"))
        critical = _reading(machine_policy.get("critical_c"))
        if warning is not None and critical is not None:
            return {
                "warning_c": warning, "critical_c": critical,
                "source": BAND_SOURCE_MACHINE_CLASS,
            }
    return None


def gpu_temperature_block(
    readings: Optional[Mapping[str, Any]],
    limits: Optional[Mapping[str, Any]] = None,
    machine_policy: Optional[Mapping[str, Any]] = None,
    class_known: bool = True,
) -> Dict[str, Any]:
    """A node's GPU temperature as a payload block: value, sensor, bands, note.

    ``note`` is the sentence that goes with the bands: empty when they are the
    GPU's own, :data:`MACHINE_CLASS_BANDS_NOTE` when they are the machine
    class's, and :data:`NO_GPU_TEMPERATURE_LIMIT` when there are none. A node
    with no reading has no sensor, no bands and no note.
    """
    value, sensor = gpu_temperature_reading(readings)
    bands = gpu_temperature_bands(sensor, limits, machine_policy)
    if sensor is None:
        note = ""
    elif bands is None:
        note = NO_GPU_TEMPERATURE_LIMIT if class_known else UNKNOWN_CLASS_NO_BANDS
    elif bands["source"] == BAND_SOURCE_MACHINE_CLASS:
        note = MACHINE_CLASS_BANDS_NOTE
    else:
        note = ""
    return {"value_c": value, "sensor": sensor, "bands": bands, "note": note}


def limits_from_records(records: Any) -> Dict[str, Dict[str, float]]:
    """The primary GPU's own temperature limits by sensor, or ``{}`` when it reports none.

    ``records`` is `accelerators.discover_accelerators`'s answer. The hwmon
    sensor's label names the sensor (``edge``) and the limits are the ones the
    driver publishes for it. On the Strix Halo boxes the driver publishes none
    (S0: no ``temp1_crit``, every ``amd-smi static`` limit N/A), and the result
    is empty - never an invented limit.
    """
    if not isinstance(records, (list, tuple)) or not records:
        return {}
    primary = records[0] if isinstance(records[0], Mapping) else {}
    limits = primary.get("temperature_limits")
    if not isinstance(limits, Mapping) or not limits:
        return {}
    # Keyed by the same name the reading carries (review S-20), and bounded
    # like every other limit (review S-17): a published 0 is not a limit.
    kept = {
        name: value for name in ("warning_c", "critical_c")
        for value in (plausible_temperature_limit(limits.get(name)),) if value is not None
    }
    return {sensor_from_label(primary.get("temperature_label")): kept} if kept else {}


def limits_from_driver(driver: Any) -> Dict[str, Dict[str, float]]:
    """:func:`limits_from_records` for a platform driver's discovered accelerators."""
    discover = getattr(driver, "accelerators", None)
    if not callable(discover):
        return {}
    try:
        return limits_from_records(discover())
    except (OSError, TypeError, ValueError):
        return {}
