"""The primary GPU's live telemetry keys: which reading is which, and where each comes from.

Moved out of `platforms.accelerators` at the thousand-line limit (VD-147 review);
`accelerators` re-exports :func:`accelerator_telemetry`, so every caller keeps
working. Discovery stays there; this module only decides what a discovered
record's readings may be stored AS.

Two decisions live here, each once:

* **GPU power** (:func:`hwmon_power_is_gpu_power`). The kernel's hwmon
  ``power1_average`` is the board's power on a discrete card and the whole
  package's (processor and graphics together) on an integrated one. Storing
  the package figure as ``gpu_power_watts`` is the LESSONS 4 / VD-040
  substitution. On an AMD integrated part in the device table the graphics
  engine's own power comes from ``amd-smi``; on any other AMD part the hwmon
  figure is stored only when the driver also publishes a board power cap
  (``power1_cap``), which a discrete board has and an integrated part does not
  (review S-16). Otherwise no GPU power is stored: an absence, never the
  package number.
* **Which temperature, by which name** - through `platforms.gpu_temperature`,
  with the hwmon sensor named by the driver's own label (review S-20).
"""

from __future__ import annotations

import shutil
import subprocess
from typing import Any, Callable, Dict, List, Optional

from ..telemetry_bounds import POWER_CAP_KEY, REFUSED_FIELDS_KEY, plausible
from .gpu_temperature import SENSOR_GRAPHICS_ENGINE, gpu_temperature_reading
from .graphics_software import GPU_GFX_POWER_FIELD, integrated_amd_gpu

#: The telemetry key that carries the hwmon sensor's own label, so a reading of
#: that sensor is named as the driver names it.
GPU_TEMPERATURE_LABEL_KEY = "gpu_temperature_label"
#: The `gpu_power_source` a discrete card's hwmon reading is stored under; the
#: Assistant names it "board" from this one spelling (ACC-201).
HWMON_POWER_SOURCE = "hwmon power1_average"
#: What measured a GPU power reading, sent beside it (`gpu_power_sensor`) so
#: every screen names it from this owner: a discrete card's hwmon figure is
#: the board's; a reading the vendor tool took is the graphics engine's own.
POWER_SENSOR_BOARD = "board"


def gpu_power_sensor(row: Any) -> Optional[str]:
    """The name of what measured a telemetry row's GPU power, or None when nothing says.

    Decided from what a STORED row keeps, so a worker's row says the same as
    the controller's live sample (review 2 of ACC-204): the ingest path keeps
    numbers only, so the sensor word and ``gpu_power_source`` never reach the
    store, but the vendor-status code does. :data:`NOT_APPLICABLE` is set only
    where the power came from hwmon (a discrete card), a code that carries
    readings means the vendor tool read the graphics engine, and a row with
    neither (an agent from before the code) names no sensor rather than guess.
    """
    from ..gpu_vendor_status import CODES_WITH_READINGS, NOT_APPLICABLE, STATUS_FIELD, known_code

    row = row if isinstance(row, dict) else {}
    power = row.get("gpu_power_watts")
    if isinstance(power, bool) or not isinstance(power, (int, float)):
        return None
    if row.get("gpu_power_sensor"):
        return str(row["gpu_power_sensor"])
    # A hwmon reading always comes with NOT_APPLICABLE (below), so its source
    # name decides nothing on its own (review 3, M4).
    if row.get("gpu_power_source") == GPU_GFX_POWER_FIELD:
        return SENSOR_GRAPHICS_ENGINE
    code = known_code(row.get(STATUS_FIELD))
    if code == NOT_APPLICABLE:
        return POWER_SENSOR_BOARD
    return SENSOR_GRAPHICS_ENGINE if code in CODES_WITH_READINGS else None


#: AMD parts confirmed to be discrete boards, by PCI vendor and device id: their
#: hwmon power is the board's own. Only a part listed here has its hwmon power
#: stored as GPU power (pass-2 review SC-C): an integrated part's hwmon power is
#: the whole package, and neither "not in the integrated table" nor "publishes
#: a power cap" proves a part is a board - an APU can publish a cap. A card
#: added here is one somebody has seen enumerate as a discrete board.
DISCRETE_AMD_DEVICES = {
    ("0x1002", "0x744c"): "Radeon RX 7900 XT / XTX (Navi 31)",
    ("0x1002", "0x7480"): "Radeon RX 7600 (Navi 33)",
}


def hwmon_power_is_gpu_power(record: Dict[str, Any]) -> bool:
    """Whether a record's hwmon power may be stored as the GPU's power (reviews S-16, SC-C).

    A non-AMD part: yes. An AMD part: only when the device table confirms it
    is a discrete board (:data:`DISCRETE_AMD_DEVICES`); otherwise the reading
    is left out, never stored as the GPU's.
    """
    if record.get("vendor") != "AMD":
        return discrete_non_amd_gpu(record)
    key = (str(record.get("vendor_id") or "").lower(), str(record.get("device_id") or "").lower())
    return key in DISCRETE_AMD_DEVICES


#: NVIDIA's PCI vendor id: its graphics devices on a PC are discrete cards.
NVIDIA_VENDOR_ID = "0x10de"


def discrete_non_amd_gpu(record: Dict[str, Any]) -> bool:
    """Whether a non-AMD part is shown to be a discrete card (review 3, nit 7).

    Its hwmon power is the board's only then. An integrated part's hwmon power
    is the package's, so a part that may be integrated - unified memory, or an
    Intel part with no memory of its own - stores no GPU power, and its screens
    name no sensor rather than call the package "board".
    """
    if record.get("unified_memory") is True:
        return False
    if str(record.get("vendor_id") or "").lower() == NVIDIA_VENDOR_ID:
        return True
    vram = record.get("vram_total_bytes")
    return isinstance(vram, (int, float)) and not isinstance(vram, bool) and vram > 0


def accelerator_telemetry(
    accelerators: Optional[List[Dict[str, Any]]] = None,
    sys_root: str = "/sys",
    finder: Callable[[str], Optional[str]] = shutil.which,
    runner: Callable[..., Any] = subprocess.run,
    *,
    now: Optional[float] = None,
    vendor: Optional[Callable[[], Dict[str, Any]]] = None,
) -> Dict[str, Any]:
    """Return the ``gpu_*`` telemetry keys for the primary accelerator.

    Keys are omitted entirely when the underlying sysfs attribute is absent.
    A missing sensor is not zero, and reporting ``0`` for an unreadable GPU
    temperature is how a dashboard ends up claiming a running GPU is at 0 °C.

    Power is sourced by capability (see :func:`_add_gpu_power`): hwmon for a
    discrete card, the ``amd-smi`` graphics channel for an integrated Radeon.
    ``finder`` and ``runner`` reach that tool and let a test inject it; the poll
    path takes the default cached reader. ``vendor`` replaces that read whole: a
    worker's emitter never runs the tool and hands in its sampler's reading.
    """
    from .accelerators import discover_accelerators

    records = (
        discover_accelerators(sys_root) if accelerators is None else accelerators
    )
    if not records:
        return {}
    primary = records[0]
    mapping = (
        ("gpu_temperature_c", "temperature_c"),
        ("gpu_clock_mhz", "clock_mhz"),
        ("gpu_busy_percent", "busy_percent"),
        ("gpu_vram_used_bytes", "vram_used_bytes"),
        ("gpu_vram_total_bytes", "vram_total_bytes"),
        ("gpu_gtt_used_bytes", "gtt_used_bytes"),
        ("gpu_gtt_total_bytes", "gtt_total_bytes"),
    )
    telemetry: Dict[str, Any] = {}
    for key, source in mapping:
        value = primary.get(source)
        if value is not None:
            telemetry[key] = value
    if "gpu_temperature_c" in telemetry and primary.get("temperature_label"):
        telemetry[GPU_TEMPERATURE_LABEL_KEY] = str(primary["temperature_label"])
    _add_gpu_power(telemetry, primary, finder, runner, now, vendor)
    power_sensor = gpu_power_sensor(telemetry)
    if power_sensor:  # what measured the power, by name (review of ACC-204)
        telemetry["gpu_power_sensor"] = power_sensor
    sensor = gpu_temperature_reading(telemetry)[1]
    if sensor:  # which of the two temperatures the screens show, by name
        telemetry["gpu_temperature_sensor"] = sensor
    if telemetry:
        telemetry["gpu_name"] = primary.get("name")
    return telemetry


def _add_gpu_power(
    telemetry: Dict[str, Any],
    primary: Dict[str, Any],
    finder: Callable[[str], Optional[str]],
    runner: Callable[..., Any],
    now: Optional[float],
    vendor: Optional[Callable[[], Dict[str, Any]]] = None,
) -> None:
    """Set ``gpu_power_watts`` from the channel that is correct for this part.

    On an integrated Radeon, ``amd-smi`` graphics-engine power is the engine's
    own draw; the hwmon ``power1_average`` is SoC package power there and must
    NOT stand in for it (LESSONS 4 / VD-040). When the gfx channel is
    unreadable the reading is omitted, never back-filled with the package
    number. The graphics-engine temperature and the status code saying whether
    the tool was read come from the same cached call (VD-147).
    """
    # Imported here: `gpu_vendor_status` reads this package's own modules.
    from ..gpu_vendor_status import NOT_APPLICABLE, STATUS_FIELD, controller_vendor_fields
    from .accelerators import cached_vendor_tool_metrics

    if integrated_amd_gpu(primary):
        if vendor is None:
            metrics = cached_vendor_tool_metrics([primary], finder, runner, now=now)
            telemetry.update(controller_vendor_fields(metrics))
        else:
            telemetry.update(vendor())
        return
    telemetry[STATUS_FIELD] = NOT_APPLICABLE
    power = primary.get("power_watts")
    cap = primary.get("power_cap_watts")
    if power is not None and hwmon_power_is_gpu_power(primary) and not plausible("gpu_power_watts", power, cap):
        # Refused against its own cap: not shown, and counted where the row is
        # built, so the owner is told a reading was discarded (pass-3 review).
        telemetry.setdefault(REFUSED_FIELDS_KEY, []).append("gpu_power_watts")
    elif power is not None and hwmon_power_is_gpu_power(primary):
        telemetry["gpu_power_watts"] = power
        telemetry["gpu_power_source"] = HWMON_POWER_SOURCE
        if cap is not None:
            # Carried with the reading so the store's own bound (telemetry_flatten)
            # applies the same cap (review SC-C).
            telemetry[POWER_CAP_KEY] = cap
