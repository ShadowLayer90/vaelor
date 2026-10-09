"""Decide appliance health, and say exactly which categories were evaluated.

Health used to be computed from ``cpu_temperature`` and ``memory_percent``
alone while the interface reading it asserted "No thermal, memory, storage, or
service alerts." — four categories claimed, two evaluated. On a workstation
with a hot GPU that sentence was simply false, and it fed the sidebar badge,
the status pill and the Assistant's own health answer, so one over-claim
surfaced in three places at once.

The fix is structural rather than editorial. This returns ``checked``: the
categories that were genuinely evaluated on *this* sample, built from the
readings that were actually present. A caller enumerates that list instead of
hard-coding a sentence, so the claim grows as coverage grows and cannot
outrun it. A category whose reading is missing is not listed — "not measured"
and "measured and fine" are different answers and must not render the same.
"""

from __future__ import annotations

from typing import Any, Dict, List, Mapping, Optional, Tuple

from .platforms.gpu_temperature import (
    BAND_SOURCE_MACHINE_CLASS, SENSOR_EDGE, SENSOR_GRAPHICS_ENGINE, gpu_temperature_block,
)


#: Memory utilisation bands. Unlike temperature these are not machine-class
#: dependent: 95% of RAM in use is the same problem on every appliance.
MEMORY_WARNING_PERCENT = 85.0
MEMORY_CRITICAL_PERCENT = 95.0

#: Fallback temperature bands, used only when a driver supplies no policy.
#: The Raspberry Pi numbers, because they are the conservative pair.
DEFAULT_WARNING_C = 70.0
DEFAULT_CRITICAL_C = 80.0

#: What a machine's card says when its class is not known (VD-147 pass-2 review
#: NB-1): its processor temperature is shown but not judged - never judged on
#: another class's numbers, the Pi's included. The GPU says the same through
#: `gpu_temperature.UNKNOWN_CLASS_NO_BANDS`.
UNKNOWN_CLASS_CPU_NOT_JUDGED = (
    "Vaelor does not know what kind of machine this is, so its processor "
    "temperature is shown but not judged."
)

#: Human names for the categories, in the order a caller should list them.
PROCESSOR = "processor"
MEMORY = "memory"
GRAPHICS = "graphics"

CATEGORY_LABELS = {
    PROCESSOR: "processor",
    MEMORY: "memory",
    GRAPHICS: "graphics",
}


def _number(value: Any) -> Optional[float]:
    """A real reading, or ``None``. Booleans are not temperatures."""
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        return None
    numeric = float(value)
    return numeric if numeric == numeric else None


def _band(
    reading: Optional[float], warning: Optional[float], critical: Optional[float]
) -> Optional[str]:
    """The band a reading falls in; a mark that is not known is not applied."""
    if reading is None or (warning is None and critical is None):
        return None
    if critical is not None and reading >= critical:
        return "critical"
    if warning is not None and reading >= warning:
        return "degraded"
    return "healthy"


#: What each GPU sensor is called at the start of a health sentence, so the
#: sentence names what was measured (VD-147, owner decision D5; LESSONS 5).
GPU_SENSOR_NOUNS = {
    SENSOR_GRAPHICS_ENGINE: "Graphics-engine temperature",
    SENSOR_EDGE: "GPU edge temperature",
}


def _worst(current: str, candidate: Optional[str]) -> str:
    order = {"healthy": 0, "degraded": 1, "critical": 2}
    if candidate is None:
        return current
    return candidate if order[candidate] > order[current] else current


def evaluate_health(
    data: Optional[Mapping[str, Any]] = None,
    thermal_policy: Optional[Mapping[str, Any]] = None,
    gpu_limits: Optional[Mapping[str, Any]] = None,
    class_known: bool = True,
) -> Dict[str, Any]:
    """Status, reasons, and the categories this sample could actually judge.

    ``class_known`` False (a worker whose machine class was not recognised)
    judges NEITHER temperature against class bands: the processor's is not
    judged at all and ``notes`` says so, and the GPU's is judged only against
    limits the GPU reports for itself. The conservative fallback bands are for
    the controller's own driver with no policy, never for an unknown worker.

    ``thermal_policy`` is the machine-class policy **of the node whose sample
    this is** - never another node's. ``gpu_limits`` are the limits that node's
    GPU reports for itself, by sensor (``{}`` when it reports none).

    **Graphics is judged on the one owner's reading, against that node's own
    GPU bands** (VD-147, owner decision D5). The reading is
    `gpu_temperature_reading`'s: the graphics engine when it was read, else
    edge, and the sentence names which. The bands are, in order, the GPU's own
    limit for that sensor, else this node's machine-class bands (the
    temperatures at which an operator should act on this class of machine),
    else none - and with none, graphics is not judged and not claimed. It used
    to be judged on the edge sensor under the name "Graphics temperature", and
    for a worker against the controller's bands.
    """
    readings = dict(data or {})
    policy = dict(thermal_policy or {})
    warning_c = float(policy.get("warning_c", DEFAULT_WARNING_C))
    critical_c = float(policy.get("critical_c", DEFAULT_CRITICAL_C))

    cpu = _number(readings.get("cpu_temperature"))
    memory = _number(readings.get("memory_percent"))
    notes: List[str] = []
    if not class_known:
        policy = {}
        if cpu is not None:
            notes.append(UNKNOWN_CLASS_CPU_NOT_JUDGED)
        cpu = None
    gpu_block = gpu_temperature_block(readings, gpu_limits, policy or None, class_known=class_known)
    gpu = gpu_block["value_c"]
    gpu_bands = gpu_block["bands"] or {}
    # A reading always has a sensor; with no reading the noun is never used.
    gpu_noun = GPU_SENSOR_NOUNS.get(
        gpu_block["sensor"], "GPU temperature ({})".format(gpu_block["sensor"] or ""),
    )

    evaluated: List[Tuple[str, Optional[str], str, str]] = [
        (
            PROCESSOR,
            _band(cpu, warning_c, critical_c),
            "CPU temperature is critical",
            "CPU temperature is elevated",
        ),
        (
            MEMORY,
            _band(memory, MEMORY_WARNING_PERCENT, MEMORY_CRITICAL_PERCENT),
            "Memory utilization is critical",
            "Memory utilization is elevated",
        ),
        (
            GRAPHICS,
            _band(gpu, gpu_bands.get("warning_c"), gpu_bands.get("critical_c")),
            gpu_noun + " is critical",
            gpu_noun + " is elevated",
        ),
    ]

    status = "healthy"
    reasons: List[str] = []
    checked: List[str] = []
    for category, band, critical_reason, degraded_reason in evaluated:
        if band is None:
            # No reading means this category was not evaluated. Saying so is
            # the point: an unlisted category is one the caller must not claim.
            continue
        checked.append(CATEGORY_LABELS[category])
        status = _worst(status, band)
        if band == "critical":
            reasons.append(critical_reason)
        elif band == "degraded":
            reasons.append(degraded_reason)
    return {
        "status": status,
        "reasons": reasons,
        # Exactly what this sample judged, in the order a sentence should list
        # it. The client enumerates this rather than hard-coding categories.
        "checked": checked,
        # What was read but deliberately not judged, in a sentence each.
        "notes": notes,
        "thermal_policy": {
            "warning_c": warning_c,
            "critical_c": critical_c,
            "rationale": policy.get("rationale", ""),
            # Graphics is listed only when its bands ARE these machine-class
            # bands; a GPU with its own reported limit is judged on that.
            "applies_to": ["processor"] + (
                ["graphics"] if gpu_bands.get("source") == BAND_SOURCE_MACHINE_CLASS else []
            ),
        },
        # The GPU reading that was judged: its value, the sensor it came from,
        # the bands and where they came from, and the sentence that goes with
        # them. ``sensor`` is None when the machine reported no GPU temperature.
        "gpu_temperature": gpu_block,
    }
