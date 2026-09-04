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


#: Memory utilisation bands. Unlike temperature these are not machine-class
#: dependent: 95% of RAM in use is the same problem on every appliance.
MEMORY_WARNING_PERCENT = 85.0
MEMORY_CRITICAL_PERCENT = 95.0

#: Fallback temperature bands, used only when a driver supplies no policy.
#: The Raspberry Pi numbers, because they are the conservative pair.
DEFAULT_WARNING_C = 70.0
DEFAULT_CRITICAL_C = 80.0

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
    reading: Optional[float], warning: float, critical: float
) -> Optional[str]:
    if reading is None:
        return None
    if reading >= critical:
        return "critical"
    if reading >= warning:
        return "degraded"
    return "healthy"


def _worst(current: str, candidate: Optional[str]) -> str:
    order = {"healthy": 0, "degraded": 1, "critical": 2}
    if candidate is None:
        return current
    return candidate if order[candidate] > order[current] else current


def evaluate_health(
    data: Optional[Mapping[str, Any]] = None,
    thermal_policy: Optional[Mapping[str, Any]] = None,
) -> Dict[str, Any]:
    """Status, reasons, and the categories this sample could actually judge.

    Graphics is evaluated against the same ``thermal_policy`` bands the driver
    already serves for the processor. That is deliberate and not laziness:
    those bands are the machine class's "an operator should act" temperatures,
    they are published to the client already, and inventing a second pair for
    the GPU would be asserting numbers nobody measured. When a GPU temperature
    is present it is judged; when it is absent, graphics is not claimed.
    """
    readings = dict(data or {})
    policy = dict(thermal_policy or {})
    warning_c = float(policy.get("warning_c", DEFAULT_WARNING_C))
    critical_c = float(policy.get("critical_c", DEFAULT_CRITICAL_C))

    cpu = _number(readings.get("cpu_temperature"))
    memory = _number(readings.get("memory_percent"))
    gpu = _number(readings.get("gpu_temperature_c"))

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
            _band(gpu, warning_c, critical_c),
            "Graphics temperature is critical",
            "Graphics temperature is elevated",
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
        "thermal_policy": {
            "warning_c": warning_c,
            "critical_c": critical_c,
            "rationale": policy.get("rationale", ""),
            # Named so a reader can tell that a GPU at 95 °C is being judged
            # against the processor's published bands rather than a second set
            # of numbers invented for it.
            "applies_to": ["processor", "graphics"],
        },
    }
