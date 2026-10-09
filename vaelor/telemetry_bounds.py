"""Physical-plausibility bounds for stored telemetry: one table, every node.

VD-147 (review S8, N-S10). A sensor can hand back garbage: a unit error
(milliwatts read as watts), an overflow, an uninitialised register. Stored as
it stands, that one number bends every chart and average that includes it. So a
bounded field whose value is outside its range is **dropped** - never clamped,
because a clamped value is an invented reading - and the drop is counted so the
owner is told.

**These are limits on what is physically possible, not on what is usual.** The
table rejects sensor garbage only. It holds no figure taken from the machines
Vaelor happens to run on today: a discrete GPU drawing 450 W at 88 C is
ordinary and must be stored. A bound tight enough to "look right" on a Strix
Halo would silently delete the first larger machine's readings, which is the
"declared unavailable, actually readable" defect (LESSONS 4) with a number in
place of a sentence.

One table for both sides. It is applied where a machine reading becomes a
stored row (`telemetry_flatten.flatten_storable`, which the controller's writer
and the worker emitter share) and checked again at the ingest route, as a
defence against a modified emitter.

Standard library only: this module rides in the worker emitter's bundle.
"""

from __future__ import annotations

import math
from typing import Any, Dict, List, Mapping, Optional, Tuple

#: Power, in watts. 2 kW is past any single accelerator or processor package
#: Vaelor can be pointed at; a reading above it is a unit error or an overflow.
POWER_MIN_WATTS = 0.0
POWER_MAX_WATTS = 2000.0

#: When the device reports its own power cap, a reading above this many times
#: the cap is implausible. Twice, because a smoothed reading can briefly sit
#: above a cap the firmware enforces on an average.
POWER_CAP_FACTOR = 2.0

#: Temperature, in degrees Celsius. Silicon does not run below -40 or survive
#: 150; a reading outside is a disconnected or mis-scaled sensor.
TEMPERATURE_MIN_C = -40.0
TEMPERATURE_MAX_C = 150.0

#: A temperature LIMIT (a warning or critical mark the hardware reports for
#: itself). A limit at or below freezing is not a limit anything enforces.
TEMPERATURE_LIMIT_MIN_C = 0.0
TEMPERATURE_LIMIT_MAX_C = 150.0

#: The age of a reading, in seconds: never negative, and older than a day is a
#: clock fault rather than an age.
AGE_MIN_SECONDS = 0.0
AGE_MAX_SECONDS = 86400.0

#: Bytes on one filesystem (VD-205 item 6): 2**60 (one exbibyte) is past any
#: single filesystem a machine Vaelor manages can mount; above it is an overflow.
DISK_MIN_BYTES = 0.0
DISK_MAX_BYTES = float(2 ** 60)

#: Network throughput summed over a machine's NICs, in bytes per second: 10**12
#: is 8 Tbit/s, past any set of interfaces one machine carries.
NETWORK_MIN_BYTES_PER_SECOND = 0.0
NETWORK_MAX_BYTES_PER_SECOND = 1e12

#: Fan speed in RPM. No fan turns at 100,000; a tachometer register reading
#: above it is garbage. 0 is a stopped fan the tachometer saw, and is real.
FAN_MIN_RPM = 0.0
FAN_MAX_RPM = 100000.0

#: How many fans gave a reading. At least 1 when sent: with no readable fan the
#: emitter sends no fan fields, so a 0 here is a zero standing in for "not
#: read" and is refused (LESSONS 8).
FAN_COUNT_MIN = 1.0
FAN_COUNT_MAX = 256.0

_POWER = (POWER_MIN_WATTS, POWER_MAX_WATTS)
_DISK = (DISK_MIN_BYTES, DISK_MAX_BYTES)
_NETWORK = (NETWORK_MIN_BYTES_PER_SECOND, NETWORK_MAX_BYTES_PER_SECOND)
_TEMPERATURE = (TEMPERATURE_MIN_C, TEMPERATURE_MAX_C)
_AGE = (AGE_MIN_SECONDS, AGE_MAX_SECONDS)

#: Every bounded field and its inclusive range. A field not listed here is not
#: bounded; adding a stored power, temperature, age, disk, network or fan field
#: means adding it here.
BOUNDS: Dict[str, Tuple[float, float]] = {
    "gpu_power_watts": _POWER,
    "gpu_socket_power_watts": _POWER,
    "cpu_package_power_watts": _POWER,
    "npu_power_watts": _POWER,
    "gpu_temperature_c": _TEMPERATURE,
    "gpu_gfx_temperature_c": _TEMPERATURE,
    "cpu_temperature": _TEMPERATURE,
    "gpu_vendor_age_seconds": _AGE,
    "disk_root_total_bytes": _DISK,
    "disk_root_used_bytes": _DISK,
    "disk_root_free_bytes": _DISK,
    "disk_data_total_bytes": _DISK,
    "disk_data_used_bytes": _DISK,
    "disk_data_free_bytes": _DISK,
    "net_rx_bytes_per_second": _NETWORK,
    "net_tx_bytes_per_second": _NETWORK,
    "fan_rpm": (FAN_MIN_RPM, FAN_MAX_RPM),
    "fan_count": (FAN_COUNT_MIN, FAN_COUNT_MAX),
}

#: Fields that hold a code rather than a measurement, and the codes they may
#: hold. ``gpu_vendor_status`` is the status table of `gpu_vendor_status` (0-8);
#: a test pins the two together, and the set is restated here rather than
#: imported so the ingest validator stays free of the platform readers.
ALLOWED_VALUES: Dict[str, frozenset] = {
    "gpu_vendor_status": frozenset(float(code) for code in range(9)),
}

#: The power fields, the only ones a reported cap can tighten.
_POWER_FIELDS = frozenset(name for name, bound in BOUNDS.items() if bound is _POWER)

#: The sentence an owner reads when readings were discarded, on the machine's
#: card (review S-15). ``{count}`` and ``{node}`` are filled by
#: :func:`implausible_note`.
IMPLAUSIBLE_READINGS = "{count} readings from {node} were outside physical limits and were not stored."
IMPLAUSIBLE_READING_ONE = "1 reading from {node} was outside physical limits and was not stored."


def implausible_note(count: Any, node: str) -> str:
    """The card's sentence for ``count`` discarded readings, or ``""`` for none."""
    number = int(count) if isinstance(count, int) and not isinstance(count, bool) else 0
    if number <= 0:
        return ""
    template = IMPLAUSIBLE_READING_ONE if number == 1 else IMPLAUSIBLE_READINGS
    return template.format(count=number, node=node or "an unnamed machine")


def finite_number(value: Any) -> Optional[float]:
    """``value`` as a finite float, or ``None``. Total: never raises.

    A JSON document can hold NaN, infinity and integers too large for a float
    (review B2); each of those is "not a usable number", not an exception.
    """
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        return None
    try:
        number = float(value)
    except (OverflowError, ValueError):
        return None
    return number if math.isfinite(number) else None


_finite = finite_number


def _is_number(value: Any) -> bool:
    return isinstance(value, (int, float)) and not isinstance(value, bool)


def bound_for(field: str, power_cap_watts: Optional[float] = None) -> Optional[Tuple[float, float]]:
    """The inclusive range for ``field``, or ``None`` when it is not bounded.

    ``power_cap_watts`` is the cap the device reports for itself, when it
    reports one; it lowers the ceiling of a power field to
    :data:`POWER_CAP_FACTOR` times that cap and never raises it past
    :data:`POWER_MAX_WATTS`. A cap that is itself not a positive finite number
    is ignored.
    """
    bound = BOUNDS.get(field)
    if bound is None:
        return None
    cap = _finite(power_cap_watts)
    if field in _POWER_FIELDS and cap is not None and cap > 0:
        return bound[0], min(bound[1], POWER_CAP_FACTOR * cap)
    return bound


def plausible(field: str, value: Any, power_cap_watts: Optional[float] = None) -> bool:
    """Whether ``value`` may be stored under ``field``.

    An unbounded field is always plausible; so is a value that is not a number
    (a label, a name), because the bounds speak only about measurements. A
    coded field (:data:`ALLOWED_VALUES`) must hold one of its codes. A NUMBER
    that is not finite (NaN, infinity, an integer past a float's range) is
    never a plausible measurement of a bounded field.
    """
    number = _finite(value)
    allowed = ALLOWED_VALUES.get(field)
    if allowed is not None:
        return number is not None and number in allowed
    bound = bound_for(field, power_cap_watts)
    if bound is None:
        return True
    if number is None:
        return not _is_number(value)
    return bound[0] <= number <= bound[1]


#: The telemetry key a reading's own reported power cap travels under, so the
#: flatten seam applies it to that row's power fields (pass-2 review SC-C).
POWER_CAP_KEY = "gpu_power_cap_watts"

#: The telemetry key under which a reader lists the fields it refused before
#: the row was built (a power reading past its own cap). A list, so it is never
#: stored; the flatten seam adds it to the dropped names it counts (pass-3 review).
REFUSED_FIELDS_KEY = "refused_fields"

#: The line-protocol field a worker's emitter reports its own discards in: a
#: count, never stored. The ingest route adds it to the machine's implausible
#: count, so a discard on the worker is not a silent gap on the controller.
DISCARDED_COUNT_FIELD = "implausible_discarded"
#: The name recorded for each of those discards (the worker names no field).
DISCARDED_ON_MACHINE = "discarded on the machine"
#: The most discards one line may report; a larger number is a malformed line.
MAX_DISCARDED_PER_LINE = 64


def apply_bounds(
    row: Mapping[str, Any], power_cap_watts: Optional[float] = None,
) -> Tuple[Dict[str, Any], List[str]]:
    """``(kept, dropped)``: the row without its implausible values, and their names.

    Nothing is clamped. A dropped field is simply absent from ``kept``, which
    every reader already treats as "not measured".
    """
    kept: Dict[str, Any] = {}
    dropped: List[str] = []
    for field, value in row.items():
        if plausible(field, value, power_cap_watts):
            kept[field] = value
        else:
            dropped.append(field)
    return kept, dropped


def plausible_temperature_limit(value: Any) -> Optional[float]:
    """A reported temperature limit in Celsius, or ``None`` when implausible."""
    number = _finite(value)
    if number is None or not TEMPERATURE_LIMIT_MIN_C < number <= TEMPERATURE_LIMIT_MAX_C:
        return None
    return number
