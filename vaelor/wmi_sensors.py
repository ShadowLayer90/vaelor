"""HP WMI sensors: fans and board temperatures the kernel already publishes.

``hp-wmi-sensors`` ships with the distribution and was simply never loaded. It
is a *different* driver from ``hp-wmi``: probing as an unprivileged user found
``hwmon`` node ``hp`` exposing no inputs, and Vaelor concluded that HP keeps
everything in the embedded controller. It does not. Loading the module creates
a node named ``hp_wmi_sensors`` carrying three labelled fans and seven labelled
temperatures, so a machine with three readable fans was being told it had none.

Two rules this module exists to hold:

**Match on name, never on index.** The node was ``hwmon10`` when the module was
loaded by hand, and the number moves with probe order. Anything keyed to an
index is a reading of whatever happened to enumerate there.

**Identical readings are not corroborated readings.** Five of the seven
temperatures read exactly 31.0 °C on an idle machine. That is plausible near
ambient and it is also exactly what a BIOS returning a placeholder for an
unimplemented channel looks like, and those two possibilities are
indistinguishable from a single sample. This module therefore reports such
channels as *read but not corroborated*, and only marks a channel distinct once
it has actually been observed to differ from its peers. Presenting seven
identical numbers as seven measurements would be the same defect this codebase
has spent weeks removing, in a new place.
"""

from __future__ import annotations

import json
import re
import threading
import time
from pathlib import Path
from typing import Any, Dict, List, Optional

from .runtime_paths import state_path


#: The hwmon ``name`` the driver publishes. Matched exactly, case-insensitively.
WMI_SENSOR_NAME = "hp_wmi_sensors"

#: The kernel module that creates it, and the WMI GUID whose presence means the
#: firmware actually offers these sensors. The installer uses both.
WMI_SENSOR_MODULE = "hp-wmi-sensors"
WMI_SENSOR_GUID = "5FB7F034-2C63-45e9-BE91-3D44E2C707E4"

#: Channels reading the *same* value as at least this many peers are treated as
#: uncorroborated until one of them is seen to move independently. Two channels
#: agreeing is unremarkable; five agreeing to the milli-degree is the shape of a
#: firmware placeholder.
PLACEHOLDER_GROUP_SIZE = 3

#: Channels confirmed to be real sensors by a load test on the target hardware,
#: with the range each moved. Three concurrent generations on
#: ``gpt-oss-20b-GGUF`` drove GPU 0 to 95% busy; 22 samples over 63 s.
#:
#: The uniform 31 °C at idle was a genuinely idle machine near ambient, not a
#: firmware placeholder. Withholding these until they were shown to move was
#: the right default and the measurement lifts it — a fresh appliance seeds
#: from this rather than making every owner load their own machine to earn
#: readings someone has already earned.
#:
#: **The range is provenance, not a threshold.** Ambient and the M.2 areas moved
#: 1–2 °C under a load that moved the CPU 28 °C: they are measuring a different
#: thing, slowly, not measuring badly. Any rule that discarded a channel for a
#: small range would throw away four real sensors, so no such rule exists here
#: and :func:`_mark_corroboration` never consults these numbers.
CONFIRMED_DISTINCT_CHANNELS = {
    "CPU": {"idle_c": 32.0, "load_c": 60.0, "range_c": 28.0},
    "CPU VDDCR VR": {"idle_c": 32.0, "load_c": 41.0, "range_c": 9.0},
    "CPU VDDCR_CCD VR": {"idle_c": 32.0, "load_c": 37.0, "range_c": 5.0},
    "M.2 SSD1 Area": {"idle_c": 31.0, "load_c": 33.0, "range_c": 2.0},
    "M.2 SSD2 Area": {"idle_c": 32.0, "load_c": 34.0, "range_c": 2.0},
    "North Ambient": {"idle_c": 31.0, "load_c": 33.0, "range_c": 2.0},
    "South Ambient": {"idle_c": 30.0, "load_c": 31.0, "range_c": 1.0},
}

#: How that measurement was taken, so the next person can repeat it rather than
#: trusting the table.
CONFIRMATION_METHOD = (
    "Three concurrent generations on gpt-oss-20b-GGUF drove GPU 0 to 95% busy; "
    "22 samples over 63 s. Script: tools/z2_thermal.sh."
)

#: A fan whose speed does not change under a partial load has not thereby shown
#: itself to be broken. The power-supply fan held ~2160 RPM throughout a
#: GPU-only load, which is what a power-supply fan should do; moving it would
#: need a sustained whole-system load. Recorded so nobody later reads a flat
#: tachometer as a dead one.
FLAT_FAN_IS_NOT_EVIDENCE = (
    "A fan holding a steady speed under a partial load is behaving correctly. "
    "Only a fan reporting a fault, or one that never reads at all, is evidence "
    "of a problem."
)

_NUMBER = re.compile(r"-?\d+")


def _read(path: Path) -> str:
    try:
        return path.read_text(encoding="utf-8", errors="replace").strip("\x00\n ")
    except OSError:
        return ""


def _integer(path: Path) -> Optional[int]:
    match = _NUMBER.fullmatch(_read(path))
    return int(match.group()) if match else None


def find_node(sys_root: str = "/sys") -> Optional[Path]:
    """The hwmon directory whose ``name`` is ``hp_wmi_sensors``, or ``None``.

    By name, because the index moved the moment the module was loaded and will
    move again on any probe-order change.
    """
    try:
        entries = sorted((Path(sys_root) / "class" / "hwmon").iterdir())
    except OSError:
        return None
    for entry in entries:
        if _read(entry / "name").lower() == WMI_SENSOR_NAME:
            return entry
    return None


def _channels(node: Path, prefix: str, scale: int) -> List[Dict[str, Any]]:
    readings: List[Dict[str, Any]] = []
    try:
        inputs = sorted(node.glob("{}*_input".format(prefix)))
    except OSError:
        return readings
    for sensor in inputs:
        raw = _integer(sensor)
        if raw is None:
            continue
        stem = sensor.name[: -len("_input")]
        # Every channel carries a fault flag. A faulted channel is reported as
        # faulted rather than dropped: "this sensor says it is broken" is a
        # different fact from "this sensor is not there", and an operator
        # chasing a dead fan needs the first one.
        fault = _integer(node / "{}_fault".format(stem))
        readings.append({
            "channel": stem,
            "label": _read(node / "{}_label".format(stem)) or stem,
            "value": raw / scale if scale != 1 else raw,
            "raw": raw,
            "fault": bool(fault) if fault is not None else None,
        })
    return readings


def _mark_corroboration(
    readings: List[Dict[str, Any]], trusted: Optional[set] = None
) -> List[Dict[str, Any]]:
    """Flag channels that share a value with several peers.

    ``trusted`` carries labels already known to be separate sensors — either
    confirmed by the load test in :data:`CONFIRMED_DISTINCT_CHANNELS` or
    observed to move independently on this machine. Those stay distinct even
    when they happen to agree now, because two sensors reading the same number
    at the same moment is normal once they have been shown to be two sensors.
    That is exactly the idle case: seven real channels all sitting at ambient.

    Note what this does *not* do. It never looks at how far a channel moved.
    South Ambient's whole range under load was 1 °C against the CPU's 28 °C,
    and any magnitude rule tuned to the CPU would discard four real sensors
    for measuring something slower.
    """
    known = set(CONFIRMED_DISTINCT_CHANNELS) | set(trusted or ())
    counts: Dict[Any, int] = {}
    for reading in readings:
        counts[reading["raw"]] = counts.get(reading["raw"], 0) + 1
    result = []
    for reading in readings:
        shared = counts[reading["raw"]]
        corroborated = reading["label"] in known or shared < PLACEHOLDER_GROUP_SIZE
        entry = dict(reading)
        entry["distinct"] = corroborated
        entry["shares_value_with"] = shared - 1
        entry["reason"] = "" if corroborated else (
            "This channel reads exactly the same value as {} others. That is "
            "plausible on an idle machine and is also what firmware returns "
            "for a channel it does not implement, so Vaelor has not yet "
            "established that it is a separate sensor."
        ).format(shared - 1)
        result.append(entry)
    return result


def read_sensors(
    sys_root: str = "/sys", observer: Optional["DivergenceObserver"] = None
) -> Dict[str, Any]:
    """Fans and temperatures from the WMI sensor driver, or a stated absence."""
    node = find_node(sys_root)
    if node is None:
        return {
            "available": False,
            "fans": [],
            "temperatures": [],
            "reason": (
                "The {} kernel module is not loaded, so its fan and board "
                "temperature sensors are not readable. It ships with the "
                "distribution; loading it makes them available."
            ).format(WMI_SENSOR_MODULE),
            "module": WMI_SENSOR_MODULE,
            "node": None,
        }
    fans = _channels(node, "fan", 1)
    temperatures = _channels(node, "temp", 1000)
    if observer is not None:
        observer.record(temperatures)
    corroborated = _mark_corroboration(
        temperatures, observer.distinct_labels() if observer else None
    )
    return {
        "available": True,
        "node": node.name,
        "module": WMI_SENSOR_MODULE,
        "reason": "",
        # Fans are reported unconditionally: their readings were observed to
        # move between samples, so they are measurements rather than constants.
        "fans": fans,
        "temperatures": corroborated,
        "uncorroborated_temperatures": [
            entry["label"] for entry in corroborated if not entry["distinct"]
        ],
    }


#: Module-level observer, so divergence learned on one poll is remembered by
#: the next. Built lazily: importing this module must not touch the state
#: directory.
_OBSERVER: Optional["DivergenceObserver"] = None


def _shared_observer() -> "DivergenceObserver":
    global _OBSERVER
    if _OBSERVER is None:
        _OBSERVER = DivergenceObserver()
    return _OBSERVER


def wmi_sensor_telemetry(
    sys_root: str = "/sys", die_celsius: Optional[float] = None
) -> Dict[str, Any]:
    """Telemetry keys for the poll loop, omitted entirely when unreadable.

    All seven temperature channels were confirmed real by a load test, so they
    are reported. The corroboration machinery stays in place for a channel this
    table does not cover — another HP board, a firmware revision that adds one —
    and it still withholds anything unproven rather than assuming this machine's
    result generalises.

    ``die_celsius`` lets the caller pass the ``k10temp`` reading so the two
    independent CPU sources can be cross-checked. Both are reported; the
    comparison is published beside them.
    """
    sensors = read_sensors(sys_root, observer=_shared_observer())
    if not sensors.get("available"):
        return {}
    telemetry: Dict[str, Any] = {}
    fans = [
        {
            "label": entry["label"],
            "rpm": entry["value"],
            "fault": entry["fault"],
        }
        for entry in sensors["fans"]
    ]
    if fans:
        telemetry["wmi_fans"] = fans
    distinct = distinct_temperatures(sensors)
    if distinct:
        telemetry["wmi_temperatures"] = [
            {
                "label": entry["label"],
                "celsius": entry["value"],
                "fault": entry["fault"],
            }
            for entry in distinct
        ]
    if sensors.get("uncorroborated_temperatures"):
        telemetry["wmi_temperatures_uncorroborated"] = list(
            sensors["uncorroborated_temperatures"]
        )
    # Two independent paths to the CPU temperature, both reported. The
    # agreement is the useful signal: a drift between them later says
    # something neither reading says on its own.
    agreement = cross_source_agreement(sensors, die_celsius)
    if agreement.get("agrees") is not None:
        telemetry["cpu_temperature_sources_agree"] = agreement["agrees"]
        telemetry["cpu_temperature_source_delta_c"] = agreement["delta_c"]
        if agreement.get("reason"):
            telemetry["cpu_temperature_source_warning"] = agreement["reason"]
    return telemetry


# ---------------------------------------------------------------------------
# Cross-source agreement
# ---------------------------------------------------------------------------

#: The WMI ``CPU`` channel and ``k10temp``'s ``Tctl`` are two independent paths
#: to the same quantity — one through firmware, one off the die. Under load
#: they tracked each other to within 1.4 °C across a 29 °C swing.
#:
#: That is a reason to report *both* rather than to pick a winner. Two sources
#: agreeing is a stronger claim than either alone, and the agreement is worth
#: monitoring in its own right: if they later drift apart, something has
#: changed — a firmware update, a failing sensor, a mis-seated cooler — and
#: neither reading alone would say so.
CROSS_SOURCE_LABEL = "CPU"
CROSS_SOURCE_OBSERVED_DELTA_C = 1.4

#: How far apart they may drift before it is worth reporting. Set well above
#: the observed 1.4 °C: the two are sampled at different instants through
#: different paths, so a small transient difference is normal and only a
#: sustained gap means anything.
CROSS_SOURCE_ALERT_DELTA_C = 5.0


def cross_source_agreement(
    sensors: Dict[str, Any], die_celsius: Optional[float]
) -> Dict[str, Any]:
    """Compare the firmware CPU channel with the on-die reading.

    Returns ``agrees: None`` when either source is missing — an absent reading
    is not a disagreement, and reporting it as one would invent a fault out of
    a sensor nobody loaded.
    """
    channel = next(
        (
            entry for entry in sensors.get("temperatures", [])
            if entry.get("label") == CROSS_SOURCE_LABEL
        ),
        None,
    )
    firmware = channel.get("value") if channel else None
    if not isinstance(firmware, (int, float)) or not isinstance(
        die_celsius, (int, float)
    ):
        return {
            "agrees": None,
            "delta_c": None,
            "reason": (
                "Only one CPU temperature source is readable, so there is "
                "nothing to cross-check it against."
            ),
        }
    delta = round(abs(float(firmware) - float(die_celsius)), 2)
    agrees = delta <= CROSS_SOURCE_ALERT_DELTA_C
    return {
        "agrees": agrees,
        "delta_c": delta,
        "firmware_c": float(firmware),
        "die_c": float(die_celsius),
        "observed_delta_c": CROSS_SOURCE_OBSERVED_DELTA_C,
        "reason": "" if agrees else (
            "The firmware CPU sensor reads {:.1f} °C and the on-die sensor "
            "reads {:.1f} °C, {:.1f} °C apart. Under load these were measured "
            "agreeing to within {:.1f} °C, so a gap this size means one of "
            "them has changed."
        ).format(
            float(firmware), float(die_celsius), delta,
            CROSS_SOURCE_OBSERVED_DELTA_C,
        ),
    }


def distinct_temperatures(sensors: Dict[str, Any]) -> List[Dict[str, Any]]:
    """Only the channels shown to be separate sensors.

    What a surface should present as a list of measurements. The rest are
    available in ``temperatures`` with the reason they are not here.
    """
    return [
        entry for entry in sensors.get("temperatures", [])
        if entry.get("distinct")
    ]


# ---------------------------------------------------------------------------
# Divergence
# ---------------------------------------------------------------------------

def store_path() -> str:
    import os

    return os.environ.get("VAELOR_SENSOR_DIVERGENCE_PATH", "").strip() or state_path(
        "sensor-divergence.json"
    )


class DivergenceObserver:
    """Remember which temperature channels have been seen to move apart.

    A single sample cannot tell a real sensor from a firmware placeholder. Two
    samples in which the channels disagree can. This records that, so the
    appliance learns which channels are real by watching the machine work
    rather than by anyone asserting it — and so the answer survives a restart
    instead of being re-guessed on every boot.
    """

    def __init__(self, path: Optional[str] = None):
        self._path = path
        self._lock = threading.Lock()
        self._distinct: Optional[set] = None

    def _file(self) -> Path:
        return Path(self._path or store_path())

    def _load(self) -> set:
        if self._distinct is not None:
            return self._distinct
        found: set = set()
        try:
            raw = json.loads(self._file().read_text(encoding="utf-8"))
            found = {str(label) for label in raw.get("distinct", [])}
        except (AttributeError, OSError, TypeError, ValueError):
            found = set()
        self._distinct = found
        return found

    def _persist(self) -> None:
        try:
            path = self._file()
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text(
                json.dumps(
                    {
                        "distinct": sorted(self._distinct or ()),
                        "updated_at": int(time.time()),
                    },
                    separators=(",", ":"),
                ),
                encoding="utf-8",
            )
        except OSError:
            # An unwritable state directory costs the memory of this
            # observation, not the observation itself.
            pass

    def distinct_labels(self) -> set:
        """Confirmed channels plus anything this machine has since proved.

        The confirmed set is a prior, not something this observer learned, and
        it is deliberately not written to the store: the file records what
        *this* appliance saw, and mixing a shipped measurement into it would
        make the two indistinguishable a year from now.
        """
        with self._lock:
            return set(CONFIRMED_DISTINCT_CHANNELS) | set(self._load())

    def record(self, readings: List[Dict[str, Any]]) -> set:
        """Note one sample; return the labels now known to be distinct.

        A channel earns "distinct" by holding a value no sizeable group of its
        peers shares. That is the observation that separates a real sensor from
        a placeholder, and once earned it is kept.
        """
        with self._lock:
            known = set(self._load())
            counts: Dict[Any, int] = {}
            for reading in readings:
                counts[reading.get("raw")] = counts.get(reading.get("raw"), 0) + 1
            learned = {
                str(reading.get("label"))
                for reading in readings
                if counts.get(reading.get("raw"), 0) < PLACEHOLDER_GROUP_SIZE
            }
            if learned - known:
                self._distinct = known | learned
                self._persist()
            return set(self._distinct or known)

    def reset(self) -> None:
        with self._lock:
            self._distinct = set()
            self._persist()
