"""Labelled hwmon sensor selection.

Taking ``max()`` over ``/sys/class/thermal/thermal_zone*/temp`` is only correct
where the only zone is the CPU. On an AMD workstation the sole ACPI zone reads
the board at ~35 °C while the processor sits at 40 °C in ``k10temp``, so the
maximum silently reports the board as the CPU. Sensors have names and labels;
this module reads them.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, List, Optional


#: hwmon driver names that report a CPU package temperature, best first, with
#: the sensor labels that identify the package reading within each driver.
#:
#: ``hp_wmi_sensors`` publishes a channel explicitly labelled ``CPU`` and is
#: deliberately **not** in this list. ``k10temp/Tctl`` stays primary for three
#: reasons: it is the processor's own on-die sensor rather than a board reading
#: relayed through firmware; it is the control temperature AMD's own boost and
#: throttle logic acts on, so it is the number that explains what the processor
#: is doing; and it is polled directly rather than through a WMI call whose
#: update rate the firmware chooses. The WMI ``CPU`` channel is reported beside
#: it as a board-side reading — which is also, at present, one of the channels
#: reading exactly 31.0 °C alongside four others and therefore the *least*
#: corroborated candidate for a primary source.
CPU_SENSORS = (
    ("k10temp", ("tdie", "tctl")),
    ("zenpower", ("tdie", "tctl")),
    ("coretemp", ("package id 0",)),
    ("cpu_thermal", ()),
    ("soc_thermal", ()),
    ("cpu-thermal", ()),
)

#: Thermal-zone types that genuinely name the CPU, used before any fallback.
CPU_ZONE_TYPES = ("cpu-thermal", "cpu_thermal", "x86_pkg_temp", "soc_thermal")

#: hwmon names whose readings belong to storage and networking, reported
#: separately rather than folded into the CPU number.
STORAGE_SENSORS = ("nvme", "drivetemp")
NETWORK_SENSOR_PREFIXES = ("r8169", "e1000", "igc", "iwlwifi", "mt7", "ath1", "rtw")


def _read(path: Path) -> str:
    try:
        return path.read_text(encoding="utf-8", errors="replace").strip("\x00\n ")
    except OSError:
        return ""


def _milli_celsius(path: Path) -> Optional[float]:
    raw = _read(path)
    try:
        value = float(raw)
    except (TypeError, ValueError):
        return None
    celsius = value / 1000 if abs(value) > 200 else value
    if not -20 <= celsius <= 150:
        return None
    return round(celsius, 1)


def _hwmon_nodes(sys_root: Path) -> List[Dict[str, Any]]:
    nodes: List[Dict[str, Any]] = []
    try:
        entries = sorted((sys_root / "class" / "hwmon").iterdir())
    except OSError:
        return nodes
    for entry in entries:
        name = _read(entry / "name")
        if not name:
            continue
        readings = []
        try:
            inputs = sorted(entry.glob("temp*_input"))
        except OSError:
            inputs = []
        for sensor in inputs:
            value = _milli_celsius(sensor)
            if value is None:
                continue
            prefix = sensor.name[: -len("_input")]
            readings.append({
                "label": _read(entry / f"{prefix}_label") or prefix,
                "celsius": value,
            })
        nodes.append({"name": name, "path": entry, "readings": readings})
    return nodes


def cpu_temperature(sys_root: str = "/sys") -> Dict[str, Any]:
    """Return the CPU temperature and say where it came from.

    The source string is part of the contract: a reading whose provenance is
    "the hottest thermal zone" is a guess, and callers deserve to know that.
    """
    root = Path(sys_root)
    nodes = _hwmon_nodes(root)
    for driver, labels in CPU_SENSORS:
        for node in nodes:
            if node["name"].lower() != driver:
                continue
            for reading in node["readings"]:
                label = reading["label"].lower()
                if not labels or label in labels:
                    return {
                        "celsius": reading["celsius"],
                        "source": "{}/{}".format(driver, reading["label"]),
                        "labelled": True,
                    }
    try:
        zones = sorted((root / "class" / "thermal").glob("thermal_zone*"))
    except OSError:
        zones = []
    readings = []
    for zone in zones:
        zone_type = _read(zone / "type").lower()
        value = _milli_celsius(zone / "temp")
        if value is None:
            continue
        if zone_type in CPU_ZONE_TYPES:
            return {
                "celsius": value,
                "source": "thermal/{}".format(zone_type),
                "labelled": True,
            }
        readings.append((value, zone_type))
    if not readings:
        return {"celsius": None, "source": None, "labelled": False}
    # Last resort. Flagged as unlabelled so nothing downstream mistakes a
    # board or chipset sensor for a processor reading.
    hottest = max(readings)
    return {
        "celsius": hottest[0],
        "source": "thermal/{}".format(hottest[1] or "unknown"),
        "labelled": False,
    }


def core_temperatures(sys_root: str = "/sys") -> List[Dict[str, Any]]:
    """Per-die or per-core temperatures where the driver exposes them."""
    cores = []
    for node in _hwmon_nodes(Path(sys_root)):
        name = node["name"].lower()
        if name not in {"k10temp", "zenpower", "coretemp"}:
            continue
        for reading in node["readings"]:
            label = reading["label"].lower()
            if label.startswith("tccd") or label.startswith("core "):
                cores.append({
                    "label": reading["label"],
                    "celsius": reading["celsius"],
                })
    return cores


def peripheral_temperatures(sys_root: str = "/sys") -> Dict[str, Any]:
    """Storage and network sensor readings, reported in their own right."""
    storage = []
    network = []
    for node in _hwmon_nodes(Path(sys_root)):
        name = node["name"].lower()
        if name in STORAGE_SENSORS:
            for reading in node["readings"]:
                storage.append({
                    "device": node["name"],
                    "label": reading["label"],
                    "celsius": reading["celsius"],
                })
        elif name.startswith(NETWORK_SENSOR_PREFIXES):
            for reading in node["readings"]:
                network.append({
                    "device": node["name"],
                    "label": reading["label"],
                    "celsius": reading["celsius"],
                })
    return {"storage": storage, "network": network}
