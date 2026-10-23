"""What the Assistant's model is shown of each reading: the fields the question needs, compactly (VD-219).

**The owner's rule (2026-10-17): the prompt is as small as it can be while
keeping what the question needs.** A history ask had built a 14-16k-token
prompt on a 16,384-token NPU window, and every extra prompt token costs about
1.6 ms before the first word. So this module designs to a budget:

* **A reading is formatted, not forwarded.** Each reading the model sees is
  either its *detail* view - the fields an owner asks about, with units in the
  value ("38.8 of 47.2 GB (82%)") instead of raw bytes, ids, provenance and
  agent guidance - or its one-line *summary*. The question's own subject gets
  the detail; everything else gets the line (LESSONS 11: the figure the
  question asks about must reach the model; the rest is context).
* **Telemetry is split by section** (CPU, memory, temperatures, fans, disk,
  network, GPU, NPU, power, boot) and a question is shown the sections it
  names, plus the one-line status.
* **History is downsampled**: per asked series, min/avg/max and a handful of
  evenly spaced points, never the raw rows.
* **Nothing is cut mid-field.** A reading that does not fit falls back to its
  line, then is dropped whole (:func:`fit_readings`).
* **One figure, one source** (LESSONS 6): the GPU, NPU and disk figures are in
  their own readings, so telemetry's copies are shown only when those readings
  are not.

This shapes the model's view only; the gathered facts and every built-in answer
read from them are untouched.
"""

from __future__ import annotations

import json
import re
from datetime import datetime, timezone
from typing import Any, Callable, Dict, FrozenSet, Iterable, List, Optional, Sequence, Tuple

from .assistant_vocabulary import TOOL_PHRASES, UPTIME_PHRASES, literal_phrases
from .network_interfaces import is_plumbing_name
from .phrase_match import mentions

#: Telemetry sections, and the words that ask for each. The shared vocabulary
#: owns the words for the readings that have a tool; CPU, memory, power and
#: boot have none, because telemetry is gathered for every question.
_SECTION_WORDS: Dict[str, Tuple[str, ...]] = {
    "cpu": ("cpu", "cpus", "processor", "processors", "chip", "core", "cores", "thread", "threads",
            "load", "busy", "clock", "frequency", "ghz"),
    "memory": ("memory", "ram", "swap", "ecc"),
    "temps": literal_phrases("cooling.status"),
    "fans": ("fan", "fans", "rpm", "cooling", "airflow"),
    "disk": literal_phrases("storage.status"),
    "network": literal_phrases("network.status"),
    "gpu": literal_phrases("gpu.status"),
    "npu": literal_phrases("npu.status"),
    "power": ("power", "watt", "watts", "wattage", "draw", "drawing", "energy"),
    "boot": ("boot", "booted", "reboot", "rebooted", "restart", "restarted") + UPTIME_PHRASES,
}

#: History series per section; the default when a question names none.
_HISTORY_SERIES: Dict[str, Tuple[str, ...]] = {
    "cpu": ("cpu_percent", "cpu_temperature", "cpu_freq"),
    "memory": ("memory_percent",),
    "temps": ("cpu_temperature", "gpu_temperature_c"),
    "disk": ("disk_primary_percent",),
    "network": ("network_download_speed", "network_upload_speed"),
    "gpu": ("gpu_busy_percent", "gpu_temperature_c", "gpu_power_watts"),
    "npu": ("npu_activity_percent", "npu_power_watts"),
    "power": ("cpu_package_power_watts", "gpu_power_watts"),
}
_DEFAULT_SERIES = ("cpu_percent", "cpu_temperature", "memory_percent", "gpu_busy_percent")

#: Points shown per history series, evenly spaced over the window.
HISTORY_POINTS = 6

#: Items a list shows before it is counted instead of listed.
LIST_ITEMS = 8

#: A string longer than this is summarised to its first sentence, or dropped.
SHORT_TEXT = 140

#: Keys that are bookkeeping, provenance or agent guidance, never a reading.
_NOT_READINGS = frozenset((
    "id", "ids", "sha256", "sha256_known", "revision", "mtime_ns", "source", "sources", "basis",
    "age_seconds", "vendor_status", "path", "file", "image", "runtime_container_id", "app_instance_id",
    "template_id", "utilisation_warning", "guidance", "rationale", "crosscheck", "topology_source",
))


def _bookkeeping(key: str) -> bool:
    """Whether ``key`` names bookkeeping: one of `_NOT_READINGS`, whole or as a ``_suffix``."""
    return key in _NOT_READINGS or any(key.endswith("_" + name) for name in _NOT_READINGS)


GB = 1e9


def topics_of(question: str) -> FrozenSet[str]:
    """The telemetry sections ``question`` names."""
    lower = str(question or "").lower()
    return frozenset(name for name, words in _SECTION_WORDS.items() if mentions(lower, words))


# --- value formatting --------------------------------------------------------


def _num(value: Any) -> Optional[float]:
    return float(value) if isinstance(value, (int, float)) and not isinstance(value, bool) else None


def _gb(value: Any) -> str:
    number = _num(value)
    return "" if number is None else "{:.1f} GB".format(number / GB)


def _of(used: Any, total: Any) -> str:
    """``"38.8 of 47.2 GB used (82%)"`` - "used" said, so the share is never read as free (kb N200)."""
    used_n, total_n = _num(used), _num(total)
    if used_n is None or not total_n:
        return ""
    return "{:.1f} of {:.1f} GB used ({:.0f}%)".format(used_n / GB, total_n / GB, 100 * used_n / total_n)


def _round(value: Any) -> Any:
    number = _num(value)
    if number is None or isinstance(value, int):
        return value
    return round(number, 1) if abs(number) >= 1 else round(number, 2)


def _when(value: Any) -> str:
    """Seconds or milliseconds since the epoch, as a UTC date and time."""
    number = _num(value)
    if not number:
        return ""
    if number > 1e11:
        number /= 1000.0
    return datetime.fromtimestamp(number, timezone.utc).strftime("%Y-%m-%d %H:%M UTC")


#: What a shortened text ends with, so a cut is never read as the whole.
CUT_MARK = " [...]"


def _short(text: Any, limit: int = SHORT_TEXT) -> str:
    """``text`` whole when short, else its first sentence when that is short, else cut at a word with a mark.

    Never ``""`` for a text that says something (kb R158): the Z2's 144-character
    network summary was shown as nothing in every sweep.
    """
    text = " ".join(str(text or "").split())
    if len(text) <= limit:
        return text
    first = re.split(r"(?<=[.!?])\s", text, maxsplit=1)[0]
    if len(first) <= limit:
        return first
    cut = text[:limit - len(CUT_MARK)].rsplit(" ", 1)[0]
    return cut + CUT_MARK


#: Job states whose message is the reason something went wrong, shown whole.
_FAILED_STATES = frozenset(("failed", "error", "errored", "blocked", "cancelled", "unhealthy"))


def _join(parts: Iterable[str], separator: str = ", ") -> str:
    return separator.join(part for part in parts if part)


def _compact(value: Any, depth: int = 0) -> Any:
    """Any reading, without bookkeeping: units in values, long text summarised, long lists counted."""
    if isinstance(value, dict):
        shown: Dict[str, Any] = {}
        for key, item in value.items():
            key = str(key)
            if item is None or item == "" or _bookkeeping(key) or depth >= 3 and isinstance(item, (dict, list)):
                continue
            if key.endswith("_bytes") and _num(item) is not None:
                shown[key[:-6]] = _gb(item)
                continue
            item = _compact(item, depth + 1)
            if item == "" or item is None:
                continue
            shown[key] = item
        return shown
    if isinstance(value, list):
        items = [_compact(item, depth + 1) for item in value[:LIST_ITEMS]]
        if len(value) > LIST_ITEMS:
            items.append("(+{} more)".format(len(value) - LIST_ITEMS))
        return items
    if isinstance(value, str):
        return _short(value)
    return _round(value)


# --- per-reading views: (detail, line) ---------------------------------------


def _limits(norms: Any, borrowed: bool = False) -> str:
    """``"(normal below 97 C, act at 100 C)"`` from a reading's own thermal norms, or ``""``.

    VD-219 data audit: 1,785 of 2,610 routed prompts showed a temperature
    without its threshold, so the answer could only say none was supplied.
    """
    if not isinstance(norms, dict) or not norms.get("available", True):
        return ""
    low = norms.get("investigate_above_c", norms.get("warning_c"))
    high = norms.get("act_at_c", norms.get("critical_c"))
    if _num(low) is None or _num(high) is None:
        return ""
    # A GPU has no limit of its own here: the machine-class processor limits
    # stand in for it, and the health reading says so (kb R168).
    return "(normal below {} C, act at {} C{})".format(_round(low), _round(high), "; processor limits" if borrowed else "")


def _telemetry(value: Dict[str, Any], topics: FrozenSet[str], shown_beside: FrozenSet[str]) -> Tuple[Any, str]:
    get = value.get
    limits = _limits(get("thermal_norms"))
    fans = [fan for fan in get("wmi_fans") or [] if isinstance(fan, dict)]
    faulty = [str(fan.get("label")) for fan in fans if fan.get("fault")]
    cpu = _join([
        _join(["{} C".format(_round(get("cpu_temperature"))), limits], " ")
        if _num(get("cpu_temperature")) is not None else "",
        "{}% busy".format(_round(get("cpu_percent"))) if _num(get("cpu_percent")) is not None else "",
        "{:.1f} GHz".format(_num(get("cpu_freq")) / 1000) if _num(get("cpu_freq")) else "",
        "{} cores, {} threads".format(get("cpu_cores"), get("cpu_threads")) if get("cpu_threads") else "",
    ])
    memory = _join([_of(get("memory_used"), get("memory_total")),
                    "{} available".format(_gb(get("memory_available"))) if get("memory_available") else ""])
    # Two CPU temperatures from two sensors are said as such (kb R159, LESSONS 6).
    if get("cpu_temperature_source_warning"):
        cpu = _join([cpu, "sensors disagree: " + _short(get("cpu_temperature_source_warning"))])
    sections: Dict[str, str] = {
        "cpu": cpu,
        "memory": _join([memory, "ECC {}".format(get("memory_ecc_state")) if get("memory_ecc_state") else ""]),
        "fans": _join(["{} {} rpm".format(fan.get("label"), fan.get("rpm")) for fan in fans])
        + ("; faults: " + ", ".join(faulty) if faulty else (", no faults" if fans else "")),
        "temps": _join([
            _join(["board sensors:"] + [", ".join(
                "{} {} C".format(entry.get("label"), _round(entry.get("celsius")))
                for entry in (get("wmi_temperatures") or []) if isinstance(entry, dict))], " ")
            if get("wmi_temperatures") else ""]
            + ["{} {} {} C".format(entry.get("device"), entry.get("label"), _round(entry.get("celsius")))
               for key in ("storage_temperatures", "network_temperatures")
               for entry in (get(key) or []) if isinstance(entry, dict)]),
        "disk": _join(["{} {}".format(get("disk_primary_mount") or "/", _of(get("disk_primary_used"), get("disk_primary_total")))]
                      if get("disk_primary_total") else []),
        "network": "down {} B/s, up {} B/s".format(_round(get("network_download_speed")), _round(get("network_upload_speed")))
        if get("network_download_speed") is not None else "",
        "gpu": _join([
            "{}% busy".format(get("gpu_busy_percent")) if get("gpu_busy_percent") is not None else "",
            _join(["{} C".format(_round(get("gpu_gfx_temperature_c") or get("gpu_temperature_c"))),
                   _limits(get("thermal_norms"), borrowed=True)], " ")
            if get("gpu_temperature_c") else "",
            "{} W".format(_round(get("gpu_power_watts"))) if get("gpu_power_watts") is not None else "",
            "GTT {}".format(_of(get("gpu_gtt_used_bytes"), get("gpu_gtt_total_bytes"))) if get("gpu_gtt_total_bytes") else "",
        ]),
        "npu": _join([
            "{}% activity".format(_round(get("npu_activity_percent"))) if get("npu_activity_percent") is not None else "",
            "{} W".format(_round(get("npu_power_watts"))) if get("npu_power_watts") is not None else "",
        ]),
        "power": "CPU package {} W".format(_round(get("cpu_package_power_watts"))) if get("cpu_package_power_watts") is not None else "",
        "boot": "booted {}".format(_when(get("boot_time"))) if get("boot_time") else "",
    }
    owned = {"gpu": "gpu.status", "npu": "npu.status", "disk": "storage.status", "fans": "cooling.status"}
    usable = {name: text for name, text in sections.items() if text and owned.get(name) not in shown_beside}
    line = _join(["CPU " + usable["cpu"] if "cpu" in usable else "",
                  "memory " + usable["memory"].split(", ECC")[0] if "memory" in usable else "",
                  "disk " + usable["disk"] if "disk" in usable else "",
                  "fans " + usable["fans"] if "fans" in usable else "",
                  "GPU " + usable["gpu"] if "gpu" in usable else ""], "; ")
    norms = get("thermal_norms") if isinstance(get("thermal_norms"), dict) else {}
    detail: Dict[str, Any] = {name: usable[name] for name in ("cpu", "memory") if name in usable}
    detail.update({name: usable[name] for name in sorted(topics) if name in usable})
    if norms.get("available") and topics & {"cpu", "temps", "gpu"}:
        detail["thermal_limits"] = "lower is normal for this machine class"
    return detail, line


def _gpu(value: Dict[str, Any], topics: FrozenSet[str], shown_beside: FrozenSet[str]) -> Tuple[Any, str]:
    if value.get("detected") is False:
        reason = _short(value.get("reason")) or "no GPU detected"
        return {"detected": False, "reason": reason}, "GPU: " + reason
    if not isinstance(value.get("adapters"), list):
        raise ValueError("not the GPU reading's shape")  # shown compacted - never read as absent (LESSONS 4)
    adapters, lines = [], []
    for adapter in value.get("adapters") or []:
        if not isinstance(adapter, dict):
            continue
        memory = adapter.get("memory") if isinstance(adapter.get("memory"), dict) else {}
        state = _join([
            "{}% busy".format(adapter.get("utilisation_percent")) if adapter.get("utilisation_percent") is not None else "",
            _join(["{} C".format(_round(adapter.get("temperature_c"))),
                   _limits(value.get("thermal_norms"), borrowed=True)], " ")
            if adapter.get("temperature_c") is not None else "",
            "{} W".format(_round(adapter.get("power_watts"))) if adapter.get("power_watts") is not None else "",
            "{} MHz".format(adapter.get("clock_mhz")) if adapter.get("clock_mhz") else "",
        ])
        gtt = _of(memory.get("gtt_used_bytes"), memory.get("gtt_total_bytes"))
        vram = _of(memory.get("vram_used_bytes"), memory.get("vram_total_bytes"))
        entry = {"name": adapter.get("name"), "state": state}
        if memory.get("unified_memory"):
            entry["memory"] = _join(["GTT (shared system memory, where models load) " + gtt if gtt else "",
                                     "VRAM (small firmware carve-out) " + vram if vram else ""], "; ")
        else:
            entry["memory"] = _join(["VRAM " + vram if vram else "", "GTT " + gtt if gtt else ""], "; ")
        adapters.append(entry)
        lines.append(_join([str(adapter.get("name") or "GPU"), state, "GTT " + gtt if gtt else ""]))
    detail: Dict[str, Any] = {"adapters": adapters}
    detail.update({key: value[key] for key in ("rocm_version", "mesa_version") if value.get(key)})
    return detail, "; ".join(lines)


def _npu(value: Dict[str, Any], topics: FrozenSet[str], shown_beside: FrozenSet[str]) -> Tuple[Any, str]:
    if value.get("detected") is False:
        reason = _short(value.get("reason")) or "no NPU detected"
        return {"detected": False, "reason": reason}, "NPU: " + reason
    if not isinstance(value.get("accelerators"), list):
        raise ValueError("not the NPU reading's shape")  # shown compacted - never read as absent (LESSONS 4)
    details, lines = [], []
    for accelerator in value.get("accelerators") or []:
        if not isinstance(accelerator, dict):
            continue
        power = accelerator.get("power") if isinstance(accelerator.get("power"), dict) else {}
        clock = accelerator.get("clock") if isinstance(accelerator.get("clock"), dict) else {}
        state = _join([
            "{}% activity".format(_round(accelerator.get("utilisation_percent")))
            if accelerator.get("utilisation_percent") is not None else "",
            "{} W".format(_round(power.get("watts"))) if power.get("watts") is not None else "",
            "{} MHz".format(clock.get("mhz")) if clock.get("mhz") else "",
        ])
        entry = {"name": accelerator.get("name"), "driver": accelerator.get("driver"), "state": state,
                 "used_by_vaelor_inference": bool(accelerator.get("usable_for_inference")),
                 "firmware": accelerator.get("firmware_version"),
                 "rated": "{} TOPS {} (datasheet peak)".format(accelerator.get("rated_tops"), accelerator.get("rated_tops_precision"))
                 if accelerator.get("rated_tops") else ""}
        if not accelerator.get("usable_for_inference"):
            entry["why_not"] = _short(accelerator.get("usability_reason"))
        details.append({key: item for key, item in entry.items() if item not in ("", None)})
        lines.append(_join([str(accelerator.get("name") or "NPU"), state]))
    return {"accelerators": details}, "; ".join(lines)


def _storage(value: Dict[str, Any], topics: FrozenSet[str], shown_beside: FrozenSet[str]) -> Tuple[Any, str]:
    volumes = [volume for volume in value.get("volumes") or [] if isinstance(volume, dict)]
    shown_volumes = [_join([str(volume.get("mountpoint") or volume.get("name")),
                            str(volume.get("filesystem") or ""),
                            "{}% used".format(volume.get("used_percent")) if volume.get("used_percent") is not None else "",
                            "{} free of {}".format(_gb(volume.get("free_bytes")), _gb(volume.get("total_bytes"))),
                            "{} reserved for the system".format(_gb(volume.get("reserved_bytes"))) if volume.get("reserved_bytes") else "",
                            str(volume.get("model") or "")])
                     for volume in volumes]

    def unmounted(device: Dict[str, Any]) -> bool:
        children = [child for child in device.get("children") or [] if isinstance(child, dict)]
        return (device.get("type") != "loop" and not device.get("mountpoints")
                and not any(child.get("mountpoints") for child in children))

    spare = [_join([str(device.get("name")), _gb(device.get("size")), str(device.get("model") or "")])
             for device in value.get("devices") or [] if isinstance(device, dict) and unmounted(device)]
    detail: Dict[str, Any] = {"summary": value.get("summary"), "volumes": shown_volumes}
    if spare:
        detail["not_mounted"] = spare
    if value.get("smartctl_available") is False:
        detail["drive_health"] = "SMART not readable"
    line = "; ".join(shown_volumes[:3]) or _short(value.get("summary"))
    line += "; not mounted: " + ", ".join(spare) if spare else ""
    return {key: item for key, item in detail.items() if item}, line


def _network(value: Dict[str, Any], topics: FrozenSet[str], shown_beside: FrozenSet[str]) -> Tuple[Any, str]:
    interfaces, links = [], 0
    for interface in value.get("interfaces") or []:
        if not isinstance(interface, dict):
            continue
        name = str(interface.get("name") or "")
        # Container and VM plumbing, by name only: the reading may describe
        # another machine, whose /sys this one cannot look in (VD-133 1.5.5 #1).
        if is_plumbing_name(name):
            links += 1
            continue
        if name == "lo":
            continue
        addresses = [address.get("address") + "/" + str(address.get("prefix"))
                     for address in interface.get("addresses") or []
                     if isinstance(address, dict) and address.get("family") == "inet"]
        interfaces.append(_join([name, str(interface.get("state") or ""), " ".join(addresses),
                                 "mac " + str(interface.get("mac")) if interface.get("mac") else "",
                                 "rx {} tx {}".format(_gb(interface.get("rx_bytes")), _gb(interface.get("tx_bytes")))
                                 if interface.get("rx_bytes") is not None else "",
                                 "errors {}".format(interface.get("rx_errors", 0) + interface.get("tx_errors", 0))
                                 if isinstance(interface.get("rx_errors"), int) else ""]))
    routes = [_join(["default via " + str(route.get("gateway")) if route.get("dst") == "default" else str(route.get("dst")),
                     "dev " + str(route.get("dev"))])
              for route in value.get("routes") or [] if isinstance(route, dict) and route.get("dst") == "default"]
    detail = {"summary": value.get("summary"), "interfaces": interfaces, "default_route": routes,
              "dns": value.get("dns")}
    if links:
        detail["container_links_not_listed"] = links
    line = " ".join(str(value.get("summary") or "").split()) or "; ".join(interfaces[:3])
    return {key: item for key, item in detail.items() if item}, line


def _services(value: Any, topics: FrozenSet[str], shown_beside: FrozenSet[str]) -> Tuple[Any, str]:
    units = [unit for unit in value if isinstance(unit, dict)] if isinstance(value, list) else []
    if not units:
        return _compact(value), json.dumps(_compact(value), separators=(",", ":"), default=str)
    rows = ["{}: not installed".format(unit.get("id")) if unit.get("available") is False else
            "{}: {}/{}, {} restarts{}".format(unit.get("id"), unit.get("active"), unit.get("substate"),
                                               unit.get("restarts", 0),
                                               ", exit {}".format(unit.get("exit_status")) if unit.get("exit_status") else "")
            for unit in units]
    def said(unit: Dict[str, Any]) -> str:
        return str(unit.get("id") or unit.get("unit") or "").replace("-", " ")

    # A unit this machine does not have (available: false) is not a stopped
    # service: it is said as not installed and not counted (kb R168).
    absent = [said(unit) for unit in units if unit.get("available") is False]
    units = [unit for unit in units if unit.get("available") is not False]
    down = [said(unit) for unit in units if unit.get("active") != "active"]
    up = [said(unit) for unit in units if unit.get("active") == "active"]
    restarts = sum(unit.get("restarts") or 0 for unit in units if isinstance(unit.get("restarts"), int))
    # VD-219 data audit: "is the control plane running" got "6 of 6 active" and
    # no name to say yes about; each unit is named, briefly.
    line = "managed services, {} active: {}{}{}; {} restarts".format(
        "all {}".format(len(units)) if not down else "{} of {}".format(len(up), len(units)),
        ", ".join(up) or "none", "; not active: " + ", ".join(down) if down else "",
        "; not installed: " + ", ".join(absent) if absent else "", restarts)
    return rows, line


def _health(value: Dict[str, Any], topics: FrozenSet[str], shown_beside: FrozenSet[str]) -> Tuple[Any, str]:
    policy = value.get("thermal_policy") if isinstance(value.get("thermal_policy"), dict) else {}
    detail = {"status": value.get("status"), "reasons": value.get("reasons"), "checked": value.get("checked"),
              "notes": value.get("notes"),
              "temperature_limits": _limits(policy)}
    reasons = [str(reason) for reason in value.get("reasons") or []]
    line = "health: {}{}{}".format(value.get("status"), " - " + "; ".join(reasons) if reasons else
                                   " (checked {})".format(", ".join(map(str, value.get("checked") or []))),
                                   "; temperature " + _limits(policy) if _limits(policy) else "")
    return {key: item for key, item in detail.items() if item not in ("", None)}, line


def _cooling(value: Dict[str, Any], topics: FrozenSet[str], shown_beside: FrozenSet[str]) -> Tuple[Any, str]:
    cpu = value.get("cpu") if isinstance(value.get("cpu"), dict) else {}
    case = value.get("case") if isinstance(value.get("case"), dict) else {}
    named, fans = set(), []
    for fan in list(cpu.get("fans") or []) + list(case.get("board_fans") or []):
        if isinstance(fan, dict) and fan.get("label") not in named:
            named.add(fan.get("label"))
            fans.append("{} {} rpm{}".format(fan.get("label"), fan.get("rpm"), " FAULT" if fan.get("fault") else ""))
    detail = {"fans_read": fans or "no fan reading", "cpu_temperature": "{} C".format(_round(cpu.get("temperature")))
              if cpu.get("temperature") is not None else "",
              "fan_control": "not available from Linux" if cpu.get("control") == "unavailable" else cpu.get("mode"),
              "enclosure_fans": case.get("fan_count") if case.get("detected") else "none fitted"}
    line = "fans: " + (", ".join(fans) if fans else "none read")
    return {key: item for key, item in detail.items() if item not in ("", None)}, line


def _updates(value: Dict[str, Any], topics: FrozenSet[str], shown_beside: FrozenSet[str]) -> Tuple[Any, str]:
    if not value.get("collected", True):
        return _compact(value), "updates: " + (_short(value.get("reason")) or "not read")
    packages = [_join([str(item.get("name")), "{} -> {}".format(item.get("installed"), item.get("candidate"))])
                for item in value.get("details") or [] if isinstance(item, dict)]
    held = [_join([str(item.get("name")), _short(item.get("reason"))], ": ")
            for item in value.get("held_back") or [] if isinstance(item, dict)]
    detail = {"available": packages[:LIST_ITEMS] or "none", "count": value.get("count"),
              "downloaded_ready_to_install": bool(value.get("staged")),
              "held_back_needs_full_upgrade": held,
              "reboot_already_pending_from_earlier_changes": bool(value.get("reboot_required")),
              "reboot_after_installing_these": value.get("reboot_likelihood"),
              "estimated_minutes": value.get("estimated_install_minutes"),
              "last_completed": _when(value.get("last_completed_at"))}
    # VD-219 data audit: answers said "reboot to apply the pending update" when
    # the updates were only available. Installed or not, and the reboot, plainly.
    line = "updates: {}{}; reboot: {}".format(
        "{} available, not installed{}".format(
            value.get("count"), " (downloaded)" if value.get("staged") else " (not downloaded)")
        if value.get("count") else "no updates available",
        ", {} held back".format(len(held)) if held else "",
        "already pending from earlier changes" if value.get("reboot_required") else "none pending")
    return {key: item for key, item in detail.items() if item not in ("", None, [])}, line


def _counted(items: List[str], limit: int) -> List[str]:
    """``items`` up to ``limit``, then how many more there are - a cut list is never read as whole (kb R159)."""
    return items[:limit] + (["(+{} more)".format(len(items) - limit)] if len(items) > limit else [])


def _inventory(value: Dict[str, Any], topics: FrozenSet[str], shown_beside: FrozenSet[str],
               question: str = "") -> Tuple[Any, str]:
    lower = str(question or "").lower()

    def named(app: Dict[str, Any]) -> bool:
        names = {str(app.get(key) or "").lower() for key in ("name", "display_identity", "project")} - {""}
        names |= {name.split("/")[0] for name in names}
        return any(re.search(r"(?<![\w-])" + re.escape(name) + r"(?![\w-])", lower) for name in names)

    apps = [app for app in value.get("apps") or [] if isinstance(app, dict)]
    # The apps a question names come first, so a cut never hides the one asked about (kb R159).
    apps = sorted(apps, key=lambda app: not named(app))
    rows = []
    for app in apps:
        ports = sorted({str(port.get("host")) for port in app.get("ports") or [] if isinstance(port, dict) and port.get("host")})
        rows.append(_join([str(app.get("display_identity") or app.get("name")), str(app.get("status") or ""),
                           "ports " + " ".join(ports) if ports else "",
                           "managed" if app.get("managed") else "not managed by Vaelor"]))
    models = [_join([str(model.get("name")), str(model.get("kind") or ""), _gb(model.get("size_bytes")),
                     "in use" if model.get("in_use") else ""])
              for model in value.get("models") or [] if isinstance(model, dict)]
    # VD-133 1.5.5 #12: Vaelor's own containers (Phoenix, web search, model
    # serving) are listed apart from the owner's apps; the Assistant still
    # knows they run, said as part of Vaelor (review F6).
    own = [app for app in value.get("vaelor_containers") or [] if isinstance(app, dict)]
    detail = {"apps": _counted(rows, LIST_ITEMS * 2) or "none installed",
              "models": _counted(models, LIST_ITEMS) or "none"}
    if own:
        detail["vaelor_services"] = _counted([
            _join([str(app.get("name")), str(app.get("status") or ""),
                   "part of Vaelor (" + str(app.get("vaelor_component") or "") + ")"]) for app in own],
            LIST_ITEMS * 2)
    taken = sorted({str(port.get("host")) for app in apps + own for port in app.get("ports") or []
                    if isinstance(port, dict) and port.get("host")}, key=lambda port: (len(port), port))
    if taken:
        detail["ports_in_use"] = " ".join(taken)
    running = sum(1 for app in apps if app.get("running"))
    line = "apps: {} installed, {} running{}".format(
        len(apps), running, " (" + ", ".join(str(app.get("display_identity") or app.get("name")) for app in apps[:LIST_ITEMS]) + ")"
        if apps else "")
    if own:
        line += "; Vaelor's own services: {} running".format(sum(1 for app in own if app.get("running")))
    return detail, line


def _jobs(value: Any, topics: FrozenSet[str], shown_beside: FrozenSet[str]) -> Tuple[Any, str]:
    jobs = [job for job in value if isinstance(job, dict)] if isinstance(value, list) else []
    if not jobs:
        return _compact(value), "jobs: none recent"
    rows = [_join([str(job.get("type")), str(job.get("state")),
                   # A failure's reason survives whole (kb R158): "bind for 0.0.0.0:8096 failed".
                   " ".join(str(job.get("message") or "").split()) if str(job.get("state")) in _FAILED_STATES
                   else _short(job.get("message")),
                   "{}%".format(job.get("progress")) if job.get("progress") not in (None, 100) else "",
                   _when(job.get("updated_at") or job.get("created_at"))])
            for job in jobs[:LIST_ITEMS]]
    return rows, "last job: " + rows[0]


def _history(value: Dict[str, Any], topics: FrozenSet[str], shown_beside: FrozenSet[str]) -> Tuple[Any, str]:
    samples = [sample for sample in value.get("samples") or [] if isinstance(sample, dict)]
    if not value.get("available", True) or not samples:
        reason = _short(value.get("reason")) or "no retained samples"
        return {"available": bool(value.get("available")), "reason": reason}, "history: " + reason
    wanted = [name for topic in sorted(topics) for name in _HISTORY_SERIES.get(topic, ())] or list(_DEFAULT_SERIES)
    series: Dict[str, Any] = {}
    for name in dict.fromkeys(wanted):
        values = [_num(sample.get(name)) for sample in samples]
        values = [number for number in values if number is not None]
        if not values:
            continue
        step = max(1, (len(values) - 1) // (HISTORY_POINTS - 1)) if len(values) > HISTORY_POINTS else 1
        points = values[::step][:HISTORY_POINTS - 1] + [values[-1]] if len(values) > HISTORY_POINTS else values
        series[name] = {"min": _round(min(values)), "avg": _round(sum(values) / len(values)),
                        "max": _round(max(values)), "points_oldest_first": [_round(point) for point in points]}
    first, last = samples[0].get("time"), samples[-1].get("time")
    detail = {"samples": len(samples), "from": first, "to": last, "series": series}
    if value.get("machine"):
        detail["machine"] = value["machine"]  # whose rows these are (kb R159)
    if value.get("window_seconds"):
        detail["window_seconds"] = value["window_seconds"]
    if value.get("bucket_seconds"):
        detail["each_point"] = ("the mean of a {}-second bucket; min and max are of those means, "
                                "not single readings").format(value["bucket_seconds"])
    return detail, "history: {} samples from {} to {}".format(len(samples), first, last)


def _display(value: Dict[str, Any], topics: FrozenSet[str], shown_beside: FrozenSet[str]) -> Tuple[Any, str]:
    if value.get("detected") is False:
        return {"detected": False}, "front display: none detected"
    view = _compact(value)
    return view, "front display: {} ({})".format(value.get("hardware"), "on" if value.get("enabled") else "off")


def _lighting(value: Dict[str, Any], topics: FrozenSet[str], shown_beside: FrozenSet[str]) -> Tuple[Any, str]:
    if not value:
        return {"detected": False}, "lighting: no lighting controller"
    view = _compact(value)
    return view, "lighting: " + json.dumps(view, separators=(",", ":"), default=str)


def _capabilities(value: Dict[str, Any], topics: FrozenSet[str], shown_beside: FrozenSet[str]) -> Tuple[Any, str]:
    docker = value.get("docker") if isinstance(value.get("docker"), dict) else {}
    system = value.get("os") if isinstance(value.get("os"), dict) else {}
    inference = value.get("inference") if isinstance(value.get("inference"), dict) else {}
    access = inference.get("accelerator_access") if isinstance(inference.get("accelerator_access"), dict) else {}
    detail = {
        "docker": _join(["installed" if docker.get("installed") else "absent",
                         "compose " + str(docker.get("compose_version")) if docker.get("compose") else "",
                         "runtime " + str(docker.get("runtime")) if docker.get("runtime") else ""]),
        "os": _join([str(system.get("name") or ""), str(system.get("support_label") or "")]),
        "architecture": inference.get("architecture"),
        "accelerators_for_apps": "ready" if access.get("ready") else _short(access.get("reason")),
    }
    detail = {key: item for key, item in detail.items() if item}
    return detail, "app platform: " + "; ".join("{} {}".format(key, item) for key, item in detail.items())


def _identity(value: Any, topics: FrozenSet[str], shown_beside: FrozenSet[str]) -> Tuple[Any, str]:
    view = _compact(value)
    return view, json.dumps(view, separators=(",", ":"), default=str)


_VIEWS: Dict[str, Callable[[Any, FrozenSet[str], FrozenSet[str]], Tuple[Any, str]]] = {
    "system.telemetry": _telemetry, "gpu.status": _gpu, "npu.status": _npu, "storage.status": _storage,
    "network.status": _network, "services.status": _services, "health.status": _health,
    "cooling.status": _cooling, "updates.status": _updates, "workloads.inventory": _inventory,
    "jobs.recent": _jobs, "metrics.history": _history, "system.identity": _identity,
    "display.status": _display, "lighting.status": _lighting, "workloads.capabilities": _capabilities,
}


def views(name: str, value: Any, topics: FrozenSet[str] = frozenset(),
          shown_beside: FrozenSet[str] = frozenset(), question: str = "") -> Tuple[Any, str]:
    """``(detail, line)`` for one reading: the fields for the question, and a one-line summary."""
    if isinstance(value, dict) and set(value) == {"unavailable"}:
        return value, "{}: unavailable - {}".format(name, value["unavailable"])
    view = _VIEWS.get(name)
    if view is not None and isinstance(value, (dict, list)):
        try:
            detail, line = (_inventory(value, topics, shown_beside, question) if view is _inventory
                            else view(value, topics, shown_beside))
            if detail or line.strip():
                return detail, line
        except (AttributeError, KeyError, TypeError, ValueError):
            pass  # an unexpected shape is shown compacted, never dropped
    compact = _compact(value)
    return compact, json.dumps(compact, separators=(",", ":"), default=str)


def model_view(name: str, value: Any, topics: FrozenSet[str] = frozenset()) -> Any:
    """The detail view of one reading (see :func:`views`)."""
    return views(name, value, topics)[0]


def _size(value: Any) -> int:
    return len(json.dumps(value, separators=(",", ":"), default=str))


def fit_readings(order: Sequence[str], facts: Dict[str, Any], budget: int, *,
                 detailed: Iterable[str] = (), topics: FrozenSet[str] = frozenset(),
                 question: str = "") -> Dict[str, Any]:
    """The readings in ``order`` within ``budget`` characters: detail for ``detailed``, a line for the rest.

    Each reading is tried as its detail view (if the question asked for it),
    then as its line, then left out whole - never cut mid-field.
    """
    present = [key for key in order if key in facts]
    beside = frozenset(present)
    wanted = set(detailed)
    chosen: Dict[str, Any] = {}
    used = 0
    for key in present:
        detail, line = views(key, facts[key], topics, beside - {key}, question)
        if detail in ({}, [], "", None) and line in ("{}", "[]", '""', "", "null"):
            continue  # nothing left to say once bookkeeping is out
        for candidate in ((detail, line) if key in wanted else (line,)):
            size = _size(candidate) + len(key) + 4
            if used + size <= budget:
                chosen[key] = candidate
                used += size
                break
        else:
            if not chosen:
                # The first reading is the question's own: its line is shown
                # even over budget, so no ask reaches the model with nothing.
                chosen[key] = line
                used += _size(line) + len(key) + 4
    return chosen


#: The earlier turns and saved notes an on-device answer may carry, in characters.
LOCAL_TURN_CHARS = 1200

#: One earlier turn or note, at most.
TURN_TEXT = 400


def bounded_turns(conversation: Sequence[Any], memories: Sequence[Any]) -> Tuple[List[Any], List[Any]]:
    """Earlier turns and notes for the on-device model: each shortened, oldest turn dropped first (kb R159)."""
    def shortened(item: Any) -> Any:
        if not isinstance(item, dict) or "content" not in item:
            return item
        return dict(item, content=_short(item.get("content"), TURN_TEXT))

    notes = [shortened(item) for item in memories]
    turns = [shortened(item) for item in conversation]
    while turns and _size(turns) + _size(notes) > LOCAL_TURN_CHARS:
        turns = turns[1:]
    while notes and _size(notes) > LOCAL_TURN_CHARS:
        notes = notes[:-1]
    return turns, notes
