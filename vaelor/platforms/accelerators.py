"""Read-only accelerator discovery and telemetry from kernel sysfs.

Discovery and live telemetry come from sysfs, readable by an unprivileged
account with no vendor tooling installed. That is deliberate: on the first AMD
host Vaelor was pointed at, ``amd-smi`` reported every field as ``N/A`` while
the kernel's own hwmon files returned correct temperature, power and clock. So
sysfs stays the source of truth for anything the kernel publishes.

A few values are not in sysfs at all - the installed ROCm and Mesa versions,
adapter firmware, and neural-accelerator power - and for those an optional CLI
is consulted, with a filesystem path tried first wherever one exists. Every
such call is injectable, times out, and degrades to a stated gap.

One rule governs all of it, and it was learned the expensive way: report what
was read and where it was read from. Do not assert what the hardware can or
cannot do. Three separate values in this module were once hard-coded as
unreadable - NPU power, NPU firmware, the Mesa version - and all three were
readable; a fourth, the ROCm version, was reported absent because this module
looked in a path that no longer exists. A note naming its source corrects
itself; a claim about the world does not.
"""

from __future__ import annotations

import json
import os
import re
import shutil
import subprocess
import threading
import time
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional

from .base import read_number, text
# Re-exported: moved to `graphics_software` at the thousand-line split
# (software inventory there, sysfs discovery here); callers of either keep working.
from .graphics_software import (  # noqa: F401
    GPU_GFX_POWER_FIELD,
    _amd_smi_field,
    _amd_smi_numbers,
    _tool_output,
    gpu_gfx_power_watts,
    mesa_installation,
    rocm_installation,
)


CARD_NAME = re.compile(r"^card\d+$")
ACCEL_NAME = re.compile(r"^accel\d+$")

VENDORS = {
    "0x1002": "AMD",
    "0x10de": "NVIDIA",
    "0x8086": "Intel",
    "0x1a03": "ASPEED",
    "0x102b": "Matrox",
}

# Display adapters that exist only to paint a console are not accelerators and
# must never be offered as a local-inference target.
DISPLAY_ONLY_VENDORS = {"0x1a03", "0x102b"}

# Confirmed devices worth naming precisely. Anything absent falls back to a
# vendor-and-driver description rather than an invented marketing name.
KNOWN_DEVICES = {
    ("0x1002", "0x1586"): "AMD Radeon 8060S (Strix Halo)",
    ("0x1022", "0x17f0"): "AMD Strix Halo Neural Processing Unit",
}

#: Kernel drivers that bind a neural accelerator rather than a GPU.
NPU_DRIVERS = ("amdxdna", "intel_vpu", "ivpu")

#: The character devices a container or service needs, the group that owns each
#: one on a stock Linux install, and which accelerator it belongs to.
#:
#: ``/dev/accel/accel0`` is owned by ``render`` — the *same* group as
#: ``/dev/kfd`` and ``/dev/dri/renderD128``. Group membership therefore cannot
#: separate NPU access from GPU access: anything in ``render`` can open both.
#: Independent grants require explicit per-device passthrough.
DEVICE_ACCESS = (
    ("/dev/kfd", "render", "gpu"),
    ("/dev/dri/renderD128", "render", "gpu"),
    ("/dev/dri/card0", "video", "gpu"),
    ("/dev/accel/accel0", "render", "npu"),
)


def _driver_name(device_root: Path) -> str:
    """Which kernel driver is bound to this device.

    ``uevent`` carries ``DRIVER=`` and is a plain file, so it is read first;
    the ``driver`` symlink is the fallback.
    """
    for line in text(device_root / "uevent").splitlines():
        if line.startswith("DRIVER="):
            return line.split("=", 1)[1].strip()
    link = device_root / "driver"
    try:
        if not link.exists():
            return ""
        return os.path.basename(os.path.realpath(link))
    except OSError:
        return ""


def _hwmon_root(device_root: Path) -> Optional[Path]:
    hwmon = device_root / "hwmon"
    try:
        entries = sorted(hwmon.iterdir())
    except OSError:
        return None
    for entry in entries:
        if entry.name.startswith("hwmon"):
            return entry
    return None


def _hwmon_readings(device_root: Path) -> Dict[str, Any]:
    hwmon = _hwmon_root(device_root)
    if hwmon is None:
        return {}
    readings: Dict[str, Any] = {"hwmon_name": text(hwmon / "name")}
    temperature = read_number(hwmon / "temp1_input")
    if temperature is not None:
        readings["temperature_c"] = round(temperature / 1000, 1)
        readings["temperature_label"] = text(hwmon / "temp1_label") or "edge"
    # power1_average is the smoothed reading the vendor tools report; fall back
    # to the instantaneous one when the driver only exposes that.
    for attribute in ("power1_average", "power1_input"):
        micro_watts = read_number(hwmon / attribute)
        if micro_watts is not None and micro_watts > 0:
            readings["power_watts"] = round(micro_watts / 1_000_000, 1)
            break
    frequency = read_number(hwmon / "freq1_input")
    if frequency is not None and frequency > 0:
        readings["clock_mhz"] = round(frequency / 1_000_000)
    return readings


#: Devices confirmed to be integrated parts whose VRAM carve-out and GTT
#: aperture are both slices of system RAM, so both count towards a model
#: budget. Keyed by PCI vendor and device id, like :data:`KNOWN_DEVICES`.
#:
#: A table is used because the two numbers alone cannot tell these apart: a
#: discrete card reports a GTT aperture too — that is host memory reached over
#: PCIe, which the GPU can map but cannot hold weights in at usable speed —
#: and it reports a VRAM total exactly as an APU does. The old expression
#: ``bool(gtt_total) and bool(vram_total)`` is true for *both*, so every amdgpu
#: card was called unified and a 24 GiB discrete card in a 64 GiB host was
#: credited 48 GiB and recommended a 32B model it cannot hold.
INTEGRATED_DEVICES = {
    ("0x1002", "0x1586"): "AMD Radeon 8060S (Strix Halo)",
}


def unified_memory_verdict(
    vendor_id: str,
    device_id: str,
    vram_total: Optional[float],
    gtt_total: Optional[float],
) -> Dict[str, Any]:
    """Whether this part's GTT aperture may count towards a model budget.

    Three cases, and the reason for each is reported rather than inferred by
    whoever reads the flag:

    * **No VRAM at all, but a GTT aperture.** Nothing is dedicated, so every
      byte it can address is system memory. Unified by definition, and no
      table is needed to know it.
    * **A confirmed integrated part.** Its carve-out is taken out of system
      RAM, so carve-out plus aperture is genuinely available for weights.
    * **Anything else.** Reported as *not* unified, which under-counts a
      unified part that is not yet in the table rather than over-promising a
      discrete one. That direction is deliberate: the failure mode of guessing
      wrong here is recommending a model the machine cannot load.
    """
    vram = int(vram_total or 0)
    gtt = int(gtt_total or 0)
    if not vram and gtt:
        return {
            "unified_memory": True,
            "unified_memory_source": "no-dedicated-vram",
            "unified_memory_reason": (
                "This part reports no dedicated video memory, so its shared "
                "aperture is the only memory it has."
            ),
        }
    if (vendor_id, device_id) in INTEGRATED_DEVICES:
        return {
            "unified_memory": True,
            "unified_memory_source": "known-integrated",
            "unified_memory_reason": (
                "{} is an integrated part whose carve-out and shared aperture "
                "both come out of system memory."
            ).format(INTEGRATED_DEVICES[(vendor_id, device_id)]),
        }
    if vram and gtt:
        return {
            "unified_memory": False,
            "unified_memory_source": "unrecognised-device",
            "unified_memory_reason": (
                "This part reports both dedicated video memory and a shared "
                "aperture, which a discrete card does too. Vaelor has not "
                "confirmed it is integrated, so only its dedicated memory is "
                "counted."
            ),
        }
    return {
        "unified_memory": False,
        "unified_memory_source": "dedicated-vram",
        "unified_memory_reason": (
            "This part has dedicated video memory and no shared aperture."
        ),
    }


def _card_record(card: Path) -> Optional[Dict[str, Any]]:
    device_root = card / "device"
    vendor_id = text(device_root / "vendor").lower()
    if not vendor_id:
        return None
    device_id = text(device_root / "device").lower()
    vendor = VENDORS.get(vendor_id, vendor_id or "unknown")
    driver = _driver_name(device_root)
    name = KNOWN_DEVICES.get((vendor_id, device_id))
    if not name:
        name = "{} graphics{}".format(
            vendor, " ({})".format(driver) if driver else ""
        )
    vram_total = read_number(device_root / "mem_info_vram_total")
    vram_used = read_number(device_root / "mem_info_vram_used")
    gtt_total = read_number(device_root / "mem_info_gtt_total")
    gtt_used = read_number(device_root / "mem_info_gtt_used")
    busy = read_number(device_root / "gpu_busy_percent")
    record: Dict[str, Any] = {
        "id": card.name,
        "name": name,
        "vendor": vendor,
        "vendor_id": vendor_id,
        "device_id": device_id,
        "driver": driver,
        "kind": "display-only" if vendor_id in DISPLAY_ONLY_VENDORS else "gpu",
        "vram_total_bytes": vram_total,
        "vram_used_bytes": vram_used,
        "gtt_total_bytes": gtt_total,
        "gtt_used_bytes": gtt_used,
        "busy_percent": busy,
        "link_speed": text(device_root / "current_link_speed") or None,
    }
    record.update(unified_memory_verdict(vendor_id, device_id, vram_total, gtt_total))
    record.update(_hwmon_readings(device_root))
    return record


def discover_accelerators(sys_root: str = "/sys") -> List[Dict[str, Any]]:
    """Enumerate DRM render devices. Returns ``[]`` when there are none."""
    drm = Path(sys_root) / "class" / "drm"
    try:
        cards = sorted(
            entry for entry in drm.iterdir() if CARD_NAME.match(entry.name)
        )
    except OSError:
        return []
    records = []
    for card in cards:
        record = _card_record(card)
        if record is not None and record["kind"] == "gpu":
            records.append(record)
    return records


def discover_npus(sys_root: str = "/sys") -> List[Dict[str, Any]]:
    """Enumerate compute accelerators bound to the kernel ``accel`` subsystem.

    A neural accelerator is not an inference backend Vaelor can target: neither
    ROCm nor llama.cpp compiles for one, and on the machine this was written
    against ``rocm_agent_enumerator`` reports only the GPU. It is discovered so
    a workload that *does* support it can be granted the device, and reported
    as unusable until a userspace runtime is present.
    """
    accel = Path(sys_root) / "class" / "accel"
    try:
        nodes = sorted(
            entry for entry in accel.iterdir() if ACCEL_NAME.match(entry.name)
        )
    except OSError:
        return []
    records = []
    for node in nodes:
        device_root = node / "device"
        vendor_id = text(device_root / "vendor").lower()
        device_id = text(device_root / "device").lower()
        driver = _driver_name(device_root)
        if driver and driver not in NPU_DRIVERS and not vendor_id:
            continue
        runtime = bool(shutil.which("xrt-smi") or Path("/opt/xilinx").exists())
        records.append({
            "id": node.name,
            "name": KNOWN_DEVICES.get(
                (vendor_id, device_id),
                "{} neural accelerator".format(VENDORS.get(vendor_id, "Unknown")),
            ),
            "vendor": VENDORS.get(vendor_id, vendor_id or "unknown"),
            "vendor_id": vendor_id,
            "device_id": device_id,
            "driver": driver,
            "kind": "npu",
            "device_node": "/dev/accel/{}".format(node.name),
            # The kernel publishes this beside every other attribute on the
            # device. It was previously reported as unreadable without vendor
            # tooling, which sent an operator to install a toolchain for a
            # value sitting in a file this function already had open.
            "firmware_version": text(device_root / "fw_version") or None,
            # Discovery is not capability. Without a userspace runtime nothing
            # can submit work, and claiming otherwise would be a lie the
            # operator cannot check.
            "runtime_detected": runtime,
            "usable_for_inference": False,
            # The runtime half of this reason is a live check, not a constant.
            # It used to assert that no userspace runtime was installed even on
            # a host where discovery had just found one.
            "reason": (
                (
                    "A userspace runtime is present, but no Vaelor inference "
                    "backend targets this device."
                    if runtime else
                    "The kernel driver is bound, but no userspace runtime was "
                    "found and no Vaelor inference backend targets this device."
                )
                + " It can be granted to a workload that supports it."
            ),
        })
    return records


def resolve_group_ids(
    names: List[str], member_reader: Optional[Callable[[str], Any]] = None
) -> Dict[str, Optional[int]]:
    """Map group names to numeric GIDs on *this* host.

    Container images do not share the host's group database, so
    ``--group-add render`` fails with "unable to find group render". Numeric
    GIDs are required, and they differ between hosts, so they must be resolved
    at runtime and never written into a template.
    """
    resolved: Dict[str, Optional[int]] = {}
    try:
        import grp
    except ImportError:  # Non-POSIX developer host.
        return {name: None for name in names}
    for name in names:
        try:
            resolved[name] = int(grp.getgrnam(name).gr_gid)
        except (KeyError, OSError, TypeError, ValueError):
            resolved[name] = None
    return resolved


def device_grants(
    kinds: Optional[List[str]] = None, devices=DEVICE_ACCESS
) -> Dict[str, Any]:
    """Devices and numeric GIDs needed to grant one accelerator kind.

    ``kinds`` selects which accelerators the caller wants. Because the GPU and
    the NPU share the ``render`` group, granting one by group membership grants
    both; the per-device list is what actually separates them.
    """
    selected = [
        entry for entry in devices
        if kinds is None or entry[2] in kinds
    ]
    present = [entry for entry in selected if Path(entry[0]).exists()]
    group_names: List[str] = []
    for _path, group, _kind in present:
        if group not in group_names:
            group_names.append(group)
    group_ids = resolve_group_ids(group_names)
    shared = sorted({
        group for _p, group, _k in devices
        if len({kind for _p2, g2, kind in devices if g2 == group}) > 1
    })
    return {
        "devices": [entry[0] for entry in present],
        "groups": group_names,
        "group_ids": group_ids,
        "unresolved_groups": [
            name for name, gid in group_ids.items() if gid is None
        ],
        "shared_groups": shared,
        "isolation_note": (
            "The {} group owns more than one accelerator kind, so group "
            "membership cannot separate them. Grant devices individually."
            .format(", ".join(shared))
            if shared
            else ""
        ),
    }


#: AMD ships ``amd-smi`` in two builds: Ubuntu's own answers ``N/A`` for every
#: APU metric on a Ryzen AI Max; AMD's ROCm build publishes them. ``which``
#: returns the PATH (Ubuntu) build, so preferring the ROCm build is what makes
#: the NPU readable and stops the false "not installed" a Z2 showed (VD-040) —
#: amd-smi's total absence is the only true "not installed".
ROCM_AMD_SMI = "/opt/rocm/bin/amd-smi"


def resolve_amd_smi(
    finder: Callable[[str], Optional[str]] = shutil.which,
    rocm_amd_smi: Optional[str] = None,
) -> Optional[str]:
    """The ``amd-smi`` to run, ROCm build first (see :data:`ROCM_AMD_SMI`), then
    ``finder`` on PATH; ``rocm_amd_smi`` overrides the ROCm path, read at call
    time so a test can redirect it without binding it into a default at import."""
    candidate = rocm_amd_smi if rocm_amd_smi is not None else ROCM_AMD_SMI
    if os.access(candidate, os.X_OK):
        return candidate
    return finder("amd-smi")


def enrich_from_vendor_tool(
    accelerators: List[Dict[str, Any]],
    finder: Callable[[str], Optional[str]] = shutil.which,
    runner: Callable[..., Any] = subprocess.run,
) -> Dict[str, Any]:
    """Optional ``amd-smi`` enrichment on top of the sysfs baseline.

    sysfs stays the source of truth: it is unprivileged, costs no subprocess,
    and works on a host with no ROCm installed at all. ``amd-smi`` adds facts
    the kernel does not expose — per-core package power, DRAM bandwidth, the
    marketing name — and its absence must change nothing.

    The executable comes from :func:`resolve_amd_smi` (ROCm build preferred).

    **This spawns a process and must never be called from the telemetry poll
    loop.** ``/stream/telemetry`` samples once a second per open browser tab
    and ``current_data`` is uncached, so a direct call here is one ``amd-smi``
    process per second per tab, for as long as a tab is open. Anything on the
    poll path goes through :func:`cached_vendor_tool_metrics`, which holds the
    result for :data:`VENDOR_TOOL_CACHE_SECONDS` and reports its own age.
    """
    executable = resolve_amd_smi(finder)
    if not executable:
        return {"available": False, "reason": "amd-smi is not installed.", "metrics": {}}
    try:
        result = runner(
            [executable, "metric", "--json"],
            capture_output=True, text=True, check=False, timeout=5,
        )
    except (OSError, subprocess.SubprocessError) as error:
        return {"available": False, "reason": str(error), "metrics": {}}
    if getattr(result, "returncode", 1) != 0:
        return {
            "available": False,
            "reason": "amd-smi exited with status {}.".format(result.returncode),
            "metrics": {},
        }
    parsed = _json_body(result.stdout)
    if parsed is None:
        return {
            "available": False,
            "reason": "amd-smi returned output that is not JSON.",
            "metrics": {},
        }
    return {"available": True, "reason": "", "metrics": parsed}


def _json_body(output: str) -> Any:
    """Parse ``amd-smi --json`` output that may be preceded by a banner.

    Measured: a reader outside the ``render`` group gets 535 bytes of
    "Permission needed to access required GPU device node(s)" on *stdout*,
    ahead of a complete and valid JSON document, and ``amd-smi`` still exits 0.
    ``json.loads`` threw, the whole result was discarded, and everything
    downstream — including the NPU power and activity fields, which need no GPU
    device node at all — read as though the tool were not installed. The banner
    is a fact about one account's groups; refusing the reading over it turns
    that into a claim about the hardware.
    """
    text_out = output or ""
    brace = text_out.find("{")
    for candidate in (text_out, text_out[brace:] if brace > 0 else ""):
        if not candidate.strip():
            continue
        try:
            return json.loads(candidate)
        except ValueError:
            continue
    return None


def accelerator_telemetry(
    accelerators: Optional[List[Dict[str, Any]]] = None,
    sys_root: str = "/sys",
    finder: Callable[[str], Optional[str]] = shutil.which,
    runner: Callable[..., Any] = subprocess.run,
    *,
    now: Optional[float] = None,
) -> Dict[str, Any]:
    """Return the ``gpu_*`` telemetry keys for the primary accelerator.

    Keys are omitted entirely when the underlying sysfs attribute is absent.
    A missing sensor is not zero, and reporting ``0`` for an unreadable GPU
    temperature is how a dashboard ends up claiming a running GPU is at 0 °C.

    Power is sourced by capability (see :func:`_add_gpu_power`): hwmon for a
    discrete card, the ``amd-smi`` graphics channel for an integrated Radeon.
    ``finder`` and ``runner`` reach that tool and let a test inject it; the poll
    path takes the default cached reader.
    """
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
    _add_gpu_power(telemetry, primary, finder, runner, now)
    if telemetry:
        telemetry["gpu_name"] = primary.get("name")
    return telemetry


def _add_gpu_power(
    telemetry: Dict[str, Any],
    primary: Dict[str, Any],
    finder: Callable[[str], Optional[str]],
    runner: Callable[..., Any],
    now: Optional[float],
) -> None:
    """Set ``gpu_power_watts`` from the channel that is correct for this part.

    On an integrated Radeon, ``amd-smi`` :data:`GPU_GFX_POWER_FIELD` is the
    graphics engine's own draw; the hwmon ``power1_average`` this module reads
    for a discrete card is SoC package power there and must NOT stand in for it
    (LESSONS 4 / VD-040 — the NPU_POWER_SUBSTITUTION_WARNING class). When the
    gfx channel is unreadable the reading is omitted, never back-filled with the
    package number.
    """
    integrated = primary.get("unified_memory") is True and primary.get("vendor") == "AMD"
    if integrated:
        metrics = cached_vendor_tool_metrics([primary], finder, runner, now=now)
        gfx = gpu_gfx_power_watts(metrics.get("metrics")) if metrics.get("available") else None
        if gfx is not None:
            telemetry["gpu_power_watts"] = gfx
            telemetry["gpu_power_source"] = GPU_GFX_POWER_FIELD
        return
    power = primary.get("power_watts")
    if power is not None:
        telemetry["gpu_power_watts"] = power
        telemetry["gpu_power_source"] = "hwmon power1_average"


def _group_members(name: str) -> Optional[List[str]]:
    try:
        import grp
    except ImportError:  # Non-POSIX developer host.
        return None
    try:
        return list(grp.getgrnam(name).gr_mem)
    except (KeyError, OSError):
        return None


def device_access(
    devices=DEVICE_ACCESS,
    service_user: str = "",
    member_reader=_group_members,
) -> Dict[str, Any]:
    """Report whether a service account could actually open the GPU.

    On a stock Ubuntu install ``/dev/kfd`` and ``/dev/dri/render*`` are owned by
    the ``render`` and ``video`` groups and those groups are empty, so GPU
    discovery succeeding tells you nothing about whether inference will run.
    Readiness reporting has to say so.
    """
    entries = []
    required_groups: list[str] = []
    for path, group, kind in devices:
        present = Path(path).exists()
        entries.append({
            "path": path, "group": group, "kind": kind, "present": present,
        })
        if present and group not in required_groups:
            required_groups.append(group)
    group_ids = resolve_group_ids(required_groups)
    groups = []
    for group in required_groups:
        members = member_reader(group)
        groups.append({
            "name": group,
            # Container images have no host group database, so a grant has to
            # be expressed as a numeric GID.
            "gid": group_ids.get(group),
            "exists": members is not None,
            "members": members or [],
            "grants_service_account": bool(
                service_user and members and service_user in members
            ),
        })
    present_devices = [entry for entry in entries if entry["present"]]
    empty = [group["name"] for group in groups if not group["members"]]
    if not present_devices:
        reason = "No GPU character devices are present on this host."
        ready = False
    elif not service_user:
        reason = (
            "Add the Vaelor service account to the {} group before the GPU can "
            "be used for inference.".format(" and ".join(g["name"] for g in groups))
            if groups
            else ""
        )
        ready = False
    elif empty or not all(group["grants_service_account"] for group in groups):
        reason = (
            "The {} group does not contain the Vaelor service account, so the "
            "GPU character devices cannot be opened.".format(
                " and ".join(empty or [g["name"] for g in groups])
            )
        )
        ready = False
    else:
        reason = ""
        ready = True
    return {
        "ready": ready,
        "reason": reason,
        "devices": entries,
        "groups": groups,
    }


#: ``amd-smi`` field names for the neural accelerator. Matching is
#: case-insensitive because the two output forms disagree: the plain-text
#: report prints ``APU_AVERAGE_IPU_ACTIVITY`` while ``--json`` emits
#: ``apu_average_ipu_activity``.
NPU_ACTIVITY_FIELD = "apu_average_ipu_activity"
NPU_POWER_FIELD = "apu_average_ipu_power"
NPU_CLOCK_FIELD = "apu_average_ipuclk_frequency"


def npu_activity_percent(metrics: Any) -> Optional[float]:
    """Neural-processor utilisation from ``amd-smi metric`` output.

    The obvious source is wrong: a Lemonade server's ``system-info`` endpoint
    reports ``devices.amd_npu.utilization`` as ``0.0`` while the device is
    genuinely busy, because that field is not wired on Linux. ``amd-smi``
    reports the activity as one value per engine; the maximum is the one that
    tracks a running model.

    Returns ``None`` when the field is absent. It is never reported as zero on
    the strength of a missing measurement.
    """
    numbers = _amd_smi_numbers(_amd_smi_field(metrics, NPU_ACTIVITY_FIELD))
    return round(max(numbers), 1) if numbers else None


def npu_power_watts(metrics: Any) -> Optional[float]:
    """Neural-processor power draw from ``amd-smi metric`` output.

    This reading was previously hard-coded as permanently unavailable on the
    strength of ``xrt-smi`` printing ``Estimated Power: N/A``. That was a claim
    about one tool, generalised into a claim about the hardware: ``amd-smi``
    publishes it, and a measured 2.15 W on a Ryzen AI Max is what proved the
    constant wrong. Absence is now decided by looking, every time.
    """
    numbers = _amd_smi_numbers(_amd_smi_field(metrics, NPU_POWER_FIELD))
    return round(max(numbers), 2) if numbers else None


def npu_clock_mhz(metrics: Any) -> Optional[float]:
    """Neural-processor clock frequency from ``amd-smi metric`` output."""
    numbers = _amd_smi_numbers(_amd_smi_field(metrics, NPU_CLOCK_FIELD))
    return round(max(numbers)) if numbers else None


#: BIOS setting that configures the VRAM carve-out, exposed by ``hp_bioscfg``.
#: Reading it turns "16 GiB carve-out" from something inferred out of
#: ``mem_info_vram_total`` into the firmware setting it actually is — which is
#: what the machine-class copy already claims in prose, and what lets an owner
#: know it is a setting they could change rather than a property of the part.
BIOSCFG_ROOT = "/sys/class/firmware-attributes/hp-bioscfg/attributes"
GRAPHICS_MEMORY_ATTRIBUTE = "Dedicated Graphics Memory"


def configured_graphics_memory(sys_root: str = "/sys") -> Dict[str, Any]:
    """The carve-out as a firmware setting, or a stated absence.

    Returns bytes alongside the raw string: the attribute reads like "16 GB",
    and a caller comparing it against ``mem_info_vram_total`` needs a number.
    Vendors are inconsistent about GB-versus-GiB here, so the raw value travels
    with it rather than being silently reinterpreted.
    """
    attribute = Path(sys_root) / "class" / "firmware-attributes" / (
        "hp-bioscfg"
    ) / "attributes" / GRAPHICS_MEMORY_ATTRIBUTE / "current_value"
    raw = text(attribute)
    if not raw:
        return {
            "available": False,
            "value": "",
            "bytes": None,
            "reason": (
                "This host does not expose the {} firmware attribute, so the "
                "carve-out size is only known from what the driver reports."
            ).format(GRAPHICS_MEMORY_ATTRIBUTE),
        }
    match = re.match(r"\s*(\d+)\s*([GMgm])i?[Bb]?\s*$", raw)
    scale = {"g": 1024 ** 3, "m": 1024 ** 2}
    size = (
        int(match.group(1)) * scale[match.group(2).lower()] if match else None
    )
    return {
        "available": True,
        "value": raw,
        "bytes": size,
        "source": "hp-bioscfg/{}".format(GRAPHICS_MEMORY_ATTRIBUTE),
        "reason": "",
    }


#: Fields worth taking from the rest of ``amd-smi metric``. Deliberately not
#: the 16 per-core power, clock and C0 values: 48 numbers per poll is a cost
#: with no reader, and anything wanting them can ask for a one-off report.
AMD_SMI_HEALTH_FIELDS = (
    ("gpu_soc_temperature_c", "soc_temperature"),
    ("gpu_dram_bandwidth_gbps", "dram_bandwidth"),
    ("system_power_watts", "psys"),
)


#: Telemetry key for neural-processor utilisation. Named separately from the
#: ``gpu_*`` family because it comes from a different device and a different
#: source: sysfs publishes nothing for the NPU, so this is the one telemetry
#: reading on the strip that depends on a vendor tool being installed.
NPU_ACTIVITY_TELEMETRY_KEY = "npu_activity_percent"

#: Power and clock, computed by this module since the NPU work began and
#: published nowhere. The card said *"This device reports no power
#: measurement"* while ``apu_average_ipu_power`` sat in the output of a tool
#: the appliance already ran every ten seconds.
NPU_POWER_TELEMETRY_KEY = "npu_power_watts"
NPU_CLOCK_TELEMETRY_KEY = "npu_clock_mhz"

#: Why there is no reading, when there is none — and the distinction is not
#: academic. Measured: **Ubuntu 26.04's own ``amd-smi`` (7.2.0-3, library
#: 26.2.1) reports the whole ``usage`` block as the string ``"N/A"`` on a Ryzen
#: AI Max and publishes no ``apu_*`` field at all**, while AMD's ROCm build
#: (26.5.0) on the same machine in the same minute publishes every one. "Install
#: amd-smi" alone would send someone to the build that does not work.
NPU_TOOL_ABSENT = (
    "amd-smi is not installed, so neural-processor activity and power cannot "
    "be read. The kernel publishes neither for this device."
)
NPU_FIELDS_ABSENT = (
    "amd-smi is installed but publishes no APU metrics for this device: the "
    "fields Vaelor reads ({}) are absent from its output. Ubuntu's own build "
    "behaves this way on a Ryzen AI Max; AMD's ROCm build publishes them."
).format(", ".join((NPU_ACTIVITY_FIELD, NPU_POWER_FIELD)))
NPU_REASON_TELEMETRY_KEY = "npu_reading_reason"


#: How long one ``amd-smi`` result is reused. The tool spawns a process and the
#: telemetry stream samples once a second per open browser tab, so an uncached
#: read is process churn that scales with how many tabs are open. Ten seconds
#: bounds it to roughly six spawns a minute, and the reading's age is published
#: so a stale value is visible rather than presented as live.
VENDOR_TOOL_CACHE_SECONDS = 10.0

_VENDOR_TOOL_LOCK = threading.Lock()
_VENDOR_TOOL_CACHE: Dict[str, Any] = {}


def reset_vendor_tool_cache() -> None:
    """Drop the memoised ``amd-smi`` result (tests, and after a driver change)."""
    with _VENDOR_TOOL_LOCK:
        _VENDOR_TOOL_CACHE.clear()


def cached_vendor_tool_metrics(
    accelerators: Optional[List[Dict[str, Any]]] = None,
    finder: Callable[[str], Optional[str]] = shutil.which,
    runner: Callable[..., Any] = subprocess.run,
    *,
    now: Optional[float] = None,
    ttl: float = VENDOR_TOOL_CACHE_SECONDS,
) -> Dict[str, Any]:
    """``amd-smi`` output, reused for ``ttl`` seconds, with its age attached.

    The poll path's entry point. ``sampled_at`` and ``age_seconds`` travel with
    the result so a caller can report how old the reading is instead of
    implying it was taken now.
    """
    moment = time.time() if now is None else float(now)
    with _VENDOR_TOOL_LOCK:
        cached = _VENDOR_TOOL_CACHE.get("entry")
        if cached is not None and 0 <= moment - cached["sampled_at"] < ttl:
            return {**cached, "age_seconds": round(moment - cached["sampled_at"], 3)}
    metrics = enrich_from_vendor_tool(accelerators or [], finder, runner)
    entry = {**metrics, "sampled_at": moment}
    with _VENDOR_TOOL_LOCK:
        _VENDOR_TOOL_CACHE["entry"] = entry
    return {**entry, "age_seconds": 0.0}


def neural_accelerator_telemetry(
    sys_root: str = "/sys",
    finder: Callable[[str], Optional[str]] = shutil.which,
    runner: Callable[..., Any] = subprocess.run,
    records: Optional[List[Dict[str, Any]]] = None,
    *,
    now: Optional[float] = None,
) -> Dict[str, Any]:
    """Neural-processor utilisation, or nothing at all.

    The key is omitted whenever the reading cannot be taken: no NPU discovered,
    ``amd-smi`` not installed, the tool failing, or the field printing ``N/A``
    as it does on a host where it is not wired. Emitting ``0`` instead would be
    indistinguishable from a genuinely idle accelerator, and an invented idle
    reading is precisely the fabrication this telemetry path exists to avoid —
    the same mistake that once had a dashboard reporting a running GPU at 0 °C.

    ``APU_AVERAGE_IPU_ACTIVITY`` publishes one value per engine; the maximum is
    the device's utilisation, which is what :func:`npu_activity_percent` reads.

    Activity, power and clock are decided **one at a time**: one call, three
    fields, and a device publishing two of them must not lose all three because
    the third printed ``N/A``.

    A zero here is a reading, not a gap. On the target machine
    ``apu_average_ipu_power`` read 0.0 W with every engine at 0% while
    ``xrt-smi examine -r aie-partitions`` independently reported no hardware
    context on the device — two sources agreeing an idle accelerator is idle,
    which is what makes the zero publishable. The same field read 2.15 W with a
    model loaded, which is what proves it is wired at all.
    """
    found = discover_npus(sys_root) if records is None else records
    if not found:
        return {}
    # Cached: this runs on the telemetry poll path, and `amd-smi` is a process.
    metrics = cached_vendor_tool_metrics(found, finder, runner, now=now)
    if not metrics.get("available"):
        return {NPU_REASON_TELEMETRY_KEY: NPU_TOOL_ABSENT}
    raw = metrics.get("metrics")
    readings = (
        # value key, source key, reading, and the amd-smi field it came from.
        # Every reading names its own field (VD-040): a watt figure from the
        # NPU's own counter and one inferred from anything else are different
        # claims and must not arrive looking alike.
        (NPU_ACTIVITY_TELEMETRY_KEY, "npu_activity_source",
         npu_activity_percent(raw), NPU_ACTIVITY_FIELD),
        (NPU_POWER_TELEMETRY_KEY, "npu_power_source",
         npu_power_watts(raw), NPU_POWER_FIELD),
        (NPU_CLOCK_TELEMETRY_KEY, "npu_clock_source",
         npu_clock_mhz(raw), NPU_CLOCK_FIELD),
    )
    telemetry: Dict[str, Any] = {}
    for key, source_key, value, field in readings:
        if value is None:
            continue
        telemetry[key] = value
        telemetry[source_key] = field
    if not telemetry:
        return {NPU_REASON_TELEMETRY_KEY: NPU_FIELDS_ABSENT}
    telemetry["npu_name"] = found[0].get("name")
    # How old the whole `amd-smi` result is; all three readings came from one
    # call. A cached value presented as live is a smaller lie than a fabricated
    # one, but it is still a lie.
    telemetry["npu_activity_age_seconds"] = metrics.get("age_seconds", 0.0)
    return telemetry


def gpu_firmware(
    finder: Callable[[str], Optional[str]] = shutil.which,
    runner: Callable[..., Any] = subprocess.run,
) -> Dict[str, Any]:
    """Adapter firmware (IFWI) identity from ``amd-smi static``."""
    output = _tool_output(finder, runner, "amd-smi", ["static", "--json"])
    try:
        parsed = json.loads(output or "null")
    except ValueError:
        parsed = None
    record = _amd_smi_field(parsed, "ifwi")
    if not isinstance(record, dict):
        return {
            "version": None,
            "name": None,
            "part_number": None,
            "build_date": None,
            "note": (
                "amd-smi did not report adapter firmware on this host."
            ),
        }
    return {
        "version": str(record.get("version") or "") or None,
        "name": str(record.get("name") or "") or None,
        "part_number": str(record.get("part_number") or "") or None,
        "build_date": str(record.get("build_date") or "") or None,
        "note": "",
    }


def graphics_inventory(
    sys_root: str = "/sys",
    rocm_root: str = "/opt/rocm",
    finder: Callable[[str], Optional[str]] = shutil.which,
    runner: Callable[..., Any] = subprocess.run,
) -> Dict[str, Any]:
    """Static graphics facts for an inventory panel.

    Only values readable without a subprocess and without root appear here.
    A version this module does not read is reported as ``None`` with a note
    saying so, rather than guessed at: an inventory panel showing a plausible
    wrong version is worse than one showing a gap. The note describes what was
    read, never whether the value could be obtained some other way.
    """
    accelerators = discover_accelerators(sys_root)
    npus = discover_npus(sys_root)
    rocm = rocm_installation(rocm_root, finder, runner)
    mesa = mesa_installation(finder, runner)
    firmware = gpu_firmware(finder, runner)
    adapters = [
        {
            "name": card["name"],
            "vendor": card["vendor"],
            "driver": card["driver"],
            "pci_id": "{}:{}".format(
                card["vendor_id"].removeprefix("0x"),
                card["device_id"].removeprefix("0x"),
            ),
            "vram_total_bytes": card["vram_total_bytes"],
            "gtt_total_bytes": card["gtt_total_bytes"],
            "link_speed": card["link_speed"],
            "unified_memory": card["unified_memory"],
            "firmware": firmware,
        }
        for card in accelerators
    ]
    return {
        "adapters": adapters,
        "neural_accelerators": [
            {
                "name": npu["name"],
                "driver": npu["driver"],
                "device_node": npu["device_node"],
                "runtime_detected": npu["runtime_detected"],
                "firmware_version": npu["firmware_version"],
                # What was looked at, and where. A note naming the path it read
                # is checkable and corrects itself the moment the path is
                # wrong; "vendor tooling is required to read it" was neither,
                # and was false - the value is in this file.
                "firmware_note": (
                    "" if npu["firmware_version"] else
                    "No firmware version was readable at {}/device/fw_version."
                    .format(npu["id"])
                ),
            }
            for npu in npus
        ],
        "rocm_version": rocm["version"],
        "rocm_source": rocm["source"],
        "rocm_note": rocm["note"],
        "mesa_version": mesa["version"],
        "mesa_source": mesa["source"],
        "mesa_note": mesa["note"],
        "adapter_firmware": firmware,
    }


def accelerator_summary(
    sys_root: str = "/sys", service_user: str = ""
) -> Dict[str, Any]:
    """Discovery plus access readiness, for setup and acceptance reporting."""
    accelerators = discover_accelerators(sys_root)
    npus = discover_npus(sys_root)
    return {
        "accelerators": accelerators,
        "detected": bool(accelerators),
        "npus": npus,
        "npu_detected": bool(npus),
        "access": device_access(service_user=service_user),
        "grants": {
            "gpu": device_grants(["gpu"]),
            "npu": device_grants(["npu"]),
        },
    }
