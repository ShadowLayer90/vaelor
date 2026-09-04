"""Vendor software and adapter firmware: what is installed, not what is wired.

Split out of ``accelerators.py``, which had grown past the thousand-line limit
this project holds itself to. The seam is real rather than arbitrary:
everything here answers "what software is on this host and what firmware is on
the adapter", by shelling out to optional tools and reading version files.
Everything left behind answers "what hardware is present and what is it doing",
from sysfs, with no subprocess at all.

One rule carries across the split, and it was learned the expensive way: report
what was read and where it was read from. Do not assert what the hardware can
or cannot do.
"""

from __future__ import annotations

import re
import shutil
import subprocess
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional

from .base import text


def _tool_output(
    finder: Callable[[str], Optional[str]],
    runner: Callable[..., Any],
    name: str,
    arguments: List[str],
    timeout: int = 5,
) -> str:
    """Run one optional CLI and return its stdout, or ``""``.

    Absence of the tool is not an error and never raises: every caller has a
    filesystem answer or an honest gap to fall back to.
    """
    executable = finder(name)
    if not executable:
        return ""
    try:
        result = runner(
            [executable, *arguments],
            capture_output=True, text=True, check=False, timeout=timeout,
        )
    except (OSError, subprocess.SubprocessError):
        return ""
    if getattr(result, "returncode", 1) != 0:
        return ""
    return result.stdout or ""


def _version_key(name: str) -> tuple:
    return tuple(int(part) for part in re.findall(r"\d+", name)) or (0,)


def rocm_installation(
    rocm_root: str = "/opt/rocm",
    finder: Callable[[str], Optional[str]] = shutil.which,
    runner: Callable[..., Any] = subprocess.run,
) -> Dict[str, Any]:
    """Find the installed ROCm version, and say where the answer came from.

    This used to read ``/opt/rocm/.info/version`` and nothing else. That file
    does not exist on a current install: ROCm ships versioned directories
    behind ``update-alternatives``, so on a machine with ROCm 7.14 present the
    inventory confidently reported no ROCm at all. Looking in one place and
    reporting absence is the same defect shape as a hard-coded "unavailable" -
    the answer sounds settled and nobody re-checks the path.

    ``core/.info/version`` is preferred because the alternatives link survives
    an upgrade. The versioned directories are searched next, highest first, so
    a host mid-upgrade reports the newest it actually has.
    """
    root = Path(rocm_root)
    stable = root / "core" / ".info" / "version"
    version = text(stable)
    if version:
        return {"version": version, "source": str(stable), "note": ""}
    candidates = []
    try:
        entries = list(root.iterdir())
    except OSError:
        entries = []
    for entry in entries:
        if not entry.name.startswith("core-"):
            continue
        found = text(entry / ".info" / "version")
        if found:
            candidates.append(
                (_version_key(entry.name), found, str(entry / ".info" / "version"))
            )
    if candidates:
        _key, found, source = max(candidates)
        return {"version": found, "source": source, "note": ""}
    # Layouts before the alternatives split kept the file at the root.
    legacy = root / ".info" / "version"
    version = text(legacy)
    if version:
        return {"version": version, "source": str(legacy), "note": ""}
    reported = re.search(
        r"ROCm version:\s*([0-9][0-9A-Za-z.\-]*)",
        _tool_output(finder, runner, "amd-smi", ["version"]),
    )
    if reported:
        return {
            "version": reported.group(1),
            "source": "amd-smi version",
            "note": "",
        }
    packaged = re.search(
        r"^\S*rocm\S*\s+(\S+)$",
        _tool_output(
            finder, runner, "dpkg-query",
            ["-W", "-f=${Package} ${Version}\n", "*rocm*"],
        ),
        re.IGNORECASE | re.MULTILINE,
    )
    if packaged:
        return {
            "version": packaged.group(1),
            "source": "dpkg-query",
            "note": "",
        }
    return {
        "version": None,
        "source": "",
        "note": (
            "No ROCm version was found under {} or reported by amd-smi or "
            "dpkg-query on this host.".format(rocm_root)
        ),
    }


def mesa_installation(
    finder: Callable[[str], Optional[str]] = shutil.which,
    runner: Callable[..., Any] = subprocess.run,
) -> Dict[str, Any]:
    """Read the Mesa userspace version from the package database.

    Reported for a long time as "not readable without a display connection".
    It is readable without one: ``dpkg-query`` answers from the package
    database with no GPU context, no X or Wayland connection and no graphics
    stack involved. ``vulkaninfo`` agrees with it and needs far more.
    """
    output = _tool_output(
        finder, runner, "dpkg-query",
        ["-W", "-f=${Version}\n", "mesa-vulkan-drivers"],
    )
    version = output.strip().splitlines()[0].strip() if output.strip() else ""
    if version:
        return {
            "version": version,
            "source": "dpkg-query mesa-vulkan-drivers",
            "note": "",
        }
    return {
        "version": None,
        "source": "",
        "note": (
            "Vaelor found no mesa-vulkan-drivers package on this host, so no "
            "graphics userspace version is reported."
        ),
    }


def _amd_smi_field(metrics: Any, field: str) -> Any:
    """Find one ``amd-smi`` field anywhere in its output, or return ``None``.

    The JSON and text forms nest differently - ``--json`` puts these under
    ``gpu_data[0].usage`` and ``gpu_data[0].power`` while the text form is flat
    - so the field is searched for by name rather than by path. On a host with
    several devices this finds the first, which is the same device the rest of
    the accelerator module reports as primary.
    """
    wanted = field.lower()
    if isinstance(metrics, dict):
        for key, value in metrics.items():
            if str(key).lower() == wanted:
                return value
            found = _amd_smi_field(value, field)
            if found is not None:
                return found
    elif isinstance(metrics, list):
        for item in metrics:
            found = _amd_smi_field(item, field)
            if found is not None:
                return found
    return None


def _amd_smi_numbers(raw: Any) -> List[float]:
    """Unwrap an ``amd-smi`` reading into plain numbers.

    ``--json`` wraps every measurement as ``{"value": 97, "unit": "%"}`` while
    the text form yields bare numbers, and either may be a list or a scalar.
    Anything non-numeric - notably the literal ``"N/A"`` amd-smi prints for a
    field the device does not publish - is dropped, so an unreadable sensor
    ends up absent rather than coerced to zero.
    """
    items = raw if isinstance(raw, list) else [raw]
    numbers: List[float] = []
    for item in items:
        value = item.get("value") if isinstance(item, dict) else item
        if isinstance(value, (int, float)) and not isinstance(value, bool):
            numbers.append(float(value))
    return numbers


#: ``amd-smi`` field carrying the integrated GPU's *own* graphics-engine power.
#: Named separately from the amdgpu hwmon ``power1_average`` because on an APU
#: those two measure different things: hwmon reports whole-SoC package power
#: (measured 29 W on a Strix Halo Z2 with the GPU 0% busy) while this channel
#: reports the graphics engine's actual draw (0.03 W in the same instant). The
#: matching NPU channel is ``apu_average_ipu_power``; this is its GPU sibling.
GPU_GFX_POWER_FIELD = "apu_average_gfx_power"


def gpu_gfx_power_watts(metrics: Any) -> Optional[float]:
    """Integrated-GPU graphics-engine power from ``amd-smi metric`` output.

    Returns ``None`` when :data:`GPU_GFX_POWER_FIELD` is absent, so a caller can
    report an honest gap rather than fall back to the SoC package number. It is
    never reported as zero on the strength of a missing measurement.
    """
    numbers = _amd_smi_numbers(_amd_smi_field(metrics, GPU_GFX_POWER_FIELD))
    return round(max(numbers), 2) if numbers else None


#: ``amd-smi`` field carrying whole-socket (package) power. Read so the
#: Processor card can take its package figure from the *same* amd-smi snapshot
#: the GPU and NPU cards read, rather than from RAPL sampled on a different
#: clock — two correct instruments at two instants look out of step. Measured
#: agreement on a Strix Halo Z2: RAPL package-0 8.86 W vs this field 8.85 W.
PACKAGE_SOCKET_POWER_FIELD = "apu_average_socket_power"


def socket_power_watts(metrics: Any) -> Optional[float]:
    """Whole-socket (package) power from ``amd-smi metric`` output.

    Returns ``None`` when :data:`PACKAGE_SOCKET_POWER_FIELD` is absent, which is
    the signal to fall back to the RAPL energy-delta reading. Never zero on the
    strength of a missing measurement.
    """
    numbers = _amd_smi_numbers(_amd_smi_field(metrics, PACKAGE_SOCKET_POWER_FIELD))
    return round(max(numbers), 2) if numbers else None
