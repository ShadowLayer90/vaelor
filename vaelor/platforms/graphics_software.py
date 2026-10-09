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

import json
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


#: The units Vaelor reads for each kind of quantity, spelled exactly as printed
#: (case matters: ``mW`` is a milliwatt and ``MW`` a megawatt - review S-18),
#: with the factor that turns a value in that unit into the unit the field is
#: stored in (watts, degrees Celsius, megahertz, percent). ``amd-smi 26.5``
#: prints ``W``, ``C``, ``MHz`` and ``%``; ``mW`` is handled because other builds
#: print it for small rails, and reading 440 mW as 440 W is the defect this
#: table prevents.
POWER_UNITS: Dict[str, float] = {"W": 1.0, "mW": 0.001}
TEMPERATURE_UNITS: Dict[str, float] = {"C": 1.0, "°C": 1.0}
FREQUENCY_UNITS: Dict[str, float] = {"MHz": 1.0, "GHz": 1000.0, "kHz": 0.001}
PERCENT_UNITS: Dict[str, float] = {"%": 1.0}

#: How a wrapped value that names no unit at all is reported among the unknown
#: units: a wrapper exists to carry a unit, so one without is not read as the
#: base unit by default.
NO_UNIT = "(no unit)"


def _amd_smi_numbers(
    raw: Any, units: Optional[Dict[str, float]] = None,
    unknown: Optional[List[str]] = None,
) -> List[float]:
    """Unwrap an ``amd-smi`` reading into plain numbers, in the stored unit.

    ``--json`` wraps every measurement as ``{"value": 97, "unit": "%"}`` while
    the text form yields bare numbers, and either may be a list or a scalar.
    Anything non-numeric - notably the literal ``"N/A"`` amd-smi prints for a
    field the device does not publish - is dropped, so an unreadable sensor
    ends up absent rather than coerced to zero.

    **The unit is read, not assumed** (VD-147, review S13; LESSONS 5). ``units``
    is the table for this quantity: a wrapped value is scaled by its unit's
    factor, and a value in a unit the table does not hold is **dropped** - a
    number in an unknown unit is not a reading - with the unit's name appended
    to ``unknown`` when the caller passes a list, so the caller can say so.
    Units are matched exactly as printed. A wrapper that names no unit is not
    taken as the base unit: it is reported as :data:`NO_UNIT`. A bare number
    (the text form carries no wrapper at all) is taken as printed.
    """
    items = raw if isinstance(raw, list) else [raw]
    numbers: List[float] = []
    for item in items:
        value = item.get("value") if isinstance(item, dict) else item
        if not isinstance(value, (int, float)) or isinstance(value, bool):
            continue
        if units is not None and isinstance(item, dict):
            unit = item.get("unit")
            unit = unit.strip() if isinstance(unit, str) else ""
            factor = units.get(unit) if unit else None
            if factor is None:
                if unknown is not None:
                    unknown.append(unit or NO_UNIT)
                continue
            value = float(value) * factor
        numbers.append(float(value))
    return numbers


#: ``amd-smi`` field carrying the integrated GPU's *own* graphics-engine power.
#: Named separately from the amdgpu hwmon ``power1_average`` because on an APU
#: those two measure different things: hwmon reports whole-SoC package power
#: (measured 29 W on a Strix Halo Z2 with the GPU 0% busy) while this channel
#: reports the graphics engine's actual draw (0.03 W in the same instant). The
#: matching NPU channel is ``apu_average_ipu_power``; this is its GPU sibling.
GPU_GFX_POWER_FIELD = "apu_average_gfx_power"


def gpu_gfx_power_watts(metrics: Any, unknown: Optional[List[str]] = None) -> Optional[float]:
    """Integrated-GPU graphics-engine power from ``amd-smi metric`` output.

    Returns ``None`` when :data:`GPU_GFX_POWER_FIELD` is absent, so a caller can
    report an honest gap rather than fall back to the SoC package number. It is
    never reported as zero on the strength of a missing measurement.
    """
    numbers = _amd_smi_numbers(
        _amd_smi_field(metrics, GPU_GFX_POWER_FIELD), POWER_UNITS, unknown
    )
    return round(max(numbers), 2) if numbers else None


#: ``amd-smi`` field carrying whole-socket (package) power. Read so the
#: Processor card can take its package figure from the *same* amd-smi snapshot
#: the GPU and NPU cards read, rather than from RAPL sampled on a different
#: clock — two correct instruments at two instants look out of step. Measured
#: agreement on a Strix Halo Z2: RAPL package-0 8.86 W vs this field 8.85 W.
PACKAGE_SOCKET_POWER_FIELD = "apu_average_socket_power"


def socket_power_watts(metrics: Any, unknown: Optional[List[str]] = None) -> Optional[float]:
    """Whole-socket (package) power from ``amd-smi metric`` output.

    Returns ``None`` when :data:`PACKAGE_SOCKET_POWER_FIELD` is absent, which is
    the signal to fall back to the RAPL energy-delta reading. Never zero on the
    strength of a missing measurement.
    """
    numbers = _amd_smi_numbers(
        _amd_smi_field(metrics, PACKAGE_SOCKET_POWER_FIELD), POWER_UNITS, unknown
    )
    return round(max(numbers), 2) if numbers else None


#: ``amd-smi`` field carrying the graphics engine's own temperature. The
#: kernel's hwmon ``temp1_input`` for the same device is the *edge* sensor, a
#: different reading; the two are stored under different keys and the one owner
#: that chooses between them is `platforms.gpu_temperature`.
GPU_GFX_TEMPERATURE_FIELD = "apu_temperature_gfx"


def gpu_gfx_temperature_c(metrics: Any, unknown: Optional[List[str]] = None) -> Optional[float]:
    """Graphics-engine temperature from ``amd-smi metric`` output, in Celsius.

    ``None`` when :data:`GPU_GFX_TEMPERATURE_FIELD` is absent or prints ``N/A``:
    an unread sensor is a gap, never a zero and never the edge reading passed
    off under this name.
    """
    numbers = _amd_smi_numbers(
        _amd_smi_field(metrics, GPU_GFX_TEMPERATURE_FIELD), TEMPERATURE_UNITS, unknown
    )
    return round(max(numbers), 2) if numbers else None


#: THE table of what is taken from ``amd-smi metric`` for the GPU: the stored
#: telemetry field and the function that reads it. One table for the controller
#: and the worker's sampler (`gpu_vendor_status.readings_from`), holding only
#: fields something reads. It replaces `accelerators.AMD_SMI_HEALTH_FIELDS`,
#: three entries nothing ever read (VD-147, review nit 1; LESSONS 6) - so the
#: GPU has exactly two temperature keys, edge (``gpu_temperature_c``, from
#: hwmon) and graphics engine (``gpu_gfx_temperature_c``, from here).
AMD_SMI_GPU_FIELDS = (
    ("gpu_power_watts", gpu_gfx_power_watts),
    ("gpu_gfx_temperature_c", gpu_gfx_temperature_c),
    ("gpu_socket_power_watts", socket_power_watts),
)

#: How long one ``amd-smi`` run may take before it is abandoned, in seconds.
#: The measured run is 64-85 ms on both boxes; the owner sentence for a
#: time-out (`gpu_vendor_status`) states this figure.
VENDOR_TOOL_TIMEOUT_SECONDS = 5

#: The first line of the banner ``amd-smi`` prints on stdout, ahead of its
#: output and still exiting 0, when the caller lacks access to the GPU device
#: nodes. Matched as captured on both boxes (2026-09-29).
ACCESS_BANNER = "Permission needed to access required GPU device node(s)"


def integrated_amd_gpu(record: Any) -> bool:
    """Whether a discovered accelerator is an AMD integrated GPU.

    The one predicate behind every "this part's vendor readings come from the
    graphics-engine channels" choice: a unified-memory AMD device. Discovery
    sets ``unified_memory`` from the device table, never from a name, so this
    is capability discovery and not a platform assumption.
    """
    return (
        isinstance(record, dict)
        and record.get("unified_memory") is True
        and record.get("vendor") == "AMD"
    )


def json_after_banner(output: Any) -> Any:
    """Parse ``amd-smi --json`` output that may be preceded by a text banner.

    ``amd-smi`` prints a "Permission needed to access required GPU device
    node(s)" banner on *stdout*, ahead of a complete JSON document, and still
    exits 0. The document is an object for ``metric`` and a list for
    ``monitor``, so the body is looked for at the first line that opens either.
    ``None`` when no line starts a document that parses.
    """
    text_out = output if isinstance(output, str) else ""
    if not text_out.strip():
        return None
    try:
        return json.loads(text_out)
    except ValueError:
        pass
    offset = 0
    for line in text_out.splitlines(keepends=True):
        if line.lstrip()[:1] in ("{", "["):
            try:
                return json.loads(text_out[offset:])
            except ValueError:
                pass
        offset += len(line)
    return None
