"""The non-root GPU vendor sampler a worker runs (VD-147, owner decision D1).

The graphics engine's own power and temperature come from ``amd-smi`` and
nowhere else. The worker's telemetry emitter must never spawn a process (it
runs once a second inside Telegraf, as root), so a second, separate service
runs the tool: this one. It is a long-running loop that, every ten seconds,
runs ``amd-smi metric --json`` and writes what it read to one small file the
emitter picks up.

**It runs without root.** Its unit (`worker_telemetry_config.render_sampler_unit`)
is ``DynamicUser`` in the ``render`` and ``video`` groups with no network, no
capabilities and only the DRM device nodes allowed - the exact property set a
probe on both boxes showed ``amd-smi`` works under (S0, 2026-09-29). systemd
creates ``/run/vaelor-gpu`` for it and removes it when it stops.

**What it writes**, atomically (``mkstemp`` in the runtime directory, ``fsync``,
``os.replace``), mode ``0644``::

    {"sampled_at": <epoch seconds>, "status": <code>, "values": {...},
     "limits": {"edge": {"warning_c": ..., "critical_c": ...}}}

``status`` is a `gpu_vendor_status` code; ``values`` hold only fields that were
read (absent, never zero); ``limits`` are the temperature limits the GPU
reports for itself, by sensor, and the block is empty when it reports none -
never invented.

**Nothing is discarded.** Both stdout and stderr of the tool are captured, and
the classification (`gpu_vendor_status.read_vendor`) is the same function the
controller uses for its own call.

Its own entry module and its own bundle (`worker_telemetry_bundle`): the
emitter's bundle, and its guarantee that it never spawns a subprocess, are
untouched. Standard library only.
"""

from __future__ import annotations

import json
import os
import subprocess
import sys
import tempfile
import time
from typing import Any, Callable, Dict, Optional

from .gpu_vendor_status import (
    CODES_WITH_READINGS, UNREADABLE_OUTPUT, VENDOR_SAMPLE_INTERVAL_SECONDS, VENDOR_TOOL_TIMEOUT_SECONDS,
    read_vendor,
)
from .platforms.accelerators import discover_accelerators, resolve_amd_smi
from .platforms.gpu_temperature import limits_from_records

#: The file the emitter reads, inside the unit's runtime directory. Spelled
#: here rather than imported from `gpu_vendor_sample` (the reader) so this
#: bundle does not carry the reader; a test pins the two names together.
SAMPLE_FILE_NAME = "gpu-vendor.json"

#: Where systemd puts the runtime directory's path for the service.
RUNTIME_DIRECTORY_ENV = "RUNTIME_DIRECTORY"

#: The longest line the sampler writes to its log (the journal, through the
#: unit's stderr). amd-smi can print pages; the journal needs the first line.
LOG_LINE_MAX = 300


def _log_stderr(line: str) -> None:
    sys.stderr.write(line + "\n")
    sys.stderr.flush()


def _bounded(prefix: str, text: Any) -> str:
    """One log line: the prefix and the text's first part, never longer than the bound."""
    words = " ".join(str(text or "").split())
    return (prefix + words)[:LOG_LINE_MAX]


def run_vendor_tool(
    executable: Optional[str], runner: Callable[..., Any] = subprocess.run,
    log: Callable[[str], Any] = _log_stderr,
) -> Dict[str, Any]:
    """One bounded ``amd-smi metric --json`` run, classified.

    Returns `gpu_vendor_status.read_vendor`'s answer. A tool that is not
    installed, one that does not answer within the timeout and one that cannot
    be started are three different codes, and none of them raises. When the
    tool fails, the first part of its own error text goes to the log (review
    S-21): it was captured and thrown away.
    """
    if not executable:
        return read_vendor("", installed=False)
    try:
        result = runner(
            [executable, "metric", "--json"],
            capture_output=True, text=True, check=False,
            timeout=VENDOR_TOOL_TIMEOUT_SECONDS,
        )
    except subprocess.TimeoutExpired:
        return read_vendor("", timed_out=True)
    except (OSError, subprocess.SubprocessError):
        return read_vendor("", returncode=1)
    returncode = int(getattr(result, "returncode", 1))
    if returncode != 0:
        log(_bounded("amd-smi exited with status {}: ".format(returncode), getattr(result, "stderr", "")))
    return read_vendor(getattr(result, "stdout", "") or "", returncode=returncode)


#: The GPU's own temperature limits by sensor; the controller reads its own
#: through the same function (`platforms.gpu_temperature.limits_from_records`).
temperature_limits = limits_from_records


def build_sample(
    *, now: float, sys_root: str = "/sys",
    finder: Optional[Callable[[str], Optional[str]]] = None,
    runner: Callable[..., Any] = subprocess.run,
) -> Dict[str, Any]:
    """One sample, ready to write."""
    executable = resolve_amd_smi(finder) if finder is not None else resolve_amd_smi()
    reading = run_vendor_tool(executable, runner)
    values = reading["values"] if reading["status"] in CODES_WITH_READINGS else {}
    return {
        "sampled_at": float(now),
        "status": int(reading["status"]),
        "values": dict(values),
        "limits": temperature_limits(discover_accelerators(sys_root)),
    }


def write_sample(directory: str, sample: Dict[str, Any]) -> str:
    """Write ``sample`` into ``directory`` atomically and return the file's path.

    A reader never sees a half-written file: the bytes go to a fresh temporary
    file in the same directory, are flushed to disk, and only then take the
    final name with one ``os.replace``.
    """
    handle, temporary = tempfile.mkstemp(dir=directory, prefix=".gpu-vendor-")
    try:
        if hasattr(os, "fchmod"):
            os.fchmod(handle, 0o644)
        with os.fdopen(handle, "w", encoding="utf-8") as stream:
            json.dump(sample, stream, separators=(",", ":"), sort_keys=True)
            stream.flush()
            os.fsync(stream.fileno())
        final = os.path.join(directory, SAMPLE_FILE_NAME)
        os.replace(temporary, final)
        return final
    except BaseException:
        # Never leave a stray temporary file behind in the runtime directory.
        try:
            os.unlink(temporary)
        except OSError:
            pass  # absence-ok: the temporary file is already gone
        raise


def run(
    directory: str, *, interval: float = VENDOR_SAMPLE_INTERVAL_SECONDS,
    clock: Callable[[], float] = time.time,
    sleeper: Callable[[float], Any] = time.sleep,
    iterations: Optional[int] = None,
    sampler: Callable[..., Dict[str, Any]] = build_sample,
    log: Callable[[str], Any] = _log_stderr,
) -> int:
    """The loop: sample, write, sleep. ``iterations`` bounds it for a test.

    One failed sample - a full disk, a tool that crashed - does not end the
    loop (review S-21): the emitter would read the file as stale, and the
    sampler would stay dead until systemd's restart. The failure is logged
    once, bounded; the same failure again is not logged again.
    """
    done = 0
    last_error = ""
    while iterations is None or done < iterations:
        try:
            write_sample(directory, sampler(now=clock()))
            last_error = ""
        except Exception as error:  # noqa: BLE001 - see above: the loop outlives one sample
            message = _bounded("GPU sample not written: ", "{}: {}".format(type(error).__name__, error))
            if message != last_error:
                log(message)
            last_error = message
            try:
                # Say "unreadable" rather than leave the emitter to report the
                # sampler missing (review nit): it is running, its read failed.
                write_sample(directory, {"sampled_at": float(clock()), "status": UNREADABLE_OUTPUT,
                                         "values": {}, "limits": {}})
            except Exception:  # noqa: BLE001 - nothing more can be written; the log says why
                pass
        done += 1
        if iterations is None or done < iterations:
            sleeper(interval)
    return 0


def main() -> int:
    """Run the loop in the unit's runtime directory. Takes no input."""
    directory = os.environ.get(RUNTIME_DIRECTORY_ENV, "").split(":")[0]
    if not directory:
        sys.stderr.write("The GPU sampler has no runtime directory to write to.\n")
        return 2
    return run(directory)


if __name__ == "__main__":
    sys.exit(main())
