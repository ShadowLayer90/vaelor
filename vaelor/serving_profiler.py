"""On-demand profiling of the GPU serving process (VD-128, Phase E' 6b).

A single bounded, on-demand capture of what the GPU serving process is doing,
run only when an operator asks. This module builds argv, parses tool output and
orchestrates the captures through an injected ``run`` seam, so the whole flow is
exercised in tests with fixtures and never touches a real ``perf``, ``amd-smi``,
``bpftrace``, GPU or vLLM. The privileged run - the serving PID, tool paths, the
bounded root subprocesses and the torch trace file - lives in
:mod:`vaelor.hardware_bridge` behind ``run_serving_profile``.

The four captures: CPU (``perf`` top symbols), GPU snapshot (``amd-smi`` power/
temp/clock and the process GTT share), GPU kernel trace (vLLM's built-in torch
profiler, driven on demand on the vLLM cluster - available only in Mode B), and
eBPF (``bpftrace`` OFF-CPU blocking). Every capture that cannot run states why,
naming the real blocker; a box where none can profile shows the reasons and no
fabricated data.
"""

from __future__ import annotations

import gzip
import json
import os
import re
import shutil
import tempfile
import time
from typing import Any, Callable, Dict, List, Mapping, Optional, Sequence

from .platforms.graphics_software import json_after_banner


#: The tool paths the grounding confirmed on the Z2. Constants, not discovered
#: names, so the argv the bridge runs and the presence the probe checks cannot
#: drift; the bridge tests each path's existence and passes the booleans in.
PERF_BIN = "/usr/bin/perf"
AMD_SMI_BIN = "/usr/bin/amd-smi"
BPFTRACE_BIN = "/usr/bin/bpftrace"
#: ``curl`` drives the vLLM torch-profiler capture through the injected
#: ``run`` seam (no in-process socket), so the flow stays unit-testable.
CURL_BIN = "curl"
#: On-demand torch-profiler dumps: the HOST dir the vLLM container writes its
#: Chrome kernel traces into and the fixed container mount path. docker creates
#: the host source as root and the bridge clears it per capture, so no
#: ``install -d`` crosses the controller argv policy.
SERVING_PROFILES_DIR = "/var/lib/vaelor/serving-profiles"
SERVING_PROFILES_MOUNT = "/vaelor-serving-profiles"

#: Where the kernel exposes the non-root profiling gate. Read by the privileged
#: side and passed in; kept here so the honest-degrade note can name the file.
PERF_EVENT_PARANOID_PATH = "/proc/sys/kernel/perf_event_paranoid"

#: Capture bounds. Profiling perturbs the very box it measures, so the window is
#: a few seconds and hard-capped: a caller cannot ask the box to profile itself
#: for a minute. The default is what the route uses when no seconds are given.
PROFILE_DEFAULT_SECONDS = 3
PROFILE_MIN_SECONDS = 1
PROFILE_MAX_SECONDS = 10

#: Headroom added to the requested window for the subprocess timeout: ``perf
#: record -- sleep N`` runs for N seconds plus the cost of writing and closing
#: ``perf.data``. The bridge applies this as a HARD timeout so a wedged capture
#: can never outlive the window by more than the margin.
PROFILE_RECORD_MARGIN_SECONDS = 15

#: ``perf report`` reads a file that is already written, so it needs only a short
#: fixed bound rather than one scaled to the capture window.
PROFILE_REPORT_TIMEOUT_SECONDS = 20

#: ``amd-smi monitor`` is a one-shot snapshot; a few seconds is plenty and the
#: hard bound stops a hung tool holding a bridge thread.
PROFILE_SNAPSHOT_TIMEOUT_SECONDS = 15

#: The torch-profiler capture's bounds: fast HTTP, one short completion, then
#: a bounded poll of the host dir for the asynchronously-written trace.
PROFILE_HTTP_TIMEOUT_SECONDS = 10
PROFILE_COMPLETION_TIMEOUT_SECONDS = 30
PROFILE_COMPLETION_TOKENS = 16
PROFILE_COMPLETION_PROMPT = "Vaelor GPU kernel profile warmup."
GPU_KERNEL_TRACE_SUFFIX = ".pt.trace.json.gz"
GPU_KERNEL_TRACE_WAIT_SECONDS = 6.0
GPU_KERNEL_POLL_INTERVAL_SECONDS = 0.5
GPU_KERNEL_TOP_ROWS = 12

#: How many top symbols the CPU capture keeps. ``perf report`` can list hundreds;
#: the tab shows where the time actually goes, and the socket frame is bounded,
#: so the long tail is dropped rather than carried.
PERF_TOP_SYMBOLS = 12

#: ``perf report`` rows below this percentage are noise for a "where does the
#: time go" view; passed to ``--percent-limit`` so the tool itself drops them.
PERF_PERCENT_LIMIT = 1.0

#: How many top OFF-CPU rows the eBPF capture keeps, by blocking time descending.
#: The bpftrace map can hold one row per distinct thread ``comm``; the tab shows
#: where the blocking is, so the long tail is dropped rather than carried.
EBPF_TOP_ROWS = 12

#: The bpftrace program the eBPF capture runs, as one inline ``-e`` expression so
#: nothing extra ships to the box. It measures OFF-CPU time — where the serving
#: threads BLOCK, which ``perf``'s on-CPU sampling cannot see — attributed by
#: thread ``comm``: on each context switch it stamps a target thread going
#: off-CPU and, when that thread runs again, adds the microseconds it was blocked.
#: ``finish_task_switch*`` matches the scheduler tail on whatever kernel build the
#: box runs (it is inlined under an ``.isra`` suffix on some). An ``interval`` fires
#: ``exit()`` so the window is self-bounded, and ``END`` clears the in-flight map
#: so only the two result maps print. ``{pid}`` is the serving container's own PID
#: and ``{seconds}`` the clamped window. JSON output (``-f json``) is one object per
#: line, so a parser skips the compile-time warnings bpftrace prints as plain text.
_EBPF_PROGRAM = (
    "kprobe:finish_task_switch* {{ $prev = (struct task_struct *)arg0; "
    "if ($prev->tgid == {pid}) {{ @start[$prev->pid] = nsecs; }} "
    "if (pid == {pid} && @start[tid]) {{ "
    "$dt = (nsecs - @start[tid]) / 1000; "
    "@offcpu_us[comm] = @offcpu_us[comm] + $dt; "
    "@offcpu_us_total = @offcpu_us_total + $dt; "
    "delete(@start[tid]); }} }} "
    "interval:s:{seconds} {{ exit(); }} "
    "END {{ clear(@start); }}"
)

# --- Capture identities -----------------------------------------------------
#: The four capture kinds, in the order the tab renders them. CPU and GPU can
#: run; the other two are the honest-degrade rows that name why deep tracing is
#: not available here. Kept as constants so the module, the tests and the
#: frontend agree on one spelling per capture.
CAPTURE_CPU = "cpu"
CAPTURE_GPU = "gpu"
CAPTURE_GPU_KERNEL = "gpu_kernel"
CAPTURE_EBPF = "ebpf"

_CAPTURE_LABELS = {
    CAPTURE_CPU: "CPU profile (perf)",
    CAPTURE_GPU: "GPU snapshot (amd-smi)",
    CAPTURE_GPU_KERNEL: "GPU kernel trace (torch profiler)",
    CAPTURE_EBPF: "eBPF off-CPU trace (bpftrace)",
}

# --- Honest-degrade reasons -------------------------------------------------
#: One home per reason, so the probe and the tests cannot drift on the wording,
#: and every reason names the ACTUAL blocker rather than a generic "unavailable".
NO_SERVING_PROCESS = (
    "No serving process is running, so there is nothing to profile."
)
PERF_MISSING = (
    "perf is not installed at {}, so a CPU profile cannot be captured."
).format(PERF_BIN)
PERF_NEEDS_ROOT = (
    "perf needs root to profile another process — perf_event_paranoid is {} "
    "here, which blocks a non-root profiler; the capture runs through the root "
    "hardware bridge, which bypasses the gate."
)
AMD_SMI_MISSING = (
    "amd-smi is not installed at {}, so the GPU snapshot cannot be captured."
).format(AMD_SMI_BIN)
GPU_METRIC_NOT_READ = (
    "amd-smi's metric read did not answer, so the graphics engine's own power "
    "and temperature are not in this snapshot."
)
GPU_KERNEL_NEEDS_VLLM = (
    "GPU kernel tracing runs against the vLLM cluster engine's built-in torch "
    "profiler; the single-node llama.cpp engine (Mode A) exposes no such "
    "profiler, so start GPU clustering to capture kernels here."
)
GPU_KERNEL_NO_ENDPOINT = (
    "The local vLLM replica did not answer on its loopback serving port, so its "
    "torch profiler could not be driven; nothing was fabricated."
)
GPU_KERNEL_NO_TRACE = (
    "The vLLM torch profiler was started and stopped, but no kernel trace file "
    "appeared in the profiles directory within the wait window."
)
GPU_KERNEL_EMPTY_NOTE = (
    "Captured, but no GPU kernels ran during the profiling window."
)
GPU_KERNEL_FAILED = (
    "The GPU kernel trace could not be captured from the live vLLM server."
)
EBPF_MISSING = (
    "bpftrace is not installed at {}, so an eBPF off-CPU trace cannot be captured."
).format(BPFTRACE_BIN)
EBPF_NEEDS_ROOT = (
    "eBPF (bpftrace) needs root to attach to the serving process; this run is not "
    "root, so the off-CPU trace cannot be captured — the capture normally runs "
    "through the root hardware bridge, which provides it."
)
EBPF_EMPTY_NOTE = (
    "Captured, but the serving process did not block off-CPU during the window."
)


def clamp_seconds(seconds: Any) -> int:
    """Coerce a caller's requested window into the bounded capture range.

    A non-numeric or absent value falls back to the default; anything outside
    ``[PROFILE_MIN_SECONDS, PROFILE_MAX_SECONDS]`` is clamped. Bounding here (and
    again where the bridge sets the subprocess timeout) is what stops a caller
    asking the box to profile itself into the ground.
    """
    try:
        value = int(seconds)
    except (TypeError, ValueError):
        return PROFILE_DEFAULT_SECONDS
    return max(PROFILE_MIN_SECONDS, min(PROFILE_MAX_SECONDS, value))


def probe_capabilities(
    *,
    serving_pid: Optional[int],
    as_root: bool,
    perf_event_paranoid: Optional[int],
    perf_present: bool,
    amd_smi_present: bool,
    vllm_serving: bool,
    bpftrace_present: bool,
) -> Dict[str, Any]:
    """Decide which captures can run and, for each that cannot, the exact reason."""
    paranoid_text = "unknown" if perf_event_paranoid is None else str(perf_event_paranoid)
    have_pid = isinstance(serving_pid, int) and serving_pid > 0

    captures: List[Dict[str, Any]] = []

    # CPU — perf. Available only with a PID to attach to, the tool present, AND
    # root: perf record on another process needs it, and the paranoid gate only
    # blocks non-root (the root bridge bypasses it). The reasons are checked
    # most-specific first so the message names the nearest blocker.
    captures.append(_capture(
        CAPTURE_CPU,
        available=bool(have_pid and perf_present and as_root),
        reason=(
            "" if (have_pid and perf_present and as_root)
            else NO_SERVING_PROCESS if not have_pid
            else PERF_MISSING if not perf_present
            else PERF_NEEDS_ROOT.format(paranoid_text)
        ),
    ))

    # GPU snapshot — amd-smi. Always works where the tool is present; the
    # per-process slice needs a PID, so a missing PID drops it to the reason
    # rather than pretending a process reading exists.
    captures.append(_capture(
        CAPTURE_GPU,
        available=bool(amd_smi_present and have_pid),
        reason=(
            "" if (amd_smi_present and have_pid)
            else AMD_SMI_MISSING if not amd_smi_present
            else NO_SERVING_PROCESS
        ),
    ))

    # GPU kernel trace — vLLM's built-in torch profiler. AVAILABLE whenever a
    # vLLM cluster is serving (Mode B): the profiler is in-process in the live
    # server, driven on demand, so no attach injection and no restart.
    captures.append(_capture(
        CAPTURE_GPU_KERNEL,
        available=bool(have_pid and vllm_serving),
        reason=(
            "" if (have_pid and vllm_serving)
            else NO_SERVING_PROCESS if not have_pid
            else GPU_KERNEL_NEEDS_VLLM
        ),
    ))

    # eBPF — bpftrace. RUNS as root: root attaches to the serving PID regardless of
    # perf_event_paranoid (which gates only non-root), so a bounded OFF-CPU trace of
    # the serving threads is captured. Not available only for a real blocker — no
    # serving PID, the tool missing, or the run not being root — checked
    # most-specific first so the message names the nearest one.
    captures.append(_capture(
        CAPTURE_EBPF,
        available=bool(have_pid and bpftrace_present and as_root),
        reason=(
            "" if (have_pid and bpftrace_present and as_root)
            else NO_SERVING_PROCESS if not have_pid
            else EBPF_MISSING if not bpftrace_present
            else EBPF_NEEDS_ROOT
        ),
    ))

    return {
        "serving_pid": serving_pid if have_pid else None,
        "as_root": bool(as_root),
        "perf_event_paranoid": perf_event_paranoid,
        "captures": captures,
    }


def _capture(capture_id: str, *, available: bool, reason: str) -> Dict[str, Any]:
    """One capture verdict: its id, human label, availability and honest reason."""
    return {
        "id": capture_id,
        "label": _CAPTURE_LABELS[capture_id],
        "available": bool(available),
        "reason": reason,
    }


# --- CPU capture: perf ------------------------------------------------------
def perf_record_command(
    pid: int, seconds: int, data_path: str, *, perf_path: str = PERF_BIN
) -> List[str]:
    """A bounded ``perf record`` of one PID for ``seconds``, into ``data_path``.

    ``perf record -o <data> -p <pid> -- sleep <N>`` samples only the serving
    process's threads for exactly the window (``sleep`` is the timed anchor, not a
    workload) and writes one ``perf.data`` the report step reads. A list argv, so
    no element is word-split or glob-expanded by a shell. The PID is the serving
    container's own, resolved by the bridge from ``docker inspect`` — never a
    caller-supplied one.
    """
    return [perf_path, "record", "-o", str(data_path),
            "-p", str(int(pid)), "--", "sleep", str(int(seconds))]


def perf_report_command(data_path: str, *, perf_path: str = PERF_BIN) -> List[str]:
    """``perf report`` over a recorded ``perf.data``, as plain text to parse.

    ``--stdio`` prints the text table (no pager), ``--percent-limit`` drops the
    long tail server-side, and ``--no-children`` keeps the overhead column the
    self cost of each symbol rather than the inclusive tree, which is the "where
    is the time actually spent" view the tab wants.
    """
    return [perf_path, "report", "-i", str(data_path), "--stdio",
            "--no-children", "--percent-limit", str(PERF_PERCENT_LIMIT)]


#: A ``perf report`` data row starts with an overhead percentage; the symbol sits
#: after the ``[.]`` (user) / ``[k]`` (kernel) / ``[g]`` (guest) marker. Two
#: patterns rather than a rigid column split because the shared-object and symbol
#: fields carry spaces (C++ names) and vary in width.
_PERF_ROW = re.compile(r"^\s*(\d+(?:\.\d+)?)%\s+(.+)$")
_PERF_SYMBOL = re.compile(r"\[[.kgu?@]\]\s+(.+?)\s*$")
_PERF_OBJECT = re.compile(r"(\S+)\s+\[[.kgu?@]\]")


def parse_perf_report(stdout: Any, *, limit: int = PERF_TOP_SYMBOLS) -> List[Dict[str, Any]]:
    """The top symbols and their self-overhead from ``perf report --stdio``.

    Returns ``[{"symbol", "overhead_percent", "shared_object"}]`` for the hottest
    rows, capped at ``limit``. Comment lines (``#``) and anything that is not a
    percentage row are skipped, so a header, a blank line or a warning never
    becomes a fabricated symbol. Malformed or empty input yields an empty list —
    it never raises — so a capture that produced nothing degrades to "no symbols"
    rather than a 500.
    """
    text = stdout if isinstance(stdout, str) else ""
    rows: List[Dict[str, Any]] = []
    for line in text.splitlines():
        if line.startswith("#"):
            continue
        row = _PERF_ROW.match(line)
        if row is None:
            continue
        tail = row.group(2)
        symbol_match = _PERF_SYMBOL.search(tail)
        if symbol_match is None:
            # A percentage row with no symbol marker is a summary or a truncated
            # line, not a symbol; recording it would invent a name for time we
            # cannot attribute.
            continue
        object_match = _PERF_OBJECT.search(tail)
        rows.append({
            "symbol": symbol_match.group(1).strip(),
            "overhead_percent": round(float(row.group(1)), 2),
            "shared_object": object_match.group(1) if object_match else "",
        })
        if len(rows) >= limit:
            break
    return rows


# --- GPU capture: amd-smi ---------------------------------------------------
# The snapshot's parsing lives in `serving_profiler_gpu` (split out at the line
# ceiling, VD-147 S1b); these names are re-exported for the bridge and the tests.
from .serving_profiler_gpu import (  # noqa: E402,F401 - re-exported
    amd_smi_metric_command, discovered_integrated_gpu, parse_amd_smi,
)
from .serving_profiler_gpu import amd_smi_command as _amd_smi_command  # noqa: E402


def amd_smi_command(*, amd_smi_path: str = AMD_SMI_BIN) -> List[str]:
    """The ``amd-smi monitor`` argv (see `serving_profiler_gpu.amd_smi_command`)."""
    return _amd_smi_command(amd_smi_path=amd_smi_path)


# --- eBPF capture: bpftrace -------------------------------------------------
def ebpf_command(
    pid: int, seconds: int, *, bpftrace_path: str = BPFTRACE_BIN
) -> List[str]:
    """A bounded OFF-CPU ``bpftrace`` trace of one PID for ``seconds``.

    The program (``_EBPF_PROGRAM``) is embedded inline as a single ``-e``
    expression — the serving container's own PID and the clamped window are the
    only values it carries — so nothing extra ships to the box. ``-f json`` makes
    the output one object per line for :func:`parse_ebpf`. A list argv, so no
    element is word-split by a shell; ``bpftrace`` is run as root by the bridge,
    which is what lets it attach to the serving PID.
    """
    program = _EBPF_PROGRAM.format(pid=int(pid), seconds=int(seconds))
    return [bpftrace_path, "-f", "json", "-e", program]


def parse_ebpf(
    stdout: Any, *, seconds: int = 0, limit: int = EBPF_TOP_ROWS
) -> Dict[str, Any]:
    """The OFF-CPU blocking time by thread from ``bpftrace -f json`` output.

    Returns ``{"kind": "offcpu", "window_seconds", "total_us", "rows": [{"label",
    "value_us"} ...]}`` for the threads that blocked most, capped at ``limit``.
    Only ``{"type": "map"}`` JSON lines are read; the compile-time warnings
    bpftrace prints as plain text, and any other line, are skipped, so a warning
    never becomes a fabricated row. Malformed, empty or all-warning input yields
    an empty-but-collected result with a truthful note — it never raises — so an
    idle window degrades to "no blocking observed" rather than a 500.
    """
    text = stdout if isinstance(stdout, str) else ""
    by_comm: Dict[str, int] = {}
    total: Optional[int] = None
    for line in text.splitlines():
        stripped = line.strip()
        if not stripped.startswith("{"):
            continue
        try:
            obj = json.loads(stripped)
        except (ValueError, TypeError):
            continue
        if not isinstance(obj, dict) or obj.get("type") != "map":
            continue
        data = obj.get("data")
        if not isinstance(data, dict):
            continue
        for key, value in data.items():
            if key == "@offcpu_us" and isinstance(value, dict):
                for comm, micros in value.items():
                    number = _as_int(micros)
                    if number is not None:
                        by_comm[str(comm)] = number
            elif key == "@offcpu_us_total":
                number = _as_int(value)
                if number is not None:
                    total = number
    rows = sorted(
        ({"label": comm, "value_us": micros} for comm, micros in by_comm.items()),
        key=lambda row: row["value_us"],
        reverse=True,
    )[:limit]
    if total is None:
        total = sum(int(row["value_us"]) for row in rows)
    result: Dict[str, Any] = {
        "kind": "offcpu",
        "window_seconds": int(seconds),
        "total_us": int(total),
        "rows": rows,
    }
    if not rows:
        result["note"] = EBPF_EMPTY_NOTE
    return result


def _as_int(raw: Any) -> Optional[int]:
    """One integer from a bpftrace JSON value, or ``None`` for a non-number.

    bpftrace's JSON carries map values as plain numbers, but a garbled or string
    value degrades to ``None`` (dropped) rather than a fabricated zero, mirroring
    the absent-not-zero rule the amd-smi parser follows.
    """
    if isinstance(raw, bool):
        return None
    if isinstance(raw, (int, float)):
        return int(raw)
    if isinstance(raw, str):
        try:
            return int(float(raw.strip()))
        except ValueError:
            return None
    return None


# --- Orchestration ----------------------------------------------------------
class _Completed:
    """The minimal shape :func:`capture_profile` reads from an injected run.

    Mirrors ``subprocess.CompletedProcess`` enough for the module to stay free of
    subprocess: a return code and captured text. The bridge's real runner returns
    the genuine article; a test passes a fake, so the whole capture flow is
    exercised without a real ``perf`` or GPU.
    """

    def __init__(self, returncode: int = 0, stdout: str = "", stderr: str = ""):
        self.returncode = returncode
        self.stdout = stdout
        self.stderr = stderr


def capture_profile(
    *,
    probe: Mapping[str, Any],
    run: Callable[..., Any],
    seconds: int,
    perf_path: str = PERF_BIN,
    amd_smi_path: str = AMD_SMI_BIN,
    bpftrace_path: str = BPFTRACE_BIN,
    curl_path: str = CURL_BIN,
    torch_ports: Sequence[int] = (),
    profiles_dir: str = SERVING_PROFILES_DIR,
    temp_dir_factory: Callable[[], str] = None,  # type: ignore[assignment]
    gpu_integrated: Callable[[], Optional[bool]] = discovered_integrated_gpu,
    clock: Callable[[], float] = time.time,
    sleep: Callable[[float], None] = time.sleep,
) -> Dict[str, Any]:
    """Run whichever captures the probe allows and assemble one honest result."""
    serving_pid = probe.get("serving_pid")
    cpu_result = None
    gpu_result = None
    ebpf_result = None
    gpu_kernel_result = None
    for capture in probe.get("captures", []):
        if not capture.get("available"):
            continue
        if capture["id"] == CAPTURE_CPU and isinstance(serving_pid, int):
            cpu_result = _run_cpu_capture(
                run, serving_pid, seconds, perf_path, temp_dir_factory
            )
        elif capture["id"] == CAPTURE_GPU:
            gpu_result = _run_gpu_capture(run, amd_smi_path, serving_pid, _integrated_or_unknown(gpu_integrated))
        elif capture["id"] == CAPTURE_GPU_KERNEL:
            gpu_kernel_result = _run_gpu_kernel_capture(
                run, seconds, torch_ports, profiles_dir, curl_path, sleep, clock
            )
        elif capture["id"] == CAPTURE_EBPF and isinstance(serving_pid, int):
            ebpf_result = _run_ebpf_capture(run, serving_pid, seconds, bpftrace_path)
    return build_profile(
        probe, cpu=cpu_result, gpu=gpu_result, ebpf=ebpf_result,
        gpu_kernel=gpu_kernel_result,
        seconds=seconds, generated_at=clock(),
    )


def _run_cpu_capture(
    run: Callable[..., Any],
    pid: int,
    seconds: int,
    perf_path: str,
    temp_dir_factory: Optional[Callable[[], str]],
) -> Dict[str, Any]:
    """Record the serving PID with ``perf`` for ``seconds`` and parse the report."""
    make_dir = temp_dir_factory or (lambda: tempfile.mkdtemp(prefix="vaelor-perf-"))
    directory = make_dir()
    data_path = os.path.join(directory, "perf.data")
    try:
        record = run(
            perf_record_command(pid, seconds, data_path, perf_path=perf_path),
            timeout=int(seconds) + PROFILE_RECORD_MARGIN_SECONDS,
        )
        if getattr(record, "returncode", 1) != 0:
            return {
                "symbols": [],
                "note": _tool_stderr(record) or "perf record captured no samples.",
            }
        report = run(
            perf_report_command(data_path, perf_path=perf_path),
            timeout=PROFILE_REPORT_TIMEOUT_SECONDS,
        )
        symbols = parse_perf_report(getattr(report, "stdout", ""))
        result: Dict[str, Any] = {"symbols": symbols}
        if not symbols:
            result["note"] = _tool_stderr(report) or "perf recorded no attributable symbols."
        return result
    finally:
        # Always remove the private capture dir: perf.data (and a possible
        # perf.data.old) are transient and must not accumulate on the appliance.
        shutil.rmtree(directory, ignore_errors=True)


def _integrated_or_unknown(discover: Callable[[], Optional[bool]]) -> Optional[bool]:
    """Whether the GPU is an AMD integrated part; ``None`` ("cannot tell") if asking failed.

    A discovery that raises must not fail the whole profile (review S-28): the
    snapshot then marks the monitor's power as unidentified, as it does when
    no GPU is discovered at all.
    """
    try:
        return discover()
    except Exception:  # noqa: BLE001 - see above: "cannot tell" is an answer
        return None


def _run_gpu_capture(
    run: Callable[..., Any], amd_smi_path: str, serving_pid: Optional[int],
    integrated: Optional[bool] = None,
) -> Dict[str, Any]:
    """Take one ``amd-smi`` snapshot and parse it into structured GPU fields.

    Two bounded reads: ``monitor`` (GFX clock and use, the serving process's
    share) and ``metric`` (the graphics engine's own power and temperature,
    which ``monitor`` does not carry on an integrated part). A non-zero exit or
    unparseable ``monitor`` output yields an empty snapshot with the tool's
    stderr rather than a fabricated reading; a ``metric`` read that fails
    leaves those two figures out and says so.
    """
    result = run(
        amd_smi_command(amd_smi_path=amd_smi_path),
        timeout=PROFILE_SNAPSHOT_TIMEOUT_SECONDS,
    )
    if getattr(result, "returncode", 1) != 0:
        return {"snapshot": {}, "note": _tool_stderr(result) or "amd-smi returned no snapshot."}
    metric = run(
        amd_smi_metric_command(amd_smi_path=amd_smi_path),
        timeout=PROFILE_SNAPSHOT_TIMEOUT_SECONDS,
    )
    # A read that exited 0 but printed no document is not a read (review S-23):
    # the note must say why the two figures are missing.
    metric_read = (
        getattr(metric, "returncode", 1) == 0
        and json_after_banner(getattr(metric, "stdout", "") or "") is not None
    )
    snapshot = parse_amd_smi(
        getattr(result, "stdout", ""), serving_pid,
        metric_stdout=getattr(metric, "stdout", "") if metric_read else None,
        integrated=integrated,
    )
    payload: Dict[str, Any] = {"snapshot": snapshot}
    if not snapshot:
        payload["note"] = _tool_stderr(result) or "amd-smi returned no readable fields."
    elif not metric_read:
        payload["note"] = GPU_METRIC_NOT_READ
    return payload


def _run_ebpf_capture(
    run: Callable[..., Any], pid: int, seconds: int, bpftrace_path: str
) -> Dict[str, Any]:
    """Run one bounded OFF-CPU ``bpftrace`` trace of the serving PID and parse it.

    The subprocess timeout is the window plus the same fixed margin the CPU
    record uses, applied by ``run`` as a HARD bound: bpftrace spends a second or
    two compiling and attaching on top of the sample window, and the margin
    covers that without letting a wedged trace outlive the capture. A non-zero
    exit yields an empty-but-collected result carrying the tool's own stderr,
    never a fabricated row.
    """
    result = run(
        ebpf_command(pid, seconds, bpftrace_path=bpftrace_path),
        timeout=int(seconds) + PROFILE_RECORD_MARGIN_SECONDS,
    )
    if getattr(result, "returncode", 1) != 0:
        parsed = parse_ebpf("", seconds=seconds)
        parsed["note"] = _tool_stderr(result) or "bpftrace captured no off-CPU trace."
        return parsed
    return parse_ebpf(getattr(result, "stdout", ""), seconds=seconds)


def _tool_stderr(result: Any) -> str:
    """A short, safe tail of a tool's stderr for an honest 'why empty' note."""
    return str(getattr(result, "stderr", "") or "").strip()[:200]


def build_profile(
    probe: Mapping[str, Any],
    *,
    cpu: Optional[Mapping[str, Any]],
    gpu: Optional[Mapping[str, Any]],
    ebpf: Optional[Mapping[str, Any]] = None,
    gpu_kernel: Optional[Mapping[str, Any]] = None,
    seconds: int,
    generated_at: float,
) -> Dict[str, Any]:
    """Assemble the probe verdicts and whatever captures ran into one result."""
    captures: List[Dict[str, Any]] = []
    for capture in probe.get("captures", []):
        entry = dict(capture)
        if capture.get("available"):
            if capture["id"] == CAPTURE_CPU and cpu is not None:
                entry["result"] = dict(cpu)
            elif capture["id"] == CAPTURE_GPU and gpu is not None:
                entry["result"] = dict(gpu)
            elif capture["id"] == CAPTURE_EBPF and ebpf is not None:
                entry["result"] = dict(ebpf)
            elif capture["id"] == CAPTURE_GPU_KERNEL and gpu_kernel is not None:
                entry["result"] = dict(gpu_kernel)
        captures.append(entry)
    return {
        "seconds": int(seconds),
        "serving_pid": probe.get("serving_pid"),
        "as_root": bool(probe.get("as_root")),
        "perf_event_paranoid": probe.get("perf_event_paranoid"),
        "generated_at": generated_at,
        "captures": captures,
    }


# --- GPU kernel capture: vLLM torch profiler --------------------------------
#: Plain-English kernel categories ``(id, label, description, patterns)``, ordered so
#: the FIRST case-insensitive substring match wins - one source of truth for the spec.
_KERNEL_CATEGORIES = (
    ("attention", "Attention", "How each token attends to the others - the model's context mixing.", ("paged_attention", "flash_attn", "flashattention", "_fwd_kernel", "attention", "_attn", "scaled_dot")),
    ("kv_cache", "KV cache", "Storing and fetching past tokens so they aren't recomputed, plus the paged-KV bookkeeping.", ("reshape_and_cache", "copy_blocks", "kv_cache", "concat_and_cache", "cache_kernel", "slot_mapping", "_apply_write", "_post_update", "_prepare_pos", "seq_lens", "block_table")),
    ("norm", "Normalization", "Keeping activations stable between layers (RMS / layer norm).", ("rms_norm", "rmsnorm", "_add_rms", "layernorm", "layer_norm")),
    ("activation", "Activation & elementwise", "Activation functions, rotary position encoding, and fused elementwise math.", ("silu", "gelu", "swiglu", "rotary", "rope", "pos_encoding", "act_and_mul", "_fused_mul", "_fused_silu", "triton_poi", "triton_red", "elementwise")),
    ("sampling", "Sampling", "Turning the model's scores into the next token (temperature, top-k/top-p, softmax).", ("rocprim", "topk", "top_k", "top_p", "argmax", "sample", "temperature", "cumsum", "softmax", "logits", "sort", "scan")),
    ("matmul", "Matrix multiply", "The model's dense layers - attention projections and the MLP. The core math run for every token, so this dominating is normal and healthy.", ("wvsplitk", "splitk", "cijk", "gemm", "matmul", "tensile", "hgemm", "mfma", "_mm_")),
    ("memory", "Memory & data movement", "Copying, filling and reshaping tensors on the GPU.", ("memcpy", "memset", "fill", "copy", "cat_", "index", "reshape", "transpose", "gather", "scatter", "arange", "trampoline")),
    ("other", "Other", "Kernels outside the common categories.", ()),
)


def parse_torch_kernel_trace(
    trace: Any, *, seconds: int = 0, limit: int = GPU_KERNEL_TOP_ROWS
) -> Dict[str, Any]:
    """Top GPU kernels and their device time from a torch Chrome trace.

    ``trace`` is the ``*.pt.trace.json.gz`` the vLLM torch profiler writes on
    ``/stop_profile`` (gzipped bytes, plain JSON bytes or a JSON string). The
    ``traceEvents`` with ``cat == "kernel"`` carry a long ``name`` (the
    ``Cijk_..._ISA1151_...`` kernels) and a microsecond DEVICE ``dur``; this sums
    ``dur`` by name, counts calls, and returns the hottest ``limit`` rows. Bad
    gzip, unparseable JSON, no kernel events or a wrong shape all degrade to an
    empty ``rows`` with a truthful note and NEVER raise.
    """
    totals: Dict[str, int] = {}
    calls: Dict[str, int] = {}
    cat_us: Dict[str, int] = {}
    total = 0
    # A profiling run writes one Chrome trace PER process: the worker
    # ``rank*`` trace carries the GPU kernels, the API-server ``async_llm``
    # trace carries none. Aggregate ACROSS all of them so the kernels are
    # never missed by picking the wrong (newest) single file.
    traces = trace if isinstance(trace, (list, tuple)) else [trace]
    for one in traces:
      for event in _kernel_events(one):
        if not isinstance(event, dict) or event.get("cat") != "kernel":
            continue
        name = event.get("name")
        micros = _as_int(event.get("dur"))
        if not isinstance(name, str) or not name or micros is None:
            continue
        totals[name] = totals.get(name, 0) + micros
        calls[name] = calls.get(name, 0) + 1
        total += micros
        low = name.lower()
        cid = next((c[0] for c in _KERNEL_CATEGORIES if any(p in low for p in c[3])), "other")
        cat_us[cid] = cat_us.get(cid, 0) + micros
    rows = sorted(
        ({"name": n, "total_us": us, "calls": calls[n]} for n, us in totals.items()),
        key=lambda row: row["total_us"], reverse=True,
    )[:limit]
    cats = sorted(
        ({"id": cid, "label": label, "description": desc, "total_us": int(cat_us[cid]),
          "pct": round(cat_us[cid] / total * 100, 1) if total > 0 else 0.0}
         for cid, label, desc, _p in _KERNEL_CATEGORIES if cat_us.get(cid, 0) > 0),
        key=lambda entry: entry["total_us"], reverse=True)
    result: Dict[str, Any] = {
        "kind": "gpu_kernels", "window_seconds": int(seconds),
        "total_us": int(total), "rows": rows, "categories": cats,
    }
    if not rows:
        result["note"] = GPU_KERNEL_EMPTY_NOTE
    return result


def _kernel_events(trace: Any) -> List[Any]:
    """The ``traceEvents`` list from a torch Chrome trace, tolerant of the form."""
    if isinstance(trace, (bytes, bytearray)):
        raw = bytes(trace)
        try:
            raw = gzip.decompress(raw)
        except (OSError, EOFError, ValueError):
            pass  # not gzipped, or already-decompressed bytes
        trace = raw.decode("utf-8", errors="replace")
    if not isinstance(trace, str) or not trace.strip():
        return []
    try:
        data = json.loads(trace)
    except (ValueError, TypeError):
        return []
    events = data.get("traceEvents") if isinstance(data, dict) else None
    return events if isinstance(events, list) else []


def _run_gpu_kernel_capture(
    run: Callable[..., Any], seconds: int, ports: Sequence[int], profiles_dir: str,
    curl_path: str, sleep: Callable[[float], None], clock: Callable[[], float],
    trace_reader: Optional[Callable[[], Optional[bytes]]] = None,
) -> Dict[str, Any]:
    """Drive the live vLLM's torch profiler and parse its kernel trace.

    Best-effort and bounded: discover the local replica via ``/v1/models`` on
    each candidate loopback port (the first that answers gives the served model),
    clear the host dir, POST ``/start_profile``, drive ONE short completion, POST
    ``/stop_profile``, poll for the newest trace and parse it. Any failure is an
    honest note on an empty result and NEVER a raise, so the deepest capture
    cannot crash a profile. The profiler perturbs latency during the window.
    """
    empty = {"kind": "gpu_kernels", "window_seconds": int(seconds), "total_us": 0, "rows": []}
    try:
        endpoint = _resolve_vllm_endpoint(run, ports, curl_path)
        if endpoint is None:
            return dict(empty, note=GPU_KERNEL_NO_ENDPOINT)
        base, model = endpoint
        _prepare_profiles_dir(profiles_dir)
        started = run(_curl_post(base + "/start_profile", curl_path), timeout=PROFILE_HTTP_TIMEOUT_SECONDS)
        if getattr(started, "returncode", 1) != 0:
            return dict(empty, note=_tool_stderr(started) or GPU_KERNEL_FAILED)
        run(_curl_completion(base, model, curl_path), timeout=PROFILE_COMPLETION_TIMEOUT_SECONDS)
        run(_curl_post(base + "/stop_profile", curl_path), timeout=PROFILE_HTTP_TIMEOUT_SECONDS)
        reader = trace_reader or (lambda: _all_trace_bytes(profiles_dir))
        deadline = clock() + GPU_KERNEL_TRACE_WAIT_SECONDS
        raw = reader()
        while not raw and clock() < deadline:
            sleep(GPU_KERNEL_POLL_INTERVAL_SECONDS)
            raw = reader()
        if not raw:
            return dict(empty, note=GPU_KERNEL_NO_TRACE)
        return parse_torch_kernel_trace(raw, seconds=seconds)
    except Exception:  # noqa: BLE001 - the deepest capture must never crash a profile
        return dict(empty, note=GPU_KERNEL_FAILED)


def _resolve_vllm_endpoint(run: Callable[..., Any], ports: Sequence[int], curl_path: str):
    """``(base_url, model)`` for the first candidate port whose vLLM answers, else None."""
    for port in ports:
        base = "http://127.0.0.1:{}".format(int(port))
        answer = run(_curl_get(base + "/v1/models", curl_path), timeout=PROFILE_HTTP_TIMEOUT_SECONDS)
        if getattr(answer, "returncode", 1) != 0:
            continue
        model = _first_model_id(getattr(answer, "stdout", ""))
        if model:
            return base, model
    return None


def _first_model_id(stdout: Any) -> str:
    """The first model id from a vLLM ``/v1/models`` body, or ``""``."""
    try:
        data = json.loads(stdout) if isinstance(stdout, str) and stdout.strip() else None
    except (ValueError, TypeError):
        return ""
    items = data.get("data") if isinstance(data, dict) else None
    if isinstance(items, list) and items and isinstance(items[0], dict):
        model = items[0].get("id")
        return str(model) if model else ""
    return ""


def _curl_get(url: str, curl_path: str) -> List[str]:
    return [curl_path, "-sS", url]


def _curl_post(url: str, curl_path: str) -> List[str]:
    return [curl_path, "-sS", "-X", "POST", url]


def _curl_completion(base: str, model: str, curl_path: str) -> List[str]:
    """A curl that drives ONE short completion so GPU kernels run in the window."""
    body = json.dumps({
        "model": model, "prompt": PROFILE_COMPLETION_PROMPT,
        "max_tokens": PROFILE_COMPLETION_TOKENS, "temperature": 0,
    })
    return [curl_path, "-sS", "-X", "POST", base + "/v1/completions",
            "-H", "Content-Type: application/json", "-d", body]


def _prepare_profiles_dir(profiles_dir: str) -> None:
    """Ensure the host profiles dir exists and clear stale traces (root, in-process)."""
    try:
        os.makedirs(profiles_dir, exist_ok=True)
        for name in os.listdir(profiles_dir):
            if name.endswith(GPU_KERNEL_TRACE_SUFFIX):
                try:
                    os.remove(os.path.join(profiles_dir, name))
                except OSError:
                    pass
    except OSError:
        pass


def _all_trace_bytes(profiles_dir: str) -> List[bytes]:
    """EVERY torch Chrome trace in the dir as bytes. A run writes one per
    process (worker ``rank*`` has the kernels, API-server ``async_llm`` has
    none), so all are returned and the parser aggregates - robust to which
    file is newest. Empty list when the dir is missing or has no trace."""
    out: List[bytes] = []
    try:
        names = sorted(os.listdir(profiles_dir))
    except OSError:
        return out
    for name in names:
        if not name.endswith(GPU_KERNEL_TRACE_SUFFIX):
            continue
        try:
            with open(os.path.join(profiles_dir, name), "rb") as handle:
                out.append(handle.read())
        except OSError:
            pass
    return out
