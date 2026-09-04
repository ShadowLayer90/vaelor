"""Launch primitives for the ROCmFPX GPU model server (GPU AI-Chat).

The GPU analogue of :mod:`vaelor.flm_service`, and deliberately the simpler of
the two. The Z2's ROCmFP4 27B (``julianmb/Qwen-3.8-27B-ROCmFP4-FAST-GGUF``) was
proven by hand at ~37 t/s on the Strix Halo / Radeon 8060S (gfx1151), served by
the **ROCmFPX** llama.cpp fork - *not* stock llama.cpp, whose kernels cannot
read FP4 tensors at all. This module encodes that proven recipe as a fixed
argument vector.

Where the NPU's ``flm-real`` needs root to pin NPU pages (CAP_IPC_LOCK) and is
therefore launched through the privileged hardware bridge, ``llama-server`` on
the GPU is an **unprivileged host process**: a plain :class:`subprocess.Popen`
with no bridge, no capability grant and no tag-allowlist grammar. So the whole
of what crosses into the launch is a validated model path and a loopback port;
everything else in the argv is a module constant read off the measured machine.

Discovery/provisioning (whether the fork binary and the TheRock libraries are
present) belongs to a later phase; this module names the *provisioned locations*
as overridable constants and builds the launch, nothing more.
"""

from __future__ import annotations

import glob
import os
import re
import subprocess
import time
from pathlib import Path
from typing import Any, Callable, Dict, List, Mapping, Optional, Sequence

from .flm_service import _validate_port
from .model_catalog import RUNTIME_Z2_GPU_ROCMFP4
from .runtime_paths import LOG_ROOT


#: The ROCmFPX ``llama-server`` fork binary. A fixed path, not a ``PATH`` lookup:
#: even though this launch is unprivileged, the executable must be the fork that
#: can read FP4 tensors and nothing a mutable environment could redirect. This
#: is where a later provisioning phase places the fork build; kept as a module
#: constant (overridable for tests) rather than derived from a request.
GPU_ENGINE_BINARY = "/var/lib/vaelor/engines/rocmfpx/bin/llama-server"

#: The legacy Lemonade-snap TheRock cache: where an *older* snap kept its gfx1151
#: HIP/ROCm shared objects. Kept as a named constant (and still imported by
#: :mod:`vaelor.gpu_model_choice`) and as the FIRST candidate in
#: :data:`ROCM_RUNTIME_LIB_CANDIDATES`, but no longer hardcoded as the one launch
#: path: the current snap (v11.7.0) ships no TheRock libs and this directory does
#: not exist, so a stale hardcode silently dropped the fork onto the CPU. The
#: runtime dir is now resolved at launch by :func:`resolve_rocm_lib_dir`.
THEROCK_LIB_DIR = (
    "/var/snap/lemonade-server/common/cache/lemonade/bin/therock/gfx1151-7.13.0/lib"
)

#: The soname the fork's HIP runtime pulls first; its presence in a directory is
#: what makes that directory a real ROCm runtime lib dir rather than an empty or
#: unrelated path. Probed by :func:`resolve_rocm_lib_dir`.
ROCM_RUNTIME_PROBE_SONAME = "libamdhip64.so.7"

#: Ordered candidate roots for the gfx1151 ROCm/HIP runtime lib dir, most specific
#: first, and the reason for the order:
#:
#: 1. :data:`THEROCK_LIB_DIR` - the legacy Lemonade-snap TheRock cache. First so a
#:    box still carrying the old snap keeps working unchanged; absent on current
#:    snaps, where it simply falls through.
#: 2. ``/opt/rocm/lib`` - AMD's official gfx1151 ROCm from the apt install, the
#:    canonical location the companion installer provisions. ``ldd libggml-hip.so``
#:    against this dir resolves every HIP/ROCm/BLAS dep on the Z2.
#: 3. ``/opt/rocm/core-*/lib`` - the versioned core dir the same ROCm ships (e.g.
#:    ``core-7.14``); globbed and, when several exist, the highest version wins.
#:
#: A shell ``glob`` pattern per entry: a literal path globs to itself when it
#: exists, so all three are probed uniformly.
ROCM_RUNTIME_LIB_CANDIDATES: Sequence[str] = (
    THEROCK_LIB_DIR,
    "/opt/rocm/lib",
    "/opt/rocm/core-*/lib",
)


def _rocm_version_key(path: str) -> List[int]:
    """The integer components of a path, so ``core-7.14`` sorts above ``core-7.9``.

    A plain lexical sort would rank ``core-7.9`` above ``core-7.14`` (``'9' > '1'``)
    and pick the older runtime; comparing the numbers instead keeps "highest
    version wins" true across the two-vs-one-digit boundary.
    """
    return [int(number) for number in re.findall(r"\d+", path)]


def resolve_rocm_lib_dir(
    candidates: Sequence[str] = ROCM_RUNTIME_LIB_CANDIDATES,
    *,
    probe_soname: str = ROCM_RUNTIME_PROBE_SONAME,
) -> Optional[str]:
    """The first candidate directory that exists AND holds the ROCm runtime.

    Walks :data:`ROCM_RUNTIME_LIB_CANDIDATES` in order and returns the first
    directory that actually contains ``probe_soname`` - so a candidate that is
    absent, or present but empty of the runtime (the stale-snap failure this
    resolver exists to fix), is skipped rather than trusted. Within a single
    globbed candidate the highest version wins (:func:`_rocm_version_key`), but
    candidate ORDER is preserved across entries, so the legacy snap dir still
    beats ``/opt/rocm`` when both are real.

    Returns ``None`` when nothing resolves; the caller turns that into an honest
    failure rather than a CPU fallback that looks healthy. Pure and injectable:
    pass a different ``candidates`` (e.g. a tmp dir with a fake soname) to test it
    without a real GPU or ``/opt/rocm``.
    """
    for candidate in candidates:
        matches = sorted(glob.glob(candidate), key=_rocm_version_key, reverse=True)
        for directory in matches:
            if os.path.isfile(os.path.join(directory, probe_soname)):
                return directory
    return None


#: Loopback only. ``llama-server`` is bound here and nowhere else: AI-Chat is a
#: process on the same machine and the endpoint is never offered to the LAN. A
#: non-loopback host is refused by :func:`gpu_serve_command`.
GPU_HOST = "127.0.0.1"

#: The measured Strix-Halo ROCmFP4 launch parameters, the single source the
#: catalog entry also carries. Used as the fallback when a caller's ``runtime``
#: dict omits a field, so the argv is always the proven recipe even from a bare
#: ``{}``.
GPU_MEASURED_DEFAULTS: Dict[str, Any] = dict(RUNTIME_Z2_GPU_ROCMFP4)


class GpuModelPathError(ValueError):
    """A model path was refused before it could reach the GPU launch.

    Unlike the flm tag there is no allowlist here - the artifact is whatever the
    catalog resolved and downloaded - but the path is still the one caller-side
    value that reaches the command line, so it is checked for the shape a real
    ``.gguf`` on disk has and rejected here rather than discovered as a failed
    launch.
    """


class GpuEngineMissingError(FileNotFoundError):
    """The ROCmFPX fork binary is not present (or not executable) to launch.

    Raised at spawn time, before the log file is opened, so an unprovisioned box
    fails with a sentence naming what to do rather than a raw ``FileNotFoundError``
    from :class:`subprocess.Popen` - and with no log file descriptor left open
    behind the failed launch. A subclass of ``FileNotFoundError`` so a caller
    already catching that keeps working; the message is what changes.
    """


class GpuRuntimeLibsMissingError(FileNotFoundError):
    """No gfx1151 ROCm runtime lib dir could be resolved to launch against.

    The counterpart of :class:`GpuEngineMissingError` for the runtime the fork
    links against: raised at launch, before the log file is opened, when
    :func:`resolve_rocm_lib_dir` finds none of its candidates. This is the honest
    failure the stale hardcode used to hide - without a real ROCm lib dir on
    ``LD_LIBRARY_PATH`` ``libggml-hip.so`` cannot load and ``llama-server``
    silently falls back to the CPU while looking healthy - so the launch stops
    with a sentence naming what to install rather than serving on the CPU. A
    subclass of ``FileNotFoundError`` for the same reason ``GpuEngineMissingError``
    is: a caller already catching that keeps working; the message is what changes.
    """


def _validate_model_path(model_path: Any) -> str:
    """Return an absolute ``.gguf`` path, or raise :class:`GpuModelPathError`.

    A list argv with no shell means nothing here can be split or expanded, so
    this is not an anti-injection gate the way the flm tag grammar is; it is a
    correctness gate. An empty path, a relative one, or a non-``.gguf`` is a
    misconfiguration that would only surface as ``llama-server`` failing to
    open its model, and it is cheaper to refuse it at planning time.
    """
    text = str(model_path or "").strip()
    if not text:
        raise GpuModelPathError("A model path is required to launch the GPU server.")
    # No newline or NUL: defence in depth even behind a list argv, so a path
    # can never carry a second argument or truncate the vector.
    if any(character in text for character in "\n\r\x00"):
        raise GpuModelPathError(
            "The model path carries a control character and is refused."
        )
    if not text.endswith(".gguf"):
        raise GpuModelPathError(
            "The GPU server serves a .gguf artifact; '{}' is not one.".format(text)
        )
    # A POSIX absolute path (a leading ``/``), checked directly rather than with
    # ``os.path.isabs`` - the appliance is Linux and its model paths are always
    # ``/var/lib/vaelor/models/...``, so a leading slash is the right test and it
    # does not depend on the OS the control plane happens to be tested on.
    if not text.startswith("/"):
        raise GpuModelPathError(
            "The model path must be absolute so the launch does not depend on a "
            "working directory; '{}' is relative.".format(text)
        )
    return text


#: Port validation is :func:`vaelor.flm_service._validate_port`, imported rather
#: than re-implemented. The rule is identical - integer, 1024-65535, and the
#: reserved control-plane pair refused at the boundary (F2) - and the three
#: refusal sentences it raises are the ones ``test_duplicate_literals`` insists
#: live in one home. An unprivileged GPU launch has the same reason a root NPU
#: launch does not to bind a control-plane port, so it reuses the same gate.


def _flag(runtime: Mapping[str, Any], key: str) -> Any:
    """A runtime value, falling back to the measured default for its key.

    So a caller passing ``{}`` still gets the proven recipe and a caller
    overriding one field changes only that field. ``None`` is treated as absent
    for the same reason - an explicit null must not blank out a measured flag.
    """
    value = runtime.get(key)
    return GPU_MEASURED_DEFAULTS.get(key) if value is None else value


def gpu_serve_command(
    model_path: str,
    port: int,
    runtime: Optional[Mapping[str, Any]] = None,
    *,
    host: str = GPU_HOST,
    binary: str = GPU_ENGINE_BINARY,
) -> List[str]:
    """The fixed ``llama-server`` argv for the proven ROCmFP4 recipe.

    A list, never a string: :class:`subprocess.Popen` is given ``argv`` and no
    shell, so no element can be split, expanded or chained. The device, layer
    count, KV quantisations and speculative-decode parameters are read from
    ``runtime`` (the catalog entry's measured block) and fall back to
    :data:`GPU_MEASURED_DEFAULTS`; the model path is validated and the port is
    range-checked. The binary and the flag names are constants.

    Encodes exactly the invocation proven by hand on the Z2::

        llama-server -m <gguf> -dev Vulkan0 -ngl 99 -fa on -c 131072
            -ctk q8_0 -ctv turbo4 --spec-type draft-mtp
            --spec-draft-n-max 4 --spec-draft-p-min 0.0
            --host 127.0.0.1 --port <port>
    """
    resolved = dict(runtime or {})
    safe_model = _validate_model_path(model_path)
    safe_port = _validate_port(port)
    if str(host) not in {"127.0.0.1", "::1", "localhost"}:
        raise ValueError(
            "The GPU server is bound to loopback only; '{}' is not a loopback "
            "host.".format(host)
        )
    # ``-fa`` takes ``on``/``off``, not a bool; the measured recipe is ``on``.
    flash = "on" if bool(_flag(resolved, "flash_attn")) else "off"
    command = [
        binary,
        "-m", safe_model,
        "-dev", str(_flag(resolved, "device")),
        "-ngl", str(int(_flag(resolved, "ngl"))),
        "-fa", flash,
        "-c", str(int(_flag(resolved, "context"))),
        "-ctk", str(_flag(resolved, "kv_k")),
        "-ctv", str(_flag(resolved, "kv_v")),
    ]
    # Speculative decode is the MEASURED FP4 recipe's, not a universal default:
    # ``--spec-*`` is emitted only when ``spec_type`` resolves to a NON-EMPTY
    # value. A bare ``{}`` still backfills the measured ``draft-mtp`` (so the
    # convenience recipe is unchanged) and the FP4 runtime carries it, so both
    # emit the flags; a GENERIC runtime sets ``spec_type`` to ``""`` EXPLICITLY,
    # which :func:`_flag` returns verbatim (an empty string is present, not
    # absent, so the default is not backfilled) and this gate reads as "off". Its
    # ``turbo4`` V cache likewise cannot leak: a generic runtime sets ``kv_v``
    # explicitly, so no measured default fills it.
    if str(_flag(resolved, "spec_type") or ""):
        command += [
            "--spec-type", str(_flag(resolved, "spec_type")),
            "--spec-draft-n-max", str(int(_flag(resolved, "spec_draft_n_max"))),
            # A float flag: ``0.0`` is the measured "no probability floor".
            # Rendered from the runtime value so an override reaches it.
            "--spec-draft-p-min", str(float(_flag(resolved, "spec_draft_p_min"))),
        ]
    command += ["--host", str(host), "--port", str(safe_port)]
    return command


def gpu_serve_env(
    *,
    engine_lib_dir: str = os.path.dirname(GPU_ENGINE_BINARY),
    rocm_lib_dir: Optional[str] = None,
    resolver: Callable[[], Optional[str]] = resolve_rocm_lib_dir,
    base_env: Optional[Mapping[str, str]] = None,
) -> Dict[str, str]:
    """The launch environment: the service's own, with the GPU variables set.

    Four variables the fork needs on the Z2, none caller-controlled:

    * ``LD_LIBRARY_PATH`` gets TWO directories **prepended** to any inherited
      value: the fork binary's OWN directory and the gfx1151 ROCm runtime lib dir.
      The binary's directory holds its co-located shared objects
      (``libllama-common``, ``libggml-*``); the prebuilt is not linked with an
      ``$ORIGIN`` RUNPATH, so without its own dir on the path it dies at
      "libllama-common.so.0: cannot open shared object file" before it ever
      reaches the GPU. The runtime dir supplies the HIP/ROCm libs the fork links
      against (``libamdhip64.so.7``, ``libhipblas.so.3``, ``librocblas.so.5`` etc.)
      and is **resolved at launch** by :func:`resolve_rocm_lib_dir` rather than
      hardcoded: pass ``rocm_lib_dir`` to pin it (tests do), else the ``resolver``
      is called. Both are prepended - the engine's own dir first so its build wins
      a name clash - and any inherited value is kept behind them.
    * ``SKIP_ROCM_CHECK=1`` - the fork's start-up ROCm sanity check refuses the
      unusual gfx1151 build; skipping it is what let the server start by hand.
    * ``HSA_OVERRIDE_GFX_VERSION=11.5.1`` pins the ISA the runtime targets to the
      8060S's, so it does not misdetect the adapter.
    * ``GGML_HIP_ENABLE_UNIFIED_MEMORY=1`` lets the model spill across the Strix
      Halo unified memory aperture rather than a fixed VRAM carve-out.

    When no runtime dir resolves (``rocm_lib_dir`` unset and the resolver finds
    none of its candidates) this raises :class:`GpuRuntimeLibsMissingError` rather
    than building an environment that would drop ``llama-server`` onto the CPU -
    the honest failure the stale hardcode used to hide. It is raised here, before
    :meth:`GpuServerProcess.start` reaches the spawn or touches a running server,
    the same "validate before launch" order as the model-path and port gates.

    Everything else is inherited from the service's environment rather than
    stripped, the same reasoning as ``flm_serve_env``: a minimal hand-built
    environment is exactly the kind of plausible mechanism that is refuted the
    moment the real binary is run on the box. The security boundary is the
    validated argv, not this dict.
    """
    resolved = rocm_lib_dir if rocm_lib_dir is not None else resolver()
    if not resolved:
        raise GpuRuntimeLibsMissingError(
            "No gfx1151 ROCm runtime found (no {} under any of: {}); install "
            "AMD's gfx1151 ROCm (e.g. /opt/rocm) before deploying the GPU "
            "model.".format(
                ROCM_RUNTIME_PROBE_SONAME, ", ".join(ROCM_RUNTIME_LIB_CANDIDATES)
            )
        )
    inherited = dict(os.environ if base_env is None else base_env)
    previous = inherited.get("LD_LIBRARY_PATH", "")
    ld_dirs = [engine_lib_dir, resolved]
    if previous:
        ld_dirs.append(previous)
    inherited["LD_LIBRARY_PATH"] = os.pathsep.join(d for d in ld_dirs if d)
    inherited["SKIP_ROCM_CHECK"] = "1"
    inherited["HSA_OVERRIDE_GFX_VERSION"] = "11.5.1"
    inherited["GGML_HIP_ENABLE_UNIFIED_MEMORY"] = "1"
    return inherited


#: Where the GPU server's own stdout and stderr are kept. **Not DEVNULL**
#: (LESSONS pattern 8): a ``llama-server`` that fails to start - the fork
#: rejecting a tensor, the Vulkan adapter absent, TheRock libs unresolved - says
#: so on stderr, and discarding it makes "the GPU server could not start" and
#: "the GPU server is fine" arrive looking identical. The supervisor detects the
#: failure by the health timeout; this is where the *reason* survives.
#:
#: In ``LOG_ROOT`` beside ``flm-real.log``, because the GPU server is launched by
#: the **root hardware bridge** (which has GPU device access and no
#: ``PrivateDevices``/``MemoryDenyWriteExecute`` sandbox), not the locked-down
#: workload executor - so the process that opens this file is root and
#: ``/var/log/vaelor`` is on the bridge unit's ``ReadWritePaths``. The executor
#: could not write here (its unit makes ``/var/log/vaelor`` read-only), which is
#: one of the reasons the launch goes through the bridge rather than a direct
#: executor child.
GPU_LOG_FILE = str(Path(LOG_ROOT) / "gpu-rocmfpx.log")

#: How long to wait for a SIGTERM'd server to exit before SIGKILL. Freeing a
#: large FP4 model off the GPU is not instant; past this it is not stopping on
#: its own and is killed.
STOP_GRACE_SECONDS = 10.0


class GpuServerProcess:
    """A supervised ROCmFPX ``llama-server`` process, launched and stopped.

    The spawn is injectable so the launch argv/env can be verified - and the
    whole lifecycle exercised - without a real GPU. Production passes no
    ``spawn`` and gets :class:`subprocess.Popen` with a **fixed argv and env**,
    no shell, ``stdin`` closed and file descriptors closed. Unlike
    :class:`vaelor.flm_service.FlmProcess` there is no root boundary here: this
    is an ordinary host process.
    """

    def __init__(
        self,
        *,
        spawn: Optional[Callable[..., Any]] = None,
        binary: str = GPU_ENGINE_BINARY,
        rocm_lib_dir: Optional[str] = None,
        resolver: Callable[[], Optional[str]] = resolve_rocm_lib_dir,
    ):
        self._spawn = spawn or self._default_spawn
        self._binary = binary
        # ``rocm_lib_dir`` pins the runtime lib dir (tests do); left ``None``,
        # ``start`` resolves it at launch through ``resolver`` and raises
        # ``GpuRuntimeLibsMissingError`` if none is found - never a CPU fallback.
        self._rocm_lib_dir = rocm_lib_dir
        self._resolver = resolver
        self._process: Any = None
        self._log: Any = None
        self.model_path = ""
        self.port = 0

    def _default_spawn(self, command, env):
        # Refuse an absent or non-executable engine BEFORE opening the log or
        # spawning: `subprocess.Popen` on a missing binary raises a bare
        # `FileNotFoundError` (naming nothing to do about it) and, worse, would do
        # so with the log file descriptor just opened below still dangling. The
        # binary is `command[0]`, the fixed fork path.
        binary = command[0]
        if not (os.path.isfile(binary) and os.access(binary, os.X_OK)):
            raise GpuEngineMissingError(
                "The ROCmFPX GPU engine is not installed at {}; provision it "
                "before deploying the GPU model.".format(binary)
            )
        # stdout and stderr go to a log file, never DEVNULL: a launch that fails
        # explains itself there (LESSONS pattern 8). The handle is kept on the
        # instance so it lives as long as the process writing to it.
        log_path = Path(GPU_LOG_FILE)
        log_path.parent.mkdir(parents=True, exist_ok=True)
        self._log = open(log_path, "a", encoding="utf-8")
        try:
            return subprocess.Popen(
                command,
                env=env,
                stdin=subprocess.DEVNULL,
                stdout=self._log,
                stderr=subprocess.STDOUT,
                close_fds=True,
            )
        except Exception:
            # Any spawn failure (a race removing the binary, a resource limit)
            # must not leak the log descriptor opened a line above.
            try:
                self._log.close()
            finally:
                self._log = None
            raise

    def start(
        self, model_path: str, *, port: int,
        runtime: Optional[Mapping[str, Any]] = None,
    ) -> Dict[str, Any]:
        """Validate, then launch. A running process is stopped first.

        Validation happens before the spawn and before the previous process is
        touched, so a bad model path or port cannot take down a healthy server.
        """
        command = gpu_serve_command(
            model_path, port, runtime, binary=self._binary
        )
        env = gpu_serve_env(
            engine_lib_dir=os.path.dirname(self._binary),
            rocm_lib_dir=self._rocm_lib_dir,
            resolver=self._resolver,
        )
        if self.alive():
            self.stop()
        self._process = self._spawn(command, env)
        self.model_path = _validate_model_path(model_path)
        self.port = _validate_port(port)
        return self.status()

    def alive(self) -> bool:
        if self._process is None:
            return False
        return self._process.poll() is None

    def stop(self, *, grace_seconds: float = STOP_GRACE_SECONDS) -> Dict[str, Any]:
        """SIGTERM, wait, then SIGKILL. A process that is already gone is fine."""
        process = self._process
        if process is None:
            return {"stopped": True, "was_running": False}
        was_running = process.poll() is None
        if was_running:
            try:
                process.terminate()
            except (OSError, ProcessLookupError):
                pass
            deadline = time.monotonic() + max(0.0, float(grace_seconds))
            while time.monotonic() < deadline and process.poll() is None:
                time.sleep(0.1)
            if process.poll() is None:
                try:
                    process.kill()
                except (OSError, ProcessLookupError):
                    pass
        self._process = None
        if self._log is not None:
            try:
                self._log.close()
            except OSError:
                pass
            self._log = None
        return {"stopped": True, "was_running": was_running}

    def status(self) -> Dict[str, Any]:
        return {
            "running": self.alive(),
            "model_path": self.model_path,
            "port": self.port,
            "pid": getattr(self._process, "pid", None) if self._process else None,
        }
