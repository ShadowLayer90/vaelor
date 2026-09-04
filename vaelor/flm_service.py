"""Root-side launch primitives for the ``flm-real`` NPU model server (VD-001).

This is the boundary #178's HIGH TOCTOU reached, and everything here is written
for that: the only inputs that cross into a root-launched process are a model
tag and two integers, and each is validated against a fixed allowlist or a
numeric range *before* it is placed into a **fixed argument vector**. There is
no shell, no ``PATH`` lookup a caller could influence, and no string a caller
supplies is ever interpolated into a command. The binary path, the home
directory and the flag names are module constants, not derived from a request.

``flm-real`` needs root to pin NPU pages (CAP_IPC_LOCK). The Vaelor hardware
bridge (:mod:`vaelor.hardware_bridge`) already runs as root — it asserts
``geteuid()==0`` before it serves — so a child it spawns inherits root's full
capability set, CAP_IPC_LOCK among them, without this module manipulating a
single capability itself. The privilege comes from the bridge; the mechanism
(a validated ``flm serve`` invocation) comes from here.

The runtime is **FastFlowLM installed directly at** ``/var/lib/vaelor/flm``, not
the lemonade-server snap: ``install-vaelor.sh`` unpacks the pinned FLM tarball
(``flm`` wrapper, ``flm-real`` binary, ``lib/``, ``xclbins/``) into that
Vaelor-owned directory, and ``fetch-npu-model.sh`` installs the model under
``.../flm/models``. Vaelor invokes the ``flm`` wrapper at that fixed path with a
validated argv (VD-001). The security model is unchanged by the decouple: the
binary path, the subcommand and the flag names are module constants, the tag and
two integers are validated before they enter a **fixed argument vector**, there
is no shell and no ``PATH`` lookup — the wrapper is a fixed, trusted executable
invoked by absolute path with a fixed argv.

Discovery is separate from launch and deliberately unprivileged: whether the
binary, the device and the model are present is read from the filesystem, and a
missing precondition is reported as honest absence rather than guessed around
(the NPU tier is *not served* until all three are confirmed).
"""

from __future__ import annotations

import glob
import os
import re
import signal
import subprocess
import time
from pathlib import Path
from typing import Any, Callable, Dict, Optional

from .inference_context import NPU_CONTEXT_FLAG, NPU_NATIVE_CONTEXT_TOKENS
from .inference_model_choice import INSTALLED_FLM_MODELS
from .runtime_paths import LOG_ROOT


#: The FastFlowLM runtime + model directory Vaelor owns. FLM resolves its model
#: dir from this via ``FLM_MODEL_PATH`` (it reads ``$FLM_MODEL_PATH/models/``),
#: and it is where ``install-vaelor.sh`` unpacks the pinned FLM tarball and
#: ``fetch-npu-model.sh`` installs the model. The bridge unit runs
#: ``ProtectHome=true``, so ``~/.config`` is inaccessible and this path is what
#: FLM is pointed at instead (VD-001).
FLM_MODEL_PATH = "/var/lib/vaelor/flm"

#: The ``flm`` wrapper FastFlowLM ships. A fixed path, never a ``PATH`` lookup:
#: the launch crosses a root boundary and the executable must not be anything a
#: caller or a mutable environment could redirect. The wrapper sets
#: ``LD_LIBRARY_PATH`` + ``XILINX_XRT`` and creates the ``lib/x86_64-linux-gnu``
#: symlinks on first run, then execs ``flm-real``; it is a fixed, trusted
#: executable invoked by absolute path with a fixed argv (VD-001).
FLM_BINARY = FLM_MODEL_PATH + "/flm"

#: The default neural-accelerator device node, matching
#: :func:`vaelor.platforms.accelerators.discover_npus`. Discovery takes the
#: real node from the tier when one is known; this is the fallback name.
#:
#: **This is NOT the device-presence signal.** ``flm-real`` opens this node, but
#: the *routing decision* in :func:`discover_npu_serving` does not: the executor
#: unit runs ``PrivateDevices=yes``, which mounts a private ``/dev`` with the
#: accel node absent, so stat-ing ``/dev/accel/accel0`` inside the daemon is
#: always False even on a Z2 that is serving. Presence is read from sysfs
#: (:data:`NPU_SYSFS_CLASS_GLOB`) instead. Kept as a constant because the tier
#: still records which node it expects, and the privileged bridge (which runs
#: ``PrivateDevices=no``) is what actually opens it.
NPU_DEVICE_NODE = "/dev/accel/accel0"

#: The device-presence signal, and the reason FIX exists: a glob over the sysfs
#: accel *class* directory. ``PrivateDevices=yes`` gives the executor a private
#: ``/dev`` in which ``/dev/accel/accel0`` does not exist, but it does not touch
#: ``/sys`` — so ``/sys/class/accel/accel0`` (a symlink to the PCI device) is
#: visible inside the daemon and is the stable "an accel/NPU device exists"
#: signal. :func:`discover_npu_serving` only STATS the device to decide routing;
#: nothing here opens it, so reading presence from sysfs weakens no boundary and
#: does not require relaxing the sandbox (proven on the Z2: node absent under
#: nsenter into the executor namespace, sysfs class node present).
NPU_SYSFS_CLASS_GLOB = "/sys/class/accel/accel*"

#: Ports the control plane holds for itself. Refused at the root boundary, not
#: only in the caller-side allocator, so the process launcher itself will not
#: bind flm-real onto a control-plane port however it is reached (F2). One home
#: for the pair, imported by the supervisor rather than re-spelled there.
RESERVED_CONTROL_PLANE_PORTS = frozenset({34001, 34002})

#: Loopback only. ``flm serve`` is bound to this and nothing else: the Assistant
#: is a process on the same machine and the endpoint is never offered to the
#: LAN. A non-loopback host is refused by :func:`flm_serve_command`.
FLM_HOST = "127.0.0.1"

#: The one subcommand this module ever runs. Named so the argv builder cannot be
#: repurposed into an arbitrary ``flm`` invocation by a caller passing a verb.
FLM_SERVE_SUBCOMMAND = "serve"

#: The shape a legitimate FLM tag has, as defence in depth behind the allowlist:
#: ``family:size`` with only lowercase alphanumerics, dots and hyphens. It
#: admits ``qwen3.5:4b`` and ``gpt-oss:20b`` and rejects anything carrying a
#: space, slash, semicolon, quote or other shell-significant byte. The allowlist
#: is the real gate — a tag FLM has never been told about is "model not found"
#: however well-formed — but the pattern makes a malformed allowlist entry
#: impossible to smuggle through as well.
FLM_TAG_PATTERN = re.compile(r"^[a-z0-9]+(?:[.\-][a-z0-9]+)*:[a-z0-9]+(?:[.\-][a-z0-9]+)*$")


class FlmTagError(ValueError):
    """A model tag was refused before it could reach a root-launched process.

    LESSONS #178 boundary: the tag is the one string that crosses into the
    privileged ``flm serve`` launch, so a tag that is not on the installed
    allowlist (or is malformed) is rejected here, at planning time, rather than
    shell-interpolated and discovered as a failure at request time.
    """


def validate_flm_tag(tag: str, *, installed=INSTALLED_FLM_MODELS) -> str:
    """Return ``tag`` when it is safe to launch, else raise :class:`FlmTagError`.

    Two gates, both required. The allowlist (``installed``) is the tags
    ``flm-real`` was actually told about on the measured machine; a tag outside
    it is unknown to FLM and has no business reaching the command line. The
    pattern is defence in depth: it guarantees the value carries no
    shell-significant or path-significant byte, so even a future allowlist entry
    cannot become an injection vector.
    """
    name = str(tag or "")
    if name not in set(installed):
        raise FlmTagError(
            "'{}' is not an installed flm-real tag. Only the tags read from the "
            "machine's own inventory may be launched; a tag outside that set is "
            "'model not found' to FLM and is refused here rather than sent to a "
            "root-launched process (LESSONS #178 / VD-001).".format(name)
        )
    if not FLM_TAG_PATTERN.match(name):
        raise FlmTagError(
            "'{}' is on the installed list but does not match the flm-real tag "
            "grammar, so it is refused as a defence-in-depth measure: nothing "
            "carrying a shell- or path-significant byte reaches the launch "
            "(LESSONS #178).".format(name)
        )
    return name


def _validate_ctx_len(ctx_len: Any) -> int:
    try:
        value = int(ctx_len)
    except (TypeError, ValueError) as error:
        raise ValueError("The context length must be an integer.") from error
    if not 1 <= value <= NPU_NATIVE_CONTEXT_TOKENS:
        raise ValueError(
            "The context length must be between 1 and {}.".format(
                NPU_NATIVE_CONTEXT_TOKENS
            )
        )
    return value


def _validate_port(port: Any) -> int:
    try:
        value = int(port)
    except (TypeError, ValueError) as error:
        raise ValueError("The port must be an integer.") from error
    if not 1024 <= value <= 65535:
        raise ValueError("Choose a port from 1024 to 65535.")
    # F2: the boundary that actually launches the process refuses the reserved
    # ports too, not only the caller-side allocator. A launch reached by any
    # path cannot bind flm-real onto a control-plane port.
    if value in RESERVED_CONTROL_PLANE_PORTS:
        raise ValueError(
            "Ports {} are reserved for the Vaelor control plane.".format(
                " and ".join(str(p) for p in sorted(RESERVED_CONTROL_PLANE_PORTS))
            )
        )
    return value


def flm_serve_command(
    tag: str,
    *,
    ctx_len: int,
    port: int,
    host: str = FLM_HOST,
    installed=INSTALLED_FLM_MODELS,
) -> list:
    """The fixed argument vector for ``flm serve``, every element validated.

    A list, never a string: :class:`subprocess.Popen` is given ``argv`` and no
    shell, so no element can be split, expanded or chained. The tag is checked
    against the allowlist and the grammar; the two integers are range-checked;
    the host must be loopback. The binary, the subcommand and the context flag
    are constants. This is the whole of what crosses the root boundary.
    """
    safe_tag = validate_flm_tag(tag, installed=installed)
    safe_ctx = _validate_ctx_len(ctx_len)
    safe_port = _validate_port(port)
    if str(host) not in {"127.0.0.1", "::1", "localhost"}:
        raise ValueError(
            "flm-real is bound to loopback only; '{}' is not a loopback host."
            .format(host)
        )
    return [
        FLM_BINARY,
        FLM_SERVE_SUBCOMMAND,
        safe_tag,
        NPU_CONTEXT_FLAG,
        str(safe_ctx),
        "--port",
        str(safe_port),
        "--host",
        str(host),
        "--quiet",
    ]


def flm_serve_env() -> Dict[str, str]:
    """The launch environment: the root service's own, plus the FLM model path.

    FLM resolves its model dir from ``FLM_MODEL_PATH`` (it reads
    ``$FLM_MODEL_PATH/models/``). The bridge unit runs ``ProtectHome=true``, so
    ``~/.config`` is inaccessible and ``FLM_MODEL_PATH`` is required — without it
    FLM would look under an unreadable home and find no models (VD-001).
    ``FLM_DISABLE_UPDATE_CHECK`` pins the runtime rather than letting the wrapper
    reach out. Everything else is inherited from the privileged service's
    environment rather than stripped: the ``flm`` wrapper needs the process
    environment (its libraries, XDG dirs) to start, and a minimal hand-built
    environment is exactly the kind of plausible mechanism CLAUDE.md warns is
    refuted the moment it is run on the box. The environment is not
    caller-controlled — it is the root unit's own — so the security boundary is
    the validated argv above, not this dict.
    """
    return {
        **os.environ,
        "FLM_MODEL_PATH": FLM_MODEL_PATH,
        "FLM_DISABLE_UPDATE_CHECK": "1",
    }


def npu_serving_verdict(
    *,
    binary_present: bool,
    device_present: bool,
    model_installed: bool,
    tag: str = "",
) -> Dict[str, Any]:
    """Whether flm-real serving is available, and — when not — exactly why.

    All three preconditions must hold. Honest absence is the point: a machine
    missing any one of them (every Raspberry Pi misses all three) reports the
    specific gap rather than a bare False, so the NPU tier is described as *not
    served yet* with a reason an operator can act on, never as *unsupported*.
    """
    available = bool(binary_present and device_present and model_installed)
    if available:
        reason = ""
    else:
        missing = []
        if not binary_present:
            missing.append("the flm-real binary is not present at {}".format(FLM_BINARY))
        if not device_present:
            missing.append("no neural-accelerator device node was found")
        if not model_installed:
            missing.append(
                "the assistant model{} is not in the installed flm-real inventory"
                .format(" ({})".format(tag) if tag else "")
            )
        reason = (
            "Vaelor does not serve the neural processor on this machine: {}. "
            "The NPU tier becomes available once flm-real, the device and the "
            "model are all present (VD-001)."
        ).format("; ".join(missing))
    return {
        "available": available,
        "binary_present": bool(binary_present),
        "device_present": bool(device_present),
        "model_installed": bool(model_installed),
        "tag": str(tag or ""),
        "reason": reason,
    }


def _path_present(path: str) -> bool:
    """Filesystem presence, glob-aware. A pattern is matched, a path is stat'd.

    :data:`NPU_SYSFS_CLASS_GLOB` is a glob, so ``Path(pattern).exists()`` would
    always be False; :func:`glob.glob` is used for a pattern and a plain
    ``exists`` for a concrete path such as :data:`FLM_BINARY`.

    A path whose parent directory the caller cannot traverse raises
    ``PermissionError`` (``EACCES``) from ``stat`` rather than returning False;
    that is swallowed to False here so the caller reads "not visible to me"
    rather than propagating — the whole point of the bridge-backed
    :func:`flm_binary_present`, which is consulted next when the direct stat
    cannot see the binary.
    """
    if any(character in path for character in "*?["):
        return bool(glob.glob(path))
    try:
        return Path(path).exists()
    except OSError:
        return False


def _bridge_binary_present() -> Optional[bool]:
    """Ask the privileged bridge whether flm-real is present, as root.

    Returns True/False when the bridge answers, or None when it is unreachable
    (no socket, a fresh box, a test with no bridge) so the caller can fall back
    to the direct stat. The bridge runs ``PrivateDevices=no`` as root and owns
    the launch, so it can traverse the root-only parent directories the executor
    cannot — the same "the truthful reader is the privileged one" pattern RAPL
    and ECC already use. A pure stat with no process state touched, so it takes
    no bridge lock and cannot contend with a launch.
    """
    try:
        from .hardware_bridge import HardwareBridgeClient

        client = HardwareBridgeClient()
        if not client.available:
            return None
        present = client.flm_binary_present().get("present")
        return None if present is None else bool(present)
    except Exception:
        # Never let a presence probe raise into discovery: an unreachable or
        # misbehaving bridge is "I could not confirm", i.e. fall back to the stat.
        return None


def _sandbox_visible_binary_marker() -> bool:
    """The signal that FastFlowLM is installed, visible to the executor.

    The install marker is now the FastFlowLM runtime directory itself:
    ``Path(FLM_BINARY).parent`` is ``/var/lib/vaelor/flm``, the Vaelor-owned
    directory ``install-vaelor.sh`` unpacks the FLM tarball into. It exists iff
    FastFlowLM is installed — a fresh box or a Pi has no such directory — so its
    presence is the "flm is installed" signal, the same move ``device_present``
    made to the sysfs class node. It is a directory-level signal (not the leaf
    binary), which is why the truthful bridge probe is preferred over it whenever
    the bridge can answer.
    """
    return _path_present(str(Path(FLM_BINARY).parent))


def flm_binary_present(
    *,
    filesystem: Optional[Callable[[], bool]] = None,
    bridge: Optional[Callable[[], Optional[bool]]] = None,
    marker: Optional[Callable[[], bool]] = None,
) -> bool:
    """Whether the flm-real binary is present, by a probe that can REACH it.

    The deeper root cause behind VD-001's "the reconcile silently no-ops": the
    executor runs as ``vaelor-workloads`` and an Aug-24 snap refresh set the
    binary's parent directories (``.../bin/flm`` and ``.../bin/flm/npu``) to mode
    ``0770 root:root``. A non-root user cannot TRAVERSE those directories, so a
    direct ``Path(FLM_BINARY).exists()`` from inside the executor is False even
    though the binary itself is world-executable and the root bridge launches it
    fine. That made ``binary_present`` False → ``available`` False →
    ``serve_on_npu`` False, and the boot/failure reconcile no-oped forever — the
    same shape of sandbox-hidden signal that made ``device_present`` always False
    off the ``/dev`` node, fixed the same way: read the signal where it is true.

    Three layers, most-truthful first, each a defaulted seam:

    #. **Direct stat.** If the executor can itself stat the binary
       (``filesystem`` True), it is present — the fast path on a box with intact
       permissions, no round-trip and no ambiguity.
    #. **Privileged bridge.** Otherwise ask the bridge (``bridge``), which as root
       (``PrivateDevices=no``) traverses the root-only directories and stats
       exactly what the LAUNCHER will — the truthful leaf check. True/False is the
       bridge's answer and WINS over the marker below; None means the bridge could
       not answer (unreachable, or a bridge too old to know the query — the state
       during a live executor-only hot-patch before the bridge is updated).
    #. **Sandbox-visible install marker.** When the bridge cannot answer, fall
       back to :func:`_sandbox_visible_binary_marker` — the ``.../bin/flm`` entry
       the executor CAN stat. This is what lets the gate read present on the box
       the moment the executor restarts, without cycling the bridge (which would
       kill the running models). A genuinely fresh box has no such directory and
       reads absent, so this is a signal, not an always-True.
    """
    look_fs = filesystem or (lambda: _path_present(FLM_BINARY))
    ask_bridge = bridge or _bridge_binary_present
    look_marker = marker or _sandbox_visible_binary_marker
    if look_fs():
        return True
    verdict = ask_bridge()
    if verdict is not None:
        return bool(verdict)
    return bool(look_marker())


def discover_npu_serving(
    tag: str,
    *,
    device_node: str = NPU_DEVICE_NODE,
    installed=INSTALLED_FLM_MODELS,
    exists: Callable[[str], bool] = None,
    model_present: Callable[[], bool] = None,
) -> Dict[str, Any]:
    """Read the three preconditions off this machine and return the verdict.

    ``exists`` is injectable so a test can present a machine with or without the
    binary and device without touching the filesystem. Model presence is now a
    LIVE check: the tag must be a recognised model (:data:`INSTALLED_FLM_MODELS`)
    AND a model must actually be on disk (:func:`npu_model_present`, injectable as
    ``model_present`` for tests). The static inventory alone reported "installed"
    on a box where the model files were never fetched, so the setup card claimed
    "served on the neural processor; no download" before anything was there; the
    live probe closes that (the deferred "deeper cache probe" this docstring used
    to note). The three filesystem checks are the live half of the gate, and they
    are what make a Pi — or a fresh box with no model — honestly absent.

    **Device presence is read from sysfs, not from the ``/dev`` node.** The
    executor unit runs ``PrivateDevices=yes``, so ``/dev/accel/accel0`` does not
    exist inside the daemon even on a Z2 that is serving; stat-ing it there made
    ``device_present`` always False and the assistant deploy never routed to the
    NPU. :data:`NPU_SYSFS_CLASS_GLOB` (``/sys/class/accel/accel*``) is the signal
    the sandbox leaves visible. ``device_node`` is retained as the node identity
    the tier expects — the privileged bridge opens it — but this routing check
    deliberately does not touch it.

    **Binary presence is read where it is true, not off a root-only path.** For
    the same reason the ``/dev`` node lied about the device, an Aug-24 snap
    refresh made the binary's parent directories untraversable to the executor,
    so a direct stat of :data:`FLM_BINARY` reads absent even though the root
    bridge launches it fine. Production (``exists is None``) routes the binary
    check through :func:`flm_binary_present`, which stats it where it is visible
    (the privileged bridge). ``exists`` is the filesystem-only seam a test injects
    to present a machine with or without the binary; when injected it governs the
    binary check too, so the discovery stays testable without a bridge.
    """
    look = exists if exists is not None else _path_present
    binary_present = bool(look(FLM_BINARY)) if exists is not None else flm_binary_present()
    look_model = model_present if model_present is not None else npu_model_present
    return npu_serving_verdict(
        binary_present=binary_present,
        device_present=bool(look(NPU_SYSFS_CLASS_GLOB)),
        # A recognised tag AND a model actually on disk - not the static set alone.
        model_installed=(str(tag or "") in set(installed)) and bool(look_model()),
        tag=tag,
    )


def npu_model_present() -> bool:
    """Whether an NPU model is already installed on disk under FLM_MODEL_PATH.

    ``fetch-npu-model.sh`` installs the model into ``$FLM_MODEL_PATH/models/``
    (a ``0755 root:root`` tree the unprivileged executor can traverse and list),
    so a non-empty ``models`` directory means the model is already on disk and
    ``model.install_release`` can serve+pin it without re-downloading. A fresh
    box or a Pi has no such directory and reads absent.
    """
    models = Path(FLM_MODEL_PATH) / "models"
    try:
        return any(child.is_dir() for child in models.iterdir())
    except OSError:
        return False


#: Where flm-real's own stdout and stderr are kept. **Not discarded to
#: DEVNULL** (LESSONS pattern 8): a model server that fails to start - "model
#: not found", an NPU initialisation error - says so on stderr, and throwing
#: that away makes "flm-real could not start" and "flm-real is fine" arrive
#: looking identical. The supervisor detects the failure by the health timeout;
#: this is where the *reason* survives so an operator can read it.
FLM_LOG_FILE = str(Path(LOG_ROOT) / "flm-real.log")

#: How long to wait for a SIGTERM'd flm-real to exit before SIGKILL. A model
#: server unmapping pinned NPU pages is not instant, but it is not slow either;
#: past this the process is not stopping on its own and is killed.
STOP_GRACE_SECONDS = 10.0


class FlmProcess:
    """A supervised ``flm serve`` process, launched and stopped as root.

    Constructed and driven from inside the privileged hardware bridge. The
    spawn is injectable so the launch argv/env can be verified — and the whole
    lifecycle exercised — without a real NPU or root. Production passes no
    ``spawn`` and gets :class:`subprocess.Popen` with a **fixed argv and env**,
    no shell, ``stdin`` closed and file descriptors closed.
    """

    def __init__(
        self,
        *,
        spawn: Optional[Callable[..., Any]] = None,
        installed=INSTALLED_FLM_MODELS,
    ):
        self._spawn = spawn or self._default_spawn
        self._installed = installed
        self._process: Any = None
        self._log: Any = None
        self.tag = ""
        self.port = 0
        self.ctx_len = 0

    def _default_spawn(self, command, env):
        # stdout and stderr go to a log file, never DEVNULL: a launch that fails
        # explains itself there (LESSONS pattern 8). The handle is kept on the
        # instance so it lives as long as the process writing to it.
        log_path = Path(FLM_LOG_FILE)
        log_path.parent.mkdir(parents=True, exist_ok=True)
        self._log = open(log_path, "a", encoding="utf-8")
        return subprocess.Popen(
            command,
            env=env,
            stdin=subprocess.DEVNULL,
            stdout=self._log,
            stderr=subprocess.STDOUT,
            close_fds=True,
        )

    def start(self, tag: str, *, ctx_len: int, port: int) -> Dict[str, Any]:
        """Validate, then launch. A running process is stopped first.

        Validation happens before the spawn and before the previous process is
        touched, so a bad tag cannot take down a healthy server.
        """
        command = flm_serve_command(
            tag, ctx_len=ctx_len, port=port, installed=self._installed
        )
        if self.alive():
            self.stop()
        self._process = self._spawn(command, flm_serve_env())
        self.tag = validate_flm_tag(tag, installed=self._installed)
        self.ctx_len = _validate_ctx_len(ctx_len)
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
            "tag": self.tag,
            "port": self.port,
            "ctx_len": self.ctx_len,
            "pid": getattr(self._process, "pid", None) if self._process else None,
        }
