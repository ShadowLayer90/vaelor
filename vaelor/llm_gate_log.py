"""The ROOT side of the LLM Server gate's usage log: its directory and its rotation.

Runs inside the root hardware bridge, next to the proxy launch
(:class:`vaelor.llm_server_proxy.LlmServerProxyProcess`); the control plane reads
the log unprivileged (:mod:`vaelor.llm_gate_usage`). Two jobs, both at a root
boundary, which is why they are kept apart from the proxy's rendering.

**The directory (a root boundary).** The log directory lives under
``/run/vaelor``, which is group-writable to every ``vaelor`` service. Root then
bind-mounts it into a container whose nginx master (root) opens a file in it, so
a directory a group member could replace by a link would steer a root writer
anywhere. :func:`prepare_log_dir` therefore:

* puts the sticky bit on the parent first (as `agent_service` does for its
  config directory) and REFUSES the log when the parent is still group- or
  other-writable without it - a box installed before the tmpfiles entry gained
  the sticky bit keeps the old mode, because a wheel deploy never rewrites
  tmpfiles; with sticky, no group member can rename the root-owned directory
  aside, so the path docker re-resolves at mount time stays the one checked;
* opens the directory with ``O_DIRECTORY | O_NOFOLLOW`` and checks and changes
  it through that descriptor (``fstat``/``fchmod``), never by path, so a link
  planted at the name is refused rather than followed.

A refusal never costs the gate: the proxy is launched without the log and its
status says why (LESSONS pattern 1 - a gate that is not recording must say so).

**The rotation (never on the status path).** :class:`GateLogRotator` is the
bridge's own timer: past :data:`~vaelor.llm_gate_usage.GATE_LOG_ROTATE_BYTES`
it renames the file to ``.1`` and sends nginx ``USR1`` to reopen. It checks the
``docker kill`` answer; while no fresh file appears and ``.1`` keeps growing it
signals again; past :data:`GATE_LOG_CEILING_BYTES` it restarts the container
(and, if even that fails, truncates ``.1``) so a tmpfs cannot fill. Every
failure is caught and kept as a reason; a status read never rotates and never
raises. On reopen nginx chowns the fresh file to its worker user (uid 101 in
the stock image) and keeps mode ``0644``, so the control plane's read access,
which comes from the directory, is unchanged.
"""

from __future__ import annotations

import errno
import logging
import os
import stat
import subprocess
import threading
from typing import Any, Callable, Optional, Tuple

from .llm_gate_usage import (
    GATE_LOG_FILENAME,
    GATE_LOG_ROTATE_BYTES,
    GATE_LOG_ROTATED_SUFFIX,
    GATE_REFUSED_CAP_BYTES,
    GATE_REFUSED_FILENAME,
)

LOGGER = logging.getLogger(__name__)

#: How often the bridge checks the logs' sizes: often enough that a busy key log
#: is rotated near its cap and a refusal flood is capped quickly.
ROTATE_CHECK_SECONDS = 10.0

#: Past this, a log nginx never reopened is capped by restarting the gate.
GATE_LOG_CEILING_BYTES = 8 * GATE_LOG_ROTATE_BYTES

#: How long a ``docker kill``/``restart`` may take.
SIGNAL_TIMEOUT_SECONDS = 30

#: The two sentences this module refuses and fails with, each written once.
LOG_REFUSED = "The LLM Server is serving without its usage log, because {} {}."
ROTATION_FAILED = "The usage log could not be rotated ({})."


class LogDirFs:
    """The filesystem calls :func:`prepare_log_dir` makes: the seam a test fakes.

    POSIX semantics are the product's; on a host without ``O_DIRECTORY`` (a
    developer's Windows box, never an appliance) the directory is checked by
    ``lstat`` instead, which is not a boundary and does not claim to be.
    """

    posix = hasattr(os, "O_DIRECTORY") and hasattr(os, "O_NOFOLLOW")

    def lstat(self, path: str) -> os.stat_result:
        return os.lstat(path)

    def chmod(self, path: str, mode: int) -> None:
        os.chmod(path, mode)

    def mkdir(self, path: str, mode: int) -> None:
        os.mkdir(path, mode)

    def open_dir(self, path: str) -> Any:
        if not self.posix:
            info = os.lstat(path)
            if stat.S_ISLNK(info.st_mode):
                raise OSError(errno.ELOOP, "a link, not a directory")
            return path
        return os.open(path, os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW)

    def fstat(self, handle: Any) -> os.stat_result:
        return os.lstat(handle) if isinstance(handle, str) else os.fstat(handle)

    def fchmod(self, handle: Any, mode: int) -> None:
        if isinstance(handle, str):
            os.chmod(handle, mode)
        else:
            os.fchmod(handle, mode)

    def close(self, handle: Any) -> None:
        if not isinstance(handle, str):
            os.close(handle)

    def uid(self) -> Optional[int]:
        return os.getuid() if hasattr(os, "getuid") else None


def _unsafe_parent(mode: int) -> bool:
    return bool(mode & (stat.S_IWGRP | stat.S_IWOTH)) and not mode & stat.S_ISVTX


def prepare_log_dir(path: str, fs: Optional[LogDirFs] = None) -> Tuple[str, str]:
    """``(path, "")`` when the directory is safe to mount, else ``("", reason)``."""
    fs = fs or LogDirFs()
    parent = os.path.dirname(path.rstrip("/\\")) or path
    try:
        parent_mode = stat.S_IMODE(fs.lstat(parent).st_mode)
        # Group/other write bits and the sticky bit mean nothing on a
        # non-POSIX development host; the check is the appliance's.
        if getattr(fs, "posix", True) and _unsafe_parent(parent_mode):
            try:
                fs.chmod(parent, parent_mode | stat.S_ISVTX)
            except OSError:
                pass
            if _unsafe_parent(stat.S_IMODE(fs.lstat(parent).st_mode)):
                return "", LOG_REFUSED.format(parent, (
                    "is writable by other accounts without the sticky bit, so the "
                    "log directory inside it could be swapped"
                ))
        try:
            fs.mkdir(path, 0o755)
        except FileExistsError:
            pass
        handle = fs.open_dir(path)
        try:
            info = fs.fstat(handle)
            owner = fs.uid()
            if not stat.S_ISDIR(info.st_mode):
                raise OSError(errno.ENOTDIR, "it is not a directory")
            if owner is not None and info.st_uid != owner:
                raise OSError(errno.EPERM, "it is not owned by the hardware bridge")
            fs.fchmod(handle, 0o755)
        finally:
            fs.close(handle)
    except OSError as error:
        return "", LOG_REFUSED.format(path, "could not be used ({})".format(error.strerror or error))
    return path, ""


def _default_run(command, timeout=None):
    return subprocess.run(
        list(command), capture_output=True, text=True, check=False, timeout=timeout,
    )


class GateLogRotator:
    """Keep the usage logs bounded, on the bridge's own timer.

    ``run`` is the docker seam (the proxy's own); ``container`` the gate's name;
    ``busy`` says a proxy start or stop is in flight, when a check is skipped
    rather than racing the container being replaced. :meth:`tick` is one check
    and is what a test drives; it never raises.

    Every file is looked at with ``lstat`` and acted on only when it is a
    regular file, and truncated through an ``O_NOFOLLOW`` descriptor: the
    directory is writable from inside the gate's container, so a link planted
    there must never steer this root process (N4).
    """

    #: The longest reason kept; a repeated failure replaces it, never appends.
    REASON_LIMIT = 300

    def __init__(
        self, directory: str, *, container: str, docker: Callable[[], str],
        run: Optional[Callable[..., Any]] = None,
        busy: Optional[Callable[[], bool]] = None,
        interval_seconds: float = ROTATE_CHECK_SECONDS,
        cap_bytes: int = GATE_LOG_ROTATE_BYTES,
        ceiling_bytes: int = GATE_LOG_CEILING_BYTES,
        refused_cap_bytes: int = GATE_REFUSED_CAP_BYTES,
    ):
        self._directory = directory
        self._container = container
        self._docker = docker
        self._run = run or _default_run
        self._busy = busy or (lambda: False)
        self._interval = float(interval_seconds)
        self._cap = int(cap_bytes)
        self._ceiling = int(ceiling_bytes)
        self._refused_cap = int(refused_cap_bytes)
        self._lock = threading.Lock()
        self._stop: Optional[threading.Event] = None
        self._thread: Optional[threading.Thread] = None
        self._awaiting_reopen = False
        self._rotated_size = 0
        self._reason = ""

    @property
    def reason(self) -> str:
        """Why the last check could not do its job, "" when it could."""
        return self._reason

    def _note(self, reason: str) -> None:
        self._reason = reason[: self.REASON_LIMIT]

    def _docker_call(self, *arguments: str) -> bool:
        """One docker call; False (with a reason kept) when it did not succeed."""
        command = [self._docker(), *arguments, self._container]
        try:
            result = self._run(command, timeout=SIGNAL_TIMEOUT_SECONDS)
        except (OSError, subprocess.SubprocessError) as error:
            self._note(ROTATION_FAILED.format(error))
            return False
        if getattr(result, "returncode", 1) != 0:
            self._note(ROTATION_FAILED.format(
                (getattr(result, "stderr", "") or "docker refused").strip()[:200]
            ))
            return False
        return True

    def tick(self) -> bool:
        """One size check; True when it rotated, re-signalled or capped. Never raises."""
        if self._busy() or not self._lock.acquire(blocking=False):
            return False
        try:
            capped = self._cap_refused()
            return self._tick() or capped
        except Exception as error:  # noqa: BLE001 - a rotation fault must not end the timer
            self._note(ROTATION_FAILED.format(error))
            LOGGER.warning("%s", self._reason)
            return False
        finally:
            self._lock.release()

    def _cap_refused(self) -> bool:
        """Empty the refused log in place once past its own cap.

        nginx appends to it with ``O_APPEND``, so truncation in place is safe
        and needs no reopen; the reader sees a file shorter than its place and
        starts it again. Refusals are a lower bound under a flood by design.
        """
        path = os.path.join(self._directory, GATE_REFUSED_FILENAME)
        size = _regular_size(path)
        if size is None or size < self._refused_cap:
            return False
        _truncate_no_follow(path)
        return True

    def _tick(self) -> bool:
        current = os.path.join(self._directory, GATE_LOG_FILENAME)
        rotated = current + GATE_LOG_ROTATED_SUFFIX
        current_size = _regular_size(current)
        if current_size is not None:
            self._awaiting_reopen = False
            if current_size < self._cap:
                self._note("")
                return False
            os.replace(current, rotated)
            self._awaiting_reopen = True
            self._rotated_size = _regular_size(rotated) or 0
            if self._docker_call("kill", "--signal", "USR1"):
                self._note("")
            return True
        size = _regular_size(rotated)
        if not self._awaiting_reopen or size is None:
            return False
        if size >= self._ceiling:
            # nginx never reopened: restart the gate so it opens a fresh file.
            if self._docker_call("restart"):
                self._awaiting_reopen = False
                return True
            _truncate_no_follow(rotated)
            self._note(self._reason + " The usage log was cut short to stay bounded.")
            return True
        if size > self._rotated_size:
            # Still writing into the renamed file: ask again.
            self._rotated_size = size
            self._docker_call("kill", "--signal", "USR1")
            return True
        return False

    def _loop(self, stop: threading.Event) -> None:
        while not stop.wait(self._interval):
            self.tick()

    def start(self) -> None:
        """Start the timer; a previous one is stopped first, never left running beside it."""
        self.stop()
        stop = threading.Event()
        self._stop = stop
        self._thread = threading.Thread(
            target=self._loop, args=(stop,), name="llm-gate-log-rotator", daemon=True,
        )
        self._thread.start()

    def stop(self) -> None:
        if self._stop is not None:
            self._stop.set()
        self._stop = None
        self._thread = None


def _regular_size(path: str) -> Optional[int]:
    """The size of ``path`` when it is a regular file (never through a link), else None."""
    try:
        info = os.lstat(path)
    except FileNotFoundError:
        return None
    return info.st_size if stat.S_ISREG(info.st_mode) else None


def _truncate_no_follow(path: str) -> None:
    """Truncate ``path`` to empty through a descriptor that refuses a link."""
    descriptor = os.open(path, os.O_WRONLY | getattr(os, "O_NOFOLLOW", 0))
    try:
        if not stat.S_ISREG(os.fstat(descriptor).st_mode):
            raise OSError(errno.EINVAL, "not a regular file")
        os.ftruncate(descriptor, 0)
    finally:
        os.close(descriptor)
