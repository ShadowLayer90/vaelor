"""The root program behind the GPU kernel profile's tracing serve (VD-214).

Run only as ``python -I -m vaelor.gpu_kernel_profile_mode apply <unit>`` or
``... revert``, by ``systemd-run`` from the root hardware bridge
(`gpu_kernel_profile`): as root, outside the bridge's ``ProtectSystem=strict``
sandbox, which cannot write ``/run/systemd/system`` - and, for ``revert``,
from the deadline's timer, which systemd runs whether or not the bridge, the
console or the request that started the profile is still there.

* ``apply`` reads the server unit the bridge installed, derives the tracing
  ``ExecStart`` from it (`gpu_kernel_profile.tracing_exec_start`, every rule
  checked again here), writes the runtime drop-in, reloads systemd and queues a
  restart of the unit - ``try-restart``, so a unit the owner stopped is left
  stopped.
* ``revert`` removes every kernel-profile drop-in, reloads, and restarts each
  unit it removed one from. It then looks for any 0.27 server whose installed
  unit carries VD-165's switch but whose running process does not, however it
  got there, and restarts that too: after a revert, no copy serves with GPU
  tracing on. Idempotent - with nothing to undo it changes nothing, so the
  deadline firing after the bridge already reverted is harmless.
"""

from __future__ import annotations

import json
import os
import shutil
import stat
import subprocess
import sys
from typing import Any, Callable, List, Optional, Sequence

from . import gpu_kernel_profile as kernel
from .bridge_managed_units import UNIT_DIRECTORY
from .gpu_pool_units import SERVER_ROLE, unit_role

#: A unit file is small; anything larger is not one of ours.
_UNIT_LIMIT = 64 * 1024
#: How long one ``systemctl`` or ``docker inspect`` may take: the bounds the
#: bridge's worst case is computed from (`gpu_kernel_profile`).
_STEP_SECONDS = kernel.SYSTEMCTL_SECONDS
_INSPECT_SECONDS = kernel.INSPECT_SECONDS

_READ_FLAGS = (os.O_RDONLY | getattr(os, "O_NOFOLLOW", 0) | getattr(os, "O_NONBLOCK", 0)
               | getattr(os, "O_BINARY", 0))
_WRITE_FLAGS = (os.O_WRONLY | os.O_CREAT | os.O_EXCL | getattr(os, "O_NOFOLLOW", 0)
                | getattr(os, "O_BINARY", 0))


def _systemctl(*arguments: str) -> None:
    completed = subprocess.run(
        [shutil.which("systemctl") or kernel.SYSTEMCTL, *arguments],
        capture_output=True, text=True, timeout=_STEP_SECONDS, check=False,
    )
    if completed.returncode != 0:
        raise RuntimeError(
            str(completed.stderr or "").strip()[:300]
            or "The kernel profile's systemctl {} did not succeed.".format(arguments[0]))


def read_regular(path: str, limit: int = _UNIT_LIMIT) -> str:
    """A regular file's text, following no link, at most ``limit`` bytes."""
    handle = os.open(path, _READ_FLAGS)
    try:
        if not stat.S_ISREG(os.fstat(handle).st_mode):
            raise ValueError(kernel.UNIT_NOT_TEMPLATE)
        data = os.read(handle, limit + 1)
    finally:
        os.close(handle)
    if len(data) > limit:
        raise ValueError(kernel.UNIT_NOT_TEMPLATE)
    return data.decode("utf-8")


def _write_atomically(directory: str, name: str, text: str) -> None:
    temporary = os.path.join(directory, ".{}.{}".format(name, os.getpid()))
    handle = os.open(temporary, _WRITE_FLAGS, 0o644)
    try:
        os.write(handle, text.encode("utf-8"))
    finally:
        os.close(handle)
    os.replace(temporary, os.path.join(directory, name))


def apply(
    unit: str, *, unit_directory: str = UNIT_DIRECTORY,
    dropin_root: str = kernel.DROPIN_ROOT, systemctl: Callable[..., None] = _systemctl,
) -> dict:
    """Write the tracing drop-in for ``unit`` and queue its restart."""
    if unit_role(unit) != SERVER_ROLE:
        raise ValueError(kernel.UNIT_NOT_TEMPLATE)
    start = kernel.tracing_exec_start(unit, read_regular(os.path.join(unit_directory, unit)))
    directory = os.path.join(dropin_root, unit + ".d")
    try:
        os.mkdir(directory, 0o755)
    except FileExistsError:
        if not stat.S_ISDIR(os.lstat(directory).st_mode):
            raise ValueError(kernel.UNIT_NOT_TEMPLATE)
    _write_atomically(directory, kernel.DROPIN_NAME, kernel.dropin_text(start))
    systemctl("daemon-reload")
    systemctl("try-restart", "--no-block", unit)
    return {"applied": unit}


def _container_environ(container: str) -> Optional[bytes]:
    """The running container's main process environment, ``None`` when unread."""
    try:
        completed = subprocess.run(
            [shutil.which("docker") or "/usr/bin/docker", "inspect", "-f",
             "{{.State.Pid}}", container],
            capture_output=True, text=True, timeout=_INSPECT_SECONDS, check=False,
        )
        pid = int(str(completed.stdout or "").strip()) if completed.returncode == 0 else 0
        if pid <= 0:
            return None
        with open("/proc/{}/environ".format(pid), "rb") as handle:
            return handle.read()
    except (OSError, ValueError, subprocess.SubprocessError):
        return None


def serving_with_tracing(
    unit_directory: str = UNIT_DIRECTORY,
    environ_of: Callable[[str], Optional[bytes]] = _container_environ,
) -> List[str]:
    """Server units installed WITH the switch whose running process lacks it.

    A process whose environment could not be read is not counted: unread is
    not "tracing" (LESSONS 8), and the drop-in removal above is the main path.
    """
    switch = ("-e " + kernel.REGISTRATION_SWITCH + " ")
    found: List[str] = []
    try:
        names = sorted(os.listdir(unit_directory))
    except OSError:
        return found
    for name in names:
        try:
            if unit_role(name) != SERVER_ROLE:
                continue
            text = read_regular(os.path.join(unit_directory, name))
        except (OSError, ValueError):
            continue
        if switch not in text:
            continue
        environ = environ_of(name[: -len(".service")])
        if environ is not None and kernel.REGISTRATION_SWITCH.encode() not in environ.split(b"\0"):
            found.append(name)
    return found


def revert(
    *, unit_directory: str = UNIT_DIRECTORY, dropin_root: str = kernel.DROPIN_ROOT,
    systemctl: Callable[..., None] = _systemctl,
    environ_of: Callable[[str], Optional[bytes]] = _container_environ,
    hold: Callable[[str], None] = None,  # type: ignore[assignment]
) -> dict:
    """Remove every kernel-profile drop-in and restart what served with tracing on.

    **systemd is reloaded before ANY restart, every time** (adversarial review
    1): a drop-in removed by an earlier revert whose reload failed is still
    the loaded ``ExecStart``, and a restart without a reload would start the
    tracing serve again. A reload that fails raises, so the caller - the
    bridge, or the deadline's timer - reports a failed revert and the timer
    stays armed. Removal tolerates a drop-in another revert removed first.
    """
    removed: List[str] = []
    try:
        folders = sorted(os.listdir(dropin_root))
    except OSError:
        folders = []
    for folder in folders:
        unit = kernel.server_unit_of(folder)
        if unit is None:
            continue
        try:
            os.unlink(os.path.join(dropin_root, folder, kernel.DROPIN_NAME))
            removed.append(unit)
        except FileNotFoundError:
            pass  # not ours, or another revert removed it first
        try:
            os.rmdir(os.path.join(dropin_root, folder))
        except OSError:
            pass  # another drop-in of someone else's is still there
    # Always, even with nothing found (review 2): an earlier revert may have
    # removed a drop-in and failed to reload, and a process whose environment
    # cannot be read is not found by the check below.
    systemctl("daemon-reload")
    restarted = sorted(set(removed) | set(serving_with_tracing(unit_directory, environ_of)))
    for unit in restarted:
        # The copy is down for its restart: hold the watch, bounded (VD-214).
        (hold or _hold)(unit)
        systemctl("try-restart", "--no-block", unit)
    return {"removed": removed, "restarted": restarted}


def _hold(unit: str) -> None:
    """Hold the replica watch through this restart, at most the revert's wait."""
    name = unit[len("vaelor-vllm-"):-len("-server.service")]
    try:
        kernel.write_hold(name, seconds=kernel.REVERT_HEALTH_WAIT_SECONDS)
    except OSError:
        pass  # no /run/vaelor: no watch to hold


def main(argv: Sequence[str]) -> int:
    try:
        if list(argv[:1]) == ["revert"] and len(argv) == 1:
            outcome: Any = revert()
        elif list(argv[:1]) == ["apply"] and len(argv) == 2:
            outcome = apply(str(argv[1]))
        else:
            raise ValueError("usage: apply <unit> | revert")
    except (OSError, RuntimeError, ValueError) as error:
        print(json.dumps({"ok": False, "error": str(error)[:400]}), flush=True)
        return 1
    print(json.dumps({"ok": True, **outcome}), flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
