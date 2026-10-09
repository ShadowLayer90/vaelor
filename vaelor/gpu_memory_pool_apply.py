"""The root side of the GPU memory pool setting: write one file, rebuild the boot image.

VD-161. Two callers, one program:

* the root hardware bridge, for the controller (`hardware_bridge_host_settings`),
  which validates the request and then starts THIS module as a transient unit;
* a person at a shell (``sudo python -I -m vaelor.gpu_memory_pool_apply revert``).

**Why a transient unit and not the bridge's own process (LESSONS 13).** The
bridge unit runs under ``ProtectSystem=strict`` and ``ProtectKernelModules``:
it cannot write ``/etc/modprobe.d``, and a boot-image rebuild that cannot read
the kernel's modules cannot work. A test that called the rebuild in-process
would be green and the feature would fail on the appliance. So the bridge asks
systemd to run this program outside its sandbox, exactly as it does for the
state-root layout (`state_root_layout.start_layout_unit`), and no new unit
file has to reach the box.

**What this program will do, all of it:** write
:data:`~vaelor.gpu_memory_pool.CONFIG_PATH` with the one fixed line for a size
inside this machine's own bounds, or remove that file; then run the machine's
boot-image tool with a fixed argument. It takes one whole number and nothing
else. It never restarts the machine: the new size counts from the next
restart, which only the owner starts.

**A failed rebuild is rolled back and the roll-back is CHECKED** (review S2).
The previous file is put back and the boot image is rebuilt once more with it.
Only when both succeed does the refusal say the machine is as it was. If the
second rebuild fails, if the file could not be put back, or if the unit was
stopped part-way (``SIGTERM``, which is also what ``RuntimeMaxSec`` sends),
the refusal says exactly that and what to run - never "nothing has changed"
about a boot image nobody confirmed.

**Where the detail goes** (review S3). The unit's output is piped back to the
bridge, so ``journalctl -u`` of the transient unit holds nothing. The bridge
writes a bounded tail of this program's error output to ITS journal, and the
refusal names that unit.
"""

from __future__ import annotations

import json
import os
import re
import signal
import subprocess
import sys
from dataclasses import dataclass
from typing import Any, Callable, Dict, List, Optional

from . import gpu_memory_pool as pool

#: The boot-image tools this program knows how to drive, first found wins, each
#: with its fixed arguments. Debian and Ubuntu - the systems Vaelor's installer
#: supports - ship the first.
BOOT_IMAGE_TOOLS = (
    ("/usr/sbin/update-initramfs", ("-u",)),
)
#: One rebuild's ceiling. A change may run two (the change, then the roll-back),
#: so the unit's and the callers' ceilings below are built from twice this.
REBUILD_TIMEOUT_SECONDS = 420

VENV_PYTHON = "/opt/vaelor/venv/bin/python"
MODULE = "vaelor.gpu_memory_pool_apply"
#: The transient unit the bridge starts. ``--collect`` lets it run again after
#: a failure; ``--pipe`` brings this program's one line of JSON back. Python
#: runs isolated (``-I``): no ``PYTHON*`` variable and no directory on the
#: path but the environment's own decides what a root program imports.
APPLY_UNIT = "vaelor-gpu-memory-pool"
UNIT_RUNTIME_SECONDS = 2 * REBUILD_TIMEOUT_SECONDS + 60
APPLY_COMMAND = (
    "/usr/bin/systemd-run", "--unit=" + APPLY_UNIT, "--collect", "--wait",
    "--pipe", "--quiet", "--property=RuntimeMaxSec={}".format(UNIT_RUNTIME_SECONDS),
    VENV_PYTHON, "-I", "-m", MODULE,
)
APPLY_TIMEOUT_SECONDS = UNIT_RUNTIME_SECONDS + 120

#: The unit whose journal holds the detail of a failed change: the bridge's,
#: which logs this program's error output (the transient unit's own journal is
#: empty, because its output is piped to the bridge).
DETAIL_UNIT = "vaelor-hardware-bridge"
#: How much of the program's error output the bridge writes to its journal.
STDERR_TAIL_BYTES = 2000

ACTION_SET = "set"
ACTION_REVERT = "revert"

#: What a caller is told when the machine, not the request, was the problem.
_MACHINE_REFUSAL = (
    "The GPU memory pool setting could not be changed on this machine. Run "
    "'journalctl -u " + DETAIL_UNIT + "' there to see why."
)
#: How a person finishes a roll-back this program could not confirm.
_FINISH_BY_HAND = (
    "Run 'sudo update-initramfs -u' on this machine, then Refresh the machine "
    "settings to see what it holds."
)
_AS_IT_WAS = (
    "The boot image could not be rebuilt with the new setting. The previous "
    "setting was put back and the boot image rebuilt with it, so this machine "
    "is as it was."
)
_NOT_CONFIRMED = (
    "The boot image could not be rebuilt. The previous setting file was put "
    "back, but the boot image could not be rebuilt with it either, so it "
    "could not be confirmed which size the next restart will use. "
    + _FINISH_BY_HAND
)
_STOPPED = (
    "The change was stopped before the boot image finished rebuilding. The "
    "previous setting file was put back, but it could not be confirmed which "
    "size the next restart will use. " + _FINISH_BY_HAND
)
_NOT_PUT_BACK = (
    "The boot image could not be rebuilt and the previous setting could not "
    "be put back: the settings file on this machine {}. " + _FINISH_BY_HAND
)


class _Stopped(BaseException):
    """Raised by the ``SIGTERM`` handler; a BaseException so nothing swallows it."""


def _read_config() -> Optional[str]:
    """Vaelor's file as text; ``None`` when ABSENT. Unreadable raises ``OSError``.

    The two are never merged: a file that could not be read must not be
    "restored" by deleting it (review S6).
    """
    try:
        with open(pool.CONFIG_PATH, encoding="utf-8", errors="replace") as handle:
            return handle.read(65536)
    except FileNotFoundError:
        return None


def _write_config(text: str) -> None:
    """Replace Vaelor's file in one step, never through a link."""
    directory, name = os.path.split(pool.CONFIG_PATH)
    flags = os.O_RDONLY | getattr(os, "O_DIRECTORY", 0) | getattr(os, "O_NOFOLLOW", 0)
    dir_fd = os.open(directory, flags)
    temporary = "." + name + ".new"
    try:
        try:
            os.unlink(temporary, dir_fd=dir_fd)
        except FileNotFoundError:
            pass
        fd = os.open(
            temporary,
            os.O_WRONLY | os.O_CREAT | os.O_EXCL | getattr(os, "O_NOFOLLOW", 0),
            0o644, dir_fd=dir_fd,
        )
        with os.fdopen(fd, "w", encoding="utf-8") as handle:
            handle.write(text)
            handle.flush()
            os.fsync(handle.fileno())
        os.chmod(temporary, 0o644, dir_fd=dir_fd)
        os.rename(temporary, name, src_dir_fd=dir_fd, dst_dir_fd=dir_fd)
    finally:
        os.close(dir_fd)


def _remove_config() -> None:
    try:
        os.unlink(pool.CONFIG_PATH)
    except FileNotFoundError:
        pass


def _boot_image_tool() -> List[str]:
    for path, arguments in BOOT_IMAGE_TOOLS:
        if os.path.isfile(path) and os.access(path, os.X_OK):
            return [path, *arguments]
    return []


def _run(argv: List[str]) -> int:
    return subprocess.run(
        argv, check=False, timeout=REBUILD_TIMEOUT_SECONDS,
        stdout=sys.stderr, stderr=sys.stderr,
    ).returncode


def _local_gpu() -> Dict[str, Any]:
    from .cluster_capacity import gpu_facts_from_accelerators
    from .platforms import select_hardware_platform

    return gpu_facts_from_accelerators(select_hardware_platform().accelerators())


@dataclass(frozen=True)
class Host:
    """Everything this program touches on a machine, so a test can stand in for it."""

    read_facts: Callable[[], Dict[str, Any]] = pool.read_local_facts
    gpu: Callable[[], Dict[str, Any]] = _local_gpu
    read_config: Callable[[], Optional[str]] = _read_config
    write_config: Callable[[str], None] = _write_config
    remove_config: Callable[[], None] = _remove_config
    boot_image_tool: Callable[[], List[str]] = _boot_image_tool
    run: Callable[[List[str]], int] = _run


HOST = Host()


def local_status(host: Host = HOST) -> Dict[str, Any]:
    """This machine's pool, read where this process runs."""
    return pool.pool_status(host.read_facts(), host.gpu())


def _put(host: Host, text: Optional[str]) -> None:
    """Make Vaelor's file say ``text``; ``None`` removes it."""
    if text is None:
        host.remove_config()
    else:
        host.write_config(text)


def _rebuild(host: Host) -> bool:
    """Run the boot-image tool once; ``True`` only for a clean finish."""
    try:
        return host.run(host.boot_image_tool()) == 0
    except (OSError, subprocess.SubprocessError) as error:
        print("{}: rebuild: {}: {}".format(MODULE, type(error).__name__, error), file=sys.stderr)
        return False


def _put_back(host: Host, previous: Optional[str], now_holds: str) -> None:
    """Restore ``previous`` and prove it; refuse in words if it did not land."""
    try:
        _put(host, previous)
        landed = host.read_config() == previous
    except OSError as error:
        print("{}: put back: {}: {}".format(MODULE, type(error).__name__, error), file=sys.stderr)
        landed = False
    if not landed:
        raise RuntimeError(_NOT_PUT_BACK.format(now_holds))


def _change(host: Host, wanted: Optional[str], now_holds: str) -> Dict[str, Any]:
    """Put ``wanted`` in place and rebuild; roll back, checked, on any failure."""
    previous = host.read_config()
    try:
        _put(host, wanted)
        if _rebuild(host):
            return local_status(host)
        _put_back(host, previous, now_holds)
        raise RuntimeError(_AS_IT_WAS if _rebuild(host) else _NOT_CONFIRMED)
    except _Stopped:
        # Stopped anywhere after the first write - during the rebuild, or
        # during the roll-back itself. There is no time for another rebuild:
        # put the file back and say what is not known.
        signal.signal(signal.SIGTERM, signal.SIG_IGN)
        _put_back(host, previous, now_holds)
        raise RuntimeError(_STOPPED) from None


def _require_tool(host: Host) -> None:
    if not host.boot_image_tool():
        raise RuntimeError(
            "This machine has no boot-image tool Vaelor knows how to run "
            "(update-initramfs), so the new size could not be made to count "
            "at the next restart. Nothing was written."
        )


def apply_size(size_gib: Any, host: Host = HOST) -> Dict[str, Any]:
    """Set the pool to ``size_gib`` GiB from the next restart; answers the new status."""
    pages = pool.require_size_gib(size_gib, local_status(host))
    _require_tool(host)
    return _change(host, pool.render_config(pages), pool.file_now_holds(size_gib))


def revert(host: Host = HOST) -> Dict[str, Any]:
    """Remove Vaelor's file; the kernel's own size returns at the next restart."""
    pool.require_revert(local_status(host))
    _require_tool(host)
    return _change(host, None, pool.file_now_holds(None))


def _whole_number(text: str) -> int:
    """One to twelve ASCII digits, and nothing else, as an int (review S5)."""
    if re.fullmatch(r"[0-9]{1,12}", text) is None:
        raise ValueError(pool.WHOLE_NUMBER_REQUIRED)
    return int(text)


def _on_stop(_signum: int, _frame: Any) -> None:
    raise _Stopped()


def main(argv: Optional[List[str]] = None, host: Host = HOST) -> int:
    """``set <GiB>`` or ``revert``: one line of JSON out, 0 done, 3 refused, 4 machine."""
    arguments = list(sys.argv[1:] if argv is None else argv)
    if argv is None:
        signal.signal(signal.SIGTERM, _on_stop)
    try:
        if arguments[:1] == [ACTION_SET] and len(arguments) == 2:
            result = apply_size(_whole_number(arguments[1]), host)
        elif arguments == [ACTION_REVERT]:
            result = revert(host)
        else:
            raise ValueError("Use: set <whole GiB>, or revert.")
    except (RuntimeError, ValueError) as error:
        print(json.dumps({"ok": False, "error": str(error)}))
        return 3
    except OSError as error:
        # The detail - a path, an errno - is for the journal, not the caller.
        print("{}: {}: {}".format(MODULE, type(error).__name__, error), file=sys.stderr)
        print(json.dumps({"ok": False, "error": _MACHINE_REFUSAL}))
        return 4
    except _Stopped:
        # Stopped before anything was written.
        print(json.dumps({"ok": False, "error": (
            "The change was stopped before anything was written. "
            "Nothing was written."
        )}))
        return 3
    print(json.dumps({"ok": True, "status": result}))
    return 0


def _capture(argv: List[str]) -> "subprocess.CompletedProcess[str]":
    return subprocess.run(
        argv, check=False, capture_output=True, text=True,
        timeout=APPLY_TIMEOUT_SECONDS,
    )


def _log_detail(text: Any) -> None:
    """Write a bounded tail of the program's error output to this process's journal."""
    tail = str(text or "")[-STDERR_TAIL_BYTES:].strip()
    if tail:
        print("{}: {}".format(APPLY_UNIT, tail), file=sys.stderr, flush=True)


def run_apply_unit(
    action: str, size_gib: Optional[int] = None,
    capture: Callable[[List[str]], Any] = _capture,
    log: Callable[[Any], None] = _log_detail,
) -> Dict[str, Any]:
    """Run this program as a transient unit and answer its status, or raise.

    The caller (the bridge) has already validated ``size_gib`` as a whole
    number inside this machine's bounds; it is formatted here as digits only,
    and the program validates it again where it runs. Whatever the program
    wrote on its error output is logged by the CALLER's process, because the
    refusal points the owner at this process's journal.
    """
    if action == ACTION_SET:
        if type(size_gib) is not int:
            raise ValueError(pool.WHOLE_NUMBER_REQUIRED)
        tail = [ACTION_SET, str(size_gib)]
    elif action == ACTION_REVERT:
        tail = [ACTION_REVERT]
    else:
        raise ValueError("Unsupported GPU memory pool action.")
    try:
        completed = capture([*APPLY_COMMAND, *tail])
    except (OSError, subprocess.SubprocessError) as error:
        log("{}: {}".format(type(error).__name__, error))
        raise RuntimeError(_MACHINE_REFUSAL) from error
    answer: Any = None
    for line in reversed(str(completed.stdout or "").splitlines()):
        try:
            answer = json.loads(line)
        except ValueError:
            continue
        break
    if isinstance(answer, dict) and answer.get("ok") is True and isinstance(answer.get("status"), dict):
        return answer["status"]
    log(getattr(completed, "stderr", ""))
    if isinstance(answer, dict) and answer.get("error"):
        raise RuntimeError(str(answer["error"]))
    raise RuntimeError(_MACHINE_REFUSAL)


if __name__ == "__main__":
    sys.exit(main())
