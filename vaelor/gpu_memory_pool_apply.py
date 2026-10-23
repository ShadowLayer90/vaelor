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

import contextlib
import io
import json
import os
import re
import signal
import subprocess
import sys
import time
from dataclasses import dataclass, replace
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
#: The install and update step (`post_install_steps`): save the recommended
#: size when none is saved (VD-161 amended, owner 2026-10-20). Root only, run
#: by the installer's root shell or the update's root broker, never the bridge.
ACTION_RECOMMEND = "recommend"
#: What the step's share keeps back for itself (reading the machine, the
#: result line) after the rebuilds, so it ends before the outer kill.
RECOMMEND_MARGIN_SECONDS = 10
#: A rebuild given less than this is not started: it could not finish.
MIN_REBUILD_SECONDS = 1.0

#: What a caller is told when the machine, not the request, was the problem.
_SEE_WHY = "{} Run 'journalctl {}' there to see why."
_MACHINE_REFUSAL = _SEE_WHY.format(
    "The GPU memory pool setting could not be changed on this machine.", "-u " + DETAIL_UNIT)
#: The install step's own words for the same, pointing at where ITS detail goes.
_RECOMMEND_REFUSAL = _SEE_WHY.format(
    "This machine's GPU memory pool could not be read or changed.", "-t " + APPLY_UNIT)
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


def _kill_group(process: "subprocess.Popen[Any]") -> None:
    """Kill a timed-out rebuild and every child it started (mkinitramfs runs many)."""
    try:
        os.killpg(process.pid, signal.SIGKILL)  # type: ignore[attr-defined]
    except (OSError, AttributeError):
        process.kill()


def _run(argv: List[str], timeout: float = REBUILD_TIMEOUT_SECONDS,
         sink: Optional[Callable[[str], None]] = None) -> int:
    """Run the boot-image tool in its own process group; its exit status.

    Review (VD-161 amended, H1/2): a timeout kills the whole group, so no
    ``mkinitramfs`` child can finish a boot image after the roll-back began,
    or hold a caller's output open. With ``sink`` the tool's output is
    captured and handed to it (the install step sends it to the journal, so
    it never crowds out the step's one result line); without, it goes to this
    process's error output, as the bridge has always logged it. Raises
    ``subprocess.TimeoutExpired`` after the kill.
    """
    capture = sink is not None
    process = subprocess.Popen(
        argv, stdin=subprocess.DEVNULL, start_new_session=True, close_fds=True,
        stdout=subprocess.PIPE if capture else sys.stderr,
        stderr=subprocess.STDOUT if capture else sys.stderr,
        text=capture, errors="replace" if capture else None,
    )
    try:
        output, _ = process.communicate(timeout=timeout)
    except subprocess.TimeoutExpired:
        _kill_group(process)
        output, _ = process.communicate()
        if capture:
            sink(output or "")
        raise
    if capture:
        sink(output or "")
    return process.returncode


def _local_gpu() -> Dict[str, Any]:
    from .cluster_capacity import gpu_facts_from_accelerators
    from .platforms import select_hardware_platform

    return gpu_facts_from_accelerators(select_hardware_platform().accelerators())


#: Why the update step saves nothing after the owner's own Remove setting.
OWNER_REMOVED = (
    "Nothing saved: the last change to this machine's GPU memory pool was your own "
    "Remove setting, so it stays on the kernel's own size."
)
HISTORY_UNREAD = (
    "Nothing saved: Vaelor's job history could not be read, so whether you removed "
    "the size yourself is not known."
)


def _owner_removed_last() -> bool:
    """Whether the owner's Remove setting is the last pool change finished here.

    VD-161 amended (Adopted, coordinator 2026-10-20): an update must not undo
    it. Read from the jobs the console ran (`gpu_memory_pool_nodes.last_pool_change`,
    the history the card's own restart rule reads). A box with no job store yet
    is a fresh install with no history: never created here, because this runs
    as root and the console's own services own that file. Raises when the
    store exists and cannot be read.
    """
    from .cluster_placement import CONTROLLER_PLACEMENT_ID
    from .gpu_memory_pool_nodes import last_pool_change
    from .jobs import JobStore

    path = JobStore().database_path
    if not os.path.exists(path):
        return False
    job = last_pool_change(_read_only_jobs(path), CONTROLLER_PLACEMENT_ID)
    return job is not None and job["payload"].get("action") == ACTION_REVERT


def _read_only_jobs(path: str) -> Any:
    """The job store at ``path``, opened read-only (review 6).

    `JobStore` creates the file, sets its mode and applies its schema on every
    connection; this root step must never write the console's own database,
    so only the connection differs (``mode=ro``, no schema) and every query
    and row reader is `JobStore`'s own, so the two cannot read a job
    differently.
    """
    import sqlite3
    from pathlib import Path

    from .jobs import JobStore

    class ReadOnlyJobs(JobStore):
        def _connect(self) -> sqlite3.Connection:
            uri = Path(self.database_path).resolve().as_uri() + "?mode=ro"
            connection = sqlite3.connect(uri, uri=True, timeout=10)
            connection.row_factory = sqlite3.Row
            return connection

    return ReadOnlyJobs(path)


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
    owner_removed_last: Callable[[], bool] = _owner_removed_last


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


def save_recommended(host: Host = HOST) -> str:
    """Save the recommended size when none is saved; one sentence saying what happened.

    VD-161 amended (owner, 2026-10-20): the controller's install and every
    update run this (`post_install_steps`), so a fresh machine's card never
    opens on a pool that shrinks at the next restart. The decision is
    `gpu_memory_pool.recommended_save`, the one a worker's join takes too;
    the write is :func:`apply_size`, the console's own path, with its bounds,
    boot-image rebuild and checked roll-back. Never restarts the machine, and
    never re-saves after the owner's own Remove setting (the last finished
    pool change) - nor when that history cannot be read. Raises as
    :func:`apply_size` does when the size could not be saved.
    """
    size, why = pool.recommended_save(local_status(host))
    if size is None:
        return why
    try:
        if host.owner_removed_last():
            return OWNER_REMOVED
    except Exception as error:  # noqa: BLE001 - unreadable saves nothing, and says so
        print("{}: job history: {}: {}".format(MODULE, type(error).__name__, error), file=sys.stderr)
        return HISTORY_UNREAD
    return pool.recommended_saved_message(size, apply_size(size, host))


def _journal(text: str) -> None:
    """A bounded tail of what the step did not print, to the journal under the unit's name."""
    tail = str(text or "")[-STDERR_TAIL_BYTES:].strip()
    if not tail:
        return
    try:
        import syslog

        syslog.openlog(APPLY_UNIT)
        for line in tail.splitlines():
            syslog.syslog(syslog.LOG_INFO, line)
    except Exception:  # noqa: BLE001 - the result line stands without it
        pass


def _deadline_runner(
    deadline: float, clock: Callable[[], float], sink: Callable[[str], None],
) -> Callable[[List[str]], int]:
    """A ``Host.run`` whose every rebuild gets ALL the time left before ``deadline``.

    Review (VD-161 amended, H1): halving the share made a slow machine's
    save fail by construction. The change's rebuild gets what is left; only
    if it fails does the roll-back's rebuild get the rest - and with none
    left it is not started, which ``_change`` reports as the file put back
    and the boot image unconfirmed, with the command to run.
    """
    def run(argv: List[str]) -> int:
        left = min(float(REBUILD_TIMEOUT_SECONDS), deadline - clock())
        if left < MIN_REBUILD_SECONDS:
            sink("{}: no time left to rebuild the boot image".format(MODULE))
            return 1
        return _run(argv, left, sink)
    return run


def _recommend(
    within: float, host: Host, clock: Callable[[], float] = time.monotonic,
    journal: Callable[[str], None] = _journal,
) -> int:
    """``recommend [--within N]``: exactly one line out, last; 0 saved or kept, 3 not saved.

    ``within`` is the step's share; less :data:`RECOMMEND_MARGIN_SECONDS` it
    bounds every rebuild together (:func:`_deadline_runner`). Measured on the
    Z2 (2026-10-20) one rebuild takes 6.5 s, so the 255 s a 300 s step is
    given is ample; a slower machine spends what it needs on the change, and
    the roll-back gets the rest. Everything else - the rebuild's output, the
    detail of a failure - goes to the journal, never to standard output, so
    the post-install step's 600-character outcome is this line (review 3).
    A failure never fails the install or the update: the line says it, and
    the card still offers the choice.
    """
    details = io.StringIO()
    deadline = clock() + (within - RECOMMEND_MARGIN_SECONDS if within != float("inf")
                          else 2 * REBUILD_TIMEOUT_SECONDS)
    host = replace(host, run=_deadline_runner(deadline, clock, details.write))
    code = 0
    with contextlib.redirect_stderr(details):
        try:
            line = save_recommended(host)
        except Exception as error:  # noqa: BLE001 - one honest line, the detail to the journal
            if not isinstance(error, (RuntimeError, ValueError)):
                print("{}: {}: {}".format(MODULE, type(error).__name__, error), file=sys.stderr)
                error = RuntimeError(_RECOMMEND_REFUSAL)
            line, code = "Not saved: {} Choose a size in {}.".format(error, pool.SETTING_PLACE), 3
    journal(details.getvalue())
    print(line)
    return code


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
    if arguments[:1] == [ACTION_RECOMMEND]:
        from .post_install_steps import within_seconds

        within = within_seconds([MODULE, *arguments], (ACTION_RECOMMEND,))
        if within is None:
            print("Use: recommend [--within <seconds>].")
            return 2
        try:
            return _recommend(within, host)
        except _Stopped:
            print("Not saved: stopped before the size was saved.")
            return 3
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
