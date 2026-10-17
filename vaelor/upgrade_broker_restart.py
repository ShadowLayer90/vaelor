"""Put the update broker onto the code an update installed, once it is idle (VD-217, R105).

**Why.** The update broker (`appliance_upgrade`, ``vaelor-appliance-upgrade``)
applies an update with the code it was started with. An update never restarts
it - stopping it would kill the apply - so after a console update from 1.5.1
the box ran 1.5.3 everywhere except the broker, which kept 1.5.1's handler in
memory until a reboot: no apply lock, no self-restart, and the next console
update unguarded again (kb R105, measured on the Z2 2026-10-16: the broker's
start time stayed before the install). A broker from 1.5.3 on restarts itself
after each apply (`upgrade_apply_lock.restart_self`); an older one cannot.

**How, and the root boundary (LESSONS 18).** ``vaelor-refresh-models --apply``
(`deployment_refresh`) is the one command every broker - the old one included -
runs from the NEW package, as root, after its health gate. As root it calls
:func:`schedule`, which hands systemd one fixed command
(:func:`watch_argv`: this interpreter, ``-I``, this module, ``watch``) as a
transient unit of its own, outside the broker's process group, so a restart of
the broker cannot kill it. No socket, verb, argument or file a client chooses
is added: the control plane, the bridge and every group-``vaelor`` service are
untouched, and the transient unit's argv carries nothing but fixed words.

**Never during an apply.** :func:`watch` restarts the broker only when all of
these hold, twice, :data:`QUIET_SECONDS` apart:

* the broker started before the installed package was written (its process
  start against the package's own ``RECORD``); otherwise it already runs the
  new code and nothing is done;
* the apply lock (`upgrade_apply_lock`) is free - and it is then HELD across
  the restart, so a new-code apply or refresh cannot begin in between;
* no update plan is staged (a plan staged is an apply about to start);
* no apply is in flight: every apply, an old broker's included, re-creates the
  root-only ``verified`` folder when it starts and writes the result file when
  it ends (`appliance_upgrade._verify_staged_wheel`, the ``finally`` of
  `appliance_upgrade.perform_upgrade_apply`), so a ``verified`` folder newer
  than the result is an apply that has not ended;
* the broker has no child process (pip, systemctl, the refresh and its builds
  all run as its children).

It gives up after :data:`WATCH_SECONDS`, leaving the broker as it was; the next
reboot or installer run starts it on the new code, as before this fix. Each
outcome is printed and written to the journal.
"""

from __future__ import annotations

import os
import subprocess
import sys
import time
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Sequence

#: The transient unit the watch runs as. A second schedule while one runs is
#: refused by systemd, which is the point: one watch at a time.
WATCH_UNIT = "vaelor-upgrade-broker-refresh"

#: How long the watch waits for the broker to fall idle before giving up.
#: Longer than any apply (`appliance_upgrade.REFRESH_TIMEOUT_SECONDS` plus the
#: health gate and the install), so the apply that scheduled it always ends first.
WATCH_SECONDS = 1200

#: How often the watch looks.
POLL_SECONDS = 2

#: How long every idle sign must hold before the restart.
QUIET_SECONDS = 6

#: What the restart itself may take.
RESTART_SECONDS = 90

#: How the journal names these lines.
JOURNAL_TAG = "vaelor-upgrade-broker"

CURRENT = "the update broker already runs the installed code"
NOT_RUNNING = "the update broker is not running; it starts on the installed code"
UNREAD = "the update broker's start or the install time could not be read ({}), so it was left as it was"
GAVE_UP = (
    "the update broker stayed busy for {} s, so it was left on its previous code; "
    "the next restart of the machine or run of the installer starts it on the new code"
)
APPLYING = "an update is being applied"
NOT_SCHEDULED = "Vaelor update broker refresh: could not be scheduled ({})"
RESTARTED = "the update broker was restarted onto the installed code"
NOT_RESTARTED = "the update broker could not be restarted ({}); it keeps its previous code until the next restart"


def watch_argv(python: str = sys.executable) -> List[str]:
    """The fixed command :func:`schedule` hands systemd: nothing a client chooses."""
    return ["systemd-run", "--unit", WATCH_UNIT, "--collect", "--no-block", "--quiet",
            "--property=RuntimeMaxSec={}".format(WATCH_SECONDS + 60),
            python, "-I", "-m", "vaelor.upgrade_broker_restart", "watch"]


def schedule(python: Optional[str] = None,
             run: Callable[..., Any] = subprocess.run) -> str:
    """Start :func:`watch` as its own transient unit; one line saying what happened. Never raises."""
    try:
        done = run(watch_argv(python or sys.executable), capture_output=True, text=True,
                   timeout=30, check=False, stdin=subprocess.DEVNULL)
    except (OSError, subprocess.SubprocessError) as error:
        return NOT_SCHEDULED.format(
            str(error)[:160] or type(error).__name__)
    if getattr(done, "returncode", 1) != 0:
        said = str(getattr(done, "stderr", "") or getattr(done, "stdout", "") or "").strip()
        return NOT_SCHEDULED.format(
            said[:160] or "systemd-run refused")
    return "Vaelor update broker refresh: scheduled for when the update broker is idle"


# --- What the watch reads ----------------------------------------------------------

def _show(run: Callable[..., Any], unit: str) -> Dict[str, str]:
    done = run(["systemctl", "show", unit, "-p", "MainPID", "-p", "ActiveState",
                "-p", "ControlGroup"], capture_output=True, text=True, timeout=15, check=False)
    pairs = (line.split("=", 1) for line in str(getattr(done, "stdout", "") or "").splitlines())
    return {key: value for key, value in (pair for pair in pairs if len(pair) == 2)}


def process_started_at(pid: int, proc: str = "/proc") -> float:
    """When ``pid`` started, in seconds since the epoch, from the kernel's own counters."""
    stat = Path(proc, str(pid), "stat").read_text(encoding="ascii")
    # Field 22 (start time, in clock ticks after boot) counts from after the
    # command name, which may hold spaces, so split after its closing bracket.
    fields = stat[stat.rindex(")") + 2:].split()
    ticks = int(fields[19])
    boot = next(int(line.split()[1]) for line in
                Path(proc, "stat").read_text(encoding="ascii").splitlines()
                if line.startswith("btime "))
    return boot + ticks / float(os.sysconf("SC_CLK_TCK"))


def broker_state(run: Callable[..., Any] = subprocess.run, unit: Optional[str] = None,
                 proc: str = "/proc", cgroups: str = "/sys/fs/cgroup") -> Dict[str, Any]:
    """The broker's process, when it started, and how many other processes its unit holds."""
    from .appliance_upgrade import UPGRADE_ORCHESTRATOR_EXCLUSIONS

    shown = _show(run, unit or UPGRADE_ORCHESTRATOR_EXCLUSIONS[0])
    pid = int(shown.get("MainPID", "0") or 0)
    if shown.get("ActiveState") != "active" or pid <= 0:
        return {"running": False}
    group = shown.get("ControlGroup", "")
    members = [int(word) for word in
               Path(cgroups + group, "cgroup.procs").read_text(encoding="ascii").split()]
    return {"running": True, "pid": pid, "started_at": process_started_at(pid, proc),
            "children": len([member for member in members if member != pid])}


def installed_at(distribution: Optional[str] = None) -> float:
    """When the installed package was written: its ``RECORD``, which every install rewrites."""
    from importlib import metadata

    from .appliance_upgrade import PACKAGE_NAME

    found = metadata.distribution(distribution or PACKAGE_NAME)
    for entry in found.files or ():
        if entry.name == "RECORD" and entry.parent.name.endswith(".dist-info"):
            return os.stat(str(entry.locate())).st_mtime
    raise FileNotFoundError("the installed package lists no RECORD")


def apply_in_flight(retained: Optional[Path] = None, result: Optional[Path] = None,
                    plan: Optional[Path] = None) -> Optional[str]:
    """Why an apply may be running or about to, or ``None`` (module docstring)."""
    from . import appliance_upgrade as upgrade

    retained = Path(retained or upgrade.RETAINED_DIR)
    result = Path(result or upgrade.RESULT_PATH)
    plan = Path(plan or upgrade.PLAN_PATH)
    if plan.exists():
        return "an update plan is staged"
    try:
        started = (retained / upgrade.VERIFIED_SUBDIR).stat().st_mtime
    except FileNotFoundError:
        return None
    try:
        ended = result.stat().st_mtime
    except FileNotFoundError:
        return APPLYING
    return APPLYING if started > ended else None


# --- The watch ----------------------------------------------------------------------

def idle_reason(state: Dict[str, Any], in_flight: Optional[str]) -> Optional[str]:
    """Why the broker is not idle now, or ``None`` when every idle sign holds."""
    if in_flight is not None:
        return in_flight
    if state.get("children"):
        return "the update broker is running a command"
    return None


def watch(*, run: Callable[..., Any] = subprocess.run,
          state: Optional[Callable[[], Dict[str, Any]]] = None,
          installed: Optional[Callable[[], float]] = None,
          in_flight: Optional[Callable[[], Optional[str]]] = None,
          lock_dir: Optional[Path] = None,
          clock: Callable[[], float] = time.monotonic,
          sleep: Callable[[float], None] = time.sleep,
          seconds: float = WATCH_SECONDS) -> str:
    """Restart the broker onto the installed code once it is idle; what was done. Never raises."""
    from . import upgrade_apply_lock as apply_lock
    from .appliance_upgrade import RETAINED_DIR, UPGRADE_ORCHESTRATOR_EXCLUSIONS

    state = state or (lambda: broker_state(run))
    installed = installed or installed_at
    in_flight = in_flight or apply_in_flight
    lock_dir = Path(lock_dir or RETAINED_DIR)
    deadline = clock() + float(seconds)
    quiet_since: Optional[float] = None
    while True:
        try:
            now_state = state()
            if not now_state.get("running"):
                return NOT_RUNNING
            if float(now_state["started_at"]) >= float(installed()):
                return CURRENT
        except (OSError, ValueError, KeyError, StopIteration, ImportError) as error:
            return UNREAD.format(str(error)[:160] or type(error).__name__)
        handle = apply_lock.acquire(lock_dir)
        try:
            busy = "an update holds the update lock" if handle is None else idle_reason(
                now_state, in_flight())
            if busy is not None:
                quiet_since = None
            elif quiet_since is None:
                quiet_since = clock()
            elif clock() - quiet_since >= QUIET_SECONDS:
                return _restart(run, UPGRADE_ORCHESTRATOR_EXCLUSIONS[0])
        finally:
            apply_lock.release(handle)
        if clock() >= deadline:
            return GAVE_UP.format(int(seconds))
        sleep(POLL_SECONDS)


def _restart(run: Callable[..., Any], unit: str) -> str:
    """Restart ``unit``; the caller holds the apply lock across it."""
    try:
        done = run(["systemctl", "restart", unit], capture_output=True, text=True,
                   timeout=RESTART_SECONDS, check=False)
    except (OSError, subprocess.SubprocessError) as error:
        return NOT_RESTARTED.format(str(error)[:160] or type(error).__name__)
    if getattr(done, "returncode", 1) != 0:
        return NOT_RESTARTED.format(
            str(getattr(done, "stderr", "") or "").strip()[:160] or "systemctl refused")
    return RESTARTED


def _journal(line: str) -> None:
    try:
        import syslog

        syslog.openlog(JOURNAL_TAG)
        syslog.syslog(syslog.LOG_INFO, line)
    except Exception:  # noqa: BLE001 - the printed line stands without it
        pass


def main(argv: Sequence[str]) -> int:
    """``watch``, as root: the transient unit :func:`schedule` starts. Never fails."""
    if list(argv[1:]) != ["watch"]:
        sys.stderr.write("usage: python -I -m vaelor.upgrade_broker_restart watch\n")
        return 2
    geteuid = getattr(os, "geteuid", None)
    if geteuid is None or geteuid() != 0:
        sys.stderr.write("Restarting the update broker needs root.\n")
        return 2
    line = "Vaelor update broker refresh: " + watch()
    print(line, flush=True)
    _journal(line)
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv))
