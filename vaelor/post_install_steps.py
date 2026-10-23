"""The root build steps every install of a release runs before its models are reloaded (VD-217).

An update must leave a box in the state the installer leaves it in (VD-217).
The installer builds four things for the 30B as root after its health gate -
the vLLM attention change fetched from GitHub and built (VD-215,
`vllm_overlay_build`), Vaelor's HIP expert kernels (VD-216,
`vllm_moe_hip_build`), the 4-bit guesser (VD-216 item 2,
`vllm_guesser_build`) - and removes what the release retired
(`vllm_overlay_prune`). The console's Update Vaelor ran none of them, so a box
updated from the console served the 30B without its guesser (the drafter needs
the fetched attention file) until the owner re-ran the installer.

**One list, two callers, no new door** (LESSONS 6: two hand-kept lists of the
same steps drift). :data:`STEPS` is the list. It runs from
`deployment_refresh.main` - ``vaelor-refresh-models --apply`` - BEFORE the
refresh queues any reload, and only as root. That command is the one both
paths already run after their health gate: the installer by name, and the
console update's root broker (`appliance_upgrade.REFRESH_COMMAND`, a fixed
argv). Riding on it is what makes the very first console update from an
older release work: the broker that applies an update is the OLD release's
code, still in memory, and the only thing it runs from the NEW package after
the restart is that command. So the root boundary gains no socket, no verb,
no argument and no input a client chooses (LESSONS 18): the broker's argv is
unchanged, each step is ``<this interpreter> -I -m vaelor.<module> <verb>``
from :data:`STEPS` alone, and ``-I`` keeps the caller's environment and working
folder off the step's import path, so it runs the package the update just
verified and installed into the root-owned, ``go-w`` venv, nothing else.

**Bounded so the reload still happens** (:data:`BUDGET_SECONDS`). The broker
of every release so far gives that command 600 s and kills it on overrun
(`appliance_upgrade._default_refresh`), which would lose the reload. So the
steps share a budget well under that; each step gets the smaller of its own
bound and what is left, runs in its own process group, and is killed with it
when it overruns. A step that builds in a container is told its share
(``--within``, :func:`within_seconds`) and keeps every wait and build inside
it, so its own timeout path - which removes its build container by name -
fires before the outer kill; and after any outer kill the build containers
that step's own process group named are removed, no other's (R101)
(R98: a killed ``docker`` client leaves its container running). Every step is non-fatal on its own terms (each module's
``main`` prints and returns 0), and every artifact is checked against its pin
again at each use, so a step that did not finish leaves a box that serves with
the honest fallback - no guesser, no patched file, vLLM's own expert kernels -
and says so where the owner reads it: the Load's note and the console's
"speed fix missing" line. The root bridge also builds each one at the first
model start that needs it. Each outcome is printed and written to the journal
(the broker keeps no command output).

**Order** (`tests/test_vd217_update_parity.py`): the attention change is
fetched before it is built; the builds run before the prune, so the current
versions are in place before anything retired is removed; and all of it runs
before the reload, so the reload serves what was built rather than starting a
background build of its own.
"""

from __future__ import annotations

import os
import signal
import subprocess
import sys
import time
from typing import Any, Callable, List, NamedTuple, Optional, Sequence


class Step(NamedTuple):
    """One root build step: a module of this package, its one verb, its own bound."""

    module: str
    verb: str
    seconds: int
    label: str
    #: Whether the step takes :data:`WITHIN_FLAG` (it builds in a container).
    bounded: bool = False
    #: What the step's own out-of-time line says instead of the builds'
    #: OUT_OF_TIME (``{seconds}`` is its share), when that one would be untrue.
    late: str = ""


#: THE list (module docstring). The installer has no copy of it: it runs
#: ``vaelor-refresh-models --apply``, which runs this, then queues the reloads.
#: Each bound is the step's own worst case: a fetch's deadline; reading the
#: image's file; a lock wait plus a build for the two compilers; the prune's.
STEPS = (
    # First, because it takes seconds and the builds below can use the whole
    # budget: an update that ran out of time must still turn InfluxDB's
    # request log off (`influxdb_config`, VD-133 1.5.5 #13).
    # 150 s: at least `influxdb_config.WORST_CASE_SECONDS` (a test holds it).
    Step("influxdb_config", "apply", 150,
         "keep InfluxDB on loopback with its request and query logs off (VD-133)",
         late=("not finished within {seconds} s, so it was stopped; InfluxDB may keep "
               "its old settings, request log included, until the next update or "
               "install applies them")),
    # The release scripts Remove Vaelor and repair run. Here as well as in the
    # update broker because the broker that applies the first update to this
    # release is the previous release's, which never refreshes them
    # (`release_snapshot`, VD-217).
    Step("release_snapshot", "refresh", 30,
         "refresh the release scripts Remove Vaelor runs (VD-217)"),
    # VD-161 amended (owner, 2026-10-20): a machine with no saved GPU memory
    # pool size gets the recommended one, on install AND update, so both paths
    # end in the same state (VD-217). Seconds when a size is already saved;
    # the first time, one boot-image rebuild (6.5 s measured on the Z2,
    # 2026-10-20). Of its 300 s, 45 s is kept for clean-up (--within 255) and
    # 10 s for itself, so its rebuilds share 245 s: the change gets all that
    # is left, a roll-back the rest. Before the builds, which the first model
    # start builds anyway if time runs out.
    Step("gpu_memory_pool_apply", "recommend", 300,
         "save the recommended GPU memory pool size when none is saved (VD-161)", True,
         late=("not finished within {seconds} s, so it was stopped; whether the size was "
               "saved is not known - see Cluster > Setup > Machine settings")),
    Step("vllm_overlay_build", "fetch", 150, "fetch the vLLM attention change (VD-215)"),
    Step("vllm_overlay_build", "build", 90, "build the patched attention file (VD-215)", True),
    Step("vllm_moe_hip_build", "build", 390, "build Vaelor's HIP expert kernels (VD-216)", True),
    Step("vllm_guesser_build", "build", 1230, "build the 4-bit guesser (VD-216)", True),
    Step("vllm_overlay_prune", "apply", 60, "remove retired kernel versions (VD-216)"),
    # Starts a background move and returns: the NPU package is gigabytes, more
    # than this budget, and the reload must not wait for it (`npu_runtime_switch`).
    Step("npu_runtime_switch", "start", 30,
         "move the NPU Assistant to its pinned FastFlowLM and model (VD-219)"),
)

#: What all the steps together may take. Under the 600 s every broker so far
#: gives ``vaelor-refresh-models`` (`appliance_upgrade.REFRESH_TIMEOUT_SECONDS`)
#: with room for the refresh itself, which only reads and queues.
BUDGET_SECONDS = 450

#: A step started with less than this left is not started: it could not finish.
MIN_STEP_SECONDS = 20

#: The flag a bounded step takes its share of the budget by, in whole seconds.
WITHIN_FLAG = "--within"

#: What a step keeps back from its share for its own clean-up after a timeout,
#: so that clean-up runs before the outer kill (R98).
CLEANUP_SECONDS = 45

#: What a build container's removal is given, whatever is left of the share.
REMOVE_SECONDS = 30

#: How the journal names these lines.
JOURNAL_TAG = "vaelor-post-install"

OUT_OF_TIME = (
    "not finished within {seconds} s, so it was stopped; this box serves without "
    "it until it is built (the first model start that needs it builds it)"
)
NOT_STARTED = (
    "not started: the time this update allows for its builds ran out; the first "
    "model start that needs it builds it"
)


def step_argv(step: Step, python: str, seconds: Optional[float] = None) -> List[str]:
    """The fixed command one step runs as: this interpreter, isolated, the module, its verb.

    A bounded step is also given its share, less :data:`CLEANUP_SECONDS`.
    """
    argv = [python, "-I", "-m", "vaelor." + step.module, step.verb]
    if step.bounded and seconds is not None:
        argv += [WITHIN_FLAG, str(max(1, int(seconds) - CLEANUP_SECONDS))]
    return argv


def within_seconds(argv: Sequence[str], verbs: Sequence[str]) -> Optional[float]:
    """A step module's own reading of its argv: ``<verb> [--within N]``.

    The seconds it may take (``inf`` with no flag: the installer's own
    bounds), or ``None`` for anything else, which the module refuses.
    """
    rest = list(argv[1:])
    if not rest or rest[0] not in verbs:
        return None
    if len(rest) == 1:
        return float("inf")
    if len(rest) == 3 and rest[1] == WITHIN_FLAG and rest[2].isascii() and rest[2].isdigit():
        return float(rest[2])
    return None


def step_timeout(argv: Sequence[str], deadline: float, bound: float,
                 clock: Callable[[], float] = time.monotonic) -> float:
    """A step's timeout for one command: its own bound, cut to the deadline; a removal its own."""
    if list(argv[:3]) == ["docker", "rm", "-f"]:
        return float(REMOVE_SECONDS)
    return max(1.0, min(float(bound), deadline - clock()))


def group_tag(group: Optional[int] = None) -> str:
    """Six hex digits naming a process group, for build container names (R101).

    A step runs in its own session (:func:`_run_one`), so its process group is
    its own process id, and no other live group has it: the root bridge's
    background builds and a worker's builds name other groups.
    """
    if group is None:
        group = os.getpgrp() if hasattr(os, "getpgrp") else os.getpid()
    return "{:06x}".format(int(group) & 0xFFFFFF)


def build_prefixes(group: int) -> List[str]:
    """The name prefixes of one process group's build containers: nothing else is ever removed."""
    from .vllm_guesser_build import CONTAINER_PREFIX as GUESSER
    from .vllm_moe_hip_build import CONTAINER_PREFIX as HIP

    return [prefix + group_tag(group) for prefix in (GUESSER, HIP)]


def build_containers(group: int, run: Callable[..., Any] = subprocess.run) -> List[str]:
    """Every container, running or not, that ``group``'s builds named (:func:`build_prefixes`)."""
    try:
        listed = run(["docker", "ps", "-a", "--format", "{{.Names}}"], capture_output=True,
                     text=True, timeout=REMOVE_SECONDS, check=False)
    except (OSError, subprocess.SubprocessError):
        return []
    prefixes = tuple(build_prefixes(group))
    return [name for name in str(getattr(listed, "stdout", "") or "").split()
            if name.startswith(prefixes)]


def remove_group_containers(group: int,
                            run: Callable[..., Any] = subprocess.run) -> List[str]:
    """Remove the build containers a stopped step's group started (R98, R101); their names."""
    removed = []
    for name in build_containers(group, run):
        try:
            run(["docker", "rm", "-f", name], capture_output=True, text=True,
                timeout=REMOVE_SECONDS, check=False)
            removed.append(name)
        except (OSError, subprocess.SubprocessError):
            pass
    return removed


def _journal(line: str) -> None:
    try:
        import syslog

        syslog.openlog(JOURNAL_TAG)
        syslog.syslog(syslog.LOG_INFO, line)
    except Exception:  # noqa: BLE001 - the printed line stands without it
        pass


#: The group and umask the installer's root shell runs the steps with. The
#: update's broker runs ``Group=vaelor-jobs UMask=0007`` (`appliance_upgrade`),
#: and a step it starts would otherwise leave every built file group
#: ``vaelor-jobs`` where the installer leaves ``root`` (measured on the Z2,
#: 2026-10-16: same bytes and modes, other group). So a root step runs as the
#: installer's would, and both paths leave the same folders (VD-217).
INSTALLER_GROUP = 0
INSTALLER_UMASK = 0o022


def _as_installer() -> dict:
    """Popen's arguments for :data:`INSTALLER_GROUP` and :data:`INSTALLER_UMASK`, as root only."""
    if os.name != "posix" or not is_root():
        return {}
    return {"group": INSTALLER_GROUP, "extra_groups": [], "umask": INSTALLER_UMASK}


def _run_one(argv: List[str], seconds: float) -> str:
    """Run one step in its own process group; its last line, or why it did not finish."""
    try:
        process = subprocess.Popen(
            argv, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True,
            start_new_session=True, close_fds=True, stdin=subprocess.DEVNULL,
            **_as_installer(),
        )
    except OSError as error:
        return "could not start ({})".format(str(error)[:160] or type(error).__name__)
    try:
        output, _ = process.communicate(timeout=seconds)
    except subprocess.TimeoutExpired:
        try:
            os.killpg(process.pid, signal.SIGKILL)
        except (OSError, AttributeError):
            process.kill()
        process.communicate()
        removed = remove_group_containers(process.pid) if is_root() else []
        return OUT_OF_TIME.format(seconds=int(seconds)) + (
            "; its build container was removed" if removed else "")
    lines = [line for line in (output or "").splitlines() if line.strip()]
    if process.returncode != 0:
        return "ended with status {}: {}".format(
            process.returncode, (lines[-1] if lines else "no output")[:300])
    return " / ".join(lines)[:600] or "done"


#: How a step ended, for the update panel (R104): built or current, not started
#: for want of time, or held back with the step's own reason.
STEP_DONE, STEP_SKIPPED, STEP_HELD_BACK = "done", "skipped", "held-back"


def step_status(step: Step, outcome: str) -> str:
    """:data:`STEP_DONE`, :data:`STEP_SKIPPED` or :data:`STEP_HELD_BACK` for one step's outcome.

    A fetch or build module prints ``<what> (<key>): done`` per item when it
    is current and its reason otherwise, so anything but ``done`` on every
    line is held back; the prune says what it removed, and only a runner's
    own failure holds it back.
    """
    if outcome == NOT_STARTED:
        return STEP_SKIPPED
    if outcome.startswith(("not finished within", "ended with status", "could not start")):
        return STEP_HELD_BACK
    if step.verb not in ("fetch", "build"):
        return STEP_DONE
    parts = [part.strip() for part in outcome.split(" / ") if part.strip()]
    return STEP_DONE if parts and all(
        part == "done" or part.endswith(": done") for part in parts) else STEP_HELD_BACK


def run_steps(
    *, python: Optional[str] = None, budget: float = BUDGET_SECONDS,
    run: Callable[[List[str], float], str] = _run_one,
    clock: Callable[[], float] = time.monotonic,
    say: Callable[[str], None] = print, steps: Sequence[Step] = STEPS,
    record: Optional[Callable[[dict], None]] = None,
) -> List[str]:
    """Run :data:`STEPS` in order within ``budget``; one line per step. Never raises.

    ``record`` is told each step's label, status (:func:`step_status`) and
    outcome as it ends (R104: the update panel shows them as they happen).
    """
    python = python or sys.executable
    deadline = clock() + float(budget)
    lines: List[str] = []
    for step in steps:
        left = deadline - clock()
        if left < MIN_STEP_SECONDS:
            outcome = NOT_STARTED
        else:
            try:
                share = min(float(step.seconds), left)
                outcome = run(step_argv(step, python, share), share)
                if step.late and outcome.startswith(OUT_OF_TIME.split("{")[0]):
                    outcome = step.late.format(seconds=int(share))
            except Exception as error:  # noqa: BLE001 - a step never fails the update
                outcome = str(error)[:300] or type(error).__name__
        line = "Vaelor post-install step, {}: {}".format(step.label, outcome)
        lines.append(line)
        say(line)
        _journal(line)
        if record is not None:
            try:
                record({"label": step.label, "status": step_status(step, outcome),
                        "outcome": outcome})
            except Exception:  # noqa: BLE001 - the printed line stands without it
                pass
    return lines


def is_root() -> bool:
    """Whether this process may run the steps: they write root-only folders."""
    geteuid = getattr(os, "geteuid", None)
    return geteuid is not None and geteuid() == 0


def main(argv: Sequence[str]) -> int:
    """``run``, as root: the steps alone, for a test against a bench box. Never fails."""
    if list(argv[1:]) != ["run"]:
        sys.stderr.write("usage: python -I -m vaelor.post_install_steps run\n")
        return 2
    if not is_root():
        sys.stderr.write("The post-install steps build root-only folders; run them as root.\n")
        return 2
    run_steps()
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv))
