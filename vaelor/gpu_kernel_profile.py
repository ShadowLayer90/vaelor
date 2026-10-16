"""The GPU kernel profile: a time-limited serve with GPU tracing on (VD-214).

vLLM 0.27's containers start with rocprofiler-register switched off
(`vllm_images.VLLM_027_ROCM10`, VD-165), because with it on a ROCr thread holds
one CPU core at 100% for as long as the model serves. With it off, the torch
profiler's trace holds CPU events only, so the Performance tab's on-demand
profile cannot time GPU kernels (VD-213). The owner chose a profile mode that
restarts the controller's copy of the model with registration on, captures,
and restarts it normally - "meant for testing to time limit this to avoid
permanently pegging the CPU core" (VD-214).

**Which value turns registration on.** rocprofiler-register reads
``ROCPROFILER_REGISTER_ENABLED`` with a default of ``true`` (``get_env(...,
true)`` in ``rocprofiler_register_library_api_table``, ROCm/rocprofiler-register
amd-mainline, read 2026-10-12; the pinned image's own copy was not re-read).
So the tracing serve leaves the variable OUT - the exact state VD-165 measured
before its fix - rather than spelling a ``=1`` whose parsing was not read.

**How the tracing serve is made.** Never by a second render: the controller's
server unit is the one the root bridge rendered and installed
(`bridge_managed_units`). :func:`tracing_exec_start` takes that installed text,
proves it is the template's (`check_rendered_unit`), removes the one
``-e ROCPROFILER_REGISTER_ENABLED=0`` pair, and proves the result is still the
template's. It goes into a RUNTIME drop-in under ``/run/systemd/system`` - a
tmpfs, gone at the next boot - that overrides ``ExecStart``. No value a client
sends reaches it: the unit is named from the mode file, the text is root's.

**Why it cannot be left on (the owner's limit).** Before the drop-in is
written, a transient systemd timer is armed (:func:`arm_command`) that runs the
revert program (`gpu_kernel_profile_mode`) :data:`TRACING_DEADLINE_SECONDS`
later. It is owned by systemd, not by the console, the request or the bridge:
a closed browser, a dropped connection, a failed capture, a crashed or
restarted bridge all leave it armed. A reboot clears the drop-in with ``/run``.
If the timer cannot be armed, nothing is changed. The revert removes every
such drop-in, restarts the units it removed one from, and restarts any 0.27
server still running without the switch however it got there.

**The watch must not tear the cluster down meanwhile.** A replicated row with
no replica answering for `gpu_pool_replicas.REPLICA_DOWN_PASSES` passes (about
two minutes) is failed and torn down, and on a one-machine deployment each of
the two restarts is that long. A bounded hold (:func:`write_hold`) tells the
watch that this controller's copy is restarting on purpose; it is root-owned,
names the deployment, and is honoured only until its time - never more than
:data:`HOLD_SECONDS` ahead, so a clock step or a forged file cannot hold a
dead cluster up for longer (LESSONS 22). It says nothing about tracing.
"""

from __future__ import annotations

import json
import os
import stat
import tempfile
import time
from typing import Any, Callable, FrozenSet, List, Mapping, Optional

from .bridge_managed_units import check_rendered_unit
from .cluster_placement import CONTROLLER_PLACEMENT_ID
from .gpu_pool_balancer import replica_entries
from .gpu_pool_units import SERVER_ROLE, VERDICT_REPLICATED, record_mode, unit_role
from .runtime_paths import run_path
from .serving_profile_traces import TRACE_READ_BUDGET_SECONDS
from .state_root_layout import VENV_PYTHON
from .vllm_images import VLLM_027_ROCM10
from .vllm_serve_options import image_on_record

#: VD-165's switch, as the 0.27 image's fixed environment spells it.
REGISTRATION_SWITCH = next(
    env for env in VLLM_027_ROCM10.environment if env.startswith("ROCPROFILER_")
)

#: How long the restarted model may take to answer ``/health``. The Z2's 30B
#: restarts in about 1-2 minutes with a warm compile cache (owner, VD-214
#: brief; not re-measured in code); 300 s is 2.5 times the slow end. A cold
#: compile is longer and is not what this mode is for: past this the capture
#: is abandoned and the model restarted normally.
RESTART_HEALTH_WAIT_SECONDS = 300

#: Reading the traces once more after the poll: one read is cut at
#: `serving_profile_traces.TRACE_READ_BUDGET_SECONDS` (20 s) whatever the
#: traces hold - a real 507 MB capture read in 2.9 s on the Z2, a crafted one
#: within every size cap took 49.6 s before that budget (VD-216 review R61) -
#: plus 10 s for the parse and the reply.
TRACE_READ_SECONDS = int(TRACE_READ_BUDGET_SECONDS) + 10

#: The longest the existing profile takes once the model answers: the serving
#: PID (``docker inspect``, 10 s), the drafter's counters at both ends (2 x 5 s),
#: the CPU record and report (window 10 s + 15 s margin + 20 s report), two
#: ``amd-smi`` reads (2 x 15 s), the off-CPU trace (25 s), and the kernel
#: capture (model lookup, start, one completion, stop: 10 + 10 + 30 + 10 s,
#: then up to 16 s for a trace still being written) - 200 s at every bound
#: (`serving_profiler`) - and the bounded read of the traces.
CAPTURE_ALLOWANCE_SECONDS = 200 + TRACE_READ_SECONDS

#: Room for the bridge's own steps between them (unit read, timer, drop-in,
#: daemon-reload, the restart job being queued).
STEP_SLACK_SECONDS = 60

#: The owner's limit: tracing is on for at most this long, whatever happens to
#: the request that started it. The sum of the three above - 590 s.
TRACING_DEADLINE_SECONDS = (
    RESTART_HEALTH_WAIT_SECONDS + CAPTURE_ALLOWANCE_SECONDS + STEP_SLACK_SECONDS
)

#: The same limit in whole minutes, ROUNDED UP - the one figure every sentence
#: and the console state (review 2: seconds // 60 said 9 for 570 s).
TRACING_DEADLINE_MINUTES = -(-TRACING_DEADLINE_SECONDS // 60)

#: How long the normal restart after the capture may take to answer again.
REVERT_HEALTH_WAIT_SECONDS = RESTART_HEALTH_WAIT_SECONDS

#: How often the bridge asks ``/health`` while it waits, and how long one
#: ``/health`` probe may take.
HEALTH_POLL_SECONDS = 5
HEALTH_PROBE_SECONDS = 3

#: How long one quick ``systemctl`` or a ``systemd-run`` that does not wait
#: may take, and one ``docker inspect`` of the serving container.
SYSTEMCTL_SECONDS = 30
INSPECT_SECONDS = 30

#: Where the runtime drop-in lives: ``/run`` (a tmpfs, gone at boot).
DROPIN_ROOT = "/run/systemd/system"
DROPIN_NAME = "50-vaelor-kernel-profile.conf"

#: The transient units: the deadline's timer (and its service), the apply,
#: and the revert run at once when the capture ends.
REVERT_TIMER_UNIT = "vaelor-gpu-kernel-profile-revert"
APPLY_UNIT = "vaelor-gpu-kernel-profile-apply"
REVERT_NOW_UNIT = "vaelor-gpu-kernel-profile-revert-now"
PROGRAM_MODULE = "vaelor.gpu_kernel_profile_mode"
SYSTEMD_RUN = "/usr/bin/systemd-run"
SYSTEMCTL = "/usr/bin/systemctl"
#: The apply and the immediate revert each queue a restart and return; a
#: transient unit that has not finished in this long is killed by systemd,
#: and the bridge waits for ``systemd-run --wait`` at most half a minute more.
PROGRAM_SECONDS = 120
PROGRAM_WAIT_SECONDS = PROGRAM_SECONDS + 30

#: The hold the replica watch reads (``/run/vaelor``, root-written, 0644).
HOLD_PATH = run_path("gpu-kernel-profile.json")
_HOLD_LIMIT = 4096

# --- refusals, in the owner's words ----------------------------------------
NO_CLUSTER_MODEL = "No cluster model is serving on this controller, so there is nothing to profile."
NOT_HEALTHY = (
    "The cluster model is not serving right now, so a kernel profile cannot "
    "restart it. Try again once it serves.")
SPLIT_REFUSED = (
    "This model is split across machines; a kernel profile restarts one "
    "machine's copy, and a split has no copy of its own to restart.")
NOT_REGISTRATION_OFF = (
    "This deployment's vLLM image does not serve with GPU tracing switched "
    "off (only vLLM 0.27 does, VD-165), so the ordinary profile is the one to run.")
NO_CONTROLLER_COPY = "This controller holds no copy of the model to restart."
CONTROLLER_COPY_DOWN = (
    "This controller's copy of the model is not answering, so it was not "
    "restarted for a kernel profile.")
UNIT_NOT_TEMPLATE = (
    "The serving unit installed on this controller is not the one Vaelor "
    "renders, so it was not changed.")
DEPLOY_IN_FLIGHT = (
    "The cluster model is being deployed, loaded or unloaded right now; run "
    "the kernel profile once that has finished.")
UNIT_CHANGED = (
    "The model's serving unit was changed while it restarted for the kernel "
    "profile, so no capture ran.")
ALREADY_TRACING = (
    "GPU tracing from an earlier kernel profile is still in place; it is "
    "switched off by itself within {} minutes.").format(TRACING_DEADLINE_MINUTES)


# --- what the bridge's verb says ---------------------------------------------
BUSY = "A GPU kernel profile is already running on this controller."
DEADLINE_NOT_ARMED = (
    "The time limit for GPU tracing could not be set on this machine, so the "
    "model was not restarted: {}")
NEVER_ANSWERED = (
    "The model did not answer within {} s of restarting with GPU tracing on, "
    "so no capture ran.").format(RESTART_HEALTH_WAIT_SECONDS)
APPLY_FAILED = "The model could not be restarted with GPU tracing on: {}"
REVERTED = "The model was restarted normally, with GPU tracing off."
REVERT_PENDING = (
    "Restarting the model normally did not complete here; GPU tracing is "
    "switched off by itself within {} minutes.").format(TRACING_DEADLINE_MINUTES)
BACK_SLOW = (
    "The model was restarted normally but had not answered after {} s; it may "
    "still be loading.").format(REVERT_HEALTH_WAIT_SECONDS)


def serving_record(records: Any, deployment_name: str) -> Optional[Mapping[str, Any]]:
    """The stored row of the deployment the mode file says this controller serves."""
    if not deployment_name:
        return None
    for record in records or ():
        if isinstance(record, Mapping) and str(record.get("name", "")) == deployment_name:
            return record
    return None


def record_refusal(record: Optional[Mapping[str, Any]]) -> Optional[str]:
    """Why the console will not start a kernel profile of ``record``, else ``None``.

    The control plane's half of the checks, on the deployment's stored row;
    the root bridge checks the installed unit itself and trusts none of this.
    """
    if not record:
        return NO_CLUSTER_MODEL
    if str(record.get("state", "")) != "healthy":
        return NOT_HEALTHY
    if record_mode(record) != VERDICT_REPLICATED:
        return SPLIT_REFUSED
    if REGISTRATION_SWITCH not in image_on_record(record.get("units") or {}).environment:
        return NOT_REGISTRATION_OFF
    own = [entry for entry in replica_entries(record)
           if str(entry.get("node_id", "")) == CONTROLLER_PLACEMENT_ID]
    if not own:
        return NO_CONTROLLER_COPY
    if not own[0].get("alive"):
        return CONTROLLER_COPY_DOWN
    return None


def tracing_exec_start(unit: str, text: str) -> str:
    """The installed unit's ``ExecStart`` with VD-165's switch removed, or a refusal.

    ``text`` must be the managed template exactly (`check_rendered_unit`), a
    server's, not a split's, carrying the switch once; the result must be the
    template too. Any other shape raises ``ValueError`` and nothing is written.
    """
    try:
        if unit_role(unit) != SERVER_ROLE:
            raise ValueError(UNIT_NOT_TEMPLATE)
        check_rendered_unit(unit, text)
    except ValueError:
        raise ValueError(UNIT_NOT_TEMPLATE) from None
    lines = text.split("\n")
    starts = [line for line in lines if line.startswith("ExecStart=")]
    if len(starts) != 1 or any(line.startswith("BindsTo=") for line in lines):
        raise ValueError(SPLIT_REFUSED)
    start = starts[0][len("ExecStart="):]
    words = start.split(" ")
    if "--distributed-executor-backend" in words:
        raise ValueError(SPLIT_REFUSED)
    at = [index for index in range(len(words) - 1)
          if words[index] == "-e" and words[index + 1] == REGISTRATION_SWITCH]
    if len(at) != 1:
        raise ValueError(NOT_REGISTRATION_OFF)
    tracing = " ".join(words[: at[0]] + words[at[0] + 2:])
    try:
        check_rendered_unit(unit, text.replace(starts[0], "ExecStart=" + tracing))
    except ValueError:
        raise ValueError(UNIT_NOT_TEMPLATE) from None
    return tracing


def dropin_text(tracing_start: str) -> str:
    """The runtime drop-in: the tracing ``ExecStart`` in place of the unit's own."""
    return "\n".join((
        "# Vaelor GPU kernel profile (VD-214): GPU tracing on, for at most "
        "{} s.".format(TRACING_DEADLINE_SECONDS),
        "# Removed by {}.timer; /run is cleared at boot.".format(REVERT_TIMER_UNIT),
        "[Service]",
        "ExecStart=",
        "ExecStart=" + tracing_start,
        "",
    ))


def dropin_directory(unit: str) -> str:
    return "{}/{}.d".format(DROPIN_ROOT, unit)


def server_unit_of(directory_name: str) -> Optional[str]:
    """The managed server unit a ``<unit>.d`` folder name belongs to, else ``None``."""
    if not directory_name.endswith(".service.d"):
        return None
    unit = directory_name[: -len(".d")]
    try:
        return unit if unit_role(unit) == SERVER_ROLE else None
    except ValueError:
        return None


def _program(*arguments: str) -> List[str]:
    return [VENV_PYTHON, "-I", "-m", PROGRAM_MODULE, *arguments]


def arm_command() -> List[str]:
    """The deadline: a transient timer that runs the revert, owned by systemd."""
    return [
        SYSTEMD_RUN, "--unit=" + REVERT_TIMER_UNIT, "--collect", "--quiet",
        "--on-active={}s".format(TRACING_DEADLINE_SECONDS),
        "--timer-property=AccuracySec=1s",
        "--property=RuntimeMaxSec={}".format(PROGRAM_SECONDS),
        "--description=Vaelor GPU kernel profile deadline (VD-214)",
        *_program("revert"),
    ]


def apply_command(unit: str) -> List[str]:
    """The apply, as root outside the bridge's sandbox: writes the drop-in, restarts."""
    return [
        SYSTEMD_RUN, "--unit=" + APPLY_UNIT, "--collect", "--wait", "--pipe", "--quiet",
        "--property=RuntimeMaxSec={}".format(PROGRAM_SECONDS),
        *_program("apply", unit),
    ]


def revert_now_command() -> List[str]:
    return [
        SYSTEMD_RUN, "--unit=" + REVERT_NOW_UNIT, "--collect", "--wait", "--pipe", "--quiet",
        "--property=RuntimeMaxSec={}".format(PROGRAM_SECONDS),
        *_program("revert"),
    ]


def disarm_commands() -> List[List[str]]:
    """Stop a timer left from an earlier run and clear its failed service."""
    return [
        [SYSTEMCTL, "stop", REVERT_TIMER_UNIT + ".timer"],
        [SYSTEMCTL, "reset-failed", REVERT_TIMER_UNIT + ".service", REVERT_TIMER_UNIT + ".timer"],
    ]


def reset_commands() -> List[List[str]]:
    """Clear failed transient units of an earlier run, so their names can be reused."""
    return [[SYSTEMCTL, "reset-failed", APPLY_UNIT + ".service", REVERT_NOW_UNIT + ".service"]]


# --- how long the bridge's verb can take, from the bounds its steps use ------

def _wait_seconds(limit: int) -> int:
    """A health wait: its limit, and one more poll begun just before it."""
    return limit + HEALTH_POLL_SECONDS + INSPECT_SECONDS + HEALTH_PROBE_SECONDS


def tracing_phase_seconds() -> int:
    """From the hold to the revert: clear, arm, apply, wait, re-read, capture."""
    return (
        (len(disarm_commands()) + len(reset_commands()) + 1) * SYSTEMCTL_SECONDS
        + PROGRAM_WAIT_SECONDS + _wait_seconds(RESTART_HEALTH_WAIT_SECONDS)
        + CAPTURE_ALLOWANCE_SECONDS
    )


def revert_phase_seconds() -> int:
    """The revert: run it, disarm the deadline, wait for the model, check it."""
    return (
        PROGRAM_WAIT_SECONDS + len(disarm_commands()) * SYSTEMCTL_SECONDS
        + _wait_seconds(REVERT_HEALTH_WAIT_SECONDS) + INSPECT_SECONDS
    )


def bridge_worst_case_seconds() -> int:
    """The verb at every step's own bound: the first probe and both phases.

    Computed, not written down, from the same constants the steps are given
    (review 2: the prose sum left out the polls' and commands' bounds, and
    the console gave up first).
    """
    return HEALTH_PROBE_SECONDS + tracing_phase_seconds() + revert_phase_seconds()


#: The watch's hold, each phase's own: written at the start for the tracing
#: phase and again for the revert, cleared when the verb ends. The longer of
#: the two is also the furthest ahead a hold may ever be.
HOLD_SECONDS = max(tracing_phase_seconds(), revert_phase_seconds())

#: How long the client waits for the bridge: its worst case and a minute.
REQUEST_SECONDS = bridge_worst_case_seconds() + 60


# --- the watch's hold -------------------------------------------------------

_READ_FLAGS = (os.O_RDONLY | getattr(os, "O_NOFOLLOW", 0) | getattr(os, "O_NONBLOCK", 0)
               | getattr(os, "O_BINARY", 0))


#: The hold's mode, set on the file itself so no umask narrows it: the bridge
#: runs with ``UMask=0007``, which made a plain create 0640 root:root, and the
#: watch's account (``vaelor-workloads``, group ``vaelor``) reads it as
#: "other" - it may cross ``/run/vaelor`` (1770 root:vaelor) but owns nothing
#: in it (LESSONS 13, adversarial review 1).
HOLD_MODE = 0o644


def write_hold(deployment: str, *, seconds: int = HOLD_SECONDS, path: str = HOLD_PATH,
               clock: Callable[[], float] = time.time) -> None:
    """Hold the replica watch for ``deployment`` for ``seconds`` (root writes it).

    The temporary is a fresh random name (``mkstemp``: ``O_EXCL``, no link
    followed), so a group member of the sticky ``/run/vaelor`` cannot plant
    the name first and refuse the profile; it is renamed over the hold.
    """
    body = json.dumps({"deployment": str(deployment),
                       "until": float(clock()) + min(int(seconds), HOLD_SECONDS)})
    directory, name = os.path.split(path)
    handle, temporary = tempfile.mkstemp(prefix="." + name + ".", dir=directory or None)
    try:
        try:
            if hasattr(os, "fchmod"):
                os.fchmod(handle, HOLD_MODE)
            os.write(handle, body.encode("utf-8"))
        finally:
            os.close(handle)
        os.replace(temporary, path)
    except BaseException:
        try:
            os.unlink(temporary)
        except OSError:
            pass
        raise


def clear_hold(path: str = HOLD_PATH) -> None:
    try:
        os.unlink(path)
    except FileNotFoundError:
        pass


def _root_owners() -> FrozenSet[int]:
    return frozenset({0})


def held(deployment: Any, *, path: str = HOLD_PATH, clock: Callable[[], float] = time.time,
         owners: Optional[FrozenSet[int]] = None) -> bool:
    """Whether the watch should not count a silent pass of ``deployment`` now.

    Only a regular, single-named file owned by root (``owners``) naming this
    deployment, whose time is still ahead and not more than :data:`HOLD_SECONDS`
    ahead. Anything else - absent, a link, another writer, a past or far-future
    time, unreadable - is no hold: the watch counts as it always did.
    """
    allowed = _root_owners() if owners is None else owners
    try:
        handle = os.open(path, _READ_FLAGS)
    except OSError:
        return False
    try:
        status = os.fstat(handle)
        if (not stat.S_ISREG(status.st_mode) or status.st_nlink != 1
                or (allowed and status.st_uid not in allowed)):
            return False
        data = os.read(handle, _HOLD_LIMIT + 1)
    except OSError:
        return False
    finally:
        os.close(handle)
    try:
        hold = json.loads(data.decode("utf-8")) if len(data) <= _HOLD_LIMIT else None
        name, until = str(hold["deployment"]), float(hold["until"])
    except (ValueError, TypeError, KeyError):
        return False
    now = float(clock())
    return name == str(deployment) and now < until <= now + HOLD_SECONDS
