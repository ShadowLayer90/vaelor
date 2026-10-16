"""The root bridge's GPU kernel profile verb (VD-214), as a runtime mixin.

``run_gpu_kernel_profile`` restarts the controller's copy of the cluster model
with GPU tracing on, runs the existing bounded profile (`ServingProfileMixin`),
and restarts it normally. The rules, the deadline and why it cannot be left on
are in `gpu_kernel_profile`; the root work outside the sandbox is
`gpu_kernel_profile_mode`. Only the capture window crosses the socket: the
unit is named from the mode file and its text is the one this bridge installed
(VD-143, LESSONS 18).

**Order is the guarantee.** Check everything, then hold the watch, then arm
the deadline, and only then write the drop-in. If the deadline cannot be armed
nothing is changed. Whatever follows - a model that never answers, a failed
capture, an exception - the ``finally`` reverts at once; and if that revert
fails, or this process dies first, the deadline's timer still does it.

Never takes the bridge's lock: the model verbs it serialises must not wait
minutes behind a profile (the profile verb's constraint (b)). One kernel
profile at a time is this module's own lock.
"""

from __future__ import annotations

import os
import subprocess
import threading
import time
import urllib.error
import urllib.request
from typing import Any, Dict, List, Optional


def _kernel():
    """`gpu_kernel_profile`, imported when used: it reaches the unit template,
    whose import chain comes back to the bridge module this mixin is part of."""
    from . import gpu_kernel_profile

    return gpu_kernel_profile


_KERNEL_PROFILE_LOCK = threading.Lock()


class KernelProfileMixin:
    """``run_gpu_kernel_profile`` and the seams it is tested through."""

    def run_gpu_kernel_profile(self, seconds: Any) -> Dict[str, Any]:
        kernel = _kernel()
        if not _KERNEL_PROFILE_LOCK.acquire(blocking=False):
            raise RuntimeError(kernel.BUSY)
        try:
            return self._kernel_profile(seconds)
        finally:
            _KERNEL_PROFILE_LOCK.release()

    def _kernel_profile(self, seconds: Any) -> Dict[str, Any]:
        from .gpu_pool_units import SERVER_ROLE, container_name, unit_name
        kernel = _kernel()

        deployment, port = self._kernel_target()
        unit = unit_name(deployment, SERVER_ROLE)
        container = container_name(deployment, SERVER_ROLE)
        # Refuses a split, a 0.22 unit, or a unit that is not the template.
        source = self._kernel_unit_text(unit)
        kernel.tracing_exec_start(unit, source)
        if self._kernel_dropin_present(unit):
            raise RuntimeError(kernel.ALREADY_TRACING)
        if not self._kernel_health(port):
            raise RuntimeError(kernel.CONTROLLER_COPY_DOWN)
        kernel.write_hold(deployment, seconds=kernel.tracing_phase_seconds(),
                          clock=self._kernel_clock)
        for command in kernel.disarm_commands() + kernel.reset_commands():
            self._kernel_run(command)  # a stale timer only; no drop-in is in place
        armed = self._kernel_run(kernel.arm_command())  # returns once the timer is set
        if armed.returncode != 0:
            kernel.clear_hold()
            raise RuntimeError(kernel.DEADLINE_NOT_ARMED.format(_tail(armed) or "systemd-run failed."))
        outcome: Dict[str, Any] = {"deadline_seconds": kernel.TRACING_DEADLINE_SECONDS}
        failure: Optional[str] = None
        result: Optional[Dict[str, Any]] = None
        try:
            applied = self._kernel_run(kernel.apply_command(unit), kernel.PROGRAM_WAIT_SECONDS)
            if applied.returncode != 0:
                failure = kernel.APPLY_FAILED.format(_tail(applied) or "the restart was refused.")
            elif not self._kernel_wait(container, port, kernel.RESTART_HEALTH_WAIT_SECONDS,
                                       tracing=True):
                failure = kernel.NEVER_ANSWERED
            elif self._kernel_unit_text(unit) != source:
                # A deploy or load re-rendered the unit meanwhile: the drop-in
                # was derived from the old text, so nothing is captured from it.
                failure = kernel.UNIT_CHANGED
            else:
                # The wait saw the new process run without the switch.
                outcome["tracing_confirmed"] = True
                result = self.run_serving_profile(seconds)
        except Exception as error:  # noqa: BLE001 - reverted below whatever it was
            failure = failure or "The kernel profile stopped: {}".format(error)
        finally:
            # The hold, renewed for the revert's own phase (`revert_phase_seconds`).
            kernel.write_hold(deployment, seconds=kernel.revert_phase_seconds(),
                              clock=self._kernel_clock)
            outcome.update(self._kernel_revert(container, port))
        if failure is not None or result is None:
            raise RuntimeError("{} {}".format(failure or kernel.NEVER_ANSWERED, outcome["revert_note"]))
        return dict(result, kernel_profile=outcome)

    def _kernel_revert(self, container: str, port: int) -> Dict[str, Any]:
        """Revert at once; disarm the deadline only when the revert ran.

        The watch's hold is cleared on every path (adversarial review 1): it
        masks a dead model while it stands, so it ends with this verb, and a
        later revert by the timer writes its own, shorter one.
        """
        kernel = _kernel()
        try:
            reverted = self._kernel_run(kernel.revert_now_command(), kernel.PROGRAM_WAIT_SECONDS)
            if reverted.returncode != 0:
                # The timer stays armed: it is now the revert (LESSONS 1, said).
                return {"reverted": False, "revert_note": kernel.REVERT_PENDING}
            for command in kernel.disarm_commands():
                self._kernel_run(command)
            back = self._kernel_wait(container, port, kernel.REVERT_HEALTH_WAIT_SECONDS, tracing=False)
            off = self._kernel_switch_absent(container) is False
            return {"reverted": True, "serving_again": back,
                    "tracing_off_confirmed": off,
                    "revert_note": kernel.REVERTED if back else kernel.BACK_SLOW}
        finally:
            kernel.clear_hold()

    # --- seams ---------------------------------------------------------------

    @staticmethod
    def _kernel_target():
        """``(deployment, controller replica port)`` from the mode file, or a refusal."""
        from .gpu_cluster_mode_state import MODE_CLUSTER, ClusterModeStore
        kernel = _kernel()

        state = ClusterModeStore().read()
        base = int(getattr(state, "cluster_port", 0) or 0)
        if state.mode != MODE_CLUSTER or not state.deployment_name or base <= 0:
            raise RuntimeError(kernel.NO_CLUSTER_MODEL)
        if state.deploy_in_flight:
            raise RuntimeError(kernel.DEPLOY_IN_FLIGHT)
        # The controller's replica listens one above the balancer (VD-129).
        return state.deployment_name, base + 1

    @staticmethod
    def _kernel_unit_text(unit: str) -> str:
        from .gpu_kernel_profile_mode import read_regular
        from .bridge_managed_units import UNIT_DIRECTORY
        kernel = _kernel()

        try:
            return read_regular(os.path.join(UNIT_DIRECTORY, unit))
        except OSError:
            raise RuntimeError(kernel.NO_CONTROLLER_COPY) from None

    @staticmethod
    def _kernel_dropin_present(unit: str) -> bool:
        kernel = _kernel()
        return os.path.lexists(os.path.join(kernel.dropin_directory(unit), kernel.DROPIN_NAME))

    @staticmethod
    def _kernel_run(argv: List[str], timeout: Optional[int] = None) -> subprocess.CompletedProcess:
        """One fixed root command, bounded by ``timeout`` (a quick ``systemctl``'s
        by default); a spawn failure or timeout is a non-zero result."""
        kernel = _kernel()
        try:
            return subprocess.run(
                argv, capture_output=True, text=True,
                timeout=kernel.SYSTEMCTL_SECONDS if timeout is None else timeout, check=False,
            )
        except (OSError, subprocess.SubprocessError) as error:
            return subprocess.CompletedProcess(argv, 1, "", str(error))

    @staticmethod
    def _kernel_health(port: int) -> bool:
        """Whether the replica's own ``/health`` answers 200 on loopback."""
        kernel = _kernel()
        try:
            with urllib.request.urlopen(
                "http://127.0.0.1:{}/health".format(int(port)),
                timeout=kernel.HEALTH_PROBE_SECONDS,
            ) as answer:
                return answer.status == 200
        except (OSError, ValueError, urllib.error.URLError):
            return False

    def _kernel_wait(self, container: str, port: int, limit: int, *, tracing: bool) -> bool:
        """Wait for the RESTARTED process: the switch as wanted, and ``/health`` 200.

        The restart is queued, not done, so the old process may still answer
        ``/health`` for a while; only a process whose own environment shows
        the wanted side of VD-165's switch is the new one (LESSONS 1: verify
        what you got). Back from tracing, an unreadable environment with a
        healthy answer is accepted, since only the tracing process can be
        told apart by it; it is then not reported as confirmed.
        """
        kernel = _kernel()
        deadline = self._kernel_clock() + limit
        while True:
            self._kernel_sleep(kernel.HEALTH_POLL_SECONDS)
            absent = self._kernel_switch_absent(container)
            wanted = absent is True if tracing else absent is not True
            if wanted and self._kernel_health(port):
                return True
            if self._kernel_clock() >= deadline:
                return False

    def _kernel_switch_absent(self, container: str) -> Optional[bool]:
        """Whether the serving process runs WITHOUT VD-165's switch; ``None`` if unread."""
        from .gpu_kernel_profile_mode import _container_environ
        kernel = _kernel()

        environ = _container_environ(container)
        if environ is None:
            return None
        return kernel.REGISTRATION_SWITCH.encode() not in environ.split(b"\0")

    @staticmethod
    def _kernel_clock() -> float:
        return time.time()

    @staticmethod
    def _kernel_sleep(seconds: float) -> None:
        time.sleep(seconds)


def _tail(completed: Any) -> str:
    text = str(getattr(completed, "stderr", "") or getattr(completed, "stdout", "") or "")
    return text.strip()[:300]
