"""The on-demand serving profile verb of the root hardware bridge (VD-128).

Extracted from :mod:`vaelor.hardware_bridge` so the bridge stays under the
module ceiling; :class:`vaelor.hardware_bridge._HardwareRuntime` inherits it, so
the verb, its helpers and every patch target on the runtime class are unchanged.
Runs in ONE process: the root bridge. The capture never takes the bridge's lock
(constraint (b)): a profile window is long, and the model verbs the lock
serialises must not wait on it.
"""

from __future__ import annotations

import os
from typing import Any, Optional


class ServingProfileMixin:
    """``run_serving_profile`` and the root-side facts it derives."""

    def run_serving_profile(self, seconds: Any) -> dict[str, Any]:
        """Run ONE bounded on-demand profile of the GPU serving process, as root."""
        from . import serving_profiler

        bounded = serving_profiler.clamp_seconds(seconds)
        serving_pid = self._serving_container_pid(self._serving_container_name())
        vllm_serving, torch_ports = self._cluster_serving()
        probe = serving_profiler.probe_capabilities(
            serving_pid=serving_pid,
            # ``geteuid`` is POSIX-only; the bridge is a Linux root service, but
            # the getattr keeps the module importable/testable off-Linux, where it
            # reports "not root" (the honest not-available branch).
            as_root=(getattr(os, "geteuid", lambda: -1)() == 0),
            perf_event_paranoid=self._perf_event_paranoid(),
            perf_present=os.path.exists(serving_profiler.PERF_BIN),
            amd_smi_present=os.path.exists(serving_profiler.AMD_SMI_BIN),
            vllm_serving=vllm_serving,
            bpftrace_present=os.path.exists(serving_profiler.BPFTRACE_BIN),
        )
        # No lock: see constraint (b). The captures run through the injected
        # `_profile_run` seam, so the flow is unit-tested without a real perf/GPU.
        return serving_profiler.capture_profile(
            probe=probe, run=self._profile_run, seconds=bounded,
            torch_ports=torch_ports,
            profiles_dir=serving_profiler.SERVING_PROFILES_DIR,
        )

    @staticmethod
    def _serving_container_name() -> str:
        """The container to profile, by the active mode: Mode B vLLM server, else Mode A."""
        from .gpu_rocmfpx_service import GPU_CONTAINER_NAME
        try:
            from .gpu_cluster_mode_state import MODE_CLUSTER, ClusterModeStore
            from .gpu_pool_units import SERVER_ROLE, container_name
            state = ClusterModeStore().read()
            if state.mode == MODE_CLUSTER and state.deployment_name:
                return container_name(state.deployment_name, SERVER_ROLE)
        except Exception:  # noqa: BLE001 - a profile must never fail on this read
            pass
        return GPU_CONTAINER_NAME

    @staticmethod
    def _cluster_serving():
        """(vLLM serving?, loopback ports to drive) from the mode file (VD-128).

        Mode B only. The ports are the controller replica (balancer port + 1)
        then the distributed lead (balancer port), so the torch capture drives
        the controller's OWN loopback server, whose trace lands in the host
        profiles dir the bridge reads.
        """
        try:
            from .gpu_cluster_mode_state import MODE_CLUSTER, ClusterModeStore
            state = ClusterModeStore().read()
            base = int(getattr(state, "cluster_port", 0) or 0)
            if state.mode == MODE_CLUSTER and base > 0:
                return True, [base + 1, base]
        except Exception:  # noqa: BLE001 - a profile must never fail on this read
            pass
        return False, []

    @staticmethod
    def _serving_container_pid(container_name: str) -> Optional[int]:
        """The serving container's own main-process PID, from ``docker inspect``.

        Constraint (c): derived here, never taken from the caller. Docker prints
        ``0`` for a created-but-not-running container — not a process to profile —
        so that and any inspect failure return ``None`` and the probe reports "no
        serving process is running".
        """
        import shutil
        import subprocess

        docker = shutil.which("docker") or "/usr/bin/docker"
        try:
            completed = subprocess.run(
                [docker, "inspect", "-f", "{{.State.Pid}}", str(container_name)],
                capture_output=True, text=True, timeout=10, check=False,
            )
        except (OSError, subprocess.SubprocessError):
            return None
        if completed.returncode != 0:
            return None
        try:
            pid = int(str(completed.stdout or "").strip())
        except (TypeError, ValueError):
            return None
        return pid if pid > 0 else None

    @staticmethod
    def _perf_event_paranoid() -> Optional[int]:
        """The kernel's non-root profiling gate, so the honest note can name it.

        Root ``perf`` bypasses this value, but reading it lets the CPU capture's
        not-available note (on a hypothetical non-root run) state the actual gate.
        Unreadable or non-integer degrades to ``None`` ("unknown"), never a guess.
        """
        from .serving_profiler import PERF_EVENT_PARANOID_PATH

        try:
            with open(PERF_EVENT_PARANOID_PATH, encoding="utf-8") as handle:
                return int(handle.read().strip())
        except (OSError, ValueError):
            return None

    @staticmethod
    def _profile_run(argv: Any, *, timeout: Any):
        """Run one bounded capture subprocess as root, never raising on failure.

        Constraint (a): a HARD subprocess timeout bounds every capture, capped at
        the longest a single capture may take; a timeout or spawn error becomes a
        non-zero ``CompletedProcess`` the pure orchestrator degrades on (an honest
        "captured nothing" note) rather than killing the handler thread.
        """
        import subprocess

        from .serving_profiler import (
            PROFILE_MAX_SECONDS,
            PROFILE_RECORD_MARGIN_SECONDS,
        )

        ceiling = PROFILE_MAX_SECONDS + PROFILE_RECORD_MARGIN_SECONDS
        try:
            hard = max(1, min(int(timeout), ceiling))
        except (TypeError, ValueError):
            hard = ceiling
        args = [str(item) for item in argv]
        try:
            return subprocess.run(
                args, capture_output=True, text=True, timeout=hard, check=False,
            )
        except subprocess.TimeoutExpired:
            return subprocess.CompletedProcess(
                args, 124, "", "The capture did not finish within {} s.".format(hard)
            )
        except OSError as error:
            return subprocess.CompletedProcess(args, 127, "", str(error))
