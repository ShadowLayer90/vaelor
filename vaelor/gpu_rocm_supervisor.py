"""Control-plane supervision of the ROCmFPX GPU model server (GPU AI-Chat).

The GPU analogue of :mod:`vaelor.flm_supervisor`, and the simpler half of the
GPU subsystem: it allocates a loopback port, launches ``llama-server`` (the
ROCmFPX fork), watches the endpoint until it answers, and restarts it when it
stops. The policy - port allocation, health, retry, restart - lives here where
it can be unit-tested with a faked server and no GPU.

Like the NPU supervisor, the launch crosses a **privileged bridge**: the GPU
server needs /dev/dri + /dev/kfd for HIP/Vulkan, a writable ``/var/log/vaelor``
for its log, and no ``MemoryDenyWriteExecute`` for the fork's JIT — none of
which the sandboxed workload executor has — so :class:`GpuBridgeLauncher` drives
it through the root hardware bridge, the same account that launches flm-real.
The policy above the launcher (port, health, retry, restart) stays here and
stays testable with a fake launcher and no GPU. The health probe and the port
helper are *reused* from :mod:`vaelor.flm_supervisor` and
:mod:`vaelor.executor_network` rather than duplicated - an OpenAI-compatible
``llama-server`` answers on the same ``/health`` and ``/v1/models`` surface the
NPU server does.
"""

from __future__ import annotations

import socket
import time
from typing import Any, Callable, Dict, Optional

from .executor_network import available_model_port
from .flm_service import RESERVED_CONTROL_PLANE_PORTS
from .flm_supervisor import HEALTH_POLL_SECONDS, _endpoint_healthy


#: How long to wait for a freshly launched GPU server to answer on loopback
#: before giving up. A 27B FP4 model load plus the graph build on the Strix Halo
#: is markedly slower than the NPU's ~90 s ceiling, so this is set well above it.
#:
#: **240 s is a provisional ceiling, to be confirmed on the Z2.** It was chosen
#: to comfortably exceed the by-hand load observed while proving the recipe, not
#: measured as the worst case; the real value wants a cold-load measurement on
#: the box, the same way the flm deadline was set from what VD-001 saw.
GPU_HEALTH_DEADLINE_SECONDS = 240.0


class GpuRocmSupervisor:
    """Start, health-check, restart and stop the ROCmFPX GPU model server.

    Driven through three injected seams so it is testable with a faked server,
    mirroring :class:`vaelor.flm_supervisor.FlmSupervisor`:

    * ``launcher`` - an object with ``start(model_path, port, runtime)``,
      ``stop()`` and ``status()``. Production wires :class:`GpuBridgeLauncher`,
      which launches the server through the root hardware bridge; a test passes a
      fake that records calls and reports liveness.
    * ``health`` - ``health(port) -> bool``. Production probes loopback via the
      reused :func:`vaelor.flm_supervisor._endpoint_healthy`; a test scripts a
      sequence of up/down answers.
    * ``allocate_port`` - chooses a free loopback port. Production reuses
      :func:`vaelor.executor_network.available_model_port`.
    """

    def __init__(
        self,
        launcher: Any,
        *,
        health: Optional[Callable[[int], bool]] = None,
        allocate_port: Optional[Callable[[], int]] = None,
        deadline_seconds: float = GPU_HEALTH_DEADLINE_SECONDS,
        poll_seconds: float = HEALTH_POLL_SECONDS,
        sleep: Callable[[float], None] = time.sleep,
        monotonic: Callable[[], float] = time.monotonic,
    ):
        self._launcher = launcher
        self._health = health or _endpoint_healthy
        self._allocate_port = allocate_port or (
            lambda: available_model_port(socket.socket)
        )
        self._deadline = float(deadline_seconds)
        self._poll = float(poll_seconds)
        self._sleep = sleep
        self._monotonic = monotonic

    def allocate_port(self, requested: int = 0) -> int:
        """A validated loopback port: the requested one, or a free one chosen.

        The same guard the llama.cpp and NPU deploys apply, so a GPU deploy
        cannot be pointed at a privileged port or at the reserved control-plane
        ports.
        """
        port = int(requested or 0) or int(self._allocate_port())
        if not 1024 <= port <= 65535 or port in RESERVED_CONTROL_PLANE_PORTS:
            raise ValueError(
                "Choose an available loopback port from 1024 to 65535 for the "
                "GPU model server."
            )
        return port

    def serve(
        self, model_path: str, *, port: int,
        runtime: Optional[Dict[str, Any]] = None,
        announce: Optional[Callable[[str], None]] = None,
    ) -> Dict[str, Any]:
        """Launch the GPU server and wait until it answers, or fail with a reason.

        A launch that never becomes healthy within the deadline is stopped and
        reported as a failure rather than left running unreachable - the same
        rollback discipline the llama.cpp and NPU deploys follow.
        """
        self._launcher.start(model_path, port=port, runtime=runtime)
        if announce:
            announce("Waiting for the GPU to load the model")
        deadline = self._monotonic() + self._deadline
        while self._monotonic() < deadline:
            if self._health(port):
                return {
                    "healthy": True, "model_path": model_path, "port": port,
                    "endpoint": "http://127.0.0.1:{}/v1".format(port),
                    "health_url": "http://127.0.0.1:{}/health".format(port),
                }
            self._sleep(self._poll)
        self._launcher.stop()
        raise RuntimeError(
            "The GPU model server started but did not answer on loopback within "
            "{:.0f} seconds; it was stopped.".format(self._deadline)
        )

    def healthy(self, port: int) -> bool:
        return bool(self._health(port))

    def restart_if_unhealthy(
        self, model_path: str, *, port: int,
        runtime: Optional[Dict[str, Any]] = None,
        announce: Optional[Callable[[str], None]] = None,
    ) -> Dict[str, Any]:
        """Restart the server only when it is actually not answering.

        A healthy server is left alone - restarting a working GPU tier evicts a
        loaded 27B model to no purpose, and reloading it costs the full deadline
        above. An unhealthy one is stopped and served again, and the result says
        which happened.
        """
        if self._health(port):
            return {"restarted": False, "healthy": True, "port": port}
        self._launcher.stop()
        result = self.serve(
            model_path, port=port, runtime=runtime, announce=announce
        )
        return {"restarted": True, **result}

    def stop(self) -> Dict[str, Any]:
        outcome = self._launcher.stop()
        return outcome if isinstance(outcome, dict) else {"stopped": True}


class GpuBridgeLauncher:
    """Adapts the privileged hardware bridge to the supervisor's launcher seam.

    The GPU analogue of :class:`vaelor.flm_supervisor.BridgeLauncher`. The
    supervisor knows nothing about where the GPU server is launched; this is the
    one place that does, and it is a thin pass-through to the bridge client,
    which carries the request to the root bridge where the /dev/dri + /dev/kfd
    access, the writable ``/var/log/vaelor`` and the absence of
    ``MemoryDenyWriteExecute`` actually are (the sandboxed executor has none of
    them). Kept apart from the supervisor so the policy stays testable without a
    bridge.
    """

    def __init__(self, client: Any):
        self._client = client

    def start(
        self, model_path: str, *, port: int,
        runtime: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        return self._client.gpu_start(model_path, port, runtime)

    def stop(self) -> Dict[str, Any]:
        return self._client.gpu_stop()

    def status(self) -> Dict[str, Any]:
        return self._client.gpu_status()
