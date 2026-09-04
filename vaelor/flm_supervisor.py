"""Control-plane supervision of the flm-real NPU server (VD-001 / VD-002).

The unprivileged half of the subsystem. It decides *whether* the Assistant
should be served on the NPU, allocates a loopback port, asks the privileged
bridge to launch flm-real, watches the endpoint until it answers, and restarts
it when it stops. The privilege lives in :mod:`vaelor.hardware_bridge`; the
*policy* — port allocation, health, retry, restart — lives here, where it can
be unit-tested with a faked flm-real and no root.

Nothing here interpolates a model tag into anything. The launcher it drives
carries the tag to :mod:`vaelor.flm_service`, which validates it at the root
boundary; the supervisor's own inputs are the plan (Vaelor constants) and a
port it chose itself.
"""

from __future__ import annotations

import socket
import time
import urllib.error
import urllib.request
from typing import Any, Callable, Dict, Mapping, Optional

from .executor_network import available_model_port
from .flm_service import RESERVED_CONTROL_PLANE_PORTS


#: How long to wait for a freshly launched flm-real to answer on loopback
#: before giving up. FLM maps its model into pinned NPU pages at start, which is
#: not instant; this is the ceiling on that wait, after which the launch is a
#: failure rather than an indefinite hang.
HEALTH_DEADLINE_SECONDS = 90.0

#: Seconds between health probes while waiting for the server to come up.
HEALTH_POLL_SECONDS = 2.0

#: Per-probe network timeout. A loopback endpoint that is up answers in
#: milliseconds; one that is still loading refuses or drops, and must not hold
#: the poll loop for longer than one interval.
PROBE_TIMEOUT_SECONDS = 3.0

#: Paths tried, in order, to decide the server is answering. ``/health`` is the
#: cheapest liveness signal; ``/v1/models`` confirms the OpenAI surface is up
#: for a server that does not publish ``/health``.
HEALTH_PATHS = ("/health", "/v1/models")


def should_serve_on_npu(
    hardware: Mapping[str, Any],
    payload: Mapping[str, Any],
    plan: Mapping[str, Any],
    capability: Mapping[str, Any],
) -> bool:
    """Whether this deploy belongs on the NPU rather than on llama.cpp.

    Every clause must hold, and the first is the isolation guarantee: a machine
    with no neural accelerator (every Raspberry Pi) fails here and the deploy
    takes the unchanged llama.cpp path byte-for-byte. The surface clause keeps
    the GPU AI-Chat tier (gpt-oss on llama-server) untouched — only the
    ``assistant`` surface is a candidate. The capability clause is honest
    discovery (binary + device + model). The plan clause refuses a model the
    capability table has ruled unfit: an unusable model has no launchable tag,
    and serving nothing is better than serving something that cannot answer.
    """
    if not (hardware or {}).get("neural_accelerators"):
        return False
    surface = str((payload or {}).get("surface") or "assistant")
    if surface != "assistant":
        return False
    if not (capability or {}).get("available"):
        return False
    return bool((plan or {}).get("usable")) and bool((plan or {}).get("flm_tag"))


def _endpoint_healthy(
    port: int,
    *,
    timeout: float = PROBE_TIMEOUT_SECONDS,
    opener: Optional[Callable[..., Any]] = None,
) -> bool:
    """Whether an OpenAI-compatible server is answering on this loopback port."""
    request_opener = opener or urllib.request.urlopen
    for path in HEALTH_PATHS:
        url = "http://127.0.0.1:{}{}".format(int(port), path)
        try:
            with request_opener(url, timeout=timeout) as response:
                if getattr(response, "status", 200) == 200:
                    return True
        except (OSError, urllib.error.URLError, ValueError):
            continue
    return False


class FlmSupervisor:
    """Start, health-check, restart and stop the flm-real NPU server.

    Driven through three injected seams so it is testable with a faked
    flm-real:

    * ``launcher`` — an object with ``start(tag, ctx_len, port)``, ``stop()``
      and ``status()``. Production wires :class:`BridgeLauncher`; a test passes
      a fake that records calls and reports liveness.
    * ``health`` — ``health(port) -> bool``. Production probes loopback; a test
      scripts a sequence of up/down answers.
    * ``allocate_port`` — chooses a free loopback port. Production uses
      :func:`vaelor.executor_network.available_model_port`.
    """

    def __init__(
        self,
        launcher: Any,
        *,
        health: Optional[Callable[[int], bool]] = None,
        allocate_port: Optional[Callable[[], int]] = None,
        deadline_seconds: float = HEALTH_DEADLINE_SECONDS,
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

        The same guard the llama.cpp deploy applies, so an NPU deploy cannot be
        pointed at a privileged port or at the reserved control-plane ports.
        """
        port = int(requested or 0) or int(self._allocate_port())
        if not 1024 <= port <= 65535 or port in RESERVED_CONTROL_PLANE_PORTS:
            raise ValueError(
                "Choose an available loopback port from 1024 to 65535 for the "
                "flm-real server."
            )
        return port

    def serve(
        self, tag: str, *, ctx_len: int, port: int,
        announce: Optional[Callable[[str], None]] = None,
    ) -> Dict[str, Any]:
        """Launch flm-real and wait until it answers, or fail with a reason.

        A launch that never becomes healthy within the deadline is stopped and
        reported as a failure rather than left running unreachable — the same
        rollback discipline the llama.cpp deploy follows.
        """
        self._launcher.start(tag, ctx_len=ctx_len, port=port)
        if announce:
            announce("Waiting for the neural processor to load the model")
        deadline = self._monotonic() + self._deadline
        while self._monotonic() < deadline:
            if self._health(port):
                return {
                    "healthy": True, "tag": tag, "port": port,
                    "ctx_len": ctx_len,
                    "endpoint": "http://127.0.0.1:{}/v1".format(port),
                    "health_url": "http://127.0.0.1:{}/health".format(port),
                }
            self._sleep(self._poll)
        self._launcher.stop()
        raise RuntimeError(
            "flm-real started but did not answer on loopback within {:.0f} "
            "seconds; it was stopped.".format(self._deadline)
        )

    def healthy(self, port: int) -> bool:
        return bool(self._health(port))

    def restart_if_unhealthy(
        self, tag: str, *, ctx_len: int, port: int,
        announce: Optional[Callable[[str], None]] = None,
    ) -> Dict[str, Any]:
        """Restart the server only when it is actually not answering.

        A healthy server is left alone — restarting a working NPU tier evicts a
        loaded model to no purpose. An unhealthy one is stopped and served
        again, and the result says which happened.
        """
        if self._health(port):
            return {"restarted": False, "healthy": True, "port": port}
        self._launcher.stop()
        result = self.serve(tag, ctx_len=ctx_len, port=port, announce=announce)
        return {"restarted": True, **result}

    def stop(self) -> Dict[str, Any]:
        outcome = self._launcher.stop()
        return outcome if isinstance(outcome, dict) else {"stopped": True}


class BridgeLauncher:
    """Adapts the privileged hardware bridge to the supervisor's launcher seam.

    The supervisor knows nothing about how flm-real is actually launched as
    root; this is the one place that does, and it is a thin pass-through to the
    bridge client. Kept apart from the supervisor so the policy stays testable
    without a bridge.
    """

    def __init__(self, client: Any):
        self._client = client

    def start(self, tag: str, *, ctx_len: int, port: int) -> Dict[str, Any]:
        return self._client.flm_start(tag, ctx_len, port)

    def stop(self) -> Dict[str, Any]:
        return self._client.flm_stop()

    def status(self) -> Dict[str, Any]:
        return self._client.flm_status()
