"""The managed Arize Phoenix trace-collector CONTAINER (VD-128, Phase E').

Phoenix receives the inference gateway's OTLP spans and shows per-request traces.
It runs as ONE Vaelor-controlled container, launched by the root hardware bridge
(``phoenix_start``/``phoenix_stop``/``phoenix_status``) exactly as the LLM Server
auth proxy is (:mod:`vaelor.llm_server_proxy`): same ``docker run -d`` shape, same
injected ``run`` seam so the launch is verifiable without docker, same
``start``/``stop``/``alive``/``status`` process and the same
``apply``/``converged`` controller. The two differences from the proxy are the
only ones that matter here:

* **Loopback publish, no ``--network host``.** Phoenix binds ``0.0.0.0:6006``
  inside the container, and the launch maps ONLY the host's loopback to it
  (``-p 127.0.0.1:6006:6006``). So the collector and its UI are reachable on the
  controller's loopback - by the gateway on the same host, and by an operator
  through the console - but never unkeyed on the LAN. The appliance already fronts
  TLS and auth; an open trace UI on the LAN would undo that.
* **No key, no config file.** Phoenix is unkeyed by design (loopback-only), so
  there is no root-owned ``0600`` config to render and nothing secret on the argv.

The image is DIGEST-pinned for the reason the model and proxy images are: a moving
tag means the box that first pulls Phoenix and the box that redeploys later could
run different collectors under one name.
"""

from __future__ import annotations

import shutil
import subprocess
from typing import Any, Callable, Dict, List, Mapping, Optional

from .flm_service import _validate_port
from .phoenix_state import (
    PHOENIX_LOOPBACK_HOST, PHOENIX_PORT, PhoenixSettings, trace_endpoint,
)

#: The Arize Phoenix image the collector runs.
#:
#: **Pinned by DIGEST, in the ``name:tag@sha256:`` form docker accepts for
#: ``pull``/``run``/``image inspect`` alike**, exactly as
#: :data:`vaelor.gpu_rocmfpx_service.GPU_CONTAINER_IMAGE` and
#: :data:`vaelor.llm_server_proxy.PROXY_IMAGE` are, and for the same reason: this
#: is the trace collector the observability story rests on, and "whatever the
#: registry served that day" would make a serving/telemetry regression
#: unattributable. The tag is kept beside the digest because it is what a reader
#: recognises and what a re-pin starts from. ``version-20.9.0`` is the current
#: stable release; the digest is the multi-arch manifest-list digest, so
#: ``docker run`` resolves the controller's amd64 image from it automatically.
#: **To re-pin:** ``docker pull arizephoenix/phoenix:<tag>`` on the appliance,
#: then ``docker image inspect --format '{{index .RepoDigests 0}}'
#: arizephoenix/phoenix:<tag>`` for the digest and ``docker image ls`` for the
#: size.
PHOENIX_IMAGE = (
    "arizephoenix/phoenix:version-20.9.0"
    "@sha256:f8a80ff0ffb5ae394846baffaa0cc90b3add4e0f3515eaa968ca11079137bca4"
)

#: The fixed name of the single Phoenix container, so ``stop``/``status`` can
#: address it without tracking an id across the bridge boundary (mirrors
#: :data:`vaelor.llm_server_proxy.PROXY_CONTAINER_NAME`).
PHOENIX_CONTAINER_NAME = "vaelor-phoenix"

#: Bounds on the docker calls, mirroring the proxy launch's. A cold Phoenix pull
#: is larger than nginx, so the pull bound is generous.
DOCKER_RUN_TIMEOUT = 120
DOCKER_PULL_TIMEOUT = 900
DOCKER_QUERY_TIMEOUT = 30

#: How long ``docker stop`` may drain Phoenix before SIGKILL.
STOP_GRACE_SECONDS = 10


class PhoenixImageMissingError(FileNotFoundError):
    """The Phoenix image is not present and could not be pulled.

    A subclass of ``FileNotFoundError`` (hence ``OSError``) so a caller's
    fall-back set catches it like the proxy/model image-missing errors.
    """


def phoenix_container_command(
    *,
    port: int = PHOENIX_PORT,
    bind_host: str = PHOENIX_LOOPBACK_HOST,
    docker: str = "docker",
    image: str = PHOENIX_IMAGE,
    container_name: str = PHOENIX_CONTAINER_NAME,
) -> List[str]:
    """The ``docker run -d`` argv for the Phoenix container.

    ::

        docker run -d --name vaelor-phoenix -p 127.0.0.1:6006:6006 <image>

    The ``-p 127.0.0.1:6006:6006`` publish is the whole boundary: Phoenix listens
    on all interfaces INSIDE the container, and only the host's loopback is mapped
    to it, so the collector and UI answer on the controller's loopback and nowhere
    on the LAN. A list, never a string, so no element can be split or chained by a
    shell. The bind host is validated as loopback here at the launch boundary so a
    caller cannot widen the publish to ``0.0.0.0``.
    """
    listen = _validate_loopback(bind_host)
    listen_port = _validate_port(port)
    return [
        docker, "run", "-d", "--name", container_name,
        "-p", "{}:{}:{}".format(listen, listen_port, listen_port),
        image,
    ]


def _validate_loopback(value: Any) -> str:
    """Return the bind host, or raise unless it is loopback.

    The security invariant of this launch: Phoenix is published on loopback ONLY,
    so the one place the argv is built refuses any other bind rather than trusting
    the caller. ``127.0.0.1`` is the sole accepted value (IPv6 loopback ``::1`` is
    not used because the publish maps a v4 loopback).
    """
    text = str(value or "").strip()
    if text != PHOENIX_LOOPBACK_HOST:
        raise ValueError(
            "SECURITY / Phoenix: refusing to publish the trace collector on {!r}. "
            "Phoenix is unkeyed and must bind loopback ({}) only - the appliance "
            "fronts TLS and auth, and an open trace UI on the LAN would undo that."
            .format(text, PHOENIX_LOOPBACK_HOST)
        )
    return text


def _default_run(command: Any, *, timeout: Optional[int] = None):
    """Run a docker command, capturing output, never raising on non-zero.

    stdout/stderr are captured (not DEVNULL): a ``docker run`` that fails explains
    itself on stderr, which the caller surfaces in the raised error.
    """
    return subprocess.run(
        list(command), capture_output=True, text=True, check=False, timeout=timeout
    )


class PhoenixServerProcess:
    """A supervised Phoenix CONTAINER, launched and stopped root-side.

    The trace-collector analogue of
    :class:`vaelor.llm_server_proxy.LlmServerProxyProcess`: it manages a ``docker
    run -d`` Phoenix container published on loopback, with the same
    ``start``/``stop``/``alive``/``status`` shape and the same injected ``run``
    seam so the launch is verifiable without docker.
    """

    #: What this container is called in its own failure sentence.
    label = "Phoenix trace collector"

    def __init__(
        self,
        *,
        run: Optional[Callable[..., Any]] = None,
        docker: Optional[str] = None,
        image: str = PHOENIX_IMAGE,
        container_name: str = PHOENIX_CONTAINER_NAME,
    ):
        self._run = run or _default_run
        self._docker = docker
        self._image = image
        self._name = container_name
        self.bind_host = ""
        self.port = 0

    def _docker_binary(self) -> str:
        return self._docker or shutil.which("docker") or "/usr/bin/docker"

    def ensure_image(self, docker: Optional[str] = None) -> None:
        """Make sure the Phoenix image is present, pulling it once if not.

        ``docker image inspect`` is the cheap presence probe; only a miss triggers
        a ``docker pull``. A failed fetch raises :class:`PhoenixImageMissingError`
        naming it, so a box that cannot reach the registry degrades with a clear
        reason rather than a bare docker error.
        """
        docker = docker or self._docker_binary()
        inspected = self._run(
            [docker, "image", "inspect", self._image], timeout=DOCKER_QUERY_TIMEOUT
        )
        if getattr(inspected, "returncode", 1) == 0:
            return
        pulled = self._run([docker, "pull", self._image], timeout=DOCKER_PULL_TIMEOUT)
        if getattr(pulled, "returncode", 1) != 0:
            raise PhoenixImageMissingError(
                "The Phoenix image {} is not present and could not be pulled: {}"
                .format(self._image, (getattr(pulled, "stderr", "") or "").strip())
            )

    def start(
        self, *, port: int = PHOENIX_PORT, bind_host: str = PHOENIX_LOOPBACK_HOST,
        skip_ensure: bool = False,
    ) -> dict:
        """Validate, then launch the Phoenix container, replacing any running one.

        The bind host and port are validated (loopback only) while building the
        argv, BEFORE the previous container is touched, so a malformed request
        cannot take down a healthy collector.

        ``skip_ensure`` lets the caller pull the image FIRST, off the bridge lock:
        ``ensure_image`` can ``docker pull`` for minutes on first enable, and the
        bridge holds its lock across this whole method, so pulling here would
        freeze the model tiers the bridge also serializes. The bridge verb pulls
        unlocked and passes ``skip_ensure=True`` so only the fast run is locked.
        """
        command = phoenix_container_command(
            port=int(port), bind_host=bind_host, docker=self._docker_binary(),
            image=self._image, container_name=self._name,
        )
        docker = self._docker_binary()
        if not skip_ensure:
            self.ensure_image(docker)
        if self.alive():
            self.stop()
        else:
            self._run([docker, "rm", "-f", self._name], timeout=DOCKER_QUERY_TIMEOUT)
        result = self._run(command, timeout=DOCKER_RUN_TIMEOUT)
        if getattr(result, "returncode", 1) != 0:
            raise RuntimeError(
                "The {} container failed to launch: {}".format(
                    self.label, (getattr(result, "stderr", "") or "").strip()
                )
            )
        self.bind_host = _validate_loopback(bind_host)
        self.port = _validate_port(port)
        return self.status()

    def alive(self) -> bool:
        """Whether our Phoenix container exists and is running, per ``docker inspect``."""
        docker = self._docker_binary()
        try:
            result = self._run(
                [docker, "inspect", "-f", "{{.State.Running}}", self._name],
                timeout=DOCKER_QUERY_TIMEOUT,
            )
        except (OSError, subprocess.SubprocessError):
            return False
        return (
            getattr(result, "returncode", 1) == 0
            and (getattr(result, "stdout", "") or "").strip() == "true"
        )

    def stop(self, *, grace_seconds: int = STOP_GRACE_SECONDS) -> dict:
        """``docker stop`` then ``docker rm``. Best-effort so a stop never raises."""
        docker = self._docker_binary()
        was_running = self.alive()
        try:
            self._run(
                [docker, "stop", "--time", str(int(grace_seconds)), self._name],
                timeout=DOCKER_RUN_TIMEOUT,
            )
            self._run([docker, "rm", "-f", self._name], timeout=DOCKER_QUERY_TIMEOUT)
        except (OSError, subprocess.SubprocessError):
            pass
        return {"stopped": True, "was_running": was_running}

    def status(self) -> dict:
        return {
            "running": self.alive(),
            "bind_host": self.bind_host,
            "port": self.port,
            "container": self._name,
        }


class PhoenixController:
    """Converge the Phoenix container to the persisted enable flag, over the bridge.

    The executor-side seam the ``phoenix.apply`` job and the boot/failure reconcile
    drive, the trace-collector analogue of
    :class:`vaelor.llm_server_proxy.LlmServerProxyController`. It holds a
    bridge-client-like object with ``phoenix_start``/``phoenix_stop``/
    ``phoenix_status`` (production: the root
    :class:`vaelor.hardware_bridge_client.HardwareBridgeClient`; a test: a
    recording fake) and turns the persisted :class:`PhoenixSettings` into the one
    right call:

    * ENABLED -> ``phoenix_start`` the loopback collector;
    * DISABLED -> :meth:`stop`: ``phoenix_stop`` and remove it.

    :meth:`converged` is the "is the collector already right?" read the reconcile
    consults so it does not replace the container on every pass (the flap the LLM
    proxy's second live run found): enabled means running on loopback, disabled
    means not running. Fail-safe: an unreadable status is "not converged", so the
    caller applies rather than trusts.
    """

    def __init__(self, client: Any, *, port: int = PHOENIX_PORT):
        self._client = client
        self._port = port

    def apply(self, settings: PhoenixSettings) -> dict:
        if settings.enabled:
            return self._client.phoenix_start(self._port, PHOENIX_LOOPBACK_HOST)
        return self.stop()

    def converged(self, settings: PhoenixSettings) -> bool:
        try:
            status = self.status()
        except Exception:  # noqa: BLE001 - an unreadable collector is unconverged
            return False
        running = bool(status.get("running"))
        if not settings.enabled:
            return not running
        return (
            running
            and str(status.get("bind_host") or "") == PHOENIX_LOOPBACK_HOST
            and _port_or_zero(status.get("port")) == int(self._port)
        )

    def stop(self) -> dict:
        """Take the collector down and remove it, over the bridge."""
        return self._client.phoenix_stop()

    def status(self) -> Mapping[str, Any]:
        return self._client.phoenix_status()


def _port_or_zero(value: Any) -> int:
    """A status field as a port, or 0 for anything a port cannot be read from."""
    try:
        return int(value or 0)
    except (TypeError, ValueError):
        return 0


def trace_collector_status(store: Any, bridge: Any) -> Dict[str, Any]:
    """The honest trace-collector status: the persisted intent plus a live read.

    The ONE builder the admin route and the Performance-tab ``traces`` section
    both read, so the surface and the tab cannot disagree. ``enabled`` is the
    persisted flag (the intent); ``running`` is a BEST-EFFORT live read over the
    bridge - attempted only when the bridge socket is present
    (``bridge.available``, a cheap stat) so a down bridge never hangs a snapshot,
    and swallowed to ``False`` on any trouble, so the tab degrades to "enabled,
    not up yet" rather than crashing. ``otlp_endpoint`` is where the gateway ships
    spans when enabled; ``ui_port`` and ``ui_host`` are where the Phoenix UI answers:
    the controller's loopback, the address the launch publishes on.
    """
    settings = store.read()
    running = False
    if settings.enabled and getattr(bridge, "available", False):
        try:
            running = bool(bridge.phoenix_status().get("running"))
        except Exception:  # noqa: BLE001 - a down bridge is "not running", not a crash
            running = False
    return {
        "enabled": bool(settings.enabled),
        "running": running,
        "otlp_endpoint": trace_endpoint(settings),
        "ui_port": PHOENIX_PORT,
        # W6-6: the address the UI answers on - the one the launch publishes
        # on - so the console says it from here rather than from a constant.
        "ui_host": PHOENIX_LOOPBACK_HOST,
    }
