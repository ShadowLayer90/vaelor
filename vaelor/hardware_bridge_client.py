"""The unprivileged client of the root hardware bridge, and its wire contract.

Moved out of `hardware_bridge` (VD-129) when the replica balancer's three
verbs took that module past the 1,000-line ceiling `CLAUDE.md` sets - the same
move `credential_broker_client` made for the credential broker, for the same
reason, and with the same shape: the server module re-exports every name here,
so its readers (`bridge_transport`, `gpu_serving_target`, the executor's
launchers, the tests that patch ``vaelor.hardware_bridge.HardwareBridgeClient``)
import what they always did.

The client is small and fails closed: one request per Unix-socket connection,
a JSON line each way, a frame bound the server shares, and every failure - a
missing socket, a timeout, an invalid reply, a refused action - raised as
:class:`HardwareBridgeError`. Nothing here is validated on behalf of the root
side: the boundary that matters is the one that spawns a process or writes a
root-owned file, and that is the server's (LESSONS #178).
"""

from __future__ import annotations

import json
import socket
from pathlib import Path
from typing import Any, Optional

from .host_power import POWER_ACTION_REFUSAL
from .runtime_paths import env_value


SOCKET_PATH = env_value(
    "VAELOR_HARDWARE_BRIDGE_SOCKET",
    "PM_HARDWARE_BRIDGE_SOCKET",
    "/run/vaelor/hardwared.sock",
)
MAX_REQUEST_BYTES = 64 * 1024

#: How long an allowlisted ``run_argv`` may run before the bridge kills it, and
#: the ceiling a caller may ask for. The image pull is the long pole (the vLLM
#: container is multi-GB), so the ceiling matches the hour the SSH path already
#: gives that one command; the default is the ordinary systemd/cat round trip.
RUN_ARGV_TIMEOUT_SECONDS = 120
RUN_ARGV_MAX_TIMEOUT_SECONDS = 3600

#: How many BYTES of a command's stdout cross the socket. `SshTransport.run`
#: bounds its own read the same way (1 MiB); the smaller bound here is set by
#: this socket's :data:`MAX_REQUEST_BYTES` frame, which the JSON reply must fit
#: inside. Every command whose OUTPUT is parsed - ``cat`` of a group file, a
#: routing table or a progress JSON, ``systemctl show``, ``sha256sum`` - is far
#: under it. Hitting it is REPORTED (``truncated``) rather than left to look
#: like a short answer, because a silently cut group file or routing table
#: parses cleanly into the wrong facts.
RUN_ARGV_MAX_OUTPUT_BYTES = 32 * 1024

#: How long the socket read may block for an on-demand serving profile (VD-128).
#: The default 3 s client timeout fits the quick status verbs, but a profile runs
#: two bounded ``perf`` steps and an ``amd-smi`` snapshot back to back on the
#: privileged side; this ceiling comfortably outlives their summed hard timeouts
#: so the reply is not cut off mid-capture, while still bounding a wedged bridge.
RUN_PROFILE_SOCKET_TIMEOUT = 120

#: How long a client waits for ``agent_start`` / ``agent_rekey``. The bridge runs
#: ``systemctl restart`` and replaces the gate container under its lock (a gate
#: image pull, when one is needed, runs before the lock); the 3 s default cut a
#: slow launch off on the client while the bridge finished it, so a relaunch or
#: re-key read as failed. The reconcile re-reads the live status on its next
#: pass either way, so this bounds a wedged bridge rather than deciding truth.
AGENT_LAUNCH_SOCKET_TIMEOUT = 120

#: How long a client waits for ``proxy_start`` to answer. The bridge replaces
#: the gate under its own lock - image inspect, liveness inspect, ``docker
#: stop`` and ``rm`` of the old container, ``docker run`` of the new - which
#: `vaelor.llm_server_proxy` bounds at 4 x 30 s + 2 x 120 s = 360 s; this adds
#: a margin. The 3 s default let a slow start time out HERE while it went on to
#: completion THERE, after a later start had already landed - so the gate ran
#: the older key set while the executor believed the newer one (2026-09-28
#: review). A first-time image pull is not covered; a client that times out on
#: one is corrected by the next reconcile, which compares the key set the
#: running gate reports, not the one the caller last asked for.
PROXY_START_SOCKET_TIMEOUT = 390

#: The same for ``proxy_stop``: liveness inspect, ``docker stop`` (bounded at
#: 120 s) and ``rm`` - 180 s bridge-side, plus a margin.
PROXY_STOP_SOCKET_TIMEOUT = 210

#: How long a client waits for ``managed_unit_install`` (VD-143). Bridge-side the
#: install is three file operations and up to three ``systemctl`` calls
#: (``daemon-reload``, then ``enable --now`` - or ``reset-failed`` and ``start
#: --no-block`` for a pull), each bounded at :data:`RUN_ARGV_TIMEOUT_SECONDS`;
#: this adds a margin, so a slow reload is reported as its own failure rather
#: than as an unavailable bridge.
UNIT_INSTALL_SOCKET_TIMEOUT = 3 * RUN_ARGV_TIMEOUT_SECONDS + 30

#: ``model_cache_remove`` walks and unlinks one repo's weights, which for a
#: large model is many files. It is idempotent, so a client that gives up first
#: is corrected by the next removal, but the ordinary case should be answered.
CACHE_REMOVE_SOCKET_TIMEOUT = 600

#: ``pull_progress_read`` is one bounded file read.
PROGRESS_READ_SOCKET_TIMEOUT = 30
#: The replica balancer IS the proxy's launch (`gpu_pool_replicas.BalancerProcess`
#: inherits `_launch` and `stop`), so its verbs wait exactly as long (ACC-067):
#: at 3 s a restore after a bridge restart timed out here while it went on to
#: completion there, and the bridge wrote its answer into a closed socket.
BALANCER_START_SOCKET_TIMEOUT = PROXY_START_SOCKET_TIMEOUT
BALANCER_STOP_SOCKET_TIMEOUT = PROXY_STOP_SOCKET_TIMEOUT
#: **That wait is held under the GPU serving mutex** (review S8). The Mode B
#: reconcile converges the balancer inside `ClusterModeSwitch.reconcile`, which
#: holds ``watch_lock``, so a start that runs to its bound keeps the watch pass,
#: and every verb waiting on that mutex (enter, leave, repoint, a health write),
#: waiting up to 390 s with it. Accepted: an image already present restores in
#: seconds; the bound is a cold pull's. The 3 s it replaced gave up while the
#: bridge went on, so the next pass started a second restore behind the first.

#: ``phoenix_start`` bridge-side: image inspect, liveness inspect, the stop's
#: own liveness inspect, ``docker stop`` (bounded at 120 s) and ``rm`` of the
#: old collector, ``docker run`` of the new one and the status read -
#: `vaelor.phoenix_service` bounds those at 5 x 30 s + 2 x 120 s = 390 s; this
#: adds a margin. A first-time image pull
#: is not covered: a client that times out on one is corrected by the next
#: reconcile, which reads the collector's live status (ACC-067).
PHOENIX_START_SOCKET_TIMEOUT = 420

#: ``phoenix_stop``: liveness inspect, ``docker stop`` (120 s) and ``rm``.
PHOENIX_STOP_SOCKET_TIMEOUT = 210

AF_UNIX = getattr(socket, "AF_UNIX", -1)


#: How long a GPU memory pool change may take to answer: the root side
#: rebuilds the boot image before it replies, and rebuilds it a second
#: time to roll a failed change back (its own ceiling is 1020 s:
#: `gpu_memory_pool_apply.APPLY_TIMEOUT_SECONDS`, which a test compares).
GPU_MEMORY_POOL_SOCKET_TIMEOUT = 1050.0


class HardwareBridgeError(RuntimeError):
    """Raised when the optional Pironman hardware bridge is unavailable."""



class HardwareBridgeClient:
    """Small fail-closed client used by the unprivileged control plane."""

    def __init__(self, socket_path: str = SOCKET_PATH, timeout: float = 3.0):
        self.socket_path = socket_path
        self.timeout = timeout

    @property
    def available(self) -> bool:
        return Path(self.socket_path).is_socket()

    def _request(
        self,
        action: str,
        payload: dict[str, Any] | None = None,
        *,
        socket_timeout: float | None = None,
    ) -> Any:
        request = {"action": action, "payload": payload or {}}
        encoded = (json.dumps(request, separators=(",", ":")) + "\n").encode()
        if len(encoded) > MAX_REQUEST_BYTES:
            raise HardwareBridgeError("Hardware request is too large.")
        try:
            with socket.socket(AF_UNIX, socket.SOCK_STREAM) as connection:
                # A per-call override lets one long verb (the on-demand profile)
                # wait out its bounded capture without widening the quick verbs'
                # 3 s default.
                connection.settimeout(socket_timeout or self.timeout)
                connection.connect(self.socket_path)
                connection.sendall(encoded)
                response = b""
                while not response.endswith(b"\n"):
                    chunk = connection.recv(8192)
                    if not chunk:
                        break
                    response += chunk
                    if len(response) > MAX_REQUEST_BYTES:
                        raise HardwareBridgeError("Hardware response is too large.")
        except (OSError, TimeoutError) as error:
            raise HardwareBridgeError("Pironman hardware service is unavailable.") from error
        try:
            decoded = json.loads(response.decode("utf-8"))
        except (UnicodeDecodeError, json.JSONDecodeError) as error:
            raise HardwareBridgeError("Pironman hardware service returned invalid data.") from error
        if not decoded.get("ok"):
            raise HardwareBridgeError(
                str(decoded.get("error") or "Pironman hardware request failed.")
            )
        return decoded.get("data")

    def snapshot(self) -> dict[str, Any]:
        value = self._request("snapshot")
        return value if isinstance(value, dict) else {}

    def device_info(self) -> dict[str, Any]:
        return dict(self.snapshot().get("device_info") or {})

    def current_data(self) -> dict[str, Any]:
        return dict(self.snapshot().get("data") or {})

    def read_config(self) -> dict[str, Any]:
        return dict(self.snapshot().get("config") or {})

    def update_config(self, patch: dict[str, Any]) -> dict[str, Any]:
        if not isinstance(patch, dict) or not isinstance(patch.get("system"), dict):
            raise ValueError("Hardware configuration must contain a system object.")
        value = self._request("update_config", patch)
        return value if isinstance(value, dict) else {}

    def power(self, action: str) -> dict[str, Any]:
        if action not in {"restart_service", "reboot", "shutdown"}:
            raise ValueError(POWER_ACTION_REFUSAL)
        value = self._request("power", {"action": action})
        return value if isinstance(value, dict) else {}

    def package_power(self, interval_seconds: float = 2.0) -> dict[str, Any]:
        """Package watts, measured by the privileged side across two samples.

        RAPL counters are root-only, so this is the only account that can read
        them. Reporting package power as unavailable from the control plane
        described the reader, not the hardware.
        """
        value = self._request(
            "package_power", {"interval_seconds": interval_seconds}
        )
        return value if isinstance(value, dict) else {}

    def rapl_energy(self) -> dict[str, Any]:
        """The raw counters, with no sleep, for a caller measuring its own gap.

        :meth:`package_power` holds a bridge thread for the length of the
        interval it is given, which is fine for a one-shot report and wrong on
        a telemetry poll. This returns the counters as they read right now and
        leaves the division to the sampler on the other side.
        """
        value = self._request("rapl_energy")
        return value if isinstance(value, dict) else {}

    def memory_ecc(self) -> dict[str, Any]:
        """Whether the memory has ECC, from EDAC and the SMBIOS memory records.

        Here for the same reason as RAPL: ``/sys/firmware/dmi/entries`` is mode
        0400 root, so the unprivileged control plane cannot tell "no ECC" from
        "not readable" and was publishing the wrong one of the two.
        """
        value = self._request("memory_ecc")
        return value if isinstance(value, dict) else {}

    def wmi_sensors(self) -> dict[str, Any]:
        """Fans and board temperatures from ``hp_wmi_sensors``, if loaded."""
        value = self._request("wmi_sensors")
        return value if isinstance(value, dict) else {}

    def drive_health(self) -> dict[str, Any]:
        """NVMe SMART wear, power-on hours, unsafe shutdowns and media errors."""
        value = self._request("drive_health")
        return value if isinstance(value, dict) else {}

    def flm_start(self, tag: str, ctx_len: int, port: int) -> dict[str, Any]:
        """Launch the flm-real NPU server as root (VD-001).

        The privileged side needs root for CAP_IPC_LOCK; this client holds none
        of it and only carries the request. The tag is validated at the root
        boundary (:mod:`vaelor.flm_service`), not here — a client-side check
        would be advisory, and the boundary that matters is the one that
        actually launches the process (LESSONS #178).
        """
        value = self._request(
            "flm_start",
            {"tag": str(tag), "ctx_len": int(ctx_len), "port": int(port)},
        )
        return value if isinstance(value, dict) else {}

    def flm_stop(self) -> dict[str, Any]:
        """Stop the flm-real NPU server (service stop only, never a reboot;
        VD-019)."""
        value = self._request("flm_stop")
        return value if isinstance(value, dict) else {}

    def flm_status(self) -> dict[str, Any]:
        """Whether the flm-real NPU server is running, and on what tag/port."""
        value = self._request("flm_status")
        return value if isinstance(value, dict) else {}

    def flm_binary_present(self) -> dict[str, Any]:
        """Whether the flm-real binary is present, checked by the root bridge.

        The executor cannot traverse the binary's ``0770 root:root`` parent
        directories, so its own stat reads absent even on a box that serves
        (VD-001); this asks the privileged side, which can. A courier call like
        the rest — the stat that matters happens at the root boundary.
        """
        value = self._request("flm_binary_present")
        return value if isinstance(value, dict) else {}

    def flm_install_release(self, source_url: str, expected_sha256: str) -> dict[str, Any]:
        """Install a fine-tuned NPU model from a pinned release, as root.

        A fine-tune (ff-4b-h) is not in the snap's public flm catalog, so it is
        delivered as a release the appliance downloads, verifies and unpacks into
        the snap paths flm-real serves from. Those paths are root-owned and this
        client holds no root, so the privileged side does the download-and-unpack
        and verifies the bytes against ``expected_sha256`` - which the caller
        pins from the model-catalog entry, the same trust model as the flm tag it
        passes to :meth:`flm_start`.
        """
        value = self._request(
            "flm_install_release",
            {"source_url": str(source_url), "expected_sha256": str(expected_sha256)},
        )
        return value if isinstance(value, dict) else {}

    def gpu_start(
        self, model_path: str, port: int, runtime: dict[str, Any] | None
    ) -> dict[str, Any]:
        """Launch the ROCmFPX GPU model server through the root bridge.

        Same reason the NPU goes through here (:meth:`flm_start`), a different
        privilege: the GPU server needs the bridge unit's /dev/dri + /dev/kfd
        access, its writable ``/var/log/vaelor`` and its lack of
        ``MemoryDenyWriteExecute`` — none of which the sandboxed workload
        executor has (``PrivateDevices=true`` hides the GPU, and its
        ``/var/log/vaelor`` is read-only). The model path and port are validated
        at the root boundary inside :mod:`vaelor.gpu_rocmfpx_service`, not here —
        the payload is carried through so the one place the rule lives is the one
        that launches the process (LESSONS #178).
        """
        value = self._request(
            "gpu_start",
            {"model_path": model_path, "port": port, "runtime": runtime},
        )
        return value if isinstance(value, dict) else {}

    def gpu_stop(self) -> dict[str, Any]:
        """Stop the GPU model server (service stop only, never a reboot)."""
        value = self._request("gpu_stop")
        return value if isinstance(value, dict) else {}

    def gpu_status(self) -> dict[str, Any]:
        """Whether the GPU model server is running, and on what model/port."""
        value = self._request("gpu_status")
        return value if isinstance(value, dict) else {}

    def proxy_start(
        self, listen_host: str, listen_port: int, model_port: Optional[int],
        api_keys: Any, wake_door: bool = False, unloaded_notice: bool = False,
        loading: bool = False,
    ) -> dict[str, Any]:
        """Launch the LLM Server auth proxy (nginx) through the root bridge.

        The LAN gate in front of the loopback GPU model: it runs ``--network host``
        so it can reach the model on ``127.0.0.1:<model_port>``, and enforces
        ``Authorization: Bearer <api_key>``. It is launched here, not from the
        executor, for the same reason the model is (:meth:`gpu_start`): the executor
        sandbox cannot ``docker run`` a host-network container that binds a
        privileged-adjacent LAN port and writes a root-owned config. The key is
        written into a root-owned config FILE at the root boundary inside
        :mod:`vaelor.llm_server_proxy` and never placed on any argv (LESSONS #178);
        it is carried through here unaltered, the same trust model as the model
        runtime that crosses this bridge.
        """
        payload = {
            "listen_host": listen_host, "listen_port": listen_port,
            "model_port": model_port, "api_keys": list(api_keys),
        }
        if wake_door:  # ACC-058: front the control plane's wake responder
            payload["wake_door"] = True
        if unloaded_notice:  # VD-159: front nothing; the door answers itself
            payload["unloaded_notice"] = True
        if loading:  # ACC-193: ...and says the model is loading
            payload["loading"] = True
        value = self._request(
            "proxy_start", payload, socket_timeout=PROXY_START_SOCKET_TIMEOUT,
        )
        return value if isinstance(value, dict) else {}

    def proxy_stop(self) -> dict[str, Any]:
        """Stop the LLM Server auth proxy and remove its key-bearing config."""
        value = self._request("proxy_stop", socket_timeout=PROXY_STOP_SOCKET_TIMEOUT)
        return value if isinstance(value, dict) else {}

    def proxy_status(self) -> dict[str, Any]:
        """Whether the LLM Server auth proxy is running, and on what binding."""
        value = self._request("proxy_status")
        return value if isinstance(value, dict) else {}

    def phoenix_start(self, port: int, bind_host: str) -> dict[str, Any]:
        """Launch the Phoenix trace collector (VD-128) through the root bridge.

        The OTLP trace collector the inference gateway ships spans to. It is
        launched here, not from the executor, for the same reason the model and
        the auth proxy are (:meth:`proxy_start`): the executor sandbox cannot
        ``docker run`` a container that publishes a host port. The loopback-only
        publish is enforced at the root boundary inside
        :mod:`vaelor.phoenix_service`; the payload is carried through unaltered
        (LESSONS #178).
        """
        value = self._request(
            "phoenix_start", {"port": port, "bind_host": bind_host},
            socket_timeout=PHOENIX_START_SOCKET_TIMEOUT,
        )
        return value if isinstance(value, dict) else {}

    def phoenix_stop(self) -> dict[str, Any]:
        """Stop the Phoenix trace collector and remove it."""
        value = self._request(
            "phoenix_stop", socket_timeout=PHOENIX_STOP_SOCKET_TIMEOUT
        )
        return value if isinstance(value, dict) else {}

    def phoenix_status(self) -> dict[str, Any]:
        """Whether the Phoenix trace collector is running, and on what binding."""
        value = self._request("phoenix_status")
        return value if isinstance(value, dict) else {}

    def agent_start(
        self, *, name: str, port: Any, config_content: str, gate: dict[str, Any]
    ) -> dict[str, Any]:
        """Launch one deployed cluster agent through the root bridge.

        The agent tier's twin of :meth:`phoenix_start`: the fixed ExecStart, the
        loopback runtime and the inbound single-key gate are the root boundary's
        (:mod:`vaelor.agent_service`). The 0600 ``config_content`` and the ``gate``
        dict carry the model key and the ``vsk_`` door key over the socket to that
        boundary unaltered - they are placed on no argv (LESSONS #178).
        """
        value = self._request(
            "agent_start",
            {
                "name": str(name), "port": port,
                "config_content": str(config_content or ""), "gate": gate,
            },
            socket_timeout=AGENT_LAUNCH_SOCKET_TIMEOUT,
        )
        return value if isinstance(value, dict) else {}

    def agent_stop(self, name: str) -> dict[str, Any]:
        """Stop one cluster agent unit and gate and unlink its 0600 configs."""
        value = self._request("agent_stop", {"name": str(name)})
        return value if isinstance(value, dict) else {}

    def agent_rekey(
        self, *, name: str, port: Any, gate: dict[str, Any],
        socket_timeout: float = AGENT_LAUNCH_SOCKET_TIMEOUT,
    ) -> dict[str, Any]:
        """Replace one agent's gate with its CURRENT key set; the runtime keeps serving.

        The ``gate`` dict carries the listen host and port and the plaintext
        ``api_keys`` over the socket to the root boundary unaltered, on no argv
        (LESSONS #178). An empty key list stops the gate: no keyless door.
        """
        value = self._request(
            "agent_rekey", {"name": str(name), "port": port, "gate": gate},
            socket_timeout=socket_timeout,
        )
        return value if isinstance(value, dict) else {}

    def agent_status(self, name: str) -> dict[str, Any]:
        """Whether one agent's unit and gate run, and the key-set hash and
        surface digest they run with (never a key)."""
        value = self._request("agent_status", {"name": str(name)})
        return value if isinstance(value, dict) else {}

    def balancer_start(
        self, name: str, port: int, upstreams: list[str], api_key: str
    ) -> dict[str, Any]:
        """Launch a replicated deployment's balancer (nginx) through the root bridge.

        The throughput intent's endpoint (VD-129): the second nginx container
        the controller runs, shaped exactly like :meth:`proxy_start` - the
        config is rendered root-side under ``/run/vaelor`` at ``0600`` inside
        :mod:`vaelor.gpu_pool_replicas`, mounted read-only, and the key is
        carried in the payload to that boundary and placed on no argv. The
        deployment ``name`` picks the container (``vaelor-vllm-<name>-balancer``)
        and is validated root-side by the one name rule.
        """
        value = self._request(
            "balancer_start",
            {
                "name": name, "port": port,
                "upstreams": [str(item) for item in upstreams],
                "api_key": api_key,
            },
            socket_timeout=BALANCER_START_SOCKET_TIMEOUT,
        )
        return value if isinstance(value, dict) else {}

    def balancer_stop(self, name: str) -> dict[str, Any]:
        """Stop a deployment's balancer and remove its key-bearing config."""
        value = self._request(
            "balancer_stop", {"name": name},
            socket_timeout=BALANCER_STOP_SOCKET_TIMEOUT,
        )
        return value if isinstance(value, dict) else {}

    def balancer_status(self, name: str) -> dict[str, Any]:
        """Whether a deployment's balancer is running, on what port, pooling whom."""
        value = self._request("balancer_status", {"name": name})
        return value if isinstance(value, dict) else {}

    def run_argv(
        self,
        argv: list[str],
        *,
        stdin_text: str = "",
        timeout: int | None = None,
    ) -> dict[str, Any]:
        """Run one reviewed argv on THIS machine, as root, and return stdout.

        The controller's half of the GPU cluster. A GPU deploy talks to every
        participating node through a transport that writes systemd units, runs
        ``docker`` and reads a handful of files; on an enrolled worker that is
        SSH, and on the controller itself it is this - the privilege boundary
        the appliance already has. :class:`vaelor.bridge_transport.BridgeTransport`
        is the adapter that makes the two look identical to the runtime.

        There is no ``sudo`` step and no ``sudo`` flag, exactly as there is none
        for :meth:`gpu_start`: the bridge IS the privilege boundary, so the
        command runs as this service's own user (root) or not at all. What may
        be run is bounded instead by :mod:`vaelor.bridge_argv_policy` - the one
        home of that rule - and the check that matters is the ROOT one on the
        far side of this socket, not this client's.
        """
        payload: dict[str, Any] = {
            "argv": [str(item) for item in argv],
            "stdin_text": str(stdin_text),
        }
        if timeout is not None:
            payload["timeout"] = int(timeout)
        value = self._request("run_argv", payload)
        return value if isinstance(value, dict) else {}

    def managed_unit_install(
        self, kind: str, values: dict[str, Any],
    ) -> dict[str, Any]:
        """Have the root side render, write and start one managed GPU unit.

        VD-143. Only the unit's KIND and its typed VALUES cross the socket -
        never unit text - and the root side renders the unit from the GPU
        runtime's own template, refusing a kind or a value name that template
        does not take.
        """
        value = self._request(
            "managed_unit_install", {"kind": str(kind), "values": dict(values)},
            socket_timeout=UNIT_INSTALL_SOCKET_TIMEOUT,
        )
        return value if isinstance(value, dict) else {}

    def ray_plane_prepare(
        self, name: str, token: str, link: str, address: str, peers: list[str],
        shared: bool = False,
    ) -> dict[str, Any]:
        """Have the root side guard this controller's part of a split (ACC-163).

        A name, the token and IPv4 addresses cross the socket; the root side
        renders the firewall from its fixed template and writes both files.
        """
        value = self._request(
            "ray_plane_fence",
            {"name": str(name), "token": str(token), "link": str(link),
             "address": str(address), "peers": [str(peer) for peer in peers],
             "shared": bool(shared)},
            socket_timeout=UNIT_INSTALL_SOCKET_TIMEOUT,
        )
        return value if isinstance(value, dict) else {}

    def ray_container_cgroup(self, name: str, role: str) -> dict[str, Any]:
        """Where one of a split's Ray containers on this controller runs (ACC-187)."""
        value = self._request(
            "ray_container_cgroup", {"name": str(name), "role": str(role)},
            socket_timeout=UNIT_INSTALL_SOCKET_TIMEOUT,
        )
        return value if isinstance(value, dict) else {}

    def ray_plane_clear(self, name: str) -> dict[str, Any]:
        """Have the root side remove this controller's part of a split's Ray plane."""
        value = self._request(
            "ray_plane_clear", {"name": str(name)},
            socket_timeout=UNIT_INSTALL_SOCKET_TIMEOUT,
        )
        return value if isinstance(value, dict) else {}

    def pull_progress_read(self, repo: str) -> dict[str, Any]:
        """``{"text": ...}``: a model pull's progress record, read root-side.

        The root side opens it without following a link anywhere on the path,
        so a link planted in the model store is never a root read of another
        file (VD-143).
        """
        value = self._request(
            "pull_progress_read", {"repo": str(repo)},
            socket_timeout=PROGRESS_READ_SOCKET_TIMEOUT,
        )
        return value if isinstance(value, dict) else {}

    def model_cache_remove(self, repo: str) -> dict[str, Any]:
        """``{"path": ...}``: remove one repo's cached weights, root-side, link-safe."""
        value = self._request(
            "model_cache_remove", {"repo": str(repo)},
            socket_timeout=CACHE_REMOVE_SOCKET_TIMEOUT,
        )
        return value if isinstance(value, dict) else {}

    def gpu_memory_pool_set(self, size_gib: Any) -> dict[str, Any]:
        """Set this machine's GPU memory pool from its next restart (VD-161).

        One whole number of GiB crosses the socket, unaltered: the root
        side decides whether it is a whole number and whether it is
        inside this machine's bounds, then writes its own fixed file and
        rebuilds the boot image. The reply is the pool's new status. The
        socket waits out the rebuild, which takes a minute or more.
        """
        value = self._request(
            "gpu_memory_pool_set", {"size_gib": size_gib},
            socket_timeout=GPU_MEMORY_POOL_SOCKET_TIMEOUT,
        )
        return value if isinstance(value, dict) else {}

    def gpu_memory_pool_revert(self) -> dict[str, Any]:
        """Remove Vaelor's GPU memory pool file; the reply is the new status."""
        value = self._request(
            "gpu_memory_pool_revert", {},
            socket_timeout=GPU_MEMORY_POOL_SOCKET_TIMEOUT,
        )
        return value if isinstance(value, dict) else {}

    def run_serving_profile(self, seconds: int | None = None) -> dict[str, Any]:
        """Run one bounded on-demand profile of the GPU serving process, as root.

        VD-128 §6b. The privileged side finds the serving container's PID from
        ``docker inspect`` (never a caller-supplied one), probes which captures can
        run, executes the bounded ``perf`` / ``amd-smi`` captures the probe allows,
        and returns the structured result with an honest reason for every capture
        that could not run. This client only carries the capture *window*; the PID,
        the tool paths and the paranoid gate are all read at the root boundary
        (LESSONS #178). The socket is given a longer read timeout because the
        capture runs for several seconds before the one reply comes back.
        """
        payload: dict[str, Any] = {}
        if seconds is not None:
            payload["seconds"] = int(seconds)
        value = self._request(
            "run_serving_profile", payload, socket_timeout=RUN_PROFILE_SOCKET_TIMEOUT
        )
        return value if isinstance(value, dict) else {}
