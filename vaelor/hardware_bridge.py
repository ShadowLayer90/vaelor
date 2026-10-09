"""Narrow Unix-socket bridge to the preserved Pironman hardware runtime."""

from __future__ import annotations

import json
import os
import signal
import socketserver
import sys
import threading
import types
from dataclasses import asdict, is_dataclass
from pathlib import Path
from typing import Any

from .hardware_bridge_agents import AGENT_ACTIONS, AgentBridgeVerbs, dispatch_agent_action
from .hardware_bridge_host_settings import (
    HOST_SETTING_ACTIONS, HostSettingVerbs, dispatch_host_setting,
)
from .bridge_peers import BRIDGE_SOCKET_GROUP, admit, admitted_uids, socket_group
from .state_root_layout import start_layout_unit
from .hardware_bridge_profile import ServingProfileMixin
from .host_power import POWER_ACTION_REFUSAL
from .hardware_bridge_client import (  # noqa: F401 - the wire contract, re-exported
    AF_UNIX,
    MAX_REQUEST_BYTES,
    RUN_ARGV_MAX_OUTPUT_BYTES,
    RUN_ARGV_MAX_TIMEOUT_SECONDS,
    RUN_ARGV_TIMEOUT_SECONDS,
    SOCKET_PATH,
    HardwareBridgeClient,
    HardwareBridgeError,
)
from .hardware_bridge_commands import CONTROLLER_UNIT_VERBS, ControllerCommandsMixin

PIRONMAN_CONFIG = Path("/opt/pironman5/config.json")
PIRONMAN_VENV = Path("/opt/pironman5/venv")

#: What the three balancer verbs (VD-129) answer to a payload that is not an
#: object - one sentence for the three, because they take one payload shape.
_INVALID_BALANCER_REQUEST = "Invalid replica balancer request."

#: What a caller is told when the root side hit an operating-system error: the
#: detail - a path, an errno - is written to the journal instead.
_OS_REFUSAL = (
    "The hardware bridge could not complete that on this machine; its journal "
    "has the detail."
)


def _json_safe(value: Any) -> Any:
    if value is None or isinstance(value, (bool, int, float, str)):
        return value
    if is_dataclass(value) and not isinstance(value, type):
        return _json_safe(asdict(value))
    if hasattr(value, "_asdict"):
        return _json_safe(value._asdict())
    if isinstance(value, dict):
        return {str(key): _json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple, set)):
        return [_json_safe(item) for item in value]
    if hasattr(value, "__dict__"):
        return {
            str(key): _json_safe(item)
            for key, item in vars(value).items()
            if not str(key).startswith("_")
        }
    return str(value)


#: Guards the creation of a runtime's per-family locks (:meth:`_family_lock`).
_FAMILY_LOCKS_GUARD = threading.Lock()
#: How many family locks one runtime holds at most: ``construct``, ``phoenix``
#: and one per replicated deployment's balancer, far more than a box runs.
MAX_FAMILY_LOCKS = 128


class _HardwareRuntime(
    ServingProfileMixin, ControllerCommandsMixin, AgentBridgeVerbs, HostSettingVerbs,
):
    """Own PMAuto without starting the legacy web dashboard.

    **Two kinds of lock (ACC-067).** ``self._lock`` is VD-001's: it serialises
    the verbs that put a model on an accelerator (NPU, GPU) or change what
    systemd runs, and the LLM Server gate that fronts them. Everything else a
    verb touches - the lazy construction of a supervised process, the Phoenix
    collector, one deployment's balancer - takes only its OWN family lock
    (:meth:`_family_lock`). Restoring Phoenix or a balancer after a bridge
    restart runs ``docker stop``/``run`` for seconds to minutes; under the
    shared lock that froze every status read behind it, so callers timed out
    after 3 s and the bridge wrote its answer into a closed socket.
    """

    def _family_lock(self, family: str) -> threading.RLock:
        """The lock one family of verbs serialises on, created on first use.

        Created lazily (and under a module guard, so two threads cannot mint
        two) rather than in ``__init__``, because a runtime built without the
        enclosure returns from ``__init__`` early and tests build one with
        ``__new__``. A family is a container the bridge supervises on its own:
        ``construct``, ``phoenix`` and ``balancer:<name>`` - the name validated
        before a lock is minted for it (:meth:`_balancer_lock`), and at most
        :data:`MAX_FAMILY_LOCKS` of them.
        """
        with _FAMILY_LOCKS_GUARD:
            locks = self.__dict__.setdefault("_family_locks", {})
            if family not in locks and len(locks) >= MAX_FAMILY_LOCKS:
                raise ValueError("The hardware bridge supervises too many replica balancers.")
            return locks.setdefault(family, threading.RLock())

    def _balancer_lock(self, name: Any) -> threading.RLock:
        """One deployment's balancer family lock, for a VALID deployment name only."""
        from .gpu_pool_units import deployment_name

        return self._family_lock("balancer:" + deployment_name(name))

    def __init__(self) -> None:
        self._lock = threading.RLock()
        # An enclosure is optional. The bridge's other job — holding the
        # privilege needed to power the host — applies to every machine, and
        # refusing to start without a Pironman is what left an x86 workstation
        # with no way to reboot from its own control plane.
        try:
            self.appliance = self._load_appliance()
        except (ImportError, OSError, RuntimeError, ValueError) as error:
            self.appliance = None
            self.enclosure_error = str(error)
            self.device_info = {}
            return
        self.enclosure_error = ""
        self.device_info = {
            "name": self.appliance.pm_auto.device_info.get(
                "name", "Pironman appliance"
            ),
            "id": self.appliance.pm_auto.device_info.get("id", "pironman5"),
            "peripherals": list(self.appliance.peripherals),
            "version": self.appliance.pm_auto.device_info.get("version", "unknown"),
            "app_name": "vaelor-hardware",
            "config_path": str(PIRONMAN_CONFIG),
        }

    @staticmethod
    def _load_appliance():
        site_packages = sorted(PIRONMAN_VENV.glob("lib/python*/site-packages"))
        if not site_packages:
            raise RuntimeError("The Pironman hardware runtime is not installed.")
        for location in reversed(site_packages):
            sys.path.insert(1, str(location))
        # SunFounder's venv intentionally includes Debian's hardware bindings
        # (for example python3-lgpio) from the system dist-packages directory.
        system_packages = Path("/usr/lib/python3/dist-packages")
        if system_packages.is_dir():
            sys.path.insert(1, str(system_packages))
        dashboard_stub = types.ModuleType("pm_dashboard.pm_dashboard")
        dashboard_stub.PMDashboard = None
        sys.modules["pm_dashboard.pm_dashboard"] = dashboard_stub
        from pironman5 import pironman5 as appliance_module

        # The Vaelor process owns the only web listener. The stub above prevents
        # the legacy package from composing another control plane while this
        # service loads PMAuto and its GPIO/I2C/SPI addons.
        return appliance_module.Pironman5(config_path=str(PIRONMAN_CONFIG))

    def _require_enclosure(self) -> None:
        if self.appliance is None:
            raise RuntimeError(
                self.enclosure_error
                or "No enclosure hardware is fitted to this machine."
            )

    def start(self) -> None:
        if self.appliance is None:
            return
        self.appliance.pm_auto.start()

    def stop(self) -> None:
        if self.appliance is None:
            return
        self.appliance.pm_auto.stop()

    def snapshot(self) -> dict[str, Any]:
        with self._lock:
            self._require_enclosure()
            return {
                "device_info": dict(self.device_info),
                "config": _json_safe(self.appliance.read_config()),
                "data": _json_safe(self.appliance.pm_auto.read() or {}),
            }

    def update_config(self, patch: dict[str, Any]) -> dict[str, Any]:
        with self._lock:
            self._require_enclosure()
            return self.appliance.update_config(patch)

    def power(self, action: str) -> dict[str, Any]:
        """Perform a host power action with the privilege this service holds.

        The mechanism comes from the platform driver through systemd-logind on
        every host, Pironman included: this service already runs as root, so
        ``systemctl reboot``/``poweroff`` work directly, and the enclosure's
        ``sf_rpi_status`` sudo wrappers cannot run under the unit's
        ``NoNewPrivileges=yes`` (#208, VD-098). The bridge supplies the
        privilege; the seam supplies the mechanism.
        """
        if action not in {"restart_service", "reboot", "shutdown"}:
            raise ValueError(POWER_ACTION_REFUSAL)
        record = self._power_actions().get(action) or {}
        run = record.get("run")
        if run is None:
            raise RuntimeError(
                record.get("reason")
                or "This host does not provide the requested power action."
            )
        result = run()
        return result if isinstance(result, dict) else {
            "action": action, "scheduled": True
        }

    def package_power(self, interval_seconds: float = 2.0) -> dict[str, Any]:
        """Read RAPL and return watts, measured across two samples.

        Lives here because the counters are root-only and this service is the
        thing that runs as root. The control plane reported package power as
        unavailable on every x86 host, which described the account it was
        asking as rather than the machine.

        No enclosure requirement: RAPL is a processor feature, and gating it on
        a Pironman would repeat the mistake this whole port exists to correct.
        """
        from .package_power import package_power

        return package_power(interval_seconds=interval_seconds)

    def rapl_energy(self) -> dict[str, Any]:
        """The counters as they read now. No sleep, so no thread is held open.

        Same privilege, same absence of an enclosure requirement as
        :meth:`package_power`; the difference is only who measures the
        interval. A poll loop measures its own and must not be blocked while
        this side measures one for it.
        """
        from .package_power import domains

        return {"domains": domains()}

    def memory_ecc(self) -> dict[str, Any]:
        """ECC state from EDAC and the root-only SMBIOS memory records."""
        from .memory_ecc import memory_ecc

        return memory_ecc()

    def wmi_sensors(self) -> dict[str, Any]:
        """Fans and board temperatures from ``hp_wmi_sensors``, if loaded."""
        from .wmi_sensors import read_sensors

        return read_sensors()

    def drive_health(self) -> dict[str, Any]:
        """NVMe SMART, read through the admin-passthrough ioctl.

        Here for the same reason as RAPL: the ioctl needs CAP_SYS_ADMIN and
        this service has it. No `nvme-cli` dependency - requiring a package to
        be installed before an appliance can say its drive is wearing out makes
        the answer depend on a fact about the host rather than the hardware.
        """
        from .nvme_smart import drive_health

        return drive_health()

    @staticmethod
    def _power_actions() -> dict[str, dict[str, Any]]:
        from .platforms import select_hardware_platform

        return select_hardware_platform().power_actions()

    def _flm(self):
        """The supervised flm-real process, created on first use.

        Lazily imported and constructed so a host that never serves the NPU
        never touches :mod:`vaelor.flm_service`, and so the bridge starts on a
        machine where flm-real does not exist.
        """
        with self._family_lock("construct"):
            existing = getattr(self, "_flm_process", None)
            if existing is None:
                from .flm_service import FlmProcess

                existing = FlmProcess()
                self._flm_process = existing
            return existing

    def flm_start(self, tag: str, ctx_len: int, port: int) -> dict[str, Any]:
        """Launch flm-real as root (VD-001).

        The root child inherits CAP_IPC_LOCK to pin NPU pages. The tag is validated
        inside :mod:`vaelor.flm_service` against the installed allowlist and the tag
        grammar before entering a fixed argument vector: no shell, no interpolation,
        no ``PATH`` lookup (LESSONS #178). Held under ``self._lock`` for the whole op
        (VD-001): `FlmProcess.start` is a non-atomic check-then-act, so two threads
        unlocked could orphan a second flm-real pinning NPU pages; the reentrant
        RLock serializes it so only one is ever tracked.
        """
        with self._lock:
            return self._flm().start(str(tag), ctx_len=int(ctx_len), port=int(port))

    def flm_stop(self) -> dict[str, Any]:
        """Stop flm-real (a service stop, never a reboot, VD-019), under ``self._lock`` as :meth:`flm_start`."""
        with self._lock:
            return self._flm().stop()

    def flm_status(self) -> dict[str, Any]:
        """Whether flm-real is running, and on what tag and port."""
        return self._flm().status()

    def flm_binary_present(self) -> dict[str, Any]:
        """Whether the flm-real binary is present and launchable, checked as root.

        The truthful reader for the binary, exactly as this unit is for RAPL and
        ECC: the executor runs as ``vaelor-workloads`` and an Aug-24 snap refresh
        set the binary's parent directories (``.../bin/flm``, ``.../bin/flm/npu``)
        to ``0770 root:root``, so the unprivileged discovery cannot traverse them
        to stat the binary even though it is world-executable and this root unit
        launches it fine (VD-001). This service runs ``PrivateDevices=no`` as root
        and can, so :func:`vaelor.flm_service.discover_npu_serving` reads binary
        presence through here.

        A pure filesystem stat that touches no process state, so — unlike
        :meth:`flm_start`/:meth:`flm_stop` — it takes no lock and cannot contend
        with a launch.
        """
        from pathlib import Path

        from .flm_service import FLM_BINARY

        binary = Path(FLM_BINARY)
        try:
            present = binary.exists()
            executable = bool(present) and os.access(str(binary), os.X_OK)
        except OSError:
            present = executable = False
        return {"present": bool(present), "executable": bool(executable)}

    def flm_install_release(self, source_url: str, expected_sha256: str) -> dict[str, Any]:
        """Download, verify and unpack the on-device NPU model release, as root.

        Serialised with :meth:`flm_start` / :meth:`flm_stop` under ``self._lock``:
        the model files it replaces are the ones a concurrent launch would read,
        so an install must not interleave with a start. The fetch-and-unpack lives
        in :mod:`vaelor.flm_npu_release` — it lists the GitHub release, downloads
        the split model parts, reassembles and verifies them against the pinned
        sha256, and unpacks the model into ``FLM_MODEL_PATH/models`` (the exact
        artifact and directory ``deploy/fetch-npu-model.sh`` uses, so the UI
        install and the installer's first-boot fetch agree). The FLM **runtime**
        is provisioned separately by the installer; this artifact carries only the
        model, so once it lands the existing supervisor serves the tag with no
        other change - the turnkey equivalent of the manual place-model step, done
        as root behind the bridge.

        **Only the shipped release.** The pair the caller sends must be exactly
        a release this wheel's catalog pins (`model_catalog.pinned_release`),
        and the URL and digest used are then the catalog's: a caller cannot
        have root fetch, trust and unpack an archive of its own choosing by
        naming its own URL beside its own digest.
        """
        from pathlib import Path

        from . import flm_npu_release
        from .flm_service import FLM_MODEL_PATH
        from .model_catalog import pinned_release

        release = pinned_release(source_url, expected_sha256)
        if release is None:
            raise ValueError(
                "Only the on-device model release this appliance ships with can "
                "be installed."
            )
        with self._lock:
            # Stop any running flm-real before replacing the model files it may
            # have open. A no-op on a fresh box; on a re-install over a live model
            # it frees the files, and the deploy step relaunches flm-real after.
            self._flm().stop()
            return flm_npu_release.install(
                release["source_url"],
                Path(FLM_MODEL_PATH),
                expected_sha256=release["sha256"],
            )

    def _gpu_server(self):
        """The supervised ROCmFPX GPU server process, created on first use.

        Lazily imported and constructed like :meth:`_flm`, so a host that never
        serves the GPU tier never touches
        :mod:`vaelor.gpu_rocmfpx_service`, and so the bridge starts on a machine
        where the ROCmFPX fork does not exist.
        """
        with self._family_lock("construct"):
            existing = getattr(self, "_gpu_process", None)
            if existing is None:
                from .gpu_rocmfpx_service import GpuServerProcess, configured_model_cache_dir

                existing = GpuServerProcess(model_cache_dir=configured_model_cache_dir())
                self._gpu_process = existing
            return existing

    def gpu_start(
        self, model_path: Any, port: Any, runtime: Any
    ) -> dict[str, Any]:
        """Launch the ROCmFPX GPU model server as root.

        Launched here, not from the workload executor, for privileges only this
        unit holds: /dev/dri and /dev/kfd for HIP/Vulkan (the executor's
        ``PrivateDevices=true`` hides them), a writable ``/var/log/vaelor``, and no
        ``MemoryDenyWriteExecute`` for the fork's JIT. The model path and port are
        validated inside :mod:`vaelor.gpu_rocmfpx_service` before they enter a fixed
        argument vector: no shell, no interpolation (LESSONS #178). Held under
        ``self._lock`` for the whole op, as :meth:`flm_start` is (VD-001): the
        launch is a non-atomic check-then-act, so two threads unlocked could orphan
        a second server holding VRAM; the reentrant RLock serializes it.
        """
        with self._lock:
            return self._gpu_server().start(
                str(model_path), port=int(port), runtime=runtime
            )

    def gpu_stop(self) -> dict[str, Any]:
        """Stop the GPU model server. A service stop only, never a reboot.

        Under ``self._lock`` for the same reason as :meth:`gpu_start`: a stop
        racing a concurrent start must not interleave the check-then-act that
        tracks the single GPU process (VD-001).
        """
        with self._lock:
            return self._gpu_server().stop()

    def gpu_status(self) -> dict[str, Any]:
        """Whether the GPU model server is running, and on what model and port."""
        return self._gpu_server().status()

    def _proxy_server(self):
        """The supervised LLM Server auth-proxy process, created on first use.

        Lazily imported and constructed like :meth:`_gpu_server`, so a host that
        never enables the LLM Server never touches :mod:`vaelor.llm_server_proxy`,
        and so the bridge starts on a machine without nginx or docker.
        """
        with self._family_lock("construct"):
            existing = getattr(self, "_proxy_process", None)
            if existing is None:
                from .llm_server_proxy import LlmServerProxyProcess

                existing = LlmServerProxyProcess()
                self._proxy_process = existing
            return existing

    def proxy_start(
        self, listen_host: Any, listen_port: Any, model_port: Any, api_keys: Any,
        wake_door: bool = False, unloaded_notice: bool = False, loading: bool = False,
    ) -> dict[str, Any]:
        """Launch the LLM Server auth proxy as root.

        Held under ``self._lock`` for the whole op, as :meth:`gpu_start` (VD-001):
        the launch is a non-atomic check-then-act a toggle racing the reconcile
        could otherwise double. The key/coupling and every validation happen inside
        :mod:`vaelor.llm_server_proxy` at this root boundary, the payload carried
        through unaltered so the rule lives where the root-owned config is written
        (LESSONS #178).
        """
        with self._lock:
            return self._proxy_server().start(
                listen_host=str(listen_host), listen_port=int(listen_port),
                model_port=(
                    None if wake_door or unloaded_notice else int(model_port)
                ),
                api_keys=[str(key) for key in (api_keys or [])],
                wake_door=bool(wake_door),
                unloaded_notice=bool(unloaded_notice),
                loading=bool(loading),
            )

    def proxy_stop(self) -> dict[str, Any]:
        """Stop the LLM Server auth proxy and remove its config, under ``self._lock`` as :meth:`proxy_start`."""
        with self._lock:
            return self._proxy_server().stop()

    def proxy_status(self) -> dict[str, Any]:
        """Whether the LLM Server auth proxy is running, and on what binding."""
        return self._proxy_server().status()

    def _phoenix_server(self):
        """The supervised Phoenix trace-collector process, created on first use.

        Lazily imported and constructed like :meth:`_proxy_server`, so a host that
        never enables tracing never touches :mod:`vaelor.phoenix_service`, and so
        the bridge starts on a machine without docker or the Phoenix image.
        """
        with self._family_lock("construct"):
            existing = getattr(self, "_phoenix_process", None)
            if existing is None:
                from .phoenix_service import PhoenixServerProcess

                existing = PhoenixServerProcess()
                self._phoenix_process = existing
            return existing

    def phoenix_start(self, port: Any, bind_host: Any) -> dict[str, Any]:
        """Launch the Phoenix trace collector as root.

        The check-then-act tracking the single container is serialised on the
        ``phoenix`` family lock (ACC-067), NOT the shared model lock: the replace
        is a ``docker stop`` and ``run`` of a collector that holds no model, and
        holding the shared lock across it froze every other verb (and every
        status read) for its length. The loopback-only publish is enforced
        inside :mod:`vaelor.phoenix_service` (LESSONS #178). The image PULL runs
        first, outside any lock; it is idempotent, so the lock guards only the
        replace via ``skip_ensure``.
        """
        server = self._phoenix_server()
        server.ensure_image()
        with self._family_lock("phoenix"):
            return server.start(
                port=int(port), bind_host=str(bind_host), skip_ensure=True,
            )

    def phoenix_stop(self) -> dict[str, Any]:
        """Stop the Phoenix trace collector and remove it, on its own family lock as :meth:`phoenix_start`."""
        with self._family_lock("phoenix"):
            return self._phoenix_server().stop()

    def phoenix_status(self) -> dict[str, Any]:
        """Whether the Phoenix trace collector is running, and on what binding."""
        return self._phoenix_server().status()

    # The cluster-agent verbs (agent_start/stop/status/rekey) live in
    # `hardware_bridge_agents.AgentBridgeVerbs`, mixed in above.

    def _balancer_server(self, name: Any):
        """The supervised balancer process for one deployment, created on first use.

        Per deployment name, because the container is named for it
        (``vaelor-vllm-<name>-balancer``); lazily imported like
        :meth:`_proxy_server`, so a host that never replicates never touches
        :mod:`vaelor.gpu_pool_replicas`. The name is validated by the one
        deployment-name rule inside `BalancerProcess`, at this root boundary.
        """
        with self._family_lock("construct"):
            from .gpu_pool_replicas import BalancerProcess

            servers = getattr(self, "_balancer_processes", None)
            if servers is None:
                servers = {}
                self._balancer_processes = servers
            key = str(name)
            if key not in servers:
                servers[key] = BalancerProcess(key)
            return servers[key]

    def balancer_start(
        self, name: Any, port: Any, upstreams: Any, api_key: Any
    ) -> dict[str, Any]:
        """Launch a replicated deployment's balancer as root (VD-129).

        Serialised for the WHOLE op on this deployment's own family lock
        (ACC-067): the launch is a non-atomic check-then-act, and the mode
        reconcile converging the balancer must not race the deploy starting it
        - but the balancer holds no model, so the shared lock is not taken and
        a restore after a bridge restart no longer stalls every other verb.
        The config, the upstream pool and the key are validated inside
        :mod:`vaelor.gpu_pool_replicas` at this root boundary - the payload is
        carried through unaltered so the one place the rule lives is the one
        that writes the root-owned config (LESSONS #178).
        """
        if not isinstance(upstreams, list):
            raise ValueError("The balancer upstreams must be a list.")
        with self._balancer_lock(name):
            return self._balancer_server(name).start(
                port=int(port), upstreams=[str(item) for item in upstreams],
                api_key=str(api_key),
            )

    def balancer_stop(self, name: Any) -> dict[str, Any]:
        """Stop a deployment's balancer and remove its config. Service stop only."""
        with self._balancer_lock(name):
            return self._balancer_server(name).stop()

    def balancer_status(self, name: Any) -> dict[str, Any]:
        """Whether a deployment's balancer is running, on what port, pooling whom."""
        return self._balancer_server(name).status()



class _Handler(socketserver.StreamRequestHandler):
    def handle(self) -> None:
        request_bytes = self.rfile.readline(MAX_REQUEST_BYTES + 1)
        response: dict[str, Any]
        action: Any = None
        try:
            if not request_bytes or len(request_bytes) > MAX_REQUEST_BYTES:
                raise ValueError("Invalid hardware request size.")
            request = json.loads(request_bytes.decode("utf-8"))
            action = request.get("action")
            payload = request.get("payload") or {}
            runtime: _HardwareRuntime = self.server.runtime  # type: ignore[attr-defined]
            if action == "snapshot":
                data = runtime.snapshot()
            elif action == "update_config":
                if not isinstance(payload, dict):
                    raise ValueError("Invalid hardware configuration.")
                data = runtime.update_config(payload)
            elif action == "power":
                if not isinstance(payload, dict):
                    raise ValueError("Invalid power request.")
                data = runtime.power(str(payload.get("action") or ""))
            elif action == "package_power":
                if not isinstance(payload, dict):
                    raise ValueError("Invalid package power request.")
                # Bounded: this call sleeps for the interval it is given, and
                # an unbounded one would hold a bridge thread open for as long
                # as the caller liked.
                interval = payload.get("interval_seconds", 2.0)
                try:
                    interval = min(5.0, max(0.5, float(interval)))
                except (TypeError, ValueError):
                    interval = 2.0
                data = runtime.package_power(interval)
            elif action == "rapl_energy":
                data = runtime.rapl_energy()
            elif action == "memory_ecc":
                data = runtime.memory_ecc()
            elif action == "wmi_sensors":
                data = runtime.wmi_sensors()
            elif action == "drive_health":
                data = runtime.drive_health()
            elif action == "flm_start":
                if not isinstance(payload, dict):
                    raise ValueError("Invalid flm-real launch request.")
                # The tag validation that matters happens at the root boundary
                # inside `flm_service`; the payload is passed through unaltered
                # rather than pre-filtered here, so there is one place the rule
                # lives and it is the one that launches the process (#178).
                data = runtime.flm_start(
                    str(payload.get("tag") or ""),
                    payload.get("ctx_len"),
                    payload.get("port"),
                )
            elif action == "flm_stop":
                data = runtime.flm_stop()
            elif action == "flm_status":
                data = runtime.flm_status()
            elif action == "flm_binary_present":
                data = runtime.flm_binary_present()
            elif action == "flm_install_release":
                if not isinstance(payload, dict):
                    raise ValueError("Invalid model install request.")
                # The bytes are verified against the pinned digest inside
                # `flm_npu_release` at the root boundary, the same as flm_start's
                # tag: the payload is passed through unaltered so there is one
                # place the rule lives and it is the one that unpacks as root.
                data = runtime.flm_install_release(
                    str(payload.get("source_url") or ""),
                    str(payload.get("expected_sha256") or ""),
                )
            elif action == "gpu_start":
                if not isinstance(payload, dict):
                    raise ValueError("Invalid GPU model server launch request.")
                # As with flm_start, the path/port validation that matters
                # happens at the root boundary inside `gpu_rocmfpx_service`; the
                # payload is passed through unaltered so there is one place the
                # rule lives and it is the one that launches the process (#178).
                data = runtime.gpu_start(
                    payload.get("model_path"),
                    payload.get("port"),
                    payload.get("runtime"),
                )
            elif action == "gpu_stop":
                data = runtime.gpu_stop()
            elif action == "gpu_status":
                data = runtime.gpu_status()
            elif action == "proxy_start":
                if not isinstance(payload, dict):
                    raise ValueError("Invalid LLM Server proxy launch request.")
                # As with gpu_start, the key/coupling and validation that matter
                # happen at the root boundary inside `llm_server_proxy`; the payload
                # is passed through unaltered so there is one place the rule lives
                # and it is the one that writes the root-owned config (#178).
                data = runtime.proxy_start(
                    payload.get("listen_host"),
                    payload.get("listen_port"),
                    payload.get("model_port"),
                    payload.get("api_keys"),
                    wake_door=payload.get("wake_door") is True,
                    unloaded_notice=payload.get("unloaded_notice") is True,
                    loading=payload.get("loading") is True,
                )
            elif action == "proxy_stop":
                data = runtime.proxy_stop()
            elif action == "proxy_status":
                data = runtime.proxy_status()
            elif action == "phoenix_start":
                if not isinstance(payload, dict):
                    raise ValueError("Invalid Phoenix launch request.")
                # As with proxy_start, the loopback-only publish is enforced at the
                # root boundary inside `phoenix_service`; the payload is passed
                # through unaltered so there is one place the rule lives and it is
                # the one that launches the container (#178).
                data = runtime.phoenix_start(
                    payload.get("port"), payload.get("bind_host"),
                )
            elif action == "phoenix_stop":
                data = runtime.phoenix_stop()
            elif action == "phoenix_status":
                data = runtime.phoenix_status()
            elif action in AGENT_ACTIONS:
                # Dedicated verbs like phoenix_start: the config body and gate
                # keys ride the payload unaltered to the root boundary in
                # `agent_service`, which owns the fixed ExecStart. No
                # `bridge_argv_policy` entry, no `run_argv` (#178).
                data = dispatch_agent_action(runtime, action, payload)
            elif action == "balancer_start":
                if not isinstance(payload, dict):
                    raise ValueError(_INVALID_BALANCER_REQUEST)
                # As with proxy_start, the name, the pool and the key are
                # validated at the root boundary inside `gpu_pool_replicas`;
                # the payload is passed through unaltered so there is one place
                # the rule lives and it is the one that writes the root-owned
                # config (#178).
                data = runtime.balancer_start(
                    payload.get("name"), payload.get("port"),
                    payload.get("upstreams"), payload.get("api_key"),
                )
            elif action == "balancer_stop":
                if not isinstance(payload, dict):
                    raise ValueError(_INVALID_BALANCER_REQUEST)
                data = runtime.balancer_stop(payload.get("name"))
            elif action == "balancer_status":
                if not isinstance(payload, dict):
                    raise ValueError(_INVALID_BALANCER_REQUEST)
                data = runtime.balancer_status(payload.get("name"))
            elif action == "run_argv":
                if not isinstance(payload, dict):
                    raise ValueError("Invalid controller command request.")
                # The argv policy that matters is enforced at the root boundary
                # inside `run_argv`; the payload is carried through unaltered so
                # there is one place the rule lives and it is the one that
                # spawns the process (#178).
                data = runtime.run_argv(
                    payload.get("argv"),
                    payload.get("stdin_text", ""),
                    payload.get("timeout"),
                )
            elif action == "run_serving_profile":
                if not isinstance(payload, dict):
                    raise ValueError("Invalid serving profile request.")
                # Only the capture WINDOW crosses the wire; the PID is derived
                # root-side from `docker inspect`, never taken from the caller
                # (no profiling an attacker-named process). The window is clamped
                # and every capture is bounded inside the runtime (#178).
                data = runtime.run_serving_profile(payload.get("seconds"))
            elif action in HOST_SETTING_ACTIONS:
                # The GPU memory pool (VD-161): one whole number, or nothing,
                # crosses the socket; path, line and command are root-side.
                data = dispatch_host_setting(runtime, action, payload)
            elif action in CONTROLLER_UNIT_VERBS:
                # The controller's unit and model-store verbs (VD-143): typed
                # values in, every rule and every file operation root-side.
                data = runtime.controller_verb(action, payload)
            else:
                raise ValueError("Unsupported hardware action.")
            response = {"ok": True, "data": data}
        except OSError as error:
            # An OSError's text carries root-side paths and errno detail the
            # caller has no business reading: it goes to the journal, and the
            # caller gets a plain sentence (VD-143).
            print(
                "vaelor-hardware-bridge: {} failed: {}: {}".format(
                    str(action or "request"), type(error).__name__, error,
                ),
                file=sys.stderr, flush=True,
            )
            response = {"ok": False, "error": _OS_REFUSAL}
        except (AttributeError, OverflowError, RuntimeError, TypeError, ValueError) as error:
            response = {"ok": False, "error": str(error)}
        self.wfile.write(
            (json.dumps(response, separators=(",", ":")) + "\n").encode("utf-8")
        )


class _UnixStreamServer(socketserver.TCPServer):
    address_family = AF_UNIX


class _Server(socketserver.ThreadingMixIn, _UnixStreamServer):
    daemon_threads = True
    allow_reuse_address = True
    #: The uids served (VD-143, `bridge_peers`). Empty until `main` resolves
    #: them, so a server built any other way serves nobody.
    admitted_uids: frozenset = frozenset()

    def verify_request(self, request, client_address) -> bool:
        """Serve a connection only from an admitted peer, before reading a byte."""
        return admit(request, self.admitted_uids)


def _start_state_root_layout() -> threading.Thread:
    """Lay the state root out once per start, off the serving path (review B1).

    A hot-patched or upgraded box restarts this service; systemd runs the
    layout outside this unit's ``ProtectSystem=strict`` sandbox.
    """
    worker = threading.Thread(
        target=start_layout_unit, name="vaelor-state-root-layout", daemon=True,
    )
    worker.start()
    return worker


def main() -> None:
    if os.geteuid() != 0:
        raise SystemExit("The Pironman hardware bridge must run as root.")
    if AF_UNIX == -1:
        raise SystemExit("Unix sockets are required for the hardware bridge.")
    socket_path = Path(SOCKET_PATH)
    socket_path.parent.mkdir(parents=True, exist_ok=True)
    socket_path.unlink(missing_ok=True)
    runtime = _HardwareRuntime()
    server = _Server(str(socket_path), _Handler)
    server.runtime = runtime  # type: ignore[attr-defined]
    os.chmod(socket_path, 0o660)
    try:
        server.admitted_uids = admitted_uids()
        group_id, group_name = socket_group()
        os.chown(socket_path, 0, group_id)
        if group_name != BRIDGE_SOCKET_GROUP:
            # Every start, loudly, until the installer has run: the fallback
            # group is one the untrusted-content account also has, so only the
            # uid check stands between it and this socket.
            print(
                "vaelor-hardware-bridge: WARNING: the {} group does not exist, "
                "so the socket falls back to group {} (which vaelor-research also "
                "has). Every peer but root, vaelor and vaelor-workloads is still "
                "refused by uid. Re-run the Vaelor installer to create the "
                "group.".format(BRIDGE_SOCKET_GROUP, group_name),
                file=sys.stderr, flush=True,
            )
    except (KeyError, OSError):
        server.server_close()
        socket_path.unlink(missing_ok=True)
        raise

    stopped = threading.Event()

    def stop(_signum=None, _frame=None) -> None:
        if not stopped.is_set():
            stopped.set()
            threading.Thread(target=server.shutdown, daemon=True).start()

    signal.signal(signal.SIGINT, stop)
    signal.signal(signal.SIGTERM, stop)
    runtime.start()
    _start_state_root_layout()
    try:
        server.serve_forever(poll_interval=0.25)
    finally:
        server.server_close()
        runtime.stop()
        socket_path.unlink(missing_ok=True)


if __name__ == "__main__":
    main()
