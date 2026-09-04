"""Narrow Unix-socket bridge to the preserved Pironman hardware runtime."""

from __future__ import annotations

import json
import os
import signal
import socket
import socketserver
import sys
import threading
import types
from dataclasses import asdict, is_dataclass
from pathlib import Path
from typing import Any

from .runtime_paths import env_value


SOCKET_PATH = env_value(
    "VAELOR_HARDWARE_BRIDGE_SOCKET",
    "PM_HARDWARE_BRIDGE_SOCKET",
    "/run/vaelor/hardwared.sock",
)
PIRONMAN_CONFIG = Path("/opt/pironman5/config.json")
PIRONMAN_VENV = Path("/opt/pironman5/venv")
MAX_REQUEST_BYTES = 64 * 1024
AF_UNIX = getattr(socket, "AF_UNIX", -1)


class HardwareBridgeError(RuntimeError):
    """Raised when the optional Pironman hardware bridge is unavailable."""


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


class HardwareBridgeClient:
    """Small fail-closed client used by the unprivileged control plane."""

    def __init__(self, socket_path: str = SOCKET_PATH, timeout: float = 3.0):
        self.socket_path = socket_path
        self.timeout = timeout

    @property
    def available(self) -> bool:
        return Path(self.socket_path).is_socket()

    def _request(self, action: str, payload: dict[str, Any] | None = None) -> Any:
        request = {"action": action, "payload": payload or {}}
        encoded = (json.dumps(request, separators=(",", ":")) + "\n").encode()
        if len(encoded) > MAX_REQUEST_BYTES:
            raise HardwareBridgeError("Hardware request is too large.")
        try:
            with socket.socket(AF_UNIX, socket.SOCK_STREAM) as connection:
                connection.settimeout(self.timeout)
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
            raise ValueError("Choose restart_service, reboot, or shutdown.")
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


class _HardwareRuntime:
    """Own PMAuto without starting the legacy web dashboard."""

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
            raise ValueError("Choose restart_service, reboot, or shutdown.")
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
        with self._lock:
            existing = getattr(self, "_flm_process", None)
            if existing is None:
                from .flm_service import FlmProcess

                existing = FlmProcess()
                self._flm_process = existing
            return existing

    def flm_start(self, tag: str, ctx_len: int, port: int) -> dict[str, Any]:
        """Launch flm-real as root (VD-001).

        This service already runs as root - it asserted ``geteuid()==0`` before
        serving - so the child inherits CAP_IPC_LOCK to pin NPU pages without
        any capability juggling here. The tag is validated inside
        :mod:`vaelor.flm_service` against the installed allowlist and the tag
        grammar before it is placed in a fixed argument vector: no shell, no
        interpolation, no ``PATH`` lookup (LESSONS #178).

        Held under ``self._lock`` for the WHOLE op, not just the singleton
        lookup (VD-001). `_Server` is a `ThreadingMixIn` with `daemon_threads`,
        so two `flm_start` requests - the boot reconcile and a manual
        `model.deploy` - run on concurrent threads. `FlmProcess.start` is a
        non-atomic check-then-act (`if alive(): stop()` then spawn); without the
        lock both callers can see `alive()==False` and both spawn, orphaning a
        second flm-real that pins NPU pages. The RLock is reentrant, so the
        nested `_flm()` acquire is fine, and it serializes launches so only one
        flm-real is ever tracked.
        """
        with self._lock:
            return self._flm().start(str(tag), ctx_len=int(ctx_len), port=int(port))

    def flm_stop(self) -> dict[str, Any]:
        """Stop flm-real. A service stop only - never a reboot (VD-019).

        Under ``self._lock`` for the same reason as :meth:`flm_start`: a stop
        racing a concurrent start must not interleave the check-then-act that
        tracks the single flm-real process (VD-001).
        """
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
        """Download, verify and unpack a fine-tuned NPU model release, as root.

        Serialised with :meth:`flm_start` / :meth:`flm_stop` under ``self._lock``:
        the model files it replaces are the ones a concurrent launch would read,
        so an install must not interleave with a start. The verify-and-unpack
        lives in :mod:`vaelor.flm_npu_release` (sha256 pin, path-traversal guard,
        streamed to disk); this targets the fixed snap home and the ``flm-real``
        binary directory (`flm_service.FLM_HOME` / `FLM_BINARY`), so once the
        model AND its bundled runtime land there the existing supervisor serves
        the installed tag with no other change - the turnkey equivalent of the
        manual place-model-and-swap-runtime step, done as root behind the bridge.
        """
        from pathlib import Path

        from . import flm_npu_release
        from .flm_service import FLM_BINARY, FLM_HOME

        with self._lock:
            # Stop any running flm-real before overwriting its binary: an
            # executable still being run cannot be replaced (ETXTBSY, "Text file
            # busy"). A no-op on a fresh box; on a re-install over a live model it
            # frees the binary, and the deploy step relaunches flm-real after.
            self._flm().stop()
            return flm_npu_release.install(
                str(source_url),
                Path(FLM_HOME),
                Path(FLM_BINARY).parent,
                expected_sha256=str(expected_sha256),
            )

    def _gpu_server(self):
        """The supervised ROCmFPX GPU server process, created on first use.

        Lazily imported and constructed like :meth:`_flm`, so a host that never
        serves the GPU tier never touches
        :mod:`vaelor.gpu_rocmfpx_service`, and so the bridge starts on a machine
        where the ROCmFPX fork does not exist.
        """
        with self._lock:
            existing = getattr(self, "_gpu_process", None)
            if existing is None:
                from .gpu_rocmfpx_service import GpuServerProcess

                existing = GpuServerProcess()
                self._gpu_process = existing
            return existing

    def gpu_start(
        self, model_path: Any, port: Any, runtime: Any
    ) -> dict[str, Any]:
        """Launch the ROCmFPX GPU model server as root.

        The GPU server is launched here, not from the workload executor, for the
        privileges only this unit holds: /dev/dri and /dev/kfd for HIP/Vulkan (the
        executor's ``PrivateDevices=true`` hides them, so the fork sees no GPU and
        falls back to the CPU), a writable ``/var/log/vaelor`` for the server log
        (the executor's is read-only — the ``[Errno 30] Read-only file system``
        that blocked the deploy), and no ``MemoryDenyWriteExecute`` for the fork's
        JIT. This service already asserted ``geteuid()==0`` before serving, so the
        child inherits all of it without any capability juggling here. The model
        path and port are validated inside :mod:`vaelor.gpu_rocmfpx_service`
        against the path and port rules before they enter a fixed argument
        vector: no shell, no interpolation (LESSONS #178).

        Held under ``self._lock`` for the WHOLE op, exactly as :meth:`flm_start`
        is (VD-001): ``_Server`` is a ``ThreadingMixIn`` with ``daemon_threads``,
        so a boot reconcile and a manual ``model.deploy`` can call ``gpu_start``
        on two threads at once. ``GpuServerProcess.start`` is a non-atomic
        check-then-act (``if alive(): stop()`` then spawn); without the lock both
        callers can see ``alive()==False`` and both spawn, orphaning a second
        server holding VRAM. The RLock is reentrant, so the nested
        ``_gpu_server()`` acquire is fine, and it serializes launches so only one
        server is ever tracked.
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


class _Handler(socketserver.StreamRequestHandler):
    def handle(self) -> None:
        request_bytes = self.rfile.readline(MAX_REQUEST_BYTES + 1)
        response: dict[str, Any]
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
            else:
                raise ValueError("Unsupported hardware action.")
            response = {"ok": True, "data": data}
        except (AttributeError, OSError, RuntimeError, TypeError, ValueError) as error:
            response = {"ok": False, "error": str(error)}
        self.wfile.write(
            (json.dumps(response, separators=(",", ":")) + "\n").encode("utf-8")
        )


class _UnixStreamServer(socketserver.TCPServer):
    address_family = AF_UNIX


class _Server(socketserver.ThreadingMixIn, _UnixStreamServer):
    daemon_threads = True
    allow_reuse_address = True


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
        import grp

        os.chown(socket_path, 0, grp.getgrnam("vaelor").gr_gid)
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
    try:
        server.serve_forever(poll_interval=0.25)
    finally:
        server.server_close()
        runtime.stop()
        socket_path.unlink(missing_ok=True)


if __name__ == "__main__":
    main()
