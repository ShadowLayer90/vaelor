"""Default Linux adapters for Vaelor's stable platform contracts."""

from __future__ import annotations

import platform
import re
import shutil
import subprocess
from typing import Any, Callable, Dict, Optional

from .docker_health import runtime_health
from .linux_storage import LinuxStorageProvider
from .linux_telemetry import LinuxTelemetryProvider
from .platforms import read_os_release, select_hardware_platform
from .platforms.accelerators import accelerator_summary


Snapshot = Dict[str, Any]

VAELOR_LINUX_SERVICES = {
    "control-plane": ("vaelor-control-plane.service", "pironman5.service"),
    "workload-executor": (
        "vaelor-workload-executor.service",
        "pironman-workload-executor.service",
    ),
    "workload-broker": ("vaelor-workload-broker.service",),
    "credential-broker": (
        "vaelor-credential-broker.service",
        "pironman-credential-broker.service",
    ),
    "vnc-gateway": (
        "vaelor-vnc-gateway.service",
        "pironman-vnc-gateway.service",
    ),
    "vnc-tls": (
        "vaelor-vnc-tls-proxy.service",
        "pironman-vnc-tls-proxy.service",
    ),
}


class DetectedHardwarePlatform:
    """Adapter that resolves the platform driver through the registry.

    The driver is chosen once and reused, so a single request does not probe
    the board, SMBIOS and the DRM tree several times over.
    """

    def __init__(self, driver=None):
        self._driver = driver or select_hardware_platform()

    @property
    def driver(self):
        return self._driver

    @property
    def machine_class(self) -> str:
        return self._driver.machine_class

    def snapshot(
        self,
        raw: Optional[Snapshot],
        metrics: Optional[Snapshot],
        inventory: Optional[Snapshot],
    ) -> Snapshot:
        return self._driver.snapshot(raw, metrics, inventory)

    def machine(
        self,
        raw: Optional[Snapshot] = None,
        metrics: Optional[Snapshot] = None,
        inventory: Optional[Snapshot] = None,
    ) -> Snapshot:
        return self._driver.machine(raw, metrics, inventory)

    def capabilities(
        self,
        raw: Optional[Snapshot] = None,
        metrics: Optional[Snapshot] = None,
        inventory: Optional[Snapshot] = None,
    ) -> Snapshot:
        return self._driver.capabilities(raw, metrics, inventory)

    def product(self, raw: Optional[Snapshot] = None) -> Snapshot:
        return self._driver.resolve_product(raw)

    def accelerators(self) -> list[Snapshot]:
        return self._driver.accelerators()

    def neural_accelerators(self) -> list[Snapshot]:
        return self._driver.neural_accelerators()

    def graphics(self) -> Snapshot:
        return self._driver.graphics()

    def thermal_policy(self) -> Snapshot:
        return self._driver.thermal_policy()


class LinuxOperatingSystemDriver:
    def __init__(self, reader: Callable[[], Snapshot] = read_os_release):
        self._reader = reader

    def snapshot(self) -> Snapshot:
        return dict(self._reader())

    #: Distributions whose package manager Vaelor knows how to drive. This is
    #: not a list of operating systems a worker is allowed to run — it is the
    #: list Vaelor can *install Docker on*, which is a different question and
    #: only arises when the worker has not already got it.
    APT_MANAGED_WORKER_OS = frozenset({"debian", "raspbian", "ubuntu"})

    def worker_compatibility(
        self,
        inventory: Snapshot,
        allowed_architectures: set[str],
    ) -> Snapshot:
        """Whether a probed worker can run cluster workloads, and why not.

        This used to refuse any worker whose ``os_id`` was outside the
        Debian family, which filtered on the wrong axis: the OS *name* was
        standing in for two real constraints. The first — architecture — is
        now the enrollment gate (VD-031) and is checked before this. The
        second is a capability, and capabilities are discovered: does the
        worker have a container runtime, and if not, can Vaelor install one
        here? A Fedora host that already runs Docker was being turned away
        for a reason that did not apply to it.
        """
        architecture = str(inventory.get("architecture", "")).lower()
        if architecture not in allowed_architectures:
            return {
                "compatible": False,
                "reason": (
                    "The worker architecture is not supported by this "
                    "cluster profile."
                ),
            }
        if bool(inventory.get("docker")):
            return {"compatible": True, "reason": ""}
        os_id = str(inventory.get("os_id", "")).lower()
        if os_id and os_id not in self.APT_MANAGED_WORKER_OS:
            return {
                "compatible": False,
                "reason": (
                    "This worker has no container runtime, and Vaelor can "
                    "only install one for it on a Debian-family operating "
                    "system. Install Docker on the worker, then enrol it "
                    "again."
                ),
            }
        return {"compatible": True, "reason": ""}

    def managed_services(self) -> Dict[str, tuple[str, ...]]:
        return dict(VAELOR_LINUX_SERVICES)


class AptPackageManager:
    def __init__(self, finder: Callable[[str], Optional[str]] = shutil.which):
        self._finder = finder

    def installation_capability(self, os_info: Snapshot) -> Snapshot:
        supported = (
            os_info.get("family") == "debian"
            and os_info.get("support_level") in {"verified", "compatible"}
            and self._finder("apt-get") is not None
        )
        if supported:
            reason = "Packages can be installed automatically on this host."
        elif os_info.get("support_level") == "limited":
            reason = (
                "This appliance OS manages packages itself; Vaelor will not "
                "modify its package set."
            )
        else:
            reason = (
                "Automatic package installation is not validated for this "
                "operating system."
            )
        return {
            "available": supported,
            "id": "apt" if supported else None,
            "method": "Debian package manager" if supported else None,
            "reason": reason,
        }

    def _executable(self) -> str:
        executable = self._finder("apt-get")
        if not executable:
            raise RuntimeError("The Debian package manager is unavailable.")
        return executable

    def update_command(self) -> list[str]:
        return [
            self._executable(),
            "-o",
            "APT::Sandbox::User=root",
            "update",
        ]

    def install_command(self, packages: list[str]) -> list[str]:
        if not packages or any(
            not re.fullmatch(r"[a-z0-9][a-z0-9+.-]{0,127}", package)
            for package in packages
        ):
            raise ValueError("Package names must use the guarded allowlist format.")
        return [
            self._executable(),
            "-o",
            "APT::Sandbox::User=root",
            "--yes",
            "install",
            *packages,
        ]

    def list_upgradable_command(self) -> list[str]:
        executable = self._finder("apt")
        if not executable:
            raise RuntimeError("The Debian package inventory tool is unavailable.")
        return [executable, "list", "--upgradable"]

    def parse_upgradable(self, output: str) -> list[Snapshot]:
        details = []
        for line in str(output).splitlines()[1:101]:
            if "/" not in line:
                continue
            match = re.match(
                r"^(?P<name>[^/]+)/(?P<source>\S+)\s+"
                r"(?P<candidate>\S+)\s+\S+\s+\[upgradable from: "
                r"(?P<installed>[^\]]+)\]",
                line,
            )
            if match:
                item = match.groupdict()
                source = item["source"].lower()
                name = item["name"].lower()
                item["classification"] = (
                    "Security" if "security" in source
                    else "Firmware" if name.startswith("linux-firmware")
                    else "Desktop" if any(
                        marker in name for marker in (
                            "gnome", "gdm", "ubuntu-desktop", "session", "papers"
                        )
                    )
                    else "System"
                )
                details.append(item)
            else:
                details.append({
                    "name": line.split("/", 1)[0],
                    "source": "package repository",
                    "candidate": "available",
                    "installed": "installed",
                    "classification": "System",
                })
        return details

    def stage_upgrade_commands(self) -> list[list[str]]:
        return [
            self.update_command(),
            [
                self._executable(),
                "-o",
                "APT::Sandbox::User=root",
                "--yes",
                "--download-only",
                "upgrade",
            ],
        ]

    def apply_upgrade_commands(self) -> list[list[str]]:
        return [[
            self._executable(),
            "-o",
            "APT::Sandbox::User=root",
            "--yes",
            "upgrade",
        ]]

    def _dpkg(self) -> str:
        executable = self._finder("dpkg")
        if not executable:
            raise RuntimeError("The Debian package configuration tool is unavailable.")
        return executable

    def audit_command(self) -> list[str]:
        """Read-only check that lists any half-installed/half-configured package.

        Empty output means the package database is whole. Used to decide whether
        a recovery attempt actually healed the system before the upgrade is
        retried, so a genuine break is never reported as recovered.
        """
        return [self._dpkg(), "--audit"]

    def repair_commands(self) -> list[list[str]]:
        """Finish a dpkg run left half-done by a trigger-ordering race.

        Ubuntu 26.04 builds the initramfs with ``dracut``, whose trigger can
        fire for a freshly-staged kernel before that kernel's
        ``linux-modules-*`` package has finished unpacking - so ``dpkg`` aborts
        the ``apt upgrade`` with the kernel left *half-configured* even though
        nothing is actually wrong. By the time the run has aborted the modules
        ARE unpacked, so ``dpkg --configure -a`` completes the deferred
        configuration and ``apt-get -f install`` closes any dependency the abort
        left open. Both are idempotent and a no-op on a healthy system, so
        running them can only move the machine toward a whole state.
        """
        return [
            [self._dpkg(), "--configure", "-a"],
            [
                self._executable(),
                "-o",
                "APT::Sandbox::User=root",
                "--fix-broken",
                "--yes",
                "install",
            ],
        ]


class DockerComposeScheduler:
    def __init__(
        self,
        package_manager: AptPackageManager,
        os_driver: LinuxOperatingSystemDriver,
        finder: Callable[[str], Optional[str]] = shutil.which,
        runner: Callable[..., Any] = subprocess.run,
        runtime_probe: Optional[Callable[[Optional[str]], dict]] = None,
    ):
        self._package_manager = package_manager
        self._os_driver = os_driver
        self._finder = finder
        self._runner = runner
        self._runtime_probe = runtime_probe

    def capabilities(self) -> Snapshot:
        docker_path = self._finder("docker")
        compose_version = None
        # Whether the Compose probe reached a definite answer. It stays True
        # when Docker is absent (no engine means Compose is definitively
        # absent) or when the probe ran to a return code. It goes False only
        # when the probe could not run at all - a daemon still starting, a
        # busy socket, or the 3s timeout - because "could not check" is not
        # "not installed" (#120, LESSONS 8: a timeout and a true negative must
        # not arrive looking the same).
        compose_checked = True
        if docker_path:
            try:
                result = self._runner(
                    [docker_path, "compose", "version", "--short"],
                    capture_output=True,
                    check=False,
                    text=True,
                    timeout=3,
                )
                if result.returncode == 0:
                    compose_version = result.stdout.strip()[:64]
            except (OSError, subprocess.SubprocessError):
                compose_checked = False
        installation = self._package_manager.installation_capability(
            self._os_driver.snapshot()
        )
        if installation["available"]:
            installation = {
                **installation,
                "reason": (
                    "Docker and Compose can be installed automatically on "
                    "this host."
                ),
            }
        # A daemon that answers is not a daemon that can start containers. This
        # cheap structural probe (one `docker info`, one stat - never a run)
        # catches a wiped/broken data-root, which otherwise reads as a healthy
        # node while every `docker run` fails. Failing safe: only a confirmed
        # missing storage area is called broken (see docker_health).
        health = (self._runtime_probe or (
            lambda dp: runtime_health(dp, runner=self._runner)
        ))(docker_path)
        return {
            "id": "docker-compose",
            "installed": docker_path is not None,
            "compose": compose_version is not None,
            "runtime": health["state"],
            "runtime_reason": health["reason"],
            # The third state, kept apart from `compose`. `compose` False with
            # this True is a definite absence; False with this False is a
            # probe that could not run, and a consumer must not render the
            # second as the first (offer an install, say "not ready") - that
            # collapse is #120.
            "compose_checked": compose_checked,
            "compose_version": compose_version,
            "installation": installation,
        }


class LinuxRemoteAccessProvider:
    def __init__(
        self,
        os_driver: LinuxOperatingSystemDriver,
        package_manager: AptPackageManager,
        finder: Callable[[str], Optional[str]] = shutil.which,
    ):
        self._os_driver = os_driver
        self._package_manager = package_manager
        self._finder = finder

    def capabilities(self) -> Snapshot:
        os_info = self._os_driver.snapshot()
        package_install = self._package_manager.installation_capability(os_info)
        os_name = os_info.get("name") or "Linux"
        native_rdp = (
            os_info.get("id") == "ubuntu"
            and self._finder("grdctl") is not None
        )
        return {
            "os": os_info,
            "browser_desktop_setup": bool(package_install["available"]),
            "native_rdp_setup": native_rdp,
            "native_rdp_name": (
                "Ubuntu Remote Login"
                if os_info.get("id") == "ubuntu"
                else f"{os_name} Remote Desktop"
            ),
            "console_kind": "ssh",
        }


class ManagedLocalInferenceBackend:
    def __init__(self, accelerator_probe: Callable[[], Snapshot] = accelerator_summary):
        self._accelerator_probe = accelerator_probe

    def capabilities(self) -> Snapshot:
        architecture = platform.machine() or "unknown"
        try:
            accelerators = self._accelerator_probe()
        except OSError:
            accelerators = {"accelerators": [], "detected": False, "access": {}}
        return {
            "id": "managed-local",
            "architecture": architecture,
            "openai_compatible": True,
            "model_discovery": True,
            "streaming": True,
            "accelerator_detected": bool(accelerators.get("detected")),
            "accelerators": accelerators.get("accelerators", []),
            "accelerator_access": accelerators.get("access", {}),
        }


class HostPowerController:
    def __init__(
        self,
        *,
        restart_service: Optional[Callable[[], Any]] = None,
        reboot: Optional[Callable[[], Any]] = None,
        shutdown: Optional[Callable[[], Any]] = None,
        telemetry_probe: Optional[Callable[[], bool]] = None,
        unavailable_reasons: Optional[Dict[str, str]] = None,
    ):
        self._actions = {
            "restart_service": restart_service,
            "reboot": reboot,
            "shutdown": shutdown,
        }
        # The generic fallback text claimed the *driver* did not provide the
        # action, which on x86 was untrue: the driver provides it and the
        # process simply lacked privilege. Callers may supply the real reason.
        self._reasons = dict(unavailable_reasons or {})
        # Whether the host reports power is a platform fact. Asking
        # `shutil.which("vcgencmd")` made every machine answer the question
        # with a Broadcom firmware tool.
        self._telemetry_probe = telemetry_probe or (
            lambda: select_hardware_platform().power_telemetry()
        )

    def capabilities(self) -> Snapshot:
        try:
            telemetry = bool(self._telemetry_probe())
        except (OSError, RuntimeError, ValueError):
            telemetry = False
        return {
            "id": "host-power",
            "telemetry": telemetry,
            "actions": {
                name: {
                    "available": callback is not None,
                    "reason": (
                        ""
                        if callback is not None
                        else self._reasons.get(name)
                        or "The installed platform driver does not provide this action."
                    ),
                }
                for name, callback in self._actions.items()
            },
            "target_atx": False,
        }

    def execute(self, action: str) -> Any:
        if action not in self._actions:
            raise ValueError("Choose restart_service, reboot, or shutdown.")
        callback = self._actions[action]
        if callback is None:
            raise RuntimeError(
                "The installed platform driver does not provide this power action."
            )
        return callback()


def default_platform_drivers() -> Snapshot:
    os_driver = LinuxOperatingSystemDriver()
    package_manager = AptPackageManager()
    return {
        "hardware": DetectedHardwarePlatform(),
        "operating_system": os_driver,
        "package_manager": package_manager,
        "container_scheduler": DockerComposeScheduler(
            package_manager, os_driver
        ),
        "inference_backend": ManagedLocalInferenceBackend(),
        "storage_provider": LinuxStorageProvider(),
        "remote_access_provider": LinuxRemoteAccessProvider(
            os_driver, package_manager
        ),
        "telemetry_provider": LinuxTelemetryProvider(),
        "power_controller": HostPowerController(),
    }
