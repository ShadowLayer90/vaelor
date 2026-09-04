"""Stable capability contracts between Vaelor and host-specific adapters."""

from __future__ import annotations

from typing import Any, Dict, Optional, Protocol, runtime_checkable


Snapshot = Dict[str, Any]


@runtime_checkable
class HardwarePlatform(Protocol):
    def snapshot(
        self,
        raw: Optional[Snapshot],
        metrics: Optional[Snapshot],
        inventory: Optional[Snapshot],
    ) -> Snapshot:
        """Describe the host, enclosure, power, and feature policy."""


@runtime_checkable
class OperatingSystemDriver(Protocol):
    def snapshot(self) -> Snapshot:
        """Return normalized OS identity and support metadata."""

    def worker_compatibility(
        self,
        inventory: Snapshot,
        allowed_architectures: set[str],
    ) -> Snapshot:
        """Validate a remote worker against this platform policy."""

    def managed_services(self) -> Dict[str, tuple[str, ...]]:
        """Return logical service IDs and ordered native-unit candidates."""


@runtime_checkable
class PackageManager(Protocol):
    def installation_capability(self, os_info: Snapshot) -> Snapshot:
        """Describe whether guarded package installation is available."""

    def update_command(self) -> list[str]:
        """Return a fixed-argument package metadata refresh command."""

    def install_command(self, packages: list[str]) -> list[str]:
        """Return a fixed-argument package installation command."""

    def list_upgradable_command(self) -> list[str]:
        """Return a read-only command that lists pending upgrades."""

    def parse_upgradable(self, output: str) -> list[Snapshot]:
        """Normalize bounded package-manager output into update records."""

    def stage_upgrade_commands(self) -> list[list[str]]:
        """Return fixed commands that download, but do not apply, upgrades."""

    def apply_upgrade_commands(self) -> list[list[str]]:
        """Return fixed commands that apply staged system upgrades."""

    def audit_command(self) -> list[str]:
        """Return a read-only command listing any half-configured package."""

    def repair_commands(self) -> list[list[str]]:
        """Return idempotent commands that finish an interrupted dpkg run."""


@runtime_checkable
class ContainerScheduler(Protocol):
    def capabilities(self) -> Snapshot:
        """Describe the local or clustered workload scheduler."""


@runtime_checkable
class InferenceBackend(Protocol):
    def capabilities(self) -> Snapshot:
        """Describe an inference runtime without exposing credentials."""


@runtime_checkable
class StorageProvider(Protocol):
    def capabilities(self) -> Snapshot:
        """Describe storage media and supported lifecycle operations."""

    def inventory(self) -> Snapshot:
        """Return bounded physical-device and mounted-volume inventory."""


@runtime_checkable
class RemoteAccessProvider(Protocol):
    def capabilities(self) -> Snapshot:
        """Describe native desktop, browser desktop, and console support."""


@runtime_checkable
class TelemetryProvider(Protocol):
    def capabilities(self) -> Snapshot:
        """Describe available telemetry sources."""

    def snapshot(self, metrics: Optional[Snapshot] = None) -> Snapshot:
        """Merge platform telemetry with any hardware-adapter metrics."""


@runtime_checkable
class PowerController(Protocol):
    def capabilities(self) -> Snapshot:
        """Describe observable and controllable power features."""

    def execute(self, action: str) -> Any:
        """Execute one allowlisted action through the platform provider."""
