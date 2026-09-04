"""Dynamic access to the optional root-owned Pironman hardware bridge."""

from __future__ import annotations

from typing import Any

from .hardware_bridge import HardwareBridgeClient, HardwareBridgeError


class RuntimeHardware:
    """Resolve bridge availability per request instead of only during startup."""

    def __init__(self, bridge: HardwareBridgeClient | None = None) -> None:
        self.bridge = bridge or HardwareBridgeClient()

    def _read(self, method: str) -> dict[str, Any]:
        if not self.bridge.available:
            return {}
        try:
            value = getattr(self.bridge, method)()
        except (HardwareBridgeError, OSError, RuntimeError, ValueError):
            return {}
        return value if isinstance(value, dict) else {}

    def device_info(self) -> dict[str, Any]:
        return self._read("device_info")

    def current_data(self) -> dict[str, Any]:
        return self._read("current_data")

    def read_config(self) -> dict[str, Any]:
        return self._read("read_config")

    def update_config(self, patch: dict[str, Any]) -> dict[str, Any]:
        if not self.bridge.available:
            raise RuntimeError("Pironman hardware service is unavailable.")
        return self.bridge.update_config(patch)

    def ip_data(self) -> dict[str, Any]:
        return {
            key: value
            for key, value in self.current_data().items()
            if key in {"ips", "network_type"}
            or key.startswith("ip_")
            or key.startswith("mac_")
        }
