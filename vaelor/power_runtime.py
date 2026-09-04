"""Select the host power provider exposed by the control plane.

Two things can perform host power, in preference order:

1. The privileged hardware bridge. It runs as root, so logind authorizes it
   unconditionally, and it is the only path available to the unprivileged
   control-plane account.
2. The platform driver directly, when *this* process is privileged enough —
   which is true for the bridge itself and for a legacy root deployment, and
   false for the control plane.

Availability used to be gated on whether a SunFounder Raspberry Pi package was
importable, which reported "no shutdown driver" on every x86 machine.
"""

from __future__ import annotations

from collections.abc import Callable
from typing import Any, Optional

from .platform_drivers import HostPowerController


def platform_power_actions() -> dict[str, dict[str, Any]]:
    """Power actions from the selected platform driver."""
    from .platforms import select_hardware_platform

    try:
        return select_hardware_platform().power_actions()
    except (OSError, RuntimeError, ValueError):
        return {}


class RuntimePowerController:
    """Resolve an optional privileged bridge at request time."""

    def __init__(
        self,
        *,
        hardware_bridge: Any | None,
        restart_service: Optional[Callable[[], Any]] = None,
        actions_probe: Optional[Callable[[], dict[str, dict[str, Any]]]] = None,
    ):
        self._hardware_bridge = hardware_bridge
        self._restart_service = restart_service
        # Resolved at call time rather than bound as a default argument, so the
        # seam is observable: a caller can inject one, and the real probe is
        # genuinely exercised when nobody does.
        self._actions_probe = actions_probe

    def _selected(self) -> HostPowerController:
        bridge = self._hardware_bridge
        if bridge is not None and bridge.available:
            return HostPowerController(
                restart_service=lambda: bridge.power("restart_service"),
                reboot=lambda: bridge.power("reboot"),
                shutdown=lambda: bridge.power("shutdown"),
            )
        probe = self._actions_probe or platform_power_actions
        actions = probe() or {}

        def resolve(name: str) -> Optional[Callable[[], Any]]:
            record = actions.get(name) or {}
            return record.get("run")

        # Deliberately no fallback to a caller-supplied restart hook: the
        # control plane's default is a no-op lambda, and wiring that in here
        # would report a successful service restart that never happened.
        return HostPowerController(
            restart_service=resolve("restart_service"),
            reboot=resolve("reboot"),
            shutdown=resolve("shutdown"),
            unavailable_reasons={
                name: (actions.get(name) or {}).get("reason", "")
                for name in ("restart_service", "reboot", "shutdown")
            },
        )

    def capabilities(self) -> dict[str, Any]:
        return self._selected().capabilities()

    def execute(self, action: str) -> Any:
        return self._selected().execute(action)


def build_host_power_controller(
    *,
    hardware_bridge: Any | None,
    restart_service: Optional[Callable[[], Any]] = None,
    actions_probe: Optional[Callable[[], dict[str, dict[str, Any]]]] = None,
    **_legacy: Any,
) -> RuntimePowerController:
    """Build the controller.

    ``**_legacy`` absorbs the retired ``legacy_available``/``reboot``/
    ``shutdown`` arguments so an out-of-tree caller does not break; they no
    longer influence the result.
    """
    return RuntimePowerController(
        hardware_bridge=hardware_bridge,
        restart_service=restart_service,
        actions_probe=actions_probe,
    )
