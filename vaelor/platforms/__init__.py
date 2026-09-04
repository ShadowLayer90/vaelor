"""Hardware platform driver registry and discovery-based selection.

``ARCHITECTURE.md`` described a registry here long before one existed. This is
it. Drivers register a probe; selection runs the probes in registration order
and takes the first that claims the host. Nothing infers support from an
architecture name — the probes read the machine.
"""

from __future__ import annotations

from typing import Any, Callable, Dict, List, Optional, Tuple

from ..runtime_paths import env_value
from .base import (
    CAPABILITY_KEYS,
    GENERIC,
    GENERIC_PRODUCT,
    MACHINE_CLASSES,
    PI_APPLIANCE,
    WORKSTATION,
    HardwarePlatformBase,
    board_info,
    capability,
    dmi_identity,
    read_os_release,
    thermal_policy,
)
from .raspberry_pi import (
    ALIASES,
    PRODUCTS,
    RaspberryPiPlatform,
    detect_product,
    enclosure_detected,
    observed_capabilities,
    plausible_products,
    power_snapshot,
)
from .workstation import GenericPlatform, WorkstationPlatform


Probe = Callable[[Dict[str, Any]], bool]
Factory = Callable[..., HardwarePlatformBase]

__all__ = [
    "ALIASES",
    "CAPABILITY_KEYS",
    "GENERIC",
    "GENERIC_PRODUCT",
    "GenericPlatform",
    "HardwarePlatformBase",
    "MACHINE_CLASSES",
    "PI_APPLIANCE",
    "PRODUCTS",
    "RaspberryPiPlatform",
    "WORKSTATION",
    "WorkstationPlatform",
    "available_drivers",
    "board_info",
    "capability",
    "detect_product",
    "dmi_identity",
    "enclosure_detected",
    "observed_capabilities",
    "plausible_products",
    "power_snapshot",
    "read_os_release",
    "register_driver",
    "select_hardware_platform",
    "thermal_policy",
]


def _is_raspberry_pi(board: Dict[str, Any]) -> bool:
    return bool(board.get("is_raspberry_pi"))


def _is_workstation(board: Dict[str, Any]) -> bool:
    return board.get("firmware_identity") == "smbios"


def _always(_board: Dict[str, Any]) -> bool:
    return True


_REGISTRY: List[Tuple[str, Probe, Factory]] = [
    ("raspberry-pi", _is_raspberry_pi, RaspberryPiPlatform),
    ("workstation", _is_workstation, WorkstationPlatform),
    ("generic", _always, GenericPlatform),
]


def register_driver(name: str, probe: Probe, factory: Factory) -> None:
    """Add a driver ahead of the generic fallback.

    Registration order is selection order, so a new driver takes precedence
    over every driver registered after it and over the fallback.
    """
    global _REGISTRY
    _REGISTRY = [entry for entry in _REGISTRY if entry[0] != name]
    _REGISTRY.insert(max(0, len(_REGISTRY) - 1), (name, probe, factory))


def available_drivers() -> List[str]:
    return [name for name, _probe, _factory in _REGISTRY]


def _factory_for(name: str) -> Optional[Factory]:
    for registered, _probe, factory in _REGISTRY:
        if registered == name:
            return factory
    return None


def select_hardware_platform(
    board: Optional[Dict[str, Any]] = None, **kwargs: Any
) -> HardwarePlatformBase:
    """Choose the driver whose probe matches this host.

    ``VAELOR_PLATFORM_DRIVER`` forces a named driver. It exists so acceptance
    runs and tests can exercise a platform they are not executing on; it is not
    a supported production setting and an unknown name is ignored rather than
    silently selecting the wrong hardware.
    """
    forced = env_value("VAELOR_PLATFORM_DRIVER", "PM_PLATFORM_DRIVER", "").strip()
    if forced:
        factory = _factory_for(forced)
        if factory is not None:
            return factory(**kwargs)
    facts = board if board is not None else board_info()
    for _name, probe, factory in _REGISTRY:
        if probe(facts):
            return factory(**kwargs)
    return GenericPlatform(**kwargs)
