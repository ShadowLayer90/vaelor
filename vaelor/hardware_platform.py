"""Compatibility surface over the ``vaelor.platforms`` driver package.

The Raspberry Pi enclosure logic that used to live here is now one driver among
others in ``vaelor/platforms/``. This module keeps the historical import names
working for existing callers, tests, and the physical acceptance tool.
"""

from __future__ import annotations

from typing import Any, Dict, Optional

from .platforms import (
    ALIASES,
    GENERIC_PRODUCT,
    PRODUCTS,
    board_info,
    detect_product,
    enclosure_detected,
    power_snapshot,
    read_os_release,
    select_hardware_platform,
)


__all__ = [
    "ALIASES",
    "GENERIC_PRODUCT",
    "PRODUCTS",
    "board_info",
    "detect_product",
    "enclosure_detected",
    "platform_snapshot",
    "power_snapshot",
    "read_os_release",
    "select_hardware_platform",
]


def platform_snapshot(
    raw: Optional[Dict[str, Any]],
    metrics: Optional[Dict[str, Any]],
    hardware: Optional[Dict[str, Any]],
) -> Dict[str, Any]:
    """Describe this host through whichever platform driver claims it."""
    return select_hardware_platform().snapshot(raw, metrics, hardware)
