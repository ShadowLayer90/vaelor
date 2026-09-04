"""Machine class and capability resolution shared by the API route modules.

Route handlers must not decide for themselves whether a control exists. They
ask the selected platform driver, and when the answer is "no" they surface the
driver's reason verbatim so the operator reads why *this* machine cannot do it.
"""

from __future__ import annotations

from typing import Any, Dict, Optional

from .npu_serving_state import annotate_npu_serving
from .platform_drivers import default_platform_drivers


def platform_driver(callbacks: Dict[str, Any]):
    drivers = callbacks.get("platform_drivers") or default_platform_drivers()
    return drivers["hardware"]


def machine_capabilities(
    callbacks: Dict[str, Any],
    raw: Optional[Dict[str, Any]] = None,
    metrics: Optional[Dict[str, Any]] = None,
) -> Dict[str, Dict[str, Any]]:
    driver = platform_driver(callbacks)
    try:
        return driver.capabilities(raw, metrics, None)
    except (AttributeError, NotImplementedError, OSError):
        return {}


def capability_gate(
    callbacks: Dict[str, Any],
    name: str,
    *,
    code: str,
    fallback: str,
) -> Optional[Dict[str, str]]:
    """Return an API error body when ``name`` is unavailable, else ``None``.

    Fails closed: a driver that cannot answer is treated as "not available"
    rather than letting the mutation through to a bridge that will raise.
    """
    device_info = callbacks.get("device_info")
    current_data = callbacks.get("current_data")
    raw = (device_info() if device_info else {}) or {}
    metrics = (current_data() if current_data else {}) or {}
    capabilities = machine_capabilities(callbacks, raw, metrics)
    record = capabilities.get(name)
    if record is None:
        return {"code": code, "message": fallback}
    if record.get("available"):
        return None
    return {"code": code, "message": record.get("reason") or fallback}


def machine_payload(callbacks: Dict[str, Any]) -> Dict[str, Any]:
    """The ``GET /api/v2/system/machine`` body."""
    device_info = callbacks.get("device_info")
    current_data = callbacks.get("current_data")
    inventory_probe = callbacks.get("hardware_inventory")
    raw = (device_info() if device_info else {}) or {}
    metrics = (current_data() if current_data else {}) or {}
    inventory = (inventory_probe() if inventory_probe else {}) or {}
    driver = platform_driver(callbacks)
    payload = driver.machine(raw, metrics, inventory)
    payload["accelerators"] = driver.accelerators()
    # The driver reports pure hardware discovery; whether the Assistant is
    # actually served on the neural processor is control-plane state the record
    # only learns here, where the running deployment is in reach (see
    # `npu_serving_state`). A machine with no NPU is returned unchanged.
    payload["neural_accelerators"] = annotate_npu_serving(
        driver.neural_accelerators(), callbacks
    )
    payload["graphics"] = driver.graphics()
    payload["thermal_policy"] = driver.thermal_policy()
    return payload
