"""Network-address validation shared by cluster planning and execution."""

from __future__ import annotations

import ipaddress
from typing import Any


def private_controller_ipv4(value: Any) -> str:
    """Return a safe private controller address or raise a useful error."""
    try:
        address = ipaddress.ip_address(str(value).strip())
    except ValueError as error:
        raise ValueError("Choose a private IPv4 controller address.") from error
    if (
        address.version != 4
        or not address.is_private
        or address.is_loopback
        or address.is_link_local
        or address.is_multicast
    ):
        raise ValueError("Choose a private, non-loopback IPv4 controller address.")
    return str(address)
