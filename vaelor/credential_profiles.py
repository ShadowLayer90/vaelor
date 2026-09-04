"""Validation for encrypted non-AI credential profiles."""

from __future__ import annotations

import ipaddress
import json
import re
import socket
from typing import Any, Dict, Type


def validate_ssh_profile(
    secret_value: str,
    error_type: Type[ValueError] = ValueError,
) -> Dict[str, Any]:
    """Validate and normalize a LAN-only SSH enrollment profile."""
    try:
        profile = json.loads(secret_value)
    except (TypeError, json.JSONDecodeError) as error:
        raise error_type("The SSH profile is invalid.") from error
    if not isinstance(profile, dict):
        raise error_type("The SSH profile is invalid.")
    host = str(profile.get("host", "")).strip()
    username = str(profile.get("username", "")).strip()
    password = str(profile.get("password", ""))
    fingerprint = str(profile.get("host_key_fingerprint", "")).strip()
    try:
        port = int(profile.get("port", 22))
    except (TypeError, ValueError) as error:
        raise error_type("The SSH port is invalid.") from error
    if not host or len(host) > 253 or not username or len(username) > 64:
        raise error_type("Enter a valid SSH host and user name.")
    if not re.fullmatch(r"[A-Za-z0-9._-]+", username):
        raise error_type("The SSH user name contains unsupported characters.")
    if not 1 <= port <= 65535 or not password or len(password.encode("utf-8")) > 4096:
        raise error_type("Enter a valid SSH port and password.")
    if not re.fullmatch(r"SHA256:[A-Za-z0-9+/]{20,60}={0,2}", fingerprint):
        raise error_type("Confirm the server's SHA256 host-key fingerprint.")
    try:
        resolved = {
            ipaddress.ip_address(item[4][0])
            for item in socket.getaddrinfo(host, port, type=socket.SOCK_STREAM)
        }
    except (OSError, ValueError) as error:
        raise error_type("The SSH host could not be resolved.") from error
    if not resolved or any(
        not address.is_private
        or address.is_loopback
        or address.is_link_local
        or address.is_multicast
        or address.is_unspecified
        for address in resolved
    ):
        raise error_type("Cluster nodes must use a private, non-loopback LAN address.")
    return {
        "host": host,
        "port": port,
        "username": username,
        "password": password,
        "host_key_fingerprint": fingerprint,
        "sudo_uses_login_password": bool(profile.get("sudo_uses_login_password", True)),
    }
