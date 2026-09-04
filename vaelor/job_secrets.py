"""Decide whether a job payload carries a credential, so it is never stored."""

from __future__ import annotations

import re
from typing import Any


SENSITIVE_KEYS = {"password", "passwd", "token", "secret", "api_key", "apikey"}
SENSITIVE_KEY_PARTS = {
    "authorization", "cookie", "credentials", "password", "passwd",
    "private_key", "secret", "token", "api_key", "apikey", "access_key",
}


def sensitive_key(key: Any) -> bool:
    normalized = re.sub(r"[^a-z0-9]+", "_", str(key).strip().lower()).strip("_")
    return (
        normalized in SENSITIVE_KEYS
        or normalized in SENSITIVE_KEY_PARTS
        or bool(re.search(
            r"(?:^|_)(?:password|passwd|token|secret|api_key|apikey|authorization|"
            r"private_key|client_secret|access_key|credential|credentials)(?:_|$)",
            normalized,
        ))
    )


def sensitive_string(value: str) -> bool:
    return bool(
        re.search(
            r"""(?ix)
            (?:^|[\s,;])(?:[A-Z0-9_]*(?:PASSWORD|PASSWD|TOKEN|SECRET|API[_-]?KEY|AUTHORIZATION|PRIVATE[_-]?KEY|CLIENT[_-]?SECRET))\s*[:=]\s*\S+
            |authorization\s*:\s*(?:bearer|basic)\s+\S+
            |-----BEGIN\s+(?:RSA\s+|EC\s+|OPENSSH\s+)?PRIVATE\s+KEY-----
            |https?://[^/\s:@]+:[^@\s]+@
            """,
            value,
        )
    )


def contains_secret(value: Any) -> bool:
    if isinstance(value, dict):
        for key, child in value.items():
            if sensitive_key(key):
                return True
            if contains_secret(child):
                return True
    elif isinstance(value, list):
        return any(contains_secret(item) for item in value)
    elif isinstance(value, str):
        return sensitive_string(value)
    return False
