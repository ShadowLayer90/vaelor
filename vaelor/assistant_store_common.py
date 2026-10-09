"""Neutral primitives shared by every assistant/cluster SQLite store.

These helpers began in :mod:`vaelor.mcp_catalog` (F1) and were lifted here so the
F2 skills library, and the F3/F4/F5 stores that follow, reuse one home rather
than importing PRIVATE names across modules. Nothing here is store-specific: the
clock wrapper, the request-digest helper, and the secret-shaping guards are the
same for a catalog server, a skill manifest, or any later record that must keep
credentials in the broker and reference them by id.

Validators raise :class:`StoreInputError`, a neutral ``ValueError``. Each calling
store wraps these in a thin local helper that translates ``StoreInputError`` into
its own module error type, so a store's external behaviour is unchanged.
"""

from __future__ import annotations

import hashlib
import json
import time
from typing import Any, Mapping


class StoreInputError(ValueError):
    """A safe, user-presentable input error raised by the shared validators."""


#: Field names whose *presence anywhere in an input* means a secret value was
#: passed where only an identifier belongs.
SENSITIVE_NAMES = {
    "secret", "token", "password", "authorization", "credential_value",
    "plaintext", "api_key", "apikey", "bearer", "private_key",
}


def now() -> float:
    return time.time()


def digest(value: Mapping[str, Any]) -> str:
    return hashlib.sha256(
        json.dumps(value, sort_keys=True, separators=(",", ":"), default=str).encode()
    ).hexdigest()


def contains_sensitive(value: Any) -> bool:
    if isinstance(value, Mapping):
        return any(
            any(name in str(key).lower() for name in SENSITIVE_NAMES)
            or contains_sensitive(item)
            for key, item in value.items()
        )
    if isinstance(value, (list, tuple)):
        return any(contains_sensitive(item) for item in value)
    return False


def reject_sensitive(value: Mapping[str, Any], field: str = "input") -> None:
    if contains_sensitive(value):
        raise StoreInputError(
            "{} cannot contain secret-shaped fields; store credentials in the "
            "broker and reference them by id.".format(field)
        )


def credential_id(value: Any) -> str:
    """An *identifier* for a broker credential, never the secret itself."""
    text = str(value or "").strip()
    if not text:
        return ""
    # Broker ids are opaque tokens; keep the shape tight - bounded length, no
    # secret-shaped substring, and only id-safe characters - so a value cannot
    # be smuggled in as an "id".
    if (len(text) > 120
            or any(name in text.lower() for name in SENSITIVE_NAMES)
            or not all(character.isalnum() or character in "_-.:" for character in text)):
        raise StoreInputError("credential_id must be a broker reference, not a secret.")
    return text
