"""Deterministic, server-owned routing from chat requests to custom agents."""

from __future__ import annotations

import re
from typing import Any, Iterable, Mapping


_REQUEST_WORDS = {"ask", "have", "run", "use"}
_MATCH_STOPWORDS = {"agent", "assistant", "custom", "my", "the", "a", "an"}


def _tokens(value: Any) -> set[str]:
    return set(re.findall(r"[a-z0-9]+", str(value).lower()))


def _safe_app_grants(profile: Mapping[str, Any]) -> list[dict[str, Any]]:
    """Return bounded review metadata without transport or credential details."""
    result: list[dict[str, Any]] = []
    for raw in list(profile.get("app_grants_summary", []))[:10]:
        if not isinstance(raw, Mapping):
            continue
        operations = []
        for operation in list(raw.get("operations", []))[:20]:
            if not isinstance(operation, Mapping):
                continue
            access = str(operation.get("access", "read")).strip().lower()
            operations.append({
                "id": str(operation.get("id", ""))[:80],
                "name": str(operation.get("name", ""))[:100],
                "access": access if access in {"read", "write"} else "read",
                "risk": str(operation.get("risk", "low"))[:40],
            })
        result.append({
            "grant_id": str(raw.get("grant_id", ""))[:96],
            "app_instance_id": str(raw.get("app_instance_id", ""))[:160],
            "app_name": str(raw.get("app_name", "Installed app"))[:100],
            "manifest_version": str(raw.get("manifest_version", ""))[:40],
            "manifest_digest": str(raw.get("manifest_digest", ""))[:64],
            "operations": operations,
        })
    return result


def custom_agent_proposal(
    message: Any,
    profiles: Iterable[Mapping[str, Any]],
    *,
    explicit: bool = False,
) -> dict[str, Any] | None:
    """Match only an explicit, unambiguous request to an enabled custom agent."""
    message_tokens = _tokens(message)
    if not explicit and (not message_tokens.intersection(_REQUEST_WORDS) or "agent" not in message_tokens):
        return None
    enabled = [profile for profile in profiles if profile.get("enabled", True)]
    candidates = []
    for profile in enabled:
        distinctive = {
            token for token in _tokens(profile.get("name", "")) - _MATCH_STOPWORDS
            if len(token) > 1
        }
        score = len(message_tokens.intersection(distinctive))
        if score:
            candidates.append((score, len(distinctive), profile))
    if not candidates:
        if len(enabled) != 1:
            return None
        selected = enabled[0]
    else:
        candidates.sort(key=lambda item: (item[0], -item[1]), reverse=True)
        if len(candidates) > 1 and candidates[0][0] == candidates[1][0]:
            return None
        selected = candidates[0][2]

    app_grants = _safe_app_grants(selected)
    connector_names = [
        str(item.get("name", "")).strip()[:100]
        for item in list(selected.get("connectors", []))[:10]
        if isinstance(item, Mapping) and str(item.get("name", "")).strip()
    ]
    app_names = [item["app_name"] for item in app_grants if item["app_name"]]
    return {
        "profile_id": str(selected["id"]),
        "profile_name": str(selected["name"])[:100],
        "profile_version": int(selected.get("version", 0)),
        "task": str(message).strip()[:4000],
        "capabilities": [str(item)[:100] for item in selected.get("scopes", [])],
        "integrations": list(dict.fromkeys(connector_names + app_names)),
        "app_grants": app_grants,
    }
