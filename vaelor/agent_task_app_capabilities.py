"""Private policy helpers for custom-agent app capability snapshots and plans."""

from __future__ import annotations

import json
from typing import Any, Mapping


_APP_GRANT_LIMIT = 32
_APP_OPERATION_LIMIT = 64


def _safe_text(value: Any, maximum: int = 160) -> str:
    return str(value or "").strip()[:maximum]


def _context_value(source: Any, name: str, default: Any = None) -> Any:
    if isinstance(source, Mapping):
        return source.get(name, default)
    return getattr(source, name, default)


def _app_grant_source(dependency: Any, actor: str, agent_id: str, version: int, definition: Mapping[str, Any]) -> tuple[list[Any], str]:
    """Read app authorization facts through a small, optional dependency seam."""
    if dependency is None:
        return [], "not_configured"
    try:
        if hasattr(dependency, "snapshot_for_agent"):
            method = dependency.snapshot_for_agent
            try:
                raw = method(actor, agent_id, version, definition=definition)
            except TypeError:
                raw = method(actor, agent_id, version)
        elif hasattr(dependency, "snapshot"):
            method = dependency.snapshot
            try:
                raw = method(actor, agent_id, version, definition=definition)
            except TypeError:
                raw = method(actor, agent_id, version)
        elif callable(dependency):
            try:
                raw = dependency(actor, agent_id, version, definition)
            except TypeError:
                raw = dependency(actor, agent_id, version)
        elif hasattr(dependency, "list"):
            method = dependency.list
            try:
                raw = method(actor, agent_id=agent_id, limit=_APP_GRANT_LIMIT)
            except TypeError:
                raw = method(actor, agent_id, _APP_GRANT_LIMIT)
        else:
            return [], "unavailable"
    except Exception:
        return [], "unavailable"
    grants = raw.get("grants", raw.get("app_grants", [])) if isinstance(raw, Mapping) else raw
    return (grants[:_APP_GRANT_LIMIT], "available") if isinstance(grants, list) else ([], "unavailable")


def _registry_facts(dependency: Any, grant: Mapping[str, Any]) -> tuple[Any, Any]:
    registry = _context_value(dependency, "registry")
    if registry is None:
        return None, None
    try:
        app_id = str(grant.get("app_instance_id", ""))
        app = registry.get_app_instance(app_id) if hasattr(registry, "get_app_instance") else None
        digest = str(grant.get("manifest_digest", ""))
        manifest = registry.get_manifest(digest) if hasattr(registry, "get_manifest") else None
        return app, manifest
    except Exception:
        return None, None


def _safe_app_grant_snapshot(dependency: Any, actor: str, agent_id: str, version: int, definition: Mapping[str, Any]) -> tuple[list[dict[str, Any]], str]:
    """Project only server-owned, non-secret pins into a task approval snapshot."""
    raw_grants, status = _app_grant_source(dependency, actor, agent_id, version, definition)
    snapshot: list[dict[str, Any]] = []
    for raw in raw_grants:
        if not isinstance(raw, Mapping):
            continue
        if raw.get("revoked") or raw.get("revoked_at") is not None:
            continue
        raw_actor = raw.get("actor", raw.get("owner"))
        if raw_actor is not None and str(raw_actor) != actor:
            continue
        if str(raw.get("agent_id", raw.get("agent", agent_id))) != agent_id:
            continue
        try:
            if int(raw.get("agent_version", raw.get("version", -1))) != version:
                continue
        except (TypeError, ValueError):
            continue
        app, manifest = _registry_facts(dependency, raw)
        app_instance = raw.get("app_instance") if isinstance(raw.get("app_instance"), Mapping) else {}
        manifest_value = raw.get("manifest") if isinstance(raw.get("manifest"), Mapping) else {}
        app_name = _safe_text(raw.get("app_name", raw.get("app_label", app_instance.get("app_label", app_instance.get("name", "")))) or (app.get("app_label") if isinstance(app, Mapping) else "") or (getattr(manifest, "app_label", "") if manifest is not None else "") or raw.get("app_instance_id", ""), 120)
        manifest_version = raw.get("manifest_version", manifest_value.get("manifest_version"))
        app_version = raw.get("app_version", manifest_value.get("app_version"))
        if manifest_version is None and manifest is not None:
            manifest_version = getattr(manifest, "manifest_version", None)
        if app_version is None and manifest is not None:
            app_version = getattr(manifest, "app_version", None)
        raw_operations = raw.get("operations", raw.get("operation_details", []))
        operation_map: dict[str, Mapping[str, Any]] = {}
        if isinstance(raw_operations, Mapping):
            operation_map = {str(key): value for key, value in raw_operations.items() if isinstance(value, Mapping)}
        elif isinstance(raw_operations, list):
            operation_map = {str(item.get("operation_id", item.get("id", ""))): item for item in raw_operations if isinstance(item, Mapping)}
        operation_ids = raw.get("operation_ids", raw.get("operations", []))
        if isinstance(operation_ids, Mapping):
            operation_ids = list(operation_ids)
        if isinstance(operation_ids, list) and operation_ids and isinstance(operation_ids[0], Mapping):
            operation_ids = [item.get("operation_id", item.get("id", "")) for item in operation_ids]
        if not isinstance(operation_ids, list):
            operation_ids = list(operation_map)
        if not operation_ids and manifest is not None:
            operation_ids = [getattr(item, "operation_id", "") for item in getattr(manifest, "operations", ())]
        operations: list[dict[str, Any]] = []
        for operation_id in operation_ids[:_APP_OPERATION_LIMIT]:
            operation_id = str(operation_id).strip()[:128]
            if not operation_id:
                continue
            operation = operation_map.get(operation_id)
            manifest_operation = manifest.operation(operation_id) if manifest is not None and hasattr(manifest, "operation") else None
            operations.append({
                "operation_id": operation_id,
                "mode": _safe_text((operation or {}).get("mode", getattr(manifest_operation, "mode", "unknown")), 16).lower(),
                "risk": _safe_text((operation or {}).get("risk", getattr(manifest_operation, "risk", "unknown")), 32).lower(),
                "approval": _safe_text((operation or {}).get("approval", (operation or {}).get("approval_policy", getattr(manifest_operation, "approval_policy", "unknown"))), 32).lower(),
            })
        if not operations:
            continue
        manifest_digest = raw.get("manifest_digest", manifest_value.get("manifest_digest", app.get("manifest_digest", "") if isinstance(app, Mapping) else ""))
        snapshot.append({
            "grant_id": _safe_text(raw.get("grant_id", raw.get("id", "")), 160),
            "app_instance_id": _safe_text(raw.get("app_instance_id", app_instance.get("instance_id", "")), 128),
            "app_name": app_name,
            "manifest_digest": _safe_text(manifest_digest, 128),
            "manifest_version": int(manifest_version) if isinstance(manifest_version, int) and not isinstance(manifest_version, bool) else None,
            "app_version": _safe_text(app_version, 64) if app_version is not None else "",
            "connection_id": _safe_text(raw.get("connection_id", ""), 128),
            "connection_secret_version": int(raw.get("connection_secret_version", 0)) if str(raw.get("connection_secret_version", "")).isdigit() else 0,
            "operations": operations,
        })
    return snapshot, status


def plan_custom_app_calls(agent: Any, task: str, definition: Mapping[str, Any], grants: list[Mapping[str, Any]], timeout_seconds: int, mode: str = "") -> list[dict[str, Any]]:
    """Ask the selected model for a strict plan over already-pinned app operations."""
    from .inference_client import chat_completion, inference_timeout, parse_model_object
    from .inference_client import allowed_inference_endpoint
    from .model_calibration import APP_CALL_PLAN_SCHEMA
    from .model_profiles import with_structured_response_format
    from .provider_runtime import generation_parameters, provider_system_prompt
    import urllib.request

    if not grants:
        return []
    connection = agent._connection(mode)
    if not connection or not allowed_inference_endpoint(connection):
        raise ValueError("App capability planning requires a safe selected AI model.")
    safe_grants = json.loads(json.dumps(grants[:_APP_GRANT_LIMIT], separators=(",", ":"), default=str))
    headers = {"Content-Type": "application/json", "Accept": "application/json"}
    if connection.get("api_key"):
        headers["Authorization"] = "Bearer " + connection["api_key"]
    model = str(connection.get("model", ""))
    if not model:
        request = urllib.request.Request(connection["base_url"] + "/models", headers=headers)
        with urllib.request.urlopen(request, timeout=timeout_seconds) as response:
            model = str(json.loads(response.read(1024 * 1024).decode("utf-8"))["data"][0]["id"])
    prompt = (
        "Choose zero or more calls for this custom agent from only the supplied pinned operations. "
        "Never invent grants or operations. Never include URLs, endpoints, headers, credentials, "
        "tokens, secrets, shell, or code. Return strict JSON only with exactly "
        '{"calls":[{"grant_id":"","operation_id":"","input":{}}]}. '
        "Use at most five calls. Read calls may be executed. Write calls are preview-only and require "
        "separate human approval. Return an empty calls list when no app call is needed."
    )
    body = chat_completion(
        dict(connection),
        with_structured_response_format({
            "model": model,
            "messages": [
                {"role": "system", "content": provider_system_prompt(prompt, dict(connection))},
                {"role": "user", "content": json.dumps({
                    "task": str(task)[:4000],
                    "agent_name": str(definition.get("name", "Custom agent"))[:120],
                    "pinned_app_operations": safe_grants,
                }, ensure_ascii=False, separators=(",", ":"))},
            ],
            **generation_parameters(dict(connection), max_tokens=700, temperature=0),
        }, connection, APP_CALL_PLAN_SCHEMA),
        headers, inference_timeout(dict(connection), timeout_seconds),
    )
    parsed = parse_model_object(body["choices"][0]["message"]["content"])
    if not isinstance(parsed, dict) or set(parsed) != {"calls"} or not isinstance(parsed["calls"], list):
        raise ValueError("The model returned an invalid app-call plan.")
    allowed = {
        str(grant.get("grant_id")): {str(item.get("operation_id")): item for item in grant.get("operations", []) if isinstance(item, Mapping)}
        for grant in safe_grants if isinstance(grant, Mapping)
    }

    def unsafe(value: Any) -> bool:
        if isinstance(value, Mapping):
            for key, child in value.items():
                parts = {part for part in str(key).lower().replace("-", "_").split("_") if part}
                if parts & {"url", "uri", "endpoint", "header", "authorization", "credential", "password", "secret", "token", "api", "private"}:
                    return True
                if unsafe(child):
                    return True
        elif isinstance(value, list):
            return any(unsafe(child) for child in value)
        elif isinstance(value, str) and value.lower().startswith(("http://", "https://")):
            return True
        return False

    result = []
    for item in parsed["calls"][:5]:
        if not isinstance(item, Mapping) or set(item) != {"grant_id", "operation_id", "input"}:
            raise ValueError("The model returned an invalid app call.")
        grant_id = str(item["grant_id"])[:160]
        operation_id = str(item["operation_id"])[:128]
        parameters = item["input"]
        if not isinstance(parameters, dict) or unsafe(parameters):
            raise ValueError("The model returned unsafe app-call input.")
        if grant_id not in allowed or operation_id not in allowed[grant_id]:
            raise ValueError("The model selected an operation outside the pinned grants.")
        encoded = json.dumps(parameters, ensure_ascii=False, separators=(",", ":"), allow_nan=False)
        if len(encoded.encode("utf-8")) > 16 * 1024:
            raise ValueError("The model returned oversized app-call input.")
        result.append({"grant_id": grant_id, "operation_id": operation_id, "input": json.loads(encoded)})
    return result
