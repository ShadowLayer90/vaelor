"""Authenticated HTTP boundary for installed-app capabilities.

This module deliberately has no composition side effects.  The application
composition layer registers :func:`register_integration_routes` and supplies
the four control-plane callbacks documented by the P0 contract.

The route layer is an adapter, not an authorization source: all identities,
manifest digests, connection ownership, current agent versions, and lifecycle
facts are resolved from the server-side stores on every mutation.
"""

from __future__ import annotations

import time
import uuid
from datetime import datetime, timezone
from typing import Any, Mapping, Optional

from flask import g, request

from .agent_app_grants import AgentAppGrantError
from .api_common import ApiContext, payload as _payload
from .app_capability_manifest import AppCapabilityManifest, AppOperation
from .app_capability_registry import (
    AppCapabilityRegistry,
    RegistryError,
)
from .custom_agents import CustomAgentError
from .integration_connections import IntegrationConnectionError


_PREVIEW_TTL_SECONDS = 300
_MAX_PREVIEWS = 128
_APP_STATUSES = {"active", "degraded", "stopped", "incompatible", "removed"}
_SENSITIVE_KEYS = {
    "actor", "api_key", "authorization", "credential",
    "endpoint", "password", "private_key", "secret", "token", "url", "uri",
}


def _now() -> float:
    return time.time()


def _iso(timestamp: Optional[float]) -> Optional[str]:
    if timestamp is None:
        return None
    return datetime.fromtimestamp(float(timestamp), timezone.utc).isoformat()


def _text(value: Any, field: str, maximum: int = 256) -> str:
    result = str(value or "").strip()
    if not result or len(result) > maximum:
        raise ValueError(f"{field} is invalid.")
    return result


def _body() -> dict[str, Any]:
    value = request.get_json(silent=True)
    if not isinstance(value, dict):
        raise ValueError("A JSON object is required.")
    return value


def _reject_sensitive_keys(value: Any, path: str = "request") -> None:
    if isinstance(value, Mapping):
        for key, child in value.items():
            normalized = "".join(
                character.lower() if character.isalnum() else "_"
                for character in str(key)
            )
            if normalized in {"credential_ref", "credentialref"}:
                # This is the only credential-shaped field accepted here, and
                # the connection store validates it as an opaque broker ID.
                continue
            parts = {part for part in normalized.split("_") if part}
            if parts & _SENSITIVE_KEYS:
                raise ValueError(f"{path}.{key} is not accepted by this endpoint.")
            _reject_sensitive_keys(child, f"{path}.{key}")
    elif isinstance(value, (list, tuple)):
        for index, child in enumerate(value):
            _reject_sensitive_keys(child, f"{path}[{index}]")


def _value(body: Mapping[str, Any], *names: str, default: Any = None) -> Any:
    for name in names:
        if name in body:
            return body[name]
    return default


def _version(value: Any, field: str = "version") -> int:
    text = _text(value, field, 32)
    if text.lower().startswith("v"):
        text = text[1:]
    try:
        result = int(text)
    except ValueError as error:
        raise ValueError(f"{field} is invalid.") from error
    if result < 1:
        raise ValueError(f"{field} is invalid.")
    return result


def _error_code(error: Exception) -> str:
    message = str(error).lower()
    if "not found" in message or "was not found" in message:
        return "not_found"
    if "stale" in message or "changed" in message or "incompatible" in message:
        return "stale_request"
    if "unhealthy" in message or "health" in message:
        return "unavailable"
    return "invalid_request"


def _status_for_ui(state: str) -> str:
    if state in _APP_STATUSES:
        return state
    if state in {"discovered", "configured"}:
        return "stopped"
    return "degraded"


def _recovery_reasons(app: Mapping[str, Any]) -> list[str]:
    reasons: list[str] = []
    state = str(app.get("state", ""))
    health = str(app.get("health", ""))
    compatibility = str(app.get("compatibility", ""))
    if state != "active":
        reasons.append({
            "removed": "Restore or reinstall the managed app before granting access.",
            "stopped": "Start the managed app before granting access.",
            "degraded": "Resolve the app lifecycle issue before granting access.",
        }.get(state, "Start the managed app before granting access."))
    if health != "healthy":
        reasons.append("Restore the app health check before using its capabilities.")
    if compatibility != "compatible":
        reasons.append("Review the installed manifest and recreate affected grants.")
    return list(dict.fromkeys(reasons))


def _safe_connection(item: Mapping[str, Any]) -> dict[str, Any]:
    """Return UI-safe connection metadata; credential references never leave the API."""
    status = str(item.get("status") or item.get("test_status") or "pending")
    expires_at = item.get("expires_at")
    if status not in {"revoked", "degraded"} and expires_at is not None and float(expires_at) <= _now():
        status = "expired"
    if status == "failed":
        status = "degraded"
    issue = {
        "revoked": "Connection revoked.",
        "expired": "Connection expired; rotate or reconnect it.",
        "degraded": "The last connection test did not pass.",
        "pending": "Test the connection before granting access.",
    }.get(status)
    return {
        "id": str(item.get("id", "")),
        "label": str(item.get("provider", "Integration connection")),
        "status": status,
        "scopes": [str(scope) for scope in item.get("scopes", [])],
        "expiresAt": _iso(expires_at),
        "lastTestedAt": _iso(item.get("last_test_at")),
        "issue": issue,
    }


def _safe_grant(item: Mapping[str, Any], agent: Optional[Mapping[str, Any]] = None,
                status: Optional[str] = None, reason: str = "") -> dict[str, Any]:
    agent_name = str((agent or {}).get("name") or item.get("agent_id") or "Custom agent")
    return {
        "id": str(item.get("id", "")),
        "status": status or ("revoked" if item.get("revoked") else "active"),
        "agentName": agent_name,
        "agentVersionLabel": f"v{int(item.get('agent_version', 0))}",
        "operationIds": [str(operation) for operation in item.get("operation_ids", [])],
        "connectionId": str(item.get("connection_id", "")) or None,
        "blockedReason": reason or None,
    }


def _agent_version(agent: Mapping[str, Any]) -> int:
    return _version(agent.get("version"), "agent_version")


def _require_services(callbacks: Mapping[str, Any]):
    registry = callbacks.get("app_capability_registry")
    connections = callbacks.get("integration_connections")
    grants = callbacks.get("agent_app_grants")
    agents = callbacks.get("custom_agents")
    missing = [
        name for name, value in (
            ("app_capability_registry", registry),
            ("integration_connections", connections),
            ("agent_app_grants", grants),
            ("custom_agents", agents),
        ) if value is None
    ]
    if missing:
        raise RuntimeError("Integration services are unavailable: " + ", ".join(missing))
    return registry, connections, grants, agents


def _app_and_manifest(registry: AppCapabilityRegistry, instance_id: str):
    app = registry.get_app_instance(_text(instance_id, "app_instance_id", 128))
    if app is None:
        raise LookupError("The app instance was not found.")
    manifest = registry.get_manifest(app["manifest_digest"])
    if manifest is None:
        raise LookupError("The installed app manifest is unavailable.")
    return app, manifest


def _assert_app_ready(app: Mapping[str, Any], manifest: AppCapabilityManifest) -> None:
    if app.get("state") != "active":
        raise ValueError("The managed app is not active.")
    if app.get("health") != "healthy":
        raise ValueError("The managed app is unhealthy.")
    if app.get("compatibility") != "compatible":
        raise ValueError("The managed app is incompatible with its pinned manifest.")
    if app.get("manifest_digest") != manifest.manifest_digest:
        raise ValueError("The installed app manifest digest is stale.")
    if app.get("observed_manifest_digest", manifest.manifest_digest) != manifest.manifest_digest:
        raise ValueError("The installed app manifest changed; review the app before granting access.")


def _assert_connection_ready(item: Mapping[str, Any], actor: str) -> int:
    if item.get("actor") != actor:
        raise LookupError("The integration connection was not found.")
    if item.get("revoked") or item.get("revoked_at") is not None:
        raise ValueError("The integration connection is revoked.")
    if str(item.get("test_status")) != "healthy":
        raise ValueError("The integration connection has not passed its health test.")
    expires_at = item.get("expires_at")
    if expires_at is not None and float(expires_at) <= _now():
        raise ValueError("The integration connection has expired.")
    return _version(item.get("secret_version"), "secret_version")


def _operation_projection(operation: AppOperation, available: bool, reason: str = "") -> dict[str, Any]:
    return {
        "id": operation.operation_id,
        "label": operation.label,
        "description": f"{operation.mode.title()} operation available through the installed app.",
        "kind": operation.mode,
        "risk": operation.risk,
        "availability": "available" if available else "incompatible",
        "unavailableReason": reason or None,
    }


def _detail(
    registry: AppCapabilityRegistry,
    connections: Any,
    grants: Any,
    agents: Any,
    actor: str,
    app: Mapping[str, Any],
    manifest: AppCapabilityManifest,
) -> dict[str, Any]:
    ready = (
        app.get("state") == "active"
        and app.get("health") == "healthy"
        and app.get("compatibility") == "compatible"
        and app.get("observed_manifest_digest", manifest.manifest_digest) == manifest.manifest_digest
    )
    unavailable_reason = " ".join(_recovery_reasons(app))
    operation_items = [
        _operation_projection(operation, ready, unavailable_reason)
        for operation in manifest.operations
    ]
    connection_items = [
        _safe_connection(item) for item in connections.list(actor, limit=100)
    ]
    agent_items: list[dict[str, Any]] = []
    for agent in agents.list(actor, include_disabled=True):
        try:
            version = _agent_version(agent)
        except (TypeError, ValueError):
            continue
        agent_items.append({
            "agentId": str(agent["id"]),
            "agentName": str(agent.get("name", agent["id"])),
            "versionId": str(version),
            "versionLabel": f"v{version}",
            "compatibleOperationIds": [operation.operation_id for operation in manifest.operations]
            if ready and agent.get("enabled", True) else [],
            "status": "active" if agent.get("enabled", True) else "archived",
        })
    grant_items = [
        item for item in grants.list(actor, limit=200)
        if item.get("app_instance_id") == app["instance_id"]
    ]
    evaluated: list[dict[str, Any]] = []
    for item in grant_items:
        agent = agents.get(item["agent_id"], actor)
        decision = grants.evaluate(
            item["id"], actor,
            agent_version=(agent or {}).get("version"),
            app_state=app.get("state"),
            manifest_digest=app.get("manifest_digest"),
            available_operations=[operation.operation_id for operation in manifest.operations],
        )
        reasons = decision.get("reasons", [])
        evaluated.append(_safe_grant(
            item, agent, str(decision.get("status", "blocked")),
            "; ".join(str(reason).replace("_", " ").capitalize() for reason in reasons),
        ))
    dependents = registry.list_dependents(app["instance_id"], limit=100).get("items", [])
    dependents_projection = [
        {
            "id": str(item.get("dependent_id", "")),
            "label": str(item.get("reference", {}).get("label") or item.get("dependent_id", "")),
            "kind": "agent" if item.get("dependent_type") == "agent" else "task" if item.get("dependent_type") == "task" else "automation",
            "status": "blocked" if not ready else "active",
            "impact": str(item.get("reference", {}).get("impact") or "Capability access depends on this app."),
            "recoveryAction": str(item.get("reference", {}).get("recovery_action") or "Review or revoke the dependent grant."),
        }
        for item in dependents
    ]
    dependents_projection.extend({
        "id": item["id"],
        "label": item["agentName"],
        "kind": "agent",
        "status": "blocked" if item["status"] != "active" else "active",
        "impact": "This agent version has access to the installed app.",
        "recoveryAction": "Review the grant or create a replacement version.",
    } for item in evaluated)
    status = _status_for_ui(str(app.get("state", "degraded")))
    return {
        "appInstanceId": app["instance_id"],
        "appName": app["app_label"],
        "appVersion": manifest.app_version,
        "manifestVersion": str(manifest.manifest_version),
        "status": status,
        "healthSummary": "Healthy and ready for capability access." if ready else (" ".join(_recovery_reasons(app)) or "Capability access is blocked until the app is ready."),
        "compatibilitySummary": "Manifest is pinned and compatible." if app.get("compatibility") == "compatible" else "Installed manifest is not compatible with the grant pin.",
        "connectionRequired": bool(manifest.requires_connection or any(operation.requires_connection for operation in manifest.operations)),
        "operations": operation_items,
        "connections": connection_items,
        "agentVersions": agent_items,
        "existingGrant": evaluated[0] if evaluated else None,
        "grants": evaluated,
        "dependents": dependents_projection,
        "recoveryActions": _recovery_reasons(app),
    }


def _grant_request(body: Mapping[str, Any]) -> tuple[str, int, Optional[str], list[str], Optional[int]]:
    _reject_sensitive_keys(body)
    app_instance_id = _text(_value(body, "appInstanceId", "app_instance_id"), "app_instance_id", 128)
    agent_version = _version(_value(body, "agentVersionId", "agent_version", "version"), "agent_version")
    connection_id = _value(body, "connectionId", "connection_id")
    if connection_id is not None:
        connection_id = _text(connection_id, "connection_id", 128)
    operations = _value(body, "operationIds", "operation_ids")
    if not isinstance(operations, list) or not operations:
        raise ValueError("operationIds must be a non-empty list.")
    operation_ids = sorted({_text(item, "operation_id", 128) for item in operations})
    supplied_manifest_version = _value(body, "manifestVersion", "manifest_version")
    manifest_version = None if supplied_manifest_version is None else _version(supplied_manifest_version, "manifest_version")
    return app_instance_id, agent_version, connection_id, operation_ids, manifest_version


def register_integration_routes(context: ApiContext) -> None:
    """Register the authenticated P0 integration endpoints on ``context``."""
    blueprint = context.blueprint
    callbacks = context.callbacks
    security = context.security
    require_auth = context.require_auth
    previews: dict[str, dict[str, Any]] = {}

    def services():
        refresh = callbacks.get("integration_reconcile")
        if refresh is not None:
            try:
                refresh()
            except Exception as error:
                raise RuntimeError("Current managed-app state could not be verified.") from error
        return _require_services(callbacks)

    def audit(action: str, result: str, target: str = "", details: Optional[Mapping[str, Any]] = None):
        try:
            security.audit(
                g.auth_session.username, action, result, target=target,
                remote_addr=request.remote_addr or "", details=dict(details or {}),
            )
        except (AttributeError, OSError, TypeError, ValueError):
            pass

    @blueprint.get("/integrations/apps")
    @require_auth("operator")
    def integration_apps():
        try:
            registry, connections, grants, agents = services()
            result = registry.list_app_instances(
                limit=int(request.args.get("limit", 50)), cursor=request.args.get("cursor")
            )
            actor = g.auth_session.username
            items = []
            for item in result["items"]:
                manifest = registry.get_manifest(item["manifest_digest"])
                if manifest is None:
                    continue
                items.append(_detail(registry, connections, grants, agents, actor, item, manifest))
            return _payload({"items": items, "nextCursor": result.get("next_cursor"), "limit": result.get("limit")})
        except (RegistryError, ValueError, RuntimeError) as error:
            return _payload(error={"code": _error_code(error), "message": str(error)}, status=503 if isinstance(error, RuntimeError) else 400)

    @blueprint.get("/integrations/apps/<instance_id>")
    @require_auth("operator")
    def integration_app_detail(instance_id):
        try:
            registry, connections, grants, agents = services()
            app, manifest = _app_and_manifest(registry, instance_id)
            return _payload(_detail(registry, connections, grants, agents, g.auth_session.username, app, manifest))
        except LookupError as error:
            return _payload(error={"code": "not_found", "message": str(error)}, status=404)
        except (RegistryError, ValueError, RuntimeError) as error:
            return _payload(error={"code": _error_code(error), "message": str(error)}, status=503 if isinstance(error, RuntimeError) else 400)

    @blueprint.get("/integrations/apps/<instance_id>/dependents")
    @require_auth("operator")
    def integration_app_dependents(instance_id):
        try:
            registry, _connections, grants, _agents = services()
            _app_and_manifest(registry, instance_id)
            result = registry.list_dependents(instance_id, limit=int(request.args.get("limit", 50)), cursor=request.args.get("cursor"))
            grant_dependents = grants.list(g.auth_session.username, limit=200)
            result["items"].extend({
                "dependent_type": "agent",
                "dependent_id": item["id"],
                "reference": {"impact": "This agent version has a grant for the app."},
            } for item in grant_dependents if item.get("app_instance_id") == instance_id)
            result["nextCursor"] = result.pop("next_cursor", None)
            return _payload(result)
        except LookupError as error:
            return _payload(error={"code": "not_found", "message": str(error)}, status=404)
        except (RegistryError, ValueError, RuntimeError) as error:
            return _payload(error={"code": _error_code(error), "message": str(error)}, status=503 if isinstance(error, RuntimeError) else 400)

    @blueprint.get("/integrations/connections")
    @require_auth("operator")
    def integration_connections():
        try:
            _registry, connections, _grants, _agents = services()
            return _payload({"items": [_safe_connection(item) for item in connections.list(g.auth_session.username, limit=200)]})
        except (IntegrationConnectionError, ValueError, RuntimeError) as error:
            return _payload(error={"code": _error_code(error), "message": str(error)}, status=503 if isinstance(error, RuntimeError) else 400)

    @blueprint.post("/integrations/connections")
    @require_auth("administrator", csrf=True)
    def integration_connection_create():
        try:
            body = _body()
            _reject_sensitive_keys(body)
            allowed = {"provider", "credential_ref", "credentialRef", "scopes", "secret_version", "secretVersion", "expires_at", "expiresAt", "idempotency_key", "idempotencyKey"}
            if set(body) - allowed:
                raise ValueError("The connection request contains unsupported fields.")
            _registry, connections, _grants, _agents = services()
            item = connections.create(
                g.auth_session.username,
                _text(_value(body, "provider"), "provider", 100),
                _text(_value(body, "credential_ref", "credentialRef"), "credential_ref", 128),
                _value(body, "scopes", default=[]),
                secret_version=_value(body, "secret_version", "secretVersion", default=1),
                expires_at=_value(body, "expires_at", "expiresAt"),
                idempotency_key=_value(body, "idempotency_key", "idempotencyKey"),
            )
            audit("integrations.connection.create", "success", item["id"])
            return _payload(_safe_connection(item), status=201)
        except (IntegrationConnectionError, ValueError, RuntimeError, TypeError) as error:
            return _payload(error={"code": _error_code(error), "message": str(error)}, status=503 if isinstance(error, RuntimeError) else 400)

    @blueprint.post("/integrations/apps/<instance_id>/connections")
    @require_auth("administrator", csrf=True)
    def integration_app_connection_create(instance_id):
        # Keep the typed UI's app-scoped creation flow while delegating storage
        # to the same connection endpoint contract.
        try:
            _app_and_manifest(services()[0], instance_id)
        except LookupError as error:
            return _payload(error={"code": "not_found", "message": str(error)}, status=404)
        return integration_connection_create()

    @blueprint.post("/integrations/connections/<connection_id>/test")
    @require_auth("administrator", csrf=True)
    def integration_connection_test(connection_id):
        try:
            _registry, connections, _grants, _agents = services()
            actor = g.auth_session.username
            item = connections.get(connection_id, actor)
            if item is None:
                return _payload(error={"code": "not_found", "message": "The integration connection was not found."}, status=404)
            tester = callbacks.get("integration_connection_test") or callbacks.get("connection_tester")
            if tester is None:
                return _payload(error={"code": "connection_test_unavailable", "message": "No managed connection tester is configured."}, status=503)
            outcome = tester(item)
            if isinstance(outcome, Mapping):
                healthy = bool(outcome.get("healthy", outcome.get("ok", False)))
                detail = str(outcome.get("detail", outcome.get("message", "")))
            else:
                healthy, detail = bool(outcome), "Connection test completed." if outcome else "Connection test failed."
            updated = connections.test(connection_id, actor, healthy, detail)
            audit("integrations.connection.test", "success" if healthy else "failure", connection_id)
            return _payload(_safe_connection(updated), status=200 if healthy else 409)
        except (IntegrationConnectionError, ValueError, RuntimeError, TypeError) as error:
            return _payload(error={"code": _error_code(error), "message": str(error)}, status=503 if isinstance(error, RuntimeError) else 400)

    @blueprint.delete("/integrations/connections/<connection_id>")
    @require_auth("administrator", csrf=True)
    def integration_connection_revoke(connection_id):
        try:
            _registry, connections, grants, _agents = services()
            actor = g.auth_session.username
            before = connections.get(connection_id, actor)
            if before is None:
                return _payload(error={"code": "not_found", "message": "The integration connection was not found."}, status=404)
            impact = grants.dependents(connection_id, actor) if hasattr(grants, "dependents") else connections.dependents(connection_id, actor)
            revoked = connections.revoke(connection_id, actor, str((_body() if request.data else {}).get("reason", "")))
            audit("integrations.connection.revoke", "success", connection_id, {"active_dependents": impact.get("counts", {}).get("active", 0)})
            return _payload({"connection": _safe_connection(revoked), "impact": impact, "recoveryAction": "Create and test a replacement connection, then recreate blocked grants."})
        except (IntegrationConnectionError, AgentAppGrantError, ValueError, RuntimeError, TypeError) as error:
            return _payload(error={"code": _error_code(error), "message": str(error)}, status=503 if isinstance(error, RuntimeError) else 400)

    def _preview_grant(actor: str, body: Mapping[str, Any]) -> dict[str, Any]:
        now = _now()
        for preview_id, preview in list(previews.items()):
            if float(preview.get("expires_at", 0)) <= now:
                previews.pop(preview_id, None)
        if len(previews) >= _MAX_PREVIEWS:
            raise RuntimeError("The server-owned preview capacity is temporarily full; retry shortly.")
        registry, connections, grants, agents = services()
        app_instance_id, source_version, connection_id, operation_ids, manifest_version = _grant_request(body)
        app, manifest = _app_and_manifest(registry, app_instance_id)
        _assert_app_ready(app, manifest)
        if manifest_version is not None and manifest_version != manifest.manifest_version:
            raise ValueError("The requested manifest version is stale.")
        agent = agents.get(body.get("agentId") or body.get("agent_id"), actor)
        if agent is None:
            raise LookupError("The custom agent was not found.")
        if _agent_version(agent) != source_version:
            raise ValueError("The custom agent version is stale; refresh and review again.")
        if not agent.get("enabled", True):
            raise ValueError("The custom agent is disabled.")
        if not connection_id:
            raise ValueError("A tested connection is required to save app access.")
        connection = connections.get(connection_id, actor)
        if connection is None:
            raise LookupError("The integration connection was not found.")
        secret_version = _assert_connection_ready(connection, actor)
        operations: list[AppOperation] = []
        for operation_id in operation_ids:
            operation = manifest.operation(operation_id)
            if operation is None:
                raise ValueError("The requested operation is not present in the current manifest.")
            operations.append(operation)
        preview_id = "preview_" + uuid.uuid4().hex
        expires_at = now + _PREVIEW_TTL_SECONDS
        previews[preview_id] = {
            "actor": actor,
            "agent_id": agent["id"],
            "source_version": source_version,
            "app_instance_id": app_instance_id,
            "manifest_digest": manifest.manifest_digest,
            "manifest_version": manifest.manifest_version,
            "connection_id": connection_id,
            "secret_version": secret_version,
            "operation_ids": operation_ids,
            "expires_at": expires_at,
        }
        return {
            "previewId": preview_id,
            "expiresAt": _iso(expires_at),
            "summary": f"Grant {len(operation_ids)} pinned operation(s) to {agent.get('name', agent['id'])}.",
            "items": [
                {"label": operation.label, "kind": operation.mode, "risk": operation.risk,
                 "summary": "Exact operation access will be pinned to this app manifest and agent version."}
                for operation in operations
            ],
            "warnings": ["Write operations remain behind an exact task preview and approval."] if any(operation.mode == "write" for operation in operations) else [],
        }

    @blueprint.post("/assistant/custom-agents/<agent_id>/app-grants/preview")
    @require_auth("administrator", csrf=True)
    def integration_grant_preview(agent_id):
        try:
            body = _body()
            body = {**body, "agentId": agent_id}
            return _payload(_preview_grant(g.auth_session.username, body))
        except LookupError as error:
            return _payload(error={"code": "not_found", "message": str(error)}, status=404)
        except (AgentAppGrantError, CustomAgentError, RegistryError, ValueError, RuntimeError, TypeError) as error:
            return _payload(error={"code": _error_code(error), "message": str(error)}, status=503 if isinstance(error, RuntimeError) else 409 if "stale" in str(error).lower() else 400)

    @blueprint.get("/assistant/custom-agents/<agent_id>/app-grants")
    @require_auth("operator")
    def integration_grants(agent_id):
        try:
            registry, _connections, grants, agents = services()
            actor = g.auth_session.username
            agent = agents.get(agent_id, actor)
            if agent is None:
                return _payload(error={"code": "not_found", "message": "The custom agent was not found."}, status=404)
            items = []
            for item in grants.list(actor, agent_id=agent_id, limit=200):
                app = registry.get_app_instance(item["app_instance_id"])
                manifest = registry.get_manifest(app["manifest_digest"]) if app else None
                decision = grants.evaluate(
                    item["id"], actor,
                    agent_version=agent.get("version"),
                    app_state=app.get("state") if app else "removed",
                    manifest_digest=app.get("manifest_digest") if app else "",
                    available_operations=[operation.operation_id for operation in manifest.operations] if manifest else [],
                )
                items.append(_safe_grant(item, agent, decision.get("status"), "; ".join(decision.get("reasons", []))))
            return _payload({"items": items, "agentId": agent_id, "agentVersion": agent.get("version")})
        except (AgentAppGrantError, ValueError, RuntimeError) as error:
            return _payload(error={"code": _error_code(error), "message": str(error)}, status=503 if isinstance(error, RuntimeError) else 400)


    @blueprint.post("/assistant/custom-agents/<agent_id>/app-grants")
    @require_auth("administrator", csrf=True)
    def integration_grant_create(agent_id):
        try:
            body = _body()
            preview_id = _value(body, "previewId", "preview_id")
            if not preview_id or preview_id not in previews:
                raise ValueError("A current server-owned preview is required.")
            preview = previews[preview_id]
            actor = g.auth_session.username
            if preview["actor"] != actor or preview["agent_id"] != agent_id or preview["expires_at"] <= _now():
                raise ValueError("The grant preview is stale or belongs to another actor.")
            selection = {**body, "agentId": agent_id}
            app_instance_id, source_version, connection_id, operation_ids, manifest_version = _grant_request(selection)
            if (app_instance_id, source_version, connection_id, operation_ids, manifest_version) != (
                preview["app_instance_id"], preview["source_version"], preview["connection_id"], preview["operation_ids"], preview["manifest_version"]
            ):
                raise ValueError("The grant selection changed after preview.")
            registry, connections, grants, agents = services()
            app, manifest = _app_and_manifest(registry, app_instance_id)
            _assert_app_ready(app, manifest)
            if manifest.manifest_digest != preview["manifest_digest"]:
                raise ValueError("The app manifest changed after preview.")
            agent = agents.get(agent_id, actor)
            if agent is None or _agent_version(agent) != source_version:
                raise ValueError("The custom agent version changed after preview.")
            connection = connections.get(connection_id, actor)
            if connection is None or _assert_connection_ready(connection, actor) != preview["secret_version"]:
                raise ValueError("The connection changed after preview.")
            if not hasattr(grants, "clone_version"):
                raise RuntimeError("The grant store cannot pin an agent version safely.")
            updated = agents.update(agent_id, actor, {})
            target_version = _agent_version(updated)
            cloned = grants.clone_version(actor, agent_id, source_version, target_version)
            created = None
            try:
                created = grants.create(
                    actor, agent_id, target_version, app_instance_id,
                    manifest.manifest_digest, connection_id, operation_ids,
                    connection_secret_version=preview["secret_version"],
                    idempotency_key=f"p0-grant-{preview_id}",
                )
            except Exception:
                for clone in cloned.get("grants", []):
                    try:
                        grants.revoke(clone["id"], actor, "Compensated failed grant creation.")
                    except Exception:
                        pass
                raise
            previews.pop(preview_id, None)
            audit("integrations.grant.create", "success", created["id"], {"agent_version": target_version})
            return _payload({"grant": _safe_grant(created, updated), "agentVersion": target_version, "clonedGrants": cloned.get("cloned", 0)}, status=201)
        except LookupError as error:
            return _payload(error={"code": "not_found", "message": str(error)}, status=404)
        except (AgentAppGrantError, CustomAgentError, RegistryError, ValueError, RuntimeError, TypeError) as error:
            return _payload(error={"code": _error_code(error), "message": str(error)}, status=503 if isinstance(error, RuntimeError) else 409 if "stale" in str(error).lower() or "preview" in str(error).lower() else 400)

    @blueprint.delete("/assistant/custom-agents/<agent_id>/app-grants/<grant_id>")
    @require_auth("administrator", csrf=True)
    def integration_grant_revoke(agent_id, grant_id):
        try:
            _registry, _connections, grants, agents = services()
            actor = g.auth_session.username
            item = grants.get(grant_id, actor)
            if item is None or item.get("agent_id") != agent_id:
                return _payload(error={"code": "not_found", "message": "The capability grant was not found."}, status=404)
            revoked = grants.revoke(grant_id, actor, "Revoked by administrator.")
            agent = agents.get(agent_id, actor)
            audit("integrations.grant.revoke", "success", grant_id)
            return _payload({"grant": _safe_grant(revoked, agent, "revoked", "Revoked by administrator."), "recoveryAction": "Create a new preview if this agent needs access again."})
        except (AgentAppGrantError, ValueError, RuntimeError) as error:
            return _payload(error={"code": _error_code(error), "message": str(error)}, status=503 if isinstance(error, RuntimeError) else 400)
