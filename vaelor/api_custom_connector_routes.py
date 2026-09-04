"""Authenticated lifecycle routes for custom-agent connector grants."""

from __future__ import annotations

from flask import g, request

from .api_common import ApiContext, payload as _payload
from .custom_agents import CustomAgentError
from .custom_connector_runtime import ConnectorRuntimeError


def register_custom_connector_routes(context: ApiContext) -> None:
    blueprint = context.blueprint
    callbacks = context.callbacks
    security = context.security
    require_auth = context.require_auth

    def services():
        store = callbacks.get("custom_agents")
        runtime = callbacks.get("connector_runtime")
        if store is None or runtime is None:
            raise ConnectorRuntimeError("connector_runtime_unavailable: connector service is unavailable.")
        return store, runtime

    def profile(agent_id):
        store, runtime = services()
        item = store.get(agent_id, g.auth_session.username)
        if item is None:
            raise ConnectorRuntimeError("agent_not_found: custom agent was not found.")
        return store, runtime, item

    @blueprint.post("/assistant/custom-agents/<agent_id>/connectors/<connector_id>/test")
    @require_auth("operator", csrf=True)
    def connector_test(agent_id, connector_id):
        try:
            _store, runtime, item = profile(agent_id)
            body = request.get_json(silent=True) or {}
            if set(body) != {"operation_id", "arguments"}:
                raise ConnectorRuntimeError("invalid_request: operation_id and arguments are required.")
            result = runtime.test_connection(
                item, connector_id, str(body["operation_id"]), body["arguments"],
                actor=g.auth_session.username,
            )
        except (ConnectorRuntimeError, CustomAgentError, TypeError) as error:
            return _payload(error={"code": "connector_test_failed", "message": str(error)}, status=400)
        security.audit(
            g.auth_session.username, "assistant.connector.test", "success",
            target=agent_id, remote_addr=request.remote_addr or "",
            details={"connector_id": connector_id, "operation_id": result["operation_id"]},
        )
        return _payload(result)

    @blueprint.post(
        "/assistant/custom-agents/<agent_id>/connectors/<connector_id>/operations/<operation_id>/execute"
    )
    @require_auth("operator", csrf=True)
    def connector_execute(agent_id, connector_id, operation_id):
        try:
            _store, runtime, item = profile(agent_id)
            body = request.get_json(silent=True) or {}
            if set(body) != {"arguments"} or not isinstance(body["arguments"], dict):
                raise ConnectorRuntimeError("invalid_request: arguments object is required.")
            result = runtime.execute(
                item, connector_id, operation_id, body["arguments"],
                actor=g.auth_session.username,
            )
        except (ConnectorRuntimeError, CustomAgentError, TypeError) as error:
            return _payload(error={"code": "connector_execution_failed", "message": str(error)}, status=400)
        security.audit(
            g.auth_session.username, "assistant.connector.execute", "success",
            target=agent_id, remote_addr=request.remote_addr or "",
            details={"connector_id": connector_id, "operation_id": operation_id,
                     "state": result["state"]},
        )
        return _payload(result, status=202 if result["state"] == "approval_required" else 200)

    @blueprint.post(
        "/assistant/custom-agents/<agent_id>/connector-approvals/<approval_id>/approve"
    )
    @require_auth("operator", csrf=True)
    def connector_approve(agent_id, approval_id):
        try:
            _store, runtime, item = profile(agent_id)
            result = runtime.approve(
                approval_id, item, actor=g.auth_session.username,
            )
        except (ConnectorRuntimeError, CustomAgentError) as error:
            return _payload(error={"code": "connector_approval_failed", "message": str(error)}, status=409)
        security.audit(
            g.auth_session.username, "assistant.connector.approve", "success",
            target=agent_id, remote_addr=request.remote_addr or "",
            details={"approval_id": approval_id, "state": result["state"]},
        )
        return _payload(result)

    @blueprint.get("/assistant/custom-agents/<agent_id>/connector-audit")
    @require_auth("operator")
    def connector_audit(agent_id):
        try:
            _store, runtime, _item = profile(agent_id)
            return _payload(runtime.audit(
                agent_id, g.auth_session.username, request.args.get("limit", 100)
            ))
        except (ConnectorRuntimeError, ValueError) as error:
            return _payload(error={"code": "connector_audit_failed", "message": str(error)}, status=400)

    @blueprint.delete("/assistant/custom-agents/<agent_id>/connectors/<connector_id>")
    @require_auth("operator", csrf=True)
    def connector_revoke(agent_id, connector_id):
        try:
            store, _runtime, item = profile(agent_id)
            remaining = [
                connector for connector in item.get("connectors", [])
                if connector["id"] != connector_id
            ]
            if len(remaining) == len(item.get("connectors", [])):
                raise ConnectorRuntimeError("connector_not_found: connector grant was not found.")
            updated = store.update(
                agent_id, g.auth_session.username, {"connectors": remaining}
            )
        except (ConnectorRuntimeError, CustomAgentError) as error:
            return _payload(error={"code": "connector_revoke_failed", "message": str(error)}, status=400)
        security.audit(
            g.auth_session.username, "assistant.connector.revoke", "success",
            target=agent_id, remote_addr=request.remote_addr or "",
            details={"connector_id": connector_id, "version": updated["version"]},
        )
        return _payload({"revoked": True, "connector_id": connector_id,
                         "agent_version": updated["version"]})
