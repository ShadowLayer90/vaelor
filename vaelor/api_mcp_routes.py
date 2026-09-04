"""HTTP transport for Vaelor's MCP server, and for enrolled external servers.

Two directions meet here and they are deliberately not symmetric.

*Inbound* - ``POST /api/v2/mcp`` - is Vaelor answering an MCP client. It reuses
the same session, the same roles and the same audit log as every other v2
route, and the session it builds can only ever hold appliance scopes the
caller's role already has.

*Outbound* - the enrolled external servers - is Vaelor calling somebody else.
That never happens on an inbound MCP session: ``McpSession.for_role`` can only
narrow, and no role carries ``external:mcp``, so an MCP client cannot reach
through Vaelor into another server. Running an external tool is an
administrator action with an explicit confirmation, and it is audited under its
own event.
"""

from __future__ import annotations

from flask import g, jsonify, request

from .api_common import ApiContext, payload
from .mcp_client import EXTERNAL_SCOPE, McpClientError
from .mcp_server import (
    PROTOCOL_VERSION,
    McpSession,
    VaelorMcpServer,
    handle_batch,
)


def register_mcp_routes(context: ApiContext) -> None:
    blueprint = context.blueprint
    require_auth = context.require_auth
    security = context.security
    callbacks = context.callbacks

    def server():
        registry = callbacks.get("assistant_tools")
        if registry is None:
            return None
        return VaelorMcpServer(registry, external=callbacks.get("external_mcp_tools"))

    def session():
        return McpSession.for_role(
            g.auth_session.username, g.auth_session.role
        )

    @blueprint.get("/mcp")
    @require_auth("viewer")
    def mcp_descriptor():
        """What this endpoint is, and what this caller would be granted.

        A client that can see its own authorisation before it connects does not
        have to discover it by being refused.
        """
        current = session()
        return payload({
            "protocol_version": PROTOCOL_VERSION,
            "transport": "json-rpc over POST /api/v2/mcp",
            "authentication": "Vaelor session cookie and CSRF token",
            "granted_scopes": sorted(current.scopes),
            "administrator": current.administrator,
            "read_only": True,
            "note": (
                "Guarded public research is never available over MCP: it is "
                "authorised per agent with its own domain allowlist."
            ),
        })

    @blueprint.post("/mcp")
    @require_auth("operator", csrf=True)
    def mcp_endpoint():
        instance = server()
        if instance is None:
            return payload(
                error={
                    "code": "assistant_tools_unavailable",
                    "message": "Assistant tools are unavailable.",
                },
                status=503,
            )
        body = request.get_json(silent=True)
        if body is None:
            return payload(
                error={
                    "code": "invalid_mcp_request",
                    "message": "Send a JSON-RPC 2.0 request body.",
                },
                status=400,
            )
        current = session()
        response = handle_batch(instance, body, current)
        method = body.get("method", "") if isinstance(body, dict) else "batch"
        security.audit(
            g.auth_session.username, "mcp.request", "success",
            target=str(method)[:64], remote_addr=request.remote_addr or "",
            details={"scopes": sorted(current.scopes)},
        )
        # A batch of notifications has no reply at all, which is what JSON-RPC
        # requires and what an MCP client expects.
        return ("", 202) if response is None else (jsonify(response), 200)

    @blueprint.get("/mcp/servers")
    @require_auth("operator")
    def mcp_servers():
        external = callbacks.get("external_mcp_tools")
        if external is None:
            return payload({"servers": [], "configured": False})
        return payload({
            "configured": True,
            "servers": [
                {
                    "name": item["name"],
                    "label": item["label"],
                    "endpoint": item["endpoint"],
                    "enabled": item["enabled"],
                    "approved_tools": list(item["approved_tools"]),
                    "origin": "external",
                    "scope": EXTERNAL_SCOPE,
                }
                for item in external.servers
            ],
            "note": (
                "External tools are not appliance tools. They hold no "
                "appliance scope, are never reachable from an inbound MCP "
                "session, and every call needs an administrator confirmation."
            ),
        })

    @blueprint.post("/mcp/servers/<server_name>/tools/<tool_name>/run")
    @require_auth("administrator", csrf=True)
    def mcp_external_run(server_name, tool_name):
        external = callbacks.get("external_mcp_tools")
        if external is None:
            return payload(
                error={
                    "code": "external_mcp_unavailable",
                    "message": "No external MCP servers are configured.",
                },
                status=503,
            )
        body = request.get_json(silent=True) or {}
        qualified = external.qualified(str(server_name), str(tool_name))
        # The confirmation is the approval, typed by a human, for this exact
        # tool. Without it the call is refused - which is why the approval
        # callback is built here rather than held open on the client.
        confirmed = body.get("confirmation") == qualified
        call_session = McpSession(
            actor=g.auth_session.username,
            administrator=True,
            scopes=frozenset({EXTERNAL_SCOPE}),
        )
        try:
            result = external.call(
                qualified,
                body.get("arguments", {}),
                call_session,
            ) if confirmed else _refuse(qualified)
        except McpClientError as error:
            security.audit(
                g.auth_session.username, "mcp.external.run", "rejected",
                target=qualified[:64], remote_addr=request.remote_addr or "",
            )
            return payload(
                error={"code": "external_mcp_rejected", "message": str(error)},
                status=400,
            )
        security.audit(
            g.auth_session.username, "mcp.external.run", "success",
            target=qualified[:64], remote_addr=request.remote_addr or "",
            details={"server": str(server_name)[:32]},
        )
        return payload(result)


def _refuse(qualified: str):
    raise McpClientError(
        "approval_required: confirm this call by sending confirmation="
        "{}.".format(qualified)
    )
