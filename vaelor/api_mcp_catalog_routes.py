"""HTTP routes for the agent-facing MCP catalog and per-agent tool grants (F1).

These routes administer the *outbound* MCP surface a deployed agent runtime will
use. They are deliberately separate from ``api_mcp_routes`` — the inbound
``POST /api/v2/mcp`` server and the operator-confirmed external-run route — and
share nothing with it: no record touched here is ever passed to
``VaelorMcpServer(external=...)`` or minted into an ``EXTERNAL_SCOPE`` session.

Reads are ``operator``; every mutation is ``administrator`` + CSRF + audit,
matching the conventions in ``api_mcp_routes`` and ``api_integration_routes``.
"""

from __future__ import annotations

from flask import g, request

from .api_common import ApiContext, payload
from .mcp_catalog import McpCatalogError, refresh_server_health, start_stale_refresh


def register_mcp_catalog_routes(context: ApiContext) -> None:
    blueprint = context.blueprint
    require_auth = context.require_auth
    security = context.security
    callbacks = context.callbacks

    def catalog():
        return callbacks.get("mcp_catalog")

    def grants():
        return callbacks.get("agent_mcp_grants")

    def _unavailable():
        return payload(
            error={
                "code": "mcp_catalog_unavailable",
                "message": "The MCP catalog is unavailable.",
            },
            status=503,
        )

    def _audit(action: str, outcome: str, target: str = "", **details):
        security.audit(
            g.auth_session.username, action, outcome,
            target=str(target)[:64], remote_addr=request.remote_addr or "",
            details=details or None,
        )

    def _server_view(item):
        """A server as an administrator sees it. No secret ever appears; the
        broker reference is an id, and the endpoint is the operator-typed URL."""
        return {
            "id": item["id"],
            "name": item["name"],
            "kind": item["kind"],
            "endpoint": item["endpoint"],
            "credential_id": item["credential_id"],
            "enabled": item["enabled"],
            "approved_tools": list(item["approved_tools"]),
            "health": item["health"],
            "created_at": item["created_at"],
            "updated_at": item["updated_at"],
        }

    # -- catalog servers --------------------------------------------------
    @blueprint.get("/mcp/catalog/servers")
    @require_auth("operator")
    def mcp_catalog_servers():
        store = catalog()
        if store is None:
            return payload({"servers": [], "configured": False})
        servers = store.list()
        # ACC-077: a stale health reading is re-probed in the background, so
        # the list stops reading healthy for a server that went down.
        refreshing = set(start_stale_refresh(
            store, servers, spawn=callbacks.get("mcp_catalog_refresh_spawn"),
        ))
        return payload({
            "configured": True,
            "servers": [
                {**_server_view(item), "health_refreshing": item["id"] in refreshing}
                for item in servers
            ],
            "note": (
                "These servers feed a deployed agent's own outbound MCP client "
                "only. They are never reachable from the inbound POST /api/v2/mcp "
                "server and hold no appliance scope."
            ),
        })

    @blueprint.post("/mcp/catalog/servers")
    @require_auth("administrator", csrf=True)
    def mcp_catalog_register():
        store = catalog()
        if store is None:
            return _unavailable()
        body = request.get_json(silent=True) or {}
        try:
            created = store.register(
                name=body.get("name", ""),
                kind=body.get("kind", ""),
                endpoint=body.get("endpoint", ""),
                credential_id=body.get("credential_id", ""),
                enabled=bool(body.get("enabled", False)),
                approved_tools=body.get("approved_tools", []),
            )
        except McpCatalogError as error:
            _audit("mcp.catalog.register", "rejected", str(body.get("name", ""))[:64])
            return payload(
                error={"code": "mcp_catalog_rejected", "message": str(error)},
                status=400,
            )
        _audit("mcp.catalog.register", "success", created["id"], kind=created["kind"])
        return payload(_server_view(created), status=201)

    @blueprint.post("/mcp/catalog/servers/<server_id>")
    @require_auth("administrator", csrf=True)
    def mcp_catalog_update(server_id):
        store = catalog()
        if store is None:
            return _unavailable()
        body = request.get_json(silent=True) or {}
        try:
            updated = store.update(str(server_id), body)
        except McpCatalogError as error:
            _audit("mcp.catalog.update", "rejected", str(server_id))
            return payload(
                error={"code": "mcp_catalog_rejected", "message": str(error)},
                status=400,
            )
        _audit("mcp.catalog.update", "success", updated["id"])
        return payload(_server_view(updated))

    @blueprint.delete("/mcp/catalog/servers/<server_id>")
    @require_auth("administrator", csrf=True)
    def mcp_catalog_remove(server_id):
        store = catalog()
        if store is None:
            return _unavailable()
        try:
            result = store.remove(str(server_id))
        except McpCatalogError as error:
            _audit("mcp.catalog.remove", "rejected", str(server_id))
            return payload(
                error={"code": "mcp_catalog_rejected", "message": str(error)},
                status=404,
            )
        _audit("mcp.catalog.remove", "success", str(server_id))
        return payload(result)

    @blueprint.post("/mcp/catalog/servers/<server_id>/health")
    @require_auth("administrator", csrf=True)
    def mcp_catalog_health(server_id):
        store = catalog()
        if store is None:
            return _unavailable()
        try:
            updated = refresh_server_health(store, str(server_id))
        except McpCatalogError as error:
            _audit("mcp.catalog.health", "rejected", str(server_id))
            return payload(
                error={"code": "mcp_catalog_rejected", "message": str(error)},
                status=404,
            )
        _audit(
            "mcp.catalog.health", "success", updated["id"],
            status=updated["health"].get("status", ""),
        )
        return payload({"id": updated["id"], "health": updated["health"]})

    # -- per-agent grants -------------------------------------------------
    def _grant_view(item):
        return {
            "id": item["id"],
            "created_by": item["created_by"],
            "agent_id": item["agent_id"],
            "agent_version": item["agent_version"],
            "server_id": item["server_id"],
            "allowed_tools": list(item["allowed_tools"]),
            "revoked": item["revoked"],
            "revoked_at": item.get("revoked_at"),
            "created_at": item["created_at"],
            "updated_at": item["updated_at"],
        }

    @blueprint.get("/mcp/catalog/grants")
    @require_auth("operator")
    def mcp_catalog_grants():
        store = grants()
        if store is None:
            return payload({"grants": [], "configured": False})
        agent_id = request.args.get("agent_id") or None
        items = store.list(agent_id=agent_id, limit=200)
        return payload({"configured": True, "grants": [_grant_view(item) for item in items]})

    @blueprint.post("/mcp/catalog/grants")
    @require_auth("administrator", csrf=True)
    def mcp_catalog_attach():
        store = grants()
        if store is None:
            return _unavailable()
        body = request.get_json(silent=True) or {}
        try:
            created = store.create(
                g.auth_session.username,
                body.get("agent_id", ""),
                body.get("agent_version", 0),
                body.get("server_id", ""),
                body.get("allowed_tools", []),
                idempotency_key=body.get("idempotency_key"),
            )
        except McpCatalogError as error:
            _audit("mcp.grant.attach", "rejected", str(body.get("server_id", ""))[:64])
            return payload(
                error={"code": "mcp_grant_rejected", "message": str(error)},
                status=400,
            )
        _audit(
            "mcp.grant.attach", "success", created["id"],
            agent_id=created["agent_id"], server_id=created["server_id"],
        )
        return payload(_grant_view(created), status=201)

    @blueprint.delete("/mcp/catalog/grants/<grant_id>")
    @require_auth("administrator", csrf=True)
    def mcp_catalog_revoke(grant_id):
        store = grants()
        if store is None:
            return _unavailable()
        try:
            revoked = store.revoke(str(grant_id), "MCP grant revoked by an administrator.")
        except McpCatalogError as error:
            _audit("mcp.grant.revoke", "rejected", str(grant_id))
            return payload(
                error={"code": "mcp_grant_rejected", "message": str(error)},
                status=404,
            )
        _audit("mcp.grant.revoke", "success", str(grant_id))
        return payload(_grant_view(revoked))
