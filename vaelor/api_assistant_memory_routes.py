"""Assistant long-term memory API routes."""

from flask import g, request

from .api_common import ApiContext, payload as _payload
from .assistant_memory import AssistantMemoryError


def register_assistant_memory_routes(context: ApiContext) -> None:
    """Register administrator-owned long-term memory endpoints."""
    blueprint = context.blueprint
    callbacks = context.callbacks
    security = context.security
    require_auth = context.require_auth

    @blueprint.get("/assistant/memories")
    @require_auth("administrator")
    def assistant_memories():
        assistant_store = callbacks.get("assistant_memory")
        if assistant_store is None:
            return _payload(
                error={"code": "assistant_memory_unavailable", "message": "Assistant memory is unavailable."},
                status=503,
            )
        try:
            memories = assistant_store.list_memories(
                query=request.args.get("query", ""),
                memory_type=request.args.get("type", ""),
                limit=request.args.get("limit", 100),
            )
        except (AssistantMemoryError, TypeError, ValueError) as error:
            return _payload(
                error={"code": "invalid_memory_query", "message": str(error)}, status=400
            )
        return _payload(memories)

    @blueprint.post("/assistant/memories")
    @require_auth("administrator", csrf=True)
    def assistant_memory_create():
        assistant_store = callbacks.get("assistant_memory")
        if assistant_store is None:
            return _payload(
                error={"code": "assistant_memory_unavailable", "message": "Assistant memory is unavailable."},
                status=503,
            )
        body = request.get_json(silent=True) or {}
        try:
            memory = assistant_store.add_memory(
                content=body.get("content", ""),
                memory_type=body.get("memory_type", ""),
                scope=body.get("scope", ""),
                source=body.get("source", ""),
                confidence=body.get("confidence", 1),
                pinned=body.get("pinned", False),
                actor=g.auth_session.username,
                expires_at=body.get("expires_at"),
            )
        except (AssistantMemoryError, TypeError, ValueError) as error:
            security.audit(
                g.auth_session.username, "assistant.memory.create", "failure",
                remote_addr=request.remote_addr or "",
            )
            return _payload(
                error={"code": "invalid_memory", "message": str(error)}, status=400
            )
        security.audit(
            g.auth_session.username, "assistant.memory.create", "success",
            target=memory["id"], remote_addr=request.remote_addr or "",
            details={"type": memory["memory_type"], "scope": memory["scope"]},
        )
        return _payload(memory, status=201)

    @blueprint.patch("/assistant/memories/<memory_id>")
    @require_auth("administrator", csrf=True)
    def assistant_memory_update(memory_id):
        assistant_store = callbacks.get("assistant_memory")
        if assistant_store is None:
            return _payload(
                error={"code": "assistant_memory_unavailable", "message": "Assistant memory is unavailable."},
                status=503,
            )
        try:
            memory = assistant_store.update_memory(
                memory_id, request.get_json(silent=True) or {}, g.auth_session.username
            )
        except (AssistantMemoryError, TypeError, ValueError) as error:
            return _payload(
                error={"code": "invalid_memory", "message": str(error)}, status=400
            )
        security.audit(
            g.auth_session.username, "assistant.memory.update", "success",
            target=memory_id, remote_addr=request.remote_addr or "",
        )
        return _payload(memory)

    @blueprint.delete("/assistant/memories/<memory_id>")
    @require_auth("administrator", csrf=True)
    def assistant_memory_delete(memory_id):
        assistant_store = callbacks.get("assistant_memory")
        if assistant_store is None:
            return _payload(
                error={"code": "assistant_memory_unavailable", "message": "Assistant memory is unavailable."},
                status=503,
            )
        deleted = assistant_store.delete_memory(memory_id, g.auth_session.username)
        if not deleted:
            return _payload(
                error={"code": "memory_not_found", "message": "Memory was not found."},
                status=404,
            )
        security.audit(
            g.auth_session.username, "assistant.memory.delete", "success",
            target=memory_id, remote_addr=request.remote_addr or "",
        )
        return _payload({"deleted": True})
