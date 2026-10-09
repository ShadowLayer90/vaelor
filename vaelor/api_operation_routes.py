"""Authenticated routes for the unified ``vaelor.operation.v1`` contract."""

from __future__ import annotations

from typing import Any

from flask import g, request

from .agent_tasks import AgentTaskError
from .api_common import ApiContext, payload
from .operation_projection import (
    OPERATION_SCHEMA,
    OperationActionError,
    OperationNotFound,
    OperationProjection,
)


MAX_LIMIT = 200


def register_operation_routes(context: ApiContext) -> None:
    """Register list/detail/action routes over the two durable source ledgers."""
    blueprint = context.blueprint
    callbacks = context.callbacks
    require_auth = context.require_auth
    security = context.security

    def projection() -> OperationProjection:
        return OperationProjection(
            callbacks.get("job_store"), callbacks.get("agent_tasks"),
            automations=callbacks.get("automations"),
        )

    def actor_scope() -> str | None:
        return None if g.auth_session.role == "administrator" else g.auth_session.username

    def audit(action: str, result: str, target: str, **details: Any) -> None:
        security.audit(
            g.auth_session.username,
            action,
            result,
            target=str(target)[:256],
            remote_addr=request.remote_addr or "",
            details=details,
        )

    def operation_error(error: BaseException, operation_id: str = ""):
        if isinstance(error, OperationNotFound) or isinstance(error, KeyError):
            return payload(
                error={
                    "code": "operation_not_found",
                    "message": "The operation was not found or is not visible to this account.",
                },
                status=404,
            )
        if isinstance(error, (OperationActionError, AgentTaskError, ValueError)):
            return payload(
                error={
                    "code": "operation_action_invalid",
                    "message": str(error)[:500],
                },
                status=409,
            )
        return payload(
            error={
                "code": "operation_unavailable",
                "message": "The operation service could not complete the request safely.",
            },
            status=503,
        )

    @blueprint.get("/operations")
    @require_auth("viewer")
    def operation_list():
        ledger = str(request.args.get("ledger", "")).strip() or None
        # #205 coverage: a non-integer ?limit must not leak Python's raw
        # "invalid literal for int() with base 10: 'abc'" to the caller.
        raw_limit = str(request.args.get("limit", "50")).strip()
        if not raw_limit.lstrip("-").isdigit():
            return payload(
                error={
                    "code": "operation_query_invalid",
                    "message": "The 'limit' parameter must be a whole number.",
                },
                status=400,
            )
        # ACC-125: `?bucket=` narrows the window to one Activity tile; absent
        # or `all` means every operation. `summary` always partitions all of
        # them, so the tiles never count only the window below them.
        bucket = str(request.args.get("bucket", "")).strip() or None
        if bucket == "all":
            bucket = None
        try:
            limit = max(1, min(int(raw_limit), MAX_LIMIT))
            page = projection().page(
                actor_scope(), ledger=ledger, limit=limit, bucket=bucket)
        except (TypeError, ValueError) as error:
            return payload(
                error={"code": "operation_query_invalid", "message": str(error)[:500]},
                status=400,
            )
        operations = page["operations"]
        return payload({
            "schema": OPERATION_SCHEMA,
            "operations": operations,
            "items": operations,
            "count": len(operations),
            "summary": page["summary"],
            "matched": page["matched"],
            "bucket": page["bucket"] or "all",
        })

    @blueprint.get("/operations/<path:operation_id>")
    @require_auth("viewer")
    def operation_detail(operation_id: str):
        try:
            result = projection().get(operation_id, actor_scope(), detail=True)
        except Exception as error:
            return operation_error(error, operation_id)
        return payload(result)

    @blueprint.get("/operations/<path:operation_id>/audit")
    @require_auth("viewer")
    def operation_audit(operation_id: str):
        try:
            operation = projection().get(operation_id, actor_scope(), detail=False)
        except Exception as error:
            return operation_error(error, operation_id)
        visible_targets = {
            operation["operation_id"], operation["operation_key"], operation["source_id"],
        }
        # W5-D5: a route that queues a job audits the change under its own
        # target and names the job in details.job_id; that id is this
        # operation's source id. Matched by id, never by time window.
        events = [
            item for item in security.list_audit(200)
            if item.get("target") in visible_targets
            or item.get("details", {}).get("operation_id") == operation["operation_id"]
            or (
                operation.get("ledger") == "jobs"
                and item.get("details", {}).get("job_id") == operation["source_id"]
            )
        ]
        if g.auth_session.role != "administrator":
            events = [
                item for item in events
                if item.get("actor") == g.auth_session.username
            ]
        return payload({
            "schema": OPERATION_SCHEMA,
            "operation_id": operation["operation_id"],
            "events": events,
        })

    @blueprint.post("/operations/<path:operation_id>/cancel")
    @require_auth("operator", csrf=True)
    def operation_cancel(operation_id: str):
        try:
            result = projection().cancel(
                operation_id,
                g.auth_session.username,
                allow_all=g.auth_session.role == "administrator",
            )
        except Exception as error:
            audit("operation.cancel", "failure", operation_id, error_code=type(error).__name__)
            return operation_error(error, operation_id)
        audit("operation.cancel", "success", result["operation_id"], ledger=result["ledger"])
        return payload({"operation": result, "action": "cancel"})

    @blueprint.post("/operations/<path:operation_id>/dismiss")
    @require_auth("operator", csrf=True)
    def operation_dismiss(operation_id: str):
        """The owner's "I've dealt with this" for an attention item (VD-139)."""
        try:
            result = projection().dismiss(
                operation_id,
                g.auth_session.username,
                allow_all=g.auth_session.role == "administrator",
            )
        except Exception as error:
            audit("operation.dismiss", "failure", operation_id, error_code=type(error).__name__)
            return operation_error(error, operation_id)
        audit("operation.dismiss", "success", result["operation_id"], ledger=result["ledger"])
        return payload({"operation": result, "action": "dismiss"})

    @blueprint.post("/operations/<path:operation_id>/retry")
    @require_auth("operator", csrf=True)
    def operation_retry(operation_id: str):
        try:
            result = projection().retry(
                operation_id,
                g.auth_session.username,
                allow_all=g.auth_session.role == "administrator",
            )
        except Exception as error:
            audit("operation.retry", "failure", operation_id, error_code=type(error).__name__)
            return operation_error(error, operation_id)
        audit(
            "operation.retry",
            "success",
            result["operation_id"],
            source_operation_id=operation_id,
            ledger=result["ledger"],
        )
        return payload({"operation": result, "action": "retry"})
