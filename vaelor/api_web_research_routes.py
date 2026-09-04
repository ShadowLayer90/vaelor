"""Truthful setup and lifecycle routes for guarded public web research."""

from __future__ import annotations

from flask import g, request

from .api_common import ApiContext, payload
from .web_research import WebResearchError, WebResearchManager


def register_web_research_routes(context: ApiContext) -> None:
    blueprint = context.blueprint
    require_auth = context.require_auth
    security = context.security

    def manager() -> WebResearchManager:
        configured = context.callbacks.get("web_research_manager")
        return configured if configured is not None else WebResearchManager()

    @blueprint.get("/applications/research-service")
    @require_auth("viewer")
    def web_research_status():
        return payload(manager().status())

    @blueprint.post("/applications/research-service/plan")
    @require_auth("administrator", csrf=True)
    def web_research_plan():
        body = request.get_json(silent=True)
        action = body.get("action") if isinstance(body, dict) else ""
        try:
            result = manager().plan(str(action))
        except WebResearchError as error:
            return payload(error={"code": "invalid_research_action", "message": str(error)}, status=400)
        security.audit(
            g.auth_session.username, "web-research.plan", "success",
            target=str(action), remote_addr=request.remote_addr or "",
        )
        return payload(result)

    @blueprint.post("/applications/research-service/actions")
    @require_auth("administrator", csrf=True)
    def web_research_action():
        body = request.get_json(silent=True)
        if not isinstance(body, dict) or set(body) - {"action", "confirmation", "purge"}:
            return payload(
                error={"code": "invalid_research_action", "message": "Send a reviewed research-service action."},
                status=400,
            )
        action = str(body.get("action", ""))
        try:
            plan = manager().plan(action)
        except WebResearchError as error:
            return payload(error={"code": "invalid_research_action", "message": str(error)}, status=400)
        if body.get("confirmation") != plan["confirmation"]:
            return payload(
                error={
                    "code": "research_action_confirmation_required",
                    "message": "Review the plan and type its exact confirmation.",
                },
                status=400,
            )
        job_store = context.callbacks.get("job_store")
        if job_store is None:
            return payload(
                error={"code": "research_job_queue_unavailable", "message": "The durable setup queue is unavailable."},
                status=503,
            )
        job = job_store.create(
            "host.web-research.manage", g.auth_session.username,
            {
                "action": action,
                "confirmation": plan["confirmation"],
                "purge": bool(body.get("purge", False)),
            },
        )
        security.audit(
            g.auth_session.username, "web-research.queue", "queued",
            target=action, remote_addr=request.remote_addr or "",
            details={"job_id": job["id"]},
        )
        return payload({"plan": plan, "job": job}, status=202)
