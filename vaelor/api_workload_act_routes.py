"""Administer the default-inert ``workloads:act`` grant (VD-100 #96).

The acting seam - the operator's Assistant proposing a restart, an installed
agent proposing one - is inert until ``workloads:act`` is granted. These routes
are the only thing that grants it, and they are administrator-only: no role holds
the scope by default, and it is never inferred from a role, only written here.

There is no execution anywhere in this module. Granting the scope lets a proposal
*reach* the operator's existing ``POST /api/v2/jobs`` approval; it mints no job
and runs nothing. Widening an agent past the server-owned envelope
(`INSTALLED_AGENT_OPERATIONS`) is refused by the store, so this surface cannot
introduce a job type the executor was never meant to run (#34).

It lives in its own module because the assistant route files are at their size
ceiling; its blueprint is registered from `api_v2.py`.
"""

from __future__ import annotations

from flask import g, request

from .api_common import ApiContext, payload as _payload
from .workload_act_grants import WorkloadActGrantError


def register_workload_act_routes(context: ApiContext) -> None:
    blueprint = context.blueprint
    callbacks = context.callbacks
    security = context.security
    require_auth = context.require_auth

    def _store():
        return callbacks.get("workload_act_grants")

    def _unavailable():
        return _payload(
            error={
                "code": "workload_act_unavailable",
                "message": "The acting-grant service is unavailable.",
            },
            status=503,
        )

    @blueprint.post("/assistant/workload-act/operators/<username>")
    @require_auth("administrator", csrf=True)
    def grant_operator(username):
        store = _store()
        if store is None:
            return _unavailable()
        try:
            granted = store.grant_operator_scope(username, g.auth_session.username)
        except WorkloadActGrantError as error:
            return _payload(error={"code": "invalid_grant", "message": str(error)}, status=400)
        security.audit(
            g.auth_session.username, "workload_act.operator.grant", "success",
            target=str(username), remote_addr=request.remote_addr or "",
        )
        return _payload(granted, status=201)

    @blueprint.delete("/assistant/workload-act/operators/<username>")
    @require_auth("administrator", csrf=True)
    def revoke_operator(username):
        store = _store()
        if store is None:
            return _unavailable()
        removed = store.revoke_operator_scope(username)
        security.audit(
            g.auth_session.username, "workload_act.operator.revoke",
            "success" if removed else "noop",
            target=str(username), remote_addr=request.remote_addr or "",
        )
        return _payload({"revoked": bool(removed)})

    @blueprint.post(
        "/assistant/workload-act/agents/<profile>/versions/<int:version>/operations/<operation>"
    )
    @require_auth("administrator", csrf=True)
    def grant_agent_operation(profile, version, operation):
        store = _store()
        if store is None:
            return _unavailable()
        try:
            granted = store.grant_agent_operation(
                profile, version, operation, g.auth_session.username
            )
        except WorkloadActGrantError as error:
            return _payload(error={"code": "invalid_grant", "message": str(error)}, status=400)
        security.audit(
            g.auth_session.username, "workload_act.agent.grant", "success",
            target="{}:{}:{}".format(profile, version, operation),
            remote_addr=request.remote_addr or "",
        )
        return _payload(granted, status=201)

    @blueprint.delete(
        "/assistant/workload-act/agents/<profile>/versions/<int:version>/operations/<operation>"
    )
    @require_auth("administrator", csrf=True)
    def revoke_agent_operation(profile, version, operation):
        store = _store()
        if store is None:
            return _unavailable()
        removed = store.revoke_agent_operation(profile, version, operation)
        security.audit(
            g.auth_session.username, "workload_act.agent.revoke",
            "success" if removed else "noop",
            target="{}:{}:{}".format(profile, version, operation),
            remote_addr=request.remote_addr or "",
        )
        return _payload({"revoked": bool(removed)})
