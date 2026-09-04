"""Per-operation approval for the operations an installed agent proposes.

This is the acting counterpart to `assistant_task_write_approve`
(`api_assistant_routes.py`). An installed agent that finishes a task may append
``proposed_operations`` entries to its result - the same shape as
``proposed_writes`` - and each one is a *request*, not an action. An operator
approves them one at a time here.

The route owns no execution. It resolves the proposal against the agent's
server-owned pinned grant snapshot and against live inventory
(`vaelor.assistant_action_proposals`), then routes to the same
``job_store.create`` + ``security.audit`` path the human jobs endpoint uses.
There is one executor, and this endpoint is a producer of jobs for it, never a
second way to run one.

It lives in its own module because `api_workload_routes.py` is at its size
ceiling; its blueprint is registered from `api_v2.py`.
"""

from __future__ import annotations

from typing import Any, Dict

from flask import g, request

from .agent_tasks import AgentTaskError
from .api_common import ApiContext, payload as _payload
from .assistant_action_proposals import (
    ActingProposalError,
    managed_projects,
    resolve_agent_operation,
)


def _live_managed_projects(callbacks: Dict[str, Any]) -> frozenset[str]:
    """The managed apps the server can see right now, for resource scoping.

    Read from live inventory rather than from the agent's result: #34 says the
    resource an operation acts on is bound by server truth, not by a project
    name the model wrote.
    """
    inventory = callbacks.get("workload_inventory")
    snapshot: Any = None
    if inventory is not None:
        if hasattr(inventory, "snapshot"):
            snapshot = inventory.snapshot()
        elif hasattr(inventory, "list_all"):
            snapshot = inventory.list_all()
        elif callable(inventory):
            snapshot = inventory()
    return managed_projects(snapshot)


def register_agent_operation_routes(context: ApiContext) -> None:
    blueprint = context.blueprint
    callbacks = context.callbacks
    security = context.security
    require_auth = context.require_auth

    @blueprint.post(
        "/assistant/tasks/<task_id>/operations/<int:operation_index>/approve"
    )
    @require_auth("operator", csrf=True)
    def assistant_task_operation_approve(task_id, operation_index):
        task_store = callbacks.get("agent_tasks")
        job_store = callbacks.get("job_store")
        if job_store is None:
            return _payload(
                error={
                    "code": "job_service_unavailable",
                    "message": "The privileged job service is unavailable, so operations cannot be approved.",
                },
                status=503,
            )
        task = task_store.get(task_id, g.auth_session.username) if task_store else None
        if task is None or task["state"] != "completed":
            return _payload(
                error={
                    "code": "agent_operation_not_found",
                    "message": "A completed agent operation proposal was not found.",
                },
                status=404,
            )
        operations = task.get("result", {}).get("proposed_operations", [])
        if not 0 <= operation_index < len(operations):
            return _payload(
                error={
                    "code": "agent_operation_not_found",
                    "message": "Agent operation proposal was not found.",
                },
                status=404,
            )
        entry = operations[operation_index]
        if not isinstance(entry, dict) or entry.get("executed"):
            return _payload(
                error={
                    "code": "agent_operation_rejected",
                    "message": "This operation is unsupported or already completed.",
                },
                status=409,
            )
        try:
            proposal = resolve_agent_operation(
                entry, task, _live_managed_projects(callbacks)
            )
        except ActingProposalError as error:
            security.audit(
                g.auth_session.username,
                "assistant.agent_operation.approve",
                "rejected",
                target=task_id,
                remote_addr=request.remote_addr or "",
                details={"operation": str(entry.get("operation", ""))[:64]},
            )
            return _payload(
                error={"code": "agent_operation_rejected", "message": str(error)},
                status=409,
            )
        try:
            job = job_store.create(
                proposal["type"], g.auth_session.username, proposal["payload"]
            )
        except ValueError as error:
            return _payload(
                error={"code": "invalid_job_request", "message": str(error)},
                status=400,
            )
        updated_result = dict(task["result"])
        updated_operations = [dict(item) for item in operations]
        updated_operations[operation_index].update(
            {
                "executed": True,
                "job_id": job["id"],
                "job_type": proposal["type"],
                "approved_by": g.auth_session.username,
            }
        )
        updated_result["proposed_operations"] = updated_operations
        recording_error = ""
        try:
            updated = task_store.update_completed_result(
                task_id, task["actor"], updated_result, "agent_operation_approved"
            )
        except (AgentTaskError, AttributeError, KeyError, ValueError) as error:
            # The job is already minted and will run; only the executed marker
            # failed to persist. Surface that rather than swallowing it, so the
            # operator is not shown a proposal that has in fact been approved.
            updated = task
            recording_error = str(error)[:240]
        security.audit(
            g.auth_session.username,
            "assistant.agent_operation.approve",
            "success",
            target=job["id"],
            remote_addr=request.remote_addr or "",
            details={
                "task_id": task_id,
                "operation": str(entry.get("operation", ""))[:64],
                "type": proposal["type"],
            },
        )
        body: Dict[str, Any] = {"task": updated, "job": job}
        if recording_error:
            body["recording_error"] = recording_error
        return _payload(body, status=202)
