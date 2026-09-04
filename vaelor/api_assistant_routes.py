"""Assistant skills, tasks, automations, conversations, and memory routes."""

from __future__ import annotations

from flask import g, request

from .agent_tasks import AgentTaskError
from .assistant_memory import AssistantMemoryError
from .assistant_skills import AssistantSkillError
from .assistant_tools import AssistantToolError
from .automations import AutomationError
from .custom_agents import CustomAgentError
from .rag_chat import RagChatError
from .api_common import ApiContext, assistant_model_status, payload as _payload
from .api_assistant_memory_routes import register_assistant_memory_routes

def register_assistant_routes(context: ApiContext) -> None:
    blueprint = context.blueprint
    callbacks = context.callbacks
    security = context.security
    limiter = context.limiter
    require_auth = context.require_auth
    appliance_address = context.appliance_address
    register_assistant_memory_routes(context)

    def validate_agent_collections(body):
        rag_store = callbacks.get("rag_chat")
        references = [
            *body.get("read_collection_ids", []),
            body.get("write_collection_id", ""),
        ]
        references = {str(item) for item in references if item}
        if not references:
            return
        owned = {
            item["id"] for item in rag_store.collections(g.auth_session.username)
        } if rag_store is not None else set()
        if not references.issubset(owned):
            raise CustomAgentError("Choose only RAG collections owned by this account.")

    @blueprint.get("/assistant/status")
    @require_auth("administrator")
    def assistant_memory_status():
        assistant_store = callbacks.get("assistant_memory")
        if assistant_store is None:
            return _payload(
                error={"code": "assistant_memory_unavailable", "message": "Assistant memory is unavailable."},
                status=503,
            )
        return _payload(assistant_store.stats(g.auth_session.username))

    @blueprint.get("/assistant/skills")
    @require_auth("administrator")
    def assistant_skill_list():
        skill_store = callbacks.get("assistant_skills")
        if skill_store is None:
            return _payload(error={"code": "skills_unavailable", "message": "Assistant skills are unavailable."}, status=503)
        model = assistant_model_status(callbacks, g.auth_session.username)
        return _payload([
            {**skill, "model_ready": model["ready"]}
            for skill in skill_store.list(request.args.get("status"))
        ])

    @blueprint.post("/assistant/skills")
    @require_auth("administrator", csrf=True)
    def assistant_skill_propose():
        skill_store = callbacks.get("assistant_skills")
        body = request.get_json(silent=True) or {}
        try:
            skill = skill_store.propose(
                g.auth_session.username, body.get("name", ""),
                body.get("description", ""), body.get("content", ""),
            )
        except (AttributeError, AssistantSkillError) as error:
            return _payload(error={"code": "skill_proposal_rejected", "message": str(error)}, status=400)
        security.audit(
            g.auth_session.username, "assistant.skill.propose", "success",
            target=skill["id"], remote_addr=request.remote_addr or "",
        )
        return _payload(skill, status=201)

    @blueprint.post("/assistant/skills/<skill_id>/review")
    @require_auth("administrator", csrf=True)
    def assistant_skill_review(skill_id):
        skill_store = callbacks.get("assistant_skills")
        body = request.get_json(silent=True) or {}
        if (
            str(body.get("decision", "")) == "active"
            and not assistant_model_status(
                callbacks, g.auth_session.username
            )["ready"]
        ):
            return _payload(
                error={
                    "code": "assistant_model_required",
                    "message": "Select a local or connected model before activating skills.",
                },
                status=409,
            )
        try:
            skill = skill_store.review(
                skill_id, g.auth_session.username, str(body.get("decision", ""))
            )
        except (AttributeError, AssistantSkillError) as error:
            return _payload(error={"code": "skill_review_rejected", "message": str(error)}, status=400)
        security.audit(
            g.auth_session.username, "assistant.skill.review", "success",
            target=skill_id, remote_addr=request.remote_addr or "",
            details={"decision": skill["status"]},
        )
        return _payload(skill)

    @blueprint.patch("/assistant/skills/<skill_id>")
    @require_auth("administrator", csrf=True)
    def assistant_skill_update(skill_id):
        skill_store = callbacks.get("assistant_skills")
        body = request.get_json(silent=True) or {}
        try:
            skill = skill_store.update(
                skill_id, g.auth_session.username, body.get("name", ""),
                body.get("description", ""), body.get("content", ""),
            )
        except (AttributeError, AssistantSkillError) as error:
            return _payload(
                error={"code": "skill_update_rejected", "message": str(error)},
                status=400,
            )
        security.audit(
            g.auth_session.username, "assistant.skill.update", "success",
            target=skill_id, remote_addr=request.remote_addr or "",
            details={"version": skill["version"], "status": skill["status"]},
        )
        return _payload(skill)

    @blueprint.delete("/assistant/skills/<skill_id>")
    @require_auth("administrator", csrf=True)
    def assistant_skill_delete(skill_id):
        skill_store = callbacks.get("assistant_skills")
        try:
            result = skill_store.delete(skill_id)
        except (AttributeError, AssistantSkillError) as error:
            return _payload(
                error={"code": "skill_delete_rejected", "message": str(error)},
                status=400,
            )
        security.audit(
            g.auth_session.username, "assistant.skill.delete", "success",
            target=skill_id, remote_addr=request.remote_addr or "",
        )
        return _payload(result)

    @blueprint.get("/assistant/tools")
    @require_auth("viewer")
    def assistant_tool_catalog():
        registry = callbacks.get("assistant_tools")
        if registry is None:
            return _payload(
                error={"code": "assistant_tools_unavailable", "message": "Assistant tools are unavailable."},
                status=503,
            )
        return _payload(list(registry.catalog()))

    @blueprint.post("/assistant/tools/<tool_name>/run")
    @require_auth("operator", csrf=True)
    def assistant_tool_run(tool_name):
        registry = callbacks.get("assistant_tools")
        if registry is None:
            return _payload(
                error={"code": "assistant_tools_unavailable", "message": "Assistant tools are unavailable."},
                status=503,
            )
        body = request.get_json(silent=True) or {}
        try:
            result = registry.run(
                tool_name,
                body.get("arguments", {}),
                actor=g.auth_session.username,
                administrator=g.auth_session.role == "administrator",
            )
        except AssistantToolError as error:
            security.audit(
                g.auth_session.username, "assistant.tool.run", "rejected",
                target=tool_name, remote_addr=request.remote_addr or "",
            )
            return _payload(
                error={"code": "assistant_tool_rejected", "message": str(error)},
                status=400,
            )
        security.audit(
            g.auth_session.username, "assistant.tool.run", "success",
            target=tool_name, remote_addr=request.remote_addr or "",
            details={
                "scope": result["scope"],
                "risk": result["risk"],
                "duration_ms": result["duration_ms"],
            },
        )
        return _payload(result)

    @blueprint.get("/assistant/profiles")
    @require_auth("viewer")
    def assistant_profiles():
        task_store = callbacks.get("agent_tasks")
        if task_store is None:
            return _payload(
                error={"code": "agent_tasks_unavailable", "message": "Agent tasks are unavailable."},
                status=503,
            )
        registry = callbacks.get("assistant_tools")
        tools = list(registry.catalog()) if registry is not None else []
        available_scopes = {item["scope"] for item in tools}
        model = assistant_model_status(callbacks, g.auth_session.username)
        profiles = []
        for profile in task_store.profiles(g.auth_session.username):
            required = set(profile["scopes"])
            profiles.append({
                **profile,
                "operational": bool(profile.get("enabled", True)) and model["ready"] and required.issubset(available_scopes),
                "model_required": True,
                "model_ready": model["ready"],
                "tools": [
                    item["name"] for item in tools if item["scope"] in required
                ],
            })
        return _payload(profiles)

    # Authoring an agent is administration; running and reading one is not.
    # The Routines tab that hosts authoring is already administrator-only in the
    # UI, and the deciding evidence is inside this same resource: POST and
    # DELETE on `/assistant/custom-agents/<id>/app-grants` require an
    # administrator while their GET stays operator. Skills, the other
    # administrator-gated surface, is administrator on every verb. The API was
    # the side that disagreed, so it is the side that moves. Reads stay at
    # operator: an operator still has to see the agent it is allowed to run.
    @blueprint.post("/assistant/custom-agents")
    @require_auth("administrator", csrf=True)
    def custom_agent_create():
        store = callbacks.get("custom_agents")
        if store is None:
            return _payload(error={"code": "custom_agents_unavailable", "message": "Custom agents are unavailable."}, status=503)
        try:
            body = request.get_json(silent=True) or {}
            validate_agent_collections(body)
            agent = store.create(g.auth_session.username, body)
        except CustomAgentError as error:
            return _payload(error={"code": "invalid_custom_agent", "message": str(error)}, status=400)
        security.audit(
            g.auth_session.username, "assistant.agent.create", "success",
            target=agent["id"], remote_addr=request.remote_addr or "",
            details={"version": agent["version"], "scopes": agent["scopes"]},
        )
        return _payload(agent, status=201)

    @blueprint.post("/assistant/custom-agents/draft")
    @require_auth("administrator", csrf=True)
    def custom_agent_draft():
        store = callbacks.get("custom_agents")
        body = request.get_json(silent=True) or {}
        try:
            draft = store.draft(body.get("request", ""))
        except (AttributeError, CustomAgentError) as error:
            return _payload(error={"code": "custom_agent_draft_failed", "message": str(error)}, status=400)
        return _payload(draft)

    @blueprint.patch("/assistant/custom-agents/<agent_id>")
    @require_auth("administrator", csrf=True)
    def custom_agent_update(agent_id):
        store = callbacks.get("custom_agents")
        actor = g.auth_session.username
        previous = store.get(agent_id, actor) if store is not None else None
        try:
            body = request.get_json(silent=True) or {}
            validate_agent_collections(body)
            agent = store.update(agent_id, actor, body)
            grants = callbacks.get("agent_app_grants")
            if (
                grants is not None
                and previous is not None
                and int(previous.get("version", 0)) != int(agent.get("version", 0))
            ):
                grants.clone_version(
                    actor, agent_id, previous["version"], agent["version"]
                )
        except (AttributeError, CustomAgentError, ValueError) as error:
            return _payload(error={"code": "custom_agent_update_failed", "message": str(error)}, status=400)
        security.audit(
            g.auth_session.username, "assistant.agent.update", "success",
            target=agent_id, remote_addr=request.remote_addr or "",
            details={"version": agent["version"], "enabled": agent["enabled"]},
        )
        return _payload(agent)

    @blueprint.get("/assistant/custom-agents/<agent_id>/revisions")
    @require_auth("operator")
    def custom_agent_revisions(agent_id):
        store = callbacks.get("custom_agents")
        try:
            return _payload(store.revisions(agent_id, g.auth_session.username))
        except (AttributeError, CustomAgentError) as error:
            return _payload(error={"code": "custom_agent_not_found", "message": str(error)}, status=404)

    @blueprint.post("/assistant/custom-agents/<agent_id>/rollback")
    @require_auth("administrator", csrf=True)
    def custom_agent_rollback(agent_id):
        store = callbacks.get("custom_agents")
        body = request.get_json(silent=True) or {}
        try:
            version = int(body.get("version"))
            previous = store.get(agent_id, g.auth_session.username)
            agent = store.rollback(agent_id, g.auth_session.username, version)
            grants = callbacks.get("agent_app_grants")
            if grants is not None and previous is not None and int(previous.get("version", 0)) != int(agent.get("version", 0)):
                grants.clone_version(
                    g.auth_session.username, agent_id,
                    previous["version"], agent["version"],
                )
        except (AttributeError, CustomAgentError, TypeError, ValueError) as error:
            return _payload(error={"code": "custom_agent_rollback_failed", "message": str(error)}, status=400)
        security.audit(
            g.auth_session.username, "assistant.agent.rollback", "success",
            target=agent_id, remote_addr=request.remote_addr or "",
            details={"from_version": version, "new_version": agent["version"]},
        )
        return _payload(agent)

    @blueprint.delete("/assistant/custom-agents/<agent_id>")
    @require_auth("administrator", csrf=True)
    def custom_agent_delete(agent_id):
        store = callbacks.get("custom_agents")
        task_store = callbacks.get("agent_tasks")
        active = [
            item for item in task_store.list(actor=g.auth_session.username, limit=200)
            if item["profile"] == agent_id
            and item["state"] not in {"completed", "failed", "cancelled", "archived"}
        ]
        if active:
            return _payload(
                error={"code": "custom_agent_in_use", "message": "Disable or finish this agent's active tasks before deleting it."},
                status=409,
            )
        actor = g.auth_session.username
        grant_store = callbacks.get("agent_app_grants")
        active_grants = [
            item for item in (grant_store.list(actor, agent_id=agent_id, limit=200) if grant_store else [])
            if not item.get("revoked")
        ]
        automation_store = callbacks.get("automations")
        active_automations = [
            item for item in (automation_store.list(actor) if automation_store else [])
            if item.get("profile") == agent_id and item.get("enabled")
        ]
        if active_grants or active_automations:
            return _payload(
                error={
                    "code": "custom_agent_dependencies",
                    "message": "Revoke this agent's app grants and disable its schedules before deleting it.",
                    "details": {
                        "active_app_grants": len(active_grants),
                        "active_automations": len(active_automations),
                    },
                },
                status=409,
            )
        try:
            result = store.delete(agent_id, actor)
        except (AttributeError, CustomAgentError) as error:
            return _payload(error={"code": "custom_agent_not_found", "message": str(error)}, status=404)
        security.audit(
            g.auth_session.username, "assistant.agent.delete", "success",
            target=agent_id, remote_addr=request.remote_addr or "",
        )
        return _payload(result)

    @blueprint.get("/assistant/tasks")
    @require_auth("operator")
    def assistant_tasks():
        task_store = callbacks.get("agent_tasks")
        if task_store is None:
            return _payload(
                error={"code": "agent_tasks_unavailable", "message": "Agent tasks are unavailable."},
                status=503,
            )
        actor = None if g.auth_session.role == "administrator" else g.auth_session.username
        archived = request.args.get("archived", "").lower() in {"1", "true", "yes"}
        tasks = task_store.list(actor=actor, limit=request.args.get("limit", 100))
        return _payload([
            task for task in tasks
            if (task.get("state") == "archived") == archived
        ])

    @blueprint.post("/assistant/tasks/archive-finished")
    @require_auth("operator", csrf=True)
    def assistant_tasks_archive_finished():
        task_store = callbacks.get("agent_tasks")
        if task_store is None:
            return _payload(
                error={"code": "agent_tasks_unavailable", "message": "Agent tasks are unavailable."},
                status=503,
            )
        archived = task_store.archive_finished(g.auth_session.username)
        security.audit(
            g.auth_session.username, "assistant.tasks.archive_finished", "success",
            remote_addr=request.remote_addr or "", details={"count": archived},
        )
        return _payload({"archived": archived})

    @blueprint.post("/assistant/tasks")
    @require_auth("operator", csrf=True)
    def assistant_task_create():
        task_store = callbacks.get("agent_tasks")
        if task_store is None:
            return _payload(
                error={"code": "agent_tasks_unavailable", "message": "Agent tasks are unavailable."},
                status=503,
            )
        body = request.get_json(silent=True) or {}
        # Escalation changes only the MODEL, never the approval/role/read-only
        # envelope: a capable run drives the GPU ai-chat model instead of the NPU
        # one. It carries no NPU execution binding (the NPU readiness gate below
        # is for the NPU binding), so the run_once GPU check is its real gate.
        use_capable = bool(body.get("use_capable_model"))
        model = assistant_model_status(callbacks, g.auth_session.username)
        if not use_capable and not model["ready"]:
            return _payload(
                error={
                    "code": "assistant_model_required",
                    "message": "Select a local or connected model before running specialist agents.",
                },
                status=409,
            )
        try:
            task = task_store.create(
                g.auth_session.username,
                body.get("title", ""),
                body.get("description", ""),
                kind="durable",
                profile=body.get("profile", "system"),
                approval_required=True,
                idempotency_key=body.get("idempotency_key"),
                dependencies=body.get("dependencies", []),
                profile_version=body.get("profile_version"),
                model_tier="capable" if use_capable else "",
                execution_binding=None if use_capable else {
                    "mode": model["mode"],
                    "provider": model["provider"],
                    "model": model["model"],
                },
            )
        except (AgentTaskError, TypeError, ValueError) as error:
            return _payload(
                error={"code": "invalid_agent_task", "message": str(error)}, status=400
            )
        security.audit(
            g.auth_session.username, "assistant.task.create", "success",
            target=task["id"], remote_addr=request.remote_addr or "",
            details={"profile": task["profile"], "state": task["state"]},
        )
        return _payload(task, status=201)

    @blueprint.patch("/assistant/tasks/<task_id>/binding")
    @require_auth("operator", csrf=True)
    def assistant_task_binding(task_id):
        """Bind an interrupted run to the currently selected model runtime."""
        task_store = callbacks.get("agent_tasks")
        model = assistant_model_status(callbacks, g.auth_session.username)
        if task_store is None:
            return _payload(
                error={"code": "agent_tasks_unavailable", "message": "Agent tasks are unavailable."},
                status=503,
            )
        if not model["ready"]:
            return _payload(
                error={
                    "code": "assistant_model_required",
                    "message": "Select and verify a local or connected model before replacing this execution binding.",
                },
                status=409,
            )
        actor = None if g.auth_session.role == "administrator" else g.auth_session.username
        visible = task_store.get(task_id, actor)
        if visible is None:
            return _payload(error={"code": "agent_task_not_found", "message": "Task was not found."}, status=404)
        try:
            task = task_store.rebind_execution(task_id, visible["actor"], {
                "mode": model["mode"],
                "provider": model["provider"],
                "model": model["model"],
            })
        except (AttributeError, AgentTaskError, TypeError, ValueError) as error:
            return _payload(error={"code": "execution_binding_rejected", "message": str(error)}, status=409)
        security.audit(
            g.auth_session.username, "assistant.task.binding.replace", "success",
            target=task_id, remote_addr=request.remote_addr or "",
            details={"mode": model["mode"], "provider": model["provider"], "model": model["model"]},
        )
        return _payload(task)

    @blueprint.get("/assistant/tasks/<task_id>")
    @require_auth("operator")
    def assistant_task_detail(task_id):
        task_store = callbacks.get("agent_tasks")
        actor = None if g.auth_session.role == "administrator" else g.auth_session.username
        task = task_store.get(task_id, actor) if task_store is not None else None
        if task is None:
            return _payload(
                error={"code": "agent_task_not_found", "message": "Task was not found."},
                status=404,
            )
        return _payload(task_store.details(task_id, task["actor"]))

    @blueprint.post("/assistant/tasks/<task_id>/writes/<int:write_index>/approve")
    @require_auth("operator", csrf=True)
    def assistant_task_write_approve(task_id, write_index):
        task_store = callbacks.get("agent_tasks")
        rag_store = callbacks.get("rag_chat")
        task = task_store.get(task_id, g.auth_session.username) if task_store else None
        if task is None or task["state"] != "completed":
            return _payload(error={"code": "agent_write_not_found", "message": "A completed agent write proposal was not found."}, status=404)
        proposals = task.get("result", {}).get("proposed_writes", [])
        if not 0 <= write_index < len(proposals):
            return _payload(error={"code": "agent_write_not_found", "message": "Agent write proposal was not found."}, status=404)
        proposal = proposals[write_index]
        if proposal.get("type") != "knowledge.document" or proposal.get("executed"):
            return _payload(error={"code": "agent_write_rejected", "message": "This write is unsupported or already completed."}, status=409)
        try:
            document = rag_store.ingest(
                g.auth_session.username, proposal.get("collection_id", ""),
                proposal.get("name", "Agent report"),
                proposal.get("content", ""), proposal.get("media_type", "text/plain"),
            )
            updated_result = dict(task["result"])
            updated_writes = [dict(item) for item in proposals]
            updated_writes[write_index].update({
                "executed": True, "document_id": document["id"],
                "approved_by": g.auth_session.username,
            })
            updated_result["proposed_writes"] = updated_writes
            updated = task_store.update_completed_result(
                task_id, task["actor"], updated_result, "knowledge_write_approved"
            )
        except (AttributeError, AgentTaskError, RagChatError) as error:
            return _payload(error={"code": "agent_write_rejected", "message": str(error)}, status=400)
        security.audit(
            g.auth_session.username, "assistant.agent_write.approve", "success",
            target=document["id"], remote_addr=request.remote_addr or "",
            details={"task_id": task_id, "collection_id": proposal["collection_id"]},
        )
        return _payload({"task": updated, "document": document})

    @blueprint.patch("/assistant/tasks/<task_id>")
    @require_auth("operator", csrf=True)
    def assistant_task_transition(task_id):
        task_store = callbacks.get("agent_tasks")
        body = request.get_json(silent=True) or {}
        requested_state = str(body.get("state", ""))
        if requested_state not in {"ready", "cancelled", "archived"}:
            return _payload(
                error={
                    "code": "invalid_task_transition",
                    "message": "You may approve, cancel, or archive a task from this interface.",
                },
                status=400,
            )
        try:
            current = task_store.get(task_id, g.auth_session.username)
            if (
                requested_state == "ready" and current is not None
                and current["state"] == "needs_approval"
            ):
                task = task_store.approve(
                    task_id, g.auth_session.username, "approved",
                    str(body.get("note", "")),
                )
            else:
                task = task_store.transition(
                    task_id, g.auth_session.username, requested_state,
                    expected_state=current["state"] if current is not None else None,
                )
        except (AttributeError, AgentTaskError) as error:
            return _payload(
                error={"code": "agent_task_transition_failed", "message": str(error)},
                status=400,
            )
        security.audit(
            g.auth_session.username, "assistant.task.transition", "success",
            target=task_id, remote_addr=request.remote_addr or "",
            details={"state": requested_state},
        )
        return _payload(task)

    @blueprint.post("/assistant/tasks/<task_id>/comments")
    @require_auth("operator", csrf=True)
    def assistant_task_comment(task_id):
        task_store = callbacks.get("agent_tasks")
        body = request.get_json(silent=True) or {}
        try:
            comment = task_store.add_comment(
                task_id, g.auth_session.username, body.get("content", "")
            )
        except (AttributeError, AgentTaskError) as error:
            return _payload(error={"code": "task_comment_rejected", "message": str(error)}, status=400)
        security.audit(
            g.auth_session.username, "assistant.task.comment", "success",
            target=task_id, remote_addr=request.remote_addr or "",
        )
        return _payload(comment, status=201)

    @blueprint.post("/assistant/tasks/<task_id>/approval")
    @require_auth("operator", csrf=True)
    def assistant_task_approval(task_id):
        task_store = callbacks.get("agent_tasks")
        body = request.get_json(silent=True) or {}
        try:
            task = task_store.approve(
                task_id, g.auth_session.username,
                str(body.get("decision", "")), str(body.get("note", "")),
            )
        except (AttributeError, AgentTaskError) as error:
            return _payload(error={"code": "task_approval_rejected", "message": str(error)}, status=400)
        security.audit(
            g.auth_session.username, "assistant.task.approval", "success",
            target=task_id, remote_addr=request.remote_addr or "",
            details={"decision": body.get("decision")},
        )
        return _payload(task)

    @blueprint.post("/assistant/tasks/<task_id>/retry")
    @require_auth("operator", csrf=True)
    def assistant_task_retry(task_id):
        task_store = callbacks.get("agent_tasks")
        body = request.get_json(silent=True) or {}
        # A user-triggered escalation re-runs the SAME task on the capable GPU
        # model. Same approval/role/read-only envelope as any retry - only the
        # model tier changes; run_once verifies the GPU is actually available.
        use_capable = bool(body.get("use_capable_model"))
        visible = task_store.get(
            task_id, None if g.auth_session.role == "administrator" else g.auth_session.username
        )
        if visible is None:
            return _payload(error={"code": "agent_task_not_found", "message": "Task was not found."}, status=404)
        try:
            task = task_store.retry(
                task_id, visible["actor"],
                model_tier="capable" if use_capable else None,
            )
        except (AttributeError, AgentTaskError) as error:
            return _payload(error={"code": "task_retry_rejected", "message": str(error)}, status=400)
        security.audit(g.auth_session.username, "assistant.task.retry", "success", target=task["id"], remote_addr=request.remote_addr or "", details={"retry_of": task_id, "model_tier": task.get("model_tier", "")})
        return _payload(task, status=201)

    @blueprint.get("/assistant/handoff-targets")
    @require_auth("operator")
    def assistant_handoff_targets():
        return _payload([
            {"username": item["username"], "role": item["role"]}
            for item in security.list_users()
            if item["enabled"] and item["role"] in {"operator", "administrator"}
        ])

    @blueprint.post("/assistant/tasks/<task_id>/handoff")
    @require_auth("operator", csrf=True)
    def assistant_task_handoff(task_id):
        task_store = callbacks.get("agent_tasks")
        body = request.get_json(silent=True) or {}
        assigned_to = str(body.get("assigned_to", "")).strip().lower()
        accounts = {
            item["username"]
            for item in security.list_users()
            if item["enabled"] and item["role"] in {"operator", "administrator"}
        }
        if assigned_to not in accounts:
            return _payload(error={"code": "handoff_account_not_found", "message": "Choose an enabled local account."}, status=400)
        visible = task_store.get(
            task_id, None if g.auth_session.role == "administrator" else g.auth_session.username
        )
        if visible is None:
            return _payload(error={"code": "agent_task_not_found", "message": "Task was not found."}, status=404)
        try:
            task = task_store.handoff(task_id, visible["actor"], assigned_to)
        except (AttributeError, AgentTaskError) as error:
            return _payload(error={"code": "task_handoff_rejected", "message": str(error)}, status=400)
        security.audit(g.auth_session.username, "assistant.task.handoff", "success", target=task_id, remote_addr=request.remote_addr or "", details={"assigned_to": assigned_to})
        return _payload(task)

    @blueprint.post("/assistant/delegations")
    @require_auth("operator", csrf=True)
    def assistant_delegate():
        coordinator = callbacks.get("subagents")
        if coordinator is None:
            return _payload(
                error={"code": "subagents_unavailable", "message": "Specialist agents are unavailable."},
                status=503,
            )
        if not assistant_model_status(
            callbacks, g.auth_session.username
        )["ready"]:
            return _payload(
                error={
                    "code": "assistant_model_required",
                    "message": "Select a local or connected model before running specialist agents.",
                },
                status=409,
            )
        body = request.get_json(silent=True) or {}
        try:
            task = coordinator.delegate(
                g.auth_session.username,
                str(body.get("profile", "")),
                str(body.get("task", "")),
                parent_id=body.get("parent_id"),
                idempotency_key=body.get("idempotency_key"),
                # Built-in `docker` and `system` already carry `jobs:read`, and
                # this endpoint is operator-reachable, so a delegation that did
                # not carry the caller's role read every account's job history.
                administrator=g.auth_session.role == "administrator",
            )
        except (AgentTaskError, ValueError) as error:
            return _payload(
                error={"code": "delegation_rejected", "message": str(error)}, status=400
            )
        security.audit(
            g.auth_session.username, "assistant.delegate", "success",
            target=task["id"], remote_addr=request.remote_addr or "",
            details={
                "profile": task["profile"],
                "state": task["state"],
                "executed_changes": False,
            },
        )
        return _payload(task, status=201)

    # Creating a schedule or an alert rule is the only operator-reachable way to
    # get an agent run that no human approves at the moment it happens, and the
    # rule outlives the session that made it. That is standing appliance
    # configuration, which is the same shape as skills and curated memory - both
    # administrator-only - not the same shape as asking for one run. The tab has
    # been administrator-only for three releases; the API now agrees with it.
    # Reading stays at operator: the listing is filtered to the caller's own rows
    # and History, which every reader has, uses it to label a run nobody typed.
    @blueprint.get("/assistant/automations")
    @require_auth("operator")
    def assistant_automations():
        store = callbacks.get("automations")
        if store is None:
            return _payload(error={"code": "automations_unavailable", "message": "Automations are unavailable."}, status=503)
        actor = None if g.auth_session.role == "administrator" else g.auth_session.username
        task_store = callbacks.get("agent_tasks")
        return _payload({
            "schedules": store.list(actor),
            "runs": store.sync_runs(task_store, limit=100),
            # Alert-rule runs are recorded separately from scheduled runs; the
            # client needs both to mark a run as automatic rather than a check.
            "trigger_runs": store.trigger_runs(limit=100),
            "triggers": store.list_triggers(actor),
        })

    @blueprint.post("/assistant/automations")
    @require_auth("administrator", csrf=True)
    def assistant_automation_create():
        store = callbacks.get("automations")
        body = request.get_json(silent=True) or {}
        if not assistant_model_status(
            callbacks, g.auth_session.username
        )["ready"]:
            return _payload(
                error={
                    "code": "assistant_model_required",
                    "message": "Select a local or connected model before scheduling agents.",
                },
                status=409,
            )
        try:
            automation = store.create(
                g.auth_session.username, body.get("name", ""), body.get("prompt", ""),
                body.get("profile", "system"), body.get("schedule", ""),
            )
        except (AttributeError, AutomationError) as error:
            return _payload(error={"code": "automation_rejected", "message": str(error)}, status=400)
        security.audit(
            g.auth_session.username, "assistant.automation.create", "success",
            target=automation["id"], remote_addr=request.remote_addr or "",
            details={"kind": automation["kind"], "profile": automation["profile"]},
        )
        return _payload(automation, status=201)

    @blueprint.patch("/assistant/automations/<automation_id>")
    @require_auth("administrator", csrf=True)
    def assistant_automation_update(automation_id):
        store = callbacks.get("automations")
        body = request.get_json(silent=True) or {}
        if not isinstance(body.get("enabled"), bool):
            return _payload(error={"code": "automation_rejected", "message": "Choose whether the schedule is enabled."}, status=400)
        try:
            automation = store.set_enabled(
                automation_id, g.auth_session.username, body["enabled"]
            )
        except (AttributeError, AutomationError) as error:
            return _payload(error={"code": "automation_rejected", "message": str(error)}, status=400)
        security.audit(
            g.auth_session.username, "assistant.automation.update", "success",
            target=automation_id, remote_addr=request.remote_addr or "",
            details={"enabled": automation["enabled"]},
        )
        return _payload(automation)

    @blueprint.delete("/assistant/automations/<automation_id>")
    @require_auth("administrator", csrf=True)
    def assistant_automation_delete(automation_id):
        store = callbacks.get("automations")
        try:
            result = store.delete(automation_id, g.auth_session.username)
        except (AttributeError, AutomationError) as error:
            return _payload(
                error={"code": "automation_rejected", "message": str(error)},
                status=400,
            )
        security.audit(
            g.auth_session.username, "assistant.automation.delete", "success",
            target=automation_id, remote_addr=request.remote_addr or "",
        )
        return _payload(result)

    @blueprint.post("/assistant/triggers")
    @require_auth("administrator", csrf=True)
    def assistant_trigger_create():
        store = callbacks.get("automations")
        body = request.get_json(silent=True) or {}
        try:
            trigger = store.create_trigger(
                g.auth_session.username, body.get("name", ""), body.get("prompt", ""),
                body.get("profile", "system"), body.get("source", ""),
                body.get("operator", ">="), body.get("threshold"),
                body.get("cooldown_seconds", 1800),
            )
        except (AttributeError, AutomationError) as error:
            return _payload(error={"code": "trigger_rejected", "message": str(error)}, status=400)
        security.audit(g.auth_session.username, "assistant.trigger.create", "success", target=trigger["id"], remote_addr=request.remote_addr or "", details={"source": trigger["source"], "threshold": trigger["threshold"]})
        return _payload(trigger, status=201)

    @blueprint.patch("/assistant/triggers/<trigger_id>")
    @require_auth("administrator", csrf=True)
    def assistant_trigger_update(trigger_id):
        store = callbacks.get("automations")
        body = request.get_json(silent=True) or {}
        if not isinstance(body.get("enabled"), bool):
            return _payload(error={"code": "trigger_rejected", "message": "Choose whether the alert is enabled."}, status=400)
        try:
            trigger = store.set_trigger_enabled(
                trigger_id, g.auth_session.username, body["enabled"]
            )
        except (AttributeError, AutomationError) as error:
            return _payload(error={"code": "trigger_rejected", "message": str(error)}, status=400)
        security.audit(g.auth_session.username, "assistant.trigger.update", "success", target=trigger_id, remote_addr=request.remote_addr or "", details={"enabled": body["enabled"]})
        return _payload(trigger)

    @blueprint.delete("/assistant/triggers/<trigger_id>")
    @require_auth("administrator", csrf=True)
    def assistant_trigger_delete(trigger_id):
        store = callbacks.get("automations")
        try:
            result = store.delete_trigger(trigger_id, g.auth_session.username)
        except (AttributeError, AutomationError) as error:
            return _payload(
                error={"code": "trigger_rejected", "message": str(error)},
                status=400,
            )
        security.audit(
            g.auth_session.username, "assistant.trigger.delete", "success",
            target=trigger_id, remote_addr=request.remote_addr or "",
        )
        return _payload(result)

    @blueprint.get("/assistant/conversations")
    @require_auth("operator")
    def assistant_conversations():
        assistant_store = callbacks.get("assistant_memory")
        if assistant_store is None:
            return _payload([])
        return _payload(
            assistant_store.list_conversations(
                g.auth_session.username,
                request.args.get("limit", 30),
                include_archived=request.args.get("archived", "").lower()
                in {"1", "true", "yes"},
            )
        )

    @blueprint.patch("/assistant/conversations/<conversation_id>")
    @require_auth("operator", csrf=True)
    def assistant_conversation_update(conversation_id):
        assistant_store = callbacks.get("assistant_memory")
        if assistant_store is None:
            return _payload(
                error={"code": "assistant_memory_unavailable", "message": "Assistant history is unavailable."},
                status=503,
            )
        try:
            conversation = assistant_store.update_conversation(
                g.auth_session.username,
                conversation_id,
                request.get_json(silent=True) or {},
            )
        except AssistantMemoryError as error:
            return _payload(error={"code": "invalid_conversation", "message": str(error)}, status=400)
        security.audit(
            g.auth_session.username, "assistant.conversation.update", "success",
            target=conversation_id, remote_addr=request.remote_addr or "",
        )
        return _payload(conversation)

    @blueprint.delete("/assistant/conversations/<conversation_id>")
    @require_auth("operator", csrf=True)
    def assistant_conversation_delete(conversation_id):
        assistant_store = callbacks.get("assistant_memory")
        if assistant_store is None or not assistant_store.delete_conversation(
            g.auth_session.username, conversation_id
        ):
            return _payload(
                error={"code": "conversation_not_found", "message": "Conversation was not found."},
                status=404,
            )
        security.audit(
            g.auth_session.username, "assistant.conversation.delete", "success",
            target=conversation_id, remote_addr=request.remote_addr or "",
        )
        return _payload({"deleted": True})

    @blueprint.get("/assistant/conversations/<conversation_id>/export")
    @require_auth("operator")
    def assistant_conversation_export(conversation_id):
        assistant_store = callbacks.get("assistant_memory")
        if assistant_store is None:
            return _payload(
                error={"code": "assistant_memory_unavailable", "message": "Assistant history is unavailable."},
                status=503,
            )
        try:
            return _payload(
                assistant_store.export_conversation(
                    g.auth_session.username, conversation_id
                )
            )
        except AssistantMemoryError as error:
            return _payload(error={"code": "conversation_not_found", "message": str(error)}, status=404)

    @blueprint.get("/assistant/conversations/<conversation_id>/messages")
    @require_auth("operator")
    def assistant_conversation_messages(conversation_id):
        assistant_store = callbacks.get("assistant_memory")
        if assistant_store is None:
            return _payload([])
        return _payload(
            assistant_store.conversation_messages(
                g.auth_session.username, conversation_id, request.args.get("limit", 100)
            )
        )

    @blueprint.get("/assistant/search")
    @require_auth("operator")
    def assistant_session_search():
        assistant_store = callbacks.get("assistant_memory")
        if assistant_store is None:
            return _payload([])
        try:
            results = assistant_store.search_messages(
                g.auth_session.username,
                request.args.get("query", ""),
                request.args.get("limit", 20),
            )
        except (TypeError, ValueError) as error:
            return _payload(
                error={"code": "invalid_session_search", "message": str(error)},
                status=400,
            )
        return _payload(results)
