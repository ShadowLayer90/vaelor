"""One-level, read-only temporary specialist coordination."""

from __future__ import annotations

import threading
from typing import Any, Dict

from .agent_knowledge import attach_write_proposal, retrieve_agent_knowledge
from .agent_tasks import AgentTaskError
from .assistant_tools import AssistantToolError
from .live_readings import one_reading_per_answer


class SubagentCoordinator:
    def __init__(
        self, task_store, tool_registry, agent, skill_store=None,
        preference_store=None, knowledge_store=None,
    ):
        self.task_store = task_store
        self.tool_registry = tool_registry
        self.agent = agent
        self.skill_store = skill_store
        self.preference_store = preference_store
        self.knowledge_store = knowledge_store
        self._local_slot = threading.BoundedSemaphore(1)

    def delegate(
        self, actor: str, profile: str, task: str, *,
        parent_id=None, idempotency_key=None, administrator: bool = False,
    ) -> Dict[str, Any]:
        """`administrator` is the caller's live role, and it decides how much a
        read-only tool may return.

        A delegation runs inside the request that asked for it, so the role is
        the session's rather than something resolved later. It defaults to
        False: a caller that does not say who is asking gets an operator's view,
        not the appliance's.
        """
        profile_definition = self.task_store.profile(profile, actor)
        if profile_definition is None or not profile_definition.get("enabled"):
            raise AgentTaskError("Unknown specialist profile.")
        child = self.task_store.create(
            actor,
            "{} review".format(profile_definition["name"]),
            task,
            kind="temporary",
            profile=profile,
            parent_id=parent_id,
            depth=1,
            idempotency_key=idempotency_key,
        )
        if child["state"] == "completed":
            return child
        acquired = self._local_slot.acquire(timeout=2)
        if not acquired:
            return self.task_store.transition(
                child["id"], actor, "blocked",
                error="Another local specialist is running. Retry when it finishes.",
            )
        try:
            self.task_store.transition(child["id"], actor, "running")
            allowed_scopes = set(profile_definition["scopes"])
            facts = {}
            # One review, one hardware sample, so two facts about the same
            # sensor cannot disagree inside a single result.
            with one_reading_per_answer():
                for tool_definition in self.tool_registry.catalog():
                    if tool_definition["scope"] not in allowed_scopes:
                        continue
                    try:
                        facts[tool_definition["name"]] = self.tool_registry.run(
                            tool_definition["name"], {}, actor=actor,
                            administrator=administrator,
                        )["result"]
                    except AssistantToolError as error:
                        facts[tool_definition["name"]] = {"unavailable": str(error)}
            knowledge = retrieve_agent_knowledge(
                self.knowledge_store, actor, profile_definition, task
            )
            if knowledge:
                facts["knowledge.sources"] = knowledge
            result = self.agent.specialist(
                profile, task, context={
                    "facts": facts,
                    "matched_skills": [
                        {
                            "name": item["name"],
                            "version": item["version"],
                            "guidance": item["content"],
                        }
                        for item in (
                            self.skill_store.match(task)
                            if self.skill_store is not None else []
                        )
                    ],
                    "policy": {
                        "direct_mutations": False,
                        "typed_write_proposals": (
                            "knowledge:write" in profile_definition.get("permissions", [])
                        ),
                        "delegation_depth": 1,
                        "memory_writes": False,
                        "credential_access": False,
                        "mutation_requires_separate_approval": True,
                    },
                    "agent_definition": {
                        "name": profile_definition["name"],
                        "instructions": profile_definition.get("instructions", ""),
                        "version": profile_definition.get("version", 0),
                    },
                },
                mode=(
                    self.preference_store.get_preference(
                        actor, "intelligence_choice", ""
                    )
                    if self.preference_store is not None else ""
                ),
            )
            if knowledge:
                result["knowledge_sources"] = [
                    {
                        "collection": item["collection_name"],
                        "document": item["document_name"],
                        "chunk": int(item["ordinal"]) + 1,
                    }
                    for item in knowledge
                ]
            result = attach_write_proposal(
                result, profile_definition, child["id"],
                self.knowledge_store, actor,
            )
            return self.task_store.transition(
                child["id"], actor, "completed", result=result
            )
        except (ValueError, AgentTaskError) as error:
            return self.task_store.transition(
                child["id"], actor, "failed", error=str(error)
            )
        finally:
            self._local_slot.release()
