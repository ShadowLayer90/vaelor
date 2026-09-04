"""Background execution for restart-safe, read-only durable agent tasks."""

from __future__ import annotations

import logging
import threading
import time
from typing import Any

from .agent_failure_messages import custom_agent_failure
from .agent_research_relevance import clarifying_questions
from .agent_search_planner import acquire_web_research
from .agent_result_shape import (
    OUTCOME_NEEDS_INPUT, OUTCOME_NO_RESULT, capable_model_name,
    escalation_offer, task_outcome,
)
from .agent_tool_loop import (
    AgentToolLoopFallback,
    current_date_utc,
    drive_custom_agent_loop,
    loop_eligible,
)
from .model_reachability import note_inference_outcome
from .agent_knowledge import attach_write_proposal, retrieve_agent_knowledge
from .agent_tasks import AgentTaskError
from .assistant_action_proposals import (
    ActingProposalError,
    managed_projects,
    resolve_agent_operation,
    sanitize_proposed_operations,
)
from .assistant_intents import order_facts_by_relevance
from .assistant_tools import AssistantToolError
from .assistant_working_context import assemble_working_context
from .live_readings import one_reading_per_answer

LOGGER = logging.getLogger(__name__)


def _task_context_policy(description: str) -> str:
    """Honor an explicit user request to avoid widening supplied context."""
    normalized = " ".join(str(description).lower().split())
    supplied_only = (
        "using only these notes", "use only these notes", "from these notes only",
        "supplied evidence only", "provided evidence only",
    )
    if any(cue in normalized for cue in supplied_only):
        return "supplied-only"
    no_external = (
        "do not search the web", "don't search the web", "no web search",
        "without web search", "offline only",
    )
    if any(cue in normalized for cue in no_external):
        return "no-external"
    return "granted"



def _safe_capability_failure(error: Any) -> str:
    message = " ".join(str(error).split())[:240]
    lowered = message.lower()
    if any(item in lowered for item in ("http://", "https://", "endpoint", "header", "credential", "password", "secret", "token", "authorization")):
        return "The app capability was blocked or unavailable. Review the pinned grant and connection."
    return message or "The app capability was blocked or unavailable."


def _bounded_capability_value(value: Any) -> Any:
    def public(item: Any) -> Any:
        if isinstance(item, dict):
            result = {}
            for key, child in item.items():
                normalized = str(key).lower().replace("-", "_")
                if any(part in normalized for part in ("url", "uri", "endpoint", "header", "authorization", "credential", "password", "secret", "token")):
                    result[str(key)] = "[redacted]"
                else:
                    result[str(key)] = public(child)
            return result
        if isinstance(item, list):
            return [public(child) for child in item]
        if isinstance(item, str) and item.lower().startswith(("http://", "https://")):
            return "[redacted]"
        return item

    try:
        import json
        encoded = json.dumps(public(value), separators=(",", ":"), default=str)
        if len(encoded.encode("utf-8")) > 16 * 1024:
            return {"status": "result_too_large"}
        return json.loads(encoded)
    except (TypeError, ValueError):
        return {"status": "result_unavailable"}

def _pinned_app_operation(grants: list[dict[str, Any]], grant_id: str, operation_id: str):
    for grant in grants:
        if str(grant.get("grant_id", "")) != grant_id:
            continue
        for operation in grant.get("operations", []):
            if isinstance(operation, dict) and str(operation.get("operation_id", "")) == operation_id:
                return grant, operation
    return None, None


def _structured_result(value: Any) -> dict[str, Any]:
    """Normalize model output into the fields the UI and audit ledger understand."""
    if not isinstance(value, dict):
        raise AgentTaskError("The selected model did not return a structured result.")

    def strings(key: str, limit: int = 8) -> list[str]:
        raw = value.get(key, [])
        if isinstance(raw, str):
            raw = [raw]
        if not isinstance(raw, list):
            return []
        return [" ".join(str(item).split())[:800] for item in raw[:limit] if str(item).strip()]

    def references(key: str) -> list[dict[str, str]]:
        raw = value.get(key, [])
        if not isinstance(raw, list):
            return []
        result = []
        for item in raw[:20]:
            if not isinstance(item, dict):
                continue
            source = str(item.get("source") or item.get("url") or item.get("title") or "").strip()[:400]
            summary = " ".join(str(item.get("summary") or item.get("excerpt") or "").split())[:800]
            if source:
                result.append({"source": source, "summary": summary})
        return result

    normalized: dict[str, Any] = {
        "source": str(value.get("source", "custom-agent-model"))[:80],
        "summary": " ".join(str(value.get("summary", "Review completed.")).split())[:1200],
        "findings": strings("findings"),
        "recommendations": strings("recommendations"),
        "next_actions": strings("next_actions"),
        "evidence": references("evidence"),
        "sources": references("sources"),
        # A model that recognises an ambiguous or under-specified task may ask
        # for more detail rather than guess. Kept here so the runner can surface
        # it; the runner's own deterministic questions (#247w) are the guarantee
        # when the model does not, since normalization otherwise drops it.
        "clarifications": strings("clarifications"),
        "warnings": strings("warnings"),
        "errors": strings("errors"),
        "executed_changes": False,
        # A run that never reached the model is not the same as one that did.
        # This was previously visible only as the first line of the findings
        # list, inside a collapsed disclosure, under a green "completed" badge.
        "degraded": bool(value.get("fallback_used")),
    }
    answer = " ".join(str(value.get("answer", "")).split())[:5000]
    if answer:
        normalized["answer"] = answer
    return normalized


#: Fact keys produced by the guarded web pipeline. Everything else in `facts` -
#: cooling, system, workloads, cluster, jobs, assistant, knowledge, app and
#: connector results - is local substance the model can answer from without the
#: web, so a junk or empty search must not flip such a run to "needs input".
_WEB_FACT_KEYS = frozenset({"research.search", "research.fetch"})


def _has_local_substance(facts: dict) -> bool:
    """Whether the run held any non-web fact the model could answer from.

    #247w finding 2: a web-enabled agent asked "check my fan health" still runs a
    search; if it returns nothing on topic, the sensor-based answer it produced
    from local facts must stay a delivered answer, not be flipped to an
    unverified "needs input" that asks for an irrelevant docs URL.
    """
    for key, value in (facts or {}).items():
        if key in _WEB_FACT_KEYS:
            continue
        if isinstance(value, dict) and value.get("unavailable"):
            continue
        if value:
            return True
    return False


def web_access_status(web_access, search, *, candidates, fetched_any, fetch_failed):
    """A *granted* web agent's down dependency, named so the answer can say it.

    LESSONS #4 (a wiring gap read as "unavailable") and #11 (a measurement taken
    and never read). #212: on the live box the guarded-search backend (SearXNG
    `system-web-research`) was absent, so `research.search` failed, the failure
    was recorded in `tool_failures`, and the run then answered with no
    candidates - so the grant read as "no access". This reads that failure and
    names which dependency; it fires only when web access is *enabled*, and a
    genuine zero-hit search stays silent (a working search is not an outage).

    ``search`` is the run's ``facts["research.search"]``: ``{"unavailable": ...}``
    when the search tool failed, else the success dict with ``results`` /
    ``excluded_results``. Returns a status dict or ``None``.
    """
    if (web_access or {}).get("enabled") is not True:
        return None
    search = search if isinstance(search, dict) else {}
    if "unavailable" in search:
        return {
            "granted": True, "dependency": "search", "state": "unavailable",
            "detail": (
                "Web access is granted, but the guarded web research search "
                "service is unavailable, so no sources could be gathered. "
                "Install or repair guarded web research, then run this again."
            ),
        }
    if candidates and not fetched_any and fetch_failed:
        return {
            "granted": True, "dependency": "fetch", "state": "unavailable",
            "detail": (
                "Web access is granted and the search returned candidates, but "
                "the research fetch broker could not read any of them. Review "
                "the fetch broker, then run this again."
            ),
        }
    domains = list((web_access or {}).get("allowed_domains") or [])
    if domains and search.get("results") == [] and search.get("excluded_results"):
        return {
            "granted": True, "dependency": "allowlist", "state": "excluded_all",
            "detail": (
                "Web access is granted, but every search result was outside the "
                "allowed domains ({}), so nothing could be read. Widen the "
                "allowed domains, then run this again.".format(", ".join(domains[:5]))
            ),
        }
    return None


class AgentTaskRunner:
    def __init__(
        self, store, tools, agent, skill_store=None, preference_store=None,
        poll_seconds: float = 2.0, knowledge_store=None,
        connector_runtime=None,
        app_capability_broker=None,
        administrator_resolver=None,
        memory_store=None,
        learning_store=None,
    ):
        self.store = store
        self.tools = tools
        self.agent = agent
        # A queued task carries an actor but no role, and the queue outlives the
        # request that filled it. Whether the owner may read appliance-wide
        # history is therefore decided here, at run time, against the live user
        # table - a demotion between queueing and running has to count. Without
        # a resolver the run is treated as an ordinary operator's.
        self.administrator_resolver = administrator_resolver
        self.skill_store = skill_store
        self.preference_store = preference_store
        self.poll_seconds = max(0.2, min(float(poll_seconds), 30))
        self.knowledge_store = knowledge_store
        # learning_store grounds a custom-agent run with actor-scoped sanitized
        # lessons (which fed no prompt before). memory_store is a DORMANT seam:
        # curated memory is global, so feeding agents needs a per-agent memory
        # permission that does not exist yet; run_once never passes it. Default None.
        self.memory_store = memory_store
        self.learning_store = learning_store
        self.connector_runtime = connector_runtime
        self.app_capability_broker = app_capability_broker
        self._stop = threading.Event()
        self._thread = None

    def start(self):
        if self._thread and self._thread.is_alive():
            return
        self.store.recover_stale()
        self._thread = threading.Thread(
            target=self._loop, name="pm-agent-task-runner", daemon=True
        )
        self._thread.start()

    def stop(self):
        self._stop.set()
        if self._thread:
            self._thread.join(timeout=3)

    def _loop(self):
        while not self._stop.is_set():
            try:
                completed = self.run_once()
            except Exception:
                # A malformed persisted definition or an unavailable backing
                # store must not kill the appliance's sole durable runner.
                completed = None
            self._stop.wait(0.2 if completed else self.poll_seconds)

    def _is_administrator(self, actor: str) -> bool:
        """Fail closed: no resolver, or an unreadable account table, is not
        grounds for handing a run every account's records."""
        if self.administrator_resolver is None:
            return False
        try:
            return bool(self.administrator_resolver(actor))
        except Exception:
            LOGGER.warning(
                "could not resolve the role for %s; treating the run as an operator's",
                actor,
            )
            return False

    def run_once(self):
        task = self.store.claim_next()
        if task is None:
            return None
        try:
            self.store.heartbeat(task["id"])
            current_profile = self.store.profile(task["profile"], task["actor"])
            if current_profile is None or not current_profile.get("enabled"):
                raise AgentTaskError("The selected agent is no longer available.")
            profile = current_profile
            if current_profile.get("custom"):
                profile_store = getattr(self.store, "profile_store", None)
                pinned = (
                    profile_store.get_version(
                        task["profile"], task["actor"], task.get("profile_version", 0)
                    )
                    if profile_store is not None and hasattr(profile_store, "get_version")
                    else None
                )
                if pinned is None:
                    raise AgentTaskError(
                        "definition_version_unavailable: the task's custom-agent version "
                        "could not be loaded."
                    )
                profile = pinned
            scopes = set(profile["scopes"])
            mode = (
                self.preference_store.get_preference(
                    task["actor"], "intelligence_choice", ""
                )
                if self.preference_store is not None else ""
            )
            binding = task.get("execution_binding") or {}
            if binding:
                bound_mode = str(binding.get("mode", "")).strip().lower()
                if bound_mode in {"local", "provider"}:
                    mode = bound_mode
                if not hasattr(self.agent, "status"):
                    return self.store.transition(
                        task["id"], task["actor"], "blocked",
                        error="execution_binding_unavailable: the selected model runtime is unavailable. Replace the binding and retry.",
                    )
                try:
                    binding_status = self.agent.status(mode=mode)
                except (OSError, TypeError, ValueError):
                    binding_status = {}
                if not binding_status.get("configured"):
                    return self.store.transition(
                        task["id"], task["actor"], "blocked",
                        error="execution_binding_unavailable: the selected model binding is unavailable. Choose a replacement model and rebind this run.",
                    )
                bound_provider = str(binding.get("provider", "")).strip()
                bound_model = str(binding.get("model", "")).strip()
                if bound_provider and str(binding_status.get("provider", "")).strip() != bound_provider:
                    return self.store.transition(
                        task["id"], task["actor"], "blocked",
                        error="execution_binding_changed: the selected provider changed. Replace the binding before retrying.",
                    )
                if bound_model and str(binding_status.get("model", "")).strip() not in {"", bound_model, "Auto-detect loaded model"}:
                    return self.store.transition(
                        task["id"], task["actor"], "blocked",
                        error="execution_binding_changed: the selected model changed. Replace the binding before retrying.",
                    )
            # A user-triggered escalation drives the ENTIRE run on the capable GPU
            # model. It carries no execution binding (the block above is skipped),
            # and this override wins for every agent connection below. Default
            # runs (model_tier "") are untouched: mode stays exactly what it was.
            model_tier = "capable" if str(task.get("model_tier") or "").strip().lower() == "capable" else ""
            if model_tier == "capable":
                mode = "capable"
            if profile.get("custom"):
                if not hasattr(self.agent, "status"):
                    raise AgentTaskError(
                        "model_configuration: custom agents require a configured AI model runtime."
                    )
                status = self.agent.status(mode)
                if not status.get("configured"):
                    raise AgentTaskError(
                        "model_configuration: custom agents require a selected local or connected AI model."
                    )
                if not status.get("endpoint_safe", False):
                    raise AgentTaskError(
                        "model_endpoint_denied: the selected AI endpoint is outside the allowed runtime policy."
                    )
            facts = {}
            tool_failures = []
            administrator = self._is_administrator(task["actor"])
            context_policy = _task_context_policy(task["description"])
            web_access = profile.get(
                "web_access", {"enabled": False, "allowed_domains": []}
            )
            domains = list(web_access.get("allowed_domains") or [])
            # Non-research (appliance) facts are gathered up front for BOTH the
            # tool loop and the one-shot path; research is NOT pre-fetched here.
            # One run, one hardware sample (one CPU temperature, quoted once).
            with one_reading_per_answer():
                for definition in self.tools.catalog():
                    if context_policy == "supplied-only":
                        continue
                    if definition["scope"] == "research:read":
                        continue
                    if definition["scope"] not in scopes:
                        continue
                    try:
                        facts[definition["name"]] = self.tools.run(
                            definition["name"], {}, actor=task["actor"],
                            administrator=administrator,
                            granted_scopes=scopes, web_access=web_access,
                        )["result"]
                    except AssistantToolError as error:
                        failure = {
                            "stage": "tool_execution",
                            "tool": definition["name"],
                            "error": str(error)[:300],
                        }
                        tool_failures.append(failure)
                        facts[definition["name"]] = {"unavailable": failure}
                    self.store.heartbeat(task["id"])
            # Research-acquisition values the audit and #247w block consume: the
            # loop path fills them from the model's calls, the fallback path from
            # acquire_web_research, a non-research run leaves the defaults.
            research_sources = []
            search = {}
            candidates = []
            relevant = []
            search_query_reason = ""
            search_attempts = []
            relevance_reason = ""
            granted_connectors = profile.get("connectors", [])
            connectors = [] if context_policy != "granted" else granted_connectors
            if connectors:
                if self.connector_runtime is None:
                    tool_failures.append({
                        "stage": "connector_runtime", "tool": "connector",
                        "error": "connector_runtime_unavailable: connector execution is not configured.",
                    })
                elif not hasattr(self.agent, "custom_agent_connector_plan"):
                    tool_failures.append({
                        "stage": "connector_planning", "tool": "connector",
                        "error": "connector_planning_unavailable: selected model runtime cannot plan connector calls.",
                    })
                else:
                    try:
                        calls = self.agent.custom_agent_connector_plan(
                            task["description"], profile, mode=mode,
                        )
                    except (OSError, ValueError, KeyError) as error:
                        calls = []
                        tool_failures.append({
                            "stage": "connector_planning", "tool": "connector",
                            "error": "connector_planning_failed: " + str(error)[:300],
                        })
                    connector_results = []
                    for call in calls[:5]:
                        try:
                            connector_results.append({
                                "connector_id": call["connector_id"],
                                "operation_id": call["operation_id"],
                                "outcome": self.connector_runtime.execute(
                                    profile, call["connector_id"], call["operation_id"],
                                    call["arguments"], actor=task["actor"], task_id=task["id"],
                                ),
                            })
                        except ValueError as error:
                            tool_failures.append({
                                "stage": "connector_execution", "tool": "connector",
                                "connector_id": str(call.get("connector_id", ""))[:48],
                                "operation_id": str(call.get("operation_id", ""))[:48],
                                "error": str(error)[:300],
                            })
                        self.store.heartbeat(task["id"])
                    if connector_results:
                        facts["connector.results"] = connector_results
            app_grants = []
            app_results = []
            app_write_proposals = []
            if profile.get("custom") and context_policy == "granted":
                approval_context = task.get("approval_context") or {}
                app_grants = approval_context.get("app_grants", [])
                if not isinstance(app_grants, list):
                    app_grants = []
                if approval_context.get("app_grants_status") == "unavailable":
                    tool_failures.append({
                        "stage": "app_capability", "tool": "app",
                        "error": "The pinned app-grant context was unavailable; app use is disabled.",
                    })
                elif app_grants:
                    if self.app_capability_broker is None:
                        tool_failures.append({
                            "stage": "app_capability", "tool": "app",
                            "error": "The app capability broker is unavailable; app use is disabled.",
                        })
                    elif not hasattr(self.agent, "custom_agent_app_plan"):
                        tool_failures.append({
                            "stage": "app_capability", "tool": "app",
                            "error": "The selected model runtime cannot plan app calls.",
                        })
                    else:
                        try:
                            calls = self.agent.custom_agent_app_plan(
                                task["description"], profile, app_grants, mode=mode
                            )
                        except Exception as error:
                            calls = []
                            tool_failures.append({
                                "stage": "app_capability", "tool": "app",
                                "error": _safe_capability_failure(error),
                            })
                        for call_index, call in enumerate(calls[:5] if isinstance(calls, list) else []):
                            if not isinstance(call, dict):
                                tool_failures.append({
                                    "stage": "app_capability", "tool": "app",
                                    "error": "The model returned an invalid app-call plan.",
                                })
                                continue
                            grant_id = str(call.get("grant_id", ""))[:160]
                            operation_id = str(call.get("operation_id", ""))[:128]
                            parameters = call.get("input")
                            grant, operation = _pinned_app_operation(app_grants, grant_id, operation_id)
                            if grant is None or operation is None:
                                tool_failures.append({
                                    "stage": "app_capability", "tool": "app",
                                    "grant_id": grant_id, "operation_id": operation_id,
                                    "error": "The model selected an operation outside the pinned grants.",
                                })
                                continue
                            if not isinstance(parameters, dict) or operation.get("mode") not in {"read", "write"}:
                                tool_failures.append({
                                    "stage": "app_capability", "tool": "app",
                                    "grant_id": grant_id, "operation_id": operation_id,
                                    "error": "The pinned app operation or input is invalid.",
                                })
                                continue
                            key = "task:{}:grant:{}:operation:{}:call:{}".format(
                                task["id"], grant_id, operation_id, call_index
                            )
                            try:
                                if operation["mode"] == "write":
                                    preview = self.app_capability_broker.preview(
                                        task["actor"], grant_id, operation_id, parameters,
                                        task_id=task["id"],
                                    )
                                    proposal = _bounded_capability_value(preview)
                                    if not isinstance(proposal, dict):
                                        proposal = {"preview": proposal}
                                    proposal.update({
                                        "status": "needs_approval",
                                        "requires_separate_approval": True,
                                        "transport_executed": False,
                                    })
                                    app_write_proposals.append(proposal)
                                else:
                                    invocation = self.app_capability_broker.invoke(
                                        task["actor"], grant_id, operation_id, parameters,
                                        task_id=task["id"], idempotency_key=key,
                                    )
                                    app_results.append({
                                        "grant_id": grant_id,
                                        "operation_id": operation_id,
                                        "status": str(invocation.get("status", "succeeded"))[:32],
                                        "result": _bounded_capability_value(invocation.get("result")),
                                        "provenance": {
                                            "audit_event_id": str((invocation.get("audit") or {}).get("event_id", ""))[:96],
                                            "manifest_digest": str(grant.get("manifest_digest", ""))[:128],
                                        },
                                    })
                            except Exception as error:
                                tool_failures.append({
                                    "stage": "app_capability", "tool": "app",
                                    "grant_id": grant_id, "operation_id": operation_id,
                                    "error": _safe_capability_failure(error),
                                })
                            self.store.heartbeat(task["id"])
                if app_results:
                    facts["app.results"] = app_results
                if app_write_proposals:
                    facts["app.write_proposals"] = app_write_proposals
            knowledge = (
                []
                if context_policy == "supplied-only"
                else retrieve_agent_knowledge(
                    self.knowledge_store, task["actor"],
                    profile, task["description"],
                )
            )
            if knowledge:
                facts["knowledge.sources"] = knowledge
            # Ask-about readings first, so a granted fact survives the context
            # budget as data rather than a truncated-away name.
            facts = order_facts_by_relevance(facts, task["description"])
            model_context = {
                    "facts": facts,
                    # The real date, stated in the prompt so an agent reads a past date in the task as past, not future.
                    "current_date": current_date_utc(),
                    "app_capabilities": app_grants,
                    "matched_skills": [
                        {"name": item["name"], "version": item["version"], "guidance": item["content"]}
                        for item in (self.skill_store.match(task["description"]) if self.skill_store else [])
                    ],
                    "policy": {
                        "durable_task": True,
                        "direct_mutations": False,
                        "typed_write_proposals": (
                            "knowledge:write" in profile.get("permissions", [])
                        ),
                        "memory_writes": False,
                        "credential_access": False,
                        "mutation_requires_separate_approval": True,
                    },
                    "agent_definition": {
                        "name": profile["name"],
                        "instructions": profile.get("instructions", ""),
                        "version": task.get("profile_version", 0),
                    },
                }
            # Ground the custom-agent ask (specialist path unchanged) in
            # actor-scoped lessons + a schema-derived output shape, so a small
            # model is not asked for structured output context-starved. A
            # narrowed context (context_policy != "granted") suppresses lessons.
            # Global curated memory is NOT fed (no per-actor scope; gated future
            # seam); knowledge stays in facts["knowledge.sources"], not doubled.
            # Delivery to the 700-char managed-local window is via the SYSTEM
            # prompt (render_output_contract), not this payload which truncates.
            if profile.get("custom"):
                grant_context = context_policy == "granted"
                model_context["working_context"] = assemble_working_context(
                    subject=task["description"],
                    actor=task["actor"],
                    surface="custom-agent",
                    learning_store=self.learning_store if grant_context else None,
                    purpose=profile.get("instructions") or profile.get("name", ""),
                )
            # LOOP-FIRST: an eligible web agent drives its own search and fetch
            # through the native tool loop, and the sources surfaced are the pages
            # IT fetched. Only when the model cannot drive the loop (no usable tool
            # call, exhausted budget, transport/shape error) does the run fall back
            # to the a83 pre-fetch + one-shot path - never worse than a83.
            tool_loop_audit = {"path": "prefetch_fallback", "invocations": [], "nudged": False}
            result = None
            loop_was_eligible = loop_eligible(profile, self.agent, context_policy, scopes, web_access)
            if loop_was_eligible:
                try:
                    result, loop_invocations, research_sources, nudged = drive_custom_agent_loop(
                        self.agent, self.tools, task["description"], profile,
                        model_context, mode, actor=task["actor"],
                        administrator=administrator, granted_scopes=scopes,
                        web_access=web_access, capable=(model_tier == "capable"),
                    )
                    tool_loop_audit = {
                        "path": "native_tool_loop", "invocations": loop_invocations,
                        "nudged": nudged,
                    }
                    search_attempts = [
                        {"query": inv["arguments"].get("query", ""), "reason": "model-native-tools"}
                        for inv in loop_invocations
                        if inv.get("name") == "research.search" and isinstance(inv.get("arguments"), dict)
                    ]
                    search_query_reason = "model-native-tools: the model wrote its own web queries as native tool calls"
                    relevance_reason = "model-selected: the model chose which results to fetch"
                    note_inference_outcome(getattr(self.agent, "_connection", lambda _: {})(mode) or {}, ok=True)
                    LOGGER.info("agent task %s ran the native tool loop (%d call(s), nudged=%s)", task["id"], len(loop_invocations), nudged)
                except (AgentToolLoopFallback, OSError, ValueError, KeyError, TypeError, AttributeError) as error:
                    # One concise line naming WHY an eligible loop fell back - no
                    # traceback; the reason string is enough for the journal.
                    LOGGER.warning("agent task %s tool loop fell back to the pre-fetch path: %s", task["id"], error)
                    tool_loop_audit = {
                        "path": "prefetch_fallback", "invocations": [],
                        "nudged": getattr(error, "nudged", False),
                    }
                    result = None
            if result is None:
                if context_policy == "granted" and "research:read" in scopes:
                    acquisition = acquire_web_research(
                        self.tools, lambda: self.store.heartbeat(task["id"]),
                        agent=self.agent, mode=mode,
                        task_description=task["description"], scopes=scopes,
                        administrator=administrator, actor=task["actor"],
                        web_access=web_access, domains=domains,
                    )
                    facts.update(acquisition.facts)
                    tool_failures.extend(acquisition.tool_failures)
                    research_sources = acquisition.research_sources
                    search = acquisition.search
                    candidates = acquisition.candidates
                    relevant = acquisition.relevant
                    search_query_reason = acquisition.query_shaping
                    search_attempts = acquisition.search_attempts
                    relevance_reason = acquisition.relevance
                try:
                    if profile.get("custom"):
                        if not hasattr(self.agent, "custom_agent"):
                            raise ValueError(
                                "The configured runtime does not support user-defined agents."
                            )
                        result = self.agent.custom_agent(
                            task["description"], profile, context=model_context, mode=mode,
                        )
                    else:
                        result = self.agent.specialist(
                            task["profile"], task["description"],
                            context=model_context, mode=mode,
                        )
                except (OSError, ValueError, KeyError) as error:
                    if profile.get("custom"):
                        connection = getattr(self.agent, "_connection", lambda _: {})(mode) or {}
                        # Without this the only record of a failed run was a
                        # sanitised sentence on screen: no traceback, no endpoint,
                        # nothing in the journal for an operator to work from.
                        LOGGER.warning(
                            "custom agent run %s failed against %s: %s",
                            task["id"], connection.get("base_url", "unknown endpoint"),
                            type(error).__name__, exc_info=error,
                        )
                        message = custom_agent_failure(error, connection)
                        # Feed the real outcome back into readiness so the status
                        # badge stops advertising a model that never answers.
                        note_inference_outcome(connection, ok=False, detail=message)
                        raise AgentTaskError(message) from error
                    raise
                if profile.get("custom"):
                    note_inference_outcome(
                        getattr(self.agent, "_connection", lambda _: {})(mode) or {}, ok=True,
                    )
                    LOGGER.info(
                        "agent task %s used the pre-fetch one-shot path", task["id"]
                    )
            if profile.get("custom") and (
                result.get("fallback_used") or result.get("source") != "custom-agent-model"
            ):
                raise AgentTaskError(
                    "The selected model did not return a usable custom-agent result. "
                    "Retry once, shorten the task, or choose a stronger model."
                )
            # VD-100 / #96 phase-1b: an installed agent's model may append
            # proposed acting operations to its raw result. Capture them before
            # normalization (which drops unknown keys). The sanitizer strips any
            # result-carried grant (#34); the pinned-envelope gate below decides
            # which survive. Only custom agents can hold a pinned envelope.
            proposed_operations = (
                sanitize_proposed_operations(result)
                if profile.get("custom") else []
            )
            result = _structured_result(result)
            # #247w / LESSONS 8: the surfaced "sources it used" must be the run's
            # real, relevance-filtered candidate set - never a URL the model
            # invented. The gate is the GRANT, not search success: a web-enabled
            # agent whose search backend was DOWN still produced no evidence, so
            # its sources must be the true (empty) set and its fabricated
            # caddyserver.com-style citation must be dropped exactly as when the
            # search ran and returned junk - the down dependency is named
            # separately by web_access_status below. A web-disabled agent keeps
            # whatever sources its own non-web facts produced.
            searched = (
                isinstance(facts.get("research.search"), dict)
                and "unavailable" not in facts["research.search"]
            )
            fetched_any = "research.fetch" in facts
            if web_access.get("enabled") or research_sources:
                result["sources"] = research_sources
            if app_results:
                result["app_results"] = app_results
            if app_write_proposals:
                result["app_write_proposals"] = app_write_proposals
            # #34 / VD-100 #96: surface an installed agent's proposed operations
            # so the operator can approve them through the per-operation route -
            # but ONLY those inside the task's SERVER-OWNED pinned envelope.
            # resolve_agent_operation double-gates each against
            # INSTALLED_AGENT_OPERATIONS AND the grant snapshot pinned into
            # approval_context at task creation (read here, never from the
            # result), then scopes it to inventory this run actually read -
            # exactly the checks the approve route re-runs at approval time. A
            # result-carried grant is already stripped; an out-of-envelope or
            # unpinned op is dropped; an agent with no pinned grant surfaces
            # nothing (fail-closed inert). NEVER derive the envelope from
            # proposed_operations - deriving authority from the model is the #34
            # failure itself.
            if proposed_operations:
                live_projects = managed_projects(facts.get("workloads.inventory"))
                surfaced = []
                for entry in proposed_operations:
                    try:
                        resolve_agent_operation(entry, task, live_projects)
                    except ActingProposalError:
                        continue
                    surfaced.append(dict(entry))
                if surfaced:
                    result["proposed_operations"] = surfaced
            result["capability_audit"] = {
                "agent_id": task["profile"],
                "definition_version": task.get("profile_version", 0),
                "scopes": sorted(scopes),
                # The same scope list returns different rows depending on the
                # owner's role at run time, so the scopes alone do not describe
                # what this run was allowed to see.
                "actor_role": "administrator" if administrator else "operator",
                "web_access": {
                    "enabled": web_access.get("enabled") is True,
                    "allowed_domains": domains,
                    # Naming this "search only" was the whole defect: the
                    # agent could collect links and never read one.
                    "mode": "allowlisted_fetch" if domains else (
                        "search_and_result_fetch" if web_access.get("enabled") else "disabled"
                    ),
                    # #247w/#247y: how the (final) search query was shaped and how
                    # its results were filtered for relevance, recorded so neither
                    # is a silent edit (LESSONS 5/9). `search_attempts` lists every
                    # query actually run - the model-formulated one and, if the
                    # first read only junk, the model-reformulated retry - so no
                    # query the run issued is invisible.
                    "query_shaping": search_query_reason,
                    "search_attempts": search_attempts,
                    "relevance": relevance_reason,
                },
                "tool_failures": tool_failures,
                # Synthesis path (native tool loop vs pre-fetch fallback) and the model's own tool calls - bounded, secrets redacted.
                "tool_calling": tool_loop_audit or {"path": "prefetch_fallback", "invocations": []},
                "user_context_policy": context_policy,
                # Which model tier actually ran: the default NPU deployment-agent
                # model, or the GPU ai-chat model a user escalated onto.
                "model_tier": "gpu/ai-chat" if model_tier == "capable" else "npu/deployment-agent",
                "direct_network": False,
                "shell": False,
                "secret_access": False,
                "execution_binding": {
                    "mode": mode,
                    "provider": str((binding or {}).get("provider") or "")[:120],
                    "model": str((binding or {}).get("model") or "")[:160],
                    "status": "bound" if binding else "active",
                },
                "connector_grants": [
                    {
                        "id": item["id"], "origin": item["base_origin"],
                        "operations": [operation["id"] for operation in item["operations"]],
                    }
                    for item in granted_connectors
                ],
            }
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
                result, profile, task["id"],
                self.knowledge_store, task["actor"],
            )
            # LESSONS #4/#11 (#212): the search failure above is already in
            # `tool_failures`, but the run answers past it, so a granted agent
            # reads as "no access" when the search backend is simply down. Name
            # the down dependency so the honest status reaches the answer, not
            # only the capability audit.
            web_status = web_access_status(
                web_access, search, candidates=candidates,
                fetched_any="research.fetch" in facts,
                fetch_failed=any(
                    failure.get("tool") == "research.fetch"
                    for failure in tool_failures
                ),
            )
            if web_status:
                result["web_access_status"] = web_status
                warnings = list(result.get("warnings") or [])
                if web_status["detail"] not in warnings:
                    result["warnings"] = [web_status["detail"], *warnings]
            # A run that finished without producing anything is recorded as
            # having produced nothing. The state stays "completed" - it ran, it
            # did not crash - but the outcome travels with the result so no
            # surface can render "Finished / 100%" over an empty hand.
            result.update(task_outcome(result))
            # When the empty hand is a down web dependency, say so as the reason
            # rather than the generic "produced nothing": the owner can act on
            # "the search service is unavailable", not on "no findings".
            if web_status and result.get("outcome") == OUTCOME_NO_RESULT:
                result["outcome_reason"] = web_status["detail"]
            # #247w + owner requirement (2026-08-21): an async run that could not
            # verify anything from public sources must ASK, not guess. When a
            # granted search ran and nothing was read - the results were junk or
            # empty, not a down dependency (which web_status already owns) - put
            # concrete, grounded questions back to the user instead of a confident
            # training-knowledge answer (LESSONS 8/11). A model that recognised an
            # ambiguous task may already have asked; keep those and lead with the
            # run's own deterministic questions, which are the guarantee.
            # The run has no evidence to stand on when a granted search ran, read
            # nothing (junk or empty, not a down dependency), AND the model had no
            # local facts to answer from. That is the #247w hallucination case:
            # only there does the runtime derive its own questions, and only there
            # does it demote the model's confident-but-unsourced output.
            unverified = (
                searched and not fetched_any and not web_status
                and not _has_local_substance(facts)
            )
            questions = list(result.get("clarifications") or [])
            if unverified:
                questions = clarifying_questions(
                    task["description"], candidates=candidates, relevant=relevant,
                ) + questions
            seen_questions: set[str] = set()
            merged_questions: list[str] = []
            for question in questions:
                text = " ".join(str(question).split())[:400]
                if text and text.lower() not in seen_questions:
                    seen_questions.add(text.lower())
                    merged_questions.append(text)
            if merged_questions:
                result["clarifications"] = merged_questions[:6]
                result["outcome"] = OUTCOME_NEEDS_INPUT
                if unverified:
                    # #247w finding 3: the model answered from training knowledge
                    # with no evidence. Demote that confident output so no view -
                    # not the run workspace, not the board's summary headline -
                    # presents the hallucination as the result. The questions are
                    # the content now; the real (empty) sources stand.
                    result["summary"] = (
                        "Could not verify this from public sources - needs more "
                        "detail (see the questions in this result)."
                    )
                    for field in ("findings", "recommendations", "next_actions"):
                        result[field] = []
                    result.pop("answer", None)
                    result["outcome_reason"] = (
                        "Could not verify this from public sources, so the run is "
                        "asking for more detail instead of presenting an "
                        "unverified answer."
                    )
                    note = (
                        "I could not confirm this from public sources for this "
                        "task, so I have not shown an unverified answer. See the "
                        "questions below and re-run with more detail."
                    )
                else:
                    # The model itself asked for clarification while it did hold
                    # evidence (fetched pages or local facts); keep its answer and
                    # surface the question alongside it.
                    result["outcome_reason"] = (
                        "This run needs more detail before it can answer "
                        "confidently; see the questions in the result."
                    )
                    note = (
                        "This run needs you to clarify before it can answer "
                        "confidently. See the questions below."
                    )
                warnings = list(result.get("warnings") or [])
                if note not in warnings:
                    result["warnings"] = [note, *warnings]
            elif not result.get("clarifications"):
                # A run that carries no questions must not ship an empty field the
                # UI would render as a stray "Questions" heading.
                result.pop("clarifications", None)
            # Escalation-available signal: an underperforming run (degraded, no
            # result / needs input, or an eligible loop that fell back so the
            # model never drove tools) can be re-run by the user on the capable
            # GPU model. The GPU lease is probed only when the run underperformed,
            # so a clean native-tool-loop delivery's path is byte-for-byte the same.
            fell_back = loop_was_eligible and (tool_loop_audit or {}).get("path") == "prefetch_fallback"
            if model_tier != "capable" and (
                result.get("degraded")
                or result.get("outcome") in (OUTCOME_NO_RESULT, OUTCOME_NEEDS_INPUT)
                or fell_back
            ):
                offer = escalation_offer(capable_model_name(self.agent), degraded=True)
                if offer:
                    result["escalation_available"] = offer
            self.store.add_artifact(
                task["id"], task["actor"], (
                    "custom-agent-result.json" if profile.get("custom")
                    else "specialist-result.json"
                ),
                "application/json", result,
            )
            return self.store.transition(
                task["id"], task["actor"], "completed", result=result
            )
        except Exception as error:
            message = " ".join(str(error).split())[:400]
            safe_error = (
                message if isinstance(error, AgentTaskError)
                else "The agent could not complete this task. Retry once or review the selected model."
            )
            # A failure is the clearest case for offering the capable GPU model -
            # the NPU one did not complete the task at all - unless this run was
            # already on the capable tier or no GPU model is available.
            failure_result = None
            try:
                offer = escalation_offer(
                    capable_model_name(self.agent),
                    model_tier=str(task.get("model_tier") or ""), failed=True,
                )
                failure_result = {"escalation_available": offer} if offer else None
            except Exception:
                failure_result = None
            return self.store.transition(
                task["id"], task["actor"], "failed", error=safe_error,
                result=failure_result,
            )
