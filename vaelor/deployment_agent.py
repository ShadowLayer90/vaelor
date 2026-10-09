"""A constrained planning agent for Docker and local-model deployments."""

from __future__ import annotations

import json
import urllib.error
import urllib.request
from typing import Any, Dict, Iterable, Optional

from .agent_prompts import (
    ASSISTANT_PROMPT,
    LOCAL_ASSISTANT_PROMPT,
    LOCAL_PLANNER_PROMPT,
    SYSTEM_PROMPT,
)
from .assistant_accelerator_answers import npu_readings_answer
from .assistant_builtin_answer import builtin_reply
from .assistant_cluster_answers import cluster_answer, workers_named
from .assistant_serving_answers import SERVING_SOURCE, serving_answer
from .assistant_log_answers import LOG_WORDS
from .phrase_match import mentions
from .assistant_fault_answers import (
    READING_BACKED_SOURCES,
    accelerator_presence_answer,
    accelerator_readings_answer,
    accelerator_slowness_answer,
    health_alert_answer,
    names_other_component_subject,
)
from .assistant_answer_scope import scoped_answer
from .assistant_local_answer import first_choice_content, local_answer, local_turn
from .assistant_recovery import recovery_answer
from .assistant_answer_presentation import (
    connected_model_failure_answer,
    with_model_failure_stated,
    general_knowledge_model_failure,
    is_appliance_question,
    is_general_knowledge_question,
    normalize_model_answer,
    with_performance,
    out_of_scope_after_model_failure,
)
from .assistant_intents import knowledge_redirect, world_followup
from .assistant_acting_wiring import acting_answer, acting_decline, acting_proposal
from .assistant_request_policy import assistant_refusal, specialist_refusal
from .assistant_scope_guard import guarded_answer
from .assistant_memory_grounding import grounded_memory_answer
from .assistant_policy import deployment_capability_answer, is_deployment_request
from .agent_failure_messages import plan_failure_warning
from .agent_result_shape import validate_specialist
from .custom_agent_model import execute_custom_agent, plan_connector_calls
from .deployment_plans import AGENT_NAME, fallback_plan, validate_plan
from .deployment_plan_router import policy_plan
from .inference_client import (
    MAX_INFERENCE_SECONDS,
    allowed_inference_endpoint as _allowed_inference_endpoint,
    chat_completion as _chat_completion,
    inference_timeout as _inference_timeout,
    parse_model_object as _parse_model_object,
)
from .assistant_machine_brief import brief_system_prompt
from .inference_tuning import offered_models
from .job_secrets import contains_secret
from .provider_runtime import (
    assistant_budget,
    assistant_context,
    answer_timeout,
    generation_parameters,
    is_complete_builtin_answer,
    is_grounded_live_answer,
    managed_local_connection,
    model_capability,
    provider_context,
    provider_system_prompt,
    provider_user_content,
)
from .model_calibration import ASSISTANT_ANSWER_SCHEMA, PLANNER_PLAN_SCHEMA
from .model_connection import resolve_model_connection
from .model_profiles import (
    reasoning_headroom_tokens,
    with_structured_response_format,
)
from .specialist_baseline import baseline_review
from .specialist_model import specialist_review
from .runtime_paths import env_value

MAX_MESSAGE_LENGTH = 4000
#: What the status says when no model is pinned: the server's first offered
#: model is used. A placeholder, never a model name (review round 3).
AUTO_DETECT_MODEL = "Auto-detect loaded model"


class DeploymentAgent:
    """Produce validated plans; execution always requires a separate approval."""

    # This runtime can drive the native tool-calling loop; the runner reads this
    # flag to prefer it for eligible web agents (agent_tool_loop).
    supports_tool_loop = True

    def __init__(
        self,
        base_url: Optional[str] = None,
        model: Optional[str] = None,
        timeout_seconds: int = 20,
        credential_broker=None,
    ):
        self.base_url = (
            base_url
            if base_url is not None
            else env_value("VAELOR_AGENT_BASE_URL", "PM_AGENT_BASE_URL", "")
        ).rstrip("/")
        self.model = model or env_value(
            "VAELOR_AGENT_MODEL", "PM_AGENT_MODEL", "local-model")
        self.api_key = env_value("VAELOR_AGENT_API_KEY", "PM_AGENT_API_KEY", "")
        self.credential_broker = credential_broker
        # Clamped to 60 here, this silently undid any larger budget the
        # caller asked for, so a big connected model could never finish.
        self.timeout_seconds = max(1, min(timeout_seconds, MAX_INFERENCE_SECONDS))

    def connection(self, mode: str = "") -> Optional[Dict[str, str]]:
        return resolve_model_connection(
            self.credential_broker, mode=mode, base_url=self.base_url,
            model=self.model, api_key=self.api_key,
        )

    def _connection(self, mode: str = "") -> Optional[Dict[str, str]]:
        """Compatibility wrapper for internal callers pending staged migration."""
        return self.connection(mode)

    def status(self, mode: str = "") -> Dict[str, Any]:
        connection = self._connection(mode)
        configured = connection is not None
        return {
            "name": AGENT_NAME,
            "configured": configured,
            "provider": connection.get("label", "OpenAI-compatible server") if connection else "built-in-planner",
            "model": (
                connection.get("model") or AUTO_DETECT_MODEL
                if connection else None
            ),
            "provider_type": connection.get("provider") if connection else "built-in",
            "effective_mode": "connected" if connection else "basic",
            "capability": model_capability(connection),
            "endpoint_safe": _allowed_inference_endpoint(connection) if connection else True,
            "approval_required": True,
            "tools": [
                "inspect workload capabilities",
                "draft Compose validation jobs",
                "draft model inspection jobs",
                "draft agent runtime deployments",
            ],
        }

    def plan(
        self, message: str, context: Optional[Dict[str, Any]] = None, mode: str = ""
    ) -> Dict[str, Any]:
        clean_message = str(message).strip()
        if not clean_message:
            raise ValueError("Describe what you want to deploy.")
        if len(clean_message) > MAX_MESSAGE_LENGTH:
            raise ValueError("Agent messages are limited to 4,000 characters.")

        policy_baseline = policy_plan(clean_message)
        # Reviewed, actionable jobs stay entirely deterministic. An unreviewed
        # app has no job to execute, so a connected model may enrich its
        # compatibility review while the policy baseline remains authoritative.
        if (
            policy_baseline is not None
            and policy_baseline.get("proposed_job") is not None
        ):
            return policy_baseline

        connection = self._connection(mode)
        if connection and _allowed_inference_endpoint(connection):
            try:
                result = self._model_plan(clean_message, context or {}, connection)
                validated = self._validate_plan(result, source="local-model")
                if policy_baseline is not None:
                    validated = self._merge_policy_review(
                        validated, policy_baseline
                    )
                return validated
            except (OSError, ValueError, KeyError, IndexError, urllib.error.URLError) as error:
                fallback = policy_baseline or self._fallback_plan(clean_message)
                fallback["warnings"] = [
                    plan_failure_warning(error, connection, clean_message),
                    *fallback.get("warnings", []),
                ][:8]
                fallback["fallback_used"] = True
                return fallback
        return self._fallback_plan(clean_message)

    @staticmethod
    def _merge_policy_review(
        model_plan: Dict[str, Any], policy_plan_result: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Keep policy conclusions while retaining safe model review details."""
        merged = dict(model_plan)
        merged["summary"] = policy_plan_result["summary"]
        merged["rationale"] = policy_plan_result["rationale"]
        merged["warnings"] = list(dict.fromkeys([
            *policy_plan_result.get("warnings", []),
            *model_plan.get("warnings", []),
        ]))[:8]
        merged["checklist"] = list(dict.fromkeys([
            *policy_plan_result.get("checklist", []),
            *model_plan.get("checklist", []),
        ]))[:8]
        merged["proposed_job"] = None
        merged["approval_required"] = False
        # Application intents are minted by deterministic policy, never by the
        # connected model. Preserve that server-owned handoff for research.
        merged["application_intent"] = policy_plan_result.get("application_intent")
        merged["policy_preflight"] = True
        return merged

    def specialist(
        self, profile: str, task: str, context: Optional[Dict[str, Any]] = None,
        mode: str = "",
    ) -> Dict[str, Any]:
        clean_task = str(task).strip()
        if not clean_task or len(clean_task) > MAX_MESSAGE_LENGTH:
            raise ValueError("Specialist tasks must be between 1 and 4,000 characters.")
        refusal = specialist_refusal(clean_task)
        if refusal:
            return self._validate_specialist(refusal, profile, "policy-refusal")
        connection = self._connection(mode)
        if connection and _allowed_inference_endpoint(connection):
            try:
                model_result = self._validate_specialist(
                    specialist_review(
                        connection, profile, clean_task, context,
                        self.timeout_seconds,
                    ),
                    profile, "local-model",
                )
                baseline = self._fallback_specialist(
                    profile, context or {}, clean_task
                )
                model_result["summary"] = baseline["summary"]
                # Findings stay grounded: every line of the baseline restates a
                # value that arrived in `facts`, and a model finding merged in
                # here is a model claim wearing evidence's clothes. What changed
                # is that the baseline now answers the request it was given, so
                # keeping it no longer means dropping the user's symptoms.
                model_result["findings"] = baseline["findings"]
                model_result["recommendations"] = list(dict.fromkeys([
                    *baseline["recommendations"],
                    *model_result["recommendations"],
                ]))[:8]
                model_result["next_actions"] = list(dict.fromkeys([
                    *baseline["next_actions"],
                    *model_result["next_actions"],
                ]))[:8]
                model_result["grounded_in_live_facts"] = True
                return model_result
            except (OSError, ValueError, KeyError, IndexError, urllib.error.URLError):
                fallback = self._fallback_specialist(
                    profile, context or {}, clean_task
                )
                fallback["findings"] = [
                    "The selected AI connection was unavailable, so this review used built-in read-only diagnostics.",
                    *fallback.get("findings", []),
                ][:8]
                fallback["fallback_used"] = True
                return fallback
        return self._fallback_specialist(profile, context or {}, clean_task)

    def custom_agent(
        self, task: str, definition: Dict[str, Any],
        context: Optional[Dict[str, Any]] = None, mode: str = "",
    ) -> Dict[str, Any]:
        """Run a true user-defined agent without specialist fallbacks."""
        clean_task = str(task).strip()
        if not clean_task or len(clean_task) > MAX_MESSAGE_LENGTH:
            raise ValueError("Custom-agent tasks must be between 1 and 4,000 characters.")
        connection = self._connection(mode)
        if not connection:
            raise ValueError("A selected AI model is required for custom agents.")
        if not _allowed_inference_endpoint(connection):
            raise ValueError("The selected AI endpoint is outside the allowed runtime policy.")
        result = execute_custom_agent(
            connection, clean_task, definition, context or {}, self.timeout_seconds,
        )
        return self._validate_specialist(
            result, str(definition.get("name", "Custom agent")), "custom-agent-model"
        )

    def custom_agent_app_plan(self, task: str, definition: Dict[str, Any], grants: list[Dict[str, Any]], mode: str = ""):
        from .agent_tasks import plan_custom_app_calls
        return plan_custom_app_calls(self, task, definition, grants, self.timeout_seconds, mode)

    def custom_agent_connector_plan(
        self, task: str, definition: Dict[str, Any], mode: str = "",
    ):
        connection = self._connection(mode)
        if not connection or not _allowed_inference_endpoint(connection):
            raise ValueError("A safe selected AI model is required for connector planning.")
        return plan_connector_calls(
            connection, task, definition, self.timeout_seconds,
        )

    def _fallback_specialist(
        self, profile: str, context: Dict[str, Any], task: str = ""
    ):
        """The built-in review, which now hears the request it was given."""
        return self._validate_specialist(
            baseline_review(profile, task, context or {}),
            profile,
            "built-in-specialist",
        )

    def answer(
        self, message: str, context: Optional[Dict[str, Any]] = None, mode: str = "",
        granted_scopes: Optional[Iterable[str]] = None,
    ):
        clean_message = str(message).strip()
        if not clean_message or len(clean_message) > MAX_MESSAGE_LENGTH:
            raise ValueError("Assistant questions must be between 1 and 4,000 characters.")
        refusal = assistant_refusal(clean_message, workers_named(clean_message, context))
        if refusal:
            # Review S4: a power or delete request is declined with the
            # control that does it, and recorded as not answered.
            return self._validate_answer(
                refusal, source="policy-refusal", question=clean_message,
                answered=bool(refusal.pop("answered", True)),
            )
        live_context = context or {}
        # The acting seam (VD-100 #96): if the operator holds workloads:act and
        # this turn's inventory names the app, carry an evidence-bound proposal.
        # It only ever PRODUCES a proposal - the operator's POST /api/v2/jobs is
        # the one approval, and this agent reaches no executor (LESSONS 14). It
        # is inert by default: `granted_scopes` without workloads:act yields
        # None, so no unscoped caller can slip a proposal in.
        acting = acting_proposal(
            clean_message,
            live_context.get("appliance", live_context).get("facts", {}),
            granted_scopes,
        )
        if acting is not None:
            return self._validate_answer(
                acting_answer(acting), source="assistant-acting",
                question=clean_message,
            )
        # Review S3: an acting request nothing proposed says nothing was done.
        declined = acting_decline(
            clean_message,
            live_context.get("appliance", live_context).get("facts", {}),
            granted_scopes,
        )
        if declined is not None:
            return self._validate_answer(
                declined, source="assistant-acting", answered=False,
                question=clean_message,
            )
        grounded = self._fallback_answer(clean_message, live_context)
        # An answer assembled entirely from readings taken on this machine is
        # the answer. It used to be computed and then thrown away, because the
        # gate that decides "is this grounded" is a vocabulary list that knew
        # no word for a fault and no word for the accelerator - so the two
        # questions this appliance is most often asked were handed to a model
        # that had neither reading in front of it.
        if grounded.get("source") in READING_BACKED_SOURCES:
            return grounded
        general_knowledge = is_general_knowledge_question(clean_message)
        # Current appliance facts outrank lexical matches in historical memory.
        # Otherwise a live CPU/fan question can be hijacked by an old checkpoint
        # that happens to mention the same hardware terms.
        if is_grounded_live_answer(clean_message, grounded) and not general_knowledge:
            return grounded
        connection = self._connection(mode)
        model_available = bool(connection and _allowed_inference_endpoint(connection))
        memory_answer = grounded_memory_answer(clean_message, live_context)
        if memory_answer is not None and not model_available:
            return self._validate_answer(
                memory_answer, source="stored-memory", question=clean_message
            )
        has_prior_turn = len(live_context.get("conversation", [])) > 1
        if (
            grounded.get("proposed_job")
            or grounded.get("application_intent")
            or grounded.get("source") == "built-in-capability"
            # (grounded-live already returned above, grounded unchanged: dead clause removed)
            or (
                is_complete_builtin_answer(grounded)
                and not has_prior_turn
                and not general_knowledge
            )
        ):
            return grounded
        # Nothing above could answer this from the appliance. If nothing on
        # this machine can evidence the question either, the model must not be
        # asked to invent one: a small model answers history and science
        # confidently and wrongly, and a wrong answer in the machine's own
        # voice is the same defect as a fabricated sensor reading. A matched
        # skill or a grounding memory is evidence, so those still answer.
        appliance = live_context.get("appliance", live_context)
        if memory_answer is None and not appliance.get("matched_skills"):
            redirect = knowledge_redirect(clean_message, appliance.get("ai_chat"))
            if redirect is not None:
                return self._validate_answer(
                    redirect, source="out-of-scope", answered=False,
                    question=clean_message,
                )
        if model_available:
            try:
                result = self._model_answer(clean_message, live_context, connection)
                validated = self._validate_answer(result, source="connected-model", question=clean_message)
                validated["performance"] = result.get("performance", {})
                validated["skills_sent"] = list(result.get("skills_sent") or [])
                return validated
            except (OSError, ValueError, KeyError, IndexError, urllib.error.URLError):
                if general_knowledge:
                    fallback = self._validate_answer(
                        general_knowledge_model_failure(),
                        source="connected-model-fallback", answered=False,
                        question=clean_message,
                    )
                    fallback["fallback_used"] = True
                    return fallback
                # Tomorrow's weather and last night's baseball are not
                # appliance facts, and reporting them purely as a model failure
                # sent people to debug an endpoint that was answering
                # correctly. The model did still fail, though, so the answer
                # says both: claiming scope is the only problem would state
                # something false about a real outage.
                if not is_appliance_question(clean_message):
                    scoped = self._validate_answer(
                        out_of_scope_after_model_failure(), source="out-of-scope",
                        answered=False, question=clean_message,
                    )
                    scoped["fallback_used"] = True
                    return scoped
                fallback = grounded
                stood_in = bool(fallback.get("answered", True))
                if not stood_in:
                    fallback["answer"] = connected_model_failure_answer()
                    # The model failed and there was no built-in answer to
                    # put in its place. Nothing was answered.
                    fallback["answered"] = False
                else:
                    # The reader is told *in the answer* that a built-in
                    # reading is standing in for a model that did not reply.
                    # See `MODEL_DID_NOT_ANSWER_PREFIX` for what this cost.
                    fallback["answer"] = with_model_failure_stated(
                        fallback.get("answer", "")
                    )
                # Review S11: only claim a built-in answer when one stood in.
                fallback["evidence"] = [{
                    "source": "assistant.fallback",
                    "summary": (
                        "The Assistant's model did not answer, so Vaelor answered "
                        "from this machine's own readings instead."
                        if stood_in else
                        "The Assistant's model did not answer, and no built-in "
                        "reading could answer this question either."
                    ),
                }, *fallback.get("evidence", [])][:10]
                fallback["fallback_used"] = True
                return fallback
        return grounded

    def _model_answer(
        self, message: str, context: Dict[str, Any], connection: Dict[str, str]
    ) -> Dict[str, Any]:
        model = connection.get("model", "")
        headers = {"Content-Type": "application/json", "Accept": "application/json"}
        if connection.get("api_key"):
            headers["Authorization"] = "Bearer {}".format(connection["api_key"])
        if not model:
            model_request = urllib.request.Request(
                "{}/models".format(connection["base_url"]), headers=headers
            )
            with urllib.request.urlopen(
                model_request, timeout=self.timeout_seconds
            ) as response:
                model_data = json.loads(response.read(1024 * 1024).decode("utf-8"))
            offered = offered_models(model_data)
            if not offered:
                # Reachable-and-empty (LM Studio with no model loaded) is a real
                # state; without this it raised an unhandled IndexError.
                raise ValueError(
                    "The model server is reachable but is not offering any "
                    "model, so there is nothing for Vaelor to send this "
                    "question to."
                )
            model = str(offered[0])
        managed = managed_local_connection(connection)
        system_content = provider_system_prompt(
            # The standing brief rides in the system message, not the JSON
            # context object (assistant_machine_brief.brief_system_prompt).
            brief_system_prompt(
                (
                    LOCAL_ASSISTANT_PROMPT
                    if connection.get("base_url", "").startswith(
                        ("http://127.0.0.1:", "http://[::1]:")
                    )
                    else ASSISTANT_PROMPT
                ),
                context,
            ),
            connection,
        )
        # VD: a managed-local model echoes a JSON prompt envelope (measured
        # 10/10 live) but answers natural language (9/9). See assistant_local_answer.
        # Review B5: memories, earlier turns and matched guidance travel with
        # the question, and only guidance actually sent may be claimed.
        if managed:
            user_content, skills_sent = local_turn(message, context, connection)
        else:
            trimmed = assistant_context(message, context, connection)
            skills_sent = [str(item.get("slug")) for item in (trimmed.get("guidance") or [])
                           if isinstance(item, dict) and item.get("slug")] if isinstance(trimmed, dict) else []
            user_content = provider_user_content({"question": message, "context": trimmed}, connection)
        request_body = {
            "model": model,
            "messages": [
                {"role": "system", "content": system_content},
                {"role": "user", "content": user_content},
            ],
            **generation_parameters(
                connection,
                max_tokens=reasoning_headroom_tokens(
                    connection, assistant_budget(connection, message)["max_tokens"]
                ),
                temperature=0.1,
            ),
            **(
                {} if managed
                else with_structured_response_format(
                    {}, connection, ASSISTANT_ANSWER_SCHEMA
                )
            ),
            **({"chat_template_kwargs": {"enable_thinking": False}} if managed else {}),
        }
        timeout = answer_timeout(connection, self.timeout_seconds)
        # Managed-local runs through local_answer, which retries once on a
        # degenerate (echoed or empty) reply before returning humanized text.
        if managed:
            result = normalize_model_answer(local_answer(request_body, headers, timeout, connection, _chat_completion))
        else:
            body = _chat_completion(connection, request_body, headers, timeout)
            result = with_performance(normalize_model_answer(_parse_model_object(first_choice_content(body))), body)
        if isinstance(result, dict):
            result["skills_sent"] = skills_sent
        return result

    @staticmethod
    def _validate_answer(
        result: Dict[str, Any], source: str, *, answered: bool = True,
        question: str = "",
    ) -> Dict[str, Any]:
        """Validate an answer's shape, and then its content against VD-042.

        Shape was all this checked. A refusal the model wrote itself never
        touches the routing gate, so it walked through here unexamined: a
        well-shaped dict carrying a sentence that declined a question about
        this machine. ``question`` is optional - a caller checking shape alone
        still can - but every path in :meth:`answer` supplies it. See
        :mod:`vaelor.assistant_scope_guard`.
        """
        if not isinstance(result, dict):
            raise ValueError("The assistant returned an invalid answer.")
        proposed = result.get("proposed_job")
        if proposed is not None:
            if (
                not isinstance(proposed, dict)
                or proposed.get("type") not in {
                    "compose.install", "compose.backup", "model.inspect", "agent.deploy",
                    # VD-100 / #96 phase-1b: the acting seam proposes one of these
                    # two reversible jobs. They are on the allowlist so an
                    # evidence-bound proposal can REACH the operator's approval;
                    # nothing here executes - the one executor runs it only after
                    # POST /api/v2/jobs (LESSONS 14).
                    "compose.restart", "compose.update",
                }
                or not isinstance(proposed.get("payload", {}), dict)
                or contains_secret(proposed.get("payload", {}))
            ):
                # Small local models occasionally put explanatory prose in this
                # field. Discard it rather than throwing away an otherwise useful
                # answer; no proposal can execute without passing this allowlist.
                proposed = None
        evidence = []
        for item in result.get("evidence", [])[:10]:
            if isinstance(item, dict):
                evidence.append({
                    "source": str(item.get("source", "appliance"))[:100],
                    "summary": str(item.get("summary", ""))[:600],
                })
        return guarded_answer(question, {
            "answer": str(result.get("answer", "I could not prepare an answer."))[:5000],
            "evidence": evidence,
            "suggested_actions": [
                str(item)[:300] for item in result.get("suggested_actions", [])[:8]
            ],
            "proposed_job": proposed,
            "approval_required": proposed is not None,
            "source": source,
            # Whether this reply actually answered the question. The audit
            # trail recorded every assistant turn as SUCCESS, including the
            # ones that told the user their model had not answered - so a card
            # promising "authenticated actions" listed six successes of which
            # two had visibly failed in front of the person reading it.
            "answered": bool(answered),
        })

    def _fallback_answer(self, message: str, context: Dict[str, Any]):
        facts = context.get("appliance", context).get("facts", {})

        recovery = recovery_answer(message, facts)
        if recovery is not None:
            return self._validate_answer(
                recovery, source="built-in-recovery", question=message
            )

        # A past-moment or identity question must not be answered with a
        # present reading about something else (#159). #144's rules for this
        # live in the prompts, which only govern a model-written answer.
        scoped = scoped_answer(message, facts, context)
        if scoped is not None:
            # The answer carries its own source and `answered`: a refusal must
            # not be audited as a success, nor claim a live reading it lacks.
            return self._validate_answer(
                scoped, source=scoped.pop("source", "built-in-live-data"),
                answered=bool(scoped.pop("answered", True)), question=message,
            )

        # Faults and accelerator slowness are answered from readings first: both
        # reached the model on the Z2 and came back invented (see those modules).
        # VD-205 live check L1: which model AI Chat, the LLM Server and the
        # Assistant use is read, never handed to the model to guess.
        serving = serving_answer(message, facts)
        if serving is not None:
            return self._validate_answer(
                serving, source=SERVING_SOURCE, question=message,
                answered=bool(serving.pop("answered", True)),
            )
        # VD-205 item 4: cluster questions are answered from the cluster
        # digest, each machine from its own readings, before any branch that
        # reads only this controller's sensors.
        cluster = cluster_answer(message, facts, context.get("appliance", context))
        if cluster is not None:
            return self._validate_answer(
                cluster, source="built-in-cluster", question=message,
                answered=bool(cluster.pop("answered", True)),
            )
        # Review S7: a question about the logs is answered from the journal,
        # not from the health verdict its "any errors" also matches.
        fault = None if mentions(message, LOG_WORDS) else health_alert_answer(message, facts)
        if fault is not None:
            return self._validate_answer(
                fault, source="built-in-health", question=message,
                answered=bool(fault.pop("answered", True)),
            )

        slowness = accelerator_slowness_answer(message, facts)
        if slowness is not None:
            return self._validate_answer(
                slowness, source="built-in-accelerator", question=message
            )

        # An accelerator question about it alone keeps its dedicated answer
        # (readings before presence, so a readings word is not intercepted
        # name-only). But a compound question naming another live subject too -
        # "CPU temperature, memory usage, GPU utilization, disk usage" - must
        # not return here dropping every other reading; it folds into the
        # accumulation below instead.
        compound_reading = names_other_component_subject(message)
        readings = _merged(accelerator_readings_answer(message, facts),
                           npu_readings_answer(message, facts))
        if readings is not None and not compound_reading:
            return self._validate_answer(
                readings, source="built-in-accelerator", question=message
            )
        presence = None if readings is not None else accelerator_presence_answer(message, facts)
        if presence is not None and not compound_reading:
            return self._validate_answer(
                presence, source="built-in-accelerator", question=message
            )

        capability = deployment_capability_answer(message, facts)
        if capability is not None:
            return self._validate_answer(
                capability, source="built-in-capability", question=message
            )

        if is_deployment_request(message):
            plan = self._fallback_plan(message)
            capabilities = facts.get("workloads.capabilities", {})
            docker = capabilities.get("docker", {}) if isinstance(
                capabilities, dict
            ) else {}
            docker_note = (
                " Docker is ready on this appliance."
                if docker.get("installed")
                else " Docker readiness will be checked before anything runs."
            )
            answer = self._validate_answer({
                "answer": "{} {}{} Nothing has run yet.".format(
                    plan["summary"], plan["rationale"], docker_note
                ),
                "evidence": [{
                    "source": "workloads.capabilities",
                    "summary": "Live Docker, storage, port, and architecture checks are required before deployment.",
                }],
                "suggested_actions": plan["checklist"],
                "proposed_job": plan["proposed_job"],
            }, source="built-in-planner", question=message)
            # Unknown applications continue through the server-owned research
            # workflow.  Keep this deterministic intent out of model output,
            # but expose it so the UI can offer one clear next step.
            answer["application_intent"] = plan.get("application_intent")
            return answer

        # Every per-subject sentence lives in `assistant_builtin_answer`. The
        # accelerator is a live-reading subject like the rest: a compound
        # question named it beside CPU/memory/storage, so its reading is
        # appended there rather than returned alone above.
        reply = builtin_reply(
            message, facts, context.get("appliance", context),
            readings if readings is not None else presence,
        )
        lines, evidence, answered = reply["lines"], reply["evidence"], reply["answered"]
        # A compound question answered for its appliance half must not drop the
        # world half in silence (#205 item 3): point that part at AI Chat rather
        # than pretend it was not asked. Only ever fires beside a real answer.
        followup = world_followup(message) if answered else None
        if followup:
            lines.append(followup["note"])
        return self._validate_answer({
            "answer": "\n\n".join(lines),
            "evidence": evidence,
            "suggested_actions": [followup["action"]] if followup else [],
            "proposed_job": None,
        }, source="hardware-guided", answered=answered, question=message)

    _validate_specialist = staticmethod(validate_specialist)

    def _model_plan(
        self, message: str, context: Dict[str, Any], connection: Dict[str, str]
    ) -> Dict[str, Any]:
        model = connection.get("model", "")
        headers = {"Content-Type": "application/json", "Accept": "application/json"}
        if connection.get("api_key"):
            headers["Authorization"] = "Bearer {}".format(connection["api_key"])
        if not model:
            model_request = urllib.request.Request(
                "{}/models".format(connection["base_url"]),
                headers=headers,
            )
            with urllib.request.urlopen(
                model_request, timeout=self.timeout_seconds
            ) as response:
                model_data = json.loads(response.read(1024 * 1024).decode("utf-8"))
            model = str(model_data["data"][0]["id"])
        request_body = with_structured_response_format({
            "model": model,
            "messages": [
                {
                    "role": "system",
                    "content": provider_system_prompt(
                        LOCAL_PLANNER_PROMPT
                        if managed_local_connection(connection)
                        else SYSTEM_PROMPT,
                        connection,
                    ),
                },
                {
                    "role": "user",
                    "content": provider_user_content(
                        {
                            "request": message,
                            "appliance": provider_context(
                                context,
                                max_chars=assistant_budget(connection, message)["context_chars"],
                            ),
                        },
                        connection,
                    ),
                },
            ],
            **generation_parameters(
                connection,
                max_tokens=reasoning_headroom_tokens(
                    connection, assistant_budget(connection, message)["max_tokens"]
                ),
                temperature=0.1,
            ),
        }, connection, PLANNER_PLAN_SCHEMA)
        body = _chat_completion(
            connection,
            request_body,
            headers,
            _inference_timeout(connection, self.timeout_seconds),
        )
        message_body = body["choices"][0]["message"]
        content = message_body.get("content", "")
        if not str(content or "").strip() and (message_body.get("reasoning") or message_body.get("reasoning_content")):
            raise ValueError(
                "The model's reasoning used the full output budget before "
                "producing a JSON plan."
            )
        return _parse_model_object(content)

    @staticmethod
    def _validate_plan(plan: Dict[str, Any], source: str) -> Dict[str, Any]:
        return validate_plan(plan, source)

    def _fallback_plan(self, message: str) -> Dict[str, Any]:
        return fallback_plan(message)


def _merged(*answers):
    """One reading-backed answer from the GPU's and the NPU's, or ``None``."""
    found = [item for item in answers if item is not None]
    if not found:
        return None
    if len(found) == 1:
        return found[0]
    merged = dict(found[0])
    merged["answer"] = " ".join(item["answer"] for item in found)
    merged["evidence"] = [entry for item in found for entry in item.get("evidence") or []]
    return merged
