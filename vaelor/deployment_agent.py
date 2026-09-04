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
from .answer_evidence import (
    add_evidence,
    describe_missing,
    first_sentence,
    present,
    sentence,
)
from .assistant_answer_topics import asks_about, capability_sentence
from .byte_units import describe_gb
from .assistant_fault_answers import (
    READING_BACKED_SOURCES,
    accelerator_presence_answer,
    accelerator_readings_answer,
    accelerator_slowness_answer,
    health_alert_answer,
    names_other_component_subject,
    overall_verdict_line,
)
from .assistant_answer_scope import scoped_answer
from .assistant_local_answer import first_choice_content, local_answer, local_user_content
from .assistant_recovery import recovery_answer
from .assistant_static_answers import static_answer
from .assistant_answer_presentation import (
    connected_model_failure_answer,
    with_model_failure_stated,
    general_knowledge_model_failure,
    is_appliance_question,
    is_general_knowledge_question,
    normalize_model_answer,
    out_of_scope_after_model_failure,
    managed_workload_summary,
    network_summary,
    storage_summary,
)
from .assistant_intents import knowledge_redirect, world_followup
from .assistant_acting_wiring import acting_answer, acting_proposal
from .assistant_request_policy import assistant_refusal, specialist_refusal
from .assistant_scope_guard import guarded_answer
from .assistant_hardware_answers import (
    case_fan_answer,
    cpu_temperature,
    display_line,
    lighting_line,
)
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
                connection.get("model") or "Auto-detect loaded model"
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
        refusal = assistant_refusal(clean_message)
        if refusal:
            return self._validate_answer(
                refusal, source="policy-refusal", question=clean_message
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
                return self._validate_answer(
                    result, source="connected-model", question=clean_message,
                )
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
                if "Connect a local model" in fallback.get("answer", ""):
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
                fallback["evidence"] = [{
                    "source": "assistant.fallback",
                    "summary": "The selected AI connection did not answer correctly, so Vaelor used built-in live-data intelligence.",
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
        user_content = (
            local_user_content(message, context, connection) if managed
            else provider_user_content(
                {
                    "question": message,
                    "context": assistant_context(message, context, connection),
                },
                connection,
            )
        )
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
            return normalize_model_answer(local_answer(
                request_body, headers, timeout, connection, _chat_completion
            ))
        body = _chat_completion(connection, request_body, headers, timeout)
        return normalize_model_answer(_parse_model_object(first_choice_content(body)))

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
        lower = message.lower()
        evidence = []
        lines = []
        cooling = facts.get("cooling.status", {})
        telemetry = facts.get("system.telemetry", {})
        identity = facts.get("system.identity", {})
        display = facts.get("display.status", {})
        lighting = facts.get("lighting.status", {})
        updates = facts.get("updates.status", {})
        services = facts.get("services.status", [])
        storage = facts.get("storage.status", {})
        network = facts.get("network.status", {})
        # The probed answer for what this machine has. Where prose written from
        # a fact object disagrees with it, this wins: the fact object may carry
        # defaults, this carries a discovery result and its reason.
        capabilities_map = facts.get("machine.capabilities", {}) or (
            facts.get("system.identity", {}) or {}
        ).get("capabilities", {})
        workloads = facts.get("workloads.inventory", {})
        jobs = facts.get("jobs.recent", [])
        specialist_results = context.get("appliance", context).get(
            "specialist_results", []
        )

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
        fault = health_alert_answer(message, facts)
        if fault is not None:
            return self._validate_answer(
                fault, source="built-in-health", question=message
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
        readings = accelerator_readings_answer(message, facts)
        if readings is not None and not compound_reading:
            return self._validate_answer(
                readings, source="built-in-accelerator", question=message
            )
        presence = accelerator_presence_answer(message, facts)
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

        asks_health = asks_about("health-verdict", lower)
        # Resolved once, before any sentence is written. Every mention of the
        # CPU temperature in this reply is this number or it is absent; two
        # readings taken moments apart must never share a paragraph.
        reading = cpu_temperature(cooling, telemetry)
        temperature_stated = False
        if asks_health and cooling:
            lines.append(overall_verdict_line(facts, reading))

        if asks_about("cooling", lower) and cooling:
            cpu = cooling.get("cpu", {})
            case = cooling.get("case", {})
            # Every clause here is conditional on a reading existing. The old
            # version defaulted `rpm` to 0 and the cooling states to the word
            # "unknown", then wrote a sentence around them - so a machine whose
            # fan speed Vaelor could not read was told its fan was at 0 RPM,
            # and given an invented explanation of why.
            stated = []
            fan_speed = sentence(
                "The CPU fan is turning at {rpm} RPM", rpm=cpu.get("rpm")
            )
            if fan_speed:
                mode = sentence(" in {mode} mode", mode=cpu.get("mode"))
                stated.append(fan_speed + mode + ".")
            else:
                stated.append(describe_missing(
                    "the processor fan's speed",
                    str(cpu.get("reason") or ""),
                ))
            cooling_state = sentence(
                "The cooling state is {current} of {maximum}.",
                current=cpu.get("current_state"), maximum=cpu.get("max_state"),
            )
            if cooling_state:
                stated.append(cooling_state)
            if reading is not None:
                stated.append("The CPU is {:.1f}°C.".format(float(reading)))
                temperature_stated = True
            # The causal story is only told where the policy that governs it was
            # read: asserting how cooling behaves on unseen hardware is invention.
            if (
                present(cpu.get("rpm")) and float(cpu.get("rpm") or 0) == 0
                and present(cpu.get("policy"))
                and reading is not None and float(reading) < 55
            ):
                stated.append(
                    "Zero RPM is expected here: this machine's cooling policy "
                    "starts the fan above {}.".format(cpu.get("policy"))
                )
            lines.append(" ".join(part for part in stated if part))
            case_line = case_fan_answer(case)
            if case_line:
                lines.append(case_line)
            add_evidence(
                evidence, "cooling.status", cpu,
                ("rpm", "mode", "current_state", "max_state", "policy"),
            ) or add_evidence(evidence, "cooling.status", cooling.get("cpu_temperature"))
        if asks_about("display", lower) and display:
            lines.append(display_line(display, capabilities_map))
            add_evidence(
                evidence, "display.status", display,
                ("hardware", "bus", "enabled", "rotation", "sleep_timeout"),
            )
        if asks_about("lighting", lower) and lighting:
            lines.append(lighting_line(lighting, capabilities_map))
            add_evidence(
                evidence, "lighting.status", lighting,
                ("led_count", "rgb_enable", "rgb_style", "rgb_brightness"),
            )
        if asks_about("updates", lower) and updates:
            lines.append(
                "{} operating-system updates are available. They are {} and have not been installed.".format(
                    updates.get("count", 0),
                    "downloaded and staged" if updates.get("staged") else "not staged",
                )
            )
            add_evidence(evidence, "updates.status", updates, ("count", "staged", "reboot_required"))
        if asks_about("services", lower) and services:
            unhealthy = [
                item.get("id", "service") for item in services
                if item.get("available") and item.get("active") != "active"
            ]
            lines.append(
                "All managed Vaelor services are active."
                if not unhealthy else "These services need attention: {}.".format(", ".join(unhealthy))
            )
            add_evidence(evidence, "services.status", services)
        if asks_about("cpu", lower) and telemetry:
            cpu_percent = telemetry.get("cpu_percent")
            details = []
            if cpu_percent is not None:
                details.append("{}% use".format(round(float(cpu_percent), 1)))
            # Saying the temperature again here is what produced two readings in
            # one paragraph. It is the same sensor, already reported above.
            if reading is not None and not temperature_stated:
                details.append("{:.1f}°C".format(reading))
            lines.append(
                "The host CPU is currently {}.".format(
                    " and ".join(details) if details else "reporting live telemetry"
                )
            )
            add_evidence(evidence, "system.telemetry", telemetry, ("cpu_percent", "cpu_temperature"))
        if asks_about("memory", lower) and telemetry:
            # `memory_total`/`memory_used` ride on every telemetry sample and
            # nothing read them, so "how much RAM" was answered with a percentage
            # of an unstated whole (#183, LESSONS 11). Whole decimal GB, the unit
            # `assistant_machine_brief` uses, so brief and answer cannot disagree.
            fitted = describe_gb(telemetry.get("memory_total"), 0)
            in_use = describe_gb(telemetry.get("memory_used"), 1)
            share = telemetry.get("memory_percent")
            used = "{}%{}".format(
                round(float(share), 1), " ({})".format(in_use) if in_use else "",
            ) if share is not None else ""
            stated = first_sentence(
                sentence(
                    "This machine has {fitted} of memory, and {used} of it is "
                    "in use.", fitted=fitted, used=used,
                ),
                sentence("This machine has {fitted} of memory.", fitted=fitted),
                sentence("Memory use is currently {used}.", used=used),
            )
            if stated:
                lines.append(
                    "{} RAM is the working space shared by the operating "
                    "system, apps, and local AI.".format(stated)
                )
                add_evidence(
                    evidence, "system.telemetry", telemetry,
                    ("memory_total", "memory_used", "memory_percent"),
                )
        if asks_about("storage", lower) and storage:
            lines.append(storage_summary(storage))
            add_evidence(evidence, "storage.status", storage)
        # `network_summary` says why the sentence carries the address itself
        # and not just the adapter name.
        if asks_about("network", lower) and network:
            lines.append(network_summary(network))
            add_evidence(evidence, "network.status", network, ("interfaces",))
        if asks_about("workloads", lower) and workloads:
            lines.append(managed_workload_summary(workloads))
            add_evidence(evidence, "workloads.inventory", workloads)
        if asks_about("jobs", lower) and jobs:
            current = jobs[0]
            lines.append(
                "The latest deployment job is {} and is {} ({}%).".format(
                    str(current.get("type", "workload")).replace(".", " "),
                    current.get("state", "unknown"),
                    current.get("progress", 0),
                )
            )
            add_evidence(evidence, "jobs.recent", jobs)
        # The accelerator is a live-reading subject like the rest: a compound
        # question named it beside CPU/memory/storage, so its reading is appended
        # here rather than returned alone above. None unless the question named it.
        accelerator = readings if readings is not None else presence
        if accelerator is not None:
            lines.append(accelerator["answer"])
            evidence.extend(accelerator.get("evidence") or [])
        # Constants, not readings. Kept in their own module so this method
        # stays about interpreting live facts; only consulted when nothing
        # measured answered.
        if not lines:
            lines = static_answer(lower) or []
        if not lines and asks_about("specialist", lower):
            if specialist_results:
                latest = specialist_results[0]
                lines = [
                    "The latest {} specialist concluded: {}".format(
                        latest.get("profile", "system"),
                        latest.get("summary", "review completed"),
                    )
                ]
                findings = latest.get("findings", [])[:3]
                if findings:
                    lines.append("Key findings: {}.".format("; ".join(findings)))
                recommendations = latest.get("recommendations", [])[:3]
                if recommendations:
                    lines.append(
                        "Recommended next steps: {}.".format(
                            "; ".join(recommendations)
                        )
                    )
            else:
                lines = [
                    "There are no completed appliance checks yet. "
                    "Run a focused read-only specialist review, then use Discuss this result "
                    "to bring its output into this conversation."
                ]
        # Nothing on this machine answered. Two things were wrong with saying
        # so (#184). The list of what this path *can* answer was typed by hand
        # and named eight subjects against the eighteen it has - a false
        # capability denial, which removes working capability and is worse
        # than the guess it prevents - so it is derived from `ANSWER_TOPICS`
        # now. And the paragraph went out through `_validate_answer`'s
        # `answered=True` default, so the audit trail recorded a refusal as a
        # success. A default cannot know what was written; the flag is set
        # here, where the refusal is.
        answered = True
        if not lines:
            model = identity.get("name") or identity.get("id") or "this Vaelor node"
            lines = [
                "I don’t have enough built-in knowledge to answer that reliably "
                "without guessing. Connect a local model, hosted API, or "
                "OpenAI-compatible endpoint for broader questions. {}".format(
                    capability_sentence(model)
                ),
            ]
            answered = False
            if identity:
                add_evidence(evidence, "system.identity", identity, ("name", "id", "model", "architecture"))
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
        if not str(content or "").strip() and message_body.get("reasoning_content"):
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
