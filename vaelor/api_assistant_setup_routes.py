"""Assistant setup, credentials, chat, and planning routes."""

from __future__ import annotations

import platform as platform_module
import re

from flask import g, request

from .assistant_action_requests import (
    decline_sentence,
    detect_action_requests,
    outstanding_actions,
)
from .assistant_acting_wiring import operator_act_scopes
from .assistant_answer_presentation import navigation_steps
from .assistant_intents import chat_destination, select_tools
from .assistant_machine_brief import machine_brief
from .assistant_memory import AssistantMemoryError
from .assistant_tools import AssistantToolError
from .chat_grounding import memory_grounding_allowed
from .chat_inference import ChatInferenceError
from .chat_turn_dedupe import IN_FLIGHT_STATUS, in_flight_error
from .live_readings import one_reading_per_answer
from .local_inference_gate import BUSY_MESSAGE, LocalModelBusy
from .model_profiles import (
    calibrate_in_background,
    calibration_pending,
    model_profile,
)
from .model_footprint import normalise_platform
from .model_reachability import probe_connection
from .model_shortcomings import model_facts
from .copilot_setup import copilot_setup_status, hardware_inventory
from .credential_broker import CredentialError
from .custom_agent_routing import custom_agent_proposal
from .api_common import ApiContext, assistant_model_status, payload as _payload
from .api_credential_routes import register_credential_routes


_custom_agent_proposal = custom_agent_proposal

def register_assistant_setup_routes(context: ApiContext) -> None:
    blueprint = context.blueprint
    callbacks = context.callbacks
    security = context.security
    limiter = context.limiter
    require_auth = context.require_auth
    appliance_address = context.appliance_address

    def active_model_connection(mode=""):
        """The connection the deployment agent would use, or None."""
        agent = callbacks.get("deployment_agent")
        if agent is None:
            return None
        try:
            connection = agent.connection(mode)
        except (AttributeError, CredentialError, OSError, TypeError, ValueError):
            return None
        return connection if connection and connection.get("base_url") else None

    def ai_chat_destination():
        """Whether AI Chat could really take a question this turn.

        Sending someone to AI Chat when AI Chat has no connection is a dead
        end, and claiming it has none when the list could not be read is a
        machine fact asserted from a default. Both are avoided by keeping
        "could not read" as its own state rather than folding it into "none".
        """
        chat = callbacks.get("chat_inference")
        if chat is None:
            return chat_destination(None)
        try:
            return chat_destination(chat.connections())
        except (
            AttributeError, ChatInferenceError, CredentialError, OSError,
            TypeError, ValueError,
        ):
            return chat_destination(None)

    def start_model_calibration(force=False):
        """Measure a newly chosen model without making the caller wait for it.

        Calibration is one real generation, and the slowest model measured on
        this appliance takes over two minutes to finish one. Selecting a model
        has to return now, so the measurement runs behind the response and every
        budget uses the documented conservative default until it lands.
        """
        connection = active_model_connection()
        if connection is None:
            return False
        if not force and connection.get("provider") == "openai":
            # A hosted frontier endpoint bills for the probe. Measuring one is
            # supported, but only when an operator asks for it.
            return False
        return calibrate_in_background(connection, force=force) is not None

    def latest_healthy_model_job(actor):
        job_store = callbacks.get("job_store")
        if job_store is None:
            return None
        return next(
            (
                job for job in job_store.list(100, actor=actor)
                if job.get("type") == "model.deploy"
                and job.get("state") == "healthy"
            ),
            None,
        )

    def reconcile_model_activation(assistant_store, actor):
        """Consume each successful model deployment once in the API owner."""
        choice = assistant_store.get_preference(actor, "intelligence_choice", "")
        job = latest_healthy_model_job(actor)
        if job is None:
            return choice
        marker_key = "last_model_activation_job"
        if assistant_store.get_preference(actor, marker_key, "") == job["id"]:
            return choice
        assistant_store.set_preference(actor, "intelligence_choice", "local")
        assistant_store.set_preference(actor, marker_key, job["id"])
        return "local"

    @blueprint.get("/agent/status")
    @require_auth("viewer")
    def deployment_agent_status():
        agent = callbacks.get("deployment_agent")
        if agent is None:
            return _payload(
                {
                    "name": "Vaelor Deployment Copilot",
                    "configured": False,
                    "provider": "unavailable",
                    "approval_required": True,
                    "tools": [],
                }
            )
        assistant_store = callbacks.get("assistant_memory")
        mode = (
            assistant_store.get_preference(
                g.auth_session.username, "intelligence_choice", ""
            )
            if assistant_store is not None else ""
        )
        status = agent.status(mode=mode)
        # `configured` says a model was chosen; it never said the endpoint
        # answers. Reporting the first as the second is what let the UI show
        # "MODEL READY" while every request through it failed.
        if status.get("configured"):
            connection = active_model_connection(mode)
            try:
                probe = probe_connection(connection)
            except (AttributeError, OSError, TypeError, ValueError):
                probe = {"reachable": True, "detail": "", "endpoint": ""}
            status["reachable"] = bool(probe.get("reachable"))
            status["unreachable_reason"] = (
                "" if probe.get("reachable") else str(probe.get("detail", ""))
            )
            # A reachable server offering no model is not an unreachable one,
            # and saying so was the difference between the Assistant's verdict
            # and AI Chat's on the same endpoint at the same moment.
            status["offering_models"] = probe.get("offering_models")
            status["model_availability_reason"] = str(
                probe.get("model_availability_reason", "")
            )
            status["endpoint"] = str(probe.get("endpoint", ""))
            # What this model was measured to need, so the sizing Vaelor applies
            # is inspectable rather than a number buried in the source.
            status["model_profile"] = model_profile(connection)
            status["calibrating"] = calibration_pending(connection)
            # VD-071's shortcomings, and VD-073's idle-unload periods, both
            # keyed on the model this connection actually answers with.
            #
            # `connection` is None whenever the agent cannot resolve one - no
            # credential, an endpoint that will not parse, a broker that is
            # down - and `configured` above says only that a model was *chosen*.
            # Every other line here already tolerates that; the first version of
            # this one did not, and took the whole endpoint down with a 500 on
            # the appliance while every test passed a connection in.
            #
            # `or {}` rather than a `.get` default, because `model_profile` is a
            # key that exists and holds None when nothing was measured, and a
            # default only applies to a key that is absent.
            profile = status.get("model_profile") or {}
            status["model_facts"] = model_facts(
                str((connection or {}).get("model") or status.get("model") or ""),
                normalise_platform(platform_module.machine()) or "",
                int(profile.get("context_tokens") or 0)
                or int(profile.get("context") or 0),
            )
        return _payload(status)

    @blueprint.post("/agent/model/calibrate")
    @require_auth("administrator", csrf=True)
    def deployment_agent_calibrate():
        """Re-measure the selected model on request.

        An owner who changes what the endpoint serves, or who suspects the
        stored profile no longer fits, can force a fresh measurement without
        deleting and re-adding the credential.
        """
        connection = active_model_connection()
        if connection is None:
            return _payload(
                error={
                    "code": "model_not_selected",
                    "message": "Select an AI model before calibrating it.",
                },
                status=400,
            )
        started = start_model_calibration(force=True)
        security.audit(
            g.auth_session.username, "assistant.model.calibrate", "success",
            target=str(connection.get("model", ""))[:100],
            remote_addr=request.remote_addr or "",
        )
        return _payload({
            "started": started,
            "calibrating": calibration_pending(connection),
            "model_profile": model_profile(connection),
        })

    @blueprint.get("/copilot/setup")
    @require_auth("viewer")
    def copilot_setup():
        probe = callbacks.get("hardware_inventory")
        hardware = probe() if probe is not None else hardware_inventory()
        setup = copilot_setup_status(hardware)
        broker = callbacks.get("credential_broker")
        if broker is not None:
            try:
                capabilities = broker.capabilities()
                setup["credential_storage_ready"] = bool(capabilities.get("ready"))
                available = {
                    item["id"]: item for item in capabilities.get("providers", [])
                }
                for provider in setup["providers"]:
                    if provider["id"] in available:
                        provider["available"] = True
            except CredentialError:
                pass
        return _payload(setup)

    @blueprint.post("/copilot/install-npu-model")
    @require_auth("operator", csrf=True)
    def install_npu_model():
        """Install a fine-tuned on-device NPU model from its pinned release.

        A release-sourced NPU model is not a Hugging Face GGUF, so it does not go
        through inspect -> download -> deploy: the setup screen's Install button
        hits this instead. The tag names which model; the source URL and the
        sha256 the deploy verifies against are read from the catalog shipped in
        the wheel by the executor, never from the client, so the job carries only
        the tag - the trust anchor is the code.
        """
        from .model_catalog import catalog_release_for_tag

        body = request.get_json(silent=True) or {}
        tag = str(body.get("tag") or "").strip()
        if catalog_release_for_tag(tag) is None:
            return _payload(
                error={
                    "code": "unknown_on_device_model",
                    "message": "That on-device model is not available on this appliance.",
                },
                status=400,
            )
        job_store = callbacks.get("job_store")
        if job_store is None:
            return _payload(
                error={"code": "jobs_unavailable", "message": "The job service is unavailable."},
                status=503,
            )
        try:
            job = job_store.create(
                "model.install_release", g.auth_session.username, {"tag": tag}
            )
        except ValueError as error:
            return _payload(
                error={"code": "invalid_job_request", "message": str(error)}, status=400,
            )
        security.audit(
            g.auth_session.username, "model.install_release", "accepted",
            target=tag, remote_addr=request.remote_addr or "",
        )
        return _payload({"job_id": job["id"], "tag": tag})

    @blueprint.get("/assistant/preferences")
    @require_auth("viewer")
    def assistant_preferences():
        assistant_store = callbacks.get("assistant_memory")
        choice = ""
        if assistant_store is not None:
            choice = reconcile_model_activation(
                assistant_store, g.auth_session.username
            )
        return _payload({"intelligence_choice": choice})

    @blueprint.patch("/assistant/preferences")
    @require_auth("operator", csrf=True)
    def assistant_preferences_update():
        assistant_store = callbacks.get("assistant_memory")
        if assistant_store is None:
            return _payload(
                error={
                    "code": "assistant_memory_unavailable",
                    "message": "Assistant preferences are unavailable.",
                },
                status=503,
            )
        body = request.get_json(silent=True) or {}
        choice = str(body.get("intelligence_choice", "")).strip().lower()
        if choice not in {"basic", "local", "provider"}:
            return _payload(
                error={
                    "code": "invalid_intelligence_choice",
                    "message": "Choose built-in help, a local model, or a connected provider.",
                },
                status=400,
            )
        assistant_store.set_preference(
            g.auth_session.username, "intelligence_choice", choice
        )
        latest_model_job = latest_healthy_model_job(g.auth_session.username)
        if latest_model_job is not None:
            assistant_store.set_preference(
                g.auth_session.username,
                "last_model_activation_job",
                latest_model_job["id"],
            )
        security.audit(
            g.auth_session.username,
            "assistant.preference.update",
            "success",
            target="intelligence_choice",
            remote_addr=request.remote_addr or "",
            details={"choice": choice},
        )
        return _payload({"intelligence_choice": choice})

    # The administrator-only /credentials CRUD routes live in a focused module
    # to keep this one under the 1,000-line production ceiling (VD-111). They
    # share the background-calibration helper, threaded in so activation and
    # model selection still measure the new model exactly as before.
    register_credential_routes(context, start_model_calibration)

    @blueprint.post("/assistant/chat")
    @require_auth("operator", csrf=True)
    def assistant_chat():
        agent = callbacks.get("deployment_agent")
        registry = callbacks.get("assistant_tools")
        assistant_store = callbacks.get("assistant_memory")
        if agent is None or registry is None or assistant_store is None:
            return _payload(
                error={
                    "code": "assistant_unavailable",
                    "message": "The conversational assistant is unavailable.",
                },
                status=503,
            )
        body = request.get_json(silent=True) or {}
        message = str(body.get("message", "")).strip()
        # VD-112 follow-up. A retried send carries the same client key, so a
        # duplicate replays the accepted turn rather than writing a second user
        # turn or starting a second inference. Claim before this turn's message
        # is written; complete on success, release on failure.
        actor = g.auth_session.username
        dedupe = callbacks.get("chat_turn_dedupe")
        requested_conversation = str(body.get("conversation_id") or "")
        idempotency_key = str(body.get("idempotency_key", "")).strip()
        if dedupe is not None:
            claim = dedupe.claim(actor, requested_conversation, idempotency_key)
            if claim.replay is not None:
                return _payload(claim.replay)
            if claim.in_flight:
                return _payload(error=in_flight_error(), status=IN_FLIGHT_STATUS)

        def abandon():
            # The in-flight claim must be released on EVERY failure exit,
            # not only the two error types named below. A leaked claim makes
            # the client's resend collide with a phantom in-flight turn (503)
            # and arm a resume-poll for a reply that never lands.
            if dedupe is not None:
                dedupe.abandon(actor, requested_conversation, idempotency_key)

        try:
            custom_store = callbacks.get("custom_agents")
            proposed_agent_task = _custom_agent_proposal(
                message,
                custom_store.list(g.auth_session.username, include_disabled=False)
                if custom_store is not None else [],
            )
            # Word-boundary intent matching so ordinary phrasing ("running hot",
            # "too warm") reaches the same facts as the literal word "temperature",
            # and short tokens no longer pull unrelated topics into the answer.
            selected = select_tools(message)
            # A change request must be answered as a change request, so gather the
            # facts for whatever the user asked to alter even if they never asked
            # about its current state.
            action_requests = detect_action_requests(message)
            available_tools = registry.names()
            for action in action_requests:
                tool_name = "{}.status".format(action["area"])
                if tool_name in available_tools:
                    selected.add(tool_name)
            facts = {}
            # One question, one hardware sample. Gathering each fact tool with its
            # own reading is what let a single reply quote two CPU temperatures
            # taken moments apart.
            with one_reading_per_answer():
                for tool_name in sorted(selected):
                    try:
                        facts[tool_name] = registry.run(
                            tool_name,
                            {"limit": 8}
                            if tool_name in {"jobs.recent", "recovery.checkpoints"}
                            else {},
                            actor=g.auth_session.username,
                            administrator=g.auth_session.role == "administrator",
                        )["result"]
                    except AssistantToolError as error:
                        facts[tool_name] = {"unavailable": str(error)}
            task_store = callbacks.get("agent_tasks")
            specialist_results = []
            if task_store is not None:
                specialist_results = [
                    {
                        "id": item["id"],
                        "title": item["title"],
                        "profile": item["profile"],
                        "summary": item.get("result", {}).get("summary", ""),
                        "findings": item.get("result", {}).get("findings", [])[:8],
                        "recommendations": item.get("result", {}).get("recommendations", [])[:8],
                        "next_actions": item.get("result", {}).get("next_actions", [])[:8],
                    }
                    for item in task_store.list(
                        actor=g.auth_session.username,
                        limit=6,
                    )
                    if item.get("state") == "completed"
                ][:3]
            model_status = assistant_model_status(
                callbacks, g.auth_session.username
            )
            skill_store = callbacks.get("assistant_skills")
            matched_skills = [
                {
                    "slug": item["slug"],
                    "name": item["name"],
                    "version": item["version"],
                    "guidance": item["content"],
                }
                for item in (
                    skill_store.match(message)
                    if skill_store is not None and model_status["ready"] else []
                )
            ]
            slot_cache = callbacks.get("assistant_slot_cache")
            conversation = assistant_store.ensure_conversation(
                g.auth_session.username,
                body.get("conversation_id"),
                title=message,
            )
            # The standing brief travels with every answer. Facts alone
            # do not say which machine produced them, and a model
            # without that reads a workstation's normal 92 °C as the
            # Raspberry Pi emergency it is not. Built once: the same text
            # rides the request and identifies any saved KV prefix.
            brief = machine_brief(callbacks)
            mode = assistant_store.get_preference(
                g.auth_session.username, "intelligence_choice", ""
            )
            # Before this turn's user message is written, because the saved
            # prefix is only provably current while the recorded last message
            # is still the conversation's last message (VD-076). A refused or
            # failed restore costs nothing but the full prefill.
            if slot_cache is not None:
                slot_cache.restore(
                    agent.connection(mode),
                    actor=g.auth_session.username,
                    conversation_id=conversation["id"],
                    brief_text=brief["text"],
                )
            assistant_store.append_message(
                conversation["id"], g.auth_session.username, "user", message
            )
            context = assistant_store.context_for(
                query=message,
                actor=g.auth_session.username,
                conversation_id=conversation["id"],
                live_context={
                    "machine_brief": brief,
                    "facts": facts,
                    "telemetry_history_range": callbacks.get("telemetry_history_range"),
                    "specialist_results": specialist_results,
                    "matched_skills": matched_skills,
                    # A question this appliance holds no evidence for is sent
                    # to AI Chat, so the answer has to know whether AI Chat is
                    # actually set up before it names it as the destination.
                    "ai_chat": ai_chat_destination(),
                },
                # Curated memory is administrator-owned: every verb on
                # /assistant/memories is administrator-only. Grounding an
                # answer on a memory hands its text to the selected model, so
                # it is a read of that content and stays on the same role.
                # Widening it so an operator's answer is as well informed as an
                # administrator's is a real product question - it decides where
                # curated appliance facts are allowed to travel - and it needs
                # to be decided deliberately, with its own test, not as a side
                # effect of making both surfaces behave alike.
                include_curated_memory=memory_grounding_allowed(
                    g.auth_session.role
                ),
            )
            # Default-inert acting scopes (VD-100 #96): no grant -> nothing proposed.
            answer = agent.answer(
                message, context=context, mode=mode,
                granted_scopes=operator_act_scopes(
                    callbacks.get("workload_act_grants"), g.auth_session.username
                ),
            )
            # A change request must never be answered with only the current
            # state. If nothing downstream raised an approval for it, say
            # plainly that no change was made and where to make it, keeping the
            # value the user actually asked for.
            # Proposing a change and refusing it in the same reply is a
            # contradiction the user cannot resolve, so one function decides
            # which requests are still unanswered. It also keeps declining only
            # the areas nothing downstream took up, so "restart grafana and turn
            # the lights purple" never drops the lighting half.
            outstanding = outstanding_actions(
                action_requests, answer, proposed_agent_task
            )
            if proposed_agent_task is not None:
                answer.update({
                    "answer": (
                        "I matched this request to {} version {}. Review the exact "
                        "agent run before Vaelor queues it."
                    ).format(
                        proposed_agent_task["profile_name"],
                        proposed_agent_task["profile_version"],
                    ),
                    "source": "agent-router",
                    "proposed_job": None,
                    "application_intent": None,
                    "approval_required": True,
                    "evidence": [{
                        "source": "custom-agent.{}".format(
                            proposed_agent_task["profile_id"]
                        ),
                        "summary": "Matched the enabled, account-owned agent definition.",
                    }],
                    "suggested_actions": [
                        "Review the pinned agent version and granted capabilities before queuing it."
                    ],
                })
            # Declines are appended LAST so nothing downstream can overwrite
            # them. Rewriting the answer for a matched agent previously threw
            # away the statement that a requested change had not been made.
            if outstanding:
                answer["answer"] = "\n\n".join([
                    str(answer.get("answer", "")).strip(),
                    *[decline_sentence(action) for action in outstanding],
                ]).strip()
                answer["suggested_actions"] = [
                    "Open {} to change {}.".format(action["screen"], action["subject"])
                    for action in outstanding
                ] + list(answer.get("suggested_actions") or [])
                answer["unperformed_actions"] = outstanding
            for skill in matched_skills:
                answer["evidence"].append({
                    "source": "skill.{}".format(skill["slug"]),
                    "summary": "Applied reviewed {} guidance (version {}).".format(
                        skill["name"], skill["version"]
                    ),
                })
            answer["evidence"] = answer["evidence"][:10]
            # Naming a destination is not the same as getting the user there.
            # Every screen the answer names is emitted as a real hash route the
            # frontend can render as a link, alongside the human sentence.
            answer["next_steps"] = navigation_steps(answer, outstanding)
            reply = assistant_store.append_message(
                conversation["id"],
                g.auth_session.username,
                "assistant",
                answer["answer"],
                metadata={
                    "source": answer["source"],
                    "evidence": answer["evidence"],
                    "suggested_actions": answer["suggested_actions"],
                    "proposed_job": answer["proposed_job"],
                    "application_intent": answer.get("application_intent"),
                    "proposed_agent_task": proposed_agent_task,
                    "next_steps": answer["next_steps"],
                },
            )
            # The save rides the write that already exists (owner's design,
            # VD-076): llama-server owns the idle timer and never announces a
            # sleep, and saving where the reply is appended makes knowing it
            # unnecessary. Only when the model itself answered - a grounded
            # answer never touched the engine, so the slot holds some other
            # exchange and recording it against this conversation would claim
            # a prefix this transcript cannot prove.
            if slot_cache is not None and answer.get("source") == "connected-model":
                slot_cache.save(
                    agent.connection(mode),
                    actor=g.auth_session.username,
                    conversation_id=conversation["id"],
                    message_id=reply["id"],
                    brief_text=brief["text"],
                )
        except LocalModelBusy as error:
            # LocalModelBusy is a RuntimeError, outside answer()'s except tuple
            # (OSError, ValueError, KeyError, URLError), so it arrives uncaught
            # and used to escape as an HTTP 500 that read as an outage. Mapped
            # to a truthful 503 like AI Chat's chat_model_busy; caught by type.
            if dedupe is not None:
                dedupe.abandon(actor, requested_conversation, idempotency_key)
            return _payload(
                error={"code": "assistant_model_busy",
                       "message": str(error) or BUSY_MESSAGE},
                status=503,
            )
        except (AssistantMemoryError, ValueError) as error:
            if dedupe is not None:
                dedupe.abandon(actor, requested_conversation, idempotency_key)
            return _payload(
                error={"code": "invalid_assistant_request", "message": str(error)},
                status=400,
            )
        except Exception:
            # Any other failure between the claim and the terminal
            # complete() must still release the claim. Re-raised so the
            # framework still surfaces the underlying error.
            abandon()
            raise
        answer["conversation_id"] = conversation["id"]
        answer["proposed_agent_task"] = proposed_agent_task
        # A turn that told the user their model did not answer is not a
        # success, and recording it as one put two visible failures into a
        # list of six SUCCESS rows on a card promising authenticated actions.
        answered = answer.get("answered", True)
        try:
            # These terminal steps sit inside the claim's guarded region so a
            # failure still releases the claim, or the client's resend collides
            # with a phantom in-flight turn (503) - the very leak this closes.
            # audit() is the SQLite writer that can raise under a concurrent-
            # writer lock; complete() is an in-memory store write wrapped here
            # for symmetry (harmless, and it cannot wrongly drop a completed
            # claim - abandon() only removes in-flight records).
            security.audit(
                g.auth_session.username,
                "assistant.chat",
                "success" if answered else "failure",
                target=conversation["id"],
                remote_addr=request.remote_addr or "",
                details={
                    "source": answer["source"],
                    "tools": sorted(facts),
                    "approval_required": answer["approval_required"],
                    "answered": answered,
                },
            )
            if dedupe is not None:
                dedupe.complete(actor, requested_conversation, idempotency_key, answer)
        except Exception:
            abandon()
            raise
        return _payload(answer)

    @blueprint.post("/agent/plan")
    @require_auth("operator", csrf=True)
    def deployment_agent_plan():
        agent = callbacks.get("deployment_agent")
        if agent is None:
            return _payload(
                error={
                    "code": "deployment_agent_unavailable",
                    "message": "The deployment agent is not configured.",
                },
                status=503,
            )
        body = request.get_json(silent=True) or {}
        message = str(body.get("message", "")).strip()
        bootstrap = str(body.get("bootstrap", "")).strip().lower()
        model_bootstrap = (
            bootstrap == "model-install"
            and re.fullmatch(
                r"[A-Za-z0-9_.-]+/[A-Za-z0-9_.-]+(?:\s+[A-Za-z0-9_.-]+){0,4}",
                message,
            ) is not None
        )
        model_status = assistant_model_status(
            callbacks, g.auth_session.username
        )
        probe = callbacks.get("workload_capabilities")
        live_context = probe() if probe is not None else {}
        skill_store = callbacks.get("assistant_skills")
        if skill_store is not None:
            matched = skill_store.match(str(message))
            live_context = dict(live_context)
            live_context["matched_skills"] = [
                {
                    "name": item["name"],
                    "version": item["version"],
                    "guidance": item["content"],
                    "provenance": item["provenance"],
                }
                for item in matched
            ]
        assistant_store = callbacks.get("assistant_memory")
        # A catalogue selection is a click, not something the owner said.
        # Persisting it created a saved chat titled
        # "unsloth/Qwen3-4B-Instruct-2507-GGUF Q4_0" holding the click as if
        # it had been typed, and returning to the Assistant afterwards landed
        # the reader in that machine-written conversation instead of their own
        # (#160). Sixteen of them had accumulated on the appliance. The plan
        # is still produced and still returned; only the transcript is
        # withheld. **The `or conversation_id` escape came off:**
        # `Workloads.tsx` holds the id in component state and reuses it, so an
        # owner who asked one real question and then opened the catalogue had
        # every click appended into their own conversation. Both call sites
        # that send this marker are clicks, never typed messages.
        remember = not model_bootstrap
        conversation = None
        try:
            if assistant_store is not None and remember:
                conversation = assistant_store.ensure_conversation(
                    g.auth_session.username,
                    body.get("conversation_id"),
                    title=str(message),
                )
                context = assistant_store.context_for(
                    query=str(message),
                    actor=g.auth_session.username,
                    conversation_id=conversation["id"],
                    live_context=live_context,
                    # Same grounding policy as /assistant/chat: curated memory
                    # is read by the role that may read curated memory.
                    include_curated_memory=memory_grounding_allowed(
                        g.auth_session.role
                    ),
                )
                assistant_store.append_message(
                    conversation["id"], g.auth_session.username, "user", str(message)
                )
            else:
                context = live_context
            # A catalog selection is already a structured, allowlisted model
            # query. Route it through the deterministic planner even when a
            # local/provider model is active. Small models may describe a valid
            # install yet omit the executable model.inspect job, which would
            # otherwise strand the user without an approval action.
            mode = (
                "basic"
                if model_bootstrap or not model_status["ready"]
                else assistant_store.get_preference(
                    g.auth_session.username, "intelligence_choice", ""
                )
                if assistant_store is not None
                else ""
            )
            plan = agent.plan(message, context=context, mode=mode)
            if assistant_store is not None and conversation is not None:
                assistant_store.append_message(
                    conversation["id"],
                    g.auth_session.username,
                    "assistant",
                    "{}\n\n{}".format(
                        plan.get("summary", ""), plan.get("rationale", "")
                    ).strip(),
                    metadata={
                        "source": plan.get("source"),
                        "proposed_job": plan.get("proposed_job"),
                    },
                )
                plan["conversation_id"] = conversation["id"]
        except (ValueError, AssistantMemoryError) as error:
            return _payload(
                error={"code": "invalid_agent_request", "message": str(error)},
                status=400,
            )
        security.audit(
            g.auth_session.username,
            "agent.plan",
            "success",
            target="deployment-copilot",
            remote_addr=request.remote_addr or "",
            details={
                "source": plan.get("source"),
                "proposed_job_type": (
                    plan.get("proposed_job", {}) or {}
                ).get("type"),
            },
        )
        return _payload(plan)
