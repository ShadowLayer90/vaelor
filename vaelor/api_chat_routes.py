"""Independent AI Chat, model selection, and local RAG collection routes."""

from __future__ import annotations

import re
import sqlite3
from collections.abc import Mapping

from flask import g, request

from .api_common import ApiContext, payload as _payload
from .chat_connections import describe_connections, local_tier_map
from .copilot_setup import copilot_setup_status, hardware_inventory
from .inference_status import inference_status
from .local_endpoint_discovery import discover_cached as discover_local_endpoints
from .model_reachability import probe_connection
from .chat_appliance_scope import (
    APPLIANCE_SCOPE_PROVIDER,
    appliance_scope_decline,
    record_decline,
    retrieval_answers_question,
)
from .chat_grounding import (
    curated_memories,
    memory_grounding_allowed,
    resolve_collections,
    retrieval_summary,
    searchable_collections,
)
from .chat_inference import ChatInferenceError
from .chat_turn_dedupe import IN_FLIGHT_STATUS, in_flight_error
from .custom_agent_routing import custom_agent_proposal
from .rag_chat import RagChatError


def register_chat_routes(context: ApiContext) -> None:
    blueprint = context.blueprint
    callbacks = context.callbacks
    security = context.security
    require_auth = context.require_auth

    def services():
        return callbacks.get("rag_chat"), callbacks.get("chat_inference")

    def grounding_memories(message):
        """Curated memory for callers whose role is allowed to read it.

        Memory and knowledge collections are not the same corpus: collections
        are searched and cited per request, memory is standing background
        context. The inference runtime labels them separately so a remembered
        fact can never surface as an [S#] citation.

        Every verb on /api/v2/assistant/memories is administrator-only, and
        this route is open to operators. Grounding an answer on a memory hands
        that memory's text to the selected model - which may be a cloud
        provider - so it is a read of administrator-owned content and is gated
        on the same role. An operator's answer is simply built without it.
        """
        if not memory_grounding_allowed(getattr(g.auth_session, "role", "")):
            return {"memories": [], "available": True}
        return curated_memories(callbacks.get("assistant_memory"), message)

    def grounded_retrieval(actor, collection_ids, retrieved, memory):
        """Report what actually grounded this answer, including what did not.

        Counting a collection the actor cannot read as "searched", or staying
        silent when the memory store could not be opened, both describe a
        better-grounded answer than the one that was produced. `actor` is
        required so the count reflects collections this reader can actually
        search - two callers already passed it while this signature did not
        accept it, which raised a TypeError on the regenerate path.
        """
        store = callbacks.get("rag_chat")
        owned = collection_ids
        if store is not None and hasattr(store, "owned_collections"):
            try:
                # `owned_collections(actor, collection_ids)` already returns the
                # readable subset. Calling it with the actor alone raised a
                # TypeError that the old broad `except` swallowed, silently
                # leaving `owned` unfiltered - the readable filter never ran.
                owned = store.owned_collections(actor, collection_ids)
            except (AttributeError, TypeError, ValueError, sqlite3.Error):
                owned = collection_ids
        summary = retrieval_summary(owned, retrieved)
        if not memory.get("available", True):
            summary["memory_unavailable"] = True
        return summary

    def conversation_patch(model, collection_ids, collections_chosen):
        """Only write the knowledge selection back when the client chose it.

        Sending a request used to overwrite the conversation's collections with
        whatever the client happened to hold, so one visit to a chat created
        before a collection existed replayed its empty list and made the
        setting permanent. An unspecified selection now leaves the stored one
        untouched.

        The model is a claim about what answered in this conversation, so an
        empty ``model`` leaves the stored one alone. Writing it before the
        request was sent meant a model that produced nothing still relabelled
        the chat, and the sidebar faithfully showed a conversation as belonging
        to a model that never spoke in it.
        """
        patch = {}
        if model:
            patch["model"] = model
        if collections_chosen:
            patch["collections"] = collection_ids
        return patch

    def apply_patch(store, actor, conversation, patch):
        return store.update_conversation(actor, conversation["id"], patch) if patch else conversation

    def chat_failure(
        error, fallback_code: str, fallback_status: int, **details,
    ):
        return _payload(
            error={
                "code": getattr(error, "code", fallback_code),
                "message": str(error),
                **details,
            },
            status=getattr(error, "status", fallback_status),
        )

    def citations_for(retrieved):
        return [
            {
                "id": item["id"], "collection_id": item["collection_id"],
                "collection": item["collection_name"],
                "document_id": item["document_id"],
                "document": item["document_name"],
                "chunk": int(item["ordinal"]) + 1,
                "excerpt": item["content"][:400],
            }
            for item in retrieved
        ]

    def custom_profiles(actor):
        """Return only enabled, actor-owned custom profiles from server stores."""
        task_store = callbacks.get("agent_tasks")
        profiles = []
        try:
            profiles = task_store.profiles(actor) if task_store is not None else []
        except (AttributeError, TypeError, ValueError):
            profiles = []
        if not profiles:
            store = callbacks.get("custom_agents")
            try:
                profiles = store.list(actor, include_disabled=True) if store is not None else []
            except (AttributeError, TypeError, ValueError):
                profiles = []
        return [
            profile for profile in profiles
            if isinstance(profile, Mapping)
            and bool(profile.get("custom"))
            and bool(profile.get("enabled", True))
        ]

    def named_profile_match(message, profiles):
        """Require a name token before the shared matcher can select a profile."""
        message_tokens = set(re.findall(r"[a-z0-9]+", str(message).lower()))
        stopwords = {"agent", "assistant", "custom", "my", "the", "a", "an"}
        for profile in profiles:
            name_tokens = {
                token for token in re.findall(r"[a-z0-9]+", str(profile.get("name", "")).lower())
                if token not in stopwords and len(token) > 1
            }
            if name_tokens.intersection(message_tokens):
                return True
        return False

    def grant_summaries(actor, profile):
        """Decorate a proposal with bounded, non-secret current app access."""
        grants = callbacks.get("agent_app_grants")
        registry = callbacks.get("app_capability_registry")
        if grants is None:
            return []
        try:
            rows = grants.list(actor, agent_id=str(profile.get("id", "")), limit=50)
        except (AttributeError, TypeError, ValueError):
            return []
        try:
            selected_version = int(profile.get("version", 0))
        except (TypeError, ValueError):
            selected_version = 0
        summaries = []
        for grant in rows:
            if not isinstance(grant, Mapping) or bool(grant.get("revoked")):
                continue
            try:
                if int(grant.get("agent_version", -1)) != selected_version:
                    continue
            except (TypeError, ValueError):
                continue
            app = None
            manifest = None
            try:
                if registry is not None:
                    app = registry.get_app_instance(str(grant.get("app_instance_id", "")))
                    manifest = registry.get_manifest(str(grant.get("manifest_digest", "")))
            except (AttributeError, TypeError, ValueError):
                app = None
                manifest = None
            operations_by_id = {}
            if manifest is not None:
                operations_by_id = {
                    str(operation.operation_id): operation
                    for operation in getattr(manifest, "operations", ())
                }
            operations = []
            for operation_id in list(grant.get("operation_ids", []))[:20]:
                operation = operations_by_id.get(str(operation_id))
                mode = str(getattr(operation, "mode", "read"))
                operations.append({
                    "id": str(operation_id)[:80],
                    "name": str(getattr(operation, "label", operation_id))[:100],
                    "access": mode if mode in {"read", "write"} else "read",
                    "risk": str(getattr(operation, "risk", "low"))[:40],
                })
            summaries.append({
                "grant_id": str(grant.get("id", ""))[:96],
                "app_instance_id": str(grant.get("app_instance_id", ""))[:160],
                "app_name": str((app or {}).get("app_label", "Installed app"))[:100],
                "manifest_version": str(getattr(manifest, "manifest_version", ""))[:40],
                "manifest_digest": str(grant.get("manifest_digest", ""))[:64],
                "operations": operations,
            })
        return summaries

    def agent_proposal(actor, message, requested_profile_id=""):
        profiles = custom_profiles(actor)
        requested_profile_id = str(requested_profile_id or "").strip()
        if requested_profile_id:
            selected_profiles = [
                profile for profile in profiles
                if str(profile.get("id", "")) == requested_profile_id
            ]
            if not selected_profiles:
                return None
            profiles = selected_profiles
        elif not profiles or not named_profile_match(message, profiles):
            return None
        proposal = custom_agent_proposal(message, profiles, explicit=bool(requested_profile_id))
        if proposal is None:
            return None
        selected = next(
            profile for profile in profiles
            if str(profile.get("id")) == proposal["profile_id"]
        )
        proposal["app_grants"] = grant_summaries(actor, selected)
        proposal["approval_required"] = True
        return proposal

    def proposal_text(proposal):
        capabilities = ", ".join(proposal.get("capabilities", [])) or "none listed"
        integrations = ", ".join(proposal.get("integrations", [])) or "none"
        app_access = []
        for grant in proposal.get("app_grants", []):
            operations = ", ".join(
                "{} ({})".format(item.get("name", item.get("id", "")), item.get("access", "read"))
                for item in grant.get("operations", [])
            ) or "no operations listed"
            app_access.append("{}: {}".format(grant.get("app_name", "Installed app"), operations))
        # The internal profile id belongs in the structured proposal the
        # client already receives, not in prose. Reading a raw
        # "custom_c4f7dc80..." GUID back to the user names nothing they can
        # act on and looks like a leak.
        return (
            "I matched this request to the enabled custom agent '{}' (version {}). "
            "No work has started. The bounded task is: {}. Capabilities: {}. "
            "App access: {}. Integrations: {}. Approval required: yes. Review this exact "
            "run, then approve it under Assistant > Agents, in this agent's run history."
        ).format(
            proposal["profile_name"], proposal["profile_version"],
            proposal["task"], capabilities, "; ".join(app_access) or "none granted",
            integrations,
        )

    @blueprint.get("/ai-chat/setup")
    @require_auth("operator")
    def chat_setup():
        store, inference = services()
        if store is None or inference is None:
            return _payload(error={"code": "ai_chat_unavailable", "message": "AI Chat is unavailable."}, status=503)
        try:
            connections = inference.connections()
        except ChatInferenceError as error:
            return chat_failure(error, "chat_connection_unavailable", 503)
        active = next(
            (item for item in connections if "ai-chat" in item.get("active_for", [])),
            None,
        )
        try:
            unavailable_models = inference.known_bad_models()
        except (AttributeError, ChatInferenceError):
            unavailable_models = {}
        plan = {}
        try:
            probe = callbacks.get("hardware_inventory")
            plan = copilot_setup_status(
                probe() if probe is not None else hardware_inventory()
            ).get("inference_plan", {})
        except (OSError, RuntimeError, ValueError):
            plan = {}

        def loaded_for(connection):
            """Which models this server is holding, or None if it does not say.

            None is not "none loaded". A server that publishes no load state
            gets a picker warning, not a false claim that nothing is resident.
            """
            base_url = str(connection.get("base_url") or "")
            if not base_url:
                return None
            result = probe_connection(dict(connection))
            if not result.get("reachable"):
                return None
            return result.get("loaded_models")

        described = describe_connections(
            connections,
            loaded_for=loaded_for,
            failures_for=lambda _connection: unavailable_models,
        )
        # Only endpoints not already registered: an owner who has accepted one
        # should not keep being offered it.
        known = {
            str(item.get("base_url") or "").rstrip("/") for item in described
        }
        # The cached form, never the blocking sweep: this is a page-load
        # endpoint. Calling `discover()` here made it wait on twelve serial
        # probes - eighteen seconds on a host that drops rather than refuses -
        # and browsers aborted it mid-flight.
        discovered = discover_local_endpoints()
        discovered["endpoints"] = [
            item for item in discovered["endpoints"]
            if str(item.get("base_url") or "").rstrip("/") not in known
        ]
        return _payload({
            # Every connection, each with `local`, its models, and per-model
            # load state. The privacy grouping the picker draws is a
            # presentation of `local`, not a second code path: `model.deploy`
            # already registers the appliance's own server as a connection.
            "connections": described,
            "active_connection": active,
            # Which locally-served model runs on which engine, so a local model
            # is identifiable as local *and* placeable.
            "local_tiers": local_tier_map(plan),
            # Model servers already running on this machine that Vaelor did not
            # start. Offered, never adopted: a machine running the recommended
            # two-tier setup reported MODEL REQUIRED because the only path that
            # could register a local model was the one where Vaelor deployed it
            # itself. `discovered` is what the picker offers; nothing here is
            # connected until an owner accepts it.
            "discovered_endpoints": discovered,
            "preference": store.preference(g.auth_session.username),
            "collections": store.collections(g.auth_session.username),
            "limits": {"document_bytes": 2 * 1024 * 1024, "collections": 50},
            "retrieval": "Full-text search over your knowledge, with cited passages",
            # Models on the active connection that have really failed here, with
            # the reason, so the picker can mark one before the user sends a
            # question into it rather than after. Measured from real requests -
            # nothing is probed or predicted - and a model that starts answering
            # drops out of this map on its own.
            "unavailable_models": unavailable_models,
        })

    @blueprint.get("/inference/status")
    @require_auth("viewer")
    def inference_engine_status():
        """What each engine is holding, and what memory that accounts for.

        Assembled from facts already established elsewhere rather than probed
        afresh, and every gap is stated: an engine nobody configured reports
        that nothing was asked, and memory the driver says is in use but no
        engine accounts for is reported as unattributed rather than dropped.
        """
        probe = callbacks.get("hardware_inventory")
        hardware = probe() if probe is not None else hardware_inventory()
        _, inference = services()
        connections = []
        if inference is not None:
            try:
                connections = inference.connections()
            except ChatInferenceError:
                connections = []
        return _payload(inference_status(
            hardware,
            connections=connections,
            probe=lambda connection: probe_connection(dict(connection)),
            # The listed records are redacted (no base_url); this resolves the
            # active managed-local lease so the CPU engine can be probed and show
            # the deployed model rather than model=null (#205 Step 3, VD-071).
            resolve_local=(
                inference.active_local_connection if inference is not None else None
            ),
        ))

    @blueprint.post("/ai-chat/connections/<credential_id>/activate")
    @require_auth("operator", csrf=True)
    def chat_connection_activate(credential_id):
        _, inference = services()
        try:
            result = inference.activate(credential_id)
        except (AttributeError, ChatInferenceError) as error:
            return chat_failure(error, "chat_connection_rejected", 400)
        security.audit(
            g.auth_session.username, "ai_chat.connection.activate", "success",
            target=credential_id, remote_addr=request.remote_addr or "",
        )
        return _payload(result)

    @blueprint.get("/ai-chat/connections/<credential_id>/models")
    @require_auth("operator")
    def chat_connection_models(credential_id):
        _, inference = services()
        try:
            return _payload(inference.models(credential_id))
        except (AttributeError, ChatInferenceError) as error:
            return chat_failure(error, "chat_models_unavailable", 400)

    @blueprint.patch("/ai-chat/preferences")
    @require_auth("operator", csrf=True)
    def chat_preferences():
        store, _ = services()
        body = request.get_json(silent=True) or {}
        try:
            result = store.set_preference(
                g.auth_session.username,
                body.get("model", ""),
                body.get("collection_ids", []),
            )
        except (AttributeError, RagChatError, TypeError) as error:
            return _payload(error={"code": "chat_preference_rejected", "message": str(error)}, status=400)
        return _payload(result)

    @blueprint.post("/ai-chat/collections")
    @require_auth("operator", csrf=True)
    def collection_create():
        store, _ = services()
        body = request.get_json(silent=True) or {}
        try:
            item = store.create_collection(
                g.auth_session.username, body.get("name", ""), body.get("description", "")
            )
        except (AttributeError, RagChatError) as error:
            return _payload(error={"code": "collection_rejected", "message": str(error)}, status=400)
        security.audit(
            g.auth_session.username, "ai_chat.collection.create", "success",
            target=item["id"], remote_addr=request.remote_addr or "",
        )
        return _payload(item, status=201)

    @blueprint.delete("/ai-chat/collections/<collection_id>")
    @require_auth("operator", csrf=True)
    def collection_delete(collection_id):
        store, _ = services()
        try:
            result = store.delete_collection(g.auth_session.username, collection_id)
        except (AttributeError, RagChatError) as error:
            return _payload(error={"code": "collection_not_found", "message": str(error)}, status=404)
        security.audit(
            g.auth_session.username, "ai_chat.collection.delete", "success",
            target=collection_id, remote_addr=request.remote_addr or "",
        )
        return _payload(result)

    @blueprint.get("/ai-chat/collections/<collection_id>/documents")
    @require_auth("operator")
    def document_list(collection_id):
        store, _ = services()
        try:
            return _payload(store.documents(g.auth_session.username, collection_id))
        except (AttributeError, RagChatError) as error:
            return _payload(error={"code": "collection_not_found", "message": str(error)}, status=404)

    @blueprint.post("/ai-chat/collections/<collection_id>/documents")
    @require_auth("operator", csrf=True)
    def document_ingest(collection_id):
        store, _ = services()
        body = request.get_json(silent=True) or {}
        try:
            document = store.ingest(
                g.auth_session.username, collection_id, body.get("name", ""),
                body.get("content", ""), body.get("media_type", "text/plain"),
            )
        except (AttributeError, RagChatError) as error:
            return _payload(error={"code": "document_rejected", "message": str(error)}, status=400)
        security.audit(
            g.auth_session.username, "ai_chat.document.ingest", "success",
            target=document["id"], remote_addr=request.remote_addr or "",
            details={"collection_id": collection_id, "size_bytes": document["size_bytes"]},
        )
        return _payload(document, status=201)

    @blueprint.delete("/ai-chat/documents/<document_id>")
    @require_auth("operator", csrf=True)
    def document_delete(document_id):
        store, _ = services()
        try:
            result = store.delete_document(g.auth_session.username, document_id)
        except (AttributeError, RagChatError) as error:
            return _payload(error={"code": "document_not_found", "message": str(error)}, status=404)
        security.audit(
            g.auth_session.username, "ai_chat.document.delete", "success",
            target=document_id, remote_addr=request.remote_addr or "",
        )
        return _payload(result)

    @blueprint.get("/ai-chat/conversations")
    @require_auth("operator")
    def chat_conversations():
        store, _ = services()
        archived = request.args.get("archived", "").lower() in {"1", "true", "yes"}
        return _payload(store.conversations(
            g.auth_session.username, archived=archived,
            query=request.args.get("q", ""),
        ))

    @blueprint.get("/ai-chat/conversations/<conversation_id>/messages")
    @require_auth("operator")
    def chat_messages(conversation_id):
        store, _ = services()
        try:
            return _payload(store.messages(g.auth_session.username, conversation_id))
        except (AttributeError, RagChatError) as error:
            return _payload(error={"code": "chat_not_found", "message": str(error)}, status=404)

    @blueprint.patch("/ai-chat/conversations/<conversation_id>")
    @require_auth("operator", csrf=True)
    def chat_conversation_update(conversation_id):
        store, _ = services()
        try:
            return _payload(store.update_conversation(
                g.auth_session.username, conversation_id,
                request.get_json(silent=True) or {},
            ))
        except (AttributeError, RagChatError) as error:
            return _payload(error={"code": "chat_update_rejected", "message": str(error)}, status=400)

    @blueprint.delete("/ai-chat/conversations/<conversation_id>")
    @require_auth("operator", csrf=True)
    def chat_conversation_delete(conversation_id):
        store, _ = services()
        try:
            return _payload(store.delete_conversation(g.auth_session.username, conversation_id))
        except (AttributeError, RagChatError) as error:
            return _payload(error={"code": "chat_not_found", "message": str(error)}, status=404)

    @blueprint.post("/ai-chat/conversations/<conversation_id>/fork")
    @require_auth("operator", csrf=True)
    def chat_conversation_fork(conversation_id):
        store, _ = services()
        body = request.get_json(silent=True) or {}
        message_id = body.get("through_message_id")
        try:
            branch = store.fork_conversation(
                g.auth_session.username, conversation_id,
                int(message_id) if message_id is not None else None,
            )
        except (AttributeError, RagChatError, TypeError, ValueError) as error:
            return _payload(error={"code": "chat_fork_rejected", "message": str(error)}, status=400)
        security.audit(
            g.auth_session.username, "ai_chat.conversation.fork", "success",
            target=branch["id"], remote_addr=request.remote_addr or "",
            details={"source": conversation_id},
        )
        return _payload(branch, status=201)

    @blueprint.post("/ai-chat/conversations/<conversation_id>/regenerate")
    @require_auth("operator", csrf=True)
    def chat_conversation_regenerate(conversation_id):
        store, inference = services()
        body = request.get_json(silent=True) or {}
        try:
            prepared = store.prepare_regeneration(
                g.auth_session.username, conversation_id,
                int(body.get("message_id")), body.get("content"),
            )
            conversation = prepared["conversation"]
            searchable = searchable_collections(
                store, g.auth_session.username, conversation["collections"]
            )
            # Regeneration re-asks the question, so the boundary applies to it
            # exactly as it does to the first send: retrieve first, and let the
            # appliance-scope gate decline only what the documents could not
            # answer. Without this an edited prompt was a way around the gate.
            retrieved = store.retrieve(
                g.auth_session.username, prepared["prompt"], searchable, limit=6,
            )
            if not retrieval_answers_question(prepared["prompt"], retrieved):
                decline = appliance_scope_decline(prepared["prompt"])
                if decline is not None:
                    declined = store.add_message(
                        g.auth_session.username, conversation_id, "assistant",
                        decline["answer"], [], None, APPLIANCE_SCOPE_PROVIDER,
                        decline["metadata"],
                    )
                    return _payload({
                        "messages": store.messages(
                            g.auth_session.username, conversation_id
                        ),
                        "message": declined,
                        "model": conversation["model"],
                        "provider": APPLIANCE_SCOPE_PROVIDER,
                    })
            memory = grounding_memories(prepared["prompt"])
            retrieval = grounded_retrieval(
                g.auth_session.username, searchable, retrieved, memory
            )
            result = inference.answer(
                prepared["prompt"], model=conversation["model"],
                history=prepared["history"], retrieved=retrieved,
                memories=memory["memories"],
            )
            assistant_message = store.add_message(
                g.auth_session.username, conversation_id, "assistant",
                result["answer"], citations_for(retrieved), retrieval,
                result["model"],
            )
        except (
            AttributeError, ChatInferenceError, RagChatError, TypeError, ValueError,
            sqlite3.Error,
        ) as error:
            return chat_failure(error, "chat_regenerate_failed", 400)
        security.audit(
            g.auth_session.username, "ai_chat.message.regenerate", "success",
            target=conversation_id, remote_addr=request.remote_addr or "",
            details={"model": result["model"], **retrieval},
        )
        return _payload({
            "messages": store.messages(g.auth_session.username, conversation_id),
            "message": assistant_message,
            "model": result["model"], "provider": result["provider"],
            "retrieval": retrieval,
        })

    @blueprint.post("/ai-chat/messages")
    @require_auth("operator", csrf=True)
    def chat_message():
        store, inference = services()
        body = request.get_json(silent=True) or {}
        actor = g.auth_session.username
        message = str(body.get("message", "")).strip()
        preference = store.preference(actor)
        model = str(body.get("model") or preference["model"]).strip()
        temporary = body.get("temporary") is True
        requested_conversation = str(body.get("conversation_id", ""))
        # VD-112 follow-up. A retried send carries the same client key, so a
        # duplicate replays the accepted turn instead of writing a second one or
        # starting a second inference. Claim before any turn is written; record
        # on every success, release on failure so a genuine resend still runs.
        dedupe = callbacks.get("chat_turn_dedupe")
        idempotency_key = str(body.get("idempotency_key", "")).strip()
        if dedupe is not None:
            claim = dedupe.claim(actor, requested_conversation, idempotency_key)
            if claim.replay is not None:
                return _payload(claim.replay)
            if claim.in_flight:
                return _payload(error=in_flight_error(), status=IN_FLIGHT_STATUS)

        def accept(payload):
            if dedupe is not None:
                dedupe.complete(
                    actor, requested_conversation, idempotency_key, payload
                )
            return _payload(payload)

        def release():
            if dedupe is not None:
                dedupe.abandon(actor, requested_conversation, idempotency_key)

        # Resolving the selection needs the conversation first, because "not
        # specified" falls back to what this chat already searches before it
        # falls back to the account preference.
        try:
            existing = (
                store.ensure_conversation(actor, requested_conversation)
                if requested_conversation and not temporary else None
            )
        except (AttributeError, RagChatError) as error:
            release()
            return chat_failure(error, "ai_chat_failed", 400)
        collection_ids, collections_chosen = resolve_collections(
            body, preference, existing
        )
        requested_agent_id = str(body.get("agent_id", "")).strip()
        proposal = agent_proposal(actor, message, requested_agent_id)
        if requested_agent_id and proposal is None:
            # Every other early exit releases the claim; this one used to return
            # 400 while leaving the turn marked in-flight, so the client's resend
            # collided with a phantom in-flight turn instead of retrying.
            release()
            return _payload(
                error={
                    "code": "agent_unavailable",
                    "message": "The selected agent is unavailable, disabled, or not owned by this account.",
                },
                status=400,
            )
        if proposal is not None:
            answer = proposal_text(proposal)
            proposal_message = {
                "role": "assistant",
                "content": answer,
                "citations": [],
                "metadata": {
                    "source": "agent-router",
                    "proposed_agent_task": proposal,
                    "approval_required": True,
                },
            }
            if temporary:
                return accept({
                    "conversation_id": "",
                    "message": proposal_message,
                    "model": model,
                    "provider": "agent-router",
                    "temporary": True,
                    "proposed_agent_task": proposal,
                    "approval_required": True,
                })
            # The persisted proposal path writes to the store, which can raise
            # exactly as the main answer path can. Without this guard a store
            # error here escaped uncaught: an HTTP 500 that also left the claim
            # in-flight, so the resend got 503 rather than a real retry.
            try:
                conversation = existing
                if conversation is None:
                    conversation = store.ensure_conversation(
                        actor, title=message[:100], model=model, collections=collection_ids,
                    )
                conversation = apply_patch(
                    store, actor, conversation,
                    conversation_patch(model, collection_ids, collections_chosen),
                )
                assistant_message = store.add_exchange(
                    actor, conversation["id"], message, answer, [], None, "agent-router",
                )
            except (AttributeError, RagChatError, TypeError, sqlite3.Error) as error:
                release()
                return chat_failure(error, "ai_chat_failed", 400)
            assistant_message["metadata"] = proposal_message["metadata"]
            try:
                # audit is a SQLite write; a lock or write error must release the
                # claim and surface as a server error (release(); raise, mirroring
                # the assistant route), NOT the store path's client-facing 400 -
                # a transient DB lock is not a bad request, and 400 is not a code
                # the client auto-resumes, so the released turn would never retry.
                # (Any exception type is caught: a non-sqlite audit failure must
                # not slip a narrow tuple and re-leak the claim.)
                security.audit(
                    actor, "ai_chat.agent.proposal", "success",
                    target=conversation["id"], remote_addr=request.remote_addr or "",
                    details={
                        "profile": proposal["profile_id"],
                        "profile_version": proposal["profile_version"],
                        "approval_required": True,
                        "auto_run": False,
                    },
                )
            except Exception:
                release()
                raise
            return accept({
                "conversation_id": conversation["id"],
                "message": assistant_message,
                "model": model,
                "provider": "agent-router",
                "proposed_agent_task": proposal,
                "approval_required": True,
            })
        # VD-042's inward half, retrieve-first (#247o). AI Chat cannot read this
        # machine, so a question about it must never reach a provider that would
        # invent a reading - "how hot is my CPU" answered from nothing looks
        # exactly like a measured figure. But when a knowledge collection is
        # active the same words may be answerable from the user's own documents,
        # so retrieval runs first and the appliance-scope gate declines only what
        # the documents could not answer. With no collection active there is
        # nothing to retrieve, so the gate applies up front exactly as before -
        # placed after the agent branch so a named custom agent still runs.
        searchable = searchable_collections(store, actor, collection_ids)
        if not searchable:
            decline = appliance_scope_decline(message)
            if decline is not None:
                try:
                    return accept(record_decline(
                        store, actor, decline, message, model=model,
                        temporary=temporary, conversation=existing,
                        collections=collection_ids,
                    ))
                except (AttributeError, RagChatError, TypeError, sqlite3.Error) as error:
                    release()
                    return chat_failure(error, "ai_chat_failed", 400)
        conversation = None
        try:
            retrieved = store.retrieve(actor, message, searchable, limit=6)
            if searchable and not retrieval_answers_question(message, retrieved):
                # Knowledge is active but nothing in it is about the question.
                # An appliance question now falls back to the deflection rather
                # than to a provider that would fabricate a reading; a general
                # question (decline is None) still answers normally.
                decline = appliance_scope_decline(message)
                if decline is not None:
                    return accept(record_decline(
                        store, actor, decline, message, model=model,
                        temporary=temporary, conversation=existing,
                        collections=collection_ids,
                    ))
            if temporary:
                history = [
                    {
                        "role": item.get("role"),
                        "content": str(item.get("content", ""))[:3000],
                    }
                    for item in body.get("history", [])[-8:]
                    if isinstance(item, dict)
                    and item.get("role") in {"user", "assistant"}
                ]
                memory = grounding_memories(message)
                retrieval = grounded_retrieval(
                    actor, searchable, retrieved, memory
                )
                result = inference.answer(
                    message, model=model, history=history, retrieved=retrieved,
                    memories=memory["memories"],
                )
                return accept({
                    "conversation_id": "",
                    "message": {
                        "role": "assistant", "content": result["answer"],
                        "citations": citations_for(retrieved),
                        "retrieval": retrieval, "model": result["model"],
                    },
                    "model": result["model"], "provider": result["provider"],
                    "temporary": True, "retrieval": retrieval,
                })
            conversation = existing
            history = (
                store.messages(actor, conversation["id"])
                if conversation else []
            )
            if conversation is None:
                conversation = store.ensure_conversation(
                    actor, title=message[:100], model=model, collections=collection_ids,
                )
            # The knowledge selection is the user's choice and is recorded now.
            # The model is not: it is a record of what answered here, and
            # nothing has answered yet.
            conversation = apply_patch(
                store, actor, conversation,
                conversation_patch("", collection_ids, collections_chosen),
            )
            store.add_message(actor, conversation["id"], "user", message)
            memory = grounding_memories(message)
            retrieval = grounded_retrieval(actor, searchable, retrieved, memory)
            result = inference.answer(
                message, model=model, history=history, retrieved=retrieved,
                memories=memory["memories"],
            )
            citations = citations_for(retrieved)
            assistant_message = store.add_message(
                actor, conversation["id"], "assistant", result["answer"], citations,
                retrieval, result["model"],
            )
            # Something answered, so the conversation may now say what.
            conversation = apply_patch(
                store, actor, conversation,
                conversation_patch(result["model"] or model, [], False),
            )
        except (
            AttributeError, ChatInferenceError, RagChatError, TypeError,
            sqlite3.Error,
        ) as error:
            release()
            if not temporary and conversation is not None:
                try:
                    failure_message = store.add_message(
                        actor,
                        conversation["id"],
                        "assistant",
                        "This request failed: {}".format(str(error)),
                        model=model,
                        # Recorded, never inferred. A model may legitimately
                        # write "this request failed", so the text cannot be
                        # the marker; without a stored one, a reload showed a
                        # failure as though it were an answer and the Retry
                        # affordance disappeared with the page.
                        metadata={
                            "failed": True,
                            "error_code": getattr(error, "code", "ai_chat_failed"),
                            "failed_model": model,
                        },
                    )
                except (AttributeError, RagChatError, TypeError, sqlite3.Error):
                    return chat_failure(error, "ai_chat_failed", 400)
                security.audit(
                    actor, "ai_chat.message", "failure",
                    target=conversation["id"], remote_addr=request.remote_addr or "",
                    details={
                        "model": model,
                        "code": getattr(error, "code", "ai_chat_failed"),
                        "saved": True,
                    },
                )
                return chat_failure(
                    error, "ai_chat_failed", 400,
                    conversation_id=conversation["id"],
                    message_record=failure_message,
                )
            return chat_failure(error, "ai_chat_failed", 400)
        security.audit(
            actor, "ai_chat.message", "success",
            target=conversation["id"], remote_addr=request.remote_addr or "",
            details={
                "model": result["model"], "citations": len(citations), **retrieval,
            },
        )
        return accept({
            "conversation_id": conversation["id"],
            "message": assistant_message,
            "model": result["model"], "provider": result["provider"],
            "retrieval": retrieval,
        })
