"""Independent AI Chat, model selection, and local RAG collection routes."""

from __future__ import annotations

import base64
import binascii
import sqlite3

from flask import g, request

from .api_common import ApiContext, payload as _payload
from .chat_connections import LOCAL_BY_ADDRESS, LOCALITY_UNKNOWN, describe_connections
from .copilot_setup import hardware_inventory
from .credential_listing import redact_inference_status_for_role
from .gpu_cluster_mode_state import ClusterModeStore
from .gpu_serving_target import deployment_unload_cause
from .inference_status import cluster_on_this_adapter, inference_status
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
    wants_whole_document,
)
from .chat_agent_proposals import agent_proposal, proposal_text
from .api_chat_thinking_routes import thinking_choice
from .chat_inference import ChatInferenceError
from .chat_turn_dedupe import IN_FLIGHT_STATUS, in_flight_error
from .document_text import SUPPORTED_EXTENSIONS
from .rag_chat import AGENT_ROUTER_AUTHOR, MAX_DOCUMENT_BYTES, RagChatError
from .session_affinity import conversation_session_key, opening_session_key


#: What the active connection says about where prompts go while the GPU
#: cluster serves AI Chat (its address is the controller's balancer).
CLUSTER_ACTIVE_LOCALITY = (
    "The GPU cluster answers this connection; its replicas can run on other "
    "machines in the cluster, so prompts may leave this appliance for them."
)


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

    def conversation_patch(collection_ids, collections_chosen):
        """Only write the knowledge selection back when the client chose it.

        Sending a request used to overwrite the conversation's collections with
        whatever the client happened to hold, so one visit to a chat created
        before a collection existed replayed its empty list and made the
        setting permanent. An unspecified selection now leaves the stored one
        untouched.

        The model is not written at all: which model answered in a
        conversation is derived from its turns by the store (ACC-116), so a
        model that produced nothing cannot relabel the chat.
        """
        return {"collections": collection_ids} if collections_chosen else {}

    def apply_patch(store, actor, conversation, patch):
        return store.update_conversation(actor, conversation["id"], patch) if patch else conversation

    def chat_failure(
        error, fallback_code: str, fallback_status: int, **details,
    ):
        # The model the request actually went to, when inference got that far,
        # so the client blames that model and not whatever its picker shows.
        if getattr(error, "model", ""):
            details.setdefault("model", error.model)
        return _payload(
            error={
                "code": getattr(error, "code", fallback_code),
                "message": str(error),
                **details,
            },
            status=getattr(error, "status", fallback_status),
        )

    def turn_metadata(result):
        """What a stored answer says about itself: its timing and, when the
        provider returned one, its thinking summary (VD-209 item 4)."""
        metadata = {"performance": result.get("performance") or {}}
        if result.get("thinking"):
            metadata["thinking"] = result["thinking"]
        return metadata

    def cluster_credential_id():
        """The credential the GPU cluster serves through, from the mode file; ""."""
        modes = callbacks.get("cluster_mode_store") or ClusterModeStore()
        try:
            return str(getattr(modes.read(), "cluster_credential_id", "") or "")
        except Exception:  # noqa: BLE001 - absence-ok: no cluster credential, no relabel
            return ""

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

    @blueprint.get("/ai-chat/setup")
    @require_auth("operator")
    def chat_setup():
        store, inference = services()
        if store is None or inference is None:
            return _payload(error={"code": "ai_chat_unavailable", "message": "AI Chat is unavailable."}, status=503)
        try:
            # VD-210 owner rule: AI Chat's choices, never the Assistant's NPU
            # model (`ChatInference.choices`). A Pi's shared CPU model and the
            # Z2's GPU model carry no NPU marker and stay listed.
            connections = inference.choices()
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
        try:
            assignment_refusal = inference.assignment_refusal()
            # VD-210: per connection - only this machine's own GPU model is
            # refused while the cluster holds the GPU; the picker greys that
            # one row, with this sentence beside it, and nothing else.
            connection_refusals = {
                str(item.get("id")): refusal for item in connections
                for refusal in [inference.assignment_refusal(str(item.get("id") or ""))]
                if refusal
            }
        except (AttributeError, TypeError):
            assignment_refusal, connection_refusals = "", {}
        # Only the connections the owner added, and Vaelor's own managed
        # serving. Nothing here looks for AI servers on this machine or the
        # network: a server enters Vaelor only when the owner adds it (VD-137).
        described = describe_connections(connections)
        # ACC-101: the active connection is the DESCRIBED one, so it carries the
        # locality `describe_connections` decided (`local`, `local_source`,
        # `local_reason`). It was the raw broker row, which carries none, so the
        # Details panel's "prompts never leave the machine" note - keyed on
        # `local_source` - could never appear. One answer, read by both.
        cluster_credential = cluster_credential_id()
        if active is not None:
            active = next(
                (item for item in described if item.get("id") == active.get("id")),
                active,
            )
            # When AI Chat's connection IS the cluster's (the mode file's
            # credential), its loopback address is the controller's balancer,
            # which forwards to replicas on other machines: a loopback address
            # does not make it local, so it is not called that (w2-creds
            # review). Asked of the mode file, not of a refusal: since VD-210
            # the owner may pick any other connection while clustered.
            if (active.get("id") and active.get("id") == cluster_credential
                    and active.get("local_source") == LOCAL_BY_ADDRESS):
                active.update(
                    local=None, local_source=LOCALITY_UNKNOWN,
                    local_reason=CLUSTER_ACTIVE_LOCALITY,
                )
        return _payload({
            # Every connection, each with `local` and how that was decided.
            # The privacy grouping the picker draws is a presentation of
            # `local`, not a second code path: `model.deploy` already
            # registers the appliance's own server as a connection.
            "connections": described,
            "active_connection": active,
            # Why AI Chat's connection cannot be changed at all right now, or
            # "". Since VD-210 nothing refuses the whole picker (it is "" in
            # Mode B too); the per-connection refusals below say which row.
            "assignment_refusal": assignment_refusal,
            # VD-210: the connections AI Chat cannot take right now, each with
            # its sentence (this machine's own GPU model while clustered).
            "connection_refusals": connection_refusals,
            # The connection the GPU cluster serves through (the mode file's), or
            # "": the picker groups that one under "Cluster" (VD-210).
            "cluster_credential_id": cluster_credential,
            "preference": store.preference(g.auth_session.username),
            "collections": store.collections(g.auth_session.username),
            "limits": {
                "document_bytes": MAX_DOCUMENT_BYTES,
                "collections": 50,
                "document_extensions": list(SUPPORTED_EXTENSIONS),
            },
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
                # Every connection, NPU model included: this describes what each
                # engine holds, not what AI Chat may pick (`choices`, VD-210).
                connections = inference.connections()
            except ChatInferenceError:
                connections = []
        # Which GPU cluster deployments run a server here, so a clustered GPU
        # is described as the cluster's and never as llama.cpp's (ACC-114).
        # An unreadable store leaves the mode file to say the GPU is clustered.
        operations = callbacks.get("cluster_operations")
        try:
            cluster_records = operations.store.list_pooled_deployments()
        except (AttributeError, OSError, ValueError, sqlite3.Error):
            cluster_records = None
        cluster_mode = bool(
            inference is not None and getattr(inference, "cluster_mode_active", None)
            and inference.cluster_mode_active()
        )
        # Who unloaded the cluster deployment on this adapter, asked of the one
        # unload-cause rule (VD-136) rather than decided here: it says whether
        # the next AI Chat request loads it again.
        unload_cause = ""
        placed = cluster_on_this_adapter(cluster_records)
        if placed is not None and str(placed.get("state") or "") == "unloaded":
            unload_cause = deployment_unload_cause(
                getattr(inference, "broker", None), ClusterModeStore().read(),
                str(placed.get("name") or ""),
            )
        # VD-204: an engine's failure sentence can name a connection or its
        # address; a viewer gets the fixed sentence (`credential_listing`).
        return _payload(redact_inference_status_for_role(inference_status(
            hardware,
            connections=connections,
            probe=lambda connection: probe_connection(dict(connection)),
            # The listed records are redacted (no base_url); this resolves the
            # active managed-local lease so the CPU engine can be probed and show
            # the deployed model rather than model=null (#205 Step 3, VD-071).
            resolve_local=(
                inference.active_local_connection if inference is not None else None
            ),
            cluster_deployments=cluster_records,
            cluster_mode=cluster_mode,
            cluster_unload_cause=unload_cause,
        ), g.auth_session.role))

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
        raw = None
        encoded = body.get("content_b64")
        if encoded is not None:
            # base64 inflates ~33%; refuse an oversize body before decoding it
            # into memory. The decoded bytes are re-checked against the limit.
            if not isinstance(encoded, str) or len(encoded) > MAX_DOCUMENT_BYTES // 3 * 4 + 8:
                return _payload(error={
                    "code": "document_rejected",
                    "message": "The uploaded file is too large.",
                }, status=400)
            try:
                raw = base64.b64decode(encoded, validate=True)
            except (binascii.Error, ValueError):
                return _payload(error={
                    "code": "document_rejected",
                    "message": "The uploaded file could not be decoded.",
                }, status=400)
        try:
            document = store.ingest(
                g.auth_session.username, collection_id, body.get("name", ""),
                body.get("content", ""), body.get("media_type", "text/plain"),
                raw=raw,
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

    @blueprint.get("/ai-chat/conversations/<conversation_id>")
    @require_auth("operator")
    def chat_conversation(conversation_id):
        """One conversation by id, archived or not (ACC-108).

        The list holds the newest hundred unarchived chats, so an address
        naming an archived or older one could not be reopened from it, and the
        screen said "New chat" while sends went on into the hidden one.
        """
        store, _ = services()
        try:
            return _payload(store.ensure_conversation(g.auth_session.username, conversation_id))
        except (AttributeError, RagChatError) as error:
            return _payload(error={"code": "chat_not_found", "message": str(error)}, status=404)

    @blueprint.get("/ai-chat/conversations/<conversation_id>/messages")
    @require_auth("operator")
    def chat_messages(conversation_id):
        store, _ = services()
        try:
            return _payload(store.messages(g.auth_session.username, conversation_id))
        except (AttributeError, RagChatError) as error:
            return _payload(error={"code": "chat_not_found", "message": str(error)}, status=404)

    @blueprint.get("/ai-chat/conversations/<conversation_id>/export")
    @require_auth("operator")
    def chat_export(conversation_id):
        """Every turn for the Markdown export, bounded, saying if it was cut (ACC-112)."""
        store, _ = services()
        try:
            return _payload(store.export_messages(g.auth_session.username, conversation_id))
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
            result = store.delete_conversation(g.auth_session.username, conversation_id)
        except (AttributeError, RagChatError) as error:
            return _payload(error={"code": "chat_not_found", "message": str(error)}, status=404)
        # W7 retest: deleting an Assistant conversation was audited and deleting
        # an AI Chat one was not (LESSONS 6: two surfaces, one kind of act).
        security.audit(
            g.auth_session.username, "ai_chat.conversation.delete", "success",
            target=conversation_id, remote_addr=request.remote_addr or "",
        )
        return _payload(result)

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
        # The model the picker shows, exactly as a send resolves it. This used
        # to be the conversation's stored model, so a regenerate went to a
        # model the reader could no longer see - in cluster mode one vLLM did
        # not serve, 404 - and the failure was blamed on the picker's model.
        model = str(
            body.get("model") or store.preference(g.auth_session.username)["model"]
        ).strip()
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
            # exactly as it does to the first send: retrieve first (including the
            # same document-directed `lead_on_empty` fallback, so "summarize it"
            # regenerates from the document instead of falling through to a
            # decline), and let the appliance-scope gate decline only what the
            # documents could not answer. Without this an edited prompt was a way
            # around the gate.
            retrieved = store.retrieve(
                g.auth_session.username, prepared["prompt"], searchable, limit=6,
                lead_on_empty=wants_whole_document(prepared["prompt"]),
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
                        "model": model,
                        "provider": APPLIANCE_SCOPE_PROVIDER,
                    })
            memory = grounding_memories(prepared["prompt"])
            retrieval = grounded_retrieval(
                g.auth_session.username, searchable, retrieved, memory
            )
            result = inference.answer(
                prepared["prompt"], model=model,
                history=prepared["history"], retrieved=retrieved,
                memories=memory["memories"],
                session_key=conversation_session_key(
                    g.auth_session.username, conversation_id
                ),
                thinking=thinking_choice(store, body, g.auth_session.username),
            )
            assistant_message = store.add_message(
                g.auth_session.username, conversation_id, "assistant",
                result["answer"], citations_for(retrieved), retrieval,
                result["model"], metadata=turn_metadata(result),
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
        proposal = agent_proposal(callbacks, actor, message, requested_agent_id)
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
                    "provider": AGENT_ROUTER_AUTHOR,
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
                        actor, title=message[:100], collections=collection_ids,
                    )
                conversation = apply_patch(
                    store, actor, conversation,
                    conversation_patch(collection_ids, collections_chosen),
                )
                assistant_message = store.add_exchange(
                    actor, conversation["id"], message, answer, [], None,
                    AGENT_ROUTER_AUTHOR,
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
                "provider": AGENT_ROUTER_AUTHOR,
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
            # `lead_on_empty`: a document-directed ask ("summarize it") shares no
            # words with the document, so keyword FTS returns nothing and the
            # model reports it sees none. For those asks only, fall back to the
            # document's leading chunks - a plain keyword search that matched
            # nothing must still report zero passages, not have a document
            # injected under it (`test_a_search_that_matched_nothing_says_so`).
            retrieved = store.retrieve(
                actor, message, searchable, limit=6,
                lead_on_empty=wants_whole_document(message),
            )
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
                        # A failure notice the client is holding is not an
                        # answer; `conversational_history` leaves it out.
                        "failed": item.get("failed") is True,
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
                    # A temporary chat has no id: its first message names it
                    # for as long as it lasts (VD-158) - read off everything
                    # the client sent, not the last eight turns the model sees.
                    session_key=opening_session_key(actor, [
                        *(body.get("history") if isinstance(body.get("history"), list) else []),
                        {"role": "user", "content": message},
                    ]),
                    thinking=thinking_choice(store, body, actor),
                )
                return accept({
                    "conversation_id": "",
                    "message": {
                        "role": "assistant", "content": result["answer"],
                        "citations": citations_for(retrieved),
                        "retrieval": retrieval, "model": result["model"],
                        "metadata": turn_metadata(result),
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
                    actor, title=message[:100], collections=collection_ids,
                )
            # The knowledge selection is the user's choice and is recorded now.
            # The model is not recorded anywhere but on the turn it writes.
            conversation = apply_patch(
                store, actor, conversation,
                conversation_patch(collection_ids, collections_chosen),
            )
            store.add_message(actor, conversation["id"], "user", message)
            memory = grounding_memories(message)
            retrieval = grounded_retrieval(actor, searchable, retrieved, memory)
            result = inference.answer(
                message, model=model, history=history, retrieved=retrieved,
                memories=memory["memories"],
                session_key=conversation_session_key(actor, conversation["id"]),
                thinking=thinking_choice(store, body, actor),
            )
            citations = citations_for(retrieved)
            assistant_message = store.add_message(
                actor, conversation["id"], "assistant", result["answer"], citations,
                retrieval, result["model"], metadata=turn_metadata(result),
            )
        except (
            AttributeError, ChatInferenceError, RagChatError, TypeError,
            sqlite3.Error,
        ) as error:
            release()
            if not temporary and conversation is not None:
                # The model the request really went to, which is the picker's
                # choice or - with none chosen - the one the server offered.
                failed_model = getattr(error, "model", "") or model
                try:
                    failure_message = store.add_message(
                        actor,
                        conversation["id"],
                        "assistant",
                        "This request failed: {}".format(str(error)),
                        model=failed_model,
                        # Recorded, never inferred. A model may legitimately
                        # write "this request failed", so the text cannot be
                        # the marker; without a stored one, a reload showed a
                        # failure as though it were an answer and the Retry
                        # affordance disappeared with the page.
                        metadata={
                            "failed": True,
                            "error_code": getattr(error, "code", "ai_chat_failed"),
                            "failed_model": failed_model,
                        },
                    )
                except (AttributeError, RagChatError, TypeError, sqlite3.Error):
                    return chat_failure(error, "ai_chat_failed", 400)
                security.audit(
                    actor, "ai_chat.message", "failure",
                    target=conversation["id"], remote_addr=request.remote_addr or "",
                    details={
                        "model": failed_model,
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
