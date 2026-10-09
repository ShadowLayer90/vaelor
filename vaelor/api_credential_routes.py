"""Managed-credential CRUD routes, split out of the setup-route module.

These administrator-only routes create, test, activate, inspect, re-model and
delete the stored provider credentials AI Chat and the model-download path use.
They were carved out of ``api_assistant_setup_routes`` to keep that module
under the 1,000-line production ceiling (VD-111 follow-up).

**None of them gives the Assistant a connection** (VD-049 / VD-201 item 3).
Adding a connection, "Use in AI Chat" and choosing a model act on AI Chat's
lease through `ChatInference.activate`, so the Mode B lock answers its own
sentence; and the vault refuses the Assistant's purpose for anything but the
Assistant's own install, and AI Chat's for the Assistant's NPU model
(`model_credential_roles`, VD-202). They used to activate
the Assistant and then measure its new model in the background; with the
Assistant's model no longer theirs to change, that measurement left with it.
"""

from __future__ import annotations

import json
import logging

from flask import g, request

from .api_common import (
    ApiContext,
    CREDENTIAL_BROKER_UNAVAILABLE,
    payload as _payload,
)
from .agent_pool_operations import AGENT_ENDPOINT_PREFIX
from .agent_reconcile import apply_agent_endpoint_keys
from .api_cluster_agent_routes import agent_store_from
from .chat_inference import ChatInference, ChatInferenceError
from .credential_broker import CREDENTIAL_NOT_FOUND, CredentialError
from .credential_listing import (
    ASSISTANT_KEY_IN_USE, MODEL_SET_BY_VAELOR,
    connection_listing, delete_refusal, read_references, resting_test_answer,
)
from .gpu_cluster_mode import ClusterModeStore
from .gpu_serving_target import AI_CHAT_PURPOSE
from .hosted_providers import HOSTED_COMPATIBLE, HOSTED_KINDS
from .model_credential_roles import ASSISTANT_PURPOSE, MANAGED_LOCAL_PREFIX, ai_chat_may_use
from .inference_gateway import cluster_model_rest_state
from .llm_server_state import LLM_SERVER_ENDPOINT_ID, enqueue_apply

LOGGER = logging.getLogger(__name__)


def register_credential_routes(context: ApiContext) -> None:
    """Register the /credentials routes on the shared API v2 blueprint."""
    blueprint = context.blueprint
    callbacks = context.callbacks
    security = context.security
    require_auth = context.require_auth

    def apply_endpoint_keys(endpoint_id, action) -> dict:
        """Ask for the LLM Server's running gate to be re-keyed, and say so.

        The gate renders its key SET into its config, so a key minted, rotated
        or revoked here is not admitted (or not refused) until the gate is
        re-applied. This enqueues the apply a toggle does and answers what it
        managed - ``queued`` with the job, or ``pending`` when no job could be
        queued and the executor's reconcile is what re-keys it - so the
        console can say "within about 30 seconds" rather than "immediately".
        Never raises: it runs after the broker already minted or rotated, and
        the one-time plaintext must still reach the caller
        (`llm_server_state.enqueue_apply`). A deployed agent's gate is re-keyed
        in this request (ACC-070, `agent_reconcile.apply_agent_endpoint_keys`),
        ``applied`` or ``pending`` on its 30 s reconcile; any other endpoint's
        gate answers nothing here.
        """
        if endpoint_id.startswith(AGENT_ENDPOINT_PREFIX):
            # W6-1: the store is built inside the try. Built outside it, a store
            # that raised on first use (makedirs, SQLite init) turned an
            # already-made key change into a 500 with no audit row.
            try:
                return apply_agent_endpoint_keys(
                    endpoint_id, store=agent_store_from(callbacks),
                    broker=callbacks.get("credential_broker"),
                    bridge=callbacks.get("hardware_bridge_client"),
                )
            except Exception as error:  # noqa: BLE001 - the 30 s reconcile re-keys it
                LOGGER.warning(
                    "The agent gate for %s could not be re-keyed now; its reconcile "
                    "will: %s: %s", endpoint_id, type(error).__name__, error,
                )
                return {"apply": "pending"}
        if endpoint_id != LLM_SERVER_ENDPOINT_ID:
            return {}
        job_id = enqueue_apply(
            callbacks.get("job_store"), g.auth_session.username, action
        )
        return {"job_id": job_id, "apply": "queued" if job_id else "pending"}

    def audit_key_change(action, outcome, target, endpoint_id, applied=None, reason=""):
        """One audit row per endpoint-key change, refused or made (W6-1).

        ``applied`` is what the gate re-key answered; ``None`` means it raised,
        which is recorded as ``apply: failed``. Never carries a key.
        """
        details = {"endpoint_id": endpoint_id}
        if outcome == "success":
            details["job_id"] = (applied or {}).get("job_id") or ""
            details["apply"] = (applied or {}).get("apply") or (
                "failed" if applied is None else ""
            )
        if reason:
            details["reason"] = reason[:300]
        security.audit(
            g.auth_session.username, action, outcome,
            target=target, remote_addr=request.remote_addr or "", details=details,
        )

    def apply_and_audit(action, target, endpoint_id, verb):
        """Re-key the gate, and audit the change whatever the re-key does.

        The audit sits in a ``finally``: the broker has already changed the
        key, so a row must exist even if the re-key breaks its never-raise
        promise. Answers what the re-key said, or ``{"apply": "pending"}``.
        """
        applied = None
        try:
            applied = apply_endpoint_keys(endpoint_id, verb)
        except Exception as error:  # noqa: BLE001 - the key changed; the reconcile re-keys
            LOGGER.warning(
                "Re-keying %s after %s failed: %s: %s",
                endpoint_id, verb, type(error).__name__, error,
            )
        finally:
            audit_key_change(action, "success", target, endpoint_id, applied)
        return applied if applied is not None else {"apply": "pending"}

    def credential_visible_to_actor(broker, credential_id):
        return any(
            item.get("id") == credential_id
            for item in broker.list(g.auth_session.username)
        )

    def resting_answer(broker, credential_id):
        """What Test says for the cluster model's credential while it rests.

        Read-only (`cluster_model_rest_state` wakes nothing). ``None`` - so the
        real test runs - for any other credential or when nothing can be read.
        """
        manager = callbacks.get("cluster_manager")
        if manager is None:
            return None
        modes = callbacks.get("cluster_mode_store") or ClusterModeStore()
        try:
            cluster_credential = str(getattr(modes.read(), "cluster_credential_id", "") or "")
        except Exception:  # noqa: BLE001 - absence-ok: the real test still runs
            return None
        if not cluster_credential or cluster_credential != credential_id:
            return None
        return resting_test_answer(
            credential_id, cluster_model_rest_state(broker, modes, manager.store)
        )

    def chat_runtime(broker):
        """AI Chat's one assignment path (`ChatInference.activate`).

        The shipping runtime when the control plane wired one; otherwise one
        built over this broker, which reads the same serving-mode file, so no
        route here can step round the Mode B refusal.
        """
        return callbacks.get("chat_inference") or ChatInference(broker)

    def ai_chat_holder(broker):
        """The id AI Chat is assigned to, ``""`` for none, ``None`` if unread."""
        try:
            listing = broker.list()
        except CredentialError:
            return None
        return next((
            str(item.get("id")) for item in listing
            if AI_CHAT_PURPOSE in (item.get("active_for") or [])
            # VD-210: a stale lease on the Assistant's NPU model is no model.
            and ai_chat_may_use(item)
        ), "")

    def adopt_for_ai_chat(broker, credential_id):
        """Give a just-added connection to AI Chat when AI Chat has none.

        Answers ``(assigned, refusal)``. The connection is saved and listed
        whatever this answers: AI Chat already holding one, or a holder that
        could not be read, leaves it unassigned for the owner to pick, and the
        cluster holding AI Chat (Mode B) answers its own sentence.
        """
        chat = chat_runtime(broker)
        refusal = chat.assignment_refusal(credential_id)
        if refusal:
            return False, refusal
        if ai_chat_holder(broker) != "":
            return False, ""
        try:
            chat.activate(credential_id)
        except ChatInferenceError as error:
            return False, str(error)
        return True, ""

    def ai_chat_refused(error):
        """The answer for an AI Chat assignment the chat runtime refused."""
        return _payload(
            error={"code": error.code, "message": str(error)}, status=error.status,
        )

    def references():
        """What each store names, read fresh (`credential_listing.read_references`)."""
        return read_references(callbacks, g.auth_session.username)

    @blueprint.get("/credentials")
    @require_auth("administrator")
    def credential_list():
        broker = callbacks.get("credential_broker")
        if broker is None:
            return _payload(
                error={"code": "credential_broker_unavailable", "message": CREDENTIAL_BROKER_UNAVAILABLE},
                status=503,
            )
        try:
            # Outbound connections only, each saying who manages it; inbound
            # endpoint keys are counted, not listed (ACC-133).
            listing = connection_listing(
                broker.list(g.auth_session.username), references()
            )
            return _payload({**listing, "capabilities": broker.capabilities()})
        except CredentialError as error:
            return _payload(
                error={"code": "credential_broker_unavailable", "message": str(error)},
                status=503,
            )

    @blueprint.post("/credentials")
    @require_auth("administrator", csrf=True)
    def credential_create():
        broker = callbacks.get("credential_broker")
        if broker is None:
            return _payload(
                error={"code": "credential_broker_unavailable", "message": CREDENTIAL_BROKER_UNAVAILABLE},
                status=503,
            )
        body = request.get_json(silent=True) or {}
        credential = None
        replacement = None

        def adopt_new_connection(item):
            # VD-201 item 3: a new connection becomes AI Chat's when AI Chat
            # has none, and is never the Assistant's.
            assigned, refusal = adopt_for_ai_chat(broker, item["id"])
            item["active_for"] = [AI_CHAT_PURPOSE] if assigned else []
            item["ai_chat_refusal"] = refusal

        try:
            replacement_id = str(body.get("credential_id", "")).strip()
            if replacement_id:
                replacement = next(
                    (
                        item for item in broker.list(g.auth_session.username)
                        if item["id"] == replacement_id
                    ),
                    None,
                )
                if replacement is None:
                    raise CredentialError("The credential being replaced was not found.")
                if replacement["provider"] != str(body.get("provider", "")).strip():
                    raise CredentialError(
                        "A credential cannot change providers during replacement."
                    )
                if ASSISTANT_PURPOSE in (replacement.get("active_for") or []):
                    # VD-202: the copy is a new added connection, which the
                    # Assistant's lease can no longer take; say so up front.
                    raise CredentialError(ASSISTANT_KEY_IN_USE)
            credential = broker.put(
                body.get("provider", ""),
                body.get("label", ""),
                body.get("secret", ""),
                None,
                g.auth_session.username,
            )
            connection_test = None
            if credential["provider"] == "openai-compatible":
                connection_test = broker.test(credential["id"])
                if not connection_test.get("ok"):
                    broker.delete(credential["id"])
                    raise CredentialError(connection_test.get("message", "Connection test failed."))
                available = broker.models(credential["id"])
                requested_model = ""
                try:
                    requested_model = str(
                        json.loads(str(body.get("secret", ""))).get("model", "")
                    ).strip()
                except (TypeError, json.JSONDecodeError):
                    pass
                if requested_model:
                    broker.select_model(credential["id"], requested_model)
                    credential["selected_model"] = requested_model
                credential["discovered_models"] = available.get("models", [])
                credential["selection_required"] = (
                    not credential.get("selected_model")
                    and bool(credential["discovered_models"])
                )
                if replacement is None and not credential["selection_required"]:
                    adopt_new_connection(credential)
                credential["connection_test"] = connection_test
            elif credential["provider"] in HOSTED_KINDS:
                # VD-206: tested like OpenAI (a failed test keeps nothing), and
                # given to AI Chat only once a model is chosen - a hosted list
                # is a catalogue, and its first entry is not the owner's pick.
                connection_test = broker.test(credential["id"])
                if not connection_test.get("ok"):
                    broker.delete(credential["id"])
                    raise CredentialError(
                        connection_test.get("message") or "The hosted service did not pass its test."
                    )
                try:
                    available = broker.models(credential["id"]).get("models", [])
                except CredentialError:
                    available = None  # read and failed: said, never a zero
                requested_model = str(body.get("model", "") or "").strip()
                if credential["provider"] == HOSTED_COMPATIBLE:
                    try:
                        requested_model = str(
                            json.loads(str(body.get("secret", ""))).get("model", "")
                        ).strip()
                    except (AttributeError, TypeError, json.JSONDecodeError):
                        requested_model = ""
                if requested_model:
                    broker.select_model(credential["id"], requested_model)
                    credential["selected_model"] = requested_model
                credential["discovered_models"] = available
                credential["selection_required"] = not credential.get("selected_model")
                if replacement is None and not credential["selection_required"]:
                    adopt_new_connection(credential)
                credential["connection_test"] = connection_test
            elif credential["provider"] == "huggingface":
                if replacement is None:
                    broker.activate(credential["id"], "model-download")
                    credential["active_for"] = ["model-download"]
            elif credential["provider"] == "openai":
                connection_test = broker.test(credential["id"])
                if not connection_test.get("ok"):
                    broker.delete(credential["id"])
                    raise CredentialError(
                        connection_test.get("message", "OpenAI connection failed.")
                    )
                requested_model = str(body.get("model", "") or "").strip()
                if requested_model:
                    # The optional Model ID the dialog sends (VD-206).
                    broker.select_model(credential["id"], requested_model)
                    credential["selected_model"] = requested_model
                if replacement is None:
                    adopt_new_connection(credential)
                credential["connection_test"] = connection_test
            if replacement is not None:
                selected_model = str(replacement.get("selected_model", "")).strip()
                if selected_model:
                    broker.select_model(credential["id"], selected_model)
                for purpose in replacement.get("active_for", []):
                    broker.activate(credential["id"], purpose)
                broker.delete(replacement["id"])
                credential["replaced_credential_id"] = replacement["id"]
                credential["active_for"] = replacement.get(
                    "active_for", credential.get("active_for", [])
                )
        except CredentialError as error:
            if credential:
                try:
                    broker.delete(credential["id"])
                except CredentialError:
                    pass
            if replacement is not None:
                for purpose in replacement.get("active_for", []):
                    try:
                        broker.activate(replacement["id"], purpose)
                    except CredentialError:
                        pass
            security.audit(
                g.auth_session.username, "credential.store", "failure",
                target=str(body.get("provider", ""))[:80],
                remote_addr=request.remote_addr or "",
            )
            return _payload(
                error={"code": "invalid_credential", "message": str(error)}, status=400
            )
        security.audit(
            g.auth_session.username,
            "credential.rotate" if body.get("credential_id") else "credential.store",
            "success",
            target=credential["id"],
            remote_addr=request.remote_addr or "",
            details={"provider": credential["provider"]},
        )
        return _payload(credential, status=201)

    @blueprint.post("/credentials/<credential_id>/test")
    @require_auth("administrator", csrf=True)
    def credential_test(credential_id):
        broker = callbacks.get("credential_broker")
        if broker is None:
            return _payload(
                error={"code": "credential_broker_unavailable", "message": CREDENTIAL_BROKER_UNAVAILABLE},
                status=503,
            )
        try:
            if not credential_visible_to_actor(broker, credential_id):
                raise CredentialError(CREDENTIAL_NOT_FOUND)
            resting = resting_answer(broker, credential_id)
            if resting is not None:
                # A model resting on purpose is not a failed credential: say
                # which rest it is, and record no test result at all.
                return _payload(resting)
            result = broker.test(credential_id)
        except CredentialError as error:
            return _payload(
                error={"code": "credential_test_failed", "message": str(error)}, status=400
            )
        security.audit(
            g.auth_session.username, "credential.test",
            "success" if result["ok"] else "failure",
            target=credential_id, remote_addr=request.remote_addr or "",
            details={"provider": result["provider"]},
        )
        return _payload(result)

    @blueprint.post("/credentials/<credential_id>/activate")
    @require_auth("administrator", csrf=True)
    def credential_activate(credential_id):
        broker = callbacks.get("credential_broker")
        if broker is None:
            return _payload(
                error={
                    "code": "credential_broker_unavailable",
                    "message": CREDENTIAL_BROKER_UNAVAILABLE,
                },
                status=503,
            )
        try:
            if not credential_visible_to_actor(broker, credential_id):
                raise CredentialError(CREDENTIAL_NOT_FOUND)
            credentials = {
                item["id"]: item for item in broker.list()
            }
            credential = credentials.get(credential_id)
            if credential is None or credential.get("provider") not in {
                "openai", "openai-compatible", *HOSTED_KINDS
            }:
                raise CredentialError(
                    "Choose a saved OpenAI or OpenAI-compatible connection."
                )
            chat = chat_runtime(broker)
            # Asked before the test: a connection the cluster will not let AI
            # Chat take is not worth a round trip to the provider (VD-127).
            if chat.assignment_refusal(credential_id):
                chat.activate(credential_id)
            result = broker.test(credential_id)
            if not result.get("ok"):
                raise CredentialError(
                    result.get("message", "The connection test failed.")
                )
            if credential.get("provider") in {"openai-compatible", *HOSTED_KINDS}:
                available = broker.models(credential_id)
                if available.get("selection_required"):
                    raise CredentialError(
                        "This server provides multiple chat models. Choose and test a model before using it."
                    )
            # VD-049 / VD-201 item 3: "Use in AI Chat", never the Assistant.
            chat.activate(credential_id)
        except ChatInferenceError as error:
            return ai_chat_refused(error)
        except CredentialError as error:
            return _payload(
                error={"code": "credential_activation_failed", "message": str(error)},
                status=400,
            )
        security.audit(
            g.auth_session.username,
            "credential.activate",
            "success",
            target=credential_id,
            remote_addr=request.remote_addr or "",
            details={"purpose": AI_CHAT_PURPOSE},
        )
        return _payload({
            "credential_id": credential_id,
            "active_for": [AI_CHAT_PURPOSE],
            "connection_test": result,
        })

    @blueprint.get("/credentials/<credential_id>/models")
    @require_auth("administrator")
    def credential_models(credential_id):
        broker = callbacks.get("credential_broker")
        try:
            if not credential_visible_to_actor(broker, credential_id):
                raise CredentialError(CREDENTIAL_NOT_FOUND)
            result = broker.models(credential_id)
        except (AttributeError, CredentialError) as error:
            return _payload(
                error={"code": "model_discovery_failed", "message": str(error)},
                status=400,
            )
        return _payload(result)

    @blueprint.patch("/credentials/<credential_id>/model")
    @require_auth("administrator", csrf=True)
    def credential_model_select(credential_id):
        broker = callbacks.get("credential_broker")
        body = request.get_json(silent=True) or {}
        try:
            if not credential_visible_to_actor(broker, credential_id):
                raise CredentialError(CREDENTIAL_NOT_FOUND)
            if str(credential_id).startswith(MANAGED_LOCAL_PREFIX):
                # VD-202: a model Vaelor runs keeps the model its deploy chose;
                # re-pinning the NPU model here would un-mark it as the
                # Assistant's and let AI Chat take it.
                raise CredentialError(MODEL_SET_BY_VAELOR)
            chat = chat_runtime(broker)
            # Asked before the model is recorded: choosing a model here is using
            # it in AI Chat, which the cluster holds in Mode B (VD-127).
            if chat.assignment_refusal(credential_id):
                chat.activate(credential_id)
            result = broker.select_model(credential_id, body.get("model", ""))
            # VD-049 / VD-201 item 3: AI Chat's lease, never the Assistant's.
            chat.activate(credential_id)
        except ChatInferenceError as error:
            return ai_chat_refused(error)
        except (AttributeError, CredentialError) as error:
            return _payload(
                error={"code": "model_selection_failed", "message": str(error)},
                status=400,
            )
        security.audit(
            g.auth_session.username,
            "credential.model.select",
            "success",
            target=credential_id,
            remote_addr=request.remote_addr or "",
            details={"model": result["selected_model"]},
        )
        return _payload({
            **result,
            "active_for": [AI_CHAT_PURPOSE],
        })

    @blueprint.delete("/credentials/<credential_id>")
    @require_auth("administrator", csrf=True)
    def credential_delete(credential_id):
        broker = callbacks.get("credential_broker")
        if broker is None:
            return _payload(
                error={"code": "credential_broker_unavailable", "message": CREDENTIAL_BROKER_UNAVAILABLE},
                status=503,
            )
        try:
            item = next(
                (
                    entry for entry in broker.list(g.auth_session.username)
                    if entry.get("id") == credential_id
                ),
                None,
            )
            if item is None:
                raise CredentialError(CREDENTIAL_NOT_FOUND)
            # ACC-111: an endpoint key, or a secret Vaelor created for a node or
            # a model it runs, is not the owner's connection to disconnect; the
            # sentence says where it is managed instead.
            status, refusal = delete_refusal(item, references())
            if refusal:
                # 409: something uses it (the sentence says what). 503: a store
                # that says what uses it could not be read - fail closed.
                return _payload(
                    error={"code": "credential_managed_elsewhere" if status == 409 else "credential_use_unknown", "message": refusal},
                    status=status,
                )
            result = broker.delete(credential_id)
        except CredentialError as error:
            return _payload(
                error={"code": "credential_delete_failed", "message": str(error)}, status=400
            )
        deleted = result.get("deleted", False) if isinstance(result, dict) else bool(result)
        if not deleted:
            return _payload(
                error={"code": "credential_not_found", "message": CREDENTIAL_NOT_FOUND},
                status=404,
            )
        security.audit(
            g.auth_session.username, "credential.delete", "success",
            target=credential_id, remote_addr=request.remote_addr or "",
        )
        return _payload({"deleted": True})

    @blueprint.post("/endpoints/<endpoint_id>/keys")
    @require_auth("administrator", csrf=True)
    def endpoint_key_mint(endpoint_id):
        broker = callbacks.get("credential_broker")
        if broker is None:
            return _payload(
                error={"code": "credential_broker_unavailable", "message": CREDENTIAL_BROKER_UNAVAILABLE},
                status=503,
            )
        body = request.get_json(silent=True) or {}
        try:
            minted = broker.mint(endpoint_id, body.get("label", ""))
        except CredentialError as error:
            audit_key_change("endpoint.key.mint", "failure", endpoint_id, endpoint_id, reason=str(error))
            return _payload(
                error={"code": "endpoint_key_mint_failed", "message": str(error)},
                status=400,
            )
        # W5-D5: audited after the apply so the row names the job it queued;
        # W6-1: audited whatever the apply does.
        applied = apply_and_audit(
            "endpoint.key.mint", minted["credential_id"], endpoint_id, "mint",
        )
        minted = {**minted, **applied}
        # The plaintext key is in THIS response only - the one-time reveal. No
        # other route ever returns it. A mint whose gate re-key failed still
        # returns it (W6-1): the key exists in the broker and the owner must see
        # it once; ``apply: pending`` says the gate admits it within ~30 s.
        return _payload(minted, status=201)

    @blueprint.post("/endpoints/<endpoint_id>/keys/<credential_id>/rotate")
    @require_auth("administrator", csrf=True)
    def endpoint_key_rotate(endpoint_id, credential_id):
        broker = callbacks.get("credential_broker")
        if broker is None:
            return _payload(
                error={"code": "credential_broker_unavailable", "message": CREDENTIAL_BROKER_UNAVAILABLE},
                status=503,
            )
        try:
            # The endpoint id from the path is passed as the binding guard, so a
            # key can only be rotated under its own endpoint (the S2 boundary).
            rotated = broker.rotate(credential_id, endpoint_id)
        except CredentialError as error:
            audit_key_change("endpoint.key.rotate", "failure", credential_id, endpoint_id, reason=str(error))
            return _payload(
                error={"code": "endpoint_key_rotate_failed", "message": str(error)},
                status=400,
            )
        # The audit trail carries no key - only who rotated which credential.
        applied = apply_and_audit("endpoint.key.rotate", credential_id, endpoint_id, "rotate")
        rotated = {**rotated, **applied}
        # The fresh plaintext key is in THIS response only - a rotate is a reveal.
        return _payload(rotated, status=200)

    @blueprint.delete("/endpoints/<endpoint_id>/keys/<credential_id>")
    @require_auth("administrator", csrf=True)
    def endpoint_key_revoke(endpoint_id, credential_id):
        broker = callbacks.get("credential_broker")
        if broker is None:
            return _payload(
                error={"code": "credential_broker_unavailable", "message": CREDENTIAL_BROKER_UNAVAILABLE},
                status=503,
            )
        try:
            revoked = broker.revoke(credential_id, endpoint_id)
        except CredentialError as error:
            audit_key_change("endpoint.key.revoke", "failure", credential_id, endpoint_id, reason=str(error))
            return _payload(
                error={"code": "endpoint_key_revoke_failed", "message": str(error)},
                status=400,
            )
        applied = apply_and_audit("endpoint.key.revoke", credential_id, endpoint_id, "revoke")
        revoked = {**(revoked or {}), **applied}
        return _payload(revoked, status=200)
