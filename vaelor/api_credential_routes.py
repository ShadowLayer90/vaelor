"""Managed-credential CRUD routes, split out of the setup-route module.

These administrator-only routes create, test, activate, inspect, re-model and
delete the stored provider credentials the Assistant and model-download paths
use. They were carved out of ``api_assistant_setup_routes`` to keep that module
under the 1,000-line production ceiling (VD-111 follow-up); the behaviour is
unchanged. The one helper they share with the setup routes -
``start_model_calibration`` - is passed in rather than duplicated, so a newly
activated or re-modelled credential is still measured in the background exactly
as before.
"""

from __future__ import annotations

import json

from flask import g, request

from .api_common import (
    ApiContext,
    CREDENTIAL_BROKER_UNAVAILABLE,
    payload as _payload,
)
from .credential_broker import CredentialError


def register_credential_routes(context: ApiContext, start_model_calibration) -> None:
    """Register the /credentials routes on the shared API v2 blueprint.

    ``start_model_calibration`` is the setup-route closure that measures the
    active model behind the response; it is threaded in so activation and model
    selection here trigger the same background calibration they always did.
    """
    blueprint = context.blueprint
    callbacks = context.callbacks
    security = context.security
    require_auth = context.require_auth

    def credential_visible_to_actor(broker, credential_id):
        return any(
            item.get("id") == credential_id
            for item in broker.list(g.auth_session.username)
        )

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
            return _payload(
                {
                    "credentials": broker.list(g.auth_session.username),
                    "capabilities": broker.capabilities(),
                }
            )
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
                    broker.activate(credential["id"], "deployment-agent")
                    credential["active_for"] = ["deployment-agent"]
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
                if replacement is None:
                    broker.activate(credential["id"], "deployment-agent")
                    credential["active_for"] = ["deployment-agent"]
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
                raise CredentialError("Credential was not found.")
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
                raise CredentialError("Credential was not found.")
            credentials = {
                item["id"]: item for item in broker.list()
            }
            credential = credentials.get(credential_id)
            if credential is None or credential.get("provider") not in {
                "openai", "openai-compatible"
            }:
                raise CredentialError(
                    "Choose a saved OpenAI or OpenAI-compatible Assistant connection."
                )
            result = broker.test(credential_id)
            if not result.get("ok"):
                raise CredentialError(
                    result.get("message", "The Assistant connection test failed.")
                )
            if credential.get("provider") == "openai-compatible":
                available = broker.models(credential_id)
                if available.get("selection_required"):
                    raise CredentialError(
                        "This server provides multiple chat models. Choose and test a model before using it."
                    )
            broker.activate(credential_id, "deployment-agent")
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
            details={"purpose": "deployment-agent"},
        )
        start_model_calibration()
        return _payload({
            "credential_id": credential_id,
            "active_for": ["deployment-agent"],
            "connection_test": result,
        })

    @blueprint.get("/credentials/<credential_id>/models")
    @require_auth("administrator")
    def credential_models(credential_id):
        broker = callbacks.get("credential_broker")
        try:
            if not credential_visible_to_actor(broker, credential_id):
                raise CredentialError("Credential was not found.")
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
                raise CredentialError("Credential was not found.")
            result = broker.select_model(credential_id, body.get("model", ""))
            broker.activate(credential_id, "deployment-agent")
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
        # A different model is a different profile, so measure the new one.
        start_model_calibration()
        return _payload({
            **result,
            "active_for": ["deployment-agent"],
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
            if not credential_visible_to_actor(broker, credential_id):
                raise CredentialError("Credential was not found.")
            result = broker.delete(credential_id)
        except CredentialError as error:
            return _payload(
                error={"code": "credential_delete_failed", "message": str(error)}, status=400
            )
        deleted = result.get("deleted", False) if isinstance(result, dict) else bool(result)
        if not deleted:
            return _payload(
                error={"code": "credential_not_found", "message": "Credential was not found."},
                status=404,
            )
        security.audit(
            g.auth_session.username, "credential.delete", "success",
            target=credential_id, remote_addr=request.remote_addr or "",
        )
        return _payload({"deleted": True})
