"""Administrator routes for alert-delivery channels and their test-send.

These register on the shared ``/api/v2`` blueprint. They manage the email and
webhook channels a fired automation alert is delivered to, and let an
administrator send a synthetic alert to one channel to verify it before relying
on it. Channel secrets (SMTP password, webhook token) are never held in the
channel row: they go through the credential broker under a per-channel purpose,
and only that purpose reference is stored.
"""

from __future__ import annotations

import time

from flask import g, request

from .alert_channels import (
    AlertChannelError, CHANNEL_NOT_FOUND, channel_purpose, record_results,
)
from .alert_delivery import Senders, deliver_alert
from .api_common import (
    ApiContext,
    CREDENTIAL_BROKER_UNAVAILABLE as _BROKER_DOWN,
    payload as _payload,
)
from .credential_broker import CredentialError

_UNAVAILABLE = {"code": "alert_channels_unavailable", "message": "Alert delivery is unavailable."}


def register_alert_routes(context: ApiContext) -> None:
    blueprint = context.blueprint
    callbacks = context.callbacks
    security = context.security
    require_auth = context.require_auth

    def _store():
        return callbacks.get("alert_channels")

    def _broker():
        return callbacks.get("credential_broker")

    def _resolver(broker):
        def resolve_secret(purpose):
            lease = broker.resolve_active(purpose)
            return lease.get("token") if isinstance(lease, dict) else None
        return resolve_secret

    def _store_secret(broker, channel, actor, secret):
        """Encrypt a channel secret in the broker and bind it to the purpose."""
        purpose = channel_purpose(channel["id"])
        credential = broker.put(
            "application-secret",
            "Alert channel {}".format(channel["name"])[:80],
            secret,
            None,
            actor,
        )
        broker.activate(credential["id"], purpose)
        _store().mark_secret(channel["id"], actor, True)

    def _forget_secret(broker, channel, actor):
        purpose = channel_purpose(channel["id"])
        try:
            lease = broker.resolve_active(purpose)
            broker.deactivate(purpose)
            if isinstance(lease, dict) and lease.get("credential_id"):
                broker.delete(lease["credential_id"])
        except CredentialError:
            pass

    @blueprint.get("/assistant/alert-channels")
    @require_auth("administrator")
    def alert_channel_list():
        store = _store()
        if store is None:
            return _payload(error=_UNAVAILABLE, status=503)
        return _payload({"channels": store.list(g.auth_session.username)})

    @blueprint.post("/assistant/alert-channels")
    @require_auth("administrator", csrf=True)
    def alert_channel_create():
        store = _store()
        if store is None:
            return _payload(error=_UNAVAILABLE, status=503)
        body = request.get_json(silent=True) or {}
        actor = g.auth_session.username
        try:
            channel = store.create(actor, str(body.get("kind", "")), body)
            secret = str(body.get("secret", "")).strip()
            if secret:
                broker = _broker()
                if broker is None:
                    store.delete(channel["id"], actor)
                    raise CredentialError(_BROKER_DOWN)
                _store_secret(broker, channel, actor, secret)
                channel = store.get(channel["id"], actor)
        except (AlertChannelError, CredentialError) as error:
            return _payload(error={"code": "alert_channel_rejected", "message": str(error)}, status=400)
        security.audit(
            actor, "assistant.alert_channel.create", "success",
            target=channel["id"], remote_addr=request.remote_addr or "",
            details={"kind": channel["kind"]},
        )
        return _payload(channel, status=201)

    @blueprint.patch("/assistant/alert-channels/<channel_id>")
    @require_auth("administrator", csrf=True)
    def alert_channel_update(channel_id):
        store = _store()
        if store is None:
            return _payload(error=_UNAVAILABLE, status=503)
        body = request.get_json(silent=True) or {}
        actor = g.auth_session.username
        try:
            if isinstance(body.get("enabled"), bool) and len(body) == 1:
                channel = store.set_enabled(channel_id, actor, body["enabled"])
            else:
                channel = store.update(channel_id, actor, body)
                secret = str(body.get("secret", "")).strip()
                if secret:
                    broker = _broker()
                    if broker is None:
                        raise CredentialError(_BROKER_DOWN)
                    _store_secret(broker, channel, actor, secret)
                    channel = store.get(channel_id, actor)
        except (AlertChannelError, CredentialError) as error:
            return _payload(error={"code": "alert_channel_rejected", "message": str(error)}, status=400)
        security.audit(
            actor, "assistant.alert_channel.update", "success",
            target=channel_id, remote_addr=request.remote_addr or "",
        )
        return _payload(channel)

    @blueprint.delete("/assistant/alert-channels/<channel_id>")
    @require_auth("administrator", csrf=True)
    def alert_channel_delete(channel_id):
        store = _store()
        if store is None:
            return _payload(error=_UNAVAILABLE, status=503)
        actor = g.auth_session.username
        channel = store.get(channel_id, actor)
        if channel is None:
            return _payload(error={"code": "alert_channel_not_found", "message": CHANNEL_NOT_FOUND}, status=404)
        broker = _broker()
        if broker is not None and channel.get("has_secret"):
            _forget_secret(broker, channel, actor)
        try:
            result = store.delete(channel_id, actor)
        except AlertChannelError as error:
            return _payload(error={"code": "alert_channel_rejected", "message": str(error)}, status=400)
        security.audit(
            actor, "assistant.alert_channel.delete", "success",
            target=channel_id, remote_addr=request.remote_addr or "",
        )
        return _payload(result)

    @blueprint.post("/assistant/alert-channels/<channel_id>/test")
    @require_auth("administrator", csrf=True)
    def alert_channel_test(channel_id):
        store = _store()
        if store is None:
            return _payload(error=_UNAVAILABLE, status=503)
        actor = g.auth_session.username
        channel = store.get(channel_id, actor)
        if channel is None:
            return _payload(error={"code": "alert_channel_not_found", "message": CHANNEL_NOT_FOUND}, status=404)
        broker = _broker()
        resolve_secret = _resolver(broker) if broker is not None else (lambda _purpose: None)
        senders = callbacks.get("alert_senders") or Senders()
        alert = {
            "trigger_name": "Test alert: {}".format(channel["name"]),
            "source": "test_signal", "operator": ">=", "threshold": 0,
            "observed_value": 1, "timestamp": time.time(),
            "task_id": "",
        }
        results = deliver_alert([{**channel, "enabled": True}], alert, resolve_secret, senders)
        summary = record_results(store, results)
        outcome = results[0] if results else None
        delivered = bool(outcome and outcome.delivered)
        security.audit(
            actor, "assistant.alert_channel.test",
            "success" if delivered else "failure",
            target=channel_id, remote_addr=request.remote_addr or "",
        )
        return _payload({
            "channel_id": channel_id,
            "delivered": delivered,
            "error": outcome.error if outcome else "no_result",
            "detail": outcome.detail if outcome else "",
            "summary": summary,
        })
