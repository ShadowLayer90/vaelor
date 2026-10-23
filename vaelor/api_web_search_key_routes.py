"""The owner's optional web search key (VD-222 item 2), administrator only.

An owner may add one search API key (Brave Search API). It is stored in the
credential broker like every other secret the owner adds, leased under the
``web-search`` purpose, and never shown back: these routes answer only whether a
key is set, its fingerprint tail and when it was added. Adding or removing it
queues the search service's key action (set or remove), which rewrites
settings.yml with or without SearXNG's ``braveapi`` engine and recreates the search container.

Vaelor never adds a key by itself (the no-automatic-AI-server-discovery rule:
outside services are only ever the owner's choice). With a key, every guarded
search also goes to Brave, which the Connections row says before it is added.
"""

from __future__ import annotations

from flask import g, request

from .api_common import ApiContext, CREDENTIAL_BROKER_UNAVAILABLE, payload
from .credential_broker import PROVIDERS, WEB_SEARCH_PROVIDER, WEB_SEARCH_PURPOSE, CredentialError
from .web_research import (
    KEY_REMOVE_ACTION, KEY_SET_ACTION, SEARCH_KEY_SHAPE, SETTINGS_CONFIRMATION,
)

#: The service the key is for, as the broker names the kind.
PROVIDER_NAME = PROVIDERS[WEB_SEARCH_PROVIDER]["name"]
#: The label the stored key carries in the broker and on the Connections list.
WEB_SEARCH_KEY_LABEL = "Web search key (Brave Search API)"
#: Said when the body's key is not one opaque token (the shape settings.yml takes).
KEY_REFUSED = (
    "Paste the key exactly as Brave shows it: letters, digits, '-' and '_', "
    "16 to 200 characters."
)
QUEUE_UNAVAILABLE = (
    "The search service's update queue is unavailable, so nothing was changed. "
    "Try again in a moment."
)


def _view(items) -> dict:
    """What an administrator may see of the key: never the key itself."""
    keys = [item for item in items if item.get("provider") == WEB_SEARCH_PROVIDER]
    active = next((item for item in keys if WEB_SEARCH_PURPOSE in (item.get("active_for") or [])), None)
    if active is None:
        return {"configured": False, "provider": PROVIDER_NAME}
    return {
        "configured": True,
        "provider": PROVIDER_NAME,
        "fingerprint_tail": str(active.get("fingerprint") or "")[-6:],
        "added_at": active.get("updated_at") or active.get("created_at"),
        "added_by": active.get("owner") or "",
    }


def register_web_search_key_routes(context: ApiContext) -> None:
    blueprint = context.blueprint
    require_auth = context.require_auth
    security = context.security
    callbacks = context.callbacks

    def unavailable(message: str, code: str):
        return payload(error={"code": code, "message": message}, status=503)

    def queue_settings(job_store, action):
        # Never folded into an earlier settings job: one already running has
        # read the key it writes, and this change would be lost behind it.
        return job_store.create(
            "host.web-research.manage", g.auth_session.username,
            {"action": action, "confirmation": SETTINGS_CONFIRMATION, "purge": False},
            deduplicate=False,
        )

    @blueprint.get("/web-search/key")
    @require_auth("administrator")
    def web_search_key_status():
        broker = callbacks.get("credential_broker")
        if broker is None:
            return unavailable(CREDENTIAL_BROKER_UNAVAILABLE, "credential_broker_unavailable")
        try:
            return payload(_view(broker.list(g.auth_session.username)))
        except CredentialError as error:
            return unavailable(str(error), "credential_broker_unavailable")

    @blueprint.put("/web-search/key")
    @require_auth("administrator", csrf=True)
    def web_search_key_store():
        body = request.get_json(silent=True)
        key = str(body.get("key", "")).strip() if isinstance(body, dict) else ""
        if not isinstance(body, dict) or set(body) - {"key"} or not SEARCH_KEY_SHAPE.fullmatch(key):
            return payload(error={"code": "invalid_web_search_key", "message": KEY_REFUSED}, status=400)
        broker = callbacks.get("credential_broker")
        job_store = callbacks.get("job_store")
        if broker is None:
            return unavailable(CREDENTIAL_BROKER_UNAVAILABLE, "credential_broker_unavailable")
        if job_store is None:
            return unavailable(QUEUE_UNAVAILABLE, "research_job_queue_unavailable")
        try:
            earlier = [
                item["id"] for item in broker.list(g.auth_session.username)
                if item.get("provider") == WEB_SEARCH_PROVIDER
            ]
            stored = broker.put(WEB_SEARCH_PROVIDER, WEB_SEARCH_KEY_LABEL, key, None, g.auth_session.username)
            try:
                broker.activate(stored["id"], WEB_SEARCH_PURPOSE)
            except CredentialError:
                broker.delete(stored["id"])
                raise
            for credential_id in earlier:
                broker.delete(credential_id)
            view = _view(broker.list(g.auth_session.username))
        except CredentialError as error:
            security.audit(
                g.auth_session.username, "web-search-key.store", "failure",
                remote_addr=request.remote_addr or "",
            )
            return payload(error={"code": "web_search_key_not_stored", "message": str(error)}, status=400)
        job = queue_settings(job_store, KEY_SET_ACTION)
        security.audit(
            g.auth_session.username, "web-search-key.store", "success",
            target=stored["id"], remote_addr=request.remote_addr or "",
            details={"job_id": job["id"], "replaced": len(earlier)},
        )
        return payload({**view, "job": job}, status=202)

    @blueprint.delete("/web-search/key")
    @require_auth("administrator", csrf=True)
    def web_search_key_remove():
        broker = callbacks.get("credential_broker")
        job_store = callbacks.get("job_store")
        if broker is None:
            return unavailable(CREDENTIAL_BROKER_UNAVAILABLE, "credential_broker_unavailable")
        if job_store is None:
            return unavailable(QUEUE_UNAVAILABLE, "research_job_queue_unavailable")
        try:
            removed = [
                item["id"] for item in broker.list(g.auth_session.username)
                if item.get("provider") == WEB_SEARCH_PROVIDER
            ]
            for credential_id in removed:
                broker.delete(credential_id)
        except CredentialError as error:
            return unavailable(str(error), "credential_broker_unavailable")
        # Queued even when nothing was stored: a settings.yml that still names a
        # key the broker no longer holds is rewritten without it. The removal
        # is the job's own action, so it never depends on the broker answering.
        job = queue_settings(job_store, KEY_REMOVE_ACTION)
        security.audit(
            g.auth_session.username, "web-search-key.remove", "success",
            remote_addr=request.remote_addr or "",
            details={"job_id": job["id"], "removed": len(removed)},
        )
        return payload({"configured": False, "provider": PROVIDER_NAME, "job": job}, status=202)
