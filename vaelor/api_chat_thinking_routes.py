"""AI Chat's thinking control routes (VD-209): read the control, remember a step.

- ``GET /ai-chat/thinking?model=`` answers whether the active connection's
  model can think, which steps it takes, and the step remembered for that
  connection (or the default it starts on).
- ``PATCH /ai-chat/thinking`` remembers ``{credential_id, step}``.

A send carries the step it was asked with (`thinking_choice`), so a click
just before Send is never lost to the PATCH still in flight; without one the
connection's remembered step is used.
"""

from __future__ import annotations

import sqlite3
import threading
from typing import Any, Callable, Dict, Mapping, Union

from flask import g, request

from .api_common import ApiContext, payload as _payload
from .chat_inference import ChatInferenceError
from .chat_thinking import THINKING_STEPS
from .chat_thinking_store import ThinkingPreferenceError, ThinkingPreferences

#: Answered when AI Chat's own services are not wired (an appliance mid-start).
THINKING_UNAVAILABLE = "AI Chat's thinking control is unavailable right now."

_STORES: Dict[str, ThinkingPreferences] = {}
_LOCK = threading.Lock()


def thinking_preferences(store: Any) -> ThinkingPreferences:
    """The preference table beside AI Chat's own store (one per database file)."""
    path = str(store.database_path)
    with _LOCK:
        if path not in _STORES:
            _STORES[path] = ThinkingPreferences(path)
        return _STORES[path]


def thinking_choice(store: Any, body: Mapping[str, Any], actor: str) -> Union[str, Callable[[str], str]]:
    """The step one send asks for: the request's own, else the remembered one."""
    asked = str(body.get("thinking") or "").strip().lower()
    if asked in THINKING_STEPS:
        return asked
    if store is None or not hasattr(store, "database_path"):
        return ""

    def remembered(credential_id: str) -> str:
        try:
            return thinking_preferences(store).step(actor, credential_id)
        except sqlite3.Error:
            return ""

    return remembered


def register_chat_thinking_routes(context: ApiContext) -> None:
    blueprint = context.blueprint
    callbacks = context.callbacks
    require_auth = context.require_auth

    @blueprint.get("/ai-chat/thinking")
    @require_auth("operator")
    def chat_thinking():
        store, inference = callbacks.get("rag_chat"), callbacks.get("chat_inference")
        if store is None or inference is None:
            return _payload(error={"code": "ai_chat_unavailable", "message": THINKING_UNAVAILABLE}, status=503)
        try:
            view = inference.thinking(str(request.args.get("model", "")).strip()[:200])
        except ChatInferenceError as error:
            return _payload(error={"code": error.code, "message": str(error)}, status=error.status)
        remembered = ""
        if view["available"] and view["credential_id"]:
            try:
                remembered = thinking_preferences(store).step(
                    g.auth_session.username, view["credential_id"])
            except sqlite3.Error:
                remembered = ""
        # The step the model will actually be sent: the remembered one if
        # this model takes it, else the default (a model with no Off is
        # never shown Off as chosen).
        view["remembered_step"] = remembered
        view["step"] = view["default_step"] if remembered not in view["steps"] else remembered
        return _payload(view)

    @blueprint.patch("/ai-chat/thinking")
    @require_auth("operator", csrf=True)
    def chat_thinking_update():
        store = callbacks.get("rag_chat")
        body = request.get_json(silent=True) or {}
        if store is None:
            return _payload(error={"code": "ai_chat_unavailable", "message": THINKING_UNAVAILABLE}, status=503)
        try:
            step = thinking_preferences(store).set_step(
                g.auth_session.username, body.get("credential_id", ""), body.get("step", ""))
        except (ThinkingPreferenceError, sqlite3.Error) as error:
            return _payload(error={"code": "chat_thinking_rejected", "message": str(error)}, status=400)
        return _payload({"credential_id": str(body.get("credential_id")), "step": step})

