"""Revocable token store and OpenAI-compatible Vaelor Assistant API."""

from __future__ import annotations

import hashlib
import json
import os
import secrets
import sqlite3
import time
import uuid
from contextlib import closing
from pathlib import Path
from typing import Optional

from flask import Blueprint, jsonify, request

from .runtime_paths import env_value, state_path
from .local_inference_gate import BUSY_MESSAGE, LocalModelBusy

TOKEN_SCOPES = {"assistant", "inference"}
ASSISTANT_MODEL_ID = "vaelor-assistant"
LEGACY_ASSISTANT_MODEL_ID = "pironman-copilot"


class AgentApiTokenStore:
    def __init__(self, database_path: Optional[str] = None):
        self.database_path = database_path or env_value(
            "VAELOR_AGENT_API_DB", "PM_AGENT_API_DB",
            state_path("assistant/api-tokens.sqlite3"),
        )
        Path(self.database_path).parent.mkdir(parents=True, exist_ok=True, mode=0o700)
        with closing(sqlite3.connect(self.database_path)) as connection:
            connection.execute(
                """
                CREATE TABLE IF NOT EXISTS api_tokens (
                    id TEXT PRIMARY KEY, label TEXT NOT NULL, token_hash TEXT NOT NULL UNIQUE,
                    prefix TEXT NOT NULL, enabled INTEGER NOT NULL,
                    created_at REAL NOT NULL, last_used_at REAL,
                    scopes TEXT NOT NULL DEFAULT '["assistant"]'
                )
                """
            )
            columns = {
                row[1] for row in connection.execute("PRAGMA table_info(api_tokens)")
            }
            if "scopes" not in columns:
                connection.execute(
                    """ALTER TABLE api_tokens
                       ADD COLUMN scopes TEXT NOT NULL DEFAULT '["assistant"]'"""
                )
            connection.commit()
        try:
            os.chmod(self.database_path, 0o600)
        except OSError:
            pass

    @staticmethod
    def _hash(token: str):
        return hashlib.sha256(token.encode("utf-8")).hexdigest()

    @staticmethod
    def _normalize_scopes(scopes=None):
        values = scopes if isinstance(scopes, list) else ["assistant"]
        normalized = sorted({
            str(scope).strip().lower() for scope in values
            if str(scope).strip()
        })
        if not normalized or any(scope not in TOKEN_SCOPES for scope in normalized):
            raise ValueError("Choose Assistant access, inference access, or both.")
        return normalized

    def create(self, label: str, scopes=None):
        label = str(label).strip()[:100]
        if not label:
            raise ValueError("Enter a name for this API connection.")
        normalized_scopes = self._normalize_scopes(scopes)
        token = "vak_{}".format(secrets.token_urlsafe(32))
        item_id = uuid.uuid4().hex
        now = time.time()
        with closing(sqlite3.connect(self.database_path)) as connection:
            connection.execute(
                """INSERT INTO api_tokens(
                       id,label,token_hash,prefix,enabled,created_at,scopes
                   ) VALUES(?,?,?,?,1,?,?)""",
                (
                    item_id, label, self._hash(token), token[:12], now,
                    json.dumps(normalized_scopes, separators=(",", ":")),
                ),
            )
            connection.commit()
        return {"token": token, **self.get(item_id)}

    def get(self, item_id: str):
        with closing(sqlite3.connect(self.database_path)) as connection:
            connection.row_factory = sqlite3.Row
            row = connection.execute(
                "SELECT id,label,prefix,enabled,created_at,last_used_at,scopes FROM api_tokens WHERE id=?",
                (item_id,),
            ).fetchone()
        if row is None:
            raise KeyError(item_id)
        result = dict(row)
        result["enabled"] = bool(result["enabled"])
        result["scopes"] = json.loads(result["scopes"])
        return result

    def list(self):
        with closing(sqlite3.connect(self.database_path)) as connection:
            connection.row_factory = sqlite3.Row
            rows = connection.execute(
                "SELECT id,label,prefix,enabled,created_at,last_used_at,scopes FROM api_tokens ORDER BY created_at DESC"
            ).fetchall()
        return [{
            **dict(row),
            "enabled": bool(row["enabled"]),
            "scopes": json.loads(row["scopes"]),
        } for row in rows]

    def revoke(self, item_id: str):
        with closing(sqlite3.connect(self.database_path)) as connection:
            cursor = connection.execute(
                "UPDATE api_tokens SET enabled=0 WHERE id=?", (item_id,)
            )
            connection.commit()
        if not cursor.rowcount:
            raise KeyError(item_id)
        return self.get(item_id)

    def authenticate(self, token: str, required_scope: Optional[str] = None):
        if (
            not token.startswith(("vak_", "pmk_"))
            or len(token) > 256
        ):
            return None
        token_hash = self._hash(token)
        now = time.time()
        with closing(sqlite3.connect(self.database_path)) as connection:
            connection.row_factory = sqlite3.Row
            row = connection.execute(
                "SELECT id,label,scopes FROM api_tokens WHERE token_hash=? AND enabled=1",
                (token_hash,),
            ).fetchone()
            scopes = json.loads(row["scopes"]) if row else []
            if row and (required_scope is None or required_scope in scopes):
                connection.execute(
                    "UPDATE api_tokens SET last_used_at=? WHERE id=?", (now, row["id"])
                )
                connection.commit()
                return {"id": row["id"], "label": row["label"], "scopes": scopes}
        return None


def create_agent_api_blueprint(tokens, agent, context_provider):
    blueprint = Blueprint("agent_api", __name__, url_prefix="/v1")

    def authenticated():
        header = request.headers.get("Authorization", "")
        token = header[7:] if header.startswith("Bearer ") else ""
        return tokens.authenticate(token, "assistant")

    @blueprint.get("/models")
    def models():
        identity = authenticated()
        if identity is None:
            return jsonify({"error": {"message": "Invalid or revoked API token.", "type": "authentication_error"}}), 401
        return jsonify({"object": "list", "data": [
            {
                "id": ASSISTANT_MODEL_ID, "object": "model",
                "created": 0, "owned_by": "vaelor",
            },
            {
                "id": LEGACY_ASSISTANT_MODEL_ID, "object": "model",
                "created": 0, "owned_by": "vaelor",
                "deprecated": True,
            },
        ]})

    @blueprint.post("/chat/completions")
    def chat_completions():
        identity = authenticated()
        if identity is None:
            return jsonify({"error": {"message": "Invalid or revoked API token.", "type": "authentication_error"}}), 401
        body = request.get_json(silent=True) or {}
        if body.get("stream"):
            return jsonify({"error": {"message": "Streaming is not enabled for this appliance endpoint.", "type": "invalid_request_error"}}), 400
        messages = body.get("messages", [])
        if not isinstance(messages, list) or len(messages) > 50:
            return jsonify({"error": {"message": "Provide up to 50 messages.", "type": "invalid_request_error"}}), 400
        user_messages = [
            str(item.get("content", "")) for item in messages
            if isinstance(item, dict) and item.get("role") == "user"
        ]
        prompt = user_messages[-1].strip() if user_messages else ""
        if not prompt or len(prompt) > 4000:
            return jsonify({"error": {"message": "The final user message must contain 1-4,000 characters.", "type": "invalid_request_error"}}), 400
        try:
            answer = agent.answer(prompt, context=context_provider())
        except LocalModelBusy as error:
            # The single-endpoint model is generating for another caller. Return
            # a truthful 503 an OpenAI-compatible client can read, not an opaque
            # 500 that reads as an outage (#223) - the v2 assistant route does
            # the same. Caught by exact type so real model errors are not masked.
            return jsonify({"error": {
                "message": str(error) or BUSY_MESSAGE,
                "type": "server_error", "code": "model_busy",
            }}), 503
        requested_model = str(body.get("model", ASSISTANT_MODEL_ID))
        response_model = (
            requested_model
            if requested_model in {
                ASSISTANT_MODEL_ID, LEGACY_ASSISTANT_MODEL_ID
            }
            else ASSISTANT_MODEL_ID
        )
        content = str(answer.get("answer", "")).strip()
        if answer.get("proposed_job"):
            content += "\n\nA protected action is available in the Vaelor UI for operator review and approval."
        created = int(time.time())
        return jsonify({
            "id": "chatcmpl_{}".format(uuid.uuid4().hex),
            "object": "chat.completion",
            "created": created,
            "model": response_model,
            "choices": [{"index": 0, "message": {"role": "assistant", "content": content}, "finish_reason": "stop"}],
            "usage": {"prompt_tokens": max(1, len(prompt) // 4), "completion_tokens": max(1, len(content) // 4), "total_tokens": max(2, (len(prompt) + len(content)) // 4)},
        })

    return blueprint
