"""Revocable token store gating Vaelor's inference API access."""

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

from .runtime_paths import env_value, state_path

TOKEN_SCOPES = {"inference"}


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
                    scopes TEXT NOT NULL DEFAULT '["inference"]'
                )
                """
            )
            columns = {
                row[1] for row in connection.execute("PRAGMA table_info(api_tokens)")
            }
            if "scopes" not in columns:
                connection.execute(
                    """ALTER TABLE api_tokens
                       ADD COLUMN scopes TEXT NOT NULL DEFAULT '["inference"]'"""
                )
            if "revoked_at" not in columns:
                # ACC-115: a revoke flipped `enabled` and recorded no time, so
                # the console could not say when a key stopped working. Keys
                # revoked before this column existed keep NULL ("date not
                # recorded"), never a guessed date.
                connection.execute("ALTER TABLE api_tokens ADD COLUMN revoked_at REAL")
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
        values = scopes if isinstance(scopes, list) else ["inference"]
        normalized = sorted({
            str(scope).strip().lower() for scope in values
            if str(scope).strip()
        })
        if not normalized or any(scope not in TOKEN_SCOPES for scope in normalized):
            raise ValueError("Choose inference access.")
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
                "SELECT id,label,prefix,enabled,created_at,last_used_at,scopes,revoked_at FROM api_tokens WHERE id=?",
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
                "SELECT id,label,prefix,enabled,created_at,last_used_at,scopes,revoked_at FROM api_tokens ORDER BY created_at DESC"
            ).fetchall()
        return [{
            **dict(row),
            "enabled": bool(row["enabled"]),
            "scopes": json.loads(row["scopes"]),
        } for row in rows]

    def revoke(self, item_id: str):
        with closing(sqlite3.connect(self.database_path)) as connection:
            # The first revoke's time is kept: revoking again is not a new event.
            cursor = connection.execute(
                "UPDATE api_tokens SET enabled=0, revoked_at=COALESCE(revoked_at, ?) WHERE id=?",
                (time.time(), item_id),
            )
            connection.commit()
        if not cursor.rowcount:
            raise KeyError(item_id)
        return self.get(item_id)

    def delete(self, item_id: str):
        """Remove a REVOKED key's record (ACC-115); a working key is refused.

        Revoked keys otherwise piled up for ever. Only the row goes: the audit
        log keeps the create, revoke and this delete, and the usage meter keeps
        its per-key totals. A key that still works must be revoked first, so a
        delete can never be the thing that silently cuts a client off.
        """
        with closing(sqlite3.connect(self.database_path)) as connection:
            row = connection.execute(
                "SELECT enabled FROM api_tokens WHERE id=?", (item_id,)
            ).fetchone()
            if row is None:
                raise KeyError(item_id)
            if row[0]:
                raise ValueError(
                    "This key still works. Revoke it first; its record can be "
                    "removed once it is revoked."
                )
            connection.execute("DELETE FROM api_tokens WHERE id=? AND enabled=0", (item_id,))
            connection.commit()
        return {"deleted": True, "id": item_id}

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
