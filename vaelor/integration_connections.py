"""Durable, secret-free metadata for installed-app integration connections."""

from __future__ import annotations

import hashlib
import json
import os
import re
import sqlite3
import time
import uuid
from contextlib import contextmanager
from typing import Any, Dict, Iterable, Optional

from .runtime_paths import state_path


class IntegrationConnectionError(ValueError):
    """Safe, bounded error for connection metadata operations."""


_REF = re.compile(r"^cred_[A-Za-z0-9_-]{8,120}$")
_TEST_STATES = {"pending", "healthy", "degraded", "failed"}
_MAX_SCOPES = 64
_MAX_AUDIT_DETAIL = 256


def _now() -> float:
    return time.time()


def _text(value: Any, field: str, maximum: int = 160) -> str:
    result = str(value or "").strip()
    if not result or len(result) > maximum:
        raise IntegrationConnectionError("{} is invalid.".format(field))
    return result


def _secret_version(value: Any) -> int:
    try:
        result = int(value)
    except (TypeError, ValueError) as error:
        raise IntegrationConnectionError("secret_version is invalid.") from error
    if result < 1:
        raise IntegrationConnectionError("secret_version is invalid.")
    return result


def _scopes(values: Iterable[Any]) -> list[str]:
    if isinstance(values, (str, bytes)):
        raise IntegrationConnectionError("scopes must be a list.")
    result = sorted({_text(item, "scope", 100) for item in values})
    if len(result) > _MAX_SCOPES:
        raise IntegrationConnectionError("Too many connection scopes.")
    return result


def _safe_digest(value: Dict[str, Any]) -> str:
    return hashlib.sha256(
        json.dumps(value, sort_keys=True, separators=(",", ":"), default=str).encode()
    ).hexdigest()


class IntegrationConnectionStore:
    """Own connection lifecycle metadata; credential values never enter this store."""

    def __init__(self, database_path: Optional[str] = None, clock=_now):
        self.database_path = database_path or state_path("assistant/integrations.sqlite3")
        self.clock = clock
        parent = os.path.dirname(self.database_path)
        if parent:
            os.makedirs(parent, mode=0o700, exist_ok=True)
        self._initialize()
        try:
            os.chmod(self.database_path, 0o600)
        except OSError:
            pass

    def _connect(self) -> sqlite3.Connection:
        connection = sqlite3.connect(self.database_path, timeout=15)
        connection.row_factory = sqlite3.Row
        connection.execute("PRAGMA foreign_keys=ON")  # pairs-with: sqlite-foreign-keys-tight
        connection.execute("PRAGMA busy_timeout=15000")
        return connection

    @contextmanager
    def _connection(self):
        connection = self._connect()
        try:
            with connection:
                yield connection
        finally:
            connection.close()

    def _initialize(self) -> None:
        with self._connection() as connection:
            connection.execute("PRAGMA journal_mode=WAL")  # pairs-with: sqlite-journal-mode-tight
            connection.executescript(
                """
                CREATE TABLE IF NOT EXISTS integration_connections (
                    id TEXT PRIMARY KEY,
                    actor TEXT NOT NULL,
                    provider TEXT NOT NULL,
                    credential_ref TEXT NOT NULL,
                    scopes_json TEXT NOT NULL,
                    consent_at REAL NOT NULL,
                    secret_version INTEGER NOT NULL,
                    expires_at REAL,
                    test_status TEXT NOT NULL,
                    last_test_at REAL,
                    last_test_detail TEXT NOT NULL DEFAULT '',
                    rotation_count INTEGER NOT NULL DEFAULT 0,
                    rotated_at REAL,
                    revoked_at REAL,
                    revocation_reason TEXT NOT NULL DEFAULT '',
                    idempotency_key TEXT,
                    request_digest TEXT NOT NULL,
                    created_at REAL NOT NULL,
                    updated_at REAL NOT NULL,
                    UNIQUE(actor, idempotency_key)
                );
                CREATE INDEX IF NOT EXISTS idx_integration_connections_actor
                    ON integration_connections(actor, updated_at DESC);
                """
            )

    @staticmethod
    def _row(row: Optional[sqlite3.Row]) -> Optional[Dict[str, Any]]:
        if row is None:
            return None
        item = dict(row)
        item["scopes"] = json.loads(item.pop("scopes_json"))
        item.pop("request_digest", None)
        item["revoked"] = item["revoked_at"] is not None
        item["status"] = "revoked" if item["revoked"] else item["test_status"]
        return item

    def _dependent_counts(self, connection_id: str, actor: str) -> Dict[str, int]:
        try:
            with self._connection() as connection:
                rows = connection.execute(
                    "SELECT revoked_at FROM agent_app_grants "
                    "WHERE connection_id=? AND actor=?", (connection_id, actor)
                ).fetchall()
        except sqlite3.OperationalError:
            rows = []
        return {
            "total": len(rows),
            "active": sum(row["revoked_at"] is None for row in rows),
            "revoked": sum(row["revoked_at"] is not None for row in rows),
        }

    def _with_dependents(self, item: Optional[Dict[str, Any]]) -> Optional[Dict[str, Any]]:
        if item is not None:
            item["dependent_counts"] = self._dependent_counts(item["id"], item["actor"])
        return item

    def create(
        self, actor: str, provider: str, credential_ref: str, scopes: Iterable[Any],
        *, secret_version: Any = 1, consent_at: Optional[float] = None,
        expires_at: Optional[float] = None, idempotency_key: Optional[str] = None,
    ) -> Dict[str, Any]:
        actor = _text(actor, "actor", 120)
        provider = _text(provider, "provider", 100)
        credential_ref = _text(credential_ref, "credential_ref", 128)
        if not _REF.fullmatch(credential_ref):
            raise IntegrationConnectionError("credential_ref must be a broker reference.")
        clean_scopes = _scopes(scopes)
        version = _secret_version(secret_version)
        consent = self.clock() if consent_at is None else float(consent_at)
        expiry = None if expires_at is None else float(expires_at)
        if expiry is not None and expiry <= consent:
            raise IntegrationConnectionError("expiry must be after consent.")
        key = None if idempotency_key is None else _text(idempotency_key, "idempotency_key", 160)
        payload = {
            "actor": actor, "provider": provider, "credential_ref": credential_ref,
            "scopes": clean_scopes, "secret_version": version,
            "consent_at": None if consent_at is None else consent, "expires_at": expiry,
        }
        digest = _safe_digest(payload)
        connection_id = "conn_" + uuid.uuid4().hex
        now = self.clock()
        with self._connection() as connection:
            if key:
                existing = connection.execute(
                    "SELECT * FROM integration_connections WHERE actor=? AND idempotency_key=?",
                    (actor, key),
                ).fetchone()
                if existing:
                    if existing["request_digest"] != digest:
                        raise IntegrationConnectionError("idempotency_key conflicts with an earlier request.")
                    return self._with_dependents(self._row(existing))  # type: ignore[arg-type]
            connection.execute(
                """INSERT INTO integration_connections
                (id,actor,provider,credential_ref,scopes_json,consent_at,secret_version,
                 expires_at,test_status,last_test_at,last_test_detail,rotation_count,
                 rotated_at,revoked_at,revocation_reason,idempotency_key,request_digest,
                 created_at,updated_at)
                VALUES(?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)""",
                (connection_id, actor, provider, credential_ref,
                 json.dumps(clean_scopes, separators=(",", ":")), consent, version,
                 expiry, "pending", None, "", 0, None, None, "", key, digest, now, now),
            )
        return self.get(connection_id, actor)  # type: ignore[return-value]

    def get(self, connection_id: str, actor: Optional[str] = None) -> Optional[Dict[str, Any]]:
        query = "SELECT * FROM integration_connections WHERE id=?"
        values: list[Any] = [str(connection_id)]
        if actor is not None:
            query += " AND actor=?"
            values.append(_text(actor, "actor", 120))
        with self._connection() as connection:
            row = connection.execute(query, values).fetchone()
        return self._with_dependents(self._row(row))

    def list(self, actor: str, limit: int = 100) -> list[Dict[str, Any]]:
        actor = _text(actor, "actor", 120)
        limit = max(1, min(int(limit), 200))
        with self._connection() as connection:
            rows = connection.execute(
                "SELECT * FROM integration_connections WHERE actor=? "
                "ORDER BY updated_at DESC LIMIT ?", (actor, limit)
            ).fetchall()
        return [self._with_dependents(self._row(row)) for row in rows]  # type: ignore[list-item]

    def test(self, connection_id: str, actor: str, healthy: bool, detail: str = "") -> Dict[str, Any]:
        actor = _text(actor, "actor", 120)
        status = "healthy" if bool(healthy) else "degraded"
        detail = str(detail or "").strip()[:_MAX_AUDIT_DETAIL]
        now = self.clock()
        with self._connection() as connection:
            updated = connection.execute(
                "UPDATE integration_connections SET test_status=?,last_test_at=?,"
                "last_test_detail=?,updated_at=? WHERE id=? AND actor=? AND revoked_at IS NULL",
                (status, now, detail, now, str(connection_id), actor),
            ).rowcount
        if updated != 1:
            raise IntegrationConnectionError("Connection was not found or is revoked.")
        return self.get(connection_id, actor)  # type: ignore[return-value]

    record_test = test

    def rotate(self, connection_id: str, actor: str, secret_version: Any) -> Dict[str, Any]:
        actor = _text(actor, "actor", 120)
        version = _secret_version(secret_version)
        now = self.clock()
        with self._connection() as connection:
            current = connection.execute(
                "SELECT secret_version FROM integration_connections WHERE id=? AND actor=? AND revoked_at IS NULL",
                (str(connection_id), actor),
            ).fetchone()
            if current is None:
                raise IntegrationConnectionError("Connection was not found or is revoked.")
            if int(current["secret_version"]) == version:
                return self.get(connection_id, actor)  # type: ignore[return-value]
            connection.execute(
                "UPDATE integration_connections SET secret_version=?,test_status='pending',"
                "last_test_at=NULL,last_test_detail='',rotation_count=rotation_count+1,"
                "rotated_at=?,updated_at=? WHERE id=? AND actor=? AND revoked_at IS NULL",
                (version, now, now, str(connection_id), actor),
            )
        return self.get(connection_id, actor)  # type: ignore[return-value]

    def revoke(self, connection_id: str, actor: str, reason: str = "") -> Dict[str, Any]:
        actor = _text(actor, "actor", 120)
        now = self.clock()
        reason = str(reason or "revoked by actor").strip()[:_MAX_AUDIT_DETAIL]
        with self._connection() as connection:
            connection.execute(
                "UPDATE integration_connections SET revoked_at=?,revocation_reason=?,updated_at=? "
                "WHERE id=? AND actor=? AND revoked_at IS NULL",
                (now, reason, now, str(connection_id), actor),
            )
        item = self.get(connection_id, actor)
        if item is None:
            raise IntegrationConnectionError("Connection was not found.")
        return item

    def dependents(self, connection_id: str, actor: str) -> Dict[str, Any]:
        item = self.get(connection_id, actor)
        if item is None:
            raise IntegrationConnectionError("Connection was not found.")
        counts = item["dependent_counts"]
        try:
            with self._connection() as connection:
                rows = connection.execute(
                    "SELECT id,agent_id,agent_version,app_instance_id,manifest_digest,revoked_at "
                    "FROM agent_app_grants WHERE connection_id=? AND actor=? ORDER BY created_at DESC",
                    (str(connection_id), actor),
                ).fetchall()
        except sqlite3.OperationalError:
            rows = []
        return {"connection_id": str(connection_id), "counts": counts, "grants": [dict(row) for row in rows]}


ConnectionStore = IntegrationConnectionStore
