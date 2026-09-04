"""Version-pinned app grants and bounded invocation audit."""

from __future__ import annotations

import hashlib
import json
import os
import sqlite3
import time
import uuid
from contextlib import contextmanager
from typing import Any, Dict, Iterable, Mapping, Optional

from .runtime_paths import state_path


class AgentAppGrantError(ValueError):
    """Safe authorization and persistence error."""


class GrantDeniedError(AgentAppGrantError):
    """Raised when a grant is not active for the supplied current facts."""

    def __init__(self, decision: Mapping[str, Any]):
        self.decision = dict(decision)
        super().__init__("Grant unavailable: {}.".format(", ".join(decision.get("reasons", ["unknown"]))))


_MAX_OPERATIONS = 64
_MAX_PROVENANCE_BYTES = 4096
_SENSITIVE_NAMES = {"secret", "token", "password", "authorization", "credential_value", "plaintext"}


def _now() -> float:
    return time.time()


def _text(value: Any, field: str, maximum: int = 160) -> str:
    result = str(value or "").strip()
    if not result or len(result) > maximum:
        raise AgentAppGrantError("{} is invalid.".format(field))
    return result


def _version(value: Any) -> int:
    try:
        result = int(value)
    except (TypeError, ValueError) as error:
        raise AgentAppGrantError("agent_version is invalid.") from error
    if result < 1:
        raise AgentAppGrantError("agent_version is invalid.")
    return result


def _operations(values: Iterable[Any]) -> list[str]:
    if isinstance(values, (str, bytes)):
        raise AgentAppGrantError("operation_ids must be a list.")
    result = sorted({_text(item, "operation_id", 120) for item in values})
    if not result or len(result) > _MAX_OPERATIONS:
        raise AgentAppGrantError("The grant must contain a bounded operation selection.")
    return result


def _digest(value: Mapping[str, Any]) -> str:
    return hashlib.sha256(
        json.dumps(value, sort_keys=True, separators=(",", ":"), default=str).encode()
    ).hexdigest()


def _contains_sensitive(value: Any) -> bool:
    if isinstance(value, Mapping):
        return any(
            any(name in str(key).lower() for name in _SENSITIVE_NAMES) or _contains_sensitive(item)
            for key, item in value.items()
        )
    if isinstance(value, (list, tuple)):
        return any(_contains_sensitive(item) for item in value)
    return False


class AgentAppGrantStore:
    """Store only stable grant pins; runtime truth is supplied to ``evaluate``."""

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
                CREATE TABLE IF NOT EXISTS agent_app_grants (
                    id TEXT PRIMARY KEY,
                    actor TEXT NOT NULL,
                    agent_id TEXT NOT NULL,
                    agent_version INTEGER NOT NULL,
                    app_instance_id TEXT NOT NULL,
                    manifest_digest TEXT NOT NULL,
                    connection_id TEXT NOT NULL,
                    connection_secret_version INTEGER NOT NULL,
                    operation_ids_json TEXT NOT NULL,
                    source_grant_id TEXT,
                    revoked_at REAL,
                    revocation_reason TEXT NOT NULL DEFAULT '',
                    idempotency_key TEXT,
                    request_digest TEXT NOT NULL,
                    created_at REAL NOT NULL,
                    updated_at REAL NOT NULL,
                    UNIQUE(actor, idempotency_key)
                );
                CREATE INDEX IF NOT EXISTS idx_agent_app_grants_connection
                    ON agent_app_grants(actor, connection_id, created_at DESC);
                CREATE INDEX IF NOT EXISTS idx_agent_app_grants_agent
                    ON agent_app_grants(actor, agent_id, agent_version);
                CREATE TABLE IF NOT EXISTS capability_invocation_audit (
                    event_id TEXT PRIMARY KEY,
                    actor TEXT NOT NULL,
                    grant_id TEXT NOT NULL,
                    task_id TEXT NOT NULL DEFAULT '',
                    agent_id TEXT NOT NULL,
                    agent_version INTEGER NOT NULL,
                    app_instance_id TEXT NOT NULL,
                    manifest_digest TEXT NOT NULL,
                    connection_id TEXT NOT NULL,
                    operation_id TEXT NOT NULL DEFAULT '',
                    approval TEXT NOT NULL DEFAULT 'not_required',
                    duration_ms INTEGER NOT NULL,
                    outcome TEXT NOT NULL,
                    result_provenance_json TEXT NOT NULL DEFAULT '{}',
                    detail TEXT NOT NULL DEFAULT '',
                    idempotency_key TEXT,
                    created_at REAL NOT NULL,
                    UNIQUE(actor, idempotency_key)
                );
                CREATE INDEX IF NOT EXISTS idx_capability_audit_actor
                    ON capability_invocation_audit(actor, created_at DESC);
                """
            )
            columns = {
                row["name"] for row in connection.execute(
                    "PRAGMA table_info(agent_app_grants)"
                )
            }
            if "source_grant_id" not in columns:
                connection.execute(
                    "ALTER TABLE agent_app_grants ADD COLUMN source_grant_id TEXT"
                )
            connection.execute(
                "CREATE UNIQUE INDEX IF NOT EXISTS idx_agent_app_grants_clone "
                "ON agent_app_grants(actor, agent_id, agent_version, source_grant_id) "
                "WHERE source_grant_id IS NOT NULL"
            )


    @staticmethod
    def _grant_row(row: Optional[sqlite3.Row]) -> Optional[Dict[str, Any]]:
        if row is None:
            return None
        item = dict(row)
        item["operation_ids"] = json.loads(item.pop("operation_ids_json"))
        item["revoked"] = item["revoked_at"] is not None
        item.pop("request_digest", None)
        return item

    @staticmethod
    def _audit_row(row: sqlite3.Row) -> Dict[str, Any]:
        item = dict(row)
        item["result_provenance"] = json.loads(item.pop("result_provenance_json") or "{}")
        return item

    def create(
        self, actor: str, agent_id: str, agent_version: Any, app_instance_id: str,
        manifest_digest: str, connection_id: str, operation_ids: Iterable[Any],
        *, connection_secret_version: Any = 1,
        idempotency_key: Optional[str] = None,
    ) -> Dict[str, Any]:
        actor = _text(actor, "actor", 120)
        agent_id = _text(agent_id, "agent_id", 120)
        version = _version(agent_version)
        app_instance_id = _text(app_instance_id, "app_instance_id", 120)
        manifest_digest = _text(manifest_digest, "manifest_digest", 200)
        connection_id = _text(connection_id, "connection_id", 120)
        secret_version = _version(connection_secret_version)
        clean_operations = _operations(operation_ids)
        key = None if idempotency_key is None else _text(idempotency_key, "idempotency_key", 160)
        payload = {
            "actor": actor, "agent_id": agent_id, "agent_version": version,
            "app_instance_id": app_instance_id, "manifest_digest": manifest_digest,
            "connection_id": connection_id, "connection_secret_version": secret_version,
            "operation_ids": clean_operations,
        }
        request_digest = _digest(payload)
        grant_id = "grant_" + uuid.uuid4().hex
        now = self.clock()
        with self._connection() as connection:
            current = connection.execute(
                "SELECT secret_version,actor,revoked_at FROM integration_connections WHERE id=?",
                (connection_id,),
            ).fetchone()
            if current is None or current["actor"] != actor or current["revoked_at"] is not None:
                raise AgentAppGrantError("Connection is unavailable to this actor.")
            if int(current["secret_version"]) != secret_version:
                raise AgentAppGrantError("connection secret version is stale.")
            if key:
                existing = connection.execute(
                    "SELECT * FROM agent_app_grants WHERE actor=? AND idempotency_key=?",
                    (actor, key),
                ).fetchone()
                if existing:
                    if existing["request_digest"] != request_digest:
                        raise AgentAppGrantError("idempotency_key conflicts with an earlier request.")
                    return self._grant_row(existing)  # type: ignore[return-value]
            connection.execute(
                """INSERT INTO agent_app_grants
                (id,actor,agent_id,agent_version,app_instance_id,manifest_digest,
                 connection_id,connection_secret_version,operation_ids_json,source_grant_id,
                 revoked_at,revocation_reason,idempotency_key,request_digest,created_at,updated_at)
                VALUES(?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)""",
                (grant_id, actor, agent_id, version, app_instance_id, manifest_digest,
                 connection_id, secret_version,
                 json.dumps(clean_operations, separators=(",", ":")), None, None, "", key,
                 request_digest, now, now),
            )
        return self.get(grant_id, actor)  # type: ignore[return-value]

    def get(self, grant_id: str, actor: Optional[str] = None) -> Optional[Dict[str, Any]]:
        query = "SELECT * FROM agent_app_grants WHERE id=?"
        values: list[Any] = [str(grant_id)]
        if actor is not None:
            query += " AND actor=?"
            values.append(_text(actor, "actor", 120))
        with self._connection() as connection:
            row = connection.execute(query, values).fetchone()
        return self._grant_row(row)

    def list(self, actor: str, agent_id: Optional[str] = None, limit: int = 100) -> list[Dict[str, Any]]:
        actor = _text(actor, "actor", 120)
        limit = max(1, min(int(limit), 200))
        query = "SELECT * FROM agent_app_grants WHERE actor=?"
        values: list[Any] = [actor]
        if agent_id is not None:
            query += " AND agent_id=?"
            values.append(_text(agent_id, "agent_id", 120))
        query += " ORDER BY created_at DESC LIMIT ?"
        values.append(limit)
        with self._connection() as connection:
            rows = connection.execute(query, values).fetchall()
        return [self._grant_row(row) for row in rows]  # type: ignore[list-item]

    def revoke(self, grant_id: str, actor: str, reason: str = "") -> Dict[str, Any]:
        actor = _text(actor, "actor", 120)
        now = self.clock()
        with self._connection() as connection:
            connection.execute(
                "UPDATE agent_app_grants SET revoked_at=?,revocation_reason=?,updated_at=? "
                "WHERE id=? AND actor=? AND revoked_at IS NULL",
                (now, str(reason or "revoked by actor").strip()[:256], now, str(grant_id), actor),
            )
        item = self.get(grant_id, actor)
        if item is None:
            raise AgentAppGrantError("Grant was not found.")
        return item

    @staticmethod
    def _facts(current_state: Optional[Mapping[str, Any]], facts: Mapping[str, Any]) -> Dict[str, Any]:
        result = dict(current_state or {})
        result.update(facts)
        for key in ("app", "agent", "connection"):
            nested = result.get(key)
            if isinstance(nested, Mapping):
                result.update(nested)
        return result

    def evaluate(
        self, grant_id: str, actor: str, current_state: Optional[Mapping[str, Any]] = None,
        **facts: Any,
    ) -> Dict[str, Any]:
        actor = _text(actor, "actor", 120)
        grant = self.get(grant_id, actor)
        if grant is None:
            raise AgentAppGrantError("Grant was not found.")
        state = self._facts(current_state, facts)
        reasons: list[str] = []
        incompatible: list[str] = []
        if grant["revoked"]:
            return {"grant_id": grant["id"], "status": "revoked", "reasons": ["grant_revoked"], "recovery_action": "Create a new grant."}
        if state.get("agent_version") is None:
            reasons.append("agent_version_unavailable")
        elif int(state["agent_version"]) != grant["agent_version"]:
            incompatible.append("agent_version_stale")
        if not state.get("manifest_digest"):
            reasons.append("manifest_digest_unavailable")
        elif str(state["manifest_digest"]) != grant["manifest_digest"]:
            incompatible.append("manifest_digest_stale")
        app_state = str(state.get("app_state", state.get("app_status", ""))).lower()
        if not app_state:
            reasons.append("app_state_unavailable")
        elif app_state in {"removed", "stopped", "degraded", "discovered", "configured"}:
            reasons.append("app_unavailable" if app_state != "degraded" else "app_degraded")
        elif app_state == "incompatible":
            incompatible.append("app_incompatible")
        elif app_state != "active":
            reasons.append("app_state_denied")
        available = state.get("available_operations")
        if available is None:
            reasons.append("operation_state_unavailable")
        else:
            available_ids = {str(item) for item in available}
            if not set(grant["operation_ids"]).issubset(available_ids):
                incompatible.append("operation_missing")
        with self._connection() as connection:
            row = connection.execute(
                "SELECT actor,secret_version,test_status,expires_at,revoked_at FROM integration_connections WHERE id=?",
                (grant["connection_id"],),
            ).fetchone()
        if row is None or row["actor"] != actor:
            reasons.append("connection_unavailable")
        else:
            connection_status = str(state.get("connection_status", row["test_status"])).lower()
            if row["revoked_at"] is not None or connection_status == "revoked":
                reasons.append("connection_revoked")
            elif connection_status != "healthy":
                reasons.append("connection_unhealthy")
            expires_at = row["expires_at"]
            if expires_at is not None and float(expires_at) <= self.clock():
                reasons.append("connection_expired")
            if int(row["secret_version"]) != grant["connection_secret_version"]:
                reasons.append("secret_version_stale")
            supplied_secret = state.get("secret_version")
            if supplied_secret is not None and int(supplied_secret) != grant["connection_secret_version"]:
                reasons.append("secret_version_stale")
        if incompatible:
            status = "incompatible"
        elif reasons:
            status = "blocked"
        else:
            status = "active"
        return {
            "grant_id": grant["id"], "status": status,
            "reasons": sorted(set(incompatible + reasons)),
            "recovery_action": "Review the pinned agent, app manifest, operations, and connection." if status != "active" else "",
        }

    status = evaluate

    def require_active(self, grant_id: str, actor: str, current_state: Optional[Mapping[str, Any]] = None, **facts: Any) -> Dict[str, Any]:
        decision = self.evaluate(grant_id, actor, current_state, **facts)
        if decision["status"] != "active":
            raise GrantDeniedError(decision)
        return self.get(grant_id, actor)  # type: ignore[return-value]

    authorize = require_active

    def clone_version(
        self, actor: str, agent_id: str, from_version: Any, to_version: Any,
    ) -> Dict[str, Any]:
        """Carry forward only live grants into an explicitly requested agent version."""
        actor = _text(actor, "actor", 120)
        agent_id = _text(agent_id, "agent_id", 120)
        source_version = _version(from_version)
        target_version = _version(to_version)
        if source_version == target_version:
            raise AgentAppGrantError("from_version and to_version must differ.")
        now = self.clock()
        with self._connection() as connection:
            sources = connection.execute(
                "SELECT * FROM agent_app_grants WHERE actor=? AND agent_id=? "
                "AND agent_version=? AND revoked_at IS NULL ORDER BY created_at ASC",
                (actor, agent_id, source_version),
            ).fetchall()
            cloned = []
            for source in sources:
                existing = connection.execute(
                    "SELECT * FROM agent_app_grants WHERE actor=? AND agent_id=? "
                    "AND agent_version=? AND source_grant_id=?",
                    (actor, agent_id, target_version, source["id"]),
                ).fetchone()
                if existing is not None:
                    cloned.append(self._grant_row(existing))
                    continue
                grant_id = "grant_" + uuid.uuid4().hex
                connection.execute(
                    """INSERT INTO agent_app_grants
                    (id,actor,agent_id,agent_version,app_instance_id,manifest_digest,
                     connection_id,connection_secret_version,operation_ids_json,source_grant_id,
                     revoked_at,revocation_reason,idempotency_key,request_digest,created_at,updated_at)
                    VALUES(?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)""",
                    (grant_id, actor, agent_id, target_version, source["app_instance_id"],
                     source["manifest_digest"], source["connection_id"],
                     source["connection_secret_version"], source["operation_ids_json"],
                     source["id"], None, "", None, source["request_digest"], now, now),
                )
                created = connection.execute(
                    "SELECT * FROM agent_app_grants WHERE id=?", (grant_id,)
                ).fetchone()
                cloned.append(self._grant_row(created))
        return {
            "actor": actor, "agent_id": agent_id,
            "from_version": source_version, "to_version": target_version,
            "cloned": len(cloned), "grants": cloned,
        }

    def dependents(self, connection_id: str, actor: str) -> Dict[str, Any]:
        actor = _text(actor, "actor", 120)
        with self._connection() as connection:
            rows = connection.execute(
                "SELECT id,agent_id,agent_version,app_instance_id,manifest_digest,operation_ids_json,revoked_at "
                "FROM agent_app_grants WHERE connection_id=? AND actor=? ORDER BY created_at DESC",
                (str(connection_id), actor),
            ).fetchall()
        grants = []
        for row in rows:
            item = dict(row)
            item["operation_ids"] = json.loads(item.pop("operation_ids_json"))
            grants.append(item)
        return {"connection_id": str(connection_id), "counts": {
            "total": len(grants),
            "active": sum(item["revoked_at"] is None for item in grants),
            "revoked": sum(item["revoked_at"] is not None for item in grants),
        }, "grants": grants}

    dependency_impact = dependents

    def record_invocation(
        self, actor: str, grant_id: str, *, task_id: str = "", operation_id: str = "",
        approval: str = "not_required", duration_ms: int = 0, outcome: str,
        result_provenance: Optional[Mapping[str, Any]] = None, detail: str = "",
        idempotency_key: Optional[str] = None,
    ) -> Dict[str, Any]:
        actor = _text(actor, "actor", 120)
        grant = self.get(grant_id, actor)
        if grant is None:
            raise AgentAppGrantError("Grant was not found.")
        if operation_id and operation_id not in grant["operation_ids"]:
            raise AgentAppGrantError("operation_id is not granted.")
        try:
            duration = int(duration_ms)
        except (TypeError, ValueError) as error:
            raise AgentAppGrantError("duration_ms is invalid.") from error
        if not 0 <= duration <= 86_400_000:
            raise AgentAppGrantError("duration_ms is invalid.")
        outcome = _text(outcome, "outcome", 80)
        approval = _text(approval, "approval", 80)
        provenance = dict(result_provenance or {})
        if _contains_sensitive(provenance):
            raise AgentAppGrantError("result provenance cannot contain secret-shaped fields.")
        provenance_json = json.dumps(provenance, sort_keys=True, separators=(",", ":"), default=str)
        if len(provenance_json.encode()) > _MAX_PROVENANCE_BYTES:
            raise AgentAppGrantError("result provenance is too large.")
        key = None if idempotency_key is None else _text(idempotency_key, "idempotency_key", 160)
        event_id = "inv_" + uuid.uuid4().hex
        now = self.clock()
        with self._connection() as connection:
            if key:
                existing = connection.execute(
                    "SELECT * FROM capability_invocation_audit WHERE actor=? AND idempotency_key=?",
                    (actor, key),
                ).fetchone()
                if existing:
                    return self._audit_row(existing)
            connection.execute(
                """INSERT INTO capability_invocation_audit
                (event_id,actor,grant_id,task_id,agent_id,agent_version,app_instance_id,
                 manifest_digest,connection_id,operation_id,approval,duration_ms,outcome,
                 result_provenance_json,detail,idempotency_key,created_at)
                VALUES(?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)""",
                (event_id, actor, grant_id, str(task_id or "")[:160], grant["agent_id"],
                 grant["agent_version"], grant["app_instance_id"], grant["manifest_digest"],
                 grant["connection_id"], str(operation_id or "")[:120], approval, duration,
                 outcome, provenance_json, str(detail or "")[:256], key, now),
            )
            row = connection.execute(
                "SELECT * FROM capability_invocation_audit WHERE event_id=?", (event_id,)
            ).fetchone()
        return self._audit_row(row)  # type: ignore[arg-type]

    audit_invocation = record_invocation

    def audit(self, actor: str, limit: int = 100) -> list[Dict[str, Any]]:
        actor = _text(actor, "actor", 120)
        limit = max(1, min(int(limit), 200))
        with self._connection() as connection:
            rows = connection.execute(
                "SELECT * FROM capability_invocation_audit WHERE actor=? "
                "ORDER BY created_at DESC LIMIT ?", (actor, limit)
            ).fetchall()
        return [self._audit_row(row) for row in rows]

    list_audit = audit


GrantStore = AgentAppGrantStore
