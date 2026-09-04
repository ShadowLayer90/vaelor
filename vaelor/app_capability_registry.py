"""Durable registry for installed-app manifests and managed app instances.

This module owns only the app boundary.  Connections, grants, and invocation
records consume its ID/digest-based authorization primitive but are persisted
by their respective control-plane workers.
"""

from __future__ import annotations

import base64
import hashlib
import json
import os
import re
import sqlite3
import threading
from dataclasses import dataclass, field
import time
from contextlib import closing
from pathlib import Path
from typing import Any, Mapping, Optional

from .app_capability_manifest import (
    AppCapabilityManifest,
    AppOperation,
    validate_manifest,
)
from .runtime_paths import env_value, state_path


class RegistryError(ValueError):
    """Base error for invalid registry operations or blocked authorization."""


class IncompatibleManifestError(RegistryError):
    """Raised when a running app instance presents a different manifest digest."""


class AuthorizationError(RegistryError):
    """Raised whenever an app capability cannot be authorized fail-closed."""


@dataclass(frozen=True)
class AppInstanceRegistration:
    """Typed projection of a persisted managed app instance."""

    instance_id: str
    project: str
    service: str
    app_id: str
    app_label: str
    manifest_digest: str
    state: str
    health: str
    compatibility: str
    observed_manifest_digest: str = ""
    reason: str = ""
    health_evidence: Mapping[str, Any] = field(default_factory=dict)
    created_at: int = 0
    updated_at: int = 0
    last_seen_at: Optional[int] = None

    @classmethod
    def from_dict(cls, value: Mapping[str, Any]) -> "AppInstanceRegistration":
        return cls(
            instance_id=str(value["instance_id"]),
            project=str(value["project"]),
            service=str(value["service"]),
            app_id=str(value["app_id"]),
            app_label=str(value["app_label"]),
            manifest_digest=str(value["manifest_digest"]),
            state=str(value["state"]),
            health=str(value["health"]),
            compatibility=str(value["compatibility"]),
            observed_manifest_digest=str(value.get("observed_manifest_digest", "")),
            reason=str(value.get("reason", "")),
            health_evidence=dict(value.get("health_evidence", {})),
            created_at=int(value.get("created_at", 0)),
            updated_at=int(value.get("updated_at", 0)),
            last_seen_at=value.get("last_seen_at"),
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "instance_id": self.instance_id,
            "project": self.project,
            "service": self.service,
            "app_id": self.app_id,
            "app_label": self.app_label,
            "manifest_digest": self.manifest_digest,
            "observed_manifest_digest": self.observed_manifest_digest or self.manifest_digest,
            "state": self.state,
            "health": self.health,
            "compatibility": self.compatibility,
            "reason": self.reason,
            "health_evidence": dict(self.health_evidence),
            "created_at": self.created_at,
            "updated_at": self.updated_at,
            "last_seen_at": self.last_seen_at,
        }

APP_STATES = {"discovered", "configured", "active", "degraded", "stopped", "incompatible", "removed"}
HEALTH_STATES = {"unknown", "healthy", "unhealthy"}
COMPATIBILITY_STATES = {"unknown", "compatible", "incompatible"}
_WORKLOAD_ID_RE = re.compile(r"^[A-Za-z0-9][A-Za-z0-9_.-]{0,127}$")
_MAX_PAGE_SIZE = 100


def _bounded_text(value: Any, field_name: str, maximum: int = 256) -> str:
    if not isinstance(value, str):
        raise RegistryError(f"{field_name} must be text.")
    value = value.strip()
    if not value or len(value) > maximum:
        raise RegistryError(f"{field_name} must be non-empty and bounded.")
    return value


def _workload_part(value: Any, field_name: str) -> str:
    value = _bounded_text(value, field_name, 128)
    if not _WORKLOAD_ID_RE.fullmatch(value):
        raise RegistryError(f"{field_name} is not a valid managed workload identity.")
    return value


def _canonical_workload_identity(project: Any, service: Any) -> tuple[str, str]:
    """Validate the collision domain used to derive stable app-instance IDs."""
    return _workload_part(project, "project"), _workload_part(service, "service")


def app_instance_id_for_workload(project: str, service: str) -> str:
    """Return the stable server-owned ID for a managed project/service pair."""
    project, service = _canonical_workload_identity(project, service)
    encoded = json.dumps(
        {"project": project, "service": service},
        sort_keys=True, separators=(",", ":"), ensure_ascii=True,
    ).encode("ascii")
    return "appinst_" + hashlib.sha256(encoded).hexdigest()[:32]


derive_app_instance_id = app_instance_id_for_workload


def _now() -> int:
    return int(time.time())


def _json_object(value: Any, field_name: str, *, max_bytes: int = 128 * 1024) -> dict[str, Any]:
    if not isinstance(value, Mapping):
        raise RegistryError(f"{field_name} must be an object.")
    try:
        encoded = json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"), allow_nan=False)
    except (TypeError, ValueError) as error:
        raise RegistryError(f"{field_name} must contain JSON values.") from error
    if len(encoded.encode("utf-8")) > max_bytes:
        raise RegistryError(f"{field_name} is too large.")
    return dict(value)


def _page_size(value: Any) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise RegistryError("limit must be an integer.")
    if value < 1:
        raise RegistryError("limit must be at least 1.")
    return min(value, _MAX_PAGE_SIZE)


def _encode_cursor(created_at: int, instance_id: str) -> str:
    raw = json.dumps([created_at, instance_id], separators=(",", ":")).encode("utf-8")
    return base64.urlsafe_b64encode(raw).decode("ascii").rstrip("=")


def _decode_cursor(value: Optional[str]) -> Optional[tuple[int, str]]:
    if not value:
        return None
    if not isinstance(value, str) or len(value) > 256:
        raise RegistryError("cursor is invalid.")
    try:
        raw = base64.urlsafe_b64decode(value + "=" * (-len(value) % 4))
        decoded = json.loads(raw.decode("utf-8"))
        if not isinstance(decoded, list) or len(decoded) != 2:
            raise ValueError
        created_at, instance_id = decoded
        if isinstance(created_at, bool) or not isinstance(created_at, int) or not isinstance(instance_id, str):
            raise ValueError
        return created_at, instance_id
    except (ValueError, TypeError, UnicodeError, json.JSONDecodeError) as error:
        raise RegistryError("cursor is invalid.") from error


def _validate_state(value: Any, allowed: set[str], field_name: str) -> str:
    value = _bounded_text(value, field_name, 32).lower()
    if value not in allowed:
        raise RegistryError(f"{field_name} is unsupported.")
    return value


class AppCapabilityRegistry:
    """SQLite-backed manifest and installed-app instance registry."""

    def __init__(self, database_path: Optional[str] = None):
        self.database_path = database_path or env_value(
            "VAELOR_CAPABILITY_REGISTRY_DB", "PM_CAPABILITY_REGISTRY_DB",
            state_path("integrations/app-capability-registry.sqlite3"),
        )
        self._schema_ready = False
        self._schema_lock = threading.Lock()
        self._connect().close()

    def _connect(self) -> sqlite3.Connection:
        path = Path(self.database_path)
        path.parent.mkdir(parents=True, exist_ok=True)
        connection = sqlite3.connect(str(path), timeout=15)
        connection.row_factory = sqlite3.Row
        connection.execute("PRAGMA foreign_keys = ON")  # pairs-with: sqlite-foreign-keys-spaced
        connection.execute("PRAGMA busy_timeout = 15000")
        self._ensure_schema(connection)
        try:
            os.chmod(path, 0o660)
        except PermissionError:
            pass
        return connection

    def _ensure_schema(self, connection: sqlite3.Connection) -> None:
        if self._schema_ready:
            return
        with self._schema_lock:
            if self._schema_ready:
                return
            connection.execute("PRAGMA journal_mode = WAL")  # pairs-with: sqlite-journal-mode-spaced
            connection.executescript(
                """
                CREATE TABLE IF NOT EXISTS app_capability_manifests (
                    manifest_digest TEXT PRIMARY KEY,
                    app_id TEXT NOT NULL,
                    app_version TEXT NOT NULL,
                    manifest_version INTEGER NOT NULL,
                    manifest_json TEXT NOT NULL,
                    created_at INTEGER NOT NULL
                );
                CREATE TABLE IF NOT EXISTS app_instances (
                    instance_id TEXT PRIMARY KEY,
                    project TEXT NOT NULL,
                    service TEXT NOT NULL,
                    app_id TEXT NOT NULL,
                    app_label TEXT NOT NULL,
                    manifest_digest TEXT NOT NULL,
                    observed_manifest_digest TEXT NOT NULL DEFAULT '',
                    state TEXT NOT NULL,
                    health TEXT NOT NULL,
                    compatibility TEXT NOT NULL,
                    reason TEXT NOT NULL DEFAULT '',
                    health_evidence_json TEXT NOT NULL DEFAULT '{}',
                    created_at INTEGER NOT NULL,
                    updated_at INTEGER NOT NULL,
                    last_seen_at INTEGER,
                    UNIQUE(project, service),
                    FOREIGN KEY (manifest_digest) REFERENCES app_capability_manifests(manifest_digest)
                );
                CREATE TABLE IF NOT EXISTS app_dependency_references (
                    instance_id TEXT NOT NULL,
                    dependent_type TEXT NOT NULL,
                    dependent_id TEXT NOT NULL,
                    reference_json TEXT NOT NULL DEFAULT '{}',
                    created_at INTEGER NOT NULL,
                    updated_at INTEGER NOT NULL,
                    PRIMARY KEY (instance_id, dependent_type, dependent_id),
                    FOREIGN KEY (instance_id) REFERENCES app_instances(instance_id) ON DELETE CASCADE
                );
                CREATE INDEX IF NOT EXISTS app_instances_order_idx
                    ON app_instances(created_at, instance_id);
                CREATE INDEX IF NOT EXISTS app_dependency_instance_idx
                    ON app_dependency_references(instance_id, created_at, dependent_id);
                """
            )
            connection.commit()
            self._schema_ready = True

    @staticmethod
    def _manifest_row(manifest: AppCapabilityManifest, now: int) -> tuple[Any, ...]:
        return (
            manifest.manifest_digest, manifest.app_id, manifest.app_version,
            manifest.manifest_version,
            json.dumps(manifest.to_dict(), ensure_ascii=False, sort_keys=True, separators=(",", ":")), now,
        )

    def _store_manifest(self, connection: sqlite3.Connection, manifest: AppCapabilityManifest) -> None:
        connection.execute(
            """INSERT OR IGNORE INTO app_capability_manifests(
                manifest_digest, app_id, app_version, manifest_version, manifest_json, created_at
            ) VALUES (?, ?, ?, ?, ?, ?)""",
            self._manifest_row(manifest, _now()),
        )

    @staticmethod
    def _instance(row: sqlite3.Row) -> dict[str, Any]:
        evidence = json.loads(row["health_evidence_json"])
        return {
            "instance_id": row["instance_id"],
            "project": row["project"],
            "service": row["service"],
            "app_id": row["app_id"],
            "app_label": row["app_label"],
            "manifest_digest": row["manifest_digest"],
            "observed_manifest_digest": row["observed_manifest_digest"] or row["manifest_digest"],
            "state": row["state"],
            "health": row["health"],
            "compatibility": row["compatibility"],
            "reason": row["reason"],
            "health_evidence": evidence,
            "created_at": row["created_at"],
            "updated_at": row["updated_at"],
            "last_seen_at": row["last_seen_at"],
        }

    def register_app_instance(
        self,
        *,
        project: str,
        service: str,
        manifest: AppCapabilityManifest | Mapping[str, Any],
        state: str = "discovered",
        health: str = "unknown",
        compatibility: str = "compatible",
        health_evidence: Optional[Mapping[str, Any]] = None,
        instance_id: Optional[str] = None,
    ) -> dict[str, Any]:
        """Register an instance; recreate-safe identity comes from project/service."""
        project, service = _canonical_workload_identity(project, service)
        expected_id = app_instance_id_for_workload(project, service)
        if instance_id is not None and instance_id != expected_id:
            raise RegistryError("instance_id must match the server-derived workload identity.")
        manifest = validate_manifest(manifest)
        state = _validate_state(state, APP_STATES, "state")
        health = _validate_state(health, HEALTH_STATES, "health")
        compatibility = _validate_state(compatibility, COMPATIBILITY_STATES, "compatibility")
        evidence = _json_object(health_evidence or {}, "health_evidence")
        now = _now()
        with closing(self._connect()) as connection:
            self._store_manifest(connection, manifest)
            existing = connection.execute(
                "SELECT * FROM app_instances WHERE instance_id=?", (expected_id,)
            ).fetchone()
            if existing is not None and existing["manifest_digest"] != manifest.manifest_digest:
                connection.execute(
                    """UPDATE app_instances SET observed_manifest_digest=?, state='incompatible',
                       compatibility='incompatible', reason=?, health_evidence_json=?,
                       updated_at=?, last_seen_at=? WHERE instance_id=?""",
                    (manifest.manifest_digest, "The installed manifest digest changed; review and recreate the grant.",
                     json.dumps(evidence, sort_keys=True, separators=(",", ":")), now, now, expected_id),
                )
                connection.commit()
                raise IncompatibleManifestError("The app instance presented an incompatible manifest digest.")
            if existing is None:
                connection.execute(
                    """INSERT INTO app_instances(
                        instance_id, project, service, app_id, app_label, manifest_digest,
                        observed_manifest_digest, state, health, compatibility, reason,
                        health_evidence_json, created_at, updated_at, last_seen_at
                    ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, '', ?, ?, ?, ?)""",
                    (expected_id, project, service, manifest.app_id, manifest.app_label,
                     manifest.manifest_digest, manifest.manifest_digest, state, health,
                     compatibility, json.dumps(evidence, sort_keys=True, separators=(",", ":")), now, now, now),
                )
            else:
                connection.execute(
                    """UPDATE app_instances SET app_id=?, app_label=?, state=?, health=?,
                       compatibility=?, reason='', health_evidence_json=?, updated_at=?, last_seen_at=?
                       WHERE instance_id=?""",
                    (manifest.app_id, manifest.app_label, state, health, compatibility,
                     json.dumps(evidence, sort_keys=True, separators=(",", ":")), now, now, expected_id),
                )
            connection.commit()
            row = connection.execute("SELECT * FROM app_instances WHERE instance_id=?", (expected_id,)).fetchone()
        return self._instance(row)

    register = register_app_instance

    def reconcile_app_instance(
        self,
        instance_id: str,
        *,
        state: Optional[str] = None,
        health: Optional[str] = None,
        compatibility: Optional[str] = None,
        observed_manifest_digest: Optional[str] = None,
        runtime_container_id: Optional[str] = None,
        health_evidence: Optional[Mapping[str, Any]] = None,
        reason: Optional[str] = None,
    ) -> dict[str, Any]:
        """Reconcile mutable runtime facts without changing stable identity."""
        instance_id = _bounded_text(instance_id, "instance_id", 128)
        evidence = None if health_evidence is None else _json_object(health_evidence, "health_evidence")
        if runtime_container_id is not None:
            runtime_container_id = _bounded_text(runtime_container_id, "runtime_container_id", 256)
            if evidence is None:
                evidence = {}
            runtime = evidence.setdefault("runtime", {})
            if not isinstance(runtime, dict):
                raise RegistryError("health_evidence.runtime must be an object.")
            runtime["container_id"] = runtime_container_id
        now = _now()
        with closing(self._connect()) as connection:
            current = connection.execute("SELECT * FROM app_instances WHERE instance_id=?", (instance_id,)).fetchone()
            if current is None:
                raise RegistryError("The app instance was not found.")
            if health_evidence is None and runtime_container_id is not None:
                prior_evidence = json.loads(current["health_evidence_json"])
                prior_evidence.update(evidence)
                evidence = prior_evidence
            if evidence is None:
                evidence = json.loads(current["health_evidence_json"])
            current_state = state or current["state"]
            current_health = health or current["health"]
            current_compat = compatibility or current["compatibility"]
            current_state = _validate_state(current_state, APP_STATES, "state")
            current_health = _validate_state(current_health, HEALTH_STATES, "health")
            current_compat = _validate_state(current_compat, COMPATIBILITY_STATES, "compatibility")
            observed = observed_manifest_digest if observed_manifest_digest is not None else current["observed_manifest_digest"]
            if observed != current["manifest_digest"]:
                current_state, current_compat = "incompatible", "incompatible"
                reason = reason or "The installed manifest digest changed; review and recreate the grant."
            connection.execute(
                """UPDATE app_instances SET state=?, health=?, compatibility=?,
                   observed_manifest_digest=?, reason=?, health_evidence_json=?, updated_at=?, last_seen_at=?
                   WHERE instance_id=?""",
                (current_state, current_health, current_compat, observed, reason or current["reason"],
                 json.dumps(evidence, sort_keys=True, separators=(",", ":")), now, now, instance_id),
            )
            connection.commit()
            row = connection.execute("SELECT * FROM app_instances WHERE instance_id=?", (instance_id,)).fetchone()
        return self._instance(row)

    reconcile = reconcile_app_instance

    def get_manifest(self, manifest_digest: str) -> Optional[AppCapabilityManifest]:
        if not isinstance(manifest_digest, str) or not re.fullmatch(r"[0-9a-f]{64}", manifest_digest):
            raise RegistryError("manifest_digest is invalid.")
        with closing(self._connect()) as connection:
            row = connection.execute(
                "SELECT manifest_json FROM app_capability_manifests WHERE manifest_digest=?",
                (manifest_digest,),
            ).fetchone()
        return validate_manifest(json.loads(row["manifest_json"])) if row else None
    def get_app_instance(self, instance_id: str) -> Optional[dict[str, Any]]:
        instance_id = _bounded_text(instance_id, "instance_id", 128)
        with closing(self._connect()) as connection:
            row = connection.execute("SELECT * FROM app_instances WHERE instance_id=?", (instance_id,)).fetchone()
        return self._instance(row) if row else None

    get = get_app_instance

    def list_app_instances(self, *, limit: int = 50, cursor: Optional[str] = None) -> dict[str, Any]:
        limit = _page_size(limit)
        decoded = _decode_cursor(cursor)
        query = "SELECT * FROM app_instances"
        values: list[Any] = []
        if decoded:
            query += " WHERE created_at > ? OR (created_at = ? AND instance_id > ?)"
            values.extend((decoded[0], decoded[0], decoded[1]))
        query += " ORDER BY created_at, instance_id LIMIT ?"
        values.append(limit + 1)
        with closing(self._connect()) as connection:
            rows = connection.execute(query, values).fetchall()
        items = [self._instance(row) for row in rows[:limit]]
        next_cursor = _encode_cursor(rows[limit]["created_at"], rows[limit]["instance_id"]) if len(rows) > limit else None
        return {"items": items, "next_cursor": next_cursor, "limit": limit}

    list = list_app_instances

    def register_dependency_reference(
        self,
        instance_id: str,
        *,
        dependent_type: str,
        dependent_id: str,
        reference: Optional[Mapping[str, Any]] = None,
    ) -> dict[str, Any]:
        instance_id = _bounded_text(instance_id, "instance_id", 128)
        dependent_type = _bounded_text(dependent_type, "dependent_type", 64)
        dependent_id = _bounded_text(dependent_id, "dependent_id", 128)
        reference = _json_object(reference or {}, "reference", max_bytes=32 * 1024)
        now = _now()
        with closing(self._connect()) as connection:
            if connection.execute("SELECT 1 FROM app_instances WHERE instance_id=?", (instance_id,)).fetchone() is None:
                raise RegistryError("The app instance was not found.")
            connection.execute(
                """INSERT INTO app_dependency_references(
                    instance_id, dependent_type, dependent_id, reference_json, created_at, updated_at
                ) VALUES (?, ?, ?, ?, ?, ?)
                ON CONFLICT(instance_id, dependent_type, dependent_id) DO UPDATE SET
                    reference_json=excluded.reference_json, updated_at=excluded.updated_at""",
                (instance_id, dependent_type, dependent_id,
                 json.dumps(reference, sort_keys=True, separators=(",", ":")), now, now),
            )
            connection.commit()
        return {
            "instance_id": instance_id, "dependent_type": dependent_type,
            "dependent_id": dependent_id, "reference": reference,
        }

    add_dependency_reference = register_dependency_reference

    def remove_dependency_reference(self, instance_id: str, *, dependent_type: str, dependent_id: str) -> bool:
        with closing(self._connect()) as connection:
            cursor = connection.execute(
                "DELETE FROM app_dependency_references WHERE instance_id=? AND dependent_type=? AND dependent_id=?",
                (str(instance_id), str(dependent_type), str(dependent_id)),
            )
            connection.commit()
        return cursor.rowcount == 1

    def list_dependents(self, instance_id: str, *, limit: int = 50, cursor: Optional[str] = None) -> dict[str, Any]:
        instance_id = _bounded_text(instance_id, "instance_id", 128)
        limit = _page_size(limit)
        decoded = _decode_cursor(cursor)
        query = """SELECT * FROM app_dependency_references WHERE instance_id=?"""
        values: list[Any] = [instance_id]
        if decoded:
            query += " AND (created_at > ? OR (created_at = ? AND dependent_id > ?))"
            values.extend((decoded[0], decoded[0], decoded[1]))
        query += " ORDER BY created_at, dependent_id LIMIT ?"
        values.append(limit + 1)
        with closing(self._connect()) as connection:
            rows = connection.execute(query, values).fetchall()
        items = [
            {"instance_id": row["instance_id"], "dependent_type": row["dependent_type"],
             "dependent_id": row["dependent_id"], "reference": json.loads(row["reference_json"]),
             "created_at": row["created_at"], "updated_at": row["updated_at"]}
            for row in rows[:limit]
        ]
        next_cursor = _encode_cursor(rows[limit]["created_at"], rows[limit]["dependent_id"]) if len(rows) > limit else None
        return {"items": items, "next_cursor": next_cursor, "limit": limit}

    dependents = list_dependents
    list_dependency_references = list_dependents
    dependency_references = list_dependents

    def authorize_operation(
        self,
        instance_id: str,
        *,
        manifest_digest: str,
        operation_id: str,
        connection_healthy: Optional[bool] = None,
    ) -> AppOperation:
        """Authorize only a pinned instance/digest/operation tuple.

        The method intentionally has no label argument.  Required connections
        are denied unless the connection worker supplies an explicit healthy
        result.
        """
        instance = self.get_app_instance(instance_id)
        if instance is None:
            raise AuthorizationError("The app instance is unavailable.")
        if instance["state"] != "active":
            raise AuthorizationError(f"The app instance is {instance['state']}.")
        if instance["health"] != "healthy":
            raise AuthorizationError("The app instance health is not healthy.")
        if instance["compatibility"] != "compatible":
            raise AuthorizationError("The app instance is not compatible with its pinned manifest.")
        if manifest_digest != instance["manifest_digest"]:
            raise AuthorizationError("The requested manifest digest is not pinned to this instance.")
        with closing(self._connect()) as connection:
            row = connection.execute(
                "SELECT manifest_json FROM app_capability_manifests WHERE manifest_digest=?",
                (manifest_digest,),
            ).fetchone()
        if row is None:
            raise AuthorizationError("The pinned manifest is unavailable.")
        manifest = validate_manifest(json.loads(row["manifest_json"]))
        operation = manifest.operation(operation_id)
        if operation is None:
            raise AuthorizationError("The requested operation is not present in the pinned manifest.")
        if (manifest.requires_connection or operation.requires_connection) and connection_healthy is not True:
            raise AuthorizationError("The required app connection is not healthy.")
        return operation

    def can_invoke(self, instance_id: str, *, manifest_digest: str, operation_id: str, connection_healthy: Optional[bool] = None) -> bool:
        try:
            self.authorize_operation(
                instance_id, manifest_digest=manifest_digest, operation_id=operation_id,
                connection_healthy=connection_healthy,
            )
        except AuthorizationError:
            return False
        return True


__all__ = [
    "APP_STATES", "AppInstanceRegistration", "AuthorizationError",
    "COMPATIBILITY_STATES", "HEALTH_STATES", "AppCapabilityRegistry",
    "IncompatibleManifestError", "RegistryError", "app_instance_id_for_workload",
    "derive_app_instance_id",
]
