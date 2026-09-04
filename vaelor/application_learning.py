"""Sanitized operational lessons from researched application deployments.

This store is deliberately separate from Assistant chat memory.  It records
only server-verified identities, content fingerprints, approvals, and bounded
outcomes.  Raw research content, prompts, credentials, and user-provided
configuration values never cross this boundary.
"""

from __future__ import annotations

import hashlib
import json
import os
import re
import sqlite3
import threading
import time
import unicodedata
from contextlib import closing
from pathlib import Path
from typing import Any, Dict, Optional
from urllib.parse import urlsplit, urlunsplit

try:
    import grp
except ImportError:
    grp = None

from .application_deployments import content_digest
from .runtime_paths import env_value, jobs_group_id, state_path


_DIGEST = re.compile(r"^sha256:[0-9a-f]{64}$")
_SAFE_ID = re.compile(r"^[a-z0-9](?:[a-z0-9-]{0,62}[a-z0-9])?$")
_SECRET = re.compile(
    r"(?:-----BEGIN [A-Z ]*PRIVATE KEY-----|\b(?:password|token|secret|api[_-]?key)\s*[:=])",
    re.IGNORECASE,
)
EVENTS = frozenset({"research", "approval", "deployment", "recovery"})
COMPATIBILITY = frozenset({"verified", "unsupported", "unknown", ""})
OUTCOMES = frozenset({"", "approved", "succeeded", "failed", "rolled_back"})
RECOVERY_CATEGORIES = frozenset({
    "", "cancelled", "runtime_unavailable", "image_acquisition",
    "configuration_validation", "health_verification", "project_conflict",
    "authorization", "rollback_completed", "rollback_incomplete",
    "deployment_failed",
})


class ApplicationLearningError(ValueError):
    """Safe validation failure for operational deployment lessons."""


def _text(value: Any, name: str, limit: int, *, required: bool = False) -> str:
    text = unicodedata.normalize("NFKC", str(value or "")).strip()
    if required and not text:
        raise ApplicationLearningError(f"Enter {name}.")
    if len(text) > limit:
        raise ApplicationLearningError(f"{name.capitalize()} is too long.")
    if any(unicodedata.category(char) in {"Cc", "Cf", "Cs"} for char in text):
        raise ApplicationLearningError(f"{name.capitalize()} contains unsafe characters.")
    if _SECRET.search(text):
        raise ApplicationLearningError("Secrets cannot be stored as deployment lessons.")
    return text


def _digest(value: Any, name: str, *, required: bool = False) -> str:
    digest = str(value or "").strip().lower()
    if not digest and not required:
        return ""
    if not _DIGEST.fullmatch(digest):
        raise ApplicationLearningError(f"{name.capitalize()} must be a SHA-256 digest.")
    return digest


def _source_fingerprints(manifest: Dict[str, Any]) -> list[Dict[str, str]]:
    result: list[Dict[str, str]] = []
    for source in list(manifest.get("sources") or [])[:24]:
        if not isinstance(source, dict) or not source.get("verified"):
            continue
        parsed = urlsplit(str(source.get("url") or ""))
        if parsed.scheme.lower() != "https" or not parsed.hostname:
            continue
        identity = urlunsplit(("https", parsed.netloc.lower(), parsed.path, parsed.query, ""))
        digest = _digest(source.get("sha256"), "source content digest", required=True)
        result.append({
            "identity_sha256": "sha256:" + hashlib.sha256(identity.encode("utf-8")).hexdigest(),
            "content_sha256": digest,
        })
    return sorted(result, key=lambda item: (item["identity_sha256"], item["content_sha256"]))


def deployment_failure_category(error: BaseException, *, rollback_completed: bool) -> str:
    """Map an operational error to a non-sensitive, reusable category."""
    text = str(error).lower()
    if "cancel" in text:
        return "cancelled"
    if not rollback_completed:
        return "rollback_incomplete"
    if "docker is not installed" in text or "runtime" in text:
        return "runtime_unavailable"
    if "pull" in text or "image" in text or "manifest" in text:
        return "image_acquisition"
    if "config" in text or "compose" in text or "policy" in text:
        return "configuration_validation"
    if "health" in text or "running" in text or "status" in text:
        return "health_verification"
    if "already uses" in text or "conflict" in text:
        return "project_conflict"
    if "credential" in text or "permission" in text or "authoriz" in text:
        return "authorization"
    return "deployment_failed"


class ApplicationLearningStore:
    """Append-only SQLite ledger of bounded researched-deployment lessons."""

    def __init__(self, database_path: Optional[str] = None):
        self.database_path = database_path or env_value(
            "VAELOR_APPLICATION_LEARNING_DB", "PM_APPLICATION_LEARNING_DB",
            state_path("applications/deployment-lessons.sqlite3"),
        )
        self._schema_ready = False
        self._schema_lock = threading.Lock()
        self._connect().close()

    def _connect(self) -> sqlite3.Connection:
        path = Path(self.database_path)
        path.parent.mkdir(parents=True, exist_ok=True)
        try:
            descriptor = os.open(str(path), os.O_CREAT | os.O_EXCL | os.O_RDWR, 0o660)
        except FileExistsError:
            pass
        else:
            os.close(descriptor)
        try:
            os.chmod(path, 0o660)
        except PermissionError:
            if path.stat().st_mode & 0o777 != 0o660:
                raise
        group_id = jobs_group_id(grp)
        if group_id is not None:
            try:
                os.chown(path, -1, group_id)
            except PermissionError:
                pass
        connection = sqlite3.connect(str(path), timeout=10)
        connection.row_factory = sqlite3.Row
        self._ensure_schema(connection)
        return connection

    def _ensure_schema(self, connection: sqlite3.Connection) -> None:
        if self._schema_ready:
            return
        with self._schema_lock:
            if self._schema_ready:
                return
            connection.execute("PRAGMA journal_mode = WAL")  # pairs-with: sqlite-journal-mode-spaced
            connection.executescript("""
                CREATE TABLE IF NOT EXISTS application_deployment_lessons (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    draft_id TEXT NOT NULL,
                    actor TEXT NOT NULL,
                    event TEXT NOT NULL,
                    app_id TEXT NOT NULL,
                    application_query TEXT NOT NULL,
                    source_fingerprints_json TEXT NOT NULL DEFAULT '[]',
                    compatibility TEXT NOT NULL DEFAULT '',
                    compatibility_reason TEXT NOT NULL DEFAULT '',
                    manifest_digest TEXT NOT NULL DEFAULT '',
                    configuration_digest TEXT NOT NULL DEFAULT '',
                    compose_digest TEXT NOT NULL DEFAULT '',
                    outcome TEXT NOT NULL DEFAULT '',
                    recovery_category TEXT NOT NULL DEFAULT '',
                    created_at INTEGER NOT NULL
                );
                CREATE INDEX IF NOT EXISTS idx_application_lessons_actor
                    ON application_deployment_lessons(actor, created_at DESC, id DESC);
                CREATE INDEX IF NOT EXISTS idx_application_lessons_app
                    ON application_deployment_lessons(actor, app_id, created_at DESC);
            """)
            connection.commit()
            self._schema_ready = True

    def record_research(self, draft: Dict[str, Any]) -> Dict[str, Any]:
        manifest = draft.get("manifest") or {}
        compatibility = manifest.get("compatibility") or {}
        return self._append(
            draft, "research",
            source_fingerprints=_source_fingerprints(manifest),
            compatibility=str(compatibility.get("status") or ""),
            compatibility_reason=compatibility.get("reason"),
        )

    def record_approval(self, draft: Dict[str, Any]) -> Dict[str, Any]:
        return self._append(draft, "approval", outcome="approved")

    def record_deployment(
        self, draft: Dict[str, Any], *, succeeded: bool,
        recovery_category: str = "",
    ) -> Dict[str, Any]:
        return self._append(
            draft, "deployment",
            outcome="succeeded" if succeeded else "failed",
            recovery_category=recovery_category,
        )

    def record_recovery(
        self, draft: Dict[str, Any], *, completed: bool,
        recovery_category: str,
    ) -> Dict[str, Any]:
        return self._append(
            draft, "recovery",
            outcome="rolled_back" if completed else "failed",
            recovery_category=recovery_category,
        )

    def _append(self, draft: Dict[str, Any], event: str, **details: Any) -> Dict[str, Any]:
        if event not in EVENTS or not isinstance(draft, dict):
            raise ApplicationLearningError("Choose a valid deployment lesson event.")
        intent = draft.get("intent") or {}
        manifest = draft.get("manifest") or {}
        app = manifest.get("application") or {}
        app_id = str(app.get("id") or "unknown").strip().lower()
        if app_id != "unknown" and not _SAFE_ID.fullmatch(app_id):
            raise ApplicationLearningError("Application identity is invalid.")
        compatibility = str(details.get("compatibility") or "").lower()
        outcome = str(details.get("outcome") or "").lower()
        recovery = str(details.get("recovery_category") or "").lower()
        if compatibility not in COMPATIBILITY or outcome not in OUTCOMES:
            raise ApplicationLearningError("Deployment lesson outcome is invalid.")
        if recovery not in RECOVERY_CATEGORIES:
            raise ApplicationLearningError("Deployment recovery category is invalid.")
        fingerprints = details.get("source_fingerprints")
        if fingerprints is None:
            fingerprints = _source_fingerprints(manifest)
        if not isinstance(fingerprints, list) or len(fingerprints) > 24:
            raise ApplicationLearningError("Source fingerprints are invalid.")
        configuration = draft.get("configuration")
        configuration_digest = content_digest(configuration) if isinstance(configuration, dict) else ""
        values = (
            _text(draft.get("id"), "a draft id", 120, required=True),
            _text(draft.get("actor"), "an actor", 120, required=True),
            event, app_id,
            _text(intent.get("application_query"), "an application query", 160, required=True),
            json.dumps(fingerprints, sort_keys=True, separators=(",", ":")),
            compatibility,
            _text(details.get("compatibility_reason"), "a compatibility reason", 800),
            _digest(draft.get("manifest_digest"), "manifest digest"),
            configuration_digest,
            _digest(draft.get("compose_digest"), "Compose digest"),
            outcome, recovery, int(time.time()),
        )
        with closing(self._connect()) as connection:
            cursor = connection.execute("""
                INSERT INTO application_deployment_lessons(
                    draft_id,actor,event,app_id,application_query,
                    source_fingerprints_json,compatibility,compatibility_reason,
                    manifest_digest,configuration_digest,compose_digest,outcome,
                    recovery_category,created_at
                ) VALUES(?,?,?,?,?,?,?,?,?,?,?,?,?,?)
            """, values)
            connection.commit()
            lesson_id = cursor.lastrowid
        return self.get(int(lesson_id), values[1])

    @staticmethod
    def _record(row: sqlite3.Row) -> Dict[str, Any]:
        return {
            "id": int(row["id"]), "draft_id": row["draft_id"],
            "event": row["event"], "app_id": row["app_id"],
            "application_query": row["application_query"],
            "source_fingerprints": json.loads(row["source_fingerprints_json"]),
            "compatibility": row["compatibility"],
            "compatibility_reason": row["compatibility_reason"],
            "manifest_digest": row["manifest_digest"],
            "configuration_digest": row["configuration_digest"],
            "compose_digest": row["compose_digest"], "outcome": row["outcome"],
            "recovery_category": row["recovery_category"],
            "created_at": int(row["created_at"]),
        }

    def get(self, lesson_id: int, actor: str) -> Dict[str, Any]:
        with closing(self._connect()) as connection:
            row = connection.execute(
                "SELECT * FROM application_deployment_lessons WHERE id=? AND actor=?",
                (int(lesson_id), _text(actor, "an actor", 120, required=True)),
            ).fetchone()
        if row is None:
            raise ApplicationLearningError("Deployment lesson was not found.")
        return self._record(row)

    def list(self, actor: str, *, app_id: str = "", limit: int = 25) -> list[Dict[str, Any]]:
        actor_name = _text(actor, "an actor", 120, required=True)
        count = max(1, min(int(limit), 100))
        query = "SELECT * FROM application_deployment_lessons WHERE actor=?"
        values: list[Any] = [actor_name]
        if app_id:
            normalized_id = str(app_id).strip().lower()
            if normalized_id != "unknown" and not _SAFE_ID.fullmatch(normalized_id):
                raise ApplicationLearningError("Application identity is invalid.")
            query += " AND app_id=?"
            values.append(normalized_id)
        query += " ORDER BY created_at DESC,id DESC LIMIT ?"
        values.append(count)
        with closing(self._connect()) as connection:
            rows = connection.execute(query, values).fetchall()
        return [self._record(row) for row in rows]
