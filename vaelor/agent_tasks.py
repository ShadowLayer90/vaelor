"""Persistent task ledger for temporary delegations and durable agent work."""

from __future__ import annotations

import json
import os
import sqlite3
import time
import uuid
from contextlib import contextmanager
from typing import Any, Dict, Optional

from .agent_task_app_capabilities import (  # noqa: F401
    _safe_app_grant_snapshot,
    plan_custom_app_calls,  # re-exported; deployment_agent imports it from here
)
from .runtime_paths import env_value, state_path


TASK_STATES = {
    "triage", "ready", "running", "needs_approval", "blocked",
    "completed", "failed", "cancelled", "archived",
}
TASK_KINDS = {"temporary", "durable"}
PROFILES = {
    "docker": {
        "name": "Docker and app troubleshooter",
        "description": "Finds appliance app, Compose, port, storage, health, and recovery problems.",
        "scopes": ["workloads:read", "jobs:read"],
    },
    "models": {
        "name": "Local AI troubleshooter",
        "description": "Finds appliance model compatibility, memory, storage, runtime, and health problems.",
        "scopes": ["workloads:read", "system:read"],
    },
    "system": {
        "name": "System health troubleshooter",
        "description": "Finds appliance telemetry, cooling, service, storage, and network problems.",
        "scopes": ["system:read", "cooling:read", "jobs:read"],
    },
    "security": {
        "name": "Appliance security check",
        "description": "Checks appliance exposure, approvals, credentials, and risky workload settings.",
        "scopes": ["system:read", "workloads:read", "assistant:read"],
    },
    "kvm": {
        "name": "Remote access troubleshooter",
        "description": "Finds appliance KVM, VNC, connectivity, session, and access problems.",
        "scopes": ["system:read", "workloads:read"],
    },
    "research": {
        "name": "Setup evidence check",
        "description": "Checks bounded setup evidence for appliance compatibility and recovery risks.",
        "scopes": ["system:read", "workloads:read"],
    },
}


class AgentTaskError(ValueError):
    pass


class AgentTaskStore:
    def __init__(self, database_path: Optional[str] = None, profile_store=None, app_grant_context=None,
                 workload_act_grants=None):
        self.profile_store = profile_store
        self.app_grant_context = app_grant_context
        # The server-owned workload acting grant (VD-100 #96). Default None means
        # no agent is ever pinned to a workload operation, so `resolve_agent_
        # operation` resolves the empty set and nothing is proposable - the
        # fail-closed default the acting seam requires.
        self.workload_act_grants = workload_act_grants
        self.database_path = database_path or env_value(
            "VAELOR_AGENT_TASK_DB", "PM_AGENT_TASK_DB",
            state_path("assistant/tasks.sqlite3"),
        )
        parent = os.path.dirname(self.database_path)
        if parent:
            os.makedirs(parent, mode=0o700, exist_ok=True)
        self._initialize()
        try:
            os.chmod(self.database_path, 0o600)
        except OSError:
            pass

    def _workload_act_snapshot(self, profile, version, definition):
        """The pinned workload-acting grant for this profile, or ``[]``.

        Only custom agents can hold one, and only if an administrator pinned it.
        A store that raises must not block task creation; it fails closed to an
        empty snapshot, which means the agent may propose no workload operation -
        never the other way around (#34).
        """
        if self.workload_act_grants is None or not definition.get("custom"):
            return []
        try:
            return self.workload_act_grants.operation_grant_snapshot(profile, int(version))
        except Exception:
            return []

    def _connect(self):
        connection = sqlite3.connect(self.database_path, timeout=15)
        connection.row_factory = sqlite3.Row
        connection.execute("PRAGMA foreign_keys=ON")  # pairs-with: sqlite-foreign-keys-tight
        return connection

    @contextmanager
    def _connection(self):
        connection = self._connect()
        try:
            with connection:
                yield connection
        finally:
            connection.close()

    def _initialize(self):
        with self._connection() as connection:
            connection.execute("PRAGMA journal_mode=WAL")  # pairs-with: sqlite-journal-mode-tight
            connection.executescript(
                """
                CREATE TABLE IF NOT EXISTS agent_tasks (
                    id TEXT PRIMARY KEY,
                    actor TEXT NOT NULL,
                    assigned_to TEXT,
                    title TEXT NOT NULL,
                    description TEXT NOT NULL,
                    kind TEXT NOT NULL,
                    profile TEXT NOT NULL,
                    profile_version INTEGER NOT NULL DEFAULT 0,
                    approval_context_json TEXT NOT NULL DEFAULT '{}',
                    execution_binding_json TEXT NOT NULL DEFAULT '{}',
                    model_tier TEXT NOT NULL DEFAULT '',
                    state TEXT NOT NULL,
                    parent_id TEXT,
                    depth INTEGER NOT NULL DEFAULT 0,
                    approval_required INTEGER NOT NULL DEFAULT 0,
                    idempotency_key TEXT,
                    result_json TEXT NOT NULL DEFAULT '{}',
                    error TEXT NOT NULL DEFAULT '',
                    heartbeat_at REAL,
                    created_at REAL NOT NULL,
                    updated_at REAL NOT NULL,
                    retry_of TEXT,
                    FOREIGN KEY(parent_id) REFERENCES agent_tasks(id)
                );
                CREATE UNIQUE INDEX IF NOT EXISTS idx_agent_tasks_idempotency
                ON agent_tasks(actor, idempotency_key)
                WHERE idempotency_key IS NOT NULL;
                CREATE INDEX IF NOT EXISTS idx_agent_tasks_actor_updated
                ON agent_tasks(actor, updated_at DESC);
                CREATE TABLE IF NOT EXISTS agent_task_events (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    task_id TEXT NOT NULL,
                    actor TEXT NOT NULL,
                    event TEXT NOT NULL,
                    details_json TEXT NOT NULL DEFAULT '{}',
                    created_at REAL NOT NULL,
                    FOREIGN KEY(task_id) REFERENCES agent_tasks(id) ON DELETE CASCADE
                );
                CREATE TABLE IF NOT EXISTS agent_task_dependencies (
                    task_id TEXT NOT NULL,
                    depends_on TEXT NOT NULL,
                    PRIMARY KEY(task_id, depends_on),
                    FOREIGN KEY(task_id) REFERENCES agent_tasks(id) ON DELETE CASCADE,
                    FOREIGN KEY(depends_on) REFERENCES agent_tasks(id) ON DELETE CASCADE
                );
                CREATE TABLE IF NOT EXISTS agent_task_comments (
                    id TEXT PRIMARY KEY,
                    task_id TEXT NOT NULL,
                    actor TEXT NOT NULL,
                    content TEXT NOT NULL,
                    created_at REAL NOT NULL,
                    FOREIGN KEY(task_id) REFERENCES agent_tasks(id) ON DELETE CASCADE
                );
                CREATE TABLE IF NOT EXISTS agent_task_artifacts (
                    id TEXT PRIMARY KEY,
                    task_id TEXT NOT NULL,
                    name TEXT NOT NULL,
                    media_type TEXT NOT NULL,
                    content_json TEXT NOT NULL,
                    created_at REAL NOT NULL,
                    FOREIGN KEY(task_id) REFERENCES agent_tasks(id) ON DELETE CASCADE
                );
                CREATE TABLE IF NOT EXISTS agent_task_approvals (
                    id TEXT PRIMARY KEY,
                    task_id TEXT NOT NULL,
                    actor TEXT NOT NULL,
                    decision TEXT NOT NULL,
                    note TEXT NOT NULL,
                    created_at REAL NOT NULL,
                    FOREIGN KEY(task_id) REFERENCES agent_tasks(id) ON DELETE CASCADE
                );
                """
            )
            columns = {
                row["name"] for row in connection.execute("PRAGMA table_info(agent_tasks)")
            }
            if "assigned_to" not in columns:
                connection.execute("ALTER TABLE agent_tasks ADD COLUMN assigned_to TEXT")
            if "retry_of" not in columns:
                connection.execute("ALTER TABLE agent_tasks ADD COLUMN retry_of TEXT")
            if "profile_version" not in columns:
                connection.execute(
                    "ALTER TABLE agent_tasks ADD COLUMN profile_version INTEGER NOT NULL DEFAULT 0"
                )
            if "approval_context_json" not in columns:
                connection.execute(
                    "ALTER TABLE agent_tasks ADD COLUMN approval_context_json TEXT NOT NULL DEFAULT '{}'"
                )
            if "execution_binding_json" not in columns:
                connection.execute(
                    "ALTER TABLE agent_tasks ADD COLUMN execution_binding_json TEXT NOT NULL DEFAULT '{}'"
                )
            # The model tier this run uses: '' is the default NPU deployment-agent
            # model; 'capable' is the user-triggered escalation onto the GPU
            # ai-chat model. Older rows predate escalation and are NPU runs.
            if "model_tier" not in columns:
                connection.execute(
                    "ALTER TABLE agent_tasks ADD COLUMN model_tier TEXT NOT NULL DEFAULT ''"
                )
            # Older rows recorded retry lineage only in the append-only event
            # stream. Backfill the queryable projection before retry
            # deduplication is relied on by the UI or worker.
            legacy_retries = connection.execute(
                "SELECT task_id, details_json FROM agent_task_events WHERE event='retried'"
            ).fetchall()
            for event in legacy_retries:
                try:
                    retry_of = json.loads(event["details_json"] or "{}").get("retry_of")
                except (TypeError, ValueError):
                    retry_of = None
                if retry_of:
                    connection.execute(
                        "UPDATE agent_tasks SET retry_of=? WHERE id=? AND COALESCE(retry_of, '')=''",
                        (str(retry_of), event["task_id"]),
                    )
            connection.execute(
                "CREATE UNIQUE INDEX IF NOT EXISTS idx_agent_tasks_active_retry "
                "ON agent_tasks(retry_of) WHERE retry_of IS NOT NULL "
                "AND state IN ('triage', 'ready', 'running', 'needs_approval')"
            )

    def profiles(self, actor: Optional[str] = None):
        builtins = [
            {"id": key, **value, "custom": False, "version": 0, "enabled": True}
            for key, value in PROFILES.items()
        ]
        return builtins + (
            self.profile_store.list(actor) if self.profile_store and actor else []
        )

    def profile(self, profile_id: str, actor: str):
        if profile_id in PROFILES:
            return {
                "id": profile_id, **PROFILES[profile_id],
                "custom": False, "version": 0, "enabled": True,
            }
        return self.profile_store.get(profile_id, actor) if self.profile_store else None

    @staticmethod
    def _row(row):
        if row is None:
            return None
        item = dict(row)
        item["approval_required"] = bool(item["approval_required"])
        item["result"] = json.loads(item.pop("result_json") or "{}")
        item["approval_context"] = json.loads(
            item.pop("approval_context_json", "{}") or "{}"
        )
        item["execution_binding"] = json.loads(
            item.pop("execution_binding_json", "{}") or "{}"
        )
        item["model_tier"] = str(item.get("model_tier") or "")
        return item

    @staticmethod
    def _validated_execution_binding(binding: Optional[Dict[str, Any]]) -> Dict[str, Any]:
        """Keep model selection metadata durable without storing credentials."""
        if binding is None:
            return {}
        if not isinstance(binding, dict):
            raise AgentTaskError("The execution model binding is invalid.")
        mode = str(binding.get("mode", "")).strip().lower()
        if mode not in {"local", "provider"}:
            raise AgentTaskError("The execution model binding must name local or provider mode.")
        return {
            "mode": mode,
            "provider": str(binding.get("provider", ""))[:120],
            "model": str(binding.get("model", ""))[:160],
            "bound_at": float(binding.get("bound_at", time.time())),
        }

    @staticmethod
    def _attach_revisions(connection, records):
        """Bind task projections to append-only task-event sequence IDs."""
        if not records:
            return
        ids = [str(item["id"]) for item in records]
        placeholders = ",".join("?" for _ in ids)
        rows = connection.execute(
            f"SELECT task_id, MAX(id) AS revision FROM agent_task_events WHERE task_id IN ({placeholders}) GROUP BY task_id",
            ids,
        ).fetchall()
        revisions = {str(row["task_id"]): int(row["revision"] or 1) for row in rows}
        for item in records:
            item["revision"] = revisions.get(str(item["id"]), 1)

    def create(
        self, actor: str, title: str, description: str, *,
        kind: str = "durable", profile: str = "system",
        parent_id: Optional[str] = None, depth: int = 0,
        approval_required: bool = False,
        idempotency_key: Optional[str] = None,
        dependencies: Optional[list[str]] = None,
        profile_version: Optional[int] = None,
        _approval_context: Optional[Dict[str, Any]] = None,
        execution_binding: Optional[Dict[str, Any]] = None,
        model_tier: str = "",
        retry_of: Optional[str] = None,
    ) -> Dict[str, Any]:
        title = str(title).strip()[:120]
        description = str(description).strip()[:4000]
        if not title or not description:
            raise AgentTaskError("A task title and description are required.")
        definition = self.profile(profile, actor)
        if kind not in TASK_KINDS or not definition or not definition.get("enabled"):
            raise AgentTaskError("Unsupported task kind or agent profile.")
        selected_definition = definition
        selected_version = int(definition.get("version", 0))
        if profile_version is not None:
            requested_version = int(profile_version)
            if not definition.get("custom") or self.profile_store is None or not hasattr(
                self.profile_store, "get_version"
            ):
                raise AgentTaskError("Only custom agents support pinned definition versions.")
            selected_definition = self.profile_store.get_version(
                profile, actor, requested_version
            )
            if selected_definition is None:
                raise AgentTaskError("The pinned custom-agent definition is unavailable.")
            selected_version = requested_version
        app_grants, app_grants_status = ([], "not_applicable")
        if selected_definition.get("custom"):
            app_grants, app_grants_status = _safe_app_grant_snapshot(
                self.app_grant_context, actor, profile, selected_version, selected_definition
            )
        approval_context = {
            "profile_name": str(selected_definition.get("name", profile))[:100],
            "profile_version": selected_version,
            "capabilities": sorted({
                str(item)[:100]
                for item in (
                    list(selected_definition.get("scopes", []))
                    + list(selected_definition.get("permissions", []))
                )
                if str(item).strip()
            }),
            "integrations": [
                str(connector.get("name", ""))[:100]
                for connector in selected_definition.get("connectors", [])
                if str(connector.get("name", "")).strip()
            ],
            "app_grants": app_grants,
            "app_grants_status": app_grants_status,
            # Pinned once, from the server-owned store, at task-creation time.
            # #34: the acting envelope an installed agent may propose within is
            # NEVER derived from its own result; it is this snapshot, taken here
            # and read back unchanged by `resolve_agent_operation` at approval.
            # Its own field, so it never rides the app-capability path exposed to
            # the model. Empty unless an administrator pinned the profile.
            "workload_act_grants": self._workload_act_snapshot(
                profile, selected_version, selected_definition
            ),
        }
        if _approval_context is not None:
            approval_context = json.loads(json.dumps(_approval_context, separators=(",", ":"), default=str))
        binding = self._validated_execution_binding(execution_binding)
        # Only the capable escalation tier is a non-default value; anything else
        # is the default NPU model. Validated here so an unknown tier can never
        # be persisted and read back as an escalation.
        tier = "capable" if str(model_tier).strip().lower() == "capable" else ""
        if depth not in (0, 1):
            raise AgentTaskError("Delegation depth is limited to one.")
        if kind == "temporary" and depth != 1:
            raise AgentTaskError("Temporary delegated tasks must be one-level delegations.")
        key = str(idempotency_key).strip()[:120] if idempotency_key else None
        now = time.time()
        task_id = uuid.uuid4().hex
        state = "needs_approval" if approval_required else "ready"
        with self._connection() as connection:
            if key:
                existing = connection.execute(
                    "SELECT * FROM agent_tasks WHERE actor=? AND idempotency_key=?",
                    (actor, key),
                ).fetchone()
                if existing:
                    return self._row(existing)
            connection.execute(
                """
                INSERT INTO agent_tasks
                (id, actor, assigned_to, title, description, kind, profile,
                 profile_version, approval_context_json, execution_binding_json, model_tier, state, parent_id, depth,
                 approval_required, idempotency_key, created_at, updated_at, retry_of)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (task_id, actor, actor, title, description, kind, profile,
                 selected_version, json.dumps(approval_context), json.dumps(binding), tier, state, parent_id,
                 depth, int(approval_required), key, now, now, retry_of),
            )
            for dependency_id in dependencies or []:
                dependency = connection.execute(
                    "SELECT actor FROM agent_tasks WHERE id=?", (dependency_id,)
                ).fetchone()
                if dependency is None or dependency["actor"] != actor:
                    raise AgentTaskError("A task dependency was not found.")
                connection.execute(
                    "INSERT INTO agent_task_dependencies(task_id,depends_on) VALUES(?,?)",
                    (task_id, dependency_id),
                )
            connection.execute(
                "INSERT INTO agent_task_events(task_id,actor,event,details_json,created_at) VALUES(?,?,?,?,?)",
                (task_id, actor, "created", "{}", now),
            )
        return self.get(task_id, actor)

    def get(self, task_id: str, actor: Optional[str] = None):
        query = "SELECT * FROM agent_tasks WHERE id=?"
        values = [task_id]
        if actor is not None:
            query += " AND (actor=? OR assigned_to=?)"
            values.extend((actor, actor))
        with self._connection() as connection:
            row = connection.execute(query, values).fetchone()
            if row is None:
                return None
            item = self._row(row)
            self._attach_revisions(connection, [item])
        return item

    def list(self, actor: Optional[str] = None, limit: int = 100):
        limit = max(1, min(int(limit), 200))
        query = "SELECT * FROM agent_tasks"
        values = []
        if actor is not None:
            query += " WHERE actor=? OR assigned_to=?"
            values.extend((actor, actor))
        query += " ORDER BY updated_at DESC LIMIT ?"
        values.append(limit)
        with self._connection() as connection:
            records = [self._row(row) for row in connection.execute(query, values)]
            self._attach_revisions(connection, records)
        return records

    def archive_finished(self, actor: str) -> int:
        """Hide finished work from the recent ledger without erasing its audit trail."""
        now = time.time()
        with self._connection() as connection:
            rows = connection.execute(
                """
                SELECT id FROM agent_tasks
                WHERE actor = ? AND state IN ('completed', 'failed', 'cancelled')
                """,
                (actor,),
            ).fetchall()
            for row in rows:
                connection.execute(
                    "UPDATE agent_tasks SET state = 'archived', updated_at = ? WHERE id = ?",
                    (now, row["id"]),
                )
                connection.execute(
                    """
                    INSERT INTO agent_task_events
                    (task_id, actor, event, details_json, created_at)
                    VALUES (?, ?, 'archived', '{}', ?)
                    """,
                    (row["id"], actor, now),
                )
        return len(rows)

    def transition(
        self, task_id: str, actor: str, state: str, *,
        result: Optional[Dict[str, Any]] = None, error: str = "",
        expected_state: Optional[str] = None,
    ):
        if state not in TASK_STATES:
            raise AgentTaskError("Unsupported task state.")
        current = self.get(task_id, actor)
        if current is None:
            raise AgentTaskError("Task not found.")
        terminal = {"completed", "failed", "cancelled", "archived"}
        if current["state"] in terminal and state != "archived":
            raise AgentTaskError("This task is already finished.")
        now = time.time()
        bounded_result = json.dumps(
            current["result"] if result is None else result,
            separators=(",", ":"), default=str,
        )
        if len(bounded_result.encode("utf-8")) > 64 * 1024:
            raise AgentTaskError("Task result exceeds the safe output limit.")
        with self._connection() as connection:
            cursor = connection.execute(
                """
                UPDATE agent_tasks SET state=?, result_json=?, error=?,
                heartbeat_at=?, updated_at=? WHERE id=? AND actor=?
                AND (actor=? OR assigned_to=?) AND state=?
                """,
                (state, bounded_result, str(error)[:2000],
                 now if state == "running" else current.get("heartbeat_at"),
                 now, task_id, current["actor"], current["actor"], actor,
                 expected_state or current["state"]),
            )
            if cursor.rowcount != 1:
                raise AgentTaskError("The task state changed before this action completed. Refresh and try again.")
            connection.execute(
                "INSERT INTO agent_task_events(task_id,actor,event,details_json,created_at) VALUES(?,?,?,?,?)",
                (task_id, actor, state, "{}", now),
            )
        return self.get(task_id, actor)

    def rebind_execution(self, task_id: str, actor: str, binding: Dict[str, Any]):
        """Replace an unavailable model binding while preserving task identity."""
        current = self.get(task_id, actor)
        if current is None:
            raise AgentTaskError("Task not found.")
        if current["state"] in {"completed", "archived"}:
            raise AgentTaskError("Completed task history cannot be rebound.")
        next_binding = self._validated_execution_binding(binding)
        now = time.time()
        next_state = "needs_approval" if current["state"] in {"blocked", "failed"} else current["state"]
        with self._connection() as connection:
            connection.execute(
                """
                UPDATE agent_tasks SET execution_binding_json=?, state=?, error='',
                    heartbeat_at=NULL, updated_at=? WHERE id=? AND actor=?
                    AND (actor=? OR assigned_to=?)
                """,
                (json.dumps(next_binding, separators=(",", ":")), next_state, now,
                 task_id, current["actor"], current["actor"], actor),
            )
            connection.execute(
                """
                INSERT INTO agent_task_events(task_id,actor,event,details_json,created_at)
                VALUES(?,?,?,?,?)
                """,
                (task_id, actor, "execution_binding_replaced", json.dumps({
                    "mode": next_binding["mode"],
                    "provider": next_binding["provider"],
                    "model": next_binding["model"],
                    "state": next_state,
                }), now),
            )
        return self.get(task_id, actor)

    def cancel(self, task_id: str, actor: str, *, allow_all: bool = False):
        """Cancel one task through the task ledger, preserving its event history."""
        current = self.get(task_id, None if allow_all else actor)
        if current is None:
            raise AgentTaskError("Task not found.")
        if current["state"] in {"completed", "failed", "cancelled", "archived"}:
            raise AgentTaskError("This task is already finished.")
        now = time.time()
        with self._connection() as connection:
            cursor = connection.execute(
                """
                UPDATE agent_tasks SET state='cancelled', error=?,
                    heartbeat_at=NULL, updated_at=? WHERE id=? AND actor=? AND state NOT IN
                    ('completed','failed','cancelled','archived')
                """,
                ("Cancelled by operator", now, task_id, current["actor"]),
            )
            if cursor.rowcount != 1:
                raise AgentTaskError(
                    "The task state changed before cancellation completed. Refresh and try again."
                )
            connection.execute(
                """
                INSERT INTO agent_task_events
                    (task_id,actor,event,details_json,created_at)
                VALUES(?,?,?,?,?)
                """,
                (task_id, actor, "cancelled", json.dumps({"operation_action": "cancel"}), now),
            )
        return self.get(task_id, None if allow_all else actor)

    def update_completed_result(
        self, task_id: str, actor: str, result: Dict[str, Any], event: str
    ):
        current = self.get(task_id, actor)
        if current is None or current["state"] != "completed":
            raise AgentTaskError("Only a completed task result can be updated.")
        encoded = json.dumps(result, separators=(",", ":"), default=str)
        if len(encoded.encode("utf-8")) > 64 * 1024:
            raise AgentTaskError("Task result exceeds the safe output limit.")
        now = time.time()
        with self._connection() as connection:
            connection.execute(
                "UPDATE agent_tasks SET result_json=?,updated_at=? WHERE id=? AND actor=?",
                (encoded, now, task_id, current["actor"]),
            )
            connection.execute(
                """
                INSERT INTO agent_task_events
                (task_id,actor,event,details_json,created_at) VALUES(?,?,?,?,?)
                """,
                (task_id, actor, str(event)[:80], "{}", now),
            )
        return self.get(task_id, actor)

    def events(self, task_id: str, actor: str):
        if self.get(task_id, actor) is None:
            raise AgentTaskError("Task not found.")
        with self._connection() as connection:
            return [
                {**dict(row), "details": json.loads(row["details_json"] or "{}")}
                for row in connection.execute(
                    "SELECT * FROM agent_task_events WHERE task_id=? ORDER BY id",
                    (task_id,),
                )
            ]

    def details(self, task_id: str, actor: str):
        task = self.get(task_id, actor)
        if task is None:
            raise AgentTaskError("Task not found.")
        with self._connection() as connection:
            task["dependencies"] = [
                row["depends_on"] for row in connection.execute(
                    "SELECT depends_on FROM agent_task_dependencies WHERE task_id=?",
                    (task_id,),
                )
            ]
            task["comments"] = [
                dict(row) for row in connection.execute(
                    "SELECT * FROM agent_task_comments WHERE task_id=? ORDER BY created_at",
                    (task_id,),
                )
            ]
            task["artifacts"] = [
                {**dict(row), "content": json.loads(row["content_json"])}
                for row in connection.execute(
                    "SELECT * FROM agent_task_artifacts WHERE task_id=? ORDER BY created_at",
                    (task_id,),
                )
            ]
            task["approvals"] = [
                dict(row) for row in connection.execute(
                    "SELECT * FROM agent_task_approvals WHERE task_id=? ORDER BY created_at",
                    (task_id,),
                )
            ]
            self._attach_revisions(connection, [task])
        task["events"] = self.events(task_id, actor)
        return task

    def add_comment(self, task_id: str, actor: str, content: str):
        if self.get(task_id, actor) is None:
            raise AgentTaskError("Task not found.")
        content = str(content).strip()
        if not content or len(content) > 4000:
            raise AgentTaskError("Comments must be between 1 and 4,000 characters.")
        item = {"id": uuid.uuid4().hex, "task_id": task_id, "actor": actor,
                "content": content, "created_at": time.time()}
        with self._connection() as connection:
            connection.execute(
                "INSERT INTO agent_task_comments(id,task_id,actor,content,created_at) VALUES(:id,:task_id,:actor,:content,:created_at)",
                item,
            )
        return item

    def add_artifact(self, task_id: str, actor: str, name: str, media_type: str, content: Any):
        if self.get(task_id, actor) is None:
            raise AgentTaskError("Task not found.")
        name = str(name).strip()[:120]
        media_type = str(media_type).strip()[:80] or "application/json"
        encoded = json.dumps(content, separators=(",", ":"), default=str)
        if not name or len(encoded.encode("utf-8")) > 64 * 1024:
            raise AgentTaskError("Artifact name is required and content is limited to 64 KiB.")
        item = {"id": uuid.uuid4().hex, "task_id": task_id, "name": name,
                "media_type": media_type, "content": json.loads(encoded),
                "created_at": time.time()}
        with self._connection() as connection:
            connection.execute(
                "INSERT INTO agent_task_artifacts(id,task_id,name,media_type,content_json,created_at) VALUES(?,?,?,?,?,?)",
                (item["id"], task_id, name, media_type, encoded, item["created_at"]),
            )
        return item

    def approve(self, task_id: str, actor: str, decision: str, note: str = ""):
        current = self.get(task_id, actor)
        if current is None:
            raise AgentTaskError("Task not found.")
        if current["state"] != "needs_approval":
            raise AgentTaskError("This task is not waiting for approval.")
        if decision not in {"approved", "rejected"}:
            raise AgentTaskError("Choose approved or rejected.")
        task = self.transition(
            task_id, actor, "ready" if decision == "approved" else "cancelled",
            error="" if decision == "approved" else "Rejected by operator",
            expected_state="needs_approval",
        )
        now = time.time()
        with self._connection() as connection:
            connection.execute(
                "INSERT INTO agent_task_approvals(id,task_id,actor,decision,note,created_at) VALUES(?,?,?,?,?,?)",
                (uuid.uuid4().hex, task_id, actor, decision, str(note)[:1000], now),
            )
        return task

    def heartbeat(self, task_id: str):
        now = time.time()
        with self._connection() as connection:
            cursor = connection.execute(
                "UPDATE agent_tasks SET heartbeat_at=?,updated_at=? WHERE id=? AND state='running'",
                (now, now, task_id),
            )
        return bool(cursor.rowcount)

    def handoff(self, task_id: str, actor: str, assigned_to: str):
        current = self.get(task_id, actor)
        if current is None:
            raise AgentTaskError("Task not found.")
        if current["state"] in {"running", "completed", "archived"}:
            raise AgentTaskError("Running or finished tasks cannot be handed off.")
        assigned_to = str(assigned_to).strip().lower()[:64]
        if not assigned_to:
            raise AgentTaskError("Choose an account for the handoff.")
        now = time.time()
        with self._connection() as connection:
            connection.execute(
                "UPDATE agent_tasks SET assigned_to=?,updated_at=? WHERE id=?",
                (assigned_to, now, task_id),
            )
            connection.execute(
                "INSERT INTO agent_task_events(task_id,actor,event,details_json,created_at) VALUES(?,?,?,?,?)",
                (task_id, actor, "handoff", json.dumps({"assigned_to": assigned_to}), now),
            )
        return self.get(task_id)

    def retry(
        self, task_id: str, actor: str, *, allow_all: bool = False,
        model_tier: Optional[str] = None,
    ):
        current = self.get(task_id, None if allow_all else actor)
        if current is None:
            raise AgentTaskError("Task not found.")
        if current["state"] not in {"failed", "cancelled", "blocked"}:
            raise AgentTaskError("Only failed, cancelled, or blocked tasks can be retried.")
        # A user-triggered escalation re-runs the SAME task on the capable GPU
        # model. It carries no NPU execution binding (the copied one names the
        # NPU model that just underperformed), so run_once drives the whole run
        # on the capable connection instead. A plain retry keeps the original
        # tier and binding unchanged.
        escalate = str(model_tier or "").strip().lower() == "capable"
        # A duplicate click must return the existing child instead of creating
        # a second active execution. The query is backed by a partial unique
        # index and the insert below carries retry_of with task creation, so
        # concurrent clicks cannot both win.
        active_states = {"triage", "ready", "running", "needs_approval"}
        with self._connection() as connection:
            existing = connection.execute(
                "SELECT * FROM agent_tasks WHERE retry_of=? AND state IN (?,?,?,?) "
                "ORDER BY created_at DESC LIMIT 1",
                (task_id, *active_states),
            ).fetchone()
            if existing is not None:
                candidate = self._row(existing)
                if allow_all or candidate.get("actor") == actor or candidate.get("assigned_to") == actor:
                    candidate["deduplicated"] = True
                    return candidate
        try:
            retried = self.create(
                current["actor"],
                current["title"],
                current["description"],
                kind=current["kind"],
                profile=current["profile"],
                approval_required=current["approval_required"],
                idempotency_key="retry:{}:{}".format(task_id, uuid.uuid4().hex),
                profile_version=(
                    current["profile_version"] if current["profile_version"] else None
                ),
                _approval_context=current.get("approval_context") or {},
                execution_binding=(
                    None if escalate else (current.get("execution_binding") or None)
                ),
                model_tier="capable" if escalate else (current.get("model_tier") or ""),
                retry_of=task_id,
            )
        except sqlite3.IntegrityError:
            # The unique partial index is the serialization point for two
            # simultaneous retry requests. Return the winner as stable data.
            with self._connection() as connection:
                existing = connection.execute(
                    "SELECT * FROM agent_tasks WHERE retry_of=? AND state IN (?,?,?,?) "
                    "ORDER BY created_at DESC LIMIT 1",
                    (task_id, *active_states),
                ).fetchone()
                if existing is None:
                    raise
                candidate = self._row(existing)
                candidate["deduplicated"] = True
                return candidate
        now = time.time()
        with self._connection() as connection:
            connection.execute(
                "UPDATE agent_tasks SET assigned_to=?,updated_at=? WHERE id=?",
                (current.get("assigned_to") or current["actor"], now, retried["id"]),
            )
            connection.execute(
                "INSERT INTO agent_task_events(task_id,actor,event,details_json,created_at) VALUES(?,?,?,?,?)",
                (retried["id"], actor, "retried", json.dumps({"retry_of": task_id}), now),
            )
        result = self.get(retried["id"], None if allow_all else actor)
        result["deduplicated"] = False
        return result

    def recover_stale(self, stale_seconds: int = 120):
        cutoff = time.time() - max(30, stale_seconds)
        with self._connection() as connection:
            rows = connection.execute(
                "SELECT id,actor,approval_required FROM agent_tasks WHERE kind='durable' AND state='running' AND COALESCE(heartbeat_at,updated_at)<?",
                (cutoff,),
            ).fetchall()
            for row in rows:
                state = "needs_approval" if row["approval_required"] else "ready"
                connection.execute(
                    "UPDATE agent_tasks SET state=?,error='Recovered after interrupted worker',updated_at=? WHERE id=?",
                    (state, time.time(), row["id"]),
                )
        return len(rows)

    def claim_next(self):
        claimed = None
        with self._connection() as connection:
            connection.execute("BEGIN IMMEDIATE")  # pairs-with: sqlite-begin-immediate
            candidates = connection.execute(
                "SELECT * FROM agent_tasks WHERE kind='durable' AND state='ready' ORDER BY created_at"
            ).fetchall()
            for row in candidates:
                dependencies = connection.execute(
                    """
                    SELECT t.state FROM agent_task_dependencies d
                    JOIN agent_tasks t ON t.id=d.depends_on WHERE d.task_id=?
                    """, (row["id"],),
                ).fetchall()
                states = {item["state"] for item in dependencies}
                if states & {"failed", "cancelled", "blocked"}:
                    connection.execute(
                        "UPDATE agent_tasks SET state='blocked',error='A dependency did not complete',updated_at=? WHERE id=?",
                        (time.time(), row["id"]),
                    )
                    continue
                if dependencies and any(item["state"] != "completed" for item in dependencies):
                    continue
                now = time.time()
                connection.execute(
                    "UPDATE agent_tasks SET state='running',heartbeat_at=?,updated_at=? WHERE id=? AND state='ready'",
                    (now, now, row["id"]),
                )
                claimed = (row["id"], row["actor"])
                break
        return self.get(*claimed) if claimed else None
