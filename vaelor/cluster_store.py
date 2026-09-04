"""Persistent, secret-free inventory for controller-managed cluster nodes."""

from __future__ import annotations

import json
import os
import sqlite3
import threading
import time
import uuid
from contextlib import closing
from pathlib import Path
from typing import Any, Dict, Optional

from .runtime_paths import env_value, state_path


class ClusterStore:
    """SQLite fleet inventory. SSH material remains in the credential broker."""

    def __init__(self, database_path: Optional[str] = None):
        self.database_path = database_path or env_value(
            "VAELOR_CLUSTER_DB", "PM_CLUSTER_DB",
            state_path("cluster/fleet.sqlite3"),
        )
        self._schema_ready = False
        self._schema_lock = threading.Lock()

    def _connect(self) -> sqlite3.Connection:
        path = Path(self.database_path)
        path.parent.mkdir(parents=True, exist_ok=True)
        connection = sqlite3.connect(str(path), timeout=10)
        connection.row_factory = sqlite3.Row
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
            connection.executescript(
                """
                CREATE TABLE IF NOT EXISTS cluster_state (
                    key TEXT PRIMARY KEY,
                    value_json TEXT NOT NULL,
                    updated_at INTEGER NOT NULL
                );
                CREATE TABLE IF NOT EXISTS cluster_nodes (
                    id TEXT PRIMARY KEY,
                    name TEXT NOT NULL,
                    host TEXT NOT NULL,
                    port INTEGER NOT NULL,
                    credential_id TEXT NOT NULL,
                    host_key_fingerprint TEXT NOT NULL,
                    role TEXT NOT NULL,
                    state TEXT NOT NULL,
                    inventory_json TEXT NOT NULL DEFAULT '{}',
                    labels_json TEXT NOT NULL DEFAULT '{}',
                    last_seen INTEGER,
                    created_at INTEGER NOT NULL,
                    updated_at INTEGER NOT NULL,
                    UNIQUE(host, port)
                );
                CREATE TABLE IF NOT EXISTS pooled_deployments (
                    name TEXT PRIMARY KEY,
                    state TEXT NOT NULL,
                    model_id TEXT NOT NULL,
                    node_ids_json TEXT NOT NULL,
                    units_json TEXT NOT NULL DEFAULT '{}',
                    endpoint TEXT NOT NULL DEFAULT '',
                    credential_id TEXT NOT NULL DEFAULT '',
                    created_at INTEGER NOT NULL,
                    updated_at INTEGER NOT NULL
                );
                """
            )
            connection.commit()
            self._schema_ready = True

    @staticmethod
    def _node(row: sqlite3.Row) -> Dict[str, Any]:
        return {
            "id": row["id"],
            "name": row["name"],
            "host": row["host"],
            "port": row["port"],
            "role": row["role"],
            "state": row["state"],
            "host_key_fingerprint": row["host_key_fingerprint"],
            "inventory": json.loads(row["inventory_json"]),
            "labels": json.loads(row["labels_json"]),
            "last_seen": row["last_seen"],
            "created_at": row["created_at"],
            "updated_at": row["updated_at"],
        }

    def controller(self) -> Dict[str, Any]:
        with closing(self._connect()) as connection:
            row = connection.execute(
                "SELECT value_json FROM cluster_state WHERE key='controller'"
            ).fetchone()
        return json.loads(row["value_json"]) if row else {
            "initialized": False,
            "driver": "docker-swarm",
            "role": "head-controller",
        }

    def set_controller(self, value: Dict[str, Any]) -> Dict[str, Any]:
        normalized = {
            "initialized": bool(value.get("initialized")),
            "driver": "docker-swarm",
            "role": "head-controller",
            "cluster_id": str(value.get("cluster_id", ""))[:128],
            "advertise_address": str(value.get("advertise_address", ""))[:253],
        }
        now = int(time.time())
        with closing(self._connect()) as connection:
            connection.execute(
                """
                INSERT INTO cluster_state(key,value_json,updated_at)
                VALUES('controller',?,?)
                ON CONFLICT(key) DO UPDATE SET
                    value_json=excluded.value_json,updated_at=excluded.updated_at
                """,
                (json.dumps(normalized, separators=(",", ":")), now),
            )
            connection.commit()
        return normalized

    def add_node(
        self,
        *,
        name: str,
        host: str,
        port: int,
        credential_id: str,
        fingerprint: str,
        inventory: Dict[str, Any],
    ) -> Dict[str, Any]:
        node_id = "node_{}".format(uuid.uuid4().hex[:20])
        now = int(time.time())
        with closing(self._connect()) as connection:
            try:
                connection.execute(
                    """
                    INSERT INTO cluster_nodes(
                        id,name,host,port,credential_id,host_key_fingerprint,
                        role,state,inventory_json,labels_json,last_seen,created_at,updated_at
                    ) VALUES(?,?,?,?,?,?,'worker','enrolled',?,'{}',?,?,?)
                    """,
                    (
                        node_id, str(name).strip()[:80], host, int(port),
                        credential_id, fingerprint,
                        json.dumps(inventory, separators=(",", ":")),
                        now, now, now,
                    ),
                )
            except sqlite3.IntegrityError as error:
                raise ValueError("That SSH host is already enrolled.") from error
            connection.commit()
            row = connection.execute(
                "SELECT * FROM cluster_nodes WHERE id=?", (node_id,)
            ).fetchone()
        return self._node(row)

    def list_nodes(self, limit: int = 100) -> list[Dict[str, Any]]:
        with closing(self._connect()) as connection:
            rows = connection.execute(
                "SELECT * FROM cluster_nodes ORDER BY created_at LIMIT ?",
                (max(1, min(int(limit), 200)),),
            ).fetchall()
        return [self._node(row) for row in rows]

    def get_node(self, node_id: str, *, include_credential: bool = False):
        with closing(self._connect()) as connection:
            row = connection.execute(
                "SELECT * FROM cluster_nodes WHERE id=?", (str(node_id),)
            ).fetchone()
        if row is None:
            return None
        node = self._node(row)
        if include_credential:
            node["credential_id"] = row["credential_id"]
        return node

    def update_node(self, node_id: str, **changes) -> Dict[str, Any]:
        allowed = {"name", "role", "state", "inventory", "labels", "last_seen"}
        fields, values = [], []
        for key, value in changes.items():
            if key not in allowed:
                continue
            column = f"{key}_json" if key in {"inventory", "labels"} else key
            fields.append(f"{column}=?")
            values.append(
                json.dumps(value, separators=(",", ":"))
                if key in {"inventory", "labels"} else value
            )
        if not fields:
            node = self.get_node(node_id)
            if node is None:
                raise ValueError("Cluster node was not found.")
            return node
        fields.append("updated_at=?")
        values.extend([int(time.time()), str(node_id)])
        with closing(self._connect()) as connection:
            cursor = connection.execute(
                f"UPDATE cluster_nodes SET {','.join(fields)} WHERE id=?", values
            )
            if cursor.rowcount != 1:
                raise ValueError("Cluster node was not found.")
            connection.commit()
        return self.get_node(node_id)

    def delete_node(self, node_id: str) -> Optional[str]:
        with closing(self._connect()) as connection:
            row = connection.execute(
                "SELECT credential_id FROM cluster_nodes WHERE id=?", (str(node_id),)
            ).fetchone()
            if row is None:
                return None
            connection.execute("DELETE FROM cluster_nodes WHERE id=?", (str(node_id),))
            connection.commit()
        return str(row["credential_id"])

    #: How many removals the fleet view remembers. Enough to explain a node
    #: that is no longer listed; this is not the audit log, which lives in the
    #: security store and is never trimmed by this cap.
    EVICTION_HISTORY = 20

    def list_evictions(self) -> list[Dict[str, Any]]:
        """Nodes this controller drained and removed, newest first (VD-033).

        A removed node leaves no row in `cluster_nodes`, so without this the
        fleet view would simply stop showing it. Silent removal is its own kind
        of lie, and this is what the view reports instead.
        """
        with closing(self._connect()) as connection:
            row = connection.execute(
                "SELECT value_json FROM cluster_state "
                "WHERE key='architecture_evictions'"
            ).fetchone()
        if row is None:
            return []
        records = json.loads(row["value_json"])
        return records if isinstance(records, list) else []

    def record_evictions(
        self, records: list[Dict[str, Any]]
    ) -> list[Dict[str, Any]]:
        """Append removal records and return the retained history."""
        history = ([dict(item) for item in records] + self.list_evictions())
        retained = history[: self.EVICTION_HISTORY]
        now = int(time.time())
        with closing(self._connect()) as connection:
            connection.execute(
                """
                INSERT INTO cluster_state(key,value_json,updated_at)
                VALUES('architecture_evictions',?,?)
                ON CONFLICT(key) DO UPDATE SET
                    value_json=excluded.value_json,updated_at=excluded.updated_at
                """,
                (json.dumps(retained, separators=(",", ":")), now),
            )
            connection.commit()
        return retained

    @staticmethod
    def _pooled(row: sqlite3.Row) -> Dict[str, Any]:
        return {
            "name": row["name"],
            "state": row["state"],
            "model_id": row["model_id"],
            "node_ids": json.loads(row["node_ids_json"]),
            "units": json.loads(row["units_json"]),
            "endpoint": row["endpoint"],
            "credential_id": row["credential_id"],
            "created_at": row["created_at"],
            "updated_at": row["updated_at"],
        }

    def put_pooled_deployment(
        self, *, name: str, state: str, model_id: str,
        node_ids: list[str], units: Optional[Dict[str, Any]] = None,
        endpoint: str = "", credential_id: str = "",
    ) -> Dict[str, Any]:
        now = int(time.time())
        with closing(self._connect()) as connection:
            connection.execute(
                """
                INSERT INTO pooled_deployments(
                    name,state,model_id,node_ids_json,units_json,endpoint,
                    credential_id,created_at,updated_at
                ) VALUES(?,?,?,?,?,?,?,?,?)
                ON CONFLICT(name) DO UPDATE SET
                    state=excluded.state,
                    model_id=excluded.model_id,
                    node_ids_json=excluded.node_ids_json,
                    units_json=excluded.units_json,
                    endpoint=excluded.endpoint,
                    credential_id=excluded.credential_id,
                    updated_at=excluded.updated_at
                """,
                (
                    name, state, model_id,
                    json.dumps(node_ids, separators=(",", ":")),
                    json.dumps(units or {}, separators=(",", ":")),
                    endpoint, credential_id, now, now,
                ),
            )
            connection.commit()
            row = connection.execute(
                "SELECT * FROM pooled_deployments WHERE name=?", (name,)
            ).fetchone()
        return self._pooled(row)

    def list_pooled_deployments(self) -> list[Dict[str, Any]]:
        with closing(self._connect()) as connection:
            rows = connection.execute(
                "SELECT * FROM pooled_deployments ORDER BY created_at"
            ).fetchall()
        return [self._pooled(row) for row in rows]

    def get_pooled_deployment(self, name: str):
        with closing(self._connect()) as connection:
            row = connection.execute(
                "SELECT * FROM pooled_deployments WHERE name=?", (str(name),)
            ).fetchone()
        return self._pooled(row) if row else None

    def delete_pooled_deployment(self, name: str) -> bool:
        with closing(self._connect()) as connection:
            cursor = connection.execute(
                "DELETE FROM pooled_deployments WHERE name=?", (str(name),)
            )
            connection.commit()
        return cursor.rowcount == 1
