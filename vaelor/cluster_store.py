"""Persistent, secret-free inventory for controller-managed cluster nodes."""

from __future__ import annotations

import json
import logging
import os
import sqlite3
import threading
import time
import uuid
from contextlib import closing
from pathlib import Path
from typing import Any, Dict, Optional

from .runtime_paths import env_value, state_path

LOGGER = logging.getLogger(__name__)

#: The one wording for "no enrolled node has that id", imported by every
#: cluster module that refuses on it rather than spelled in each.
NODE_NOT_FOUND = "Cluster node was not found."


#: The cluster store's file under the state root: the store's default and the
#: installer's read-only GPU check (`deployment_refresh.cluster_database`)
#: both take it from here, so the two cannot name different files (review A4).
CLUSTER_DATABASE = "cluster/fleet.sqlite3"


#: Every pooled deployment row, oldest first: the store's own listing and the
#: installer's read-only one (`deployment_refresh.gpu_refresh_plan`) alike.
POOLED_ROWS_QUERY = "SELECT * FROM pooled_deployments ORDER BY created_at"


def _decode_profile(raw: Optional[str]) -> Dict[str, Any]:
    """One stored software record (VD-194 round 3 R3-6c).

    A row that is not a JSON object is reported as ``{"unreadable": why}`` for
    that worker alone - one corrupt row never takes every worker's reading down
    with it, and is never decoded as something it is not (LESSONS 8).
    """
    try:
        value = json.loads(raw or "{}")
    except ValueError as error:
        return {"unreadable": "it is not valid JSON ({})".format(str(error)[:80])}
    if not isinstance(value, dict):
        return {"unreadable": "it is not a record"}
    return value


class ClusterStore:
    """SQLite fleet inventory. SSH material remains in the credential broker."""

    def __init__(self, database_path: Optional[str] = None):
        self.database_path = database_path or env_value(
            "VAELOR_CLUSTER_DB", "PM_CLUSTER_DB",
            state_path(CLUSTER_DATABASE),
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
                    ingest_key_hash TEXT NOT NULL DEFAULT '',
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
                CREATE TABLE IF NOT EXISTS model_cache (
                    node_id TEXT NOT NULL,
                    repo TEXT NOT NULL,
                    revision TEXT NOT NULL DEFAULT '',
                    state TEXT NOT NULL,
                    bytes_on_disk INTEGER NOT NULL DEFAULT 0,
                    message TEXT NOT NULL DEFAULT '',
                    unit TEXT NOT NULL DEFAULT '',
                    created_at INTEGER NOT NULL,
                    updated_at INTEGER NOT NULL,
                    PRIMARY KEY(node_id, repo, revision)
                );
                """
            )
            # A store created before E2a has no `ingest_key_hash` column; add it
            # to an existing table the same way `agent_api` migrates its own
            # token store, so an upgraded controller keeps its enrolled nodes.
            columns = {
                row[1]
                for row in connection.execute("PRAGMA table_info(cluster_nodes)")
            }
            if "ingest_key_hash" not in columns:
                connection.execute(
                    "ALTER TABLE cluster_nodes "
                    "ADD COLUMN ingest_key_hash TEXT NOT NULL DEFAULT ''"
                )
            # VD-194 P1: the last worker-software reading and attempt, added the
            # same way so an upgraded controller keeps its enrolled workers.
            if "profile_json" not in columns:
                connection.execute(
                    "ALTER TABLE cluster_nodes "
                    "ADD COLUMN profile_json TEXT NOT NULL DEFAULT '{}'"
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

    @staticmethod
    def _refuse_controller_key(key: str) -> None:
        """The controller record is not a named setting; keep it out of reach."""
        if key == "controller":
            raise ValueError("The controller record has its own accessors.")

    def state_value(self, key: str) -> Any:
        """One named cluster-wide setting, or ``None`` when it was never set.

        The ``cluster_state`` table beside the controller record: small
        settings both the control plane and the workload executor read (the
        cluster link, VD-162). The reserved ``controller`` key keeps its own
        accessors and is refused here.
        """
        self._refuse_controller_key(key)
        with closing(self._connect()) as connection:
            row = connection.execute(
                "SELECT value_json FROM cluster_state WHERE key=?", (str(key),)
            ).fetchone()
        return json.loads(row["value_json"]) if row else None

    def put_state_value(self, key: str, value: Any) -> None:
        """Store one named cluster-wide setting (see `state_value`)."""
        self._refuse_controller_key(key)
        with closing(self._connect()) as connection:
            connection.execute(
                """
                INSERT INTO cluster_state(key,value_json,updated_at)
                VALUES(?,?,?)
                ON CONFLICT(key) DO UPDATE SET
                    value_json=excluded.value_json,updated_at=excluded.updated_at
                """,
                (str(key), json.dumps(value, separators=(",", ":")), int(time.time())),
            )
            connection.commit()

    def delete_state_value(self, key: str) -> None:
        """Forget one named cluster-wide setting; absent is not an error."""
        self._refuse_controller_key(key)
        with closing(self._connect()) as connection:
            connection.execute("DELETE FROM cluster_state WHERE key=?", (str(key),))
            connection.commit()

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

    def _node_rows(self, limit: int) -> list:
        with closing(self._connect()) as connection:
            return connection.execute(
                "SELECT * FROM cluster_nodes ORDER BY created_at LIMIT ?",
                (max(1, min(int(limit), 200)),),
            ).fetchall()

    def _node_row(self, node_id: str):
        with closing(self._connect()) as connection:
            return connection.execute(
                "SELECT * FROM cluster_nodes WHERE id=?", (str(node_id),)
            ).fetchone()

    def list_nodes(self, limit: int = 100) -> list[Dict[str, Any]]:
        return [self._node(row) for row in self._node_rows(limit)]

    def list_nodes_tolerant(self, limit: int = 100) -> list[Dict[str, Any]]:
        """Every node, each row decoded on its own (PH-R1).

        A row whose JSON will not decode is logged with its traceback and
        listed with what does decode and ``record_unreadable: True``, never
        allowed to fail the whole Fleet view (LESSONS 8).
        """
        nodes = []
        for row in self._node_rows(limit):
            try:
                nodes.append(self._node(row))
            except ValueError:
                LOGGER.exception("node %s: its row in the fleet store will not decode", row["id"])
                nodes.append(self._core(row))
        return nodes

    @staticmethod
    def _core(row: sqlite3.Row) -> Dict[str, Any]:
        """A node row with each JSON field kept only where it decodes."""
        core = {key: row[key] for key in ("id", "name", "host", "port", "role", "state",
                                          "host_key_fingerprint", "last_seen", "created_at", "updated_at")}
        for key in ("inventory", "labels"):
            try:
                value = json.loads(row[key + "_json"])
            except ValueError:
                value = None
            core[key] = value if isinstance(value, dict) else {}
        core["record_unreadable"] = True
        return core

    def node_core(self, node_id: str) -> Optional[Dict[str, Any]]:
        """A node whose record will not decode, as far as it does, with its
        credential id (PH-R2: the removal that is its exit). ``None`` when gone."""
        row = self._node_row(node_id)
        if row is None:
            return None
        return {**self._core(row), "credential_id": row["credential_id"]}

    def node_credential_ids(self) -> set:
        """Every SSH credential id an enrolled node signs in with (ids only).

        What Settings > Connections reads to tell a cluster sign-in still in
        use from one no node names any more (`credential_listing`).
        """
        with closing(self._connect()) as connection:
            rows = connection.execute("SELECT credential_id FROM cluster_nodes").fetchall()
        return {str(row["credential_id"]) for row in rows if row["credential_id"]}

    def get_node(self, node_id: str, *, include_credential: bool = False):
        row = self._node_row(node_id)
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
                raise ValueError(NODE_NOT_FOUND)
            return node
        fields.append("updated_at=?")
        values.extend([int(time.time()), str(node_id)])
        with closing(self._connect()) as connection:
            cursor = connection.execute(
                f"UPDATE cluster_nodes SET {','.join(fields)} WHERE id=?", values
            )
            if cursor.rowcount != 1:
                raise ValueError(NODE_NOT_FOUND)
            connection.commit()
        return self.get_node(node_id)

    def set_ingest_key_hash(self, node_id: str, key_hash: str) -> bool:
        """Store the sha256 hash of a node's telemetry ingest key.

        Only the hash is kept — never the plaintext key — so the store can
        authenticate a presented key without being able to reproduce it. The
        hash lives on the node row, so `delete_node` drops it with the node and
        no separate cleanup is needed.
        """
        with closing(self._connect()) as connection:
            cursor = connection.execute(
                "UPDATE cluster_nodes SET ingest_key_hash=?, updated_at=? WHERE id=?",
                (str(key_hash), int(time.time()), str(node_id)),
            )
            connection.commit()
        return cursor.rowcount == 1

    def node_profile(self, node_id: str) -> Dict[str, Any]:
        """A worker's last software reading and attempt (VD-194), ``{}`` when never read.

        Kept off the node record (`_node`) on purpose: the raw reading is for
        the projection `worker_profile_state` serves, never for every payload
        that carries a node.
        """
        with closing(self._connect()) as connection:
            row = connection.execute(
                "SELECT profile_json FROM cluster_nodes WHERE id=?", (str(node_id),)
            ).fetchone()
        return _decode_profile(row["profile_json"]) if row else {}

    def node_profiles(self) -> Dict[str, Dict[str, Any]]:
        """Every enrolled node's software record, by node id; each row decoded on its own."""
        with closing(self._connect()) as connection:
            rows = connection.execute("SELECT id, profile_json FROM cluster_nodes").fetchall()
        return {row["id"]: _decode_profile(row["profile_json"]) for row in rows}

    def set_node_profile(self, node_id: str, record: Dict[str, Any]) -> bool:
        """Store a worker's software record; False when the node is gone."""
        with closing(self._connect()) as connection:
            cursor = connection.execute(
                "UPDATE cluster_nodes SET profile_json=? WHERE id=?",
                (json.dumps(record, separators=(",", ":")), str(node_id)),
            )
            connection.commit()
        return cursor.rowcount == 1

    def ingest_key_hashes(self) -> list[tuple[str, str]]:
        """`(node_id, ingest_key_hash)` for every node that has a hash set.

        Returned as a list so the manager can compare a presented key against
        each in constant time. Nodes with no hash are omitted, so a node that
        was never issued a key can never match.
        """
        with closing(self._connect()) as connection:
            rows = connection.execute(
                "SELECT id, ingest_key_hash FROM cluster_nodes "
                "WHERE ingest_key_hash != ''"
            ).fetchall()
        return [(row["id"], row["ingest_key_hash"]) for row in rows]

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
            rows = connection.execute(POOLED_ROWS_QUERY).fetchall()
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

    #: The states a cached-weights row may hold. Owned here, its one home, so a
    #: pull that has started (``pulling``), finished (``ready``), or failed
    #: (``error``) is described the same way wherever it is read. Held as one
    #: constant rather than re-spelled at each call so the model library and this
    #: store cannot drift on the word for the same state.
    MODEL_CACHE_STATES = ("pulling", "ready", "error")

    #: The single-row read shared by the put (read-back) and get paths, named
    #: once so the two cannot drift on the key columns.
    _MODEL_CACHE_ROW_QUERY = (
        "SELECT * FROM model_cache WHERE node_id=? AND repo=? AND revision=?"
    )

    @staticmethod
    def _model_cache(row: sqlite3.Row) -> Dict[str, Any]:
        return {
            "node_id": row["node_id"],
            "repo": row["repo"],
            "revision": row["revision"],
            "state": row["state"],
            "bytes_on_disk": row["bytes_on_disk"],
            "message": row["message"],
            "unit": row["unit"],
            "created_at": row["created_at"],
            "updated_at": row["updated_at"],
        }

    def put_model_cache(
        self, *, node_id: str, repo: str, revision: str = "", state: str,
        bytes_on_disk: int = 0, message: str = "", unit: str = "",
    ) -> Dict[str, Any]:
        """Upsert one (node, repo, revision) cached-weights row in the house style.

        ``created_at`` is set on insert and left untouched by the update branch,
        so re-recording a pull's progress keeps the moment it began.
        """
        if state not in self.MODEL_CACHE_STATES:
            raise ValueError("A cached-model state must be pulling, ready, or error.")
        now = int(time.time())
        with closing(self._connect()) as connection:
            connection.execute(
                """
                INSERT INTO model_cache(
                    node_id,repo,revision,state,bytes_on_disk,message,unit,
                    created_at,updated_at
                ) VALUES(?,?,?,?,?,?,?,?,?)
                ON CONFLICT(node_id,repo,revision) DO UPDATE SET
                    state=excluded.state,
                    bytes_on_disk=excluded.bytes_on_disk,
                    message=excluded.message,
                    unit=excluded.unit,
                    updated_at=excluded.updated_at
                """,
                (
                    str(node_id), str(repo), str(revision or ""), state,
                    int(bytes_on_disk or 0), str(message or ""), str(unit or ""),
                    now, now,
                ),
            )
            connection.commit()
            row = connection.execute(
                self._MODEL_CACHE_ROW_QUERY,
                (str(node_id), str(repo), str(revision or "")),
            ).fetchone()
        return self._model_cache(row)

    def list_model_cache(self) -> list[Dict[str, Any]]:
        with closing(self._connect()) as connection:
            rows = connection.execute(
                "SELECT * FROM model_cache ORDER BY created_at, node_id, repo"
            ).fetchall()
        return [self._model_cache(row) for row in rows]

    def get_model_cache(self, node_id: str, repo: str, revision: str = ""):
        with closing(self._connect()) as connection:
            row = connection.execute(
                self._MODEL_CACHE_ROW_QUERY,
                (str(node_id), str(repo), str(revision or "")),
            ).fetchone()
        return self._model_cache(row) if row else None

    def delete_model_cache(self, node_id: str, repo: str, revision: str = "") -> bool:
        with closing(self._connect()) as connection:
            cursor = connection.execute(
                "DELETE FROM model_cache WHERE node_id=? AND repo=? AND revision=?",
                (str(node_id), str(repo), str(revision or "")),
            )
            connection.commit()
        return cursor.rowcount == 1

    def delete_model_cache_repo(self, node_id: str, repo: str) -> int:
        """Drop every cached-weights row for one repo on one node, any revision.

        The Hugging Face hub lays a repo out as one ``models--org--name``
        directory holding *all* its revisions, so removing that directory on a
        node removes every revision's weights at once. Deleting only the one
        ``(node,repo,revision)`` row would leave a sibling revision's row reading
        ``ready`` while its weights are gone - an untruthful inventory. Removal is
        therefore repo-granular: this drops all revisions' rows together. Returns
        how many rows were removed.
        """
        with closing(self._connect()) as connection:
            cursor = connection.execute(
                "DELETE FROM model_cache WHERE node_id=? AND repo=?",
                (str(node_id), str(repo)),
            )
            connection.commit()
        return cursor.rowcount
