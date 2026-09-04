"""The operator-facing grant that turns the acting surface on (VD-100 #96).

``workloads:act`` is default-inert: no role and no agent carries it, so the
acting seam (`assistant_acting_wiring`, the installed-agent operation route)
proposes nothing until an administrator grants it here. This store is that grant,
and it is the ONLY thing that widens the surface - the model never does (#34).

Two kinds of grant, one store, both empty until an administrator writes them:

* **Operator scope.** An operator account may be granted ``workloads:act`` so
  its own Assistant chat can carry an acting proposal. ``scopes_for(actor)`` is
  read on the answer path and threaded into the acting registry's scope gate.
* **Agent operations.** A custom-agent profile may be pinned to a subset of the
  phase-1 envelope (``workloads.restart`` / ``workloads.redeploy``).
  ``operation_grant_snapshot(profile, version)`` returns the server-owned grant
  the task-creation path pins into ``approval_context``, and which
  `assistant_action_proposals.resolve_agent_operation` intersects with the
  envelope at approval time. An operation outside the envelope is refused on
  write, so a typo or a widened row can never smuggle a new job type in.
"""

from __future__ import annotations

import os
import sqlite3
import time
from contextlib import closing
from pathlib import Path
from typing import Any, Dict, List, Optional

from .assistant_action_proposals import INSTALLED_AGENT_OPERATIONS, WORKLOADS_ACT_SCOPE
from .runtime_paths import env_value, state_path


class WorkloadActGrantError(ValueError):
    """A safe, user-presentable refusal to write a workload-act grant."""


class WorkloadActGrantStore:
    """Records which operators hold ``workloads:act`` and which agents are pinned.

    Empty by construction: a fresh appliance grants nothing, so the acting
    surface is inert until an administrator acts. Nothing here reads the model's
    output; the envelope is validated against the server-owned constant on every
    write.
    """

    def __init__(self, database_path: Optional[str] = None):
        self.database_path = database_path or env_value(
            "VAELOR_WORKLOAD_ACT_DB", "PM_WORKLOAD_ACT_DB",
            state_path("assistant/workload-act-grants.sqlite3"),
        )
        Path(self.database_path).parent.mkdir(parents=True, exist_ok=True, mode=0o700)
        with closing(self._connect()) as connection:
            connection.execute(
                """
                CREATE TABLE IF NOT EXISTS operator_act_scopes (
                    actor TEXT NOT NULL,
                    scope TEXT NOT NULL,
                    granted_by TEXT NOT NULL,
                    created_at REAL NOT NULL,
                    PRIMARY KEY (actor, scope)
                )
                """
            )
            connection.execute(
                """
                CREATE TABLE IF NOT EXISTS agent_operation_grants (
                    profile TEXT NOT NULL,
                    profile_version INTEGER NOT NULL,
                    operation TEXT NOT NULL,
                    granted_by TEXT NOT NULL,
                    created_at REAL NOT NULL,
                    PRIMARY KEY (profile, profile_version, operation)
                )
                """
            )
            connection.commit()
        try:
            os.chmod(self.database_path, 0o600)
        except OSError:
            pass

    def _connect(self) -> sqlite3.Connection:
        connection = sqlite3.connect(self.database_path)
        connection.row_factory = sqlite3.Row
        return connection

    # -- Operator scope grants -------------------------------------------------

    def scopes_for(self, actor: str) -> frozenset[str]:
        """The act-scopes an operator holds now. Empty by default (fail-closed)."""
        actor = str(actor or "").strip()
        if not actor:
            return frozenset()
        with closing(self._connect()) as connection:
            rows = connection.execute(
                "SELECT scope FROM operator_act_scopes WHERE actor=?", (actor,)
            ).fetchall()
        return frozenset(row["scope"] for row in rows)

    def grant_operator_scope(self, actor: str, granted_by: str) -> Dict[str, Any]:
        actor = str(actor or "").strip()
        if not actor:
            raise WorkloadActGrantError("Name the operator to grant workloads:act.")
        now = time.time()
        with closing(self._connect()) as connection:
            connection.execute(
                """
                INSERT OR REPLACE INTO operator_act_scopes
                    (actor, scope, granted_by, created_at)
                VALUES (?, ?, ?, ?)
                """,
                (actor, WORKLOADS_ACT_SCOPE, str(granted_by or "")[:100], now),
            )
            connection.commit()
        return {"actor": actor, "scope": WORKLOADS_ACT_SCOPE}

    def revoke_operator_scope(self, actor: str) -> bool:
        actor = str(actor or "").strip()
        with closing(self._connect()) as connection:
            cursor = connection.execute(
                "DELETE FROM operator_act_scopes WHERE actor=? AND scope=?",
                (actor, WORKLOADS_ACT_SCOPE),
            )
            connection.commit()
        return cursor.rowcount > 0

    # -- Agent operation grants ------------------------------------------------

    def _pinned_operations(self, profile: str, version: int) -> List[str]:
        with closing(self._connect()) as connection:
            rows = connection.execute(
                """
                SELECT operation FROM agent_operation_grants
                WHERE profile=? AND profile_version=?
                ORDER BY operation
                """,
                (str(profile), int(version)),
            ).fetchall()
        # Intersect with the server-owned envelope on read as well as write: a
        # constant that shrank must never leave a stale row proposable (#34).
        return [row["operation"] for row in rows if row["operation"] in INSTALLED_AGENT_OPERATIONS]

    def operation_grant_snapshot(self, profile: str, version: int) -> List[Dict[str, Any]]:
        """The pinned workload-operation grant, in the shape the resolver reads.

        Returns ``[]`` when nothing is pinned - `authorized_operations` then
        resolves to the empty set and the agent may propose nothing, which is the
        correct fail-closed default. When operations are pinned it returns one
        grant dict of ``{"operations": [{"operation_id": op}, ...]}``, the exact
        shape `assistant_action_proposals.agent_grant_snapshot` consumes.
        """
        operations = self._pinned_operations(profile, version)
        if not operations:
            return []
        return [{
            "grant_type": "workloads.act",
            "operations": [{"operation_id": op} for op in operations],
        }]

    def grant_agent_operation(
        self, profile: str, version: int, operation: str, granted_by: str
    ) -> Dict[str, Any]:
        operation = str(operation or "").strip()
        if operation not in INSTALLED_AGENT_OPERATIONS:
            raise WorkloadActGrantError(
                "#34 / VD-100 #96: an agent may be pinned only to the server-owned "
                "envelope {}. '{}' is outside it and is refused here, so a widened "
                "row can never introduce a job type the executor was not meant to "
                "run.".format(sorted(INSTALLED_AGENT_OPERATIONS), operation)
            )
        now = time.time()
        with closing(self._connect()) as connection:
            connection.execute(
                """
                INSERT OR REPLACE INTO agent_operation_grants
                    (profile, profile_version, operation, granted_by, created_at)
                VALUES (?, ?, ?, ?, ?)
                """,
                (str(profile), int(version), operation, str(granted_by or "")[:100], now),
            )
            connection.commit()
        return {"profile": str(profile), "profile_version": int(version), "operation": operation}

    def revoke_agent_operation(self, profile: str, version: int, operation: str) -> bool:
        with closing(self._connect()) as connection:
            cursor = connection.execute(
                """
                DELETE FROM agent_operation_grants
                WHERE profile=? AND profile_version=? AND operation=?
                """,
                (str(profile), int(version), str(operation or "").strip()),
            )
            connection.commit()
        return cursor.rowcount > 0
