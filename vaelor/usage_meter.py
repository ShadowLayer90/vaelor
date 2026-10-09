"""Durable cumulative per-key usage of the inference gateway (Phase G, VD-128).

The F-phase API keys answer *who* reached the managed inference gateway; this is
the accounting layer on top: how many requests and prompt/completion tokens each
key consumed, kept cumulatively so the number survives a restart and a
telemetry-retention lapse. It mirrors :mod:`vaelor.inference_metrics` for its
connection, chmod and self-heal discipline, but where that store's detail is a
bounded rolling log, this one is an UPSERT counter: nothing is ever pruned, and
a count only ever moves up.

**What it no longer answers.** How much each DEPLOYMENT served is the model's
own count (:mod:`vaelor.model_usage`, ACC-044/045) - this meter saw only the
gateway and AI Chat, keyed deployments by an internal lease id, and split a
deployment's totals on every redeploy. Its per-deployment reading and its daily
roll-up, which nothing ever read (ACC-047), were removed; windowed questions are
answered from the ten-minute roll-ups (:mod:`vaelor.usage_rollup`).

Only the control-plane gateway (the ``vaelor`` process, Path A) writes it, so
``0o600`` is the correct mode - it is not shared with the executor account. The
writes happen on the request path and the UI reads concurrently, so the store
runs in WAL. Every write is best-effort: a failure here degrades to "not
counted" and never surfaces into the inference response.

``usage_total`` is keyed ``(key_id, deployment)``, where ``deployment`` is the
lease credential id (server-authoritative), never a client-supplied model
string, so a client cannot mint unbounded rows. A store written by an older
release may still hold a ``usage_daily`` table; nothing reads or writes it.
"""

from __future__ import annotations

import os
import sqlite3
import time
from contextlib import closing
from pathlib import Path
from typing import List, Dict, Any, Optional

from .runtime_paths import env_value, state_path


class UsageMeter:
    def __init__(self, database_path: Optional[str] = None):
        self.database_path = database_path or env_value(
            "VAELOR_USAGE_METER_DB", "PM_USAGE_METER_DB",
            state_path("cluster/usage-meter.sqlite3"),
        )
        Path(self.database_path).parent.mkdir(parents=True, exist_ok=True, mode=0o700)
        with closing(sqlite3.connect(self.database_path)) as connection:
            connection.execute(
                """CREATE TABLE IF NOT EXISTS usage_total (
                    key_id TEXT NOT NULL, key_label TEXT NOT NULL DEFAULT '',
                    deployment TEXT NOT NULL,
                    request_count INTEGER NOT NULL DEFAULT 0,
                    prompt_tokens INTEGER NOT NULL DEFAULT 0,
                    completion_tokens INTEGER NOT NULL DEFAULT 0,
                    first_used_at REAL NOT NULL, last_used_at REAL NOT NULL,
                    PRIMARY KEY (key_id, deployment)
                )"""
            )
            connection.execute("PRAGMA journal_mode=WAL")  # pairs-with: sqlite-journal-mode-tight
            connection.commit()
        self._heal_permissions()

    def _heal_permissions(self):
        """Re-assert ``0o600`` on the store file, swallowing a platform refusal.

        Done on construction and after every write open, so a file left behind
        with a looser mode is tightened the next time the gateway touches it,
        mirroring how :class:`~vaelor.agent_api.AgentApiTokenStore` self-heals.
        """
        try:
            os.chmod(self.database_path, 0o600)
        except OSError:
            pass

    def record(
        self, *, key_id: str, deployment: str, label: str = "",
        prompt_tokens: int = 0, completion_tokens: int = 0,
    ):
        """Fold one completed request into the cumulative counters.

        Best-effort by contract: any store failure is swallowed so a metering
        problem can never fail the inference that is already being served. Counts
        are clamped non-negative and only ever incremented, so the lifetime
        figures are monotonic. One call is exactly one request; tokens are added
        as measured (0 when no usage was parsed).
        """
        key_id = str(key_id or "")
        if not key_id:
            return
        deployment = str(deployment or "")
        prompt_tokens = max(0, int(prompt_tokens or 0))
        completion_tokens = max(0, int(completion_tokens or 0))
        now = time.time()
        try:
            with closing(sqlite3.connect(self.database_path)) as connection:
                connection.execute(
                    """INSERT INTO usage_total(
                        key_id, key_label, deployment, request_count,
                        prompt_tokens, completion_tokens, first_used_at, last_used_at
                    ) VALUES(?,?,?,1,?,?,?,?)
                    ON CONFLICT(key_id, deployment) DO UPDATE SET
                        key_label=excluded.key_label,
                        request_count=usage_total.request_count + 1,
                        prompt_tokens=usage_total.prompt_tokens + excluded.prompt_tokens,
                        completion_tokens=usage_total.completion_tokens
                            + excluded.completion_tokens,
                        last_used_at=excluded.last_used_at""",
                    (
                        key_id, str(label or ""), deployment,
                        prompt_tokens, completion_tokens, now, now,
                    ),
                )
                connection.commit()
            self._heal_permissions()
        except sqlite3.Error:
            return

    def per_key_totals(self) -> List[Dict[str, Any]]:
        """Lifetime counters summed across deployments, one row per key id.

        The label is a denormalized snapshot for a key the token store no longer
        lists; the read API prefers the live label and uses this only as a
        fallback. Ordered by most recently used so the busy keys lead.
        """
        return self._grouped(
            """SELECT key_id, MAX(key_label) AS key_label,
                SUM(request_count) AS request_count,
                SUM(prompt_tokens) AS prompt_tokens,
                SUM(completion_tokens) AS completion_tokens,
                MAX(last_used_at) AS last_used_at
            FROM usage_total GROUP BY key_id ORDER BY last_used_at DESC"""
        )

    def _grouped(self, query: str, parameters=()) -> List[Dict[str, Any]]:
        """Run a read query and return plain dict rows, empty on any store error."""
        try:
            with closing(sqlite3.connect(self.database_path)) as connection:
                connection.row_factory = sqlite3.Row
                rows = connection.execute(query, parameters).fetchall()
            return [dict(row) for row in rows]
        except sqlite3.Error:
            return []
