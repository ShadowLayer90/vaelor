"""Persistent, bounded jobs for privileged appliance operations."""

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
from .runtime_paths import data_path, env_value

from .platform_drivers import default_platform_drivers
from .job_control import JobCancelled
from .application_research_workflow import canonical_phase, RESEARCH_BLOCKERS
from .job_failure_text import user_facing_failure
from .job_projection import job_projection
from .job_schema import apply_schema
from .job_secrets import contains_secret
from .job_vocabulary import (
    ACTIVE_JOB_STATES,
    ALLOWED_JOB_TYPES,
    JOB_STATES,
    TERMINAL_JOB_STATES,
)
from .operation_projection import ACTIVE_OPERATION_STATES


logger = logging.getLogger(__name__)


class JobStore:
    """SQLite job queue whose records never contain raw secret values."""

    def __init__(self, database_path: Optional[str] = None):
        self.database_path = database_path or env_value(
            "VAELOR_JOBS_DB", "PM_DASHBOARD_JOBS_DB",
            data_path("jobs/jobs.sqlite3"),
        )
        self._schema_ready = False
        self._schema_lock = threading.Lock()

    def _connect(self) -> sqlite3.Connection:
        path = Path(self.database_path)
        path.parent.mkdir(parents=True, exist_ok=True)
        try:
            descriptor = os.open(
                str(path),
                os.O_CREAT | os.O_EXCL | os.O_RDWR,
                0o660,
            )
        except FileExistsError:
            pass
        else:
            os.close(descriptor)
        try:
            os.chmod(path, 0o660)
        except PermissionError:
            # The executor may share a root-owned, group-writable queue with
            # the dashboard and does not need to change its mode on every poll.
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
            apply_schema(connection)
            self._schema_ready = True

    @staticmethod
    def _record(row: sqlite3.Row) -> Dict[str, Any]:
        record = {
            "id": row["id"],
            "type": row["type"],
            "state": row["state"],
            "actor": row["actor"],
            "payload": json.loads(row["payload_json"]),
            "progress": row["progress"],
            "message": row["message"],
            "phase": (
                canonical_phase(row["phase"])
                if row["type"] == "application.research"
                else row["phase"]
            ),
            "blocker_layer": row["blocker_layer"],
            "result": json.loads(row["result_json"]),
            "created_at": row["created_at"],
            "updated_at": row["updated_at"],
            "attempt": row["attempt"],
            "retry_of": row["retry_of"],
        }
        record.update(job_projection(record))
        return record

    @staticmethod
    def _attach_revisions(
        connection: sqlite3.Connection, records: list[Dict[str, Any]]
    ) -> None:
        """Bind projections to append-only event sequence revisions."""
        if not records:
            return
        ids = [str(record["id"]) for record in records]
        placeholders = ",".join("?" for _ in ids)
        rows = connection.execute(
            f"SELECT job_id, MAX(id) AS revision FROM job_events WHERE job_id IN ({placeholders}) GROUP BY job_id",
            ids,
        ).fetchall()
        revisions = {str(row["job_id"]): int(row["revision"] or 1) for row in rows}
        for record in records:
            record["revision"] = revisions.get(str(record["id"]), 1)

    def create(
        self,
        job_type: str,
        actor: str,
        payload: Dict[str, Any],
        *,
        deduplicate: bool = True,
    ) -> Dict[str, Any]:
        if job_type not in ALLOWED_JOB_TYPES:
            raise ValueError("Choose a supported workload job type.")
        if not isinstance(payload, dict):
            raise ValueError("Job payload must be an object.")
        if contains_secret(payload):
            raise ValueError("Store credentials separately and submit only a secret reference.")
        encoded = json.dumps(
            payload,
            separators=(",", ":"),
            default=str,
            sort_keys=True,
        )
        if len(encoded.encode("utf-8")) > 64 * 1024:
            raise ValueError("Job payload exceeds the 64 KiB limit.")
        job_id = str(uuid.uuid4())
        now = int(time.time() * 1000)
        phase = "interpreting" if job_type == "application.research" else ""
        with closing(self._connect()) as connection:
            connection.execute("BEGIN IMMEDIATE")  # pairs-with: sqlite-begin-immediate
            if deduplicate:
                active = connection.execute(
                    """
                    SELECT * FROM jobs
                    WHERE type=? AND actor=? AND payload_json=?
                      AND state IN
                        ('queued','validating','downloading','starting','cancelling')
                    ORDER BY created_at
                    LIMIT 1
                    """,
                    (job_type, actor, encoded),
                ).fetchone()
                if active is not None:
                    connection.commit()
                    record = self._record(active)
                    self._attach_revisions(connection, [record])
                    record["deduplicated"] = True
                    return record
            connection.execute(
                """
                INSERT INTO jobs
                    (id, type, state, actor, payload_json, progress, message,
                     phase, blocker_layer, result_json, created_at, updated_at)
                VALUES (?, ?, 'queued', ?, ?, 0, '', ?, '', '{}', ?, ?)
                """,
                (job_id, job_type, actor, encoded, phase, now, now),
            )
            connection.execute(
                """INSERT INTO job_events
                   (job_id,state,progress,message,phase,blocker_layer,created_at)
                   VALUES(?,?,?,?,?,'',?)""",
                (job_id, "queued", 0, "Queued for validation", phase, now),
            )
            connection.commit()
            row = connection.execute("SELECT * FROM jobs WHERE id = ?", (job_id,)).fetchone()
            record = self._record(row)
            self._attach_revisions(connection, [record])
        record["deduplicated"] = False
        return record

    def list(self, limit: int = 50, actor: Optional[str] = None) -> list[Dict[str, Any]]:
        safe_limit = max(1, min(int(limit), 200))
        with closing(self._connect()) as connection:
            if actor is None:
                rows = connection.execute(
                    "SELECT * FROM jobs ORDER BY created_at DESC LIMIT ?",
                    (safe_limit,),
                ).fetchall()
            else:
                rows = connection.execute(
                    "SELECT * FROM jobs WHERE actor=? ORDER BY created_at DESC LIMIT ?",
                    (actor, safe_limit),
                ).fetchall()
            records = [self._record(row) for row in rows]
            self._attach_revisions(connection, records)
        return records

    @staticmethod
    def _retry_lineage(records: list[Dict[str, Any]]) -> set[str]:
        """Attach complete retry ancestry and return uniquely resolved failures."""
        by_id = {record["id"]: record for record in records}
        resolved_failures: set[str] = set()
        for record in records:
            ancestry: list[str] = []
            ancestor = record.get("retry_of")
            visited: set[str] = set()
            while ancestor and ancestor not in visited:
                visited.add(ancestor)
                prior = by_id.get(ancestor)
                if prior is None:
                    break
                ancestry.append(prior["id"])
                if (
                    record["operation_state"] in {"completed", "healthy"}
                    and prior["operation_state"] in {"failed", "cancelled"}
                ):
                    resolved_failures.add(prior["id"])
                ancestor = prior.get("retry_of")
            record["retry_ancestry"] = ancestry
            record["retry_depth"] = len(ancestry)
            record["retry_root_id"] = ancestry[-1] if ancestry else record["id"]
        for record in records:
            record["resolved_by_retry"] = record["id"] in resolved_failures
        return resolved_failures

    def ledger(self, limit: int = 50, actor: Optional[str] = None) -> Dict[str, Any]:
        """Read rows and counters from one consistent job-ledger snapshot."""
        safe_limit = max(1, min(int(limit), 200))
        with closing(self._connect()) as connection:
            if actor is None:
                rows = connection.execute(
                    "SELECT * FROM jobs ORDER BY created_at DESC"
                ).fetchall()
            else:
                rows = connection.execute(
                    "SELECT * FROM jobs WHERE actor=? ORDER BY created_at DESC",
                    (actor,),
                ).fetchall()
            records = [self._record(row) for row in rows]
            self._attach_revisions(connection, records)
        resolved_failures = self._retry_lineage(records)
        # #201. This used to be a hand-written five, against the console's six
        # in `frontend/src/lib/operationOwner.ts`. Both were right for the
        # ledger each reads - `ready` and `draft` are unreachable for a job row
        # and reachable for an agent task - and *that* is why it would drift: a
        # fourteenth operation state would have been added to one side. The
        # count is unchanged by reading the wider set, because no job row can
        # hold the two states it adds; `tests/test_operation_activity.py` keeps
        # that measurement executable rather than in a comment.
        active = [
            record for record in records
            if record["operation_state"] in ACTIVE_OPERATION_STATES
        ]
        attention = [
            record for record in records
            if record["attention"] and not record["resolved_by_retry"]
        ]
        retryable = [
            record for record in records
            if record["retryable"] and not record["resolved_by_retry"]
        ]
        return {
            "jobs": records[:safe_limit],
            "counts": {
                "total": len(records),
                "active": len(active),
                "attention": len(attention),
                "retryable": len(retryable),
                "resolved_failures": len(resolved_failures),
            },
            "resolved_failure_ids": sorted(resolved_failures),
            "attention_ids": [record["id"] for record in attention],
            "retryable_ids": [record["id"] for record in retryable],
        }

    def get(self, job_id: str, actor: Optional[str] = None) -> Optional[Dict[str, Any]]:
        with closing(self._connect()) as connection:
            if actor is None:
                row = connection.execute(
                    "SELECT * FROM jobs WHERE id = ?", (job_id,)
                ).fetchone()
            else:
                row = connection.execute(
                    "SELECT * FROM jobs WHERE id = ? AND actor = ?", (job_id, actor)
                ).fetchone()
            if row is None:
                return None
            record = self._record(row)
            self._attach_revisions(connection, [record])
        return record

    def claim_next(self) -> Optional[Dict[str, Any]]:
        now = int(time.time() * 1000)
        with closing(self._connect()) as connection:
            connection.execute("BEGIN IMMEDIATE")  # pairs-with: sqlite-begin-immediate
            row = connection.execute(
                "SELECT id,type FROM jobs WHERE state = 'queued' ORDER BY created_at LIMIT 1"
            ).fetchone()
            if row is None:
                connection.rollback()
                return None
            phase = "interpreting" if row["type"] == "application.research" else ""
            connection.execute(
                """
                UPDATE jobs
                SET state = 'validating', progress = 5,
                    message = 'Validating request', phase = ?,
                    blocker_layer = '', updated_at = ?
                WHERE id = ? AND state = 'queued'
                """,
                (phase, now, row["id"]),
            )
            connection.execute(
                """INSERT INTO job_events
                   (job_id,state,progress,message,phase,blocker_layer,created_at)
                   VALUES(?,?,?,?,?,'',?)""",
                (row["id"], "validating", 5, "Validating request", phase, now),
            )
            connection.commit()
        return self.get(row["id"])

    def recover_interrupted(self) -> int:
        """Resolve work left by a previous executor without reviving it."""
        now = int(time.time() * 1000)
        placeholders = ",".join("?" for _ in ACTIVE_JOB_STATES)
        with closing(self._connect()) as connection:
            connection.execute("BEGIN IMMEDIATE")  # pairs-with: sqlite-begin-immediate
            rows = connection.execute(
                f"SELECT * FROM jobs WHERE state IN ({placeholders})",
                tuple(sorted(ACTIVE_JOB_STATES)),
            ).fetchall()
            for row in rows:
                research = row["type"] == "application.research"
                cancelling = row["state"] == "cancelling"
                state = "cancelled" if cancelling else "failed"
                message = (
                    "Cancellation completed while the workload service restarted."
                    if cancelling else
                    "Interrupted when the workload service restarted. "
                    "Review current appliance state before retrying."
                )
                phase = (
                    canonical_phase(row["phase"])
                    if research else row["phase"]
                )
                result = {
                    "code": "cancelled_on_restart"
                    if cancelling else "interrupted_worker",
                    "recovery": (
                        "No further work was started."
                        if cancelling else
                        "Inspect the workload or cluster state before starting a retry."
                    ),
                }
                blocker = "execution" if research and not cancelling else row["blocker_layer"]
                connection.execute(
                    """
                    UPDATE jobs
                    SET state=?,message=?,result_json=?,phase=?,
                        blocker_layer=?,updated_at=?
                    WHERE id=? AND state NOT IN
                        ('completed','failed','cancelled','healthy','rejected','superseded')
                    """,
                    (
                        state, message, json.dumps(result, separators=(",", ":")),
                        phase, blocker, now, row["id"],
                    ),
                )
                connection.execute(
                    """
                    INSERT INTO job_events
                        (job_id,state,progress,message,phase,blocker_layer,created_at)
                    VALUES (?,?,?,?,?,?,?)
                    """,
                    (
                        row["id"], state, 0, message, phase, blocker, now,
                    ),
                )
            connection.commit()
        return len(rows)
    def finish(
        self,
        job_id: str,
        *,
        state: str,
        message: str,
        result: Optional[Dict[str, Any]] = None,
        phase: Optional[str] = None,
        blocker_layer: Optional[str] = None,
    ) -> Dict[str, Any]:
        if state not in TERMINAL_JOB_STATES:
            raise ValueError("Choose a supported terminal job state.")
        now = int(time.time() * 1000)
        with closing(self._connect()) as connection:
            connection.execute("BEGIN IMMEDIATE")  # pairs-with: sqlite-begin-immediate
            row = connection.execute(
                "SELECT * FROM jobs WHERE id=?", (job_id,)
            ).fetchone()
            if row is None:
                connection.rollback()
                raise KeyError(job_id)
            current = self._record(row)
            if row["state"] in TERMINAL_JOB_STATES:
                connection.rollback()
                return current
            research = row["type"] == "application.research"
            current_phase = (
                canonical_phase(row["phase"]) if research else row["phase"]
            )
            if phase is not None:
                phase_value = canonical_phase(phase) if research else phase
            elif research and state in {"completed", "healthy"}:
                phase_value = "ready_for_review"
            elif research and state == "failed":
                phase_value = "needs_input"
            else:
                phase_value = current_phase
            blocker_value = (
                current["blocker_layer"] if blocker_layer is None else blocker_layer
            )
            if research and blocker_value not in RESEARCH_BLOCKERS and blocker_value != "":
                connection.rollback()
                raise ValueError(
                    "Choose a supported application-research blocker layer."
                )
            final_state = state
            final_message = str(message)[:1000]
            final_result = result or {}
            # Docker and systemd failure text reached users verbatim, including
            # `journalctl` invocations, on a product that promises no terminal
            # is required. Translating here covers every producer that finishes
            # a job. The original is not swallowed: it stays in the result for
            # a technical disclosure and is logged for operators.
            if final_state == "failed":
                translated = user_facing_failure(final_message)
                if translated is not None:
                    logger.warning(
                        "job %s (%s) failed: %s",
                        job_id, row["type"], translated["technical"],
                    )
                    final_result = {
                        **final_result,
                        "cause": translated["cause"],
                        "recovery": translated["recovery"],
                        "technical_detail": translated["technical"],
                    }
                    final_message = translated["summary"][:1000]
            if row["state"] == "cancelling" and state != "cancelled":
                final_state = "cancelled"
                final_message = "Cancelled safely"
                final_result = {
                    "recovery": "Review the workload state before retrying.",
                }
            if final_state == "cancelled" and not final_result:
                final_result = {
                    "recovery": "Review the workload state before retrying.",
                }
            progress = 100 if final_state in {"completed", "healthy"} else 0
            cursor = connection.execute(
                """
                UPDATE jobs
                SET state=?,progress=?,message=?,result_json=?,
                    phase=?,blocker_layer=?,updated_at=?
                WHERE id=? AND state NOT IN
                    ('completed','failed','cancelled','healthy','rejected','superseded')
                """,
                (
                    final_state, progress, final_message,
                    json.dumps(final_result, separators=(",", ":"), default=str),
                    phase_value, blocker_value, now, job_id,
                ),
            )
            if cursor.rowcount != 1:
                connection.rollback()
                return self.get(job_id)
            connection.execute(
                """
                INSERT INTO job_events
                    (job_id,state,progress,message,phase,blocker_layer,created_at)
                VALUES (?,?,?,?,?,?,?)
                """,
                (
                    job_id, final_state, progress, final_message,
                    phase_value, blocker_value, now,
                ),
            )
            connection.commit()
        return self.get(job_id)

    def events(self, job_id: str, actor: Optional[str] = None) -> list[Dict[str, Any]]:
        parent = self.get(job_id, actor=actor)
        if parent is None:
            raise KeyError(job_id)
        with closing(self._connect()) as connection:
            rows = connection.execute(
                """
                SELECT id,state,progress,message,phase,blocker_layer,created_at
                FROM job_events WHERE job_id=? ORDER BY id
                """,
                (job_id,),
            ).fetchall()
        events = [dict(row) for row in rows]
        if parent["type"] == "application.research":
            for event in events:
                event["phase"] = canonical_phase(event["phase"])
        return events

    def progress(
        self, job_id: str, state: str, percent: int, message: str, *,
        phase: Optional[str] = None, blocker_layer: Optional[str] = None,
    ) -> Dict[str, Any]:
        if state not in JOB_STATES:
            raise ValueError("Choose a supported job state.")
        percent = max(0, min(int(percent), 100))
        now = int(time.time() * 1000)
        with closing(self._connect()) as connection:
            connection.execute("BEGIN IMMEDIATE")  # pairs-with: sqlite-begin-immediate
            row = connection.execute(
                "SELECT * FROM jobs WHERE id=?", (job_id,)
            ).fetchone()
            if row is None:
                connection.rollback()
                raise KeyError(job_id)
            if row["state"] in TERMINAL_JOB_STATES:
                connection.rollback()
                raise ValueError("This job is already finished.")
            if row["state"] == "cancelling" and state != "cancelling":
                connection.rollback()
                raise JobCancelled()
            research = row["type"] == "application.research"
            phase_value = (
                canonical_phase(phase) if research and phase is not None
                else canonical_phase(row["phase"]) if research
                else row["phase"] if phase is None else phase
            )
            blocker_value = (
                row["blocker_layer"] if blocker_layer is None else blocker_layer
            )
            cursor = connection.execute(
                """
                UPDATE jobs
                SET state=?,progress=?,message=?,phase=?,
                    blocker_layer=?,updated_at=?
                WHERE id=? AND state NOT IN
                    ('completed','failed','cancelled','healthy','rejected','superseded','cancelling')
                """,
                (
                    state, percent, str(message)[:1000], phase_value,
                    blocker_value, now, job_id,
                ),
            )
            if cursor.rowcount != 1:
                connection.rollback()
                raise JobCancelled()
            connection.execute(
                """
                INSERT INTO job_events
                    (job_id,state,progress,message,phase,blocker_layer,created_at)
                VALUES (?,?,?,?,?,?,?)
                """,
                (
                    job_id, state, percent, str(message)[:1000],
                    phase_value, blocker_value, now,
                ),
            )
            connection.commit()
        return self.get(job_id)

    def request_cancel(self, job_id: str, actor: Optional[str] = None) -> Dict[str, Any]:
        now = int(time.time() * 1000)
        with closing(self._connect()) as connection:
            connection.execute("BEGIN IMMEDIATE")  # pairs-with: sqlite-begin-immediate
            row = connection.execute(
                "SELECT * FROM jobs WHERE id=? AND (? IS NULL OR actor=?)",
                (job_id, actor, actor),
            ).fetchone()
            if row is None:
                connection.rollback()
                raise KeyError(job_id)
            if row["state"] in TERMINAL_JOB_STATES:
                connection.rollback()
                raise ValueError("This job is already finished.")
            phase = (
                canonical_phase(row["phase"])
                if row["type"] == "application.research" else row["phase"]
            )
            if row["state"] == "cancelling":
                connection.rollback()
                return self._record(row)
            if row["state"] == "queued":
                state = "cancelled"
                message = "Cancelled before execution"
                progress = 0
                result = {"recovery": "No appliance changes were made."}
            else:
                state = "cancelling"
                message = "Cancellation requested"
                progress = row["progress"]
                result = json.loads(row["result_json"])
            connection.execute(
                """
                UPDATE jobs
                SET state=?,progress=?,message=?,result_json=?,phase=?,updated_at=?
                WHERE id=? AND state NOT IN
                    ('completed','failed','cancelled','healthy','rejected','superseded')
                """,
                (
                    state, progress, message, json.dumps(result, separators=(",", ":")),
                    phase, now, job_id,
                ),
            )
            connection.execute(
                """
                INSERT INTO job_events
                    (job_id,state,progress,message,phase,blocker_layer,created_at)
                VALUES (?,?,?,?,?,?,?)
                """,
                (job_id, state, progress, message, phase, row["blocker_layer"], now),
            )
            connection.commit()
        return self.get(job_id)

    def dismiss_application_research(
        self, draft_id: str, actor: str, *, reason: str = "",
    ) -> list[Dict[str, Any]]:
        """Retire terminal research records once their saved draft is resolved.

        Called both when a draft is discarded and when it is approved into a
        deployment: in either case the research is no longer "waiting to be
        resumed", so retiring it here is what stops the resume banner from
        dangling a record the reader can no longer act on. ``reason`` overrides
        the default discard wording for the approval path. Job history remains
        append-only through a superseded event, and active work is intentionally
        left to the separate cancellation workflow.
        """
        dismissal = reason or "Saved application research discarded."
        retired_message = reason or "Saved application research discarded"
        target = str(draft_id).strip()
        if not target or not actor:
            return []
        now = int(time.time() * 1000)
        dismissed_ids: list[str] = []
        with closing(self._connect()) as connection:
            connection.execute("BEGIN IMMEDIATE")  # pairs-with: sqlite-begin-immediate
            rows = connection.execute(
                """
                SELECT * FROM jobs
                WHERE actor=? AND type='application.research'
                  AND state IN ('completed','failed','healthy')
                ORDER BY created_at DESC
                """,
                (actor,),
            ).fetchall()
            for row in rows:
                try:
                    payload = json.loads(row["payload_json"])
                except (TypeError, ValueError):
                    continue
                if str(payload.get("draft_id", "")) != target:
                    continue
                result = json.loads(row["result_json"])
                result["dismissal"] = dismissal
                connection.execute(
                    """
                    UPDATE jobs
                    SET state='superseded',progress=0,message=?,result_json=?,updated_at=?
                    WHERE id=? AND state IN ('completed','failed','healthy')
                    """,
                    (
                        retired_message,
                        json.dumps(result, separators=(",", ":"), default=str),
                        now,
                        row["id"],
                    ),
                )
                connection.execute(
                    """
                    INSERT INTO job_events
                        (job_id,state,progress,message,phase,blocker_layer,created_at)
                    VALUES (?,?,?,?,?,?,?)
                    """,
                    (
                        row["id"], "superseded", 0,
                        retired_message,
                        canonical_phase(row["phase"]), row["blocker_layer"], now,
                    ),
                )
                dismissed_ids.append(row["id"])
            connection.commit()
        return [record for job_id in dismissed_ids if (record := self.get(job_id))]

    def is_cancelling(self, job_id: str) -> bool:
        current = self.get(job_id)
        return current is not None and current["state"] == "cancelling"

    def retry(self, job_id: str, actor: str, *, allow_all: bool = False) -> Dict[str, Any]:
        now = int(time.time() * 1000)
        with closing(self._connect()) as connection:
            connection.execute("BEGIN IMMEDIATE")  # pairs-with: sqlite-begin-immediate
            original = connection.execute(
                "SELECT * FROM jobs WHERE id=? AND (? OR actor=?)",
                (job_id, 1 if allow_all else 0, actor),
            ).fetchone()
            if original is None:
                connection.rollback()
                raise KeyError(job_id)
            if original["state"] not in {"failed", "cancelled"}:
                connection.rollback()
                raise ValueError("Only failed or cancelled jobs can be retried.")
            active_retry = connection.execute(
                """
                SELECT * FROM jobs
                WHERE retry_of=? AND state IN
                    ('queued','validating','downloading','starting','cancelling')
                ORDER BY created_at LIMIT 1
                """,
                (job_id,),
            ).fetchone()
            if active_retry is not None:
                connection.commit()
                record = self._record(active_retry)
                self._attach_revisions(connection, [record])
                record["deduplicated"] = True
                return record
            child_id = str(uuid.uuid4())
            phase = "interpreting" if original["type"] == "application.research" else ""
            attempt = int(original["attempt"] or 1) + 1
            connection.execute(
                """
                INSERT INTO jobs
                    (id,type,state,actor,payload_json,progress,message,phase,
                     blocker_layer,result_json,created_at,updated_at,attempt,retry_of)
                VALUES (?,?, 'queued', ?, ?, 0, 'Queued for validation', ?, '',
                        '{}', ?, ?, ?, ?)
                """,
                (
                    child_id, original["type"], actor, original["payload_json"],
                    phase, now, now, attempt, job_id,
                ),
            )
            connection.execute(
                """
                INSERT INTO job_events
                    (job_id,state,progress,message,phase,blocker_layer,created_at)
                VALUES (?,?,?,?,?,?,?)
                """,
                (child_id, "queued", 0, "Queued for validation", phase, "", now),
            )
            connection.commit()
            row = connection.execute(
                "SELECT * FROM jobs WHERE id=?", (child_id,)
            ).fetchone()
            record = self._record(row)
            self._attach_revisions(connection, [record])
        record["deduplicated"] = False
        return record

def workload_capabilities(drivers=None) -> Dict[str, Any]:
    """Report host workload tools using fixed, non-shell commands."""

    drivers = drivers or default_platform_drivers()
    os_info = drivers["operating_system"].snapshot()
    docker = drivers["container_scheduler"].capabilities()
    inference = drivers["inference_backend"].capabilities()
    return {
        "docker": {
            key: value for key, value in docker.items() if key != "id"
        },
        "os": os_info,
        "inference": inference,
        "runtime": {
            "architecture": inference.get("architecture", "unknown"),
        },
        "roots": {
            "workloads": data_path("workloads"),
            "models": data_path("models"),
        },
        "job_types": sorted(ALLOWED_JOB_TYPES),
    }
