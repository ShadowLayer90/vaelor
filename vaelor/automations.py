"""Durable one-shot and recurring read-only assistant schedules."""

from __future__ import annotations

import logging
import os
import re
import sqlite3
import threading
import time
import uuid
from contextlib import closing
from datetime import datetime, timezone
from typing import Optional

from .agent_tasks import PROFILES
from .automation_status import (
    last_run_view,
    next_slot_after,
    schedule_status,
    trigger_status,
)
from .runtime_paths import env_value, state_path

LOGGER = logging.getLogger(__name__)

# The `worker` flag marks a signal a per-worker alert can be built on: one the
# controller derives from a worker's reported row (control_plane_runtime
# WORKER_SIGNAL_FIELDS), which today is cpu_temperature and memory_percent. A
# worker also reports disk, network and fan readings (VD-205 item 6), but no
# worker alert signal is derived from them yet, and service failures and the
# fan-failure signal are controller-only. A rule on any of those aimed at a
# worker would silently never fire, so it is refused at creation - honest
# degradation rather than a rule that can never trigger.
TRIGGER_SOURCES = {
    "cpu_temperature": {"label": "CPU temperature", "minimum": 40, "maximum": 100, "worker": True},
    "memory_percent": {"label": "Memory use", "minimum": 1, "maximum": 100, "worker": True},
    "storage_percent": {"label": "Storage use", "minimum": 1, "maximum": 100, "worker": False},
    "service_failures": {"label": "Failed Vaelor services", "minimum": 1, "maximum": 10, "worker": False},
    "fan_failure": {"label": "Fan failure signal", "minimum": 1, "maximum": 1, "worker": False},
}


OWNER_REVOKED = (
    "owner_not_authorized: the account that created this rule is no longer an "
    "administrator, so its unattended runs are suspended."
)
WRITE_POLICY = (
    "Runs read only. Anything that would change something is prepared as a "
    "proposal and waits for a separate human approval."
)


class AutomationError(ValueError):
    pass


#: Recorded on a rule whose evaluation raised (a locked store, an unreadable
#: value). The raw Python error goes to the log only; the card shows this
#: sentence, and the next evaluation that reads a value clears it, so one
#: transient failure cannot leave the rule red until it next fires.
EVALUATION_FAILED = (
    "evaluation_failed: The last check of this rule failed before it could "
    "read the signal. It is retried every few seconds; the control-plane log "
    "has the details."
)

#: Refused when a one-time schedule that already ran is re-enabled. Enabling it
#: used to store it as enabled with nothing left to run, so it read green and
#: could never fire again (ACC-139).
ONE_SHOT_ALREADY_RAN = (
    "This one-time schedule has already run, so it cannot run again. "
    "Create a new schedule instead."
)


def machine_label(node: str, machine_names=None) -> str:
    """The machine a rule watches, by the name the owner knows it by.

    ``""`` is the controller. A worker is named from the fleet's own records;
    when that name cannot be read the label says so in words rather than
    printing the node id (ACC-087, ACC-127).
    """
    if not node:
        return "the controller"
    name = str((machine_names or {}).get(node) or "").strip()
    return name or "an enrolled worker"


def unattended_disclosure(profile: str, version, definition) -> dict:
    """State what an unattended run of this pinned definition may do.

    Creating the rule is the approval for every run it will ever make, so the
    person creating it has to be able to read what they are approving. This is
    the pinned definition's own capability list, not the agent's current one:
    the run loads the version recorded here, so a scope added later does not
    reach it.
    """
    if not definition:
        return {
            "agent": str(profile)[:100],
            "definition_version": int(version or 0),
            "pinned_definition_available": False,
            "reads": [],
            "web_access": "unavailable",
            "integrations": [],
            "writes": WRITE_POLICY,
        }
    web = definition.get("web_access") or {}
    domains = [str(item)[:120] for item in (web.get("allowed_domains") or [])][:20]
    return {
        "agent": str(definition.get("name", profile))[:100],
        "definition_version": int(version or 0),
        "pinned_definition_available": True,
        "reads": sorted({
            str(item)[:60]
            for item in (
                list(definition.get("scopes", []))
                + list(definition.get("permissions", []))
            )
            if str(item).strip()
        }),
        "web_access": (
            "disabled" if not web.get("enabled")
            else "allowlisted: " + ", ".join(domains) if domains
            else "guarded search, and only the pages that search returns"
        ),
        "integrations": [
            str(item.get("name", ""))[:100]
            for item in definition.get("connectors", [])
            if str(item.get("name", "")).strip()
        ][:20],
        "writes": WRITE_POLICY,
    }


def owner_still_authorized(predicate, actor: str) -> bool:
    """Re-check at fire time that the rule's owner may still run it unattended.

    Gating creation alone would have left every rule an operator already
    planted running forever, and would let a demoted administrator keep an
    unattended agent running in their name after their access was taken away.
    Authorization is a property of now, not of the moment the row was written.
    An unreadable answer is treated as "no": a rule nobody is watching must not
    run on a failure to check who owns it.
    """
    if predicate is None:
        return True
    try:
        return bool(predicate(actor))
    except Exception:
        return False


def parse_schedule(text: str, now: Optional[float] = None):
    clean = str(text).strip().lower()
    current = now if now is not None else time.time()
    match = re.fullmatch(r"in\s+(\d+)\s+(minute|minutes|hour|hours)", clean)
    if match:
        amount = int(match.group(1))
        seconds = amount * (3600 if "hour" in match.group(2) else 60)
        if not 60 <= seconds <= 365 * 86400:
            raise AutomationError("One-shot schedules must be between one minute and one year.")
        return {"kind": "once", "next_run_at": current + seconds, "interval_seconds": None}
    match = re.fullmatch(r"every\s+(\d+)\s+(minute|minutes|hour|hours|day|days)", clean)
    if match:
        amount = int(match.group(1))
        unit = match.group(2)
        multiplier = 86400 if "day" in unit else 3600 if "hour" in unit else 60
        seconds = amount * multiplier
        if not 300 <= seconds <= 30 * 86400:
            raise AutomationError("Recurring schedules must run between every five minutes and every 30 days.")
        return {"kind": "interval", "next_run_at": current + seconds, "interval_seconds": seconds}
    try:
        parsed = datetime.fromisoformat(clean.replace("z", "+00:00"))
        if parsed.tzinfo is None:
            parsed = parsed.replace(tzinfo=timezone.utc)
        timestamp = parsed.timestamp()
        if timestamp <= current:
            raise AutomationError("Choose a future date and time.")
        return {"kind": "once", "next_run_at": timestamp, "interval_seconds": None}
    except ValueError as error:
        raise AutomationError(
            "Use “in 30 minutes”, “every 6 hours”, or an ISO date and time."
        ) from error


class AutomationStore:
    def __init__(self, database_path: Optional[str] = None, profile_store=None,
                 delivery_async: bool = True):
        self.profile_store = profile_store
        # Out-of-band alert delivery (email/webhook) blocks on the network. It
        # runs off the trigger-evaluation thread by default so a hung relay
        # cannot delay OTHER triggers' evaluation; tests set this False to
        # deliver inline and deterministically.
        self._delivery_async = delivery_async
        self.database_path = database_path or env_value(
            "VAELOR_AUTOMATIONS_DB", "PM_AUTOMATIONS_DB",
            state_path("assistant/automations.sqlite3"),
        )
        parent = os.path.dirname(self.database_path)
        if parent:
            os.makedirs(parent, mode=0o700, exist_ok=True)
        self._initialize()
        try:
            os.chmod(self.database_path, 0o600)
        except OSError:
            pass

    def _connect(self):
        connection = sqlite3.connect(self.database_path, timeout=15)
        connection.row_factory = sqlite3.Row
        return connection

    def _initialize(self):
        with closing(self._connect()) as connection:
            connection.executescript(
                """
                CREATE TABLE IF NOT EXISTS automations (
                    id TEXT PRIMARY KEY,
                    actor TEXT NOT NULL,
                    name TEXT NOT NULL,
                    prompt TEXT NOT NULL,
                    profile TEXT NOT NULL,
                    kind TEXT NOT NULL,
                    schedule_text TEXT NOT NULL,
                    next_run_at REAL,
                    interval_seconds INTEGER,
                    enabled INTEGER NOT NULL,
                    created_at REAL NOT NULL,
                    updated_at REAL NOT NULL
                );
                CREATE TABLE IF NOT EXISTS automation_runs (
                    id TEXT PRIMARY KEY,
                    automation_id TEXT NOT NULL,
                    scheduled_for REAL NOT NULL,
                    task_id TEXT,
                    state TEXT NOT NULL,
                    message TEXT NOT NULL,
                    created_at REAL NOT NULL,
                    FOREIGN KEY(automation_id) REFERENCES automations(id) ON DELETE CASCADE
                );
                CREATE UNIQUE INDEX IF NOT EXISTS idx_automation_run_once
                ON automation_runs(automation_id,scheduled_for);
                CREATE INDEX IF NOT EXISTS idx_automation_runs_latest
                ON automation_runs(automation_id,created_at);
                CREATE INDEX IF NOT EXISTS idx_automation_runs_task
                ON automation_runs(task_id);
                CREATE TABLE IF NOT EXISTS automation_triggers (
                    id TEXT PRIMARY KEY, actor TEXT NOT NULL, name TEXT NOT NULL,
                    prompt TEXT NOT NULL, profile TEXT NOT NULL, source TEXT NOT NULL,
                    operator TEXT NOT NULL, threshold REAL NOT NULL,
                    cooldown_seconds INTEGER NOT NULL, enabled INTEGER NOT NULL,
                    last_triggered_at REAL, last_value REAL, node TEXT NOT NULL DEFAULT '',
                    created_at REAL NOT NULL, updated_at REAL NOT NULL
                );
                CREATE TABLE IF NOT EXISTS automation_trigger_runs (
                    id TEXT PRIMARY KEY, trigger_id TEXT NOT NULL, value REAL NOT NULL,
                    task_id TEXT NOT NULL, created_at REAL NOT NULL,
                    FOREIGN KEY(trigger_id) REFERENCES automation_triggers(id) ON DELETE CASCADE
                );
                CREATE INDEX IF NOT EXISTS idx_automation_trigger_runs_task
                ON automation_trigger_runs(task_id);
                """
            )
            for table in ("automations", "automation_triggers"):
                columns = {
                    row["name"] for row in connection.execute(
                        "PRAGMA table_info({})".format(table)
                    )
                }
                if "profile_version" not in columns:
                    connection.execute(
                        "ALTER TABLE {} ADD COLUMN profile_version INTEGER NOT NULL DEFAULT 0"
                        .format(table)
                    )
            trigger_columns = {
                row["name"] for row in connection.execute(
                    "PRAGMA table_info(automation_triggers)"
                )
            }
            if "last_error" not in trigger_columns:
                connection.execute(
                    "ALTER TABLE automation_triggers ADD COLUMN last_error TEXT NOT NULL DEFAULT ''"
                )
            if "last_delivery" not in trigger_columns:
                connection.execute(
                    "ALTER TABLE automation_triggers ADD COLUMN last_delivery TEXT NOT NULL DEFAULT ''"
                )
            # Per-worker alert thresholds (VD-128). An appliance DB written before
            # this column existed must gain it in place, or every trigger read
            # would raise on the missing `node`. Empty is the controller, so the
            # migration leaves every existing rule pointing at the controller —
            # exactly the machine it evaluated against before.
            if "node" not in trigger_columns:
                connection.execute(
                    "ALTER TABLE automation_triggers ADD COLUMN node TEXT NOT NULL DEFAULT ''"
                )
            # When the rule last READ its signal (ACC-086). `last_value` alone
            # said nothing about its age, so a machine that stopped reporting
            # left a frozen number under a green "watching" pill.
            if "last_value_at" not in trigger_columns:
                connection.execute(
                    "ALTER TABLE automation_triggers ADD COLUMN last_value_at REAL"
                )
            connection.commit()

    def _row(self, row):
        item = dict(row)
        item["enabled"] = bool(item["enabled"])
        # A trigger names the machine it watches; "" is the controller. The
        # migration makes the column always present, but a defensive default
        # keeps a pre-migration read (schedules have no node) from raising.
        item.setdefault("node", "")
        item["capability_disclosure"] = self._disclosure(
            item["profile"], item.get("profile_version", 0), item["actor"]
        )
        return item

    def _disclosure(self, profile: str, version, actor: str) -> dict:
        definition = PROFILES.get(profile)
        if definition is None and self.profile_store is not None:
            try:
                definition = self.profile_store.get_version(
                    profile, actor, int(version or 0)
                )
            except (AttributeError, OSError, TypeError, ValueError):
                definition = None
        return unattended_disclosure(profile, version, definition)

    def create(self, actor: str, name: str, prompt: str, profile: str, schedule_text: str):
        name = str(name).strip()[:100]
        prompt = str(prompt).strip()[:4000]
        if not name or not prompt:
            raise AutomationError("A schedule name and task are required.")
        profile_version = self._profile_version(profile, actor)
        parsed = parse_schedule(schedule_text)
        now = time.time()
        item_id = uuid.uuid4().hex
        with closing(self._connect()) as connection:
            connection.execute(
                """
                INSERT INTO automations
                (id,actor,name,prompt,profile,profile_version,kind,schedule_text,next_run_at,
                 interval_seconds,enabled,created_at,updated_at)
                VALUES(?,?,?,?,?,?,?,?,?,?,1,?,?)
                """,
                (item_id, actor, name, prompt, profile, profile_version, parsed["kind"],
                 str(schedule_text).strip()[:120], parsed["next_run_at"],
                 parsed["interval_seconds"], now, now),
            )
            connection.commit()
        return self.get(item_id, actor)

    def _profile_version(self, profile: str, actor: str) -> int:
        if profile in PROFILES:
            return 0
        definition = self.profile_store.get(profile, actor) if self.profile_store else None
        if (
            not definition
            or not definition.get("enabled")
            or str(definition.get("surface", "assistant")) == "inference"
        ):
            raise AutomationError("Choose an enabled built-in or custom agent.")
        return int(definition.get("version", 0))

    def get(self, item_id: str, actor: Optional[str] = None):
        query = "SELECT * FROM automations WHERE id=?"
        values = [item_id]
        if actor is not None:
            query += " AND actor=?"
            values.append(actor)
        with closing(self._connect()) as connection:
            row = connection.execute(query, values).fetchone()
            return self._schedule_views(connection, [self._row(row)])[0] if row else None

    def list(self, actor: Optional[str] = None):
        query = "SELECT * FROM automations"
        values = []
        if actor is not None:
            query += " WHERE actor=?"
            values.append(actor)
        query += " ORDER BY created_at DESC"
        with closing(self._connect()) as connection:
            items = [self._row(row) for row in connection.execute(query, values)]
            return self._schedule_views(connection, items)

    @staticmethod
    def _schedule_views(connection, items):
        """Attach each schedule's latest recorded run and its derived status.

        Failed and blocked runs were recorded in ``automation_runs`` and read
        by nothing, so a schedule whose every run failed stayed green (ACC-135).
        """
        latest = {
            row["automation_id"]: dict(row)
            for row in AutomationStore._latest_runs(connection, [item["id"] for item in items])
        }
        now = time.time()
        for item in items:
            run = latest.get(item["id"])
            item["last_run"] = last_run_view(run)
            item["status"] = schedule_status(item, run, now)
        return items

    @staticmethod
    def _latest_runs(connection, automation_ids):
        """The newest recorded run of each schedule - one row each, chosen in SQL.

        A schedule running every five minutes records ~8,600 runs a month; the
        list must not read all of them to show one. One ``LIMIT 1`` read per
        schedule walks the ``(automation_id, created_at)`` index backwards, so
        each costs one index seek however long the history grows.
        """
        rows = []
        for automation_id in automation_ids:
            row = connection.execute(
                "SELECT * FROM automation_runs WHERE automation_id=? "
                "ORDER BY created_at DESC, rowid DESC LIMIT 1",
                (automation_id,),
            ).fetchone()
            if row is not None:
                rows.append(row)
        return rows

    def task_origins(self, task_ids):
        """Which recorded run started each task: ``alert_rule`` or ``schedule``.

        The recorded fact, not the task's idempotency key: a caller of
        ``POST /assistant/tasks`` chooses its own key, so a key shaped
        ``trigger:...`` proves nothing (ACC-127 review, S3). A task id absent
        from both run tables was not started by a rule or a schedule.
        """
        ids = [str(item) for item in task_ids if item]
        if not ids:
            return {}
        marks = ",".join("?" for _ in ids)
        origins = {}
        with closing(self._connect()) as connection:
            for row in connection.execute(
                "SELECT task_id FROM automation_runs WHERE task_id IN ({})".format(marks), ids,
            ):
                origins[row["task_id"]] = "schedule"
            for row in connection.execute(
                "SELECT task_id FROM automation_trigger_runs WHERE task_id IN ({})".format(marks), ids,
            ):
                origins[row["task_id"]] = "alert_rule"
        return origins

    def set_enabled(self, item_id: str, actor: str, enabled: bool):
        current = self.get(item_id, actor)
        if current is None:
            raise AutomationError("Schedule not found.")
        if enabled and current["kind"] == "once" and current["next_run_at"] is None:
            raise AutomationError(ONE_SHOT_ALREADY_RAN)
        with closing(self._connect()) as connection:
            cursor = connection.execute(
                "UPDATE automations SET enabled=?,updated_at=? WHERE id=? AND actor=?",
                (int(enabled), time.time(), item_id, actor),
            )
            connection.commit()
        if not cursor.rowcount:
            raise AutomationError("Schedule not found.")
        return self.get(item_id, actor)

    def delete(self, item_id: str, actor: str):
        with closing(self._connect()) as connection:
            row = connection.execute(
                "SELECT id FROM automations WHERE id=? AND actor=?",
                (item_id, actor),
            ).fetchone()
            if row is None:
                raise AutomationError("Schedule not found.")
            connection.execute(
                "DELETE FROM automation_runs WHERE automation_id=?", (item_id,)
            )
            connection.execute("DELETE FROM automations WHERE id=?", (item_id,))
            connection.commit()
        return {"deleted": True, "id": item_id}

    def due(self, now: Optional[float] = None, limit: int = 20):
        current = now if now is not None else time.time()
        with closing(self._connect()) as connection:
            return [
                self._row(row) for row in connection.execute(
                    "SELECT * FROM automations WHERE enabled=1 AND next_run_at IS NOT NULL AND next_run_at<=? ORDER BY next_run_at LIMIT ?",
                    (current, max(1, min(limit, 100))),
                )
            ]

    def record_run(
        self, automation, task_id: Optional[str], *, state: str = "queued",
        message: str = "Read-only agent task created",
    ):
        scheduled = automation["next_run_at"]
        now = time.time()
        with closing(self._connect()) as connection:
            connection.execute(
                "INSERT OR IGNORE INTO automation_runs(id,automation_id,scheduled_for,task_id,state,message,created_at) VALUES(?,?,?,?,?,?,?)",
                (uuid.uuid4().hex, automation["id"], scheduled, task_id,
                 state, str(message)[:500], now),
            )
            if automation["kind"] == "once":
                connection.execute(
                    "UPDATE automations SET enabled=0,next_run_at=NULL,updated_at=? WHERE id=?",
                    (now, automation["id"]),
                )
            else:
                # The next slot AFTER now, not after the slot that just ran: a
                # paused or missed schedule runs once and resumes its cadence
                # instead of firing one catch-up run per poll (ACC-139).
                connection.execute(
                    "UPDATE automations SET next_run_at=?,updated_at=? WHERE id=?",
                    (next_slot_after(scheduled, automation["interval_seconds"], now),
                     now, automation["id"]),
                )
            connection.commit()

    def runs(self, automation_id: Optional[str] = None, limit: int = 100):
        query = "SELECT * FROM automation_runs"
        values = []
        if automation_id:
            query += " WHERE automation_id=?"
            values.append(automation_id)
        query += " ORDER BY created_at DESC LIMIT ?"
        values.append(max(1, min(limit, 200)))
        with closing(self._connect()) as connection:
            return [dict(row) for row in connection.execute(query, values)]

    def trigger_runs(self, limit: int = 100):
        """Alert-rule runs, so the client can label them automatic.

        The trigger evaluator records each fired alert in
        ``automation_trigger_runs``. Those task ids are authoritative for "an
        alert rule started this", exactly as ``automation_runs`` is for a
        schedule. Without returning them, an alert-rule run had no automatic
        marker and classified under Checks beside runs the operator typed.
        """
        with closing(self._connect()) as connection:
            rows = connection.execute(
                "SELECT id,trigger_id,value,task_id,created_at "
                "FROM automation_trigger_runs "
                "ORDER BY created_at DESC LIMIT ?",
                (max(1, min(limit, 200)),),
            )
            return [dict(row) for row in rows]

    def trigger_runs_between(self, since: float, until: float):
        """Fired alert runs in a time span with their rule's machine, oldest first.

        ``node`` is ``""`` for a rule on the controller (VD-205 item 3, review B-2).
        """
        with closing(self._connect()) as connection:
            rows = connection.execute(
                "SELECT r.trigger_id, r.value, r.created_at, t.name, t.source, t.node "
                "FROM automation_trigger_runs r JOIN automation_triggers t ON t.id = r.trigger_id "
                "WHERE r.created_at >= ? AND r.created_at <= ? ORDER BY r.created_at ASC LIMIT 500",
                (float(since), float(until)),
            )
            return [dict(row) for row in rows]

    def sync_runs(self, task_store, limit: int = 200):
        """Persist the current task state in scheduled-run history."""
        runs = self.runs(limit=limit)
        if task_store is None:
            return runs
        updates = []
        for run in runs:
            task = task_store.get(run.get("task_id"), None)
            if task is None:
                continue
            state = str(task.get("state", "queued"))
            message = (
                str(task.get("error", "")).strip()
                or str((task.get("result") or {}).get("summary", "")).strip()
                or "Agent task {}".format(state.replace("_", " "))
            )[:500]
            if state != run["state"] or message != run["message"]:
                updates.append((state, message, run["id"]))
        if updates:
            with closing(self._connect()) as connection:
                connection.executemany(
                    "UPDATE automation_runs SET state=?,message=? WHERE id=?",
                    updates,
                )
                connection.commit()
        return self.runs(limit=limit)

    def create_trigger(
        self, actor: str, name: str, prompt: str, profile: str,
        source: str, operator: str, threshold: float, cooldown_seconds: int = 1800,
        node: str = "",
    ):
        name = str(name).strip()[:100]
        prompt = str(prompt).strip()[:4000]
        node = str(node).strip()[:64]
        if not name or not prompt:
            raise AutomationError("An alert name and specialist task are required.")
        profile_version = self._profile_version(profile, actor)
        if source not in TRIGGER_SOURCES:
            raise AutomationError("Choose a supported alert signal.")
        if operator not in {">=", "<="}:
            raise AutomationError("Choose above or below threshold.")
        # Only cpu_temperature and memory_percent are derived as alert signals
        # for a worker, so a rule aimed at one may watch only those. Refusing a controller-only signal
        # here keeps a per-worker rule from being stored in a state that could
        # never fire. The controller ("") keeps all five. Whether the node is a
        # real enrolled worker is the route's check, not the store's — the store
        # only owns the source/node compatibility rule.
        if node and not TRIGGER_SOURCES[source].get("worker"):
            raise AutomationError("That signal can raise an alert only for the controller.")
        try:
            threshold = float(threshold)
            cooldown_seconds = int(cooldown_seconds)
        except (TypeError, ValueError) as error:
            raise AutomationError("Enter a valid threshold and cooldown.") from error
        limits = TRIGGER_SOURCES[source]
        if not limits["minimum"] <= threshold <= limits["maximum"]:
            raise AutomationError("The alert threshold is outside the safe range.")
        if not 300 <= cooldown_seconds <= 7 * 86400:
            raise AutomationError("Alert cooldown must be between five minutes and seven days.")
        now = time.time()
        trigger_id = uuid.uuid4().hex
        with closing(self._connect()) as connection:
            connection.execute(
                """
                INSERT INTO automation_triggers
                (id,actor,name,prompt,profile,profile_version,source,operator,threshold,cooldown_seconds,
                 node,enabled,created_at,updated_at)
                VALUES(?,?,?,?,?,?,?,?,?,?,?,1,?,?)
                """,
                (trigger_id, actor, name, prompt, profile, profile_version, source, operator,
                 threshold, cooldown_seconds, node, now, now),
            )
            connection.commit()
        return self.get_trigger(trigger_id, actor)

    def _trigger_row(self, row, now: float):
        item = self._row(row)
        item["status"] = trigger_status(item, now)
        # The signal's words from TRIGGER_SOURCES, the one table that owns
        # them, so the cards never keep a second copy of the labels.
        item["signal_label"] = TRIGGER_SOURCES.get(item["source"], {}).get(
            "label", "An unrecognised signal"
        )
        return item

    def get_trigger(self, trigger_id: str, actor: Optional[str] = None):
        query = "SELECT * FROM automation_triggers WHERE id=?"
        values = [trigger_id]
        if actor is not None:
            query += " AND actor=?"
            values.append(actor)
        with closing(self._connect()) as connection:
            row = connection.execute(query, values).fetchone()
        return self._trigger_row(row, time.time()) if row else None

    def list_triggers(self, actor: Optional[str] = None):
        query = "SELECT * FROM automation_triggers"
        values = []
        if actor is not None:
            query += " WHERE actor=?"
            values.append(actor)
        query += " ORDER BY created_at DESC"
        now = time.time()
        with closing(self._connect()) as connection:
            return [self._trigger_row(row, now) for row in connection.execute(query, values)]

    def set_trigger_enabled(self, trigger_id: str, actor: str, enabled: bool):
        with closing(self._connect()) as connection:
            cursor = connection.execute(
                "UPDATE automation_triggers SET enabled=?,updated_at=? WHERE id=? AND actor=?",
                (int(enabled), time.time(), trigger_id, actor),
            )
            connection.commit()
        if not cursor.rowcount:
            raise AutomationError("Alert rule not found.")
        return self.get_trigger(trigger_id, actor)

    def delete_trigger(self, trigger_id: str, actor: str):
        with closing(self._connect()) as connection:
            row = connection.execute(
                "SELECT id FROM automation_triggers WHERE id=? AND actor=?",
                (trigger_id, actor),
            ).fetchone()
            if row is None:
                raise AutomationError("Alert rule not found.")
            connection.execute(
                "DELETE FROM automation_trigger_runs WHERE trigger_id=?",
                (trigger_id,),
            )
            connection.execute(
                "DELETE FROM automation_triggers WHERE id=?", (trigger_id,)
            )
            connection.commit()
        return {"deleted": True, "id": trigger_id}

    def _record_trigger_error(self, trigger_id: str, message: str, now: float):
        with closing(self._connect()) as connection:
            connection.execute(
                "UPDATE automation_triggers SET last_error=?,updated_at=? WHERE id=?",
                (str(message)[:480], now, trigger_id),
            )
            connection.commit()

    def _record_trigger_delivery(self, trigger_id: str, summary: str, now: float):
        with closing(self._connect()) as connection:
            connection.execute(
                "UPDATE automation_triggers SET last_delivery=?,updated_at=? WHERE id=?",
                (str(summary)[:480], now, trigger_id),
            )
            connection.commit()

    def evaluate_triggers(self, values_by_node: dict[str, dict[str, float]], task_store,
                          owner_authorized=None, deliver=None, machine_names=None):
        """Evaluate every enabled rule against its own machine's readings.

        ``values_by_node`` is keyed by machine: ``""`` holds the controller's
        readings and each worker node-id key holds that worker's. A rule reads
        only its own node's dict, so a worker's memory has no bearing on a
        controller rule and vice versa. A node ABSENT from the dict is a machine
        that did not report this pass (stale, never enrolled, or telemetry off);
        its rules are skipped rather than fired, because a missing reading is not
        a reading of zero and must never raise a false alarm.

        Each rule is evaluated on its own: one rule that raises records its own
        ``last_error`` (shown on its card) and the pass moves on, instead of the
        whole pass being dropped in silence (ACC-135). ``machine_names`` maps a
        node id to the name the owner knows it by, so a fired alert and its run
        name the machine rather than printing its id (ACC-087, ACC-127).
        """
        now = time.time()
        created = []
        for trigger in self.list_triggers():
            if not trigger["enabled"]:
                continue
            try:
                task = self._evaluate_trigger(
                    trigger, values_by_node, task_store, owner_authorized,
                    deliver, machine_names, now,
                )
            except (sqlite3.Error, OSError, TypeError, ValueError) as error:
                LOGGER.warning("Alert rule %s could not be evaluated: %s", trigger["id"], error)
                try:
                    self._record_trigger_error(trigger["id"], EVALUATION_FAILED, now)
                except sqlite3.Error as record_error:
                    LOGGER.warning("Could not record the alert rule error: %s", record_error)
                continue
            if task is not None:
                created.append(task)
        return created

    def _evaluate_trigger(self, trigger, values_by_node, task_store, owner_authorized,
                          deliver, machine_names, now):
        """Evaluate one enabled rule; the task it created, or ``None``."""
        node_values = values_by_node.get(trigger["node"])
        # The node is not reporting this pass: skip, do not fire. Distinct
        # from a node that reported without this field (below), which is also
        # a skip but for a different reason.
        if node_values is None or trigger["source"] not in node_values:
            return None
        if not owner_still_authorized(owner_authorized, trigger["actor"]):
            self._record_trigger_error(trigger["id"], OWNER_REVOKED, now)
            return None
        value = float(node_values[trigger["source"]])
        matched = (
            value >= trigger["threshold"]
            if trigger["operator"] == ">="
            else value <= trigger["threshold"]
        )
        cooled = (
            trigger["last_triggered_at"] is None
            or now - trigger["last_triggered_at"] >= trigger["cooldown_seconds"]
        )
        with closing(self._connect()) as connection:
            # The reading and WHEN it was read. The two errors that describe
            # a check rather than a fired run - a revoked owner and a failed
            # evaluation - clear as soon as a check succeeds; a failed task
            # creation stays until the rule next fires cleanly.
            connection.execute(
                "UPDATE automation_triggers SET last_value=?,last_value_at=?,"
                "last_error=CASE WHEN last_error IN (?,?) THEN '' ELSE last_error END,"
                "updated_at=? WHERE id=?",
                (value, now, OWNER_REVOKED, EVALUATION_FAILED, now, trigger["id"]),
            )
            connection.commit()
        if not matched or not cooled:
            return None
        # Name the machine that crossed. A fired alert that only said the
        # signal and value left the reader guessing whether it was the
        # controller or one of several workers; the diagnostic run and the
        # out-of-band alert both have to say which box to look at - by its
        # name, not its node id.
        machine = machine_label(trigger["node"], machine_names)
        signal = TRIGGER_SOURCES.get(trigger["source"], {}).get("label", trigger["source"])
        try:
            task = task_store.create(
                trigger["actor"],
                "Alert: {} on {}".format(trigger["name"], machine),
                "{}\n\nObserved {} {} {} {} on {}.".format(
                    trigger["prompt"], signal, value, trigger["operator"],
                    trigger["threshold"], machine,
                ),
                kind="durable", profile=trigger["profile"], approval_required=False,
                profile_version=(trigger["profile_version"] or None),
                idempotency_key="trigger:{}:{}".format(trigger["id"], int(now)),
            )
        except ValueError as error:
            self._record_trigger_error(
                trigger["id"], "task_creation: " + str(error)[:480], now
            )
            return None
        with closing(self._connect()) as connection:
            connection.execute(
                "UPDATE automation_triggers SET last_triggered_at=?,last_value=?,last_error='',updated_at=? WHERE id=?",
                (now, value, now, trigger["id"]),
            )
            connection.execute(
                "INSERT INTO automation_trigger_runs(id,trigger_id,value,task_id,created_at) VALUES(?,?,?,?,?)",
                (uuid.uuid4().hex, trigger["id"], value, task["id"], now),
            )
            connection.commit()
        # Out-of-band delivery is best-effort and strictly after the run is
        # durable: a broken email relay or webhook must never fail the
        # trigger, lose the task, or raise into this loop. It also blocks on
        # the network, so it runs OFF this evaluation thread by default - a
        # hung relay must not delay the other triggers in this pass or the
        # next poll cycle. The recorded outcome lands when delivery finishes.
        if deliver is not None:
            alert = {
                "trigger_name": trigger["name"],
                "source": trigger["source"],
                "signal": signal,
                # The machine by name, for a person reading the email or
                # the chat message; `node` keeps the id for a program.
                "machine": machine,
                "operator": trigger["operator"],
                "threshold": trigger["threshold"],
                "observed_value": value,
                # Which machine crossed, so an emailed/webhooked alert names
                # the box. "controller" rather than "" for a human reader.
                "node": trigger["node"] or "controller",
                "timestamp": now,
                "task_id": task["id"],
            }
            if self._delivery_async:
                threading.Thread(
                    target=self._run_delivery,
                    args=(deliver, trigger["id"], alert, now),
                    name="pm-alert-delivery", daemon=True,
                ).start()
            else:
                self._run_delivery(deliver, trigger["id"], alert, now)
        return task

    def _run_delivery(self, deliver, trigger_id: str, alert: dict, now: float):
        """Deliver one fired alert and record its outcome. Never raises.

        Runs on a delivery thread (or inline in tests). The bound ``deliver``
        callback captures its own per-channel failures and returns a short
        summary; anything it still throws is swallowed and recorded so a broken
        channel can never escape into the caller.
        """
        try:
            summary = deliver(alert) or ""
        except Exception as error:  # noqa: BLE001 - delivery is non-fatal
            summary = "delivery_error: {}".format(str(error)[:200])
        try:
            self._record_trigger_delivery(trigger_id, summary, now)
        except Exception:  # noqa: BLE001 - a locked store must not kill the thread
            pass


class AutomationRunner:
    def __init__(self, store, task_store, context_provider=None,
                 poll_seconds: float = 5, owner_authorized=None, deliver=None,
                 machine_names=None):
        self.store = store
        self.task_store = task_store
        self.poll_seconds = max(1, min(float(poll_seconds), 60))
        self.context_provider = context_provider
        self.owner_authorized = owner_authorized
        # Optional out-of-band alert delivery. Left None keeps the historical
        # behaviour (a fired trigger creates a task and nothing is sent), so
        # existing callers and tests are unaffected.
        self.deliver = deliver
        # Optional: node id -> the name the owner knows the machine by.
        self.machine_names = machine_names
        self._stop = threading.Event()
        self._thread = None

    def start(self):
        if self._thread and self._thread.is_alive():
            return
        self._thread = threading.Thread(
            target=self._loop, name="pm-automation-runner", daemon=True
        )
        self._thread.start()

    def _loop(self):
        while not self._stop.is_set():
            try:
                self.run_once()
            except Exception:  # noqa: BLE001 - one bad pass must not end the scheduler
                # Logged with its traceback: an exception here used to end the
                # thread, and every schedule and alert rule with it, silently.
                LOGGER.exception("The schedule and alert-rule pass failed")
            self._stop.wait(self.poll_seconds)

    def run_once(self, now: Optional[float] = None):
        created = []
        for automation in self.store.due(now):
            if not owner_still_authorized(self.owner_authorized, automation["actor"]):
                self.store.record_run(
                    automation, None, state="blocked", message=OWNER_REVOKED,
                )
                continue
            try:
                task = self.task_store.create(
                    automation["actor"],
                    "Scheduled: {}".format(automation["name"]),
                    automation["prompt"], kind="durable",
                    profile=automation["profile"],
                    profile_version=(automation["profile_version"] or None),
                    approval_required=False,
                    idempotency_key="automation:{}:{}".format(
                        automation["id"], int(automation["next_run_at"])
                    ),
                )
            except ValueError as error:
                self.store.record_run(
                    automation, None, state="failed",
                    message="task_creation: " + str(error),
                )
                continue
            self.store.record_run(automation, task["id"])
            created.append(task)
        if self.context_provider is not None:
            try:
                created.extend(
                    self.store.evaluate_triggers(
                        self.context_provider(), self.task_store,
                        owner_authorized=self.owner_authorized,
                        deliver=self.deliver,
                        machine_names=self._machine_names(),
                    )
                )
            except (AutomationError, OSError, ValueError, sqlite3.Error) as error:
                # Swallowed with `pass` before (ACC-135): no reading, no rule
                # evaluated, and nothing anywhere said so. Rules that did not
                # get a fresh reading show "Not reporting" on their cards.
                LOGGER.warning("Alert rules could not be evaluated this pass: %s", error)
        self.store.sync_runs(self.task_store)
        return created

    def _machine_names(self):
        if self.machine_names is None:
            return {}
        try:
            return dict(self.machine_names() or {})
        except Exception as error:  # noqa: BLE001 - names only label an alert
            LOGGER.warning("Machine names for alerts could not be read: %s", error)
            return {}
