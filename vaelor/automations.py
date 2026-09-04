"""Durable one-shot and recurring read-only assistant schedules."""

from __future__ import annotations

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
from .runtime_paths import env_value, state_path

TRIGGER_SOURCES = {
    "cpu_temperature": {"label": "CPU temperature", "minimum": 40, "maximum": 100},
    "memory_percent": {"label": "Memory use", "minimum": 1, "maximum": 100},
    "storage_percent": {"label": "Storage use", "minimum": 1, "maximum": 100},
    "service_failures": {"label": "Failed Vaelor services", "minimum": 1, "maximum": 10},
    "fan_failure": {"label": "Fan failure signal", "minimum": 1, "maximum": 1},
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
                CREATE TABLE IF NOT EXISTS automation_triggers (
                    id TEXT PRIMARY KEY, actor TEXT NOT NULL, name TEXT NOT NULL,
                    prompt TEXT NOT NULL, profile TEXT NOT NULL, source TEXT NOT NULL,
                    operator TEXT NOT NULL, threshold REAL NOT NULL,
                    cooldown_seconds INTEGER NOT NULL, enabled INTEGER NOT NULL,
                    last_triggered_at REAL, last_value REAL,
                    created_at REAL NOT NULL, updated_at REAL NOT NULL
                );
                CREATE TABLE IF NOT EXISTS automation_trigger_runs (
                    id TEXT PRIMARY KEY, trigger_id TEXT NOT NULL, value REAL NOT NULL,
                    task_id TEXT NOT NULL, created_at REAL NOT NULL,
                    FOREIGN KEY(trigger_id) REFERENCES automation_triggers(id) ON DELETE CASCADE
                );
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
            connection.commit()

    def _row(self, row):
        item = dict(row)
        item["enabled"] = bool(item["enabled"])
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
        if not definition or not definition.get("enabled"):
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
        return self._row(row) if row else None

    def list(self, actor: Optional[str] = None):
        query = "SELECT * FROM automations"
        values = []
        if actor is not None:
            query += " WHERE actor=?"
            values.append(actor)
        query += " ORDER BY created_at DESC"
        with closing(self._connect()) as connection:
            return [self._row(row) for row in connection.execute(query, values)]

    def set_enabled(self, item_id: str, actor: str, enabled: bool):
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
                connection.execute(
                    "UPDATE automations SET next_run_at=?,updated_at=? WHERE id=?",
                    (scheduled + automation["interval_seconds"], now, automation["id"]),
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
    ):
        name = str(name).strip()[:100]
        prompt = str(prompt).strip()[:4000]
        if not name or not prompt:
            raise AutomationError("An alert name and specialist task are required.")
        profile_version = self._profile_version(profile, actor)
        if source not in TRIGGER_SOURCES:
            raise AutomationError("Choose a supported alert signal.")
        if operator not in {">=", "<="}:
            raise AutomationError("Choose above or below threshold.")
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
                 enabled,created_at,updated_at)
                VALUES(?,?,?,?,?,?,?,?,?,?,1,?,?)
                """,
                (trigger_id, actor, name, prompt, profile, profile_version, source, operator,
                 threshold, cooldown_seconds, now, now),
            )
            connection.commit()
        return self.get_trigger(trigger_id, actor)

    def get_trigger(self, trigger_id: str, actor: Optional[str] = None):
        query = "SELECT * FROM automation_triggers WHERE id=?"
        values = [trigger_id]
        if actor is not None:
            query += " AND actor=?"
            values.append(actor)
        with closing(self._connect()) as connection:
            row = connection.execute(query, values).fetchone()
        return self._row(row) if row else None

    def list_triggers(self, actor: Optional[str] = None):
        query = "SELECT * FROM automation_triggers"
        values = []
        if actor is not None:
            query += " WHERE actor=?"
            values.append(actor)
        query += " ORDER BY created_at DESC"
        with closing(self._connect()) as connection:
            return [self._row(row) for row in connection.execute(query, values)]

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

    def evaluate_triggers(self, values: dict[str, float], task_store,
                          owner_authorized=None, deliver=None):
        now = time.time()
        created = []
        for trigger in self.list_triggers():
            if not trigger["enabled"] or trigger["source"] not in values:
                continue
            if not owner_still_authorized(owner_authorized, trigger["actor"]):
                self._record_trigger_error(trigger["id"], OWNER_REVOKED, now)
                continue
            value = float(values[trigger["source"]])
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
                connection.execute(
                    "UPDATE automation_triggers SET last_value=?,updated_at=? WHERE id=?",
                    (value, now, trigger["id"]),
                )
                connection.commit()
            if not matched or not cooled:
                continue
            try:
                task = task_store.create(
                    trigger["actor"],
                    "Alert: {}".format(trigger["name"]),
                    "{}\n\nObserved {} {} {}.".format(
                        trigger["prompt"], trigger["source"], trigger["operator"], value
                    ),
                    kind="durable", profile=trigger["profile"], approval_required=False,
                    profile_version=(trigger["profile_version"] or None),
                    idempotency_key="trigger:{}:{}".format(trigger["id"], int(now)),
                )
            except ValueError as error:
                self._record_trigger_error(
                    trigger["id"], "task_creation: " + str(error)[:480], now
                )
                continue
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
                    "operator": trigger["operator"],
                    "threshold": trigger["threshold"],
                    "observed_value": value,
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
            created.append(task)
        return created

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
                 poll_seconds: float = 5, owner_authorized=None, deliver=None):
        self.store = store
        self.task_store = task_store
        self.poll_seconds = max(1, min(float(poll_seconds), 60))
        self.context_provider = context_provider
        self.owner_authorized = owner_authorized
        # Optional out-of-band alert delivery. Left None keeps the historical
        # behaviour (a fired trigger creates a task and nothing is sent), so
        # existing callers and tests are unaffected.
        self.deliver = deliver
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
            self.run_once()
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
                    )
                )
            except (AutomationError, OSError, ValueError):
                pass
        self.store.sync_runs(self.task_store)
        return created
