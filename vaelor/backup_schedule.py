"""Scheduled, retained, optionally off-site backups of Vaelor state.

Backup/restore of appliance state already exists as
:class:`vaelor.portable_state.PortableState`. This module does *not* reinvent
the archive format; it drives ``PortableState.export`` on a timer, keeps a
bounded set of the resulting archives on disk, optionally pushes each finished
archive off the box through :mod:`vaelor.offsite_delivery`, and records every
run so an administrator can see what happened and restore from the list.

The supervision follows the house pattern used by the NPU/GPU autostart loops
and :class:`vaelor.automations.AutomationRunner`: a daemon thread runs
``reconcile()`` forever, and every seam a test needs to make that deterministic
— the clock, the sleep, the stop event, the exporter, the secret resolver, and
the off-site transport — is injected. A test therefore never sleeps, never
writes a real 512 MiB archive (it stubs ``export`` to a tiny file), and never
opens a socket.

The passphrase that encrypts a scheduled archive is **never** stored in the
config table. It is held in the credential broker under a purpose string and
resolved at run time through the injected ``secret_resolver``. Off-site
destination credentials are handled the same way: the config stores only the
endpoint/bucket/prefix and the credential *purpose*, never the secret.
"""

from __future__ import annotations

import hashlib
import json
import os
import sqlite3
import threading
import time
import uuid
from contextlib import closing
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional

from .offsite_delivery import deliver_archive, normalize_offsite_config
from .portable_state import PortableState, PortableStateError
from .runtime_paths import env_value, state_path

#: Host-bound operational config + run history. Deliberately NOT part of
#: ``PortableState`` scope (schedules and off-site endpoints belong to *this*
#: box, not to the state that moves between boxes) and erased by the factory
#: reset (see ``appliance_recovery.SQLITE_FILES``).
BACKUP_SCHEDULE_DB = Path(state_path("recovery/backups.sqlite3"))
#: Scheduled archives land beside the on-demand portable-state exports.
BACKUP_EXPORT_ROOT = Path(state_path("recovery/exports"))
#: Scheduled archives carry this prefix so retention and the archive listing can
#: tell them apart from the transient ``{uuid}.vaelor`` on-demand portable-state
#: exports that share this directory. Without it an aborted on-demand download's
#: orphan would be listed as a backup and, being newer, could evict a real one.
SCHEDULED_ARCHIVE_PREFIX = "vaelor-backup-"
SCHEDULED_ARCHIVE_GLOB = SCHEDULED_ARCHIVE_PREFIX + "*.vaelor"
#: Broker purposes. Referenced by the config; the secret itself lives only in
#: the credential vault.
BACKUP_PASSPHRASE_PURPOSE = "backup-passphrase"
BACKUP_OFFSITE_PURPOSE = "backup-offsite"

CONFIG_ID = "default"
MIN_INTERVAL_SECONDS = 900
MAX_INTERVAL_SECONDS = 30 * 86400
MIN_RETENTION_KEEP = 1
MAX_RETENTION_KEEP = 365
MAX_RETENTION_AGE_SECONDS = 365 * 86400
RUN_HISTORY_CAP = 500
BACKUP_SUPERVISE_INTERVAL_SECONDS = 300

TRIGGER_SCHEDULED = "scheduled"
TRIGGER_MANUAL = "manual"
STATUS_OK = "ok"
STATUS_ERROR = "error"
OFFSITE_SKIPPED = "skipped"
OFFSITE_OK = "ok"
OFFSITE_FAILED = "failed"


#: Shared so the on-demand push and the restore route speak with one voice.
ARCHIVE_NOT_FOUND = "That backup archive was not found."


class BackupScheduleError(ValueError):
    pass


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


class BackupScheduleStore:
    """SQLite store for the single schedule config row and run history."""

    def __init__(self, database_path: Optional[str] = None):
        self.database_path = database_path or env_value(
            "VAELOR_BACKUP_SCHEDULE_DB", "PM_BACKUP_SCHEDULE_DB",
            str(BACKUP_SCHEDULE_DB),
        )
        parent = os.path.dirname(self.database_path)
        if parent:
            os.makedirs(parent, mode=0o700, exist_ok=True)
        self._initialize()
        try:
            os.chmod(self.database_path, 0o600)
        except OSError:
            pass

    def _connect(self) -> sqlite3.Connection:
        connection = sqlite3.connect(self.database_path, timeout=15)
        connection.row_factory = sqlite3.Row
        return connection

    def _initialize(self) -> None:
        with closing(self._connect()) as connection:
            connection.executescript(
                """
                CREATE TABLE IF NOT EXISTS backup_config (
                    id TEXT PRIMARY KEY,
                    enabled INTEGER NOT NULL DEFAULT 0,
                    interval_seconds INTEGER NOT NULL,
                    retention_keep INTEGER NOT NULL,
                    retention_max_age_seconds INTEGER NOT NULL DEFAULT 0,
                    passphrase_purpose TEXT NOT NULL DEFAULT '',
                    offsite_config TEXT NOT NULL DEFAULT '',
                    next_run_at REAL,
                    updated_at REAL NOT NULL
                );
                CREATE TABLE IF NOT EXISTS backup_runs (
                    id TEXT PRIMARY KEY,
                    created_at REAL NOT NULL,
                    trigger TEXT NOT NULL,
                    archive_name TEXT NOT NULL DEFAULT '',
                    size_bytes INTEGER NOT NULL DEFAULT 0,
                    sha256 TEXT NOT NULL DEFAULT '',
                    status TEXT NOT NULL,
                    error TEXT NOT NULL DEFAULT '',
                    offsite_status TEXT NOT NULL DEFAULT 'skipped',
                    offsite_detail TEXT NOT NULL DEFAULT ''
                );
                CREATE INDEX IF NOT EXISTS idx_backup_runs_created
                ON backup_runs(created_at DESC);
                """
            )
            connection.commit()

    # -- configuration ---------------------------------------------------

    @staticmethod
    def _default_config() -> Dict[str, Any]:
        return {
            "enabled": False,
            "interval_seconds": 86400,
            "retention_keep": 7,
            "retention_max_age_seconds": 0,
            "passphrase_purpose": "",
            "offsite": {},
            "next_run_at": None,
        }

    def get_config(self) -> Dict[str, Any]:
        with closing(self._connect()) as connection:
            row = connection.execute(
                "SELECT * FROM backup_config WHERE id=?", (CONFIG_ID,)
            ).fetchone()
        if row is None:
            return self._default_config()
        raw = str(row["offsite_config"] or "").strip()
        try:
            offsite = normalize_offsite_config(json.loads(raw) if raw else {})
        except (ValueError, TypeError):
            offsite = {}
        return {
            "enabled": bool(row["enabled"]),
            "interval_seconds": int(row["interval_seconds"]),
            "retention_keep": int(row["retention_keep"]),
            "retention_max_age_seconds": int(row["retention_max_age_seconds"]),
            "passphrase_purpose": str(row["passphrase_purpose"] or ""),
            "offsite": offsite,
            "next_run_at": row["next_run_at"],
        }

    def set_schedule(
        self,
        *,
        enabled: bool,
        interval_seconds: int,
        retention_keep: int,
        retention_max_age_seconds: int = 0,
        passphrase_purpose: Optional[str] = None,
        now: Optional[float] = None,
    ) -> Dict[str, Any]:
        interval_seconds = int(interval_seconds)
        retention_keep = int(retention_keep)
        retention_max_age_seconds = int(retention_max_age_seconds)
        if not MIN_INTERVAL_SECONDS <= interval_seconds <= MAX_INTERVAL_SECONDS:
            raise BackupScheduleError(
                "Choose a backup interval between 15 minutes and 30 days."
            )
        if not MIN_RETENTION_KEEP <= retention_keep <= MAX_RETENTION_KEEP:
            raise BackupScheduleError("Keep between 1 and 365 backups.")
        if not 0 <= retention_max_age_seconds <= MAX_RETENTION_AGE_SECONDS:
            raise BackupScheduleError("A maximum age must be between zero and one year.")
        current = self.get_config()
        purpose = (
            current["passphrase_purpose"]
            if passphrase_purpose is None
            else str(passphrase_purpose).strip()[:80]
        )
        if enabled and not purpose:
            raise BackupScheduleError(
                "Set a backup passphrase before enabling scheduled backups."
            )
        moment = now if now is not None else time.time()
        # Enabling (or changing the interval while enabled) schedules the next
        # run one interval out; disabling clears it so a re-enable does not fire
        # a backlog of missed runs at once.
        next_run_at = moment + interval_seconds if enabled else None
        offsite_raw = self._encode_offsite(current["offsite"])
        with closing(self._connect()) as connection:
            connection.execute(
                """
                INSERT INTO backup_config
                    (id, enabled, interval_seconds, retention_keep,
                     retention_max_age_seconds, passphrase_purpose,
                     offsite_config, next_run_at, updated_at)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
                ON CONFLICT(id) DO UPDATE SET
                    enabled=excluded.enabled,
                    interval_seconds=excluded.interval_seconds,
                    retention_keep=excluded.retention_keep,
                    retention_max_age_seconds=excluded.retention_max_age_seconds,
                    passphrase_purpose=excluded.passphrase_purpose,
                    next_run_at=excluded.next_run_at,
                    updated_at=excluded.updated_at
                """,
                (
                    CONFIG_ID, int(bool(enabled)), interval_seconds, retention_keep,
                    retention_max_age_seconds, purpose, offsite_raw, next_run_at, moment,
                ),
            )
            connection.commit()
        return self.get_config()

    @staticmethod
    def _encode_offsite(offsite: Dict[str, Any]) -> str:
        clean = normalize_offsite_config(offsite)
        return json.dumps(clean, separators=(",", ":"), sort_keys=True) if clean else ""

    def set_offsite(self, offsite: Optional[Dict[str, Any]], now: Optional[float] = None) -> Dict[str, Any]:
        """Replace the off-site target. ``None``/empty clears it."""
        encoded = self._encode_offsite(offsite or {})
        moment = now if now is not None else time.time()
        # Ensure a config row exists so an off-site target can be set before a
        # schedule is; the defaults are a disabled daily schedule.
        current = self.get_config()
        with closing(self._connect()) as connection:
            connection.execute(
                """
                INSERT INTO backup_config
                    (id, enabled, interval_seconds, retention_keep,
                     retention_max_age_seconds, passphrase_purpose,
                     offsite_config, next_run_at, updated_at)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
                ON CONFLICT(id) DO UPDATE SET
                    offsite_config=excluded.offsite_config,
                    updated_at=excluded.updated_at
                """,
                (
                    CONFIG_ID, int(current["enabled"]), current["interval_seconds"],
                    current["retention_keep"], current["retention_max_age_seconds"],
                    current["passphrase_purpose"], encoded, current["next_run_at"], moment,
                ),
            )
            connection.commit()
        return self.get_config()

    def set_passphrase_purpose(self, purpose: str, now: Optional[float] = None) -> Dict[str, Any]:
        """Persist the broker purpose that resolves the backup passphrase.

        Kept separate from :meth:`set_schedule` so storing (or rotating) the
        passphrase never disturbs the schedule's ``next_run_at``.
        """
        clean = str(purpose or "").strip()[:80]
        current = self.get_config()
        moment = now if now is not None else time.time()
        with closing(self._connect()) as connection:
            connection.execute(
                """
                INSERT INTO backup_config
                    (id, enabled, interval_seconds, retention_keep,
                     retention_max_age_seconds, passphrase_purpose,
                     offsite_config, next_run_at, updated_at)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
                ON CONFLICT(id) DO UPDATE SET
                    passphrase_purpose=excluded.passphrase_purpose,
                    updated_at=excluded.updated_at
                """,
                (
                    CONFIG_ID, int(current["enabled"]), current["interval_seconds"],
                    current["retention_keep"], current["retention_max_age_seconds"],
                    clean, self._encode_offsite(current["offsite"]),
                    current["next_run_at"], moment,
                ),
            )
            connection.commit()
        return self.get_config()

    def set_next_run_at(self, value: Optional[float]) -> None:
        with closing(self._connect()) as connection:
            connection.execute(
                "UPDATE backup_config SET next_run_at=? WHERE id=?", (value, CONFIG_ID)
            )
            connection.commit()

    # -- run history -----------------------------------------------------

    def record_run(
        self,
        *,
        trigger: str,
        status: str,
        archive_name: str = "",
        size_bytes: int = 0,
        sha256: str = "",
        error: str = "",
        offsite_status: str = OFFSITE_SKIPPED,
        offsite_detail: str = "",
        now: Optional[float] = None,
    ) -> Dict[str, Any]:
        run_id = uuid.uuid4().hex
        moment = now if now is not None else time.time()
        with closing(self._connect()) as connection:
            connection.execute(
                """
                INSERT INTO backup_runs
                    (id, created_at, trigger, archive_name, size_bytes, sha256,
                     status, error, offsite_status, offsite_detail)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    run_id, moment, str(trigger)[:32], str(archive_name)[:200],
                    int(size_bytes), str(sha256)[:64], str(status)[:16],
                    str(error)[:500], str(offsite_status)[:16], str(offsite_detail)[:300],
                ),
            )
            connection.commit()
        return self.get_run(run_id)

    def set_offsite_result(self, run_id: str, status: str, detail: str) -> None:
        with closing(self._connect()) as connection:
            connection.execute(
                "UPDATE backup_runs SET offsite_status=?, offsite_detail=? WHERE id=?",
                (str(status)[:16], str(detail)[:300], run_id),
            )
            connection.commit()

    def get_run(self, run_id: str) -> Optional[Dict[str, Any]]:
        with closing(self._connect()) as connection:
            row = connection.execute(
                "SELECT * FROM backup_runs WHERE id=?", (run_id,)
            ).fetchone()
        return dict(row) if row else None

    def list_runs(self, limit: int = 100) -> List[Dict[str, Any]]:
        with closing(self._connect()) as connection:
            rows = connection.execute(
                "SELECT * FROM backup_runs ORDER BY created_at DESC LIMIT ?",
                (max(1, min(int(limit), RUN_HISTORY_CAP)),),
            ).fetchall()
        return [dict(row) for row in rows]

    def latest_run_for(self, archive_name: str) -> Optional[Dict[str, Any]]:
        with closing(self._connect()) as connection:
            row = connection.execute(
                "SELECT * FROM backup_runs WHERE archive_name=? "
                "ORDER BY created_at DESC LIMIT 1",
                (str(archive_name),),
            ).fetchone()
        return dict(row) if row else None

    def prune_runs(self, keep: int = RUN_HISTORY_CAP) -> int:
        with closing(self._connect()) as connection:
            cursor = connection.execute(
                "DELETE FROM backup_runs WHERE id NOT IN "
                "(SELECT id FROM backup_runs ORDER BY created_at DESC LIMIT ?)",
                (max(1, int(keep)),),
            )
            connection.commit()
        return cursor.rowcount


def apply_retention(
    export_root: Path,
    keep: int,
    max_age_seconds: int,
    *,
    now: Optional[float] = None,
) -> List[str]:
    """Delete scheduled archives beyond ``keep`` newest or older than the max age.

    Retention is a property of the archives on disk, so it reads the directory
    rather than the run table: a run row whose archive was already removed, and
    an archive with no run row, are both handled correctly. Zero archives is a
    normal state and returns an empty list, never an error. In-progress exports
    (``*.vaelor.tmp``) are ignored so a half-written archive is never counted or
    pruned.
    """
    root = Path(export_root)
    if not root.is_dir():
        return []
    moment = now if now is not None else time.time()
    archives = [
        path for path in root.glob(SCHEDULED_ARCHIVE_GLOB)
        if path.is_file() and not path.name.endswith(".tmp")
    ]
    archives.sort(key=lambda path: path.stat().st_mtime, reverse=True)
    doomed: List[Path] = []
    for index, path in enumerate(archives):
        too_many = index >= max(1, int(keep))
        too_old = (
            max_age_seconds > 0
            and (moment - path.stat().st_mtime) > max_age_seconds
        )
        if too_many or too_old:
            doomed.append(path)
    removed: List[str] = []
    for path in doomed:
        try:
            path.unlink()
            removed.append(path.name)
        except OSError:
            continue
    return removed


class BackupScheduler:
    """Runs the export, records it, delivers off-site, and prunes.

    ``secret_resolver`` maps a broker *purpose* to the plaintext secret for that
    purpose (in production, ``lambda p: broker.resolve_active(p)["token"]``).
    ``deliver`` is the off-site seam (default :func:`offsite_delivery.deliver_archive`).
    Neither the exporter, the resolver, nor the deliverer is allowed to crash the
    supervise loop: a failure at any stage is recorded on the run and the loop
    continues.
    """

    def __init__(
        self,
        store: BackupScheduleStore,
        portable_state: Optional[PortableState] = None,
        *,
        export_root: Optional[Path] = None,
        secret_resolver: Optional[Callable[[str], str]] = None,
        deliver: Callable[..., Dict[str, Any]] = deliver_archive,
        transport=None,
        clock: Callable[[], float] = time.time,
    ):
        self.store = store
        self.portable_state = portable_state or PortableState()
        self.export_root = Path(export_root) if export_root is not None else BACKUP_EXPORT_ROOT
        self.secret_resolver = secret_resolver
        self.deliver = deliver
        self.transport = transport
        self.clock = clock

    def _resolve(self, purpose: str) -> str:
        if not purpose:
            raise BackupScheduleError("No credential purpose is configured.")
        if self.secret_resolver is None:
            raise BackupScheduleError("The credential broker is unavailable.")
        secret = self.secret_resolver(purpose)
        if not secret:
            raise BackupScheduleError("No secret is active for this purpose.")
        return secret

    def reconcile(self, now: Optional[float] = None) -> Optional[Dict[str, Any]]:
        """One supervise pass. Never raises; returns the run it made, if any."""
        moment = now if now is not None else self.clock()
        try:
            config = self.store.get_config()
        except sqlite3.Error:
            return None
        if not config["enabled"]:
            return None
        next_run_at = config["next_run_at"]
        if next_run_at is None:
            # Enabled with no scheduled time (e.g. after a crash between the
            # config write and the first run): arm it one interval out.
            self.store.set_next_run_at(moment + config["interval_seconds"])
            return None
        if moment < next_run_at:
            return None
        try:
            return self.run_backup(
                now=moment, trigger=TRIGGER_SCHEDULED, config=config
            )
        finally:
            # Advance the clock even if the run raised (run_backup aims never to,
            # but a store write could still fail), so a transient failure cannot
            # make every supervise pass re-fire the same due backup. A still-
            # unwritable store simply advances on the next pass.
            try:
                self.store.set_next_run_at(moment + config["interval_seconds"])
            except Exception:
                pass

    def run_backup(
        self,
        now: Optional[float] = None,
        *,
        trigger: str = TRIGGER_MANUAL,
        config: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        """Export once, record it, deliver off-site, prune. Never raises."""
        moment = now if now is not None else self.clock()
        if config is None:
            try:
                config = self.store.get_config()
            except Exception as error:
                # The backup config DB is unreadable (a lock). Nothing can be
                # recorded because the store that would record it is the one that
                # failed, so return a synthetic error result rather than raising.
                return {
                    "status": STATUS_ERROR, "trigger": trigger,
                    "error": f"config: {error}"[:500], "recorded": False,
                }
        try:
            passphrase = self._resolve(config["passphrase_purpose"])
        except Exception as error:  # broker/resolver failure must not crash the loop
            return self.store.record_run(
                trigger=trigger, status=STATUS_ERROR,
                error=f"passphrase: {error}"[:500], now=moment,
            )
        self.export_root.mkdir(parents=True, exist_ok=True)
        stamp = time.strftime("%Y%m%d-%H%M%S", time.gmtime(moment))
        archive = self.export_root / (
            f"{SCHEDULED_ARCHIVE_PREFIX}{stamp}-{uuid.uuid4().hex[:8]}.vaelor"
        )
        try:
            self.portable_state.export(archive, passphrase)
        except (OSError, PortableStateError, ValueError, sqlite3.Error) as error:
            # sqlite3.Error included: export snapshots the live state databases,
            # so a concurrent-writer lock surfaces here and must be recorded, not
            # raised into the loop.
            archive.unlink(missing_ok=True)
            return self.store.record_run(
                trigger=trigger, status=STATUS_ERROR,
                error=f"export: {error}"[:500], now=moment,
            )
        try:
            size = archive.stat().st_size
            sha = _sha256_file(archive)
        except OSError as error:
            return self.store.record_run(
                trigger=trigger, status=STATUS_ERROR,
                error=f"verify: {error}"[:500], now=moment,
            )
        offsite_status, offsite_detail = self._deliver(config, archive, moment)
        run = self.store.record_run(
            trigger=trigger, status=STATUS_OK, archive_name=archive.name,
            size_bytes=size, sha256=sha, offsite_status=offsite_status,
            offsite_detail=offsite_detail, now=moment,
        )
        # Retention runs only after a successful archive is recorded, so a keep
        # of N never prunes below N by counting an archive that just failed.
        try:
            apply_retention(
                self.export_root, config["retention_keep"],
                config["retention_max_age_seconds"], now=moment,
            )
            self.store.prune_runs()
        except OSError:
            pass
        return run

    def _deliver(
        self, config: Dict[str, Any], archive: Path, moment: float
    ) -> tuple[str, str]:
        offsite = config.get("offsite") or {}
        if not offsite:
            return OFFSITE_SKIPPED, ""
        try:
            credentials = self._resolve(offsite.get("credential_purpose", ""))
        except Exception as error:  # resolver/broker failure is a delivery failure
            return OFFSITE_FAILED, f"credential: {error}"[:300]
        try:
            kwargs: Dict[str, Any] = {"now": moment}
            if self.transport is not None:
                kwargs["transport"] = self.transport
            result = self.deliver(archive, offsite, credentials, **kwargs)
        except Exception as error:  # a backend must not crash the loop
            return OFFSITE_FAILED, f"delivery: {error}"[:300]
        # A failed push deliberately leaves the local archive in place.
        return (
            OFFSITE_OK if result.get("ok") else OFFSITE_FAILED,
            str(result.get("detail", ""))[:300],
        )

    def push_offsite(self, archive_name: str, now: Optional[float] = None) -> Dict[str, Any]:
        """Push an already-existing archive off-site on demand.

        Records the outcome against the archive's most recent run row, or a new
        row if the archive has none, and never removes the local archive.
        """
        moment = now if now is not None else self.clock()
        safe = Path(str(archive_name)).name
        archive = self.export_root / safe
        if not safe.endswith(".vaelor") or not archive.is_file():
            raise BackupScheduleError(ARCHIVE_NOT_FOUND)
        config = self.store.get_config()
        if not config.get("offsite"):
            raise BackupScheduleError("Configure an off-site target first.")
        status, detail = self._deliver(config, archive, moment)
        run = self.store.latest_run_for(safe)
        if run is None:
            run = self.store.record_run(
                trigger="offsite", status=STATUS_OK, archive_name=safe,
                size_bytes=archive.stat().st_size, offsite_status=status,
                offsite_detail=detail, now=moment,
            )
        else:
            self.store.set_offsite_result(run["id"], status, detail)
            run = self.store.get_run(run["id"])
        return run


def launch_backup_autostart(
    scheduler: BackupScheduler,
    *,
    interval_seconds: float = BACKUP_SUPERVISE_INTERVAL_SECONDS,
    sleep: Callable[[float], None] = time.sleep,
    stop: Optional[threading.Event] = None,
) -> threading.Thread:
    """Supervise the schedule off the request path, boot and failure alike.

    Mirrors :func:`vaelor.executor_service.launch_gpu_chat_autostart`: the first
    pass after boot catches up a due backup, and every ``interval_seconds`` pass
    after is the failure watch. ``reconcile`` is idempotent and never raises, so
    one bad pass can neither kill the thread nor stop the next. ``sleep`` and
    ``stop`` are injected seams: production wires neither and the loop runs
    forever; a test passes a fake ``sleep`` and a :class:`threading.Event` to run
    a bounded number of passes.
    """
    def _supervise() -> None:
        while stop is None or not stop.is_set():
            try:
                scheduler.reconcile()
            except Exception:
                # Belt-and-braces: reconcile is built not to raise, but a bad
                # pass (e.g. a transient DB error) must never kill the thread or
                # stop the next pass.
                pass
            if stop is not None and stop.is_set():
                return
            sleep(interval_seconds)

    thread = threading.Thread(
        target=_supervise, name="vaelor-backup-autostart", daemon=True
    )
    thread.start()
    return thread
