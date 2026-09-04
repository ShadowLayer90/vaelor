"""Administrator routes for scheduled and off-site backups.

These live in their own module (registered by ``register_backup_routes``) so
``api_auth_routes`` stays under the production line ceiling. They build on the
existing portable-state primitive rather than reinventing it: an on-demand
backup drives the same ``PortableState.export`` the scheduler uses, and a
restore-from-list stages the archive into the existing portable-import review so
the destructive replacement still runs through ``appliance_recovery`` behind its
own typed confirmation.

Secrets never touch the config table. A passphrase or an off-site credential
supplied here is written straight into the credential broker and referenced only
by its purpose string; the config keeps the reference, never the secret.
"""

from __future__ import annotations

import shutil
import uuid
from pathlib import Path

from flask import g, request

from .api_common import ApiContext, payload as _payload
from .backup_schedule import (
    ARCHIVE_NOT_FOUND,
    BACKUP_OFFSITE_PURPOSE,
    BACKUP_PASSPHRASE_PURPOSE,
    SCHEDULED_ARCHIVE_GLOB,
    BackupScheduleError,
)
from .offsite_delivery import OffsiteError, normalize_offsite_config
from .portable_state import PortableStateError

_UNAVAILABLE = {
    "code": "backup_scheduler_unavailable",
    "message": "The backup scheduler is unavailable.",
}


def _archive_listing(scheduler) -> list[dict]:
    root = Path(scheduler.export_root)
    if not root.is_dir():
        return []
    items = []
    # Only scheduled archives, not the transient {uuid}.vaelor on-demand exports
    # that share this directory (see SCHEDULED_ARCHIVE_GLOB).
    for path in root.glob(SCHEDULED_ARCHIVE_GLOB):
        if not path.is_file() or path.name.endswith(".tmp"):
            continue
        try:
            stat = path.stat()
        except OSError:
            continue
        run = scheduler.store.latest_run_for(path.name)
        items.append({
            "name": path.name,
            "size_bytes": stat.st_size,
            "modified_at": int(stat.st_mtime),
            "offsite_status": (run or {}).get("offsite_status", "skipped"),
            "sha256": (run or {}).get("sha256", ""),
        })
    items.sort(key=lambda item: item["modified_at"], reverse=True)
    return items


def register_backup_routes(context: ApiContext) -> None:
    blueprint = context.blueprint
    callbacks = context.callbacks
    security = context.security
    require_auth = context.require_auth

    def _scheduler():
        return callbacks.get("backup_scheduler")

    @blueprint.get("/admin/backups")
    @require_auth("administrator")
    def backups_status():
        scheduler = _scheduler()
        if scheduler is None:
            return _payload(error=_UNAVAILABLE, status=503)
        config = scheduler.store.get_config()
        return _payload({
            "config": {
                "enabled": config["enabled"],
                "interval_seconds": config["interval_seconds"],
                "retention_keep": config["retention_keep"],
                "retention_max_age_seconds": config["retention_max_age_seconds"],
                "passphrase_configured": bool(config["passphrase_purpose"]),
                "next_run_at": config["next_run_at"],
                "offsite": config["offsite"],
            },
            "runs": scheduler.store.list_runs(limit=100),
            "archives": _archive_listing(scheduler),
        })

    @blueprint.put("/admin/backups/schedule")
    @require_auth("administrator", csrf=True)
    def backups_set_schedule():
        scheduler = _scheduler()
        if scheduler is None:
            return _payload(error=_UNAVAILABLE, status=503)
        body = request.get_json(silent=True) or {}
        try:
            config = scheduler.store.set_schedule(
                enabled=bool(body.get("enabled", False)),
                interval_seconds=int(body.get("interval_seconds", 86400)),
                retention_keep=int(body.get("retention_keep", 7)),
                retention_max_age_seconds=int(body.get("retention_max_age_seconds", 0)),
            )
        except (BackupScheduleError, TypeError, ValueError) as error:
            return _payload(
                error={"code": "backup_schedule_invalid", "message": str(error)},
                status=400,
            )
        security.audit(
            g.auth_session.username, "appliance.backup.schedule", "success",
            remote_addr=request.remote_addr or "",
        )
        return _payload({"config": config})

    @blueprint.post("/admin/backups/passphrase")
    @require_auth("administrator", csrf=True)
    def backups_set_passphrase():
        scheduler = _scheduler()
        broker = callbacks.get("credential_broker")
        if scheduler is None or broker is None:
            return _payload(error=_UNAVAILABLE, status=503)
        body = request.get_json(silent=True) or {}
        passphrase = str(body.get("passphrase", ""))
        if len(passphrase) < 16:
            return _payload(
                error={
                    "code": "backup_passphrase_weak",
                    "message": "Use a backup passphrase with at least 16 characters.",
                },
                status=400,
            )
        try:
            credential = broker.put(
                "application-secret", "Scheduled backup passphrase",
                passphrase, owner=g.auth_session.username,
            )
            broker.activate(credential["id"], BACKUP_PASSPHRASE_PURPOSE)
        except Exception as error:  # broker rejection is a safe, user-facing string
            return _payload(
                error={"code": "backup_passphrase_failed", "message": str(error)},
                status=400,
            )
        scheduler.store.set_passphrase_purpose(BACKUP_PASSPHRASE_PURPOSE)
        security.audit(
            g.auth_session.username, "appliance.backup.passphrase", "success",
            remote_addr=request.remote_addr or "",
        )
        return _payload({"passphrase_configured": True}, status=201)

    @blueprint.put("/admin/backups/offsite")
    @require_auth("administrator", csrf=True)
    def backups_set_offsite():
        scheduler = _scheduler()
        broker = callbacks.get("credential_broker")
        if scheduler is None:
            return _payload(error=_UNAVAILABLE, status=503)
        body = request.get_json(silent=True) or {}
        backend = str(body.get("backend", "")).strip().lower()
        if not backend:
            scheduler.store.set_offsite({})
            security.audit(
                g.auth_session.username, "appliance.backup.offsite", "success",
                remote_addr=request.remote_addr or "",
            )
            return _payload({"offsite": {}})
        secret = str(body.get("credentials", "")).strip()
        config = {
            "backend": backend,
            "endpoint": body.get("endpoint", ""),
            "bucket": body.get("bucket", ""),
            "prefix": body.get("prefix", ""),
            "region": body.get("region", ""),
            "credential_purpose": BACKUP_OFFSITE_PURPOSE,
        }
        try:
            normalize_offsite_config(config)
        except OffsiteError as error:
            return _payload(
                error={"code": "backup_offsite_invalid", "message": str(error)},
                status=400,
            )
        if secret:
            if broker is None:
                return _payload(error=_UNAVAILABLE, status=503)
            try:
                credential = broker.put(
                    "application-secret", "Scheduled backup off-site target",
                    secret, owner=g.auth_session.username,
                )
                broker.activate(credential["id"], BACKUP_OFFSITE_PURPOSE)
            except Exception as error:
                return _payload(
                    error={"code": "backup_offsite_credential_failed", "message": str(error)},
                    status=400,
                )
        stored = scheduler.store.set_offsite(config)
        security.audit(
            g.auth_session.username, "appliance.backup.offsite", "success",
            remote_addr=request.remote_addr or "",
        )
        return _payload({"offsite": stored["offsite"]})

    @blueprint.post("/admin/backups")
    @require_auth("administrator", csrf=True)
    def backups_run_now():
        scheduler = _scheduler()
        if scheduler is None:
            return _payload(error=_UNAVAILABLE, status=503)
        run = scheduler.run_backup(trigger="manual")
        security.audit(
            g.auth_session.username, "appliance.backup.run", run["status"],
            remote_addr=request.remote_addr or "",
        )
        if run["status"] != "ok":
            return _payload(
                error={
                    "code": "backup_run_failed",
                    "message": run.get("error") or "The backup did not complete.",
                    "run": run,
                },
                status=400,
            )
        return _payload({"run": run}, status=201)

    @blueprint.post("/admin/backups/<name>/offsite")
    @require_auth("administrator", csrf=True)
    def backups_push_offsite(name):
        scheduler = _scheduler()
        if scheduler is None:
            return _payload(error=_UNAVAILABLE, status=503)
        try:
            run = scheduler.push_offsite(name)
        except BackupScheduleError as error:
            return _payload(
                error={"code": "backup_offsite_push_failed", "message": str(error)},
                status=400,
            )
        security.audit(
            g.auth_session.username, "appliance.backup.offsite_push",
            run.get("offsite_status", ""), remote_addr=request.remote_addr or "",
        )
        return _payload({"run": run})

    @blueprint.post("/admin/backups/<name>/restore")
    @require_auth("administrator", csrf=True)
    def backups_restore(name):
        scheduler = _scheduler()
        plans = callbacks.get("portable_import_plans")
        if scheduler is None or plans is None:
            return _payload(error=_UNAVAILABLE, status=503)
        safe = Path(str(name)).name
        archive = Path(scheduler.export_root) / safe
        if not safe.endswith(".vaelor") or not archive.is_file():
            return _payload(
                error={"code": "backup_not_found", "message": ARCHIVE_NOT_FOUND},
                status=404,
            )
        body = request.get_json(silent=True) or {}
        passphrase = str(body.get("passphrase", ""))
        # Drive the existing portable-import review: copy the listed archive
        # into the import directory the recovery broker consumes, then stage it.
        # The destructive replacement still runs only after the typed
        # confirmation on POST /admin/portable-state/import.
        plans.import_root.mkdir(parents=True, exist_ok=True)
        staged_copy = plans.import_root / f"{uuid.uuid4().hex}.vaelor"
        try:
            shutil.copy2(archive, staged_copy)
            staged = plans.stage(g.auth_session.username, staged_copy, passphrase)
        except (OSError, PortableStateError, ValueError) as error:
            staged_copy.unlink(missing_ok=True)
            return _payload(
                error={"code": "backup_restore_invalid", "message": str(error)},
                status=400,
            )
        security.audit(
            g.auth_session.username, "appliance.backup.restore_stage", "success",
            target=staged["plan"]["id"] if staged.get("plan") else "",
            remote_addr=request.remote_addr or "",
        )
        return _payload(staged, status=201)
