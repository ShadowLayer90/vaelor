"""Versioned, reversible migration from legacy Pironman control-plane paths."""

from __future__ import annotations

import argparse
import json
import os
import shutil
import time
from pathlib import Path
from typing import Any, Dict


MIGRATION_VERSION = 1
CONFIRMATION = "migrate-control-plane-to-vaelor"
ROLLBACK_CONFIRMATION = "rollback-vaelor-migration"
PATH_MAPPINGS = (
    (
        "etc/credstore.encrypted/pironman-master-key",
        "etc/vaelor/credentials/master-key.cred",
    ),
    ("opt/pironman5/workloads", "var/lib/vaelor/workloads"),
    ("opt/pironman5/models", "var/lib/vaelor/models"),
    ("opt/pironman5/backups", "var/lib/vaelor/backups"),
    ("opt/pironman5/jobs", "var/lib/vaelor/jobs"),
    ("opt/pironman5/vnc", "var/lib/vaelor/vnc"),
    ("opt/pironman5/security.sqlite3", "var/lib/vaelor/security.sqlite3"),
    ("opt/pironman5/totp.key", "var/lib/vaelor/totp.key"),
    ("var/lib/pironman5/assistant", "var/lib/vaelor/assistant"),
    ("var/lib/pironman5/cluster", "var/lib/vaelor/cluster"),
    ("var/lib/pironman5/credentials", "var/lib/vaelor/credentials"),
    ("var/lib/pironman5/kvm", "var/lib/vaelor/kvm"),
    (
        "var/lib/pironman5/device-identity.json",
        "var/lib/vaelor/device-identity.json",
    ),
    (
        "var/lib/pironman5/system-update-state.json",
        "var/lib/vaelor/system-update-state.json",
    ),
    ("var/log/pironman5", "var/log/vaelor"),
)


class VaelorMigration:
    def __init__(self, root: str = "/"):
        self.root = Path(root).resolve()
        self.manifest = self.root / "var/lib/vaelor/migration-v1.json"

    def _path(self, relative: str) -> Path:
        relative_path = Path(relative)
        if relative_path.is_absolute() or ".." in relative_path.parts:
            raise ValueError("Migration paths must stay inside the selected root.")
        # Keep the final path lexical. Resolving it would follow a compatibility
        # symlink created by an earlier migration and make the Vaelor target look
        # like fresh legacy state during an idempotent repair.
        return self.root / relative_path

    @staticmethod
    def _occupied(path: Path) -> bool:
        if not path.exists():
            return False
        return path.is_file() or path.is_symlink() or any(path.iterdir())

    def inspect(self) -> Dict[str, Any]:
        items = []
        blocked = False
        for legacy_name, vaelor_name in PATH_MAPPINGS:
            legacy = self._path(legacy_name)
            vaelor = self._path(vaelor_name)
            source_exists = legacy.exists() and not legacy.is_symlink()
            destination_exists = self._occupied(vaelor)
            conflict = source_exists and destination_exists
            blocked = blocked or conflict
            items.append({
                "legacy": str(legacy),
                "vaelor": str(vaelor),
                "source_exists": source_exists,
                "destination_exists": destination_exists,
                "conflict": conflict,
            })
        return {
            "version": MIGRATION_VERSION,
            "blocked": blocked,
            "already_applied": self.manifest.is_file(),
            "items": items,
        }

    def apply(self, confirmation: str) -> Dict[str, Any]:
        if confirmation != CONFIRMATION:
            raise ValueError("Confirm the versioned Vaelor data migration.")
        if self.manifest.exists():
            raise ValueError("This Vaelor migration is already recorded.")
        plan = self.inspect()
        if plan["blocked"]:
            raise ValueError(
                "Legacy and Vaelor data both exist. Export or reconcile them first."
            )
        moved = []
        try:
            for item in plan["items"]:
                if not item["source_exists"]:
                    continue
                legacy = Path(item["legacy"])
                vaelor = Path(item["vaelor"])
                vaelor.parent.mkdir(parents=True, exist_ok=True)
                destination_was_empty = (
                    vaelor.is_dir()
                    and not vaelor.is_symlink()
                    and not any(vaelor.iterdir())
                )
                if destination_was_empty:
                    vaelor.rmdir()
                shutil.move(str(legacy), str(vaelor))
                legacy.parent.mkdir(parents=True, exist_ok=True)
                legacy.symlink_to(vaelor, target_is_directory=vaelor.is_dir())
                moved.append({
                    "legacy": str(legacy),
                    "vaelor": str(vaelor),
                    "destination_was_empty": destination_was_empty,
                })
            self.manifest.parent.mkdir(parents=True, exist_ok=True)
            record = {
                "version": MIGRATION_VERSION,
                "applied_at": int(time.time()),
                "moved": moved,
            }
            self.manifest.write_text(
                json.dumps(record, indent=2, sort_keys=True), encoding="utf-8"
            )
            os.chmod(self.manifest, 0o600)
            return record
        except Exception:
            self._reverse(moved)
            raise

    def rollback(self, confirmation: str) -> Dict[str, Any]:
        if confirmation != ROLLBACK_CONFIRMATION:
            raise ValueError("Confirm rollback of the Vaelor data migration.")
        try:
            record = json.loads(self.manifest.read_text("utf-8"))
        except (OSError, ValueError, json.JSONDecodeError) as error:
            raise ValueError("No valid Vaelor migration record was found.") from error
        moved = list(record.get("moved", []))
        self._reverse(moved)
        self.manifest.unlink(missing_ok=True)
        return {"rolled_back": True, "version": record.get("version"), "moved": moved}

    @staticmethod
    def _reverse(moved) -> None:
        for item in reversed(moved):
            legacy = Path(item["legacy"])
            vaelor = Path(item["vaelor"])
            if legacy.is_symlink():
                legacy.unlink()
            if vaelor.exists() and not legacy.exists():
                legacy.parent.mkdir(parents=True, exist_ok=True)
                shutil.move(str(vaelor), str(legacy))
            if item.get("destination_was_empty") and not vaelor.exists():
                vaelor.mkdir(parents=True, exist_ok=True)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Inspect or apply the Vaelor v1 compatibility migration."
    )
    parser.add_argument("--root", default="/")
    parser.add_argument("--apply", action="store_true")
    parser.add_argument("--rollback", action="store_true")
    parser.add_argument("--confirm", default="")
    args = parser.parse_args()
    if args.apply and args.rollback:
        parser.error("Choose apply or rollback, not both.")
    if (args.apply or args.rollback) and Path(args.root).resolve() == Path("/"):
        if hasattr(os, "geteuid") and os.geteuid() != 0:
            parser.error("Apply and rollback require root privileges.")
    migration = VaelorMigration(args.root)
    result = (
        migration.apply(args.confirm)
        if args.apply else
        migration.rollback(args.confirm)
        if args.rollback else
        migration.inspect()
    )
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
