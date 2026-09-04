"""Verified controller-side inventory for worker-local cluster volume backups."""

from __future__ import annotations

import hashlib
import json
import os
import re
import threading
import time
import uuid
from pathlib import Path
from typing import Any, Dict, Optional

from .runtime_paths import data_path


BACKUP_ID = re.compile(r"cluster-[a-f0-9]{16}")


class ClusterBackupStore:
    def __init__(self, root: Optional[str] = None):
        self.root = Path(
            root or data_path("backups/cluster")
        ).resolve()
        self._lock = threading.RLock()

    def _ensure_root(self) -> None:
        self.root.mkdir(parents=True, exist_ok=True)
        try:
            os.chmod(self.root, 0o750)
        except PermissionError:
            pass

    def archive_path(self, backup_id: str) -> Path:
        clean = str(backup_id).strip()
        if not BACKUP_ID.fullmatch(clean):
            raise ValueError("Choose a valid cluster backup.")
        path = (self.root / f"{clean}.tar.gz").resolve()
        if self.root not in path.parents:
            raise ValueError("Choose a valid cluster backup.")
        return path

    def _metadata_path(self, backup_id: str) -> Path:
        self.archive_path(backup_id)
        return (self.root / f"{backup_id}.json").resolve()

    @staticmethod
    def checksum(path: Path) -> str:
        digest = hashlib.sha256()
        with path.open("rb") as stream:
            while chunk := stream.read(1024 * 1024):
                digest.update(chunk)
        return digest.hexdigest()

    def record(
        self,
        *,
        service_name: str,
        node_id: str,
        volume: str,
        staged_archive: Path,
        reason: str = "manual",
    ) -> Dict[str, Any]:
        if not staged_archive.is_file():
            raise ValueError("The staged cluster backup is unavailable.")
        self._ensure_root()
        backup_id = f"cluster-{uuid.uuid4().hex[:16]}"
        destination = self.archive_path(backup_id)
        temporary = destination.with_suffix(".tar.gz.next")
        with self._lock:
            staged_archive.replace(temporary)
            os.chmod(temporary, 0o640)
            temporary.replace(destination)
            metadata = {
                "id": backup_id,
                "service_name": str(service_name)[:80],
                "node_id": str(node_id)[:80],
                "volume": str(volume)[:160],
                "size_bytes": destination.stat().st_size,
                "sha256": self.checksum(destination),
                "reason": str(reason)[:40],
                "created_at": int(time.time()),
            }
            metadata_path = self._metadata_path(backup_id)
            metadata_next = metadata_path.with_suffix(".json.next")
            metadata_next.write_text(
                json.dumps(metadata, separators=(",", ":"), sort_keys=True),
                encoding="utf-8",
            )
            os.chmod(metadata_next, 0o640)
            metadata_next.replace(metadata_path)
        return metadata

    def get(self, backup_id: str, *, verify: bool = False) -> Dict[str, Any]:
        metadata_path = self._metadata_path(backup_id)
        archive = self.archive_path(backup_id)
        if not metadata_path.is_file() or not archive.is_file():
            raise ValueError("The cluster backup was not found.")
        try:
            value = json.loads(metadata_path.read_text(encoding="utf-8"))
        except (json.JSONDecodeError, OSError) as error:
            raise ValueError("The cluster backup metadata is invalid.") from error
        if not isinstance(value, dict) or value.get("id") != backup_id:
            raise ValueError("The cluster backup metadata is invalid.")
        result = {**value, "archive_path": str(archive)}
        if verify:
            actual = self.checksum(archive)
            if actual != value.get("sha256"):
                raise ValueError("The cluster backup checksum does not match.")
            result["verified"] = True
        return result

    def list(self, *, service_name: str = "", limit: int = 100):
        self._ensure_root()
        values = []
        for path in self.root.glob("cluster-*.json"):
            backup_id = path.stem
            try:
                item = self.get(backup_id)
            except ValueError:
                continue
            if service_name and item.get("service_name") != service_name:
                continue
            item.pop("archive_path", None)
            values.append(item)
        values.sort(key=lambda item: item.get("created_at", 0), reverse=True)
        return values[:max(1, min(int(limit), 500))]

    def delete(self, backup_id: str, *, confirmation: str) -> Dict[str, Any]:
        item = self.get(backup_id)
        if confirmation != item["service_name"]:
            raise ValueError("Type the service name to delete this backup.")
        with self._lock:
            self.archive_path(backup_id).unlink(missing_ok=True)
            self._metadata_path(backup_id).unlink(missing_ok=True)
        return {"id": backup_id, "deleted": True}
