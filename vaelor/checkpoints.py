"""Read-only inventory of verified workload recovery checkpoints."""

from __future__ import annotations

import hashlib
import json
import re
import shutil
import tarfile
from hmac import compare_digest
from pathlib import Path, PurePosixPath
from typing import Any

from .runtime_paths import data_path
from .workload_broker import WorkloadBrokerClient


_CHECKPOINT_PATTERN = re.compile(
    r"(?P<project>[a-z0-9][a-z0-9_-]{1,47})-(?P<created_at_ms>[0-9]+)\.tar\.gz"
)
_PROJECT_PATTERN = re.compile(r"[a-z0-9][a-z0-9_-]{1,47}")
_DIGEST_PATTERN = re.compile(r"[0-9a-f]{64}")
_MAX_MEMBERS = 5000
_MAX_LINEAGE_BYTES = 64 * 1024
_LINEAGE_MEMBER = ".vaelor-checkpoint.json"
_LINEAGE_SCHEMA = "vaelor.checkpoint.lineage.v1"
LINEAGE_MEMBER = _LINEAGE_MEMBER
LINEAGE_SCHEMA = _LINEAGE_SCHEMA


class CheckpointInventory:
    """Discover and bind only recovery archives that are valid on disk.

    Checkpoint filenames are lookup hints written by the executor:
    <project>-<created-at-milliseconds>.tar.gz. The archive is authoritative:
    it must contain one project root, a Compose file, and canonical lineage
    metadata whose project and creation timestamp match the filename. The
    inventory computes a content-bound manifest and digest from the bytes
    currently on disk every time a checkpoint is exposed or bound for restore.
    """

    LINEAGE_MEMBER = LINEAGE_MEMBER
    LINEAGE_SCHEMA = LINEAGE_SCHEMA

    def __init__(self, backup_root: str | None = None):
        configured_root = backup_root or data_path("backups/workloads")
        self.backup_root = Path(configured_root).resolve()
        self.broker = WorkloadBrokerClient() if backup_root is None else None

    @staticmethod
    def _parse_checkpoint(checkpoint: str) -> re.Match[str]:
        match = _CHECKPOINT_PATTERN.fullmatch(checkpoint)
        if match is None:
            raise ValueError("Choose a valid managed checkpoint.")
        return match

    def _archive_path(self, checkpoint: str) -> tuple[Path, re.Match[str]]:
        match = self._parse_checkpoint(checkpoint)
        path = (self.backup_root / checkpoint).resolve()
        if path.parent != self.backup_root or not path.is_file():
            raise ValueError("Checkpoint was not found.")
        return path, match

    @staticmethod
    def _manifest_digest(entries: list[dict[str, Any]]) -> str:
        encoded = json.dumps(
            entries,
            ensure_ascii=True,
            sort_keys=True,
            separators=(",", ":"),
        ).encode("utf-8")
        return hashlib.sha256(encoded).hexdigest()

    @staticmethod
    def _member_path(name: str, project: str) -> str:
        if not name or name.startswith(("/", "\\")) or "\\" in name:
            raise ValueError("Checkpoint contains an unsafe path.")
        path = PurePosixPath(name)
        parts = path.parts
        if not parts or any(part in {"", ".", ".."} for part in parts):
            raise ValueError("Checkpoint contains an unsafe path.")
        if parts[0] != project:
            raise ValueError("Checkpoint belongs to a different project.")
        return "/".join(parts)

    @staticmethod
    def canonical_lineage(metadata: dict[str, Any]) -> bytes:
        return CheckpointInventory._canonical_lineage(metadata)

    @staticmethod
    def _member_digest(
        archive: tarfile.TarFile,
        member: tarfile.TarInfo,
        capture: bool = False,
    ) -> tuple[str, bytes | None]:
        extracted = archive.extractfile(member)
        if extracted is None:
            raise ValueError("Checkpoint file contents could not be read.")
        digest = hashlib.sha256()
        captured = bytearray() if capture else None
        while True:
            chunk = extracted.read(4 * 1024 * 1024)
            if not chunk:
                break
            digest.update(chunk)
            if captured is not None:
                if len(captured) + len(chunk) > _MAX_LINEAGE_BYTES:
                    raise ValueError("Checkpoint lineage metadata is too large.")
                captured.extend(chunk)
        return digest.hexdigest(), bytes(captured) if captured is not None else None

    @staticmethod
    def _canonical_lineage(metadata: dict[str, Any]) -> bytes:
        return (
            json.dumps(
                metadata,
                ensure_ascii=True,
                sort_keys=True,
                separators=(",", ":"),
            )
            + "\n"
        ).encode("utf-8")

    @classmethod
    def _validate_lineage(
        cls,
        raw: bytes,
        project: str,
        expected_created_at_ms: int,
    ) -> dict[str, Any]:
        try:
            metadata = json.loads(raw.decode("utf-8"))
        except (UnicodeDecodeError, json.JSONDecodeError) as error:
            raise ValueError("Checkpoint lineage metadata is invalid.") from error
        if not isinstance(metadata, dict):
            raise ValueError("Checkpoint lineage metadata is invalid.")
        if raw != cls._canonical_lineage(metadata):
            raise ValueError("Checkpoint lineage metadata is not canonical.")
        if metadata.get("schema") != _LINEAGE_SCHEMA:
            raise ValueError("Checkpoint lineage metadata is stale or unsupported.")
        if metadata.get("project") != project:
            raise ValueError("Checkpoint lineage project does not match the archive.")
        created_at_ms = metadata.get("created_at_ms")
        if (
            isinstance(created_at_ms, bool)
            or not isinstance(created_at_ms, int)
            or created_at_ms <= 0
            or created_at_ms != expected_created_at_ms
        ):
            raise ValueError("Checkpoint creation timestamp does not match its archive.")
        for key in ("creator", "created_by", "actor"):
            value = metadata.get(key)
            if not isinstance(value, str) or not value.strip() or len(value) > 200:
                raise ValueError("Checkpoint lineage creator metadata is incomplete.")
        job_id = metadata.get("job_id")
        creating_job_id = metadata.get("creating_job_id")
        if job_id != creating_job_id:
            raise ValueError("Checkpoint creating-job lineage does not match.")
        if job_id is not None and (
            not isinstance(job_id, str) or not job_id.strip() or len(job_id) > 200
        ):
            raise ValueError("Checkpoint creating-job lineage is invalid.")
        return metadata

    def _inspect_archive(
        self, path: Path, expected_project: str, expected_created_at_ms: int
    ) -> tuple[list[dict[str, Any]], int, dict[str, Any]]:
        entries: list[dict[str, Any]] = []
        names: set[str] = set()
        total_bytes = 0
        has_compose = False
        lineage_bytes: bytes | None = None
        lineage_member_name = ""
        try:
            with tarfile.open(path, mode="r:gz") as archive:
                members = archive.getmembers()
                if not members or len(members) > _MAX_MEMBERS:
                    raise ValueError("Checkpoint contents are invalid.")
                top_levels = set()
                for member in members:
                    raw_parts = PurePosixPath(member.name).parts
                    if not raw_parts:
                        raise ValueError("Checkpoint contains an unsafe path.")
                    top_levels.add(raw_parts[0])
                if len(top_levels) != 1:
                    raise ValueError("Checkpoint must contain one project root.")
                project = next(iter(top_levels))
                if not _PROJECT_PATTERN.fullmatch(project):
                    raise ValueError("Checkpoint contains an invalid project root.")
                if project != expected_project:
                    raise ValueError("Checkpoint belongs to a different project.")
                for member in members:
                    name = self._member_path(member.name, project)
                    if name in names:
                        raise ValueError("Checkpoint contains duplicate paths.")
                    names.add(name)
                    if member.isdir():
                        entry_type = "directory"
                        size = 0
                        entry = {"path": name, "type": entry_type, "size_bytes": size}
                    elif member.isfile():
                        entry_type = "file"
                        size = int(member.size)
                        if size < 0:
                            raise ValueError("Checkpoint contains an invalid file size.")
                        total_bytes += size
                        is_lineage = name == f"{project}/{_LINEAGE_MEMBER}"
                        member_digest, captured = self._member_digest(
                            archive, member, is_lineage
                        )
                        if is_lineage:
                            lineage_bytes = captured
                            lineage_member_name = name
                        if name == f"{project}/compose.yaml":
                            has_compose = True
                        entry = {
                            "path": name,
                            "type": entry_type,
                            "size_bytes": size,
                            "sha256": member_digest,
                        }
                    else:
                        raise ValueError("Checkpoint contains an unsupported file type.")
                    entries.append(entry)
        except (OSError, tarfile.TarError, EOFError) as error:
            raise ValueError("Checkpoint archive could not be read.") from error
        if not has_compose:
            raise ValueError("Checkpoint does not contain compose.yaml.")
        if lineage_bytes is None or lineage_member_name != f"{expected_project}/{_LINEAGE_MEMBER}":
            raise ValueError("Checkpoint lineage metadata is missing.")
        lineage = self._validate_lineage(
            lineage_bytes, expected_project, expected_created_at_ms
        )
        entries.sort(key=lambda entry: str(entry["path"]))
        return entries, total_bytes, lineage

    @staticmethod
    def _sha256(path: Path) -> str:
        digest = hashlib.sha256()
        try:
            with path.open("rb") as source:
                for chunk in iter(lambda: source.read(4 * 1024 * 1024), b""):
                    digest.update(chunk)
        except OSError as error:
            raise ValueError("Checkpoint could not be read from disk.") from error
        return digest.hexdigest()

    def _verified_checkpoint(self, checkpoint: str) -> dict[str, Any]:
        path, match = self._archive_path(checkpoint)
        project = match.group("project")
        try:
            stat_before = path.stat()
        except OSError as error:
            raise ValueError("Checkpoint was not found.") from error
        if stat_before.st_size <= 0:
            raise ValueError("Checkpoint has no recoverable archive data.")
        created_at_ms = int(match.group("created_at_ms"))
        entries, content_bytes, lineage = self._inspect_archive(
            path, project, created_at_ms
        )
        digest = self._sha256(path)
        try:
            stat_after = path.stat()
        except OSError as error:
            raise ValueError("Checkpoint changed while it was being verified.") from error
        if (
            stat_before.st_size != stat_after.st_size
            or stat_before.st_mtime_ns != stat_after.st_mtime_ns
        ):
            raise ValueError("Checkpoint changed while it was being verified.")
        manifest_digest = self._manifest_digest(entries)
        preflight = {
            "state": "archive_verified",
            "structural_valid": True,
            "compose_config": "not_run",
            "startup": "unproven",
            "startup_proven": False,
            "message": "Archive bytes and Compose presence are verified; startup is not proven.",
        }
        creation_lineage = dict(lineage)
        creation_lineage["archive"] = checkpoint
        return {
            "id": checkpoint,
            "project": project,
            "owner": project,
            "created_at": created_at_ms / 1000,
            "created_at_ms": created_at_ms,
            "creation_lineage": creation_lineage,
            "artifact_path": checkpoint,
            "size_bytes": stat_after.st_size,
            "content_bytes": content_bytes,
            "sha256": digest,
            "manifest_digest": manifest_digest,
            "manifest_entries": len(entries),
            "verified": True,
            "restorable": True,
            "restorable_state": "preflight_ready",
            "preflight": preflight,
            "startup_proven": False,
        }

    def list(self, limit: int = 100):
        try:
            candidates = sorted(
                self.backup_root.glob("*.tar.gz"),
                key=lambda path: path.stat().st_mtime,
                reverse=True,
            )
        except OSError:
            candidates = []
        output = []
        for path in candidates:
            try:
                record = self._verified_checkpoint(path.name)
            except (OSError, ValueError, tarfile.TarError):
                continue
            output.append(record)
            if len(output) >= max(1, min(int(limit), 200)):
                break
        return output

    def verify(self, checkpoint: str):
        return self._verified_checkpoint(checkpoint)

    def checksum(self, checkpoint: str):
        """Compatibility alias for the API's existing verification endpoint."""

        return self.verify(checkpoint)

    def bind_restore(
        self, checkpoint: str, project: str, sha256: str
    ) -> dict[str, Any]:
        if not _PROJECT_PATTERN.fullmatch(str(project)):
            raise ValueError("Choose a managed project.")
        if not _DIGEST_PATTERN.fullmatch(str(sha256)):
            raise ValueError("Restore requires the verified checkpoint digest.")
        record = self.verify(checkpoint)
        if record["project"] != project:
            raise ValueError("Checkpoint belongs to a different project.")
        if not compare_digest(record["sha256"], sha256):
            raise ValueError("Checkpoint changed since it was verified.")
        return {
            "checkpoint": record["id"],
            "project": record["project"],
            "sha256": record["sha256"],
            "manifest_digest": record["manifest_digest"],
            "verified": True,
            "restorable": True,
            "restorable_state": record["restorable_state"],
            "preflight": record["preflight"],
            "startup_proven": record["startup_proven"],
            "creation_lineage": record["creation_lineage"],
        }

    def delete(self, checkpoint: str, confirmation: str):
        record = self.verify(checkpoint)
        project = record["project"]
        if confirmation != project:
            raise ValueError("Type the project name to confirm checkpoint deletion.")
        if self.broker is not None:
            return self.broker.delete_checkpoint(checkpoint, confirmation)
        path, _ = self._archive_path(checkpoint)
        recovery_copy = path.with_suffix("").with_suffix(".tar.project")
        path.unlink()
        if recovery_copy.is_file():
            recovery_copy.unlink()
        elif recovery_copy.is_dir():
            shutil.rmtree(recovery_copy)
        return {
            "id": checkpoint,
            "project": project,
            "sha256": record["sha256"],
            "deleted": True,
        }
