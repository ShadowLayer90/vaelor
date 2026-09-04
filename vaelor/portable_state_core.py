"""Encrypted, versioned transfer of portable Vaelor control-plane state."""

from __future__ import annotations

import base64
import hashlib
import io
import json
import os
import shutil
import sqlite3
import tempfile
import time
import uuid
import zipfile
from contextlib import closing
from pathlib import Path, PurePosixPath
from typing import Any

from cryptography.exceptions import InvalidTag
from cryptography.hazmat.primitives.ciphers.aead import AESGCM
from cryptography.hazmat.primitives.kdf.scrypt import Scrypt

from .runtime_paths import STATE_ROOT
from .version import __version__


MAGIC = b"VAELOR-PORTABLE\x00\x01"
SCHEMA = 1
MAX_ARCHIVE_BYTES = 512 * 1024 * 1024
MAX_TRANSFER_FILES = 2048
MIN_PASSPHRASE_LENGTH = 16
DATABASES = (
    "security.sqlite3",
    "jobs/jobs.sqlite3",
    "assistant/assistant.sqlite3",
    "assistant/tasks.sqlite3",
    "assistant/skills.sqlite3",
    "assistant/automations.sqlite3",
    "assistant/custom-agents.sqlite3",
    "assistant/rag-chat.sqlite3",
    "integrations/app-capability-registry.sqlite3",
    "assistant/integrations.sqlite3",
)
FILES = (
    "system-update-state.json",
    "totp.key",
)
WORKLOAD_FILENAMES = {
    "compose.yaml",
    "compose.yml",
    "docker-compose.yaml",
    "docker-compose.yml",
    "metadata.json",
}
EXCLUDED = (
    "AI/provider credentials and SSH enrollment secrets",
    "external Assistant and inference API tokens",
    "credential vaults and plaintext/provider secret values (never exported)",
    "host-bound integration credential references and live connection test evidence (sanitized to a disconnected state)",
    "active login, CSRF, VNC, and KVM sessions",
    "TLS private keys and host-bound encrypted credential keys",
    "models, container volumes, backups, and hardware identity",
)


class PortableStateError(ValueError):
    pass


def _derive_key(passphrase: str, salt: bytes) -> bytes:
    if len(passphrase) < MIN_PASSPHRASE_LENGTH:
        raise PortableStateError(
            f"Use a transfer passphrase with at least {MIN_PASSPHRASE_LENGTH} characters."
        )
    return Scrypt(salt=salt, length=32, n=2**15, r=8, p=1).derive(
        passphrase.encode("utf-8")
    )


def _sha256(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def _safe_relative(value: str) -> Path:
    candidate = PurePosixPath(value)
    if candidate.is_absolute() or ".." in candidate.parts or not candidate.parts:
        raise PortableStateError("The transfer archive contains an unsafe path.")
    return Path(*candidate.parts)


class PortableState:
    def __init__(self, root: Path | str | None = None):
        self.root = Path(root) if root is not None else STATE_ROOT

    @staticmethod
    def scope() -> dict[str, Any]:
        return {
            "schema": SCHEMA,
            "includes": [
                "users, password hashes, MFA configuration, and audit history",
                "Assistant chats, memories, agents, skills, tasks, and automations",
                "AI Chat conversations, preferences, and RAG knowledge",
                "installed app capability manifests, stable app registrations, dependencies, version-pinned agent grants, connection metadata, and invocation audit",
                "deployment job history and portable workload definitions",
                "staged operating-system update metadata",
            ],
            "excludes": list(EXCLUDED),
        }

    @staticmethod
    def _sanitize_database(path: Path, relative: str) -> None:
        """Make imported state safe for a different host before it is used.

        The credential vault is outside ``DATABASES``. The integration database
        contains only references, but those references point into a host-bound
        vault and therefore cannot be restored as healthy credentials. The app
        registry also contains runtime observations from the source host; keep
        the durable manifest and identity while requiring local reconciliation.
        This is intentionally idempotent so it protects older archives too.
        """
        if relative not in {
            "assistant/integrations.sqlite3",
            "integrations/app-capability-registry.sqlite3",
        } or not path.is_file():
            return
        with closing(sqlite3.connect(path)) as database:
            database.execute("PRAGMA busy_timeout=15000")
            tables = {
                row[0]
                for row in database.execute(
                    "SELECT name FROM sqlite_master WHERE type='table'"
                ).fetchall()
            }
            if relative == "assistant/integrations.sqlite3":
                if "integration_connections" in tables:
                    columns = {
                        row[1]
                        for row in database.execute(
                            "PRAGMA table_info(integration_connections)"
                        ).fetchall()
                    }
                    required = {
                        "id", "credential_ref", "test_status",
                        "last_test_at", "last_test_detail",
                    }
                    if required <= columns:
                        rows = database.execute(
                            "SELECT id FROM integration_connections"
                        ).fetchall()
                        for (connection_id,) in rows:
                            marker = hashlib.sha256(
                                str(connection_id).encode("utf-8")
                            ).hexdigest()[:24]
                            database.execute(
                                "UPDATE integration_connections SET "
                                "credential_ref=?, test_status='pending', "
                                "last_test_at=NULL, last_test_detail='' WHERE id=?",
                                (f"cred_transfer_disabled_{marker}", connection_id),
                            )
            elif "app_instances" in tables:
                columns = {
                    row[1]
                    for row in database.execute(
                        "PRAGMA table_info(app_instances)"
                    ).fetchall()
                }
                assignments: list[str] = []
                values: list[Any] = []
                for column, value in (
                    ("state", "stopped"),
                    ("health", "unknown"),
                    ("compatibility", "unknown"),
                    ("reason", "Imported state requires local workload reconciliation."),
                    ("health_evidence_json", "{}"),
                    ("last_seen_at", None),
                ):
                    if column in columns:
                        assignments.append(f"{column}=?")
                        values.append(value)
                if assignments:
                    database.execute(
                        "UPDATE app_instances SET " + ", ".join(assignments),
                        values,
                    )
            database.commit()

    @classmethod
    def _database_snapshot(
        cls, source: Path, destination: Path, relative: str
    ) -> None:
        destination.parent.mkdir(parents=True, exist_ok=True)
        with closing(sqlite3.connect(
            f"file:{source.as_posix()}?mode=ro", uri=True, timeout=20
        )) as incoming:
            with closing(sqlite3.connect(destination)) as outgoing:
                incoming.backup(outgoing)
        if source.name == "security.sqlite3":
            with closing(sqlite3.connect(destination)) as database:
                database.execute("DELETE FROM sessions")
                database.commit()
        cls._sanitize_database(destination, relative)

    def _workload_files(self) -> list[Path]:
        root = self.root / "workloads"
        if not root.is_dir():
            return []
        return [
            path
            for path in sorted(root.rglob("*"))
            if path.is_file()
            and not path.is_symlink()
            and path.name in WORKLOAD_FILENAMES
            and path.stat().st_size <= 2 * 1024 * 1024
        ]

    def export(self, output: Path | str, passphrase: str) -> dict[str, Any]:
        output_path = Path(output)
        records: list[dict[str, Any]] = []
        buffer = io.BytesIO()
        with tempfile.TemporaryDirectory(prefix="vaelor-export-") as temporary:
            snapshot_root = Path(temporary)
            for relative in DATABASES:
                source = self.root / relative
                if not source.is_file():
                    continue
                snapshot = snapshot_root / relative
                self._database_snapshot(source, snapshot, relative)
                payload = snapshot.read_bytes()
                records.append(
                    {"path": relative, "bytes": len(payload), "sha256": _sha256(payload)}
                )
            for relative in FILES:
                source = self.root / relative
                if source.is_file() and not source.is_symlink():
                    payload = source.read_bytes()
                    records.append(
                        {"path": relative, "bytes": len(payload), "sha256": _sha256(payload)}
                    )
            for source in self._workload_files():
                relative = source.relative_to(self.root).as_posix()
                payload = source.read_bytes()
                records.append(
                    {"path": relative, "bytes": len(payload), "sha256": _sha256(payload)}
                )
            records.sort(key=lambda item: item["path"])
            manifest = {
                "schema": SCHEMA,
                "product": "Vaelor",
                "version": __version__,
                "created_at": int(time.time()),
                "files": records,
                "excluded": list(EXCLUDED),
            }
            with zipfile.ZipFile(
                buffer, mode="w", compression=zipfile.ZIP_DEFLATED, compresslevel=9
            ) as archive:
                for record in records:
                    relative = str(record["path"])
                    snapshot = snapshot_root / relative
                    source = snapshot if snapshot.exists() else self.root / relative
                    archive.writestr(relative, source.read_bytes())
                archive.writestr(
                    "manifest.json",
                    json.dumps(manifest, separators=(",", ":"), sort_keys=True),
                )

        plaintext = buffer.getvalue()
        if len(plaintext) > MAX_ARCHIVE_BYTES:
            raise PortableStateError("Portable state exceeds the 512 MiB transfer limit.")
        salt, nonce = os.urandom(16), os.urandom(12)
        header = json.dumps(
            {
                "schema": SCHEMA,
                "kdf": "scrypt",
                "n": 2**15,
                "r": 8,
                "p": 1,
                "salt": base64.b64encode(salt).decode("ascii"),
                "nonce": base64.b64encode(nonce).decode("ascii"),
            },
            separators=(",", ":"),
            sort_keys=True,
        ).encode("utf-8")
        aad = MAGIC + header
        encrypted = AESGCM(_derive_key(passphrase, salt)).encrypt(
            nonce, plaintext, aad
        )
        output_path.parent.mkdir(parents=True, exist_ok=True)
        temporary_output = output_path.with_suffix(output_path.suffix + ".tmp")
        descriptor = os.open(
            temporary_output,
            os.O_CREAT | os.O_EXCL | os.O_WRONLY,
            0o600,
        )
        with os.fdopen(descriptor, "wb") as stream:
            stream.write(MAGIC)
            stream.write(len(header).to_bytes(4, "big"))
            stream.write(header)
            stream.write(encrypted)
        temporary_output.replace(output_path)
        return manifest

    @staticmethod
    def _decrypt(source: Path, passphrase: str) -> bytes:
        if source.stat().st_size > MAX_ARCHIVE_BYTES:
            raise PortableStateError("The transfer archive exceeds the size limit.")
        with source.open("rb") as stream:
            if stream.read(len(MAGIC)) != MAGIC:
                raise PortableStateError("This is not a supported Vaelor transfer archive.")
            header_size = int.from_bytes(stream.read(4), "big")
            if not 32 <= header_size <= 4096:
                raise PortableStateError("The transfer archive header is invalid.")
            header = stream.read(header_size)
            encrypted = stream.read()
        try:
            metadata = json.loads(header)
            salt = base64.b64decode(metadata["salt"], validate=True)
            nonce = base64.b64decode(metadata["nonce"], validate=True)
            if metadata.get("schema") != SCHEMA or metadata.get("kdf") != "scrypt":
                raise PortableStateError("The transfer archive version is unsupported.")
            return AESGCM(_derive_key(passphrase, salt)).decrypt(
                nonce, encrypted, MAGIC + header
            )
        except (InvalidTag, KeyError, TypeError, ValueError) as error:
            raise PortableStateError(
                "The transfer passphrase is incorrect or the archive was changed."
            ) from error

    def inspect(self, source: Path | str, passphrase: str) -> dict[str, Any]:
        plaintext = self._decrypt(Path(source), passphrase)
        try:
            with zipfile.ZipFile(io.BytesIO(plaintext)) as archive:
                manifest = json.loads(archive.read("manifest.json"))
                if manifest.get("schema") != SCHEMA or not isinstance(
                    manifest.get("files"), list
                ):
                    raise PortableStateError("The transfer manifest is unsupported.")
                records = manifest["files"]
                if len(records) > MAX_TRANSFER_FILES:
                    raise PortableStateError("The transfer contains too many files.")
                declared = set()
                total = 0
                for record in records:
                    relative = _safe_relative(str(record.get("path", ""))).as_posix()
                    if relative in declared:
                        raise PortableStateError("The transfer contains a duplicate path.")
                    declared.add(relative)
                    expected = int(record.get("bytes", -1))
                    if expected < 0:
                        raise PortableStateError("A transferred file has an invalid size.")
                    total += expected
                    if total > MAX_ARCHIVE_BYTES:
                        raise PortableStateError(
                            "The expanded transfer exceeds the 512 MiB limit."
                        )
                    info = archive.getinfo(relative)
                    if info.is_dir() or info.file_size != expected:
                        raise PortableStateError("A transferred file has the wrong size.")
                    payload = archive.read(info)
                    if _sha256(payload) != record.get("sha256"):
                        raise PortableStateError("A transferred file failed verification.")
                actual = {
                    name for name in archive.namelist() if name != "manifest.json"
                }
                if actual != declared:
                    raise PortableStateError(
                        "The transfer file list does not match its manifest."
                    )
                return manifest
        except (
            KeyError,
            TypeError,
            ValueError,
            zipfile.BadZipFile,
            json.JSONDecodeError,
        ) as error:
            if isinstance(error, PortableStateError):
                raise
            raise PortableStateError(
                "The transfer archive or manifest is invalid."
            ) from error

    def import_archive(
        self,
        source: Path | str,
        passphrase: str,
        *,
        replace: bool = False,
    ) -> dict[str, Any]:
        source_path = Path(source)
        plaintext = self._decrypt(source_path, passphrase)
        manifest = self.inspect(source_path, passphrase)
        targets = [
            self.root / _safe_relative(str(item["path"]))
            for item in manifest["files"]
        ]
        existing = [str(path.relative_to(self.root)) for path in targets if path.exists()]
        if existing and not replace:
            raise PortableStateError(
                "Portable state already exists. Review a replacement import first."
            )
        backup_root = self.root / "import-backups" / uuid.uuid4().hex
        if existing:
            for path in targets:
                if path.is_file():
                    destination = backup_root / path.relative_to(self.root)
                    destination.parent.mkdir(parents=True, exist_ok=True)
                    shutil.copy2(path, destination)
        result = {
            "imported": len(manifest["files"]),
            "source_version": manifest.get("version"),
            "backup": str(backup_root) if existing else None,
            "files": [str(path.relative_to(self.root)) for path in targets],
            "sessions_restored": False,
            "credentials_restored": False,
        }
        staged: list[tuple[Path, Path]] = []
        try:
            with zipfile.ZipFile(io.BytesIO(plaintext)) as archive:
                for item in manifest["files"]:
                    relative = _safe_relative(str(item["path"]))
                    destination = self.root / relative
                    destination.parent.mkdir(parents=True, exist_ok=True)
                    temporary = destination.with_name(
                        f".{destination.name}.{uuid.uuid4().hex}.import"
                    )
                    descriptor = os.open(
                        temporary,
                        os.O_CREAT | os.O_EXCL | os.O_WRONLY,
                        0o600,
                    )
                    with os.fdopen(descriptor, "wb") as stream:
                        stream.write(archive.read(relative.as_posix()))
                    self._sanitize_database(temporary, relative.as_posix())
                    owner_source = (
                        destination if destination.exists() else destination.parent
                    )
                    owner = owner_source.stat()
                    if hasattr(os, "chown"):
                        os.chown(temporary, owner.st_uid, owner.st_gid)
                    os.chmod(
                        temporary,
                        0o660
                        if relative.parts[0] in {"jobs", "workloads"}
                        else 0o600,
                    )
                    staged.append((temporary, destination))
            for temporary, destination in staged:
                temporary.replace(destination)
                for suffix in ("-wal", "-shm"):
                    Path(str(destination) + suffix).unlink(missing_ok=True)
            return result
        except Exception:
            for temporary, _destination in staged:
                temporary.unlink(missing_ok=True)
            self.rollback_import(result)
            raise

    def rollback_import(self, result: dict[str, Any]) -> dict[str, Any]:
        files = [
            _safe_relative(str(value))
            for value in result.get("files", [])
        ]
        backup_value = result.get("backup")
        backup_root = Path(str(backup_value)) if backup_value else None
        if backup_root is not None:
            allowed_root = (self.root / "import-backups").resolve()
            try:
                backup_root.resolve().relative_to(allowed_root)
            except ValueError as error:
                raise PortableStateError("The import rollback path is invalid.") from error
        restored = 0
        for relative in files:
            destination = self.root / relative
            backup = backup_root / relative if backup_root is not None else None
            for suffix in ("", "-wal", "-shm"):
                Path(str(destination) + suffix).unlink(missing_ok=True)
            if backup is not None and backup.is_file():
                destination.parent.mkdir(parents=True, exist_ok=True)
                shutil.copy2(backup, destination)
                restored += 1
        return {"rolled_back": True, "restored": restored}
