"""Compose deployment persistence, backup, removal, and restore operations."""

from __future__ import annotations

import hashlib
import hmac
import io
import json
import os
import re
import shutil
import subprocess
import tarfile
import tempfile
import time
import urllib.error
import urllib.request
from pathlib import Path
from typing import Any, Dict
from urllib.parse import urlsplit

from .application_executor import (
    application_draft_environment,
    complete_application_import,
    condense_docker_error,
    record_application_failure,
    resolve_application_import,
    rollback_application_compose,
    wait_for_application_health,
)
from .checkpoints import CheckpointInventory
from .deployment_refresh import MANAGED_PROJECT


#: Managed projects a Compose import may never create or overwrite.
#:
#: ``MANAGED_PROJECT`` ("model-assistant") is the running local AI runtime,
#: managed through the runtime removal/redeploy path - which already refuses to
#: treat it as an app (``workload_dependencies`` raises "Manage the local AI
#: server as a runtime..."). The import path had no matching guard, so a raw
#: ``compose.import`` with ``project: model-assistant`` would overwrite the
#: runtime's compose and ``up --remove-orphans`` it (#205 finding 3). Named from
#: the one owner rather than a second literal, per LESSONS 6 / #98.
RESERVED_IMPORT_PROJECTS = frozenset({MANAGED_PROJECT})

#: A held import lock older than this is treated as abandoned regardless of who
#: it names. It only has to exceed the longest an import can legitimately run: a
#: single ``_import_compose`` is bounded by its own step timeouts (config 30s,
#: pull 900s, up 180s, health 90s) to roughly twenty minutes, so an hour is
#: comfortable headroom and also the backstop against PID reuse - a dead
#: importer's PID reassigned to some unrelated live process would otherwise read
#: as "still held" for ever.
IMPORT_LOCK_MAX_AGE_SECONDS = 3600


def _process_alive(pid: int) -> bool:
    """Best-effort liveness check for the PID recorded in an import lock.

    Portable because the test suite and the appliance are different platforms:
    signal 0 on POSIX, a query-only process handle on Windows. A check that
    cannot decide errs toward *alive* so a live holder is never reclaimed out
    from under a running import.
    """
    if pid <= 0:
        return False
    if os.name == "nt":  # pragma: no cover - exercised only on Windows
        import ctypes

        PROCESS_QUERY_LIMITED_INFORMATION = 0x1000
        STILL_ACTIVE = 259
        kernel32 = ctypes.windll.kernel32
        handle = kernel32.OpenProcess(PROCESS_QUERY_LIMITED_INFORMATION, False, pid)
        if not handle:
            return False
        try:
            code = ctypes.c_ulong()
            if kernel32.GetExitCodeProcess(handle, ctypes.byref(code)):
                return code.value == STILL_ACTIVE
            return True
        finally:
            kernel32.CloseHandle(handle)
    try:
        os.kill(pid, 0)
    except ProcessLookupError:
        return False
    except PermissionError:
        return True  # exists, owned by someone else
    except OSError:
        return True
    return True


def _import_lock_is_stale(lock_path: Path) -> bool:
    """True when an import lock has no live holder and may be reclaimed.

    Stale means the writer is gone (dead PID), the lock is implausibly old
    (PID-reuse backstop), or the file is unreadable/malformed - none of which
    is a live import. A live PID within the age bound is *not* stale and must
    keep blocking (#205 M2).
    """
    try:
        raw = lock_path.read_text(encoding="utf-8").strip()
    except OSError:
        return True
    parts = raw.split()
    if len(parts) != 2:
        return True
    try:
        pid = int(parts[0])
        created = float(parts[1])
    except ValueError:
        return True
    if time.time() - created > IMPORT_LOCK_MAX_AGE_SECONDS:
        return True
    return not _process_alive(pid)


def _acquire_import_lock(lock_path: Path) -> None:
    """Claim the per-project import lock, healing one a crash left behind.

    ``O_EXCL`` serialises two *live* imports of one project. But a hard crash
    (SIGKILL/OOM/power loss) between the create and the ``finally`` that removes
    it would otherwise strand the file, and then every future import of that
    project is refused for ever - the re-brick-on-crash class of #205 finding 1,
    reopened (M2). So a lock whose holder is gone is reclaimed. The reclaim is
    race-safe: two processes that both see one stale lock both unlink and both
    retry ``O_EXCL``, which still lets exactly one win; the loser then reads the
    winner's fresh live identity and is refused. The identity written is
    ``"<pid> <epoch>"``.
    """
    for reclaimed in (False, True):
        try:
            lock_fd = os.open(
                str(lock_path), os.O_CREAT | os.O_EXCL | os.O_WRONLY, 0o660
            )
        except FileExistsError:
            if not reclaimed and _import_lock_is_stale(lock_path):
                try:
                    os.unlink(str(lock_path))
                except FileNotFoundError:
                    pass
                continue
            raise RuntimeError(
                "Another import is already changing this managed project. Wait "
                "for it to finish before importing again."
            )
        else:
            try:
                os.write(
                    lock_fd, "{} {}".format(os.getpid(), time.time()).encode()
                )
            finally:
                os.close(lock_fd)
            return


class _NoRedirectHandler(urllib.request.HTTPRedirectHandler):
    def redirect_request(self, *_args, **_kwargs):
        return None


class ExecutorComposeLifecycleMixin:
    """Implement the stateful Compose lifecycle used by ``JobExecutor``."""

    def _import_compose(self, payload: Dict[str, Any], actor: str = "") -> Dict[str, Any]:
        application_draft, payload = resolve_application_import(
            payload, self.application_deployments, actor,
            broker=self.credential_broker,
        )
        project = self._project_directory(str(payload.get("project", "")))
        if project.name in RESERVED_IMPORT_PROJECTS:
            raise ValueError(
                "'{}' is the managed local AI runtime and cannot be replaced by "
                "a Compose import. Manage it as a runtime instead.".format(project.name)
            )
        content = payload.get("content", "")
        if not isinstance(content, str) or not content.strip() or len(content.encode()) > 64 * 1024:
            raise ValueError("Enter a Compose configuration smaller than 64 KB.")
        self._reject_secret_lines(content)
        self._reject_unsafe_compose_keys(content)
        docker = shutil.which("docker")
        if docker is None:
            raise RuntimeError("Docker is not installed.")
        project.mkdir(parents=True, exist_ok=True)
        project.chmod(0o2770)
        compose_file = project / "compose.yaml"
        # One import per project at a time. Two concurrent imports of the same
        # *new* project each read `previous_content is None`, so without this
        # both would write, `up`, and report healthy (#205 finding 3). The lock
        # self-heals a crash that stranded it rather than deadlocking every
        # future import of the project (#205 M2); see `_acquire_import_lock`.
        lock_path = project / ".import.lock"
        _acquire_import_lock(lock_path)
        temporary_path = None
        candidate_committed = False
        previous_content = (
            compose_file.read_text(encoding="utf-8")
            if compose_file.exists() else None
        )
        try:
            # An existing managed project is replaced only through the
            # draft-gated approval flow. A raw import (no draft) carries no
            # approval and previously overwrote silently (#205 finding 3b).
            if previous_content is not None and not (
                application_draft is not None
                and bool(application_draft["compose"].get("x-vaelor-replace-existing"))
            ):
                raise ValueError(
                    "An application already uses this managed project. Review and explicitly approve replacement."
                )
            with tempfile.NamedTemporaryFile(
                "w", encoding="utf-8", suffix=".yaml", dir=project, delete=False
            ) as temporary:
                temporary.write(content)
                temporary_path = Path(temporary.name)
            temporary_path.chmod(0o660)
            secret_environment = application_draft_environment(application_draft)
            self._checkpoint(20, "Normalizing Compose configuration")
            result = subprocess.run(
                [docker, "compose", "--project-directory", str(project), "-f",
                 str(temporary_path), "config", "--format", "json"],
                capture_output=True, check=False, text=True, timeout=30,
                env=secret_environment,
            )
            if result.returncode != 0:
                raise ValueError(
                    condense_docker_error(result.stderr, "Docker rejected this Compose file")
                )
            normalized = json.loads(result.stdout)
            self._validate_normalized_compose(normalized, project)
            if compose_file.exists():
                history = project / ".history"
                history.mkdir(mode=0o2770, exist_ok=True)
                history.chmod(0o2770)
                backup = history / "compose.{}.yaml".format(int(time.time()))
                shutil.copy2(compose_file, backup)
                backup.chmod(0o660)
            temporary_path.replace(compose_file)
            candidate_committed = True
            self._checkpoint(40, "Downloading container images", "downloading")
            self._compose(project, "pull", timeout=900)
            self._checkpoint(80, "Starting custom app", "starting")
            self._compose(project, "up", "-d", "--remove-orphans", timeout=180)
            health = wait_for_application_health(
                lambda: self._compose(project, "ps", "--format", "json", timeout=30),
                sorted(normalized["services"]),
            )
            result = {
                "project": project.name, "services": sorted(normalized["services"]),
                "compose_file": str(compose_file), "policy": "validated",
                "health": health,
            }
            endpoint_candidates = [
                {"service": service_name, "url": f"http://127.0.0.1:{published}", "status": "candidate"}
                for service_name, service in normalized["services"].items()
                for published in self._published_http_ports(service)
            ]
            result["endpoint_candidates"] = endpoint_candidates
            result["verified_endpoints"] = [
                {**candidate, "status": "verified", "proof": proof}
                for candidate in endpoint_candidates
                if (proof := self._probe_http_endpoint(candidate["url"])) is not None
            ]
            if result["verified_endpoints"]:
                result["open_url"] = result["verified_endpoints"][0]["url"]
            return complete_application_import(
                self.application_deployments, application_draft, result, actor,
                self.application_learning,
            )
        except Exception as deployment_error:
            rollback_completed = not candidate_committed
            if candidate_committed:
                try:
                    rollback_application_compose(
                        self._compose, project, compose_file, previous_content
                    )
                    rollback_completed = True
                except Exception as rollback_error:
                    record_application_failure(
                        self.application_learning,
                        application_draft, deployment_error,
                        rollback_completed=False,
                    )
                    raise RuntimeError(
                        "Deployment failed and automatic rollback is incomplete: {}"
                        .format(str(rollback_error)[:500])
                    ) from deployment_error
            record_application_failure(
                self.application_learning,
                application_draft, deployment_error,
                rollback_completed=rollback_completed,
            )
            raise
        finally:
            if temporary_path is not None:
                temporary_path.unlink(missing_ok=True)
            lock_path.unlink(missing_ok=True)
            if (
                previous_content is None
                and project.exists()
                and not any(project.iterdir())
            ):
                project.rmdir()

    @staticmethod
    def _published_http_ports(service: Dict[str, Any]) -> list[int]:
        ports = service.get("ports", []) if isinstance(service, dict) else []
        result: list[int] = []
        for binding in ports if isinstance(ports, list) else []:
            if not isinstance(binding, dict):
                continue
            published = binding.get("published")
            target = binding.get("target")
            protocol = str(binding.get("protocol", "tcp")).lower()
            if protocol in {"tcp", "http", "https"} and isinstance(published, int) and isinstance(target, int):
                result.append(published)
        return result

    @staticmethod
    def _probe_http_endpoint(url: str) -> Dict[str, Any] | None:
        """Return proof that a local published port actually speaks HTTP.

        Compose metadata can identify a TCP mapping but cannot identify the
        application protocol.  Only a bounded loopback HTTP response may
        promote a candidate into ``verified_endpoints`` and ``open_url``.
        """
        parsed = urlsplit(str(url))
        if parsed.scheme not in {"http", "https"} or parsed.hostname not in {
            "127.0.0.1", "localhost", "::1"
        } or parsed.username or parsed.password:
            return None
        try:
            port = parsed.port
        except ValueError:
            return None
        if not isinstance(port, int) or not 1 <= port <= 65535:
            return None
        request = urllib.request.Request(
            url, headers={"Accept": "*/*", "User-Agent": "Vaelor endpoint verifier"}
        )
        opener = urllib.request.build_opener(
            _NoRedirectHandler(),
        )
        try:
            with opener.open(request, timeout=3) as response:
                response.read(256)
                status = int(response.getcode() or 0)
                content_type = str(response.headers.get("content-type", ""))[:120]
        except urllib.error.HTTPError as error:
            # A 4xx response still proves that an HTTP server answered; a 5xx
            # response is an application-health failure and must not expose an
            # Open action.
            status = int(error.code or 0)
            if not 200 <= status < 500:
                return None
            content_type = str(error.headers.get("content-type", ""))[:120]
        except (OSError, ValueError, urllib.error.URLError):
            return None
        if not 200 <= status < 500:
            return None
        return {"status": status, "content_type": content_type}

    def _backup_compose(self, payload: Dict[str, Any]) -> Dict[str, Any]:
        project = self._project_directory(str(payload.get("project", "")))
        if not (project / "compose.yaml").is_file():
            raise ValueError("The managed project was not found.")
        backup_root = (self.backups_root / "workloads").resolve()
        backup_root.mkdir(parents=True, exist_ok=True)
        backup_root.chmod(0o2770)
        created_at_ms = int(time.time() * 1000)
        destination = backup_root / "{}-{}.tar.gz".format(
            project.name, created_at_ms
        )
        actor = str(
            payload.get("_actor")
            or payload.get("actor")
            or "executor"
        ).strip()[:200] or "executor"
        raw_job_id = payload.get("_job_id", getattr(self, "_active_job_id", None))
        job_id = str(raw_job_id).strip()[:200] if raw_job_id else None
        lineage = {
            "actor": actor,
            "created_by": actor,
            "creator": actor,
            "created_at_ms": created_at_ms,
            "creating_job_id": job_id,
            "job_id": job_id,
            "project": project.name,
            "schema": CheckpointInventory.LINEAGE_SCHEMA,
        }
        lineage_bytes = CheckpointInventory.canonical_lineage(lineage)
        self._checkpoint(40, "Creating project backup")
        with tarfile.open(destination, "w:gz") as archive:
            def archive_filter(member: tarfile.TarInfo):
                if member.name == "{}/{}".format(project.name, CheckpointInventory.LINEAGE_MEMBER):
                    return None
                return member

            archive.add(
                project,
                arcname=project.name,
                recursive=True,
                filter=archive_filter,
            )
            metadata = tarfile.TarInfo(
                "{}/{}".format(project.name, CheckpointInventory.LINEAGE_MEMBER)
            )
            metadata.mode = 0o600
            metadata.mtime = created_at_ms / 1000
            metadata.size = len(lineage_bytes)
            archive.addfile(metadata, fileobj=io.BytesIO(lineage_bytes))
        destination.chmod(0o660)
        if not destination.is_file() or destination.stat().st_size == 0:
            raise RuntimeError("Backup verification failed.")
        record = CheckpointInventory(str(backup_root)).verify(destination.name)
        return {
            "project": project.name,
            "backup": str(destination),
            "size_bytes": record["size_bytes"],
            "sha256": record["sha256"],
            "manifest_digest": record["manifest_digest"],
            "creation_lineage": record["creation_lineage"],
            "preflight": record["preflight"],
        }

    def _remove_compose(self, payload: Dict[str, Any]) -> Dict[str, Any]:
        project = self._project_directory(str(payload.get("project", "")))
        if str(payload.get("confirm", "")) != project.name:
            raise ValueError("Type the project name to confirm removal.")
        retain_data = bool(
            payload.get("retain_data", not payload.get("remove_volumes", False))
        )
        create_backup = bool(payload.get("create_backup", True))
        backup = (
            self._backup_compose({
                "project": project.name,
                "_actor": payload.get("_actor") or payload.get("actor"),
                "_job_id": payload.get("_job_id") or payload.get("job_id"),
            })
            if create_backup else None
        )
        self._checkpoint(55, "Stopping project", "starting")
        arguments = ["down", "--remove-orphans"]
        if not retain_data:
            arguments.append("--volumes")
        self._compose(project, *arguments, timeout=180)
        recovered = None
        if backup is not None:
            recovered = Path(backup["backup"]).with_suffix(".project")
            if recovered.exists():
                raise RuntimeError("A recovery destination already exists.")
            shutil.move(str(project), str(recovered))
        else:
            shutil.rmtree(project)
        return {
            "project": project.name, "removed": True,
            "volumes_removed": not retain_data, "backup": backup,
            "recovery_copy": str(recovered) if recovered else None,
        }

    def _preflight_staged_compose(self, restored: Path) -> Dict[str, Any]:
        compose_file = restored / "compose.yaml"
        if not compose_file.is_file():
            raise ValueError("Checkpoint does not contain compose.yaml.")
        self._compose(restored, "config", "--quiet", timeout=30)
        return {
            "state": "compose_config_validated",
            "structural_valid": True,
            "compose_config": "validated",
            "startup": "unproven",
            "startup_proven": False,
            "message": "Staged Compose configuration passed preflight; startup is not proven.",
        }

    def _restore_checkpoint(self, payload: Dict[str, Any]) -> Dict[str, Any]:
        checkpoint = str(payload.get("checkpoint", "")).strip()
        project_name = str(payload.get("project", "")).strip()
        digest = str(payload.get("sha256", "")).strip().lower()
        manifest_digest = str(payload.get("manifest_digest", "")).strip().lower()
        if not re.fullmatch(r"[a-z0-9][a-z0-9_-]{1,47}", project_name):
            raise ValueError("Choose a managed project.")
        if str(payload.get("confirm", "")) != project_name:
            raise ValueError("Type the project name to confirm restoration.")
        if not re.fullmatch(r"[0-9a-f]{64}", digest):
            raise ValueError("Restore requires the verified checkpoint digest.")
        if not re.fullmatch(r"[0-9a-f]{64}", manifest_digest):
            raise ValueError("Restore requires the verified checkpoint manifest.")
        backup_root = (self.backups_root / "workloads").resolve()
        archive_path = (backup_root / checkpoint).resolve()
        if archive_path.parent != backup_root:
            raise ValueError("Choose a valid managed checkpoint.")
        project = self._project_directory(project_name)
        if not (project / "compose.yaml").is_file():
            raise ValueError("The current managed project was not found.")
        inventory = CheckpointInventory(str(backup_root))
        binding = inventory.bind_restore(checkpoint, project_name, digest)
        if not hmac.compare_digest(binding["manifest_digest"], manifest_digest):
            raise ValueError("Checkpoint manifest changed since it was verified.")
        with tempfile.TemporaryDirectory(dir=self.workloads_root) as temporary:
            stage = Path(temporary).resolve()
            staged_archive = stage / checkpoint
            staged_digest = hashlib.sha256()
            try:
                with archive_path.open("rb") as source, staged_archive.open("wb") as target:
                    for chunk in iter(lambda: source.read(4 * 1024 * 1024), b""):
                        staged_digest.update(chunk)
                        target.write(chunk)
            except OSError as error:
                raise ValueError("Checkpoint could not be read from disk.") from error
            if not hmac.compare_digest(staged_digest.hexdigest(), digest):
                raise ValueError("Checkpoint changed since it was verified.")
            staged_binding = CheckpointInventory(str(stage)).bind_restore(
                checkpoint, project_name, digest
            )
            if not hmac.compare_digest(
                staged_binding["manifest_digest"], manifest_digest
            ):
                raise ValueError("Checkpoint manifest changed since it was verified.")
            restore_root = stage / "restore"
            restore_root.mkdir()
            with tarfile.open(staged_archive, "r:gz") as archive:
                members = archive.getmembers()
                if not members or len(members) > 5000:
                    raise ValueError("Checkpoint contents are invalid.")
                for member in members:
                    target = (restore_root / member.name).resolve()
                    if restore_root not in target.parents and target != restore_root:
                        raise ValueError("Checkpoint contains an unsafe path.")
                    if member.issym() or member.islnk() or member.isdev():
                        raise ValueError("Checkpoint contains an unsupported file type.")
                    first = Path(member.name).parts[0] if Path(member.name).parts else ""
                    if first != project_name:
                        raise ValueError("Checkpoint belongs to a different project.")
                archive.extractall(restore_root, members=members, filter="data")
            restored = restore_root / project_name
            preflight = self._preflight_staged_compose(restored)
            lineage_file = restored / CheckpointInventory.LINEAGE_MEMBER
            if lineage_file.is_file():
                lineage_file.unlink()
            self._checkpoint(35, "Verified staged checkpoint", "validating")
            safety_backup = self._backup_compose({
                "project": project_name,
                "_actor": payload.get("_actor"),
                "_job_id": payload.get("_job_id"),
            })
            rollback = project.with_name(".{}-restore-rollback".format(project.name))
            if rollback.exists():
                shutil.rmtree(rollback)
            self._compose(project, "down", "--remove-orphans", timeout=180)
            project.replace(rollback)
            try:
                shutil.copytree(restored, project)
                self._compose(project, "config", "--quiet", timeout=30)
                self._checkpoint(75, "Starting restored application", "starting")
                self._compose(project, "up", "-d", "--remove-orphans", timeout=180)
            except Exception:
                if project.exists():
                    shutil.rmtree(project)
                rollback.replace(project)
                self._compose(project, "up", "-d", "--remove-orphans", timeout=180)
                raise
            shutil.rmtree(rollback)
        return {
            "project": project_name,
            "restored_from": checkpoint,
            "pre_restore_backup": safety_backup,
            "preflight": preflight,
        }
