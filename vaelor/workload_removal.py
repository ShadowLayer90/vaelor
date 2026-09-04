"""Execution helpers for reviewed dependency-aware workload removal."""

from __future__ import annotations

import hashlib
import json
import shutil
import tempfile
import time
from pathlib import Path
from typing import Any, Callable, Dict

from .managed_local_credentials import PREFIX as MANAGED_CREDENTIAL_PREFIX
from .workload_inventory import model_file_identity


def _backup_model(model: Path, backups_root: Path) -> Dict[str, Any]:
    destination_root = (backups_root / "models").resolve()
    destination_root.mkdir(parents=True, exist_ok=True)
    destination = destination_root / "{}-{}.gguf".format(
        model.stem, time.time_ns()
    )
    try:
        shutil.copy2(model, destination)
        destination.chmod(0o660)
        hasher = hashlib.sha256()
        with destination.open("rb") as source:
            for chunk in iter(lambda: source.read(4 * 1024 * 1024), b""):
                hasher.update(chunk)
        result = {
            "path": str(destination), "size_bytes": destination.stat().st_size,
            "sha256": hasher.hexdigest(),
        }
        if not destination.is_file() or result["size_bytes"] == 0:
            raise RuntimeError("Model backup verification failed.")
        return result
    except Exception:
        try:
            destination.unlink()
        except OSError:
            pass
        raise


def _managed_credentials(broker) -> list[Dict[str, Any]]:
    return [
        item for item in broker.list()
        if str(item.get("id", "")).startswith(MANAGED_CREDENTIAL_PREFIX)
    ]


def _deactivate_managed_consumers(
    broker, credentials: list[Dict[str, Any]], detached: list[str] | None = None
):
    detached = detached if detached is not None else []
    for credential in credentials:
        for purpose in credential.get("active_for", []):
            if purpose in {"deployment-agent", "ai-chat"}:
                # Record the assignment before the broker call. A broker may
                # mutate state and then raise, and that partial mutation still
                # needs compensation.
                if purpose not in detached:
                    detached.append(purpose)
                broker.deactivate(purpose)
    return sorted(set(detached))


def _delete_managed_credentials(broker, credentials, attempted):
    for credential in credentials:
        credential_id = str(credential.get("id", ""))
        attempted.append(credential_id)
        broker.delete(credential_id)


def _capture_credential_profiles(broker, credentials):
    """Capture only trusted local leases needed to recreate deleted credentials."""
    profiles = {}
    resolver = getattr(broker, "resolve_active", None)
    if not callable(resolver):
        return profiles
    for credential in credentials:
        credential_id = str(credential.get("id", ""))
        for purpose in credential.get("active_for", []):
            if purpose not in {"deployment-agent", "ai-chat"}:
                continue
            try:
                lease = resolver(purpose)
            except Exception:
                continue
            if str(lease.get("credential_id", "")) != credential_id:
                continue
            provider = str(lease.get("provider", ""))
            if provider == "openai-compatible":
                secret = json.dumps({
                    "base_url": lease.get("base_url", ""),
                    "model": lease.get("model", ""),
                    "api_key": lease.get("api_key", ""),
                }, separators=(",", ":"))
            else:
                secret = str(lease.get("token", ""))
            if secret:
                profiles[credential_id] = {
                    "provider": provider,
                    "label": str(lease.get("label", credential_id)),
                    "secret": secret,
                }
    return profiles


def _restore_managed_consumers(
    broker, credentials, profiles, detached, deleted_ids
):
    try:
        current = {
            str(item.get("id", "")) for item in broker.list()
        }
    except Exception as error:
        raise RuntimeError("Managed credential state could not be restored.") from error
    for credential_id in deleted_ids:
        if credential_id in current:
            continue
        profile = profiles.get(credential_id)
        put = getattr(broker, "put", None)
        if not profile or not callable(put):
            raise RuntimeError(
                "A deleted managed credential could not be restored."
            )
        put(
            profile["provider"], profile["label"], profile["secret"],
            credential_id=credential_id,
        )
        current.add(credential_id)
    activate = getattr(broker, "activate", None)
    if detached and not callable(activate):
        raise RuntimeError("Managed credential assignments could not be restored.")
    for purpose in detached:
        credential = next(
            (item for item in credentials
             if purpose in item.get("active_for", [])), None
        )
        if credential is not None:
            activate(str(credential["id"]), purpose)


def _model_path(resource: Dict[str, Any], models_root: Path) -> Path:
    path = Path(str(resource.get("path", ""))).resolve()
    root = models_root.resolve()
    if (
        path.suffix.lower() != ".gguf"
        or root not in path.parents
        or not path.is_file()
    ):
        raise ValueError("The reviewed model file is outside managed storage.")
    return path


def _revalidate_model(resource: Dict[str, Any], models_root: Path) -> Path:
    path = _model_path(resource, models_root)
    identity = model_file_identity(path)
    expected = {
        field: resource.get(field)
        for field in ("path", "sha256", "size_bytes", "mtime_ns")
    }
    if any(identity[field] != expected[field] for field in expected):
        raise ValueError("The reviewed model file changed after approval.")
    return path


def _snapshot_project(project: Path, backups_root: Path):
    if not project.is_dir():
        return None
    backups_root = backups_root.resolve()
    backups_root.mkdir(parents=True, exist_ok=True)
    root = Path(tempfile.mkdtemp(prefix=".vaelor-removal-", dir=str(backups_root)))
    snapshot = root / project.name
    try:
        shutil.copytree(project, snapshot)
    except Exception:
        shutil.rmtree(root, ignore_errors=True)
        raise
    return {"root": root, "path": snapshot, "project": project}


def _restore_project(
    snapshot: Dict[str, Any], project: Path, compose_remove, compose_restore
):
    if not snapshot:
        return
    snapshot_path = Path(snapshot["path"])
    if project.is_dir():
        shutil.rmtree(project)
    elif project.exists():
        project.unlink()
    shutil.copytree(snapshot_path, project)
    restore_payload = {
        "project": project.name, "confirm": project.name,
        "create_backup": False, "retain_data": True,
    }
    if callable(compose_restore):
        compose_restore(restore_payload)
        return
    owner = getattr(compose_remove, "__self__", None)
    compose_up = getattr(owner, "_compose", None)
    if callable(compose_up):
        compose_up(project, "up", "-d", "--remove-orphans", timeout=180)


def _cleanup_snapshot(snapshot):
    if snapshot:
        shutil.rmtree(Path(snapshot["root"]), ignore_errors=True)


def _cleanup_safety_backup(backup):
    if not backup:
        return
    path = Path(backup["path"])
    try:
        path.unlink()
    except OSError:
        return
    try:
        path.parent.rmdir()
    except OSError:
        pass


def _restore_model(model_path: Path, backup: Dict[str, Any], expected):
    if model_path.exists():
        current = model_file_identity(model_path)
        if current == expected:
            return
        raise RuntimeError("The model path changed during removal rollback.")
    model_path.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(backup["path"], model_path)
    restored = model_file_identity(model_path)
    if any(restored[field] != expected[field] for field in expected):
        raise RuntimeError("Model rollback verification failed.")


def execute_managed_removal(
    payload: Dict[str, Any],
    actor: str,
    dependencies,
    compose_remove: Callable[[Dict[str, Any]], Dict[str, Any]],
    broker,
    models_root: Path,
    backups_root: Path,
    compose_restore: Callable[[Dict[str, Any]], Dict[str, Any]] | None = None,
    job_id: str | None = None,
) -> Dict[str, Any]:
    """Revalidate a plan and execute it as a compensating transaction."""
    report = dependencies.validate_removal(payload, actor)
    kind = report["resource"]["kind"]
    dependency_strategy = payload["dependency_strategy"]
    cascade = dependency_strategy == "cascade"
    create_backup = payload["create_backup"]
    retain_data = payload["retain_data"]
    result: Dict[str, Any] = {
        "resource": report["resource"],
        "resource_id": report["resource"]["id"],
        "display_identity": report["display_identity"],
        "plan_digest": report["plan_digest"],
        "dependency_strategy": dependency_strategy,
        "cascade": cascade,
        "retain_data": retain_data,
        "create_backup": create_backup,
    }

    inventory_root = dependencies.inventory.workloads_root.resolve()
    project_name = report["resource"].get("project", "")
    if kind == "runtime" or (kind == "model" and cascade):
        project_name = "model-assistant"
    project = (inventory_root / str(project_name)).resolve() if project_name else None
    if project is not None and inventory_root not in project.parents:
        raise ValueError("The reviewed Compose project is outside managed storage.")
    compose_file = project / "compose.yaml" if project is not None else None
    remove_runtime = bool(
        project is not None and compose_file.is_file()
        and (kind == "runtime" or cascade)
    )
    delete_model = kind == "model" or (kind == "runtime" and not retain_data)
    model_resource = None
    if delete_model:
        model_resource = (
            report["resource"] if kind == "model"
            else report["resource"].get("model")
        )
        if not isinstance(model_resource, dict):
            raise ValueError("The reviewed model file is no longer available.")

    model_path = None
    model_backup = None
    model_deleted = False
    project_snapshot = None
    compose_attempted = False
    credentials = []
    credential_profiles = {}
    detached = []
    deleted_ids = []
    try:
        if delete_model:
            model_path = _revalidate_model(model_resource, models_root)
            model_backup = _backup_model(model_path, backups_root)
            expected = {
                field: model_resource[field]
                for field in ("path", "sha256", "size_bytes", "mtime_ns")
            }
            if any(model_backup[field] != expected[field]
                   for field in ("sha256", "size_bytes")):
                raise RuntimeError("Model backup verification failed.")
            _revalidate_model(model_resource, models_root)

        if project is not None and ((kind == "app") or remove_runtime):
            project_snapshot = _snapshot_project(project, backups_root)

        if kind == "app":
            compose_attempted = True
            result["removal"] = compose_remove({
                "project": report["resource"]["project"],
                "confirm": report["resource"]["project"],
                "create_backup": create_backup,
                "retain_data": retain_data,
                "_actor": actor, "_job_id": job_id,
            })
        else:
            if remove_runtime:
                credentials = _managed_credentials(broker)
                credential_profiles = _capture_credential_profiles(
                    broker, credentials
                )
                if model_resource is not None:
                    _revalidate_model(model_resource, models_root)
                compose_attempted = True
                result["runtime_removal"] = compose_remove({
                    "project": "model-assistant", "confirm": "model-assistant",
                    "create_backup": create_backup, "retain_data": True,
                    "_actor": actor, "_job_id": job_id,
                })
                detached = _deactivate_managed_consumers(
                    broker, credentials, detached
                )
                result["detached_consumers"] = detached

            if delete_model:
                _revalidate_model(model_resource, models_root)
                size = model_path.stat().st_size
                model_path.unlink()
                model_deleted = True
                result["model_removal"] = {
                    "path": str(model_path), "reclaimed_bytes": size,
                }

            if remove_runtime:
                _delete_managed_credentials(broker, credentials, deleted_ids)

        if delete_model:
            if create_backup:
                result["model_backup"] = model_backup
            else:
                _cleanup_safety_backup(model_backup)
            try:
                model_path.parent.rmdir()
            except OSError:
                pass
        _cleanup_snapshot(project_snapshot)
        return result
    except Exception as error:
        rollback_errors = []
        if delete_model and model_backup and model_path is not None:
            try:
                if model_deleted or not model_path.exists():
                    _restore_model(model_path, model_backup, expected)
            except Exception as rollback_error:
                rollback_errors.append(rollback_error)
        if detached or deleted_ids:
            try:
                _restore_managed_consumers(
                    broker, credentials, credential_profiles,
                    detached, deleted_ids,
                )
            except Exception as rollback_error:
                rollback_errors.append(rollback_error)
        if compose_attempted and project_snapshot:
            try:
                _restore_project(
                    project_snapshot, project, compose_remove, compose_restore,
                )
            except Exception as rollback_error:
                rollback_errors.append(rollback_error)
        if not create_backup:
            _cleanup_safety_backup(model_backup)
        _cleanup_snapshot(project_snapshot)
        if rollback_errors:
            raise RuntimeError(
                "Managed removal failed and compensation was incomplete: {}"
                .format(rollback_errors[0])
            ) from error
        raise
