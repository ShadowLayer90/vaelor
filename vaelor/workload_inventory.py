"""Safe inventory and app-scoped management for Docker workloads and GGUF models."""

from __future__ import annotations

import hashlib
import json
import os
import re
import shutil
import subprocess
import tempfile
import time
from pathlib import Path
from collections.abc import Mapping
from typing import Any, Callable

from .compose_policy import reject_secret_lines, reject_unsafe_keys, validate_normalized
from .app_capability_registry import app_instance_id_for_workload
from .managed_app_capabilities import safe_published_ports
from .runtime_paths import data_path
from .workload_broker import WorkloadBrokerClient


APP_ID = re.compile(r"^[a-f0-9]{12,64}$")
MANAGED_TEMPLATE_ID = re.compile(r"^[a-z0-9][a-z0-9.-]{0,63}$")
MANAGED_MODEL_CREDENTIAL = re.compile(r"^cred_managed_local_[a-f0-9]{12,64}$")
MODEL_HASH_CHUNK_BYTES = 4 * 1024 * 1024
SECRET_LINE = re.compile(
    r"(?im)^(\s*(?:password|token|api[_-]?key|secret)\s*[:=]\s*).+$"
)


def _default_run(command: list[str], **kwargs):
    return subprocess.run(command, **kwargs)

def model_file_identity(path: Path) -> dict[str, Any]:
    """Return a content-and-metadata binding for one managed model file.

    Reads the whole file, every time. On the appliance that is 60-plus seconds
    of SHA-256 over microSD, which is why it belongs only where the binding is
    the point - confirming the file about to be deleted is the file that was
    reviewed - and never on a listing. :func:`model_file_summary` is what a
    listing gets.

    **It is not cached, and a cache keyed on path, size and modification time
    would be wrong.** `tests/test_workload_dependencies.py` demonstrates the
    exact attack it would miss: replace the bytes with different bytes of the
    same length, restore the original mtime, and all three key components are
    unchanged while the content is not. That test caught this function being
    given such a cache, at the one call site where a stale digest would let a
    swapped model through a reviewed removal.
    """
    resolved = path.resolve()
    before = resolved.stat()
    hasher = hashlib.sha256()
    with resolved.open("rb") as source:
        for chunk in iter(lambda: source.read(MODEL_HASH_CHUNK_BYTES), b""):
            hasher.update(chunk)
    after = resolved.stat()
    if (
        before.st_size != after.st_size
        or before.st_mtime_ns != after.st_mtime_ns
    ):
        raise OSError("The model file changed while its identity was being read.")
    return {
        "path": str(resolved),
        "sha256": hasher.hexdigest(),
        "size_bytes": before.st_size,
        "mtime_ns": before.st_mtime_ns,
    }


def model_file_summary(path: Path) -> dict[str, Any]:
    """What a *listing* can know about a model file without reading it.

    Measured on the appliance on 2026-08-09: `list_all()` took 65-100 seconds
    across three consecutive runs, of which 68 seconds was hashing every GGUF
    on disk - about 7 GB - on every load of the Manage tab. No browser request
    ever completed, and until alpha 25 the screen rendered that failure as
    *"No apps installed yet"* on an appliance with five containers running.

    **Nothing on that screen used the digest.** It is used where it earns its
    cost, by `workload_removal` and `workload_dependencies`, one file at a time
    at the moment of a destructive action.

    `sha256` stays in the payload because `/api/v2` is a public interface and
    removing a field is a breaking change - but it is always empty here, with
    ``sha256_known`` saying so. **An empty string is not a digest and must
    never be compared against one**; a caller that needs the binding calls
    :func:`model_file_identity` and pays for it.
    """
    resolved = path.resolve()
    stat = resolved.stat()
    return {
        "path": str(resolved),
        "sha256": "",
        "sha256_known": False,
        "size_bytes": stat.st_size,
        "mtime_ns": stat.st_mtime_ns,
    }


class WorkloadInventory:
    def __init__(
        self,
        workloads_root: str = data_path("workloads"),
        models_root: str = data_path("models"),
        runner: Callable[..., Any] = _default_run,
        credential_broker: Any | None = None,
    ):
        self.workloads_root = Path(workloads_root).resolve()
        self.models_root = Path(models_root).resolve()
        self.runner = runner
        self.broker = WorkloadBrokerClient() if runner is _default_run else None
        self.credential_broker = credential_broker

    @staticmethod
    def _redact(value: str) -> str:
        value = SECRET_LINE.sub(r"\1[redacted]", value)
        return re.sub(
            r"(?i)(bearer\s+|(?:token|password|api[_-]?key)=)[^\s]+",
            r"\1[redacted]",
            value,
        )

    def _run(self, command: list[str], timeout: int = 8):
        if self.broker is not None and command and command[0] == "docker":
            return self.broker.run(command, timeout)
        return self.runner(
            command,
            capture_output=True,
            text=True,
            timeout=timeout,
            check=False,
        )

    def _inside(self, path: Path, root: Path) -> bool:
        try:
            path.resolve().relative_to(root)
            return True
        except (OSError, ValueError):
            return False

    def _docker_apps(self) -> list[dict[str, Any]]:
        if shutil.which("docker") is None:
            return []
        listed = self._run(["docker", "ps", "-aq", "--no-trunc"])
        ids = [item for item in listed.stdout.splitlines() if APP_ID.fullmatch(item)][:100]
        if not ids:
            return []
        inspected = self._run(["docker", "inspect", *ids], timeout=12)
        if inspected.returncode != 0:
            return []
        try:
            records = json.loads(inspected.stdout)
        except (TypeError, json.JSONDecodeError):
            return []
        apps = []
        for record in records:
            labels = (record.get("Config") or {}).get("Labels") or {}
            state = record.get("State") or {}
            network = record.get("NetworkSettings") or {}
            working_dir = labels.get("com.docker.compose.project.working_dir", "")
            managed = bool(working_dir and self._inside(Path(working_dir), self.workloads_root))
            ports = []
            for container_port, bindings in (network.get("Ports") or {}).items():
                if not isinstance(container_port, str):
                    continue
                for binding in bindings or []:
                    if not isinstance(binding, Mapping):
                        continue
                    ports.append(
                        {
                            "container": container_port,
                            "host": binding.get("HostPort", ""),
                            "address": binding.get("HostIp", ""),
                        }
                    )
            try:
                published_ports = safe_published_ports(ports)
            except ValueError:
                # Keep the app visible but never expose an unvalidated route.
                published_ports = []
            health = (state.get("Health") or {}).get("Status")
            remote_port = next(
                (item["host"] for item in ports if item["container"].startswith("5900/")),
                None,
            )
            app_id = str(record.get("Id", ""))
            if not APP_ID.fullmatch(app_id):
                continue
            project = labels.get("com.docker.compose.project")
            service = labels.get("com.docker.compose.service")
            managed_identity = None
            if managed and isinstance(project, str) and isinstance(service, str):
                try:
                    managed_identity = app_instance_id_for_workload(project, service)
                except (TypeError, ValueError):
                    managed_identity = None
            template_id = labels.get("io.pironman.template") if managed_identity else None
            if not isinstance(template_id, str) or not MANAGED_TEMPLATE_ID.fullmatch(template_id.strip().lower()):
                template_id = None
            elif template_id.strip() != template_id:
                template_id = None
            container_name = str(record.get("Name", "")).lstrip("/") or app_id[:12]
            display_identity = (
                f"{project}/{service}"
                if managed_identity and isinstance(project, str) and isinstance(service, str)
                else container_name
            )
            apps.append(
                {
                    "id": app_id,
                    "name": container_name,
                    "display_identity": display_identity,
                    "image": str((record.get("Config") or {}).get("Image", "Unknown image")),
                    "status": str(state.get("Status", "unknown")),
                    "health": health,
                    "running": bool(state.get("Running")),
                    "project": project,
                    "service": service,
                    "app_instance_id": managed_identity,
                    "template_id": template_id,
                    "runtime_container_id": app_id if managed_identity else None,
                    "published_ports": published_ports,
                    "ports": ports,
                    "managed": managed,
                    "capabilities": {
                        "logs": True,
                        "configuration": managed,
                        "console": bool(state.get("Running")),
                        "remote_desktop": bool(remote_port),
                    },
                    "remote_desktop": (
                        {"kind": "vnc", "host_port": remote_port} if remote_port else None
                    ),
                }
            )
        return sorted(apps, key=lambda item: item["name"].lower())

    def _managed_model_assignment(self) -> dict[str, Any]:
        assignment: dict[str, Any] = {"active_for": []}
        if self.credential_broker is None:
            return assignment
        for purpose in ("deployment-agent", "ai-chat"):
            try:
                lease = self.credential_broker.resolve_active(purpose)
            except (OSError, RuntimeError, ValueError):
                continue
            credential_id = str((lease or {}).get("credential_id", ""))
            if not MANAGED_MODEL_CREDENTIAL.fullmatch(credential_id):
                continue
            if assignment.get("credential_id") not in (None, credential_id):
                continue
            assignment["credential_id"] = credential_id
            assignment["active_for"].append(purpose)
            selected = str(
                (lease or {}).get("model") or (lease or {}).get("selected_model") or ""
            ).strip()
            if selected:
                assignment["model"] = selected
            endpoint = str((lease or {}).get("base_url", "")).strip()
            if endpoint:
                assignment["endpoint"] = endpoint
        return assignment

    def _managed_model_runtime(self) -> dict[str, Any] | None:
        """Resolve one managed llama.cpp identity from Docker, not job history."""
        if shutil.which("docker") is None:
            return None
        try:
            listed = self._run([
                "docker", "ps", "-aq", "--no-trunc",
                "--filter", "label=com.docker.compose.project=model-assistant",
            ])
        except (OSError, RuntimeError, subprocess.SubprocessError):
            return None
        ids = [item for item in listed.stdout.splitlines() if APP_ID.fullmatch(item)][:4]
        if listed.returncode or not ids:
            return None
        try:
            inspected = self._run(["docker", "inspect", *ids], timeout=12)
        except (OSError, RuntimeError, subprocess.SubprocessError):
            return None
        if inspected.returncode:
            return None
        try:
            records = json.loads(inspected.stdout)
        except (TypeError, json.JSONDecodeError):
            return None
        assignment = self._managed_model_assignment()
        for record in records if isinstance(records, list) else []:
            labels = (record.get("Config") or {}).get("Labels") or {}
            state = record.get("State") or {}
            if (
                labels.get("com.docker.compose.project") != "model-assistant"
                or labels.get("com.docker.compose.service") != "llama-server"
            ):
                continue
            working = str(labels.get("com.docker.compose.project.working_dir", ""))
            if not working or not self._inside(Path(working), self.workloads_root):
                continue
            environment = {}
            for item in (record.get("Config") or {}).get("Env") or []:
                key, separator, value = str(item).partition("=")
                if separator:
                    environment[key] = value
            container_model = environment.get("LLAMA_ARG_MODEL", "")
            if not container_model.startswith("/models/"):
                continue
            host_path = None
            for mount in record.get("Mounts") or []:
                if mount.get("Type") != "bind" or mount.get("Destination") != "/models":
                    continue
                candidate = Path(str(mount.get("Source", ""))) / Path(container_model).name
                try:
                    candidate.resolve().relative_to(Path(str(mount.get("Source", ""))).resolve())
                except (OSError, ValueError):
                    continue
                host_path = candidate
                break
            name = Path(container_model).stem
            selected = str(assignment.get("model", ""))
            selected_name = Path(selected).stem if selected else ""
            if selected_name and selected_name != name:
                assignment["identity_warning"] = (
                    "The active credential reports a different model from the managed runtime."
                )
            return {
                "container_id": str(record.get("Id", "")),
                "container_name": str(record.get("Name", "")).lstrip("/"),
                "project": "model-assistant",
                "service": "llama-server",
                "running": bool(state.get("Running")),
                "health": (state.get("Health") or {}).get("Status"),
                "container_model": container_model,
                "host_path": str(host_path) if host_path is not None else "",
                "name": name,
                **assignment,
            }
        return None

    def _models(self) -> list[dict[str, Any]]:
        runtime = self._managed_model_runtime()
        try:
            active_config = (
                self.workloads_root / "model-assistant" / "compose.yaml"
            ).read_text(encoding="utf-8")
        except OSError:
            active_config = ""
        models = []
        paths = list(self.models_root.rglob("*.gguf"))[:200] if self.models_root.exists() else []
        runtime_path = Path(str((runtime or {}).get("host_path", "")))
        if runtime_path.is_file() and all(
            path.resolve() != runtime_path.resolve() for path in paths
        ):
            paths.append(runtime_path)
        for path in paths:
            path = path.resolve()
            if not self._inside(path, self.models_root):
                if not runtime or path.resolve() != runtime_path.resolve():
                    continue
            try:
                # Stat, not SHA-256. See `model_file_summary`: hashing here
                # cost 68 s of the 65-100 s this listing took on the appliance,
                # for a digest no caller of this endpoint reads.
                identity = model_file_summary(path)
            except (OSError, ValueError):
                continue
            relative = (
                str(path.relative_to(self.models_root))
                if self._inside(path, self.models_root) else path.name
            )
            runtime_match = bool(
                runtime and (
                    path.resolve() == runtime_path.resolve()
                    or path.name == Path(str(runtime.get("container_model", ""))).name
                )
            )
            # #147: seven files on disk, three in the catalog, and every row
            # offered "Use model". The extra four are evaluation candidates
            # and the fine-tune — present because we put them there, not
            # because an owner installed them. A file the catalog does not
            # name has no verified identity and no measured footprint, so it
            # is listed with why it is here, and is not offerable.
            from .model_footprint import identify_by_file

            models.append(
                {
                    "id": hashlib.sha256(relative.encode()).hexdigest()[:16],
                    "name": path.stem,
                    "file": relative,
                    **identity,
                    "modified_at": int(identity["mtime_ns"] / 1_000_000),
                    "status": "ready",
                    "catalog": identify_by_file(path.name) is not None,
                    "in_use": bool(
                        runtime_match and runtime.get("running")
                    ) or (str(path.parent) in active_config and path.name in active_config),
                    **({"runtime": runtime} if runtime_match else {}),
                }
            )
        if runtime and not any(item.get("runtime") for item in models):
            models.append({
                "id": hashlib.sha256(
                    ("runtime:" + str(runtime.get("container_model", ""))).encode()
                ).hexdigest()[:16],
                "name": runtime["name"],
                "file": Path(str(runtime.get("container_model", ""))).name,
                "path": str(runtime.get("host_path", "")),
                "size_bytes": 0,
                "modified_at": 0,
                "sha256": "",
                "mtime_ns": 0,
                "status": "degraded",
                "status_reason": "The running model file is not readable from managed storage.",
                "in_use": bool(runtime.get("running")),
                "runtime": runtime,
            })
        return sorted(models, key=lambda item: item["name"].lower())

    def list_all(self) -> dict[str, Any]:
        return {
            "apps": self._docker_apps(),
            "models": self._models(),
            "capabilities": {
                "logs": True,
                "configuration": "managed-projects-only",
                "console": "diagnostics-only",
                "remote_desktop": "detected-vnc-only",
            },
        }

    def _app(self, app_id: str) -> dict[str, Any]:
        if not APP_ID.fullmatch(app_id):
            raise ValueError("Invalid app identifier.")
        app = next((item for item in self._docker_apps() if item["id"] == app_id), None)
        if app is None:
            raise ValueError("App was not found.")
        return app

    def logs(self, app_id: str, tail: int = 200) -> dict[str, Any]:
        self._app(app_id)
        tail = min(max(int(tail), 20), 500)
        # `--timestamps` makes Docker prepend its own RFC3339Nano instant to
        # every line, whichever stream it came from. Without it a container
        # that writes no timestamp of its own (searxng's uwsgi banners, Python
        # tracebacks) gave the log viewer nothing to show but "Time not
        # reported"; with it the viewer always has a real, authoritative time.
        result = self._run(
            ["docker", "logs", "--timestamps", "--tail", str(tail), app_id],
            timeout=10,
        )
        output = (result.stdout or "") + (result.stderr or "")
        return {"output": self._redact(output[-65536:]), "tail": tail}

    def diagnostics(self, app_id: str, tool: str) -> dict[str, Any]:
        self._app(app_id)
        commands = {
            "stats": [
                "docker", "stats", "--no-stream", "--format",
                "CPU {{.CPUPerc}} | Memory {{.MemUsage}} | Network {{.NetIO}}", app_id,
            ],
            "processes": [
                "docker", "top", app_id, "-eo",
                "pid,ppid,user,stat,etime,comm",
            ],
        }
        if tool not in commands:
            raise ValueError("Choose either stats or processes.")
        result = self._run(commands[tool], timeout=10)
        output = (result.stdout or "") + (result.stderr or "")
        return {"tool": tool, "output": self._redact(output[-65536:])}

    def vnc_target(self, app_id: str) -> int:
        app = self._app(app_id)
        remote = app.get("remote_desktop") or {}
        try:
            port = int(remote.get("host_port", 0))
        except (TypeError, ValueError):
            port = 0
        if not app["running"] or not 5900 <= port <= 65535:
            raise ValueError("This app does not have a running VNC desktop.")
        return port

    def _config_path(self, app_id: str) -> Path:
        app = self._app(app_id)
        if not app["managed"] or not app.get("project"):
            raise ValueError("Configuration editing is only available for managed apps.")
        project = self.workloads_root / str(app["project"])
        for filename in ("compose.yaml", "compose.yml", "docker-compose.yml"):
            candidate = project / filename
            if candidate.is_file() and self._inside(candidate, self.workloads_root):
                return candidate
        raise ValueError("No managed Compose configuration was found.")

    def read_config(self, app_id: str) -> dict[str, Any]:
        path = self._config_path(app_id)
        content = path.read_text(encoding="utf-8")
        if len(content.encode()) > 65536:
            raise ValueError("Configuration is too large for the web editor.")
        return {"content": self._redact(content), "filename": path.name, "editable": True}

    def save_config(self, app_id: str, content: str) -> dict[str, Any]:
        if not isinstance(content, str) or not content.strip() or len(content.encode()) > 65536:
            raise ValueError("Enter a configuration smaller than 64 KB.")
        if SECRET_LINE.search(content):
            raise ValueError("Store credentials in the credential broker, not in app configuration.")
        reject_secret_lines(content)
        reject_unsafe_keys(content)
        path = self._config_path(app_id)
        original = path.stat()
        with tempfile.NamedTemporaryFile(
            "w", encoding="utf-8", suffix=".yaml", dir=path.parent, delete=False
        ) as temporary:
            temporary.write(content)
            temporary_path = Path(temporary.name)
        temporary_path.chmod(0o660)
        try:
            result = self._run(
                [
                    "docker", "compose", "--project-directory", str(path.parent),
                    "-f", str(temporary_path), "config", "--format", "json",
                ],
                timeout=15,
            )
            if result.returncode != 0:
                raise ValueError(self._redact(result.stderr.strip()) or "Docker rejected this configuration.")
            try:
                normalized = json.loads(result.stdout)
            except (TypeError, json.JSONDecodeError) as error:
                raise ValueError(
                    "Docker returned an unreadable normalized configuration."
                ) from error
            validate_normalized(normalized, self.workloads_root)
            history = path.parent / ".history"
            history.mkdir(mode=0o2770, exist_ok=True)
            try:
                history.chmod(0o2770)
            except PermissionError:
                # A history directory created by the executor is already
                # group-writable; group members cannot change its mode.
                pass
            if hasattr(os, "geteuid") and os.geteuid() == 0:
                os.chown(history, original.st_uid, original.st_gid)
            for prior_backup in list(history.glob("*.yaml"))[:100]:
                if not prior_backup.is_file():
                    continue
                try:
                    prior_backup.chmod(0o660)
                except PermissionError:
                    pass
                if hasattr(os, "geteuid") and os.geteuid() == 0:
                    os.chown(
                        prior_backup,
                        original.st_uid,
                        original.st_gid,
                    )
            backup = history / f"{path.stem}.{int(time.time())}.yaml"
            backup.write_bytes(path.read_bytes())
            backup.chmod(0o660)
            if hasattr(os, "geteuid") and os.geteuid() == 0:
                os.chown(backup, original.st_uid, original.st_gid)
            # The dashboard can run as root while the allowlisted lifecycle
            # The executor runs as vaelor-workloads. Preserve the managed file's
            # ownership and access mode when replacing it so a normal edit does
            # not lock the executor out of its own Compose project.
            temporary_path.chmod(original.st_mode & 0o777)
            if hasattr(os, "geteuid") and os.geteuid() == 0:
                os.chown(temporary_path, original.st_uid, original.st_gid)
            temporary_path.replace(path)
        finally:
            temporary_path.unlink(missing_ok=True)
        return {"saved": True, "filename": path.name, "backup": backup.name}
