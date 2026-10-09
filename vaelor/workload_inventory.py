"""Safe inventory and app-scoped management for Docker workloads and GGUF models."""

from __future__ import annotations

import hashlib
import ipaddress
import json
import os
import re
import shutil
import subprocess
import tempfile
import time
from contextlib import contextmanager
from pathlib import Path
from collections.abc import Mapping
from typing import Any, Callable

from .app_port_claims import model_port_holders
from .compose_policy import reject_secret_lines, reject_unsafe_keys, validate_normalized
from .app_capability_registry import app_instance_id_for_workload
from .app_catalog import APP_TEMPLATES, install_env_from_compose
from .managed_app_capabilities import safe_published_ports, template_id_from_labels
from .runtime_paths import data_path
from .workload_broker import WorkloadBrokerClient
#: The model listing (which files are downloaded, and which are serving) lives
#: in `workload_models`, with the managed-credential id rule it reads.
from .workload_models import INSPECT_BATCH, WorkloadModelsMixin


APP_ID = re.compile(r"^[a-f0-9]{12,64}$")
MANAGED_TEMPLATE_ID = re.compile(r"^[a-z0-9][a-z0-9.-]{0,63}$")
MODEL_HASH_CHUNK_BYTES = 4 * 1024 * 1024
#: A configuration file the app-files editor will read or write must be text
#: and this small; 256 KiB is generous for a YAML config and bounds both the
#: docker-cp payload and what a browser editor loads.
CONFIG_FILE_MAX_BYTES = 256 * 1024
#: Cap on console command output returned to the browser (bytes). The broker
#: also bounds every command to a 30-second timeout.
CONSOLE_OUTPUT_MAX_BYTES = 64 * 1024
#: Refusals shared by the read and write paths, kept once each so the two sides
#: cannot drift apart (and so the message is not written twice in this module).
_UNDECLARED_FILE = "That file is not an editable configuration file for this app."
_FILE_TOO_LARGE = "This configuration file is too large for the web editor."
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


NOT_MANAGED = "This is not a managed Vaelor application."
NOT_FROM_CATALOG = (
    "Vaelor manages this app, but it was not installed from the app catalog, so "
    "its files cannot be browsed or edited here."
)


def catalog_template(app: Mapping[str, Any]) -> dict[str, Any] | None:
    """The catalog template a managed app was installed from, or None.

    The one rule (W4d-D4) for whether the template-scoped tools - the file
    manager, generated credentials, declared config files - can act on an app:
    "managed" alone also covers Vaelor's own projects (the web-research search
    app), which have no template and which the broker will not let them reach.
    """
    template_id = app.get("template_id")
    if not app.get("managed") or not isinstance(template_id, str):
        return None
    return APP_TEMPLATES.get(template_id)


def _host_port(entry: Mapping[str, Any]) -> int | None:
    host = str(entry.get("host", ""))
    return int(host) if host.isdigit() else None


def _loopback(address: Any) -> bool:
    try:
        return ipaddress.ip_address(str(address).strip("[]")).is_loopback
    except ValueError:
        return False  # "" or a name: Docker publishes on every interface


def local_only_ports(ports: list[dict[str, Any]]) -> list[int]:
    """Host ports every binding of which is a loopback address (W4d-D5)."""
    by_port: dict[int, list[bool]] = {}
    for entry in ports:
        host = _host_port(entry)
        if host is not None:
            by_port.setdefault(host, []).append(_loopback(entry.get("address", "")))
    return sorted(port for port, flags in by_port.items() if all(flags))


class WorkloadInventory(WorkloadModelsMixin):
    def __init__(
        self,
        workloads_root: str = data_path("workloads"),
        models_root: str = data_path("models"),
        runner: Callable[..., Any] = _default_run,
        credential_broker: Any | None = None,
        npu_models_root: str | None = None,
        npu_status: Callable[[], Any] | None = None,
    ):
        self.workloads_root = Path(workloads_root).resolve()
        self.models_root = Path(models_root).resolve()
        self.runner = runner
        self.broker = WorkloadBrokerClient() if runner is _default_run else None
        self.credential_broker = credential_broker
        # The on-device (NPU) models and what the NPU is serving (ACC-106),
        # read by `WorkloadModelsMixin`; ``npu_status`` is the hardware bridge's
        # ``flm_status``, or ``None`` where no one can ask it.
        self.npu_models_root = Path(npu_models_root) if npu_models_root else None
        self.npu_status = npu_status

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
        # The one container read (`_container_records`): every id, in the
        # batches the workload broker admits; unreadable lists no apps.
        records = self._container_records() or []
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
            template_id = template_id_from_labels(labels) if managed_identity else None
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
                    "web_port": self._web_port(template_id, ports, published_ports),
                    # W4d-D5: ports only this machine can reach. Not links.
                    "local_only_ports": local_only_ports(ports),
                    # W6 sweep: Docker lists no ports for a host-network
                    # container (vaelor-llm-proxy serves 11434 that way).
                    "host_network": str(
                        (record.get("HostConfig") or {}).get("NetworkMode", "")
                    ) == "host",
                    "managed": managed,
                    "capabilities": {
                        "logs": True,
                        "configuration": managed,
                        "console": bool(state.get("Running")),
                        "remote_desktop": bool(remote_port),
                        # W4d-D4: the file manager, credentials and config
                        # files need a catalog template; same rule as
                        # `_managed_template`, which refuses the rest.
                        "file_manager": catalog_template(
                            {"managed": managed, "template_id": template_id}
                        ) is not None,
                    },
                    "remote_desktop": (
                        {"kind": "vnc", "host_port": remote_port} if remote_port else None
                    ),
                }
            )
        return sorted(apps, key=lambda item: item["name"].lower())

    def _web_port(
        self,
        template_id: str | None,
        ports: list[dict[str, Any]],
        published_ports: list[dict[str, Any]],
    ) -> int | None:
        """Host port that reaches the app's web UI, not an incidental port.

        A multi-port app publishes more than one host port and the first one
        Docker reports can be the non-HTTP extra port - Syncthing publishes
        22000 (sync) and 8384 (GUI), AdGuard 53 (DNS) and 3000 (admin). The
        "Open app" link must land on the GUI, so resolve the published host
        port whose CONTAINER side is the template's web ``container_port``. When
        the template is unknown, fall back to the current first-published-port
        behaviour rather than guess.
        """
        # W4d-D5: a port bound only to loopback cannot be opened from the
        # owner's browser, so it is never the link (searxng on 127.0.0.1:8888,
        # Phoenix on 127.0.0.1:6006 were offered as LAN links and refused).
        local = set(local_only_ports(ports))
        ports = [entry for entry in ports if _host_port(entry) not in local]
        published_ports = [
            entry for entry in published_ports if entry.get("host_port") not in local
        ]
        template = APP_TEMPLATES.get(template_id) if template_id else None
        if template is not None:
            want = template.get("container_port")
            for entry in ports:
                container = str(entry.get("container", ""))
                head = container.split("/", 1)[0]
                host = str(entry.get("host", ""))
                if head.isdigit() and int(head) == want and host.isdigit():
                    return int(host)
        for entry in published_ports:
            host_port = entry.get("host_port")
            if isinstance(host_port, int) and not isinstance(host_port, bool):
                return host_port
        for entry in ports:
            host = str(entry.get("host", ""))
            if host.isdigit():
                return int(host)
        return None

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

    def stored_models(self) -> list[dict[str, Any]]:
        """The ``models`` `list_all` lists, without its app read: one Docker read, not two."""
        return self._models()

    def app_names(self, app_ids: Any) -> dict[str, str] | None:
        """``{container id: name}`` for the wanted ids, read lightly (W8-3).

        The audit trail names an app row with one ``docker ps`` and one
        ``docker inspect`` of only the ids it shows - never ``list_all``, which
        inspects every container, reads every model file, asks the NPU bridge
        and discovers the GPU just to name a few apps. Both reads are the ones
        the workload broker already admits. An id Docker no longer lists is
        absent from the answer (the app is gone); ``None`` means Docker could
        not be read, which the caller says, never as gone (LESSONS 8).
        """
        wanted = sorted({str(item) for item in app_ids if APP_ID.fullmatch(str(item))})
        if not wanted:
            return {}
        if shutil.which("docker") is None:
            return None
        try:
            listed = self._run(["docker", "ps", "-aq", "--no-trunc"])
        except (OSError, RuntimeError, subprocess.SubprocessError):
            return None
        if listed.returncode:
            return None
        present = listed.stdout.split()
        # An audit row may carry a short id; Docker lists full ones.
        full = {
            item: next((each for each in present if each.startswith(item)), None)
            for item in wanted
        }
        ids = sorted({value for value in full.values() if value})
        found: dict[str, str] = {}
        for start in range(0, len(ids), INSPECT_BATCH):
            records = self._inspect(ids[start:start + INSPECT_BATCH])
            if records is None:
                return None
            for record in records:
                app_id = str(record.get("Id", ""))
                # The same name the Manage list shows for this container.
                found[app_id] = str(record.get("Name", "")).lstrip("/") or app_id[:12]
        return {item: found[value] for item, value in full.items() if value in found}

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
            # W7-2: an edit applies on the next Restart (W4d-D12), so a port a
            # stored model comes back on is refused here too.
            validate_normalized(
                normalized, self.workloads_root,
                model_ports=model_port_holders(self.credential_broker),
            )
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
        # W4d-D12: the save writes the file only. The running containers keep
        # their configuration until Vaelor recreates them (start, restart and
        # update all run `docker compose up -d`); stop does not.
        return {
            "saved": True, "applied": False, "filename": path.name,
            "backup": backup.name,
            "applies_when": (
                "The running app keeps its current configuration until you "
                "restart, start or update it from Vaelor."
            ),
        }

    def _managed_template(self, app_id: str) -> tuple[dict[str, Any], str, dict[str, Any]]:
        """Resolve a managed app to its curated template, or refuse.

        Both credential reveal and file editing act only on apps Vaelor itself
        deployed from a known template: the template is what says which secrets
        exist and which files may be edited. An unmanaged or unknown-template
        container has neither, so it is refused rather than guessed at.
        """
        app = self._app(app_id)
        template = catalog_template(app)
        if template is None:
            raise ValueError(NOT_FROM_CATALOG if app.get("managed") else NOT_MANAGED)
        return app, app["template_id"], template

    def read_secrets(self, app_id: str) -> dict[str, str]:
        """Return the UN-redacted values of the template's ``secret_env`` keys.

        The Configuration tab redacts these (code-server's generated PASSWORD,
        for one), so an operator has no other way to retrieve a secret Vaelor
        minted for them. Only keys the template declares as ``secret_env`` are
        returned - ``install_env_from_compose`` reads them straight back out of
        the compose file Vaelor wrote - and apps with none return ``{}``.
        """
        _app, template_id, template = self._managed_template(app_id)
        if not template.get("secret_env"):
            return {}
        content = self._config_path(app_id).read_text(encoding="utf-8")
        return install_env_from_compose(template_id, content)

    def _config_file_target(
        self, template_id: str, template: dict[str, Any], rel_path: Any
    ) -> tuple[str, str]:
        """Map a caller path to (container name, in-container path), or refuse.

        ``rel_path`` must be one of the template's declared ``config_files`` by
        exact match. Exact membership is what forecloses traversal - a value
        holding ``..`` or a leading ``/`` simply is not in the declared set - and
        the explicit guard restates that so the invariant cannot be weakened by
        a future edit to the template list.
        """
        declared = list(template.get("config_files") or [])
        if not isinstance(rel_path, str) or rel_path not in declared:
            raise ValueError(_UNDECLARED_FILE)
        if rel_path.startswith("/") or ".." in rel_path.replace("\\", "/").split("/"):
            raise ValueError(_UNDECLARED_FILE)
        volume = template.get("volume")
        if not volume:
            raise ValueError("This app has no configuration volume.")
        _name, mount = volume
        return f"vaelor-{template_id}", f"{str(mount).rstrip('/')}/{rel_path}"

    @contextmanager
    def _private_workdir(self):
        """A per-operation setgid 2770 directory under the workloads root.

        ``docker cp`` runs in the broker, which is a *different* OS user from
        the control plane: the control plane runs as ``vaelor`` and the broker
        runs docker as ``vaelor-workloads``. They share only the group
        ``vaelor-jobs``, and the workloads root is setgid ``2770 vaelor-jobs``.
        So the cp target directory must be enterable and writable by *both*
        users, which means group access through ``vaelor-jobs`` - a 0700
        directory owned by ``vaelor`` locks the broker out entirely (reads land
        nothing and every file reports "not created yet"; writes never persist).

        Mode 2770 restores that cross-user access while keeping L2's intent: the
        only members of ``vaelor-jobs`` are Vaelor's own trusted service users
        (control plane + broker), so 2770 grants no access to anyone outside
        that group, and the unpredictable ``mkdtemp`` name means a hostile peer
        would have to both already be in ``vaelor-jobs`` *and* guess the random
        directory name to race the ``docker cp`` - a far higher bar than the
        pre-L2 predictable temp file directly in the group-writable root. The
        setgid bit (inherited from the root, restated here) keeps the group as
        ``vaelor-jobs`` for anything created inside. The directory still
        resolves inside the workloads root, which the broker's
        ``_host_config_workfile`` requires of the host cp target.

        The directory (and everything under it) is removed on the way out,
        whether the body returns or raises.
        """
        self.workloads_root.mkdir(parents=True, exist_ok=True)
        workdir = Path(tempfile.mkdtemp(dir=self.workloads_root, prefix=".appfile-"))
        try:
            workdir.chmod(0o2770)
            yield workdir
        finally:
            shutil.rmtree(workdir, ignore_errors=True)

    @contextmanager
    def _copy_from_container(self, container: str, container_path: str):
        """Land one container file on the host via ``docker cp``, yielding its
        path, or ``None`` if the file is absent.

        ``docker cp`` needs no shell in the container. Copying to a host temp
        file (rather than the ``-`` tar stream) keeps every byte off the
        broker's text channel: the command produces no stdout. The landed file
        is *not* read here - the caller stats it to enforce the size cap before
        deciding whether to read any bytes at all - and the private directory
        holding it is cleaned up when the ``with`` block exits.
        """
        with self._private_workdir() as workdir:
            temporary_path = workdir / "file"
            result = self._run(
                ["docker", "cp", f"{container}:{container_path}", str(temporary_path)],
                timeout=15,
            )
            yield temporary_path if result.returncode == 0 else None

    def list_app_files(self, app_id: str) -> list[dict[str, Any]]:
        """Existence and size of each of the template's declared config files."""
        _app, template_id, template = self._managed_template(app_id)
        files: list[dict[str, Any]] = []
        for relative in list(template.get("config_files") or []):
            container, container_path = self._config_file_target(template_id, template, relative)
            # Stat, not read: the listing needs only existence and size, so the
            # file's bytes never enter memory here regardless of how large it is.
            with self._copy_from_container(container, container_path) as landed:
                exists = landed is not None
                size = landed.stat().st_size if landed is not None else 0
            files.append({
                "path": relative,
                "exists": exists,
                "size": size,
            })
        return files

    def read_app_file(self, app_id: str, rel_path: Any) -> dict[str, Any]:
        """Return the text content of one declared config file."""
        _app, template_id, template = self._managed_template(app_id)
        container, container_path = self._config_file_target(template_id, template, rel_path)
        with self._copy_from_container(container, container_path) as landed:
            if landed is None:
                raise ValueError("That configuration file was not found in the app.")
            # Enforce the cap from the on-disk size BEFORE loading the file, so
            # an oversized file is refused without ever being read into memory.
            if landed.stat().st_size > CONFIG_FILE_MAX_BYTES:
                raise ValueError(_FILE_TOO_LARGE)
            data = landed.read_bytes()
        if b"\x00" in data:
            raise ValueError("This configuration file is not text and cannot be edited here.")
        try:
            content = data.decode("utf-8")
        except UnicodeDecodeError as error:
            raise ValueError("This configuration file is not valid UTF-8 text.") from error
        return {"path": rel_path, "content": content}

    def write_app_file(self, app_id: str, rel_path: Any, content: Any) -> dict[str, Any]:
        """Write text into one declared config file, via ``docker cp``.

        The write only ever targets a file the template already declares; it
        never creates a file outside that set. Content is text, bounded, and
        NUL-free, and reaches the container through a host temp file so no byte
        crosses the broker's text channel.
        """
        _app, template_id, template = self._managed_template(app_id)
        container, container_path = self._config_file_target(template_id, template, rel_path)
        if not isinstance(content, str):
            raise ValueError("Provide the configuration file content as text.")
        encoded = content.encode("utf-8")
        if len(encoded) > CONFIG_FILE_MAX_BYTES:
            raise ValueError(_FILE_TOO_LARGE)
        if b"\x00" in encoded:
            raise ValueError("Configuration files must be text.")
        with self._private_workdir() as workdir:
            temporary_path = workdir / "file"
            temporary_path.write_bytes(encoded)
            # The control plane (``vaelor``) writes this temp file, but the
            # broker (``vaelor-workloads``) is the one that runs ``docker cp``
            # and so must READ it. A file created here is 0600 by default, which
            # the broker cannot read; 0660 grants read (and write) to the shared
            # ``vaelor-jobs`` group so the cross-user cp works, while still
            # giving no access to anyone outside that trusted group.
            temporary_path.chmod(0o660)
            result = self._run(
                ["docker", "cp", str(temporary_path), f"{container}:{container_path}"],
                timeout=15,
            )
            if result.returncode != 0:
                raise ValueError(
                    self._redact((result.stderr or "").strip())
                    or "The configuration file could not be written to the app."
                )
        return {"ok": True}

    def exec_in_app(self, app_id: str, command: Any) -> dict[str, Any]:
        """Run an admin-authorized shell command inside a managed app container.

        This deliberately relaxes the diagnostics-only console: an administrator
        can run a real command (reset a password, change a setting) inside the
        app. It is confined to the app's own container - ``docker exec`` runs in
        that container's namespace, the blueprints mount no host paths and grant
        no privileged/socket access - and the broker independently allows only
        the ``docker exec <managed-container> sh -c <command>`` shape. The
        endpoint gates it on the administrator role + CSRF and audits the
        command.

        Output is bounded AT THE SOURCE: the command's merged stdout+stderr is
        piped through ``head -c`` inside the container, so the broker never
        buffers more than the cap and a runaway producer (``yes``) is stopped by
        SIGPIPE once it exceeds it. The broker also kills the ``docker exec``
        client at 30 seconds; a process that keeps running in the container after
        that stays confined by the app's ``mem_limit``/``cpus`` limits (the
        exec exit code then reflects that bounded pipeline, not the raw command).
        """
        _app, template_id, _template = self._managed_template(app_id)
        if not isinstance(command, str) or not command.strip():
            raise ValueError("Enter a command to run.")
        container = f"vaelor-{template_id}"
        bounded = f"( {command} ) 2>&1 | head -c {CONSOLE_OUTPUT_MAX_BYTES}"
        result = self._run(["docker", "exec", container, "sh", "-c", bounded], timeout=30)
        combined = result.stdout or ""
        encoded = combined.encode("utf-8", "replace")
        truncated = len(encoded) >= CONSOLE_OUTPUT_MAX_BYTES
        if len(encoded) > CONSOLE_OUTPUT_MAX_BYTES:
            combined = encoded[:CONSOLE_OUTPUT_MAX_BYTES].decode("utf-8", "replace")
        return {
            "output": combined,
            "exit_code": int(result.returncode),
            "truncated": truncated,
        }
