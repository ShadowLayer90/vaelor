"""Allowlisted Docker access for the unprivileged Vaelor control plane."""

from __future__ import annotations

import json
import os
import re
import shutil
import socket
import subprocess
from pathlib import Path
from types import SimpleNamespace
from typing import Any

from .app_catalog import (
    APP_TEMPLATES,
    data_path_is_within_roots,
    declared_config_targets,
)
from .runtime_paths import data_path, run_path


APP_ID = re.compile(r"^[a-f0-9]{12,64}$")

#: The exact cluster-status reads ``DockerSwarmDriver.status()`` issues, held
#: here so the driver and this allowlist cannot drift apart (#141). The control
#: plane deliberately has no docker group — that is what this broker is for —
#: yet the cluster screen queried the socket directly and told the owner the
#: runtime was unreachable while five brokered containers ran beside it. These
#: are reads with fixed argument lists, the same risk class as ``docker ps``
#: above. Swarm *mutations* stay out: they run as approval-gated jobs in the
#: workload executor, which holds the privilege itself.
SWARM_INFO_COMMAND = ("docker", "info", "--format", "{{json .Swarm}}")
SWARM_NODES_COMMAND = (
    "docker", "node", "ls", "--format",
    '{"id":{{json .ID}},"hostname":{{json .Hostname}},'
    '"status":{{json .Status}},"availability":{{json .Availability}},'
    '"manager_status":{{json .ManagerStatus}},"engine":{{json .EngineVersion}}}',
)
SWARM_SERVICES_COMMAND = (
    "docker", "service", "ls", "--format",
    '{"id":{{json .ID}},"name":{{json .Name}},'
    '"mode":{{json .Mode}},"replicas":{{json .Replicas}},'
    '"image":{{json .Image}},"ports":{{json .Ports}}}',
)
CLUSTER_STATUS_COMMANDS = (
    SWARM_INFO_COMMAND,
    SWARM_NODES_COMMAND,
    SWARM_SERVICES_COMMAND,
)

#: The task-row template ``DockerSwarmDriver.service_details`` sends with
#: ``docker service ps`` — shared for the same no-drift reason as above.
SERVICE_TASKS_FORMAT = (
    '{"id":{{json .ID}},"name":{{json .Name}},'
    '"image":{{json .Image}},"node":{{json .Node}},'
    '"desired":{{json .DesiredState}},"current":{{json .CurrentState}},'
    '"error":{{json .Error}}}'
)


#: The file manager's three fixed in-container scripts, held here as module
#: constants so the inventory sends BYTE-IDENTICAL text to what this allowlist
#: admits (the same no-drift discipline as ``SERVICE_TASKS_FORMAT``). The
#: browsable path is always passed as the POSITIONAL argument ``$1`` and is
#: never interpolated into the script text, so a hostile name cannot break out
#: of the single-quoted argument into the shell. Each script targets busybox/
#: alpine, debian and ubuntu ``sh``.
FS_LIST_SCRIPT = (
    'd=$1; [ -d "$d" ] || { echo __vaelor_notdir__; exit 3; }; cd "$d" || exit 3; '
    'for e in * .[!.]* ..?*; do [ -e "$e" ] || [ -L "$e" ] || continue; '
    'if [ -d "$e" ]; then t=d; s=0; else t=f; s=$(wc -c < "$e" 2>/dev/null | tr -d " "); fi; '  # absence-ok: an entry that vanished mid-listing or cannot be sized reports 0 via ${s:-0}, not an error line
    'printf "%s\\t%s\\t%s\\n" "$t" "${s:-0}" "$e"; done'
)
FS_MKDIR_SCRIPT = 'mkdir -- "$1"'
FS_DELETE_SCRIPT = 'rm -rf -- "$1"'
FS_SCRIPTS = {FS_LIST_SCRIPT, FS_MKDIR_SCRIPT, FS_DELETE_SCRIPT}

#: The one fixed script the file manager runs as ROOT inside a container, and
#: the only root exec this broker admits at all. ``docker cp`` preserves the
#: host temp file's owner (the control-plane user, uid ~997) and mode 0660 into
#: the container, so an uploaded file lands owned by a user the app is NOT, and
#: an app that runs as its own non-root user gets "permission denied" on its own
#: upload. This chowns the just-uploaded file to match its PARENT directory's
#: owner:group - the app's own identity for that data root - so the file becomes
#: native to the app (readable and writable by it). ``$1`` is the file, ``$2``
#: is the reference directory; both are passed POSITIONALLY and never
#: interpolated. ``chown -h`` never follows a symlink, and ``stat -c '%u:%g'``
#: is portable across busybox/alpine, debian and ubuntu. It is kept OUT of
#: ``FS_SCRIPTS`` so the non-root FS exec branch can never run it, and the root
#: branch below can never run an ``FS_SCRIPTS`` command.
FS_CHOWN_SCRIPT = 'own=$(stat -c "%u:%g" -- "$2") && chown -h "$own" -- "$1"'


def _managed_service(value: Any) -> bool:
    """The same name rule the driver enforces, checked independently here.

    The broker validates its own inputs rather than trusting a caller's
    validation: only Vaelor-managed service names pass, so the service reads
    below cannot be pointed at an arbitrary Swarm service.
    """
    return (
        isinstance(value, str)
        and value.startswith((
            "vaelor-app-", "vaelor-llm-",
        ))
        and len(value) <= 63
        and all(char.isalnum() or char in "._-" for char in value)
    )
SOCKET_PATH = Path(
    os.environ.get("VAELOR_WORKLOAD_BROKER_SOCKET", run_path("workloadd.sock"))
)
WORKLOADS_ROOT = Path(data_path("workloads")).resolve()
BACKUPS_ROOT = Path(data_path("backups/workloads")).resolve()
MAX_REQUEST = 128 * 1024
MAX_OUTPUT = 2 * 1024 * 1024


def _inside(path: Path, root: Path | None = None) -> bool:
    root = WORKLOADS_ROOT if root is None else root
    try:
        path.resolve().relative_to(root)
        return True
    except (OSError, ValueError):
        return False


def _bounded_ids(values: list[str]) -> bool:
    return 0 < len(values) <= 100 and all(APP_ID.fullmatch(item) for item in values)


def _managed_config_target(reference: Any) -> bool:
    """True only for ``vaelor-<template>:<one declared config file>``.

    The broker validates this independently of the caller, the same way
    ``_managed_service`` does for Swarm reads: the container must be a curated
    template's own container name (``vaelor-<template_id>``) and the
    in-container path must be one of that template's declared editable config
    files. ``declared_config_targets`` returns the exact allowed paths and an
    empty list for any unknown or file-less template, so membership is an exact
    match that fails closed - a path with ``..`` or a leading ``/`` simply is
    not in the set, and the explicit ``..`` guard restates that invariant.
    """
    if not isinstance(reference, str):
        return False
    name, separator, container_path = reference.partition(":")
    if not separator or not name.startswith("vaelor-"):
        return False
    if ".." in container_path.split("/"):
        return False
    template_id = name[len("vaelor-"):]
    return container_path in set(declared_config_targets(template_id))


def _managed_data_path(container: Any, container_path: Any) -> bool:
    """True only for a curated container plus a path inside its data roots.

    The file manager operates over each managed app's browsable data roots
    (its volume mount, plus a media mount for media apps). As with the other
    broker checks, the container is re-derived from the curated template set
    rather than trusted from the caller: ``container`` must be
    ``vaelor-<template_id>`` for a real ``APP_TEMPLATES`` entry, and
    ``container_path`` must be absolute, ``..``-free and equal to or under one
    of that template's ``declared_data_roots`` (segment-prefix, so ``/config``
    does not admit ``/configX``). ``data_path_is_within_roots`` is the single
    shared encoding of that rule; a template with no data roots admits nothing.
    """
    if not isinstance(container, str) or not container.startswith("vaelor-"):
        return False
    template_id = container[len("vaelor-"):]
    if template_id not in APP_TEMPLATES:
        return False
    return data_path_is_within_roots(template_id, container_path)


def _managed_data_target(reference: Any) -> bool:
    """True only for ``vaelor-<template>:<path under a data root>``.

    The data-file counterpart of ``_managed_config_target``: the ``docker cp``
    that uploads or downloads a file in the file manager is bounded to a
    curated container name and an in-container path this broker independently
    confirms lies inside that template's browsable data roots.
    """
    if not isinstance(reference, str):
        return False
    name, separator, container_path = reference.partition(":")
    if not separator:
        return False
    return _managed_data_path(name, container_path)


def _managed_container(name: Any) -> bool:
    """True only for ``vaelor-<template_id>`` naming a curated template's container.

    The admin console runs a shell inside ONE managed app container. As with the
    other broker checks, the container is re-derived here from the curated
    template set rather than trusted from the caller: only a name that maps to a
    real ``APP_TEMPLATES`` entry passes, so the exec can never target another
    container, an image, or the host. The blueprints mount no host paths and set
    no ``privileged``/docker-socket access, so the shell is confined to the
    container.
    """
    return (
        isinstance(name, str)
        and name.startswith("vaelor-")
        and name[len("vaelor-"):] in APP_TEMPLATES
    )


def _host_config_workfile(path: Any) -> bool:
    """A plain host temp file inside the managed workloads root.

    The other side of an allowlisted ``docker cp``. It is never a container
    reference (``_inside`` resolves it under ``WORKLOADS_ROOT``, which no
    ``name:path`` spec does) and never the ``-`` tar stream (``Path('-')``
    resolves outside the root), so the two cp branches below admit exactly one
    container target paired with one bounded host file, in either direction.
    """
    return isinstance(path, str) and bool(path) and _inside(Path(path))


def _validate(command: Any) -> list[str]:
    if not isinstance(command, list) or not all(
        isinstance(item, str) for item in command
    ):
        raise ValueError("Command must be a string list.")
    if command == ["docker", "ps", "-aq", "--no-trunc"]:
        return command
    if tuple(command) in CLUSTER_STATUS_COMMANDS:
        return command
    # The cluster screen's per-service reads (#141 review): inspect, task
    # rows, and logs, each read-only with a fixed argv shape and a
    # Vaelor-managed service name. Swarm mutations stay out of this broker.
    if (
        len(command) == 4
        and command[:3] == ["docker", "service", "inspect"]
        and _managed_service(command[3])
    ):
        return command
    if (
        len(command) == 7
        and command[:3] == ["docker", "service", "ps"]
        and _managed_service(command[3])
        and command[4:] == ["--no-trunc", "--format", SERVICE_TASKS_FORMAT]
    ):
        return command
    if (
        len(command) == 8
        and command[:6] == [
            "docker", "service", "logs", "--raw", "--timestamps", "--tail",
        ]
        and command[6].isdigit()
        and 20 <= int(command[6]) <= 500
        and _managed_service(command[7])
    ):
        return command
    if len(command) >= 3 and command[:2] == ["docker", "inspect"]:
        if _bounded_ids(command[2:]):
            return command
    if (
        len(command) == 6
        and command[:4] == ["docker", "logs", "--timestamps", "--tail"]
        and command[4].isdigit()
        and 20 <= int(command[4]) <= 500
        and APP_ID.fullmatch(command[5])
    ):
        return command
    if (
        len(command) == 6
        and command[:5]
        == ["docker", "stats", "--no-stream", "--format",
            "CPU {{.CPUPerc}} | Memory {{.MemUsage}} | Network {{.NetIO}}"]
        and APP_ID.fullmatch(command[5])
    ):
        return command
    if (
        len(command) == 5
        and command[:2] == ["docker", "top"]
        and APP_ID.fullmatch(command[2])
        and command[3:] == ["-eo", "pid,ppid,user,stat,etime,comm"]
    ):
        return command
    # App config-file editing: `docker cp` one declared config file between a
    # managed container and a workloads-root temp file. Exactly two shapes, each
    # constrained to a curated template's container name and its own declared
    # file (see `_managed_config_target`) with a bounded host temp file on the
    # other side. No shell, no `docker exec`, no `-` tar stream, and never two
    # host paths or two container paths - one of each, in either direction.
    if len(command) == 4 and command[:2] == ["docker", "cp"]:
        source, target = command[2], command[3]
        if _managed_config_target(source) and _host_config_workfile(target):
            return command
        if _host_config_workfile(source) and _managed_config_target(target):
            return command
    if (
        len(command) == 9
        and command[:3] == ["docker", "compose", "--project-directory"]
        and command[4] == "-f"
        and command[6:] == ["config", "--format", "json"]
    ):
        project = Path(command[3])
        config = Path(command[5])
        if (
            _inside(project)
            and _inside(config)
            and config.parent.resolve() == project.resolve()
            and config.is_file()
        ):
            return command
    # Admin console: run a shell inside ONE managed app container. The endpoint
    # gates this on administrator + CSRF and audits the command; the broker only
    # enforces that the target is a curated template's own container and the argv
    # is exactly `docker exec <container> sh -c <command>` - no -u/user, no added
    # flags, no other container or image, no `-`/host path. The command string is
    # arbitrary and interpreted only by `sh -c` inside that container, which has
    # no host mounts and no privileged/socket access, so the shell is confined.
    if (
        len(command) == 6
        and command[:2] == ["docker", "exec"]
        and _managed_container(command[2])
        and command[3:5] == ["sh", "-c"]
        and isinstance(command[5], str)
    ):
        return command
    # File manager (list/mkdir/delete): exactly
    # `docker exec <managed-container> sh -c <fixed-script> _ <path>`. The
    # script is one of the three module constants (no arbitrary command, unlike
    # the console above), the browsable path is the POSITIONAL argument `$1` (a
    # `_` placeholder fills `$0`), and both the container and the path are
    # re-validated here against the curated template's own data roots, so the
    # operation is confined to that app's data even though `sh -c` interprets
    # the fixed script inside the container.
    if (
        len(command) == 8
        and command[:2] == ["docker", "exec"]
        and _managed_container(command[2])
        and command[3:5] == ["sh", "-c"]
        and command[5] in FS_SCRIPTS
        and command[6] == "_"
        and _managed_data_path(command[2], command[7])
    ):
        return command
    # File manager (post-upload ownership): the ONLY root exec this broker
    # admits, and it is deliberately the narrowest shape in this file. `docker
    # cp` lands an uploaded file owned by the control-plane user (uid ~997) mode
    # 0660, which an app running as its own non-root user cannot read; this
    # chowns that one file to its parent data-root directory's owner:group so it
    # becomes native to the app. It is bounded on every axis so it cannot be
    # widened into an arbitrary root command or a write outside the app's data:
    #   - the argv is EXACTLY `docker exec -u 0 <container> sh -c <script> _ f d`
    #     (11 elements); no extra flags, no other `-u` value than root's `0`;
    #   - the script must be `FS_CHOWN_SCRIPT` by EXACT identity - it is not in
    #     `FS_SCRIPTS`, so this branch runs neither the list/mkdir/delete scripts
    #     nor any caller-supplied command, only this single fixed chown;
    #   - `_` fills `$0`; the file (`$1`) and the reference directory (`$2`) are
    #     positional and never interpolated, and BOTH must pass
    #     `_managed_data_path` for the same curated container - so root can only
    #     chown a file inside this app's own data roots, to an owner copied from
    #     another directory inside those same roots, and nothing else.
    if (
        len(command) == 11
        and command[:2] == ["docker", "exec"]
        and command[2:4] == ["-u", "0"]
        and _managed_container(command[4])
        and command[5:7] == ["sh", "-c"]
        and command[7] == FS_CHOWN_SCRIPT
        and command[8] == "_"
        and _managed_data_path(command[4], command[9])
        and _managed_data_path(command[4], command[10])
    ):
        return command
    # File manager (upload/download): `docker cp` one data-root file between a
    # managed container and a workloads-root temp file, in either direction -
    # the same bounded shape as the config-file cp above, but validated against
    # the template's browsable data roots (see `_managed_data_target`).
    if len(command) == 4 and command[:2] == ["docker", "cp"]:
        source, target = command[2], command[3]
        if _managed_data_target(source) and _host_config_workfile(target):
            return command
        if _host_config_workfile(source) and _managed_data_target(target):
            return command
    raise ValueError("Docker operation is not allowed.")


def execute(command: Any, timeout: Any = 8) -> dict[str, Any]:
    validated = _validate(command)
    docker = shutil.which("docker")
    if docker is None:
        return {"returncode": 127, "stdout": "", "stderr": "Docker is unavailable."}
    validated[0] = docker
    try:
        limit = min(max(int(timeout), 1), 30)
    except (TypeError, ValueError):
        limit = 8
    try:
        result = subprocess.run(
            validated,
            capture_output=True,
            text=True,
            check=False,
            timeout=limit,
        )
        return {
            "returncode": result.returncode,
            "stdout": (result.stdout or "")[-MAX_OUTPUT:],
            "stderr": (result.stderr or "")[-MAX_OUTPUT:],
        }
    except subprocess.TimeoutExpired:
        return {"returncode": 124, "stdout": "", "stderr": "Docker operation timed out."}


def delete_checkpoint(checkpoint: Any, confirmation: Any) -> dict[str, Any]:
    if not isinstance(checkpoint, str) or not re.fullmatch(
        r"[a-z0-9_-]+-\d+\.tar\.gz", checkpoint
    ):
        raise ValueError("Choose a valid managed checkpoint.")
    match = re.fullmatch(r"(.+)-\d+\.tar\.gz", checkpoint)
    project = match.group(1) if match else ""
    if confirmation != project:
        raise ValueError("Type the project name to confirm checkpoint deletion.")
    path = (BACKUPS_ROOT / checkpoint).resolve()
    if BACKUPS_ROOT not in path.parents or not path.is_file():
        raise ValueError("Checkpoint was not found.")
    recovery_copy = path.with_suffix("").with_suffix(".tar.project")
    path.unlink()
    if recovery_copy.is_file():
        recovery_copy.unlink()
    elif recovery_copy.is_dir():
        shutil.rmtree(recovery_copy)
    return {"id": checkpoint, "project": project, "deleted": True}


class WorkloadBrokerClient:
    """Small Unix-socket client that preserves subprocess-like results."""

    def __init__(self, socket_path: Path | str = SOCKET_PATH):
        self.socket_path = str(socket_path)

    def _request(self, payload: dict[str, Any], timeout: int = 8) -> dict[str, Any]:
        encoded = json.dumps(payload, separators=(",", ":")).encode() + b"\n"
        try:
            with socket.socket(socket.AF_UNIX, socket.SOCK_STREAM) as client:
                client.settimeout(min(max(timeout, 1), 30) + 2)
                client.connect(self.socket_path)
                client.sendall(encoded)
                response = b""
                while b"\n" not in response and len(response) <= MAX_OUTPUT * 2 + 4096:
                    chunk = client.recv(65536)
                    if not chunk:
                        break
                    response += chunk
            return json.loads(response.split(b"\n", 1)[0])
        except (OSError, ValueError, json.JSONDecodeError, AttributeError) as error:
            # Name the hop (#141 review): a bare errno here used to surface as
            # "Vaelor could not read its cluster state: [Errno 2] …", which
            # reads as a Docker fault when the thing that did not answer is
            # this broker's socket.
            return {
                "returncode": 1,
                "stdout": "",
                "stderr": "The workload broker did not answer: {}".format(error),
            }

    def run(self, command: list[str], timeout: int = 8):
        result = self._request(
            {"action": "docker", "command": command, "timeout": timeout}, timeout
        )
        try:
            return SimpleNamespace(
                returncode=int(result.get("returncode", 1)),
                stdout=str(result.get("stdout", "")),
                stderr=str(result.get("stderr", "")),
            )
        except (TypeError, ValueError):
            return SimpleNamespace(returncode=1, stdout="", stderr="Invalid broker reply.")

    def delete_checkpoint(self, checkpoint: str, confirmation: str) -> dict[str, Any]:
        result = self._request(
            {
                "action": "checkpoint.delete",
                "checkpoint": checkpoint,
                "confirmation": confirmation,
            }
        )
        if result.get("returncode", 1) != 0:
            raise ValueError(str(result.get("stderr", "Checkpoint deletion failed.")))
        value = result.get("result")
        if not isinstance(value, dict):
            raise ValueError("Invalid checkpoint deletion reply.")
        return value


def _serve_connection(connection: socket.socket) -> None:
    request = b""
    while b"\n" not in request and len(request) <= MAX_REQUEST:
        chunk = connection.recv(65536)
        if not chunk:
            break
        request += chunk
    try:
        if len(request) > MAX_REQUEST:
            raise ValueError("Request is too large.")
        payload = json.loads(request.split(b"\n", 1)[0])
        action = payload.get("action", "docker")
        if action == "docker":
            result = execute(payload.get("command"), payload.get("timeout", 8))
        elif action == "checkpoint.delete":
            result = {
                "returncode": 0,
                "result": delete_checkpoint(
                    payload.get("checkpoint"), payload.get("confirmation")
                ),
            }
        else:
            raise ValueError("Broker operation is not allowed.")
    except (OSError, ValueError, json.JSONDecodeError, AttributeError) as error:
        result = {"returncode": 2, "stdout": "", "stderr": str(error)}
    connection.sendall(json.dumps(result, separators=(",", ":")).encode() + b"\n")


def main() -> None:
    SOCKET_PATH.parent.mkdir(parents=True, exist_ok=True)
    SOCKET_PATH.unlink(missing_ok=True)
    with socket.socket(socket.AF_UNIX, socket.SOCK_STREAM) as server:
        server.bind(str(SOCKET_PATH))
        group_id = os.getgid()
        try:
            import grp

            group_id = grp.getgrnam("vaelor-jobs").gr_gid
        except (ImportError, KeyError):
            pass
        os.chown(SOCKET_PATH, os.getuid(), group_id)
        os.chmod(SOCKET_PATH, 0o660)
        server.listen(16)
        while True:
            connection, _ = server.accept()
            with connection:
                _serve_connection(connection)


if __name__ == "__main__":
    main()
