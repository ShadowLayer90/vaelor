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
        len(command) == 5
        and command[:3] == ["docker", "logs", "--tail"]
        and command[3].isdigit()
        and 20 <= int(command[3]) <= 500
        and APP_ID.fullmatch(command[4])
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
