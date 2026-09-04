"""Root-isolated, fixed-command broker for Debian system updates."""

from __future__ import annotations

import json
import os
import socket
import socketserver
import subprocess
import time
from pathlib import Path
from typing import Any, Callable, Dict

from .platform_drivers import default_platform_drivers
from .runtime_paths import env_value, jobs_group_id, run_path, state_path

try:
    import grp
except ImportError:  # pragma: no cover - deployment target is Linux
    grp = None


SOCKET_PATH = env_value(
    "VAELOR_SYSTEM_UPDATE_SOCKET", "PM_SYSTEM_UPDATE_SOCKET",
    run_path("system-update.sock"),
)
STATE_PATH = Path(
    env_value(
        "VAELOR_SYSTEM_UPDATE_STATE", "PM_SYSTEM_UPDATE_STATE",
        state_path("system-update-state.json"),
    )
)
MAX_REQUEST_BYTES = 4096
ALLOWED_ACTIONS = {"stage", "apply"}

#: Signatures of a dpkg run that aborted with the package database left
#: half-finished - the recoverable class (a trigger-ordering race such as
#: Ubuntu 26.04's dracut initramfs trigger firing before a staged kernel's
#: modules finished unpacking), as opposed to a real failure like a denied
#: permission, a repo it could not reach, or a held package. Only these get a
#: repair-and-retry pass; everything else is reported as-is so a genuine failure
#: is never dressed up as success (VD-035 honest-degrade).
_RECOVERABLE_MARKERS = (
    "dpkg: error processing",
    "dpkg was interrupted",
    "update-initramfs",
    "dracut",
    "half-configured",
    "--configure -a",
)


def _looks_recoverable(output: str) -> bool:
    text = str(output).lower()
    return any(marker in text for marker in _RECOVERABLE_MARKERS)


def _attempt_recovery(
    package_manager, runner: Callable[..., Any], environment: Dict[str, str]
) -> bool:
    """Finish an interrupted dpkg run, returning True only if it is now whole.

    Runs the driver's idempotent repair commands, then re-reads the audit: the
    upgrade is retried only when the package database reports no half-configured
    package, so a break the repair could not heal still surfaces honestly.
    """
    for command in package_manager.repair_commands():
        runner(
            command,
            capture_output=True,
            text=True,
            check=False,
            timeout=3600,
            env=environment,
        )
    audit = runner(
        package_manager.audit_command(),
        capture_output=True,
        text=True,
        check=False,
        timeout=60,
        env=environment,
    )
    return audit.returncode == 0 and not audit.stdout.strip()


def _packages(package_manager=None, runner: Callable[..., Any] = subprocess.run) -> list[str]:
    package_manager = (
        package_manager or default_platform_drivers()["package_manager"]
    )
    result = runner(
        package_manager.list_upgradable_command(),
        capture_output=True,
        text=True,
        check=False,
        timeout=30,
    )
    return [
        line.split("/", 1)[0]
        for line in result.stdout.splitlines()[1:]
        if "/" in line
    ][:500]


def _write_state(data: Dict[str, Any]) -> None:
    STATE_PATH.parent.mkdir(parents=True, exist_ok=True)
    temporary = STATE_PATH.with_suffix(".tmp")
    temporary.write_text(
        json.dumps(data, separators=(",", ":"), sort_keys=True),
        encoding="utf-8",
    )
    os.chmod(temporary, 0o644)
    temporary.replace(STATE_PATH)


def perform_update(
    action: str,
    package_manager=None,
    runner: Callable[..., Any] = subprocess.run,
) -> Dict[str, Any]:
    if action not in ALLOWED_ACTIONS:
        raise ValueError("Choose stage or apply.")
    package_manager = (
        package_manager or default_platform_drivers()["package_manager"]
    )
    before = _packages(package_manager, runner)
    environment = {
        **os.environ,
        "DEBIAN_FRONTEND": "noninteractive",
        "APT_LISTCHANGES_FRONTEND": "none",
    }
    commands = (
        package_manager.stage_upgrade_commands()
        if action == "stage"
        else package_manager.apply_upgrade_commands()
    )
    output = ""
    for command in commands:
        result = runner(
            command,
            capture_output=True,
            text=True,
            check=False,
            timeout=3600,
            env=environment,
        )
        output = (result.stdout + "\n" + result.stderr)[-8192:]
        if result.returncode != 0:
            # A dpkg run left half-finished by a trigger-ordering race heals
            # with one repair pass; retry the command once and keep going if
            # the database is whole again. Anything the repair cannot heal (or
            # a failure that is not this class) still raises - a real break is
            # never reported as success.
            if _looks_recoverable(output) and _attempt_recovery(
                package_manager, runner, environment
            ):
                result = runner(
                    command,
                    capture_output=True,
                    text=True,
                    check=False,
                    timeout=3600,
                    env=environment,
                )
                output = (result.stdout + "\n" + result.stderr)[-8192:]
            if result.returncode != 0:
                raise RuntimeError(output.strip() or "The package manager failed.")
    after = _packages(package_manager, runner)
    state = {
        "action": action,
        "completed_at": int(time.time() * 1000),
        "packages_before": before,
        "packages_remaining": after,
        "reboot_required": Path("/var/run/reboot-required").exists(),
    }
    _write_state(state)
    return state


class _Handler(socketserver.StreamRequestHandler):
    def handle(self) -> None:
        raw = self.rfile.readline(MAX_REQUEST_BYTES + 1)
        try:
            if len(raw) > MAX_REQUEST_BYTES:
                raise ValueError("Update request is too large.")
            request = json.loads(raw.decode("utf-8"))
            if not isinstance(request, dict) or set(request) != {"action"}:
                raise ValueError("Update request is invalid.")
            result = {"ok": True, "result": perform_update(str(request["action"]))}
        except (
            json.JSONDecodeError,
            OSError,
            subprocess.SubprocessError,
            RuntimeError,
            ValueError,
        ) as error:
            result = {"ok": False, "error": str(error)[:1000]}
        self.wfile.write(
            json.dumps(result, separators=(",", ":")).encode("utf-8") + b"\n"
        )


_UnixServerBase = getattr(
    socketserver, "ThreadingUnixStreamServer", socketserver.ThreadingTCPServer
)


class _Server(_UnixServerBase):
    daemon_threads = True


def serve() -> None:
    socket_path = Path(SOCKET_PATH)
    socket_path.unlink(missing_ok=True)
    server = _Server(str(socket_path), _Handler)
    os.chmod(socket_path, 0o660)
    if grp is not None:
        group_id = jobs_group_id(grp)
        if group_id is not None:
            os.chown(socket_path, 0, group_id)
    try:
        server.serve_forever(poll_interval=0.5)
    finally:
        server.server_close()
        socket_path.unlink(missing_ok=True)


class SystemUpdateClient:
    def __init__(self, socket_path: str = SOCKET_PATH, timeout: int = 3700):
        self.socket_path = socket_path
        self.timeout = timeout

    def run(self, action: str) -> Dict[str, Any]:
        if action not in ALLOWED_ACTIONS:
            raise ValueError("Choose stage or apply.")
        with socket.socket(socket.AF_UNIX, socket.SOCK_STREAM) as connection:
            connection.settimeout(self.timeout)
            connection.connect(self.socket_path)
            connection.sendall(
                json.dumps({"action": action}, separators=(",", ":")).encode("utf-8")
                + b"\n"
            )
            response = b""
            while not response.endswith(b"\n"):
                chunk = connection.recv(4096)
                if not chunk:
                    break
                response += chunk
                if len(response) > 64 * 1024:
                    raise RuntimeError("Update broker response is too large.")
        payload = json.loads(response.decode("utf-8"))
        if not payload.get("ok"):
            raise RuntimeError(payload.get("error") or "System update failed.")
        return payload["result"]


def main() -> None:
    serve()


if __name__ == "__main__":
    main()
