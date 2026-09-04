"""Unix-socket server entrypoint for the encrypted credential vault."""

from __future__ import annotations

import argparse
import json
import os
import socketserver
from pathlib import Path
from typing import Any, Dict, Optional

from .credential_broker import (
    MAX_REQUEST_BYTES,
    CredentialError,
    CredentialVault,
)
from .runtime_paths import env_value, run_path, state_path


def dispatch(vault: CredentialVault, request: Dict[str, Any]) -> Any:
    operation = request.get("operation")
    payload = request.get("payload") or {}
    operations = {
        "capabilities": lambda: vault.capabilities(),
        "list": lambda: vault.list(payload.get("actor")),
        "put": lambda: vault.put(
            payload.get("provider", ""), payload.get("label", ""),
            payload.get("secret", ""), payload.get("credential_id"),
            payload.get("owner", ""),
        ),
        "delete": lambda: {
            "deleted": vault.delete(payload.get("credential_id", ""))
        },
        "test": lambda: vault.test(payload.get("credential_id", "")),
        "activate": lambda: vault.activate(
            payload.get("credential_id", ""), payload.get("purpose", "")
        ),
        "deactivate": lambda: {
            "deactivated": vault.deactivate(payload.get("purpose", ""))
        },
        "models": lambda: vault.models(payload.get("credential_id", "")),
        "select_model": lambda: vault.select_model(
            payload.get("credential_id", ""), payload.get("model", "")
        ),
        "resolve_active": lambda: vault.resolve_active(
            payload.get("purpose", "")
        ),
        "resolve": lambda: vault.resolve(
            payload.get("credential_id", ""), payload.get("purpose", ""),
            payload.get("actor", ""),
        ),
    }
    if operation not in operations:
        raise CredentialError("Unsupported credential broker operation.")
    return operations[operation]()


class CredentialRequestHandler(socketserver.StreamRequestHandler):
    def handle(self):
        raw = self.rfile.readline(MAX_REQUEST_BYTES + 1)
        try:
            if len(raw) > MAX_REQUEST_BYTES:
                raise CredentialError("Credential broker request is too large.")
            data = dispatch(self.server.vault, json.loads(raw.decode("utf-8")))
            response = {"ok": True, "data": data}
        except (CredentialError, json.JSONDecodeError, UnicodeDecodeError) as error:
            response = {"ok": False, "error": str(error)[:240]}
        self.wfile.write(
            json.dumps(response, separators=(",", ":")).encode("utf-8") + b"\n"
        )


if hasattr(socketserver, "UnixStreamServer"):
    class CredentialBrokerServer(
        socketserver.ThreadingMixIn, socketserver.UnixStreamServer
    ):
        daemon_threads = True

        def __init__(self, socket_path: str, vault: CredentialVault):
            path = Path(socket_path)
            path.parent.mkdir(parents=True, exist_ok=True)
            if path.exists():
                path.unlink()
            self.vault = vault
            super().__init__(socket_path, CredentialRequestHandler)
            os.chmod(socket_path, 0o660)
else:
    class CredentialBrokerServer:
        def __init__(self, socket_path: str, vault: CredentialVault):
            raise OSError("Unix-domain sockets are required for the credential broker.")


def read_master_key(path: Optional[str] = None) -> bytes:
    credential_directory = os.environ.get("CREDENTIALS_DIRECTORY", "")
    resolved = path or (
        str(Path(credential_directory) / "master.key")
        if credential_directory else ""
    )
    resolved = resolved or env_value(
        "VAELOR_CREDENTIAL_MASTER_KEY_FILE",
        "PM_CREDENTIAL_MASTER_KEY_FILE",
        "",
    )
    if not resolved:
        raise CredentialError("No credential master key was supplied.")
    key = Path(resolved).read_bytes()
    if len(key) != 32:
        raise CredentialError("Credential master key must be exactly 32 bytes.")
    return key


def main():
    parser = argparse.ArgumentParser(description="Vaelor credential broker")
    parser.add_argument(
        "--socket",
        default=env_value(
            "VAELOR_CREDENTIAL_BROKER_SOCKET", "PM_CREDENTIAL_BROKER_SOCKET",
            run_path("credentiald.sock"),
        ),
    )
    parser.add_argument(
        "--database",
        default=env_value(
            "VAELOR_CREDENTIAL_VAULT_DB", "PM_CREDENTIAL_VAULT_DB",
            state_path("credentials/vault.sqlite3"),
        ),
    )
    parser.add_argument("--master-key-file", default=None)
    arguments = parser.parse_args()
    vault = CredentialVault(
        arguments.database, read_master_key(arguments.master_key_file)
    )
    server = CredentialBrokerServer(arguments.socket, vault)
    try:
        server.serve_forever()
    finally:
        server.server_close()
        try:
            Path(arguments.socket).unlink()
        except FileNotFoundError:
            pass
