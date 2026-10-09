"""Unix-socket server entrypoint for the encrypted credential vault."""

from __future__ import annotations

import argparse
import json
import logging
import os
import socketserver
from pathlib import Path
from typing import Any, Dict, Optional

from .credential_broker import (
    MAX_REQUEST_BYTES,
    REQUEST_TOO_LARGE,
    CredentialError,
    CredentialVault,
)
from .credential_broker_client import VERB_FAILED
from .credential_use import record_use
from .runtime_paths import env_value, run_path, state_path

LOGGER = logging.getLogger(__name__)

#: A verb that raised anything but a `CredentialError` answers with
#: `credential_broker_client.VERB_FAILED` (its owner, where the client reads it
#: as "no answer, retry"). The traceback goes to the journal; the client gets a
#: sentence, never an empty frame - an empty frame is what the first live
#: ``FileNotFoundError`` inside ``vault.list`` produced, and it reached the AI
#: Chat page as an HTTP 500 rather than as the degraded answer a broker error
#: is (VD-127).


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
        "mint": lambda: vault.mint(
            payload.get("endpoint_id", ""), payload.get("label", ""),
        ),
        "rotate": lambda: vault.rotate(
            payload.get("credential_id", ""), payload.get("endpoint_id", ""),
        ),
        "revoke": lambda: vault.revoke(
            payload.get("credential_id", ""), payload.get("endpoint_id", ""),
        ),
        # F3b-ii: the executor's gate renderer sources its key SET over the socket
        # (design section 2's "a socket verb is added THEN"), and the control-plane
        # migration imports the pre-F3b-ii key over it. Both hand plaintext only to
        # an in-group caller reaching the 0o660 socket, the same trust boundary
        # ``resolve`` leases a secret across; ``import_key`` refuses a populated
        # endpoint, so it can only ever establish a first key, never plant a second.
        "endpoint_keys": lambda: {
            "keys": vault.endpoint_keys(payload.get("endpoint_id", "")),
        },
        "import_key": lambda: vault.import_key(
            payload.get("endpoint_id", ""), payload.get("key", ""),
            payload.get("label", ""),
        ),
        # ACC-107 / ACC-043: the one writer of `last_used_at` for a real use. It
        # decrypts nothing and only moves the stamp forward, so an in-group
        # caller can at worst make a key look recently used - never read one.
        "record_use": lambda: record_use(
            vault, payload.get("credential_id", ""), payload.get("used_at"),
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
                raise CredentialError(REQUEST_TOO_LARGE)
            data = dispatch(self.server.vault, json.loads(raw.decode("utf-8")))
            response = {"ok": True, "data": data}
        except (CredentialError, json.JSONDecodeError, UnicodeDecodeError) as error:
            response = {"ok": False, "error": str(error)[:240]}
        except Exception:  # noqa: BLE001 - the verb's failure is the answer, not silence
            LOGGER.exception("A credential broker operation failed.")
            response = {"ok": False, "error": VERB_FAILED}
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
