"""Encrypted credential vault and least-privilege Unix-socket broker."""

from __future__ import annotations

import hashlib
import ipaddress
import json
import os
import secrets
import socket
import sqlite3
import threading
import time
import urllib.error
import urllib.parse
import urllib.request
from contextlib import closing
from pathlib import Path
from typing import Any, Callable, Dict, Optional

from cryptography.hazmat.primitives.ciphers.aead import AESGCM
from cryptography.exceptions import InvalidTag

from .credential_profiles import validate_ssh_profile as _validate_ssh_profile
from .runtime_paths import env_value, run_path


MAX_SECRET_BYTES = 8192
MAX_REQUEST_BYTES = 16384
MAX_PROVIDER_RESPONSE_BYTES = 1024 * 1024
PROVIDERS = {
    "openai": {
        "name": "OpenAI API",
        "auth": "api_key",
        "test_url": "https://api.openai.com/v1/models",
        "header": "Authorization",
        "prefix": "Bearer ",
    },
    "huggingface": {
        "name": "Hugging Face",
        "auth": "access_token",
        "test_url": "https://huggingface.co/api/whoami-v2",
        "header": "Authorization",
        "prefix": "Bearer ",
    },
    "openai-compatible": {
        "name": "OpenAI-compatible server",
        "auth": "optional_api_key",
    },
    "ssh": {
        "name": "Cluster node SSH",
        "auth": "password_or_private_key",
    },
    "application-secret": {
        "name": "Application secret",
        "auth": "managed_secret",
    },
}

ASSIGNMENT_PROVIDERS = {
    "deployment-agent": {"openai-compatible", "openai"},
    "ai-chat": {"openai-compatible", "openai"},
    "model-download": {"huggingface"},
    "hosted-agent": {"openai"},
    "cluster-node": {"ssh"},
    "cluster-inference": {"openai-compatible"},
    "application-deploy": {"application-secret"},
    "custom-agent-connector": {"application-secret"},
    "backup-passphrase": {"application-secret"},
    "backup-offsite": {"application-secret"},
}
ASSIGNMENT_PURPOSES = set(ASSIGNMENT_PROVIDERS)

#: Alert channels each bind one managed secret (an SMTP password or webhook
#: token) under a per-channel purpose ``alert-channel-<id>`` - a prefix family,
#: not a fixed name, since channels are created at runtime. All are backed by
#: the ``application-secret`` provider, so no new provider kind is introduced.
ALERT_PURPOSE_PREFIX = "alert-channel-"


def purpose_provider_set(purpose: str) -> Optional[set]:
    """The providers a purpose may bind, or ``None`` if unsupported.

    A bounded prefix admits the alert-channel family without letting a caller
    smuggle an arbitrary string past the fixed assignment gate.
    """
    if purpose in ASSIGNMENT_PROVIDERS:
        return ASSIGNMENT_PROVIDERS[purpose]
    if purpose.startswith(ALERT_PURPOSE_PREFIX) and 0 < len(purpose) <= 80:
        return {"application-secret"}
    return None

NON_CHAT_MODEL_MARKERS = (
    "embed", "embedding", "rerank", "reranker", "whisper", "transcrib",
    "speech", "tts", "text-to-speech", "image", "vision-encoder",
)


def _is_chat_model_id(model_id: str) -> bool:
    """Conservatively exclude model IDs that clearly describe non-chat engines."""
    lower = str(model_id).strip().lower()
    return bool(lower) and not any(marker in lower for marker in NON_CHAT_MODEL_MARKERS)


class CredentialError(ValueError):
    """Safe error suitable for returning through the local broker protocol."""


class _NoRedirect(urllib.request.HTTPRedirectHandler):
    def redirect_request(self, request, file_pointer, code, message, headers, new_url):
        return None


def validate_compatible_profile(secret_value: str) -> Dict[str, str]:
    """Validate and normalize an encrypted compatible-endpoint profile."""
    try:
        profile = json.loads(secret_value)
    except (TypeError, json.JSONDecodeError) as error:
        raise CredentialError("The compatible server profile is invalid.") from error
    if not isinstance(profile, dict):
        raise CredentialError("The compatible server profile is invalid.")

    base_url = str(profile.get("base_url", "")).strip().rstrip("/")
    model = str(profile.get("model", "")).strip()
    api_key = str(profile.get("api_key", "")).strip()
    if len(base_url) > 500 or len(model) > 200 or len(api_key) > MAX_SECRET_BYTES:
        raise CredentialError("The compatible server profile is too large.")

    parsed = urllib.parse.urlsplit(base_url)
    if (
        parsed.scheme not in {"http", "https"}
        or not parsed.hostname
        or parsed.username
        or parsed.password
        or parsed.query
        or parsed.fragment
    ):
        raise CredentialError("Enter a complete HTTP or HTTPS server URL without credentials or query text.")
    try:
        port = parsed.port
    except ValueError as error:
        raise CredentialError("The compatible server port is invalid.") from error
    if port is not None and not 1 <= port <= 65535:
        raise CredentialError("The compatible server port is invalid.")

    try:
        resolved = {
            ipaddress.ip_address(item[4][0])
            for item in socket.getaddrinfo(parsed.hostname, port or (443 if parsed.scheme == "https" else 80))
        }
    except (OSError, ValueError) as error:
        raise CredentialError("The compatible server address could not be resolved.") from error
    if not resolved or any(
        not (address.is_private or address.is_loopback)
        or address.is_link_local
        or address.is_multicast
        or address.is_unspecified
        for address in resolved
    ):
        raise CredentialError("For safety, compatible servers must use a private LAN or loopback address.")

    return {"base_url": base_url, "model": model, "api_key": api_key}


def validate_ssh_profile(secret_value: str) -> Dict[str, Any]:
    return _validate_ssh_profile(secret_value, CredentialError)


def _compatible_request(
    profile: Dict[str, str],
    path: str,
    *,
    payload: Optional[Dict[str, Any]] = None,
    timeout: int = 20,
) -> Dict[str, Any]:
    headers = {
        "Accept": "application/json",
        "User-Agent": "Vaelor-Credential-Broker/1",
    }
    if profile["api_key"]:
        headers["Authorization"] = "Bearer {}".format(profile["api_key"])
    data = None
    method = "GET"
    if payload is not None:
        data = json.dumps(payload, separators=(",", ":")).encode("utf-8")
        headers["Content-Type"] = "application/json"
        method = "POST"
    request = urllib.request.Request(
        "{}{}".format(profile["base_url"], path),
        data=data,
        headers=headers,
        method=method,
    )
    opener = urllib.request.build_opener(_NoRedirect)
    with opener.open(request, timeout=timeout) as response:
        if not 200 <= response.status < 300:
            raise CredentialError("The compatible server rejected the connection test.")
        raw = response.read(1024 * 1024 + 1)
    if len(raw) > 1024 * 1024:
        raise CredentialError("The compatible server response was too large.")
    try:
        result = json.loads(raw.decode("utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError) as error:
        raise CredentialError("The compatible server did not return valid JSON.") from error
    if not isinstance(result, dict):
        raise CredentialError("The compatible server returned an invalid response.")
    return result


class CredentialVault:
    """AES-GCM encrypted SQLite vault. No public method returns plaintext."""

    def __init__(
        self,
        database_path: str,
        master_key: bytes,
        provider_tester: Optional[Callable[[str, str], Dict[str, Any]]] = None,
    ):
        if len(master_key) != 32:
            raise CredentialError("Credential master key must be exactly 32 bytes.")
        self.database_path = database_path
        self._cipher = AESGCM(master_key)
        self._provider_tester = provider_tester or self._test_provider
        self._schema_ready = False
        self._schema_lock = threading.Lock()
        self._connect().close()

    def _connect(self) -> sqlite3.Connection:
        path = Path(self.database_path)
        path.parent.mkdir(parents=True, exist_ok=True)
        try:
            descriptor = os.open(str(path), os.O_CREAT | os.O_EXCL | os.O_RDWR, 0o600)
        except FileExistsError:
            pass
        else:
            os.close(descriptor)
        os.chmod(path, 0o600)
        connection = sqlite3.connect(str(path), timeout=10)
        connection.row_factory = sqlite3.Row
        connection.execute("PRAGMA foreign_keys = ON")  # pairs-with: sqlite-foreign-keys-spaced
        self._ensure_schema(connection)
        for suffix in ("", "-wal", "-shm"):
            candidate = Path("{}{}".format(path, suffix))
            if candidate.exists():
                os.chmod(candidate, 0o600)
        return connection

    def _ensure_schema(self, connection: sqlite3.Connection) -> None:
        if self._schema_ready:
            return
        with self._schema_lock:
            if self._schema_ready:
                return
            connection.execute("PRAGMA journal_mode = WAL")  # pairs-with: sqlite-journal-mode-spaced
            connection.executescript(
                """
            CREATE TABLE IF NOT EXISTS credentials (
                id TEXT PRIMARY KEY,
                provider TEXT NOT NULL,
                label TEXT NOT NULL,
                fingerprint TEXT NOT NULL,
                nonce BLOB NOT NULL,
                ciphertext BLOB NOT NULL,
                version INTEGER NOT NULL DEFAULT 1,
                owner TEXT NOT NULL DEFAULT '',
                created_at INTEGER NOT NULL,
                updated_at INTEGER NOT NULL,
                last_used_at INTEGER,
                last_test_status TEXT,
                last_tested_at INTEGER
            );

            CREATE TABLE IF NOT EXISTS broker_audit (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                created_at INTEGER NOT NULL,
                action TEXT NOT NULL,
                target TEXT NOT NULL,
                provider TEXT NOT NULL,
                result TEXT NOT NULL
            );

            CREATE TABLE IF NOT EXISTS credential_assignments (
                purpose TEXT PRIMARY KEY,
                credential_id TEXT NOT NULL,
                updated_at INTEGER NOT NULL,
                FOREIGN KEY (credential_id) REFERENCES credentials(id)
            );

            CREATE TABLE IF NOT EXISTS credential_model_preferences (
                credential_id TEXT PRIMARY KEY,
                selected_model TEXT NOT NULL,
                updated_at INTEGER NOT NULL,
                FOREIGN KEY (credential_id) REFERENCES credentials(id)
            );
            """
            )
            columns = {
                row[1] for row in connection.execute(
                    "PRAGMA table_info(credentials)"
                ).fetchall()
            }
            if "owner" not in columns:
                connection.execute(
                    "ALTER TABLE credentials ADD COLUMN owner TEXT NOT NULL DEFAULT ''"
                )
            if "last_tested_at" not in columns:
                # #143: `last_test_status` alone let a pass from weeks ago
                # wear the same green as one from a minute ago. A result is
                # only readable next to when it was measured; existing rows
                # keep NULL, which the UI reads as "date not recorded".
                connection.execute(
                    "ALTER TABLE credentials ADD COLUMN last_tested_at INTEGER"
                )
            connection.commit()
            self._schema_ready = True

    def _audit(
        self, action: str, target: str = "", provider: str = "", result: str = "success"
    ) -> None:
        with closing(self._connect()) as connection:
            connection.execute(
                """
                INSERT INTO broker_audit
                    (created_at, action, target, provider, result)
                VALUES (?, ?, ?, ?, ?)
                """,
                (
                    int(time.time()), action[:80], target[:80],
                    provider[:40], result[:20],
                ),
            )
            connection.commit()

    @staticmethod
    def capabilities() -> Dict[str, Any]:
        return {
            "ready": True,
            "providers": [
                {
                    "id": provider_id,
                    "name": policy["name"],
                    "auth": policy["auth"],
                    "oauth": False,
                }
                for provider_id, policy in PROVIDERS.items()
            ],
            "plaintext_export": False,
        }

    @staticmethod
    def _validate_provider(provider: str) -> str:
        clean = str(provider).strip().lower()
        if clean not in PROVIDERS:
            raise CredentialError("This provider is not supported by the credential broker.")
        return clean

    @staticmethod
    def _metadata(
        row: sqlite3.Row, active_for: Optional[list[str]] = None
    ) -> Dict[str, Any]:
        metadata = {
            "id": row["id"],
            "provider": row["provider"],
            "label": row["label"],
            "fingerprint": row["fingerprint"],
            "version": int(row["version"]),
            "owner": row["owner"],
            "created_at": row["created_at"],
            "updated_at": row["updated_at"],
            "last_used_at": row["last_used_at"],
            "last_test_status": row["last_test_status"],
            "last_tested_at": row["last_tested_at"],
        }
        metadata["active_for"] = active_for or []
        return metadata

    @staticmethod
    def _associated_data(credential_id: str, provider: str, version: int) -> bytes:
        return "{}:{}:{}".format(credential_id, provider, version).encode("utf-8")

    def list(self, actor: Optional[str] = None) -> list[Dict[str, Any]]:
        clean_actor = str(actor or "").strip()
        with closing(self._connect()) as connection:
            query = (
                """
                SELECT id, provider, label, fingerprint, version, owner,
                       created_at, updated_at,
                       last_used_at, last_test_status, last_tested_at
                FROM credentials
                """
            )
            values = []
            if clean_actor:
                query += " WHERE provider!='application-secret' OR owner=?"
                values.append(clean_actor)
            query += " ORDER BY created_at DESC"
            rows = connection.execute(query, values).fetchall()
            assignments = connection.execute(
                "SELECT purpose, credential_id FROM credential_assignments"
            ).fetchall()
            model_preferences = {
                row["credential_id"]: row["selected_model"]
                for row in connection.execute(
                    "SELECT credential_id, selected_model FROM credential_model_preferences"
                ).fetchall()
            }
        active = {}
        for assignment in assignments:
            active.setdefault(assignment["credential_id"], []).append(assignment["purpose"])
        result = []
        for row in rows:
            item = self._metadata(row, active.get(row["id"], []))
            item["selected_model"] = model_preferences.get(row["id"], "")
            result.append(item)
        self._audit("credential.list")
        return result

    def put(
        self,
        provider: str,
        label: str,
        secret_value: str,
        credential_id: Optional[str] = None,
        owner: str = "",
    ) -> Dict[str, Any]:
        clean_provider = self._validate_provider(provider)
        clean_label = str(label).strip()[:80] or PROVIDERS[clean_provider]["name"]
        secret_bytes = str(secret_value).strip().encode("utf-8")
        if len(secret_bytes) < 8 or len(secret_bytes) > MAX_SECRET_BYTES:
            raise CredentialError("Credential must contain between 8 and 8,192 bytes.")
        if clean_provider == "openai-compatible":
            validate_compatible_profile(secret_bytes.decode("utf-8"))
        elif clean_provider == "ssh":
            validate_ssh_profile(secret_bytes.decode("utf-8"))
        now = int(time.time())
        item_id = credential_id or "cred_{}".format(secrets.token_hex(12))
        clean_owner = str(owner or "").strip()[:120]
        if clean_provider == "application-secret" and not clean_owner:
            raise CredentialError("Application secrets require an owning administrator.")
        version = 1
        with closing(self._connect()) as connection:
            existing = connection.execute(
                "SELECT provider, version, owner, created_at FROM credentials WHERE id = ?",
                (item_id,),
            ).fetchone()
            if existing is not None:
                if existing["provider"] != clean_provider:
                    raise CredentialError("A credential cannot change providers during rotation.")
                if (
                    clean_provider == "application-secret"
                    and existing["owner"] != clean_owner
                ):
                    raise CredentialError("Only the owning administrator can rotate this application secret.")
                version = int(existing["version"]) + 1
                created_at = int(existing["created_at"])
            else:
                created_at = now
            nonce = os.urandom(12)
            ciphertext = self._cipher.encrypt(
                nonce,
                secret_bytes,
                self._associated_data(item_id, clean_provider, version),
            )
            fingerprint = hashlib.sha256(secret_bytes).hexdigest()[-8:].upper()
            connection.execute(
                """
                INSERT INTO credentials (
                    id, provider, label, fingerprint, nonce, ciphertext, version, owner,
                    created_at, updated_at, last_used_at, last_test_status
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, NULL, NULL)
                ON CONFLICT(id) DO UPDATE SET
                    label=excluded.label, fingerprint=excluded.fingerprint,
                    nonce=excluded.nonce, ciphertext=excluded.ciphertext,
                    version=excluded.version, updated_at=excluded.updated_at,
                    last_test_status=NULL, last_tested_at=NULL
                """,
                (
                    item_id, clean_provider, clean_label, fingerprint, nonce,
                    ciphertext, version, clean_owner, created_at, now,
                ),
            )
            connection.commit()
            row = connection.execute(
                """
                SELECT id, provider, label, fingerprint, version, owner,
                       created_at, updated_at,
                       last_used_at, last_test_status, last_tested_at
                FROM credentials WHERE id = ?
                """,
                (item_id,),
            ).fetchone()
        metadata = self._metadata(row)
        self._audit(
            "credential.rotate" if existing is not None else "credential.store",
            item_id,
            clean_provider,
        )
        return metadata

    def delete(self, credential_id: str) -> bool:
        with closing(self._connect()) as connection:
            connection.execute(
                "DELETE FROM credential_assignments WHERE credential_id = ?",
                (str(credential_id),),
            )
            connection.execute(
                "DELETE FROM credential_model_preferences WHERE credential_id = ?",
                (str(credential_id),),
            )
            cursor = connection.execute(
                "DELETE FROM credentials WHERE id = ?", (str(credential_id),)
            )
            connection.commit()
        deleted = cursor.rowcount == 1
        self._audit(
            "credential.delete", str(credential_id),
            result="success" if deleted else "not_found",
        )
        return deleted

    def activate(self, credential_id: str, purpose: str) -> Dict[str, Any]:
        clean_purpose = str(purpose).strip().lower()
        providers = purpose_provider_set(clean_purpose)
        if providers is None:
            raise CredentialError("This credential assignment is not supported.")
        with closing(self._connect()) as connection:
            row = connection.execute(
                "SELECT id, provider FROM credentials WHERE id = ?",
                (str(credential_id),),
            ).fetchone()
            if row is None:
                raise CredentialError("Credential was not found.")
            if row["provider"] not in providers:
                raise CredentialError(
                    "This credential provider cannot be used for that purpose."
                )
            connection.execute(
                """
                INSERT INTO credential_assignments (purpose, credential_id, updated_at)
                VALUES (?, ?, ?)
                ON CONFLICT(purpose) DO UPDATE SET
                    credential_id=excluded.credential_id,
                    updated_at=excluded.updated_at
                """,
                (clean_purpose, row["id"], int(time.time())),
            )
            connection.commit()
        self._audit("credential.activate", row["id"], row["provider"])
        return {"credential_id": row["id"], "purpose": clean_purpose}

    def deactivate(self, purpose: str) -> bool:
        clean_purpose = str(purpose).strip().lower()
        if purpose_provider_set(clean_purpose) is None:
            raise CredentialError("This credential assignment is not supported.")
        with closing(self._connect()) as connection:
            cursor = connection.execute(
                "DELETE FROM credential_assignments WHERE purpose = ?",
                (clean_purpose,),
            )
            connection.commit()
        removed = cursor.rowcount == 1
        self._audit(
            "credential.deactivate",
            clean_purpose,
            result="success" if removed else "not_found",
        )
        return removed

    def resolve_active(self, purpose: str) -> Dict[str, Any]:
        """Return one purpose-bound lease. This is never exposed by the HTTP API."""
        clean_purpose = str(purpose).strip().lower()
        if purpose_provider_set(clean_purpose) is None:
            raise CredentialError("This credential assignment is not supported.")
        with closing(self._connect()) as connection:
            row = connection.execute(
                """
                SELECT c.id, c.provider, c.label, c.nonce, c.ciphertext, c.version,
                       COALESCE(p.selected_model, '') AS selected_model
                FROM credential_assignments a
                JOIN credentials c ON c.id = a.credential_id
                LEFT JOIN credential_model_preferences p
                    ON p.credential_id = c.id
                WHERE a.purpose = ?
                """,
                (clean_purpose,),
            ).fetchone()
        if row is None:
            raise CredentialError("No credential is active for this purpose.")
        plaintext = self._decrypt(row)
        try:
            profile = (
                validate_compatible_profile(plaintext)
                if row["provider"] == "openai-compatible"
                else {"token": plaintext}
            )
        finally:
            plaintext = ""
        if row["selected_model"]:
            profile["model"] = row["selected_model"]
        self._audit("credential.lease", row["id"], row["provider"])
        return {
            **profile,
            "credential_id": row["id"],
            "label": row["label"],
            "provider": row["provider"],
        }

    def resolve(
        self, credential_id: str, purpose: str, actor: str = ""
    ) -> Dict[str, Any]:
        """Return a credential-specific purpose lease to trusted local services."""
        clean_purpose = str(purpose).strip().lower()
        providers = purpose_provider_set(clean_purpose)
        if providers is None:
            raise CredentialError("This credential assignment is not supported.")
        with closing(self._connect()) as connection:
            row = connection.execute(
                """
                SELECT id, provider, label, nonce, ciphertext, version, owner
                FROM credentials WHERE id = ?
                """,
                (str(credential_id),),
            ).fetchone()
        if row is None:
            raise CredentialError("Credential was not found.")
        if row["provider"] not in providers:
            raise CredentialError("This credential provider cannot be used for that purpose.")
        if (
            clean_purpose == "custom-agent-connector"
            and (not str(actor).strip() or row["owner"] != str(actor).strip())
        ):
            raise CredentialError("This application secret is not owned by the requesting administrator.")
        plaintext = self._decrypt(row)
        try:
            profile = (
                validate_ssh_profile(plaintext)
                if row["provider"] == "ssh"
                else validate_compatible_profile(plaintext)
                if row["provider"] == "openai-compatible"
                else {"token": plaintext}
            )
        finally:
            plaintext = ""
        self._audit("credential.lease", row["id"], row["provider"])
        return {
            **profile,
            "credential_id": row["id"],
            "label": row["label"],
            "provider": row["provider"],
            "credential_version": int(row["version"]),
        }

    def models(self, credential_id: str) -> Dict[str, Any]:
        with closing(self._connect()) as connection:
            row = connection.execute(
                """
                SELECT c.id, c.provider, c.nonce, c.ciphertext, c.version,
                       COALESCE(p.selected_model, '') AS selected_model
                FROM credentials c
                LEFT JOIN credential_model_preferences p
                    ON p.credential_id = c.id
                WHERE c.id = ?
                """,
                (str(credential_id),),
            ).fetchone()
        if row is None:
            raise CredentialError("Credential was not found.")
        if row["provider"] not in {"openai", "openai-compatible"}:
            raise CredentialError("This credential does not provide chat models.")
        plaintext = self._decrypt(row)
        try:
            if row["provider"] == "openai-compatible":
                profile = validate_compatible_profile(plaintext)
                payload = _compatible_request(profile, "/models")
            else:
                request = urllib.request.Request(
                    PROVIDERS["openai"]["test_url"],
                    headers={
                        "Authorization": "Bearer {}".format(plaintext),
                        "Accept": "application/json",
                        "User-Agent": "Vaelor-Credential-Broker/1",
                    },
                )
                with urllib.request.urlopen(request, timeout=20) as response:
                    payload = json.loads(
                        response.read(MAX_PROVIDER_RESPONSE_BYTES).decode("utf-8")
                    )
        except urllib.error.HTTPError as error:
            if error.code in {401, 403}:
                raise CredentialError("The provider rejected this credential.") from error
            raise CredentialError(
                "The provider returned HTTP {}.".format(error.code)
            ) from error
        except (urllib.error.URLError, TimeoutError, ValueError) as error:
            raise CredentialError("The model list could not be loaded.") from error
        finally:
            plaintext = ""
        models = sorted({
            str(item.get("id", "")).strip()
            for item in payload.get("data", [])
            if (
                isinstance(item, dict)
                and _is_chat_model_id(str(item.get("id", "")).strip())
            )
        })
        if row["provider"] == "openai":
            models = [
                model for model in models
                if model.startswith(("gpt-", "o1", "o3", "o4"))
                and not any(part in model for part in (
                    "audio", "image", "realtime", "search", "transcribe", "tts"
                ))
            ]
        if not models:
            raise CredentialError("The provider did not report any usable chat models.")
        selected_model = str(row["selected_model"] or "")
        selected_model_available = bool(
            selected_model and selected_model in models
        )
        return {
            "credential_id": row["id"],
            "provider": row["provider"],
            "models": models[:200],
            "selected_model": selected_model,
            "selected_model_available": selected_model_available,
            "selection_required": (
                row["provider"] == "openai-compatible"
                and bool(models)
                and not selected_model_available
            ),
        }

    def select_model(self, credential_id: str, model: str) -> Dict[str, Any]:
        selected = str(model).strip()
        if not selected or len(selected) > 200:
            raise CredentialError("Choose a valid model.")
        available = self.models(credential_id)
        if selected not in available["models"]:
            raise CredentialError("The selected model is not available from this provider.")
        if available["provider"] == "openai-compatible":
            with closing(self._connect()) as connection:
                row = connection.execute(
                    """
                    SELECT id, provider, nonce, ciphertext, version
                    FROM credentials WHERE id = ?
                    """,
                    (str(credential_id),),
                ).fetchone()
            plaintext = self._decrypt(row)
            try:
                profile = validate_compatible_profile(plaintext)
                profile["model"] = selected
                result = self._provider_tester(
                    "openai-compatible",
                    json.dumps(profile, separators=(",", ":")),
                )
            finally:
                plaintext = ""
            if not result.get("ok"):
                raise CredentialError(
                    str(result.get("message", "The selected model did not pass its chat test."))
                )
        with closing(self._connect()) as connection:
            connection.execute(
                """
                INSERT INTO credential_model_preferences (
                    credential_id, selected_model, updated_at
                ) VALUES (?, ?, ?)
                ON CONFLICT(credential_id) DO UPDATE SET
                    selected_model=excluded.selected_model,
                    updated_at=excluded.updated_at
                """,
                (str(credential_id), selected, int(time.time())),
            )
            connection.commit()
        self._audit(
            "credential.model.select", str(credential_id), available["provider"]
        )
        return {
            "credential_id": str(credential_id),
            "provider": available["provider"],
            "selected_model": selected,
        }

    def _decrypt(self, row: sqlite3.Row) -> str:
        try:
            return self._cipher.decrypt(
                row["nonce"],
                row["ciphertext"],
                self._associated_data(row["id"], row["provider"], row["version"]),
            ).decode("utf-8")
        except (InvalidTag, UnicodeDecodeError) as error:
            raise CredentialError("Credential could not be decrypted.") from error

    def test(self, credential_id: str) -> Dict[str, Any]:
        with closing(self._connect()) as connection:
            row = connection.execute(
                """
                SELECT c.id, c.provider, c.nonce, c.ciphertext, c.version,
                       COALESCE(p.selected_model, '') AS selected_model
                FROM credentials c
                LEFT JOIN credential_model_preferences p
                    ON p.credential_id = c.id
                WHERE c.id = ?
                """,
                (str(credential_id),),
            ).fetchone()
        if row is None:
            raise CredentialError("Credential was not found.")
        try:
            plaintext = self._decrypt(row)
        except CredentialError as error:
            self._audit(
                "credential.test", row["id"], row["provider"], "decrypt_failed"
            )
            raise error
        try:
            if row["provider"] == "openai-compatible" and row["selected_model"]:
                profile = validate_compatible_profile(plaintext)
                profile["model"] = row["selected_model"]
                plaintext = json.dumps(profile, separators=(",", ":"))
            result = self._provider_tester(row["provider"], plaintext)
            status = "success" if result.get("ok") else "failed"
        finally:
            plaintext = ""
        now = int(time.time())
        with closing(self._connect()) as connection:
            connection.execute(
                """
                UPDATE credentials
                SET last_used_at = ?, last_test_status = ?, last_tested_at = ?
                WHERE id = ?
                """,
                (now, status, now, row["id"]),
            )
            connection.commit()
        self._audit("credential.test", row["id"], row["provider"], status)
        return {
            "id": row["id"],
            "provider": row["provider"],
            "ok": status == "success",
            "message": str(result.get("message", "Connection test completed."))[:240],
            "tested_at": now,
        }

    @staticmethod
    def _test_provider(provider: str, secret_value: str) -> Dict[str, Any]:
        if provider == "application-secret":
            return {
                "ok": True,
                "message": "The encrypted application secret is available to approved deployments.",
            }
        if provider == "openai-compatible":
            profile = validate_compatible_profile(secret_value)
            try:
                model_data = _compatible_request(profile, "/models")
                models = model_data.get("data", [])
                if not isinstance(models, list) or not models:
                    return {"ok": False, "message": "The server did not report any loaded models."}
                selected_model = profile["model"] or str(models[0].get("id", "")).strip()
                if not selected_model:
                    return {"ok": False, "message": "The server did not report a usable model ID."}
                if profile["model"] and not any(
                    isinstance(item, dict) and item.get("id") == selected_model
                    for item in models
                ):
                    return {"ok": False, "message": "The selected model is not loaded on this server."}
                chat = _compatible_request(
                    profile,
                    "/chat/completions",
                    payload={
                        "model": selected_model,
                        "messages": [{"role": "user", "content": "Reply with OK."}],
                        "temperature": 0,
                        "max_tokens": 2,
                    },
                    timeout=45,
                )
                if not isinstance(chat.get("choices"), list):
                    return {"ok": False, "message": "The chat-completions response was not compatible."}
                return {
                    "ok": True,
                    "message": "Connected to {}. Models and chat completions are working.".format(selected_model),
                }
            except urllib.error.HTTPError as error:
                if error.code in {401, 403}:
                    return {"ok": False, "message": "The server rejected the API key."}
                return {"ok": False, "message": "The server returned HTTP {}.".format(error.code)}
            except (urllib.error.URLError, TimeoutError):
                return {"ok": False, "message": "The compatible server could not be reached from this Vaelor node."}
        policy = PROVIDERS[provider]
        request = urllib.request.Request(
            policy["test_url"],
            headers={
                policy["header"]: "{}{}".format(policy["prefix"], secret_value),
                "Accept": "application/json",
                "User-Agent": "Vaelor-Credential-Broker/1",
            },
        )
        try:
            with urllib.request.urlopen(request, timeout=12) as response:
                ok = 200 <= response.status < 300
            return {
                "ok": ok,
                "message": "Provider accepted this credential." if ok else "Provider rejected this credential.",
            }
        except urllib.error.HTTPError as error:
            if error.code in {401, 403}:
                return {"ok": False, "message": "Provider rejected this credential."}
            return {"ok": False, "message": "Provider returned HTTP {}.".format(error.code)}
        except urllib.error.URLError:
            return {"ok": False, "message": "Provider could not be reached from this device."}


class CredentialBrokerClient:
    """Small JSON-over-Unix-socket client. It never offers a plaintext get."""

    def __init__(self, socket_path: Optional[str] = None, timeout_seconds: int = 60):
        self.socket_path = socket_path or env_value(
            "VAELOR_CREDENTIAL_BROKER_SOCKET", "PM_CREDENTIAL_BROKER_SOCKET",
            run_path("credentiald.sock"),
        )
        self.timeout_seconds = timeout_seconds

    def _request(self, operation: str, **payload) -> Any:
        request_body = json.dumps(
            {"operation": operation, "payload": payload}, separators=(",", ":")
        ).encode("utf-8") + b"\n"
        if len(request_body) > MAX_REQUEST_BYTES:
            raise CredentialError("Credential broker request is too large.")
        connection = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
        connection.settimeout(self.timeout_seconds)
        try:
            connection.connect(self.socket_path)
            connection.sendall(request_body)
            response = b""
            while b"\n" not in response and len(response) <= MAX_REQUEST_BYTES:
                chunk = connection.recv(4096)
                if not chunk:
                    break
                response += chunk
        except OSError as error:
            raise CredentialError("Credential broker is unavailable.") from error
        finally:
            connection.close()
        decoded = json.loads(response.split(b"\n", 1)[0].decode("utf-8"))
        if not decoded.get("ok"):
            raise CredentialError(decoded.get("error", "Credential broker rejected the request."))
        return decoded.get("data")

    def capabilities(self):
        return self._request("capabilities")

    def list(self, actor=None):
        return self._request("list", actor=actor)

    def put(self, provider, label, secret_value, credential_id=None, owner=""):
        return self._request(
            "put", provider=provider, label=label, secret=secret_value,
            credential_id=credential_id, owner=owner,
        )

    def delete(self, credential_id):
        return self._request("delete", credential_id=credential_id)

    def test(self, credential_id):
        return self._request("test", credential_id=credential_id)

    def activate(self, credential_id, purpose):
        return self._request(
            "activate", credential_id=credential_id, purpose=purpose
        )

    def deactivate(self, purpose):
        return self._request("deactivate", purpose=purpose)

    def models(self, credential_id):
        return self._request("models", credential_id=credential_id)

    def select_model(self, credential_id, model):
        return self._request(
            "select_model", credential_id=credential_id, model=model
        )

    def resolve_active(self, purpose):
        return self._request("resolve_active", purpose=purpose)

    def resolve(self, credential_id, purpose, actor=""):
        return self._request(
            "resolve", credential_id=credential_id, purpose=purpose, actor=actor
        )


def main():
    from .credential_broker_server import main as server_main
    server_main()


if __name__ == "__main__":
    main()
