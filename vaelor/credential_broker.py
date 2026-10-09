"""Encrypted credential vault and least-privilege Unix-socket broker."""

from __future__ import annotations

import hashlib
import json
import os
import secrets
import sqlite3
import threading
import time
import urllib.error
import urllib.request
from contextlib import closing
from pathlib import Path
from typing import Any, Callable, Dict, Optional

from cryptography.hazmat.primitives.ciphers.aead import AESGCM
from cryptography.exceptions import InvalidTag

# The wire contract and the client live in the leaf module; re-exported here
# because this is the spelling every importer of the broker uses.
from .credential_broker_client import (  # noqa: F401 - re-exported
    CREDENTIAL_NOT_FOUND,
    MAX_REQUEST_BYTES,
    REQUEST_TOO_LARGE,
    CredentialBrokerClient,
    CredentialError,
)
from .credential_profiles import validate_ssh_profile as _validate_ssh_profile
# The connection tests live in their own module (split for the line ceiling);
# `validate_compatible_profile` is re-exported because importers spell it here,
# and `_compatible_request` keeps its old name so `models` reads it from this
# module's namespace exactly as before.
from .credential_provider_probe import (  # noqa: F401 - re-exported
    MAX_SECRET_BYTES,
    compatible_request as _compatible_request,
    probe_provider,
    validate_compatible_profile,
)
from . import served_endpoint_keys
from .hosted_providers import (
    HOSTED_COMPATIBLE,
    HOSTED_KINDS,
    HOSTED_PROVIDERS,
    lease_profile as hosted_lease_profile,
    list_models as hosted_list_models,
    validate_hosted_profile,
)
from .model_credential_roles import (  # noqa: F401 - ASSISTANT_PURPOSE re-exported
    ASSISTANT_PURPOSE,
    role_refusal,
)
from .served_endpoint_keys import SERVED_ENDPOINT_PROVIDER

MAX_PROVIDER_RESPONSE_BYTES = 1024 * 1024
#: How many chat models a model list returns (Anthropic's own list maximum).
MAX_LISTED_MODELS = 1000
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
    # Inbound, broker-minted endpoint key. Created ONLY through
    # ``CredentialVault.mint``; ``put`` refuses it and ``capabilities`` hides it
    # so the generic create route can never store a weak, unbound one.
    SERVED_ENDPOINT_PROVIDER: served_endpoint_keys.PROVIDER_DESCRIPTOR,
    # VD-206: the hosted services AI Chat may use, one kind each
    # (`hosted_providers` owns the table and says why they are kinds).
    **{
        kind: {"name": facts["name"], "auth": "api_key"}
        for kind, facts in HOSTED_PROVIDERS.items()
    },
}

ASSIGNMENT_PROVIDERS = {
    # VD-207: the Assistant's own lease takes no external kind - OpenAI's row
    # here was unreachable (the vault's role check refused it) and is gone.
    "deployment-agent": {"openai-compatible"},
    # VD-206 item 3: the hosted kinds serve AI Chat and nothing else; the
    # Assistant's purpose above never lists them (VD-049, VD-201).
    "ai-chat": {"openai-compatible", "openai", *HOSTED_KINDS},
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

#: VD-049 / VD-201 / VD-202: which credential may take the Assistant's lease and
#: which may take AI Chat's is answered in `model_credential_roles` and enforced
#: HERE, in `CredentialVault.activate`, not in each route - three routes once
#: assigned the Assistant on their own (LESSONS 6).

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
    if served_endpoint_keys.is_served_endpoint_purpose(purpose):
        return {SERVED_ENDPOINT_PROVIDER}
    return None

NON_CHAT_MODEL_MARKERS = (
    "embed", "embedding", "rerank", "reranker", "whisper", "transcrib",
    "speech", "tts", "text-to-speech", "image", "vision-encoder",
)


def _is_chat_model_id(model_id: str) -> bool:
    """Conservatively exclude model IDs that clearly describe non-chat engines."""
    lower = str(model_id).strip().lower()
    return bool(lower) and not any(marker in lower for marker in NON_CHAT_MODEL_MARKERS)


#: What `resolve_active` answers for a purpose nothing is assigned to. Named so
#: a reader can tell "no lease" from "the broker could not be asked" - both
#: arrive as `CredentialError` - without re-spelling the sentence (VD-127, B4).
NO_ACTIVE_CREDENTIAL = "No credential is active for this purpose."


def validate_ssh_profile(secret_value: str) -> Dict[str, Any]:
    return _validate_ssh_profile(secret_value, CredentialError)


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
        # The WAL sidecars exist only while a connection holds them: SQLite
        # unlinks both when the last connection closes, so a sidecar seen a
        # moment ago can be gone by the chmod (the broker's first live
        # ``FileNotFoundError`` on ``-shm``, VD-127). A vanished sidecar is
        # nothing to protect; the vault file itself was chmod'ed above and a
        # failure there still raises.
        for suffix in ("-wal", "-shm"):
            try:
                os.chmod("{}{}".format(path, suffix), 0o600)
            except FileNotFoundError:
                continue
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
                last_tested_at INTEGER,
                last4 TEXT,
                revoked_at INTEGER
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
            if "last4" not in columns:
                # A served-endpoint key's last four characters are shown (never
                # the key) so two keys on one endpoint are tellable apart; rows
                # that predate minting keep NULL.
                connection.execute(
                    "ALTER TABLE credentials ADD COLUMN last4 TEXT"
                )
            if "revoked_at" not in columns:
                # A served-endpoint key is revoked by stamping this column, not
                # by deleting the row: the record stays for audit while the key
                # stops being active. Rows that predate revoke keep NULL (live).
                connection.execute(
                    "ALTER TABLE credentials ADD COLUMN revoked_at INTEGER"
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
                if provider_id != SERVED_ENDPOINT_PROVIDER
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
        metadata["last4"] = row["last4"] if "last4" in row.keys() else None
        if row["provider"] == SERVED_ENDPOINT_PROVIDER:
            metadata.update(served_endpoint_keys.metadata_view(row))
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
                       last_used_at, last_test_status, last_tested_at, last4,
                       revoked_at
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
        # Review A11: a listing carries metadata and fingerprints, never a
        # secret. An actor-less one is a service polling it - the mode watch
        # every 30 s among them, about 5,760 rows a day - so only a listing
        # made for a named person is audited; every lease still is.
        if clean_actor:
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
        if clean_provider == SERVED_ENDPOINT_PROVIDER:
            raise CredentialError(served_endpoint_keys.PUT_REJECTED)
        clean_label = str(label).strip()[:80] or PROVIDERS[clean_provider]["name"]
        secret_bytes = str(secret_value).strip().encode("utf-8")
        if len(secret_bytes) < 8 or len(secret_bytes) > MAX_SECRET_BYTES:
            raise CredentialError("Credential must contain between 8 and 8,192 bytes.")
        if clean_provider == "openai-compatible":
            validate_compatible_profile(secret_bytes.decode("utf-8"))
        elif clean_provider == "ssh":
            validate_ssh_profile(secret_bytes.decode("utf-8"))
        elif clean_provider == HOSTED_COMPATIBLE:
            validate_hosted_profile(secret_bytes.decode("utf-8"))
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
                       last_used_at, last_test_status, last_tested_at, last4,
                       revoked_at
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

    def mint(self, endpoint_id: str, label: str = "") -> Dict[str, Any]:
        """Mint a fresh inbound endpoint key, revealing the plaintext ONCE.

        The served-endpoint path lives in :mod:`vaelor.served_endpoint_keys`
        (split out for the line ceiling); it uses this vault's own cipher,
        connection and audit. The returned ``key`` is the one-time reveal - it
        is never re-readable through ``list`` or any other verb.
        """
        return served_endpoint_keys.mint(self, endpoint_id, label)

    def rotate(self, credential_id: str, endpoint_id: str = "") -> Dict[str, Any]:
        """Rotate a served-endpoint key in place, revealing the new plaintext ONCE."""
        return served_endpoint_keys.rotate(self, credential_id, endpoint_id)

    def revoke(self, credential_id: str, endpoint_id: str = "") -> Dict[str, Any]:
        """Mark a served-endpoint key revoked (kept for audit), never deleted."""
        return served_endpoint_keys.revoke(self, credential_id, endpoint_id)

    def endpoint_keys(self, endpoint_id: str) -> list:
        """The active plaintext key set for a gate (see the leaf for the boundary)."""
        return served_endpoint_keys.endpoint_keys(self, endpoint_id)

    def import_key(
        self, endpoint_id: str, key: str, label: str = ""
    ) -> Dict[str, Any]:
        """Import a GIVEN key as an endpoint's first key (migration only; see leaf)."""
        return served_endpoint_keys.import_key(self, endpoint_id, key, label)

    def delete(self, credential_id: str) -> bool:
        with closing(self._connect()) as connection:
            kind = connection.execute(
                "SELECT provider FROM credentials WHERE id = ?",
                (str(credential_id),),
            ).fetchone()
            if kind is not None and kind["provider"] == SERVED_ENDPOINT_PROVIDER:
                # ACC-111: an endpoint key is revoked and KEPT (the audit rule
                # `revoke` enforces); a generic delete would erase the record
                # and pull the key out from under a running gate. Refused here,
                # at the vault, so no caller of any route can get round it.
                self._audit(
                    "credential.delete", str(credential_id),
                    SERVED_ENDPOINT_PROVIDER, "refused",
                )
                raise CredentialError(served_endpoint_keys.DELETE_REFUSED)
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
                """
                SELECT c.id, c.provider, c.owner,
                       COALESCE(p.selected_model, '') AS selected_model
                FROM credentials c
                LEFT JOIN credential_model_preferences p ON p.credential_id = c.id
                WHERE c.id = ?
                """,
                (str(credential_id),),
            ).fetchone()
            if row is None:
                raise CredentialError(CREDENTIAL_NOT_FOUND)
            if row["provider"] not in providers:
                raise CredentialError("This credential provider cannot be used for that purpose.")
            # VD-049 / VD-201 / VD-202: the one place both leases are guarded.
            refused = role_refusal(clean_purpose, row)
            if not refused:
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
        if refused:
            self._audit("credential.activate", row["id"], row["provider"], "refused")
            raise CredentialError(refused)
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
            raise CredentialError(NO_ACTIVE_CREDENTIAL)
        plaintext = self._decrypt(row)
        try:
            profile = (
                validate_compatible_profile(plaintext)
                if row["provider"] == "openai-compatible"
                else hosted_lease_profile(row["provider"], plaintext)
                if row["provider"] in HOSTED_KINDS
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
                SELECT id, provider, label, nonce, ciphertext, version, owner,
                       revoked_at
                FROM credentials WHERE id = ?
                """,
                (str(credential_id),),
            ).fetchone()
        if row is None:
            raise CredentialError(CREDENTIAL_NOT_FOUND)
        if row["provider"] not in providers:
            raise CredentialError("This credential provider cannot be used for that purpose.")
        if clean_purpose.startswith(served_endpoint_keys.SERVED_ENDPOINT_PURPOSE_PREFIX):
            bound_endpoint = clean_purpose[len(served_endpoint_keys.SERVED_ENDPOINT_PURPOSE_PREFIX):]
            if bound_endpoint != str(row["owner"]).strip().lower():
                raise CredentialError("This endpoint key is bound to a different endpoint.")
            if row["revoked_at"] is not None:
                raise CredentialError("This endpoint key has been revoked.")
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
                else hosted_lease_profile(row["provider"], plaintext)
                if row["provider"] in HOSTED_KINDS
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
            raise CredentialError(CREDENTIAL_NOT_FOUND)
        if row["provider"] not in {"openai", "openai-compatible", *HOSTED_KINDS}:
            raise CredentialError("This credential does not provide chat models.")
        plaintext = self._decrypt(row)
        try:
            if row["provider"] in HOSTED_KINDS:
                # In the service's own order (Anthropic lists newest first),
                # not sorted: the first listed is what the picker offers first.
                payload = {"data": [
                    {"id": model} for model in hosted_list_models(row["provider"], plaintext)
                ]}
            elif row["provider"] == "openai-compatible":
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
        except CredentialError:
            raise  # a hosted service's refusal, already in the owner's words
        except urllib.error.HTTPError as error:
            if error.code in {401, 403}:
                raise CredentialError("The provider rejected this credential.") from error
            raise CredentialError("The provider returned HTTP {}.".format(error.code)) from error
        except (OSError, ValueError) as error:  # URLError, a timeout, TLS, bad JSON
            raise CredentialError("The model list could not be loaded.") from error
        finally:
            plaintext = ""
        listed = [
            str(item.get("id", "")).strip()
            for item in payload.get("data", [])
            if (
                isinstance(item, dict)
                and _is_chat_model_id(str(item.get("id", "")).strip())
            )
        ]
        models = (
            list(dict.fromkeys(listed)) if row["provider"] in HOSTED_KINDS
            else sorted(set(listed))
        )
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
            # OpenRouter lists several hundred; a cut at 200 made a model the
            # owner typed unselectable (`select_model` checks this list).
            "models": models[:MAX_LISTED_MODELS],
            "selected_model": selected_model,
            "selected_model_available": selected_model_available,
            "selection_required": (
                row["provider"] in {"openai-compatible", *HOSTED_KINDS}
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
                raise CredentialError(str(result.get("message", "The selected model did not pass its chat test.")))
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
            raise CredentialError(CREDENTIAL_NOT_FOUND)
        if row["provider"] == SERVED_ENDPOINT_PROVIDER:
            raise CredentialError("Inbound endpoint keys are not connection-tested.")
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
                SET last_test_status = ?, last_tested_at = ?
                WHERE id = ?
                """,
                (status, now, row["id"]),
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
        """The production connection test (`vaelor.credential_provider_probe`)."""
        return probe_provider(provider, secret_value, PROVIDERS.get(provider, {}))


def main():
    from .credential_broker_server import main as server_main
    server_main()


if __name__ == "__main__":
    main()
