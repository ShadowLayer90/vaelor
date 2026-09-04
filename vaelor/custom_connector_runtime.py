"""Least-privilege execution and audit for custom-agent REST connectors."""

from __future__ import annotations

import hashlib
import http.client
import json
import os
import sqlite3
import ssl
import time
import uuid
from contextlib import closing
from typing import Any, Callable, Dict, Optional
from urllib.parse import quote, urlencode, urlsplit, urlunsplit

from .application_research import (
    REDIRECT_STATUSES, ResearchSecurityError,
    _PinnedHTTPSConnection, _validate_public_address,
    system_resolver,
)
from .connector_policy import (
    PATH_PARAMETER, READ_METHODS, ConnectorPolicyError, policy_fingerprint,
    validate_instance,
)
from .credential_broker import CredentialError
from .runtime_paths import state_path


class ConnectorRuntimeError(ValueError):
    pass


APPROVAL_TTL_SECONDS = 15 * 60


class ConnectorRuntime:
    def __init__(
        self, credential_broker, database_path: Optional[str] = None,
        resolver: Callable[[str, int], tuple[str, ...]] = system_resolver,
        connection_factory: Callable[..., Any] = _PinnedHTTPSConnection,
        clock: Callable[[], float] = time.time,
    ):
        self.credential_broker = credential_broker
        self.database_path = database_path or state_path("assistant/connector-audit.sqlite3")
        self.resolver = resolver
        self.connection_factory = connection_factory
        self.clock = clock
        parent = os.path.dirname(self.database_path)
        if parent:
            os.makedirs(parent, mode=0o700, exist_ok=True)
        self._initialize()

    def _connect(self):
        connection = sqlite3.connect(self.database_path, timeout=15)
        connection.row_factory = sqlite3.Row
        return connection

    def _initialize(self):
        with closing(self._connect()) as connection:
            connection.executescript("""
                CREATE TABLE IF NOT EXISTS connector_audit (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    event_id TEXT NOT NULL UNIQUE,
                    actor TEXT NOT NULL,
                    agent_id TEXT NOT NULL,
                    definition_version INTEGER NOT NULL,
                    connector_id TEXT NOT NULL,
                    operation_id TEXT NOT NULL,
                    method TEXT NOT NULL,
                    path_template TEXT NOT NULL,
                    policy_fingerprint TEXT NOT NULL,
                    request_sha256 TEXT NOT NULL,
                    response_sha256 TEXT NOT NULL DEFAULT '',
                    response_bytes INTEGER NOT NULL DEFAULT 0,
                    status TEXT NOT NULL,
                    detail TEXT NOT NULL,
                    created_at REAL NOT NULL
                );
                CREATE TABLE IF NOT EXISTS connector_approvals (
                    id TEXT PRIMARY KEY,
                    actor TEXT NOT NULL,
                    agent_id TEXT NOT NULL,
                    definition_version INTEGER NOT NULL,
                    connector_id TEXT NOT NULL,
                    operation_id TEXT NOT NULL,
                    policy_fingerprint TEXT NOT NULL,
                    arguments_json TEXT NOT NULL,
                    state TEXT NOT NULL,
                    created_at REAL NOT NULL,
                    decided_at REAL,
                    expires_at REAL NOT NULL DEFAULT 0,
                    credential_version INTEGER NOT NULL DEFAULT 0
                );
                CREATE TABLE IF NOT EXISTS connector_rate_events (
                    event_id TEXT PRIMARY KEY,
                    agent_id TEXT NOT NULL,
                    connector_id TEXT NOT NULL,
                    operation_id TEXT NOT NULL,
                    created_at REAL NOT NULL
                );
            """)
            columns = {
                row[1] for row in connection.execute(
                    "PRAGMA table_info(connector_approvals)"
                ).fetchall()
            }
            if "expires_at" not in columns:
                connection.execute(
                    "ALTER TABLE connector_approvals ADD COLUMN expires_at REAL NOT NULL DEFAULT 0"
                )
            if "credential_version" not in columns:
                connection.execute(
                    "ALTER TABLE connector_approvals ADD COLUMN credential_version INTEGER NOT NULL DEFAULT 0"
                )
            connection.commit()
        try:
            os.chmod(self.database_path, 0o600)
        except OSError:
            pass

    @staticmethod
    def _find(profile: Dict[str, Any], connector_id: str, operation_id: str):
        connector = next((item for item in profile.get("connectors", []) if item["id"] == connector_id), None)
        if connector is None:
            raise ConnectorRuntimeError("connector_revoked: this connector grant is unavailable.")
        operation = next((item for item in connector["operations"] if item["id"] == operation_id), None)
        if operation is None:
            raise ConnectorRuntimeError("operation_denied: this connector operation is not granted.")
        return connector, operation

    def execute(
        self, profile: Dict[str, Any], connector_id: str, operation_id: str,
        arguments: Dict[str, Any], *, actor: str, task_id: str = "",
        approved: bool = False,
    ) -> Dict[str, Any]:
        connector, operation = self._find(profile, connector_id, operation_id)
        try:
            clean_arguments = validate_instance(arguments or {}, operation["request_schema"])
        except ConnectorPolicyError as error:
            self._audit(
                profile, connector, operation,
                arguments if isinstance(arguments, dict) else {},
                "denied", "request_schema_denied: {}".format(error),
            )
            raise ConnectorRuntimeError("request_schema_denied: {}".format(error)) from error
        fingerprint = policy_fingerprint(connector, operation)
        if operation["method"] not in READ_METHODS and not approved:
            approval_id = "connector_approval_" + uuid.uuid4().hex
            created_at = self.clock()
            credential_version = self._credential_version(connector, actor)
            with closing(self._connect()) as connection:
                connection.execute(
                    "INSERT INTO connector_approvals "
                    "(id,actor,agent_id,definition_version,connector_id,operation_id,"
                    "policy_fingerprint,arguments_json,state,created_at,decided_at,"
                    "expires_at,credential_version) VALUES(?,?,?,?,?,?,?,?,?,?,NULL,?,?)",
                    (approval_id, actor, profile["id"], int(profile["version"]),
                     connector_id, operation_id, fingerprint,
                     json.dumps(clean_arguments, separators=(",", ":")),
                     "pending", created_at, created_at + APPROVAL_TTL_SECONDS,
                     credential_version),
                )
                connection.commit()
            preview = self._preview(connector, operation, clean_arguments)
            self._audit(profile, connector, operation, clean_arguments, "approval_required", approval_id)
            return {
                "state": "approval_required", "approval_id": approval_id,
                "preview": preview, "executed": False,
                "expires_at": created_at + APPROVAL_TTL_SECONDS,
            }
        try:
            return self._perform(
                profile, connector, operation, clean_arguments, actor, task_id
            )
        except ConnectorRuntimeError as error:
            self._audit(
                profile, connector, operation, clean_arguments, "failed", str(error)
            )
            raise

    def approve(
        self, approval_id: str, profile: Dict[str, Any], *, actor: str,
    ) -> Dict[str, Any]:
        now = self.clock()
        with closing(self._connect()) as connection:
            connection.execute(
                "UPDATE connector_approvals SET state='expired',decided_at=? "
                "WHERE id=? AND actor=? AND state='pending' AND expires_at<?",
                (now, str(approval_id), actor, now),
            )
            row = connection.execute(
                "SELECT * FROM connector_approvals WHERE id=? AND actor=? "
                "AND agent_id=? AND state='pending' AND expires_at>=?",
                (str(approval_id), actor, str(profile.get("id", "")), now),
            ).fetchone()
            if row is not None:
                claimed = connection.execute(
                    "UPDATE connector_approvals SET state='executing',decided_at=? "
                    "WHERE id=? AND actor=? AND agent_id=? AND state='pending' "
                    "AND expires_at>=?",
                    (now, approval_id, actor, str(profile.get("id", "")), now),
                ).rowcount
                connection.commit()
            else:
                claimed = 0
        if row is None or claimed != 1:
            raise ConnectorRuntimeError("approval_unavailable: connector approval was not found or already decided.")
        try:
            connector, operation = self._find(
                profile, row["connector_id"], row["operation_id"]
            )
        except ConnectorRuntimeError as error:
            with closing(self._connect()) as connection:
                connection.execute(
                    "UPDATE connector_approvals SET state='stale' WHERE id=? AND state='executing'",
                    (approval_id,),
                )
                connection.commit()
            raise ConnectorRuntimeError("approval_stale: connector grant was revoked.") from error
        if (
            int(profile["version"]) != row["definition_version"]
            or policy_fingerprint(connector, operation) != row["policy_fingerprint"]
            or self._credential_version(connector, actor) != row["credential_version"]
        ):
            with closing(self._connect()) as connection:
                connection.execute(
                    "UPDATE connector_approvals SET state='stale' WHERE id=? AND state='executing'",
                    (approval_id,),
                )
                connection.commit()
            raise ConnectorRuntimeError("approval_stale: the connector grant changed or was revoked.")
        arguments = json.loads(row["arguments_json"])
        try:
            result = self._perform(profile, connector, operation, arguments, actor, approval_id)
        except Exception as error:
            with closing(self._connect()) as connection:
                connection.execute(
                    "UPDATE connector_approvals SET state='failed' WHERE id=? AND state='executing'",
                    (approval_id,),
                )
                connection.commit()
            self._audit(
                profile, connector, operation, arguments, "failed", str(error)
            )
            raise
        with closing(self._connect()) as connection:
            connection.execute(
                "UPDATE connector_approvals SET state='executed',decided_at=? "
                "WHERE id=? AND state='executing'",
                (self.clock(), approval_id),
            )
            connection.commit()
        return result

    def test_connection(
        self, profile: Dict[str, Any], connector_id: str, operation_id: str,
        arguments: Dict[str, Any], *, actor: str,
    ) -> Dict[str, Any]:
        connector, operation = self._find(profile, connector_id, operation_id)
        if operation["method"] not in READ_METHODS:
            raise ConnectorRuntimeError("test_denied: connection tests require a granted read operation.")
        result = self.execute(
            profile, connector_id, operation_id, arguments, actor=actor,
            task_id="connection-test",
        )
        return {"ok": True, "connector_id": connector_id, "operation_id": operation_id,
                "response_bytes": result["metadata"]["response_bytes"]}

    def _perform(self, profile, connector, operation, arguments, actor, task_id):
        self._reserve_rate(profile["id"], connector["id"], operation)
        url, body = self._request_parts(connector, operation, arguments)
        headers = {"Accept": "application/json", "User-Agent": "Vaelor-Connector/1"}
        if body is not None:
            headers["Content-Type"] = "application/json"
        if connector["auth"] != "none":
            try:
                lease = self.credential_broker.resolve(
                    connector["credential_ref"], "custom-agent-connector", actor
                )
            except (CredentialError, ValueError) as error:
                raise ConnectorRuntimeError("credential_unavailable: {}".format(error)) from error
            token = str(lease.get("token", ""))
            if not token:
                raise ConnectorRuntimeError("credential_unavailable: credential has no usable token.")
            header = "Authorization" if connector["auth"] == "bearer" else "X-API-Key"
            headers[header] = ("Bearer " + token) if connector["auth"] == "bearer" else token
        try:
            status, raw = self._fetch(
                url, operation["method"], headers, body,
                operation["timeout_seconds"], operation["max_response_bytes"],
            )
        finally:
            headers.pop("Authorization", None)
            headers.pop("X-API-Key", None)
        if operation["method"] == "HEAD" and not raw:
            decoded: Any = {}
        else:
            try:
                decoded = json.loads(raw.decode("utf-8"))
            except (UnicodeDecodeError, json.JSONDecodeError) as error:
                raise ConnectorRuntimeError("response_schema_denied: connector did not return valid JSON.") from error
        try:
            validated = validate_instance(decoded, operation["response_schema"], "response")
        except ConnectorPolicyError as error:
            raise ConnectorRuntimeError("response_schema_denied: {}".format(error)) from error
        digest = hashlib.sha256(raw).hexdigest()
        self._audit(
            profile, connector, operation, arguments, "completed",
            "HTTP {} task={}".format(status, str(task_id)[:80]), digest, len(raw),
        )
        return {
            "state": "completed", "executed": True, "result": validated,
            "metadata": {"http_status": status, "response_bytes": len(raw),
                         "response_sha256": digest},
        }

    def _fetch(self, url, method, headers, body, timeout, limit):
        parsed = urlsplit(url)
        host = parsed.hostname or ""
        addresses = tuple(dict.fromkeys(self.resolver(host, 443)))
        if not addresses or len(addresses) > 8:
            raise ConnectorRuntimeError("network_denied: connector DNS response is unavailable or too broad.")
        for address in addresses:
            try:
                _validate_public_address(address)
            except ResearchSecurityError as error:
                raise ConnectorRuntimeError("network_denied: connector resolved to a private or unsafe address.") from error
        target = urlunsplit(("", "", parsed.path or "/", parsed.query, ""))
        last_error = None
        for peer in addresses:
            connection = None
            try:
                connection = self.connection_factory(host, 443, peer, timeout)
                connection.request(method, target, body=body, headers=headers)
                response = connection.getresponse()
                if response.status in REDIRECT_STATUSES:
                    raise ConnectorRuntimeError("redirect_denied: connectors do not follow redirects.")
                raw = response.read(limit + 1)
                if len(raw) > limit:
                    raise ConnectorRuntimeError("response_too_large: connector exceeded its response limit.")
                if not 200 <= response.status < 300:
                    raise ConnectorRuntimeError("connector_http_error: remote API returned HTTP {}.".format(response.status))
                media = str(response.getheader("Content-Type", "")).split(";", 1)[0].lower()
                if method != "HEAD" and media not in {"application/json", "application/problem+json"}:
                    raise ConnectorRuntimeError("response_type_denied: connectors accept JSON responses only.")
                return response.status, raw
            except ConnectorRuntimeError:
                raise
            except (OSError, ssl.SSLError, http.client.HTTPException, TimeoutError) as error:
                last_error = error
            finally:
                if connection is not None:
                    connection.close()
        raise ConnectorRuntimeError("connector_unreachable: public API could not be reached securely.") from last_error

    @staticmethod
    def _request_parts(connector, operation, arguments):
        used = set(PATH_PARAMETER.findall(operation["path"]))
        path = PATH_PARAMETER.sub(lambda match: quote(str(arguments[match.group(1)]), safe=""), operation["path"])
        remaining = {key: value for key, value in arguments.items() if key not in used}
        if operation["input_location"] == "query":
            query = urlencode(remaining, doseq=True)
            return connector["base_origin"] + path + (("?" + query) if query else ""), None
        return connector["base_origin"] + path, json.dumps(remaining, separators=(",", ":")).encode("utf-8")

    def _preview(self, connector, operation, arguments):
        url, body = self._request_parts(connector, operation, arguments)
        return {
            "connector": connector["name"], "operation": operation["id"],
            "method": operation["method"], "url": url,
            "body_sha256": hashlib.sha256(body or b"").hexdigest(),
            "body_bytes": len(body or b""), "credential_ref": connector["credential_ref"],
            "credential_value_included": False,
        }

    def _credential_version(self, connector, actor):
        if connector["auth"] == "none":
            return 0
        try:
            metadata = next(
                item for item in self.credential_broker.list(actor)
                if item.get("id") == connector["credential_ref"]
            )
        except (StopIteration, AttributeError, TypeError, ValueError) as error:
            raise ConnectorRuntimeError(
                "credential_unavailable: credential metadata is unavailable."
            ) from error
        if metadata.get("provider") != "application-secret":
            raise ConnectorRuntimeError(
                "credential_unavailable: connector credential type is invalid."
            )
        return int(metadata.get("version", 0))

    def _reserve_rate(self, agent_id, connector_id, operation):
        now = self.clock()
        cutoff = now - 60
        with closing(self._connect()) as connection:
            connection.execute("BEGIN IMMEDIATE")  # pairs-with: sqlite-begin-immediate
            connection.execute(
                "DELETE FROM connector_rate_events WHERE created_at<?", (now - 3600,)
            )
            count = connection.execute(
                "SELECT COUNT(*) FROM connector_rate_events WHERE agent_id=? "
                "AND connector_id=? AND operation_id=? AND created_at>=?",
                (agent_id, connector_id, operation["id"], cutoff),
            ).fetchone()[0]
            if count >= operation["rate_limit_per_minute"]:
                connection.rollback()
                raise ConnectorRuntimeError(
                    "rate_limit_denied: operation rate limit reached."
                )
            connection.execute(
                "INSERT INTO connector_rate_events VALUES(?,?,?,?,?)",
                (uuid.uuid4().hex, agent_id, connector_id, operation["id"], now),
            )
            connection.commit()

    def _audit(self, profile, connector, operation, arguments, status, detail,
               response_sha256="", response_bytes=0):
        request_digest = hashlib.sha256(json.dumps(
            arguments, sort_keys=True, separators=(",", ":")
        ).encode("utf-8")).hexdigest()
        with closing(self._connect()) as connection:
            connection.execute(
                "INSERT INTO connector_audit "
                "(event_id,actor,agent_id,definition_version,connector_id,operation_id,method,"
                "path_template,policy_fingerprint,request_sha256,response_sha256,response_bytes,"
                "status,detail,created_at) VALUES(?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)",
                (uuid.uuid4().hex, str(profile.get("actor", "")), profile["id"],
                 int(profile["version"]), connector["id"], operation["id"],
                 operation["method"], operation["path"], policy_fingerprint(connector, operation),
                 request_digest, response_sha256, response_bytes, status, str(detail)[:500], self.clock()),
            )
            connection.commit()

    def audit(self, agent_id: str, actor: str, limit: int = 100):
        with closing(self._connect()) as connection:
            rows = connection.execute(
                "SELECT event_id,definition_version,connector_id,operation_id,method,path_template,"
                "request_sha256,response_sha256,response_bytes,status,detail,created_at "
                "FROM connector_audit WHERE agent_id=? AND actor=? ORDER BY id DESC LIMIT ?",
                (agent_id, actor, max(1, min(int(limit), 200))),
            ).fetchall()
        return [dict(row) for row in rows]
