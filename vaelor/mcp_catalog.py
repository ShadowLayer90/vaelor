"""The agent-facing MCP catalog and per-agent, version-pinned tool grants.

This is the F1 backend for cluster agents that reach *out* to MCP servers. It is
deliberately kept separate from the inbound ``VaelorMcpServer`` surface: nothing
here is ever handed to ``VaelorMcpServer(external=...)`` and no record here mints
an ``EXTERNAL_SCOPE`` session. Two stores share one SQLite file, mirroring the
rigor of :mod:`vaelor.agent_app_grants`:

* :class:`McpCatalogStore` — an administrator-curated registry of MCP servers a
  deployed agent may be granted. A ``builtin`` server is Vaelor itself and has
  no network address, so a deployed agent's outbound client can never call it:
  its health says so rather than "healthy" (ACC-076), and a deploy that grants
  it is refused with that reason. A ``url`` server is somebody else's software
  whose endpoint must pass the exact same SSRF guard the outbound
  ``mcp_client`` uses; its health is re-probed when it is older than
  :data:`HEALTH_STALE_SECONDS` (ACC-077).
* :class:`AgentMcpGrantStore` — version-pinned ``allowed_tools`` grants that
  attach a catalog server to one agent version. At creation ``allowed_tools``
  MUST be a subset of the server's ``approved_tools`` (the catalog-side half of
  the intersection invariant; the hard call-seam refusal is F4).

The *authorisation truth an agent runtime enforces* is supplied to
:meth:`AgentMcpGrantStore.evaluate`, which is deny-by-default: a missing,
disabled, revoked, or version-stale grant yields a non-``active`` decision, and
``allowed_tools`` is re-intersected with the server's *current* ``approved_tools``
so shrinking the catalog shrinks every grant.
"""

from __future__ import annotations

import json
import os
import sqlite3
import threading
import uuid
from contextlib import contextmanager
from typing import Any, Callable, Dict, Iterable, List, Mapping, Optional

# Reuse the outbound client's SSRF guard and bounds rather than reimplement
# them: `_endpoint` is the exact validator `mcp_client.validate_server` applies
# (HTTPS + `public_host`, loopback-only plain HTTP), and the name/tool shapes and
# limits are the same ones an enrolled external server is held to.
from .mcp_client import (
    MAX_TOOLS_PER_SERVER,
    SERVER_NAME,
    TOOL_NAME,
    McpClientError,
    _endpoint as validate_endpoint,
    http_transport,
    public_host,
)
from .runtime_paths import env_value, state_path
from .assistant_store_common import (
    StoreInputError,
    SENSITIVE_NAMES as _SENSITIVE_NAMES,
    contains_sensitive as _contains_sensitive,
    credential_id as _common_credential_id,
    digest as _digest,
    now as _now,
    reject_sensitive as _common_reject_sensitive,
)


class McpCatalogError(ValueError):
    """A safe, user-presentable catalog or grant error."""


class GrantDeniedError(McpCatalogError):
    """Raised when a grant is not active for the supplied current facts."""

    def __init__(self, decision: Mapping[str, Any]):
        self.decision = dict(decision)
        super().__init__(
            "MCP grant unavailable: {}.".format(
                ", ".join(decision.get("reasons", ["unknown"]))
            )
        )


KINDS = ("builtin", "url")
HEALTH_STATES = {"unknown", "healthy", "unreachable", "error", "degraded"}
MAX_ALLOWED_TOOLS = MAX_TOOLS_PER_SERVER
_CREDENTIAL_ID = "credential_id"

#: Error sentences written at more than one call site keep a single home here,
#: so the message cannot drift between two raises. This is the duplicate-literal
#: guard's "give the sentence one home" repair for a within-module repeat,
#: applied in place rather than tolerated.
_SERVER_NOT_FOUND = "Server was not found."
_GRANT_NOT_FOUND = "MCP grant was not found."
_NAME_TAKEN = "A server with this name already exists."
_ENABLED_MESSAGE = "enabled must be true or false."
_SELECT_GRANT_BY_ID = "SELECT * FROM agent_mcp_grants WHERE id=?"


def _text(value: Any, field: str, maximum: int = 160) -> str:
    result = str(value or "").strip()
    if not result or len(result) > maximum:
        raise McpCatalogError("{} is invalid.".format(field))
    return result


def _server_name(value: Any) -> str:
    name = str(value or "").strip().lower()
    if not SERVER_NAME.fullmatch(name):
        raise McpCatalogError(
            "A server name uses lower-case letters, digits and hyphens."
        )
    return name


def _kind(value: Any) -> str:
    kind = str(value or "").strip().lower()
    if kind not in KINDS:
        raise McpCatalogError("A server kind is one of: {}.".format(", ".join(KINDS)))
    return kind


def _url_endpoint(value: Any) -> str:
    """Validate a ``url`` server's endpoint through mcp_client's own SSRF guard.

    The guard is reused verbatim; only its error type is translated so every
    catalog rejection is a :class:`McpCatalogError`.
    """
    try:
        return validate_endpoint(value)
    except McpClientError as error:
        raise McpCatalogError(str(error)) from error


def _credential_id(value: Any) -> str:
    """An *identifier* for a broker credential, validated by the shared rule."""
    try:
        return _common_credential_id(value)
    except StoreInputError as error:
        raise McpCatalogError(str(error)) from error


def _tool_list(values: Any, field: str, maximum: int = MAX_TOOLS_PER_SERVER) -> List[str]:
    if isinstance(values, (str, bytes)) or not isinstance(values, Iterable):
        raise McpCatalogError("{} must be a list of tool names.".format(field))
    tools: List[str] = []
    for item in values:
        tool = str(item).strip()
        if not TOOL_NAME.fullmatch(tool):
            raise McpCatalogError("{} contains an invalid tool name.".format(field))
        if tool not in tools:
            tools.append(tool)
    if len(tools) > maximum:
        raise McpCatalogError(
            "{} holds at most {} tool names.".format(field, maximum)
        )
    return tools


def _reject_sensitive(value: Mapping[str, Any], field: str = "input") -> None:
    try:
        _common_reject_sensitive(value, field)
    except StoreInputError as error:
        raise McpCatalogError(str(error)) from error


def _version(value: Any) -> int:
    try:
        result = int(value)
        if result < 1:
            raise ValueError
    except (TypeError, ValueError) as error:
        raise McpCatalogError("The grant agent_version is invalid.") from error
    return result


def _default_health() -> Dict[str, Any]:
    return {"status": "unknown", "checked_at": None, "tools_count": 0, "detail": ""}


def _builtin_health() -> Dict[str, Any]:
    """The built-in server's honest health, as a deployed agent meets it.

    A deployed agent reaches tool servers through its outbound MCP client, which
    needs an address; Vaelor's built-in server has none, so an agent that was
    granted it could not start while this read "healthy" (ACC-076).
    """
    return {
        "status": "unreachable", "checked_at": _now(), "tools_count": 0,
        "detail": (
            "A deployed cluster agent cannot call Vaelor's built-in server: it "
            "has no network address. Vaelor's own read-only tools reach a "
            "deployed agent through the scopes its definition grants."
        ),
    }


#: A ``url`` server's health older than this is re-probed in the background
#: when the catalog is read, so a server that went down stops reading healthy
#: without anyone pressing Re-check (ACC-077).
HEALTH_STALE_SECONDS = 300.0

_REFRESH_LOCK = threading.Lock()


def stale_server_ids(servers: Iterable[Mapping[str, Any]], now: float) -> List[str]:
    """The ``url`` servers whose last health check is missing or stale."""
    stale: List[str] = []
    for server in servers:
        if str(server.get("kind")) != "url":
            continue
        checked = (server.get("health") or {}).get("checked_at")
        try:
            fresh = checked is not None and now - float(checked) < HEALTH_STALE_SECONDS
        except (TypeError, ValueError):
            fresh = False
        if not fresh:
            stale.append(str(server.get("id")))
    return stale


def start_stale_refresh(
    store: "McpCatalogStore", servers: Iterable[Mapping[str, Any]], *,
    now: Optional[float] = None, spawn: Optional[Callable[[Callable[[], None]], Any]] = None,
) -> List[str]:
    """Re-probe stale servers OFF the request, one refresh at a time.

    Returns the ids handed to the refresh (empty when none are stale or one is
    already running). The read that triggered it answers at once with what is
    stored; the next read sees the fresh result.
    """
    ids = stale_server_ids(servers, _now() if now is None else now)
    if not ids or not _REFRESH_LOCK.acquire(blocking=False):
        return []

    def _run() -> None:
        try:
            for server_id in ids:
                try:
                    refresh_server_health(store, server_id)
                except (McpCatalogError, OSError, ValueError):
                    continue
        finally:
            _REFRESH_LOCK.release()

    starter = spawn or (lambda target: threading.Thread(target=target, daemon=True).start())
    try:
        starter(_run)
    except BaseException:
        # A refresh that never started must not hold the lock for good.
        _REFRESH_LOCK.release()
        raise
    return ids


def _normalise_health(value: Any) -> Dict[str, Any]:
    if not isinstance(value, Mapping):
        raise McpCatalogError("health must be an object.")
    _reject_sensitive(value, "health")
    status = str(value.get("status", "unknown")).strip().lower()
    if status not in HEALTH_STATES:
        raise McpCatalogError(
            "health status is one of: {}.".format(", ".join(sorted(HEALTH_STATES)))
        )
    checked_at = value.get("checked_at")
    if checked_at is not None:
        try:
            checked_at = float(checked_at)
        except (TypeError, ValueError) as error:
            raise McpCatalogError("health checked_at is invalid.") from error
    try:
        tools_count = int(value.get("tools_count", 0))
    except (TypeError, ValueError) as error:
        raise McpCatalogError("health tools_count is invalid.") from error
    if not 0 <= tools_count <= 10000:
        raise McpCatalogError("health tools_count is out of range.")
    detail = " ".join(str(value.get("detail", "")).split())[:400]
    return {
        "status": status, "checked_at": checked_at,
        "tools_count": tools_count, "detail": detail,
    }


def _ensure_schema(connection: sqlite3.Connection) -> None:
    """Create BOTH catalog tables. Either store can bootstrap the shared file, so
    constructing ``AgentMcpGrantStore`` alone never hits a missing-table error.

    Grants are administrator-managed, NOT siloed by the creating admin: the
    ``created_by`` column is an audit field, never an access key. Any administrator
    sees and revokes any grant, and the F4 runtime authorises a grant by id without
    a username. (The route's ``require_auth("administrator")`` is the real gate.)
    """
    connection.executescript(
        """
        CREATE TABLE IF NOT EXISTS mcp_catalog_servers (
            id TEXT PRIMARY KEY,
            name TEXT NOT NULL,
            kind TEXT NOT NULL,
            endpoint TEXT NOT NULL DEFAULT '',
            credential_id TEXT NOT NULL DEFAULT '',
            enabled INTEGER NOT NULL DEFAULT 0,
            approved_tools_json TEXT NOT NULL DEFAULT '[]',
            health_json TEXT NOT NULL DEFAULT '{}',
            created_at REAL NOT NULL,
            updated_at REAL NOT NULL,
            UNIQUE(name)
        );
        CREATE INDEX IF NOT EXISTS idx_mcp_catalog_kind
            ON mcp_catalog_servers(kind, name);
        CREATE TABLE IF NOT EXISTS agent_mcp_grants (
            id TEXT PRIMARY KEY,
            created_by TEXT NOT NULL,
            agent_id TEXT NOT NULL,
            agent_version INTEGER NOT NULL,
            server_id TEXT NOT NULL,
            allowed_tools_json TEXT NOT NULL,
            source_grant_id TEXT,
            revoked_at REAL,
            revocation_reason TEXT NOT NULL DEFAULT '',
            idempotency_key TEXT,
            request_digest TEXT NOT NULL,
            created_at REAL NOT NULL,
            updated_at REAL NOT NULL,
            UNIQUE(created_by, idempotency_key)
        );
        CREATE INDEX IF NOT EXISTS idx_agent_mcp_grants_agent
            ON agent_mcp_grants(agent_id, agent_version);
        CREATE INDEX IF NOT EXISTS idx_agent_mcp_grants_server
            ON agent_mcp_grants(server_id, created_at DESC);
        CREATE UNIQUE INDEX IF NOT EXISTS idx_agent_mcp_grants_clone
            ON agent_mcp_grants(agent_id, agent_version, source_grant_id)
            WHERE source_grant_id IS NOT NULL;
        """
    )


class McpCatalogStore:
    """An administrator-curated registry of agent-reachable MCP servers."""

    def __init__(self, database_path: Optional[str] = None, clock: Callable[[], float] = _now):
        self.database_path = database_path or env_value(
            "VAELOR_MCP_CATALOG_DB", "PM_MCP_CATALOG_DB",
            state_path("assistant/mcp-catalog.sqlite3"),
        )
        self.clock = clock
        parent = os.path.dirname(self.database_path)
        if parent:
            os.makedirs(parent, mode=0o700, exist_ok=True)
        self._initialize()
        try:
            # 0o660, not 0o600: these stores are shared across the control-plane
            # (vaelor) and the workload executor (vaelor-workloads) via the setgid
            # vaelor-jobs assistant dir; a cluster.agent.deploy runs in the executor
            # and must open the catalog/skills/deployment stores the control-plane
            # also writes. A non-owner tolerates an already-correct mode.
            os.chmod(self.database_path, 0o660)
        except PermissionError:
            if os.stat(self.database_path).st_mode & 0o777 != 0o660:
                raise

    def _connect(self) -> sqlite3.Connection:
        connection = sqlite3.connect(self.database_path, timeout=15)
        connection.row_factory = sqlite3.Row
        connection.execute("PRAGMA foreign_keys=ON")  # pairs-with: sqlite-foreign-keys-tight
        connection.execute("PRAGMA busy_timeout=15000")
        return connection

    @contextmanager
    def _connection(self):
        connection = self._connect()
        try:
            with connection:
                yield connection
        finally:
            connection.close()

    def _initialize(self) -> None:
        with self._connection() as connection:
            connection.execute("PRAGMA journal_mode=WAL")  # pairs-with: sqlite-journal-mode-tight
            _ensure_schema(connection)

    @staticmethod
    def _row(row: Optional[sqlite3.Row]) -> Optional[Dict[str, Any]]:
        if row is None:
            return None
        item = dict(row)
        item["approved_tools"] = json.loads(item.pop("approved_tools_json") or "[]")
        item["health"] = json.loads(item.pop("health_json") or "{}") or _default_health()
        item["enabled"] = bool(item["enabled"])
        return item

    def register(
        self, *, name: str, kind: str, endpoint: str = "",
        credential_id: str = "", enabled: bool = False,
        approved_tools: Iterable[Any] = (),
    ) -> Dict[str, Any]:
        """Enrol a server, or refuse it. Enrolling does not approve any tool."""
        name = _server_name(name)
        kind = _kind(kind)
        credential = _credential_id(credential_id)
        if not isinstance(enabled, bool):
            raise McpCatalogError(_ENABLED_MESSAGE)
        tools = _tool_list(approved_tools, "approved_tools")
        if kind == "builtin":
            # Vaelor itself: no endpoint, no credential, always healthy.
            endpoint_value = ""
            credential = ""
            health = _builtin_health()
        else:
            endpoint_value = _url_endpoint(endpoint)
            health = _default_health()
        server_id = "mcpsrv_" + uuid.uuid4().hex
        now = self.clock()
        try:
            with self._connection() as connection:
                connection.execute(
                    """INSERT INTO mcp_catalog_servers
                    (id,name,kind,endpoint,credential_id,enabled,approved_tools_json,
                     health_json,created_at,updated_at)
                    VALUES(?,?,?,?,?,?,?,?,?,?)""",
                    (server_id, name, kind, endpoint_value, credential, int(enabled),
                     json.dumps(tools, separators=(",", ":")),
                     json.dumps(health, separators=(",", ":")), now, now),
                )
        except sqlite3.IntegrityError as error:
            raise McpCatalogError(_NAME_TAKEN) from error
        return self.get(server_id)  # type: ignore[return-value]

    def get(self, server_id: str) -> Optional[Dict[str, Any]]:
        with self._connection() as connection:
            row = connection.execute(
                "SELECT * FROM mcp_catalog_servers WHERE id=?", (str(server_id),)
            ).fetchone()
        return self._row(row)

    def list(self, *, enabled_only: bool = False, limit: int = 200) -> List[Dict[str, Any]]:
        limit = max(1, min(int(limit), 500))
        query = "SELECT * FROM mcp_catalog_servers"
        if enabled_only:
            query += " WHERE enabled=1"
        query += " ORDER BY name LIMIT ?"
        with self._connection() as connection:
            rows = connection.execute(query, (limit,)).fetchall()
        return [self._row(row) for row in rows]  # type: ignore[list-item]

    def update(self, server_id: str, patch: Mapping[str, Any]) -> Dict[str, Any]:
        """Amend a server's mutable fields. ``kind`` is immutable."""
        if not isinstance(patch, Mapping):
            raise McpCatalogError("A server update is an object.")
        _reject_sensitive(patch, "server update")
        if "kind" in patch:
            raise McpCatalogError("A server's kind cannot be changed after enrolment.")
        current = self.get(server_id)
        if current is None:
            raise McpCatalogError(_SERVER_NOT_FOUND)
        name = _server_name(patch.get("name", current["name"]))
        if "enabled" in patch and not isinstance(patch["enabled"], bool):
            raise McpCatalogError(_ENABLED_MESSAGE)
        enabled = bool(patch.get("enabled", current["enabled"]))
        tools = _tool_list(
            patch.get("approved_tools", current["approved_tools"]), "approved_tools"
        )
        if current["kind"] == "builtin":
            endpoint_value = ""
            credential = ""
        else:
            endpoint_value = _url_endpoint(patch.get("endpoint", current["endpoint"]))
            credential = _credential_id(patch.get("credential_id", current["credential_id"]))
        now = self.clock()
        try:
            with self._connection() as connection:
                connection.execute(
                    "UPDATE mcp_catalog_servers SET name=?,endpoint=?,credential_id=?,"
                    "enabled=?,approved_tools_json=?,updated_at=? WHERE id=?",
                    (name, endpoint_value, credential, int(enabled),
                     json.dumps(tools, separators=(",", ":")), now, str(server_id)),
                )
        except sqlite3.IntegrityError as error:
            raise McpCatalogError(_NAME_TAKEN) from error
        return self.get(server_id)  # type: ignore[return-value]

    def remove(self, server_id: str) -> Dict[str, Any]:
        with self._connection() as connection:
            cursor = connection.execute(
                "DELETE FROM mcp_catalog_servers WHERE id=?", (str(server_id),)
            )
        if cursor.rowcount != 1:
            raise McpCatalogError(_SERVER_NOT_FOUND)
        return {"removed": True, "id": str(server_id)}

    def record_health(self, server_id: str, health: Mapping[str, Any]) -> Dict[str, Any]:
        """Persist a probe result. A ``builtin`` server keeps its fixed reading."""
        current = self.get(server_id)
        if current is None:
            raise McpCatalogError(_SERVER_NOT_FOUND)
        normalised = _builtin_health() if current["kind"] == "builtin" else _normalise_health(health)
        now = self.clock()
        with self._connection() as connection:
            connection.execute(
                "UPDATE mcp_catalog_servers SET health_json=?,updated_at=? WHERE id=?",
                (json.dumps(normalised, separators=(",", ":")), now, str(server_id)),
            )
        return self.get(server_id)  # type: ignore[return-value]


def probe_server_health(
    server: Mapping[str, Any], transport: Callable[..., Any] = http_transport,
) -> Dict[str, Any]:
    """Probe one server and return an honest health dict — never fabricate.

    A ``builtin`` server is Vaelor itself and needs no network. A ``url`` server
    is asked to ``initialize`` and to list its tools; failure records
    ``unreachable`` with a short detail rather than a healthy status.
    """
    if str(server.get("kind")) == "builtin":
        return _builtin_health()
    endpoint = str(server.get("endpoint") or "")
    if not endpoint:
        return {"status": "error", "checked_at": _now(), "tools_count": 0,
                "detail": "No endpoint is configured for this server."}
    # Re-check the SSRF guard at probe time: a name that resolved public at
    # enrolment must still be public now, or we do not contact it. This matches
    # the existing outbound path; a validate-then-connect DNS-rebind gap remains.
    # F4: the real per-call agent seam should resolve once and connect to the
    # pinned public IP (Host/SNI preserved), closing the rebind window for good.
    wire = {"name": str(server.get("name") or "server"), "endpoint": endpoint}
    try:
        validate_endpoint(endpoint)
        transport(wire, "initialize", {
            "protocolVersion": "2025-06-18",
            "capabilities": {},
            "clientInfo": {"name": "vaelor-catalog-probe", "version": "1"},
        })
        listing = transport(wire, "tools/list", {})
    except (McpClientError, OSError, TypeError, ValueError) as error:
        return {"status": "unreachable", "checked_at": _now(), "tools_count": 0,
                "detail": " ".join(str(error).split())[:400]}
    tools = listing.get("tools") if isinstance(listing, dict) else None
    count = len(tools) if isinstance(tools, list) else 0
    return {"status": "healthy", "checked_at": _now(), "tools_count": count,
            "detail": "Server answered initialize and advertised {} tools.".format(count)}


def refresh_server_health(
    store: McpCatalogStore, server_id: str,
    transport: Callable[..., Any] = http_transport,
) -> Dict[str, Any]:
    """Probe a stored server and persist the result via ``record_health``."""
    server = store.get(server_id)
    if server is None:
        raise McpCatalogError(_SERVER_NOT_FOUND)
    return store.record_health(server_id, probe_server_health(server, transport))


class AgentMcpGrantStore:
    """Version-pinned per-agent MCP tool grants, deny-by-default on evaluate.

    Shares one SQLite file with :class:`McpCatalogStore` so grant creation can
    read the server's live ``approved_tools`` in the same transaction, exactly as
    ``AgentAppGrantStore`` reads ``integration_connections``.
    """

    def __init__(self, database_path: Optional[str] = None, clock: Callable[[], float] = _now):
        self.database_path = database_path or env_value(
            "VAELOR_MCP_CATALOG_DB", "PM_MCP_CATALOG_DB",
            state_path("assistant/mcp-catalog.sqlite3"),
        )
        self.clock = clock
        parent = os.path.dirname(self.database_path)
        if parent:
            os.makedirs(parent, mode=0o700, exist_ok=True)
        self._initialize()
        try:
            # 0o660, not 0o600: these stores are shared across the control-plane
            # (vaelor) and the workload executor (vaelor-workloads) via the setgid
            # vaelor-jobs assistant dir; a cluster.agent.deploy runs in the executor
            # and must open the catalog/skills/deployment stores the control-plane
            # also writes. A non-owner tolerates an already-correct mode.
            os.chmod(self.database_path, 0o660)
        except PermissionError:
            if os.stat(self.database_path).st_mode & 0o777 != 0o660:
                raise

    def _connect(self) -> sqlite3.Connection:
        connection = sqlite3.connect(self.database_path, timeout=15)
        connection.row_factory = sqlite3.Row
        connection.execute("PRAGMA foreign_keys=ON")  # pairs-with: sqlite-foreign-keys-tight
        connection.execute("PRAGMA busy_timeout=15000")
        return connection

    @contextmanager
    def _connection(self):
        connection = self._connect()
        try:
            with connection:
                yield connection
        finally:
            connection.close()

    def _initialize(self) -> None:
        with self._connection() as connection:
            connection.execute("PRAGMA journal_mode=WAL")  # pairs-with: sqlite-journal-mode-tight
            _ensure_schema(connection)

    @staticmethod
    def _row(row: Optional[sqlite3.Row]) -> Optional[Dict[str, Any]]:
        if row is None:
            return None
        item = dict(row)
        item["allowed_tools"] = json.loads(item.pop("allowed_tools_json"))
        item["revoked"] = item["revoked_at"] is not None
        item.pop("request_digest", None)
        return item

    @staticmethod
    def _server(connection: sqlite3.Connection, server_id: str) -> Optional[Dict[str, Any]]:
        row = connection.execute(
            "SELECT id,name,kind,enabled,approved_tools_json FROM mcp_catalog_servers WHERE id=?",
            (str(server_id),),
        ).fetchone()
        if row is None:
            return None
        item = dict(row)
        item["approved_tools"] = json.loads(item.pop("approved_tools_json") or "[]")
        item["enabled"] = bool(item["enabled"])
        return item

    def create(
        self, created_by: str, agent_id: str, agent_version: Any, server_id: str,
        allowed_tools: Iterable[Any], *, idempotency_key: Optional[str] = None,
    ) -> Dict[str, Any]:
        created_by = _text(created_by, "created_by", 120)
        agent_id = _text(agent_id, "agent_id", 120)
        version = _version(agent_version)
        server_id = _text(server_id, "server_id", 120)
        requested = _tool_list(allowed_tools, "allowed_tools", MAX_ALLOWED_TOOLS)
        if not requested:
            raise McpCatalogError("A grant must allow at least one tool.")
        key = None if idempotency_key is None else _text(idempotency_key, "idempotency_key", 160)
        payload = {
            "created_by": created_by, "agent_id": agent_id, "agent_version": version,
            "server_id": server_id, "allowed_tools": requested,
        }
        request_digest = _digest(payload)
        grant_id = "mcpgrant_" + uuid.uuid4().hex
        now = self.clock()
        with self._connection() as connection:
            server = self._server(connection, server_id)
            if server is None:
                raise McpCatalogError("No catalog server with that id is enrolled.")
            # The catalog-side half of the intersection invariant: a grant can
            # never allow a tool the catalog has not approved. The hard call-seam
            # refusal is F4; this refuses to even record an over-broad grant.
            approved = set(server["approved_tools"])
            excess = [tool for tool in requested if tool not in approved]
            if excess:
                raise McpCatalogError(
                    "allowed_tools must be a subset of the server's approved_tools; "
                    "not approved: {}.".format(", ".join(sorted(excess))[:200])
                )
            if key:
                existing = connection.execute(
                    "SELECT * FROM agent_mcp_grants WHERE created_by=? AND idempotency_key=?",
                    (created_by, key),
                ).fetchone()
                if existing:
                    if existing["request_digest"] != request_digest:
                        raise McpCatalogError(
                            "The idempotency_key conflicts with an earlier MCP grant request."
                        )
                    return self._row(existing)  # type: ignore[return-value]
            connection.execute(
                """INSERT INTO agent_mcp_grants
                (id,created_by,agent_id,agent_version,server_id,allowed_tools_json,
                 source_grant_id,revoked_at,revocation_reason,idempotency_key,
                 request_digest,created_at,updated_at)
                VALUES(?,?,?,?,?,?,?,?,?,?,?,?,?)""",
                (grant_id, created_by, agent_id, version, server_id,
                 json.dumps(requested, separators=(",", ":")), None, None, "", key,
                 request_digest, now, now),
            )
            row = connection.execute(_SELECT_GRANT_BY_ID, (grant_id,)).fetchone()
        return self._row(row)  # type: ignore[return-value]

    def get(self, grant_id: str) -> Optional[Dict[str, Any]]:
        # Administrator-global: any administrator resolves any grant by id. The
        # route's require_auth("administrator") is the gate, not the creator.
        with self._connection() as connection:
            row = connection.execute(_SELECT_GRANT_BY_ID, (str(grant_id),)).fetchone()
        return self._row(row)

    def list(
        self, agent_id: Optional[str] = None,
        *, agent_version: Optional[Any] = None, limit: int = 200,
    ) -> List[Dict[str, Any]]:
        limit = max(1, min(int(limit), 500))
        query = "SELECT * FROM agent_mcp_grants"
        clauses: List[str] = []
        values: List[Any] = []
        if agent_id is not None:
            clauses.append("agent_id=?")
            values.append(_text(agent_id, "agent_id", 120))
        if agent_version is not None:
            clauses.append("agent_version=?")
            values.append(_version(agent_version))
        if clauses:
            query += " WHERE " + " AND ".join(clauses)
        query += " ORDER BY created_at DESC LIMIT ?"
        values.append(limit)
        with self._connection() as connection:
            rows = connection.execute(query, values).fetchall()
        return [self._row(row) for row in rows]  # type: ignore[list-item]

    def revoke(self, grant_id: str, reason: str = "") -> Dict[str, Any]:
        now = self.clock()
        with self._connection() as connection:
            connection.execute(
                "UPDATE agent_mcp_grants SET revoked_at=?,revocation_reason=?,updated_at=? "
                "WHERE id=? AND revoked_at IS NULL",
                (now, str(reason or "revoked by an administrator").strip()[:256], now, str(grant_id)),
            )
        item = self.get(grant_id)
        if item is None:
            raise McpCatalogError(_GRANT_NOT_FOUND)
        return item

    def evaluate(
        self, grant_id: str, current_state: Optional[Mapping[str, Any]] = None,
        **facts: Any,
    ) -> Dict[str, Any]:
        """Deny-by-default decision for the runtime's supplied current facts.

        The stored grant pins only stable identifiers. Whether it is usable *now*
        depends on the live catalog (does the server still exist, is it enabled,
        does it still approve these tools) and the runtime's current agent version,
        supplied here rather than trusted from the row. Resolved by grant id — the
        F4 runtime authorises without needing the creating administrator's name.
        """
        grant = self.get(grant_id)
        if grant is None:
            raise McpCatalogError(_GRANT_NOT_FOUND)
        state = dict(current_state or {})
        state.update(facts)
        if grant["revoked"]:
            return {
                "grant_id": grant["id"], "status": "revoked",
                "reasons": ["grant_revoked"], "permitted_tools": [],
                "recovery_action": "Create a new MCP grant.",
            }
        reasons: List[str] = []
        incompatible: List[str] = []
        if state.get("agent_version") is None:
            reasons.append("agent_version_unavailable")
        elif int(state["agent_version"]) != grant["agent_version"]:
            incompatible.append("agent_version_stale")
        with self._connection() as connection:
            server = self._server(connection, grant["server_id"])
        if server is None:
            reasons.append("server_unavailable")
            approved: set = set()
        else:
            approved = set(server["approved_tools"])
            if not server["enabled"]:
                reasons.append("server_disabled")
        # Re-intersect with the catalog's *current* approval. A tool removed from
        # the server after the grant was written is no longer permitted, and if
        # every granted tool is gone the grant is incompatible, not merely thin.
        permitted = [tool for tool in grant["allowed_tools"] if tool in approved]
        if server is not None and not permitted:
            incompatible.append("all_tools_revoked")
        elif server is not None and len(permitted) != len(grant["allowed_tools"]):
            incompatible.append("some_tools_revoked")
        if incompatible:
            status = "incompatible"
        elif reasons:
            status = "blocked"
        else:
            status = "active"
        return {
            "grant_id": grant["id"], "status": status,
            "reasons": sorted(set(incompatible + reasons)),
            "permitted_tools": sorted(permitted),
            "recovery_action": (
                "Review the pinned agent version and the catalog server's approved tools."
                if status != "active" else ""
            ),
        }

    status = evaluate

    def require_active(
        self, grant_id: str, current_state: Optional[Mapping[str, Any]] = None,
        **facts: Any,
    ) -> Dict[str, Any]:
        decision = self.evaluate(grant_id, current_state, **facts)
        if decision["status"] != "active":
            raise GrantDeniedError(decision)
        record = self.get(grant_id)
        # Hand back the catalog-intersected authoritative list, so an F4 caller
        # never accidentally trusts the row's raw ``allowed_tools``.
        record["permitted_tools"] = decision["permitted_tools"]  # type: ignore[index]
        return record  # type: ignore[return-value]

    authorize = require_active

    def clone_version(
        self, agent_id: str, from_version: Any, to_version: Any,
    ) -> Dict[str, Any]:
        """Carry live grants forward into an explicitly requested agent version.

        The cloned ``allowed_tools`` is copied verbatim from the source grant; it is
        NOT re-intersected here. That is safe because ``evaluate`` re-intersects with
        the server's *current* ``approved_tools`` at authorization time, so a tool
        removed from the catalog since the source grant was written can never be
        exercised through the clone.
        """
        agent_id = _text(agent_id, "agent_id", 120)
        source_version = _version(from_version)
        target_version = _version(to_version)
        if source_version == target_version:
            raise McpCatalogError("The clone from_version and to_version must differ.")
        clone_lookup = (
            "SELECT * FROM agent_mcp_grants WHERE agent_id=? "
            "AND agent_version=? AND source_grant_id=?"
        )
        now = self.clock()
        with self._connection() as connection:
            sources = connection.execute(
                "SELECT * FROM agent_mcp_grants WHERE agent_id=? "
                "AND agent_version=? AND revoked_at IS NULL ORDER BY created_at ASC",
                (agent_id, source_version),
            ).fetchall()
            cloned: List[Dict[str, Any]] = []
            for source in sources:
                existing = connection.execute(
                    clone_lookup, (agent_id, target_version, source["id"]),
                ).fetchone()
                if existing is not None:
                    cloned.append(self._row(existing))  # type: ignore[arg-type]
                    continue
                grant_id = "mcpgrant_" + uuid.uuid4().hex
                try:
                    connection.execute(
                        """INSERT INTO agent_mcp_grants
                        (id,created_by,agent_id,agent_version,server_id,allowed_tools_json,
                         source_grant_id,revoked_at,revocation_reason,idempotency_key,
                         request_digest,created_at,updated_at)
                        VALUES(?,?,?,?,?,?,?,?,?,?,?,?,?)""",
                        (grant_id, source["created_by"], agent_id, target_version,
                         source["server_id"], source["allowed_tools_json"], source["id"],
                         None, "", None, source["request_digest"], now, now),
                    )
                except sqlite3.IntegrityError:
                    # A concurrent clone won the unique clone index; adopt its row.
                    existing = connection.execute(
                        clone_lookup, (agent_id, target_version, source["id"]),
                    ).fetchone()
                    if existing is not None:
                        cloned.append(self._row(existing))  # type: ignore[arg-type]
                    continue
                created = connection.execute(_SELECT_GRANT_BY_ID, (grant_id,)).fetchone()
                cloned.append(self._row(created))  # type: ignore[arg-type]
        return {
            "agent_id": agent_id,
            "from_version": source_version, "to_version": target_version,
            "cloned": len(cloned), "grants": cloned,
        }

    def dependents(self, server_id: str) -> Dict[str, Any]:
        """Which grants would be affected by removing or disabling a server."""
        with self._connection() as connection:
            rows = connection.execute(
                "SELECT id,created_by,agent_id,agent_version,allowed_tools_json,revoked_at "
                "FROM agent_mcp_grants WHERE server_id=? ORDER BY created_at DESC",
                (str(server_id),),
            ).fetchall()
        grants = []
        for row in rows:
            item = dict(row)
            item["allowed_tools"] = json.loads(item.pop("allowed_tools_json"))
            grants.append(item)
        return {
            "server_id": str(server_id),
            "counts": {
                "total": len(grants),
                "active": sum(item["revoked_at"] is None for item in grants),
                "revoked": sum(item["revoked_at"] is not None for item in grants),
            },
            "grants": grants,
        }

    dependency_impact = dependents
