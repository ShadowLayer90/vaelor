"""The F4a persistence layer for cluster-deployed Agents (store only).

This is the record store the F4b agent jobs and runtime will drive; it holds no
job, launches nothing, and reaches no node. It is built and unit-tested
standalone, exactly as F1's :mod:`vaelor.mcp_catalog` and F2's
:mod:`vaelor.skills_library` were built before their consumers, and it mirrors
their idioms: one bounded, validated, deny-by-default SQLite store; the shared
:mod:`vaelor.assistant_store_common` primitives (the clock, the sensitive-field
guard, the broker-reference credential-id shape); and version-pinned records in
the shape of :mod:`vaelor.agent_app_grants` (``custom_agent_version`` is an
``INTEGER NOT NULL`` pin, and an attached grant or skill list is tied to the
agent id AND that version).

A deployment row is::

    {id, name, state, custom_agent_id, custom_agent_version,
     backing{model_deployment_name}, mcp_grants[], skills[],
     placement{node_ids}, endpoint, api_key_id, memory_policy,
     created_at, updated_at}

Three invariants are the security spine, and each is exercised by a test:

1. **No secret value is ever stored.** Every free-text and structured input is
   screened with :func:`assistant_store_common.contains_sensitive`, and the
   ``api_key_id`` column holds ONLY a registry id - never the key value. An
   attempt to persist anything key-shaped (a ``vsk_`` served key, a bearer
   token, a bare ``token_urlsafe`` secret) is refused rather than written.
2. **Version pinning is load-bearing.** ``custom_agent_version`` is pinned at
   creation and never mutated, and :meth:`AgentDeploymentStore.attach_grants`
   and :meth:`AgentDeploymentStore.attach_skills` refuse a list whose supplied
   agent version does not match the pin - the grant/skill set belongs to one
   (agent id, version) pair, exactly as ``agent_app_grants`` binds them.
3. **State transitions are validated.** A row moves ``deploying`` ->
   ``healthy`` | ``failed`` | ``removing``, and a ``healthy`` row may still go
   ``failed`` or ``removing``; nothing reaches ``healthy`` except from
   ``deploying``, and once a row is ``removing`` the only operation it accepts
   is :meth:`AgentDeploymentStore.delete`.

Owner decision (design section 5): a cluster agent is backed by a cluster MODEL
deployment (``backing.model_deployment_name``) - there is no provider-credential
field here at all. ``memory_policy`` is opaque state reserved for F5: it is
screened for secrets and stored verbatim, never interpreted.
"""

from __future__ import annotations

import hashlib
import hmac
import json
import os
import sqlite3
import uuid
from contextlib import contextmanager
from typing import Any, Callable, Dict, Iterable, List, Mapping, Optional

# The cluster deployment-name rule has one home; reuse it so an agent
# deployment's name is held to exactly what a pooled GPU deployment's name is,
# rather than spelling a second slug rule that could drift from it.
from .gpu_pool_units import deployment_name as _cluster_name
# The served-endpoint key VALUE prefix, imported so this store can refuse a raw
# served key where only its registry id belongs (invariant 1). Importing the
# one constant keeps the refusal coupled to the minter, not a copied literal.
from .served_endpoint_keys import VSK_PREFIX
from .assistant_store_common import (
    StoreInputError,
    contains_sensitive as _contains_sensitive,
    credential_id as _common_credential_id,
    now as _now,
    reject_sensitive as _common_reject_sensitive,
)
from .runtime_paths import env_value, state_path


class AgentDeploymentError(ValueError):
    """A safe, user-presentable agent-deployment persistence error."""


class StateTransitionError(AgentDeploymentError):
    """Raised when a requested lifecycle move is not allowed from the current state."""


#: The lifecycle a deployment row travels. ``deploying`` is where a row begins;
#: a deploy that answers healthy is promoted, one that never does is failed, and
#: a teardown moves any row to ``removing``, after which the row accepts only a
#: delete. Held as individual single-token names (never one set-of-strings
#: literal) so the serving vocabulary this store carries is not mistaken for a
#: cross-module word set.
STATE_DEPLOYING = "deploying"
STATE_HEALTHY = "healthy"
STATE_FAILED = "failed"
STATE_REMOVING = "removing"

#: The one allowed-transition table. Nothing reaches ``healthy`` except from
#: ``deploying`` (no health from nothing); a ``removing`` row is terminal until
#: it is deleted. Built from the state names above, so a state can never be
#: mistyped into the table without the name existing.
_ALLOWED_TRANSITIONS: Dict[str, frozenset] = {
    STATE_DEPLOYING: frozenset({STATE_HEALTHY, STATE_FAILED, STATE_REMOVING}),
    STATE_HEALTHY: frozenset({STATE_FAILED, STATE_REMOVING}),
    STATE_FAILED: frozenset({STATE_REMOVING}),
    STATE_REMOVING: frozenset(),
}
_STATES = frozenset(_ALLOWED_TRANSITIONS)

#: The most tools one MCP grant may pin onto an agent version, and the most
#: grants and skills one deployment may carry. Bounded so a single row cannot be
#: grown without limit through the attach seams.
_MAX_TOOLS = 64
_MAX_GRANTS = 64
_MAX_SKILLS = 64
_MAX_NODES = 64
_MAX_MEMORY_POLICY_BYTES = 4096
_ID_PREFIX = "agentdep_"
_BUSY_TIMEOUT_MS = 15000

#: Sentences written at more than one call site keep a single home here so the
#: wording cannot drift between two raises, and each is deliberately phrased so
#: it collides with no other backend module's equivalent.
_NOT_FOUND = "Agent deployment not found."
_NAME_TAKEN = "An agent deployment already uses this name."
_REMOVING_LOCKED = "This agent deployment is being removed and accepts no further changes."
_STALE_VERSION = "The supplied custom agent version does not match this deployment's pin."
_SELECT_BY_NAME = "SELECT * FROM agent_deployments WHERE name=?"


def _name(value: Any) -> str:
    """A deployment name, validated by the shared cluster-name rule."""
    try:
        return _cluster_name(value)
    except ValueError as error:
        raise AgentDeploymentError(str(error)) from error


def _identifier(value: Any, field: str, maximum: int = 120) -> str:
    text = str(value or "").strip()
    if not text or len(text) > maximum:
        raise AgentDeploymentError("The {} value is not usable.".format(field))
    return text


def _version(value: Any) -> int:
    try:
        result = int(value)
        if result < 1:
            raise ValueError
    except (TypeError, ValueError) as error:
        raise AgentDeploymentError("A custom agent version must be a positive integer.") from error
    return result


def _state(value: Any) -> str:
    text = str(value or "").strip().lower()
    if text not in _STATES:
        raise AgentDeploymentError(
            "A deployment state is one of: {}.".format(", ".join(sorted(_STATES)))
        )
    return text


def _reject_sensitive(value: Any, field: str) -> None:
    """Refuse a secret-shaped field anywhere in an input, translating the error."""
    if isinstance(value, Mapping):
        try:
            _common_reject_sensitive(value, field)
        except StoreInputError as error:
            raise AgentDeploymentError(str(error)) from error
    elif _contains_sensitive(value):
        raise AgentDeploymentError(
            "{} cannot carry secret-shaped fields; keep credentials in the "
            "broker and name them by id.".format(field)
        )


def new_deployment_id() -> str:
    """A fresh deployment row id, the one shape :meth:`create` accepts."""
    return _ID_PREFIX + uuid.uuid4().hex


def _deployment_id(value: Any) -> str:
    """A caller-chosen row id, refused unless it is exactly the minted shape."""
    text = str(value or "")
    suffix = text[len(_ID_PREFIX):]
    if not text.startswith(_ID_PREFIX) or len(suffix) != 32 or any(
        char not in "0123456789abcdef" for char in suffix
    ):
        raise AgentDeploymentError("The agent deployment id is not a valid id.")
    return text


def _api_key_id(value: Any) -> str:
    """A registry id for the agent's inbound key - NEVER the key value.

    Invariant 1's home for the key field. The store keeps only the id of the F4
    key record (a minted ``<type>_<body>`` reference such as the broker's
    ``cred_...``); the ``vsk_`` served key itself, a bearer token, and a bare
    high-entropy ``token_urlsafe`` secret are all refused here so a leaked value
    can never be written where an id belongs. Empty is allowed: the key is
    assigned at deploy time, after the row already exists.
    """
    text = str(value or "").strip()
    if not text:
        return ""
    if text.startswith(VSK_PREFIX):
        raise AgentDeploymentError(
            "Store the key record id, not the served key value itself."
        )
    # The shared shape rejects a secret-name substring, whitespace (so a bearer
    # header cannot slip through), an over-long value, and any non-id character.
    try:
        text = _common_credential_id(text)
    except StoreInputError as error:
        raise AgentDeploymentError(str(error)) from error
    # A minted id is structured ``<short-alpha-type>_<body>``; a raw token has
    # no such prefix, so this refuses a bare ``token_urlsafe`` secret that would
    # otherwise pass the id-character rule above.
    prefix, separator, body = text.partition("_")
    if not (separator and body and prefix.isalpha() and prefix.islower()
            and 2 <= len(prefix) <= 16):
        raise AgentDeploymentError(
            "An api_key_id must be a minted key record reference, not a raw token."
        )
    return text


def _node_ids(values: Any) -> List[str]:
    if isinstance(values, (str, bytes)) or not isinstance(values, Iterable):
        raise AgentDeploymentError("placement node_ids must be a list of node ids.")
    result: List[str] = []
    for item in values:
        node = str(item).strip()
        if node and node not in result:
            result.append(node)
    if len(result) > _MAX_NODES:
        raise AgentDeploymentError(
            "A placement names at most {} nodes.".format(_MAX_NODES)
        )
    return result


def _tool_names(values: Any) -> List[str]:
    if isinstance(values, (str, bytes)) or not isinstance(values, Iterable):
        raise AgentDeploymentError("An mcp grant's allowed_tools must be a list.")
    tools: List[str] = []
    for item in values:
        tool = str(item or "").strip()
        if not tool or len(tool) > 120:
            raise AgentDeploymentError("An mcp grant names an invalid tool.")
        if tool not in tools:
            tools.append(tool)
    if len(tools) > _MAX_TOOLS:
        raise AgentDeploymentError(
            "An mcp grant allows at most {} tools.".format(_MAX_TOOLS)
        )
    return tools


def _mcp_grants(values: Any) -> List[Dict[str, Any]]:
    """Validate ``[{server_id, allowed_tools[]}]`` - server ids and tool names only."""
    if isinstance(values, (str, bytes)) or not isinstance(values, Iterable):
        raise AgentDeploymentError("mcp_grants must be a list of grant objects.")
    _reject_sensitive(list(values), "mcp_grants")
    grants: List[Dict[str, Any]] = []
    seen: set = set()
    for entry in values:
        if not isinstance(entry, Mapping):
            raise AgentDeploymentError("Each mcp grant is an object with a server_id.")
        server_id = _identifier(entry.get("server_id"), "server_id")
        if server_id in seen:
            raise AgentDeploymentError("An mcp grant names the same server twice.")
        seen.add(server_id)
        grants.append({
            "server_id": server_id,
            "allowed_tools": _tool_names(entry.get("allowed_tools", ())),
        })
    if len(grants) > _MAX_GRANTS:
        raise AgentDeploymentError(
            "A deployment holds at most {} mcp grants.".format(_MAX_GRANTS)
        )
    return grants


def _skills(values: Any) -> List[Dict[str, Any]]:
    """Validate ``[{skill_id, version}]`` - a skill id and its pinned version."""
    if isinstance(values, (str, bytes)) or not isinstance(values, Iterable):
        raise AgentDeploymentError("skills must be a list of skill objects.")
    _reject_sensitive(list(values), "skills")
    skills: List[Dict[str, Any]] = []
    seen: set = set()
    for entry in values:
        if not isinstance(entry, Mapping):
            raise AgentDeploymentError("Each skill is an object with a skill_id.")
        skill_id = _identifier(entry.get("skill_id"), "skill_id")
        if skill_id in seen:
            raise AgentDeploymentError("A skill is listed twice.")
        seen.add(skill_id)
        skills.append({"skill_id": skill_id, "version": _version(entry.get("version"))})
    if len(skills) > _MAX_SKILLS:
        raise AgentDeploymentError(
            "A deployment holds at most {} skills.".format(_MAX_SKILLS)
        )
    return skills


def _memory_policy(value: Any) -> Any:
    """Opaque F5 state: screened for secrets, size-bounded, stored verbatim."""
    if value is None:
        return None
    if not isinstance(value, (Mapping, str)):
        raise AgentDeploymentError("A memory_policy is a small object or a string.")
    _reject_sensitive(value, "memory_policy")
    encoded = json.dumps(value, separators=(",", ":"), default=str)
    if len(encoded.encode()) > _MAX_MEMORY_POLICY_BYTES:
        raise AgentDeploymentError("The memory_policy is too large to store.")
    return json.loads(encoded)


def _ensure_schema(connection: sqlite3.Connection) -> None:
    connection.executescript(
        """
        CREATE TABLE IF NOT EXISTS agent_deployments (
            id TEXT PRIMARY KEY,
            name TEXT NOT NULL,
            state TEXT NOT NULL,
            custom_agent_id TEXT NOT NULL,
            custom_agent_version INTEGER NOT NULL,
            model_deployment_name TEXT NOT NULL,
            mcp_grants_json TEXT NOT NULL DEFAULT '[]',
            skills_json TEXT NOT NULL DEFAULT '[]',
            node_ids_json TEXT NOT NULL DEFAULT '[]',
            endpoint TEXT NOT NULL DEFAULT '',
            api_key_id TEXT NOT NULL DEFAULT '',
            memory_policy_json TEXT NOT NULL DEFAULT 'null',
            actor TEXT NOT NULL DEFAULT '',
            memory_key_hash TEXT NOT NULL DEFAULT '',
            created_at REAL NOT NULL,
            updated_at REAL NOT NULL,
            UNIQUE(name)
        );
        CREATE INDEX IF NOT EXISTS idx_agent_deployments_state
            ON agent_deployments(state, name);
        """
    )
    # Additive migration for a store created before the reboot reconcile
    # (F4b-ii-C2) needed the deploying admin recorded. A fresh table already
    # carries the column from the CREATE above; an existing one gains it here.
    try:
        connection.execute(
            "ALTER TABLE agent_deployments ADD COLUMN actor TEXT NOT NULL DEFAULT ''"
        )
    except sqlite3.OperationalError:
        pass
    # Additive migration for the F5 per-agent memory-key hash, added the same
    # way the actor column was: a fresh table already carries it from the
    # CREATE above, and an existing store gains it here without a rebuild.
    try:
        connection.execute(
            "ALTER TABLE agent_deployments ADD COLUMN memory_key_hash TEXT NOT NULL DEFAULT ''"
        )
    except sqlite3.OperationalError:
        pass


class AgentDeploymentStore:
    """A deny-by-default, name-keyed store of cluster agent deployment records.

    Keyed on the deployment ``name`` (unique, validated by the shared cluster
    rule), the handle every operation takes. The row carries its own opaque
    ``id`` too, minted for a stable reference the F4b runtime can quote.
    """

    def __init__(self, database_path: Optional[str] = None, clock: Callable[[], float] = _now):
        self.database_path = database_path or env_value(
            "VAELOR_AGENT_DEPLOYMENTS_DB", "PM_AGENT_DEPLOYMENTS_DB",
            state_path("assistant/agent-deployments.sqlite3"),
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
        connection.execute("PRAGMA busy_timeout = {:d}".format(_BUSY_TIMEOUT_MS))
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
        item["mcp_grants"] = json.loads(item.pop("mcp_grants_json") or "[]")
        item["skills"] = json.loads(item.pop("skills_json") or "[]")
        item["placement"] = {"node_ids": json.loads(item.pop("node_ids_json") or "[]")}
        item["backing"] = {"model_deployment_name": item.pop("model_deployment_name")}
        item["memory_policy"] = json.loads(item.pop("memory_policy_json") or "null")
        # The endpoint is nullable until the deploy sets it; present it as None
        # rather than an empty string so a not-yet-deployed row reads honestly.
        item["endpoint"] = item["endpoint"] or None
        # The memory-key hash is an authentication secret's digest, never part
        # of a projected row; the resolve path reads it through its own query.
        item.pop("memory_key_hash", None)
        return item

    def _require(self, connection: sqlite3.Connection, name: str) -> Dict[str, Any]:
        row = connection.execute(_SELECT_BY_NAME, (name,)).fetchone()
        current = self._row(row)
        if current is None:
            raise AgentDeploymentError(_NOT_FOUND)
        return current

    @staticmethod
    def _guard_mutable(current: Mapping[str, Any]) -> None:
        """Refuse any change to a row that is already being removed."""
        if current["state"] == STATE_REMOVING:
            raise StateTransitionError(_REMOVING_LOCKED)

    def create(
        self, *, name: str, custom_agent_id: str, custom_agent_version: Any,
        model_deployment_name: str, node_ids: Iterable[Any] = (),
        memory_policy: Any = None, api_key_id: str = "", endpoint: str = "",
        deployment_id: str = "",
    ) -> Dict[str, Any]:
        """Write a new deployment row in the ``deploying`` state, or refuse it.

        The custom-agent pin (id + version) and the backing cluster model are
        set here and the version is never mutated afterwards. Grants and skills
        start empty and are attached, version-pinned, through their own seams.
        ``deployment_id`` is the id the control plane chose so it could mint
        the agent's first key under ``agent:<id>`` and reveal it once before
        queueing the deploy (GG14); it must be one this store would mint.
        """
        name = _name(name)
        custom_agent_id = _identifier(custom_agent_id, "custom_agent_id")
        version = _version(custom_agent_version)
        model = _identifier(model_deployment_name, "model_deployment_name", 200)
        nodes = _node_ids(node_ids)
        policy = _memory_policy(memory_policy)
        key_id = _api_key_id(api_key_id)
        endpoint_value = _identifier(endpoint, "endpoint", 400) if endpoint else ""
        deployment_id = _deployment_id(deployment_id) if deployment_id else new_deployment_id()
        now = self.clock()
        try:
            with self._connection() as connection:
                connection.execute(
                    """INSERT INTO agent_deployments
                    (id,name,state,custom_agent_id,custom_agent_version,
                     model_deployment_name,mcp_grants_json,skills_json,node_ids_json,
                     endpoint,api_key_id,memory_policy_json,created_at,updated_at)
                    VALUES(?,?,?,?,?,?,?,?,?,?,?,?,?,?)""",
                    (deployment_id, name, STATE_DEPLOYING, custom_agent_id, version,
                     model, "[]", "[]",
                     json.dumps(nodes, separators=(",", ":")),
                     endpoint_value, key_id,
                     json.dumps(policy, separators=(",", ":")), now, now),
                )
        except sqlite3.IntegrityError as error:
            raise AgentDeploymentError(_NAME_TAKEN) from error
        return self.get(name)  # type: ignore[return-value]

    def get(self, name: str) -> Optional[Dict[str, Any]]:
        with self._connection() as connection:
            row = connection.execute(_SELECT_BY_NAME, (str(name),)).fetchone()
        return self._row(row)

    def list(self, *, state: Optional[str] = None, limit: int = 200) -> List[Dict[str, Any]]:
        limit = max(1, min(int(limit), 500))
        if state is None:
            query = "SELECT * FROM agent_deployments ORDER BY created_at DESC LIMIT ?"
            values: tuple = (limit,)
        else:
            query = ("SELECT * FROM agent_deployments WHERE state=? "
                     "ORDER BY created_at DESC LIMIT ?")
            values = (_state(state), limit)
        with self._connection() as connection:
            rows = connection.execute(query, values).fetchall()
        return [self._row(row) for row in rows]  # type: ignore[list-item]

    def update_state(self, name: str, state: str) -> Dict[str, Any]:
        """Move a row to ``state`` if the transition is allowed, else refuse it."""
        target = _state(state)
        now = self.clock()
        with self._connection() as connection:
            current = self._require(connection, str(name))
            source = current["state"]
            if target == source:
                return current
            if target not in _ALLOWED_TRANSITIONS[source]:
                raise StateTransitionError(
                    "An agent deployment cannot move from {} to {}.".format(source, target)
                )
            connection.execute(
                "UPDATE agent_deployments SET state=?,updated_at=? WHERE name=?",
                (target, now, str(name)),
            )
            return self._require(connection, str(name))

    def set_endpoint(self, name: str, endpoint: str) -> Dict[str, Any]:
        """Record the agent's OpenAI-compatible endpoint (set at deploy time)."""
        endpoint_value = _identifier(endpoint, "endpoint", 400) if endpoint else ""
        now = self.clock()
        with self._connection() as connection:
            current = self._require(connection, str(name))
            self._guard_mutable(current)
            connection.execute(
                "UPDATE agent_deployments SET endpoint=?,updated_at=? WHERE name=?",
                (endpoint_value, now, str(name)),
            )
            return self._require(connection, str(name))

    def set_api_key_id(self, name: str, api_key_id: str) -> Dict[str, Any]:
        """Record the F4 key-registry id for the agent's inbound key (id only)."""
        key_id = _api_key_id(api_key_id)
        now = self.clock()
        with self._connection() as connection:
            current = self._require(connection, str(name))
            self._guard_mutable(current)
            connection.execute(
                "UPDATE agent_deployments SET api_key_id=?,updated_at=? WHERE name=?",
                (key_id, now, str(name)),
            )
            return self._require(connection, str(name))

    def set_actor(self, name: str, actor: str) -> Dict[str, Any]:
        """Record the deploying admin so the reboot reconcile (SF-A) can re-read
        the pinned custom-agent definition when it re-renders this agent's
        runtime after a reboot. This is an identity, never a secret."""
        value = _identifier(actor, "actor", 120)
        now = self.clock()
        with self._connection() as connection:
            current = self._require(connection, str(name))
            self._guard_mutable(current)
            connection.execute(
                "UPDATE agent_deployments SET actor=?,updated_at=? WHERE name=?",
                (value, now, str(name)),
            )
            return self._require(connection, str(name))

    @staticmethod
    def _memory_key_hash(token: str) -> str:
        """The sha256 hex of a per-agent memory token - the only form kept."""
        return hashlib.sha256(str(token).encode("utf-8")).hexdigest()

    def set_memory_key_hash(self, name: str, key_hash: str) -> Dict[str, Any]:
        """Store the sha256 hash of this agent's memory token (id-direction secret).

        Only the hash is held, never the token, so the control plane can resolve
        a presented token without being able to reproduce it. Deliberately NOT
        guarded against a removing row: a teardown clears the key while the row
        is already ``removing``, and rotating this credential must still work
        then, exactly as the cluster ingest key clears on an enrolled node.
        """
        now = self.clock()
        with self._connection() as connection:
            self._require(connection, str(name))
            connection.execute(
                "UPDATE agent_deployments SET memory_key_hash=?,updated_at=? WHERE name=?",
                (str(key_hash), now, str(name)),
            )
            return self._require(connection, str(name))

    def clear_memory_key(self, name: str) -> Dict[str, Any]:
        """Revoke this agent's memory token by emptying its stored hash.

        The resolve path skips a row whose hash is empty, so the old token
        authenticates to nothing the moment this runs - the revoke a teardown
        needs without deleting the deployment row.
        """
        return self.set_memory_key_hash(name, "")

    def agent_id_for_memory_key(self, presented_token: Any):
        """The ``(name, id)`` a presented memory token binds to, or ``None``.

        The token is hashed and compared, in CONSTANT time via
        `hmac.compare_digest`, against EVERY row's stored hash. Every row is
        compared even after a match, so the time taken reveals neither which
        agent matched nor how far down it sat. A row with an empty hash is
        excluded from the query, so an agent never issued a token cannot match,
        and an empty or non-string token matches nothing. This is the ONE seam
        that turns a bearer secret into the isolated agent identity the memory
        store tags by; the token itself is never returned or logged.
        """
        if not isinstance(presented_token, str) or not presented_token:
            return None
        presented_hash = self._memory_key_hash(presented_token)
        matched = None
        with self._connection() as connection:
            rows = connection.execute(
                "SELECT name, id, memory_key_hash FROM agent_deployments "
                "WHERE memory_key_hash != ''"
            ).fetchall()
        for row in rows:
            if hmac.compare_digest(presented_hash, str(row["memory_key_hash"])):
                matched = (row["name"], row["id"])
        return matched

    def attach_grants(
        self, name: str, agent_version: Any, mcp_grants: Iterable[Any],
    ) -> Dict[str, Any]:
        """Pin an mcp-grant list onto this deployment for a matching agent version.

        The list belongs to one (custom_agent_id, custom_agent_version) pair, so
        the supplied ``agent_version`` must equal the row's pin; a stale version
        is refused rather than allowed to attach a grant set to a definition the
        row is no longer pinned to.
        """
        version = _version(agent_version)
        grants = _mcp_grants(mcp_grants)
        now = self.clock()
        with self._connection() as connection:
            current = self._require(connection, str(name))
            self._guard_mutable(current)
            if version != current["custom_agent_version"]:
                raise StateTransitionError(_STALE_VERSION)
            connection.execute(
                "UPDATE agent_deployments SET mcp_grants_json=?,updated_at=? WHERE name=?",
                (json.dumps(grants, separators=(",", ":")), now, str(name)),
            )
            return self._require(connection, str(name))

    def attach_skills(
        self, name: str, agent_version: Any, skills: Iterable[Any],
    ) -> Dict[str, Any]:
        """Pin a skill list onto this deployment for a matching agent version.

        Version-pinned exactly like :meth:`attach_grants`: the skill list is
        tied to the row's pinned custom agent version, and a mismatch is refused.
        """
        version = _version(agent_version)
        skill_list = _skills(skills)
        now = self.clock()
        with self._connection() as connection:
            current = self._require(connection, str(name))
            self._guard_mutable(current)
            if version != current["custom_agent_version"]:
                raise StateTransitionError(_STALE_VERSION)
            connection.execute(
                "UPDATE agent_deployments SET skills_json=?,updated_at=? WHERE name=?",
                (json.dumps(skill_list, separators=(",", ":")), now, str(name)),
            )
            return self._require(connection, str(name))

    def delete(self, name: str) -> Dict[str, Any]:
        """Drop a deployment row. Allowed from any state - it is the removal escape."""
        with self._connection() as connection:
            cursor = connection.execute(
                "DELETE FROM agent_deployments WHERE name=?", (str(name),)
            )
        if cursor.rowcount != 1:
            raise AgentDeploymentError(_NOT_FOUND)
        return {"removed": True, "name": str(name)}
