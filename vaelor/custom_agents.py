"""Persistent, versioned general-purpose custom agent definitions."""

from __future__ import annotations

import json
import ipaddress
import os
import re
import sqlite3
import time
import uuid
from contextlib import closing
from typing import Any, Dict, Optional
from urllib.parse import urlsplit

from .agent_naming import draft_agent_name
from .agent_research_needs import needs_public_research
from .runtime_paths import env_value, state_path
from .connector_policy import ConnectorPolicyError, validate_connectors


ALLOWED_SCOPES = {
    "assistant:read", "cluster:read", "cooling:read", "jobs:read",
    "research:read", "system:read", "workloads:read",
}
ALLOWED_PERMISSIONS = {"knowledge:read", "knowledge:write", "workloads:propose"}
MAX_WEB_DOMAINS = 20
UNSAFE_INSTRUCTIONS = re.compile(
    r"(ignore\s+(all\s+)?previous|system\s+prompt|reveal\s+(a\s+)?secret|"
    r"(password|token|secret|api[_ -]?key)\s*[:=]|sudo\b|shell\s+command)",
    re.I,
)


class CustomAgentError(ValueError):
    pass


def _domain_name(value: Any) -> str:
    """Normalize a public DNS suffix without resolving or contacting it."""
    text = str(value or "").strip().lower().rstrip(".")
    if "://" in text:
        try:
            parsed = urlsplit(text)
        except ValueError:
            parsed = None
        if parsed is None or parsed.scheme != "https" or parsed.path not in ("", "/"):
            raise CustomAgentError("Web allowlist entries must be domain names, not URLs.")
        text = parsed.hostname or ""
    try:
        ascii_name = text.encode("idna").decode("ascii")
    except UnicodeError as error:
        raise CustomAgentError("A web allowlist domain is invalid.") from error
    if not ascii_name or len(ascii_name) > 253 or ascii_name == "localhost":
        raise CustomAgentError("Web access can allow only public domain names.")
    try:
        ipaddress.ip_address(ascii_name)
    except ValueError:
        pass
    else:
        raise CustomAgentError("Web access can allow only public domain names, not IP addresses.")
    labels = ascii_name.split(".")
    if len(labels) < 2 or any(
        not re.fullmatch(r"[a-z0-9](?:[a-z0-9-]{0,61}[a-z0-9])?", label)
        for label in labels
    ) or labels[-1] in {"local", "localhost", "internal", "home", "lan"}:
        raise CustomAgentError("Web access can allow only valid public domain names.")
    return ascii_name


def _web_access(value: Dict[str, Any], scopes: list[str]) -> Dict[str, Any]:
    supplied = "web_access" in value
    raw = value.get("web_access", {"enabled": False, "allowed_domains": []})
    if not isinstance(raw, dict) or set(raw) - {"enabled", "allowed_domains"}:
        raise CustomAgentError("Web access must contain only enabled and allowed_domains.")
    enabled = raw.get("enabled", False)
    domains = raw.get("allowed_domains", [])
    if not isinstance(enabled, bool) or not isinstance(domains, list):
        raise CustomAgentError("Web access needs a boolean enabled value and a domain list.")
    if len(domains) > MAX_WEB_DOMAINS:
        raise CustomAgentError("Web access is limited to 20 allowed domains.")
    normalized = []
    for item in domains:
        domain = _domain_name(item)
        if domain not in normalized:
            normalized.append(domain)
    research = "research:read" in scopes
    if research and (not supplied or not enabled):
        raise CustomAgentError(
            "Research capability requires explicit web_access.enabled=true."
        )
    if enabled and not research:
        raise CustomAgentError("Enabled web access requires the research:read capability.")
    if normalized and not enabled:
        raise CustomAgentError("Enable web access before adding allowed domains.")
    return {"enabled": enabled, "allowed_domains": sorted(normalized)}


class CustomAgentStore:
    def __init__(self, database_path: Optional[str] = None, credential_broker=None):
        self.credential_broker = credential_broker
        self.database_path = database_path or env_value(
            "VAELOR_CUSTOM_AGENTS_DB", "PM_CUSTOM_AGENTS_DB",
            state_path("assistant/custom-agents.sqlite3"),
        )
        parent = os.path.dirname(self.database_path)
        if parent:
            os.makedirs(parent, mode=0o700, exist_ok=True)
        self._initialize()
        try:
            os.chmod(self.database_path, 0o660)
        except PermissionError:
            if os.stat(self.database_path).st_mode & 0o777 != 0o660:
                raise

    def _connect(self):
        connection = sqlite3.connect(self.database_path, timeout=15)
        connection.row_factory = sqlite3.Row
        connection.execute("PRAGMA foreign_keys=ON")  # pairs-with: sqlite-foreign-keys-tight
        return connection

    def _initialize(self):
        with closing(self._connect()) as connection:
            connection.executescript(
                """
                CREATE TABLE IF NOT EXISTS custom_agents (
                    id TEXT PRIMARY KEY,
                    actor TEXT NOT NULL,
                    name TEXT NOT NULL,
                    description TEXT NOT NULL,
                    instructions TEXT NOT NULL,
                    scopes_json TEXT NOT NULL,
                    version INTEGER NOT NULL,
                    enabled INTEGER NOT NULL,
                    created_at REAL NOT NULL,
                    updated_at REAL NOT NULL,
                    UNIQUE(actor, name)
                );
                CREATE TABLE IF NOT EXISTS custom_agent_revisions (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    agent_id TEXT NOT NULL,
                    version INTEGER NOT NULL,
                    definition_json TEXT NOT NULL,
                    actor TEXT NOT NULL,
                    action TEXT NOT NULL,
                    created_at REAL NOT NULL,
                    FOREIGN KEY(agent_id) REFERENCES custom_agents(id) ON DELETE CASCADE
                );
                """
            )
            columns = {
                row["name"] for row in connection.execute(
                    "PRAGMA table_info(custom_agents)"
                )
            }
            if "permissions_json" not in columns:
                connection.execute(
                    "ALTER TABLE custom_agents ADD COLUMN permissions_json TEXT NOT NULL DEFAULT '[]'"
                )
            if "read_collections_json" not in columns:
                connection.execute(
                    "ALTER TABLE custom_agents ADD COLUMN read_collections_json TEXT NOT NULL DEFAULT '[]'"
                )
            if "write_collection_id" not in columns:
                connection.execute(
                    "ALTER TABLE custom_agents ADD COLUMN write_collection_id TEXT NOT NULL DEFAULT ''"
                )
            if "web_policy_json" not in columns:
                connection.execute(
                    "ALTER TABLE custom_agents ADD COLUMN web_policy_json TEXT NOT NULL "
                    "DEFAULT '{\"enabled\":false,\"allowed_domains\":[]}'"
                )
            if "connectors_json" not in columns:
                connection.execute(
                    "ALTER TABLE custom_agents ADD COLUMN connectors_json TEXT NOT NULL DEFAULT '[]'"
                )
            connection.commit()

    @staticmethod
    def _row(row):
        if row is None:
            return None
        item = dict(row)
        item["scopes"] = json.loads(item.pop("scopes_json"))
        item["permissions"] = json.loads(item.pop("permissions_json"))
        item["read_collection_ids"] = json.loads(item.pop("read_collections_json"))
        item["web_access"] = json.loads(item.pop("web_policy_json"))
        item["connectors"] = json.loads(item.pop("connectors_json"))
        item["enabled"] = bool(item["enabled"])
        item["custom"] = True
        return item

    @staticmethod
    def _validated_definition(value: Dict[str, Any]):
        name = str(value.get("name", "")).strip()[:100]
        description = str(value.get("description", "")).strip()[:400]
        instructions = str(value.get("instructions", "")).strip()
        raw_scopes = list(map(str, value.get("scopes", [])))
        scopes = sorted(set(raw_scopes))
        if not name or not description or not instructions:
            raise CustomAgentError("Name, purpose, and operating instructions are required.")
        if len(instructions) > 6000:
            raise CustomAgentError("Agent instructions are limited to 6,000 characters.")
        if len(raw_scopes) != len(scopes) or any(
            scope not in ALLOWED_SCOPES for scope in scopes
        ):
            raise CustomAgentError("Choose only supported optional capabilities.")
        if UNSAFE_INSTRUCTIONS.search(instructions):
            raise CustomAgentError("Agent instructions request unsafe or secret-shaped access.")
        raw_permissions = list(map(str, value.get("permissions", [])))
        permissions = sorted({
            item for item in raw_permissions if item in ALLOWED_PERMISSIONS
        })
        if len(permissions) != len(set(raw_permissions)):
            raise CustomAgentError("Choose only supported guarded permissions.")
        read_collections = [
            str(item) for item in value.get("read_collection_ids", [])[:10]
            if re.fullmatch(r"rag_[a-f0-9]{32}", str(item))
        ]
        write_collection = str(value.get("write_collection_id", ""))
        if "knowledge:write" in permissions and not re.fullmatch(
            r"rag_[a-f0-9]{32}", write_collection
        ):
            raise CustomAgentError("Choose a destination collection for knowledge writes.")
        if "knowledge:read" not in permissions:
            read_collections = []
        web_access = _web_access(value, scopes)
        try:
            connectors = validate_connectors(value.get("connectors", []))
        except ConnectorPolicyError as error:
            raise CustomAgentError(str(error)) from error
        return (
            name, description, instructions, scopes, permissions,
            read_collections, write_collection, web_access, connectors,
        )

    def _validate_credential_references(self, connectors, actor):
        references = {
            item["credential_ref"] for item in connectors if item["credential_ref"]
        }
        if not references:
            return
        if self.credential_broker is None:
            raise CustomAgentError(
                "Authenticated connectors require the encrypted credential broker."
            )
        try:
            credentials = {
                item["id"]: item
                for item in self.credential_broker.list(actor)
            }
        except Exception as error:
            raise CustomAgentError("Credential references could not be verified.") from error
        for reference in references:
            credential = credentials.get(reference)
            if (
                credential is None
                or credential.get("provider") != "application-secret"
                or credential.get("owner") != actor
            ):
                raise CustomAgentError(
                    "Connector credential references must identify an application-secret credential."
                )

    @staticmethod
    def draft(request: str):
        prompt = str(request).strip()
        if not prompt or len(prompt) > 2000:
            raise CustomAgentError("Describe the agent in 1–2,000 characters.")
        if UNSAFE_INSTRUCTIONS.search(prompt):
            raise CustomAgentError("The requested agent asks for unsafe or secret-shaped access.")
        lower = prompt.lower()
        selectors = (
            ("cooling:read", ("fan", "cool", "thermal", "temperature")),
            ("workloads:read", ("docker", "container", "app", "model", "llm")),
            ("jobs:read", ("job", "deploy", "install", "update", "history")),
            ("assistant:read", ("assistant", "memory", "conversation")),
            ("cluster:read", ("cluster", "fleet", "node", "worker", "swarm")),
        )
        scopes = [
            scope for scope, keywords in selectors
            if any(keyword in lower for keyword in keywords)
        ]
        # Whether the agent can reach the web must follow from the job the user
        # described, not from which synonym they reached for. A substring list
        # gave public research to "find me the latest MLB scores" and withheld
        # it from "read the Raspberry Pi news page", so a user could save a
        # web-research agent that could not research the web.
        if needs_public_research(prompt):
            scopes.append("research:read")
        daily_health = any(
            phrase in lower for phrase in ("daily health", "morning health", "health check", "daily check")
        )
        if daily_health:
            scopes.extend(("cooling:read", "system:read"))
        draft = {
            "name": draft_agent_name(prompt),
            "description": prompt[:400],
            "instructions": (
                "Complete the user's stated task using only the explicitly granted "
                "capabilities. Distinguish live Vaelor facts from untrusted public research, "
                "cite evidence, explain uncertainty in beginner-friendly language, and return "
                "practical next steps. Focus on: {}"
            ).format(prompt[:1000]),
            "scopes": sorted(set(scopes)),
            "permissions": [],
            "read_collection_ids": [],
            "write_collection_id": "",
            "web_access": {
                "enabled": "research:read" in scopes,
                "allowed_domains": [],
            },
            "connectors": [],
        }
        if daily_health:
            draft["schedule_recommendation"] = {
                "name": "Morning health check",
                "prompt": "Review cooling and system health, cite the live facts, and recommend safe next steps.",
                "schedule": "every 24 hours",
                "rationale": "Daily health requests need Cooling and System read access plus an explicit morning cadence.",
            }
        return draft

    def list(self, actor: str, *, include_disabled: bool = True):
        query = "SELECT * FROM custom_agents WHERE actor=?"
        values = [actor]
        if not include_disabled:
            query += " AND enabled=1"
        query += " ORDER BY name"
        with closing(self._connect()) as connection:
            return [self._row(row) for row in connection.execute(query, values)]

    def enabled_dependencies(self):
        """Return only non-secret global identifiers needed by dependency checks."""
        with closing(self._connect()) as connection:
            rows = connection.execute(
                "SELECT id,name FROM custom_agents WHERE enabled=1 ORDER BY name"
            ).fetchall()
        return [{"id": row["id"], "name": row["name"]} for row in rows]

    def get(self, agent_id: str, actor: str):
        with closing(self._connect()) as connection:
            row = connection.execute(
                "SELECT * FROM custom_agents WHERE id=? AND actor=?",
                (agent_id, actor),
            ).fetchone()
        return self._row(row)

    def create(self, actor: str, definition: Dict[str, Any]):
        (
            name, description, instructions, scopes, permissions,
            read_collections, write_collection, web_access, connectors,
        ) = self._validated_definition(definition)
        self._validate_credential_references(connectors, actor)
        agent_id = "custom_" + uuid.uuid4().hex
        now = time.time()
        snapshot = {
            "name": name, "description": description,
            "instructions": instructions, "scopes": scopes,
            "permissions": permissions, "read_collection_ids": read_collections,
            "write_collection_id": write_collection,
            "web_access": web_access,
            "connectors": connectors,
        }
        try:
            with closing(self._connect()) as connection:
                connection.execute(
                    """
                    INSERT INTO custom_agents
                    (id,actor,name,description,instructions,scopes_json,version,
                     enabled,created_at,updated_at,permissions_json,
                     read_collections_json,write_collection_id,web_policy_json,connectors_json)
                    VALUES(?,?,?,?,?,?,1,1,?,?,?,?,?,?,?)
                    """,
                    (agent_id, actor, name, description, instructions,
                     json.dumps(scopes), now, now, json.dumps(permissions),
                     json.dumps(read_collections), write_collection,
                     json.dumps(web_access), json.dumps(connectors)),
                )
                connection.execute(
                    """
                    INSERT INTO custom_agent_revisions
                    (agent_id,version,definition_json,actor,action,created_at)
                    VALUES(?,1,?,?,?,?)
                    """,
                    (agent_id, json.dumps(snapshot), actor, "created", now),
                )
                connection.commit()
        except sqlite3.IntegrityError as error:
            raise CustomAgentError("An agent with this name already exists.") from error
        return self.get(agent_id, actor)

    def update(self, agent_id: str, actor: str, patch: Dict[str, Any]):
        current = self.get(agent_id, actor)
        if current is None:
            raise CustomAgentError("Custom agent was not found.")
        merged = {**current, **patch}
        (
            name, description, instructions, scopes, permissions,
            read_collections, write_collection, web_access, connectors,
        ) = self._validated_definition(merged)
        self._validate_credential_references(connectors, actor)
        enabled = patch.get("enabled", current["enabled"])
        if not isinstance(enabled, bool):
            raise CustomAgentError("Enabled must be true or false.")
        # Activation and archival flip only `enabled`, and a flag flip is not a
        # new version: minting one told the operator the agent "is now version 2"
        # after a no-op activate, and moved the active definition off the version
        # a pinned automation still points at. An enabled-only patch is the sole
        # exemption. An *empty* patch is NOT exempt - the app-grant save calls
        # ``update(agent_id, actor, {})`` deliberately to mint the version its
        # grants pin to (agent_app_grants.clone_for_version), so it must still
        # bump; and any patch that touches the definition mints a version as
        # before.
        enabled_only = set(patch) == {"enabled"}
        version = current["version"] + (0 if enabled_only else 1)
        now = time.time()
        snapshot = {
            "name": name, "description": description, "instructions": instructions,
            "scopes": scopes, "enabled": enabled,
            "permissions": permissions, "read_collection_ids": read_collections,
            "write_collection_id": write_collection,
            "web_access": web_access,
            "connectors": connectors,
        }
        try:
            with closing(self._connect()) as connection:
                connection.execute(
                    """
                    UPDATE custom_agents SET name=?,description=?,instructions=?,
                    scopes_json=?,version=?,enabled=?,updated_at=?,
                    permissions_json=?,read_collections_json=?,write_collection_id=?
                    ,web_policy_json=?,connectors_json=?
                    WHERE id=? AND actor=?
                    """,
                    (name, description, instructions, json.dumps(scopes), version,
                     int(enabled), now, json.dumps(permissions),
                     json.dumps(read_collections), write_collection,
                     json.dumps(web_access), json.dumps(connectors), agent_id, actor),
                )
                if not enabled_only:
                    connection.execute(
                        """
                        INSERT INTO custom_agent_revisions
                        (agent_id,version,definition_json,actor,action,created_at)
                        VALUES(?,?,?,?,?,?)
                        """,
                        (agent_id, version, json.dumps(snapshot), actor, "updated", now),
                    )
                connection.commit()
        except sqlite3.IntegrityError as error:
            raise CustomAgentError("An agent with this name already exists.") from error
        return self.get(agent_id, actor)

    def get_version(self, agent_id: str, actor: str, version: int):
        """Return the immutable definition pinned to a durable task."""
        current = self.get(agent_id, actor)
        if current is None:
            return None
        with closing(self._connect()) as connection:
            row = connection.execute(
                "SELECT definition_json FROM custom_agent_revisions "
                "WHERE agent_id=? AND version=?",
                (agent_id, int(version)),
            ).fetchone()
        if row is None:
            return None
        definition = json.loads(row["definition_json"])
        definition.setdefault("enabled", True)
        definition.setdefault("permissions", [])
        definition.setdefault("read_collection_ids", [])
        definition.setdefault("write_collection_id", "")
        # Revisions created before web policy existed must not inherit access.
        definition.setdefault("web_access", {"enabled": False, "allowed_domains": []})
        definition.setdefault("connectors", [])
        return {
            **definition,
            "id": agent_id,
            "actor": actor,
            "version": int(version),
            "custom": True,
        }

    def rollback(self, agent_id: str, actor: str, version: int):
        """Create a new active version from an immutable prior definition."""
        definition = self.get_version(agent_id, actor, version)
        if definition is None:
            raise CustomAgentError("The requested agent version was not found.")
        patch = {
            key: definition[key]
            for key in (
                "name", "description", "instructions", "scopes", "permissions",
                "read_collection_ids", "write_collection_id", "web_access", "connectors",
            )
        }
        return self.update(agent_id, actor, patch)

    def revisions(self, agent_id: str, actor: str):
        if self.get(agent_id, actor) is None:
            raise CustomAgentError("Custom agent was not found.")
        with closing(self._connect()) as connection:
            return [
                {**dict(row), "definition": json.loads(row["definition_json"])}
                for row in connection.execute(
                    """
                    SELECT version,definition_json,actor,action,created_at
                    FROM custom_agent_revisions WHERE agent_id=?
                    ORDER BY version DESC
                    """,
                    (agent_id,),
                )
            ]

    def delete(self, agent_id: str, actor: str):
        with closing(self._connect()) as connection:
            cursor = connection.execute(
                "DELETE FROM custom_agents WHERE id=? AND actor=?", (agent_id, actor)
            )
            connection.commit()
        if cursor.rowcount != 1:
            raise CustomAgentError("Custom agent was not found.")
        return {"deleted": True, "id": agent_id}
