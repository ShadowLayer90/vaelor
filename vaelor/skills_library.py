"""The agent-facing skills/plugins library and per-target attach records.

This is the F2 backend: an administrator curates skill manifests here, and each
manifest can be *attached* to one model or one agent version. It mirrors the
rigor of the F1 MCP catalog backend and shares its shape - two stores over
one SQLite file, bounded/validated input, deny-by-default evaluation, and
version-pinned, administrator-global attach records.

* :class:`SkillsLibraryStore` - the curated manifests. A manifest names the
  capability ``grants`` a skill assumes and the broker ``credentials`` (by id
  only, never a secret) it needs. Two invariants are enforced at register and
  update time. **B3:** every entry in ``grants`` must already be a member of
  ``custom_agents.ALLOWED_SCOPES`` or ``ALLOWED_PERMISSIONS`` - a skill can
  never introduce a capability the scope model does not already name. And the
  ``description`` must pass ``custom_agents.UNSAFE_INSTRUCTIONS`` so a manifest
  cannot smuggle secret-shaped instructions.
* :class:`SkillAttachmentStore` - version-pinned ``{target, skill}`` records,
  ``model`` or ``agent`` targets, deny-by-default on :meth:`evaluate`.

**Who reads what, today.** The library is read at runtime: a cluster agent
deploy pins skill ids on its deployment record, and
``agent_pool_operations`` resolves each id against this library at deploy and
relaunch, accepts only read-only skills, and adds an accepted skill's read
scopes and instructions to that agent. The attachment store is NOT read by any
runtime and no screen writes it: its rows are created and revoked only through
the ``/skills/attachments`` API routes, and nothing consults :meth:`evaluate`
or :meth:`require_active` outside the tests. An attachment row therefore
grants nothing and withholds nothing from a running agent. What the store does
do, if called: :meth:`evaluate` re-derives, from the *current* library and
scope vocabulary, whether the skill is still usable, and fails closed when the
skill was removed or disabled or when a grant it declares has since left the
vocabulary.
"""

from __future__ import annotations

import json
import os
import sqlite3
import time
import uuid
from contextlib import contextmanager
from typing import Any, Callable, Dict, Iterable, List, Mapping, Optional

# Reuse the scope vocabulary and the unsafe-instruction guard the custom-agent
# definitions are held to, so a skill can never name a capability an agent
# could not, and its description is screened the same way an agent's operating
# instructions are.
from .custom_agents import (
    ALLOWED_PERMISSIONS,
    ALLOWED_SCOPES,
    UNSAFE_INSTRUCTIONS,
    UNSAFE_INSTRUCTIONS_BODY,
)

# Reuse the shared store primitives rather than re-spell them: the clock and
# request-digest helpers, plus the credential-id shape rule and sensitive-field
# rejection, live in one home (:mod:`assistant_store_common`) and are imported
# here, their neutral error translated so every rejection this module raises is
# a :class:`SkillsLibraryError`.
from .assistant_store_common import (
    StoreInputError,
    credential_id as _common_credential_id,
    digest as _digest,
    now as _now,
    reject_sensitive as _common_reject_sensitive,
)
from .runtime_paths import env_value, state_path


class SkillsLibraryError(ValueError):
    """A safe, user-presentable library or attachment error."""


class AttachmentDeniedError(SkillsLibraryError):
    """Raised when an attachment is not active for the supplied current facts."""

    def __init__(self, decision: Mapping[str, Any]):
        self.decision = dict(decision)
        super().__init__(
            "Skill attachment unavailable: {}.".format(
                ", ".join(decision.get("reasons", ["unknown"]))
            )
        )


KINDS = ("skill", "plugin")
TARGET_TYPES = ("model", "agent")
MAX_GRANTS = 32
MAX_CREDENTIALS = 16
MAX_DESCRIPTION = 2000
MAX_INSTRUCTIONS = 50000

#: Error sentences written at more than one call site keep a single home here so
#: the wording cannot drift between two raises. This is the duplicate-literal
#: guard's "give the sentence one home" repair for a within-module repeat,
#: applied in place. Each is deliberately skill-specific so none collides with
#: the F1 catalog's equivalents.
_SKILL_NOT_FOUND = "Skill was not found."
_ATTACHMENT_NOT_FOUND = "Skill attachment was not found."
_NAME_TAKEN = "A skill by this name is already in the library."
_ENABLED_MESSAGE = "The enabled flag must be true or false."
_SELECT_ATTACHMENT_BY_ID = "SELECT * FROM skill_attachments WHERE id=?"
_SELECT_ATTACHMENT_BY_IDEMPOTENCY = (
    "SELECT * FROM skill_attachments WHERE created_by=? AND idempotency_key=?"
)
#: The library's own newest-first page order. The attachment store uses the same
#: shared idiom; the library orders by name instead, spelled distinctly so the
#: two are not one literal.
_ORDER_NEWEST = " ORDER BY created_at DESC LIMIT ?"
#: A bounded-collection refusal shared by the grant and credential list
#: validators; one home so the two cannot drift.
_AT_MOST = "{} holds at most {} entries."


def _allowed_grants() -> set:
    """The scope + permission vocabulary a skill's grants must lie within.

    Read from the module globals on every call, not frozen at import, so a test
    (or a real future tightening) that shrinks ``ALLOWED_SCOPES`` immediately
    narrows what a stored attachment is allowed to carry.
    """
    return set(ALLOWED_SCOPES) | set(ALLOWED_PERMISSIONS)


def _text(value: Any, field: str, maximum: int = 120) -> str:
    result = str(value or "").strip()
    if not result or len(result) > maximum:
        raise SkillsLibraryError("{} is not valid.".format(field))
    return result


def _kind(value: Any) -> str:
    kind = str(value or "").strip().lower()
    if kind not in KINDS:
        raise SkillsLibraryError("A skill kind is one of: {}.".format(", ".join(KINDS)))
    return kind


def _target_type(value: Any) -> str:
    target = str(value or "").strip().lower()
    if target not in TARGET_TYPES:
        raise SkillsLibraryError(
            "A target type is one of: {}.".format(", ".join(TARGET_TYPES))
        )
    return target


def _target_version(value: Any) -> int:
    try:
        result = int(value)
        if result < 1:
            raise ValueError
    except (TypeError, ValueError) as error:
        raise SkillsLibraryError("The attachment target_version is invalid.") from error
    return result


def _description(value: Any) -> str:
    text = " ".join(str(value or "").split())
    if not text or len(text) > MAX_DESCRIPTION:
        raise SkillsLibraryError(
            "A skill description is required and bounded in length."
        )
    if UNSAFE_INSTRUCTIONS.search(text):
        raise SkillsLibraryError(
            "The skill description requests unsafe or secret-shaped access."
        )
    return text


def _instructions(value: Any) -> str:
    """A skill's how-to body (a SKILL.md import), kept as markdown.

    Unlike :func:`_description`, this does NOT collapse whitespace: an
    instructions body is a markdown document whose newlines and layout carry
    meaning, so only trailing whitespace is stripped. An empty body is allowed
    (a skill may legitimately carry no instructions) and returns ''; a non-empty
    body is bounded to :data:`MAX_INSTRUCTIONS` and run through the
    body-specific ``UNSAFE_INSTRUCTIONS_BODY`` screen. That screen refuses an
    imperative override ("ignore all previous instructions") or a secret
    exfiltration ("disclose the api_key") rather than the secret-shaped
    tokens the description screen rejects, so a genuine SKILL.md - which
    routinely shows ``sudo`` and ``--env HF_TOKEN=...`` - imports cleanly
    while an injected override body still cannot.
    """
    text = str(value or "").rstrip()
    if not text:
        return ""
    if len(text) > MAX_INSTRUCTIONS:
        raise SkillsLibraryError("A skill instructions body is bounded in length.")
    if UNSAFE_INSTRUCTIONS_BODY.search(text):
        raise SkillsLibraryError(
            "The skill instructions request an override or secret disclosure."
        )
    return text


def _skill_name(value: Any) -> str:
    """A skill name, screened like the description.

    The name is instruction-bearing too: F4 surfaces ``{name, description}``
    into a deployed agent's context, so an unscreened name would be an
    injection channel beside the screened description. Screen it the same way.
    """
    name = _text(value, "name", 80)
    if UNSAFE_INSTRUCTIONS.search(name):
        raise SkillsLibraryError(
            "The skill name requests unsafe or secret-shaped access."
        )
    return name


def _reject_sensitive(value: Mapping[str, Any], field: str = "input") -> None:
    """Refuse a secret-shaped field, reusing the shared detector and error text."""
    try:
        _common_reject_sensitive(value, field)
    except StoreInputError as error:
        raise SkillsLibraryError(str(error)) from error


def _credential_id(value: Any) -> str:
    """A broker credential *identifier*, never the secret - the shared shape rule."""
    try:
        return _common_credential_id(value)
    except StoreInputError as error:
        raise SkillsLibraryError(str(error)) from error


def _grant_list(values: Any, field: str = "grants") -> List[str]:
    """Validated capability grants, each a member of the scope vocabulary.

    This is the B3 invariant's home: a grant outside
    ``ALLOWED_SCOPES ∪ ALLOWED_PERMISSIONS`` is refused, so a manifest can never
    name a capability the scope model does not already define.
    """
    if isinstance(values, (str, bytes)) or not isinstance(values, Iterable):
        raise SkillsLibraryError("{} must be a list of scopes or permissions.".format(field))
    grants: List[str] = []
    for item in values:
        grant = str(item).strip()
        if grant and grant not in grants:
            grants.append(grant)
    if len(grants) > MAX_GRANTS:
        raise SkillsLibraryError(_AT_MOST.format(field, MAX_GRANTS))
    allowed = _allowed_grants()
    unknown = [grant for grant in grants if grant not in allowed]
    if unknown:
        raise SkillsLibraryError(
            "grants may name only known scopes and permissions; not recognised: "
            "{}.".format(", ".join(sorted(unknown))[:200])
        )
    return grants


def _credential_list(values: Any, field: str = "credentials") -> List[str]:
    if isinstance(values, (str, bytes)) or not isinstance(values, Iterable):
        raise SkillsLibraryError("{} must be a list of broker credential ids.".format(field))
    credentials: List[str] = []
    for item in values:
        credential = _credential_id(item)
        if credential and credential not in credentials:
            credentials.append(credential)
    if len(credentials) > MAX_CREDENTIALS:
        raise SkillsLibraryError(_AT_MOST.format(field, MAX_CREDENTIALS))
    return credentials


def register_refusals(
    *, name: Any, kind: Any = "skill", description: Any = "",
    instructions: Any = "", grants: Any = (), credentials: Any = (),
) -> List[str]:
    """Every reason :meth:`SkillsLibraryStore.register` would refuse these fields.

    Nothing is written. It runs the very validators ``register`` runs - this
    module stays the one owner of the register rules - but each one on its own,
    so a preview (the GitHub import) can name every reason at once instead of
    only the first. An empty list means the field rules accept the manifest; the
    unique-name rule needs the store, see :meth:`SkillsLibraryStore.refusals`.
    """
    checks = (
        lambda: _skill_name(name),
        lambda: _kind(kind),
        lambda: _description(description),
        lambda: _instructions(instructions),
        lambda: _grant_list(grants),
        lambda: _credential_list(credentials),
    )
    reasons: List[str] = []
    for check in checks:
        try:
            check()
        except SkillsLibraryError as error:
            reasons.append(str(error))
    return reasons


def _ensure_schema(connection: sqlite3.Connection) -> None:
    """Create BOTH library tables. Either store can bootstrap the shared file,
    so constructing :class:`SkillAttachmentStore` alone never hits a
    missing-table error.

    Attachments are administrator-managed, NOT siloed by the creating admin: the
    ``created_by`` column is an audit field, never an access key. Any
    administrator sees and revokes any attachment, and :meth:`evaluate` looks an
    attachment up by id without a username (no runtime calls it yet). The
    route's ``require_auth("administrator")`` is the real gate.
    """
    connection.executescript(
        """
        CREATE TABLE IF NOT EXISTS skills_library (
            id TEXT PRIMARY KEY,
            name TEXT NOT NULL,
            description TEXT NOT NULL DEFAULT '',
            instructions TEXT NOT NULL DEFAULT '',
            kind TEXT NOT NULL,
            enabled INTEGER NOT NULL DEFAULT 1,
            grants_json TEXT NOT NULL DEFAULT '[]',
            credentials_json TEXT NOT NULL DEFAULT '[]',
            created_at REAL NOT NULL,
            updated_at REAL NOT NULL,
            UNIQUE(name)
        );
        CREATE INDEX IF NOT EXISTS idx_skills_library_kind
            ON skills_library(kind, name);
        CREATE TABLE IF NOT EXISTS skill_attachments (
            id TEXT PRIMARY KEY,
            created_by TEXT NOT NULL,
            target_type TEXT NOT NULL,
            target_id TEXT NOT NULL,
            target_version INTEGER NOT NULL,
            skill_id TEXT NOT NULL,
            source_attachment_id TEXT,
            revoked_at REAL,
            revocation_reason TEXT NOT NULL DEFAULT '',
            idempotency_key TEXT,
            request_digest TEXT NOT NULL,
            created_at REAL NOT NULL,
            updated_at REAL NOT NULL,
            UNIQUE(created_by, idempotency_key)
        );
        CREATE INDEX IF NOT EXISTS idx_skill_attachments_target
            ON skill_attachments(target_type, target_id, target_version);
        CREATE INDEX IF NOT EXISTS idx_skill_attachments_skill
            ON skill_attachments(skill_id, created_at DESC);
        CREATE UNIQUE INDEX IF NOT EXISTS idx_skill_attachments_clone
            ON skill_attachments(target_type, target_id, target_version, source_attachment_id)
            WHERE source_attachment_id IS NOT NULL;
        """
    )
    # Idempotent add-column migration for a library created before the
    # instructions body existed (the custom_agents PRAGMA table_info + ALTER
    # idiom): a pre-existing row reads back with instructions ''.
    columns = {
        row["name"] for row in connection.execute("PRAGMA table_info(skills_library)")
    }
    if "instructions" not in columns:
        connection.execute(
            "ALTER TABLE skills_library ADD COLUMN instructions TEXT NOT NULL DEFAULT ''"
        )


class SkillsLibraryStore:
    """An administrator-curated library of attachable skill manifests."""

    def __init__(self, database_path: Optional[str] = None, clock: Callable[[], float] = _now):
        self.database_path = database_path or env_value(
            "VAELOR_SKILLS_LIBRARY_DB", "PM_SKILLS_LIBRARY_DB",
            state_path("assistant/skills-library.sqlite3"),
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
        item["grants"] = json.loads(item.pop("grants_json") or "[]")
        item["credentials"] = json.loads(item.pop("credentials_json") or "[]")
        item["enabled"] = bool(item["enabled"])
        item["instructions"] = item.get("instructions", "") or ""
        return item

    def register(
        self, *, name: str, kind: str, description: str = "",
        instructions: str = "",
        grants: Iterable[Any] = (), credentials: Iterable[Any] = (),
        enabled: bool = True,
    ) -> Dict[str, Any]:
        """Enrol a skill manifest, or refuse it (B3 + secret screening)."""
        name = _skill_name(name)
        kind = _kind(kind)
        description_text = _description(description)
        instructions_text = _instructions(instructions)
        if not isinstance(enabled, bool):
            raise SkillsLibraryError(_ENABLED_MESSAGE)
        grant_list = _grant_list(grants)
        credential_list = _credential_list(credentials)
        skill_id = "skill_" + uuid.uuid4().hex
        now = self.clock()
        try:
            with self._connection() as connection:
                connection.execute(
                    """INSERT INTO skills_library
                    (id,name,description,instructions,kind,enabled,grants_json,
                     credentials_json,created_at,updated_at)
                    VALUES(?,?,?,?,?,?,?,?,?,?)""",
                    (skill_id, name, description_text, instructions_text, kind,
                     int(enabled), json.dumps(grant_list, separators=(",", ":")),
                     json.dumps(credential_list, separators=(",", ":")), now, now),
                )
        except sqlite3.IntegrityError as error:
            raise SkillsLibraryError(_NAME_TAKEN) from error
        return self.get(skill_id)  # type: ignore[return-value]

    def refusals(self, **fields: Any) -> List[str]:
        """:func:`register_refusals` plus the store's unique-name rule.

        The name is compared exactly as ``register`` stores it (stripped, and
        matched by the column's own ``UNIQUE`` collation), so a name reported
        free here is one ``register`` accepts, and a taken one carries the same
        sentence ``register`` raises.
        """
        reasons = register_refusals(**fields)
        try:
            name = _skill_name(fields.get("name"))
        except SkillsLibraryError:
            return reasons
        with self._connection() as connection:
            taken = connection.execute(
                "SELECT 1 FROM skills_library WHERE name=? LIMIT 1", (name,)
            ).fetchone()
        if taken is not None:
            reasons.append(_NAME_TAKEN)
        return reasons

    def get(self, skill_id: str) -> Optional[Dict[str, Any]]:
        with self._connection() as connection:
            row = connection.execute(
                "SELECT * FROM skills_library WHERE id=?", (str(skill_id),)
            ).fetchone()
        return self._row(row)

    def list(self, *, enabled_only: bool = False, limit: int = 200) -> List[Dict[str, Any]]:
        limit = max(1, min(int(limit), 500))
        query = "SELECT * FROM skills_library"
        if enabled_only:
            query += " WHERE enabled = 1"
        query += " ORDER BY name ASC LIMIT ?"
        with self._connection() as connection:
            rows = connection.execute(query, (limit,)).fetchall()
        return [self._row(row) for row in rows]  # type: ignore[list-item]

    def update(self, skill_id: str, patch: Mapping[str, Any]) -> Dict[str, Any]:
        """Amend a manifest's mutable fields. ``kind`` is immutable."""
        if not isinstance(patch, Mapping):
            raise SkillsLibraryError("A skill update is an object.")
        _reject_sensitive(patch, "skill update")
        if "kind" in patch:
            raise SkillsLibraryError("A skill's kind cannot be changed after enrolment.")
        current = self.get(skill_id)
        if current is None:
            raise SkillsLibraryError(_SKILL_NOT_FOUND)
        name = _skill_name(patch.get("name", current["name"]))
        description_text = _description(patch.get("description", current["description"]))
        instructions_text = _instructions(
            patch.get("instructions", current.get("instructions", ""))
        )
        if "enabled" in patch and not isinstance(patch["enabled"], bool):
            raise SkillsLibraryError(_ENABLED_MESSAGE)
        enabled = bool(patch.get("enabled", current["enabled"]))
        grant_list = _grant_list(patch.get("grants", current["grants"]))
        credential_list = _credential_list(patch.get("credentials", current["credentials"]))
        now = self.clock()
        try:
            with self._connection() as connection:
                connection.execute(
                    "UPDATE skills_library SET name=?,description=?,instructions=?,"
                    "enabled=?,grants_json=?,credentials_json=?,updated_at=? WHERE id=?",
                    (name, description_text, instructions_text, int(enabled),
                     json.dumps(grant_list, separators=(",", ":")),
                     json.dumps(credential_list, separators=(",", ":")),
                     now, str(skill_id)),
                )
        except sqlite3.IntegrityError as error:
            raise SkillsLibraryError(_NAME_TAKEN) from error
        return self.get(skill_id)  # type: ignore[return-value]

    def remove(self, skill_id: str) -> Dict[str, Any]:
        with self._connection() as connection:
            cursor = connection.execute(
                "DELETE FROM skills_library WHERE id=?", (str(skill_id),)
            )
        if cursor.rowcount != 1:
            raise SkillsLibraryError(_SKILL_NOT_FOUND)
        return {"removed": True, "id": str(skill_id)}


class SkillAttachmentStore:
    """Version-pinned per-target skill attachments, deny-by-default on evaluate.

    Shares one SQLite file with :class:`SkillsLibraryStore` so an attachment can
    read the skill's live manifest in the same transaction. A row records that
    an administrator attached a skill to a model or an agent version through the
    API; no runtime reads these rows and no screen writes them, so a row has no
    effect on what a deployed agent may do (see the module docstring).
    :meth:`evaluate` re-derives usability from the current library and the
    current scope vocabulary and fails closed.
    """

    def __init__(self, database_path: Optional[str] = None, clock: Callable[[], float] = _now):
        self.database_path = database_path or env_value(
            "VAELOR_SKILLS_LIBRARY_DB", "PM_SKILLS_LIBRARY_DB",
            state_path("assistant/skills-library.sqlite3"),
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
        item["revoked"] = item["revoked_at"] is not None
        item.pop("request_digest", None)
        return item

    @staticmethod
    def _skill(connection: sqlite3.Connection, skill_id: str) -> Optional[Dict[str, Any]]:
        row = connection.execute(
            "SELECT id,name,enabled,grants_json FROM skills_library WHERE id=?",
            (str(skill_id),),
        ).fetchone()
        if row is None:
            return None
        item = dict(row)
        item["grants"] = json.loads(item.pop("grants_json") or "[]")
        item["enabled"] = bool(item["enabled"])
        return item

    def create(
        self, created_by: str, target_type: str, target_id: str,
        target_version: Any, skill_id: str, *,
        idempotency_key: Optional[str] = None,
    ) -> Dict[str, Any]:
        created_by = _text(created_by, "created_by", 120)
        target_type = _target_type(target_type)
        target_id = _text(target_id, "target_id", 120)
        version = _target_version(target_version)
        skill_id = _text(skill_id, "skill_id", 120)
        key = None if idempotency_key is None else _text(idempotency_key, "idempotency_key", 160)
        payload = {
            "created_by": created_by, "target_type": target_type,
            "target_id": target_id, "target_version": version, "skill_id": skill_id,
        }
        request_digest = _digest(payload)
        attachment_id = "skillatt_" + uuid.uuid4().hex
        now = self.clock()
        with self._connection() as connection:
            skill = self._skill(connection, skill_id)
            if skill is None:
                raise SkillsLibraryError("No skill with that id is in the library.")
            if key:
                existing = connection.execute(
                    _SELECT_ATTACHMENT_BY_IDEMPOTENCY,
                    (created_by, key),
                ).fetchone()
                if existing:
                    if existing["request_digest"] != request_digest:
                        raise SkillsLibraryError(
                            "The idempotency_key conflicts with an earlier attach request."
                        )
                    return self._row(existing)  # type: ignore[return-value]
            try:
                connection.execute(
                    """INSERT INTO skill_attachments
                    (id,created_by,target_type,target_id,target_version,skill_id,
                     source_attachment_id,revoked_at,revocation_reason,idempotency_key,
                     request_digest,created_at,updated_at)
                    VALUES(?,?,?,?,?,?,?,?,?,?,?,?,?)""",
                    (attachment_id, created_by, target_type, target_id, version, skill_id,
                     None, None, "", key, request_digest, now, now),
                )
            except sqlite3.IntegrityError:
                # A concurrent submit with the same (created_by, idempotency_key)
                # won the unique index; adopt its row rather than raising a 500.
                if key:
                    existing = connection.execute(
                        _SELECT_ATTACHMENT_BY_IDEMPOTENCY,
                        (created_by, key),
                    ).fetchone()
                    if existing is not None:
                        return self._row(existing)  # type: ignore[return-value]
                raise
            row = connection.execute(_SELECT_ATTACHMENT_BY_ID, (attachment_id,)).fetchone()
        return self._row(row)  # type: ignore[return-value]

    def get(self, attachment_id: str) -> Optional[Dict[str, Any]]:
        # Administrator-global: any administrator resolves any attachment by id.
        # The route's require_auth("administrator") is the gate, not the creator.
        with self._connection() as connection:
            row = connection.execute(
                _SELECT_ATTACHMENT_BY_ID, (str(attachment_id),)
            ).fetchone()
        return self._row(row)

    def list(
        self, target_id: Optional[str] = None,
        *, target_type: Optional[str] = None,
        target_version: Optional[Any] = None, limit: int = 200,
    ) -> List[Dict[str, Any]]:
        limit = max(1, min(int(limit), 500))
        query = "SELECT * FROM skill_attachments"
        clauses: List[str] = []
        values: List[Any] = []
        if target_id is not None:
            clauses.append("target_id=?")
            values.append(_text(target_id, "target_id", 120))
        if target_type is not None:
            clauses.append("target_type=?")
            values.append(_target_type(target_type))
        if target_version is not None:
            clauses.append("target_version=?")
            values.append(_target_version(target_version))
        if clauses:
            query += " WHERE " + " AND ".join(clauses)
        query += _ORDER_NEWEST
        values.append(limit)
        with self._connection() as connection:
            rows = connection.execute(query, values).fetchall()
        return [self._row(row) for row in rows]  # type: ignore[list-item]

    def revoke(self, attachment_id: str, reason: str = "") -> Dict[str, Any]:
        now = self.clock()
        with self._connection() as connection:
            connection.execute(
                "UPDATE skill_attachments SET revoked_at=?,revocation_reason=?,updated_at=? "
                "WHERE id=? AND revoked_at IS NULL",
                (now, str(reason or "revoked by a cluster administrator").strip()[:256],
                 now, str(attachment_id)),
            )
        item = self.get(attachment_id)
        if item is None:
            raise SkillsLibraryError(_ATTACHMENT_NOT_FOUND)
        return item

    def evaluate(
        self, attachment_id: str, current_state: Optional[Mapping[str, Any]] = None,
        **facts: Any,
    ) -> Dict[str, Any]:
        """Deny-by-default decision for the runtime's supplied current facts.

        The stored attachment pins only stable identifiers. Whether it is usable
        *now* depends on the live library (does the skill still exist, is it
        enabled, are its declared grants still within the scope vocabulary) and
        the runtime's current target version, supplied here rather than trusted
        from the row. Resolved by id - the F4 runtime authorises without needing
        the creating administrator's name.
        """
        attachment = self.get(attachment_id)
        if attachment is None:
            raise SkillsLibraryError(_ATTACHMENT_NOT_FOUND)
        state = dict(current_state or {})
        state.update(facts)
        if attachment["revoked"]:
            return {
                "attachment_id": attachment["id"], "status": "revoked",
                "reasons": ["attachment_revoked"], "permitted_grants": [],
                "recovery_action": "Create a new skill attachment.",
            }
        reasons: List[str] = []
        incompatible: List[str] = []
        supplied_version = state.get("target_version")
        if supplied_version is None:
            reasons.append("target_version_unavailable")
        else:
            try:
                supplied_version = int(supplied_version)
            except (TypeError, ValueError):
                supplied_version = None
            if supplied_version is None:
                reasons.append("target_version_unavailable")
            elif supplied_version != attachment["target_version"]:
                incompatible.append("target_version_stale")
        with self._connection() as connection:
            skill = self._skill(connection, attachment["skill_id"])
        if skill is None:
            reasons.append("skill_unavailable")
            declared: List[str] = []
        else:
            declared = list(skill["grants"])
            if not skill["enabled"]:
                reasons.append("skill_disabled")
        # Re-intersect the skill's declared grants with the *current* scope
        # vocabulary. A grant that has since left ALLOWED_SCOPES/PERMISSIONS is
        # no longer permitted, and if every declared grant is gone the
        # attachment is incompatible, not merely thin.
        allowed = _allowed_grants()
        permitted = [grant for grant in declared if grant in allowed]
        if skill is not None and declared and not permitted:
            incompatible.append("all_grants_denied")
        elif skill is not None and len(permitted) != len(declared):
            incompatible.append("some_grants_denied")
        if incompatible:
            status = "incompatible"
        elif reasons:
            status = "blocked"
        else:
            status = "active"
        return {
            "attachment_id": attachment["id"], "status": status,
            "reasons": sorted(set(incompatible + reasons)),
            "permitted_grants": sorted(permitted),
            "recovery_action": (
                "Review the pinned target version and the skill's declared grants."
                if status != "active" else ""
            ),
        }

    status = evaluate

    def require_active(
        self, attachment_id: str, current_state: Optional[Mapping[str, Any]] = None,
        **facts: Any,
    ) -> Dict[str, Any]:
        decision = self.evaluate(attachment_id, current_state, **facts)
        if decision["status"] != "active":
            raise AttachmentDeniedError(decision)
        record = self.get(attachment_id)
        # Hand back the vocabulary-intersected authoritative grant list, so an F4
        # caller never accidentally trusts a stale manifest.
        record["permitted_grants"] = decision["permitted_grants"]  # type: ignore[index]
        return record  # type: ignore[return-value]

    authorize = require_active

    def clone_version(
        self, target_id: str, from_version: Any, to_version: Any,
        *, target_type: str = "agent",
    ) -> Dict[str, Any]:
        """Carry live attachments forward into a new agent version.

        Meaningful for an ``agent`` target, whose definition is versioned; a
        model target is not versioned this way. The cloned rows copy the source
        ``skill_id`` verbatim - not re-checked here, because :meth:`evaluate`
        re-derives usability from the live library and scope vocabulary at
        authorization time, so a skill removed or a grant dropped since the
        source attachment was written can never be exercised through the clone.
        """
        target_id = _text(target_id, "target_id", 120)
        target_type = _target_type(target_type)
        source_version = _target_version(from_version)
        target_version = _target_version(to_version)
        if source_version == target_version:
            raise SkillsLibraryError("The clone source and target versions must differ.")
        clone_lookup = (
            "SELECT * FROM skill_attachments WHERE target_type=? AND target_id=? "
            "AND target_version=? AND source_attachment_id=?"
        )
        now = self.clock()
        with self._connection() as connection:
            sources = connection.execute(
                "SELECT * FROM skill_attachments WHERE target_type=? AND target_id=? "
                "AND target_version=? AND revoked_at IS NULL ORDER BY created_at ASC",
                (target_type, target_id, source_version),
            ).fetchall()
            cloned: List[Dict[str, Any]] = []
            for source in sources:
                existing = connection.execute(
                    clone_lookup, (target_type, target_id, target_version, source["id"]),
                ).fetchone()
                if existing is not None:
                    cloned.append(self._row(existing))  # type: ignore[arg-type]
                    continue
                attachment_id = "skillatt_" + uuid.uuid4().hex
                try:
                    connection.execute(
                        """INSERT INTO skill_attachments
                        (id,created_by,target_type,target_id,target_version,skill_id,
                         source_attachment_id,revoked_at,revocation_reason,idempotency_key,
                         request_digest,created_at,updated_at)
                        VALUES(?,?,?,?,?,?,?,?,?,?,?,?,?)""",
                        (attachment_id, source["created_by"], target_type, target_id,
                         target_version, source["skill_id"], source["id"],
                         None, "", None, source["request_digest"], now, now),
                    )
                except sqlite3.IntegrityError:
                    # A concurrent clone won the unique clone index; adopt its row.
                    existing = connection.execute(
                        clone_lookup, (target_type, target_id, target_version, source["id"]),
                    ).fetchone()
                    if existing is not None:
                        cloned.append(self._row(existing))  # type: ignore[arg-type]
                    continue
                created = connection.execute(
                    _SELECT_ATTACHMENT_BY_ID, (attachment_id,)
                ).fetchone()
                cloned.append(self._row(created))  # type: ignore[arg-type]
        return {
            "target_type": target_type, "target_id": target_id,
            "from_version": source_version, "to_version": target_version,
            "cloned": len(cloned), "attachments": cloned,
        }

    def dependents(self, skill_id: str) -> Dict[str, Any]:
        """Which attachments would be affected by removing or disabling a skill."""
        with self._connection() as connection:
            rows = connection.execute(
                "SELECT id,created_by,target_type,target_id,target_version,revoked_at "
                "FROM skill_attachments WHERE skill_id=?" + _ORDER_NEWEST,
                (str(skill_id), 500),
            ).fetchall()
        attachments = [dict(row) for row in rows]
        return {
            "skill_id": str(skill_id),
            "counts": {
                "total": len(attachments),
                "active": sum(item["revoked_at"] is None for item in attachments),
                "revoked": sum(item["revoked_at"] is not None for item in attachments),
            },
            "attachments": attachments,
        }

    dependency_impact = dependents
