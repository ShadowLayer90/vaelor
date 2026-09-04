"""Reviewed, versioned appliance skills loaded only when relevant."""

from __future__ import annotations

import json
import os
import re
import sqlite3
import time
import uuid
from contextlib import closing
from typing import Optional

from .runtime_paths import env_value, state_path


BUILTIN_SKILLS = [
    ("deployment", "Safe application deployment", "Plan and verify guarded Docker Compose deployments with explicit resource, port, storage, and approval checks.", "docker compose app deploy ports storage",
     "Inspect capabilities first. Prefer a curated template. Validate ports, storage, resource limits, and Compose policy. Present one approval before creating a job."),
    ("model-selection", "Local model selection", "Compare local AI models against this node's memory, storage, privacy, licensing, and performance constraints.", "llm model gguf hugging face memory quantization",
     "Compare GGUF size plus runtime overhead with available RAM and storage. Prefer Q4_K_M for balanced Pi use. Explain privacy, license, context, and speed before approval."),
    ("troubleshooting", "System troubleshooting", "Diagnose appliance health from read-only telemetry, logs, services, networking, cooling, and hardware evidence.", "diagnose health temperature fan service logs network",
     "Start with read-only telemetry and service facts. Separate symptoms from evidence. Recommend the least disruptive recovery and never claim a change ran."),
    ("recovery", "Safe recovery", "Recover failed or cancelled work using audited history, bounded retries, checkpoints, and reversible repair steps.", "failed job retry rollback recover repair",
     "Read the job event history and recovery guidance. Retry only failed or cancelled idempotent work. Prefer checkpoint restore over manual file edits."),
    ("backup", "Backup and restore", "Create, verify, and restore versioned appliance backups while preserving a clear approval and recovery trail.", "backup export restore verify",
     "Inventory what will be protected, create a versioned checkpoint, verify its manifest and readability, then require approval before restore."),
    ("remote-access", "Remote access", "Set up and operate secure KVM, VNC, console, and remote-control sessions with bounded privileges and auditing.", "kvm vnc remote desktop console session",
     "Check hardware and gateway capability first. Use one-use tickets, minimum privileges, explicit control ownership, bounded clipboard, and audited power actions."),
]
UNSAFE = re.compile(
    r"(ignore\s+(all\s+)?previous|system\s+prompt|do\s+not\s+tell\s+the\s+user|"
    r"(password|token|secret|api[_ -]?key)\s*[:=])", re.I
)
WEAK_MATCH_TERMS = {
    "a", "and", "app", "application", "check", "clear", "local", "read", "safe",
    "service", "system", "this", "using", "while", "with",
}
# Naming the subject precisely fixed the false matches and cost real ones: a
# skill called "Check my disk space" stopped answering "is my drive full?",
# because a user does not know which word the skill was named after. These are
# equivalences within one subject, applied to the question only - a skill still
# has to name its own subject to be matched at all, so this can widen recall
# without letting filler back in as evidence.
SUBJECT_SYNONYMS = {
    "disk": {"disks", "drive", "drives", "ssd", "hdd", "nvme", "microsd", "sdcard"},
    "storage": {"space", "capacity", "full", "free"},
    "temperature": {"temp", "hot", "heat", "warm", "thermal", "overheating"},
    "cooling": {"fan", "fans", "cooler", "airflow"},
    "network": {"networking", "wifi", "ethernet", "lan", "connectivity"},
    "backup": {"backups", "snapshot", "snapshots"},
}
# Both directions, so "drive" reaches a "disk" skill and "disk" reaches a
# "drive" one.
_SYNONYM_INDEX: dict[str, set] = {}
for _subject, _equivalents in SUBJECT_SYNONYMS.items():
    for _term in {_subject, *_equivalents}:
        _SYNONYM_INDEX.setdefault(_term, set()).update({_subject, *_equivalents})


def _subject_keywords(name: str, description: str) -> str:
    """Keywords that say what a proposed skill is *about*.

    Storing the whole description here made every filler word part of a
    skill's subject, so one disk-space skill matched cooling questions and
    sports questions alike and was cited as evidence on both. The subject is
    the distinctive words in the name, with the description kept as
    supporting text that can strengthen a match but never create one.
    """
    tokens = [
        token for token in re.findall(r"[a-z0-9]+", str(name).lower())
        if len(token) > 2 and token not in WEAK_MATCH_TERMS
    ]
    if not tokens:
        tokens = [
            token for token in re.findall(r"[a-z0-9]+", str(description).lower())
            if len(token) > 2 and token not in WEAK_MATCH_TERMS
        ][:6]
    return " ".join(dict.fromkeys(tokens))


class AssistantSkillError(ValueError):
    pass


class AssistantSkillStore:
    def __init__(self, database_path: Optional[str] = None):
        self.database_path = database_path or env_value(
            "VAELOR_ASSISTANT_SKILLS_DB", "PM_ASSISTANT_SKILLS_DB",
            state_path("assistant/skills.sqlite3"),
        )
        parent = os.path.dirname(self.database_path)
        if parent:
            os.makedirs(parent, mode=0o700, exist_ok=True)
        self._initialize()
        try:
            os.chmod(self.database_path, 0o600)
        except OSError:
            pass

    def _connect(self):
        connection = sqlite3.connect(self.database_path, timeout=15)
        connection.row_factory = sqlite3.Row
        return connection

    def _initialize(self):
        with closing(self._connect()) as connection:
            connection.executescript(
                """
                CREATE TABLE IF NOT EXISTS assistant_skills (
                    id TEXT PRIMARY KEY,
                    slug TEXT NOT NULL UNIQUE,
                    name TEXT NOT NULL,
                    description TEXT NOT NULL,
                    keywords TEXT NOT NULL,
                    content TEXT NOT NULL,
                    version INTEGER NOT NULL,
                    status TEXT NOT NULL,
                    provenance TEXT NOT NULL,
                    created_by TEXT NOT NULL,
                    validation_json TEXT NOT NULL,
                    created_at REAL NOT NULL,
                    updated_at REAL NOT NULL
                );
                CREATE TABLE IF NOT EXISTS assistant_skill_revisions (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    skill_id TEXT NOT NULL,
                    version INTEGER NOT NULL,
                    content TEXT NOT NULL,
                    actor TEXT NOT NULL,
                    action TEXT NOT NULL,
                    created_at REAL NOT NULL,
                    FOREIGN KEY(skill_id) REFERENCES assistant_skills(id) ON DELETE CASCADE
                );
                """
            )
            columns = {
                row["name"]
                for row in connection.execute(
                    "PRAGMA table_info(assistant_skills)"
                ).fetchall()
            }
            if "use_count" not in columns:
                connection.execute(
                    "ALTER TABLE assistant_skills ADD COLUMN use_count INTEGER NOT NULL DEFAULT 0"
                )
            if "last_used_at" not in columns:
                connection.execute(
                    "ALTER TABLE assistant_skills ADD COLUMN last_used_at REAL"
                )
            # Keywords are a skill's *subject*, and rows written before
            # _subject_keywords existed stored the whole description there.
            # The matcher now weights the subject at twice the description, so
            # those rows do not merely keep their old behaviour - they match
            # more than they used to, and a disk-space skill on an appliance
            # that has been running for a while is cited as evidence on cooling
            # questions. Nothing rewrites keywords except propose() and
            # update(), so an appliance that never edits a skill would carry
            # this forever. Rewrite them once, in place, on store open.
            if "subject_keywords" not in columns:
                # The column carries no per-row meaning; its existence is the
                # record that the rewrite below has already run.
                connection.execute(
                    "ALTER TABLE assistant_skills "
                    "ADD COLUMN subject_keywords INTEGER NOT NULL DEFAULT 1"
                )
                legacy = connection.execute(
                    """
                    SELECT id,name,description FROM assistant_skills
                    WHERE provenance NOT IN ('vaelor-builtin','pironman-builtin')
                    """
                ).fetchall()
                # Built-in rows are excluded on purpose: their keywords are
                # curated subject terms already, not a copy of the description.
                connection.executemany(
                    "UPDATE assistant_skills SET keywords=?,subject_keywords=1 WHERE id=?",
                    [
                        (_subject_keywords(row["name"], row["description"]), row["id"])
                        for row in legacy
                    ],
                )
            now = time.time()
            for slug, name, description, keywords, content in BUILTIN_SKILLS:
                connection.execute(
                    """
                    INSERT OR IGNORE INTO assistant_skills
                    (id,slug,name,description,keywords,content,version,status,
                     provenance,created_by,validation_json,created_at,updated_at)
                    VALUES(?,?,?,?,?,?,1,'active','vaelor-builtin','system',?,?,?)
                    """,
                    (uuid.uuid4().hex, slug, name, description, keywords, content,
                     json.dumps({"reviewed": True, "secret_scan": "passed"}),
                     now, now),
                )
                connection.execute(
                    """
                    UPDATE assistant_skills SET description=?, updated_at=?
                    WHERE slug=? AND provenance='vaelor-builtin' AND description=name
                    """,
                    (description, now, slug),
                )
            connection.commit()

    @staticmethod
    def _row(row):
        item = dict(row)
        item["validation"] = json.loads(item.pop("validation_json"))
        return item

    def list(self, status: Optional[str] = None):
        query = "SELECT * FROM assistant_skills"
        values = []
        if status:
            query += " WHERE status=?"
            values.append(status)
        query += " ORDER BY status,name"
        with closing(self._connect()) as connection:
            return [self._row(row) for row in connection.execute(query, values)]

    @staticmethod
    def _match_terms(text: str) -> set:
        """Tokens that carry topic, with filler and stop words removed.

        Filler used to count as evidence. A skill whose description contained
        "check how much space is free" matched "how hot is my cpu?" on
        {how, is, my} alone, so one disk-space skill was cited as evidence on
        cooling questions and on "who won the world series" - and the
        evidence-backed badge on those answers meant nothing.
        """
        return {
            token for token in re.findall(r"[a-z0-9]+", str(text).lower())
            if len(token) > 2 and token not in WEAK_MATCH_TERMS
        }

    @staticmethod
    def _asked_terms(query: str) -> set:
        """The question's topic words, plus their within-subject equivalents."""
        terms = AssistantSkillStore._match_terms(query)
        expanded = set(terms)
        for term in terms:
            expanded.update(_SYNONYM_INDEX.get(term, ()))
        return expanded

    def match(self, query: str, limit: int = 3):
        terms = self._asked_terms(query)
        ranked = []
        for skill in self.list("active"):
            # What a skill is *about* is its name and keywords. Its description
            # is supporting prose and can only strengthen a match the subject
            # already carries; it can no longer create one on its own.
            subject = self._match_terms(
                "{} {}".format(skill["name"], skill["keywords"])
            )
            supporting = self._match_terms(skill["description"]) - subject
            named = terms & subject
            if not named:
                continue
            score = 2 * len(named) + len(terms & supporting)
            ranked.append((score, skill))
        selected = [skill for _, skill in sorted(
            ranked, key=lambda item: (-item[0], item[1]["name"])
        )[:max(1, min(limit, 5))]]
        if selected:
            now = time.time()
            with closing(self._connect()) as connection:
                connection.executemany(
                    """
                    UPDATE assistant_skills
                    SET use_count = use_count + 1, last_used_at = ?
                    WHERE id = ?
                    """,
                    [(now, item["id"]) for item in selected],
                )
                connection.commit()
            for item in selected:
                item["use_count"] = int(item.get("use_count") or 0) + 1
                item["last_used_at"] = now
        return selected

    def propose(self, actor: str, name: str, description: str, content: str):
        name = str(name).strip()[:100]
        description = str(description).strip()[:300]
        content = str(content).strip()
        if not name or not description or not content or len(content) > 8000:
            raise AssistantSkillError("Name, description, and content are required; content is limited to 8,000 characters.")
        if UNSAFE.search(content) or any(
            ord(character) in range(0x202A, 0x202F) for character in content
        ):
            raise AssistantSkillError("The proposed skill contains unsafe instructions or secret-shaped data.")
        slug = "-".join(re.findall(r"[a-z0-9]+", name.lower()))[:64]
        if not slug:
            raise AssistantSkillError("Choose a valid skill name.")
        now = time.time()
        skill_id = uuid.uuid4().hex
        validation = {"secret_scan": "passed", "instruction_scan": "passed", "reviewed": False}
        try:
            with closing(self._connect()) as connection:
                connection.execute(
                    """
                    INSERT INTO assistant_skills
                    (id,slug,name,description,keywords,content,version,status,
                     provenance,created_by,validation_json,created_at,updated_at)
                    VALUES(?,?,?,?,?,?,1,'proposed','assistant-proposal',?,?,?,?)
                    """,
                    (skill_id, slug, name, description,
                     _subject_keywords(name, description), content,
                     actor, json.dumps(validation), now, now),
                )
                connection.execute(
                    "INSERT INTO assistant_skill_revisions(skill_id,version,content,actor,action,created_at) VALUES(?,1,?,?,?,?)",
                    (skill_id, content, actor, "proposed", now),
                )
                connection.commit()
        except sqlite3.IntegrityError as error:
            raise AssistantSkillError("A skill with this name already exists.") from error
        return next(item for item in self.list() if item["id"] == skill_id)

    def review(self, skill_id: str, actor: str, decision: str):
        if decision not in {"active", "rejected"}:
            raise AssistantSkillError("Choose active or rejected.")
        with closing(self._connect()) as connection:
            row = connection.execute(
                "SELECT * FROM assistant_skills WHERE id=? AND status='proposed'",
                (skill_id,),
            ).fetchone()
            if row is None:
                raise AssistantSkillError("The proposed skill was not found.")
            validation = json.loads(row["validation_json"])
            validation["reviewed"] = decision == "active"
            validation["reviewed_by"] = actor
            now = time.time()
            connection.execute(
                "UPDATE assistant_skills SET status=?,validation_json=?,updated_at=? WHERE id=?",
                (decision, json.dumps(validation), now, skill_id),
            )
            connection.execute(
                "INSERT INTO assistant_skill_revisions(skill_id,version,content,actor,action,created_at) VALUES(?,?,?,?,?,?)",
                (skill_id, row["version"], row["content"], actor, decision, now),
            )
            connection.commit()
        return next(item for item in self.list() if item["id"] == skill_id)

    def update(
        self, skill_id: str, actor: str, name: str, description: str, content: str
    ):
        name = str(name).strip()[:100]
        description = str(description).strip()[:300]
        content = str(content).strip()
        if not name or not description or not content or len(content) > 8000:
            raise AssistantSkillError(
                "Name, description, and content are required; content is limited to 8,000 characters."
            )
        if UNSAFE.search(content) or any(
            ord(character) in range(0x202A, 0x202F) for character in content
        ):
            raise AssistantSkillError(
                "The skill contains unsafe instructions or secret-shaped data."
            )
        slug = "-".join(re.findall(r"[a-z0-9]+", name.lower()))[:64]
        if not slug:
            raise AssistantSkillError("Choose a valid skill name.")
        with closing(self._connect()) as connection:
            row = connection.execute(
                "SELECT * FROM assistant_skills WHERE id=?", (skill_id,)
            ).fetchone()
            if row is None:
                raise AssistantSkillError("The skill was not found.")
            if row["provenance"] in {"vaelor-builtin", "pironman-builtin"}:
                raise AssistantSkillError("Built-in skills cannot be edited.")
            version = int(row["version"]) + 1
            now = time.time()
            validation = {
                "secret_scan": "passed",
                "instruction_scan": "passed",
                "reviewed": False,
            }
            try:
                connection.execute(
                    """
                    UPDATE assistant_skills
                    SET slug=?,name=?,description=?,keywords=?,content=?,version=?,
                        status='proposed',validation_json=?,updated_at=?
                    WHERE id=?
                    """,
                    (
                        slug, name, description,
                        _subject_keywords(name, description), content,
                        version, json.dumps(validation), now, skill_id,
                    ),
                )
                connection.execute(
                    """
                    INSERT INTO assistant_skill_revisions
                    (skill_id,version,content,actor,action,created_at)
                    VALUES(?,?,?,?,?,?)
                    """,
                    (skill_id, version, content, actor, "updated", now),
                )
                connection.commit()
            except sqlite3.IntegrityError as error:
                raise AssistantSkillError(
                    "A skill with this name already exists."
                ) from error
        return next(item for item in self.list() if item["id"] == skill_id)

    def delete(self, skill_id: str):
        with closing(self._connect()) as connection:
            row = connection.execute(
                "SELECT provenance FROM assistant_skills WHERE id=?", (skill_id,)
            ).fetchone()
            if row is None:
                raise AssistantSkillError("The skill was not found.")
            if row["provenance"] in {"vaelor-builtin", "pironman-builtin"}:
                raise AssistantSkillError("Built-in skills cannot be deleted.")
            connection.execute(
                "DELETE FROM assistant_skill_revisions WHERE skill_id=?", (skill_id,)
            )
            connection.execute(
                "DELETE FROM assistant_skills WHERE id=?", (skill_id,)
            )
            connection.commit()
        return {"deleted": True, "id": skill_id}
