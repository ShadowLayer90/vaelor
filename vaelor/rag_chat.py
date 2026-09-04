"""Persistent AI Chat conversations and cited local RAG collections."""

from __future__ import annotations

import hashlib
import json
import os
import re
import sqlite3
import time
import uuid
from contextlib import closing
from typing import Any, Dict, Optional, Sequence

from .conversation_titles import conversation_title
from .runtime_paths import env_value, state_path


MAX_DOCUMENT_BYTES = 2 * 1024 * 1024
MAX_COLLECTIONS = 50
MAX_DOCUMENTS_PER_COLLECTION = 200
MAX_CHUNKS_PER_DOCUMENT = 4000
# One request may search at most this many collections.
MAX_SELECTED_COLLECTIONS = 10


class RagChatError(ValueError):
    pass


# Keys a stored message may carry about itself. `failed` is the one that
# matters: without it the only way to know a turn failed was to have watched it
# happen, so a reload turned a recorded failure into something indistinguishable
# from an answer and took the Retry affordance with it. Reading it back out of
# the message text was considered and rejected - a model may legitimately write
# "this request failed" - so it is recorded, bounded, and nothing else.
MESSAGE_METADATA_KEYS = ("failed", "error_code", "failed_model")


def _stored_json(raw: Any, default: Any) -> Any:
    """Read a JSON column, treating anything unreadable as absent.

    `messages()` is the only way to open a saved conversation, so a column that
    will not parse used to make the entire chat unreachable rather than making
    one message's citations or self-description unknown. An empty column was
    already tolerated; a corrupt one is the same problem and gets the same
    answer.
    """
    try:
        return json.loads(raw) if raw else default
    except (TypeError, ValueError):
        return default


def _message_metadata(value: Any) -> Dict[str, Any]:
    """A bounded, allowlisted view of what a message claims about itself."""
    if not isinstance(value, dict):
        return {}
    metadata: Dict[str, Any] = {}
    if value.get("failed") is True:
        metadata["failed"] = True
    for key in ("error_code", "failed_model"):
        text = str(value.get(key, "")).strip()[:200]
        if text:
            metadata[key] = text
    return metadata


def _chunks(text: str, size: int = 900, overlap: int = 140):
    clean = re.sub(r"\r\n?", "\n", text).strip()
    if not clean:
        return []
    result = []
    start = 0
    while start < len(clean) and len(result) < MAX_CHUNKS_PER_DOCUMENT:
        end = min(len(clean), start + size)
        if end < len(clean):
            boundary = max(clean.rfind("\n", start, end), clean.rfind(" ", start, end))
            if boundary > start + size // 2:
                end = boundary
        part = clean[start:end].strip()
        if part:
            result.append(part)
        if end >= len(clean):
            break
        start = max(start + 1, end - overlap)
    return result


class RagChatStore:
    def __init__(self, database_path: Optional[str] = None):
        self.database_path = database_path or env_value(
            "VAELOR_RAG_CHAT_DB", "PM_RAG_CHAT_DB",
            state_path("assistant/rag-chat.sqlite3"),
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
        connection.execute("PRAGMA foreign_keys=ON")  # pairs-with: sqlite-foreign-keys-tight
        return connection

    def _initialize(self):
        with closing(self._connect()) as connection:
            connection.executescript(
                """
                PRAGMA journal_mode=WAL;
                CREATE TABLE IF NOT EXISTS rag_collections (
                    id TEXT PRIMARY KEY, actor TEXT NOT NULL, name TEXT NOT NULL,
                    description TEXT NOT NULL, created_at REAL NOT NULL,
                    updated_at REAL NOT NULL, UNIQUE(actor,name)
                );
                CREATE TABLE IF NOT EXISTS rag_documents (
                    id TEXT PRIMARY KEY, collection_id TEXT NOT NULL, actor TEXT NOT NULL,
                    name TEXT NOT NULL, media_type TEXT NOT NULL, sha256 TEXT NOT NULL,
                    size_bytes INTEGER NOT NULL, chunk_count INTEGER NOT NULL,
                    created_at REAL NOT NULL,
                    FOREIGN KEY(collection_id) REFERENCES rag_collections(id) ON DELETE CASCADE,
                    UNIQUE(collection_id,sha256)
                );
                CREATE TABLE IF NOT EXISTS rag_chunks (
                    id TEXT PRIMARY KEY, document_id TEXT NOT NULL,
                    collection_id TEXT NOT NULL, ordinal INTEGER NOT NULL,
                    content TEXT NOT NULL,
                    FOREIGN KEY(document_id) REFERENCES rag_documents(id) ON DELETE CASCADE,
                    FOREIGN KEY(collection_id) REFERENCES rag_collections(id) ON DELETE CASCADE
                );
                CREATE VIRTUAL TABLE IF NOT EXISTS rag_chunks_fts USING fts5(
                    content, chunk_id UNINDEXED, collection_id UNINDEXED,
                    tokenize='porter unicode61'
                );
                CREATE TABLE IF NOT EXISTS ai_chat_conversations (
                    id TEXT PRIMARY KEY, actor TEXT NOT NULL, title TEXT NOT NULL,
                    model TEXT NOT NULL, collections_json TEXT NOT NULL,
                    archived INTEGER NOT NULL DEFAULT 0, created_at REAL NOT NULL,
                    updated_at REAL NOT NULL
                );
                CREATE TABLE IF NOT EXISTS ai_chat_messages (
                    id INTEGER PRIMARY KEY AUTOINCREMENT, conversation_id TEXT NOT NULL,
                    role TEXT NOT NULL, content TEXT NOT NULL,
                    citations_json TEXT NOT NULL DEFAULT '[]', created_at REAL NOT NULL,
                    retrieval_json TEXT NOT NULL DEFAULT '{}',
                    model TEXT NOT NULL DEFAULT '',
                    metadata_json TEXT NOT NULL DEFAULT '{}',
                    FOREIGN KEY(conversation_id) REFERENCES ai_chat_conversations(id) ON DELETE CASCADE
                );
                CREATE TABLE IF NOT EXISTS ai_chat_preferences (
                    actor TEXT PRIMARY KEY, model TEXT NOT NULL,
                    collection_ids_json TEXT NOT NULL, updated_at REAL NOT NULL
                );
                """
            )
            # Whether a search ran is part of the answer, so a chat that was
            # saved before this column existed must still open. Older messages
            # simply report no recorded search rather than claiming one.
            columns = {
                row["name"]
                for row in connection.execute("PRAGMA table_info(ai_chat_messages)")
            }
            if "retrieval_json" not in columns:
                connection.execute(
                    "ALTER TABLE ai_chat_messages "
                    "ADD COLUMN retrieval_json TEXT NOT NULL DEFAULT '{}'"
                )
            # The model that produced an answer is a property of that answer,
            # not of the picker's current value. Older messages record no
            # model rather than borrowing today's selection.
            if "model" not in columns:
                connection.execute(
                    "ALTER TABLE ai_chat_messages ADD COLUMN model TEXT NOT NULL DEFAULT ''"
                )
            # Whether a stored turn is a recorded failure or a real reply is
            # not something a reader can infer from the text: a model may
            # legitimately write "this request failed". Before this, a reload
            # turned a failure into something that looked like an answer, and
            # the Retry affordance vanished with the page. It is recorded
            # explicitly, and messages saved earlier simply claim nothing.
            if "metadata_json" not in columns:
                connection.execute(
                    "ALTER TABLE ai_chat_messages "
                    "ADD COLUMN metadata_json TEXT NOT NULL DEFAULT '{}'"
                )
            connection.commit()

    def preference(self, actor: str):
        with closing(self._connect()) as connection:
            row = connection.execute(
                "SELECT * FROM ai_chat_preferences WHERE actor=?", (actor,)
            ).fetchone()
            if row is None:
                return {"model": "", "collection_ids": []}
            raw = json.loads(row["collection_ids_json"])
            owned = {
                item["id"] for item in connection.execute(
                    "SELECT id FROM rag_collections WHERE actor=?", (actor,)
                )
            }
            selected = [item for item in raw if item in owned]
            if selected != raw:
                connection.execute(
                    """
                    UPDATE ai_chat_preferences
                    SET collection_ids_json=?,updated_at=? WHERE actor=?
                    """,
                    (json.dumps(selected), time.time(), actor),
                )
                connection.commit()
        return {"model": row["model"], "collection_ids": selected}

    def set_preference(self, actor: str, model: str, collection_ids: list[str]):
        with closing(self._connect()) as connection:
            owned = {
                row["id"] for row in connection.execute(
                    "SELECT id FROM rag_collections WHERE actor=?", (actor,)
                )
            }
            selected = [
                item for item in collection_ids if item in owned
            ][:MAX_SELECTED_COLLECTIONS]
            model_text = str(model).strip()[:200]
            # An empty model means "leave what answered in this chat alone", not
            # "forget it" (LESSONS 6 / #205). `api_chat_routes.conversation_patch`
            # already preserves the stored model on an empty value and says why -
            # "a model that produced nothing still relabelled the chat" - and
            # this writer answered the same "did anything actually answer"
            # question the opposite way, blanking the preference on
            # PATCH {model:""}. Two mechanisms, one question; they must agree.
            # The DO UPDATE clause omits `model` when empty so an existing row
            # keeps it; the VALUES row supplies '' only when no row exists yet.
            model_update = (
                "model=excluded.model," if model_text else ""
            )
            connection.execute(
                """
                INSERT INTO ai_chat_preferences(actor,model,collection_ids_json,updated_at)
                VALUES(?,?,?,?)
                ON CONFLICT(actor) DO UPDATE SET {}
                collection_ids_json=excluded.collection_ids_json,
                updated_at=excluded.updated_at
                """.format(model_update),
                (actor, model_text, json.dumps(selected), time.time()),
            )
            connection.commit()
        return self.preference(actor)

    def collections(self, actor: str):
        with closing(self._connect()) as connection:
            rows = connection.execute(
                """
                SELECT c.*, COUNT(DISTINCT d.id) AS document_count,
                       COUNT(ch.id) AS chunk_count,
                       COALESCE(SUM(DISTINCT d.size_bytes),0) AS size_bytes
                FROM rag_collections c
                LEFT JOIN rag_documents d ON d.collection_id=c.id
                LEFT JOIN rag_chunks ch ON ch.document_id=d.id
                WHERE c.actor=? GROUP BY c.id ORDER BY c.name
                """,
                (actor,),
            ).fetchall()
        return [dict(row) for row in rows]

    def create_collection(self, actor: str, name: str, description: str = ""):
        name = str(name).strip()[:100]
        description = str(description).strip()[:500]
        if not name:
            raise RagChatError("Collection name is required.")
        if len(self.collections(actor)) >= MAX_COLLECTIONS:
            raise RagChatError("Collection limit reached.")
        item_id = "rag_" + uuid.uuid4().hex
        now = time.time()
        try:
            with closing(self._connect()) as connection:
                connection.execute(
                    "INSERT INTO rag_collections VALUES(?,?,?,?,?,?)",
                    (item_id, actor, name, description, now, now),
                )
                connection.commit()
        except sqlite3.IntegrityError as error:
            raise RagChatError("A collection with this name already exists.") from error
        return next(item for item in self.collections(actor) if item["id"] == item_id)

    def _owned_collection(self, connection, actor: str, collection_id: str):
        row = connection.execute(
            "SELECT * FROM rag_collections WHERE id=? AND actor=?",
            (collection_id, actor),
        ).fetchone()
        if row is None:
            raise RagChatError("RAG collection was not found.")
        return row

    def ingest(
        self, actor: str, collection_id: str, name: str,
        content: str, media_type: str = "text/plain",
    ):
        name = str(name).strip()[:180]
        content = str(content)
        encoded = content.encode("utf-8")
        if not name or not encoded or len(encoded) > MAX_DOCUMENT_BYTES:
            raise RagChatError("Document name and text are required; maximum size is 2 MiB.")
        parts = _chunks(content)
        if not parts:
            raise RagChatError("The document did not contain indexable text.")
        digest = hashlib.sha256(encoded).hexdigest()
        document_id = "doc_" + uuid.uuid4().hex
        now = time.time()
        try:
            with closing(self._connect()) as connection:
                self._owned_collection(connection, actor, collection_id)
                count = connection.execute(
                    "SELECT COUNT(*) FROM rag_documents WHERE collection_id=?",
                    (collection_id,),
                ).fetchone()[0]
                if count >= MAX_DOCUMENTS_PER_COLLECTION:
                    raise RagChatError("This collection has reached its document limit.")
                connection.execute(
                    """
                    INSERT INTO rag_documents
                    (id,collection_id,actor,name,media_type,sha256,size_bytes,
                     chunk_count,created_at) VALUES(?,?,?,?,?,?,?,?,?)
                    """,
                    (document_id, collection_id, actor, name, str(media_type)[:100],
                     digest, len(encoded), len(parts), now),
                )
                for ordinal, part in enumerate(parts):
                    chunk_id = "chunk_" + uuid.uuid4().hex
                    connection.execute(
                        "INSERT INTO rag_chunks VALUES(?,?,?,?,?)",
                        (chunk_id, document_id, collection_id, ordinal, part),
                    )
                    connection.execute(
                        "INSERT INTO rag_chunks_fts(content,chunk_id,collection_id) VALUES(?,?,?)",
                        (part, chunk_id, collection_id),
                    )
                connection.execute(
                    "UPDATE rag_collections SET updated_at=? WHERE id=?",
                    (now, collection_id),
                )
                connection.commit()
        except sqlite3.IntegrityError as error:
            raise RagChatError("This exact document is already indexed.") from error
        return self.document(actor, document_id)

    def documents(self, actor: str, collection_id: str):
        with closing(self._connect()) as connection:
            self._owned_collection(connection, actor, collection_id)
            return [
                dict(row) for row in connection.execute(
                    """
                    SELECT id,collection_id,name,media_type,sha256,size_bytes,
                           chunk_count,created_at FROM rag_documents
                    WHERE collection_id=? AND actor=? ORDER BY created_at DESC
                    """,
                    (collection_id, actor),
                )
            ]

    def document(self, actor: str, document_id: str):
        with closing(self._connect()) as connection:
            row = connection.execute(
                "SELECT * FROM rag_documents WHERE id=? AND actor=?",
                (document_id, actor),
            ).fetchone()
        if row is None:
            raise RagChatError("Document was not found.")
        return dict(row)

    def owned_collections(self, actor: str, collection_ids: Sequence[str]) -> list[str]:
        """The subset of a requested selection this actor can actually read.

        Callers need this before they can honestly report what was searched,
        and asking here keeps ownership a property of the store rather than
        something each route re-derives.
        """
        requested = [str(item) for item in collection_ids][:MAX_SELECTED_COLLECTIONS]
        if not requested:
            return []
        with closing(self._connect()) as connection:
            owned = {
                row["id"] for row in connection.execute(
                    "SELECT id FROM rag_collections WHERE actor=?", (actor,)
                )
            }
        return [item for item in requested if item in owned]

    def retrieve(self, actor: str, query: str, collection_ids: list[str], limit: int = 5):
        terms = re.findall(r"[A-Za-z0-9]{2,}", str(query))[:20]
        if not terms or not collection_ids:
            return []
        expression = " OR ".join('"{}"'.format(term.replace('"', "")) for term in terms)
        with closing(self._connect()) as connection:
            owned = {
                row["id"] for row in connection.execute(
                    "SELECT id FROM rag_collections WHERE actor=?", (actor,)
                )
            }
            selected = [
                item for item in collection_ids if item in owned
            ][:MAX_SELECTED_COLLECTIONS]
            if not selected:
                return []
            # The placeholders and the bound values must come from the same
            # list. Building them from the unfiltered request and binding the
            # filtered one turned "an administrator deleted a collection while
            # this tab was open" into an unhandled 500.
            placeholders = ",".join("?" for _ in selected)
            rows = connection.execute(
                f"""
                SELECT ch.id,ch.ordinal,ch.content,d.id AS document_id,
                       d.name AS document_name,c.id AS collection_id,
                       c.name AS collection_name,bm25(rag_chunks_fts) AS score
                FROM rag_chunks_fts
                JOIN rag_chunks ch ON ch.id=rag_chunks_fts.chunk_id
                JOIN rag_documents d ON d.id=ch.document_id
                JOIN rag_collections c ON c.id=ch.collection_id
                WHERE rag_chunks_fts MATCH ?
                  AND ch.collection_id IN ({placeholders})
                ORDER BY score LIMIT ?
                """,
                (expression, *selected, max(1, min(limit, 10))),
            ).fetchall()
        return [dict(row) for row in rows]

    def delete_document(self, actor: str, document_id: str):
        with closing(self._connect()) as connection:
            row = connection.execute(
                "SELECT id FROM rag_documents WHERE id=? AND actor=?",
                (document_id, actor),
            ).fetchone()
            if row is None:
                raise RagChatError("Document was not found.")
            chunk_ids = [
                item["id"] for item in connection.execute(
                    "SELECT id FROM rag_chunks WHERE document_id=?", (document_id,)
                )
            ]
            connection.executemany(
                "DELETE FROM rag_chunks_fts WHERE chunk_id=?", [(item,) for item in chunk_ids]
            )
            connection.execute("DELETE FROM rag_documents WHERE id=?", (document_id,))
            connection.commit()
        return {"deleted": True, "id": document_id}

    def delete_collection(self, actor: str, collection_id: str):
        for document in self.documents(actor, collection_id):
            self.delete_document(actor, document["id"])
        with closing(self._connect()) as connection:
            cursor = connection.execute(
                "DELETE FROM rag_collections WHERE id=? AND actor=?",
                (collection_id, actor),
            )
            connection.commit()
        if cursor.rowcount != 1:
            raise RagChatError("RAG collection was not found.")
        preference = self.preference(actor)
        if collection_id in preference["collection_ids"]:
            self.set_preference(
                actor,
                preference["model"],
                [
                    item for item in preference["collection_ids"]
                    if item != collection_id
                ],
            )
        return {"deleted": True, "id": collection_id}

    def ensure_conversation(
        self, actor: str, conversation_id: str = "", *,
        title: str = "", model: str = "", collections: Optional[list[str]] = None,
    ):
        if conversation_id:
            with closing(self._connect()) as connection:
                row = connection.execute(
                    "SELECT * FROM ai_chat_conversations WHERE id=? AND actor=?",
                    (conversation_id, actor),
                ).fetchone()
            if row is None:
                raise RagChatError("AI Chat conversation was not found.")
            return self._conversation_row(row)
        item_id = "chat_" + uuid.uuid4().hex
        now = time.time()
        with closing(self._connect()) as connection:
            connection.execute(
                """
                INSERT INTO ai_chat_conversations
                (id,actor,title,model,collections_json,created_at,updated_at)
                VALUES(?,?,?,?,?,?,?)
                """,
                (item_id, actor,
                 conversation_title(title, limit=100, fallback="New AI chat"),
                 str(model)[:200], json.dumps(collections or []), now, now),
            )
            connection.commit()
        return self.ensure_conversation(actor, item_id)

    @staticmethod
    def _conversation_row(row):
        item = dict(row)
        item["collections"] = json.loads(item.pop("collections_json"))
        item["archived"] = bool(item["archived"])
        return item

    def add_message(
        self, actor: str, conversation_id: str, role: str,
        content: str, citations: Optional[list[Dict[str, Any]]] = None,
        retrieval: Optional[Dict[str, Any]] = None, model: str = "",
        metadata: Optional[Dict[str, Any]] = None,
    ):
        """Store one turn.

        ``metadata`` carries facts about the turn that its text cannot be
        trusted to reveal - above all, whether this is a recorded failure
        rather than an answer. It is stored, not inferred, because a model can
        legitimately write the same words a failure notice uses.
        """
        if role not in {"user", "assistant"} or not str(content).strip():
            raise RagChatError("A valid chat message is required.")
        self.ensure_conversation(actor, conversation_id)
        now = time.time()
        stored_metadata = _message_metadata(metadata)
        with closing(self._connect()) as connection:
            cursor = connection.execute(
                """
                INSERT INTO ai_chat_messages
                (conversation_id,role,content,citations_json,created_at,retrieval_json,model,metadata_json)
                VALUES(?,?,?,?,?,?,?,?)
                """,
                (conversation_id, role, str(content)[:12000],
                 json.dumps(citations or []), now, json.dumps(retrieval or {}),
                 str(model)[:200], json.dumps(stored_metadata)),
            )
            connection.execute(
                "UPDATE ai_chat_conversations SET updated_at=? WHERE id=?",
                (now, conversation_id),
            )
            connection.commit()
        return {
            "id": cursor.lastrowid, "conversation_id": conversation_id,
            "role": role, "content": str(content)[:12000],
            "citations": citations or [], "created_at": now,
            "retrieval": retrieval or {}, "model": str(model)[:200],
            "metadata": stored_metadata,
        }

    def add_exchange(
        self, actor: str, conversation_id: str, user_content: str,
        assistant_content: str, citations: Optional[list[Dict[str, Any]]] = None,
        retrieval: Optional[Dict[str, Any]] = None, model: str = "",
    ):
        user_text = str(user_content).strip()
        assistant_text = str(assistant_content).strip()
        if not user_text or not assistant_text:
            raise RagChatError("A complete user and assistant exchange is required.")
        self.ensure_conversation(actor, conversation_id)
        now = time.time()
        with closing(self._connect()) as connection:
            connection.execute(
                """
                INSERT INTO ai_chat_messages
                (conversation_id,role,content,citations_json,created_at)
                VALUES(?,?,?,?,?)
                """,
                (conversation_id, "user", user_text[:12000], "[]", now),
            )
            cursor = connection.execute(
                """
                INSERT INTO ai_chat_messages
                (conversation_id,role,content,citations_json,created_at,retrieval_json,model)
                VALUES(?,?,?,?,?,?,?)
                """,
                (
                    conversation_id, "assistant", assistant_text[:12000],
                    json.dumps(citations or []), now, json.dumps(retrieval or {}),
                    str(model)[:200],
                ),
            )
            connection.execute(
                "UPDATE ai_chat_conversations SET updated_at=? WHERE id=?",
                (now, conversation_id),
            )
            connection.commit()
        return {
            "id": cursor.lastrowid, "conversation_id": conversation_id,
            "role": "assistant", "content": assistant_text[:12000],
            "citations": citations or [], "created_at": now,
            "retrieval": retrieval or {}, "model": str(model)[:200],
            "metadata": {},
        }

    def messages(self, actor: str, conversation_id: str, limit: int = 100):
        self.ensure_conversation(actor, conversation_id)
        with closing(self._connect()) as connection:
            rows = connection.execute(
                """
                SELECT * FROM ai_chat_messages WHERE conversation_id=?
                ORDER BY id DESC LIMIT ?
                """,
                (conversation_id, max(1, min(limit, 200))),
            ).fetchall()
        return [
            {
                **dict(row),
                "citations": _stored_json(row["citations_json"], []),
                "retrieval": _stored_json(row["retrieval_json"], {}),
                # A reload must be able to tell a recorded failure from a
                # reply. Older rows carry nothing and claim nothing.
                "metadata": _message_metadata(
                    _stored_json(dict(row).get("metadata_json"), {})
                ),
            }
            for row in reversed(rows)
        ]

    def conversations(self, actor: str, *, archived: bool = False, query: str = ""):
        query = str(query).strip()[:100]
        with closing(self._connect()) as connection:
            if query:
                rows = connection.execute(
                    """
                    SELECT DISTINCT c.* FROM ai_chat_conversations c
                    LEFT JOIN ai_chat_messages m ON m.conversation_id=c.id
                    WHERE c.actor=? AND c.archived=?
                      AND (LOWER(c.title) LIKE ? OR LOWER(m.content) LIKE ?)
                    ORDER BY c.updated_at DESC LIMIT 100
                    """,
                    (actor, int(archived), f"%{query.lower()}%", f"%{query.lower()}%"),
                ).fetchall()
            else:
                rows = connection.execute(
                    """
                    SELECT * FROM ai_chat_conversations
                    WHERE actor=? AND archived=? ORDER BY updated_at DESC LIMIT 100
                    """,
                    (actor, int(archived)),
                ).fetchall()
        return [self._conversation_row(row) for row in rows]

    def update_conversation(self, actor: str, conversation_id: str, patch: Dict[str, Any]):
        current = self.ensure_conversation(actor, conversation_id)
        title = str(patch.get("title", current["title"])).strip()[:100]
        archived = patch.get("archived", current["archived"])
        model = str(patch.get("model", current["model"])).strip()[:200]
        requested = patch.get("collections", current["collections"])
        if not title or not isinstance(archived, bool):
            raise RagChatError("Enter a title and valid archived state.")
        with closing(self._connect()) as connection:
            owned = {
                row["id"] for row in connection.execute(
                    "SELECT id FROM rag_collections WHERE actor=?", (actor,)
                )
            }
            collections = [
                item for item in requested
                if isinstance(item, str) and item in owned
            ][:10]
            connection.execute(
                """
                UPDATE ai_chat_conversations
                SET title=?,archived=?,model=?,collections_json=?,updated_at=?
                WHERE id=? AND actor=?
                """,
                (
                    title, int(archived), model, json.dumps(collections),
                    time.time(), conversation_id, actor,
                ),
            )
            connection.commit()
        return self.ensure_conversation(actor, conversation_id)

    def fork_conversation(
        self, actor: str, conversation_id: str, through_message_id: Optional[int] = None,
    ):
        source = self.ensure_conversation(actor, conversation_id)
        messages = self.messages(actor, conversation_id, limit=200)
        if through_message_id is not None:
            ids = [item["id"] for item in messages]
            if through_message_id not in ids:
                raise RagChatError("The selected message was not found.")
            messages = messages[:ids.index(through_message_id) + 1]
        branch = self.ensure_conversation(
            actor, title=f"{source['title']} (branch)",
            model=source["model"], collections=source["collections"],
        )
        for message in messages:
            self.add_message(
                actor, branch["id"], message["role"], message["content"],
                message.get("citations", []), message.get("retrieval", {}),
                message.get("model", ""), message.get("metadata", {}),
            )
        return self.ensure_conversation(actor, branch["id"])

    def prepare_regeneration(
        self, actor: str, conversation_id: str, message_id: int,
        content: Optional[str] = None,
    ):
        conversation = self.ensure_conversation(actor, conversation_id)
        with closing(self._connect()) as connection:
            target = connection.execute(
                """
                SELECT * FROM ai_chat_messages
                WHERE id=? AND conversation_id=?
                """,
                (message_id, conversation_id),
            ).fetchone()
            if target is None:
                raise RagChatError("The selected message was not found.")
            if target["role"] == "assistant":
                target = connection.execute(
                    """
                    SELECT * FROM ai_chat_messages
                    WHERE conversation_id=? AND role='user' AND id<?
                    ORDER BY id DESC LIMIT 1
                    """,
                    (conversation_id, message_id),
                ).fetchone()
            if target is None or target["role"] != "user":
                raise RagChatError("A user prompt is required to regenerate an answer.")
            prompt = str(content if content is not None else target["content"]).strip()
            if not prompt or len(prompt) > 8000:
                raise RagChatError("Edited messages must be 1–8,000 characters.")
            connection.execute(
                "UPDATE ai_chat_messages SET content=? WHERE id=?",
                (prompt, target["id"]),
            )
            connection.execute(
                "DELETE FROM ai_chat_messages WHERE conversation_id=? AND id>?",
                (conversation_id, target["id"]),
            )
            connection.execute(
                "UPDATE ai_chat_conversations SET updated_at=? WHERE id=?",
                (time.time(), conversation_id),
            )
            connection.commit()
        history = [
            item for item in self.messages(actor, conversation_id, limit=200)
            if item["id"] != target["id"]
        ]
        return {
            "conversation": conversation,
            "prompt": prompt,
            "history": history,
            "user_message_id": target["id"],
        }

    def delete_conversation(self, actor: str, conversation_id: str):
        with closing(self._connect()) as connection:
            cursor = connection.execute(
                "DELETE FROM ai_chat_conversations WHERE id=? AND actor=?",
                (conversation_id, actor),
            )
            connection.commit()
        if cursor.rowcount != 1:
            raise RagChatError("AI Chat conversation was not found.")
        return {"deleted": True, "id": conversation_id}
