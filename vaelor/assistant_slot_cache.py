"""Bind the Assistant model's saved KV prefix to the conversation it covers.

VD-076 chose the Pi's Assistant model for its wake speed and measured the
mechanism that delivers it: llama-server's ``--slot-save-path`` with a save
after the reply and a restore before the next question. Measured on the
appliance, the first answer after an idle period costs 95.5 s without the
round-trip and 20.4 s with it, because waking otherwise pays the model reload
*and* a full standing-brief prefill. The flag has shipped since alpha 29;
nothing called ``/slots/{id}?action=save|restore`` until this module.

**The slot is keyed by the conversation** (owner's design, 2026-08-11). The KV
cache is a derived artifact of exactly the transcript plus the standing brief,
so the conversation record is its identity, its staleness guard's home, and its
cleanup path: the mapping row cascades away with the conversation.

**Two fail-closed guards, then the engine's own comparison.** A restore is
attempted only when the brief's sha256 recorded at save matches the brief this
machine would send now (the brief carries the Vaelor version, so an upgrade
correctly strands every saved slot), and when the recorded last message is
still the conversation's last message (a transcript that moved makes the
prefix unprovable - a crash between the SQLite commit and the slot write, or
the reverse, is exactly this shape). Either mismatch drops the mapping and the
turn pays the full prefill. Never serve a prefix that cannot be proved
current. Beneath both guards sits llama-server's own token-by-token prefix
comparison: a restored cache that does not match the incoming prompt is simply
not reused (VD-076 measured that as the hybrid model's 0% hit), so the failure
mode of a wrong restore is a wasted 40 ms, not a wrong answer.

**The storage bound is a pool of three fixed filenames**, not a deletion
policy. ~122 MB per saved slot means the owner's cap of the three most recent
conversations is ~366 MB. A fourth conversation's save claims the least
recently saved filename and the evicted conversation's mapping row is removed;
the engine overwrites the file in place. Vaelor never needs write access to
``SLOT_CACHE_DIR`` - the mount is written only by the container - and no
lifecycle ever deletes a file. Deleting a conversation removes its mapping
(nothing can restore the bytes without one); the bytes themselves persist on
the appliance's own disk until the pool slot is reused.

Assistant only. AI Chat has its own store and, on the Pi, no model (#143).
A server deployed without ``slot_save`` in its catalog runtime refuses the
slot endpoints; that error is treated as "no fast wake", never as a failed
turn.
"""

from __future__ import annotations

import hashlib
import http.client
import json
import sqlite3
import threading
import time
import urllib.request
from contextlib import closing
from typing import Any, Dict, Optional, Tuple

from .provider_runtime import managed_local_connection

#: Everything a slot operation may legitimately fail with: the engine
#: (connection refused, HTTP error, a status line or body cut off mid-read by
#: an OOM-killed container), a malformed response, or SQLite contention from
#: the threaded control plane. "Nothing here may fail the turn" is only a
#: contract if this list is what the public methods actually absorb - the
#: first version caught two of the four and a half-dead engine could 500 a
#: chat turn whose answer already existed.
RECOVERABLE_ERRORS = (
    OSError, ValueError, http.client.HTTPException, sqlite3.Error,
)

#: `--parallel 1` (#109) means the server has exactly one slot.
SLOT_ID = 0

#: The owner's cap, expressed as names rather than as a count to enforce:
#: three files exist at most because only three names are ever written.
SLOT_POOL: Tuple[str, ...] = (
    "assistant-recent-0.kvslot",
    "assistant-recent-1.kvslot",
    "assistant-recent-2.kvslot",
)

#: Save measured 90-134 ms and restore 22-40 ms on the appliance (VD-076);
#: the timeout is headroom for a loaded machine, not an expectation.
SLOT_TIMEOUT_SECONDS = 15


def brief_fingerprint(brief_text: str) -> str:
    """The identity of the standing brief, as bytes rather than as meaning."""
    return hashlib.sha256(str(brief_text or "").encode("utf-8")).hexdigest()


def slots_url(base_url: str) -> str:
    """The engine's slot endpoint, from an OpenAI-style base URL.

    The chat connection carries ``.../v1``; the slot API lives at the server
    root beside it.
    """
    base = str(base_url or "").rstrip("/")
    if base.endswith("/v1"):
        base = base[: -len("/v1")].rstrip("/")
    return "{}/slots/{}".format(base, SLOT_ID)


class AssistantSlotCache:
    """The conversation-to-slot mapping, and the save/restore calls it gates.

    Owns its own table in the assistant store's database rather than growing
    the store: the mapping is derived cache state, the store holds the primary
    record, and the foreign key is what ties their lifecycles - deleting a
    conversation cascades the mapping away through the store's own connection.
    """

    def __init__(
        self,
        store: Any,
        *,
        timeout_seconds: int = SLOT_TIMEOUT_SECONDS,
        opener=urllib.request.urlopen,
    ):
        self.database_path = store.database_path
        self.timeout_seconds = int(timeout_seconds)
        self._opener = opener
        # Serializes claim-and-record: without it two first-saves in the
        # threaded control plane both read the pool free, both claim the same
        # name, and the loser dies on UNIQUE(filename). The lock does NOT span
        # the model answer, so interleaved conversations can still save each
        # other's engine state under their own name - that costs the fast
        # wake (the engine's token comparison reuses nothing), never an
        # answer, and is accepted rather than serializing whole chat turns.
        self._save_lock = threading.Lock()
        with closing(self._connect()) as connection:
            connection.execute(
                """
                CREATE TABLE IF NOT EXISTS assistant_kv_slots (
                    conversation_id TEXT PRIMARY KEY
                        REFERENCES assistant_conversations(id) ON DELETE CASCADE,
                    filename TEXT NOT NULL UNIQUE,
                    message_id INTEGER NOT NULL,
                    brief_sha256 TEXT NOT NULL,
                    saved_at INTEGER NOT NULL
                )
                """
            )
            connection.commit()

    def _connect(self) -> sqlite3.Connection:
        connection = sqlite3.connect(str(self.database_path), timeout=10)
        connection.row_factory = sqlite3.Row
        connection.execute("PRAGMA foreign_keys = ON")  # pairs-with: sqlite-foreign-keys-spaced
        return connection

    def slot_for(self, actor: str, conversation_id: str) -> Optional[Dict[str, Any]]:
        with closing(self._connect()) as connection:
            row = connection.execute(
                """
                SELECT s.conversation_id, s.filename, s.message_id,
                       s.brief_sha256, s.saved_at
                FROM assistant_kv_slots s
                JOIN assistant_conversations c ON c.id = s.conversation_id
                WHERE s.conversation_id = ? AND c.actor = ?
                """,
                (conversation_id, actor),
            ).fetchone()
        return dict(row) if row is not None else None

    def drop(self, conversation_id: str) -> None:
        with closing(self._connect()) as connection:
            connection.execute(
                "DELETE FROM assistant_kv_slots WHERE conversation_id = ?",
                (conversation_id,),
            )
            connection.commit()

    def _last_message_id(
        self, connection: sqlite3.Connection, conversation_id: str
    ) -> Optional[int]:
        row = connection.execute(
            """
            SELECT id FROM assistant_messages
            WHERE conversation_id = ? ORDER BY id DESC LIMIT 1
            """,
            (conversation_id,),
        ).fetchone()
        return int(row["id"]) if row is not None else None

    def _claim_filename(self, actor: str, conversation_id: str) -> Optional[str]:
        """The pool name this conversation's save may write.

        Its own name if it has one; a never-used name otherwise; failing both,
        the least recently saved conversation is evicted and its name reused.
        The evicted row is removed *before* the overwrite is attempted, so no
        moment exists where a mapping points at a file being rewritten.
        """
        with closing(self._connect()) as connection:
            owner = connection.execute(
                "SELECT id FROM assistant_conversations WHERE id = ? AND actor = ?",
                (conversation_id, actor),
            ).fetchone()
            if owner is None:
                return None
            mine = connection.execute(
                "SELECT filename FROM assistant_kv_slots WHERE conversation_id = ?",
                (conversation_id,),
            ).fetchone()
            if mine is not None and str(mine["filename"]) in SLOT_POOL:
                return str(mine["filename"])
            if mine is not None:
                # The same tampered-table defence the restore path has: a
                # stored name outside the pool's own vocabulary is never
                # handed to the engine, not even to overwrite.
                connection.execute(
                    "DELETE FROM assistant_kv_slots WHERE conversation_id = ?",
                    (conversation_id,),
                )
                connection.commit()
            used = {
                str(row["filename"]): row
                for row in connection.execute(
                    "SELECT filename, conversation_id, saved_at FROM assistant_kv_slots"
                ).fetchall()
            }
            for name in SLOT_POOL:
                if name not in used:
                    return name
            evicted = min(
                (row for row in used.values() if row["filename"] in SLOT_POOL),
                key=lambda row: (int(row["saved_at"]), str(row["filename"])),
                default=None,
            )
            if evicted is None:
                # Every pool name is held by a row outside the pool's own
                # vocabulary - a tampered table. Refuse rather than invent.
                return None
            connection.execute(
                "DELETE FROM assistant_kv_slots WHERE conversation_id = ?",
                (str(evicted["conversation_id"]),),
            )
            connection.commit()
            return str(evicted["filename"])

    def _call(self, connection: Dict[str, str], action: str, filename: str) -> None:
        request = urllib.request.Request(
            "{}?action={}".format(slots_url(connection.get("base_url", "")), action),
            data=json.dumps({"filename": filename}).encode("utf-8"),
            headers={"Content-Type": "application/json"},
            method="POST",
        )
        with self._opener(request, timeout=self.timeout_seconds) as response:
            response.read(64 * 1024)

    def restore(
        self,
        connection: Optional[Dict[str, str]],
        *,
        actor: str,
        conversation_id: str,
        brief_text: str,
    ) -> Dict[str, Any]:
        """Warm the slot with this conversation's saved prefix, or say why not.

        Every refusal is fail-closed: the mapping is dropped so the claim
        cannot be retried against state that has already disproved it, and the
        turn continues into a full prefill. Nothing here may fail the turn.
        """
        if not connection or not managed_local_connection(connection):
            return {"restored": False, "reason": "not-managed-local"}
        try:
            row = self.slot_for(actor, str(conversation_id))
            if row is None:
                return {"restored": False, "reason": "no-saved-slot"}
            if row["filename"] not in SLOT_POOL:
                self.drop(str(conversation_id))
                return {"restored": False, "reason": "unknown-filename"}
            if row["brief_sha256"] != brief_fingerprint(brief_text):
                # The brief carries the Vaelor version (#144), so an upgrade
                # lands here: the machine the prefix described no longer
                # exists.
                self.drop(str(conversation_id))
                return {"restored": False, "reason": "brief-changed"}
            with closing(self._connect()) as database:
                last = self._last_message_id(database, str(conversation_id))
            if last != int(row["message_id"]):
                self.drop(str(conversation_id))
                return {"restored": False, "reason": "transcript-moved"}
            try:
                self._call(connection, "restore", str(row["filename"]))
            except RECOVERABLE_ERRORS:
                self.drop(str(conversation_id))
                return {"restored": False, "reason": "restore-failed"}
            return {"restored": True, "filename": str(row["filename"])}
        except RECOVERABLE_ERRORS:
            # SQLite contention, a half-dead engine, anything else on the
            # recoverable list: the turn proceeds into the full prefill.
            return {"restored": False, "reason": "slot-cache-unavailable"}

    def save(
        self,
        connection: Optional[Dict[str, str]],
        *,
        actor: str,
        conversation_id: str,
        message_id: int,
        brief_text: str,
    ) -> Dict[str, Any]:
        """Save the slot where the reply was appended, and record what it covers.

        The write already happening is the correct moment (owner's design):
        llama-server owns the idle timer and never announces a sleep, and a
        save at reply time makes knowing the timer unnecessary. The row is
        recorded only after the engine confirms the save; a failed save drops
        any previous claim this conversation held, because the file may now be
        half-written. Nothing here may fail the turn.
        """
        if not connection or not managed_local_connection(connection):
            return {"saved": False, "reason": "not-managed-local"}
        try:
            # Claim and record under one lock, or two concurrent first-saves
            # both see the same pool name free and the loser dies on
            # UNIQUE(filename) after its answer already exists.
            with self._save_lock:
                filename = self._claim_filename(actor, str(conversation_id))
                if filename is None:
                    return {"saved": False, "reason": "conversation-unknown"}
                try:
                    self._call(connection, "save", filename)
                except RECOVERABLE_ERRORS:
                    self.drop(str(conversation_id))
                    return {"saved": False, "reason": "save-failed"}
                now = int(time.time())
                with closing(self._connect()) as database:
                    database.execute(
                        """
                        INSERT INTO assistant_kv_slots
                            (conversation_id, filename, message_id,
                             brief_sha256, saved_at)
                        VALUES (?, ?, ?, ?, ?)
                        ON CONFLICT(conversation_id) DO UPDATE SET
                            filename = excluded.filename,
                            message_id = excluded.message_id,
                            brief_sha256 = excluded.brief_sha256,
                            saved_at = excluded.saved_at
                        """,
                        (
                            str(conversation_id), filename, int(message_id),
                            brief_fingerprint(brief_text), now,
                        ),
                    )
                    database.commit()
            return {"saved": True, "filename": filename}
        except RECOVERABLE_ERRORS:
            return {"saved": False, "reason": "slot-cache-unavailable"}
