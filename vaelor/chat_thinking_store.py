"""The owner's thinking step per AI Chat connection, and the stored summary view (VD-209).

The step is remembered **per connection** (VD-209 item 2), in the same SQLite
file as the rest of AI Chat's preferences (`rag_chat.RagChatStore`), in a
table of its own keyed by ``(actor, credential_id)``: the existing preference
row is one per actor, and a connection's step must not follow the owner to a
different connection.

`stored_thinking` is the bounded, allowlisted view of a turn's thinking that a
saved message may carry. Leaf module: standard library and `chat_thinking`'s
constants only.
"""

from __future__ import annotations

import sqlite3
import time
from contextlib import closing
from typing import Any, Dict

from .chat_thinking import THINKING_STEPS, bounded_summary

#: The longest credential id stored; broker ids are far shorter.
MAX_CREDENTIAL_ID = 200


class ThinkingPreferenceError(ValueError):
    """A step or connection the store refuses, in the owner's words."""


class ThinkingPreferences:
    def __init__(self, database_path: str):
        self.database_path = database_path
        with closing(self._connect()) as connection:
            connection.execute(
                """
                CREATE TABLE IF NOT EXISTS ai_chat_thinking (
                    actor TEXT NOT NULL, credential_id TEXT NOT NULL,
                    step TEXT NOT NULL, updated_at REAL NOT NULL,
                    PRIMARY KEY(actor, credential_id)
                )
                """
            )
            connection.commit()

    def _connect(self):
        connection = sqlite3.connect(self.database_path, timeout=15)
        connection.row_factory = sqlite3.Row
        return connection

    def step(self, actor: str, credential_id: str) -> str:
        """The step the owner chose for this connection, ``""`` if none yet."""
        with closing(self._connect()) as connection:
            row = connection.execute(
                "SELECT step FROM ai_chat_thinking WHERE actor=? AND credential_id=?",
                (str(actor), str(credential_id)),
            ).fetchone()
        step = str(row["step"]) if row is not None else ""
        return step if step in THINKING_STEPS else ""

    def set_step(self, actor: str, credential_id: str, step: str) -> str:
        clean = str(step or "").strip().lower()
        if clean not in THINKING_STEPS:
            raise ThinkingPreferenceError("Choose Off, Low, Medium or High.")
        credential = str(credential_id or "").strip()
        if not credential or len(credential) > MAX_CREDENTIAL_ID:
            raise ThinkingPreferenceError("Choose an AI Chat connection first.")
        with closing(self._connect()) as connection:
            connection.execute(
                """
                INSERT INTO ai_chat_thinking(actor, credential_id, step, updated_at)
                VALUES(?,?,?,?)
                ON CONFLICT(actor, credential_id) DO UPDATE SET
                step=excluded.step, updated_at=excluded.updated_at
                """,
                (str(actor), credential, clean, time.time()),
            )
            connection.commit()
        return clean


def stored_thinking(value: Any) -> Dict[str, Any]:
    """A saved turn's thinking: summary (bounded), step and measured seconds only."""
    if not isinstance(value, dict):
        return {}
    summary, truncated = bounded_summary(value.get("summary"))
    if not summary:
        return {}
    view: Dict[str, Any] = {"summary": summary}
    step = str(value.get("step") or "")
    if step in THINKING_STEPS:
        view["step"] = step
    try:
        seconds = float(value["seconds"])
    except (KeyError, TypeError, ValueError):
        seconds = -1.0
    if 0 <= seconds < 86400:
        view["seconds"] = round(seconds, 1)
    if truncated or value.get("truncated") is True:
        view["truncated"] = True
    return view
