"""Idempotent turn dedupe for the two conversational chat POST endpoints.

VD-112 follow-up (the durable half). ``frontend/src/lib/api.ts`` re-sends a
rejected POST when a pooled socket resets fast, and A97 narrowed that retry to a
one-second window — but a client link blip in the first second after a chat POST
the appliance already accepted could still re-send it, writing a SECOND user
turn and starting a SECOND inference from one user action. The ``/jobs`` deploy
path is already safe because it carries a ``crypto.randomUUID()`` idempotency
key; the two chat endpoints now carry one too, and this store is the server half
that makes a duplicate a no-op.

Each accepted turn is recorded the moment it is claimed, keyed by the sending
actor, the conversation the client addressed, and the client-minted key:

* first sight of a key → the caller proceeds, then records the finished payload;
* a duplicate arriving while the first is still answering → ``in_flight``: the
  handler returns the dropped-connection response the live-drop resume-poll
  already reconciles (VD-110 #247i), never a second inference;
* a duplicate arriving after the first finished → the stored payload is replayed
  verbatim, so no second user turn is written.

Scope uses the conversation id the client *sent*, not the one the appliance
creates, so a first-turn retry (which repeats ``conversation_id=""``) resolves to
the same record as its original. The store is bounded by TTL and entry count so
it cannot grow without limit on a long-lived control plane.
"""

from __future__ import annotations

import threading
import time
from collections import OrderedDict
from dataclasses import dataclass
from typing import Any, Callable, Dict, Optional, Tuple

# The client budgets ~900 s for an answer, so a key must outlive the longest
# turn it could still be a duplicate of; past that the original has landed and a
# late re-send is a genuinely new ask.
_DEFAULT_TTL_SECONDS = 900.0
# A control plane serves one household; a few hundred recent turns is generous
# headroom while still bounding memory hard.
_DEFAULT_MAX_ENTRIES = 512
_MAX_KEY_LENGTH = 200

# Returned for a duplicate that arrives while the first send is still answering.
# ``service_unavailable`` is exactly the code the frontend treats as a
# connection dropped mid-answer (useResumedAnswer.DROPPED_CONNECTION_CODES), so
# both chat surfaces arm their resume-poll and surface the reply the first send
# will persist — the same reconciliation a real drop takes, not an error.
IN_FLIGHT_ERROR_CODE = "service_unavailable"
IN_FLIGHT_STATUS = 503
IN_FLIGHT_MESSAGE = (
    "This turn is already being answered from an earlier send. "
    "The reply appears here as soon as it lands — you do not need to ask again."
)


def in_flight_error() -> Dict[str, str]:
    """The error envelope body for an in-flight duplicate turn."""
    return {"code": IN_FLIGHT_ERROR_CODE, "message": IN_FLIGHT_MESSAGE}


ScopeKey = Tuple[str, str, str]


@dataclass(frozen=True)
class TurnClaim:
    """The outcome of claiming a turn key. Exactly one branch is actionable.

    * ``proceed`` is True — first sight of this key. Run the turn, then call
      :meth:`ChatTurnDedupe.complete` with its payload (or :meth:`abandon` if it
      failed before producing one).
    * ``replay`` is not None — the turn already finished. Return this stored
      payload verbatim; do not start a second inference.
    * ``in_flight`` is True — a duplicate arrived while the first is answering.
      Return the :func:`in_flight_error` response so the client reconciles by
      poll instead of launching a second inference.
    """

    proceed: bool
    in_flight: bool
    replay: Optional[Dict[str, Any]]


@dataclass
class _Record:
    # ``payload is None`` marks an in-flight claim; a dict marks a finished turn.
    payload: Optional[Dict[str, Any]]
    expires_at: float


def _key_token(key: Any) -> str:
    if key is None:
        return ""
    return str(key).strip()[:_MAX_KEY_LENGTH]


def _text(value: Any) -> str:
    return "" if value is None else str(value)


class ChatTurnDedupe:
    """Thread-safe, bounded record of accepted chat-turn idempotency keys."""

    def __init__(
        self,
        *,
        ttl_seconds: float = _DEFAULT_TTL_SECONDS,
        max_entries: int = _DEFAULT_MAX_ENTRIES,
        clock: Callable[[], float] = time.monotonic,
    ) -> None:
        self._ttl = float(ttl_seconds)
        self._max = max(1, int(max_entries))
        self._clock = clock
        self._lock = threading.Lock()
        self._entries: "OrderedDict[ScopeKey, _Record]" = OrderedDict()

    def claim(self, actor: Any, conversation_id: Any, key: Any) -> TurnClaim:
        """Register a turn key, or report that a duplicate was already seen.

        A falsy key means the caller opted out of dedupe (a legacy client that
        sent none): the turn always proceeds, exactly as before this store
        existed, so no keyless caller ever collides with another.
        """
        token = _key_token(key)
        if not token:
            return TurnClaim(proceed=True, in_flight=False, replay=None)
        scope = (_text(actor), _text(conversation_id), token)
        with self._lock:
            self._purge_expired()
            record = self._entries.get(scope)
            if record is not None:
                self._entries.move_to_end(scope)
                if record.payload is not None:
                    return TurnClaim(False, False, dict(record.payload))
                return TurnClaim(False, True, None)
            self._entries[scope] = _Record(
                payload=None, expires_at=self._clock() + self._ttl
            )
            self._entries.move_to_end(scope)
            self._evict_overflow()
            return TurnClaim(True, False, None)

    def complete(
        self, actor: Any, conversation_id: Any, key: Any, payload: Dict[str, Any]
    ) -> None:
        """Record the finished payload so later duplicates replay it."""
        token = _key_token(key)
        if not token:
            return
        scope = (_text(actor), _text(conversation_id), token)
        with self._lock:
            self._entries[scope] = _Record(
                payload=dict(payload), expires_at=self._clock() + self._ttl
            )
            self._entries.move_to_end(scope)
            self._evict_overflow()

    def abandon(self, actor: Any, conversation_id: Any, key: Any) -> None:
        """Drop a still-in-flight claim whose turn failed before completing.

        Only an in-flight record is removed; a completed one must survive so its
        replay stays available, and guarding on that keeps this failure-path
        call from racing a concurrent :meth:`complete`.
        """
        token = _key_token(key)
        if not token:
            return
        scope = (_text(actor), _text(conversation_id), token)
        with self._lock:
            record = self._entries.get(scope)
            if record is not None and record.payload is None:
                del self._entries[scope]

    def _purge_expired(self) -> None:
        now = self._clock()
        stale = [s for s, r in self._entries.items() if r.expires_at <= now]
        for scope in stale:
            del self._entries[scope]

    def _evict_overflow(self) -> None:
        while len(self._entries) > self._max:
            self._entries.popitem(last=False)
