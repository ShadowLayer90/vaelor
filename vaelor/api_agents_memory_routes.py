"""Session-less bearer routes for a deployed agent's private memory (F5a).

``POST /api/v2/agents/memory/record`` and ``.../recall`` let a cluster agent
append to, and read back, its OWN recent-interaction memory. Like the telemetry
ingest surface these accept requests off the network from an agent process that
holds no browser session, so they are NOT ``require_auth``; each carries its own
bearer check against the per-agent memory-key hash on the deployment row,
compared in constant time, and the presented token is never logged or echoed.

The security spine is one sentence: **the agent identity is resolved from the
token and every store call is made under that resolved id - a record's own
contents never choose the partition.** A request that packs an ``agent_id`` into
its body writes into, and reads from, the agent the TOKEN binds to, exactly as
telemetry ingest tags the authenticated node and ignores the body's node tag.

The guards run cheap-first, mirroring the ingest route:

1. **Bearer -> agent.** A missing or malformed header, an unwired deployment
   store, or a token matching no deployment row all resolve to a uniform ``401``.
2. **Body-size cap**, from the declared length and again from the bytes read,
   before parsing: over :data:`MAX_MEMORY_BODY_BYTES` is ``413``.
3. **Per-agent, per-operation rate cap.** About one accepted call per agent per
   :data:`MEMORY_MIN_INTERVAL_SECONDS` for each of record and recall; a faster
   caller is ``429`` and a rejected call does not advance the budget.
4. **Store call under the resolved id.** A malformed request body is ``400``; a
   memory store that is unwired or cannot serve the call is ``503``.

Every branch degrades honestly: hostile input is never a ``500``, and a memory
subsystem that is down answers ``503`` rather than hanging the agent.
"""

from __future__ import annotations

import json
import logging
import threading
import time
from typing import Any, Dict, Optional, Tuple

from flask import request
from werkzeug.exceptions import RequestEntityTooLarge

from .api_common import ApiContext, payload as _payload
from .agent_memory import AgentMemoryError, DEFAULT_RECALL_LIMIT

LOGGER = logging.getLogger(__name__)

#: Largest memory request body the routes read, in bytes. A record call carries a
#: handful of short summaries and a recall call carries a small limit; 64 KiB is
#: generous for both and bounds a rejected call cheaply.
MAX_MEMORY_BODY_BYTES = 64 * 1024

#: The floor between accepted calls from one agent for one operation, in seconds.
#: An agent recalls once and records once per served completion, seconds apart,
#: so this sub-second floor never throttles ordinary cadence while a burst inside
#: it is turned away. In-memory and keyed per (operation, agent) on this process.
MEMORY_MIN_INTERVAL_SECONDS = 0.5

#: One home for the 413 wording, shared by the declared-length and bytes-read
#: checks so the two cannot drift apart on the message.
_MEMORY_BODY_TOO_LARGE = "The agent memory request body exceeds the accepted size."

#: One home each for the two sentences both handlers would otherwise repeat -
#: the 401 with no resolvable key, and the 503 with no store wired - so the
#: record and recall paths cannot drift apart on either wording.
_MISSING_MEMORY_KEY = "Present a valid agent memory key."
_MEMORY_NOT_WIRED = "Agent memory is not wired into this control plane."

_MEMORY_RATE_LOCK = threading.Lock()
_LAST_MEMORY_ACCEPTED: Dict[str, float] = {}


def _rate_key(operation: str, agent_id: str) -> str:
    """A budget key naming both the operation and the agent, kept independent."""
    return "{}:{}".format(operation, agent_id)


def _memory_rate_limited(operation: str, agent_id: str, now: float) -> bool:
    """Whether this agent ran ``operation`` too recently to accept another."""
    with _MEMORY_RATE_LOCK:
        last = _LAST_MEMORY_ACCEPTED.get(_rate_key(operation, agent_id))
        return last is not None and (now - last) < MEMORY_MIN_INTERVAL_SECONDS


def _mark_memory_accepted(operation: str, agent_id: str, now: float) -> None:
    with _MEMORY_RATE_LOCK:
        _LAST_MEMORY_ACCEPTED[_rate_key(operation, agent_id)] = now


def _error(code: str, message: str, status: int):
    return _payload(error={"code": code, "message": message}, status=status)


def _resolve_agent(callbacks: Dict[str, Any]) -> Optional[Tuple[str, str]]:
    """The ``(name, deployment_id)`` a presented memory token binds to, or ``None``.

    Fails closed: a missing or non-bearer header, an unwired deployment store, or
    a token matching no row all resolve to ``None`` for a uniform 401 at the call
    site. The token itself never reaches a log or an error message.
    """
    header = request.headers.get("Authorization", "")
    if not header.startswith("Bearer "):
        return None
    key = header[7:]
    if not key:
        return None
    store = callbacks.get("agent_deployments")
    if store is None:
        return None
    try:
        return store.agent_id_for_memory_key(key)
    except (AttributeError, OSError, TypeError, ValueError):
        return None


def _read_capped_body() -> Optional[bytes]:
    """The request body, or ``None`` when it exceeds the size cap.

    The declared length is checked first, then a hard ceiling on the read itself
    so a chunked body with no Content-Length cannot be buffered whole before the
    check, then the bytes read as belt-and-suspenders.
    """
    if request.content_length is not None and request.content_length > MAX_MEMORY_BODY_BYTES:
        return None
    request.max_content_length = MAX_MEMORY_BODY_BYTES
    try:
        raw = request.get_data(cache=False)
    except RequestEntityTooLarge:
        return None
    if len(raw) > MAX_MEMORY_BODY_BYTES:
        return None
    return raw


def _decode_object(raw: bytes) -> Optional[Dict[str, Any]]:
    """Parse the body as a JSON object, or ``None`` when it is not one.

    An empty body decodes to an empty object, so a recall call may carry no body
    and still take the default limit.
    """
    if not raw:
        return {}
    try:
        parsed = json.loads(raw.decode("utf-8"))
    except (UnicodeDecodeError, ValueError):
        return None
    if not isinstance(parsed, dict):
        return None
    return parsed


def register_agents_memory_routes(context: ApiContext) -> None:
    blueprint = context.blueprint
    callbacks = context.callbacks

    @blueprint.post("/agents/memory/record")
    def agents_memory_record():
        identity = _resolve_agent(callbacks)
        if identity is None:
            return _error(
                "authentication_required",
                _MISSING_MEMORY_KEY,
                401,
            )
        name, agent_id = identity
        raw = _read_capped_body()
        if raw is None:
            return _error("payload_too_large", _MEMORY_BODY_TOO_LARGE, 413)
        now = time.time()
        if _memory_rate_limited("record", agent_id, now):
            return _error(
                "rate_limited",
                "This agent is writing memory faster than accepted.",
                429,
            )
        body = _decode_object(raw)
        if body is None:
            return _error(
                "invalid_payload", "The record body must be a JSON object.", 400
            )
        store = callbacks.get("agent_memory")
        if store is None:
            LOGGER.warning(_MEMORY_NOT_WIRED)
            return _error(
                "memory_unavailable",
                _MEMORY_NOT_WIRED,
                503,
            )
        try:
            stored = store.record(agent_id, body.get("records"))
        except AgentMemoryError as error:
            return _error("invalid_payload", str(error), 400)
        except Exception as error:  # never a 500 back to the network: report unavailable
            # Logged at WARNING with the agent's name and the store's reason - an
            # agent whose memory silently fails is the defect this surface had.
            LOGGER.warning("Agent memory write for %s failed: %s", name, error)
            return _error(
                "memory_unavailable",
                "The agent memory store could not accept the write.",
                503,
            )
        _mark_memory_accepted("record", agent_id, now)
        return _payload({"agent": name, "stored": int(stored)})

    @blueprint.post("/agents/memory/recall")
    def agents_memory_recall():
        identity = _resolve_agent(callbacks)
        if identity is None:
            return _error(
                "authentication_required",
                _MISSING_MEMORY_KEY,
                401,
            )
        name, agent_id = identity
        raw = _read_capped_body()
        if raw is None:
            return _error("payload_too_large", _MEMORY_BODY_TOO_LARGE, 413)
        now = time.time()
        if _memory_rate_limited("recall", agent_id, now):
            return _error(
                "rate_limited",
                "This agent is reading memory faster than accepted.",
                429,
            )
        body = _decode_object(raw)
        if body is None:
            return _error(
                "invalid_payload", "The recall body must be a JSON object.", 400
            )
        store = callbacks.get("agent_memory")
        if store is None:
            LOGGER.warning(_MEMORY_NOT_WIRED)
            return _error(
                "memory_unavailable",
                _MEMORY_NOT_WIRED,
                503,
            )
        try:
            records = store.recall(agent_id, body.get("limit", DEFAULT_RECALL_LIMIT))
        except AgentMemoryError as error:
            return _error("invalid_payload", str(error), 400)
        except Exception as error:  # never a 500 back to the network: report unavailable
            LOGGER.warning("Agent memory recall for %s failed: %s", name, error)
            return _error(
                "memory_unavailable",
                "The agent memory store could not serve the recall.",
                503,
            )
        _mark_memory_accepted("recall", agent_id, now)
        return _payload({"agent": name, "records": list(records)})
