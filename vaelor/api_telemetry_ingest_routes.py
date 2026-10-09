"""Keyed HTTP ingest for per-node telemetry (Phase E2a).

``POST /api/v2/telemetry/ingest`` is the ONE new network surface for per-node
telemetry: an enrolled worker posts InfluxDB line protocol here and the
controller stores it, tagged with the worker's node id. **It accepts writes from
the network, so it is attacker-adjacent and every guard here is deliberate.**

It is NOT ``require_auth``: that authenticates browser *users* by session
cookie, and a worker agent has no session. So this carries its own bearer check
against the ingest-key hash stored on the node record, comparing in constant
time. The order of the guards is the point — the cheap ones run first:

1. **Bearer → node.** A missing or malformed ``Authorization`` header, or a key
   that resolves to no node, is ``401``. The key never appears in a log or an
   error.
2. **Body-size cap**, from ``Content-Length`` and again from the bytes actually
   read, before any parsing: over ``MAX_BODY_BYTES`` is ``413``.
3. **Per-node rate cap.** About one accepted post per node per
   ``INGEST_MIN_INTERVAL_SECONDS`` (an in-memory last-accepted time). The floor
   sits a jitter tolerance below the interval (``INGEST_ACCEPT_FLOOR_SECONDS``)
   so an agent flushing *at* the interval is not throttled by sub-second jitter;
   a much faster caller is ``429``. Rejected posts do not advance the budget, and
   are bounded by the body cap, so a flood is cheap.
4. **Validation** via ``telemetry_ingest.read_ingest``, which is fail-closed
   and enforces measurement, node-tag and field policy. A refusal is a terse
   ``400`` that never echoes the raw body.
5. **Write**, tagged with the AUTHENTICATED node id — never the payload's — via
   the ``telemetry_ingest_write`` callback. A store that cannot take the write
   is ``503``.

Every accepted post, and every post refused because its timestamps sat outside
the accepted window, is recorded in ``telemetry_ingest_status`` with the
worker's measured clock offset (ACC-126). The refusal stays a ``400`` - nothing
is backfilled or future-dated - but it is no longer silent: the Fleet screen
reads why, the journal gets a rate-limited warning naming the node and the
offset, and the self-heal reconcile can tell a wrong clock from a dead agent.

The route never returns ``500`` for hostile input: a decode or validation
failure is ``400``, an auth failure ``401``, an oversize body ``413``, a too-fast
caller ``429``, and a store problem ``503``.
"""

from __future__ import annotations

import threading
import time
from typing import Any, Dict, Optional

from flask import request
from werkzeug.exceptions import RequestEntityTooLarge

from .api_common import ApiContext, payload as _payload
from .telemetry_ingest import (
    BACKLOG_REFUSAL_KIND,
    CLOCK_REFUSAL_KIND,
    IngestBatch,
    TelemetryIngestError,
    read_ingest,
)
from .telemetry_ingest_status import (
    record_accepted,
    record_backlog_refused,
    record_implausible,
    record_clock_refused,
)
from .telemetry_store import DEFAULT_DATA_INTERVAL, TelemetryStoreError

#: Largest ingest body the route reads, in bytes. A worker's post is a handful
#: of short lines; 64 KiB is generous for that and small enough that a rejected
#: post is cheap to bound.
MAX_BODY_BYTES = 64 * 1024

#: The nominal gap between accepted posts from one node, in seconds: a worker
#: samples one row per interval, so the store's sampling interval is the natural
#: bound and a much faster caller is throttled with a 429. In-memory and
#: per-node, so the bound is "about one stored post per node per interval on this
#: process".
INGEST_MIN_INTERVAL_SECONDS = DEFAULT_DATA_INTERVAL

#: Jitter headroom below the interval. The Telegraf agent flushes every
#: ``INGEST_MIN_INTERVAL_SECONDS``, but its posts land with sub-second jitter
#: (measured ~0.98 s apart) and it re-sends a rejected metric on its next flush,
#: so a floor set at *exactly* the interval rejects roughly every other
#: on-cadence post — a ~0.98 s cadence alternates accept/reject, halving the rate
#: and filling the agent's journal with 429s. Accepting a post once it is within
#: this tolerance of the interval lets an on-cadence agent through untouched,
#: while a genuine flood (many posts within a fraction of a second) is still
#: rejected and the worst case stays about two stored rows per node per second.
INGEST_INTERVAL_TOLERANCE_SECONDS = DEFAULT_DATA_INTERVAL / 2

#: The effective floor: a post is throttled only when it arrives sooner than this
#: after the node's last accepted post.
INGEST_ACCEPT_FLOOR_SECONDS = (
    INGEST_MIN_INTERVAL_SECONDS - INGEST_INTERVAL_TOLERANCE_SECONDS
)

#: One home for the 413 reason, used by both the declared-length and the
#: bytes-read caps, so the two checks cannot drift on the wording.
_BODY_TOO_LARGE = "The ingest body is too large."

_RATE_LOCK = threading.Lock()
_LAST_ACCEPTED: Dict[str, float] = {}


def _rate_limited(node_id: str, now: float) -> bool:
    """Whether ``node_id`` posted an accepted body too recently to accept another."""
    with _RATE_LOCK:
        last = _LAST_ACCEPTED.get(node_id)
        return last is not None and (now - last) < INGEST_ACCEPT_FLOOR_SECONDS


def _mark_accepted(node_id: str, now: float) -> None:
    with _RATE_LOCK:
        _LAST_ACCEPTED[node_id] = now


def _error(code: str, message: str, status: int):
    return _payload(error={"code": code, "message": message}, status=status)


def _resolve_node(callbacks: Dict[str, Any]) -> Optional[str]:
    """The node id the presented bearer key resolves to, or ``None``.

    Fails closed: a missing/malformed header, an unwired cluster manager, or a
    key matching no node record all resolve to ``None`` (a uniform 401 at the
    call site), and the key itself is never logged or echoed.
    """
    header = request.headers.get("Authorization", "")
    if not header.startswith("Bearer "):
        return None
    key = header[7:]
    if not key:
        return None
    manager = callbacks.get("cluster_manager")
    if manager is None:
        return None
    try:
        return manager.node_for_ingest_key(key)
    except (AttributeError, OSError, TypeError, ValueError):
        return None


def register_telemetry_ingest_routes(context: ApiContext) -> None:
    blueprint = context.blueprint
    callbacks = context.callbacks

    @blueprint.post("/telemetry/ingest")
    def telemetry_ingest():
        node_id = _resolve_node(callbacks)
        if node_id is None:
            return _error(
                "authentication_required",
                "Present a valid node ingest key.",
                401,
            )
        # Body-size cap before reading or parsing: the declared length first,
        # then a hard per-request ceiling on the READ itself so a chunked body
        # with no Content-Length cannot be buffered whole before the check (the
        # app sets no global MAX_CONTENT_LENGTH; the sibling upload route caps
        # per-request the same way). `get_data` raises once the ceiling is
        # crossed; the post-read length check stays as belt-and-suspenders.
        if request.content_length is not None and request.content_length > MAX_BODY_BYTES:
            return _error("payload_too_large", _BODY_TOO_LARGE, 413)
        request.max_content_length = MAX_BODY_BYTES
        try:
            raw = request.get_data(cache=False)
        except RequestEntityTooLarge:
            return _error("payload_too_large", _BODY_TOO_LARGE, 413)
        if len(raw) > MAX_BODY_BYTES:
            return _error("payload_too_large", _BODY_TOO_LARGE, 413)
        now = time.time()
        if _rate_limited(node_id, now):
            return _error(
                "rate_limited",
                "This node is posting telemetry faster than accepted.",
                429,
            )
        try:
            text = raw.decode("utf-8")
        except UnicodeDecodeError:
            return _error("invalid_payload", "The ingest body is not UTF-8 text.", 400)
        try:
            batch: IngestBatch = read_ingest(text, node_id)
        except TelemetryIngestError as error:
            if error.refusal_kind == CLOCK_REFUSAL_KIND:
                # Refused for the worker's clock, not for its shape: remember
                # it (and warn, rate-limited) so the screen and the self-heal
                # can say so instead of seeing a silent worker.
                record_clock_refused(node_id, error.clock_offset_seconds, now)
            elif error.refusal_kind == BACKLOG_REFUSAL_KIND:
                # A buffered backlog, not a clock: recorded as such so the
                # self-heal restarts the agent instead of blaming its clock.
                record_backlog_refused(node_id, now)
            # Terse and derived from the validator's reason; the raw body is
            # never reflected back to the caller.
            return _error("invalid_payload", str(error), 400)
        points = batch.points
        writer = callbacks.get("telemetry_ingest_write")
        if writer is None:
            return _error(
                "telemetry_unavailable",
                "Telemetry ingest is not wired into this control plane.",
                503,
            )
        try:
            written = writer(node_id, points)
        except TelemetryStoreError as error:
            return _error("telemetry_unavailable", str(error), 503)
        except Exception:  # never a 500 back to the network: report unavailable
            return _error(
                "telemetry_unavailable",
                "The telemetry store could not accept the write.",
                503,
            )
        _mark_accepted(node_id, now)
        record_accepted(node_id, batch.clock_offset_seconds, now)
        record_implausible(node_id, batch.implausible, now)
        return _payload({"node": node_id, "accepted_points": int(written)})
