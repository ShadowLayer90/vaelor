"""Fail-closed validator for per-node telemetry writes (Phase E2a).

The controller accepts telemetry from the network at
``POST /api/v2/telemetry/ingest`` (see ``api_telemetry_ingest_routes``). That
route authenticates the caller to a *node id* and then hands the raw request
body here. **Everything in this module treats the body as hostile**: it is
written by a worker over the LAN, and a worker must never be able to write into
another node's series, invent a new column in the shared ``history``
measurement, backfill or future-date a row, or crash the parser.

So this is a small, strict InfluxDB line-protocol reader, not a permissive one:

* **Measurement.** Only ``history`` is accepted; any other measurement refuses
  the whole call. There is one telemetry measurement and this is it.
* **Node tag.** The authenticated ``node_id`` is authoritative and is the tag
  the route stores, always. A line may carry a ``node`` tag, but only if it
  equals ``node_id``; a line trying to write ``node=controller`` (or any other
  node) from a worker is refused. That is the central threat this guards.
* **Field names.** A field key must match ``[a-z0-9_]{1,64}`` *and* be one of
  the telemetry keys the store already holds (``ACCEPTED_FIELDS``); an unknown
  key is dropped, not stored, so a worker cannot mint arbitrary series. A value
  must be a number or a short unquoted string; a non-finite float is dropped
  exactly as the store's ``_storable`` drops it.
* **Caps.** At most ``MAX_INGEST_LINES`` lines per call, and each timestamp must
  sit within ``INGEST_CLOCK_SKEW_SECONDS`` of now or the line is refused, so a
  worker can neither flood the store in one call nor stamp a row into the past
  or the future.
* **Clock offset.** Every read measures how far the worker's clock sits from
  the controller's, from the newest line timestamp, and hands it back - on an
  accepted batch (``IngestBatch.clock_offset_seconds``) and on the timestamp
  refusal alike (``TelemetryIngestError.clock_offset_seconds``) - so a worker
  whose clock has drifted is named rather than silently refused (ACC-126).

Structure is strict (a stray backslash, a control character, the wrong number
of space-separated sections, an unterminated string, a tag that is not ``node``
all refuse the call); *content* of a well-formed field is lenient (an unknown or
unparseable field is dropped rather than refusing the call), because a worker
running a slightly different Telegraf build should have its recognised metrics
recorded rather than its whole post thrown away.

This module imports no Flask and touches no store, so the whole threat model is
unit-testable in isolation.
"""

from __future__ import annotations

import math
import re
import time
from typing import Dict, List, NamedTuple, Optional, Tuple

from .telemetry_bounds import DISCARDED_COUNT_FIELD, DISCARDED_ON_MACHINE, MAX_DISCARDED_PER_LINE, plausible
from .telemetry_store import HISTORY_MEASUREMENT

#: The one measurement an ingest line may name. Reused from the store so the
#: name a worker may write and the name the reader reads cannot drift apart.
INGEST_MEASUREMENT = HISTORY_MEASUREMENT

#: The tag key that names which node a row belongs to. A single token, so it is
#: not a duplicated sentence; it is the one word both the writer and the reader
#: key the per-node series on.
NODE_TAG_KEY = "node"

#: The telemetry field names a worker may write into the ``history``
#: measurement: the machine-detail's influx fields (behind
#: ``api_telemetry_history_routes._SERIES_FIELDS``) and the readings listed
#: below them. An ingest line may write only
#: these; any other field key is DROPPED, so a worker cannot create arbitrary
#: columns or series in the shared ``history`` measurement. Kept as a closed
#: vocabulary rather than an open list: each name here has a reader. Most are
#: the metrics the machine-detail renders; the disk, network and fan fields
#: (VD-205 item 6) are read by the Assistant from a worker's latest row and are
#: not charted on any screen.
ACCEPTED_FIELDS = frozenset({
    "cpu_percent",
    "memory_percent",
    "gpu_busy_percent",
    "gpu_gtt_used_bytes",
    "gpu_gtt_total_bytes",
    "cpu_temperature",
    "gpu_temperature_c",
    # The GPU vendor readings a worker's sampler hands its emitter (VD-147):
    # graphics-engine power and temperature, package power, and - numbers, not
    # words - the status code and age of that reading (`gpu_vendor_status`).
    "gpu_power_watts",
    "gpu_gfx_temperature_c",
    "gpu_socket_power_watts",
    "gpu_vendor_status",
    "gpu_vendor_age_seconds",
    # A worker's disk, network and fans (VD-205 item 6), each absent when its
    # source is unreadable (`worker_telemetry_emitter`).
    "disk_root_total_bytes",
    "disk_root_used_bytes",
    "disk_root_free_bytes",
    "disk_data_total_bytes",
    "disk_data_used_bytes",
    "disk_data_free_bytes",
    "net_rx_bytes_per_second",
    "net_tx_bytes_per_second",
    "fan_rpm",
    "fan_count",
})

#: Most lines one call may carry. A worker samples one row per interval, so a
#: post is a handful of lines; this cap keeps a single authenticated call from
#: writing an unbounded batch, and is enforced before any line is parsed.
MAX_INGEST_LINES = 200

#: How far a line's own timestamp may sit from the controller's clock, in
#: seconds. A row further out is refused rather than stored, so a worker cannot
#: backfill history or future-date a row past the freshness the reader trusts.
INGEST_CLOCK_SKEW_SECONDS = 120

#: Why a post was refused for its timestamps, carried on the refusal so the
#: route records the right thing. ``CLOCK_REFUSAL_KIND``: the NEWEST line is
#: itself outside the window, so the worker's clock is wrong.
#: ``BACKLOG_REFUSAL_KIND``: the newest line is current but older lines are not -
#: an agent resending readings it buffered while the controller was away - which
#: says nothing about the clock and is healed by restarting the agent.
CLOCK_REFUSAL_KIND = "clock"
BACKLOG_REFUSAL_KIND = "backlog"

#: What a node id may look like: letters, digits, hyphen and underscore, 1-64
#: characters. The one home for the shape, imported by ``database`` so the read
#: path's query-injection guard and this write path agree on it.
NODE_ID_PATTERN = re.compile(r"[A-Za-z0-9_-]{1,64}")

_FIELD_KEY = re.compile(r"[a-z0-9_]{1,64}")
_INTEGER = re.compile(r"-?\d+[iu]")
#: A line-protocol float, restricted to a leading-digit form (`12`, `12.5`,
#: `1e3`) — the shape a telemetry agent emits. No `.5` branch, so the pattern
#: carries no alternation to hide a second matching rule behind.
_FLOAT = re.compile(r"[-+]?\d+\.?\d*([eE][-+]?\d+)?")


def valid_node_id(node: object) -> bool:
    """Whether ``node`` is a well-formed node id (letters/digits/-/_ , 1-64)."""
    return isinstance(node, str) and bool(NODE_ID_PATTERN.fullmatch(node))


class TelemetryIngestError(ValueError):
    """A telemetry ingest payload the validator refuses.

    One error type for every structural refusal, so the route has exactly one
    thing to catch and can map it to a single terse ``400`` without echoing the
    raw body back to the caller.

    ``clock_offset_seconds`` is set only on the timestamp-window refusal: the
    worker clock minus the controller clock, in seconds, measured from the
    newest line timestamp. Its presence is what tells the route that this post
    was refused because the worker's clock is wrong, not because it was
    malformed.
    """

    def __init__(
        self, reason: str, clock_offset_seconds: Optional[float] = None,
        refusal_kind: str = "",
    ) -> None:
        super().__init__(reason)
        self.clock_offset_seconds = clock_offset_seconds
        self.refusal_kind = refusal_kind


class Point(NamedTuple):
    """One validated row: its stored fields and an optional clamped ns time.

    ``time`` is ``None`` when the line carried no timestamp, in which case the
    store stamps the row with the server clock. The node tag is not carried
    here: the route stamps it from the authenticated ``node_id``, never from the
    payload.
    """

    fields: Dict[str, object]
    time: Optional[int]


class IngestBatch(NamedTuple):
    """A validated post: its points and the worker's measured clock offset.

    ``clock_offset_seconds`` is the worker clock minus the controller clock,
    from the newest line timestamp (positive: the worker is ahead), or ``None``
    when no line carried a timestamp. It includes the agent's flush delay, a
    second or so, which is noise beside the skew it exists to expose.
    """

    points: List[Point]
    clock_offset_seconds: Optional[float]
    #: The fields dropped because their value was physically implausible (or a
    #: status code the table does not hold). Re-checked here, against the same
    #: one table the emitter applied, as defence against a modified emitter.
    implausible: Tuple[str, ...] = ()


def _refuse(reason: str) -> "TelemetryIngestError":
    return TelemetryIngestError(reason)


def _parse_value(value: str) -> object:
    """One field value as a stored FLOAT, or ``_DROP`` to drop the field.

    Every accepted field is numeric, so a value is stored only if it parses to a
    finite float. A quoted string is checked for well-formedness (an
    unterminated or inner-quote string is a structural refusal that raises) and
    then DROPPED: storing a string into a field the controller writes as a float
    is an InfluxDB per-measurement type conflict that drops the whole row. A
    number that overflows the float range (a huge ``…i`` integer), is non-finite,
    or is unparseable is dropped too, and booleans are not accepted — so a stray
    field never fails the whole call and never changes a numeric field's stored
    type. Mirrors the store's ``_storable`` keeping one numeric type per field.
    """
    if value.startswith('"'):
        if len(value) < 2 or not value.endswith('"') or '"' in value[1:-1]:
            raise _refuse("a string field value is not terminated")
        return _DROP
    try:
        if _INTEGER.fullmatch(value):
            return float(int(value[:-1]))
        if _FLOAT.fullmatch(value):
            number = float(value)
            return number if math.isfinite(number) else _DROP
    except (ValueError, OverflowError):
        return _DROP
    return _DROP


#: Sentinel meaning "this field is not stored" — distinct from a real value so
#: a dropped field cannot be confused with a stored ``None`` or ``0``.
_DROP = object()


def _parse_fields(section: str, implausible: Optional[List[str]] = None) -> Dict[str, object]:
    """The field set of one line: recognised, allowed, plausible fields only.

    Field *structure* is strict — a token with no ``=`` or an empty key refuses
    the line — while field *content* is lenient: a key outside the allowlist or
    a value that will not parse is dropped, so a worker cannot mint columns yet
    a recognised metric beside an unrecognised one is still recorded. A value
    outside the physical-plausibility bounds is dropped too, never clamped, and
    its field name is appended to ``implausible``.
    """
    fields: Dict[str, object] = {}
    for token in section.split(","):
        key, sep, value = token.partition("=")
        if sep != "=" or not key or not value:
            raise _refuse("a field is not a key=value pair")
        if key == DISCARDED_COUNT_FIELD:
            # The emitter's own discards: counted, never stored (pass-3 review).
            reported = _parse_value(value)
            if implausible is not None and isinstance(reported, float) and 0 < reported <= MAX_DISCARDED_PER_LINE:
                implausible.extend([DISCARDED_ON_MACHINE] * int(reported))
            continue
        if _FIELD_KEY.fullmatch(key) is None or key not in ACCEPTED_FIELDS:
            continue
        parsed = _parse_value(value)
        if parsed is _DROP:
            continue
        if not plausible(key, parsed):
            if implausible is not None:
                implausible.append(key)
            continue
        fields[key] = parsed
    return fields


def _line_time(section: Optional[str]) -> Optional[int]:
    """A line's timestamp in ns, ``None`` if absent; refused if not an integer.

    The skew window is applied by `read_ingest` once every line is read, so
    the offset it reports is measured from the NEWEST line rather than from
    whichever line happened to trip the window first.
    """
    if section is None:
        return None
    if re.fullmatch(r"-?\d+", section) is None:
        raise _refuse("a timestamp is not an integer")
    return int(section)


def _parse_line(
    line: str, node_id: str, implausible: Optional[List[str]] = None,
) -> Tuple[Optional[Dict[str, object]], Optional[int]]:
    """One line to its stored fields (``None`` when none) and its ns time.

    Refuses the call (raises) on any structural problem: a backslash or control
    character, the wrong number of space-separated sections, a measurement that
    is not ``history``, a tag that is not the node tag, or a node tag naming a
    different node than the authenticated one.
    """
    if "\\" in line or any(ord(char) < 0x20 for char in line):
        raise _refuse("a line holds an escape or control character")
    parts = line.split(" ")
    if len(parts) not in (2, 3) or any(part == "" for part in parts):
        raise _refuse("a line is not measurement, fields and optional time")
    head, field_section = parts[0], parts[1]
    time_section = parts[2] if len(parts) == 3 else None

    head_parts = head.split(",")
    if head_parts[0] != INGEST_MEASUREMENT:
        raise _refuse("the measurement is not accepted")
    for tag in head_parts[1:]:
        key, sep, value = tag.partition("=")
        if sep != "=" or key != NODE_TAG_KEY:
            raise _refuse("a tag other than the node tag is not accepted")
        if value != node_id:
            raise _refuse("the node tag does not match the authenticated node")

    fields = _parse_fields(field_section, implausible)
    return (fields or None), _line_time(time_section)


def read_ingest(
    raw_text: object, node_id: str, now_ns: Optional[int] = None
) -> IngestBatch:
    """Validate a raw ingest body against ``node_id``, measuring its clock.

    Raises ``TelemetryIngestError`` for any structural or policy violation and
    returns the stored points otherwise (empty when every line's fields were
    unrecognised - nothing to store, but nothing wrong). The caller stamps
    ``node=node_id`` on every returned point; this never trusts the payload's
    own node tag beyond checking it does not name a different node.

    A stored line stamped more than ``INGEST_CLOCK_SKEW_SECONDS`` from the
    controller's clock refuses the whole post, as it always has; the refusal
    now carries the measured offset, so the route can record and explain it.
    """
    if not valid_node_id(node_id):
        raise _refuse("the authenticated node id is not well formed")
    if not isinstance(raw_text, str):
        raise _refuse("the ingest body is not text")
    lines = [line for line in raw_text.split("\n") if line.strip()]
    if len(lines) > MAX_INGEST_LINES:
        raise _refuse("the ingest body holds too many lines")
    now = time.time_ns() if now_ns is None else now_ns
    implausible: List[str] = []
    parsed = [_parse_line(line, node_id, implausible) for line in lines]
    stamps = [stamp for _fields, stamp in parsed if stamp is not None]
    offset = (
        round((max(stamps) - now) / 1_000_000_000, 3) if stamps else None
    )
    window_ns = INGEST_CLOCK_SKEW_SECONDS * 1_000_000_000
    # Only the NEWEST stamp can say the worker's clock is wrong. A post whose
    # newest line is current but whose older lines are out of window is a
    # buffered backlog (an agent resending what it held while the controller
    # was unreachable); calling that a clock fault told the owner to fix a
    # clock that was right and stopped the reconcile restarting the agent.
    newest_out = bool(stamps) and abs(max(stamps) - now) > window_ns
    points: List[Point] = []
    for fields, stamp in parsed:
        if fields is None:
            continue
        if stamp is not None and abs(stamp - now) > window_ns:
            if newest_out:
                raise TelemetryIngestError(
                    "a timestamp is outside the accepted window",
                    clock_offset_seconds=offset,
                    refusal_kind=CLOCK_REFUSAL_KIND,
                )
            raise TelemetryIngestError(
                "the post resends readings older than the accepted window",
                refusal_kind=BACKLOG_REFUSAL_KIND,
            )
        points.append(Point(fields=fields, time=stamp))
    return IngestBatch(
        points=points, clock_offset_seconds=offset, implausible=tuple(implausible),
    )


def parse_ingest(raw_text: object, node_id: str) -> List[Point]:
    """The stored points of `read_ingest`, for callers with no use for the clock."""
    return read_ingest(raw_text, node_id).points
