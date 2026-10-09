"""The Performance dashboard's own read path into InfluxDB (VD-147 S4, spec §2.3).

`database.Database` holds ONE InfluxDB client behind ONE lock for every write
and every read. A multi-day dashboard read through it would hold the lock the
data logger's once-a-second write and every worker's ingest write wait on - and
a stalled write is a machine that looks "not reporting". So the dashboard reads
through this module's own client and its own lock, and nothing else uses it.

Three reads, each bounded:

* :meth:`TelemetryReader.latest_per_node` - the now layer's one query: every
  node's newest row in the last five minutes, ``GROUP BY "node"`` with no node
  predicate (the measurement holds only the controller's rows and rows the
  keyed ingest stamped with an authenticated node, so "every node in it" is the
  fleet, removed nodes included);
* :meth:`TelemetryReader.host_buckets` - host series for a range, from the raw
  ``history`` below :data:`ROLLUP_FROM_SECONDS` and from the one-minute
  continuous query's ``history_1m`` at and above it (`telemetry_reader_rollup`)
  - but only once the roll-up is created and backfilled; until then the raw
  rows answer every range (pass-3 review S4-B2);
* :meth:`TelemetryReader.latest_for` - one node's newest row with no time bound,
  asked only when the bounded read found nothing for that node.

A tag value that is not a valid node id is dropped and logged, never rendered;
the untagged legacy series (``node = ''``) is the controller's. A filtered read
goes through :func:`nodes_clause`, which validates every id and raises on an
empty list rather than widening the read.

Every call to the store - the roll-up's creation and backfill included - goes
through :meth:`TelemetryReader.locked`, one lock around one client with a
short timeout and few retries (pass-3 review): the client's HTTP session is
never shared across threads, and a stalled store costs a request seconds, not
the influxdb client's default of waiting forever three times.
"""

from __future__ import annotations

import logging
import threading
from typing import Any, Callable, Dict, Iterable, List, Optional, Tuple

from .cluster_placement import CONTROLLER_PLACEMENT_ID
from .telemetry_ingest import valid_node_id
from .telemetry_store import HISTORY_MEASUREMENT

LOGGER = logging.getLogger(__name__)

#: The now layer's bound: a node with no row in this long is not reporting.
LATEST_WINDOW_SECONDS = 300

#: Host ranges at least this long read the one-minute roll-up rather than the
#: raw one-second rows (spec §2.3: about 20 thousand rows over two nodes at 7 d
#: instead of about 1.2 million).
ROLLUP_FROM_SECONDS = 6 * 3600

#: The roll-up measurement the continuous query writes.
ROLLUP_MEASUREMENT = "history_1m"

#: Host fields averaged over a bucket, and host fields kept as the bucket's
#: highest reading. The roll-up stores ``<field>_sum`` and ``<field>_n`` for the
#: first and ``<field>_max`` for the second, so a dashboard bucket over several
#: minutes is Σsum ÷ Σn - a partial minute weighted by what it holds.
MEAN_FIELDS = (
    "cpu_percent", "memory_percent", "gpu_busy_percent", "gpu_power_watts",
    "gpu_socket_power_watts", "cpu_package_power_watts", "gpu_gtt_used_bytes",
)
MAX_FIELDS = ("gpu_gfx_temperature_c", "gpu_temperature_c", "gpu_gtt_total_bytes")

#: The field whose sample count stands for a node's host coverage in a bucket.
#: Each averaged field also carries its own count as ``<field>_n``, so a field
#: read less often than the CPU (GPU power every few seconds) is judged on its
#: own coverage (pass-3 review).
COVERAGE_FIELD = "cpu_percent"
SAMPLES_KEY = "samples"
COUNT_SUFFIX = "_n"

#: Seconds one store call may take, and how often the client retries it.
READ_TIMEOUT_SECONDS = 5
READ_RETRIES = 2
#: How long a call waits for the reader's lock before giving up (pass-4
#: review): a hung store must cost a request seconds, not a queue behind it.
LOCK_WAIT_SECONDS = 3.0
#: Why a read gave up waiting for an earlier one.
READER_BUSY = "The telemetry store is still answering an earlier dashboard read; the next refresh tries again."


class TelemetryReaderError(RuntimeError):
    """A dashboard read that could not be made; the message is the owner's sentence."""


def nodes_clause(nodes: Iterable[str], keyword: str) -> str:
    """``"node" = 'a' OR "node" = 'b'``, parenthesised and joined by ``keyword``.

    Every id is validated before it is interpolated; the controller expands to
    its tagged-or-untagged pair. An empty list raises: a filter that filtered
    nothing would silently read every node.
    """
    parts: List[str] = []
    for node in nodes:
        if not valid_node_id(node):
            raise ValueError("A dashboard node filter takes only well-formed node ids.")
        if node == CONTROLLER_PLACEMENT_ID:
            parts.append("\"node\" = '{}' OR \"node\" = ''".format(node))
        else:
            parts.append("\"node\" = '{}'".format(node))
    if not parts:
        raise ValueError("A node filter needs at least one node.")
    return "{}({})".format(keyword, " OR ".join(parts))


def _default_client_factory() -> Any:
    from influxdb import InfluxDBClient

    return InfluxDBClient(host="localhost", port=8086, timeout=READ_TIMEOUT_SECONDS, retries=READ_RETRIES)


class TelemetryReader:
    """Dashboard reads through a client and a lock of their own."""

    def __init__(
        self, database_name: str,
        client_factory: Callable[[], Any] = _default_client_factory,
        log: Optional[logging.Logger] = None,
        lock_wait_seconds: float = LOCK_WAIT_SECONDS,
    ) -> None:
        self.database_name = database_name
        self._client_factory = client_factory
        self._client: Any = None
        self._lock = threading.Lock()
        self._lock_wait = float(lock_wait_seconds)
        self._log = log or LOGGER

    def locked(self) -> "_LockedClient":
        """This reader's client behind its lock: the only way anything reaches the store."""
        return _LockedClient(self)

    def _call(self, method: str, *args: Any, **options: Any) -> Any:
        """One client call under the lock; a failure drops the client so the next call reconnects."""
        if not self._lock.acquire(timeout=self._lock_wait):
            raise TelemetryReaderError(READER_BUSY)
        try:
            if self._client is None:
                self._client = self._client_factory()
            return getattr(self._client, method)(*args, **options)
        except Exception:
            self._client = None
            raise
        finally:
            self._lock.release()

    def _query(self, statement: str, **options: Any) -> List[Dict[str, Any]]:
        """Run one statement; the raw series list, or :class:`TelemetryReaderError`."""
        try:
            result = self._call("query", statement, database=self.database_name, **options)
        except TelemetryReaderError:
            raise
        except Exception as error:  # noqa: BLE001 - one sentence for every way a read fails
            raise TelemetryReaderError(
                "The telemetry store could not be read for the dashboard: {}".format(
                    " ".join(str(error).split())[:200])
            ) from error
        raw = getattr(result, "raw", result)
        series = raw.get("series") if isinstance(raw, dict) else None
        return list(series or [])

    def _node_of(self, series: Dict[str, Any]) -> Optional[str]:
        """The node a series belongs to, or ``None`` for a tag that is not a node id."""
        tag = str((series.get("tags") or {}).get("node") or "")
        if tag == "":
            return CONTROLLER_PLACEMENT_ID
        if not valid_node_id(tag):
            self._log.warning("Dropped telemetry rows tagged with a malformed node id.")
            return None
        return tag

    @staticmethod
    def _rows(series: Dict[str, Any]) -> List[Dict[str, Any]]:
        columns = list(series.get("columns") or [])
        return [dict(zip(columns, values)) for values in series.get("values") or []]

    def latest_per_node(self, window_seconds: int = LATEST_WINDOW_SECONDS) -> Dict[str, Dict[str, Any]]:
        """Every node's newest row in the window, ``time`` in epoch seconds.

        Where the controller has both a tagged and an untagged (legacy) series,
        the newer row wins.
        """
        statement = (
            'SELECT * FROM "{}" WHERE time > now() - {}s GROUP BY "node" ORDER BY time DESC LIMIT 1'
        ).format(HISTORY_MEASUREMENT, max(1, int(window_seconds)))
        found: Dict[str, Dict[str, Any]] = {}
        for series in self._query(statement, epoch="s"):
            node = self._node_of(series)
            rows = self._rows(series)
            if node is None or not rows:
                continue
            row = rows[0]
            if node not in found or (row.get("time") or 0) > (found[node].get("time") or 0):
                found[node] = row
        return found

    def latest_for(self, node: str) -> Optional[Dict[str, Any]]:
        """One node's newest row with no time bound - asked only when the bounded read had none."""
        statement = 'SELECT * FROM "{}"{} ORDER BY time DESC LIMIT 1'.format(
            HISTORY_MEASUREMENT, nodes_clause([node], " WHERE "),
        )
        series = self._query(statement, epoch="s")
        rows = self._rows(series[0]) if series else []
        return rows[0] if rows else None

    def host_buckets(
        self, start: int, end: int, step: int, rollup_ready: bool = False,
    ) -> Tuple[Dict[str, Dict[int, Dict[str, Any]]], str]:
        """``({node: {bucket start: row}}, source)`` for a range; ``source`` is ``raw`` or ``rollup``.

        Each row carries the mean fields, the max fields, each averaged field's
        own count and ``samples`` (the CPU count); a bucket with no sample is
        absent from the node's map, never zero. The roll-up is read only when
        the caller says it is ready: a missing or half-backfilled roll-up would
        draw every machine as absent.
        """
        rollup = rollup_ready and (end - start) >= ROLLUP_FROM_SECONDS
        statement = rollup_select(start, end, step) if rollup else raw_select(start, end, step)
        buckets: Dict[str, Dict[int, Dict[str, Any]]] = {}
        for series in self._query(statement, epoch="s"):
            node = self._node_of(series)
            if node is None:
                continue
            target = buckets.setdefault(node, {})
            for row in self._rows(series):
                shaped = _from_rollup(row) if rollup else _from_raw(row)
                if shaped is None:
                    continue
                moment = int(row.get("time") or 0)
                if moment in target:
                    target[moment] = _merge(target[moment], shaped)
                else:
                    target[moment] = shaped
        return buckets, "rollup" if rollup else "raw"


#: One bucketed read: the columns, the measurement, the span and the step.
_BUCKETED_SELECT = (
    'SELECT {} FROM "{}" WHERE time >= {}s AND time < {}s GROUP BY time({}s), "node" fill(null)'
)


def raw_select(start: int, end: int, step: int) -> str:
    """The raw-rows bucketed read (ranges below :data:`ROLLUP_FROM_SECONDS`)."""
    columns = ['mean("{0}") AS "{0}"'.format(field) for field in MEAN_FIELDS]
    columns += ['count("{0}") AS "{0}{1}"'.format(field, COUNT_SUFFIX) for field in MEAN_FIELDS]
    columns += ['max("{0}") AS "{0}"'.format(field) for field in MAX_FIELDS]
    columns.append('count("{}") AS "{}"'.format(COVERAGE_FIELD, SAMPLES_KEY))
    return _BUCKETED_SELECT.format(", ".join(columns), HISTORY_MEASUREMENT, int(start), int(end), int(step))


def rollup_select(start: int, end: int, step: int) -> str:
    """The roll-up bucketed read (ranges at and above :data:`ROLLUP_FROM_SECONDS`)."""
    columns = []
    for field in MEAN_FIELDS:
        columns.append('sum("{0}_sum") AS "{0}_sum"'.format(field))
        columns.append('sum("{0}_n") AS "{0}_n"'.format(field))
    columns += ['max("{0}_max") AS "{0}"'.format(field) for field in MAX_FIELDS]
    return _BUCKETED_SELECT.format(", ".join(columns), ROLLUP_MEASUREMENT, int(start), int(end), int(step))


def _number(value: Any) -> Optional[float]:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        return None
    try:
        number = float(value)
    except (OverflowError, ValueError):
        return None
    return number if number == number and abs(number) != float("inf") else None


def _from_raw(row: Dict[str, Any]) -> Optional[Dict[str, Any]]:
    shaped = {field: _number(row.get(field)) for field in MEAN_FIELDS + MAX_FIELDS}
    if all(value is None for value in shaped.values()) and not _number(row.get(SAMPLES_KEY)):
        return None
    for field in MEAN_FIELDS:
        shaped[field + COUNT_SUFFIX] = _number(row.get(field + COUNT_SUFFIX)) or 0.0
    shaped[SAMPLES_KEY] = _number(row.get(SAMPLES_KEY)) or 0.0
    return shaped


def _from_rollup(row: Dict[str, Any]) -> Optional[Dict[str, Any]]:
    shaped: Dict[str, Any] = {}
    for field in MEAN_FIELDS:
        total, count = _number(row.get(field + "_sum")), _number(row.get(field + "_n"))
        shaped[field] = total / count if total is not None and count else None
        shaped[field + COUNT_SUFFIX] = count or 0.0
    for field in MAX_FIELDS:
        shaped[field] = _number(row.get(field))
    shaped[SAMPLES_KEY] = shaped[COVERAGE_FIELD + COUNT_SUFFIX]
    if not shaped[SAMPLES_KEY] and all(shaped.get(field) is None for field in MEAN_FIELDS + MAX_FIELDS):
        return None
    return shaped


def _merge(first: Dict[str, Any], second: Dict[str, Any]) -> Dict[str, Any]:
    """The controller's tagged and legacy untagged series in one bucket, weighted by samples."""
    merged: Dict[str, Any] = {}
    for field in MEAN_FIELDS:
        counted = field + COUNT_SUFFIX
        weight_a, weight_b = first.get(counted) or 0.0, second.get(counted) or 0.0
        merged[counted] = weight_a + weight_b
        a, b = first.get(field), second.get(field)
        if a is None or b is None or weight_a + weight_b <= 0:
            merged[field] = a if b is None else b
        else:
            merged[field] = (a * weight_a + b * weight_b) / (weight_a + weight_b)
    for field in MAX_FIELDS:
        values = [value for value in (first.get(field), second.get(field)) if value is not None]
        merged[field] = max(values) if values else None
    merged[SAMPLES_KEY] = (first.get(SAMPLES_KEY) or 0.0) + (second.get(SAMPLES_KEY) or 0.0)
    return merged


class _LockedClient:
    """The reader's client as the roll-up sees it: every call under the reader's lock."""

    def __init__(self, reader: TelemetryReader) -> None:
        self._reader = reader

    def query(self, statement: str, **options: Any) -> Any:
        return self._reader._call("query", statement, **options)

    def write_points(self, points: Any, **options: Any) -> Any:
        return self._reader._call("write_points", points, **options)
