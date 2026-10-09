"""Per-replica serving buckets: what each machine's engine did, twenty seconds at a time.

VD-147 (slice S3). The Performance dashboard draws serving figures per machine
over time. They are kept here, in the usage ledger's own database
(:mod:`vaelor.model_usage`) and written inside the ledger's own transaction,
because the ledger already holds each replica's persisted counter baselines
and already turns two readings into an increase. A second counter-diff
mechanism beside it would be two owners of "how much did this replica do"
(LESSONS 6).

Every function here takes the ledger's open connection and never commits: the
ledger writes usage totals, the usage bucket and these rows in ONE transaction.

**Usage and rates follow different rules, on purpose.** Usage wants every
token counted, so after a restart or a gap the ledger adds the whole new
reading (Prometheus's ``increase``). A RATE needs to know how long the increase
took, and after a restart or a gap that is not known. So a tick is one of:

* **rateable** - the increase spans a time inside the bound: its increases go
  into this table's columns and the measured span into ``covered_seconds``;
* **a reset** - the engine restarted: ``reset_ticks`` goes up, nothing else;
* **a gap** - the span is past the bound, or the wall clock stepped back:
  ``gap_ticks`` goes up, nothing else;
* **a first read** - only a baseline.

**The span is this table's own** (review B1): the time since the performance
baseline these counters are diffed against was read - never the usage
baseline's, which a usage-only reading can move in between. When both readings
carry the poller's stamp from the same run (`serving_metrics_poller.TickClock`),
the span is measured on the poller's monotonic clock, so a wall-clock step
cannot fake a spike or a dip (review S-7), and "one tick apart" is judged by
tick number, so a slow but healthy tick is not a gap (review S-8)
(`serving_buckets.stamped_span`). Otherwise it is wall-clock time, judged by
`serving_buckets.span_is_rateable`.

In every case ``attempted_seconds`` grows by the tick's period, which is what
makes coverage (read time over attempted time) measurable, and the gauges that
are readings of "now" (KV cache, running, waiting) are recorded, because those
do not depend on a span. A tick in which the replica could not be read at all
(:func:`record_miss`) adds attempted seconds only.

**Speed bands** are stored as per-bin increases of the engine's own histogram,
only where non-zero, against a numbered edge set kept once in a side table.
Rows of different edge sets are never pooled (`serving_buckets.merge_bins`).
"""

from __future__ import annotations

import json
import sqlite3
from typing import Any, Dict, List, Mapping, Optional, Sequence, Tuple

from .serving_buckets import bin_increases, span_is_rateable, stamped_span
from .serving_metrics import FAMILY_PER_REQUEST, SERVING_POLL_SECONDS
from .usage_rollup import performance_bucket_start, performance_prune_before

#: How a deployment's copies are laid out, as a row records it: one full copy
#: per machine, one model spanning machines, or this machine's own engine.
#: ``replicated`` and ``distributed`` are the record's own words
#: (`cluster_gpu_sizing`); ``single`` is single-machine serving.
PLACEMENT_SINGLE = "single"

#: The cumulative counters a tick's performance reading may carry, each mapped
#: to the column its increase is added to.
COUNTER_COLUMNS = (
    "itl_seconds", "itl_intervals", "prefix_hits", "prefix_queries",
    "preemptions", "prompt_computed", "prompt_cached", "prompt_seconds",
)
#: The usage counters, whose increase the ledger has already worked out.
_USAGE_COLUMNS = ("prompt_tokens", "completion_tokens", "requests")
#: The gauges, each kept as the highest reading in the bucket.
_GAUGE_COLUMNS = (("kv_max", "kv_cache_fraction"), ("running_max", "requests_processing"),
                  ("waiting_max", "requests_deferred"))

SCHEMA = """
CREATE TABLE IF NOT EXISTS model_usage_replica_buckets (
    identity TEXT NOT NULL,
    replica  TEXT NOT NULL,
    bucket   INTEGER NOT NULL,
    node_id  TEXT NOT NULL,
    node_name TEXT NOT NULL DEFAULT '',
    model TEXT NOT NULL DEFAULT '', engine TEXT NOT NULL DEFAULT '',
    placement TEXT NOT NULL DEFAULT '',
    attempted_seconds REAL NOT NULL DEFAULT 0,
    covered_seconds   REAL NOT NULL DEFAULT 0,
    reset_ticks INTEGER NOT NULL DEFAULT 0,
    gap_ticks INTEGER NOT NULL DEFAULT 0,
    models_seen INTEGER NOT NULL DEFAULT 1,
    band_includes_short_replies INTEGER NOT NULL DEFAULT 0,
    prompt_tokens REAL, completion_tokens REAL, requests REAL,
    itl_seconds REAL, itl_intervals REAL,
    prefix_hits REAL, prefix_queries REAL, preemptions REAL,
    prompt_computed REAL, prompt_cached REAL, prompt_seconds REAL,
    kv_max REAL, running_max REAL, waiting_max REAL,
    PRIMARY KEY (identity, replica, bucket)
);
CREATE INDEX IF NOT EXISTS model_usage_replica_buckets_by_time
    ON model_usage_replica_buckets(bucket);
CREATE TABLE IF NOT EXISTS model_usage_edge_sets (
    edge_set_id INTEGER PRIMARY KEY,
    family INTEGER NOT NULL,
    fingerprint TEXT NOT NULL UNIQUE,
    edges TEXT NOT NULL
);
CREATE TABLE IF NOT EXISTS model_usage_replica_histogram (
    identity TEXT NOT NULL, replica TEXT NOT NULL, bucket INTEGER NOT NULL,
    edge_set_id INTEGER NOT NULL,
    edge_index INTEGER NOT NULL,
    count INTEGER NOT NULL,
    PRIMARY KEY (identity, replica, bucket, edge_set_id, edge_index)
) WITHOUT ROWID;
CREATE INDEX IF NOT EXISTS model_usage_replica_histogram_by_time
    ON model_usage_replica_histogram(bucket);
CREATE TABLE IF NOT EXISTS model_usage_replica_transitions (
    replica TEXT NOT NULL, at REAL NOT NULL, kind TEXT NOT NULL,
    identity TEXT NOT NULL DEFAULT '', node_id TEXT NOT NULL DEFAULT '',
    node_name TEXT NOT NULL DEFAULT '', placement TEXT NOT NULL DEFAULT '',
    PRIMARY KEY (replica, at, kind)
);
CREATE INDEX IF NOT EXISTS model_usage_replica_transitions_by_time
    ON model_usage_replica_transitions(at);
CREATE TABLE IF NOT EXISTS model_usage_replica_baselines (
    identity TEXT NOT NULL, replica TEXT NOT NULL,
    reading TEXT NOT NULL,
    PRIMARY KEY (identity, replica)
);
"""

_ADD_COLUMNS = _USAGE_COLUMNS + COUNTER_COLUMNS
_MAX_COLUMNS = tuple(column for column, _gauge in _GAUGE_COLUMNS)


def _upsert_sql() -> str:
    """The one statement that folds a tick into its bucket row.

    An added column stays NULL until a tick measures it (absent is not zero);
    a gauge column keeps the higher of what is there and what arrived.
    """
    columns = (
        ["identity", "replica", "bucket", "node_id", "node_name", "model", "engine",
         "placement", "attempted_seconds", "covered_seconds", "reset_ticks", "gap_ticks",
         "band_includes_short_replies"]
        + list(_ADD_COLUMNS) + list(_MAX_COLUMNS)
    )
    table = "model_usage_replica_buckets"
    updates = [
        "node_id = CASE WHEN excluded.node_id = '' THEN {t}.node_id ELSE excluded.node_id END",
        "node_name = CASE WHEN excluded.node_name = '' THEN {t}.node_name ELSE excluded.node_name END",
        # A second model inside one bucket is counted, so the bucket can be
        # withheld from the charts: its figures belong to neither model.
        "models_seen = {t}.models_seen + (CASE WHEN excluded.model != '' AND {t}.model != ''"
        " AND excluded.model != {t}.model THEN 1 ELSE 0 END)",
        "model = CASE WHEN excluded.model = '' THEN {t}.model ELSE excluded.model END",
        "engine = CASE WHEN excluded.engine = '' THEN {t}.engine ELSE excluded.engine END",
        "placement = CASE WHEN excluded.placement = '' THEN {t}.placement ELSE excluded.placement END",
        "attempted_seconds = {t}.attempted_seconds + excluded.attempted_seconds",
        "covered_seconds = {t}.covered_seconds + excluded.covered_seconds",
        "reset_ticks = {t}.reset_ticks + excluded.reset_ticks",
        "gap_ticks = {t}.gap_ticks + excluded.gap_ticks",
        "band_includes_short_replies = MAX({t}.band_includes_short_replies,"
        " excluded.band_includes_short_replies)",
    ]
    updates += [
        "{c} = CASE WHEN excluded.{c} IS NULL THEN {{t}}.{c}"
        " ELSE COALESCE({{t}}.{c}, 0) + excluded.{c} END".format(c=column)
        for column in _ADD_COLUMNS
    ]
    updates += [
        "{c} = CASE WHEN excluded.{c} IS NULL THEN {{t}}.{c}"
        " WHEN {{t}}.{c} IS NULL THEN excluded.{c}"
        " ELSE MAX({{t}}.{c}, excluded.{c}) END".format(c=column)
        for column in _MAX_COLUMNS
    ]
    return (
        "INSERT INTO {t}({cols}) VALUES ({marks}) "
        "ON CONFLICT(identity, replica, bucket) DO UPDATE SET {updates}"
    ).format(
        t=table, cols=", ".join(columns), marks=", ".join("?" for _ in columns),
        updates=", ".join(update.format(t=table) for update in updates),
    ), columns


_UPSERT, _UPSERT_COLUMNS = _upsert_sql()


def ensure_schema(connection: sqlite3.Connection) -> None:
    connection.executescript(SCHEMA)


def _number(value: Any) -> Optional[float]:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        return None
    try:
        number = float(value)
    except (OverflowError, ValueError):
        return None
    return number if number == number and abs(number) != float("inf") else None


def _write_row(connection: sqlite3.Connection, values: Mapping[str, Any]) -> None:
    connection.execute(_UPSERT, [values.get(column) for column in _UPSERT_COLUMNS])


def _base_row(
    identity: str, replica: str, clock: float, meta: Mapping[str, Any], attempted: float,
) -> Dict[str, Any]:
    return {
        "identity": identity, "replica": replica, "bucket": performance_bucket_start(clock),
        "node_id": str(meta.get("node_id") or ""), "node_name": str(meta.get("node_name") or ""),
        "model": str(meta.get("model") or ""), "engine": str(meta.get("engine") or ""),
        "placement": str(meta.get("placement") or ""),
        "attempted_seconds": max(0.0, float(attempted)), "covered_seconds": 0.0,
        "reset_ticks": 0, "gap_ticks": 0, "band_includes_short_replies": 0,
    }


def _tick_seconds(performance: Mapping[str, Any]) -> float:
    """How long this tick was: the poller's measured period, else the nominal one."""
    period = _number(performance.get("tick_period"))
    return period if period is not None and period > 0 else SERVING_POLL_SECONDS


def _edge_set_id(connection: sqlite3.Connection, family: int, edges: Sequence[float]) -> int:
    """The id of an edge set, created on first sight. One row per distinct edge list."""
    text = json.dumps([float(edge) for edge in edges])
    fingerprint = "{}:{}".format(int(family), text)
    connection.execute(
        "INSERT OR IGNORE INTO model_usage_edge_sets(family, fingerprint, edges) VALUES (?,?,?)",
        (int(family), fingerprint, text),
    )
    row = connection.execute(
        "SELECT edge_set_id FROM model_usage_edge_sets WHERE fingerprint = ?", (fingerprint,),
    ).fetchone()
    return int(row[0])


def _reading(performance: Mapping[str, Any], clock: float) -> Dict[str, Any]:
    """The part of a tick's performance reading that is kept as the next baseline.

    With WHEN it was read: the wall clock always, and the poller's stamp
    (session, tick number, monotonic reading) when the reading carried one.
    """
    counters = performance.get("counters") if isinstance(performance.get("counters"), Mapping) else {}
    histogram = performance.get("histogram") if isinstance(performance.get("histogram"), Mapping) else None
    reading: Dict[str, Any] = {
        "counters": {
            name: _number(counters.get(name)) for name in COUNTER_COLUMNS
            if _number(counters.get(name)) is not None
        },
        "created": _number(performance.get("created")),
        "read_at": float(clock),
    }
    session, tick, monotonic = performance.get("session"), performance.get("tick"), _number(performance.get("monotonic"))
    if isinstance(session, str) and isinstance(tick, int) and not isinstance(tick, bool) and monotonic is not None:
        reading.update({"session": session, "tick": tick, "monotonic": monotonic})
    if histogram:
        reading["histogram"] = {
            "family": int(histogram.get("family") or 0),
            "edges": [float(edge) for edge in histogram.get("edges") or ()],
            "cumulative": [float(count) for count in histogram.get("cumulative") or ()],
            "short_replies": _number(histogram.get("short_replies")),
        }
    return reading


def _restarted(prior: Mapping[str, Any], reading: Mapping[str, Any]) -> bool:
    """Whether the engine restarted between two performance readings.

    Its creation stamp changed, a counter went backwards, or the inter-token
    pair moved inconsistently (either half backwards): in each case an increase
    taken across the two would span time the new process did not run.
    """
    before, after = prior.get("created"), reading.get("created")
    if before is not None and after is not None and before != after:
        return True
    old = prior.get("counters") or {}
    return any(
        name in old and value < old[name]
        for name, value in (reading.get("counters") or {}).items()
    )


def _histogram_bins(
    prior: Mapping[str, Any], reading: Mapping[str, Any],
) -> Optional[Dict[str, Any]]:
    """``{family, edges, bins, short_known}`` for the tick, or ``None``.

    ``None`` when either reading has no histogram, the family or the edges
    changed, or a count went backwards. For the per-request family, requests
    that wrote 0 or 1 token are taken out of the fastest bin, where the engine
    files them with a time-per-token of zero.
    """
    before, after = prior.get("histogram"), reading.get("histogram")
    if not before or not after:
        return None
    if before.get("family") != after.get("family") or before.get("edges") != after.get("edges"):
        return None
    bins = bin_increases(before.get("cumulative"), after.get("cumulative") or ())
    if bins is None:
        return None
    short_known = True
    if after.get("family") == FAMILY_PER_REQUEST:
        old_short, new_short = before.get("short_replies"), after.get("short_replies")
        if old_short is None or new_short is None or new_short < old_short:
            short_known = False
        elif bins:
            bins[0] = max(0, bins[0] - int(round(new_short - old_short)))
    return {"family": after["family"], "edges": after["edges"], "bins": bins, "short_known": short_known}


def _span(prior: Mapping[str, Any], reading: Mapping[str, Any], tick_period: Any) -> Tuple[Optional[float], bool]:
    """``(seconds, rateable)`` between the stored performance baseline and this reading."""
    stamped = stamped_span(prior, reading)
    if stamped is not None:
        return stamped
    before = _number(prior.get("read_at"))
    if before is None:
        return None, False
    span = float(reading["read_at"]) - before
    return span, span_is_rateable(span, tick_period)


def record_tick(
    connection: sqlite3.Connection, *, identity: str, replica: str, clock: float,
    usage_restarted: bool, usage_added: Mapping[str, Any],
    performance: Mapping[str, Any], meta: Mapping[str, Any],
) -> str:
    """Fold one successful read of a replica into its bucket; return what the tick was.

    ``usage_restarted`` and ``usage_added`` are the ledger's own verdict and
    increases for the usage counters. The span is measured from this table's
    own baseline (see the module docstring). Returns ``"first"``, ``"reset"``,
    ``"gap"`` or ``"rate"``.
    """
    tick = _tick_seconds(performance)
    row = _base_row(identity, replica, clock, meta, tick)
    gauges = performance.get("gauges") if isinstance(performance.get("gauges"), Mapping) else {}
    for column, gauge in _GAUGE_COLUMNS:
        row[column] = _number(gauges.get(gauge))
    stored = connection.execute(
        "SELECT reading FROM model_usage_replica_baselines WHERE identity = ? AND replica = ?",
        (identity, replica),
    ).fetchone()
    prior = json.loads(stored[0]) if stored else None
    reading = _reading(performance, clock)
    connection.execute(
        """INSERT INTO model_usage_replica_baselines(identity, replica, reading) VALUES (?,?,?)
           ON CONFLICT(identity, replica) DO UPDATE SET reading = excluded.reading""",
        (identity, replica, json.dumps(reading)),
    )
    span, rateable = (None, False) if prior is None else _span(prior, reading, performance.get("tick_period"))
    if prior is None or span is None:
        kind = "first"
    elif usage_restarted or _restarted(prior, reading):
        kind = "reset"
        row["reset_ticks"] = 1
    elif not rateable:
        kind = "gap"
        row["gap_ticks"] = 1
    else:
        kind = "rate"
        row["covered_seconds"] = span
        for column in _USAGE_COLUMNS:
            if column in usage_added:
                row[column] = float(usage_added[column])
        old = prior.get("counters") or {}
        for column, value in (reading.get("counters") or {}).items():
            if column in old:
                row[column] = value - old[column]
        histogram = _histogram_bins(prior, reading)
        if histogram is not None:
            if not histogram["short_known"]:
                row["band_includes_short_replies"] = 1
            _write_bins(connection, identity, replica, row["bucket"], histogram)
    _write_row(connection, row)
    return kind


def _write_bins(
    connection: sqlite3.Connection, identity: str, replica: str, bucket: int,
    histogram: Mapping[str, Any],
) -> None:
    """Add a tick's non-zero bin increases to the bucket's histogram rows."""
    nonzero = [(index, count) for index, count in enumerate(histogram["bins"]) if count > 0]
    if not nonzero:
        return
    edge_set = _edge_set_id(connection, histogram["family"], histogram["edges"])
    connection.executemany(
        """INSERT INTO model_usage_replica_histogram(identity, replica, bucket, edge_set_id,
               edge_index, count) VALUES (?,?,?,?,?,?)
           ON CONFLICT(identity, replica, bucket, edge_set_id, edge_index)
           DO UPDATE SET count = model_usage_replica_histogram.count + excluded.count""",
        [(identity, replica, bucket, edge_set, index, count) for index, count in nonzero],
    )


def record_miss(
    connection: sqlite3.Connection, *, identity: str, replica: str, clock: float,
    seconds: Any, meta: Mapping[str, Any],
) -> None:
    """A tick in which the replica was a target and could not be read.

    Adds the tick's seconds to ``attempted_seconds`` and nothing else, so the
    bucket's coverage says how much of the time was really read.
    """
    period = _number(seconds)
    attempted = period if period is not None and period > 0 else SERVING_POLL_SECONDS
    _write_row(connection, _base_row(identity, replica, clock, meta, attempted))


def prune(connection: sqlite3.Connection, clock: float) -> None:
    """Drop performance rows past their retention, in the ledger's own prune pass."""
    oldest = performance_prune_before(clock)
    connection.execute("DELETE FROM model_usage_replica_buckets WHERE bucket < ?", (oldest,))
    # A replica's newest start or stop is kept however old it is (pass-6): it
    # says whether the replica is serving, which no later row would say.
    connection.execute(
        """DELETE FROM model_usage_replica_transitions WHERE at < ? AND EXISTS (
               SELECT 1 FROM model_usage_replica_transitions newer
               WHERE newer.replica = model_usage_replica_transitions.replica
                 AND newer.at > model_usage_replica_transitions.at)""",
        (oldest,),
    )
    removed = connection.execute(
        "DELETE FROM model_usage_replica_histogram WHERE bucket < ?", (oldest,),
    ).rowcount
    if removed:
        # Only when histogram rows went: scanning for unreferenced edge sets on
        # every write cost 6-20 ms at two to eight replicas (review nit).
        connection.execute(
            """DELETE FROM model_usage_edge_sets WHERE edge_set_id NOT IN
                   (SELECT DISTINCT edge_set_id FROM model_usage_replica_histogram)"""
        )


def bucket_rows(
    connection: sqlite3.Connection, since: float, until: Optional[float] = None,
) -> List[Dict[str, Any]]:
    """Every replica bucket row whose bucket starts in ``[since, until)``, oldest first."""
    clause, arguments = "bucket >= ?", [performance_bucket_start(since)]
    if until is not None:
        clause += " AND bucket < ?"
        arguments.append(int(until))
    rows = connection.execute(
        "SELECT * FROM model_usage_replica_buckets WHERE {} ORDER BY bucket, identity, replica".format(clause),
        arguments,
    ).fetchall()
    return [dict(row) for row in rows]


#: A replica joined the poller's target set, or left it (pass-4 review B1). The
#: dashboard's Started / Stopped markers come from these rows only, never from
#: a gap: a slow tick, a locked ledger or a moment the deployment record could
#: not be read leaves buckets without a row while the replica still served.
TRANSITION_STARTED = "started"
TRANSITION_STOPPED = "stopped"
TRANSITIONS = frozenset({"started", "stopped"})


def record_transition(
    connection: sqlite3.Connection, *, replica: str, kind: str, at: float, meta: Mapping[str, Any],
) -> None:
    """Write one start or stop of a replica; the same one twice is kept once.

    A start begins a new engine run: the performance baselines of the replica's
    previous run are dropped, so its first reading is a first read, never a
    counter reset. A fresh deploy was called "Engine restarted" (ACC-200); a
    restart is only ever a restart of an engine that was serving.
    """
    if kind not in TRANSITIONS:
        raise ValueError("unknown transition")
    if kind == TRANSITION_STARTED:
        connection.execute("DELETE FROM model_usage_replica_baselines WHERE replica = ?", (replica,))
    connection.execute(
        "INSERT OR IGNORE INTO model_usage_replica_transitions"
        "(replica, at, kind, identity, node_id, node_name, placement) VALUES (?, ?, ?, ?, ?, ?, ?)",
        (replica, float(at), kind, str(meta.get("identity") or ""), str(meta.get("node_id") or ""),
         str(meta.get("node_name") or ""), str(meta.get("placement") or "")),
    )


def last_transition_kinds(connection: sqlite3.Connection) -> Dict[str, str]:
    """``{replica: "started" | "stopped"}``: what each replica's newest record says."""
    rows = connection.execute(
        """SELECT t.replica, t.kind FROM model_usage_replica_transitions t
           JOIN (SELECT replica, MAX(at) AS at FROM model_usage_replica_transitions GROUP BY replica) last
             ON last.replica = t.replica AND last.at = t.at""",
    ).fetchall()
    return {str(row[0]): str(row[1]) for row in rows}


def transition_rows(connection: sqlite3.Connection, since: float, until: float) -> List[Dict[str, Any]]:
    """Every start and stop in ``[since, until)``, plus each replica's last one before ``since``.

    The earlier row says whether a replica was serving when the range began,
    so a bucket at its start is judged by the record, not by its emptiness.
    """
    before = connection.execute(
        """SELECT t.* FROM model_usage_replica_transitions t
           JOIN (SELECT replica, MAX(at) AS at FROM model_usage_replica_transitions
                 WHERE at < ? GROUP BY replica) last
             ON last.replica = t.replica AND last.at = t.at""", (float(since),),
    ).fetchall()
    inside = connection.execute(
        "SELECT * FROM model_usage_replica_transitions WHERE at >= ? AND at < ? ORDER BY at, replica",
        (float(since), float(until)),
    ).fetchall()
    return [dict(row) for row in before] + [dict(row) for row in inside]


def _stepped_sql() -> str:
    """One row per replica, layout and step bucket: sums summed, gauges and flags at their highest."""
    sums = ["attempted_seconds", "covered_seconds", "reset_ticks", "gap_ticks"] + list(_ADD_COLUMNS)
    highest = ["models_seen", "band_includes_short_replies"] + list(_MAX_COLUMNS)
    columns = ["SUM({0}) AS {0}".format(column) for column in sums]
    columns += ["MAX({0}) AS {0}".format(column) for column in highest]
    return (
        "SELECT identity, replica, node_id, MAX(node_name) AS node_name, model, engine, placement, "
        "? + ((bucket - ?) / ?) * ? AS bucket, {columns} "
        "FROM model_usage_replica_buckets WHERE bucket >= ? AND bucket < ? "
        "GROUP BY identity, replica, node_id, model, engine, placement, (bucket - ?) / ? "
        "ORDER BY bucket, identity, replica"
    ).format(columns=", ".join(columns))


_STEPPED = _stepped_sql()


def stepped_bucket_rows(
    connection: sqlite3.Connection, since: int, until: int, step: int,
) -> List[Dict[str, Any]]:
    """The rows in ``[since, until)`` already grouped into ``step``-second buckets from ``since``.

    The dashboard's range read (pass-3 review): the database does the fold, so
    eight replicas over seven days are 1 920 rows handed back, not 242 thousand.
    A bucket that held two models comes back as two rows, which the dashboard's
    fold marks as a model change rather than mixing.
    """
    start, end, grain = int(since), int(until), max(1, int(step))
    rows = connection.execute(_STEPPED, (start, start, grain, grain, start, end, start, grain)).fetchall()
    return [dict(row) for row in rows]


def stepped_histogram_rows(
    connection: sqlite3.Connection, since: int, until: int, step: int,
) -> List[Dict[str, Any]]:
    """Speed-band rows already summed into ``step``-second buckets from ``since`` (pass-4 review).

    The same shape as :func:`histogram_rows`: a bucket that held two edge sets
    stays two entries, which `serving_buckets.merge_bins` refuses to pool.
    """
    start, end, grain = int(since), int(until), max(1, int(step))
    rows = connection.execute(
        """SELECT h.identity, h.replica, ? + ((h.bucket - ?) / ?) * ? AS bucket, h.edge_set_id, h.edge_index,
                  SUM(h.count) AS count, e.family, e.edges
           FROM model_usage_replica_histogram h
           JOIN model_usage_edge_sets e ON e.edge_set_id = h.edge_set_id
           WHERE h.bucket >= ? AND h.bucket < ?
           GROUP BY h.identity, h.replica, (h.bucket - ?) / ?, h.edge_set_id, h.edge_index
           ORDER BY bucket, h.identity, h.replica, h.edge_set_id, h.edge_index""",
        (start, start, grain, grain, start, end, start, grain),
    ).fetchall()
    return _group_bins(rows)


def _group_bins(rows: Sequence[Any]) -> List[Dict[str, Any]]:
    grouped: Dict[tuple, Dict[str, Any]] = {}
    for row in rows:
        key = (row["identity"], row["replica"], row["bucket"], row["edge_set_id"])
        entry = grouped.setdefault(key, {
            "identity": row["identity"], "replica": row["replica"], "bucket": row["bucket"],
            "edge_set_id": row["edge_set_id"], "family": row["family"],
            "edges": json.loads(row["edges"]), "bins": {},
        })
        entry["bins"][int(row["edge_index"])] = int(row["count"])
    return list(grouped.values())


def histogram_rows(
    connection: sqlite3.Connection, since: float, until: Optional[float] = None,
) -> List[Dict[str, Any]]:
    """Speed-band rows in a span: ``{identity, replica, bucket, family, edges, bins}``.

    One entry per replica, bucket and edge set, ``bins`` mapping a bin index to
    its count. A bucket holding two edge sets comes back as two entries, which
    `serving_buckets.merge_bins` refuses to pool.
    """
    clause, arguments = "h.bucket >= ?", [performance_bucket_start(since)]
    if until is not None:
        clause += " AND h.bucket < ?"
        arguments.append(int(until))
    rows = connection.execute(
        """SELECT h.identity, h.replica, h.bucket, h.edge_set_id, h.edge_index, h.count,
                  e.family, e.edges
           FROM model_usage_replica_histogram h
           JOIN model_usage_edge_sets e ON e.edge_set_id = h.edge_set_id
           WHERE {} ORDER BY h.bucket, h.identity, h.replica, h.edge_set_id, h.edge_index""".format(clause),
        arguments,
    ).fetchall()
    return _group_bins(rows)
