"""The one-minute host roll-up the dashboard reads for ranges of six hours and more (VD-147 S4, §2.3).

A continuous query keeps ``history_1m``: per node and per minute, the SUM and
the COUNT of each averaged field and the MAX of each peak field. Sums and
counts rather than means, so a dashboard bucket spanning several minutes is
Σsum ÷ Σcount and a partly filled minute weighs what it holds.

* :func:`ensure_continuous_query` creates it once - it asks
  ``SHOW CONTINUOUS QUERIES`` first, so a second start changes nothing;
* :func:`backfill` fills the history an existing appliance already holds, one
  hour per statement (:data:`BACKFILL_CHUNK_SECONDS`), newest last, and writes
  its progress to a marker measurement after every chunk, so a restart resumes
  and the screen can say "Older history is being prepared (N % done)" rather
  than draw a gap (:func:`backfill_progress`);
* :func:`refresh_marker` rewrites the finished marker now and then: the marker
  lives under the same retention as the rows, and a marker written once would
  age out after a week and start the whole backfill again (pass-3 review).

``FOR 3m`` recomputes the last three minutes on each run, which covers a
worker's rows arriving up to the ingest route's accepted clock skew late.
The retention policy is the telemetry store's own, so the roll-up is kept for
exactly as long as the rows it summarises.
"""

from __future__ import annotations

import logging
import re
import time
from typing import Any, Callable, Dict, Optional

from .telemetry_reader import MAX_FIELDS, MEAN_FIELDS, ROLLUP_MEASUREMENT
from .telemetry_store import HISTORY_MEASUREMENT, RETENTION_POLICY_NAME

LOGGER = logging.getLogger(__name__)

#: The continuous query's name; its existence is what makes creation idempotent.
CONTINUOUS_QUERY_NAME = "cq_history_1m"

#: One backfill statement covers this much history. An hour of one-second rows
#: from two machines is about 7200 rows per field: small enough that one
#: statement never holds the store for long.
BACKFILL_CHUNK_SECONDS = 3600

#: The marker measurement the backfill writes its progress to.
BACKFILL_MARKER = "history_1m_backfill"

#: What the screen says while older history is still being rolled up.
BACKFILL_PREPARING = "Older history is being prepared ({percent} % done)."


def _aggregates() -> str:
    columns = []
    for field in MEAN_FIELDS:
        columns.append('sum("{0}") AS "{0}_sum", count("{0}") AS "{0}_n"'.format(field))
    columns += ['max("{0}") AS "{0}_max"'.format(field) for field in MAX_FIELDS]
    return ", ".join(columns)


def _target(database: str) -> str:
    return '"{}"."{}"."{}"'.format(database, RETENTION_POLICY_NAME, ROLLUP_MEASUREMENT)


def continuous_query_statement(database: str) -> str:
    """The ``CREATE CONTINUOUS QUERY`` the store runs every minute."""
    return (
        'CREATE CONTINUOUS QUERY "{name}" ON "{db}" RESAMPLE EVERY 1m FOR 3m BEGIN '
        'SELECT {aggregates} INTO {target} FROM "{source}" GROUP BY time(1m), "node" END'
    ).format(
        name=CONTINUOUS_QUERY_NAME, db=database, aggregates=_aggregates(),
        target=_target(database), source=HISTORY_MEASUREMENT,
    )


def backfill_statement(database: str, start: int, end: int) -> str:
    """One chunk of the backfill: the same aggregation over ``[start, end)``."""
    return (
        'SELECT {aggregates} INTO {target} FROM "{source}" '
        'WHERE time >= {start}s AND time < {end}s GROUP BY time(1m), "node"'
    ).format(
        aggregates=_aggregates(), target=_target(database), source=HISTORY_MEASUREMENT,
        start=int(start), end=int(end),
    )


def _normalised(text: str) -> str:
    """Lower case, no quotes, one space, none around brackets and commas."""
    flat = re.sub(r"\s+", " ", str(text).replace('"', "").lower())
    return re.sub(r"\s*([(),])\s*", r"\1", flat).strip()


_UNIT_SECONDS = {"s": 1, "m": 60, "h": 3600, "d": 86400}
_DURATION = r"(\d+)(s|m|h|d)"


def _seconds(amount: str, unit: str) -> int:
    return int(amount) * _UNIT_SECONDS[unit]


def computes(text: str) -> Dict[str, Any]:
    """What a continuous query computes, however InfluxDB spells it (pass-5 review).

    The aggregates as ``(function, field, alias)``, the measurement read and
    the one written (without database or policy), the grouping interval and
    the resample window, all in seconds - so quoting, case, spacing, the
    database and policy prefixes InfluxDB adds and ``1m`` against ``60s``
    change nothing, and a different computation always shows.
    """
    flat = _normalised(text)
    aggregates = set(re.findall(r"(\w+)\(([\w.]+)\)\s*as\s*([\w.]+)", flat))
    into = re.search(r"\binto\s*([\w.]+)", flat)
    source = re.search(r"\bfrom\s*([\w.]+)", flat)
    group = re.search(r"group by time\(" + _DURATION + r"\)", flat)
    tags = re.search(r"group by time\(\w+\)((?:,[\w*]+)*)", flat)
    target = into.group(1).split(".") if into else []
    resample = re.search(r"resample every " + _DURATION + r" for " + _DURATION, flat)
    return {
        "aggregates": {(function, field.rsplit(".", 1)[-1], alias) for function, field, alias in aggregates},
        "into": into.group(1).rsplit(".", 1)[-1] if into else "",
        # The retention policy written into (pass-6): a roll-up kept for a
        # different period than its rows is a different roll-up.
        "into_policy": target[-2] if len(target) >= 2 else "",
        # The tags grouped by (``node`` must be there, or every machine is one).
        "tags": frozenset(tag for tag in (tags.group(1).split(",") if tags else []) if tag),
        "where": bool(re.search(r"\bwhere\b", flat)),
        "from": source.group(1).rsplit(".", 1)[-1] if source else "",
        "group": _seconds(*group.groups()) if group else None,
        "resample": (_seconds(*resample.groups()[:2]), _seconds(*resample.groups()[2:])) if resample else None,
    }


def query_is_current(shown: str) -> bool:
    """Whether a stored continuous query computes exactly what this version reads."""
    return computes(shown) == computes(continuous_query_statement("vaelor"))


#: Whether this process has already said it replaced the query (said once).
_REPLACE_REPORTED: Dict[str, bool] = {}


def ensure_continuous_query(
    client: Any, database: str, before_replace: Optional[Callable[[], None]] = None,
) -> bool:
    """Create the continuous query, or replace one that computes something else; True when it did.

    A replaced query's roll-up is rebuilt: ``before_replace`` runs first (the
    dashboard stops reading the roll-up), then the marker is reset, so the
    backfill runs again and overwrites every minute in the old shape. A replace
    is logged once per process.
    """
    shown = client.query("SHOW CONTINUOUS QUERIES")
    raw = getattr(shown, "raw", shown)
    for series in (raw.get("series") if isinstance(raw, dict) else None) or []:
        for row in series.get("values") or []:
            if row and row[0] == CONTINUOUS_QUERY_NAME:
                if len(row) > 1 and query_is_current(str(row[1])):
                    return False
                if before_replace is not None:
                    before_replace()
                if not _REPLACE_REPORTED.get(database):
                    _REPLACE_REPORTED[database] = True
                    LOGGER.warning("The one-minute roll-up's continuous query computed something else; "
                                   "it was replaced and its history is being rebuilt.")
                client.query('DROP CONTINUOUS QUERY "{}" ON "{}"'.format(CONTINUOUS_QUERY_NAME, database))
                refresh_marker(client, database, {"done": 0, "total": 0, "complete": False, "start": 0})
    client.query(continuous_query_statement(database))
    return True


def backfill_progress(client: Any, database: str) -> Dict[str, Any]:
    """``{"done", "total", "complete", "start"}`` from the marker; an absent marker is "not started"."""
    result = client.query(
        'SELECT last("done") AS "done", last("total") AS "total", last("complete") AS "complete", '
        'last("start") AS "start" FROM "{}"'.format(BACKFILL_MARKER), database=database,
    )
    raw = getattr(result, "raw", result)
    series = (raw.get("series") if isinstance(raw, dict) else None) or []
    if not series or not series[0].get("values"):
        return {"done": 0, "total": 0, "complete": False, "start": None}
    row = dict(zip(series[0].get("columns") or [], series[0]["values"][0]))
    return {
        "done": int(row.get("done") or 0), "total": int(row.get("total") or 0),
        "complete": bool(row.get("complete")), "start": row.get("start"),
    }


def preparing_sentence(progress: Optional[Dict[str, Any]]) -> str:
    """The screen's sentence while the backfill runs, or ``""`` when done or not needed."""
    if not progress or progress.get("complete") or not progress.get("total"):
        return ""
    percent = int(100 * min(1.0, progress.get("done", 0) / progress["total"]))
    return BACKFILL_PREPARING.format(percent=percent)


def refresh_marker(client: Any, database: str, progress: Dict[str, Any], now: Optional[float] = None) -> None:
    """Write the marker again, stamped now, so it never ages out of the retention policy."""
    moment = int(time.time() if now is None else now)
    client.write_points([{
        "measurement": BACKFILL_MARKER, "time": moment * 1_000_000_000,
        "fields": {"done": float(progress.get("done") or 0), "total": float(progress.get("total") or 0),
                   "complete": float(bool(progress.get("complete"))), "start": float(progress.get("start") or 0)},
    }], database=database)


def backfill(
    client: Any, database: str, *, retention_seconds: int, now: Optional[float] = None,
    max_chunks: Optional[int] = None,
) -> Dict[str, Any]:
    """Roll up the history the store already holds, resuming from the marker.

    Walks from the oldest retained hour to the minute the continuous query
    started covering, one chunk per statement, recording progress after each.
    ``max_chunks`` bounds one call (a background thread calls it repeatedly).
    Returns the progress it left.
    """
    moment = int(time.time() if now is None else now)
    progress = backfill_progress(client, database)
    if progress["complete"]:
        return progress
    start = int(progress["start"]) if progress["start"] else moment - int(retention_seconds)
    start -= start % BACKFILL_CHUNK_SECONDS
    total = max(1, -(-(moment - start) // BACKFILL_CHUNK_SECONDS))
    done = progress["done"]
    ran = 0
    while done < total and (max_chunks is None or ran < max_chunks):
        chunk_start = start + done * BACKFILL_CHUNK_SECONDS
        client.query(backfill_statement(database, chunk_start, chunk_start + BACKFILL_CHUNK_SECONDS),
                     database=database)
        done += 1
        ran += 1
        client.write_points([{
            "measurement": BACKFILL_MARKER,
            "fields": {"done": float(done), "total": float(total),
                       "complete": float(done >= total), "start": float(start)},
        }], database=database)
    return {"done": done, "total": total, "complete": done >= total, "start": start}
