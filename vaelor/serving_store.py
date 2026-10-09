"""The serving measurement: what the ``/metrics`` scrape stores and the tab reads.

Split out of :mod:`vaelor.telemetry_store` (VD-147, slice S1a) with no behaviour
change. The telemetry store keeps retention and the per-second machine history;
this module keeps the second measurement beside it - the GPU serving engine's
gauges, written every ten seconds by the serving pollers
(:mod:`vaelor.serving_metrics_poller`) and read by the Performance snapshot.

It reuses the store's own connection (``Database.set_tagged`` / ``Database.get``)
rather than opening a second client, and both directions are honest by
construction: nothing to write writes nothing, and nothing to read is ``None``,
never a zero-filled row.
"""

from __future__ import annotations

from datetime import datetime, timezone
from typing import Any, Dict, Optional

from .cluster_placement import CONTROLLER_PLACEMENT_ID

#: The measurement the serving-metrics scrape writes and the Performance tab
#: reads (Phase E′). Separate from `HISTORY_MEASUREMENT` because it is a
#: different cadence and a different shape — the GPU engine's live serving gauges
#: (decode/prefill rate, queue depth) rather than the per-second machine
#: telemetry — and one name, for the same "two spellings finds nothing" reason.
SERVING_MEASUREMENT = "serving"


def record_serving(
    database: Any,
    gauges: Dict[str, Any],
    node: str = CONTROLLER_PLACEMENT_ID,
) -> bool:
    """Store one scraped serving-metrics sample, tagged for the controller.

    The write half of the Phase E′ serving scrape. It **reuses the store's
    existing connection** through `Database.set_tagged` rather than opening a
    second InfluxDB client — the same write path per-node telemetry ingest uses —
    and writes into `SERVING_MEASUREMENT` beside the history rows.

    **Honest by construction, both directions.** `database is None` is retention
    switched off (or not yet running), and an empty `gauges` is a scrape that
    found nothing worth recording; either writes nothing and returns `False`,
    never a zero-filled row. `set_tagged` refuses a store that is not ready, and
    any store exception is caught here and reported as `False`, so a bad write
    cannot crash the poll loop that calls this.

    The `node` tag stamps the sample as the CONTROLLER's — the GPU engine runs on
    the controller's loopback — so a future per-worker vLLM scrape can write the
    same measurement under its own node without the read confusing the two.
    """
    if database is None or not gauges:
        return False
    try:
        accepted, _detail = database.set_tagged(
            SERVING_MEASUREMENT, {"node": node}, dict(gauges)
        )
    except Exception:  # noqa: BLE001 - a scrape write may never crash the loop
        return False
    return bool(accepted)


def row_time_seconds(value: Any) -> Optional[float]:
    """An InfluxDB row ``time`` as epoch seconds, or None when unreadable.

    `Database.get` queries without an ``epoch`` argument, so InfluxDB answers
    with an RFC 3339 string such as ``2026-09-28T10:11:12.123456789Z`` - nine
    fractional digits, which ``datetime.fromisoformat`` refuses before Python
    3.11. The fraction is cut to microseconds here rather than rejected. An
    integer or float is already epoch seconds and passes through. Anything
    else is None: a reading whose age cannot be established is not stamped
    with a guessed one.
    """
    if isinstance(value, bool):
        return None
    if isinstance(value, (int, float)):
        return float(value)
    if not isinstance(value, str) or not value.strip():
        return None
    text = value.strip()
    if text.endswith("Z"):
        text = text[:-1] + "+00:00"
    head, dot, rest = text.partition(".")
    if dot:
        digits = ""
        while rest and rest[0].isdigit():
            digits, rest = digits + rest[0], rest[1:]
        text = head + "." + (digits[:6] or "0").ljust(6, "0") + rest
    try:
        parsed = datetime.fromisoformat(text)
    except ValueError:
        return None
    if parsed.tzinfo is None:
        parsed = parsed.replace(tzinfo=timezone.utc)
    return parsed.timestamp()


def latest_serving(database: Any) -> Optional[Dict[str, Any]]:
    """The newest stored serving sample as a dict, or None when there is none.

    The read half of the scrape, for the Performance route. `None` is the honest
    answer for every "nothing to show" case — retention off (`database is None`),
    a store that cannot be read, or a measurement nothing has been scraped into
    yet — rather than a fabricated zero-filled reading. It reuses `Database.get`,
    which returns the single most recent row (or `None`), so no second query path
    is introduced.

    The row is stamped with ``sampled_at`` (epoch seconds from the row's own
    ``time``, or None when that cannot be read) so the reader can refuse to
    present an old sample as live (ACC-049): the poller stops writing the moment
    the engine stops, unloads or crashes, and without the stamp the last row it
    wrote would read as the engine's current state for ever.
    """
    if database is None:
        return None
    try:
        row = database.get(SERVING_MEASUREMENT, n=1)
    except Exception:  # noqa: BLE001 - an unreadable store degrades to "nothing"
        return None
    if not isinstance(row, dict):
        return None
    return {**row, "sampled_at": row_time_seconds(row.get("time"))}
