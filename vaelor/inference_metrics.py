"""Persistent bounded telemetry for the managed inference gateway.

Two layers in one file. ``requests`` is the DETAIL: one row per request, the
newest ``DETAIL_ROWS`` only, which the Performance tab reads for latency
percentiles. ``request_buckets`` is the ROLL-UP: the same requests counted per
:mod:`vaelor.usage_rollup` bucket and never capped by row count, so a window's
counts stay complete however busy the gateway was (ACC-047). Once the detail
has been pruned, :meth:`InferenceGatewayMetrics.detail_retained_since` says from
when it still covers, so a figure built from detail can say so.

Every row names its ``source``: the inference gateway's own requests, or AI
Chat's, which the control plane records here too (Observability Unit 2). Only
the gateway's are external API traffic (ACC-046). Rows written before the
column existed cannot be told apart - AI Chat's and the gateway's looked the
same - so they are marked ``unattributed`` rather than guessed as gateway.

**The roll-up starts when it is created.** A store upgraded from before it has
detail rows but no buckets, so a window reaches back only to that moment:
:meth:`InferenceGatewayMetrics.window_totals` reports ``since`` as the later of
the window's start and the roll-up's, and a screen states it (ACC-047).
"""

from __future__ import annotations

import os
import sqlite3
import time
from contextlib import closing
from pathlib import Path
from typing import Any, Dict, Iterable, Optional

from .usage_rollup import (
    CLIENT_ERROR_STATUS, FAILURE_STATUS, SOURCE_AI_CHAT, SOURCE_GATEWAY,
    SOURCE_UNATTRIBUTED, STATUS_CLIENT_ERROR, STATUS_FAILURE, bucket_start,
    prune_before, status_class, window_start,
)
from .runtime_paths import env_value, state_path

# The failure / client-error line (ACC-056) has ONE owner, `usage_rollup`, which
# every door's roll-up also classifies with; these names stay importable here
# for the RED windows, the 24-hour summary and the request trace.
__all__ = ["CLIENT_ERROR_STATUS", "FAILURE_STATUS", "is_client_error", "is_failure"]


def is_failure(status) -> bool:
    """Whether a recorded status is a server-side failure (5xx)."""
    return status_class(status) == STATUS_FAILURE


def is_client_error(status) -> bool:
    """Whether a recorded status is the caller's error (4xx)."""
    return status_class(status) == STATUS_CLIENT_ERROR


def detail_coverage(
    detail_retained_since: Optional[float], *, window_start: float, lifetime: bool,
    kept: Optional[int] = None,
) -> Dict[str, Any]:
    """Whether figures built from the request DETAIL cover the span they claim.

    ``partial`` is True once the detail has been pruned and the span reaches
    back past what it still holds: a lifetime reading always, a window when it
    starts before the oldest retained request. ``since`` is that oldest
    request's time, for the screen to state (ACC-047). ``kept`` is how many
    requests the store keeps one by one (the gateway's own cap by default).
    """
    kept = InferenceGatewayMetrics.DETAIL_ROWS if kept is None else int(kept)
    if detail_retained_since is None:
        return {"partial": False, "since": None, "kept": kept}
    partial = bool(lifetime) or float(detail_retained_since) > float(window_start)
    return {"partial": partial, "since": float(detail_retained_since), "kept": kept}


class InferenceGatewayMetrics:
    #: How many detail rows are kept. Older rows are pruned; the roll-up keeps
    #: their counts (see the module docstring).
    DETAIL_ROWS = 10000

    def __init__(self, database_path: Optional[str] = None):
        self.database_path = database_path or env_value(
            "VAELOR_INFERENCE_METRICS_DB", "PM_INFERENCE_METRICS_DB",
            state_path("cluster/inference-metrics.sqlite3"),
        )
        Path(self.database_path).parent.mkdir(parents=True, exist_ok=True, mode=0o700)
        with closing(sqlite3.connect(self.database_path)) as connection:
            connection.execute(
                """CREATE TABLE IF NOT EXISTS requests (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    created_at REAL NOT NULL, status INTEGER NOT NULL,
                    duration_ms INTEGER NOT NULL, streaming INTEGER NOT NULL,
                    prompt_tokens INTEGER NOT NULL, completion_tokens INTEGER NOT NULL,
                    response_bytes INTEGER NOT NULL
                )"""
            )
            columns = {row[1] for row in connection.execute("PRAGMA table_info(requests)")}
            if "source" not in columns:
                connection.execute(
                    "ALTER TABLE requests ADD COLUMN source TEXT NOT NULL DEFAULT '{}'".format(
                        SOURCE_UNATTRIBUTED
                    )
                )
            connection.executescript(
                """
                CREATE TABLE IF NOT EXISTS request_buckets (
                    source TEXT NOT NULL, bucket INTEGER NOT NULL,
                    requests INTEGER NOT NULL DEFAULT 0,
                    failures INTEGER NOT NULL DEFAULT 0,
                    client_errors INTEGER NOT NULL DEFAULT 0,
                    streaming INTEGER NOT NULL DEFAULT 0,
                    prompt_tokens INTEGER NOT NULL DEFAULT 0,
                    completion_tokens INTEGER NOT NULL DEFAULT 0,
                    response_bytes INTEGER NOT NULL DEFAULT 0,
                    duration_ms_sum REAL NOT NULL DEFAULT 0,
                    duration_ms_max REAL NOT NULL DEFAULT 0,
                    PRIMARY KEY (source, bucket)
                );
                CREATE TABLE IF NOT EXISTS detail_retention (
                    id INTEGER PRIMARY KEY CHECK (id = 1), pruned_at REAL NOT NULL
                );
                CREATE TABLE IF NOT EXISTS rollup_started (
                    id INTEGER PRIMARY KEY CHECK (id = 1), started_at REAL NOT NULL
                );
                """
            )
            connection.execute(
                "INSERT OR IGNORE INTO rollup_started(id, started_at) VALUES (1, ?)",
                (time.time(),),
            )
            connection.commit()
        try:
            os.chmod(self.database_path, 0o600)
        except OSError:
            pass

    def record(
        self, *, status: int, duration_ms: int, streaming: bool,
        prompt_tokens: int = 0, completion_tokens: int = 0,
        response_bytes: int = 0, source: str = SOURCE_GATEWAY,
    ):
        now = time.time()
        kind = status_class(status)
        prompt_tokens = max(0, int(prompt_tokens))
        completion_tokens = max(0, int(completion_tokens))
        response_bytes = max(0, int(response_bytes))
        with closing(sqlite3.connect(self.database_path)) as connection:
            connection.execute(
                """INSERT INTO requests(
                    created_at,status,duration_ms,streaming,prompt_tokens,
                    completion_tokens,response_bytes,source
                ) VALUES(?,?,?,?,?,?,?,?)""",
                (
                    now, int(status), int(duration_ms), int(streaming),
                    prompt_tokens, completion_tokens, response_bytes, str(source),
                ),
            )
            connection.execute(
                """INSERT INTO request_buckets(source, bucket, requests, failures,
                       client_errors, streaming, prompt_tokens, completion_tokens,
                       response_bytes, duration_ms_sum, duration_ms_max)
                   VALUES (?,?,1,?,?,?,?,?,?,?,?)
                   ON CONFLICT(source, bucket) DO UPDATE SET
                       requests = request_buckets.requests + 1,
                       failures = request_buckets.failures + excluded.failures,
                       client_errors = request_buckets.client_errors + excluded.client_errors,
                       streaming = request_buckets.streaming + excluded.streaming,
                       prompt_tokens = request_buckets.prompt_tokens + excluded.prompt_tokens,
                       completion_tokens = request_buckets.completion_tokens
                           + excluded.completion_tokens,
                       response_bytes = request_buckets.response_bytes + excluded.response_bytes,
                       duration_ms_sum = request_buckets.duration_ms_sum + excluded.duration_ms_sum,
                       duration_ms_max = MAX(request_buckets.duration_ms_max,
                           excluded.duration_ms_max)""",
                (
                    str(source), bucket_start(now), int(kind == STATUS_FAILURE),
                    int(kind == STATUS_CLIENT_ERROR), int(bool(streaming)),
                    prompt_tokens, completion_tokens, response_bytes,
                    float(duration_ms), float(duration_ms),
                ),
            )
            pruned = connection.execute(
                """DELETE FROM requests WHERE id NOT IN (
                    SELECT id FROM requests ORDER BY id DESC LIMIT ?
                )""",
                (self.DETAIL_ROWS,),
            ).rowcount
            if pruned > 0:
                connection.execute(
                    "INSERT OR REPLACE INTO detail_retention(id, pruned_at) VALUES (1, ?)",
                    (now,),
                )
            connection.execute(
                "DELETE FROM request_buckets WHERE bucket < ?", (prune_before(now),)
            )
            connection.commit()

    def detail_retained_since(self) -> Optional[float]:
        """When the per-request detail starts, or None while it still holds everything.

        None until the first prune; after it, the oldest retained request's
        time, so a window reaching back further than that is known to be
        partial rather than presented as complete (ACC-047).
        """
        with closing(sqlite3.connect(self.database_path)) as connection:
            pruned = connection.execute(
                "SELECT pruned_at FROM detail_retention WHERE id = 1"
            ).fetchone()
            if pruned is None:
                return None
            oldest = connection.execute("SELECT MIN(created_at) FROM requests").fetchone()
        return float(oldest[0]) if oldest and oldest[0] is not None else None

    def window_totals(
        self, window_seconds: float, *, sources: Optional[Iterable[str]] = None,
        now: Optional[float] = None,
    ) -> Dict[str, Any]:
        """Counts over a window from the roll-up - complete whatever the detail cap.

        ``sources`` narrows to those sources (the External API card reads only
        the gateway's). ``since`` is the real start of the span counted: the
        earliest bucket read, or when the roll-up began if that is later.
        Failures are server-side (5xx); a caller's 4xx is ``client_errors``.
        """
        clock = time.time() if now is None else float(now)
        since = window_start(clock, window_seconds)
        wanted = [str(item) for item in (sources or ())]
        where = "bucket >= ?"
        parameters: list = [since]
        if wanted:
            where += " AND source IN ({})".format(",".join("?" * len(wanted)))
            parameters.extend(wanted)
        with closing(sqlite3.connect(self.database_path)) as connection:
            connection.row_factory = sqlite3.Row
            row = connection.execute(
                """SELECT COALESCE(SUM(requests), 0) AS requests,
                          COALESCE(SUM(failures), 0) AS failures,
                          COALESCE(SUM(client_errors), 0) AS client_errors,
                          COALESCE(SUM(streaming), 0) AS streaming_requests,
                          COALESCE(SUM(prompt_tokens), 0) AS prompt_tokens,
                          COALESCE(SUM(completion_tokens), 0) AS completion_tokens,
                          COALESCE(SUM(response_bytes), 0) AS response_bytes,
                          COALESCE(SUM(duration_ms_sum), 0) AS duration_ms_sum,
                          COALESCE(MAX(duration_ms_max), 0) AS maximum_latency_ms
                   FROM request_buckets WHERE """ + where,
                parameters,
            ).fetchone()
            started = connection.execute(
                "SELECT started_at FROM rollup_started WHERE id = 1"
            ).fetchone()
        requests = int(row["requests"])
        return {
            "since": max(since, float(started["started_at"])) if started else since,
            "requests": requests,
            "failures": int(row["failures"]),
            "client_errors": int(row["client_errors"]),
            "streaming_requests": int(row["streaming_requests"]),
            "average_latency_ms": (
                round(float(row["duration_ms_sum"]) / requests, 1) if requests else 0.0
            ),
            "maximum_latency_ms": int(row["maximum_latency_ms"]),
            "prompt_tokens": int(row["prompt_tokens"]),
            "completion_tokens": int(row["completion_tokens"]),
            "response_bytes": int(row["response_bytes"]),
            "duration_ms_sum": float(row["duration_ms_sum"]),
        }

    def rows_between(self, start_time: float, end_time: float):
        """Raw ``(created_at, status, duration_ms, tokens)`` rows in a half-open
        window, oldest first.

        ``snapshot`` aggregates in SQL and so can only answer count/avg/max for a
        single window anchored at *now*; it cannot give a p95 (SQLite has no
        percentile) or compare one window against an earlier one. The Performance
        tab needs both: percentiles are computed in Python from these durations,
        and a recent window is diffed against the prior one to name what changed.
        The range is half-open ``[start, end)`` so two adjacent windows share no
        row. Columns are the few the percentile and token-rate maths need, not
        the whole row, so this stays a read the agent-parseable snapshot can lean
        on rather than a raw matrix dump. ``source`` says which door each
        request came through (the gateway's own, AI Chat's, or unattributed),
        so the Performance card can show every door apart.
        """
        with closing(sqlite3.connect(self.database_path)) as connection:
            connection.row_factory = sqlite3.Row
            rows = connection.execute(
                """SELECT created_at, status, duration_ms,
                    prompt_tokens, completion_tokens, source
                FROM requests
                WHERE created_at >= ? AND created_at < ?
                ORDER BY created_at ASC""",
                (float(start_time), float(end_time)),
            ).fetchall()
        return [dict(row) for row in rows]

    def snapshot(self, window_seconds: int = 86400):
        cutoff = time.time() - max(60, min(int(window_seconds), 30 * 86400))
        with closing(sqlite3.connect(self.database_path)) as connection:
            connection.row_factory = sqlite3.Row
            row = connection.execute(
                """SELECT COUNT(*) AS requests,
                    SUM(CASE WHEN status >= ? THEN 1 ELSE 0 END) AS failures,
                    SUM(CASE WHEN status >= ? AND status < ? THEN 1 ELSE 0 END)
                        AS client_errors,
                    SUM(streaming) AS streaming_requests,
                    COALESCE(AVG(duration_ms), 0) AS average_latency_ms,
                    COALESCE(MAX(duration_ms), 0) AS maximum_latency_ms,
                    COALESCE(SUM(prompt_tokens), 0) AS prompt_tokens,
                    COALESCE(SUM(completion_tokens), 0) AS completion_tokens,
                    COALESCE(SUM(response_bytes), 0) AS response_bytes
                FROM requests WHERE created_at >= ?""",
                (FAILURE_STATUS, CLIENT_ERROR_STATUS, FAILURE_STATUS, cutoff),
            ).fetchone()
        return {
            key: round(value, 1) if key == "average_latency_ms" else int(value or 0)
            for key, value in dict(row).items()
        }
