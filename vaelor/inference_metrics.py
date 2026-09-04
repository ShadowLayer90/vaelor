"""Persistent bounded telemetry for the managed inference gateway."""

from __future__ import annotations

import os
import sqlite3
import time
from contextlib import closing
from pathlib import Path
from typing import Optional

from .runtime_paths import env_value, state_path


class InferenceGatewayMetrics:
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
            connection.commit()
        try:
            os.chmod(self.database_path, 0o600)
        except OSError:
            pass

    def record(
        self, *, status: int, duration_ms: int, streaming: bool,
        prompt_tokens: int = 0, completion_tokens: int = 0,
        response_bytes: int = 0,
    ):
        with closing(sqlite3.connect(self.database_path)) as connection:
            connection.execute(
                """INSERT INTO requests(
                    created_at,status,duration_ms,streaming,prompt_tokens,
                    completion_tokens,response_bytes
                ) VALUES(?,?,?,?,?,?,?)""",
                (
                    time.time(), int(status), int(duration_ms), int(streaming),
                    max(0, int(prompt_tokens)), max(0, int(completion_tokens)),
                    max(0, int(response_bytes)),
                ),
            )
            connection.execute(
                """DELETE FROM requests WHERE id NOT IN (
                    SELECT id FROM requests ORDER BY id DESC LIMIT 10000
                )"""
            )
            connection.commit()

    def snapshot(self, window_seconds: int = 86400):
        cutoff = time.time() - max(60, min(int(window_seconds), 30 * 86400))
        with closing(sqlite3.connect(self.database_path)) as connection:
            connection.row_factory = sqlite3.Row
            row = connection.execute(
                """SELECT COUNT(*) AS requests,
                    SUM(CASE WHEN status >= 400 THEN 1 ELSE 0 END) AS failures,
                    SUM(streaming) AS streaming_requests,
                    COALESCE(AVG(duration_ms), 0) AS average_latency_ms,
                    COALESCE(MAX(duration_ms), 0) AS maximum_latency_ms,
                    COALESCE(SUM(prompt_tokens), 0) AS prompt_tokens,
                    COALESCE(SUM(completion_tokens), 0) AS completion_tokens,
                    COALESCE(SUM(response_bytes), 0) AS response_bytes
                FROM requests WHERE created_at >= ?""",
                (cutoff,),
            ).fetchone()
        return {
            key: round(value, 1) if key == "average_latency_ms" else int(value or 0)
            for key, value in dict(row).items()
        }
