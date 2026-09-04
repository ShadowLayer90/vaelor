"""Create the jobs database and bring an older one forward in place."""

from __future__ import annotations

import sqlite3


SCHEMA = """
CREATE TABLE IF NOT EXISTS jobs (
    id TEXT PRIMARY KEY,
    type TEXT NOT NULL,
    state TEXT NOT NULL,
    actor TEXT NOT NULL,
    payload_json TEXT NOT NULL,
    progress INTEGER NOT NULL DEFAULT 0,
    message TEXT NOT NULL DEFAULT '',
    phase TEXT NOT NULL DEFAULT '',
    blocker_layer TEXT NOT NULL DEFAULT '',
    result_json TEXT NOT NULL DEFAULT '{}',
    created_at INTEGER NOT NULL,
    updated_at INTEGER NOT NULL
);
CREATE INDEX IF NOT EXISTS idx_jobs_created
    ON jobs(created_at DESC);
CREATE TABLE IF NOT EXISTS job_events (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    job_id TEXT NOT NULL,
    state TEXT NOT NULL,
    progress INTEGER NOT NULL,
    message TEXT NOT NULL,
    phase TEXT NOT NULL DEFAULT '',
    blocker_layer TEXT NOT NULL DEFAULT '',
    created_at INTEGER NOT NULL,
    FOREIGN KEY(job_id) REFERENCES jobs(id) ON DELETE CASCADE
);
"""

# Columns added after the first release, each with the definition an existing
# table is brought forward with.
ADDED_JOB_COLUMNS = (
    ("result_json", "TEXT NOT NULL DEFAULT '{}'"),
    ("attempt", "INTEGER NOT NULL DEFAULT 1"),
    ("retry_of", "TEXT"),
    ("phase", "TEXT NOT NULL DEFAULT ''"),
    ("blocker_layer", "TEXT NOT NULL DEFAULT ''"),
)
ADDED_EVENT_COLUMNS = (
    ("phase", "TEXT NOT NULL DEFAULT ''"),
    ("blocker_layer", "TEXT NOT NULL DEFAULT ''"),
)

# Research phases were renamed, and rows written under the old names would
# otherwise render as an unknown phase for the life of the appliance.
_RENAMED_RESEARCH_PHASES = """
CASE phase
    WHEN "queued" THEN "interpreting"
    WHEN "needs-input" THEN "needs_input"
    WHEN "ready-for-review" THEN "ready_for_review"
    WHEN "failed" THEN "needs_input"
    WHEN "cancelled" THEN "needs_input"
    WHEN "completed" THEN "ready_for_review"
    ELSE phase
END
"""


def _existing_columns(connection: sqlite3.Connection, table: str) -> set[str]:
    return {
        row["name"]
        for row in connection.execute(f"PRAGMA table_info({table})").fetchall()
    }


def apply_schema(connection: sqlite3.Connection) -> None:
    """Create the tables, add any missing columns, and normalize phase names."""
    connection.executescript(SCHEMA)
    for table, additions in (
        ("jobs", ADDED_JOB_COLUMNS), ("job_events", ADDED_EVENT_COLUMNS),
    ):
        columns = _existing_columns(connection, table)
        for name, definition in additions:
            if name not in columns:
                connection.execute(
                    f"ALTER TABLE {table} ADD COLUMN {name} {definition}"
                )
    connection.execute(
        "UPDATE jobs SET phase = {} WHERE type = \"application.research\"".format(
            _RENAMED_RESEARCH_PHASES
        )
    )
    connection.execute(
        "UPDATE job_events SET phase = {} WHERE job_id IN ("
        "SELECT id FROM jobs WHERE type = \"application.research\")".format(
            _RENAMED_RESEARCH_PHASES
        )
    )
    connection.commit()
