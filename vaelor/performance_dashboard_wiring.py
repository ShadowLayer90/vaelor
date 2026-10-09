"""How the Performance dashboard is plugged into the control plane (VD-147 S4).

`control_plane_runtime.py` is at its size limit, so the dashboard's wiring is
here and the host adds exactly three calls:

* :func:`dashboard_callbacks` - spread into the API callbacks: the dashboard's
  own telemetry reader, the chart-slot store, and the roll-up's state (is it
  ready, and the progress sentence while it is not);
* :func:`register_dashboard` - the two routes, called beside
  ``register_cluster_performance_routes``;
* :func:`start_rollup` - once the telemetry store is running: creates the
  one-minute continuous query (idempotent) and backfills the history already
  held, in the background, one chunk at a time.

The roll-up is only read once it is ready (pass-3 review S4-B2). A failure to
create or backfill it is retried every :data:`ROLLUP_RETRY_SECONDS`, and the
finished marker is rewritten every :data:`MARKER_REFRESH_SECONDS` so it never
ages out of the retention policy. Until it is ready, every range reads the
raw rows.
"""

from __future__ import annotations

import logging
import threading
from typing import Any, Callable, Dict, Optional

from .api_performance_dashboard_routes import register_performance_dashboard_routes
from .performance_dashboard_slots import ChartSlots
from .telemetry_reader import TelemetryReader
from .telemetry_reader_rollup import (
    backfill, backfill_progress, ensure_continuous_query, preparing_sentence, refresh_marker,
)
from .telemetry_store import DEFAULT_RETENTION_DAYS

LOGGER = logging.getLogger(__name__)

#: Chunks one backfill pass runs before yielding to the store for a moment.
BACKFILL_CHUNKS_PER_PASS = 4
#: The pause between backfill passes, in seconds.
BACKFILL_PAUSE_SECONDS = 2.0
#: How long after a failure the roll-up is tried again.
ROLLUP_RETRY_SECONDS = 300.0
#: How often a finished roll-up's marker is rewritten (and the query checked).
MARKER_REFRESH_SECONDS = 86400.0

#: What a viewer reads while the roll-up could not be prepared.
ROLLUP_UNAVAILABLE = (
    "Older history could not be prepared yet; Vaelor tries again every few minutes "
    "and reads the full-rate history meanwhile."
)

#: The callback key the control plane reads the state from to start the thread.
ROLLUP_STATE_KEY = "telemetry_rollup_state"


class RollupState:
    """What the roll-up thread knows: created, ready, the last error, the progress."""

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self.created = False
        self.ready = False
        self.error = ""
        self.progress: Dict[str, Any] = {}

    def update(self, **changes: Any) -> None:
        with self._lock:
            for key, value in changes.items():
                setattr(self, key, value)

    def is_ready(self) -> bool:
        with self._lock:
            return bool(self.ready)

    def note(self) -> str:
        """The screen's sentence while older history is not rolled up yet; ``""`` once it is.

        A failed attempt is said in plain words (pass-4 review) - never the
        store's own error text, which can carry its address.
        """
        with self._lock:
            if self.ready:
                return ""
            if self.error and not self.progress:
                return ROLLUP_UNAVAILABLE
            return preparing_sentence(dict(self.progress))


def dashboard_callbacks(database_name: str, retention_days: int = DEFAULT_RETENTION_DAYS) -> Dict[str, Any]:
    """The dashboard's sources, for the API callbacks."""
    reader = TelemetryReader(database_name)
    state = RollupState()
    return {
        "telemetry_reader": reader,
        "chart_slots": ChartSlots(retention_days=retention_days),
        "telemetry_rollup_ready": state.is_ready,
        "telemetry_rollup_note": state.note,
        ROLLUP_STATE_KEY: state,
    }


def register_dashboard(context: Any) -> None:
    register_performance_dashboard_routes(context)


def start_rollup(
    reader: TelemetryReader, retention_days: int = DEFAULT_RETENTION_DAYS,
    state: Optional[RollupState] = None,
    wait: Callable[[float], bool] = threading.Event().wait,
    log: Optional[logging.Logger] = None,
) -> threading.Thread:
    """Create the roll-up and backfill it on a daemon thread; failures are logged and retried.

    ``wait(seconds)`` returning true stops the thread (an ``Event.wait``).
    """
    log = log or LOGGER
    state = state or RollupState()
    client = reader.locked()

    def prepare() -> bool:
        """One attempt: the query, then the backfill to completion. False when stopped."""
        # A query about to be replaced stops being read first (pass-5 review).
        ensure_continuous_query(client, reader.database_name, before_replace=lambda: state.update(ready=False))
        state.update(created=True)
        progress = backfill_progress(client, reader.database_name)
        if not progress["complete"]:
            # A roll-up with no finished marker (a wiped store) is not whole yet.
            state.update(ready=False, progress=progress)
        while not progress["complete"]:
            progress = backfill(client, reader.database_name,
                                retention_seconds=int(retention_days) * 86400,
                                max_chunks=BACKFILL_CHUNKS_PER_PASS)
            state.update(progress=progress)
            if not progress["complete"] and wait(BACKFILL_PAUSE_SECONDS):
                return False
        refresh_marker(client, reader.database_name, progress)
        state.update(ready=True, error="", progress=progress)
        return True

    def run() -> None:
        reported = ""
        while True:
            try:
                if not prepare():
                    return
                pause = MARKER_REFRESH_SECONDS
            except Exception as error:  # noqa: BLE001 - the dashboard degrades to raw rows; retried
                message = " ".join(str(error).split())[:200]
                # Whatever failed, the roll-up is not trusted until a pass completes.
                state.update(error=message, ready=False)
                if message != reported:
                    log.warning("The one-minute telemetry roll-up could not be prepared; "
                                "retrying in %d s: %s", int(ROLLUP_RETRY_SECONDS), message)
                    reported = message
                pause = ROLLUP_RETRY_SECONDS
            if wait(pause):
                return

    thread = threading.Thread(target=run, name="telemetry-rollup", daemon=True)
    thread.start()
    return thread
