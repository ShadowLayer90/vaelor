"""How much each served model did, counted by the model itself.

**Why the model is the source (owner decision, 2026-09-28).** Usage used to be
counted where a request passed through a Vaelor process - the inference gateway
and AI Chat. The LLM Server's LAN door (nginx straight to the model), agents,
and anything else reaching the model never passed through either, so an
external app used the model all day and the per-deployment card never moved
(ACC-044). Every request, whichever door it came through, is counted by the
serving engine's own Prometheus counters, so those are the truth for "how much
did the model do":

* **vLLM (a cluster deployment)**: ``vllm:prompt_tokens_total``,
  ``vllm:generation_tokens_total`` and ``vllm:request_success_total`` per
  replica - requests AND tokens.
* **llama.cpp (this machine's GPU model)**: ``llamacpp:prompt_tokens_total`` and
  ``llamacpp:tokens_predicted_total``. llama.cpp keeps no request counter, so
  on this engine requests are reported as NOT COUNTED, never as zero.

**One scrape, not two.** The counters arrive through the serving pollers that
already read ``/metrics`` every ten seconds for the Performance card
(:mod:`vaelor.serving_metrics_poller`), handed over per replica through their
``usage_sink``; this module adds no scrape of its own.

**Counters are cumulative per replica and reset when the engine restarts**
(an unload and reload, a crash, a redeploy). Each replica's last reading is
kept as a baseline, persisted so a control-plane restart neither re-counts nor
loses what the model did meanwhile. A reading at or above the baseline adds the
difference; a reading BELOW it means the engine restarted, and the whole new
reading is what it did since, which is Prometheus's own ``increase`` rule. The
first reading of a replica never seen before only sets its baseline: what the
engine did before Vaelor first read it was not counted here, and the totals say
when counting began rather than claim that history.

**Keyed by what the owner deployed, not by an internal id (ACC-045).** A cluster
deployment is keyed by its name, which a redeploy keeps, so totals continue
across a redeploy instead of starting a new anonymous row; this machine's model
is keyed by the model it serves. Each row carries the owner-readable name.

**The same readings also feed the Performance dashboard** (VD-147): when a
poller hands over a replica's performance reading beside its usage counters,
:meth:`ModelUsageLedger.observe` writes the replica's twenty-second bucket
(`model_usage_replica`) in the same transaction, from the same baseline. Usage
and rates part ways only on a restart or a gap: usage counts the whole new
reading, a rate claims nothing for a span it cannot measure.

Written and read by the control plane only (``0600``).
"""

from __future__ import annotations

import logging
import os
import sqlite3
import time
from contextlib import closing
from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional, Sequence

from . import model_usage_replica as replica_store
from .gpu_pool_units import LLAMACPP_ENGINE, VLLM_ENGINE
from .runtime_paths import env_value, state_path
from .serving_metrics import counter_increase
from .usage_rollup import bucket_start, prune_before, window_start

LOGGER = logging.getLogger(__name__)

#: The engines a row can come from; the card names the engine beside the row.
#: The words are `gpu_pool_units`'s.
ENGINE_VLLM = VLLM_ENGINE
ENGINE_LLAMACPP = LLAMACPP_ENGINE

#: The three counters a reading may carry, in the store's column names.
COUNTERS = ("prompt_tokens", "completion_tokens", "requests")

#: A deployment the model was read from within this long is "counting now";
#: older than this it is shown with when it was last read. Three poll ticks.
LIVE_WITHIN_SECONDS = 30.0

#: Baselines of replicas not read for this long are forgotten: that replica is
#: gone, and a new one on the same address starts a fresh baseline.
BASELINE_RETENTION_SECONDS = 30 * 86400

#: Identity prefixes: a cluster deployment by its name, this machine's model by
#: the model it serves.
CLUSTER_PREFIX = "cluster:"
LOCAL_PREFIX = "local:"


def counters_from_vllm(aggregate: Mapping[str, Any]) -> Dict[str, float]:
    """The ledger's counters from one replica's parsed vLLM aggregate.

    ``aggregate`` is ``parse_vllm_serving(body)["aggregate"]``. A family the
    engine did not report is absent, never zero.
    """
    mapping = {
        "prompt_tokens": "prompt_tokens_total",
        "completion_tokens": "generation_tokens_total",
        "requests": "request_success_total",
    }
    return {
        column: float(aggregate[key]) for column, key in mapping.items()
        if isinstance(aggregate.get(key), (int, float))
    }


def counters_from_llamacpp(counters: Mapping[str, Any]) -> Dict[str, float]:
    """The ledger's counters from :func:`~vaelor.serving_metrics.parse_llamacpp_counters`.

    llama.cpp counts tokens only; ``requests`` is never present, which is how
    the ledger knows this engine does not count them.
    """
    mapping = {
        "prompt_tokens": "prompt_tokens_total",
        "completion_tokens": "tokens_predicted_total",
    }
    return {
        column: float(counters[key]) for column, key in mapping.items()
        if isinstance(counters.get(key), (int, float))
    }


class ModelUsageLedger:
    """Per-deployment usage accumulated from the model's own counters."""

    def __init__(self, database_path: Optional[str] = None):
        self.database_path = database_path or env_value(
            "VAELOR_MODEL_USAGE_DB", "PM_MODEL_USAGE_DB",
            state_path("cluster/model-usage.sqlite3"),
        )
        Path(self.database_path).parent.mkdir(parents=True, exist_ok=True, mode=0o700)
        with closing(self._connect()) as connection:
            connection.executescript(
                """
                CREATE TABLE IF NOT EXISTS model_usage_totals (
                    identity TEXT PRIMARY KEY,
                    name TEXT NOT NULL DEFAULT '', model TEXT NOT NULL DEFAULT '',
                    engine TEXT NOT NULL DEFAULT '',
                    prompt_tokens INTEGER NOT NULL DEFAULT 0,
                    completion_tokens INTEGER NOT NULL DEFAULT 0,
                    requests INTEGER NOT NULL DEFAULT 0,
                    requests_counted INTEGER NOT NULL DEFAULT 0,
                    counting_since REAL NOT NULL,
                    last_used_at REAL, last_read_at REAL NOT NULL
                );
                CREATE TABLE IF NOT EXISTS model_usage_buckets (
                    identity TEXT NOT NULL, bucket INTEGER NOT NULL,
                    prompt_tokens INTEGER NOT NULL DEFAULT 0,
                    completion_tokens INTEGER NOT NULL DEFAULT 0,
                    requests INTEGER NOT NULL DEFAULT 0,
                    PRIMARY KEY (identity, bucket)
                );
                CREATE INDEX IF NOT EXISTS model_usage_buckets_by_time
                    ON model_usage_buckets(bucket);
                CREATE TABLE IF NOT EXISTS model_usage_baselines (
                    identity TEXT NOT NULL, replica TEXT NOT NULL,
                    prompt_tokens REAL, completion_tokens REAL, requests REAL,
                    read_at REAL NOT NULL,
                    PRIMARY KEY (identity, replica)
                );
                """
            )
            replica_store.ensure_schema(connection)
            # One writer (the serving poller) and short dashboard reads: with a
            # write-ahead log neither blocks the other. Set once; it persists
            # in the file. Switching needs a moment with no other connection:
            # on a busy first start after an upgrade the file stays in its old
            # journal mode (correct, only slower) and the next start switches
            # it (review nit).
            try:
                connection.execute("PRAGMA journal_mode=WAL")  # pairs-with: sqlite-journal-mode-tight
            except sqlite3.OperationalError as error:
                LOGGER.info("Usage ledger stays in its current journal mode for now: %s", error)
            connection.commit()
        try:
            os.chmod(self.database_path, 0o600)
        except OSError:
            pass

    def _connect(self) -> sqlite3.Connection:
        connection = sqlite3.connect(self.database_path, timeout=15)
        connection.row_factory = sqlite3.Row
        return connection

    def observe(
        self, *, identity: str, name: str, model: str, engine: str,
        replica: str, counters: Mapping[str, float], now: Optional[float] = None,
        performance: Optional[Mapping[str, Any]] = None,
        node_id: str = "", node_name: str = "", placement: str = "",
    ) -> Dict[str, int]:
        """Fold one replica's cumulative counters in; returns what was added.

        See the module docstring for the reset rule. A reading with no counters
        at all changes nothing (the engine answered without these families).
        ``requests_counted`` is set once any reading carried a request counter,
        so a llama.cpp row never shows a request count it does not have.

        ``performance`` is the replica's performance reading for this tick (its
        cumulative decode, cache and histogram counters, its gauges and the
        poller's measured tick period). With it, the replica's performance
        bucket is written in this same transaction, tagged with the machine
        (``node_id``, and ``node_name`` as it is called NOW, so a machine
        removed later keeps its name in history), the model and the
        ``placement``.
        """
        readings = {
            column: max(0.0, float(counters[column]))
            for column in COUNTERS if column in counters
        }
        if not identity or not replica or not readings:
            return {}
        clock = time.time() if now is None else float(now)
        with closing(self._connect()) as connection:
            prior = connection.execute(
                "SELECT * FROM model_usage_baselines WHERE identity = ? AND replica = ?",
                (identity, replica),
            ).fetchone()
            added, restarted = _increase(prior, readings)
            connection.execute(
                """INSERT INTO model_usage_baselines(identity, replica, prompt_tokens,
                       completion_tokens, requests, read_at) VALUES (?,?,?,?,?,?)
                   ON CONFLICT(identity, replica) DO UPDATE SET
                       prompt_tokens = excluded.prompt_tokens,
                       completion_tokens = excluded.completion_tokens,
                       requests = excluded.requests, read_at = excluded.read_at""",
                (
                    identity, replica, readings.get("prompt_tokens"),
                    readings.get("completion_tokens"), readings.get("requests"), clock,
                ),
            )
            used = any(added.values())
            connection.execute(
                """INSERT INTO model_usage_totals(identity, name, model, engine,
                       prompt_tokens, completion_tokens, requests, requests_counted,
                       counting_since, last_used_at, last_read_at)
                   VALUES (?,?,?,?,?,?,?,?,?,?,?)
                   ON CONFLICT(identity) DO UPDATE SET
                       name = excluded.name, model = excluded.model,
                       engine = excluded.engine,
                       prompt_tokens = model_usage_totals.prompt_tokens + excluded.prompt_tokens,
                       completion_tokens = model_usage_totals.completion_tokens
                           + excluded.completion_tokens,
                       requests = model_usage_totals.requests + excluded.requests,
                       requests_counted = MAX(model_usage_totals.requests_counted,
                           excluded.requests_counted),
                       last_used_at = COALESCE(excluded.last_used_at,
                           model_usage_totals.last_used_at),
                       last_read_at = excluded.last_read_at""",
                (
                    identity, str(name or ""), str(model or ""), str(engine or ""),
                    added["prompt_tokens"], added["completion_tokens"], added["requests"],
                    1 if "requests" in readings else 0, clock,
                    clock if used else None, clock,
                ),
            )
            if used:
                connection.execute(
                    """INSERT INTO model_usage_buckets(identity, bucket, prompt_tokens,
                           completion_tokens, requests) VALUES (?,?,?,?,?)
                       ON CONFLICT(identity, bucket) DO UPDATE SET
                           prompt_tokens = model_usage_buckets.prompt_tokens
                               + excluded.prompt_tokens,
                           completion_tokens = model_usage_buckets.completion_tokens
                               + excluded.completion_tokens,
                           requests = model_usage_buckets.requests + excluded.requests""",
                    (
                        identity, bucket_start(clock), added["prompt_tokens"],
                        added["completion_tokens"], added["requests"],
                    ),
                )
            if performance is not None:
                replica_store.record_tick(
                    connection, identity=identity, replica=replica, clock=clock,
                    usage_restarted=restarted, performance=performance,
                    usage_added={column: added[column] for column in readings},
                    meta={
                        "node_id": node_id, "node_name": node_name, "model": model,
                        "engine": engine, "placement": placement,
                    },
                )
            self._prune(connection, clock)
            connection.commit()
        return added

    @staticmethod
    def _prune(connection: sqlite3.Connection, clock: float) -> None:
        """Drop rows past their retention: usage buckets, baselines, performance rows."""
        connection.execute(
            "DELETE FROM model_usage_buckets WHERE bucket < ?", (prune_before(clock),)
        )
        connection.execute(
            "DELETE FROM model_usage_baselines WHERE read_at < ?",
            (clock - BASELINE_RETENTION_SECONDS,),
        )
        connection.execute(
            """DELETE FROM model_usage_replica_baselines WHERE (identity, replica) NOT IN
                   (SELECT identity, replica FROM model_usage_baselines)"""
        )
        replica_store.prune(connection, clock)

    def observe_miss(
        self, *, identity: str, replica: str, seconds: Optional[float] = None,
        model: str = "", engine: str = "", node_id: str = "", node_name: str = "",
        placement: str = "", now: Optional[float] = None,
    ) -> None:
        """Record a tick in which a replica was a target and could not be read.

        Nothing is counted: the tick's seconds are added to the bucket's
        attempted time, so its coverage says how much of the interval was
        really read and a partial total is never drawn as a whole one.
        """
        if not identity or not replica:
            return
        clock = time.time() if now is None else float(now)
        with closing(self._connect()) as connection:
            replica_store.record_miss(
                connection, identity=identity, replica=replica, clock=clock, seconds=seconds,
                meta={
                    "node_id": node_id, "node_name": node_name, "model": model,
                    "engine": engine, "placement": placement,
                },
            )
            # A replica that is never read still ages out (review nit).
            self._prune(connection, clock)
            connection.commit()

    def replica_buckets(
        self, since: float, until: Optional[float] = None,
    ) -> List[Dict[str, Any]]:
        """The per-replica performance buckets starting in ``[since, until)``."""
        with closing(self._connect()) as connection:
            return replica_store.bucket_rows(connection, since, until)

    def replica_buckets_stepped(self, since: float, until: float, step: int) -> List[Dict[str, Any]]:
        """The same rows grouped into ``step``-second buckets by the database (`stepped_bucket_rows`)."""
        with closing(self._connect()) as connection:
            return replica_store.stepped_bucket_rows(connection, int(since), int(until), int(step))

    def record_transition(
        self, *, replica: str, kind: str, at: Optional[float] = None, identity: str = "",
        node_id: str = "", node_name: str = "", placement: str = "",
    ) -> None:
        """A replica became a serving target, or stopped being one (`model_usage_replica.record_transition`)."""
        moment = time.time() if at is None else float(at)
        with closing(self._connect()) as connection:
            replica_store.record_transition(connection, replica=replica, kind=kind, at=moment, meta={
                "identity": identity, "node_id": node_id, "node_name": node_name, "placement": placement})
            connection.commit()

    def replica_histograms_stepped(self, since: float, until: float, step: int) -> List[Dict[str, Any]]:
        """The speed-band rows summed into ``step``-second buckets by the database."""
        with closing(self._connect()) as connection:
            return replica_store.stepped_histogram_rows(connection, int(since), int(until), int(step))

    def last_transitions(self) -> Dict[str, str]:
        """Each replica's last recorded start or stop (`model_usage_replica.last_transition_kinds`)."""
        with closing(self._connect()) as connection:
            return replica_store.last_transition_kinds(connection)

    def replica_transitions(self, since: float, until: float) -> List[Dict[str, Any]]:
        """The starts and stops for a span, with each replica's state as it began."""
        with closing(self._connect()) as connection:
            return replica_store.transition_rows(connection, since, until)

    def replica_histograms(
        self, since: float, until: Optional[float] = None,
    ) -> List[Dict[str, Any]]:
        """The speed-band rows for the same span (`model_usage_replica.histogram_rows`)."""
        with closing(self._connect()) as connection:
            return replica_store.histogram_rows(connection, since, until)

    def deployments(self, now: Optional[float] = None) -> List[Dict[str, Any]]:
        """Every deployment the model has been read for, most recently used first."""
        clock = time.time() if now is None else float(now)
        with closing(self._connect()) as connection:
            rows = connection.execute(
                """SELECT * FROM model_usage_totals
                   ORDER BY COALESCE(last_used_at, 0) DESC, identity"""
            ).fetchall()
        return [
            {
                "identity": row["identity"],
                "name": row["name"] or row["identity"],
                "model": row["model"],
                "engine": row["engine"],
                "prompt_tokens": int(row["prompt_tokens"]),
                "completion_tokens": int(row["completion_tokens"]),
                # None, not 0, when this engine does not count requests.
                "requests": int(row["requests"]) if row["requests_counted"] else None,
                "counting_since": row["counting_since"],
                "last_used_at": row["last_used_at"],
                "last_read_at": row["last_read_at"],
                "live": clock - float(row["last_read_at"]) <= LIVE_WITHIN_SECONDS,
            }
            for row in rows
        ]

    def window_totals(
        self, window_seconds: float, *, now: Optional[float] = None,
    ) -> Dict[str, Any]:
        """Tokens (and counted requests) across every deployment over a window.

        ``since`` is the real start of the span counted: the earliest bucket
        read, or when counting began if that is later; ``engines`` names the
        engines that served in it. ``requests`` is
        None when no deployment in the window counts requests, and
        ``requests_partial`` is True when some do and some (llama.cpp) do not,
        so a screen does not present a partial count as the whole.
        """
        clock = time.time() if now is None else float(now)
        since = window_start(clock, window_seconds)
        with closing(self._connect()) as connection:
            row = connection.execute(
                """SELECT COALESCE(SUM(b.prompt_tokens), 0) AS prompt_tokens,
                          COALESCE(SUM(b.completion_tokens), 0) AS completion_tokens,
                          COALESCE(SUM(b.requests), 0) AS requests,
                          COALESCE(MAX(t.requests_counted), 0) AS requests_counted,
                          COALESCE(MIN(t.requests_counted), 0) AS every_counted
                   FROM model_usage_buckets b
                   LEFT JOIN model_usage_totals t ON t.identity = b.identity
                   WHERE b.bucket >= ?""",
                (since,),
            ).fetchone()
            started = connection.execute(
                "SELECT MIN(counting_since) AS started FROM model_usage_totals"
            ).fetchone()
            engines = [
                str(item[0]) for item in connection.execute(
                    """SELECT DISTINCT t.engine FROM model_usage_buckets b
                       JOIN model_usage_totals t ON t.identity = b.identity
                       WHERE b.bucket >= ?""",
                    (since,),
                )
            ]
        return {
            "since": max(since, started["started"]) if started and started["started"] else since,
            "engines": engines,
            "prompt_tokens": int(row["prompt_tokens"]),
            "completion_tokens": int(row["completion_tokens"]),
            "requests": int(row["requests"]) if row["requests_counted"] else None,
            "requests_partial": bool(row["requests_counted"]) and not row["every_counted"],
            "counting_since": started["started"] if started else None,
        }


def deployment_usage(rows: Optional[Sequence[Mapping[str, Any]]]) -> List[Dict[str, Any]]:
    """The Performance card's per-deployment rows, from :meth:`ModelUsageLedger.deployments`.

    Named by the deployment the owner made, never by an internal id (ACC-045);
    ``requests`` stays None for an engine that does not count them. ``None``
    input (the ledger unwired or unreadable) is an empty list here - the
    snapshot says separately whether the ledger could be read.
    """
    normalized: List[Dict[str, Any]] = []
    for row in rows or []:
        if not isinstance(row, Mapping) or not row.get("identity"):
            continue
        requests = row.get("requests")
        normalized.append({
            "identity": str(row["identity"]),
            "name": str(row.get("name") or row["identity"]),
            "model": str(row.get("model") or ""),
            "engine": str(row.get("engine") or ""),
            "prompt_tokens": int(row.get("prompt_tokens") or 0),
            "completion_tokens": int(row.get("completion_tokens") or 0),
            "requests": None if requests is None else int(requests),
            "counting_since": row.get("counting_since"),
            "last_used_at": row.get("last_used_at"),
            "last_read_at": row.get("last_read_at"),
            "live": bool(row.get("live")),
        })
    return normalized


def _increase(
    prior: Optional[sqlite3.Row], readings: Mapping[str, float],
) -> "tuple[Dict[str, int], bool]":
    """``(added, restarted)``: what each counter added since ``prior``, and whether
    the engine restarted in between.

    No prior reading adds nothing: it only sets the baseline. Any counter lower
    than its baseline means the engine restarted (`serving_metrics.counter_increase`),
    and then every counter's whole reading is new, since one restart resets
    them all together.
    """
    added = {column: 0 for column in COUNTERS}
    if prior is None:
        return added, False
    moves = {
        column: counter_increase(float(prior[column]), value)
        for column, value in readings.items() if prior[column] is not None
    }
    restarted = any(reset for _increase_, reset in moves.values())
    for column, value in readings.items():
        if restarted:
            added[column] = int(round(value))
        elif column in moves:
            added[column] = int(round(moves[column][0]))
        # A family this replica did not report before: this reading is its
        # baseline, exactly as a replica's first reading is (adds nothing).
    return added, restarted
