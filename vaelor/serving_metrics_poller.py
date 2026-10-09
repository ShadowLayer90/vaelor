"""Scrape the GPU serving engine's ``/metrics`` on a loop (Phase E′).

The controller-side half of the serving-metrics feature: a light background
poller that, only while the GPU AI-Chat engine is actually serving on a loopback
port, GETs its Prometheus ``/metrics`` endpoint, parses it with
:func:`vaelor.serving_metrics.parse_llamacpp_metrics`, and hands the normalized
gauges to the telemetry store. The parse is pure and lives next door; this
module owns the I/O, the timing and the lifecycle.

**Every collaborator is an injected seam, so the whole thing is testable with no
engine, no socket and no database:**

* ``endpoint_source`` — ``() -> Optional[str]``: the engine's ``/metrics`` URL
  right now, or ``None`` when nothing is serving there. Returning ``None`` is the
  discovery half of honest degradation — the poller writes nothing rather than
  scraping a stale or wrong endpoint.
* ``record`` — ``(gauges) -> bool``: stores one sample, honest-degrading on the
  retention state (see :func:`vaelor.serving_store.record_serving`).
* ``fetch`` — ``(url) -> str``: the HTTP GET. Production uses a short-timeout
  loopback ``urllib`` read; a test scripts the body (or an error).

**It never invents a reading and it never crashes the loop.** No endpoint, a
fetch that raises, a body that parses to nothing, or a write that fails each end
the tick with nothing stored; the read side then reports "not collected", which
is the truth. Any unexpected error inside a tick is caught and logged so a single
bad scrape cannot take the thread down.
"""

from __future__ import annotations

import logging
import threading
import time
import urllib.request
import uuid
from typing import Any, Callable, Dict, Iterable, List, Optional, Tuple

from .cluster_placement import CONTROLLER_PLACEMENT_ID
from .generation_health import (
    DECODE_SECONDS_FIELD, DECODE_TOKENS_FIELD, TTFT_COUNT_FIELD,
    TTFT_SECONDS_FIELD, ttft_bucket_field,
)
from .serving_buckets import maximum, pooled_rate, pooled_ratio
from .serving_metrics import (  # noqa: F401 - the two poll constants are re-exported
    GOODPUT_BUDGET_SECONDS,
    SERVING_FETCH_TIMEOUT_SECONDS,
    SERVING_POLL_SECONDS,
    TTFT_BUCKET_KEY_PREFIX,
    counter_increase,
    histogram_in_use,
    parse_llamacpp_counters,
    parse_llamacpp_metrics,
    parse_vllm_performance,
    parse_vllm_serving,
)

LOGGER = logging.getLogger(__name__)


def _http_get(url: str) -> str:
    """GET ``url`` and return its body as text, with a bounded timeout.

    Only ever called with a loopback ``/metrics`` URL the endpoint source built,
    so there is no untrusted-host or redirect surface to guard here; the timeout
    is the one thing that matters, so a dead engine cannot wedge the loop.
    """
    with urllib.request.urlopen(url, timeout=SERVING_FETCH_TIMEOUT_SECONDS) as response:  # noqa: S310 - loopback URL built by the caller
        return response.read().decode("utf-8", "replace")


#: What a poller hands its ``usage_sink``: a replica's name (``node:port`` for a
#: vLLM replica, the ``/metrics`` URL for this machine's llama.cpp), the
#: CUMULATIVE counters that one body carried, and the replica's performance
#: reading for the tick (VD-147): its cumulative decode, cache and histogram
#: counters, its gauges, and the poller's measured tick period. The usage ledger
#: (:mod:`vaelor.model_usage`) turns successive readings into what the model did
#: and into the dashboard's per-replica buckets; the pollers only pass on what
#: they already scraped, so neither adds a scrape. A sink may also carry a
#: ``miss(replica, seconds)`` method, called for a tick in which the replica was
#: a target and could not be read - that is what makes coverage measurable.
UsageSink = Callable[[str, Dict[str, float], Dict[str, Any]], None]


def _feed_usage(
    sink: Optional[UsageSink], replica: str, counters: Dict[str, float],
    log: logging.Logger, performance: Optional[Dict[str, Any]] = None,
) -> None:
    """Hand one replica's counters to the usage sink, never failing the tick."""
    if sink is None or not counters:
        return
    try:
        sink(replica, dict(counters), dict(performance or {}))
    except Exception as error:  # noqa: BLE001 - usage must never stop the card's scrape
        log.debug("Model usage from %s was not recorded: %s", replica, error)


#: The counters that make a vLLM body a reading of what the replica did. A
#: body with only its queue gauges (truncated, or a half-started engine) is
#: recorded as a miss, not as a read that did nothing.
_VLLM_USAGE_KEYS = ("generation_tokens_total", "prompt_tokens_total", "request_success_total")


class DiscoveryUnknown(RuntimeError):
    """Discovery could not tell what is serving (a store it reads failed); the targets are unchanged."""


def read_mode_strictly(store: Any = None) -> Any:
    """The cluster mode record, or :class:`DiscoveryUnknown` when it exists and cannot be read.

    `ClusterModeStore.read` falls back to single-machine mode on a read error -
    right for the screens, wrong for a start/stop record, where it would write a
    false "Stopped serving" (pass-5 review B1). An absent file is single-machine
    mode; a present file that will not read is not known.
    """
    import json
    from pathlib import Path

    from .gpu_cluster_mode_state import ClusterModeStore

    store = ClusterModeStore() if store is None else store
    path = getattr(store, "_path", None)
    if isinstance(path, Path) and path.exists():
        try:
            raw = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, ValueError) as error:
            raise DiscoveryUnknown("the cluster mode record could not be read") from error
        if not mode_record_is_valid(raw):
            # Readable but not a record the store writes: not known, not Mode A (pass-6).
            raise DiscoveryUnknown("the cluster mode record is not a valid record")
    return store.read()


def mode_record_is_valid(raw: Any) -> bool:
    """Whether a mode record is one the store could have written: a known mode and, for a cluster, its fields."""
    from .gpu_serving_target import MODE_CLUSTER, MODE_SINGLE

    if not isinstance(raw, dict):
        return False
    mode = raw.get("mode", MODE_SINGLE)
    if mode not in (MODE_SINGLE, MODE_CLUSTER):
        return False
    if mode != MODE_CLUSTER:
        return True
    port = raw.get("cluster_port", 0)
    return (isinstance(raw.get("deployment_name", ""), str)
            and isinstance(port, int) and not isinstance(port, bool) and port >= 0)


class TargetTracker:
    """Tells the sink which replicas joined or left a poller's targets (pass-4 B1, pass-5).

    Fed only after a discovery that succeeded, so a failed read never becomes
    a stop. On the first tick the record is reconciled: a target whose last
    recorded transition was a stop has started, and a replica recorded as
    serving that is no longer a target has stopped; a target with no record
    at all claims nothing.
    """

    def __init__(self, sink: Optional["UsageSink"], log: logging.Logger, local: bool) -> None:
        self._sink, self._log, self._local = sink, log, local
        self._current: Optional[set] = None

    def note(self, current: set) -> None:
        # Starts and stops kept from a locked ledger are written every tick,
        # not only with the next one (pass-6 review).
        flush = getattr(self._sink, "flush_transitions", None)
        if flush is not None:
            try:
                flush()
            except Exception as error:  # noqa: BLE001 - a marker must never stop the card's scrape
                self._log.debug("Kept serving starts and stops were not written yet: %s", error)
        previous, self._current = self._current, set(current)
        if previous is None:
            last = self._last()
            started = {replica for replica in current if last.get(replica) == "stopped"}
            stopped = {replica for replica, kind in last.items() if kind == "started" and replica not in current}
        else:
            started, stopped = current - previous, previous - current
        for replica in sorted(started):
            _feed_transition(self._sink, replica, "started", self._log)
        for replica in sorted(stopped):
            _feed_transition(self._sink, replica, "stopped", self._log)

    def _last(self) -> Dict[str, str]:
        reader = getattr(self._sink, "last_transitions", None)
        if reader is None:
            return {}
        try:
            found = reader() or {}
        except Exception as error:  # noqa: BLE001 - reconciling must never stop the card's scrape
            self._log.debug("The last recorded serving starts and stops were not read: %s", error)
            return {}
        return {replica: kind for replica, kind in found.items()
                if str(replica).startswith(LOCAL_REPLICA_PREFIX) == self._local}


#: A llama.cpp replica is named by its metrics URL; a vLLM one by "<node>:<port>".
LOCAL_REPLICA_PREFIX = "http"


def _feed_transition(sink: Optional[UsageSink], replica: str, kind: str, log: logging.Logger) -> None:
    """Tell the sink a replica joined or left the target set, never failing the tick."""
    transition = getattr(sink, "transition", None)
    if transition is None:
        return
    try:
        transition(replica, kind)
    except Exception as error:  # noqa: BLE001 - a marker must never stop the card's scrape
        log.debug("A serving start or stop of %s was not recorded: %s", replica, error)


def _feed_miss(
    sink: Optional[UsageSink], replica: str, seconds: Optional[float], log: logging.Logger,
) -> None:
    """Tell the sink a target replica could not be read this tick, never failing it."""
    miss = getattr(sink, "miss", None)
    if miss is None:
        return
    try:
        miss(replica, seconds)
    except Exception as error:  # noqa: BLE001 - coverage must never stop the card's scrape
        log.debug("A missed read of %s was not recorded: %s", replica, error)


class TickClock:
    """The measured time between the starts of a poller's ticks.

    A tick is not a fixed length: the poller sleeps its interval and then reads
    each replica in turn, so the real period is the interval plus however long
    the reads took. The dashboard's rate rule needs the real one (a slow but
    healthy tick must not be called a gap), so it is measured here, on the
    monotonic clock, and handed to the sink with every reading.
    """

    def __init__(self, clock: Callable[[], float] = time.monotonic) -> None:
        self._clock = clock
        self._last: Optional[float] = None
        self._session = uuid.uuid4().hex[:16]
        self._index = 0

    def start(self) -> Optional[float]:
        """Mark a tick's start; return the period since the previous start, or None."""
        now = self._clock()
        period = None if self._last is None else now - self._last
        self._last = now
        self._index += 1
        return period

    def stamp(self) -> Dict[str, Any]:
        """When a replica was read, on this poller's own clock (review S-7, S-8).

        ``session`` names this poller's run: a monotonic reading means nothing
        to another process, so the ledger measures a span on this clock only
        between two stamps of the same session. ``tick`` is the tick's number
        in the session, so two reads one tick apart are recognised as such
        however long the reads inside the tick took.
        """
        return {"session": self._session, "tick": self._index, "monotonic": self._clock()}

    def forget(self) -> None:
        """Nothing is being served: the next tick has no previous one to measure from."""
        self._last = None
        self._session = uuid.uuid4().hex[:16]


def llamacpp_performance(
    counters: Dict[str, float], gauges: Dict[str, float], tick_period: Optional[float],
    stamp: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    """This machine's llama.cpp reading as the ledger's performance input.

    llama.cpp adds a reply's tokens and the seconds spent writing them when the
    reply finishes; their ratio is the writing speed, so they fill the same two
    columns vLLM's inter-token counters fill. It reports no KV-cache occupancy,
    no prefix-cache counters and no per-request histogram, and none is invented.
    """
    mapping = (
        ("itl_intervals", "tokens_predicted_total"),
        ("itl_seconds", "tokens_predicted_seconds_total"),
        ("prompt_seconds", "prompt_seconds_total"),
    )
    return {
        **(stamp or {}), "node_id": CONTROLLER_PLACEMENT_ID, "tick_period": tick_period,
        "gauges": {
            key: gauges[key] for key in ("requests_processing", "requests_deferred") if key in gauges
        },
        "counters": {column: counters[key] for column, key in mapping if key in counters},
        "histogram": None, "created": None,
    }


def vllm_performance(
    node_id: str, port: int, aggregate: Dict[str, float], body: str,
    tick_period: Optional[float], stamp: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    """One vLLM replica's reading as the ledger's performance input."""
    parsed = parse_vllm_performance(body)
    mapping = (
        ("itl_seconds", "tpot_seconds_sum"), ("itl_intervals", "tpot_seconds_count"),
        ("prefix_hits", "prefix_cache_hits_total"), ("prefix_queries", "prefix_cache_queries_total"),
        ("preemptions", "preemptions_total"),
    )
    counters = {column: aggregate[key] for column, key in mapping if key in aggregate}
    sources = parsed["prompt_sources"]
    for column, source in (("prompt_computed", "local_compute"), ("prompt_cached", "local_cache_hit")):
        if source in sources:
            counters[column] = sources[source]
    used = histogram_in_use(parsed)
    histogram = None
    if used is not None:
        family, shape = used
        histogram = {
            "family": family, "edges": shape["edges"], "cumulative": shape["cumulative"],
            "short_replies": parsed["short_replies"],
        }
    return {
        **(stamp or {}), "node_id": str(node_id), "port": int(port), "tick_period": tick_period,
        "gauges": {
            key: aggregate[key]
            for key in ("kv_cache_fraction", "requests_processing", "requests_deferred")
            if key in aggregate
        },
        "counters": counters, "histogram": histogram, "created": parsed["created"],
    }


class WindowBaselines:
    """What a cumulative counter family read last, per source, for the window fields.

    `generation_health` judges a window on what changed during it, so each
    tick writes the increase of a family of counters since the previous tick.
    A family is compared as a whole: if ANY counter in it went backwards, or
    the family's shape changed (a bucket appeared), the source restarted and
    its increase this tick means nothing, so it sits the tick out (``None``)
    and the new reading becomes the baseline - never a negative, never a
    spike. The first read of a source is only a baseline.
    """

    def __init__(self) -> None:
        self._last: Dict[Tuple[str, str], Dict[str, float]] = {}

    def increase(
        self, source: str, family: str, values: Dict[str, float],
    ) -> Optional[Dict[str, float]]:
        key = (str(source), str(family))
        prior = self._last.get(key)
        self._last[key] = dict(values)
        if prior is None or set(prior) != set(values):
            return None
        if any(values[name] < prior[name] for name in values):
            return None
        return {name: values[name] - prior[name] for name in values}

    def forget(self, source: str) -> None:
        """Drop one source's baselines: its engine restarted (review S-11)."""
        for key in [key for key in self._last if key[0] == str(source)]:
            self._last.pop(key, None)

    def keep_only(self, sources: Iterable[str]) -> None:
        """Forget every source not in ``sources`` (a replica gone, nothing serving)."""
        wanted = {str(source) for source in sources}
        for key in [key for key in self._last if key[0] not in wanted]:
            self._last.pop(key, None)


def llamacpp_decode_window(
    baselines: WindowBaselines, endpoint: str, counters: Dict[str, float],
) -> Dict[str, float]:
    """The tokens written and the seconds spent writing them during one tick.

    From llama.cpp's own ``tokens_predicted_total`` and
    ``tokens_predicted_seconds_total``: their ratio over a window is the
    per-answer writing speed. Empty when either counter is missing, on the
    first read, and across a restart.
    """
    tokens = counters.get("tokens_predicted_total")
    seconds = counters.get("tokens_predicted_seconds_total")
    if tokens is None or seconds is None:
        return {}
    change = baselines.increase(endpoint, "decode", {"tokens": tokens, "seconds": seconds})
    if change is None:
        return {}
    return {
        DECODE_TOKENS_FIELD: float(change["tokens"]),
        DECODE_SECONDS_FIELD: float(change["seconds"]),
    }


class ServingMetricsPoller:
    """A background thread that scrapes serving metrics while the engine serves.

    Lifecycle mirrors the runtime's other pollers: :meth:`start` spawns a daemon
    thread, :meth:`stop` asks it to finish. :meth:`poll_once` is the whole tick
    as a pure-ish, synchronous step — it is what the loop calls and what a test
    drives directly.
    """

    def __init__(
        self,
        *,
        endpoint_source: Callable[[], Optional[str]],
        record: Optional[Callable[[Dict[str, float]], bool]],
        interval_seconds: float = SERVING_POLL_SECONDS,
        fetch: Optional[Callable[[str], str]] = None,
        log: Optional[logging.Logger] = None,
        usage_sink: Optional[UsageSink] = None,
        clock: Callable[[], float] = time.monotonic,
    ):
        self._ticks = TickClock(clock)
        self._endpoint_source = endpoint_source
        self._record = record
        self._usage_sink = usage_sink
        self._interval = float(interval_seconds)
        self._fetch = fetch or _http_get
        self._log = log or LOGGER
        self._stop = threading.Event()
        self._thread: Optional[threading.Thread] = None
        self._windows = WindowBaselines()
        #: Mode A writes its starts and stops too (pass-5 review).
        self._targets = TargetTracker(usage_sink, self._log, local=True)

    def poll_once(self) -> bool:
        """One scrape → parse → store step. True only when a sample was stored.

        Returns False — writing nothing — for every honest-degrade case: no
        engine serving (no endpoint), a fetch that fails, a body that carries
        none of the gauges, or no place to record it. Never raises: a failure is
        logged at debug (a not-serving engine is the normal state, not an error
        worth shouting about) and reported as False.
        """
        if self._record is None:
            return False
        try:
            endpoint = self._endpoint_source()
        except Exception as error:  # noqa: BLE001 - discovery must not crash the loop
            self._log.debug("Serving-metrics endpoint discovery failed: %s", error)
            return False
        self._targets.note({endpoint} if endpoint else set())
        if not endpoint:
            self._windows.keep_only(())
            self._ticks.forget()
            return False
        period = self._ticks.start()
        self._windows.keep_only((endpoint,))
        try:
            body = self._fetch(endpoint)
        except Exception as error:  # noqa: BLE001 - a dead endpoint is the normal not-serving case
            self._log.debug("Serving-metrics scrape of %s failed: %s", endpoint, error)
            _feed_miss(self._usage_sink, endpoint, period, self._log)
            return False
        counters = parse_llamacpp_counters(body)
        gauges = parse_llamacpp_metrics(body)
        if not counters:
            # An answer that is not the engine's metrics (an error page, an
            # empty body) is a tick the replica was not read (review S-9).
            _feed_miss(self._usage_sink, endpoint, period, self._log)
        _feed_usage(
            self._usage_sink, endpoint, counters, self._log,
            llamacpp_performance(counters, gauges, period, self._ticks.stamp()),
        )
        if not gauges:
            return False
        # What was written during this tick, for the window's speed verdict.
        gauges.update(llamacpp_decode_window(self._windows, endpoint, counters))
        try:
            return bool(self._record(gauges))
        except Exception as error:  # noqa: BLE001 - a store hiccup must not crash the loop
            self._log.debug("Serving-metrics write failed: %s", error)
            return False

    def _loop(self) -> None:
        # `wait` doubles as the sleep and the stop signal, so a stop is acted on
        # immediately rather than after the full interval. The first scrape waits
        # one interval too — the engine may still be loading right after boot.
        while not self._stop.wait(self._interval):
            self.poll_once()

    def start(self) -> None:
        """Start the poll thread, unless it is already running or has no store.

        With no ``record`` callback there is nowhere to write, so no thread is
        spawned — the same honest-degrade the tick makes, decided once at start
        rather than every ten seconds.
        """
        if self._record is None:
            self._log.info(
                "Serving-metrics poller not started: no telemetry store to write to."
            )
            return
        if self._thread is not None and self._thread.is_alive():
            return
        self._stop.clear()
        self._thread = threading.Thread(
            target=self._loop, name="serving-metrics-poller", daemon=True
        )
        self._thread.start()

    def stop(self) -> None:
        """Ask the poll thread to finish, and wait briefly for it to."""
        self._stop.set()
        thread = self._thread
        if thread is not None:
            thread.join(timeout=self._interval + SERVING_FETCH_TIMEOUT_SECONDS)
        self._thread = None


#: The smallest interval (seconds) a token delta may be divided by. A tick that
#: somehow arrives back-to-back cannot turn a handful of tokens into a spurious
#: spike: below this the rate is reported as zero for that tick instead.
_MIN_RATE_INTERVAL_SECONDS = 1e-3


class VllmServingPoller:
    """Feed the serving card from a vLLM cluster's per-replica ``/metrics``.

    The cluster counterpart of :class:`ServingMetricsPoller`. Where that one GETs
    a single co-located llama.cpp endpoint, this one asks its ``targets_source``
    for the ``(node_id, port)`` vLLM servers behind the current Mode-B deployment
    and reads each through ``scrape_body`` - the SAME controller-loopback GET /
    worker ``python3`` transport the idle watch uses, so this holds no serving or
    bridge lock while it waits on a socket. It parses each body with
    :func:`vaelor.serving_metrics.parse_vllm_serving` and writes ONE aggregate
    sample per tick: the running/waiting queue SUMMED across replicas, the
    output and prompt token THROUGHPUT summed from each replica's own counter
    delta, the per-stream DECODE SPEED from the inter-token-latency histogram's
    delta over the tick, how many replicas were read out of how many there are,
    the KV-cache fraction as the MAX across replicas (a ratio is not additive), a
    pooled TTFT/TPOT average (sum of the histogram sums over the sum of the
    counts, omitted when no observations exist), and the summed success counter.
    It also derives the lifetime prefix-cache hit rate, the summed preemption
    counter, the goodput share within :data:`GOODPUT_BUDGET_SECONDS`, and
    per-model RED (request and error counts and an average end-to-end latency,
    encoded as ``model:<name>:<field>`` gauges the snapshot decodes).

    **Decode speed is not throughput (ACC-051).** Tokens counted over a
    ten-second tick and summed across replicas answer "how much did the cluster
    produce", not "how fast does a stream type": a stream that ran for three of
    the ten seconds, or two replicas each half busy, drag that figure well below
    what a user watching the answer sees. So the tick writes the two separately:
    ``decode_tokens_per_second`` is the inter-token intervals observed during the
    tick divided by the seconds they took (vLLM's ``inter_token_latency_seconds``
    ``_count``/``_sum`` deltas, pooled across replicas) - the rate a stream
    decodes at while it is generating, the same quantity llama.cpp's
    ``predicted_tokens_seconds`` reports - and it is ABSENT when nothing was
    decoded during the tick; ``generation_throughput_tokens_per_second`` and
    ``prompt_throughput_tokens_per_second`` carry the counted aggregate under a
    name that says so. One known limit: under speculative decoding a single
    step can emit several tokens but is observed as one inter-token interval,
    so this formula UNDERSTATES the speed a user sees on such a deployment.

    **A partial sum says it is partial (ACC-053).** ``replicas_total`` and
    ``replicas_read`` ride on every sample, so a tick that could read one replica
    of two is stored as "1 of 2", never as a whole-cluster total.

    **Why an aggregate, not one sample per replica.** The card reads the single
    newest serving row (``serving_store.latest_serving``); writing a row per
    replica would leave it showing whichever replica was scraped last, understating
    a two-node cluster as one node. The whole-engine numbers a user asks the card
    for are the sum of throughput and queue across replicas, so the tick writes
    that one aggregate under the controller's node tag - the same tag and the same
    ``record`` seam the llama.cpp poller uses, so the read side is unchanged.

    **Honest-degrade end to end.** No targets (not serving vLLM), a replica that
    cannot be read, or a body that parses to nothing each contribute nothing and
    never raise; a tick that gathered nothing from any replica writes nothing, so
    the card degrades to its "not collected" state rather than a fabricated zero. A
    replica whose counter went backwards (a server restart) contributes NO rate
    that tick - not a negative, not a spike and not a zero: the interval the new
    reading spans is not known, and ``replicas_read`` already says the row is
    partial (VD-147; it used to contribute a fabricated ``0.0``). A replica seen
    for the first time contributes the queue it reports but no rate yet - there
    is no earlier sample to take a delta against.

    The pooled figures (decode speed, prefix-cache ratio, KV-cache maximum) are
    computed by `serving_buckets`, the same functions the dashboard's buckets
    are read through, so the card and a chart cannot disagree about a tick.
    """

    def __init__(
        self,
        *,
        targets_source: Callable[[], List[Tuple[str, int]]],
        scrape_body: Callable[[str, int], Optional[str]],
        record: Optional[Callable[[Dict[str, float]], bool]],
        interval_seconds: float = SERVING_POLL_SECONDS,
        clock: Callable[[], float] = time.monotonic,
        log: Optional[logging.Logger] = None,
        usage_sink: Optional[UsageSink] = None,
    ):
        self._targets_source = targets_source
        self._scrape_body = scrape_body
        self._record = record
        self._usage_sink = usage_sink
        self._interval = float(interval_seconds)
        self._clock = clock
        self._log = log or LOGGER
        self._stop = threading.Event()
        self._thread: Optional[threading.Thread] = None
        #: node_id -> {counter_key: (value, monotonic_ts)} held across ticks so a
        #: cumulative counter becomes a per-second rate. Cleared whenever nothing
        #: is serving, so a later redeploy starts its rate baseline fresh.
        self._previous: Dict[str, Dict[str, Tuple[float, float]]] = {}
        #: The time-to-first-token family's last reading per replica, for the
        #: window fields `generation_health` sums.
        self._windows = WindowBaselines()
        #: The measured tick period, on this poller's own (monotonic) clock.
        self._ticks = TickClock(clock)
        #: node_id -> the engine's creation stamp at its last read. A new stamp
        #: is a restart even when every counter has already passed its old
        #: value, the same rule the ledger applies (review S-11).
        self._created: Dict[str, float] = {}
        #: The replicas whose counters restarted during the current tick.
        self._restarted: set = set()
        #: Which replicas joined or left the targets, told to the sink.
        self._targets = TargetTracker(usage_sink, self._log, local=False)

    def _delta(
        self, node_id: str, key: str, current: float, now: float
    ) -> Optional[Tuple[float, float]]:
        """``(increase, elapsed seconds)`` of one replica's counter since the last tick.

        ``None`` when there is no earlier sample (the first read of this replica,
        so no honest delta exists yet), and ``None`` when the counter went
        backwards (a restart, read by `serving_metrics.counter_increase`): a
        rate across a restart would divide by time the new process did not run,
        so the replica sits the tick out rather than being counted as zero.
        Always records ``current`` as the new baseline, so the next tick can
        measure against it.
        """
        prior = self._previous.get(node_id, {}).get(key)
        self._previous.setdefault(node_id, {})[key] = (current, now)
        if prior is None:
            return None
        previous_value, previous_ts = prior
        increase, reset = counter_increase(previous_value, current)
        if reset:
            self._restarted.add(node_id)
            return None
        return (increase, now - previous_ts)

    def _rate(self, node_id: str, key: str, current: float, now: float) -> Optional[float]:
        """Tokens/sec for one replica's cumulative counter since the last tick.

        ``None`` when there is no earlier sample, the counter went backwards (a
        restart) or the tick arrived implausibly fast - no measured span, so no
        rate, and never a fabricated ``0.0`` (pass-2 review nit); otherwise the
        non-negative delta over the elapsed seconds.
        """
        delta = self._delta(node_id, key, current, now)
        if delta is None:
            return None
        increase, elapsed = delta
        if elapsed < _MIN_RATE_INTERVAL_SECONDS:
            return None
        return increase / elapsed

    def poll_once(self) -> bool:
        """One scrape-all-replicas -> aggregate -> store step. True only on a write.

        Returns False - writing nothing - for every honest-degrade case: no store,
        nothing serving vLLM, and a tick no replica answered. Never raises.
        """
        if self._record is None:
            return False
        try:
            targets = self._targets_source()
        except Exception as error:  # noqa: BLE001 - discovery must not crash the loop
            self._log.debug("vLLM serving target discovery failed: %s", error)
            return False
        self._targets.note({"%s:%s" % (node, port) for node, port in (targets or [])})
        if not targets:
            self._previous.clear()
            self._created.clear()
            self._windows.keep_only(())
            self._ticks.forget()
            return False
        period = self._ticks.start()
        now = self._clock()
        self._restarted = set()
        # One row per replica for the shared pooled-figure functions: what it
        # decoded this tick, and the cumulative cache counters and gauges it read.
        tick_rows: List[Dict[str, float]] = []
        replica_rows: List[Dict[str, float]] = []
        running = waiting = 0.0
        decode = prefill = 0.0
        have_queue = False
        decode_samples = prefill_samples = 0
        # The inter-token intervals observed during this tick and the seconds
        # they took, pooled across replicas: their ratio is the per-stream
        # decode speed (ACC-051), token-weighted rather than a mean of means.
        itl_seconds = itl_intervals = 0.0
        # Whether any replica gave a decode increase this tick (zero included:
        # read, and nothing written, is a measurement), and the answers that
        # STARTED this tick by time-to-first-token bucket - the window fields.
        decode_window_seen = False
        ttft_window_seen = False
        ttft_window: Dict[str, float] = {}
        # The deep aggregate signals: KV-cache is a fraction (MAX across
        # replicas, never summed); TTFT/TPOT are a pooled average built from
        # the summed histogram sum and count; success is a summed counter.
        ttft_sum = ttft_count = tpot_sum = tpot_count = 0.0
        success_total = 0.0
        success_seen = False
        preemptions = 0.0
        preemptions_seen = False
        goodput_within = goodput_total = 0.0
        goodput_seen = False
        # node -> nothing; per model_name we roll the request/error counters and
        # the e2e sum/count across replicas, then encode one flat gauge per model.
        per_model_totals: Dict[str, Dict[str, float]] = {}
        seen: set = set()
        read_targets: set = set()
        for node_id, port in targets:
            node = str(node_id)
            replica = "%s:%s" % (node, port)
            try:
                body = self._scrape_body(node_id, port)
            except Exception as error:  # noqa: BLE001 - an unreachable replica is skipped
                self._log.debug("vLLM serving scrape of %s:%s did not answer: %s", node, port, error)
                body = None
            if not body:
                # A target that could not be read: counted as attempted time,
                # so the bucket's coverage says so.
                _feed_miss(self._usage_sink, replica, period, self._log)
                continue
            parsed = parse_vllm_serving(body)
            aggregate = parsed.get("aggregate", {})
            if not aggregate or not any(key in aggregate for key in _VLLM_USAGE_KEYS):
                # An answer that is not vLLM's metrics - or one cut short before
                # its usage counters - was not a read (review S-9).
                _feed_miss(self._usage_sink, replica, period, self._log)
                continue
            performance = vllm_performance(node, port, aggregate, body, period, self._ticks.stamp())
            created = performance.get("created")
            if created is not None:
                if self._created.get(node) not in (None, created):
                    # The engine was started again since the last read: every
                    # baseline it had belongs to the old process.
                    self._previous.pop(node, None)
                    self._windows.forget(replica)
                    self._restarted.add(node)
                self._created[node] = created
            _feed_usage(self._usage_sink, replica, aggregate, self._log, performance)
            replica_rows.append(dict(aggregate))
            per_model = parsed.get("models", {})
            seen.add(node)
            read_targets.add((node, port))
            processing = aggregate.get("requests_processing")
            deferred = aggregate.get("requests_deferred")
            if processing is not None:
                running += processing
                have_queue = True
            if deferred is not None:
                waiting += deferred
                have_queue = True
            generation = aggregate.get("generation_tokens_total")
            if generation is not None:
                rate = self._rate(node, "generation_tokens_total", generation, now)
                if rate is not None:
                    decode += rate
                    decode_samples += 1
            prompt = aggregate.get("prompt_tokens_total")
            if prompt is not None:
                rate = self._rate(node, "prompt_tokens_total", prompt, now)
                if rate is not None:
                    prefill += rate
                    prefill_samples += 1
            ttft_series_sum = aggregate.get("ttft_seconds_sum")
            ttft_series_count = aggregate.get("ttft_seconds_count")
            if ttft_series_sum is not None and ttft_series_count is not None:
                ttft_sum += ttft_series_sum
                ttft_count += ttft_series_count
            tpot_series_sum = aggregate.get("tpot_seconds_sum")
            tpot_series_count = aggregate.get("tpot_seconds_count")
            if tpot_series_sum is not None and tpot_series_count is not None:
                tpot_sum += tpot_series_sum
                tpot_count += tpot_series_count
                # The histogram's sum and count are one pair: if EITHER went
                # backwards the server restarted (or the series was reset), and
                # a delta of one against the other would be a meaningless speed.
                # Both baselines are re-taken and the replica sits this tick out.
                prior = self._previous.get(node, {})
                prior_seconds = prior.get("tpot_seconds_sum")
                prior_intervals = prior.get("tpot_seconds_count")
                pair_reset = (
                    (prior_seconds is not None and tpot_series_sum < prior_seconds[0])
                    or (prior_intervals is not None and tpot_series_count < prior_intervals[0])
                )
                seconds_delta = self._delta(node, "tpot_seconds_sum", tpot_series_sum, now)
                intervals_delta = self._delta(node, "tpot_seconds_count", tpot_series_count, now)
                if not pair_reset and seconds_delta is not None and intervals_delta is not None:
                    itl_seconds += seconds_delta[0]
                    itl_intervals += intervals_delta[0]
                    decode_window_seen = True
                    tick_rows.append({
                        "itl_seconds": seconds_delta[0], "itl_intervals": intervals_delta[0],
                    })
            if ttft_series_sum is not None and ttft_series_count is not None:
                family = {
                    key: value for key, value in aggregate.items()
                    if key.startswith(TTFT_BUCKET_KEY_PREFIX)
                }
                family[TTFT_COUNT_FIELD] = ttft_series_count
                family[TTFT_SECONDS_FIELD] = ttft_series_sum
                change = self._windows.increase("%s:%s" % (node, port), "ttft", family)
                if change is not None:
                    ttft_window_seen = True
                    for key, increase in change.items():
                        if key.startswith(TTFT_BUCKET_KEY_PREFIX):
                            key = ttft_bucket_field(float(key[len(TTFT_BUCKET_KEY_PREFIX):]))
                        ttft_window[key] = ttft_window.get(key, 0.0) + increase
            success = aggregate.get("request_success_total")
            if success is not None:
                success_total += success
                success_seen = True
            preempted = aggregate.get("preemptions_total")
            if preempted is not None:
                preemptions += preempted
                preemptions_seen = True
            within = aggregate.get("e2e_within_budget_count")
            e2e_count = aggregate.get("e2e_total_count")
            if within is not None and e2e_count is not None:
                goodput_within += within
                goodput_total += e2e_count
                goodput_seen = True
            for model_name, model_metrics in per_model.items():
                requests = model_metrics.get("request_success_total")
                if requests is None:
                    continue
                bucket = per_model_totals.setdefault(
                    model_name,
                    {"requests": 0.0, "errors": 0.0, "e2e_sum": 0.0, "e2e_count": 0.0},
                )
                bucket["requests"] += requests
                bucket["errors"] += model_metrics.get("request_errors_total", 0.0)
                model_e2e_sum = model_metrics.get("e2e_seconds_sum")
                model_e2e_count = model_metrics.get("e2e_total_count")
                if model_e2e_sum is not None and model_e2e_count is not None:
                    bucket["e2e_sum"] += model_e2e_sum
                    bucket["e2e_count"] += model_e2e_count
        for stale in [node for node in self._previous if node not in seen]:
            self._previous.pop(stale, None)
        self._windows.keep_only("%s:%s" % pair for pair in read_targets)
        if not seen:
            return False
        gauges: Dict[str, float] = {}
        if have_queue:
            gauges["requests_processing"] = running
            gauges["requests_deferred"] = waiting
        if decode_samples:
            gauges["generation_throughput_tokens_per_second"] = decode
        if prefill_samples:
            gauges["prompt_throughput_tokens_per_second"] = prefill
        # Per-stream decode speed only when tokens were actually decoded during
        # the tick: an idle tick has no speed to report, which is an absent key,
        # never a 0 tok/s that reads as a stalled engine.
        decode_speed = pooled_rate(tick_rows, "itl_intervals", "itl_seconds")
        if decode_speed:
            gauges["decode_tokens_per_second"] = decode_speed
        # What changed during the tick, for the window's speed verdict
        # (`generation_health`): written as floats (one field type in the
        # store), zeros included, whenever a replica gave an increase.
        if decode_window_seen:
            gauges[DECODE_TOKENS_FIELD] = float(itl_intervals)
            gauges[DECODE_SECONDS_FIELD] = float(itl_seconds)
        if ttft_window_seen:
            for key, increase in ttft_window.items():
                gauges[key] = float(increase)
        # The deep signals: each is written only when its inputs were present
        # this tick - an absent family is an omitted key, never a written 0. The
        # TTFT/TPOT averages guard divide-by-zero through the count > 0 test, so
        # a window with the histogram declared but no observations shows no
        # average rather than a fabricated one.
        kv_cache = maximum(replica_rows, "kv_cache_fraction")
        if kv_cache is not None:
            gauges["kv_cache_fraction"] = kv_cache
        if ttft_count > 0:
            gauges["ttft_seconds"] = ttft_sum / ttft_count
        if tpot_count > 0:
            gauges["tpot_seconds"] = tpot_sum / tpot_count
        if success_seen:
            gauges["request_success_total"] = success_total
        # The E' deep signals, each written only when its inputs were present:
        # a lifetime prefix-cache hit ratio, the summed preemption counter, and
        # the goodput share of requests finishing within the budget (with the
        # budget itself, so the read side never assumes it).
        prefix_rate = pooled_ratio(
            replica_rows, "prefix_cache_hits_total", "prefix_cache_queries_total",
        )
        if prefix_rate is not None:
            gauges["prefix_cache_hit_rate"] = prefix_rate
        if preemptions_seen:
            gauges["preemptions_total"] = preemptions
        if goodput_seen and goodput_total > 0:
            gauges["goodput_ratio"] = goodput_within / goodput_total
            gauges["goodput_budget_seconds"] = GOODPUT_BUDGET_SECONDS
        # Per-model RED is encoded into the same flat gauge dict the store keeps.
        # Each model with traffic contributes keys of the exact form
        # 'model:<model_name>:requests_total', 'model:<model_name>:errors_total'
        # and 'model:<model_name>:avg_e2e_seconds'. A model_name carries '/' and
        # '-'; the snapshot decodes by splitting the final ':key' off with one
        # rsplit, so the name stays intact whatever it contains.
        for model_name, totals in per_model_totals.items():
            if totals["requests"] <= 0:
                continue
            gauges["model:%s:requests_total" % model_name] = totals["requests"]
            gauges["model:%s:errors_total" % model_name] = totals["errors"]
            if totals["e2e_count"] > 0:
                gauges["model:%s:avg_e2e_seconds" % model_name] = (
                    totals["e2e_sum"] / totals["e2e_count"]
                )
        if not gauges:
            return False
        # Coverage rides on the sample so a partial sum is never read as the
        # whole cluster's (ACC-053): the read side states "1 of 2" from these.
        gauges["replicas_total"] = float(len({(str(node), port) for node, port in targets}))
        gauges["replicas_read"] = float(len(read_targets))
        # A replica read this tick whose engine restarted contributes no rate,
        # so the throughput totals are partial even when every replica answered
        # (review S-10): the read side says so.
        restarted = {node for node in self._restarted if node in seen}
        if restarted:
            gauges["replicas_restarted"] = float(len(restarted))
        try:
            return bool(self._record(gauges))
        except Exception as error:  # noqa: BLE001 - a store hiccup must not crash the loop
            self._log.debug("vLLM serving write failed: %s", error)
            return False

    def _loop(self) -> None:
        while not self._stop.wait(self._interval):
            self.poll_once()

    def start(self) -> None:
        """Start the poll thread, unless it is already running or has no store."""
        if self._record is None:
            self._log.info(
                "vLLM serving-metrics poller not started: no telemetry store to write to."
            )
            return
        if self._thread is not None and self._thread.is_alive():
            return
        self._stop.clear()
        self._thread = threading.Thread(
            target=self._loop, name="vllm-serving-metrics-poller", daemon=True
        )
        self._thread.start()

    def stop(self) -> None:
        """Ask the poll thread to finish, and wait briefly for it to."""
        self._stop.set()
        thread = self._thread
        if thread is not None:
            thread.join(timeout=self._interval + SERVING_FETCH_TIMEOUT_SECONDS)
        self._thread = None
