"""Telemetry retention: the store Vaelor owns, and the one error it raises.

VD-095 / task #187. Retention had never run under Vaelor. Several separate
breaks each explained the symptom on their own, and every one of them survived
for the same reason: **this path had never returned a row**, so a test that
mocks the store into success passes against all of them at once.

This module exists so they cannot recur independently.

* `TELEMETRY_STORE_NAME` is the only place the store is named, so the name the
  reader opens and the name the writer creates cannot drift apart.
* `start_retention` is a start path that the deployment actually reaches.
  `ControlPlaneRuntime.run()` is the legacy Pironman entrypoint and no Vaelor
  process constructs the class that calls it, so `Database.start()` had never
  executed.
* `TelemetryStoreError` is the single exception a caller catches, raised for
  every store failure. `InfluxDBClientError` inherits straight from `Exception`
  - its MRO is `[InfluxDBClientError, Exception, BaseException, object]` - so
  the caller's `(AttributeError, OSError, TypeError, ValueError)` tuple could
  never catch the one failure this path actually produces. Lengthening that
  tuple was the alternative repair and it leaves the *next* store exception
  equally unreportable.
* `history_samples` returns rows oldest to newest, which is what the note handed
  to the model already claimed while the query said `ORDER BY time DESC`.

Readiness deliberately does **not** live here. "Can this database be read" is
the store's own question, so it is `Database.unavailable_reason()`: the old
`is_ready()` pinged the InfluxDB *server* and never the database, so the check
passed and the query then failed.

**`RetentionState` is the part to read if you read nothing else.** Retention has
four states and they are four different sentences for the owner, with four
different repairs. Two review rounds each found the same class of defect - two
of them collapsed into one - and each collapse reproduced the exact complaint
this work exists to remove, "history is not retained on this appliance", while
retention was switched on:

* `STARTING` - the control plane is up and retention has not finished starting.
  Real and recurring: `server.py` calls `serve_forever()` immediately after
  spawning the thread, so Flask answers requests throughout a window with a
  floor of `Database.start()`'s unconditional two-second sleep. Every restart
  reopens it. This is the module-level default, because at import time nothing
  has started, and defaulting it to "off" is what made the window invisible.
* `OFF` - the owner switched retention off. Nothing is wrong.
* `FAILED` - retention tried and could not start; `reason` says why. The
  appliance configuration is read over the hardware bridge's Unix socket, and
  `RuntimeHardware._read` answers `{}` - not an error - while that socket is
  unbound. `vaelor-hardware-bridge.service` is `Type=simple`, which systemd
  calls active the moment it forks, so `After=` orders the bridge's start and
  not its readiness. Reading `enable_history` out of `{}` gave `False`, so a
  transient ordering race became a permanent silent stop. That is LESSONS
  pattern 8: the observer manufactured the absence.
* `RUNNING` - the store is open.

So `telemetry_settings` **raises** rather than returning a value it could not
read: no `TelemetrySettings` instance can mean "unknown". `resolve_settings`
retries within a stated ceiling because the race is short. And `history_source`
takes the state explicitly, so there is no combination of module defaults that
reads as an owner setting.
"""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass
from enum import Enum
from typing import Any, Callable, Dict, List, Optional

LOGGER = logging.getLogger(__name__)

#: The telemetry store is named `vaelor`, decided by the owner on 2026-08-13
#: (VD-095). `pironman5-max` is a Pironman-era name and the compatibility
#: boundary in `CLAUDE.md` forbids a new dependency on a legacy name in a public
#: read path. The rows that database still holds are abandoned, not migrated.
TELEMETRY_STORE_NAME = "vaelor"

#: Retention period, in days. **Seven, decided in VD-089 on 2026-08-11.**
#:
#: This is not a fallback that happens to be seven. VD-089 recorded that thirty
#: "was never chosen" - it is a Pironman-era value that can persist in the
#: legacy `/opt/pironman5/config.json`, not a number any owner set. A fresh
#: Vaelor install never writes it: `VaelorControlPlane.__init__` fills an absent
#: `database_retention_days` with `DEFAULT_RETENTION_DAYS` (7). Only a box
#: upgraded over a legacy Pironman config still carries the 30, and there it is
#: read but not obeyed - the store applies seven regardless. (The docstrings
#: here once claimed 30 was `control_plane.py`'s own default; the code default
#: became 7 and this text did not, until 2026-08-16 - LESSONS 6, a reason that
#: drifted from the code it explained.)
#:
#: A legacy `database_retention_days: 30` is therefore deliberately **not** the
#: operative value; see `telemetry_settings`. Owner visibility and control over
#: the period is task #186 and is not built here.
DEFAULT_RETENTION_DAYS = 7

#: Sampling period used when the configuration does not carry `data_interval`,
#: in seconds.
DEFAULT_DATA_INTERVAL = 1

#: Most rows a single history read may ask the store for.
MAX_HISTORY_SAMPLES = 120

#: Most buckets a single *time-ranged* history read may return, whatever window
#: is asked for. 168 is seven days at one bucket an hour - the retention period
#: at its coarsest useful resolution. The cap exists because a small model reads
#: this payload: seven days of one-second samples is ~600,000 rows and would
#: bury the answer, so the window is always downsampled to at most this many
#: buckets.
MAX_HISTORY_BUCKETS = 168

#: The finest bucket a ranged read will use, in seconds. A short window divided
#: by `MAX_HISTORY_BUCKETS` can ask for sub-second buckets, finer than the
#: sampling interval and only multiplying near-identical rows; this floors it at
#: the default sampling period.
MIN_BUCKET_SECONDS = DEFAULT_DATA_INTERVAL

#: The longest window a ranged read will honour, in seconds - seven days, the
#: retention period. A longer span can only return what the store kept, so it is
#: clamped here rather than querying time the store cannot cover.
MAX_HISTORY_WINDOW_SECONDS = DEFAULT_RETENTION_DAYS * 24 * 60 * 60

#: The measurement the data logger writes and the history reader reads. One
#: name, because two spellings of it is how a read path silently finds nothing.
HISTORY_MEASUREMENT = "history"

#: The retention policy this store manages. InfluxDB's own `autogen` policy is
#: created with `create_database` and never expires, so a named policy of our
#: own is what actually enforces a period.
RETENTION_POLICY_NAME = "default_policy"

#: How many times the configuration read is retried, and how long between
#: attempts, before retention gives up and says so. The hardware bridge binds
#: its socket in well under a second; this ceiling is generous for that race and
#: bounded so a bridge that never arrives produces a reported failure rather
#: than a thread waiting for ever. Roughly 30 seconds in total.
#:
#: Read inside `resolve_settings` rather than bound as its default arguments,
#: so patching either of these reaches the production call. Bound as defaults
#: they were unreachable, and no test could exercise the real values without
#: spending 30 real seconds.
CONFIG_READ_ATTEMPTS = 30
CONFIG_READ_RETRY_SECONDS = 1.0


class TelemetryStoreError(RuntimeError):
    """Any failure to start, reach, or read the telemetry store.

    One error for the whole store layer, so a caller has exactly one thing to
    catch and its `reason` field is always reachable.
    """


class RetentionState(Enum):
    """Which of the four retention states this process is in.

    Four, not two. Collapsing any pair of these produces a sentence that is
    false about the appliance, and the two collapses already found both
    produced the *same* false sentence - that history is not retained.
    """

    #: Nothing has started yet. The module-level default, on purpose.
    STARTING = "starting"
    #: The owner switched retention off.
    OFF = "off"
    #: Retention tried to start and could not; a reason travels with it.
    FAILED = "failed"
    #: The store is open and the writer is running.
    RUNNING = "running"


@dataclass(frozen=True)
class TelemetrySettings:
    """What the appliance configuration says about retention.

    Only ever describes a configuration that was genuinely read; see
    `telemetry_settings`.
    """

    enabled: bool
    retention_days: int
    interval: float
    #: What `database_retention_days` said, when it said anything. Recorded so
    #: an override can be logged rather than applied silently, and never used
    #: as the store's period.
    configured_retention_days: Optional[int] = None


@dataclass
class TelemetryRetention:
    """A started store and the writer filling it, so both stop together."""

    database: Any
    data_logger: Any

    def stop(self) -> None:
        self.data_logger.stop()
        self.database.close()


def _positive_int(value: Any, fallback: Optional[int]) -> Optional[int]:
    try:
        number = int(value)
    except (TypeError, ValueError):
        return fallback
    return number if number > 0 else fallback


def _positive_number(value: Any, fallback: float) -> float:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return fallback
    return number if number > 0 else fallback


def telemetry_settings(config: Any) -> TelemetrySettings:
    """Read the retention settings out of a control-plane config.

    `read_config()` returns `{"system": {...}}` on this appliance, the same
    shape `VaelorControlPlane.__init__` reads, so both take the settings from
    one place rather than two.

    **Raises `TelemetryStoreError` when there is no `system` object to read.**
    That is the whole point of this function's shape: a configuration that could
    not be fetched must not arrive looking like a configuration that says
    retention is off. `RuntimeHardware._read` returns `{}` when the hardware
    bridge socket is not bound, and returning `enabled=False` for that made a
    transient startup race permanently indistinguishable from an owner setting.

    **`database_retention_days` is read but not obeyed**, and that is a
    deliberate, recorded choice rather than an oversight. A box upgraded over a
    legacy Pironman config can carry 30 there; VD-089 decided 7 and recorded
    that 30 was never chosen by anyone. A fresh install does not carry it at all:
    `VaelorControlPlane.__init__` writes `DEFAULT_RETENTION_DAYS` (7) when the
    key is absent. Honouring a legacy 30 would ship a ledger that says 7 against
    an appliance doing 30.
    It is returned as `configured_retention_days` so the override can be logged
    where an operator will see it, and giving the owner a real choice of period
    is task #186.
    """
    system: Optional[Dict[str, Any]] = None
    if isinstance(config, dict):
        candidate = config.get("system")
        if isinstance(candidate, dict):
            system = candidate
    if system is None:
        raise TelemetryStoreError(
            "the appliance configuration carries no system object, so whether "
            "telemetry history is switched on could not be determined"
        )
    return TelemetrySettings(
        enabled=bool(system.get("enable_history", False)),
        retention_days=DEFAULT_RETENTION_DAYS,
        interval=_positive_number(system.get("data_interval"), DEFAULT_DATA_INTERVAL),
        configured_retention_days=_positive_int(
            system.get("database_retention_days"), None
        ),
    )


def vaelor_managed_settings() -> TelemetrySettings:
    """Retention settings for a machine with no Pironman HAT (VD-095 / VD-096).

    On an x86 workstation there is no HAT, so no `/opt/pironman5/config.json`
    exists to carry a `system.enable_history` flag - that object lives only in
    the HAT's config on a Pi. The telemetry store is Vaelor's own InfluxDB, and
    `VaelorControlPlane.__init__` already writes `DEFAULT_RETENTION_DAYS` on a
    fresh install, so on a HAT-less host retention is Vaelor-managed and on by
    default rather than gated on a flag that can never be present.

    Returned as a genuine, fully-read `TelemetrySettings` - not a fallback for a
    read that failed - so it carries no `configured_retention_days`, because
    there is no HAT configuration asking for a different period to override.
    """
    return TelemetrySettings(
        enabled=True,
        retention_days=DEFAULT_RETENTION_DAYS,
        interval=DEFAULT_DATA_INTERVAL,
        configured_retention_days=None,
    )


def resolve_settings(
    read_config: Callable[[], Any],
    attempts: Optional[int] = None,
    delay: Optional[float] = None,
    sleep: Optional[Callable[[float], None]] = None,
    log: Optional[logging.Logger] = None,
    hat_config_present: Optional[Callable[[], bool]] = None,
) -> TelemetrySettings:
    """The retention settings, waiting a bounded while for the bridge to answer.

    The control plane can reach this before `vaelor-hardware-bridge.service` has
    bound its socket, and `After=` on a `Type=simple` unit is not a readiness
    barrier. The window is short, so retrying is the proportionate repair; the
    ceiling is stated so a bridge that never arrives is reported rather than
    waited on for ever.

    **`hat_config_present` separates "no HAT" from "the HAT has not answered
    yet", which are a permanent state and a transient one with opposite repairs
    (VD-095 / VD-096).** On an x86 workstation the bridge is up and its socket is
    bound, but there is no Pironman HAT, so `read_config()` returns `{}` for ever
    - there is no `system.enable_history` flag to wait for. Retrying that thirty
    times and then reporting "the hardware bridge may not have bound its socket"
    described a socket race that is not happening and left retention permanently
    `FAILED` on a machine whose store was fine. When the caller supplies this
    signal and it reports the HAT config source absent, retention is
    Vaelor-managed and resolves immediately to `vaelor_managed_settings()`; the
    signal is whether that config source EXISTS, not whether the bridge answered,
    because only the first tells the permanent case from the Pi boot race below.
    A caller that supplies nothing keeps the historic behaviour - the retry loop
    - so the Pi path is untouched and only the caller that knows the hardware
    (the control plane) short-circuits it.

    The module constants are read here rather than bound as default arguments,
    so patching them reaches this call from production rather than only from a
    test that passes them explicitly.

    Raises `TelemetryStoreError` naming the ceiling when it runs out. The
    sentence leads with what an operator can act on, because the reason is
    trimmed for the model further down and a truncated tail costs the useful
    half.
    """
    attempts = CONFIG_READ_ATTEMPTS if attempts is None else attempts
    delay = CONFIG_READ_RETRY_SECONDS if delay is None else delay
    sleep = time.sleep if sleep is None else sleep
    log = log or LOGGER
    # No Pironman HAT on this machine: there is no `system.enable_history` flag
    # anywhere to wait for, so retrying and then blaming an unbound socket would
    # be describing a race that cannot happen here. This is permanent, unlike the
    # transient Pi boot race the loop below handles, and the two are told apart
    # by whether the HAT config source exists - not by whether the bridge
    # answered, which is `{}` in both cases (VD-095 / VD-096).
    if hat_config_present is not None and not hat_config_present():
        log.info(
            "No Pironman HAT configuration on this machine; telemetry retention "
            "is Vaelor-managed and enabled with the %s day default.",
            DEFAULT_RETENTION_DAYS,
        )
        return vaelor_managed_settings()
    ceiling = max(1, attempts)
    last: Any = None
    for attempt in range(1, ceiling + 1):
        try:
            return telemetry_settings(read_config())
        except TelemetryStoreError as error:
            last = error
        except Exception as error:  # converted, never discarded: reported below
            last = TelemetryStoreError(
                "the appliance configuration could not be read: {}".format(
                    " ".join(str(error).split())
                )
            )
        if attempt == 1:
            log.info(
                "Telemetry retention is waiting for the appliance configuration: %s",
                last,
            )
        if attempt < ceiling:
            sleep(delay)
    raise TelemetryStoreError(
        "the appliance configuration could not be read, so whether history is "
        "switched on is unknown; the hardware bridge may not have bound its "
        "socket ({} attempts over about {:.0f}s)".format(ceiling, ceiling * delay)
    )


def not_started(reason: Any) -> TelemetryStoreError:
    """The error the read path raises when retention tried and failed.

    Distinct from `None` - the owner switched retention off - and from a store
    that exists but cannot be read. Different states, different sentences,
    because they have different repairs, and collapsing any two of them is how
    an appliance came to say it retained nothing while retention was on.
    """
    return TelemetryStoreError(
        "Telemetry retention is not running on this appliance: {}".format(reason)
    )


def still_starting() -> TelemetryStoreError:
    """The error for the boot window, which is a real and recurring state.

    Not an error the owner needs to act on - it says "ask again shortly" - but
    it must not be reported as retention being switched off, because that sends
    them to a setting that is already correct.
    """
    return TelemetryStoreError(
        "Telemetry retention has not finished starting since the control plane "
        "last restarted, so the stored history cannot be read yet. It is not "
        "switched off; try again in a moment."
    )


def history_source(
    database: Any, state: RetentionState, reason: Optional[str] = None
) -> Any:
    """Which store the history read should open, given the retention state.

    Four states in, four outcomes out. It takes the state explicitly rather
    than inferring it from `database is None` and an `enabled` flag, because
    that inference is what made the boot window indistinguishable from an owner
    switching retention off - both are "no store and not enabled".

    It lives here rather than inline in `control_plane` so it can be exercised
    without importing Flask, and so the decision has one home.
    """
    if state is RetentionState.STARTING:
        raise still_starting()
    if state is RetentionState.FAILED:
        raise not_started(reason)
    if state is RetentionState.OFF:
        return None
    return database


def describe_retention(
    state: RetentionState,
    settings: Optional[TelemetrySettings],
    failure: Optional[str] = None,
) -> Dict[str, Any]:
    """The configured-vs-applied retention period and state, for /api/v2 (#186).

    **The override was invisible for two weeks (VD-089 / #186).** A legacy
    appliance configuration can carry `database_retention_days: 30`; the store
    applies the `DEFAULT_RETENTION_DAYS` (7) decided in VD-089. That override was logged once
    at boot and shown on no surface, so an owner reading their own 30 had no way
    to learn that 7 was in force. This states both, which one applies, and the
    retention state, so "off", "still starting" and "failed" read as the four
    different things they are rather than collapsing to one (LESSONS 4/11).

    It lives here rather than in `control_plane` so it can be exercised without
    importing Flask, exactly as `history_source` is - the same reason the whole
    `RetentionState` decision does.

    `settings` is None until the configuration has been read at least once, so
    `configured_retention_days` is honestly None in the `STARTING` and
    early-`FAILED` states rather than a guessed number.

    **The applied period is read from the settings the store was built with, not
    from `DEFAULT_RETENTION_DAYS` restated here (LESSONS 6).** `build_store` hands
    `settings.retention_days` to the `Database`, so that field *is* the enforced
    period; re-deriving it from the module default would be a second mechanism
    answering "what is applied", equal today only by construction and a silent
    lie the moment the period becomes configurable (task #186 itself). Before the
    configuration is read the store has enforced nothing, so the default constant
    is the honest best estimate for that window alone.
    """
    applied = (
        settings.retention_days if settings is not None else DEFAULT_RETENTION_DAYS
    )
    configured = (
        settings.configured_retention_days if settings is not None else None
    )
    override_active = configured is not None and configured != applied
    if state is RetentionState.OFF:
        reason = (
            "Telemetry history retention is switched off in the appliance "
            "configuration."
        )
    elif state is RetentionState.STARTING:
        reason = (
            "Telemetry retention has not finished starting since the control "
            "plane last restarted; ask again shortly."
        )
    elif state is RetentionState.FAILED:
        reason = failure or "Telemetry retention tried to start and could not."
    else:
        reason = "Telemetry history is retained for {} days.".format(applied)
    if override_active:
        reason = (
            "{} The appliance configuration asks for {} days; Vaelor applies "
            "the {} days decided in VD-089, and choosing the period is task #186."
        ).format(reason, configured, applied)
    return {
        "state": state.value,
        "applied_retention_days": applied,
        "configured_retention_days": configured,
        "override_active": override_active,
        "reason": reason,
        # Only meaningful in FAILED; None elsewhere so a reader never shows a
        # stale failure beside a healthy state.
        "failure": failure if state is RetentionState.FAILED else None,
    }


def build_store(
    settings: TelemetrySettings,
    log: Optional[logging.Logger] = None,
    database_factory: Optional[Callable[..., Any]] = None,
) -> Any:
    """Construct the store object without touching InfluxDB.

    Construction is separated from `start_retention` so a caller can publish the
    store before starting it. That is what lets a failed start be reported as
    "switched on and unreadable" rather than "switched off" - two different
    sentences with two different repairs.
    """
    log = log or LOGGER
    if (
        settings.configured_retention_days is not None
        and settings.configured_retention_days != settings.retention_days
    ):
        # Logged rather than applied, and logged rather than silent: an operator
        # who did hand-edit that key deserves to see it being overridden.
        log.warning(
            "Appliance configuration asks for %s day telemetry retention; "
            "applying the %s days decided in VD-089 instead. Choosing the "
            "period is task #186.",
            settings.configured_retention_days,
            settings.retention_days,
        )
    if database_factory is None:
        from .database import Database

        database_factory = Database
    return database_factory(
        TELEMETRY_STORE_NAME,
        log=log,
        retention_days=settings.retention_days,
    )


def start_retention(
    database: Any,
    settings: TelemetrySettings,
    read_data: Callable[[], Dict[str, Any]],
    log: Optional[logging.Logger] = None,
    data_logger_factory: Optional[Callable[..., Any]] = None,
) -> TelemetryRetention:
    """Create the database, apply its retention policy, and start writing.

    Raises `TelemetryStoreError` when the store cannot be brought up. The caller
    decides whether that is fatal; under Flask it is not, because an appliance
    whose InfluxDB is down must still serve everything else.

    **The writer is asked to prove it can write, then that it started.**
    `DataLogger.start`, `loop` and `get_data` are all wrapped in `@log_error`,
    which catches `Exception`, logs it and returns `None`, so without these
    checks a `TelemetryRetention` comes back looking healthy while nothing is
    writing - and a readable, empty store reads to a model as "the appliance has
    no record of last night", the original complaint arriving from a
    "successful" start.

    An earlier version of this docstring declined the write proof on the grounds
    that it costs a sampling interval. **That was wrong**: one synchronous
    `database.set` round-trips the entire chain - read the machine, shape the
    row, hand it to InfluxDB against the selected database - in milliseconds,
    and the row it writes is a real first sample rather than a probe, so it also
    shortens the window in which the store is legitimately empty.

    An empty first *reading* is deliberately **not** fatal. "Nothing to record
    yet" is not "cannot record", and because `FAILED` is terminal until the
    process restarts, turning a transient empty read into permanent death is the
    worse defect. It is logged, the writer starts anyway and retries every
    interval, and `metrics_history` reports an empty store honestly.
    """
    log = log or LOGGER
    if data_logger_factory is None:
        from .data_logger import DataLogger

        data_logger_factory = DataLogger
    try:
        database.start()
    except Exception as error:  # converted, never discarded: the caller reports it
        raise TelemetryStoreError(
            "the telemetry store '{}' could not be started: {}".format(
                TELEMETRY_STORE_NAME, " ".join(str(error).split())
            )
        ) from error
    reason = database.unavailable_reason()
    if reason is not None:
        raise TelemetryStoreError(
            "the telemetry store '{}' did not come up: {}".format(
                TELEMETRY_STORE_NAME, reason
            )
        )
    data_logger = data_logger_factory(
        database=database, interval=settings.interval, log=log
    )
    data_logger.set_read_data(read_data)
    _prove_the_writer_can_write(database, data_logger, log)
    data_logger.start()
    _require_running_writer(data_logger)
    log.info(
        "Telemetry retention started: store '%s', %s day retention, %ss interval.",
        TELEMETRY_STORE_NAME,
        settings.retention_days,
        settings.interval,
    )
    return TelemetryRetention(database=database, data_logger=data_logger)


def _prove_the_writer_can_write(
    database: Any, data_logger: Any, log: logging.Logger
) -> None:
    """Write one real sample synchronously, so the whole chain is proved.

    Cheap - one round-trip - and the row is a legitimate first sample, not a
    probe to be cleaned up. It is the only check here that exercises
    `read_data`, the row shaping and the store's acceptance of a write against
    the selected database together.
    """
    sample = data_logger.get_data()
    if not sample:
        log.warning(
            "Telemetry retention has nothing to record yet: the machine reading "
            "came back empty. The writer will retry every interval."
        )
        return
    try:
        accepted, detail = database.set(HISTORY_MEASUREMENT, sample)
    except Exception as error:  # converted, never discarded: reported to the caller
        # `Database.set` is not total: its `InfluxDBClientError` handler does
        # `json.loads(e.content)`, which raises again when the server's body is
        # not JSON, and an exception raised inside an `except` clause is not
        # caught by the later one. So this cannot assume a tuple comes back.
        raise TelemetryStoreError(
            "the telemetry store could not be written to, so retention would "
            "have recorded nothing: {}".format(" ".join(str(error).split()))
        ) from error
    if not accepted:
        raise TelemetryStoreError(
            "the telemetry store rejected the first sample, so retention would "
            "have recorded nothing: {}".format(" ".join(str(detail).split()))
        )


def _require_running_writer(data_logger: Any) -> None:
    """Refuse to call retention started when the writer is not running."""
    if not getattr(data_logger, "running", False):
        raise TelemetryStoreError(
            "the telemetry writer did not start, so nothing would be recorded; "
            "its failure was swallowed by @log_error and logged"
        )
    thread = getattr(data_logger, "thread", None)
    if thread is None or not thread.is_alive():
        raise TelemetryStoreError(
            "the telemetry writer reports running but has no live thread, so "
            "nothing would be recorded"
        )


def _switched_off_error() -> TelemetryStoreError:
    """Retention is off, so only the live sample exists. One home for the
    sentence both the count read and the ranged read raise, so they cannot
    drift into two spellings of the same state.
    """
    return TelemetryStoreError(
        "Telemetry history retention is switched off on this appliance, so "
        "only the current sample exists."
    )


def _store_read_error(error: Any) -> TelemetryStoreError:
    """The store raised while being read. Shared by both read paths so the
    sentence the owner sees is identical whichever one hit it.
    """
    return TelemetryStoreError(
        "the telemetry store '{}' could not be read: {}".format(
            TELEMETRY_STORE_NAME, " ".join(str(error).split())
        )
    )


def _on_but_unreadable_error(reason: Any) -> TelemetryStoreError:
    """Retention is on but the store cannot be read - a different state from
    off, and the same sentence for both read paths.
    """
    return TelemetryStoreError(
        "Telemetry history is switched on, but the store cannot be read: "
        "{}".format(reason)
    )


def history_samples(database: Any, limit: int = 30) -> List[Dict[str, Any]]:
    """Recent retained telemetry rows, **oldest to newest**, or a stated reason.

    The ordering is the contract `metrics_history` hands the model in its note.
    It is satisfied here rather than left to the caller, because a caller that
    forgets reads every trend backwards and nothing looks wrong.

    Every way this can fail raises `TelemetryStoreError`, including retention
    being switched off, so the caller has one thing to catch and always has a
    sentence to show the owner.
    """
    if database is None:
        raise _switched_off_error()
    rows: Any = []
    try:
        reason = database.unavailable_reason()
        if reason is None:
            rows = database.get(HISTORY_MEASUREMENT, n=_row_count(limit))
    except Exception as error:  # converted, never discarded: the caller reports it
        raise _store_read_error(error) from error
    if reason is not None:
        raise _on_but_unreadable_error(reason)
    if rows is None:
        return []
    if isinstance(rows, dict):
        return [rows]
    return list(rows)


def _row_count(limit: Any) -> int:
    try:
        requested = int(limit)
    except (TypeError, ValueError):
        requested = 30
    return max(1, min(requested, MAX_HISTORY_SAMPLES))


def clamp_window_seconds(since_seconds: Any) -> int:
    """A requested window narrowed to what this store can actually answer.

    One place decides the bounds so the tool layer and the store layer cannot
    disagree about them. Below a bucket the answer is a single point; above the
    retention period the extra span is empty, so it is clamped to the seven days
    the store keeps rather than queried.
    """
    try:
        seconds = int(since_seconds)
    except (TypeError, ValueError):
        seconds = MIN_BUCKET_SECONDS
    return max(MIN_BUCKET_SECONDS, min(seconds, MAX_HISTORY_WINDOW_SECONDS))


def _bucket_seconds(window_seconds: int, max_points: int) -> int:
    """The coarsest bucket that keeps a window within `max_points` buckets.

    Ceiling division, not floor: a floor can leave `window / bucket` one bucket
    over the cap (23h59m of data at a 24-bucket-wide bucket is 25 rows), and the
    cap is the whole point. Floored at `MIN_BUCKET_SECONDS` so a short window is
    not sliced finer than the sampling interval.
    """
    points = max(1, min(int(max_points), MAX_HISTORY_BUCKETS))
    bucket = -(-window_seconds // points)  # ceil(window / points)
    return max(MIN_BUCKET_SECONDS, bucket)


def history_range(
    database: Any,
    since_seconds: Any,
    max_points: Optional[int] = None,
    function: str = "mean",
) -> Dict[str, Any]:
    """Downsampled telemetry over a *time window*, oldest to newest, or a reason.

    The count-based `history_samples` answers "the newest N rows"; this answers
    "the last H hours", which is the question a trend actually is. A window is
    downsampled to at most `MAX_HISTORY_BUCKETS` buckets - `database.get_trend`
    applies `function` to each bucket - so a seven-day pull is ~168 rows rather
    than the ~600,000 one-second samples it covers, small enough to hand a model.

    Every way it can fail raises `TelemetryStoreError`, exactly as
    `history_samples` does and for the same reason: the caller has one thing to
    catch and always has a sentence to show the owner. A `None` database is
    retention switched off, not an error to swallow. The four retention states
    are already resolved by `history_source` before this is called, so this need
    only handle "switched off" and "on but unreadable".

    Returns the buckets alongside the window and bucket size actually used, so
    the caller can tell the model the resolution it is reading ("hourly means
    over the last 24 hours") rather than leaving it to guess from row spacing.
    """
    if database is None:
        raise _switched_off_error()
    window = clamp_window_seconds(since_seconds)
    points = MAX_HISTORY_BUCKETS if max_points is None else max_points
    bucket = _bucket_seconds(window, points)
    rows: Any = []
    try:
        reason = database.unavailable_reason()
        if reason is None:
            rows = database.get_trend(
                HISTORY_MEASUREMENT, window, bucket, function=function
            )
    except Exception as error:  # converted, never discarded: the caller reports it
        raise _store_read_error(error) from error
    if reason is not None:
        raise _on_but_unreadable_error(reason)
    return {
        "window_seconds": window,
        "bucket_seconds": bucket,
        "function": function,
        "buckets": [] if rows is None else list(rows),
    }
