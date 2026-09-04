import sys
import threading
import logging
import os
from os import path

import flask
from flask import redirect, request
from flask_cors import cross_origin
from importlib.resources import files as resource_files
from .wsgi_server import ControlPlaneServer

from .data_logger import DataLogger
from .database import Database
from .telemetry_store import (
    DEFAULT_RETENTION_DAYS,
    TELEMETRY_STORE_NAME,
    RetentionState,
    TelemetryStoreError,
    build_store,
    describe_retention,
    history_range,
    history_samples,
    history_source,
    resolve_settings,
    start_retention,
)
from .utils import log_error
from .runtime_paths import LOG_ROOT, env_value
from .legacy_v1_routes import API_PREFIX, register_legacy_v1_routes
from .power_runtime import build_host_power_controller
from .hardware_runtime import RuntimeHardware
from .hardware_bridge import PIRONMAN_CONFIG
# One home for the v2 cache-control strings. The routes in frontend_routes.py set
# these per response; the after_request block below only supplies them as a
# fallback, so both must name the same values - importing avoids the #98 drift of
# an immutable/no-cache string written verbatim in two modules.
from .frontend_routes import (
    IMMUTABLE_ASSET_CACHE,
    NO_STORE_CACHE,
    REVALIDATE_HTML_CACHE,
)

AVAILABLE_OLED_PAGES = []

__package_name__ = __name__.split('.')[0]
__log_path__ = str(LOG_ROOT)
__www_v2_path__ = str(resource_files(__package_name__).joinpath('www_v2'))
__api_prefix__ = API_PREFIX
__host__ = '0.0.0.0'
__port__ = 34001
__log__ = None
__restart_service__ = lambda: None

__db__ = None
__data_logger__ = None
__telemetry_retention__ = None
__telemetry_retention_lock__ = threading.Lock()
#: Which of the four retention states this process is in. `STARTING` is the
#: right default: at import time nothing has started, and Flask begins
#: answering requests before it has. Defaulting this to "off" is what made the
#: boot window - every restart - read to the owner as an owner setting.
__telemetry_state__ = RetentionState.STARTING
#: Why retention failed, when it did. Read only in the `FAILED` state.
__telemetry_start_failure__ = None
#: The retention settings once they were read, so the /api/v2 retention status
#: can show the configured period beside the applied one (#186). None until the
#: configuration has been read at least once - the `STARTING`/`FAILED`-early
#: states, where the configured period is genuinely not yet known.
__telemetry_settings__ = None
__app__ = flask.Flask(__name__, static_folder=__www_v2_path__)

__app__.logger.setLevel(logging.WARN)
logging.getLogger('werkzeug').setLevel(logging.ERROR)

# The control plane is same-origin. Do not grant arbitrary websites access to
# authenticated appliance responses.
__cors__ = None
__device_info__ = {}
__enable_history__ = False

__read_data__ = lambda: {}
__get_ip_data__ = lambda: {}
__read_config__ = lambda: {}
__on_config_changed__ = lambda config: None
__test_smtp__ = lambda: False
__play_pipower5_buzzer__ = lambda: None

# Native Vaelor installations keep the original, root-owned Pironman hardware
# runtime behind a narrow Unix socket. Resolve its availability on every
# request because systemd may start the control plane before the bridge socket.
# Legacy embedded launches replace these callbacks through the compatibility
# PMDashboard class.
# setters below.
__runtime_hardware__ = RuntimeHardware()
__hardware_bridge__ = __runtime_hardware__.bridge
__read_data__ = __runtime_hardware__.current_data
__get_ip_data__ = __runtime_hardware__.ip_data
__read_config__ = __runtime_hardware__.read_config
__on_config_changed__ = __runtime_hardware__.update_config

# Host power resolves through the privileged bridge when it is running, and
# otherwise through the platform driver. Which mechanism that is - SunFounder's
# sequenced shutdown or systemd-logind - is the driver's decision, not this
# module's.
__power_controller__ = build_host_power_controller(
    hardware_bridge=__hardware_bridge__,
    restart_service=lambda: __restart_service__(),
)


def __shutdown__():
    return __power_controller__.execute("shutdown")


def __reboot__():
    return __power_controller__.execute("reboot")


# Host dashboard page
@__app__.route('/')
@cross_origin()
def dashboard():
    return redirect('/v2/', code=302)

# Host dashboard css
@__app__.route('/index.css')
@cross_origin()
def dashboard_css():
    return redirect('/v2/', code=302)

# Host favicon
@__app__.route('/favicon.ico')
@cross_origin()
def favicon():
    return redirect('/v2/', code=302)

# Host static files for dashboard page
@__app__.route('/static/<path:path>')
@cross_origin()
def serve_static(path):
    return redirect('/v2/', code=302)

# The v1 surface reads the live callback globals above, so it takes this
# module as its state rather than a snapshot taken at registration time.
register_legacy_v1_routes(__app__, sys.modules[__name__])


def _telemetry_source():
    """This process's retention state, as the thing to read.

    The decision itself is `telemetry_store.history_source`; this only supplies
    the globals that hold the state. Keeping the rule there is what lets it be
    tested without a Flask import.

    **No lock is taken here, and that is deliberate rather than an oversight.**
    `__telemetry_retention_lock__` serialises writers; it gives a reader no
    coherent view of several globals at once. What makes this safe is publication
    order: `start_telemetry_retention` assigns `__db__` and
    `__telemetry_start_failure__` *before* it assigns `__telemetry_state__`, and
    a reader must load the state **first**. So a reader either sees a state whose
    companions are already in place, or an earlier state - never a state
    promising a store that is not there yet.

    **`state` is bound on its own statement for exactly that reason, and the
    obvious one-liner is wrong.** Written as
    `history_source(__db__, __telemetry_state__, ...)` the arguments evaluate
    left to right, so the bytecode loads `__db__` *before* the state - the
    inverse of the discipline above, and the pairing it admits is
    `(None, RUNNING)`, which `history_source` answers with the "switched off"
    sentence this whole batch exists to remove. It is not reachable on CPython
    today, because no eval breaker is checked between adjacent `LOAD_GLOBAL`s
    and a 400,000-iteration two-thread probe found none - but a discipline that
    holds only by an interpreter detail is not a discipline, and a free-threaded
    build would not honour it.
    """
    state = __telemetry_state__
    return history_source(__db__, state, __telemetry_start_failure__)


def _v2_current_data():
    if __data_logger__ is None:
        return __read_data__() or {}
    return __data_logger__.get_data()


def _v2_power_action(action):
    return __power_controller__.execute(action)


def _telemetry_write_source():
    """The reading the retention writer records: the live-UI snapshot itself.

    The writer used to read `__runtime_hardware__.current_data` - the raw
    hardware-bridge reading. On a Raspberry Pi the HAT publishes flat scalars
    there and the writer recorded them; on a HAT-less x86 workstation the bridge
    is up but there is no HAT publishing a machine reading, so that call returns
    `{}` and the writer proved it could write, found nothing to record, and
    filled the store with nothing every interval (VD-095's residual on x86).

    The gauges the Home screen shows on that same box do not come from the raw
    bridge read: they come from `ControlPlaneRuntime.current_data`, the platform
    telemetry provider's snapshot, which fills CPU, memory, temperature,
    storage, network and accelerator readings from `/proc` and `/sys` when the
    bridge omits them. Feeding the writer that snapshot records the same real
    telemetry the UI already shows.

    It does not disturb the Pi. The snapshot starts from the bridge reading and
    only *fills* keys the bridge did not supply, so a Pi's HAT scalars still win
    and are written as before - the x86 host simply stops writing an empty row.
    `DataLogger.get_data` then flattens and keeps the storable scalars.
    """
    return __control_plane__.current_data()


from .control_plane_runtime import ControlPlaneRuntime

__control_plane__ = ControlPlaneRuntime(__app__, {
    "device_info": lambda: __device_info__ or __runtime_hardware__.device_info(),
    "current_data": _v2_current_data,
    "power_action": _v2_power_action,
    "power_capabilities": __power_controller__.capabilities,
    "read_config": lambda: __read_config__(),
    "update_config": lambda patch: __on_config_changed__(patch),
    "telemetry_history": lambda limit=30: history_samples(
        _telemetry_source(), limit
    ),
    # The same store, read by *time* rather than by row count, so the Assistant
    # can answer "the last 24 hours" or "the last 7 days" with a downsampled
    # trend instead of only the newest handful of per-interval rows. Reads the
    # same `_telemetry_source`, so it honours the identical four retention
    # states (off / starting / failed / running) rather than a second rule.
    "telemetry_history_range": lambda since_seconds, max_points=None: history_range(
        _telemetry_source(), since_seconds, max_points
    ),
    # The configured-vs-applied retention period and state, so the override
    # (#186) is visible on /api/v2 rather than only in the boot journal.
    # Deferred through a lambda: the function is defined further down, after the
    # retention start path it sits beside.
    "telemetry_retention": lambda: telemetry_retention_status(),
})


@__app__.after_request
def apply_v2_security_headers(response):
    if request.path.startswith("/v2") or request.path.startswith("/api/v2"):
        response.headers["X-Content-Type-Options"] = "nosniff"
        response.headers["X-Frame-Options"] = "DENY"
        response.headers["Referrer-Policy"] = "no-referrer"
        response.headers["Permissions-Policy"] = (
            "camera=(), microphone=(), geolocation=(), payment=()"
        )
        response.headers["Content-Security-Policy"] = (
            "default-src 'self'; "
            "script-src 'self'; "
            "style-src 'self'; "
            "img-src 'self' data:; "
            "connect-src 'self'; "
            "frame-src 'self' http://*:34002 https://*:34002; "
            "font-src 'self'; "
            "object-src 'none'; "
            "base-uri 'self'; "
            "form-action 'self'; "
            "frame-ancestors 'none'"
        )
        if request.is_secure:
            response.headers["Strict-Transport-Security"] = "max-age=31536000"
    if request.path.startswith("/v2"):
        # The /v2 routes in frontend_routes.py now set Cache-Control explicitly
        # for the cases that matter: content-hashed assets/ files are immutable,
        # the stale-asset recovery shim is no-store, and index.html is no-cache
        # so a new deploy's asset hashes are always picked up. `setdefault` keeps
        # those route decisions authoritative; this block only supplies a
        # fallback for any /v2 response (a 404, a favicon) that set none.
        if response.headers.get("X-Vaelor-Stale-Asset-Recovery") == "1":
            response.headers.setdefault("Cache-Control", NO_STORE_CACHE)
        elif response.mimetype == "text/html":
            response.headers.setdefault("Cache-Control", REVALIDATE_HTML_CACHE)
        elif request.path.startswith("/v2/assets/"):
            response.headers.setdefault("Cache-Control", IMMUTABLE_ASSET_CACHE)
        else:
            response.headers.setdefault("Cache-Control", REVALIDATE_HTML_CACHE)
    return response


from .frontend_routes import register_frontend_routes

register_frontend_routes(__app__, __www_v2_path__)


def _pironman_hat_present() -> bool:
    """Whether this machine carries the Pironman HAT's configuration source.

    Capability discovery, resolved on every call rather than at import, and off
    the same `PIRONMAN_CONFIG` path `HardwareBridgeClient` already keys off - not
    a new notion and not an architecture string. It is the honest signal for "no
    HAT": on an x86 workstation the file is absent for ever, while on a Pi it is
    present even during the boot window where the bridge socket has not yet bound
    (VD-095 / VD-096). `resolve_settings` uses exactly that distinction to tell a
    permanent no-HAT machine from a Pi whose bridge has not answered yet.
    """
    return PIRONMAN_CONFIG.exists()


def start_telemetry_retention():
    """Create and start the telemetry store this process reads from.

    **Why this is not in `VaelorControlPlane.run()`.** The Vaelor deployment
    serves through `vaelor.server.main()`, which imports `__app__` from this
    module and never constructs `VaelorControlPlane`. `run()` ends in
    `serve_forever()` and is the legacy Pironman entrypoint, so on an appliance
    it has never executed: nothing created the database, nothing applied the
    retention policy, and the data logger never wrote a row. Two weeks of
    telemetry were lost that way (VD-095 defect 2).

    **The globals are set here on purpose.** `__telemetry_state__`, `__db__`
    and `__telemetry_start_failure__` are what the v2 telemetry callback reads
    through `_telemetry_source`; `__enable_history__` is the v1 routes' own
    switch and is kept for them. They only ever meant anything in the process
    that set them - a value read from a freshly imported `control_plane` is the
    module default, which is how an earlier check reported "history is not
    retained" about an appliance holding 179 MB of samples.

    **Assignment order is load-bearing, not cosmetic.** Each state is published
    *after* the values it promises, because `_telemetry_source` reads the state
    first and takes it as a guarantee that its companions are in place. Inverting
    any of those pairs admits a torn read whose sentence is the false one.

    `__data_logger__` is deliberately *not* published. `_v2_current_data` falls
    back to `__read_data__()` when it is None, and `DataLogger.get_data()` drops
    list fields (and flattens nested dicts to scalars), so publishing it would
    quietly strip `ips` and the disk list out of `/api/v2` current data.

    The control plane is a single process - one `vaelor` under
    `vaelor-control-plane.service`, Werkzeug `threaded=True`, no pre-fork worker
    fan-out - so this starts one data logger. The lock and the early return keep
    that true if a second caller ever appears; the other ten Vaelor units have
    their own `main()` and must never call this.

    Never fatal. An appliance whose InfluxDB is down must still serve
    everything else, so a failure is logged and left for the read path to
    report as a reason.

    **Four outcomes, never two.** `STARTING` - this has not run yet, which is
    every restart until it does; `OFF` - the owner switched retention off;
    `FAILED` - it ran and could not start, with a reason; `RUNNING` - the store
    is open. Two review rounds each found a pair of these collapsed into one,
    and both collapses produced the same false sentence, that history is not
    retained. `FAILED` came from the hardware bridge: `RuntimeHardware._read`
    answers `{}` while its socket is unbound, and treating that as "switched
    off" made a transient ordering race permanently invisible. `STARTING` came
    from the module defaults being indistinguishable from `OFF`.

    **An x86 workstation is not a Pi with a slow bridge, and used to read as
    one (VD-095 / VD-096, defect #225).** With no HAT there is no
    `system.enable_history` flag anywhere, so `read_config()` returns `{}` for
    ever; the old code retried it thirty times and then reported "the hardware
    bridge may not have bound its socket", a socket race that was not happening,
    leaving retention permanently `FAILED` on a machine whose store was healthy.
    `_pironman_hat_present` is passed to `resolve_settings` so the permanent
    no-HAT case resolves to Vaelor-managed settings (enabled, the default
    period) instead, keyed off whether the HAT config source exists rather than
    off whether the bridge answered.

    **`FAILED` is terminal until the process restarts, and that is chosen.**
    `resolve_settings` retries within its own ceiling; nothing retries after it
    gives up. On a Pi a bridge that binds later than the ceiling still leaves
    retention dead until a restart, with a reason that says the socket "may not
    have bound". Recorded in VD-095 as a residual rather than repaired, because
    no production caller re-enters this and a background re-arm is a larger
    change than it looks. The x86 sub-case of that residual is now closed: a
    HAT-less host no longer waits on a bridge race it can never win.
    """
    global __db__, __enable_history__, __telemetry_retention__
    global __telemetry_start_failure__, __telemetry_state__, __telemetry_settings__
    log = __log__ or logging.getLogger(__name__)
    with __telemetry_retention_lock__:
        if __telemetry_retention__ is not None:
            return __telemetry_retention__
        try:
            # Retries within a stated ceiling: the hardware bridge is
            # `Type=simple`, so `After=` orders its start and not its readiness,
            # and this thread can arrive before the socket is bound. But that
            # race is a Pi's; on an x86 workstation there is no HAT and so no
            # `system.enable_history` flag to wait for, so `_pironman_hat_present`
            # lets `resolve_settings` tell the permanent no-HAT case from the
            # transient one and return Vaelor-managed settings instead of failing
            # thirty seconds later with an unbound-socket reason (VD-095/VD-096).
            settings = resolve_settings(
                __read_config__, log=log,
                hat_config_present=_pironman_hat_present,
            )
        except Exception as error:  # noqa: BLE001 - retention must not stop the boot
            # Recorded, not just logged. Left as a bare log line this reads to
            # the owner as "retention is switched off", which is false and sends
            # them to the wrong setting.
            __telemetry_start_failure__ = " ".join(str(error).split())
            __telemetry_state__ = RetentionState.FAILED
            log.error("Telemetry retention did not start: %s", error)
            return None
        # The configuration was read, so the configured period is now known and
        # can be shown beside the applied one on /api/v2 (#186), whatever happens
        # to the store below.
        __telemetry_settings__ = settings
        __enable_history__ = settings.enabled
        if not settings.enabled:
            __telemetry_state__ = RetentionState.OFF
            log.info("Telemetry history is switched off in the appliance configuration.")
            return None
        # Published before the start is attempted, so a failure reads as
        # "switched on and unreadable" rather than "switched off". Those are
        # different sentences for the owner and different repairs.
        __db__ = build_store(settings, log=log)
        try:
            retention = start_retention(
                __db__, settings, _telemetry_write_source, log=log
            )
        except Exception as error:  # noqa: BLE001 - retention must not stop the boot
            # A `TelemetryStoreError` is the store layer saying why in a
            # sentence built to be read, so the traceback adds nothing.
            # Anything else reaching here is unexpected and worth one.
            reporter = log.error if isinstance(error, TelemetryStoreError) else log.exception
            __telemetry_start_failure__ = " ".join(str(error).split())
            __telemetry_state__ = RetentionState.FAILED
            # Deliberately not the same sentence as the configuration failure
            # above: an operator reading the journal needs to know which of the
            # two stopped it, and one shared line makes them the same event.
            reporter("Telemetry retention could not open its store: %s", error)
            return None
        # State last, always. `_telemetry_source` reads it first, so assigning
        # it before `__db__` would publish a promise of a store that is not
        # there yet.
        __telemetry_retention__ = retention
        __telemetry_state__ = RetentionState.RUNNING
        return __telemetry_retention__


def telemetry_retention_status():
    """This process's retention state and periods, for the /api/v2 route (#186).

    A read-only assembly of the same globals `_telemetry_source` reads, handed
    to `telemetry_store.describe_retention`, which owns the wording so it can be
    tested without importing Flask - the same split the `RetentionState`
    decision already uses.
    """
    return describe_retention(
        __telemetry_state__, __telemetry_settings__, __telemetry_start_failure__
    )


class VaelorControlPlane:
    def __init__(self, device_info=None, database=TELEMETRY_STORE_NAME, config=None, log=None, get_logger=None):
        global __device_info__, __log_path__, __enable_history__
        global __data_logger__, __db__, __log__, __restart_service__
        global AVAILABLE_OLED_PAGES

        __device_info__ = device_info
        if 'app_name' in __device_info__:
            app_name = __device_info__['app_name']
        else:
            app_name = __device_info__['id']
        __log_path__ = f'/var/log/{app_name}'

        if get_logger:
            self.log = get_logger(__name__)
        else:
            self.log = log or logging.getLogger(__name__)
        __log__ = self.log

        if 'enable_history' not in config['system']:
            config['system']['enable_history'] = False
        __enable_history__ = config['system']['enable_history']
        if 'database_retention_days' not in config['system']:
            config['system']['database_retention_days'] = DEFAULT_RETENTION_DAYS
        database_retention_days = config['system']['database_retention_days']

        if __enable_history__:
            __db__ = Database(database, log=log, retention_days=database_retention_days)
        self.data_logger = DataLogger(
            database=__db__,
            interval=config['system']['data_interval'],
            log=self.log)
        __data_logger__ = self.data_logger

        self.started = False

        AVAILABLE_OLED_PAGES = []
        for item in __device_info__['peripherals']:
            if item.startswith("oled_page_"):
                AVAILABLE_OLED_PAGES.append(item.split("oled_page_")[1])

    @log_error
    def set_debug_level(self, level):
        self.log.setLevel(level)

    @log_error
    def set_test_smtp(self, func):
        global __test_smtp__
        __test_smtp__ = func

    @log_error
    def start(self):
        self.log.debug("Initializing Dashboard Server")
        tls_cert = env_value("VAELOR_TLS_CERT", "PM_TLS_CERT", "")
        tls_key = env_value("VAELOR_TLS_KEY", "PM_TLS_KEY", "")
        ssl_context = None
        if tls_cert or tls_key:
            if not (
                tls_cert
                and tls_key
                and path.isfile(tls_cert)
                and path.isfile(tls_key)
            ):
                raise RuntimeError(
                    "VAELOR_TLS_CERT and VAELOR_TLS_KEY must be readable; "
                    "legacy PM aliases are accepted."
                )
            ssl_context = (tls_cert, tls_key)
        scheme = "https" if ssl_context else "http"
        self.log.info(f"Starting Dashboard Server on {scheme}://{__host__}:{__port__}")
        # Production WSGI serving, not werkzeug.serving.make_server: that dev
        # server discloses framework/interpreter versions in its Server header
        # (#205). ControlPlaneServer keeps the same serve_forever/shutdown/
        # server_close handle the code below drives, and terminates TLS in front
        # of waitress. (This legacy path does not run on Vaelor appliances - see
        # start_telemetry_retention's docstring - but must not carry the pattern.)
        self.server = ControlPlaneServer(
            __host__, __port__, __app__, ssl_context=ssl_context, log=self.log
        )
        self.ctx = __app__.app_context()
        self.ctx.push()
        self.thread = threading.Thread(target=self.run, daemon=True)
        self.thread.start()
        self.log.info("Dashboard Server Started")

    @log_error
    def update_config(self, config):
        patch = {}
        if 'data_interval' in config:
            self.data_logger.set_interval(config['data_interval'])
            patch['data_interval'] = config['data_interval']
        if 'database_retention_days' in config:
            __db__.set_retention_days(config['database_retention_days'])
            patch['database_retention_days'] = config['database_retention_days']
        return patch

    @log_error
    def set_read_data(self, func):
        global __read_data__
        __read_data__ = func
        self.data_logger.set_read_data(func)

    @log_error
    def set_get_ip_data(self, func):
        global __get_ip_data__
        __get_ip_data__ = func

    @log_error
    def set_read_config(self, func):
        global __read_config__
        __read_config__ = func

    @log_error
    def set_on_config_changed(self, func):
        global __on_config_changed__
        __on_config_changed__ = func

    @log_error
    def set_on_restart_service(self, func):
        global __restart_service__
        __restart_service__ = func

    @log_error
    def set_play_pipower5_buzzer(self, func):
        global __play_pipower5_buzzer__
        __play_pipower5_buzzer__ = func

    @log_error
    def run(self):
        self.started = True
        if __enable_history__:
            self.log.debug("Starting Database")
            __db__.start()
            self.log.info("Database Started")
        self.log.debug("Starting Data Logger")
        self.data_logger.start()
        self.log.info("Data Logger Started")
        self.server.serve_forever()
        self.log.info("Dashboard Server started")

    @log_error
    def shutdown(self):
        self.server.shutdown()

    @log_error
    def stop(self):
        self.log.debug("Stopping Dashboard Server")
        if self.started:
            self.data_logger.stop()
            if __db__:
                __db__.close()
            self.server.shutdown()
            self.server.server_close()
            self.started = False
            self.thread.join(timeout=5)
            self.log.info("Dashboard Server stopped")
