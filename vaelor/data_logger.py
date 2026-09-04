import math
import time
import logging
import threading

from influxdb import InfluxDBClient

from .utils import log_error


def _storable(value):
    """One scalar InfluxDB can hold as a field, or None to drop it.

    `bool` is checked before `int` because it is a subclass of it, and is
    written as 0/1 so a field that reads as a number on one platform and a flag
    on another keeps a single stored type. Numbers and strings pass through;
    everything else (a list, a `None`, a nested object already handled by the
    caller) returns None so the writer drops it rather than handing InfluxDB a
    value it cannot store.
    """
    if isinstance(value, bool):
        return int(value)
    if isinstance(value, int):
        # One stored type per numeric field. A telemetry field can read `int 0`
        # at exact idle (`round(max(0, 0.0), 1)` collapses to int) and `float`
        # otherwise; InfluxDB rejects a write whose field type differs from the
        # stored one and drops that interval's whole row, so numbers are stored
        # as float uniformly.
        return float(value)
    if isinstance(value, float):
        # NaN and inf are not valid InfluxDB line-protocol field values; the
        # server rejects the write. A non-finite value in the FIRST sample would
        # fail retention startup for the whole box (VD-095 residual), so drop it
        # here rather than hand it to the store.
        return value if math.isfinite(value) else None
    if isinstance(value, str):
        return value
    return None


def _flatten_storable(data, prefix=""):
    """Flatten a machine reading into the flat scalar row InfluxDB stores.

    The writer used to keep only top-level scalars and drop every `list` and
    `dict`. On a Raspberry Pi the HAT reads back flat scalars, so that kept
    everything that mattered; on an x86 workstation the same reading arrives
    either empty or with its numbers nested one level down - a HAT-less host's
    telemetry is assembled from `/proc` and `/sys`, and a provider can group it
    as ``{"cpu": {"percent": 3, "temperature": 33}}``. Dropping every dict threw
    those numbers away, so `get_data` came back empty and the store recorded
    nothing every interval (the "nothing to record yet" warning on the x86 box).

    Flattening is platform-agnostic and is why this is not an ``if workstation``
    branch: a flat Pi reading passes through unchanged - there are no nested
    dicts to descend into - and a nested reading yields ``cpu_percent``,
    ``cpu_temperature`` and the like. Lists and any value that is not a number,
    bool or string are dropped rather than crashing the writer: InfluxDB cannot
    store them as a field, and a first sample that raised would read to the
    caller as "cannot record" rather than the empty reading it actually is.
    """
    flat = {}
    for key, value in data.items():
        name = "{}{}".format(prefix, key)
        if isinstance(value, dict):
            flat.update(_flatten_storable(value, "{}_".format(name)))
            continue
        stored = _storable(value)
        if stored is not None:
            flat[name] = stored
    return flat


class DataLogger:

    @log_error
    def __init__(self, database=None, interval=1, log=None):
        self.log = log or logging.getLogger(app_name)
        self._is_ready = False

        try:
            self.client = InfluxDBClient(host='localhost', port=8086)
        except Exception as e:
            self.log.error(f"Failed to connect to influxdb: {e}")
            return

        self.thread = None
        self.running = False

        self.db = database
        self.interval = interval

        self.status = {}
        self.__read_data__ = None

    @log_error
    def set_read_data(self, func):
        self.__read_data__ = func

    @log_error
    def set_interval(self, interval):
        self.interval = interval

    @log_error
    def get_data(self):
        if self.__read_data__ is None:
            self.log.error("No read data function set")
            return {}
        data = self.__read_data__()
        if not isinstance(data, dict) or not data:
            return {}
        return _flatten_storable(data)

    @log_error
    def loop(self):
        start = time.time()
        while self.running:
            data = self.get_data()
            if data != {}:
                if self.db is not None:
                    status, msg = self.db.set('history', data)
                    if not status:
                        self.log.error(f"Failed to set data: {msg}")

            elapsed = time.time() - start
            if elapsed < self.interval:
                time.sleep(self.interval - elapsed)
            start += self.interval

    @log_error
    def start(self):
        if self.running:
            self.log.warning("Already running")
            return
        self.running = True
        self.thread = threading.Thread(target=self.loop)
        self.thread.start()
        self.log.info("Data Logger Start")

    @log_error
    def stop(self):
        self.log.debug("Stopping Data Logger")
        if self.running:
            self.running = False
            self.thread.join()
        self.log.info("Data Logger stopped")
