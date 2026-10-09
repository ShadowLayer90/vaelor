import time
import logging
import threading

from influxdb import InfluxDBClient

from .cluster_placement import CONTROLLER_PLACEMENT_ID
from .telemetry_store import HISTORY_MEASUREMENT
# The flatten lives in a dependency-free module so the E2b worker telemetry
# emitter can reuse this exact behaviour without dragging in `influxdb` above.
# One copy, imported here under the names this module and its tests have always
# used, so the controller's stored row and a worker's cannot drift.
from .telemetry_flatten import _flatten_storable, _storable  # noqa: F401
from .utils import log_error


class DataLogger:

    @log_error
    def __init__(self, database=None, interval=1, log=None, node=None):
        self.log = log or logging.getLogger(__name__)
        self._is_ready = False
        # Which node's series this writer's rows belong to. On the controller
        # (the only caller today) it is the controller placement id, so its own
        # rows carry `node=controller` and the read path can filter to them the
        # same way it filters an enrolled worker's E2b rows.
        self.node = node or CONTROLLER_PLACEMENT_ID
        # How many readings the bounds table has discarded from this writer's
        # rows, and which fields the last time (VD-147): the controller's
        # counterpart of the per-worker count `telemetry_ingest_status` keeps.
        # Set before anything can return early (review nit).
        self.implausible_dropped = 0
        self.implausible_fields = []

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
        dropped = []
        row = _flatten_storable(data, dropped=dropped)
        if dropped:
            if not self.implausible_dropped:
                # Said once, loudly; counted every time after that.
                self.log.warning(
                    "Discarded an implausible reading (%s); it is not stored.",
                    ", ".join(sorted(dropped)),
                )
            self.implausible_dropped += len(dropped)
            self.implausible_fields = sorted(dropped)
        return row

    @log_error
    def loop(self):
        start = time.time()
        while self.running:
            data = self.get_data()
            if data != {}:
                if self.db is not None:
                    status, msg = self.db.set_tagged(
                        HISTORY_MEASUREMENT, {"node": self.node}, data
                    )
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
