from influxdb import InfluxDBClient
from influxdb.exceptions import InfluxDBClientError
import json
import logging
import subprocess
import time
from math import floor
import re

from .utils import log_error
from .telemetry_store import DEFAULT_RETENTION_DAYS, RETENTION_POLICY_NAME
import threading

class Database:
    def __init__(self, database, log=None, retention_days=DEFAULT_RETENTION_DAYS):
        self.log = log or logging.getLogger(__name__)
        self.lock = threading.Lock()

        self.database = database
        self.retention_days = retention_days
        self.influx_manually_started = False
        self.starting = False

        # initialize InfluxDB client
        self.client = InfluxDBClient(host='localhost', port=8086)

    def start(self):
        self.is_starting = True
        if not Database.is_influxdb_running():
            self.log.info("Starting influxdb service")
            self.start_influxdb()
        else:
            self.log.info("Influxdb service is already running")
        # Wait 2 seconds for InfluxDB to start
        time.sleep(2)

        self.log.debug("Waiting for InfluxDB to be ready")
        for _ in range(10):
            if not self.is_starting:
                return False
            # `server_is_ready`, not `is_ready`: this loop runs *before* the
            # database exists and is what creates it, so waiting on the database
            # here would wait forever and never create it (VD-095 defect 3).
            if self.server_is_ready():
                self.log.info("Influxdb is ready")
                break
            else:
                time.sleep(1)
        else:
            self.log.error("Timeout waiting for InfluxDB to be ready")
            return False

        databases = self.client.get_list_database()
        if not any(db['name'] == self.database for db in databases):
            self.client.create_database(self.database)
            self.log.info(f"Database '{self.database}' created successfully")

        self.client.switch_database(self.database)
        self.ensure_retention_policy_consistency()
        return True

    def server_is_ready(self):
        """Whether an InfluxDB server answers at all.

        This is the *server's* liveness and nothing more. It is the right
        question only while starting up, before the database can exist. Every
        read and write asks `is_ready()` instead.
        """
        try:
            return bool(self.client.ping())
        except Exception as e:
            self.log.error(f"Failed to connect to InfluxDB: {e}")
            return False

    def unavailable_reason(self):
        """Why *this database* cannot be read right now, or None when it can.

        VD-095 defect 3: readiness used to ping the InfluxDB server and never
        the database, so the check passed on an appliance where the database had
        never been created and the query then failed with `database not found`.
        A guard that reports on a neighbour of the thing it claims to check is
        worse than no guard, because it reads as evidence.

        The reason and the boolean come from one implementation so they can
        never disagree about the same instant.
        """
        try:
            if not self.client.ping():
                return "the InfluxDB server did not answer a ping"
            names = {db.get('name') for db in self.client.get_list_database()}
        except Exception as e:
            return f"the InfluxDB server could not be reached: {e}"
        if self.database not in names:
            return (
                f"the InfluxDB server is running but database "
                f"'{self.database}' does not exist"
            )
        return None

    def is_ready(self):
        """Whether this database can be read - not merely whether a server is up."""
        return self.unavailable_reason() is None

    @staticmethod
    def is_influxdb_running():
        try:
            # Use 'pgrep' to find the process
            subprocess.check_output(["pgrep", "influxd"])
            return True
        except subprocess.CalledProcessError:
            return False

    def start_influxdb(self):
        # Prefer the package-managed service so InfluxDB runs with its dedicated
        # identity and survives dashboard restarts. Keep a bounded fallback for
        # non-systemd development environments.
        try:
            result = subprocess.run(
                ["systemctl", "start", "influxdb.service"],
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,  # absence-ok: returncode and is_influxdb_running() are checked below
                timeout=15,
                check=False,
            )
            if result.returncode == 0 and self.is_influxdb_running():
                self.influx_manually_started = False
                return
        except (OSError, subprocess.TimeoutExpired):
            pass
        subprocess.Popen(["influxd"], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)  # absence-ok: daemon launch, not a probe; liveness read separately
        self.influx_manually_started = True

    def stop_influxdb(self):
        subprocess.Popen(["pkill", "influxd"])

    def parse_influxdb_duration(self, duration_str):
        """解析InfluxDB返回的各种时间格式为天数"""
        if duration_str == "INF":
            return float('inf')

        # 处理简单格式：Xd
        if duration_str.endswith('d'):
            try:
                return int(duration_str.rstrip('d'))
            except ValueError:
                pass

        # Compound durations: XhYmZs, XhYm, Xh, and so on.
        #
        # `fullmatch`, and at least one group must be present. Every group here
        # is optional, so `re.match` matched the *empty string* against any
        # input - "garbage" parsed as zero seconds, the caller compared 0 days
        # against the configured period and rewrote the policy, and the
        # "Unsupported duration format" warning below was unreachable. Since
        # VD-095 this decides whether the 7-day retention policy is applied, so
        # an unparseable duration has to be reported rather than read as zero.
        match = re.fullmatch(r'(?:(\d+)h)?(?:(\d+)m)?(?:(\d+)s)?', duration_str)
        if match and any(group is not None for group in match.groups()):
            hours = int(match.group(1)) if match.group(1) else 0
            minutes = int(match.group(2)) if match.group(2) else 0
            seconds = int(match.group(3)) if match.group(3) else 0

            # 转换为天
            total_seconds = hours * 3600 + minutes * 60 + seconds
            return total_seconds / (24 * 3600)

        # 无法解析的格式
        self.log.warning(f"Unsupported duration format: {duration_str}")
        return None

    @log_error
    def ensure_retention_policy_consistency(self):
        """确保保留策略与配置一致"""
        try:
            policies = self.client.get_list_retention_policies(self.database)
            default_policy = next((p for p in policies if p["default"]), None)

            # 预期的保留时长（天）和InfluxDB格式
            expected_days = self.retention_days
            expected_duration = f"{expected_days}d"

            if not default_policy:
                # 创建默认保留策略
                self.set_retention_days(expected_days)
                self.log.info(f"Created default retention policy: {expected_duration}")
            else:
                # 检查现有默认策略是否符合配置
                current_duration = default_policy["duration"]

                # 使用增强的解析函数
                current_days = self.parse_influxdb_duration(current_duration)

                if current_days is None:
                    self.log.error(f"Failed to parse retention duration: {current_duration}")
                    return

                if current_days != expected_days:
                    # 更新策略
                    self.set_retention_days(expected_days)
                    self.log.info(f"Updated inconsistent retention policy: {current_duration} → {expected_duration}")
                else:
                    self.log.debug(f"Retention policy is consistent: {expected_duration}")

        except Exception as e:
            self.log.exception(f"Failed to check retention policy consistency: {e}")

    @log_error
    def set_retention_days(self, days):
        """设置数据库的默认保留天数"""
        try:
            policy_name = RETENTION_POLICY_NAME
            duration = f"{days}d"

            # 检查策略是否存在
            policies = self.client.get_list_retention_policies(self.database)
            policy_names = [p["name"] for p in policies]

            if policy_name in policy_names:
                # 修改现有策略
                self.client.alter_retention_policy(
                    policy_name,
                    database=self.database,
                    duration=duration,
                    replication=1,
                    default=True
                )
                self.log.info(f"Updated retention policy to {duration}")
            else:
                # 创建新策略
                self.client.create_retention_policy(
                    policy_name,
                    duration=duration,
                    replication=1,
                    database=self.database,
                    default=True
                )
                self.log.info(f"Created retention policy {policy_name} with duration {duration}")

            return True, f"Retention policy set to {days} days"

        except Exception as e:
            self.log.exception(f"Failed to set retention policy: {e}")
            return False, str(e)

    def set(self, measurement, data):
        # self.log.debug(f"Setting data to database: measurement={measurement}, data={data}")
        if not self.is_ready():
            self.log.error('Database is not ready')
            return False, 'Database is not ready'
        json_body = [
            {
                "measurement": measurement,
                "fields": data
            }
        ]
        try:
            with self.lock:
                self.client.write_points(json_body)
            return True, json_body
        except InfluxDBClientError as e:
            return False, json.loads(e.content)["error"]
        except Exception as e:
            return False, str(e)

    def get_data_by_time_range(self, measurement, start_time, end_time, keys="*", function="mean", max_size=300):
        # self.log.warning(f"Getting data from database: measurement={measurement}, keys={keys}, start_time={start_time}, end_time={end_time}, function={function}, max_size={max_size}")
        if not self.is_ready():
            self.log.error('Database is not ready')
            return []
        if function not in ["mean", "sum", "min", "max", "count"]:
            self.log.error(f"Invalid function: {function}")
            return []
        if keys != "*":
            newKeys = []
            for k in keys.split(","):
                newKeys.append(f'{function}("{k}") as "{k}"')
            keys = ",".join(newKeys)
        duration = int(end_time) - int(start_time)
        duration_in_seconds = duration / 1000000000
        interval = 1
        if duration_in_seconds > max_size:
            interval = duration_in_seconds / max_size
            interval = floor(interval)
        query = f'SELECT {keys} FROM {measurement} WHERE time >= {start_time} AND time <= {end_time} GROUP BY time({interval}s)'
        # self.log.warning(f"Query: {query}")
        with self.lock:
            result = self.client.query(query)
        return list(result.get_points())

    #: The aggregate functions `get_trend` will apply. Restricted so an untrusted
    #: caller cannot smuggle arbitrary text into the `SELECT`; anything else falls
    #: back to `mean`.
    TREND_FUNCTIONS = ("mean", "min", "max", "count", "sum")

    def get_trend(self, measurement, since_seconds, bucket_seconds, function="mean"):
        """Every numeric field, aggregated into fixed time buckets, **ascending**.

        The counterpart to `get`, which returns the newest `n` rows by count:
        this returns one row per `bucket_seconds` window over the last
        `since_seconds`, so a caller can read a trend over a *span of time*
        without pulling every per-interval sample. Seven days at one row a
        second is ~600,000 rows; bucketed hourly it is 168.

        `MEAN(*)` (or the chosen aggregate) is applied to `*`, which InfluxDB
        evaluates only over numeric fields - the string labels the telemetry
        provider also writes (`cpu_temperature_source` and the like) are ignored
        rather than erroring, which is exactly what a trend wants. The `mean_`
        prefix InfluxDB puts on each aggregated column is stripped, so a bucket
        row carries the same field names as a raw `get` row (`cpu_temperature`,
        not `mean_cpu_temperature`) and everything downstream treats the two the
        same. `fill(none)` leaves a window with no samples out rather than
        inventing a zero for it.
        """
        if not self.is_ready():
            self.log.error('Database is not ready')
            return []
        function = function if function in self.TREND_FUNCTIONS else "mean"
        since = max(1, int(since_seconds))
        bucket = max(1, int(bucket_seconds))
        query = (
            f'SELECT {function.upper()}(*) FROM {measurement} '
            f'WHERE time > now() - {since}s '
            f'GROUP BY time({bucket}s) fill(none) ORDER BY time ASC'
        )
        with self.lock:
            result = self.client.query(query)
        prefix = f"{function}_"
        rows = []
        for point in result.get_points():
            row = {}
            for key, value in point.items():
                if key != "time" and key.startswith(prefix):
                    row[key[len(prefix):]] = value
                else:
                    row[key] = value
            rows.append(row)
        return rows

    def if_too_many_nulls(self, result, threshold=0.5):
        for point in result:
            error_length = len([key for key, value in point.items() if value is None])
            error_ratio = error_length / len(point)
            if error_ratio > threshold:
                return True
        return False

    def get(self, measurement, key="*", n=1):
        """The most recent `n` rows, returned **oldest to newest**.

        `ORDER BY time DESC` is how InfluxDB is asked for the *latest* n rows,
        so the query keeps it and the rows are reversed afterwards. Without the
        reversal every trend reads backwards: `metrics_history` tells the model
        "Samples are ordered oldest to newest", and `get_data_by_time_range`
        already returns ascending, so descending here made two methods of one
        class answer the same question differently (VD-095).
        """
        # self.log.debug(f"Getting data from database: measurement={measurement}, key={key}, n={n}")
        if not self.is_ready():
            self.log.error('Database is not ready')
            return []

        # Read data from last 1 second
        time_filter = "time < now() - 1s"
        query = f"SELECT {key} FROM {measurement} WHERE {time_filter} ORDER BY time DESC LIMIT {n}"
        with self.lock:
            result = self.client.query(query)

        result = list(reversed(list(result.get_points())))
        if n == 1:
            if len(result) == 0:
                self.log.debug(f"No data found for query: {query}")
                result = None
            else:
                result = result[0]
                if key != "*" and key != "time" and "," not in key:
                    result = result[key]
        # self.log.debug(f"Got data from database: {result}")
        return result

    def clear_measurement(self, measurement):
        self.log.info(f"Clearing database: {self.database}")
        if not self.is_ready():
            self.log.error('Database is not ready')
            return False
        self.client.drop_measurement(measurement)
        self.log.info(f"Database '{self.database}' cleared successfully")
        return True

    def close(self):
        self.client.close()
        if self.influx_manually_started:
            self.stop_influxdb()
        self.is_starting = False
        self.log.info("Database closed")
