"""Per-key use of the LLM Server, read from the gate's own access log.

The LLM Server's LAN door is an nginx container (:mod:`vaelor.llm_server_proxy`)
in front of a loopback model. A client that uses it never passes through any
Vaelor process, so nothing Vaelor ran could say which key was used, how often,
or when: the key table read "Never used" after an external app had used it all
day (ACC-043), and Settings counted none of its requests (ACC-048). The gate is
the one place every such request crosses, so it is the source (owner decision,
2026-09-28): it writes one line per request to the API naming the admitting key
by its FINGERPRINT, and this module turns those lines into counts.

**What the line carries, and what it never does.** ``v2 <epoch.ms> <FP> <status>
<seconds> <method>`` (a ``v1`` line, without the method, is still read): the
fingerprint is the vault's own eight-hex digest of the key
(:func:`vaelor.served_endpoint_keys.served_key_fingerprint`), set by the gate's
config inside the branch that matched that key, so the Authorization header is
never logged. A refused request (wrong or missing key) is logged under
:data:`GATE_REFUSED_MARK` instead, so guessing at keys is counted rather than
invisible - in a separate, separately capped file (:data:`GATE_REFUSED_FILENAME`),
so a flood of refusals can never push an admitted line out before it is read.
No path, no client address, no body. Tokens are not in it:
nginx does not read the response, and how much the model did is the model's
own count (:mod:`vaelor.model_usage`).

**Who writes and who reads (two processes, two users).**

* The gate's nginx master (root, in a container the ROOT hardware bridge runs)
  writes :data:`GATE_LOG_FILENAME` in :data:`GATE_LOG_HOST_DIR`, a ``0755`` root
  directory the bridge prepares (:mod:`vaelor.llm_gate_log`, which also keeps
  it bounded by rotating it to ``<name>.1``).
* The control plane (``vaelor``, in group ``vaelor``) reads it: the files are
  ``0644`` inside a directory it can traverse, so the reader needs no
  privilege and no new bridge verb. It keeps its place as ``(inode, offset)``,
  follows the rotation, and commits its counts and its new place in ONE
  transaction, so a crash between them re-reads nothing twice; a rotation it
  missed entirely is recorded as a gap, never silently.

Counts land in :class:`GateUsageStore` (control plane only, ``0600``), and each
key's newest use is handed to the credential broker's ``record_use`` verb, so
``last_used_at`` stays the one column every key listing reads
(:mod:`vaelor.credential_use`).
"""

from __future__ import annotations

import logging
import os
import re
import sqlite3
import threading
import weakref
import time
from contextlib import closing
from pathlib import Path
from typing import Any, Callable, Dict, Iterable, List, Mapping, NamedTuple, Optional, Tuple

from .runtime_paths import env_value, run_path, state_path
from .usage_rollup import (
    STATUS_CLIENT_ERROR, STATUS_FAILURE, bucket_start, prune_before, status_class,
    window_start,
)

LOGGER = logging.getLogger(__name__)

#: The gate's log directory on the host, and the one file name nginx writes.
GATE_LOG_DIRNAME = "llm-gate"
GATE_LOG_HOST_DIR = run_path(GATE_LOG_DIRNAME)
GATE_LOG_FILENAME = "access.log"
#: What the bridge renames a full log to before nginx reopens a fresh one.
GATE_LOG_ROTATED_SUFFIX = ".1"
#: Where that directory is mounted inside the gate's container.
GATE_LOG_CONTAINER_DIR = "/var/log/vaelor-gate"
#: The size past which the bridge rotates the log. About 36 bytes a line, so
#: roughly 29,000 requests; the reader drains every
#: :data:`GATE_DRAIN_SECONDS`, and would have to miss two rotations to lose one.
GATE_LOG_ROTATE_BYTES = 1024 * 1024

#: The nginx ``log_format`` name and body. The body is the wire format
#: :func:`parse_gate_line` reads; the version word leads so a future format can
#: be told apart rather than misread.
GATE_LOG_FORMAT_NAME = "vaelor_gate_v2"
GATE_LOG_VERSION = "v2"
#: Lines a gate rendered before the method was logged; read as before.
GATE_LEGACY_VERSION = "v1"
GATE_KEY_VARIABLE = "$vaelor_key_fp"
#: The one line shape; the key slot differs between the two files. The request
#: METHOD closes it (ACC-192), never the path: every OpenAI inference call is a
#: POST, and a GET (``/v1/models``, the key check every client starts with) is
#: a use of the key but not a request the model served, so it counts for the
#: key and stays out of the per-request times the Performance tab judges.
_LINE_SHAPE = "v2 $msec {} $status $request_time $request_method"
#: The method an inference request arrives with.
INFERENCE_METHOD = "POST"
_METHOD = re.compile(r"[A-Z]{1,16}")
GATE_LOG_FORMAT = _LINE_SHAPE.format(GATE_KEY_VARIABLE)
#: What a refused request is logged under in place of a fingerprint.
GATE_REFUSED_MARK = "-"

#: Refused requests (a wrong or missing key) go to their OWN file, capped on its
#: own and truncated in place when full, so no amount of refused traffic - which
#: any LAN client can send without a key - can push an admitted line out of the
#: key log before it is read. Refusals are therefore a lower bound under a
#: flood; admitted use never is. Same line shape, the refused mark in the key
#: slot, so one parser reads both.
GATE_REFUSED_FILENAME = "refused.log"
GATE_REFUSED_VARIABLE = "$vaelor_refused"
GATE_REFUSED_FORMAT_NAME = "vaelor_gate_refused_v2"
GATE_REFUSED_FORMAT = _LINE_SHAPE.format(GATE_REFUSED_MARK)
GATE_REFUSED_CAP_BYTES = 256 * 1024

#: The bucket columns that count only the requests the model served (POSTs,
#: ACC-192), so the 24 h window and the Performance tab count one population.
INFERENCE_BUCKET_COLUMNS = (
    ("inference_requests", "INTEGER"), ("inference_failures", "INTEGER"),
    ("inference_client_errors", "INTEGER"), ("inference_seconds", "REAL"),
)

#: How often the control plane drains the log.
GATE_DRAIN_SECONDS = 15.0
#: The most the reader takes in one drain, so a backlog cannot balloon memory;
#: the rest is read on the next drain. Larger than a rotated log and its
#: successor together, so a busy drain interval never skips a rotation.
MAX_READ_BYTES = 16 * 1024 * 1024

#: How long a proxy status read is reused, so the console's GETs do not each
#: cross the root bridge.
PROXY_STATUS_TTL_SECONDS = 5.0
#: How long a fingerprint the broker does not know is not asked about again.
UNKNOWN_FINGERPRINT_TTL_SECONDS = 300.0
#: A reader that has not completed a read in this long is not counting.
READER_STALE_SECONDS = 4 * GATE_DRAIN_SECONDS

#: The endpoint whose keys this log names (the LLM Server's registry id).
LLM_SERVER_ENDPOINT = "llm-server"

#: How many admitted requests are kept one by one, with their own time taken,
#: for the Performance card's latency percentiles. The bucket counts are never
#: capped; only this per-request detail is, as the gateway's is, and a span
#: reaching back past it says so (:meth:`GateUsageStore.detail_retained_since`).
GATE_DETAIL_ROWS = 10000

#: A fingerprint as the vault writes it: the last eight hex of a SHA-256, upper.
_FINGERPRINT = re.compile(r"[0-9A-F]{8}")

#: Whether the key-use figures are current, as the key table and Settings read
#: it. The one owner of these words; `frontend/src/lib/llmServerStatus.ts`
#: holds the one marked copy (`KEY_USAGE_STATES`).
USAGE_STATES = ("counting", "unreadable", "not-reading", "not-logging")
USAGE_COUNTING, USAGE_UNREADABLE, USAGE_NOT_READING, USAGE_NOT_LOGGING = USAGE_STATES


class GateRecord(NamedTuple):
    """One request to the API, as the gate logged it (refused ones included)."""

    at: float
    fingerprint: str
    status: int
    seconds: float
    #: ``""`` on a legacy line, which did not say.
    method: str = ""

    @property
    def refused(self) -> bool:
        return self.fingerprint == GATE_REFUSED_MARK

    @property
    def inference(self) -> bool:
        """Whether the model served this request (a legacy line is taken as it was)."""
        return self.method in ("", INFERENCE_METHOD)


def parse_gate_line(line: str) -> Optional[GateRecord]:
    """One log line as a :class:`GateRecord`, or ``None`` for anything else.

    Strict on purpose: a line in another version, with a key mark that is
    neither the vault's fingerprint shape nor the refused mark, or with a field
    that is not a number is skipped rather than counted under a guessed key.
    """
    fields = str(line or "").split()
    if len(fields) == 6 and fields[0] == GATE_LOG_VERSION and _METHOD.fullmatch(fields[5]):
        method = fields.pop()
    elif len(fields) == 5 and fields[0] == GATE_LEGACY_VERSION:
        method = ""
    else:
        return None
    _version, at, fingerprint, status, seconds = fields
    if fingerprint != GATE_REFUSED_MARK and not _FINGERPRINT.fullmatch(fingerprint):
        return None
    try:
        return GateRecord(float(at), fingerprint, int(status), max(0.0, float(seconds)), method)
    except ValueError:
        return None


#: Keyed by the bridge's socket path for a real client (a stable identity:
#: every client of one bridge shares its reading), else by the object itself,
#: held weakly so an entry can never outlive it. It was keyed by ``id(bridge)``,
#: which CPython reuses the moment an object is freed, so a new bridge could be
#: handed a dead one's status (review SC2).
_STATUS_CACHE: Dict[Any, Tuple[float, Any, Any]] = {}
_STATUS_LOCK = threading.Lock()


def _status_key(bridge: Any) -> Tuple[Any, Any]:
    """``(key, ref)``: the socket path, or the object with a weak reference to it."""
    path = getattr(bridge, "socket_path", None)
    if isinstance(path, str) and path:
        return ("socket", path), None
    try:
        ref = weakref.ref(bridge)
    except TypeError:  # an object that cannot be weakly referenced is not cached
        return None, None
    return ("object", id(bridge)), ref


def reset_status_cache() -> None:
    """Forget every cached proxy status (tests; a bridge restart reads afresh anyway)."""
    with _STATUS_LOCK:
        _STATUS_CACHE.clear()


def cached_proxy_status(bridge: Any, *, now: Optional[float] = None) -> Optional[Mapping[str, Any]]:
    """The LLM Server proxy's status over ``bridge``, reused for a few seconds.

    ``None`` when the bridge could not be asked. The console reads this on
    every GET of two screens; one read per :data:`PROXY_STATUS_TTL_SECONDS` is
    plenty for a flag that changes when the gate is restarted.
    """
    if bridge is None:
        return None
    clock = time.monotonic() if now is None else float(now)
    key, ref = _status_key(bridge)
    with _STATUS_LOCK:
        cached = _STATUS_CACHE.get(key) if key is not None else None
        # An object-keyed entry counts only for the very object it was read from.
        if cached and cached[2] is not None and cached[2]() is not bridge:
            cached = None
        if cached and clock - cached[0] < PROXY_STATUS_TTL_SECONDS:
            return cached[1]
    try:
        from .llm_server_proxy import LlmServerProxyController

        status = LlmServerProxyController(bridge).status()
    except Exception:  # noqa: BLE001 - an unasked gate is "not known"
        status = None
    if key is not None:
        with _STATUS_LOCK:
            _STATUS_CACHE[key] = (clock, status, ref)
    return status


def gate_logging_state(enabled: bool, status: Optional[Mapping[str, Any]]) -> Optional[Dict[str, str]]:
    """Whether an ENABLED LLM Server's running gate is recording use.

    ``None`` when there is nothing to say: the server is off, or its gate is not
    running (then nothing reaches the model through it). Otherwise a
    :data:`USAGE_NOT_LOGGING` state with the bridge's own reason when the gate
    serves without its log - or when the bridge could not be asked - because
    "Never used" read off a gate that was not recording is absence produced by
    the observer (LESSONS pattern 8). The one reading both the key table and
    Settings take.
    """
    if not enabled:
        return None
    if not isinstance(status, Mapping):
        return {"state": USAGE_NOT_LOGGING, "detail": (
            "Vaelor could not ask the hardware bridge whether the LLM Server is "
            "recording key use, so key use shown here may be incomplete."
        )}
    if not status.get("running") or status.get("usage_log"):
        return None
    return {"state": USAGE_NOT_LOGGING, "detail": str(status.get("usage_log_reason") or (
        "The LLM Server is serving without its usage log, so key use is not "
        "being counted."
    ))}


def _stat(path: str) -> Optional[os.stat_result]:
    try:
        return os.stat(path)
    except FileNotFoundError:
        return None


def _read_segment(
    path: str, inode: int, offset: int, limit: int,
) -> Optional[Tuple[bytes, bool]]:
    """Complete lines of ``path`` from ``offset``, if it is still file ``inode``.

    Returns ``(lines, whole)``: ``whole`` is True when the read reached the end
    of the file within ``limit``, so nothing but an unfinished last line is
    left. ``None`` when the name now points at another file (it was rotated
    between the stat and the open); the caller keeps its place and tries on the
    next drain. Only whole lines are returned, so a line nginx is still writing
    is read next time rather than split.
    """
    # Never through a link: the directory is written by a container.
    flags = os.O_RDONLY | getattr(os, "O_NOFOLLOW", 0) | getattr(os, "O_BINARY", 0)
    with os.fdopen(os.open(path, flags), "rb") as handle:
        if os.fstat(handle.fileno()).st_ino != inode:
            return None
        handle.seek(offset)
        data = handle.read(max(0, limit))
    return data[: data.rfind(b"\n") + 1], len(data) < max(0, limit)


def read_new_lines(
    directory: str, cursor: Optional[Tuple[int, int]], limit: int = MAX_READ_BYTES,
    filename: str = GATE_LOG_FILENAME,
) -> Tuple[List[str], Optional[Tuple[int, int]], bool]:
    """The lines written since ``cursor``, the new cursor, and whether any were lost.

    ``cursor`` is ``(inode, offset)`` of the file the reader stopped in, or
    ``None`` before the first read. Three cases, in order:

    * the cursor's file is now the rotated one: finish it, then read the new
      current file from its start;
    * the cursor's file is still current: read on from the offset (from the
      start if the file is now shorter than the offset);
    * the cursor's file is gone - rotated twice since the last read - or there
      is no cursor: read the rotated file and the current one whole. With a
      cursor, that is a GAP: the lines between the offset and the end of the
      lost file were never read, and the caller records it.

    A file shorter than the place the reader had in it was cut short (the
    bridge's last resort against an unbounded log), which is a gap too. The
    reader stays in the rotated file while it is unfinished or while nginx has
    not yet reopened a fresh current file, so no line is read twice.
    """
    current = os.path.join(directory, filename)
    rotated = current + GATE_LOG_ROTATED_SUFFIX
    current_stat = _stat(current)
    rotated_stat = _stat(rotated)
    inode, offset = cursor if cursor else (None, 0)
    chunks: List[bytes] = []
    gap = False

    rotated_from: Optional[int] = None
    if inode is not None and rotated_stat is not None and rotated_stat.st_ino == inode:
        rotated_from = offset
        if rotated_stat.st_size < offset:
            rotated_from, gap = 0, True
    elif inode is not None and current_stat is not None and current_stat.st_ino == inode:
        if current_stat.st_size < offset:
            offset, gap = 0, True
    else:
        gap = inode is not None
        inode, offset = (current_stat.st_ino, 0) if current_stat else (None, 0)
        rotated_from = 0 if rotated_stat is not None else None

    budget = limit
    if rotated_from is not None and rotated_stat is not None:
        segment = _read_segment(rotated, rotated_stat.st_ino, rotated_from, budget)
        if segment is None:
            return [], cursor, False
        data, whole = segment
        chunks.append(data)
        budget -= len(data)
        if not whole or current_stat is None:
            return _lines(chunks), (rotated_stat.st_ino, rotated_from + len(data)), gap
        inode, offset = current_stat.st_ino, 0

    if inode is not None and current_stat is not None and current_stat.st_ino == inode:
        segment = _read_segment(current, inode, offset, budget)
        if segment is not None:
            chunks.append(segment[0])
            offset += len(segment[0])
    return _lines(chunks), ((inode, offset) if inode is not None else None), gap


def _lines(chunks: Iterable[bytes]) -> List[str]:
    return [
        line for chunk in chunks
        for line in chunk.decode("utf-8", "replace").splitlines() if line.strip()
    ]


class GateUsageStore:
    """Durable per-key counts from the gate log (control plane only, ``0600``).

    ``gate_key_totals`` holds lifetime counts per fingerprint and never prunes;
    ``gate_buckets`` holds the same counts per :mod:`vaelor.usage_rollup` bucket
    for windowed questions (refused requests under :data:`GATE_REFUSED_MARK`);
    ``gate_requests`` holds the newest :data:`GATE_DETAIL_ROWS` admitted
    requests one by one - time, status and the gate's own time taken - so the
    Performance card's percentiles are computed from measured requests, never
    from a bucket's average;
    ``gate_reader`` holds the reader's place in the log, its last error and its
    last gap, and when counting began. The time of the last successful read is
    kept in memory: it changes every drain, and the reader lives in this process.
    """

    def __init__(self, database_path: Optional[str] = None):
        self.database_path = database_path or env_value(
            "VAELOR_LLM_GATE_USAGE_DB", "PM_LLM_GATE_USAGE_DB",
            state_path("cluster/llm-gate-usage.sqlite3"),
        )
        self.last_read_at: Optional[float] = None
        Path(self.database_path).parent.mkdir(parents=True, exist_ok=True, mode=0o700)
        with closing(self._connect()) as connection:
            connection.executescript(
                """
                CREATE TABLE IF NOT EXISTS gate_key_totals (
                    endpoint TEXT NOT NULL, fingerprint TEXT NOT NULL,
                    credential_id TEXT NOT NULL DEFAULT '',
                    requests INTEGER NOT NULL DEFAULT 0,
                    failures INTEGER NOT NULL DEFAULT 0,
                    client_errors INTEGER NOT NULL DEFAULT 0,
                    first_used_at REAL NOT NULL, last_used_at REAL NOT NULL,
                    PRIMARY KEY (endpoint, fingerprint)
                );
                CREATE TABLE IF NOT EXISTS gate_buckets (
                    endpoint TEXT NOT NULL, fingerprint TEXT NOT NULL,
                    bucket INTEGER NOT NULL,
                    requests INTEGER NOT NULL DEFAULT 0,
                    failures INTEGER NOT NULL DEFAULT 0,
                    client_errors INTEGER NOT NULL DEFAULT 0,
                    seconds_sum REAL NOT NULL DEFAULT 0,
                    PRIMARY KEY (endpoint, fingerprint, bucket)
                );
                CREATE INDEX IF NOT EXISTS gate_buckets_by_time ON gate_buckets(bucket);
                CREATE TABLE IF NOT EXISTS gate_refused_reader (
                    id INTEGER PRIMARY KEY CHECK (id = 1),
                    inode TEXT NOT NULL DEFAULT '', offset INTEGER NOT NULL DEFAULT 0
                );
                CREATE TABLE IF NOT EXISTS gate_reader (
                    id INTEGER PRIMARY KEY CHECK (id = 1),
                    inode TEXT NOT NULL DEFAULT '', offset INTEGER NOT NULL DEFAULT 0,
                    last_error TEXT NOT NULL DEFAULT '', last_gap_at REAL,
                    started_at REAL NOT NULL
                );
                CREATE TABLE IF NOT EXISTS gate_requests (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    at REAL NOT NULL, status INTEGER NOT NULL, seconds REAL NOT NULL
                );
                CREATE INDEX IF NOT EXISTS gate_requests_by_time ON gate_requests(at);
                CREATE TABLE IF NOT EXISTS gate_detail (
                    id INTEGER PRIMARY KEY CHECK (id = 1),
                    started_at REAL NOT NULL, pruned_at REAL
                );
                """
            )
            connection.execute(
                "INSERT OR IGNORE INTO gate_detail(id, started_at) VALUES (1, ?)", (time.time(),)
            )
            # W4d-D7 (LESSONS 6): the window's requests are the ones the model
            # served (ACC-192), counted per bucket beside every admitted use. A
            # store from before these columns says so through `inference_since`.
            bucket_columns = {row[1] for row in connection.execute("PRAGMA table_info(gate_buckets)")}
            for column, kind in INFERENCE_BUCKET_COLUMNS:
                if column not in bucket_columns:
                    connection.execute(
                        "ALTER TABLE gate_buckets ADD COLUMN {} {} NOT NULL DEFAULT 0".format(column, kind)
                    )
            columns = {row[1] for row in connection.execute("PRAGMA table_info(gate_reader)")}
            if "inference_since" not in columns:
                connection.execute("ALTER TABLE gate_reader ADD COLUMN inference_since REAL")
                connection.execute("UPDATE gate_reader SET inference_since = ?", (time.time(),))
            if "started_at" not in columns:
                connection.execute("ALTER TABLE gate_reader ADD COLUMN started_at REAL")
            connection.execute(
                "INSERT OR IGNORE INTO gate_reader(id, started_at) VALUES (1, ?)", (time.time(),)
            )
            connection.execute(
                "UPDATE gate_reader SET started_at = ? WHERE started_at IS NULL", (time.time(),)
            )
            connection.commit()
        try:
            os.chmod(self.database_path, 0o600)
        except OSError:
            pass

    def _connect(self) -> sqlite3.Connection:
        connection = sqlite3.connect(self.database_path, timeout=15)
        connection.row_factory = sqlite3.Row
        return connection

    def _reader_row(self) -> sqlite3.Row:
        with closing(self._connect()) as connection:
            return connection.execute("SELECT * FROM gate_reader WHERE id = 1").fetchone()

    # -- the reader's place ------------------------------------------------

    def cursor(self) -> Optional[Tuple[int, int]]:
        row = self._reader_row()
        if row is None or not str(row["inode"]):
            return None
        return int(row["inode"]), int(row["offset"])

    def refused_cursor(self) -> Optional[Tuple[int, int]]:
        with closing(self._connect()) as connection:
            row = connection.execute(
                "SELECT inode, offset FROM gate_refused_reader WHERE id = 1"
            ).fetchone()
        if row is None or not str(row["inode"]):
            return None
        return int(row["inode"]), int(row["offset"])

    def note_error(self, error: str) -> None:
        """Record a failed read (only when the reason changed)."""
        if str(self._reader_row()["last_error"]) == error[:300]:
            return
        with closing(self._connect()) as connection:
            connection.execute("UPDATE gate_reader SET last_error = ? WHERE id = 1", (error[:300],))
            connection.commit()

    def reader_state(self, now: Optional[float] = None) -> Dict[str, Any]:
        """Whether the log is being read: counting, unreadable, or not reading.

        The key table and Settings show "Never used" or a zero only while this
        says ``counting``; otherwise they say the count is not known, because a
        reader that is failing or stopped produces absence (LESSONS pattern 8).
        ``last_gap_at`` is when lines were last lost to a rotation the reader
        missed, for the screen to say so.
        """
        clock = time.time() if now is None else float(now)
        row = self._reader_row()
        error = str(row["last_error"] or "") if row else ""
        if error:
            state, detail = USAGE_UNREADABLE, (
                "Vaelor could not read the LLM Server's usage log, so key use "
                "is not known: " + error
            )
        elif self.last_read_at is None or clock - self.last_read_at > READER_STALE_SECONDS:
            state, detail = USAGE_NOT_READING, (
                "Vaelor has not read the LLM Server's usage log recently, so key "
                "use shown here may be out of date."
            )
        else:
            state, detail = USAGE_COUNTING, ""
        return {
            "state": state, "detail": detail, "last_read_at": self.last_read_at,
            "last_gap_at": row["last_gap_at"] if row else None,
            "started_at": row["started_at"] if row else None,
        }

    # -- counts ------------------------------------------------------------

    def commit_drain(
        self, records: Iterable[GateRecord], credentials: Mapping[str, str], *,
        cursor: Optional[Tuple[int, int]], gap: bool = False,
        refused_cursor: Optional[Tuple[int, int]] = None,
        endpoint: str = LLM_SERVER_ENDPOINT, now: Optional[float] = None,
    ) -> int:
        """Fold a drain's records AND move the reader's place, in one transaction.

        A crash before the commit leaves both as they were, so the next drain
        reads the same lines again and counts them once; a crash after it has
        nothing left to redo. ``credentials`` maps a fingerprint to the vault
        credential it belongs to; a fingerprint with no entry (a key rotated or
        revoked since) is still counted, under its fingerprint alone. Refused
        requests are counted per bucket only. Returns the admitted requests.
        """
        totals: Dict[str, Dict[str, Any]] = {}
        buckets: Dict[Tuple[str, int], Dict[str, float]] = {}
        detail: List[Tuple[float, int, float]] = []
        for record in records:
            kind = status_class(record.status)
            bucket = buckets.setdefault((record.fingerprint, bucket_start(record.at)), {
                "requests": 0, "failures": 0, "client_errors": 0, "seconds": 0.0,
                "inference_requests": 0, "inference_failures": 0,
                "inference_client_errors": 0, "inference_seconds": 0.0,
            })
            targets = [bucket]
            if not record.refused:
                total = totals.setdefault(record.fingerprint, {
                    "requests": 0, "failures": 0, "client_errors": 0,
                    "first": record.at, "last": record.at,
                })
                total["first"] = min(total["first"], record.at)
                total["last"] = max(total["last"], record.at)
                targets.append(total)
            for target in targets:
                target["requests"] += 1
                target["failures"] += kind == STATUS_FAILURE
                target["client_errors"] += kind == STATUS_CLIENT_ERROR
            bucket["seconds"] += record.seconds
            if record.inference:
                bucket["inference_requests"] += 1
                bucket["inference_failures"] += kind == STATUS_FAILURE
                bucket["inference_client_errors"] += kind == STATUS_CLIENT_ERROR
                bucket["inference_seconds"] += record.seconds
            # Only a request the model served is a request the Performance
            # tab times and counts (ACC-192); the key's use is counted above.
            if not record.refused and record.inference:
                detail.append((record.at, record.status, record.seconds))
        clock = time.time() if now is None else float(now)
        with closing(self._connect()) as connection:
            if detail:
                self._keep_detail(connection, detail, clock)
            for fingerprint, total in totals.items():
                connection.execute(
                    """INSERT INTO gate_key_totals(endpoint, fingerprint, credential_id,
                           requests, failures, client_errors, first_used_at, last_used_at)
                       VALUES (?,?,?,?,?,?,?,?)
                       ON CONFLICT(endpoint, fingerprint) DO UPDATE SET
                           credential_id = CASE WHEN excluded.credential_id != ''
                               THEN excluded.credential_id ELSE gate_key_totals.credential_id END,
                           requests = gate_key_totals.requests + excluded.requests,
                           failures = gate_key_totals.failures + excluded.failures,
                           client_errors = gate_key_totals.client_errors + excluded.client_errors,
                           first_used_at = MIN(gate_key_totals.first_used_at, excluded.first_used_at),
                           last_used_at = MAX(gate_key_totals.last_used_at, excluded.last_used_at)""",
                    (
                        endpoint, fingerprint, str(credentials.get(fingerprint) or ""),
                        total["requests"], total["failures"], total["client_errors"],
                        total["first"], total["last"],
                    ),
                )
            for (fingerprint, start), bucket in buckets.items():
                connection.execute(
                    """INSERT INTO gate_buckets(endpoint, fingerprint, bucket, requests,
                           failures, client_errors, seconds_sum, inference_requests,
                           inference_failures, inference_client_errors, inference_seconds)
                       VALUES (?,?,?,?,?,?,?,?,?,?,?)
                       ON CONFLICT(endpoint, fingerprint, bucket) DO UPDATE SET
                           requests = gate_buckets.requests + excluded.requests,
                           failures = gate_buckets.failures + excluded.failures,
                           client_errors = gate_buckets.client_errors + excluded.client_errors,
                           seconds_sum = gate_buckets.seconds_sum + excluded.seconds_sum,
                           inference_requests = gate_buckets.inference_requests
                               + excluded.inference_requests,
                           inference_failures = gate_buckets.inference_failures
                               + excluded.inference_failures,
                           inference_client_errors = gate_buckets.inference_client_errors
                               + excluded.inference_client_errors,
                           inference_seconds = gate_buckets.inference_seconds
                               + excluded.inference_seconds""",
                    (
                        endpoint, fingerprint, start, bucket["requests"],
                        bucket["failures"], bucket["client_errors"], bucket["seconds"],
                        bucket["inference_requests"], bucket["inference_failures"],
                        bucket["inference_client_errors"], bucket["inference_seconds"],
                    ),
                )
            if buckets:
                connection.execute(
                    "DELETE FROM gate_buckets WHERE bucket < ?", (prune_before(clock),)
                )
            row = connection.execute("SELECT inode, offset, last_error FROM gate_reader WHERE id = 1").fetchone()
            inode, offset = cursor if cursor else ("", 0)
            if (str(row["inode"]), int(row["offset"])) != (str(inode), int(offset)) or row["last_error"]:
                connection.execute(
                    "UPDATE gate_reader SET inode = ?, offset = ?, last_error = '' WHERE id = 1",
                    (str(inode), int(offset)),
                )
            if gap:
                connection.execute(
                    "UPDATE gate_reader SET last_gap_at = ? WHERE id = 1", (clock,)
                )
            if refused_cursor is not None:
                connection.execute(
                    "INSERT OR REPLACE INTO gate_refused_reader(id, inode, offset) VALUES (1, ?, ?)",
                    (str(refused_cursor[0]), int(refused_cursor[1])),
                )
            connection.commit()
        self.last_read_at = clock
        return sum(total["requests"] for total in totals.values())

    @staticmethod
    def _keep_detail(
        connection: sqlite3.Connection, detail: List[Tuple[float, int, float]], clock: float,
    ) -> None:
        """Add a drain's admitted requests to the per-request detail, then bound it.

        Runs inside :meth:`commit_drain`'s transaction, so the detail moves with
        the counts and the cursor or not at all. Anything older than the bucket
        retention or beyond :data:`GATE_DETAIL_ROWS` is pruned, and the prune is
        recorded so a span reaching past it is known to be partial.
        """
        connection.executemany(
            "INSERT INTO gate_requests(at, status, seconds) VALUES (?,?,?)", detail,
        )
        pruned = connection.execute(
            """DELETE FROM gate_requests WHERE at < ? OR id <= (
                   SELECT id FROM gate_requests ORDER BY id DESC LIMIT 1 OFFSET ?)""",
            (prune_before(clock), GATE_DETAIL_ROWS),
        ).rowcount
        if pruned > 0:
            connection.execute("UPDATE gate_detail SET pruned_at = ? WHERE id = 1", (clock,))

    def rows_between(self, start_time: float, end_time: float) -> List[Dict[str, Any]]:
        """Admitted requests in ``[start, end)``, oldest first, in the gateway's row shape.

        ``created_at``, ``status`` and ``duration_ms`` - the gate's own
        ``$request_time``, from the request's first byte to the response's
        last - so the Performance card computes this door's percentiles exactly
        as it does the gateway's. The gate cannot see tokens, so there are none.
        """
        with closing(self._connect()) as connection:
            rows = connection.execute(
                """SELECT at, status, seconds FROM gate_requests
                   WHERE at >= ? AND at < ? ORDER BY at ASC""",
                (float(start_time), float(end_time)),
            ).fetchall()
        return [
            {"created_at": float(row["at"]), "status": int(row["status"]),
             "duration_ms": float(row["seconds"]) * 1000.0}
            for row in rows
        ]

    def detail_retained_since(self) -> Optional[float]:
        """From when the per-request detail is complete, or None while it holds everything.

        After a prune, the oldest request still held. Before one, the moment
        the detail began if counting began in an earlier bucket (a store
        upgraded from before requests were kept one by one): those earlier
        requests were counted but never timed one by one, so a span reaching
        back past it is partial and a screen says so.
        """
        with closing(self._connect()) as connection:
            detail = connection.execute(
                "SELECT started_at, pruned_at FROM gate_detail WHERE id = 1"
            ).fetchone()
            if detail is None:
                return None
            if detail["pruned_at"] is not None:
                oldest = connection.execute("SELECT MIN(at) FROM gate_requests").fetchone()
                return float(oldest[0]) if oldest and oldest[0] is not None else float(detail["pruned_at"])
            counted = connection.execute(
                "SELECT MIN(bucket) FROM gate_buckets WHERE fingerprint != ?", (GATE_REFUSED_MARK,),
            ).fetchone()
        began = float(detail["started_at"])
        if counted and counted[0] is not None and float(counted[0]) < bucket_start(began):
            return began
        return None

    def key_totals(self, endpoint: str = LLM_SERVER_ENDPOINT) -> List[Dict[str, Any]]:
        """Lifetime counts per fingerprint, each with the credential it resolved to."""
        with closing(self._connect()) as connection:
            rows = connection.execute(
                """SELECT fingerprint, credential_id, requests, failures,
                          client_errors, first_used_at, last_used_at
                   FROM gate_key_totals WHERE endpoint = ?""",
                (endpoint,),
            ).fetchall()
        return [dict(row) for row in rows]

    def window_totals(
        self, window_seconds: float, *, now: Optional[float] = None,
        endpoint: str = LLM_SERVER_ENDPOINT,
    ) -> Dict[str, Any]:
        """The model's requests, failures and latency, every use, and refusals.

        W4d-D7 (LESSONS 6): ``requests``, ``failures``, ``client_errors`` and
        ``duration_ms_sum`` count only the requests the model served (POSTs,
        ACC-192) - the population the Performance tab times. Counting every
        admitted line made Settings say 17 of 20 failed at 0 ms while
        Performance said 0 errors: the failures were key checks (``GET
        /v1/models``) answered at once by a door with no model behind it.
        ``uses`` keeps every admitted line - the key's use. ``since`` is the
        real start of the span counted: the earliest bucket read (see
        :mod:`vaelor.usage_rollup`), or when counting began if that is later.
        """
        clock = time.time() if now is None else float(now)
        since = window_start(clock, window_seconds)
        with closing(self._connect()) as connection:
            row = connection.execute(
                """SELECT COALESCE(SUM(CASE WHEN fingerprint != ? THEN inference_requests END), 0)
                              AS requests,
                          COALESCE(SUM(CASE WHEN fingerprint != ? THEN inference_failures END), 0)
                              AS failures,
                          COALESCE(SUM(CASE WHEN fingerprint != ? THEN inference_client_errors END), 0)
                              AS client_errors,
                          COALESCE(SUM(CASE WHEN fingerprint != ? THEN inference_seconds END), 0)
                              AS seconds_sum,
                          COALESCE(SUM(CASE WHEN fingerprint != ? THEN requests END), 0) AS uses,
                          COALESCE(SUM(CASE WHEN fingerprint = ? THEN requests END), 0) AS refused
                   FROM gate_buckets WHERE endpoint = ? AND bucket >= ?""",
                (GATE_REFUSED_MARK,) * 6 + (endpoint, since),
            ).fetchone()
            started = connection.execute(
                "SELECT started_at, inference_since FROM gate_reader WHERE id = 1"
            ).fetchone()
        began = max(
            float(started["started_at"]), float(started["inference_since"] or 0)
        ) if started else since
        return {
            "since": max(since, began),
            "requests": int(row["requests"]),
            "failures": int(row["failures"]),
            "client_errors": int(row["client_errors"]),
            "duration_ms_sum": float(row["seconds_sum"]) * 1000.0,
            "uses": int(row["uses"]),
            "refused": int(row["refused"]),
        }


def key_usage_for_rows(
    store: Optional[GateUsageStore], rows: Iterable[Mapping[str, Any]],
) -> Dict[str, Dict[str, Any]]:
    """Per-credential request counts for the key rows a surface lists.

    A key's count is every line logged under its CURRENT fingerprint plus every
    line an earlier drain already attributed to its credential id - so a
    rotation does not reset what the table shows. Keyed by credential id; a
    store that cannot be read answers ``{}`` and the caller says so.
    """
    if store is None:
        return {}
    try:
        totals = store.key_totals()
    except sqlite3.Error:
        return {}
    usage: Dict[str, Dict[str, Any]] = {}
    for row in rows:
        credential_id = str(row.get("id") or row.get("credential_id") or "")
        fingerprint = str(row.get("fingerprint") or row.get("key_fingerprint") or "")
        if not credential_id:
            continue
        matched = [
            total for total in totals
            if (credential_id and total["credential_id"] == credential_id)
            or (fingerprint and total["fingerprint"] == fingerprint)
        ]
        usage[credential_id] = {
            "requests": sum(int(total["requests"]) for total in matched),
            "failures": sum(int(total["failures"]) for total in matched),
        }
    return usage


def _gate_credentials(broker: Any) -> Dict[str, str]:
    """Fingerprint -> credential id for every LLM Server key the vault holds.

    Revoked keys included, so a request logged just before a revoke still
    lands on its key. The listing is fingerprint-only; nothing is decrypted.
    """
    from .served_endpoint_keys import SERVED_ENDPOINT_PROVIDER

    mapping: Dict[str, str] = {}
    for row in broker.list() or []:
        if (
            row.get("provider") == SERVED_ENDPOINT_PROVIDER
            and str(row.get("endpoint_id") or "") == LLM_SERVER_ENDPOINT
            and row.get("fingerprint")
        ):
            mapping[str(row["fingerprint"])] = str(row.get("id") or "")
    return mapping


class GateUsageCollector:
    """Drain the gate log into :class:`GateUsageStore` on a loop (control plane).

    One drain reads the lines since the stored cursor and commits the counts
    with the new cursor in one transaction. The fingerprint-to-key map is
    listed from the broker only when a fingerprint it does not know appears,
    not on every drain. Then every key's newest recorded use is reported to the
    broker's ``record_use`` whenever it is newer than what this process last
    reported - so a use a broker outage missed is reported on a later drain,
    and ``record_use`` moving only forward makes a repeat harmless.
    """

    def __init__(
        self,
        store: GateUsageStore,
        *,
        broker: Any = None,
        directory: str = GATE_LOG_HOST_DIR,
        interval_seconds: float = GATE_DRAIN_SECONDS,
        clock: Callable[[], float] = time.time,
    ):
        self.store = store
        self._broker = broker
        self._directory = directory
        self._interval = float(interval_seconds)
        self._clock = clock
        self._stop = threading.Event()
        self._thread: Optional[threading.Thread] = None
        self._credentials: Dict[str, str] = {}
        self._unknown_until: Dict[str, float] = {}
        self._reported: Dict[str, float] = {}

    def _credentials_for(self, records: List[GateRecord]) -> Dict[str, str]:
        """The fingerprint map, listed from the broker only for a new fingerprint.

        A fingerprint the listing does not hold (a key purged since, or a line
        no key wrote) is not asked about again for
        :data:`UNKNOWN_FINGERPRINT_TTL_SECONDS`.
        """
        now = self._clock()
        unknown = {
            record.fingerprint for record in records
            if not record.refused and record.fingerprint not in self._credentials
            and self._unknown_until.get(record.fingerprint, 0.0) <= now
        }
        if unknown and self._broker is not None:
            try:
                self._credentials = _gate_credentials(self._broker)
            except Exception as error:  # noqa: BLE001 - counts must not wait on the broker
                LOGGER.debug("LLM Server keys could not be listed: %s", error)
                return self._credentials
            for fingerprint in unknown - set(self._credentials):
                self._unknown_until[fingerprint] = now + UNKNOWN_FINGERPRINT_TTL_SECONDS
        return self._credentials

    def drain_once(self) -> int:
        """One read -> commit -> report step. Returns the requests folded; never raises."""
        now = self._clock()
        try:
            cursor = self.store.cursor()
            lines, new_cursor, gap = read_new_lines(self._directory, cursor)
            refused_lines, refused_cursor, _refused_gap = read_new_lines(
                self._directory, self.store.refused_cursor(),
                filename=GATE_REFUSED_FILENAME,
            )
        except OSError as error:
            reason = error.strerror or type(error).__name__
            try:
                self.store.note_error(reason)
            except sqlite3.Error:
                pass
            LOGGER.warning("The LLM Server usage log could not be read: %s", reason)
            return 0
        except sqlite3.Error as error:
            LOGGER.warning("The LLM Server usage store could not be read: %s", error)
            return 0
        # Each file holds only its own kind: a key's use from the key log, a
        # refusal from the refused log.
        records = [record for record in map(parse_gate_line, lines) if record and not record.refused]
        refused = [record for record in map(parse_gate_line, refused_lines) if record and record.refused]
        credentials = self._credentials_for(records)
        try:
            folded = self.store.commit_drain(
                records + refused, credentials, cursor=new_cursor, gap=gap,
                refused_cursor=refused_cursor, now=now,
            )
        except sqlite3.Error as error:
            LOGGER.warning("LLM Server usage could not be stored: %s", error)
            return 0
        if gap:
            LOGGER.warning(
                "The LLM Server usage log rotated twice between reads; the "
                "requests in between were not counted."
            )
        self._report_uses()
        return folded

    def _report_uses(self) -> None:
        if self._broker is None or not callable(getattr(self._broker, "record_use", None)):
            return
        newest: Dict[str, float] = {}
        try:
            totals = self.store.key_totals()
        except sqlite3.Error:
            return
        for total in totals:
            credential_id = total["credential_id"] or self._credentials.get(total["fingerprint"], "")
            if credential_id:
                newest[credential_id] = max(newest.get(credential_id, 0.0), float(total["last_used_at"]))
        for credential_id, used_at in newest.items():
            if self._reported.get(credential_id, 0.0) >= used_at:
                continue
            try:
                self._broker.record_use(credential_id, used_at)
                self._reported[credential_id] = used_at
            except Exception as error:  # noqa: BLE001 - reported again on the next drain
                LOGGER.debug("A key's last use could not be recorded: %s", error)

    def _loop(self) -> None:
        while not self._stop.wait(self._interval):
            try:
                self.drain_once()
            except Exception:  # noqa: BLE001 - one bad drain must not end counting
                LOGGER.exception("The LLM Server usage drain failed.")

    def start(self) -> None:
        if self._thread is not None and self._thread.is_alive():
            return
        self._stop.clear()
        self._thread = threading.Thread(
            target=self._loop, name="llm-gate-usage", daemon=True
        )
        self._thread.start()

    def stop(self) -> None:
        self._stop.set()
        thread = self._thread
        if thread is not None:
            thread.join(timeout=self._interval + 5)
        self._thread = None
