"""Read vLLM's torch traces out of the profiles folder, as root, bounded (VD-214).

The on-demand profile (VD-128) reads the Chrome traces vLLM's torch profiler
writes on ``/stop_profile``. The folder (`serving_profiler.SERVING_PROFILES_DIR`)
is mounted read-write into the serving container, so what is in it is whatever
that container - or anything that took it over - chose to write, and the root
hardware bridge reads it (LESSONS 18: the boundary is who can write the folder,
not who reads it). So each file is:

* opened with ``O_NOFOLLOW | O_NONBLOCK`` - a link is refused by the open
  itself, and a FIFO cannot block the reader;
* accepted only when ``fstat`` says it is a regular file;
* STREAMED, never held whole: read a step at a time, unpacked a step at a
  time, and each ``traceEvents`` entry decoded on its own and folded into a
  :data:`summary <TraceSummary>` - per kernel name its device time and call
  count, and whether the trace holds GPU-side or worker events. Nothing else
  of the trace is kept.

**What bounds the memory, by construction.** At any moment the reader holds
one read step (:data:`_STEP`), one unpacked step, at most :data:`_SLICE` of
decoded text beyond one partial event of at most :data:`TRACE_EVENT_MAX_CHARS`,
and ONE summary for the whole capture, whose size is at most
:data:`TRACE_KERNEL_NAMES_MAX` names of at most :data:`TRACE_KERNEL_NAME_MAX_CHARS`
characters - every file is folded into it as it is read, so the bound is the
capture's, not each file's. An event over its cap, a trace nested deeper than
the decoder's stack, or a capture with more distinct kernel names is refused,
not kept. VD-216 housekeeping, measured on the Z2 (2026-10-14): the cap on
names was per file and every file's summary was kept, so 620 crafted files
(78 MB packed, 49,999 short names each) took the reader and the parser's merge
to a 12.6 GB peak in 67 s; one 5 KB gzip holding a 5 MiB event to 124 MB.

**What bounds the time.** Decoding is in C (``raw_decode``) but every event
still becomes a small Python object, so time grows with the unpacked size;
:data:`TRACE_INFLATED_MAX_BYTES` and :data:`TRACE_TOTAL_INFLATED_MAX_BYTES` cap
it, set from traces measured on the Z2 (below).

**Every refusal is said** (LESSONS 1, 8): :class:`TraceRead` carries the name
and the reason of each file it would not read, and the capture puts them in its
result as ``traces_refused``. A trace still being written (a gzip that has not
ended, an event array that has not closed) is neither read nor refused: it is
named in :attr:`TraceRead.pending`, and the capture's poll waits for it.
"""

from __future__ import annotations

import codecs
import json
import os
import stat
import time
import zlib
from dataclasses import dataclass, field
from typing import Any, Dict, Iterable, List, Optional

from .serving_profiler import GPU_KERNEL_TRACE_SUFFIX, _GPU_SIDE_TRACE_CATEGORIES, _as_int

_MIB = 1024 * 1024

#: Measured on the Z2 (rank0 = the worker's trace, which holds the kernels;
#: async_llm = the API server's), packed -> unpacked:
#: - the console's own profile under 8-user prompt-reading load (2026-10-13):
#:   rank0 3,501,176 B -> 55,240,911 B, 78 kernel names
#: - CPU-only product capture (2026-10-12): rank0 7,178,602 B -> 120,416,758 B
#: - GPU-traced bench captures, 3 requests (2026-10-12): rank0 21,568,539 B ->
#:   393,687,339 B and 26,646,101 B -> 466,265,489 B (the largest found, p029)
#: - GEAK's whole-model Run A profiles (2026-10-13): 2.3-2.4 MB -> 52-55 MB
#: - async_llm: up to 3,180,513 B -> 41,255,703 B
#: Every one held at most 82 distinct kernel names and no event over 1,223
#: characters; p029 and its API-server trace (507 MB unpacked) read in 2.9 s at
#: a 43 MB peak. VD-216 housekeeping (2026-10-14).

#: The most read from ONE trace file: about five times the 26.6 MB p029 rank0,
#: so a console profile under heavier load than any measured is still read.
TRACE_FILE_MAX_BYTES = 128 * _MIB

#: The most read from ALL of a capture's traces together: one rank0 at its cap
#: and as much again for the API server's and any other process's.
TRACE_TOTAL_MAX_BYTES = 2 * TRACE_FILE_MAX_BYTES

#: The most ONE trace may unpack to: about 4.6 times the 466 MB p029 rank0.
#: Bounds the time a trace takes to read, not the memory (above), which no
#: longer grows with it.
TRACE_INFLATED_MAX_BYTES = 2048 * _MIB

#: The most ALL of a capture's traces may unpack to: one rank0 at its cap and a
#: quarter of a GiB again (an async_llm trace unpacks to at most about 41 MB).
TRACE_TOTAL_INFLATED_MAX_BYTES = TRACE_INFLATED_MAX_BYTES + 256 * _MIB

#: The longest single ``traceEvents`` entry: 53 times the longest measured
#: (1,223 characters, a kernel launch with its arguments).
TRACE_EVENT_MAX_CHARS = 64 * 1024

#: How many distinct kernel names ONE CAPTURE may report, across all its
#: files. A serving model runs about 80 (measured above); a capture with
#: fifty times that was not written by a model.
TRACE_KERNEL_NAMES_MAX = 4096

#: A kernel name is counted under at most this many of its first characters.
TRACE_KERNEL_NAME_MAX_CHARS = 1024

#: How many ``traceEvents`` entries ONE CAPTURE may hold, across all its files:
#: about nine times the 1.81 million of p029 and its API-server trace. This,
#: not the unpacked size, is what bounds the time, since every entry costs the
#: decoder the same: on the Z2, 128 MiB of empty events (44.7 million) took
#: 12.6 s, so at the unpacked cap alone that shape would take minutes; 16
#: million short kernel events take about 12 s.
TRACE_EVENTS_MAX = 16_000_000

#: How many trace files one capture reads (a capture writes two: the worker's
#: and the API server's), and how many folder entries are looked at to find
#: them. Past either, the rest is refused in one line, never one line each.
TRACE_FILES_MAX = 32

#: The longest ONE read of a capture's traces may take, whatever they hold.
#: The caps above bound real traces (p029 and its API-server trace, 507 MB,
#: read in 2.9 s on the Z2); a crafted trace within every cap - events that
#: each straddle a decode slice - read in 49.6 s (VD-216 review R61). So time
#: is bounded by time: past this the file being read, and every one after it,
#: is refused and named. Seven times the largest real read.
TRACE_READ_BUDGET_SECONDS = 20.0

#: The clock the budget is measured on (a seam for the tests).
_clock = time.monotonic
TRACE_FOLDER_ENTRIES_MAX = 4096

#: How much is read, and unpacked, per step.
_STEP = 4 * _MIB

#: How much decoded text is handed to the event decoder at once, so what one
#: ``raw_decode`` can build is bounded by this and one event, not by a step.
_SLICE = 64 * 1024

#: The most read before the event array begins (the trace's header fields).
_HEADER_MAX_CHARS = 16 * _MIB

#: ``O_BINARY`` is Windows' alone (the suite's other platform), where a read
#: without it stops at the first 0x1A byte of a gzip.
_READ_FLAGS = (os.O_RDONLY | getattr(os, "O_NOFOLLOW", 0) | getattr(os, "O_NONBLOCK", 0)
               | getattr(os, "O_BINARY", 0))

_EVENTS_KEY = '"traceEvents"'
_WHITESPACE = " \t\r\n"
_DECODER = json.JSONDecoder()

_NOT_A_TRACE = "not read: not a Chrome trace"
_EVENT_TOO_BIG = "not read: an event is larger than {} KiB or malformed".format(
    TRACE_EVENT_MAX_CHARS // 1024)
_TOO_DEEP = "not read: an event is nested deeper than the reader will follow"
_TOO_MANY_NAMES = "not read: this capture's traces hold more than {} distinct kernel names"
_TOO_MANY_EVENTS = "not read: this capture's traces hold more than {:,} events"
_TOO_MANY_FILES = ("not read, with {} other file(s): a capture writes two traces, and this "
                   "folder holds more than {}")
_TIME_PASSED = "not read: reading this capture's traces passed {:g} s"
_FOLDER_TOO_FULL = ("not looked at: the profiles folder holds more than {} entries, so the "
                    "rest of it was not read")

#: What a file is refused with when it would unpack past its own cap, or past
#: what is left of the capture's total (review 2: the second named the
#: remainder as though it were this file's cap).
_ONE_UNPACKED = "not read: it would unpack to more than {} MiB"
_TOTAL_UNPACKED = "not read: this capture's traces together would unpack to more than {} MiB"

#: What a file the capture's total cap left unread is refused with.
_TOTAL_PASSED = "not read: this capture's traces together passed {} MiB"

#: One trace, reduced to what the kernel table needs:
#: ``{"kernels": {name: [device_us, calls]}, "gpu_side": bool, "worker": bool}``.
TraceSummary = Dict[str, Any]


@dataclass
class TraceRead:
    """The capture's traces folded into one summary, and every file refused with the reason.

    ``blobs`` holds that one summary, or nothing when no file was read whole.
    ``pending`` names the files still being written (a gzip that has not
    ended), which the capture's poll waits for (`serving_profiler`).
    """

    blobs: List[TraceSummary] = field(default_factory=list)
    refused: List[Dict[str, str]] = field(default_factory=list)
    pending: List[str] = field(default_factory=list)


class _Refused(ValueError):
    """A file this reader will not read, with the reason the result carries."""


def is_summary(value: Any) -> bool:
    return isinstance(value, dict) and isinstance(value.get("kernels"), dict)


def new_summary() -> TraceSummary:
    return {"kernels": {}, "gpu_side": False, "worker": False}


def add_event(summary: TraceSummary, event: Any, names_max: Optional[int] = None) -> None:
    """Fold one ``traceEvents`` entry into ``summary`` (the parser's own rules).

    ``names_max`` is how many distinct names ``summary`` may hold (the
    capture's remainder; :data:`TRACE_KERNEL_NAMES_MAX` when not given). A
    duration that is not a finite number is no duration (the event is skipped).
    """
    if not isinstance(event, dict):
        return
    category = event.get("cat")
    summary["gpu_side"] = summary["gpu_side"] or category in _GPU_SIDE_TRACE_CATEGORIES
    summary["worker"] = summary["worker"] or category == "cpu_op"
    if category != "kernel":
        return
    try:
        name, micros = event.get("name"), _as_int(event.get("dur"))
    except (ValueError, OverflowError):  # NaN and Infinity decode as floats
        return
    if not isinstance(name, str) or not name or micros is None:
        return
    key = name[:TRACE_KERNEL_NAME_MAX_CHARS]
    kernels = summary["kernels"]
    entry = kernels.get(key)
    if entry is None:
        if len(kernels) >= (TRACE_KERNEL_NAMES_MAX if names_max is None else names_max):
            raise _Refused(_TOO_MANY_NAMES.format(TRACE_KERNEL_NAMES_MAX))
        entry = kernels[key] = [0, 0]
    entry[0] += micros
    entry[1] += 1


def _fold(capture: TraceSummary, part: TraceSummary) -> None:
    """Add one file's summary into the capture's."""
    capture["gpu_side"] = capture["gpu_side"] or part["gpu_side"]
    capture["worker"] = capture["worker"] or part["worker"]
    kernels = capture["kernels"]
    for name, (micros, calls) in part["kernels"].items():
        entry = kernels.setdefault(name, [0, 0])
        entry[0] += micros
        entry[1] += calls


def summarise_events(events: Iterable[Any]) -> TraceSummary:
    """A summary of events already in memory; one with too many names is empty."""
    summary = new_summary()
    try:
        for event in events:
            add_event(summary, event)
    except _Refused:
        return new_summary()
    return summary


def read_traces(
    profiles_dir: str, skip: Any = (), *,
    file_limit: Optional[int] = None, total_limit: Optional[int] = None,
    inflated_limit: Optional[int] = None,
) -> TraceRead:
    """Every trace in ``profiles_dir`` but the names in ``skip``, streamed and bounded.

    The limits default to this module's constants, read at call time. Every
    budget is the CAPTURE's: what a file read, unpacked and decoded counts
    against it whether the file was then accepted or refused, so a folder of
    files each refused at its last byte costs no more than one capture.
    """
    per_file = TRACE_FILE_MAX_BYTES if file_limit is None else int(file_limit)
    total = TRACE_TOTAL_MAX_BYTES if total_limit is None else int(total_limit)
    inflated = TRACE_INFLATED_MAX_BYTES if inflated_limit is None else int(inflated_limit)
    result = TraceRead()
    deadline = _clock() + TRACE_READ_BUDGET_SECONDS
    names = _trace_names(profiles_dir, skip, result)
    over = names[TRACE_FILES_MAX:]
    if over:
        result.refused.append(_refusal(over[0], _TOO_MANY_FILES.format(
            len(over) - 1, TRACE_FILES_MAX)))
    spent = [0, 0, 0]  # bytes read, bytes unpacked, events decoded, across this capture
    capture: Optional[TraceSummary] = None  # every file read whole, folded together
    for name in names[:TRACE_FILES_MAX]:
        if _clock() > deadline:
            result.refused.append(_refusal(name, _TIME_PASSED.format(TRACE_READ_BUDGET_SECONDS)))
            continue
        budget = min(per_file, total - spent[0])
        unpack_budget = min(inflated, TRACE_TOTAL_INFLATED_MAX_BYTES - spent[1])
        if budget <= 0 or unpack_budget <= 0:
            result.refused.append(_refusal(name, _TOTAL_PASSED.format(total // _MIB)))
            continue
        names_left = TRACE_KERNEL_NAMES_MAX - (len(capture["kernels"]) if capture else 0)
        try:
            unpack_reason = (
                _TOTAL_UNPACKED.format(TRACE_TOTAL_INFLATED_MAX_BYTES // _MIB)
                if unpack_budget < inflated else _ONE_UNPACKED.format(inflated // _MIB))
            summary = _read_one(os.path.join(profiles_dir, name), budget, unpack_budget,
                                per_file, total, spent, unpack_reason, names_left, deadline)
        except _Refused as refused:
            result.refused.append(_refusal(name, str(refused)))
            continue
        except OSError:
            # A link (ELOOP from O_NOFOLLOW) or a file that vanished mid-read.
            result.refused.append(_refusal(name, "not read: it is a link or could not be opened"))
            continue
        if summary is None:
            result.pending.append(name)
        elif capture is None:
            capture = summary
        else:
            _fold(capture, summary)
    if capture is not None:
        result.blobs.append(capture)
    return result


def _trace_names(profiles_dir: str, skip: Any, result: TraceRead) -> List[str]:
    """The trace names to read, sorted; at most :data:`TRACE_FOLDER_ENTRIES_MAX` entries looked at."""
    names: List[str] = []
    try:
        with os.scandir(profiles_dir) as entries:
            for seen, entry in enumerate(entries):
                if seen >= TRACE_FOLDER_ENTRIES_MAX:
                    result.refused.append(_refusal(
                        "(the rest of the folder)", _FOLDER_TOO_FULL.format(TRACE_FOLDER_ENTRIES_MAX)))
                    break
                if entry.name.endswith(GPU_KERNEL_TRACE_SUFFIX) and entry.name not in skip:
                    names.append(entry.name)
    except OSError:
        return []
    return sorted(names)


def _refusal(name: str, reason: str) -> Dict[str, str]:
    return {"name": name, "reason": reason}


def _read_one(path: str, budget: int, unpack_budget: int, per_file: int, total: int,
              spent: List[int], unpack_reason: str,
              names_max: Optional[int] = None,
              deadline: Optional[float] = None) -> Optional[TraceSummary]:
    """One trace streamed into a summary; ``None`` while it is still being written.

    What it read, unpacked and decoded is added to ``spent`` however it ends.
    """
    events = _EventStream(unpack_budget, unpack_reason, names_max,
                          TRACE_EVENTS_MAX - spent[2], deadline)
    read = 0
    handle = os.open(path, _READ_FLAGS)
    try:
        if not stat.S_ISREG(os.fstat(handle).st_mode):
            raise _Refused("not read: not a regular file")
        inflater = None
        while True:
            chunk = os.read(handle, _STEP)
            if not chunk:
                break
            read += len(chunk)
            if read > budget:
                raise _Refused(_TOTAL_PASSED.format(total // _MIB) if budget < per_file
                               else "not read: larger than {} MiB".format(per_file // _MIB))
            if inflater is None:
                gzipped = chunk[:2] == b"\x1f\x8b"
                inflater = zlib.decompressobj(16 + zlib.MAX_WBITS) if gzipped else False
            if inflater is False:
                events.feed(chunk)
            elif not _inflate(inflater, chunk, events):
                break  # the gzip ended
            if events.done:
                break
        if events.done:
            return events.summary
        if inflater not in (None, False) and inflater.eof:
            # The whole gzip, and still no closed event list: not a trace.
            raise _Refused("not read: not a complete Chrome trace")
        return None  # cut short: the writer has not finished
    finally:
        os.close(handle)
        spent[0] += read
        spent[1] += events.unpacked
        spent[2] += events.count


def _inflate(inflater: Any, chunk: bytes, events: "_EventStream") -> bool:
    """Unpack ``chunk`` a step at a time into ``events``; ``False`` once the gzip ended."""
    pending = chunk
    try:
        while pending and not inflater.eof and not events.done:
            events.feed(inflater.decompress(pending, _STEP))
            pending = inflater.unconsumed_tail
    except zlib.error:
        raise _Refused("not read: not a readable gzip") from None
    return not inflater.eof


class _EventStream:
    """Unpacked bytes in, one ``traceEvents`` entry at a time out, into a summary."""

    def __init__(self, limit: int, reason: str, names_max: Optional[int] = None,
                 events_max: Optional[int] = None, deadline: Optional[float] = None):
        self.limit = limit
        self.reason = reason
        self.names_max = names_max
        self.events_max = TRACE_EVENTS_MAX if events_max is None else events_max
        self.count = 0  # events decoded
        self.deadline = deadline
        self.unpacked = 0
        self.summary = new_summary()
        self.done = False
        self._text = codecs.getincrementaldecoder("utf-8")(errors="replace")
        self._buffer = ""
        self._seen = 0  # characters dropped while looking for the array
        self._in_array = False

    def feed(self, data: bytes) -> None:
        if self.done or not data:
            return
        if self.deadline is not None and _clock() > self.deadline:
            raise _Refused(_TIME_PASSED.format(TRACE_READ_BUDGET_SECONDS))
        self.unpacked += len(data)
        if self.unpacked > self.limit:
            raise _Refused(self.reason)
        text = self._text.decode(data)
        # A slice at a time: the decoder never sees more than one partial
        # event and one slice, so a step of tiny nested values cannot become
        # a step's worth of Python objects at once.
        for at in range(0, len(text), _SLICE):
            if self.done:
                return
            self._buffer += text[at:at + _SLICE]
            if self._in_array or self._find_array():
                self._drain()

    def _find_array(self) -> bool:
        at = self._buffer.find(_EVENTS_KEY)
        if at < 0:
            keep = len(_EVENTS_KEY)
            self._seen += max(0, len(self._buffer) - keep)
            self._buffer = self._buffer[-keep:]
            if self._seen > _HEADER_MAX_CHARS:
                raise _Refused("not read: no event list in its first {} MiB".format(_HEADER_MAX_CHARS // _MIB))
            return False
        rest = self._buffer[at + len(_EVENTS_KEY):].lstrip(_WHITESPACE)
        if not rest.startswith(":"):
            if rest:
                raise _Refused(_NOT_A_TRACE)
            return False
        rest = rest[1:].lstrip(_WHITESPACE)
        if not rest:
            return False
        if not rest.startswith("["):
            raise _Refused(_NOT_A_TRACE)
        self._buffer = rest[1:]
        self._in_array = True
        return True

    def _drain(self) -> None:
        buffer, at, end = self._buffer, 0, len(self._buffer)
        while True:
            while at < end and buffer[at] in _WHITESPACE + ",":
                at += 1
            if at >= end:
                break
            if buffer[at] == "]":
                self.done = True
                break
            if buffer[at] != "{":
                raise _Refused(_NOT_A_TRACE)
            try:
                event, after = _DECODER.raw_decode(buffer, at)
            except RecursionError:
                raise _Refused(_TOO_DEEP) from None
            except ValueError:
                if end - at > TRACE_EVENT_MAX_CHARS:
                    raise _Refused(_EVENT_TOO_BIG) from None
                break  # the rest of this event has not arrived yet
            if after - at > TRACE_EVENT_MAX_CHARS:
                raise _Refused(_EVENT_TOO_BIG)
            self.count += 1
            if self.count > self.events_max:
                raise _Refused(_TOO_MANY_EVENTS.format(TRACE_EVENTS_MAX))
            add_event(self.summary, event, self.names_max)
            at = after
        self._buffer = buffer[at:]
