"""The draft model's acceptance in the on-demand serving profile (VD-213, VD-128).

While a vLLM replica serves with a draft model (speculative decoding), vLLM
counts every draft it verifies on its Prometheus ``/metrics``. This capture
reads that endpoint once when the profile window opens and once when it closes,
through the same injected ``run`` seam and the same ``curl`` the torch-profiler
capture uses, and reports what the drafter did in between: how many drafts,
how many guessed tokens, how many were kept, the tokens kept per step and how
often each guess position is right.

**The metric names are vLLM's own, read from the pinned image** (LESSONS 15,
not guessed): ``vllm/v1/spec_decode/metrics.py`` in
``rocm/vllm@sha256:b8a082f3...`` (vLLM 0.27.1.dev5, prometheus_client 0.26.0)
declares the counters ``vllm:spec_decode_num_drafts``,
``vllm:spec_decode_num_draft_tokens``, ``vllm:spec_decode_num_accepted_tokens``
and ``vllm:spec_decode_num_accepted_tokens_per_pos`` (labelled ``position``,
one series per lookahead slot ``0 .. num_speculative_tokens - 1``), each
labelled ``model_name`` and ``engine``. prometheus_client exposes a counter as
``<name>_total`` (and a ``<name>_created`` gauge this ignores). vLLM only
creates them when a speculative config is set, so their absence is the truthful
"no draft model is serving". Its own log line defines the two derived figures
this reports: mean acceptance length ``1 + accepted / drafts`` (the ``1`` is the
token the main model writes itself every step) and the per-position rate
``accepted_per_pos[i] / drafts``.

Pure apart from the injected ``run`` and ``clock``: nothing here opens a socket,
so the whole flow is tested with real-shaped ``/metrics`` text.
"""

from __future__ import annotations

import math
from typing import Any, Callable, Dict, List, Mapping, Optional, Sequence

from .serving_metrics import counter_increase

CAPTURE_SPEC = "speculative_decoding"
SPEC_LABEL = "Draft model acceptance (vLLM metrics)"

#: The exposed sample names (vLLM's counter name + prometheus_client's
#: ``_total``). One home, so the parser and the tests cannot drift.
SPEC_DRAFTS_SERIES = "vllm:spec_decode_num_drafts_total"
SPEC_DRAFT_TOKENS_SERIES = "vllm:spec_decode_num_draft_tokens_total"
SPEC_ACCEPTED_SERIES = "vllm:spec_decode_num_accepted_tokens_total"
SPEC_PER_POSITION_SERIES = "vllm:spec_decode_num_accepted_tokens_per_pos_total"
_TOTALS = {
    SPEC_DRAFTS_SERIES: "drafts",
    SPEC_DRAFT_TOKENS_SERIES: "draft_tokens",
    SPEC_ACCEPTED_SERIES: "accepted",
}
_POSITION_LABEL = 'position="'
#: More digits than this is not a lookahead slot (VD-213 caps the lookahead at 15).
_POSITION_DIGITS = 2

#: Each ``/metrics`` read is a short loopback GET; bounded so two reads add at
#: most this twice to a profile whose socket ceiling is fixed.
SPEC_METRICS_TIMEOUT_SECONDS = 5

#: The only shape this capture ever runs: ``curl -sS http://127.0.0.1:<port>/metrics``.
#: The port comes from the root bridge's own mode file, never from the caller.
_LOOPBACK = "http://127.0.0.1:{}/metrics"

SPEC_NEEDS_VLLM = (
    "Draft model acceptance is read from the vLLM cluster engine's metrics; the "
    "single-node llama.cpp engine (Mode A) is not serving through vLLM, so there "
    "is nothing to read here."
)
SPEC_NO_ENDPOINT = (
    "The local vLLM replica's metrics did not answer on its loopback serving "
    "port, so the draft model's acceptance could not be read; nothing was "
    "fabricated."
)
SPEC_NO_DRAFTER = (
    "No draft model is serving: vLLM's metrics carry no speculative-decoding "
    "counters, which it only creates when a draft model is configured."
)
SPEC_NO_DRAFTS = (
    "A draft model is serving, but it made no guesses during the window (no "
    "reply was being written), so there is no acceptance to report."
)
SPEC_END_UNREAD = (
    "The window's closing metrics read did not answer, so there is no end "
    "reading to compare against; nothing was fabricated."
)
SPEC_RESTARTED_NOTE = (
    "The serving engine restarted during the window, so these figures count "
    "from its restart, not from the window's start."
)
SPEC_OWN_REQUEST_NOTE = (
    "The window can include the profile's own short test request (the GPU "
    "kernel trace drives one), so on an otherwise idle box these figures may "
    "describe only that request."
)
SPEC_POSITION_NOTE = (
    "Each position's rate is per draft, as vLLM defines it: when the lookahead "
    "changes with load, later positions are guessed on fewer drafts, which also "
    "lowers their rate."
)


def metrics_argv(port: Any, curl_path: str) -> List[str]:
    """The one argv this capture runs, for a loopback port; refuses anything else.

    The URL shape is fixed here, at the privileged side (LESSONS 18): host
    ``127.0.0.1``, path ``/metrics``, and an integer TCP port. A bool, a string
    or an out-of-range number raises rather than building a different URL.
    """
    if isinstance(port, bool) or not isinstance(port, int) or not 0 < port < 65536:
        raise ValueError("not a TCP port: {!r}".format(port))
    return [curl_path, "-sS", _LOOPBACK.format(port)]


def parse_spec_counters(text: Any) -> Optional[Dict[str, Any]]:
    """The cumulative drafter counters in one ``/metrics`` body, or ``None``.

    Sums each counter over every series (one per engine and model), and the
    per-position counter per ``position``. ``None`` when the body carries none
    of the three totals - vLLM only creates them with a draft model - so absence
    is never read as zero. A malformed or non-finite value is skipped; anything
    that is not text yields ``None`` and never raises.
    """
    if not isinstance(text, str):
        return None
    totals: Dict[str, float] = {}
    positions: Dict[int, float] = {}
    for raw in text.splitlines():
        line = raw.strip()
        if not line or line.startswith("#"):
            continue
        fields = _split_labelled(line) if "}" in line else line.split()[:2]
        if fields is None or len(fields) != 2:
            continue
        series, raw_value = fields
        if "{" in series and not series.endswith("}"):
            continue  # an unclosed label set is not a sample
        name = series.split("{", 1)[0]
        value = _counter_value(raw_value)
        if value is None:
            continue
        if name in _TOTALS:
            key = _TOTALS[name]
            totals[key] = totals.get(key, 0.0) + value
        elif name == SPEC_PER_POSITION_SERIES:
            index = _position(series)
            if index is not None:
                positions[index] = positions.get(index, 0.0) + value
    if not totals:
        return None
    return {
        "drafts": totals.get("drafts", 0.0),
        "draft_tokens": totals.get("draft_tokens", 0.0),
        "accepted": totals.get("accepted", 0.0),
        "positions": positions,
    }


def _position(series: str) -> Optional[int]:
    """The ``position`` label of one per-position series, or ``None`` if unreadable."""
    start = series.find(_POSITION_LABEL)
    if start == -1:
        return None
    start += len(_POSITION_LABEL)
    end = series.find('"', start)
    raw = series[start:end] if end != -1 else ""
    # ``str.isdigit`` alone accepts "²" or "٣", which ``int`` then rejects or
    # reads as a digit vLLM never wrote: ASCII only, and no more digits than a
    # real lookahead slot (vLLM numbers them 0 .. num_speculative_tokens - 1).
    if not (raw.isascii() and raw.isdigit() and len(raw) <= _POSITION_DIGITS):
        return None
    return int(raw)


def _counter_value(raw: str) -> Optional[float]:
    """One counter sample's value, or ``None`` for anything a counter cannot be.

    ``float`` alone is too kind for a parser that must never fabricate: it reads
    "٣" (a non-ASCII digit) as 3.0 and "1_0" as 10.0. Only ASCII text is read,
    without underscores, and a non-finite or negative result is dropped (a
    Prometheus counter never goes below zero).
    """
    if not raw.isascii() or "_" in raw:
        return None
    try:
        value = float(raw)
    except (ValueError, OverflowError):
        return None
    return value if math.isfinite(value) and value >= 0 else None


def _split_labelled(line: str) -> Optional[List[str]]:
    """``[series, value]`` for a labelled sample (a label value may hold spaces;
    a trailing timestamp is ignored)."""
    end = line.rfind("}")
    rest = line[end + 1:].split()
    return [line[:end + 1], rest[0]] if rest else None


def window_delta(before: Mapping[str, Any], after: Mapping[str, Any]) -> Dict[str, Any]:
    """What the drafter did between two readings, with ``restarted`` if it restarted.

    Uses the project's one counter reading (:func:`serving_metrics.counter_increase`,
    Prometheus's ``increase``), so a counter lower than before - an engine
    restart - counts the whole new reading rather than a negative number.
    """
    restarted = False
    out: Dict[str, Any] = {}
    for key in ("drafts", "draft_tokens", "accepted"):
        out[key], reset = counter_increase(float(before.get(key, 0.0)), float(after.get(key, 0.0)))
        restarted = restarted or reset
    earlier = before.get("positions") or {}
    positions: Dict[int, float] = {}
    for index, value in (after.get("positions") or {}).items():
        positions[index], reset = counter_increase(float(earlier.get(index, 0.0)), float(value))
        restarted = restarted or reset
    out["positions"] = positions
    out["restarted"] = restarted
    return out


def summarise(delta: Mapping[str, Any], *, window_seconds: float) -> Dict[str, Any]:
    """The capture's result from a window delta that saw at least one draft."""
    drafts = int(delta["drafts"])
    drafted = int(delta["draft_tokens"])
    accepted = int(delta["accepted"])
    per_draft = accepted / drafts
    result: Dict[str, Any] = {
        "kind": CAPTURE_SPEC,
        "window_seconds": round(float(window_seconds), 1),
        "drafts": drafts,
        "drafted_tokens": drafted,
        "accepted_tokens": accepted,
        "accepted_per_step": round(per_draft, 2),
        "mean_acceptance_length": round(1.0 + per_draft, 2),
        "positions": [
            {"position": index + 1, "accepted": int(count),
             "rate": round(count / drafts, 3)}
            for index, count in sorted(delta["positions"].items())
        ],
    }
    if drafted > 0:
        result["acceptance_rate"] = round(accepted / drafted, 3)
    return result


class SpecWindow:
    """One capture window: the opening reading, then :meth:`close` on the profile."""

    def __init__(self, *, reason: str = "", run: Optional[Callable[..., Any]] = None,
                 port: Optional[int] = None, before: Optional[Dict[str, Any]] = None,
                 started: float = 0.0, curl_path: str = "curl",
                 clock: Optional[Callable[[], float]] = None):
        self.reason, self.run, self.port, self.before = reason, run, port, before
        self.started, self.curl_path, self.clock = started, curl_path, clock

    def close(self, profile: Dict[str, Any]) -> Dict[str, Any]:
        """Take the closing reading and append this capture's entry to ``profile``."""
        entry = {"id": CAPTURE_SPEC, "label": SPEC_LABEL, "available": False,
                 "reason": self.reason}
        if not self.reason:
            try:
                entry.update(self._measure(profile))
            except Exception:  # noqa: BLE001 - one capture must never crash a profile
                entry["reason"] = SPEC_END_UNREAD
        profile.setdefault("captures", []).append(entry)
        return profile

    def _measure(self, profile: Mapping[str, Any]) -> Dict[str, Any]:
        body = _read(self.run, self.port, self.curl_path)
        if body is None:
            return {"reason": SPEC_END_UNREAD}
        after = parse_spec_counters(body)
        if after is None:
            return {"reason": SPEC_NO_DRAFTER}
        delta = window_delta(self.before or {}, after)
        if delta["drafts"] <= 0:
            return {"reason": SPEC_NO_DRAFTS}
        result = summarise(delta, window_seconds=self.clock() - self.started)
        notes = [SPEC_RESTARTED_NOTE] if delta["restarted"] or self.before is None else []
        if _kernel_trace_ran(profile):
            notes.append(SPEC_OWN_REQUEST_NOTE)
        if len(result["positions"]) > 1:
            notes.append(SPEC_POSITION_NOTE)
        if notes:
            result["note"] = " ".join(notes)
        return {"available": True, "reason": "", "result": result}


def open_window(
    probe: Mapping[str, Any], run: Callable[..., Any], ports: Sequence[int],
    curl_path: str, clock: Callable[[], float], no_process_reason: str,
) -> SpecWindow:
    """Take the opening reading, or record why this capture cannot run.

    Runs only with a serving PID and the vLLM replica's loopback ports (Mode B);
    the first port whose ``/metrics`` answers with a vLLM body is the one read
    again at the close, so both readings come from the same process.
    """
    serving_pid = probe.get("serving_pid")
    if not (isinstance(serving_pid, int) and serving_pid > 0):
        return SpecWindow(reason=no_process_reason)
    if not ports:
        return SpecWindow(reason=SPEC_NEEDS_VLLM)
    try:
        for port in ports:
            body = _read(run, port, curl_path)
            if body is None:
                continue
            return SpecWindow(run=run, port=int(port), before=parse_spec_counters(body),
                              started=clock(), curl_path=curl_path, clock=clock)
    except Exception:  # noqa: BLE001 - one capture must never crash a profile
        pass
    return SpecWindow(reason=SPEC_NO_ENDPOINT)


def _read(run: Callable[..., Any], port: Any, curl_path: str) -> Optional[str]:
    """One ``/metrics`` body from a vLLM replica, or ``None`` if it did not answer as one."""
    answer = run(metrics_argv(port, curl_path), timeout=SPEC_METRICS_TIMEOUT_SECONDS)
    body = getattr(answer, "stdout", "")
    if getattr(answer, "returncode", 1) != 0 or not isinstance(body, str):
        return None
    # curl without -f exits 0 on an HTTP error page; only a vLLM body counts.
    return body if "\nvllm:" in "\n" + body else None


def _kernel_trace_ran(profile: Mapping[str, Any]) -> bool:
    """Whether this profile's GPU kernel capture drove its own short request.

    Read from the fact the capture records when it sends the completion
    (``request_driven``), never inferred from its note, its rows or its
    availability: a trace with no GPU events still drove the request (VD-165).
    """
    from .serving_profiler import CAPTURE_GPU_KERNEL

    for capture in profile.get("captures", []):
        result = capture.get("result")
        if capture.get("id") == CAPTURE_GPU_KERNEL and isinstance(result, Mapping):
            return result.get("request_driven") is True
    return False
