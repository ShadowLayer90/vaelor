"""Ask the configured model endpoint whether it is actually there.

Alpha 12 reported "MODEL READY" and "CONNECTED MODEL" from configuration alone.
An endpoint that was switched off, moved, or never reachable from the appliance
still rendered as verified health, and the first sign of trouble was an agent
run that failed a minute later blaming the user's choice of model.
"""

from __future__ import annotations

import json
import threading
import time
import urllib.error
import urllib.request
from typing import Any, Dict, Mapping, Optional, Sequence

from .inference_tuning import detect_model_eviction, loaded_models, offered_models

# Long enough to survive a slow first response, short enough that a dead
# endpoint does not hold an API request open.
PROBE_TIMEOUT_SECONDS = 2.5

# A probe result is reused for this long. Status is read on nearly every screen
# paint; without a cache a healthy endpoint would field a request per render.
CACHE_SECONDS = 15.0

_CACHE: Dict[str, Dict[str, Any]] = {}
_LOCK = threading.Lock()


def _cache_key(connection: Mapping[str, str]) -> str:
    return "{}|{}".format(
        connection.get("base_url", ""), connection.get("model", "")
    )


def _timed_out(timeout_seconds: Optional[float], subject: str) -> str:
    """Name the request that timed out, and only a number it actually used.

    This said *"The model server did not answer within 2.5 seconds"* on every
    timeout anywhere, because it read :data:`PROBE_TIMEOUT_SECONDS` - the
    readiness probe's own budget - regardless of which request had failed. A
    live tester watched an Assistant question run for 125 seconds beside a
    banner promising 2.5, which reads as a broken timeout and is actually a
    banner describing a different request. A caller that does not know its own
    budget now states no figure rather than borrowing one.
    """
    if timeout_seconds is None:
        return "{} did not answer in time.".format(subject)
    return "{} did not answer within {:g} seconds.".format(
        subject, timeout_seconds
    )


def describe_endpoint_failure(
    error: Exception,
    timeout_seconds: Optional[float] = None,
    subject: str = "The model server",
) -> str:
    """Explain a failed model request in the user's terms, never a raw traceback.

    Shared rather than private: the Ask path reported this as
    ``modelUnreachableReason`` while an agent run reported a bare HTTP status
    and a URL, so the same outage read as a model problem on one tab and as the
    user's own mistake on the other. One endpoint, one explanation.

    ``timeout_seconds`` is the budget of *the request that failed*. It has no
    default figure on purpose: the only caller entitled to name a number is the
    one that set it.
    """
    if isinstance(error, urllib.error.HTTPError):
        if error.code in (401, 403):
            return "The model server refused the credentials Vaelor has for it."
        return "The model server answered with HTTP {}.".format(error.code)
    if isinstance(error, TimeoutError):
        return _timed_out(timeout_seconds, subject)
    reason = getattr(error, "reason", None)
    if isinstance(reason, Exception):
        error = reason
    if isinstance(error, TimeoutError):
        return _timed_out(timeout_seconds, subject)
    text = str(error)
    if "refused" in text.lower():
        return "Nothing is listening at that address."
    if "name or service not known" in text.lower() or "name resolution" in text.lower():
        return "That host name could not be resolved from this appliance."
    return "The model server could not be reached."


def probe_connection(
    connection: Optional[Mapping[str, str]], *, now: Optional[float] = None
) -> Dict[str, Any]:
    """Return `{reachable, detail, endpoint, checked_at}` for one connection."""
    if not connection or not connection.get("base_url"):
        return {
            "reachable": False, "endpoint": "",
            "detail": "No AI model is selected yet.", "checked_at": 0.0,
        }
    moment = time.time() if now is None else now
    key = _cache_key(connection)
    # A run that actually failed outranks a socket that happens to open.
    unusable = inference_failure(connection, now=moment)
    if unusable:
        return {
            "reachable": False, "endpoint": str(connection["base_url"]).rstrip("/"),
            "detail": unusable, "checked_at": moment,
        }
    with _LOCK:
        cached = _CACHE.get(key)
        if cached and moment - cached["checked_at"] < CACHE_SECONDS:
            return dict(cached)

    endpoint = str(connection["base_url"]).rstrip("/")
    headers = {"Accept": "application/json"}
    if connection.get("api_key"):
        headers["Authorization"] = "Bearer {}".format(connection["api_key"])
    result: Dict[str, Any] = {
        "reachable": False, "endpoint": endpoint,
        "detail": "", "checked_at": moment, "loaded_models": [],
        # Reachable and offering nothing is a third state, not a synonym for
        # unreachable. A tester saw the Assistant report MODEL UNREACHABLE at
        # the same moment AI Chat correctly said the server "is reachable but
        # is not offering any model" - and AI Chat was right. `None` means the
        # question was not reached, which is not the same as "none offered".
        "offering_models": None, "model_availability_reason": "",
        # The offered model *names*, kept beside the `offering_models` bool. A
        # managed-local server deployed with no stored model preference has an
        # empty `selected_model`, so its name is only knowable from what the
        # server offers here - never from `loaded_models`, which is [] on the
        # Pi's llama.cpp build (#205 Step 3).
        "offered_model_names": [],
    }
    try:
        request = urllib.request.Request(endpoint + "/models", headers=headers)
        with urllib.request.urlopen(request, timeout=PROBE_TIMEOUT_SECONDS) as response:
            listing = json.loads(response.read(256 * 1024).decode("utf-8", "replace"))
        result["reachable"] = True
        # Which models the server says are *loaded*, not merely offered. This is
        # the only evidence available that a second tier evicted the first: a
        # server whose `max_loaded_models` is 1 unloads silently and reports
        # success, so nothing else in the exchange records it.
        result["loaded_models"] = loaded_models(listing)
        offered = offered_models(listing)
        result["offering_models"] = bool(offered)
        result["offered_model_names"] = list(offered)
        if not result["offering_models"]:
            result["model_availability_reason"] = (
                "The model server is reachable but is not offering any model, "
                "so there is nothing for Vaelor to send a request to."
            )
    except (OSError, ValueError, urllib.error.URLError) as error:
        result["detail"] = describe_endpoint_failure(
            error, PROBE_TIMEOUT_SECONDS, "The readiness check"
        )

    with _LOCK:
        _CACHE[key] = dict(result)
    return dict(result)


def evicted_models(
    connection: Optional[Mapping[str, str]],
    expected: Sequence[str],
    *,
    exit_code: int = 0,
    now: Optional[float] = None,
) -> Dict[str, Any]:
    """Report tiers that were loaded on this endpoint and are not any more.

    Vaelor's two-tier design keeps an assistant model resident on the neural
    processor and a chat model on the GPU. A model server left at
    ``max_loaded_models: 1`` unloads the first when the second loads, returns
    ``rc 0``, and warns about nothing — a destructive action reporting success,
    which is precisely the failure class this project has spent weeks removing.
    Vaelor sets the value; this is what catches it when the value did not take.
    """
    probe = probe_connection(connection, now=now)
    if not probe.get("reachable"):
        return {
            "evicted": [], "ok": True, "exit_code": int(exit_code),
            "loaded": [], "message": "",
            "unknown": True, "detail": probe.get("detail", ""),
        }
    report = detect_model_eviction(
        expected, {"data": [
            {"id": name, "loaded": True} for name in probe.get("loaded_models", [])
        ]}, exit_code=exit_code,
    )
    report["unknown"] = False
    return report


def reset_cache() -> None:
    """Drop memoised probe results (used by tests and after a model change)."""
    with _LOCK:
        _CACHE.clear()
        _FAILURES.clear()


# Answering `/models` is not the same as answering a question. A server that
# lists its models instantly and then times out on every request was still
# being advertised as "MODEL READY", so the badge stayed green through a
# hundred per cent failure rate. Real inference outcomes are recorded here and
# folded into readiness.
CONSECUTIVE_FAILURES_BEFORE_UNUSABLE = 2
FAILURE_MEMORY_SECONDS = 900.0

_FAILURES: Dict[str, Dict[str, Any]] = {}


def note_inference_outcome(
    connection: Optional[Mapping[str, str]], *, ok: bool, detail: str = "",
    now: Optional[float] = None,
) -> None:
    """Record whether a real request to this endpoint produced an answer."""
    if not connection or not connection.get("base_url"):
        return
    moment = time.time() if now is None else now
    key = _cache_key(connection)
    with _LOCK:
        if ok:
            _FAILURES.pop(key, None)
            return
        entry = _FAILURES.get(key)
        if entry is None or moment - entry["at"] > FAILURE_MEMORY_SECONDS:
            entry = {"count": 0}
        entry["count"] += 1
        entry["at"] = moment
        entry["detail"] = detail[:300]
        _FAILURES[key] = entry


def recorded_failures(
    base_url: str, *, now: Optional[float] = None
) -> Dict[str, str]:
    """Models at this endpoint that have actually failed, and why.

    A model picker that lists everything the server advertises sends users into
    an HTTP 400 they cannot predict. Nothing here probes or guesses: these are
    real requests that really failed, keyed by the model they were sent to, so a
    surface can mark a known-bad choice before the user commits to it. A model
    that starts answering clears its own entry.
    """
    endpoint = str(base_url or "").strip()
    if not endpoint:
        return {}
    moment = time.time() if now is None else now
    prefix = endpoint + "|"
    failures: Dict[str, str] = {}
    with _LOCK:
        entries = list(_FAILURES.items())
    for key, entry in entries:
        if not key.startswith(prefix):
            continue
        model = key[len(prefix):]
        if not model or moment - entry["at"] > FAILURE_MEMORY_SECONDS:
            continue
        failures[model] = entry.get("detail") or (
            "The last {} requests to this model did not produce an answer.".format(
                entry["count"]
            )
        )
    return failures


def inference_failure(
    connection: Optional[Mapping[str, str]], *, now: Optional[float] = None
) -> str:
    """Explain why this endpoint is not usable, or "" when it is."""
    if not connection or not connection.get("base_url"):
        return ""
    moment = time.time() if now is None else now
    with _LOCK:
        entry = _FAILURES.get(_cache_key(connection))
        if not entry or entry["count"] < CONSECUTIVE_FAILURES_BEFORE_UNUSABLE:
            return ""
        if moment - entry["at"] > FAILURE_MEMORY_SECONDS:
            return ""
        return entry.get("detail") or (
            "The model server accepts connections but has not answered the last "
            "{} requests.".format(entry["count"])
        )
