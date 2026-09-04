"""Keep measured model profiles, and answer budget questions from them.

Calibration is a network round trip that can take minutes on a reasoning model,
so it never happens inside a user's request. It runs when a model is selected,
tested, or explicitly re-measured by an operator, and everything a request path
asks here is a dictionary lookup with a documented fallback.
"""

from __future__ import annotations

import json
import logging
import math
import os
import threading
import time
import urllib.error
from pathlib import Path
from typing import Any, Dict, Mapping, Optional

from .model_calibration import (
    MANAGED_LOCAL_MAX_TOKENS,
    MAX_AGENT_TOKENS,
    PROBE_TIMEOUT_SECONDS,
    PROFILE_MAX_AGE_SECONDS,
    REASONING_MARGIN,
    derive_profile,
    measure_connection,
    profile_key,
    response_format_for,
    unmeasured_profile,
)
from .runtime_paths import state_path

LOGGER = logging.getLogger(__name__)

_LOCK = threading.Lock()
_PROFILES: Dict[str, Dict[str, Any]] = {}
_LOADED_FROM: Optional[str] = None
_IN_FLIGHT: set[str] = set()
# Why the last measurement of an endpoint did not happen, in memory only.
# `/agent/status` reports this profile, and "not measured yet" with no reason
# is the difference between an operator who knows to retry and one who does
# not. It is deliberately not persisted: it describes one attempt against one
# server, and a restart is entitled to try again.
_FAILURES: Dict[str, str] = {}
# Structured-output contracts an endpoint has been *observed* to refuse, keyed
# by profile. Written only when the identical request succeeded with the field
# removed, so it records a server's answer rather than a guess about one.
# In memory only, for the same reason as `_FAILURES`: it describes one server's
# behaviour right now, and a restart is entitled to ask again.
_REJECTED_FORMATS: Dict[str, set[str]] = {}

# Bounded so a long-lived appliance that has tried many endpoints cannot grow
# this file without limit.
MAX_STORED_PROFILES = 40

# Every number a stored profile promises, because each one reaches an `int()`
# or a `float()` on a request path.
_NUMERIC_FIELDS = ("measured_at", "max_tokens", "reasoning_ratio", "timeout_seconds")


def store_path() -> str:
    """Where measured profiles live between restarts."""
    return os.environ.get("VAELOR_MODEL_PROFILE_PATH", "").strip() or state_path(
        "model-profiles.json"
    )


def _usable(value: Any) -> bool:
    """Whether a stored entry can answer a budget question without raising.

    Admitting any dict with a truthy `measured` was not tolerating an unusable
    file, it was deferring it: a profile whose numbers were strings sailed
    through here and raised `ValueError` inside `float()` on a live request,
    two calls away from anything that knew what the file was. An entry that
    does not carry real, finite numbers is not a measurement, so it is treated
    as the absence of one.
    """
    if not isinstance(value, dict) or not value.get("measured"):
        return False
    for field in _NUMERIC_FIELDS:
        number = value.get(field)
        if isinstance(number, bool) or not isinstance(number, (int, float)):
            return False
        try:
            # An integer too large to be a float overflows the conversion, and
            # a JSON `1e400` parses straight to infinity.
            if not math.isfinite(float(number)):
                return False
        except OverflowError:
            return False
    return True


def _load() -> Dict[str, Dict[str, Any]]:
    """Read persisted profiles once per path, tolerating any unusable file."""
    global _LOADED_FROM
    path = store_path()
    if _LOADED_FROM == path:
        return _PROFILES
    _PROFILES.clear()
    try:
        raw = json.loads(Path(path).read_text(encoding="utf-8"))
        for key, value in (raw or {}).items():
            if _usable(value):
                _PROFILES[str(key)] = value
    except (AttributeError, OSError, TypeError, ValueError):
        # A missing or corrupt profile file means "not measured yet", which is
        # a state this module already handles. It must never fail a request.
        _PROFILES.clear()
    _LOADED_FROM = path
    return _PROFILES


def _persist() -> None:
    path = Path(store_path())
    try:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(
            json.dumps(_PROFILES, separators=(",", ":"), sort_keys=True),
            encoding="utf-8",
        )
    except OSError as error:
        # An appliance with a read-only or unwritable state directory still
        # benefits from the in-memory profile for the life of the process.
        LOGGER.warning("Model profile could not be saved to %s: %s", path, error)


def remember(profile: Mapping[str, Any]) -> Dict[str, Any]:
    """Record one measured profile and make it survive a restart."""
    stored = dict(profile)
    with _LOCK:
        _load()
        key = profile_key(stored)
        # A measurement answers the question the recorded failure was standing
        # in for, so the excuse goes with it.
        _FAILURES.pop(key, None)
        _PROFILES[key] = stored
        if len(_PROFILES) > MAX_STORED_PROFILES:
            for key, _value in sorted(
                _PROFILES.items(), key=lambda item: item[1].get("measured_at", 0.0)
            )[: len(_PROFILES) - MAX_STORED_PROFILES]:
                _PROFILES.pop(key, None)
        _persist()
    return stored


def reset_cache() -> None:
    """Drop remembered profiles (used by tests and after a state-path change)."""
    global _LOADED_FROM
    with _LOCK:
        _PROFILES.clear()
        _FAILURES.clear()
        _REJECTED_FORMATS.clear()
        _IN_FLIGHT.clear()
        _LOADED_FROM = None


def model_profile(
    connection: Optional[Mapping[str, str]], *, now: Optional[float] = None
) -> Dict[str, Any]:
    """The measured profile for this endpoint and model, or a stated default.

    This is on every budget decision, so it never touches the network.
    """
    if not connection or not connection.get("base_url"):
        return unmeasured_profile(connection, "No AI model is selected yet.")
    with _LOCK:
        key = profile_key(connection)
        stored = _load().get(key)
        reason = _FAILURES.get(key, "")
    if not stored:
        # If a measurement was attempted and could not run, say which one and
        # why rather than reporting the same "not measured yet" an endpoint
        # nobody has tried gets.
        return unmeasured_profile(connection, reason)
    profile = dict(stored)
    moment = time.time() if now is None else now
    age = moment - float(profile.get("measured_at", 0.0))
    # A stale measurement still describes this model better than a guess does,
    # so it stays in force and is only reported as due for a refresh.
    profile["stale"] = age > PROFILE_MAX_AGE_SECONDS
    return profile


def calibrate(
    connection: Mapping[str, str], *, force: bool = False,
    timeout: float = PROBE_TIMEOUT_SECONDS, now: Optional[float] = None,
) -> Dict[str, Any]:
    """Measure this model once and remember the result.

    A failed measurement is never fatal: it returns the documented conservative
    default with the reason it could not be measured, so selecting a model whose
    server is briefly down still leaves a usable appliance.
    """
    if not force:
        existing = model_profile(connection, now=now)
        if existing.get("measured") and not existing.get("stale"):
            return existing
    try:
        observation = measure_connection(connection, timeout=timeout)
        # Derivation is inside the guarantee too. A `completion_tokens` of
        # 1e308 costs nothing to send and overflows the arithmetic that sizes
        # a budget from it, which happened after the old `try` had ended.
        profile = derive_profile(connection, observation)
    except urllib.error.HTTPError as error:
        return _record_failure(connection, (
            "Calibration could not run: the model server answered HTTP {}. "
            "Vaelor is using a conservative default budget."
        ).format(error.code))
    except (
        # A model server is not a trusted peer. `AttributeError` is what a
        # `message` of the wrong type or a body that is a list produces, and
        # `ArithmeticError` covers an absurd number reached through arithmetic;
        # `LookupError` keeps the KeyError and IndexError this always caught.
        ArithmeticError, AttributeError, LookupError, OSError, TypeError,
        ValueError,
    ) as error:
        return _record_failure(connection, (
            "Calibration could not run ({}). Vaelor is using a conservative "
            "default budget."
        ).format(type(error).__name__))
    return remember(profile)


def _record_failure(connection: Mapping[str, str], detail: str) -> Dict[str, Any]:
    """Return the conservative default, and remember why it is the default."""
    with _LOCK:
        # Bounded like the profiles themselves. Reasons are cheap to rebuild -
        # the next attempt writes a fresh one - so the whole set is dropped
        # rather than carrying an eviction order it does not need.
        if len(_FAILURES) >= MAX_STORED_PROFILES:
            _FAILURES.clear()
        _FAILURES[profile_key(connection)] = detail
    return unmeasured_profile(connection, detail)


def calibrate_in_background(connection: Mapping[str, str], *, force: bool = False):
    """Start a measurement without holding up whoever asked for it.

    Selecting a model must return immediately; the slowest model measured on
    this appliance takes over two minutes to answer one agent task. Until the
    profile lands, every budget uses the conservative default.
    """
    key = profile_key(connection)
    with _LOCK:
        if key in _IN_FLIGHT:
            return None
        _IN_FLIGHT.add(key)

    def run() -> None:
        try:
            calibrate(dict(connection), force=force)
        except Exception:  # pragma: no cover - a probe must never crash a thread
            LOGGER.exception("Model calibration failed for %s", key)
        finally:
            with _LOCK:
                _IN_FLIGHT.discard(key)

    thread = threading.Thread(target=run, name="vaelor-model-calibration", daemon=True)
    thread.start()
    return thread


def calibrate_if_unmeasured(connection: Optional[Mapping[str, str]]) -> bool:
    """Measure an endpoint that has a model selected but no profile yet.

    Calibration used to run only when a model was selected, tested, or
    re-measured on request. An appliance whose model was chosen before that code
    shipped therefore stayed unmeasured for ever, and unmeasured is not a
    neutral state: it is what made every agent run guess its structured-output
    contract, size its budget from a default, and time out on a floor derived
    from nothing. The appliance this was found on had no stored profile at all.

    The measurement is one real generation - 161 to 173 seconds against the
    model on this appliance - so it always runs on its own thread and this
    returns at once. It starts at most one per endpoint and model: a
    measurement in flight, one already stored, or one that has already failed
    all decline to start another, so a server that is down is probed once
    rather than on a loop.

    Returns True when a measurement was started, for callers that want to say so.
    """
    if not connection or not connection.get("base_url"):
        return False
    if connection.get("provider") == "openai":
        # A hosted frontier endpoint bills for the probe. Measuring one stays
        # something an operator asks for, never something a restart buys.
        return False
    key = profile_key(connection)
    with _LOCK:
        if key in _IN_FLIGHT or key in _FAILURES or _load().get(key):
            return False
    return calibrate_in_background(dict(connection)) is not None


def calibration_pending(connection: Optional[Mapping[str, str]]) -> bool:
    """True while a measurement for this connection is still running."""
    if not connection:
        return False
    with _LOCK:
        return profile_key(connection) in _IN_FLIGHT


def reasoning_headroom_tokens(
    connection: Optional[Mapping[str, str]], visible_tokens: int
) -> int:
    """Room for `visible_tokens` of answer plus this model's own thinking.

    A reasoning model emits its hidden pass first, out of the same allowance.
    Asking one for 900 tokens of JSON and giving it 900 tokens total is how a
    perfectly capable model returns nothing at all.
    """
    wanted = max(1, int(visible_tokens))
    profile = model_profile(connection)
    ratio = max(0.0, float(profile.get("reasoning_ratio", 0.0)))
    tokens = int(math.ceil(wanted * (1.0 + ratio * REASONING_MARGIN)))
    if tokens > wanted:
        tokens = int(math.ceil(tokens / 64.0) * 64)
    # The profile's own `max_tokens` is not the cap here: it sizes one
    # agent-shaped answer, and a caller asking for a deliberately longer reply
    # should get it. What does cap this is the hardware the model runs on.
    ceiling = (
        MANAGED_LOCAL_MAX_TOKENS if profile.get("managed_local") else MAX_AGENT_TOKENS
    )
    return max(1, min(tokens, ceiling))


def managed_local_token_ceiling() -> int:
    """The upper bound any managed-local request is held to."""
    return MANAGED_LOCAL_MAX_TOKENS


def profile_timeout_floor(connection: Optional[Mapping[str, str]]) -> int:
    """Seconds this model was measured to need, with headroom, or zero.

    Only ever used to *raise* a timeout. Measurement may tell Vaelor that a
    model needs longer than configured; it must never shorten a budget an
    operator set deliberately.
    """
    profile = model_profile(connection)
    return int(profile.get("timeout_seconds", 0)) if profile.get("measured") else 0


def note_structured_rejection(
    connection: Optional[Mapping[str, str]], response_format: Any
) -> None:
    """Remember that this endpoint refused this structured-output contract.

    Called only once the identical request has succeeded with the field
    removed, so this is the server's own answer and not an inference from a
    status code that could have meant anything.

    Without it, an endpoint that rejects a contract rejects it again on every
    single request, forever. On the appliance's own LM Studio that cost a wasted
    round trip *and* a just-in-time model load before the real request could
    start - measured at ten seconds and a 400 before any work began - and left a
    long generation exposed to being evicted by the next request for a different
    model, which is what finally surfaced as "HTTP 400" two minutes into a run.
    """
    kind = ""
    if isinstance(response_format, Mapping):
        kind = str(response_format.get("type", ""))
    if not kind or not connection or not connection.get("base_url"):
        return
    key = profile_key(connection)
    with _LOCK:
        if len(_REJECTED_FORMATS) >= MAX_STORED_PROFILES:
            _REJECTED_FORMATS.clear()
        _REJECTED_FORMATS.setdefault(key, set()).add(kind)
    LOGGER.info(
        "%s refused response_format %r; Vaelor will not send it again.", key, kind
    )


def structured_output_rejected(
    connection: Optional[Mapping[str, str]], kind: str
) -> bool:
    """True when this endpoint has already refused this contract."""
    if not connection or not connection.get("base_url"):
        return False
    with _LOCK:
        return kind in _REJECTED_FORMATS.get(profile_key(connection), set())


def structured_response_format(
    connection: Optional[Mapping[str, str]],
    schema: Optional[Dict[str, Any]] = None,
) -> Optional[Dict[str, Any]]:
    """The `response_format` this server was measured to accept, if any.

    An unmeasured endpoint used to be sent `{"type":"json_object"}` on the
    theory that the client's drop-on-400 retry would sort out a server that
    refused it. LM Studio - the appliance's own model server - refuses exactly
    that, so every agent run began with a guaranteed rejection. It is not a
    guess worth making: `json_schema` says strictly more, is accepted by every
    server Vaelor has been measured against that supports structured output at
    all, and is only sent when the caller supplied the shape it asked for. A
    caller that declared no shape sends no contract, because the alternative is
    asking a server to answer a question nobody put.
    """
    profile = model_profile(connection)
    mode = (
        str(profile.get("structured_output", "prompt"))
        if profile.get("measured") else "json_schema"
    )
    response_format = response_format_for(mode, schema)
    if response_format is not None and structured_output_rejected(
        connection, str(response_format.get("type", ""))
    ):
        return None
    return response_format


def with_structured_response_format(
    payload: Dict[str, Any], connection: Optional[Mapping[str, str]],
    schema: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    """Attach the structured-output contract this server was measured to accept.

    LM Studio rejects `{"type":"json_object"}` with HTTP 400 and asks for
    `json_schema`. The shared client copes by dropping `response_format`
    entirely, which works but throws away the server's guarantee that the reply
    is a JSON object at all. Where calibration established that `json_schema` is
    available and the caller declared the shape it asked for, use it; where
    nothing structured is accepted, send nothing rather than spending a round
    trip rediscovering that on every request.

    A caller that passes no `schema` never receives one. Vaelor asks for three
    different JSON shapes from three different prompts, and a schema is only
    correct for the prompt it was written for.
    """
    response_format = structured_response_format(connection, schema)
    if response_format is None:
        return payload
    return {**payload, "response_format": response_format}
