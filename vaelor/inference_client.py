"""Bounded OpenAI-compatible inference client helpers."""

from __future__ import annotations

import json
import logging
import re
import time
import urllib.error
import urllib.request
from typing import Any, Dict, Optional

from .credential_broker import CredentialError, validate_compatible_profile
from .model_profiles import note_structured_rejection, profile_timeout_floor
from .runtime_paths import env_value
from .provider_runtime import managed_local_connection
from .local_inference_gate import local_inference_slot

LOGGER = logging.getLogger(__name__)

MAX_PROVIDER_RESPONSE_BYTES = 2 * 1024 * 1024
OPENAI_BASE_URL = "https://api.openai.com/v1"

#: Where a normalized per-answer timing summary is stashed on a parsed response
#: body, so a caller that discards the raw body can still surface it. See
#: :func:`normalize_performance`.
PERFORMANCE_KEY = "performance"


def _positive_float(*candidates: Any) -> Optional[float]:
    """The first candidate that reads as a number greater than zero, or None."""
    for value in candidates:
        if value is None:
            continue
        try:
            number = float(value)
        except (TypeError, ValueError):
            continue
        if number > 0:
            return number
    return None


def _seconds_from_ms(value: Any) -> Optional[float]:
    number = _positive_float(value)
    return number / 1000.0 if number is not None else None


def normalize_performance(
    body: Any, total_seconds: Optional[float] = None,
) -> Dict[str, float]:
    """Fold a model server's own timing report into one compact summary.

    Two serving stacks report timings in their own vocabulary, and this is the
    one place that knows both:

    * FastFlowLM (the NPU Assistant) returns an OpenAI-style ``usage`` object
      carrying ``prefill_duration_ttft`` (seconds to first token),
      ``prefill_speed_tps``, ``decoding_speed_tps`` and ``decoding_duration``
      (seconds).
    * llama.cpp (the GPU AI Chat) returns a ``timings`` object with
      ``prompt_per_second`` / ``predicted_per_second`` and ``prompt_ms`` /
      ``predicted_ms``.

    Both collapse to ``total_seconds`` / ``ttft_seconds`` / ``prefill_tps`` /
    ``decode_tps``. Every field is optional, and a provider that reports no
    timings at all yields ``{}`` - so a hosted endpoint that answers without any
    ``usage`` or ``timings`` block shows no line, even though a wall-clock was
    measured around it: a bare "total" with nothing to attribute it to is not
    worth the row. When the model does report timings, ``total_seconds`` prefers
    the caller's wall-clock - the most faithful time to complete - and falls back
    to the model's own ttft-plus-decode durations only when none was measured.
    """
    usage = body.get("usage") if isinstance(body, dict) else None
    timings = body.get("timings") if isinstance(body, dict) else None
    usage = usage if isinstance(usage, dict) else {}
    timings = timings if isinstance(timings, dict) else {}
    ttft = _positive_float(
        usage.get("prefill_duration_ttft"), _seconds_from_ms(timings.get("prompt_ms")),
    )
    prefill_tps = _positive_float(
        usage.get("prefill_speed_tps"), timings.get("prompt_per_second"),
    )
    decode_tps = _positive_float(
        usage.get("decoding_speed_tps"), timings.get("predicted_per_second"),
    )
    decode_seconds = _positive_float(
        usage.get("decoding_duration"), _seconds_from_ms(timings.get("predicted_ms")),
    )
    if ttft is None and prefill_tps is None and decode_tps is None and decode_seconds is None:
        return {}
    total = _positive_float(total_seconds)
    if total is None and ttft is not None and decode_seconds is not None:
        total = ttft + decode_seconds
    performance: Dict[str, float] = {}
    if total is not None:
        performance["total_seconds"] = round(total, 3)
    if ttft is not None:
        performance["ttft_seconds"] = round(ttft, 3)
    if prefill_tps is not None:
        performance["prefill_tps"] = round(prefill_tps, 1)
    if decode_tps is not None:
        performance["decode_tps"] = round(decode_tps, 1)
    return performance


def allowed_inference_endpoint(connection: Dict[str, str]) -> bool:
    if connection.get("provider") == "openai":
        return connection.get("base_url") == OPENAI_BASE_URL
    try:
        validate_compatible_profile(json.dumps({
            "base_url": connection.get("base_url", ""),
            "model": "",
            "api_key": "",
        }))
        return True
    except CredentialError:
        return False


def _drop_trailing_comma(out: list) -> None:
    """Remove a comma (and any whitespace after it) sitting at the end of ``out``.

    Called just before a closing ``}``/``]`` is emitted and once more at the end
    of a truncation, so ``{"a":1,}`` and ``[1,2,]`` lose the stray comma a strict
    parser rejects. Only a genuine trailing comma is touched; a comma between two
    values is never at the tail of ``out`` when a closer arrives, so it is left
    alone.
    """
    i = len(out) - 1
    while i >= 0 and out[i] in " \t\r\n":
        i -= 1
    if i >= 0 and out[i] == ",":
        del out[i:]


def _balance_json_fragment(fragment: str) -> str | None:
    """Best-effort structural repair of near-JSON, or ``None`` if unsafe.

    A single left-to-right pass tracks whether we are inside a string literal
    (respecting backslash escapes) so that braces and quotes *inside* strings are
    never miscounted, and keeps a stack of the open ``{``/``[`` delimiters. It
    repairs exactly two malformations common in truncated or sloppy local-model
    output, and nothing else:

      * trailing commas before a ``}``/``]`` (stripped via ``_drop_trailing_comma``)
      * a value cut off before it closed - the missing ``}``/``]`` are appended in
        the correct order, and a string truncated mid-token is closed first.

    It deliberately does NOT invent missing commas *between* values: that would be
    guessing at content, so such input falls through and is reported as malformed.
    Any structural contradiction (a closer with nothing open, or a ``}`` closing a
    ``[``) returns ``None`` rather than a reshaped guess. Scanning stops once the
    first top-level value closes, so trailing prose after a complete object is
    ignored the same way ``raw_decode`` would.
    """
    stack: list = []
    out: list = []
    in_string = False
    escaped = False
    for ch in fragment:
        if in_string:
            out.append(ch)
            if escaped:
                escaped = False
            elif ch == "\\":
                escaped = True
            elif ch == '"':
                in_string = False
            continue
        if ch == '"':
            in_string = True
            out.append(ch)
            continue
        if ch in "{[":
            stack.append(ch)
            out.append(ch)
            continue
        if ch in "}]":
            _drop_trailing_comma(out)
            if not stack:
                return None  # a closer with nothing open: too broken to trust
            opener = stack.pop()
            if (opener == "{") != (ch == "}"):
                return None  # mismatched pair, e.g. "[...}": do not guess
            out.append(ch)
            if not stack:
                break  # first top-level value complete; ignore any trailing prose
            continue
        out.append(ch)
    if stack:  # truncated: close the string (if any) then the open delimiters
        if in_string:
            if escaped:
                out.pop()  # a dangling backslash would escape our closing quote
            out.append('"')
        _drop_trailing_comma(out)
        while stack:
            out.append("}" if stack.pop() == "{" else "]")
    return "".join(out)


def _recover_json_object(fragment: str) -> Any:
    """Try to parse ``fragment`` after a conservative structural repair.

    Returns the decoded value, or ``None`` when repair is impossible or the
    repaired text still will not parse. ``raw_decode`` (not ``json.loads``) is
    used so trailing prose after the object is tolerated, matching the earlier
    unrepaired attempt.
    """
    repaired = _balance_json_fragment(fragment)
    if repaired is None:
        return None
    try:
        result, _ = json.JSONDecoder().raw_decode(repaired)
    except json.JSONDecodeError:
        return None
    return result


def parse_model_object(content: Any) -> Dict[str, Any]:
    """Accept a JSON object even when a small model wraps it in a code fence."""
    if isinstance(content, dict):
        return content
    text = str(content or "").strip()
    if text.startswith("```"):
        text = re.sub(r"^```(?:json)?\s*", "", text, count=1, flags=re.IGNORECASE)
        text = re.sub(r"\s*```$", "", text, count=1)
    try:
        result = json.loads(text)
    except json.JSONDecodeError:
        start = text.find("{")
        if start < 0:
            raise ValueError("The connected AI endpoint did not return JSON.")
        try:
            result, _ = json.JSONDecoder().raw_decode(text[start:])
        except json.JSONDecodeError as error:
            # Last resort, only after both strict attempts failed: repair the
            # common small-model malformations (trailing commas, a truncated
            # response). This never changes the outcome for input that already
            # parsed, and gives up cleanly rather than returning a wrong object.
            result = _recover_json_object(text[start:])
            if result is None:
                raise ValueError(
                    "The connected AI endpoint returned malformed JSON."
                ) from error
    if not isinstance(result, dict):
        raise ValueError("The connected AI endpoint did not return a JSON object.")
    return result


# A connected endpoint may be a large model on another machine. This is the
# floor for those; raise it on slow hardware, lower it if you would rather fail
# fast than wait.
DEFAULT_REMOTE_INFERENCE_SECONDS = 240
MAX_INFERENCE_SECONDS = 900


def remote_inference_budget() -> int:
    """Seconds a connected (non-managed-local) endpoint gets to answer.

    This is read per request, so an unparseable setting must degrade to the
    default rather than raise: `inf` and `nan` both survive `float()` and then
    fail `int()` with OverflowError or ValueError, which would turn one typo in
    the environment into a 500 on every message. Silence is its own defect
    though - `60s` and `0x10` used to become 240 with nothing said anywhere -
    so anything unusable is logged with the value that caused it.
    """
    raw = env_value(
        "VAELOR_INFERENCE_TIMEOUT_SECONDS",
        "PM_INFERENCE_TIMEOUT_SECONDS",
        str(DEFAULT_REMOTE_INFERENCE_SECONDS),
    )
    try:
        parsed = float(raw)
        if parsed != parsed or parsed in (float("inf"), float("-inf")):
            raise ValueError("timeout must be a finite number of seconds")
        seconds = int(parsed)
    except (OverflowError, TypeError, ValueError):
        LOGGER.warning(
            "VAELOR_INFERENCE_TIMEOUT_SECONDS=%r is not a number of seconds; "
            "using %s instead", raw, DEFAULT_REMOTE_INFERENCE_SECONDS,
        )
        seconds = DEFAULT_REMOTE_INFERENCE_SECONDS
    return max(30, min(seconds, MAX_INFERENCE_SECONDS))


def inference_timeout(connection: Dict[str, str], configured: int) -> int:
    """How long one completion may take.

    A capable model is not a fast one. A 27B answering with structured JSON
    over a LAN routinely needs two to three minutes, and the old ceiling of 60
    seconds cut every one of those requests off mid-answer - so a model that
    was working perfectly well reported as a timeout on every single call. The
    budget now follows what the endpoint actually is, and stays overridable for
    hardware slower or faster than this.

    "Small model on this box" is not the same as "fast", which is why the
    measured floor matters: the 1.7B on this appliance's own Pi needs about 104
    seconds for one agent task, more than twice the 45-second floor that was
    meant to make it fail fast. Measurement can only *raise* the budget here, so
    a `VAELOR_INFERENCE_TIMEOUT_SECONDS` an operator set deliberately is never
    silently shortened by a probe.
    """
    budget = max(int(configured), 1)
    if managed_local_connection(connection):
        budget = max(budget, 45)
    else:
        budget = max(budget, remote_inference_budget())
    return min(max(budget, profile_timeout_floor(connection)), MAX_INFERENCE_SECONDS)


def chat_completion(
    connection: Dict[str, str],
    request_body: Dict[str, Any],
    headers: Dict[str, str],
    timeout: int,
) -> Dict[str, Any]:
    """Call chat completions, tolerating servers without JSON-mode support.

    The retry is a fallback, not a routine: a rejection proved by the retry is
    recorded against the endpoint so the same contract is never sent to it
    again. A validation rejection that repeats on every request is not free -
    on LM Studio it costs a just-in-time model load before the 400, and it
    leaves the real request to run in whatever time is left.
    """
    url = "{}/chat/completions".format(connection["base_url"])

    def send(payload):
        request = urllib.request.Request(
            url,
            data=json.dumps(payload).encode("utf-8"),
            headers=headers,
            method="POST",
        )
        with urllib.request.urlopen(request, timeout=timeout) as response:
            content = response.read(MAX_PROVIDER_RESPONSE_BYTES + 1)
        if len(content) > MAX_PROVIDER_RESPONSE_BYTES:
            raise ValueError("The connected AI endpoint returned too much data.")
        return json.loads(content.decode("utf-8"))

    # #223 / VD-085: hold one local-inference slot across the whole call -
    # including the response_format fallback retry, which is the same logical
    # generation - so a single managed local model is never asked to run two at
    # once. This is the shared path for agent tasks, research, custom agents and
    # the deployment agent; AI Chat has its own gated call. A busy model raises
    # LocalModelBusy, which the caller surfaces rather than piling onto a worker
    # until the pool saturates. Remote endpoints are not gated.
    # Wall-clock across the whole generation (including the response_format
    # fallback retry, which is the same logical call), so a caller that keeps
    # only the answer text can still report the most faithful time-to-complete.
    start = time.monotonic()
    with local_inference_slot(connection):
        try:
            body = send(request_body)
        except urllib.error.HTTPError as error:
            if (
                error.code not in {400, 422}
                or "response_format" not in request_body
                or connection.get("provider") == "openai"
            ):
                raise
            compatible_body = dict(request_body)
            compatible_body.pop("response_format", None)
            body = send(compatible_body)
            # Only now is the field the proven cause: the same request without
            # it succeeded. A 400 on its own means nothing in particular - a
            # generation evicted mid-flight reports one too - so nothing is
            # recorded until the server has answered the question both ways.
            note_structured_rejection(connection, request_body.get("response_format"))
    if isinstance(body, dict):
        body[PERFORMANCE_KEY] = normalize_performance(body, time.monotonic() - start)
    return body
