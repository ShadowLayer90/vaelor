"""Bounded OpenAI-compatible inference client helpers."""

from __future__ import annotations

import http.client
import io
import json
import logging
import re
import socket
import threading
import time
import urllib.error
import urllib.parse
import urllib.request
import uuid
from typing import Any, Dict, Optional

from .credential_broker import CredentialError, validate_compatible_profile
from .flm_service import FLM_PENALTIES
from .model_credential_roles import is_the_assistants_npu_model
from .hosted_providers import HOSTED_KINDS
from .model_profiles import note_structured_rejection, profile_timeout_floor
from .runtime_paths import env_value
from .provider_runtime import managed_local_connection
from .local_inference_gate import local_inference_slot
from .session_affinity import session_headers

LOGGER = logging.getLogger(__name__)

MAX_PROVIDER_RESPONSE_BYTES = 2 * 1024 * 1024
OPENAI_BASE_URL = "https://api.openai.com/v1"

#: What the shared plain-urllib inference path answers for a hosted AI Chat
#: connection (VD-206 / VD-207); every caller turns a ValueError into its own
#: degraded result.
HOSTED_NOT_ON_THIS_PATH = (
    "A hosted AI Chat connection is not used for appliance work; it is reached "
    "only from AI Chat."
)

#: Where a normalized per-answer timing summary is stashed on a parsed response
#: body, so a caller that discards the raw body can still surface it. See
#: :func:`normalize_performance`.
PERFORMANCE_KEY = "performance"

#: A model reply over :data:`MAX_PROVIDER_RESPONSE_BYTES`, streamed or not.
TOO_MUCH_DATA = "The connected AI endpoint returned too much data."


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
    if connection.get("provider") in HOSTED_KINDS:
        # VD-220: only a custom agent's run on AI Chat's own hosted model, which
        # `chat_completion` sends through `hosted_transport`; refused otherwise.
        from .ai_chat_model_lease import CUSTOM_AGENT_ON_AI_CHAT
        from .hosted_transport import parse_hosted_url

        try:
            parse_hosted_url(str(connection.get("base_url") or ""))
        except CredentialError:
            return False
        return bool(connection.get(CUSTOM_AGENT_ON_AI_CHAT))
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


def appliance_session_headers(connection: Dict[str, str], key: Any) -> Dict[str, str]:
    """``X-Session-Id`` for a model this appliance serves, else nothing.

    The key only means something to Vaelor's own replica balancer (VD-157), so
    a hosted service or another machine's endpoint is told nothing about which
    conversation or task a request belongs to (VD-158). The one rule for every
    caller that names its work: AI Chat and the multi-call agent, research and
    Assistant paths (VD-216 fix 1).
    """
    return session_headers(key) if managed_local_connection(connection) else {}


#: VD-218 / R136, measured on the Z2 2026-10-17: FastFlowLM 1.0.2 keeps one of
#: its ten connection places for every request whose client closes the socket
#: mid-generation; after ten it refuses every connection until it is restarted.
#: A request read to its end costs nothing (36 Stops, R143). So a FastFlowLM
#: generation is never ended by closing its connection: it is named with a
#: ``request_id`` (FLM's handler reads that key from the request body, else
#: invents ``req_<n>``), asked to end with ``POST /api/cancel`` - which 1.0.2
#: accepts and ignores (R143), so nothing waits on it - and read to its end.
#: Other loopback servers (llama-server, the vLLM balancer) end cleanly when
#: their connection closes and have no cancel request; they keep that.
CANCEL_PATH = "/api/cancel"

#: Seconds the cancel request itself may take.
CANCEL_SECONDS = 2.0

#: The floor on how long a cancelled or over-budget generation may take to end
#: before its connection is closed anyway (the last resort, logged).
DRAIN_SECONDS = 60.0

#: The slowest a FastFlowLM generation writes. Measured 13.0-14.5 tokens a
#: second on the Z2 NPU (kb N108, R143); 10 leaves room for contention. A
#: generation cannot outlive ``max_tokens`` at this rate unless it is wedged.
SLOWEST_DECODE_TPS = 10.0

#: Logged when even the drain ran out and the connection was closed anyway.
LAST_RESORT = ("The model server did not end a generation within its drain time; "
               "closing its connection, which may cost FastFlowLM a connection place (R136).")


def drain_seconds(request_body: Dict[str, Any]) -> float:
    """How long a cancelled generation may take to end: the most it can still write.

    ``max_tokens`` at :data:`SLOWEST_DECODE_TPS`, and never less than
    :data:`DRAIN_SECONDS` - e.g. 768 tokens is 77 s, 320 tokens is the 60 s floor.
    """
    tokens = request_body.get("max_tokens") or request_body.get("max_completion_tokens") or 0
    try:
        return max(DRAIN_SECONDS, float(tokens) / SLOWEST_DECODE_TPS)
    except (TypeError, ValueError):
        return DRAIN_SECONDS


def flm_connection(connection: Dict[str, str]) -> bool:
    """Whether this connection reaches FastFlowLM, the Assistant's NPU server.

    The product's own rule for its NPU model (:func:`is_the_assistants_npu_model`):
    a model Vaelor runs here, marked as the NPU's by its credential id or by
    the FLM tag it is pinned to. Never just "a loopback address", which the GPU
    servers share, and never just the tag: an owner-added Ollama endpoint
    serves ``qwen3.5:4b`` too (R149).
    """
    return managed_local_connection(connection) and is_the_assistants_npu_model({
        "id": connection.get("credential_id") or "", "provider": connection.get("provider") or "",
        "owner": connection.get("owner") or "", "selected_model": connection.get("model") or ""})


def hang_up(sock: Optional[socket.socket], why: str = "") -> None:
    """Close ``sock`` now, from any thread, waking a read blocked on it.

    shutdown() wakes a blocked recv on Linux; on Windows only closing the
    handle does. Detaching first marks the socket object closed, so the
    reader's next call fails instead of using a handle that may be reused.
    """
    if sock is None:
        return
    if why:
        LOGGER.warning(why)
    try:
        sock.shutdown(socket.SHUT_RDWR)
    except OSError:
        pass
    try:
        socket.close(sock.detach())
    except OSError:
        pass


def new_request_id() -> str:
    """A name for one generation, which the server's cancel request accepts."""
    return "vaelor-{}".format(uuid.uuid4().hex)


def cancel_request(connection: Dict[str, str], request_id: str) -> bool:
    """Ask the managed local server to end the generation named ``request_id``.

    ``True`` when the server accepted the request without an ``error`` body -
    which is NOT proof the generation stopped. Measured on the Z2 2026-10-17
    (FLM 1.0.2, 33 cancels naming the running request's id): every one was
    accepted, none was logged as cancelled, and every generation ran to its
    token limit. So nothing here relies on the cancel: the caller always
    drains, and the drain is what keeps the server's connection places
    (R136). Never raises.
    """
    parts = urllib.parse.urlsplit(str(connection.get("base_url") or ""))
    if not request_id or parts.scheme not in ("http", "https") or not parts.netloc:
        return False
    url = urllib.parse.urlunsplit((parts.scheme, parts.netloc, CANCEL_PATH, "", ""))
    request = urllib.request.Request(
        url, data=json.dumps({"request_id": request_id}).encode("utf-8"), method="POST",
        headers={"Content-Type": "application/json"})
    try:
        with urllib.request.urlopen(request, timeout=CANCEL_SECONDS) as response:
            body = json.loads(response.read(4096).decode("utf-8", "replace") or "{}")
            return 200 <= response.status < 300 and not (isinstance(body, dict) and body.get("error"))
    except (OSError, ValueError, urllib.error.URLError) as error:
        LOGGER.warning("The model server did not accept a cancel request: %s", error)
        return False


class CancelAtDeadline:
    """Cancel one named generation when its budget runs out.

    The client then waits for the server to end it rather than closing the
    connection on it (R136). ``fired`` says the budget ran out; ``on_fire``
    is how the generation is cancelled.
    """

    def __init__(self, seconds: float, on_fire: Any) -> None:
        self.fired = False
        self._on_fire = on_fire
        self._timer = threading.Timer(max(0.05, float(seconds)), self._fire)
        self._timer.daemon = True
        self._timer.start()

    def _fire(self) -> None:
        self.fired = True
        self._on_fire()

    def cancel(self) -> None:
        self._timer.cancel()


#: What a generation cancelled at its budget raises, once the server ended it.
OVER_BUDGET = "The model did not finish within its time budget, so the appliance cancelled it."


def flm_plain_post(connection: Dict[str, str], url: str, payload: Dict[str, Any],
                   headers: Dict[str, str], timeout: float, max_bytes: int) -> bytes:
    """One non-streamed FastFlowLM generation, never hung up on while it runs (R136).

    FastFlowLM sends nothing until a non-streamed generation has ended, so a
    per-read socket timeout is no bound at all - it either closes on a working
    generation or never fires. The bound is wall-clock instead: at ``timeout``
    the generation is cancelled by name; at ``timeout`` plus
    :func:`drain_seconds` (the most it could still write) the connection is
    closed as the logged last resort. Raises ``urllib.error.HTTPError`` for an
    error status, as the urllib path does, and ``TimeoutError`` over budget.
    """
    request_id = new_request_id()
    parts = urllib.parse.urlsplit(url)
    client = http.client.HTTPConnection(parts.hostname, parts.port or 80, timeout=None)
    budget = CancelAtDeadline(timeout, lambda: cancel_request(connection, request_id))
    held: Dict[str, Any] = {}  # the socket; http.client drops its own reference
    give_up = threading.Timer(float(timeout) + drain_seconds(payload),
                              lambda: hang_up(held.get("sock"), LAST_RESORT))
    give_up.daemon = True
    try:
        client.connect()
        held["sock"] = client.sock
        give_up.start()
        # VD-220 (owner): every FastFlowLM request states its penalties (FLM_PENALTIES).
        sent = dict(payload, request_id=request_id, **FLM_PENALTIES)
        client.request("POST", parts.path, body=json.dumps(sent).encode("utf-8"),
                       headers=headers)
        response = client.getresponse()
        data = response.read(max_bytes + 1)
        status, reason, response_headers = response.status, response.reason, response.headers
    except (OSError, http.client.HTTPException) as error:
        if budget.fired:
            raise TimeoutError(OVER_BUDGET) from error
        if isinstance(error, OSError):
            raise
        raise ConnectionError(str(error)) from error
    finally:
        budget.cancel()
        give_up.cancel()
        client.close()
    if status >= 400:
        raise urllib.error.HTTPError(url, status, reason, response_headers, io.BytesIO(data))
    if budget.fired:
        raise TimeoutError(OVER_BUDGET)
    return data


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
    from .ai_chat_model_lease import (
        CUSTOM_AGENT_ON_AI_CHAT, ai_chat_turn, hosted_agent_completion,
        off_machine_agent_body,
    )

    on_ai_chat = bool(connection.get(CUSTOM_AGENT_ON_AI_CHAT))
    if str(connection.get("provider") or "") in HOSTED_KINDS:
        # VD-206 / VD-207: a hosted AI Chat connection is reached only through
        # `hosted_transport` (checked, pinned, no redirects), and only for a
        # custom agent's run on AI Chat's model (VD-220). This shared path is
        # plain urllib, so anything else is refused before anything is sent.
        if not on_ai_chat:
            raise ValueError(HOSTED_NOT_ON_THIS_PATH)
        with ai_chat_turn(connection):
            return hosted_agent_completion(connection, request_body, timeout)
    if on_ai_chat and connection.get("provider") == "openai":
        request_body = off_machine_agent_body(connection, dict(request_body))
    url = "{}/chat/completions".format(connection["base_url"])
    flm = flm_connection(connection)

    def send(payload):
        if flm:
            # VD-218 / R136: never hang up on a running FastFlowLM generation.
            content = flm_plain_post(connection, url, payload, headers, timeout,
                                     MAX_PROVIDER_RESPONSE_BYTES)
        else:
            request = urllib.request.Request(
                url,
                data=json.dumps(payload).encode("utf-8"),
                headers=headers,
                method="POST",
            )
            with urllib.request.urlopen(request, timeout=timeout) as response:
                content = response.read(MAX_PROVIDER_RESPONSE_BYTES + 1)
        if len(content) > MAX_PROVIDER_RESPONSE_BYTES:
            raise ValueError(TOO_MUCH_DATA)
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
    # VD-220: a custom agent's call takes AI Chat's own gate (the cluster
    # permits in Mode B, the endpoint's one slot in Mode A). The gates refuse
    # rather than queue, so the agent's call waits its turn, bounded, and then
    # fails as busy (`ai_chat_turn`); every other caller still refuses at once.
    slot = ai_chat_turn(connection) if on_ai_chat else local_inference_slot(connection)
    with slot:
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
