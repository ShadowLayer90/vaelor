"""Bounded OpenAI-compatible inference client helpers."""

from __future__ import annotations

import json
import logging
import re
import urllib.error
import urllib.request
from typing import Any, Dict

from .credential_broker import CredentialError, validate_compatible_profile
from .model_profiles import note_structured_rejection, profile_timeout_floor
from .runtime_paths import env_value
from .provider_runtime import managed_local_connection
from .local_inference_gate import local_inference_slot

LOGGER = logging.getLogger(__name__)

MAX_PROVIDER_RESPONSE_BYTES = 2 * 1024 * 1024
OPENAI_BASE_URL = "https://api.openai.com/v1"


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
    with local_inference_slot(connection):
        try:
            return send(request_body)
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
            return body
