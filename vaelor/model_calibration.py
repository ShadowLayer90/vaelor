"""Size a model's response budget by measuring it, not by guessing its name.

Every token budget in Vaelor used to be a constant chosen for whichever model
the author had in front of them. That is wrong in both directions, and the
appliance's own hardware proves it:

- `qwen/qwen3.6-27b` on LM Studio, asked one agent-shaped question at 1200
  tokens, spent all 1200 on 5175-5238 characters of hidden reasoning and
  returned *zero* characters of visible answer - reproduced twice. Given room
  it answers normally in 1376-1624 completion tokens. The product reported the
  starved case as "the model did not return a usable structured response",
  blaming the model for a limit we set.
- `openai/gpt-oss-20b` on the same server answers the same question in 240-367
  tokens with a quarter as much reasoning as answer, in under three seconds.
- The managed-local `Qwen3-1.7B-Q4_K_M` on the Pi answers in 250 tokens with
  *no* reasoning at all, but takes 104 seconds - making the smallest model the
  slowest one on this appliance, the opposite of what a size heuristic predicts.

No name match and no parameter-count heuristic orders those three correctly.
So this module asks the model instead: one representative structured task at a
generous ceiling, recorded as a profile keyed by endpoint and model id. The
profile is measured when a model is selected or tested, never inside a user's
request, and every lookup here is a dictionary read.

When calibration cannot run, the profile is returned unmeasured with a
conservative default and says so, so a missing measurement degrades the sizing
rather than the feature.
"""

from __future__ import annotations

import json
import math
import time
import urllib.error
import urllib.request
from typing import Any, Dict, Mapping, Optional

from .provider_runtime import managed_local_connection
from .token_estimate import estimate_tokens

MAX_PROBE_RESPONSE_BYTES = 1024 * 1024

# The probe must never run at a shipped default, or it measures a truncated
# failure and calibrates to it. This ceiling is deliberately larger than any
# budget Vaelor issues.
PROBE_CEILING_TOKENS = 6000

# Seconds one calibration pass may take. The slowest model measured on this
# appliance needed 158s for this task, so the budget allows for hardware
# meaningfully slower than that before giving up and recording "unmeasured".
PROBE_TIMEOUT_SECONDS = 420

# A capability check is a one-token request (max_tokens=1) whose only purpose is
# the HTTP status. It still pays for the model to be loaded first: on a CPU-only
# Pi with an idle-unloaded model, the first request waits ~67s for the cold load
# before it can answer even a one-token probe. At 30s this was shorter than that
# load and fired on every calibration, raising TimeoutError before the
# generation probe below ever ran and pinning the model to the conservative
# default budget (#205 deep pass, Vaelor 2.1.0a54, Qwen3-4B on CPU). It must
# clear a realistic Pi cold response; generation stays bounded to one token, so
# a generous budget on a background thread cannot make the probe linger.
CAPABILITY_TIMEOUT_SECONDS = 180

# Floor and ceiling for a derived agent budget. The floor keeps a suspiciously
# small measurement from producing a budget that cannot hold a structured
# answer; the ceiling keeps a runaway measurement from handing any endpoint an
# unbounded generation.
MIN_AGENT_TOKENS = 512
MAX_AGENT_TOKENS = 8000

# Managed-local means "on this Pi". The measured throughput of the 1.7B model
# there is roughly 2.4 tokens per second, so 1536 tokens is about ten minutes of
# worst-case generation - inside `MAX_INFERENCE_SECONDS` (900) only because a
# model that reaches its ceiling is already failing. Raising this trades the
# appliance's responsiveness for answers no local model has been observed to
# need, so raise it only with a measurement that says so.
MANAGED_LOCAL_MAX_TOKENS = 1536

# Headroom over what was observed. A single sample tells us what one task
# needed, not what the next one will, so the budget is a multiple of it. Heavy
# reasoners get more, because their hidden pass grows with task difficulty
# faster than their visible answer does.
HEADROOM_BASE = 2.5
HEADROOM_PER_REASONING_RATIO = 0.25
MAX_REASONING_RATIO = 4.0

# Wall clock is measured once, so allow the real thing to take longer.
LATENCY_HEADROOM = 2.0

# A timeout derived from the probe's own wall clock says how long *that* request
# took, not how long the budget the probe justifies will take. Measured on this
# appliance: `qwen/qwen3.6-27b` answered the probe in 119s and was granted a
# 239-second budget from it, then needed longer than that for one real agent
# task and was cut off - a model failing on a limit derived from its own good
# behaviour. So the budget is also priced at the observed throughput: if Vaelor
# is going to let a model write `max_tokens`, it has to be willing to wait for
# them. This margin covers evaluating a real prompt, which the probe's short one
# does not measure.
GENERATION_TIMEOUT_MARGIN = 1.25

# One measurement of the reasoning pass, so let it run a quarter longer than it
# did when sizing a request around it.
REASONING_MARGIN = 1.25

# What an unmeasured endpoint is assumed to be. A connected endpoint is assumed
# to reason as heavily as anything measured here, because under-budgeting a
# reasoning model returns an empty answer while over-budgeting it only reserves
# space the model never uses. Managed-local is assumed not to reason, because
# Vaelor sends `/no_think` and `enable_thinking: false` to it and the local
# model measured on this appliance emitted exactly zero reasoning characters.
UNMEASURED_REMOTE_RATIO = MAX_REASONING_RATIO
UNMEASURED_LOCAL_RATIO = 0.0
UNMEASURED_REMOTE_TOKENS = 4000
UNMEASURED_LOCAL_TOKENS = 1024

# A profile older than this is still used - a stale measurement beats a guess -
# but is reported as stale so an operator or a caller can refresh it.
PROFILE_MAX_AGE_SECONDS = 30 * 24 * 3600

# Reasoning at least as long as the visible answer.
HEAVY_REASONING_RATIO = 1.0

# Servers disagree about how a structured response is requested. LM Studio
# rejects `json_object` outright with HTTP 400 and wants `json_schema`; llama.cpp
# and OpenAI accept `json_object`. Which one this endpoint takes is measured,
# not assumed.
#
# A schema is per call site, never global. Vaelor asks three different questions
# in three different shapes - a review, a plan, an assistant answer - and forcing
# one schema on all of them would quietly answer the wrong question. Only a
# caller that knows what it asked for may supply a schema, and callers that have
# not declared one send no `response_format` at all rather than a wrong one.
_STRINGS = {"type": "array", "items": {"type": "string"}}

AGENT_RESULT_SCHEMA: Dict[str, Any] = {
    "name": "vaelor_agent_result",
    # Not strict: `agent_result_shape` also accepts `answer`, `warnings`,
    # `sources`, and `evidence`, and a user-defined agent whose whole purpose is
    # one direct reply must still be able to send `answer`.
    "strict": False,
    "schema": {
        "type": "object",
        "properties": {
            "summary": {"type": "string"},
            "findings": _STRINGS,
            "recommendations": _STRINGS,
            "next_actions": _STRINGS,
            "answer": {"type": "string"},
            "warnings": _STRINGS,
            # #247w: specific questions the agent puts back to the user when it
            # cannot verify an answer or the task is ambiguous, rather than
            # guessing. The runner also derives these deterministically.
            "clarifications": _STRINGS,
        },
        "required": ["summary", "findings", "recommendations", "next_actions"],
        "additionalProperties": True,
    },
}

# The deployment planner. A different question from the agent result above, and
# forcing the agent shape on it would have the planner answer something nobody
# asked, which is why a schema belongs to a call site and never to the module.
PLANNER_PLAN_SCHEMA: Dict[str, Any] = {
    "name": "vaelor_deployment_plan",
    "strict": False,
    "schema": {
        "type": "object",
        "properties": {
            "summary": {"type": "string"},
            "rationale": {"type": "string"},
            "warnings": _STRINGS,
            "checklist": _STRINGS,
            "proposed_job": {"type": ["object", "null"]},
        },
        "required": ["summary", "rationale", "warnings", "checklist"],
        "additionalProperties": True,
    },
}

# The conversational assistant answer.
ASSISTANT_ANSWER_SCHEMA: Dict[str, Any] = {
    "name": "vaelor_assistant_answer",
    "strict": False,
    "schema": {
        "type": "object",
        "properties": {
            "answer": {"type": "string"},
            "evidence": {"type": "array", "items": {"type": "object"}},
            "suggested_actions": _STRINGS,
            "proposed_job": {"type": ["object", "null"]},
        },
        "required": ["answer"],
        "additionalProperties": True,
    },
}

CONNECTOR_PLAN_SCHEMA: Dict[str, Any] = {
    "name": "vaelor_connector_plan",
    "strict": False,
    "schema": {
        "type": "object",
        "properties": {
            "calls": {
                "type": "array",
                "items": {
                    "type": "object",
                    "properties": {
                        "connector_id": {"type": "string"},
                        "operation_id": {"type": "string"},
                        "arguments": {"type": "object"},
                    },
                    "required": ["connector_id", "operation_id", "arguments"],
                },
            },
        },
        "required": ["calls"],
        "additionalProperties": False,
    },
}

APP_CALL_PLAN_SCHEMA: Dict[str, Any] = {
    "name": "vaelor_app_call_plan",
    "strict": False,
    "schema": {
        "type": "object",
        "properties": {
            "calls": {
                "type": "array",
                "items": {
                    "type": "object",
                    "properties": {
                        "grant_id": {"type": "string"},
                        "operation_id": {"type": "string"},
                        "input": {"type": "object"},
                    },
                    "required": ["grant_id", "operation_id", "input"],
                },
            },
        },
        "required": ["calls"],
        "additionalProperties": False,
    },
}

# One small, representative agent task: a structured answer about a few granted
# facts. It is the same shape as real work, so what it costs predicts real work.
PROBE_MESSAGES = [
    {
        "role": "system",
        "content": (
            "You are a Vaelor appliance agent. Use only the granted facts. "
            "Reply with ONLY a JSON object with the keys summary (string), "
            "findings (array of strings), recommendations (array of strings), "
            "and next_actions (array of strings)."
        ),
    },
    {
        "role": "user",
        "content": json.dumps({
            "task": "Assess appliance cooling from the granted facts only.",
            "granted_context": {
                "cpu_temperature_c": 44,
                "cpu_fan": {"rpm": 0, "mode": "automatic"},
                "case_fans": {"count": 2, "shared_control": True, "running": True},
                "services": "all active",
            },
        }, separators=(",", ":")),
    },
]


def profile_key(connection: Mapping[str, str]) -> str:
    """Identify a profile by what was measured: this endpoint, this model.

    Accepts either a connection (`base_url`) or a stored profile (`endpoint`),
    because a profile has to be filed under the same key it is later found by.
    """
    source = connection or {}
    endpoint = source.get("base_url") or source.get("endpoint") or ""
    return "{}|{}".format(str(endpoint).rstrip("/"), str(source.get("model", "")))


def _round_up(value: float, step: int = 64) -> int:
    return int(math.ceil(max(value, 0) / step) * step)


def _token_field(connection: Mapping[str, str]) -> str:
    return (
        "max_completion_tokens"
        if (connection or {}).get("provider") == "openai" else "max_tokens"
    )


def reasoning_ratio_of(content_chars: int, reasoning_chars: int) -> float:
    """Hidden reasoning per character of visible answer.

    An answer of zero characters is the failure this whole module exists to
    prevent, and it carries no usable ratio, so it is treated as the most
    reasoning-heavy case rather than as a division by zero.
    """
    if content_chars > 0:
        return max(0.0, reasoning_chars / content_chars)
    return MAX_REASONING_RATIO if reasoning_chars > 0 else 0.0


def observation_from_response(
    body: Mapping[str, Any],
    *,
    latency_seconds: float,
    ceiling_tokens: int = PROBE_CEILING_TOKENS,
    structured_output: str = "prompt",
) -> Dict[str, Any]:
    """Reduce one completion response to the numbers a profile is derived from."""
    choices = body.get("choices") or [{}]
    choice = choices[0] if isinstance(choices[0], Mapping) else {}
    message = choice.get("message") or {}
    content = str(message.get("content") or "")
    reasoning = str(
        message.get("reasoning_content") or message.get("reasoning") or ""
    )
    usage = body.get("usage") or {}
    try:
        completion_tokens = int(usage.get("completion_tokens") or 0)
    except (AttributeError, TypeError, ValueError):
        # A `usage` that is not an object says exactly as much about the token
        # count as one that is absent, and absence already has a fallback.
        completion_tokens = 0
    measured_tokens = completion_tokens > 0
    if not measured_tokens:
        # Some servers omit `usage`. This carried its own third copy of the
        # characters-per-token factor, at 3, justified as erring high "so an
        # estimate biases towards a larger budget" - a 38% over-count sold as
        # headroom, on the number every derived budget is scaled from.
        completion_tokens = max(1, estimate_tokens(content + reasoning))
    valid_json = False
    if content.strip():
        try:
            valid_json = isinstance(json.loads(content), dict)
        except (TypeError, ValueError):
            valid_json = False
    return {
        "completion_tokens": completion_tokens,
        # Whether that figure is the server's own count or this module's
        # estimate. Both used to arrive under one key, so a profile derived
        # from a guess was indistinguishable from one derived from a
        # measurement - in a module whose entire premise is that measuring
        # beats guessing.
        "completion_tokens_measured": measured_tokens,
        "content_chars": len(content),
        "reasoning_chars": len(reasoning),
        "finish_reason": str(choice.get("finish_reason") or ""),
        "returned_valid_json": valid_json,
        "structured_output": structured_output,
        "latency_seconds": round(max(0.0, float(latency_seconds)), 2),
        "ceiling_tokens": int(ceiling_tokens),
    }


def derive_profile(
    connection: Mapping[str, str], observation: Mapping[str, Any]
) -> Dict[str, Any]:
    """Turn one measurement into the budget, timeout, and shape of a model."""
    local = managed_local_connection(dict(connection or {}))
    ratio = reasoning_ratio_of(
        int(observation.get("content_chars", 0)),
        int(observation.get("reasoning_chars", 0)),
    )
    truncated = str(observation.get("finish_reason", "")) == "length"
    multiplier = HEADROOM_BASE + min(
        ratio, MAX_REASONING_RATIO
    ) * HEADROOM_PER_REASONING_RATIO
    if truncated:
        # The model was still writing when the ceiling stopped it, so the
        # tokens it managed are not what it needed - the ceiling is, and even
        # that is only a lower bound. Size from the limit, not from the loss.
        need = int(observation.get("ceiling_tokens", PROBE_CEILING_TOKENS))
    else:
        need = max(1, int(observation.get("completion_tokens", 1)))
    tokens = min(max(_round_up(need * multiplier), MIN_AGENT_TOKENS), MAX_AGENT_TOKENS)
    if local:
        tokens = min(tokens, MANAGED_LOCAL_MAX_TOKENS)
    latency = float(observation.get("latency_seconds", 0.0))
    observed_tokens = max(1, int(observation.get("completion_tokens", 1)))
    # Tokens per second, from the one generation that was actually watched.
    throughput = observed_tokens / latency if latency > 0 else 0.0
    generation_seconds = tokens / throughput if throughput > 0 else 0.0
    timeout_seconds = int(math.ceil(max(
        latency * LATENCY_HEADROOM, generation_seconds * GENERATION_TIMEOUT_MARGIN,
    )))
    return {
        "endpoint": str((connection or {}).get("base_url", "")).rstrip("/"),
        "model": str((connection or {}).get("model", "")),
        "measured": True,
        "measured_at": time.time(),
        "stale": False,
        "managed_local": local,
        "max_tokens": tokens,
        "timeout_seconds": timeout_seconds,
        "observed_completion_tokens": int(observation.get("completion_tokens", 0)),
        # Whether the server counted those tokens or this module estimated
        # them from the character count. A profile that says "measured" while
        # its central number is a guess is the failure this module was written
        # to end. Absent on an observation supplied directly - by a caller or
        # by a stored profile - which is a given count, not an estimate.
        "observed_tokens_measured": bool(
            observation.get("completion_tokens_measured", True)
        ),
        "observed_latency_seconds": round(latency, 2),
        "reasoning_ratio": round(ratio, 3),
        "reasons_heavily": ratio >= HEAVY_REASONING_RATIO,
        "structured_output": str(observation.get("structured_output", "prompt")),
        "returned_valid_json": bool(observation.get("returned_valid_json")),
        "complete": not truncated,
        "detail": (
            "Measured: this model {} {} tokens and {:g}s for one agent task, "
            "and wrote {:g} characters of reasoning per character of answer."
        ).format(
            "needed" if observation.get("completion_tokens_measured", True)
            else "returned no token count, so this estimates",
            int(observation.get("completion_tokens", 0)), round(latency, 1),
            round(ratio, 2),
        ) if not truncated else (
            "Measured, but the model was still writing at {} tokens, so this "
            "budget is a lower bound rather than an observed need."
        ).format(int(observation.get("ceiling_tokens", PROBE_CEILING_TOKENS))),
    }


def unmeasured_profile(
    connection: Optional[Mapping[str, str]], detail: str = ""
) -> Dict[str, Any]:
    """The documented conservative default, labelled honestly as a guess."""
    local = managed_local_connection(dict(connection or {}))
    return {
        "endpoint": str((connection or {}).get("base_url", "")).rstrip("/"),
        "model": str((connection or {}).get("model", "")),
        "measured": False,
        "measured_at": 0.0,
        "stale": False,
        "managed_local": local,
        "max_tokens": UNMEASURED_LOCAL_TOKENS if local else UNMEASURED_REMOTE_TOKENS,
        "timeout_seconds": 0,
        "observed_completion_tokens": 0,
        "observed_tokens_measured": False,
        "observed_latency_seconds": 0.0,
        "reasoning_ratio": (
            UNMEASURED_LOCAL_RATIO if local else UNMEASURED_REMOTE_RATIO
        ),
        "reasons_heavily": not local,
        "structured_output": "unknown",
        "returned_valid_json": False,
        "complete": True,
        "detail": detail or (
            "This model has not been measured yet, so Vaelor is using a "
            "conservative default budget."
        ),
    }


def _post(
    connection: Mapping[str, str], payload: Dict[str, Any], timeout: float
) -> Dict[str, Any]:
    headers = {"Content-Type": "application/json", "Accept": "application/json"}
    if connection.get("api_key"):
        headers["Authorization"] = "Bearer {}".format(connection["api_key"])
    request = urllib.request.Request(
        "{}/chat/completions".format(str(connection["base_url"]).rstrip("/")),
        data=json.dumps(payload).encode("utf-8"),
        headers=headers,
        method="POST",
    )
    with urllib.request.urlopen(request, timeout=timeout) as response:
        raw = response.read(MAX_PROBE_RESPONSE_BYTES)
    return json.loads(raw.decode("utf-8", "replace"))


def _first_model(connection: Mapping[str, str], timeout: float) -> str:
    headers = {"Accept": "application/json"}
    if connection.get("api_key"):
        headers["Authorization"] = "Bearer {}".format(connection["api_key"])
    request = urllib.request.Request(
        "{}/models".format(str(connection["base_url"]).rstrip("/")), headers=headers
    )
    with urllib.request.urlopen(request, timeout=timeout) as response:
        body = json.loads(response.read(MAX_PROBE_RESPONSE_BYTES).decode("utf-8"))
    return str((body.get("data") or [{}])[0].get("id", ""))


def response_format_for(
    mode: str, schema: Optional[Dict[str, Any]] = None
) -> Optional[Dict[str, Any]]:
    """The `response_format` payload for a measured structured-output mode.

    A `json_schema` server is only asked for a schema when the caller declared
    one. Substituting a house schema for a caller that did not would change the
    question being asked, which is worse than sending no contract at all.
    """
    if mode == "json_schema":
        return {"type": "json_schema", "json_schema": schema} if schema else None
    if mode == "json_object":
        return {"type": "json_object"}
    return None


def measure_structured_output(
    connection: Mapping[str, str], model: str,
    timeout: float = CAPABILITY_TIMEOUT_SECONDS,
) -> str:
    """Which structured-output contract this server accepts.

    Asked with a one-token request, because a server validates `response_format`
    before it generates anything: the status code is the whole answer, and a
    reasoning model does not spend two minutes supplying it. Dropping the field
    on a 400 - which is what the shared client does - works, but it throws away
    the server's own guarantee of a JSON object when `json_schema` was available
    all along.
    """
    for mode in ("json_schema", "json_object"):
        payload = {
            "model": model,
            "messages": [{"role": "user", "content": "ok"}],
            _token_field(connection): 1,
            "response_format": response_format_for(mode, AGENT_RESULT_SCHEMA),
        }
        try:
            _post(connection, payload, timeout)
            return mode
        except urllib.error.HTTPError as error:
            if error.code not in {400, 404, 422, 500, 501}:
                raise
        except ValueError:
            continue
    return "prompt"


def measure_connection(
    connection: Mapping[str, str], *, timeout: float = PROBE_TIMEOUT_SECONDS
) -> Dict[str, Any]:
    """Run one representative agent task and report what it cost."""
    model = str((connection or {}).get("model") or "")
    if not model:
        model = _first_model(connection, CAPABILITY_TIMEOUT_SECONDS)
    structured = measure_structured_output(connection, model)
    payload: Dict[str, Any] = {
        "model": model,
        "messages": list(PROBE_MESSAGES),
        _token_field(connection): PROBE_CEILING_TOKENS,
    }
    # The probe asks the agent-result question, so it is entitled to the
    # agent-result schema.
    response_format = response_format_for(structured, AGENT_RESULT_SCHEMA)
    if response_format is not None:
        payload["response_format"] = response_format
    started = time.monotonic()
    body = _post(connection, payload, timeout)
    return observation_from_response(
        body,
        latency_seconds=time.monotonic() - started,
        ceiling_tokens=PROBE_CEILING_TOKENS,
        structured_output=structured,
    )
