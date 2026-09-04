"""Honest, model-aware explanations for planning fallbacks."""

from __future__ import annotations

import urllib.error
from typing import Dict

from .custom_agent_model import ReasoningBudgetError, UngroundedEntityError
from .model_reachability import describe_endpoint_failure
from .provider_runtime import model_capability, request_complexity

# Kept verbatim: this is the honest description of a model that answered but
# not in the required shape, and it must not leak parser internals.
MALFORMED_RESULT_MESSAGE = (
    "The selected model did not return a usable structured response. "
    "Retry once, shorten the task, or choose a stronger model."
)

# The first thing the reader needs, on every surface: the AI, not their agent,
# is what failed. Someone living on the Agents tab saw "HTTP 400" and an
# endpoint URL and reasonably concluded they had built the agent wrong.
MODEL_NOT_ANSWERING = (
    "The AI model is not answering, so this agent run produced no result."
)


def _unreachable(
    error: Exception, endpoint: str, advice: str, operator_note: str
) -> str:
    """One sentence on what happened, one on what to do, then the endpoint.

    Leading with "Check the model server log at http://192.168.0.72:6354/v1" was
    the entire remediation offered to a person who wanted baseball scores. The
    address still matters - an operator cannot debug without it - so it keeps
    its place at the end, where it does not stand in for an explanation.
    """
    return "{lead} {reason} Nothing on this appliance changed. {advice} {note}".format(
        lead=MODEL_NOT_ANSWERING,
        reason=describe_endpoint_failure(error),
        advice=advice,
        note=operator_note.format(endpoint=endpoint),
    )


def custom_agent_failure(error: Exception, connection: Dict[str, str]) -> str:
    """Explain why one custom-agent run failed, without guessing at the cause.

    Every cause used to collapse into "choose a stronger model", which is wrong
    advice for an endpoint that is simply not running and leaves the user with
    no way to tell the two apart.
    """
    endpoint = str((connection or {}).get("base_url") or "the selected AI endpoint")[:200]
    model = str((connection or {}).get("model") or "The selected model")[:100]
    if isinstance(error, UngroundedEntityError):
        return (
            "The agent's answer named {}, which Vaelor could not find in any "
            "source this agent is allowed to read. Grant public research access, "
            "or name the source it should use, then run it again.".format(
                ", ".join(error.entities[:3])
            )
        )
    if isinstance(error, ReasoningBudgetError):
        return (
            "{} spent its whole response allowance reasoning before it wrote an "
            "answer, so nothing came back. This is a budget limit, not a fault "
            "in the model - retry, shorten the task, or pick a model that "
            "reasons less before answering.".format(model)
        )
    if isinstance(error, urllib.error.HTTPError):
        return _unreachable(
            error, endpoint,
            "Run the agent again; if it keeps failing, pick a different model "
            "on the assistant's model screen. Your agent's instructions are "
            "not the problem.",
            "For operators: {endpoint} refused the request, and its server log "
            "records why.",
        )
    if isinstance(error, TimeoutError):
        return (
            "{} {} did not answer before the inference timeout. Retry, shorten "
            "the task, or choose a faster model.".format(MODEL_NOT_ANSWERING, model)
        )
    if isinstance(error, (urllib.error.URLError, OSError)):
        return _unreachable(
            error, endpoint,
            "Check that the AI model is switched on, then run the agent again.",
            "For operators: Vaelor could not reach {endpoint} from this appliance.",
        )
    return MALFORMED_RESULT_MESSAGE


def plan_failure_warning(
    error: Exception, connection: Dict[str, str], request: str = ""
) -> str:
    model = str(connection.get("model") or "The selected model")[:100]
    detail = str(error).lower()
    capability = model_capability(connection)
    stronger = (
        " This request needs more reliable multi-step reasoning than the selected "
        "lightweight model produced; choose a stronger local or hosted model for "
        "the research synthesis. Built-in safety checks remain available."
        if capability.get("tier") == "basic-local"
        and request_complexity(request) == "complex"
        else ""
    )
    if "reasoning used the full output budget" in detail:
        message = (
            f"{model} used its full response budget for reasoning before producing "
            "the required JSON plan. Vaelor used built-in safety checks; retry or "
            "choose a less verbose instruction model."
        )
    elif "did not return json" in detail or "malformed json" in detail:
        message = (
            f"{model} responded without a valid JSON plan. Vaelor used built-in "
            "safety checks; choose a model with structured JSON support or retry."
        )
    elif "unsupported response schema" in detail:
        message = (
            f"{model} returned an unsupported planning format. Vaelor used built-in "
            "safety checks; choose an instruction model that follows structured JSON prompts."
        )
    elif isinstance(error, urllib.error.HTTPError):
        message = (
            f"{model} was rejected by the model server (HTTP {error.code}). Vaelor "
            "used built-in safety checks; inspect the model server log."
        )
    elif isinstance(error, (TimeoutError, urllib.error.URLError)):
        message = (
            f"{model} did not finish before the inference timeout. Vaelor used "
            "built-in safety checks; retry or choose a faster model."
        )
    else:
        message = (
            f"{model} returned an incomplete planning response. Vaelor used built-in "
            "safety checks; retry or choose another instruction model."
        )
    return message + stronger
