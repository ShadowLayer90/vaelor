"""Shape and bound model results before they reach a task or a screen.

Every model-produced review - specialist or user-defined agent - passes through
here on its way out, so this is the one place that decides which fields survive
and how long they may be.
"""

from __future__ import annotations

from typing import Any, Dict

from .agent_tasks import AgentTaskError

_task_error = AgentTaskError

# Fields a run may carry beyond the fixed review skeleton. `answer` matters
# most: a custom agent's whole purpose can be one direct reply ("did they
# win?"), and dropping it left the run output showing bullet lists and no
# answer at all.
OPTIONAL_RESULT_FIELDS = ("answer", "warnings", "sources", "evidence")

MAX_SUMMARY_CHARS = 1200
#: The summary a review that wrote none carries.
REVIEW_COMPLETED = "Review completed."
MAX_ITEM_CHARS = 400
MAX_ITEMS = 8


def bounded_items(result: Dict[str, Any], name: str) -> list[str]:
    """Coerce one list-shaped field into at most `MAX_ITEMS` bounded strings."""
    value = result.get(name, [])
    if value is None:
        return []
    if isinstance(value, str):
        value = [value]
    if not isinstance(value, list):
        return []
    return [str(item)[:MAX_ITEM_CHARS] for item in value[:MAX_ITEMS]]


#: Fields whose presence means the run produced something a reader can use.
SUBSTANCE_FIELDS = ("findings", "recommendations", "next_actions", "sources", "evidence")

#: What a run delivered, independent of whether it *ran*. A task that finished
#: cleanly having produced nothing is still a task that produced nothing, and
#: reporting it as ``Finished / 100%`` behind a green badge is the
#: success-for-work-not-done defect in the agent path: a run whose whole output
#: was "I cannot provide the most recent NFL scores because the search tool
#: returned no results" was indistinguishable from one that did the job.
OUTCOME_DELIVERED = "delivered"
OUTCOME_NO_RESULT = "no_result"
#: The run could not safely answer and is asking the user for more detail rather
#: than guessing. #247w: a research agent that read nothing relevant from public
#: sources surfaces concrete clarifying questions (in ``clarifications``) instead
#: of a confident training-knowledge answer. Like ``no_result`` this travels in
#: the result under a "completed" state - the run finished, it did not crash - so
#: no surface renders "Finished / 100%" over an unanswerable task.
OUTCOME_NEEDS_INPUT = "needs_input"


def task_outcome(result: Dict[str, Any]) -> Dict[str, Any]:
    """Whether this run produced anything, and what to say when it did not.

    Deliberately structural rather than phrase-matching. A run counts as having
    delivered when it carries findings, recommendations, next actions, sources
    or evidence, or when it changed something. Prose alone does not count for
    an *agent task*: these exist to do work and report it, and an agent that
    explains why it could not has explained rather than done.
    """
    if not isinstance(result, dict):
        return {
            "outcome": OUTCOME_NO_RESULT,
            "outcome_reason": "The agent returned no usable result.",
        }
    if result.get("executed_changes"):
        return {"outcome": OUTCOME_DELIVERED, "outcome_reason": ""}
    if any(result.get(field) for field in SUBSTANCE_FIELDS):
        return {"outcome": OUTCOME_DELIVERED, "outcome_reason": ""}
    stated = " ".join(str(result.get("answer") or result.get("summary") or "").split())
    return {
        "outcome": OUTCOME_NO_RESULT,
        "outcome_reason": (
            "This run finished without producing findings, recommendations, "
            "next actions or sources."
            + (" The agent reported: {}".format(stated[:300]) if stated else "")
        ),
    }


def capable_model_name(agent: Any) -> str:
    """The GPU ai-chat model name an agent would escalate onto, or "".

    "" means the graphics model is unavailable and no escalation is offered.
    Fails closed: any broker trouble reads as unavailable, never an escape.
    """
    try:
        connection = getattr(agent, "_connection", lambda _mode: None)("capable")
    except Exception:
        return ""
    if not connection:
        return ""
    return str(connection.get("model") or connection.get("label") or "").strip() or "the graphics model"


def escalation_offer(
    capable_model: str, *, model_tier: str = "", failed: bool = False,
    degraded: bool = False, outcome: Any = None, fell_back: bool = False,
) -> Dict[str, Any] | None:
    """The capable-model re-run to offer the user, or ``None``.

    Escalation is user-triggered, never automatic. It is offered only when a
    capable GPU model exists (``capable_model`` truthy), the run is not ALREADY
    on the capable tier, and the run underperformed: it failed, delivered
    degraded, produced no result or is asking for input, or an eligible tool loop
    fell back so the model never drove tools. A clean native-tool-loop delivery
    offers nothing. ``model`` names what the re-run would use so the UI can say so.
    """
    if not capable_model or str(model_tier).strip().lower() == "capable":
        return None
    if not (
        failed or degraded
        or outcome in (OUTCOME_NO_RESULT, OUTCOME_NEEDS_INPUT) or fell_back
    ):
        return None
    return {"model_tier": "capable", "model": str(capable_model)[:160], "surface": "ai-chat"}


def failure_escalation(agent: Any, task: Dict[str, Any], *, custom: bool) -> Dict[str, Any] | None:
    """The result a failed run carries: the capable re-run offer, or ``None``.

    A failure is the clearest case for offering the capable GPU model - the NPU
    one did not complete the task at all - unless the run was already on the
    capable tier or no GPU model is available. A custom agent is offered
    nothing: it already runs on AI Chat's model (VD-220). Fails closed.
    """
    if custom:
        return None
    try:
        offer = escalation_offer(
            capable_model_name(agent),
            model_tier=str(task.get("model_tier") or ""), failed=True,
        )
    except Exception:  # noqa: BLE001 - absence-ok: no offer is the safe answer
        return None
    return {"escalation_available": offer} if offer else None


def validate_specialist(
    result: Dict[str, Any], profile: str, source: str
) -> Dict[str, Any]:
    """Return the bounded review skeleton plus any optional fields present."""
    if not isinstance(result, dict):
        raise ValueError("The specialist returned an invalid result.")
    validated: Dict[str, Any] = {
        "profile": profile,
        "source": source,
        "summary": str(result.get("summary", REVIEW_COMPLETED))[:MAX_SUMMARY_CHARS],
        "findings": bounded_items(result, "findings"),
        "recommendations": bounded_items(result, "recommendations"),
        "next_actions": bounded_items(result, "next_actions"),
        "executed_changes": False,
    }
    for key in OPTIONAL_RESULT_FIELDS:
        if result.get(key):
            validated[key] = result[key]
    return validated


def structured_result(value: Any) -> dict[str, Any]:
    """Normalize model output into the fields the UI and audit ledger understand."""
    if not isinstance(value, dict):
        raise _task_error("The selected model did not return a structured result.")

    def strings(key: str, limit: int = 8) -> list[str]:
        raw = value.get(key, [])
        if isinstance(raw, str):
            raw = [raw]
        if not isinstance(raw, list):
            return []
        return [" ".join(str(item).split())[:800] for item in raw[:limit] if str(item).strip()]

    def references(key: str) -> list[dict[str, str]]:
        raw = value.get(key, [])
        if not isinstance(raw, list):
            return []
        result = []
        for item in raw[:20]:
            if not isinstance(item, dict):
                continue
            source = str(item.get("source") or item.get("url") or item.get("title") or "").strip()[:400]
            summary = " ".join(str(item.get("summary") or item.get("excerpt") or "").split())[:800]
            if source:
                result.append({"source": source, "summary": summary})
        return result

    normalized: dict[str, Any] = {
        "source": str(value.get("source", "custom-agent-model"))[:80],
        "summary": " ".join(str(value.get("summary", REVIEW_COMPLETED)).split())[:1200],
        "findings": strings("findings"),
        "recommendations": strings("recommendations"),
        "next_actions": strings("next_actions"),
        "evidence": references("evidence"),
        "sources": references("sources"),
        # A model that recognises an ambiguous or under-specified task may ask
        # for more detail rather than guess. Kept here so the runner can surface
        # it; the runner's own deterministic questions (#247w) are the guarantee
        # when the model does not, since normalization otherwise drops it.
        "clarifications": strings("clarifications"),
        "warnings": strings("warnings"),
        "errors": strings("errors"),
        "executed_changes": False,
        # A run that never reached the model is not the same as one that did.
        # This was previously visible only as the first line of the findings
        # list, inside a collapsed disclosure, under a green "completed" badge.
        "degraded": bool(value.get("fallback_used")),
    }
    answer = " ".join(str(value.get("answer", "")).split())[:5000]
    if answer:
        normalized["answer"] = answer
    return normalized
