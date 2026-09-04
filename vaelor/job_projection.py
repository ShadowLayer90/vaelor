"""Canonical operator-facing projections for persisted jobs."""

from __future__ import annotations

from typing import Any, Dict, Optional

from .operation_projection import ACTIVE_OPERATION_STATES


_REJECTION_CODES = {
    "dependency_plan_required",
    "invalid_job_request",
    "policy_rejected",
    "validation_failed",
    "validation_rejected",
}
_REJECTION_PREFIXES = (
    "choose ",
    "confirm ",
    "custom compose ",
    "custom services ",
    "docker rejected ",
    "enter a compose ",
    "every custom service ",
    "the compose project name is invalid",
    "type the ",
)
_TERMINAL_OPERATION_STATES = {  # vocabulary: terminal-state
    "completed", "healthy", "failed", "rejected", "cancelled", "superseded",
}


def _is_rejected(job: Dict[str, Any], raw_state: str) -> bool:
    if raw_state == "rejected":
        return True
    if raw_state != "failed":
        return False
    result = job.get("result") if isinstance(job.get("result"), dict) else {}
    code = str(result.get("code", "")).strip().lower()
    if code in _REJECTION_CODES or code.endswith("_rejected"):
        return True
    message = str(job.get("message", "")).strip().lower()
    return message.startswith(_REJECTION_PREFIXES)


def _operation_state(job: Dict[str, Any]) -> str:
    raw_state = str(job.get("state", "unknown")).strip().lower()
    if _is_rejected(job, raw_state):
        return "rejected"
    if raw_state == "queued":
        return "queued"
    if raw_state in {"validating", "downloading", "starting", "running", "cancelling"}:  # vocabulary: active-job-state
        return "running"
    if raw_state == "waiting":
        return "waiting"
    if raw_state in {"needs_approval", "needs-input", "needs_input"}:
        return "needs_approval"
    if raw_state in {"paused", "superseded"}:
        return raw_state
    if raw_state in _TERMINAL_OPERATION_STATES:
        return raw_state
    phase = str(job.get("phase", "")).strip().lower()
    if phase in {"needs_input", "ready_for_review"}:
        return "needs_approval"
    return "unknown"


def _resource_identity(job: Dict[str, Any]) -> tuple[Optional[str], Optional[str]]:
    payload = job.get("payload") if isinstance(job.get("payload"), dict) else {}
    resource_id = next(
        (
            value for value in (
                job.get("resource_id"),
                payload.get("resource_id"),
                payload.get("app_id"),
                payload.get("instance_id"),
                payload.get("model_id"),
                payload.get("id"),
                payload.get("project"),
                payload.get("checkpoint"),
            ) if value is not None and value != ""
        ),
        None,
    )
    display_identity = next(
        (
            value for value in (
                job.get("display_identity"),
                payload.get("display_identity"),
                payload.get("name"),
                payload.get("app_name"),
                payload.get("project"),
                payload.get("model"),
                payload.get("repo"),
            ) if value is not None and value != ""
        ),
        None,
    )
    return (
        str(resource_id) if resource_id is not None else None,
        str(display_identity) if display_identity is not None else None,
    )


def job_projection(job: Dict[str, Any]) -> Dict[str, Any]:
    """Return the one operator-facing projection for a persisted job row.

    Raw executor states are intentionally translated here. Callers should use
    these fields for status, health, readiness, attention, and identity rather
    than deriving a second vocabulary from state or payload.
    """
    operation = _operation_state(job)
    resource_id, display_identity = _resource_identity(job)
    # #179. `liveness` reports whether *this operation* is still going, and it
    # is derived from the same partition `JobStore.ledger` counts `active`
    # from, so the two answers in one `/api/v2/jobs?summary=true` body cannot
    # disagree. They did: `healthy` used to project `live`, and `healthy` is
    # terminal. A JobStore row is immutable once terminal and nothing re-reads
    # the resource, so that was a present-tense claim computed from a fact
    # about the past - true for the instant the job ended and never checked
    # again. Eleven of the forty-five liveness fields in the appliance's own
    # captures were `live`, one of them on a `cluster.initialize` job that ran
    # for 1.6 seconds.
    #
    # Nothing here can say whether the deployed resource is up now; that needs
    # a live probe. What the row does know - that the job ended healthy and
    # ready - is what `health` and `readiness` below carry, and they are
    # records of an observation rather than claims about the present.
    if operation == "running":
        liveness = "running"
    elif operation in ACTIVE_OPERATION_STATES:
        liveness = "waiting"
    elif operation in _TERMINAL_OPERATION_STATES:
        liveness = "stopped"
    else:
        liveness = "unknown"
    if operation == "healthy":
        health = "healthy"
    elif operation == "failed":
        health = "failed"
    elif operation == "rejected":
        health = "rejected"
    else:
        health = "unknown"
    if operation in {"healthy", "completed"}:
        readiness = "ready"
    elif operation in {"failed", "rejected", "cancelled", "superseded"}:
        readiness = "not_ready"
    elif operation == "unknown":
        readiness = "unknown"
    else:
        readiness = "pending"
    attention = operation in {"failed", "needs_approval", "paused", "unknown"}
    retryable = operation in {"failed", "cancelled"}
    retry_of = job.get("retry_of")
    return {
        "operation_state": operation,
        "resource_liveness": liveness,
        "liveness": liveness,
        "health": health,
        "readiness": readiness,
        "attention": attention,
        "retryable": retryable,
        "resource_id": resource_id,
        "display_identity": display_identity,
        "retry_of": str(retry_of) if retry_of else None,
        "phase": str(job.get("phase", "")),
        "blocker_layer": str(job.get("blocker_layer", "")),
    }
