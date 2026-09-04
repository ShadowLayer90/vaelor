"""Server-owned ``vaelor.operation.v1`` projections across durable ledgers.

The JobStore and AgentTaskStore remain the immutable source ledgers.  This
module adapts their current records into one operator-facing contract; it
does not update, merge, or otherwise rewrite either ledger while projecting.
"""

from __future__ import annotations

import re
import time
from typing import Any, Dict, Optional
from urllib.parse import quote


OPERATION_SCHEMA = "vaelor.operation.v1"
CANONICAL_STATES = frozenset({
    "draft", "needs_approval", "ready", "queued", "running", "waiting",
    "paused", "completed", "healthy", "failed", "rejected", "cancelled",
    "superseded",
})
LEDGER_JOBS = "jobs"
LEDGER_AGENT_TASKS = "agent_tasks"
LEDGERS = frozenset({LEDGER_JOBS, LEDGER_AGENT_TASKS})
TERMINAL_STATES = frozenset({  # vocabulary: terminal-state
    "completed", "healthy", "failed", "rejected", "cancelled", "superseded",
})
#: The states in which an operation has not finished and something is expected
#: to move it. With ``TERMINAL_STATES`` and ``draft`` this partitions
#: ``CANONICAL_STATES`` exactly, and
#: ``tests/test_operation_activity.py`` checks that in both directions, so a
#: fourteenth canonical state cannot be added without saying which side it is
#: on. ``draft`` is on neither: a saved draft is unfinished and nothing is
#: working on it.
#:
#: Written out rather than derived from the two sets above, because the guard
#: in ``tests/test_wire_vocabularies.py`` reads an owning constant out of the
#: source tree and a subtraction is not a declaration a reader can see. The
#: partition test is what keeps the two spellings honest.
#:
#: #201: ``vaelor/jobs.py`` counted the ledger's ``active`` rows from five of
#: these and ``frontend/src/lib/operationOwner.ts`` gated its polling on all
#: six. Both were right for the ledger each reads - ``ready`` is unreachable
#: for a job row and reachable for an agent task - and nothing related them, so
#: a state added to one side would have moved only that side.
ACTIVE_OPERATION_STATES = frozenset({
    "needs_approval", "ready", "queued", "running", "waiting", "paused",
})
#: #180. Staleness is a server-owned verdict, not a client computation.
#: ``frontend/src/components/OperationOwner.tsx`` decided on its own whether a
#: ``running`` operation had gone quiet, holding both the one-hour threshold and
#: the "which states can even stall" set; the projection carried nothing, so the
#: contract could not answer the question the card was already drawing. There is
#: no heartbeat - ``heartbeat_at`` is NULL on every row - so the only signal is
#: the last update, and the only state where silence means "stopped reporting"
#: rather than "waiting for a human" (``needs_approval``/``paused``) or "working
#: slowly on a large pull" (``queued``/``ready``/``waiting``) is ``running``.
#: The frontend narrowed to exactly this single-member set for the same reason;
#: the server owns it here and the frontend renders ``staleness.stale``.
OPERATION_STALE_AFTER_SECONDS = 60 * 60
STALE_REPORTING_STATES = frozenset({"running"})
_OPERATION_ID = re.compile(r"^(jobs|agent_tasks):([^:]+)$")
_SECRET_KEY = re.compile(
    r"(?:password|passwd|token|secret|api[_-]?key|authorization|credential|private[_-]?key)",
    re.IGNORECASE,
)
_SECRET_VALUE = re.compile(
    r"(?:password|passwd|token|secret|api[_-]?key|authorization)\s*[:=]\s*\S+|"
    r"\b(?:bearer|basic)\s+\S+|-----BEGIN [A-Z ]*PRIVATE KEY-----",
    re.IGNORECASE,
)


class OperationNotFound(KeyError):
    """Raised when an operation is not visible to the requesting actor."""


class OperationActionError(ValueError):
    """Raised when a source-ledger action cannot be applied safely."""


def operation_id(ledger: str, source_id: str) -> str:
    """Return the immutable, globally unique operation identifier."""
    if ledger not in LEDGERS:
        raise ValueError("Choose a supported operation ledger.")
    value = str(source_id).strip()
    if not value or ":" in value:
        raise ValueError("The source operation identifier is invalid.")
    return f"{ledger}:{value}"


def parse_operation_id(value: str) -> tuple[str, str]:
    match = _OPERATION_ID.fullmatch(str(value))
    if match is None:
        raise OperationNotFound(value)
    return match.group(1), match.group(2)


def _redact(value: Any, *, depth: int = 0) -> Any:
    """Return bounded, JSON-safe output without exposing credential values."""
    if depth > 6:
        return "[truncated]"
    if isinstance(value, dict):
        result: dict[str, Any] = {}
        for key, child in value.items():
            name = str(key)
            result[name] = "[redacted]" if _SECRET_KEY.search(name) else _redact(
                child, depth=depth + 1
            )
        return result
    if isinstance(value, (list, tuple)):
        return [_redact(item, depth=depth + 1) for item in list(value)[:200]]
    if isinstance(value, str):
        return "[redacted]" if _SECRET_VALUE.search(value) else value[:4000]
    if isinstance(value, (int, float, bool)) or value is None:
        return value
    return str(value)[:500]


def _source_revision(record: Dict[str, Any], *, fallback: int = 1) -> int:
    value = record.get("revision")
    if isinstance(value, int) and not isinstance(value, bool) and value > 0:
        return value
    updated = record.get("updated_at")
    try:
        # Jobs use integer milliseconds; tasks use floating-point seconds.
        numeric = float(updated)
        return max(fallback, int(numeric if numeric > 10_000_000_000 else numeric * 1000))
    except (TypeError, ValueError):
        return fallback


def _updated_sort_value(record: Dict[str, Any]) -> float:
    """Normalize JobStore milliseconds and task-store seconds for one sort."""
    try:
        value = float(record.get("updated_at", 0) or 0)
    except (TypeError, ValueError):
        return 0.0
    return value / 1000.0 if value > 100_000_000_000 else value


def _activity_seconds(record: Dict[str, Any]) -> Optional[float]:
    """When this row last moved, in seconds, or ``None`` if it never has.

    JobStore writes integer milliseconds and the task store floating-point
    seconds (see ``_source_revision``); both normalize here the way
    ``_updated_sort_value`` does, so one threshold measures both. ``started_at``
    is the fallback the frontend used, for a job that began and reported nothing
    since.
    """
    for key in ("updated_at", "started_at"):
        try:
            value = float(record.get(key))
        except (TypeError, ValueError):
            continue
        if value <= 0:
            continue
        return value / 1000.0 if value > 100_000_000_000 else value
    return None


def _staleness(record: Dict[str, Any], state: str, *, now: float) -> Dict[str, Any]:
    """The server's verdict on whether a reporting operation has gone quiet.

    Only ``running`` can be stale (``STALE_REPORTING_STATES``); a missing
    timestamp is not evidence of a stall, so it is reported as not stale rather
    than invented in either direction. ``now`` is injected so the verdict is a
    pure function of the record and the clock, and the guard can pin the hour
    boundary without racing a real one.
    """
    reporting = state in STALE_REPORTING_STATES
    last = _activity_seconds(record) if reporting else None
    silent = None if last is None else max(0.0, now - last)
    return {
        "reporting": reporting,
        "stale": bool(silent is not None and silent > OPERATION_STALE_AFTER_SECONDS),
        "silent_seconds": None if silent is None else int(silent),
        "stale_after_seconds": OPERATION_STALE_AFTER_SECONDS,
    }


def _timestamp_fields(record: Dict[str, Any], state: str) -> Dict[str, Any]:
    created = record.get("created_at")
    updated = record.get("updated_at")
    return {
        "created_at": created,
        "updated_at": updated,
        "started_at": record.get("started_at"),
        "completed_at": updated if state in TERMINAL_STATES else None,
    }


def _progress(
    record: Dict[str, Any], state: str, *, has_progress: bool = True
) -> Dict[str, Any]:
    raw = record.get("progress") if has_progress else None
    if state in {"queued", "draft", "ready", "waiting", "needs_approval", "paused"}:
        value = None
        determinate = False
    elif state in {"completed", "healthy"}:
        value = 100
        determinate = True
    elif isinstance(raw, (int, float)) and not isinstance(raw, bool):
        value = max(0, min(100, int(raw)))
        determinate = True
    else:
        value = None
        determinate = False
    return {
        "value": value,
        "percent": value,
        "determinate": determinate,
        "label": "Complete" if value == 100 else (
            f"{value}%" if determinate and value is not None else "In progress"
        ),
    }


def _job_state(record: Dict[str, Any]) -> str:
    projected = str(record.get("operation_state", "")).strip().lower()
    if projected in CANONICAL_STATES:
        return projected
    raw = str(record.get("state", "")).strip().lower()
    if raw in {"queued"}:
        return "queued"
    if raw in {"validating", "downloading", "starting", "running", "cancelling"}:  # vocabulary: active-job-state
        return "running"
    if raw in {"waiting", "blocked"}:
        return "waiting"
    if raw in {"needs_approval", "needs-input", "needs_input"}:
        return "needs_approval"
    if raw in CANONICAL_STATES:
        return raw
    return "waiting"


def _task_state(record: Dict[str, Any]) -> str:
    raw = str(record.get("state", "")).strip().lower()
    return {
        "triage": "draft",
        "ready": "ready",
        "running": "running",
        "needs_approval": "needs_approval",
        "blocked": "waiting",
        "completed": "completed",
        "failed": "failed",
        "cancelled": "cancelled",
        "archived": "superseded",
    }.get(raw, "waiting")


def _job_owner(record: Dict[str, Any]) -> tuple[str, Optional[str]]:
    payload = record.get("payload") if isinstance(record.get("payload"), dict) else {}
    job_type = str(record.get("type", ""))
    resource = next(
        (
            value for value in (
                record.get("resource_id"), payload.get("resource_id"),
                payload.get("draft_id"), payload.get("app_id"),
                payload.get("instance_id"), payload.get("model_id"),
                payload.get("project"), payload.get("id"),
            ) if value not in (None, "")
        ),
        None,
    )
    if job_type.startswith("application.") or job_type.startswith("compose."):
        route = "/workloads/applications" if payload.get("draft_id") else "/workloads"
    elif job_type.startswith("model.") or job_type == "agent.deploy":
        route = "/workloads/models"
    elif job_type.startswith("cluster."):
        route = "/fleet"
    elif job_type.startswith("host.") or job_type.startswith("system."):
        route = "/system"
    else:
        route = "/operations"
    return route, str(resource) if resource not in (None, "") else None


def _task_owner(record: Dict[str, Any]) -> tuple[str, Optional[str]]:
    profile = str(record.get("profile", "")).strip()
    return "/assistant/agents", str(record.get("id", "")) or profile or None


def _error(record: Dict[str, Any], state: str) -> tuple[Dict[str, Any], Dict[str, Any]]:
    result = record.get("result") if isinstance(record.get("result"), dict) else {}
    message = str(record.get("error") or result.get("message") or "")[:2000]
    recoverable = state in {"failed", "waiting", "needs_approval", "paused", "cancelled"}
    error = {
        "code": str(result.get("code", "operation_failed" if message else ""))[:120],
        "message": message,
        "recoverable": recoverable,
    }
    correction = {
        "available": recoverable and bool(message),
        "message": message if recoverable and message else "",
        "fields": _redact(result.get("correction_fields", {}))
        if isinstance(result.get("correction_fields", {}), dict) else {},
    }
    return error, correction


def _lineage(
    ledger: str, record: Dict[str, Any], by_source_id: Optional[Dict[str, Dict[str, Any]]] = None
) -> Dict[str, Any]:
    source_id = str(record.get("id", ""))
    parent_source = record.get("retry_of")
    ancestry: list[str] = []
    visited: set[str] = set()
    while parent_source and str(parent_source) not in visited:
        parent_key = str(parent_source)
        visited.add(parent_key)
        ancestry.append(operation_id(ledger, parent_key))
        parent = (by_source_id or {}).get(parent_key)
        if parent is None:
            break
        parent_source = parent.get("retry_of")
    return {
        "parent_operation_id": operation_id(ledger, str(record["retry_of"]))
        if record.get("retry_of") else None,
        "root_operation_id": ancestry[-1] if ancestry else operation_id(ledger, source_id),
        "depth": len(ancestry),
        "attempt": int(record.get("attempt") or record.get("depth") or 1),
        "ancestry": ancestry,
    }


def _endpoints(
    op_id: str, permissions: Dict[str, bool], *, include_discard: bool = False
) -> Dict[str, Optional[str]]:
    base = f"/api/v2/operations/{quote(op_id, safe=':')}"
    endpoints: Dict[str, Optional[str]] = {
        "self": base,
        "cancel": f"{base}/cancel" if permissions.get("cancel") else None,
        "retry": f"{base}/retry" if permissions.get("retry") else None,
        "resume": f"{base}/resume" if permissions.get("resume") else None,
        "audit": f"{base}/audit",
    }
    if include_discard:
        endpoints["discard"] = f"{base}/discard"
    return endpoints


def project_job(
    record: Dict[str, Any], *, by_source_id: Optional[Dict[str, Dict[str, Any]]] = None,
    detail: bool = False, now: Optional[float] = None,
) -> Dict[str, Any]:
    """Adapt one JobStore record into the versioned operation contract."""
    now = time.time() if now is None else float(now)
    source_id = str(record["id"])
    op_id = operation_id(LEDGER_JOBS, source_id)
    state = _job_state(record)
    route, resource = _job_owner(record)
    raw_state = str(record.get("state", ""))
    permissions = {
        "cancel": raw_state not in {"completed", "healthy", "failed", "rejected", "cancelled", "superseded"},  # vocabulary: terminal-state
        "retry": (raw_state == "failed" and state == "failed") or raw_state == "cancelled",
        "resume": False,
        "discard": False,
    }
    permissions["can_cancel"] = permissions["cancel"]
    permissions["can_retry"] = permissions["retry"]
    error, correction = _error(record, state)
    result = _redact(record.get("result") if isinstance(record.get("result"), dict) else {})
    cleanup_state = "requested" if raw_state == "cancelling" else (
        "completed" if state == "cancelled" else "not_required"
    )
    operation = {
        "schema": OPERATION_SCHEMA,
        "operation_id": op_id,
        "operation_key": f"jobs/{source_id}",
        "ledger": LEDGER_JOBS,
        "source_id": source_id,
        "revision": _source_revision(record),
        "owner": {"route": route, "resource": resource},
        "owner_route": route,
        "owner_resource": resource,
        "type": str(record.get("type", "job")),
        "state": state,
        "canonical_state": state,
        "source_state": raw_state,
        "phase": str(record.get("phase", "")),
        "progress": _progress(record, state),
        "message": str(record.get("message", ""))[:2000],
        "error": error,
        "recoverable_error": error,
        "correction": correction,
        "permissions": permissions,
        "result_summary": {"available": bool(result), "data": result},
        "result": result,
        "artifacts": result.get("artifacts", []) if isinstance(result, dict) and isinstance(result.get("artifacts"), list) else [],
        "endpoint": f"/api/v2/operations/{quote(op_id, safe=':')}",
        "audit_link": f"/api/v2/operations/{quote(op_id, safe=':')}/audit",
        "retry_lineage": _lineage(LEDGER_JOBS, record, by_source_id),
        "timestamps": _timestamp_fields(record, state),
        "staleness": _staleness(record, state, now=now),
        "cleanup": {"state": cleanup_state, "message": ""},
    }
    operation["action_endpoints"] = _endpoints(op_id, permissions)
    operation["endpoints"] = operation["action_endpoints"]
    if detail:
        operation["events"] = _redact(record.get("events", []))
    return operation


def project_agent_task(
    record: Dict[str, Any], *, by_source_id: Optional[Dict[str, Dict[str, Any]]] = None,
    detail: bool = False, now: Optional[float] = None,
) -> Dict[str, Any]:
    """Adapt one AgentTaskStore record into the versioned operation contract."""
    now = time.time() if now is None else float(now)
    source_id = str(record["id"])
    op_id = operation_id(LEDGER_AGENT_TASKS, source_id)
    state = _task_state(record)
    route, resource = _task_owner(record)
    raw_state = str(record.get("state", ""))
    permissions = {
        "cancel": raw_state not in {"completed", "failed", "cancelled", "archived"},
        "retry": raw_state in {"failed", "cancelled", "blocked"},
        "resume": False,
        "discard": False,
    }
    permissions["can_cancel"] = permissions["cancel"]
    permissions["can_retry"] = permissions["retry"]
    error, correction = _error(record, state)
    result = _redact(record.get("result") if isinstance(record.get("result"), dict) else {})
    operation = {
        "schema": OPERATION_SCHEMA,
        "operation_id": op_id,
        "operation_key": f"agent_tasks/{source_id}",
        "ledger": LEDGER_AGENT_TASKS,
        "source_id": source_id,
        "revision": _source_revision(record),
        "owner": {"route": route, "resource": resource},
        "owner_route": route,
        "owner_resource": resource,
        "type": "agent.task",
        "state": state,
        "canonical_state": state,
        "source_state": raw_state,
        "phase": (
            "approval" if state == "needs_approval" else
            "execution" if state == "running" else
            "result" if state in TERMINAL_STATES else "planning"
        ),
        "progress": _progress(record, state, has_progress=False),
        "message": str(record.get("error") or record.get("title") or "")[:2000],
        "error": error,
        "recoverable_error": error,
        "correction": correction,
        "permissions": permissions,
        "result_summary": {"available": bool(result), "data": result},
        "result": result,
        "artifacts": _redact(record.get("artifacts", [])) if isinstance(record.get("artifacts"), list) else [],
        "endpoint": f"/api/v2/operations/{quote(op_id, safe=':')}",
        "audit_link": f"/api/v2/operations/{quote(op_id, safe=':')}/audit",
        "retry_lineage": _lineage(LEDGER_AGENT_TASKS, record, by_source_id),
        "timestamps": _timestamp_fields(record, state),
        "staleness": _staleness(record, state, now=now),
        "cleanup": {
            "state": "completed" if state == "cancelled" else "not_required",
            "message": "",
        },
    }
    operation["action_endpoints"] = _endpoints(op_id, permissions)
    operation["endpoints"] = operation["action_endpoints"]
    if detail:
        operation["events"] = _redact(record.get("events", []))
        operation["comments"] = _redact(record.get("comments", []))
        operation["approvals"] = _redact(record.get("approvals", []))
        operation["dependencies"] = _redact(record.get("dependencies", []))
    return operation


class OperationProjection:
    """Read/adapt both source ledgers and delegate actions to their owners."""

    def __init__(self, job_store: Any, agent_task_store: Any):
        self.job_store = job_store
        self.agent_task_store = agent_task_store

    def list(self, actor: Optional[str] = None, *, ledger: Optional[str] = None, limit: int = 50) -> list[Dict[str, Any]]:
        safe_limit = max(1, min(int(limit), 200))
        if ledger is not None and ledger not in LEDGERS:
            raise ValueError("Choose jobs or agent_tasks as the operation ledger.")
        rows: list[tuple[str, Dict[str, Any]]] = []
        if ledger in (None, LEDGER_JOBS) and self.job_store is not None:
            try:
                source = self.job_store.ledger(200, actor=actor).get("jobs", [])
            except AttributeError:
                source = self.job_store.list(safe_limit, actor=actor)
            rows.extend((LEDGER_JOBS, item) for item in source)
        if ledger in (None, LEDGER_AGENT_TASKS) and self.agent_task_store is not None:
            rows.extend(
                (LEDGER_AGENT_TASKS, item)
                for item in self.agent_task_store.list(actor=actor, limit=200)
            )
        rows.sort(key=lambda item: _updated_sort_value(item[1]), reverse=True)
        job_map = {str(item["id"]): item for source, item in rows if source == LEDGER_JOBS}
        task_map = {str(item["id"]): item for source, item in rows if source == LEDGER_AGENT_TASKS}
        projected = [
            project_job(item, by_source_id=job_map)
            if source == LEDGER_JOBS else project_agent_task(item, by_source_id=task_map)
            for source, item in rows[:safe_limit]
        ]
        return projected

    def get(self, value: str, actor: Optional[str] = None, *, detail: bool = True) -> Dict[str, Any]:
        ledger, source_id = parse_operation_id(value)
        if ledger == LEDGER_JOBS:
            record = self.job_store.get(source_id, actor=actor) if self.job_store is not None else None
            if record is None:
                raise OperationNotFound(value)
            record["events"] = self.job_store.events(source_id, actor=actor) if detail else []
            source = self.job_store.list(200, actor=actor)
            by_id = {str(item["id"]): item for item in source}
            return project_job(record, by_source_id=by_id, detail=detail)
        if self.agent_task_store is None:
            raise OperationNotFound(value)
        visible = self.agent_task_store.get(source_id, actor=actor)
        if visible is None:
            raise OperationNotFound(value)
        record = (
            self.agent_task_store.details(source_id, visible["actor"])
            if detail else visible
        )
        source = self.agent_task_store.list(actor=actor, limit=200)
        by_id = {str(item["id"]): item for item in source}
        return project_agent_task(record, by_source_id=by_id, detail=detail)

    def cancel(self, value: str, actor: str, *, allow_all: bool = False) -> Dict[str, Any]:
        ledger, source_id = parse_operation_id(value)
        # #205 coverage: a terminal operation cannot be cancelled - say so,
        # rather than letting the store raise a bare KeyError whose str() is
        # just the quoted id. Mirrors `retry`'s pre-check below; the projection
        # already sets permissions.can_cancel=False for terminal ops.
        visible = self.get(value, None if allow_all else actor, detail=False)
        if not visible["permissions"].get("cancel", False):
            raise OperationActionError(
                "This operation cannot be cancelled: it has already ended."
            )
        if ledger == LEDGER_JOBS:
            if self.job_store is None:
                raise OperationActionError("Job operations are unavailable.")
            try:
                record = self.job_store.request_cancel(source_id, actor=None if allow_all else actor)
            except (KeyError, ValueError) as error:
                raise OperationActionError(str(error)) from error
        else:
            if self.agent_task_store is None:
                raise OperationActionError("Agent task operations are unavailable.")
            try:
                record = self.agent_task_store.cancel(source_id, actor, allow_all=allow_all)
            except (KeyError, ValueError) as error:
                raise OperationActionError(str(error)) from error
        return self.get(operation_id(ledger, source_id), None if allow_all else actor)

    def retry(self, value: str, actor: str, *, allow_all: bool = False) -> Dict[str, Any]:
        ledger, source_id = parse_operation_id(value)
        visible = self.get(value, None if allow_all else actor, detail=False)
        if not visible["permissions"].get("retry", False):
            raise OperationActionError("This operation cannot be retried in its current state.")
        if ledger == LEDGER_JOBS:
            if self.job_store is None:
                raise OperationActionError("Job operations are unavailable.")
            try:
                created = self.job_store.retry(source_id, actor, allow_all=allow_all)
            except (KeyError, ValueError) as error:
                raise OperationActionError(str(error)) from error
        else:
            if self.agent_task_store is None:
                raise OperationActionError("Agent task operations are unavailable.")
            try:
                created = self.agent_task_store.retry(source_id, actor, allow_all=allow_all)
            except (KeyError, ValueError) as error:
                raise OperationActionError(str(error)) from error
        child_id = str(created.get("id", ""))
        return self.get(operation_id(ledger, child_id), None if allow_all else actor)


__all__ = [
    "CANONICAL_STATES", "LEDGER_AGENT_TASKS", "LEDGER_JOBS", "LEDGERS",
    "OPERATION_SCHEMA", "OPERATION_STALE_AFTER_SECONDS", "STALE_REPORTING_STATES",
    "OperationActionError", "OperationNotFound",
    "OperationProjection", "operation_id", "parse_operation_id",
    "project_agent_task", "project_job",
]
