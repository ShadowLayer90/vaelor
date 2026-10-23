"""What resolves a failed operation: a successful retry, or a later converged run.

VD-139 resolved a failure only by a successful retry of it. VD-133 1.5.5 #4:
a worker update cut off by a controller restart was brought up to date by the
automatic recheck four minutes later - a new job, not a retry - and stayed red.
So a later run of the same operation on the same target resolves it too, when
that run actually converged the target. The failure is never rewritten: what
resolved it and when is attached beside it.

The attention verdict that reads these is `operation_projection.needs_attention`.
"""

from __future__ import annotations

import bisect
import json
from typing import Any, Callable, Dict, Mapping, Optional

from .job_vocabulary import CLUSTER_NODE_PROFILE_JOB

#: A later attempt in one of these states may resolve a failure.
RETRY_RESOLVING_STATES = frozenset({"completed", "healthy"})
#: The states a later attempt can resolve (the ones an operation is retried from).
RETRY_RESOLVABLE_STATES = frozenset({"failed", "cancelled"})


def _profile_converged(payload: Mapping[str, Any], result: Mapping[str, Any]) -> bool:
    """A worker profile run converged only when it did what its action asks.

    ``apply`` converged when it applied or found the worker current
    (`worker_profile_apply.SUCCESSES`); ``remove`` when it took the profile
    off (`worker_profile_job.REMOVED`). A run that ends ``completed`` because
    the worker did not answer (deferred) or was held serving changed nothing
    and says nothing about the worker (review F1). Imported lazily: the
    profile code pulls in the SSH transport.
    """
    if payload.get("action") == "remove":
        from .worker_profile_job import REMOVED

        return result.get("outcome") == REMOVED
    from .worker_profile_apply import SUCCESSES

    return result.get("outcome") in SUCCESSES


class ConvergentType:
    """A job type that re-reads its target and converges it.

    Two runs carry different payloads - why it was queued, a digest read at
    queue time - and still act on the same thing, so ``keys`` name the
    payload fields that ARE the target. ``converged(payload, result)`` says whether a
    run that ended well actually brought the target to the wanted state;
    only such a run resolves an earlier failure.
    """

    def __init__(self, keys: tuple, converged: Callable[[Mapping[str, Any], Mapping[str, Any]], bool]):
        self.keys = keys
        self.converged = converged


CONVERGENT_TYPES: Dict[str, ConvergentType] = {
    CLUSTER_NODE_PROFILE_JOB: ConvergentType(("node_id", "action"), _profile_converged),
}


def operation_target(job_type: object, payload: Any) -> Optional[str]:
    """What a job acts on, as one comparable key, or ``None`` when unreadable.

    Every type but a convergent one is targeted by its whole payload (an
    identical request, which is what a retry copies). A convergent type
    missing one of its target keys has no target: never matched by guesswork.
    """
    if not isinstance(payload, dict):
        return None
    kind = CONVERGENT_TYPES.get(str(job_type))
    if kind is None:
        named: Any = payload
    elif any(payload.get(key) in (None, "") for key in kind.keys):
        return None
    else:
        named = {key: payload[key] for key in kind.keys}
    return json.dumps([str(job_type), named], sort_keys=True, separators=(",", ":"), default=str)


def job_succeeded(record: Mapping[str, Any]) -> bool:
    """Whether a job row may resolve an earlier failure: it ended well and,
    for a convergent type, it converged its target (review F1)."""
    if record.get("operation_state") not in RETRY_RESOLVING_STATES:
        return False
    kind = CONVERGENT_TYPES.get(str(record.get("type", "")))
    if kind is None:
        return True
    result, payload = record.get("result"), record.get("payload")
    return (isinstance(result, Mapping) and isinstance(payload, Mapping)
            and bool(kind.converged(payload, result)))


def retry_resolvers(records: Any, succeeded: Any, state_of: Any) -> Dict[str, Dict[str, Any]]:
    """``{id: the attempt that resolved it}`` for every record a later
    SUCCESSFUL attempt retried, transitively.

    ``records`` carry ``id`` and ``retry_of``; ``succeeded(record)`` says
    whether an attempt may resolve and ``state_of(record)`` is its canonical
    state. A succeeding attempt resolves every ancestor on its ``retry_of``
    chain in :data:`RETRY_RESOLVABLE_STATES`. An attempt still running, or one
    that failed too, resolves nothing. Ancestors outside ``records`` end the walk.
    """
    by_id = {str(record["id"]): record for record in records}
    resolved: Dict[str, Dict[str, Any]] = {}
    for record in records:
        if not succeeded(record):
            continue
        ancestor = record.get("retry_of")
        visited: set = set()
        while ancestor and str(ancestor) not in visited:
            visited.add(str(ancestor))
            prior = by_id.get(str(ancestor))
            if prior is None:
                break
            if state_of(prior) in RETRY_RESOLVABLE_STATES:
                resolved.setdefault(str(prior["id"]), record)
            ancestor = prior.get("retry_of")
    return resolved


def retry_resolved_ids(records: Any, state_of: Any) -> set:
    """The ids a later attempt in :data:`RETRY_RESOLVING_STATES` retried."""
    return set(retry_resolvers(
        records, lambda record: state_of(record) in RETRY_RESOLVING_STATES, state_of))


def _number(value: Any) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return float("nan")


def later_run_resolvers(
    records: Any, succeeded: Any, state_of: Any, target_of: Any,
) -> Dict[str, Dict[str, Any]]:
    """``{id: the run that resolved it}``: failures a LATER succeeding run of
    the same operation on the same target settled (VD-133 1.5.5 #4).

    ``target_of`` is :func:`operation_target` (``None`` never matches). A
    resolver succeeded, was created no earlier than the failure and finished no
    earlier; the first such run by creation is named. Successes are sorted once
    per target and found by bisection, so a target with thousands of runs (a
    scale-to-zero load/unload loop) stays near linear (review F4).
    """
    successes: Dict[str, list] = {}
    for record in records:
        target = target_of(record)
        created = _number(record.get("created_at"))
        if target is not None and created == created and succeeded(record):
            successes.setdefault(target, []).append((created, _number(record.get("updated_at")), record))
    starts: Dict[str, list] = {}
    for target, runs in successes.items():
        runs.sort(key=lambda item: item[0])
        starts[target] = [item[0] for item in runs]
    resolved: Dict[str, Dict[str, Any]] = {}
    for record in records:
        if state_of(record) not in RETRY_RESOLVABLE_STATES:
            continue
        target = target_of(record)
        runs = successes.get(target) if target is not None else None
        created, finished = _number(record.get("created_at")), _number(record.get("updated_at"))
        if not runs or created != created or finished != finished:
            continue
        for _start, end, run in runs[bisect.bisect_left(starts[target], created):]:
            if run is not record and end >= finished:
                resolved[str(record["id"])] = run
                break
    return resolved
