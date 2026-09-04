"""Mint job *proposals* for the acting tools. Nothing here executes anything.

VD-051 is emphatic that acting must not open a second execution path: there is
one executor, one `JobStore`, one operator-approval seam (the authenticated,
CSRF-checked ``POST /api/v2/jobs``). An acting tool is a new *producer* of
proposals, never a second consumer of them. So every function in this module
returns a descriptor - a job ``type`` drawn from the existing
``ALLOWED_JOB_TYPES`` allowlist plus a bounded ``payload`` - and touches no
store, no executor, and no Docker. The operator's click on the existing job
endpoint is what turns a descriptor into a job; the out-of-process executor is
what runs it; ``security.audit`` is what records it.

Two callers exist:

* The Assistant's own acting tools (`vaelor.assistant_acting_tools`) call
  :func:`build_job_proposal`, which binds the proposal to a fact the read tools
  produced *this turn* - VD-051 constraint 2, "nothing acts on evidence it did
  not establish".
* The installed-agent path (`vaelor.api_agent_operation_routes`) calls
  :func:`resolve_agent_operation`, which additionally refuses any operation
  outside the agent's **server-owned** pinned grant snapshot - the #34 rule,
  that an agent's envelope is never derived from the model's own output.

Phase 1 (the Pi gate) ships exactly the two reversible operations VD-051
constraint 3 puts first: restart before redeploy, redeploy before remove. Both
route to executor handlers that already exist (`compose.restart`,
`compose.update` - the health-checked, auto-rolling-back update). No new
executor code is added, here or anywhere.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, Mapping, Optional


class ActingProposalError(ValueError):
    """A safe, user-presentable refusal to mint an acting proposal."""


@dataclass(frozen=True)
class _ActingOperation:
    """One acting operation: the tool name a producer offers, and the existing
    executor job it routes to. ``job_type`` is validated against
    ``ALLOWED_JOB_TYPES`` at import so a typo here cannot invent a new job."""

    tool: str
    job_type: str
    scope: str
    verb: str
    evidence_tool: str
    reversible_note: str


#: The scope an acting tool requires. New ``*:act`` scopes are inert by
#: default: no role and no agent carries one unless an operator grants it (see
#: `assistant_acting_tools` and the granted-scopes gate it reuses).
WORKLOADS_ACT_SCOPE = "workloads:act"

_ACTING_OPERATIONS: Dict[str, _ActingOperation] = {
    "workloads.restart": _ActingOperation(
        tool="workloads.restart",
        job_type="compose.restart",
        scope=WORKLOADS_ACT_SCOPE,
        verb="restart",
        evidence_tool="workloads.inventory",
        reversible_note="A restart is reversible: the app comes back on the same configuration.",
    ),
    "workloads.redeploy": _ActingOperation(
        tool="workloads.redeploy",
        job_type="compose.update",
        scope=WORKLOADS_ACT_SCOPE,
        verb="redeploy",
        evidence_tool="workloads.inventory",
        reversible_note=(
            "A redeploy pulls replacement images, verifies health, and rolls "
            "back to the previous images automatically if the new ones are "
            "unhealthy."
        ),
    ),
}

#: The complete envelope an *installed agent* may propose within. It is a
#: server-owned constant, never read from a task result or a model answer - the
#: #34 rule. An agent still needs a pinned grant for each operation on top of
#: this (see :func:`authorized_operations`); the envelope is the outer bound.
INSTALLED_AGENT_OPERATIONS = frozenset(_ACTING_OPERATIONS)


def _assert_job_types_exist() -> None:
    """Every operation routes to an existing executor job, checked at import.

    LESSONS 15 / VD-051 #96: acting adds no executor code. If a `job_type` here
    is not already in `ALLOWED_JOB_TYPES`, this raises at import rather than
    letting a proposal be minted for a job the one executor cannot run.
    """
    from .job_vocabulary import ALLOWED_JOB_TYPES

    missing = sorted(
        operation.job_type
        for operation in _ACTING_OPERATIONS.values()
        if operation.job_type not in ALLOWED_JOB_TYPES
    )
    if missing:
        raise RuntimeError(
            "LESSONS 15 / VD-051 #96: acting tools must route to an EXISTING "
            "executor job type. These are not in ALLOWED_JOB_TYPES "
            "(vaelor/job_vocabulary.py) and no new executor code may be added "
            "to carry them: {}. Route the operation to a job the executor "
            "already handles, or do not ship it.".format(missing)
        )


_assert_job_types_exist()


def operation_catalog() -> list[Dict[str, str]]:
    """The acting operations and the existing jobs they route to, for display."""
    return [
        {
            "tool": operation.tool,
            "job_type": operation.job_type,
            "scope": operation.scope,
            "verb": operation.verb,
            "evidence_tool": operation.evidence_tool,
        }
        for operation in sorted(_ACTING_OPERATIONS.values(), key=lambda item: item.tool)
    ]


def managed_projects(inventory_fact: Any) -> frozenset[str]:
    """The managed Compose projects a `workloads.inventory` reading named.

    A proposal may only target one of these. The set comes from an inventory
    *reading* - either this turn's fact (the Assistant path) or a live snapshot
    taken at approval time (the installed-agent path) - never from a name the
    model supplied on its own.
    """
    apps = inventory_fact.get("apps") if isinstance(inventory_fact, Mapping) else None
    names: set[str] = set()
    for app in apps or []:
        if not isinstance(app, Mapping):
            continue
        project = app.get("project")
        if app.get("managed") and isinstance(project, str) and project.strip():
            names.add(project.strip())
    return frozenset(names)


def _target_project(arguments: Any, verb: str) -> str:
    if not isinstance(arguments, Mapping):
        raise ActingProposalError("Acting tool arguments must be an object.")
    project = str(arguments.get("project", "")).strip()
    if not project:
        raise ActingProposalError(
            "Name the managed app (its Compose project) to {}.".format(verb)
        )
    return project


def build_job_proposal(
    operation: str,
    arguments: Mapping[str, Any],
    evidence_facts: Optional[Mapping[str, Any]],
) -> Dict[str, Any]:
    """Return an inert job proposal for one acting operation, or refuse.

    VD-051 constraint 2 in one function: the target must appear among the
    managed apps that the operation's read tool reported *this turn*. If the
    read tools never established the app, no proposal is minted - an Assistant
    cannot act on a fault it invented, because it can only propose against what
    it just read.
    """
    spec = _ACTING_OPERATIONS.get(str(operation))
    if spec is None:
        raise ActingProposalError("Unknown acting operation: {}.".format(operation))
    project = _target_project(arguments, spec.verb)
    known = managed_projects((evidence_facts or {}).get(spec.evidence_tool))
    if project not in known:
        raise ActingProposalError(
            "evidence_unbound: VD-051 #96 constraint 2 forbids acting on a fact "
            "this turn did not read. '{project}' is not among the managed apps "
            "{tool} reported this turn ({known}). Read {tool} first - an acting "
            "proposal binds to evidence the read tools established, never to a "
            "name the model supplied.".format(
                project=project,
                tool=spec.evidence_tool,
                known=sorted(known) or "none",
            )
        )
    return {
        "type": spec.job_type,
        "payload": {"project": project},
        "confirmation": "{verb} the managed app '{project}'. {note}".format(
            verb=spec.verb.capitalize(), project=project, note=spec.reversible_note
        ),
        "evidence": {"tool": spec.evidence_tool, "resource": project},
    }


def sanitize_proposed_operations(result: Any) -> list[Dict[str, str]]:
    """Bound the acting operations an installed-agent RESULT proposed (#34).

    An agent runtime that surfaces ``proposed_operations`` from a model result
    must pass them through here first. Kept to ``operation`` / ``project``
    strings only: any ``grants`` the model tried to append are dropped, because
    authority is read from the server-owned pinned snapshot at approval time
    (`resolve_agent_operation`), never from the result. Bounded to five so a
    runaway result cannot flood the approval surface.

    This lives here rather than in the runner because the runner is another
    agent's file; the runner need only call this and attach the returned list.
    """
    if not isinstance(result, Mapping):
        return []
    raw = result.get("proposed_operations", [])
    if not isinstance(raw, list):
        return []
    operations: list[Dict[str, str]] = []
    for item in raw[:5]:
        if not isinstance(item, Mapping):
            continue
        operation = str(item.get("operation", "")).strip()[:64]
        project = str(item.get("project", "")).strip()[:128]
        if operation and project:
            operations.append({"operation": operation, "project": project})
    return operations


def agent_grant_snapshot(task: Mapping[str, Any]) -> list[Mapping[str, Any]]:
    """The server-owned workload-acting grant snapshot on an installed agent's task.

    #34: this is read from ``approval_context`` - the snapshot the server pinned
    when the task was created - and never from the task *result*, which the agent
    (and thus the model) authors.

    LESSONS 6 / VD-106 (#34 / VD-100): exactly ONE field is read, ``workload_act_grants``.
    The two grant namespaces must never cross-authorize. ``app_grants`` (from
    `agent_task_app_capabilities._safe_app_grant_snapshot`) carries integration /
    app-capability grants whose ``operation_id`` is free-form and can be literally
    ``workloads.restart``. This function once also read ``app_grants``, on the
    now-corrected belief that integration grants "harmlessly fall outside the
    envelope intersection" - false: an app grant named ``workloads.restart``
    intersected straight into the workload acting envelope and minted a live
    ``compose.restart`` with no dedicated workload grant. App-capability authority
    is resolved only on the app path; workload acting authority is resolved only
    against this field, so the two hand-maintained lists (the LESSONS-6
    two-namespace class - #98 is its packaging-lists sibling) cannot drift into
    cross-authorization.
    """
    context = task.get("approval_context") if isinstance(task, Mapping) else None
    if not isinstance(context, Mapping):
        return []
    grants: list[Mapping[str, Any]] = []
    for grant in context.get("workload_act_grants") or []:
        if isinstance(grant, Mapping):
            grants.append(grant)
    return grants


def authorized_operations(grant_snapshot: Any) -> frozenset[str]:
    """The operations an installed agent may propose: envelope AND pinned grant.

    #34: the returned set is the intersection of the server-owned envelope
    constant and the operation ids the server pinned into the grant snapshot.
    An agent with no pinned workload-operation grant may propose nothing, which
    is the correct fail-closed default - `*:act` is never granted implicitly.
    """
    pinned: set[str] = set()
    for grant in grant_snapshot or []:
        if not isinstance(grant, Mapping):
            continue
        for operation in grant.get("operations", []) or []:
            if isinstance(operation, Mapping):
                identifier = str(operation.get("operation_id", "")).strip()
                if identifier:
                    pinned.add(identifier)
    return frozenset(INSTALLED_AGENT_OPERATIONS & pinned)


def resolve_agent_operation(
    entry: Mapping[str, Any],
    task: Mapping[str, Any],
    live_projects: frozenset[str],
) -> Dict[str, Any]:
    """Turn one agent-proposed operation into a job proposal, or refuse it.

    Three gates, in order of how much they matter to #34:

    1. The operation is within the server-owned envelope constant.
    2. The operation is pinned by the agent's server-owned grant snapshot -
       *this* is the pinned-grant check, extended from `plan_custom_app_calls`.
       A grant read from `entry` (the model's output) would defeat the rule, so
       the snapshot is read from `approval_context` only.
    3. The target app is one the server can see managed *right now* - resource
       scoping against live truth, not against a project name the model typed.
    """
    if not isinstance(entry, Mapping):
        raise ActingProposalError("The proposed operation is not an object.")
    operation = str(entry.get("operation", "")).strip()
    if operation not in INSTALLED_AGENT_OPERATIONS:
        raise ActingProposalError(
            "operation_outside_envelope: #34 / VD-051 #96 - an installed agent "
            "may propose only {envelope}. '{operation}' is not one of them, so "
            "no job is minted. The envelope is a server-owned constant and is "
            "never widened by anything an agent writes into its result.".format(
                envelope=sorted(INSTALLED_AGENT_OPERATIONS), operation=operation
            )
        )
    allowed = authorized_operations(agent_grant_snapshot(task))
    if operation not in allowed:
        raise ActingProposalError(
            "operation_outside_grant: #34 / VD-051 #96 - this agent's "
            "server-owned grant snapshot does not pin '{operation}' (it pins "
            "{allowed}). The envelope is checked against the pinned grant, not "
            "against the agent's own proposed_operations - deriving the "
            "envelope from the model is exactly the failure #34 names.".format(
                operation=operation, allowed=sorted(allowed) or "nothing"
            )
        )
    spec = _ACTING_OPERATIONS[operation]
    project = _target_project(entry, spec.verb)
    if project not in live_projects:
        raise ActingProposalError(
            "resource_unbound: '{project}' is not a managed app the server can "
            "see now ({known}). The target is scoped against live inventory, "
            "not against the name in the agent's result.".format(
                project=project, known=sorted(live_projects) or "none"
            )
        )
    return {
        "type": spec.job_type,
        "payload": {"project": project},
        "evidence": {"tool": spec.evidence_tool, "resource": project},
    }
