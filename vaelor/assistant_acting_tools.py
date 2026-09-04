"""The acting-tool registry: producers of job proposals, never a second executor.

`assistant_tools.AssistantToolRegistry` is the *read* registry - 25-odd tools
that inspect appliance state and change nothing. This registry is its acting
counterpart, kept deliberately separate so the read surface stays read-only by
construction (VD-051's line 1 concern) and the acting surface is small enough
to reason about whole.

The one architectural rule, from VD-051: **an acting tool never executes.** Its
handler returns a ``{"proposed_job": {...}}`` descriptor and nothing else - no
`JobStore`, no executor, no Docker is reachable from it. The proposal names an
existing allowlisted job type; the operator's authenticated, CSRF-checked click
on ``POST /api/v2/jobs`` is what mints the job, the out-of-process executor runs
it, and ``security.audit`` records it. There is one execution path and acting
tools are not on it.

Two gates, reused rather than reinvented:

* **Scope.** Every acting tool requires a ``*:act`` scope, enforced by the same
  ``granted_scopes`` check the read registry uses. No role and no agent carries
  ``workloads:act`` by default, so the whole surface is inert until an operator
  grants it.
* **Evidence.** A proposal binds to a fact the read tools produced this turn
  (`assistant_action_proposals.build_job_proposal`), so an Assistant cannot act
  on a fault it invented - VD-051 constraint 2.
"""

from __future__ import annotations

from typing import Any, Dict, Iterable, Optional

from .assistant_tools import (
    ARGUMENTS_MUST_BE_OBJECT,
    AssistantToolError,
    ToolDefinition,
    UNSUPPORTED_ARGUMENTS,
)
from .assistant_action_proposals import (
    ActingProposalError,
    WORKLOADS_ACT_SCOPE,
    build_job_proposal,
)


_PROJECT_PARAMETERS = {
    "type": "object",
    "properties": {
        "project": {"type": "string", "minLength": 1, "maxLength": 128},
    },
    "required": ["project"],
    "additionalProperties": False,
}


def _proposal_handler(operation: str):
    """A handler that returns a proposal and reaches nothing that executes.

    It takes the turn's evidence facts as a second argument (the read registry's
    handlers take only ``args``; acting handlers need the facts to bind the
    proposal), and it returns exactly ``{"proposed_job": ...}``. It closes over
    no store and no executor - that inertness is what
    `tests/test_assistant_acting_tools.py` proves with a `JobStore` spy.
    """

    def run(arguments: Dict[str, Any], evidence_facts: Dict[str, Any]) -> Dict[str, Any]:
        return {"proposed_job": build_job_proposal(operation, arguments, evidence_facts)}

    return run


def _build_acting_tools() -> tuple[ToolDefinition, ...]:
    return (
        ToolDefinition(
            "workloads.restart",
            "Propose restarting one managed app. Returns a proposal an operator "
            "approves on the jobs endpoint; it never restarts anything itself.",
            WORKLOADS_ACT_SCOPE,
            "reversible_action",
            5,
            _proposal_handler("workloads.restart"),
            _PROJECT_PARAMETERS,
            mutation=True,
            approval="operator",
        ),
        ToolDefinition(
            "workloads.redeploy",
            "Propose redeploying one managed app: pull replacement images, verify "
            "health, and roll back automatically on failure. Returns a proposal "
            "an operator approves; it never redeploys anything itself.",
            WORKLOADS_ACT_SCOPE,
            "reversible_action",
            5,
            _proposal_handler("workloads.redeploy"),
            _PROJECT_PARAMETERS,
            mutation=True,
            approval="operator",
        ),
    )


class AssistantActingToolRegistry:
    """Register and run acting tools, which produce proposals and never execute.

    ``callbacks`` mirrors the read registry's shape so a caller can wire the
    same appliance seams (a ``job_store`` among them). Nothing in this class
    reads ``job_store``; that a store can be present and stay untouched is the
    point the guard asserts.
    """

    def __init__(self, callbacks: Optional[Dict[str, Any]] = None):
        self.callbacks = callbacks or {}
        self._tools: Dict[str, ToolDefinition] = {
            tool.name: tool for tool in _build_acting_tools()
        }

    def catalog(self) -> list[Dict[str, Any]]:
        return [self._tools[name].public() for name in sorted(self._tools)]

    def names(self) -> frozenset[str]:
        return frozenset(self._tools)

    def run(
        self,
        name: str,
        arguments: Optional[Dict[str, Any]] = None,
        *,
        granted_scopes: Optional[Iterable[str]] = None,
        evidence_facts: Optional[Dict[str, Any]] = None,
        actor: Optional[str] = None,
    ) -> Dict[str, Any]:
        tool = self._tools.get(str(name))
        if tool is None:
            raise AssistantToolError("Unknown assistant acting tool.")
        # LESSONS 2 / VD-051 #96: the acting surface is inert until a scope
        # grants it. Deleting this gate must fail a test - `granted_scopes`
        # None means "no gating", so an acting tool run with a scope set that
        # omits `workloads:act` is refused here, exactly as the read registry
        # refuses a read tool whose scope is not granted.
        if granted_scopes is not None and tool.scope not in set(granted_scopes):
            raise AssistantToolError(
                "capability_denied: {name} requires {scope}. LESSONS 2 / "
                "VD-051 #96: an acting tool is inert until a *:act scope grants "
                "it, and no role or agent carries {scope} by default.".format(
                    name=tool.name, scope=tool.scope
                )
            )
        args = arguments or {}
        if not isinstance(args, dict):
            raise AssistantToolError(ARGUMENTS_MUST_BE_OBJECT)
        allowed = set(tool.parameters.get("properties", {}))
        if set(args) - allowed:
            raise AssistantToolError(UNSUPPORTED_ARGUMENTS)
        try:
            raw = tool.handler(args, evidence_facts or {})
        except ActingProposalError as error:
            raise AssistantToolError(str(error)) from error
        # An acting tool returns a proposal, never a mutation result. The shape
        # is asserted here so a handler that ever returned anything else - an
        # execution outcome, say - is caught at the seam rather than surfaced.
        if not isinstance(raw, dict) or set(raw) != {"proposed_job"}:
            raise AssistantToolError(
                "LESSONS 14 / VD-051 #96: an acting tool must return exactly "
                "{{'proposed_job': ...}} and never a mutation result - it is a "
                "producer of proposals, not a second execution path."
            )
        return {
            "tool": tool.name,
            "scope": tool.scope,
            "risk": tool.risk,
            "mutation": True,
            "approval_required": True,
            "result": raw,
        }
