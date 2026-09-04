"""Consult the acting registry from the Assistant answer loop (VD-100 #96).

The acting seam - `assistant_acting_tools.AssistantActingToolRegistry` and
`assistant_action_proposals.build_job_proposal` - was built and committed, but
nothing in a running answer instantiated it, so the Assistant could never carry
an acting proposal. This module is that consultation, kept out of
`deployment_agent.py` so that module stays under its 1,000-line ceiling.

It is a **producer of a proposal and reaches nothing that executes** (VD-100,
LESSONS 14). Given this turn's evidence facts and the operator's granted scopes,
it asks the acting registry for a ``proposed_job`` and returns it, or returns
``None``. The operator's existing, CSRF-checked ``POST /api/v2/jobs`` is still
the only approval; there is one executor and this module is not on it.

Two gates, both reused from the registry rather than reinvented here:

* **Default-inert scope.** No proposal is produced unless the caller passes a
  granted-scope set that contains ``workloads:act``. ``granted_scopes=None`` is
  treated as "no scopes" - the fail-closed reading - so any caller that does not
  resolve an operator grant (the external-agent path, a bare unit test) gets an
  inert answer, never an accidental proposal.
* **Evidence binding.** The target app must appear among the managed apps
  ``workloads.inventory`` reported *this turn*. The project name is taken from
  that live reading, never from free text the model emitted, so the Assistant
  cannot propose against an app it invented (VD-100 constraint 2).
"""

from __future__ import annotations

import re
from typing import Any, Dict, Iterable, Mapping, Optional

from .assistant_acting_tools import AssistantActingToolRegistry
from .assistant_action_proposals import (
    WORKLOADS_ACT_SCOPE,
    managed_projects,
)
from .assistant_tools import AssistantToolError

#: The read fact an acting proposal binds to. It must be gathered this turn for
#: the evidence gate to pass, so the route adds it to the selected read tools
#: whenever `mentions_workload_action` is true.
EVIDENCE_TOOL = "workloads.inventory"

# restart-family and redeploy-family verbs, mapped to the two acting operations
# the phase-1 envelope allows (VD-100: restart before redeploy, nothing else).
_OPERATION_VERBS = (
    (
        re.compile(r"\b(?:restart|reboot|reload|bounce|recycle|cycle|relaunch)\b", re.IGNORECASE),
        "workloads.restart",
    ),
    (
        re.compile(r"\b(?:redeploy|re-?deploy|roll\s*out|pull\s+(?:and\s+)?restart)\b", re.IGNORECASE),
        "workloads.redeploy",
    ),
)

# Vocabulary that makes a clause a question about an app rather than an
# instruction to act on it: "why did nextcloud restart", "did grafana restart".
# Presence *before* the verb disqualifies the clause, exactly as
# `assistant_action_requests` disqualifies a lighting question. Polite request
# forms ("could you", "would you") are deliberately NOT here - they open an
# instruction, not a question, so they must still reach a proposal.
_QUESTION_BEFORE = re.compile(
    r"\b(?:why|what|which|when|where|who|how|whether|"
    r"is|are|was|were|does|did|has|have|had|"
    r"tell\s+me|show\s+me)\b",
    re.IGNORECASE,
)


def _matched_operation(message: str) -> tuple[Optional[str], int]:
    """The acting operation this message asks for, and where its verb starts.

    Redeploy is checked before restart because "pull and restart" is a redeploy,
    not a restart: the more specific family wins so a redeploy request is never
    downgraded to a bare restart proposal.
    """
    best: tuple[Optional[str], int] = (None, -1)
    for pattern, operation in _OPERATION_VERBS:
        match = pattern.search(message)
        if match is not None and (best[0] is None or operation == "workloads.redeploy"):
            best = (operation, match.start())
    return best


def _target_project(message: str, known: Iterable[str]) -> Optional[str]:
    """The managed app named in ``message``, taken from live evidence only.

    Iterating the apps ``workloads.inventory`` reported (longest first, so
    "nextcloud-db" wins over "nextcloud") means the project can only ever be one
    the read tools established this turn - the evidence binding, enforced before
    the registry is even consulted.
    """
    lowered = message.lower()
    for project in sorted(known, key=len, reverse=True):
        if re.search(r"\b" + re.escape(project.lower()) + r"\b", lowered):
            return project
    return None


def mentions_workload_action(message: str) -> bool:
    """Cheap gate the route uses to read `workloads.inventory` this turn.

    Deliberately loose - it only decides whether to *gather the evidence*, never
    whether to propose. The real decision (scope, verb, evidence-bound project)
    is made in `acting_proposal`.
    """
    return _matched_operation(str(message or ""))[0] is not None


def operator_act_scopes(store: Any, actor: str) -> frozenset[str]:
    """The acting scopes an operator holds now - empty when no store or no grant.

    Kept here so the answer route resolves the operator's grant in one line and
    stays under its size ceiling. Default-inert: a missing store or an operator
    with no administrator grant both yield the empty set (VD-100 #96).
    """
    if store is None:
        return frozenset()
    return store.scopes_for(actor)


def acting_answer(proposal: Mapping[str, Any]) -> Dict[str, Any]:
    """Wrap a minted proposal in the answer the Assistant returns.

    Kept here rather than in `deployment_agent` so that module stays under its
    line ceiling. The proposal travels in ``proposed_job``; the prose only
    states what will be reviewed, and that nothing has run.
    """
    evidence = proposal.get("evidence", {}) if isinstance(proposal, Mapping) else {}
    resource = evidence.get("resource", "") if isinstance(evidence, Mapping) else ""
    return {
        "answer": "{} Review and approve it before anything runs; nothing has "
        "changed yet.".format(proposal.get("confirmation", "")).strip(),
        "evidence": [{
            "source": evidence.get("tool", EVIDENCE_TOOL) if isinstance(evidence, Mapping) else EVIDENCE_TOOL,
            "summary": "Bound to the managed app '{}' this turn's inventory reported; "
            "the proposal cannot target an app it did not read.".format(resource),
        }],
        "suggested_actions": [
            "Approve the proposed job to apply the change; it does not run until you do.",
        ],
        "proposed_job": dict(proposal),
    }


def acting_proposal(
    message: str,
    evidence_facts: Optional[Mapping[str, Any]],
    granted_scopes: Optional[Iterable[str]],
    *,
    registry: Optional[AssistantActingToolRegistry] = None,
) -> Optional[Dict[str, Any]]:
    """Return a job proposal for an acting request, or ``None``.

    ``None`` for every inert case: no acting verb, the scope is not granted (the
    default), no managed app named in the message, or the registry refused the
    proposal (evidence unbound). A returned value is the inert ``proposed_job``
    descriptor the registry minted - a request the operator approves, never an
    action this function took.
    """
    text = str(message or "")
    # Default-inert: absent an explicit workloads:act grant, nothing is proposed.
    if WORKLOADS_ACT_SCOPE not in set(granted_scopes or ()):  # noqa: SIM  (fail-closed)
        return None
    operation, verb_start = _matched_operation(text)
    if operation is None:
        return None
    # A question that happens to contain the verb ("why did grafana restart")
    # is not an instruction to act. Disqualify it exactly as the owner-performed
    # action detector disqualifies a lighting question.
    if _QUESTION_BEFORE.search(text[:verb_start]):
        return None
    facts = evidence_facts or {}
    known = managed_projects(facts.get(EVIDENCE_TOOL))
    project = _target_project(text, known)
    if project is None:
        return None
    registry = registry or AssistantActingToolRegistry()
    try:
        outcome = registry.run(
            operation,
            {"project": project},
            granted_scopes=granted_scopes,
            evidence_facts=dict(facts),
        )
    except AssistantToolError:
        # Scope denied or evidence unbound: the registry refused, so the
        # Assistant carries nothing. Inert, not an error the user must see.
        return None
    proposal = outcome.get("result", {}).get("proposed_job")
    return proposal if isinstance(proposal, dict) else None
