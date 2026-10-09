"""Adaptive, non-blocking intelligence guidance for application research."""

from __future__ import annotations

import re
from typing import Any, Dict, Iterable, Mapping

from .provider_runtime import assistant_budget, model_capability, request_complexity


#: Discovery-degraded flags the research pipeline records when a fallback path
#: actually fired (a flaky model plan, a failed source selection, a search-query
#: fallback, or unreachable guarded web research). Escalation to the capable GPU
#: model is only worthwhile when one of these is set: it means DISCOVERY - not
#: extraction - underperformed, so a stronger model may find an authoritative
#: source the assistant missed.
_DISCOVERY_DEGRADED_FLAGS = (
    "discovery_fallback",
    "source_selection_fallback",
    "query_fallback_used",
    "web_research_unavailable",
)

#: The two discovery-empty risk names ``manifest_research_risks`` reports and
#: ``escalation_recommended`` reads back. Named once so the two functions cannot
#: drift out of agreement on the exact spelling.
_RISK_UNVERIFIED_COMPATIBILITY = "unverified compatibility"
_RISK_NO_DIGEST_IMAGE = "no digest-pinned image"


_REVIEWED_SIMPLE = {
    "grafana", "nginx", "plex", "uptime kuma", "home assistant", "portainer",
}
_COMPLEX_TERMS = {
    "cluster", "distributed", "custom", "database", "migration", "gpu", "cuda",
    "reverse proxy", "oauth", "sso", "high availability", "multi-node", "kubernetes",
    "private registry", "build from source", "native package", "host network",
}
_GENERIC_NAMES = {"app", "application", "server", "service", "container", "unknown application"}


def application_research_capability(
    request: str,
    connection: Dict[str, str] | None = None,
    *,
    evidence_count: int = 0,
    risk_flags: Iterable[str] = (),
) -> Dict[str, Any]:
    """Describe how far the selected intelligence can assist without blocking.

    Deterministic source, architecture, digest, and Compose policy remains the
    authority. A stronger model recommendation changes explanation quality,
    never the safety gate or whether bounded research may start.
    """
    query = " ".join(str(request or "").strip().split())[:500]
    lower = query.lower()
    capability = model_capability(connection)
    risks = sorted({str(item).strip().lower()[:80] for item in risk_flags if str(item).strip()})
    reviewed_simple = any(name in lower for name in _REVIEWED_SIMPLE)
    complex_matches = sorted(term for term in _COMPLEX_TERMS if term in lower)
    generic = lower in _GENERIC_NAMES or len(re.findall(r"[a-z0-9]+", lower)) < 2
    score = len(complex_matches) + min(3, len(risks))
    if request_complexity(query) == "complex" and not reviewed_simple:
        score += 1
    complexity = "complex" if score >= 3 else "moderate" if score else "simple"

    questions = []
    if generic:
        questions.append("What is the application's exact name or official project URL?")
    if any(term in lower for term in ("custom", "build from source", "private registry")):
        questions.append("Which official repository or registry should Vaelor treat as authoritative?")
    if any(term in lower for term in ("oauth", "sso", "database", "reverse proxy")):
        questions.append("Should Vaelor connect to an existing identity, database, or proxy service?")
    if any(term in lower for term in ("cluster", "multi-node", "distributed", "high availability")):
        questions.append("How many nodes and replicas should this application use?")

    selected_tier = capability["tier"]
    limited_engine = selected_tier in {"rules", "basic-local"}
    stronger = bool(limited_engine and (complexity == "complex" or len(risks) >= 2))
    budget = (
        assistant_budget(connection, query)
        if connection else
        {"context_chars": 4000, "max_tokens": 256}
    )
    if reviewed_simple and selected_tier == "basic-local":
        summary = "The selected lightweight local model is suitable for this well-known app; Vaelor still verifies every source, digest, architecture, and setting."
    elif stronger:
        summary = "Bounded research can continue, but a stronger installed or connected model may explain this multi-step or higher-risk deployment more reliably."
    else:
        summary = "The selected intelligence can assist this research while Vaelor's deterministic checks remain authoritative."
    limitations = list(capability.get("limitations", []))
    if not evidence_count:
        limitations.append("No source evidence has been captured yet; compatibility cannot be claimed.")
    if risks:
        limitations.append("Higher-risk details require explicit evidence: {}.".format(", ".join(risks[:4])))
    return {
        "schema_version": 1,
        "complexity": complexity,
        "can_proceed": True,
        "selected_intelligence": {
            "tier": selected_tier,
            "label": capability["label"],
            "description": capability["description"],
        },
        "summary": summary,
        "limitations": list(dict.fromkeys(limitations))[:8],
        "follow_up_required": bool(questions),
        "follow_up_questions": questions[:3],
        "stronger_model_recommended": stronger,
        "recommendation": (
            "Select a capable local or hosted model for richer synthesis; bounded research and all safety checks remain available now."
            if stronger else None
        ),
        "context_budget": budget,
        "evidence_count": max(0, int(evidence_count)),
        "risk_flags": risks,
        "policy_authority": "deterministic-vaelor-policy",
    }


def manifest_research_risks(manifest: Any) -> list[str]:
    if not isinstance(manifest, dict):
        return []
    compatibility = manifest.get("compatibility", {})
    risks = []
    if isinstance(compatibility, dict) and compatibility.get("status") != "verified":
        risks.append(_RISK_UNVERIFIED_COMPATIBILITY)
    if not manifest.get("images"):
        risks.append(_RISK_NO_DIGEST_IMAGE)
    variables = manifest.get("variables", [])
    if isinstance(variables, list) and any(isinstance(item, dict) and item.get("secret") for item in variables):
        risks.append("credential integration")
    if len(manifest.get("images", [])) > 1:
        risks.append("multiple services")
    return risks


def escalation_recommended(manifest: Any, discovery_flags: Any) -> bool:
    """True only when the assistant's DISCOVERY was weak enough to escalate.

    For an unreviewed application, ports and volumes are supplied by the operator
    at configure time, never by the model - so a "verified image, 0 ports, 0
    volumes" manifest is the correct shape, and this NEVER keys on those counts.
    The capable GPU model's real leverage is discovery: finding an authoritative
    source or verifiable image the assistant missed. So escalation is recommended
    only when both are true:

    * discovery came up empty - ``manifest_research_risks`` reports both
      "unverified compatibility" (``compatibility.status`` is not ``verified``)
      and "no digest-pinned image" (no image was proven); and
    * a discovery-degraded flag actually fired (a flaky model plan, a failed
      source selection, a query fallback, or unreachable guarded web research),
      so the weakness is a discovery gap a stronger model can plausibly close -
      not an app whose facts are simply absent from the public evidence.

    Pure and side-effect free: the caller decides availability (a capable lease)
    and fire-once (the job's own ``mode``) separately.
    """
    risks = manifest_research_risks(manifest)
    if not (
        _RISK_UNVERIFIED_COMPATIBILITY in risks and _RISK_NO_DIGEST_IMAGE in risks
    ):
        return False
    flags = discovery_flags if isinstance(discovery_flags, Mapping) else {}
    return any(bool(flags.get(flag)) for flag in _DISCOVERY_DEGRADED_FLAGS)
