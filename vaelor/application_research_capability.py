"""Adaptive, non-blocking intelligence guidance for application research."""

from __future__ import annotations

import re
from typing import Any, Dict, Iterable

from .provider_runtime import assistant_budget, model_capability, request_complexity


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
        risks.append("unverified compatibility")
    if not manifest.get("images"):
        risks.append("no digest-pinned image")
    variables = manifest.get("variables", [])
    if isinstance(variables, list) and any(isinstance(item, dict) and item.get("secret") for item in variables):
        risks.append("credential integration")
    if len(manifest.get("images", [])) > 1:
        risks.append("multiple services")
    return risks
