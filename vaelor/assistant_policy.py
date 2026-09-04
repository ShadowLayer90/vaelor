"""Deterministic capability policy for appliance-assistant answers.

**#181 - the architecture was a constant.** This module told every owner that
*"Other **ARM64** apps, including Plex, are not pre-approved"*, on Raspberry
Pis and on x86-64 workstations alike, and offered to review "another ARM64
Compose workload" on both. `system.identity` carries `architecture`, gathered
on every request, and nothing here read it: a hardware fact this appliance
never took, used to justify refusing an app. That is the same shape as the
model inventing *"Plex requires GPU acceleration and inference support"* to
justify the same refusal, one layer down and in code rather than in prose.

`cluster_plan_services` had already made this correction for the deploy step,
which used to say "reviewed ARM64-capable image" on every cluster in
existence; `architecture_label` is the same function, and an unread
architecture is named as nothing rather than as the likelier guess.
"""

from __future__ import annotations

import re
from typing import Any, Dict, Optional

from .app_catalog import public_catalog
from .application_deployments import classify_application_intent
from .cluster_architecture import architecture_class, architecture_label


def is_deployment_request(message: str) -> bool:
    """Match an explicit change request without substring collisions."""
    if classify_application_intent(message) is not None:
        return True
    words = set(re.findall(r"[a-z0-9]+", message.lower()))
    asks_for_change = bool(
        words.intersection({
            "deploy", "deployment", "host", "hosting", "install", "installation",
            "launch", "run", "create", "spin", "setup",
        })
    ) or "set up" in message.lower()
    workload_terms = {
        "agent", "app", "assistant", "compose", "container", "docker", "grafana",
        "game", "llm", "model", "plex", "portainer", "server", "service", "webui",
    }
    return asks_for_change and bool(words.intersection(workload_terms))


def deployment_capability_answer(
    message: str, facts: Dict[str, Any]
) -> Optional[Dict[str, Any]]:
    """Answer catalog questions without asking a small model to guess."""
    lower = " ".join(str(message).lower().split())
    capability_question = (
        bool(re.search(r"\bwhat (?:apps? )?can (?:you|vaelor|pironman) (?:deploy|install|run)\b", lower))
        or bool(re.search(r"\bwhich apps? (?:can|do) (?:you|vaelor|pironman) (?:deploy|install|run)\b", lower))
        or "supported app" in lower
        or "reviewed app" in lower
        or "deployment catalog" in lower
    )
    if not capability_question:
        return None

    names = [item["name"] for item in public_catalog()]
    capabilities = facts.get("workloads.capabilities", {})
    docker = capabilities.get("docker", {}) if isinstance(capabilities, dict) else {}
    engine = (
        "Docker is installed and ready."
        if docker.get("installed")
        else "Docker must pass the one-click readiness check before an app can run."
    )
    identity = facts.get("system.identity")
    discovered = (identity if isinstance(identity, dict) else {}).get("architecture")
    # `architecture_label` answers "an unknown architecture" for a failed
    # probe, which is the right answer for a cluster refusal and the wrong one
    # in a sentence about what can be deployed: it would read as a claim that
    # this machine's silicon is strange. Say nothing about the silicon here
    # instead, which is what an unread reading supports.
    silicon = (
        architecture_label(discovered) if architecture_class(discovered) else ""
    )
    return {
        "answer": (
            "Vaelor can deploy these reviewed one-click apps: {}. {} Other {}"
            "apps, including Plex, are "
            "not pre-approved; Vaelor can validate a supplied Compose stack, but "
            "it will not claim compatibility until image architecture, ports, "
            "storage, memory, and health checks pass."
        ).format(", ".join(names), engine, "{} ".format(silicon) if silicon else ""),
        "evidence": [
            {
                "source": "apps.catalog",
                "summary": "{} reviewed one-click templates are currently installed in the catalog.".format(
                    len(names)
                ),
            },
            {
                "source": "workloads.capabilities",
                "summary": "Live Docker readiness and guarded Compose validation determine what else can run.",
            },
        ],
        "suggested_actions": [
            "Open Workloads > Install to choose a reviewed one-click app.",
            "Use Import a Docker stack to review another {}Compose workload "
            "before approval.".format("{} ".format(silicon) if silicon else ""),
        ],
        "proposed_job": None,
    }
