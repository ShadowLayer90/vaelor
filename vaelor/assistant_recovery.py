"""Grounded backup and recovery answers for the appliance assistant."""

from __future__ import annotations

from typing import Any, Dict, Optional

from .phrase_match import mentions


#: Matched as whole words (#182), with the inflections written out. A bare
#: word boundary alone would have lost "backups", "checkpoints" and
#: "restored", turning answerable questions into refusals - which is the
#: expensive half of this repair everywhere it was made.
RECOVERY_TERMS = (
    "backup", "backups", "checkpoint", "checkpoints", "restore", "restores",
    "restored", "recovery", "rollback", "rollbacks", "roll back",
)
CREATE_PHRASES = (
    "create a checkpoint",
    "create checkpoint",
    "make a checkpoint",
    "create a backup",
    "make a backup",
    "back up",
)


def recovery_answer(message: str, facts: Dict[str, Any]) -> Optional[Dict[str, Any]]:
    lower = str(message).lower()
    if not mentions(lower, RECOVERY_TERMS):
        return None
    workloads = facts.get("workloads.inventory", {})
    checkpoints = facts.get("recovery.checkpoints", [])
    apps = workloads.get("apps", []) if isinstance(workloads, dict) else []
    projects = sorted({
        str(item.get("project"))
        for item in apps
        if item.get("managed") and item.get("project")
    })
    create_requested = any(phrase in lower for phrase in CREATE_PHRASES)
    evidence = [{
        "source": "recovery.checkpoints",
        "summary": "Current managed configuration restore-point inventory.",
    }]

    if create_requested and len(projects) == 1:
        project = projects[0]
        return {
            "answer": (
                "I can create a configuration checkpoint for {}. It protects "
                "the managed Compose files and Vaelor settings for this app; "
                "it does not copy Docker volume contents or external model files. "
                "Nothing has run yet."
            ).format(project),
            "evidence": [
                {
                    "source": "workloads.inventory",
                    "summary": "One managed application project is available to checkpoint.",
                },
                *evidence,
            ],
            "suggested_actions": [
                "Review the target and approve Create checkpoint.",
                "Use the app manager to verify or restore it later.",
            ],
            "proposed_job": {
                "type": "compose.backup",
                "payload": {"project": project},
            },
        }

    if create_requested and len(projects) > 1:
        answer = (
            "A checkpoint must belong to one managed app. Choose which app to "
            "protect: {}."
        ).format(", ".join(projects[:8]))
        actions = [f"Create a checkpoint for {project}" for project in projects[:8]]
    elif create_requested:
        answer = "There is no managed application project available to checkpoint yet."
        actions = []
    else:
        count = len(checkpoints) if isinstance(checkpoints, list) else 0
        answer = (
            "A Vaelor checkpoint is a versioned configuration restore point for "
            "one managed app. It contains the app's managed Compose project files, "
            "not Docker volume data or external model files. {} checkpoint{} "
            "currently available. Restoring first creates a safety checkpoint, "
            "validates the archive, replaces the project configuration, and starts "
            "the app; if startup fails, Vaelor puts the prior configuration back."
        ).format(count, " is" if count == 1 else "s are")
        actions = ["Open the app manager to create, verify, restore, or delete a checkpoint."]
    return {
        "answer": answer,
        "evidence": evidence,
        "suggested_actions": actions,
        "proposed_job": None,
    }
