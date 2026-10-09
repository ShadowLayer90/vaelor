"""Match an AI Chat request to one of the caller's custom agents.

AI Chat never runs an agent. When a request names an enabled, account-owned
custom agent (or the reader picked one explicitly), the chat answers with a
bounded *proposal*: which agent, which version, what it may touch, and that
nothing has started. The run itself is queued and approved on the Assistant.

Moved out of `api_chat_routes` unchanged so that module stays under the line
ceiling; the route still owns when a proposal is offered and how it is stored.
"""

from __future__ import annotations

import re
from collections.abc import Mapping

from .custom_agent_routing import custom_agent_proposal


def custom_profiles(callbacks, actor):
    """Return only enabled, actor-owned assistant custom profiles.

    Chat proposals run on the assistant, so an inference (cluster) agent
    must never be matchable here - it would be named, proposed, and then
    dead-end at enqueue. Both the profiles seam and the store fallback are
    constrained to the assistant surface.
    """
    task_store = callbacks.get("agent_tasks")
    profiles = []
    try:
        profiles = (
            task_store.profiles(actor, surface="assistant")
            if task_store is not None else []
        )
    except (AttributeError, TypeError, ValueError):
        profiles = []
    if not profiles:
        store = callbacks.get("custom_agents")
        try:
            profiles = store.list(actor, include_disabled=True) if store is not None else []
        except (AttributeError, TypeError, ValueError):
            profiles = []
    return [
        profile for profile in profiles
        if isinstance(profile, Mapping)
        and bool(profile.get("custom"))
        and bool(profile.get("enabled", True))
        and str(profile.get("surface", "assistant")) == "assistant"
    ]


def named_profile_match(message, profiles):
    """Require a name token before the shared matcher can select a profile."""
    message_tokens = set(re.findall(r"[a-z0-9]+", str(message).lower()))
    stopwords = {"agent", "assistant", "custom", "my", "the", "a", "an"}
    for profile in profiles:
        name_tokens = {
            token for token in re.findall(r"[a-z0-9]+", str(profile.get("name", "")).lower())
            if token not in stopwords and len(token) > 1
        }
        if name_tokens.intersection(message_tokens):
            return True
    return False


def grant_summaries(callbacks, actor, profile):
    """Decorate a proposal with bounded, non-secret current app access."""
    grants = callbacks.get("agent_app_grants")
    registry = callbacks.get("app_capability_registry")
    if grants is None:
        return []
    try:
        rows = grants.list(actor, agent_id=str(profile.get("id", "")), limit=50)
    except (AttributeError, TypeError, ValueError):
        return []
    try:
        selected_version = int(profile.get("version", 0))
    except (TypeError, ValueError):
        selected_version = 0
    summaries = []
    for grant in rows:
        if not isinstance(grant, Mapping) or bool(grant.get("revoked")):
            continue
        try:
            if int(grant.get("agent_version", -1)) != selected_version:
                continue
        except (TypeError, ValueError):
            continue
        app = None
        manifest = None
        try:
            if registry is not None:
                app = registry.get_app_instance(str(grant.get("app_instance_id", "")))
                manifest = registry.get_manifest(str(grant.get("manifest_digest", "")))
        except (AttributeError, TypeError, ValueError):
            app = None
            manifest = None
        operations_by_id = {}
        if manifest is not None:
            operations_by_id = {
                str(operation.operation_id): operation
                for operation in getattr(manifest, "operations", ())
            }
        operations = []
        for operation_id in list(grant.get("operation_ids", []))[:20]:
            operation = operations_by_id.get(str(operation_id))
            mode = str(getattr(operation, "mode", "read"))
            operations.append({
                "id": str(operation_id)[:80],
                "name": str(getattr(operation, "label", operation_id))[:100],
                "access": mode if mode in {"read", "write"} else "read",
                "risk": str(getattr(operation, "risk", "low"))[:40],
            })
        summaries.append({
            "grant_id": str(grant.get("id", ""))[:96],
            "app_instance_id": str(grant.get("app_instance_id", ""))[:160],
            "app_name": str((app or {}).get("app_label", "Installed app"))[:100],
            "manifest_version": str(getattr(manifest, "manifest_version", ""))[:40],
            "manifest_digest": str(grant.get("manifest_digest", ""))[:64],
            "operations": operations,
        })
    return summaries


def agent_proposal(callbacks, actor, message, requested_profile_id=""):
    profiles = custom_profiles(callbacks, actor)
    requested_profile_id = str(requested_profile_id or "").strip()
    if requested_profile_id:
        selected_profiles = [
            profile for profile in profiles
            if str(profile.get("id", "")) == requested_profile_id
        ]
        if not selected_profiles:
            return None
        profiles = selected_profiles
    elif not profiles or not named_profile_match(message, profiles):
        return None
    proposal = custom_agent_proposal(message, profiles, explicit=bool(requested_profile_id))
    if proposal is None:
        return None
    selected = next(
        profile for profile in profiles
        if str(profile.get("id")) == proposal["profile_id"]
    )
    proposal["app_grants"] = grant_summaries(callbacks, actor, selected)
    proposal["approval_required"] = True
    return proposal


def proposal_text(proposal):
    capabilities = ", ".join(proposal.get("capabilities", [])) or "none listed"
    integrations = ", ".join(proposal.get("integrations", [])) or "none"
    app_access = []
    for grant in proposal.get("app_grants", []):
        operations = ", ".join(
            "{} ({})".format(item.get("name", item.get("id", "")), item.get("access", "read"))
            for item in grant.get("operations", [])
        ) or "no operations listed"
        app_access.append("{}: {}".format(grant.get("app_name", "Installed app"), operations))
    # The internal profile id belongs in the structured proposal the
    # client already receives, not in prose. Reading a raw
    # "custom_c4f7dc80..." GUID back to the user names nothing they can
    # act on and looks like a leak.
    return (
        "I matched this request to the enabled custom agent '{}' (version {}). "
        "No work has started. The bounded task is: {}. Capabilities: {}. "
        "App access: {}. Integrations: {}. Approval required: yes. Review this exact "
        "run, then approve it under Assistant > Agents, in this agent's run history."
    ).format(
        proposal["profile_name"], proposal["profile_version"],
        proposal["task"], capabilities, "; ".join(app_access) or "none granted",
        integrations,
    )
