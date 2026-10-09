"""Two routes a deployed cluster agent's lifecycle needed and had nowhere to live.

``POST /api/v2/agents/wake`` - session-less, called BY a deployed agent's
runtime when its model does not answer (owner decision 2026-09-28: a model
scaled to zero after sitting idle counts as available and wakes on demand).
It carries the agent's own per-deployment bearer - the memory token, checked
exactly as the memory routes check it, in constant time against the stored
hash - so only a running agent can ask, and only a request that already passed
that agent's keyed gate reaches the runtime that asks. The agent is identified
by the TOKEN; the body chooses nothing. What it may wake is the agent's
RECORDED backing deployment (``agent_backing.wake_backing``): an idle unload
enqueues the same warm load the inference gateway's wake enqueues; a manual
unload, a failed or a missing deployment enqueue nothing and are named.

``POST /api/v2/cluster/agents/skills-preview`` - the deploy review screen's
account of what the chosen skills will do (ACC-078, ACC-138): the read scopes
each grants, and exactly what guidance is sent - whole, shortened, or not sent
and why. It is the deploy's own derivation (``agent_skill_surface``), so the
screen and the deploy cannot disagree.
"""

from __future__ import annotations

import threading
import time
from typing import Any, Dict, Optional, Tuple

from flask import request

from .agent_backing import wake_backing
from .agent_skill_surface import (
    MAX_SKILL_BLOCK_CHARS, MAX_SKILL_INSTRUCTIONS_CHARS, MAX_SKILLS_IN_INSTRUCTIONS,
    skill_surface,
)
from .api_common import ApiContext, payload
from .gpu_idle_watch import enqueue_cluster_load

#: A wake answer is reused for this long per agent, so a burst of failing
#: requests from one agent asks the control plane once.
WAKE_ANSWER_REUSE_SECONDS = 2.0

_WAKE_LOCK = threading.Lock()
_LAST_WAKE: Dict[str, Tuple[float, Dict[str, Any]]] = {}

_WAKE_KEY_MISSING = "Present this agent's own bearer to ask for a wake."


def agent_store_from(callbacks: Dict[str, Any]) -> Any:
    """The agent deployment store: the wired one, else the cluster's own."""
    store = callbacks.get("agent_deployments")
    if store is not None:
        return store
    operations = callbacks.get("cluster_operations")
    reader = getattr(operations, "_agent_deployment_store", None)
    return reader() if callable(reader) else None


def _presented_agent(callbacks: Dict[str, Any]) -> Optional[Tuple[str, str]]:
    """The ``(name, id)`` the presented bearer binds to, or ``None`` (a uniform 401)."""
    header = request.headers.get("Authorization", "")
    if not header.startswith("Bearer ") or not header[7:]:
        return None
    store = agent_store_from(callbacks)
    if store is None:
        return None
    try:
        return store.agent_id_for_memory_key(header[7:])
    except (AttributeError, OSError, TypeError, ValueError):
        return None


def register_cluster_agent_routes(context: ApiContext) -> None:
    blueprint = context.blueprint
    callbacks = context.callbacks
    require_auth = context.require_auth

    @blueprint.post("/agents/wake")
    def agents_wake():
        identity = _presented_agent(callbacks)
        if identity is None:
            return payload(
                error={"code": "authentication_required", "message": _WAKE_KEY_MISSING},
                status=401,
            )
        name, agent_id = identity
        now = time.monotonic()
        with _WAKE_LOCK:
            cached = _LAST_WAKE.get(agent_id)
            if cached is not None and now - cached[0] < WAKE_ANSWER_REUSE_SECONDS:
                return payload(cached[1])
        store = agent_store_from(callbacks)
        operations = callbacks.get("cluster_operations")
        try:
            row = store.get(name) or {}
            backing = str((row.get("backing") or {}).get("model_deployment_name") or "")
            answer = wake_backing(
                cluster_store=operations.store,
                broker=callbacks.get("credential_broker"),
                name=backing,
                enqueue_load=lambda deployment: enqueue_cluster_load(
                    callbacks.get("job_store"), deployment
                ),
                mode_store=callbacks.get("cluster_mode_store"),
            )
        except Exception:  # noqa: BLE001 - never a 500 back to an agent: unavailable
            return payload(
                error={
                    "code": "wake_unavailable",
                    "message": "Vaelor could not read this agent's model deployment.",
                },
                status=503,
            )
        with _WAKE_LOCK:
            _LAST_WAKE[agent_id] = (now, answer)
        return payload(answer)

    @blueprint.post("/cluster/agents/skills-preview")
    @require_auth("operator", csrf=True)
    def cluster_agent_skills_preview():
        body = request.get_json(silent=True) or {}
        raw = body.get("skills") if isinstance(body, dict) else None
        skills = [
            item if isinstance(item, dict) else {"skill_id": str(item)}
            for item in (raw if isinstance(raw, list) else [])
        ]
        operations = callbacks.get("cluster_operations")
        reader = getattr(operations, "_agent_skills_library", None)
        library = reader() if callable(reader) else None
        surface = skill_surface(library, skills, strict=False)
        return payload({
            "scopes": surface["scopes"],
            "guidance": surface["guidance"],
            "refusals": surface["refusals"],
            "limits": {
                "skills": MAX_SKILLS_IN_INSTRUCTIONS,
                "instructions_per_skill": MAX_SKILL_INSTRUCTIONS_CHARS,
                "block": MAX_SKILL_BLOCK_CHARS,
            },
        })
