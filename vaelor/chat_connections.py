"""Present every AI Chat connection, with load state and where it runs.

Two things the single-``active_connection`` contract could not express, and
both cost users something real:

**A local model is a connection.** ``model.deploy`` stands up an
OpenAI-compatible llama.cpp server on ``127.0.0.1`` and registers a credential
for it, so the appliance's own model already travels the same path as LM
Studio on the LAN or a hosted endpoint. The privacy distinction the interface
needs to draw is therefore a ``local`` flag on a list, not a second code path,
and grouping by it is presentation.

**Listing a model is not loading it.** A picker built from ``/v1/models``
shows everything a server *offers*. On LM Studio, choosing one that is not
resident returns ``HTTP 400`` after the user has committed to it, with the
picker sitting silently on the selection that caused it. Load state belongs in
the contract so the picker can say so first.

Where a server does not publish load state, this reports ``None`` — not
``False``. "This model is not loaded" and "this server does not say" are
different claims, and rendering the second as the first would trade one wrong
answer for another.
"""

from __future__ import annotations

from typing import Any, Callable, Dict, List, Mapping, Optional, Sequence

from .managed_local_credentials import PREFIX as MANAGED_LOCAL_PREFIX
from .provider_runtime import managed_local_connection


#: How a connection's ``local`` flag was decided, so a caller can tell a
#: measured answer from a structural one.
LOCAL_BY_PREFIX = "vaelor-managed"
LOCAL_BY_ADDRESS = "loopback-address"
NOT_LOCAL = "remote"


def connection_locality(connection: Mapping[str, Any]) -> Dict[str, Any]:
    """Whether this connection is served by this appliance, and how we know.

    Two independent signals, because either alone has a gap: a credential
    Vaelor generated for its own deploy carries the managed prefix, and any
    connection whose base URL is a loopback address is on this machine
    regardless of who created it. A hosted provider matches neither.
    """
    identifier = str(
        connection.get("id") or connection.get("credential_id") or ""
    )
    if identifier.startswith(MANAGED_LOCAL_PREFIX):
        return {
            "local": True,
            "local_source": LOCAL_BY_PREFIX,
            "local_reason": (
                "Vaelor deployed this model on this appliance, so prompts sent "
                "to it never leave the machine."
            ),
        }
    if connection.get("base_url") and managed_local_connection(dict(connection)):
        return {
            "local": True,
            "local_source": LOCAL_BY_ADDRESS,
            "local_reason": (
                "This endpoint is a loopback address on this appliance, so "
                "prompts sent to it never leave the machine."
            ),
        }
    if connection.get("provider") == "openai":
        return {
            "local": False,
            "local_source": NOT_LOCAL,
            "local_reason": "Prompts sent to this connection go to OpenAI.",
        }
    return {
        "local": False,
        "local_source": NOT_LOCAL,
        "local_reason": (
            "This endpoint is on another machine, so prompts sent to it leave "
            "this appliance."
        ),
    }


def model_load_state(
    models: Sequence[str],
    loaded: Optional[Sequence[str]],
    failures: Optional[Mapping[str, str]] = None,
) -> List[Dict[str, Any]]:
    """Per-model availability, separating "offered", "loaded" and "failed".

    ``loaded=None`` means the server published no load state. Every model is
    then reported with ``loaded: None`` and a reason, so the picker can warn
    that selecting one may trigger a load rather than pretending to know.
    """
    known_bad = dict(failures or {})
    resident = None if loaded is None else {str(name) for name in loaded}
    result: List[Dict[str, Any]] = []
    for name in models:
        model = str(name)
        if resident is None:
            state, reason = None, (
                "This server does not report which models are loaded, so "
                "choosing this one may start a load first."
            )
        elif model in resident:
            state, reason = True, ""
        else:
            state, reason = False, (
                "This server lists this model but is not holding it. Choosing "
                "it starts a load, which can take a while or fail outright."
            )
        result.append({
            "model": model,
            # Offered by the server. Kept distinct from `loaded` because that
            # distinction is the whole point of this contract.
            "listed": True,
            "loaded": state,
            "load_reason": reason,
            # A model this appliance has really failed to use, with why.
            # Measured from real requests, never probed.
            "available": model not in known_bad,
            "reason": known_bad.get(model, ""),
        })
    return result


def describe_connections(
    connections: Sequence[Mapping[str, Any]],
    *,
    loaded_for: Optional[Callable[[Mapping[str, Any]], Optional[Sequence[str]]]] = None,
    failures_for: Optional[Callable[[Mapping[str, Any]], Mapping[str, str]]] = None,
    models_for: Optional[Callable[[Mapping[str, Any]], Sequence[str]]] = None,
) -> List[Dict[str, Any]]:
    """Every connection the picker may offer, each carrying its own facts.

    The callables are injected so this stays a pure shaping function: a route
    supplies live lookups, a test supplies fixtures, and neither needs the
    other's machinery.
    """
    described: List[Dict[str, Any]] = []
    for connection in connections:
        entry = dict(connection)
        entry.update(connection_locality(entry))
        models = list(models_for(entry)) if models_for else list(
            entry.get("models") or []
        )
        loaded = loaded_for(entry) if loaded_for else None
        failures = failures_for(entry) if failures_for else {}
        entry["models"] = models
        entry["model_states"] = model_load_state(models, loaded, failures)
        entry["loaded_models"] = None if loaded is None else list(loaded)
        entry["load_state_known"] = loaded is not None
        # The selected model's own state, lifted so a picker does not have to
        # scan the list to find out whether its current choice is usable.
        selected = str(entry.get("selected_model") or entry.get("model") or "")
        chosen = next(
            (item for item in entry["model_states"] if item["model"] == selected),
            None,
        )
        entry["selected_model_loaded"] = chosen["loaded"] if chosen else None
        entry["selected_model_reason"] = chosen["load_reason"] if chosen else ""
        described.append(entry)
    return described


def local_tier_map(
    inference_plan: Optional[Mapping[str, Any]] = None,
) -> Dict[str, Any]:
    """Which locally-served model belongs to which engine.

    Lets a picker label a local model with the engine that will run it rather
    than presenting a flat list in which the NPU assistant model and the GPU
    chat model look interchangeable.
    """
    plan = dict(inference_plan or {})
    tiers = plan.get("tiers") or []
    by_model: Dict[str, Any] = {}
    for tier in tiers:
        model = str(tier.get("model") or "")
        if not model:
            continue
        by_model[model] = {
            "kind": tier.get("kind"),
            "role": tier.get("role"),
            "context_tokens": tier.get("context_tokens"),
        }
    return {
        "models": by_model,
        "tiers": [
            {
                "kind": tier.get("kind"),
                "role": tier.get("role"),
                "model": tier.get("model"),
                "context_tokens": tier.get("context_tokens"),
            }
            for tier in tiers
        ],
    }
