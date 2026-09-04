"""Resolve the model selected for appliance-assisted work."""

from __future__ import annotations

from typing import Any, Dict, Optional

from .credential_broker import CredentialError
from .inference_client import OPENAI_BASE_URL
from .runtime_paths import env_value

OPENAI_DEFAULT_MODEL = "gpt-5.6-terra"

#: The user-triggered escalation tiers. Default work runs on the NPU
#: ``deployment-agent`` model; these modes re-run the SAME task on the more
#: capable GPU model that powers AI Chat (the ``ai-chat`` lease). Escalation is
#: never automatic - a run only reaches here when the operator asked for it.
CAPABLE_MODES = frozenset({"capable", "gpu"})


def _lease_connection(lease: Dict[str, str]) -> Dict[str, str]:
    """Shape a broker lease into an inference connection (openai vs local).

    The same normalization the deployment-agent path has always used, so the
    capable GPU lease and the default NPU lease are shaped identically.
    """
    if lease.get("provider") == "openai":
        return {
            **lease,
            "base_url": OPENAI_BASE_URL,
            "model": lease.get("model") or OPENAI_DEFAULT_MODEL,
            "api_key": lease.get("token", ""),
        }
    return lease


def assistant_model_configured(credential_broker: Any) -> bool:
    """True when a working Assistant model connection is configured.

    The signal is an active ``deployment-agent`` lease - the same connection
    the restart-on-boot reconcile keys on, and true for a local NPU/llama.cpp
    model or a hosted provider alike. This gates *availability*, not liveness:
    an installed-but-momentarily-down model still reports configured, and the
    action itself surfaces the error if the model is truly unreachable.

    Fails closed: this check sits on the Workloads page's load path, so any
    broker trouble - no active lease, a refused/half-closed socket, an empty
    reply from a briefly write-locked vault (surfacing client-side as a
    JSONDecodeError) - must read as "no model, workflow off", never escape as a
    500. The env read this replaced could not fail; this must not either
    (VD-109 review).
    """
    if credential_broker is None:
        return False
    try:
        credential_broker.resolve_active("deployment-agent")
    except Exception:
        return False
    return True


def resolve_model_connection(
    credential_broker: Any = None,
    *,
    mode: str = "",
    base_url: Optional[str] = None,
    model: Optional[str] = None,
    api_key: Optional[str] = None,
) -> Optional[Dict[str, str]]:
    """Return the active deployment model without exposing broker internals."""
    normalized_mode = str(mode).strip().lower()
    if normalized_mode == "basic":
        return None
    if normalized_mode in CAPABLE_MODES:
        # Escalation: run this on the capable GPU model that powers AI Chat.
        # It is a DIFFERENT lease ("ai-chat"), never the NPU deployment-agent,
        # and it never falls through to the environment NPU configuration: if no
        # GPU lease is active the graphics model is genuinely unavailable and the
        # caller must see None so it can tell the user, not silently downshift.
        if credential_broker is None:
            return None
        try:
            return _lease_connection(credential_broker.resolve_active("ai-chat"))
        except CredentialError:
            return None
    if credential_broker is not None:
        try:
            return _lease_connection(credential_broker.resolve_active("deployment-agent"))
        except CredentialError:
            pass
    endpoint = (
        base_url
        if base_url is not None
        else env_value("VAELOR_AGENT_BASE_URL", "PM_AGENT_BASE_URL", "")
    ).rstrip("/")
    if not endpoint:
        return None
    return {
        "base_url": endpoint,
        "model": model or env_value(
            "VAELOR_AGENT_MODEL", "PM_AGENT_MODEL", "local-model"
        ),
        "api_key": api_key if api_key is not None else env_value(
            "VAELOR_AGENT_API_KEY", "PM_AGENT_API_KEY", ""
        ),
        "label": "Environment configuration",
        "provider": "openai-compatible",
    }
