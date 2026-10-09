"""Resolve the model selected for appliance-assisted work."""

from __future__ import annotations

from typing import Any, Dict, Optional

from .credential_broker import CredentialError
from .hosted_providers import OFF_MACHINE_KINDS
from .inference_client import OPENAI_BASE_URL
from .model_credential_roles import ai_chat_may_use_lease
from .runtime_paths import env_value

OPENAI_DEFAULT_MODEL = "gpt-5.6-terra"

#: VD-207: why escalation to AI Chat's model is unavailable when that model is
#: off this machine. The owner decided such an escalation needs their approval
#: each time; that approval is not built yet, so escalation fails closed.
ESCALATION_NEEDS_APPROVAL = (
    "AI Chat's model is a hosted service off this machine, and sending "
    "research to it needs your approval each time, which Vaelor cannot ask "
    "for yet. The Assistant's own model's result stands."
)


def off_machine_lease(lease: Any) -> bool:
    """Whether a lease sends prompts off this machine (VD-206 / VD-207)."""
    return isinstance(lease, dict) and str(lease.get("provider") or "") in OFF_MACHINE_KINDS


def escalation_refusal(credential_broker: Any) -> str:
    """Why research cannot escalate to AI Chat's model right now, or ``""``.

    Only the off-machine case is named here; no lease at all is the existing
    "the graphics model is unavailable" answer, which callers already give.
    """
    if credential_broker is None:
        return ""
    try:
        lease = credential_broker.resolve_active("ai-chat")
    except Exception:  # noqa: BLE001 - absence-ok: unread is the callers' own unavailable answer
        return ""
    return ESCALATION_NEEDS_APPROVAL if off_machine_lease(lease) else ""

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
            lease = credential_broker.resolve_active("ai-chat")
            listed = credential_broker.list()
        except CredentialError:
            return None
        if not ai_chat_may_use_lease(lease, listed):
            # VD-210 owner rule: a stale AI Chat lease on the Assistant's NPU
            # model is not a capable GPU model, and AI Chat never uses it.
            return None
        if off_machine_lease(lease):
            # VD-207: escalating to an external model needs the owner's
            # approval each time, and that approval is not built yet - so it
            # fails closed for every off-machine kind, OpenAI included.
            return None
        conn = _lease_connection(lease)
        # Stamp the escalation marker server-side, from the broker lease, so
        # every tier consumer treats the capable pass as capable regardless of
        # the model NAME the lease carries. The GPU fork lease deliberately
        # stores an EMPTY model (its identity is in the label), which
        # `model_capability` would otherwise read as a tiny model and mis-tier
        # as `basic-local` - running the degraded compact research flow on the
        # LARGER model. This marker is set ONLY here and never from untrusted
        # input; it affects discovery richness and capability display, never a
        # safety gate. Return a COPY so the broker's lease dict is not mutated.
        return {**conn, "escalated_capable": True}
    if credential_broker is not None:
        try:
            lease = credential_broker.resolve_active("deployment-agent")
        except CredentialError:
            lease = None
        # VD-049 / VD-207: the Assistant's own lease never names an external
        # model (the vault refuses it); refused here too, so a stale row cannot.
        if lease is not None and not off_machine_lease(lease):
            return _lease_connection(lease)
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
