"""Fast, policy-first routing for deployment plans."""

from __future__ import annotations

from typing import Any, Dict, Optional

from .assistant_policy import is_deployment_request
from .deployment_plans import fallback_plan


def policy_plan(message: str) -> Optional[Dict[str, Any]]:
    """Return a deterministic plan when the request implies a real change.

    A connected model may explain ambiguous questions, but it must not delay or
    redefine an allowlisted installation or an unreviewed-app compatibility
    review.  Those decisions belong to the appliance policy layer.
    """
    if not is_deployment_request(message):
        return None
    plan = fallback_plan(message)
    plan["policy_preflight"] = True
    return plan
