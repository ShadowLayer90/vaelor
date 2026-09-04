"""Feature gates for staged application discovery and deployment."""

from __future__ import annotations

from typing import Dict


def application_features(model_present: bool) -> Dict[str, bool]:
    """Return the staged application workflow capabilities.

    The whole workflow - research, then drafting, then deploy/install - is
    gated on one prerequisite: a working Assistant model connection is
    configured (an active ``deployment-agent`` credential). There is no
    per-stage environment toggle. Owner decision (2026-08-19, VD-109):
    "once the Assistant model is downloaded and installed it should work;
    there should be no separate toggle or additional work needed." The three
    stages move together because a working model is the single thing they all
    depend on; the per-install approval and the operator/admin role remain the
    per-action safety gates.

    Kept pure - it takes the presence signal as a ``bool`` so it stays
    trivially testable and free of any broker dependency. Compute
    ``model_present`` with ``assistant_model_configured`` at the call site.
    """
    return {
        "research": model_present,
        "drafts": model_present,
        "deploy": model_present,
    }
