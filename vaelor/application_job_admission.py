"""Whether an application research or compose-draft job may be queued.

Moved, unchanged in behaviour, out of ``api_workload_routes.create_job``, which
sat at the 1,000-line ceiling. The two staged application capabilities are
admitted only when the feature is enabled and the payload names exactly one
server-owned draft (and, for research, optional source URLs).
"""

from __future__ import annotations

from typing import Any, Mapping, Optional, Tuple

from .application_features import application_features
from .model_connection import assistant_model_configured

#: The job types this admission check owns.
APPLICATION_JOB_TYPES = frozenset({"application.research", "compose.draft"})


def application_job_refusal(
    job_type: str, job_payload: Any, callbacks: Mapping[str, Any],
) -> Optional[Tuple[str, str, int]]:
    """``(code, message, status)`` refusing the job, or ``None`` to admit it."""
    if job_type not in APPLICATION_JOB_TYPES:
        return None
    features = application_features(
        assistant_model_configured(callbacks.get("credential_broker"))
    )
    required = "research" if job_type == "application.research" else "drafts"
    if not features[required]:
        return (
            "application_feature_disabled",
            "This staged application capability is not enabled.", 409,
        )
    if not isinstance(job_payload, dict) or set(job_payload) - {
        "draft_id", "source_urls"
    } or not str(job_payload.get("draft_id", "")).startswith("appdraft_"):
        return ("invalid_application_job", "Choose a server-owned application draft.", 400)
    if job_type == "compose.draft" and "source_urls" in job_payload:
        return ("invalid_application_job", "Compose drafts accept only a draft ID.", 400)
    return None
