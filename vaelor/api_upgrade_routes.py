"""Authenticated routes for staging and applying a pinned Vaelor upgrade.

``GET /api/v2/upgrade`` reports what the appliance is running, what release is
offered, and whether this appliance is eligible for it. ``POST /api/v2/upgrade``
creates the gated ``appliance.upgrade`` job through the same
``job_store.create`` + confirmation + ``security.audit`` path every other
destructive job uses, so an upgrade is queued, ledgered and audited exactly like
a factory reset rather than through a private side door.

``GET /api/v2/upgrade/running-version`` is the loopback-only surface the root
upgrade broker reads back after it restarts the control plane. It is the honest
answer to "which code is running now": it comes from *this* process - the one
that just restarted - and not from a constant a mid-upgrade executor imported
(#185/#171). It is unauthenticated because the broker has no session, and
restricted to loopback because the running version is the appliance's own
business.
"""

from __future__ import annotations

import ipaddress
from typing import Any, Optional

from flask import g, request

from .api_common import ApiContext, payload as _payload
from .appliance_upgrade import CONFIRMATION, UpgradeApplyPlans, last_result
from .release_source import (
    ReleaseError,
    StubReleaseSource,
    upgrade_eligibility,
)
from .version import __version__


def _release_source(callbacks: dict):
    source = callbacks.get("release_source")
    return source if source is not None else StubReleaseSource()


def _loopback(remote_addr: str) -> bool:
    try:
        return ipaddress.ip_address(remote_addr or "").is_loopback
    except ValueError:
        return False


def register_upgrade_routes(context: ApiContext) -> None:
    blueprint = context.blueprint
    callbacks = context.callbacks
    security = context.security
    require_auth = context.require_auth

    def audit(action: str, result: str, target: str, **details: Any) -> None:
        security.audit(
            g.auth_session.username,
            action,
            result,
            target=str(target)[:256],
            remote_addr=request.remote_addr or "",
            details=details,
        )

    def _offered() -> tuple[Optional[Any], Optional[dict]]:
        """The offered manifest and its eligibility for the running version."""
        source = _release_source(callbacks)
        try:
            manifest = source.latest_manifest()
        except ReleaseError as error:
            return None, {"eligible": False, "reason": str(error)}
        if manifest is None:
            return None, None
        return manifest, upgrade_eligibility(__version__, manifest)

    @blueprint.get("/upgrade/running-version")
    def running_version():
        # The broker's live readback. Loopback-only: this is not a public fact.
        if not _loopback(request.remote_addr or ""):
            return _payload(
                error={
                    "code": "not_found",
                    "message": "This surface is available on the appliance only.",
                },
                status=404,
            )
        return _payload({"running_version": __version__})

    @blueprint.get("/upgrade")
    @require_auth("viewer")
    def upgrade_status():
        source = _release_source(callbacks)
        manifest, eligibility = _offered()
        return _payload({
            "running_version": __version__,
            "source": getattr(source, "name", "unknown"),
            "manifest": manifest.public() if manifest is not None else None,
            "eligibility": eligibility,
            "confirmation": CONFIRMATION,
            "last_result": last_result(),
        })

    @blueprint.post("/upgrade")
    @require_auth("administrator", csrf=True)
    def create_upgrade():
        job_store = callbacks.get("job_store")
        if job_store is None:
            return _payload(
                error={
                    "code": "job_service_unavailable",
                    "message": "The upgrade job service is not available on this appliance.",
                },
                status=503,
            )
        body = request.get_json(silent=True) or {}
        job_payload = body.get("payload", {})
        manifest, eligibility = _offered()
        if manifest is None:
            return _payload(
                error={
                    "code": "no_release_available",
                    "message": "No pinned Vaelor release is available to install.",
                },
                status=409,
            )
        if not eligibility or not eligibility.get("eligible"):
            return _payload(
                error={
                    "code": "upgrade_not_eligible",
                    "message": (eligibility or {}).get("reason")
                    or "This appliance is not eligible for the offered release.",
                },
                status=409,
            )
        if (
            not isinstance(job_payload, dict)
            or job_payload.get("confirmation") != CONFIRMATION
            or job_payload.get("to_version") != manifest.version
        ):
            audit("upgrade.create", "failure", manifest.version)
            return _payload(
                error={
                    "code": "upgrade_confirmation_required",
                    "message": (
                        "Review the offered release and type the exact "
                        "confirmation to apply it."
                    ),
                },
                status=400,
            )
        # A fresh plan slot: the executor stages the verified wheel into it.
        UpgradeApplyPlans().cancel()
        try:
            job = job_store.create(
                "appliance.upgrade",
                g.auth_session.username,
                {
                    "confirmation": CONFIRMATION,
                    "to_version": manifest.version,
                },
            )
        except ValueError as error:
            return _payload(
                error={"code": "invalid_job_request", "message": str(error)},
                status=400,
            )
        audit(
            "upgrade.create", "success", job["id"],
            to_version=manifest.version, sha256=manifest.sha256,
        )
        return _payload(job, status=202)
