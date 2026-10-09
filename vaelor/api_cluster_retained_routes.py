"""Routes for the data volumes a cluster app removal kept (W4d-D20).

Split out of ``api_cluster_routes`` (1,000-line ceiling). The records and the
delete live in `cluster_retained_volumes`; these routes read the live service
list through the cluster manager's brokered driver, as every other cluster read
in this process does (#141).
"""

from __future__ import annotations

from flask import g, request

from .api_common import ApiContext, payload
from .cluster_driver import ClusterDriverError
from .cluster_retained_volumes import RetainedVolumeStore, delete_retained_volume
from .cluster_service_state import live_service_names
from .credential_broker import CredentialError
from .ssh_transport import SshTransport


def _live_services(callbacks) -> set:
    """Running service names; empty when unreadable (the worker's docker still
    refuses to delete a volume a running container uses)."""
    try:
        return set(live_service_names(callbacks.get("cluster_manager").driver.status()))
    except (AttributeError, ClusterDriverError, ValueError):
        return set()


def register_cluster_retained_routes(context: ApiContext) -> None:
    blueprint = context.blueprint
    callbacks = context.callbacks
    require_auth = context.require_auth
    security = context.security

    @blueprint.get("/cluster/retained-volumes")
    @require_auth("operator")
    def cluster_retained_volumes():
        backups = callbacks.get("cluster_backups")
        if backups is None:
            return payload(
                error={
                    "code": "cluster_retained_unavailable",
                    "message": "The list of retained cluster data is unavailable.",
                },
                status=503,
            )

        def backups_for(service_name: str) -> int:
            try:
                return len(backups.list(service_name=service_name, limit=500))
            except (OSError, ValueError):
                return 0

        try:
            return payload(RetainedVolumeStore(backups).list(
                live_services=_live_services(callbacks), backups_for=backups_for,
            ))
        except (AttributeError, OSError) as error:
            return payload(
                error={"code": "cluster_retained_unavailable", "message": str(error)},
                status=503,
            )

    @blueprint.delete("/cluster/retained-volumes/<record_id>")
    @require_auth("administrator", csrf=True)
    def cluster_retained_volume_delete(record_id):
        body = request.get_json(silent=True) or {}
        try:
            result = delete_retained_volume(
                callbacks.get("cluster_operations"), record_id,
                str(body.get("confirmation", "")),
                live_services=_live_services(callbacks),
                transport_factory=SshTransport,
                forget_only=bool(body.get("forget_only")),
                removed_ack=str(body.get("removed_ack", "")),
            )
        except (AttributeError, CredentialError, OSError, ValueError) as error:
            return payload(
                error={"code": "cluster_retained_delete_failed", "message": str(error)},
                status=400,
            )
        security.audit(
            g.auth_session.username,
            "cluster.retained_volume.delete",
            "success",
            target=result.get("volume", record_id),
            remote_addr=request.remote_addr or "",
            details={"forgotten": bool(result.get("forgotten")),
                     "already_gone": bool(result.get("already_gone"))},
        )
        return payload(result)
