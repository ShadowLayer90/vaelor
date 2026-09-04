"""Administrator-only fleet enrollment and cluster planning routes."""

from __future__ import annotations

from flask import g, request

from .api_common import ApiContext, payload
from .credential_broker import CredentialError
from .cluster_driver import ClusterDriverError
from .ssh_transport import SshTransportError
from .cluster_placement import add_controller_placement
from .cluster_plan_contract import ClusterPlanContext, ClusterPlanError
from .cluster_plan_inference import inference_runtimes
from .cluster_plans import build_cluster_plan


def register_cluster_routes(context: ApiContext) -> None:
    blueprint = context.blueprint
    callbacks = context.callbacks
    require_auth = context.require_auth
    security = context.security

    @blueprint.get("/cluster")
    @require_auth("operator")
    def cluster_summary():
        manager = callbacks.get("cluster_manager")
        if manager is None:
            return payload(
                error={"code": "cluster_unavailable", "message": "Fleet management is unavailable."},
                status=503,
            )
        result = manager.summary(context.appliance_address())
        probe = callbacks.get("hardware_inventory")
        hardware = probe() if probe is not None else {}
        add_controller_placement(result, hardware)
        return payload(result)

    @blueprint.get("/cluster/services/<service_name>")
    @require_auth("operator")
    def cluster_service_details(service_name):
        manager = callbacks.get("cluster_manager")
        try:
            return payload(manager.driver.service_details(service_name))
        except (AttributeError, ClusterDriverError, ValueError) as error:
            return payload(
                error={
                    "code": "cluster_service_unavailable",
                    "message": str(error),
                },
                status=400,
            )

    @blueprint.get("/cluster/services/<service_name>/logs")
    @require_auth("operator")
    def cluster_service_logs(service_name):
        manager = callbacks.get("cluster_manager")
        try:
            lines = int(request.args.get("lines", 200))
            return payload(
                manager.driver.service_logs(service_name, lines=lines)
            )
        except (AttributeError, ClusterDriverError, TypeError, ValueError) as error:
            return payload(
                error={
                    "code": "cluster_service_logs_unavailable",
                    "message": str(error),
                },
                status=400,
            )

    @blueprint.post("/cluster/services/<service_name>/diagnostics")
    @require_auth("operator", csrf=True)
    def cluster_service_diagnostics(service_name):
        operations = callbacks.get("cluster_operations")
        body = request.get_json(silent=True) or {}
        try:
            result = operations.service_diagnostics(
                service_name,
                str(body.get("tool", "stats")),
            )
        except (
            AttributeError, CredentialError, ClusterDriverError,
            SshTransportError, ValueError,
        ) as error:
            return payload(
                error={
                    "code": "cluster_service_diagnostic_failed",
                    "message": str(error),
                },
                status=400,
            )
        security.audit(
            g.auth_session.username,
            "cluster.service.diagnostic",
            "success",
            target=service_name,
            remote_addr=request.remote_addr or "",
            details={"tool": result["tool"], "node_id": result["node_id"]},
        )
        return payload(result)

    @blueprint.get("/cluster/backups")
    @require_auth("operator")
    def cluster_backups():
        store = callbacks.get("cluster_backups")
        if store is None:
            return payload(
                error={
                    "code": "cluster_backups_unavailable",
                    "message": "Cluster backup inventory is unavailable.",
                },
                status=503,
            )
        return payload(store.list(
            service_name=str(request.args.get("service_name", ""))[:80],
            limit=request.args.get("limit", 100),
        ))

    @blueprint.get("/cluster/backups/<backup_id>")
    @require_auth("operator")
    def cluster_backup_details(backup_id):
        store = callbacks.get("cluster_backups")
        try:
            result = store.get(
                backup_id,
                verify=str(request.args.get("verify", "")).lower()
                in {"1", "true", "yes"},
            )
            result.pop("archive_path", None)
            return payload(result)
        except (AttributeError, OSError, ValueError) as error:
            return payload(
                error={
                    "code": "cluster_backup_unavailable",
                    "message": str(error),
                },
                status=404,
            )

    @blueprint.delete("/cluster/backups/<backup_id>")
    @require_auth("administrator", csrf=True)
    def cluster_backup_delete(backup_id):
        store = callbacks.get("cluster_backups")
        body = request.get_json(silent=True) or {}
        try:
            result = store.delete(
                backup_id,
                confirmation=str(body.get("confirmation", "")),
            )
        except (AttributeError, OSError, ValueError) as error:
            return payload(
                error={
                    "code": "cluster_backup_delete_failed",
                    "message": str(error),
                },
                status=400,
            )
        security.audit(
            g.auth_session.username,
            "cluster.backup.delete",
            "success",
            target=backup_id,
            remote_addr=request.remote_addr or "",
        )
        return payload(result)

    @blueprint.get("/cluster/inference/runtimes")
    @require_auth("operator")
    def cluster_inference_runtimes():
        return payload(inference_runtimes())

    @blueprint.post("/cluster/ssh-fingerprint")
    @require_auth("administrator", csrf=True)
    def cluster_fingerprint():
        body = request.get_json(silent=True) or {}
        manager = callbacks.get("cluster_manager")
        try:
            result = manager.inspect_host(body.get("host", ""), body.get("port", 22))
        except (AttributeError, OSError, SshTransportError, ValueError) as error:
            return payload(
                error={"code": "ssh_inspection_failed", "message": str(error)},
                status=400,
            )
        security.audit(
            g.auth_session.username, "cluster.ssh.inspect", "success",
            target=result["host"], remote_addr=request.remote_addr or "",
        )
        return payload(result)

    @blueprint.post("/cluster/nodes")
    @require_auth("administrator", csrf=True)
    def cluster_enroll():
        body = request.get_json(silent=True) or {}
        manager = callbacks.get("cluster_manager")
        try:
            node = manager.enroll(body)
        except (
            AttributeError, CredentialError, OSError, SshTransportError, ValueError
        ) as error:
            security.audit(
                g.auth_session.username, "cluster.node.enroll", "failure",
                target=str(body.get("host", ""))[:80],
                remote_addr=request.remote_addr or "",
            )
            return payload(
                error={"code": "cluster_enrollment_failed", "message": str(error)},
                status=400,
            )
        security.audit(
            g.auth_session.username, "cluster.node.enroll", "success",
            target=node["id"], remote_addr=request.remote_addr or "",
        )
        return payload(node, status=201)

    @blueprint.post("/cluster/nodes/<node_id>/refresh")
    @require_auth("administrator", csrf=True)
    def cluster_refresh(node_id):
        manager = callbacks.get("cluster_manager")
        try:
            node = manager.refresh_node(node_id)
        except (
            AttributeError, CredentialError, OSError, SshTransportError, ValueError
        ) as error:
            return payload(
                error={"code": "cluster_refresh_failed", "message": str(error)},
                status=400,
            )
        return payload(node)

    @blueprint.delete("/cluster/nodes/<node_id>")
    @require_auth("administrator", csrf=True)
    def cluster_remove(node_id):
        body = request.get_json(silent=True) or {}
        if body.get("confirm") != "remove-enrollment":
            return payload(
                error={
                    "code": "confirmation_required",
                    "message": "Confirm removal of this unjoined enrollment.",
                },
                status=400,
            )
        manager = callbacks.get("cluster_manager")
        try:
            removed = manager.remove_node(node_id)
        except ValueError as error:
            return payload(
                error={"code": "cluster_node_joined", "message": str(error)},
                status=400,
            )
        if not removed:
            return payload(
                error={"code": "cluster_node_not_found", "message": "Cluster node was not found."},
                status=404,
            )
        security.audit(
            g.auth_session.username, "cluster.node.remove", "success",
            target=node_id, remote_addr=request.remote_addr or "",
        )
        return payload({"removed": True})

    @blueprint.post("/cluster/architecture/evictions")
    @require_auth("administrator", csrf=True)
    def cluster_evict_mismatched():
        """Drain and unenrol nodes of the wrong architecture (VD-033).

        Audited per node and after the fact, so the trail records what was
        actually destroyed rather than what was asked for. A node that could
        not be classified is never a target, and a target that could not be
        removed is audited as a failure rather than quietly dropped.
        """
        operations = callbacks.get("cluster_operations")
        if operations is None:
            return payload(
                error={
                    "code": "cluster_unavailable",
                    "message": "Fleet management is unavailable.",
                },
                status=503,
            )
        body = request.get_json(silent=True) or {}
        try:
            result = operations.evict_mismatched_nodes(body)
        except (
            AttributeError, CredentialError, ClusterDriverError,
            SshTransportError, ValueError,
        ) as error:
            return payload(
                error={
                    "code": "cluster_eviction_failed",
                    "message": str(error),
                },
                status=400,
            )
        for record in result["evicted"] + result["failed"]:
            security.audit(
                g.auth_session.username,
                "cluster.node.architecture-eviction",
                "success" if record.get("removed") else "failure",
                target=str(record.get("node_id", "")),
                remote_addr=request.remote_addr or "",
                details={
                    "host": record.get("host", ""),
                    "node_architecture": record.get("architecture", ""),
                    "controller_architecture": record.get(
                        "controller_architecture", ""
                    ),
                    "reason": record.get("reason", ""),
                    "drained": bool(record.get("drained")),
                    "forced": bool(record.get("forced")),
                    "error": record.get("error", ""),
                },
            )
        return payload(result)

    @blueprint.post("/cluster/plan")
    @require_auth("administrator", csrf=True)
    def cluster_plan():
        body = request.get_json(silent=True) or {}
        manager = callbacks.get("cluster_manager")
        summary = manager.summary()
        summary["controller"]["candidate_address"] = context.appliance_address()
        probe = callbacks.get("hardware_inventory")
        hardware = probe() if probe is not None else {}
        add_controller_placement(summary, hardware)
        try:
            return payload(build_cluster_plan(
                str(body.get("action", "")).strip(),
                body,
                ClusterPlanContext(
                    manager=manager, summary=summary, callbacks=callbacks
                ),
            ))
        except ClusterPlanError as error:
            return payload(
                error={"code": error.code, "message": error.message},
                status=error.status,
            )
