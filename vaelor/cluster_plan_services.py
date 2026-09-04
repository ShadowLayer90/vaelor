"""Approval plans for the lifecycle of Vaelor-managed cluster services."""

from __future__ import annotations

from typing import Any, Dict

from .app_catalog import APP_TEMPLATES
from .cluster_architecture import architecture_label
from .cluster_driver import ClusterDriverError
from .cluster_plan_contract import (
    ClusterPlanContext,
    ClusterPlanError,
    managed_service_name,
)


RESERVED_APP_PORTS = {34001, 34002}

SERVICE_UPDATE_STEPS = {
    "restart": [
        "Restart one task at a time without changing its image.",
        "Automatically roll back if Docker reports an update failure.",
        "Wait for every configured replica before reporting success.",
    ],
    "refresh": [
        "Resolve the current reviewed image tag to its available digest.",
        "Replace one task at a time with automatic failure rollback.",
        "Wait for every configured replica before reporting success.",
    ],
    "rollback": [
        "Ask Docker Swarm to restore the previous service specification.",
        "Replace one task at a time using the stored rollback policy.",
        "Wait for every configured replica before reporting success.",
    ],
}

SERVICE_UPDATE_IMPACTS = {
    "restart": (
        "A single-replica service may be briefly unavailable while "
        "its replacement starts."
    ),
    "refresh": (
        "A mutable image tag may contain application changes. "
        "Docker will retain the previous specification for rollback."
    ),
    "rollback": (
        "The service returns to Docker's immediately previous "
        "specification; node-local data is not changed."
    ),
}


def plan_deploy_app(
    action: str, body: Dict[str, Any], context: ClusterPlanContext
) -> Dict[str, Any]:
    node = context.manager.store.get_node(str(body.get("node_id", "")))
    if node is None or not node.get("labels", {}).get("swarm_node_id"):
        raise ClusterPlanError(
            "cluster_worker_required",
            "Choose a worker that has joined this cluster.",
        )
    template = APP_TEMPLATES.get(str(body.get("template_id", "")))
    if template is None:
        raise ClusterPlanError(
            "cluster_template_required",
            "Choose a reviewed application from the catalog.",
        )
    try:
        port = int(body.get("port", template["default_port"]))
    except (TypeError, ValueError):
        port = 0
    if (
        not 1024 <= port <= 65535
        or port in RESERVED_APP_PORTS
        or 8100 <= port <= 8199
    ):
        raise ClusterPlanError(
            "cluster_app_port", "Choose an available application port."
        )
    return {
        "title": f"Deploy {template['name']} on {node['name']}",
        "steps": [
            # Named from what was probed on that worker. The step used to read
            # "reviewed ARM64-capable image" on every cluster in existence,
            # which was a claim about the fleet rather than a fact about it.
            (
                "Pull the reviewed {} image {} on the selected worker.".format(
                    architecture_label(
                        node.get("inventory", {}).get("architecture")
                    ),
                    template["image"],
                )
            ),
            (
                f"Reserve {template['memory']} and preserve at least 1 GB "
                "for the worker operating system."
            ),
            (
                f"Publish port {port} through Swarm ingress and wait for "
                "one running replica."
            ),
            (
                "Keep persistent data on the selected worker."
                if template.get("volume")
                else "Run without persistent application storage."
            ),
        ],
        "impact": (
            "Persistent data is node-local; drain or removal requires a backup first."
            if template.get("volume")
            else "The stateless service can be removed and recreated from its template."
        ),
        "approval_required": True,
    }


def plan_remove_service(
    action: str, body: Dict[str, Any], context: ClusterPlanContext
) -> Dict[str, Any]:
    service_name = managed_service_name(body, context)
    return {
        "title": f"Remove {service_name}",
        "steps": [
            "Stop the service and remove its Swarm definition.",
            "Release its published port and scheduling reservation.",
            "Retain node-local volumes for recovery unless separately removed.",
        ],
        "impact": "The service will stop immediately after approval.",
        "approval_required": True,
    }


def plan_service_update(
    action: str, body: Dict[str, Any], context: ClusterPlanContext
) -> Dict[str, Any]:
    service_name = managed_service_name(body, context)
    verb = action.removesuffix("-service")
    titles = {
        "restart": f"Restart {service_name}",
        "refresh": f"Refresh {service_name} from its current image tag",
        "rollback": f"Roll back {service_name}",
    }
    return {
        "title": titles[verb],
        "steps": SERVICE_UPDATE_STEPS[verb],
        "impact": SERVICE_UPDATE_IMPACTS[verb],
        "approval_required": True,
    }


def plan_configure_service(
    action: str, body: Dict[str, Any], context: ClusterPlanContext
) -> Dict[str, Any]:
    service_name = managed_service_name(body, context)
    try:
        replicas = int(body.get("replicas", 0))
        memory_limit = int(body.get("memory_limit_mib", 0))
        memory_reservation = int(body.get("memory_reservation_mib", 0))
        parallelism = int(body.get("update_parallelism", 0))
    except (TypeError, ValueError) as error:
        raise ClusterPlanError(
            "cluster_service_configuration",
            "Replica, memory, and rolling-update settings must be numbers.",
        ) from error
    order = str(body.get("update_order", "")).strip().lower()
    if (
        not 1 <= replicas <= 32
        or not 128 <= memory_limit <= 131072
        or not 64 <= memory_reservation <= memory_limit
        or not 1 <= parallelism <= min(replicas, 8)
        or order not in {"start-first", "stop-first"}
    ):
        raise ClusterPlanError(
            "cluster_service_configuration",
            (
                "Choose 1–32 replicas, a 128 MiB–128 GiB memory "
                "limit, a reservation no larger than that limit, "
                "and a valid rolling-update policy."
            ),
        )
    order_label = (
        "Start replacements before stopping old tasks"
        if order == "start-first"
        else "Stop old tasks before starting replacements"
    )
    return {
        "title": f"Change settings for {service_name}",
        "steps": [
            f"Set the service to {replicas} replica"
            f"{'' if replicas == 1 else 's'}.",
            (
                f"Reserve {memory_reservation} MiB and enforce a "
                f"{memory_limit} MiB memory limit per replica."
            ),
            (
                f"Update {parallelism} task"
                f"{'' if parallelism == 1 else 's'} at a time. "
                f"{order_label}."
            ),
            "Roll back automatically if Docker reports an update failure.",
            "Wait for every configured replica before reporting success.",
        ],
        "impact": (
            "Increasing replicas or memory can consume worker capacity. "
            "Start-first updates temporarily need room for both old and "
            "replacement tasks."
        ),
        "approval_required": True,
    }


def plan_backup_service(
    action: str, body: Dict[str, Any], context: ClusterPlanContext
) -> Dict[str, Any]:
    service_name = str(body.get("service_name", "")).strip()
    if service_name not in context.live_service_names():
        raise ClusterPlanError(
            "cluster_service_not_found",
            "Choose a live managed cluster service.",
            status=404,
        )
    try:
        details = context.manager.driver.service_details(service_name)
    except (ClusterDriverError, ValueError) as error:
        raise ClusterPlanError(
            "cluster_service_unavailable", str(error)
        ) from error
    volumes = [
        item for item in details.get("mounts", [])
        if item.get("type") == "volume"
    ]
    if len(volumes) != 1:
        raise ClusterPlanError(
            "cluster_backup_volume",
            "Backup currently requires exactly one managed named volume.",
        )
    return {
        "title": f"Back up {service_name}",
        "steps": [
            "Re-check the pinned worker identity and service placement.",
            f"Archive named volume {volumes[0]['source']} without changing it.",
            "Copy the archive to the controller and verify its SHA-256 checksum.",
            "Keep the SSH credential and plaintext password out of backup metadata.",
        ],
        "impact": (
            "The service remains online. A busy database may require its "
            "application-native export for transactional consistency."
        ),
        "approval_required": True,
    }


def plan_restore_service(
    action: str, body: Dict[str, Any], context: ClusterPlanContext
) -> Dict[str, Any]:
    service_name = str(body.get("service_name", "")).strip()
    backup_id = str(body.get("backup_id", "")).strip()
    store = context.callbacks.get("cluster_backups")
    try:
        backup = store.get(backup_id, verify=True)
    except (AttributeError, OSError, ValueError) as error:
        raise ClusterPlanError(
            "cluster_backup_unavailable", str(error), status=404
        ) from error
    if backup.get("service_name") != service_name:
        raise ClusterPlanError(
            "cluster_backup_service_mismatch",
            "Choose a backup created for this service.",
        )
    return {
        "title": f"Restore {service_name}",
        "steps": [
            "Verify the selected archive checksum and current worker placement.",
            "Create and verify a new safety backup of the current volume.",
            "Stop every service replica before replacing the volume contents.",
            "Restore the selected archive and wait for every replica to recover.",
            "Automatically restore the safety backup if the operation fails.",
        ],
        "impact": (
            "The service is unavailable during restore. Data written after "
            "the selected backup is replaced, but retained in the safety backup."
        ),
        "approval_required": True,
        "backup": {
            key: backup[key]
            for key in (
                "id", "service_name", "volume", "size_bytes",
                "sha256", "created_at",
            )
        },
    }
