"""Cluster-specific privileged job execution."""

from __future__ import annotations

from typing import Any, Callable, Dict

from .cluster_operations import ClusterOperations
from .jobs import JobStore


def execute_cluster_job(
    job: Dict[str, Any],
    operations: ClusterOperations,
    store: JobStore,
    checkpoint: Callable[[int, str, str], None],
) -> Dict[str, Any]:
    job_type = job["type"]
    if job_type == "cluster.initialize":
        checkpoint(25, "Initializing the encrypted cluster control plane", "starting")
        result = operations.initialize(job["payload"])
        return store.finish(
            job["id"], state="healthy",
            message="This Vaelor node is now the head controller", result=result,
        )
    if job_type == "cluster.node.join":
        checkpoint(20, "Verifying the worker and preparing Docker", "starting")
        result = operations.join_node(job["payload"])
        return store.finish(
            job["id"], state="healthy",
            message="Worker joined and placement labels applied", result=result,
        )
    if job_type == "cluster.node.availability":
        checkpoint(30, "Changing worker scheduling availability", "starting")
        result = operations.set_node_availability(job["payload"])
        return store.finish(
            job["id"], state="completed",
            message="Worker scheduling availability updated", result=result,
        )
    if job_type == "cluster.node.remove":
        checkpoint(20, "Draining and removing the selected worker", "starting")
        result = operations.remove_node(job["payload"])
        return store.finish(
            job["id"], state="completed",
            message="Worker removed and its SSH credential deleted", result=result,
        )
    if job_type == "cluster.llm.deploy":
        checkpoint(10, "Validating worker capacity and model fit", "validating")
        result = operations.deploy_llm(
            job["payload"],
            progress=lambda percent, message: checkpoint(
                percent, message, "starting"
            ),
        )
        return store.finish(
            job["id"], state="healthy",
            message="Cluster LLM is healthy and registered", result=result,
        )
    if job_type == "cluster.app.deploy":
        checkpoint(10, "Validating worker capacity and application policy", "validating")
        result = operations.deploy_app(
            job["payload"],
            progress=lambda percent, message: checkpoint(
                percent, message, "starting"
            ),
        )
        return store.finish(
            job["id"], state="healthy",
            message="Cluster application is running", result=result,
        )
    if job_type == "cluster.service.manage":
        action = str(job["payload"].get("action", "")).strip().lower()
        checkpoint(
            25,
            f"Preparing reviewed cluster service {action}",
            "starting",
        )
        result = operations.manage_service(job["payload"])
        return store.finish(
            job["id"], state="healthy",
            message=f"Cluster service {action} completed", result=result,
        )
    if job_type == "cluster.service.backup":
        checkpoint(20, "Preparing a verified worker volume backup", "starting")
        result = operations.backup_service(job["payload"])
        return store.finish(
            job["id"], state="completed",
            message="Cluster service backup verified", result=result,
        )
    if job_type == "cluster.service.configure":
        checkpoint(
            20,
            "Applying reviewed cluster service settings",
            "starting",
        )
        result = operations.configure_service(job["payload"])
        return store.finish(
            job["id"], state="healthy",
            message="Cluster service configuration applied", result=result,
        )
    if job_type == "cluster.service.restore":
        checkpoint(
            15,
            "Creating a safety backup before service restore",
            "starting",
        )
        result = operations.restore_service(job["payload"])
        return store.finish(
            job["id"], state="healthy",
            message="Cluster service backup restored", result=result,
        )
    if job_type == "cluster.service.remove":
        checkpoint(30, "Removing the reviewed cluster service", "starting")
        result = operations.remove_service(job["payload"])
        return store.finish(
            job["id"], state="completed",
            message="Cluster service removed", result=result,
        )
    if job_type == "cluster.pooled.remove":
        checkpoint(20, "Stopping the pooled root and worker services", "starting")
        result = operations.remove_pooled_inference(job["payload"])
        return store.finish(
            job["id"], state="completed",
            message="Pooled inference deployment removed", result=result,
        )
    raise ValueError("Choose a supported cluster job type.")
