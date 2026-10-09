"""Cluster-specific privileged job execution."""

from __future__ import annotations

from typing import Any, Callable, Dict

from .cluster_node_removal import removal_message
from .cluster_operations import ClusterOperations
from .gpu_memory_pool_nodes import (
    NODE_GPU_MEMORY_JOB, NODE_REBOOT_JOB, PROGRESS_MESSAGE, change_worker_pool,
    reboot_worker,
)
from .job_vocabulary import (
    CLUSTER_AGENT_DEPLOY_JOB,
    CLUSTER_AGENT_REMOVE_JOB,
    CLUSTER_GPU_LOAD_JOB,
    CLUSTER_GPU_REFRESH_JOB,
    CLUSTER_GPU_UNLOAD_JOB,
    CLUSTER_LLM_DEPLOY_JOB,
    CLUSTER_NODE_PROFILE_JOB,
)
from .jobs import JobStore
from .worker_profile_job import JOINED_REASON, queue_profile_job, run_profile_job


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
        # VD-194 P2: the join ends by queuing the worker's profile, which runs
        # next (one job at a time) and lays the telemetry agent down with it.
        queue_profile_job(store, result["node_id"], JOINED_REASON)
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
        # ACC-117: the message says what was removed AND what was not, from
        # the result, instead of calling every removal clean.
        return store.finish(
            job["id"], state="completed",
            message=removal_message(result), result=result,
        )
    if job_type == CLUSTER_LLM_DEPLOY_JOB:
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
    if job_type == "cluster.app.deploy-researched":
        checkpoint(
            10, "Validating the approved application and cluster policy",
            "validating",
        )
        result = operations.deploy_researched_app(
            job["payload"],
            job["actor"],
            progress=lambda percent, message: checkpoint(
                percent, message, "starting"
            ),
        )
        return store.finish(
            job["id"], state="healthy",
            message="Researched cluster application is running", result=result,
        )
    if job_type == "cluster.app.remove-researched":
        checkpoint(
            30, "Removing the researched application and its network", "starting"
        )
        result = operations.remove_researched_app(job["payload"])
        return store.finish(
            job["id"], state="completed",
            message="Researched cluster application removed", result=result,
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
    if job_type == "cluster.gpu.remove":
        checkpoint(20, "Stopping the vLLM server and Ray units", "starting")
        result = operations.remove_gpu_inference(job["payload"])
        return store.finish(
            job["id"], state="completed",
            message="GPU inference deployment removed", result=result,
        )
    if job_type == CLUSTER_GPU_UNLOAD_JOB:
        checkpoint(20, "Stopping the vLLM units to reclaim the GPU", "starting")
        result = operations.unload_gpu_inference(job["payload"])
        return store.finish(
            job["id"], state="completed",
            message="GPU inference deployment unloaded", result=result,
        )
    if job_type == CLUSTER_GPU_LOAD_JOB:
        checkpoint(10, "Loading the vLLM model back onto the GPU", "starting")
        result = operations.load_gpu_inference(
            job["payload"],
            progress=lambda percent, message: checkpoint(
                percent, message, "starting"
            ),
        )
        return store.finish(
            job["id"], state="healthy",
            message="GPU inference deployment loaded and serving", result=result,
        )
    if job_type == CLUSTER_GPU_REFRESH_JOB:
        checkpoint(5, "Checking whether this release renders the model's units differently", "starting")
        result = operations.refresh_gpu_inference(
            job["payload"],
            progress=lambda percent, message: checkpoint(
                percent, message, "starting"
            ),
        )
        return store.finish(
            job["id"], state="completed",
            message=(
                "GPU inference deployment re-rendered and serving"
                if result.get("refreshed") else
                "GPU inference deployment left as it was: " + str(result.get("reason", ""))
            ),
            result=result,
        )
    if job_type == "cluster.gpu.rotate-key":
        checkpoint(10, "Rotating the internal cluster serving key", "starting")
        result = operations.rotate_gpu_cluster_key(
            job["payload"],
            progress=lambda percent, message: checkpoint(
                percent, message, "starting"
            ),
        )
        return store.finish(
            job["id"], state="completed",
            message="Cluster serving key rotated", result=result,
        )
    if job_type == CLUSTER_AGENT_DEPLOY_JOB:
        checkpoint(10, "Validating the cluster agent and its tool boundary", "validating")
        result = operations.deploy_agent(
            job["payload"],
            progress=lambda percent, message: checkpoint(
                percent, message, "starting"
            ),
        )
        # The one-time inbound gate key is returned to the deploy caller for a
        # single reveal; it must never enter the durable job ledger (and so the
        # operation projection), which only ever carries the safe id + last4.
        result = {key: value for key, value in dict(result).items() if key != "key"}
        return store.finish(
            job["id"], state="healthy",
            message="Cluster agent is healthy behind its inbound gate",
            result=result,
        )
    if job_type == CLUSTER_AGENT_REMOVE_JOB:
        checkpoint(20, "Stopping the cluster agent unit and its inbound gate", "starting")
        result = operations.remove_agent(job["payload"])
        return store.finish(
            job["id"], state="completed",
            message="Cluster agent deployment removed", result=result,
        )
    if job_type == "cluster.model.pull":
        checkpoint(
            5, "Preparing to cache the model on the selected nodes", "starting"
        )
        result = operations.model_library.pull(
            job["payload"],
            progress=lambda percent, message: checkpoint(
                percent, message, "starting"
            ),
        )
        return store.finish(
            job["id"], state="completed",
            message="Model weights cached on the selected nodes", result=result,
        )
    if job_type == "cluster.model.remove":
        checkpoint(20, "Removing the cached model weights", "starting")
        result = operations.model_library.remove(job["payload"])
        return store.finish(
            job["id"], state="completed",
            message="Cached model weights removed", result=result,
        )
    if job_type == CLUSTER_NODE_PROFILE_JOB:
        return run_profile_job(job, operations, store, checkpoint)
    if job_type == NODE_GPU_MEMORY_JOB:
        checkpoint(20, PROGRESS_MESSAGE, "starting")
        result = change_worker_pool(operations, job["payload"])
        return store.finish(
            job["id"], state="completed", message=result["message"], result=result,
        )
    if job_type == NODE_REBOOT_JOB:
        checkpoint(30, "Asking the machine to restart", "starting")
        # The job follows the machine until it answers again; the wait
        # reports through the checkpoint, which is what lets a cancel end it.
        result = reboot_worker(
            operations, job["payload"],
            progress=lambda percent, message: checkpoint(percent, message, "starting"),
            queued_at_ms=int(job["created_at"]),
        )
        return store.finish(
            job["id"], state="completed", message=result["message"], result=result,
        )
    raise ValueError("Choose a supported cluster job type.")
