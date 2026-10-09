"""The confirm token each cluster job type requires, in one home.

This is the pure data mapping `api_workload_routes.create_job` checks a request's
``confirm`` field against before a cluster mutation is allowed to become a job.
It lived inline in `api_workload_routes.py`, which sat at the 1,000-line module
ceiling; moving it here is behaviour-preserving (the same job-type -> token map,
looked up the same way) and gives that route module the headroom a new token
would otherwise have crowded out.

Two cluster job types are deliberately absent because their confirm token is not
a constant: ``cluster.node.availability`` (``set-worker-<availability>``) and
``cluster.service.manage`` (``restart``/``refresh``/``rollback``-``cluster-service``)
compute the expected token from the payload and are checked separately at the
route. Everything whose token is fixed is here.
"""

from __future__ import annotations

from typing import Dict, Optional

from .job_vocabulary import (
    CLUSTER_AGENT_DEPLOY_JOB,
    CLUSTER_AGENT_REMOVE_JOB,
    CLUSTER_GPU_LOAD_JOB,
    CLUSTER_GPU_REFRESH_JOB,
    CLUSTER_GPU_UNLOAD_JOB,
    CLUSTER_LLM_DEPLOY_JOB,
    CLUSTER_NODE_PROFILE_JOB,
)

#: job type -> the exact ``confirm`` value a request must carry. The GPU-remove
#: entry (``cluster.gpu.remove`` -> ``remove-gpu-inference``) mirrors the pooled
#: one and matches the token `gpu_pool_operations.GpuPoolOperations.remove`
#: checks, so the route gate and the operation agree on one string.
CLUSTER_JOB_CONFIRMATIONS: Dict[str, str] = {
    "cluster.initialize": "initialize-head-controller",
    "cluster.node.join": "join-worker-node",
    "cluster.node.remove": "remove-worker-node",
    "cluster.node.gpu-memory": "change-worker-gpu-memory",
    "cluster.node.reboot": "reboot-worker-node",
    CLUSTER_NODE_PROFILE_JOB: "apply-worker-profile",
    "cluster.app.deploy": "deploy-cluster-app",
    "cluster.app.deploy-researched": "deploy-researched-app",
    "cluster.app.remove-researched": "remove-researched-app",
    CLUSTER_LLM_DEPLOY_JOB: "deploy-cluster-llm",
    CLUSTER_AGENT_DEPLOY_JOB: "deploy-cluster-agent",
    CLUSTER_AGENT_REMOVE_JOB: "remove-cluster-agent",
    "cluster.service.backup": "backup-cluster-service",
    "cluster.service.configure": "configure-cluster-service",
    "cluster.service.remove": "remove-cluster-service",
    "cluster.service.restore": "restore-cluster-service",
    "cluster.pooled.remove": "remove-pooled-inference",
    "cluster.gpu.remove": "remove-gpu-inference",
    "cluster.gpu.rotate-key": "rotate-cluster-key",
    CLUSTER_GPU_UNLOAD_JOB: "unload-gpu-inference",
    CLUSTER_GPU_LOAD_JOB: "load-gpu-inference",
    CLUSTER_GPU_REFRESH_JOB: "refresh-gpu-inference",
    "cluster.model.pull": "pull-cluster-model",
    "cluster.model.remove": "remove-cluster-model",
}


#: The one job whose token depends on a flag: an approved FORCED worker removal
#: (owner decision 2026-09-28) carries its own token, so a plain confirm can
#: never be replayed as a forced one. `cluster_node_removal` checks the same
#: pair, plus the typed acknowledgement, when the job runs.
FORCED_NODE_REMOVAL_CONFIRM = "force-remove-worker-node"

#: Taking a worker's profile off carries its own token, so an apply's confirm
#: can never be replayed as a removal (VD-194 P2).
PROFILE_REMOVAL_CONFIRM = "remove-worker-profile"


def confirmation_accepted(job_type: str, payload: object) -> bool:
    """Whether ``payload`` carries the confirm token ``job_type`` requires."""
    if not isinstance(payload, dict):
        return False
    expected = CLUSTER_JOB_CONFIRMATIONS.get(job_type)
    if job_type == "cluster.node.remove" and payload.get("force") is True:
        expected = FORCED_NODE_REMOVAL_CONFIRM
    if job_type == CLUSTER_NODE_PROFILE_JOB and payload.get("action") == "remove":
        expected = PROFILE_REMOVAL_CONFIRM
    return payload.get("confirm") == expected


def confirmation_for(job_type: str) -> Optional[str]:
    """The fixed confirm token for ``job_type``, or ``None`` when it has none."""
    return CLUSTER_JOB_CONFIRMATIONS.get(job_type)
