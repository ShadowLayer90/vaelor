"""Approval plans for single, replicated, and pooled-memory model deployment."""

from __future__ import annotations

from typing import Any, Dict, List

from .cluster_llm_sizing import swarm_llm_memory_limit_mib
from .cluster_placement import CONTROLLER_PLACEMENT_ID, placement_node
from .cluster_plan_contract import ClusterPlanContext, ClusterPlanError
from .pooled_inference import (
    artifact_architecture,
    runtime_artifact,
    runtime_manifest,
    validate_pooled_deployment,
)


DEPLOYMENT_MODES = {"single", "replicated", "pooled"}


def inference_runtimes() -> Dict[str, Any]:
    """The replicated and pooled runtimes an operator may choose between."""
    return {
        "replicated": {
            "engine": "llama.cpp",
            "model_format": "GGUF",
            "node_counts": list(range(1, 9)),
        },
        "pooled": runtime_manifest(),
    }


def _selected_node_ids(deployment_mode: str, body: Dict[str, Any]) -> List[str]:
    if deployment_mode == "single":
        return [str(body.get("node_id", "")).strip()]
    raw_node_ids = body.get("node_ids", [])
    node_ids = (
        list(dict.fromkeys(
            str(node_id).strip()
            for node_id in raw_node_ids
            if str(node_id).strip()
        ))
        if isinstance(raw_node_ids, list) else
        []
    )
    valid_count = (
        len(node_ids) in {2, 4, 8}
        if deployment_mode == "pooled"
        else 2 <= len(node_ids) <= 8
    )
    if not valid_count:
        raise ClusterPlanError(
            "cluster_replica_workers",
            (
                "Choose exactly 2, 4, or 8 joined nodes for "
                "pooled-memory inference."
                if deployment_mode == "pooled"
                else "Choose between 2 and 8 joined workers for "
                "replicated inference."
            ),
        )
    return node_ids


def _pooled_plan(
    body: Dict[str, Any], context: ClusterPlanContext, nodes: List[Dict[str, Any]]
) -> Dict[str, Any]:
    try:
        preflight = validate_pooled_deployment(
            str(body.get("pooled_model", "")), nodes
        )
    except ValueError as error:
        raise ClusterPlanError("pooled_inference_fit", str(error)) from error
    # Selected from the pool's own discovered architecture. This asked for the
    # ARM64 artifact unconditionally, so an x86-64 pool the manifest says is
    # supported could never be approved.
    artifact = runtime_artifact(preflight["architecture"])
    if artifact is None:
        raise ClusterPlanError(
            "pooled_runtime_missing",
            (
                "Install the verified {} Distributed Llama runtime artifact "
                "before approving pooled inference.".format(
                    artifact_architecture(preflight["architecture"])
                )
            ),
            status=409,
        )
    return {
        "title": "Deploy pooled-memory inference",
        "steps": [
            (
                "Verify the pinned Distributed Llama runtime "
                f"{preflight['runtime_commit'][:12]} on every node."
            ),
            (
                f"Store the complete {preflight['model']['name']} "
                f"model on root node {nodes[0]['name']} only."
            ),
            (
                f"Start {len(nodes) - 1} private worker processes, "
                "then start the root API only after all workers answer."
            ),
            (
                "Keep the unencrypted worker protocol on the private "
                "cluster network and expose clients only through the "
                "authenticated Vaelor gateway."
            ),
        ],
        "impact": (
            "Every token depends on every selected node and the wired "
            "network. Losing one node stops this pooled model; Vaelor "
            "must fall back to a replicated or local model."
        ),
        "approval_required": True,
        "preflight": preflight,
        "runtime_artifact_required": True,
        "runtime_artifact": {
            "architecture": artifact["architecture"],
            "commit": artifact["commit"],
            "sha256": artifact["sha256"],
            "verified": True,
        },
        "cluster": context.summary["controller"],
    }


def _refuse_unfit_nodes(
    nodes: List[Dict[str, Any]], model_size: int, service_memory: int
) -> None:
    for node in nodes:
        memory_bytes = int(node.get("inventory", {}).get("memory_bytes", 0))
        storage_bytes = int(node.get("inventory", {}).get("root_free_bytes", 0))
        if service_memory > max(0, memory_bytes - 1024 ** 3):
            raise ClusterPlanError(
                "model_memory_fit",
                (
                    f"This model does not fit on {node['name']} while "
                    "preserving worker OS headroom."
                ),
            )
        if int(model_size * 1.2) + 1024 ** 3 > storage_bytes:
            raise ClusterPlanError(
                "model_storage_fit",
                (
                    f"{node['name']} does not have enough free storage "
                    "for the model."
                ),
            )


def plan_deploy_llm(
    action: str, body: Dict[str, Any], context: ClusterPlanContext
) -> Dict[str, Any]:
    deployment_mode = str(body.get("deployment_mode", "single")).strip().lower()
    if deployment_mode not in DEPLOYMENT_MODES:
        raise ClusterPlanError(
            "cluster_llm_mode",
            "Choose single-worker, replicated, or pooled-memory inference.",
        )
    node_ids = _selected_node_ids(deployment_mode, body)
    if deployment_mode == "pooled" and CONTROLLER_PLACEMENT_ID in node_ids:
        raise ClusterPlanError(
            "pooled_controller_unsupported",
            (
                "Pooled-memory inference currently uses SSH-managed "
                "workers; choose workers only for this mode."
            ),
        )
    nodes = [
        placement_node(context.manager, context.summary, node_id)
        for node_id in node_ids
    ]
    if any(
        node is None or not node.get("labels", {}).get("swarm_node_id")
        for node in nodes
    ):
        raise ClusterPlanError(
            "cluster_worker_required",
            (
                "Choose this initialized controller or active workers "
                "that have joined the cluster."
            ),
        )
    if deployment_mode == "pooled":
        return _pooled_plan(body, context, nodes)
    try:
        model_size = int(body.get("model_size_bytes", 0))
    except (TypeError, ValueError):
        model_size = 0
    if model_size < 32 * 1024 ** 2:
        raise ClusterPlanError(
            "model_size_required",
            "Enter the exact size reported for the reviewed GGUF file.",
        )
    # #138: the same measured sizing the deploy will use, so the number the
    # owner approves is the number the service gets. The estimate this
    # replaces (`model_size * 1.35 + 512 MB`) capped the shipped 4B below its
    # measured cgroup steady state.
    try:
        service_memory = swarm_llm_memory_limit_mib(
            str(body.get("name", "")).strip() or "this service",
            str(body.get("model_repo", "")).strip(),
            str(body.get("model_file", "")).strip(),
            nodes,
        ) * 1024 ** 2
    except ValueError as error:
        raise ClusterPlanError("model_footprint_unmeasured", str(error)) from error
    _refuse_unfit_nodes(nodes, model_size, service_memory)
    worker_names = ", ".join(node["name"] for node in nodes)
    replica_count = len(nodes)
    return {
        "title": (
            "Deploy replicated OpenAI-compatible inference"
            if deployment_mode == "replicated"
            else "Deploy a managed OpenAI-compatible LLM server"
        ),
        "steps": [
            (
                f"Reserve 1 GB for each node OS and cap every replica "
                f"at {service_memory / 1024 ** 3:.1f} GiB."
            ),
            (
                f"Pull {body.get('model_repo', '')}:{body.get('model_file', '')} "
                f"into a node-local cache on {worker_names}."
            ),
            (
                f"Place {replica_count} llama.cpp replica"
                f"{'' if replica_count == 1 else 's'}, at most one per "
                "selected node, behind Swarm ingress."
            ),
            (
                "Require every replica, then test model discovery and chat "
                "completion before registering the endpoint."
            ),
        ],
        "impact": (
            f"Each of {replica_count} nodes downloads its own model copy. "
            "Vaelor refuses any target that cannot preserve RAM or storage headroom."
        ),
        "approval_required": True,
        "terminology": (
            "The inference API is OpenAI-compatible. Vaelor also publishes an "
            "OpenAPI description for client generation."
        ),
        "cluster": context.summary["controller"],
    }


def plan_remove_pooled(
    action: str, body: Dict[str, Any], context: ClusterPlanContext
) -> Dict[str, Any]:
    deployment_name = str(body.get("name", "")).strip().lower()
    deployment = context.manager.store.get_pooled_deployment(deployment_name)
    if deployment is None:
        raise ClusterPlanError(
            "pooled_deployment_not_found",
            "Choose a managed pooled inference deployment.",
            status=404,
        )
    return {
        "title": f"Remove pooled inference {deployment_name}",
        "steps": [
            "Stop the root API before stopping each private worker.",
            "Disable and remove the managed systemd service definitions.",
            "Revoke the gateway credential and remove the deployment record.",
            "Retain the verified runtime and converted model cache for reuse.",
        ],
        "impact": (
            "This pooled model stops immediately. Cached model files remain "
            "on the nodes and can be reused by a later approved deployment."
        ),
        "approval_required": True,
    }
