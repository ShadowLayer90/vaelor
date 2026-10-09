"""The single-worker and replicated branch of the cluster LLM deployment.

This is the validation + service-creation + readiness-wait + rollback body of
`cluster_operations.ClusterOperations.deploy_llm`, housed here so that module
stays under the 1,000-line ceiling with real headroom for the GPU and pooled
deploy delegations that share its entry point. It is a strict, behaviour-
preserving move: the pooled and GPU modes return earlier from `deploy_llm`, so
what reaches here is only ``single`` or ``replicated`` inference on the CPU
Swarm path (`self.driver.deploy_llm`), and the existing `deploy_llm` tests
exercise it unchanged through the same public method.

It is written as a function taking the `ClusterOperations` instance rather than a
class of its own because it is a continuation of one method, not a new
responsibility: it reaches back through the operations facade for the driver, the
credential broker, and the node-resolution helpers (`_placement_node`,
`_validate_worker`) that the rest of `deploy_llm` already owns. The endpoint
readiness probe uses ``urllib.request.urlopen`` exactly as before; because that
resolves to the shared ``urllib.request`` module object, the deploy tests'
``@patch("vaelor.cluster_operations.urllib.request.urlopen")`` still intercepts
it here.
"""

from __future__ import annotations

import hashlib
import json
import re
import time
import urllib.error
import urllib.request
from typing import Any, Dict

from .cluster_llm_sizing import swarm_llm_memory_limit_mib
from .cluster_placement import CONTROLLER_PLACEMENT_ID
from .cluster_capacity import worker_placement_state
from .model_footprint import MIB

#: The live placement states in which a worker takes no NEW model.
OUT_OF_SERVICE_STATES = frozenset({"drained", "paused"})


def refuse_out_of_service_nodes(nodes, status) -> None:
    """Refuse a model deployment onto a drained or paused worker.

    ACC-118, and review SC1/SC2. Every model path asks here: the CPU Swarm
    path, the GPU (vLLM) path and the pooled CPU path - the last two run
    systemd units over SSH and never pass through Swarm's scheduler, so a drain
    did nothing to stop them landing on the machine.

    **The answer is the LIVE placement reading**, `worker_placement_state` over
    the cluster's own node list (``status``), the same one owner the Fleet
    ledger uses - not the stored ``node.state``. A worker drained outside
    Vaelor (``docker node update --availability drain``) is refused, and so is
    a row an older Recheck reset to ``enrolled`` while Swarm still holds it
    drained. The controller is never drained.
    """
    for node in nodes:
        if not node or str(node.get("id", "")) == CONTROLLER_PLACEMENT_ID:
            continue
        verdict = worker_placement_state(node, status)
        # Review R6: when the live list cannot be read, the drain Vaelor
        # itself recorded still stands - never place work on a machine this
        # controller drained just because Swarm did not answer.
        unread_but_drained = verdict["state"] == "unknown" and node.get(
            "state"
        ) in {"drain", "pause"}
        if verdict["state"] in OUT_OF_SERVICE_STATES or unread_but_drained:
            raise ValueError(
                f"Return {node['name']} to service before deploying a model."
            )


def deploy_single_or_replicated(
    operations, payload: Dict[str, Any], deployment_mode: str, progress=None
) -> Dict[str, Any]:
    """Validate, create, and health-check a single or replicated cluster LLM.

    ``operations`` is the `ClusterOperations` whose ``deploy_llm`` delegated here
    after handling the ``pooled`` and ``gpu`` modes; ``deployment_mode`` is the
    already-normalised ``single`` or ``replicated`` string it computed.
    """
    if deployment_mode not in {"single", "replicated"}:
        raise ValueError("Choose single-worker or replicated inference.")
    if deployment_mode == "replicated":
        raw_node_ids = payload.get("node_ids", [])
        if not isinstance(raw_node_ids, list):
            raise ValueError("Choose the workers for replicated inference.")
        node_ids = list(dict.fromkeys(
            str(node_id).strip()
            for node_id in raw_node_ids
            if str(node_id).strip()
        ))
        if not 2 <= len(node_ids) <= 8:
            raise ValueError(
                "Choose between 2 and 8 workers for replicated inference."
            )
    else:
        node_ids = [str(payload.get("node_id", "")).strip()]
    # The pooled branch returns at the top of `deploy_llm`, so a "pooled +
    # controller" guard here is unreachable. The SSH-only protection lives
    # in pooled_operations.deploy, on the path pooled deploys actually run
    # (#Recovery-7b).
    nodes = [operations._placement_node(node_id) for node_id in node_ids]
    refuse_out_of_service_nodes(nodes, operations.driver.status())
    for node in nodes:
        if node["id"] != CONTROLLER_PLACEMENT_ID:
            operations._validate_worker(
                node.get("inventory", {}),
                where=str(node.get("host") or node.get("name") or ""),
            )

    name = str(payload.get("name", "")).strip()
    repository = str(payload.get("model_repo", "")).strip()
    model_file = str(payload.get("model_file", "")).strip()
    if not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9._-]{0,39}", name):
        raise ValueError("Use a short letters-and-numbers deployment name.")
    if not re.fullmatch(
        r"[A-Za-z0-9][A-Za-z0-9._-]{0,95}/[A-Za-z0-9][A-Za-z0-9._-]{0,95}",
        repository,
    ):
        raise ValueError("Choose a valid reviewed Hugging Face repository.")
    if not re.fullmatch(
        r"[A-Za-z0-9][A-Za-z0-9._-]{0,199}\.gguf", model_file, re.I
    ):
        raise ValueError("Choose a reviewed GGUF model file.")
    try:
        model_size = int(payload.get("model_size_bytes", 0))
        port = int(payload.get("port", 8100))
    except (TypeError, ValueError) as error:
        raise ValueError("Model size and API port must be numbers.") from error
    if model_size < 32 * 1024 ** 2:
        raise ValueError("Use the inspected GGUF file size before deployment.")
    if not 8100 <= port <= 8199:
        raise ValueError("Cluster LLM ports must be between 8100 and 8199.")

    # #138: measured sizing, shared with the approval plan.
    memory_limit_mib = swarm_llm_memory_limit_mib(
        name, repository, model_file, nodes
    )
    required_memory = memory_limit_mib * MIB
    required_storage = int(model_size * 1.2) + 1024 ** 3
    for node in nodes:
        inventory = node.get("inventory", {})
        memory_bytes = int(inventory.get("memory_bytes", 0))
        if required_memory > max(0, memory_bytes - 1024 ** 3):
            raise ValueError(
                f"This model does not fit on {node['name']} while preserving "
                "1 GB for its operating system."
            )
        if int(inventory.get("root_free_bytes", 0)) < required_storage:
            raise ValueError(
                f"{node['name']} does not have enough free storage for this model."
            )
    report = progress or (lambda _percent, _message: None)
    swarm_node_ids = [
        str(node.get("labels", {}).get("swarm_node_id", ""))
        for node in nodes
    ]
    if CONTROLLER_PLACEMENT_ID in node_ids:
        controller_index = node_ids.index(CONTROLLER_PLACEMENT_ID)
        operations.driver.label_node(
            swarm_node_ids[controller_index],
            {"vaelor.node_id": CONTROLLER_PLACEMENT_ID},
        )
    pool_label = ""
    labeled_nodes: list[str] = []
    result = None
    try:
        if deployment_mode == "replicated":
            pool_label = "vaelor.llm_pool_{}".format(
                hashlib.sha256(
                    "{}:{}:{}".format(
                        name, repository, ",".join(sorted(node_ids))
                    ).encode()
                ).hexdigest()[:12]
            )
            for swarm_node_id in swarm_node_ids:
                operations.driver.label_node(
                    swarm_node_id, {pool_label: "true"}
                )
                labeled_nodes.append(swarm_node_id)
        report(
            20,
            (
                f"Creating {len(nodes)} replicated inference services"
                if deployment_mode == "replicated"
                else "Creating the constrained cluster inference service"
            ),
        )
        result = operations.driver.deploy_llm(
            name=name,
            node_label=nodes[0]["id"] if deployment_mode == "single" else None,
            pool_label=pool_label or None,
            replicas=len(nodes),
            model_repo=repository,
            model_file=model_file,
            memory_limit_mib=memory_limit_mib,
            port=port,
        )
        report(55, "Waiting for every inference replica to become ready")
        operations.driver.wait_service(
            result["name"],
            timeout=max(
                30,
                min(int(payload.get("startup_timeout_seconds", 900)), 1800),
            ),
            expected_replicas=len(nodes),
        )
    except Exception:
        if result:
            operations.driver.remove_service(result["name"])
        else:
            for swarm_node_id in labeled_nodes:
                try:
                    operations.driver.remove_node_label(swarm_node_id, pool_label)
                except Exception:
                    pass
        raise

    endpoint = f"http://127.0.0.1:{port}/v1"
    health = f"http://127.0.0.1:{port}/health"
    deadline = time.monotonic() + max(
        30, min(int(payload.get("startup_timeout_seconds", 900)), 1800)
    )
    last_error = ""
    try:
        while time.monotonic() < deadline:
            try:
                with urllib.request.urlopen(health, timeout=4) as response:
                    if response.status == 200:
                        break
            except (OSError, urllib.error.URLError) as error:
                last_error = str(error)[:240]
            report(75, "Waiting for the model to load across the Swarm network")
            time.sleep(3)
        else:
            raise RuntimeError(
                "The cluster model did not become healthy: {}".format(last_error)
            )
        profile = json.dumps({
            "base_url": endpoint, "model": "", "api_key": "",
        }, separators=(",", ":"))
        credential = operations.broker.put(
            "openai-compatible", f"Cluster LLM · {name}", profile
        )
        try:
            test = operations.broker.test(credential["id"])
            if not test.get("ok"):
                raise RuntimeError(
                    str(test.get("message", "The cluster model API test failed."))
                )
            # VD-202 item 1: never the Assistant's lease (refused at deploy_llm).
            operations.broker.activate(credential["id"], "cluster-inference")
        except Exception:
            operations.broker.delete(credential["id"])
            raise
    except Exception:
        operations.driver.remove_service(result["name"])
        raise
    return {
        **result,
        "node_id": nodes[0]["id"],
        "node_ids": node_ids,
        "deployment_mode": deployment_mode,
        "replicas": len(nodes),
        "endpoint": endpoint,
        "health": health,
        "credential_id": credential["id"],
        "assistant_active": False,
        "model_size_bytes": model_size,
    }
