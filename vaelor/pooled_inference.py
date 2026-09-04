"""Distributed Llama runtime manifest and beginner-safe topology validation."""

from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
from typing import Any, Dict, Iterable

DISTRIBUTED_LLAMA_REPOSITORY = "https://github.com/b4rtaz/distributed-llama.git"
DISTRIBUTED_LLAMA_COMMIT = "59af889085c6c0316a4524a92b42f04caa4bcc6d"
SUPPORTED_NODE_COUNTS = {2, 4, 8}

POOLED_MODEL_CATALOG = {
    "qwen3_0.6b_q40": {
        "name": "Qwen 3 0.6B Q40",
        "download_bytes": 900_000_000,
        "kv_heads": 8,
        "weights_float_type": "q40",
        "buffer_float_type": "q80",
        "max_sequence_length": 4096,
    },
    "llama3_2_3b_instruct_q40": {
        "name": "Llama 3.2 3B Instruct Q40",
        "download_bytes": 3_400_000_000,
        "kv_heads": 8,
        "weights_float_type": "q40",
        "buffer_float_type": "q80",
        "max_sequence_length": 4096,
    },
    "qwen3_8b_q40": {
        "name": "Qwen 3 8B Q40",
        "download_bytes": 6_700_000_000,
        "kv_heads": 8,
        "weights_float_type": "q40",
        "buffer_float_type": "q80",
        "max_sequence_length": 4096,
    },
}


#: Architectures the distributed runtime is built for. This is the same list
#: ``runtime_manifest()`` publishes; keeping one constant stops the manifest
#: and the admission check from disagreeing again.
SUPPORTED_POOLED_ARCHITECTURES = frozenset({"aarch64", "x86_64"})


def _normalize_architecture(value: Any) -> str:
    normalized = str(value or "").strip().lower()
    return {
        "arm64": "aarch64",
        "amd64": "x86_64",
        "x86-64": "x86_64",
    }.get(normalized, normalized)


def runtime_manifest() -> Dict[str, Any]:
    return {
        "engine": "distributed-llama",
        "repository": DISTRIBUTED_LLAMA_REPOSITORY,
        "commit": DISTRIBUTED_LLAMA_COMMIT,
        "license": "MIT",
        "architectures": sorted(SUPPORTED_POOLED_ARCHITECTURES),
        "node_counts": sorted(SUPPORTED_NODE_COUNTS),
        "network": "private wired Ethernet",
        "worker_port": 9999,
        "models": [
            {"id": model_id, **details}
            for model_id, details in POOLED_MODEL_CATALOG.items()
        ],
    }


def artifact_architecture(architecture: str = "arm64") -> str:
    """The name the runtime artifact for an architecture is published under.

    The manifest speaks `uname -m` (`aarch64`) and the artifact filenames speak
    Docker (`arm64`). Callers that name the missing artifact in a refusal go
    through this, so the message and the file it looked for cannot disagree.
    """
    return str(architecture).lower().replace("aarch64", "arm64")


def runtime_artifact(
    architecture: str = "arm64", search_roots=None
) -> Dict[str, Any] | None:
    normalized = artifact_architecture(architecture)
    configured = os.environ.get("VAELOR_RUNTIME_ARTIFACTS", "")
    roots = list(search_roots or (
        ([Path(configured)] if configured else [])
        + [
            Path(__file__).resolve().parents[1] / "dist" / "runtime-artifacts",
            Path("/opt/vaelor/runtime-artifacts"),
        ]
    ))
    for root in roots:
        root = Path(root)
        manifest_path = root / f"distributed-llama-{normalized}.json"
        try:
            manifest = json.loads(manifest_path.read_text("utf-8"))
            archive = root / str(manifest["archive"])
            if (
                manifest.get("commit") != DISTRIBUTED_LLAMA_COMMIT
                or manifest.get("architecture") != normalized
                or not archive.is_file()
            ):
                continue
            digest = hashlib.sha256(archive.read_bytes()).hexdigest()
            if digest != manifest.get("sha256"):
                continue
            return {
                **manifest,
                "path": str(archive),
                "verified": True,
            }
        except (KeyError, OSError, ValueError, json.JSONDecodeError):
            continue
    return None


def node_thread_count(inventory: Dict[str, Any]) -> int:
    """The logical/thread count for one node, with the shared field fallback.

    #113: ``cpu_threads`` is the logical count. A record persisted before #113 -
    or enrolled then and upgraded without a re-probe (LESSONS 13, a deployment
    fact the suite cannot construct) - has only ``cpu_count``, which was
    cores == threads on the non-SMT parts that shipped then, so it is the honest
    fallback. The admission gate below and ``pooled_operations.worker_thread_count``
    both resolve the field through here, so a legacy node can never be accepted
    by one and rejected by the other - which is the asymmetry this closes.
    """
    raw = inventory.get("cpu_threads", inventory.get("cpu_count", 4))
    try:
        return int(raw)
    except (TypeError, ValueError):
        return 4


def validate_pooled_deployment(
    model_id: str, nodes: Iterable[Dict[str, Any]]
) -> Dict[str, Any]:
    selected = list(nodes)
    model = POOLED_MODEL_CATALOG.get(str(model_id))
    if model is None:
        raise ValueError(
            "Choose a converted model from the pooled-memory catalog."
        )
    if len(selected) not in SUPPORTED_NODE_COUNTS:
        raise ValueError("Pooled-memory inference requires exactly 2, 4, or 8 nodes.")
    if len(selected) > int(model["kv_heads"]):
        raise ValueError(
            "This model has fewer KV heads than the selected node count."
        )
    architectures = {
        _normalize_architecture(
            node.get("inventory", {}).get("architecture", "")
        )
        for node in selected
    }
    # The runtime manifest has always advertised aarch64 and x86_64, but this
    # check accepted only aarch64, so an x86 node could never join a pool the
    # manifest said it supported. Mixing architectures in one pool is still
    # refused: the runtime distributes identical tensor shards and a
    # heterogeneous pool has no verified behaviour.
    if len(architectures) != 1:
        raise ValueError(
            "Pooled-memory inference requires every worker to share one "
            "processor architecture."
        )
    if not architectures.issubset(SUPPORTED_POOLED_ARCHITECTURES):
        raise ValueError(
            "The pooled-memory runtime supports 64-bit ARM or x86-64 workers."
        )
    for node in selected:
        if node.get("state") != "joined":
            raise ValueError(f"{node.get('name', 'A worker')} is not active.")
        inventory = node.get("inventory", {})
        # #113: this gate sizes the pool by the worker's thread count, so it
        # reads `cpu_threads` (logical CPUs), not `cpu_count` (physical cores),
        # and its message names threads. It resolves the field through
        # `node_thread_count` - the SAME fallback `worker_thread_count` uses - so
        # a legacy record carrying only `cpu_count` cannot pass one consumer and
        # fail the other. Reading `cpu_threads` with a 0 default here rejected
        # exactly those upgraded pre-#113 nodes.
        if node_thread_count(inventory) < 4:
            raise ValueError(
                f"{node.get('name', 'A worker')} needs at least four CPU threads."
            )

    model_bytes = int(model["download_bytes"])
    shared_model_bytes = model_bytes // len(selected)
    worker_required = int(shared_model_bytes * 1.35) + 768 * 1024 ** 2
    root_required = worker_required + 768 * 1024 ** 2
    for index, node in enumerate(selected):
        available = max(
            0, int(node.get("inventory", {}).get("memory_bytes", 0)) - 768 * 1024 ** 2
        )
        required = root_required if index == 0 else worker_required
        if available < required:
            role = "root" if index == 0 else "worker"
            raise ValueError(
                f"{node.get('name', 'A node')} does not have enough reserved "
                f"memory to act as the pooled {role}."
            )
    root_free = int(
        selected[0].get("inventory", {}).get("root_free_bytes", 0)
    )
    if root_free < model_bytes + 2 * 1024 ** 3:
        raise ValueError(
            "The pooled root node needs the complete converted model plus 2 GB free."
        )
    return {
        "engine": "distributed-llama",
        "runtime_commit": DISTRIBUTED_LLAMA_COMMIT,
        # The pool is homogeneous by the check above, so this single value is
        # the whole pool's architecture. Both the plan and the operation select
        # the runtime artifact from it; they used to ask for the ARM64 artifact
        # unconditionally, which no x86-64 pool could ever satisfy.
        "architecture": sorted(architectures)[0],
        "model": {"id": model_id, **model},
        "model_path": (
            f"models/{model_id}/dllama_model_{model_id}.m"
        ),
        "tokenizer_path": (
            f"models/{model_id}/dllama_tokenizer_{model_id}.t"
        ),
        "node_count": len(selected),
        "root_node_id": selected[0]["id"],
        "worker_node_ids": [node["id"] for node in selected[1:]],
        "memory": {
            "estimated_root_bytes": root_required,
            "estimated_worker_bytes": worker_required,
            "model_stored_on_root_only": True,
        },
        "network": {
            "transport": "private wired Ethernet",
            "worker_port": 9999,
            "encrypted": False,
            "gateway_required": True,
        },
    }
