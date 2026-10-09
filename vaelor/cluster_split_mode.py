"""How a model split across machines is split: pipeline (the default) or tensor.

VD-156 made every split a pipeline: each machine holds whole layers and passes
its result on once a step. The owner re-opened the other way on 2026-10-01 as an
EXPLICIT choice (VD-167): tensor-parallel splits every layer across the
machines, so they exchange data on every layer. It is never picked for the
owner. The fit and the form say what was measured on the lab pair - one user
got 24 tokens a second against 41 for the pipeline split, and the link carried
about 101 times the traffic - and recommend the fastest link the machines share
for it (`cluster_link_recommendation`); the owner decides, and Ethernet is
allowed for it, with that honest note.

What the choice changes is only the parallel degrees across machines
(:func:`degrees`); sizing is the split's own, and a machine is still
tensor-parallel across its own GPUs either way. vLLM refuses a tensor degree
that does not divide the model's attention heads, so the fit and the deploy
refuse it first, in the same words (:func:`tensor_refusal`).
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional, Tuple

from .cluster_gpu_sizing import VERDICT_DISTRIBUTED

#: The deploy and fit field, and its two values.
SPLIT_MODE_FIELD = "split_mode"
SPLIT_PIPELINE = "pipeline"
SPLIT_TENSOR = "tensor"
SPLIT_MODES = (SPLIT_PIPELINE, SPLIT_TENSOR)

#: What the fit says beside a tensor-parallel split, in plain words.
TENSOR_NOTE = (
    "Tensor-parallel splits every layer across the machines, so they exchange "
    "data on every layer. Measured between two machines, one user got 24 tokens "
    "a second against 41 for a pipeline split, and the link carried about 101 "
    "times the traffic. Over ordinary Ethernet it will be slower still; a "
    "fast dedicated link helps. The choice is yours."
)

#: The refusal when the model's heads do not divide across the machines.
TENSOR_HEADS_REFUSED = (
    "This model cannot be split tensor-parallel {degree} ways ({count} "
    "machines): its {heads} attention heads do not divide evenly between them. "
    "Split it as a pipeline instead."
)


def parse_split_mode(value: Any) -> str:
    """The owner's choice; nothing chosen is the pipeline split."""
    if value is None or value == "":
        return SPLIT_PIPELINE
    if value not in SPLIT_MODES:
        raise ValueError(
            "A split is either 'pipeline' or 'tensor'; '{}' is neither.".format(str(value)[:40])
        )
    return str(value)


def tensor_refusal(spec: Any, machines: int, devices: int = 1) -> Optional[str]:
    """Why vLLM would refuse this tensor degree, or ``None`` when it divides.

    The rule vLLM applies: the attention heads divide by the degree, and the
    KV heads either divide by it or are a divisor of it (then replicated).
    """
    degree, _pipeline = degrees(SPLIT_TENSOR, max(1, int(machines)), machine_devices(devices))
    heads = int(getattr(spec, "attention_heads", 0) or 0)
    kv_heads = int(getattr(spec, "kv_heads", 0) or 0)
    if heads <= 0 or heads % degree or (kv_heads > 0 and kv_heads % degree and degree % kv_heads):
        return TENSOR_HEADS_REFUSED.format(degree=degree, count=machines, heads=heads or "unknown")
    return None


def machine_devices(value: Any) -> int:
    """GPUs a machine contributes, as the deploy counts them (`gpu_fit_node`)."""
    try:
        return max(1, int(value or 1))
    except (TypeError, ValueError):
        return 1


def degrees(mode: str, machines: int, devices: int) -> Tuple[int, int]:
    """``(tensor_parallel_size, pipeline_parallel_size)`` for a split."""
    if mode == SPLIT_TENSOR:
        return int(machines) * int(devices), 1
    return int(devices), int(machines)


UNEQUAL_GPUS = (
    "The machines of this split do not have the same number of GPUs ({counts}). "
    "A split runs the same share on every machine, so choose machines with "
    "the same number of GPUs."
)


def split_devices(named_counts: Any) -> int:
    """The GPUs every machine of a split has, or the refusal naming each machine.

    The ONE derivation the preview and the deploy share (review 2, SC4): the
    degrees are machines x this number, and machines with unequal counts would
    ask Ray for GPUs some machine does not have - a placement that waits for
    ever - so they are refused, in plain words.
    """
    counts = [(str(name), machine_devices(count)) for name, count in named_counts]
    if len({count for _name, count in counts}) > 1:
        raise ValueError(UNEQUAL_GPUS.format(counts=", ".join(
            "{} has {}".format(name, count) for name, count in counts)))
    return counts[0][1] if counts else 1


def placed_devices(decision: Dict[str, Any], ledger: Any) -> List[Tuple[str, int]]:
    """``[(name, GPUs)]`` for each machine the fit placed, from the capacity ledger."""
    by_id = {node.get("node_id"): node for node in (ledger or {}).get("nodes") or []}
    found = []
    for view in (decision.get("placement") or {}).get("nodes") or []:
        gpu = (((by_id.get(view.get("node_id")) or {}).get("capacity") or {}).get("gpu") or {})
        found.append((str(view.get("name") or view.get("node_id") or ""),
                      machine_devices(gpu.get("device_count"))))
    return found


def annotate_fit(
    decision: Dict[str, Any], mode: str, spec: Any,
    recommendation: Optional[Dict[str, Any]] = None, device_counts: Any = (),
) -> Dict[str, Any]:
    """``decision`` saying which split was asked for, and what that means.

    Only a split (``distributed``) changes: machines with unequal GPU counts
    are refused in either mode (`split_devices`); a tensor choice carries the
    note, the link recommendation and, when the heads do not divide, the
    refusal.
    """
    decision = {**decision, SPLIT_MODE_FIELD: mode}
    if decision.get("verdict") != VERDICT_DISTRIBUTED:
        return decision
    try:
        devices = split_devices(device_counts)
    except ValueError as error:
        if not decision.get("refusal"):
            decision["refusal"] = {"code": "unequal_gpus", "message": str(error)}
        return decision
    if mode != SPLIT_TENSOR:
        return decision
    placement = dict(decision.get("placement") or {})
    placement["parallelism"] = SPLIT_TENSOR
    placement["parallelism_reason"] = TENSOR_NOTE
    decision = {**decision, "placement": placement}
    if recommendation is not None:
        decision["link_recommendation"] = recommendation
    # One derivation with the deploy: machines x each machine's (equal) GPUs.
    refusal = tensor_refusal(spec, int(placement.get("node_count") or 0), devices)
    if refusal is not None and not decision.get("refusal"):
        decision["refusal"] = {"code": "tensor_split_refused", "message": refusal}
    return decision
