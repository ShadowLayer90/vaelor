"""Will this model fit on the cluster's GPUs, and how should it run across them.

Phase 2a of the cluster capacity foundation. Phase 1
(`cluster_capacity.compute_capacity_ledger`) answers *what does each node have
and what is left*; this module answers the next question the deploy UI asks
*before* the owner commits: **can this model run on the GPU memory the cluster
actually has free, and if so, on one node or sharded across several?**

Like its Phase-1 sibling the decision is a **pure function** of already-fetched
facts - a model spec and a list of GPU-capable node dicts - so every verdict is
unit-tested with no SSH and no Docker (`plan_gpu_fit`). `api_cluster_routes`
does the fetching (through `cluster_manager.capacity_ledger()`) and hands the
GPU nodes here. The spec itself - the request body validated into a
:class:`~vaelor.cluster_gpu_model_spec.GpuModelSpec` - is
`cluster_gpu_model_spec`'s, re-exported here so callers import one module.

Five honest verdicts, and no silent failure. (A sixth, ``single`` - one GPU
node's free memory holds the whole thing - is what ``single_refused`` below
replaced under the capacity intent; the engine no longer produces it, and the
deploy makes no placement from anything but ``distributed`` and ``replicated``.)

* ``distributed`` - too big for any one node, but the N largest GPU nodes hold
  it split between them. **The split is pipeline-parallel, always** (owner
  decision, 2026-09-30): each machine holds whole layers and hands its result
  on once a step. Tensor-parallel across machines is never chosen, whatever
  link the request names - measured between two machines over Thunderbolt it
  moved about 100 times the data and halved one user's speed. The split is
  for CAPACITY (a model, or a context, one machine cannot hold), and the
  verdict says so; it is not the way to more speed, which is ``replicated``.
* ``replicated`` (VD-129, the throughput intent) - one copy of the whole model
  on EACH selected machine, behind one balancer on this controller. Every node
  must hold the whole model at the fraction vLLM will be launched with:
  ``min(usable, gpu_memory_utilization x addressable) >= required``, and the
  preview is handed the same utilisation the deploy launches with, so the two
  cannot disagree. Answers are not faster; the cluster holds its response
  budget at higher concurrency, and the sentence says so with the measured
  figures rather than a promise.
* ``wont_fit`` - it does not fit, stated with the deficit in GB and the
  concrete ways out: raise each node's GPU memory pool to an achievable value
  sized from its own system RAM minus headroom, choose a smaller quantization,
  or add a node (capacity) - or split it instead (throughput). A replica that
  a bigger pool or a shorter context WOULD hold is told so, with the numbers
  (:func:`_replica_ways_out`).
* ``single_refused`` (VD-125, D6) - it fits one of the machines the operator
  selected, and clustering is therefore the wrong tool: the product serves a
  one-box model as AI Chat on llama.cpp (Mode A), and vLLM is never offered for
  a single node. Refused HERE, in the pure engine, so the ``/cluster/fit``
  preview and the deploy say the same sentence rather than the preview promising
  a placement the deploy then refuses in its own words. Under the throughput
  intent a ONE-machine selection is refused with the same sentence: one replica
  is no replication.
* ``unsupported_intent`` (D6/D10, VD-129) - the throughput intent's one slice
  not built: replication WITHOUT this controller holding a replica. The
  endpoint and its balancer run on the controller, so a selection that leaves
  it out is refused with the sentence that says to select it.

**Mode A's memory is counted when the switch will free it (D1).** On the
controller the AI-Chat model holds most of the aperture, and the mode switch
stops it before vLLM starts - so sizing against what is free while the 27B is
loaded refuses a model that would comfortably fit. A node may therefore carry
``reclaimable_gpu_bytes``, decided by
:func:`vaelor.gpu_serving_target.gpu_chat_reclaimable` (managed-local target AND
the model actually running) and derived by
:func:`vaelor.cluster_capacity.reclaimable_gpu_bytes`, and the verdict's own
summary says so out loud rather than reporting a free figure nobody can see.
"""

from __future__ import annotations

from dataclasses import replace
from typing import Any, Dict, List, Optional

from .cluster_gpu_model_spec import (  # noqa: F401 - the engine's public spec API
    BYTES_PER_PARAMETER, DEFAULT_KV_SLOTS, RUNTIME_RESERVE_BYTES,
    WEIGHT_OVERHEAD_FRACTION, GpuModelSpec,
    build_model_spec, bytes_per_parameter, context_that_fits,
    normalize_quantization, with_launch_cache, with_speculation,
)
#: What would make a refused model fit, and the one name of the machine
#: setting those sentences point at, live in `cluster_gpu_ways_out`.
from .cluster_gpu_ways_out import (  # noqa: F401 - re-exported for their readers
    CONTEXT_WAY_OUT, OPTION_ADD_MACHINE, OPTION_RAISE_POOL,
    OPTION_SHORTER_CONTEXT, OPTION_SMALLER_MODEL, POOL_SETTING_NAME,
    POOL_WAY_OUT, SPLIT_POOL_WAY_OUT, STARTUP_NO_BIGGER_POOL,
    STARTUP_RAISE_POOL, STARTUP_WONT_FIT_SUMMARY, achievable_gtt_ceiling,
    pool_headroom, replica_ways_out, with_raised_pools, wont_fit_options,
)
from .cluster_gpu_ways_out import gb as _gb
from .cluster_placement import CONTROLLER_PLACEMENT_ID


#: How a model is split across machines unless the owner chooses tensor-parallel
#: (VD-167, `cluster_split_mode`), and why that is the default
#: (owner decision 2026-09-30, from the two-machine benchmark of that night:
#: pipeline moved about 5.8 KB a token over the link and ran as fast on 1 GbE
#: as on Thunderbolt; tensor-parallel moved about 592 KB a token and gave one
#: user 24 tokens a second where the pipeline gave 41). There is no link fast
#: enough among machines to change the answer, so the request's ``link`` word
#: is echoed and decides nothing.
SPLIT_PARALLELISM = "pipeline"
SPLIT_PARALLELISM_REASON = (
    "A model split across machines runs as a pipeline: each machine holds "
    "whole layers and passes its result on once a step. Splitting every "
    "layer across machines instead was measured between two machines over "
    "Thunderbolt: it moved about 100 times the data and halved one user's "
    "speed, so a split is a pipeline unless tensor-parallel is chosen."
)
#: What a split is for, said with the verdict. Measured with the recommended
#: 30B model across two machines: 22.9 ms a token against 19.5 on one.
SPLIT_IS_CAPACITY = (
    "This is for capacity, not speed: it holds a model or a context too big "
    "for one machine. Measured across two machines, each word took about 15% "
    "longer than on one."
)

#: Plain-language names for the KV and overhead math, returned on the wire so a
#: reader never has to reconstruct where a number came from.
KV_CACHE_FORMULA = (
    "context_tokens x kv_layers x kv_heads x head_dim, for each of K and V, "
    "at the cache dtype bit-width (f16 = 2 bytes per element), times the slot "
    "count; kv_layers is every hidden layer unless the model's config.json "
    "marks some as linear attention, whose fixed per-sequence state is added "
    "per slot instead"
)
OVERHEAD_FORMULA = (
    "WEIGHT_OVERHEAD_FRACTION x weights, plus a fixed runtime reserve for "
    "each serving process (compiled graphs, activations, the allocator), "
    "larger for a hybrid model and for one that drafts tokens; measured on "
    "vLLM 0.27"
)

#: What the operator is asking clustering FOR (VD-125, D6). ``capacity`` is the
#: bigger-than-one-box case this engine shards; ``throughput`` is the same model
#: replicated for concurrency (VD-129). The word is a parameter rather than an
#: assumption because "single when it fits" only ever produces the capacity case,
#: so a throughput request answered by a capacity verdict would be a wrong answer
#: dressed as a right one.
INTENT_CAPACITY = "capacity"
INTENT_THROUGHPUT = "throughput"

#: The launch fraction each intent defaults to, and the one home of both. The
#: distributed head keeps the 0.90 the pipeline proof ran on; a replica defaults
#: to 0.80 because the 0.90 default left no CUDA-graph capture headroom on a
#: 30 GiB unified aperture (VD-127 cleanup item 9c: the 32B FP8 sized its KV
#: cache to the brim and died in the largest capture). The fit sizes a replica
#: at THIS fraction, the deploy launches it at this fraction, and the route
#: reports it, so preview equals deploy on the number that decides a fit.
DISTRIBUTED_GPU_MEMORY_UTILIZATION = 0.90
REPLICA_GPU_MEMORY_UTILIZATION = 0.80
GPU_MEMORY_UTILIZATION_MIN = 0.10
#: The highest launch fraction the fit takes. vLLM's start-up briefly holds
#: more than its share - measured up to about 1.8 GiB above 0.80 of the pool
#: (`cluster_gpu_model_spec`) - so a fraction near 1 leaves that nowhere to go
#: and runs the machine's GPU memory pool out. 0.95 keeps about 2 GiB of a
#: 44 GiB pool free for it.
GPU_MEMORY_UTILIZATION_MAX = 0.95

#: The two verdicts a placement is made from, and the verdicts it cannot be
#: made from with the sentence each carries. All are produced by the pure
#: engine so `/cluster/fit` and the deploy refuse with identical words:
#: `gpu_pool_operations._placement` places ``distributed`` and ``replicated``
#: and raises every other verdict's summary verbatim rather than writing a
#: refusal of its own. ``replicated`` is also what a replicated record's
#: ``units["mode"]`` says (`gpu_pool_units` reads it), so the word is one.
VERDICT_DISTRIBUTED = "distributed"
VERDICT_REPLICATED = "replicated"
VERDICT_WONT_FIT = "wont_fit"
VERDICT_SINGLE_REFUSED = "single_refused"
VERDICT_UNSUPPORTED_INTENT = "unsupported_intent"

SINGLE_REFUSED_SUMMARY = (
    "This model fits one machine ({}). Serve it as the AI Chat model on that "
    "machine instead of clustering."
)

#: The throughput intent's three sentences (VD-129), verbatim from the design.
#: The controller-less refusal is the ``unsupported_intent`` this intent still
#: has; the per-node refusal is its ``wont_fit``; the third is the verdict.
REPLICATED_CONTROLLER_REQUIRED_SUMMARY = (
    "Replicating without this controller is not available yet: the endpoint "
    "and its balancer run on the controller, so select it too."
)
REPLICATED_WONT_FIT_SUMMARY = (
    "Replicating runs the whole model on each machine. It needs about "
    "{required} GiB; {node} has {usable} GiB at {utilisation:.0%} GPU memory "
    "use. Choose 'Run a model bigger than one machine' to split it across "
    "machines, or a smaller model."
)
#: "At least", not "about": `_replica_requests` is a floor (the whole-context
#: requests the KV budget holds after the weights and the overhead), and on
#: the real Strix shape an 8B at its full 32768 context at 0.80 answers 1 -
#: "about 1" would read as an estimate where the figure is a guarantee. The
#: noun agrees with the count through `replicated_summary`.
REPLICATED_SUMMARY = (
    "Runs one copy of the model on each of {N} machines behind one endpoint on "
    "this controller, with room for at least {K} full-context {requests} per "
    "machine. Answers are not faster; the cluster keeps its response budget at "
    "higher concurrency — about {N} machines' worth. Measured on this "
    "class with an 8B model: two machines held the budget at about 64 "
    "simultaneous requests where one held it at about 32."
)

#: What a typed launch context is refused with: it is the context vLLM is
#: launched at (``--max-model-len``), and vLLM refuses one above the model's
#: own without an override this product does not set.
MAX_MODEL_LEN_RANGE = (
    "The max context must be a whole number between 1 and the model's context "
    "length ({})."
)

#: How a node's reclaimable memory is described, so the operator can see that a
#: figure larger than the node's free memory is not a mistake. Written once and
#: formatted into every verdict summary that depends on it.
RECLAIM_PHRASE = "{} GiB free plus {} GiB reclaimed by stopping the AI Chat model"
FREE_PHRASE = "{} GiB free"

#: What a verdict says when a node's Mode A liveness could not be read at all
#: (VD-127). Without it a bridge timeout reached the operator as a plain "will
#: not fit" - a refusal presented as a measurement - for exactly the models the
#: mode switch exists to make room for. Appended to whichever verdict was
#: reached, because the caveat applies to all of them equally.
UNREADABLE_RECLAIM_NOTE = (
    " Sized without the AI Chat model's memory because its state could not be "
    "read."
)


def replicated_summary(count: int, room: int) -> str:
    """:data:`REPLICATED_SUMMARY` for ``count`` machines with ``room`` requests each."""
    return REPLICATED_SUMMARY.format(
        N=count, K=room, requests="request" if room == 1 else "requests",
    )


def validate_max_model_len(value: Any, spec: GpuModelSpec) -> int:
    """The context the launch uses: the operator's ``max_model_len``, else the model's.

    ONE derivation for the fit and the deploy (VD-129, the one-derivation
    rule): the fit sizes the KV cache at this number and the deploy launches
    ``--max-model-len`` at it, so a typed context cannot be previewed at the
    model's full window and launched at a fraction of it. ``None`` and an empty
    string mean the model's own context; anything else must be a whole number
    in ``1..context_length``.
    """
    if value is None or value == "":
        return int(spec.context_length)
    try:
        length = int(value)
    except (TypeError, ValueError):
        raise ValueError(MAX_MODEL_LEN_RANGE.format(spec.context_length))
    if not 1 <= length <= int(spec.context_length):
        raise ValueError(MAX_MODEL_LEN_RANGE.format(spec.context_length))
    return length


def launch_spec(spec: GpuModelSpec, max_model_len: Any) -> GpuModelSpec:
    """``spec`` sized at the context the launch uses (`validate_max_model_len`)."""
    return replace(spec, context_length=validate_max_model_len(max_model_len, spec))


def default_gpu_memory_utilization(intent: Any) -> float:
    """The launch fraction an intent defaults to, when the operator set none."""
    if str(intent or INTENT_CAPACITY) == INTENT_THROUGHPUT:
        return REPLICA_GPU_MEMORY_UTILIZATION
    return DISTRIBUTED_GPU_MEMORY_UTILIZATION


def validate_gpu_memory_utilization(
    value: Any, *, intent: Any = INTENT_CAPACITY
) -> float:
    """The fraction vLLM is launched at, validated; the intent's default for none.

    One rule for the band, read by the fit (which sizes a replica at it), the
    deploy (which launches at it) and the runtime's ``--gpu-memory-utilization``
    (which refuses anything outside it) - three readers, one spelling
    (LESSONS pattern 6). ``None`` and an empty string mean "the default", so a
    form that sends nothing and a form that sends the default agree.
    """
    if value is None or value == "":
        return default_gpu_memory_utilization(intent)
    try:
        fraction = float(value)
    except (TypeError, ValueError):
        raise ValueError("The GPU memory utilization must be a number.")
    if not GPU_MEMORY_UTILIZATION_MIN <= fraction <= GPU_MEMORY_UTILIZATION_MAX:
        raise ValueError(
            "The GPU memory utilization must be between {:.2f} and {:.2f}.".format(
                GPU_MEMORY_UTILIZATION_MIN, GPU_MEMORY_UTILIZATION_MAX
            )
        )
    return fraction


def gpu_nodes_from_ledger(ledger: Dict[str, Any]) -> List[Dict[str, Any]]:
    """The GPU-capable nodes of a capacity ledger, shaped for `plan_gpu_fit`.

    A node whose `capacity.gpu.present` is false (no accelerator, or enrolled
    before GPU discovery) is dropped, so the planner never treats an honest
    absence as zero free memory it might size against.
    """
    nodes: List[Dict[str, Any]] = []
    for node in ledger.get("nodes", []) or []:
        capacity = node.get("capacity", {}) or {}
        gpu = capacity.get("gpu", {}) or {}
        if not gpu.get("present"):
            continue
        free = node.get("free", {}) or {}
        nodes.append({
            "node_id": str(node.get("node_id", "")),
            "name": str(node.get("name", "")),
            "free_gpu_bytes": int(free.get("gpu_memory_bytes", 0) or 0),
            # The ledger derived this from the node's own `mode_a_reclaimable`
            # fact through `cluster_capacity.reclaimable_gpu_bytes`, the same
            # call `cluster_capacity.gpu_fit_node` makes for the deploy - one
            # engine, so the preview and the deploy free the same memory.
            "reclaimable_gpu_bytes": int(gpu.get("reclaimable_bytes", 0) or 0),
            # Empty unless the reclaimable question could not be ASKED, in which
            # case the verdict says it was sized without that memory.
            "reclaimable_reason": str(
                gpu.get("mode_a_reclaimable_reason", "") or ""
            ),
            "addressable_bytes": int(gpu.get("addressable_bytes", 0) or 0),
            "vram_total_bytes": int(gpu.get("vram_total_bytes", 0) or 0),
            "gtt_total_bytes": int(gpu.get("gtt_total_bytes", 0) or 0),
            "system_ram_bytes": gpu.get("system_ram_bytes"),
            "gfx_target_version": str(gpu.get("gfx_target_version", "") or ""),
        })
    return nodes


def _node_free(node: Dict[str, Any]) -> int:
    """What is free on this node right now, with nothing stopped."""
    return int(node.get("free_gpu_bytes", 0) or 0)


def _node_reclaimable(node: Dict[str, Any]) -> int:
    """What the mode switch would give this node back by stopping Mode A (D1)."""
    return int(node.get("reclaimable_gpu_bytes", 0) or 0)


def _node_usable(node: Dict[str, Any]) -> int:
    """The memory a cluster placement may actually count on for this node.

    Free plus reclaimable, because the deploy stops the AI-Chat model before vLLM
    starts - so the preview and the deploy have to size against the same number
    or the preview promises room the deploy would then refuse (VD-B3b-1's rule,
    applied to memory the switch frees rather than memory a probe reported).
    """
    return _node_free(node) + _node_reclaimable(node)


def _node_addressable(node: Dict[str, Any]) -> int:
    return int(node.get("addressable_bytes", 0) or 0)


def _launch_budget(node: Dict[str, Any], utilisation: float) -> int:
    """What a replica may actually take on this node (VD-129).

    vLLM reserves ``gpu_memory_utilization x addressable`` at launch, whatever
    is free, so a replica's budget is the smaller of that reservation and the
    usable memory: a node with room to spare cannot lend a fraction it never
    reserves, and a node whose reservation exceeds what the switch frees is
    the OOM the fit exists to predict.
    """
    return min(_node_usable(node), int(_node_addressable(node) * utilisation))


def _free_phrase(node: Dict[str, Any]) -> str:
    """How much this node contributes, said so a reader can check the sum."""
    reclaimable = _node_reclaimable(node)
    if reclaimable <= 0:
        return FREE_PHRASE.format(_gb(_node_free(node)))
    return RECLAIM_PHRASE.format(_gb(_node_free(node)), _gb(reclaimable))


def _unreadable_note(nodes: List[Dict[str, Any]]) -> str:
    """The caveat for a fleet whose reclaimable memory could not be read.

    One sentence however many nodes carry the reason, because the operator's
    question is "was this figure sized with the AI Chat model's memory or
    without it", not which machine failed to answer - the log carries that.
    """
    if any(str(node.get("reclaimable_reason", "") or "") for node in nodes):
        return UNREADABLE_RECLAIM_NOTE
    return ""


def _reclaim_note(nodes: List[Dict[str, Any]]) -> str:
    """A trailing sentence naming every node whose memory is being reclaimed.

    Empty when nothing is reclaimed, so a verdict on a fleet that is not serving
    anything reads exactly as it did before.
    """
    reclaiming = [node for node in nodes if _node_reclaimable(node) > 0]
    if not reclaiming:
        return ""
    return " " + " ".join(
        "{} contributes {}.".format(
            node.get("name") or node.get("node_id"), _free_phrase(node)
        )
        for node in reclaiming
    )


def _node_view(node: Dict[str, Any]) -> Dict[str, Any]:
    """The subset of a GPU node the decision echoes back, free memory first."""
    return {
        "node_id": node.get("node_id", ""),
        "name": node.get("name", ""),
        "free_gpu_bytes": _node_free(node),
        "free_gpu_gb": _gb(_node_free(node)),
        # Reported beside the free figure rather than folded into it, so a
        # surface can show both and nobody has to reconcile a "free" number
        # against what the GPU is visibly holding.
        "reclaimable_gpu_bytes": _node_reclaimable(node),
        "reclaimable_gpu_gb": _gb(_node_reclaimable(node)),
        "reclaimable_reason": str(node.get("reclaimable_reason", "") or ""),
        "usable_gpu_bytes": _node_usable(node),
        "usable_gpu_gb": _gb(_node_usable(node)),
        "addressable_bytes": _node_addressable(node),
        "gtt_total_bytes": int(node.get("gtt_total_bytes", 0) or 0),
        "system_ram_bytes": node.get("system_ram_bytes"),
        "gfx_target_version": node.get("gfx_target_version", ""),
    }


def _model_view(spec: GpuModelSpec, required: Dict[str, int]) -> Dict[str, Any]:
    return {
        "name": spec.name,
        "weight_bytes": required["weight_bytes"],
        "weight_gb": _gb(required["weight_bytes"]),
        "weight_estimated": spec.weight_estimated,
        "weight_source": (
            "estimated-from-parameters" if spec.weight_estimated else "measured"
        ),
        "parameter_billions": spec.parameter_billions,
        "quantization": spec.quantization,
        "bytes_per_parameter": spec.bytes_per_parameter,
        "hidden_layers": spec.hidden_layers,
        "kv_layers": (
            spec.hidden_layers if spec.kv_layers is None else spec.kv_layers
        ),
        "recurrent_state_bytes": spec.recurrent_state_bytes,
        "attention_heads": spec.attention_heads,
        "kv_heads": spec.kv_heads,
        "head_dim": spec.head_dim,
        "context_length": spec.context_length,
        "slots": spec.slots,
    }


def _budget_view(required: Dict[str, int]) -> Dict[str, Any]:
    return {
        "weight_bytes": required["weight_bytes"],
        "kv_cache_bytes": required["kv_cache_bytes"],
        "overhead_bytes": required["overhead_bytes"],
        "margin_bytes": required["margin_bytes"],
        "required_bytes": required["required_bytes"],
        "required_gb": _gb(required["required_bytes"]),
        "overhead_fraction": WEIGHT_OVERHEAD_FRACTION,
        "kv_cache_formula": KV_CACHE_FORMULA,
        "overhead_formula": OVERHEAD_FORMULA,
    }


def _in_placement(node: Dict[str, Any], placement: Dict[str, Any]) -> bool:
    """Whether this node is one the distributed placement actually chose."""
    return any(
        view.get("node_id") == node.get("node_id")
        for view in placement.get("nodes", [])
    )


def _try_distributed(
    spec: GpuModelSpec, nodes: List[Dict[str, Any]], link: str,
    utilisation: float,
) -> Optional[Dict[str, Any]]:
    """The fewest GPU nodes that hold the model sharded, or ``None``.

    For each N from 2 up, the N nodes with the most free GPU memory are the
    candidates. They hold the model when their **free memory sums to at least
    the whole requirement** AND an even shard of that requirement fits each of
    them (which, since ``weights < required``, means ``weights / N`` fits each
    too - the task's "weights/N fits each node"). The smallest N that satisfies
    both is chosen, because fewer nodes means fewer network hops. The split
    is a pipeline whatever ``link`` says (:data:`SPLIT_PARALLELISM`).

    The requirement is N machines' worth: each runs its own serving process
    and needs its own runtime reserve (`GpuModelSpec.required_bytes`), and
    each must be able to START its share (`startup_pool_bytes`). Each machine
    gives what vLLM is launched with there (`_launch_budget`), as a replica's
    does: a split sized against the whole pool was admitted where the launch
    fraction of it could not hold its share.
    """
    ordered = sorted(
        nodes, key=lambda node: _launch_budget(node, utilisation), reverse=True,
    )
    for n in range(2, len(ordered) + 1):
        chosen = ordered[:n]
        required = spec.required_bytes(stages=n)
        total_required = required["required_bytes"]
        usable_values = [_launch_budget(node, utilisation) for node in chosen]
        total_usable = sum(usable_values)
        smallest_usable = min(usable_values)
        # Even split of the whole budget across the N nodes. `weights / N` is
        # strictly smaller, so a shard that fits also fits the weights alone.
        shard_bytes = -(-total_required // n)  # ceil division
        weight_shard_bytes = -(-required["weight_bytes"] // n)
        starts = all(
            _node_addressable(node) >= spec.startup_pool_bytes(stages=n)
            for node in chosen
        )
        if starts and total_usable >= total_required and smallest_usable >= shard_bytes:
            return {
                "node_count": n,
                "required_bytes": total_required,
                "parallelism": SPLIT_PARALLELISM,
                "parallelism_reason": SPLIT_PARALLELISM_REASON,
                "link": normalize_quantization(link) or "cross-node",
                "shard_bytes": shard_bytes,
                "shard_gb": _gb(shard_bytes),
                "weight_shard_bytes": weight_shard_bytes,
                # The same three figures the cluster view carries, with the
                # same meanings (see `_decide_gpu_fit`): ``free`` is what the
                # nodes have now, ``usable`` is what the switch will make it.
                "total_free_gpu_bytes": sum(_node_free(node) for node in chosen),
                "total_usable_gpu_bytes": sum(_node_usable(node) for node in chosen),
                "nodes": [_node_view(node) for node in chosen],
            }
    return None


def plan_gpu_fit(
    spec: GpuModelSpec,
    gpu_nodes: List[Dict[str, Any]],
    *,
    link: str = "cross-node",
    intent: str = INTENT_CAPACITY,
    gpu_memory_utilization: Any = None,
    max_model_len: Any = None,
) -> Dict[str, Any]:
    """Decide whether and how ``spec`` runs on ``gpu_nodes``. Pure and testable.

    ``gpu_nodes`` is a list of ``{node_id, name, free_gpu_bytes,
    reclaimable_gpu_bytes, addressable_bytes, gtt_total_bytes, system_ram_bytes,
    ...}`` dicts - the shape `gpu_nodes_from_ledger` produces from a Phase-1
    capacity ledger. No SSH, no Docker: the whole verdict is arithmetic over
    these dicts and the spec, which is what lets every branch be unit-tested
    directly.

    ``intent`` is what clustering is being asked for (:data:`INTENT_CAPACITY` or
    :data:`INTENT_THROUGHPUT`); see the module docstring for why the two cannot
    share one verdict. ``gpu_memory_utilization`` is the fraction the deploy
    will launch vLLM at - ``None`` for the intent's default - and the verdict
    echoes it, because under the throughput intent it is what a replica's fit
    is decided on (VD-129). ``max_model_len`` is the context the deploy will
    launch at - ``None`` for the model's own - and the verdict echoes that too
    (`validate_max_model_len`): the KV cache is sized at the launch context,
    whichever intent asked, so a typed context previews as it launches.

    The one thing this wrapper adds to :func:`_decide_gpu_fit` is the caveat for
    a fleet whose reclaimable memory could not be READ. Appending it here rather
    than inside each branch is the point: there are many return paths and the
    caveat is true of all of them, so a branch that forgot it would be the
    silent "will not fit" this exists to prevent.
    """
    decision = _decide_gpu_fit(
        launch_spec(spec, max_model_len), gpu_nodes, link=link, intent=intent,
        gpu_memory_utilization=gpu_memory_utilization,
    )
    note = _unreadable_note(gpu_nodes)
    if note:
        decision["summary"] = str(decision.get("summary", "")) + note
    return decision


def _decide_gpu_fit(
    spec: GpuModelSpec,
    gpu_nodes: List[Dict[str, Any]],
    *,
    link: str = "cross-node",
    intent: str = INTENT_CAPACITY,
    gpu_memory_utilization: Any = None,
) -> Dict[str, Any]:
    """The verdict itself; see :func:`plan_gpu_fit` for the contract."""
    required = spec.required_bytes()
    model_view = _model_view(spec, required)
    budget_view = _budget_view(required)
    asked = str(intent or INTENT_CAPACITY)
    utilisation = validate_gpu_memory_utilization(
        gpu_memory_utilization, intent=asked
    )
    # Three totals with the same meanings as the per-node view's three figures
    # (LESSONS pattern 5): ``free`` is the sum of what each node has free right
    # now, ``reclaimable`` what the mode switch would give back, and ``usable``
    # - free plus reclaimable - the figure every fit decision below is made on.
    # The first cut filed the usable sum under ``total_free``, so a controller
    # holding a 22 GiB model read as having 22 GiB more free than its own row.
    total_free = sum(_node_free(node) for node in gpu_nodes)
    reclaimable = sum(_node_reclaimable(node) for node in gpu_nodes)
    total_usable = sum(_node_usable(node) for node in gpu_nodes)
    cluster_view = {
        "gpu_node_count": len(gpu_nodes),
        "total_free_gpu_bytes": total_free,
        "total_free_gpu_gb": _gb(total_free),
        "total_reclaimable_gpu_bytes": reclaimable,
        "total_reclaimable_gpu_gb": _gb(reclaimable),
        "total_usable_gpu_bytes": total_usable,
        "total_usable_gpu_gb": _gb(total_usable),
        "nodes": [_node_view(node) for node in gpu_nodes],
    }
    base = {
        "model": model_view,
        "budget": budget_view,
        "cluster": cluster_view,
        "intent": asked,
        # Echoed on every verdict so the form's fit key and the deploy carry the
        # same numbers, whichever intent asked (VD-129: preview equals deploy).
        # The context is the spec's because `plan_gpu_fit` sized the spec at
        # the launch context before handing it here; the deploy launches at
        # what this echoes rather than deriving a number of its own.
        "gpu_memory_utilization": utilisation,
        "max_model_len": int(spec.context_length),
    }

    if not gpu_nodes:
        deficit = required["required_bytes"]
        return {
            **base,
            "verdict": VERDICT_WONT_FIT,
            "deficit_bytes": deficit,
            "deficit_gb": _gb(deficit),
            "options": wont_fit_options(
                spec, gpu_nodes, required, {OPTION_SMALLER_MODEL, OPTION_ADD_MACHINE},
            ),
            "summary": (
                "No GPU-capable nodes are enrolled, so this model cannot run on "
                "GPU. It needs about {} GiB. Enroll a GPU node, or serve it on "
                "CPU."
            ).format(budget_view["required_gb"]),
        }

    if asked == INTENT_THROUGHPUT:
        return {**base, **_decide_replicated(spec, gpu_nodes, required, utilisation)}

    # 1. Single node: the largest free GPU that holds the whole thing, so the
    #    verdict leaves the most headroom rather than filling a node to its rim.
    #    "Holds" is at the fraction vLLM is launched with, as for a replica:
    #    a machine with 44 GiB free does not hold a 43 GiB need at 0.90 of it,
    #    and calling that a one-machine model sent a context only the split
    #    can carry (the 30B's 262,144) to AI Chat instead.
    fitting = [
        node for node in gpu_nodes
        if _launch_budget(node, utilisation) >= required["required_bytes"]
        and _node_addressable(node) >= spec.startup_pool_bytes()
    ]
    if fitting:
        # D6, and it does not depend on how many machines were ticked. The
        # model fits one of them, and clustering is the wrong tool for that:
        # Mode A serves a one-box model on llama.cpp and **vLLM is never
        # offered for a single node** (VD-125/VD-127), so the answer is the same
        # sentence whether the operator selected one machine or five. The first
        # cut refused only at two or more, which made a one-node selection the
        # one way to reach a vLLM deploy of a model AI Chat should have served -
        # a preview that said "fits" and a deploy that agreed, both wrong.
        return {**base, **_single_refused(max(fitting, key=_node_usable), required)}

    # 2. Distributed: the fewest GPU nodes whose free memory holds it sharded.
    # 1b. A model that would fit one machine but cannot START in its pool
    #     (a hybrid model's start-up, `startup_pool_bytes`) is refused as a
    #     replica would be - with the pool it needs - never split to dodge
    #     the rule: a split's own start-up need was measured nowhere.
    held_back = [
        node for node in gpu_nodes
        if _launch_budget(node, utilisation) >= required["required_bytes"]
        and _node_addressable(node) < spec.startup_pool_bytes()
    ]
    if held_back:
        worst = max(held_back, key=_node_addressable)
        return {**base, **_startup_refusal(
            spec, worst, _launch_budget(worst, utilisation), required,
            utilisation, gpu_nodes,
        )}

    distributed = _try_distributed(spec, gpu_nodes, link, utilisation)
    if distributed is not None:
        return {
            **base,
            "budget": _budget_view(
                spec.required_bytes(stages=distributed["node_count"])
            ),
            "verdict": VERDICT_DISTRIBUTED,
            "placement": distributed,
            "summary": (
                "Too big for one machine, so it is split across {} machines, "
                "each holding a share of its layers. {}{}"
            ).format(
                distributed["node_count"], SPLIT_IS_CAPACITY,
                _reclaim_note(
                    [node for node in gpu_nodes if _in_placement(node, distributed)]
                ),
            ),
        }

    # 3. Won't fit: name the deficit and the concrete ways out. Sized for a
    #    split across every selected machine - one reserve for one machine.
    required = spec.required_bytes(stages=max(1, len(gpu_nodes)))
    total_usable = sum(_launch_budget(node, utilisation) for node in gpu_nodes)
    budget_view = _budget_view(required)
    base["budget"] = budget_view
    total_required = required["required_bytes"]
    # A bigger pool is a way out only if the engine itself would then stop
    # refusing: asked again of the same nodes with every pool at the ceiling
    # its machine's RAM allows (which leaves nothing to raise, so it asks
    # once). A pool already at its ceiling, or short for some other reason,
    # is not named - the rule a replica's refusal follows.
    raisable = [node for node in gpu_nodes if pool_headroom(node) > 0]
    pool_helps = bool(raisable) and _decide_gpu_fit(
        spec, with_raised_pools(gpu_nodes), link=link, intent=asked,
        gpu_memory_utilization=utilisation,
    )["verdict"] != VERDICT_WONT_FIT
    ways = {OPTION_SMALLER_MODEL, OPTION_ADD_MACHINE} | (
        {OPTION_RAISE_POOL} if pool_helps else set()
    )
    if total_usable < total_required:
        # Not enough GPU memory in the whole cluster. The figure is the usable
        # one; when any of it is reclaimed, `_reclaim_note` below says which
        # node contributes what, so the sum can be checked by the reader.
        deficit = total_required - total_usable
        deficit_reason = (
            "the cluster has {} GiB free across {} GPU node(s) at {:.0%} GPU "
            "memory use and the model needs about {} GiB"
        ).format(
            _gb(total_usable), len(gpu_nodes), utilisation,
            budget_view["required_gb"],
        )
    else:
        # Enough in total, but the nodes are too unequal for an even shard: the
        # smallest node cannot hold one N-th of the requirement, and identical
        # shards are what the split places. The deficit is what that weakest
        # node lacks.
        smallest_usable = min(_launch_budget(node, utilisation) for node in gpu_nodes)
        shard = -(-total_required // len(gpu_nodes))
        deficit = max(0, shard - smallest_usable)
        deficit_reason = (
            "the cluster's {} GiB total would cover it, but an even shard across "
            "{} nodes is {} GiB and the smallest node holds only {} GiB, so no "
            "even shard fits"
        ).format(
            _gb(total_usable), len(gpu_nodes), _gb(shard), _gb(smallest_usable)
        )
    return {
        **base,
        "verdict": VERDICT_WONT_FIT,
        "deficit_bytes": deficit,
        "deficit_gb": _gb(deficit),
        "options": wont_fit_options(spec, gpu_nodes, required, ways),
        "summary": (
            "Will not fit: {}. Short by about {} GiB.{} Choose a smaller model "
            "or a shorter Max context, or add a machine.{}"
        ).format(
            deficit_reason, _gb(deficit),
            SPLIT_POOL_WAY_OUT.format(nodes=" and ".join(
                str(node.get("name") or node.get("node_id")) for node in raisable
            )) if pool_helps else "",
            _reclaim_note(gpu_nodes),
        ),
    }


def _single_refused(chosen: Dict[str, Any], required: Dict[str, int]) -> Dict[str, Any]:
    """The ``single_refused`` verdict for the one machine that holds the model."""
    headroom = _node_usable(chosen) - required["required_bytes"]
    name = chosen.get("name") or chosen.get("node_id")
    return {
        "verdict": VERDICT_SINGLE_REFUSED,
        "placement": {
            **_node_view(chosen),
            "headroom_bytes": headroom,
            "headroom_gb": _gb(headroom),
        },
        "summary": SINGLE_REFUSED_SUMMARY.format(name),
    }


def _replica_requests(spec: GpuModelSpec, budget: int) -> int:
    """How many full-context requests one replica's budget holds (VD-129).

    Solved from the same arithmetic `GpuModelSpec.required_bytes` uses: the
    budget less the weights and the overhead on them, shared among requests'
    worth of KV cache. A node that fits the spec holds at least its ``slots``
    requests by construction.
    """
    per_request = spec.kv_cache_bytes() // max(1, int(spec.slots))
    if per_request <= 0:
        return 0
    fixed = spec.required_bytes()
    room = budget - fixed["weight_bytes"] - fixed["overhead_bytes"]
    return max(0, int(room // per_request))


def _startup_refusal(
    spec: GpuModelSpec, worst: Dict[str, Any], budget: int,
    required: Dict[str, int], utilisation: float, gpu_nodes: List[Dict[str, Any]],
) -> Dict[str, Any]:
    """The refusal of a model that cannot START in ``worst``'s pool, whatever its context.

    One sentence for a copy on each machine and for the capacity intent
    alike: the pool it needs, and either that the pool can be raised to it
    (the machine's RAM allows it) or that it cannot. No context is offered,
    because none would help.
    """
    ways = replica_ways_out(spec, worst, budget, required["required_bytes"], utilisation)
    startup = spec.startup_pool_bytes()
    deficit = max(
        required["required_bytes"] - budget, startup - _node_addressable(worst), 0,
    )
    kinds = ways["kinds"] - {OPTION_SHORTER_CONTEXT}
    return {
        "verdict": VERDICT_WONT_FIT,
        "deficit_bytes": deficit,
        "deficit_gb": _gb(deficit),
        "max_context_that_fits": 0,
        "options": wont_fit_options(spec, gpu_nodes, required, kinds | {OPTION_SMALLER_MODEL}),
        "summary": STARTUP_WONT_FIT_SUMMARY.format(
            needed=_gb(startup), weights=_gb(required["weight_bytes"]),
            node=worst.get("name") or worst.get("node_id"),
            pool=_gb(_node_addressable(worst)),
        ) + (
            STARTUP_RAISE_POOL if OPTION_RAISE_POOL in kinds else STARTUP_NO_BIGGER_POOL
        ),
    }


def _decide_replicated(
    spec: GpuModelSpec,
    gpu_nodes: List[Dict[str, Any]],
    required: Dict[str, int],
    utilisation: float,
) -> Dict[str, Any]:
    """The throughput intent's verdict (VD-129): one whole copy per machine.

    In the order a reader would want the answer: the arithmetic first (a
    machine that cannot hold the model at the launch fraction is a refusal
    naming that machine and the fraction, with the sharding intent offered as
    the way out); then the two policy refusals - one machine is no replication
    (D6's sentence, so the console shows one sentence for one machine whatever
    the intent), and a set without this controller has nowhere to run the
    endpoint and its balancer; then the placement, with the per-machine request
    room the KV budget allows, minimum over the set because the balancer holds
    the budget only as well as its weakest replica.
    """
    budgets = {
        str(node.get("node_id", "")): _launch_budget(node, utilisation)
        for node in gpu_nodes
    }
    startup = spec.startup_pool_bytes()
    short = [
        node for node in gpu_nodes
        if budgets[str(node.get("node_id", ""))] < required["required_bytes"]
        or _node_addressable(node) < startup
    ]
    if short:
        worst = min(short, key=lambda node: budgets[str(node.get("node_id", ""))])
        budget = budgets[str(worst.get("node_id", ""))]
        deficit = max(0, required["required_bytes"] - budget)
        ways = replica_ways_out(
            spec, worst, budget, required["required_bytes"], utilisation,
        )
        if _node_addressable(worst) < startup:
            # The model cannot START in this pool, whatever its context.
            return _startup_refusal(
                spec, worst, budget, required, utilisation, gpu_nodes,
            )
        summary = REPLICATED_WONT_FIT_SUMMARY.format(
            required=_gb(required["required_bytes"]),
            node=worst.get("name") or worst.get("node_id"),
            usable=_gb(budget), utilisation=utilisation,
        ) + ways["note"]
        return {
            "verdict": VERDICT_WONT_FIT,
            "deficit_bytes": deficit,
            "deficit_gb": _gb(deficit),
            "max_context_that_fits": ways["max_context_that_fits"],
            # Splitting is the way out, not another node: `add_node` is the
            # capacity intent's option and would be a wrong door here. A
            # shorter context and a bigger pool are offered only where they
            # would work (`replica_ways_out`); a smaller model always is.
            "options": wont_fit_options(
                spec, gpu_nodes, required, ways["kinds"] | {OPTION_SMALLER_MODEL},
            ),
            "summary": summary,
        }
    if len(gpu_nodes) == 1:
        return _single_refused(gpu_nodes[0], required)
    if not any(
        str(node.get("node_id", "")) == CONTROLLER_PLACEMENT_ID
        for node in gpu_nodes
    ):
        return {
            "verdict": VERDICT_UNSUPPORTED_INTENT,
            "summary": REPLICATED_CONTROLLER_REQUIRED_SUMMARY,
        }
    requests = {
        node_id: _replica_requests(spec, budget)
        for node_id, budget in budgets.items()
    }
    count = len(gpu_nodes)
    room = min(requests.values())
    return {
        "verdict": VERDICT_REPLICATED,
        "placement": {
            "node_count": count,
            "replica_count": count,
            "gpu_memory_utilization": utilisation,
            "requests_per_machine": room,
            "nodes": [
                {
                    **_node_view(node),
                    "launch_budget_bytes": budgets[str(node.get("node_id", ""))],
                    "launch_budget_gb": _gb(budgets[str(node.get("node_id", ""))]),
                    "requests": requests[str(node.get("node_id", ""))],
                }
                for node in gpu_nodes
            ],
        },
        "summary": replicated_summary(count, room) + _reclaim_note(gpu_nodes),
    }


def fit_decision(
    spec: Dict[str, Any],
    ledger: Dict[str, Any],
    *,
    link: str = "cross-node",
    node_ids: Optional[List[str]] = None,
    intent: str = INTENT_CAPACITY,
    gpu_memory_utilization: Any = None,
    max_model_len: Any = None,
    mtp_tokens: Any = None,
    vllm_image: Any = None,
    built: Optional[GpuModelSpec] = None,
) -> Dict[str, Any]:
    """End-to-end: validate a spec body, read the ledger's GPU nodes, decide.

    The one call the route makes. `build_model_spec` raises ``ValueError`` on a
    bad body (the route turns that into a 400), and `gpu_nodes_from_ledger`
    drops honest GPU absences, so `plan_gpu_fit` only ever sees real free memory.

    ``node_ids`` scopes the decision to exactly the selected workers (VD-B3b-1):
    the deploy path sizes over the nodes the operator picked, so the fit *preview*
    must size over the same set or it can promise a green "fits" the deploy then
    refuses. A non-empty ``node_ids`` filters the GPU nodes to that set before
    planning; ``None`` or an empty list keeps the whole-fleet behaviour the
    ``GET /cluster/capacity`` siblings assume, so existing callers are unchanged.
    Ids that name no GPU node are simply absent from the filtered set rather than
    erroring, so a stale selection degrades to the nodes that do exist.
    ``gpu_memory_utilization`` and ``max_model_len`` are the launch fraction
    and the launch context the form will deploy with (VD-129); a bad value of
    either is a ``ValueError`` here, before any sizing. ``mtp_tokens`` and
    ``vllm_image`` are the deploy's multi-token-prediction and image choices:
    with the selected machines' GPUs they decide the cache the launch keeps,
    sized by the call the deploy makes (`with_launch_cache`). ``built`` is
    ``spec`` already validated by a caller that needed its facts too, so the
    body is not read and built twice.
    """
    gpu_nodes = gpu_nodes_from_ledger(ledger)
    if node_ids:
        selected = {str(node_id) for node_id in node_ids}
        gpu_nodes = [node for node in gpu_nodes if node["node_id"] in selected]
    model_spec = with_launch_cache(
        built or build_model_spec(spec), mtp_tokens, vllm_image=vllm_image,
        gpu_targets=[node["gfx_target_version"] for node in gpu_nodes],
    )
    return plan_gpu_fit(
        model_spec, gpu_nodes, link=str(link or "cross-node"),
        intent=str(intent or INTENT_CAPACITY),
        gpu_memory_utilization=gpu_memory_utilization,
        max_model_len=max_model_len,
    )
