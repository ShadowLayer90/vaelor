"""What would make a model fit, said only when it would: the fit's ways out.

Housed out of `cluster_gpu_sizing` (at the 1,000-line ceiling `CLAUDE.md`
sets) when the refusals learned to name a way out with its numbers. A
"will not fit" that stops there is a dead end, and one that lists every
conceivable remedy sends its reader to settings that change nothing. So each
way out here is offered under ONE rule, the same for a copy on every machine
and for a model split across them:

* **a shorter context**, when one fits the memory the machine has now;
* **a bigger GPU memory pool**, only when the POOL is what is short - not
  memory something else is holding - and the machine's own RAM allows a pool
  that would do (:func:`achievable_gtt_ceiling`). A pool already at its
  ceiling, or big enough but in use, is never named.

The sentence and the verdict's ``options`` list are built from the same
answer, so the line under a refusal offers exactly what the refusal does.

**The setting has one name here** (:data:`POOL_SETTING_NAME`). The setting
itself is built elsewhere; every sentence that points at it takes its name
from this constant, so there is one place to change it.
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional

from .cluster_gpu_model_spec import GpuModelSpec, context_that_fits
from .model_sizing import GIB, SYSTEM_RESERVE_FRACTION

#: What the machine setting that sizes the GPU's share of memory is called,
#: wherever a fit sentence points an owner at it.
POOL_SETTING_NAME = "GPU memory pool"

#: The ways out a replica's refusal names when they would work, with the
#: numbers. "Max context" is the deploy form's field; the pool is the
#: machine's own setting, which takes a restart.
CONTEXT_WAY_OUT = " It would fit at a Max context of {tokens:,} tokens or fewer."
#: Where the setting is and what changing it costs, said once for every
#: sentence that sends an owner to it.
RAISE_THE_POOL = (
    "raise the " + POOL_SETTING_NAME + " in that machine's settings, which "
    "takes effect after a restart."
)
POOL_WAY_OUT = (
    " {node}'s " + POOL_SETTING_NAME + " is {pool} GiB and this needs a pool "
    "of about {needed} GiB: " + RAISE_THE_POOL
)
#: The same way out for a model split across machines, naming each machine
#: whose pool could be raised.
SPLIT_POOL_WAY_OUT = (
    " A bigger " + POOL_SETTING_NAME + " on {nodes} would hold it: " + RAISE_THE_POOL
)
#: What a refusal says when a model cannot START in a machine's pool, however
#: short its context (`cluster_gpu_model_spec.HYBRID_STARTUP_BYTES`).
STARTUP_WONT_FIT_SUMMARY = (
    "This kind of model needs about {needed} GiB of " + POOL_SETTING_NAME
    + " on a machine while it starts, whatever its context: {weights} GiB of "
    "weights and the room its start-up takes. {node}'s pool is {pool} GiB."
)
STARTUP_RAISE_POOL = (
    " Raise the " + POOL_SETTING_NAME + " in that machine's settings to at "
    "least that; it takes effect after a restart."
)
STARTUP_NO_BIGGER_POOL = (
    " That machine cannot have a pool that big. Choose a smaller model."
)

#: The ``kind`` of each way out in a verdict's ``options``. The pool's keeps
#: the word the wire has always carried for it.
OPTION_SHORTER_CONTEXT = "shorter_context"
OPTION_RAISE_POOL = "raise_gtt_ceiling"
OPTION_SMALLER_MODEL = "smaller_quantization"
OPTION_ADD_MACHINE = "add_node"


def gb(bytes_value: int) -> float:
    """Bytes as GiB, rounded to two places for a legible on-wire figure."""
    return round(int(bytes_value) / GIB, 2)


def _whole(node: Dict[str, Any], field: str) -> int:
    return int(node.get(field, 0) or 0)


def achievable_gtt_ceiling(system_ram_bytes: Any) -> Optional[int]:
    """The largest GPU memory pool a node could set, from its system RAM.

    The pool (GTT) is carved out of system RAM, so it can never legitimately
    exceed what is left after the OS/Docker/control-plane reserve
    (`model_sizing.SYSTEM_RESERVE_FRACTION`). ``None`` when the node never
    reported its RAM, so the option is offered as "refresh the node to size
    this" rather than against a number nobody has.
    """
    if not isinstance(system_ram_bytes, int) or system_ram_bytes <= 0:
        return None
    return int(system_ram_bytes * (1 - SYSTEM_RESERVE_FRACTION))


def pool_headroom(node: Dict[str, Any]) -> int:
    """How much bigger this node's pool could be made; 0 when unknown or none."""
    ceiling = achievable_gtt_ceiling(node.get("system_ram_bytes"))
    if ceiling is None:
        return 0
    return max(0, ceiling - _whole(node, "addressable_bytes"))


def with_raised_pools(nodes: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """``nodes`` as they would be with every pool at its achievable ceiling.

    For asking the fit engine itself whether a bigger pool is a way out: the
    memory a raised pool adds is free memory, and nothing else changes.
    """
    return [
        {
            **node,
            "free_gpu_bytes": _whole(node, "free_gpu_bytes") + pool_headroom(node),
            "addressable_bytes": _whole(node, "addressable_bytes") + pool_headroom(node),
        }
        for node in nodes
    ]


def replica_ways_out(
    spec: GpuModelSpec, worst: Dict[str, Any], budget: int, required: int,
    utilisation: float,
) -> Dict[str, Any]:
    """What would make the weakest machine hold a replica, with the numbers.

    ``budget`` is what the machine can give a replica now and ``required``
    what one needs. The pool is short when the launch fraction of it is under
    the requirement, or when the model cannot start in it at all; it is named
    only then, and only when a pool that would do is one this machine could
    have. A shorter context is offered when one fits the budget and the model
    can start. ``kinds`` is the same answer for the verdict's ``options``.
    """
    name = worst.get("name") or worst.get("node_id")
    pool = _whole(worst, "addressable_bytes")
    startup = spec.startup_pool_bytes()
    starts = pool >= startup
    fitting = context_that_fits(spec, budget) if starts else 0
    note = CONTEXT_WAY_OUT.format(tokens=fitting) if fitting else ""
    kinds = {OPTION_SHORTER_CONTEXT} if fitting else set()
    needed = max(int(required / utilisation) + 1, startup)
    ceiling = achievable_gtt_ceiling(worst.get("system_ram_bytes"))
    pool_short = int(pool * utilisation) < required or not starts
    if pool_short and ceiling is not None and needed <= ceiling:
        note += POOL_WAY_OUT.format(node=name, pool=gb(pool), needed=gb(needed))
        kinds.add(OPTION_RAISE_POOL)
    return {"max_context_that_fits": fitting, "note": note, "kinds": kinds}


def startup_pool_refusal(spec: GpuModelSpec, nodes: List[Dict[str, Any]], stages: int = 1) -> str:
    """The start-up sentence for the first machine whose pool cannot START the model; ``""`` if none.

    The hybrid start-up rule alone (`GpuModelSpec.startup_pool_bytes`), for a
    Load: the model was placed when it was deployed, and a pool shrunk since
    must not have it started where it cannot start. ``stages`` is 1 for a
    copy on each machine and the machine count for a split. A machine whose
    pool was never reported is not judged here.
    """
    startup = spec.startup_pool_bytes(stages)
    for node in nodes:
        pool = _whole(node, "addressable_bytes")
        if startup <= 0 or pool <= 0 or pool >= startup:
            continue
        ceiling = achievable_gtt_ceiling(node.get("system_ram_bytes"))
        return STARTUP_WONT_FIT_SUMMARY.format(
            needed=gb(startup), weights=gb(-(-int(spec.weight_bytes) // max(1, int(stages)))),
            node=node.get("name") or node.get("node_id"), pool=gb(pool),
        ) + (STARTUP_RAISE_POOL if ceiling is not None and startup <= ceiling
             else STARTUP_NO_BIGGER_POOL)
    return ""


def wont_fit_options(
    spec: GpuModelSpec, nodes: List[Dict[str, Any]], required: Dict[str, int],
    kinds: Any,
) -> List[Dict[str, Any]]:
    """The ways out of a wont-fit named in ``kinds``, each with its numbers.

    ``kinds`` is the set the verdict's sentence offers, so a way out that
    would not help here is not in the list either. A shorter context carries
    the context that fits as ``max_context`` when the caller found one.
    """
    per_node = []
    for node in nodes:
        ceiling = achievable_gtt_ceiling(node.get("system_ram_bytes"))
        per_node.append({
            "node_id": node.get("node_id", ""),
            "name": node.get("name", ""),
            "current_addressable_bytes": _whole(node, "addressable_bytes"),
            "current_free_gpu_bytes": _whole(node, "free_gpu_bytes"),
            "reclaimable_gpu_bytes": _whole(node, "reclaimable_gpu_bytes"),
            "system_ram_bytes": node.get("system_ram_bytes"),
            "achievable_gtt_ceiling_bytes": ceiling,
            "achievable_gtt_ceiling_gb": gb(ceiling) if ceiling else None,
            "reason": (
                "The GPU memory pool is taken from the machine's own memory; it "
                "can reach that memory minus a {:.0%} reserve for the system."
                .format(SYSTEM_RESERVE_FRACTION)
                if ceiling is not None else
                "This node has not reported its system RAM; refresh it to size a "
                "raisable ceiling."
            ),
        })
    every = [
        {
            "kind": OPTION_SHORTER_CONTEXT,
            "detail": "Set a shorter Max context: the KV cache shrinks with it.",
        },
        {
            "kind": OPTION_RAISE_POOL,
            "detail": (
                "Raise each machine's GPU memory pool so the GPU can use more "
                "of the machine's memory. The largest pool each machine could "
                "have is listed below."
            ),
            "per_node": per_node,
        },
        {
            "kind": OPTION_SMALLER_MODEL,
            "detail": (
                "A smaller quantization shrinks the weights: the model needs "
                "about {} GiB of weights at {}, and a lighter format lowers that "
                "directly."
            ).format(
                gb(required["weight_bytes"]), spec.quantization or "this format"
            ),
        },
        {
            "kind": OPTION_ADD_MACHINE,
            "detail": (
                "Add a GPU node to the cluster; distributed placement can then "
                "shard the model across more free memory."
            ),
        },
    ]
    return [option for option in every if option["kind"] in kinds]
