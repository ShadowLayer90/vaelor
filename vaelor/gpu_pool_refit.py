"""What a deployment is served with once its model's layout is read: fit again, then serve.

Housed out of `gpu_pool_operations` (at the 1,000-line ceiling `CLAUDE.md`
sets) and read by it and by `gpu_pool_reload`. A cluster deploy can begin
before the model's ``config.json`` is in this controller's model library -
a pasted repo whose geometry was typed by hand - and the file arrives with
the deploy's own weight fetch. The file decides things the typed geometry
could not: whether the model is hybrid (Gated DeltaNet), what its cache
settings are on these GPUs, and so how much memory it takes to start. So:

* **A deploy fits the model again once the fetch has read the file**
  (:func:`refit_after_fetch`), before any unit starts, with the same call the
  first fit made. If it no longer fits, the deploy is refused with the fit's
  own sentence - a pasted 27B on a machine with the kernel's default pool used
  to pass the first fit on its typed geometry and be launched into the
  configuration that ran out of memory. If it now fits somewhere else, the
  deploy is refused too: the machines the owner was shown are the machines it
  serves on, or nothing.
* **A model whose NAME says it is of a hybrid family and whose file can never
  be read is not launched on a guess** (:data:`vllm_serve_options.
  CONFIG_NEVER_READ`): its start-up need is exactly what the guess cannot
  know. A name that says nothing is served as typed, as before.
* **Every Load asks the start-up rule again** (:func:`options_for_load_checked`),
  whether or not its options change: a hybrid model placed on a pool that
  has been shrunk since its deploy is refused with the pool it needs, before
  anything starts, rather than started where it cannot start.
* **A Load that finds the file since** (:func:`options_for_load_checked`)
  adds only what the file newly decides - the cache settings - to the options
  the deployment was made with, keeping every choice its owner made; and it
  serves them only if the fit, asked again with the record's own weight size,
  context and fraction, still places the model. Otherwise it serves the unit
  as deployed and says why.

:func:`fit_nodes` is the one reading of the selected machines both make: the
deploy's own, moved here unchanged, so the Load's fit counts the memory the
cluster mode switch will free exactly as the deploy's did.
"""

from __future__ import annotations

from typing import Any, Dict, List, Mapping, Optional, Tuple

from . import hf_cached_config
from .cluster_capacity import gpu_fit_node
from .cluster_gpu_ways_out import startup_pool_refusal
from .cluster_gpu_sizing import (
    INTENT_CAPACITY, INTENT_THROUGHPUT, VERDICT_DISTRIBUTED, VERDICT_REPLICATED,
    build_model_spec, plan_gpu_fit, with_launch_cache,
)
from .cluster_placement import CONTROLLER_PLACEMENT_ID
from .gpu_serving_target import switch_gpu_chat_reclaimable
from .vllm_model_profile import facts_from_hf_config, profile_for
from .vllm_serve_options import (
    CONFIG_NEVER_READ, CONFIG_READ_FIELD, IMAGE_FIELD, LOAD_KEPT_AS_DEPLOYED,
    LOAD_NOT_CHECKED, MTP_FIELD, ServeOptions, names_hybrid_family,
    options_for, options_for_load,
)

#: The record key for the weight size the deploy's fit was made on, which a
#: Load's fit needs and the record never carried before.
WEIGHT_FIELD = "weight_bytes"

#: What a deploy is refused with when the model's own layout, read after the
#: fetch, would place it on other machines than the ones the owner chose.
LAYOUT_MOVES_IT = (
    "This model's own layout, read once its files were fetched, places it "
    "differently from the preview it was deployed from. Nothing was started. "
    "Preview it again and deploy the placement shown."
)

#: What a Load is refused with when a machine's pool can no longer START the
#: model (`cluster_gpu_ways_out.startup_pool_refusal` gives the reason).
LOAD_CANNOT_START = "The deployment was not loaded, and nothing was started. {reason}"


def fit_nodes(broker: Any, mode_switch: Any, nodes: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """The fit engine's reading of ``nodes``, with the controller's reclaimable memory.

    The controller's fit node counts the memory the mode switch will free
    (D1), so a deploy - and a Load - size against the number the preview
    did. Every other node's held memory is somebody else's and is not
    reclaimable by anything here.
    """
    reclaimable, reclaim_reason = switch_gpu_chat_reclaimable(broker, mode_switch)
    return [
        gpu_fit_node(
            node,
            reclaimable=reclaimable and node["id"] == CONTROLLER_PLACEMENT_ID,
            reclaimable_reason=(
                reclaim_reason if node["id"] == CONTROLLER_PLACEMENT_ID else ""
            ),
        )
        for node in nodes
    ]


def refit_after_fetch(
    ops: Any, payload: Mapping[str, Any], repo: str, revision: Optional[str], *,
    spec: Any, nodes: List[Dict[str, Any]], by_id: Mapping[str, Any],
    participants: List[Dict[str, Any]], intent: str, fraction: float,
    gpu_targets: List[Any], replicated: bool,
) -> Tuple[Any, Optional[Mapping[str, Any]], Optional[Dict[str, Any]]]:
    """``(spec, facts, options)`` a deploy serves with once its fetch has run.

    ``spec`` is what the first fit sized; with facts already known it is kept
    and ``options`` is ``None`` (the first decision stands). Otherwise the
    model library is read again: a file there means the fit is asked again
    with it - refused in its own words if it no longer places the model, or
    places it elsewhere - and the options are decided from it; no file means
    a hybrid-family name is refused (:data:`CONFIG_NEVER_READ`) and any other
    is served as typed.
    """
    if spec.facts is not None:
        return spec, spec.facts, None
    config = hf_cached_config.cached_config(repo, revision)
    if config is None:
        if names_hybrid_family(profile_for(repo, None)):
            raise ValueError(CONFIG_NEVER_READ)
        return spec, None, None
    _repo, _revision, fetched = ops._model_and_spec(payload)
    fetched = with_launch_cache(
        fetched, payload.get(MTP_FIELD), vllm_image=payload.get(IMAGE_FIELD),
        gpu_targets=gpu_targets,
    )
    plan = plan_gpu_fit(
        fetched, nodes, link=str(payload.get("link", "cross-node")),
        intent=intent, gpu_memory_utilization=fraction,
        max_model_len=payload.get("max_model_len"),
    )
    placement = ops._placement(plan, by_id, payload, fetched)
    if sorted(node["id"] for node in placement["nodes"]) != sorted(
        node["id"] for node in participants
    ) or (placement["mode"] == VERDICT_REPLICATED) != replicated:
        raise ValueError(LAYOUT_MOVES_IT)
    facts = fetched.facts
    options = options_for(
        profile_for(repo, facts), payload, replicated=replicated,
        gpu_targets=gpu_targets,
    ).as_record()
    return fetched, facts, options


def options_for_load_checked(
    ops: Any, units: Mapping[str, Any], repo: str, revision: Optional[str], *,
    participants: List[Dict[str, Any]], replicated: bool, fraction: float,
    context: int,
) -> Tuple[ServeOptions, bool, str]:
    """``(options, config_read, note)`` a Load serves a stored deployment with.

    A record that does not say its config was unread is served as stored. One
    that does, whose file can now be read, gains only the cache settings the
    file decides (`vllm_serve_options.options_for_load`) - and only if the fit
    still places the model at the record's own weight size, context and
    fraction; otherwise the unit is served as deployed, the record keeps
    saying the config is unread, and ``note`` says why in plain words.
    """
    targets = [
        str(((node.get("inventory") or {}).get("gpu") or {}).get("gfx_target_version", "") or "")
        for node in participants
    ]
    unread = (units or {}).get(CONFIG_READ_FIELD) is False
    config = hf_cached_config.cached_config(repo, revision)
    weight = (units or {}).get(WEIGHT_FIELD)
    weighed = not isinstance(weight, bool) and isinstance(weight, int) and weight > 0
    if config is not None and weighed:
        # The start-up rule, on the pools as they are now (J5).
        reason = startup_pool_refusal(
            build_model_spec({"hf_config": config, "weight_bytes": weight}),
            fit_nodes(ops.broker, ops.mode_switch, participants),
            stages=1 if replicated else len(participants),
        )
        if reason:
            raise ValueError(LOAD_CANNOT_START.format(reason=reason))
    facts = facts_from_hf_config(config) if config is not None and unread else None
    options, config_read = options_for_load(
        units, repo, replicated=replicated, facts=facts, gpu_targets=targets,
    )
    stored, _ = options_for_load(units, repo, replicated=replicated, facts=None)
    if options == stored:
        return stored, config_read, ""
    if not weighed:
        return stored, False, LOAD_NOT_CHECKED
    spec = with_launch_cache(
        build_model_spec({"hf_config": config, "weight_bytes": weight}),
        stored.mtp_tokens or None, vllm_image=stored.vllm_image, gpu_targets=targets,
    )
    plan = plan_gpu_fit(
        spec, fit_nodes(ops.broker, ops.mode_switch, participants),
        intent=INTENT_THROUGHPUT if replicated else INTENT_CAPACITY,
        gpu_memory_utilization=fraction, max_model_len=context,
    )
    if plan["verdict"] != (VERDICT_REPLICATED if replicated else VERDICT_DISTRIBUTED):
        return stored, False, LOAD_KEPT_AS_DEPLOYED.format(reason=plan["summary"])
    return options, True, ""
