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
from .cluster_gpu_model_spec import with_decided_drafter
from .cluster_gpu_ways_out import startup_pool_refusal
from .cluster_gpu_sizing import (
    INTENT_CAPACITY, INTENT_THROUGHPUT, VERDICT_DISTRIBUTED, VERDICT_REPLICATED,
    build_model_spec, plan_gpu_fit, with_launch_cache,
)
from .cluster_placement import CONTROLLER_PLACEMENT_ID
from .gpu_serving_target import switch_gpu_chat_reclaimable
from .vllm_model_profile import facts_from_hf_config, profile_for
from .vllm_guesser_build import GUESSER_FIELD
from .vllm_images import image_profile
from .vllm_serve_advanced import (
    ADVANCED_FIELDS, FUSION_FIELD, SPEEDUPS_MEASURED, speedups_note,
)
from .vllm_release_defaults import (
    RELEASE_DEFAULTS_NOT_CHECKED, RELEASE_DEFAULTS_NOT_FIT, held_back, release_defaults_note,
)
from .vllm_serve_options import (
    OPTIONS_FIELD, CONFIG_NEVER_READ, CONFIG_READ_FIELD, IMAGE_FIELD, LOAD_KEPT_AS_DEPLOYED,
    LOAD_NOT_CHECKED, MTP_FIELD, ServeOptions, names_hybrid_family, options_draft_bytes,
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
    fetched = with_decided_drafter(with_launch_cache(
        fetched, payload.get(MTP_FIELD), vllm_image=payload.get(IMAGE_FIELD),
        gpu_targets=gpu_targets,
    ), repo, payload, gpu_targets)
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


def gpu_targets_of(participants: List[Dict[str, Any]]) -> List[str]:
    """Each machine's ``gfx_target_version``, read as the deploy reads it (`gpu_fit_node`).

    One reading for a Load, its note and the upgrade's refresh check (VD-217,
    LESSONS 6), off the raw field the probe writes - never a derived one.
    """
    return [
        str(((node.get("inventory") or {}).get("gpu") or {}).get("gfx_target_version", "") or "")
        for node in participants
    ]


def options_for_load_checked(
    ops: Any, units: Mapping[str, Any], repo: str, revision: Optional[str], *,
    participants: List[Dict[str, Any]], replicated: bool, fraction: float,
    context: int, drafter: Optional[bool] = True,
) -> Tuple[ServeOptions, bool, str]:
    """:func:`_load_checked`, and the stand-aside sentence when the Load dropped
    VD-216 item 2's speed-ups because a machine now reads as another GPU (R86),
    or gave a record made before this release's measured settings none of
    them for the same reason (VD-217)."""
    options, config_read, note = _load_checked(
        ops, units, repo, revision, participants=participants, replicated=replicated,
        fraction=fraction, context=context, drafter=drafter)
    recorded = (units or {}).get(OPTIONS_FIELD)
    carried = isinstance(recorded, Mapping) and (
        recorded.get(FUSION_FIELD) is not None or recorded.get(GUESSER_FIELD) is not None)
    older = replicated and isinstance(recorded, Mapping) and not any(
        field in recorded for field in ADVANCED_FIELDS) and (
        options.vllm_image, str(repo or "").strip()) in SPEEDUPS_MEASURED
    if (carried or older) and not (options.qk_norm_rope_fusion or options.guesser):
        machines = [
            {"name": node.get("name") or node.get("id"), "gfx_target_version": target}
            for node, target in zip(participants, gpu_targets_of(participants))]
        note = _joined(note, speedups_note(repo, image_profile(options.vllm_image), machines)
                       if carried else release_defaults_note(machines))
    return options, config_read, note


def _fit_refusal(ops: Any, body: Mapping[str, Any], options: ServeOptions, *,
                 participants: List[Dict[str, Any]], replicated: bool, fraction: float,
                 context: int, targets: List[str]) -> str:
    """The fit's own sentence when ``options`` would not place this record, ``""`` when it would."""
    spec = with_launch_cache(
        build_model_spec(body), options.mtp_tokens or None,
        vllm_image=options.vllm_image, gpu_targets=targets,
    )
    nodes = fit_nodes(ops.broker, ops.mode_switch, participants)
    refusal = startup_pool_refusal(spec, nodes, stages=1 if replicated else len(participants))
    if refusal:
        return refusal
    plan = plan_gpu_fit(
        spec, nodes, intent=INTENT_THROUGHPUT if replicated else INTENT_CAPACITY,
        gpu_memory_utilization=fraction, max_model_len=context,
    )
    if plan["verdict"] != (VERDICT_REPLICATED if replicated else VERDICT_DISTRIBUTED):
        return str(plan["summary"])
    return ""


def _load_checked(
    ops: Any, units: Mapping[str, Any], repo: str, revision: Optional[str], *,
    participants: List[Dict[str, Any]], replicated: bool, fraction: float,
    context: int, drafter: Optional[bool] = True,
) -> Tuple[ServeOptions, bool, str]:
    """``(options, config_read, note)`` a Load serves a stored deployment with.

    A record that does not say its config was unread is served as stored. One
    that does, whose file can now be read, gains only the cache settings the
    file decides (`vllm_serve_options.options_for_load`) - and only if the fit
    still places the model at the record's own weight size, context and
    fraction; otherwise the unit is served as deployed, the record keeps
    saying the config is unread, and ``note`` says why in plain words.

    VD-217: a record that gains this release's draft model (made before it,
    `vllm_serve_options.with_release_defaults`) gains it only if the fit,
    charged with the drafter's weights as a deploy charges them, still places
    the model. Otherwise, or when that cannot be checked, it gets the
    release's other settings without the draft model and what needs it (R96),
    and ``note`` names exactly what was held back and why (R97). ``drafter``
    is `vllm_serve_options.options_for_load`'s: an owner's Load offers it
    again (``True``), the upgrade's refresh respects an earlier decline.
    """
    targets = gpu_targets_of(participants)
    unread = (units or {}).get(CONFIG_READ_FIELD) is False
    config = hf_cached_config.cached_config(repo, revision)
    weight = (units or {}).get(WEIGHT_FIELD)
    weighed = not isinstance(weight, bool) and isinstance(weight, int) and weight > 0

    def decided(facts: Any = None, defaults: bool = True,
                offer: Optional[bool] = drafter) -> Tuple[ServeOptions, bool]:
        # With the machines' GPUs, so VD-216 item 2's speed-ups a Load adds
        # (`vllm_serve_options.with_measured_speedups`) are in both values and
        # never read as the config-driven change below.
        return options_for_load(units, repo, replicated=replicated, facts=facts,
                                gpu_targets=targets, release_defaults=defaults,
                                drafter=offer)

    baseline, _ = decided(defaults=False)
    # The draft model the record serves with is charged as the deploy charged
    # it (VD-213, LESSONS 6): the record's own, never one it does not carry.
    load_body = {"hf_config": config, "weight_bytes": weight,
                 "draft_weight_bytes": options_draft_bytes(baseline)}
    if config is not None and weighed:
        # The start-up rule, on the pools as they are now (J5).
        reason = startup_pool_refusal(
            build_model_spec(load_body),
            fit_nodes(ops.broker, ops.mode_switch, participants),
            stages=1 if replicated else len(participants),
        )
        if reason:
            raise ValueError(LOAD_CANNOT_START.format(reason=reason))
    stored, _ = decided()
    offer, note = drafter, ""
    if options_draft_bytes(stored) != options_draft_bytes(baseline):
        if config is None or not weighed:
            refusal = RELEASE_DEFAULTS_NOT_CHECKED.format(reason=(
                "this model's config.json is not in the model library" if config is None
                else "the deployment's record does not say how big its weights are"))
        else:
            fit = _fit_refusal(
                ops, {**load_body, "draft_weight_bytes": options_draft_bytes(stored)},
                stored, participants=participants, replicated=replicated,
                fraction=fraction, context=context, targets=targets)
            refusal = RELEASE_DEFAULTS_NOT_FIT.format(reason=fit) if fit else ""
        if refusal:
            # R96: the release's other settings stay; only the draft model and
            # what needs it are held back, and the record says it was declined.
            offer = False
            proposed, (stored, _) = stored, decided(offer=False)
            note = held_back(proposed, stored, refusal)
    load_body["draft_weight_bytes"] = options_draft_bytes(stored)
    facts = facts_from_hf_config(config) if config is not None and unread else None
    options, config_read = decided(facts, offer=offer)
    if options == stored:
        return stored, config_read, note
    if not weighed:
        return stored, False, _joined(note, LOAD_NOT_CHECKED)
    spec = with_launch_cache(
        build_model_spec(load_body),
        stored.mtp_tokens or None, vllm_image=stored.vllm_image, gpu_targets=targets,
    )
    plan = plan_gpu_fit(
        spec, fit_nodes(ops.broker, ops.mode_switch, participants),
        intent=INTENT_THROUGHPUT if replicated else INTENT_CAPACITY,
        gpu_memory_utilization=fraction, max_model_len=context,
    )
    if plan["verdict"] != (VERDICT_REPLICATED if replicated else VERDICT_DISTRIBUTED):
        return stored, False, _joined(note, LOAD_KEPT_AS_DEPLOYED.format(reason=plan["summary"]))
    return options, True, note


def _joined(*notes: str) -> str:
    return " ".join(note for note in notes if note)
