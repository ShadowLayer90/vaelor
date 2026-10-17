"""What a Load gives a deployment made before this release's measured settings (VD-217).

A Load re-serves a stored record. A record made before the draft model
(VD-213), the patched attention file (VD-215) or Vaelor's expert kernels
(VD-216) kept serving without them, so a 1.5.1 deployment of the 30B kept
1.5.1's speed until its owner removed and redeployed it. Here a Load gives it
what a fresh deploy of the same model gets on the same machines - decided by
`vllm_serve_options._deploy_choices`, the deploy's own decision, never a copy
(LESSONS 6) - where it was measured, never over an owner's choice and never
past the fit (`gpu_pool_refit`).

**A declined draft model** (R96). When the draft model would not fit, or
could not be fetched, the Load serves the release's other settings without it
and what needs it, and the record says so (:data:`DRAFTER_DECLINED_FIELD`), so
the upgrade's refresh check does not count it again at every install; an
owner's Load offers it again. :func:`held_back` says exactly what was held
back and why (R97, LESSONS 10). :func:`load_changes` is what the refresh check
counts (kb G71).

Housed out of `vllm_serve_options` at the 1,000-line ceiling (`CLAUDE.md`).
"""

from __future__ import annotations

from dataclasses import replace
from typing import Any, Iterable, List, Mapping, Optional

from .vllm_guesser_build import GUESSER_FIELD
from .vllm_images import image_profile
from .vllm_moe_kernels import MOE_KERNEL_FIELD, decide_moe_kernel
from .vllm_overlays import OVERLAY_FIELD, decide_overlay
from .vllm_serve_advanced import (
    ADVANCED_FIELDS, BATCHED_TOKENS_FIELD, DRAFTER_FIELD, FUSION_FIELD, MAX_NUM_SEQS_FIELD,
    NEW_RUNNER_FIELD, SPEEDUPS_MEASURED, measured_gpus,
)
from .vllm_serve_options import ServeOptions, _deploy_choices, options_from_record


def release_defaults_apply(
    options: ServeOptions, repo: Any, *, replicated: bool, gpu_targets: Iterable[Any],
) -> bool:
    """Whether a Load may give a record what this release measured (VD-217).

    Only where it was measured: the pinned 0.27 image and the AWQ 30B
    (`vllm_serve_advanced.SPEEDUPS_MEASURED`), one copy per machine, no
    multi-token prediction, and EVERY machine reading gfx1151. A machine that
    reads as another GPU, or cannot be read, gets nothing added (LESSONS 8).
    """
    return (replicated and not options.mtp_tokens
            and (options.vllm_image, str(repo or "").strip()) in SPEEDUPS_MEASURED
            and measured_gpus(list(gpu_targets or ())))


def with_release_defaults(
    options: ServeOptions, stored: Any, repo: Any, *, replicated: bool,
    gpu_targets: Iterable[Any], drafter: Optional[bool] = True,
    declined: Optional[str] = None,
) -> ServeOptions:
    """``options`` (a record's) with what a fresh deploy of it gets here now (VD-217).

    A record made before the draft model (VD-213), the attention file
    (VD-215) or Vaelor's expert kernel (VD-216) serves without them at a Load,
    so a 1.5.1 deployment of the 30B kept 1.5.1's speed until its owner
    removed and redeployed it. Here a Load gives it what :func:`_deploy_choices`
    gives an untouched deploy form, where :func:`release_defaults_apply` says
    it was measured. ``stored`` is the record's own ``vllm_options`` mapping.

    **An owner's choice is never overridden.** The Advanced settings are
    written only when set, and on the measured model one copy per machine a
    deploy always sets the request cap (the catalog's default when the owner
    leaves it), so a record that carries ANY Advanced field was made by a
    release that offered them: its missing draft model was turned off, and is
    left off. Only a record that carries none - made before they were offered
    - gains the defaults. The attention file and the expert kernel are no
    owner's choice (no form sets either); each is decided again, the way a
    deploy decides it, only when the record does not carry it.

    **A declined draft model** (R96). ``drafter=False`` gives the defaults
    without the draft model and what works only with it (its attention file
    and, later, its 4-bit guesser) - what a Load serves when the drafter would
    not fit or could not be fetched; the rest of the release's settings are
    kept. ``declined`` is the draft model key such a Load recorded
    (:data:`DRAFTER_DECLINED_FIELD`): the record's Advanced settings were
    written by it, not by an owner, so they are still this release's defaults.
    ``drafter=None`` offers the draft model unless it is the one declined - a
    release with another draft model offers that one (R103) - ``True`` offers
    it anyway (an owner's Load re-checks), ``False`` never.
    """
    if not release_defaults_apply(options, repo, replicated=replicated, gpu_targets=gpu_targets):
        return options
    stored = stored if isinstance(stored, Mapping) else {}
    image = image_profile(options.vllm_image)
    repo_text = str(repo or "").strip()
    if any(field in stored for field in ADVANCED_FIELDS) and not declined:
        # Offered when it was made: the owner's Advanced settings stand.
        advanced = options.advanced
        overlay = decide_overlay(image, advanced[2], True)
        kernel = decide_moe_kernel(image, repo_text, True)
    else:
        advanced, overlay, kernel = _deploy_choices({}, repo_text, image, 0, True)
        if drafter is False or (drafter is None and declined is not None
                                and declined == advanced[2]):
            advanced, overlay, kernel = _deploy_choices(
                {DRAFTER_FIELD: None}, repo_text, image, 0, True)
    return replace(
        options, advanced=advanced,
        attention_overlay=options.attention_overlay if OVERLAY_FIELD in stored else overlay,
        moe_kernel=options.moe_kernel if MOE_KERNEL_FIELD in stored else kernel,
    )


#: The record key (in ``units``, beside ``vllm_options``, so no bridge reads
#: it) that says a Load served this release's defaults WITHOUT the draft model
#: because it would not fit or could not be fetched (R96). Its presence keeps
#: the upgrade's refresh check from counting the drafter again at every
#: install; an owner's Load offers it again, so the decline is re-checked.
DRAFTER_DECLINED_FIELD = "release_drafter_declined"

#: What a Load says when it served a record made before this release's draft
#: model without it (VD-217, R97): exactly what was held back, and why.
DRAFTER_HELD_BACK = "Loaded without {held}: {why} Load it again to try again."
#: The ``why`` of :data:`DRAFTER_HELD_BACK`, one per cause.
RELEASE_DEFAULTS_NOT_FIT = "with the draft model the model would not fit these machines ({reason})."
RELEASE_DEFAULTS_NOT_CHECKED = (
    "whether the draft model fits these machines could not be checked, because {reason}."
)
DRAFTER_NOT_FETCHED = "the draft model could not be fetched ({reason})."

#: How :func:`held_back` names each field a declined drafter takes with it.
_HELD_NAMES = (
    (DRAFTER_FIELD, "this release's draft model"),
    (OVERLAY_FIELD, "the patched attention file the draft model needs"),
    (GUESSER_FIELD, "the 4-bit guesser built from the draft model"),
    (MAX_NUM_SEQS_FIELD, "the request cap"), (BATCHED_TOKENS_FIELD, "the batch budget"),
    (NEW_RUNNER_FIELD, "the newer runner"), (MOE_KERNEL_FIELD, "Vaelor's expert kernels"),
    (FUSION_FIELD, "the fused qk-norm/RoPE step"),
)


#: What a Load says when a record made before this release's measured
#: settings does not gain them because a machine is not the GPU they were
#: measured on (VD-217; R86's pattern, naming every setting it holds back).
RELEASE_DEFAULTS_STAND_ASIDE = (
    "This deployment was made before this release's measured settings for "
    "this model (the draft model, the patched attention file, the newer "
    "runner, Vaelor's expert kernels, the fused qk-norm/RoPE step and the "
    "4-bit guesser), and they are left off: they were measured only on gfx1151 "
    "GPUs (the Radeon 8060S), and {names} {verb} not report one."
)


def release_defaults_note(gpu_nodes: Iterable[Any]) -> str:
    """:data:`RELEASE_DEFAULTS_STAND_ASIDE` naming the machines that did not read gfx1151, or ``""``."""
    names = [
        str(node.get("name") or node.get("node_id") or "a machine")
        for node in gpu_nodes or () if isinstance(node, Mapping)
        and not measured_gpus([node.get("gfx_target_version")])
    ]
    if not names:
        return ""
    return RELEASE_DEFAULTS_STAND_ASIDE.format(
        names=" and ".join(names), verb="did" if len(names) == 1 else "do")


def declined_drafter(proposed: ServeOptions, served: ServeOptions) -> Optional[str]:
    """The draft model ``proposed`` offered and ``served`` does not carry, or ``None``.

    What a Load writes under :data:`DRAFTER_DECLINED_FIELD` (R96).
    """
    return proposed.advanced[2] if served.advanced[2] is None else None


def held_back(proposed: ServeOptions, served: ServeOptions, why: str) -> str:
    """:data:`DRAFTER_HELD_BACK` naming every field ``proposed`` has and ``served`` lacks."""
    wanted, got = proposed.as_record(), served.as_record()
    names = [name for field, name in _HELD_NAMES
             if field in wanted and wanted.get(field) != got.get(field)]
    held = ", ".join(names[:-1]) + " and " + names[-1] if len(names) > 1 else (
        names[0] if names else _HELD_NAMES[0][1])
    return DRAFTER_HELD_BACK.format(held=held, why=why)


def load_changes(units: Mapping[str, Any], repo: Any, options: ServeOptions,
                 *, replicated: bool) -> List[str]:
    """The ``vllm_options`` fields a Load serving ``options`` changes on this record (VD-217).

    The upgrade's refresh check counts these as well as the unit text (kb
    G71: it compared only the text, so an install that changed only what a
    Load adds skipped the reload). Compared as the record would store them,
    so a field is counted exactly when the record would change.
    """
    before = options_from_record(units, repo, replicated=replicated).as_record()
    after = options.as_record()
    return sorted(field for field in set(before) | set(after)
                  if before.get(field) != after.get(field))
