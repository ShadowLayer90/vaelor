"""The deploy form's Advanced serving settings: request cap, batch budget, draft model, newer runner.

VD-213 (owner decision 2026-10-12): the measured settings become the serving
default for the catalog's Qwen3-30B-A3B, and the owner can set them in the
deploy form's Advanced section. Four owner choices, each optional in the
deploy payload:

* ``max_num_seqs`` - the most requests one machine serves at once
  (``--max-num-seqs``; vLLM's own default is 256);
* ``max_num_batched_tokens`` - the tokens one engine step may batch
  (``--max-num-batched-tokens``);
* ``drafter`` - ``null`` (off) or ``{"key", "schedule"}``: a pinned draft model
  (`vllm_drafters`) and its lookahead per load, rendered as vLLM's dotted
  ``--speculative-config.*`` words;
* ``new_runner`` - vLLM 0.27's newer model runner, the container variable
  `vllm_images.V2_RUNNER_ENV`.

**A field the payload leaves out is the model's catalog default** (`gpu_model_
catalog.default_serving_for`, read by repo) on a one-copy-per-machine
deployment - where it was measured - as far as the chosen image can take it;
a model with no defaults, or a split, gets vLLM's own. ``drafter: null`` and
``new_runner: false`` are the owner turning them off.

**One rule, every reader** (LESSONS 6). :func:`check_advanced` is the
combination rule the deploy (:func:`advanced_for`), the renderer on both
sides of the root bridge (`vllm_serve_options.require_options`) and the form's
choices (:func:`advanced_choices`) share; :func:`advanced_words` and
:func:`advanced_environment` are the one place each setting becomes a word.
A refusal says what to change, in plain words (LESSONS 1).
"""

from __future__ import annotations

from typing import Any, Dict, List, Mapping, Optional, Tuple

from .gpu_model_catalog import default_serving_for
from .vllm_drafters import (
    Drafter, Schedule, drafter_profile, drafters_for, fit_schedule,
    max_lookahead, require_schedule, schedule_word, varies,
)
from .vllm_images import V2_RUNNER_ENV, VLLM_027_ROCM10, VllmImage, image_profile

#: The pinned image the item-2 speed-ups were measured on, by key.
VLLM_027_ROCM10_KEY = VLLM_027_ROCM10.key

#: The payload's and the record's names for the four settings, spelled once.
MAX_NUM_SEQS_FIELD = "max_num_seqs"
BATCHED_TOKENS_FIELD = "max_num_batched_tokens"
DRAFTER_FIELD = "drafter"
NEW_RUNNER_FIELD = "new_runner"
ADVANCED_FIELDS = (MAX_NUM_SEQS_FIELD, BATCHED_TOKENS_FIELD, DRAFTER_FIELD, NEW_RUNNER_FIELD)

#: The ranges the deploy accepts and the form offers (the TS form mirrors them).
MAX_NUM_SEQS_RANGE = (1, 64)
BATCHED_TOKENS_RANGE = (1024, 65536)

#: What a draft model or the newer runner on a model split is refused with.
SPLIT_REFUSED = (
    "The draft model and the newer runner are offered on a "
    "one-copy-per-machine deployment only; neither has been verified "
    "across a model split between machines."
)

#: What one deployment's Advanced settings are once decided: ``(max_num_seqs,
#: max_num_batched_tokens, drafter key, schedule, new_runner)``.
Advanced = Tuple[Optional[int], Optional[int], Optional[str], Schedule, bool]
NOTHING: Advanced = (None, None, None, (), False)


def _whole_in(value: Any, bounds: Tuple[int, int], what: str) -> int:
    low, high = bounds
    if isinstance(value, bool) or not isinstance(value, int) or not low <= value <= high:
        raise ValueError("{} must be a whole number from {} to {}.".format(what, low, high))
    return value


def check_advanced(
    max_num_seqs: Any, batched: Any, drafter: Any, schedule: Any, new_runner: Any,
    *, image: VllmImage, mtp_tokens: int,
) -> Advanced:
    """The settings as typed values, or a plain refusal of the first thing wrong.

    Refused: a count out of its range; the newer runner on an image whose
    vLLM was never run with it; a draft model that is not pinned, on an image
    it was not run on, beside multi-token prediction, or without a request
    cap; a lookahead table that does not cover 1 to the cap
    (`vllm_drafters.require_schedule`); a table that changes with load on the
    older runner (VD-213: it forces piecewise GPU graphs there).
    """
    cap = None if max_num_seqs is None else _whole_in(
        max_num_seqs, MAX_NUM_SEQS_RANGE, "Max requests per machine")
    budget = None if batched is None else _whole_in(
        batched, BATCHED_TOKENS_RANGE, "The batch token budget")
    if not isinstance(new_runner, bool):
        raise ValueError("The newer-runner setting must be true or false.")
    if new_runner and not image.v2_runner:
        raise ValueError(
            "The newer vLLM runner is not available in the {} image. Choose "
            "vLLM 0.27, or turn the newer runner off.".format(image.name)
        )
    if drafter is None:
        return cap, budget, None, (), new_runner
    profile = drafter_profile(drafter)
    if image.key not in profile.images:
        raise ValueError(
            "The {} draft model was not run on the {} image. Choose vLLM 0.27, "
            "or turn the draft model off.".format(profile.label, image.name)
        )
    if mtp_tokens:
        raise ValueError(
            "A draft model and multi-token prediction both draft tokens; turn "
            "one of them off."
        )
    table = require_schedule(schedule, cap, max_tokens=profile.max_tokens)
    if varies(table) and not new_runner:
        raise ValueError(
            "A lookahead that changes with load needs the newer vLLM runner: on "
            "the older one it makes the GPU run slower graphs. Turn the newer "
            "runner on, or use one lookahead for every load."
        )
    return cap, budget, profile.key, table, new_runner


def require_target(drafter: Optional[str], repo: Any) -> None:
    """Refuse a draft model for a model it does not draft for."""
    if drafter is not None and str(repo or "").strip() not in drafter_profile(drafter).targets:
        raise ValueError(
            "The {} draft model drafts only for {}; it cannot be used with "
            "this model.".format(
                drafter_profile(drafter).label, " and ".join(drafter_profile(drafter).targets),
            )
        )


def _requested(payload: Mapping[str, Any], field: str) -> Tuple[bool, Any]:
    """``(given, value)``: a field the payload leaves out, or sets to null, is not given."""
    value = payload.get(field)
    return value is not None, value


def advanced_for(
    payload: Mapping[str, Any], *, repo: Any, image: VllmImage, mtp_tokens: int,
    replicated: bool,
) -> Advanced:
    """The Advanced settings a NEW deployment gets: the payload's, else the defaults.

    The catalog defaults (`gpu_model_catalog.default_serving_for`) apply only
    to one copy per machine, where they were measured, and only as far as
    ``image`` takes them: a default drafter the image cannot run is left off,
    and so is the newer runner on an image without it. A default lookahead
    table follows a cap the owner changed (`vllm_drafters.fit_schedule`). A
    draft model or the newer runner the owner asks for on a split is refused,
    as multi-token prediction is: neither was run across machines.
    """
    defaults = default_serving_for(repo) if replicated else {}
    given_cap, cap = _requested(payload, MAX_NUM_SEQS_FIELD)
    if not given_cap:
        cap = defaults.get(MAX_NUM_SEQS_FIELD)
    given_budget, budget = _requested(payload, BATCHED_TOKENS_FIELD)
    if not given_budget:
        budget = defaults.get(BATCHED_TOKENS_FIELD)
    drafter, schedule = None, None
    if DRAFTER_FIELD in payload:
        raw = payload[DRAFTER_FIELD]
        if raw is not None:
            if not isinstance(raw, Mapping) or set(raw) != {"key", "schedule"}:
                raise ValueError(
                    "The draft model choice must name a draft model and its "
                    "lookahead table."
                )
            drafter, schedule = raw["key"], raw["schedule"]
            require_target(drafter, repo)
    elif defaults.get(DRAFTER_FIELD) and not mtp_tokens:
        default = defaults[DRAFTER_FIELD]
        if any(item.key == default["key"] for item in drafters_for(repo, image.key)):
            drafter = default["key"]
            schedule = default["schedule"]
            if isinstance(cap, int) and not isinstance(cap, bool) and cap >= 1:
                schedule = fit_schedule(tuple(tuple(entry) for entry in schedule), cap)
    given_runner, new_runner = _requested(payload, NEW_RUNNER_FIELD)
    if not given_runner:
        new_runner = bool(defaults.get(NEW_RUNNER_FIELD)) and image.v2_runner
    if not replicated and (drafter is not None or new_runner is True):
        raise ValueError(SPLIT_REFUSED)
    return check_advanced(
        cap, budget, drafter, schedule, new_runner, image=image, mtp_tokens=mtp_tokens,
    )


def default_drafter(repo: Any) -> Optional[Drafter]:
    """The draft model a NEW deployment of ``repo`` serves with by default, or ``None``.

    VD-214: the model library's download fetches it beside the model
    (`gpu_model_library`). Asked of :func:`advanced_for` with an empty payload
    on the default image, one copy per machine - exactly what an untouched
    deploy form decides - so the library cannot fetch a drafter the deploy
    would not use, or miss one it would (LESSONS 6).
    """
    key = advanced_for(
        {}, repo=repo, image=image_profile(None), mtp_tokens=0, replicated=True,
    )[2]
    return None if key is None else drafter_profile(key)


def advanced_record(advanced: Advanced) -> Dict[str, Any]:
    """The settings as record fields: ONLY those that are set (an older bridge knows none)."""
    cap, budget, drafter, schedule, new_runner = advanced
    record: Dict[str, Any] = {}
    if cap is not None:
        record[MAX_NUM_SEQS_FIELD] = cap
    if budget is not None:
        record[BATCHED_TOKENS_FIELD] = budget
    if drafter is not None:
        record[DRAFTER_FIELD] = {
            "key": drafter, "schedule": [list(entry) for entry in schedule],
        }
    if new_runner:
        record[NEW_RUNNER_FIELD] = True
    return record


def advanced_from_record(
    value: Mapping[str, Any], *, image: VllmImage, mtp_tokens: int,
) -> Advanced:
    """The settings a stored value carries, checked; absent fields are off.

    A field is written only when it is set (:func:`advanced_record`), so a
    present field holding null, ``false`` or another shape is refused rather
    than read as off.
    """
    if any(field in value and value[field] is None for field in ADVANCED_FIELDS):
        raise ValueError("A stored Advanced serving setting is empty.")
    if NEW_RUNNER_FIELD in value and value[NEW_RUNNER_FIELD] is not True:
        raise ValueError("A stored newer-runner setting must be on when present.")
    drafter, schedule = None, None
    if DRAFTER_FIELD in value:
        raw = value[DRAFTER_FIELD]
        if not isinstance(raw, Mapping) or set(raw) != {"key", "schedule"}:
            raise ValueError("A stored draft model setting is not one this version reads.")
        drafter, schedule = raw["key"], raw["schedule"]
    return check_advanced(
        value.get(MAX_NUM_SEQS_FIELD), value.get(BATCHED_TOKENS_FIELD), drafter,
        schedule, value.get(NEW_RUNNER_FIELD, False), image=image, mtp_tokens=mtp_tokens,
    )


def advanced_words(advanced: Advanced, *, drafter_path: Optional[str] = None) -> List[str]:
    """The ``vllm serve`` words, in one fixed order: cap, budget, then the draft model.

    The draft model is vLLM's dotted ``--speculative-config.*`` form, as
    multi-token prediction is (`vllm_serve_options`), pinned to the drafter's
    revision the way ``--revision`` pins a served model: the pull fetched
    that commit into the mounted hub cache, so the container finds it there.
    ``drafter_path`` is the container folder of the 4-bit guesser built from
    that drafter on this machine (VD-216 work order item 2,
    `vllm_guesser_build`): the model word names it and no revision is given,
    because a folder has none. ``None`` is the pinned bf16 drafter itself.
    """
    cap, budget, drafter, schedule, _new_runner = advanced
    words: List[str] = []
    if cap is not None:
        words += ["--max-num-seqs", str(cap)]
    if budget is not None:
        words += ["--max-num-batched-tokens", str(budget)]
    if drafter is not None:
        profile = drafter_profile(drafter)
        words.append("--speculative-config.method={}".format(profile.method))
        if drafter_path is None:
            words += [
                "--speculative-config.model={}".format(profile.repo),
                "--speculative-config.revision={}".format(profile.revision),
            ]
        else:
            words.append("--speculative-config.model={}".format(drafter_path))
        words += [
            "--speculative-config.num_speculative_tokens={}".format(max_lookahead(schedule)),
            "--speculative-config.num_speculative_tokens_per_batch_size={}".format(
                schedule_word(schedule)),
        ]
    return words


def advanced_environment(advanced: Advanced) -> List[str]:
    """The container variable the newer runner adds, or nothing."""
    return [V2_RUNNER_ENV] if advanced[4] else []


# --- VD-216 work order item 2: the speed-ups measured on gfx1151 only ----------

#: The one GPU both item-2 speed-ups were measured on: gfx1151 (the Radeon
#: 8060S of the Z2 and the ZBook), as the kfd topology spells it
#: (``gfx_target_version``; `vllm_images.gfx_family_major`). Exactly this
#: number, not its family: nothing else was run with them.
MEASURED_GPU = "110501"

#: The record field the qk-norm/RoPE fusion is kept under (written only when on).
FUSION_FIELD = "qk_norm_rope_fusion"

#: **The decision hook.** ``False`` renders every deployment as before item 2.
FUSION_SERVES = True

#: The word vLLM 0.27 reads into ``CompilationConfig.pass_config``: its own
#: fused qk-norm + RoPE kernel (``torch.ops._C.fused_qk_norm_rope``) in place of
#: four per layer. WO2 (2026-10-15, kb N46/N57): exact within one bf16 ulp,
#: 887 -> 738 dispatches per step, steps 0.9-2.2% faster, HumanEval unchanged.
FUSION_WORD = "--compilation-config.pass_config.enable_qk_norm_rope_fusion=true"

#: The (image, model) pairs both item-2 speed-ups were measured on: the pinned
#: 0.27 image and the catalog's 30B, one copy per machine.
SPEEDUPS_MEASURED = frozenset({(VLLM_027_ROCM10_KEY, "cyankiwi/Qwen3-30B-A3B-Instruct-2507-AWQ-4bit")})

#: What the form says when the 30B would get the item-2 speed-ups but a chosen
#: machine's GPU is not the one they were measured on (D43's pattern: the
#: speed-up stands aside on that hardware, and says so).
SPEEDUPS_STAND_ASIDE = (
    "The fused qk-norm/RoPE step and the 4-bit guesser are left off: they were "
    "measured only on gfx1151 GPUs (the Radeon 8060S), and {names} {verb} not "
    "report one. The model serves with vLLM's own steps and the full-size guesser."
)


def measured_gpus(gpu_targets: Any) -> bool:
    """Whether EVERY machine of a deployment reported :data:`MEASURED_GPU`; none is ``False``."""
    targets = [str(target or "").strip() for target in (gpu_targets or ())]
    return bool(targets) and all(target == MEASURED_GPU for target in targets)


def decide_fusion(image: VllmImage, repo: Any, replicated: bool, gpu_targets: Any) -> bool:
    """Whether a deployment serves with the fusion: measured image, model and GPUs, one copy per machine."""
    return (FUSION_SERVES and replicated and (image.key, str(repo or "").strip()) in SPEEDUPS_MEASURED
            and measured_gpus(gpu_targets))


def require_fusion(value: Any, *, image: VllmImage) -> bool:
    """The renderer's rule for a stored fusion field, on both sides of the bridge."""
    if value is None:
        return False
    if value is not True:
        raise ValueError("A stored qk-norm/RoPE fusion setting must be on when present.")
    if image.key not in {pair[0] for pair in SPEEDUPS_MEASURED}:
        raise ValueError("The qk-norm/RoPE fusion was measured on the vLLM 0.27 image only.")
    # The hook off (R86): every unit renders as before item 2, stored ones too.
    return FUSION_SERVES


def fusion_words(on: bool) -> List[str]:
    """The ``vllm serve`` word the fusion adds, or nothing."""
    return [FUSION_WORD] if on else []


def speedups_note(repo: Any, image: VllmImage, gpu_nodes: Any) -> str:
    """:data:`SPEEDUPS_STAND_ASIDE` naming the chosen machines that are not gfx1151, or ``""``.

    Said only for the measured model on the measured image: anything else
    never had the speed-ups to stand aside from. A machine that reported no
    GPU version is named too - a GPU nobody read is not assumed to be gfx1151.
    """
    if (image.key, str(repo or "").strip()) not in SPEEDUPS_MEASURED:
        return ""
    names = [
        str(node.get("name") or node.get("node_id") or "a machine")
        for node in gpu_nodes or () if isinstance(node, Mapping)
        and not measured_gpus([node.get("gfx_target_version")])
    ]
    if not names:
        return ""
    return SPEEDUPS_STAND_ASIDE.format(names=" and ".join(names),
                                       verb="did" if len(names) == 1 else "do")


#: What the form says of the newer runner, in plain words.
NEW_RUNNER_OFFERED = (
    "vLLM 0.27's newer model runner. A draft model whose lookahead changes "
    "with load needs it: on the older runner that setting made one user "
    "markedly slower (measured on an HP Z2 Mini G1a)."
)
NEW_RUNNER_NOT_OFFERED = (
    "This vLLM image does not have the newer runner; choose vLLM 0.27 to use it."
)
ADVANCED_MEASURED = (
    "Recommended for this model on one copy per machine: up to {cap} requests "
    "per machine{drafter}{runner}. Measured on an HP Z2 Mini G1a, where total "
    "speed rose at every load measured and answer quality was unchanged. A "
    "model split across machines uses vLLM's own settings."
)
ADVANCED_UNMEASURED = (
    "No settings were measured for this model, so vLLM's own are used unless "
    "you set these."
)


def advanced_choices(repo: Any, image: VllmImage, gpu_nodes: Any = ()) -> Dict[str, Any]:
    """The form's ``advanced`` object for one model on one image (the VD-213 contract).

    ``gpu_nodes`` are the machines the form has chosen: the note also says
    when the item-2 speed-ups stand aside on them (:func:`speedups_note`).
    """
    defaults = default_serving_for(repo)
    offered = drafters_for(repo, image.key)
    default_drafter = defaults.get(DRAFTER_FIELD)
    if default_drafter and not any(item.key == default_drafter["key"] for item in offered):
        default_drafter = None
    runner_default = bool(defaults.get(NEW_RUNNER_FIELD)) and image.v2_runner
    if defaults:
        schedule = (default_drafter or {}).get("schedule") or []
        drafter_words = "" if not schedule else (
            ", the draft model drafting {} tokens for one request and {} for "
            "{}-{}".format(schedule[0][2], schedule[-1][2], schedule[-1][0], schedule[-1][1])
            if len(schedule) > 1 else ", the draft model drafting {} tokens".format(schedule[0][2])
        )
        note = ADVANCED_MEASURED.format(
            cap=defaults.get(MAX_NUM_SEQS_FIELD), drafter=drafter_words,
            runner=", on the newer runner" if runner_default else "",
        )
    else:
        note = ADVANCED_UNMEASURED
    aside = speedups_note(repo, image, gpu_nodes)
    if aside:
        note = "{} {}".format(note, aside)
    return {
        MAX_NUM_SEQS_FIELD: {
            "default": defaults.get(MAX_NUM_SEQS_FIELD),
            "min": MAX_NUM_SEQS_RANGE[0], "max": MAX_NUM_SEQS_RANGE[1],
        },
        BATCHED_TOKENS_FIELD: {
            "default": defaults.get(BATCHED_TOKENS_FIELD),
            "min": BATCHED_TOKENS_RANGE[0], "max": BATCHED_TOKENS_RANGE[1],
        },
        "drafters": [
            {"key": item.key, "label": item.label, "repo": item.repo,
             "weight_bytes": item.weight_bytes, "max_tokens": item.max_tokens}
            for item in offered
        ],
        "drafter_default": (
            {"key": default_drafter["key"], "schedule": default_drafter["schedule"]}
            if default_drafter else None
        ),
        NEW_RUNNER_FIELD: {
            "offered": image.v2_runner, "default": runner_default,
            "note": NEW_RUNNER_OFFERED if image.v2_runner else NEW_RUNNER_NOT_OFFERED,
        },
        "note": note,
    }

