"""The per-model ``vllm serve`` options a cluster unit carries, as ONE typed value.

A cluster deployment's units are rendered from typed values, never from text a
client composed (VD-143): on the controller the root bridge renders them
itself. This module is the one rule for the value that says how THIS model is
served - the image it runs in (`vllm_images`), its tool-call and reasoning
parsers, whether its vision tower is skipped, whether its multi-token
prediction head drafts tokens, whether its container starts through the
entry program that shows vLLM its tuned MoE tables (`vllm_entry_program`),
and how a hybrid model's cache is kept on a GPU without FP8 hardware - and
the one place each becomes a ``vllm serve`` word or a container variable:

* :func:`options_for` decides the value at deploy time, from the model's
  profile (`vllm_model_profile`), the GPUs it will run on and the owner's
  choices in the payload;
* :func:`require_options` is the renderer's check, on both sides of the bridge:
  a mapping of exactly these fields, each of its own type, the parser names
  from the allowlists vLLM registers and the image from the pinned table. A
  string where a count belongs, a parser or image name that is not listed, an
  extra key - all refused, so no value can carry a word, a flag or a shell
  character into a unit;
* :func:`serve_arguments` and :func:`container_environment` render it.

**A hybrid model's cache, on RDNA without FP8 hardware, on vLLM 0.27**
(:func:`hybrid_cache`; owner decision 2026-09-30, measured on the 27B). The
Qwen3.5-architecture models mix Gated-DeltaNet layers with full attention. On
gfx1151 vLLM's automatic choices cost them their long-context speed twice: a
checkpoint that declares an FP8 KV scheme gets an FP8 cache, which this GPU
emulates; and their 256-wide heads rule out ROCm's own attention kernel. So
such a model is started with ``--kv-cache-dtype bfloat16 --attention-backend
TRITON_ATTN`` (8k-context decode went from 272 to 80 ms a token; the cost is
about 19% on an 8k prompt's prefill) and, for chat, with the aligned prefix
cache: ``--enable-prefix-caching --mamba-cache-mode align --prefix-match-unit
16`` (turn two of a conversation started in 6.3 s instead of 20.8). vLLM keeps
prefix caching off for hybrid models unless asked, and calls ``align``
experimental. **Multi-token prediction and the aligned prefix cache were each
measured alone, never together**, so a deployment with prediction on gets the
bf16 flags and no prefix cache. The decision is the GPU family's and the
config's, never a name's.

**Every flag is vLLM's own, read at tags v0.22.1 and v0.27.0 and parsed by the
0.27 image's own argument parser** (LESSONS 15): ``--tool-call-parser`` and
``--reasoning-parser``; ``--language-model-only`` (every modality limit
becomes zero, and ``_mark_tower_model`` then skips building and loading the
vision tower); and ``--speculative-config`` in vLLM's dotted form,
``--speculative-config.method=mtp --speculative-config.num_speculative_tokens=N``,
which ``FlexibleArgumentParser.parse_args`` folds into the same JSON object the
model card passes - so no brace or quote reaches a systemd ``ExecStart`` or the
lead's ``bash -c`` script. ``mtp`` loads the checkpoint's own MTP layer, so it
is only rendered for a model whose config declares one.

**A record without the value is served as its model's profile says**, on the
image it was deployed on (:func:`options_from_record`;
`vllm_images.RECORDED_IMAGE_KEY`, 0.22.1 - never the default of the day), so
a Qwen3 record gains its reasoning parser at its next Load and an unknown
model's unit is unchanged. A record that stored the value before the cache
flags existed carries six fields and is rendered as it was deployed: without
them.

**The value crosses to a root bridge that may be older than this code** (an
upgrade installs the control plane and the bridge in one wheel, but each keeps
running until it is restarted). So a value whose cache switches are both off
is WRITTEN with the six fields an older bridge knows (:meth:`ServeOptions.
as_record`): every deployment that does not need the new switches loads
against an older bridge exactly as before. One that does need them is refused
by that bridge, and the refusal is said as what to do about it
(:data:`BRIDGE_TOO_OLD`). The Advanced settings (VD-213: request cap, batch
budget, draft model, newer runner; `vllm_serve_advanced`) follow the same
rule: each is written only when it is set, so a value without them is the
record an older bridge accepts byte for byte. So does the patched-attention
overlay a draft model serves with (VD-215, `vllm_overlays`): decided for a new
deployment, kept as recorded on a Load. So does Vaelor's expert kernel (VD-216,
`vllm_moe_kernels`). So do VD-216 work order item 2's two speed-ups - the
qk-norm/RoPE fusion (`vllm_serve_advanced`) and the 4-bit guesser
(`vllm_guesser_build`) - which are also ADDED at a Load where they were
measured (:func:`with_measured_speedups`), never removed by one. VD-217: and
a Load gives a record made before the Advanced settings, the attention file
or the expert kernel what a fresh deploy gets, where it was measured, never
over an owner's choice (`vllm_release_defaults`).
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, replace
from typing import Any, Dict, Iterable, List, Mapping, Optional, Tuple

from .gpu_model_catalog import served_repo
from .vllm_drafters import drafter_profile
from .vllm_entry_program import TUNED_FOLDER_ENV
from .vllm_serve_advanced import (
    ADVANCED_FIELDS, FUSION_FIELD, NOTHING, SPLIT_REFUSED, Advanced, advanced_choices,
    advanced_environment, advanced_for, advanced_from_record, advanced_record, advanced_words,
    decide_fusion, fusion_words, require_fusion,
)
from .vllm_guesser_build import (
    CONTAINER_DIR as GUESSER_CONTAINER_DIR, GUESSER_FIELD, GuesserBuild, decide_guesser,
    refused_fields, require_guesser,
)
from .vllm_overlays import (
    OVERLAY_FIELD, VllmOverlay, decide_overlay, overlay_environment, overlay_words,
    require_overlay,
)
from .vllm_moe_kernels import (
    MOE_KERNEL_FIELD, VllmMoeKernel, decide_moe_kernel, kernel_environment, require_moe_kernel,
)
from .vllm_images import (
    DEFAULT_IMAGE_KEY, IMAGE_FIELD, IMAGES, RECORDED_IMAGE_KEY, VllmImage,
    all_rdna3_family, gfx_family_major, image_profile,
)
from .vllm_model_profile import (
    DEFAULT_TOOL_CALL_PARSER, FAST_W4A16, REASONING_PARSERS, TOOL_CALL_PARSERS,
    ModelFacts, ModelProfile, kernel_note, profile_for,
)

#: The unit value, and the record key, the options travel under.
OPTIONS_FIELD = "vllm_options"

#: The deploy payload's owner choices besides the image (:data:`IMAGE_FIELD`).
MTP_FIELD = "mtp_tokens"
TEXT_ONLY_FIELD = "text_only"

#: The most draft tokens one MTP step may propose. The Qwen3.8 INT4 card runs
#: three; vLLM warns that more than one reruns the same MTP layer and lowers
#: acceptance, so four bounds the value without inviting a setting that is
#: slower than none.
MAX_MTP_TOKENS = 4

#: The count the deploy form suggests when its owner turns prediction on: the
#: one that was measured (the 0.27 benchmark's run c, 2026-09-30).
MTP_SUGGESTED_TOKENS = 2

_FIELDS = (
    "tool_call_parser", "reasoning_parser", "mtp_tokens", "text_only",
    "tuned_moe", IMAGE_FIELD,
)
#: The two cache switches added after the first six. A stored value from
#: before them carries neither and reads as both off - the unit it was
#: deployed with; a value that carries one must carry both.
_CACHE_FIELDS = ("bf16_attention", "aligned_prefix_cache")

#: The words each cache switch renders to, spelled once. vLLM's own flags,
#: read at tag v0.27.0 (``vllm/engine/arg_utils.py``, ``vllm/config/cache.py``)
#: and run on the 0.27 image.
_BF16_ATTENTION_WORDS = (
    "--kv-cache-dtype", "bfloat16", "--attention-backend", "TRITON_ATTN",
)
_ALIGNED_PREFIX_WORDS = (
    "--enable-prefix-caching", "--mamba-cache-mode", "align",
    "--prefix-match-unit", "16",
)


@dataclass(frozen=True)
class ServeOptions:
    """How one deployment's model is served. The defaults are the pre-profile unit.

    The default image is the one a unit that names none was always rendered
    for (`vllm_images.RECORDED_IMAGE_KEY`), not the default a new deployment
    gets: a value built with no image describes a deployment from before the
    choice. `options_for` names the image of every new one.
    """

    tool_call_parser: str = DEFAULT_TOOL_CALL_PARSER
    reasoning_parser: Optional[str] = None
    mtp_tokens: int = 0
    text_only: bool = False
    tuned_moe: bool = False
    vllm_image: str = RECORDED_IMAGE_KEY
    #: A bf16 KV cache on the Triton attention backend (:func:`hybrid_cache`).
    bf16_attention: bool = False
    #: The aligned prefix cache of a hybrid model (:func:`hybrid_cache`).
    aligned_prefix_cache: bool = False
    #: The Advanced settings (VD-213), `vllm_serve_advanced.Advanced`:
    #: request cap, batch budget, draft model and its lookahead table, newer
    #: runner. All off is a unit as it was before they existed.
    advanced: Advanced = NOTHING
    #: The patched-attention overlay a draft model serves with (VD-215,
    #: `vllm_overlays`), by its table key; ``None`` is the image's own file.
    attention_overlay: Optional[str] = None
    #: Vaelor's expert-kernel plugin (VD-216, `vllm_moe_kernels`), by its
    #: table key; ``None`` is vLLM's own expert kernel.
    moe_kernel: Optional[str] = None
    #: vLLM's fused qk-norm/RoPE step (VD-216 item 2, `vllm_serve_advanced`).
    qk_norm_rope_fusion: bool = False
    #: The 4-bit guesser build (VD-216 item 2, `vllm_guesser_build`), by its
    #: table key; ``None`` is the pinned bf16 drafter itself.
    guesser: Optional[str] = None

    def as_record(self) -> Dict[str, Any]:
        """The value as a unit parameter and a record field: plain JSON types.

        The two cache switches are written only when one of them is on, so a
        value without them is the six fields it always was - byte for byte
        what a bridge from before the switches accepts (module docstring).
        """
        record = asdict(self)
        del record["advanced"]
        # Written only when set (VD-215), as the Advanced settings are: a
        # deployment without it is the record an older bridge accepts.
        if self.attention_overlay is None:
            del record[OVERLAY_FIELD]
        if self.moe_kernel is None:
            del record[MOE_KERNEL_FIELD]
        # VD-216 item 2, the same rule: absent unless set.
        if not self.qk_norm_rope_fusion:
            del record[FUSION_FIELD]
        if self.guesser is None:
            del record[GUESSER_FIELD]
        if not (self.bf16_attention or self.aligned_prefix_cache):
            for field in _CACHE_FIELDS:
                del record[field]
        record.update(advanced_record(self.advanced))
        return record


def _refuse(field: str) -> ValueError:
    return ValueError("The vLLM serving option '{}' is invalid.".format(field))


def require_options(value: Any) -> ServeOptions:
    """The renderer's rule for ``vllm_options``: exactly its typed fields, or refused.

    ``None`` is a unit rendered without the value (every caller before it
    existed): the pre-profile defaults, on the image those units ran. Anything
    else must be a mapping of exactly :data:`_FIELDS`, with both or neither of
    :data:`_CACHE_FIELDS` (an already-built :class:`ServeOptions` is checked as
    its record); a bool is not a count, and a count is not a bool. The MoE
    entry program and the cache switches are refused for an image that cannot
    take them (`vllm_images`), and the aligned prefix cache is refused
    without the bf16 cache or with multi-token prediction: the renderer emits
    only the combinations that were measured (:func:`hybrid_cache`).
    """
    if value is None:
        return ServeOptions()
    if isinstance(value, ServeOptions):
        value = value.as_record()
    if not isinstance(value, Mapping):
        raise ValueError("The vLLM serving options must be named values.")
    keys = set(str(key) for key in value) - set(ADVANCED_FIELDS) - {
        OVERLAY_FIELD, MOE_KERNEL_FIELD, FUSION_FIELD, GUESSER_FIELD}
    if keys not in (set(_FIELDS), set(_FIELDS + _CACHE_FIELDS)):
        raise _refuse(", ".join(sorted(
            keys.symmetric_difference(_FIELDS + _CACHE_FIELDS)
        )))
    tool_parser = value["tool_call_parser"]
    if not isinstance(tool_parser, str) or tool_parser not in TOOL_CALL_PARSERS:
        raise _refuse("tool_call_parser")
    reasoning = value["reasoning_parser"]
    if reasoning is not None and (
        not isinstance(reasoning, str) or reasoning not in REASONING_PARSERS
    ):
        raise _refuse("reasoning_parser")
    tokens = value["mtp_tokens"]
    if isinstance(tokens, bool) or not isinstance(tokens, int) or not (
        0 <= tokens <= MAX_MTP_TOKENS
    ):
        raise _refuse("mtp_tokens")
    cache = {field: value.get(field, False) for field in _CACHE_FIELDS}
    for flag, setting in (
        ("text_only", value["text_only"]), ("tuned_moe", value["tuned_moe"]),
        *cache.items(),
    ):
        if not isinstance(setting, bool):
            raise _refuse(flag)
    try:
        # A stored value that names no image is one from before an image was
        # named: the recorded image, never today's default (`vllm_images`).
        stored_image = value[IMAGE_FIELD]
        image = image_profile(RECORDED_IMAGE_KEY if stored_image is None else stored_image)
    except ValueError:
        raise _refuse(IMAGE_FIELD)
    if value["tuned_moe"] and not image.moe_entry:
        raise _refuse("tuned_moe")
    bf16, aligned = cache["bf16_attention"], cache["aligned_prefix_cache"]
    if bf16 and not image.hybrid_cache:
        raise _refuse("bf16_attention")
    if aligned and (not bf16 or tokens):
        raise _refuse("aligned_prefix_cache")
    # The Advanced settings' one rule (`vllm_serve_advanced.check_advanced`),
    # the same the deploy decided them with.
    advanced = advanced_from_record(value, image=image, mtp_tokens=tokens)
    # VD-215: a stored overlay key, present only when set; a present null is
    # refused rather than read as off, like the Advanced fields.
    if OVERLAY_FIELD in value and value[OVERLAY_FIELD] is None:
        raise _refuse(OVERLAY_FIELD)
    overlay = require_overlay(value.get(OVERLAY_FIELD), image=image, drafter=advanced[2])
    # VD-216: the same rule for Vaelor's expert kernel.
    if MOE_KERNEL_FIELD in value and value[MOE_KERNEL_FIELD] is None:
        raise _refuse(MOE_KERNEL_FIELD)
    kernel = require_moe_kernel(value.get(MOE_KERNEL_FIELD), image=image)
    # VD-216 item 2: present only when set, and a present null is refused.
    for field in (FUSION_FIELD, GUESSER_FIELD):
        if field in value and value[field] is None:
            raise _refuse(field)
    fusion = require_fusion(value.get(FUSION_FIELD), image=image)
    guesser = require_guesser(value.get(GUESSER_FIELD), image=image, drafter=advanced[2])
    return ServeOptions(
        tool_call_parser=tool_parser, reasoning_parser=reasoning,
        mtp_tokens=tokens, text_only=value["text_only"],
        tuned_moe=value["tuned_moe"], vllm_image=image.key,
        bf16_attention=bf16, aligned_prefix_cache=aligned, advanced=advanced,
        attention_overlay=None if overlay is None else overlay.key,
        moe_kernel=None if kernel is None else kernel.key,
        qk_norm_rope_fusion=fusion, guesser=None if guesser is None else guesser.key,
    )


def image_of(value: Any) -> VllmImage:
    """The pinned image a unit rendered from ``value`` runs."""
    return image_profile(require_options(value).vllm_image)


def requested_mtp_tokens(raw: Any, mtp_layers: Optional[int]) -> int:
    """The owner's MTP choice, refused where the model cannot honour it.

    ``mtp_layers`` is how many MTP layers the model's config declares, ``None``
    when no config was read. One rule for the fit preview and the deploy, so
    both size the draft layer the launch will load (`cluster_gpu_model_spec`).
    """
    if raw is None or raw == 0:
        return 0
    if isinstance(raw, bool) or not isinstance(raw, int) or not 1 <= raw <= MAX_MTP_TOKENS:
        raise ValueError(
            "Multi-token prediction takes a whole number of draft tokens from "
            "1 to {} (0 turns it off).".format(MAX_MTP_TOKENS)
        )
    if mtp_layers is None:
        raise ValueError(
            "Multi-token prediction can only be turned on for a model whose "
            "config.json Vaelor has read - fetch the model into the model "
            "library on this controller first: that file is what says the "
            "model ships a prediction head."
        )
    if mtp_layers <= 0:
        raise ValueError(
            "This model's config.json declares no multi-token-prediction head, "
            "so multi-token prediction cannot be turned on for it."
        )
    return raw


def _require_replicated(tokens: int, replicated: bool, advanced: Advanced = NOTHING) -> None:
    if (advanced[2] is not None or advanced[4]) and not replicated:
        raise ValueError(SPLIT_REFUSED)
    if tokens and not replicated:
        raise ValueError(
            "Multi-token prediction is offered on a one-copy-per-machine "
            "deployment only; it has not been verified across a model split "
            "between machines."
        )


def hybrid_cache(
    hybrid: bool, image: VllmImage, gpu_targets: Iterable[Any], mtp_tokens: int,
) -> Tuple[bool, bool]:
    """``(bf16_attention, aligned_prefix_cache)`` for one deployment (module docstring).

    Three facts and one choice, none of them a name: ``hybrid`` is whether
    the model's ``config.json`` marks linear-attention layers
    (`vllm_model_profile`); ``image`` must be one whose vLLM takes the flags;
    ``gpu_targets`` are the ``gfx_target_version`` each selected machine's GPU
    reported, and EVERY one must be of the family without FP8 hardware
    (`vllm_images.all_rdna3_family`) - a machine that reported none, or no
    machines at all, gets nothing. The aligned prefix cache is then on unless
    multi-token prediction is. One rule for the launch and for the fit, which
    sizes the cache these flags make (`cluster_gpu_model_spec`).
    """
    bf16 = bool(hybrid) and image.hybrid_cache and all_rdna3_family(gpu_targets)
    return bf16, bf16 and not mtp_tokens


#: What the fit and the deploy say when a model that would get the hybrid
#: cache settings does not, because a fact they are decided from could not be
#: read. Plain words, and what to do: a model started without them is not
#: broken, only slow on long conversations, and nothing else would say why.
CACHE_GPU_UNREAD = (
    "Vaelor could not read which GPU family {names} has, so this model is "
    "started without the settings that keep long conversations fast on that "
    "kind of GPU. Refresh {pronoun} on the Fleet page and try again."
)
CACHE_NO_CONFIG = (
    "Vaelor has not read this model's layout yet: its config.json is not in "
    "the model library on this controller. The deploy fetches the model "
    "first and checks the fit again with it; if the file still cannot be "
    "read, the deploy stops before starting the model."
)
#: What a deploy is refused with when that happens (`gpu_pool_refit`): a model
#: of a hybrid family is not launched on a guess at its layout.
CONFIG_NEVER_READ = (
    "The deploy stopped before starting the model: its config.json could not "
    "be read from the model library on this controller, even after fetching "
    "it, and this kind of model needs its layout read to be sized and started "
    "safely. Fetch the model into the model library, check it is listed "
    "there, and deploy again."
)
#: What a Load says when the layout it has since read allows settings that
#: would not fit, or could not be checked (`gpu_pool_refit`).
LOAD_KEPT_AS_DEPLOYED = (
    "Loaded as it was deployed. This model's layout has been read since, but "
    "the settings it allows would not fit these machines: {reason}"
)
LOAD_NOT_CHECKED = (
    "Loaded as it was deployed. This model's layout has been read since, but "
    "the settings it allows could not be checked against these machines, "
    "because the deployment's record does not say how big its weights are. "
    "Deploy it again to use them."
)
#: The name-families whose models are hybrid, for the one case the config
#: cannot speak: it was never read.
_HYBRID_FAMILIES = frozenset({"qwen3_5", "qwen3_5_moe"})


def names_hybrid_family(profile: ModelProfile) -> bool:
    """Whether a model's NAME, with no config read, says it is of a hybrid family."""
    return not profile.config_read and profile.family in _HYBRID_FAMILIES


def cache_note(profile: ModelProfile, image: VllmImage, gpu_nodes: Iterable[Any]) -> str:
    """One plain sentence when the hybrid cache settings could not be decided, or ``""``.

    ``gpu_nodes`` is each selected machine as a mapping with its ``name`` and
    ``gfx_target_version``. Nothing is said for an image that takes no such
    settings, for a model that is not hybrid, or for machines whose GPUs were
    read and are simply of another family. Two things are said: the model's
    ``config.json`` was not read and its NAME is of a hybrid family - so
    whether it is hybrid is not known - and a hybrid model on a machine whose
    GPU family is unread.
    """
    if not image.hybrid_cache:
        return ""
    if not profile.config_read:
        return CACHE_NO_CONFIG if names_hybrid_family(profile) else ""
    if not profile.hybrid:
        return ""
    unread = [
        str(node.get("name") or node.get("node_id") or "a machine")
        for node in gpu_nodes or ()
        if not gfx_family_major(node.get("gfx_target_version"))
    ]
    if not unread:
        return ""
    return CACHE_GPU_UNREAD.format(
        names=" and ".join(unread),
        pronoun="that machine" if len(unread) == 1 else "those machines",
    )


#: What a Load or a deploy is refused with when the root bridge on this
#: controller is older than the control plane and does not know a value this
#: deployment needs (module docstring). The bridge's own refusal names a
#: field; this names what to do.
BRIDGE_TOO_OLD = (
    "The hardware bridge on this controller is still running the previous "
    "Vaelor version, which does not know a setting this deployment needs. "
    "Restart the hardware bridge (or restart the machine) and try again."
)


def bridge_refusal(error: Any, values: Mapping[str, Any]) -> Optional[str]:
    """:data:`BRIDGE_TOO_OLD` when ``error`` is an older bridge refusing a newer value.

    Every value was rendered locally before it crossed, so a bridge that
    refuses one by NAME is a bridge that does not have the name: the cache
    switches in ``vllm_options``, or the image a model pull names. Anything
    else is ``None``, and the bridge's own words stand.
    """
    options = values.get(OPTIONS_FIELD)
    carried = [
        name for name in _CACHE_FIELDS + ADVANCED_FIELDS + (
            OVERLAY_FIELD, MOE_KERNEL_FIELD, FUSION_FIELD, GUESSER_FIELD)
        if isinstance(options, Mapping) and name in options
    ]
    if IMAGE_FIELD in values:
        carried.append(IMAGE_FIELD)
    text = str(error)
    # VD-216 item 2 (G64 residue): "guesser" is an ordinary word in the
    # bridge's own sentences, so item 2's two fields count only when the
    # refusal LISTS them by name - never because a sentence mentions one.
    exact = {FUSION_FIELD, GUESSER_FIELD}
    listed = refused_fields(text)
    return BRIDGE_TOO_OLD if any(
        (name in listed) if name in exact else (name in text) for name in carried) else None


def options_for(
    profile: ModelProfile, payload: Mapping[str, Any], *, replicated: bool,
    gpu_targets: Iterable[Any] = (),
) -> ServeOptions:
    """The options a deploy serves ``profile``'s model with, from the payload's choices.

    ``text_only`` defaults ON for a model with a vision tower: the cluster
    serves chat, and skipping the tower frees its memory and its start-up
    profiling. It means nothing for a text model and is recorded ``False``.
    A value that is not a real boolean is refused rather than read for
    truthiness. MTP is off unless asked for (:func:`requested_mtp_tokens`).
    A payload that names no image gets the default (`vllm_images`).
    ``gpu_targets`` is each selected machine's ``gfx_target_version``, for
    the hybrid cache switches (:func:`hybrid_cache`).
    """
    raw_text_only = payload.get(TEXT_ONLY_FIELD)
    if raw_text_only is not None and not isinstance(raw_text_only, bool):
        raise ValueError("The text-only setting must be true or false.")
    image = image_profile(payload.get(IMAGE_FIELD))
    tokens = requested_mtp_tokens(
        payload.get(MTP_FIELD), profile.mtp_layers if profile.config_read else None,
    )
    _require_replicated(tokens, replicated)
    bf16, aligned = hybrid_cache(profile.hybrid, image, gpu_targets, tokens)
    advanced, overlay, kernel = _deploy_choices(
        payload, _payload_repo(payload), image, tokens, replicated)
    return with_measured_speedups(ServeOptions(
        tool_call_parser=profile.tool_call_parser,
        reasoning_parser=profile.reasoning_parser,
        mtp_tokens=tokens,
        text_only=profile.vision and (raw_text_only is not False),
        tuned_moe=profile.moe and image.moe_entry,
        vllm_image=image.key,
        bf16_attention=bf16, aligned_prefix_cache=aligned, advanced=advanced,
        attention_overlay=overlay, moe_kernel=kernel,
    ), _payload_repo(payload), replicated=replicated, gpu_targets=gpu_targets)


def _deploy_choices(
    payload: Mapping[str, Any], repo: Optional[str], image: VllmImage, tokens: int,
    replicated: bool,
) -> Tuple[Advanced, Optional[str], Optional[str]]:
    """``(advanced, overlay, kernel)`` a deploy decides: THE one decision (LESSONS 6).

    Read by :func:`options_for` with the deploy's payload and by
    :func:`with_release_defaults` with an empty one - what an untouched
    deploy form gets - so a Load cannot add something a deploy would not.
    """
    # The Advanced settings (VD-213): the payload's, else this repo's catalog
    # defaults on one copy per machine (`vllm_serve_advanced.advanced_for`).
    advanced = advanced_for(
        payload, repo=repo, image=image, mtp_tokens=tokens, replicated=replicated,
    )
    # VD-215: a draft model on one copy per machine serves with the patched
    # attention file where its image has one (`vllm_overlays`); VD-216: the
    # 30B on one copy per machine serves its small expert calls with Vaelor's
    # own kernel (`vllm_moe_kernels`).
    return (advanced, decide_overlay(image, advanced[2], replicated),
            decide_moe_kernel(image, repo, replicated))


def with_measured_speedups(
    options: ServeOptions, repo: Any, *, replicated: bool, gpu_targets: Iterable[Any],
) -> ServeOptions:
    """``options`` with VD-216 item 2's speed-ups added where they were measured.

    The qk-norm/RoPE fusion (`vllm_serve_advanced.decide_fusion`) and the
    4-bit guesser (`vllm_guesser_build.decide_guesser`): the 30B on the 0.27
    image, one copy per machine, every machine gfx1151 - read from the
    ``gfx_target_version`` each machine reported, the facts the hybrid cache
    is decided from. A deploy decides them, and so does every Load, from the
    machines as they are now: a record made before them gains them (LESSONS
    22), and one whose machines now READ as another GPU loses them (R86; the
    Load says so, `gpu_pool_refit`). Only when a machine's GPU cannot be read
    is the record kept as it is - an unread version is not evidence the GPU
    changed (LESSONS 8) - and nothing is added either.
    """
    targets = list(gpu_targets or ())
    image = image_profile(options.vllm_image)
    if not targets or not all(gfx_family_major(target) for target in targets):
        return options
    fusion = decide_fusion(image, repo, replicated, targets)
    guesser = decide_guesser(image, options.advanced[2], replicated, targets, repo=str(repo or ""))
    if fusion == options.qk_norm_rope_fusion and guesser == options.guesser:
        return options
    return replace(options, qk_norm_rope_fusion=fusion, guesser=guesser)


def _payload_repo(payload: Mapping[str, Any]) -> Optional[str]:
    """The repo a deploy payload's ``model_source`` serves, ``None`` when it names none.

    A source the deploy refuses is refused by the deploy's own resolution
    first; here it only means there are no defaults to read.
    """
    if not payload.get("model_source"):
        return None
    try:
        return served_repo(payload.get("model_source"))[0]
    except ValueError:
        return None


def options_draft_bytes(value: Any) -> int:
    """The weight bytes of the draft model ``value`` serves with, 0 for none (VD-213)."""
    key = require_options(value).advanced[2]
    return drafter_profile(key).weight_bytes if key is not None else 0


def payload_draft_bytes(
    repo: Any, facts: Optional[ModelFacts], payload: Mapping[str, Any],
    gpu_targets: Iterable[Any] = (),
) -> int:
    """The draft model a whole copy of this deploy would load, in bytes (VD-213).

    Decided by :func:`options_for` itself, as a one-copy-per-machine
    deployment, so the fit charges exactly the drafter the units will run.
    """
    return options_draft_bytes(options_for(
        profile_for(repo, facts), payload, replicated=True, gpu_targets=gpu_targets,
    ))


def options_from_record(
    units: Mapping[str, Any], repo: Any, *, replicated: bool = True,
) -> ServeOptions:
    """The options a stored deployment is re-rendered with on a Load.

    The record's own value when it has one - re-checked, including that MTP
    is still only on a one-copy-per-machine deployment - otherwise its repo's
    profile with no ``config.json``, with no MTP and the tower kept: what a
    record from before this value was launched with, plus the parsers its
    name implies.

    **Such a record keeps the image it was deployed on**
    (`vllm_images.RECORDED_IMAGE_KEY`): there was one image before a
    deployment could name one, and it is not the default any more. Reading
    the default here would move a running deployment onto another vLLM, with
    a cold compile cache, the next time it was loaded - which nobody asked
    for. A Load then records the image, so it is said from then on.
    """
    stored = (units or {}).get(OPTIONS_FIELD)
    if stored is not None:
        options = require_options(stored)
        _require_replicated(options.mtp_tokens, replicated, options.advanced)
        if options.qk_norm_rope_fusion and not replicated:
            raise ValueError(SPLIT_REFUSED)
        return options
    profile = profile_for(repo, None)
    recorded = image_profile(RECORDED_IMAGE_KEY)
    return ServeOptions(
        tool_call_parser=profile.tool_call_parser,
        reasoning_parser=profile.reasoning_parser,
        tuned_moe=profile.moe and recorded.moe_entry,
        vllm_image=recorded.key,
    )


#: The record key that says whether the options were decided with the model's
#: ``config.json`` read. ``False`` is a deployment whose layout was not known
#: when it was made; a Load decides them again once the file can be read.
CONFIG_READ_FIELD = "vllm_config_read"


def options_for_load(
    units: Mapping[str, Any], repo: Any, *, replicated: bool,
    facts: Optional[ModelFacts], gpu_targets: Iterable[Any] = (),
    release_defaults: bool = True, drafter: Optional[bool] = None,
) -> Tuple[ServeOptions, bool]:
    """``(options, config_read)`` a Load may serve a stored deployment with.

    VD-217: ``release_defaults`` adds what a fresh deploy gets here now
    (`vllm_release_defaults`); ``drafter`` says whether its draft model
    is offered - ``True`` offers it even where an earlier Load declined it
    (an owner's Load re-checks), ``False`` never (the fit or the fetch said
    no), ``None`` unless the record says it was declined
    (`vllm_release_defaults.DRAFTER_DECLINED_FIELD`; the upgrade's refresh check and job).

    :func:`options_from_record`, with one addition. A deployment made before
    its model's ``config.json`` was in the model library was decided without
    it - and its record says so (:data:`CONFIG_READ_FIELD` is ``False``). When
    ``facts`` - the file as the library holds it now - can be read, the ONE
    thing the file newly decides is added: the hybrid cache settings
    (:func:`hybrid_cache`), on the image the record names. Every other field
    is the one the deployment was made with - its text-only choice above all,
    so a vision model whose owner kept its image reader keeps it (review R5).
    Whether the added settings FIT is the caller's to ask
    (`gpu_pool_refit.options_for_load_checked`). A record that says nothing
    about the config, or says it was read, is rendered as stored.
    """
    # VD-217: what a fresh deploy gets here now, unless the fit said no
    # (`gpu_pool_refit`), for a record made before it (:func:`with_release_defaults`).
    from .vllm_release_defaults import DRAFTER_DECLINED_FIELD, with_release_defaults

    recorded = options_from_record(units, repo, replicated=replicated)
    if release_defaults:
        marker = (units or {}).get(DRAFTER_DECLINED_FIELD)
        declined = str(marker.get("key") or "") if isinstance(marker, Mapping) else None
        recorded = with_release_defaults(
            recorded, (units or {}).get(OPTIONS_FIELD), repo, replicated=replicated,
            gpu_targets=gpu_targets, declined=declined, drafter=drafter)
    # VD-216 item 2: what this release measured, added to the stored options
    # on the machines as they are now (:func:`with_measured_speedups`).
    stored = with_measured_speedups(
        recorded, repo, replicated=replicated, gpu_targets=gpu_targets,
    )
    if (units or {}).get(CONFIG_READ_FIELD) is not False or facts is None:
        return stored, (units or {}).get(CONFIG_READ_FIELD) is not False
    bf16, aligned = hybrid_cache(
        facts.linear is not None, image_profile(stored.vllm_image), gpu_targets,
        stored.mtp_tokens,
    )
    return replace(stored, bf16_attention=bf16, aligned_prefix_cache=aligned), True


def image_on_record(units: Mapping[str, Any]) -> VllmImage:
    """The pinned image a stored deployment runs: its own, or the recorded one.

    For the console's detail view. A record whose stored value cannot be read
    is shown as the recorded image rather than failing the listing.
    """
    try:
        return image_of((units or {}).get(OPTIONS_FIELD))
    except ValueError:
        return image_profile(RECORDED_IMAGE_KEY)


def serve_arguments(value: Any) -> List[str]:
    """The ``vllm serve`` words ``value`` renders to, in one fixed order.

    The tool-call pair stays where every unit has always carried it, so a
    unit rendered with the defaults is byte-identical to one rendered before
    the value existed. The cache switches follow the text-only word, and
    multi-token prediction stays last.
    """
    options = require_options(value)
    words = ["--enable-auto-tool-choice", "--tool-call-parser", options.tool_call_parser]
    if options.reasoning_parser is not None:
        words += ["--reasoning-parser", options.reasoning_parser]
    if options.text_only:
        words.append("--language-model-only")
    if options.bf16_attention:
        words += _BF16_ATTENTION_WORDS
    if options.aligned_prefix_cache:
        words += _ALIGNED_PREFIX_WORDS
    if options.mtp_tokens:
        words += [
            "--speculative-config.method=mtp",
            "--speculative-config.num_speculative_tokens={}".format(options.mtp_tokens),
        ]
    # The Advanced settings (VD-213) after everything else, so a value
    # without them renders byte for byte as before; then the overlay's
    # backend (VD-215), never twice.
    words += advanced_words(options.advanced, drafter_path=(
        None if options.guesser is None else GUESSER_CONTAINER_DIR))
    if options.attention_overlay is not None:
        words += overlay_words(options.bf16_attention)
    # VD-216 item 2: last, so a value without it renders byte for byte as before.
    words += fusion_words(options.qk_norm_rope_fusion)
    return words


#: What the deploy form says about each choice, in plain words.
MTP_OFFERED = (
    "Off by default. The model can draft extra words each step: measured on "
    "the 27B, that made a short chat with one user faster (12.3 to 17.0 "
    "tokens a second) and made long prompts and many users at once slower. "
    "2 extra words is the setting that was measured."
)
MTP_NO_CONFIG = (
    "Fetch this model into the model library on this controller first: its "
    "config.json is what says whether it has a prediction head."
)
MTP_NO_HEAD = "This model has no prediction head, so this does not apply."
TEXT_ONLY_OFFERED = (
    "This model can also read images. Serving text only skips loading that "
    "part, which frees memory; on by default for chat."
)


#: What the deploy form says of an image's 4-bit kernel (ACC-205). Before the
#: model's config is read the weights' format is not known, so the note says
#: what the image does and asks for the fetch; once it is read, the fast path
#: is said too rather than left blank.
KERNEL_FAST_UNREAD = (
    "This image has the fast 4-bit kernel for this GPU; fetch the model to "
    "check that its weights can use it."
)
KERNEL_SLOW_UNREAD = (
    "This image runs 4-bit weights through a slow general kernel on this GPU; "
    "fetch the model to check whether its weights are 4-bit."
)
KERNEL_FAST_READ = "This image runs this model's 4-bit weights on the fast kernel for this GPU."


def form_kernel_note(profile: ModelProfile, image: VllmImage) -> str:
    """The form's note on how fast this model's weights run in ``image``: what is known, never blank by default."""
    if not profile.config_read:
        return KERNEL_FAST_UNREAD if image.fast_w4a16 else KERNEL_SLOW_UNREAD
    if profile.weight_format == FAST_W4A16 and image.fast_w4a16:
        return KERNEL_FAST_READ
    return kernel_note(profile.weight_format, image.fast_w4a16)


def serving_choices(
    repo: Any, facts: Optional[ModelFacts], *, vllm_image: Any = None,
    gpu_nodes: Iterable[Any] = (),
) -> Dict[str, Any]:
    """What the deploy form may offer for one model: the images, MTP, text-only.

    From the same profile the deploy decides with, so the form never offers a
    choice the deploy would refuse; each image carries the form's kernel note
    for this model's weights in it (:func:`form_kernel_note`: what the image
    does with 4-bit weights and "fetch the model to check" before the config
    is read, the fast or slow kernel after it). The
    default image is listed first. ``cache_note`` is :func:`cache_note` for
    the image the form has chosen and the machines it has selected.
    """
    profile = profile_for(repo, facts)
    gpu_nodes = list(gpu_nodes or ())
    has_head = profile.config_read and profile.mtp_layers > 0
    images = sorted(IMAGES.values(), key=lambda image: image.key != DEFAULT_IMAGE_KEY)
    return {
        "images": [
            {"key": image.key, "label": image.label,
             "kernel_note": form_kernel_note(profile, image)}
            for image in images
        ],
        "default_image": DEFAULT_IMAGE_KEY,
        "cache_note": cache_note(profile, image_profile(vllm_image), gpu_nodes),
        "mtp": {
            "available": has_head,
            "max_tokens": MAX_MTP_TOKENS,
            "suggested_tokens": MTP_SUGGESTED_TOKENS,
            "detail": MTP_OFFERED if has_head else (
                MTP_NO_HEAD if profile.config_read else MTP_NO_CONFIG
            ),
        },
        "text_only": {
            "applies": profile.vision,
            "detail": TEXT_ONLY_OFFERED if profile.vision else "",
        },
        # VD-213: the Advanced section's choices and defaults for this model
        # on the image the form has chosen (`vllm_serve_advanced`).
        "advanced": advanced_choices(repo, image_profile(vllm_image), gpu_nodes),
    }


def container_environment(value: Any) -> List[str]:
    """The container variables ``value`` adds: the tuned-table folder for an MoE
    model, then the newer runner's variable (VD-213)."""
    options = require_options(value)
    overlay = overlay_of(options)
    kernel = moe_kernel_of(options)
    return ([TUNED_FOLDER_ENV] if options.tuned_moe else []) + advanced_environment(
        options.advanced) + (
        [] if overlay is None else overlay_environment(overlay, options.advanced[3])) + (
        [] if kernel is None else kernel_environment(kernel))


def overlay_of(value: Any) -> Optional[VllmOverlay]:
    """The overlay a unit rendered from ``value`` mounts (VD-215), or ``None``."""
    options = require_options(value)
    return require_overlay(
        options.attention_overlay, image=image_profile(options.vllm_image),
        drafter=options.advanced[2],
    )


def moe_kernel_of(value: Any) -> Optional[VllmMoeKernel]:
    """Vaelor's expert-kernel plugin a unit rendered from ``value`` mounts (VD-216), or ``None``."""
    options = require_options(value)
    return require_moe_kernel(options.moe_kernel, image=image_profile(options.vllm_image))


def guesser_of(value: Any) -> Optional[GuesserBuild]:
    """The 4-bit guesser build a unit rendered from ``value`` mounts (VD-216 item 2), or ``None``."""
    options = require_options(value)
    return require_guesser(options.guesser, image=image_profile(options.vllm_image),
                           drafter=options.advanced[2])


def wants_tuned_tables(value: Any) -> bool:
    """Whether a unit rendered from ``value`` starts through the entry program
    that links vLLM's packaged tuned MoE tables (`vllm_entry_program`)."""
    return require_options(value).tuned_moe
