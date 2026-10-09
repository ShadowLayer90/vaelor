"""What a cluster model IS, read off its Hugging Face ``config.json``: the one owner.

The vLLM cluster serves whatever repo the owner points it at, and four things
the launch and the fit need are facts about that model, not choices: which
tool-call and reasoning parsers its chat template speaks, how many of its
layers hold a KV cache, and whether it carries a vision tower or a
multi-token-prediction (MTP) head. Until this module the launch hard-coded
``--tool-call-parser hermes`` for every model and the fit counted a KV cache on
every layer. Both are wrong for the Qwen3.5-architecture models (Qwen3.5, 3.6
and 3.8): their tool calls are XML (``qwen3_xml``), and three of every four
layers are Gated-DeltaNet linear attention, holding a small fixed recurrent
state instead of a KV cache that grows with the context.

**Where the facts come from, in order.**

1. The repo's own ``config.json``, when the deploy carries it (``model_spec``'s
   ``hf_config``): ``model_type``, ``layer_types``, ``vision_config``,
   ``mtp_num_hidden_layers`` and the expert count are read straight off it by
   :func:`facts_from_hf_config`. It is the file vLLM itself reads.
2. Otherwise - only when NO config was read - the repo NAME, for the Qwen
   families only (:func:`family_from_name`), which is enough to pick parsers
   and never enough to claim a vision tower, an MTP head or a hybrid layout:
   those stay unknown and are treated as absent. A config that was read and
   names a ``model_type`` outside :data:`FAMILIES` is believed over the name:
   the default family, whatever the repo is called.
3. Otherwise nothing: an unknown family is served exactly as before, with the
   ``hermes`` parser and no reasoning parser (:data:`DEFAULT_FAMILY`).

**What the config cannot say, the name does.** A Qwen3 ``-Instruct-2507`` and
the hybrid-thinking Qwen3 it was split from have the same ``config.json``
shape; only the chat template differs. So the name refines the family
(:func:`profile_for`). An ``-Instruct-2507`` (and a base, embedding, reranker
or guard model) gets NO reasoning parser: vLLM 0.22.1's
``Qwen3ReasoningParser`` reads output with no ``</think>`` as unfinished
reasoning while thinking is on (its default), which would move a non-thinking
model's whole answer out of ``content``. A ``Coder`` gets the ``qwen3_coder``
tool parser its card names. And whether the template takes the
``enable_thinking`` switch at all (:func:`has_thinking_switch`) is read off
the name alone: the switch lives in the official Qwen chat template, which no
``config.json`` describes, so a same-architecture fine-tune under another name
is not promised one.

**One family table.** :data:`FAMILIES` is the one answer to "which Qwen family
is this and what does its template speak" - the parsers here, and the thinking
switch `model_thinking` puts a default on. A family is added here, once its
template has been read, and nowhere else.

Every parser name here was read in vLLM 0.22.1's registries at the pinned tag
(``vllm/tool_parsers/__init__.py`` and ``vllm/reasoning/__init__.py`` at
commit ``0decac0``) and found in the served image (the 2026-09-29 model
shortlist); the root bridge refuses any other (`vllm_serve_options`). The
0.27 benchmark (2026-09-30) ran the matrix both ways on the 0.27 image: a
Qwen3 served with ``qwen3_xml``, or a Qwen3.8 served with ``hermes``, returned
its tool call as plain text, and lost it altogether when streaming.

**A hybrid model's linear-attention state is kept in parts**
(:class:`LinearState`), because vLLM sizes its cache from them: the attention
block is as many tokens as it takes to cover one layer's state page, and the
convolution state grows by one step for every draft token of multi-token
prediction (`cluster_gpu_model_spec.with_launch_cache`).
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from typing import Any, Dict, Mapping, Optional

#: The tool-call parsers a Vaelor unit may name, each registered in vLLM 0.22.1.
#: ``hermes`` is Qwen3's ``<tool_call>{json}</tool_call>``; ``qwen3_xml`` and
#: ``qwen3_coder`` read the XML ``<function=...><parameter=...>`` form the
#: Qwen3.5-architecture and Qwen3-Coder templates emit.
TOOL_CALL_PARSERS = frozenset({"hermes", "qwen3_xml", "qwen3_coder"})

#: The reasoning parsers a Vaelor unit may name. vLLM 0.22.1's ``qwen3`` covers
#: Qwen3 and Qwen3.5 alike: ``<think>`` generated, or placed in the prompt.
REASONING_PARSERS = frozenset({"qwen3"})

#: The parser a family whose template nobody has read is served with. It is
#: what every unit carried before this module, so an unknown model launches
#: exactly as it did rather than on a guess.
DEFAULT_TOOL_CALL_PARSER = "hermes"

#: The largest ``config.json`` accepted, serialised. Qwen3.8-27B-INT4's is
#: about 21 KB, most of it the quantization ``ignore`` list.
_MAX_CONFIG_BYTES = 256 * 1024

#: Layer kinds that hold a KV cache (``sliding_attention`` holds a windowed
#: one, counted whole: the fit errs high, never low), and the kind that holds
#: a fixed recurrent state instead.
_KV_LAYER_KINDS = frozenset({"full_attention", "sliding_attention", "attention"})
_LINEAR_LAYER_KIND = "linear_attention"

#: Bytes per element of the recurrent state, by the config's
#: ``mamba_ssm_dtype`` (bf16 when the config names none), and of the
#: convolution state, which is kept in the model's own 16-bit dtype.
_STATE_DTYPE_BYTES = {"float32": 4, "float16": 2, "bfloat16": 2}
_CONV_STATE_BYTES = 2


@dataclass(frozen=True)
class Family:
    """The parsers one model family's chat template speaks."""

    name: str
    tool_call_parser: str
    reasoning_parser: Optional[str]
    thinking_switch: bool = False


#: The families whose templates were read, keyed by the HF ``model_type``.
FAMILIES: Dict[str, Family] = {
    "qwen3": Family("qwen3", "hermes", "qwen3", thinking_switch=True),
    "qwen3_moe": Family("qwen3_moe", "hermes", "qwen3", thinking_switch=True),
    "qwen3_5": Family("qwen3_5", "qwen3_xml", "qwen3", thinking_switch=True),
    "qwen3_5_moe": Family("qwen3_5_moe", "qwen3_xml", "qwen3", thinking_switch=True),
}
DEFAULT_FAMILY = Family("", DEFAULT_TOOL_CALL_PARSER, None)

#: The prefix every Qwen3-family release name starts with: ``Qwen3-8B`` is
#: Qwen3; ``Qwen3.5-4B``, ``Qwen3.6-35B-A3B`` and ``Qwen3.8-27B-INT4`` (a
#: version after the dot) are the Qwen3.5 architecture (`_name_family`).
_QWEN3_PREFIX = "qwen3"
#: Variants whose template takes no reasoning parser, or whose tools are the
#: Coder XML - both visible only in the name (module docstring).
_NO_REASONING_MARKERS = (
    "-instruct-2507", "-base", "embedding", "reranker", "guard", "-next-", "-omni-",
)
_CODER_MARKER = "-coder"
#: Variants whose template carries no ``enable_thinking`` switch: the 2507
#: split releases (``-Instruct-2507`` never thinks, ``-Thinking-2507`` always
#: does) and the coder, vision, base, embedding, reranker and guard models.
_NO_SWITCH_MARKERS = (
    "-instruct-2507", "-thinking-2507", "coder", "-vl-", "-base",
    "embedding", "reranker", "guard", "-next-", "-omni-",
)
#: A Qwen vision-language release names itself ``-VL-``; its ``model_type``
#: (``qwen3_vl``) is not a family read here, so the default parsers apply.
_VISION_NAME_MARKER = "-vl-"


@dataclass(frozen=True)
class LinearState:
    """One sequence's Gated-DeltaNet state, in the parts vLLM sizes it from.

    ``state_bytes`` is one layer's recurrent state; ``conv_step_bytes`` is one
    kept step of its causal-convolution state, and ``conv_steps`` how many the
    model keeps (``kernel - 1``). vLLM keeps one more step for each draft
    token of multi-token prediction, so a layer's page grows with them.
    """

    layers: int
    state_bytes: int
    conv_step_bytes: int
    conv_steps: int

    def page_bytes(self, draft_tokens: int = 0) -> int:
        """One layer's state for one sequence, with ``draft_tokens`` drafted."""
        return self.state_bytes + self.conv_step_bytes * (self.conv_steps + draft_tokens)


@dataclass(frozen=True)
class ModelFacts:
    """What one ``config.json`` says about a model.

    ``kv_layers`` is ``None`` when every layer holds a KV cache or the layout
    is unknown - the fit then counts all ``hidden_layers``, as it always did.
    ``recurrent_state_bytes`` is one sequence's fixed linear-attention state,
    and ``linear`` the same state in its parts - ``None`` for a model with no
    linear-attention layer, which is what "not a hybrid model" means here.
    """

    model_type: str = ""
    kv_layers: Optional[int] = None
    recurrent_state_bytes: int = 0
    vision: bool = False
    mtp_layers: int = 0
    experts: int = 0
    #: The weights' 4-bit format as the kernels see it (:func:`weight_format`).
    weight_format: str = ""
    linear: Optional[LinearState] = None


@dataclass(frozen=True)
class ModelProfile:
    """The family's parsers joined to the model's facts, for one deploy."""

    family: str
    tool_call_parser: str
    reasoning_parser: Optional[str]
    vision: bool
    mtp_layers: int
    moe: bool
    config_read: bool
    thinking_switch: bool = False
    weight_format: str = ""
    #: Whether the config marks linear-attention (Gated DeltaNet) layers. From
    #: the config alone: a name never claims a hybrid layout.
    hybrid: bool = False


def _repo_name(repo: Any) -> str:
    return str(repo or "").strip().rsplit("/", 1)[-1]


def _is_active_parameter_tag(word: str) -> bool:
    """``a3b``, ``a22b``, ``a3.5b``: the active-parameter tag only an MoE release carries."""
    middle = word[1:-1]
    return (
        len(word) > 2 and word[0] == "a" and word[-1] == "b"
        and middle.replace(".", "", 1).isdigit()
    )


def family_from_name(repo: Any) -> str:
    """The ``model_type`` a Qwen repo's name implies, or ``""`` for anything else.

    Read word by word (the name split on ``-``), with no pattern: the first
    word is ``qwen3`` or ``qwen3.<version>``, and an MoE release carries an
    active-parameter word such as ``A3B``.
    """
    lowered = _repo_name(repo).lower()
    if _VISION_NAME_MARKER in "-" + lowered + "-":
        return ""
    words = lowered.split("-")
    if len(words) < 2 or not words[0].startswith(_QWEN3_PREFIX):
        return ""
    version = words[0][len(_QWEN3_PREFIX):]
    moe = any(_is_active_parameter_tag(word) for word in words[1:])
    if version == "":
        return "qwen3_moe" if moe else "qwen3"
    if version[0] == "." and version[1:].isdigit():
        return "qwen3_5_moe" if moe else "qwen3_5"
    return ""


#: The Qwen3 versions whose published chat template was READ and found to take
#: ``enable_thinking`` (adversarial review, 2026-09-29: Qwen3-8B/32B/8B-FP8/
#: 30B-A3B/4B-AWQ, Qwen3.5-4B/27B, Qwen3.6-27B; and Qwen3.8-27B, whose
#: ``chat_template.jinja`` - the same file in RedHatAI's INT4 - reads
#: ``enable_thinking`` and ``preserve_thinking`` and writes ``<function=`` tool
#: calls, read 2026-09-29). ``""`` is Qwen3 itself. A newer version is treated
#: as having no switch until its template is read and added here.
_SWITCH_VERSIONS = ("", ".5", ".6", ".8")


def has_thinking_switch(repo: Any) -> bool:
    """Whether ``org/name``'s official chat template reads ``enable_thinking``.

    From the name alone (module docstring): a Qwen3 chat release of a version
    whose template was read (:data:`_SWITCH_VERSIONS`), minus the variants that
    carry no switch. `model_thinking` asks here; this is its one answer.
    """
    family = FAMILIES.get(family_from_name(repo))
    lowered = _repo_name(repo).lower()
    version = lowered.split("-", 1)[0][len(_QWEN3_PREFIX):]
    return (
        bool(family and family.thinking_switch) and version in _SWITCH_VERSIONS
        and not any(marker in lowered for marker in _NO_SWITCH_MARKERS)
    )


def _whole(value: Any, field: str, *, minimum: int = 0) -> int:
    """A config integer, refused rather than coerced: ``True`` is not a count."""
    if isinstance(value, bool) or not isinstance(value, int) or value < minimum:
        raise ValueError(
            "The model's config.json has an invalid '{}'.".format(field)
        )
    return value


def text_config(config: Mapping[str, Any]) -> Mapping[str, Any]:
    """The language model's half of a config: ``text_config`` when nested."""
    nested = config.get("text_config")
    return nested if isinstance(nested, Mapping) else config


#: The deepest a config.json nests (Qwen3.8's is 4: the root, ``text_config``,
#: ``rope_parameters`` and a list). Anything deeper is not a config, and a
#: walk over one is refused before any recursive call can meet it.
_MAX_NESTING = 16


def _nesting(value: Any) -> int:
    """How deeply ``value`` nests mappings and lists, walked without recursion."""
    deepest, stack = 0, [(value, 1)]
    while stack:
        item, depth = stack.pop()
        if isinstance(item, Mapping):
            children = list(item.values())
        elif isinstance(item, list):
            children = item
        else:
            continue
        deepest = max(deepest, depth)
        if deepest > _MAX_NESTING:
            return deepest
        stack.extend((child, depth + 1) for child in children)
    return deepest


def require_config(config: Any) -> Mapping[str, Any]:
    """``config`` when it is a ``config.json`` object of a sane size; refused otherwise."""
    refusal = "'hf_config' must be the model's config.json object."
    if not isinstance(config, Mapping) or _nesting(config) > _MAX_NESTING:
        raise ValueError(refusal)
    try:
        size = len(json.dumps(config))
    except (TypeError, ValueError, RecursionError):
        # A body nested deeper than the interpreter recurses is not a config.
        raise ValueError(refusal)
    if size > _MAX_CONFIG_BYTES:
        raise ValueError("'hf_config' is larger than any model config.json.")
    return config


def geometry_from_hf_config(config: Any) -> Dict[str, Any]:
    """The fit-geometry fields ``build_model_spec`` takes, read off a config.

    ``num_hidden_layers``, ``num_attention_heads``, ``num_key_value_heads``
    (the attention heads when absent: plain multi-head attention),
    ``head_dim`` or ``hidden_size``, and ``max_position_embeddings``. Each is
    left out when the config does not carry it, so the spec's own validation
    names the field that is missing.
    """
    text = text_config(require_config(config))
    fields = {
        "hidden_layers": text.get("num_hidden_layers"),
        "attention_heads": text.get("num_attention_heads"),
        "kv_heads": text.get("num_key_value_heads", text.get("num_attention_heads")),
        "head_dim": text.get("head_dim"),
        "hidden_size": text.get("hidden_size"),
        "context_length": text.get("max_position_embeddings"),
    }
    return {key: value for key, value in fields.items() if value is not None}


def _linear_state(text: Mapping[str, Any], linear_layers: int) -> Optional[LinearState]:
    """One sequence's Gated-DeltaNet state across ``linear_layers`` layers.

    Per layer: the recurrent state, ``value heads x key dim x value dim`` in
    ``mamba_ssm_dtype``, plus the causal-convolution state,
    ``(2 x key heads x key dim + value heads x value dim) x (kernel - 1)`` in
    16 bits - the two buffers vLLM's hybrid cache keeps for each sequence. A
    config that marks linear layers without these fields is refused: sizing
    them at zero is the optimistic guess the fit exists to avoid.
    """
    if linear_layers == 0:
        return None
    key_heads = _whole(text.get("linear_num_key_heads"), "linear_num_key_heads", minimum=1)
    value_heads = _whole(text.get("linear_num_value_heads"), "linear_num_value_heads", minimum=1)
    key_dim = _whole(text.get("linear_key_head_dim"), "linear_key_head_dim", minimum=1)
    value_dim = _whole(text.get("linear_value_head_dim"), "linear_value_head_dim", minimum=1)
    kernel = _whole(text.get("linear_conv_kernel_dim"), "linear_conv_kernel_dim", minimum=1)
    state_bytes = _STATE_DTYPE_BYTES.get(str(text.get("mamba_ssm_dtype") or "bfloat16"))
    if state_bytes is None:
        raise ValueError("The model's config.json has an invalid 'mamba_ssm_dtype'.")
    return LinearState(
        layers=linear_layers,
        state_bytes=value_heads * key_dim * value_dim * state_bytes,
        conv_step_bytes=(
            (2 * key_heads * key_dim + value_heads * value_dim) * _CONV_STATE_BYTES
        ),
        conv_steps=kernel - 1,
    )


def _layer_split(text: Mapping[str, Any]) -> tuple:
    """``(kv_layers, linear_layers)``: ``kv_layers`` is ``None`` for a plain transformer.

    ``layer_types`` is the authority; a hybrid config without it
    (``full_attention_interval`` alone) is read through the interval.
    """
    layers = text.get("num_hidden_layers")
    layer_types = text.get("layer_types")
    if layer_types is not None:
        if not isinstance(layer_types, list) or not all(
            isinstance(kind, str) for kind in layer_types
        ):
            raise ValueError("The model's config.json has an invalid 'layer_types'.")
        if layers is not None and len(layer_types) != _whole(
            layers, "num_hidden_layers", minimum=1,
        ):
            raise ValueError(
                "The model's config.json lists {} layer types for {} layers.".format(
                    len(layer_types), layers,
                )
            )
        unknown = sorted(set(layer_types) - _KV_LAYER_KINDS - {_LINEAR_LAYER_KIND})
        if unknown:
            raise ValueError(
                "The model's config.json has a layer kind Vaelor cannot size: "
                "{}.".format(", ".join(unknown))
            )
        linear = layer_types.count(_LINEAR_LAYER_KIND)
        return (len(layer_types) - linear if linear else None), linear
    interval = text.get("full_attention_interval")
    if interval is not None and layers is not None:
        interval = _whole(interval, "full_attention_interval", minimum=1)
        total = _whole(layers, "num_hidden_layers", minimum=1)
        if interval > 1:
            return total // interval, total - total // interval
    return None, 0


#: The 4-bit weight formats :func:`weight_format` names. ``w4a16-rdna`` is one
#: vLLM 0.27's RDNAHybridW4A16 kernel takes on gfx1151 (GPTQ or
#: compressed-tensors, symmetric or zero-point, no ``g_idx``, groups of 32, 64
#: or 128); ``w4a16-other`` is a 4-bit layout it refuses; ``awq`` is AutoAWQ,
#: which vLLM routes to its Triton AWQ kernels on ROCm in 0.22.1 and 0.27 alike.
FAST_W4A16 = "w4a16-rdna"
OTHER_W4A16 = "w4a16-other"
AWQ_FORMAT = "awq"
_FAST_GROUPS = frozenset({32, 64, 128})
#: The one compressed-tensors activation order that gives a layer a ``g_idx``
#: (``has_g_idx = actorder == ActivationOrdering.GROUP`` in vLLM 0.27's
#: ``compressed_tensors_wNa16.py``); ``weight``/``static`` reorder at quantize
#: time and load without one.
_GIDX_ORDER = "group"


def weight_format(config: Mapping[str, Any]) -> str:
    """The weights' 4-bit format as vLLM's kernels choose on, or ``""``."""
    quantization = config.get("quantization_config")
    if not isinstance(quantization, Mapping):
        quantization = text_config(config).get("quantization_config")
    if not isinstance(quantization, Mapping):
        return ""
    method = str(quantization.get("quant_method") or "").lower()
    if method == "awq":
        return AWQ_FORMAT
    if method == "gptq":
        fast = (
            quantization.get("bits") == 4 and not quantization.get("desc_act")
            and quantization.get("group_size") in _FAST_GROUPS
        )
        return FAST_W4A16 if fast else OTHER_W4A16
    if method != "compressed-tensors":
        return ""
    groups = quantization.get("config_groups")
    weights = [
        group.get("weights") for group in (groups or {}).values()
        if isinstance(group, Mapping) and isinstance(group.get("weights"), Mapping)
    ] if isinstance(groups, Mapping) else []
    four_bit = [w for w in weights if w.get("num_bits") == 4 and w.get("type") == "int"]
    if not four_bit:
        return ""
    fast = all(
        w.get("group_size") in _FAST_GROUPS and w.get("actorder") != _GIDX_ORDER
        for w in four_bit
    )
    return FAST_W4A16 if fast else OTHER_W4A16


def kernel_note(weights: str, fast_w4a16: bool) -> str:
    """One plain sentence on how fast this model's 4-bit weights will run, or ``""``.

    Measured on the Z2 on vLLM 0.22.1: 4-bit weights ran through Triton at
    about 30 GB/s where BF16 ran at about 210 GB/s, so a 4-bit model there can
    answer no faster than the full-size one. vLLM 0.27's RDNAHybridW4A16 is the
    fast path, for the formats :func:`weight_format` marks.
    """
    if weights == FAST_W4A16 and fast_w4a16:
        return ""
    if weights == FAST_W4A16:
        return (
            "This image runs 4-bit weights through a slow general kernel on "
            "this GPU; the vLLM 0.27 image has the fast one for this model."
        )
    if weights == AWQ_FORMAT:
        return (
            "AWQ weights run through vLLM's slow Triton AWQ kernel on this GPU "
            "with every image; a GPTQ or compressed-tensors 4-bit copy of the "
            "model runs the fast kernel in the vLLM 0.27 image."
        )
    if weights == OTHER_W4A16:
        return (
            "These 4-bit weights use a layout the fast kernel does not take "
            "(activation order, or a group size other than 32, 64 or 128), "
            "so they run through vLLM's slow general kernel on this GPU."
        )
    return ""


def facts_from_hf_config(config: Any) -> ModelFacts:
    """Everything the launch and the fit need from one ``config.json``."""
    config = require_config(config)
    text = text_config(config)
    kv_layers, linear_layers = _layer_split(text)
    linear = _linear_state(text, linear_layers)
    mtp = text.get("mtp_num_hidden_layers", text.get("num_nextn_predict_layers", 0))
    experts = text.get("num_experts", text.get("n_routed_experts", 0))
    return ModelFacts(
        model_type=str(config.get("model_type") or text.get("model_type") or ""),
        kv_layers=kv_layers,
        recurrent_state_bytes=linear.layers * linear.page_bytes() if linear else 0,
        linear=linear,
        vision=isinstance(config.get("vision_config"), Mapping),
        mtp_layers=_whole(mtp or 0, "mtp_num_hidden_layers"),
        experts=_whole(experts or 0, "num_experts"),
        weight_format=weight_format(config),
    )


def profile_for(repo: Any, facts: Optional[ModelFacts]) -> ModelProfile:
    """The one answer to "what does vLLM need to know about THIS model".

    ``facts`` is what the deploy's ``config.json`` said, ``None`` when it
    carried none. The family comes from its ``model_type``, else from the
    name; the name then refines the variant (module docstring). Vision and
    MTP come from the config alone. MoE comes from the config's expert count
    or the family, and a name's ``-A3B`` is enough for that one: the only
    thing it turns on (on the vLLM 0.27 image) is starting the container
    through the entry program that links vLLM's own tuned MoE tables
    (`vllm_entry_program`), which adds a moment to the start and changes
    nothing where no table matches.

    **The reasoning parser needs the NAME to vouch for the template**
    (review round 2, S2). vLLM's ``qwen3`` parser treats output with no
    ``</think>`` as unfinished reasoning while thinking is on, which is the
    template's own default unless Vaelor renders one - so a renamed or
    fine-tuned model whose config says ``qwen3`` but whose template never
    thinks would lose its whole answer into ``reasoning``. It is set only
    when the repo name is an official Qwen3-family release
    (:func:`family_from_name`) not marked as a non-thinking variant; a
    ``-Thinking-2507`` keeps it, since it always thinks.
    """
    read = facts is not None
    facts = facts or ModelFacts()
    # A config that was read is believed over the name (S1): a model_type
    # outside the table is the default family however the repo is called.
    model_type = facts.model_type if read else family_from_name(repo)
    family = FAMILIES.get(model_type, DEFAULT_FAMILY)
    lowered = _repo_name(repo).lower()
    tool_parser = family.tool_call_parser
    reasoning = family.reasoning_parser
    if family is not DEFAULT_FAMILY:
        if _CODER_MARKER in lowered:
            tool_parser, reasoning = "qwen3_coder", None
        elif any(marker in lowered for marker in _NO_REASONING_MARKERS):
            reasoning = None
    if not family_from_name(repo):
        reasoning = None
    return ModelProfile(
        family=family.name,
        tool_call_parser=tool_parser,
        reasoning_parser=reasoning,
        vision=facts.vision,
        mtp_layers=facts.mtp_layers,
        moe=facts.experts > 0 or model_type.endswith("_moe"),
        config_read=read,
        thinking_switch=has_thinking_switch(repo),
        weight_format=facts.weight_format,
        hybrid=facts.linear is not None,
    )
