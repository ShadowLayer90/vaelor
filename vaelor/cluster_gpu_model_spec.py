"""The model facts a GPU fit decision is made from, validated once.

Housed out of `cluster_gpu_sizing` for the 1,000-line ceiling `CLAUDE.md` sets
when the throughput intent's ``replicated`` verdict (VD-129) landed there, and
a coherent job on its own: turn a request body - the catalog's
`gpu_model_catalog.catalog_spec_body`, or a pasted Hugging Face repo's
geometry - into a :class:`GpuModelSpec` the engine can do arithmetic on, or
refuse it with the field that is wrong. The engine imports everything here and
re-exports the names its callers already use, so nothing above it moved.

**Every number is derivable and documented.** The KV-cache formula is
`model_sizing.kv_cache_bytes` (the same one the single-node planner uses),
applied to the layers that hold a KV cache - all of them, unless the model's
own ``config.json`` (``hf_config``, read by `vllm_model_profile`) marks some as
linear attention, whose fixed per-sequence state is counted instead; the
bytes-per-parameter table is nominal GGUF bit-widths, used only when a measured
weight byte size is not supplied; and what the runtime needs beyond weights
and KV is a small fraction of the weights plus a fixed reserve, calibrated on
the vLLM 0.27 benchmark (:data:`WEIGHT_OVERHEAD_FRACTION`). A caller that has
a measured artifact size (the catalog's `download_bytes`) passes it and the
estimate is never reached.
"""

from __future__ import annotations

from dataclasses import dataclass, replace
from typing import Any, Dict, Iterable, Optional

from .model_sizing import GIB, kv_cache_bytes
#: A pasted repo's ``config.json`` is read by the one owner of model facts.
from .vllm_model_profile import ModelFacts, facts_from_hf_config, geometry_from_hf_config


#: What a served model needs beyond its weight files and its KV cache
#: (coordinator decision 2026-09-30, replacing "20% of weights plus KV", which
#: refused contexts that were measured to start: it charged a fifth of the KV
#: cache, and nothing in the runtime grows with the cache).
#:
#: **Calibrated on what vLLM 0.27 itself reported on the pair** (the 0.27
#: benchmark, `bench_data.json`): at each start vLLM logs the KV memory it
#: found, which is its budget (``gpu_memory_utilization`` x the pool) less
#: everything else it holds. "Beyond the weights" below is that remainder less
#: the checkpoint's weight FILES, in GiB:
#:
#: ====================  ======  =======  ========  ===========  =========
#: model (cold start)    pool    weights  KV found  beyond them  this rule
#: ====================  ======  =======  ========  ===========  =========
#: Qwen3-8B BF16         30.4    15.26    6.39      2.67         2.20
#: Qwen3-8B AWQ          30.4    5.68     15.94     2.70         2.01
#: Qwen3-8B w4a16        30.4    5.68     16.76     1.88         2.01
#: Qwen3-30B-A3B w4a16   44.0    16.85    16.13     2.22         2.24
#: Qwen3.8-27B, bf16 KV  44.0    18.12    14.09     2.99         3.89
#: Qwen3.8-27B, 2 drafts 44.0    18.12    11.43     5.65         5.89
#: ====================  ======  =======  ========  ===========  =========
#:
#: The same three 8B models started warm (compile cache filled) with 0.66 to
#: 0.93 GiB beyond their weights, so most of a cold start's figure is a
#: compile transient.
#:
#: **The rule is anchored on the 30B, the recommended model, and on vLLM's
#: own words.** Asked to start it at its full 262,144-token context on one
#: 44 GiB machine, vLLM refused and said the longest context it could hold
#: there was 175,984 tokens (16.11 GiB of KV). The reserve is set so that
#: this rule's longest context for that model on that pool is that number -
#: it admits 175,984 and refuses 176,000 - rather than the 136,000 the old
#: rule stopped at. It is under the two cold 8B rows above by up to 0.7 GiB
#: (about 5,000 tokens of an 8B's cache), which a start at the very brim of
#: the budget would meet as vLLM's own "context too long for the KV cache"
#: refusal; it is over every other row.
#:
#: The fraction is the little by which loaded weights exceed their files (the
#: 30B loaded 17.1 GiB from 16.85). The reserve is the runtime itself:
#: compiled graphs, activations, the allocator.
WEIGHT_OVERHEAD_FRACTION = 0.02
RUNTIME_RESERVE_BYTES = 1944 * 1024 * 1024
#: A model whose weight size was ESTIMATED (no measured bytes) is not one of
#: the rows above: it keeps the largest reserve any plain model was measured
#: to need, on top of an estimate that already errs high.
UNMEASURED_RUNTIME_RESERVE_BYTES = int(2.75 * GIB)
#: A hybrid (Gated DeltaNet) model captures many more graphs, and drafting
#: with its prediction head more again: the two 27B rows.
HYBRID_RESERVE_BYTES = int(1.625 * GIB)
DRAFTING_RESERVE_BYTES = int(2 * GIB)
#: The share of the context the fit will size to (coordinator decision
#: 2026-09-30, after the re-review): it offers and admits at most about 97%
#: of the longest context its arithmetic computes, so an offer is never the
#: brim of the budget - where a compile transient, or a runtime older than the
#: one this rule was calibrated on (vLLM 0.27), would meet vLLM's own "context
#: too long for the KV cache" refusal. Charged as its own line of the budget
#: (``margin_bytes``): the KV cache divided by this, less the KV cache.
CONTEXT_MARGIN = 0.97

#: What a hybrid model needs of the POOL - not of its launch fraction - while
#: it starts: its weights plus this. Measured, the 27B (16.7 GiB loaded) ran
#: a 30.4 GiB pool out of memory while vLLM profiled its graphs, twice, at a
#: 16k context whose steady need was about 23 GiB; it started in 44 GiB. The
#: figure sits between the pool that failed (about 13.7 GiB beyond the loaded
#: weights) and the one that worked (27.3), nearer the failure: nothing in
#: between was measured, and a machine with 40 GiB should not be refused on a
#: guess. The fit used to admit this model at 32k on the pool it died in.
HYBRID_STARTUP_BYTES = 16 * GIB

#: Default KV-cache slots for a fit decision: one. The question this engine
#: answers is "does the model fit at all", and the smallest honest KV budget is
#: one slot holding the whole context window - the Assistant's single-owner
#: shape (`model_sizing.ASSISTANT_PARALLEL_SLOTS`). A served endpoint that will
#: run more slots passes its own count; the KV cost then scales with it exactly
#: as `kv_cache_bytes` multiplies by ``parallel``.
DEFAULT_KV_SLOTS = 1

#: Nominal **bytes per weight parameter** by quantization, used ONLY when a
#: measured weight byte size is not supplied. Each value is the format's nominal
#: bit-width divided by eight (e.g. ``q4_k_m`` ~4.83 bits -> ~0.604 bytes). They
#: are approximations: two publishers' Q4_K_M builds of one model differ by
#: ~1,056 bytes (VD-065), and a real GGUF carries per-tensor mixed quantization
#: these single figures cannot capture. So the honest input is `weight_bytes`
#: read off the artifact (the catalog's digest-verified `download_bytes`); this
#: table is the fallback that lets a bare "4B at Q4" still produce a legible,
#: deliberately slightly-high estimate rather than nothing.
BYTES_PER_PARAMETER: Dict[str, float] = {
    "f32": 4.0,
    "f16": 2.0,
    "bf16": 2.0,
    "q8_0": 8.5 / 8,
    "q6_k": 6.5625 / 8,
    "q5_k_m": 5.5 / 8,
    "q5_1": 6.0 / 8,
    "q5_0": 5.5 / 8,
    "q4_k_m": 4.83 / 8,
    "q4_k_s": 4.5 / 8,
    "q4_1": 5.0 / 8,
    "q4_0": 4.5 / 8,
    "iq4_nl": 4.5 / 8,
    "q4nx": 4.5 / 8,
    "mxfp4": 4.25 / 8,
    "rocmfp4": 4.25 / 8,
    "q3_k_m": 3.9 / 8,
    "q2_k": 3.35 / 8,
}


def normalize_quantization(value: Any) -> str:
    """A quantization label folded to the ``BYTES_PER_PARAMETER`` key shape."""
    return str(value or "").strip().lower().replace("-", "_")


def bytes_per_parameter(quantization: str) -> Optional[float]:
    """Nominal bytes per weight for a quantization, or ``None`` if unknown.

    ``None`` rather than a guessed default: an unrecognised quantization with no
    measured `weight_bytes` is a spec this engine cannot size, and saying so is
    the honest-degradation rule the whole capacity foundation follows.
    """
    return BYTES_PER_PARAMETER.get(normalize_quantization(quantization))


@dataclass(frozen=True)
class GpuModelSpec:
    """The model facts a GPU fit decision needs, sourced the catalog's way.

    ``weight_bytes`` is the resident weight size. Prefer the measured artifact
    size (`model_catalog` entries carry a digest-verified ``download_bytes``);
    it is only derived from ``parameter_billions`` x `bytes_per_parameter` when
    a measured size is absent, and ``weight_estimated`` records which happened so
    the verdict can say so.

    ``head_dim`` defaults to ``hidden_size // attention_heads`` when a hidden
    size is given, because that identity is how a transformer's per-head width
    is defined; a spec that supplies neither is refused by `build_model_spec`
    rather than sized against a guessed head width.

    ``kv_layers`` is how many layers hold a KV cache: ``None`` for all of
    ``hidden_layers``, fewer for a hybrid model whose other layers are linear
    attention, each of those holding ``recurrent_state_bytes`` (summed, per
    sequence) that does not grow with the context. ``facts`` is what the
    model's ``config.json`` said, when the deploy carried one - the parsers,
    vision tower and MTP head the launch reads (`vllm_model_profile`).

    ``kv_block_tokens`` is the size of the blocks the KV cache is kept in when
    that size matters to the total: zero for the 16-token blocks of a plain
    model, and the block a hybrid model's state page forces when it is
    launched with a bf16 KV cache (:func:`with_launch_cache`). A context is
    then held in whole blocks.
    """

    name: str
    weight_bytes: int
    hidden_layers: int
    attention_heads: int
    kv_heads: int
    head_dim: int
    context_length: int
    quantization: str = ""
    parameter_billions: float = 0.0
    slots: int = DEFAULT_KV_SLOTS
    weight_estimated: bool = False
    bytes_per_parameter: Optional[float] = None
    kv_layers: Optional[int] = None
    recurrent_state_bytes: int = 0
    facts: Optional[ModelFacts] = None
    kv_block_tokens: int = 0
    #: The draft tokens of multi-token prediction this launch runs with.
    draft_tokens: int = 0

    def kv_cache_bytes(self) -> int:
        """The minimum KV budget: the whole context window at ``slots`` slots.

        Over the layers that hold a KV cache, plus each slot's fixed
        linear-attention state for a hybrid model. The context is rounded up
        to whole blocks where the launch keeps it in big ones.
        """
        layers = self.hidden_layers if self.kv_layers is None else self.kv_layers
        tokens = int(self.context_length)
        if self.kv_block_tokens > 0:
            tokens = -(-tokens // self.kv_block_tokens) * self.kv_block_tokens
        return kv_cache_bytes(
            tokens,
            layers=layers,
            kv_heads=self.kv_heads,
            head_dim=self.head_dim,
            parallel=self.slots,
        ) + int(self.recurrent_state_bytes) * int(self.slots)

    def runtime_reserve_bytes(self) -> int:
        """What ONE serving process needs beyond weights and KV (module constants)."""
        reserve = (
            UNMEASURED_RUNTIME_RESERVE_BYTES if self.weight_estimated
            else RUNTIME_RESERVE_BYTES
        )
        if self.recurrent_state_bytes > 0:
            reserve += HYBRID_RESERVE_BYTES
        if self.draft_tokens > 0:
            reserve += DRAFTING_RESERVE_BYTES
        return reserve

    def startup_pool_bytes(self, stages: int = 1) -> int:
        """The pool one machine needs while the model STARTS; 0 for a plain model.

        A hybrid model's start needs far more than its steady state
        (:data:`HYBRID_STARTUP_BYTES`); split across ``stages`` machines each
        holds its share of the weights.
        """
        if self.recurrent_state_bytes <= 0:
            return 0
        return -(-int(self.weight_bytes) // max(1, int(stages))) + HYBRID_STARTUP_BYTES

    def required_bytes(self, stages: int = 1) -> Dict[str, int]:
        """Weights + KV + overhead, and each part, all as integers of bytes.

        The overhead is :data:`WEIGHT_OVERHEAD_FRACTION` of the weights plus
        one runtime reserve for each of the ``stages`` processes that serve
        the model: one for a whole copy, one per machine for a split. The
        margin is :data:`CONTEXT_MARGIN`'s share of the cache.
        """
        weights = int(self.weight_bytes)
        kv = int(self.kv_cache_bytes())
        overhead = (
            int(weights * WEIGHT_OVERHEAD_FRACTION)
            + max(1, int(stages)) * self.runtime_reserve_bytes()
        )
        margin = int(kv * (1 / CONTEXT_MARGIN - 1))
        return {
            "weight_bytes": weights,
            "kv_cache_bytes": kv,
            "overhead_bytes": overhead,
            "margin_bytes": margin,
            "required_bytes": weights + kv + overhead + margin,
        }


def with_speculation(spec: GpuModelSpec, mtp_tokens: Any) -> GpuModelSpec:
    """``spec`` sized for multi-token prediction with ``mtp_tokens`` draft tokens.

    The owner's choice is checked by the one rule the deploy uses
    (`vllm_serve_options.requested_mtp_tokens`), so the fit preview refuses
    what the deploy would. Then the two things MTP adds to one sequence's
    memory, as vLLM sizes them (read at tag v0.27.0): the checkpoint's MTP
    layers are full-attention layers with their own KV cache, and a hybrid
    model keeps its linear-attention state for every speculative token as well
    as the accepted one - ``MambaSpec.max_memory_usage_bytes`` is the state's
    page times ``1 + num_speculative_blocks`` in the default ``none`` cache
    mode, with ``num_speculative_blocks`` the speculative token count
    (``vllm/v1/kv_cache_interface.py``, ``layers/mamba/abstract.py``). The
    draft layer's weights are already in a measured ``weight_bytes`` that
    counts the checkpoint's MTP file.
    """
    from .vllm_serve_options import requested_mtp_tokens

    facts = spec.facts
    tokens = requested_mtp_tokens(mtp_tokens, facts.mtp_layers if facts else None)
    if not tokens:
        return spec
    layers = spec.hidden_layers if spec.kv_layers is None else spec.kv_layers
    return replace(
        spec,
        kv_layers=layers + facts.mtp_layers,
        recurrent_state_bytes=spec.recurrent_state_bytes * (1 + tokens),
        draft_tokens=tokens,
    )


#: Bytes of one bf16 element, and the multiple vLLM rounds a hybrid model's
#: attention block up to (``block_alignment_bytes`` in
#: ``HybridAttentionMambaModelConfig``, read at tag v0.27.0).
_BF16_BYTES = 2
_BLOCK_ALIGNMENT_TOKENS = 16


def with_launch_cache(
    spec: GpuModelSpec, mtp_tokens: Any, *, vllm_image: Any = None,
    gpu_targets: Iterable[Any] = (),
) -> GpuModelSpec:
    """``spec`` sized for the cache this launch will actually keep.

    The ONE sizing call the fit preview and the deploy both make, with the
    same three inputs the launch is decided from: the multi-token-prediction
    choice (:func:`with_speculation`), the image (``None`` for the default)
    and each selected machine's ``gfx_target_version``. Whether the hybrid
    cache flags apply is `vllm_serve_options.hybrid_cache`'s answer - the
    rule the launch itself uses - so the fit cannot size a cache the unit
    does not ask for.

    **With those flags, the layout is vLLM 0.27's** (read at tag v0.27.0:
    ``vllm/model_executor/models/config.py`` and ``MambaSpec`` in
    ``vllm/v1/kv_cache_interface.py``), at a bf16 KV:

    * the attention block is the smallest multiple of 16 tokens whose page
      (one layer's K and V) covers one linear-attention layer's state page,
      so the context is held in whole blocks of that size - 784 tokens for
      Qwen3.8-27B, 800 with two draft tokens, as the server logged;
    * each linear-attention layer's state page is padded to that attention
      page, and a sequence keeps two of them with the aligned prefix cache
      (``mamba_cache_mode == "align"``: ``2 + num_speculative_blocks``), or
      one plus one for each draft token without it.

    Without the flags the spec is :func:`with_speculation`'s, unchanged.
    """
    from .vllm_images import image_profile
    from .vllm_serve_options import hybrid_cache

    sized = with_speculation(spec, mtp_tokens)
    linear = spec.facts.linear if spec.facts is not None else None
    tokens = int(mtp_tokens or 0)
    bf16, aligned = hybrid_cache(
        linear is not None, image_profile(vllm_image), gpu_targets, tokens,
    )
    if not bf16:
        return sized
    token_layer = 2 * spec.kv_heads * spec.head_dim * _BF16_BYTES
    step = _BLOCK_ALIGNMENT_TOKENS * token_layer
    block = _BLOCK_ALIGNMENT_TOKENS * -(-linear.page_bytes(tokens) // step)
    pages = (2 if aligned else 1) + tokens
    return replace(
        sized, kv_block_tokens=block,
        recurrent_state_bytes=linear.layers * block * token_layer * pages,
    )


#: The block vLLM keeps a plain model's KV cache in, and so the step a
#: fitting context is offered in; a hybrid model's is its own (784 tokens for
#: the 27B at a bf16 cache, `with_launch_cache`).
_PLAIN_BLOCK_TOKENS = 16


def context_that_fits(spec: GpuModelSpec, budget: int) -> int:
    """The longest context, in whole KV blocks, that ``budget`` bytes hold; 0 for none.

    Found by asking the spec itself (`GpuModelSpec.required_bytes` grows with
    the context), so it is the fit's own arithmetic - its margin included, so
    an offer is at most about :data:`CONTEXT_MARGIN` of the brim - and a
    context it offers is one the same fit then accepts.
    """
    step = int(spec.kv_block_tokens) or _PLAIN_BLOCK_TOKENS

    def holds(tokens: int) -> bool:
        sized = replace(spec, context_length=tokens)
        return sized.required_bytes()["required_bytes"] <= budget

    low, high = 0, int(spec.context_length) // step
    while low < high:
        middle = (low + high + 1) // 2
        if holds(middle * step):
            low = middle
        else:
            high = middle - 1
    return low * step


def _positive_int(value: Any, field: str) -> int:
    try:
        number = int(value)
    except (TypeError, ValueError):
        raise ValueError("'{}' must be a whole number.".format(field))
    if number <= 0:
        raise ValueError("'{}' must be greater than zero.".format(field))
    return number


def build_model_spec(spec: Dict[str, Any]) -> GpuModelSpec:
    """Validate a request body into a `GpuModelSpec`, or raise ``ValueError``.

    Defensive on every field so the route can turn a bad body into a 400 with
    the reason, never a stack trace: the transformer geometry
    (``hidden_layers``, ``attention_heads``, ``kv_heads``, ``context_length``)
    is required and must be positive; the weight size is either measured
    (``weight_bytes``) or derived from ``parameter_billions`` and a known
    ``quantization``, and a spec that provides neither path is refused with what
    it is missing rather than sized against a guess.
    """
    if not isinstance(spec, dict):
        raise ValueError("A model spec object is required.")
    facts: Optional[ModelFacts] = None
    if spec.get("hf_config") is not None:
        # The repo's own config.json: it fills every geometry field the body
        # left out (a field the body states still wins), and says which layers
        # hold a KV cache and what the model carries besides.
        facts = facts_from_hf_config(spec["hf_config"])
        stated = {key: value for key, value in spec.items() if value is not None}
        spec = {**geometry_from_hf_config(spec["hf_config"]), **stated}

    hidden_layers = _positive_int(spec.get("hidden_layers"), "hidden_layers")
    attention_heads = _positive_int(
        spec.get("attention_heads"), "attention_heads"
    )
    kv_heads = _positive_int(spec.get("kv_heads"), "kv_heads")
    context_length = _positive_int(spec.get("context_length"), "context_length")
    if kv_heads > attention_heads:
        raise ValueError(
            "'kv_heads' cannot exceed 'attention_heads' ({} > {}).".format(
                kv_heads, attention_heads
            )
        )
    if attention_heads % kv_heads != 0:
        # A valid grouped-query geometry divides query heads evenly among the
        # KV heads; without this a tensor-parallel split that clears the KV-head
        # test could still leave query heads that do not divide across nodes.
        raise ValueError(
            "'attention_heads' must be a multiple of 'kv_heads' "
            "({} % {} != 0).".format(attention_heads, kv_heads)
        )

    head_dim = spec.get("head_dim")
    hidden_size = spec.get("hidden_size")
    if head_dim is not None:
        head_dim = _positive_int(head_dim, "head_dim")
    elif hidden_size is not None:
        head_dim = _positive_int(hidden_size, "hidden_size") // attention_heads
        if head_dim <= 0:
            raise ValueError(
                "'hidden_size' is smaller than 'attention_heads', so the "
                "per-head width would be zero."
            )
    else:
        raise ValueError(
            "Provide 'head_dim', or 'hidden_size' to derive it from "
            "'attention_heads'."
        )

    quantization = str(spec.get("quantization", "") or "")
    raw_params = spec.get("parameter_billions", 0.0)
    try:
        parameter_billions = float(raw_params) if raw_params else 0.0
    except (TypeError, ValueError):
        raise ValueError("'parameter_billions' must be a number.")
    slots = spec.get("slots", DEFAULT_KV_SLOTS)
    slots = _positive_int(slots if slots is not None else DEFAULT_KV_SLOTS, "slots")
    kv_layers = facts.kv_layers if facts is not None else None
    if spec.get("kv_layers") is not None:
        kv_layers = _positive_int(spec.get("kv_layers"), "kv_layers")
    if kv_layers is not None and kv_layers > hidden_layers:
        raise ValueError(
            "'kv_layers' cannot exceed 'hidden_layers' ({} > {}).".format(
                kv_layers, hidden_layers
            )
        )

    weight_estimated = False
    per_parameter: Optional[float] = None
    raw_weight = spec.get("weight_bytes")
    if raw_weight is not None:
        weight_bytes = _positive_int(raw_weight, "weight_bytes")
    else:
        if parameter_billions <= 0:
            raise ValueError(
                "Provide 'weight_bytes', or 'parameter_billions' and "
                "'quantization' to derive the weight size."
            )
        per_parameter = bytes_per_parameter(quantization)
        if per_parameter is None:
            raise ValueError(
                "No measured 'weight_bytes' and quantization '{}' is not in the "
                "bytes-per-parameter table, so the weight size cannot be "
                "derived. Supply 'weight_bytes' or a known quantization.".format(
                    quantization or "(empty)"
                )
            )
        weight_bytes = int(parameter_billions * 1e9 * per_parameter)
        weight_estimated = True
        if weight_bytes <= 0:
            raise ValueError("The derived weight size is zero; check the spec.")

    return GpuModelSpec(
        name=str(spec.get("name", "") or ""),
        weight_bytes=weight_bytes,
        hidden_layers=hidden_layers,
        attention_heads=attention_heads,
        kv_heads=kv_heads,
        head_dim=head_dim,
        context_length=context_length,
        quantization=quantization,
        parameter_billions=parameter_billions,
        slots=slots,
        weight_estimated=weight_estimated,
        bytes_per_parameter=per_parameter,
        kv_layers=kv_layers,
        recurrent_state_bytes=facts.recurrent_state_bytes if facts is not None else 0,
        facts=facts,
    )
