"""Accelerator-aware model sizing, memory budgeting, and KV-cache planning.

Sizing used to be a single ladder over total system RAM. That is the right
answer for a CPU-only appliance and the wrong answer for anything with a GPU:
on a workstation with a 16 GiB VRAM carve-out and a 22 GiB GTT aperture it
recommended an 8 B model, and it sized the KV cache against 45 GiB of system
memory that the GPU cannot address.
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional

from .inference_tuning import (
    FLASH_ATTENTION_MODE,
    NPU_NATIVE_CONTEXT_TOKENS,
    NPU_RECOMMENDED_CONTEXT_TOKENS,
    RECOMMENDED_CONTEXT_TOKENS,
    npu_tier_plan,
    resolve_kv_cache_types,
)


GIB = 1024 ** 3

#: Bits per cache element. ``f16`` is llama.cpp's default and the only type
#: guaranteed to work on every backend.
KV_CACHE_BITS = {"f16": 16, "bf16": 16, "q8_0": 8, "q5_1": 6, "q5_0": 5.5, "q4_1": 4.5, "q4_0": 4}

#: Share of system memory kept for the operating system, Docker and the
#: control plane. GTT is carved out of system RAM, so an accelerator budget can
#: never legitimately exceed what is left after this reserve.
SYSTEM_RESERVE_FRACTION = 0.25

#: Conservative upper bound on KV cache size when GGUF metadata is unavailable.
#: Real values for common 7-30 B GQA models sit well below this at f16; using a
#: ceiling means the planner errs towards leaving memory free.
#:
#: **Per token of one slot.** The cache is paid for ``n_ctx × n_parallel``
#: times, and every function here that turns tokens into bytes therefore takes
#: the slot count. Nothing multiplied by it before, which is how a 4,096-token
#: request cost 16,384 tokens of KV on the Pi with no sizing surface aware of
#: it — see :data:`SLOT_MULTIPLICATION_MEASUREMENT`.
KV_BYTES_PER_TOKEN_F16 = 128 * 1024

#: The Assistant is one owner at one keyboard. A second slot buys concurrency
#: nobody is asking for and costs a second full window of KV, so this surface
#: runs one slot with the whole window in it.
ASSISTANT_PARALLEL_SLOTS = 1

#: The OpenAI-compatible endpoint is a different surface with a different
#: question behind it: ``LLAMA_ARG_HOST`` is ``0.0.0.0`` and
#: :mod:`vaelor.cluster_plan_inference` exists to place it on worker nodes, so
#: more than one caller at a time is the normal case rather than an accident.
#: Four is not a preference — it is the number this image already chose when
#: nothing stated one (below), now stated and, crucially, budgeted for.
SERVED_PARALLEL_SLOTS = 4

#: What an unstated slot count actually did, measured on the Pi appliance.
#:
#: Vaelor set ``LLAMA_ARG_CTX_SIZE=4096`` and no ``--parallel``. The server
#: logged ``srv load_model: initializing, n_slots = 4, n_ctx_slot = 4096,
#: kv_unified = 'true'`` — four slots each holding the *full* requested window,
#: 16,384 tokens of KV against the 4,096 that were asked for. Qwen3-1.7B is 28
#: layers, 8 KV heads, head_dim 128, so f16 KV costs 112 KiB per token: 1.88 GB
#: at 16,384 tokens, plus ~1.03 GB of weights, against a 2,816 MiB container
#: limit sized for one slot. The kernel's OOM record was 2,444,384 kB anonymous
#: plus 493,660 kB file — 2.87 GB — four minutes after the deploy reported
#: success.
#:
#: The container limit is *not* the defect and must not be raised to absorb
#: this: it is sized correctly for the window Vaelor requested.
SLOT_MULTIPLICATION_MEASUREMENT = {
    "requested_context_tokens": 4096,
    "observed_slots": 4,
    "observed_context_tokens": 16384,
    "model": "Qwen3-1.7B Q4",
    "layers": 28,
    "kv_heads": 8,
    "head_dim": 128,
    "kv_bytes_per_token_f16": 112 * 1024,
    "kv_bytes_observed": 16384 * 112 * 1024,
    "container_limit_mib": 2816,
    "oom_anon_kib": 2444384,
    "oom_file_kib": 493660,
}

#: Which surface gets how many slots, and why. Held as a table rather than as
#: one global setting because "how many callers at once" is a property of the
#: surface being served and not of the machine: the same appliance answers one
#: owner on the Assistant and a LAN full of clients on the served endpoint.
PARALLEL_SLOT_POLICY = {
    "assistant": (
        ASSISTANT_PARALLEL_SLOTS,
        "The Assistant serves one owner at one keyboard, so one slot holds the "
        "whole context window rather than a fraction of it, and no KV is "
        "allocated for concurrency nobody asked for.",
    ),
    "served": (
        SERVED_PARALLEL_SLOTS,
        "The OpenAI-compatible endpoint is reachable by more than one caller, "
        "so it is launched with {} slots — stated rather than inherited, and "
        "budgeted at {} times the context window of KV.".format(
            SERVED_PARALLEL_SLOTS, SERVED_PARALLEL_SLOTS
        ),
    ),
}


def parallel_slots(surface: str = "assistant") -> Dict[str, Any]:
    """How many slots this surface is launched with, and on what reasoning.

    **Always an answer, never an inherited default.** The launch that produced
    :data:`SLOT_MULTIPLICATION_MEASUREMENT` asked for a context window and said
    nothing about slots, so llama.cpp's own default applied and each of its four
    slots took the request in full. An unrecognised surface therefore falls back
    to the Assistant's single slot — the smaller claim on memory — and says that
    is what happened, rather than leaving the number to the runtime.
    """
    known = surface in PARALLEL_SLOT_POLICY
    slots, reason = PARALLEL_SLOT_POLICY.get(
        surface, PARALLEL_SLOT_POLICY["assistant"]
    )
    return {
        "surface": surface if known else "assistant",
        "requested_surface": surface,
        "slots": slots,
        "stated": True,
        "reason": reason if known else (
            "No surface named '{}' has a slot policy, so the single-slot "
            "Assistant policy was applied: the slot count is stated at launch "
            "either way, because an unstated one is what allocated four full "
            "windows on the appliance."
        ).format(surface),
    }

CPU_TIERS = (
    (6, "1.7B"),
    (11, "4B"),
    (float("inf"), "8B"),
)

ACCELERATOR_TIERS = (
    (6, "1.7B"),
    (11, "4B"),
    (18, "8B"),
    (28, "14B"),
    (56, "32B"),
    (float("inf"), "70B"),
)


def accelerator_memory_bytes(accelerators: Optional[List[Dict[str, Any]]]) -> int:
    """Memory the primary accelerator can address, VRAM carve-out plus GTT.

    On a unified part the GTT aperture is genuinely usable for weights, so
    ignoring it under-sells the machine by more than half. On a discrete card
    GTT is host memory reached over PCIe; it is reported but not counted.
    """
    for accelerator in accelerators or []:
        vram = int(accelerator.get("vram_total_bytes") or 0)
        gtt = int(accelerator.get("gtt_total_bytes") or 0)
        if not vram and not gtt:
            continue
        if accelerator.get("unified_memory"):
            return vram + gtt
        # A part with no carve-out at all is addressed entirely through its
        # aperture. Returning `vram` here reported zero usable memory for a
        # GPU that plainly has some, which is the mirror of crediting a
        # discrete card with host memory it cannot hold weights in.
        return vram or gtt


def model_memory_budget(
    system_memory_bytes: int,
    accelerators: Optional[List[Dict[str, Any]]] = None,
) -> Dict[str, Any]:
    """The memory a local model may actually occupy, and where it came from."""
    system_memory_bytes = max(0, int(system_memory_bytes or 0))
    reserve = int(system_memory_bytes * SYSTEM_RESERVE_FRACTION)
    host_budget = max(0, system_memory_bytes - reserve)
    accelerator_bytes = accelerator_memory_bytes(accelerators)
    if accelerator_bytes:
        # GTT is system RAM under another name, so the accelerator budget is
        # clamped by what the host can spare. Skipping this clamp is how a
        # unified part gets promised more memory than the machine has.
        budget = min(accelerator_bytes, host_budget) if host_budget else accelerator_bytes
        return {
            "budget_bytes": budget,
            "source": "accelerator",
            "accelerator_bytes": accelerator_bytes,
            "host_bytes": host_budget,
            "reserved_for_system_bytes": reserve,
        }
    return {
        "budget_bytes": host_budget,
        "source": "system-memory",
        "accelerator_bytes": 0,
        "host_bytes": host_budget,
        "reserved_for_system_bytes": reserve,
    }


def recommended_model_tier(
    system_memory_bytes: int,
    accelerators: Optional[List[Dict[str, Any]]] = None,
) -> str:
    """Largest parameter class worth recommending on this machine.

    The CPU ladder deliberately stops at 8 B: above that, generation on a CPU
    is too slow to be a usable assistant regardless of how much RAM is fitted.
    An accelerator changes that, so it gets its own ladder.
    """
    budget = model_memory_budget(system_memory_bytes, accelerators)
    gib = budget["budget_bytes"] / GIB
    tiers = (
        ACCELERATOR_TIERS if budget["source"] == "accelerator" else CPU_TIERS
    )
    # The CPU ladder is expressed against total RAM, which is how it has always
    # behaved; the accelerator ladder is expressed against the usable budget.
    measure = (
        gib
        if budget["source"] == "accelerator"
        else max(0, int(system_memory_bytes or 0)) / GIB
    )
    for ceiling, tier in tiers:
        if measure < ceiling:
            return tier
    return tiers[-1][1]


def feature_policy(
    board: Dict[str, Any],
    inventory: Dict[str, Any],
    accelerators: Optional[List[Dict[str, Any]]] = None,
) -> Dict[str, Any]:
    memory_bytes = int(inventory.get("memory_total_bytes", 0) or 0) or int(
        board.get("memory_total_bytes", 0) or 0
    )
    memory_gib = memory_bytes / GIB
    cores = int(board.get("cpu_cores", 1) or 1)
    budget = model_memory_budget(memory_bytes, accelerators)
    return {
        "local_ai": memory_gib >= 3,
        "recommended_local_model_max": recommended_model_tier(
            memory_bytes, accelerators
        ),
        "containers": cores >= 2 and memory_gib >= 1,
        "desktop_streaming": memory_gib >= 2,
        "model_budget_bytes": budget["budget_bytes"],
        "model_budget_source": budget["source"],
        "accelerated": budget["source"] == "accelerator",
    }


#: Where a local model can run. This is a dimension of a model choice, not a
#: property of the appliance: an x86 host can hold a resident NPU model for
#: interactive work and a larger GPU model for heavy work at the same time, so
#: "the local model" is not a singleton.
ACCELERATOR_TIERS_BY_KIND = ("npu", "gpu", "cpu")


def local_inference_tiers(
    system_memory_bytes: int,
    accelerators: Optional[List[Dict[str, Any]]] = None,
    neural_accelerators: Optional[List[Dict[str, Any]]] = None,
    resident_bytes: Optional[Dict[str, int]] = None,
    npu_context_tokens: int = NPU_RECOMMENDED_CONTEXT_TOKENS,
    *,
    npu_serving: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    """Describe the local inference tiers this machine can host, jointly.

    On a unified-memory part the NPU and the GPU draw on the same pool, so a
    model resident on one permanently reduces what the other can load. Sizing
    each tier in isolation over-commits the machine. ``resident_bytes`` carries
    what each tier has already claimed.

    The split between tiers is deliberately **not** fixed here. What a resident
    NPU model costs the GPU tier on real hardware is a measurement, not a
    constant, and hard-coding a ratio would be inventing one.
    """
    resident = dict(resident_bytes or {})
    budget = model_memory_budget(system_memory_bytes, accelerators)
    claimed = sum(max(0, int(value or 0)) for value in resident.values())
    shared_pool = max(0, budget["budget_bytes"] - claimed)
    tiers = []
    if neural_accelerators:
        npu = neural_accelerators[0]
        # The NPU server is launched with an explicit context window. Left
        # unset, FLM preallocates KV for its full 131,072-token window: a 3B
        # measured 17.4 GB that way against 4.0 GB at 8192, and was marginally
        # faster at the smaller window. The plan carries the flags so the tier
        # cannot be started without them.
        plan = npu_tier_plan(context_tokens=npu_context_tokens)
        # The gate is real capability discovery now, not a hard False (VD-001).
        # Vaelor serves the NPU itself through flm-real, so this tier is
        # available exactly when a caller has confirmed the binary, the device
        # and the model are all present AND the selected model is fit to run.
        # `npu_serving` carries that verdict (see `flm_service.discover_npu_serving`);
        # a caller that passes nothing gets honest absence rather than a claim,
        # and a model the capability table rules out keeps the tier unavailable
        # even where flm-real is present - an unusable model has nothing to
        # launch. Every Raspberry Pi fails all of this and the tier stays
        # unavailable, which is the Pi-isolation guarantee.
        serving = dict(npu_serving or {})
        npu_available = bool(serving.get("available")) and bool(plan["usable"])
        if npu_available:
            npu_reason = ""
        elif not plan["usable"]:
            npu_reason = str(plan["capability_reason"])
        elif serving:
            npu_reason = str(serving.get("reason") or "")
        else:
            npu_reason = (
                "Vaelor has not confirmed flm-real serving on this machine, so "
                "the neural processor tier is not being served. It becomes "
                "available once the flm-real binary, the device and the model "
                "are all present (VD-001)."
            )
        tiers.append({
            "kind": "npu",
            "device": npu.get("name"),
            "device_node": npu.get("device_node"),
            "role": "interactive",
            "resident_bytes": max(0, int(resident.get("npu", 0) or 0)),
            # Vaelor serves this tier itself, through the flm-real OpenAI-
            # compatible server it launches and supervises (VD-001/VD-002).
            "backend": "flm-real",
            "available": npu_available,
            "reason": npu_reason,
            "context_tokens": plan["context_tokens"],
            "native_context_tokens": NPU_NATIVE_CONTEXT_TOKENS,
            "arguments": plan["arguments"],
            "context_reason": plan["context_reason"],
            "reclaimed_bytes": plan["reclaimed_bytes"],
            # `reclaimed_bytes` is None for an unmeasured window, and a reader
            # cannot tell that from "nothing was reclaimed" without this flag.
            # The plan has always carried it; the tier used to drop it.
            "footprint_measured": plan["footprint_measured"],
            "model": plan["model"],
            # The name `flm-real` answers to, carried beside Vaelor's own. The
            # two are different strings and a tier that only publishes the
            # Vaelor identifier hands its caller an unlaunchable name.
            "flm_tag": plan["flm_tag"],
            "flm_tag_known": plan["flm_tag_known"],
            "model_selection": plan["selection"],
            # The capability guard travels with the model it guards. Copying
            # `model` and `selection` while leaving the verdict behind is what
            # let an unusable model reach a tier that looked deployable.
            "model_usable": plan["usable"],
            "model_capability_reason": plan["capability_reason"],
            "native_tool_calling": plan["native_tool_calling"],
        })
    if accelerators:
        tiers.append({
            "kind": "gpu",
            "device": accelerators[0].get("name"),
            "role": "heavy",
            "resident_bytes": max(0, int(resident.get("gpu", 0) or 0)),
            "backend": "llama.cpp",
            "available": True,
            "reason": None,
        })
    tiers.append({
        "kind": "cpu",
        "device": None,
        "role": "fallback",
        "resident_bytes": max(0, int(resident.get("cpu", 0) or 0)),
        "backend": "llama.cpp",
        "available": True,
        "reason": None,
    })
    return {
        "tiers": tiers,
        "shared_memory": budget["source"] == "accelerator",
        "total_budget_bytes": budget["budget_bytes"],
        "claimed_bytes": claimed,
        "available_bytes": shared_pool,
        "note": (
            "The accelerators share one memory pool, so a model resident on "
            "one tier reduces what the others can load."
            if budget["source"] == "accelerator"
            else ""
        ),
    }


def resident_tier_kinds(hardware: Optional[Dict[str, Any]] = None) -> List[str]:
    """Which local tiers this machine keeps resident, gated on its hardware.

    The single predicate behind every count of resident models, and the same
    gating :func:`local_inference_tiers` applies to its engine list: an NPU tier
    only with a neural accelerator, a GPU tier only with a GPU, otherwise one
    CPU model. Centralised so :func:`~vaelor.inference_tuning.recommended_deployment`,
    the deploy executor, and the status surface cannot drift on it (#98). The
    two-tier NPU+GPU plan is Z2-only (VD-071/VD-072); on a CPU-only Pi this
    returns ``["cpu"]`` and the one model serves both surfaces.
    """
    facts = dict(hardware or {})
    kinds: List[str] = []
    if facts.get("neural_accelerators"):
        kinds.append("npu")
    if facts.get("accelerators"):
        kinds.append("gpu")
    return kinds or ["cpu"]


def cpu_tier_plan(
    model: str = "", *, context_tokens: int = RECOMMENDED_CONTEXT_TOKENS
) -> Dict[str, Any]:
    """The single local tier for a machine with no NPU and no GPU (VD-071).

    Sibling of :func:`local_inference_tiers`' CPU fallback: when neither
    accelerator is present, one model runs on the CPU and serves both the
    Assistant and AI Chat, so there is no second resident tier to reserve
    memory for. Emitting the Z2's NPU+GPU pair here regardless of hardware is
    the #205 defect (VD-071/VD-072).
    """
    window = max(1, int(context_tokens or RECOMMENDED_CONTEXT_TOKENS))
    # No accelerated backend, so no flash attention and the KV cache stays f16 -
    # the same choice `_deploy_model` makes for an unaccelerated host. Quantizing
    # it here would apply an accelerator measurement to hardware nobody
    # benchmarked (VD-071 is Pi-scoped).
    return {
        "kind": "cpu",
        "role": "local",
        "model": str(model or ""),
        "backend": "llama.cpp",
        "context_tokens": window,
        "cache_type_k": "f16",
        "cache_type_v": "f16",
        "arguments": ["--ctx-size", str(window), "-ctk", "f16", "-ctv", "f16"],
        "reason": (
            "This machine has no neural accelerator and no GPU, so one model "
            "runs on the CPU and serves both the Assistant and AI Chat "
            "(VD-071). The two-tier NPU-plus-GPU plan is Z2-only."
        ),
    }


def describe_inference_machine(hardware: Optional[Dict[str, Any]] = None) -> str:
    """Describe the machine an inference status is *about*, from its own facts.

    Kept apart from :data:`~vaelor.inference_tuning.MEASURED_ON`, which names the
    reference bench box the tuning was benchmarked on. Emitting that constant as
    an arbitrary machine's own measured status is the #205 lie: a CPU-only Pi
    reported the HP Z2's identity as its own.
    """
    facts = dict(hardware or {})
    device = str(facts.get("device") or "").strip()
    parts = []
    if facts.get("neural_accelerators"):
        parts.append("a neural accelerator")
    if facts.get("accelerators"):
        parts.append("a GPU")
    phrase = (
        " and ".join(parts) if parts
        else "no neural accelerator and no GPU (CPU only)"
    )
    return (
        "{} - {}".format(device, phrase) if device
        else "This machine has {}.".format(phrase)
    )


def loaded_model_capacity(hardware: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
    """``max_loaded_models`` for this machine's resident tiers, gated on hardware.

    One derivation for every caller that reports the setting, so a CPU-only Pi
    never advertises the Z2's two-model capacity (VD-071/VD-072).
    """
    from .inference_tuning import loaded_model_settings

    kinds = resident_tier_kinds(hardware)
    return loaded_model_settings(len(kinds), kinds=kinds)


def kv_cache_bytes(
    context_tokens: int,
    *,
    layers: int,
    kv_heads: int,
    head_dim: int,
    cache_type_k: str = "f16",
    cache_type_v: str = "f16",
    parallel: int = 1,
) -> int:
    """Exact KV cache size when GGUF metadata is known.

    ``context × layers × kv_heads × head_dim`` elements for each of K and V,
    at the bit width of that cache's dtype, **times the number of slots**.

    The slot count is a parameter rather than an assumption because it is the
    one factor a request cannot see: the caller asks for a window and the
    runtime decides how many copies of it to hold. On the appliance it decided
    four, and this function, computing one, agreed with a container limit that
    was then exceeded by 1.4 GB.
    """
    bits_k = KV_CACHE_BITS.get(cache_type_k, 16)
    bits_v = KV_CACHE_BITS.get(cache_type_v, 16)
    elements = max(0, int(context_tokens)) * max(1, int(layers)) * max(1, int(kv_heads)) * max(1, int(head_dim))
    return int(elements * max(1, int(parallel)) * (bits_k + bits_v) / 8)


def estimate_kv_cache_bytes(
    context_tokens: int,
    cache_type_k: str = "f16",
    cache_type_v: str = "f16",
    parallel: int = 1,
) -> int:
    """Upper-bound KV cache size when GGUF metadata is not available.

    Prefer :func:`kv_cache_bytes` whenever layer and head counts can be read
    from the model. This estimate exists so the planner never sizes a context
    window while pretending the cache is free — and ``parallel`` exists so it
    never sizes one while pretending there is only one of it.
    """
    bits_k = KV_CACHE_BITS.get(cache_type_k, 16)
    bits_v = KV_CACHE_BITS.get(cache_type_v, 16)
    scale = (bits_k + bits_v) / 32
    tokens = max(0, int(context_tokens)) * max(1, int(parallel))
    return int(tokens * KV_BYTES_PER_TOKEN_F16 * scale)


def plan_kv_cache(
    cache_type_k: str = "f16",
    cache_type_v: str = "f16",
    *,
    flash_attention: Optional[bool] = None,
) -> Dict[str, Any]:
    """Resolve a KV cache configuration that llama.cpp will actually accept.

    Both cache types are one decision, taken by
    :func:`~vaelor.inference_tuning.resolve_kv_cache_types`. This function used
    to take it here and took it half way: it downgraded V to f16 without flash
    attention and left K quantized, on the stated grounds that "K-only
    quantization is a fully supported outcome". Supported it is; usable it is
    not. That pair was measured at **1.01 tokens per second** decode against
    64.22 with both caches quantized and flash attention on, and 14.19 with
    neither — a server that starts, serves, answers, and warns nobody.

    ``flash_attention=None`` still means "not established on this backend" and
    is still treated as unavailable. What changed is what that costs: it now
    degrades to plain f16 rather than to the worst measured configuration.
    """
    requested_k = cache_type_k if cache_type_k in KV_CACHE_BITS else "f16"
    requested_v = cache_type_v if cache_type_v in KV_CACHE_BITS else "f16"
    unknown_type = (
        cache_type_k not in KV_CACHE_BITS or cache_type_v not in KV_CACHE_BITS
    )
    available = bool(flash_attention)
    resolved = resolve_kv_cache_types(
        requested_k, requested_v, flash_attention_available=available
    )
    resolved_k = resolved["cache_type_k"]
    resolved_v = resolved["cache_type_v"]
    adjusted = bool(resolved["adjusted"]) or unknown_type
    reason = resolved["reason"]
    if not reason and unknown_type:
        reason = "An unsupported cache type was requested; f16 was used instead."
    # Not re-derived here. This was the fourth copy of the rule and it decided
    # from V alone, so `k=q8_0, v=f16` reported `flash_attention: False` beside
    # a quantized K — the 1.01 tok/s pairing, announced as unadjusted, while
    # `build_model_compose` re-resolved and shipped something else.
    use_flash_attention = bool(resolved["flash_attention"])
    return {
        "cache_type_k": resolved_k,
        "cache_type_v": resolved_v,
        "flash_attention": use_flash_attention,
        # *How* it is turned on, never *that* it is forced on. `-fa 1` was
        # measured as a large loss on some model and backend pairs (848 against
        # 1544 prefill on a 20B under Vulkan) and a large win on others, so
        # llama.cpp is left to decide per model.
        "flash_attention_mode": FLASH_ATTENTION_MODE if use_flash_attention else "",
        "flash_attention_available": available,
        "requested_cache_type_k": requested_k,
        "requested_cache_type_v": requested_v,
        "adjusted": adjusted,
        "reason": reason,
    }
