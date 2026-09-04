"""K and V cache quantization is one decision, and it needs flash attention.

This module exists because the decision was implemented twice, independently,
and both implementations got the same half of it wrong: they downgraded the V
cache to ``f16`` when flash attention was unavailable and left the K cache
quantized. That is not a conservative middle ground. It is the worst
configuration measured on this hardware.

Measured on the Z2 on a 3 B at 32 k context, two independent runs agreeing to
two decimal places, memory identical to the tenth of a megabyte:

===============================================  =============
KV configuration                                 decode tok/s
===============================================  =============
f16 K / f16 V, no flash attention                14.19
**q8_0 K only, no flash attention**              **1.01**
q4_0 K only, no flash attention                  1.83
q8_0 K + q8_0 V, flash attention on              64.22
q8_0 K + q8_0 V, no flash attention              refused to start
===============================================  =============

The asymmetry is the whole trap. llama.cpp **refuses** a quantized V cache
without flash attention — ``quantized V cache requires flash_attn to be
enabled`` — so that mistake cannot be made silently. It does **not** refuse a
quantized K cache. The server starts, serves, answers coherently, and runs 64x
slower than the correct configuration and 14x slower than no quantization at
all, with GTT usage jumping from ~374 MB to ~1,961 MB as the fallback path
spills out of VRAM. Nothing warns.

So the two cache types are not independent settings. They are one decision
that either applies with flash attention or does not apply at all.
"""

from __future__ import annotations

from typing import Any, Dict


#: The cache type that works on every backend with or without flash attention.
UNQUANTIZED_CACHE_TYPE = "f16"

#: Cache types that are not quantized, so are safe without flash attention.
UNQUANTIZED_CACHE_TYPES = frozenset({"", "f16", "bf16"})

#: What quantizing the K cache *without* flash attention costs, measured.
#: Kept beside the code so a reviewer can check the claim without leaving the
#: file — the previous half-fix was justified in a comment that described the
#: other failure mode entirely.
KV_WITHOUT_FLASH_ATTENTION_MEASUREMENT = {
    "backend": "vulkan",
    "model": "Llama-3.2-3B-Instruct-UD-Q4_K_XL",
    "context_tokens": 32768,
    "prompt_tokens": 12882,
    "runs": 2,
    "f16_no_flash_attention_decode_tps": 14.19,
    "q8_0_k_only_no_flash_attention_decode_tps": 1.01,
    "q4_0_k_only_no_flash_attention_decode_tps": 1.83,
    "q8_0_both_with_flash_attention_decode_tps": 64.22,
    "quantized_v_without_flash_attention": "refused to start",
}


def kv_cache_is_quantized(cache_type: Any) -> bool:
    """Whether this cache type is anything other than an unquantized one."""
    return (
        str(cache_type or UNQUANTIZED_CACHE_TYPE).strip().casefold()
        not in UNQUANTIZED_CACHE_TYPES
    )


def resolve_kv_cache_types(
    cache_type_k: str = UNQUANTIZED_CACHE_TYPE,
    cache_type_v: str = UNQUANTIZED_CACHE_TYPE,
    *,
    flash_attention_available: bool = False,
) -> Dict[str, Any]:
    """Resolve K and V cache types as **one** decision, not two.

    Without flash attention, both caches are ``f16``. With it, both keep what
    was asked for. There is no third outcome, because the third outcome is the
    1.01 tok/s row.

    **Whether to turn flash attention on is part of the same decision**, and is
    returned as ``flash_attention`` rather than re-derived by each caller. It
    was re-derived, four times, and the last copy asked the question of ``V``
    alone — ``available and resolved_v != "f16"`` — which is the identical
    asymmetry this module exists to remove: with ``k=q8_0, v=f16`` it answered
    "no flash attention" for a configuration whose K cache is quantized, so the
    pairing reported to the operator was the 1.01 tok/s row while the container
    shipped something else. Quantizing *either* cache is what needs the flag,
    so ``quantized`` and ``flash_attention`` are the same fact seen twice.
    """
    requested_k = str(cache_type_k or UNQUANTIZED_CACHE_TYPE)
    requested_v = str(cache_type_v or UNQUANTIZED_CACHE_TYPE)
    available = bool(flash_attention_available)
    quantization_wanted = (
        kv_cache_is_quantized(requested_k) or kv_cache_is_quantized(requested_v)
    )
    if available or not quantization_wanted:
        return {
            "cache_type_k": requested_k,
            "cache_type_v": requested_v,
            "flash_attention_available": available,
            "quantized": quantization_wanted,
            # Only turned on where it buys something, so a CPU or unverified
            # backend is never handed a flag it may not implement.
            "flash_attention": available and quantization_wanted,
            "requested_cache_type_k": requested_k,
            "requested_cache_type_v": requested_v,
            "adjusted": False,
            "reason": "",
        }
    return {
        "cache_type_k": UNQUANTIZED_CACHE_TYPE,
        "cache_type_v": UNQUANTIZED_CACHE_TYPE,
        "flash_attention_available": False,
        "quantized": False,
        "flash_attention": False,
        "requested_cache_type_k": requested_k,
        "requested_cache_type_v": requested_v,
        "adjusted": True,
        "reason": (
            "Flash attention is not available on this backend, so both KV cache "
            "types stay {}. Quantizing K on its own is not a safe half measure: "
            "it starts and answers, and decode was measured at {:g} tokens per "
            "second against {:g} with both caches quantized and flash attention "
            "on, and {:g} with neither. Quantizing V on its own is refused at "
            "load instead."
        ).format(
            UNQUANTIZED_CACHE_TYPE,
            KV_WITHOUT_FLASH_ATTENTION_MEASUREMENT[
                "q8_0_k_only_no_flash_attention_decode_tps"
            ],
            KV_WITHOUT_FLASH_ATTENTION_MEASUREMENT[
                "q8_0_both_with_flash_attention_decode_tps"
            ],
            KV_WITHOUT_FLASH_ATTENTION_MEASUREMENT[
                "f16_no_flash_attention_decode_tps"
            ],
        ),
    }
