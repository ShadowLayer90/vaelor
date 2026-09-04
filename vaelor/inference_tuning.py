"""Measured inference tuning: the configuration the benchmarks recommend.

Everything Vaelor knows about *how* to run a local model well is a
measurement, not a preference, and every measurement lives in this module so
there is exactly one file to edit when the benchmark matrix lands a new number.
Nothing below is a guess; each constant carries the observation it came from.

The four things this module exists to make reachable, because Vaelor's own code
path could not previously produce any of them. Each authoritative measurement
lives on the constant it belongs to; these are pointers, not the record:

1. **Flash attention is discoverable** (``flash_attention_support``), so
   ``plan_kv_cache`` no longer silently downgrades ``cache_type_v`` to ``f16``.
2. **Context is sized against the prompt, not a fixed mode** (VD-075).
3. **A second model may stay loaded** on the two-tier Z2, and eviction is
   detected when a server left at 1 unloads silently. Z2-only (VD-071/VD-072).
4. **The NPU tier gets an explicit context window** so FLM does not preallocate
   KV for its full 131,072-token window.
"""

from __future__ import annotations

from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence


GIB = 1024 ** 3
MIB = 1024 ** 2

# The backend benchmark catalogue and its provenance live in a sibling module
# (#206 split; this module was at the 1,000-line ceiling). They are re-exported
# here because this stays the one public name for tuning facts. `MEASURED_ON` is
# the reference bench box every figure below was taken on - provenance on any
# other machine, never its own status (#205, #57).
from .inference_measurements import (  # noqa: F401
    BACKEND_MEASUREMENTS,
    BACKEND_WINS,
    KV_QUANTIZATION_MEASUREMENT,
    LEMONADE_SNAP_REVISION,
    LEMONADE_VERSION,
    MEASURED_ON,
    MODEL_FLAG_MEASUREMENTS,
    SUPERSEDED_SNAP_REVISION,
    SUPERSEDED_VERSION,
    VULKAN_CARVE_OUT_CEILING_OBSERVED,
    VULKAN_CARVE_OUT_HEADROOM_FRACTION,
)


# Context windows, and the model each tier runs, live in focused modules so
# they can be read without the decision code around them. They are re-exported
# here because this module stays the one public name for tuning facts.
from .inference_context import (  # noqa: F401
    AGENT_PROMPT_TOKENS,
    GPU_RECOMMENDED_CONTEXT_TOKENS,
    NPU_CONTEXT_FLAG,
    NPU_CONTEXT_FOOTPRINT_BYTES,
    NPU_NATIVE_CONTEXT_TOKENS,
    NPU_RECOMMENDED_CONTEXT_TOKENS,
    RECOMMENDED_CONTEXT_TOKENS,
    RECOMMENDED_RUNTIME_MODE,
)
from .inference_model_choice import (  # noqa: F401
    ASSISTANT_MODEL_PIN,
    CAPABILITY_PROBE,
    CHAT_TIER_MODEL,
    INSTALLED_FLM_MODELS,
    MINIMUM_SCHEMA_VALID_RATE,
    NPU_MODEL_CAPABILITY,
    NPU_NATIVE_TOOL_CALLING,
    QWEN35_NPU_MEASUREMENT,
    SELECTION_CRITERIA,
    flm_model_tag,
    npu_model_capability,
    recommended_npu_model,
)
# The K/V coupling is its own module because it is one rule with one
# measurement behind it, and because it was previously implemented twice — in
# `gpu_tier_plan` here and in `plan_kv_cache` in `model_sizing` — with the same
# half of it wrong in both. There is now one place to get it right.
from .kv_cache_policy import (  # noqa: F401
    KV_WITHOUT_FLASH_ATTENTION_MEASUREMENT,
    UNQUANTIZED_CACHE_TYPE,
    kv_cache_is_quantized,
    resolve_kv_cache_types,
)


# ---------------------------------------------------------------------------
# Backend selection
# ---------------------------------------------------------------------------

#: Vulkan allocates from the VRAM carve-out; ROCm allocates from the larger GTT
#: pool. Under the old policy this was the *only* reason to reach for ROCm and
#: it was a capacity reason. It now runs the same direction as the performance
#: evidence rather than against it: the default backend is also the one with
#: more room, so there is no size at which the appliance has to trade speed for
#: headroom, and no fallback rung to get wrong.
BACKEND_MEMORY_POOL = {
    "vulkan": "vram-carve-out",
    "rocm": "gtt",
    "cpu": "system-memory",
}

#: Server and runtime allowance on top of weights and KV cache. llama-server's
#: compute buffers, the Vulkan/HIP scratch allocations and the model's own
#: graph do not fit in zero bytes, and planning as if they do is how a model
#: that "fits" fails at load.
RUNTIME_OVERHEAD_BYTES = 1 * GIB

#: Character devices each backend needs. Vulkan needs only the render node;
#: ROCm additionally needs ``/dev/kfd``. Under the old policy this counted for
#: Vulkan. It is now a cost the default pays: the measured prefill advantage on
#: a prefill-dominated workload outweighs one extra character device, and the
#: difference is recorded here so the trade stays visible rather than becoming
#: invisible by being the default.
BACKEND_DEVICES = {
    "vulkan": ("/dev/dri/renderD128", "/dev/dri/card0"),
    "rocm": ("/dev/kfd", "/dev/dri/renderD128", "/dev/dri/card0"),
    "cpu": (),
}

#: The backend chosen for GPU inference unless an operator pins another.
#:
#: ROCm, on measured prefill for this workload. Prefill dominates every request
#: the appliance actually makes - 68% of an answer, measured - and on revision
#: 378 ROCm leads prefill on 7 of 8 models. This used to cite a ~4,668-token
#: agent prompt as the reason; the prompt is 1,344 (VD-075) and prefill still
#: dominates, because it is the question and the retrieved context that make it
#: so, not the standing brief.
#:
#: This is a *standing* default rather than one conditioned on the installed
#: runtime version. Vulkan currently leads decode, and a future ROCm update is
#: expected to close that; making the choice version-conditional would flip the
#: appliance's backend underneath users as point releases land, which is a
#: worse property than being slightly behind on decode for a while.
DEFAULT_GPU_BACKEND = "rocm"

#: ROCm is a *telemetry* dependency and stays installed regardless of which
#: backend runs inference: ``amd-smi`` is where NPU activity, NPU power and
#: adapter firmware come from, and nothing else publishes them. Uninstalling
#: ROCm because inference moved to Vulkan would blind the hardware pages.
ROCM_TELEMETRY_TOOL = "amd-smi"
ROCM_TELEMETRY_FIELDS = (
    "NPU utilisation", "NPU power", "NPU clock", "adapter firmware",
)


def backend_devices(backend: str) -> List[str]:
    """The character devices this backend actually needs."""
    return list(BACKEND_DEVICES.get(str(backend or "").lower(), ()))


def rocm_requirement(backend: str) -> Dict[str, Any]:
    """Separate ROCm-for-telemetry from ROCm-for-inference, explicitly.

    These are two different dependencies that happen to share a name, and
    conflating them is how "we run Vulkan now" turns into a hardware page with
    no NPU readings on it.
    """
    inference = str(backend or "").lower() == "rocm"
    return {
        "telemetry": True,
        "telemetry_tool": ROCM_TELEMETRY_TOOL,
        "telemetry_reason": (
            "{} is the only source for {} on this hardware, so ROCm stays "
            "installed whichever backend runs inference."
        ).format(ROCM_TELEMETRY_TOOL, ", ".join(ROCM_TELEMETRY_FIELDS)),
        "inference": inference,
        "inference_reason": (
            "ROCm is the measured default for GPU inference on this hardware: "
            "it leads prefill on {} of {} models on snap revision {}, and this "
            "appliance's requests are prefill-dominated."
            .format(
                BACKEND_WINS["prefill"]["rocm"],
                BACKEND_WINS["prefill"]["models"],
                LEMONADE_SNAP_REVISION,
            )
            if inference else
            "Inference does not use ROCm on this deployment."
        ),
    }


def _primary(accelerators: Optional[Sequence[Mapping[str, Any]]]) -> Dict[str, Any]:
    for accelerator in accelerators or []:
        if accelerator.get("vram_total_bytes") or accelerator.get("gtt_total_bytes"):
            return dict(accelerator)
    return {}


def select_inference_backend(
    *,
    model_bytes: int,
    kv_cache_bytes: int = 0,
    accelerators: Optional[Sequence[Mapping[str, Any]]] = None,
    override: str = "",
    overhead_bytes: int = RUNTIME_OVERHEAD_BYTES,
    system_memory_bytes: int = 0,
) -> Dict[str, Any]:
    """Choose an accelerated backend from measurement, and say why.

    **ROCm is the default**, on measured prefill for this workload. Every
    request the appliance makes is prefill-dominated - 68% of an answer is
    prefill, measured on the Pi - and on snap revision 378 ROCm leads prefill on 7 of 8
    models — ``Llama-3.2-3B`` at 2445.1 tok/s against Vulkan's 2022.4, both
    taken through Lemonade. The 2334.0 this used to quote for Vulkan came from
    a native container, and comparing across harnesses understated the lead.

    This reverses the previous policy, and the reversal is a measurement rather
    than a change of mind. The earlier figures were taken on revision 360, the
    machine auto-refreshed to 378 mid-benchmark, and ROCm prefill rose by up to
    +262% while Vulkan and the NPU did not move at all. Reasoning from the
    superseded numbers is what made Vulkan look like the obvious default.

    There is no longer a capacity rung. Vulkan allocates from the 16 GiB VRAM
    carve-out and ROCm from the 22.78 GiB shared pool, so the faster backend is
    also the roomier one and the old "fall back to ROCm above the carve-out"
    special case has nothing left to do. Vulkan stays reachable as an explicit
    override — it still leads decode on 7 of 8 — and an override that will not
    fit the carve-out is told so.
    """
    accelerator = _primary(accelerators)
    carve_out = max(0, int(accelerator.get("vram_total_bytes") or 0))
    gtt = max(0, int(accelerator.get("gtt_total_bytes") or 0))
    required = (
        max(0, int(model_bytes or 0))
        + max(0, int(kv_cache_bytes or 0))
        + max(0, int(overhead_bytes or 0))
    )
    usable_carve_out = int(carve_out * VULKAN_CARVE_OUT_HEADROOM_FRACTION)
    system_memory = max(0, int(system_memory_bytes or 0))
    # Headroom against the carve-out when there is one, else against this
    # machine's own system memory, and clamped at zero either way: computed
    # against a zero carve-out it went to -1 GiB on the CPU-only Pi (#205), a
    # negative reported as this machine's status. Never below zero.
    headroom_bytes = max(
        0, (usable_carve_out if carve_out else system_memory) - required
    )
    decision: Dict[str, Any] = {
        "backend": "cpu",
        "reason": "",
        "measured": True,
        # Benchmark provenance, not this machine (was `measured_on`, #205).
        "benchmarked_on": MEASURED_ON,
        "required_bytes": required,
        "vram_carve_out_bytes": carve_out,
        "usable_carve_out_bytes": usable_carve_out,
        "gtt_pool_bytes": gtt,
        "headroom_bytes": headroom_bytes,
        "memory_pool": BACKEND_MEMORY_POOL["cpu"],
        "devices": [],
        "override": "",
        "measurements": BACKEND_MEASUREMENTS,
    }
    chosen = str(override or "").strip().lower()
    if chosen in BACKEND_MEMORY_POOL:
        decision.update({
            "backend": chosen,
            "override": chosen,
            "measured": False,
            "memory_pool": BACKEND_MEMORY_POOL[chosen],
            "devices": backend_devices(chosen),
            "reason": _override_reason(chosen, required, carve_out, usable_carve_out),
        })
        decision["rocm"] = rocm_requirement(chosen)
        return decision
    if not accelerator:
        decision["reason"] = "No accelerator was discovered, so inference runs on the CPU."
        # Absence chose the CPU here, not a benchmark, and this machine has no
        # accelerator to have produced the Z2's throughput matrix. Leaving
        # `measured: true` beside that foreign catalogue reads as "this machine
        # measured these", which is the #205 leak one level down (LESSONS 5/6).
        # Drop the catalogue and its bench-box identity; say where the figures
        # do live instead.
        decision["measured"] = False
        decision.pop("measurements", None)
        decision.pop("benchmarked_on", None)
        decision["measurements_note"] = (
            "Backend throughput figures are measured on the reference bench "
            "box, not on this machine, which has no accelerator to benchmark."
        )
        decision["rocm"] = rocm_requirement("cpu")
        return decision
    rocm = BACKEND_MEASUREMENTS["llama-3.2-3b"]["rocm"]
    vulkan = BACKEND_MEASUREMENTS["llama-3.2-3b"]["vulkan"]
    decision.update({
        "backend": DEFAULT_GPU_BACKEND,
        "memory_pool": BACKEND_MEMORY_POOL[DEFAULT_GPU_BACKEND],
        "devices": backend_devices(DEFAULT_GPU_BACKEND),
        "snap_revision": LEMONADE_SNAP_REVISION,
        "reason": (
            "ROCm, on measured prefill. This appliance's agent prompt is {} "
            "tokens, so its requests are prefill-dominated, and on snap "
            "revision {} ({}) ROCm leads prefill on {} of {} models — "
            "{:g} against {:g} tok/s on Llama-3.2-3B, both through the same "
            "harness. It also draws on the "
            "{:.1f} GiB shared pool rather than the {:.1f} GiB carve-out, so "
            "this model's {:.1f} GiB needs no fallback. Vulkan still leads "
            "decode on {} of {} and remains available as an override."
        ).format(
            AGENT_PROMPT_TOKENS,
            LEMONADE_SNAP_REVISION, LEMONADE_VERSION,
            BACKEND_WINS["prefill"]["rocm"], BACKEND_WINS["prefill"]["models"],
            rocm["lemonade_prefill_tps"], vulkan["lemonade_prefill_tps"],
            gtt / GIB, carve_out / GIB, required / GIB,
            BACKEND_WINS["decode"]["vulkan"], BACKEND_WINS["decode"]["models"],
        ),
    })
    decision["rocm"] = rocm_requirement(DEFAULT_GPU_BACKEND)
    return decision


def _override_reason(
    chosen: str, required: int, carve_out: int, usable_carve_out: int
) -> str:
    """Say that an operator pinned this, and what it costs if it will not fit.

    A Vulkan override is the one that can silently hurt: it allocates from the
    carve-out, and ``Qwen3.6-27B`` was measured there at 16,268 MB of 16,384.
    An operator who pins it for a model that size should be told before the
    appliance runs it, not after. Both recorded peak temperatures are quoted
    with their runs, because they disagree by 13 °C and picking one silently
    would be a claim rather than a measurement.
    """
    base = (
        "An operator pinned the {} backend, so measurement did not choose it. "
        "The measured default is {}."
    ).format(chosen, DEFAULT_GPU_BACKEND)
    if chosen != "vulkan" or not carve_out or required <= usable_carve_out:
        return base
    return base + (
        " This model needs {:.1f} GiB and Vulkan allocates from the {:.1f} GiB "
        "carve-out, which has room for {:.1f} GiB at {:.0f}% occupancy. {} was "
        "measured at {} MB of {} ({:.1%}) there, and peaked at {} °C in a {} "
        "({} °C in the {})."
    ).format(
        required / GIB, carve_out / GIB, usable_carve_out / GIB,
        VULKAN_CARVE_OUT_HEADROOM_FRACTION * 100,
        VULKAN_CARVE_OUT_CEILING_OBSERVED["model"],
        VULKAN_CARVE_OUT_CEILING_OBSERVED["used_mib"],
        VULKAN_CARVE_OUT_CEILING_OBSERVED["carve_out_mib"],
        VULKAN_CARVE_OUT_CEILING_OBSERVED["utilisation"],
        VULKAN_CARVE_OUT_CEILING_OBSERVED["peak_temperature_c"],
        VULKAN_CARVE_OUT_CEILING_OBSERVED["peak_temperature_source"],
        VULKAN_CARVE_OUT_CEILING_OBSERVED["matrix_peak_temperature_c"],
        VULKAN_CARVE_OUT_CEILING_OBSERVED["matrix_peak_temperature_source"],
    )


# ---------------------------------------------------------------------------
# Flash attention
# ---------------------------------------------------------------------------

#: Backends on which quantizing the V cache has actually been observed to work.
#: The CPU backend is deliberately absent: it has not been measured here, and
#: "unmeasured" is not "available".
FLASH_ATTENTION_BACKENDS = frozenset({"vulkan", "rocm"})

#: What ``-fa`` is set to when flash attention is used. Never ``1``: forcing it
#: on is a measured *loss* on some model and backend pairs — 848 against 1544
#: prefill on the 20 B under Vulkan — and a large win on others, so llama.cpp
#: decides per model and Vaelor does not overrule it globally.
FLASH_ATTENTION_MODE = "auto"

def flash_attention_support(
    accelerators: Optional[Sequence[Mapping[str, Any]]] = None,
    backend: str = "",
) -> Dict[str, Any]:
    """Whether the V cache may be quantized on this host, and on what evidence.

    This is the missing fact. Without it every ``cache_type_v`` request was
    downgraded to f16 with a reason nobody ever saw, because the key the
    planner reads was never populated by the inventory that feeds it.
    """
    requested = str(backend or "").strip().lower()
    candidates = (
        {requested} if requested
        else set(FLASH_ATTENTION_BACKENDS) if _primary(accelerators) else set()
    )
    usable = sorted(candidates & FLASH_ATTENTION_BACKENDS)
    if usable:
        return {
            "available": True,
            "mode": FLASH_ATTENTION_MODE,
            "backends": usable,
            "source": "measured",
            "reason": (
                "Flash attention and a quantized V cache are available on the "
                "{} backend{}. The win was measured on {}: q8_0 K+V on a 3 B "
                "at {} tokens gave {:.0%} VRAM, {:+.1%} prefill, {:+.1%} "
                "decode and {:+.1%} TTFT against f16. `-fa` stays `{}` so "
                "llama.cpp decides per model."
            ).format(
                ", ".join(usable), "" if len(usable) == 1 else "s",
                KV_QUANTIZATION_MEASUREMENT["backend"],
                KV_QUANTIZATION_MEASUREMENT["context_tokens"],
                KV_QUANTIZATION_MEASUREMENT["vram_change"],
                KV_QUANTIZATION_MEASUREMENT["prefill_change"],
                KV_QUANTIZATION_MEASUREMENT["decode_change"],
                KV_QUANTIZATION_MEASUREMENT["ttft_change"],
                FLASH_ATTENTION_MODE,
            ),
        }
    return {
        "available": False,
        "mode": "",
        "backends": [],
        "source": "unmeasured",
        "reason": (
            "No accelerated backend with measured flash-attention support was "
            "found, so the KV cache stays f16."
        ),
    }


# ---------------------------------------------------------------------------
# Per-model runtime flags
# ---------------------------------------------------------------------------

#: Flags every model gets unless it says otherwise.
#:
#: ``ubatch`` of 1024 stays the default, but **its headline number is no longer
#: current**. The +18.9% prefill and −15.8% TTFT on the 20 B were measured on
#: snap revision 360 under ROCm and have never been re-validated on 378 - the
#: build in which ROCm behaviour changed enough to reverse the backend
#: conclusion. The figures are kept as ``superseded_*`` in
#: ``MODEL_FLAG_MEASUREMENTS``, the same way the other revision-360 numbers
#: were handled, and nothing quotes them as measured until they are re-run.
#: The flag itself is retained because it was also best-in-class under Vulkan
#: stacked with a q8 KV cache, and reverting a setting on the strength of a
#: stale measurement would be the same mistake in the other direction.
DEFAULT_MODEL_FLAGS = {
    "flash_attention_mode": FLASH_ATTENTION_MODE,
    "ubatch": 1024,
    "cache_type_k": "q8_0",
    "cache_type_v": "q8_0",
    "context_tokens": RECOMMENDED_CONTEXT_TOKENS,
}

#: Per-model overrides, matched case-insensitively as a substring of the model
#: name or file. Flags are per-model because the measurements are: `-fa` forced
#: on is a large win on the 3 B under ROCm and a large loss on the 20 B under
#: Vulkan, so there is no one global answer and structuring it as one would be
#: inventing agreement the data does not have.
MODEL_FLAG_OVERRIDES: Dict[str, Dict[str, Any]] = {
    "gpt-oss-20b": {
        # Measured 848 prefill with `-fa 1` against 1544 with `-fa auto` under
        # Vulkan. `auto` is not a default here, it is the measured answer.
        "flash_attention_mode": "auto",
        "ubatch": 1024,
    },
    "gemma-4-26b-a4b": {
        # A quantized KV cache is not universally free. This model loses 26% of
        # its prefill to q8_0 K+V on ROCm, reproduced in both benchmark passes,
        # so the global default is wrong for it and it keeps f16.
        "cache_type_k": "f16",
        "cache_type_v": "f16",
    },
}

def model_runtime_flags(model_name: str = "") -> Dict[str, Any]:
    """Measured llama.cpp flags for one model.

    Per-model rather than global, because ``-fa`` is measured as a large win on
    one model and a large loss on another. A caller that wants a global answer
    is asking a question the measurements do not have.
    """
    flags = dict(DEFAULT_MODEL_FLAGS)
    needle = str(model_name or "").casefold()
    for key, override in MODEL_FLAG_OVERRIDES.items():
        if key.casefold() in needle:
            flags.update(override)
            flags["override_matched"] = key
            break
    return flags


# ---------------------------------------------------------------------------
# Loaded-model capacity
# ---------------------------------------------------------------------------

#: How many models the two-tier Z2 design needs resident at once: the assistant
#: model on the NPU and the chat model on the GPU. Z2-only (VD-071/VD-072) - a
#: CPU-only Pi keeps 1. A server left at 1 where 2 are needed evicts the first
#: when the second loads, returns ``rc 0`` and warns about nothing.
RECOMMENDED_MAX_LOADED_MODELS = 2

#: The knob's name on the servers Vaelor drives, so the setting can be written
#: rather than described.
MAX_LOADED_MODELS_SETTING = "max_loaded_models"


def loaded_model_settings(
    tiers: int = 1, *, kinds: Optional[Sequence[str]] = None
) -> Dict[str, Any]:
    """``max_loaded_models`` for the tiers this machine actually keeps resident.

    Two-tier residency (an NPU assistant model and a GPU chat model) is Z2-only
    (VD-071/VD-072): it needs both accelerators. A CPU-only Pi runs one model for
    both surfaces, so its setting is 1; forcing 2 reserves memory for a model it
    never loads. ``kinds`` is the resident tier kinds, and the default is 1, not
    2 - a bare call must not assume the two-tier machine.
    """
    present = [str(k) for k in (kinds or [])]
    wanted = max(1, int(tiers or 1))
    if wanted >= 2 and "npu" in present and "gpu" in present:
        reason = (
            "Vaelor keeps {} models resident: the assistant model on the neural "
            "processor and the chat model on the GPU. At the default of 1 the "
            "second load evicts the first, exits 0, and reports nothing."
        ).format(wanted)
    else:
        where = {
            "cpu": "the CPU", "gpu": "the accelerator",
            "npu": "the neural accelerator",
        }.get(present[0] if present else "cpu", "this machine")
        reason = (
            "This machine keeps {} local model resident on {}, serving both the "
            "Assistant and AI Chat, so max_loaded_models is {}. Keeping two "
            "models resident at once is a Z2-only design (VD-071) that needs "
            "accelerators this machine does not have."
        ).format(wanted, where, wanted)
    return {MAX_LOADED_MODELS_SETTING: wanted, "reason": reason}


def loaded_models(listing: Any) -> List[str]:
    """Model ids a server reports as currently loaded.

    Tolerant of the shapes real servers use: ``{"data": [...]}`` as OpenAI
    defines it, a bare list, and either a ``loaded`` boolean or a ``state`` of
    ``"loaded"`` per entry. A listing that marks nothing as loaded is reported
    as nothing loaded rather than as everything loaded.
    """
    entries: Iterable[Any]
    if isinstance(listing, Mapping):
        entries = listing.get("data") or listing.get("models") or []
    elif isinstance(listing, (list, tuple)):
        entries = listing
    else:
        return []
    result: List[str] = []
    for entry in entries:
        if isinstance(entry, str):
            result.append(entry)
            continue
        if not isinstance(entry, Mapping):
            continue
        name = str(entry.get("id") or entry.get("model") or entry.get("name") or "")
        if not name:
            continue
        state = str(entry.get("state") or entry.get("status") or "").casefold()
        if entry.get("loaded") is True or state == "loaded":
            result.append(name)
    return result


def offered_models(listing: Any) -> List[str]:
    """Every model id a server lists, loaded or not.

    ``loaded_models`` answers "what is resident"; this answers "is there
    anything here at all". They are different questions and only the second one
    distinguishes a server that is reachable and empty from a server that is
    not answering - a distinction the Assistant was not making while AI Chat
    was, on the same endpoint, in the same second.
    """
    entries: Iterable[Any]
    if isinstance(listing, Mapping):
        entries = listing.get("data") or listing.get("models") or []
    elif isinstance(listing, (list, tuple)):
        entries = listing
    else:
        return []
    result: List[str] = []
    for entry in entries:
        if isinstance(entry, str) and entry:
            result.append(entry)
            continue
        if not isinstance(entry, Mapping):
            continue
        name = str(entry.get("id") or entry.get("model") or entry.get("name") or "")
        if name:
            result.append(name)
    return result


def detect_model_eviction(
    expected: Sequence[str], listing: Any, *, exit_code: int = 0
) -> Dict[str, Any]:
    """Report a model that was loaded and is not any more.

    A server whose ``max_loaded_models`` is 1 unloads the resident model to make
    room for the next one, and says so nowhere: the request succeeds, the
    process exits 0, and the tier that was working silently stops existing. An
    exit code of 0 is therefore not evidence of anything here, which is why it is
    an input to this function rather than a short-circuit out of it.
    """
    wanted = [str(name) for name in expected if str(name)]
    still_loaded = set(loaded_models(listing))
    evicted = [name for name in wanted if name not in still_loaded]
    if not evicted:
        return {
            "evicted": [],
            "ok": True,
            "exit_code": int(exit_code),
            "loaded": sorted(still_loaded),
            "message": "",
        }
    return {
        "evicted": evicted,
        "ok": False,
        "exit_code": int(exit_code),
        "loaded": sorted(still_loaded),
        "message": (
            "{} was loaded and is not any more. The model server unloaded it to "
            "make room, and reported success (exit {}) while doing so. Raise {} "
            "to {} so both tiers can stay resident."
        ).format(
            ", ".join(evicted), int(exit_code), MAX_LOADED_MODELS_SETTING,
            RECOMMENDED_MAX_LOADED_MODELS,
        ),
    }


# ---------------------------------------------------------------------------
# The deployable plan
# ---------------------------------------------------------------------------

def npu_tier_plan(
    model: str = "", context_tokens: int = NPU_RECOMMENDED_CONTEXT_TOKENS
) -> Dict[str, Any]:
    """Arguments for the assistant tier — when there is a model fit to run.

    The context flag is present on every plan this emits, because omitting it
    is not a neutral default on FLM: it preallocates KV for the full
    131,072-token window whether or not anything will use it. A 3 B measured
    17.4 GB that way and 4.0 GB at 8192 — and was marginally *faster* at the
    smaller window. 13.4 GB was being returned for nothing.

    **The capability guard now reaches the plan.** ``recommended_npu_model``
    computes ``usable`` from the measured schema-valid rate, and this function
    used to return a model name and a startable argument list regardless of it:
    a caller reading ``plan["model"]`` got a deployable configuration for a
    model the selector had just said could not do structured agent work. That
    is the same shape as the KV-cache defect — a guard computed and then not
    applied to the thing it guards — and it is latent only because four models in
    the shipped table reach 100%. Re-measuring makes it live. When the guard
    fails, this returns no model and no arguments: there is nothing to launch,
    rather than something that launches and cannot work.
    """
    selection = recommended_npu_model()
    pinned = str(model or "")
    chosen = pinned or str(selection["model"])
    # A pinned model is judged on its own row, not on the selection's. The
    # selection describes the model *it* picked, so reusing its verdict for a
    # model the operator named would be answering a question nobody asked.
    capability = npu_model_capability(pinned) if pinned else {
        "model": chosen,
        "usable": bool(selection["usable"]),
        "measured": True,
        "schema_valid_rate": selection.get("schema_valid_rate"),
        "reason": "" if selection["usable"] else str(selection["reason"]),
    }
    usable = bool(capability["usable"])
    # Vaelor's identifier for a model is not the name `flm-real` answers to.
    # Under VD-001 and VD-002 Vaelor launches that process itself, so the plan
    # carries the tag that goes on the command line rather than leaving each
    # caller to derive one: `gemma3-4b-FLM` sent to `flm-real serve` comes back
    # as "model not found" on a model that is installed and working.
    tag = flm_model_tag(chosen) if usable else {
        "tag": "",
        "known": False,
        "installed": False,
        "reason": "This tier emits no model, so there is no tag to launch with.",
    }
    capability_reason = str(capability["reason"]) if usable else (
        "{} No model and no arguments are emitted for this tier: a "
        "configuration that starts and cannot answer in the required shape is "
        "worse than none, because it fails at request time instead of at "
        "planning time."
    ).format(capability["reason"])
    window = max(1, int(context_tokens or NPU_RECOMMENDED_CONTEXT_TOKENS))
    default_bytes = NPU_CONTEXT_FOOTPRINT_BYTES.get(NPU_NATIVE_CONTEXT_TOKENS, 0)
    # Only two windows have been measured. Interpolating between them would be
    # inventing a measurement, and the naive `.get(window, 0)` was worse than
    # that: an unmeasured window scored zero, so the saving came out as the
    # whole 17.4 GB and a *larger* window than 8192 appeared to reclaim more
    # than a smaller one. An unmeasured window now reports no figure at all.
    sized_bytes = NPU_CONTEXT_FOOTPRINT_BYTES.get(window)
    measured_window = sized_bytes is not None
    # Not "the flag is set" when nothing is emitted: this string is shown to
    # operators, and a plan with an empty argument list claiming to set a window
    # would be describing a deployment that is not going to happen.
    if not usable:
        context_reason = "This tier emits no arguments, so no context window is set."
    elif measured_window:
        context_reason = (
            "{} {} is set explicitly. Left unset, FLM preallocates KV "
            "for its full {}-token window: a 3 B measured {:.1f} GB that way "
            "against {:.1f} GB here, and was marginally faster at the smaller "
            "window."
        ).format(
            NPU_CONTEXT_FLAG, window, NPU_NATIVE_CONTEXT_TOKENS,
            default_bytes / 1_000_000_000, sized_bytes / 1_000_000_000,
        )
    else:
        context_reason = (
            "{} {} is set explicitly so FLM does not preallocate KV for "
            "its full {}-token window. This window has not been measured, so "
            "how much that saves is not known."
        ).format(NPU_CONTEXT_FLAG, window, NPU_NATIVE_CONTEXT_TOKENS)
    return {
        "kind": "npu",
        "role": "assistant",
        # Blank, not the rejected name: a caller reading this key is reading it
        # to launch something, and there is nothing here fit to launch.
        "model": chosen if usable else "",
        # The name that goes to `flm-real serve`, beside the name Vaelor uses.
        # Blank when nothing is launchable, for the same reason `model` is.
        "flm_tag": str(tag["tag"]),
        "flm_tag_known": bool(tag["known"]),
        "flm_tag_installed": bool(tag["installed"]),
        "flm_tag_reason": str(tag["reason"]),
        "usable": usable,
        "capability": capability,
        "capability_reason": capability_reason,
        "context_tokens": window,
        "arguments": [NPU_CONTEXT_FLAG, str(window)] if usable else [],
        "selection": selection,
        "native_tool_calling": NPU_NATIVE_TOOL_CALLING,
        "orchestration": (
            "Vaelor orchestrates this tier with structured JSON intent it "
            "executes itself. FLM has no usable native tool calling."
        ),
        "context_reason": context_reason,
        # None, not zero: "we did not measure this window" is not "this window
        # saves nothing".
        "reclaimed_bytes": (
            max(0, default_bytes - sized_bytes) if measured_window else None
        ),
        "footprint_measured": measured_window,
        "measured_windows": sorted(NPU_CONTEXT_FOOTPRINT_BYTES),
    }


def gpu_tier_plan(
    model: str = CHAT_TIER_MODEL,
    *,
    context_tokens: int = GPU_RECOMMENDED_CONTEXT_TOKENS,
    flash_attention_available: bool = True,
) -> Dict[str, Any]:
    """Arguments for the AI Chat tier: the measured GPU configuration.

    The KV cache types are resolved by :func:`resolve_kv_cache_types` rather
    than decided here, because this function used to decide them itself and got
    it half right: it downgraded ``cache_type_v`` when flash attention was
    unavailable and left ``cache_type_k`` at ``q8_0``. The plan it emitted was
    ``-ctk q8_0 -ctv f16`` with no ``-fa`` — the configuration measured at 1.01
    tokens per second against 64.22, which starts and serves and warns nobody.

    **#247s reconciliation.** ``CHAT_TIER_MODEL`` is ``gpt-oss-20b-GGUF``, and
    the ``model`` this returns is what the idle-GPU System card names. That was a
    phantom (it named a model the catalog did not stock); #247m added gpt-oss-20b
    to the catalog AND the stock-GGUF-on-GPU AI-Chat route
    (:meth:`ExecutorModelDeployMixin._deploy_model` on a ``surface="ai-chat"``
    entry), so the name is now a real, offered, deployable chat model whose
    measured flags (from :func:`model_runtime_flags`) are gpt-oss's — the tier's
    name and its arguments are the same model. The catalog's *headline* GPU
    recommendation stays the owner-deployed ROCmFP4 27B, which is a different
    tier (the fork path, not these stock llama.cpp arguments), and
    ``inference_status`` already shows the actually-resident model whenever an
    ai-chat lease is active, so the two only differ on an idle GPU. See
    ``tests/test_inference_tuning.py::…is_no_longer_a_phantom`` for the guard
    tying this constant to the catalog.
    """
    flags = model_runtime_flags(model)
    window = max(1, int(context_tokens or GPU_RECOMMENDED_CONTEXT_TOKENS))
    cache = resolve_kv_cache_types(
        str(flags["cache_type_k"]),
        str(flags["cache_type_v"]),
        flash_attention_available=flash_attention_available,
    )
    cache_k = cache["cache_type_k"]
    cache_v = cache["cache_type_v"]
    arguments = ["--ctx-size", str(window), "-ctk", cache_k, "-ctv", cache_v]
    if flash_attention_available:
        arguments += ["-fa", str(flags["flash_attention_mode"])]
    arguments += ["-ub", str(flags["ubatch"])]
    return {
        "kind": "gpu",
        "role": "chat",
        "model": str(model or CHAT_TIER_MODEL),
        "context_tokens": window,
        "cache_type_k": cache_k,
        "cache_type_v": cache_v,
        "cache_adjusted": bool(cache["adjusted"]),
        "cache_reason": cache["reason"],
        "flash_attention_mode": (
            str(flags["flash_attention_mode"]) if flash_attention_available else ""
        ),
        "ubatch": int(flags["ubatch"]),
        "arguments": arguments,
        "reason": (
            "-ub {} was best-in-class under Vulkan stacked with a q8 KV cache. "
            "Its ROCm figures ({:+.1%} prefill, {:+.1%} TTFT on the 20 B) are "
            "from snap revision {} and await re-measurement on {}. `-fa` "
            "stays `{}` per model: forcing it on measured {:g} prefill against "
            "{:g} on this model under Vulkan."
        ).format(
            flags["ubatch"],
            MODEL_FLAG_MEASUREMENTS["gpt-oss-20b"][
                "superseded_ubatch_1024_prefill_change"
            ],
            MODEL_FLAG_MEASUREMENTS["gpt-oss-20b"][
                "superseded_ubatch_1024_ttft_change"
            ],
            SUPERSEDED_SNAP_REVISION, LEMONADE_SNAP_REVISION,
            FLASH_ATTENTION_MODE,
            MODEL_FLAG_MEASUREMENTS["gpt-oss-20b"]["flash_attention_forced_on_prefill_tps"],
            MODEL_FLAG_MEASUREMENTS["gpt-oss-20b"]["flash_attention_auto_prefill_tps"],
        ),
    }


def recommended_deployment(
    hardware: Optional[Mapping[str, Any]] = None,
    *,
    model_bytes: int = 0,
    kv_cache_bytes: int = 0,
    override: str = "",
) -> Dict[str, Any]:
    """The whole configuration the benchmarks recommend, in one object.

    What the Assistant and the UI report from, so every element carries its
    reason. Tiers and counts are gated on ``hardware`` (VD-071/VD-072).
    """
    # Deferred: `model_sizing` owns the gating predicate and imports the tier
    # planners from this module (#98).
    from .model_sizing import (
        cpu_tier_plan, describe_inference_machine, resident_tier_kinds,
    )
    facts = dict(hardware or {})
    accelerators = facts.get("accelerators") or []
    neural = facts.get("neural_accelerators") or []
    backend = select_inference_backend(
        model_bytes=model_bytes,
        kv_cache_bytes=kv_cache_bytes,
        accelerators=accelerators,
        override=override,
        system_memory_bytes=int(facts.get("memory_total_bytes") or 0),
    )
    flash = flash_attention_support(accelerators, backend["backend"])
    # NPU only with a neural accelerator, GPU only with a GPU, else one CPU model
    # for both surfaces. Emitting NPU+GPU regardless is the #205 defect - it
    # reported the Z2's two-tier plan on a CPU-only Pi.
    builders = {
        "npu": npu_tier_plan,
        "gpu": lambda: gpu_tier_plan(flash_attention_available=flash["available"]),
        "cpu": cpu_tier_plan,
    }
    kinds = resident_tier_kinds(facts)
    tiers = [builders[kind]() for kind in kinds]
    return {
        # This machine, not `MEASURED_ON` (the Z2 bench box, carried as labelled
        # provenance below). Reporting the reference box as this machine's own
        # status is the #205 lie.
        "measured_on": describe_inference_machine(facts),
        "benchmarked_on": MEASURED_ON,
        "is_reference_machine": bool(neural and accelerators),
        "backend": backend,
        "flash_attention": flash,
        "loaded_models": loaded_model_settings(len(tiers), kinds=kinds),
        "context": {
            "recommended_tokens": RECOMMENDED_CONTEXT_TOKENS,
            "agent_prompt_tokens": AGENT_PROMPT_TOKENS,
            "reason": (
                "The appliance's own agent prompt measures {} tokens, so a "
                "{}-token window cannot hold it. {} is the recommended window."
            ).format(AGENT_PROMPT_TOKENS, 4096, RECOMMENDED_CONTEXT_TOKENS),
        },
        "tiers": tiers,
    }
