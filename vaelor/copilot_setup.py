"""Hardware discovery and conservative model guidance for the setup assistant."""

from __future__ import annotations

import shutil
from pathlib import Path
from typing import Any, Dict

from .hardware_platform import board_info
from .inference_tuning import (
    AGENT_PROMPT_TOKENS,
    GPU_RECOMMENDED_CONTEXT_TOKENS,
    RECOMMENDED_CONTEXT_TOKENS,
    RECOMMENDED_RUNTIME_MODE,
    flash_attention_support,
    recommended_deployment,
)
from .model_catalog import (
    catalog_models,
    catalog_runtime,
    headroom_note,
    largest_within,
    npu_release_models,
)
from .model_footprint import identify, normalise_platform, reservation_bytes
from .model_sizing import (
    estimate_kv_cache_bytes,
    model_memory_budget,
    parallel_slots,
    plan_kv_cache,
    recommended_model_tier,
)
from .platforms.accelerators import accelerator_summary
from .runtime_paths import data_path


GIB = 1024 ** 3
MIB = 1024 ** 2
#: Applied to a *measured* footprint before it becomes a container limit.
#: Estimates are not scaled by it - an estimate is already uncertain in an
#: unknown direction, and padding one would disguise that.
MEASURED_LIMIT_MARGIN = 1.15
#: Decimal gigabytes, for anything whose label says "GB". Disk vendors, `df`
#: and every other surface in this product use decimal units; dividing by 1024³
#: and writing GB is a 7% understatement wearing the right suffix.
GB = 1000 ** 3


def _linux_memory() -> tuple[int, int]:
    values: Dict[str, int] = {}
    try:
        for line in Path("/proc/meminfo").read_text(encoding="utf-8").splitlines():
            name, raw = line.split(":", 1)
            values[name] = int(raw.strip().split()[0]) * 1024
    except (OSError, ValueError):
        return 0, 0
    return values.get("MemTotal", 0), values.get("MemAvailable", 0)


def hardware_inventory(model_root: str = data_path("models")) -> Dict[str, Any]:
    """Return only the hardware facts needed to make a model recommendation."""
    memory_total, memory_available = _linux_memory()
    board = board_info()
    storage_path = Path(model_root)
    while not storage_path.exists() and storage_path.parent != storage_path:
        storage_path = storage_path.parent
    try:
        storage_free = shutil.disk_usage(storage_path).free
    except OSError:
        storage_free = 0
    summary = accelerator_summary()
    # Flash attention is the fact the KV planner reads and this inventory never
    # supplied. Without the key `plan_kv_cache` saw `None`, treated it as
    # "unavailable", and silently downgraded every requested `cache_type_v` to
    # f16 — so the measured KV-quantization win (-29% VRAM, +16% prefill, +21%
    # decode, -14% TTFT on a 3B at 32k) could not be produced on any machine.
    flash = flash_attention_support(summary["accelerators"])
    return {
        "device": board["model"],
        "architecture": board["architecture"],
        # Cores and threads, kept apart. `cpu_cores` used to be
        # `os.cpu_count()` — logical CPUs — under a name that says otherwise,
        # and every reader inherited the confusion.
        "cpu_cores": board["cpu_cores"],
        "cpu_threads": board.get("cpu_threads") or board["cpu_cores"],
        "cpu_cores_are_threads": bool(board.get("cpu_cores_are_threads")),
        "is_raspberry_pi": board["is_raspberry_pi"],
        "vendor": board.get("vendor", ""),
        # Model sizing and setup readiness both need to know whether there is
        # an accelerator and whether the service account could actually open
        # it. Discovery alone is not permission.
        "accelerators": summary["accelerators"],
        "accelerator_detected": summary["detected"],
        "accelerator_access": summary["access"],
        "neural_accelerators": summary["npus"],
        "device_grants": summary["grants"],
        "flash_attention": flash["available"],
        "flash_attention_mode": flash["mode"],
        "flash_attention_source": flash["source"],
        "flash_attention_reason": flash["reason"],
        "memory_total_bytes": memory_total,
        "memory_available_bytes": memory_available,
        "storage_free_bytes": storage_free,
        "model_storage": model_root,
    }


_RUNTIME_MODES = [
    {
        "id": "efficient",
        "name": "Memory saver",
        "description": "Shorter context and a smaller RAM ceiling leave more room for apps.",
    },
    {
        "id": "balanced",
        "name": "Balanced",
        "description": "Recommended mix of response quality, context, and free system memory.",
    },
    {
        "id": "quality",
        "name": "Long context",
        "description": "Uses more memory for longer conversations and larger prompts.",
    },
]

#: Space left free after a model download, so the appliance still has room to
#: log, update and write a checkpoint.
#:
#: Declared before the recommendation that reads it because both the CPU and
#: the accelerated paths now size against a specific model rather than a flat
#: floor: a size a caller cannot compare to free space is not a check, and a
#: flat floor offered a 42 GB model on a disk with 8 GiB left.
STORAGE_HEADROOM_BYTES = 4 * GIB


def _storage_verdict(
    hardware: Dict[str, Any], download_bytes: int
) -> tuple[bool, str]:
    """Whether this specific model fits the disk, and what the numbers were.

    The old check was a flat 8 GiB floor applied to every tier, so a 42 GB
    model was offered as installable on a disk with 8 GiB left. A size the
    caller cannot compare against free space is not a check.

    **The sentence divides by 1000³ because it says GB.** It divided by 1024³
    and said GB, which understates free space by 7% - small enough to look
    plausible and large enough to be wrong, which is how it survived. The
    thresholds stay in binary units, where they were chosen; only the display
    changes.
    """
    free = int(hardware.get("storage_free_bytes", 0) or 0)
    needed = max(0, int(download_bytes or 0))
    if not free:
        # Free space unknown. The download path re-checks before writing, so
        # this does not block; it says which it is.
        return True, "Free disk space could not be read, so it was not checked."
    if not needed:
        ok = free >= 8 * GIB
        return ok, (
            "" if ok else
            "This appliance has {:.1f} GB free, which is below the {:.1f} GB floor."
            .format(free / GB, 8 * GIB / GB)
        )
    required = needed + STORAGE_HEADROOM_BYTES
    if free >= required:
        return True, ""
    return False, (
        "This model is about {:.1f} GB and this appliance has {:.1f} GB free. "
        "It needs {:.1f} GB to download and leave room to work."
    ).format(needed / GB, free / GB, required / GB)


#: Context window offered with each tier of recommendation.
#:
#: **Every rung must hold the appliance's own agent prompt.** These were 2048,
#: 4096 and 8192, and while
#: :data:`~vaelor.inference_context.AGENT_PROMPT_TOKENS` read 4,668 the first
#: two were below it, so a machine on either rung was offered a model that could
#: never be given its instructions. Measuring the prompt put it at 1,344
#: (VD-075) and every rung now clears it - which is a change in the arithmetic,
#: not in the rule. Derived from the constant rather than written beside it, so
#: a change to the prompt cannot leave a rung behind; that derivation is why
#: this needed no edit when the constant moved.
_TIER_CONTEXT_TOKENS = {
    tier: max(tokens, RECOMMENDED_CONTEXT_TOKENS)
    for tier, tokens in {"compact": 2048, "balanced": 4096, "capable": 8192}.items()
}

#: How conservative each tier is, expressed as a parameter ceiling rather than
#: as a model. A CPU appliance with 8 GB of RAM *can* hold a 4 B model and is
#: still offered the smaller one first, because generation speed on a CPU is
#: the binding constraint rather than memory; the larger one sits beside it as
#: an alternative. Written as a ceiling so that no model name exists anywhere
#: outside :mod:`vaelor.model_catalog` - a name here could outlive the entry it
#: refers to, which is precisely the failure being fixed.
_TIER_CEILING = {
    "compact": "1.7B",
    # 1.7B until 2026-08-08, on the reasoning that CPU generation above it is
    # too slow to be a usable assistant. Measurement overrides the caution: on
    # a Pi the 4B answers in ~20 s against the 1.7B's ~10 s, and scores 92.2%
    # against 68.9% - diagnosis 92.9% against 48.2%. **The 1.7B cannot
    # diagnose**, which is most of what an appliance assistant is for, so
    # halving the latency of a wrong answer is not the better trade (VD-070,
    # VD-071).
    #
    # This is a *recommendation*, not an install. Nothing here fetches a model;
    # the owner chooses, and the 1.7B remains in the catalog beside it for
    # anyone who wants the faster, smaller one.
    "balanced": "4B",
    "capable": "4B",
    # The accelerated ladder is bounded by the hardware, not by caution.
    "accelerated": "999B",
}


def _entry_footprint_fits(hardware: Dict[str, Any], item: Dict[str, Any]) -> bool:
    """Whether this catalogued model's FOOTPRINT fits this machine, honestly.

    The one gate that lets a mixture-of-experts model be surfaced by its small
    active size without over-offering (#247m): it runs the SAME sizing the
    deploy runs - :func:`local_runtime_settings`, which refuses with a
    ``ValueError`` when the model cannot be held while preserving memory for the
    operating system - so a surfaced alternative is one the deploy will accept.
    Reusing the deploy's own fit rather than a second byte comparison is what
    keeps the recommendation and the deploy from disagreeing (LESSONS 6). Only a
    ``ValueError`` (the fit refusal) is caught; anything else is a real fault and
    is not swallowed.
    """
    try:
        local_runtime_settings(
            hardware,
            int(item.get("download_bytes") or 0),
            RECOMMENDED_RUNTIME_MODE,
            repo=str(item.get("repo") or ""),
            file=str(item.get("file") or ""),
        )
    except ValueError:
        return False
    return True


def recommend_local_models(hardware: Dict[str, Any]) -> Dict[str, Any]:
    """Choose a local model **from the catalog**, sized to this machine.

    The sizing ladder is unchanged: the CPU one stops at 8 B because CPU
    generation above that is too slow to be a usable assistant, and an
    accelerator gets its own ladder measured against GPU memory. What changed
    is what the ladder's answer is *used for*. It used to be looked up in a
    table of model names, so a machine whose budget reached the 32 B rung was
    told to "Install Qwen3 32B" - a model no part of this appliance could
    fetch, beside a size belonging to a different model, above a button that
    installed a third one.

    The rung is now a **ceiling** over :mod:`vaelor.model_catalog`, which holds
    only models with a repository and a file behind them. A machine that
    outruns the catalog is told that about its hardware rather than promised a
    model, because the honest thing to widen is the catalog, not the sentence.
    """
    memory_bytes = int(hardware.get("memory_total_bytes", 0) or 0)
    memory_gib = memory_bytes / GIB
    accelerators = hardware.get("accelerators") or []
    budget = model_memory_budget(memory_bytes, accelerators)
    hardware_tier = recommended_model_tier(memory_bytes, accelerators)
    accelerated = budget["source"] == "accelerator"
    tier = (
        "accelerated" if accelerated
        else "compact" if memory_gib < 5
        else "balanced" if memory_gib < 11
        else "capable"
    )
    # Two ceilings, and the lower one wins: what the hardware can hold, and how
    # far this tier is willing to go on it. The primary stays on NOMINAL params
    # (`largest_within`): a machine is recommended a conservative dense pick, and
    # a mixture-of-experts model never becomes the primary through a low active
    # count it cannot back with footprint.
    affordable = largest_within(hardware_tier)
    primary = largest_within(_TIER_CEILING[tier])
    if primary["parameter_billions"] > affordable["parameter_billions"]:
        primary = affordable
    affordable_billions = affordable["parameter_billions"]
    affordable_effective = affordable.get("effective_billions", affordable_billions)
    alternatives = []
    for item in catalog_models():
        if item["id"] == primary["id"]:
            continue
        if item["parameter_billions"] <= affordable_billions:
            # Unchanged: an entry the nominal ladder already affords. Every dense
            # model reaches here exactly as before, so nothing about the existing
            # recommendations moves.
            alternatives.append(item)
        elif (
            item.get("effective_billions", item["parameter_billions"])
            <= affordable_effective
            and _entry_footprint_fits(hardware, item)
        ):
            # #247m / LESSONS 12/5: a mixture-of-experts model the nominal ladder
            # hid, whose ACTIVE size is within the affordable tier AND whose
            # FOOTPRINT the honest deploy-time byte fit accepts on this machine.
            # The byte fit is what keeps this from over-offering: a ~17 GiB MoE
            # with a tiny active count is surfaced only where its bytes actually
            # fit, never on a box too small to hold it - so the recommendation
            # and the deploy cannot disagree (LESSONS 6). Dense entries never
            # reach this branch (their effective size equals their nominal one,
            # already handled above).
            alternatives.append(item)
    storage_ok, storage_reason = _storage_verdict(
        hardware, int(primary.get("download_bytes") or 0)
    )
    headroom = headroom_note(hardware_tier)
    rationale = (
        "Sized against the accelerator's memory carve-out and shared aperture "
        "rather than total system RAM."
        if accelerated else
        "A {} Q4 model leaves room for the operating system, Docker, and the "
        "model's working memory.".format(primary["parameter_size"])
    )
    context_tokens = 16384 if accelerated else _TIER_CONTEXT_TOKENS[tier]

    def _memory_note(item) -> str:
        """The running memory cost, stated where the choice is made (#146).

        The only figure the setup card carried was the download size — disk,
        not RAM — so the one route to the running cost before committing sat
        behind the button that changes the machine. Measured figures come
        from the footprint table; an unmeasured combination says so instead
        of guessing.
        """
        from .model_footprint import normalise_platform, reservation_bytes

        platform = normalise_platform(hardware.get("architecture"))
        if not platform:
            return ""
        reservation = reservation_bytes(
            str(item.get("id") or ""), platform, context_tokens, 1
        )
        if reservation.get("bytes"):
            return (
                "While serving it holds about {:.1f} GiB of its own memory "
                "allowance, measured on this hardware."
            ).format(reservation["bytes"] / GIB)
        return (
            "Its running memory cost has not been measured on this hardware "
            "at this window yet."
        )

    # A fine-tuned on-device NPU model, on a machine that HAS a neural processor
    # to run it, is the recommended Assistant: faster and more private than any
    # CPU/GPU GGUF, and what this appliance was built around. It is NOT on the RAM
    # ladder above (it runs on the NPU, not system RAM), so it is applied here,
    # after the ladder chose a GGUF - which becomes the top alternative. A machine
    # with no NPU keeps the ladder's pick unchanged, byte-for-byte. It installs
    # through `POST /copilot/install-npu-model` (model.install_release), which the
    # frontend routes to on `backend == "flm-npu"`, not the HF inspect/download.
    catalog_list = list(catalog_models())
    npu_release = list(npu_release_models())
    if hardware.get("neural_accelerators") and npu_release:
        alternatives = [primary] + alternatives
        primary = npu_release[0]
        catalog_list = npu_release + catalog_list
        rationale = (
            "This machine has a neural processor, so its fine-tuned on-device "
            "model is the recommended Assistant - the fastest, most private "
            "option, running on the NPU rather than system RAM."
        )
        context_tokens = 16384

    return {
        "tier": tier,
        # The recommended model's own size, so the header that renders this
        # cannot name a parameter class the catalog does not stock.
        "parameter_range": primary["parameter_size"],
        # What the hardware could hold, kept separate from what is offered.
        # Conflating the two is the whole defect.
        "hardware_tier": hardware_tier,
        "exceeds_catalog": bool(headroom),
        "catalog_note": headroom,
        "catalog": tuple(catalog_list),
        "quantization": primary["quantization"],
        "context_tokens": context_tokens,
        "primary": {**primary, "memory_note": _memory_note(primary)},
        "alternatives": [
            {**item, "memory_note": _memory_note(item)} for item in alternatives
        ],
        "storage_ok": storage_ok,
        "can_install": storage_ok and memory_gib >= 3,
        "storage_reason": storage_reason,
        "accelerator": accelerators[0].get("name") if accelerators else None,
        "accelerator_memory_bytes": budget["accelerator_bytes"],
        "budget_bytes": budget["budget_bytes"],
        "rationale": rationale,
        "verification_required": True,
        "runtime_modes": _RUNTIME_MODES,
    }


def local_runtime_settings(
    hardware: Dict[str, Any],
    model_size_bytes: int,
    requested_mode: str,
    *,
    cache_type_k: str = "f16",
    cache_type_v: str = "f16",
    surface: str = "assistant",
    repo: str = "",
    file: str = "",
) -> Dict[str, Any]:
    """Size llama.cpp while preserving a host-memory reserve.

    When an accelerator is present the weights and the KV cache live in its
    memory, not the host's, so the budget comes from the VRAM carve-out and the
    GTT aperture. Sizing purely from system RAM is what capped a workstation
    with a 16 GiB carve-out and a 22 GiB aperture at an 8 B model.

    ``requested_mode`` of ``"recommended"`` — or of nothing at all — asks for
    the largest window this machine can afford rather than a fixed middle
    setting. That distinction is the whole defect: deploys asked for
    ``"balanced"``, whose window is 4096 tokens, on a machine with a ~34 GB
    budget, while the appliance's own agent prompt measures
    :data:`~vaelor.inference_tuning.AGENT_PROMPT_TOKENS` tokens. The selector
    only ever walks *down* from what it is asked for, so asking for the middle
    setting made the recommended window unreachable on any hardware.

    ``surface`` decides how many slots the server is launched with, and the KV
    cache is charged for every one of them. This is the half of #109 that is
    not about a flag: the container limit returned here was derived from a
    per-token figure with no slot multiplier anywhere in it, so it was correct
    for the window Vaelor asked for and wrong by the slot count for the window
    llama.cpp actually built.
    """
    total = max(0, int(hardware.get("memory_total_bytes", 0) or 0))
    available = max(0, int(hardware.get("memory_available_bytes", total) or 0))
    # llama.cpp's `--threads` schedules on logical CPUs, so this is the one
    # place that genuinely wants the thread count rather than the core count.
    # It read `cpu_cores` when that field meant threads; now the field means
    # cores, so the thread count is asked for by name.
    cores = max(1, int(
        hardware.get("cpu_threads") or hardware.get("cpu_cores", 1) or 1
    ))
    model_size = max(1, int(model_size_bytes or 0))
    accelerators = hardware.get("accelerators") or []
    accelerator_budget = model_memory_budget(total, accelerators)
    reserve = max(int(1.5 * GIB), int(total * 0.25))
    safe_budget = max(
        int(1.5 * GIB),
        min(
            max(0, total - reserve),
            max(int(1.5 * GIB), available - int(0.5 * GIB)),
        ),
    )
    if accelerator_budget["source"] == "accelerator":
        safe_budget = max(safe_budget, accelerator_budget["budget_bytes"])
    cache = plan_kv_cache(
        cache_type_k,
        cache_type_v,
        flash_attention=hardware.get("flash_attention"),
    )
    slots = parallel_slots(surface)
    slot_count = max(1, int(slots["slots"]))
    # **Whose measurement sizes this deploy.** `RECOMMENDED_CONTEXT_TOKENS` is
    # the CPU tier's window, measured on a Pi. This function is the only one on
    # the deploy path that renders a compose file, so reading that constant
    # unconditionally meant a Pi measurement sized every machine: when VD-075
    # took it from 8,192 to 4,096, the Z2's managed local model silently halved
    # with it, on a box with 128 GB of RAM and a GPU, while its memory limit
    # stayed where it was. VD-077 split the constant per accelerator and fixed
    # the two *planning* surfaces; this one kept the shared name and was the
    # only one that deploys.
    #
    # Selected from `accelerator_budget["source"]` rather than by re-inspecting
    # `accelerators`, because this function has already asked that question and
    # a second way to ask it is how the two answers start disagreeing.
    window = (
        GPU_RECOMMENDED_CONTEXT_TOKENS
        if accelerator_budget["source"] == "accelerator"
        else RECOMMENDED_CONTEXT_TOKENS
    )
    modes = {
        "efficient": {
            "context": 2048,
            "threads": min(3, cores),
            "overhead": int(0.75 * GIB),
        },
        "balanced": {
            "context": 4096,
            "threads": min(4, cores),
            "overhead": int(1.25 * GIB),
        },
        "quality": {
            "context": window,
            "threads": min(4, cores),
            "overhead": int(2 * GIB),
        },
    }
    order = ["efficient", "balanced", "quality"]
    # The mode whose window is the recommended one. Anything that is not a named
    # mode - including the empty string and the literal "recommended" - asks for
    # the best this machine can afford, because the selector below can only walk
    # downwards and a middle default therefore caps every machine at 4096.
    # Falls back to the largest mode rather than raising, so lowering the
    # recommended window below every mode's context is a configuration change
    # and not a crash on the deploy path.
    recommended = next(
        (
            name for name in reversed(order)
            if modes[name]["context"] >= window
        ),
        order[-1],
    )
    mode = requested_mode if requested_mode in modes else recommended

    def _estimate(candidate: str) -> int:
        # The per-mode overhead was tuned against **one** f16 KV cache, so it
        # already carries one. Everything the shipped configuration costs
        # beyond that single cache is settled here: a quantized cache is
        # credited as the saving it is, and every slot past the first is
        # charged as the second full copy of the window that it is.
        context_tokens = modes[candidate]["context"]
        planned = estimate_kv_cache_bytes(
            context_tokens,
            cache["cache_type_k"],
            cache["cache_type_v"],
            parallel=slot_count,
        )
        covered = estimate_kv_cache_bytes(context_tokens, "f16", "f16")
        overhead = max(
            modes[candidate]["overhead"] // 2,
            modes[candidate]["overhead"] + (planned - covered),
        )
        return int(model_size * 1.25) + overhead

    selected = mode
    for candidate in reversed(order[:order.index(mode) + 1]):
        if _estimate(candidate) <= safe_budget:
            selected = candidate
            break
    estimate = _estimate(selected)
    if estimate > safe_budget:
        raise ValueError(
            "This model does not fit while preserving memory for the operating system."
        )
    context = modes[selected]["context"]

    # **A window measured on this model outranks the mode's.** The three modes
    # are memory profiles, not evidence: `quality` asks for
    # `RECOMMENDED_CONTEXT_TOKENS`, which was 8,192 because that was the
    # smallest power of two above `AGENT_PROMPT_TOKENS = 4668` - a constant
    # nothing in the tree derived and which measurement put at 1,344. That
    # window's KV cache cost ~460 MiB the Pi did not have and stalled the
    # machine badly enough to need a physical power cycle (VD-075). The
    # constant is now measured and the recommendation is 4,096, but the
    # ceiling below stays: a mode is still a memory profile rather than
    # evidence, so where a catalog entry carries a window somebody actually
    # measured, that window wins here.
    #
    # A *smaller* mode is still honoured: choosing "Memory saver" is a real
    # request for less memory, and nothing measured says 2,048 is unsafe. Only
    # asking for more than the measurement is refused.
    measured_runtime = catalog_runtime(repo, file)
    measured_context = int(measured_runtime.get("context") or 0)
    context_capped_by_measurement = bool(
        measured_context and context > measured_context
    )
    if context_capped_by_measurement:
        context = measured_context

    # A MEASURED FOOTPRINT OUTRANKS THE ESTIMATE. `_estimate` is
    # `model_size * 1.25 + a per-mode overhead`, and for Qwen3 1.7B at 2,048
    # tokens that is roughly 2,600 MiB against a measured steady state of
    # 3,157 MB - 18% low, in the direction that sets a container limit below
    # what the process needs. Three OOM kills on 2026-08-08 came from limits
    # derived this way (#117, VD-070).
    #
    # The lookup is by repository and file, which is what the download path
    # already verifies, rather than by byte count: two publishers' builds of
    # the same model share a filename 1,056 bytes apart (VD-065).
    measured = None
    footprint_source = "estimated"
    model_id = identify(repo, file)
    platform = normalise_platform(
        hardware.get("architecture") or hardware.get("uname_architecture")
    )
    if model_id and platform:
        reservation = reservation_bytes(model_id, platform, context, slot_count)
        if reservation["bytes"] is not None:
            measured = int(reservation["bytes"])
            footprint_source = reservation["source"]

    if measured is not None:
        # Deliberately not `min(measured, safe_budget)`. If the measured
        # requirement exceeds what this machine can spare, that is a refusal to
        # surface, not a number to trim - silently issuing a limit below the
        # measured need is how the appliance was killed under sustained load
        # while its configuration looked correct.
        if measured > safe_budget:
            raise ValueError(
                "This model needs {} MiB measured on this platform at {} tokens, "
                "and only {} MiB can be spared while preserving memory for the "
                "operating system.".format(
                    measured // MIB, context, safe_budget // MIB
                )
            )
        # A DELIBERATE MARGIN, NOT WHATEVER THE ROUNDING LEAVES. Rounding the
        # measured figure up to the next 256 MiB boundary gave the Qwen3 4B at
        # 8,192 exactly 90 MiB of headroom - an accident of arithmetic, not a
        # decision. The plateau behind these numbers was established over 40
        # prompts (#117); an appliance runs for weeks, and the cost of being
        # 2% short is an OOM kill under sustained load.
        #
        # 15% is chosen from the one configuration with a long clean run: the
        # 1.7B at 2,048 held 3,157 MB under a 3,584 MB limit - 13.5% - through
        # 200 evaluation items with zero errors.
        #
        # **CLAMPED, because the margin is part of what the container may take
        # and therefore part of what the machine must be able to spare.** The
        # refusal above tests `measured`; the margin was applied after it and
        # never re-tested, so a model that fitted by 10 MiB was issued a limit
        # 848 MiB beyond what could be spared.
        #
        # That is worse than issuing too little, and the appliance demonstrated
        # why on 2026-08-09. At a 5,120 MiB limit the container exceeded its own
        # cgroup, the kernel killed llama-server, and the machine carried on. At
        # 6,400 MiB the container stayed *inside* its limit, so nothing was ever
        # killed - and everything outside the cgroup starved instead. journald
        # logged "Under memory pressure, flushing caches" and the machine went
        # silent six minutes later with no OOM record at all. It needed a
        # physical power cycle.
        #
        # **The cgroup limit is the machine's protection from the model.**
        # Raising it past what the machine can spare does not buy the model
        # headroom; it removes the one mechanism that keeps a hungry model from
        # taking the appliance down with it, and converts a survivable
        # container restart into an unrecoverable system stall.
        limit_mib = min(
            (
                (int(measured * MEASURED_LIMIT_MARGIN) + 255 * MIB)
                // (256 * MIB)
            ) * 256,
            safe_budget // MIB,
        )
    else:
        limit_mib = max(
            1536,
            min(
                safe_budget // MIB,
                ((estimate + 255 * MIB) // (256 * MIB)) * 256,
            ),
        )
    fits_prompt = context >= AGENT_PROMPT_TOKENS
    return {
        "requested_mode": mode,
        "recommended_mode": recommended,
        "mode": selected,
        "context": context,
        # Whether a mode's window was reduced to one that was measured, so the
        # screen can say so rather than showing "Long context" beside a window
        # that is not long.
        "context_capped_by_measurement": context_capped_by_measurement,
        # Everything else the chosen model needs at runtime, straight from the
        # catalog entry: today that is `slot_save`, which is the only reason
        # the standing brief survives VD-073's idle unload. Without this the
        # deploy path never sees it and `--slot-save-path` never gets rendered
        # on a real appliance - which is exactly what shipped in alpha 29.
        **{
            key: value for key, value in measured_runtime.items()
            if key != "context"
        },
        "threads": modes[selected]["threads"],
        "memory_limit_mib": int(limit_mib),
        # Whether the limit above is an observation or an arithmetic
        # guess. VD-040: the claim carries its basis, and these two are
        # 18% apart on the one model where both exist.
        "footprint_source": footprint_source,
        # How much room the issued limit actually leaves over the measured
        # need, after the clamp. The intended margin is 15%; when the machine
        # cannot spare that, what remains is the number that matters and it is
        # stated rather than left to be derived from three other fields. A
        # single-figure margin here means this model at this window is at the
        # edge of what this appliance can hold with everything else running.
        "limit_margin_percent": (
            round(100.0 * (int(limit_mib) * MIB - measured) / measured, 1)
            if measured else None
        ),
        "measured_footprint_bytes": measured,
        "reserved_for_system_bytes": reserve,
        "safe_budget_bytes": safe_budget,
        "adjusted": selected != mode,
        "budget_source": accelerator_budget["source"],
        "accelerator_budget_bytes": accelerator_budget["accelerator_bytes"],
        # Every slot's cache, because every slot's cache is what has to fit.
        # This key fed the backend decision and the memory limit while counting
        # one slot on a server that built four.
        "kv_cache_bytes": estimate_kv_cache_bytes(
            context,
            cache["cache_type_k"],
            cache["cache_type_v"],
            parallel=slot_count,
        ),
        "kv_cache_bytes_per_slot": estimate_kv_cache_bytes(
            context, cache["cache_type_k"], cache["cache_type_v"]
        ),
        # Stated at launch, never left to the runtime. `context` is per slot,
        # so the tokens of KV this server holds are the product of the two.
        "parallel_slots": slot_count,
        "parallel_slots_surface": slots["surface"],
        "parallel_slots_reason": slots["reason"],
        "context_tokens_total": context * slot_count,
        "cache_type_k": cache["cache_type_k"],
        "cache_type_v": cache["cache_type_v"],
        "flash_attention": cache["flash_attention"],
        "flash_attention_mode": cache["flash_attention_mode"],
        "cache_adjusted": cache["adjusted"],
        "cache_reason": cache["reason"],
        # A window smaller than the appliance's own agent prompt is not a
        # quieter setting, it is a model that cannot be given its instructions.
        # Said out loud rather than discovered as a truncated run.
        "agent_prompt_tokens": AGENT_PROMPT_TOKENS,
        "fits_agent_prompt": fits_prompt,
        "context_reason": "" if fits_prompt else (
            "This {}-token window is smaller than the appliance's own "
            "{}-token agent prompt, so the assistant cannot send its full "
            "instructions to this model."
        ).format(context, AGENT_PROMPT_TOKENS),
    }


def _npu_display_name(model_id: str) -> str:
    """A readable name for an NPU model id, e.g. ``qwen3.5-4b-FLM`` -> ``Qwen3.5 4B``.

    Presentation only. The authoritative identifiers — the model ``id`` and the
    ``flm_tag`` that launches it — are carried verbatim; this only titlecases the
    family and upper-cases a parameter token so the wizard shows a name rather
    than a slug.
    """
    base = model_id[:-4] if model_id.endswith("-FLM") else model_id
    words = []
    for word in base.replace("-", " ").split():
        words.append(word.upper() if word[-1:].lower() == "b" and any(
            character.isdigit() for character in word
        ) else word.capitalize())
    return " ".join(words) or model_id


def _npu_assistant_recommendation(hardware: Dict[str, Any]) -> Dict[str, Any]:
    """Fields overriding the GGUF recommendation when this machine serves on NPU.

    Returns ``{}`` — leaving the recommendation byte-identical — on every machine
    that does not actually serve the assistant on a neural processor. That is the
    Pi isolation: a CPU-only machine has no ``neural_accelerators``, so it never
    reaches discovery and its GGUF recommendation is unchanged.

    When the NPU serving tier IS available (a neural accelerator is present, and
    real capability discovery confirms flm-real, the sysfs device node and a
    usable, installed FLM model), the wizard must present the NPU assistant model
    — VD-108's ``qwen3.5-4b-FLM`` served on the neural processor via flm-real —
    rather than the Pi-style GGUF download the Z2 would never run for its
    assistant. The old status derived its recommendation only from
    ``recommended_model_tier`` over the GGUF catalog and never called
    ``recommended_npu_model``, so on the Z2 it recommended "Qwen3 4B Instruct —
    the Pi's measured best" for a tier that does not run GGUF at all (VD-108).
    """
    neural = hardware.get("neural_accelerators") or []
    if not neural:
        return {}
    from .flm_service import discover_npu_serving
    from .inference_model_choice import NPU_MODEL_CAPABILITY, recommended_npu_model
    from .inference_tuning import npu_tier_plan

    plan = npu_tier_plan()
    if not plan.get("usable") or not plan.get("flm_tag"):
        return {}
    capability = discover_npu_serving(str(plan.get("flm_tag") or ""))
    if not capability.get("available"):
        return {}

    selection = recommended_npu_model()
    model_id = str(selection.get("model") or "")
    if not model_id:
        return {}
    facts = NPU_MODEL_CAPABILITY.get(model_id, {})
    parameter_size = str(facts.get("parameters") or "")
    resident = int(facts.get("resident_bytes") or 0)
    context_tokens = int(plan.get("context_tokens") or 0)
    # One home for the served quantization label: it is both the model's own
    # figure and the tier-level one, and a sentence written twice in a module is
    # exactly what LESSONS 6 / #98 records (it drifts). Read off the machine's
    # inventory (q4nx via FastFlowLM); this tier is served, never downloaded, so
    # there is no GGUF quantization or download size to state.
    npu_quantization = "q4nx (FLM)"
    memory_note = (
        "Holds about {:.1f} GiB of neural-processor memory while serving, "
        "measured on this hardware (VD-108).".format(resident / GIB)
        if resident else
        "Its neural-processor memory cost has not been measured on this "
        "hardware yet."
    )
    primary = {
        "id": model_id,
        "name": _npu_display_name(model_id),
        "parameter_size": parameter_size,
        "quantization": npu_quantization,
        "download_bytes": 0,
        "size_source": "measured",
        "size_note": "Served on the neural processor; no download.",
        "memory_note": memory_note,
        "search_query": "",
        "experience": (
            "Runs the Assistant on the neural processor (XDNA2 NPU) via "
            "flm-real, chosen on reasoning accuracy (VD-108)."
        ),
        # Markers a reader can branch on without parsing the strings above.
        "backend": "flm-real",
        "device": "npu",
        "served_on_npu": True,
        "flm_tag": str(plan.get("flm_tag") or ""),
    }
    return {
        "tier": "accelerated",
        "parameter_range": parameter_size,
        "quantization": npu_quantization,
        "context_tokens": context_tokens,
        "primary": primary,
        # The NPU assistant is the one model for this tier; the GGUF catalog
        # entries are not alternatives for it (they do not run on the NPU).
        "alternatives": [],
        "storage_ok": True,
        "can_install": True,
        "storage_reason": "",
        "rationale": (
            "Served on this machine's neural processor via flm-real; no model "
            "is downloaded, and the neural processor holds the model's memory."
        ),
        # So a reader can tell an NPU-served recommendation from a GGUF one.
        "served_on_npu": True,
        "assistant_selection": selection,
    }


def _gpu_chat_recommendation(hardware: Dict[str, Any]) -> Dict[str, Any]:
    """Fields overriding the recommendation when this machine serves the GPU model.

    Thin by design: the capability gate and the override assembly both live in
    :mod:`vaelor.gpu_model_choice`, the way ``_npu_assistant_recommendation``
    delegates its discovery to :mod:`vaelor.flm_service`. Returns ``{}`` on every
    machine that cannot run the ROCmFP4 27B - a Pi, an x86 box without a gfx1151
    GPU, or a gfx1151 box where the fork is not provisioned - so the recommendation
    is byte-identical there and the 27B is never recommended where it cannot run.
    """
    from .gpu_model_choice import gpu_chat_recommendation

    return gpu_chat_recommendation(hardware)


def copilot_setup_status(hardware: Dict[str, Any]) -> Dict[str, Any]:
    # TWO recommendations from ONE base, because this appliance runs two local
    # models in two roles and they need different picks:
    #
    #   * the **Assistant** is served on the neural processor (an FLM model like
    #     qwen3.5:4b, never a downloadable catalog GGUF), and
    #   * **AI Chat** is served on the GPU (the ROCmFP4 27B, a catalog GGUF via
    #     the ROCmFPX fork).
    #
    # A single shared ``recommendation`` was the bug (HIGH severity): the GPU
    # override — correct for the installable catalog — was merged *on top of* the
    # NPU override, so the "Set up assistant" screen recommended the GPU 27B as
    # the Assistant's brain. The two surfaces that read this payload want
    # opposite things, so the recommendation is split by role and neither
    # override leaks into the other's slot:
    #
    #   * ``recommendation`` = base + the **NPU** override only. This is the
    #     ASSISTANT's model (CopilotSetup + AgentAssistantPanel read it). On a Z2
    #     it is the NPU model; on a Pi the NPU override returns ``{}`` so it is
    #     the base small GGUF — unchanged for the Pi. The GPU override is NEVER
    #     merged here: the 27B is not the Assistant.
    #   * ``catalog_recommendation`` = base + the **GPU** override only. This is
    #     the installable-catalog pick (ModelCatalog reads it). On a gfx1151 box
    #     with the fork it is the 27B; on a non-GPU/Pi box the GPU override
    #     returns ``{}`` so it is the base small installable GGUF — exactly what
    #     the catalog should recommend there. The NPU override is NEVER merged
    #     here: the NPU model is not a downloadable catalog GGUF.
    base = recommend_local_models(hardware)
    npu_override = _npu_assistant_recommendation(hardware)
    recommendation = {**base, **npu_override} if npu_override else base
    gpu_override = _gpu_chat_recommendation(hardware)
    catalog_recommendation = {**base, **gpu_override} if gpu_override else base
    return {
        "hardware": hardware,
        "recommendation": recommendation,
        # The installable-catalog recommendation, split from the Assistant's own
        # so the GPU 27B is offered as an AI-Chat/local install rather than as
        # the Assistant's brain. See the block above for why the two differ.
        "catalog_recommendation": catalog_recommendation,
        # The measured configuration, with the reason for every element, so the
        # Assistant and the UI report what was chosen and why rather than
        # printing a backend name with nothing behind it.
        "inference_plan": recommended_deployment(hardware),
        "providers": [
            {
                "id": "local",
                "name": "Local and private",
                "auth": "none",
                "available": True,
                "description": "Runs on this Vaelor node. No usage fee and prompts stay on the device.",
            },
            {
                "id": "openai",
                "name": "OpenAI API",
                "auth": "api_key",
                "available": False,
                "description": "Frontier intelligence with usage-based API billing.",
                "note": "ChatGPT subscriptions and API billing are separate.",
            },
            {
                "id": "openai-compatible",
                "name": "OpenAI-compatible server",
                "auth": "optional_api_key",
                "available": False,
                "description": "Connect Lemonade, LM Studio, llama.cpp, or another compatible LAN server.",
                "note": "API keys are optional for local servers and encrypted when supplied.",
            },
        ],
        "approval_required": True,
        "credential_storage_ready": False,
    }
