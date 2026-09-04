"""Compose definition for the managed local llama.cpp service.

Split out of ``executor.py`` because accelerator offload is a genuinely
different code path — a different image, device passthrough, supplementary
groups and no thread tuning — rather than a parameter change.
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional

from .accelerator_runtime import runtime_environment
from .inference_tuning import (
    backend_devices,
    model_runtime_flags,
    resolve_kv_cache_types,
    rocm_requirement,
    select_inference_backend,
)
from .model_sizing import ASSISTANT_PARALLEL_SLOTS
from .platforms.accelerators import device_grants
from .runtime_paths import env_value


#: The CPU inference engine, **by digest rather than by tag**.
#:
#: ``:server`` is a moving tag and it moved. Read off the appliance on
#: 2026-08-10, the running container was build ``10335 (74ce15741)`` at the
#: digest below - and nothing in this tree recorded which build any earlier
#: measurement was taken on, so the ledger's throughput and accuracy figures
#: belong to a build nobody can now name. That is the whole of #130: a
#: measurement whose subject can change underneath it is not reproducible, and
#: an appliance that silently upgrades its own inference engine on the next
#: pull is not an appliance.
#:
#: **Earlier figures are not retro-attributed to this digest.** They were taken
#: before it was recorded, and stamping them with a build they may not have run
#: on would be the provenance defect this constant exists to prevent. From here
#: forward a measurement can name its engine.
#:
#: Read with `docker image inspect ghcr.io/ggml-org/llama.cpp:server --format
#: '{{index .RepoDigests 0}}'`. To move it, pull the new tag, record the new
#: digest and the build it reports, and re-run whatever the change is claimed
#: to affect.
CPU_IMAGE_DIGEST = (
    "sha256:2a8440d3aa0be70bf1d1824d2721fc5001616d46be4348b1c1b38fa30af4fe1c"
)
CPU_IMAGE_BUILD = "10335 (74ce15741)"
CPU_IMAGE = "ghcr.io/ggml-org/llama.cpp@{}".format(CPU_IMAGE_DIGEST)

#: Seconds of inactivity before llama-server unloads the model's weights
#: (VD-073). Fifteen minutes: long enough that an ordinary session is never
#: interrupted, short enough that an appliance nobody is using gets its memory
#: back within a coffee break. The cost of being wrong is one 21-second wait.
#:
#: **This does not reduce the memory the container must be allowed.** The model
#: still needs its full footprint when awake, so the cgroup limit and the
#: sizing path are unchanged by this - it lowers average residency, not peak.
SLEEP_IDLE_SECONDS = 15 * 60

#: Where a saved KV prefix lives, on the host and inside the container. Outside
#: the model directory, which is mounted read-only and this is written to.
SLOT_CACHE_DIR = "/var/lib/vaelor/kv-cache"
SLOT_CACHE_MOUNT = "/kv"

#: Share of the container's limit the RAM prompt cache may occupy, and the
#: bounds that share is held between.
#:
#: **llama-server defaults `--cache-ram` to 8192 MiB. The Pi has 7,931 MB of
#: RAM in total.** The engine was being told it may keep a prompt cache larger
#: than the entire machine, so it cached a KV prefix per distinct prompt and
#: never evicted - it cannot reach a ceiling the hardware does not have. The
#: cgroup limit arrives first.
#:
#: Measured on the appliance 2026-08-10, Qwen3-4B-Instruct-2507 Q4_0 at 4,096,
#: 80 prompts, everything else identical:
#:
#:     default 8192 MiB   anon 2831 -> 3430 by prompt 40, +13 MB per prompt,
#:                        no plateau, cgroup 87% of limit and climbing
#:     bounded  256 MiB   anon 2831 -> 3091 by prompt 20 and flat thereafter;
#:                        3122 at prompt 80, +0.5 MB per prompt residual,
#:                        cgroup steady at 71% of limit
#:
#: It costs nothing: 29.2 s per answer bounded against 30.3 s unbounded, so the
#: cache that mattered - the standing brief's prefix, VD-076 - still fits. What
#: was accumulating was every *other* prefix, kept forever against a ceiling
#: that could never be reached.
#:
#: 5% of the limit reproduces the 256 MiB that was measured at the Pi's 5,120,
#: and the floor keeps a small container able to hold a brief at all.
PROMPT_CACHE_LIMIT_SHARE = 0.05
PROMPT_CACHE_MIN_MIB = 128
PROMPT_CACHE_MAX_MIB = 512

#: What the limit is when the caller does not know its own container size.
#: Deliberately the measured floor rather than the engine default: an unknown
#: budget is a reason to be conservative, not a reason to inherit 8 GiB.
PROMPT_CACHE_FALLBACK_MIB = PROMPT_CACHE_MIN_MIB


#: The largest share of the container's limit the prompt cache may take when
#: a measured prompt state is being fitted into it. 5% is the ordinary share;
#: this is the ceiling on raising it to fit one state, so a model whose state
#: is enormous does not eat the container to cache a single prefix.
PROMPT_CACHE_FIT_SHARE = 0.15


def prompt_cache_mib(
    memory_limit_mib=None, requested_mib=None, prompt_state_mib=None
) -> int:
    """How large the RAM prompt cache may be, from the container's own limit.

    Derived rather than fixed, because the number that matters is a share of
    what the machine can spare. A literal would be right on the Pi and wrong on
    the next box, which is the shape of VD-077.

    ``requested_mib`` is a measured value carried on a catalog entry - what a
    particular model was observed to need. **It is a ceiling, not a floor.** A
    catalog number is a measurement of a model; the derived number is a
    property of the machine in front of it, and the smaller of the two is the
    only answer that satisfies both. Taking the catalog value outright would
    put the Pi's 256 MiB on a 1 GB box, which is the defect one layer down.
    """
    try:
        limit = int(memory_limit_mib or 0)
    except (TypeError, ValueError):
        limit = 0
    if limit <= 0:
        derived = PROMPT_CACHE_FALLBACK_MIB
    else:
        share = int(limit * PROMPT_CACHE_LIMIT_SHARE)
        derived = max(PROMPT_CACHE_MIN_MIB, min(PROMPT_CACHE_MAX_MIB, share))
    try:
        requested = int(requested_mib or 0)
    except (TypeError, ValueError):
        requested = 0
    bounded = min(derived, requested) if requested > 0 else derived
    # **A ceiling below one prompt state is a cache that cannot hold anything.**
    # Measured on the appliance 2026-08-11: llama-server logged "prompt state
    # size 288.024 MiB exceeds cache size limit 217.000 MiB, skipping", so the
    # 5% share had switched prefix caching off entirely and every request paid
    # a full prefill - the 68% of answer time #128 names, and a large part of
    # why the Assistant was being cancelled at all (#156, #157).
    #
    # The share above is a property of the machine and stays the ceiling for
    # the ordinary case. This is the floor: if one measured state fits inside
    # `PROMPT_CACHE_FIT_SHARE` of the container, the cache is raised to hold
    # it, because a cache holding one prefix is the entire feature and a cache
    # holding none is reserved memory doing nothing. If it does not fit, the
    # bound is left alone rather than eating the container for one prefix.
    #
    # **No catalog entry supplies `prompt_state_mib` today, so this branch is
    # unreachable in production and the prompt cache is still off (#157).**
    # Saying so here because the alternative is code that reads as live: the
    # one measured value, 288.024 MiB, was withdrawn from `RUNTIME_PI_ASSISTANT`
    # during review when it turned out to be 25 KiB below the state it had to
    # hold. The mechanism is kept rather than deleted because the measurement
    # that would make it correct is a known, scoped piece of work and this is
    # where its answer belongs - but a reader must not mistake a parameter
    # nobody passes for a feature that runs.
    try:
        state = int(prompt_state_mib or 0)
    except (TypeError, ValueError):
        state = 0
    if state > 0 and limit > 0 and state <= int(limit * PROMPT_CACHE_FIT_SHARE):
        return max(bounded, state)
    return bounded

#: Accelerated images, with their measured pull sizes. The ROCm image is
#: 30.6 GB against 1.22 GB for the Vulkan one, which is a serious cost on an
#: appliance — and ROCm is now the measured default, so that cost is one the
#: default path pays. It is recorded here rather than buried because a first
#: deploy on a slow link is a genuinely different experience at 30.6 GB, and
#: because `deployment_agent` quotes these strings to the operator before the
#: pull starts. ROCm also has to be installed on the host, which it already is:
#: `amd-smi` is the only source of NPU telemetry regardless of backend.
#:
#: See :data:`vaelor.inference_tuning.DEFAULT_GPU_BACKEND` for why the measured
#: prefill advantage was judged to outweigh this.
#:
#: **Pinned by digest, #247m.** These were floating ``:server-rocm`` /
#: ``:server-vulkan`` tags, which is the #130 defect for the GPU tier: a tag
#: moves, and an appliance that pulls it silently upgrades its own inference
#: engine, so a measurement's subject can change underneath it. The stock GPU
#: AI-Chat route (#247m) makes that concrete - gpt-oss-20b ships MXFP4 tensors
#: that only a recent llama.cpp reads, so serving it against a floating tag is a
#: bet that the tag never regresses MXFP4. Both are pinned to build **b10548**
#: (`org.opencontainers.image.revision a298422da78eb75e440a7de0ca408af64d323d93`),
#: comfortably past the build that added gpt-oss/MXFP4 support, resolved from the
#: registry against `ghcr.io/ggml-org/llama.cpp` on 2026-08-21.
#:
#: **Provenance caveat, kept honest (the reason the old note declined to pin).**
#: :data:`CPU_IMAGE`'s digest was read off the Pi that runs it; these two have
#: NOT yet been run on the Z2 - the pin was resolved from the registry, not from
#: the host. So this fixes reproducibility (the engine can no longer change under
#: a measurement) but does not by itself establish that these builds serve MXFP4
#: on the Z2's GPU: that is a live single-GPU verification the owner runs
#: separately (#130 remains open for the Z2 until then). Pinning a specific,
#: known-recent digest is strictly safer than a moving tag for that verification,
#: because what is tested is then what ships. To move it, pull the new tag on the
#: Z2, record the digest and the build it reports, and re-run the GPU tier.
VULKAN_IMAGE_DIGEST = (
    "sha256:70815abd05cfd67c9a686f295573075aac96a3099ad979443b909666c291ad2e"
)
ROCM_IMAGE_DIGEST = (
    "sha256:1c5c31b27450c8419ca9a430993701f3f5933a8e202a52228862989dcc28b7d2"
)
ACCELERATED_IMAGE_BUILD = "b10548 (a298422da78eb75e440a7de0ca408af64d323d93)"
ACCELERATED_IMAGES = {
    "vulkan": (
        "ghcr.io/ggml-org/llama.cpp@{}".format(VULKAN_IMAGE_DIGEST),
        "about 1.2 GB",
    ),
    "rocm": (
        "ghcr.io/ggml-org/llama.cpp@{}".format(ROCM_IMAGE_DIGEST),
        "about 30.6 GB",
    ),
}

BACKENDS = ("cpu", "vulkan", "rocm")

#: Layers offered to the accelerator when one is asked for. Recorded on the
#: plan whether or not the offload is granted, because "how many layers were
#: requested" is what makes a launch that ended up on the CPU legible as a
#: *declined* accelerator request rather than a CPU deployment by choice.
ACCELERATED_GPU_LAYERS = 999

#: What an unset ``VAELOR_LOCAL_MODEL_BACKEND`` means: decide from measurement.
#: This used to be ``cpu``, on the stated grounds that nothing had been proven
#: end to end on the target hardware. That is no longer true — four models were
#: measured on it — and while it was true the recommended configuration was
#: unreachable without an operator knowing an environment variable existed.
#: ``auto`` is not "guess": it runs :func:`select_inference_backend` against the
#: model's actual size and the accelerator's actual carve-out, and everything
#: downstream still fails closed with a reason.
MEASURED_BACKEND = "auto"


def requested_backend() -> str:
    """Which backend the operator asked for, or ``auto`` to measure one.

    An explicit ``cpu``, ``vulkan`` or ``rocm`` is honoured exactly: an operator
    who pins a backend has overruled the measurement on purpose, and that is
    reported as an override rather than dressed up as a decision.
    """
    value = env_value(
        "VAELOR_LOCAL_MODEL_BACKEND", "PM_LOCAL_MODEL_BACKEND", MEASURED_BACKEND
    ).strip().lower()
    return value if value in BACKENDS + (MEASURED_BACKEND,) else MEASURED_BACKEND


def accelerator_plan(
    hardware: Dict[str, Any],
    backend: Optional[str] = None,
    grants: Optional[Dict[str, Any]] = None,
    *,
    model_bytes: int = 0,
    kv_cache_bytes: int = 0,
) -> Dict[str, Any]:
    """Decide whether the model service can be given the accelerator, and which.

    Fails closed with a reason at every step. Detection is not permission, and
    permission is not a working runtime.

    The choice between Vulkan and ROCm is a measurement, not a preference.
    Vulkan won on essentially everything tried, needs only ``/dev/dri/renderD128``
    where ROCm also needs ``/dev/kfd``, and drives the CPU about four times less
    — but it allocates from the VRAM carve-out while ROCm draws on the larger
    shared pool, so a model that does not fit the carve-out with headroom goes
    to ROCm. ``decision`` carries the numbers that settled it.
    """
    backend = backend or requested_backend()
    decision: Dict[str, Any] = {}
    if backend == MEASURED_BACKEND:
        decision = select_inference_backend(
            model_bytes=model_bytes,
            kv_cache_bytes=kv_cache_bytes,
            accelerators=hardware.get("accelerators") or [],
        )
        backend = decision["backend"]
    plan: Dict[str, Any] = {
        "backend": "cpu",
        "requested_backend": backend,
        "image": CPU_IMAGE,
        "devices": [],
        "group_add": [],
        "gpu_layers": 0,
        # What was asked for, kept even when every gate below refuses it. The
        # verification that follows the deploy reads this rather than the
        # backend it settled on, so a declined offload cannot be reported as
        # "no accelerator was expected".
        "requested_gpu_layers": (
            ACCELERATED_GPU_LAYERS if backend in ACCELERATED_IMAGES else 0
        ),
        "reason": "",
        "decision": decision,
        # ROCm is a telemetry dependency whatever runs inference: `amd-smi` is
        # the only source of NPU utilisation, NPU power and adapter firmware.
        "rocm": rocm_requirement(backend),
    }
    if backend == "cpu":
        plan["reason"] = decision.get("reason") or (
            "GPU offload is not enabled for the managed local model."
        )
        return plan
    if not hardware.get("accelerators"):
        plan["reason"] = "No accelerator was discovered on this host."
        return plan
    resolved = grants if grants is not None else device_grants(["gpu"])
    if not resolved.get("devices"):
        plan["reason"] = "The GPU character devices are not present on this host."
        return plan
    if resolved.get("unresolved_groups"):
        # `--group-add render` fails: a container image has no host group
        # database. Numeric GIDs are required, and they differ per host, so an
        # unresolvable group means the container could not open the device.
        plan["reason"] = (
            "The {} group could not be resolved to a numeric id on this host, "
            "so the container could not be granted the GPU devices.".format(
                ", ".join(resolved["unresolved_groups"])
            )
        )
        return plan
    access = hardware.get("accelerator_access") or {}
    if access and not access.get("ready"):
        plan["reason"] = access.get("reason") or (
            "The service account cannot open the GPU character devices."
        )
        return plan
    image, _size = ACCELERATED_IMAGES[backend]
    # Only the devices this backend actually needs. Vulkan runs on the render
    # node alone; handing it `/dev/kfd` as well would widen the privilege
    # surface of the container for a device it never opens.
    wanted = backend_devices(backend)
    devices = [path for path in resolved["devices"] if path in wanted] or list(
        resolved["devices"]
    )
    plan.update({
        "backend": backend,
        "image": image,
        "devices": devices,
        "group_add": [
            int(gid) for gid in resolved["group_ids"].values() if gid is not None
        ],
        "gpu_layers": ACCELERATED_GPU_LAYERS,
        "reason": decision.get("reason", ""),
        "shared_group_note": resolved.get("isolation_note", ""),
    })
    return plan


def _indent_list(values: List[Any], key: str, indent: str = "    ") -> str:
    if not values:
        return ""
    lines = ["{}{}:".format(indent, key)]
    lines.extend('{}  - "{}"'.format(indent, value) for value in values)
    return "\n".join(lines) + "\n"


def build_model_compose(
    *,
    model_dir: str,
    model_name: str,
    port: int,
    runtime: Dict[str, Any],
    plan: Optional[Dict[str, Any]] = None,
) -> str:
    """Render the managed-model compose file."""
    plan = plan or {
        "backend": "cpu", "image": CPU_IMAGE, "devices": [], "group_add": [],
        "gpu_layers": 0,
    }
    accelerated = plan.get("backend") != "cpu"
    # Flags are resolved per model, not globally, because the measurements are
    # per model: the same flag is a large win on one and a large loss on another.
    flags = model_runtime_flags(model_name)
    environment = [
        ('LLAMA_ARG_MODEL', '/models/{}'.format(model_name)),
        ('LLAMA_ARG_CTX_SIZE', str(runtime["context"])),
        # State the slot count; never inherit it — the same rule the library
        # path follows two blocks down, for the same reason. Left unset, this
        # image started **four** slots and gave each of them the full 4,096
        # tokens that were requested: 16,384 tokens of KV, ~1.88 GB on a
        # 1.7 B model, inside a 2,816 MiB container sized for one. Nothing in
        # the HTTP surface said so; the appliance simply went dark four minutes
        # later. See `model_sizing.SLOT_MULTIPLICATION_MEASUREMENT`.
        #
        # A runtime that names no slot count gets one, not llama.cpp's default:
        # this is the last layer before launch, and the smaller claim is the
        # safe direction to fail in.
        (
            'LLAMA_ARG_N_PARALLEL',
            str(max(1, int(runtime.get("parallel_slots") or ASSISTANT_PARALLEL_SLOTS))),
        ),
    ]
    if not accelerated:
        environment.append(("LLAMA_ARG_THREADS", str(runtime["threads"])))
    else:
        environment.append(
            ("LLAMA_ARG_N_GPU_LAYERS", str(plan.get("gpu_layers", 999)))
        )
        # State the library path; never inherit it. `libggml-hip.so` has
        # `RUNPATH: $ORIGIN` and nothing else, so with no accelerator library
        # directory on `LD_LIBRARY_PATH` it cannot resolve, llama.cpp loads the
        # CPU backend instead, and it serves happily at 198 tok/s prefill
        # against 1835 with zero GPU memory and no error of any kind. Asking
        # for `-ngl 99` does not make that fail; it just does not happen.
        environment.extend(sorted(runtime_environment(plan.get("backend", "")).items()))
    environment.extend([
        ("LLAMA_ARG_HOST", "0.0.0.0"),
        ("LLAMA_ARG_PORT", "8080"),
        ("LLAMA_ARG_ENDPOINT_METRICS", "1"),
        ("LLAMA_ARG_JINJA", "1"),
    ])
    # KV cache dtype, resolved here as well as in the planner because this is
    # the last layer before llama.cpp is actually launched. Neither cache may
    # be quantized without flash attention: a quantized V is refused at load,
    # and a quantized K *is not* — it starts, serves, and decodes at 1.01
    # tokens per second against 64.22. This file used to emit whatever it was
    # handed, so a caller that built its own runtime dict could write the trap
    # straight into the shipped configuration.
    cache = resolve_kv_cache_types(
        str(runtime.get("cache_type_k", "f16")),
        str(runtime.get("cache_type_v", "f16")),
        flash_attention_available=bool(runtime.get("flash_attention")),
    )
    cache_k = cache["cache_type_k"]
    cache_v = cache["cache_type_v"]
    if cache_k != "f16":
        environment.append(("LLAMA_ARG_CACHE_TYPE_K", cache_k))
    if cache_v != "f16":
        environment.append(("LLAMA_ARG_CACHE_TYPE_V", cache_v))
    # `cache["flash_attention"]` is what the *cache configuration* requires,
    # and it implies this condition rather than replacing it: a caller with an
    # available backend and two f16 caches is passing an availability hint for
    # a flag that is `auto` anyway, which is neither the trap nor a claim about
    # what shipped. Narrowing this to the resolver's answer would change the
    # argument vector for `gemma-4-26b-a4b`, whose f16 override was measured
    # about KV quantization and says nothing about flash attention.
    if runtime.get("flash_attention"):
        # `auto`, never `1`. Forcing flash attention on was measured at 848
        # prefill against 1544 on the 20B under Vulkan, and as a large win on a
        # 3B under ROCm; there is no global answer, so llama.cpp is left to pick
        # per model. Per-model overrides live in `inference_tuning`.
        environment.append((
            "LLAMA_ARG_FLASH_ATTN",
            str(runtime.get("flash_attention_mode") or flags["flash_attention_mode"]),
        ))
    # -ub 1024: +18.9% prefill and -15.8% TTFT on the 20B under ROCm, and
    # best-in-class under Vulkan stacked with a q8 KV cache.
    if flags.get("ubatch"):
        environment.append(("LLAMA_ARG_UBATCH", str(int(flags["ubatch"]))))
    rendered_environment = "\n".join(
        '      {}: "{}"'.format(key, value) for key, value in environment
    )
    devices = _indent_list(plan.get("devices") or [], "devices")
    group_add = _indent_list(plan.get("group_add") or [], "group_add")
    limits = "          memory: {}M\n".format(runtime["memory_limit_mib"])
    if not accelerated:
        limits += '          cpus: "4.0"\n'
    # VD-073. llama-server unloads the model's weights after this many seconds
    # of inactivity and reloads them on the next request. Measured on the Pi,
    # the 4B at 8,192: 5,708 MB resident falls to 713 MB - 88% returned - and
    # the first question afterwards blocks for 21 s and is answered. That is
    # the whole of the on-demand behaviour VD-069 built four releases of
    # machinery for.
    #
    # **On the command line, not in `environment`.** `LLAMA_ARG_SLEEP_IDLE_SECONDS`
    # is silently ignored - the README says every flag has an environment
    # equivalent and this one does not work. Measured 2026-08-09: identical
    # containers, env var never slept, command-line flag released 1,168 MB.
    # Everything else here is configured by environment, so this is the only
    # `command:` in the file and it is deliberate.
    #
    # `is None`, not `or`. A caller passing 0 means *never sleep*, and `or`
    # reads that as absent and hands back the default - an explicit opt-out
    # turned into an opt-in, which is the one direction this must not fail in.
    requested = runtime.get("sleep_idle_seconds")
    idle = SLEEP_IDLE_SECONDS if requested is None else int(requested)
    arguments = []
    if idle > 0:
        arguments.extend(["--sleep-idle-seconds", str(idle)])
    # `--slot-save-path` is what lets the standing brief survive the sleep
    # above. Left to themselves the two features fight: sleeping returns
    # ~3.6 GiB and discards the RAM prompt cache with it, so the first question
    # after a quiet period pays the model reload *and* a full 786-token
    # prefill. Measured on the appliance 2026-08-10 - 95.5s without, 20.4s
    # with (VD-076).
    #
    # This makes the capability present on every box. The calls that use it
    # live in `assistant_slot_cache`: a save where the Assistant's reply is
    # appended, a guarded restore before the next question in that
    # conversation (task #152).
    if runtime.get("slot_save"):
        arguments.extend(["--slot-save-path", SLOT_CACHE_MOUNT])
    # The prompt cache is bounded to something this machine actually has.
    # Always, on every deploy - see `prompt_cache_mib` for the measurement.
    arguments.extend([
        "--cache-ram", str(prompt_cache_mib(
            runtime.get("memory_limit_mib"), runtime.get("cache_ram_mib"),
            runtime.get("prompt_state_mib"))),
    ])
    command = (
        "    command: [{}]\n".format(
            ", ".join('"{}"'.format(item) for item in arguments))
        if arguments else ""
    )
    slot_volume = (
        '      - "{}:{}"\n'.format(SLOT_CACHE_DIR, SLOT_CACHE_MOUNT)
        if runtime.get("slot_save") else ""
    )
    return (
        "services:\n"
        "  llama-server:\n"
        "    image: {image}\n"
        "    restart: unless-stopped\n"
        "{command}"
        "    ports:\n"
        '      - "127.0.0.1:{port}:8080"\n'
        "    volumes:\n"
        '      - "{model_dir}:/models:ro"\n'
        "{slot_volume}"
        "{devices}{group_add}"
        "    environment:\n"
        "{environment}\n"
        "      LLAMA_ARG_CHAT_TEMPLATE_KWARGS: '{{\"enable_thinking\":false}}'\n"
        '      LLAMA_ARG_REASONING_BUDGET: "0"\n'
        "    deploy:\n"
        "      resources:\n"
        "        limits:\n"
        "{limits}"
    ).format(
        image=plan.get("image", CPU_IMAGE),
        command=command,
        port=port,
        model_dir=model_dir,
        slot_volume=slot_volume,
        devices=devices,
        group_add=group_add,
        environment=rendered_environment,
        limits=limits,
    )
