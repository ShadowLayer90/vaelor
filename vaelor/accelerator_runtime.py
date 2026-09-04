"""Make the accelerator runtime explicit, and prove the accelerator is used.

This module exists because of one measurement, and it is the failure Vaelor
was built to prevent: **an appliance reporting a healthy GPU-backed service
that is not GPU-backed.**

With no ROCm directory on ``LD_LIBRARY_PATH``, ``libggml-hip.so`` cannot
resolve ``libamdhip64.so.7``, ``libhipblas.so.3`` or ``librocblas.so.5``. Its
``RUNPATH`` is ``$ORIGIN`` and nothing else, and ``/opt/rocm`` is not in
``/etc/ld.so.conf.d/``, so there is no fallback for the loader to find. What
llama.cpp does next is the problem: it loads the CPU backend and carries on.
Measured on the Z2, twice, with ``-ngl 99`` requested throughout:

===========================  ==============  ==============  =============
arm                          GPU memory      prefill tok/s   decode tok/s
===========================  ==============  ==============  =============
no ROCm dir on the path      **0.0 MB**      **198.2**       17.6
ROCm on the path             5903.4 MB       1834.9          37.1
===========================  ==============  ==============  =============

**No error. No warning. ``/health`` answered 200 throughout.** A server in
that state looks completely fine from the outside; it just answers nine times
slower, on the CPU, on a machine bought for its GPU.

Three things follow, and this module is all three:

1. **State the environment; never inherit it.** :func:`runtime_environment`
   returns the library path a backend needs, so the launch does not depend on
   whatever the supervising process happened to have set.
2. **Verify the accelerator is in use, not that the process is running.**
   :func:`verify_accelerator_in_use` reads what the driver says is allocated.
   A GPU-backed server holding zero GPU memory is CPU-serving, and that is the
   cheapest signal available.
3. **Say so.** *"Running on CPU — the GPU library did not load"* is a legible
   state. A silent 9x degradation is not.

Note what is *not* claimed here. "Unknown" is a real answer: an accelerator
that does not report its memory usage cannot be checked this way, and that is
reported as unverified rather than as either success or failure — the same
distinction :mod:`vaelor.model_reachability` draws between ``unknown`` and
``false``.

**"State the environment" only reaches a launch that asked for one.**
:func:`runtime_environment` is consulted by
:func:`~vaelor.model_service_compose.build_model_compose` on the accelerated
arm alone, because the CPU image has no accelerator library to find. So when
the plan decides ``rocm`` and a later gate refuses it — no character devices,
an unresolvable ``render`` group, a service account that cannot open the
device — the container is launched from the CPU image and no ``LD_LIBRARY_PATH``
is ever set for it. That is not the library path failing to *cover* a path; it
is a launch that never took the arm that states one, and the two need telling
apart because only one of them is fixed by adding a directory. The state for
it is ``not-offloaded``: requested, declined, announced, with the refusal's own
reason quoted rather than left in another field.

**And the same discipline applies to the failure.** Announcing a CPU fallback
is the loudest thing this module does, and announcing one that is not there
inverts it into the defect it exists to prevent: it asserts a 9x degradation
on a healthy machine. A redeploy makes that easy — the previous model is still
resident when a baseline is read too early, so the new model appears to hold
nothing. So the fallback is claimed only when the accelerator holds nothing
**in absolute terms**, and a delta that cannot be attributed is reported as
``unattributed``. Every reading also carries the ``basis`` it was taken on, so
a differential and an absolute reading of the same fact can be compared rather
than silently disagreeing.

**The same shape, applied to the context window.** :func:`verify_context_built`
is the second reading taken after ``/health`` answers, and it is here rather
than in a module of its own because it is the same failure: ask llama.cpp for a
slot context above the model's training context and it *caps* rather than
fails. ``W srv load_model: the slot context (147456) exceeds the training
context of the model (131072) - capping``, the server starts, ``/health``
returns 200, and the HTTP surface never mentions it. The overshoot is not free
either: 147,456 requested built the same 131,072-token slot and consumed
16,147 MB against 15,619 MB — 528 MB spent on nothing.

That is the third silent degradation measured on this machine, after the 1
tok/s KV-cache trap and the CPU fallback above, and they are one class. After
starting any runtime, verify **what you actually got**, not that the process
answers.
"""

from __future__ import annotations

import json
import urllib.error
import urllib.request
from typing import Any, Callable, Dict, Mapping, Optional, Sequence


MIB = 1024 ** 2

#: Directories each backend's shared libraries live in, searched in order.
#:
#: These are *prepended* to the loader's search order, so naming a directory
#: that does not exist costs nothing and the image's own ``RUNPATH`` and
#: ``ld.so.cache`` still apply behind them. The conventional system
#: directories are included so that setting this variable can never remove a
#: path the image was relying on.
BACKEND_LIBRARY_PATHS: Dict[str, tuple] = {
    "rocm": (
        "/opt/rocm/lib",
        "/opt/rocm/lib64",
        "/usr/local/lib",
        "/usr/lib/x86_64-linux-gnu",
        "/usr/lib",
    ),
    "vulkan": (
        "/usr/local/lib",
        "/usr/lib/x86_64-linux-gnu",
        "/usr/lib",
    ),
    "cpu": (),
}

#: The compute library each accelerated backend must have loaded. Recorded so
#: the diagnosis can name the file that failed to resolve rather than saying
#: "the GPU did not work".
BACKEND_COMPUTE_LIBRARY = {
    "rocm": "libggml-hip.so",
    "vulkan": "libggml-vulkan.so",
}

#: The measurement this module is built on. Kept beside the code so a reviewer
#: can check the claim without leaving the file.
#:
#: **It is a reference, not a reading.** It was taken on one model, on one
#: machine, on a day that is not today, and it travelled inside every
#: deployment result under the key ``measurement`` — beside a deployment of a
#: *different* model, whose own throughput was never measured. A reader saw
#: ``decode_tps`` and ``gpu_memory_bytes`` next to their deploy and had no way
#: to tell that neither number came from it. So the framing is part of the
#: constant rather than something a caller is trusted to add, and the result
#: key that carries it is :data:`REFERENCE_MEASUREMENT_KEY`.
SILENT_CPU_FALLBACK_MEASUREMENT = {
    "about": (
        "Reference measurement from a previous run on the model named below. "
        "It is the evidence for how this failure behaves, not a reading taken "
        "from the deployment it is shown beside."
    ),
    "backend": "rocm",
    "model": "Llama-3.2-3B-Instruct-UD-Q4_K_XL",
    "runs": 2,
    "gpu_layers_requested": 99,
    "without_library_path": {
        "gpu_memory_bytes": 0,
        "prefill_tps": 198.17,
        "decode_tps": 17.63,
        "hip_backend_loaded": False,
    },
    "with_library_path": {
        "gpu_memory_bytes": 5903 * MIB,
        "prefill_tps": 1834.88,
        "decode_tps": 37.07,
        "hip_backend_loaded": True,
    },
    "prefill_ratio": 9.3,
    "decode_ratio": 2.1,
    "symptom": "no error, no warning, /health answered 200",
    "cause": (
        "libggml-hip.so has RUNPATH $ORIGIN only and /opt/rocm is not in "
        "/etc/ld.so.conf.d/, so the HIP backend cannot be resolved and "
        "llama.cpp falls back to the CPU backend"
    ),
}

#: Least GPU memory a genuinely accelerated server is expected to be holding.
#:
#: The failing arm held **exactly zero** and the working arm held 5,903 MB, so
#: this threshold is nowhere near either. It is set above an idle adapter's own
#: allocation — 160 MB VRAM plus 26 MB GTT was measured on an idle Z2 — so a
#: display buffer alone cannot be mistaken for a loaded model.
MINIMUM_ACCELERATED_BYTES = 512 * MIB

#: The key a reference measurement travels under. Named rather than spelled
#: ``"measurement"`` because that word, beside a deployment, reads as *this
#: deployment's* measurement — which it never was.
REFERENCE_MEASUREMENT_KEY = "reference_measurement"


def library_path(backend: str) -> str:
    """The ``LD_LIBRARY_PATH`` value this backend needs, or ``""`` for none."""
    return ":".join(BACKEND_LIBRARY_PATHS.get(str(backend or "").lower(), ()))


def runtime_environment(backend: str) -> Dict[str, str]:
    """Environment variables that must be set where llama.cpp is launched.

    Empty for the CPU backend, which needs nothing beyond what any process
    already has. For an accelerated backend this is the difference between a
    GPU-backed server and a CPU-backed one that claims to be GPU-backed.
    """
    value = library_path(backend)
    return {"LD_LIBRARY_PATH": value} if value else {}


def reported_gpu_memory_bytes(
    accelerators: Optional[Sequence[Mapping[str, Any]]],
) -> Optional[int]:
    """Total memory the primary accelerator says is allocated, or ``None``.

    ``None`` means the adapter did not report either figure. That is not zero:
    an adapter that cannot be read tells you nothing about whether a model is
    resident on it, and reporting it as zero would turn "not measurable" into
    "definitely on the CPU".
    """
    for accelerator in accelerators or []:
        vram = accelerator.get("vram_used_bytes")
        gtt = accelerator.get("gtt_used_bytes")
        if vram is None and gtt is None:
            continue
        return max(0, int(vram or 0)) + max(0, int(gtt or 0))
    return None


#: How a reading was taken, and therefore what it is entitled to claim.
#:
#: ``differential`` subtracts a baseline read while no managed model was
#: resident, so what is left is this deployment's own allocation.
#: ``absolute`` has no baseline at all: it can say the accelerator is holding a
#: model's worth of memory, but not *whose*. Both are honest; they answer
#: different questions, and a reading that does not say which one it answered
#: invites two callers to contradict each other over one fact.
BASIS_DIFFERENTIAL = "differential"
BASIS_ABSOLUTE = "absolute"


def _accelerator_was_requested(backend: str, gpu_layers: Any) -> bool:
    """Whether an accelerator was *asked for*, whatever was then launched.

    This is deliberately a question about the request. ``expected_accelerator``
    used to be read off the backend the container actually got, so a deployment
    that decided ``rocm``, asked for 99 GPU layers and then launched on the CPU
    image reported ``expected_accelerator: false`` and told the owner *"this
    model runs on the CPU by configuration, so no accelerator was expected"* —
    in the same payload that carried ``backend_decision.backend: "rocm"``. The
    owner is told not to look, which is worse than the silence this module was
    built to end.
    """
    if str(backend or "").lower() in BACKEND_COMPUTE_LIBRARY:
        return True
    try:
        return int(gpu_layers or 0) > 0
    except (TypeError, ValueError):
        return False


def verify_accelerator_in_use(
    backend: str,
    accelerators: Optional[Sequence[Mapping[str, Any]]] = None,
    *,
    baseline_bytes: Optional[int] = None,
    minimum_bytes: int = MINIMUM_ACCELERATED_BYTES,
    requested_backend: Optional[str] = None,
    gpu_layers_requested: Any = 0,
    declined_reason: str = "",
) -> Dict[str, Any]:
    """Whether the accelerator this deployment asked for is actually serving.

    The process running is not the question — the failing arm ran perfectly.
    The question is whether the accelerator is holding anything, and
    ``baseline_bytes`` (read before the model started) makes that a delta
    rather than a guess about what else is on the machine.

    Returns ``in_use`` as ``True``, ``False`` or ``None``. ``None`` is
    load-bearing: it means the check could not be made, and a caller must not
    render it as either outcome.

    **A delta of zero is not by itself a CPU fallback.** ``cpu-fallback`` is
    the loudest thing this module says, so it is claimed only when the
    accelerator is holding less than a model's worth **in absolute terms** —
    which is the measured failing arm, at 0.0 MB. A delta that is small while
    the adapter is holding gigabytes means the baseline already covered them:
    that is consistent with a model this deployment did not put there *and*
    with a baseline that was read too late, and asserting either would be
    asserting a degradation that has not been established. It is reported as
    ``unattributed`` — an unanswered question, not a verdict.

    ``requested_backend`` and ``gpu_layers_requested`` describe what was asked
    for; ``backend`` is what was launched. They are usually the same, and where
    they are not the difference is the whole answer: an accelerator that was
    requested and then not given is a degradation of exactly the size this
    module exists to report, and it must not be described as a CPU deployment
    "by configuration".
    """
    name = str(backend or "").lower()
    asked_for = str(
        requested_backend if requested_backend is not None else backend or ""
    ).lower()
    accelerated = name in BACKEND_COMPUTE_LIBRARY
    expected = _accelerator_was_requested(asked_for, gpu_layers_requested)
    library = BACKEND_COMPUTE_LIBRARY.get(name) or BACKEND_COMPUTE_LIBRARY.get(
        asked_for, ""
    )
    observed = reported_gpu_memory_bytes(accelerators)
    basis = BASIS_DIFFERENTIAL if baseline_bytes is not None else BASIS_ABSOLUTE
    result: Dict[str, Any] = {
        "backend": name,
        "requested_backend": asked_for,
        "gpu_layers_requested": int(gpu_layers_requested or 0),
        # Answered from the request, never from what the launch settled on.
        "expected_accelerator": expected,
        "compute_library": library,
        "gpu_memory_bytes": observed,
        "baseline_bytes": baseline_bytes,
        "basis": basis,
        "in_use": None,
        "state": "unknown",
        "detail": "",
    }
    if not accelerated:
        if expected:
            # Requested and not given. The container never ran an accelerated
            # image, so `runtime_environment` was never applied to it either —
            # the stated library path cannot reach a launch that did not take
            # the accelerated arm. Say which one happened.
            result.update({
                "in_use": False,
                "state": "not-offloaded",
                REFERENCE_MEASUREMENT_KEY: SILENT_CPU_FALLBACK_MEASUREMENT,
                "detail": (
                    "An accelerator was requested — the {} backend with {} GPU "
                    "layers — but this model was launched on the {} backend, so "
                    "nothing was offloaded and no accelerator library was set "
                    "for it. {}A comparable shortfall was measured at {:.0f} "
                    "against {:.0f} tokens per second of prefill on another "
                    "model, so the cost of serving this way is large."
                ).format(
                    asked_for or "requested",
                    int(gpu_layers_requested or 0),
                    name or "cpu",
                    "The reason given was: {} ".format(
                        str(declined_reason).strip()
                    ) if str(declined_reason).strip() else "",
                    SILENT_CPU_FALLBACK_MEASUREMENT["without_library_path"][
                        "prefill_tps"
                    ],
                    SILENT_CPU_FALLBACK_MEASUREMENT["with_library_path"][
                        "prefill_tps"
                    ],
                ),
            })
            return result
        result.update({
            "in_use": False,
            "state": "cpu",
            "detail": (
                "This model runs on the CPU by configuration, so no accelerator "
                "was expected."
            ),
        })
        return result
    if observed is None:
        result["detail"] = (
            "The accelerator does not report how much of its memory is "
            "allocated, so whether {} actually loaded could not be verified. "
            "This is not a failure; it is an unanswered question."
        ).format(library or "the GPU library")
        return result
    threshold = max(0, int(minimum_bytes))
    held = observed - max(0, int(baseline_bytes or 0)) if baseline_bytes is not None else observed
    if held >= threshold:
        result.update({
            "in_use": True,
            "state": "accelerated",
            "held_bytes": held,
            "detail": (
                "The {} backend is holding {:.0f} MiB of accelerator memory, so "
                "{} loaded and the model is on the GPU."
            ).format(name, held / MIB, library) if basis == BASIS_DIFFERENTIAL else (
                "The accelerator is holding {:.0f} MiB, which is a loaded "
                "model's worth and not an idle adapter's. This reading is "
                "absolute — nothing was measured before this server started — "
                "so it establishes that {} is in use on this machine, not that "
                "this server is what is using it."
            ).format(held / MIB, library or "the GPU library"),
        })
        return result
    if observed >= threshold:
        result.update({
            "state": "unattributed",
            "held_bytes": held,
            "detail": (
                "The accelerator is holding {:.0f} MiB, but it was already "
                "holding {:.0f} MiB before this server started, so none of it "
                "can be attributed to this one. That is consistent with a "
                "model another engine loaded and with a baseline taken while "
                "the previous model was still resident, and this reading "
                "cannot tell those apart. Whether {} loaded here is unanswered "
                "— it is not a CPU fallback, which is the reading where the "
                "accelerator holds nothing at all."
            ).format(
                observed / MIB, max(0, int(baseline_bytes or 0)) / MIB,
                library or "the GPU library",
            ),
        })
        return result
    result.update({
        "in_use": False,
        "state": "cpu-fallback",
        "held_bytes": held,
        REFERENCE_MEASUREMENT_KEY: SILENT_CPU_FALLBACK_MEASUREMENT,
        "detail": (
            "Running on CPU — the GPU library did not load. The server started "
            "and answers, but the accelerator is holding {:.0f} MiB, and a "
            "{}-backed model would hold far more. {} could not be resolved, "
            "most likely because no accelerator library directory was on "
            "LD_LIBRARY_PATH. The same failure was measured at {:.0f} against "
            "{:.0f} tokens per second of prefill, with no error and no warning."
        ).format(
            held / MIB, name, library or "The GPU backend",
            SILENT_CPU_FALLBACK_MEASUREMENT["without_library_path"]["prefill_tps"],
            SILENT_CPU_FALLBACK_MEASUREMENT["with_library_path"]["prefill_tps"],
        ),
    })
    return result


# ---------------------------------------------------------------------------
# The context window that was actually built
# ---------------------------------------------------------------------------

#: llama.cpp's own readback of the server it built. Nothing in Vaelor read it
#: before: ``/props``, ``n_ctx_train`` and ``default_generation_settings``
#: returned zero hits across the whole tree, so the requested context window was
#: the only figure anything knew, and it is the one figure that can be wrong.
CONTEXT_PROPS_PATH = "/props"

#: The measurement this check is built on, kept beside the code the same way
#: the CPU-fallback measurement is.
#:
#: Overshooting buys nothing and costs memory: the slot is capped to the
#: training context either way, and the difference was paid in resident bytes
#: that no request can use. Never request above a model's trained context.
CONTEXT_CAP_MEASUREMENT = {
    "about": (
        "Reference measurement from a previous run. It is the evidence for how "
        "capping behaves, not a reading taken from the deployment it is shown "
        "beside."
    ),
    "requested_tokens": 147456,
    "built_tokens": 131072,
    "trained_tokens": 131072,
    "resident_mib_requested": 16147,
    "resident_mib_at_training_context": 15619,
    "wasted_mib": 528,
    "symptom": (
        "the server started, /health answered 200, and the only notice was a "
        "line in the process log: W srv load_model: the slot context (147456) "
        "exceeds the training context of the model (131072) - capping"
    ),
}

#: The same failure with its sign reversed, measured on the Pi appliance.
#:
#: A window can silently *multiply* as well as silently cap, and this check was
#: written to catch one direction only: it compared the per-slot figure with the
#: request, and a per-slot figure equal to the request read as success no matter
#: how many slots were holding one. Vaelor asked for 4,096 tokens, stated no
#: slot count, and the server started four slots of 4,096 — four times the KV
#: that was requested and budgeted for. ``/health`` answered 200, the deploy
#: reported success, and the container was OOM-killed four minutes later.
CONTEXT_MULTIPLIED_MEASUREMENT = {
    "about": (
        "Reference measurement from a previous run. It is the evidence for how "
        "an unstated slot count behaves, not a reading taken from the "
        "deployment it is shown beside."
    ),
    "requested_tokens": 4096,
    "slots": 4,
    "allocated_tokens": 16384,
    "kv_bytes_allocated": 16384 * 112 * 1024,
    "container_limit_mib": 2816,
    "oom_anon_kib": 2444384,
    "oom_file_kib": 493660,
    "symptom": (
        "the server started, /health answered 200, and the only notice was a "
        "line in the process log: srv load_model: initializing, n_slots = 4, "
        "n_ctx_slot = 4096, kv_unified = 'true' - followed four minutes later "
        "by an out-of-memory kill with CONSTRAINT_MEMCG"
    ),
}

#: Context states that are a degradation and must be announced. ``capped`` was
#: the only one, which is why nothing said anything when the window went the
#: other way: a caller asking *"was it capped?"* is asking half the question.
#:
#: ``below-brief`` is the third question, and the appliance answered it the
#: expensive way on 2026-08-09: a redeploy sized itself down to a 4,096-token
#: window and reported *"the server built the 4096 token context it was asked
#: for"*, which is true and useless. The window was smaller than the standing
#: brief Vaelor puts in front of every Assistant question, so the deployment
#: was broken at the moment it was called healthy. **Verifying that the server
#: honoured the request says nothing about whether the request was sane.**
ANNOUNCED_CONTEXT_STATES = ("capped", "multiplied", "below-brief")


def _first_int(*values: Any) -> Optional[int]:
    """The first value that is a usable positive integer, or ``None``."""
    for value in values:
        if value is None or isinstance(value, bool):
            continue
        try:
            number = int(value)
        except (TypeError, ValueError):
            continue
        if number > 0:
            return number
    return None


def _settings(props: Optional[Mapping[str, Any]]) -> Mapping[str, Any]:
    settings = (props or {}).get("default_generation_settings")
    return settings if isinstance(settings, Mapping) else {}


def built_context_tokens(props: Optional[Mapping[str, Any]]) -> Optional[int]:
    """The slot context llama.cpp says it built, or ``None`` if it did not say.

    ``default_generation_settings.n_ctx`` is the field the capping warning
    itself quotes — *"the slot context (147456)"* — so it is read first and the
    top-level fallbacks are only for servers that publish it elsewhere.
    """
    settings = _settings(props)
    params = settings.get("params")
    return _first_int(
        settings.get("n_ctx"),
        (props or {}).get("n_ctx"),
        params.get("n_ctx") if isinstance(params, Mapping) else None,
    )


def trained_context_tokens(props: Optional[Mapping[str, Any]]) -> Optional[int]:
    """The model's training context, where the server publishes it.

    Not every build puts ``n_ctx_train`` on ``/props``, so this is genuinely
    optional: the cap is detected from what was built against what was asked
    for, and the training context only explains it.
    """
    settings = _settings(props)
    meta = (props or {}).get("model_meta") or (props or {}).get("meta")
    return _first_int(
        (props or {}).get("n_ctx_train"),
        settings.get("n_ctx_train"),
        meta.get("n_ctx_train") if isinstance(meta, Mapping) else None,
    )


def verify_context_built(
    requested: Any,
    props: Optional[Mapping[str, Any]] = None,
    *,
    requested_slots: Any = 1,
) -> Dict[str, Any]:
    """Whether the server built the context window it was asked for.

    The same question :func:`verify_accelerator_in_use` asks about the GPU, and
    for the same reason: llama.cpp caps a slot context above the model's
    training context, starts anyway, answers ``/health`` with 200, and says so
    only in its own log.

    **The window can also multiply, and this check used to miss that entirely.**
    It compared the per-slot figure against the request, so four slots each
    holding the full 4,096 tokens that were asked for looked exactly like one
    slot holding them — while costing four times the KV cache. A guard written
    for *"the runtime did not give you what you asked for"* that only tests
    ``built < requested`` is answering half its own question, and the half it
    skipped is the one that ended in an OOM kill
    (:data:`CONTEXT_MULTIPLIED_MEASUREMENT`).

    So the comparison is now made twice, against the two numbers that matter:
    ``built`` against the request is what a *prompt* can use, and
    ``built × slots`` against the request is what *memory* has to hold.

    ``unknown`` is load-bearing exactly as ``in_use: None`` is. A server that
    does not publish its context has told us nothing — and *nothing* must not
    render as "capped". A caller that treats a missing reading as a cap will
    report a defect that is not there, which is the same class of error as
    reporting health that is not there.
    """
    asked = _first_int(requested) or 0
    built = built_context_tokens(props)
    trained = trained_context_tokens(props)
    slots = _first_int((props or {}).get("total_slots")) or 1
    wanted_slots = _first_int(requested_slots) or 1
    result: Dict[str, Any] = {
        "requested": asked,
        "built": built,
        "trained": trained,
        "total_slots": slots,
        "requested_slots": wanted_slots,
        "slots_as_requested": slots == wanted_slots,
        # Tokens of KV the server is holding: the per-slot window in every one
        # of its slots. `built` alone was the whole reading, and `built` alone
        # cannot see this number.
        "allocated": built * slots if built is not None else None,
        "capped": None,
        "multiplied": None,
        "unknown": True,
        "state": "unknown",
        "detail": "",
    }
    if not asked:
        result["detail"] = (
            "No context window was requested, so there is nothing to check the "
            "server's own figure against."
        )
        return result
    if built is None:
        result["detail"] = (
            "The model server does not publish the context window it built, so "
            "whether the {} tokens requested were honoured could not be "
            "verified. This is not a failure; it is an unanswered question."
        ).format(asked)
        return result
    allocated = built * slots
    slot_clause = (
        "" if slots == wanted_slots else
        " {} slot{} were asked for and the server started {}.".format(
            wanted_slots, "" if wanted_slots == 1 else "s", slots
        )
    )
    if allocated > asked:
        result.update({
            "capped": False,
            "multiplied": True,
            "unknown": False,
            "state": "multiplied",
            REFERENCE_MEASUREMENT_KEY: CONTEXT_MULTIPLIED_MEASUREMENT,
            "detail": (
                "The context window was multiplied, not honoured: {} tokens "
                "were requested and the server holds {} of {} tokens each, "
                "which is {} tokens of KV cache.{} Nothing in the HTTP surface "
                "says so — the server starts and /health answers 200 — and the "
                "memory was budgeted for the {} tokens that were asked for. On "
                "this appliance the same shape reached {} tokens against {} "
                "requested and the container was killed for exceeding its "
                "limit four minutes after the deploy reported success."
            ).format(
                asked,
                "{} slots".format(slots) if slots > 1 else "one slot",
                built, allocated, slot_clause,
                asked,
                CONTEXT_MULTIPLIED_MEASUREMENT["allocated_tokens"],
                CONTEXT_MULTIPLIED_MEASUREMENT["requested_tokens"],
            ),
        })
        return result
    if slots > 1 and allocated == asked:
        # One window divided between slots. The total is exactly what was
        # asked for, so memory is as budgeted, but no single request can use
        # more than its own slot — and that is worth saying rather than
        # calling the deployment simply correct.
        result.update({
            "capped": False,
            "multiplied": False,
            "unknown": False,
            "state": "divided",
            "detail": (
                "The {} tokens requested were divided across {} slots of {} "
                "tokens each, so the memory is as budgeted but no single "
                "request can use more than {} tokens.{}"
            ).format(asked, slots, built, built, slot_clause),
        })
        return result
    if built >= asked:
        # The server did what it was told. Whether it was told something
        # workable is a different question, and until this branch asked it a
        # window too small for Vaelor's own standing brief passed as a success.
        from .inference_context import AGENT_PROMPT_TOKENS

        if built < AGENT_PROMPT_TOKENS:
            result.update({
                "capped": False,
                "multiplied": False,
                "unknown": False,
                "state": "below-brief",
                "detail": (
                    "The server built the {} token context it was asked for, "
                    "but that is smaller than the {} token brief Vaelor puts "
                    "in front of every Assistant question, so the Assistant "
                    "cannot fit its own instructions. This is not the server "
                    "declining anything - it built what it was asked for, and "
                    "what it was asked for is too small. Deploy the model "
                    "again when the machine is not also holding the model "
                    "this one replaced."
                ).format(built, AGENT_PROMPT_TOKENS),
            })
            return result
        result.update({
            "capped": False,
            "multiplied": False,
            "unknown": False,
            "state": "as-requested",
            "detail": (
                "The server built the {} token context it was asked for."
            ).format(asked),
        })
        return result
    trained_clause = (
        " The model was trained on {} tokens, and a slot cannot exceed that."
        .format(trained) if trained else
        " The server did not publish the model's training context, so how far "
        "it could have gone is not known here."
    ) + (
        # Short of the request *and* split across slots. The two shortfalls are
        # different faults and a reader needs both numbers to tell which one
        # this is, so the slot count travels with the cap.
        " The window is also divided across {} slots, so {} tokens of KV are "
        "held in total.".format(slots, allocated) if slots > 1 else ""
    ) + slot_clause
    result.update({
        "capped": True,
        "multiplied": False,
        "unknown": False,
        "state": "capped",
        REFERENCE_MEASUREMENT_KEY: CONTEXT_CAP_MEASUREMENT,
        "detail": (
            "The context window was capped: {} tokens were requested and the "
            "server built {}.{} Nothing in the HTTP surface says so — the "
            "server starts and /health answers 200 — and the overshoot is not "
            "free: {} requested built the same {} token slot and held {} MiB "
            "against {} MiB, {} MiB that no request can use."
        ).format(
            asked, built, trained_clause,
            CONTEXT_CAP_MEASUREMENT["requested_tokens"],
            CONTEXT_CAP_MEASUREMENT["built_tokens"],
            CONTEXT_CAP_MEASUREMENT["resident_mib_requested"],
            CONTEXT_CAP_MEASUREMENT["resident_mib_at_training_context"],
            CONTEXT_CAP_MEASUREMENT["wasted_mib"],
        ),
    })
    return result


def _read_json(url: str, timeout: float) -> Optional[Any]:
    request = urllib.request.Request(url, headers={"Accept": "application/json"})
    try:
        with urllib.request.urlopen(request, timeout=timeout) as response:
            return json.loads(response.read(256 * 1024).decode("utf-8", "replace"))
    except (OSError, ValueError, urllib.error.URLError):
        return None


def verify_context_from_server(
    base_url: str,
    requested: Any,
    *,
    requested_slots: Any = 1,
    timeout: float = 3.0,
    fetch: Optional[Callable[[str, float], Optional[Any]]] = None,
) -> Dict[str, Any]:
    """Read ``/props`` from a server that has just started, and check its window.

    ``fetch`` is injected so the check is testable without a network, and
    resolved at call time rather than bound as a default so that patching the
    reader works — a default argument captures the function at import and
    silently ignores the substitution.

    **Nothing this raises may undo a deployment that succeeded.** This is a
    diagnostic read taken *after* the server is healthy and the credential is
    active; a server that answers ``/props`` with something unexpected has
    failed to answer a question, which is ``unknown``, not a failed deploy.
    """
    url = "{}{}".format(str(base_url or "").rstrip("/"), CONTEXT_PROPS_PATH)
    try:
        props = (fetch or _read_json)(url, timeout)
    except Exception:
        props = None
    return verify_context_built(
        requested,
        props if isinstance(props, Mapping) else None,
        requested_slots=requested_slots,
    )
