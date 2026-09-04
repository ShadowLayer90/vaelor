"""Whether THIS machine can serve the ROCmFP4 GPU model, and the pick that follows.

The GPU analogue of the NPU path (:func:`vaelor.flm_service.discover_npu_serving`
plus :mod:`vaelor.inference_model_choice`): a capability gate that fails closed,
and a thin builder that hands :mod:`vaelor.copilot_setup` the override making the
tested 27B (``qwen3-8-27b-rocmfp4-fast``) the recommended local model.

The defect this exists to prevent is the phantom model. The ROCmFP4 27B loads
only in the ROCmFPX llama.cpp fork on a gfx1151 GPU, so recommending it on a Pi,
on an x86 box with no such GPU, or on a gfx1151 box where the fork is not yet
provisioned would name a model the machine cannot run - which is exactly what the
catalog's engine gate exists to stop. The discovery below keeps the invitation
honest: all three conditions must hold before the 27B is offered.

It deliberately does **not** require the model already downloaded. The
recommendation IS the invitation to install it, so gating on the artifact would
be chicken-and-egg - the model can never be present until it has first been
recommended. The download path still runs its own RAM/format/byte checks before
anything is fetched; this gate only decides whether the offer is truthful.
"""

from __future__ import annotations

import os
from typing import Any, Callable, Dict, Iterable, Mapping, Optional, Sequence

from .gpu_rocmfpx_service import (
    GPU_ENGINE_BINARY,
    ROCM_RUNTIME_LIB_CANDIDATES,
    ROCM_RUNTIME_PROBE_SONAME,
    resolve_rocm_lib_dir,
)
from .model_catalog import catalog_model

#: The one engine-gated catalog entry this whole module is about. The same id the
#: catalog carries (``engine="rocmfpx"``); read from the catalog rather than
#: re-spelled, so a rename there cannot leave this pointing at nothing.
GPU_CHAT_MODEL_ID = "qwen3-8-27b-rocmfp4-fast"

#: How a gfx1151 accelerator names itself in ``discover_accelerators`` records
#: (:data:`vaelor.platforms.accelerators.KNOWN_DEVICES` maps the PCI id to
#: "AMD Radeon 8060S (Strix Halo)"). Matched on the marker tokens, not the whole
#: string, so a later name revision that keeps "8060S" or "Strix" still resolves,
#: and a bare "gfx1151" form is admitted for the same reason. Lower-cased tokens;
#: the match is case-insensitive.
GFX1151_NAME_MARKERS = ("gfx1151", "8060s", "strix")


def _names_a_gfx1151(accelerators: Iterable[Any]) -> bool:
    """True when a Strix Halo / Radeon 8060S (gfx1151) GPU is in the list.

    Reads each accelerator the way the rest of the code does - off the ``name``
    the DRM discovery assigned - and matches the marker tokens rather than the
    exact string. Anything that is not a mapping with a name is skipped rather
    than raising, so a malformed inventory fails closed to "absent" instead of
    throwing on the way to a recommendation.
    """
    for accelerator in accelerators or ():
        if not isinstance(accelerator, Mapping):
            continue
        text = str(accelerator.get("name") or "").lower()
        if any(marker in text for marker in GFX1151_NAME_MARKERS):
            return True
    return False


def names_a_gfx1151(accelerators: Iterable[Any]) -> bool:
    """Public marker check: is a gfx1151 GPU in this accelerator inventory?

    A one-line delegate to :func:`_names_a_gfx1151` so another module can ask the
    same capability question without importing a private name or, worse,
    re-spelling the marker tokens. The capability signal has one owner here; the
    credential-adoption policy (:func:`vaelor.managed_local_credentials.
    activate_managed_local`) reads it through this door so a Z2 (a gfx1151 box
    that serves the NPU Assistant and a separate GPU AI-Chat tier) is recognised
    as dual-accelerator rather than adopting ``ai-chat`` onto the NPU.
    """
    return _names_a_gfx1151(accelerators)


def _unavailable_reason(
    gpu_present: bool,
    binary_present: bool,
    libs_present: bool,
    binary: str,
    lib_candidates: Sequence[str],
) -> str:
    """The specific gap, so an unprovisioned box says what to do, not a bare False.

    Honest absence is the point, the same discipline as ``npu_serving_verdict``:
    a machine missing any one condition names it rather than reporting a blank
    "not available", so an operator can act on the reason.
    """
    missing = []
    if not gpu_present:
        missing.append(
            "no gfx1151 GPU (Strix Halo / Radeon 8060S) is present"
        )
    if not binary_present:
        missing.append(
            "the ROCmFPX engine is not installed or executable at {}".format(binary)
        )
    if not libs_present:
        missing.append(
            "no gfx1151 ROCm runtime ({}) is present on any of {}".format(
                ROCM_RUNTIME_PROBE_SONAME, ", ".join(lib_candidates)
            )
        )
    return (
        "Vaelor cannot serve the ROCmFP4 GPU model on this machine: {}. It "
        "becomes available once the gfx1151 GPU, the ROCmFPX engine and the "
        "gfx1151 ROCm runtime are all present."
    ).format("; ".join(missing))


def discover_gpu_rocm_serving(
    hardware: Dict[str, Any],
    *,
    binary: str = GPU_ENGINE_BINARY,
    lib_candidates: Sequence[str] = ROCM_RUNTIME_LIB_CANDIDATES,
) -> Dict[str, Any]:
    """Whether this machine can run the ROCmFP4 GPU model, and why not when it can't.

    The GPU analogue of :func:`vaelor.flm_service.discover_npu_serving`. All three
    preconditions must hold:

    1. a **gfx1151 accelerator** (Strix Halo / Radeon 8060S) is in
       ``hardware["accelerators"]`` - the only condition read off the passed
       inventory rather than the filesystem;
    2. the **ROCmFPX fork binary** exists and is executable
       (``os.access(binary, os.X_OK)``); and
    3. a **gfx1151 ROCm runtime lib dir** is present - resolved through the SAME
       :func:`vaelor.gpu_rocmfpx_service.resolve_rocm_lib_dir` the launch path
       uses, so discovery and the launch agree on what "the libs are present"
       means. It is present iff the resolver returns a real dir (one of the
       ordered candidates that actually contains ``libamdhip64.so.7``) rather than
       ``None``. A bare ``os.path.isdir`` on one hardcoded path would report the
       stale snap dir absent while ``/opt/rocm`` served, disagreeing with the
       launch.

    It does not check that the model is downloaded - see the module docstring.
    Fails closed: any exception or missing field yields ``available`` False,
    because a recommendation made on a bad read is the phantom-model defect this
    gate exists to stop.
    """
    try:
        accelerators = (hardware or {}).get("accelerators") or []
        gpu_present = _names_a_gfx1151(accelerators)
        binary_present = bool(binary) and os.access(binary, os.X_OK)
        libs_present = resolve_rocm_lib_dir(lib_candidates) is not None
    except Exception:
        # A malformed hardware dict or an OSError from the filesystem probes must
        # not raise on the way to a recommendation; it must read as "cannot serve".
        gpu_present = binary_present = libs_present = False
    available = bool(gpu_present and binary_present and libs_present)
    return {
        "available": available,
        "gpu_present": bool(gpu_present),
        "binary_present": bool(binary_present),
        "libs_present": bool(libs_present),
        "reason": "" if available else _unavailable_reason(
            gpu_present, binary_present, libs_present, binary, lib_candidates
        ),
        "model_id": GPU_CHAT_MODEL_ID,
    }


def gpu_chat_recommendation(
    hardware: Dict[str, Any],
    *,
    discover: Optional[Callable[[Dict[str, Any]], Dict[str, Any]]] = None,
) -> Dict[str, Any]:
    """The override that makes the ROCmFP4 27B the recommended pick, or ``{}``.

    Returns ``{}`` on every machine that cannot serve the GPU model, leaving the
    base (GGUF, or NPU-overridden) recommendation untouched - the Pi / x86 /
    unprovisioned isolation. When the gate is open it hands back exactly the
    fields :func:`vaelor.copilot_setup.copilot_setup_status` merges over the base:

    * ``primary`` is the 27B catalog entry, carrying the GPU markers
      (``served_on_gpu``, ``backend``) so the card renderer can badge it without
      parsing prose - mirroring the NPU's ``served_on_npu``/``backend`` precedent.
    * the header/size fields (``parameter_range``, ``quantization``,
      ``context_tokens``) name the 27B, so "Vaelor recommends {name}" resolves to
      it.
    * **the headroom keys are cleared** - ``exceeds_catalog`` False and
      ``catalog_note`` empty - so the "this machine could run a larger model /
      the catalog stops at 4B / search Hugging Face" banner does not render
      beside the recommended 27B. The 27B *is* that larger model, now offered
      rather than described.
    * ``served_on_npu`` is set False so the single shared recommendation does not
      claim two served surfaces at once: on a Z2 that serves both, the GPU pick
      is the capable downloadable model that fills this catalog modal's headroom
      slot. ``assistant_selection`` (the NPU Assistant's own surface data) is not
      touched here, so a merge that ran the NPU override first keeps it.
    """
    # Resolved at call time (not bound as a default) so a test patching the
    # module-level ``discover_gpu_rocm_serving`` reaches this path too.
    verdict = (discover or discover_gpu_rocm_serving)(hardware)
    if not verdict.get("available"):
        return {}
    entry = catalog_model(GPU_CHAT_MODEL_ID)
    if not entry:
        # The catalog no longer stocks the entry the gate is about; refuse to
        # recommend a name with nothing behind it rather than invent one.
        return {}
    runtime = entry.get("runtime") or {}
    context_tokens = int(runtime.get("context") or 0)
    primary = {
        **entry,
        # Markers a reader (and the ModelCatalog card) branches on without
        # parsing the strings above. `backend` is the fork the deployer launches.
        "served_on_gpu": True,
        "backend": "rocmfpx",
        "device": "gpu",
    }
    return {
        "tier": "accelerated",
        "parameter_range": entry["parameter_size"],
        "quantization": entry["quantization"],
        "context_tokens": context_tokens,
        "primary": primary,
        # The headroom banner is suppressed: the recommended model IS the larger
        # model the banner used to point at, so the two must not render together.
        "exceeds_catalog": False,
        "catalog_note": "",
        "storage_ok": True,
        "can_install": True,
        "storage_reason": "",
        "rationale": (
            "Runs on this machine's gfx1151 GPU in ROCmFP4 format, through the "
            "ROCmFPX engine, for markedly stronger answers; the 27B loads "
            "nowhere else."
        ),
        # So a reader can tell a GPU-served recommendation from a GGUF or NPU one,
        # and so the single shared recommendation names one served surface.
        "served_on_gpu": True,
        "served_on_npu": False,
        "model_id": GPU_CHAT_MODEL_ID,
    }
