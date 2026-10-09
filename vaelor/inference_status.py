"""Per-engine inference status: what is loaded, and what memory it holds.

The Memory page needs to answer "where has it gone" on a machine where the
NPU, the GPU and the host draw on one pool. That question cannot be answered
from a single active connection, because a two-tier appliance has a model
resident on each engine and the interesting number is how they divide the
pool between them.

Everything here is assembly over facts other modules already establish:
:func:`~vaelor.model_sizing.local_inference_tiers` for the engines and the
shared budget, :func:`~vaelor.copilot_setup.copilot_setup_status` for the
measured plan, and :func:`~vaelor.model_reachability.probe_connection` for
which models a server currently holds. Nothing is probed that was not going
to be probed anyway, and nothing is invented where a fact is missing: an
engine whose resident size is unknown says so rather than reporting zero.
"""

from __future__ import annotations

from typing import Any, Callable, Dict, List, Mapping, Optional, Sequence

from .accelerator_runtime import (
    ACCELERATED,
    BASIS_ABSOLUTE,
    MIB,
    NOT_ESTABLISHED,
    NOTHING_SERVED,
    MINIMUM_ACCELERATED_BYTES,
    reported_gpu_memory_bytes,
    verify_accelerator_in_use,
)
from .chat_connections import connection_locality
from .cluster_placement import CONTROLLER_PLACEMENT_ID
from .gpu_serving_target import UNLOAD_CAUSE_IDLE, UNLOAD_CAUSE_MANUAL
from .inference_tuning import (
    RECOMMENDED_CONTEXT_TOKENS,
    loaded_model_settings,
    recommended_deployment,
)
from .model_sizing import local_inference_tiers
from .provider_runtime import managed_local_connection


#: Engines report memory in bytes; a caller that wants MB divides. Stated so
#: nobody has to guess from a field name.
MEMORY_UNIT = "bytes"


def _int(value: Any) -> int:
    try:
        if isinstance(value, bool):
            return 0
        return max(0, int(value or 0))
    except (TypeError, ValueError):
        return 0


def _model_display_name(identifier: str) -> str:
    """The model name a server offers, cleaned to its basename without a weight
    extension - ``/models/Qwen3-4B-Instruct.gguf`` -> ``Qwen3-4B-Instruct``.

    Only a trailing weight extension is stripped, never text after an arbitrary
    dot, so ``Qwen3.5-4B`` survives intact.
    """
    name = str(identifier or "").replace("\\", "/").rsplit("/", 1)[-1]
    for ext in (".gguf", ".q4nx", ".safetensors", ".bin"):
        if name.lower().endswith(ext):
            return name[: -len(ext)]
    return name


def _engine_health(
    tier: Mapping[str, Any], probe: Optional[Mapping[str, Any]]
) -> Dict[str, Any]:
    """Reachability for an engine, distinguishing "not asked" from "down"."""
    if probe is None:
        return {
            "state": "unknown",
            "reachable": None,
            "detail": (
                "No model server is configured for this engine, so nothing was "
                "asked."
            ),
        }
    if probe.get("reachable"):
        return {"state": "ready", "reachable": True, "detail": ""}
    return {
        "state": "unreachable",
        "reachable": False,
        "detail": probe.get("detail", "") or "The model server did not answer.",
    }


def _models_match(wanted: Any, candidate: Any) -> bool:
    """Whether two identifiers name the same model, tolerant of a path or a
    weight extension on one side.

    Factored out of :func:`_tier_connection` so the active-lease attachment can
    keep the NPU tier out with exactly the same rule the tier matcher uses: the
    NPU's assistant model never equals the GPU's chat lease, so a by-model guard
    never hands one tier's connection to the other (VD-110 #247n, LESSONS 8).
    """
    wanted = str(wanted or "").casefold()
    candidate = str(candidate or "").casefold()
    return bool(
        wanted
        and candidate
        and (candidate == wanted or wanted in candidate or candidate in wanted)
    )


def _tier_connection(
    tier: Mapping[str, Any], connections: Sequence[Mapping[str, Any]]
) -> Optional[Mapping[str, Any]]:
    """The configured connection serving this engine, if any.

    Matched on the engine's own model name rather than on position, because
    the order connections happen to be stored in says nothing about which
    engine they drive.
    """
    wanted = tier.get("model")
    for connection in connections:
        if _models_match(wanted, connection.get("model")):
            return connection
    return None


#: What a vLLM deployment record's ``units["engine"]`` says. The word is
#: `gpu_pool_units.VLLM_ENGINE`'s; spelled here rather than imported because
#: that module's import chain is the executor's, not this read path's.
_CLUSTER_ENGINE = "vllm"
#: The record states in which a deployment still owns its nodes' GPUs:
#: starting, serving, or unloaded by scale-to-zero (the mode file stays Mode B
#: and the next request loads it back). Only ``failed`` releases them.
_CLUSTER_PLACED_STATES = ("deploying", "healthy", "unloaded")
#: The readable name of the cluster engine, for the owner-facing sentences.
_CLUSTER_ENGINE_NAME = "vLLM"
#: A reading attributed from the serving record rather than from a baseline:
#: the record places a vLLM server on this machine, and the adapter's total is
#: what it reports. Neither `differential` nor `absolute` describes that.
BASIS_SERVING_RECORD = "serving-record"


def cluster_on_this_adapter(
    records: Optional[Sequence[Mapping[str, Any]]],
) -> Optional[Mapping[str, Any]]:
    """The GPU cluster deployment with a server on THIS machine, if any (ACC-114).

    Read off the stored ``pooled_deployments`` records the cluster store
    returns: a vLLM record whose ``node_ids`` include the controller runs a
    server here (`gpu_pool_units.serving_units` derives the units the same
    way), and while it is starting or serving that server - not llama.cpp,
    which the mode switch stopped - is what holds this adapter's memory. An
    unloaded record is returned too: the adapter still belongs to that
    deployment, and describing it as llama.cpp's would be the ACC-114 defect
    again in the scale-to-zero window.
    """
    for record in records or []:
        units = record.get("units") if isinstance(record, Mapping) else None
        if not isinstance(units, Mapping):
            continue
        if str(units.get("engine") or "") != _CLUSTER_ENGINE:
            continue
        if str(record.get("state") or "") not in _CLUSTER_PLACED_STATES:
            continue
        nodes = [str(node) for node in record.get("node_ids") or []]
        if CONTROLLER_PLACEMENT_ID in nodes:
            return record
    return None


#: The unloaded sentence's second half, by the cause the one unload-cause
#: rule (`gpu_serving_target.unload_cause`, VD-136) gives; any other answer,
#: "" (not known) included, promises no wake.
_UNLOADED_BY_CAUSE = {
    UNLOAD_CAUSE_IDLE: "was unloaded after sitting idle; it loads again on the next AI Chat request.",
    UNLOAD_CAUSE_MANUAL: (
        "was unloaded by hand and stays unloaded until it is loaded again from "
        "Cluster > Deployments."
    ),
}


def _cluster_acceleration(
    record: Mapping[str, Any], accelerators: Sequence[Mapping[str, Any]],
    unload_cause: str = "",
) -> Dict[str, Any]:
    """This adapter's reading while it serves a cluster deployment's replica.

    Never framed as the single-machine engine: the llama.cpp library is not
    loaded in this mode, and naming it here is what told the owner the wrong
    engine held 26 GB of memory. ``unload_cause`` is the caller's answer from
    the one unload-cause rule for an unloaded record; it is not derived here.
    """
    name = str(record.get("name") or "")
    model = str(record.get("model_id") or "")
    observed = reported_gpu_memory_bytes(accelerators)
    serving = "This machine is part of the GPU cluster deployment \"{}\"{}".format(
        name, " ({})".format(model) if model else "",
    )
    result: Dict[str, Any] = {
        "backend": _CLUSTER_ENGINE,
        "requested_backend": _CLUSTER_ENGINE,
        "compute_library": "",
        "gpu_memory_bytes": observed,
        "baseline_bytes": None,
        "basis": BASIS_SERVING_RECORD,
        "cluster_deployment": name,
        "in_use": None,
        "state": NOT_ESTABLISHED,
    }
    if str(record.get("state") or "") == "unloaded":
        label = "The GPU cluster deployment \"{}\"{}".format(
            name, " ({})".format(model) if model else "",
        )
        cause_sentence = _UNLOADED_BY_CAUSE.get(unload_cause)
        result.update({
            "state": NOTHING_SERVED,
            "detail": (
                "{} {}".format(label, cause_sentence) if cause_sentence else
                "{} is unloaded, so nothing is loaded on this graphics processor "
                "right now.".format(label)
            ),
        })
        return result
    if str(record.get("state") or "") != "healthy":
        result["detail"] = (
            "{}, and its {} server on this graphics processor is still "
            "starting.".format(serving, _CLUSTER_ENGINE_NAME)
        )
        return result
    if observed is None:
        result["detail"] = (
            "{}, served here by {}. The graphics processor does not report how "
            "much of its memory is in use, so that could not be "
            "confirmed.".format(serving, _CLUSTER_ENGINE_NAME)
        )
        return result
    # The memory guard: a healthy record over an adapter holding less than a
    # loaded model means the replica is not running here. Residual: a healthy
    # record over a dead replica container while something ELSE holds more
    # than MINIMUM_ACCELERATED_BYTES still reads as the cluster's. Probing the
    # record's endpoint would not settle it either - that is the balancer,
    # which answers while any replica does - so only a per-node unit read
    # (the executor's, not this read path's) could.
    if observed < MINIMUM_ACCELERATED_BYTES:
        result["detail"] = (
            "{}, but the graphics processor is holding only {:.0f} MiB - less "
            "than a loaded model - so its {} server does not appear to be "
            "running here.".format(serving, observed / MIB, _CLUSTER_ENGINE_NAME)
        )
        return result
    result.update({
        "in_use": True,
        "state": ACCELERATED,
        "held_bytes": observed,
        "detail": (
            "{}. {}, the cluster engine, serves it from this graphics processor, "
            "which is holding {:.0f} MiB. The single-machine engine is stopped "
            "while the graphics processor is clustered."
        ).format(serving, _CLUSTER_ENGINE_NAME, observed / MIB),
    })
    return result


def _clustered_elsewhere() -> Dict[str, Any]:
    """Mode B with no cluster server found on this machine.

    The single-machine engine was stopped when the GPU was clustered, so the
    adapter's memory is not llama.cpp's; which server holds it is not known.
    """
    return {
        "backend": _CLUSTER_ENGINE,
        "in_use": None,
        "state": NOT_ESTABLISHED,
        "basis": BASIS_SERVING_RECORD,
        "detail": (
            "This graphics processor is set aside for the GPU cluster, but no "
            "cluster deployment was found running a server on this machine, so "
            "what is using its memory is not established."
        ),
    }


def _acceleration(
    kind: str,
    backend: str,
    accelerators: Sequence[Mapping[str, Any]],
    health: Mapping[str, Any],
    configured: bool = True,
) -> Optional[Dict[str, Any]]:
    """Whether the GPU engine is actually GPU-backed, when that is askable.

    Only the GPU tier can fall back to the CPU without saying so, and the
    question is only meaningful once a server is answering: an engine with
    nothing running holds no accelerator memory for entirely innocent reasons,
    and reporting that as "running on CPU" would be a lie in the other
    direction. So an engine that is not ready reports ``unknown`` with the
    reason, never ``cpu-fallback``.

    **This reading is absolute, and says so.** A deploy can take a baseline
    while nothing of its own is resident and report a delta; a live panel has
    no "before" to compare against and never will, so it reports the total.
    The two answer the same question differently on purpose, and the
    ``basis`` :func:`~vaelor.accelerator_runtime.verify_accelerator_in_use`
    returns is what keeps that a stated difference rather than a
    contradiction: an absolute reading establishes that the accelerator is in
    use, not that this engine is what is using it.
    """
    if kind != "gpu":
        return None
    if not configured:
        # No model server is configured for this engine at all. That is a
        # plain fact about this machine, not an unanswered question.
        return {
            "backend": str(backend or ""),
            "in_use": None,
            "state": NOTHING_SERVED,
            "basis": BASIS_ABSOLUTE,
            "detail": "Vaelor is not serving an AI model on this graphics processor.",
        }
    if not health.get("reachable"):
        return {
            "backend": str(backend or ""),
            "in_use": None,
            "state": NOT_ESTABLISHED,
            "detail": (
                "No model server is answering on this engine, so whether the "
                "GPU library loaded has not been established."
            ),
        }
    return verify_accelerator_in_use(backend, accelerators)


def inference_status(
    hardware: Optional[Mapping[str, Any]] = None,
    *,
    connections: Optional[Sequence[Mapping[str, Any]]] = None,
    probe: Optional[Callable[[Mapping[str, Any]], Mapping[str, Any]]] = None,
    resident_bytes: Optional[Mapping[str, int]] = None,
    resolve_local: Optional[Callable[[], Optional[Mapping[str, Any]]]] = None,
    cluster_deployments: Optional[Sequence[Mapping[str, Any]]] = None,
    cluster_mode: bool = False,
    cluster_unload_cause: str = "",
) -> Dict[str, Any]:
    """What each engine is holding, and what that leaves.

    ``cluster_deployments`` are the stored GPU cluster records. When one of
    them runs a server on this machine, the GPU engine is that deployment's
    vLLM replica and is reported as such (`cluster_on_this_adapter`).
    ``cluster_mode`` is the serving-mode file's answer (Mode B): with it set and
    no record placing a server here, the GPU is not described as llama.cpp's,
    because the switch stopped llama.cpp when it entered the mode.
    ``cluster_unload_cause`` is `gpu_serving_target.deployment_unload_cause`'s
    answer for that record when it is unloaded, supplied by the caller.

    ``probe`` is injected so this is testable without a network and so a
    caller that already has fresh probe results does not pay for them twice.
    When it is omitted, engines report their configuration and say that
    liveness was not checked — which is a different answer from "down" and
    must not render as one.

    ``resolve_local`` returns the *resolved* active managed-local (``ai-chat``)
    connection - with a ``base_url`` and the served model - the one AI Chat
    itself reaches. The records in ``connections`` are redacted (no ``base_url``,
    model in ``selected_model``), so a local engine could be identified but not
    probed off them; the resolver supplies the endpoint. It is attached to
    whichever LOCAL tier runs the active chat model: the CPU on a Pi (#205 Step
    3, VD-071), the GPU on a Z2 (VD-110 #247n) - never the NPU.
    """
    facts = dict(hardware or {})
    accelerators = list(facts.get("accelerators") or [])
    neural = list(facts.get("neural_accelerators") or [])
    known = list(connections or [])
    resident = dict(resident_bytes or {})

    # Real capability discovery for the NPU tier's availability (VD-001), only
    # when there is a neural accelerator to serve - a CPU-only Pi never reaches
    # `flm_service` and its tier stays honestly absent for want of the device
    # `local_inference_tiers` never builds a tier for. The tag the discovery
    # checks against the installed inventory is the one the plan would launch.
    npu_serving = None
    if neural:
        from .flm_service import discover_npu_serving
        from .inference_tuning import npu_tier_plan

        plan = npu_tier_plan()
        npu_serving = discover_npu_serving(
            str(plan.get("flm_tag") or ""),
            device_node=str(neural[0].get("device_node") or "") or None,
        )
    tiers = local_inference_tiers(
        _int(facts.get("memory_total_bytes")),
        accelerators,
        neural,
        resident_bytes=resident,
        npu_serving=npu_serving,
    )
    plan = recommended_deployment(facts)
    plan_by_kind = {str(item.get("kind")): item for item in plan.get("tiers", [])}

    chosen_backend = str((plan.get("backend") or {}).get("backend") or "")
    tier_kinds = {str(item.get("kind")) for item in tiers.get("tiers", [])}
    # The listed records `known` are redacted - no base_url, model in
    # `selected_model`, identity in the `cred_managed_local_` id prefix - so a
    # local tier is identifiable off them (via `connection_locality`, NOT
    # base_url, which the live broker.list() never emits) but never probeable.
    # `resolve_local` supplies the ONE full lease AI Chat itself reaches
    # (`resolve_active("ai-chat")` -> base_url + the served model), so the health
    # widget can consult the SAME establishment signal AI Chat uses for the LOCAL
    # tier running the active chat model - the CPU on a Pi, the GPU on a Z2.
    # Without it a demonstrably-serving GPU 27B reported health=unknown ->
    # "Not established" (VD-110 #247n), and the CPU engine reported model=null on
    # a healthy Pi (#205 Step 3, VD-071).
    local_record = next(
        (item for item in known if connection_locality(item).get("local")), None
    )
    resolved_local = (
        resolve_local()
        if (local_record is not None and resolve_local is not None)
        else None
    )
    active_model = str((resolved_local or {}).get("model") or "")
    # Which local tier hosts the active chat model. The recommendation's tier
    # model can differ from what is deployed - on the Z2 the plan recommends
    # gpt-oss-20b while the GPU actually serves the 27B - so the lease's served
    # model does NOT always equal a tier's planned model; the host is the
    # accelerator tier when present, else the CPU tier. It is NEVER the NPU,
    # which serves the assistant through a separate purpose and keeps its own
    # verdict from `discover_npu_serving`; handing it this chat lease would
    # overwrite that working verdict with the wrong endpoint (LESSONS 8).
    chat_tier_kind = (
        "gpu" if "gpu" in tier_kinds else ("cpu" if "cpu" in tier_kinds else "")
    )
    cluster = cluster_on_this_adapter(cluster_deployments)
    engines: List[Dict[str, Any]] = []
    for tier in tiers.get("tiers", []):
        kind = str(tier.get("kind"))
        planned = plan_by_kind.get(kind, {})
        model = str(tier.get("model") or planned.get("model") or "")
        connection = _tier_connection({**tier, "model": model}, known)
        # No redacted record carried a base_url to probe. Fall back to the
        # resolved active-chat lease for the LOCAL tier that runs it: a by-model
        # match (the clean case, and what keeps the NPU out - its assistant model
        # never equals the chat lease's model) OR the accelerator-or-CPU chat
        # host. The NPU is never attached, so its flm verdict is untouched.
        if (
            connection is None
            and resolved_local is not None
            and kind != "npu"
            and (_models_match(model, active_model) or kind == chat_tier_kind)
        ):
            connection = resolved_local
            if kind == "cpu":
                # The CPU engine's name lives on the MATCHED LISTED record's
                # `selected_model` (the credential-store preference) - NOT the
                # resolved connection (whose `model` is '') and NOT the probe's
                # `loaded_models` ([] on the Pi's llama.cpp build). Verified on
                # the box, a53 (#205 Step 3, VD-071, LESSONS 13 - three synthetic
                # fixtures hid three real shapes; only the listed record names it).
                model = str(
                    (local_record or {}).get("selected_model")
                    or (local_record or {}).get("model")
                    or ""
                )
            elif active_model:
                # The accelerator serves the model the lease names, which on the
                # Z2 is the deployed 27B - not the plan's gpt-oss-20b
                # recommendation. Show what is actually resident so the displayed
                # name and the probed endpoint cannot disagree (LESSONS 6).
                model = active_model
        result = probe(connection) if (probe and connection) else None
        if kind == "cpu" and not model and local_record is not None and result:
            # No stored preference (empty `selected_model`), so name the model
            # from what the server *offers*. Still never `loaded_models` - [] on
            # this build - which is why the probe now carries offered names.
            offered = list(result.get("offered_model_names") or [])
            if offered:
                model = _model_display_name(str(offered[0]))
        held = _int(tier.get("resident_bytes"))
        health = _engine_health(tier, result)
        on_cluster = kind == "gpu" and cluster is not None
        if on_cluster:
            model = str(cluster.get("model_id") or model)
        engines.append({
            "kind": kind,
            "role": tier.get("role"),
            "device": tier.get("device"),
            "backend": _CLUSTER_ENGINE if on_cluster else tier.get("backend"),
            "cluster_deployment": str(cluster.get("name") or "") if on_cluster else None,
            "available": bool(tier.get("available")),
            "unavailable_reason": tier.get("reason") or "",
            "model": model or None,
            # Which models the server says it is *holding*, not which it
            # offers. An engine nobody asked reports None, not an empty list:
            # "not checked" and "holding nothing" are different facts.
            "loaded_models": (
                list(result.get("loaded_models", [])) if result is not None else None
            ),
            "context_tokens": _int(
                tier.get("context_tokens") or planned.get("context_tokens")
            ) or RECOMMENDED_CONTEXT_TOKENS,
            "resident_bytes": held,
            "resident_known": held > 0,
            "resident_reason": "" if held > 0 else (
                "Nothing has reported a resident size for this engine, so its "
                "share of the pool is not known."
            ),
            "health": health,
            # Asking for the GPU is not the same as getting it. A llama.cpp
            # that cannot resolve its accelerator library loads the CPU
            # backend and serves normally, nine times slower, with no error.
            "acceleration": (
                _cluster_acceleration(cluster, accelerators, cluster_unload_cause) if on_cluster
                else _clustered_elsewhere() if (kind == "gpu" and cluster_mode)
                else _acceleration(
                    kind, chosen_backend, accelerators, health,
                    configured=connection is not None,
                )
            ),
            "connection_id": (
                connection.get("id") or connection.get("credential_id")
                if connection else None
            ),
            "local": bool(connection and managed_local_connection(dict(connection))),
        })

    claimed = _int(tiers.get("claimed_bytes"))
    total = _int(tiers.get("total_budget_bytes"))
    # What the hardware says is actually in use, against what Vaelor can name.
    # The difference is the honest answer to "where has it gone": memory the
    # driver reports as occupied that no engine on this list accounts for -
    # another process, a display buffer, or a model Vaelor did not start.
    primary = accelerators[0] if accelerators else {}
    measured_used = _int(primary.get("vram_used_bytes")) + _int(
        primary.get("gtt_used_bytes")
    )
    measured_known = bool(
        primary.get("vram_used_bytes") is not None
        or primary.get("gtt_used_bytes") is not None
    )
    # This surface is about THIS machine, so the reference bench box's identity
    # must not appear on any other machine - not even labelled provenance. The
    # backend block carries `benchmarked_on` (the Z2) AND a `measurements`
    # catalogue of the Z2's rocm/vulkan throughput; both are honest on the Z2 and
    # a leak anywhere else (#205, LESSONS 5/6 - the same lie the top-level
    # measured_on had, one level down). Stripping only `benchmarked_on` left the
    # numbers behind with `measured: true` and no provenance, which reads as
    # "this Pi produced them" - so strip the catalogue too, mark it unmeasured
    # here, and say where the figures actually live.
    surfaced_backend = dict(plan.get("backend") or {})
    if not plan.get("is_reference_machine"):
        surfaced_backend.pop("benchmarked_on", None)
        surfaced_backend.pop("measurements", None)
        surfaced_backend["measured"] = False
        surfaced_backend["measurements_note"] = (
            "Backend throughput figures are measured on the reference bench "
            "box, not on this machine."
        )
    return {
        "memory_unit": MEMORY_UNIT,
        "engines": engines,
        "memory": {
            "total_budget_bytes": total,
            "claimed_bytes": claimed,
            "available_bytes": _int(tiers.get("available_bytes")),
            "shared_pool": bool(tiers.get("shared_memory")),
            "note": tiers.get("note", ""),
            "measured_used_bytes": measured_used if measured_known else None,
            "unattributed_bytes": (
                max(0, measured_used - claimed) if measured_known else None
            ),
            "unattributed_reason": "" if measured_known else (
                "The accelerator is not reporting how much of its memory is in "
                "use, so what Vaelor cannot account for is unknown."
            ),
        },
        # The plan already derives this from the passed hardware, so the Pi
        # reports one CPU model and the Z2 reports two - never a hardcoded 2
        # naming an NPU and a GPU this machine may not have (#205, VD-071).
        "capacity": plan.get("loaded_models") or loaded_model_settings(1),
        "backend": surfaced_backend,
        # This machine, as the plan describes it - not the Z2 bench box.
        "measured_on": plan.get("measured_on", ""),
    }
