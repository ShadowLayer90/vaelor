"""One source of truth for cluster capacity: what each node has, what is
committed where, and what is left.

This is the shared, read-only data layer two later features consume — cluster
application placement, and distributed-GPU LLM sizing. Both ask the same two
questions this module answers together for the first time: *what does each
enrolled node physically have* (CPU, system memory, and now accelerator
memory, discovered in `ssh_transport.probe`), and *what has already been
committed to it* (the memory a Swarm service reserved on that node).

The compute is deliberately a pure function of already-fetched facts —
`compute_capacity_ledger(node_facts, reservations)` touches no SSH connection
and no Docker socket, so the capacity/reserved/free arithmetic is unit-tested
without either. `ClusterManager.capacity_ledger` does the fetching and hands
the results here.

Honest degradation is the rule the accelerator half inherits from the probe: a
node with no GPU carries `{"present": false, "reason": ...}`, and a node
enrolled before GPU discovery existed (its stored inventory has no `gpu` key)
is reported absent-with-reason rather than silently credited zero capacity it
might actually have.

**Phase 1 boundary — GPU reservation is a documented stub.** Swarm reserves
*system memory* (`--reserve-memory`), which this ledger attributes to the node
each service is pinned to, so application memory reservations are real. Swarm
does not reserve GPU memory, and a replicated/pooled LLM's per-shard placement
is not resolved here; both are deferred to the distributed-GPU LLM sizing
phase. `reserved.gpu_memory_bytes` is therefore `0` on every node in Phase 1,
and a service constrained only to a pool label (no single `node_id`) is
collected into `unattributed_reservations` with the reason, not dropped and not
guessed onto a node.
"""

from __future__ import annotations

import re
from typing import Any, Dict, List, Optional

from .cluster_placement import (
    CONTROLLER_PLACEMENT_ID,
    CONTROLLER_PLACEMENT_NAME,
)


#: The Swarm placement constraint `deploy_app`/`deploy_llm` write to pin a
#: service to one enrolled node: `node.labels.vaelor.node_id==<node_id>`. Read
#: here to attribute that service's memory reservation to that node.
_NODE_ID_CONSTRAINT = re.compile(r"node\.labels\.vaelor\.node_id==(.+)")

#: The FALLBACK threshold, for a part that carries no `unified_memory` verdict:
#: the largest VRAM figure that still reads as a *carve-out* rather than a
#: dedicated pool. Only a record written before the verdict was carried gets
#: here, and the threshold is a guess where the verdict is knowledge — the same
#: gfx1151 part ships with a BIOS-configurable carve-out that can be 16 GiB, and
#: this number would call that discrete. Two GiB separates the shapes the lab's
#: own machines report, and nothing more is claimed for it.
UNIFIED_VRAM_CARVEOUT_MAX_BYTES = 2 * 1024 ** 3

#: What a node whose aperture is unreadable is told. Never "discrete": that is a
#: claim about the hardware, and no evidence for it was collected — the honest
#: answer is that the question was not answered, and a refresh is the remedy.
_NO_APERTURE_REASON = (
    "This node reported no shared aperture, so whether its memory is unified is "
    "unknown; refresh it to capture the accelerator's identity."
)

#: Phase-1 notes carried on the ledger so a reader never has to infer why a
#: number is what it is. Not a wire vocabulary — plain operator-facing prose.
_LEDGER_NOTES = {
    "gpu_reservation": (
        "GPU memory reservation is not tracked by Swarm and is deferred to "
        "the distributed-GPU LLM sizing phase, so reserved.gpu_memory_bytes is "
        "0 on every node in this phase."
    ),
    "pooled_reservation": (
        "A service constrained only to a pool label has no single node to "
        "attribute its reservation to yet; per-shard placement is resolved in "
        "the distributed-GPU LLM sizing phase. Such services are listed under "
        "unattributed_reservations."
    ),
}


def _int(value: Any) -> int:
    try:
        result = int(value)
    except (TypeError, ValueError):
        return 0
    return result if result > 0 else 0


def _normalize_gpu(inventory: Dict[str, Any]) -> Dict[str, Any]:
    """The node's accelerator facts, normalized, or an honest absence.

    A node enrolled before GPU discovery has no `gpu` key at all; that is not
    "no GPU", it is "not yet asked", and it is reported as such so a refresh —
    not a silent zero — is the remedy.
    """
    gpu = inventory.get("gpu")
    if not isinstance(gpu, dict):
        return {
            "present": False,
            "reason": (
                "This node was enrolled before accelerator discovery; refresh "
                "it to capture GPU facts."
            ),
            "gfx_target_version": "",
            "device_count": 0,
            "vram_total_bytes": 0,
            "vram_used_bytes": 0,
            "gtt_total_bytes": 0,
            "gtt_used_bytes": 0,
            "system_ram_bytes": None,
            "unified_memory": None,
            "unified_memory_source": "",
            "mode_a_reclaimable": False,
            "mode_a_reclaimable_reason": "",
            "addressable_bytes": 0,
            "reclaimable_bytes": 0,
            "memory_model": "",
            "memory_model_reason": "",
        }
    present = bool(gpu.get("present"))
    system_ram = gpu.get("system_ram_bytes")
    unified = gpu.get("unified_memory")
    normalized = {
        "present": present,
        "reason": str(gpu.get("reason", "") or ""),
        "gfx_target_version": str(gpu.get("gfx_target_version", "") or ""),
        "device_count": _int(gpu.get("device_count")),
        "vram_total_bytes": _int(gpu.get("vram_total_bytes")),
        "vram_used_bytes": _int(gpu.get("vram_used_bytes")),
        "gtt_total_bytes": _int(gpu.get("gtt_total_bytes")),
        "gtt_used_bytes": _int(gpu.get("gtt_used_bytes")),
        "system_ram_bytes": (
            int(system_ram) if isinstance(system_ram, int) else None
        ),
        # The probe's own verdict, carried rather than re-derived. `None` says
        # the record predates it and the byte fallback below decides.
        "unified_memory": unified if isinstance(unified, bool) else None,
        "unified_memory_source": str(gpu.get("unified_memory_source", "") or ""),
        # VD-125 (D1): whether the memory this GPU is holding right now belongs
        # to the Mode A AI-Chat model, which the cluster mode switch stops before
        # vLLM starts. A FACT about the node, decided by
        # `gpu_serving_target.gpu_chat_reclaimable` and carried here, so the
        # ledger derives the bytes rather than each caller re-deciding.
        "mode_a_reclaimable": bool(gpu.get("mode_a_reclaimable")),
        # Empty when the question was ANSWERED, either way; a token when it
        # could not be asked at all (VD-127, F5). The fit engine reads it so a
        # verdict sized WITHOUT the AI-Chat model's memory says so, rather than
        # reaching the operator as a bare "will not fit".
        "mode_a_reclaimable_reason": str(
            gpu.get("mode_a_reclaimable_reason", "") or ""
        ),
    }
    normalized["addressable_bytes"] = addressable_gpu_bytes(normalized)
    normalized["reclaimable_bytes"] = reclaimable_gpu_bytes(normalized)
    normalized["memory_model"] = gpu_memory_model(normalized)
    normalized["memory_model_reason"] = (
        _NO_APERTURE_REASON if normalized["memory_model"] == "unknown" else ""
    )
    return normalized


def gpu_memory_model(gpu: Dict[str, Any]) -> str:
    """`"unified"`, `"discrete"` or `"unknown"`; `""` for an absent GPU.

    **The probe's verdict decides.** `platforms.accelerators.unified_memory_verdict`
    answers this from the part's PCI identity, and its own docstring records why
    bytes cannot: a discrete card reports a GTT aperture too (host memory over
    PCIe, useless for weights) and reports a VRAM total exactly as an APU does.
    So a `gpu` block carrying `unified_memory` is simply believed — that is the
    one home of the question, and both the controller's accelerator facts and
    the SSH worker probe now carry the answer through.

    **Bytes are the fallback, for a record written before that.** A small VRAM
    figure beside an aperture reads as a carve-out. It is a guess, and a known
    imperfect one (see `UNIFIED_VRAM_CARVEOUT_MAX_BYTES`), which is why it never
    overrides a verdict.

    **No aperture at all is `"unknown"`, never `"discrete"`.** With nothing
    reported beside the VRAM there is no evidence for either shape, and calling
    it discrete would dress a missing reading as a finding. The ceiling is then
    the one figure there is; `memory_model_reason` says a refresh is the remedy.

    **And that outranks a `unified` verdict, which is the one case where the
    verdict is not enough on its own.** A part known to be integrated whose
    aperture read as 0 is a KNOWN part with a MISSING reading, not a part with
    no aperture: believing the verdict there sets the ceiling to the aperture,
    which is zero, and a node with a real 0.5 GiB carve-out then previews as
    having no GPU memory whatsoever. So an absent aperture is `"unknown"`
    whoever says the memory is unified, the ceiling falls back to the figure
    that WAS read, and the reason asks for the refresh that would supply it.
    """
    if not gpu.get("present"):
        return ""
    unified = gpu.get("unified_memory")
    gtt = _int(gpu.get("gtt_total_bytes"))
    if unified is True and not gtt:
        return "unknown"
    if isinstance(unified, bool):
        return "unified" if unified else "discrete"
    vram = _int(gpu.get("vram_total_bytes"))
    if not gtt:
        return "unknown"
    return "unified" if vram < UNIFIED_VRAM_CARVEOUT_MAX_BYTES else "discrete"


def addressable_gpu_bytes(gpu: Dict[str, Any]) -> int:
    """Bytes of GPU-addressable memory a model could be placed in.

    A unified/integrated accelerator does **not** report zero VRAM. Strix Halo
    (gfx1151, Radeon 8060S — the Z2 Mini and the ZBook Ultra) reports a VRAM
    carve-out beside a ~30 GiB GTT aperture, and the aperture is where a model
    actually becomes resident. Reading the carve-out as the ceiling is what
    refused every GPU deploy on the hardware this product targets, with a "0.5
    GiB free" preview taken from a figure no model was ever placed in.

    So `gpu_memory_model` decides, and it asks the probe's identity verdict
    rather than the byte figures: a **unified** part addresses its aperture, and
    anything else addresses its VRAM — which on a discrete card is the only
    memory weights can live in, its GTT being host memory over PCIe. A part
    whose model is `"unknown"` gets the one figure that was read, its VRAM;
    nothing is invented to fill the gap.

    The precise weights/KV budget is the sizing phase's job — this is the raw
    addressable ceiling the ledger exposes so that phase, the `/cluster/fit`
    preview and the deploy decision all start from one number.
    """
    if not gpu.get("present"):
        return 0
    if gpu_memory_model(gpu) == "unified":
        return _int(gpu.get("gtt_total_bytes"))
    return _int(gpu.get("vram_total_bytes"))


def free_gpu_bytes(gpu: Dict[str, Any]) -> int:
    """The addressable GPU memory a model could still be placed in.

    The ceiling minus what is already resident, floored at zero. **One
    derivation, called by both sides of the same question** (VD-B3b-1): this
    ledger's per-node `free.gpu_memory_bytes`, which the `/cluster/fit` preview
    reports, and `gpu_pool_operations._gpu_fit_node`, which the deploy sizes
    against. They had it two ways round - the ledger published the raw ceiling
    while the deploy subtracted residency - so a node carrying 22 GiB of GTT
    preview as having room the deploy then refused it.

    Both used bytes are added: on a unified part the aperture and the carve-out
    come out of the same pool, and on a discrete one `gtt_used_bytes` is host
    memory the card is holding open, so neither is free for weights either way.
    """
    if not gpu.get("present"):
        return 0
    used = _int(gpu.get("gtt_used_bytes")) + _int(gpu.get("vram_used_bytes"))
    return max(0, addressable_gpu_bytes(gpu) - used)


def reclaimable_gpu_bytes(gpu: Dict[str, Any]) -> int:
    """The memory stopping Mode A would give this node back (VD-125, D1).

    Zero unless the node's `mode_a_reclaimable` fact says the memory the GPU is
    holding belongs to the AI-Chat model that the cluster mode switch stops
    before vLLM starts. When it does, it is everything currently resident - both
    used figures, for the same reason `free_gpu_bytes` subtracts both: on a
    unified part the aperture and the carve-out come out of one pool.

    **One derivation, called by both sides of the same question**, exactly as
    `free_gpu_bytes` is: `gpu_nodes_from_ledger` reads what this wrote onto the
    ledger for the `/cluster/fit` preview, and
    `gpu_pool_operations._gpu_fit_node` calls it directly for the deploy. Sizing
    them apart is how a preview promises room a deploy then refuses.
    """
    if not gpu.get("present") or not gpu.get("mode_a_reclaimable"):
        return 0
    return _int(gpu.get("gtt_used_bytes")) + _int(gpu.get("vram_used_bytes"))


def gpu_facts_from_accelerators(
    accelerators: Optional[List[Dict[str, Any]]],
    system_ram_bytes: Any = None,
    *,
    mode_a_reclaimable: bool = False,
    mode_a_reclaimable_reason: str = "",
) -> Dict[str, Any]:
    """Build the node GPU shape from a controller `hardware_inventory` list.

    The enrolled-node path reads sysfs over SSH (`ssh_transport._remote_gpu`);
    the controller already has the richer `accelerators` list from
    `vaelor.platforms.accelerators`. This maps that list onto the identical
    shape so a node and the controller describe a GPU the same way in the
    ledger.

    The `unified_memory` verdict is CARRIED, not recomputed and not dropped. It
    is the answer to whether the aperture counts towards a model budget, it is
    already on the record `accelerators.unified_memory_verdict` wrote, and
    dropping it here left the ledger guessing from bytes about the one machine
    that had actually been asked. The PRIMARY card's verdict is the one taken:
    the fields beside it are summed across cards, but "is this memory unified"
    is not a quantity to add up, and a box mixing an APU with a discrete card is
    not a shape the GPU tier places a model on.

    ``mode_a_reclaimable`` is the caller's answer to "is the memory this GPU is
    holding the AI-Chat model the mode switch will stop" (VD-125, D1). Only the
    controller can be asked - it is the machine Mode A serves on - and both
    callers ask through :func:`vaelor.gpu_serving_target.gpu_chat_reclaimable`.
    ``mode_a_reclaimable_reason`` carries the case where it could not be asked,
    so a false answer for lack of a reading is distinguishable from a false one
    for lack of a resident model.
    """
    gpus = [
        item for item in (accelerators or [])
        if isinstance(item, dict) and item.get("kind") == "gpu"
    ]
    if not gpus:
        return _normalize_gpu({"gpu": {
            "present": False,
            "reason": "No GPU compute device was found on this controller.",
            "mode_a_reclaimable_reason": str(mode_a_reclaimable_reason or ""),
        }})

    def total(field: str) -> int:
        return sum(_int(item.get(field)) for item in gpus)

    primary = gpus[0]
    ram = system_ram_bytes if isinstance(system_ram_bytes, int) else None
    unified = primary.get("unified_memory")
    return _normalize_gpu({"gpu": {
        "present": True,
        "reason": "",
        "gfx_target_version": str(primary.get("gfx_target_version", "") or ""),
        "device_count": len(gpus),
        "vram_total_bytes": total("vram_total_bytes"),
        "vram_used_bytes": total("vram_used_bytes"),
        "gtt_total_bytes": total("gtt_total_bytes"),
        "gtt_used_bytes": total("gtt_used_bytes"),
        "system_ram_bytes": ram,
        "unified_memory": unified if isinstance(unified, bool) else None,
        "unified_memory_source": str(
            primary.get("unified_memory_source", "") or ""
        ),
        "mode_a_reclaimable": bool(mode_a_reclaimable),
        "mode_a_reclaimable_reason": str(mode_a_reclaimable_reason or ""),
    }})


def controller_node_facts(
    controller: Dict[str, Any],
    runtime_status: Dict[str, Any],
    hardware: Dict[str, Any],
    *,
    mode_a_reclaimable: bool = False,
    mode_a_reclaimable_reason: str = "",
) -> Dict[str, Any]:
    """Node facts for the head controller, which can itself host workloads.

    Built from the controller's own `hardware_inventory` probe so its GPU is
    described the same way an enrolled node's is (the controller is the Z2
    whose Strix Halo the sizing phase most needs to see).

    ``mode_a_reclaimable`` marks the controller's held GPU memory as memory the
    cluster mode switch will free (VD-125, D1); it is false everywhere else,
    because no other machine serves the AI-Chat model. Its ``_reason`` is set
    only when that question could not be ASKED, and is carried through so the
    fit verdict can say it was sized without that memory (VD-127).
    """
    memory_total = _int(hardware.get("memory_total_bytes"))
    cpu_cores = hardware.get("cpu_cores")
    inventory = {
        "architecture": hardware.get("architecture", "unknown"),
        "cpu_count": int(cpu_cores) if isinstance(cpu_cores, int) else None,
        "cpu_threads": _int(
            hardware.get("cpu_threads") or hardware.get("cpu_cores")
        ),
        "memory_bytes": memory_total,
        "root_free_bytes": _int(hardware.get("storage_free_bytes")),
        "gpu": gpu_facts_from_accelerators(
            hardware.get("accelerators"), memory_total or None,
            mode_a_reclaimable=mode_a_reclaimable,
            mode_a_reclaimable_reason=mode_a_reclaimable_reason,
        ),
    }
    return {
        "node_id": CONTROLLER_PLACEMENT_ID,
        "name": CONTROLLER_PLACEMENT_NAME,
        "role": "head-controller",
        "inventory": inventory,
        # Built only for an active, control-available head, which is exactly
        # when work can be placed on it (ACC-091's placement state).
        "state": "controller",
        "schedulable": True,
        "state_reason": "",
    }


def reservation_from_service_details(details: Dict[str, Any]) -> Dict[str, Any]:
    """The memory a managed Swarm service reserved, and where it lands.

    A pure read of a `DockerSwarmDriver.service_details` dict: the reservation
    is `resources.reservations.MemoryBytes`, the node is parsed from the
    `node.labels.vaelor.node_id==` placement constraint, and the pool label and
    workload come from the `vaelor.*` labels. A service pinned to a node has a
    `node_id`; a pooled/replicated LLM has only a `pool_label`.
    """
    resources = details.get("resources") or {}
    reservations = resources.get("reservations") or {}
    node_id = ""
    for constraint in details.get("constraints", []) or []:
        match = _NODE_ID_CONSTRAINT.fullmatch(str(constraint).strip())
        if match:
            node_id = match.group(1)
            break
    labels = details.get("labels") or {}
    return {
        "name": str(details.get("name", "")),
        "node_id": node_id,
        "pool_label": str(labels.get("vaelor.pool-label", "") or ""),
        "workload": str(labels.get("vaelor.workload", "") or ""),
        "reservation_bytes": _int(reservations.get("MemoryBytes")),
        "replicas": max(1, _int(details.get("desired_replicas")) or 1),
    }


def _node_capacity(inventory: Dict[str, Any]) -> Dict[str, Any]:
    cpu = inventory.get("cpu_count")
    return {
        # Physical cores; None when the node could not read its own topology
        # (#113's honesty rule — threads are never substituted for cores).
        "cpu": int(cpu) if isinstance(cpu, int) else None,
        "cpu_threads": _int(inventory.get("cpu_threads")),
        "memory_bytes": _int(inventory.get("memory_bytes")),
        "gpu": _normalize_gpu(inventory),
    }


#: Why a worker cannot take work right now, per placement state (ACC-091).
_PLACEMENT_REASONS = {
    "not-joined": (
        "This machine is enrolled but has not joined the cluster, so nothing "
        "can be placed on it yet."
    ),
    "unknown": (
        "The cluster's list of machines could not be read, so whether this "
        "machine can take work is unknown."
    ),
    "missing": (
        "The cluster no longer lists this machine; Recheck it, or join it again."
    ),
    "offline": "The cluster cannot reach this machine right now.",
    "drained": "This machine is drained for maintenance, so no new work goes to it.",
    "paused": "This machine is paused, so no new work goes to it.",
}


def worker_placement_state(
    node: Dict[str, Any], status: Dict[str, Any]
) -> Dict[str, Any]:
    """Whether new work can be placed on one enrolled worker, and why not.

    Pure: ``node`` is the cluster-store record and ``status`` the driver's
    ``docker node ls`` report already fetched. The ledger used to count every
    enrolled machine's memory as free - an enrolment that never joined, a
    drained node, one the cluster could not reach - so the fleet headline
    offered memory no deployment could use (ACC-091). Returns ``state``
    (``ready``/``not-joined``/``unknown``/``missing``/``offline``/``drained``/
    ``paused``), ``schedulable`` and a plain ``state_reason``.

    Only a joined node the live list shows Ready and Active is schedulable. A
    list that could not be read is ``unknown``, never ``ready``: the absence of
    a reading is not a healthy machine.
    """
    swarm_id = str((node.get("labels") or {}).get("swarm_node_id", "") or "")
    rows = status.get("nodes") if isinstance(status, dict) else None
    if not swarm_id:
        state = "not-joined"
    elif (
        not isinstance(status, dict)
        or not status.get("available", True)
        or not status.get("control_available")
        or not isinstance(rows, list)
        or not rows
    ):
        # A list that is absent, unparsable or EMPTY is unreadable, not "every
        # machine gone": a manager always lists itself (review nit).
        state = "unknown"
    else:
        live = {
            str(row.get("id", "")): row
            for row in rows
            if isinstance(row, dict)
        }.get(swarm_id)
        availability = str((live or {}).get("availability", "")).lower()
        if live is None:
            state = "missing"
        elif str(live.get("status", "")).lower() != "ready":
            state = "offline"
        elif availability == "drain":
            state = "drained"
        elif availability == "pause":
            state = "paused"
        else:
            state = "ready"
    return {
        "state": state,
        "schedulable": state == "ready",
        "state_reason": _PLACEMENT_REASONS.get(state, ""),
    }


def compute_capacity_ledger(
    node_facts: List[Dict[str, Any]],
    reservations: List[Dict[str, Any]],
) -> Dict[str, Any]:
    """The capacity ledger: per-node capacity/reserved/free, plus a total.

    Pure. `node_facts` are `{node_id, name, role, inventory}` records already
    fetched (from the store and the controller probe); `reservations` are the
    `reservation_from_service_details` records already parsed from Docker. No
    SSH, no Docker, so the arithmetic is unit-testable on its own.
    """
    reserved_memory: Dict[str, int] = {}
    reserved_from: Dict[str, List[str]] = {}
    unattributed: List[Dict[str, Any]] = []
    known_nodes = {str(node.get("node_id", "")) for node in node_facts}

    for record in reservations:
        node_id = str(record.get("node_id", ""))
        name = str(record.get("name", ""))
        reservation = _int(record.get("reservation_bytes"))
        # Swarm's `--reserve-memory` is per TASK, so a service pinned to one node
        # reserves its per-replica figure once for EACH of its replicas on that
        # node. A node-constrained service lands all its replicas there, so the
        # node's committed memory is the reservation times the replica count —
        # counting it once over-credited the node for any multi-replica pin.
        replicas = max(1, _int(record.get("replicas")) or 1)
        if node_id and node_id in known_nodes:
            reserved_memory[node_id] = (
                reserved_memory.get(node_id, 0) + reservation * replicas
            )
            reserved_from.setdefault(node_id, []).append(name)
            continue
        # No single node to attribute to (pooled LLM), or a service pinned to a
        # node this ledger does not list. Phase 1: recorded, never guessed on.
        unattributed.append({
            "name": name,
            "node_id": node_id,
            "pool_label": str(record.get("pool_label", "")),
            "workload": str(record.get("workload", "")),
            "reservation_bytes": reservation,
            "replicas": max(1, _int(record.get("replicas")) or 1),
            "reason": (
                "Pinned to a node not in this ledger."
                if node_id else
                "Placed across a pool; per-shard attribution is a later phase."
                if record.get("pool_label") else
                "Not pinned to a single node; spread across the swarm."
            ),
        })

    nodes: List[Dict[str, Any]] = []
    counted_ids: set = set()
    for fact in node_facts:
        node_id = str(fact.get("node_id", ""))
        # A node listed twice (e.g. the controller that is also enrolled as a
        # worker) is counted once, so its capacity never double-sums into the
        # cluster totals.
        if node_id and node_id in counted_ids:
            continue
        counted_ids.add(node_id)
        inventory = fact.get("inventory") or {}
        capacity = _node_capacity(inventory)
        reserved_bytes = reserved_memory.get(node_id, 0)
        committed_from = sorted(reserved_from.get(node_id, []))
        free_memory = max(0, capacity["memory_bytes"] - reserved_bytes)
        # ACC-091: carried from the caller's placement read. A fact that says
        # nothing (the app-placement path, which filters joined nodes itself)
        # leaves `schedulable` None, and None is counted as before.
        nodes.append({
            "node_id": node_id,
            "name": str(fact.get("name", "")),
            "role": str(fact.get("role", "worker")),
            "state": str(fact.get("state", "") or ""),
            "schedulable": fact.get("schedulable"),
            "state_reason": str(fact.get("state_reason", "") or ""),
            "capacity": capacity,
            "reserved": {
                "memory_bytes": reserved_bytes,
                # Phase 1 stub: Swarm does not reserve GPU memory. See
                # _LEDGER_NOTES["gpu_reservation"].
                "gpu_memory_bytes": 0,
                "from": committed_from,
            },
            "free": {
                "memory_bytes": free_memory,
                # No GPU reservation is tracked yet, so what is unavailable is
                # what is already RESIDENT - the same subtraction the deploy
                # sizes against, through the one helper both call (VD-B3b-1).
                "gpu_memory_bytes": free_gpu_bytes(capacity["gpu"]),
            },
        })

    return {
        "nodes": nodes,
        "cluster": _cluster_total(nodes),
        "unattributed_reservations": unattributed,
        "notes": dict(_LEDGER_NOTES),
    }


def _cluster_total(nodes: List[Dict[str, Any]]) -> Dict[str, Any]:
    cpu = sum(
        node["capacity"]["cpu"] for node in nodes
        if isinstance(node["capacity"]["cpu"], int)
    )
    cpu_threads = sum(node["capacity"]["cpu_threads"] for node in nodes)
    memory = sum(node["capacity"]["memory_bytes"] for node in nodes)
    reserved_memory = sum(node["reserved"]["memory_bytes"] for node in nodes)
    # ACC-091: free memory is memory work can be PLACED in, so a node known not
    # to take work (not joined, drained, paused, offline, unreadable) adds its
    # capacity to the total and nothing to the free figure.
    usable = [node for node in nodes if node.get("schedulable") is not False]
    free_memory = sum(node["free"]["memory_bytes"] for node in usable)
    gpu_nodes = [node for node in nodes if node["capacity"]["gpu"]["present"]]
    vram = sum(node["capacity"]["gpu"]["vram_total_bytes"] for node in gpu_nodes)
    gtt = sum(node["capacity"]["gpu"]["gtt_total_bytes"] for node in gpu_nodes)
    addressable = sum(
        node["capacity"]["gpu"]["addressable_bytes"] for node in gpu_nodes
    )
    # Summed from the nodes rather than re-derived, so the cluster line cannot
    # answer the free-memory question differently from the rows above it.
    free_gpu = sum(node["free"]["gpu_memory_bytes"] for node in nodes)
    committed_from = sorted(
        name for node in nodes for name in node["reserved"]["from"]
    )
    return {
        "node_count": len(nodes),
        "schedulable_node_count": len(usable),
        "schedulable_memory_bytes": sum(
            node["capacity"]["memory_bytes"] for node in usable
        ),
        "capacity": {
            "cpu": cpu,
            "cpu_threads": cpu_threads,
            "memory_bytes": memory,
            "gpu": {
                "present_nodes": len(gpu_nodes),
                "vram_total_bytes": vram,
                "gtt_total_bytes": gtt,
                "addressable_bytes": addressable,
            },
        },
        "reserved": {
            "memory_bytes": reserved_memory,
            "gpu_memory_bytes": 0,
            "from": committed_from,
        },
        "free": {
            "memory_bytes": free_memory,
            "gpu_memory_bytes": free_gpu,
        },
    }


def gpu_fit_node(
    node: Dict[str, Any], *, reclaimable: bool = False,
    reclaimable_reason: str = "",
) -> Dict[str, Any]:
    """Shape a resolved cluster node into the dict `plan_gpu_fit` sizes against.

    The deploy's reading of a node's GPU facts (`gpu_pool_operations`, and the
    replicated path in `gpu_pool_replicas`), housed beside the ledger's
    (`_normalize_gpu`) because the two answer one question - what may a fit
    count on for this node - from the same raw probe fields through the same
    three derivations below, and a second home was where they would have
    drifted (LESSONS pattern 6).

    Free GPU memory is the node's addressable ceiling (VRAM if discrete, else
    the Strix Halo GTT aperture) minus what is already resident. Reading the
    discovered ``inventory.gpu`` facts here keeps the fit decision a pure
    function of measured memory, exactly as `cluster_gpu_sizing` intends.

    Both figures are *derived* from the raw probe fields
    (``vram_total_bytes``/``gtt_total_bytes``), never read as a stored
    ``addressable_bytes`` key: `ssh_transport._remote_gpu` never writes that
    key onto node inventory - it exists only on the normalized capacity
    ledger. And the residency subtraction is `free_gpu_bytes`, the same call
    the ledger's own ``free.gpu_memory_bytes`` makes, so the deploy decision
    and the ``/cluster/fit`` preview cannot report different room (VD-B3b-1).

    ``reclaimable`` says this node's held GPU memory is the Mode A AI-Chat model
    the switch stops before vLLM starts (VD-125, D1), and the bytes are derived
    by `reclaimable_gpu_bytes` - the same call the ledger makes for the
    preview. Only the controller can be reclaimable; it is the machine Mode A
    serves on. ``reclaimable_reason`` is non-empty only when that question
    could not be ASKED, and travels to the fit engine so the verdict says it was
    sized without that memory rather than refusing as if it had measured it.
    """
    inventory = node.get("inventory", {})
    gpu = inventory.get("gpu", {})
    if not gpu.get("present"):
        raise ValueError(
            "{} reports no GPU; refresh it or choose GPU nodes.".format(
                node.get("name", "A node")
            )
        )
    return {
        "node_id": node["id"],
        "name": node.get("name", node["id"]),
        "free_gpu_bytes": free_gpu_bytes(gpu),
        "reclaimable_gpu_bytes": reclaimable_gpu_bytes(
            {**gpu, "mode_a_reclaimable": bool(reclaimable)}
        ),
        "reclaimable_reason": str(reclaimable_reason or ""),
        "addressable_bytes": addressable_gpu_bytes(gpu),
        "vram_total_bytes": int(gpu.get("vram_total_bytes", 0) or 0),
        "gtt_total_bytes": int(gpu.get("gtt_total_bytes", 0) or 0),
        "system_ram_bytes": gpu.get("system_ram_bytes"),
        "gfx_target_version": str(gpu.get("gfx_target_version", "") or ""),
        "device_count": max(1, int(gpu.get("device_count", 1) or 1)),
    }
