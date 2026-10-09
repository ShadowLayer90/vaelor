"""Re-render a serving cluster GPU deployment after an upgrade, through Unload and Load.

W4-D1 (2026-10-01). After an upgrade the installer queues one
``cluster.gpu.refresh`` per vLLM deployment whose recorded units this release
renders differently (`gpu_render_ledger.render_drift`); this is that job.

**It is Unload followed by Load, the owner's own two steps, and nothing else.**
No unit is rendered or started here: `gpu_pool_reload.unload_deployment` stops
what the record derives, and `gpu_pool_reload.load_deployment` re-serves it from
the record through the deploy's own serve bodies - so the replicas, the worker
gates, a split's Ray plane and its start-time slice check all come back exactly
as a Load brings them back, and a split from an older build gains the slice
check (clearing its fence note) the same way. The deployment's machines, link,
addresses, model, port and credential are the record's; a Load never changes
them.

**Nothing is stopped that cannot be started again, as far as can be known
beforehand.** The checks the Load makes before it writes anything
(`load_participants`, `load_options`, `confirm_load_bindings`) are asked of the
serving record first; every machine is asked a real question over its own
transport (its render and video group ids, the read every Load makes); and the
image the Load will run is made present on every machine while the model still
serves, so a re-pinned image downloads before the stop rather than during it.
A machine that does not answer leaves the deployment serving exactly as it
was, with a note on the row naming it - never a half refresh. What only the
serve itself can refuse (a split's firewall or slice on a machine that cannot
hold them) is not knowable here; it ends as a failed Load, said on the row.

**The interruption is a warm Load, said rather than hidden.** Load re-serves
every replica together (it mints the replicas' new internal key, re-keys the
kept credential in place and restarts the balancer), so there is no per-replica
rolling path to reuse: the model does not answer from the stop until the Load
reports healthy, typically a minute or two with the compile cache warm. The
job's progress line and the installer's line both say so.

**A Load that fails after the stop is said on the row.** Load's own rollback
leaves the row ``unloaded``; the note then says the upgrade stopped it and
could not start it again, with the Load's own sentence, and Load is the retry.
"""

from __future__ import annotations

from typing import Any, Callable, Dict, List, Optional

from . import gpu_render_ledger as render_ledger
from .gpu_cluster_mode import HEALTHY_STATE
from .gpu_pool_reload import (
    LOAD_CONFIRM, NOT_FOUND, NOT_VLLM, UNLOAD_CONFIRM, confirm_load_bindings,
    load_deployment, load_options, load_participants, unload_deployment,
)
from .vllm_images import download_line
from .gpu_pool_units import VLLM_ENGINE, deployment_name
from .ssh_transport import SshTransportError

#: The confirm token a refresh job carries; the installer's queue stamps it.
REFRESH_CONFIRM = "refresh-gpu-inference"

#: Said when a machine did not answer, before anything was stopped.
UNREACHABLE = (
    "Could not reach {} to refresh this deployment, so it was left serving as "
    "it was. Unload, then Load it once every machine is back."
)

#: The progress line while the model is down, naming the interruption - and
#: that the executor runs one job at a time, so others queue behind this one.
STOPPING = (
    "Stopping {} to apply this release's serving settings; it does not answer "
    "until the reload below is healthy (a warm Load, usually a minute or two), "
    "and other jobs wait behind this one"
)

#: What a machine that cannot be asked raises, over either transport.
_UNANSWERED = (SshTransportError, RuntimeError, OSError, ValueError)


def refresh_deployment(
    ops: Any, payload: Dict[str, Any],
    progress: Optional[Callable[[int, str], None]] = None,
    placement_status: Optional[Callable[[], Any]] = None,
) -> Dict[str, Any]:
    """Unload and Load one serving vLLM deployment whose units this release renders differently.

    A row that is not serving, or whose units already read as this release
    renders them, is answered without touching anything.
    """
    if (payload or {}).get("confirm") != REFRESH_CONFIRM:
        raise ValueError("Confirm the post-upgrade GPU deployment refresh.")
    name = deployment_name(str((payload or {}).get("name", "")))
    deployment = ops.store.get_pooled_deployment(name)
    if deployment is None:
        raise ValueError(NOT_FOUND)
    if (deployment.get("units", {}) or {}).get("engine") != VLLM_ENGINE:
        raise ValueError(NOT_VLLM)
    report = progress or (lambda _percent, _message: None)
    drift = render_ledger.render_drift(deployment)
    if not drift["refresh"]:
        return {"name": name, "refreshed": False, "reason": drift["reason"]}

    report(5, "Asking every machine of {} before anything is stopped".format(name))
    try:
        participants = load_participants(ops, deployment, placement_status)
        options = load_options(ops, deployment, participants)[0]
    except ValueError as error:
        _write_note(ops, name, {"reason": "refused", "detail": str(error)})
        raise
    transports = {node["id"]: ops._transport(node) for node in participants}
    silent = _unanswering(ops, participants, transports)
    if silent:
        _write_note(ops, name, {"reason": "unreachable", "machines": silent})
        raise RuntimeError(UNREACHABLE.format(", ".join(silent)))
    try:
        confirm_load_bindings(ops, deployment, participants, transports)
        for node in participants:
            # Present already on a warm machine (one inspect); a download, if
            # this release re-pinned the image, happens while it still serves.
            ops.runtime.ensure_image(
                transports[node["id"]], image=options.vllm_image,
                on_download=lambda profile, node=node: report(
                    8, download_line(profile, node["name"])),
            )
    except _UNANSWERED as error:
        _write_note(ops, name, {"reason": "refused", "detail": str(error)})
        raise

    # Re-read at the last moment: an idle unload may have landed since.
    current = ops.store.get_pooled_deployment(name) or {}
    if str(current.get("state", "")) != HEALTHY_STATE:
        return {"name": name, "refreshed": False,
                "reason": render_ledger.render_drift(current)["reason"]}
    report(15, STOPPING.format(name))
    try:
        unload_deployment(ops, {"name": name, "confirm": UNLOAD_CONFIRM})
    except Exception as error:
        # Unload puts the row back to serving when a unit will not stop.
        _write_note(ops, name, {"reason": "not-stopped", "detail": str(error)})
        raise
    try:
        loaded = load_deployment(
            ops, {"name": name, "confirm": LOAD_CONFIRM}, progress, placement_status,
        )
    except Exception as error:
        _write_note(ops, name, {"reason": "load-failed", "detail": str(error)})
        raise
    return {
        "name": name,
        "refreshed": True,
        "reason": drift["reason"],
        "changed": drift["changed"],
        "state": loaded.get("state"),
        "endpoint": loaded.get("endpoint"),
    }


def _unanswering(ops: Any, participants: List[Dict[str, Any]], transports) -> List[str]:
    """The machines that did not answer a real read, by name.

    The read is the group-id read every Load makes of every machine, so a
    machine that answers here is one the Load can reach; asked through the
    machine's own transport (the bridge for this controller, SSH for a worker).
    """
    silent = []
    for node in participants:
        try:
            ops.runtime.resolve_group_ids(transports[node["id"]])
        except _UNANSWERED:
            silent.append(str(node.get("name") or node.get("id") or "a machine"))
    return silent


def _write_note(ops: Any, name: str, note: Dict[str, Any]) -> None:
    """Put the refresh note on the row as it now stands, changing nothing else."""
    with ops._serving_lock():
        current = ops.store.get_pooled_deployment(name)
        if current is None:
            return
        units = dict(current.get("units", {}) or {})
        units[render_ledger.REFRESH_NOTE_FIELD] = note
        ops.store.put_pooled_deployment(
            name=name, state=str(current.get("state", "")),
            model_id=str(current.get("model_id", "")),
            node_ids=list(current.get("node_ids") or []), units=units,
            endpoint=str(current.get("endpoint", "") or ""),
            credential_id=str(current.get("credential_id", "") or ""),
        )
