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

import time
from typing import Any, Callable, Dict, List, Optional

from . import gpu_render_ledger as render_ledger
from .bridge_transport import root_renders_units
from .cluster_placement import name_in_sentence
from .vllm_guesser_build import GUESSERS, NOT_BUILT, GuesserUnavailable, UNAVAILABLE_MARK
from .vllm_guesser_build import ensure_on_worker as ensure_guesser_on_worker
from .gpu_cluster_mode import HEALTHY_STATE
from .gpu_pool_reload import (
    LOAD_CONFIRM, NOT_FOUND, NOT_VLLM, UNLOAD_CONFIRM, confirm_load_bindings,
    drafter_to_fetch, fetch_drafter, load_deployment, load_options, load_participants,
    unload_deployment,
)
from .vllm_images import download_line
from .gpu_pool_refit import gpu_targets_of
from .gpu_pool_units import VLLM_ENGINE, deployment_name, is_replicated
from .ssh_transport import SshTransportError
from .vllm_release_defaults import DRAFTER_DECLINED_FIELD, DRAFTER_NOT_FETCHED, held_back
from .vllm_serve_options import options_for_load

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

#: The job phases a refresh reports under (R104: the update panel names them);
#: `upgrade_progress` reads them back from the job's events.
PHASE_CHECK = "refresh-check"
PHASE_FETCH = "guesser-fetch"
PHASE_BUILD = "guesser-build"
PHASE_RELOAD = "model-reload"
REFRESH_PHASES = (PHASE_CHECK, PHASE_FETCH, PHASE_BUILD, PHASE_RELOAD)

#: N94 (VD-217): what building the 4-bit guesser before the stop may take, all
#: machines together. Measured 21-31 s a build on the Z2 (kb N65); a build not
#: ready by then leaves the Load serving the full-size guesser, said on the row.
GUESSER_PREBUILD_SECONDS = 240
#: How often the controller's background build is asked about.
GUESSER_POLL_SECONDS = 3
#: The progress line while it builds; and when it was not ready before the stop.
GUESSER_BUILDING = "Building the 4-bit guesser on {} while the model still serves"
#: VD-133 1.5.5 #10: the same wait at a deploy, where nothing serves yet.
GUESSER_BUILDING_FIRST = "Building the 4-bit guesser on {} so the model starts with it"
GUESSER_NOT_READY = (
    "The 4-bit guesser was not ready before the reload ({}), so the model serves "
    "the full-size guesser until its next Load"
)
GUESSER_LATE = "it did not finish within {} s"
#: R111: a worker's own build is tried again by the Load in this same refresh
#: (`gpu_pool_runtime` builds it there before the unit), so it is never "until
#: the next Load".
GUESSER_WORKER_RETRY = (
    "The 4-bit guesser could not be built before the stop on {}, so the reload "
    "builds it there again; a machine where that fails too serves the full-size "
    "guesser and its row says why"
)


def _phased(progress: Optional[Callable[..., None]], phase: str) -> Callable[[int, str], None]:
    """A two-argument report that files each line under ``phase`` (R104)."""
    if progress is None:
        return lambda _percent, _message: None
    return lambda percent, message: progress(percent, message, phase)


def refresh_deployment(
    ops: Any, payload: Dict[str, Any],
    progress: Optional[Callable[[int, str], None]] = None,
    placement_status: Optional[Callable[[], Any]] = None,
) -> Dict[str, Any]:
    """Unload and Load one serving vLLM deployment whose units this release renders differently.

    A row that is not serving, or whose units already read as this release
    renders them, is answered without touching anything. ``progress`` is
    called ``(percent, message, phase)``, the phase one of :data:`REFRESH_PHASES`.
    """
    if (payload or {}).get("confirm") != REFRESH_CONFIRM:
        raise ValueError("Confirm the post-upgrade GPU deployment refresh.")
    name = deployment_name(str((payload or {}).get("name", "")))
    deployment = ops.store.get_pooled_deployment(name)
    if deployment is None:
        raise ValueError(NOT_FOUND)
    if (deployment.get("units", {}) or {}).get("engine") != VLLM_ENGINE:
        raise ValueError(NOT_VLLM)
    report = _phased(progress, PHASE_CHECK)
    drift = render_ledger.render_drift(deployment)
    if not drift["refresh"] and str(deployment.get("state", "")) == HEALTHY_STATE:
        # VD-217 (kb G71): the units re-render unchanged, but a Load may now
        # serve this record differently - this release's measured settings for
        # a record made before them - decided on the machines' GPUs as the Load
        # reads them. Not fit-checked here: while it serves, the model's own
        # memory reads as taken, so the fit says no to what the Load after the
        # stop places (measured on the Z2). The Load re-decides with the fit
        # and serves as before, saying why, if the drafter does not fit.
        try:
            would_load = would_load_now(
                deployment, load_participants(ops, deployment, placement_status))
        except ValueError:
            would_load = None
        drift = render_ledger.render_drift(deployment, would_load)
    if not drift["refresh"]:
        return {"name": name, "refreshed": False, "reason": drift["reason"]}

    report(5, "Asking every machine of {} before anything is stopped".format(name))
    try:
        participants = load_participants(ops, deployment, placement_status)
        options = load_options(ops, deployment, participants, drafter=None)[0]
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
    # VD-217: the draft model a record made before it gains is fetched while
    # the model still serves, so the Load after the stop finds it. Decided
    # before the fit, as the refresh itself was: while it serves, the fit
    # reads the model's own memory as taken.
    wanted = would_load_now(deployment, participants)
    fetched, fetch_note = True, None
    try:
        fetch_drafter(ops, deployment, participants, transports, wanted,
                      _phased(progress, PHASE_FETCH))
    except _UNANSWERED as error:
        fetched = False
        # R96: declined on the row, so neither this refresh nor a later one
        # counts it again; the release's other settings still apply, and an
        # owner's Load offers it again.
        _mark_declined(ops, name, drafter_to_fetch(deployment.get("units", {}) or {}, wanted))
        deployment = ops.store.get_pooled_deployment(name) or deployment
        served = would_load_now(deployment, participants)
        why = held_back(wanted, served, DRAFTER_NOT_FETCHED.format(
            reason=str(error)[:200] or type(error).__name__))
        fetch_note = why  # R104: the guesser phase says the decline
        drift = render_ledger.render_drift(deployment, served)
        if not drift["refresh"]:
            return {"name": name, "refreshed": False, "reason": why, "guesser_note": why}
        drift = dict(drift, reason="{} {}".format(drift["reason"], why))
    # N94: the 4-bit guesser is built from the drafter just fetched, while the
    # model still serves, so the Load below uses it at once - never stops it.
    guesser_note = prebuild_guesser(wanted, participants, transports,
                                    _phased(progress, PHASE_BUILD)) if fetched else fetch_note

    # Re-read at the last moment: an idle unload may have landed since.
    current = ops.store.get_pooled_deployment(name) or {}
    if str(current.get("state", "")) != HEALTHY_STATE:
        return {"name": name, "refreshed": False,
                "reason": render_ledger.render_drift(current)["reason"]}
    reload_report = _phased(progress, PHASE_RELOAD)
    reload_report(15, STOPPING.format(name))
    try:
        unload_deployment(ops, {"name": name, "confirm": UNLOAD_CONFIRM})
    except Exception as error:
        # Unload puts the row back to serving when a unit will not stop.
        _write_note(ops, name, {"reason": "not-stopped", "detail": str(error)})
        raise
    try:
        loaded = load_deployment(
            ops, {"name": name, "confirm": LOAD_CONFIRM}, reload_report, placement_status,
            retry_declined=False,
        )
    except Exception as error:
        _write_note(ops, name, {"reason": "load-failed", "detail": str(error)})
        raise
    reason = drift["reason"] if guesser_note is None or not fetched else "{} {}.".format(
        drift["reason"], guesser_note)
    return {
        "name": name,
        "refreshed": True,
        "reason": reason,
        "changed": drift["changed"],
        "state": loaded.get("state"),
        "endpoint": loaded.get("endpoint"),
        # R104: what the model now serves with, for the update panel.
        "serving": serving_summary(ops.store.get_pooled_deployment(name) or {}),
        "guesser_note": guesser_note or "",
    }


def prebuild_guesser(wanted: Any, participants: List[Dict[str, Any]], transports: Dict[str, Any],
                     report: Callable[[int, str], None], *,
                     seconds: float = GUESSER_PREBUILD_SECONDS,
                     clock: Callable[[], float] = time.monotonic,
                     sleep: Callable[[float], None] = time.sleep,
                     building: str = GUESSER_BUILDING, percent: int = 12) -> Optional[str]:
    """Build the 4-bit guesser ``wanted`` serves on every machine before the stop (N94).

    ``None`` when every machine has it (or it serves none), else
    :data:`GUESSER_NOT_READY` naming each machine and why. Never raises: a
    machine without it serves the full-size guesser at the Load, as before,
    and its row says so. The controller's build runs in the background under
    its root bridge (`vllm_guesser_verb.bridge_build_state`), started first so
    it overlaps each worker's, which the controller builds there over SSH as
    its Load would (`vllm_guesser_build.ensure_on_worker`, VD-194).
    """
    row = GUESSERS.get(str(getattr(wanted, "guesser", None) or ""))
    if row is None:
        return None
    deadline = clock() + float(seconds)
    names = {node["id"]: name_in_sentence(str(node.get("name") or node.get("id") or ""))
             for node in participants}
    held: Dict[str, str] = {}
    controllers = [node["id"] for node in participants if root_renders_units(transports[node["id"]])]
    for node_id in controllers:
        report(percent, building.format(names[node_id]))
        held[node_id] = _ask_bridge(transports[node_id], row, True)
    for node in participants:
        if node["id"] in controllers:
            continue
        report(percent, building.format(names[node["id"]]))
        try:
            ensure_guesser_on_worker(transports[node["id"]], row,
                                     seconds=max(1.0, deadline - clock()))
        except (GuesserUnavailable, *_UNANSWERED) as error:
            held[node["id"]] = str(error).replace(UNAVAILABLE_MARK, "")[:300]
    for node_id in controllers:
        while held.get(node_id) == "":
            if clock() >= deadline:
                held[node_id] = GUESSER_LATE.format(int(seconds))
                break
            sleep(GUESSER_POLL_SECONDS)
            held[node_id] = _ask_bridge(transports[node_id], row, False)
    said = []
    for sentence, wanted_controller in ((GUESSER_NOT_READY, True), (GUESSER_WORKER_RETRY, False)):
        reasons = ["{}: {}".format(names[node_id], why) for node_id, why in held.items()
                   if why and (node_id in controllers) == wanted_controller]
        if reasons:
            said.append(sentence.format("; ".join(reasons)))
    return ". ".join(said) or None


#: Where a deploy's progress stands while it waits for the guesser: after
#: every machine's download (35-55) and before the first copy starts (60), so
#: the bar never runs back (adversarial L1).
FIRST_SERVE_BUILD_PERCENT = 57


def _first_serve_row(vllm_options: Any) -> Any:
    """The options a deploy serves with when they carry a 4-bit guesser, else ``None``."""
    from .vllm_serve_options import require_options

    try:
        wanted = require_options(vllm_options)
    except (TypeError, ValueError):
        return None
    return wanted if GUESSERS.get(str(getattr(wanted, "guesser", None) or "")) else None


def begin_first_serve_build(vllm_options: Any, transport: Any) -> None:
    """Start the controller's build the moment its drafter is on disk (adversarial L2).

    Called by the deploy right after the CONTROLLER's download, so the ~25 s
    build overlaps the other machines' downloads instead of following them
    while AI Chat is already down. Best effort: :func:`prebuild_for_first_serve`
    asks again, and a start already running answers "building".
    """
    wanted = _first_serve_row(vllm_options)
    if wanted is None or not root_renders_units(transport):
        return
    _ask_bridge(transport, GUESSERS[wanted.guesser], True)


def prebuild_for_first_serve(vllm_options: Any, participants: List[Dict[str, Any]],
                             transports: Dict[str, Any], report: Callable[[int, str], None],
                             **wait: Any) -> Optional[str]:
    """A deploy's wait for the 4-bit guesser it was decided with, before any unit starts.

    VD-133 1.5.5 #10: the first cold serve started the model while the
    controller's background build ran, so the model served the full-size
    guesser and its row promised "the next Load" - a restart nobody had asked
    for. The deploy fetched the drafter the build reads moments before, so the
    same bounded wait the update's refresh makes (:func:`prebuild_guesser`,
    N94) is made here and the model starts with it. A build not ready in time
    is no failure: the unit install says so on the row, as before.

    The CONTROLLER only (adversarial L2): a worker's unit install builds there
    before its unit starts, with the whole build bound, so building it here
    too would only add a build killed at this wait's deadline and redone.
    """
    wanted = _first_serve_row(vllm_options)
    if wanted is None:
        return None
    controllers = [node for node in participants
                   if root_renders_units(transports.get(node["id"]))]
    return prebuild_guesser(wanted, controllers, transports, report,
                            building=GUESSER_BUILDING_FIRST,
                            percent=FIRST_SERVE_BUILD_PERCENT, **wait)


def _ask_bridge(transport: Any, row: Any, start: bool) -> Optional[str]:
    """``None`` built, ``""`` still building, else why not - the controller's bridge's answer."""
    try:
        answer = transport.guesser_build(row.image, start)
    except _UNANSWERED as error:
        return "its hardware bridge did not answer ({})".format(str(error)[:160])
    state = str((answer or {}).get("state", ""))
    if state == "ready":
        return None
    if state in ("building", "started"):
        return ""
    return str((answer or {}).get("reason") or NOT_BUILT)[:300]


def serving_summary(record: Dict[str, Any]) -> Dict[str, Any]:
    """What a reloaded deployment serves with, and each machine's note (R104)."""
    from .gpu_pool_replicas import serving_notes
    from .vllm_serve_options import options_from_record

    units = record.get("units", {}) or {}
    try:
        options = options_from_record(
            units, str(units.get("repo", "") or record.get("model_id", "") or ""),
            replicated=is_replicated(record))
    except (ValueError, TypeError):
        return {"read": False, "notes": serving_notes({"units": units})}
    return {
        "read": True,
        "drafter": options.advanced[2] is not None,
        "guesser": "4-bit" if options.guesser else ("full-size" if options.advanced[2] else "none"),
        "expert_kernels": "vaelor" if options.moe_kernel else "vllm",
        "step_merging": bool(options.qk_norm_rope_fusion),
        "notes": serving_notes({"units": units}),
    }


def would_load_now(deployment: Dict[str, Any], participants: List[Dict[str, Any]]) -> Any:
    """The options a Load would serve ``deployment`` with on ``participants``, before the fit (VD-217)."""
    units = deployment.get("units", {}) or {}
    return options_for_load(
        units, str(units.get("repo", "") or deployment.get("model_id", "") or ""),
        replicated=is_replicated(deployment), facts=None,
        gpu_targets=gpu_targets_of(participants))[0]


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


def _mark_declined(ops: Any, name: str, key: Optional[str]) -> None:
    """Record on the row that this release's draft model was declined (R96), changing nothing else."""
    if key is None:
        return
    with ops._serving_lock():
        current = ops.store.get_pooled_deployment(name)
        if current is None:
            return
        units = dict(current.get("units", {}) or {})
        units[DRAFTER_DECLINED_FIELD] = {"key": key}
        ops.store.put_pooled_deployment(
            name=name, state=str(current.get("state", "")),
            model_id=str(current.get("model_id", "")),
            node_ids=list(current.get("node_ids") or []), units=units,
            endpoint=str(current.get("endpoint", "") or ""),
            credential_id=str(current.get("credential_id", "") or ""),
        )


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
