"""Gather the §6b Performance snapshot's inputs from the live telemetry.

This is the ONE derivation behind the "why is this slow?" snapshot, so the
Performance tab (``GET /api/v2/cluster/performance``) and the Assistant/MCP
``system.performance`` tool cannot drift: both call ``build_performance_snapshot``
and read the same numbers. The route was the only caller until the tool arrived;
its gathering moved here whole rather than being copied, because a second copy is
exactly how the tab and the agent would come to disagree.

Nothing here is a scrape that does not yet exist. The request RED comes from
every door's own request record (:func:`gather_door_requests`), the per-node USE
from the **same** ``telemetry_history_range``
callback the Fleet trend reads, the serving gauges from the E′ controller scrape,
and the health band from the same ``evaluate_health`` Home answers with. The
assembly, the percentiles, and the one-line diagnosis live in
``performance_snapshot.build_snapshot``; this module only fetches its inputs.

Every callback is read defensively: a missing one (an older wiring with no
``serving_metrics_latest``, a store that is off) degrades to the same honest
"not collected"/"not reporting" state ``build_snapshot`` already renders, never a
crash and never a fabricated zero.
"""

from __future__ import annotations

import sqlite3
import time
from typing import Any, Dict, List, Optional

from .cluster_placement import CONTROLLER_PLACEMENT_ID, CONTROLLER_PLACEMENT_NAME
# The pooled record's state words have one owner, gpu_cluster_mode, a heavy
# module; it is imported for the constants (every control-plane process has it
# loaded already) rather than respelling "healthy"/"unloaded"/"deploying" here.
from .gpu_cluster_mode import DEPLOYING_STATE, HEALTHY_STATE, UNLOADED_STATE
from .gpu_cluster_mode_state import ClusterModeStore
from .gpu_pool_units import VLLM_ENGINE, record_mode
from .gpu_serving_target import (
    UNLOAD_CAUSE_IDLE, UNLOAD_CAUSE_MANUAL, deployment_unload_cause,
    gpu_cluster_mode_active,
)
from .health_evaluation import evaluate_health
from .platforms.base import GENERIC, MACHINE_CLASSES, thermal_policy
from .platforms.gpu_temperature import limits_from_driver
from .llm_gate_usage import (
    GATE_DETAIL_ROWS, USAGE_COUNTING, USAGE_NOT_LOGGING, USAGE_NOT_READING,
)
from .performance_snapshot import (
    CLUSTER_LOADING, CLUSTER_NONE, CLUSTER_SERVING, CLUSTER_UNLOADED,
    DEPRECATED_KEYS, UNLOAD_IDLE, UNLOAD_MANUAL, build_snapshot,
)
from .request_health import (
    AI_CHAT_NOT_TIMED, DOOR_MEASURED, DOOR_NOT_KNOWN, DOOR_NOT_TIMED, DOOR_NOT_WIRED,
    DOOR_OFF, LLM_SERVER_NOT_LOGGING, LLM_SERVER_NOT_READING, LLM_SERVER_OFF,
    LLM_SERVER_UNREADABLE,
)
from .telemetry_ingest_status import worker_is_reporting
from .telemetry_store import MAX_HISTORY_BUCKETS, TelemetryStoreError
from .usage_collection import llm_server_usage_state
from .usage_rollup import SOURCE_AI_CHAT, SOURCE_GATEWAY, SOURCE_LLM_SERVER, SOURCE_UNATTRIBUTED

#: The recent window when the caller sends none, or an unreadable one: 15 minutes
#: is the "what is happening right now" span §6b diffs against the prior period.
DEFAULT_WINDOW = "15m"
DEFAULT_WINDOW_SECONDS = 15 * 60

#: Keep the window inside sane bounds. Below a minute there is no baseline to
#: speak of; a day is the longest the gateway log meaningfully covers for a
#: "why is it slow now" view, and the log is capped at 10k rows regardless.
MIN_WINDOW_SECONDS = 60
MAX_WINDOW_SECONDS = 24 * 60 * 60

#: The stored telemetry field behind each USE signal the snapshot wants from a
#: worker's history — the same field names ``telemetry_history_range`` buckets
#: carry (the aggregate prefix is already stripped by the store).
_WORKER_USE_FIELDS = (
    "cpu_percent", "gpu_busy_percent", "memory_percent", "npu_activity_percent",
    "gpu_gtt_used_bytes", "gpu_gtt_total_bytes", "cpu_temperature", "gpu_temperature_c",
    "gpu_gfx_temperature_c",
)

#: The snapshot's own cause words for the one unload-cause rule's answers
#: (``gpu_serving_target.unload_cause``, VD-136). Anything else, including the
#: rule's "not known", is "" and reads as unloaded with the cause unknown.
_UNLOAD_CAUSE_WORDS = {UNLOAD_CAUSE_IDLE: UNLOAD_IDLE, UNLOAD_CAUSE_MANUAL: UNLOAD_MANUAL}


def clamp_window(seconds: int) -> int:
    """A requested window pinned inside the bounds the snapshot can honour."""
    return max(MIN_WINDOW_SECONDS, min(int(seconds), MAX_WINDOW_SECONDS))


def _use_from_latest(row: Any) -> Dict[str, Any]:
    """The USE fields of a node's newest RAW row (``history_range()["latest"]``).

    The current reading as measured - a bucket is a mean over a window that
    may be only partly elapsed, so the newest bucket is not "now" (ACC-119).
    A field the row did not measure is simply absent, which the snapshot
    renders as unavailable rather than zero.
    """
    source = row if isinstance(row, dict) else {}
    return {
        field: source[field] for field in _WORKER_USE_FIELDS
        if source.get(field) is not None
    }


def _thermal_policy(callbacks: Dict[str, Any]) -> Dict[str, Any]:
    """This machine's thermal bands, or empty when no driver is wired.

    Read through the same platform driver ``/health`` uses, so a node's health
    band here is judged against the machine-class temperatures the rest of the
    product already uses rather than a second set invented for this view.
    """
    try:
        from .api_machine import platform_driver

        return dict(platform_driver(callbacks).thermal_policy() or {})
    except (KeyError, AttributeError, TypeError, OSError, RuntimeError):
        return {}


def worker_nodes(callbacks: Dict[str, Any]) -> List[Dict[str, str]]:
    """Enrolled workers as ``{id, name}``, or empty when the store is unwired.

    The fleet node list the snapshot lists USE for. The controller is added
    separately below; this is only the enrolled members, read from the same
    cluster store ``api_telemetry_history_routes`` validates a ``?node``
    against, so the two cannot disagree about who is in the fleet.
    """
    manager = callbacks.get("cluster_manager")
    if manager is None:
        return []
    try:
        records = manager.store.list_nodes()
    except (AttributeError, OSError, TypeError, ValueError, sqlite3.Error):
        # A locked or unreadable store lists no workers rather than failing the
        # whole snapshot (and with it the Assistant's system.performance tool).
        return []
    workers: List[Dict[str, str]] = []
    for record in records:
        node_id = str(record.get("id", ""))
        if not node_id:
            continue
        inventory = record.get("inventory") if isinstance(record.get("inventory"), dict) else {}
        workers.append({
            "id": node_id, "name": str(record.get("name", "") or node_id),
            # This worker's OWN class and GPU limits, from its enrolment
            # inventory (refreshed by the telemetry repair pass), so its
            # temperatures are judged on its bands and never the controller's.
            "thermal_policy": _class_policy(inventory.get("machine_class")),
            "machine_class_known": _class_known(inventory.get("machine_class")),
            # The driver's own name for the GPU temperature sensor (review S-20).
            "gpu_temperature_label": str(inventory.get("gpu_temperature_label") or ""),
            "gpu_temperature_limits": dict(inventory.get("gpu_temperature_limits") or {}),
        })
    return workers


def _class_policy(machine_class: Any) -> Dict[str, Any]:
    """The thermal bands of a machine class; an unrecorded class has none (review S-19).

    A generic class's numbers would be presented as this machine's: a Pi with
    no class recorded was judged on 97/100 °C. No bands is the honest answer,
    and the screen says why (`gpu_temperature.UNKNOWN_CLASS_NO_BANDS`).
    """
    if not _class_known(machine_class):
        return {}
    return dict(thermal_policy(machine_class))


def _class_known(machine_class: Any) -> bool:
    """Whether a worker's recorded class names a kind of machine.

    ``generic`` is what enrolment records when the machine's identity matched
    no class: it says "unknown", and its bands are not this machine's.
    """
    return machine_class in MACHINE_CLASSES and machine_class != GENERIC


def _controller_gpu_limits(callbacks: Dict[str, Any]) -> Dict[str, Any]:
    """The limits this controller's GPU reports for itself, or empty."""
    try:
        from .api_machine import platform_driver

        return limits_from_driver(platform_driver(callbacks))
    except (KeyError, AttributeError, TypeError, OSError, RuntimeError):
        return {}


def _worker_use(callbacks: Dict[str, Any], node_id: str, seconds: int) -> Dict[str, Any]:
    """A worker's latest USE reading from its telemetry history.

    Reuses ``telemetry_history_range`` rather than probing the worker: the
    newest raw row is the node's current utilisation, and a store that is off,
    failed, or empty yields no metrics, which the snapshot renders as "not
    reporting". So does a worker that
    :func:`vaelor.telemetry_ingest_status.worker_is_reporting` - the one owner
    the Fleet card, the alerts and the reconcile also ask (ACC-122, ACC-126,
    ACC-128) - calls not reporting, including one whose age cannot be read,
    judged on the controller's receive clock first: its last values describe
    a machine that has since gone quiet, and they must neither read as current
    nor feed the latency attribution. The ``last_sample_at`` stays, so the
    screen can say when it was last seen.
    """
    ranged = callbacks.get("telemetry_history_range")
    if ranged is None:
        return {"metrics": None, "last_sample_at": None, "last_sample_age_seconds": None}
    try:
        result = ranged(seconds, MAX_HISTORY_BUCKETS, node_id)
    except TelemetryStoreError:
        return {"metrics": None, "last_sample_at": None, "last_sample_age_seconds": None}
    fresh = worker_is_reporting(node_id, result.get("last_sample_age_seconds"))
    return {
        "metrics": (_use_from_latest(result.get("latest")) or None) if fresh else None,
        "last_sample_at": result.get("last_sample_at"),
        "last_sample_age_seconds": result.get("last_sample_age_seconds"),
    }


#: The pooled record's state, mapped onto the snapshot's cluster words. The
#: first one found wins in this order, so a healthy deployment beside an
#: unloaded one reads as serving.
_RECORD_TO_CLUSTER = (
    (HEALTHY_STATE, CLUSTER_SERVING),
    (DEPLOYING_STATE, CLUSTER_LOADING),
    (UNLOADED_STATE, CLUSTER_UNLOADED),
)


def _unload_cause(callbacks: Dict[str, Any], mode_state: Any, row: Dict[str, Any]) -> str:
    """``idle`` or ``manual`` for the unloaded Mode B record, or ``""`` if not known.

    Not decided here: :func:`vaelor.gpu_serving_target.deployment_unload_cause`
    is the one rule (VD-136) the fleet summary and the LLM Server read too, and
    this only maps its answer onto the snapshot's words. A lease that cannot be
    read is "" - unknown, never guessed manual.
    """
    cause = deployment_unload_cause(
        callbacks.get("credential_broker"), mode_state, str(row.get("name", "")),
    )
    return _UNLOAD_CAUSE_WORDS.get(cause, "")


def _is_mode_b_deployment(mode_state: Any, row: Dict[str, Any]) -> bool:
    """Whether ``row`` is the deployment the appliance's mode record serves.

    The gate :func:`deployment_unload_cause` applies: only that deployment's
    unloaded or loading record may speak for the GPU engine. Any other vLLM
    row (a leftover, a worker-led cluster) must not override a live sample
    from the engine that actually serves - llama.cpp in Mode A.
    """
    return bool(
        gpu_cluster_mode_active(mode_state)
        and str(getattr(mode_state, "deployment_name", "") or "") == str(row.get("name", ""))
    )


def _cluster_state(callbacks: Dict[str, Any], mode_state: Any = None) -> Optional[Dict[str, str]]:
    """The fleet's GPU cluster (vLLM) record as ``{state, cause}``, or None.

    Read from the same cluster store the Fleet screen's pooled-deployment rows
    come from (engine ``vllm``), so the serving card's badge and the backend's
    reason sentence answer from one record: serving (``healthy``), loading
    (``deploying``), unloaded (with the idle-or-manual cause), or none. Loading
    and unloaded count only for the Mode B deployment (see
    :func:`_is_mode_b_deployment`). ``None`` when the store is unwired or
    unreadable - unknown, not "no" - including a locked SQLite store, which
    must not fail the snapshot.
    """
    manager = callbacks.get("cluster_manager")
    if manager is None:
        return None
    try:
        rows = manager.store.list_pooled_deployments()
    except (AttributeError, OSError, TypeError, ValueError, sqlite3.Error):
        return None
    vllm = [
        row for row in rows
        if isinstance(row, dict) and (row.get("units") or {}).get("engine") == VLLM_ENGINE
    ]
    if mode_state is None:
        mode_state = ClusterModeStore().read()  # fails safe to Mode A
    for record_state, cluster_word in _RECORD_TO_CLUSTER:
        row = next((
            row for row in vllm if row.get("state") == record_state
            and (cluster_word == CLUSTER_SERVING or _is_mode_b_deployment(mode_state, row))
        ), None)
        if row is None:
            continue
        cause = _unload_cause(callbacks, mode_state, row) if cluster_word == CLUSTER_UNLOADED else ""
        # Only the deployment the mode record serves says how the serving
        # figures are laid out (review S-22): in Mode A the gauges are this
        # machine's own llama.cpp, whatever vLLM record is left in the store.
        layout = _layout(manager, row) if _is_mode_b_deployment(mode_state, row) else {}
        return {"state": cluster_word, "cause": cause, **layout}
    return {"state": CLUSTER_NONE, "cause": ""}


def _layout(manager: Any, row: Dict[str, Any]) -> Dict[str, Any]:
    """How a deployment is laid out: its placement word, its machines, its lead's name.

    ``placement`` is the record's own word (one full copy per machine, or one
    model split across machines); the lead is the first machine the record
    names, by the name the Fleet screen shows - never its id, and ``""`` when
    the name cannot be read.
    """
    node_ids = [str(item) for item in (row.get("node_ids") or []) if item]
    lead = node_ids[0] if node_ids else ""
    name = ""
    if lead == CONTROLLER_PLACEMENT_ID:
        name = CONTROLLER_PLACEMENT_NAME
    elif lead:
        try:
            records = manager.store.list_nodes()
        except (AttributeError, OSError, TypeError, ValueError, sqlite3.Error):
            records = []
        name = next(
            (str(record.get("name") or "") for record in records if str(record.get("id", "")) == lead), "",
        )
    return {"placement": record_mode(row), "machines": len(node_ids), "lead_name": name}


#: The door each gateway-store source is filed under (the words are the same).
_STORE_DOORS = (SOURCE_GATEWAY, SOURCE_AI_CHAT, SOURCE_UNATTRIBUTED)

#: The exceptions a door's store raises when it cannot be read.
_STORE_ERRORS = (AttributeError, OSError, TypeError, ValueError, sqlite3.Error)


def _llm_server_door(callbacks: Dict[str, Any], now: float) -> Dict[str, Any]:
    """The LLM Server door's state, from the one reading Settings takes too.

    :func:`vaelor.usage_collection.llm_server_usage_state` decides whether the
    gate is recording and the drain reading; this maps its answer onto the
    door words: counting is measured, a switched-off server is off, and
    anything else is not known - said about requests and their times, not
    about key use, which is what that reading's own sentence is for.
    """
    store = callbacks.get("llm_gate_usage")
    if store is None:
        return {"state": DOOR_NOT_KNOWN, "detail": DOOR_NOT_WIRED}
    try:
        usage = llm_server_usage_state(store, callbacks.get("hardware_bridge_client"), now=now)
    except _STORE_ERRORS as error:
        return {"state": DOOR_NOT_KNOWN, "detail": LLM_SERVER_UNREADABLE}
    if not usage.get("enabled"):
        return {"state": DOOR_OFF, "detail": LLM_SERVER_OFF}
    if usage.get("state") == USAGE_COUNTING:
        return {"state": DOOR_MEASURED, "detail": ""}
    return {"state": DOOR_NOT_KNOWN, "detail": {
        USAGE_NOT_LOGGING: LLM_SERVER_NOT_LOGGING, USAGE_NOT_READING: LLM_SERVER_NOT_READING,
    }.get(str(usage.get("state") or ""), LLM_SERVER_UNREADABLE)}


def gather_door_requests(
    callbacks: Dict[str, Any], *, window_seconds: int, now: float, cluster_mode: bool,
) -> Dict[str, Any]:
    """Every door's request rows for the snapshot, each tagged with its door.

    ``recent`` and ``baseline`` are the selected window and the equal one
    before it; ``lifetime`` (and its real span) is read only when no door
    recorded a request in the window, for the card's fallback. ``states``
    says, per door, whether it could be read; ``retained`` from when each
    door's per-request detail is complete. The gateway store holds the
    gateway's and AI Chat's requests under their own source; the LLM Server's
    come from its gate's record. A store that cannot be read makes its doors
    not known - never an empty, "no traffic" door.
    """
    seconds = int(window_seconds)
    spans = {"recent": (now - seconds, now), "baseline": (now - 2 * seconds, now - seconds)}
    rows: Dict[str, List[Dict[str, Any]]] = {"recent": [], "baseline": [], "lifetime": []}
    states: Dict[str, Dict[str, Any]] = {}
    retained: Dict[str, Optional[float]] = {}
    stores = []

    metrics = callbacks.get("inference_metrics")
    if metrics is None:
        for door in (SOURCE_GATEWAY, SOURCE_AI_CHAT):
            states[door] = {"state": DOOR_NOT_KNOWN, "detail": DOOR_NOT_WIRED}
    else:
        def gateway_rows(start: float, end: float) -> List[Dict[str, Any]]:
            found = metrics.rows_between(start, end)
            return [
                {**row, "door": str(row.get("source") or SOURCE_UNATTRIBUTED)}
                for row in found
                if str(row.get("source") or SOURCE_UNATTRIBUTED) in _STORE_DOORS
            ]
        stores.append((gateway_rows, _STORE_DOORS, "The inference gateway's request log"))
        try:
            since = metrics.detail_retained_since()
        except _STORE_ERRORS:
            since = None
        retained.update({door: since for door in _STORE_DOORS})
        if not cluster_mode:
            states[SOURCE_AI_CHAT] = {"state": DOOR_NOT_TIMED, "detail": AI_CHAT_NOT_TIMED}

    gate = callbacks.get("llm_gate_usage")
    states[SOURCE_LLM_SERVER] = _llm_server_door(callbacks, now)
    if gate is not None:
        def gate_rows(start: float, end: float) -> List[Dict[str, Any]]:
            return [{**row, "door": SOURCE_LLM_SERVER} for row in gate.rows_between(start, end)]
        stores.append((gate_rows, (SOURCE_LLM_SERVER,), "The LLM Server's usage record"))
        try:
            retained[SOURCE_LLM_SERVER] = gate.detail_retained_since()
        except _STORE_ERRORS:
            retained[SOURCE_LLM_SERVER] = None

    readable = []
    for read, doors, name in stores:
        try:
            for span, (start, end) in spans.items():
                rows[span].extend(read(start, end))
            readable.append(read)
        except _STORE_ERRORS as error:
            for door in doors:
                states[door] = {"state": DOOR_NOT_KNOWN, "detail": (
                    "{} could not be read, so these requests are not known ({})."
                    .format(name, type(error).__name__)
                )}
    lifetime_seconds = 0.0
    if not rows["recent"]:
        for read in readable:
            try:
                rows["lifetime"].extend(read(0.0, now))
            except _STORE_ERRORS:
                continue
        if rows["lifetime"]:
            oldest = min(float(row.get("created_at", now) or now) for row in rows["lifetime"])
            lifetime_seconds = max(1.0, now - oldest)
    for span in rows.values():
        span.sort(key=lambda row: float(row.get("created_at", 0) or 0))
    return {
        **rows, "lifetime_seconds": lifetime_seconds, "states": states, "retained": retained,
        "kept": {SOURCE_LLM_SERVER: GATE_DETAIL_ROWS},
    }


def performance_callbacks(runtime: Any, callbacks: Dict[str, Any]) -> Dict[str, Any]:
    """The sources the snapshot reads, off the control plane: the ONE list.

    The API blueprint spreads this into its callbacks and the Assistant's
    ``system.performance`` tool builds its snapshot from it, so the tab and the
    agent read the same stores - the tool used to be handed none of the
    request stores and reported every one as not known. ``callbacks`` are the
    host's own (the telemetry and serving-scrape readers).
    """
    from .phoenix_service import trace_collector_status

    usage = getattr(runtime, "usage", None)
    return {
        "current_data": runtime.current_data,
        "telemetry_history_range": callbacks.get("telemetry_history_range"),
        "serving_metrics_latest": callbacks.get("serving_metrics_latest"),
        "serving_metrics_window": callbacks.get("serving_metrics_window"),
        "platform_drivers": runtime.platform_drivers,
        "cluster_manager": runtime.cluster,
        "credential_broker": runtime.credential_broker,
        # The gateway's request log (duration rows for the percentiles, the
        # gateway's and AI Chat's apart by source), the model's own usage
        # ledger and the LLM Server gate's record. Each reader says "not
        # known" when its store cannot be read, never "none".
        "inference_metrics": runtime.inference_metrics,
        "model_usage": usage.ledger if usage is not None else None,
        "llm_gate_usage": usage.gate_store if usage is not None else None,
        "hardware_bridge_client": runtime.hardware_bridge_client,
        "phoenix_status": lambda: trace_collector_status(
            runtime.phoenix_store, runtime.hardware_bridge_client
        ),
    }


def _overall_health(callbacks: Dict[str, Any], data: Dict[str, Any]) -> Dict[str, Any]:
    """The controller's overall health band, or offline when nothing is wired."""
    current = callbacks.get("current_data")
    if current is None and not data:
        return {"status": "offline", "reasons": [], "checked": []}
    return evaluate_health(data, _thermal_policy(callbacks), _controller_gpu_limits(callbacks))


def assistant_performance_snapshot(callbacks: Dict[str, Any]) -> Dict[str, Any]:
    """The snapshot as the Assistant's ``system.performance`` returns it.

    The same snapshot as the tab's, less the deprecated alias keys (the old
    names of ``requests`` and its companions, still served on the API for one
    release): repeating the request block twice costs the small on-device
    model several hundred tokens of context for nothing it can use.
    """
    snapshot = build_performance_snapshot(callbacks)
    for key in (*DEPRECATED_KEYS, "deprecated_keys"):
        snapshot.pop(key, None)
    return snapshot


def build_performance_snapshot(
    callbacks: Dict[str, Any], *, window_seconds: int = DEFAULT_WINDOW_SECONDS
) -> Dict[str, Any]:
    """Gather the live inputs and assemble the typed Performance snapshot.

    The single source both the Performance tab and ``system.performance`` read.
    ``window_seconds`` is clamped here, so any caller — the route with a parsed
    ``?window`` or the tool with its default — lands inside the same bounds. The
    returned dict is exactly what ``build_snapshot`` produces; the route adds its
    own ``requested_window`` stamp on top, which is presentation, not gathering.
    """
    seconds = clamp_window(window_seconds)
    now = time.time()
    # Read once: whether a GPU cluster serves (Mode B) decides whether AI
    # Chat's answers are timed one by one, and the cluster record below.
    mode_state = ClusterModeStore().read()  # fails safe to Mode A

    # Every door's requests: the recent window asked for and the equal baseline
    # immediately before it, so the snapshot can name what *changed* - and,
    # only when no door recorded a request in the window, every recorded
    # request, so the card can fall back to lifetime totals.
    doors = gather_door_requests(
        callbacks, window_seconds=seconds, now=now,
        cluster_mode=gpu_cluster_mode_active(mode_state),
    )

    current_data = callbacks.get("current_data")
    controller_metrics: Optional[Dict[str, Any]] = None
    if current_data is not None:
        try:
            controller_metrics = dict(current_data() or {})
        except (OSError, RuntimeError, TypeError, ValueError):
            controller_metrics = None

    # The newest serving-metrics sample the E′ scrape stored, read through the
    # same lazy callback shape as the telemetry store above so it honours the
    # identical retention states. None when the GPU engine is not serving,
    # nothing has been scraped, or the store is off/unreadable — the snapshot
    # renders that as the honest "not collected" serving state, never a zero.
    serving_latest = callbacks.get("serving_metrics_latest")
    serving: Optional[Dict[str, Any]] = None
    if serving_latest is not None:
        try:
            serving = serving_latest()
        except (OSError, RuntimeError, TypeError, ValueError):
            serving = None

    # The model's own time to first word and writing speed over the window
    # (`generation_health`), which the request card's verdict is judged on.
    # None when unwired or unreadable, which the verdict states as "not known".
    window_reader = callbacks.get("serving_metrics_window")
    serving_window: Optional[Dict[str, Any]] = None
    if window_reader is not None:
        try:
            serving_window = window_reader(seconds)
        except (OSError, RuntimeError, TypeError, ValueError):
            serving_window = None

    # The Phoenix trace-collector status (VD-128): whether per-request tracing is
    # enabled and its collector is up, read through the same lazy-callback shape.
    # None when unwired or unreadable, which the snapshot renders as the honest
    # "tracing is off" state — never a fabricated "on".
    phoenix = callbacks.get("phoenix_status")
    traces: Optional[Dict[str, Any]] = None
    if phoenix is not None:
        try:
            traces = phoenix()
        except (OSError, RuntimeError, TypeError, ValueError):
            traces = None

    # How much each deployment's model did, from the model's own counters
    # (ACC-044/045): every door counted, keyed by the deployment the owner
    # made. None when the ledger is unwired or unreadable, which the snapshot
    # reports as "could not be read" - never as "nothing served".
    ledger = callbacks.get("model_usage")
    deployments = None
    if ledger is not None:
        try:
            deployments = ledger.deployments(now)
        except (OSError, RuntimeError, TypeError, ValueError, sqlite3.Error):
            deployments = None

    nodes: List[Dict[str, Any]] = [{
        "id": CONTROLLER_PLACEMENT_ID,
        "name": CONTROLLER_PLACEMENT_NAME,
        "role": "head-controller",
        "metrics": controller_metrics,
        "last_sample_at": None,
        "last_sample_age_seconds": None,
        "thermal_policy": _thermal_policy(callbacks),
        "gpu_temperature_limits": _controller_gpu_limits(callbacks),
    }]
    for worker in worker_nodes(callbacks):
        reading = _worker_use(callbacks, worker["id"], seconds)
        nodes.append({
            "id": worker["id"],
            "name": worker["name"],
            "role": "worker",
            "thermal_policy": worker["thermal_policy"],
            "gpu_temperature_limits": worker["gpu_temperature_limits"],
            "machine_class_known": worker["machine_class_known"],
            "gpu_temperature_label": worker["gpu_temperature_label"],
            **reading,
        })

    return build_snapshot(
        window_seconds=seconds,
        recent_rows=doors["recent"],
        baseline_rows=doors["baseline"],
        nodes=nodes,
        overall_health=_overall_health(callbacks, controller_metrics or {}),
        thermal_policy=_thermal_policy(callbacks),
        generated_at_ms=int(now * 1000),
        serving=serving,
        serving_window=serving_window,
        traces=traces,
        deployments=deployments,
        detail_retained_since=doors["retained"],
        detail_kept=doors["kept"],
        lifetime_rows=doors["lifetime"],
        lifetime_seconds=doors["lifetime_seconds"],
        now=now,
        cluster=_cluster_state(callbacks, mode_state),
        door_states=doors["states"],
    )
