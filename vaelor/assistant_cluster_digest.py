"""The cluster, in plain words, for the Assistant (VD-205 item 1).

**Baseline (VD-205, mapped 2026-10-05).** The Assistant's ``cluster.summary``
and ``system.performance`` tools existed, no question gathered them, and the
fact filter would have dropped them; worker data reached an answer only as GPU
power and temperature. This is the one compact reading a cluster, fleet,
worker or named-machine question gathers.

It reuses the owners every other surface reads rather than deriving a second
answer (LESSONS 6):

* the Performance snapshot (`performance_snapshot_source`): each machine's
  USE readings, its health band judged on its own class's thresholds, the
  serving gauges and `performance_why`'s one-line diagnosis;
* each worker's newest raw telemetry row, through ``telemetry_history_range``
  and the one freshness verdict (`telemetry_ingest_status.worker_is_reporting`),
  for what the snapshot does not carry (disk, network, fans, GPU power);
* the cluster store: names, the pooled deployments and their states, and each
  worker's software state (`cluster_worker_profile.worker_software_safe`);
* the serving-mode record (`gpu_cluster_mode_state.ClusterModeStore`).

**A summary string leads**, so the reading survives the context budget's
compaction (the same reason ``system.telemetry`` leads with one). No internal
id appears in any sentence: machines are named as the Fleet screen names them.
A reading that was not taken is absent from the machine's record and said as
"not read" in the summary - never written as a zero (LESSONS 4, 8).
"""

from __future__ import annotations

import logging
import sqlite3
from typing import Any, Callable, Dict, List, Mapping, Optional

from .answer_evidence import FIELD_NAMES
from .byte_units import describe_gb
from .cluster_placement import CONTROLLER_PLACEMENT_ID, CONTROLLER_PLACEMENT_NAME

LOGGER = logging.getLogger(__name__)

#: How far back a worker's newest row is looked for; freshness is judged by
#: the reporting verdict, not by this window.
LATEST_WINDOW_SECONDS = 600

#: Why a part of the digest is missing, said as the controller's own failure.
SNAPSHOT_UNREAD = "This controller could not assemble its performance reading, so per-machine use is not known."
STORE_UNREAD = "This controller could not read its cluster record, so the cluster's machines are not known."
STORE_NOT_WIRED = (
    "This control plane has no cluster record wired in, so it cannot list the "
    "cluster's machines. That is a wiring fault, not a statement about the cluster."
)
NO_CLUSTER = "No workers are enrolled, so there is no cluster: this controller is the only machine."

#: Two sentences the cluster answers say too (`assistant_cluster_answers`).
TIME_TO_FIRST_WORD = "Time to first word: {:.2f} s."
ACTIVE_ALERTS = "Active alerts: {}."

_STORE_ERRORS = (AttributeError, OSError, TypeError, ValueError, sqlite3.Error)

#: The raw row fields the digest carries per machine, and the record key each
#: lands under. Absent in the row means absent in the record.
_ROW_FIELDS = (
    ("cpu_percent", "cpu_percent"), ("memory_percent", "memory_percent"),
    ("gpu_busy_percent", "gpu_busy_percent"), ("gpu_gtt_used_bytes", "gtt_used_bytes"),
    ("gpu_gtt_total_bytes", "gtt_total_bytes"), ("cpu_temperature", "cpu_temperature_c"),
    ("gpu_power_watts", "gpu_power_watts"),
    ("disk_root_free_bytes", "disk_free_bytes"), ("disk_root_total_bytes", "disk_total_bytes"),
    ("net_rx_bytes_per_second", "net_rx_bytes_per_second"),
    ("net_tx_bytes_per_second", "net_tx_bytes_per_second"),
    ("fan_rpm", "fan_rpm"),
)


def _number(value: Any) -> Optional[float]:
    return float(value) if isinstance(value, (int, float)) and not isinstance(value, bool) else None


def _call(callback: Optional[Callable[[], Any]], what: str) -> Any:
    if callback is None:
        return None
    try:
        return callback()
    except (OSError, RuntimeError, TypeError, ValueError, sqlite3.Error):
        LOGGER.warning("the cluster digest could not read %s", what, exc_info=True)
        return None


def serving_mode_state(callbacks: Mapping[str, Any]) -> Any:
    """The serving-mode record (`ClusterModeStore`), or ``None`` when it could not be read.

    The one read the digest and the AI Chat / LLM Server answer
    (`assistant_serving_answers`) share, so they cannot name different modes.
    """
    reader = callbacks.get("cluster_mode_state")
    try:
        if reader is not None:
            return reader()
        from .gpu_cluster_mode_state import ClusterModeStore

        return ClusterModeStore().read()
    except (OSError, RuntimeError, TypeError, ValueError):
        LOGGER.warning("the Assistant could not read the serving mode", exc_info=True)
        return None


def _mode(callbacks: Mapping[str, Any]) -> Optional[bool]:
    """``True`` in Mode B (the cluster serves AI Chat), ``False`` in Mode A, ``None`` unread."""
    from .gpu_serving_target import gpu_cluster_mode_active

    state = serving_mode_state(callbacks)
    return None if state is None else bool(gpu_cluster_mode_active(state))


def _worker_row(callbacks: Mapping[str, Any], node_id: str) -> Dict[str, Any]:
    """The worker's newest raw row and whether it is current, from the one freshness owner."""
    from .telemetry_ingest_status import worker_is_reporting
    from .telemetry_store import MAX_HISTORY_BUCKETS, TelemetryStoreError

    ranged = callbacks.get("telemetry_history_range")
    if ranged is None:
        return {"row": None, "reporting": None}
    try:
        result = ranged(LATEST_WINDOW_SECONDS, MAX_HISTORY_BUCKETS, node_id) or {}
    except (TelemetryStoreError, OSError, TypeError, ValueError):
        return {"row": None, "reporting": None}
    latest = result.get("latest") if isinstance(result.get("latest"), Mapping) else None
    reporting = latest is not None and worker_is_reporting(node_id, result.get("last_sample_age_seconds"))
    return {"row": dict(latest) if reporting else None, "reporting": reporting}


def _readings(row: Optional[Mapping[str, Any]]) -> Dict[str, float]:
    found: Dict[str, float] = {}
    for field, key in _ROW_FIELDS:
        value = _number((row or {}).get(field))
        if value is not None:
            found[key] = value
    return found


def _serves(deployments: List[Mapping[str, Any]], node_id: str, names: Mapping[str, str]) -> List[str]:
    lines = []
    for row in deployments:
        nodes = [str(item) for item in row.get("node_ids") or []]
        if node_id not in nodes:
            continue
        mode = str((row.get("units") or {}).get("mode") or "")
        shape = {"replicated": "a full copy of", "distributed": "a share of"}.get(mode, "")
        lines.append("{} {} ({}, {})".format(
            shape or "", row.get("model_id") or row.get("name") or "a model",
            row.get("name") or "deployment", row.get("state") or "state not known").strip())
    return lines


def _megabytes(rate: float) -> str:
    """A byte rate in decimal megabytes a second (LESSONS 5: the divisor matches the unit)."""
    return "{:.2f}".format(rate / 1_000_000)


#: The readings one machine's line is made of, in the order it says them.
#: A single-reading question names some of these and is answered with those
#: alone (VD-205 live check L3); "how is X doing" gets them all.
READING_PARTS = {
    "cpu": FIELD_NAMES["cpu_percent"], "cpu_model": "processor model name",
    "memory": FIELD_NAMES["memory_percent"], "memory_free": "available memory",
    "gpu": "GPU utilisation", "gpu_memory": "GPU memory",
    "cpu_temperature": FIELD_NAMES["cpu_temperature"], "gpu_temperature": FIELD_NAMES["gpu_temperature_c"],
    "gpu_power": FIELD_NAMES["gpu_power_watts"], "disk": "disk space", "network": "network traffic",
    "fan": FIELD_NAMES["fan_rpm"],
}


def _reading_parts(machine: Mapping[str, Any]) -> List[tuple]:
    """``(part, words)`` for each reading of one reporting machine; an absent one says "not read"."""
    readings = machine.get("readings") or {}
    worker = machine.get("role") == "worker"
    parts = []
    for part, key, label in (("cpu", "cpu_percent", "CPU"), ("memory", "memory_percent", "memory"),
                             ("gpu", "gpu_busy_percent", READING_PARTS["gpu"])):
        if key in readings:
            parts.append((part, "{} {:.0f}%".format(label, readings[key])))
    if "memory_percent" in readings:
        # Review round 2 (LESSONS 5): "how much RAM is free" is the available
        # share, not the used one; a worker reports a share, not an amount.
        parts.append(("memory_free", "memory {:.0f}% available ({:.0f}% in use)".format(
            100.0 - readings["memory_percent"], readings["memory_percent"])))
    if "gtt_used_bytes" in readings and readings.get("gtt_total_bytes"):
        parts.append(("gpu_memory", "GPU memory {} of {}".format(
            describe_gb(readings["gtt_used_bytes"], 1), describe_gb(readings["gtt_total_bytes"], 1))))
    if "cpu_temperature_c" in readings:
        parts.append(("cpu_temperature", "CPU {:.0f} °C".format(readings["cpu_temperature_c"])))
    gpu_temperature = machine.get("gpu_temperature") or {}
    if _number(gpu_temperature.get("value_c")) is not None:
        parts.append(("gpu_temperature", "GPU {:.0f} °C{}".format(
            gpu_temperature["value_c"],
            " ({})".format(gpu_temperature["sensor"]) if gpu_temperature.get("sensor") else "")))
    if "gpu_power_watts" in readings:
        parts.append(("gpu_power", "GPU power {:.1f} W".format(readings["gpu_power_watts"])))
    parts.append(("disk", "disk {} free".format(describe_gb(readings["disk_free_bytes"], 1))
                  if "disk_free_bytes" in readings else "disk space not read"))
    if "net_rx_bytes_per_second" in readings and "net_tx_bytes_per_second" in readings:
        parts.append(("network", "network {} MB/s in, {} MB/s out".format(
            _megabytes(readings["net_rx_bytes_per_second"]), _megabytes(readings["net_tx_bytes_per_second"]))))
    elif worker:
        parts.append(("network", "network traffic not read"))
    if "fan_rpm" in readings:
        parts.append(("fan", "fan {:.0f} RPM".format(readings["fan_rpm"])))
    elif worker:
        parts.append(("fan", "fan speed not read"))
    return parts


def _quiet_sentence(machine: Mapping[str, Any]) -> str:
    """The line for a machine with no current readings, or ``""`` when it has them."""
    if machine.get("reporting") is False:
        return "{} ({}) is not reporting its telemetry now, so it has no current readings.".format(
            machine["name"], machine["role"])
    if machine.get("reporting") is None:
        return "{} ({}): this controller could not read its telemetry.".format(machine["name"], machine["role"])
    return ""


def reading_sentence(machine: Mapping[str, Any], wanted: tuple) -> str:
    """One machine's named readings only, with its name (VD-205 live check L3).

    A reading this machine did not report is said as not read, never dropped:
    "how much disk is free on X" must not answer with silence.
    """
    quiet = _quiet_sentence(machine)
    if quiet:
        return quiet
    parts = dict(_reading_parts(machine))
    found = [parts.get(part) or "{} not read".format(READING_PARTS[part])
             for part in READING_PARTS if part in wanted]
    return "{}: {}.".format(machine["name"], ", ".join(found))


def _machine_sentence(machine: Mapping[str, Any]) -> str:
    name = machine["name"]
    quiet = _quiet_sentence(machine)
    if quiet:
        return quiet
    parts = [words for part, words in _reading_parts(machine) if part != "memory_free"]
    health = machine.get("health") or {}
    band = str(health.get("status") or "not judged")
    sentence = "{} ({}): health {}{}; {}.".format(
        name, machine["role"], band,
        " - " + "; ".join(health.get("reasons")[:3]) if health.get("reasons") else "",
        ", ".join(parts) if parts else "no readings")
    if machine.get("serves"):
        sentence += " Serves " + "; ".join(machine["serves"]) + "."
    elif machine.get("deployments_read"):
        sentence += " Serves no cluster deployment."
    if machine.get("software"):
        sentence += " Worker software: " + machine["software"]
    return sentence


def cluster_digest(callbacks: Mapping[str, Any]) -> Dict[str, Any]:
    """The cluster digest: a leading summary, then one record per machine."""
    manager = callbacks.get("cluster_manager")
    from .performance_snapshot_source import worker_nodes

    workers = worker_nodes(dict(callbacks)) if manager is not None else []
    from .performance_snapshot_source import assistant_performance_snapshot

    snapshot = _call(callbacks.get("performance_snapshot")
                     or (lambda: assistant_performance_snapshot(dict(callbacks))), "the performance snapshot")
    snapshot = snapshot if isinstance(snapshot, Mapping) else None
    try:
        deployments = [row for row in manager.store.list_pooled_deployments()
                       if isinstance(row, Mapping)] if manager is not None else []
        store_read = manager is not None
    except _STORE_ERRORS:
        LOGGER.warning("the cluster digest could not read the pooled deployments", exc_info=True)
        deployments, store_read = [], False
    nodes_use = {str(node.get("id")): node for node in (snapshot or {}).get("nodes") or []
                 if isinstance(node, Mapping)}
    names = {CONTROLLER_PLACEMENT_ID: CONTROLLER_PLACEMENT_NAME,
             **{str(worker["id"]): str(worker["name"]) for worker in workers}}
    current = _call(callbacks.get("current_data"), "this controller's live sample") or {}
    machines: List[Dict[str, Any]] = []
    controller_use = nodes_use.get(CONTROLLER_PLACEMENT_ID) or {}
    machines.append({
        "name": CONTROLLER_PLACEMENT_NAME, "role": "controller", "reporting": True,
        "readings": _readings({**current, **{
            "disk_root_free_bytes": _number(current.get("disk_root_total")) - _number(current.get("disk_root_used"))
            if _number(current.get("disk_root_total")) is not None and _number(current.get("disk_root_used")) is not None
            else None,
            "disk_root_total_bytes": current.get("disk_root_total")}}),
        "gpu_temperature": controller_use.get("gpu_temperature") or {},
        "health": controller_use.get("health") or (snapshot or {}).get("health") or {},
        "serves": _serves(deployments, CONTROLLER_PLACEMENT_ID, names),
        "deployments_read": store_read,
    })
    profiles: Dict[str, Any] = {}
    records: Dict[str, Any] = {}
    try:
        profiles = manager.store.node_profiles() if manager is not None else {}
        # Adversarial review B-1 (LESSONS 13, 14, 6): the worker-software view
        # needs the store's own node record - its inventory carries the
        # architecture - exactly as the Fleet card passes it. `worker_nodes`
        # rows carry no inventory, and a worker that was up to date read
        # "Unknown ... Telegraf" here while the card said "Up to date".
        records = ({str(node.get("id")): node for node in manager.store.list_nodes_tolerant()}
                   if manager is not None else {})
    except _STORE_ERRORS:
        LOGGER.warning("the cluster digest could not read the worker records", exc_info=True)
    for worker in workers:
        node_id = str(worker["id"])
        row = _worker_row(callbacks, node_id)
        use = nodes_use.get(node_id) or {}
        view = manager.worker_software_safe(records.get(node_id, worker), profiles.get(node_id, {}))
        software = str(view.get("label") or "") + (": " + str(view.get("sentence")) if view.get("sentence") else "")
        machines.append({
            "name": names[node_id], "role": "worker", "reporting": row["reporting"],
            "readings": _readings(row["row"]),
            "gpu_temperature": use.get("gpu_temperature") or {},
            "health": use.get("health") or {},
            "serves": _serves(deployments, node_id, names),
            "deployments_read": store_read,
            "software": software,
        })
    mode = _mode(callbacks)
    serving = (snapshot or {}).get("serving") if isinstance((snapshot or {}).get("serving"), Mapping) else {}
    metrics = serving.get("metrics") if isinstance(serving.get("metrics"), Mapping) else {}
    generation = ((snapshot or {}).get("requests") or {}).get("generation") or {}
    decode = generation.get("decode") if isinstance(generation.get("decode"), Mapping) else {}
    ttft = generation.get("ttft") if isinstance(generation.get("ttft"), Mapping) else {}
    speed = {
        # Not "tokens_..." as a key: the tool layer redacts any key holding
        # "token" as a possible secret, and a speed is not one.
        # Two figures, kept apart (adversarial review should-fix 2): the
        # cluster's aggregate output across every request and replica, and
        # one request's writing speed. `a or b` replaced a measured 0.0 with
        # the other figure, which is a different quantity.
        "aggregate_rate": _number(metrics.get("generation_throughput_tokens_per_second")),
        "decode_rate": _number(metrics.get("decode_tokens_per_second")),
        "window_decode_rate": _number(decode.get("tokens_per_second")),
        "ttft_seconds": _number(metrics.get("ttft_seconds")),
        "ttft_p95_band_ms": ttft.get("p95_band_ms"),
        "not_read_reason": "" if serving.get("collected") else str(serving.get("reason") or ""),
        "replicas_read": _number(metrics.get("replicas_read")),
        "replicas_total": _number(metrics.get("replicas_total")),
    }
    why = (snapshot or {}).get("why") if isinstance((snapshot or {}).get("why"), Mapping) else {}
    alerts = [
        "{}: {}".format(machine["name"], reason)
        for machine in machines for reason in (machine.get("health") or {}).get("reasons") or []
    ]
    summary = _summary(machines, mode, speed, why, alerts, snapshot is not None,
                       store_read if manager is not None else None, bool(workers))
    return {
        "summary": summary,
        "mode": {True: "B", False: "A"}.get(mode),
        "machines": machines,
        "serving": speed,
        "why": str(why.get("headline") or ""),
        "alerts": alerts,
    }


def name_in_sentence(name: str) -> str:
    """A machine's name inside a sentence.

    The controller's name is a phrase, not a proper name: "on This Vaelor
    controller" reads "on this Vaelor controller". A worker's name is the
    owner's and is never changed.
    """
    return name[:1].lower() + name[1:] if name == CONTROLLER_PLACEMENT_NAME else name


def speed_sentences(speed: Mapping[str, Any]) -> List[str]:
    """The serving speed, each figure named for what it measures."""
    lines = []
    if speed.get("aggregate_rate") is not None:
        lines.append("The cluster's total output is {:.1f} tokens per second across every request{}.".format(
            speed["aggregate_rate"],
            ", {:.0f} of {:.0f} replicas read".format(speed["replicas_read"], speed["replicas_total"])
            if speed.get("replicas_read") is not None and speed.get("replicas_total") else ""))
    if speed.get("decode_rate") is not None:
        lines.append("One reply is written at {:.1f} tokens per second (the live decode gauge).".format(
            speed["decode_rate"]))
    if speed.get("window_decode_rate") is not None:
        lines.append("Across the last 15 minutes replies averaged {:.1f} tokens per second from "
                     "the model's own timings.".format(speed["window_decode_rate"]))
    return lines


def _summary(machines, mode, speed, why, alerts, snapshot_read, store_read, has_workers) -> str:
    parts: List[str] = []
    if store_read is None:
        parts.append(STORE_NOT_WIRED)
    elif not store_read:
        parts.append(STORE_UNREAD)
    elif not has_workers:
        parts.append(NO_CLUSTER)
    else:
        parts.append("The cluster has {} machines: {}.".format(
            len(machines), ", ".join("{} ({})".format(name_in_sentence(item["name"]), item["role"])
                                     for item in machines)))
    if mode is True:
        parts.append("Mode B: the cluster model serves AI Chat; the controller's own AI Chat model is stopped.")
    elif mode is False and has_workers:
        parts.append("Mode A: each machine serves its own models; AI Chat runs on the controller.")
    if not snapshot_read:
        parts.append(SNAPSHOT_UNREAD)
    parts.extend(_machine_sentence(machine) for machine in machines)
    rates = speed_sentences(speed)
    if rates:
        parts.extend(rates)
    elif speed.get("not_read_reason"):
        parts.append(speed["not_read_reason"])
    if speed.get("ttft_seconds") is not None:
        parts.append(TIME_TO_FIRST_WORD.format(speed["ttft_seconds"]))
    if why.get("headline"):
        parts.append("Diagnosis: {}".format(why["headline"]))
    unjudged = [item["name"] for item in machines if not (item.get("health") or {}).get("checked")]
    if alerts:
        parts.append(ACTIVE_ALERTS.format("; ".join(alerts)))
    elif not unjudged:
        parts.append("No machine has an active alert.")
    if unjudged:
        parts.append("Health was not judged for {}, so no alert there is not the same as clear.".format(
            ", ".join(unjudged)))
    return " ".join(parts)


#: What the tool says it reads, for the tool catalogue.
CLUSTER_DIGEST_DESCRIPTION = (
    "Read the cluster in plain words: each machine's role, whether it is reporting, "
    "its health and current CPU, memory, GPU, GPU-memory, temperature, power, disk "
    "and fan readings, what it serves, the serving mode, serving speed, the "
    "slowness diagnosis, worker software state and active alerts."
)
