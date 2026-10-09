"""Deterministic, evidence-listed answers to the common cluster questions (VD-205 item 4).

How the cluster is doing, which machine is hottest, a named machine's GPU
memory or readings, whether the model runs on every machine, and tokens per
second. The numbers come from the cluster digest (`assistant_cluster_digest`)
- readings this controller holds - and each answer lists its evidence; the
model only explains, it never supplies a figure.

A question about a worker is answered from that worker's record and never from
the controller's: "What is the GPU temperature on the worker?" was answered
with the controller's GPU (the VD-205 baseline). A role that fits several
machines is asked about, not guessed.
"""

from __future__ import annotations

from typing import Any, Dict, List, Mapping, Optional

from .assistant_fact_access import gathered_but_unread, unread, usable_fact
from .assistant_cluster_digest import ACTIVE_ALERTS, TIME_TO_FIRST_WORD, name_in_sentence, speed_sentences
from .assistant_machine_names import (
    EVERY_MACHINE_WORDS, Machine, inventory_names, no_machine_named, resolve_machines, unknown_machine,
)
from .assistant_vocabulary import SPEED_PHRASES, literal_phrases
from .byte_units import describe_gb
from .phrase_match import mentions

#: Words that make a question about the cluster as a whole.
CLUSTER_WORDS = literal_phrases("cluster.digest") + ("machines",)
_HOTTEST_WORDS = ("hottest", "warmest", "hotter")
#: The other end of the comparison (adversarial review should-fix 1: "which
#: machine is coolest" was answered with the hottest).
_COOLEST_WORDS = ("coolest", "coldest", "cooler", "colder")
_COMPARED_TEMPERATURES = "Compared each machine's CPU and GPU temperature from its latest reading."
_GPU_MEMORY_WORDS = ("gpu memory", "gtt", "vram")
_EVERYWHERE_WORDS = EVERY_MACHINE_WORDS + ("everywhere", "on both")
_MODEL_WORDS = ("model", "models", "llm", "serving", "serves", "served")
SPEED_WORDS = SPEED_PHRASES + ("tokens a second", "tokens/s", "tok/s",
                                 "generation speed", "writing speed")
_SLOW_WORDS = ("slow", "slower", "sluggish", "lag", "laggy", "bottleneck")
_ALERT_WORDS = ("alert", "alerts", "warning", "warnings", "alarm", "alarms")
_COUNT_WORDS = ("how many machines", "how many nodes", "how many workers")

_SOURCE = "cluster.digest"

#: A question about the machine as a whole gets its whole line, whatever
#: reading it also names ("how is the ZBook's disk doing" is still a check-up).
_WHOLE_MACHINE_WORDS = ("how is", "how's", "doing", "status", "health", "healthy",
                        "overview", "everything", "reporting")
_TEMPERATURE_WORDS = ("temperature", "temperatures", "temp", "temps", "hot", "warm",
                      "heat", "thermal", "degrees")
_CPU_WORDS = ("cpu", "processor")
_GPU_WORDS = ("gpu", "graphics")
#: The readings a single-reading question names, part by part
#: (`assistant_cluster_digest.READING_PARTS`). VD-205 live check L3: "How much
#: disk space is free on the ZBook?" was answered with the whole machine line.
_PART_WORDS = (
    ("disk", ("disk", "disks", "storage", "space", "drive", "drives")),
    ("network", ("network", "traffic", "bandwidth", "ethernet")),
    ("fan", ("fan", "fans", "rpm")),
    ("memory", ("memory", "ram")),
    ("gpu_power", ("power", "watts", "wattage")),
)


#: "What processor does it have" asks for the model, which no row carries
#: (review round 2): it is said as not read, never answered with CPU use.
_PROCESSOR_MODEL_WORDS = ("what processor", "which processor", "what cpu", "which cpu", "cpu model",
                          "processor model", "kind of processor", "kind of cpu", "type of processor",
                          "type of cpu")
_FREE_WORDS = ("free", "available", "left", "spare")


def readings_wanted(lower: str) -> tuple:
    """The machine readings a question names, or ``()`` for the whole line."""
    if mentions(lower, _WHOLE_MACHINE_WORDS):
        return ()
    if mentions(lower, _PROCESSOR_MODEL_WORDS):
        return ("cpu_model",)
    wanted = [part for part, words in _PART_WORDS if mentions(lower, words)]
    if "memory" in wanted and mentions(lower, _FREE_WORDS):
        wanted[wanted.index("memory")] = "memory_free"
    cpu, gpu = mentions(lower, _CPU_WORDS), mentions(lower, _GPU_WORDS)
    if mentions(lower, _TEMPERATURE_WORDS):
        wanted += [part for part, named in (("cpu_temperature", cpu), ("gpu_temperature", gpu))
                   if named or not (cpu or gpu)]
    else:
        wanted += [part for part, named in (("cpu", cpu), ("gpu", gpu and "gpu_power" not in wanted))
                   if named]
    return tuple(wanted)


def _evidence(summary: str) -> List[Dict[str, str]]:
    return [{"source": _SOURCE, "summary": summary}]


def _reply(lines: List[str], summary: str, answered: bool = True) -> Dict[str, Any]:
    return {"answer": " ".join(line for line in lines if line), "evidence": _evidence(summary),
            "suggested_actions": [], "proposed_job": None, "answered": answered}


def _record(digest: Mapping[str, Any], machine: Machine) -> Optional[Mapping[str, Any]]:
    return next((item for item in digest.get("machines") or []
                 if isinstance(item, Mapping) and item.get("name") == machine.name), None)


def _names_listed(names: List[str]) -> str:
    """Machine names joined at the start of a sentence."""
    return ", ".join(names[:1] + [name_in_sentence(name) for name in names[1:]])


def _hottest(digest: Mapping[str, Any], coolest: bool = False) -> List[str]:
    lines = []
    superlative = "coolest" if coolest else "hottest"
    for key, label in (("cpu_temperature_c", "CPU"), ("gpu", "GPU")):
        best, value = None, None
        for item in digest.get("machines") or []:
            if key == "gpu":
                reading = (item.get("gpu_temperature") or {}).get("value_c")
            else:
                reading = (item.get("readings") or {}).get(key)
            if isinstance(reading, (int, float)) and (
                    value is None or (reading < value if coolest else reading > value)):
                best, value = item, float(reading)
        if best is not None:
            lines.append("The {} {} is on {}, at {:.0f} °C.".format(
                superlative, label, name_in_sentence(best["name"]), value))
        else:
            lines.append("No machine reported a {} temperature.".format(label))
    quiet = [item["name"] for item in digest.get("machines") or [] if item.get("reporting") is not True]
    if quiet:
        lines.append("{} reported no current readings, so {} not in the comparison.".format(
            _names_listed(quiet), "it is" if len(quiet) == 1 else "they are"))
    return lines


def _gpu_memory(record: Mapping[str, Any]) -> str:
    readings = record.get("readings") or {}
    if record.get("reporting") is not True:
        return "{} has no current reading, so its GPU memory is not known now.".format(record["name"])
    used, total = readings.get("gtt_used_bytes"), readings.get("gtt_total_bytes")
    if used is None or not total:
        return "{} did not report its GPU memory (GTT) in its latest reading.".format(record["name"])
    return "{} is using {} of {} GPU memory (GTT, the shared aperture models load into), {:.0f}%.".format(
        record["name"], describe_gb(used, 1), describe_gb(total, 1), 100.0 * used / total)


def _everywhere(digest: Mapping[str, Any]) -> List[str]:
    lines = []
    serving = [item for item in digest.get("machines") or [] if item.get("serves")]
    idle = [item["name"] for item in digest.get("machines") or [] if not item.get("serves")]
    for item in serving:
        lines.append("{} serves {}.".format(item["name"], "; ".join(item["serves"])))
    if not serving:
        lines.append("No cluster deployment is recorded, so no model is spread across the machines.")
    elif idle:
        lines.append("{} {} no cluster deployment, so the model does not run on every machine.".format(
            _names_listed(idle), "serves" if len(idle) == 1 else "serve"))
    else:
        lines.append("Every machine in the cluster carries the deployment.")
    return lines


def _speed(digest: Mapping[str, Any]) -> List[str]:
    speed = digest.get("serving") or {}
    lines = speed_sentences(speed)
    if isinstance(speed.get("ttft_seconds"), (int, float)):
        lines.append(TIME_TO_FIRST_WORD.format(speed["ttft_seconds"]))
    if not lines:
        lines.append(speed.get("not_read_reason") or
                     "No serving speed was read: the model has not been timed in this window.")
    return lines


def workers_named(message: str, context: Any) -> List[str]:
    """The cluster workers ``message`` names, by name (for the power answer)."""
    appliance = (context or {}).get("appliance", context or {}) if isinstance(context, Mapping) else {}
    directory = [Machine(**item) if isinstance(item, Mapping) else item
                 for item in appliance.get("machines") or []]
    if not directory:
        return []
    resolution = resolve_machines(str(message or ""), directory)
    return [machine.name for machine in resolution.machines
            if machine.role == "worker" and not resolution.every]


def cluster_answer(message: str, facts: Mapping[str, Any], appliance: Mapping[str, Any]) -> Optional[Dict[str, Any]]:
    """The deterministic answer to a cluster question, or ``None`` when it is not one."""
    text = str(message or "")
    lower = text.lower()
    directory = [Machine(**item) if isinstance(item, Mapping) else item
                 for item in appliance.get("machines") or []]
    resolution = resolve_machines(text, directory) if directory else None
    workers = [machine for machine in directory if machine.role == "worker"]
    named_worker = bool(resolution and any(machine.role == "worker" for machine in resolution.machines))
    asks_speed = mentions(lower, SPEED_WORDS)
    about_cluster = mentions(lower, CLUSTER_WORDS + _COUNT_WORDS) or named_worker or bool(
        resolution and (resolution.every or resolution.ambiguous) and workers)
    if not (about_cluster or asks_speed):
        return None
    if gathered_but_unread(facts, _SOURCE):
        return _reply([unread("the cluster's machines")], "The cluster reading did not arrive.", answered=False)
    digest = usable_fact(facts, _SOURCE)
    if digest is None:
        return None
    if resolution is not None and resolution.ambiguous:
        return _reply([resolution.ambiguous], "More than one machine fits that description.", answered=False)
    missing = (unknown_machine(text, directory, inventory_names(facts))
               if about_cluster and not named_worker else "")
    if missing:
        return _reply([no_machine_named(missing, directory)],
            "The name in the question matches no enrolled machine.", answered=False)
    if asks_speed:
        lines = _speed(digest)
        named = [machine.name for machine in (resolution.machines if resolution else [])
                 if machine.role == "worker" and not resolution.every]
        if named:
            # Should-fix 2: the speed is read for the cluster's serving, not per
            # machine; a cluster figure is never quoted as one machine's.
            lines.insert(0, "Vaelor reads serving speed for the cluster as a whole, not per "
                            "machine, so there is no figure for {} alone. For the whole "
                            "cluster:".format(" or ".join(named)))
        return _reply(lines, "Read the serving speed from the cluster's live gauges and the model's own timings.")
    if mentions(lower, _COOLEST_WORDS):
        return _reply(_hottest(digest, coolest=True), _COMPARED_TEMPERATURES)
    if mentions(lower, _HOTTEST_WORDS):
        return _reply(_hottest(digest), _COMPARED_TEMPERATURES)
    if mentions(lower, _EVERYWHERE_WORDS) and mentions(lower, _MODEL_WORDS):
        return _reply(_everywhere(digest), "Read which machines each cluster deployment runs on.")
    targets = [machine for machine in (resolution.machines if resolution else []) if not resolution.every] \
        if resolution else []
    if targets and mentions(lower, _GPU_MEMORY_WORDS):
        lines = [_gpu_memory(record) for record in (_record(digest, machine) for machine in targets) if record]
        return _reply(lines, "Read each named machine's GPU memory from its latest telemetry.")
    if targets and any(machine.role == "worker" for machine in targets):
        from .assistant_cluster_digest import _machine_sentence, reading_sentence

        wanted = readings_wanted(lower)
        lines = [reading_sentence(record, wanted) if wanted else _machine_sentence(record)
                 for record in (_record(digest, machine) for machine in targets) if record]
        return _reply(lines, "Read the named machine's own latest telemetry, never another machine's.")
    lead: List[str] = []
    if mentions(lower, _SLOW_WORDS) and digest.get("why"):
        lead.append("Diagnosis: {}".format(digest["why"]))
    if mentions(lower, _ALERT_WORDS):
        alerts = digest.get("alerts") or []
        lead.append(ACTIVE_ALERTS.format("; ".join(alerts)) if alerts
                    else "No machine reports an active alert.")
    return _reply(lead + [str(digest.get("summary") or "")],
                  "Read every machine in the cluster from its own telemetry.")
