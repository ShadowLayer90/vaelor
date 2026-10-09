"""The neural processing unit's readings, and slowness read on the right device.

**Review B8.** With ``npu.status`` holding 97% utilisation, 2.15 W and
1,810 MHz, "What is the NPU utilisation?" and "How much power is the NPU
using?" were answered "This machine has a neural processing unit: <name>." -
the presence sentence, with every reading dropped - and because that answer is
reading-backed the model was never asked. "Why is the NPU slow?" and "Why is
inference slow?" read ``gpu.status`` only, so on the Z2 they answered with the
Radeon's numbers about a question that named the NPU.

**Review S8.** At 100% utilisation the slowness answer still said "low
utilisation is what slowness looks like ... work is not reaching the
accelerator". The interpretation now follows the measured figure.

**Review S10.** The suggestion named an "engines panel" that no screen has.

**VD-205 item 5.** A cluster's workers are part of "is the GPU the
bottleneck": their GPU readings (from their telemetry rows,
`assistant_worker_gpu`) are stated beside the controller's.
"""

from __future__ import annotations

from typing import Any, Dict, List, Mapping, Optional

from .answer_evidence import add_evidence
from .assistant_console_places import PERFORMANCE_PLACE
from .assistant_fact_access import usable_fact
from .byte_units import describe_gb
from .phrase_match import mentions

#: Utilisation at or above this is the accelerator working flat out; below
#: the idle bound, work is not reaching it. Shared with the Performance tab's
#: diagnosis so the Assistant and the tab call the same figure the same thing.
from .performance_why import GPU_IDLE_PERCENT, GPU_PEGGED_PERCENT

from .assistant_fault_answers import (  # noqa: E402 - after the shared thresholds
    _ACCELERATOR_DEVICES, CURRENTLY_REPORTS,
)

GPU_WORDS = _ACCELERATOR_DEVICES[0].words + ("gtt",)
NPU_WORDS = _ACCELERATOR_DEVICES[1].words + ("ipu",)
#: Words for a reading of the NPU. Temperature is here so it can be answered
#: as "not read", rather than by the presence sentence.
NPU_READING_WORDS = (
    "utilisation", "utilization", "usage", "busy", "load", "activity",
    "power", "watt", "watts", "wattage", "clock", "mhz", "frequency",
    "temperature", "temp", "hot", "warm", "degrees",
)
_NPU_TEMPERATURE_WORDS = ("temperature", "temp", "hot", "warm", "degrees")
_POWER_WORDS = ("power", "watt", "watts", "wattage")

#: Where to read whether a model server is serving and how fast. A real page.
SERVING_SCREEN_STEP = (
    "Open " + PERFORMANCE_PLACE + ", which shows whether the model server is "
    "serving and how fast it writes."
)


def _number(value: Any) -> Optional[float]:
    return float(value) if isinstance(value, (int, float)) and not isinstance(value, bool) else None


def utilisation_meaning(percent: Optional[float], noun: str) -> str:
    """What one measured utilisation says about slowness - never a cause on its own."""
    if percent is None:
        return (
            "It gave no utilisation reading, so whether work is reaching the "
            "{} is not known.".format(noun)
        )
    if percent >= GPU_PEGGED_PERCENT:
        return (
            "At {:.0f}% the {} is working flat out: the work is reaching it, "
            "and it is the limit right now.".format(percent, noun)
        )
    if percent < GPU_IDLE_PERCENT:
        return (
            "At {:.0f}% the {} is mostly idle. That does not by itself identify "
            "a cause: low utilisation is what slowness looks like from the "
            "outside, not why it happens - it means work "
            "is not reaching the {}, which is a question about the workload "
            "and the model server rather than about the device.".format(
                percent, noun, noun)
        )
    return (
        "At {:.0f}% the {} is partly busy, which shows neither that it is "
        "saturated nor that work is missing it.".format(percent, noun)
    )


def _npu_records(facts: Mapping[str, Any]) -> Optional[List[Mapping[str, Any]]]:
    npu = usable_fact(facts, "npu.status")
    if npu is None or not npu.get("detected"):
        return None
    return [item for item in (npu.get("accelerators") or []) if isinstance(item, Mapping)]


def _npu_line(record: Mapping[str, Any], *, temperature_asked: bool) -> str:
    name = str(record.get("name") or "The neural processing unit")
    stated: List[str] = []
    absent: List[str] = []
    utilisation = _number(record.get("utilisation_percent"))
    if utilisation is not None:
        stated.append("{:.0f}% utilisation".format(utilisation))
    else:
        absent.append("utilisation")
    power = record.get("power") if isinstance(record.get("power"), Mapping) else {}
    watts = _number(power.get("watts"))
    if watts is not None:
        stated.append("{:.2f} W".format(watts))
    else:
        absent.append("power")
    clock = record.get("clock") if isinstance(record.get("clock"), Mapping) else {}
    mhz = _number(clock.get("mhz"))
    if mhz is not None:
        stated.append("{:.0f} MHz".format(mhz))
    else:
        absent.append("clock")
    line = (
        CURRENTLY_REPORTS.format(name, ", ".join(stated)) if stated
        else "{} is present but reported none of its live readings.".format(name)
    )
    if absent and stated:
        line += " It gave no {} reading.".format(" or ".join(absent))
    if temperature_asked:
        line += (
            " Vaelor reads no temperature for the neural processing unit, so "
            "there is no NPU temperature to give."
        )
    return line


def npu_readings_answer(message: str, facts: Mapping[str, Any]) -> Optional[Dict[str, Any]]:
    """The NPU's utilisation, power and clock when a reading of it is asked for."""
    text = str(message or "")
    if not mentions(text, NPU_WORDS) or not mentions(text, NPU_READING_WORDS):
        return None
    records = _npu_records(facts)
    if not records:
        return None
    evidence: List[Dict[str, str]] = []
    lines = []
    for record in records:
        lines.append(_npu_line(record, temperature_asked=mentions(text, _NPU_TEMPERATURE_WORDS)))
        add_evidence(evidence, "npu.status", dict(record), ("utilisation_percent", "name"))
    return {"answer": " ".join(lines), "evidence": evidence,
            "suggested_actions": [], "proposed_job": None}


def worker_gpu_lines(machines: Any, evidence: List[Dict[str, str]], *, asked_power: bool = False) -> List[str]:
    """One sentence per cluster worker's GPU, from its telemetry row; a quiet one says so."""
    lines: List[str] = []
    for machine in machines if isinstance(machines, list) else []:
        if not isinstance(machine, Mapping):
            continue
        name = str(machine.get("name") or "a worker")
        if not machine.get("reporting"):
            reason = str(machine.get("reason") or "")
            if reason and reason not in lines:
                lines.append(reason)
            continue
        from .assistant_fault_answers import _adapter_readings

        stated = []
        busy = _number(machine.get("busy_percent"))
        if busy is not None:
            stated.append("{:.0f}% utilisation".format(busy))
        # The same power and temperature wording as the controller's adapter.
        stated.extend(_adapter_readings({key: machine.get(key) for key in (
            "power_watts", "power_sensor", "temperature_c", "temperature_sensor")}))
        power = _number(machine.get("power_watts"))
        used, total = _number(machine.get("gtt_used_bytes")), _number(machine.get("gtt_total_bytes"))
        if used is not None and total:
            stated.append("{} of {} GPU memory (GTT) in use".format(describe_gb(used, 1), describe_gb(total, 1)))
        add_evidence(evidence, "gpu.status", dict(machine), ("busy_percent", "temperature_c", "power_watts"))
        if stated:
            lines.append(CURRENTLY_REPORTS.format("The GPU on {}".format(name), ", ".join(stated)))
        if power is None and (asked_power or not stated) and machine.get("power_reason"):
            lines.append(str(machine.get("power_reason")))
    return lines


def _gpu_slowness(gpu: Mapping[str, Any], evidence: List[Dict[str, str]], named: bool) -> List[str]:
    from .assistant_fault_answers import _adapter_readings

    if not gpu.get("detected"):
        add_evidence(evidence, "gpu.status", dict(gpu), ("detected", "reason"))
        reason = str(gpu.get("reason") or "").rstrip(".")
        lines = ["This machine reports no compute GPU{}, so nothing here is slow because of one.".format(
            ": {}".format(reason) if reason else "")] if named else []
        return lines + worker_gpu_lines(gpu.get("cluster_machines"), evidence)
    adapters = [item for item in (gpu.get("adapters") or []) if isinstance(item, Mapping)]
    lines: List[str] = []
    for adapter in adapters[:1]:
        stated = _adapter_readings(adapter)
        add_evidence(evidence, "gpu.status", dict(adapter),
                     ("utilisation_percent", "temperature_c", "power_watts", "clock_mhz"))
        name = str(adapter.get("name") or "This machine's GPU")
        if not stated:
            lines.append(
                "{} is present but reported none of the readings that would answer "
                "this, so I have not established a cause and will not name one from "
                "an unread sensor.".format(name))
            continue
        lines.append(CURRENTLY_REPORTS.format(name, ", ".join(stated)))
        lines.append(utilisation_meaning(_number(adapter.get("utilisation_percent")), "GPU"))
    return lines + worker_gpu_lines(gpu.get("cluster_machines"), evidence)


def _npu_slowness(facts: Mapping[str, Any], evidence: List[Dict[str, str]], named: bool) -> List[str]:
    npu = usable_fact(facts, "npu.status")
    if npu is None:
        return []
    if not npu.get("detected"):
        if not named:
            return []
        add_evidence(evidence, "npu.status", dict(npu), ("detected", "reason"))
        reason = str(npu.get("reason") or "").rstrip(".")
        return ["This machine reports no neural processing unit{}, so nothing here is slow because of one.".format(
            ": {}".format(reason) if reason else "")]
    lines = []
    for record in (npu.get("accelerators") or [])[:1]:
        if not isinstance(record, Mapping):
            continue
        add_evidence(evidence, "npu.status", dict(record), ("utilisation_percent", "name"))
        lines.append(_npu_line(record, temperature_asked=False))
        lines.append(utilisation_meaning(_number(record.get("utilisation_percent")), "NPU"))
    return lines


def names_gpu(text: str) -> bool:
    """Whether a sentence names the GPU. One device per function, so no body
    reads two tiers' words (the constant index's two-tier rule)."""
    return mentions(text, GPU_WORDS)


def names_npu(text: str) -> bool:
    """Whether a sentence names the NPU."""
    return mentions(text, NPU_WORDS)


def slowness_answer(message: str, facts: Mapping[str, Any]) -> Optional[Dict[str, Any]]:
    """Answer "why is it slow" from the device the question names, never another's.

    Neither device named ("why is inference slow") reads both. Nothing is
    named as a cause that a reading did not show.
    """
    from .assistant_fault_answers import _ACCELERATOR_TERMS, _SLOW_TERMS

    text = str(message or "").lower()
    if not mentions(text, _SLOW_TERMS) or not mentions(text, _ACCELERATOR_TERMS):
        return None
    gpu_named, npu_named = names_gpu(text), names_npu(text)
    evidence: List[Dict[str, str]] = []
    lines: List[str] = []
    gpu = usable_fact(facts, "gpu.status")
    if gpu is not None and (gpu_named or not npu_named):
        lines.extend(_gpu_slowness(gpu, evidence, named=gpu_named or not npu_named))
    if npu_named or not gpu_named:
        lines.extend(_npu_slowness(facts, evidence, named=npu_named))
    if not lines:
        return None
    if all(line.startswith("This machine reports no") for line in lines):
        lines.append(
            "Slowness on this machine is then a processor, memory, storage or "
            "workload question, and those are the readings to look at.")
    else:
        lines.append(
            "What would settle it: whether the model server is running on the "
            "accelerator at all, how much accelerator memory it holds, and "
            "whether the time goes on reading the prompt or on writing the reply.")
    return {"answer": " ".join(lines), "evidence": evidence,
            "suggested_actions": [SERVING_SCREEN_STEP], "proposed_job": None}


def accelerator_sweep_lines(facts: Mapping[str, Any], evidence: List[Dict[str, str]]) -> List[str]:
    """One line per accelerator for a whole-machine summary (review B4)."""
    from .assistant_fault_answers import _adapter_readings

    lines: List[str] = []
    gpu = usable_fact(facts, "gpu.status")
    if gpu is not None and gpu.get("detected"):
        for adapter in [item for item in gpu.get("adapters") or [] if isinstance(item, Mapping)][:1]:
            stated = _adapter_readings(adapter)
            add_evidence(evidence, "gpu.status", dict(adapter), ("utilisation_percent", "temperature_c"))
            lines.append("GPU: {} {}.".format(
                adapter.get("name") or "the GPU",
                "reports " + ", ".join(stated) if stated else "reported no live readings"))
        lines.extend(worker_gpu_lines(gpu.get("cluster_machines"), evidence))
    for record in _npu_records(facts) or []:
        add_evidence(evidence, "npu.status", dict(record), ("utilisation_percent", "name"))
        lines.append("NPU: " + _npu_line(record, temperature_asked=False))
    return lines
