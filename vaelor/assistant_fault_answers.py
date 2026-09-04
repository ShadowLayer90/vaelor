"""Answer "is anything wrong" and "why is it slow" from readings, or not at all.

Two questions, measured on a Z2 running 2.1.0a19, and one failure mode.

*"What is causing the warning?"* was answered **"The warning is due to low GPU
utilisation."** Home, at that same moment, said *"All systems operational. No
processor, memory, or graphics alerts."* There was no warning. The Assistant
invented one and then explained it - the exact class :mod:`vaelor.answer_evidence`
exists to prevent, arriving one layer up: not a fabricated reading, a fabricated
*fault*.

*"Why is the GPU slow?"* was answered **"The GPU is slow due to low
utilisation."** That is the question restated as its own cause. Low utilisation
is what "slow" looks like from the outside; it is a symptom, and calling it the
cause closes an investigation that has not started.

Both fell through to the model because nothing deterministic covered them:
``deployment_agent`` has branches for cooling, display, lighting, updates,
services, CPU, memory, storage, network, workloads and jobs, and none for the
accelerator or for the health verdict. So this module answers both, and the
rule it follows is the one both failures broke:

**Report what was read. Where nothing read explains the question, say that,
and say what would.** A stated absence of cause is a usable answer; an invented
one sends someone to fix something that is not broken.
"""

from __future__ import annotations

import re
from typing import Any, Dict, List, Mapping, NamedTuple, Optional

from .answer_evidence import add_evidence, describe_missing
from .assistant_answer_topics import ANSWER_TOPICS
from .phrase_match import mentions


#: Sources whose answers are built entirely from readings taken on this
#: machine, and which therefore outrank the model.
#:
#: Without this they did not: ``is_grounded_live_answer`` needs the message to
#: contain one of its own vocabulary terms, which held no word for a fault and
#: no word for the accelerator, so a correct built-in answer was computed and
#: then discarded in favour of asking a small model the same question. That is
#: how "the warning is due to low GPU utilisation" reached a user on a machine
#: whose health verdict, in the same request, said there was no warning.
READING_BACKED_SOURCES = frozenset({
    "built-in-health", "built-in-accelerator",
})


#: A fault word, but only where the sentence is asking about *a* fault on this
#: machine rather than using the word in passing.
#:
#: The determiner is doing the work. "what does **the** error mean" and "are
#: there **any** active warnings" are asking after this appliance's state;
#: "what is a type one error in statistics" is not, and VD-041 already records
#: that widening fault vocabulary risks exactly that swallow. A bare word list
#: would have taken it - which is why this is a shape, not a list.
_FAULT_QUESTION = re.compile(
    r"\b(?:the|any|active|current|currently|outstanding|open|new|unresolved)"
    r"(?:\s+\w+){0,2}\s+"
    r"(?:error|errors|warning|warnings|alert|alerts|alarm|alarms|fault|faults)"
    r"\b",
    re.IGNORECASE,
)

#: The same question without a noun: "is anything wrong", "all clear?".
_GENERAL_FAULT_QUESTION = re.compile(
    r"\b(?:anything|something)\s+wrong\b|\ball\s+(?:clear|good)\b|"
    r"\bneeds?\s+attention\b|\banything\s+i\s+should\s+(?:know|worry)\b",
    re.IGNORECASE,
)


def asks_about_a_fault(message: str) -> bool:
    """Whether this is a question about this machine's fault state."""
    text = str(message or "")
    return bool(
        _FAULT_QUESTION.search(text) or _GENERAL_FAULT_QUESTION.search(text)
    )


def _checked_clause(checked: List[str]) -> str:
    if not checked:
        return ""
    if len(checked) == 1:
        return checked[0]
    return "{} and {}".format(", ".join(checked[:-1]), checked[-1])


def _unchecked_sentence(health: Mapping[str, Any]) -> str:
    """Name what was *not* judged, because silence there reads as "fine"."""
    from .health_evaluation import CATEGORY_LABELS

    checked = {str(item) for item in health.get("checked") or []}
    missing = [
        label for label in CATEGORY_LABELS.values() if label not in checked
    ]
    if not missing:
        return ""
    return (
        " Nothing was read for {} on this sample, so {} not judged either way."
    ).format(
        _checked_clause(missing), "they were" if len(missing) > 1 else "it was"
    )


def health_alert_answer(
    message: str, facts: Mapping[str, Any]
) -> Optional[Dict[str, Any]]:
    """State the fault situation from the same verdict every screen shows.

    Returns ``None`` when the question is not about faults, or when the health
    verdict could not be read - the second is important, because "I have no
    reading" and "there is nothing wrong" are different claims and only one of
    them is safe to make from an empty fact.
    """
    if not asks_about_a_fault(message):
        return None
    health = facts.get("health.status")
    if not isinstance(health, Mapping) or not health.get("status"):
        return None
    status = str(health.get("status"))
    reasons = [str(item) for item in (health.get("reasons") or [])]
    checked = [str(item) for item in (health.get("checked") or [])]
    evidence: List[Dict[str, str]] = []
    add_evidence(evidence, "health.status", dict(health), ("status", "checked"))
    if reasons:
        # A reason is positive evidence of a fault and stands on its own. Only
        # the *negative* claim below needs `checked`, because there the absence
        # of evidence is the entire answer.
        judged = (
            " Vaelor judged {} against this machine's own thresholds.".format(
                _checked_clause(checked)
            ) if checked else ""
        )
        return {
            "answer": (
                "There {} {} active alert{} on this machine: {}.{}{}"
            ).format(
                "is" if len(reasons) == 1 else "are", len(reasons),
                "" if len(reasons) == 1 else "s",
                "; ".join(reasons), judged, _unchecked_sentence(health),
            ),
            "evidence": evidence,
            "suggested_actions": [],
            "proposed_job": None,
        }
    if not checked:
        return {
            "answer": describe_missing(
                "processor, memory or graphics health",
                "No category could be judged from this sample, so whether "
                "anything is wrong is unanswered rather than clear.",
            ),
            "evidence": evidence,
            "suggested_actions": [],
            "proposed_job": None,
        }
    # The measured case. Home said "All systems operational"; the Assistant
    # said a warning existed and explained it. Nothing is the answer here, and
    # it is stated as a result rather than as an absence of information.
    return {
        "answer": (
            "There are no active warnings or alerts on this machine. Vaelor "
            "judged {} against this machine's own thresholds and every one is "
            "inside them, so the overall state is {}.{} If you were told "
            "otherwise somewhere else, that is worth reporting - there is no "
            "alert here for me to explain."
        ).format(
            _checked_clause(checked), status, _unchecked_sentence(health)
        ),
        "evidence": evidence,
        "suggested_actions": [],
        "proposed_job": None,
    }


def overall_verdict_line(facts: Mapping[str, Any], reading: Any = None) -> str:
    """"Overall verdict: ..." for a question about how the machine is doing.

    Lives here rather than in `deployment_agent` because this module already
    holds the rule it depends on: **the health evaluation decides what counts
    as a concern**, against this machine's own thermal policy. Deciding again
    beside it with a different threshold is how the Assistant said "attention
    is needed" while the sidebar, Home and the enclosure page all said healthy
    - LESSONS pattern 6, on one screen. Two neighbours reading one verdict are
    now one neighbourhood.

    ``reading`` is the single CPU temperature the caller has already resolved,
    so this sentence cannot introduce a second sample of the same sensor.
    """
    cooling = facts.get("cooling.status")
    cooling = cooling if isinstance(cooling, Mapping) else {}
    case = cooling.get("case") or {}
    display = facts.get("display.status") or {}
    health = facts.get("health.status")
    concerns: List[str] = []
    if isinstance(health, Mapping) and health.get("reasons"):
        concerns.extend(str(item) for item in health["reasons"][:4])
    elif isinstance(reading, (int, float)) and float(reading) >= 80:
        concerns.append("the CPU is unusually hot")
    # A fan stopped *because the selected cooling profile stops it* is the
    # product working. The tester set "02 Normal start", which holds the fans
    # below ~60 °C, and was told their machine needed attention at 41 °C.
    if case.get("running") is False and not case.get("stopped_by_policy"):
        fan_count = int(case.get("fan_count", 0) or 0)
        concerns.append(
            "the {} case fan{} are stopped".format(
                fan_count, "s" if fan_count != 1 else ""
            ) if fan_count else "the case fans are stopped"
        )
    if display and display.get("detected") and not display.get("enabled"):
        concerns.append("the OLED is detected but disabled")
    return "Overall verdict: {}.".format(
        "attention is needed because " + " and ".join(concerns)
        if concerns else "the reported hardware looks healthy"
    )


#: Words that make a question about the accelerator being slow, rather than
#: about the accelerator generally.
#:
#: **Matched as whole words, and #182 is why.** ``stall`` sits inside
#: "in*stall*ed", so *"what GPU is installed"* was routed here and answered
#: with a paragraph explaining that nothing on this appliance is slow because
#: of a GPU - a question nobody asked, on a machine that has no GPU to be slow.
#: The same substring trap that answered "how do I stop a program" with memory
#: use, one module along, in a list nobody had counted as one of the twelve.
#: Inflections are listed rather than derived, as everywhere else in this tree.
_SLOW_TERMS = (
    "slow", "slower", "slowly", "sluggish", "laggy", "lagging", "lag",
    "stall", "stalls", "stalled", "stalling", "bottleneck", "bottlenecked",
    "throttled", "throttling", "underperform", "underperforming",
    "not using", "why is it taking",
)
_ACCELERATOR_TERMS = (
    "gpu", "gpus", "graphics", "graphics card", "video card", "radeon",
    "vram", "accelerator", "accelerators", "rocm", "vulkan",
    "npu", "npus", "neural", "neural processing unit", "neural accelerator",
    "xdna", "inference",
)
#: The words that ask for a *live reading* of the accelerator rather than for
#: its presence: utilisation, temperature, memory occupancy. A question that
#: carried one of these but no slowness word ("what is my GPU utilisation and
#: temperature right now") fell past `accelerator_slowness_answer` and was
#: caught by `accelerator_presence_answer`, which answers only "this machine
#: has a compute GPU: <name>" and drops the very numbers asked for - they sat
#: in `gpu.status` the whole time. Inflections are listed, not derived.
_READING_TERMS = (
    "utilisation", "utilization", "utilised", "utilized", "usage", "used",
    "using", "busy", "load", "loaded", "temperature", "temp", "temps",
    "thermal", "hot", "warm", "heat", "degrees", "celsius",
    "memory", "gtt", "power", "watt", "watts", "wattage", "clock", "mhz",
)


class _Device(NamedTuple):
    """One accelerator this appliance probes for, and how it is named back.

    Named because the row is read four ways - the fact key, the words that
    make a question about it, where its records sit, and two spellings for
    prose - and a bare tuple of five strings is how ``row[3]`` against
    ``row[4]`` becomes "whether this machine has a compute gpu".
    """

    fact: str
    words: tuple
    records: str
    indefinite: str
    definite: str


_ACCELERATOR_DEVICES = (
    _Device(
        "gpu.status",
        ("gpu", "gpus", "graphics", "graphics card", "video card", "radeon",
         "vram", "rocm", "vulkan"),
        "adapters", "a compute GPU", "compute GPU",
    ),
    _Device(
        "npu.status",
        ("npu", "npus", "neural", "neural processing unit",
         "neural accelerator", "xdna", "tops"),
        "accelerators", "a neural processing unit", "neural processing unit",
    ),
)


#: Live-reading subjects, OTHER than the accelerator, that a compound question
#: can name. **Derived from the answer topics' own vocabularies** rather than
#: re-listed here (LESSONS 6 / VD-090: one idea, one owner - a hand copy would
#: drift and is what `test_duplicate_literals` forbids), minus the accelerator's
#: shared reading *attributes* in :data:`_READING_TERMS` - "temperature",
#: "usage", "load", "memory" and the like. Those attributes belong to a GPU
#: question as much as to any other ("what is my GPU memory usage and
#: temperature"), so excluding them keeps a pure accelerator question from being
#: misread as compound and stripped of its dedicated answer. The GPU/NPU words
#: are not present at all - the accelerator topic carries no vocabulary - so a
#: bare "gpu" never counts itself as another subject.
_READING_ATTRIBUTES = frozenset(term.lower() for term in _READING_TERMS)
_OTHER_COMPONENT_SUBJECTS = tuple(sorted({
    word
    for topic in ANSWER_TOPICS
    for word in topic.words
    if word.lower() not in _READING_ATTRIBUTES
}))


def names_other_component_subject(message: str) -> bool:
    """Whether ``message`` names a live-reading subject other than the
    accelerator, so a GPU/NPU reading is one part of a compound question.

    Used by the deterministic answer path so "What are the CPU temperature,
    memory usage, GPU utilization, and disk usage?" is answered for every
    reading named rather than returning at the GPU branch alone. Matched as
    whole words by :func:`vaelor.phrase_match.mentions`.
    """
    return mentions(str(message or ""), _OTHER_COMPONENT_SUBJECTS)


def accelerator_presence_answer(
    message: str, facts: Mapping[str, Any]
) -> Optional[Dict[str, Any]]:
    """Whether this machine has a GPU or an NPU, from the probe that looked.

    **#183: the reading was taken, delivered, and never read.** Asked *"does
    this machine have a GPU?"* on a Raspberry Pi, the appliance answered *"I
    don't have enough built-in knowledge to answer that reliably without
    guessing"* - while ``gpu.status`` sat in the same request's facts saying
    ``detected: False`` with the driver's own reason, because
    ``select_tools`` gathers it for exactly that question. There was no branch
    anywhere in the built-in path that answered presence: `_fallback_answer`
    had eleven fact branches and none for the accelerator, and
    :func:`accelerator_slowness_answer` only ever fires on a *slowness* word.

    A refusal over a reading in hand is a false denial of a capability this
    product has, which LESSONS pattern 4 records eight times and which costs
    more than the guess it was meant to prevent.

    The reason is the driver's, quoted rather than paraphrased: "no GPU
    character devices are present on this host" corrects itself if the
    hardware changes, and a sentence written here about what Pis have does not.
    """
    if not mentions(str(message or ""), _ACCELERATOR_TERMS):
        return None
    lines: List[str] = []
    evidence: List[Dict[str, str]] = []
    for device in _ACCELERATOR_DEVICES:
        reading = facts.get(device.fact)
        if not isinstance(reading, Mapping):
            continue
        if not mentions(str(message or ""), device.words):
            continue
        add_evidence(
            evidence, device.fact, dict(reading), ("detected", "reason"),
        )
        if not reading.get("detected"):
            reason = str(reading.get("reason") or "").strip().rstrip(".")
            lines.append(
                "This machine reports no {}{}.".format(
                    device.definite, ": {}".format(reason) if reason else "",
                )
            )
            continue
        found = [
            item for item in (reading.get(device.records) or [])
            if isinstance(item, Mapping)
        ]
        named = [str(item.get("name")) for item in found if item.get("name")]
        lines.append(
            "This machine has {}{}.".format(
                device.indefinite,
                ": {}".format(", ".join(named)) if named else "",
            )
        )
    if not lines:
        return None
    return {
        "answer": " ".join(lines),
        "evidence": evidence,
        "suggested_actions": [],
        "proposed_job": None,
    }


def _adapter_readings(adapter: Mapping[str, Any]) -> List[str]:
    stated: List[str] = []
    utilisation = adapter.get("utilisation_percent")
    if isinstance(utilisation, (int, float)) and not isinstance(utilisation, bool):
        stated.append("{:.0f}% utilisation".format(float(utilisation)))
    temperature = adapter.get("temperature_c")
    if isinstance(temperature, (int, float)) and not isinstance(temperature, bool):
        stated.append("{:.0f} °C".format(float(temperature)))
    memory = adapter.get("memory") if isinstance(adapter.get("memory"), Mapping) else {}
    for label, key in (("VRAM", "vram_used_percent"), ("GTT", "gtt_used_percent")):
        value = memory.get(key)
        if isinstance(value, (int, float)) and not isinstance(value, bool):
            stated.append("{} at {:.0f}%".format(label, float(value)))
    return stated


def accelerator_readings_answer(
    message: str, facts: Mapping[str, Any]
) -> Optional[Dict[str, Any]]:
    """The GPU's live readings when asked for a reading, not for presence.

    **The numbers were gathered, then dropped.** *"What is my GPU utilisation
    and temperature right now?"* carries a GPU word but no slowness word, so it
    fell past :func:`accelerator_slowness_answer` and was answered by
    :func:`accelerator_presence_answer` - which states only *"This machine has
    a compute GPU: <name>."* and never reads the utilisation and temperature
    that ``gpu.status`` had already measured. Two differently trained on-device
    models returned that same name-only sentence byte for byte, which is what a
    deterministic branch answering ahead of the model looks like: no retrain
    could touch it because the model never ran.

    This branch sits between slowness and presence. It fires only when a
    *reading* word is present, reuses :func:`_adapter_readings` - the same
    formatter the slowness path trusts - and, where the sensors returned
    nothing, says so rather than falling back to the name. Absence of the GPU
    is left to the presence answer, which quotes the driver's own reason.

    Gated on the GPU device's own words, not on any accelerator word: it reads
    ``gpu.status`` alone, so *"what is my NPU utilisation"* must fall through to
    the device loop in :func:`accelerator_presence_answer` rather than be
    answered here with the GPU's numbers.
    """
    text = str(message or "")
    if not mentions(text, _READING_TERMS):
        return None
    gpu_device = next(
        (device for device in _ACCELERATOR_DEVICES if device.fact == "gpu.status"),
        None,
    )
    if gpu_device is None or not mentions(text, gpu_device.words):
        return None
    gpu = facts.get("gpu.status")
    if not isinstance(gpu, Mapping) or not gpu.get("detected"):
        return None
    adapters = [
        item for item in (gpu.get("adapters") or []) if isinstance(item, Mapping)
    ]
    if not adapters:
        return None
    lines: List[str] = []
    evidence: List[Dict[str, str]] = []
    for adapter in adapters:
        name = str(adapter.get("name") or "The GPU")
        stated = _adapter_readings(adapter)
        add_evidence(
            evidence, "gpu.status", dict(adapter),
            ("utilisation_percent", "temperature_c"),
        )
        if stated:
            lines.append("{} currently reports {}.".format(name, ", ".join(stated)))
        else:
            lines.append(
                "{} is present but reported none of the live readings - no "
                "utilisation, temperature or memory figures were available "
                "from its sensors.".format(name)
            )
    return {
        "answer": " ".join(lines),
        "evidence": evidence,
        "suggested_actions": [],
        "proposed_job": None,
    }


def accelerator_slowness_answer(
    message: str, facts: Mapping[str, Any]
) -> Optional[Dict[str, Any]]:
    """Answer "why is the GPU slow" with readings, never with the question.

    **Low utilisation is never returned as a cause.** It was, verbatim, and it
    is the same fact as "the GPU is slow" wearing a different noun. Where the
    readings establish nothing, this says so and names what would settle it -
    which is a shorter answer than the invented one and the only honest one.
    """
    lower = str(message or "").lower()
    if not mentions(lower, _SLOW_TERMS):
        return None
    if not mentions(lower, _ACCELERATOR_TERMS):
        return None
    gpu = facts.get("gpu.status")
    if not isinstance(gpu, Mapping):
        return None
    evidence: List[Dict[str, str]] = []
    if not gpu.get("detected"):
        add_evidence(evidence, "gpu.status", dict(gpu), ("detected", "reason"))
        return {
            "answer": (
                "This machine reports no compute GPU{}, so nothing here is "
                "slow because of one. Slowness on this appliance is a "
                "processor, memory, storage or workload question, and those "
                "are the readings to look at."
            ).format(
                ": {}".format(str(gpu.get("reason")).rstrip("."))
                if gpu.get("reason") else ""
            ),
            "evidence": evidence,
            "suggested_actions": [],
            "proposed_job": None,
        }
    adapters = [item for item in (gpu.get("adapters") or []) if isinstance(item, Mapping)]
    if not adapters:
        return None
    adapter = adapters[0]
    stated = _adapter_readings(adapter)
    add_evidence(
        evidence, "gpu.status", dict(adapter),
        ("utilisation_percent", "temperature_c", "power_watts", "clock_mhz"),
    )
    name = str(adapter.get("name") or "The GPU")
    if not stated:
        return {
            "answer": (
                "{} is present but reported none of the readings that would "
                "answer this - no utilisation, no temperature, no memory "
                "figures. I have not established a cause, and I will not name "
                "one from an unread sensor."
            ).format(name),
            "evidence": evidence,
            "suggested_actions": [],
            "proposed_job": None,
        }
    return {
        "answer": (
            "{} currently reports {}. That is what was measured, and it does "
            "not by itself identify a cause: low utilisation is what slowness "
            "looks like from the outside, not why it is happening - it means "
            "work is not reaching the accelerator, which is a question about "
            "the workload and the backend rather than about the GPU. What "
            "would settle it: whether the model server is offloading to the "
            "accelerator at all, how much accelerator memory it is holding, "
            "and whether the run is prompt-bound or generation-bound."
        ).format(name, ", ".join(stated)),
        "evidence": evidence,
        "suggested_actions": [
            "Open the local AI engines panel to see whether the model server "
            "is accelerated or has fallen back to the CPU.",
        ],
        "proposed_job": None,
    }
