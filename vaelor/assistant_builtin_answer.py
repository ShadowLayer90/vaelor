"""The built-in answer's per-subject sentences, read from this turn's facts.

Moved out of `DeploymentAgent._fallback_answer`, which had reached its line
ceiling, so the subjects the review found unanswered or misanswered had room:

* **B1** - every fact is read through `assistant_fact_access.usable_fact`. A
  tool that failed is said as not read, never summarised as an empty reading,
  and ``services.status``/``jobs.recent`` are type-checked as lists (a dict
  there was an HTTP 500).
* **B2** - an update check that did not run says so; ``count`` has no default.
* **B3/B4** - the verdict is `overall_verdict_line`'s, read from
  ``health.status``, and a whole-machine question ("how is my machine
  doing", "full health check") gets the verdict plus one line per subsystem
  instead of the capability refusal.
* **B6** - the refusal names Vaelor's own model and basic mode, and sends
  outside models to AI Chat (VD-049, VD-201).
* **S5** - "running hot", "online", "need a reboot", uptime, "still
  installing" with no jobs, and "what hardware is this" are answered from
  readings already in hand.
* **S6** - "which model is loaded" reads ``inference.status``, not the app
  inventory.
* **S7** - "errors in the logs" reads the service journal.
"""

from __future__ import annotations

import re
import time
from datetime import datetime
from typing import Any, Dict, List, Mapping, Optional

from .answer_evidence import add_evidence, describe_missing, first_sentence, present, sentence
from .assistant_accelerator_answers import accelerator_sweep_lines
from .assistant_answer_presentation import managed_workload_summary, network_summary, storage_summary
from .assistant_answer_topics import asks_about, capability_sentence
from .assistant_fact_access import gathered_but_unread, unread, usable_fact
from .assistant_fault_answers import overall_verdict_line
from .assistant_intents import SWEEP_PHRASES
from .assistant_hardware_answers import case_fan_answer, cpu_temperature, display_line, lighting_line
from .assistant_log_answers import logs_line
from .assistant_static_answers import static_answer
from .reported_symptoms import ALL_SERVICES_ACTIVE
from .byte_units import describe_gb

from .phrase_match import mentions

#: The sentence written when nothing on this machine answered. The marker
#: ``enough built-in knowledge`` is read by `provider_runtime` to tell a
#: refusal from an answer, and must survive any rewording.
#:
#: **Review B6 (VD-049, VD-201).** It used to end "Connect a local model,
#: hosted API, or OpenAI-compatible endpoint" - an offer the Assistant cannot
#: take: it runs Vaelor's own model or built-in basic mode, and a model you
#: connect is used by AI Chat.
NO_BUILTIN_ANSWER = (
    "I don’t have enough built-in knowledge to answer that reliably without "
    "guessing. The Assistant runs Vaelor's own model, or built-in basic mode "
    "when that model is not running; models you connect are used in AI Chat, "
    "which is the place for broader questions."
)

#: How the update list is named when it was not read.
_UPDATE_LIST = "the operating-system update list"

#: Whole-machine phrases: the verdict plus a line for every subsystem.
_SWEEP_WORDS = SWEEP_PHRASES

#: "What hardware is this machine?" - a question about the machine itself,
#: answered from the standing brief's ``Machine:`` and processor lines.
_HARDWARE_QUESTION = re.compile(
    r"\bwhat\b[^.?!]{0,20}\b(?:hardware|kind of machine|kind of computer|kind of box|"
    r"type of machine|type of computer|type of box)\b",
    re.IGNORECASE,
)
_BRIEF_LINE = re.compile(r"^(Machine|Processor|Memory):\s*(.+?)\s*$", re.MULTILINE)

#: Job states that mean the job is still going.
_ACTIVE_JOB_STATES = frozenset({"queued", "running", "pending", "waiting", "needs_approval"})

#: Words that keep an inference question about the apps too.
_APP_WORDS = ("app", "apps", "application", "applications", "container", "containers", "installed")


def hardware_line(brief_text: str) -> str:
    """The brief's own description of the machine, or ``""`` when it carries none."""
    found = {key: value.rstrip(".") for key, value in _BRIEF_LINE.findall(str(brief_text or ""))}
    if not found.get("Machine"):
        return ""
    parts = ["This machine is {}".format(found["Machine"])]
    if found.get("Processor"):
        parts.append("its processor is {}".format(found["Processor"]))
    if found.get("Memory"):
        parts.append("memory: {}".format(found["Memory"]))
    return "; ".join(parts) + "."


def _uptime_line(telemetry: Mapping[str, Any], now: Optional[float] = None) -> str:
    boot = telemetry.get("boot_time")
    if not isinstance(boot, (int, float)) or isinstance(boot, bool) or boot <= 0:
        return describe_missing("when this machine last started")
    seconds = max(0.0, (time.time() if now is None else now) - float(boot))
    days, rest = divmod(int(seconds), 86400)
    hours, rest = divmod(rest, 3600)
    minutes = rest // 60
    span = ", ".join(part for part in (
        "{} day{}".format(days, "" if days == 1 else "s") if days else "",
        "{} hour{}".format(hours, "" if hours == 1 else "s") if hours else "",
        "{} minute{}".format(minutes, "" if minutes == 1 else "s") if not days else "",
    ) if part) or "less than a minute"
    started = datetime.fromtimestamp(float(boot)).astimezone().strftime("%Y-%m-%d %H:%M %Z").strip()
    return "This machine has been up for {}; it last started at {} (this machine's local time).".format(
        span, started)


def _updates_line(updates: Mapping[str, Any]) -> str:
    if updates.get("collected") is False or not isinstance(updates.get("count"), int):
        return describe_missing(
            _UPDATE_LIST,
            str(updates.get("reason") or "The update check did not run, so whether any updates are waiting is not known."),
        )
    return "{} operating-system update{} available. {} not been installed.".format(
        updates["count"], " is" if updates["count"] == 1 else "s are",
        ("They are downloaded and staged, and have" if updates.get("staged") else "They are not staged and have")
        if updates["count"] else "Nothing is waiting, so nothing has")


def _reboot_line(updates: Mapping[str, Any]) -> str:
    if "reboot_required" not in updates:
        return describe_missing("whether the operating system wants a restart")
    return ("Yes: the operating system has asked for a restart to finish installing updates."
            if updates.get("reboot_required") else
            "No: the operating system has not asked for a restart.")


def _jobs_line(jobs: List[Mapping[str, Any]]) -> str:
    active = [job for job in jobs if str(job.get("state", "")).lower() in _ACTIVE_JOB_STATES]
    if not jobs:
        return "No recent jobs are recorded, so nothing is installing or running right now."
    if not active:
        latest = jobs[0]
        return "Nothing is installing or running right now. The latest job, {}, is {}.".format(
            str(latest.get("type", "workload")).replace(".", " "), latest.get("state", "in an unknown state"))
    return "{} deployment job{} still going: {}.".format(
        len(active), " is" if len(active) == 1 else "s are",
        "; ".join("{} is {} ({}%)".format(
            str(job.get("type", "workload")).replace(".", " "), job.get("state"),
            job.get("progress", 0)) for job in active[:4]))


#: The Assistant with no model of its own to run; also said by the serving answer.
ASSISTANT_BASIC_MODE = (
    "The Assistant is in built-in basic mode: Vaelor's own model is not set up or not reachable."
)


def chat_connection_words(item: Mapping[str, Any]) -> str:
    """One AI Chat connection, as "label, model X, on this machine" (also the serving answer's)."""
    where = {True: "on this machine", False: "on another machine"}.get(item.get("local"), "")
    return "{}{}{}".format(
        item.get("label") or "a connected model",
        ", model {}".format(item["model"]) if item.get("model") else ", with the model picked per chat",
        ", " + where if where else "")


def inference_line(inference: Mapping[str, Any]) -> str:
    """Which model the Assistant and AI Chat each use, from ``inference.status``."""
    parts: List[str] = []
    engines = [item for item in inference.get("engines") or [] if isinstance(item, Mapping)]
    own = next(iter(engines), None)
    if own is not None and own.get("configured") and own.get("model"):
        parts.append("The Assistant runs on Vaelor's own model, {}.".format(own["model"]))
    elif own is not None:
        parts.append(ASSISTANT_BASIC_MODE)
    chat = inference.get("chat") if isinstance(inference.get("chat"), Mapping) else {}
    active = [item for item in chat.get("connections") or [] if isinstance(item, Mapping) and item.get("active")]
    if active:
        parts.append("AI Chat uses {}.".format(chat_connection_words(active[0])))
    elif chat:
        parts.append("AI Chat has no model connected.")
    local = inference.get("local_models") if isinstance(inference.get("local_models"), Mapping) else {}
    names = [str(model.get("name") or model.get("id")) for model in local.get("models") or []
             if isinstance(model, Mapping) and (model.get("name") or model.get("id"))]
    if names:
        # Review S4: these are the downloaded model files, not what serves AI
        # Chat (the serving answer owns that; in a cluster the serving model
        # may not be a file here at all), and the tool lists at most 20.
        count = local.get("count") if isinstance(local.get("count"), int) else len(names)
        shown = names[:6]
        more = max(0, count - len(shown))
        parts.append("Downloaded model files kept on this machine (stored, not necessarily "
                      "serving): {}{}.".format(", ".join(shown),
                                               "; {} more not named".format(more) if more else ""))
    return " ".join(parts)


def _cooling_lines(cooling: Mapping[str, Any], reading: Any, telemetry: Mapping[str, Any],
                   lower: str) -> List[str]:
    cpu = cooling.get("cpu") if isinstance(cooling.get("cpu"), Mapping) else {}
    case = cooling.get("case") if isinstance(cooling.get("case"), Mapping) else {}
    stated: List[str] = []
    fan_speed = sentence("The CPU fan is turning at {rpm} RPM", rpm=cpu.get("rpm"))
    if fan_speed:
        stated.append(fan_speed + sentence(" in {mode} mode", mode=cpu.get("mode")) + ".")
    else:
        stated.append(describe_missing("the processor fan's speed", str(cpu.get("reason") or "")))
    cooling_state = sentence("The cooling state is {current} of {maximum}.",
                             current=cpu.get("current_state"), maximum=cpu.get("max_state"))
    if cooling_state:
        stated.append(cooling_state)
    if reading is not None:
        stated.append("The CPU is {:.1f}°C.".format(float(reading)))
        norms = telemetry.get("thermal_norms") if isinstance(telemetry.get("thermal_norms"), Mapping) else {}
        limit = norms.get("investigate_above_c") if norms.get("available") else None
        if mentions(lower, ("hot", "hotter", "warm", "warmer", "heat", "overheat", "overheating")):
            if isinstance(limit, (int, float)):
                stated.append(
                    "That is {} this machine's investigate threshold of {:.0f}°C, so {}.".format(
                        "above" if float(reading) > float(limit) else "below", float(limit),
                        "it is hotter than this machine should run" if float(reading) > float(limit)
                        else "it is not hot for this machine"))
            else:
                stated.append("No thermal policy is published for this machine, so I cannot judge whether that is hot.")
    if (present(cpu.get("rpm")) and float(cpu.get("rpm") or 0) == 0 and present(cpu.get("policy"))
            and reading is not None and float(reading) < 55):
        stated.append("Zero RPM is expected here: this machine's cooling policy starts the fan above {}.".format(
            cpu.get("policy")))
    lines = [" ".join(part for part in stated if part)]
    case_line = case_fan_answer(case)
    if case_line:
        lines.append(case_line)
    return lines


def builtin_reply(message: str, facts: Mapping[str, Any], appliance: Mapping[str, Any],
                  accelerator: Optional[Mapping[str, Any]] = None) -> Dict[str, Any]:
    """Every subject this question names, as sentences with their evidence.

    Returns ``{"lines", "evidence", "answered"}``. ``accelerator`` is the
    GPU/NPU answer the caller already built for a compound question.
    """
    lower = str(message or "").lower()
    lines: List[str] = []
    evidence: List[Dict[str, str]] = []
    not_read: set = set()

    def noted(sentence_text: str) -> str:
        """Record a sentence that says a reading did not arrive (review B1)."""
        not_read.add(sentence_text)
        return sentence_text

    cooling = usable_fact(facts, "cooling.status") or {}
    telemetry = usable_fact(facts, "system.telemetry") or {}
    identity = usable_fact(facts, "system.identity") or {}
    capabilities_map = usable_fact(facts, "machine.capabilities") or identity.get("capabilities") or {}
    sweep = mentions(lower, _SWEEP_WORDS)
    reading = cpu_temperature(cooling, telemetry)
    temperature_stated = False
    # Literal calls, one per topic: `tests/test_assistant_builtin_answers.py`
    # reads them out of this source to check each names a declared topic.
    asked = {
        "health-verdict": asks_about("health-verdict", lower),
        "cooling": asks_about("cooling", lower),
        "display": asks_about("display", lower),
        "lighting": asks_about("lighting", lower),
        "updates": asks_about("updates", lower) or sweep,
        "reboot-required": asks_about("reboot-required", lower),
        "services": asks_about("services", lower) or sweep,
        "cpu": asks_about("cpu", lower) or sweep,
        "memory": asks_about("memory", lower) or sweep,
        "uptime": asks_about("uptime", lower),
        "storage": asks_about("storage", lower) or sweep,
        "network": asks_about("network", lower) or sweep,
        "inference": asks_about("inference", lower),
        "logs": asks_about("logs", lower),
        "workloads": asks_about("workloads", lower),
        "jobs": asks_about("jobs", lower),
    }

    # Rendered when the verdict was gathered, or when a cooling reading was:
    # with no verdict in hand the line says "not judged" rather than nothing.
    if (asked["health-verdict"] or sweep) and ("health.status" in facts or cooling):
        lines.append(overall_verdict_line(facts, reading))
        add_evidence(evidence, "health.status", usable_fact(facts, "health.status"), ("status", "checked"))
    # Review B1: an unread cooling tool still leaves the telemetry's own
    # temperature, so the fan is said as not read and the temperature stands.
    if asked["cooling"] and (cooling or reading is not None):
        lines.extend(_cooling_lines(cooling, reading, telemetry, lower))
        temperature_stated = reading is not None
        cpu = cooling.get("cpu") if isinstance(cooling.get("cpu"), Mapping) else {}
        add_evidence(evidence, "cooling.status", cpu, ("rpm", "mode", "current_state", "max_state", "policy"))             or add_evidence(evidence, "system.telemetry", telemetry, ("cpu_temperature",))
    for topic, key, subject in (("display", "display.status", "the display"),
                                ("lighting", "lighting.status", "the case lighting")):
        if asked[topic] and gathered_but_unread(facts, key):
            lines.append(noted(unread(subject)))
    display = usable_fact(facts, "display.status")
    if asked["display"] and display:
        lines.append(display_line(display, capabilities_map))
        add_evidence(evidence, "display.status", display, ("hardware", "bus", "enabled", "rotation", "sleep_timeout"))
    lighting = usable_fact(facts, "lighting.status")
    if asked["lighting"] and lighting:
        lines.append(lighting_line(lighting, capabilities_map))
        add_evidence(evidence, "lighting.status", lighting, ("led_count", "rgb_enable", "rgb_style", "rgb_brightness"))
    updates = usable_fact(facts, "updates.status")
    for topic, render in (("updates", _updates_line), ("reboot-required", _reboot_line)):
        if not asked[topic]:
            continue
        if updates is not None:
            lines.append(noted(render(updates)) if updates.get("collected") is False
                         or (topic == "reboot-required" and "reboot_required" not in updates)
                         else render(updates))
            add_evidence(evidence, "updates.status", updates, ("count", "staged", "reboot_required"))
        elif gathered_but_unread(facts, "updates.status"):
            lines.append(noted(unread(_UPDATE_LIST)))
    services = usable_fact(facts, "services.status", list)
    if asked["services"]:
        if services:
            unhealthy = [str(item.get("id", "service")) for item in services
                         if item.get("available") and item.get("active") != "active"]
            lines.append(ALL_SERVICES_ACTIVE if not unhealthy
                         else "These services need attention: {}.".format(", ".join(unhealthy)))
            add_evidence(evidence, "services.status", services)
        elif gathered_but_unread(facts, "services.status", list):
            lines.append(noted(unread("the managed services")))
    if asked["cpu"] and telemetry:
        details = []
        if telemetry.get("cpu_percent") is not None:
            details.append("{}% use".format(round(float(telemetry["cpu_percent"]), 1)))
        if reading is not None and not temperature_stated:
            details.append("{:.1f}°C".format(reading))
        lines.append("The host CPU is currently {}.".format(" and ".join(details) if details else "reporting live telemetry"))
        add_evidence(evidence, "system.telemetry", telemetry, ("cpu_percent", "cpu_temperature"))
    if asked["memory"] and telemetry:
        fitted = describe_gb(telemetry.get("memory_total"), 0)
        in_use = describe_gb(telemetry.get("memory_used"), 1)
        share = telemetry.get("memory_percent")
        used = "{}%{}".format(round(float(share), 1), " ({})".format(in_use) if in_use else "") if share is not None else ""
        stated = first_sentence(
            sentence("This machine has {fitted} of memory, and {used} of it is in use.", fitted=fitted, used=used),
            sentence("This machine has {fitted} of memory.", fitted=fitted),
            sentence("Memory use is currently {used}.", used=used),
        )
        if stated:
            lines.append("{} RAM is the working space shared by the operating system, apps, and local AI.".format(stated))
            add_evidence(evidence, "system.telemetry", telemetry, ("memory_total", "memory_used", "memory_percent"))
    if asked["uptime"] and telemetry:
        lines.append(_uptime_line(telemetry))
        add_evidence(evidence, "system.telemetry", telemetry, ("boot_time",))
    storage = usable_fact(facts, "storage.status")
    if asked["storage"] and storage is not None:
        lines.append(storage_summary(storage))
        add_evidence(evidence, "storage.status", storage)
    elif asked["storage"] and gathered_but_unread(facts, "storage.status"):
        lines.append(noted(unread("the storage")))
    network = usable_fact(facts, "network.status")
    if asked["network"] and network is not None:
        lines.append(network_summary(network))
        add_evidence(evidence, "network.status", network, ("interfaces",))
    elif asked["network"] and gathered_but_unread(facts, "network.status"):
        lines.append(noted(unread("the network interfaces")))
    inference = usable_fact(facts, "inference.status")
    inference_answered = False
    if asked["inference"] and inference is not None:
        line = inference_line(inference)
        if line:
            lines.append(line)
            inference_answered = True
            add_evidence(evidence, "inference.status", inference, ("engines", "chat"))
    if asked["logs"]:
        line = logs_line(message, facts, evidence)
        if line:
            lines.append(noted(line) if not any(item.get("source") == "logs.service" for item in evidence) else line)
    workloads_asked = asked["workloads"] and not (
        inference_answered and not mentions(lower, _APP_WORDS))
    if workloads_asked:
        workloads = usable_fact(facts, "workloads.inventory")
        if workloads is not None:
            lines.append(managed_workload_summary(workloads))
            add_evidence(evidence, "workloads.inventory", workloads)
        elif gathered_but_unread(facts, "workloads.inventory"):
            lines.append(noted(unread("the apps and models Vaelor manages")))
    jobs = usable_fact(facts, "jobs.recent", list)
    if asked["jobs"]:
        if jobs is not None:
            lines.append(_jobs_line(jobs))
            add_evidence(evidence, "jobs.recent", jobs)
        elif gathered_but_unread(facts, "jobs.recent", list):
            lines.append(noted(unread("the recent jobs")))
    if accelerator is not None:
        lines.append(accelerator["answer"])
        evidence.extend(accelerator.get("evidence") or [])
    elif sweep:
        lines.extend(accelerator_sweep_lines(facts, evidence))
    brief = appliance.get("machine_brief") if isinstance(appliance.get("machine_brief"), Mapping) else {}
    if not lines and _HARDWARE_QUESTION.search(lower):
        line = hardware_line(str(brief.get("text") or ""))
        if line:
            lines.append(line)
            evidence.append({"source": "assistant.machine-brief",
                             "summary": "Read the machine and processor lines from this appliance's standing brief."})
    if not lines:
        lines = static_answer(lower) or []
    specialist_results = appliance.get("specialist_results") or []
    if not lines and asks_about("specialist", lower):
        lines = _specialist_lines(specialist_results)
    # Review B1: a reply made only of "not read" sentences answered nothing.
    answered = any(line not in not_read for line in lines)
    if not lines:
        model = identity.get("name") or identity.get("id") or None
        lines = ["{} {}".format(NO_BUILTIN_ANSWER, capability_sentence(model))]
        answered = False
        if identity:
            add_evidence(evidence, "system.identity", identity, ("name", "id", "model", "architecture"))
    return {"lines": lines, "evidence": evidence, "answered": answered}


def _specialist_lines(specialist_results: Any) -> List[str]:
    if specialist_results:
        latest = specialist_results[0]
        lines = ["The latest {} specialist concluded: {}".format(
            latest.get("profile", "system"), latest.get("summary", "review completed"))]
        findings = latest.get("findings", [])[:3]
        if findings:
            lines.append("Key findings: {}.".format("; ".join(findings)))
        recommendations = latest.get("recommendations", [])[:3]
        if recommendations:
            lines.append("Recommended next steps: {}.".format("; ".join(recommendations)))
        return lines
    return ["There are no completed appliance checks yet. Run a focused read-only specialist review, "
            "then use Discuss this result to bring its output into this conversation."]
