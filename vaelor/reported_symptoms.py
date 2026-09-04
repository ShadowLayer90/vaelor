"""Carry what the user actually said into a health review.

A troubleshooter that never repeats the complaint has not troubleshot anything.
Asked "my machine keeps freezing and the fan is really loud, what is wrong with
it?", the built-in system review answered with a temperature, a fan RPM, a
service list and "Keep automatic CPU cooling." It never mentioned freezing, never
mentioned the noise, and - worst of all - reported 0 RPM without noticing that a
fan reported as silent cannot be the fan the user is listening to. A reading that
contradicts a reported symptom is the most valuable fact in the whole review, and
it was being dropped.

This module does two things and nothing else. It names the symptoms present in
the user's own words, and it compares them against the live readings. It never
invents a diagnosis and never claims a cause it cannot see; where the evidence
runs out it says so, because "I cannot explain this from the readings" is a
finding and pretending otherwise is not.
"""

from __future__ import annotations

import re
from typing import Any, Dict, List, Mapping, Optional, Sequence, Tuple

# Each symptom is (key, plain phrase, vocabulary). Vocabulary entries are word
# stems matched on a word boundary at the start, so "freezing", "freezes" and
# "froze" all reach the same symptom without an enumerated word list.
_SYMPTOMS: Tuple[Tuple[str, str, Tuple[str, ...]], ...] = (
    (
        "freezing", "freezing or locking up",
        ("freez", "froze", "frozen", "lock up", "locks up", "locking up",
         "lockup", "hang", "hangs", "hanging", "unresponsive", "stall",
         "stalls", "stalling", "not responding"),
    ),
    (
        "loud_fan", "a loud fan",
        ("loud", "louder", "noisy", "noise", "roaring", "screaming",
         "whining", "whine", "buzzing", "rattling", "rattle", "jet engine",
         "hair dryer"),
    ),
    (
        "hot", "running hot",
        ("overheat", "burning up", "too hot", "really hot", "very hot",
         "boiling", "scorching"),
    ),
    (
        "restarting", "restarting or shutting down on its own",
        ("reboots", "rebooting", "restarts by itself", "shuts down",
         "shutting down", "power cycle", "powers off", "turns itself off"),
    ),
    (
        "slow", "running slowly",
        ("slow", "sluggish", "laggy", "lagging", "crawl", "stutter"),
    ),
    (
        "crashing", "crashing",
        ("crash", "crashes", "crashing", "kernel panic", "blue screen"),
    ),
)

_COMPILED: Tuple[Tuple[str, str, Tuple[re.Pattern[str], ...]], ...] = tuple(
    (
        key,
        phrase,
        tuple(re.compile(r"\b" + re.escape(stem), re.IGNORECASE) for stem in stems),
    )
    for key, phrase, stems in _SYMPTOMS
)


def reported_symptoms(task: Any) -> List[Dict[str, str]]:
    """Return the symptoms this request states, in the order they are listed."""
    text = str(task or "")
    if not text.strip():
        return []
    found: List[Dict[str, str]] = []
    for key, phrase, patterns in _COMPILED:
        position = min(
            (match.start() for match in
             (pattern.search(text) for pattern in patterns) if match),
            default=None,
        )
        if position is not None:
            found.append({"key": key, "phrase": phrase, "at": position})
    found.sort(key=lambda item: item["at"])
    return [{"key": item["key"], "phrase": item["phrase"]} for item in found]


def _number(value: Any) -> Optional[float]:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        return None
    return float(value)


def symptom_findings(
    task: Any,
    *,
    temperature: Optional[float] = None,
    cpu_rpm: Any = None,
    case_running: Any = None,
    services_failing: Sequence[str] = (),
) -> Dict[str, List[str]]:
    """State the reported symptoms, and where the readings disagree with them.

    Returns ``{"findings": [...], "recommendations": [...], "next_actions": [...]}``
    with empty lists when the request describes no symptom at all, so a review
    with nothing to say adds nothing.
    """
    symptoms = reported_symptoms(task)
    if not symptoms:
        return {"findings": [], "recommendations": [], "next_actions": []}

    keys = {item["key"] for item in symptoms}
    findings = [
        "You reported {}. This review checks the live readings against that.".format(
            _join([item["phrase"] for item in symptoms])
        )
    ]
    recommendations: List[str] = []
    next_actions: List[str] = []

    rpm = _number(cpu_rpm)
    # The single most diagnostic fact available, and the one that was being
    # dropped: a fan the user can hear across the room cannot be the fan
    # reporting zero.
    if "loud_fan" in keys and rpm is not None and rpm <= 0:
        findings.append(
            "This does not add up: you describe a loud fan, but the CPU fan "
            "reports 0 RPM. Both cannot be true of the same fan. Either the "
            "noise is coming from the case fans or another component, or the "
            "CPU fan is spinning and its speed sensor is not being read. That "
            "mismatch is the first thing to settle."
        )
        next_actions.append(
            "With the case open or an ear to it, identify which fan is making "
            "the noise - CPU fan, case fans, or a drive - then compare that "
            "against the 0 RPM reading."
        )
    elif "loud_fan" in keys and rpm is not None and rpm > 0:
        findings.append(
            "The CPU fan is turning at {:.0f} RPM, which is consistent with the "
            "noise you describe.".format(rpm)
        )

    if "loud_fan" in keys and case_running is True:
        findings.append(
            "The case fans are running, so they are a candidate for the noise "
            "you can hear."
        )

    if "hot" in keys and temperature is not None and temperature < 70:
        findings.append(
            "You report the machine running hot, but the CPU is reading "
            "{:.1f}°C, which is a normal working temperature. If the case "
            "feels hot, the heat may be coming from a drive or the power "
            "supply rather than the processor.".format(temperature)
        )

    unexplained = keys & {"freezing", "restarting", "crashing", "slow"}
    if unexplained:
        explained = []
        if temperature is not None and temperature >= 80:
            explained.append(
                "the CPU is at {:.1f}°C, hot enough to throttle or shut "
                "down".format(temperature)
            )
        if services_failing:
            explained.append(
                "these managed services are not active: {}".format(
                    ", ".join(str(item) for item in services_failing[:5])
                )
            )
        if explained:
            findings.append(
                "A likely cause for {}: {}.".format(
                    _join(sorted(
                        item["phrase"] for item in symptoms
                        if item["key"] in unexplained
                    )),
                    _join(explained),
                )
            )
        else:
            # Saying "everything looks fine" to someone whose machine keeps
            # freezing is the answer that made this review useless. Say what
            # the readings do and do not cover instead.
            findings.append(
                "Nothing in these readings explains {}. They are a snapshot of "
                "this moment, and a machine that freezes intermittently is "
                "usually fine in every sample taken while it is working - so "
                "this does not mean nothing is wrong. Power supply, storage "
                "errors, memory faults and kernel messages are the usual "
                "causes, and none of them show up here.".format(
                    _join(sorted(
                        item["phrase"] for item in symptoms
                        if item["key"] in unexplained
                    ))
                )
            )
            next_actions.append(
                "Open System > Hardware & services and read the system log "
                "around the time of the last freeze; that is where power, "
                "storage and memory faults are recorded."
            )
            recommendations.append(
                "Treat this as unexplained rather than healthy: capture the "
                "time of the next freeze so the log can be read against it."
            )
    return {
        "findings": findings,
        "recommendations": recommendations,
        "next_actions": next_actions,
    }


def _join(items: Sequence[str]) -> str:
    values = [str(item) for item in items if str(item).strip()]
    if not values:
        return ""
    if len(values) == 1:
        return values[0]
    return "{} and {}".format(", ".join(values[:-1]), values[-1])


def failing_services(services: Any) -> List[str]:
    """Managed services that are installed but not active."""
    if not isinstance(services, list):
        return []
    return [
        str(item.get("id", "service"))
        for item in services
        if isinstance(item, Mapping)
        and item.get("available")
        and item.get("active") != "active"
    ]
