"""Deterministic, read-only specialist reviews built from live facts.

Extracted from ``deployment_agent`` so the agent module stays inside the
thousand-line limit and so the one thing this code must get right - describing
the appliance without a model, and without ignoring what the user said - has a
file and a test of its own.

Every review here is grounded: each sentence restates a value that arrived in
``facts``. What changed with the reported troubleshooting defect is that the
user's own words are now an input too. A review that recites readings while
saying nothing about the problem it was asked to look at is not a review.
"""

from __future__ import annotations

from typing import Any, Dict, List

from .assistant_hardware_answers import cpu_temperature
from .reported_symptoms import failing_services, symptom_findings

MAX_ITEMS = 8


def _system_review(task: Any, facts: Dict[str, Any]) -> Dict[str, List[str]]:
    cooling = facts.get("cooling.status", {})
    if not isinstance(cooling, dict):
        cooling = {}
    cpu = cooling.get("cpu", {}) if isinstance(cooling.get("cpu"), dict) else {}
    case = cooling.get("case", {}) if isinstance(cooling.get("case"), dict) else {}
    telemetry = facts.get("system.telemetry", {})
    # One temperature per review, resolved from the same place every other
    # surface reads it, so an appliance check and a chat answer taken from the
    # same sample cannot quote different numbers.
    reading = cpu_temperature(cooling, telemetry)
    temperature = float(reading or 0)
    rpm = int(cpu.get("rpm") or 0)
    failed = failing_services(facts.get("services.status", []))

    findings = [
        "CPU temperature is {:.1f}°C; the CPU fan reports {} RPM in {} mode.".format(
            temperature, rpm, cpu.get("mode", "automatic")
        )
    ]
    symptoms = symptom_findings(
        task,
        temperature=reading,
        cpu_rpm=cpu.get("rpm"),
        case_running=case.get("running"),
        services_failing=failed,
    )
    # The complaint leads. A reader who has to scroll past four readings to see
    # their own problem addressed has already concluded it was ignored.
    findings = symptoms["findings"] + findings
    if rpm == 0 and temperature < 55 and not symptoms["findings"]:
        # With a symptom present this sentence is answered above, in context.
        # On its own it is the honest explanation of a quiet fan.
        findings.append(
            "Zero CPU-fan RPM is expected at this low temperature in automatic mode."
        )
    findings.append(
        "Both case fans use one shared on/off control and are {}.".format(
            "running" if case.get("running") else "stopped or not reporting"
        )
    )
    findings.append(
        "All managed Vaelor services are active."
        if not failed else "Services needing attention: {}.".format(", ".join(failed))
    )
    recommendations = [
        *symptoms["recommendations"],
        "Keep automatic CPU cooling unless sustained temperature or throttling evidence says otherwise.",
    ]
    next_actions = [
        *symptoms["next_actions"],
        "Review System for live fan curves, service state, and logs.",
    ]
    return {
        "findings": findings,
        "recommendations": recommendations,
        "next_actions": next_actions,
    }


def _docker_review(facts: Dict[str, Any]) -> Dict[str, List[str]]:
    inventory = facts.get("workloads.inventory", {})
    apps = inventory.get("apps", [])
    running = [item for item in apps if item.get("running")]
    unhealthy = [
        item.get("name", "app") for item in apps
        if item.get("running") and item.get("health") not in (None, "", "healthy")
    ]
    findings = [
        "{} managed or discovered app(s); {} currently running.".format(
            len(apps), len(running)
        )
    ]
    if unhealthy:
        findings.append("Apps with non-healthy status: {}.".format(", ".join(unhealthy)))
    capabilities = facts.get("workloads.capabilities", {})
    docker = capabilities.get("docker", {})
    findings.append(
        "Docker is {} and Compose is {}.".format(
            "installed" if docker.get("installed") else "not installed",
            "available" if docker.get("compose") else "unavailable",
        )
    )
    return {
        "findings": findings,
        "recommendations": [
            "Validate ports, storage, resource limits, and health checks before deployment."
        ],
        "next_actions": [
            "Open Workloads to inspect logs, configuration, console, and lifecycle controls."
        ],
    }


def _models_review(facts: Dict[str, Any]) -> Dict[str, List[str]]:
    inventory = facts.get("workloads.inventory", {})
    models = inventory.get("models", [])
    telemetry = facts.get("system.telemetry", {})
    return {
        "findings": [
            "{} local GGUF model(s) are currently indexed.".format(len(models)),
            "Available memory is {} bytes and free storage is checked again before download.".format(
                int(telemetry.get("memory_available") or 0)
            ),
        ],
        "recommendations": [
            "Prefer a Q4_K_M model sized for available RAM, leaving headroom for the host operating system and runtime."
        ],
        "next_actions": [
            "Use the hardware-matched model chooser before downloading or deploying."
        ],
    }


def _security_review(facts: Dict[str, Any]) -> Dict[str, List[str]]:
    unavailable = failing_services(facts.get("services.status", []))
    memory = facts.get("assistant.memory_status", {})
    return {
        "findings": [
            "Managed security-relevant services are active."
            if not unavailable
            else "Inactive managed services: {}.".format(", ".join(unavailable)),
            "Assistant memory contains {} reviewed item(s); secret values are not exposed to this specialist.".format(
                int(memory.get("memories") or 0)
            ),
        ],
        "recommendations": [
            "Keep the dashboard on HTTPS/VPN and use least-privilege local accounts."
        ],
        "next_actions": [
            "Review Admin → Sessions and Connections for unrecognized access."
        ],
    }


def _network_finding(network: Any) -> str:
    """Report the interface count, or the reason there is no count to report."""
    if not isinstance(network, dict):
        return "Network inventory was not available for this review."
    interfaces = network.get("interfaces", [])
    if not interfaces and not network.get("collected", True):
        return str(
            network.get("detail")
            or "Vaelor could not read this machine's network interfaces."
        )
    return "Network inventory returned {} interface(s).".format(len(interfaces))


def _kvm_review(facts: Dict[str, Any]) -> Dict[str, List[str]]:
    inventory = facts.get("workloads.inventory", {})
    desktops = [
        item.get("name", "app") for item in inventory.get("apps", [])
        if (item.get("capabilities") or {}).get("remote_desktop")
    ]
    network = facts.get("network.status", {})
    return {
        "findings": [
            "{} app remote desktop(s) detected{}.".format(
                len(desktops), ": " + ", ".join(desktops) if desktops else ""
            ),
            "Remote access uses one-use VNC tickets; physical HDMI KVM remains a separate optional hardware path.",
            # "Returned 0 interface(s)" is a claim about the machine, and on
            # anything answering this request it is a false one. When the
            # inventory says it could not read them, that is what gets said.
            _network_finding(network),
        ],
        "recommendations": [
            "Prefer the host browser-desktop path when physical capture/HID hardware is absent."
        ],
        "next_actions": [
            "Open Remote console and test the host desktop before commissioning physical KVM."
        ],
    }


def _comparison_review(facts: Dict[str, Any]) -> Dict[str, List[str]]:
    inventory = facts.get("workloads.inventory", {})
    identity = facts.get("system.identity", {})
    return {
        "findings": [
            "Compared {} live fact groups for {}.".format(
                len(facts), identity.get("name") or "this Vaelor node"
            ),
            "{} installed app(s) and {} local model(s) are available for comparison.".format(
                len(inventory.get("apps", [])), len(inventory.get("models", []))
            ),
        ],
        "recommendations": [
            "Compare privacy, memory, storage, support, and rollback before choosing."
        ],
        "next_actions": [
            "Ask the main assistant to explain the trade-offs in beginner language."
        ],
    }


def baseline_review(
    profile: str, task: Any, context: Dict[str, Any]
) -> Dict[str, Any]:
    """Return the built-in review for ``profile``, grounded in ``context`` facts."""
    facts = context.get("facts", {})
    if not isinstance(facts, dict):
        facts = {}
    if profile == "system":
        parts = _system_review(task, facts)
    elif profile == "docker":
        parts = _docker_review(facts)
    elif profile == "models":
        parts = _models_review(facts)
    elif profile == "security":
        parts = _security_review(facts)
    elif profile == "kvm":
        parts = _kvm_review(facts)
    else:
        parts = _comparison_review(facts)

    findings = list(parts["findings"])
    skills = [
        item.get("name", "reviewed skill")
        for item in (context.get("matched_skills") or [])[:3]
    ]
    if skills:
        findings.append("Applied reviewed guidance: {}.".format(", ".join(skills)))
    return {
        "summary": summary_line(profile, task),
        "findings": findings[:MAX_ITEMS],
        "recommendations": parts["recommendations"][:MAX_ITEMS],
        "next_actions": parts["next_actions"][:MAX_ITEMS],
    }


def summary_line(profile: str, task: Any) -> str:
    """One sentence naming the review, and the problem it was asked about."""
    base = "{} review completed from live read-only appliance data.".format(
        str(profile).capitalize()
    )
    symptoms = symptom_findings(task).get("findings")
    if not symptoms:
        return base
    return "{} It was asked about a specific problem, and the reported symptoms " \
           "are addressed in the findings below.".format(base)
