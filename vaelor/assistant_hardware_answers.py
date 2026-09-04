"""Portable built-in explanations for live hardware facts."""

from __future__ import annotations

from typing import Any, Dict, Mapping, Optional, Sequence

from .answer_evidence import (
    contradicts_capability,
    describe_missing,
    present,
    sentence,
)


def cpu_temperature(
    cooling: Any = None, telemetry: Any = None
) -> Optional[float]:
    """The one CPU temperature an answer is allowed to quote.

    Two fact groups carry a CPU temperature: ``cooling.status`` from the fan
    controller and ``system.telemetry`` from appliance metrics. Letting each
    sentence pick its own is how one reply said "the CPU is 43.5°C" and, two
    lines later, "the host CPU is currently 26.7% use and 44.6°C".

    Telemetry wins because it is the reading every other surface already shows -
    Overview, the sidebar, health, and automation triggers - so agreeing with it
    makes the answer agree with the rest of the product. The cooling group is
    the fallback, not a second opinion: it is used only when telemetry reports
    no temperature at all.
    """
    for source, key in (
        (telemetry, "cpu_temperature"),
        ((cooling or {}).get("cpu") if isinstance(cooling, dict) else None,
         "temperature"),
    ):
        if not isinstance(source, dict):
            continue
        value = source.get(key)
        if isinstance(value, bool) or not isinstance(value, (int, float)):
            continue
        return float(value)
    return None


def thermal_summary(telemetry: Any) -> str:
    """A compact, CPU-temperature-first line that survives the context budget.

    The reading that answers "how hot is the CPU" lives in ``system.telemetry``
    - the reading every other surface already shows, see :func:`cpu_temperature`.
    But telemetry is a wide dict, and a custom-agent run compacts the granted
    context to its *leading* entries before a small model reads it, and the
    registry hands the keys back with ``cpu_temperature`` sitting past that cut.
    So a grant that promised system facts delivered ``cpu_percent`` and
    ``cpu_freq`` and truncated the temperature away, and a model with no
    temperature in hand reported it could not verify one from public sources -
    the same class of defect VD-115 fixed for NVMe usage. A short leading
    ``summary`` string, temperature first, is exactly what ``storage.status`` and
    ``network.status`` already carry for this budget, so the reading arrives as
    data rather than a truncated-away key. The value is the same sample the rest
    of the payload holds - :func:`cpu_temperature` reads it back, not a second
    poll - so the summary cannot disagree with the numbers beside it.
    """
    if not isinstance(telemetry, dict):
        telemetry = {}
    temperature = cpu_temperature(telemetry=telemetry)
    parts = [
        "CPU temperature {:.1f}°C".format(temperature)
        if temperature is not None
        else "CPU temperature not reported by this machine"
    ]
    load = telemetry.get("cpu_percent")
    if isinstance(load, (int, float)) and not isinstance(load, bool):
        parts.append("CPU load {:.0f}%".format(float(load)))
    memory = telemetry.get("memory_percent")
    if isinstance(memory, (int, float)) and not isinstance(memory, bool):
        parts.append("memory {:.0f}% used".format(float(memory)))
    return ", ".join(parts) + "."


def display_line(
    display: Mapping[str, Any], capabilities: Optional[Mapping[str, Any]] = None
) -> str:
    """The front display, described from what was read or refused outright.

    ``.get(key, default)`` does not help when the key is present and holds
    ``None``, which is exactly what an absent-hardware payload looks like.
    That produced "The front None is detected over None" - a sentence that
    asserted a detection while the machine page said otherwise.

    Where the probed capability map says the hardware is absent it wins over
    the fact object, which may carry defaults.
    """
    refusal = contradicts_capability(capabilities, "oled")
    if refusal or not display.get("detected"):
        return describe_missing("a front display", refusal)
    described = sentence(
        "The front {hardware} is detected over {bus}.",
        hardware=display.get("hardware"), bus=display.get("bus"),
    ) or "A front display is detected."
    stated = [part for part in (
        sentence("it is {state}", state=(
            "on" if display.get("enabled") else "off"
        ) if present(display.get("enabled")) else None),
        sentence("rotated {rotation}°", rotation=display.get("rotation")),
        sentence(
            "sleeps after {sleep} seconds", sleep=display.get("sleep_timeout"),
        ),
    ) if part]
    if not stated:
        return described
    return "{} {}.".format(described, ", ".join(stated)).replace(" .", ".")


def lighting_line(
    lighting: Mapping[str, Any], capabilities: Optional[Mapping[str, Any]] = None
) -> str:
    """Case lighting, with the LED count read rather than assumed.

    "The four RGB case lights" was a constant describing one enclosure.
    """
    refusal = contradicts_capability(capabilities, "case_lighting")
    if refusal or not lighting.get("detected", True):
        return describe_missing("case lighting", refusal)
    count = sentence(
        "The {count} RGB case lights are", count=lighting.get("led_count")
    ) or "The case lighting is"
    stated = [part for part in (
        sentence("using the {style} effect", style=lighting.get("rgb_style")),
        sentence(
            "at {brightness}% brightness",
            brightness=lighting.get("rgb_brightness"),
        ),
    ) if part]
    return "{} {}{}.".format(
        count, "on" if lighting.get("rgb_enable") else "off",
        " " + " ".join(stated) if stated else "",
    )


def _board_fan_clause(fan: Mapping[str, Any]) -> str:
    """One board-sensor fan, named, with whatever RPM it actually reports."""
    label = str(fan.get("label", "")).strip() or "an unlabelled fan"
    rpm = fan.get("rpm")
    if isinstance(rpm, (int, float)) and not isinstance(rpm, bool):
        stated = "{} at {:.0f} RPM".format(label, float(rpm))
    else:
        stated = "{} (no RPM reported)".format(label)
    if fan.get("fault"):
        stated += " (reporting a fault)"
    return stated


def _board_fans_sentence(board_fans: Sequence[Mapping[str, Any]]) -> str:
    """The fans the board sensor reads beyond the CPU fan already stated.

    Enclosure discovery counts Pironman case fans and knows nothing of a
    power-supply fan or a second CPU fan the board sensor reads. Those arrive
    here as ``board_fans`` so the Assistant reports every fan the machine reads
    - the System > Cooling card's own reality - rather than the one the CPU
    line happened to name.
    """
    clauses = [
        _board_fan_clause(fan)
        for fan in board_fans
        if isinstance(fan, Mapping)
    ]
    clauses = [clause for clause in clauses if clause]
    if not clauses:
        return ""
    listed = (
        clauses[0] if len(clauses) == 1
        else "{} and {}".format(", ".join(clauses[:-1]), clauses[-1])
    )
    return "This machine's board sensor also reads {}.".format(listed)


def case_fan_answer(case: Dict[str, Any]) -> str:
    """Describe reported enclosure airflow without assuming a product model.

    The count comes from enclosure discovery and is reported as found. It used
    to be floored at one, which meant a machine with no enclosure was told it
    had a case fan, and a Pironman 5 Mini (one fan) or Pro Max (three) was
    described by whatever the caller happened to pass.

    ``board_fans`` are the fans the board sensor reads that enclosure discovery
    never knew about - a second CPU fan, a power-supply fan. They are reported
    alongside the enclosure fans, or in their place where there is no enclosure,
    so the answer covers every fan the machine reads rather than saying "no case
    fans" while other fans are turning.
    """
    board_line = _board_fans_sentence(case.get("board_fans") or [])
    fan_count = int(case.get("fan_count", 0) or 0)
    if fan_count <= 0:
        if board_line:
            return board_line
        return (
            "No enclosure case fans were detected on this machine, so there is "
            "no case airflow for Vaelor to report or control."
        )
    control = (
        " on one shared on/off control."
        if case.get("shared_control")
        else "."
    )
    state = (
        "running"
        if case.get("running")
        else "stopped"
        if case.get("running") is False
        else "not reporting a live state"
    )
    rpm = (
        " and this hardware does not provide case-fan RPM"
        if case.get("rpm_available") is False
        else ""
    )
    enclosure = (
        f"The detected enclosure has {fan_count} case "
        f"fan{'' if fan_count == 1 else 's'}{control} "
        f"They are {state}{rpm}."
    )
    return "{} {}".format(enclosure, board_line) if board_line else enclosure
