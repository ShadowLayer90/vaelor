"""Say what was actually read, or claim no source at all.

The Assistant's "Evidence used" panel is the product's trust mechanism: it is
what a non-expert checks when they want to know whether an answer is grounded.
It was a fixed string. Asked about cooling on an x86 workstation, it reported
its source as "Live PWM, RPM, GPIO fan policy, and temperature" — GPIO is a
Raspberry Pi concept, there is no PWM reading on that machine, and the RPM in
the answer was invented. Boilerplate provenance is worse than none, because
none is honest and boilerplate is a claim.

Two helpers, and both fail towards silence:

* :func:`evidence_for` describes the fields a fact object actually carried, and
  returns ``None`` when it carried nothing worth citing. A caller that appends
  its result unconditionally ends up with no entry rather than a false one.
* :func:`sentence` refuses to render a template whose values are missing, so a
  ``None`` cannot reach a user as the word "None" and a placeholder cannot
  reach them as "unknown of unknown".
"""

from __future__ import annotations

from typing import Any, Dict, Iterable, Mapping, Optional, Sequence


#: Human names for the fields a fact object may carry, so a summary reads as
#: prose rather than as key names. A field absent from this map is described by
#: its own key, which is still true — it is only less pretty.
FIELD_NAMES = {
    "rpm": "fan speed",
    "mode": "fan mode",
    "current_state": "cooling state",
    "max_state": "maximum cooling state",
    "celsius": "temperature",
    "cpu_temperature": "CPU temperature",
    "cpu_percent": "CPU use",
    "memory_percent": "memory use",
    "enabled": "power state",
    "rotation": "rotation",
    "sleep_timeout": "sleep timeout",
    "rgb_enable": "lighting power",
    "rgb_style": "lighting effect",
    "rgb_brightness": "brightness",
    "count": "available update count",
    "staged": "staging state",
    "interfaces": "network interfaces",
    "fans": "fan tachometers",
    "temperatures": "board temperatures",
}


def present(value: Any) -> bool:
    """Whether this value is a real reading rather than a hole.

    ``False`` and ``0`` are real readings and stay. ``None`` and the empty
    string are not, and neither is the literal word "unknown", which is what a
    placeholder default looks like once it has been rendered.
    """
    if value is None:
        return False
    if isinstance(value, str):
        return value.strip().lower() not in {"", "unknown", "none", "n/a"}
    return True


def _named(field: str) -> str:
    return FIELD_NAMES.get(field, str(field).replace("_", " "))


def evidence_for(
    source: str,
    payload: Any,
    fields: Optional[Sequence[str]] = None,
    *,
    detail: str = "",
) -> Optional[Dict[str, str]]:
    """One evidence entry naming the fields this machine actually supplied.

    Returns ``None`` when nothing was read, so the panel shows no source rather
    than a source that describes another machine's hardware.
    """
    if not present(source):
        return None
    if isinstance(payload, Mapping):
        candidates = list(fields) if fields else list(payload.keys())
        found = [field for field in candidates if present(payload.get(field))]
        if not found:
            return None
        summary = "Read from this machine: {}.".format(
            ", ".join(_named(field) for field in found)
        )
    elif isinstance(payload, (list, tuple)):
        if not payload:
            return None
        summary = "Read from this machine: {} record{}.".format(
            len(payload), "" if len(payload) == 1 else "s"
        )
    elif present(payload):
        summary = "Read from this machine."
    else:
        return None
    if detail:
        summary = "{} {}".format(summary, detail)
    return {"source": str(source), "summary": summary}


def add_evidence(
    evidence: list,
    source: str,
    payload: Any,
    fields: Optional[Sequence[str]] = None,
    *,
    detail: str = "",
) -> bool:
    """Append an evidence entry only if there is one. Returns whether it was."""
    entry = evidence_for(source, payload, fields, detail=detail)
    if entry is None:
        return False
    evidence.append(entry)
    return True


def sentence(template: str, **values: Any) -> str:
    """Render ``template``, or return ``""`` if any value is missing.

    A sentence with a hole in it is not a degraded sentence, it is a false one:
    "The front None is detected over None" asserts a detection that did not
    happen and names hardware that does not exist. Refusing to render is the
    only outcome that does not mislead.
    """
    for value in values.values():
        if not present(value):
            return ""
    try:
        return template.format(**values)
    except (IndexError, KeyError, ValueError):
        return ""


def first_sentence(*candidates: str) -> str:
    """The first template that could be rendered completely."""
    for candidate in candidates:
        if candidate:
            return candidate
    return ""


def describe_missing(subject: str, reason: str = "") -> str:
    """State that something was not readable, and why, without guessing."""
    base = "Vaelor has no reading for {} on this machine.".format(subject)
    return "{} {}".format(base, reason.strip()) if reason.strip() else base


def contradicts_capability(
    capabilities: Optional[Mapping[str, Any]], name: str
) -> str:
    """The capability map's reason when it says this hardware is absent.

    An answer must not assert that a display "is detected" while the machine
    page says it was not. Where the two disagree the capability map wins: it is
    the probed answer, and the prose was written from a fact object that may
    carry defaults.
    """
    record = (capabilities or {}).get(name)
    if not isinstance(record, Mapping):
        return ""
    if record.get("available"):
        return ""
    return str(record.get("reason") or "")


def readable_fields(payload: Any, fields: Iterable[str]) -> Dict[str, Any]:
    """Only the fields that carry real values, for safe interpolation."""
    if not isinstance(payload, Mapping):
        return {}
    return {
        field: payload.get(field)
        for field in fields
        if present(payload.get(field))
    }
