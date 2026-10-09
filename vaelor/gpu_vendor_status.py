"""Did this machine's GPU vendor tool give a reading, and if not, why not?

VD-147 (owner decisions D1 and D5; review B2). Two of the readings the
Performance dashboard shows per machine come from ``amd-smi`` and nowhere else:
the graphics engine's own power and its own temperature. The controller runs
the tool itself; a worker's non-root sampler (`worker_gpu_sampler`) runs it and
hands the result to the telemetry emitter. Either can fail in several distinct
ways, and "no reading" with no reason is the defect this exists to prevent.

**One table, both nodes.** The status is a small integer code. Codes cross the
wire (the ingest path stores numbers only and drops strings); the sentences
stay here, on the controller, which puts the machine's name in. A code the
table does not hold is refused at ingest.

**The readings and the code are decided together**, by :func:`read_vendor`,
from one ``amd-smi metric --json`` result: the same function for the
controller's cached call and for the worker's sampler, so the two cannot come
to describe the same output differently (LESSONS 6).

Standard library only: this module rides in the worker's emitter and sampler
bundles.
"""

from __future__ import annotations

from typing import Any, Dict, Optional

from .platforms.graphics_software import (  # noqa: F401 - the timeout is re-exported
    ACCESS_BANNER, AMD_SMI_GPU_FIELDS, GPU_GFX_POWER_FIELD, VENDOR_TOOL_TIMEOUT_SECONDS,
    json_after_banner,
)

#: The stored field carrying the code, and the one carrying the reading's age.
STATUS_FIELD = "gpu_vendor_status"
AGE_FIELD = "gpu_vendor_age_seconds"

READ = 0
NOT_INSTALLED = 1
ACCESS_WARNING = 2
TIMED_OUT = 3
NO_GRAPHICS_FIELDS = 4
UNIT_NOT_RECOGNISED = 5
SAMPLER_MISSING = 6
UNREADABLE_OUTPUT = 7
NOT_APPLICABLE = 8

#: What each code means, as the owner reads it. ``{node}`` is the machine's
#: name, filled in by the controller. Code 0 has no sentence: there is nothing
#: to explain about a reading that was taken.
SENTENCES: Dict[int, str] = {
    READ: "",
    NOT_INSTALLED: "amd-smi is not installed on {node}.",
    ACCESS_WARNING: (
        "amd-smi on {node} warned that it lacks access to the GPU device, so "
        "some readings may be missing."
    ),
    TIMED_OUT: "amd-smi on {node} did not answer within 5 s.",
    NO_GRAPHICS_FIELDS: (
        "amd-smi on {node} runs, but this build does not report the "
        "graphics-engine readings."
    ),
    UNIT_NOT_RECOGNISED: (
        "amd-smi on {node} reported a value in a unit Vaelor does not recognise, "
        "so that value was left out; the other readings are shown."
    ),
    SAMPLER_MISSING: (
        "The GPU sampler on {node} has stopped or is missing; Vaelor reinstalls "
        "it at its next automatic repair."
    ),
    UNREADABLE_OUTPUT: "amd-smi on {node} returned output Vaelor could not read.",
    NOT_APPLICABLE: "No graphics readings apply to this machine.",
}

#: A worker whose rows carry no status at all is running an emitter from before
#: these readings existed. It clears by itself once the automatic repair
#: reinstalls the agent.
AGENT_PREDATES_READINGS = (
    "{node}'s telemetry agent predates GPU readings; Vaelor updates it at its "
    "next automatic repair."
)

#: The codes under which the readings that did parse are still good. Code 2 is
#: here on purpose: the access banner is a fact about one account's groups, and
#: the ``metric`` values printed after it were measured complete and correct on
#: both boxes. Discarding them would turn a warning into an absence (LESSONS 4).
#: Code 5 is here for the same reason (review S-13): the ONE value in a unit
#: Vaelor does not read is left out, and every value whose unit it did read is
#: kept - at the fields the controller stores and the sampler writes alike.
CODES_WITH_READINGS = frozenset({READ, ACCESS_WARNING, UNIT_NOT_RECOGNISED})

#: The codes that are a healthy answer rather than a fault: a reading taken,
#: and a machine with no graphics readings to take.
HEALTHY_CODES = frozenset({READ, NOT_APPLICABLE})

#: The package-power field a WORKER stores from the vendor tool. The controller
#: already stores the same figure as ``cpu_package_power_watts`` (from this very
#: amd-smi snapshot, `linux_telemetry._socket_power`), so it does not store it a
#: second time under this name: one key per node for one quantity.
WORKER_ONLY_FIELDS = frozenset({"gpu_socket_power_watts"})

#: How old a worker sampler's reading may be and still be used, in seconds. The
#: emitter reports :data:`SAMPLER_MISSING` past it. Equal to the serving tab's
#: `SERVING_STALE_AFTER_SECONDS` (three ten-second samples), which a test pins;
#: it is restated here rather than imported because this module rides in the
#: worker bundles and that one does not.
VENDOR_SAMPLE_MAX_AGE_SECONDS = 30.0

#: How often the worker's sampler runs the tool, in seconds. The controller's
#: own cache (`accelerators.VENDOR_TOOL_CACHE_SECONDS`) is the same ten.
VENDOR_SAMPLE_INTERVAL_SECONDS = 10.0

def known_code(value: Any) -> Optional[int]:
    """``value`` as a status code the table holds, or ``None``.

    Accepts the float the store and the wire carry (``2.0``) as well as an
    integer; anything that is not a whole number in the table is ``None``.
    Total (review B2): NaN, infinity and an integer too large for a float are
    ``None``, never an exception - the value comes from a file an unprivileged
    process wrote.
    """
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        return None
    if isinstance(value, int):
        return value if value in SENTENCES else None
    if value != value or value in (float("inf"), float("-inf")) or not value.is_integer():
        return None
    return int(value) if int(value) in SENTENCES else None


def status_sentence(code: Any, node_name: str) -> str:
    """The owner sentence for a stored code, with the machine named.

    ``None`` (a row that carries no status) reads as an agent from before these
    readings; an unknown number reads as unreadable output, which is what it is.
    """
    name = str(node_name or "the machine")
    if code is None:
        return AGENT_PREDATES_READINGS.format(node=name)
    known = known_code(code)
    template = SENTENCES[UNREADABLE_OUTPUT if known is None else known]
    return template.format(node=name)


def readings_from(parsed: Any, banner: bool = False) -> Dict[str, Any]:
    """``{"status": code, "values": {field: number}}`` from a parsed ``metric`` document.

    The one reading of the document, for the controller's cached call and the
    worker's sampler alike. A value printed in a unit Vaelor does not read is
    dropped (never scaled by a guess) and the result says so with code 5; the
    access banner is code 2 and **keeps** every value that parsed.
    """
    values: Dict[str, float] = {}
    unknown_units: list = []
    for field, reader in AMD_SMI_GPU_FIELDS:
        value = reader(parsed, unknown_units)
        if value is not None:
            values[field] = value
    if banner:
        status = ACCESS_WARNING
    elif unknown_units:
        status = UNIT_NOT_RECOGNISED
    elif not values:
        status = NO_GRAPHICS_FIELDS
    else:
        status = READ
    return {"status": status, "values": values}


def read_vendor(
    stdout: Any, *, installed: bool = True, timed_out: bool = False,
    returncode: int = 0,
) -> Dict[str, Any]:
    """:func:`readings_from` for one raw ``amd-smi metric --json`` run.

    The order of the checks is the order of what can be known: a tool that is
    not there, one that did not answer, one that answered with nothing
    parseable, and only then - with a parsed document in hand - what is in it.
    """
    if not installed:
        return {"status": NOT_INSTALLED, "values": {}}
    if timed_out:
        return {"status": TIMED_OUT, "values": {}}
    text = stdout if isinstance(stdout, str) else ""
    parsed = json_after_banner(text) if returncode == 0 else None
    if parsed is None:
        return {"status": UNREADABLE_OUTPUT, "values": {}}
    return readings_from(parsed, ACCESS_BANNER in text)


#: How `accelerators.enrich_from_vendor_tool` names a failed read, mapped onto
#: the codes.
_FAILURE_CODES = {"absent": NOT_INSTALLED, "timeout": TIMED_OUT, "unreadable": UNREADABLE_OUTPUT}


def controller_vendor_fields(result: Any) -> Dict[str, Any]:
    """The telemetry fields the controller stores from its cached ``amd-smi`` call.

    ``result`` is `accelerators.cached_vendor_tool_metrics`'s answer. Always
    carries the status code; carries the readings only under a code that has
    them, the reading's age, and - as before - ``gpu_power_source`` naming the
    amd-smi field the power came from. Package power is left out here (see
    :data:`WORKER_ONLY_FIELDS`).
    """
    result = result if isinstance(result, dict) else {}
    if not result.get("available"):
        code = _FAILURE_CODES.get(str(result.get("failure") or ""), UNREADABLE_OUTPUT)
        return {STATUS_FIELD: code}
    reading = readings_from(result.get("metrics"), bool(result.get("banner")))
    fields: Dict[str, Any] = {STATUS_FIELD: reading["status"]}
    if reading["status"] in CODES_WITH_READINGS:
        fields.update({
            name: value for name, value in reading["values"].items()
            if name not in WORKER_ONLY_FIELDS
        })
        if "gpu_power_watts" in fields:
            fields["gpu_power_source"] = GPU_GFX_POWER_FIELD
    age = result.get("age_seconds")
    if isinstance(age, (int, float)) and not isinstance(age, bool):
        fields[AGE_FIELD] = float(age)
    return fields
