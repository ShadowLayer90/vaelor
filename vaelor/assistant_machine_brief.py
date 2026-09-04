"""A compact standing brief describing the machine the assistant runs on.

The tool registry answers questions. It does not tell a model *which* questions
this machine can answer, and without that the model reasons about the appliance
it has read about rather than the one in front of it: it asks for case-fan speed
on a chassis whose fans belong to an embedded controller, and calls a Ryzen AI
Max boosting to 95 °C a fault because 80 °C is where a Raspberry Pi throttles.

This is the map. It is deliberately small, it holds only facts that do not move
between requests, and it is hard-capped, because a brief that grows without
limit eats the context the live evidence needs. Everything mutable - every
temperature, utilisation figure, byte count and job - stays in the tools.

**The brief is re-prefilled on every single request.** It was written on the
assumption that a fixed prefix would be cached after the first one. FLM does no
prefix caching at all - measured on the live NPU, 20 of 20 requests logged
``Prompt cache miss`` and none logged a hit - so the whole brief is paid again,
in full, forever. An interleaved A/B on the real assistant payload put that at
**821 ms of NPU prefill per request** for a 785-token ``machine`` key.

Two consequences run through this module:

* **Every cap here is a latency guard, not a context guard.** Headroom under
  the cap is not free space; it is milliseconds that will be spent on every
  answer this appliance ever gives.
* **Guidance that is only relevant next to a number does not belong here.** How
  to read the VRAM/GTT split and what counts as a hot CPU for this class of
  machine now travel on the tool responses that carry those figures, which is
  VD-008 applied properly: the explanation arrives with the data it explains,
  at the moment the model asked for it, instead of thousands of tokens earlier.
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional

#: Characters per token, calibrated against the tokeniser that actually pays
#: for this text. This module held its own copy of the figure while
#: ``provider_runtime`` and ``model_calibration`` held two more, all three
#: independently wrong; it now comes from the one place the measurement lives.
from .token_estimate import CHARS_PER_TOKEN, estimate_tokens

#: The hard ceiling. A brief is a map, not a manual: past this it is competing
#: with the evidence it exists to help interpret - and, since nothing is
#: cached, spending real prefill milliseconds on every request to do it.
#:
#: Lowered from 2000. At 2000 estimated tokens the brief could have grown by
#: another 1,009 tokens - about 1.15 s of NPU prefill on every single answer -
#: without one test failing, because the tests only asserted it was under the
#: cap. The exact-size assertions in ``tests/test_assistant_machine_brief.py``
#: are the real guard; this is the backstop behind them.
MAX_BRIEF_TOKENS = 1000
MAX_BRIEF_CHARS = MAX_BRIEF_TOKENS * CHARS_PER_TOKEN

#: Human labels for the capability keys the platform drivers answer for.
CAPABILITY_LABELS = {
    "case_fan": "Enclosure fan control",
    "case_lighting": "Enclosure RGB lighting",
    "oled": "Front OLED display",
    "cpu_fan": "CPU fan control",
    "battery": "Battery or UPS reporting",
    "gpu": "Compute GPU",
    "npu": "Neural processing unit",
}

#: What each tool is for, in the fewest words that still let a model choose
#: correctly. This is the part that closes the real gap: a model cannot ask for
#: ``gpu.status`` if nothing ever told it the tool exists.
TOOL_MAP = (
    ("system.telemetry", "live CPU, memory, temperature, fan, storage, network"),
    ("system.identity", "model, software version, detected peripherals"),
    ("gpu.status", "GPU load, VRAM/GTT split, temperature, power, clocks"),
    ("npu.status", "NPU presence, utilisation, power, clock, rated throughput"),
    ("inference.status", "which model is loaded on which engine"),
    ("cooling.status", "fan state and CPU temperature"),
    ("storage.status", "disks, filesystems, free space"),
    ("network.status", "interfaces, addresses, counters, DNS"),
    ("services.status", "managed service health"),
    ("updates.status", "available and staged OS updates"),
    ("logs.service", "recent journal lines for one managed service"),
    ("metrics.history", "recent telemetry samples, for trends"),
    ("configuration.summary", "stored appliance configuration"),
    ("workloads.inventory", "managed apps, local models, ports, health"),
    ("jobs.recent", "recent approved jobs and their progress"),
    ("cluster.summary", "head controller, workers, pooled inference"),
)

#: Tools that only mean anything on a machine that has the hardware they
#: report on, keyed by the capability that decides it.
#:
#: **The brief used to offer all sixteen everywhere.** A Raspberry Pi has no
#: neural processing unit - `hardware_inventory` says so, the tier planner emits
#: only a `cpu` tier, and the brief itself states "Neural processing unit: not
#: available" a few lines above - and then the same brief told the model it
#: could call `npu.status` for "NPU presence, utilisation, power, clock, rated
#: throughput". Offering a tool is an invitation to use it, and a model that
#: calls it has been set up to report on hardware that is not there. That is
#: the failure VD-042 and #72 are about, arriving through the tool list rather
#: than through prose.
#:
#: Only hardware-bound tools appear here. `storage.status`, `network.status`
#: and the rest describe things every machine has, and gating them on discovery
#: would invent a second way to be wrong.
HARDWARE_TOOLS = {
    "gpu.status": "gpu",
    "npu.status": "npu",
}


def available_tools(capabilities: Dict[str, Any]) -> tuple:
    """`TOOL_MAP` minus the tools whose hardware this machine does not have.

    Reads the same capability records `_capability_lines` reads, so the tool
    list and the "does not have" section cannot disagree - a brief that names a
    device as absent and then offers a tool for it is worse than either alone.

    An unknown or malformed capability record leaves the tool in place. A tool
    that exists and returns "not available" is a smaller fault than a tool that
    was silently withheld from a machine that had the hardware.
    """
    filtered = []
    for name, purpose in TOOL_MAP:
        key = HARDWARE_TOOLS.get(name)
        record = capabilities.get(key) if key else None
        if key and isinstance(record, dict) and not record.get("available"):
            continue
        filtered.append((name, purpose))
    return tuple(filtered)


_UNKNOWN = "not reported"


def _gigabytes(value: Any) -> str:
    try:
        number = int(value)
    except (TypeError, ValueError):
        return ""
    if number <= 0:
        return ""
    return "{:.0f} GB".format(number / (1000 ** 3))


def _sentence(value: str) -> str:
    text = " ".join(str(value or "").split())
    if text and not text.endswith((".", "!", "?")):
        text += "."
    return text


def _processor_line(board: Dict[str, Any]) -> str:
    """Cores and threads, told apart, because the field used to conflate them.

    ``cpu_cores`` came from ``os.cpu_count()`` — logical CPUs — and this line
    printed it as *"32 cores"* on a 16-core part, in every answer the Assistant
    ever gave about the machine. Where the core count could not be established
    the line says *logical CPUs* and does not call them cores, which is the
    same honesty the capability lines already use for absent hardware.
    """
    cores = board.get("cpu_cores")
    threads = board.get("cpu_threads")
    if not cores:
        count = _UNKNOWN
    elif board.get("cpu_cores_are_threads"):
        count = "{} logical CPUs, cores not reported".format(cores)
    elif threads and int(threads) > int(cores):
        count = "{} cores / {} threads".format(cores, threads)
    else:
        count = "{} cores".format(cores)
    return "Processor: {}, {}, {}.".format(
        board.get("cpu_model") or _UNKNOWN,
        count,
        board.get("architecture") or _UNKNOWN,
    )


def _memory_line(split: Dict[str, Any], visible: str) -> str:
    """The split, and that it is a setting rather than a property.

    Per VD-062 a value an owner can change is a reading, not a fact — so the
    carve-out is named as firmware configuration wherever it can be read, and
    where it cannot the brief says the OS figure is not the whole total instead
    of presenting it as one.
    """
    reserved = _gigabytes(split.get("graphics_reserved_bytes"))
    total = _gigabytes(split.get("accounted_total_bytes"))
    if not visible:
        return ""
    if not reserved:
        return (
            "Memory: {} visible to the operating system. No graphics "
            "reservation could be read, so how much is fitted is not known "
            "from this figure alone.".format(visible)
        )
    configurable = (
        "That split is a firmware setting an owner can change, not a property "
        "of the processor."
        if split.get("configurable")
        else "That split is reported by the display driver rather than by a "
             "firmware setting this host publishes, so its size is read but "
             "not proven changeable here."
    )
    return (
        "Memory: {} visible to the operating system, {} reserved for graphics, "
        "{} accounted for. {}"
    ).format(visible, reserved, total or "an unknown total", configurable)


def _identity(driver, inventory: Dict[str, Any]) -> List[str]:
    try:
        board = driver.driver.board() if hasattr(driver, "driver") else driver.board()
    except (AttributeError, OSError, TypeError, ValueError):
        board = {}
    try:
        product = driver.product() or {}
    except (AttributeError, OSError, TypeError, ValueError):
        product = {}
    try:
        machine_class = driver.machine_class
    except (AttributeError, TypeError, ValueError):
        machine_class = "generic"
    memory = _gigabytes(inventory.get("memory_total_bytes"))
    try:
        split = (
            driver.driver.memory_split(inventory) if hasattr(driver, "driver")
            else driver.memory_split(inventory)
        ) or {}
    except (AttributeError, OSError, TypeError, ValueError):
        split = {}
    from .version import __version__

    lines = [
        "Machine: {} ({} class).".format(
            product.get("name") or board.get("model") or "unidentified computer",
            machine_class,
        ),
        _processor_line(board),
        # #144: asked which Vaelor version it runs, the Assistant answered
        # about service health — the version was nowhere in its facts. It is
        # a standing fact of exactly this brief's kind, and putting it here
        # also means a saved KV slot goes stale across an upgrade for the
        # right reason.
        "Vaelor version: {}.".format(__version__),
    ]
    memory_line = _memory_line(split, memory)
    if memory_line:
        lines.append(memory_line)
    try:
        operating_system = (
            driver.driver.operating_system() if hasattr(driver, "driver")
            else driver.operating_system()
        ) or {}
    except (AttributeError, OSError, TypeError, ValueError):
        operating_system = {}
    if operating_system.get("name"):
        lines.append("Operating system: {} ({}).".format(
            operating_system["name"],
            operating_system.get("support_label") or "support level unknown",
        ))
    return lines


def _capability_lines(capabilities: Dict[str, Any]) -> tuple:
    """Split capabilities into what this machine has and what it does not.

    Absent capabilities carry the driver's own reason. Silence is what makes a
    model guess: told nothing about fan control it assumes fan control, and
    then reports its absence as a fault.
    """
    present: List[str] = []
    absent: List[str] = []
    for key, label in CAPABILITY_LABELS.items():
        record = capabilities.get(key)
        if not isinstance(record, dict):
            continue
        if record.get("available"):
            present.append("- {}: available.".format(label))
        else:
            absent.append("- {}: not available. {}".format(
                label, _sentence(record.get("reason") or "")
            ).rstrip())
    return present, absent


def _limitation_lines(
    capabilities: Dict[str, Any], absent: List[str]
) -> List[str]:
    """Absences only, and only ones the driver actually reported.

    This list used to assert that the neural accelerator publishes no power
    telemetry. It does publish it, through ``amd-smi``, and stating otherwise
    in a standing brief was worse than saying nothing: it reads as settled and
    nobody re-checks it. Whether a reading exists is now decided by the tool,
    at the moment of asking, and never asserted here.
    """
    return list(absent)


def machine_brief(callbacks: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
    """Build the standing brief for this machine, capped and measured."""
    callbacks = callbacks or {}
    from .api_machine import machine_capabilities, platform_driver

    driver = platform_driver(callbacks)
    device_info = callbacks.get("device_info")
    current_data = callbacks.get("current_data")
    inventory_probe = callbacks.get("hardware_inventory")
    try:
        raw = (device_info() if device_info else {}) or {}
        metrics = (current_data() if current_data else {}) or {}
        inventory = (inventory_probe() if inventory_probe else {}) or {}
    except (AttributeError, OSError, TypeError, ValueError):
        raw, metrics, inventory = {}, {}, {}
    capabilities = machine_capabilities(callbacks, raw, metrics)
    present, absent = _capability_lines(capabilities)

    sections: List[str] = [
        # The second sentence is not padding. Thermal thresholds and the
        # VRAM/GTT reading notes used to stand here, so the model always had
        # them before it saw a number. They now arrive attached to the number
        # instead, and a model that does not know that could judge a reading
        # against a norm it invented. This says where the norms are.
        "VAELOR MACHINE BRIEF - standing facts about this machine. Live values "
        "are not here; read them with the tools listed at the end, which carry "
        "the thresholds and provenance for every figure they report.",
        "\n".join(_identity(driver, inventory)),
    ]
    if present:
        sections.append("This machine has:\n" + "\n".join(present))
    limitations = _limitation_lines(capabilities, absent)
    if limitations:
        sections.append(
            "This machine does not have, or cannot control, the following. "
            "These are design facts, not faults:\n" + "\n".join(limitations)
        )
    sections.append(
        # The tool map stays. FLM exposes no native tool calling
        # (``NPU_NATIVE_TOOL_CALLING = False``, and ``response_format`` was
        # re-measured inert on this tier), so there is no tool-schema channel
        # and prose is the only way the model learns these tools exist.
        # Deleting these 222 tokens would not move a cost; it would remove the
        # capability the whole tool surface was built for.
        # Filtered by what this machine actually has. See `available_tools`:
        # the brief said "Neural processing unit: not available" and then
        # offered `npu.status` in the same breath.
        "Ask for what you need:\n"
        + "\n".join("- {}: {}".format(name, purpose)
                    for name, purpose in available_tools(capabilities))
    )
    text = "\n\n".join(sections)
    truncated = len(text) > MAX_BRIEF_CHARS
    if truncated:
        text = text[:MAX_BRIEF_CHARS].rsplit("\n", 1)[0] + "\n[brief truncated]"
    return {
        "text": text,
        "estimated_size": estimate_tokens(text),
        # The character count is the one figure here that is not an estimate.
        # It is reported because the estimate has been wrong before and nobody
        # could see it: "920 tokens" reached VD-008 from an estimator that
        # over-counted by 38%, and the live value had drifted to 991 unnoticed.
        "size_chars": len(text),
        "size_limit": MAX_BRIEF_TOKENS,
        "size_unit": "tokens, estimated at {} characters per token".format(
            CHARS_PER_TOKEN
        ),
        "truncated": truncated,
        "trust": "vaelor_verified_machine_facts",
    }


def brief_system_prompt(prompt: str, context: Any) -> str:
    """Carry the standing brief in the system message, not as a JSON value.

    The brief used to ride inside the JSON context object as a string value.
    Every newline in it was then escaped to ``\\n`` and the whole thing quoted,
    which measured **785 prompt tokens for a 717-token brief** - 68 tokens of
    envelope, about 9%, buying nothing. Nothing is cached, so that envelope was
    re-prefilled on every request for the life of the appliance.

    It is appended to the existing system prompt rather than sent as a second
    system message on purpose: several chat templates on this tier accept only
    one system turn, and a second one is silently merged or dropped depending
    on the template. One message costs the same tokens and cannot be dropped.

    The brief is Vaelor's own verified machine facts
    (``trust: vaelor_verified_machine_facts``), not host output, so the system
    role is where it belongs. Nothing read from the machine - no log line, no
    configuration value - travels this way; those stay in the user turn as
    untrusted evidence.
    """
    text = _brief_text(context)
    if not text:
        return prompt
    return "{}\n\n{}".format(prompt, text)


def _brief_text(context: Any) -> str:
    if not isinstance(context, dict):
        return ""
    appliance = context.get("appliance")
    if not isinstance(appliance, dict):
        return ""
    brief = appliance.get("machine_brief")
    if isinstance(brief, dict):
        return str(brief.get("text") or "")
    return str(brief or "")
