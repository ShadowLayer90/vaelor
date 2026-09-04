"""One home for the words an owner uses to name a reading of this machine.

**Why this module exists, and it is not tidiness.** The same question - "which
readings answer this?" - was answered by two hand-written tables in two modules,
and they were widened by hand on every change:

* ``assistant_intents._INTENT_PHRASES`` decided which fact tools to **gather**.
* ``provider_runtime._TOPIC_KEYS`` decided which of the gathered facts survive
  the small model's context budget - which readings the model actually **sees**.

They are two stages of one pipeline, so a word in the first and not the second
means a reading is fetched off the hardware and thrown away before the model
reads it. Measured on the commit that created this module: of 21 ordinary
questions, **8 gathered a reading the filter discarded**. The battery is
`tests/test_shared_assistant_vocabulary.ORDINARY_QUESTIONS`, which re-derives
the figure on every run - it was quoted in five places and lived in none of
them until 2026-08-13. "What is my ping?"
fetched ``network.status`` and showed the model nothing but identity;
"how many gigabytes free?" fetched ``storage.status`` and did the same. That is
`LESSONS` pattern 11 - the measurement was taken and not read - produced by
`LESSONS` pattern 6, one idea in two places.

`FOUNDATION_EVIDENCE.md` records this pair as section 3 instance 12, and its
design note is the reason for the shape below rather than a wider test: *the
instances that stayed fixed are the ones where the two encodings were merged,
not the ones where both sides were edited to agree.* Editing both sides to
agree is what produced #188, in one commit, by an author who could see both.

**What is shared and what is not, which is the boundary worth reading.** A
term belongs here when it selects a **fact tool** - a reading that is fetched
only if some word asked for it. ``system.identity`` and ``system.telemetry``
are gathered for every question ever asked, so no word selects them and no word
for them belongs here; the filter keeps its own small list for those, and
`provider_runtime` says why beside it. That boundary is not a matter of taste:
a filter term for a fact tool the gather table does not know is a term that can
never fire, because the reading it admits was never fetched.

**Inflections are spelled out, not stemmed.** ``throttl`` used to stand for the
whole family here, matched by a prefix rule private to `assistant_intents`. The
moment this table was shared that became a defect rather than a shorthand:
`vaelor.phrase_match.mentions` anchors both ends, so ``throttl`` matched
"throttling" for the gatherer and nothing at all for the filter - the same
entry meaning two things in the two places it was read. `_CONTROL_PLANE_WORDS`
had already settled this question the same way, for the same reason.
"""

from __future__ import annotations

from typing import Dict, Tuple

from .telemetry_trend import TREND_PATTERN

#: Always gathered: identity and telemetry ground every answer. Nothing in
#: `TOOL_PHRASES` selects these, because nothing has to.
BASE_TOOLS: Tuple[str, ...] = ("system.identity", "system.telemetry")

#: Entries beginning ``re:`` are raw regular expressions, and only the
#: gatherer's matcher can read them - see `literal_phrases`.
REGEX_PREFIX = "re:"

# "running" means workloads in "is anything running", but temperature in
# "running hot". This is the one entry that has to be a regex: it matches the
# verb only when a thermal word does not follow it.
# Any run of short qualifiers may sit between the verb and the temperature
# word: "running a little hot", "running really very warm". An enumerated
# allowlist let those through to the workload tools.
RUNNING_NOT_THERMAL = (
    r"re:\brunning\b(?!(?:\s+\w+){0,3}\s+"
    r"(?:hot|hotter|warm|warmer|cool|cooler|cold|colder)\b)"
)

#: What an owner says, and the reading it asks for.
#:
#: Phrases are matched with word boundaries by both readers. The gatherer
#: (`assistant_intents`) allows a little filler inside a multi-word phrase, so
#: "running really hot" matches "running hot"; the filter
#: (`provider_runtime`) requires the words to be adjacent. That difference is
#: in the matchers and not in any entry here, which is what keeps this table
#: readable as vocabulary rather than as two vocabularies wearing one name.
TOOL_PHRASES: Dict[str, Tuple[str, ...]] = {
    "cooling.status": (
        "fan", "fans", "rpm", "cooling", "cool down", "airflow",
        "temperature", "temperatures", "temp", "temps", "thermal", "thermals",
        # Comparatives matter: "is the cpu hotter than usual" is the same
        # question as "is the cpu hot".
        "hot", "hotter", "hottest", "warm", "warmer", "warmest",
        "cool", "cools", "cooled", "cooler", "coolest", "cold", "colder",
        "heat", "heating", "overheat", "overheating",
        "overheated", "burning up", "too hot", "running hot",
        # Spelled out rather than stemmed - see the module docstring. The
        # prefix rule this replaces matched "throttling" in one reader and
        # nothing in the other.
        "throttle", "throttles", "throttled", "throttling",
        "degrees", "celsius",
    ),
    "lighting.status": (
        "rgb", "light", "lights", "lighting", "lamp", "led", "leds",
        "colour", "color", "colours", "colors", "brightness", "rainbow",
        "case light", "case lights",
    ),
    "display.status": (
        "oled", "display", "displays", "screen", "screens", "front panel",
    ),
    # The accelerator does the work on an AI appliance, so "why is this slow"
    # has to reach it. Without these a question about speed selected CPU
    # telemetry only, and the answer described a processor at 4% while the GPU
    # was pinned.
    "gpu.status": (
        "gpu", "gpus", "graphics", "radeon", "nvidia", "vram", "gtt",
        "video memory", "rocm", "vulkan", "amdgpu", "accelerator",
        "slow", "sluggish", "laggy", "stalling", "bottleneck", "utilisation",
        "utilization",
    ),
    "npu.status": (
        "npu", "neural", "xdna", "ipu", "inference accelerator", "tops",
        "accelerator", "slow", "sluggish", "bottleneck",
    ),
    "inference.status": (
        "model", "models", "llm", "inference", "context window", "loaded",
        "which model", "loaded model", "token", "tokens", "generation",
        "slow", "sluggish",
    ),
    # The progressive and past forms were in the *answer* table and not here,
    # so "is the system updating right now" gathered nothing and the branch
    # that would have answered it never ran. See `assistant_answer_topics`:
    # the answer path may only claim a reading this table can supply.
    "updates.status": (
        "update", "updates", "updated", "updating",
        "upgrade", "upgrades", "upgraded", "upgrading",
        "patch", "patches",
        "package", "packages", "reboot", "restart required", "out of date",
    ),
    "services.status": (
        "service", "services", "daemon", "health", "healthy", "unhealthy",
        "problem", "problems", "wrong", "broken", "failing", "degraded",
        # The filter knew these and the gatherer did not, so a question about
        # a failure admitted a service reading that had never been fetched.
        "failed", "failure", "issue", "issues",
        "is everything ok", "anything wrong", "shouldn't be", "should not be",
        RUNNING_NOT_THERMAL,
    ),
    # **The fault question, which had no tool.** VD-041 gave fault vocabulary
    # to the redirect suppression list, and the redirect duly stopped firing -
    # but nothing was added *here*, so "are there any active warnings or alerts
    # on this machine right now" selected only identity and telemetry. The
    # model was handed a question about alerts and no alert state, and did what
    # its prompt tells it to do with a question it has no facts for: it said
    # the question was outside what it covers, and named AI Chat. Reproduced
    # twice on the Z2. Widening a suppression list stops a redirect; it does
    # not gather evidence, and this is the tool that does.
    "health.status": (
        "warning", "warnings", "alert", "alerts", "alarm", "alarms",
        "error", "errors", "fault", "faults", "attention",
        "anything wrong", "is anything wrong", "all clear", "all good",
    ),
    "storage.status": (
        "storage", "disk", "disks", "drive", "drives", "nvme", "sd card",
        # The two removable devices this enclosure actually boots and stores
        # on. Both were in the filter and in no gather list, so a question
        # naming either admitted a storage reading nothing had fetched.
        "microsd", "usb",
        "space", "free space", "full", "capacity", "gigabytes", "gb free",
    ),
    # **The address and the name of the machine, which had no route here.**
    # "What is my IP address?" and "What is this machine's hostname?" selected
    # identity and telemetry only - neither of which carries either - so the
    # question reached a model with no network reading in front of it while
    # `network.status` held 192.168.0.50. Only "ethernet or wifi" phrasing
    # ever reached this tool, which is the vocabulary of someone who already
    # knows how the answer is stored.
    "network.status": (
        "network", "networks", "internet", "ethernet", "wifi", "wi-fi", "dns",
        "online", "offline", "connection", "connectivity", "latency", "ping",
        "ip", "ips", "ip address", "ipv4", "ipv6", "address", "addresses",
        "hostname", "host name", "subnet", "gateway", "mac address",
    ),
    # **Retained telemetry samples, which nothing was gathering.**
    # `assistant_answer_scope.past_time_answer` reads `metrics.history` to
    # decide between offering the retained samples and stating that no record
    # covers the time asked about. The tool is registered, the samples are
    # retained - and it appeared in no phrase list, so the fact key was never
    # present, the offer branch could not execute in production, and every
    # retrospective question got the refusal regardless of what this appliance
    # actually held. The tests for that branch supplied the samples the
    # product never gathered (LESSONS pattern 14).
    #
    # **These are also the widest ordinary-English words in this table**, and
    # `tests/test_assistant_scope_consequence.py` exists because of them: they
    # moved the AI Chat redirect the day they were added here, silently, which
    # is #190. Read that guard before adding another word of this kind.
    #
    # **The trend family is one entry and it is not written here.**
    # `telemetry_trend.TREND_PATTERN` is `TREND_PHRASES` rendered as one
    # alternation, and that module also computes the trend and owns the words
    # the answer reads back. Alpha 50 shipped the other arrangement: the
    # appliance invited "ask for a trend - whether something rose, fell, or
    # held steady" and then did not fetch this reading for "Did the CPU
    # temperature rise, fall, or hold steady?", "is it trending up or down"
    # or "over the last few minutes". The offer and the vocabulary that
    # satisfies it were different sets, which is #188 for the third time.
    #
    # One entry rather than forty literals, and the reason is arithmetic:
    # `matched_phrases` counts distinct phrases and two of them corroborate a
    # sentence into scope, so "when did the Roman Empire rise and fall" would
    # count `rise` and `fall` as two witnesses and stop reaching AI Chat.
    # Measured 2026-08-13: 4 of 12 world questions moved that way as literals,
    # none as one entry. `telemetry_trend`'s docstring carries the battery.
    "metrics.history": (
        "history", "trend", "trends", "over time", "ago", "yesterday",
        "last night", "overnight", "earlier", "last hour", "past hour",
        "so far today", "at any point",
        TREND_PATTERN,
    ),
    # ``installed`` asks what this machine *has*, which is the inventory. It
    # sat in the answer table alone, so "what is installed on this machine"
    # gathered nothing at all and was refused. ``install``/``installing`` name
    # the *act* and stay with `workloads.capabilities` below.
    "workloads.inventory": (
        "app", "apps", "application", "applications", "container", "containers",
        "docker", "compose", "stack", "model", "models", "llm", "local ai",
        "workload", "workloads", "installed", RUNNING_NOT_THERMAL,
    ),
    "workloads.capabilities": (
        "deploy", "deployment", "install", "installing", "hugging", "catalog",
        "blueprint",
    ),
    "jobs.recent": (
        "job", "jobs", "task", "tasks", "activity", "history", "failed",
        "failure", "retry", "last run",
        # Job state an owner reads off the Activity page. Filter-only before,
        # so "how is the download going" admitted a job reading nothing had
        # fetched.
        "progress", "download", "downloading",
        # ``installing`` is the running job, and it reached only
        # `workloads.capabilities` - so "is anything still installing" was
        # answered from the deployment catalogue with no job reading in hand.
        # A word may name two readings; ``history`` and ``model`` already do.
        "installing",
    ),
    "recovery.checkpoints": (
        "backup", "backups", "backed up", "checkpoint", "checkpoints",
        "restore", "recovery", "rollback", "snapshot",
    ),
}


def literal_phrases(tool: str) -> Tuple[str, ...]:
    """The phrases naming ``tool`` that a whole-word matcher can read.

    Regex entries are dropped rather than passed through, because
    `vaelor.phrase_match.mentions` escapes what it is given and would match
    the pattern's own characters. Dropping is the honest direction: the filter
    then admits a reading it cannot justify only where the gatherer used a
    lookahead, and `provider_runtime` records what that costs.
    """
    return tuple(
        phrase for phrase in TOOL_PHRASES.get(tool, ())
        if not phrase.startswith(REGEX_PREFIX)
    )


def fact_tools() -> Tuple[str, ...]:
    """Every reading a word can ask for, in table order."""
    return tuple(TOOL_PHRASES)
