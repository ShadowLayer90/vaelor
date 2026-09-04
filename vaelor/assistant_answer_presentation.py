"""Plain-language presentation helpers for deterministic Assistant answers."""

from __future__ import annotations

import logging
import re
from typing import Any, Dict, List, Sequence

from .answer_evidence import describe_missing
from .assistant_action_requests import detect_action_requests
from .byte_units import describe_gb

logger = logging.getLogger(__name__)


#: A JSON-style ``\uXXXX`` escape sitting *literally* in prose - the six
#: characters backslash-u-0-0-b-0, not the degree sign they name.
#:
#: A managed-local model answers in natural language, and a connected model can
#: double-escape a value, so "34.6°C" reached the card as those literal
#: characters where a deterministic answer builds ``°`` directly and never
#: shows one. This is applied only to model-authored answer text (see
#: :meth:`DeploymentAgent._model_answer`); deterministic answers carry no such
#: escape, so it never has to run on them.
_JSON_UNICODE_ESCAPE = re.compile(r"\\u([0-9a-fA-F]{4})")


def _decode_unicode_escape(match: "re.Match[str]") -> str:
    codepoint = int(match.group(1), 16)
    # A lone surrogate escape is left exactly as written: decoding it in
    # isolation would produce an unpaired surrogate and an unusable string.
    if 0xD800 <= codepoint <= 0xDFFF:
        return match.group(0)
    return chr(codepoint)


def normalize_unicode_escapes(text: Any) -> str:
    """Turn literal JSON ``\\uXXXX`` escapes in ``text`` into real characters.

    The degree sign is the one a live tester saw ("34.6\\u00b0C"), and every
    other symbol a model spells the same way - en dash ``\\u2013``, micro
    ``\\u00b5`` - is corrected by the same pass. Real characters are untouched,
    so the function is idempotent and safe to run on an already-clean answer.
    """
    return _JSON_UNICODE_ESCAPE.sub(_decode_unicode_escape, str(text or ""))


def normalize_model_answer(result: Any) -> Any:
    """Repair literal ``\\uXXXX`` escapes in a model answer dict's text.

    Applied only to model-authored answers (connected and managed-local);
    deterministic answers build real characters and never carry an escape.
    """
    if isinstance(result, dict) and "answer" in result:
        result["answer"] = normalize_unicode_escapes(result.get("answer"))
    return result

# Screens an answer can name, with the hash route that actually reaches them.
# Every route here is one of the pages in `frontend/src/lib/navigation.ts`;
# naming a destination in prose without emitting a route is what left users
# reading "changes are made on System > Case lighting" with no way to get
# there. Ordered most specific first so "System > Cooling" is never reduced to
# a bare "System" match.
_DESTINATIONS: tuple[tuple[str, str, str, bool], ...] = (
    ("System > Case lighting", "#/system", "Case lighting", True),
    ("System > Cooling", "#/system", "Cooling", True),
    ("System > Hardware & services", "#/system", "Hardware & services", True),
    ("Workloads > Install", "#/workloads", "Install", True),
    ("Assistant > Agents", "#/assistant/agents", "Run history", True),
    ("Remote console", "#/kvm", "Remote console", True),
    # Every answer that sends a user to AI Chat named it in prose and emitted
    # no route, so "ask it in AI Chat" was a dead end on a screen the user had
    # to go and find. It is a real page; it gets a real link.
    ("AI Chat", "#/ai-chat", "AI Chat", True),
    ("app manager", "#/workloads", "App manager", True),
    ("Workloads", "#/workloads", "Workloads", False),
    ("Activity", "#/activity", "Activity", False),
    ("Fleet", "#/fleet", "Fleet", False),
    ("Admin", "#/admin", "Admin", False),
    ("Overview", "#/", "Overview", False),
)

_COMPILED_DESTINATIONS = tuple(
    (
        re.compile(
            r"(?<![\w>])" + re.escape(name).replace(r"\ ", r"\s+") + r"(?![\w<])",
            re.IGNORECASE if insensitive else 0,
        ),
        name,
        route,
        control,
    )
    for name, route, control, insensitive in _DESTINATIONS
)


def _destinations_in(text: str) -> List[Dict[str, str]]:
    """Return every navigable screen named in ``text``, most specific first."""
    remaining = str(text or "")
    found: List[Dict[str, str]] = []
    for pattern, name, route, control in _COMPILED_DESTINATIONS:
        match = pattern.search(remaining)
        if match is None:
            continue
        found.append({"label": name, "route": route, "control": control})
        # Blank the matched span so "System > Cooling" is not counted again by
        # a shorter destination overlapping the same words.
        remaining = remaining[: match.start()] + " " * (match.end() - match.start()) + remaining[match.end():]
    return found


def navigation_steps(
    answer: Dict[str, Any], actions: Sequence[Dict[str, str]] = ()
) -> List[Dict[str, str]]:
    """Return the routes for every destination this answer names.

    The human sentence is kept as written; this is the machine-readable half of
    it, so the frontend can render a link instead of asking the user to find
    the screen themselves.
    """
    texts: List[str] = [str(answer.get("answer", ""))]
    suggested = answer.get("suggested_actions") or []
    if isinstance(suggested, (list, tuple)):
        texts.extend(str(item) for item in suggested)
    steps: List[Dict[str, str]] = []
    for action in actions or ():
        route = str(action.get("route", ""))
        if route:
            steps.append({
                "label": str(action.get("screen", "")),
                "route": route,
                "control": str(action.get("control", "")),
            })
    for text in texts:
        steps.extend(_destinations_in(text))
    unique: List[Dict[str, str]] = []
    seen: set[tuple[str, str]] = set()
    for step in steps:
        key = (step["route"], step["control"])
        if key in seen:
            continue
        seen.add(key)
        unique.append(step)
    return unique[:6]


# What a person calls the media Vaelor found, keyed by the detected media kind.
# "microsd /dev/mmcblk0 has 221.5 GB free" was true and unreadable: the device
# path is an operator detail, not an answer to "how much space do I have".
_MEDIA_NAMES = {
    "microsd": "the memory card",
    "micro sd": "the memory card",
    "sd": "the memory card",
    "mmc": "the memory card",
    "nvme": "the internal SSD",
    "ssd": "the internal SSD",
    "sata": "the attached drive",
    "usb": "the USB drive",
}
_ORDINALS = ("the first", "the second", "the third", "the fourth")


def _media_name(kind: Any, position: int, duplicates: int) -> str:
    name = _MEDIA_NAMES.get(str(kind or "").strip().lower(), "the storage device")
    if duplicates < 2:
        return name
    ordinal = _ORDINALS[position] if position < len(_ORDINALS) else "another"
    return "{} {}".format(ordinal, name.split(" ", 1)[1] if name.startswith("the ") else name)


def connected_model_failure_answer() -> str:
    return (
        "The selected AI connection did not answer this request, so I cannot answer it "
        "reliably without guessing. Confirm the selected model is loaded and retry, or "
        "choose a smaller model. Built-in appliance questions about cooling, lighting, "
        "display, storage, network, services, workloads, and updates remain available."
    )


#: Prefixed to a built-in answer that stands in for a model that did not reply.
#:
#: **The sentence has to be in the answer, not in the evidence.** Measured live
#: 2026-08-11: every model answer was being cancelled, and the owner saw a
#: fluent built-in reply with "the selected AI connection did not answer
#: correctly" available only inside a collapsed Evidence panel. Asked for the
#: Vaelor version and whether the machine has a GPU, the reply was a list of
#: running services - confident, on-topic in appearance, and answering neither
#: question, with nothing visible to say why.
#:
#: That is #148's defect on a new surface: the card knew the operation had
#: failed and described it as something else. A reader must not have to open a
#: panel to learn that the thing which answered them is not the thing they
#: asked.
#: **Not "in time".** The first wording asserted a timeout, and this prefix is
#: applied for every failure the caller catches: connection refused when the
#: container is stopped, an HTTP rejection, malformed JSON, a server that
#: offers no model at all. Telling an owner their model "did not answer in
#: time" when the engine is not running sends them to wait and retry instead
#: of to start it - which is #148's defect, described as something else,
#: inside the fix for #148's defect.
MODEL_DID_NOT_ANSWER_PREFIX = (
    "The AI model did not answer, so this is what Vaelor can tell you from "
    "its own live readings instead - it may not cover what you asked. The "
    "model's own status is on Assistant setup."
)


def with_model_failure_stated(answer: str) -> str:
    """Put the model's failure in front of the built-in answer replacing it."""
    text = str(answer or "").strip()
    if not text:
        return MODEL_DID_NOT_ANSWER_PREFIX
    if text.startswith(MODEL_DID_NOT_ANSWER_PREFIX):
        return text
    return "{}\n\n{}".format(MODEL_DID_NOT_ANSWER_PREFIX, text)


# What this assistant is for. A question sharing none of these has no
# appliance facts behind it, whatever the model does.
_APPLIANCE_SIGNALS = frozenset({
    "appliance", "vaelor", "node", "pironman", "raspberry", "pi", "box",
    "fan", "cooling", "cool", "temperature", "temp", "thermal", "cpu", "gpu",
    "ram", "memory", "swap", "storage", "disk", "drive", "nvme", "microsd",
    "sd", "usb", "space", "network", "internet", "wifi", "ethernet", "dns",
    "ip", "oled", "display", "screen", "rgb", "led", "light", "lighting",
    "update", "upgrade", "package", "service", "services", "docker",
    "container", "compose", "workload", "app", "apps", "model", "models",
    "backup", "checkpoint", "restore", "recovery", "cluster", "worker",
    "job", "jobs", "log", "logs", "port", "power", "boot", "reboot",
    "shutdown", "uptime", "health", "status", "install", "installed",
    "running", "config", "configuration", "setting", "settings",
    # **Fault vocabulary.** VD-041 added these to the *redirect* suppression
    # list in `assistant_intents` and stopped there, so the sibling table that
    # decides what this Assistant is *for* still had no word for something
    # being wrong. That gap is what let "What is causing the warning?" be
    # classified as general knowledge and sent to a model, which invented one.
    "error", "warning", "alert", "fault", "alarm", "issue", "problem",
    "failure", "diagnostic", "crash", "exception", "timeout", "symptom",
    # **The machine itself.** This list named every part of the appliance -
    # cpu, fan, drive, network - and had no word for the whole, so
    # "is there anything wrong with this machine" was not an appliance
    # question. That is close to the most appliance-shaped sentence a person
    # can type, and it is the VD-051 case verbatim.
    #
    # The same drift as the fault vocabulary above, one table further on:
    # `_CONTROL_PLANE_WORDS` in `assistant_intents` has carried "machine",
    # "hardware", "sensor" and "system" since VD-041, and this table - its
    # sibling - never received them. One list suppresses a redirect, the other
    # decides what this Assistant is *for*; they look interchangeable and are
    # not, which is how they drift apart one addition at a time.
    #
    # Measured before widening, because this predicate is read on a **gate**
    # as well as on a suppression. `assistant_intents` and
    # `assistant_scope_guard` consult it to *stop* a redirect, where a false
    # positive costs nothing; `chat_appliance_scope` consults it to **decline**
    # a question on AI Chat, where a false positive blocks the very thing that
    # surface is for. Only the second side prices a mistake, so the second side
    # is what these words were scored against.
    #
    # Against the suite's own corpora - 85 appliance phrasings, 56
    # general/RAG/conversational, a 20-phrase battery written around these
    # words in their general sense, and a 10-phrase battery attaching a
    # possessive to them ("how does my washing machine work") - these four cost
    # **two** false declines on AI Chat and raise recall from 70/85 to 71/85 on
    # the general corpora and from 3/8 to 7/8 on the possessive one. The two
    # are "how does my washing machine work" and "how does my dishwasher device
    # sense a full load", and they are the stated price of a control plane
    # whose own UI calls the thing it manages a machine.
    #
    # **"system" was measured and left out, and it is the one that matters.**
    # It looks like the most obvious word on the list and it is the most
    # expensive: solar system, nervous system, immune system, tax system,
    # school system, zodiac system. Alone it took AI Chat's false declines from
    # 1 to 6 on the possessive battery for a single phrasing of recall. That is
    # the "general sense dominates" test VD-041 already used to reject `plan`,
    # `policy`, `profile`, `console`, `desktop`, `remote` and `draft`.
    #
    # **"unit" was measured and left out.** It gained one appliance phrasing
    # already covered by "asset" and "tag", and cost a false decline on "what
    # is the smallest unit of matter" plus four lost redirects.
    "machine", "hardware", "device", "sensor",
})


#: Appliance vocabulary the single-token table above cannot express.
#:
#: **The list is written in nouns and people ask in verbs.** "Is my data
#: backed up?" returned False from :func:`is_appliance_question` while "Are my
#: backups working?" returned True - the same question, one word apart, and
#: the first one walked straight past the scope guard because the guard is
#: never consulted about a question it does not think is ours. `_signal_tokens`
#: cannot close that: it reduces regular plurals, and "backed" does not reduce
#: to "backup".
#:
#: Phrases go here rather than single words going into `_APPLIANCE_SIGNALS`,
#: because the cost of this predicate is paid on a **gate** - a false positive
#: declines a question on AI Chat - and "data" or "backed" alone would decline
#: "explain how data compression works". "backed up" as a phrase does not.
#: Only entries that decide something belong here. "out of space" and
#: "powered on" were written and removed on the spot: "space" and "power" are
#: already single tokens above, so neither phrase could ever be the reason
#: this function returned True, and a rule that cannot decide is decoration
#: that reads as a guard (LESSONS pattern 2).
_APPLIANCE_PHRASES: tuple[re.Pattern[str], ...] = (
    re.compile(r"\bback(?:ed|ing)?\s+up\b", re.IGNORECASE),
)


def _signal_tokens(message: str) -> set[str]:
    """Words in ``message``, with regular plurals reduced to their singular.

    The vocabulary above is written in the singular, and it was compared against
    raw words. So "light" was in scope and "lights" was not - which is how
    "make the case lights blue" was tagged out of scope by boilerplate that
    lists lighting as in scope. The same hole swallowed "fans", "drives",
    "leds", "screens" and "temperatures".
    """
    tokens: set[str] = set()
    for word in re.findall(r"[a-z0-9]+", str(message).lower()):
        tokens.add(word)
        for suffix in ("es", "s"):
            if word.endswith(suffix) and len(word) > len(suffix) + 1:
                tokens.add(word[: -len(suffix)])
    return tokens


def is_appliance_question(message: str) -> bool:
    """Whether a question is about this machine at all.

    A bare "my" used to be enough, which made "Draft an email to my landlord"
    an appliance question. It never added reach either: every possessive that
    does name part of this box - "my CPU", "my drive" - already carries the
    signal word itself.

    Asking this surface to change something it owns is in scope too. It may not
    be able to perform the change from here, and saying so is the right answer -
    but "I cannot do that here" and "that is not something I cover" are
    different statements, and only one of them was true of "make the case lights
    blue".

    **Before adding a word above, know which side it is read on.** Two callers
    use this to *suppress* something - `assistant_intents.is_out_of_scope_question`
    and `assistant_scope_guard.misrouted_refusal` - where a false positive costs
    nothing. `chat_appliance_scope.asks_about_this_machine` uses it to
    **decline** a question on AI Chat, where a false positive blocks the one
    thing that surface exists for. A widening scored only against the first two
    is unscored, and the general sense of a candidate word is what decides it:
    "system" reads as solar, nervous, immune, tax and school far more often
    than as this box.
    """
    if _signal_tokens(message) & _APPLIANCE_SIGNALS:
        return True
    if any(pattern.search(str(message)) for pattern in _APPLIANCE_PHRASES):
        return True
    return bool(detect_action_requests(message))


# ---------------------------------------------------------------------------
# "Is this an appliance identity / version question?" - one detector, consulted
# by both the out-of-scope gate (`assistant_intents.is_out_of_scope_question`)
# and the identity answer (`assistant_answer_scope.identity_answer`).
#
# **This lives here, not in either caller, because both must not disagree.**
# LESSONS 6: "what version am I on?" was answered by `identity_answer` and
# *simultaneously* redirected to AI Chat by `is_out_of_scope_question`, on the
# live Pi during the #205 terminal gate, because the markers below lived only
# in `assistant_answer_scope` where the out-of-scope gate could not see them.
# Copying the regexes into `assistant_intents` would have been the two-lists-
# that-drift anti-pattern (#98). `assistant_answer_presentation` is the lowest
# module both callers already import without a cycle, so the detector moved
# here and every caller consults this one definition.

#: Words that mean "which Vaelor is this" rather than "what is it doing".
#:
#: **A bare `\bversion\b` was the first attempt and it was wrong.** It
#: answered "What version of Docker is installed?", "Which model version is
#: the assistant using?" and "Is there a newer version of Vaelor I should
#: install?" all with the Vaelor version - the last one while `updates.status`
#: sat unread in the facts. That is #159 committed by its own fix: a fluent
#: answer to a question nobody asked.
#:
#: Markers that name Vaelor as the subject of the version question. These are
#: unambiguous on their own, so nothing else in the sentence can exclude them -
#: which is what lets "which vaelor version is the model running under" be
#: answered instead of thrown away.
#:
#: **Every marker in this tier must name Vaelor, and one did not.** LESSONS 6 /
#: #159: `\bwhat (?:release|build) (?:is|am i)\b` lived here and names no
#: product, so `what release is the next Ubuntu?` and `what build is the new
#: Windows?` were answered "This machine is running Vaelor …" - the exact
#: fluent-answer-to-a-question-nobody-asked defect this module's version tier
#: was built to close, committed one word over in the release/build tier. It
#: matched here *before* `_OTHER_PRODUCT_NAMED` could exclude "ubuntu"/"debian",
#: which are in that list. The unqualified "what release/build is this" phrasing
#: now lives in `_VERSION_UNQUALIFIED`, subject to the other-product exclusion,
#: exactly like "what version is this". A marker that does not name Vaelor does
#: not belong in this tier.
_VAELOR_NAMED = (
    re.compile(r"\b(?:vaelor|control plane)\s+(?:version|release|build)\b"),
    re.compile(r"\b(?:version|release|build)\s+of\s+vaelor\b"),
    re.compile(r"\bwhich vaelor\b|\bwhat vaelor\b"),
    re.compile(r"\brunning vaelor\b"),
)

#: "What version is this?" - and "what release/build is this", "what build am
#: I on" - with nothing else named means this appliance. On its own it says
#: nothing about *whose* version, so it is the tier that has to yield when
#: another product is in the sentence: "what release is the next Ubuntu?" names
#: `ubuntu`, so `_OTHER_PRODUCT_NAMED` excludes it, and "what release is this?"
#: names no other product, so it stays an identity question. "release"/"build"
#: live here rather than in `_VAELOR_NAMED` for that reason (LESSONS 6 / #159).
_VERSION_UNQUALIFIED = re.compile(
    r"\b(?:what|which) (?:version|release|build) (?:is this|am i (?:running|on))\b"
)

#: Any product that is not Vaelor, named anywhere. Only ever consulted for the
#: unqualified marker above.
#:
#: **Adjacency was the wrong test and review proved it in one line.** The
#: previous exclusion required "version of docker" or "docker version", so
#: `what version is this kernel?` produced no adjacency, reached identity, and
#: was answered with Vaelor's version - #159 for the third time, on a wording
#: the module's own docstring names. Widening adjacency would only move the
#: hole. Splitting the markers by whether Vaelor is *named* removes it: an
#: explicit marker needs no exclusion, and the unqualified one excludes on the
#: mere presence of another product, which is exactly what "unqualified" means.
#:
#: **`gpu` and `npu` are deliberately absent**, though they are the obvious
#: additions. They are hardware *on this machine*, not other products whose
#: version anyone asks for, and they appear in exactly the compound question
#: this module was written for. Including them made "what version is this, and
#: does it have a GPU?" unanswerable - trading #159 for #144 rather than
#: fixing either.
_OTHER_PRODUCT_NAMED = re.compile(
    r"\b(?:docker|compose|kernel|python|ubuntu|debian|firmware|bios|rocm|"
    r"llama\.?cpp|model|npm|systemd)\b"
)

#: An update question is about a version this machine does not have, so the
#: identity answer is the wrong one however the sentence is phrased.
_UPDATE_MARKERS = re.compile(
    r"\bnewer\b|\bupdate\b|\bupgrade\b|\blatest\b|\bout of date\b|\bshould i install\b"
)


def asks_about_identity(message: str) -> bool:
    """Whether this question asks what this machine and software *are*.

    Two tiers, because "whose version" is the whole question. Naming Vaelor
    settles it however many other products the sentence mentions; not naming
    anything settles it only if nothing else is named either.

    **One detector, two readers (LESSONS 6).** This is the single source both
    `is_out_of_scope_question` and `identity_answer` consult, so the out-of-
    scope gate and the identity answer cannot disagree about whether a question
    like "what version am I on?" is appliance identity. Do not reimplement the
    markers in a caller - that reintroduces the #205 / #98 drift.
    """
    lower = " ".join(str(message or "").lower().split())
    if _UPDATE_MARKERS.search(lower):
        return False
    if any(pattern.search(lower) for pattern in _VAELOR_NAMED):
        return True
    return bool(
        _VERSION_UNQUALIFIED.search(lower)
        and not _OTHER_PRODUCT_NAMED.search(lower)
    )


# ---------------------------------------------------------------------------
# "Which OS is on this box?" and "how many CPU cores?" - standing host facts.
#
# **LESSONS 5 / #144 / #205, live on the Pi deep pass.** "What operating system
# version is installed?" routed to `workloads.inventory` on the word `installed`
# and was answered with the *app list* - a fluent answer to a different
# question, with evidence citations. "How many CPU cores does this machine
# have?" reached the CPU-telemetry branch and was answered with usage and
# temperature, never the core count. Both facts stand in the machine brief
# (`assistant_machine_brief._identity` writes the "Processor:" and "Operating
# system:" lines), so the router already holds the answer; what was missing was
# a route from these phrasings to it.
#
# These detectors live here beside `asks_about_identity`, and for the same
# reason it does: the answer path (`assistant_answer_scope.host_fact_answer`)
# and the scope gate (`assistant_intents.is_out_of_scope_question`) must not
# disagree about whether one of these is appliance business. Do not reimplement
# the markers in a caller (#98).

#: Whether a question points at *this* machine rather than the world.
#:
#: **#205 H1, the adversarial recheck.** A host-fact detector that fires on the
#: subject alone reintroduces the exact LESSONS-5 class it was built to fix:
#: "how many cores does an M4 have" and "how many cores does a Threadripper
#: 7995WX have" were answered "This machine's processor is …", with the redirect
#: suppressed so the wrong fact stood. Both detectors below now require this
#: marker, so a count/OS question about *some other* CPU or OS is not claimed.
_ABOUT_THIS_MACHINE = re.compile(
    r"\bmy\b|\bmine\b|\bthis\b|\bour\b|\bours\b|\bhere\b|\bappliance\b|"
    r"\bthis (?:box|node|pi|host|device|machine|server)\b"
)

#: A named distro/OS. The subject alone is never enough - "how does an operating
#: system work", "who invented linux", "when was ubuntu released" all name a
#: subject and are world questions - so `asks_about_os_version` additionally
#: requires a this-machine marker and rejects a how/explain/why opener.
_OS_SUBJECT = re.compile(
    r"\boperating system\b|\bos\b|\bubuntu\b|\bdebian\b|\blinux\b|"
    r"\bdistro\b|\bdistribution\b"
)
#: The OS-specific this-machine markers. ``installed`` and ``what/which os`` are
#: about-this on their own ("what operating system version is installed"); the
#: rest reuse `_ABOUT_THIS_MACHINE`. ``running`` was here and is gone: it fired
#: on "how does linux handle running processes" and "explain how linux schedules
#: running threads" (#205 H1), and every real "which OS is running here"
#: phrasing already carries ``this``/``here`` instead.
_OS_ABOUT_THIS = re.compile(
    _ABOUT_THIS_MACHINE.pattern + r"|\binstalled\b|\bwhat os\b|\bwhich os\b"
)
#: An OS question that is really an update question ("os updates installed",
#: "latest ubuntu", "is my distro up to date") is not an identity question - the
#: update branch answers it. "up to date"/"up-to-date" is #205 L2.
_OS_NOT_IDENTITY = re.compile(
    r"\bupdat|\bupgrad|\bpatch|\bnewer\b|\blatest\b|\bup[\s-]?to[\s-]?date\b"
)
#: A world/definitional opener. "how does linux …", "explain how …", "why does
#: …" are general knowledge even when a possessive slips in, so they never route
#: to the standing OS fact (#205 H1).
_WORLD_OPENER = re.compile(
    r"^\s*(?:how\s+(?:do|does|is|are|can|should|would|might)|explain\b|why\b)"
)


def asks_about_os_version(message: str) -> bool:
    """Whether this asks what operating system *this appliance* runs."""
    lower = " ".join(str(message or "").lower().split())
    if _OS_NOT_IDENTITY.search(lower):
        return False
    if _WORLD_OPENER.match(lower):
        return False
    if not _OS_SUBJECT.search(lower):
        return False
    return bool(_OS_ABOUT_THIS.search(lower))


#: A question counting the processor's cores/CPUs, told apart from a question
#: about CPU *load* or *temperature* ("how hot is my cpu", "cpu usage"), which
#: the telemetry branch answers. The count word (`cores`, `cpus`, `processors`)
#: has to be present - bare "cpu" is a load question.
_CPU_COUNT = re.compile(
    r"\bhow many\b[^.?!]{0,30}\b(?:cores?|cpus|processors)\b"
    r"|\b(?:number|count)\s+of\s+(?:cpu\s+|processor\s+)?(?:cores?|cpus|processors)\b"
    r"|\bcore count\b"
    r"|\b(?:cpu|processor)\s+cores?\b"
)


def asks_about_cpu_count(message: str) -> bool:
    """Whether this asks how many cores *this machine* has.

    The this-machine marker is required (#205 H1): "how many cores does an M4
    have" names another CPU and must not be answered with this appliance's core
    count. A bare count question with no marker is left to route elsewhere
    rather than claimed - the safe direction against a world question.
    """
    lower = " ".join(str(message or "").lower().split())
    if not _CPU_COUNT.search(lower):
        return False
    return bool(_ABOUT_THIS_MACHINE.search(lower))


def asks_about_host_facts(message: str) -> bool:
    """OS-version or CPU-core-count question - one gate for the scope reader.

    `is_out_of_scope_question` consults this to *suppress* the AI Chat redirect
    for a question the brief can answer, exactly as it consults
    `asks_about_identity`. `host_fact_answer` answers it. Two readers, one
    predicate (LESSONS 6).
    """
    return asks_about_os_version(message) or asks_about_cpu_count(message)


def out_of_scope_after_model_failure() -> dict[str, Any]:
    """Say both true things, because both happened.

    Telling someone their model is unloaded or too large when the real reason
    is that tomorrow's weather is not a fact about their appliance sends them
    to debug hardware that is working. But this answer is only ever produced
    *after* the connected model raised, so saying scope is the only problem -
    "not a problem with the connected model" - states as fact something that is
    false, and hides a real outage behind a routing decision. Scope is why
    there is no built-in answer to fall back to; the failure is still a failure,
    and the reader is told about both.
    """
    return {
        "answer": (
            "The selected AI connection did not answer this request, and this question "
            "is also outside what I cover. I answer questions about this appliance - its "
            "cooling, lighting, display, storage, network, services, workloads, updates, "
            "and jobs - so there is no built-in answer for me to fall back on here. Ask "
            "it in AI Chat, the general chat destination, which can also cite your own "
            "documents. If you expected an answer from the model, confirm it is loaded "
            "and retry."
        ),
        "evidence": [{
            "source": "assistant.scope",
            "summary": (
                "The connected model did not answer, and the question has no appliance "
                "facts behind it."
            ),
        }],
    }


def general_knowledge_model_failure() -> dict[str, Any]:
    return {
        "answer": connected_model_failure_answer(),
        "evidence": [{
            "source": "assistant.fallback",
            "summary": "The selected AI connection did not answer correctly, so Vaelor did not guess or substitute unrelated appliance data.",
        }],
    }


def _largest_volume_per_device(storage: Any) -> list[tuple[str, dict[str, Any]]]:
    """Group volumes by physical device and keep the one that sizes it."""
    if not isinstance(storage, dict):
        storage = {}
    volumes = storage.get("volumes", [])
    if not isinstance(volumes, list):
        volumes = []
    grouped: dict[str, list[dict[str, Any]]] = {}
    for item in volumes:
        if not isinstance(item, dict):
            continue
        device_id = str(
            item.get("device_id") or item.get("path") or item.get("id") or "storage"
        )
        grouped.setdefault(device_id, []).append(item)
    return [
        (
            device_id,
            max(mounts, key=lambda item: float(item.get("total_bytes", 0) or 0)),
        )
        for device_id, mounts in grouped.items()
    ]


def storage_device_detail(storage: Any) -> str:
    """Operator-facing mapping from the plain name back to the device path.

    Kept out of the answer itself but not discarded: an operator diagnosing a
    failing card still needs to know which block device it is.
    """
    entries = _largest_volume_per_device(storage)
    kinds = [str(volume.get("kind", "")).lower() for _device, volume in entries]
    return "; ".join(
        "{} = {} ({})".format(
            _media_name(
                volume.get("kind"),
                kinds[:index].count(str(volume.get("kind", "")).lower()),
                kinds.count(str(volume.get("kind", "")).lower()),
            ),
            device_id,
            volume.get("kind") or "unknown media",
        )
        for index, (device_id, volume) in enumerate(entries)
    )


#: A reserve worth mentioning. Below this the sentence would be pedantry; at
#: or above it the arithmetic visibly fails to add up on screen, which is what
#: sent a tester looking for a conversion bug that was not there.
_RESERVE_WORTH_STATING = 1_000_000_000


def _reserve_clause(volume: Any) -> str:
    """Say why free and total do not account for each other, when they do not.

    The Pi showed "220.7 GB free" in one place and, from "20.7 GB used of
    251.7 GB", an implied 231.0 GB free in another. Both were right: the first
    is what may still be written, the second includes the filesystem's reserved
    blocks. Ten gigabytes of unexplained difference between two numbers on one
    screen is indistinguishable from a bug, so it is explained.
    """
    reserved = volume.get("reserved_bytes") if isinstance(volume, dict) else None
    if not isinstance(reserved, (int, float)) or isinstance(reserved, bool):
        return ""
    if reserved < _RESERVE_WORTH_STATING:
        return ""
    # **Opens no sentence and closes none**, which is the third spelling of
    # this clause and the first that does not depend on what is printed around
    # it.
    #
    # It first opened with a bare space and ended with a full stop, while
    # `storage_summary` appends one of its own - so all three storage answers a
    # live tester saw read "...251.7 GB A further 10.3 GB is reserved... nor in
    # use..", with no break where a sentence starts and two full stops where it
    # ends. The repair moved the stop to the front, which fixed the single
    # volume and *moved* the defect rather than removing it (#192): with two
    # volumes, `storage_summary` joins the summaries with "; ", so the list ran
    # "...251.7 GB. A further 10.3 GB is reserved..., so it counts as neither
    # free nor in use; the internal SSD has 900.0 GB free of 1000.0 GB." - a
    # semicolon list broken across a sentence boundary, with the SSD's clause
    # hanging off a sentence about the memory card.
    #
    # A parenthetical owns no sentence punctuation at either end, so it reads
    # the same first, last or alone, and `storage_summary` keeps every stop -
    # which is what the comment here has claimed since the first repair.
    return (
        " (a further {} is reserved by the filesystem for the operating "
        "system, so it counts as neither free nor in use)"
    ).format(describe_gb(reserved))


def storage_summary(storage: Any) -> str:
    """Describe capacity in the user's terms, without a device path.

    "Free" here means *what can still be written*, which is the figure that
    answers the question people ask. It is not ``total - used``: on Linux those
    differ by the reserved blocks, and stating one while a sibling surface
    states the other is how the same volume acquired two free-space figures ten
    gigabytes apart.
    """
    if not isinstance(storage, dict):
        storage = {}
    entries = _largest_volume_per_device(storage)
    kinds = [str(volume.get("kind", "")).lower() for _device, volume in entries]
    summaries = []
    for index, (_device_id, volume) in enumerate(entries[:4]):
        kind = str(volume.get("kind", "")).lower()
        summaries.append(
            "{} has {} free of {}{}".format(
                _media_name(
                    volume.get("kind"), kinds[:index].count(kind), kinds.count(kind)
                ),
                describe_gb(volume.get("free_bytes") or 0),
                describe_gb(volume.get("total_bytes") or 0),
                _reserve_clause(volume),
            )
        )
    if not summaries:
        count = len(storage.get("devices") or [])
        return "Vaelor detected {} storage device{}.".format(
            count, "" if count == 1 else "s"
        )
    detail = storage_device_detail(storage)
    if detail:
        logger.debug("storage answer devices: %s", detail)
    details = "; ".join(summaries)
    return details[:1].upper() + details[1:] + "."


#: How an IPv4 address is labelled by each of the two enumerators. `ip -j`
#: emits the string "inet"; the sysfs fallback emits the same string, and a
#: raw `AF_INET` constant is accepted so a future third source cannot silently
#: produce an interface with no address.
_IPV4_FAMILIES = frozenset({"inet", 2})


def _interface_sentence(interface: Dict[str, Any]) -> str:
    """One interface, named, with whatever address it actually holds."""
    name = str(
        interface.get("name") or interface.get("interface") or ""
    ).strip()
    if not name:
        return ""
    addresses = [
        str(entry["address"])
        for entry in interface.get("addresses") or []
        if isinstance(entry, dict)
        and entry.get("address")
        and entry.get("family") in _IPV4_FAMILIES
    ]
    if not addresses:
        return "{} (no IPv4 address is assigned)".format(name)
    return "{} at {}".format(name, ", ".join(addresses[:2]))


def network_summary(network: Any) -> str:
    """Name the active interfaces *and* the addresses they hold.

    "The active network interface is eth0" is true, and it is not an answer to
    "what is my IP address?" - which is the question this branch is reached
    by. The address was in `network.status` the whole time, one key below the
    interface name that was being read out instead: a fluent answer to a
    question nobody asked, which is the shape
    :mod:`vaelor.assistant_answer_scope` exists to stop.

    An empty interface list is never reported as "this machine has no
    network". :mod:`vaelor.network_interfaces` distinguishes "nothing is up"
    from "the enumeration failed" with `collected`, and that distinction is
    the whole reason it does (LESSONS pattern 8).
    """
    if not isinstance(network, dict):
        network = {}
    interfaces = network.get("interfaces")
    described = [
        _interface_sentence(item)
        for item in (interfaces if isinstance(interfaces, list) else [])
        if isinstance(item, dict)
        and (item.get("up") or str(item.get("state", "")).lower() == "up")
    ]
    named = [sentence for sentence in described if sentence][:4]
    if named:
        return "The active network interface{} {}.".format(
            "s are" if len(named) != 1 else " is", ", ".join(named)
        )
    if network.get("collected") is False:
        return describe_missing(
            "this machine's network interfaces",
            str(network.get("detail") or ""),
        )
    return (
        "No network interface reports itself as up in the current sample, so "
        "there is no address to give you. System > Hardware & services lists "
        "what was detected."
    )


def managed_workload_summary(workloads: Any) -> str:
    if not isinstance(workloads, dict):
        workloads = {}
    apps = workloads.get("apps", [])
    models = workloads.get("models", [])
    running = [
        str(item.get("name") or item.get("id") or "managed app")
        for item in apps
        if item.get("running") or item.get("status") == "running"
    ]
    unhealthy = [
        str(item.get("name") or item.get("id") or "managed app")
        for item in apps
        if item.get("health") in {"unhealthy", "failed"}
        or item.get("status") in {"failed", "unhealthy"}
    ]
    running_summary = (
        "Running managed apps: {}.".format(", ".join(running))
        if running else "No managed apps report that they are running."
    )
    verdict = (
        "These managed apps need attention: {}.".format(", ".join(unhealthy))
        if unhealthy
        else "None of the managed apps is flagged as failed or unhealthy."
    )
    return (
        "This Vaelor node currently manages {} app{} and {} local AI model{}. {} {} "
        "This check covers Vaelor-managed workloads, not every unrelated host process."
    ).format(
        len(apps), "" if len(apps) == 1 else "s",
        len(models), "" if len(models) == 1 else "s", running_summary, verdict,
    )


#: "What is **the** CPU temperature" against "what is **a** Docker container".
#:
#: The determiner is the whole distinction and it was not being read. Both
#: open "what is", and a rule that stops at the opener sends the first one -
#: a request for a live reading, with an intact built-in path behind it - to
#: the model. On a Pi that produced the self-contradicting reply *"The selected
#: AI connection did not answer this request... Built-in appliance questions
#: about cooling, lighting, display, storage, network, services, workloads,
#: and updates remain available."* The same machine answered *"how hot is my
#: cpu?"* in 4.2 seconds.
#:
#: A definite article alone is not enough - "what is the capital of France"
#: has one - so this is paired with :func:`is_appliance_question` below, and
#: an indefinite article keeps its definitional reading whatever it names.
_DEFINITE_READING = re.compile(
    r"^\s*(?:what(?:'|’)?s|what\s+is|what\s+are|how\s+much\s+is)\s+"
    r"(?:the|my|our|its|their)\b",
    re.IGNORECASE,
)


def is_general_knowledge_question(message: str) -> bool:
    """Keep definitional questions out of live appliance keyword routing.

    A definitional *opener* is not a definitional *question*. "What is the CPU
    temperature" asks this machine for a number it holds; "What is a Docker
    container" asks for a definition and must not be answered with the app
    inventory, which is the defect the indefinite-article half of this test
    still exists to prevent.
    """
    text = str(message).strip()
    lower = text.lower()
    definitional = lower.startswith(
        ("what is ", "what are ", "define ", "explain ", "how does ")
    )
    appliance_scope = any(
        phrase in lower
        for phrase in (
            # `" our "` beside `" my "`. Without it the answer turned on a
            # pronoun: "explain how my cpu works" was appliance-scoped and
            # "explain how our cpus work" was general knowledge, sent to a
            # surface that cannot read the CPU. Inherited rather than
            # introduced, and the two are already interchangeable everywhere
            # else - `_SELF_REFERENCE` in `assistant_intents` lists "our", and
            # `_THIS_MACHINE` in `chat_appliance_scope` matches `my|our` in one
            # alternation. This table was the only place they differed.
            #
            # It is admitted only because "system" was kept out of
            # `_APPLIANCE_SIGNALS`. With "system" in, this line would have
            # turned "explain how our immune system works" and "explain how our
            # school system is funded" into declined appliance questions: the
            # possessive branch and the topic vocabulary multiply, and the
            # measurement has to be taken on the pair rather than on either.
            " my ", " our ", " this ", "current", "installed", "running",
            "on the appliance", "on this",
        )
    )
    asks_for_a_reading = bool(
        _DEFINITE_READING.match(text) and is_appliance_question(text)
    )
    return definitional and not appliance_scope and not asks_for_a_reading
