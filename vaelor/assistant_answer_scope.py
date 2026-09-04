"""Answer the question that was asked, or say which part cannot be answered.

Two shapes of question reached the reading-backed path and came back with a
fluent answer to something else. Measured live on the Pi, 2026-08-11:

- **"What was the CPU temperature ten minutes ago?"** returned the live
  reading, byte-identical to the answer given to the *current* temperature
  moments earlier, with nothing saying the past had not been read.
- **"Which version of Vaelor is this machine running, and does it have a
  GPU?"** returned a list of five running services, naming neither - while
  Home displayed the version plainly.

**Why #144 did not already cover this.** #144 put the rules in the prompts -
answer every part, say when a part has no reading, never dress the present as
the past - and a prompt only governs an answer the *model* writes. These
replies never reached the model. The reading-backed router answers first, and
it had no notion of a question's tense and no route from "which version" to
the brief that carries it. On an appliance where the model is slow or
unreachable that router is the only thing an owner ever meets, so the rule has
to live here too.

The product already knew how to do this correctly and did so in the same
session: asked for something no reading can answer, it said "I cannot answer
it reliably without guessing" and named what it could answer instead. That is
the behaviour these two cases were missing.
"""

from __future__ import annotations

import re
from typing import Any, Dict, List, NamedTuple, Optional, Tuple

from .assistant_answer_topics import asks_about
# `asks_about_identity` is defined once in `assistant_answer_presentation` and
# re-exported here so every existing caller (and `identity_answer` below) keeps
# a single implementation. It has to be single-sourced because
# `assistant_intents.is_out_of_scope_question` consults the same predicate:
# LESSONS 6 / #205, "what version am I on?" was answered here and redirected
# there at once. Do not reintroduce the markers in this module.
from .assistant_answer_presentation import (
    asks_about_cpu_count, asks_about_identity, asks_about_os_version,
)
from .assistant_intents import named_subjects
from .assistant_vocabulary import TOOL_PHRASES
from .telemetry_trend import claimed_by_a_reading
from .telemetry_trend import evidence as trend_evidence
from .telemetry_trend import (
    nothing_to_compare, off_subject, readings_named, span, trend_lines,
)
from .telemetry_store import MAX_HISTORY_WINDOW_SECONDS

#: Markers that a question is about a *past* moment rather than now.
#:
#: Deliberately conservative - each one is unambiguously retrospective on its
#: own. "since" and "before" are absent because "how long since the last
#: update" and "check before adding a workload" are present-tense questions
#: wearing temporal words, and a false positive here refuses an answer the
#: appliance could have given.
#: **Narrowed after review, and the removals matter more than what is left.**
#: The first version matched `\bwas (?:it|the|there)\b` and bare clock times,
#: which refused questions the appliance could answer and answered nothing:
#: "Was it the fan that caused the alert?" was met with "this appliance holds
#: no record covering that time" while `health_alert_answer` was holding the
#: fan reading, at 71 °C with the fan stopped. "Should I schedule the backup
#: at 2 am" and "is 3am a good time to reboot" were refused as history.
#:
#: A false positive here is the expensive direction: it withholds a live
#: reading the machine has, which is the very defect this module exists to
#: stop, arriving from the other side. So every marker below has to be
#: retrospective *on its own*, in a question, with no reading that could
#: answer it instead. A phrasing this misses is answered the ordinary way and
#: costs nothing; a phrasing it wrongly catches costs an answer.
_PAST_MARKERS = (
    re.compile(r"\bago\b"),
    re.compile(r"\byesterday\b"),
    re.compile(r"\blast night\b"),
    # "what was it earlier" is retrospective; "earlier" alone inside advice
    # ("the earlier step") is not, so it must sit beside a past-tense verb.
    re.compile(r"\b(?:was|were|did)\b[^.?!]{0,40}\bearlier\b"),
    # Bare `last hour` and friends, restored. Narrowing them to `in/over the
    # last hour` regressed "what was the cpu load last hour" to the founding
    # defect: it returned "The host CPU is currently 41.5°C" - the present as
    # the past, and a temperature for a question about load. **A miss here is
    # not neutral**, because the path that catches a miss has no tense
    # awareness at all; the reasoning that misses are cheap was wrong. These
    # reintroduce none of the false positives the review proved, which came
    # from `was (it|the|there)` and bare clock times, not from `last X`.
    re.compile(r"\blast (?:hour|day|week|month|year)\b"),
    re.compile(r"\bpast (?:hour|day|week|month|year)\b"),
    re.compile(
        r"\b(?:in|over) the (?:last|past) "
        r"(?:hour|day|week|month|year|\d+\s+\w+|(?:two|three|four|five|six|"
        r"seven|eight|nine|ten|few|couple of)\s+\w+)\b"
    ),
    # A past-tense verb *and* a past window, so "should I back up overnight"
    # stays a present-tense question about a plan.
    #
    # `today` and `tonight` join the window list here rather than becoming
    # markers of their own; the next pattern says why the bare word cannot be
    # one.
    re.compile(
        r"\b(?:was|were|did|happened)\b[^.?!]{0,40}"
        r"\b(?:today|tonight|this morning|this afternoon|this evening|"
        r"overnight|last week|last month)\b"
    ),
    # **"Has the fan been running at full speed at any point today?"**
    # Answered with "The CPU fan is turning at 932 RPM" - a live reading
    # returned as history, `answered=True`, in 0.0 s. This is the founding
    # defect of this module arriving on a phrasing it had no pattern for.
    #
    # `today` is a **window, not a tense**: "is it hot today?" is a question
    # about now, and a marker matching the bare word would refuse it and every
    # question like it. So the day words below are only ever matched beside
    # something that puts them behind us - a present-perfect construction, or
    # one of the retrospective quantifiers.
    re.compile(
        r"\b(?:at any (?:point|time)|at all|so far|earlier|already)\b"
        r"[^.?!]{0,30}\b(?:today|tonight|this (?:morning|afternoon|evening))\b"
    ),
    re.compile(
        r"\b(?:today|tonight|this (?:morning|afternoon|evening))\b"
        r"[^.?!]{0,30}\b(?:at any (?:point|time)|so far|already)\b"
    ),
    # "has ... been ... today", "have there been any alerts this morning".
    # `been` is required: without it "have I got enough space today" is a
    # question about now, and this pattern would refuse it.
    re.compile(
        r"\b(?:has|have|had)\b[^.?!]{0,40}\bbeen\b[^.?!]{0,40}"
        r"\b(?:today|tonight|this (?:morning|afternoon|evening)|overnight|"
        r"so far)\b"
    ),
    # The same construction with no day word at all: "has this ever crashed",
    # "have there been any alerts at any point". Bare `ever` is deliberately
    # absent - "have you ever" opens ordinary conversation.
    re.compile(
        r"\b(?:has|have|had)\b[^.?!]{0,60}\bat any (?:point|time)\b"
    ),
)

#: Hardware an identity question is routinely asked about in the same breath,
#: mapped to the label the standing brief uses for it.
#:
#: **The live question was compound and only half of it was answered.** "Which
#: version of Vaelor is this machine running, *and does it have a GPU?*" came
#: back with the version alone, reported as fully answered - which is #144's
#: "silently drops half the question" surviving inside the fix for it. The
#: brief already states each of these in words, so the second half is answerable
#: from what the router has in hand.
#: ``noun`` is how to name it back to a reader; lower-casing ``brief_label``
#: produced "whether this machine has a compute gpu".
class _HardwareClause(NamedTuple):
    """One device an identity question is asked about in the same breath.

    Named because the row was a bare ``(pattern, label, noun)`` triple and
    ``row[1]`` against ``row[2]`` - a label to look up in the brief against a
    noun to write into a sentence - is exactly the distinction that produced
    "whether this machine has a compute gpu" when the two were confused.
    """

    asked: "re.Pattern[str]"
    brief_label: str
    noun: str


_HARDWARE_ASKED: Tuple[_HardwareClause, ...] = (
    _HardwareClause(
        re.compile(r"\bgpu\b|\bgraphics card\b|\bvideo card\b"),
        "Compute GPU", "a GPU",
    ),
    _HardwareClause(
        re.compile(r"\bnpu\b|\bneural (?:processing unit|accelerator)\b"),
        "Neural processing unit", "a neural processing unit",
    ),
)


def _hardware_lines(message: str, brief_text: str) -> Tuple[List[str], bool]:
    """The brief's own statement about hardware this question also asks about.

    Returns the brief's words rather than a paraphrase: it already says
    "available" or "not available" with the driver's reason, and restating a
    capability in this module would be a second place for it to go stale.

    The second element is whether *every* hardware clause asked about was
    actually read. The caller needs it to decide `answered`: a reply that says
    "I could not read whether this machine has a GPU" has not answered the
    question, and reporting it as answered is the half-question defect.
    """
    lower = " ".join(str(message or "").lower().split())
    text = str(brief_text or "")
    lines: List[str] = []
    unread = 0
    for clause in _HARDWARE_ASKED:
        if not clause.asked.search(lower):
            continue
        found = re.search(
            r"^-\s*{}:\s*(.+)$".format(re.escape(clause.brief_label)),
            text, re.MULTILINE,
        )
        if found:
            lines.append("{}: {}".format(
                clause.brief_label, found.group(1).strip()
            ))
        else:
            unread += 1
            lines.append(
                "I could not read whether this machine has {}. Hardware on "
                "the Home screen lists what was detected.".format(clause.noun)
            )
    return lines, unread == 0


#: Contexts in which a past-window phrase is **not** retrospective, checked
#: before the markers above and vetoing them.
#:
#: **This exists because narrowing the positive patterns is what kept
#: regressing.** Three rounds now: broad markers caught questions the appliance
#: could answer, narrowing them lost "last hour" entirely, and restoring
#: "last hour" caught "the last day of the month". Every round moved the same
#: patterns and every round traded one direction of error for the other.
#:
#: A veto list separates the two directions so they stop fighting. A false
#: positive found in testing is added here, where it cannot weaken the markers;
#: a missed phrasing is added above, where it cannot widen the vetoes. Each
#: entry below was a demonstrated refusal of an answerable question.
_NOT_RETROSPECTIVE = (
    # Partitive: "the last day *of the month*", "the last hour *of logs*" —
    # a slice of something, not a moment that has passed. This one veto covers
    # "should the backup run on the last day of the month", "retain the last
    # week of samples", "keep only the last day of journal logs" and
    # "is the last week of the trial covered by support".
    re.compile(r"\b(?:last|past) (?:hour|day|week|month|year)\s+of\b"),
    # "past" as a preposition of motion: "how do I get past day one".
    re.compile(r"\b(?:get|got|getting|go|going|goes|move|moved|moving)\s+past\b"),
    # A retention or scheduling instruction names a window without asking
    # about one. "Rotate logs after the last day of the month" is an
    # instruction even where the partitive veto would not reach it.
    re.compile(r"\b(?:retain|rotate|prune|purge|schedule)\b"),
)


def asks_about_a_past_time(message: str) -> bool:
    """Whether this question is about a moment that has already passed."""
    lower = " ".join(str(message or "").lower().split())
    if any(pattern.search(lower) for pattern in _NOT_RETROSPECTIVE):
        return False
    return any(pattern.search(lower) for pattern in _PAST_MARKERS)


def _retained_samples(facts: Dict[str, Any]) -> Dict[str, Any]:
    history = facts.get("metrics.history")
    return history if isinstance(history, dict) else {}


#: Phrases that name *when*, not *what*. Subtracted before anything below asks
#: what a sentence is about, because ``ago``, ``yesterday``, ``trend`` and the
#: rest select `metrics.history` and name no subject at all.
_WHEN_PHRASES = frozenset(TOOL_PHRASES["metrics.history"])

#: A subordinate clause and everything in it, up to the next comma or the end
#: of the sentence.
#:
#: **Its object is an anchor or a circumstance, never the subject.** "Has the
#: CPU temperature been rising *since the last reboot*", "did the CPU
#: temperature rise *during the backup*", "did CPU usage climb *while the
#: service was restarting*" - each names a reading this store carries, and
#: each was refused because ``reboot``, ``backup`` and ``service`` are subject
#: words to :func:`assistant_intents.named_subjects`. `_WHEN_PHRASES` cannot
#: reach them: it subtracts the vocabulary that names a *time*, and these name
#: a time by naming a thing that happened.
#:
#: **Only the unclaimed-subject test reads this, and the asymmetry is
#: deliberate.** `readings_named` still reads the whole sentence, so naming a
#: reading anywhere counts; refusing needs the foreign subject in head
#: position. Both directions favour answering, which is what VD-035 and
#: LESSONS pattern 4 price higher than the converse.
#:
#: **The clause ends at a comma, not at the sentence.** "During the backup,
#: did the fan speed rise?" puts the subordinator first, and running to the
#: end of the sentence would swallow ``fan`` and answer a fan question with a
#: temperature - the defect this whole class is about, re-entered through its
#: own repair.
#:
#: **``when`` is deliberately absent.** It opens half the retrospective
#: questions an owner asks - "when did the fan rpm drop" - so treating it as a
#: subordinator strips the subject out of the sentence it introduces.
#: The branches, separately, so a test can rebuild the rule and delete one.
#:
#: `tests/test_vocabulary_reachability` counts a compiled name as one entry
#: and says what to do about an alternation inside it: a `.search()` returns
#: on the first branch to match, so a dead branch is deletable in silence -
#: three of six prompt-injection branches were, once. The rows are in
#: `tests/test_assistant_trend_answer.SUBORDINATOR_CASES` and each one is
#: dropped in turn to prove its own sentence changes verdict.
_SUBORDINATORS = "since|during|while|whilst|after|before|until"

_SUBORDINATE_CLAUSE = re.compile(
    r"\b(?:" + _SUBORDINATORS + r")\b[^,.?!;]*", re.IGNORECASE)


def _readings_asked_about(message: str) -> Tuple[Tuple[str, ...], bool]:
    """Which retained readings this question names, and whether it names a
    subject this store does not carry.

    **This was inferred from two other tables for two review rounds and each
    inference was wrong at a different boundary.** The answer-topic row was
    tried and four of its rows carry no words, so "has the backup been
    restored yesterday" named nothing and got a CPU trend. The fact tool was
    tried and `cooling.status` carries fan speed, RPM and duty cycle beside
    the one temperature this store keeps, so "has the fan speed been rising"
    was answered "CPU temperature rose 4 °C" - the original defect, narrower.
    **A tool is not a subject and a topic is not a reading.** Both halves now
    work at the granularity of the reading itself.

    1. **A reading is named and nothing else is** - answer about those.
    2. **A subject this store does not carry is named** - decline, *even where
       one of ours is named too*. "is the gpu temperature rising" and "is the
       cooling fan ok earlier" each name one of each, and half an answer is
       not an answer - the rule :func:`identity_answer` already applies.
    3. **Nothing is named at all** - "what was it an hour ago", "how is my
       machine doing over the last hour" - report every retained reading, by
       name. A sweep phrase belongs here and did not: ``how is my`` selects
       eleven tools through `_BROAD_TOOLS`, so "how is my cpu doing over the
       last few minutes" was read as naming eleven foreign subjects and
       refused with a sentence naming CPU temperature as what the appliance
       retains - about a question asking for CPU temperature. A sweep asks for
       everything; it does not ask about something else, and
       :func:`assistant_intents.named_subjects` is where that distinction now
       lives.

    **"One unclaimed subject is enough" holds in head position only.** It is
    right for "is the gpu temperature rising", where the foreign subject *is*
    what the question is about. It was wrong for the object of a subordinator:
    "did the CPU temperature rise during the backup" names a reading this
    store carries and was refused over ``backup``, with the self-contradicting
    sentence naming CPU temperature as what the appliance retains. With a
    model available `BUILTIN_REFUSAL_MARKERS` keeps that reply off the screen;
    with the model unavailable the router's last line returns it, and that is
    exactly the case this whole path exists to serve.
    """
    subjects = named_subjects(_SUBORDINATE_CLAUSE.sub(" ", str(message or "")))
    return (
        readings_named(message),
        any(not claimed_by_a_reading(phrase)
            for phrase in subjects - _WHEN_PHRASES),
    )


#: Where to look when this path cannot answer. One constant because two
#: branches end with it, and a sentence written twice in one module is what
#: `tests/test_duplicate_literals` counts.
_LOOK_ELSEWHERE = (
    "Anything I report is the current moment. If you were asking about a "
    "reading, Home shows the live values; if you were asking what happened, "
    "Activity keeps the record of jobs and changes."
)


def _scoped(
    lines: List[str], evidence: List[Dict[str, str]], *, answered: bool
) -> Dict[str, Any]:
    """The one place this module builds a reply for the router.

    **`answered` is keyword-only and has no default, so no branch can leave
    it out.** That is the whole point. #170 was two functions building this
    same six-key dict from several branches, one of which computed `answered`
    into a local and returned a dict without it - and the router's
    `scoped.pop("answered", True)` supplied `True`, so a refusal reading "this
    appliance holds no record covering that time" was recorded as a successful
    reading-backed answer. A missing key is invisible to a test that asserts
    on values, which is why it survived a test suite that covered the text of
    every branch.

    Routing it through one constructor makes the broken state unrepresentable
    rather than merely tested for, and keeps `source` derived from `answered`
    instead of set twice and allowed to disagree.
    """
    return {
        "answer": "\n\n".join(lines),
        "evidence": evidence,
        "suggested_actions": [],
        "proposed_job": None,
        "answered": answered,
        "source": "built-in-live-data" if answered else "out-of-scope",
    }


def scoped_answer(
    message: str, facts: Dict[str, Any], context: Dict[str, Any]
) -> Optional[Dict[str, Any]]:
    """The honest answer for a question this path would otherwise mis-answer.

    One entry point so the caller carries a single branch: the router in
    `deployment_agent` is the file everything touches, and it is at its line
    ceiling.
    """
    appliance = context.get("appliance", context)
    brief = appliance.get("machine_brief") or {}
    brief_text = str(brief.get("text", "")) if isinstance(brief, dict) else ""
    # The time-ranged history callback, wired into the route from the control
    # plane, lets `past_time_answer` read the *window the question names* rather
    # than only the newest rows the router pre-gathered. Absent (an older caller
    # or a control plane without the callback) it falls back to that count read.
    history_range = appliance.get("telemetry_history_range")
    return (
        past_time_answer(message, facts, history_range)
        or identity_answer(message, facts, brief_text)
        or host_fact_answer(message, facts, brief_text)
        or operational_claim_answer(message)
    )


#: A request to validate/verify a configuration - something the Assistant can
#: only satisfy by *running* a check, never from a chat reply.
_VALIDATION_REQUEST = re.compile(
    r"\bvalidat\w*\b|\bverif\w*\b|\blint\w*\b|\bdry[ -]?run\b|"
    r"\b(?:in)?valid\b|\bwell[- ]?formed\b|\bsyntax\b",
    re.IGNORECASE,
)
#: What the request is validating: a stack/config, not a live reading. "verify
#: my backups" or "check the cpu" name no config target and are left to the
#: branches that actually read them.
_CONFIG_TARGET = re.compile(
    r"\bcompose\b|\bconfig\b|\bconfiguration\b|\byaml\b|\byml\b|\bstack\b|"
    r"\bmanifest\b|\bcompose file\b|\bdeployment file\b",
    re.IGNORECASE,
)


def operational_claim_answer(message: str) -> Optional[Dict[str, Any]]:
    """Refuse to claim a validation verdict the Assistant never produced.

    **Live on the Pi #205 deep pass.** "Validate the compose config for
    model-assistant" was answered "valid, all services running" with
    `proposed_job: null` - an operational verdict from a run that never
    happened. The model cannot execute a validation from a chat turn, and there
    is no validate job type on the approval allowlist, so the only honest reply
    is that it cannot verify and where the guarded check lives. This intercepts
    the request before the model can invent a result (the same class as
    "the GPU is slow due to low utilisation" - the question returned as its own
    answer).
    """
    lower = " ".join(str(message or "").lower().split())
    if not (_VALIDATION_REQUEST.search(lower) and _CONFIG_TARGET.search(lower)):
        return None
    return {
        "answer": (
            "I cannot validate a Compose configuration from here - that needs a "
            "validation run I do not perform in this chat, so I will not tell "
            "you it is valid or invalid, or that its services are running. On "
            "Workloads > Install, Import a Docker stack runs the guarded checks "
            "- image architecture, ports, storage, memory, and health - before "
            "anything is approved."
        ),
        "evidence": [{
            "source": "assistant.capability",
            "summary": "Vaelor validates a Compose stack through the guarded "
                       "import job, not from a chat reply, so no verdict is "
                       "claimed here.",
        }],
        "suggested_actions": [
            "Open Workloads > Install and use Import a Docker stack to validate "
            "a Compose workload before approval.",
        ],
        "proposed_job": None,
        "answered": True,
        "source": "built-in-capability",
    }


#: The brief lines that carry the two standing host facts this path answers,
#: read verbatim rather than re-derived. `assistant_machine_brief._processor_line`
#: already decides how to render cores/threads/logical-CPUs, and
#: `_identity` renders the OS name and support label; quoting the line keeps one
#: author for those words (LESSONS 6). A re-derivation here is a second place
#: for "32 cores" on a 16-core part to come back (#107).
_PROCESSOR_LINE = re.compile(r"^Processor:\s*(.+?)\s*$", re.MULTILINE)
_OPERATING_SYSTEM_LINE = re.compile(r"^Operating system:\s*(.+?)\s*$", re.MULTILINE)


def host_fact_answer(
    message: str, facts: Dict[str, Any], brief_text: str = ""
) -> Optional[Dict[str, Any]]:
    """The processor and operating-system facts, from the standing brief.

    **Live on the Pi #205 deep pass (LESSONS 5 / #144).** "What operating
    system version is installed?" routed to `workloads.inventory` on the word
    `installed` and answered with the app list; "how many CPU cores does this
    machine have?" reached the telemetry branch and answered with CPU usage.
    Both answered a *different* question, with evidence citations - the shape
    this module exists to stop. The facts stand in the brief the router already
    holds; this routes the phrasing to them.

    A fact genuinely absent from the brief is declined here rather than left to
    fall through to the adjacent branch (`workloads.inventory` for OS,
    `system.telemetry` for CPU), which is the whole point: an honest "I could
    not read it" beats a fluent answer to a question nobody asked.
    """
    os_asked = asks_about_os_version(message)
    cpu_asked = asks_about_cpu_count(message)
    if not (os_asked or cpu_asked):
        return None
    text = str(brief_text or "")
    lines: List[str] = []
    evidence: List[Dict[str, str]] = []
    answered = True
    if cpu_asked:
        found = _PROCESSOR_LINE.search(text)
        if found:
            lines.append(
                "This machine's processor is {}.".format(
                    found.group(1).strip().rstrip(".")
                )
            )
            evidence.append({
                "source": "assistant.machine-brief",
                "summary": "Read the processor line, including the core count, "
                           "from this appliance's standing brief.",
            })
        else:
            answered = False
            lines.append(
                "I could not read this machine's CPU core count. The Home "
                "screen lists the detected processor."
            )
    if os_asked:
        found = _OPERATING_SYSTEM_LINE.search(text)
        if found:
            lines.append(
                "This machine's operating system is {}.".format(
                    found.group(1).strip().rstrip(".")
                )
            )
            evidence.append({
                "source": "assistant.machine-brief",
                "summary": "Read the operating-system line from this "
                           "appliance's standing brief.",
            })
        else:
            answered = False
            lines.append(
                "I could not read this appliance's operating system. The Home "
                "screen shows what was detected."
            )
    return _scoped(lines, evidence, answered=answered)


#: The number an owner puts in front of a window unit, spelled or figured. Only
#: the unambiguous words: ``a``/``an`` mean one, and ``few``/``couple`` are
#: deliberately absent - a "few minutes" is not two or three of anything
#: countable, so it is answered by :data:`_BARE_WINDOWS` with a fixed sensible
#: span rather than multiplied into a false precision.
_NUMBER_WORDS = {
    "a": 1, "an": 1, "one": 1, "two": 2, "three": 3, "four": 4, "five": 5,
    "six": 6, "seven": 7, "eight": 8, "nine": 9, "ten": 10,
}

#: A unit word an owner reaches for, mapped to the single-letter unit that
#: :func:`assistant_machine_tools._parse_window` accepts and a multiplier into
#: it. Weeks, months and years are expressed in **days** so nothing but the
#: ``s``/``m``/``h``/``d`` units that one parser knows ever leaves this module;
#: the store clamps a window past its seven-day retention to that retention
#: itself (``clamp_window_seconds``), so a "last month" that resolves to ``30d``
#: is answered with the seven days actually held rather than rejected.
_UNIT_TO_WINDOW = {
    "minute": ("m", 1), "minutes": ("m", 1), "min": ("m", 1), "mins": ("m", 1),
    "hour": ("h", 1), "hours": ("h", 1), "hr": ("h", 1), "hrs": ("h", 1),
    "day": ("d", 1), "days": ("d", 1),
    "week": ("d", 7), "weeks": ("d", 7),
    "month": ("d", 30), "months": ("d", 30),
    "year": ("d", 365), "years": ("d", 365),
}

#: A counted window: ``24 hours``, ``last 7 days``, ``two weeks``, ``an hour``.
#: The count is optional so a bare ``hour`` is caught here too, defaulting to
#: one - "an hour ago" and "over the last hour" both resolve to ``1h``.
_COUNTED_WINDOW = re.compile(
    r"\b(?:(\d+|a|an|one|two|three|four|five|six|seven|eight|nine|ten)\s+)?"
    r"(minutes?|mins?|hours?|hrs?|days?|weeks?|months?|years?)\b"
)

#: Windows an owner names without a number, in priority order (first match
#: wins). Each maps a phrase to the span this appliance reads for it. ``today``
#: and the day-part phrases are a day; the vague quantities are a deliberately
#: small, fixed window rather than a made-up count.
#:
#: **Ordered so the more specific phrase is tested first.** "few minutes" has to
#: be matched before a lone "minute" pattern could, or "the last few minutes"
#: would resolve to five minutes rather than the fifteen its own row gives it.
_BARE_WINDOWS = (
    (re.compile(r"\bfew minutes\b|\bcouple of minutes\b"), "15m"),
    (re.compile(r"\bfew hours\b|\bcouple of hours\b"), "6h"),
    (re.compile(r"\bfew days\b|\bcouple of days\b"), "3d"),
    (re.compile(r"\byesterday\b|\blast night\b|\bovernight\b"), "48h"),
    (re.compile(r"\btoday\b|\btonight\b|\bthis (?:morning|afternoon|evening)\b"),
     "24h"),
)


def window_for_question(message: str) -> Optional[str]:
    """The telemetry window this question asks about, as a ``_parse_window``
    string, or ``None`` when it names no window this store can read for.

    This is the *question*-side parser: it decides **which** window to query.
    The unit arithmetic that turns ``"24h"`` into seconds stays in the single
    :func:`assistant_machine_tools._parse_window`; this only ever emits a string
    that parser accepts, so there is one place that knows a unit is 3,600
    seconds and this is not a second one.

    Returns ``None`` for a phrase that names a slice rather than a moment that
    has passed - "the last day of the month", "retain the last week of logs" -
    by consulting the same :data:`_NOT_RETROSPECTIVE` veto
    :func:`asks_about_a_past_time` does, so a partitive never becomes a window
    to query. It also returns ``None`` for a retrospective question that names
    no window at all ("what was it earlier"), which keeps the count-based
    reading the caller already holds rather than inventing a span.
    """
    lower = " ".join(str(message or "").lower().split())
    if any(pattern.search(lower) for pattern in _NOT_RETROSPECTIVE):
        return None
    for pattern, window in _BARE_WINDOWS:
        if pattern.search(lower):
            return window
    found = _COUNTED_WINDOW.search(lower)
    if not found:
        return None
    count_text, unit_text = found.group(1), found.group(2)
    unit, multiplier = _UNIT_TO_WINDOW[unit_text]
    if count_text is None:
        count = 1
    elif count_text.isdigit():
        count = int(count_text)
    else:
        count = _NUMBER_WORDS[count_text]
    amount = count * multiplier
    if amount <= 0:
        return None
    return "{}{}".format(amount, unit)


def _windowed_history(
    message: str, history_range: Any
) -> Optional[Dict[str, Any]]:
    """The retained trend over the window this question names, or ``None``.

    Reuses the a68 query path in full: :func:`assistant_machine_tools.metrics_history`
    with a ``window`` argument runs :func:`_metrics_trend`, which downsamples the
    span to at most 168 buckets and carries the store's own honest-failure
    sentences (off, unreadable, or readable-but-empty). Nothing here re-reads the
    store or re-parses a duration; this only chooses the window and hands it to
    that one path.

    ``None`` when the question names no window, or when this control plane wired
    in no time-ranged callback - in both cases the caller falls back to the
    count-based samples already gathered, which is the pre-a68 behaviour.
    """
    if history_range is None:
        return None
    window = window_for_question(message)
    if window is None:
        return None
    from .assistant_machine_tools import metrics_history

    return metrics_history(
        {"telemetry_history_range": history_range}, {"window": window}
    )


def _reads_a_recent_window(
    history: Dict[str, Any], samples: List[Any], held: bool
) -> bool:
    """Whether this read is a recent slice of a larger store (VD-097 / #224).

    The closing span sentence must not call a partial read "everything this
    appliance has retained" when older rows exist, nor claim more is retained
    when the read already holds all there is. Two read shapes reach here and the
    signal differs:

    * **count-based** (`metrics_history` without a window) returns ``requested``,
      and getting a full page - ``len(samples) >= requested`` - means older rows
      were left behind.
    * **windowed** (`_metrics_trend`) returns ``window_seconds`` and no
      ``requested``. Older rows exist beyond the window only when the window is
      shorter than the store's retention *and* the buckets actually reach back
      to the window's **far edge**. The threshold is near-full for a reason a
      half-window one got wrong: a store aged 0.7 of the asked window covers 70%
      of it and has nothing older, yet `>= 0.5 * window` called it a slice of a
      larger store and told the owner "more is retained" - false on a fresh
      reboot. Buckets are capped at 168, so a genuinely full window measures
      ≈ window minus one bucket (~0.6% short); ``0.9`` clears that and still
      rejects the aged-0.7 store, which measures well under it. A store younger
      than the window measures far below ``0.9`` and is correctly not told more
      exists.
    """
    if not held:
        return False
    window_seconds = history.get("window_seconds")
    if isinstance(window_seconds, int) and window_seconds > 0:
        if window_seconds >= MAX_HISTORY_WINDOW_SECONDS:
            return False
        measured = span(samples)
        return (
            measured.seconds is not None
            and measured.seconds >= 0.9 * window_seconds
        )
    requested = history.get("requested")
    return (
        isinstance(requested, int) and requested > 0
        and len(samples) >= requested
    )


#: The disclosure a windowed answer owes, because its figures are not readings.
#:
#: `_metrics_trend` downsamples a span to at most 168 buckets, each the **mean**
#: of the samples in it. Stating an endpoint ("from 40°C to 47.8°C") off a mean
#: and calling it a reading is the same class of overstatement this module was
#: written to stop, one layer in: the min/max spike guard in
#: `telemetry_trend.describe` runs over the means too, so a bucket can average
#: out a real peak. The count path holds raw rows and says "retained samples"
#: unchanged; only the windowed path appends this, and it names the resolution
#: and the peak a bucket can hide. (No direction word here - the module never
#: writes one outside a computed movement, and the guard scans for it.)
_AVERAGED_BUCKETS = (
    "These figures are {}-second averaged buckets, not individual readings: "
    "they show the shape of the change across the window, and a brief spike "
    "between two buckets can be averaged away and not appear here."
)


def _windowed_evidence(
    samples: List[Any], history: Dict[str, Any]
) -> Dict[str, str]:
    """The evidence entry for a windowed read, naming what the rows are.

    `trend_evidence` says "retained telemetry samples", which is true of the
    count path's raw rows and misleading of a windowed read's downsampled means.
    This states the resolution instead, so the evidence panel and the answer's
    own caveat agree about what was read.
    """
    bucket = history.get("bucket_seconds")
    return {
        "source": "metrics.history",
        "summary": "Read {} downsampled telemetry buckets, each a {}-second "
                   "average, covering the requested window.".format(
                       span(samples).count, int(bucket) if bucket else 0),
    }


def past_time_answer(
    message: str, facts: Dict[str, Any], history_range: Any = None
) -> Optional[Dict[str, Any]]:
    """An honest answer to a question about a past moment, or ``None``.

    Never returns a current reading on its own. Either the retained samples
    are read for the direction they moved in, or the absence of a record is
    stated - and in both cases any figure is labelled with what it came from.

    **`history_range` is the a68 time-ranged callback, and it is what makes a
    windowed question truthful (A69).** Given it, a question naming a window
    ("the last hour", "the last 7 days") reads *that* window through
    `_windowed_history` - the same downsampled path `metrics.history`'s window
    branch uses - rather than the newest thirty rows by count the router
    pre-gathers, which is about half a minute and was answered "no record
    covering that time" while the seven-day store held the window exactly.
    Absent (an older caller, or a control plane with no ranged callback wired),
    or for a retrospective question that names no window, it falls back to the
    count-based fact, which is the pre-a68 behaviour every branch below keeps.

    **The samples branch is now `answered`, and the docstring it replaces said
    why it was not.** It read: "offering the retained samples still declines
    the question that was asked ('I cannot look up an exact past moment... ask
    for a trend'), so recording it as a successful reading-backed answer
    overstates what the owner got." That was exactly right while nothing in
    the tree computed a trend - the reply was an invitation, and an invitation
    is not an answer. `telemetry_trend` is the capability that objection was
    waiting for: the branch now states a direction, a magnitude and the span
    the samples actually cover, which is a reading-backed answer and is
    recorded as one.

    **The no-samples branch is untouched in what it refuses**, and lost only
    its own offer of a trend. It fires when this appliance holds nothing, so
    a sentence promising to describe a trend across the samples it retains is
    the same advertisement-without-a-capability, one branch down.

    **Four states, not two**, because samples in hand are not the same as a
    direction in hand: rows about another subject, one sample, or rows
    carrying no trendable field can each be reported honestly and cannot be
    trended. Those branches say what is held and stop. `trend_lines` returns
    nothing when `trends` does, so no branch here can write a direction word
    the computation did not produce.

    **The gate admits a trend question, and the first version admitted it in
    one branch only.** `asks_about_a_past_time("did the cpu temperature rise,
    fall, or hold steady")` is `False` - the sentence the appliance itself
    invited is present tense - and `if not retrospective: return None` sat
    *after* the movement branch, so with one sample or none this returned
    `None`, the question fell through to the live-reading assembly, and the
    owner got the current instant. That is #203 surviving in three of the four
    sample states, and they are the states VD-095 established are real: the
    boot window, a fresh store, and `metrics_history`'s "readable but holds no
    samples yet". The gate is now taken once, at the top, for every branch.
    """
    retrospective = asks_about_a_past_time(message)
    if not (retrospective or asks_about("history", message)):
        return None
    # **When the question names a window, read that window.** A69: the store
    # retains seven days, but the fact the router pre-gathers is the newest
    # thirty rows *by count* - about half a minute. Asked "how much has the CPU
    # temperature changed over the last hour", this branch read those thirty
    # seconds and, when the count read came back empty or ungathered, declared
    # "no record covering that time" while InfluxDB held the hour exactly. So
    # the window is parsed from the question and the a68 time-ranged path is
    # queried for it; the count-based fact stays the fallback for a
    # retrospective question that names no window ("what was it earlier").
    history = _windowed_history(message, history_range) or _retained_samples(facts)
    samples = history.get("samples") or []
    held = bool(history.get("available")) and bool(samples)
    named, other_subject = _readings_asked_about(message)
    if other_subject:
        # The samples are real and cover that time. They are about the wrong
        # thing, and saying so is the only true sentence available here -
        # `nothing_to_compare` would claim the rows carry nothing comparable
        # while they carry a rising temperature, and the no-record refusal
        # below would deny holding samples that exist.
        return _refused(
            off_subject(samples) if held
            else "This appliance holds no record of that, so I cannot tell "
                 "you what was true then.",
            history, samples, held,
        )
    more_retained = _reads_a_recent_window(history, samples, held)
    # A windowed read carries `bucket_seconds`; a count read does not. The
    # figures it yields are bucket means, so the answer discloses that rather
    # than presenting them as raw readings.
    windowed = history.get("bucket_seconds") is not None
    movement = (
        list(trend_lines(samples, named, more_retained=more_retained))
        if held else []
    )
    if movement:
        if windowed:
            movement.append(
                _AVERAGED_BUCKETS.format(int(history.get("bucket_seconds") or 0))
            )
        if retrospective:
            movement.append(
                "I cannot look up one exact past moment; these retained "
                "samples are what this appliance holds."
            )
        evidence = (
            _windowed_evidence(samples, history) if windowed
            else trend_evidence(samples)
        )
        return _scoped(movement, [evidence], answered=True)
    if held:
        return _refused(nothing_to_compare(samples), history, samples, held)
    # **Not every retrospective question is about a sensor.** "What did we
    # change yesterday" and "when was this installed" are about the past and
    # are not readings, and answering them with "no record covering that
    # time... open Home for the live readings" sends the reader to a screen
    # that cannot help - a confidently wrong direction, which is the failure
    # this module exists to stop. The refusal says what is missing without
    # asserting what was asked for.
    return _refused(
        "This appliance holds no record covering that time, so I cannot "
        "tell you what was true then." if retrospective else
        "This appliance holds no record of how that has changed, so I cannot "
        "tell you which way it moved.",
        history, samples, held,
    )


def _refused(
    first_line: str, history: Dict[str, Any], samples: List[Any], held: bool
) -> Dict[str, Any]:
    """A reply that declines, with the tool's own reason and where to look.

    One builder for every non-answer, for the reason `_scoped` is one builder
    for every reply: four branches each assembling the same three parts is
    four chances to leave one out, and the pointer sentence written twice in
    one module is what `tests/test_duplicate_literals` counts.

    **No trend is offered in any of them, and that removal is the point.** The
    clause that used to sit inside :data:`_LOOK_ELSEWHERE` - "I can describe a
    trend across the samples this appliance retains" - was emitted precisely
    where no trend could be computed. Every pointer it carried is kept; only
    the promise is gone, and `tests/test_assistant_trend_answer` scans the
    tree's own syntax to keep it that way.
    """
    lines = [first_line]
    reason = str(history.get("reason") or "").strip()
    if reason:
        lines.append(reason)
    lines.append(_LOOK_ELSEWHERE)
    evidence = [trend_evidence(samples)] if held else [{
        "source": "assistant.time-scope",
        "summary": "No retained sample covers the time this question "
                   "asks about.",
    }]
    return _scoped(lines, evidence, answered=False)


def identity_answer(
    message: str, facts: Dict[str, Any], brief_text: str = ""
) -> Optional[Dict[str, Any]]:
    """What this machine and this software are, from the brief and identity.

    The brief carries the Vaelor version (#144) and states absent hardware in
    words; ``system.identity`` carries it as a field. Either answers "which
    version"; a question that reaches neither used to fall through to a list
    of running services, which answers nothing that was asked.
    """
    if not asks_about_identity(message):
        return None
    identity = facts.get("system.identity")
    identity = identity if isinstance(identity, dict) else {}
    version = str(identity.get("version") or "").strip()
    if not version and brief_text:
        # `\S+` then strip the sentence's full stop: a version is full of
        # dots, so excluding them from the capture returns "2" for
        # "2.1.0a47." - which reads like an answer and is not one.
        found = re.search(r"Vaelor version:\s*(\S+)", str(brief_text))
        version = found.group(1).rstrip(".") if found else ""
    name = str(identity.get("name") or "").strip()
    lines: List[str] = []
    evidence: List[Dict[str, str]] = []
    if version:
        lines.append("This machine is running Vaelor {}.".format(version))
        evidence.append({
            "source": "system.identity",
            "summary": "Read the appliance's software version.",
        })
    else:
        lines.append(
            "I could not read this appliance's Vaelor version. Settings shows "
            "it, and so does the foot of the Home screen."
        )
    if name:
        lines.append("The detected hardware reports itself as {}.".format(name))
    hardware, hardware_read = _hardware_lines(message, brief_text)
    if hardware:
        lines.extend(hardware)
        evidence.append({
            "source": "assistant.machine-brief",
            "summary": "Read what this appliance's brief records about the "
                       "hardware the question also asked about.",
        })
    # **Every clause asked has to be resolved, not just the first one.**
    # `answered = bool(version)` reported a whole answer for a reply that read
    # "This machine is running Vaelor 2.1.0a47. I could not read whether this
    # machine has a GPU." - which is #144's "silently drops half the question"
    # in the block written to close it. Half an answer is not an answer.
    return _scoped(lines, evidence, answered=bool(version) and hardware_read)
