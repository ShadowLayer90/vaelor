"""Catch an answer that declines an appliance question, whoever wrote it.

VD-042: *everything* about the appliance belongs to the Assistant. Enforcing
that needs three points, and this is the second.

The first point is the routing gate - :func:`vaelor.assistant_intents.knowledge_redirect`
declines to redirect an appliance question outward. That gate can only stop a
redirect it is *consulted* about. The refusal a live tester actually saw on the
Z2 - *"Outside what I cover. AI Chat is the place to ask."* - never touched it:
the model wrote that sentence itself, out of its own system prompt, and the
routing gate was not bypassed so much as never asked. Adding vocabulary to a
routing table cannot stop a model writing that sentence, and no amount of
prompt wording can either, because a prompt is a request rather than a check.

So the check has to happen on the way out, on the *text*, without caring who
composed it. If the question was about this appliance, an answer that declines
it is wrong whatever produced it - the built-in redirect, a small local model,
or a frontier provider being conscientious about what it cannot see.

**What this does not do.** It never invents a reading to put in place of the
refusal. By the time an answer reaches here the reading-backed paths in
:mod:`vaelor.deployment_agent` have already had their turn and returned
directly, so a deflection arriving here means no reading answered the question.
The replacement says exactly that, and keeps the question on this surface
instead of posting it to one that cannot read the machine.
"""

from __future__ import annotations

import ast
import json
import re
from typing import Any, Dict, List, Optional, Tuple

from .assistant_answer_presentation import is_appliance_question

#: Prose that declines the *subject* of the question.
#:
#: Every pattern here has to be a refusal of scope, never a report of a
#: failure. The distinction is load-bearing and was drawn against the
#: appliance's own sentences:
#:
#: * ``connected_model_failure_answer`` - "The selected AI connection did not
#:   answer this request, so I cannot answer it reliably without guessing" - is
#:   a true statement about a real outage and must survive untouched. A pattern
#:   as loose as ``I cannot answer`` would swallow it and replace an accurate
#:   outage report with a scope notice, which is the same class of defect in
#:   the opposite direction.
#: * ``assistant_refusal`` - "I blocked that request..." - is a safety refusal.
#:   Vaelor declining to enumerate credentials is correct on an appliance
#:   question and must not be overridden by a scope guard.
#:
#: What is matched instead is the shape of a *subject* refusal: this is not
#: mine, I cannot see this machine, take it somewhere else.
#: Who the refusal is about, since the model does not reliably know.
#:
#: **The patterns were first person and the model writes second person.**
#: Every one of them was built from a sentence a tester was shown - "outside
#: what *I* cover" - and the prompt this model is given opens "*You* cover
#: this appliance only", so what it writes back is "that is outside what *you*
#: cover". Verified against the live text: `guarded_answer` did not fire on
#: it, and the refusal reached the owner reported as an answer.
#:
#: The pronoun was never the point. A refusal of the subject is wrong on an
#: appliance question whoever it says is refusing, so the person is matched
#: rather than assumed - here, once, instead of in each pattern.
_SPEAKER = r"(?:i|you|we|it|this\s+assistant|vaelor)"
_POSSESSIVE = r"(?:my|your|our|its|this\s+assistant'?s?)"

#: The machine as a whole, as the object of "I cannot see ...".
#:
#: A determiner, optionally one possessive noun ("your *device's* sensors"),
#: then a word that names the appliance rather than a reading taken from it -
#: and nothing after it, so "the system *logs*" and "the host *temperature*"
#: stay the named readings they are. See the patterns that use it for why the
#: object became mandatory (#189).
_WHOLE_MACHINE = (
    r"(?:your|this|the|my)\s+(?:\w+'s\s+)?"
    r"(?:machine|appliance|device|hardware|server|node|host|system|"
    r"environment|sensors?|readings?)(?:'s)?\b(?!\s+[a-z])"
)

_DEFLECTIONS: tuple[re.Pattern[str], ...] = (
    # "Outside what I cover", "that is outside of what you cover/handle/do".
    re.compile(
        r"\boutside\s+(?:of\s+)?(?:what\s+" + _SPEAKER +
        r"\s+(?:cover|covers|handle|handles|do|does|can\s+do)|" +
        _POSSESSIVE +
        r"\s+(?:scope|remit|coverage|area|capabilities|expertise))\b",
        re.IGNORECASE,
    ),
    # "not something I cover", "this isn't something you can help with".
    re.compile(
        r"\bnot\s+(?:something|a\s+(?:question|topic))\s+(?:that\s+)?" +
        _SPEAKER +
        r"\s+(?:cover|covers|handle|handles|deal[s]?\s+with|"
        r"can\s+help(?:\s+with)?)\b",
        re.IGNORECASE,
    ),
    # "beyond my scope", "outside my scope" is covered above; this is the
    # other preposition the same sentence is written with.
    re.compile(
        r"\bbeyond\s+(?:" + _POSSESSIVE +
        r"\s+(?:scope|remit|capabilities)|what\s+" + _SPEAKER +
        r"\s+(?:cover|covers))\b",
        re.IGNORECASE,
    ),
    # "I don't cover that", "you do not cover questions about".
    re.compile(
        r"\b" + _SPEAKER + r"\s+(?:don'?t|doesn'?t|do\s+not|does\s+not)\s+"
        r"cover\b",
        re.IGNORECASE,
    ),
    # **"Outside the scope of the available facts."** Nobody's scope is named,
    # so none of the possessive patterns above could reach it, and it is the
    # phrasing a model reaches for when it has been told to answer only from
    # supplied evidence. Three of the live refusals were this sentence.
    re.compile(
        r"\b(?:outside|beyond)\s+(?:of\s+)?the\s+scope\b"
        r"|\bnot\s+(?:with)?in\s+(?:the\s+)?scope\b"
        r"|\bno\s+(?:relevant\s+)?(?:facts?|data|information|readings?)\s+"
        r"(?:are\s+|is\s+|were\s+|was\s+)?(?:available|supplied|provided)\b",
        re.IGNORECASE,
    ),
    # Posting an appliance question to the general-chat surface. This is the
    # sentence the live tester saw, and the one the routing gate exists to
    # prevent - so it is caught here too, for the author the gate never sees.
    re.compile(
        r"\bai\s*chat\s+is\s+(?:the\s+place|where)\b"
        r"|\bask\s+(?:it|that|this)\s+(?:over\s+)?in\s+ai\s*chat\b"
        r"|\b(?:try|use|head\s+to|go\s+to)\s+ai\s*chat\b",
        re.IGNORECASE,
    ),
    # A model claiming it cannot see the machine it is speaking for. True of
    # the model in isolation, false of this Assistant, which has fact tools.
    #
    # **The object is required, and #189 is why.** These two patterns stopped
    # at the verb, so *"I cannot read the GPU temperature"* was a deflection -
    # and an answer stating the CPU temperature, the fan speed, memory use and
    # free space, with that one honest sentence at the end, was replaced whole
    # by :data:`_WITHHELD_ANSWER`. The owner asked for four readings the
    # appliance had and got a refusal, produced by the guard against refusals.
    #
    # A sentence naming *one* reading it could not take is not a refusal of
    # the subject; it is what `answer_evidence.describe_missing` exists to
    # write, and LESSONS pattern 4 is eight instances of this product denying
    # what it could read. So the claim has to be about the machine itself.
    #
    # `(?!\s+[a-z])` keeps the noun at the end of its phrase: "the *system*
    # logs" and "the *host* temperature" are named readings that happen to
    # contain a machine word, and without it both were deflections. The cost
    # is stated rather than hidden - "I cannot access your machine directly"
    # is now missed here. That direction is the cheap one: a missed refusal
    # reaches an owner as a visible bad answer, while a false positive
    # silently destroys readings this machine took, which is the defect above.
    # The general refusal is also reached by six other patterns in this table.
    re.compile(
        r"\bi\s+(?:can'?t|cannot|am\s+not\s+able\s+to|do\s+not\s+have\s+"
        r"(?:the\s+ability\s+to|access\s+to))\s+"
        r"(?:see|read|access|check|monitor|inspect|view|retrieve)\s+"
        r"(?:" + _WHOLE_MACHINE + r"|anything|it\b|that\b)",
        re.IGNORECASE,
    ),
    re.compile(
        r"\bi\s+(?:don'?t|do\s+not)\s+have\s+(?:direct\s+)?access\s+to\s+"
        + _WHOLE_MACHINE,
        re.IGNORECASE,
    ),
    # The built-in redirect's own sentence. Correct for a genuinely general
    # question, wrong for an appliance one - and the only way it reaches an
    # appliance question is a hole in the routing gate, which is exactly the
    # case worth catching twice.
    re.compile(
        r"\bnothing\s+i\s+can\s+read\s+on\s+this\s+machine\s+covers\b",
        re.IGNORECASE,
    ),
)

#: Sources whose refusal is about safety rather than subject.
#:
#: A policy refusal is a decision this appliance made on purpose, and the
#: patterns above are written not to match one. Naming the source as well is
#: belt and braces: if a future denial sentence is reworded into something that
#: reads like a scope refusal, the guard must still not overturn it.
SAFETY_SOURCES = frozenset({"policy-refusal"})

SCOPE_GUARD_EVIDENCE = "assistant.scope-guard"

_WITHHELD_ANSWER = (
    "That question is about this appliance, so it is mine to answer rather "
    "than something to hand to a general-knowledge surface. I could not answer "
    "it from what I can read on this machine just now, and I will not guess a "
    "reading in place of one. Name the reading you want - a temperature, free "
    "space, a service, an alert - or read the current values on "
    "System > Hardware & services."
)

#: Said beside the readings that survived, when only part of a reply was
#: handed away. Written not to trip :func:`declines_the_subject`, so the guard
#: can be applied to its own output without eating it a second time.
_PARTLY_WITHHELD_NOTE = (
    "One part of that reply passed this question to a general-knowledge "
    "surface. Questions about this appliance are mine, so that part was "
    "withheld; everything above was read on this machine and stands."
)

#: Sentence ends, keeping the whitespace so a rebuilt reply reads as one.
_SENTENCE_END = re.compile(r"(?<=[.!?])(\s+)")


# ---------------------------------------------------------------------------
# A model that echoes its own prompt envelope instead of answering.
#
# Measured live 2/2 on the NPU model: asking "Monitoring is paused. Please
# resume live monitoring for me." rendered, as the assistant's answer, the raw
# internal prompt envelope verbatim -
# ``{"question":"...","context":{"facts":{...,"version":"2.1.0a97"},
# "conversation":[...],"memories":["Monitor state: PAUSED..."],"sources":[...]}``
# - leaking internal context, curated memories and the version string to the
# user, and doing it even when the JSON was malformed. The raw model text is
# rendered verbatim by the answer path, so the refusal has to happen here, on
# the *text*, the same way :func:`declines_the_subject` catches a scope refusal
# whoever composed it. A degenerate echo is neither a refusal nor an answer, so
# it is caught ahead of both and never depends on the question's scope: internal
# context must not reach the user however the question was phrased.
#
# The prompt envelope's shape is its signature: the outer ``question`` and
# ``context`` frame wrapping ``facts``/``memories``/``conversation``/
# ``sources``. Detection keys on that structure - the count of these six words
# used as *object keys* - not on the mere presence of braces, and not on any one
# key being present, so a legitimate answer that quotes a small object (a compose
# fragment, ``{"context":"prod"}``) is untouched: it reproduces at most one of
# them. A reply would have to reproduce the envelope's key structure to trip
# this, which prose that merely mentions the words never does.
#
# **Quote style is not part of the signature (adversarial review).** The first
# cut hard-gated on the literal double-quoted ``"context"`` and matched only
# ``"key":``. Three leaks walked through it: an envelope with the ``context``
# wrapper renamed away, and a single-quoted Python ``repr`` of the same object.
# An object key is a key in either quote style, so the patterns match ``"key":``
# and ``'key':`` alike, the ``context`` gate is gone, and ``_parsed_object``
# also tries ``ast.literal_eval`` for the single-quoted repr.
#
# **Written as one literal `re.compile` per key, not a comprehension, because
# `tests/test_vocabulary_reachability.py` walks this tuple and places each entry
# by its source text.** A comprehension over `re.escape(key)` builds a pattern
# string that appears nowhere in the source, so the reachability guard could
# neither trace it to a line nor carry a waiver on it - the "placed in source"
# failure that guard exists to catch. Each of the six is a distinct table entry,
# exercised by `test_assistant_scope_guard`, so none is untabled.
_ENVELOPE_KEY_PATTERNS = (
    re.compile(r"""["']question["']\s*:"""),
    re.compile(r"""["']context["']\s*:"""),
    re.compile(r"""["']facts["']\s*:"""),
    re.compile(r"""["']memories["']\s*:"""),
    re.compile(r"""["']conversation["']\s*:"""),
    re.compile(r"""["']sources["']\s*:"""),
)
#: How many of the six envelope keys, used as object keys, mark an echo. Four of
#: six means the text has reproduced the envelope frame rather than mentioning
#: one of its words in prose. Each pattern names a distinct key, so counting the
#: patterns that match counts the distinct keys present.
_ENVELOPE_KEY_THRESHOLD = 4

# **The internal-context fingerprint (adversarial review).** A partial leak that
# never reproduces the full frame - ``{"memories":[...],"version":"2.1.0a97"}`` -
# still hands the user curated memories beside the appliance's own identity.
# ``memories`` as an object key beside a ``version`` or ``system.identity`` key
# is that combination, in either quote style, and is withheld on its own. Each
# alone is innocuous (an answer may mention a version); together as object keys
# they are the internal context escaping.
_MEMORIES_KEY = re.compile(r"""["']memories["']\s*:""")
_IDENTITY_KEY = re.compile(r"""["'](?:version|system\.identity)["']\s*:""")

#: Evidence source for a withheld degenerate response, kept distinct from the
#: scope guard's own source so the audit trail tells the two apart.
DEGENERATE_RESPONSE_EVIDENCE = "assistant.response-guard"

#: Shown in place of a prompt-envelope echo. An honest degrade: the model
#: produced nothing usable, so nothing usable is claimed, and the user is sent
#: to rephrase or to the screen that holds the live values.
_UNUSABLE_MODEL_ANSWER = (
    "I could not produce a clear answer to that. The connected model returned "
    "an internal, unusable response instead of an answer, so there is nothing "
    "reliable to show. Try rephrasing your question, or read the current "
    "values on System > Hardware & services."
)


def _parsed_object(text: str) -> Any:
    """The object ``text`` is, or the one embedded in it, or ``None``.

    A well-formed echo parses whole; one wrapped in a little prose parses from
    its first ``{`` to its last ``}``. Both ``json.loads`` (double-quoted) and
    ``ast.literal_eval`` (a single-quoted Python ``repr``) are tried, so quote
    style does not decide whether a leak parses. A malformed echo parses as
    neither, which is why parsing is only the *secondary* signal here - the
    structural key count catches the malformed case the model was observed to
    emit.
    """
    for candidate in (text.strip(), text[text.find("{"): text.rfind("}") + 1]):
        if not candidate:
            continue
        for loader in (json.loads, ast.literal_eval):
            try:
                return loader(candidate)
            except (ValueError, TypeError, SyntaxError, MemoryError, RecursionError):
                continue
    return None


def echoes_prompt_envelope(text: Any) -> bool:
    """Whether ``text`` is the internal prompt envelope rather than an answer.

    True when the text has reproduced the envelope's key structure - four of the
    six envelope words used as object keys, in either quote style - or carries
    the internal-context fingerprint (``memories`` beside ``version`` or
    ``system.identity`` as object keys), or parses to an object carrying the
    outer ``question``/``context`` pair. Keyed on the envelope's own structure so
    a legitimate answer that merely mentions these words in prose, or quotes a
    small unrelated object, is never mistaken for one.
    """
    candidate = str(text or "")
    if "{" not in candidate:
        return False
    # **The truncated-envelope opener (found live).** Asked "are there any issues
    # I need to know about?", the model reflected the envelope but ran out of
    # output budget partway through ``services.status``, so only three of the six
    # keys - ``question``, ``context``, ``facts`` - were ever emitted, one under
    # the threshold below. The reply leaked to the owner reported as an answer.
    # The envelope always opens with the outer ``question`` key, and a genuine
    # answer never begins ``{"question":``, so that opener is an echo whatever
    # the model managed to emit after it.
    opener = candidate.lstrip()
    if opener.startswith('{"question"') or opener.startswith("{'question'"):
        return True
    matched = sum(
        1 for pattern in _ENVELOPE_KEY_PATTERNS if pattern.search(candidate)
    )
    if matched >= _ENVELOPE_KEY_THRESHOLD:
        return True
    # Partial leak: curated memories escaping beside the appliance's identity,
    # without the full frame around them.
    if _MEMORIES_KEY.search(candidate) and _IDENTITY_KEY.search(candidate):
        return True
    parsed = _parsed_object(candidate)
    return (
        isinstance(parsed, dict)
        and "question" in parsed
        and isinstance(parsed.get("context"), (dict, list))
    )


def _withheld_echo_reply(payload: Dict[str, Any]) -> Dict[str, Any]:
    """Replace a prompt-envelope echo with the honest-failure answer.

    The raw model text and any evidence it carried are dropped whole - the
    point is that none of it reaches the user - and replaced by the guard's own
    note, so the audit trail shows a response withheld rather than a silent
    rewrite. ``answered`` is false because nothing was answered.
    """
    guarded = dict(payload)
    guarded["answer"] = _UNUSABLE_MODEL_ANSWER
    guarded["evidence"] = [{
        "source": DEGENERATE_RESPONSE_EVIDENCE,
        "summary": (
            "The connected model returned its internal prompt context instead "
            "of an answer. Vaelor withheld the raw output so no internal "
            "context, memories or version data reached the user, and produced "
            "no answer to show in its place."
        ),
    }]
    guarded["suggested_actions"] = []
    guarded["proposed_job"] = None
    guarded["approval_required"] = False
    guarded["answered"] = False
    guarded["degenerate_response"] = True
    return guarded


def declines_the_subject(text: Any) -> bool:
    """Whether ``text`` refuses the question's subject rather than reporting a failure."""
    candidate = str(text or "")
    return any(pattern.search(candidate) for pattern in _DEFLECTIONS)


def _without_the_deflections(text: Any) -> Tuple[str, int]:
    """The sentences that do not decline, and how many did.

    **Granularity is the whole of #189.** The guard asked one question of a
    whole reply - does any part of this decline - and answered it by throwing
    the reply away. A reply that states four readings and hands one clause
    elsewhere has answered four things, and replacing all of it with a refusal
    is the same loss the guard exists to prevent, arriving from inside it.
    Dropping a sentence is not dropping an answer.
    """
    parts = _SENTENCE_END.split(str(text or ""))
    kept: List[str] = []
    withheld = 0
    for index in range(0, len(parts), 2):
        sentence = parts[index].strip()
        if not sentence:
            continue
        if declines_the_subject(sentence):
            withheld += 1
        else:
            kept.append(sentence)
    return " ".join(kept), withheld


def misrouted_refusal(question: Any, answer_text: Any, source: str = "") -> bool:
    """Whether this answer declines a question VD-042 puts on this surface."""
    if str(source or "") in SAFETY_SOURCES:
        return False
    if not is_appliance_question(question):
        return False
    return declines_the_subject(answer_text)


def _not_a_success(payload: Dict[str, Any], source: str) -> Dict[str, Any]:
    """Record a reply that declined as unanswered, whatever its subject.

    **18 of 52 live answers refused in their own text and reported
    ``answered: True``.** The flag was a per-caller default -
    ``_validate_answer(..., answered=True)`` for every ``connected-model``
    reply - and a default cannot know what the model wrote. So the audit trail
    counted "that is outside what you cover" as a success, and the Assistant's
    run history showed a column of them.

    Derived from :func:`declines_the_subject`, which is the one definition of
    "this text refuses" in this module. The guard's *replacement* and the
    audit *flag* were two encodings of that idea; now they are one, and a
    pattern added above corrects both at once.

    Separate from :func:`guarded_answer`'s replacement on purpose. A refusal
    of a genuinely general question is the right words - the text stays - and
    it still did not answer anything.
    """
    if source in SAFETY_SOURCES or not payload.get("answered"):
        return payload
    if not declines_the_subject(payload.get("answer", "")):
        return payload
    return {**payload, "answered": False}


def guarded_answer(question: Any, payload: Dict[str, Any]) -> Dict[str, Any]:
    """Return ``payload``, or an honest in-scope reply where it declined.

    The replacement keeps whatever evidence the original answer carried - the
    readings it did gather are still true - and records that something was
    withheld, so the audit trail shows a guard firing rather than a silent
    rewrite. ``answered`` goes false because nothing was answered: replacing a
    refusal with a different refusal does not make it an answer, and claiming
    otherwise is the defect the ``answered`` flag was added to stop.

    **What is withheld is the declining sentence, not the reply.** Where other
    sentences survive they are kept and the notice is appended, so an answer
    that stated four readings and handed one clause away still delivers the
    four. ``answered`` is false either way, on the rule
    :func:`vaelor.assistant_answer_scope.identity_answer` already follows:
    half an answer is not an answer.
    """
    if not isinstance(payload, dict):
        return payload
    # A degenerate response - the model echoing its own prompt envelope - is
    # neither a refusal nor an answer, and is caught ahead of the scope check
    # because it must never be shown whatever the question's scope was. It
    # leaks internal context, memories and the version string otherwise.
    if echoes_prompt_envelope(payload.get("answer", "")):
        return _withheld_echo_reply(payload)
    source = str(payload.get("source", ""))
    if not misrouted_refusal(question, payload.get("answer", ""), source):
        return _not_a_success(payload, source)
    kept, withheld = _without_the_deflections(payload.get("answer", ""))
    guarded = dict(payload)
    guarded["answer"] = (
        "{}\n\n{}".format(kept, _PARTLY_WITHHELD_NOTE) if kept
        else _WITHHELD_ANSWER
    )
    guarded["evidence"] = [
        {
            "source": SCOPE_GUARD_EVIDENCE,
            "summary": (
                "{} sentence{} of the prepared answer declined a question "
                "about this appliance. VD-042 puts appliance questions on "
                "this surface, so {} withheld rather than shown.".format(
                    withheld, "" if withheld == 1 else "s",
                    "it was" if withheld == 1 else "they were",
                )
            ),
        },
        *[item for item in payload.get("evidence", []) if isinstance(item, dict)],
    ][:10]
    guarded["suggested_actions"] = []
    guarded["proposed_job"] = None
    guarded["approval_required"] = False
    guarded["answered"] = False
    guarded["scope_guarded"] = True
    return guarded


def scope_guarded(payload: Optional[Dict[str, Any]]) -> bool:
    """Whether this answer was replaced by the guard."""
    return bool(isinstance(payload, dict) and payload.get("scope_guarded"))
