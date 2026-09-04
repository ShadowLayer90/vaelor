"""Detect change requests so they are never answered as a status readout.

"can you turn the case lights to purple" was previously answered by reporting
the *current* lighting state: the request was neither performed, refused, nor
acknowledged, and the requested value ("purple") was dropped entirely. A user
reasonably reads that as "done".

This module recognises the imperative half of a message, keeps the target value
the user actually asked for, and names the screen that owns the change, so the
caller can either raise an approval or decline in plain language. It never
performs anything itself.

Two failures shaped the current design:

* The request pattern was anchored with ``^``, so it only fired when the change
  verb opened the clause. Ordinary phrasing - "I'd like the lights set to
  blue", "when you have a moment set the case lights to red", "the lights
  should be green" - was invisible, and those turns were answered as status
  readouts. Detection now strips politeness, then looks for the request
  anywhere in the clause, refusing clauses that are questions about state.
* A detected request could be *both* proposed and refused in one reply, because
  coverage was decided by testing whether the bare area key ("lighting")
  appeared as a word in the proposal text. A proposal named "compose.restart"
  or a matched custom agent never contained it, so an approval card and "I have
  not changed case lighting" were emitted together. ``outstanding_actions`` is
  now the single decision point, and it uses the same area vocabulary that
  classified the request in the first place.
"""

from __future__ import annotations

import re
from typing import Any, Dict, List, Optional, Sequence, Tuple

# Verbs that express intent to change something, as opposed to asking about it.
_CHANGE_VERB = (
    r"(?:turn|set|make|change|switch|adjust|dim|brighten|enable|disable|"
    r"put|update|apply|configure|start|stop|restart|reboot|install|remove|"
    r"delete|rename|increase|decrease|raise|lower)"
)

# A message may carry several instructions, so it is split into clauses before
# each is examined. Without this, one greedy match swallows the rest of the
# sentence and later instructions are silently dropped — the exact failure this
# module exists to prevent.
_CLAUSE_SPLIT = re.compile(r"[,;?]|\band\b|\balso\b|\bthen\b", re.IGNORECASE)

# Politeness, address, and hedging that can precede the actual instruction.
# Stripping it is what lets the request be found without anchoring on the verb,
# while still allowing the "is this clause a question?" test below to look at
# the first real word rather than at "when you have a moment".
_COURTESY_PREFIX = (
    r"hey|hi|hello|ok|okay|so|well|also|actually|just|quickly|now|please|"
    r"kindly|maybe|perhaps|vaelor|thanks|thank\s+you|"
    r"go\s+ahead\s+and|feel\s+free\s+to|"
    r"if\s+(?:possible|you\s+can|you\s+could|you\s+don'?t\s+mind)|"
    r"when(?:ever)?\s+you\s+(?:have\s+a\s+moment|get\s+a\s+chance|can|could)"
)

# Forms that make the clause a request even though no imperative verb opens it.
# "i want the case lights purple" carries no change verb at all.
_REQUEST_PREFIX = (
    r"(?:can|could|would|will)\s+(?:you|vaelor|i|we)\s*(?:please\s+)?"
    r"(?:get|have|make\s+it\s+so)?|"
    r"(?:would|do)\s+you\s+mind|"
    r"i(?:'?d|\s+would)\s+like\s+(?:you\s+to|it\s+to|to\s+have)?|"
    r"i\s+(?:want|need)\s+(?:you\s+to|it\s+to)?|"
    r"let'?s|let\s+me\s+have|give\s+me"
)

_COURTESY_PATTERN = re.compile(r"^\s*(?:{})\b[\s,]*".format(_COURTESY_PREFIX), re.IGNORECASE)
_REQUEST_PATTERN = re.compile(r"^\s*(?:{})\b[\s,]*".format(_REQUEST_PREFIX), re.IGNORECASE)

# After politeness is removed, a clause that opens with one of these is a
# question about the appliance, not an instruction to change it: "is the fan
# set to automatic", "what is the lighting set to", "did you change the
# lights". Request forms were already consumed above, so "can"/"could"/"would"
# reaching here means "can the lights be blue?" — a capability question.
_QUESTION_OPENER = re.compile(
    r"^(?:what|which|why|when|where|who|whose|whom|how|whether|"
    r"is|are|was|were|do|does|did|has|have|had|am|"
    r"can|could|would|will|shall|should|may|might)\b",
    re.IGNORECASE,
)

# The same vocabulary, tested anywhere *before* the change verb. It catches the
# embedded question that a leading-word test cannot see: "can you tell me what
# colour the lights are set to" must not be read as "set the lights".
_QUESTION_MARKER = re.compile(
    r"\b(?:what|which|why|when|where|who|whose|whom|how|whether|"
    r"is|are|was|were|do|does|did|has|have|had|am|"
    r"tell\s+me|show\s+me|know|wondering|curious)\b",
    re.IGNORECASE,
)

_CHANGE_VERB_PATTERN = re.compile(r"\b" + _CHANGE_VERB + r"\b", re.IGNORECASE)

# "the lights should be green" states a desired end state without any verb.
_DESIRED_STATE = re.compile(
    r"\b(?:should|needs?\s+to|ought\s+to|has\s+to|have\s+to)\s+be\b", re.IGNORECASE
)

# Phrases that look imperative but are actually questions about capability.
_CAPABILITY_QUESTION = re.compile(
    r"\b(?:how\s+(?:do|can|would)\s+(?:i|we|you)|what\s+happens\s+if|"
    r"is\s+it\s+possible\s+to|should\s+i)\b",
    re.IGNORECASE,
)

_COLOUR_WORDS = (
    "red", "orange", "amber", "yellow", "green", "cyan", "teal", "blue",
    "indigo", "violet", "purple", "magenta", "pink", "white", "warm white",
)
_COLOUR_PATTERN = re.compile(
    r"\b(?:" + "|".join(re.escape(word) for word in _COLOUR_WORDS) + r")\b|#[0-9a-f]{6}\b",
    re.IGNORECASE,
)

# Each area names the screen that owns the change, the hash route the frontend
# can actually navigate to, the control on that screen, and the facts that
# describe it. The route is part of the area definition so that naming a
# destination and being able to reach it can never drift apart.
_AREAS: Tuple[Tuple[str, str, str, str, str, Tuple[str, ...]], ...] = (
    (
        "lighting",
        "System > Case lighting",
        "#/system",
        "Case lighting",
        "case lighting",
        ("light", "lights", "lighting", "rgb", "led", "leds", "colour", "color",
         "brightness", "rainbow"),
    ),
    (
        "cooling",
        "System > Cooling",
        "#/system",
        "Cooling",
        "cooling and fans",
        ("fan", "fans", "cooling", "airflow", "rpm"),
    ),
    (
        "display",
        "System > Hardware & services",
        "#/system",
        "Hardware & services",
        "the front display",
        ("oled", "display", "screen"),
    ),
    (
        "workloads",
        "Workloads",
        "#/workloads",
        "Workloads",
        "installed apps",
        ("app", "apps", "container", "containers", "docker", "stack",
         "compose", "model", "models"),
    ),
    (
        "updates",
        "System > Hardware & services",
        "#/system",
        "Hardware & services",
        "system updates",
        ("update", "updates", "upgrade", "package", "packages"),
    ),
)


# A follow-up rarely names its subject again: "change them to blue instead",
# "make them a little dimmer". When the clause carries an unmistakably
# lighting-shaped value but no noun, treat it as lighting rather than dropping
# the instruction on the floor.
_LIGHTING_VALUE = re.compile(
    r"\b(?:dimmer|brighter|dim|brightness)\b|\b\d{1,3}\s*%", re.IGNORECASE
)
_PRONOUN_SUBJECT = re.compile(r"\b(?:them|it|those|these|that)\b", re.IGNORECASE)


def _area_for(clause: str) -> Optional[Tuple[str, str, str, str, str]]:
    lowered = clause.lower()
    for key, screen, route, control, subject, keywords in _AREAS:
        if any(re.search(r"\b" + re.escape(word) + r"\b", lowered) for word in keywords):
            return key, screen, route, control, subject
    if _PRONOUN_SUBJECT.search(lowered) and (
        _COLOUR_PATTERN.search(lowered) or _LIGHTING_VALUE.search(lowered)
    ):
        return "lighting", "System > Case lighting", "#/system", "Case lighting", "case lighting"
    return None


def areas_named_in(text: str) -> set[str]:
    """Return every configuration area whose vocabulary appears in ``text``.

    Used to decide whether a proposal already covers a detected request. The
    area *key* alone is not enough: a proposal for the same change is named
    "compose.restart" or "Case lights to purple", never "workloads" or
    "lighting".
    """
    lowered = str(text or "").lower()
    if not lowered.strip():
        return set()
    named = set()
    for key, _screen, _route, _control, _subject, keywords in _AREAS:
        vocabulary = (key,) + keywords
        if any(re.search(r"\b" + re.escape(word) + r"\b", lowered) for word in vocabulary):
            named.add(key)
    return named


def _strip_preamble(clause: str) -> Tuple[str, bool]:
    """Remove politeness and request framing; report whether framing was found."""
    remainder = clause.strip()
    request_form = False
    while remainder:
        request = _REQUEST_PATTERN.match(remainder)
        if request is not None and request.end() > 0:
            remainder = remainder[request.end():]
            request_form = True
            continue
        courtesy = _COURTESY_PATTERN.match(remainder)
        if courtesy is not None and courtesy.end() > 0:
            remainder = remainder[courtesy.end():]
            continue
        break
    return remainder.strip(), request_form


def requested_value(clause: str) -> str:
    """Return the concrete value the user asked for, if one is stated."""
    colour = _COLOUR_PATTERN.search(clause)
    if colour:
        return colour.group(0)
    percent = re.search(r"\b(\d{1,3})\s*%", clause)
    if percent:
        return percent.group(0)
    return ""


def _clause_request(clause: str) -> Optional[str]:
    """Return the instruction inside ``clause``, or None when it is a question.

    The instruction may sit anywhere in the clause. What disqualifies it is
    question vocabulary in front of it, not its position.
    """
    remainder, request_form = _strip_preamble(clause)
    if not remainder or _QUESTION_OPENER.match(remainder):
        return None
    verb = _CHANGE_VERB_PATTERN.search(remainder)
    if verb is not None:
        if _QUESTION_MARKER.search(remainder[: verb.start()]):
            return None
        return remainder
    # No verb: only an explicit request form ("i want the lights purple") or a
    # stated end state ("the lights should be green") counts, and only when the
    # user named the value they want.
    if not requested_value(remainder):
        return None
    desired = _DESIRED_STATE.search(remainder)
    if desired is not None:
        if _QUESTION_MARKER.search(remainder[: desired.start()]):
            return None
        return remainder
    if request_form and not _QUESTION_MARKER.search(remainder):
        return remainder
    return None


def detect_action_requests(message: str) -> List[Dict[str, str]]:
    """Return one entry per change the message asks Vaelor to make."""
    text = str(message or "")
    if not text.strip() or _CAPABILITY_QUESTION.search(text):
        return []

    requests: List[Dict[str, str]] = []
    seen: set[str] = set()
    for fragment in _CLAUSE_SPLIT.split(text):
        clause = _clause_request(fragment.strip())
        if clause is None:
            continue
        area = _area_for(clause)
        if area is None:
            continue
        key, screen, route, control, subject = area
        if key in seen:
            continue
        seen.add(key)
        requests.append({
            "area": key,
            "screen": screen,
            "route": route,
            "control": control,
            "subject": subject,
            "request": clause,
            "value": requested_value(clause),
        })
    return requests


def outstanding_actions(
    actions: Sequence[Dict[str, str]],
    answer: Dict[str, Any],
    proposed_agent_task: Optional[Dict[str, Any]] = None,
) -> List[Dict[str, str]]:
    """Return the requests this reply neither proposes nor takes up.

    A reply must propose a change or decline it, never both. Anything returned
    here may be declined; everything else is already represented by an approval
    the user can see, so a decline for it would contradict the same reply.
    """
    if not actions:
        return []
    # A matched custom agent replaces the whole reply with its own approval:
    # the turn *is* a proposal, so nothing in it may also be refused.
    if proposed_agent_task is not None:
        return []
    proposed_job = answer.get("proposed_job") or {}
    if not isinstance(proposed_job, dict):
        proposed_job = {}
    proposal_text = " ".join(
        str(value) for value in (
            proposed_job.get("type", ""),
            proposed_job.get("name", ""),
            proposed_job.get("summary", ""),
            answer.get("application_intent") or "",
        ) if value
    ).strip()
    if not proposal_text:
        return list(actions)
    covered = areas_named_in(proposal_text)
    if not covered and len(actions) == 1:
        # One request, one proposal, no shared vocabulary: the proposal is
        # still the reply to that request. Declining it here is the reported
        # self-contradiction.
        covered = {actions[0]["area"]}
    return [action for action in actions if action["area"] not in covered]


def decline_sentence(action: Dict[str, str]) -> str:
    """Plain-language statement that the change was not made, and where to make it."""
    value = action.get("value", "")
    target = ' to "{}"'.format(value) if value else ""
    return (
        "I have not changed {subject}{target}. I can read appliance facts here, "
        "but changes are made on {screen}."
    ).format(subject=action.get("subject", "that"), target=target, screen=action.get("screen", "its own screen"))
