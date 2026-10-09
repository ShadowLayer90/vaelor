"""Deterministic policy refusals that run before memory or model inference."""

from __future__ import annotations

import re
from typing import Any, Dict, Iterable, Optional

from . import assistant_console_places as places

_OVERRIDE_REQUEST = re.compile(
    r"\b(?:ignore|bypass|override|disregard)\b.{0,80}"
    r"\b(?:approval|policy|rules?|safety|safeguards?|instructions?)\b",
    re.IGNORECASE,
)
#: **A token is a secret unless the sentence is about a rate or a count
#: (review B7, then its adversarial review).** A bare ``tokens?`` refused "what
#: is my current tokens per second?" and "how many tokens does my context
#: window hold?" - the Assistant's own serving speed, refused as a password.
#: The first repair narrowed the word to credential compounds and let "show me
#: the token", "print the cluster token" and "what's the ingest token" through:
#: a VD-100 regression. The word stays a secret; only what follows it, or a
#: count in front of it, makes it a measure (`_secrets_in`).
_SECRET_TERM = re.compile(
    r"\b(?:credentials?|passwords?|passphrases?|secrets?|tokens?|"
    r"private\s+keys?|api[_ -]?keys?|ssh\s+keys?|recovery\s+codes?|"
    r"(?:ingest|join|enrol(?:l)?ment|access|signing)\s+keys?)\b",
    re.IGNORECASE,
)
#: What makes "token" a measure rather than a credential, when it follows.
#: The first branch is a word straight after "token"; the second allows up to
#: four words between, so "per second" and "throughput" live only there (in
#: the first they were branches the second already covered - every branch is
#: deleted in turn by `tests/test_assistant_pattern_branches.py`).
_TOKEN_MEASURE_AFTER = re.compile(
    r"^\s*(?:per\s+minute|/\s*s\b|a\s+second|counts?|"
    r"generated|limits?|budgets?|speed|rate)\b|"
    r"^(?:\s+\w+){0,4}\s+(?:context|window|per\s+second|throughput)\b",
    re.IGNORECASE,
)
#: ... or when a count stands in front of it.
_TOKEN_MEASURE_BEFORE = re.compile(
    r"\bhow\s+many\s+$|\b\d+\s*$|\bnumber\s+of\s+$", re.IGNORECASE)
#: A secret word that names a *screen*, not the secrets on it: "show my API
#: keys page" asks where the page is (review B7). Exempt only when nothing
#: after the screen word asks for a value ("... settings and the key").
_SCREEN_AFTER = re.compile(r"^\s*(?:page|screen|tab|settings)\b", re.IGNORECASE)
_VALUE_AFTER_SCREEN = re.compile(
    r"\b(?:keys?|tokens?|values?|secrets?|passwords?|credentials?)\b", re.IGNORECASE)
#: "what is the X token", "what's my key" - asking for the value itself.
_WHAT_IS = re.compile(r"what(?:'s|\s+is|\s+are)\b", re.IGNORECASE)


def _secrets_in(request: str, start: int = 0, end: Optional[int] = None):
    """Every secret word in a span, skipping a token that is a measure."""
    stop = len(request) if end is None else min(end, len(request))
    for match in _SECRET_TERM.finditer(request, start, stop):
        if match.group(0).lower().startswith("token"):
            if (_TOKEN_MEASURE_AFTER.search(request[match.end():])
                    or _TOKEN_MEASURE_BEFORE.search(request[:match.start()])):
                continue
        yield match


def _first_secret(request: str, start: int = 0, end: Optional[int] = None):
    return next(_secrets_in(request, start, end), None)


def _screen_only(after: str) -> bool:
    """Whether the words after a secret name only a screen, asking for no value."""
    screen = _SCREEN_AFTER.search(after)
    return bool(screen) and not _VALUE_AFTER_SCREEN.search(after[screen.end():])


_DISCLOSURE_VERB = re.compile(
    r"\b(?:give|get|provide|return|list|show|reveal|expose|print|dump|display|"
    r"send|export|retrieve|read|access|tell|cat|copy|paste)\b",
    re.IGNORECASE,
)
_STRONG_DISCLOSURE_VERB = re.compile(
    r"\b(?:reveal|expose|dump)\b", re.IGNORECASE
)
_INVENTORY_QUALIFIER = re.compile(
    r"\b(?:all|every|any|my|our|stored|saved|raw|actual|current|vault|"
    r"credential\s+broker)\b",
    re.IGNORECASE,
)
_SAFE_HELP_BETWEEN = re.compile(
    r"\b(?:how\s+to|why|change|reset|rotate|store|protect|secure|configure|"
    r"security\s+controls?)\b",
    re.IGNORECASE,
)
_SAFE_TOPIC_AFTER = re.compile(
    r"^\s*(?:policy|requirements?|reset|rotation|storage|handling|security|"
    r"status|configuration|rejected|invalid|failure|error)\b",
    re.IGNORECASE,
)
_INVENTORY_QUESTION_CUE = re.compile(r"\b(?:what|which)\b", re.IGNORECASE)
_SAFE_HELP_BEFORE = re.compile(
    r"\b(?:how\s+(?:do|can|should)\s+(?:i|we)|"
    r"where\s+can\s+(?:i|we))\s*$",
    re.IGNORECASE,
)
_HAVE_SECRET_INVENTORY = re.compile(
    r"\b(?:do|does|did)\b.{0,24}\b(?:have|store|hold|retain|keep)\b",
    re.IGNORECASE,
)
_SAFE_SECRET_CONTEXT = re.compile(
    r"\b(?:how\s+to|why|reset|rotate|rotation|protect|secure|securely|security|"
    r"policy|requirements?|storage|handling|status|configuration|rejected|"
    r"invalid|failure|error)\b",
    re.IGNORECASE,
)
_DESTRUCTIVE_IMPERATIVE = re.compile(
    r"(?:^|[.!?]\s*)(?:please\s+)?"
    r"(?:shut\s*down|reboot|delete|wipe|erase|destroy|format|disable\s+cooling|"
    r"stop\s+(?:the\s+)?fans?|"
    r"restart\s+(?:the\s+|this\s+|my\s+)?(?:machine|system|appliance|computer|box|host|worker))\b",
    re.IGNORECASE,
)

#: Which kind of destructive request it was, so the Assistant can name the
#: control that really does it (review S4). The specialist path keeps the
#: one read-only refusal; these only choose the Assistant's own wording.
_POWER_REQUEST = re.compile(
    r"\b(?:shut\s*down|reboot|power\s+off|power\s+down|power\s+cycle|restart)\b", re.IGNORECASE)
_COOLING_REQUEST = re.compile(
    r"\b(?:disable\s+cooling|stop\s+(?:the\s+)?fans?)\b", re.IGNORECASE)

#: The Assistant's answer for each kind, naming the screen that owns it.
_DESTRUCTIVE_ANSWERS = (
    (_POWER_REQUEST, (
        "I have not restarted or shut down this machine: the Assistant does "
        "not run power actions. {} are on {}, and each one asks you to "
        "confirm first.".format(places.POWER_CONTROL, places.POWER_PLACE)
    ), "Open {} to restart or shut down.".format(places.POWER_PLACE)),
    (_COOLING_REQUEST, (
        "I have not changed the fans or cooling: the Assistant does not "
        "change hardware settings. Use {} on {}.".format(
            places.COOLING_CONTROL, places.COOLING_PLACE)
    ), "Open {} to change the fans.".format(places.COOLING_PLACE)),
)
_DELETE_ANSWER = (
    "I have not deleted anything: the Assistant does not remove data. "
    "Restore points are removed on {}, and apps and their data on {}, where "
    "each removal is reviewed first.".format(places.RECOVERY_PLACE, places.APPS)
)
_DELETE_STEP = "Open {} for {}, or {} to remove an app.".format(
    places.ACTIVITY, places.RECOVERY_CONTROL, places.APPS)
#: A power request naming a cluster worker: no control restarts that machine.
_WORKER_POWER_ANSWER = (
    "I have not restarted or shut down {name}: Vaelor has no control that "
    "restarts or shuts down a worker machine; {power} ({place}) act on this "
    "controller only. Restart {name} from the machine itself."
)


def _secret_disclosure_requested(request: str) -> bool:
    for cue in _INVENTORY_QUESTION_CUE.finditer(request):
        secret = _first_secret(request, cue.end(), cue.end() + 160)
        if secret is None:
            continue
        window = request[cue.end():cue.end() + 160]
        after = request[secret.end():cue.end() + 160]
        if _SAFE_TOPIC_AFTER.search(after) or _screen_only(after):
            continue
        if _INVENTORY_QUALIFIER.search(window) or _HAVE_SECRET_INVENTORY.search(after):
            return True
        # "what is the ingest token", "what's my key": the value itself.
        if _WHAT_IS.match(request, cue.start()) and secret.start() - cue.end() <= 60:
            return True
    for verb in _DISCLOSURE_VERB.finditer(request):
        secret = _first_secret(request, verb.end(), verb.end() + 120)
        if secret is None:
            continue
        before = request[max(0, verb.start() - 40):verb.start()]
        between = request[verb.end():secret.start()]
        after = request[secret.end():secret.end() + 40]
        if _STRONG_DISCLOSURE_VERB.fullmatch(verb.group(0)):
            return True
        if _screen_only(request[secret.end():]):
            continue
        if _SAFE_HELP_BETWEEN.search(between):
            continue
        if _SAFE_HELP_BEFORE.search(before) and not _INVENTORY_QUALIFIER.search(between):
            continue
        if (
            _SAFE_TOPIC_AFTER.search(after)
            and not _INVENTORY_QUALIFIER.search(between)
        ):
            continue
        return True
    for secret in _secrets_in(request):
        before = request[max(0, secret.start() - 120):secret.start()]
        after = request[secret.end():secret.end() + 120]
        if _screen_only(request[secret.end():]):
            continue
        reverse_verb = _DISCLOSURE_VERB.search(after)
        if reverse_verb and not _SAFE_TOPIC_AFTER.search(after):
            return True
        if (
            _INVENTORY_QUALIFIER.search(before + after)
            and not _SAFE_SECRET_CONTEXT.search(before + after)
        ):
            return True
    return False


def policy_denial(request: str) -> Optional[str]:
    if _OVERRIDE_REQUEST.search(request):
        return "The request tries to override Vaelor's approval or safety policy."
    if _secret_disclosure_requested(request):
        return (
            "Vaelor cannot access, reveal, enumerate, or transmit passwords, "
            "credentials, or other secrets through the Assistant."
        )
    if _DESTRUCTIVE_IMPERATIVE.search(request):
        return "Read-only specialists cannot perform or recommend destructive actions."
    return None


def specialist_refusal(request: str) -> Optional[Dict[str, Any]]:
    denial = policy_denial(request)
    if not denial:
        return None
    return {
        "summary": "The specialist blocked an unsafe request.",
        "findings": [
            denial,
            "No changes were executed and no secrets were accessed.",
        ],
        "recommendations": [
            "Rephrase the request as a read-only diagnostic or safety review."
        ],
        "next_actions": [
            "Use a dedicated reviewed control only for a legitimate operator action."
        ],
    }


def _destructive_answer(request: str, workers: Iterable[str] = ()) -> Optional[Dict[str, Any]]:
    """The Assistant's own answer to a power, cooling or delete request (S4).

    The read-only specialist refusal was the only wording, so "Reboot the
    machine" was told that *read-only specialists* cannot act - naming a
    component the owner never asked about and no control, recorded as
    answered. Nothing was done, so ``answered`` is ``False``, and the reply
    names the screen that really does it.
    """
    if not _DESTRUCTIVE_IMPERATIVE.search(request):
        return None
    answer, step = _DELETE_ANSWER, _DELETE_STEP
    for pattern, wording, action in _DESTRUCTIVE_ANSWERS:
        if pattern.search(request):
            answer, step = wording, action
            break
    named = list(workers)
    if named and _POWER_REQUEST.search(request):
        # Adversarial review should-fix 5: "Reboot Worker A" was pointed at
        # this controller's own power controls.
        answer = _WORKER_POWER_ANSWER.format(
            name=" or ".join(named), power=places.POWER_CONTROL, place=places.POWER_PLACE)
        step = "Restart {} from the machine itself.".format(" or ".join(named))
    return {
        "answer": answer,
        "evidence": [{
            "source": "assistant.policy",
            "summary": "Power, cooling and removal changes are made on their own "
                       "screens with a confirmation, never from this chat.",
        }],
        "suggested_actions": [step],
        "proposed_job": None,
        "answered": False,
    }


def assistant_refusal(request: str, workers: Iterable[str] = ()) -> Optional[Dict[str, Any]]:
    """The policy answer for ``request``; ``workers`` are cluster workers it names."""
    denial = policy_denial(request)
    if not denial:
        return None
    if not _OVERRIDE_REQUEST.search(request) and not _secret_disclosure_requested(request):
        destructive = _destructive_answer(request, workers)
        if destructive is not None:
            return destructive
    return {
        "answer": (
            "I blocked that request. {} No changes were executed and no secrets "
            "were accessed. Ask for a read-only diagnostic, or use the dedicated "
            "reviewed control for a legitimate operator action."
        ).format(denial),
        "evidence": [{
            "source": "assistant.policy",
            "summary": "Vaelor blocked an unsafe or approval-bypassing request.",
        }],
        "suggested_actions": [
            "Rephrase this as a read-only safety or system-health question."
        ],
        "proposed_job": None,
    }
