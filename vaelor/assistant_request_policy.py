"""Deterministic policy refusals that run before memory or model inference."""

from __future__ import annotations

import re
from typing import Any, Dict, Optional

_OVERRIDE_REQUEST = re.compile(
    r"\b(?:ignore|bypass|override|disregard)\b.{0,80}"
    r"\b(?:approval|policy|rules?|safety|safeguards?|instructions?)\b",
    re.IGNORECASE,
)
_SECRET_TERM = re.compile(
    r"\b(?:credentials?|passwords?|passphrases?|secrets?|tokens?|"
    r"private\s+keys?|api[_ -]?keys?|ssh\s+keys?|recovery\s+codes?)\b",
    re.IGNORECASE,
)
_DISCLOSURE_VERB = re.compile(
    r"\b(?:give|get|provide|return|list|show|reveal|expose|print|dump|display|"
    r"send|export|retrieve|read|access|tell)\b",
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
    r"stop\s+(?:the\s+)?fans?)\b",
    re.IGNORECASE,
)


def _secret_disclosure_requested(request: str) -> bool:
    for cue in _INVENTORY_QUESTION_CUE.finditer(request):
        window = request[cue.end():cue.end() + 160]
        secret = _SECRET_TERM.search(window)
        if secret is None:
            continue
        after = window[secret.end():]
        if _SAFE_TOPIC_AFTER.search(after):
            continue
        if _INVENTORY_QUALIFIER.search(window) or _HAVE_SECRET_INVENTORY.search(after):
            return True
    for verb in _DISCLOSURE_VERB.finditer(request):
        secret = _SECRET_TERM.search(request, verb.end(), verb.end() + 120)
        if secret is None:
            continue
        before = request[max(0, verb.start() - 40):verb.start()]
        between = request[verb.end():secret.start()]
        after = request[secret.end():secret.end() + 40]
        if _STRONG_DISCLOSURE_VERB.fullmatch(verb.group(0)):
            return True
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
    for secret in _SECRET_TERM.finditer(request):
        before = request[max(0, secret.start() - 120):secret.start()]
        after = request[secret.end():secret.end() + 120]
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


def assistant_refusal(request: str) -> Optional[Dict[str, Any]]:
    denial = policy_denial(request)
    if not denial:
        return None
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
