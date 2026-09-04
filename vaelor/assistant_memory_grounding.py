"""Deterministic recall for explicit questions about reviewed Assistant memory."""

from __future__ import annotations

import re
from typing import Any, Dict, Optional

from .assistant_memory_precedence import servable_as_settled
from .assistant_request_policy import policy_denial


STOP_WORDS = {
    "about", "answer", "appliance", "could", "exact", "from", "have",
    "is", "me", "please", "reply", "tell", "that", "the", "this", "what",
    "which", "with", "would", "your",
}
QUESTION_CUES = {
    "what", "which", "tell", "remember", "memory", "stored", "recall",
}
MEMORY_CUES = {"remember", "memory", "stored", "recall"}


def _tokens(value: str) -> set[str]:
    return {
        token
        for token in re.findall(r"[a-z0-9][a-z0-9_-]{1,}", value.lower())
        if token not in STOP_WORDS
    }


def grounded_memory_answer(
    message: str, context: Dict[str, Any]
) -> Optional[Dict[str, Any]]:
    """Return an exact stored fact when one clearly matches a recall question."""
    if policy_denial(message):
        return None
    question_tokens = _tokens(message)
    cue_tokens = set(re.findall(r"[a-z]+", message.lower()))
    # Keyword overlap retrieves context, but only an explicit recall request may
    # use the deterministic no-model fallback. Ordinary questions are answered
    # by live facts or by the selected model with memory supplied as context.
    if not question_tokens or not cue_tokens.intersection(MEMORY_CUES):
        return None
    candidates = []
    for memory in context.get("memories", []):
        if not isinstance(memory, dict):
            continue
        # A fact with a live source must be DERIVED, not recalled (VD-052 /
        # #211). This deterministic no-model path cannot derive a live value, so
        # its only honest move is to decline and leave the answer to the live
        # read — exactly as detect_product refuses the .variant guess while a HAT
        # EEPROM is present. Serving it verbatim would recall a fact the live
        # source may already contradict.
        if not servable_as_settled(memory):
            continue
        content = str(memory.get("content", "")).strip()
        overlap = question_tokens.intersection(_tokens(content))
        if content and overlap:
            candidates.append((len(overlap), content, memory))
    if not candidates:
        return None
    candidates.sort(key=lambda item: item[0], reverse=True)
    threshold = (
        1 if cue_tokens.intersection(MEMORY_CUES) else 2
    )
    if candidates[0][0] < threshold:
        return None
    best_score = candidates[0][0]
    selected = [item for item in candidates if item[0] == best_score][:3]
    facts = [item[1] for item in selected]
    label = "Stored memory" if len(facts) == 1 else "Stored memories"
    return {
        "answer": "{}: {}".format(label, " ".join(facts)),
        "evidence": [
            {
                "source": "assistant.memory",
                "summary": "Used {} reviewed {} memory.".format(
                    item[2].get("scope", "stored"),
                    item[2].get("type", "fact"),
                ),
            }
            for item in selected
        ],
        "suggested_actions": [],
        "proposed_job": None,
    }
