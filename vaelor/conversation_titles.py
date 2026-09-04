"""Readable conversation titles derived from a first message.

Slicing a raw prompt at a fixed width produced titles that stopped mid-word
("...using light public research a") and made unrelated conversations
indistinguishable when they shared an opening phrase. This module trims on a
word boundary, marks visual truncation with a real ellipsis, and keeps enough
of the message for similar conversations to stay tellable apart.
"""

from __future__ import annotations

import re

TITLE_LIMIT = 80
# Below this, cutting on a word boundary loses too much to stay useful.
MIN_WORD_BOUNDARY = 24
# Words taken from the end of a truncated message to keep similar
# conversations tellable apart in the History list.
TAIL_WORDS = 3
MAX_TAIL_CHARS = 28
DEFAULT_TITLE = "New assistant conversation"

_WHITESPACE = re.compile(r"\s+")


def conversation_title(message: str, *, limit: int = TITLE_LIMIT, fallback: str = DEFAULT_TITLE) -> str:
    """Return a single-line, cleanly terminated title for ``message``.

    Long messages that share an opening phrase must not collapse to identical
    titles: three saved conversations all reading "Ask my stock agent to
    compare Microsoft and Nvidia usin..." are indistinguishable in the History
    list. When the head is truncated, a few distinguishing words from the tail
    are appended.
    """
    collapsed = _WHITESPACE.sub(" ", str(message or "")).strip()
    if not collapsed:
        return fallback
    if len(collapsed) <= limit:
        return collapsed

    # Reserve room for the ellipsis and a distinguishing tail.
    tail_words = collapsed.split()[-TAIL_WORDS:]
    tail = " ".join(tail_words).rstrip(" ,;:.!?-—–")
    # Trim a long tail rather than dropping it: discarding it entirely put
    # messages with long final words straight back to identical titles.
    while tail and len(tail) > MAX_TAIL_CHARS:
        if " " in tail:
            tail = tail.split(" ", 1)[1]
        else:
            tail = tail[-MAX_TAIL_CHARS:]
    tail_budget = len(tail) + 2 if tail else 0

    window = collapsed[: max(MIN_WORD_BOUNDARY, limit - 1 - tail_budget)]
    boundary = window.rfind(" ")
    if boundary >= MIN_WORD_BOUNDARY:
        window = window[:boundary]
    # Trailing punctuation before an ellipsis reads as a typo.
    window = window.rstrip(" ,;:.!?-—–")
    if not window:
        return fallback
    if tail_budget and tail and not collapsed.startswith(window + " " + tail):
        return "{}… {}".format(window, tail)
    return window + "…"
