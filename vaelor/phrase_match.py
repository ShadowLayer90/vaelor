"""Match whole words, not bare substrings.

Substring matching is why "Give me yesterday's MLB score ... tell me whether
they won" was answered with a greeting: "t-hey- won" contains "hey". Short
tokens like "hi", "hey", "ram", "app" and "api" hide inside ordinary English
words, so any keyword table built on `phrase in text` misfires on real
sentences.
"""

from __future__ import annotations

import re
from functools import lru_cache
from typing import Iterable, Tuple


@lru_cache(maxsize=512)
def _pattern(phrases: Tuple[str, ...]) -> re.Pattern[str]:
    alternatives = "|".join(
        r"\s+".join(re.escape(word) for word in phrase.split())
        for phrase in sorted(phrases, key=len, reverse=True)
        if phrase.strip()
    )
    return re.compile(r"\b(?:" + alternatives + r")\b", re.IGNORECASE)


def mentions(text: str, phrases: Iterable[str]) -> bool:
    """True when `text` contains any phrase as whole words."""
    candidates = tuple(str(phrase).strip() for phrase in phrases if str(phrase).strip())
    if not candidates:
        return False
    return bool(_pattern(candidates).search(str(text)))
