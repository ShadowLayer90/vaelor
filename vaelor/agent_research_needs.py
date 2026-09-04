"""Decide whether a described agent job needs information from outside.

Whether a drafted agent could reach the web used to depend on which words the
user happened to type. "Go on the internet and find me the latest MLB scores"
turned public research on; "Read the Raspberry Pi news page and tell me the
newest product announced" did not, and the user saved an agent that could never
do the one thing they described. Nothing in the second sentence is ambiguous
about needing the outside world - it just did not contain "internet", "web",
"research" or a sports word.

The heuristic here answers one question: does the described job require a fact
this appliance does not hold? Three families of evidence say yes.

1. The user named the web itself - a URL, a bare domain, "online", "the news
   page", "search the internet".
2. The user named a kind of source that only exists outside - news, articles,
   blogs, forums, release notes, advisories, documentation pages.
3. The user asked for the current state of something the appliance cannot
   observe - prices, weather, scores, fixtures, elections, or the newest
   release, product or announcement from someone else.

It is deliberately *not* triggered by recency alone. "The latest job", "the
newest version of my container" and "what is running right now" are questions
about this box, and turning on public research for them would widen an agent's
reach on the strength of the word "latest". Recency counts only when it is
attached to a noun that lives outside this appliance.
"""

from __future__ import annotations

import re
from typing import Tuple

# The web, named directly.
_DIRECT: Tuple[str, ...] = (
    r"https?://",
    r"\bwww\.",
    r"\b[a-z0-9][a-z0-9-]*\.(?:com|org|net|io|dev|ai|co|uk|edu|gov|news|info|me)\b",
    r"\b(?:internet|online|web|webpage|website|web\s?site|web\s+page|url|urls)\b",
    r"\b(?:browse|browsing)\b",
    r"\b(?:search|look)\s+(?:it\s+|them\s+)?(?:up\s+)?(?:on\s+|the\s+|through\s+)?"
    r"(?:web|internet|online|google)\b",
    r"\bgoogle\b",
    r"\b(?:news|web|product|release|download|home|landing|support|status|docs?|"
    r"documentation|blog|pricing|company|vendor|store|shop)\s+(?:page|pages|site|sites)\b",
    r"\bpublic\s+(?:research|sources?|internet|web)\b",
)

# Kinds of source that only exist outside this appliance.
_EXTERNAL_SOURCE: Tuple[str, ...] = (
    r"\bnews\b", r"\bnewsletter\b", r"\bheadlines?\b", r"\barticles?\b",
    r"\bblogs?\b", r"\bforums?\b", r"\breddit\b", r"\bwikipedia\b",
    r"\bpress\s+releases?\b", r"\brelease\s+notes\b", r"\bchangelogs?\b",
    r"\bannouncement", r"\bannounce[ds]?\b", r"\badvisor(?:y|ies)\b",
    r"\bcve\b", r"\bresearch\b", r"\bpublications?\b", r"\bpapers?\b",
    r"\bupstream\b", r"\bmanufacturer\b", r"\bofficial\s+source\b",
)

# Facts about the outside world, whatever the phrasing.
_OUTSIDE_FACT: Tuple[str, ...] = (
    r"\bweather\b", r"\bforecast\b", r"\bstocks?\b", r"\bshare\s+price\b",
    r"\bmarkets?\b", r"\bcrypto\b", r"\bbitcoin\b", r"\bexchange\s+rate\b",
    r"\bcurrenc(?:y|ies)\b", r"\bsports?\b", r"\bscores?\b", r"\bfixtures?\b",
    r"\bstandings?\b", r"\bleagues?\b", r"\bplayoffs?\b", r"\btournaments?\b",
    r"\bodds\b", r"\binjur", r"\bmlb\b", r"\bnba\b", r"\bnfl\b", r"\bnhl\b",
    r"\bpremier\s+league\b", r"\bformula\s*1\b", r"\belections?\b",
    r"\bflights?\b", r"\btimetables?\b",
)

# Recency, but only where it is attached to something outside this appliance.
_OUTSIDE_RECENCY = (
    r"\b(?:latest|newest|most\s+recent|recent|current|today'?s?|this\s+week'?s?|"
    r"up[-\s]?to[-\s]?date)\b[^.!?]{0,60}?\b(?:news|headlines?|articles?|"
    r"prices?|pricing|releases?|products?|announcements?|scores?|results?|"
    r"advisor(?:y|ies)|papers?|models?\s+released|rumou?rs?)\b"
)

_PATTERNS = tuple(
    re.compile(pattern, re.IGNORECASE)
    for pattern in (*_DIRECT, *_EXTERNAL_SOURCE, *_OUTSIDE_FACT, _OUTSIDE_RECENCY)
)


def needs_public_research(description: str) -> bool:
    """Whether this described job plainly requires information from outside."""
    text = " ".join(str(description or "").split())
    if not text:
        return False
    return any(pattern.search(text) for pattern in _PATTERNS)
