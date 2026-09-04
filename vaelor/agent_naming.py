"""Name a drafted agent after what it does.

Taking the first few words of the request produced names like "Every Morning
Give Me Yesterday agent" - grammatically broken and carrying none of the
information that tells two agents apart in a list.
"""

from __future__ import annotations

import re

# Framing words. They describe how the user phrased the request, not what the
# agent is for, so they never help identify one agent among several.
NAME_STOPWORDS = frozenset({
    "a", "an", "the", "my", "me", "i", "it", "its", "they", "them", "their",
    "and", "or", "but", "if", "is", "are", "was", "were", "be", "been",
    "for", "to", "of", "from", "on", "in", "at", "by", "with", "about",
    "that", "this", "these", "those", "there", "then", "than", "so",
    "every", "each", "daily", "day", "days", "week", "weekly", "month",
    "morning", "evening", "afternoon", "night", "today", "yesterday",
    "tomorrow", "hour", "hourly", "minute", "always", "please",
    "give", "tell", "show", "get", "pull", "fetch", "make", "let", "send",
    "want", "need", "know", "check", "keep", "have", "has", "do", "does",
    "whether", "what", "when", "how", "who", "why", "which", "can", "will",
    "little", "big", "very", "too", "really", "just", "some", "any", "out",
    "up", "down", "over", "under", "again", "also", "old",
    # Domain noise: these ride along on a URL and say nothing about the agent.
    "com", "www", "http", "https", "net", "org", "io", "co",
})

# Rendered upper-case; capitalising them ("Mlb", "Espn") reads as a typo.
NAME_ACRONYMS = frozenset({
    "mlb", "nfl", "nba", "nhl", "mls", "ufc", "espn", "bbc", "cnn",
    "cpu", "gpu", "ram", "ssd", "hdd", "nvme", "usb", "led", "rgb",
    "api", "llm", "ai", "ml", "vpn", "dns", "dhcp", "ssh", "ssl", "tls",
    "rss", "http", "https", "url", "sql", "csv", "json", "pdf", "os", "ip",
})

MAX_NAME_WORDS = 4
MAX_NAME_CHARS = 72


def draft_agent_name(prompt: str) -> str:
    """Build a short, distinguishing display name from a free-text request."""
    words = re.findall(r"[A-Za-z0-9]+", str(prompt))
    chosen: list[str] = []
    seen: set[str] = set()
    for word in words:
        key = word.lower()
        if len(word) < 2 or key in NAME_STOPWORDS or key in seen:
            continue
        seen.add(key)
        chosen.append(word)
        if len(chosen) == MAX_NAME_WORDS:
            break
    if not chosen:
        # Nothing distinctive survived; fall back to the raw opening words
        # rather than naming every such agent "Custom agent".
        chosen = words[:3] or ["Custom"]
    rendered = [
        word.upper() if word.lower() in NAME_ACRONYMS else word.capitalize()
        for word in chosen
    ]
    return " ".join(rendered)[:MAX_NAME_CHARS] + " agent"
