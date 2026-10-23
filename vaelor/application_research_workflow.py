"""Durable phase and blocker vocabulary for guarded application research."""

from __future__ import annotations

import re
from collections.abc import Callable
from typing import Any
from urllib.parse import urlsplit

from .application_deployments import QUERY_STOPWORDS


RESEARCH_PHASES = frozenset({
    "interpreting",
    "searching",
    "acquiring",
    "synthesizing",
    "validating",
    "needs_input",
    "ready_for_review",
})

LEGACY_PHASES = {
    "queued": "interpreting",
    "needs-input": "needs_input",
    "ready-for-review": "ready_for_review",
    "failed": "needs_input",
    "cancelled": "needs_input",
    "completed": "ready_for_review",
}

RESEARCH_BLOCKERS = frozenset({
    "interpretation",
    "discovery",
    "acquisition",
    "synthesis",
    "compatibility",
    "policy",
    "authorization",
    "execution",
    "verification",
})

ResearchProgress = Callable[[str, int, str], None]

# Words after an application's name that make the request about another thing:
# an alternative to it, or a tool around it. Generic, never per application.
_OTHER_THING_WORDS = frozenset({
    "alternative", "alternatives", "instead", "like", "manager", "client",
    "clients", "plugin", "plugins", "request", "requests", "companion",
    "exporter", "wrapper", "api", "bot", "integration", "addon", "extension",
})


def request_names_application(query: Any, names: Any) -> str | None:
    """The one of ``names`` this request is FOR, or ``None`` (VD-222 live).

    The one rule for "is this request for that known application": the name
    must be the first whole words of the request, and no later word may make it
    a request for something else. The request's filler words ("install",
    "the", "for my") are dropped first - the same list that turns a message
    into its application_query - so raw text and the stored query agree. The reviewed
    source path and the capability verdict both use it, so they cannot
    disagree (LESSONS 6). A substring match made "Perplexica" and "Tautulli
    for Plex" count as Plex.
    """
    words = [word for word in re.findall(r"[a-z0-9]+", str(query or "").lower())
             if word not in QUERY_STOPWORDS]
    for name in names:
        parts = str(name).split()
        if words[:len(parts)] == parts and not set(words[len(parts):]) & _OTHER_THING_WORDS:
            return str(name)
    return None


def canonical_phase(value: Any, *, default: str = "interpreting") -> str:
    """Normalize persisted/adapter values to the public phase contract."""
    phase = LEGACY_PHASES.get(value, value)
    if phase not in RESEARCH_PHASES:
        if value in (None, ""):
            return default
        raise ValueError("Choose a supported application-research phase.")
    return phase


def validate_source_urls(value: Any) -> list[str]:
    """Accept only bounded public HTTPS URLs at the workflow boundary."""
    if not isinstance(value, list) or len(value) > 8:
        raise ValueError("Choose up to eight public application sources.")
    urls: list[str] = []
    for item in value:
        if not isinstance(item, str) or not item.strip() or len(item) > 2000:
            raise ValueError("Application source URLs must be bounded strings.")
        text = item.strip()
        try:
            parsed = urlsplit(text)
            hostname = parsed.hostname
            parsed.port
        except (TypeError, ValueError):
            raise ValueError("Application source URL is malformed.") from None
        if (
            parsed.scheme.lower() != "https" or not hostname
            or parsed.username is not None or parsed.password is not None
            or parsed.fragment or any(ord(char) < 32 for char in text)
        ):
            raise ValueError("Application source URLs must be public HTTPS pages.")
        if text not in urls:
            urls.append(text)
    return urls


class ResearchWorkflowError(ValueError):
    """A safe research failure attributed to one actionable product layer."""

    def __init__(
        self, message: str, blocker_layer: str, *, phase: str = "needs_input",
    ) -> None:
        if blocker_layer not in RESEARCH_BLOCKERS:
            raise ValueError("Choose a supported application-research blocker layer.")
        phase = canonical_phase(phase)
        super().__init__(message)
        self.blocker_layer = blocker_layer
        self.phase = phase


def emit(progress: ResearchProgress | None, phase: str, percent: int, message: str) -> None:
    """Emit a validated phase without coupling research to the queue backend."""
    phase = canonical_phase(phase)
    if progress is not None:
        progress(phase, percent, message)
