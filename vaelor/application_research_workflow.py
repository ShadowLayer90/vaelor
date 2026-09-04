"""Durable phase and blocker vocabulary for guarded application research."""

from __future__ import annotations

from collections.abc import Callable
from typing import Any
from urllib.parse import urlsplit


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
