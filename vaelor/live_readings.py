"""One live hardware sample per answer.

An answer about live hardware is only as truthful as its worst reading. The
assistant gathers several fact tools per question, and two of them read the same
sensors: ``system.telemetry`` reads the appliance metrics, and ``cooling.status``
reads the fan controller, which takes its own temperature from sysfs. Each tool
sampled independently, milliseconds apart, so a single reply could - and on the
live appliance did - say "the CPU is 43.5°C" in one paragraph and "the host CPU
is currently 26.7% use and 44.6°C" two lines later. Both numbers were real. The
paragraph containing both was not.

The fix is not to round or to hide the difference. It is to take the reading
once. A reading session memoises the live callbacks for the span of one answer,
so every sentence in that answer is composed from the same sample, and the
next question takes a fresh one.

Scope is deliberately narrow. Only the callbacks named in ``COHERENT_READS`` are
memoised: those are the live sensor reads whose value moves between two calls a
few milliseconds apart. Inventory, jobs, and configuration reads are unaffected.
A session lives for the duration of one ``with`` block and is never shared
between requests: it is held in thread-local state, and the tool registry hands
it explicitly to the worker thread it runs handlers on, because a new thread
starts with an empty thread-local of its own.
"""

from __future__ import annotations

import threading
from contextlib import contextmanager
from typing import Any, Callable, Dict, Iterator, Optional

# Callbacks that read a moving sensor. Reading one of these twice inside one
# answer is what produced two temperatures in one paragraph.
COHERENT_READS = frozenset({"current_data", "cpu_fan_status"})


class ReadingSession:
    """Sample each live reading at most once, and reuse it."""

    def __init__(self) -> None:
        self._values: Dict[str, Any] = {}
        self._lock = threading.Lock()

    def read(self, name: str, produce: Callable[[], Any]) -> Any:
        """Return the sample for ``name``, taking it only if it is not held."""
        with self._lock:
            if name in self._values:
                return self._values[name]
        value = produce()
        with self._lock:
            # A concurrent caller may have stored one first. Keep the stored
            # sample rather than the newer one: within an answer, "first read
            # wins" is what makes every sentence agree.
            return self._values.setdefault(name, value)


class _Active(threading.local):
    session: Optional[ReadingSession] = None


_ACTIVE = _Active()


def active_session() -> Optional[ReadingSession]:
    """The session this thread is currently composing an answer inside."""
    return getattr(_ACTIVE, "session", None)


@contextmanager
def bound_session(session: Optional[ReadingSession]) -> Iterator[None]:
    """Run this block with ``session`` active on the calling thread.

    Used to carry a session across a thread boundary. Nesting restores whatever
    was active before, so a worker thread that is reused does not inherit a
    session from the request that finished on it.
    """
    previous = getattr(_ACTIVE, "session", None)
    _ACTIVE.session = session
    try:
        yield
    finally:
        _ACTIVE.session = previous


@contextmanager
def one_reading_per_answer() -> Iterator[ReadingSession]:
    """Compose everything in this block from a single hardware sample.

    Re-entering an active session joins it rather than starting a second one,
    so a nested helper cannot quietly reintroduce a second reading.
    """
    existing = active_session()
    if existing is not None:
        yield existing
        return
    session = ReadingSession()
    with bound_session(session):
        yield session


def coherent_reading(name: str, produce: Callable[[], Any]) -> Any:
    """Take a live reading, reusing this answer's sample when one exists."""
    session = active_session()
    if session is None or name not in COHERENT_READS:
        return produce()
    return session.read(name, produce)
