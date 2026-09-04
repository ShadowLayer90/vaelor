"""One-at-a-time gate for calls to a managed local model (#223 / VD-085).

On a Raspberry Pi the appliance runs a single local model with a small number
of inference slots, and the Assistant and AI Chat share it (VD-085). Every model
call is a synchronous, several-second ``urlopen`` that holds a control-plane
worker for its whole duration, and nothing bounded how many ran at once.

**Measured live on myhost, 2026-08-17.** The first sequential AI Chat requests
answered in 0.7-4.7 s. After requests began to overlap - the tester's own load,
and normal use where the Assistant and AI Chat hit the one model together -
``/api/v2/ai-chat/messages`` started returning 503 and resetting the connection
for *every* caller; the browser showed "The local control service is
unavailable." Yet the model was healthy (a direct ``curl`` to it answered in
1.5 s) and the control plane logged nothing. The worker pool had saturated on
generations that can only run one at a time, and a bare connection reset with no
log is the worst possible way to say "busy".

This gate admits at most :data:`SLOTS` concurrent generations **per model
endpoint**. A caller that cannot get a slot is refused **fast** - it never blocks
a worker waiting on someone else's generation - with a truthful message the
caller turns into a real HTTP 503 body, so the reader learns the model is busy
instead of that the node is unreachable. Remote providers are not gated; they
run their own concurrency and a loopback address is how a managed-local model is
told apart (see :func:`provider_runtime.managed_local_connection`).

**Per endpoint, not one global slot (a63/a64+ multi-accelerator hosts).** VD-085
assumed one shared local model, so the original gate was a single process-global
semaphore. That is wrong on a box that runs two *independent* local models on two
accelerators - the Assistant on the NPU (``http://127.0.0.1:8081``) and AI Chat
on the GPU (``http://127.0.0.1:8088``). A single global permit serialised two
models that share no hardware, and - the failure that motivated this change -
one leaked permit on that shared slot locked *both* models out until a restart:
observed live on the Z2 as every model-routed ``/assistant/chat`` returning
``LocalModelBusy`` while a thread dump showed nothing generating. The slot is now
keyed by the endpoint's identity (scheme + host + port), so two loopback ports
run concurrently and a stall on one model can never starve the other. The *same*
endpoint is still serialised one-at-a-time (or to :data:`SLOTS`), which is the
Pi's single-model behaviour unchanged - one endpoint, one semaphore, same count.

The slot count defaults to 1 (the Pi's single model) and is set from
``VAELOR_LOCAL_INFERENCE_SLOTS`` so a host that genuinely serves N in parallel
can say so. It must match the model server's own parallelism: promising two
slots to a ``--parallel 1`` server just moves the collision downstream.

**Leak instrumentation.** Each held permit records the acquiring thread's name
and a monotonic acquire time. When an acquire is *refused*, the endpoint and
every current holder (thread name + how long it has held the permit) are logged
at WARNING, so the next occurrence in ``journalctl`` names the leaker instead of
leaving a permit stuck with nothing running. No stale permit is force-released
here: a timeout-based auto-release risks a double ``release()`` and could mask
the real leak, so this change surfaces the holder rather than papering over it.
"""

from __future__ import annotations

import logging
import os
import threading
import time
from contextlib import contextmanager
from typing import Dict, Iterator, List, Optional
from urllib.parse import urlsplit

from .provider_runtime import managed_local_connection

_LOGGER = logging.getLogger(__name__)

#: The message a refused caller shows. One constant because the route and the
#: test both need the exact words, and a message that drifts from its test is
#: how "busy" quietly becomes "unavailable" again (#223 was that exact drift,
#: one layer up).
BUSY_MESSAGE = (
    "The local AI model is busy answering another request. It runs one at a "
    "time on this appliance, so it could not take this one; wait a moment and "
    "try again."
)


class LocalModelBusy(RuntimeError):
    """A local model endpoint is already generating for another caller.

    A distinct type, not a bare ``RuntimeError``, so the one call site that
    turns it into a truthful 503 catches exactly this and nothing else - a
    generic except would also swallow the model's real failures, which is the
    ``#223`` shape (a busy model reported as an unreachable node) in reverse.
    """


def _configured_slots() -> int:
    raw = (
        os.environ.get("VAELOR_LOCAL_INFERENCE_SLOTS")
        or os.environ.get("PM_LOCAL_INFERENCE_SLOTS")
    )
    try:
        return max(1, int(raw))
    except (TypeError, ValueError):
        return 1


class _Hold:
    """One held permit, for the refusal log. Identity-compared (no ``__eq__``)
    so the exact record a caller appended is the one it removes on release,
    even when two holders share a thread name and an acquire instant."""

    __slots__ = ("thread_name", "acquired_at")

    def __init__(self, thread_name: str, acquired_at: float) -> None:
        self.thread_name = thread_name
        self.acquired_at = acquired_at


_LOCK = threading.Lock()
#: Permit count per endpoint. Read under ``_LOCK`` when building a semaphore so a
#: racing ``configure()`` cannot size a gate from a half-written value.
_SLOTS: int = _configured_slots()
#: endpoint identity -> its semaphore, created lazily on first use.
_GATES: Dict[str, threading.BoundedSemaphore] = {}
#: endpoint identity -> the permits currently held there, for the refusal log.
_HOLDERS: Dict[str, List[_Hold]] = {}


def _endpoint_key(connection: Dict[str, str]) -> str:
    """The connection's endpoint identity: scheme + host + port, no path.

    ``http://127.0.0.1:8081/v1`` and ``http://127.0.0.1:8081/v1/`` are the one
    model server, so the path and any trailing slash are dropped; only the
    ``netloc`` (host and port) distinguishes the NPU's 8081 from the GPU's 8088.
    """
    parts = urlsplit((connection or {}).get("base_url", ""))
    return "{}://{}".format(parts.scheme, parts.netloc)


def _gate_for(key: str) -> threading.BoundedSemaphore:
    with _LOCK:
        gate = _GATES.get(key)
        if gate is None:
            gate = threading.BoundedSemaphore(_SLOTS)
            _GATES[key] = gate
        return gate


def configure(slots: Optional[int] = None) -> None:
    """Set the per-endpoint permit count and reset the gate table.

    Called at startup once the real slot count is known, and by tests to size
    the gates deterministically. Clearing ``_GATES`` drops every existing
    semaphore, which is only safe before the gates are in use - the app calls
    this once, before serving. A generation already in flight keeps the old
    semaphore object it captured and releases *that* one, so a ``configure()``
    racing an acquire cannot raise or double-count; the reset only governs
    permits handed out afterward.
    """
    global _SLOTS
    with _LOCK:
        _SLOTS = slots if slots and slots > 0 else _configured_slots()
        _GATES.clear()
        _HOLDERS.clear()


def _record_hold(key: str, hold: _Hold) -> None:
    with _LOCK:
        _HOLDERS.setdefault(key, []).append(hold)


def _drop_hold(key: str, hold: _Hold) -> None:
    with _LOCK:
        holders = _HOLDERS.get(key)
        if not holders:
            return
        for index, held in enumerate(holders):
            if held is hold:  # identity, not equality: remove *this* permit
                del holders[index]
                break
        if not holders:
            _HOLDERS.pop(key, None)


def _log_refusal(key: str) -> None:
    """Name the endpoint and every current holder so a leaked permit is
    identifiable in ``journalctl`` on the next refusal, not just felt."""
    now = time.monotonic()
    with _LOCK:
        holders = list(_HOLDERS.get(key, ()))
    if holders:
        detail = "; ".join(
            "{} held {:.1f}s".format(hold.thread_name, now - hold.acquired_at)
            for hold in holders
        )
    else:
        detail = (
            "no recorded holder - a permit is held outside "
            "local_inference_slot() or leaked"
        )
    _LOGGER.warning(
        "local inference slot refused for %s; current holders: %s", key, detail
    )


@contextmanager
def local_inference_slot(
    connection: Dict[str, str], *, wait_seconds: float = 0.0
) -> Iterator[None]:
    """Hold this endpoint's inference slot for the body, or refuse fast.

    Non-managed-local (remote) connections are a no-op: they are not the shared
    slots this protects. The slot is keyed by the connection's endpoint, so two
    independent local models never contend and one busy model never refuses the
    other. ``wait_seconds`` is 0 by default - a busy local model stays busy for
    the seconds a generation takes, so waiting on a worker helps nobody and
    holding one is the harm this exists to stop. The slot is always released,
    including when the body raises, so one failed generation cannot strand it
    and lock that endpoint out until a restart.
    """
    if not managed_local_connection(connection or {}):
        yield
        return
    key = _endpoint_key(connection)
    gate = _gate_for(key)
    if wait_seconds and wait_seconds > 0:
        acquired = gate.acquire(timeout=wait_seconds)
    else:
        acquired = gate.acquire(blocking=False)
    if not acquired:
        _log_refusal(key)
        raise LocalModelBusy(BUSY_MESSAGE)
    # Nothing between a successful acquire and the `try` may raise, or the permit
    # is stranded - the exact leak this module exists to prevent. So the holder
    # bookkeeping (which takes `_LOCK`) happens INSIDE the try; `hold` starts None
    # and is only dropped if it was recorded.
    hold: Optional[_Hold] = None
    try:
        hold = _Hold(threading.current_thread().name, time.monotonic())
        _record_hold(key, hold)
        yield
    finally:
        # Release the semaphore we captured, not whatever `_GATES[key]` points
        # at now: a `configure()` between acquire and release swaps the object,
        # and releasing the new one would raise or over-credit it.
        gate.release()
        if hold is not None:
            _drop_hold(key, hold)
