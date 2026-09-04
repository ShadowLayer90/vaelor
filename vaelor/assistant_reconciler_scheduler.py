"""Run the memory reconciler on down cycles (VD-101 / #97, LESSONS 8, 11).

`assistant_memory_reconciler.MemoryReconciler` was built and committed but
nothing scheduled it, so no live-source fact was ever swept in a running
appliance - the VD-100/VD-101 class of a mechanism that ships inert. This module
is the schedule.

Two rules, both inherited and both guarded:

* **Down cycles only.** The sweep must never compete with a loaded inference
  tier (VD-052). The scheduler passes a fresh ``tier_loaded`` reading into every
  ``reconcile`` call, and the reconciler itself refuses and reports ``ran=False``
  when the tier is loaded - so the deferral is re-decided each cycle, not once at
  construction.
* **Unknown is not absent; never delete.** This module adds no delete path. It
  owns the *when*; the reconciler owns the *what*, and the reconciler's contract
  (an unreadable or raising probe -> ``unknown``, never ``stale``) is unchanged.

The live probe (`registry_probe_reader`) is bound to the read tool registry: a
``device_fact`` whose ``live_source`` names e.g. ``inference.status`` /
``workloads.inventory`` is answered by running that read tool this cycle. A tool
that raises or reports itself unavailable becomes ``unreadable`` here, which the
reconciler turns into ``unknown`` - the observer produced the absence, not the
world (LESSONS 8).
"""

from __future__ import annotations

import threading
from typing import Any, Callable, Mapping, Optional

from .assistant_memory_reconciler import (
    ABSENT,
    PRESENT,
    UNREADABLE,
    MemoryReconciler,
    ProbeResult,
    ReconcileReport,
)

#: The default gap between down-cycle sweeps. Deliberately slow: reconciliation
#: is housekeeping, not a request path, and a fact that changed a minute ago is
#: still derived live by the read path regardless of the flag.
DEFAULT_INTERVAL_SECONDS = 900.0


def any_tier_loaded(inference_status: Any) -> bool:
    """True when any inference engine is holding a model (the loaded-tier signal).

    Reuses `inference_status`'s own per-engine truth rather than inventing a
    second definition of "loaded" (LESSONS 6): an engine counts as loaded when
    it reports a non-empty ``loaded_models`` list, or a known resident size.
    ``loaded_models is None`` means "not checked", which is not "loaded".
    """
    engines = inference_status.get("engines") if isinstance(inference_status, Mapping) else None
    for engine in engines or []:
        if not isinstance(engine, Mapping):
            continue
        loaded = engine.get("loaded_models")
        if isinstance(loaded, list) and loaded:
            return True
        if engine.get("resident_known") is True:
            return True
    return False


def _navigate(result: Any, selector: str) -> ProbeResult:
    """Resolve a dotted ``selector`` against a successful tool ``result``.

    The three ProbeResult statuses are not interchangeable (LESSONS 8):

    * A tool result that flags itself ``unavailable`` -> ``unreadable``. The tool
      could not answer, so this must never become ``stale``.
    * A dotted path that walks off a present mapping (the leaf key is missing on
      a container that IS there) -> ``absent``. The read succeeded and the entity
      is genuinely gone: the positive control that lets a deleted app flag stale.
    * A structural surprise - a segment that cannot be descended because the
      current node is not a mapping -> ``unreadable``, not ``absent``. We cannot
      confidently say the entity is gone, so we refuse to risk a false ``stale``.
    """
    if isinstance(result, Mapping) and result.get("unavailable"):
        return ProbeResult(status=UNREADABLE)
    current: Any = result
    for segment in [part for part in str(selector).split(".") if part]:
        if not isinstance(current, Mapping):
            # Cannot descend: the shape is not what the selector expects, so the
            # honest answer is "could not read", never "gone".
            return ProbeResult(status=UNREADABLE)
        if segment not in current:
            # Present container, missing leaf: the entity the fact named is gone.
            return ProbeResult(status=ABSENT)
        current = current[segment]
    if current is None:
        return ProbeResult(status=ABSENT)
    return ProbeResult(status=PRESENT, value=str(current))


def registry_probe_reader(registry: Any) -> Callable[[str, str], ProbeResult]:
    """A ``reader(tool, selector) -> ProbeResult`` bound to the read registry.

    The reader runs the named read tool this cycle and resolves the selector.
    Any failure to run the tool is ``unreadable`` - the reconciler turns that
    into ``unknown``, never a delete (LESSONS 8). This closure holds only the
    registry; it reaches no store and nothing that mutates.
    """

    def reader(tool: str, selector: str) -> ProbeResult:
        try:
            outcome = registry.run(str(tool), {})
        except Exception:
            # The tool could not answer. unreadable, so the reconciler leaves the
            # durable fact untouched and merely marks it unknown.
            return ProbeResult(status=UNREADABLE)
        result = outcome.get("result") if isinstance(outcome, Mapping) else outcome
        return _navigate(result, selector)

    return reader


class MemoryReconcileScheduler:
    """Own the down-cycle thread that drives one `MemoryReconciler`.

    It holds no reconciliation logic of its own: every sweep is
    ``reconciler.reconcile(tier_loaded=<fresh reading>)``, so the "defer while a
    tier is loaded" rule lives in exactly one place and is re-evaluated on every
    cycle rather than latched at start.
    """

    def __init__(
        self,
        reconciler: MemoryReconciler,
        tier_loaded: Callable[[], Any],
        *,
        interval_seconds: float = DEFAULT_INTERVAL_SECONDS,
    ) -> None:
        self._reconciler = reconciler
        self._tier_loaded = tier_loaded
        self.interval_seconds = max(1.0, float(interval_seconds))
        self._stop = threading.Event()
        self._thread: Optional[threading.Thread] = None

    def run_once(self) -> ReconcileReport:
        """Sweep now if this is a down cycle; defer otherwise.

        The ``tier_loaded`` callable is read *inside* ``reconcile`` (it is passed
        through, not evaluated here), so a tier that loaded a moment ago still
        defers this cycle. A raising signal is treated as "loaded" - a sweep that
        cannot prove the machine is idle must not compete for it.
        """

        def loaded() -> bool:
            try:
                return bool(self._tier_loaded())
            except Exception:
                return True

        return self._reconciler.reconcile(tier_loaded=loaded)

    def start(self) -> None:
        if self._thread and self._thread.is_alive():
            return
        self._stop.clear()
        self._thread = threading.Thread(
            target=self._loop, name="vaelor-memory-reconciler", daemon=True
        )
        self._thread.start()

    def stop(self) -> None:
        self._stop.set()
        thread = self._thread
        if thread and thread.is_alive():
            thread.join(timeout=2.0)

    def _loop(self) -> None:
        # Wait first, then sweep: nothing useful is stale a second after boot,
        # and the delay keeps the reconciler off the boot-time request path.
        while not self._stop.wait(self.interval_seconds):
            try:
                self.run_once()
            except Exception:
                # Housekeeping must never take the control plane down. A failed
                # sweep is retried next cycle; it deletes nothing on the way out.
                continue
