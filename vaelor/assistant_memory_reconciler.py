"""Down-cycle reconciler for live-source Assistant memories (VD-052, #97).

The owner's requirement: *"a build in agent to do a live sweep during down
cycles to check if the data is stale."* This walks the durable device/project
facts that carry a ``live_source``, reconciles each against its live source, and
writes a flag (``confirmed`` / ``stale`` / ``unknown``) that the read path then
consumes through ``assistant_memory_precedence``.

Three rules bind this module, each inherited from the rest of the project:

* **A failed or unreadable probe is ``unknown``, never a delete.** Gone and
  unreadable are different (LESSONS 8, VD-033): a sweep that cannot see a fact
  has not established the fact is wrong. A reconciler that deletes on a failed
  probe is a data-loss bug wearing a tidy-up. This module never deletes.
* **Down cycles, not idle-looking moments.** It must not compete with a loaded
  inference tier (VD-052). ``reconcile`` refuses and reports ``ran=False`` when
  the tier is loaded, rather than sweeping anyway.
* **The verdict must be read.** A measurement taken and never read is the
  VD-095 class (LESSONS 11). The flag is written to the store and consumed by
  ``context_for`` / ``grounded_memory_answer`` via the precedence resolver; the
  guard proves the read, not merely the write.

The live probe is injected as ``reader(tool, selector) -> ProbeResult`` so this
module depends on neither the tool registry nor ``assistant_acting_tools``.
"""

from __future__ import annotations

from typing import Any, Callable, Dict, List, NamedTuple, Optional

from .assistant_memory_precedence import (
    CONFIRMED,
    STALE,
    UNKNOWN,
    agrees,
    live_source_of,
)


class ProbeResult(NamedTuple):
    """What a live source said about a fact.

    ``status`` is the load-bearing field, and its three values are not
    interchangeable:

    * ``present`` — the tool read succeeded and returned ``value``.
    * ``absent`` — the tool read succeeded and the entity is genuinely gone.
      This is the only path to ``stale`` on a missing entity, and it exists so a
      genuinely-deleted app *does* get flagged (the positive control).
    * ``unreadable`` — the tool could not answer. Distinct from ``absent`` on
      purpose: this becomes ``unknown``, never ``stale``, because the observer
      produced the absence, not the world (LESSONS 8).
    """

    status: str
    value: Optional[str] = None


PRESENT = "present"
ABSENT = "absent"
UNREADABLE = "unreadable"


class ReconcileReport(NamedTuple):
    ran: bool
    reason: str
    flags: Dict[str, str]

    @property
    def stale(self) -> List[str]:
        return [mid for mid, flag in self.flags.items() if flag == STALE]

    @property
    def unknown(self) -> List[str]:
        return [mid for mid, flag in self.flags.items() if flag == UNKNOWN]


class MemoryReconciler:
    """Sweeps live-source facts on a down cycle and flags each one."""

    def __init__(
        self,
        store: Any,
        reader: Callable[[str, str], ProbeResult],
    ) -> None:
        self._store = store
        self._reader = reader

    def reconcile(self, *, tier_loaded: Any = False) -> ReconcileReport:
        """Walk every live-source fact and write its flag.

        ``tier_loaded`` may be a bool or a zero-argument callable. When it is
        true the sweep does not run: reconciliation is a down-cycle job and must
        not compete with a loaded inference tier.
        """
        loaded = tier_loaded() if callable(tier_loaded) else bool(tier_loaded)
        if loaded:
            return ReconcileReport(
                ran=False, reason="inference tier loaded; deferring to a down cycle", flags={}
            )
        flags: Dict[str, str] = {}
        for fact in self._store.live_source_facts():
            flag = self._classify(fact)
            self._store.set_reconcile_flag(fact["id"], flag)
            flags[fact["id"]] = flag
        return ReconcileReport(ran=True, reason="swept", flags=flags)

    def _classify(self, fact: Dict[str, Any]) -> str:
        """Reconcile one fact against its live source.

        The ``unknown`` branches are the whole point of the module: a raising
        reader, a missing binding, or an ``unreadable`` result never becomes
        ``stale``, so a sweep against a probe that could not run leaves the
        durable fact untouched and merely marked unknown.
        """
        source = live_source_of(fact)
        if source is None:
            return UNKNOWN
        try:
            probe = self._reader(source["tool"], source["selector"])
        except Exception:
            # A reader that raised established nothing. unknown, never a delete.
            return UNKNOWN
        if not isinstance(probe, ProbeResult) or probe.status == UNREADABLE:
            return UNKNOWN
        if probe.status == ABSENT:
            # The read succeeded and the entity is gone: the positive control.
            return STALE
        if probe.status == PRESENT:
            return CONFIRMED if agrees(str(fact.get("content", "")), str(probe.value or "")) else STALE
        return UNKNOWN
