"""Periodically self-heal joined workers' telemetry (the VD-100 inert class).

`cluster_manager.reconcile_worker_telemetry` carries the B self-heal - it brings
back a down agent, a stale controller CA, or an active-but-not-reporting worker -
but the only thing that calls it is the operator reconcile route. A worker whose
ingest key or CA drifted therefore stays broken until a human notices and clicks
Recheck: the same "a mechanism ships but nothing drives it" class the memory
reconciler already fixed with `assistant_reconciler_scheduler.MemoryReconcile
Scheduler` (VD-100/VD-101). This module is the missing cadence.

It holds no healing logic of its own and imports no cluster module. It is handed
two callables - one that lists the joined worker node ids to sweep, one that runs
the reconcile for a single node id - and it drives them on a deliberately slow
daemon loop. Two properties keep a periodic sweep safe:

* **Per-worker isolation.** Each node's reconcile runs inside its own guard, so a
  single unreachable or slow box (its SSH timeout bounds the call) is logged and
  stepped over rather than stalling the loop or the workers queued behind it.
* **Idempotent by the reconcile's own contract.** The reconcile no-ops a healthy
  agent and only reprovisions a genuinely stale one, behind its active-age
  loop-guard, so repeating the walk every interval cannot thrash a healthy
  worker or re-mint a key that is still good.
"""

from __future__ import annotations

import logging
import threading
from typing import Any, Callable, List, Optional

LOGGER = logging.getLogger(__name__)

#: The gap between sweeps, matching the memory reconciler's cadence. Deliberately
#: slow: telemetry healing is background housekeeping, not a request path, and a
#: worker that fell stale a minute ago is not worse for being swept a quarter
#: hour from now instead of instantly.
DEFAULT_INTERVAL_SECONDS = 900.0

#: How long after the control plane starts the FIRST sweep runs (ACC-194).
#: The loop used to wait a whole interval first ("nothing is meaningfully
#: stale a second after boot"), but a start is very often an upgrade, and an
#: upgrade is exactly when every worker is stale: the controller now ships a
#: newer agent and GPU sampler. The live test of build f768b46 saw a worker
#: with no sampler and an unlabelled GPU temperature until the owner pressed
#: Recheck. A minute keeps the sweep off the boot-time request path.
FIRST_SWEEP_DELAY_SECONDS = 60.0


class WorkerTelemetryReconcileScheduler:
    """Own the daemon thread that walks joined workers and drives the reconcile.

    Construction takes the two seams as callables rather than a cluster manager,
    so this module stays disjoint from `cluster_manager.py`: ``list_workers()``
    yields the joined worker node ids to visit, and ``reconcile(node_id)`` heals
    exactly one of them. Neither is evaluated at construction; both are read
    afresh on every cycle.
    """

    def __init__(
        self,
        list_workers: Callable[[], List[str]],
        reconcile: Callable[[str], Any],
        *,
        interval_seconds: float = DEFAULT_INTERVAL_SECONDS,
        first_sweep_seconds: float = FIRST_SWEEP_DELAY_SECONDS,
    ) -> None:
        self._list_workers = list_workers
        self._reconcile = reconcile
        self.interval_seconds = max(1.0, float(interval_seconds))
        self.first_sweep_seconds = min(self.interval_seconds, max(0.0, float(first_sweep_seconds)))
        self._stop = threading.Event()
        self._thread: Optional[threading.Thread] = None

    def run_once(self) -> None:
        """Sweep every joined worker once, each guarded on its own.

        Listing the workers is guarded because a store read can raise; then each
        node's reconcile is guarded independently so one failure never denies the
        remaining workers their turn. Nothing here inspects or logs a credential -
        only the node id, which is not a secret, ever reaches a log line.
        """
        try:
            workers = list(self._list_workers() or [])
        except Exception:
            LOGGER.warning(
                "worker-telemetry self-heal could not enumerate joined workers this cycle",
                exc_info=False,
            )
            return
        for node_id in workers:
            if self._stop.is_set():
                break
            try:
                self._reconcile(node_id)
            except Exception:
                LOGGER.debug(
                    "worker-telemetry self-heal passed over an unreachable worker this cycle",
                    exc_info=False,
                )
                continue

    def start(self) -> None:
        if self._thread and self._thread.is_alive():
            return
        self._stop.clear()
        self._thread = threading.Thread(
            target=self._loop, name="vaelor-worker-telemetry-reconciler", daemon=True
        )
        self._thread.start()

    def stop(self) -> None:
        self._stop.set()
        thread = self._thread
        if thread and thread.is_alive():
            thread.join(timeout=2.0)

    def _loop(self) -> None:
        # A short wait, then the first sweep (ACC-194: a start is often an
        # upgrade), then one sweep per interval.
        wait = self.first_sweep_seconds
        while not self._stop.wait(wait):
            wait = self.interval_seconds
            try:
                self.run_once()
            except Exception:
                # Housekeeping must never take the control plane down. A failed
                # sweep is simply retried on the next cycle.
                continue
