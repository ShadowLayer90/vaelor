"""Converge the Phoenix trace collector to its persisted flag (VD-128, Phase E').

The executor is the only account that can drive the root hardware bridge, so it
is the one that brings the Phoenix container up or down to match the persisted
enable flag (:mod:`vaelor.phoenix_state`). This mixin holds that reconcile:

* the ``phoenix.apply`` job an admin toggle enqueues (:meth:`run_phoenix_apply`);
* the same convergence the boot/failure autostart re-runs
  (:func:`vaelor.executor_service.launch_phoenix_autostart`), so an enabled
  collector killed by a reboot or a crash comes back without a toggle.

It is the trace-collector sibling of the LLM Server's ``apply_llm_server``
(:mod:`vaelor.executor_gpu_deploy`), kept in its own module rather than added to
that capped one - and decoupled from GPU serving, because tracing does not depend
on whether a model is being served. :meth:`converged` gates the apply so a pass
that is already right does not needlessly replace the container (the flap the LLM
proxy's second live run found).
"""

from __future__ import annotations

from typing import Any, Dict, Optional

from .phoenix_service import PHOENIX_PORT, PhoenixController
from .phoenix_state import PhoenixSettings, PhoenixStore


class ExecutorPhoenixMixin:
    """The ``phoenix.apply`` action and its boot/failure convergence."""

    def _phoenix_store(self) -> PhoenixStore:
        """The persisted enable-flag store, a lazily-cached injectable seam.

        A test injects a store over a tmp file via ``_phoenix_store_cache``;
        production reads the record from the state root.
        """
        existing = getattr(self, "_phoenix_store_cache", None)
        if existing is not None:
            return existing
        store = PhoenixStore()
        self._phoenix_store_cache = store
        return store

    def _phoenix_controller(self) -> PhoenixController:
        """The Phoenix controller over the root bridge, a lazily-cached seam.

        Production wraps a :class:`vaelor.phoenix_service.PhoenixController` over
        the same root :class:`vaelor.hardware_bridge_client.HardwareBridgeClient`
        the model, NPU and proxy launches use, because the container must be
        launched by the root bridge, not the sandboxed executor. A test injects a
        recording fake via ``_phoenix_controller_cache``.
        """
        existing = getattr(self, "_phoenix_controller_cache", None)
        if existing is not None:
            return existing
        from .hardware_bridge_client import HardwareBridgeClient

        controller = PhoenixController(HardwareBridgeClient())
        self._phoenix_controller_cache = controller
        return controller

    def apply_phoenix(self) -> Dict[str, Any]:
        """Converge the Phoenix container to the persisted enable flag.

        Enabled and not already up -> start it; disabled and still up -> stop and
        remove it; already right -> nothing, so a periodic reconcile does not
        replace a healthy container every pass. Non-fatal in the sense the caller
        needs: a genuine bridge failure propagates so the job records it, and the
        boot autostart wraps the whole pass (:func:`launch_phoenix_autostart`) so
        it never breaks the boot path.
        """
        settings: PhoenixSettings = self._phoenix_store().read()
        controller = self._phoenix_controller()
        if controller.converged(settings):
            return {"applied": False, "converged": True, "enabled": settings.enabled}
        controller.apply(settings)
        return {
            "applied": True,
            "enabled": settings.enabled,
            "port": PHOENIX_PORT,
        }

    def run_phoenix_apply(self, job: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        """Run and finish a ``phoenix.apply`` job.

        Kept here rather than inline in ``JobExecutor.run_once`` so the dispatch
        there stays a one-line delegation and this module holds the whole action
        beside the code it drives.
        """
        return self.store.finish(
            job["id"], state="completed",
            message="Phoenix trace collector settings applied",
            result=self.apply_phoenix(),
        )
