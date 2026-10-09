"""Converge the LLM Server auth PROXY to its persisted state (F3b-ii).

Split out of :mod:`vaelor.executor_gpu_deploy`, which sat at the 1,000-line
ceiling `CLAUDE.md` sets: the LLM-Server proxy-convergence surface is a cohesive
concern of its own - the ``{enabled}`` state store, the broker-held key set, the
applied-binding marker, and the apply/reconcile that brings the nginx auth proxy
up or down in front of the loopback model - so it lives here as a mixin the
executor composes beside the GPU deploy, reading the executor's own helpers
(``credential_broker``, ``models_root``, ``store``, ``_cluster_mode_state``,
``_gpu_chat_compose_project_for``) exactly as the sibling mixins do.

**The LLM Server key reaches the proxy BY FILE, never on the argv.** The keyed
gate is an nginx container the root hardware bridge launches, and the key(s) are
handed to it in a root-owned ``0600`` config file mounted read-only - never as a
``docker run`` argument, so ``docker inspect`` / ``/proc`` on the box cannot read
it (:mod:`vaelor.llm_server_proxy` builds and validates that seam). This module
only ever passes the key SET to :meth:`LlmServerProxyController.apply`; the file
discipline is the controller's.

**Two key reads, deliberately different.** The convergence POLL
(:meth:`_llm_binding_converged`, ~30 s) reads only per-key FINGERPRINTS from the
broker's ``list()`` - it decrypts nothing and writes no ``credential.endpoint-keys``
audit row - because it needs a drift signal, not the keys. A real (re)launch
(:meth:`apply_llm_server` / :meth:`_converge_llm_proxy`) reads the PLAINTEXT keys
through :func:`~vaelor.llm_server_state.active_keys` to render the gate, so the
plaintext-read audit now fires on an actual apply, not on every poll.
"""

from __future__ import annotations

import logging
import time
from typing import Any, Dict

from .credential_broker import CredentialError
from .gpu_cluster_mode import HEALTHY_STATE, UNLOADED_STATE
from .gpu_loading_door import LoadingDoor, close_door
from .gpu_serving_target import (
    KIND_CLUSTER, gpu_cluster_mode_active, resolve_gpu_serving_target,
)
from .llm_server_proxy import LLM_SERVER_PROXY_PORT, LlmServerProxyController
from .llm_server_state import (
    APPLIED_BINDING_FILENAME,
    AppliedBindingStore,
    LlmServerSettings,
    LlmServerStore,
    active_key_fingerprints,
    active_keys,
    applied_marker,
    binding_marker,
    migrate_legacy_key,
)

LOGGER = logging.getLogger(__name__)

#: What `apply_llm_server` answers when the Mode B deployment it would front is
#: not serving - unloaded (G3a/G3b) or mid-load. This apply opens no door onto
#: the model's port, which answers nothing; the mode reconcile owns the door
#: then (the unloaded door or the wake door, VD-159) and points it back at the
#: model from the CURRENT flag on the first healthy pass after the load.
CLUSTER_NOT_SERVING = "cluster-deployment-not-serving"

#: What `apply_llm_server` answers when the deployment is unloaded and the
#: door for that state could not be brought to the flag and keys (the key
#: store did not answer, or the bridge refused): nothing was applied.
UNLOADED_DOOR_NOT_APPLIED = "unloaded-door-not-applied"

#: The job's closing sentence, by whether the port was actually brought to
#: the settings. "Applied" is said only when it was.
APPLIED_MESSAGE = "LLM Server settings applied"
NOT_APPLIED_MESSAGE = (
    "LLM Server settings saved. Nothing was changed on its port this time; "
    "Vaelor brings the port to these settings on its next check."
)


class ExecutorLlmServerMixin:
    """Bring the LLM Server auth proxy to the persisted state, and keep it there."""

    def _llm_server_store(self) -> LlmServerStore:
        """The LLM Server state store (the ``{enabled}`` flag; keys live in the
        broker), a lazily-cached injectable seam a test sets over a tmp file."""
        existing = getattr(self, "_llm_server_store_cache", None)
        if existing is not None:
            return existing
        store = LlmServerStore()
        self._llm_server_store_cache = store
        return store

    def _llm_server_settings(self) -> LlmServerSettings:
        """The current LLM Server ``enabled`` flag, failing safe to disabled; the
        active key SET comes from the broker (:meth:`_llm_server_active_keys`)."""
        return self._llm_server_store().read()

    def _applied_binding_store(self) -> AppliedBindingStore:
        """The applied-binding marker store (FINDING A), a lazily-cached injectable
        seam: the bind host + key-set fingerprint the proxy was last converged to,
        in the executor-owned models root; inject ``_applied_binding_store_cache``."""
        existing = getattr(self, "_applied_binding_store_cache", None)
        if existing is not None:
            return existing
        store = AppliedBindingStore(str(self.models_root / APPLIED_BINDING_FILENAME))
        self._applied_binding_store_cache = store
        return store

    def _llm_proxy(self) -> LlmServerProxyController:
        """The LLM Server auth-proxy controller, a lazily-cached injectable seam.

        Production wraps a :class:`vaelor.llm_server_proxy.LlmServerProxyController`
        over the same root :class:`vaelor.hardware_bridge.HardwareBridgeClient` the
        model and NPU launches use, because the nginx proxy - a host-network
        container binding the LAN and mounting a root-owned config - must be
        launched by the root bridge, not the sandboxed executor. The controller
        hands the key SET to that container BY FILE (root ``0600``, read-only),
        never on the argv. A test injects a recording fake via ``_llm_proxy_cache``.
        """
        existing = getattr(self, "_llm_proxy_cache", None)
        if existing is not None:
            return existing
        from .hardware_bridge import HardwareBridgeClient

        controller = LlmServerProxyController(HardwareBridgeClient())
        self._llm_proxy_cache = controller
        return controller

    def _llm_server_active_keys(self):
        """The LLM Server's active broker key set (PLAINTEXT), or ``None`` if down.

        The RENDER path: reached only when actually (re)launching the gate, so the
        plaintext decrypt and its ``credential.endpoint-keys`` audit fire on a real
        apply, not on a poll. Migrates a pre-F3b-ii ``state.json`` key into the
        broker first (B5, idempotent), then reads the active set. BROKER-DOWN
        FAIL-SAFE: a broker that cannot answer returns ``None`` (callers leave the
        running gate as it is).
        """
        try:
            migrate_legacy_key(self._llm_server_store(), self.credential_broker)
            return active_keys(self.credential_broker)
        except CredentialError:
            return None

    def _llm_server_key_fingerprints(self):
        """The LLM Server's active per-key FINGERPRINTS (the poll's drift signal),
        or ``None`` if the broker is down.

        The POLL path (:meth:`_llm_binding_converged`): fingerprint-only, so the
        ~30 s reconcile decrypts NO key and writes NO ``credential.endpoint-keys``
        audit row - it reads the broker's fingerprint-only ``list()``. Migrates a
        pre-F3b-ii key first (idempotent; a cheap no-op once migrated), so an
        upgraded, enabled box still converges its gate. BROKER-DOWN FAIL-SAFE:
        ``None`` (the caller reads that as converged and leaves the gate as is).
        """
        try:
            migrate_legacy_key(self._llm_server_store(), self.credential_broker)
            return active_key_fingerprints(self.credential_broker)
        except CredentialError:
            return None

    def _llm_binding_converged(self, model_port: int) -> bool:
        """Whether the auth proxy already matches the persisted LLM Server state.

        The applied marker equals the desired binding AND the controller's live
        reading says the proxy is fronting ``model_port`` (the one definition the
        Mode B reconcile also uses); either failing is drift the reconcile repairs.
        The desired marker is built from per-key FINGERPRINTS (no plaintext decrypt
        on the poll), the SAME basis the applied marker is written with. A broker
        that cannot answer reads as CONVERGED so the running gate is left untouched
        (the broker-down fail-safe)."""
        fingerprints = self._llm_server_key_fingerprints()
        if fingerprints is None:
            return True
        settings = self._llm_server_settings()
        desired = binding_marker(settings.enabled, fingerprints)
        marker_ok = self._applied_binding_store().read() == desired
        # The live gate's own key set decides; the marker only stands in for a
        # status that reports none (`LlmServerProxyController.converged`).
        return self._llm_proxy().converged(
            settings.enabled, fingerprints, model_port, marker_ok=marker_ok)

    def _loading_door(self) -> LoadingDoor:
        """The way-back door's hold record, a lazily-cached injectable seam (B1)."""
        existing = getattr(self, "_loading_door_cache", None)
        if existing is None:
            existing = LoadingDoor.beside(getattr(self, "_cluster_mode_store_cache", None))
            self._loading_door_cache = existing
        return existing

    def _loading_door_now(self) -> float:
        """The clock the door's bound is read against, an injectable seam."""
        return float((getattr(self, "_loading_door_clock", None) or time.time)())

    def _mark_llm_binding_applied(self, enabled: bool, api_keys) -> None:
        """Record the proxy binding just converged to (FINDING A), the commit record:
        a half-applied toggle leaves the OLD marker, read as drift by the reconcile.

        The marker is over per-key FINGERPRINTS - the SAME basis the poll's desired
        marker uses - so it is written through :func:`applied_marker` over the
        just-applied plaintext keys, which equals what the broker's ``list()`` will
        report on the next poll. The Mode B reconcile commits through the same
        derivation into the same file."""
        self._applied_binding_store().write(applied_marker(enabled, api_keys))
        # B1: the gate was just put in front of a model (or stopped), so a
        # way-back "loading" door it replaced is no longer held.
        self._loading_door().clear()

    def _converge_llm_proxy(self, model_port: int) -> None:
        """Bring the auth proxy to the persisted state in front of ``model_port``.

        Start the keyed proxy when enabled with a key set, stop it otherwise, then
        mark it. BEST-EFFORT - a proxy that will not (re)start leaves the marker
        uncommitted for the failure-watch to retry. A broker that cannot answer
        leaves the gate as it is (fail-safe): nothing applied, no marker written.
        """
        keys = self._llm_server_active_keys()
        if keys is None:
            return
        settings = self._llm_server_settings()
        try:
            self._llm_proxy().apply(settings.enabled, keys, model_port)
        except Exception:
            return
        self._mark_llm_binding_applied(settings.enabled, keys)

    def apply_llm_server(self) -> Dict[str, Any]:
        """Converge the LLM Server auth PROXY to the persisted state.

        The action the ``llm_server.apply`` job runs whenever the user enables,
        disables or rotates the LLM Server. It:

        * resolves the GPU serving target through the ONE gate
          (:func:`~vaelor.gpu_serving_target.resolve_gpu_serving_target`):
          available for a managed-local Mode A model AND a Mode B cluster
          endpoint, not for a single-model box, a hosted lease or a
          stock-compose server;
        * starts the keyed nginx auth proxy in front of that target's LOOPBACK
          port when enabled, stops+removes it when disabled, and re-keys it on a
          rotate - engine-agnostic, because the proxy fronts a port and never
          asks what is behind it. That is what makes the LLM Server follow the
          mode switch for free: llama.cpp on 8080-8099 or vLLM on 8000-8079, one
          gate and one method.

        **The model is NOT relaunched and the AI-Chat credential is NOT touched.**
        Both stay loopback and keyless throughout, so a toggle can never disturb
        internal AI Chat or evict a loaded model - the whole point of the pivot to a
        proxy. FINDING C rollback still holds via the applied marker: the proxy is
        converged FIRST and the marker written only on success
        (:meth:`_mark_llm_binding_applied`), so a proxy that fails to start leaves
        the marker as drift for the reconcile to repair and never reports a broken
        toggle as done. A genuine proxy failure propagates so the job records it; the
        30 s failure-watch then retries.

        Non-fatal on "nothing to do": a box with no GPU serving target returns the
        gate's own ``reason`` rather than raising, and the persisted state still
        governs the NEXT deploy.

        **Under the GPU serving mutex**, the one the Mode A failure-watch and the
        Mode B reconcile hold across their own proxy applies: the job and a
        reconcile pass cannot interleave two ``proxy_start`` calls with two
        different key sets. Re-entrant, so the failure-watch's own call through
        `_healthy_binding_result` (which already holds it) is unaffected.
        """
        with self._gpu_watch_lock():
            return self._apply_llm_server_locked()

    def _apply_llm_server_locked(self) -> Dict[str, Any]:
        settings = self._llm_server_settings()
        mode_state = self._cluster_mode_state()
        # Review S4: while the Mode B deployment is UNLOADED the port is the
        # switch's unloaded-state door (VD-159), and an Enable, a Disable or a
        # key change must reach it with this job - a Disable is the owner's
        # remedy for a key they no longer trust, and it used to wait up to
        # 30 s for the next reconcile pass. Asked before the serving target,
        # because after a manual unload there is none.
        if gpu_cluster_mode_active(mode_state) and (
            self._cluster_record_state(mode_state) == UNLOADED_STATE
        ):
            return self._apply_unloaded_door(settings)
        target = resolve_gpu_serving_target(self.credential_broker, mode_state)
        if not target.available:
            # B1: nothing to front - a held way-back door must not outlive that.
            # Mode A only: in Mode B a door that fronts nothing is the switch's
            # own (loading while a deploy runs, unloaded while paused).
            if not gpu_cluster_mode_active(mode_state):
                close_door(self._loading_door(), self._llm_proxy(), "AI Chat has no model "
                           "the LLM Server fronts ({})".format(target.reason or "none"),
                           self._loading_door_now())
            return {
                "applied": False, "reason": target.reason,
                "enabled": settings.enabled,
            }
        # G3a/G3b: an unloaded (or loading) cluster deployment's door is never
        # pointed at the model from here. An idle unload holds ai-chat on the
        # cluster credential, so the target still resolves; fronting it would
        # put a keyed 502 on the LAN until the reconcile's unloaded pass moved
        # the door again. A disable still goes through, because "down" is
        # right either way.
        if (
            target.kind == KIND_CLUSTER and settings.enabled
            and not self._cluster_record_serving(mode_state)
        ):
            return {
                "applied": False, "reason": CLUSTER_NOT_SERVING,
                "enabled": settings.enabled,
            }
        # BROKER-DOWN FAIL-SAFE: without a definitive key set, leave the gate as
        # it is (a running gate stays up, none torn down or rendered keyless); the
        # failure-watch retries. Only a real answer (list, maybe empty) converges.
        keys = self._llm_server_active_keys()
        if keys is None:
            return {
                "applied": False, "reason": "broker-unavailable",
                "enabled": settings.enabled,
            }
        port = target.port
        # The LLM Server proxy fronts the fork HOST model only; a stock-GGUF ai-chat
        # runs in the ``model-chat`` compose project (its own loopback port) and is
        # out of this feature's single-node scope. A cluster target never runs in
        # that project, so the read is skipped for one.
        if target.kind != KIND_CLUSTER and (
            self._gpu_chat_compose_project_for(port) is not None
        ):
            # B1: a stock-GGUF model is never fronted, so a held way-back door
            # is closed here - a Disable or a key change always reaches it.
            close_door(self._loading_door(), self._llm_proxy(),
                       "AI Chat's model is the stock compose server, which the "
                       "LLM Server does not front", self._loading_door_now())
            return {
                "applied": False, "reason": "compose-backed",
                "enabled": settings.enabled,
            }
        # Converge the proxy, THEN commit the marker (FINDING C): a proxy start/stop
        # failure raises BEFORE the marker is written, so a half-failed toggle stays
        # visible as drift for the failure-watch reconcile rather than being reported
        # done. The proxy fronts the model's loopback ``port``.
        self._llm_proxy().apply(settings.enabled, keys, port)
        self._mark_llm_binding_applied(settings.enabled, keys)
        return {
            "applied": True,
            "enabled": settings.enabled,
            "port": port,
            "proxy_port": LLM_SERVER_PROXY_PORT,
            "kind": target.kind,
        }

    def _apply_unloaded_door(self, settings: LlmServerSettings) -> Dict[str, Any]:
        """Bring the unloaded-state door to the settings, through the switch.

        The switch owns which door that is (the wake door or the one that
        answers for itself) and converges it to the CURRENT flag and key set;
        this only asks it to do so now. ``applied`` is what the switch
        reports, so a door that could not be converged is not called applied.
        An executor with no switch (a box that never clustered cannot be
        here, but a test's may) touches nothing and says so.
        """
        switch = getattr(getattr(self, "cluster", None), "gpu_mode_switch", None)
        converged = False
        if switch is not None:
            try:
                converged = bool(switch.converge_unloaded_doors())
            except Exception as error:  # noqa: BLE001 - the reconcile retries
                LOGGER.warning(
                    "The LLM Server's door could not be brought to its settings "
                    "while the cluster model is unloaded: %s", error,
                )
        result: Dict[str, Any] = {
            "applied": converged, "enabled": settings.enabled, "kind": KIND_CLUSTER,
            "proxy_port": LLM_SERVER_PROXY_PORT,
        }
        if not converged:
            result["reason"] = UNLOADED_DOOR_NOT_APPLIED
        return result

    def _cluster_record_state(self, mode_state: Any) -> str:
        """The state of the Mode B deployment the mode file names, or ``""``.

        Read through the executor's own cluster store (``self.cluster.store``,
        the store the mode reconcile reads). ``""`` when there is no store or
        no name, and - logged - when the store cannot answer.
        """
        store = getattr(getattr(self, "cluster", None), "store", None)
        name = str(getattr(mode_state, "deployment_name", "") or "")
        if store is None or not name:
            return ""
        try:
            record = store.get_pooled_deployment(name)
        except Exception as error:  # noqa: BLE001 - the reconcile has the last word
            LOGGER.warning(
                "The cluster deployment '%s' could not be read before the LLM "
                "Server was applied: %s", name, error,
            )
            return ""
        return str((record or {}).get("state", ""))

    def _cluster_record_serving(self, mode_state: Any) -> bool:
        """Whether the Mode B deployment the mode file names reads ``healthy``.

        A state that cannot be read (:meth:`_cluster_record_state`) reads as
        serving - the answer this apply gave before G3a; the 30 s mode
        reconcile corrects the door from the record either way, so the guess
        does not outlive one tick.
        """
        state = self._cluster_record_state(mode_state)
        return not state or state == HEALTHY_STATE

    def run_llm_server_apply(self, job: Dict[str, Any]) -> Dict[str, Any]:
        """Run and finish an ``llm_server.apply`` job.

        Kept beside the action it drives rather than inline in
        ``JobExecutor.run_once`` so that module stays at its 1,000-line ceiling:
        the dispatch there is a one-line delegation, and the whole action -
        converge the auth proxy to the state and record the outcome - lives here.
        """
        result = self.apply_llm_server()
        return self.store.finish(
            job["id"], state="completed", result=result,
            message=APPLIED_MESSAGE if result.get("applied") else NOT_APPLIED_MESSAGE,
        )
