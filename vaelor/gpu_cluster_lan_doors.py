"""The GPU mode switch's two LAN doors: the LLM Server proxy and the balancer.

Housed out of `gpu_cluster_mode` for the 1,000-line ceiling `CLAUDE.md` sets,
as a mixin :class:`~vaelor.gpu_cluster_mode.ClusterModeSwitch` composes: the
switch owns WHEN a door moves (which pass, under which lock, in which order
against the lease), and this module owns HOW each one is brought to where it
should be. Everything here runs on the switch's own collaborators - ``broker``,
``llm_proxy``, ``llm_server_store`` and ``balancer`` - and inside the switch's
GPU serving mutex, because a door converged outside it could be overtaken by a
leave that had already taken it down (VD-129, in the switch's docstring).

* **The LLM Server proxy** is the keyed nginx gate in front of the serving
  model's loopback port. In Mode B it fronts the cluster's port, and
  :meth:`ClusterLanDoorsMixin._converge_proxy` is the one place Mode B brings
  it to the LLM Server's state - the CURRENT flag and the CURRENT key set,
  read on every call.
* **The replica balancer** (VD-129) is the replicated deployment's endpoint on
  this controller. It runs whenever the deployment is healthy, whatever the
  LLM Server flag says, and :meth:`ClusterLanDoorsMixin._converge_balancer`
  keeps it up on the healthy pass.
* **Both come down together** on every leave
  (:meth:`ClusterLanDoorsMixin._stop_lan_doors`), the proxy first.
* **While the deployment is unloaded the proxy stays up and keyed** (VD-159),
  and only the balancer comes down. After an IDLE unload, while the control
  plane's wake responder answers, it fronts that responder
  (:mod:`vaelor.llm_server_wake`, ACC-058), so an LLM Server client wakes the
  model the way AI Chat and the gateway do. In every other unloaded state - a
  manual unload, a cause that cannot be read, a responder that is not up - it
  fronts nothing and answers for itself: 401 without a current key, and with
  one a plain 503 that says the model is not loaded
  (:meth:`ClusterLanDoorsMixin._converge_unloaded_door`). That door needs no
  other process, which is what makes it the one a reboot cannot lose.
"""

from __future__ import annotations

import logging
from typing import Any, Mapping, Optional

from .credential_broker import CredentialError
from .gpu_pool_units import is_replicated
from .gpu_serving_target import (
    CLUSTER_INFERENCE_PURPOSE,
    KIND_CLUSTER,
    MODE_CLUSTER,
    UNLOAD_CAUSE_IDLE,
    resolve_gpu_serving_target,
    unload_cause,
)
from .llm_server_state import (
    active_key_fingerprints,
    active_keys,
    applied_marker,
    binding_marker,
    migrate_legacy_key,
)
from .llm_server_proxy import UPSTREAM_MODEL
from .llm_server_wake import wake_door_ready


LOGGER = logging.getLogger(__name__)


#: The action both directions name: the LAN gate comes down on the way in
#: (its upstream is about to stop answering) and on the way out (its upstream is
#: about to be torn down), so `enter` and `leave` are symmetric about it.
STOPPING_THE_PROXY = "stop the LLM Server proxy"

#: The replica balancer (VD-129) comes down on every leave, under the lock,
#: for the reason the switch's docstring gives; on the way in there is none yet.
STOPPING_THE_BALANCER = "stop the replica balancer"


class ClusterLanDoorsMixin:
    """Converge or stop the LLM Server proxy and the replica balancer."""

    def _converge_proxy(self, state: Any, *, wake_door: bool = False) -> bool:
        """Bring the LLM Server auth proxy to the CURRENT flag and key set.

        What the door fronts: the cluster's own loopback port by default, the
        control plane's wake responder (``wake_door``, an explicit marker the
        bridge and its status carry) on an idle unloaded pass
        (:meth:`_converge_wake_door`). Either way the target must resolve to
        the cluster first, so the door only ever opens for a cluster AI Chat
        still points at.

        **The desired state is read now, never remembered.** Enabled with a
        non-empty broker key set: the keyed gate in front of the cluster's
        loopback port. Disabled, or no keys: no gate. The flag is
        ``llm_server_store.read()`` on every call (it fails safe to disabled).
        The first cut gated this on ``llm_server_was_enabled``, a snapshot taken
        at `enter`: a box that entered Mode B with the server off, and was
        switched on afterwards, had its gate taken down by the next re-serve and
        never brought back, while the console said "Serving" (2026-09-28, eight
        days, LESSONS pattern 1). No caller gates this any more; every pass that
        can see a serving cluster converges it.

        **Drift includes the key set, read off the running gate.** The
        controller's `converged` compares the live gate (up, on this listen host
        and port, fronting this model port) AND the SET-hash of the keys the
        bridge rendered into it against the broker's fingerprint-only ``list()``
        - so a key minted, rotated or revoked while the gate is up is drift, and
        the poll decrypts nothing. The first cut read no key at all and a
        revoked key stayed admitted. The applied-binding marker (the one the
        Mode A failure-watch keeps too) is committed after a successful apply
        and consulted only for a gate whose status reports no key set: it
        records what was ASKED for, and a start that timed out on the client
        but finished on the bridge makes it disagree with what RUNS. The
        plaintext set is read only when an apply is actually needed. A
        pre-F3b-ii key still on disk is migrated into the broker first, as the
        Mode A path does, so it is not read as an empty set that stops the gate.

        Gated on the S1 target, so the proxy is only ever pointed at a port this
        appliance has independently agreed is a serving target. Best effort: a
        proxy that will not start leaves the LAN gate down and the cluster
        serving internally, and the next pass retries. A right gate is left
        alone - `apply` replaces the container, and applying every 30 s took
        the LAN endpoint down for about a second per pass (VD-127, second run).
        Broker-down fail-safe: an unreadable key set leaves the gate exactly as
        it is, never torn down and never rendered keyless.
        """
        target = resolve_gpu_serving_target(self.broker, state)
        if target.kind != KIND_CLUSTER:
            return False
        return self._converge_gate(
            int(target.port), {"wake_door": True} if wake_door else {},
        )

    def _converge_gate(self, port: int, marker: Mapping[str, bool]) -> bool:
        """Bring the gate to the current flag and key set, fronting what ``marker`` names.

        The body every door shares (:meth:`_converge_proxy` for the model and
        the wake responder, :meth:`_converge_unloaded_door` for the door that
        fronts nothing): read the flag and the fingerprints now, leave a right
        gate alone, apply only on drift, and commit the marker after a
        successful apply. ``marker`` is passed to the proxy controller as it
        is - empty for the model, so a controller is only ever handed a
        keyword it was asked to act on.
        """
        enabled = bool(self.llm_server_store.read().enabled)
        try:
            migrate_legacy_key(self.llm_server_store, self.broker)
            fingerprints = active_key_fingerprints(self.broker)
        except CredentialError:
            return False
        marker_ok = self.applied_binding.read() == binding_marker(enabled, fingerprints)
        if self.llm_proxy.converged(
            enabled, fingerprints, port, marker_ok=marker_ok, **marker,
        ):
            return True
        try:
            keys = active_keys(self.broker)
        except CredentialError:
            return False
        try:
            self.llm_proxy.apply(enabled, keys, port, **marker)
        except Exception as error:  # noqa: BLE001 - never fail a healthy deploy
            LOGGER.warning(
                "The LLM Server proxy could not be %s: %s",
                "started as the unloaded door" if marker.get("unloaded_notice")
                else "moved in front of the cluster endpoint", error,
            )
            return False
        self.applied_binding.write(applied_marker(enabled, keys))
        return True

    def _converge_wake_door(self, state: Any) -> bool:
        """The idle unloaded pass's door: keyed, fronting the wake responder.

        ACC-058. True when the door was brought to the flag in front of the
        control plane's wake responder on its unix socket. Two things must hold first, or it answers False and
        the caller takes the door down as every unloaded pass did before:

        * the unload cause is IDLE (`gpu_serving_target.unload_cause`, VD-136):
          the ``ai-chat`` lease still names the cluster. A manual unload parks
          it and is never woken; an unreadable lease is not guessed idle;
        * the responder is answering on its socket (:func:`wake_door_ready`,
          review S2): a door pointed at nothing - or at something else - would
          forward keyed requests to it.

        False too when the key set cannot be read or the door would not start.
        A disabled server or an empty key set converges to no door at all, and
        still reads True.
        """
        target = resolve_gpu_serving_target(self.broker, state)
        if unload_cause(target) != UNLOAD_CAUSE_IDLE or not wake_door_ready():
            return False
        return self._converge_proxy(state, wake_door=True)

    def _converge_unloaded_door(self) -> bool:
        """The door for every other unloaded state: keyed, answering for itself.

        VD-159. The defect it closes: after a reboot with the cluster model
        unloaded, port 11434 had no listener at all. The door used to exist
        while unloaded only as the wake door, which needs an idle unload AND
        the control plane's responder answering; in every other case - a
        manual unload, an unreadable cause, a responder not up yet - the
        unloaded pass took the door down, and a LAN client met a refused
        connection it could not tell from a machine that was off.

        This door fronts nothing, so it needs no serving target and no other
        process: only the LLM Server's own flag and keys. Enabled with keys,
        nginx answers 401 without a current key and a plain JSON 503 with
        ``Retry-After`` with one. Disabled, or with no keys, it converges to
        no door at all and still reads True. False when the key set cannot be
        read or the door would not start; the caller then asks
        :meth:`_gate_fronts_the_model` whether what is left standing is safe.
        """
        return self._converge_gate(0, {"unloaded_notice": True})

    def _hold_door_while_loading(self) -> None:
        """On the way in, stop fronting llama.cpp but keep port 11434 answering.

        ACC-193, VD-166. `enter` used to stop the gate outright, so from the moment
        llama.cpp stopped until the cluster's gate started a LAN client met a
        refused connection (about a minute in the live test). The gate is now
        moved onto the door that fronts nothing and says the model is loading:
        401 without a current key, 503 with ``Retry-After`` with one. A
        disabled server or an empty key set has no door, as before. Only when
        that door cannot be put up (the key store or the bridge did not answer)
        is the gate stopped, never left in front of the stopping model.
        """
        enabled = bool(self.llm_server_store.read().enabled)
        try:
            keys = active_keys(self.broker)
            self.llm_proxy.apply(enabled, keys, 0, unloaded_notice=True, loading=True)
        except Exception as error:  # noqa: BLE001 - the stop below is the fallback
            LOGGER.warning(
                "The LLM Server could not be kept answering while the cluster "
                "model loads, so its door is closed until it serves: %s", error,
            )
            self._quietly(lambda: self.llm_proxy.stop(), STOPPING_THE_PROXY)
            return
        self.applied_binding.write(applied_marker(enabled, keys))

    def _gate_fronts_the_model(self) -> bool:
        """Whether the RUNNING gate stands in front of the model's port.

        Asked only when no unloaded-state door could be converged (the key
        store did not answer, or the bridge refused the start). A gate on the
        stopped model answers the LAN with an nginx 502 and must come down. A
        gate that is already the wake door or the unloaded door is right as
        it stands and is left alone: taking it down would close port 11434
        for as long as the key store stays unreadable (2026-09-30 review,
        S2). Read off the gate's own status. A status that cannot be read is
        not known to be safe, so it answers True; a gate that is not running
        has nothing to take down.
        """
        try:
            status = self.llm_proxy.status() or {}
        except Exception:  # noqa: BLE001 - an unreadable gate is not known safe
            return True
        if not status.get("running"):
            return False
        return str(status.get("upstream") or UPSTREAM_MODEL) == UPSTREAM_MODEL

    def _converge_balancer(
        self, state: Any, deployment: Optional[Mapping[str, Any]]
    ) -> bool:
        """Keep a replicated deployment's balancer up on the healthy pass (VD-129).

        Only for a ``replicated`` record, and only when the controller's own
        `converged` says the live balancer is not already pooling this record's
        replicas on its port - so a right balancer is left alone the way a
        right LAN gate is. The key it forwards is the cluster credential's,
        read from the broker only when a start is needed. Best effort, like
        the proxy: a balancer that will not start is logged and the next pass
        retries; the record stays healthy because the replicas are. Runs under
        the switch's ``watch_lock``, so a start holds the mutex for as long as
        the bridge takes, up to `hardware_bridge_client.BALANCER_START_SOCKET_TIMEOUT`.
        """
        if deployment is None or not is_replicated(deployment):
            return False
        if self.balancer.converged(deployment):
            return True
        try:
            lease = self.broker.resolve(
                state.cluster_credential_id, CLUSTER_INFERENCE_PURPOSE
            )
            self.balancer.start(deployment, str(lease.get("api_key", "")))
        except Exception as error:  # noqa: BLE001 - never fail a healthy deploy
            LOGGER.warning("The replica balancer could not be started: %s", error)
            return False
        return True

    def converge_unloaded_doors(
        self, *, allow_wake: bool = True, stop_balancer: bool = True,
    ) -> bool:
        """Bring the doors to the unloaded state NOW; whether the door got there.

        For the three callers that must not wait for the next 30 s pass: the
        unload the moment its row reads unloaded - before a unit is stopped,
        so with ``allow_wake`` and ``stop_balancer`` both False, moving the
        LLM Server's door and nothing else - and again once its units are
        down; and the LLM Server's apply job, so a Disable or a revoked key
        made while unloaded takes effect with the job. False, and nothing
        touched, outside Mode B.
        """
        with self.watch_lock:
            state = self.store.read()
            if state.mode != MODE_CLUSTER:
                return False
            return bool(self._unloaded_pass(
                state, allow_wake=allow_wake, stop_balancer=stop_balancer,
            )["llm_server_converged"])

    def converge_serving_door(self, deployment: Optional[Mapping[str, Any]] = None) -> bool:
        """Point the LLM Server's door at the serving cluster NOW; and its balancer.

        For a Load that just finished and an unload that had to put its row
        back to healthy: without it the door kept answering "not loaded" in
        front of a model that was serving until the next pass. With the
        serving ``deployment`` record the replica balancer is converged too
        (re-review R1): a rolled-back unload must not leave the replicas it
        failed to stop without their front. Answers the door's result.
        False, and nothing touched, outside Mode B.
        """
        with self.watch_lock:
            state = self.store.read()
            if state.mode != MODE_CLUSTER:
                return False
            door = self._converge_proxy(state)
            if deployment is not None:
                self._converge_balancer(state, deployment)
            return door

    def _stop_lan_doors(self, deployment_name: str) -> None:
        """Take both LAN doors down, the proxy first; each stop is idempotent.

        Every leave makes exactly this pair of stops, so it is written once:
        the proxy because it fronts a port that is dead or about to be (a
        proxy left up answers the LAN with an nginx 502), the balancer because
        a healthy pass that read the record a moment earlier could otherwise
        have started it again. A no-op for a distributed deployment's
        balancer, which never had one.
        """
        self._quietly(lambda: self.llm_proxy.stop(), STOPPING_THE_PROXY)
        self._quietly(
            lambda: self.balancer.stop(deployment_name), STOPPING_THE_BALANCER
        )
