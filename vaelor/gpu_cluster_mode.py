"""The GPU serving mode switch: stop Mode A, serve on the cluster, come back.

VD-125 recorded the two-node vLLM proof and then said plainly what was still
missing: *"The stopping half of that switch is still manual (disable the LLM
Server, move AI Chat to the NPU connection so the watch no-ops, stop the model
container over SSH - no product control exists for the last step)."* This module
is that control.

**One active GPU engine at a time.** Mode A serves the AI-Chat model on
llama.cpp on this controller and optionally exposes it through the LLM Server's
auth proxy. Mode B (GPU clustering, two or more GPU nodes) disables llama.cpp and
moves both AI Chat and the LLM Server onto vLLM; tearing the cluster down reverts
to llama.cpp, which is never deleted. The rule the switch exists to enforce is
**no double residency**: the Mode A 27B holds most of the controller's ~30 GiB
aperture, so vLLM cannot start until it is gone, and "gone" is confirmed by
reading the GPU, not by having called ``stop``.

**Three verbs and a reconcile, because a deploy is not atomic.**
:meth:`ClusterModeSwitch.enter` clears the GPU before the deploy touches a node;
:meth:`~ClusterModeSwitch.repoint` moves AI Chat onto the cluster once its
endpoint is healthy; :meth:`~ClusterModeSwitch.leave` restores Mode A. Every
failure between them lands in a state :meth:`~ClusterModeSwitch.reconcile` can
finish - which is why the mode file is written FIRST, before anything is stopped
or moved. A crash after "the model is stopped and AI Chat has been moved" but
before the file said so would leave a box that reads as Mode A, no-ops its
failure-watch (the Assistant's endpoint is not a GPU tier), and never brings AI
Chat back. Recorded intent first, then act on it, is the recoverable order.

**A deploy in flight OWNS the switch, and the reconcile tells one from the job
that is running.** The window between `enter` and `repoint` is minutes to hours
long - it contains a weights fetch of tens of gigabytes per node - and for all
of it the deployment record reads ``deploying``, not ``healthy``. A reconcile
that treated "not healthy" as "no cluster" would call `leave` under a live
deploy: llama.cpp would relaunch into the aperture vLLM is loading into,
`repoint` would inherit a record the reconcile had already reset, and the pass
after that would deactivate ``ai-chat`` outright. The record carries
``deploy_in_flight``, set by `enter` and cleared by `repoint` and by `leave`;
that flag, or a deployment record in state ``deploying``, is the CLAIM. The
WITNESS that the claim is live is a fact this process already holds: the
reconcile runs on a thread of the very executor that runs the deploy job, and
:meth:`ClusterModeSwitch.reconcile` is handed ``deploy_job_active`` - whether a
``cluster.llm.deploy`` job is executing in this process right now. A claim with
no such job is a deploy that died with its executor, and the reconcile leaves
at once - Mode A is back within one 30 s pass - while the layer that can write
the cluster store marks the ``deploying`` row ``failed``. The first cut used a
wall-clock backstop instead (three pull timeouts): it held a dead box in Mode B
for six hours with llama.cpp stopped and AI Chat parked, and being blind to the
node count it would have torn a legitimate slow three-node deploy down live. A
timer stands in for a fact only when the fact is unreachable, and this one was
in the same process all along.

**And what a record RUNS is stopped before Mode A is restored - on every leave
the reconcile makes over a record.** A deploy that died after its units were
written left ``vaelor-vllm-<name>-server`` loading into the controller's
aperture (``Restart=on-failure``, ``WantedBy=multi-user.target``: a residency
that outlived a reboot) and the first abandoned pass restored the lease over
it, so the failure-watch relaunched llama.cpp beside it. The reconcile is handed
``stop_deploy_units`` by the layer that can reach the nodes and runs it under
the serving lock BEFORE `_leave_locked` (D9's order). Wired first into the
abandoned pass alone, it left the other two leaves - a record no longer
healthy, a healthy one whose credential is gone - restoring the lease over a
vLLM still serving, so every leave over a record now goes through one
:meth:`ClusterModeSwitch._stop_then_leave`.

**The switch is held by ONE deployment, and `enter` refuses a second.** The
first cut returned silently when the file already read Mode B, "so a retry after
a failed deploy keeps its restore contract" - but a failed deploy's own rollback
resets the file, so the only way to reach `enter` in Mode B is a SECOND deploy
while a first is serving. The no-op then let the second deploy's rollback
`leave` the first cluster's Mode B out from under it: llama.cpp relaunched
beside a healthy vLLM, and a healthy record the reconcile read as Mode A for
ever. Now the file must name the deployment before anything acts on it, in
one place: :meth:`ClusterModeSwitch.leave_if_owned` is the gate both the
deploy's rollback and `remove` go through.

**The replica balancer is the switch's fifth collaborator (VD-129).** A
replicated deployment serves through a second nginx container on this
controller, and it runs whenever the deployment is healthy, whatever the LLM
Server flag says: the healthy pass converges it beside the LAN gate
(:meth:`ClusterModeSwitch._converge_balancer`, from the record's replicas and
the cluster credential's key), and every leave stops it under the lock beside
the proxy - because a leave that stopped it BEFORE the lock, as `remove`'s
unit stop does, could be overtaken by a healthy pass that read the record a
moment earlier and started it again over a cluster that was gone.

**The LLM Server's gate follows the flag as it is NOW, in every mode.** The
switch never decides the gate from anything it recorded at `enter`: `repoint`,
the healthy pass and `leave` all read the current flag, and the key set the
gate carries is compared on every pass (`gpu_cluster_lan_doors`). A snapshot
of the flag taken on the way in held a gate down for eight days on a box whose
owner had switched the server on after clustering (2026-09-28).

**The mode file itself** - the record, its fail-safe read and its store - is
`gpu_cluster_mode_state`'s, re-exported here; its docstring carries the
cross-account placement rule (LESSONS pattern 13) this module's used to.
"""

from __future__ import annotations

import logging
import threading
import time
from dataclasses import replace
from typing import Any, Callable, Dict, List, Mapping, Optional

from .accelerator_runtime import MINIMUM_ACCELERATED_BYTES
from .credential_broker import NO_ACTIVE_CREDENTIAL, CredentialError
from .gpu_cluster_lan_doors import (  # noqa: F401 - the door verbs, re-exported
    STOPPING_THE_BALANCER, STOPPING_THE_PROXY, ClusterLanDoorsMixin,
)
from .gpu_pool_units import controller_unstopped
from .gpu_cluster_ai_chat import (
    heal_onto_cluster, keep_owner_choice, leave_keeps_owner_choice, settle_unload_cause,
)
from .managed_local_credentials import PREFIX as MANAGED_LOCAL_PREFIX
from .managed_local_credentials import pins_an_flm_tag
from .gpu_loading_door import LoadingDoor, leave_door
from .gpu_serving_target import (
    AI_CHAT_PURPOSE,
    MODE_CLUSTER,
    MODE_SINGLE,
)
from .gpu_cluster_mode_state import (  # noqa: F401 - the record, re-exported
    MODE_DIRECTORY, MODE_FILE, ClusterModeState, ClusterModeStore, _coerce,
)
from .llm_server_state import (
    APPLIED_BINDING_FILE,
    AppliedBindingStore,
    LlmServerStore,
)


LOGGER = logging.getLogger(__name__)


#: The process-wide GPU serving mutex (D5). One lock, held by the 30 s GPU
#: failure-watch for a whole pass, by the mode reconcile for a whole pass, and by
#: `enter`/`leave` for their whole call - because those are the four things that
#: can start or stop a GPU engine, and two of them running at once is the double
#: residency the switch exists to prevent. A relaunch already in flight when a
#: deploy calls ``enter`` must finish before ``enter`` reads the GPU, or ``enter``
#: would confirm an empty aperture that a llama.cpp start is about to fill.
#:
#: Re-entrant because :meth:`ClusterModeSwitch.reconcile` holds it and then does
#: what ``leave`` does; a plain lock would deadlock the reconcile against itself.
#: It is a module global rather than a constructor argument because the two
#: holders are built in different places - the executor mixin and
#: `cluster_operations` - and threading one object between them would be a seam
#: whose only failure mode is the two ending up with different locks.
GPU_SERVING_LOCK = threading.RLock()


#: How long `enter` will wait for the Mode A model to actually leave the GPU, and
#: how often it re-reads. Sixty seconds is far above a container stop plus the
#: driver freeing a ~21.5 GiB allocation, and far below the deploy's own weights
#: pull, so a stuck model is reported as a refusal with nothing started rather
#: than discovered by vLLM failing to allocate ten minutes later.
STOP_CONFIRM_SECONDS = 60.0
STOP_POLL_SECONDS = 2.0

#: How much GPU memory may still be held for the model to count as gone. A
#: display buffer and the driver's own scratch sit well under half a gigabyte,
#: and a resident 27B is tens of times this, so
#: :data:`~vaelor.accelerator_runtime.MINIMUM_ACCELERATED_BYTES` separates the
#: two with room to spare - the same floor the residency verdict uses for the
#: same reason, reused rather than re-picked.
RESIDENT_MODEL_MAX_BYTES = MINIMUM_ACCELERATED_BYTES

#: What `enter` raises when the model will not leave. A refusal, not a warning:
#: starting vLLM beside a resident 27B is the OOM this whole switch prevents.
MODEL_STILL_RESIDENT = (
    "The AI Chat model is still on the GPU after {:.0f} seconds, so the cluster "
    "cannot start beside it. Check the GPU model server and try again."
)

#: The one description of "unassign ai-chat", used on the way in (no Assistant to
#: degrade to) and on the way out (the model that held it is gone). Written once
#: because it names one action; two spellings of it in one module would be the
#: same sentence written twice.
CLEARING_AI_CHAT = "clear the AI Chat connection"


#: The two deployment-record states the reconcile reads. ``deploying`` is what a
#: deploy writes before it calls `enter`, so it is a SECOND claim on the switch
#: beside the mode file's own flag - either one is enough, because the two are
#: written at different moments and a reconcile that needed both would find a
#: gap between them. Neither is believed without the job fact (see the module
#: docstring): a claim with no deploy job running in this process is a deploy
#: that died.
DEPLOYING_STATE = "deploying"
HEALTHY_STATE = "healthy"

#: The state a manually unloaded deployment carries (G3a): its serving units
#: are stopped to reclaim the GPU, but its record, credential and cached
#: weights are kept so a later load brings it back warm. The reconcile reads it
#: as a KEEP-Mode-B steady state, distinct from ``deploying`` (a load or deploy
#: in flight) and from ``healthy`` (serving), so an unloaded row is never torn
#: down.
UNLOADED_STATE = "unloaded"

#: The reconcile's answers, named because the mode watch acts on them:
#: ``deploy-in-flight`` is "a deploy job is running here, wait"; the other
#: three are the LEAVES - a recorded Mode B torn down, the row it was made
#: over marked ``failed`` with a note that says which of them it was.
DEPLOY_IN_FLIGHT = "deploy-in-flight"
DEPLOY_ABANDONED = "deploy-abandoned"
NO_HEALTHY_DEPLOYMENT = "no-healthy-deployment"
CLUSTER_CREDENTIAL_MISSING = "cluster-credential-missing"

#: The reconcile's answer for an ``unloaded`` row (G3a): Mode B is kept and
#: nothing is converged. Deliberately OUTSIDE
#: `gpu_cluster_mode_watch.LEAVE_REASON_NOTES`, so the watch's ``_mark_left``
#: never fires and the unloaded row is preserved untouched - it is a steady
#: state a later load returns from, not a leave.
DEPLOYMENT_UNLOADED = "deployment-unloaded"

#: Why a leave left ``ai-chat`` UNASSIGNED instead of restoring it: a unit of
#: the record on the CONTROLLER would not stop, so vLLM may still hold the
#: aperture llama.cpp would be relaunched into. The one exception to "always
#: restore" (VD-127), decided by `gpu_pool_units.controller_unstopped`; a
#: worker's unstopped unit is harmless to the restore and gets none of this.
CONTROLLER_UNIT_UNSTOPPED = "controller-unit-unstopped"

#: What `enter` raises when the mode file already records a Mode B deployment.
#: A refusal, not a no-op, and it names the holder: see the module docstring
#: for the second deploy that tore the first cluster down through the no-op.
MODE_B_HELD = (
    "The GPU serving mode switch is held by the cluster deployment '{}', so "
    "'{}' cannot take it. Remove that deployment first."
)

#: What `enter` raises when the ``ai-chat`` lease - the restore contract - could
#: not be read. That lease is about to be moved and the model behind it stopped;
#: a read that FAILED used to be recorded as "no previous connection", so one
#: broker timeout on the way in made `leave` clear AI Chat for good on the way
#: out. Unreadable is a refusal, with nothing stopped and nothing written.
RESTORE_CONTRACT_UNREADABLE = (
    "The AI Chat connection could not be read from the credential broker, so "
    "the cluster deploy stopped nothing: {}"
)

#: How long `enter` sits silently on the GPU serving mutex before it says who it
#: is waiting for. The only other holder is the 30 s GPU failure-watch, whose
#: longest pass is a llama.cpp relaunch bounded by
#: :data:`vaelor.gpu_rocm_supervisor.GPU_HEALTH_DEADLINE_SECONDS` (240 s), so the
#: wait is finite and worth a checkpoint rather than looking like a hung deploy.
LOCK_REPORT_SECONDS = 5.0
LOCK_WAIT_NOTE = "Waiting for the AI Chat model's watch to release the GPU"

#: What `repoint` raises when the mode file does not say this deployment owns
#: the switch. Silently writing the credential onto whatever the file held was
#: how a reconcile that had already reset the record got a cluster credential
#: stamped back onto a Mode A box.
REPOINT_NOT_OWNED = (
    "The GPU serving mode file does not record '{}' as the deployment holding "
    "the switch, so AI Chat was not moved onto it."
)


def _default_gpu_used_bytes() -> Optional[int]:
    """This controller's currently-held GPU memory, or ``None`` if unreadable.

    The same reader the GPU deploy's residency verdict uses, so "is the model
    gone" and "did the model load" are answered off one measurement, never two.
    """
    from .model_start_verification import accelerator_baseline
    from .platforms.accelerators import discover_accelerators

    return accelerator_baseline(discover_accelerators())


def _default_balancer():
    """The production balancer controller (`gpu_pool_replicas.default_balancer`).

    Imported inside the call so importing this module opens no bridge socket.
    """
    from .gpu_pool_replicas import default_balancer

    return default_balancer()


class ClusterModeSwitch(ClusterLanDoorsMixin):
    """Move GPU serving between llama.cpp (Mode A) and the vLLM cluster (Mode B).

    Every collaborator is injected so the whole state machine is driven in tests
    with no GPU, no bridge and no broker socket:

    * ``broker`` - the credential broker; the switch only ever moves the
      ``ai-chat`` lease and never touches ``deployment-agent`` (D4/VD-042).
    * ``gpu_supervisor`` - :class:`vaelor.gpu_rocm_supervisor.GpuRocmSupervisor`;
      ``stop()`` retires Mode A and ``status()`` says whether it is really gone.
    * ``llm_proxy`` - :class:`vaelor.llm_server_proxy.LlmServerProxyController`;
      the LAN auth gate, stopped on the way in and converged in front of the
      cluster port on `repoint` and on every healthy pass.
    * ``llm_server_store`` - the persisted ``{enabled}`` record (the KEY(S) live
      in the broker as of F3b-ii), read afresh whenever the gate is converged;
      the switch never writes it, so a Mode B cycle leaves the owner's choice
      and the broker's key set exactly as they were.
    * ``applied_binding`` - :class:`vaelor.llm_server_state.AppliedBindingStore`
      over the marker the Mode A failure-watch keeps too, so a key minted,
      rotated or revoked while the gate is up is drift the healthy pass repairs.
    * ``balancer`` - :class:`vaelor.gpu_pool_replicas.BalancerController`; the
      replicated deployment's endpoint (VD-129), converged on the healthy pass
      and stopped on every leave.
    * ``watch_lock`` - the GPU serving mutex, :data:`GPU_SERVING_LOCK` by default.
    """

    def __init__(
        self,
        broker: Any,
        gpu_supervisor: Any,
        llm_proxy: Any,
        llm_server_store: Any = None,
        watch_lock: Any = None,
        *,
        balancer: Any = None,
        applied_binding: Any = None,
        store: Optional[ClusterModeStore] = None,
        gpu_used_bytes: Optional[Callable[[], Optional[int]]] = None,
        sleep: Callable[[float], None] = time.sleep,
        monotonic: Callable[[], float] = time.monotonic,
        now: Callable[[], float] = time.time,
    ):
        self.broker = broker
        self.gpu_supervisor = gpu_supervisor
        self.llm_proxy = llm_proxy
        self.balancer = balancer if balancer is not None else _default_balancer()
        self.llm_server_store = llm_server_store or LlmServerStore()
        self.applied_binding = applied_binding or AppliedBindingStore(
            APPLIED_BINDING_FILE
        )
        self.watch_lock = watch_lock if watch_lock is not None else GPU_SERVING_LOCK
        self.store = store or ClusterModeStore()
        self._gpu_used_bytes = gpu_used_bytes or _default_gpu_used_bytes
        self._sleep = sleep
        self._monotonic = monotonic
        self._now = now
        # B1: the hold record of the way-back door, and the watch's answer to
        # "will it front the restored model?" - wired by the executor.
        self.loading_door = LoadingDoor.beside(self.store)
        self.relaunch_fronted: Optional[Callable[[], str]] = None
        # Whether this process has already said it is waiting on an in-flight
        # deploy. Per-process rather than persisted: it exists only to keep a
        # 30 s watch from writing the same line a hundred times, and a restart
        # saying it once more is the right amount of noise.
        self._in_flight_logged = False

    def state(self) -> ClusterModeState:
        return self.store.read()

    # -- Mode A -> Mode B ------------------------------------------------

    def enter(
        self,
        deployment_name: str,
        report: Optional[Callable[[str], None]] = None,
    ) -> None:
        """Clear the GPU for a cluster deploy, and do not return until it is clear.

        Refused with :data:`MODE_B_HELD` while any deployment holds the switch,
        whatever its name: a healthy cluster is never stopped from under itself,
        and the file that names it is left exactly as it was. Refused with
        :data:`RESTORE_CONTRACT_UNREADABLE` when the ``ai-chat`` lease cannot
        be read, before anything is written or stopped - the lease is the whole
        restore contract, and a contract that was not read cannot be honoured.

        The order is otherwise recorded-intent-first (see the module docstring):
        the mode file is written before anything is stopped or moved, so every
        later failure is a state :meth:`reconcile` finishes rather than a box
        that reads as Mode A while its AI Chat has already been moved away. The
        record it writes carries ``deploy_in_flight``, which is the caller's
        claim on the switch until `repoint` or `leave` releases it.

        AI Chat is unassigned while the cluster comes up, never moved onto the
        Assistant's NPU model (W4d-D26), and says the cluster is loading.

        ``report`` is the deploy's own checkpoint callback, used for one thing:
        saying that this call is queued behind the GPU failure-watch rather than
        stuck. That wait is bounded by one watch pass - a llama.cpp relaunch
        health-gated at :data:`vaelor.gpu_rocm_supervisor.GPU_HEALTH_DEADLINE_SECONDS`
        (240 s) - so it is a delay with a ceiling, and the operator is told which
        ceiling they are waiting on.
        """
        self._acquire_serving_lock(report)
        try:
            current = self.store.read()
            if current.mode == MODE_CLUSTER:
                raise RuntimeError(MODE_B_HELD.format(
                    current.deployment_name, str(deployment_name or "")
                ))
            previous = self._restore_contract()
            self.store.write(ClusterModeState(
                mode=MODE_CLUSTER,
                deployment_name=str(deployment_name or ""),
                previous_ai_chat_credential_id=previous,
                # A recorded fact for the operator reading the file, like
                # ``entered_at``: NOTHING decides on it (the module docstring).
                llm_server_was_enabled=bool(
                    self.llm_server_store.read().enabled
                ),
                entered_at=int(self._now()),
                deploy_in_flight=True,
            ))
            # The LAN gate leaves llama.cpp first: a proxy left in front of a
            # dead upstream is the "reports healthy while badly wrong" shape
            # (LESSONS pattern 1) a client would meet as a 502. It does not
            # close: it answers "loading" until the cluster serves (ACC-193).
            self._hold_door_while_loading()
            self._degrade_ai_chat()
            self._quietly(
                lambda: self.gpu_supervisor.stop(), "stop the GPU model server"
            )
            self._confirm_model_gone()
        finally:
            self.watch_lock.release()

    def _acquire_serving_lock(
        self, report: Optional[Callable[[str], None]]
    ) -> None:
        """Take the GPU serving mutex, saying so if the watch is holding it.

        A plain ``with self.watch_lock`` is silent, and the wait it hides is the
        one an operator would read as a hung deploy: a relaunch already in
        flight holds the lock for a whole health-gated pass. So the first
        :data:`LOCK_REPORT_SECONDS` are silent and anything longer is announced
        once - to the deploy's checkpoint stream when there is one, and to the
        log either way, because a reconcile-side caller has no stream.
        """
        if self.watch_lock.acquire(timeout=LOCK_REPORT_SECONDS):
            return
        LOGGER.info("%s.", LOCK_WAIT_NOTE)
        if report is not None:
            self._quietly(lambda: report(LOCK_WAIT_NOTE), "report the GPU wait")
        self.watch_lock.acquire()

    def _restore_contract(self) -> str:
        """The ``ai-chat`` credential `leave` will put back - read, or refused.

        "Nothing is assigned" is a legitimate empty contract: the broker says so
        with :data:`~vaelor.credential_broker.NO_ACTIVE_CREDENTIAL`, and a box
        whose AI Chat was never configured clusters with nothing to restore.
        Every other failure - the broker socket down, a lease that will not
        decrypt, a malformed answer - is a lease that EXISTS and was not read,
        and is raised as :data:`RESTORE_CONTRACT_UNREADABLE` so the deploy
        fails before this switch has moved or stopped anything.
        """
        try:
            lease = self.broker.resolve_active(AI_CHAT_PURPOSE)
            # W4d-D26/D27: the NPU Assistant's credential is never AI Chat's
            # fallback. Recorded, `leave` handed AI Chat to the NPU runtime.
            if pins_an_flm_tag(lease.get("model")):
                return ""
            return str(lease.get("credential_id", ""))
        except (AttributeError, TypeError, ValueError) as error:
            if isinstance(error, CredentialError) and str(error) == NO_ACTIVE_CREDENTIAL:
                return ""
            raise RuntimeError(
                RESTORE_CONTRACT_UNREADABLE.format(error)
            ) from error

    def _degrade_ai_chat(self) -> None:
        """Take ``ai-chat`` off the model about to stop: cleared, never parked.

        It used to be parked on the Assistant's connection - the NPU model on a
        Z2 - "so AI Chat keeps answering". The owner's rule is that AI Chat never
        runs on the NPU (W4d-D26): parked there it answered from flm-real,
        failed, and offered FastFlowLM's whole uninstalled catalog. Cleared,
        AI Chat says the cluster model is not serving (`chat_inference`), and
        the unload-cause rule still reads a cleared lease as a manual unload.
        """
        self._quietly(
            lambda: self.broker.deactivate(AI_CHAT_PURPOSE),
            CLEARING_AI_CHAT,
        )

    def _confirm_model_gone(self) -> None:
        """Poll until the Mode A model is off the GPU, or refuse the switch.

        Two signals, both required, because either alone has a gap: the
        supervisor's ``status`` knows whether the container it launched is still
        running, and the accelerator reading knows whether the memory came back.
        A container that exited while the driver has not yet freed a 21.5 GiB
        allocation still cannot be started beside.

        An UNREADABLE accelerator (``None``) is accepted on the supervisor's word
        alone rather than blocking the switch for a full minute on a box whose
        adapter reports nothing - and it is logged as such, so the deploy's own
        residency verdict is what reports what actually happened next.
        """
        deadline = self._monotonic() + STOP_CONFIRM_SECONDS
        while True:
            if not self._model_running():
                used = self._read_gpu_used()
                if used is None:
                    LOGGER.warning(
                        "GPU memory could not be read while confirming the AI "
                        "Chat model stopped; continuing on the supervisor's "
                        "status alone."
                    )
                    return
                if used <= RESIDENT_MODEL_MAX_BYTES:
                    return
            if self._monotonic() >= deadline:
                raise RuntimeError(
                    MODEL_STILL_RESIDENT.format(STOP_CONFIRM_SECONDS)
                )
            self._sleep(STOP_POLL_SECONDS)

    def _model_running(self) -> bool:
        """Whether the GPU model server still reports itself up (fail-safe True).

        A supervisor that cannot be asked is treated as still running, so an
        unanswerable status delays the switch rather than clearing it - the
        direction that cannot put two engines on one GPU.
        """
        try:
            status = self.gpu_supervisor.status()
        except Exception:  # noqa: BLE001 - an unreadable status is not "gone"
            return True
        return bool((status or {}).get("running"))

    def _read_gpu_used(self) -> Optional[int]:
        try:
            return self._gpu_used_bytes()
        except Exception:  # noqa: BLE001 - an unreadable adapter says nothing
            return None

    # -- The cluster is healthy ------------------------------------------

    def repoint(
        self, credential_id: str, port: int, deployment_name: str
    ) -> Dict[str, Any]:
        """Point AI Chat at the healthy cluster, and re-arm the LAN gate.

        Called only once the deploy's own health probe has answered, so this
        never activates a credential for an endpoint that is not serving. The
        recorded ``cluster_credential_id`` is what makes
        :func:`~vaelor.gpu_serving_target.resolve_gpu_serving_target` willing to
        call this credential a serving target at all - no other
        ``openai-compatible`` credential is ever fronted by the auth proxy.

        **``deployment_name`` is re-asserted, not inherited.** The first cut read
        the file and wrote the credential onto whatever it found, so a record
        that had been reset in the meantime - by a reconcile, by an operator's
        removal - came back stamped Mode B with no restore contract in it and no
        deployment name to remove it by. Refusing instead leaves the box in the
        state the reset produced, which is Mode A with llama.cpp coming back,
        and the deploy's own rollback then stops what it started.

        **Under the GPU serving mutex, like every other verb (D5).** The read
        of the file, the write, and the lease move are one step: a reconcile
        pass that ran between them would read a file that names this deployment
        with its credential recorded and ``ai-chat`` still parked on the
        Assistant, and "converge" the lease a second time; a `leave` that ran
        between them would reset a file this call then overwrote.
        """
        name = str(deployment_name or "")
        with self.watch_lock:
            current = self.store.read()
            if current.mode != MODE_CLUSTER or current.deployment_name != name:
                LOGGER.warning(
                    "%s (the file reads mode %s for '%s').",
                    REPOINT_NOT_OWNED.format(name), current.mode,
                    current.deployment_name,
                )
                raise RuntimeError(REPOINT_NOT_OWNED.format(name))
            state = replace(
                current,
                mode=MODE_CLUSTER,
                deployment_name=name,
                cluster_credential_id=str(credential_id),
                cluster_port=int(port),
                # The deploy is done owning the switch: from here the reconcile
                # is the owner again, with a healthy record to converge on.
                deploy_in_flight=False, unload_cause="",
            )
            self.store.write(state)
            # The first serve, and the only time the switch takes AI Chat on
            # its own (VD-210): from here the owner's choice stands.
            self.broker.activate(str(credential_id), AI_CHAT_PURPOSE)
            # To the flag as it is NOW - on, off, or switched either way since
            # `enter` - never to what it was when the switch was taken.
            converged = self._converge_proxy(state)
        return {
            "mode": MODE_CLUSTER,
            "credential_id": str(credential_id),
            "port": int(port),
            "llm_server_converged": converged,
        }

    # -- Mode B -> Mode A ------------------------------------------------

    def leave(self) -> Dict[str, Any]:
        """Give AI Chat and the LLM Server back to llama.cpp.

        Idempotent (leaving Mode A is a no-op) and tolerant of the recorded
        credential having been deleted since - in which case ``ai-chat`` is
        cleared and the report says so, rather than the switch raising and
        leaving the box in a mode nothing serves.

        Nothing is relaunched here. Restoring the lease is enough: the 30 s GPU
        failure-watch sees a managed-local target on a port nothing answers and
        brings llama.cpp back within one pass and re-converges the proxy from
        the CURRENT ``enabled`` flag, which this never writes. That is the mechanism
        VD-125 measured live (08:52:10 "health: starting", 08:52:25 healthy), so
        the switch uses it rather than growing a second launcher.
        """
        with self.watch_lock:
            return self._leave_locked()

    def leave_if_owned(self, deployment_name: str) -> Dict[str, Any]:
        """`leave`, only if the mode file names THIS deployment as the holder.

        The one name gate, and both callers that give Mode A back on behalf of
        a deployment go through it: the deploy's rollback and `remove`.
        Removing an unrelated GPU deployment, or rolling back a second deploy
        that `enter` refused, must not tear down the cluster that is serving AI
        Chat - the record and the file have to agree before anything moves.
        Answers ``{}`` when the file names another deployment or none, so a
        caller can tell "left" from "not mine to leave"; the read and the
        leave are one step under the mutex, so the answer cannot go stale
        between them.
        """
        name = str(deployment_name or "")
        with self.watch_lock:
            state = self.store.read()
            if state.mode != MODE_CLUSTER or state.deployment_name != name:
                return {}
            return self._leave_locked()

    def _leave_locked(self, unassign_because: str = "") -> Dict[str, Any]:
        """Mode A back, under the lock. ``unassign_because`` is the one exception.

        Empty (every caller but `_stop_then_leave`): the previous lease is
        restored. A reason: ``ai-chat`` is cleared instead and the reason is
        reported as ``reason``, because restoring it would have the
        failure-watch relaunch llama.cpp beside whatever the reason names. The
        mode file is written back to Mode A either way - Mode B with nothing
        deployed is a box that never comes back.
        """
        state = self.store.read()
        if state.mode != MODE_CLUSTER:
            return {"mode": MODE_SINGLE, "restored": False, "reason": "not-clustering"}
        # Symmetry with `enter`: the LAN gate leaves the cluster before anything
        # moves. It is fronting the cluster's port, which is either already dead
        # or about to be, and leaving it there would publish a 502 on the LAN for
        # the length of the failure-watch's next pass. When AI Chat goes back to
        # a model the watch relaunches (W4-D8), the gate moves onto the door that
        # answers "loading" (ACC-193's door, the way back) and the relaunch puts
        # it in front of llama.cpp once it serves - 11434 never refuses in
        # between. With nothing to come back to it is stopped: "loading" would
        # never end. And the replica balancer with it, under this same lock
        # (VD-129): a stop made before the lock - `remove` stops it beside the
        # units - can be overtaken by a healthy pass that started it again.
        # B1: whether the door stays is the WATCH's answer (`gpu_loading_door`),
        # asked after the restore. The lease is restored BEFORE any credential
        # deletion (D9): `remove` deletes the cluster credential next, and the
        # broker's delete cascades its purpose assignments, so restoring
        # afterwards would race a window in which ai-chat points at nothing.
        # VD-210 item 4: the previous connection comes back only while AI Chat
        # is still on the cluster; a connection the owner chose stays.
        kept = leave_keeps_owner_choice(self, state)
        restored, reason = leave_door(
            self, state.deployment_name,
            kept is None and not unassign_because and str(
                state.previous_ai_chat_credential_id or "").startswith(MANAGED_LOCAL_PREFIX),
            lambda: self._restore_ai_chat(
                state.previous_ai_chat_credential_id, unassign_because,
            ) if kept is None else keep_owner_choice(self, kept, unassign_because),
        )
        # The LLM Server flag is NOT written here. It was never changed on the
        # way in, so it already says what the owner last chose - including a
        # Disable or an Enable made while clustered - and the keys stayed in the
        # broker throughout, so every external client survives a Mode B cycle.
        # Re-enabling from the entry-time snapshot overrode a Disable made in
        # Mode B, and minted an unrevealed key when the set was empty.
        self.store.write(ClusterModeState())
        return {
            "mode": MODE_SINGLE,
            "restored": restored,
            "reason": reason,
            "credential_id": state.previous_ai_chat_credential_id,
            "llm_server_enabled": bool(self.llm_server_store.read().enabled),
        }

    def _restore_ai_chat(self, credential_id: str, unassign_because: str = "") -> tuple:
        """Put ``ai-chat`` back on what it held before, or say why it could not."""
        if unassign_because or not credential_id:
            self._quietly(
                lambda: self.broker.deactivate(AI_CHAT_PURPOSE),
                CLEARING_AI_CHAT,
            )
            return False, unassign_because or "no-previous-connection"
        try:
            self.broker.activate(credential_id, AI_CHAT_PURPOSE)
        except CredentialError:
            # The model the box was serving before clustering has since been
            # removed. Clearing the lease is the honest end state: AI Chat asks
            # for a connection instead of pointing at a credential that is gone.
            self._quietly(
                lambda: self.broker.deactivate(AI_CHAT_PURPOSE),
                CLEARING_AI_CHAT,
            )
            return False, "previous-connection-missing"
        return True, ""

    # -- Boot and steady state -------------------------------------------

    def reconcile(
        self,
        deployment: Optional[Mapping[str, Any]],
        *,
        deploy_job_active: Callable[[], bool],
        stop_deploy_units: Callable[[], Optional[List[Dict[str, str]]]],
    ) -> Dict[str, Any]:
        """Own the mode file at boot and on every failure-watch pass.

        Five cases, and nothing else:

        * **Mode A** - no-op. The GPU failure-watch owns the managed-local model.
        * **Mode B claimed by a deploy, with the deploy job running here** -
          no-op, logged once. The flag or a ``deploying`` record says a deploy
          owns the switch, and the whole weights fetch happens inside that
          window; acting here is what tore a live Mode B down.
        * **Mode B claimed by a deploy, with NO deploy job running here** - the
          deploy died with its executor; :meth:`_stop_then_leave`, logged at
          WARNING with the deployment's name, answered :data:`DEPLOY_ABANDONED`.
        * **Mode B with a healthy record whose credential still exists** - make
          sure ``ai-chat`` is on it WHILE AI CHAT FOLLOWS THE CLUSTER, and that
          the proxy is converged. The cluster's units are
          ``WantedBy=multi-user.target`` and come back on their own after a
          reboot; an AI Chat left on nothing is put back on them. A connection
          the owner chose in Mode B is never moved (VD-210,
          `gpu_cluster_ai_chat`): this pass once undid it every 30 s.
        * **Mode B with no record, a failed one, or a vanished credential** -
          :meth:`_stop_then_leave`, answered :data:`NO_HEALTHY_DEPLOYMENT` or
          :data:`CLUSTER_CREDENTIAL_MISSING`. A cluster that is not there must
          not hold AI Chat hostage; Mode A comes back within one further pass.
          Every leave answers with the deployment's name and its reason, so
          the caller holding the cluster store can mark the row ``failed``.

        ``deployment`` is the stored record for ``deployment_name``, supplied by
        the caller that can read the cluster store (the executor's
        `ClusterOperations`). ``None`` means "no such record", which is the
        give-up case. ``deploy_job_active`` answers whether a
        ``cluster.llm.deploy`` job is executing in this process right now - the
        executor service supplies it from its own job loop, and it is required
        rather than defaulted because a caller that forgot it would tear down
        every deploy in flight. ``stop_deploy_units`` stops whatever the
        record derives on the nodes and answers what would not stop (the
        `gpu_pool_units.unstopped` shape, or nothing), supplied by the same
        caller (only it can reach a node; this switch holds no transport) and
        required for the mirror-image reason: a caller that forgot it would
        restore Mode A over a vLLM server still loading into, or serving from,
        the GPU. Never raises: it runs on a daemon thread beside the job loop,
        and one bad pass must not stop the next.
        """
        try:
            with self.watch_lock:
                return self._reconcile_locked(
                    deployment, deploy_job_active, stop_deploy_units
                )
        except Exception as error:  # noqa: BLE001 - a watch pass may never raise
            LOGGER.warning("The GPU cluster mode reconcile did not finish: %s", error)
            return {"mode": "", "reconciled": False, "reason": "error"}

    def _reconcile_locked(
        self,
        deployment: Optional[Mapping[str, Any]],
        deploy_job_active: Callable[[], bool],
        stop_deploy_units: Callable[[], None],
    ) -> Dict[str, Any]:
        state = self.store.read()
        if state.mode != MODE_CLUSTER:
            self._in_flight_logged = False
            return {"mode": MODE_SINGLE, "reconciled": False, "reason": "mode-a"}
        record_state = str((deployment or {}).get("state", ""))
        if state.deploy_in_flight or record_state == DEPLOYING_STATE:
            if deploy_job_active():
                return self._in_flight_pass()
            return self._abandoned_pass(state, deployment, stop_deploy_units)
        self._in_flight_logged = False
        state = settle_unload_cause(self, state, unloaded=record_state == UNLOADED_STATE)
        if record_state == UNLOADED_STATE:
            return self._unloaded_pass(state)
        if record_state != HEALTHY_STATE:
            return self._stop_then_leave(
                state, deployment, stop_deploy_units, NO_HEALTHY_DEPLOYMENT
            )
        if not state.cluster_credential_id or not self._credential_exists(
            state.cluster_credential_id
        ):
            return self._stop_then_leave(
                state, deployment, stop_deploy_units, CLUSTER_CREDENTIAL_MISSING
            )
        # VD-210: only while AI Chat follows the cluster - never over the owner's choice.
        moved = heal_onto_cluster(self, state)
        # The REPAIR: to the current flag and key set on every healthy pass, so
        # a gate stopped behind the switch's back, or one carrying a key set
        # the broker no longer holds, is right again within one 30 s tick.
        converged = self._converge_proxy(state)
        return {
            "mode": MODE_CLUSTER,
            "reconciled": True,
            "reason": "",
            "ai_chat_moved": moved,
            "llm_server_converged": converged,
            "balancer_converged": self._converge_balancer(state, deployment),
        }

    def _in_flight_pass(self) -> Dict[str, Any]:
        """Wait for the deploy job that owns the switch, logged ONCE per window.

        A weights fetch is minutes to hours long and this pass runs every 30 s,
        so a line per pass would bury the log in the one situation an operator
        most needs to read it.
        """
        if not self._in_flight_logged:
            LOGGER.info(
                "A GPU cluster deploy owns the serving mode switch; leaving "
                "the mode alone until it finishes."
            )
            self._in_flight_logged = True
        return {
            "mode": MODE_CLUSTER, "reconciled": False,
            "reason": DEPLOY_IN_FLIGHT,
        }

    def _unloaded_pass(
        self, state: ClusterModeState, *, allow_wake: bool = True,
        stop_balancer: bool = True,
    ) -> Dict[str, Any]:
        """An unloaded deployment (G3a/G3b): keep Mode B, and keep the LLM Server's port open.

        The serving units are stopped and the record reads ``unloaded`` while
        the mode file still names it, so Mode B is kept and ai-chat is left as
        unload parked it. The replica balancer fronts a dead port and comes
        down every pass. The LLM Server's door stays up and keyed (VD-159);
        what it fronts depends on whether a request can load the model:

        * **idle, and the wake responder answers** (ACC-058): the door fronts
          the responder, so a keyed client's request wakes the model and is
          told to retry (`_converge_wake_door`);
        * **every other unloaded state** - a manual unload, an unreadable
          cause, a responder that is not up (the state straight after a
          reboot): the door fronts nothing and answers a keyed request with a
          plain "not loaded" 503 itself (`_converge_unloaded_door`);
        * **a door that cannot be converged** (the key set unreadable, the
          bridge refusing): a gate still in front of the stopped MODEL comes
          down, since it answers the LAN with an nginx 502; a wake door or
          an unloaded door already standing is left as it is, so a key store
          that is down does not close the port (`_gate_fronts_the_model`).

        ``allow_wake`` False never opens the wake door, and ``stop_balancer``
        False leaves the balancer running: the unload's first move, made
        before its units are stopped, when a wake would load the model it is
        stopping and the balancer still fronts replicas that may yet stay up
        (an unload that cannot stop them puts the row back to healthy).
        ``llm_server_converged`` says whether the door
        is where this pass wants it. All stops are idempotent. The reason is
        :data:`DEPLOYMENT_UNLOADED`, kept out of
        `gpu_cluster_mode_watch.LEAVE_REASON_NOTES` so the row is never
        marked ``failed``.
        """
        converged = bool(
            (allow_wake and self._converge_wake_door(state))
            or self._converge_unloaded_door()
        )
        if not converged and self._gate_fronts_the_model():
            self._quietly(lambda: self.llm_proxy.stop(), STOPPING_THE_PROXY)
        if stop_balancer:
            self._quietly(
                lambda: self.balancer.stop(state.deployment_name),
                STOPPING_THE_BALANCER,
            )
        return {
            "mode": MODE_CLUSTER, "reconciled": True,
            "reason": DEPLOYMENT_UNLOADED, "llm_server_converged": converged,
        }

    def _abandoned_pass(
        self, state: ClusterModeState, deployment: Optional[Mapping[str, Any]],
        stop_deploy_units: Callable[[], None],
    ) -> Dict[str, Any]:
        """Recover a claim whose deploy job is gone: the executor died under it.

        Logged at WARNING every time it fires, naming the deployment, because
        it tears down a recorded Mode B and that is never routine. The
        teardown itself is :meth:`_stop_then_leave`, like every other leave.
        """
        LOGGER.warning(
            "The GPU cluster deploy '%s' holds the serving mode switch but no "
            "deploy job is running in this executor, so it died before it "
            "finished; stopping what it started and returning the appliance "
            "to Mode A.",
            state.deployment_name,
        )
        self._in_flight_logged = False
        return self._stop_then_leave(
            state, deployment, stop_deploy_units, DEPLOY_ABANDONED
        )

    def _stop_then_leave(
        self, state: ClusterModeState, deployment: Optional[Mapping[str, Any]],
        stop_deploy_units: Callable[[], None], reason: str,
    ) -> Dict[str, Any]:
        """Tear a recorded Mode B down: the record's units stopped FIRST, then Mode A.

        The one method every leave the reconcile makes goes through, so the
        stop-then-restore order (D9) cannot be kept by one path and skipped by
        another - which is what happened while the stop was wired into the
        abandoned pass alone. The stop runs under the serving lock BEFORE the
        lease is restored, so the failure-watch cannot relaunch llama.cpp into
        an aperture a vLLM server is loading into or serving from; through
        `_quietly`, because a stop that raised must not hold the box in Mode B
        for ever - a stranded AI Chat is the worse failure. With NO record
        there is nothing to stop: the units derive from the record's name and
        node list, so the leave happens and the stop is not asked for.

        **The one exception to "always restore" (VD-127).** What the stop
        could not stop comes back by node and unit. On a worker that is
        harmless to the restore - llama.cpp comes back on this controller's
        GPU, not that one's. On the CONTROLLER it is the double residency this
        whole switch exists to prevent, so the lease is NOT restored: Mode A
        is written (the mode must not stay B with nothing deployed), the
        cluster credential goes as it does today, and ``ai-chat`` is left
        unassigned under :data:`CONTROLLER_UNIT_UNSTOPPED` for the operator to
        resolve; the watch writes which unit, on the row.
        """
        not_stopped: List[Dict[str, str]] = []
        if deployment is not None:
            self._quietly(
                lambda: not_stopped.extend(stop_deploy_units() or []),
                "stop what the deployment started",
            )
        held = CONTROLLER_UNIT_UNSTOPPED if controller_unstopped(not_stopped) else ""
        return {
            **self._leave_locked(held),
            "reconciled": True,
            "reason": reason,
            "deployment_name": state.deployment_name,
        }

    # -- Shared broker reads ---------------------------------------------

    def _credential_exists(self, credential_id: str) -> bool:
        try:
            return any(
                str(item.get("id", "")) == str(credential_id)
                for item in self.broker.list()
            )
        except Exception:  # noqa: BLE001 - an unreadable broker is not proof
            # Believing "gone" on an unreadable broker would tear a working
            # cluster down, so an unanswered question keeps the current mode.
            return True

    @staticmethod
    def _quietly(action: Callable[[], Any], what: str) -> None:
        """Run one restore step, logging rather than aborting the rest.

        A switch step that fails must not skip the steps after it: leaving the
        proxy up is bad, but leaving the proxy up AND never restoring AI Chat is
        worse. Each failure is logged with what it was, so an operator has the
        fact rather than a silent partial switch (LESSONS pattern 1).

        ``action`` is always a thunk - ``lambda: self.llm_proxy.stop()``, never
        the bound method ``self.llm_proxy.stop`` - so the ATTRIBUTE LOOKUP
        happens inside the guard. The switch's first live run passed a bound
        method of a controller that had no such method: the ``AttributeError``
        was raised while building the argument, before this method could catch
        anything, and escaped the executor. `tests/test_gpu_cluster_mode_collaborators.py`
        refuses a call here whose action is not a lambda.
        """
        try:
            action()
        except Exception as error:  # noqa: BLE001 - never mask the cause
            LOGGER.warning("The GPU mode switch could not %s: %s", what, error)
