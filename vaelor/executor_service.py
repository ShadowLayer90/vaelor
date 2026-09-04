"""Long-running poll loop for the allowlisted workload executor."""

from __future__ import annotations

import threading
import time
from typing import Any, Callable, Dict, Optional

from .executor import JobExecutor
from .job_vocabulary import ACTIVE_JOB_STATES
from .jobs import JobStore
from .model_connection import assistant_model_configured


#: How often the GPU AI-Chat supervisor re-checks the deployed server after its
#: first boot reconcile. Restart-on-BOOT is the executor unit starting the
#: supervisor thread; restart-on-FAILURE is that same thread re-running the
#: idempotent reconcile on this interval, so a ``llama-server`` that crashes
#: while the box is up is brought back without waiting for the next reboot. The
#: GPU server is launched by the root bridge and nothing under it restarts a
#: dead child, so this loop is the only thing that heals a mid-run crash.
#:
#: Thirty seconds sits far below a 27B FP4 reload (minutes) yet far above a
#: single loopback ``/health`` GET, so the poll is negligible on an idle box and
#: a gated no-op on a non-GPU one. Each pass runs to completion before the next
#: sleep (the reconcile blocks while it health-gates a relaunch), so ticks never
#: overlap and two relaunches can never race.
GPU_CHAT_SUPERVISE_INTERVAL_SECONDS = 30.0

#: How often the NPU Assistant supervisor re-runs its reconcile after the first
#: boot pass. The NPU counterpart of :data:`GPU_CHAT_SUPERVISE_INTERVAL_SECONDS`
#: and identical in purpose: restart-on-BOOT is the executor unit starting this
#: thread; restart-on-FAILURE is that same thread re-running the idempotent
#: ``ensure_npu_assistant_served`` on this interval, so an ``flm-real`` that
#: crashes while the box is up is relaunched by the root bridge without waiting
#: for the next reboot. flm-real is a child of the bridge and nothing under it
#: restarts a dead child, so this loop is the only thing that heals a mid-run
#: crash of the Assistant tier.
#:
#: Thirty seconds sits far below an flm-real model reload (the health gate allows
#: up to 90s) yet far above a single loopback ``/health`` GET, so the poll is
#: negligible on an idle box and a gated no-op on a Pi. Each pass runs to
#: completion before the next sleep (the reconcile blocks while it health-gates a
#: relaunch), so ticks never overlap and two relaunches can never race — the
#: bridge also serialises launches under its own lock (VD-001).
NPU_ASSISTANT_SUPERVISE_INTERVAL_SECONDS = 30.0


def launch_web_research_autostart(
    executor: JobExecutor,
    *,
    interval_seconds: float = NPU_ASSISTANT_SUPERVISE_INTERVAL_SECONDS,
    sleep: Callable[[float], None] = time.sleep,
    stop: Optional[threading.Event] = None,
) -> threading.Thread:
    """Auto-provision and keep the guarded search backend up, OFF the job loop.

    A PERIODIC reconcile with two idempotent, non-fatal jobs:

    * **Auto-provision (first boot).** On a clean box
      ``ensure_web_research_started`` installs the digest-pinned SearXNG backend
      once (marker-gated, respecting a later deliberate remove), so custom-app
      research works without the operator discovering and approving a separate
      install - the capability was otherwise hidden behind a failed research
      attempt. Retried each ``interval_seconds`` because the install needs Docker,
      which may not be ready the instant the executor starts.
    * **Restart-on-boot/failure.** Once enabled, a configured-but-down service (a
      reboot, a `compose down`) is brought back the same way.

    Run before the job loop, the first pass would stall the executor's first job:
    a bring-up is a `docker compose up` plus an up-to-90s readiness poll. A daemon
    thread keeps every guarantee while letting `run_once` begin immediately. It
    was a one-shot before; the loop is what lets the first-boot install land once
    Docker becomes ready. ``sleep`` and ``stop`` are injected seams: production
    wires neither and the loop runs forever; a test passes a fake ``sleep`` and a
    :class:`threading.Event` to run a bounded number of passes.
    """
    def _supervise() -> None:
        while stop is None or not stop.is_set():
            executor.ensure_web_research_started()
            if stop is not None and stop.is_set():
                return
            sleep(interval_seconds)

    thread = threading.Thread(
        target=_supervise, name="vaelor-web-research-autostart", daemon=True
    )
    thread.start()
    return thread


def launch_npu_assistant_autostart(
    executor: JobExecutor,
    *,
    interval_seconds: float = NPU_ASSISTANT_SUPERVISE_INTERVAL_SECONDS,
    sleep: Callable[[float], None] = time.sleep,
    stop: Optional[threading.Event] = None,
) -> threading.Thread:
    """Keep the NPU Assistant up OFF the job loop's critical path (VD-001).

    The NPU counterpart of :func:`launch_gpu_chat_autostart`, and the one thing
    that gives the Assistant tier BOTH restart-on-boot AND restart-on-failure:

    * **Restart-on-boot.** flm-real runs as a child of the hardware bridge; a
      reboot or fleet restart kills it and nothing else brings it back, so the
      NPU Assistant is DOWN after every reboot/deploy until a
      ``model.deploy {surface:assistant}`` is re-run by hand - and the
      managed-local credential still points at the now-dead loopback port. The
      first pass relaunches flm-real on the SAME port for the SAME pinned model,
      so the stored credential stays valid with no re-registration.

    * **Restart-on-failure.** The bridge that launches flm-real does not restart
      a dead child, so an flm-real that crashes while the box is up would
      otherwise stay down until the next reboot. This loop re-runs the same
      idempotent reconcile every ``interval_seconds`` (default
      :data:`NPU_ASSISTANT_SUPERVISE_INTERVAL_SECONDS`): a healthy server
      answering on the stored port is left alone (relaunching would evict a
      loaded model to no purpose), a crashed one is relaunched on its pinned port
      for its pinned tag. A control-plane restart therefore RE-ADOPTS a running
      flm-real rather than cycling it.

    Run before the job loop, the first pass would stall the executor's first job
    after a reboot: flm-real's model load is health-gated for up to 90s, and the
    reconcile also absorbs a bridge-socket readiness race with its own bounded
    retry. A daemon thread keeps every guarantee (idempotent, non-fatal, gated on
    the NPU serving decision and an active managed-local Assistant connection)
    while letting ``run_once`` begin immediately. A Pi has no NPU and every pass
    is a gated no-op. ``sleep`` and ``stop`` are injected seams: production wires
    neither and the loop runs forever on real time; a test passes a fake
    ``sleep`` and a :class:`threading.Event` to run a bounded number of passes.
    """
    def _supervise() -> None:
        while stop is None or not stop.is_set():
            # ``ensure_npu_assistant_served`` is idempotent, gated and non-fatal:
            # it returns None on any error, so one bad pass can neither kill this
            # thread nor stop the next. The first pass is the boot reconcile;
            # every later pass is the failure watch.
            executor.ensure_npu_assistant_served()
            if stop is not None and stop.is_set():
                return
            sleep(interval_seconds)

    thread = threading.Thread(
        target=_supervise, name="vaelor-npu-assistant-autostart", daemon=True
    )
    thread.start()
    return thread


def launch_gpu_chat_autostart(
    executor: JobExecutor,
    *,
    interval_seconds: float = GPU_CHAT_SUPERVISE_INTERVAL_SECONDS,
    sleep: Callable[[float], None] = time.sleep,
    stop: Optional[threading.Event] = None,
) -> threading.Thread:
    """Keep the GPU AI-Chat server up OFF the job loop's critical path.

    The GPU counterpart of :func:`launch_npu_assistant_autostart`, and the one
    thing that gives the GPU tier BOTH restart-on-boot AND restart-on-failure:

    * **Restart-on-boot.** The GPU chat server is a plain host process (or a
      compose container); a reboot or fleet restart kills it and nothing else
      brings it back, so the AI-Chat tier is DOWN after every reboot until a
      deploy is re-run by hand - and the managed-local ai-chat credential still
      points at the now-dead loopback port. The first pass relaunches the server
      on the SAME port for the SAME model, so the stored credential stays valid
      with no re-registration. #247m: it dispatches by the active lease's
      MECHANISM - the fork 27B through its supervisor, or a stock GGUF through
      its ``model-chat`` compose project - so a box the owner switched to a stock
      GPU chat model comes back on that model, not the fork.

    * **Restart-on-failure.** The bridge that launches the server does not
      restart a dead child, so a ``llama-server`` that crashes while the box is
      up would otherwise stay down until the next reboot. This loop re-runs the
      same idempotent reconcile every ``interval_seconds`` (default
      :data:`GPU_CHAT_SUPERVISE_INTERVAL_SECONDS`): a healthy server is left
      alone (``restart_if_unhealthy`` evicts nothing), a crashed one is
      relaunched on its pinned port with the resolved recipe. A control-plane
      restart therefore RE-ADOPTS a running 27B rather than cycling it - reloading
      it costs minutes - which is the deliberate "don't cycle a healthy server"
      choice.

    Run before the job loop, the first pass would stall the executor's first job
    after a reboot: the 27B FP4 model load is health-gated for minutes. A daemon
    thread keeps every guarantee (idempotent, non-fatal, gated on an independent
    managed-local GPU ai-chat connection) while letting ``run_once`` begin
    immediately. A Pi or an NPU-only box has no GPU AI-Chat tier and every pass
    is a gated no-op. ``sleep`` and ``stop`` are injected seams: production wires
    neither and the loop runs forever on real time; a test passes a fake
    ``sleep`` and a :class:`threading.Event` to run a bounded number of passes.
    """
    def _supervise() -> None:
        while stop is None or not stop.is_set():
            # ``ensure_gpu_chat_served`` is idempotent, gated and non-fatal: it
            # never raises, so one bad pass can neither kill this thread nor stop
            # the next. The first pass is the boot reconcile; every later pass is
            # the failure watch.
            executor.ensure_gpu_chat_served()
            if stop is not None and stop.is_set():
                return
            sleep(interval_seconds)

    thread = threading.Thread(
        target=_supervise, name="vaelor-gpu-chat-autostart", daemon=True
    )
    thread.start()
    return thread


def assistant_deploy_pending(store: JobStore) -> bool:
    """Whether a ``model.deploy`` for the Assistant is already queued or running.

    The guard that keeps the first-boot auto-enable from stacking a second deploy
    on top of one that is already in flight. A ``model.deploy`` whose payload is
    for the Assistant surface (the default) and whose state is ``queued`` or one
    of :data:`ACTIVE_JOB_STATES` is one the executor will run, so there is nothing
    to enqueue. Non-fatal: a store it cannot read reads as "nothing pending", and
    the exact-payload dedupe in :meth:`JobStore.create` is the backstop.
    """
    try:
        records = store.list(limit=200)
    except (OSError, ValueError):
        return False
    for record in records:
        if record.get("type") != "model.deploy":
            continue
        surface = str((record.get("payload") or {}).get("surface") or "assistant")
        if surface != "assistant":
            continue
        state = str(record.get("state") or "")
        if state == "queued" or state in ACTIVE_JOB_STATES:
            return True
    return False


def first_boot_assistant_autoenable(
    store: JobStore, credential_broker: Any, *, servable: bool, tag: str
) -> Optional[Dict[str, Any]]:
    """Enqueue ONE Assistant ``model.deploy`` on a clean box, or no-op.

    On a clean box nothing runs the initial deploy, so no ``deployment-agent``
    lease exists and the UI shows no Assistant even though the NPU tier is
    servable and the model is on disk. This closes that gap once: when the tier
    is ``servable`` (its caller passes ``discover_npu_serving(...).available``)
    and a launchable ``tag``, the Assistant is not already configured, and no
    Assistant deploy is already in flight, it enqueues a ``model.deploy`` that
    serves+pins the installed model - which pins the ``deployment-agent``
    credential and makes the Assistant appear.

    Idempotent (once configured, ``assistant_model_configured`` stops it; while a
    deploy is in flight, :func:`assistant_deploy_pending` does), and the exact
    logic a test can drive without a full executor.
    """
    if not servable or not tag:
        return None
    if assistant_model_configured(credential_broker):
        return None
    if assistant_deploy_pending(store):
        return None
    return store.create(
        "model.deploy", "system", {"surface": "assistant", "tag": tag}
    )


def run_assistant_autoenable(executor: JobExecutor) -> Optional[Dict[str, Any]]:
    """Compute the NPU serving facts off this box and auto-enable the Assistant.

    Gated (a Pi with no neural accelerator, or an unservable NPU tier, no-ops),
    idempotent, and NON-FATAL: every error returns None so a boot-time reconcile
    can never keep the executor from processing jobs. The NPU serving decision
    is read exactly as the deploy path reads it - the neural-accelerator record
    for the Pi short-circuit, then :func:`discover_npu_serving` for binary +
    device + installed-model presence.
    """
    try:
        from .flm_service import discover_npu_serving
        from .inference_tuning import npu_tier_plan

        hardware = executor._model_hardware_budget()
        neural = hardware.get("neural_accelerators") or []
        if not neural:
            return None
        tag = str(npu_tier_plan().get("flm_tag") or "")
        if not tag:
            return None
        capability = discover_npu_serving(
            tag, device_node=str(neural[0].get("device_node") or "") or None
        )
        return first_boot_assistant_autoenable(
            executor.store,
            executor.credential_broker,
            servable=bool(capability.get("available")),
            tag=tag,
        )
    except Exception:
        # A first-boot reconcile must never raise into the boot path (the same
        # broad, non-fatal contract the autostart reconciles keep): a store
        # locked at boot, a bridge socket race, or a plan-access error is "did
        # not auto-enable this pass", not a crash.
        return None


def launch_assistant_autoenable(
    executor: JobExecutor,
    *,
    interval_seconds: float = NPU_ASSISTANT_SUPERVISE_INTERVAL_SECONDS,
    sleep: Callable[[float], None] = time.sleep,
    stop: Optional[threading.Event] = None,
) -> threading.Thread:
    """Auto-enable the Assistant when its model appears, OFF the job loop's path.

    A PERIODIC daemon reconcile, not a one-shot: on a clean install the NPU model
    is installed by ``deploy/fetch-npu-model.sh`` AFTER the control plane has
    started, so a single startup pass would run before the model exists and then
    no-op forever - leaving the Assistant down until the next reboot even though
    the model is now on disk. Re-running :func:`run_assistant_autoenable` every
    ``interval_seconds`` closes that gap: the first pass after the model lands
    enqueues the serve+pin deploy that makes the Assistant appear, and once it is
    configured (or a deploy is already in flight) every later pass is a guarded
    no-op - the same idempotent, non-fatal contract as the autostart reconciles.

    Blocking the executor's first job on the bridge/broker probes is the stall
    those reconciles avoid, so this runs on its own thread. A Pi/non-NPU box
    no-ops every pass. ``sleep`` and ``stop`` are injected seams: production wires
    neither and the loop runs forever on real time; a test passes a fake
    ``sleep`` and a :class:`threading.Event` to run a bounded number of passes.
    """
    def _supervise() -> None:
        while stop is None or not stop.is_set():
            # Guarded, idempotent and non-fatal: it enqueues the serve+pin deploy
            # only once the model is present and the Assistant is not already
            # configured or mid-deploy, and returns None on any error, so one bad
            # pass can neither kill this thread nor stop the next.
            run_assistant_autoenable(executor)
            if stop is not None and stop.is_set():
                return
            sleep(interval_seconds)

    thread = threading.Thread(
        target=_supervise, name="vaelor-assistant-autoenable", daemon=True
    )
    thread.start()
    return thread


def main() -> None:
    store = JobStore()
    store.recover_interrupted()
    executor = JobExecutor(store)
    launch_web_research_autostart(executor)
    launch_npu_assistant_autostart(executor)
    launch_gpu_chat_autostart(executor)
    launch_assistant_autoenable(executor)
    while True:
        completed = executor.run_once()
        time.sleep(2 if completed is None else 0.2)


if __name__ == "__main__":
    main()
