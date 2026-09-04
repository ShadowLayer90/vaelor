"""Long-running poll loop for the allowlisted workload executor."""

from __future__ import annotations

import threading
import time
from typing import Callable, Optional

from .executor import JobExecutor
from .jobs import JobStore


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


def launch_web_research_autostart(executor: JobExecutor) -> threading.Thread:
    """Bring an enabled-but-down search backend up OFF the job loop's critical
    path (#212, auto-start-on-enable).

    Enabling web research must auto-start its backend, but a genuine
    down-and-enabled recovery is a `docker compose up` followed by an up-to-90s
    readiness poll. Run before the loop, that stalls the executor's first job by
    minutes after a reboot - no jobs are processed until the search service
    converges. A daemon thread keeps every guarantee (idempotent, non-fatal,
    gated on `is_enabled()`, pinned digest and bounds) while letting `run_once`
    begin immediately.
    """
    thread = threading.Thread(
        target=executor.ensure_web_research_started,
        name="vaelor-web-research-autostart",
        daemon=True,
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


def main() -> None:
    store = JobStore()
    store.recover_interrupted()
    executor = JobExecutor(store)
    launch_web_research_autostart(executor)
    launch_npu_assistant_autostart(executor)
    launch_gpu_chat_autostart(executor)
    while True:
        completed = executor.run_once()
        time.sleep(2 if completed is None else 0.2)


if __name__ == "__main__":
    main()
