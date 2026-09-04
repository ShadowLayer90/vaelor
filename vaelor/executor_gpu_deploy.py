"""Deploy and boot-reconcile the GPU AI-Chat tier (the ROCmFPX host server).

Split out of :mod:`vaelor.executor_model_deploy` because that module is already
near the 1,000-line ceiling `CLAUDE.md` sets, and the GPU deploy is a distinct
job with its own supervisor, its own credential surface and its own boot
reconcile. The two are composed into the same executor class, so this mixin
reads the executor's helpers (``_model_file``, ``_checkpoint``,
``_model_hardware_budget``, ``credential_broker``, ``models_root``) exactly as
:class:`vaelor.executor_model_deploy.ExecutorModelDeployMixin` does.

**The GPU tier is independent of the NPU Assistant, and that independence is the
whole design (VD-085 lineage).** A dual-accelerator box (the Z2) runs the NPU
Assistant on ``deployment-agent`` and the GPU AI-Chat model on ``ai-chat``,
served by two different processes on two different loopback ports. So this path:

* routes only the engine-gated ROCmFP4 model here (the router in
  ``_deploy_model`` keys on :func:`vaelor.model_catalog.catalog_engine`);
* registers the GPU endpoint and activates it for AI Chat through
  :func:`vaelor.managed_local_credentials.activate_managed_chat`, which claims
  ``ai-chat`` and deliberately never touches ``deployment-agent``;
* never retires the NPU assistant container - that is a different tier;
* reads GPU memory before and after the load, because the silent-CPU-fallback
  trap is real: a fork that cannot resolve its TheRock libraries falls back to
  the CPU and still answers HTTP 200, so a healthy endpoint is not proof of GPU
  residency and the deploy result says which was actually observed.
"""

from __future__ import annotations

import hashlib
import json
import logging
import time
from pathlib import Path
from typing import Any, Callable, Dict, Optional

from .accelerator_runtime import BASIS_DIFFERENTIAL, MINIMUM_ACCELERATED_BYTES
from .credential_broker import CredentialBrokerClient, CredentialError
from .gpu_rocm_supervisor import GpuBridgeLauncher, GpuRocmSupervisor
from .managed_local_credentials import (
    PREFIX,
    _endpoint_of,
    activate_managed_chat,
)
from .model_catalog import (
    catalog_engine,
    catalog_models,
    catalog_runtime,
)
from .model_start_verification import accelerator_baseline
from .executor_model_deploy import (
    CONNECTION_TEST_FAILED_DETAIL,
    GPU_CHAT_COMPOSE_PROJECT,
    MANAGED_LOCAL_CREDENTIAL_LABEL,
    _artifact_identity,
    _loopback_port,
)


logger = logging.getLogger(__name__)


#: The engine value the ROCmFP4 catalog entry carries, and the value the router
#: matches to send a deploy down the GPU path. Named once here so the deploy
#: router and the boot reconcile agree by construction; the string itself is
#: :mod:`vaelor.model_catalog`'s data (``catalog_engine`` returns it), not a
#: sentence this module owns, so it is a bare identifier rather than a message.
GPU_CHAT_ENGINE = "rocmfpx"


#: How much of the model's on-disk size the GPU memory must be seen to grow by
#: for the load to count as GPU-resident rather than a silent CPU fallback.
#:
#: The failing arm of the fallback holds **zero** extra accelerator memory (the
#: HIP backend never loaded), and the working arm holds a model's worth, so this
#: fraction only has to sit clear of an idle adapter's noise while staying well
#: below a genuine load. Half is generous in both directions: the ROCmFP4 27B is
#: ~13.6 GiB of weights before its KV cache, so a real load rises by many
#: gigabytes, while a CPU fallback rises by essentially nothing. It is floored at
#: :data:`MINIMUM_ACCELERATED_BYTES` so that if the model size could not be read
#: the check still demands more than a display buffer's worth before it will
#: call the load resident.
GPU_RESIDENCY_MIN_FRACTION = 0.5

#: What ``basis`` a residency verdict carries when GPU memory could not be read
#: at all - neither the before nor the after reading was available. Distinct from
#: :data:`~vaelor.accelerator_runtime.BASIS_DIFFERENTIAL`: an unread adapter says
#: nothing about residency, and a caller must not render it as either outcome.
BASIS_UNREAD = "unread"


def gpu_residency_verdict(
    before_bytes: Optional[int], after_bytes: Optional[int], model_bytes: int
) -> Dict[str, Any]:
    """Whether the GPU actually took the model, from a before/after memory delta.

    ``before_bytes`` is read with no managed GPU model of ours resident and
    ``after_bytes`` once the server is healthy; the difference is this load's own
    allocation, the same differential :mod:`vaelor.model_start_verification`
    takes for the compose path. Either reading being ``None`` (an adapter that
    reports nothing) yields an honest "not verified" rather than a guess in
    either direction - the load-bearing distinction the accelerator check exists
    to preserve.

    Returns a block the deploy result carries verbatim, so the owner is told the
    backend/residency that was OBSERVED, never one that was assumed from a 200.
    """
    if before_bytes is None or after_bytes is None:
        return {
            "basis": BASIS_UNREAD,
            "resident": None,
            "vram_delta_bytes": None,
            "detail": (
                "GPU memory could not be read, so residency was not verified; "
                "the server answered but the backend it used is unconfirmed."
            ),
        }
    delta = int(after_bytes) - int(before_bytes)
    threshold = max(
        int(MINIMUM_ACCELERATED_BYTES),
        int(max(0, model_bytes) * GPU_RESIDENCY_MIN_FRACTION),
    )
    resident = delta >= threshold
    if resident:
        detail = "The model is resident on the GPU."
    else:
        detail = (
            "The GPU server answered, but accelerator memory did not rise by a "
            "model's worth ({} bytes, expected at least {}); it may have fallen "
            "back to the CPU. Check the ROCmFPX libraries.".format(delta, threshold)
        )
    return {
        "basis": BASIS_DIFFERENTIAL,
        "resident": resident,
        "vram_delta_bytes": delta,
        "detail": detail,
    }


def _gpu_chat_catalog_entry() -> Optional[Dict[str, Any]]:
    """The one catalog entry the GPU AI-Chat tier serves, or ``None``.

    Found by the same engine key the router uses rather than by a hard-coded id,
    so the tier follows whatever entry :mod:`vaelor.model_catalog` marks as the
    ROCmFPX model. ``None`` on a box whose catalog has no such entry, which is
    every non-GPU build.
    """
    for entry in catalog_models():
        if catalog_engine(entry["repo"], entry["file"]) == GPU_CHAT_ENGINE:
            return entry
    return None


class ExecutorGpuDeployMixin:
    """Route the ROCmFP4 model to the host-run GPU server and keep it running."""

    def _gpu_rocm_supervisor(self) -> GpuRocmSupervisor:
        """The GPU supervisor, launching through the root hardware bridge.

        A lazily-cached seam mirroring :meth:`_flm_supervisor`, injectable for
        tests. Production wraps a :class:`vaelor.gpu_rocm_supervisor.GpuBridgeLauncher`
        over the same :class:`vaelor.hardware_bridge.HardwareBridgeClient` the NPU
        deploy uses, because the GPU server must be launched by the root bridge —
        the only account with /dev/dri + /dev/kfd, a writable ``/var/log/vaelor``
        and no ``MemoryDenyWriteExecute``. The sandboxed workload executor has
        none of them (``PrivateDevices=true`` hides the GPU and its log directory
        is read-only), so a direct executor-side launch could never reach the GPU.
        The health probe and port helper stay the supervisor's own defaults -
        health via :func:`vaelor.flm_supervisor._endpoint_healthy`, port via
        :func:`vaelor.executor_network.available_model_port` - because those run
        executor-side (a loopback probe, a local port pick), not across the bridge.
        The instance is cached so a relaunch reuses the same launcher and client.
        """
        from .hardware_bridge import HardwareBridgeClient

        existing = getattr(self, "_gpu_supervisor_cache", None)
        if existing is not None:
            return existing
        supervisor = GpuRocmSupervisor(GpuBridgeLauncher(HardwareBridgeClient()))
        self._gpu_supervisor_cache = supervisor
        return supervisor

    def _gpu_chat_model_path(self) -> Optional[Path]:
        """Where the GPU AI-Chat model is on disk, if it is downloaded.

        Derived from the catalog entry through the managed-download convention
        ``models_root / repo.replace("/", "--") / file`` - the same layout
        :func:`vaelor.executor_model_deploy._artifact_identity` reverses - so the
        boot reconcile finds the model without a stored path. ``None`` when the
        entry is absent (a non-GPU catalog) or the file is not present (a box that
        never downloaded it, a Pi): both are the "nothing to serve" answer, and
        returning ``None`` keeps the reconcile from launching a server against a
        file that is not there.
        """
        entry = _gpu_chat_catalog_entry()
        if entry is None:
            return None
        directory = str(entry["repo"]).replace("/", "--")
        path = (self.models_root / directory / str(entry["file"])).resolve()
        return path if path.is_file() else None

    def _retire_gpu_chat_except(self, keep: str) -> None:
        """Retire every GPU AI-Chat server EXCEPT the one about to be deployed.

        #247m Finding 1: the stock-GGUF GPU chat compose server (the
        ``model-chat`` project) and the fork 27B host server run under DIFFERENT
        mechanisms with no shared lock, so without this they would both hold
        GPU/GTT memory and OOM the box (a 22 GiB 27B beside a 17 GiB MoE exceeds
        the ~32 GiB budget). Deploying a new GPU chat model is a switch: whatever
        else was the GPU chat model is retired, so exactly one runs. ``keep`` is
        the mechanism being (re)deployed - ``"fork"`` for the ROCmFP4 supervisor,
        ``"compose"`` for the stock ``model-chat`` project - and only the OTHER
        is retired.

        **Only GPU CHAT servers, never the Assistant.** It touches the fork
        supervisor and the ``model-chat`` compose project. The Assistant's own
        server - flm-real on the NPU, or llama.cpp in ``model-assistant`` - holds
        ``deployment-agent``, not ``ai-chat``, and is never named here. Both
        teardowns are best-effort: a redeploy must not fail because retiring the
        other server hiccuped (the residency check after start catches a real
        conflict), and in a harness without a compose seam the ``compose down``
        simply no-ops.
        """
        if keep != "fork":
            # Stop the fork 27B host server (idempotent: a no-op when none runs).
            try:
                self._gpu_rocm_supervisor().stop()
            except Exception:
                pass
        if keep != "compose":
            # `compose down` the stock GPU chat project, if it was ever deployed,
            # AND DELETE its compose file. Deleting is what closes the a81 boot-
            # reconcile hijack: the GPU chat port is per-deploy (the lowest free
            # 8080-8100 for BOTH mechanisms), so a `model-chat` file left on disk
            # can later bind the SAME port the fork's ai-chat lease ends up on -
            # and `_gpu_chat_compose_project_for` reads the FILE, not a running
            # container, so at boot it would divert the fork's reconcile to a
            # dead stock server and strand AI Chat. Switching ai-chat to the fork
            # makes any stock compose obsolete (it is regenerated on the next
            # stock deploy), so the file is removed and no stale file can match a
            # fork lease port. Best-effort: neither the down nor the delete may
            # brick the fork deploy.
            try:
                project = self._project_directory(GPU_CHAT_COMPOSE_PROJECT)
            except Exception:
                project = None
            if project is not None and (project / "compose.yaml").is_file():
                try:
                    self._compose(project, "down", "--remove-orphans", timeout=120)
                except Exception:
                    pass
                try:
                    (project / "compose.yaml").unlink(missing_ok=True)
                    project.rmdir()  # only when now empty; OSError otherwise
                except OSError:
                    pass

    def _gpu_chat_compose_project_for(self, port: int) -> Optional[Path]:
        """The stock ``model-chat`` project IFF it is the ai-chat server on ``port``.

        #247m Finding 2: after a user switches ai-chat from the fork 27B to a
        stock GGUF, the boot reconcile must bring back the COMPOSE server, not
        relaunch the fork under the stock model's lease. The discriminator is the
        port: the stock server's compose file binds ``127.0.0.1:<port>:8080`` and
        the fork runs on a different port. Retirement leaves the compose file on
        disk after a switch back to the fork, but bearing the OLD port, so a
        stale file never matches the fork's current lease port. Returns the
        project when this port IS the stock compose server, else ``None`` (the
        fork, or no compose seam in a harness).
        """
        try:
            project = self._project_directory(GPU_CHAT_COMPOSE_PROJECT)
        except Exception:
            return None
        compose_file = project / "compose.yaml"
        try:
            if not compose_file.is_file():
                return None
            text = compose_file.read_text(encoding="utf-8")
        except OSError:
            return None
        return project if "127.0.0.1:{}:8080".format(int(port)) in text else None

    def _deploy_gpu_rocm_chat(
        self, payload: Dict[str, Any], model: Path
    ) -> Dict[str, Any]:
        """Serve the ROCmFP4 model on the GPU and make it the AI-Chat model.

        The GPU counterpart of the llama.cpp compose path and the NPU flm-real
        path. Its rollback discipline is theirs: the server is launched and
        proven healthy before any credential is written, and a failure at any
        later step stops the server and removes the half-written credential, so a
        failed deploy leaves no orphan and never disturbs the NPU Assistant.

        The order of the two accelerator readings is the fail-safe for the
        silent-CPU-fallback trap: ``before`` is taken with no GPU model of ours
        resident, ``after`` once the endpoint answers, and the delta is what the
        result reports as observed residency.
        """
        runtime = catalog_runtime(**_artifact_identity(model))
        supervisor = self._gpu_rocm_supervisor()
        port = supervisor.allocate_port(int(payload.get("port") or 0))
        self._checkpoint(45, "Starting the GPU model server", "starting")
        # Stop any prior GPU model this executor is still supervising BEFORE the
        # baseline read, so ``before_bytes`` reflects no managed GPU model
        # resident. Without this, a same-process redeploy reads the outgoing 27B's
        # ~13 GB into the baseline; serve then frees it and loads the new ~13 GB,
        # the delta is ~0, and a healthy load reports a spurious CPU fallback.
        # This mirrors the compose path stopping the previous container before its
        # baseline (`_accelerator_baseline(replacing=True)`), and it accepts the
        # same tradeoff - the old model is stopped before the new one is proven -
        # which is consistent with that path. A no-op on a first deploy.
        supervisor.stop()
        # #247m Finding 1: retire the OTHER GPU chat mechanism - a stock
        # `model-chat` compose server - before the fork loads its ~13 GB, so the
        # two do not both hold GPU memory and OOM the box. `keep="fork"` leaves
        # this supervisor alone (it was just stopped above and is about to serve)
        # and never touches the Assistant.
        self._retire_gpu_chat_except("fork")
        # Read GPU memory now, with nothing of ours resident, so the after-load
        # delta below is this model's own allocation.
        before_bytes = accelerator_baseline(
            self._model_hardware_budget().get("accelerators")
        )
        # If the GPU server never answers, `serve` stops it and raises a
        # RuntimeError here - before any credential is written, so a timeout is a
        # clean failure that mutated nothing (no compose file, no credential).
        served = supervisor.serve(
            str(model), port=port, runtime=runtime,
            announce=lambda detail: self._checkpoint(75, detail, "starting"),
        )
        endpoint = served["endpoint"]
        candidate_id = "cred_managed_local_{}".format(
            hashlib.sha256(
                "gpu:{}:{}:{}".format(model, port, time.time_ns()).encode()
            ).hexdigest()[:12]
        )
        broker = CredentialBrokerClient()
        profile = json.dumps(
            {"base_url": endpoint, "model": "", "api_key": ""},
            separators=(",", ":"),
        )
        # Rollback discipline, and it must cover EVERY step after the server is up
        # - the residency read (a hardware probe that can raise) and all three
        # broker calls (which cross `CredentialBrokerClient._request` and can raise
        # json/unicode decode errors, not just `CredentialError`, when the broker
        # is bounced mid-request). A type-narrowed `except CredentialError` would
        # let those strand a running ~13 GB llama-server. So the guard is
        # `except Exception` and, on ANY failure, it stops the GPU server and
        # deletes the credential IF it was actually created, then re-raises. Not
        # BaseException: a KeyboardInterrupt/SystemExit must still stop the process.
        credential_created = False
        try:
            residency = self._gpu_residency_after_load(before_bytes, model)
            if residency["resident"] is False:
                # Surfaced prominently rather than hard-failing: a CPU fallback
                # still answers, and the owner needs to be told the tier is
                # degraded, not left with a green deploy over a slow server.
                logger.warning("GPU AI-Chat residency check: %s", residency["detail"])
            credential = broker.put(
                "openai-compatible",
                MANAGED_LOCAL_CREDENTIAL_LABEL.format(model.stem[:48]),
                profile,
                credential_id=candidate_id,
            )
            credential_created = True
            connection_test = broker.test(credential["id"])
            if not connection_test.get("ok"):
                raise RuntimeError(
                    "The GPU model started, but its chat API failed: {}".format(
                        connection_test.get("message", CONNECTION_TEST_FAILED_DETAIL)
                    )
                )
            # activate_managed_chat, NOT activate_managed_local: this claims
            # ai-chat and deliberately leaves deployment-agent on the NPU
            # Assistant, so the two accelerator tiers stay independent.
            ai_chat_active = activate_managed_chat(broker, credential["id"])
        except Exception:
            # Both cleanups are attempted independently so one failing does not
            # skip the other, and the ORIGINAL error is what re-raises.
            try:
                supervisor.stop()
            except Exception:
                pass
            if credential_created:
                try:
                    CredentialBrokerClient().delete(candidate_id)
                except Exception:
                    pass
            raise
        self._checkpoint(95, "GPU model server is healthy", "starting")
        return {
            "path": str(model),
            "endpoint": endpoint,
            "health": served["health_url"],
            "port": port,
            "context": int(runtime.get("context") or 0),
            "backend": GPU_CHAT_ENGINE,
            "device": "gpu",
            "runtime": "llama.cpp-rocmfpx",
            # This tier is AI Chat, not the Assistant: it does not hold
            # deployment-agent and does not claim to be the Assistant's server.
            "ai_chat_active": ai_chat_active,
            "assistant_active": False,
            # Observed residency, so the result never asserts a GPU it did not
            # confirm - both `acceleration` (the field the compose path fills)
            # and the fuller `gpu_residency` block carry it.
            "acceleration": residency,
            "gpu_residency": residency,
            "credential_id": credential["id"],
        }

    def _gpu_residency_after_load(
        self, before_bytes: Optional[int], model: Path
    ) -> Dict[str, Any]:
        """The residency verdict, reading GPU memory now the server is up.

        Kept tiny and behind the pure :func:`gpu_residency_verdict` so the
        threshold arithmetic is unit-testable without a GPU or a live server.
        """
        after_bytes = accelerator_baseline(
            self._model_hardware_budget().get("accelerators")
        )
        try:
            model_bytes = model.stat().st_size
        except OSError:
            # An unreadable size falls the threshold back to the accelerated-bytes
            # floor rather than to zero, so the check stays meaningful.
            model_bytes = 0
        return gpu_residency_verdict(before_bytes, after_bytes, model_bytes)

    def ensure_gpu_chat_served(
        self, *, sleep: Callable[[float], None] = time.sleep
    ) -> Optional[Dict[str, Any]]:
        """Relaunch the GPU AI-Chat server after a reboot, on its pinned port.

        The GPU counterpart of :meth:`ensure_npu_assistant_served`, run OFF the
        job loop's critical path for the same reason: a 27B FP4 load is
        health-gated for minutes, and blocking the executor on that would stall
        the first job after a reboot. The GPU server is a plain host process; a
        reboot kills it and nothing else brings it back, so the AI-Chat tier is
        DOWN after every reboot until this reconcile - or a manual deploy - runs.

        Idempotent, gated, and NON-FATAL, the same contract as the NPU reconcile:

        * Gated - it acts only when ``ai-chat`` is an independent managed-local
          GPU tier. A hosted ai-chat (no loopback), or a single-model box where
          ``ai-chat`` and ``deployment-agent`` share one endpoint (a Pi, whose
          NPU/llama.cpp reconcile owns that server), both no-op. The
          endpoint-distinctness test is the same key
          :mod:`vaelor.managed_local_credentials` uses.
        * Idempotent - a server already answering on the stored port is left
          untouched (relaunching would evict a loaded 27B to no purpose).
        * Non-fatal - nothing escapes; any failure returns ``None`` so the
          reconcile can never keep the executor from processing jobs.

        ``sleep`` is accepted for interface parity with
        :meth:`ensure_npu_assistant_served`. Unlike the NPU path there is no
        privileged bridge and so no socket-readiness race to ride out with a
        retry loop, so the parameter is currently unused; it is kept so the two
        boot reconciles present one signature to the autostart wiring.
        """
        _ = sleep
        try:
            broker = self.credential_broker
            try:
                chat = broker.resolve_active("ai-chat")
            except CredentialError:
                # No ai-chat assignment: an NPU-only box or a fresh appliance.
                return None
            if not str(chat.get("credential_id", "")).startswith(PREFIX):
                # A hosted/user ai-chat lease: nothing local to run.
                return None
            port = _loopback_port(str(chat.get("base_url") or ""))
            if port is None:
                return None
            # Only the INDEPENDENT GPU tier is supervised here. On a single-model
            # box ai-chat and deployment-agent are the same managed endpoint, and
            # the NPU/llama.cpp reconcile owns it - relaunching a GPU server on
            # that port would evict the Assistant. A distinct endpoint is what
            # makes this the independent GPU model (VD-085 key).
            chat_endpoint = _endpoint_of(chat)
            try:
                agent = broker.resolve_active("deployment-agent")
                if _endpoint_of(agent) == chat_endpoint:
                    return None
            except CredentialError:
                # No deployment-agent at all: ai-chat is the only managed tier,
                # so it is not shared with the Assistant - carry on.
                pass
            supervisor = self._gpu_rocm_supervisor()
            # #247m Finding 2: dispatch by the ACTIVE lease's mechanism, not by
            # assuming the fork. If ai-chat was switched to a stock GGUF, its
            # server is the `model-chat` compose project on this port; bring THAT
            # back rather than relaunching the fork 27B under the stock lease.
            # The fork branch below is byte-for-byte unchanged, so the reference
            # state (the 27B IS ai-chat) reconciles exactly as it did before.
            stock_project = self._gpu_chat_compose_project_for(port)
            if stock_project is not None:
                if supervisor.healthy(port):
                    return {"served": False, "reason": "already-healthy", "port": port}
                self._compose(
                    stock_project, "up", "-d", "--remove-orphans", timeout=180
                )
                return {"served": True, "mechanism": "compose", "port": port}
            model_path = self._gpu_chat_model_path()
            if model_path is None:
                # This box has no GPU chat model on disk (a Pi): nothing to serve.
                return None
            runtime = catalog_runtime(**_artifact_identity(model_path))
            if supervisor.healthy(port):
                # Already answering on the stored port: leave the loaded model be.
                return {"served": False, "reason": "already-healthy", "port": port}
            result = supervisor.restart_if_unhealthy(
                str(model_path), port=port, runtime=runtime,
            )
            return {"served": True, **result}
        except Exception:
            # Never let a boot-time reconcile keep the executor from processing
            # jobs, under ANY error - the same broad, non-BaseException catch the
            # NPU reconcile keeps, for the same reasons (a DB locked at boot, a
            # dict/int access on a malformed lease).
            return None
