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

**The gate here used to be written twice more (VD-125).** The failure-watch, the
LLM Server apply and ``api_llm_server_routes`` each spelled "is ai-chat an
independent managed-local GPU tier", in three vocabularies, and none could see
GPU clustering. :func:`vaelor.gpu_serving_target.resolve_gpu_serving_target` is
now the one home and both methods below are adapters over it, so Mode B follows
for free: the proxy fronts a cluster target, and the watch declines to relaunch
llama.cpp under one.
"""

from __future__ import annotations

import hashlib
import json
import logging
import subprocess
import time
from pathlib import Path
from typing import Any, Callable, Dict, Optional, Tuple

from .accelerator_runtime import (
    ACCELERATED, BASIS_DIFFERENTIAL, ON_CPU_BY_PLAN, MINIMUM_ACCELERATED_BYTES,
    NOT_ESTABLISHED, NOT_OFFLOADED,
)
from .app_port_claims import pick_model_port
from .credential_broker import CredentialBrokerClient
from .job_control import JobCancelled
from .gpu_model_choice import (
    discover_gpu_rocm_serving,
    generic_gpu_runtime,
    gpu_offload_plan,
)
from .gpu_cluster_mode import GPU_SERVING_LOCK, ClusterModeStore
from .gpu_loading_door import (
    RELAUNCH_COMPOSE, RELAUNCH_FP4, RELAUNCH_GENERIC, RELAUNCH_NONE, RELAUNCH_UNRESOLVED,
    watch_door,
)
from .gpu_rocm_supervisor import GpuBridgeLauncher, GpuRocmSupervisor
from .gpu_serving_target import (
    KIND_MANAGED_LOCAL,
    gpu_cluster_mode_active, resolve_gpu_serving_target,
)
from .managed_local_credentials import (
    AI_CHAT_MODEL_PREFIX,
    MANAGED_LOCAL_CREDENTIAL_LABEL,
    activate_managed_chat,
    managed_model_by_label,
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
    _artifact_identity,
)


logger = logging.getLogger(__name__)


#: The engine value the ROCmFP4 catalog entry carries, and the value the router
#: matches to send a deploy down the GPU path. Named once here so the deploy
#: router and the boot reconcile agree by construction; the string itself is
#: :mod:`vaelor.model_catalog`'s data (``catalog_engine`` returns it), not a
#: sentence this module owns, so it is a bare identifier rather than a message.
GPU_CHAT_ENGINE = "rocmfpx"


#: The two ROCmFPX-fork recipes a deploy result names, so the later frontend pass
#: can badge the recommended FP4 model apart from a user-chosen generic one
#: without parsing prose. ``rocmfp4`` is the measured 27B (``optimized`` True);
#: ``generic`` is any other artifact served on the fork with the conservative
#: :func:`vaelor.gpu_model_choice.generic_gpu_runtime` and the fit-decided offload.
GPU_RECIPE_ROCMFP4 = "rocmfp4"
GPU_RECIPE_GENERIC = "generic"

#: The fit ``reason`` the FP4 entry carries: it is the measured recipe, so it is
#: served at full offload by construction rather than by a computed fit.
GPU_FP4_MEASURED_REASON = (
    "The measured ROCmFP4 recipe offloads every layer to the GPU."
)


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

#: The fork's real serve failures - the FIX 1 fall-back set. ``RuntimeError`` is
#: a health timeout or a ``HardwareBridgeError`` (a subclass); ``OSError`` is a
#: missing engine/runtime lib (``GpuEngineMissingError``/``GpuRuntimeLibsMissingError``
#: subclass ``FileNotFoundError``); ``ValueError`` is a bad path, a
#: ``CredentialError`` (a subclass), or a decode error from a bounced broker; plus
#: any ``subprocess`` error. A ``JobCancelled`` (a ``RuntimeError`` subclass) is
#: deliberately re-raised by the helper, never fallen back on.
GPU_FORK_FALLBACK_ERRORS = (
    OSError, RuntimeError, ValueError, subprocess.SubprocessError,
)


#: What ``basis`` a residency verdict carries when GPU memory could not be read
#: at all - neither the before nor the after reading was available. Distinct from
#: :data:`~vaelor.accelerator_runtime.BASIS_DIFFERENTIAL`: an unread adapter says
#: nothing about residency, and a caller must not render it as either outcome.
BASIS_UNREAD = "unread"


def gpu_residency_verdict(
    before_bytes: Optional[int], after_bytes: Optional[int], model_bytes: int,
    *, fit_mode: str = "gpu", fit_reason: str = "",
) -> Dict[str, Any]:
    """Whether the GPU actually took the model, from a before/after memory delta.

    The block also carries ``state`` and ``in_use`` in the vocabulary every
    accelerator reading uses (`accelerator_runtime.verify_accelerator_in_use`),
    because the job card renders it as one (ACC-096): without them the card's
    Accelerator pill read "Not established" on every GPU deploy while the
    panel below it said "Running fully on the GPU" from the plan. ``fit_mode``
    is what the plan asked for: ``cpu`` expects no GPU memory at all, and
    ``partial`` expects some layers, so only the idle floor applies to it. A
    CPU plan is ``cpu-planned``, not the compose path's ``cpu`` ("by
    configuration"): the GPU plan decided it, and ``fit_reason`` - the plan's
    own sentence, which is "does not fit" or "the budget could not be read" -
    is the detail.

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
            "in_use": None,
            "state": NOT_ESTABLISHED,
            "vram_delta_bytes": None,
            "detail": (
                "GPU memory could not be read, so residency was not verified; "
                "the server answered but the backend it used is unconfirmed."
            ),
        }
    delta = int(after_bytes) - int(before_bytes)
    if fit_mode == "cpu":
        return {
            "basis": BASIS_DIFFERENTIAL,
            "resident": False,
            "in_use": False,
            "state": ON_CPU_BY_PLAN,
            "vram_delta_bytes": delta,
            "detail": fit_reason or (
                "The GPU plan placed this model on the CPU, so no GPU memory "
                "was expected."
            ),
        }
    threshold = int(MINIMUM_ACCELERATED_BYTES) if fit_mode == "partial" else max(
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
        "in_use": resident,
        "state": ACCELERATED if resident else NOT_OFFLOADED,
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

    def _gpu_rocm_chat_available(self) -> bool:
        """Whether this machine can serve a GPU AI-Chat model on the ROCmFPX fork.

        The same capability gate the recommendation uses
        (:func:`vaelor.gpu_model_choice.discover_gpu_rocm_serving`): a gfx1151
        GPU and a usable Docker, the self-contained ROCm serving container's
        prerequisites (the dead host fork no longer gates it). False everywhere
        else, so a box that cannot serve keeps the stock-compose AI-Chat path;
        the router reads this to send a non-FP4 ``ai-chat`` model to the GPU.
        """
        return bool(
            discover_gpu_rocm_serving(self._model_hardware_budget()).get("available")
        )

    def _require_gpu_rocm_serving(self) -> None:
        """Refuse a GPU-only deploy on a machine the capability gate turns away.

        ACC-061: the FP4 route asked nothing before launching, while every other
        GPU AI-Chat model passes :meth:`_gpu_rocm_chat_available`. The FP4 27B
        loads nowhere but the fork on a gfx1151 GPU, so a machine without that
        capability is told why - the gate's own sentence - before any server is
        stopped, rather than meeting a failed launch on hardware that cannot
        run it. The yes/no is :meth:`_gpu_rocm_chat_available`'s, the one
        answer every GPU route reads; the sentence is read only to refuse.
        """
        if self._gpu_rocm_chat_available():
            return
        verdict = discover_gpu_rocm_serving(self._model_hardware_budget())
        raise RuntimeError(str(verdict.get("reason") or (
            "This machine cannot serve the ROCmFP4 GPU model.")))

    def _gpu_fork_or_fallback(
        self, payload: Dict[str, Any], model: Path
    ) -> Tuple[Optional[Dict[str, Any]], str]:
        """Serve an ai-chat model on the GPU fork, or signal a compose fall-back.

        FIX 1 (HIGH-1 + MEDIUM-3). Returns ``(result, "")`` when the fork deploy
        succeeds - the caller returns it unchanged. On a fork SERVE failure it
        returns ``(None, reason)``: the fork stops the prior server and retires
        the other mechanism BEFORE it serves, so a propagated failure would leave
        the AI-Chat tier DEAD - and a bring-your-own GGUF is the model most likely
        to fail to load. So instead the caller falls through to the stock compose
        path and brings the SAME model up there (GPU-attempted via compose, CPU
        fallback if needed), and the compose result notes the fall-back. Only real
        serve failures are caught (:data:`GPU_FORK_FALLBACK_ERRORS`); a
        :class:`JobCancelled` is re-raised so a cancel is never a silent fallback.
        """
        try:
            return self._deploy_gpu_rocm_chat(payload, model), ""
        except GPU_FORK_FALLBACK_ERRORS as error:
            if isinstance(error, JobCancelled):
                raise
            self._checkpoint(
                15,
                "The GPU model server could not start; falling back to the "
                "standard local AI server",
                "starting",
            )
            return None, str(error)[:300]

    def _gpu_chat_runtime_for(self, model: Path) -> Dict[str, Any]:
        """The launch runtime for a GPU AI-Chat model: measured FP4, or generic.

        The FP4 catalog entry keeps its measured ``RUNTIME_Z2_GPU_ROCMFP4``
        verbatim, so the recommended path is byte-for-byte unchanged. Any other
        artifact - a stock catalog GGUF marked ``ai-chat``, or an off-catalog
        user ``.gguf`` - gets the conservative
        :func:`vaelor.gpu_model_choice.generic_gpu_runtime`, whose layer offload
        follows the unified-memory fit decision
        (:func:`vaelor.gpu_model_choice.gpu_offload_plan`) and which honestly
        reports a CPU fall-back when even a partial offload will not fit.

        Returns the runtime plus the descriptors the deploy result carries:
        ``recipe``, ``optimized``, and the fit ``mode``/``reason``.
        """
        identity = _artifact_identity(model)
        if identity and catalog_engine(**identity) == GPU_CHAT_ENGINE:
            return {
                "runtime": catalog_runtime(**identity),
                "recipe": GPU_RECIPE_ROCMFP4,
                "optimized": True,
                "mode": "gpu",
                "reason": GPU_FP4_MEASURED_REASON,
            }
        hardware = self._model_hardware_budget()
        try:
            model_bytes = model.stat().st_size
        except OSError:
            model_bytes = 0
        plan = gpu_offload_plan(
            model_bytes,
            hardware.get("accelerators"),
            system_memory_bytes=int(hardware.get("memory_total_bytes") or 0),
        )
        return {
            "runtime": generic_gpu_runtime(ngl=int(plan["ngl"])),
            "recipe": GPU_RECIPE_GENERIC,
            "optimized": False,
            "mode": plan["mode"],
            "reason": plan["reason"],
        }

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

    def _healthy_binding_result(self, port: int) -> Dict[str, Any]:
        """For an already-healthy model: no-op when the proxy is converged, else FORCE it.

        FINDING A. When the proxy is not converged (:meth:`_llm_binding_converged`) -
        a Disable/rotate the apply job missed, or a proxy that crashed while the model
        stayed up - :meth:`apply_llm_server` starts/stops/re-keys it to match, so the
        exposure converges within one reconcile.
        """
        if self._llm_binding_converged(port):
            return {"served": False, "reason": "already-healthy", "port": port}
        return {"served": True, "converged": True, **self.apply_llm_server()}

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
        # The FP4 27B keeps its measured runtime verbatim (recipe ``rocmfp4``,
        # ``optimized`` True); any other artifact reaching here - a stock catalog
        # GGUF or an off-catalog user ``.gguf`` marked ``ai-chat`` - is served on
        # the SAME fork with a generic runtime whose offload follows the
        # unified-memory fit decision, degrading to a CPU fall-back honestly.
        choice = self._gpu_chat_runtime_for(model)
        # The model is ALWAYS served loopback-only and keyless, whatever the LLM
        # Server state: LAN exposure is the separate auth proxy, not an engine bind
        # (the fork does not enforce a key). So the launch runtime is the pristine
        # catalog recipe - a COPY so the shared catalog dict is never mutated.
        runtime = dict(choice["runtime"])
        supervisor = self._gpu_rocm_supervisor()
        # W6-D2 reverse (LESSONS 6): never a port an installed app publishes,
        # running or stopped; an explicit one an app holds is refused by name.
        port = supervisor.allocate_port(pick_model_port(
            int(payload.get("port") or 0), self.workloads_root,
            lambda exclude=frozenset(): supervisor.allocate_port(0, exclude)))
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
        # VD-202: marked as AI Chat's model, so the Assistant's lease refuses it.
        candidate_id = AI_CHAT_MODEL_PREFIX + hashlib.sha256(
            "gpu:{}:{}:{}".format(model, port, time.time_ns()).encode()
        ).hexdigest()[:12]
        broker = CredentialBrokerClient()
        # The CLIENT credential base_url is the model's loopback endpoint
        # (``http://127.0.0.1:<port>/v1``), and its api_key is ALWAYS empty: the
        # model is unauthenticated on loopback, so internal AI Chat needs no key.
        # The LAN key lives only in the auth proxy's config, never in this
        # credential - a rotate re-keys the proxy without touching AI Chat.
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
            residency = self._gpu_residency_after_load(
                before_bytes, model, fit_mode=str(choice["mode"]),
                fit_reason=str(choice.get("reason") or ""),
            )
            if residency["state"] == NOT_OFFLOADED:
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
        # The model is now serving loopback. Bring the auth proxy to the persisted
        # LLM Server state in front of it (start the keyed gate if enabled, stop it
        # if disabled) and record the converged binding (FINDING A), so the
        # failure-watch reconcile does not needlessly re-apply a fresh deploy that
        # arrived with the LLM Server enabled. Best-effort: a proxy hiccup never
        # fails a healthy model deploy (the model stays loopback-safe).
        self._converge_llm_proxy(port)
        self._checkpoint(95, "GPU model server is healthy", "starting")
        return {
            "path": str(model),
            "endpoint": endpoint,
            "health": served["health_url"],
            "port": port,
            "context": int(runtime.get("context") or 0),
            "backend": GPU_CHAT_ENGINE,
            # The GPU is the target unless the fit decision fell the model back
            # to the CPU (a generic model too large even for a partial offload);
            # the FP4 recipe is always ``gpu`` here, so its result is unchanged.
            "device": "gpu" if choice["mode"] != "cpu" else "cpu",
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
            # Data for the later frontend pass (no rendering here): whether this
            # is the measured/recommended FP4 build, which fork recipe launched,
            # and the unified-memory fit verdict + its reason (so a CPU
            # fall-back can be shown as "does not fit the GPU", not a failure).
            "optimized": bool(choice["optimized"]),
            "recipe": choice["recipe"],
            "fit_mode": choice["mode"],
            "fit_reason": choice["reason"],
            "credential_id": credential["id"],
        }

    def _gpu_residency_after_load(
        self, before_bytes: Optional[int], model: Path, *, fit_mode: str = "gpu",
        fit_reason: str = "",
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
        return gpu_residency_verdict(
            before_bytes, after_bytes, model_bytes, fit_mode=fit_mode,
            fit_reason=fit_reason,
        )

    def ensure_gpu_chat_served(
        self, *, sleep: Callable[[float], None] = time.sleep
    ) -> Optional[Dict[str, Any]]:
        """Relaunch the GPU AI-Chat server after a reboot, on its pinned port.

        The GPU counterpart of :meth:`ensure_npu_assistant_served`, run OFF the
        job loop's critical path for the same reason: a 27B FP4 load is
        health-gated for minutes, and blocking the executor on that would stall
        the first job after a reboot. A reboot kills the GPU server and nothing
        else brings it back, so the AI-Chat tier is DOWN after every reboot until
        this reconcile - or a manual deploy - runs.

        Idempotent, gated, LOCKED and NON-FATAL:

        * Gated - it acts only on a ``managed-local`` serving target
          (:func:`~vaelor.gpu_serving_target.resolve_gpu_serving_target`). A
          hosted ai-chat, a single-model box sharing one endpoint with the
          Assistant, and a ``cluster`` target (Mode B, where vLLM holds the GPU)
          all no-op. **Mode B is why this gate must be the shared one**:
          relaunching llama.cpp under a cluster lease would put two engines on
          one GPU, the failure the mode switch exists to prevent.
        * Idempotent - a server already answering on the stored port is left
          untouched (relaunching would evict a loaded 27B to no purpose).
        * Locked - the pass holds :data:`~vaelor.gpu_cluster_mode.GPU_SERVING_LOCK`
          (D5), so a relaunch in flight and a deploy's ``enter`` cannot
          interleave: ``enter`` waits, then reads a GPU this pass is done with.
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
            with self._gpu_watch_lock():
                return self._gpu_chat_pass()
        except Exception:
            # Never let a boot-time reconcile keep the executor from processing
            # jobs, under ANY error - the same broad, non-BaseException catch the
            # NPU reconcile keeps, for the same reasons (a DB locked at boot, a
            # dict/int access on a malformed lease).
            return None

    def _gpu_watch_lock(self):
        """The process-wide GPU serving mutex, an injectable seam (D5).

        One object, shared with the mode reconcile and the switch, so everything
        that can start or stop a GPU engine is serialised. A test sets
        ``_gpu_watch_lock_cache`` to drive the contention directly.
        """
        return getattr(self, "_gpu_watch_lock_cache", None) or GPU_SERVING_LOCK

    def _cluster_mode_state(self):
        """The persisted GPU serving mode, a lazily-cached injectable seam.

        Read on every gate so a Mode B box is recognised as one; the read fails
        safe to Mode A, so a box that has never clustered behaves as before.
        """
        existing = getattr(self, "_cluster_mode_store_cache", None)
        if existing is None:
            existing = ClusterModeStore()
            self._cluster_mode_store_cache = existing
        return existing.read()

    def _refuse_deploy_under_cluster(self, sentence: str) -> None:
        """The deploy-job door: raise ``sentence`` while the file reads Mode B.

        VD-127's fourth verification found it open. A single-node ``ai-chat``
        deploy stops the supervisor (a no-op in Mode B), loads llama.cpp onto
        the GPU vLLM holds, and takes the ``ai-chat`` lease through the broker
        directly - never through `chat_inference.activate`, so the picker's
        refusal never saw it. The mode reconcile re-points the lease within
        30 s but stops nothing it did not start, so the fork stayed resident
        beside vLLM. `_deploy_model` asks this twice, one predicate and one
        sentence per condition, both the gate module's: the moment it has
        decided the surface, with ``AI_CHAT_HELD_BY_CLUSTER`` for ``ai-chat``,
        BEFORE ``supervisor.stop()`` and before any compose work (the FP4 fork
        route, the generic fork route and the compose fall-back all pass that
        decision); and again the moment the accelerator plan is settled, with
        ``GPU_HELD_BY_CLUSTER``, for any surface whose plan would put llama.cpp
        on the GPU - the Assistant's compose deploy on a single-accelerator
        gfx1151 box, which the surface gate passes by design. The sentence
        reaches the job as its failure message through the executor's
        ``RuntimeError`` arm.
        """
        if gpu_cluster_mode_active(self._cluster_mode_state()):
            raise RuntimeError(sentence)

    def _gpu_chat_pass(self) -> Optional[Dict[str, Any]]:
        """One failure-watch pass, under the lock. See :meth:`ensure_gpu_chat_served`."""
        mode_state = self._cluster_mode_state()
        # Mode B is never a relaunch, whatever the lease says: the switch
        # stopped llama.cpp so vLLM could have the aperture, and a managed-local
        # lease re-assigned under it is the mode reconcile's to re-point, not
        # this pass's to serve (VD-127). Read before the target, so the
        # supervisor is not consulted at all.
        if gpu_cluster_mode_active(mode_state):
            return None
        target = resolve_gpu_serving_target(self.credential_broker, mode_state)
        relaunch = self.gpu_chat_relaunch(target)
        # B1: a held way-back door is ruled on before anything is relaunched -
        # closed now for an arm that never fronts the gate, closed at its bound
        # for a relaunch that does not come (`gpu_loading_door`).
        watch_door(self._loading_door(), self._llm_proxy(), relaunch,
                   self._loading_door_now())
        if relaunch == RELAUNCH_NONE:
            return None
        port = target.port
        supervisor = self._gpu_rocm_supervisor()
        # #247m Finding 2: dispatch by the ACTIVE lease's mechanism, not by
        # assuming the fork. If ai-chat was switched to a stock GGUF, its server
        # is the `model-chat` compose project on this port; bring THAT back
        # rather than relaunching the fork 27B under the stock lease.
        if relaunch == RELAUNCH_COMPOSE:
            stock_project = self._gpu_chat_compose_project_for(port)
            if supervisor.healthy(port):
                return {"served": False, "reason": "already-healthy", "port": port}
            self._compose(
                stock_project, "up", "-d", "--remove-orphans", timeout=180
            )
            return {"served": True, "mechanism": "compose", "port": port}
        if relaunch != RELAUNCH_FP4:
            # A user-chosen GENERIC lease (resolved or not) re-serves from its
            # durable credential label, never as the recommended 27B.
            return self._reconcile_generic_gpu_chat(target.label, port, supervisor)
        model_path = self._gpu_chat_model_path()
        # The model is served loopback-only with the pristine catalog recipe: a
        # reboot relaunch comes back exactly as deployed, and the LLM Server's
        # LAN exposure is the separate auth proxy converged below.
        runtime = catalog_runtime(**_artifact_identity(model_path))
        if supervisor.healthy(port):
            # Answering on the stored port: a no-op if the auth proxy already
            # matches the persisted state, else FORCED to converge (FINDING A)
            # rather than the old unconditional "already-healthy" skip.
            return self._healthy_binding_result(port)
        result = supervisor.restart_if_unhealthy(
            str(model_path), port=port, runtime=runtime,
        )
        # A reboot killed the proxy too; (re)start it in front of the relaunched
        # loopback model to match the persisted LLM Server state.
        self._converge_llm_proxy(port)
        return {"served": True, **result}

    def gpu_chat_relaunch(self, target: Any = None) -> str:
        """Which arm this watch's pass relaunches AI Chat's model through - the one answer.

        `_gpu_chat_pass` dispatches on it, and the mode switch asks it on the
        way back from cluster serving whether the LLM Server's "loading" door
        will be replaced by a fronted model (B1, `gpu_loading_door`): only the
        FP4 and the resolvable generic arms converge the gate. ``none`` when
        the lease is not a managed-local GPU model; ``compose`` for a stock GGUF
        (the ``model-chat`` project, which the gate does not front);
        ``unresolved`` for a generic lease whose ``.gguf`` is not on disk.
        """
        if target is None:
            target = resolve_gpu_serving_target(
                self.credential_broker, self._cluster_mode_state())
        if target.kind != KIND_MANAGED_LOCAL:
            return RELAUNCH_NONE
        if self._gpu_chat_compose_project_for(target.port) is not None:
            return RELAUNCH_COMPOSE
        model_path = self._gpu_chat_model_path()
        # The FP4 27B only when the lease IS that model: a generic model wears
        # a label bearing its own stem and is never relaunched as the 27B.
        if model_path is not None and target.label == MANAGED_LOCAL_CREDENTIAL_LABEL.format(
                model_path.stem[:48]):
            return RELAUNCH_FP4
        if self._managed_model_by_label(target.label) is not None:
            return RELAUNCH_GENERIC
        return RELAUNCH_UNRESOLVED

    def _reconcile_generic_gpu_chat(
        self, label: str, port: int, supervisor: GpuRocmSupervisor
    ) -> Dict[str, Any]:
        """Relaunch a user-chosen GENERIC GPU chat model after a reboot.

        The generic counterpart of the FP4 arm of :meth:`ensure_gpu_chat_served`,
        reached only when the active ai-chat lease is NOT the recommended 27B. It
        shares that arm's discipline exactly: gated on the durable ai-chat lease,
        idempotent (a healthy server on the pinned port is left alone, so a loaded
        model is never evicted to no purpose), and non-fatal (a model that cannot
        be resolved on disk returns a reason, never a raise). The FP4 arm stays
        byte-for-byte.

        **The durable source is the lease itself, not a new state file.** A
        generic deploy writes the ai-chat credential with a label bearing the
        served model's file stem
        (:data:`~vaelor.managed_local_credentials.MANAGED_LOCAL_CREDENTIAL_LABEL`), and
        that label persists in the credential store across a reboot. The stem plus
        the managed models directory resolve the ``.gguf`` on disk
        (:meth:`_managed_model_by_label`), so the lease alone identifies the model
        and no parallel per-model record is introduced. The runtime is re-derived
        through :meth:`_gpu_chat_runtime_for`, so the unified-memory fit decision
        re-runs on the CURRENT hardware rather than being frozen at deploy time,
        and the model is served on the SAME fork the FP4 27B uses.
        """
        if supervisor.healthy(port):
            # Answering on the stored port: a no-op when the auth proxy already
            # matches the persisted state, else FORCED to converge (FINDING A).
            # Asked BEFORE the file is resolved: a serving model's gate is
            # re-keyed even when its file cannot be found, or a key revoked
            # with no apply job queued stayed admitted here for ever.
            return self._healthy_binding_result(port)
        model_path = self._managed_model_by_label(label)
        if model_path is None:
            # The lease names a model that is not (unambiguously) on disk: no
            # pointless launch against a file that is not there.
            return {"served": False, "reason": "generic-model-unresolved", "port": port}
        runtime = self._gpu_chat_runtime_for(model_path)["runtime"]
        result = supervisor.restart_if_unhealthy(
            str(model_path), port=port, runtime=runtime,
        )
        # A reboot killed the proxy too; (re)start it in front of the relaunched
        # loopback model to match the persisted LLM Server state.
        self._converge_llm_proxy(port)
        return {"served": True, **result}

    def _managed_model_by_label(self, label: str) -> Optional[Path]:
        """:func:`~vaelor.managed_local_credentials.managed_model_by_label` over this executor's models root."""
        return managed_model_by_label(self.models_root, label)
