"""Least-privileged executor for validated workload jobs."""

from __future__ import annotations

import json
import shutil
import time
import subprocess
import urllib.parse
import urllib.request
import urllib.error
import tempfile
import hashlib
import os
import re
import socket
from pathlib import Path
from typing import Any, Dict, Optional

from .compose_policy import (
    reject_secret_lines,
    reject_unsafe_keys,
    validate_normalized,
)
from .assistant_memory import AssistantMemoryStore
from .jobs import JobStore
from .app_catalog import APP_TEMPLATES, build_install_env, media_host_dir, render_compose
from .credential_broker import CredentialBrokerClient, CredentialError
from .copilot_setup import local_runtime_settings
from .inference_tuning import RECOMMENDED_RUNTIME_MODE
from .system_update import SystemUpdateClient
from .appliance_recovery import (
    ApplianceRecoveryClient, CONFIRMATION as RESET_CONFIRMATION,
    IMPORT_CONFIRMATION, UNINSTALL_CONFIRMATION,
)
from .host_desktop import HostDesktopClient
from .cluster_operations import ClusterOperations
from .cluster_jobs import execute_cluster_job
from .local_model_files import model_file, model_hardware_budget, remove_model
from .model_discovery import inspect_models, verify_inspected_selection
from .runtime_paths import data_path
from .workload_inventory import WorkloadInventory
from .workload_dependencies import WorkloadDependencyService
from .workload_removal import execute_managed_removal
from .custom_agents import CustomAgentStore
from .job_control import JobCancelled
from .application_deployments import ApplicationDeploymentStore
from .application_learning import ApplicationLearningStore
from .application_research_intelligence import ApplicationResearchIntelligence
from .application_research_server import ApplicationResearchClient
from .model_connection import resolve_model_connection
from .application_search import SearxSearchClient
from .application_executor import (
    application_learning_store,
    condense_docker_error,
    stored_application_environment,
    wait_for_application_health,
)
from .executor_application_jobs import ExecutorApplicationJobsMixin
from .executor_appliance_upgrade import ExecutorApplianceUpgradeMixin
from .executor_compose_lifecycle import ExecutorComposeLifecycleMixin
from .executor_gpu_deploy import ExecutorGpuDeployMixin
from .executor_model_deploy import ExecutorModelDeployMixin
from .executor_network import (
    available_model_port, deployment_copilot_result, ensure_extra_host_ports_available,
    ensure_host_port_available,
)
from .web_research import SEARCH_URL, WebResearchError, WebResearchManager


class JobExecutor(
    ExecutorApplicationJobsMixin, ExecutorApplianceUpgradeMixin,
    ExecutorComposeLifecycleMixin, ExecutorModelDeployMixin,
    ExecutorGpuDeployMixin,
):
    """Process one allowlisted job without invoking a shell."""

    def __init__(
        self,
        store: JobStore,
        workloads_root: str = data_path("workloads"),
        models_root: str = data_path("models"),
        backups_root: str = data_path("backups"),
        timeout_seconds: int = 30,
        application_deployments: Optional[ApplicationDeploymentStore] = None,
        application_learning: Optional[ApplicationLearningStore] = None,
        workload_dependencies: Optional[WorkloadDependencyService] = None,
        credential_broker=None,
        web_research: Optional[WebResearchManager] = None,
        application_research=None,
        preference_store: Optional[AssistantMemoryStore] = None,
    ):
        self.store = store
        self.preference_store = preference_store
        self.workloads_root = Path(workloads_root).resolve()
        self.models_root = Path(models_root).resolve()
        self.backups_root = Path(backups_root).resolve()
        self.timeout_seconds = max(1, min(timeout_seconds, 300))
        self._active_job_id = None
        self.system_update = SystemUpdateClient()
        self.appliance_recovery = ApplianceRecoveryClient()
        self.host_desktop = HostDesktopClient()
        self.web_research = web_research or WebResearchManager()
        self.cluster = ClusterOperations()
        self.application_deployments = (
            application_deployments
            or ApplicationDeploymentStore(str(
                self.workloads_root.parent
                / "applications"
                / "applications.sqlite3"
            ))
        )
        self.application_learning = application_learning or application_learning_store(self.workloads_root)
        self.credential_broker = credential_broker or CredentialBrokerClient()
        self.application_research = application_research or ApplicationResearchIntelligence(
            ApplicationResearchClient(),
            lambda: resolve_model_connection(self.credential_broker),
            search_client=SearxSearchClient(SEARCH_URL),
            # #247r: the same owned lifecycle the executor autostart uses, so an
            # application-research job auto-provisions guarded web research on
            # demand rather than dead-ending when it was never set up.
            web_research=self.web_research,
            timeout_seconds=60,
        )
        self.workload_dependencies = workload_dependencies or WorkloadDependencyService(
            WorkloadInventory(str(self.workloads_root), str(self.models_root)),
            self.credential_broker,
            CustomAgentStore(str((
                self.workloads_root.parent if self.workloads_root.name == "workloads"
                else self.workloads_root
            ) / "assistant" / "custom-agents.sqlite3")),
        )

    def ensure_web_research_started(self) -> Optional[Dict[str, Any]]:
        """Bring an already-enabled search backend back up automatically.

        #212 / owner decision (auto-start-on-enable): once web research
        is enabled its backend must be up for a granted agent to fetch, with no
        separate manual container deploy. This runs on executor start-up and
        recovers a configured-but-down service (a `compose down` left in place, a
        boot where Docker's own restart policy did not win). It is idempotent
        (a ready service is left untouched) and reuses the one owned compose
        project with its pinned digest and bounds - no second docker mechanism.

        Non-fatal: it only acts when the service is already configured, and any
        Docker failure is swallowed so it can never keep the executor from
        processing jobs.
        """
        try:
            # First-boot auto-provision: install the guarded search backend once
            # on a clean box so custom-application research works without the
            # operator first discovering and approving the separate install
            # (marker-gated, respects a later deliberate remove, no-op once done).
            self.web_research.autoprovision()
            if not self.web_research.is_enabled():
                return None
            return self.web_research.ensure_running()
        except (WebResearchError, OSError, subprocess.SubprocessError):
            return None

    def run_once(self) -> Optional[Dict[str, Any]]:
        job = self.store.claim_next()
        if job is None:
            return None
        self._active_job_id = job["id"]
        try:
            self._checkpoint(10, "Request accepted")
            if job["type"] in {"application.research", "compose.draft"}:
                return self._run_application_job(job)
            if job["type"] == "model.inspect":
                result = self._inspect_models(job["payload"])
                return self.store.finish(
                    job["id"],
                    state="completed",
                    message="Model candidates inspected",
                    result=result,
                )
            if job["type"] == "model.download":
                result = self._download_model(job["payload"])
                return self.store.finish(
                    job["id"], state="completed",
                    message="Model downloaded and verified", result=result,
                )
            if job["type"] == "model.deploy":
                result = self._deploy_model(job["payload"])
                if self.preference_store is not None:
                    self.preference_store.set_preference(
                        job["actor"], "intelligence_choice", "local"
                    )
                return self.store.finish(
                    job["id"], state="healthy",
                    message="Local AI server is healthy", result=result,
                )
            if job["type"] == "model.install_release":
                result = self._install_release_model(job["payload"])
                if self.preference_store is not None:
                    self.preference_store.set_preference(
                        job["actor"], "intelligence_choice", "local"
                    )
                return self.store.finish(
                    job["id"], state="healthy",
                    message="On-device NPU model installed and serving",
                    result=result,
                )
            if job["type"] == "model.remove":
                raise ValueError(
                    "Legacy model removal is blocked. Review a managed removal plan."
                )
            if job["type"] == "managed.remove":
                result = self._remove_managed(
                    job["payload"], job.get("actor", ""), job["id"]
                )
                return self.store.finish(
                    job["id"], state="completed",
                    message="Managed resource removed", result=result,
                )
            if job["type"] == "compose.validate":
                result = self._validate_compose(job["payload"])
                return self.store.finish(
                    job["id"],
                    state="completed",
                    message="Compose project validated",
                    result=result,
                )
            if job["type"] == "compose.install":
                result = self._install_template(job["payload"])
                return self.store.finish(
                    job["id"], state="healthy",
                    message="App installed and started", result=result,
                )
            if job["type"] == "compose.import":
                result = self._import_compose(job["payload"], job["actor"])
                return self.store.finish(
                    job["id"], state="healthy",
                    message="Custom Compose project validated and deployed", result=result,
                )
            if job["type"] == "compose.backup":
                result = self._backup_compose({
                    **job["payload"],
                    "_actor": job["actor"],
                    "_job_id": job["id"],
                })
                return self.store.finish(
                    job["id"], state="completed",
                    message="Compose project backup verified", result=result,
                )
            if job["type"] == "compose.remove":
                raise ValueError(
                    "Legacy application removal is blocked. Review a managed removal plan."
                )
            if job["type"] == "checkpoint.restore":
                result = self._restore_checkpoint({
                    **job["payload"],
                    "_actor": job["actor"],
                    "_job_id": job["id"],
                })
                return self.store.finish(
                    job["id"], state="healthy",
                    message="Checkpoint restored and application started", result=result,
                )
            if job["type"] in {"compose.start", "compose.restart", "compose.stop", "compose.update"}:
                result = self._lifecycle(job["type"].split(".", 1)[1], job["payload"])
                return self.store.finish(
                    job["id"], state="completed",
                    message=f"App {job['type'].split('.', 1)[1]} completed", result=result,
                )
            if job["type"] == "compose.pull":
                project = self._project_directory(str(job["payload"].get("project", "")))
                self._compose(project, "pull", timeout=900)
                return self.store.finish(
                    job["id"], state="completed", message="App images downloaded",
                    result={"project": project.name, "action": "pull"},
                )
            if job["type"] == "compose.deploy":
                project = self._project_directory(str(job["payload"].get("project", "")))
                self._compose(project, "up", "-d", "--remove-orphans", timeout=180)
                return self.store.finish(
                    job["id"], state="healthy", message="App deployed",
                    result={"project": project.name, "action": "deploy"},
                )
            if job["type"] == "agent.deploy":
                result = deployment_copilot_result(job["payload"])
                return self.store.finish(
                    job["id"],
                    state="healthy",
                    message="Deployment Copilot is available",
                    result=result,
                )
            if job["type"] in {"system.update.stage", "system.update.apply"}:
                action = job["type"].rsplit(".", 1)[1]
                if action == "apply" and job["payload"].get("confirm") != "apply-updates":
                    raise ValueError("Confirm that the staged system updates should be installed.")
                self._checkpoint(
                    25,
                    "Downloading package updates" if action == "stage"
                    else "Installing staged package updates",
                    "downloading" if action == "stage" else "starting",
                )
                result = self.system_update.run(action)
                return self.store.finish(
                    job["id"],
                    state="completed",
                    message="Updates downloaded and staged" if action == "stage"
                    else "System updates installed",
                    result=result,
                )
            if job["type"] == "appliance.factory-reset":
                payload = job["payload"]
                if payload.get("confirmation") != RESET_CONFIRMATION:
                    raise ValueError("Factory reset confirmation did not match.")
                self._checkpoint(
                    25,
                    "Factory reset staged; transferring to the recovery broker",
                    "starting",
                )
                result = self.appliance_recovery.reset(
                    str(payload.get("plan_id", "")),
                    str(payload.get("confirmation", "")),
                )
                return self.store.finish(
                    job["id"],
                    state="completed",
                    message="Factory reset accepted; the dashboard will return to first-run setup",
                    result=result,
                )
            # Mirrors the appliance.factory-reset branch above: validate the typed
            # confirmation, then hand the staged plan to the recovery broker - here
            # ApplianceRecoveryClient.uninstall instead of .reset.
            if job["type"] == "appliance.remove-vaelor":
                payload = job["payload"]
                if payload.get("confirmation") != UNINSTALL_CONFIRMATION:
                    raise ValueError("The removal confirmation did not match.")
                self._checkpoint(
                    25,
                    "Removal staged; transferring to the recovery broker",
                    "starting",
                )
                result = self.appliance_recovery.uninstall(
                    str(payload.get("plan_id", "")),
                    str(payload.get("confirmation", "")),
                )
                return self.store.finish(
                    job["id"],
                    state="completed",
                    message="Removal accepted; every Vaelor service and the OS stack it installed are being removed",
                    result=result,
                )
            if job["type"] == "appliance.portable-import":
                payload = job["payload"]
                if payload.get("confirmation") != IMPORT_CONFIRMATION:
                    raise ValueError("Portable import confirmation did not match.")
                self._checkpoint(
                    25, "Portable state verified; transferring to the recovery broker",
                    "starting",
                )
                result = self.appliance_recovery.portable_import(
                    str(payload.get("plan_id", "")),
                    str(payload.get("confirmation", "")),
                )
                return self.store.finish(
                    job["id"],
                    state="completed",
                    message="Portable import accepted; sign in again after Vaelor restarts",
                    result=result,
                )
            if job["type"] == "appliance.upgrade":
                return self._run_appliance_upgrade(job)
            if job["type"] == "host.vnc.enable":
                if job["payload"].get("confirm") != "enable-host-vnc":
                    raise ValueError("Confirm host browser desktop setup.")
                self._checkpoint(
                    20,
                    "Installing the optional loopback browser desktop",
                    "downloading",
                )
                result = self.host_desktop.commission_vnc()
                return self.store.finish(
                    job["id"],
                    state="completed",
                    message="Optional host browser desktop is ready",
                    result=result,
                )
            if job["type"] == "host.docker.install":
                if job["payload"].get("confirm") != "install-docker":
                    raise ValueError("Confirm that Docker should be installed on this host.")
                self._checkpoint(20, "Installing the OS-supported Docker runtime", "downloading")
                result = self.host_desktop.install_docker()
                return self.store.finish(
                    job["id"],
                    state="completed",
                    message="Docker and Compose installed; Vaelor access is being refreshed",
                    result=result,
                )
            if job["type"] == "host.memory.optimize":
                profile = str(job["payload"].get("profile", ""))
                if job["payload"].get("confirm") != "apply-memory-profile":
                    raise ValueError("Confirm the reviewed memory profile.")
                self._checkpoint(30, "Applying the reviewed Linux memory policy", "starting")
                result = self.host_desktop.optimize_memory(profile)
                return self.store.finish(
                    job["id"],
                    state="completed",
                    message="Memory policy applied without a reboot",
                    result=result,
                )
            if job["type"] == "host.web-research.manage":
                action = str(job["payload"].get("action", ""))
                self._checkpoint(
                    20,
                    "Applying the guarded web research lifecycle",
                    "starting",
                )
                result = self.web_research.execute(
                    action,
                    str(job["payload"].get("confirmation", "")),
                    purge=bool(job["payload"].get("purge", False)),
                )
                messages = {
                    "install": "Guarded web research installed and verified",
                    "repair": "Guarded web research repaired and verified",
                    "remove": "Guarded web research removed",
                }
                return self.store.finish(
                    job["id"],
                    state="completed",
                    message=messages.get(action, "Guarded web research lifecycle completed"),
                    result=result,
                )
            if job["type"].startswith("cluster."):
                return execute_cluster_job(
                    job, self.cluster, self.store, self._checkpoint
                )
            return self.store.finish(
                job["id"],
                state="failed",
                message="This job type requires the privileged appliance service.",
                result={"code": "executor_not_commissioned"},
            )
        except JobCancelled:
            return self.store.finish(
                job["id"], state="cancelled", message="Cancelled safely",
                result={"recovery": "Review the workload state before retrying."},
            )
        except (OSError, RuntimeError, ValueError, subprocess.SubprocessError) as error:
            return self.store.finish(
                job["id"],
                state="failed",
                message=str(error),
                result={"code": "job_failed"},
            )
        finally:
            self._active_job_id = None

    def _checkpoint(
        self, percent: int, message: str, state: str = "validating", *,
        phase: str | None = None, blocker_layer: str | None = None,
    ):
        if self._active_job_id is None:
            return
        if self.store.is_cancelling(self._active_job_id):
            raise JobCancelled()
        self.store.progress(
            self._active_job_id, state, percent, message,
            phase=phase, blocker_layer=blocker_layer,
        )

    def _inspect_models(self, payload: Dict[str, Any]) -> Dict[str, Any]:
        return inspect_models(
            payload, self._open_huggingface, self._checkpoint,
            self._model_hardware_budget(),
        )

    @staticmethod
    def _validate_repo_file(repo: str, filename: str):
        if not re.fullmatch(r"[A-Za-z0-9_.-]+/[A-Za-z0-9_.-]+", repo):
            raise ValueError("Choose a valid Hugging Face repository.")
        if (
            not filename.lower().endswith(".gguf")
            or filename.startswith(("/", "\\"))
            or ".." in Path(filename).parts
            or "\\" in filename
        ):
            raise ValueError("Choose a safe GGUF file from the selected repository.")

    def _install_release_model(self, payload: Dict[str, Any]) -> Dict[str, Any]:
        """Install and serve a fine-tuned NPU model from its pinned release.

        Unlike a Hugging Face GGUF (download then deploy), a fine-tune arrives as
        a release: the job carries only the flm tag, and the catalog holds the
        pinned source URL and sha256 - the trust anchor is the code shipped in the
        wheel, never the job payload. The download, sha-verify and unpack into the
        snap paths flm serves from happen as root behind the hardware bridge
        (:meth:`HardwareBridgeClient.flm_install_release`), then the shared NPU-
        assistant deploy the reconcile uses brings the model up and health-checks
        it - here driven by an explicit plan built from the known tag rather than
        the fit ladder, which does not stock release models. A long client timeout
        covers the multi-GB download; the bridge serialises it under its flm lock.
        """
        from .flm_service import npu_model_present
        from .hardware_bridge import HardwareBridgeClient
        from .model_catalog import catalog_release_for_tag

        release = catalog_release_for_tag(str(payload.get("tag") or ""))
        if release is None:
            raise ValueError(
                "No release-installable on-device model matches this request."
            )
        # Skip the multi-GB download when the model is already installed on disk.
        # `fetch-npu-model.sh` installs the NPU model into
        # `/var/lib/vaelor/flm/models` on a clean box, and the first-boot
        # auto-enable enqueues this deploy to serve+pin it. Re-downloading it
        # would be pointless (and would fail on a box with no release source), so
        # a present model dir goes straight to the serve+pin path below.
        if not npu_model_present():
            HardwareBridgeClient(timeout=1200).flm_install_release(
                release["source_url"], release["sha256"]
            )
        # The install placed OUR flm-real + model in the snap paths, so serve the
        # known tag directly. A pathless `_deploy_model` would consult the fit
        # ladder (`should_serve_on_npu`), which stocks only Hugging Face GGUFs and
        # cannot pick a release model on a fresh box - it would fall through to the
        # llama.cpp path and fail with "Choose a downloaded managed GGUF model".
        # We know the tag, so we build the plan `_deploy_npu_assistant` needs and
        # call it: launch flm-real, health-check, pin + activate the managed
        # credential on the served tag - the same path the reconcile uses.
        plan = {
            "flm_tag": release["tag"],
            "context_tokens": 16384,
            "model": release["name"],
            "context_reason": "",
        }
        return self._deploy_npu_assistant({"surface": "assistant"}, {"plan": plan})

    def _download_model(self, payload: Dict[str, Any]) -> Dict[str, Any]:
        repo = str(payload.get("repo", "")).strip()
        filename = str(payload.get("file", "")).strip()
        self._validate_repo_file(repo, filename)
        expected_size = int(payload.get("size_bytes") or 0)
        if expected_size < 0 or expected_size > 16 * 1024 ** 3:
            raise ValueError("The selected model file is outside the supported size limit.")
        verify_inspected_selection(
            self.store, payload.get("inspection_job_id"), repo, filename,
            expected_size,
        )
        runtime_preview = None
        if expected_size:
            runtime_preview = local_runtime_settings(
                self._model_hardware_budget(), expected_size,
                RECOMMENDED_RUNTIME_MODE,
                # Named so the sizing can use a measured footprint instead of
                # `model_size * 1.25 + overhead`, which is 1,024 MiB low for
                # Qwen3 1.7B at 2,048 tokens (#117, VD-070). Without the
                # artifact identity the estimate is all that is available.
                repo=repo, file=filename,
            )
        model_dir = (self.models_root / repo.replace("/", "--")).resolve()
        if self.models_root not in model_dir.parents:
            raise ValueError("The model destination is outside the managed model root.")
        model_dir.mkdir(parents=True, exist_ok=True)
        destination = (model_dir / Path(filename).name).resolve()
        if model_dir not in destination.parents:
            raise ValueError("The model file path is invalid.")
        # Reuse a file already present in managed storage instead of re-fetching
        # gigabytes from Hugging Face. A model preserved from a prior install or
        # copied in by hand sits at exactly this path; when its byte count matches
        # the catalog's exact expected size (the same `!=` size gate the download
        # enforces below), verify its bytes and return the SAME result a completed
        # download would - no network, no re-download. Only the completed file is
        # honoured, never a stale ``.part``, and the size match must be exact.
        if expected_size and destination.is_file() and (
            destination.stat().st_size == expected_size
        ):
            self._checkpoint(94, "Verifying the existing model file", "validating")
            digest = hashlib.sha256()
            with destination.open("rb") as source:
                for chunk in iter(lambda: source.read(4 * 1024 * 1024), b""):
                    digest.update(chunk)
            return {
                "repo": repo, "file": filename, "path": str(destination),
                "size_bytes": destination.stat().st_size,
                "sha256": digest.hexdigest(),
                "resumable": True, "runtime_preview": runtime_preview,
            }
        partial = destination.with_suffix(destination.suffix + ".part")
        free = shutil.disk_usage(model_dir).free
        required = expected_size or 8 * 1024 ** 3
        if free - required < 1024 ** 3:
            raise ValueError("Keep at least 1 GB free after downloading this model.")
        existing = partial.stat().st_size if partial.exists() else 0
        url = "https://huggingface.co/{}/resolve/main/{}".format(
            repo, urllib.parse.quote(filename, safe="/")
        )
        headers = {"Accept": "application/octet-stream", "User-Agent": "Vaelor-Control-Plane/2"}
        if existing:
            headers["Range"] = "bytes={}-".format(existing)
        request = urllib.request.Request(url, headers=headers)
        self._checkpoint(15, "Connecting to Hugging Face", "downloading")
        with self._open_huggingface(request) as response:
            status = getattr(response, "status", 200)
            if existing and status != 206:
                existing = 0
                partial.unlink(missing_ok=True)
            total = expected_size or (
                existing + int(response.headers.get("Content-Length") or 0)
            )
            mode = "ab" if existing else "wb"
            downloaded = existing
            last_progress = -1
            with partial.open(mode) as output:
                while True:
                    if self._active_job_id and self.store.is_cancelling(self._active_job_id):
                        raise JobCancelled()
                    chunk = response.read(1024 * 1024)
                    if not chunk:
                        break
                    output.write(chunk)
                    downloaded += len(chunk)
                    if total:
                        progress = min(90, 15 + int(downloaded / total * 75))
                        if progress >= last_progress + 2:
                            self._checkpoint(progress, "Downloaded {} of {} MiB".format(
                                downloaded // 1024 ** 2, total // 1024 ** 2
                            ), "downloading")
                            last_progress = progress
                output.flush()
                os.fsync(output.fileno())
        if expected_size and downloaded != expected_size:
            raise ValueError("The downloaded size did not match the selected file.")
        self._checkpoint(94, "Verifying downloaded model", "validating")
        digest = hashlib.sha256()
        with partial.open("rb") as source:
            for chunk in iter(lambda: source.read(4 * 1024 * 1024), b""):
                digest.update(chunk)
        partial.replace(destination)
        return {
            "repo": repo, "file": filename, "path": str(destination),
            "size_bytes": downloaded, "sha256": digest.hexdigest(),
            "resumable": True, "runtime_preview": runtime_preview,
        }

    def _open_huggingface(self, request):
        try:
            return urllib.request.urlopen(request, timeout=self.timeout_seconds)
        except urllib.error.HTTPError as error:
            if error.code not in (401, 403):
                raise
            try:
                lease = CredentialBrokerClient().resolve_active("model-download")
            except (CredentialError, OSError) as broker_error:
                raise ValueError(
                    "This model needs Hugging Face access. Connect a Hugging Face token in Assistant setup, then retry."
                ) from broker_error
            token = str(lease.get("token", ""))
            if not token:
                raise ValueError("The Hugging Face credential is unavailable.")
            headers = dict(request.header_items())
            headers["Authorization"] = "Bearer {}".format(token)
            authenticated = urllib.request.Request(
                request.full_url,
                data=request.data,
                headers=headers,
                method=request.get_method(),
            )
            try:
                return urllib.request.urlopen(
                    authenticated, timeout=self.timeout_seconds
                )
            except urllib.error.HTTPError as authenticated_error:
                if authenticated_error.code in (401, 403):
                    raise ValueError(
                        "Hugging Face did not allow this model. Accept its license and check the connected token, then retry."
                    ) from authenticated_error
                raise

    def _model_file(self, value: str) -> Path:
        return model_file(value, self.models_root)

    def _remove_model(self, payload: Dict[str, Any]) -> Dict[str, Any]:
        return remove_model(payload, self.models_root, self.workloads_root)

    def _remove_managed(
        self, payload: Dict[str, Any], actor: str, job_id: str | None = None
    ) -> Dict[str, Any]:
        return execute_managed_removal(
            payload, actor, self.workload_dependencies, self._remove_compose,
            self.credential_broker, self.models_root, self.backups_root,
            job_id=job_id,
        )

    def _model_hardware_budget(self) -> Dict[str, Any]:
        return model_hardware_budget(self.models_root, self.workloads_root)

    @staticmethod
    def _available_model_port() -> int:
        return available_model_port(socket.socket)

    def _project_directory(self, project_name: str) -> Path:
        if not re.fullmatch(r"[a-z0-9][a-z0-9_-]{1,47}", project_name):
            raise ValueError("The Compose project name is invalid.")
        project = (self.workloads_root / project_name).resolve()
        if self.workloads_root not in project.parents:
            raise ValueError("The Compose project is outside the workload root.")
        return project

    def _validate_compose(self, payload: Dict[str, Any]) -> Dict[str, Any]:
        project = self._project_directory(str(payload.get("project", "")))
        compose_file = (project / "compose.yaml").resolve()
        if project not in compose_file.parents or not compose_file.is_file():
            raise ValueError("Create compose.yaml in the managed project before validation.")
        docker = shutil.which("docker")
        if docker is None:
            raise RuntimeError("Docker is not installed.")
        self._checkpoint(35, "Validating Compose configuration")
        result = subprocess.run(
            [
                docker,
                "compose",
                "--project-directory",
                str(project),
                "-f",
                str(compose_file),
                "config",
                "--quiet",
            ],
            capture_output=True,
            check=False,
            text=True,
            timeout=self.timeout_seconds,
        )
        if result.returncode != 0:
            raise ValueError((result.stderr or "Compose validation failed.")[:1000])
        return {"project": project.name, "compose_file": str(compose_file)}

    def _compose(self, project: Path, *arguments: str, timeout: int = 120):
        docker = shutil.which("docker")
        if docker is None:
            raise RuntimeError("Docker is not installed.")
        compose_file = project / "compose.yaml"
        command = [
                docker, "compose", "--project-name", project.name,
                "--project-directory", str(project), "-f", str(compose_file),
                *arguments,
            ]
        environment = stored_application_environment(compose_file)
        with tempfile.TemporaryFile(mode="w+t", encoding="utf-8") as output:
            process = subprocess.Popen(
                command, stdout=output, stderr=output, text=True,
                env=environment,
            )
            deadline = time.monotonic() + timeout
            while process.poll() is None:
                if self._active_job_id and self.store.is_cancelling(self._active_job_id):
                    process.terminate()
                    try:
                        process.wait(timeout=5)
                    except subprocess.TimeoutExpired:
                        process.kill()
                    raise JobCancelled()
                if time.monotonic() >= deadline:
                    process.kill()
                    raise subprocess.TimeoutExpired(command, timeout)
                time.sleep(0.5)
            output.seek(0)
            text = output.read()[-65536:]
        if process.returncode != 0:
            raise ValueError(condense_docker_error(text, "Docker operation failed"))
        return {"returncode": process.returncode, "output": text}

    def _install_template(self, payload: Dict[str, Any]) -> Dict[str, Any]:
        template_id = str(payload.get("template", "")).strip()
        if template_id not in APP_TEMPLATES:
            raise ValueError("Choose a supported application.")
        project = self._project_directory(template_id)
        compose_file = project / "compose.yaml"
        raw_port = payload.get("port", APP_TEMPLATES[template_id]["default_port"])
        if isinstance(raw_port, bool) or not isinstance(raw_port, int):
            raise ValueError("Choose a whole-number application port.")
        port = raw_port
        compose_existed = compose_file.exists()
        existing = compose_file.read_text(encoding="utf-8") if compose_existed else None
        # A re-install must reuse the secret already persisted in the existing
        # compose (otherwise the freshly generated one would differ and trip the
        # "different managed configuration" guard, and rotate the password); a
        # first install mints a fresh, unique secret for each secret_env key.
        install_env = build_install_env(template_id, existing)
        content = render_compose(template_id, port, install_env)
        if compose_existed and existing != content:
            raise ValueError("This app already has a different managed configuration.")
        if not compose_existed:
            ensure_host_port_available(port, socket.socket)
            ensure_extra_host_ports_available(
                APP_TEMPLATES[template_id]["name"],
                APP_TEMPLATES[template_id].get("extra_ports", []),
                socket.socket,
            )
        project.mkdir(parents=True, exist_ok=True)
        project.chmod(0o2770)
        # A media app renders one controlled bind mounting a managed host dir at
        # its media_mount. Pre-create it with the SAME managed posture as the
        # project dir (2770 vaelor-jobs) so the file manager and the app share
        # it through the group, rather than letting `docker compose up`
        # auto-create it root:root 0755. Uses the identical media_host_dir the
        # renderer used, so the created path and the bind's source cannot drift.
        media_dir = media_host_dir(template_id)
        if media_dir is not None:
            media_path = Path(media_dir)
            media_path.mkdir(parents=True, exist_ok=True)
            media_path.chmod(0o2770)
        temporary = project / ".compose.yaml.tmp"
        try:
            if not compose_existed:
                temporary.write_text(content, encoding="utf-8")
                temporary.chmod(0o660)
                temporary.replace(compose_file)
            self._checkpoint(20, "Validating app configuration")
            self._compose(project, "config", "--quiet", timeout=30)
            self._checkpoint(35, "Downloading app image", "downloading")
            self._compose(project, "pull", timeout=900)
            self._checkpoint(80, "Starting app", "starting")
            self._compose(project, "up", "-d", "--remove-orphans", timeout=180)
        except Exception:
            temporary.unlink(missing_ok=True)
            if not compose_existed:
                # This run created the managed project and it did not finish.
                # The old cleanup guarded on `compose_file.exists() and not
                # any(project.iterdir())` - a contradiction once the compose
                # file is written, so it never fired and left an orphan project
                # that /managed cannot show and that bricks every later install
                # of this app (#205 finding 1). Mirror `_import_compose`: tear
                # down anything that started, then remove the whole directory.
                self._discard_incomplete_install(project, compose_file)
            raise
        return {
            "project": project.name,
            "template": template_id,
            "port": port,
            "endpoint": f"http://DEVICE:{port}",
        }

    def _discard_incomplete_install(self, project: Path, compose_file: Path) -> None:
        """Remove a managed project this run created but could not complete."""
        if compose_file.exists():
            try:
                self._compose(project, "down", "--remove-orphans", timeout=180)
            except Exception:
                # A failed teardown must not stop us removing the orphan
                # directory, which is the part that bricks re-install.
                pass
        compose_file.unlink(missing_ok=True)
        try:
            if project.exists() and not any(project.iterdir()):
                project.rmdir()
        except OSError:
            pass

    def _lifecycle(self, action: str, payload: Dict[str, Any]) -> Dict[str, Any]:
        project = self._project_directory(str(payload.get("project", "")))
        if not (project / "compose.yaml").is_file():
            raise ValueError("The managed app configuration was not found.")
        commands = {
            "start": ("up", "-d"),
            "restart": ("restart",),
            "stop": ("stop",),
            "update": ("pull",),
        }
        if action not in commands:
            raise ValueError("Choose a supported app action.")
        self._checkpoint(30, "{} app".format(action.capitalize()), "starting")
        if action != "update":
            self._compose(project, *commands[action], timeout=180)
            return {"project": project.name, "action": action}

        previous_images = self._capture_compose_images(project)
        if not previous_images:
            raise RuntimeError(
                "Update stopped before changing the app because the running image set could not be captured for rollback."
            )
        rollback_file = project / ".vaelor-update-rollback.json"
        self._checkpoint(40, "Downloading replacement images", "downloading")
        self._compose(project, "pull", timeout=900)
        try:
            self._checkpoint(70, "Starting replacement images", "starting")
            self._compose(project, "up", "-d", "--remove-orphans", timeout=180)
            health = wait_for_application_health(
                lambda: self._compose(project, "ps", "--format", "json", timeout=30),
                sorted(previous_images),
            )
        except Exception as update_error:
            rollback_file.write_text(
                json.dumps({
                    "services": {
                        service: {"image": image_id}
                        for service, image_id in previous_images.items()
                    }
                }, sort_keys=True),
                encoding="utf-8",
            )
            rollback_file.chmod(0o660)
            try:
                self._checkpoint(85, "Restoring the previous application images", "starting")
                self._compose(
                    project, "-f", str(rollback_file), "up", "-d", "--remove-orphans",
                    timeout=180,
                )
                wait_for_application_health(
                    lambda: self._compose(project, "ps", "--format", "json", timeout=30),
                    sorted(previous_images),
                )
            except Exception as rollback_error:
                raise RuntimeError(
                    "Application update failed and automatic image rollback is incomplete: {}"
                    .format(str(rollback_error)[:500])
                ) from update_error
            finally:
                rollback_file.unlink(missing_ok=True)
            raise RuntimeError(
                "Application update failed; Vaelor restored and verified the previous image set."
            ) from update_error
        return {
            "project": project.name,
            "action": action,
            "health": health,
            "rollback": {
                "available": True,
                "applied": False,
                "captured_services": sorted(previous_images),
            },
        }

    def _capture_compose_images(self, project: Path) -> Dict[str, str]:
        """Capture immutable image IDs for every running Compose service."""
        result = self._compose(project, "ps", "--format", "json", timeout=30)
        raw = str(result.get("output", "")).strip()
        if not raw:
            return {}
        try:
            parsed = json.loads(raw)
            rows = parsed if isinstance(parsed, list) else [parsed]
        except json.JSONDecodeError:
            try:
                rows = [json.loads(line) for line in raw.splitlines() if line.strip()]
            except json.JSONDecodeError:
                return {}
        docker = shutil.which("docker")
        if docker is None:
            return {}
        captured: Dict[str, str] = {}
        for row in rows:
            if not isinstance(row, dict):
                continue
            service = str(row.get("Service", row.get("service", ""))).strip()
            container_id = str(row.get("ID", row.get("Id", row.get("id", "")))).strip()
            if not service or not container_id:
                continue
            inspected = subprocess.run(
                [docker, "inspect", "--format", "{{.Image}}", container_id],
                capture_output=True, check=False, text=True, timeout=30,
            )
            image_id = inspected.stdout.strip()
            if inspected.returncode != 0 or not re.fullmatch(r"sha256:[a-f0-9]{64}", image_id):
                return {}
            captured[service] = image_id
        return captured

    @staticmethod
    def _reject_secret_lines(content: str):
        reject_secret_lines(content)

    @staticmethod
    def _reject_unsafe_compose_keys(content: str):
        reject_unsafe_keys(content)

    def _validate_normalized_compose(self, normalized: Dict[str, Any], project: Path):
        validate_normalized(normalized, self.workloads_root)
