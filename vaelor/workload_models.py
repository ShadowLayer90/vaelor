"""Which models are downloaded on this appliance, and which are serving right now.

Extracted from :mod:`vaelor.workload_inventory` (whose ``WorkloadInventory``
inherits :class:`WorkloadModelsMixin`, so every caller and patch target is
unchanged). Runs in two processes: the control plane (the Manage > Downloaded
models listing, reading Docker through the workload broker) and the workload
executor (the removal's re-validation, reading Docker directly).

**ACC-106.** The listing said the running GPU AI Chat model was merely "ready"
and let it be removed, never listed the on-device (NPU) model, and called a
STOPPED container's model "In use" because a compose file named it. Now:

* ``in_use`` is measured - a RUNNING container is serving the file: the
  Assistant's llama.cpp (``model-assistant``), AI Chat's (``model-chat``), or
  the GPU AI Chat server the hardware bridge runs (``vaelor-gpu-rocmfpx``,
  whose model is read off its mount and its command line); ``served_by`` names
  which;
* ``in_use_known`` is ``False`` when Docker could not be read, so an unknown is
  never shown as "not in use";
* ``selected`` is the old compose-text match: configured for the Assistant,
  running or not;
* the NPU models under FastFlowLM's models folder are listed too, marked not
  removable here, with what the NPU is serving read from the hardware bridge.
"""

from __future__ import annotations

import hashlib
import json
import os
import re
import shutil
import subprocess
from collections.abc import Mapping
from pathlib import Path
from typing import Any, Callable, Optional

from . import workload_broker
# Every id a managed-local deploy writes, old and marked (VD-202), is answered
# by `model_credential_roles`; this module kept its own regex once and lost the
# marked ids (LESSONS 6 / 14).
from .model_credential_roles import is_managed_local_credential_id
#: The most ids one admitted ``docker inspect`` may name (`workload_broker`).
INSPECT_BATCH = 100

#: The servers ``served_by`` names.
ASSISTANT_SERVER = "assistant"
AI_CHAT_SERVER = "ai-chat"
GPU_AI_CHAT_SERVER = "gpu-ai-chat"
NPU_ASSISTANT_SERVER = "npu-assistant"

#: The compose projects whose ``llama-server`` serves a managed model, and who.
_COMPOSE_SERVERS = {"model-assistant": ASSISTANT_SERVER, "model-chat": AI_CHAT_SERVER}

#: Why an NPU model has no Remove here.
NPU_NOT_REMOVABLE = (
    "Installed and managed by Vaelor's hardware service for the on-device "
    "Assistant, so it is not removed from this list."
)

#: The most files an NPU model folder is walked for its size.
_NPU_WALK_LIMIT = 2000


def _host_path(record: Mapping[str, Any], container_path: str) -> str:
    """The host file a container path names, through the container's bind mounts."""
    for mount in record.get("Mounts") or []:
        if not isinstance(mount, Mapping) or mount.get("Type") != "bind":
            continue
        destination = str(mount.get("Destination", "")).rstrip("/")
        if destination and container_path.startswith(destination + "/"):
            relative = container_path[len(destination) + 1:]
            if ".." in relative.split("/"):
                return ""
            return str(Path(str(mount.get("Source", ""))) / relative)
    return ""


def model_servers(records: list[dict[str, Any]], workloads_root: Path) -> list[dict[str, Any]]:
    """``[{host_path, running, server}]`` - every container serving a model file.

    Read from the inspect records: a managed compose ``llama-server`` names its
    model in ``LLAMA_ARG_MODEL``; the GPU AI Chat server names it on its
    command line (``-m /models/...`` for the standard engine, the positional
    ``/app/models/...`` for the FP4 fork). Both are mapped to the host through
    the container's own bind mount, so the path is what Docker mounted, not
    what a file claims.
    """
    from .gpu_rocmfpx_service import GPU_CONTAINER_NAME

    servers: list[dict[str, Any]] = []
    for record in records or []:
        config = record.get("Config") or {}
        labels = config.get("Labels") or {}
        running = bool((record.get("State") or {}).get("Running"))
        project = labels.get("com.docker.compose.project")
        if project in _COMPOSE_SERVERS and labels.get("com.docker.compose.service") == "llama-server":
            working = str(labels.get("com.docker.compose.project.working_dir", ""))
            try:
                Path(working).resolve().relative_to(workloads_root)
            except (OSError, ValueError):
                continue
            environment = dict(
                str(item).partition("=")[::2] for item in config.get("Env") or []
            )
            host = _host_path(record, str(environment.get("LLAMA_ARG_MODEL", "")))
            if host:
                servers.append({"host_path": host, "running": running,
                                "server": _COMPOSE_SERVERS[project]})
        elif str(record.get("Name", "")).lstrip("/") == GPU_CONTAINER_NAME:
            arguments = [str(item) for item in (record.get("Args") or config.get("Cmd") or [])]
            for argument in arguments:
                host = _host_path(record, argument) if argument.lower().endswith(".gguf") else ""
                if host:
                    servers.append({"host_path": host, "running": running,
                                    "server": GPU_AI_CHAT_SERVER})
                    break
    return servers


def _same_file(first: str, second: str) -> bool:
    try:
        return Path(first).resolve() == Path(second).resolve()
    except OSError:
        return first == second


def usage_fields(path: Path, servers: list[dict[str, Any]] | None) -> dict[str, Any]:
    """``in_use``, ``in_use_known`` and ``served_by`` for one downloaded file."""
    if servers is None:
        return {"in_use": False, "in_use_known": False, "served_by": []}
    serving = sorted({
        server["server"] for server in servers
        if server["running"] and _same_file(server["host_path"], str(path))
    })
    return {"in_use": bool(serving), "in_use_known": True, "served_by": serving}


def _normalized(text: str) -> str:
    return re.sub(r"[^a-z0-9]", "", str(text or "").lower())


#: Why an on-device model's size is not shown (W4d-D3): an observer that
#: cannot read the folder has measured nothing, so it says so, not "0 B".
NPU_SIZE_UNREADABLE = (
    "The size could not be read: Vaelor's hardware service keeps this model's "
    "folder private to itself ({})."
)


def npu_models(root: Path | None, status: Callable[[], Any] | None) -> list[dict[str, Any]]:
    """The on-device (NPU) models installed under FastFlowLM's models folder.

    One row per model folder, sized from its files. ``in_use`` comes from the
    hardware bridge's ``flm_status`` (running, and the tag it serves): the one
    folder when there is one, else the folder whose name the tag names. A
    status that cannot be read, or a tag that names no single folder, is
    ``in_use_known`` false - never a guess.
    """
    if root is None:
        from .flm_service import FLM_MODEL_PATH

        root = Path(FLM_MODEL_PATH) / "models"
    try:
        folders = sorted(child for child in root.iterdir() if child.is_dir())
    except OSError:
        return []
    if not folders:
        return []
    reading = None
    if status is not None:
        try:
            reading = status() or {}
        except Exception:  # noqa: BLE001 - an unanswered bridge is "not known"
            reading = None
    serving_folder = None
    known = isinstance(reading, Mapping)
    if known and reading.get("running"):
        tag = _normalized(reading.get("tag", ""))
        named = [folder for folder in folders if tag and _normalized(folder.name).startswith(tag)]
        if len(folders) == 1:
            serving_folder = folders[0]
        elif len(named) == 1:
            serving_folder = named[0]
        else:
            known = False
    rows = []
    for folder in folders:
        size: Optional[int] = 0
        newest = 0
        size_reason = ""
        try:
            # W4d-D3 (LESSONS 4, 8): `rglob` yields nothing for a folder this
            # process cannot list (FastFlowLM's are root 0770), and that empty
            # walk was shown as "0 B" for 3.4 GB. Asked first, and said.
            with os.scandir(folder):
                pass
        except OSError as error:
            size, size_reason = None, NPU_SIZE_UNREADABLE.format(error.strerror or error)
        for count, item in enumerate(folder.rglob("*") if size is not None else ()):
            if count >= _NPU_WALK_LIMIT:
                break
            try:
                if item.is_file():
                    stat = item.stat()
                    size = int(size or 0) + stat.st_size
                    newest = max(newest, stat.st_mtime_ns)
            except OSError:
                continue
        in_use = folder == serving_folder
        rows.append({
            "id": hashlib.sha256(("npu:" + folder.name).encode()).hexdigest()[:16],
            "name": folder.name,
            "file": folder.name,
            "path": str(folder),
            "sha256": "",
            "sha256_known": False,
            "size_bytes": size,
            **({"size_reason": size_reason} if size_reason else {}),
            "mtime_ns": newest,
            "modified_at": int(newest / 1_000_000),
            "status": "ready",
            "status_reason": NPU_NOT_REMOVABLE,
            "catalog": True,
            "surface": "assistant",
            "kind": "npu",
            "removable": False,
            "in_use": in_use,
            "in_use_known": known,
            "served_by": [NPU_ASSISTANT_SERVER] if in_use else [],
            "selected": False,
        })
    return rows


#: Why a downloaded model is not removed, one sentence per cause (ACC-106).
REMOVAL_REFUSALS = {
    "npu": (
        "This is the on-device (NPU) Assistant model. Vaelor's hardware "
        "service installed it and manages it, so it cannot be removed here."
    ),
    "unknown": (
        "Vaelor could not check whether this model is running right now, so "
        "it was not removed. Try again in a moment."
    ),
    "ai-chat": (
        "This model is serving AI Chat right now. Choose a different AI Chat "
        "model first, then remove this one."
    ),
}


def model_removal_refusal(model: Mapping[str, Any]) -> str:
    """The plain reason a listed model cannot be removed, or ``""``.

    The one rule the removal plan and its re-validation both apply: an NPU
    model is not Vaelor's to remove here; a model whose use could not be read
    is not removed on a guess; a model serving AI Chat is refused outright,
    because the only cascade the removal knows stops the ASSISTANT's runtime
    and would leave AI Chat's server holding a deleted file. The Assistant's
    own model keeps its reviewed cascade (`workload_dependencies`).
    """
    if model.get("kind") == "npu" or model.get("removable") is False:
        return REMOVAL_REFUSALS["npu"]
    if model.get("in_use_known") is False:
        return REMOVAL_REFUSALS["unknown"]
    served = set(model.get("served_by") or [])
    if served & {AI_CHAT_SERVER, GPU_AI_CHAT_SERVER}:
        return REMOVAL_REFUSALS["ai-chat"]
    return ""


class WorkloadModelsMixin:
    """The model half of `WorkloadInventory`: its ``_models`` and helpers."""

    def _managed_model_assignment(self) -> dict[str, Any]:
        assignment: dict[str, Any] = {"active_for": []}
        if self.credential_broker is None:
            return assignment
        for purpose in ("deployment-agent", "ai-chat"):
            try:
                lease = self.credential_broker.resolve_active(purpose)
            except (OSError, RuntimeError, ValueError):
                continue
            credential_id = str((lease or {}).get("credential_id", ""))
            if not is_managed_local_credential_id(credential_id):
                continue
            if assignment.get("credential_id") not in (None, credential_id):
                continue
            assignment["credential_id"] = credential_id
            assignment["active_for"].append(purpose)
            selected = str(
                (lease or {}).get("model") or (lease or {}).get("selected_model") or ""
            ).strip()
            if selected:
                assignment["model"] = selected
            endpoint = str((lease or {}).get("base_url", "")).strip()
            if endpoint:
                assignment["endpoint"] = endpoint
        return assignment

    def _ai_chat_model_stem(self) -> str:
        """The file stem of the model currently serving AI Chat, or ``""``.

        Read from the DURABLE ai-chat credential lease the GPU deploy writes: its
        label carries the served model's file stem
        (``MANAGED_LOCAL_CREDENTIAL_LABEL`` = ``"Managed local model · <stem>"``).
        This is the one signal that lets an OFF-catalog user ``.gguf`` deployed as
        the GPU AI-Chat model be reported with ``surface`` ``"ai-chat"`` (see
        :meth:`_models`), so re-activating it from the Manage panel routes it back
        to the GPU chat tier rather than defaulting to the Assistant - the same
        stem-in-the-label the boot reconcile resolves the model by, read here so
        the inventory and the reconcile agree on which model IS chat.

        ``""`` when there is no managed-local ai-chat lease at all (a hosted
        provider, an NPU-only box, or no assignment), so a genuine Assistant model
        is never relabelled off this path.
        """
        if self.credential_broker is None:
            return ""
        try:
            lease = self.credential_broker.resolve_active("ai-chat")
        except Exception:
            # This enrichment fills a blank surface; it must never keep a listing
            # (or the boot reconcile that drives it) from returning. Any broker
            # failure - no lease (CredentialError), an unreachable socket, or a
            # host with no AF_UNIX at all - degrades to "" (the pre-enrichment
            # value), so the model is simply surfaced by the catalog alone.
            return ""
        credential_id = str((lease or {}).get("credential_id", ""))
        if not is_managed_local_credential_id(credential_id):
            return ""
        from .executor_model_deploy import MANAGED_LOCAL_CREDENTIAL_LABEL

        prefix = MANAGED_LOCAL_CREDENTIAL_LABEL.format("")
        label = str((lease or {}).get("label") or "")
        return label[len(prefix):] if label.startswith(prefix) else ""

    def _container_records(self) -> list[dict[str, Any]] | None:
        """Every container's ``docker inspect`` record, or ``None`` if unreadable.

        ACC-106: the exact two reads the workload broker admits for the
        control plane (``docker ps -aq --no-trunc``, then ``docker inspect`` of
        up to 100 ids, repeated in batches of 100 so a box with more
        containers still has its serving model seen) - the model runtime used
        to add a ``--filter`` the broker refuses, so on the appliance the
        running model was never seen and "In use" came from a compose file's
        text instead. ``None`` means the answer is not known, never "nothing is
        running"; one batch that cannot be read makes the whole answer unknown.
        """
        if shutil.which("docker") is None:
            return None
        try:
            listed = self._run(["docker", "ps", "-aq", "--no-trunc"])
        except (OSError, RuntimeError, subprocess.SubprocessError):
            return None
        if listed.returncode:
            return None
        ids = [
            item for item in listed.stdout.splitlines()
            if workload_broker.APP_ID.fullmatch(item)
        ]
        records: list[dict[str, Any]] = []
        for start in range(0, len(ids), INSPECT_BATCH):
            batch = self._inspect(ids[start:start + INSPECT_BATCH])
            if batch is None:
                return None
            records += batch
        return records

    def _inspect(self, ids: list[str]) -> list[dict[str, Any]] | None:
        """One admitted ``docker inspect`` of at most :data:`INSPECT_BATCH` ids."""
        try:
            inspected = self._run(["docker", "inspect", *ids], timeout=12)
        except (OSError, RuntimeError, subprocess.SubprocessError):
            return None
        if inspected.returncode:
            return None
        try:
            records = json.loads(inspected.stdout)
        except (TypeError, json.JSONDecodeError):
            return None
        return [record for record in records if isinstance(record, Mapping)] if isinstance(records, list) else None

    def _managed_model_runtime(self, records: list[dict[str, Any]] | None = None) -> dict[str, Any] | None:
        """Resolve one managed llama.cpp identity from Docker, not job history."""
        if records is None:
            records = self._container_records()
        if not records:
            return None
        assignment = self._managed_model_assignment()
        for record in records if isinstance(records, list) else []:
            labels = (record.get("Config") or {}).get("Labels") or {}
            state = record.get("State") or {}
            if (
                labels.get("com.docker.compose.project") != "model-assistant"
                or labels.get("com.docker.compose.service") != "llama-server"
            ):
                continue
            working = str(labels.get("com.docker.compose.project.working_dir", ""))
            if not working or not self._inside(Path(working), self.workloads_root):
                continue
            environment = {}
            for item in (record.get("Config") or {}).get("Env") or []:
                key, separator, value = str(item).partition("=")
                if separator:
                    environment[key] = value
            container_model = environment.get("LLAMA_ARG_MODEL", "")
            if not container_model.startswith("/models/"):
                continue
            host_path = None
            for mount in record.get("Mounts") or []:
                if mount.get("Type") != "bind" or mount.get("Destination") != "/models":
                    continue
                candidate = Path(str(mount.get("Source", ""))) / Path(container_model).name
                try:
                    candidate.resolve().relative_to(Path(str(mount.get("Source", ""))).resolve())
                except (OSError, ValueError):
                    continue
                host_path = candidate
                break
            name = Path(container_model).stem
            selected = str(assignment.get("model", ""))
            selected_name = Path(selected).stem if selected else ""
            if selected_name and selected_name != name:
                assignment["identity_warning"] = (
                    "The active credential reports a different model from the managed runtime."
                )
            return {
                "container_id": str(record.get("Id", "")),
                "container_name": str(record.get("Name", "")).lstrip("/"),
                "project": "model-assistant",
                "service": "llama-server",
                "running": bool(state.get("Running")),
                "health": (state.get("Health") or {}).get("Status"),
                "container_model": container_model,
                "host_path": str(host_path) if host_path is not None else "",
                "name": name,
                **assignment,
            }
        return None

    def _models(self) -> list[dict[str, Any]]:
        from .workload_inventory import model_file_summary

        records = self._container_records()
        runtime = self._managed_model_runtime(records)
        servers = model_servers(records, self.workloads_root) if records is not None else None
        # The file the durable ai-chat lease names, so an off-catalog GPU chat
        # model is surfaced as ``ai-chat`` and re-activates onto the chat tier.
        ai_chat_stem = self._ai_chat_model_stem()
        try:
            active_config = (
                self.workloads_root / "model-assistant" / "compose.yaml"
            ).read_text(encoding="utf-8")
        except OSError:
            active_config = ""
        models = []
        # W4d-D13/D28: asked at most once, and only for a catalog Assistant
        # GGUF - the deploy's own rule (`deploy_surface`) decides the label.
        npu_reading: list = []

        gpu_reading: list = []

        def gpu_facts() -> tuple:
            """(the fork serves AI Chat here, Mode B holds it), read once."""
            if not gpu_reading:
                from .copilot_setup import hardware_inventory
                from .gpu_cluster_mode_state import ClusterModeStore
                from .gpu_model_choice import discover_gpu_rocm_serving
                from .gpu_serving_target import gpu_cluster_mode_active

                try:
                    fork = bool(discover_gpu_rocm_serving(hardware_inventory()).get("available"))
                except Exception:  # noqa: BLE001 - unknown: the profiles stay offered
                    fork = False
                try:
                    clustered = bool(gpu_cluster_mode_active(ClusterModeStore().read()))
                except Exception:  # noqa: BLE001 - the deploy refuses for itself
                    clustered = False
                gpu_reading.append((fork, clustered))
            return gpu_reading[0]

        def npu_holds_assistant() -> bool:
            if not npu_reading:
                from .flm_supervisor import npu_serves_assistant

                try:
                    npu_reading.append(npu_serves_assistant())
                except Exception:  # noqa: BLE001 - a listing never fails on it
                    npu_reading.append(False)
            return npu_reading[0]

        paths = list(self.models_root.rglob("*.gguf"))[:200] if self.models_root.exists() else []
        runtime_path = Path(str((runtime or {}).get("host_path", "")))
        if runtime_path.is_file() and all(
            path.resolve() != runtime_path.resolve() for path in paths
        ):
            paths.append(runtime_path)
        for path in paths:
            path = path.resolve()
            if not self._inside(path, self.models_root):
                if not runtime or path.resolve() != runtime_path.resolve():
                    continue
            try:
                # Stat, not SHA-256. See `model_file_summary`: hashing here
                # cost 68 s of the 65-100 s this listing took on the appliance,
                # for a digest no caller of this endpoint reads.
                identity = model_file_summary(path)
            except (OSError, ValueError):
                continue
            relative = (
                str(path.relative_to(self.models_root))
                if self._inside(path, self.models_root) else path.name
            )
            runtime_match = bool(
                runtime and (
                    path.resolve() == runtime_path.resolve()
                    or path.name == Path(str(runtime.get("container_model", ""))).name
                )
            )
            # #147: seven files on disk, three in the catalog, and every row
            # offered "Use model". The extra four are evaluation candidates
            # and the fine-tune — present because we put them there, not
            # because an owner installed them. A file the catalog does not
            # name has no verified identity and no measured footprint, so it
            # is listed with why it is here, and is not offerable.
            from .model_footprint import identify_by_file
            from .model_catalog import catalog_surface_by_file, deploy_surface

            # Which tier "Use model" switches. The catalog names it for a stocked
            # file; an OFF-catalog file resolves to "" there, so a user .gguf that
            # was deployed as the GPU AI-Chat model would default back to the
            # Assistant on re-activation. Fill that blank from the durable ai-chat
            # lease: the file the lease names IS chat, so it is surfaced as such.
            # Only the empty catalog surface is filled - a genuine Assistant/NPU
            # model keeps the surface the catalog gives it.
            surface = catalog_surface_by_file(path.name)
            if not surface and ai_chat_stem and path.stem[:48] == ai_chat_stem:
                surface = "ai-chat"
            if surface:
                # The same rule the deploy applies, so the switch panel's title
                # and the surface it sends back are what the deploy will serve.
                surface = deploy_surface(
                    "", surface, names_file=True,
                    npu_holds_assistant=npu_holds_assistant,
                )

            # W4d-D28: what the switch will run, and whether it may run now.
            serving = {}
            if surface == "ai-chat":
                from .executor_model_deploy import _artifact_identity
                from .gpu_model_choice import gpu_chat_context
                from .gpu_serving_target import AI_CHAT_HELD_BY_CLUSTER

                fork, clustered = gpu_facts()
                context = gpu_chat_context(_artifact_identity(path), fork)
                if context:
                    serving["serving_context"] = context
                if clustered:
                    serving["switch_refusal"] = AI_CHAT_HELD_BY_CLUSTER

            models.append(
                {
                    **serving,
                    "id": hashlib.sha256(relative.encode()).hexdigest()[:16],
                    "name": path.stem,
                    "file": relative,
                    **identity,
                    "modified_at": int(identity["mtime_ns"] / 1_000_000),
                    "status": "ready",
                    "catalog": identify_by_file(path.name) is not None,
                    # Which tier "Use model" switches - the GPU AI-Chat model or
                    # the NPU Assistant's. "" when the catalog does not name the
                    # file. The UI labels the switch by this and sends it back on
                    # the deploy so a GPU chat model is never routed to the
                    # Assistant surface by a resolution miss.
                    "surface": surface,
                    # ACC-106: "In use" is a RUNNING container serving this
                    # file (the Assistant's, AI Chat's compose, or the GPU AI
                    # Chat server), measured from Docker; the compose file
                    # naming it is `selected`, a configuration and not a use.
                    **usage_fields(path, servers),
                    "selected": str(path.parent) in active_config and path.name in active_config,
                    **({"runtime": runtime} if runtime_match else {}),
                }
            )
        if runtime and not any(item.get("runtime") for item in models):
            models.append({
                "id": hashlib.sha256(
                    ("runtime:" + str(runtime.get("container_model", ""))).encode()
                ).hexdigest()[:16],
                "name": runtime["name"],
                "file": Path(str(runtime.get("container_model", ""))).name,
                "path": str(runtime.get("host_path", "")),
                "size_bytes": 0,
                "modified_at": 0,
                "sha256": "",
                "mtime_ns": 0,
                "status": "degraded",
                "status_reason": "The running model file is not readable from managed storage.",
                "in_use": bool(runtime.get("running")),
                "in_use_known": True,
                "served_by": [ASSISTANT_SERVER] if runtime.get("running") else [],
                "selected": True,
                "runtime": runtime,
            })
        models.extend(npu_models(self.npu_models_root, self.npu_status))
        return sorted(models, key=lambda item: item["name"].lower())
