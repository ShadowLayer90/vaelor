"""Privileged, approval-gated head-controller cluster operations."""

from __future__ import annotations

import json
import re
import urllib.request
from pathlib import Path
from typing import Any, Callable, Dict, Optional

# The single/replicated readiness probe moved to `cluster_llm_deploy` with the
# rest of that branch, but it still calls `urllib.request.urlopen` on the shared
# module object. This import keeps `cluster_operations.urllib` resolvable so the
# deploy tests' `@patch("vaelor.cluster_operations.urllib.request.urlopen")`
# seam - which reaches the same shared object - continues to intercept it.
_ = urllib.request

from .cluster_architecture import controller_architecture, join_compatibility
from .cluster_backups import ClusterBackupStore
from .cluster_capacity import gpu_facts_from_accelerators
from .cluster_link import link_for_split
from .gpu_node_facts import local_cluster_interface
from .cluster_driver import ClusterDriverError, DockerSwarmDriver
from .cluster_llm_deploy import (
    deploy_single_or_replicated,
    refuse_out_of_service_nodes,
)
from .cluster_eviction import ArchitectureEviction
from .cluster_node_removal import remove_worker_node
from .cluster_store import NODE_NOT_FOUND, ClusterStore
from .credential_broker import CredentialBrokerClient
from .ssh_transport import SshTransport
from .cluster_app_deploy import (
    deploy_catalog_app,
    remove_controller_label_if_unused,
)
from .cluster_app_multideploy import deploy_researched_app as _multideploy_app
from .cluster_app_multideploy import remove_researched_app as _remove_multideploy_app
from . import cluster_retained_volumes, cluster_service_backup
from .cluster_service_reconfigure import (
    build_label_constraints,
    cpu_reservation_cores,
    reconcile_service_change,
    validate_cpu_limit,
)
from .cluster_service_state import (
    STATEFUL_DRAIN_ACK, REMOVAL_WITHOUT_BACKUP, STATEFUL_REMOVE_ACK,
    live_service_names, require_data_loss_ack, services_needing_backup,
)
from .pooled_runtime import PooledRuntime
from .pooled_operations import PooledDeploymentOperations
from .gpu_cluster_mode_watch import reconcile_gpu_cluster_mode as _mode_watch_pass
from .gpu_model_library import GpuModelLibrary
from .gpu_pool_operations import ASSISTANT_NOT_CLUSTERED, GpuPoolOperations
from .gpu_pool_reload import load_deployment, unload_deployment
from .gpu_pool_refresh import refresh_deployment
from .cluster_agent_operations import ClusterAgentOperationsMixin
from .gpu_pool_runtime import GpuPoolRuntime
from .gpu_serving_target import controller_gpu_chat_reclaimable
from .platform_drivers import default_platform_drivers
from .copilot_setup import hardware_inventory
from .cluster_placement import CONTROLLER_PLACEMENT_ID
from .cluster_network import private_controller_ipv4
from .runtime_paths import data_path


class ClusterOperations(ClusterAgentOperationsMixin):
    def __init__(
        self,
        store: Optional[ClusterStore] = None,
        broker: Optional[CredentialBrokerClient] = None,
        driver: Optional[DockerSwarmDriver] = None,
        pooled_runtime: Optional[PooledRuntime] = None,
        backup_store: Optional[ClusterBackupStore] = None,
        os_driver=None,
        inventory_probe=None,
        gpu_mode_switch=None,
        application_deployments=None,
        workloads_root=None,
    ):
        self.store = store or ClusterStore()
        self.broker = broker or CredentialBrokerClient(timeout_seconds=30)
        self.driver = driver or DockerSwarmDriver(timeout=600)
        self.pooled_runtime = pooled_runtime or PooledRuntime()
        self.backup_store = backup_store or ClusterBackupStore()
        self.os_driver = (
            os_driver
            or default_platform_drivers()["operating_system"]
        )
        self.inventory_probe = inventory_probe or hardware_inventory
        # The approved-draft store D4a resolves through (executor-injected only).
        self.application_deployments = application_deployments
        self.workloads_root = Path(workloads_root or data_path("workloads"))
        # VD-125: the GPU serving mode switch, OPTIONAL by design (D12). The
        # control plane builds this class too, and a preview-side instance must
        # never stop llama.cpp or move AI Chat's lease; only the EXECUTOR - the
        # process that holds the 30 s GPU failure-watch and can drive the root
        # bridge - injects a real one, and everything below is gated on it.
        self.gpu_mode_switch = gpu_mode_switch
        self._controller_architecture = None
        self.pooled_operations = PooledDeploymentOperations(
            store=self.store,
            broker=self.broker,
            runtime=self.pooled_runtime,
            joined_node=self._joined_node,
            advertise_address=self._advertise_address,
        )
        # The weights-on-the-box layer (Phase B2): pull/remove model caches on
        # selected nodes and read the per-node cache inventory. It reuses this
        # controller's store, broker, and node resolution, mirroring how
        # `pooled_operations` is composed; the executor's cluster job dispatch
        # and the `GET /cluster/models` route both reach it here.
        # VD-125: its targets include this controller, so it resolves nodes
        # through `_gpu_placement_node` like the serving layer does - a machine
        # that can serve a model has to be able to cache one.
        self.model_library = GpuModelLibrary(
            store=self.store,
            broker=self.broker,
            joined_node=self._gpu_placement_node,
            advertise_address=self._advertise_address,
        )
        # The GPU serving layer (Phase B3a): deploy and remove vLLM (single-GPU
        # or Ray/RCCL distributed) inference on selected workers. Composed
        # exactly as `pooled_operations` is, reusing this controller's store,
        # broker, and node resolution; `deploy_llm` delegates the ``gpu`` mode
        # here and the `cluster.gpu.remove` job reaches it through
        # `remove_gpu_inference`.
        # VD-125: its participants are the enrolled GPU workers AND this
        # controller, so it resolves nodes through `_gpu_placement_node` rather
        # than `_joined_node` - the controller is not a Swarm worker and never
        # will be, but its GPU is half of a two-node cluster.
        self.gpu_runtime = GpuPoolRuntime()
        self.gpu_operations = GpuPoolOperations(
            store=self.store,
            broker=self.broker,
            runtime=self.gpu_runtime,
            joined_node=self._gpu_placement_node,
            advertise_address=self._advertise_address,
            mode_switch=self.gpu_mode_switch,
            # VD-162: the owner's chosen cluster link, read per split deploy.
            split_link=lambda: link_for_split(self.store),
        )
        # VD-033. Reuses `remove_node`, so an evicted host loses its stored SSH
        # credential exactly as a refused one does.
        self.architecture_eviction = ArchitectureEviction(
            store=self.store,
            controller_architecture=self.controller_architecture,
            remove_node=self.remove_node,
        )

    @classmethod
    def for_executor(
        cls, application_deployments=None, workloads_root=None
    ) -> "ClusterOperations":
        """This class as the WORKLOAD EXECUTOR builds it: with a real mode switch.

        The one place the GPU serving mode switch is injected (VD-125, D12). The
        executor is the process that holds the 30 s GPU failure-watch and can
        drive the root bridge, so it is the only one that can stop llama.cpp and
        be sure it stayed stopped. Every other construction of this class - the
        control plane's above all - gets no switch, and its GPU operations
        therefore never move AI Chat's lease or touch the GPU model.
        """
        from .gpu_cluster_mode_watch import executor_mode_switch

        return cls(
            gpu_mode_switch=executor_mode_switch(),
            application_deployments=application_deployments,
            workloads_root=workloads_root,
        )

    @staticmethod
    def _advertise_address(value: Any) -> str:
        return private_controller_ipv4(value)

    def initialize(self, payload: Dict[str, Any]) -> Dict[str, Any]:
        if payload.get("confirm") != "initialize-head-controller":
            raise ValueError("Confirm head-controller initialization.")
        address = self._advertise_address(payload.get("advertise_address", ""))
        status = self.driver.status()
        result = (
            status
            if status.get("initialized") and status.get("control_available")
            else self.driver.initialize(address)
        )
        self.store.set_controller({
            "initialized": True,
            "cluster_id": result.get("node_id", ""),
            "advertise_address": address,
        })
        return result

    def join_node(self, payload: Dict[str, Any]) -> Dict[str, Any]:
        if payload.get("confirm") != "join-worker-node":
            raise ValueError("Confirm worker-node joining.")
        node = self.store.get_node(str(payload.get("node_id", "")), include_credential=True)
        if node is None:
            raise ValueError(NODE_NOT_FOUND)
        controller = self.store.controller()
        if not controller.get("initialized") or not controller.get("advertise_address"):
            raise ValueError("Initialize the head controller before joining workers.")
        existing_swarm_id = str(node.get("labels", {}).get("swarm_node_id", ""))
        if node.get("state") == "joined" and existing_swarm_id:
            runtime_ids = {
                str(item.get("id", "")) for item in self.driver.status().get("nodes", [])
            }
            if existing_swarm_id in runtime_ids:
                return {
                    "joined": True,
                    "already_joined": True,
                    "node_id": node["id"],
                    "swarm_node_id": existing_swarm_id,
                    "labels": node.get("labels", {}),
                }
        inventory = node.get("inventory", {})
        self._validate_worker(
            inventory, where=str(node.get("host") or node.get("name") or "")
        )
        profile = self.broker.resolve(node["credential_id"], "cluster-node")
        result = self.driver.join_worker(
            SshTransport(profile, timeout=120),
            controller["advertise_address"],
            install_docker=not bool(inventory.get("docker")),
        )
        labels = {
            "vaelor.node_id": node["id"],
            "vaelor.architecture": str(inventory.get("architecture", "unknown")),
            "vaelor.memory_tier": self._memory_tier(
                inventory.get("memory_bytes", 0)
            ),
            # Retained for one compatibility window so existing Swarm state
            # remains inspectable during the Vaelor migration.
            "pironman.node_id": node["id"],
            "pironman.architecture": str(inventory.get("architecture", "unknown")),
            "pironman.memory_tier": self._memory_tier(inventory.get("memory_bytes", 0)),
        }
        self.driver.label_node(result["swarm_node_id"], labels)
        stored_labels = {**labels, "swarm_node_id": result["swarm_node_id"]}
        self.store.update_node(
            node["id"], state="joined", role="worker", labels=stored_labels
        )
        return {**result, "node_id": node["id"], "labels": stored_labels}

    #: Architectures Vaelor runs cluster workloads on at all. Membership here
    #: does not mean two of them may share a cluster — VD-031 keeps one
    #: architecture per fleet, and that is checked separately below.
    SUPPORTED_WORKER_ARCHITECTURES = frozenset(
        {"aarch64", "arm64", "x86_64", "amd64"}
    )

    def controller_architecture(self) -> str:
        """This controller's architecture class, from its own hardware probe.

        Memoised: a replicated deployment validates up to eight workers in one
        call, and the probe behind this reads accelerators and disk usage. The
        machine does not change silicon while the process is running.
        """
        if getattr(self, "_controller_architecture", None) is None:
            try:
                self._controller_architecture = controller_architecture(
                    self.inventory_probe()
                )
            except (AttributeError, OSError, TypeError, ValueError):
                self._controller_architecture = ""
        return self._controller_architecture

    def _validate_worker(
        self, inventory: Dict[str, Any], *, where: str = ""
    ) -> None:
        # VD-031 is checked here rather than only at enrollment because a node
        # admitted before the rule existed is still in the store, and quietly
        # scheduling onto it is the same failure as admitting it.
        verdict = join_compatibility(
            self.controller_architecture(),
            inventory.get("architecture"),
            where=where,
        )
        if not verdict["compatible"]:
            raise ValueError(verdict["reason"])
        compatibility = self.os_driver.worker_compatibility(
            inventory, set(self.SUPPORTED_WORKER_ARCHITECTURES)
        )
        if not compatibility["compatible"]:
            raise ValueError(compatibility["reason"])
        if int(inventory.get("memory_bytes", 0) or 0) < 1024 ** 3:
            raise ValueError("The worker needs at least 1 GB of physical memory.")
        free_bytes = int(inventory.get("root_free_bytes", 6 * 1024 ** 3) or 0)
        if free_bytes < 5 * 1024 ** 3:
            raise ValueError("Free at least 5 GB on the worker before joining it.")

    def set_node_availability(self, payload: Dict[str, Any]) -> Dict[str, Any]:
        availability = str(payload.get("availability", "")).lower()
        expected = f"set-worker-{availability}"
        if availability not in {"active", "pause", "drain"} or payload.get("confirm") != expected:
            raise ValueError("Confirm the reviewed worker availability change.")
        node = self._joined_node(payload.get("node_id"))
        # GATE (item 2): a drain of a pinned stateful app with no backup is a
        # data loss with no migration path, so it needs the typed ack.
        if availability == "drain" and self._stateful_names_needing_backup(
            pinned_to=str(node["id"])
        ):
            require_data_loss_ack(
                expected_ack=STATEFUL_DRAIN_ACK,
                provided_ack=payload.get("data_loss_ack"),
                subject="This worker hosts a pinned app that",
            )
        swarm_node_id = str(node["labels"]["swarm_node_id"])
        result = self.driver.set_node_availability(swarm_node_id, availability)
        state = "joined" if availability == "active" else availability
        self.store.update_node(node["id"], state=state)
        return {**result, "node_id": node["id"], "state": state}

    def remove_node(self, payload: Dict[str, Any]) -> Dict[str, Any]:
        # ACC-117: refused while a deployment names the node, telemetry agent
        # taken off first, what stays said - `cluster_node_removal` owns it.
        return remove_worker_node(self, payload)

    def deploy_llm(self, payload: Dict[str, Any], progress=None) -> Dict[str, Any]:
        if payload.get("confirm") != "deploy-cluster-llm":
            raise ValueError("Confirm the reviewed cluster LLM deployment.")
        if payload.get("use_for_assistant"):
            # VD-049 / VD-202 item 1: no cluster model serves the Assistant, in
            # any mode. Refused here, before any mode is chosen or work starts.
            raise ValueError(ASSISTANT_NOT_CLUSTERED)
        deployment_mode = str(
            payload.get("deployment_mode", "single")
        ).strip().lower()
        # `pooled` (CPU distributed-llama) and `gpu` (vLLM) each own their whole
        # lifecycle in their operations module and reuse this same
        # `cluster.llm.deploy` job type and its `deploy-cluster-llm` confirm gate.
        # The single/replicated CPU-Swarm path is housed in `cluster_llm_deploy`
        # to keep this module under the line ceiling.
        if deployment_mode in ("pooled", "gpu"):
            # ACC-118 / SC1: both SSH-run paths ask the drain question too,
            # of nodes resolved by their own resolver, against the live list.
            path = (
                self.pooled_operations if deployment_mode == "pooled"
                else self.gpu_operations
            )
            raw_ids = payload.get("node_ids")
            refuse_out_of_service_nodes(
                [
                    path.joined_node(node_id)
                    for node_id in (raw_ids if isinstance(raw_ids, list) else [])
                    if str(node_id) != CONTROLLER_PLACEMENT_ID
                ],
                self.driver.status(),
            )
            return path.deploy(payload, progress)
        return deploy_single_or_replicated(
            self, payload, deployment_mode, progress
        )

    def deploy_app(self, payload: Dict[str, Any], progress=None) -> Dict[str, Any]:
        # The placement (run-once/spread/pin) + replica + env body is housed in
        # `cluster_app_deploy` so this module stays under the line ceiling, the
        # same extraction the LLM path made. D2 boundary: scaling a SPREAD
        # service past its eligible-machine count leaves tasks Pending and must
        # route through `cluster_app_placement.reconcile` — that lives in
        # `configure_service`/`scale_service`, which D1 does not touch.
        if payload.get("confirm") != "deploy-cluster-app":
            raise ValueError("Confirm the reviewed cluster application deployment.")
        return deploy_catalog_app(self, payload, progress)

    def deploy_researched_app(
        self, payload: Dict[str, Any], actor: str, progress=None
    ) -> Dict[str, Any]:
        # D4a: an APPROVED draft deployed as N Swarm services atomically; the
        # impure body is in `cluster_app_multideploy` (not the catalog gate).
        if payload.get("confirm") != "deploy-researched-app":
            raise ValueError(
                "Confirm the reviewed researched application deployment."
            )
        return _multideploy_app(self, payload, actor, progress)

    def remove_researched_app(self, payload: Dict[str, Any]) -> Dict[str, Any]:
        # D4c: remove a researched multi-service app as a unit — the group
        # data-loss gate, then every service, then the overlay network through the
        # resilient teardown, retaining node-local volumes. The single-name
        # `remove_service` path is untouched; this is its own group-aware op.
        if payload.get("confirm") != "remove-researched-app":
            raise ValueError(
                "Confirm removal of the reviewed researched application."
            )
        return _remove_multideploy_app(self, payload)

    def remove_service(self, payload: Dict[str, Any]) -> Dict[str, Any]:
        if payload.get("confirm") != "remove-cluster-service":
            raise ValueError("Confirm removal of the reviewed cluster service.")
        name = str(payload.get("service_name", "")).strip()
        if not re.fullmatch(r"vaelor-(?:app|llm)-[a-z0-9-]{1,52}", name):
            raise ValueError("Choose a Vaelor-managed cluster service.")
        if name not in set(live_service_names(self.driver.status())):
            raise ValueError("The managed cluster service was not found.")
        # GATE (item 2): removing a stateful app with no backup is a data loss.
        if self._stateful_names_needing_backup([name]):
            require_data_loss_ack(
                expected_ack=STATEFUL_REMOVE_ACK,
                provided_ack=payload.get("data_loss_ack"),
                subject="This app",
                consequence=REMOVAL_WITHOUT_BACKUP,
            )
        retained = self._volumes_left_by(name)
        self.driver.remove_service(name)
        # Strip any controller node label this service's placement added, unless
        # another managed service still pins the controller (B1).
        remove_controller_label_if_unused(self, removed_service=name)
        # W4d-D20: `service rm` keeps the volume; list it so it can be reused
        # or deleted instead of sitting on the worker unseen.
        return {
            "removed": True, "name": name,
            **cluster_retained_volumes.record_retained(self, [retained]),
        }

    def _volumes_left_by(self, name: str):
        """``(service, volumes, holders)`` a removal of ``name`` will leave behind."""
        try:
            details = self.driver.service_details(name)
        except (ClusterDriverError, ValueError):
            return (name, [], "")
        if not isinstance(details, dict):
            return (name, [], "")
        volumes = cluster_retained_volumes.volume_mounts(details)
        pinned = cluster_retained_volumes.pinned_node(details.get("constraints"))
        # F3: an unpinned service's volume is wherever its tasks ran.
        return (
            name, volumes,
            cluster_retained_volumes.holders_of(self, details, pinned) if volumes else [pinned],
        )

    def _stateful_names_needing_backup(self, candidate_names=None, *, pinned_to=""):
        """Managed stateful app services with no recovery backup — the ones the
        remove/drain gates refuse. ``candidate_names`` defaults to every live app
        service (the drain gate's whole-node question); the pure
        `cluster_service_state.services_needing_backup` decides from two reads."""
        if candidate_names is None:
            candidate_names = live_service_names(self.driver.status())

        def details_for(name):
            try:
                return self.driver.service_details(name)
            except (ClusterDriverError, ValueError):
                return None

        return services_needing_backup(
            candidate_names, details_for=details_for,
            has_backup=lambda name: bool(self.backup_store.list(service_name=name, limit=1)),
            pinned_to=pinned_to,
        )

    def manage_service(self, payload: Dict[str, Any]) -> Dict[str, Any]:
        action = str(payload.get("action", "")).strip().lower()
        confirmations = {
            "restart": "restart-cluster-service",
            "refresh": "refresh-cluster-service",
            "rollback": "rollback-cluster-service",
        }
        if action not in confirmations:
            raise ValueError("Choose restart, refresh, or rollback.")
        if payload.get("confirm") != confirmations[action]:
            raise ValueError(f"Confirm cluster service {action}.")
        name = str(payload.get("service_name", "")).strip()
        if not re.fullmatch(r"vaelor-(?:app|llm)-[a-z0-9-]{1,52}", name):
            raise ValueError("Choose a Vaelor-managed cluster service.")
        if name not in set(live_service_names(self.driver.status())):
            raise ValueError("The managed cluster service was not found.")
        operations = {
            "restart": self.driver.restart_service,
            "refresh": self.driver.refresh_service,
            "rollback": self.driver.rollback_service,
        }
        return operations[action](name)

    def configure_service(self, payload: Dict[str, Any]) -> Dict[str, Any]:
        if payload.get("confirm") != "configure-cluster-service":
            raise ValueError("Confirm the reviewed cluster service settings.")
        name = str(payload.get("service_name", "")).strip()
        if not re.fullmatch(r"vaelor-(?:app|llm)-[a-z0-9-]{1,52}", name):
            raise ValueError("Choose a Vaelor-managed cluster service.")
        if name not in set(live_service_names(self.driver.status())):
            raise ValueError("The managed cluster service was not found.")
        try:
            replicas = int(payload.get("replicas", 0))
            memory_limit_mib = int(payload.get("memory_limit_mib", 0))
            memory_reservation_mib = int(
                payload.get("memory_reservation_mib", 0)
            )
            update_parallelism = int(payload.get("update_parallelism", 0))
        except (TypeError, ValueError) as error:
            raise ValueError(
                "Replica, memory, and rolling-update settings must be numbers."
            ) from error
        update_order = str(payload.get("update_order", "")).strip().lower()
        if not 1 <= replicas <= 32:
            raise ValueError("Choose between 1 and 32 replicas.")
        if not 16 <= memory_limit_mib <= 131072:
            raise ValueError(
                "Choose a memory limit from 16 MiB to 128 GiB."
            )
        if not 16 <= memory_reservation_mib <= memory_limit_mib:
            raise ValueError(
                "Memory reservation must be at least 16 MiB and no larger "
                "than the limit."
            )
        if not 1 <= update_parallelism <= min(replicas, 8):
            raise ValueError(
                "Update parallelism must be between 1 and the replica count."
            )
        if update_order not in {"start-first", "stop-first"}:
            raise ValueError(
                "Choose start-first or stop-first rolling updates."
            )
        # D2: validate the CPU limit + label constraints, then re-run placement
        # eligibility for the NEW replicas/reservation through the shared
        # reconcile — the SAME derivation the preview ran, so a scale a preview
        # refused is refused here too. `reconcile_service_change` raises a
        # PlacementError (a ValueError) with the arithmetic on a refusal.
        cpu_limit = validate_cpu_limit(payload.get("cpu_limit"))
        label_constraints = build_label_constraints(
            payload.get("label_constraints")
        )
        decision = reconcile_service_change(
            name,
            replicas=replicas,
            memory_reservation_mib=memory_reservation_mib,
            store=self.store,
            driver=self.driver,
            inventory_probe=self.inventory_probe,
        )
        additions = [
            constraint for constraint in label_constraints
            if constraint not in decision.existing_constraints
        ]
        return self.driver.configure_service(
            name,
            replicas=replicas,
            memory_limit_mib=memory_limit_mib,
            memory_reservation_mib=memory_reservation_mib,
            update_parallelism=update_parallelism,
            update_order=update_order,
            cpu_limit=cpu_limit,
            cpu_reservation=cpu_reservation_cores(cpu_limit),
            label_constraints_add=additions,
        )

    def _backup_target(self, service_name: str):
        return cluster_service_backup.backup_target(self, service_name)

    def _service_worker(self, details: Dict[str, Any]) -> Dict[str, Any]:
        for constraint in details.get("constraints", []):
            match = re.fullmatch(
                r"node\.labels\.vaelor\.node_id==(.+)",
                str(constraint),
            )
            if match:
                return self._joined_node(
                    match.group(1), include_credential=True
                )
        running_hosts = {
            str(task.get("node", ""))
            for task in details.get("tasks", [])
            if str(task.get("current", "")).lower().startswith("running")
        }
        for listed in self.store.list_nodes():
            node = self.store.get_node(
                listed["id"], include_credential=True
            )
            if node is None:
                continue
            hostname = str(node.get("inventory", {}).get("hostname", ""))
            if hostname in running_hosts or node.get("name") in running_hosts:
                if node.get("labels", {}).get("swarm_node_id"):
                    return node
        raise ValueError(
            "No enrolled running worker could be resolved for this service."
        )

    def service_diagnostics(
        self, service_name: str, tool: str, *, details: Dict[str, Any]
    ) -> Dict[str, Any]:
        """One app-scoped docker read on the worker running ``service_name``.

        ``details`` is the service as the CALLER read it (W4d-D18). This runs
        in the control plane, whose ``self.driver`` is a direct docker client
        and which holds no docker group (#141), so reading the service here
        answered "permission denied" on every appliance. The route reads it
        through the cluster manager's brokered driver instead; the worker side
        is an elevated SSH read, as before.
        """
        name = str(service_name).strip()
        if not re.fullmatch(r"vaelor-(?:app|llm)-[a-z0-9-]{1,52}", name):
            raise ValueError("Choose a Vaelor-managed cluster service.")
        selected_tool = str(tool).strip().lower()
        if selected_tool not in {"stats", "processes", "health"}:
            raise ValueError("Choose resource use, processes, or health.")
        node = self._service_worker(details)
        profile = self.broker.resolve(node["credential_id"], "cluster-node")
        transport = SshTransport(profile)
        container_ids = transport.run([
            "docker", "ps",
            "--filter", f"label=com.docker.swarm.service.name={name}",
            "--filter", "status=running",
            "--format", "{{.ID}}",
        ], sudo=True).splitlines()
        if not container_ids:
            raise ValueError("No running service task is available on this worker.")
        container_id = container_ids[0].strip()
        commands = {
            "stats": [
                "docker", "stats", "--no-stream", "--format",
                (
                    '{"name":{{json .Name}},"cpu":{{json .CPUPerc}},'
                    '"memory":{{json .MemUsage}},"memory_percent":'
                    '{{json .MemPerc}},"network":{{json .NetIO}},'
                    '"block":{{json .BlockIO}}}'
                ),
                container_id,
            ],
            "processes": [
                "docker", "top", container_id, "-eo", "pid,user,comm",
            ],
            "health": [
                "docker", "inspect", "--format",
                "{{json .State}}",
                container_id,
            ],
        }
        output = transport.run(
            commands[selected_tool], sudo=True, timeout=30
        )
        structured = None
        if selected_tool in {"stats", "health"}:
            try:
                structured = json.loads(output)
            except json.JSONDecodeError:
                structured = None
        if selected_tool == "health" and isinstance(structured, dict):
            health = structured.get("Health")
            structured = {
                "status": str(structured.get("Status", "")),
                "health": (
                    str(health.get("Status", ""))
                    if isinstance(health, dict)
                    else "not-configured"
                ),
                "started_at": str(structured.get("StartedAt", "")),
                "finished_at": str(structured.get("FinishedAt", "")),
                "error": str(structured.get("Error", ""))[:500],
            }
        return {
            "service_name": name,
            "node_id": node["id"],
            "node_name": node["name"],
            "tool": selected_tool,
            "data": structured,
            "output": output[:65536] if structured is None else "",
        }

    def backup_service(self, payload: Dict[str, Any]) -> Dict[str, Any]:
        return cluster_service_backup.backup_service(self, payload)

    def _apply_cluster_archive(
        self,
        *,
        transport: SshTransport,
        archive: Dict[str, Any],
        volume: str,
    ) -> None:
        cluster_service_backup.apply_cluster_archive(
            self, transport=transport, archive=archive, volume=volume
        )

    def restore_service(self, payload: Dict[str, Any]) -> Dict[str, Any]:
        return cluster_service_backup.restore_service(self, payload)

    def remove_pooled_inference(
        self, payload: Dict[str, Any]
    ) -> Dict[str, Any]:
        return self.pooled_operations.remove(payload)

    def remove_gpu_inference(
        self, payload: Dict[str, Any]
    ) -> Dict[str, Any]:
        return self.gpu_operations.remove(payload)

    def unload_gpu_inference(self, payload: Dict[str, Any]) -> Dict[str, Any]:
        """G3a: stop a vLLM deployment's units to reclaim the GPU, keeping it."""
        return unload_deployment(self.gpu_operations, payload)

    def load_gpu_inference(
        self, payload: Dict[str, Any], progress=None
    ) -> Dict[str, Any]:
        """G3a: re-serve an unloaded vLLM deployment warm from its record."""
        return load_deployment(self.gpu_operations, payload, progress, self.driver.status)

    def refresh_gpu_inference(
        self, payload: Dict[str, Any], progress=None
    ) -> Dict[str, Any]:
        """W4-D1: re-render a serving vLLM deployment after an upgrade (Unload, Load)."""
        return refresh_deployment(self.gpu_operations, payload, progress, self.driver.status)


    def rotate_gpu_cluster_key(
        self, payload: Dict[str, Any], progress=None
    ) -> Dict[str, Any]:
        return self.gpu_operations.rotate_cluster_serving_key(payload, progress)

    def reconcile_gpu_cluster_mode(
        self, *, deploy_job_active: Callable[[], bool], job_store: Any = None
    ) -> Dict[str, Any]:
        """The 30 s mode watch's whole pass; see `gpu_cluster_mode_watch`.

        This layer contributes the three things the switch cannot reach - the
        cluster store, the executor's job loop through ``deploy_job_active``,
        and the nodes through the GPU operations' `stop_units_for` (the stop
        the switch runs before it restores Mode A over an abandoned deploy) -
        and the switch owns every decision made from them (VD-125). Non-fatal
        by construction, and a no-op without a switch (D12).
        """
        return _mode_watch_pass(
            self.gpu_mode_switch, self.store, deploy_job_active,
            self.gpu_operations.stop_units_for,
            self.gpu_operations.replica_health,
            ops=self.gpu_operations, job_store=job_store,
        )

    def architecture_survey(self) -> Dict[str, Any]:
        return self.architecture_eviction.survey()

    def evict_mismatched_nodes(
        self, payload: Dict[str, Any]
    ) -> Dict[str, Any]:
        return self.architecture_eviction.evict(payload)

    def _joined_node(self, node_id: Any, *, include_credential: bool = False) -> Dict[str, Any]:
        node = self.store.get_node(
            str(node_id or ""), include_credential=include_credential
        )
        if node is None:
            raise ValueError(NODE_NOT_FOUND)
        if not node.get("labels", {}).get("swarm_node_id"):
            raise ValueError("Join this worker before managing its cluster state.")
        return node

    def _placement_node(self, node_id: Any) -> Dict[str, Any]:
        """Resolve a joined worker or the local Swarm manager for inference."""
        if str(node_id or "") != CONTROLLER_PLACEMENT_ID:
            return self._joined_node(node_id)
        status = self.driver.status()
        controller = self.store.controller()
        if not (
            status.get("initialized")
            and status.get("control_available")
            and controller.get("initialized")
        ):
            raise ValueError(
                "Initialize this controller before placing an LLM server on it."
            )
        swarm_node_id = str(
            controller.get("cluster_id") or status.get("node_id") or ""
        )
        if not swarm_node_id:
            raise ValueError("The controller Swarm node identity is unavailable.")
        hardware = self.inventory_probe()
        return {
            "id": CONTROLLER_PLACEMENT_ID,
            "name": "This Vaelor controller",
            "state": "joined",
            "role": "head-controller",
            "labels": {"swarm_node_id": swarm_node_id},
            "inventory": {
                "architecture": hardware.get("architecture", "unknown"),
                "memory_bytes": int(
                    hardware.get("memory_total_bytes", 0) or 0
                ),
                "root_free_bytes": int(
                    hardware.get("storage_free_bytes", 0) or 0
                ),
            },
        }

    def _gpu_placement_node(
        self, node_id: Any, *, include_credential: bool = False
    ) -> Dict[str, Any]:
        """A GPU-serving participant: an enrolled worker, or this controller.

        VD-125. `GpuPoolOperations` resolves every chosen node through this, so
        the controller reaches the deploy as an ordinary participant. What it
        needs beyond `_placement_node`'s record is what a PROBED worker carries
        and the fit engine and the collective binding read:

        * ``host`` - the address the cluster reaches this machine on, which for
          the controller is its own Swarm advertise address (there is no SSH
          host to borrow). `GpuPoolOperations` passes it through
          `_advertise_address`, so it is validated as a private IPv4 exactly as
          a worker's is.
        * ``inventory["gpu"]`` - read from the LOCAL accelerator probe through
          `cluster_capacity.gpu_facts_from_accelerators`, the one reader that
          already describes this controller's GPU for `/cluster/capacity` and
          the `/cluster/fit` preview. No second sysfs parser exists, so the
          preview and the deploy cannot disagree about the same machine.
        * ``inventory["cluster_interface"]`` - the NIC carrying that address,
          from the same `gpu_node_facts.interface_for_address` rule the enrolled
          probe uses, in the same ``{name, address}`` (or ``reason``) shape.

        The controller is never a Swarm *worker*, so this deliberately does not
        route through `_joined_node` for it; every other node still does, and a
        node that has not joined is refused there as before.
        """
        if str(node_id or "") != CONTROLLER_PLACEMENT_ID:
            return self._joined_node(
                node_id, include_credential=include_credential
            )
        node = self._placement_node(CONTROLLER_PLACEMENT_ID)
        try:
            address = self._advertise_address(
                self.store.controller().get("advertise_address", "")
            )
        except ValueError as error:
            # The address rule's own sentence says what is wrong with the value
            # but not whose entry to go and fix, and this deploy has several
            # participants. Prefixed by the machine, never respelled - the same
            # shape a per-node interface pin's refusal takes.
            raise ValueError(f"{node['name']}: {error}") from error
        node["host"] = address
        node["inventory"] = {
            **node["inventory"],
            "gpu": self._controller_gpu_facts(),
            "cluster_interface": self._controller_cluster_interface(address),
        }
        return node

    def _controller_gpu_facts(self) -> Dict[str, Any]:
        """This controller's GPU, in the shape `ssh_transport` writes for a node.

        One reader (`cluster_capacity.gpu_facts_from_accelerators`) over the
        controller's own `hardware_inventory` accelerators, so a node and the
        head describe a GPU identically. That reader returns the ledger's
        NORMALIZED shape, which carries fields derived from the raw ones; those
        are dropped here because raw node inventory never holds a derived value
        (`gpu_pool_operations._gpu_fit_node` re-derives the ceiling itself, and
        a deploy reading a stored derivation off inventory is the exact defect
        that rule exists to prevent).

        `unified_memory` and `unified_memory_source` are NOT derived and stay:
        they are the accelerator probe's own verdict about this part, exactly as
        the SSH probe writes them for a worker, and the ledger's ceiling rule
        reads them rather than guessing from byte figures (VD-125). Dropping
        them here would make the controller the one machine whose measured
        answer was thrown away before the ledger could use it.
        """
        try:
            hardware = self.inventory_probe() or {}
        except (AttributeError, OSError, TypeError, ValueError):
            hardware = {}
        memory_total = int(hardware.get("memory_total_bytes", 0) or 0)
        # VD-125 (D1): raw inventory carries the FACT - "the memory this GPU is
        # holding is the AI-Chat model the switch stops" - and never the bytes
        # derived from it; `gpu_pool_operations._gpu_fit_node` derives those
        # through the same `cluster_capacity` call the ledger uses. The reason
        # rides with it so an UNREADABLE answer is not a negative one (VD-127).
        reclaimable, reclaim_reason = controller_gpu_chat_reclaimable()
        gpu = dict(gpu_facts_from_accelerators(
            hardware.get("accelerators"), memory_total or None,
            mode_a_reclaimable=reclaimable,
            mode_a_reclaimable_reason=reclaim_reason,
        ))
        for derived in (
            "addressable_bytes", "reclaimable_bytes", "memory_model",
            "memory_model_reason",
        ):
            gpu.pop(derived, None)
        return gpu

    @staticmethod
    def _controller_cluster_interface(address: str) -> Dict[str, Any]:
        """The NIC carrying the controller's cluster address, or the reason why not.

        A thin call to `gpu_node_facts.local_cluster_interface`, which is the ONE
        home of this read (the display path -
        `cluster_manager.summary` - shares the same helper, so the deploy and the
        fleet UI cannot disagree about this one machine's link). Absent WITH a
        reason, never guessed: an unreadable table or an address on no connected
        subnet leaves ``name`` empty and says why, and the GPU deploy is what
        turns that into a refusal, by node, before anything is started.
        """
        return local_cluster_interface(address)

    @staticmethod
    def _memory_tier(bytes_value: Any) -> str:
        try:
            gib = int(bytes_value) / 1024 ** 3
        except (TypeError, ValueError):
            return "unknown"
        if gib >= 15:
            return "16gb-plus"
        if gib >= 7:
            return "8gb"
        if gib >= 3:
            return "4gb"
        return "low-memory"
