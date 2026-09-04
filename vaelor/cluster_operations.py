"""Privileged, approval-gated head-controller cluster operations."""

from __future__ import annotations

import hashlib
import json
import os
import re
import tempfile
import time
import urllib.error
import urllib.request
import uuid
from pathlib import Path
from typing import Any, Dict, Optional

from .cluster_architecture import controller_architecture, join_compatibility
from .cluster_backups import ClusterBackupStore
from .cluster_driver import DockerSwarmDriver
from .cluster_llm_sizing import swarm_llm_memory_limit_mib
from .model_footprint import MIB
from .cluster_eviction import ArchitectureEviction
from .cluster_store import ClusterStore
from .credential_broker import CredentialBrokerClient
from .ssh_transport import SshTransport
from .app_catalog import APP_TEMPLATES
from .pooled_runtime import PooledRuntime
from .pooled_operations import PooledDeploymentOperations
from .platform_drivers import default_platform_drivers
from .copilot_setup import hardware_inventory
from .cluster_placement import CONTROLLER_PLACEMENT_ID
from .cluster_network import private_controller_ipv4


class ClusterOperations:
    def __init__(
        self,
        store: Optional[ClusterStore] = None,
        broker: Optional[CredentialBrokerClient] = None,
        driver: Optional[DockerSwarmDriver] = None,
        pooled_runtime: Optional[PooledRuntime] = None,
        backup_store: Optional[ClusterBackupStore] = None,
        os_driver=None,
        inventory_probe=None,
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
        self._controller_architecture = None
        self.pooled_operations = PooledDeploymentOperations(
            store=self.store,
            broker=self.broker,
            runtime=self.pooled_runtime,
            joined_node=self._joined_node,
            advertise_address=self._advertise_address,
        )
        # VD-033. Reuses `remove_node`, so an evicted host loses its stored SSH
        # credential exactly as a refused one does.
        self.architecture_eviction = ArchitectureEviction(
            store=self.store,
            controller_architecture=self.controller_architecture,
            remove_node=self.remove_node,
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
            raise ValueError("Cluster node was not found.")
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
        swarm_node_id = str(node["labels"]["swarm_node_id"])
        result = self.driver.set_node_availability(swarm_node_id, availability)
        state = "joined" if availability == "active" else availability
        self.store.update_node(node["id"], state=state)
        return {**result, "node_id": node["id"], "state": state}

    def remove_node(self, payload: Dict[str, Any]) -> Dict[str, Any]:
        force = bool(payload.get("force", False))
        expected = "force-remove-worker-node" if force else "remove-worker-node"
        if payload.get("confirm") != expected:
            raise ValueError("Confirm the reviewed worker removal.")
        node = self.store.get_node(
            str(payload.get("node_id", "")), include_credential=True
        )
        if node is None:
            raise ValueError("Cluster node was not found.")
        swarm_node_id = str(node.get("labels", {}).get("swarm_node_id", ""))
        result = {"removed": True, "forced": False, "swarm_node_id": ""}
        if swarm_node_id:
            profile = self.broker.resolve(node["credential_id"], "cluster-node")
            result = self.driver.remove_worker(
                SshTransport(profile, timeout=120),
                swarm_node_id,
                force=force,
            )
        credential_id = self.store.delete_node(node["id"])
        if credential_id:
            self.broker.delete(credential_id)
        return {**result, "node_id": node["id"]}

    def deploy_llm(self, payload: Dict[str, Any], progress=None) -> Dict[str, Any]:
        if payload.get("confirm") != "deploy-cluster-llm":
            raise ValueError("Confirm the reviewed cluster LLM deployment.")
        deployment_mode = str(
            payload.get("deployment_mode", "single")
        ).strip().lower()
        if deployment_mode == "pooled":
            return self.pooled_operations.deploy(payload, progress)
        if deployment_mode not in {"single", "replicated"}:
            raise ValueError("Choose single-worker or replicated inference.")
        if deployment_mode == "replicated":
            raw_node_ids = payload.get("node_ids", [])
            if not isinstance(raw_node_ids, list):
                raise ValueError("Choose the workers for replicated inference.")
            node_ids = list(dict.fromkeys(
                str(node_id).strip()
                for node_id in raw_node_ids
                if str(node_id).strip()
            ))
            if not 2 <= len(node_ids) <= 8:
                raise ValueError(
                    "Choose between 2 and 8 workers for replicated inference."
                )
        else:
            node_ids = [str(payload.get("node_id", "")).strip()]
        # The pooled branch returns at the top of this method, so a "pooled +
        # controller" guard here is unreachable. The SSH-only protection lives
        # in pooled_operations.deploy, on the path pooled deploys actually run
        # (#Recovery-7b).
        nodes = [self._placement_node(node_id) for node_id in node_ids]
        for node in nodes:
            if node.get("state") in {"drain", "pause"}:
                raise ValueError(
                    f"Return {node['name']} to service before deploying a model."
                )
            if node["id"] != CONTROLLER_PLACEMENT_ID:
                self._validate_worker(
                    node.get("inventory", {}),
                    where=str(node.get("host") or node.get("name") or ""),
                )

        name = str(payload.get("name", "")).strip()
        repository = str(payload.get("model_repo", "")).strip()
        model_file = str(payload.get("model_file", "")).strip()
        if not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9._-]{0,39}", name):
            raise ValueError("Use a short letters-and-numbers deployment name.")
        if not re.fullmatch(
            r"[A-Za-z0-9][A-Za-z0-9._-]{0,95}/[A-Za-z0-9][A-Za-z0-9._-]{0,95}",
            repository,
        ):
            raise ValueError("Choose a valid reviewed Hugging Face repository.")
        if not re.fullmatch(
            r"[A-Za-z0-9][A-Za-z0-9._-]{0,199}\.gguf", model_file, re.I
        ):
            raise ValueError("Choose a reviewed GGUF model file.")
        try:
            model_size = int(payload.get("model_size_bytes", 0))
            port = int(payload.get("port", 8100))
        except (TypeError, ValueError) as error:
            raise ValueError("Model size and API port must be numbers.") from error
        if model_size < 32 * 1024 ** 2:
            raise ValueError("Use the inspected GGUF file size before deployment.")
        if not 8100 <= port <= 8199:
            raise ValueError("Cluster LLM ports must be between 8100 and 8199.")

        # #138: measured sizing, shared with the approval plan.
        memory_limit_mib = swarm_llm_memory_limit_mib(
            name, repository, model_file, nodes
        )
        required_memory = memory_limit_mib * MIB
        required_storage = int(model_size * 1.2) + 1024 ** 3
        for node in nodes:
            inventory = node.get("inventory", {})
            memory_bytes = int(inventory.get("memory_bytes", 0))
            if required_memory > max(0, memory_bytes - 1024 ** 3):
                raise ValueError(
                    f"This model does not fit on {node['name']} while preserving "
                    "1 GB for its operating system."
                )
            if int(inventory.get("root_free_bytes", 0)) < required_storage:
                raise ValueError(
                    f"{node['name']} does not have enough free storage for this model."
                )
        report = progress or (lambda _percent, _message: None)
        swarm_node_ids = [
            str(node.get("labels", {}).get("swarm_node_id", ""))
            for node in nodes
        ]
        if CONTROLLER_PLACEMENT_ID in node_ids:
            controller_index = node_ids.index(CONTROLLER_PLACEMENT_ID)
            self.driver.label_node(
                swarm_node_ids[controller_index],
                {"vaelor.node_id": CONTROLLER_PLACEMENT_ID},
            )
        pool_label = ""
        labeled_nodes: list[str] = []
        result = None
        try:
            if deployment_mode == "replicated":
                pool_label = "vaelor.llm_pool_{}".format(
                    hashlib.sha256(
                        "{}:{}:{}".format(
                            name, repository, ",".join(sorted(node_ids))
                        ).encode()
                    ).hexdigest()[:12]
                )
                for swarm_node_id in swarm_node_ids:
                    self.driver.label_node(
                        swarm_node_id, {pool_label: "true"}
                    )
                    labeled_nodes.append(swarm_node_id)
            report(
                20,
                (
                    f"Creating {len(nodes)} replicated inference services"
                    if deployment_mode == "replicated"
                    else "Creating the constrained cluster inference service"
                ),
            )
            result = self.driver.deploy_llm(
                name=name,
                node_label=nodes[0]["id"] if deployment_mode == "single" else None,
                pool_label=pool_label or None,
                replicas=len(nodes),
                model_repo=repository,
                model_file=model_file,
                memory_limit_mib=memory_limit_mib,
                port=port,
            )
            report(55, "Waiting for every inference replica to become ready")
            self.driver.wait_service(
                result["name"],
                timeout=max(
                    30,
                    min(int(payload.get("startup_timeout_seconds", 900)), 1800),
                ),
                expected_replicas=len(nodes),
            )
        except Exception:
            if result:
                self.driver.remove_service(result["name"])
            else:
                for swarm_node_id in labeled_nodes:
                    try:
                        self.driver.remove_node_label(swarm_node_id, pool_label)
                    except Exception:
                        pass
            raise

        endpoint = f"http://127.0.0.1:{port}/v1"
        health = f"http://127.0.0.1:{port}/health"
        deadline = time.monotonic() + max(
            30, min(int(payload.get("startup_timeout_seconds", 900)), 1800)
        )
        last_error = ""
        try:
            while time.monotonic() < deadline:
                try:
                    with urllib.request.urlopen(health, timeout=4) as response:
                        if response.status == 200:
                            break
                except (OSError, urllib.error.URLError) as error:
                    last_error = str(error)[:240]
                report(75, "Waiting for the model to load across the Swarm network")
                time.sleep(3)
            else:
                raise RuntimeError(
                    "The cluster model did not become healthy: {}".format(last_error)
                )
            profile = json.dumps({
                "base_url": endpoint, "model": "", "api_key": "",
            }, separators=(",", ":"))
            credential = self.broker.put(
                "openai-compatible", f"Cluster LLM · {name}", profile
            )
            try:
                test = self.broker.test(credential["id"])
                if not test.get("ok"):
                    raise RuntimeError(
                        str(test.get("message", "The cluster model API test failed."))
                    )
                self.broker.activate(credential["id"], "cluster-inference")
                if payload.get("use_for_assistant"):
                    self.broker.activate(credential["id"], "deployment-agent")
            except Exception:
                self.broker.delete(credential["id"])
                raise
        except Exception:
            self.driver.remove_service(result["name"])
            raise
        return {
            **result,
            "node_id": nodes[0]["id"],
            "node_ids": node_ids,
            "deployment_mode": deployment_mode,
            "replicas": len(nodes),
            "endpoint": endpoint,
            "health": health,
            "credential_id": credential["id"],
            "assistant_active": bool(payload.get("use_for_assistant")),
            "model_size_bytes": model_size,
        }

    def deploy_app(self, payload: Dict[str, Any], progress=None) -> Dict[str, Any]:
        if payload.get("confirm") != "deploy-cluster-app":
            raise ValueError("Confirm the reviewed cluster application deployment.")
        node = self._joined_node(payload.get("node_id"))
        if node.get("state") in {"drain", "pause"}:
            raise ValueError("Return this worker to service before deploying an app.")
        self._validate_worker(
            node.get("inventory", {}),
            where=str(node.get("host") or node.get("name") or ""),
        )
        template_id = str(payload.get("template_id", "")).strip()
        template = APP_TEMPLATES.get(template_id)
        if template is None:
            raise ValueError("Choose a reviewed application from the catalog.")
        name = str(payload.get("name", "")).strip()
        if not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9-]{0,39}", name):
            raise ValueError("Use a short letters-and-numbers deployment name.")
        try:
            port = int(payload.get("port", template["default_port"]))
        except (TypeError, ValueError) as error:
            raise ValueError("The published application port must be a number.") from error
        if (
            not 1024 <= port <= 65535
            or port in {34001, 34002}
            or 8100 <= port <= 8199
        ):
            raise ValueError("Choose an available app port from 1024 to 65535.")
        memory_mib = int(template["memory_mib"])
        memory_bytes = int(node.get("inventory", {}).get("memory_bytes", 0))
        if memory_mib * 1024 ** 2 > max(0, memory_bytes - 1024 ** 3):
            raise ValueError(
                "This worker cannot preserve 1 GB of OS headroom for that app."
            )
        report = progress or (lambda _percent, _message: None)
        report(25, "Creating the resource-bounded application service")
        result = self.driver.deploy_app(
            name=name,
            node_label=node["id"],
            image=template["image"],
            container_port=int(template["container_port"]),
            published_port=port,
            memory_limit_mib=memory_mib,
            template_id=template_id,
            volume=template.get("volume"),
        )
        try:
            report(70, "Waiting for the application replica to become ready")
            health = self.driver.wait_service(
                result["name"],
                timeout=max(30, min(int(payload.get("startup_timeout_seconds", 180)), 600)),
            )
        except Exception:
            self.driver.remove_service(result["name"])
            raise
        return {**result, **health}

    def remove_service(self, payload: Dict[str, Any]) -> Dict[str, Any]:
        if payload.get("confirm") != "remove-cluster-service":
            raise ValueError("Confirm removal of the reviewed cluster service.")
        name = str(payload.get("service_name", "")).strip()
        if not re.fullmatch(
            r"vaelor-(?:app|llm)-[a-z0-9-]{1,40}", name
        ):
            raise ValueError("Choose a Vaelor-managed cluster service.")
        services = {
            str(item.get("name", ""))
            for item in self.driver.status().get("services", [])
        }
        if name not in services:
            raise ValueError("The managed cluster service was not found.")
        self.driver.remove_service(name)
        return {"removed": True, "name": name}

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
        if not re.fullmatch(
            r"vaelor-(?:app|llm)-[a-z0-9-]{1,40}", name
        ):
            raise ValueError("Choose a Vaelor-managed cluster service.")
        services = {
            str(item.get("name", ""))
            for item in self.driver.status().get("services", [])
        }
        if name not in services:
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
        if not re.fullmatch(
            r"vaelor-(?:app|llm)-[a-z0-9-]{1,40}", name
        ):
            raise ValueError("Choose a Vaelor-managed cluster service.")
        services = {
            str(item.get("name", ""))
            for item in self.driver.status().get("services", [])
        }
        if name not in services:
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
        if not 128 <= memory_limit_mib <= 131072:
            raise ValueError(
                "Choose a memory limit from 128 MiB to 128 GiB."
            )
        if not 64 <= memory_reservation_mib <= memory_limit_mib:
            raise ValueError(
                "Memory reservation must be at least 64 MiB and no larger "
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
        return self.driver.configure_service(
            name,
            replicas=replicas,
            memory_limit_mib=memory_limit_mib,
            memory_reservation_mib=memory_reservation_mib,
            update_parallelism=update_parallelism,
            update_order=update_order,
        )

    def _backup_target(self, service_name: str):
        details = self.driver.service_details(service_name)
        node_id = ""
        for constraint in details.get("constraints", []):
            match = re.fullmatch(
                r"node\.labels\.vaelor\.node_id==(.+)",
                str(constraint),
            )
            if match:
                node_id = match.group(1)
                break
        if not node_id:
            raise ValueError(
                "This service is not pinned to an enrolled Vaelor worker."
            )
        mounts = [
            item for item in details.get("mounts", [])
            if item.get("type") == "volume"
            and re.fullmatch(r"[A-Za-z0-9_.-]{1,160}", item.get("source", ""))
        ]
        if len(mounts) != 1:
            raise ValueError(
                "Cluster backup currently requires exactly one managed named volume."
            )
        node = self._joined_node(node_id, include_credential=True)
        profile = self.broker.resolve(node["credential_id"], "cluster-node")
        return details, node, profile, mounts[0]

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
        self, service_name: str, tool: str
    ) -> Dict[str, Any]:
        name = str(service_name).strip()
        if not re.fullmatch(
            r"vaelor-(?:app|llm)-[a-z0-9-]{1,40}", name
        ):
            raise ValueError("Choose a Vaelor-managed cluster service.")
        selected_tool = str(tool).strip().lower()
        if selected_tool not in {"stats", "processes", "health"}:
            raise ValueError("Choose resource use, processes, or health.")
        details = self.driver.service_details(name)
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
        if payload.get("confirm") != "backup-cluster-service":
            raise ValueError("Confirm the reviewed cluster service backup.")
        name = str(payload.get("service_name", "")).strip()
        details, node, profile, mount = self._backup_target(name)
        transport = SshTransport(profile)
        remote_name = f"/tmp/vaelor-cluster-{uuid.uuid4().hex}.tar.gz"
        staging_root = Path(
            self.backup_store.root.parent / ".cluster-staging"
        )
        staging_root.mkdir(parents=True, exist_ok=True)
        descriptor, staged_name = tempfile.mkstemp(
            prefix="cluster-", suffix=".tar.gz", dir=staging_root
        )
        os.close(descriptor)
        staged_path = Path(staged_name)
        try:
            staged_path.unlink(missing_ok=True)
            transport.run(
                [
                    "docker", "run", "--rm",
                    "--network", "none",
                    "--read-only",
                    "--cap-drop", "ALL",
                    "-v", f"{mount['source']}:/data:ro",
                    "-v", "/tmp:/backup",
                    "alpine:3.20",
                    "tar", "-czf", f"/backup/{Path(remote_name).name}",
                    "-C", "/data", ".",
                ],
                sudo=True,
                timeout=1800,
            )
            transport.run(
                ["chown", str(profile["username"]), remote_name],
                sudo=True,
            )
            transport.get_cluster_archive(remote_name, str(staged_path))
            result = self.backup_store.record(
                service_name=name,
                node_id=node["id"],
                volume=mount["source"],
                staged_archive=staged_path,
                reason=str(payload.get("reason", "manual")),
            )
            return {
                **result,
                "node_name": node["name"],
                "desired_replicas": details["desired_replicas"],
                "verified": True,
            }
        finally:
            staged_path.unlink(missing_ok=True)
            try:
                transport.run(["rm", "-f", remote_name], sudo=True)
            except Exception:
                pass

    def _apply_cluster_archive(
        self,
        *,
        transport: SshTransport,
        archive: Dict[str, Any],
        volume: str,
    ) -> None:
        remote_name = f"/tmp/vaelor-cluster-{uuid.uuid4().hex}.tar.gz"
        try:
            transport.put_cluster_archive(
                archive["archive_path"], remote_name
            )
            observed = transport.run(["sha256sum", remote_name]).split()[0]
            if observed != archive["sha256"]:
                raise ValueError("The worker received an invalid backup archive.")
            # The restore container drops every capability, including DAC
            # override. Keep the staged archive private while making its owner
            # match the container's root process.
            transport.run(
                ["chown", "root:root", remote_name],
                sudo=True,
            )
            transport.run(
                ["chmod", "0600", remote_name],
                sudo=True,
            )
            transport.run(
                [
                    "docker", "run", "--rm",
                    "--network", "none",
                    "--read-only",
                    "--cap-drop", "ALL",
                    "-v", f"{volume}:/data",
                    "-v", "/tmp:/backup:ro",
                    "alpine:3.20",
                    "sh", "-ceu",
                    (
                        'find /data -mindepth 1 -maxdepth 1 '
                        '-exec rm -rf -- "{}" +; '
                        'tar -xzf "$1" -C /data'
                    ),
                    "vaelor-restore",
                    f"/backup/{Path(remote_name).name}",
                ],
                sudo=True,
                timeout=1800,
            )
        finally:
            try:
                transport.run(["rm", "-f", remote_name], sudo=True)
            except Exception:
                pass

    def restore_service(self, payload: Dict[str, Any]) -> Dict[str, Any]:
        if payload.get("confirm") != "restore-cluster-service":
            raise ValueError("Confirm the reviewed cluster service restore.")
        name = str(payload.get("service_name", "")).strip()
        backup = self.backup_store.get(
            str(payload.get("backup_id", "")), verify=True
        )
        if backup["service_name"] != name:
            raise ValueError("Choose a backup created for this service.")
        details, node, profile, mount = self._backup_target(name)
        if (
            backup["node_id"] != node["id"]
            or backup["volume"] != mount["source"]
        ):
            raise ValueError(
                "The backup does not match this worker volume placement."
            )
        safety = self.backup_service({
            "confirm": "backup-cluster-service",
            "service_name": name,
            "reason": "pre-restore",
        })
        safety_archive = self.backup_store.get(safety["id"], verify=True)
        transport = SshTransport(profile)
        replicas = int(details["desired_replicas"])
        self.driver.scale_service(name, 0)
        try:
            self._apply_cluster_archive(
                transport=transport,
                archive=backup,
                volume=mount["source"],
            )
            ready = self.driver.scale_service(name, replicas)
        except Exception:
            try:
                self._apply_cluster_archive(
                    transport=transport,
                    archive=safety_archive,
                    volume=mount["source"],
                )
                self.driver.scale_service(name, replicas)
            except Exception:
                pass
            raise
        return {
            "restored": True,
            "service_name": name,
            "backup_id": backup["id"],
            "safety_backup_id": safety["id"],
            **ready,
        }

    def remove_pooled_inference(
        self, payload: Dict[str, Any]
    ) -> Dict[str, Any]:
        return self.pooled_operations.remove(payload)

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
            raise ValueError("Cluster node was not found.")
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
