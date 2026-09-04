"""Lifecycle orchestration for SSH-managed pooled-memory inference."""

from __future__ import annotations

import json
import re
import time
import urllib.error
import urllib.request
from typing import Any, Callable, Dict

from .cluster_placement import CONTROLLER_PLACEMENT_ID
from .pooled_inference import (
    artifact_architecture,
    node_thread_count,
    runtime_artifact,
    validate_pooled_deployment,
)
from .ssh_transport import SshTransport


def worker_thread_count(inventory: Dict[str, Any]) -> int:
    """Threads for the llama runtime on one node: logical CPUs, capped at 8.

    #113: the field resolution (``cpu_threads`` -> ``cpu_count`` -> default)
    lives in ``pooled_inference.node_thread_count`` so this consumer and the
    admission gate can never disagree on a record missing ``cpu_threads``. This
    caps at 8 for the llama runtime; the gate applies only the four-thread
    minimum. Never reads ``cpu_count`` as cores: that is physical cores now, and
    sizing ``--threads`` from cores throws away SMT.
    """
    return max(1, min(node_thread_count(inventory), 8))


class PooledDeploymentOperations:
    def __init__(
        self, *, store, broker, runtime,
        joined_node: Callable[..., Dict[str, Any]],
        advertise_address: Callable[[Any], str],
    ):
        self.store = store
        self.broker = broker
        self.runtime = runtime
        self.joined_node = joined_node
        self.advertise_address = advertise_address

    def _transports(self, nodes):
        return [
            SshTransport(
                self.broker.resolve(node["credential_id"], "cluster-node"),
                timeout=120,
            )
            for node in nodes
        ]

    def _stop_started(self, started) -> None:
        for transport, unit in reversed(started):
            try:
                self.runtime.stop_unit(transport, unit)
            except Exception:
                pass

    def deploy(self, payload: Dict[str, Any], progress=None) -> Dict[str, Any]:
        raw_ids = payload.get("node_ids", [])
        if not isinstance(raw_ids, list):
            raise ValueError("Choose the pooled root and worker nodes.")
        node_ids = list(dict.fromkeys(
            str(node_id).strip() for node_id in raw_ids
            if str(node_id).strip()
        ))
        # Pooled inference runs the llama.cpp RPC mesh over SSH-managed workers;
        # the local controller placement has no SSH credential and cannot host a
        # pool member. Guard here on the path that actually runs pooled deploys -
        # the identical check in cluster_operations.deploy_llm was unreachable
        # dead code because the pooled branch returns before it (#Recovery-7b).
        if CONTROLLER_PLACEMENT_ID in node_ids:
            raise ValueError(
                "Pooled-memory inference currently supports SSH-managed workers only."
            )
        nodes = [
            self.joined_node(node_id, include_credential=True)
            for node_id in node_ids
        ]
        model_id = str(payload.get("pooled_model", ""))
        preflight = validate_pooled_deployment(model_id, nodes)
        # From the pool's own discovered architecture, so this cannot disagree
        # with the plan the operator just approved.
        artifact = runtime_artifact(preflight["architecture"])
        if artifact is None:
            raise ValueError(
                "Install the verified {} pooled runtime artifact first.".format(
                    artifact_architecture(preflight["architecture"])
                )
            )
        name = str(payload.get("name", "")).strip().lower()
        if not re.fullmatch(r"[a-z0-9][a-z0-9-]{0,38}", name):
            raise ValueError("Use a short lowercase pooled deployment name.")
        existing = self.store.get_pooled_deployment(name)
        if existing and existing.get("state") in {"deploying", "healthy"}:
            raise ValueError("A pooled deployment already uses that name.")

        report = progress or (lambda _percent, _message: None)
        transports = self._transports(nodes)
        destinations = []
        started = []
        credential_id = ""
        self.store.put_pooled_deployment(
            name=name, state="deploying", model_id=model_id, node_ids=node_ids
        )
        try:
            for index, transport in enumerate(transports):
                report(
                    10 + int(25 * index / len(nodes)),
                    f"Installing the verified runtime on {nodes[index]['name']}",
                )
                destinations.append(self.runtime.install(transport, artifact))
            report(40, f"Downloading {preflight['model']['name']} on the root node")
            self.runtime.download_model(
                transports[0], destinations[0], preflight["model"]["id"]
            )

            worker_units = []
            worker_addresses = []
            for index in range(1, len(nodes)):
                address = self.advertise_address(nodes[index]["host"])
                worker_addresses.append(address)
                report(60, f"Starting pooled worker {nodes[index]['name']}")
                unit = self.runtime.start_worker(
                    transports[index],
                    name=name,
                    destination=destinations[index],
                    address=address,
                    threads=worker_thread_count(nodes[index]["inventory"]),
                )
                worker_units.append(unit)
                started.append((transports[index], unit))

            root_address = self.advertise_address(nodes[0]["host"])
            report(72, f"Starting pooled root API on {nodes[0]['name']}")
            root_unit = self.runtime.start_root(
                transports[0],
                name=name,
                destination=destinations[0],
                address=root_address,
                threads=worker_thread_count(nodes[0]["inventory"]),
                model=preflight["model"],
                workers=worker_addresses,
            )
            started.append((transports[0], root_unit))
            endpoint = f"http://{root_address}:9998/v1"
            self._wait_for_api(endpoint, payload, report)

            profile = json.dumps(
                {"base_url": endpoint, "model": "", "api_key": ""},
                separators=(",", ":"),
            )
            credential = self.broker.put(
                "openai-compatible", f"Pooled LLM · {name}", profile
            )
            credential_id = credential["id"]
            test = self.broker.test(credential_id)
            if not test.get("ok"):
                raise RuntimeError(
                    str(test.get("message", "The pooled model API test failed."))
                )
            self.broker.activate(credential_id, "cluster-inference")
            if payload.get("use_for_assistant"):
                self.broker.activate(credential_id, "deployment-agent")
            units = {"root": root_unit, "workers": worker_units}
            self.store.put_pooled_deployment(
                name=name, state="healthy", model_id=model_id,
                node_ids=node_ids, units=units, endpoint=endpoint,
                credential_id=credential_id,
            )
        except Exception:
            if credential_id:
                try:
                    self.broker.delete(credential_id)
                except Exception:
                    pass
            self._stop_started(started)
            self.store.put_pooled_deployment(
                name=name, state="failed", model_id=model_id, node_ids=node_ids
            )
            raise

        report(100, "Pooled inference is healthy behind the managed gateway")
        return {
            "name": name,
            "deployment_mode": "pooled",
            "engine": "distributed-llama",
            "node_ids": node_ids,
            "root_node_id": nodes[0]["id"],
            "worker_node_ids": node_ids[1:],
            "units": units,
            "endpoint": endpoint,
            "credential_id": credential_id,
            "runtime": {
                "commit": artifact["commit"],
                "sha256": artifact["sha256"],
            },
            "model": preflight["model"],
        }

    @staticmethod
    def _wait_for_api(endpoint, payload, report) -> None:
        deadline = time.monotonic() + max(
            60, min(int(payload.get("startup_timeout_seconds", 900)), 1800)
        )
        while time.monotonic() < deadline:
            try:
                with urllib.request.urlopen(
                    endpoint + "/models", timeout=5
                ) as response:
                    if response.status == 200:
                        return
            except (OSError, urllib.error.URLError):
                pass
            report(82, "Waiting for every pooled node and the root API")
            time.sleep(3)
        raise RuntimeError("The pooled root API did not become healthy.")

    def remove(self, payload: Dict[str, Any]) -> Dict[str, Any]:
        if payload.get("confirm") != "remove-pooled-inference":
            raise ValueError("Confirm the reviewed pooled deployment removal.")
        name = str(payload.get("name", "")).strip().lower()
        deployment = self.store.get_pooled_deployment(name)
        if deployment is None:
            raise ValueError("The pooled deployment was not found.")
        nodes = [
            self.joined_node(node_id, include_credential=True)
            for node_id in deployment["node_ids"]
        ]
        transports = self._transports(nodes)
        units = deployment.get("units", {})
        root_unit = str(units.get("root", ""))
        worker_units = list(units.get("workers", []))
        if root_unit:
            self.runtime.stop_unit(transports[0], root_unit)
        for index, unit in enumerate(worker_units, start=1):
            self.runtime.stop_unit(transports[index], str(unit))
        credential_id = str(deployment.get("credential_id", ""))
        if credential_id:
            self.broker.delete(credential_id)
        self.store.delete_pooled_deployment(name)
        return {
            "name": name,
            "removed": True,
            "model_cache_retained": True,
            "runtime_retained": True,
        }
