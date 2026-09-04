"""Composition boundary for managed-app capabilities and custom-agent grants."""

from __future__ import annotations

import time
from typing import Any, Mapping, Optional

from .agent_app_grants import AgentAppGrantStore
from .app_capability_broker import AppCapabilityBroker
from .app_capability_registry import AppCapabilityRegistry
from .integration_connections import IntegrationConnectionStore
from .managed_app_capabilities import ManagedAppCapabilityReconciler
from .managed_app_transport import ManagedAppTransport


class IntegrationRuntime:
    """Keep inventory, health probes, stores, and the invocation broker coherent."""

    def __init__(
        self,
        workloads: Any,
        custom_agents: Any,
        credential_broker: Any,
        *,
        registry: Optional[AppCapabilityRegistry] = None,
        connections: Optional[IntegrationConnectionStore] = None,
        grants: Optional[AgentAppGrantStore] = None,
        transport: Any = None,
        clock=time.time,
        probe_ttl_seconds: float = 60.0,
    ) -> None:
        self.workloads = workloads
        self.custom_agents = custom_agents
        self.clock = clock
        self.probe_ttl_seconds = max(5.0, min(float(probe_ttl_seconds), 300.0))
        self.registry = registry or AppCapabilityRegistry()
        self.connections = connections or IntegrationConnectionStore()
        self.grants = grants or AgentAppGrantStore()
        self.reconciler = ManagedAppCapabilityReconciler(self.registry)
        self.transport = transport or ManagedAppTransport(
            self.registry, self.connections, credential_broker
        )
        self._health_probes: dict[str, dict[str, Any]] = {}
        self.broker = AppCapabilityBroker(
            self.registry,
            self.connections,
            self.grants,
            agent_facts=self._agent_revision,
            transport=self.transport,
            refresh_state=self.reconcile,
            clock=clock,
        )

    def _agent_revision(self, actor: str, agent_id: str, version: int):
        return self.custom_agents.get_version(agent_id, actor, version)

    def _inventory_apps(self) -> list[Mapping[str, Any]]:
        inventory = self.workloads.list_all()
        apps = inventory.get("apps", []) if isinstance(inventory, Mapping) else []
        if not isinstance(apps, list):
            raise RuntimeError("Managed app inventory is unavailable.")
        return [item for item in apps if isinstance(item, Mapping)]

    def _current_probe_results(self) -> dict[str, dict[str, Any]]:
        now = float(self.clock())
        for instance_id, item in list(self._health_probes.items()):
            if float(item.get("expires_at", 0)) <= now:
                self._health_probes.pop(instance_id, None)
        return {
            instance_id: {"ok": bool(item["ok"])}
            for instance_id, item in self._health_probes.items()
        }

    def reconcile(self) -> dict[str, Any]:
        return self.reconciler.reconcile(
            self._inventory_apps(), self._current_probe_results()
        )

    def _app_candidates(self, connection: Mapping[str, Any]) -> list[dict[str, Any]]:
        provider = str(connection.get("provider", ""))
        scoped = {
            value[4:]
            for value in connection.get("scopes", [])
            if isinstance(value, str) and value.startswith("app:")
        }
        items: list[dict[str, Any]] = []
        cursor = None
        while True:
            page = self.registry.list_app_instances(limit=100, cursor=cursor)
            for item in page.get("items", []):
                if item.get("app_id") != provider or item.get("state") == "removed":
                    continue
                if scoped and item.get("instance_id") not in scoped:
                    continue
                items.append(item)
            cursor = page.get("next_cursor")
            if not cursor:
                return items

    def test_connection(self, connection: Mapping[str, Any]) -> dict[str, Any]:
        """Bootstrap one pending connection with a fixed server-owned health probe."""
        self.reconcile()
        candidates = self._app_candidates(connection)
        if len(candidates) != 1:
            return {
                "healthy": False,
                "detail": "Select exactly one matching managed app for this connection.",
            }
        instance_id = str(candidates[0]["instance_id"])
        try:
            outcome = self.transport.test_connection(
                str(connection.get("actor", "")), instance_id, str(connection.get("id", ""))
            )
            healthy = bool(outcome.get("healthy", outcome.get("ok", False)))
        except Exception:
            healthy = False
        self._health_probes[instance_id] = {
            "ok": healthy,
            "expires_at": float(self.clock()) + self.probe_ttl_seconds,
        }
        self.reconcile()
        return {
            "healthy": healthy,
            "detail": (
                "Managed Grafana connection verified."
                if healthy else "The managed Grafana health check did not pass."
            ),
            "app_instance_id": instance_id,
        }

    def snapshot_for_agent(
        self, actor: str, agent_id: str, version: int, *, definition: Any = None
    ) -> dict[str, Any]:
        """Project exact grant pins and safe operation policy into a task snapshot."""
        result: list[dict[str, Any]] = []
        for grant in self.grants.list(actor, agent_id=agent_id, limit=200):
            if int(grant.get("agent_version", 0)) != int(version) or grant.get("revoked"):
                continue
            app = self.registry.get_app_instance(str(grant.get("app_instance_id", "")))
            manifest = self.registry.get_manifest(str(grant.get("manifest_digest", "")))
            if app is None or manifest is None:
                continue
            operations = []
            for operation_id in grant.get("operation_ids", []):
                operation = manifest.operation(str(operation_id))
                if operation is None:
                    continue
                operations.append({
                    "operation_id": operation.operation_id,
                    "label": operation.label,
                    "mode": operation.mode,
                    "risk": operation.risk,
                    "approval_policy": operation.approval_policy,
                })
            if operations:
                result.append({
                    **grant,
                    "app_name": app.get("app_label", manifest.app_label),
                    "manifest_version": manifest.manifest_version,
                    "app_version": manifest.app_version,
                    "operations": operations,
                })
        return {"grants": result}


__all__ = ["IntegrationRuntime"]
