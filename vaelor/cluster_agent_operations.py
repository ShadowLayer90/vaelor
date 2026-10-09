"""The cluster-agent half of :class:`vaelor.cluster_operations.ClusterOperations`.

Moved out of that module (at the line ceiling) as a mixin, behaviour-preserving,
so the agent tier could gain its live read (ACC-071): the deployed agents'
list and detail now carry each agent's ACTIVE keys and a ``runtime`` read off
what is running (``agent_runtime_state``), instead of the row's stored state
and a placeholder key. The mixin reads ``self.store`` (the cluster store, where
an agent's backing deployment is recorded) and ``self.broker`` from the class
it is mixed into.
"""

from __future__ import annotations

import sqlite3
from typing import Any, Dict, Optional

from .agent_deployments import (
    AgentDeploymentStore,
    STATE_FAILED as AGENT_STATE_FAILED,
    STATE_REMOVING as AGENT_STATE_REMOVING,
)
from .agent_pool_operations import (
    AgentPoolOperations,
    AGENT_ENDPOINT_PREFIX,
    DEPLOY_CONFIRM as AGENT_DEPLOY_CONFIRM,
    REMOVE_CONFIRM as AGENT_REMOVE_CONFIRM,
)
from .agent_reconcile import broker_listing
from .agent_runtime_state import live_agent_view
from .agent_service import AGENT_PORT_BAND_END, AGENT_PORT_BAND_START
from .cluster_placement import CONTROLLER_PLACEMENT_ID
from .custom_agents import CustomAgentStore
from .mcp_catalog import McpCatalogStore
from .skills_library import SkillsLibraryStore


def _agent_deployment_view(row: Dict[str, Any]) -> Dict[str, Any]:
    """The GET-safe read model of one agent deployment row.

    Carries the endpoint, state, backing cluster model, the inbound key's
    registry id and the canonical broker endpoint id that key's rotate and
    revoke routes address - and NEVER a key value, because the store holds
    only the id.
    Placement is honest: a deployed agent runs on THIS controller behind its
    inbound gate, so that is what is reported rather than the worker id the
    request happened to ask for.
    """
    row = row or {}
    # The addressing id the inbound key's routes use, built exactly as mint
    # does (AGENT_ENDPOINT_PREFIX + the row id). Non-secret, like api_key_id;
    # blank only for a malformed row with no id, so the UI offers no key
    # actions it could not route.
    row_id = row.get("id")
    endpoint_id = AGENT_ENDPOINT_PREFIX + str(row_id) if row_id else ""
    return {
        "name": row.get("name"),
        "state": row.get("state"),
        "endpoint": row.get("endpoint"),
        "backing": row.get("backing", {}),
        "custom_agent_id": row.get("custom_agent_id"),
        "custom_agent_version": row.get("custom_agent_version"),
        "api_key_id": row.get("api_key_id", ""),
        "endpoint_id": endpoint_id,
        "placement": {"runs_on": CONTROLLER_PLACEMENT_ID},
        "created_at": row.get("created_at"),
        "updated_at": row.get("updated_at"),
    }


class ClusterAgentOperationsMixin:
    """Deploy, remove, reconcile and read the cluster's deployed agents."""

    store: Any
    broker: Any

    # -- cluster agents (F4b-ii) ---------------------------------------------

    def deploy_agent(self, payload: Dict[str, Any], progress=None) -> Dict[str, Any]:
        # Composes the agent orchestration with the real collaborators and
        # delegates, exactly as `deploy_llm` composes and delegates to
        # `GpuPoolOperations`. The runtime and its inbound gate run on THIS
        # controller (loopback fronted by the gate on the advertise address the
        # route injected), so placement is the controller, not a worker.
        if payload.get("confirm") != AGENT_DEPLOY_CONFIRM:
            raise ValueError("Confirm the cluster agent deployment before dispatching it.")
        operations = self._agent_pool_operations()
        result = operations.deploy(payload, progress)
        actor = str(payload.get("actor", "") or "")
        if actor and isinstance(result, dict) and result.get("name"):
            # Persist the deploying admin so the reboot reconcile can re-read the
            # pinned definition when it re-renders this agent (SF-A). Best-effort:
            # a healthy deploy is never failed over a missed actor stamp.
            try:
                operations.store.set_actor(result["name"], actor)
            except Exception:  # noqa: BLE001
                pass
        return result

    def remove_agent(self, payload: Dict[str, Any]) -> Dict[str, Any]:
        if payload.get("confirm") != AGENT_REMOVE_CONFIRM:
            raise ValueError("Confirm the cluster agent removal before dispatching it.")
        return self._agent_pool_operations().remove(payload)

    def reconcile_cluster_agents(self, job_store: Any = None) -> Dict[str, Any]:
        # Restart-on-boot AND restart-on-failure (SF-A, ACC-070/071/074): each
        # healthy agent's runtime and gate are converged to its CURRENT keys
        # and surface, read off what is running (`agent_reconcile`).
        from .agent_reconcile import pending_deploy_ids, reconcile_cluster_agents as _reconcile

        try:
            pending = pending_deploy_ids(job_store) if job_store is not None else None
        except (OSError, ValueError, sqlite3.Error):
            pending = None
        if pending is None:
            # Without the job ledger no first key can be known to be orphaned.
            return _reconcile(self._agent_pool_operations(), sweep_orphans=False)
        return _reconcile(self._agent_pool_operations(), pending_deployment_ids=pending)

    def list_agent_deployments(self, *, live: bool = True) -> list:
        """Every agent row's safe view, with its keys and live ``runtime``.

        ``live`` reads the broker once and, per agent, the bridge and the
        runtime's loopback health (each bounded); ``False`` is the stored view
        alone, for callers that need no reading of what runs.
        """
        rows = self._agent_deployment_store().list()
        if not live:
            return [_agent_deployment_view(row) for row in rows]
        operations = self._agent_pool_operations()
        listing = broker_listing(self.broker)
        return [
            {**_agent_deployment_view(row),
             **live_agent_view(operations, row, listing, bridge=self._agent_bridge())}
            for row in rows
        ]

    def get_agent_deployment(self, name: str) -> Optional[Dict[str, Any]]:
        row = self._agent_deployment_store().get(str(name))
        if not row:
            return None
        listing = broker_listing(self.broker)
        return {
            **_agent_deployment_view(row),
            **live_agent_view(
                self._agent_pool_operations(), row, listing, bridge=self._agent_bridge(),
            ),
        }

    def _agent_bridge(self) -> Any:
        """The root bridge client the live read asks; a test sets ``agent_bridge``."""
        bridge = getattr(self, "agent_bridge", None)
        if bridge is None:
            from .hardware_bridge_client import HardwareBridgeClient

            bridge = HardwareBridgeClient()
            self.agent_bridge = bridge
        return bridge

    def _agent_pool_operations(self) -> AgentPoolOperations:
        return AgentPoolOperations(
            store=self._agent_deployment_store(),
            broker=self.broker,
            catalog=self._agent_mcp_catalog(),
            custom_agents=self._agent_custom_agents(),
            skills_library=self._agent_skills_library(),
            keys=self.broker,
            allocate_port=self._allocate_agent_port,
            # The recorded backing deployment lives in the cluster store: the
            # agent resolves ITS credential (ACC-075), and the surface digest
            # names it.
            cluster_store=self.store,
            bridge=getattr(self, "agent_bridge", None),
        )

    def _agent_deployment_store(self) -> AgentDeploymentStore:
        if getattr(self, "_agent_store", None) is None:
            self._agent_store = AgentDeploymentStore()
        return self._agent_store

    def _agent_mcp_catalog(self) -> McpCatalogStore:
        if getattr(self, "_agent_catalog", None) is None:
            self._agent_catalog = McpCatalogStore()
        return self._agent_catalog

    def _agent_skills_library(self) -> SkillsLibraryStore:
        if getattr(self, "_agent_skills", None) is None:
            self._agent_skills = SkillsLibraryStore()
        return self._agent_skills

    def _agent_custom_agents(self) -> CustomAgentStore:
        if getattr(self, "_agent_custom", None) is None:
            self._agent_custom = CustomAgentStore()
        return self._agent_custom

    def _allocate_agent_port(self) -> int:
        # The lowest free port in the agent band not held by a non-terminal
        # agent row, derived by parsing each row's endpoint - so two live
        # agents never collide and a torn-down one's port is reusable.
        import urllib.parse

        used = set()
        for row in self._agent_deployment_store().list():
            if str(row.get("state")) in {AGENT_STATE_FAILED, AGENT_STATE_REMOVING}:
                continue
            port = urllib.parse.urlsplit(str(row.get("endpoint") or "")).port
            if port:
                used.add(int(port))
        for candidate in range(AGENT_PORT_BAND_START, AGENT_PORT_BAND_END + 1):
            if candidate not in used:
                return candidate
        raise ValueError("Every port in the cluster agent band is already in use.")
