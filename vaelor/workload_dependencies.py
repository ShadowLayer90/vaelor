"""Dependency-aware removal plans for managed apps and local AI resources."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any, Dict

from .managed_local_credentials import PREFIX as MANAGED_CREDENTIAL_PREFIX
from .workload_inventory import model_file_identity


RESOURCE_KINDS = {"app", "model", "runtime"}
DEPENDENCY_STRATEGIES = {"resolve", "cascade"}
REMOVAL_FIELDS = {
    "kind", "id", "display_identity", "plan_digest", "confirmation",
    "dependency_strategy", "retain_data", "create_backup",
}


class DependencyError(ValueError):
    """A removal target or approval payload is unsafe or stale."""


class WorkloadDependencyService:
    def __init__(self, inventory, credential_broker=None, custom_agents=None):
        self.inventory = inventory
        self.credential_broker = credential_broker
        self.custom_agents = custom_agents

    @staticmethod
    def _edge(kind, item_id, name, relationship, *, active=False, blocking=False):
        return {
            "kind": kind,
            "id": str(item_id),
            "name": str(name),
            "relationship": relationship,
            "active": bool(active),
            "blocking": bool(blocking),
        }

    def _credentials(self) -> list[Dict[str, Any]]:
        if self.credential_broker is None:
            return []
        try:
            return list(self.credential_broker.list())
        except (AttributeError, OSError, RuntimeError, ValueError):
            # Failing open would allow removal while assignments are unknown.
            raise DependencyError(
                "Managed AI assignments could not be inspected. Removal is blocked."
            )

    def _agents(self, actor: str) -> list[Dict[str, Any]]:
        if self.custom_agents is None:
            return []
        try:
            if hasattr(self.custom_agents, "enabled_dependencies"):
                return list(self.custom_agents.enabled_dependencies())
            if not actor:
                return []
            return list(self.custom_agents.list(actor, include_disabled=False))
        except (AttributeError, OSError, RuntimeError, ValueError):
            raise DependencyError(
                "Custom agent assignments could not be inspected. Removal is blocked."
            )

    def _runtime_edges(self, data: Dict[str, Any], actor: str):
        dependencies = []
        managed_assignments = set()
        for credential in self._credentials():
            credential_id = str(credential.get("id", ""))
            if not credential_id.startswith(MANAGED_CREDENTIAL_PREFIX):
                continue
            for purpose in credential.get("active_for", []):
                if purpose not in {"deployment-agent", "ai-chat"}:
                    continue
                managed_assignments.add(str(purpose))
                label = "Vaelor Assistant" if purpose == "deployment-agent" else "AI Chat"
                dependencies.append(self._edge(
                    "consumer", purpose, label, "uses-managed-runtime",
                    active=True, blocking=True,
                ))
        if "deployment-agent" in managed_assignments:
            dependencies.append(self._edge(
                "agent", "specialist-agents", "Specialist agents",
                "uses-assistant-model", active=True, blocking=True,
            ))
            for agent in self._agents(actor):
                dependencies.append(self._edge(
                    "agent", agent.get("id", "custom-agent"),
                    agent.get("name", "Custom agent"), "uses-assistant-model",
                    active=True, blocking=True,
                ))
        for model in data["models"]:
            if model.get("in_use"):
                dependencies.append(self._edge(
                    "model", model["id"], model["name"], "runtime-loads-model",
                    active=True, blocking=False,
                ))
        return dependencies
    @staticmethod
    def _app_identity(item: Dict[str, Any]) -> tuple[str, str]:
        """Return the recreate-safe ID and the one operator-facing name."""
        stable_id = str(item.get("app_instance_id") or item.get("id") or "").strip()
        display_identity = str(
            item.get("display_identity") or item.get("name") or stable_id
        ).strip()
        if not stable_id or not display_identity:
            raise DependencyError("The managed application has no usable identity.")
        return stable_id, display_identity

    @staticmethod
    def _model_binding(item: Dict[str, Any]) -> Dict[str, Any]:
        raw_path = str(item.get("path", "")).strip()
        if not raw_path:
            return {
                "id": str(item.get("id", "")),
                "name": str(item.get("display_identity") or item.get("name", "")),
                "file": str(item.get("file", "")),
                "path": "",
                "sha256": "",
                "size_bytes": 0,
                "mtime_ns": 0,
            }
        path = Path(raw_path).resolve()
        try:
            identity = model_file_identity(path)
        except (OSError, ValueError):
            try:
                size_bytes = int(item.get("size_bytes", 0) or 0)
            except (TypeError, ValueError):
                size_bytes = 0
            identity = {
                "path": str(path),
                "sha256": str(item.get("sha256", "")),
                "size_bytes": size_bytes,
                "mtime_ns": int(item.get("mtime_ns", 0) or 0),
            }
        return {
            "id": str(item.get("id", "")),
            "name": str(item.get("display_identity") or item.get("name", "")),
            "file": str(item.get("file", "")),
            **identity,
        }


    @staticmethod
    def _digest(value: Dict[str, Any]) -> str:
        encoded = json.dumps(value, sort_keys=True, separators=(",", ":"))
        return hashlib.sha256(encoded.encode("utf-8")).hexdigest()

    def report(self, kind: str, resource_id: str, actor: str = "") -> Dict[str, Any]:
        clean_kind = str(kind).strip().lower()
        clean_id = str(resource_id).strip()
        if clean_kind not in RESOURCE_KINDS or not clean_id:
            raise DependencyError("Choose a managed app, model, or runtime.")
        data = self.inventory.list_all()
        apps = list(data.get("apps", []))
        models = list(data.get("models", []))
        dependencies: list[Dict[str, Any]] = []
        affected: list[Dict[str, Any]] = []

        if clean_kind == "app":
            app = next(
                (
                    item for item in apps
                    if item.get("id") == clean_id
                    or item.get("app_instance_id") == clean_id
                ),
                None,
            )
            if app is None or not app.get("managed") or not app.get("project"):
                raise DependencyError("Choose a managed application.")
            project = str(app["project"])
            if project == "model-assistant":
                raise DependencyError(
                    "Manage the local AI server as a runtime so its consumers are protected."
                )
            resource_id, display_identity = self._app_identity(app)
            resource = {
                "kind": "app", "id": resource_id, "name": display_identity,
                "display_identity": display_identity, "project": project,
            }
            for sibling in apps:
                if sibling.get("project") != project:
                    continue
                sibling_id, sibling_identity = self._app_identity(sibling)
                affected.append(self._edge(
                    "service", sibling_id, sibling_identity,
                    "removed-with-compose-project", active=sibling.get("running", False),
                ))
            # Selecting any container means approving removal of its whole project.
            dependencies.extend(
                {**edge, "blocking": edge["id"] != resource_id}
                for edge in affected
                if edge["id"] != resource_id
            )
            confirmation = display_identity
            retain_supported = True
        elif clean_kind == "runtime":
            if clean_id != "model-assistant":
                raise DependencyError("Choose the managed local AI runtime.")
            compose = self.inventory.workloads_root / "model-assistant" / "compose.yaml"
            runtime_apps = [item for item in apps if item.get("project") == clean_id]
            if not compose.is_file() and not runtime_apps:
                raise DependencyError("The managed local AI runtime was not found.")
            resource = {
                "kind": "runtime", "id": clean_id,
                "name": "Managed llama.cpp runtime",
                "display_identity": "Managed llama.cpp runtime",
                "project": clean_id,
            }
            active_model = next(
                (item for item in models if item.get("in_use")), None
            )
            dependencies = self._runtime_edges(data, actor)
            if active_model is not None:
                resource["model"] = self._model_binding(active_model)
            affected = [
                self._edge(
                    "service", self._app_identity(item)[0],
                    self._app_identity(item)[1],
                    "runtime-service", active=item.get("running", False),
                )
                for item in runtime_apps
            ]
            confirmation = resource["display_identity"]
            retain_supported = True
        else:
            model = next((item for item in models if item.get("id") == clean_id), None)
            if model is None:
                raise DependencyError("Choose a downloaded managed model.")
            display_identity = str(model.get("display_identity") or model["name"])
            resource = {
                "kind": "model", "name": display_identity,
                "display_identity": display_identity,
                **self._model_binding(model),
            }
            if model.get("in_use"):
                dependencies.append(self._edge(
                    "runtime", "model-assistant", "Managed llama.cpp runtime",
                    "loads-model", active=True, blocking=True,
                ))
                dependencies.extend(self._runtime_edges(data, actor))
                dependencies = [
                    edge for edge in dependencies
                    if not (edge["kind"] == "model" and edge["id"] == clean_id)
                ]
            confirmation = display_identity
            retain_supported = False

        dependencies = sorted(
            dependencies,
            key=lambda item: (item["kind"], item["id"], item["relationship"]),
        )
        affected = sorted(affected, key=lambda item: (item["kind"], item["id"]))
        blocking = [item for item in dependencies if item["blocking"]]
        graph = {
            "resource": resource,
            "dependencies": dependencies,
            "affected_resources": affected,
            "display_identity": confirmation,
            "confirmation": confirmation,
        }
        return {
            **graph,
            "blocked": bool(blocking),
            "plan_digest": self._digest(graph),
            "defaults": {
                "dependency_strategy": None,
                "retain_data": retain_supported,
                "create_backup": True,
            },
            "requirements": {
                "dependency_strategy_required": True,
                "dependency_strategies": sorted(DEPENDENCY_STRATEGIES),
                "cascade_required": bool(blocking),
                "retain_data_supported": retain_supported,
                "backup_supported": True,
            },
            "disclosures": [
                "Active dependents block removal unless cascade is explicitly approved.",
                "Backup, dependent removal, and retained data are separate choices.",
                "The dependency graph is checked again immediately before execution.",
            ],
        }

    def validate_removal(self, payload: Any, actor: str = "") -> Dict[str, Any]:
        if not isinstance(payload, dict) or set(payload) != REMOVAL_FIELDS:
            raise DependencyError(
                "Submit the exact reviewed removal payload, including every disclosure choice."
            )
        for field in ("retain_data", "create_backup"):
            if type(payload[field]) is not bool:  # bool is intentionally exact here.
                raise DependencyError("Removal disclosure choices must be true or false.")
        if payload["dependency_strategy"] not in DEPENDENCY_STRATEGIES:
            raise DependencyError("Choose how Vaelor should handle dependencies before approving removal.")
        report = self.report(payload["kind"], payload["id"], actor)
        if payload["plan_digest"] != report["plan_digest"]:
            raise DependencyError(
                "Dependencies changed after review. Load a fresh removal plan."
            )
        if payload["id"] != report["resource"]["id"]:
            raise DependencyError("The reviewed resource is no longer the same managed resource.")
        if payload["display_identity"] != report["display_identity"]:
            raise DependencyError("The displayed resource identity changed. Load a fresh removal plan.")
        if payload["confirmation"] != report["display_identity"]:
            raise DependencyError("Type the displayed resource name to confirm removal.")
        if report["blocked"] and payload["dependency_strategy"] != "cascade":
            raise DependencyError(
                "Active dependents block removal. Review them and explicitly approve cascade removal."
            )
        if not report["requirements"]["retain_data_supported"] and payload["retain_data"]:
            raise DependencyError("A model file cannot be removed while retaining that file.")
        if not report["blocked"] and payload["dependency_strategy"] == "cascade":
            raise DependencyError(
                "Cascade removal is only available when an active dependency is present."
            )
        return report
