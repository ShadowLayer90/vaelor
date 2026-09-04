"""Discovery and reconciliation for capabilities on managed Compose apps.

Only the managed Compose project/service pair identifies an app instance. A
Docker container ID is retained as runtime evidence and is never used for
capability identity or grant lookup.
"""

from __future__ import annotations

import re
from collections.abc import Iterable, Mapping
from typing import Any

from .app_capability_manifest import AppCapabilityManifest, AppOperation
from .app_capability_registry import AppCapabilityRegistry, RegistryError, app_instance_id_for_workload
from .app_catalog import APP_TEMPLATES

MANAGED_TEMPLATE_LABEL = "io.vaelor.template"
#: Pironman-era label still carried by containers deployed before the rename.
#: Read only as a fallback; never emitted on new workloads (CLAUDE.md forbids
#: new dependencies on legacy names).
LEGACY_MANAGED_TEMPLATE_LABEL = "io.pironman.template"
_PORT_CONTAINER_RE = re.compile(r"^(\d{1,5})/(tcp|udp|sctp)$")
_TEMPLATE_ID_RE = re.compile(r"^[a-z0-9][a-z0-9.-]{0,63}$")


def template_id_from_labels(labels: Mapping[str, Any]) -> Any:
    """Return the managed-template label value, preferring the Vaelor key.

    Already-deployed containers may still carry the legacy
    ``io.pironman.template`` label, so fall back to it when the Vaelor label is
    absent.
    """
    if not isinstance(labels, Mapping):
        return None
    if MANAGED_TEMPLATE_LABEL in labels:
        return labels[MANAGED_TEMPLATE_LABEL]
    return labels.get(LEGACY_MANAGED_TEMPLATE_LABEL)


def safe_published_ports(value: Any) -> list[dict[str, int]]:
    """Project Docker port bindings without addresses, URLs, or free-form text."""
    if not isinstance(value, (list, tuple)) or len(value) > 32:
        raise ValueError("published ports must be a bounded list")
    projected: list[dict[str, int]] = []
    seen: set[tuple[int, int]] = set()
    for item in value:
        if not isinstance(item, Mapping):
            raise ValueError("published port entries must be objects")
        container = item.get("container", item.get("container_port"))
        match = _PORT_CONTAINER_RE.fullmatch(container) if isinstance(container, str) else None
        if match is None:
            raise ValueError("container port is malformed")
        container_port = int(match.group(1))
        host = item.get("host", item.get("host_port"))
        if isinstance(host, bool) or not isinstance(host, (str, int)):
            raise ValueError("host port must be numeric")
        host_text = str(host)
        if not host_text.isdigit():
            raise ValueError("host port must be numeric")
        host_port = int(host_text)
        if not 1 <= container_port <= 65535 or not 1 <= host_port <= 65535:
            raise ValueError("ports must be between 1 and 65535")
        pair = (container_port, host_port)
        if pair not in seen:
            seen.add(pair)
            projected.append({"container_port": container_port, "host_port": host_port})
    return projected


def _safe_template_id(value: Any) -> str | None:
    if not isinstance(value, str):
        return None
    value = value.strip().lower()
    return value if _TEMPLATE_ID_RE.fullmatch(value) else None


def _managed_workload_parts(app: Mapping[str, Any]) -> tuple[str, str] | None:
    if app.get("managed") is not True:
        return None
    project, service = app.get("project"), app.get("service")
    if not isinstance(project, str) or not isinstance(service, str):
        return None
    try:
        app_instance_id_for_workload(project, service)
    except (RegistryError, TypeError):
        return None
    return project.strip(), service.strip()


def stable_managed_app_id(app: Mapping[str, Any]) -> str | None:
    """Derive an ID only from a validated managed project/service pair."""
    parts = _managed_workload_parts(app)
    return app_instance_id_for_workload(*parts) if parts else None


def _grafana_manifest() -> AppCapabilityManifest:
    """Secret-free Grafana policy; transport routes live outside manifests."""
    return AppCapabilityManifest(
        app_id="grafana",
        app_label="Grafana",
        app_version="curated",
        manifest_version=1,
        operations=(
            AppOperation(
                operation_id="read_health", label="Read Grafana health", mode="read",
                output_schema={"type": "object"}, requires_connection=True,
            ),
            AppOperation(
                operation_id="read_dashboards", label="List Grafana dashboards", mode="read",
                output_schema={"type": "array"}, requires_connection=True,
            ),
            AppOperation(
                operation_id="write_annotation", label="Create a Grafana annotation", mode="write",
                input_schema={"type": "object", "required": ["text"]},
                output_schema={"type": "object"}, risk="medium", approval_policy="operator",
                requires_connection=True,
            ),
        ),
        health_probe={"operation_id": "read_health", "kind": "http_api"},
        requires_connection=True,
        metadata={
            "connection_policy": "credential_broker_managed",
            "transport": "server_owned_grafana_adapter",
        },
    )


def builtin_manifests() -> dict[str, AppCapabilityManifest]:
    """Return only curated manifests whose operations are explicitly known."""
    return {"grafana": _grafana_manifest()}


def manifest_for_template(template_id: Any) -> AppCapabilityManifest | None:
    template_id = _safe_template_id(template_id)
    if template_id is None or template_id not in APP_TEMPLATES:
        return None
    return builtin_manifests().get(template_id)


def _health_probe_outcome(value: Any) -> tuple[bool | None, dict[str, Any]]:
    """Accept only server-owned probe status and bounded numeric evidence."""
    if isinstance(value, bool):
        return value, {"ok": value}
    if not isinstance(value, Mapping) or not isinstance(value.get("ok"), bool):
        return None, {"ok": None, "valid": False}
    evidence: dict[str, Any] = {"ok": value["ok"]}
    status_code = value.get("status_code")
    if isinstance(status_code, int) and not isinstance(status_code, bool) and 100 <= status_code <= 599:
        evidence["status_code"] = status_code
    return value["ok"], evidence


def _runtime_state(
    app: Mapping[str, Any],
    manifest: AppCapabilityManifest,
    probe_result: Any,
) -> tuple[str, str, str, str, dict[str, Any]]:
    """Map lifecycle and explicit server health evidence to fail-closed state."""
    running = app.get("running") is True
    health = str(app.get("health") or "unknown").strip().lower()
    status = str(app.get("status") or "unknown").strip().lower()
    if not running or status in {"created", "exited", "dead", "removing"}:
        return "stopped", "unknown", "compatible", "Managed workload is not running.", {}
    if manifest.health_probe:
        outcome, evidence = _health_probe_outcome(probe_result)
        if outcome is True:
            return "active", "healthy", "compatible", "", evidence
        if outcome is False:
            return "degraded", "unhealthy", "compatible", "Server-owned health probe failed.", evidence
        return "degraded", "unknown", "compatible", "Server-owned health probe has not succeeded.", evidence
    if health == "healthy":
        return "active", "healthy", "compatible", "", {}
    if health == "unhealthy":
        return "degraded", "unhealthy", "compatible", "Managed workload health is unhealthy.", {}
    return "degraded", "unknown", "compatible", "Managed workload health is not verified.", {}


def reconcile_managed_workloads(
    registry: AppCapabilityRegistry,
    inventory_apps: Iterable[Mapping[str, Any]],
    health_probe_results: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """Reconcile known managed apps and fail closed for absent registrations."""
    current_ids: set[str] = set()
    registered: list[dict[str, Any]] = []
    updated: list[dict[str, Any]] = []
    unsupported: list[dict[str, Any]] = []
    for app in inventory_apps:
        if not isinstance(app, Mapping) or app.get("managed") is not True:
            continue
        instance_id = stable_managed_app_id(app)
        if instance_id is None:
            unsupported.append({"reason": "invalid managed project/service identity"})
            continue
        current_ids.add(instance_id)
        template_id = _safe_template_id(app.get("template_id"))
        manifest = manifest_for_template(template_id)
        if manifest is None:
            unsupported.append({
                "instance_id": instance_id, "template_id": template_id,
                "project": app.get("project"), "service": app.get("service"),
                "reason": "No curated capability manifest exists for this template.",
            })
            existing = registry.get_app_instance(instance_id)
            if existing is not None:
                updated.append(registry.reconcile_app_instance(
                    instance_id, state="incompatible", health="unknown",
                    compatibility="incompatible",
                    reason="The installed template is unsupported; capabilities are blocked.",
                    health_evidence={"template_id": template_id},
                ))
            continue
        probe_result = health_probe_results.get(instance_id) if isinstance(health_probe_results, Mapping) else None
        state, health, compatibility, reason, probe_evidence = _runtime_state(app, manifest, probe_result)
        try:
            published_ports = safe_published_ports(app.get("published_ports", []))
            published_ports_valid = True
        except ValueError:
            published_ports = []
            published_ports_valid = False
        evidence = {
            "status": str(app.get("status") or "unknown")[:32],
            "running": bool(app.get("running")), "template_id": template_id,
            "published_ports": published_ports,
            "published_ports_valid": published_ports_valid,
        }
        if manifest.health_probe:
            evidence["health_probe"] = probe_evidence
        runtime_container_id = app.get("runtime_container_id") or app.get("id")
        if isinstance(runtime_container_id, str) and runtime_container_id:
            evidence["runtime_container_id"] = runtime_container_id[:256]
        existing = registry.get_app_instance(instance_id)
        if existing is not None and existing["state"] == "removed":
            # Removal is an explicit grant boundary. Reappearance must be
            # re-registered by an operator before any old grant can recover.
            updated.append(registry.reconcile_app_instance(
                instance_id, state="removed", health="unknown",
                reason="The workload reappeared after removal; explicit re-registration is required.",
                health_evidence={**evidence, "reappeared_after_removal": True},
            ))
            continue
        if existing is None:
            registered.append(registry.register_app_instance(
                project=str(app["project"]), service=str(app["service"]), manifest=manifest,
                state=state, health=health, compatibility=compatibility,
                health_evidence=evidence,
            ))
        else:
            updated.append(registry.reconcile_app_instance(
                instance_id, state=state, health=health, compatibility=compatibility,
                observed_manifest_digest=manifest.manifest_digest,
                runtime_container_id=(runtime_container_id if isinstance(runtime_container_id, str) else None),
                health_evidence=evidence, reason=reason,
            ))
    removed: list[dict[str, Any]] = []
    cursor = None
    while True:
        page = registry.list_app_instances(limit=100, cursor=cursor)
        for item in page.get("items", []):
            instance_id = item["instance_id"]
            if instance_id in current_ids or item["state"] == "removed":
                continue
            removed.append(registry.reconcile_app_instance(
                instance_id, state="removed", health="unknown",
                reason="Managed workload is absent from the current inventory.",
                health_evidence={"present": False},
            ))
        cursor = page.get("next_cursor")
        if not cursor:
            break
    return {"registered": registered, "updated": updated, "removed": removed, "unsupported": unsupported}


class ManagedAppCapabilityReconciler:
    """Small injectable adapter for control-plane composition."""

    def __init__(self, registry: AppCapabilityRegistry):
        self.registry = registry

    def reconcile(self, inventory_apps: Iterable[Mapping[str, Any]], health_probe_results: Mapping[str, Any] | None = None) -> dict[str, Any]:
        return reconcile_managed_workloads(self.registry, inventory_apps, health_probe_results)


__all__ = [
    "LEGACY_MANAGED_TEMPLATE_LABEL", "MANAGED_TEMPLATE_LABEL", "ManagedAppCapabilityReconciler",
    "builtin_manifests", "manifest_for_template", "reconcile_managed_workloads", "safe_published_ports",
    "stable_managed_app_id", "template_id_from_labels",
]
