"""The refusal type, shared context, and guards every cluster plan builder uses."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, Set


MANAGED_SERVICE_PREFIXES = (
    "vaelor-app-", "vaelor-llm-",
)


class ClusterPlanError(Exception):
    """A refusal a plan builder wants rendered as an API error."""

    def __init__(self, code: str, message: str, status: int = 400):
        super().__init__(message)
        self.code = code
        self.message = message
        self.status = status


@dataclass
class ClusterPlanContext:
    """Everything a plan builder may read while planning one action."""

    manager: Any
    summary: Dict[str, Any] = field(default_factory=dict)
    callbacks: Dict[str, Any] = field(default_factory=dict)

    def live_service_names(self) -> Set[str]:
        return {
            str(item.get("name", ""))
            for item in self.summary.get("runtime", {}).get("services", [])
        }

    def enrolled_node(self, node_id: Any, message: str) -> Dict[str, Any]:
        """Resolve an enrolled node record, or refuse with `message`."""
        node = self.manager.store.get_node(str(node_id))
        if node is None:
            raise ClusterPlanError("cluster_node_not_found", message, status=404)
        return node


def managed_service_name(body: Dict[str, Any], context: ClusterPlanContext) -> str:
    """Resolve a live, Vaelor-managed service name, or refuse."""
    service_name = str(body.get("service_name", "")).strip()
    if (
        service_name not in context.live_service_names()
        or not service_name.startswith(MANAGED_SERVICE_PREFIXES)
    ):
        raise ClusterPlanError(
            "cluster_service_not_found",
            "Choose a Vaelor-managed cluster service.",
            status=404,
        )
    return service_name
