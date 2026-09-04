"""Dispatch a requested cluster action to the builder that plans it."""

from __future__ import annotations

from typing import Any, Callable, Dict

from .cluster_plan_contract import ClusterPlanContext, ClusterPlanError
from .cluster_plan_membership import (
    plan_evict_mismatched,
    plan_initialize,
    plan_join_node,
    plan_node_availability,
    plan_remove_node,
)
from .cluster_plan_inference import plan_deploy_llm, plan_remove_pooled
from .cluster_plan_services import (
    plan_backup_service,
    plan_configure_service,
    plan_deploy_app,
    plan_remove_service,
    plan_restore_service,
    plan_service_update,
)


PlanBuilder = Callable[[str, Dict[str, Any], ClusterPlanContext], Dict[str, Any]]

PLAN_BUILDERS: Dict[str, PlanBuilder] = {
    "initialize": plan_initialize,
    "join-node": plan_join_node,
    "drain-node": plan_node_availability,
    "resume-node": plan_node_availability,
    "remove-node": plan_remove_node,
    "evict-mismatched": plan_evict_mismatched,
    "deploy-llm": plan_deploy_llm,
    "remove-pooled": plan_remove_pooled,
    "deploy-app": plan_deploy_app,
    "remove-service": plan_remove_service,
    "restart-service": plan_service_update,
    "refresh-service": plan_service_update,
    "rollback-service": plan_service_update,
    "configure-service": plan_configure_service,
    "backup-service": plan_backup_service,
    "restore-service": plan_restore_service,
}


def build_cluster_plan(
    action: str, body: Dict[str, Any], context: ClusterPlanContext
) -> Dict[str, Any]:
    """Build the approval plan for `action`, or raise `ClusterPlanError`."""
    builder = PLAN_BUILDERS.get(action)
    if builder is None:
        raise ClusterPlanError(
            "unsupported_cluster_action",
            "Choose a supported cluster action.",
        )
    return {"action": action, **builder(action, body, context)}
