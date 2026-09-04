"""Composable authenticated v2 API for the Vaelor control plane."""

from typing import Any, Dict, Optional

from flask import Blueprint

from .api_assistant_routes import register_assistant_routes
from .api_assistant_setup_routes import register_assistant_setup_routes
from .api_agent_operation_routes import register_agent_operation_routes
from .api_alert_routes import register_alert_routes
from .api_backup_routes import register_backup_routes
from .api_workload_act_routes import register_workload_act_routes
from .api_application_routes import register_application_routes
from .api_auth_routes import register_auth_routes
from .api_common import ApiContext
from .api_chat_routes import register_chat_routes
from .api_cluster_routes import register_cluster_routes
from .chat_turn_dedupe import ChatTurnDedupe
from .api_custom_connector_routes import register_custom_connector_routes
from .api_enclosure_routes import register_enclosure_routes
from .api_hardware_routes import register_hardware_routes
from .api_integration_routes import register_integration_routes
from .api_mcp_routes import register_mcp_routes
from .api_operation_routes import register_operation_routes
from .api_upgrade_routes import register_upgrade_routes
from .api_workload_routes import register_workload_routes
from .api_web_research_routes import register_web_research_routes
from .security import SecurityStore


def create_api_v2_blueprint(
    callbacks: Dict[str, Any],
    store: Optional[SecurityStore] = None,
) -> Blueprint:
    """Build the API from independently replaceable route domains."""
    # One dedupe store, shared by both chat POST route modules, so a retried
    # send that reaches either endpoint replays the accepted turn instead of
    # starting a second inference (VD-112 follow-up). setdefault leaves a caller
    # that pre-wired its own instance untouched.
    callbacks.setdefault("chat_turn_dedupe", ChatTurnDedupe())
    context = ApiContext(callbacks, store)
    register_auth_routes(context)
    register_hardware_routes(context)
    register_enclosure_routes(context)
    register_assistant_setup_routes(context)
    register_assistant_routes(context)
    register_alert_routes(context)
    register_backup_routes(context)
    register_agent_operation_routes(context)
    register_workload_act_routes(context)
    register_custom_connector_routes(context)
    register_integration_routes(context)
    register_application_routes(context)
    register_web_research_routes(context)
    register_chat_routes(context)
    register_cluster_routes(context)
    register_workload_routes(context)
    register_operation_routes(context)
    register_upgrade_routes(context)
    register_mcp_routes(context)
    return context.blueprint
