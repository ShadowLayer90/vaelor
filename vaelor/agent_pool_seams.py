"""The injected seams of cluster-agent orchestration, and its error.

Moved out of :mod:`vaelor.agent_pool_operations` (at the line ceiling) so that
module keeps the deploy/relaunch/remove sequence and nothing else: the default
launch bridge, port allocator, health probe, recorded-backing resolve and
pre-launch runtime check each have one home here, and a test replaces any of
them through the operations constructor.
"""

from __future__ import annotations

import threading
from typing import Any, Dict, Tuple

from .agent_backing import BackingUnavailable, resolve_recorded_backing
from .mcp_client import NAMESPACE

#: Where a deployed agent's loopback OpenAI runtime binds - fixed to this host so
#: the runtime is never LAN-open on its own, only through the inbound gate.
RUNTIME_LOOPBACK_HOST = "127.0.0.1"

_BACKING_UNAVAILABLE = (
    "This agent is backed by {}, which is not available on the cluster right now."
)
_NO_FREE_PORT = (
    "No loopback port is free in the cluster agent band for a new deployment."
)


#: Serializes every change to what runs for a deployed agent in the executor:
#: a deploy, a remove and a whole reconcile pass (B1). Without it a pass that
#: listed a row as healthy could relaunch it while a remove was stopping it,
#: leaving a running agent no row describes, admitting keys the remove revoked.
AGENT_LIFECYCLE_LOCK = threading.RLock()


class AgentPoolError(RuntimeError):
    """A safe, user-presentable cluster-agent orchestration error."""


def _default_bridge() -> Any:
    """The production launch bridge: the root ``agent_start`` / ``agent_stop`` client.

    Imported lazily so importing this module never opens a socket and a test that
    injects a fake bridge never constructs the real one.
    """
    from .hardware_bridge import HardwareBridgeClient

    return HardwareBridgeClient()


def _default_allocate_port(taken: Tuple[int, ...] = ()) -> int:
    """Pick a free loopback port in the agent band by bind-probing each in turn.

    The band and its bound are :mod:`vaelor.agent_service`'s to own; this reads
    them so the two cannot drift. A port already handed out in this same deploy
    pass is skipped via ``taken`` even before the OS would report it busy.
    """
    import socket

    from .agent_service import AGENT_PORT_BAND_END, AGENT_PORT_BAND_START

    for port in range(AGENT_PORT_BAND_START, AGENT_PORT_BAND_END + 1):
        if port in taken:
            continue
        probe = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        try:
            probe.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
            probe.bind((RUNTIME_LOOPBACK_HOST, port))
        except OSError:
            continue
        finally:
            probe.close()
        return port
    raise AgentPoolError(_NO_FREE_PORT)


def _default_probe(url: str) -> bool:
    """Whether the inbound gate answers its unauthenticated ``/health`` with 200.

    Any trouble is reported as not-healthy, the fail-safe direction: the deploy
    then waits or rolls back rather than promoting a runtime that never answered.
    """
    import urllib.error
    import urllib.request

    try:
        with urllib.request.urlopen(url, timeout=5) as response:  # noqa: S310 - loopback/LAN gate
            return int(getattr(response, "status", 200)) == 200
    except (OSError, urllib.error.URLError, ValueError):
        return False


def _resolve_recorded(broker: Any, cluster_store: Any, model_deployment_name: str) -> Dict[str, Any]:
    """The RECORDED backing deployment's ``{base_url, model, api_key}`` profile.

    Resolved through that deployment's own credential, never the
    ``cluster-inference`` lease, which follows whichever deployment is active
    (ACC-075: the agent must use the deployment it was deployed against and
    kept warm for). A gone deployment names itself; a broker that cannot
    resolve the credential is the honest "backing is not available" signal.
    """
    from .credential_broker_client import CredentialError

    if cluster_store is None:
        raise AgentPoolError(_BACKING_UNAVAILABLE.format(model_deployment_name))
    try:
        return resolve_recorded_backing(broker, cluster_store, model_deployment_name)
    except BackingUnavailable as error:
        raise AgentPoolError(str(error)) from error
    except CredentialError as error:
        raise AgentPoolError(
            _BACKING_UNAVAILABLE.format(model_deployment_name)
        ) from error


def _split_endpoint(endpoint: str) -> Tuple[str, int]:
    """The advertise host and loopback-fronting port from an ``.../v1`` endpoint.

    The reboot reconcile re-renders a healthy row from what the row already
    holds; the endpoint the deploy recorded carries both the LAN address the
    gate binds and the runtime port it fronts, so the relaunch reuses them
    rather than allocating a new port a live client would not be reaching.
    """
    import urllib.parse

    parts = urllib.parse.urlsplit(str(endpoint or ""))
    return str(parts.hostname or ""), int(parts.port or 0)


def _default_validate_runtime(config_content: str) -> None:
    """Build the runtime from ``config_content`` exactly as its unit will.

    ``agent_server.build_runtime`` is the function the unit's process runs at
    start; running it here, in process, before the launch means a config it
    refuses (an MCP server with no usable address, a model endpoint it will not
    call, an acting permission) fails the deploy with that refusal rather than
    a crash-looping unit. It binds no socket and calls nothing.
    """
    import json

    from .agent_server import build_runtime

    build_runtime(json.loads(config_content))


def _function_spec(server_name: str, tool: str) -> Dict[str, Any]:
    """A minimal OpenAI function spec for one effective ``mcp.<server>.<tool>`` pair.

    The catalog holds tool NAMES, not per-tool descriptions, so a generic
    description is used; the parameters are left open. Only an effective pair ever
    reaches here, so the model is only ever shown a granted, in-effect tool.
    """
    qualified = "{}{}.{}".format(NAMESPACE, server_name, tool)
    return {
        "type": "function",
        "function": {
            "name": qualified,
            "description": "Call the {} tool on the {} MCP server.".format(
                tool, server_name
            ),
            "parameters": {"type": "object"},
        },
    }
