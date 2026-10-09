"""The root bridge's cluster-agent verbs, moved out of :mod:`vaelor.hardware_bridge`.

That module sat at the 1,000-line ceiling, and the agent tier needed two more
verbs (ACC-070): ``agent_status`` - what the running unit and gate are running
WITH (the gate's key-set hash and the agent config's surface digest, never a
key) - and ``agent_rekey`` - replace only the gate with the endpoint's current
key set. The start and stop verbs moved here unchanged beside them.

Each is a dedicated verb, NOT a ``bridge_argv_policy`` widening (the
same-allowlist failure): the bridge-FIXED ExecStart and every validation live
in :mod:`vaelor.agent_service`, and ``run_argv`` is never reached. The gate
image is pulled OUTSIDE the bridge lock (a pull can run for minutes); the lock
guards only the fast launch. The launcher is stateless (keyed by name), so a
fresh one is built per call.
"""

from __future__ import annotations

from typing import Any, Dict

#: The verbs :func:`dispatch_agent_action` answers.
AGENT_ACTIONS = frozenset({"agent_start", "agent_stop", "agent_status", "agent_rekey"})


class AgentBridgeVerbs:
    """Mixed into the bridge runtime, which supplies ``self._lock``."""

    _lock: Any

    def agent_start(self, name: Any, port: Any, config_content: Any, gate: Any) -> Dict[str, Any]:
        """Launch one deployed cluster agent: its loopback unit and inbound gate."""
        from .agent_service import AgentServerProcess

        server = AgentServerProcess()
        server.ensure_gate_image()
        with self._lock:
            return server.start(
                name=str(name), port=port,
                config_content=str(config_content or ""),
                gate=gate, skip_ensure=True,
            )

    def agent_stop(self, name: Any) -> Dict[str, Any]:
        """Stop one agent's unit and gate and unlink its configs, under ``self._lock``."""
        from .agent_service import AgentServerProcess

        with self._lock:
            return AgentServerProcess().stop(str(name))

    def agent_rekey(self, name: Any, port: Any, gate: Any) -> Dict[str, Any]:
        """Replace one agent's gate with its current key set; the runtime keeps serving.

        ``gate["pull"] is False`` (the console's in-request re-key, S5) never
        pulls the image: a missing one refuses at once, and the reconcile - which
        may pull - applies the change instead.
        """
        from .agent_service import AgentLaunchError, AgentServerProcess

        server = AgentServerProcess()
        if isinstance(gate, dict) and gate.get("pull") is False:
            if not server.gate_image_present():
                raise AgentLaunchError("The gate image is not present; the reconcile applies this change.")
        else:
            server.ensure_gate_image()
        with self._lock:
            return server.rekey(name=str(name), port=port, gate=gate, skip_ensure=True)

    def agent_status(self, name: Any) -> Dict[str, Any]:
        """What one agent's unit and gate are running, and with which keys and surface."""
        from .agent_service import AgentServerProcess

        return AgentServerProcess().status(str(name))


def dispatch_agent_action(runtime: Any, action: str, payload: Any) -> Dict[str, Any]:
    """Answer one agent verb. The payload rides unaltered to the root boundary (#178)."""
    if not isinstance(payload, dict):
        raise ValueError("Invalid cluster agent request.")
    if action == "agent_start":
        return runtime.agent_start(
            payload.get("name"), payload.get("port"),
            payload.get("config_content"), payload.get("gate"),
        )
    if action == "agent_rekey":
        return runtime.agent_rekey(
            payload.get("name"), payload.get("port"), payload.get("gate"),
        )
    if action == "agent_status":
        return runtime.agent_status(payload.get("name"))
    return runtime.agent_stop(payload.get("name"))
