"""What a deployed cluster agent is actually doing, for the console's card.

The agent card read its state and "1 active" key from the deployment row and a
placeholder, so after a revoke, a reboot or a failed deploy it still said
Serving with a key and a Rotate that failed (ACC-071/072). Each agent in
``GET /cluster/agents`` now carries:

* ``keys`` - the endpoint's ACTIVE keys from the broker's fingerprint-only
  listing (label, fingerprint, last four, created), never a placeholder, with
  ``keys_known: false`` when the broker could not answer;
* ``runtime`` - ``{state, reason, detail}``, one word of
  :data:`AGENT_RUNTIME_STATES` read from what is running: the agent's row, the
  RECORDED backing deployment (``agent_backing``), the bridge's ``agent_status``
  (unit, gate, the gate's key set and the runtime's surface digest), and the
  runtime's own ``/health`` on loopback, which now says whether its model
  answers. ``Serving`` is said only when every link does.

Every sentence says what IS and what the owner can do, and names the timing
Vaelor's reconcile keeps (about 30 seconds) rather than "immediately".
"""

from __future__ import annotations

import json
import urllib.request
from typing import Any, Callable, Dict, List, Mapping, Optional

from .agent_backing import (
    BACKING_FAILED, BACKING_MISSING, BACKING_STARTING, BACKING_UNKNOWN,
    BACKING_UNLOADED_IDLE, BACKING_UNLOADED_MANUAL, BACKING_UNLOADED_UNKNOWN, backing_state, default_mode_store,
)
from .agent_pool_seams import RUNTIME_LOOPBACK_HOST, _split_endpoint
from .agent_reconcile import (
    agent_endpoint_id, agent_status, desired_key_set, endpoint_key_rows,
)

#: Every word ``runtime.state`` may carry on an agent - the wire vocabulary the
#: console's card switches on (`frontend/src/lib/agentRuntimeStatus.ts` holds
#: the one copy, marked, and `tests/test_wire_vocabularies.py` pins the pair).
AGENT_RUNTIME_STATES = frozenset({
    "serving", "starting", "failed", "removing", "model-missing",
    "model-unloaded", "model-paused", "unknown", "not-running", "no-keys",
    "applying-keys", "applying-changes", "model-not-answering",
})

#: How long the loopback ``/health`` read may take; the runtime is on this
#: machine and answers from a cached model probe.
HEALTH_READ_SECONDS = 3.0

#: The sentence for each state, keyed ``"state/reason"`` where one state has
#: several causes, else by state.
RUNTIME_DETAILS = {
    "starting": (
        "The agent is being deployed. It serves once its runtime and gate "
        "answer."
    ),
    "starting/model-loading": (
        "The agent is running, and its model is loading. It answers once the "
        "model is serving."
    ),
    "failed": (
        "The deploy did not finish and was rolled back: its runtime was "
        "stopped and its keys revoked. The deploy job's result says why. "
        "Remove this agent and deploy it again."
    ),
    "removing": "The agent is being removed.",
    "model-missing": (
        "The model deployment this agent was deployed against is no longer on "
        "the cluster, so the agent has nothing to answer with. Remove the "
        "agent and deploy it against a serving model."
    ),
    "model-not-answering/model-failed": (
        "The model deployment this agent was deployed against has failed, so "
        "the agent cannot answer. Fix or redeploy the model from Cluster > "
        "Deployments."
    ),
    "model-unloaded": (
        "The agent's model was unloaded by hand, so the agent cannot answer. "
        "Load the model from Cluster > Deployments; a request to the agent "
        "does not load it."
    ),
    "model-unloaded/cause-unknown": (
        "The agent's model is unloaded, and Vaelor could not tell whether a "
        "request to the agent will load it again. Load the model from Cluster "
        "> Deployments."
    ),
    "model-paused": (
        "The agent's model was unloaded after sitting idle. The agent's next "
        "request loads it again: that request is answered with a retry hint, "
        "and requests succeed once the model is serving."
    ),
    "unknown": (
        "Vaelor could not ask the hardware bridge what this agent is running, "
        "so it is not known to be serving."
    ),
    "unknown/model-unknown": (
        "Vaelor could not read whether this agent's model is serving, so the "
        "agent is not known to be serving."
    ),
    "not-running/runtime-down": (
        "The agent's runtime is not running, so requests to it fail. Vaelor "
        "restarts it within about 30 seconds; if this stays, remove and "
        "redeploy the agent."
    ),
    "not-running/gate-down": (
        "The agent's LAN gate is not running, so clients get connection "
        "refused. Vaelor restarts it within about 30 seconds."
    ),
    "not-running/runtime-silent": (
        "The agent's runtime is up but did not answer its health check, so "
        "requests to it may fail."
    ),
    "no-keys": (
        "Every key for this agent has been revoked, so Vaelor keeps its LAN "
        "gate closed rather than open it without one. Issue a new key to open "
        "it."
    ),
    "applying-keys": (
        "A key was issued, rotated or revoked and the agent's gate still "
        "carries the older set: a new key may be refused and a revoked one "
        "still accepted until Vaelor re-keys the gate, within about 30 "
        "seconds."
    ),
    "applying-changes": (
        "This agent's tools, skills, instructions or model changed since it "
        "was started. Vaelor restarts it with the new set within about 30 "
        "seconds; until then it answers with the old set."
    ),
    "model-not-answering": (
        "The agent is running, but its model did not answer, so its requests "
        "fail. Check the model deployment under Cluster > Deployments."
    ),
}


#: A row that is not serving by its own lifecycle, and the state it reads as.
_ROW_LIFECYCLE = {"deploying": "starting", "failed": "failed", "removing": "removing"}


def _runtime(state: str, reason: str = "") -> Dict[str, str]:
    detail = RUNTIME_DETAILS.get("{}/{}".format(state, reason)) or RUNTIME_DETAILS.get(state, "")
    return {"state": state, "reason": reason, "detail": "" if state == "serving" else str(detail)}


def read_health(endpoint: str, *, timeout: float = HEALTH_READ_SECONDS) -> Optional[Mapping[str, Any]]:
    """The runtime's own ``/health`` body on loopback, or ``None`` if it did not answer.

    Read on the runtime's loopback port, not through the gate, so it answers
    whether or not a gate runs (a keyless agent runs none).
    """
    _host, port = _split_endpoint(endpoint)
    if not port:
        return None
    url = "http://{}:{}/health".format(RUNTIME_LOOPBACK_HOST, port)
    try:
        with urllib.request.urlopen(url, timeout=timeout) as response:  # noqa: S310 - loopback
            body = json.loads(response.read(64 * 1024).decode("utf-8"))
    except (OSError, ValueError):
        return None
    return body if isinstance(body, Mapping) else None


def model_answers(health: Optional[Mapping[str, Any]]) -> Optional[bool]:
    """Whether the runtime's health says its model answered; ``None`` if it does not say."""
    if health is None:
        return None
    model = health.get("model")
    if not isinstance(model, Mapping):
        return None
    return bool(model.get("reachable"))


def key_profiles(rows: List[Mapping[str, Any]]) -> List[Dict[str, Any]]:
    """The non-secret profile of each active key - never plaintext."""
    return [
        {
            "credential_id": row.get("id"),
            "label": row.get("label"),
            "key_fingerprint": row.get("fingerprint"),
            "last4": row.get("last4"),
            "created_at": row.get("created_at"),
            "last_used_at": row.get("last_used_at"),
        }
        for row in rows
    ]


def agent_runtime(
    row: Mapping[str, Any], *, backing: str, status: Optional[Mapping[str, Any]],
    key_rows: Optional[List[Mapping[str, Any]]], desired_digest: Optional[str],
    health: Optional[Mapping[str, Any]],
) -> Dict[str, str]:
    """The card's ``{state, reason, detail}`` from what was read. Pure.

    ``key_rows`` and ``desired_digest`` are ``None`` when their owner could not
    say, and then decide nothing; ``status`` is ``None`` when the bridge could
    not be asked. The order is the order of causes: a row that is not serving
    by its own lifecycle, then its model, then its runtime and gate, then what
    they carry.
    """
    row_state = str(row.get("state") or "")
    if row_state in _ROW_LIFECYCLE:
        return _runtime(_ROW_LIFECYCLE[row_state])
    if backing == BACKING_MISSING:
        return _runtime("model-missing")
    if backing == BACKING_UNLOADED_MANUAL:
        return _runtime("model-unloaded")
    if backing == BACKING_UNLOADED_UNKNOWN:
        return _runtime("model-unloaded", "cause-unknown")
    if backing == BACKING_UNLOADED_IDLE:
        return _runtime("model-paused")
    if status is None:
        return _runtime("unknown")
    if not status.get("running"):
        return _runtime("not-running", "runtime-down")
    if key_rows is not None and not key_rows:
        return _runtime("no-keys")
    if not status.get("gate_running"):
        return _runtime("not-running", "gate-down")
    if key_rows is not None and str(status.get("key_set") or "") != desired_key_set(key_rows):
        return _runtime("applying-keys")
    if desired_digest is not None and str(status.get("surface_digest") or "") != desired_digest:
        return _runtime("applying-changes")
    if backing == BACKING_STARTING:
        return _runtime("starting", "model-loading")
    if backing == BACKING_FAILED:
        return _runtime("model-not-answering", "model-failed")
    if health is None:
        return _runtime("not-running", "runtime-silent")
    answers = model_answers(health)
    if answers is False:
        return _runtime("model-not-answering")
    if answers is None and backing == BACKING_UNKNOWN:
        return _runtime("unknown", "model-unknown")
    return _runtime("serving")


def live_agent_view(
    ops: Any, row: Mapping[str, Any], listing: Optional[List[Mapping[str, Any]]],
    *, bridge: Any, health_reader: Callable[[str], Optional[Mapping[str, Any]]] = read_health,
    mode_store: Any = None,
) -> Dict[str, Any]:
    """``{keys, keys_known, runtime}`` for one agent row, read live.

    Every read is bounded and none raises: an unanswered broker makes the keys
    unknown, an unanswered bridge makes the runtime ``unknown``, and a digest
    that cannot be computed (a definition gone) decides nothing.
    """
    endpoint_id = agent_endpoint_id(row)
    key_rows = None if listing is None else endpoint_key_rows(listing, endpoint_id)
    name = str((row.get("backing") or {}).get("model_deployment_name") or "")
    backing = (
        backing_state(
            ops.cluster_store, name, broker=ops.broker,
            mode_store=mode_store if mode_store is not None else default_mode_store(),
        )
        if ops.cluster_store is not None else BACKING_UNKNOWN
    )
    status = None
    if bridge is not None and getattr(bridge, "available", False):
        status = agent_status(bridge, str(row.get("name", "")))
    digest = None
    if listing is not None and str(row.get("state") or "") == "healthy":
        try:
            digest = ops.desired_digest(row, listing)
        except Exception:  # noqa: BLE001 - absence-ok: a digest nobody can compute decides nothing
            digest = None
    health = health_reader(str(row.get("endpoint") or "")) if status and status.get("running") else None
    return {
        "keys": key_profiles(key_rows or []),
        "keys_known": key_rows is not None,
        "runtime": agent_runtime(
            row, backing=backing, status=status, key_rows=key_rows,
            desired_digest=digest, health=health,
        ),
    }
