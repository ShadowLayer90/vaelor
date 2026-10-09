"""Keep every deployed cluster agent running what it should, with the keys it should.

The agent-tier sibling of the LLM Server gate's convergence (DECISIONS VD-127,
the 2026-09-28 amendment), applied to ACC-070/071/074. Before this, a deployed
agent's gate was rendered once with the deploy's one key and never again: a key
rotated or minted in the console was refused, a revoked one kept working until
a controller reboot, and after that reboot the relaunch resolved the deploy's
key id - revoked by then - and left the agent down for good while the console
read Serving.

**What is running is read off the running agent** (``agent_status`` over the
root bridge): whether the unit and the gate run, the SET-hash of the keys the
gate was rendered with, and the ``surface_digest`` the runtime was started
with. **What should run is read from the owners**: the broker's CURRENT active
keys for the agent's endpoint (its fingerprint-only listing, no decrypt), and
the digest a relaunch would stamp now (``AgentPoolOperations.desired_digest``).
One pass then does the least that converges:

* the unit is down (a reboot cleared its tmpfs config, or it will not start),
  or its surface is stale (a tool, skill, instruction or backing change) - a
  full relaunch with the current key set;
* otherwise, the gate's key set is not the current one, or a gate runs with no
  active key, or none runs with some - a gate-only re-key; the runtime keeps
  serving;
* otherwise nothing.

The plaintext key set is read over the broker socket (``endpoint_keys``) only
when a relaunch or re-key is about to use it, so a converged pass decrypts
nothing. The loop runs on the executor every 30 s (``executor_service.
launch_agent_reconcile``): the first pass is the boot reconcile and every later
one the failure watch, so a crash mid-run is healed as well as a reboot.

**Accepted, visible fail-open** (the LLM Server's, restated): while the broker
cannot answer, nothing is re-keyed, so a key revoked in that window stays
admitted; the console shows ``applying-keys`` for as long as the running set
differs, and Remove stops the gate outright. A bridge too old to answer
``agent_status`` falls back to the pre-ACC-070 health-probe relaunch, now with
the current key set.
"""

from __future__ import annotations

import logging
import time
from typing import Any, Callable, Dict, Iterable, List, Mapping, Optional, Tuple

from .agent_pool_operations import AGENT_ENDPOINT_PREFIX
from .agent_pool_seams import AGENT_LIFECYCLE_LOCK, _split_endpoint
from .llm_server_state import binding_marker, row_fingerprints
from .job_vocabulary import CLUSTER_AGENT_DEPLOY_JOB
from .served_endpoint_keys import SERVED_ENDPOINT_PROVIDER

LOGGER = logging.getLogger(__name__)


def agent_endpoint_id(row: Mapping[str, Any]) -> str:
    """``agent:<row id>``, or ``""`` for a row with no id."""
    row_id = str(row.get("id") or "")
    return AGENT_ENDPOINT_PREFIX + row_id if row_id else ""


def endpoint_key_rows(listing: Any, endpoint_id: str) -> List[Mapping[str, Any]]:
    """The ACTIVE served-endpoint rows bound to ``endpoint_id`` in a broker listing."""
    return [
        row for row in (listing or [])
        if row.get("provider") == SERVED_ENDPOINT_PROVIDER
        and str(row.get("endpoint_id") or "") == endpoint_id
        and not row.get("revoked")
    ]


def desired_key_set(rows: Any) -> str:
    """The SET-hash the gate must carry for these rows; ``""`` means no gate.

    The same basis the gate's own marker is written with
    (:func:`vaelor.llm_server_state.applied_marker` over the plaintext keys), so
    a gate carrying exactly the broker's current keys compares equal.
    """
    return binding_marker(True, row_fingerprints(rows)).key_fingerprint


def broker_listing(broker: Any) -> Optional[List[Mapping[str, Any]]]:
    """The broker's fingerprint-only listing, or ``None`` when it cannot answer."""
    try:
        return list(broker.list() or [])
    except Exception:  # noqa: BLE001 - None is the "broker could not say" answer
        return None


def agent_status(bridge: Any, name: str) -> Optional[Dict[str, Any]]:
    """The bridge's reading of one agent, or ``None`` when it cannot be read."""
    try:
        status = bridge.agent_status(name)
    except Exception:  # noqa: BLE001 - None is the "bridge could not say" answer
        return None
    return dict(status) if isinstance(status, Mapping) else None


def gate_converged(status: Mapping[str, Any], rows: Any) -> bool:
    """Whether the running gate admits exactly the endpoint's active keys."""
    wanted = desired_key_set(rows)
    if not wanted:
        return not bool(status.get("gate_running"))
    return bool(status.get("gate_running")) and str(status.get("key_set") or "") == wanted


def _health_url(endpoint: str) -> str:
    """The inbound gate's ``/health`` URL derived from a stored ``.../v1`` endpoint."""
    text = str(endpoint or "").strip()
    if not text:
        return ""
    base = text[: -len("/v1")] if text.endswith("/v1") else text.rstrip("/")
    return base + "/health"


#: A row whose relaunch or re-key failed this many times in a row is retried
#: on a growing interval (S3), so a runtime that cannot start is not rebuilt
#: every 30 s forever; the interval doubles from the first to the longest.
BACKOFF_AFTER_FAILURES = 3
BACKOFF_FIRST_SECONDS = 60.0
BACKOFF_LONGEST_SECONDS = 1800.0

#: An active ``agent:*`` key with no deployment row, older than this and not
#: waiting on a queued deploy, is revoked by the pass (S2): a first key whose
#: deploy was refused, cancelled or lost opens nothing, but it must not stay a
#: live credential.
ORPHAN_KEY_AGE_SECONDS = 3600.0

#: Per-row (by row id) failure memory for the backoff and the change-of-reason
#: warning.
_FAILURES: Dict[str, Dict[str, Any]] = {}


def reconcile_cluster_agents(
    ops: Any, *, probe: Optional[Callable[[str], bool]] = None,
    clock: Callable[[], float] = time.time,
    pending_deployment_ids: Iterable[str] = (), sweep_orphans: bool = True,
) -> Dict[str, Any]:
    """One pass over every ``healthy`` agent row; idempotent and non-fatal per row.

    Held under :data:`~vaelor.agent_pool_seams.AGENT_LIFECYCLE_LOCK` for the
    whole pass, the lock a deploy and a remove hold too (B1), so a pass never
    relaunches an agent that a remove is tearing down. Returns ``{relaunched,
    rekeyed, healthy, surfaced, backing_off, skipped, orphan_keys_revoked}``. A
    row that cannot be healed is LEFT IN PLACE and surfaced with the reason
    (logged at WARNING when the reason changes), never crashed and never
    marked failed.
    """
    outcome: Dict[str, Any] = {
        "relaunched": [], "rekeyed": [], "healthy": [], "surfaced": [],
        "backing_off": [], "skipped": [], "orphan_keys_revoked": [],
    }
    with AGENT_LIFECYCLE_LOCK:
        listing = broker_listing(ops.broker)
        now = clock()
        for row in ops.store.list(state="healthy"):
            name = str(row.get("name", ""))
            # Keyed by the row id, so a redeploy under the same name starts clean.
            key = str(row.get("id") or name)
            memory = _FAILURES.get(key) or {}
            if now < float(memory.get("retry_at") or 0):
                outcome["backing_off"].append(name)
                continue
            try:
                action = _converge_row(ops, row, listing, probe or ops._probe)
            except Exception as error:  # noqa: BLE001 - surface an unhealable row, never crash
                outcome["surfaced"].append({"name": name, "reason": str(error)})
                _note_failure(key, name, str(error), now)
                continue
            _FAILURES.pop(key, None)
            outcome[action].append(name)
        if listing is not None and sweep_orphans:
            outcome["orphan_keys_revoked"] = sweep_orphan_keys(
                ops, listing, now=now, pending_deployment_ids=pending_deployment_ids,
            )
    return outcome


def pending_deploy_ids(job_store: Any) -> List[str]:
    """Deployment ids named by agent deploy jobs not yet finished (queued or running).

    Their first keys have no row yet and must not be swept as orphans. Asked of
    the job store by type and state, never through a newest-rows window, so an
    old queued deploy is always seen (N3). Raises what the job store raises;
    the caller then sweeps nothing this pass.
    """
    return [
        str((record.get("payload") or {}).get("deployment_id") or "")
        for record in job_store.unfinished_of_type(CLUSTER_AGENT_DEPLOY_JOB)
    ]


def _note_failure(key: str, name: str, reason: str, now: float) -> None:
    """Count a failed pass for row ``key``; back off after repeated failures."""
    memory = _FAILURES.setdefault(key, {"count": 0, "reason": "", "retry_at": 0.0})
    memory["count"] += 1
    if memory["reason"] != reason:
        LOGGER.warning("Cluster agent %s could not be brought back: %s", name, reason)
        memory["reason"] = reason
    extra = memory["count"] - BACKOFF_AFTER_FAILURES
    if extra >= 0:
        memory["retry_at"] = now + min(
            BACKOFF_FIRST_SECONDS * (2 ** extra), BACKOFF_LONGEST_SECONDS,
        )


def sweep_orphan_keys(
    ops: Any, listing: Any, *, now: float, pending_deployment_ids: Iterable[str] = (),
) -> List[str]:
    """Revoke active ``agent:*`` keys no deployment row owns (S2). Best-effort.

    A key is an orphan when no agent row has its deployment id, it is older
    than :data:`ORPHAN_KEY_AGE_SECONDS`, and no queued or running deploy job
    names that id. Returns the revoked credential ids (never a key).
    """
    try:
        owned = {str(row.get("id") or "") for row in ops.store.list()}
    except Exception:  # noqa: BLE001 - absence-ok: an unreadable store revokes nothing
        return []
    pending = {str(item) for item in pending_deployment_ids}
    revoked: List[str] = []
    for row in listing or []:
        endpoint_id = str(row.get("endpoint_id") or "")
        if (row.get("provider") != SERVED_ENDPOINT_PROVIDER or row.get("revoked")
                or not endpoint_id.startswith(AGENT_ENDPOINT_PREFIX)):
            continue
        deployment_id = endpoint_id[len(AGENT_ENDPOINT_PREFIX):]
        if deployment_id in owned or deployment_id in pending:
            continue
        if now - float(row.get("created_at") or 0) < ORPHAN_KEY_AGE_SECONDS:
            continue
        try:
            ops.keys.revoke(str(row.get("id")), endpoint_id)
        except Exception as error:  # noqa: BLE001 - retried on the next pass
            LOGGER.warning("Could not revoke an orphaned agent key: %s", error)
            continue
        revoked.append(str(row.get("id")))
    return revoked


def _converge_row(ops: Any, row: Mapping[str, Any], listing: Any, probe: Callable[[str], bool]) -> str:
    """Bring one row to what it should run; the name of what was done.

    The row is RE-READ and must still be ``healthy`` immediately before any
    relaunch or re-key (B1): a row that is being removed is left alone.
    """
    name = str(row.get("name", ""))
    endpoint_id = agent_endpoint_id(row)
    status = agent_status(ops.bridge, name)
    if status is None:
        # A bridge that cannot answer agent_status (an older bridge process):
        # the health probe decides, as before ACC-070, with the CURRENT keys.
        health_url = _health_url(str(row.get("endpoint") or ""))
        if health_url and probe(health_url):
            return "healthy"
        action = "relaunched"
    else:
        stale = False
        if listing is not None:
            stale = str(status.get("surface_digest") or "") != ops.desired_digest(row, listing)
        if not status.get("running") or stale:
            action = "relaunched"
        elif listing is not None and not gate_converged(
                status, endpoint_key_rows(listing, endpoint_id)):
            action = "rekeyed"
        else:
            return "healthy"
    current = ops.store.get(name)
    if not isinstance(current, Mapping) or str(current.get("state") or "") != "healthy":
        return "skipped"
    keys, read_at = _current_keys(ops, current, endpoint_id)
    if action == "relaunched":
        answer = ops.relaunch(current, gate_keys=keys, keys_read_at=read_at)
    else:
        answer = ops.rekey(current, keys, keys_read_at=read_at)
    if isinstance(answer, Mapping) and answer.get("gate_change") == "stale":
        _require_current_gate(ops, name, endpoint_id)
    return action


def _require_current_gate(ops: Any, name: str, endpoint_id: str) -> None:
    """After a "stale" answer, fail unless the running gate carries the current set.

    "Stale" is right when a newer set (the console's) is already running; it is
    wrong when the gate's recorded read time is merely ahead of the clock. A
    pass that did not converge is a failure - surfaced, warned and backed off
    (N2) - never reported as re-keyed while a revoked key stays admitted.
    """
    status = agent_status(ops.bridge, name)
    listing = broker_listing(ops.broker)
    if status is None or listing is None:
        return
    if not gate_converged(status, endpoint_key_rows(listing, endpoint_id)):
        raise RuntimeError(
            "The agent's gate kept an older key set than the current one; "
            "Vaelor retries it on the next pass."
        )


def _current_keys(ops: Any, row: Mapping[str, Any], endpoint_id: str) -> Tuple[List[str], float]:
    """The endpoint's active plaintext keys over the broker socket, and when read.

    Raises when the broker cannot answer: a relaunch must never go ahead with
    an EMPTY set it did not read (that would close a working door). The time
    is taken BEFORE the read, so the gate's newer-set check is conservative.
    """
    if not endpoint_id:
        raise ValueError("This agent row has no id, so its keys cannot be found.")
    read_at = time.time()
    return [str(key) for key in ops.broker.endpoint_keys(endpoint_id)], read_at


#: How long the console's in-request re-key may wait on the bridge (S5).
CONSOLE_REKEY_SECONDS = 10.0


def apply_agent_endpoint_keys(
    endpoint_id: str, *, store: Any, broker: Any, bridge: Any,
) -> Dict[str, Any]:
    """Re-key one agent's gate NOW, after a mint, rotate or revoke in the console.

    Called by the key routes on the control plane so a changed key reaches the
    running gate within the request, not only at the next 30 s pass. Answers
    ``{"apply": "applied"}`` when the bridge replaced the gate, ``{"apply":
    "pending"}`` when it could not (the reconcile then applies it on its next
    pass), and ``{}`` for an endpoint no deployed agent owns. Never raises: the
    broker already changed the key and a one-time plaintext must still reach
    the caller.
    """
    if not endpoint_id.startswith(AGENT_ENDPOINT_PREFIX):
        return {}
    row_id = endpoint_id[len(AGENT_ENDPOINT_PREFIX):]
    try:
        row = next(
            (item for item in store.list() if str(item.get("id") or "") == row_id), None
        )
    except Exception:  # noqa: BLE001 - the reconcile applies it instead
        return {"apply": "pending"}
    if row is None or str(row.get("state") or "") != "healthy":
        return {}
    try:
        if not getattr(bridge, "available", False):
            return {"apply": "pending"}
        read_at = time.time()
        keys = [str(key) for key in broker.endpoint_keys(endpoint_id)]
        advertise, port = _split_endpoint(str(row.get("endpoint") or ""))
        # S5: bounded well under a request's patience and never pulling an
        # image in the request; a slower bridge leaves it to the reconcile.
        bridge.agent_rekey(
            name=str(row.get("name", "")), port=port,
            gate={"listen_host": advertise, "listen_port": port, "api_keys": keys,
                  "keys_read_at": read_at, "pull": False},
            socket_timeout=CONSOLE_REKEY_SECONDS,
        )
    except Exception:  # noqa: BLE001 - the reconcile applies it instead
        return {"apply": "pending"}
    return {"apply": "applied"}
