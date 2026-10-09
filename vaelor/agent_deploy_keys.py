"""The first key of a UI-deployed agent, minted where it can be shown once (GG14).

The deploy runs as a job on the executor, and the job ledger must never carry a
plaintext key, so the deploy's own mint was stripped from the result before
anyone saw it: no client could ever authenticate to an agent deployed from the
console. The key is now minted by the control plane in the request that queues
the deploy, exactly as the Endpoints panel's mint route does it, and returned
in THAT response for a single reveal. The job carries only the key's id and
the deployment id the key is bound under (``agent:<id>``); the executor
resolves the plaintext by id over the broker socket, as the reconcile does.

The same request stamps the trusted server state the deploy needs - the
controller's LAN advertise address and the acting administrator - which it
did before this module existed, inline in the jobs route.
"""

from __future__ import annotations

from typing import Any, Dict, Mapping, Optional

from .agent_deployments import new_deployment_id
from .agent_pool_operations import AGENT_ENDPOINT_PREFIX

#: The deploy request's answer when the first key could not be minted (S6):
#: plain, and never the broker's own error text.
FIRST_KEY_MINT_FAILED = (
    "Vaelor could not create the agent's first API key, so the deploy was not "
    "queued. Check that the credential service is running, then deploy again."
)

#: Why a failed agent deploy is not retried as it was (S7): its first key was
#: revoked when it failed (or its job was cancelled), and a retry would reuse
#: that key and the deployment id the failed row still holds.
AGENT_DEPLOY_RETRY_REFUSED = (
    "A failed agent deploy is not retried as it was: its first API key was "
    "revoked when it stopped. Remove the failed agent, then deploy it again to "
    "get a new key."
)

_AGENT_DEPLOY_JOB = "cluster.agent.deploy"


def stamp_agent_deploy(job_payload: Dict[str, Any], callbacks: Mapping[str, Any], actor: str) -> None:
    """Stamp the controller's advertise address and the actor, from server state.

    Both come from trusted server state, never from the request body.
    """
    operations = callbacks.get("cluster_operations")
    try:
        controller = operations.store.controller() if operations else {}
    except (AttributeError, OSError, ValueError):
        controller = {}
    job_payload["advertise_address"] = str((controller or {}).get("advertise_address", "") or "")
    job_payload["actor"] = actor


def mint_first_agent_key(job_payload: Dict[str, Any], broker: Any) -> Dict[str, Any]:
    """Choose the deployment id, mint its first key, and return the one-time reveal.

    The payload gains ``deployment_id`` and ``first_key_id`` - ids only. Raises the
    broker's error when it cannot mint: an agent deployed with a key nobody was
    shown is the defect this exists to end, so the deploy is refused instead.
    """
    deployment_id = new_deployment_id()
    endpoint_id = AGENT_ENDPOINT_PREFIX + deployment_id
    label = str(job_payload.get("name") or "").strip()[:80] or endpoint_id
    minted = broker.mint(endpoint_id, label)
    job_payload["deployment_id"] = deployment_id
    # "first_key_id", not "api_key_id": the job store refuses secret-shaped
    # field names, and this is only an id.
    job_payload["first_key_id"] = str(minted["credential_id"])
    return {
        "credential_id": str(minted["credential_id"]),
        "endpoint_id": endpoint_id,
        "label": str(minted.get("label") or label),
        "key_fingerprint": str(minted.get("key_fingerprint") or ""),
        "last4": str(minted.get("last4") or ""),
        "key": str(minted["key"]),
    }


def agent_deploy_retry_refused(job_store: Any, job_id: str) -> bool:
    """Whether ``job_id`` is an agent deploy, which is never retried as it was."""
    try:
        job = job_store.get(job_id)
    except Exception:  # noqa: BLE001 - absence-ok: the retry route answers not-found itself
        return False
    return bool(job) and job.get("type") == _AGENT_DEPLOY_JOB


def revoke_cancelled_deploy_key(broker: Any, job: Optional[Mapping[str, Any]]) -> None:
    """Revoke the first key of an agent deploy cancelled before it ran (S2).

    A deploy cancelled while queued never runs, so no rollback revokes the key
    the control plane minted for it. One cancelled while running is rolled
    back by the executor, which revokes it.
    """
    if not job or job.get("type") != _AGENT_DEPLOY_JOB or job.get("state") != "cancelled":
        return
    payload = job.get("payload") or {}
    key_id = str(payload.get("first_key_id") or "")
    deployment_id = str(payload.get("deployment_id") or "")
    if key_id and deployment_id:
        revoke_unqueued_key(broker, {
            "credential_id": key_id, "endpoint_id": AGENT_ENDPOINT_PREFIX + deployment_id,
        })


def revoke_unqueued_key(broker: Any, reveal: Optional[Mapping[str, Any]]) -> None:
    """Revoke a first key whose deploy could not be queued, best-effort."""
    if not reveal:
        return
    try:
        broker.revoke(str(reveal.get("credential_id")), str(reveal.get("endpoint_id")))
    except Exception:  # noqa: BLE001 - the key binds an endpoint no agent will ever own
        pass
