"""The model deployment a cluster agent was deployed against: one owner.

A cluster agent names its backing deployment on its row
(``backing.model_deployment_name``), and the GPU idle watch keeps THAT
deployment warm for it (``gpu_idle_watch.pinned_warm``). The agent itself used
to call whichever cluster deployment currently held the
``cluster-inference`` lease, so with two deployments the one that was kept warm
and the one that answered could differ (ACC-075). Every question about an
agent's model is now answered here, from the recorded deployment:

* :func:`resolve_recorded_backing` - the ``{base_url, model, api_key}`` the
  agent calls, read off that deployment's OWN credential (by id, not by lease);
* :func:`backing_state` - whether that deployment is serving, starting,
  unloaded (and by whom), failed or gone, from its record, never from a lease;
* :func:`wake_backing` - the wake a deployed agent asks for when its model does
  not answer: an idle scale-to-zero unload enqueues the same warm load the
  inference gateway's wake enqueues; a manual unload does not (the owner unloaded
  it on purpose, and loads it from Cluster > Deployments).

Idle and manual unloads write the same record state. Which one it was is
answered by the one owner of that question,
:func:`vaelor.gpu_serving_target.deployment_unload_cause` (VD-136): idle only
while the mode record is active, names THIS deployment, and AI Chat's resolved
target is still the available cluster; a lease that cannot be read is
"unknown", never guessed either way. This module holds no copy of that rule.
"""

from __future__ import annotations

import hashlib
import json
from typing import Any, Callable, Dict, Iterable, Mapping, Optional

from .gpu_serving_target import (
    CLUSTER_INFERENCE_PURPOSE, UNLOAD_CAUSE_IDLE, UNLOAD_CAUSE_MANUAL,
    deployment_unload_cause,
)

#: The backing deployment's condition, as an agent sees it.
BACKING_SERVING = "serving"
BACKING_STARTING = "starting"
BACKING_UNLOADED_IDLE = "unloaded-idle"
BACKING_UNLOADED_MANUAL = "unloaded-manual"
#: Unloaded, and who unloaded it could not be read (VD-136: never guessed).
BACKING_UNLOADED_UNKNOWN = "unloaded-unknown"
BACKING_FAILED = "failed"
BACKING_MISSING = "missing"
BACKING_UNKNOWN = "unknown"

#: How long a waking client is asked to wait before retrying: a warm load of a
#: cached model is tens of seconds to a few minutes on the pair.
WAKE_RETRY_AFTER_SECONDS = 30

#: What a deployed agent is told when it asks for a wake, per backing state.
#: Each says what IS and what the owner can do.
WAKE_DETAILS = {
    BACKING_UNLOADED_IDLE: (
        "The agent's model was unloaded after sitting idle and is loading "
        "again now. Retry in about {} seconds."
    ),
    BACKING_STARTING: (
        "The agent's model is loading. Retry in about {} seconds."
    ),
    BACKING_UNLOADED_MANUAL: (
        "The agent's model {} was unloaded by hand, so the agent cannot answer "
        "until it is loaded again from Cluster > Deployments."
    ),
    BACKING_MISSING: (
        "The model deployment {} this agent was deployed against is no longer "
        "on the cluster. Remove the agent and deploy it against a serving model."
    ),
    BACKING_FAILED: (
        "The model deployment {} this agent was deployed against has failed. "
        "Fix or redeploy it from Cluster > Deployments."
    ),
    BACKING_UNLOADED_UNKNOWN: (
        "The agent's model {} is unloaded, and Vaelor could not tell whether a "
        "request wakes it, so it did not try. Load it from Cluster > Deployments."
    ),
    BACKING_UNKNOWN: (
        "Vaelor could not read the state of the model deployment {} behind "
        "this agent, so it did not try to wake it. Check it under Cluster > "
        "Deployments."
    ),
}

_BACKING_GONE = (
    "The model deployment {} this agent was deployed against is not on the "
    "cluster, so the agent has no model to answer with."
)
_BACKING_NO_CREDENTIAL = (
    "The model deployment {} this agent was deployed against has no serving "
    "credential recorded, so the agent cannot reach it."
)


class BackingUnavailable(RuntimeError):
    """The recorded backing deployment cannot be resolved; the message says why."""


def backing_record(cluster_store: Any, name: str) -> Optional[Dict[str, Any]]:
    """The recorded pooled deployment ``name``, or ``None`` when there is none.

    Raises whatever the store raises: an unreadable store is not an absent
    deployment (LESSONS pattern 8), and each caller decides what that means.
    """
    record = cluster_store.get_pooled_deployment(str(name))
    return dict(record) if isinstance(record, Mapping) else None


def resolve_recorded_backing(
    broker: Any, cluster_store: Any, name: str,
) -> Dict[str, Any]:
    """The recorded deployment's ``{base_url, model, api_key}``, by its own credential.

    Not the ``cluster-inference`` lease: that lease follows whichever deployment
    is active, which is not necessarily the one this agent was deployed against
    and kept warm for (ACC-075). A credential that cannot be read raises the
    broker's error; a deployment that is gone, or has no credential, raises
    :class:`BackingUnavailable` naming it.
    """
    record = backing_record(cluster_store, name)
    if record is None:
        raise BackingUnavailable(_BACKING_GONE.format(name))
    credential_id = str(record.get("credential_id") or "")
    if not credential_id:
        raise BackingUnavailable(_BACKING_NO_CREDENTIAL.format(name))
    lease = dict(broker.resolve(credential_id, CLUSTER_INFERENCE_PURPOSE) or {})
    lease["credential_id"] = credential_id
    return lease


def _credential_row(rows: Iterable[Mapping[str, Any]], credential_id: str) -> Mapping[str, Any]:
    for row in rows or []:
        if str(row.get("id") or "") == credential_id:
            return row
    return {}


def _unload_cause(broker: Any, mode_store: Any, name: str) -> str:
    """The foundation's answer for one unloaded deployment, mapped to ours.

    ``deployment_unload_cause`` says ``unloaded-idle``, ``unloaded-manual`` or
    ``""`` (unknown); a mode record that cannot be read is unknown too. The
    deployment IS unloaded either way; only its cause is unknown.
    """
    try:
        mode_state = mode_store.read()
    except Exception:  # noqa: BLE001 - an unreadable mode record: cause unknown
        return BACKING_UNLOADED_UNKNOWN
    cause = deployment_unload_cause(broker, mode_state, name)
    if cause == UNLOAD_CAUSE_IDLE:
        return BACKING_UNLOADED_IDLE
    if cause == UNLOAD_CAUSE_MANUAL:
        return BACKING_UNLOADED_MANUAL
    return BACKING_UNLOADED_UNKNOWN


def default_mode_store() -> Any:
    """The appliance's cluster mode record, read-only here."""
    from .gpu_cluster_mode_state import ClusterModeStore

    return ClusterModeStore()


def backing_state(cluster_store: Any, name: str, *, broker: Any, mode_store: Any) -> str:
    """The recorded deployment's condition. Unreadable is :data:`BACKING_UNKNOWN`.

    An ``unloaded`` record's cause comes from
    :func:`vaelor.gpu_serving_target.deployment_unload_cause` over ``broker``
    and the mode record ``mode_store`` reads.
    """
    try:
        record = backing_record(cluster_store, name)
    except Exception:  # noqa: BLE001 - absence-ok: reported as unknown, not as gone
        return BACKING_UNKNOWN
    if record is None:
        return BACKING_MISSING
    state = str(record.get("state") or "")
    if state == "healthy":
        return BACKING_SERVING
    if state == "deploying":
        return BACKING_STARTING
    if state == "unloaded":
        return _unload_cause(broker, mode_store, name)
    if state == "failed":
        return BACKING_FAILED
    return BACKING_UNKNOWN


def backing_identity(
    cluster_store: Any, name: str, credential_rows: Iterable[Mapping[str, Any]],
) -> Dict[str, Any]:
    """What the agent's model connection depends on, with no secret in it.

    The deployment name, its credential id, that credential's version (a
    re-keyed cluster credential bumps it) and the recorded endpoint - the parts
    of the backing whose change means the running agent holds a stale
    connection. Folded into the agent's surface digest.
    """
    record = backing_record(cluster_store, name) or {}
    credential_id = str(record.get("credential_id") or "")
    row = _credential_row(credential_rows, credential_id)
    return {
        "deployment": str(name),
        "credential_id": credential_id,
        "credential_version": row.get("version"),
        "endpoint": str(record.get("endpoint") or ""),
    }


def surface_digest(parts: Mapping[str, Any]) -> str:
    """A short, secret-free digest of what a running agent was started with."""
    canonical = json.dumps(parts, sort_keys=True, separators=(",", ":"), default=str)
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()[:16]


def wake_backing(
    *, cluster_store: Any, broker: Any, name: str,
    enqueue_load: Callable[[str], Any], mode_store: Any = None,
) -> Dict[str, Any]:
    """Answer a deployed agent's wake request for its recorded deployment.

    An idle scale-to-zero unload enqueues one warm load (the same
    ``cluster.gpu.load`` the inference gateway's wake enqueues, deduplicated
    by the job store) and answers ``unloaded-idle`` with a retry hint; a load
    already under way answers ``starting``; a manual unload, a failed or a
    missing deployment answer their own state and enqueue nothing. A serving
    deployment answers ``serving`` - the model is up and simply did not answer.
    """
    state = backing_state(
        cluster_store, name, broker=broker,
        mode_store=mode_store if mode_store is not None else default_mode_store(),
    )
    if state == BACKING_UNLOADED_IDLE:
        enqueue_load(str(name))
    detail = WAKE_DETAILS.get(state, "")
    if state in (BACKING_UNLOADED_IDLE, BACKING_STARTING):
        detail = detail.format(WAKE_RETRY_AFTER_SECONDS)
    elif detail:
        detail = detail.format(name)
    return {
        "state": state,
        "deployment": str(name),
        "retry_after": WAKE_RETRY_AFTER_SECONDS
        if state in (BACKING_UNLOADED_IDLE, BACKING_STARTING) else 0,
        "detail": detail,
    }
