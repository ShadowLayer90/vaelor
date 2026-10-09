"""Honest backend state + one-line reason for a cluster APP service (D3).

Pure. Fed the already-fetched ``docker service inspect`` object and the
``docker service ls`` replica field, so the SAME derivation runs in a unit test
as in `cluster_manager.summary`. It exists because the Deployments view read an
app row's status from the replica string alone — a regex that called a service
which had silently rolled back, or a stateful app whose data node was gone,
"Running". Swarm's real answer is in ``UpdateStatus`` and the placement
constraints, which only ``service inspect`` carries; this turns that into the
same ``state`` + ``reason`` the pooled rows publish, so `ClusterDeployments`
renders an app row through the identical status tones instead of the regex.

The state module also houses the stateful data-loss GATE (item 2): a stateful
app has no failover — its data lives on the one machine it runs on — so removing
it, or draining that machine, without a recovery backup is an irreversible loss.
`require_data_loss_ack` refuses that unless the operator typed the exact
acknowledgement, mirroring the `cluster_job_confirmations` typed-string pattern.
Kept here because "is this service stateful" is the one fact both the state
derivation and the gate turn on, and stating it once is the point.
"""

from __future__ import annotations

import re
from typing import Any, Dict, Iterable

#: The six honest states an app row can carry, each mapped on the frontend to a
#: `deployment-row__status` tone/label (Running/Deploying/Updating/Rolled back/
#: Failed/Unavailable) — the same tones the pooled rows already use.
HEALTHY = "healthy"
DEPLOYING = "deploying"
UPDATING = "updating"
ROLLED_BACK = "rolled-back"
FAILED = "failed"
UNAVAILABLE = "unavailable"

#: The Swarm ``UpdateStatus.State`` values, grouped by what they honestly mean.
_ROLLED_BACK_STATES = ("rollback_started", "rollback_completed")

#: The typed acknowledgements a stateful data-loss action needs when no recovery
#: backup exists, mirroring the fixed strings in `cluster_job_confirmations`.
#: Removal keeps the volume (W4d-D20), so the acknowledgement is that the data
#: is left with no backup, not that it is destroyed.
STATEFUL_REMOVE_ACK = "remove-stateful-without-backup"
STATEFUL_DRAIN_ACK = "drain-stateful-and-lose-data"

#: The managed pin constraint Swarm holds for a run-once/pin service — the same
#: expression `cluster_service_reconfigure` reads, kept independent here because
#: this module is pure and imports nothing from the driver stack.
_NODE_PIN = re.compile(r"node\.labels\.vaelor\.node_id==(.+)")

_REPLICAS = re.compile(r"\s*(\d+)\s*/\s*(\d+)")


def _running_desired(replicas: Any, inspect: Dict[str, Any]) -> tuple[int, int]:
    """Running and desired replica counts from the ``ls`` string, desired
    falling back to the inspect's ``Replicated.Replicas`` when the string is
    absent (a service just created reads no ``N/M`` yet)."""
    match = _REPLICAS.match(str(replicas or ""))
    running = int(match.group(1)) if match else 0
    desired = int(match.group(2)) if match else 0
    if not desired:
        mode = (inspect.get("Spec", {}) or {}).get("Mode", {}) or {}
        replicated = mode.get("Replicated", {}) or {}
        try:
            desired = int(replicated.get("Replicas", 1) or 1)
        except (TypeError, ValueError):
            desired = 1
    return running, max(desired, 0)


def _container_mounts(inspect: Dict[str, Any]) -> list:
    task = (inspect.get("Spec", {}) or {}).get("TaskTemplate", {}) or {}
    container = task.get("ContainerSpec", {}) or {}
    mounts = container.get("Mounts", []) or []
    return mounts if isinstance(mounts, list) else []


def inspect_is_stateful(inspect: Dict[str, Any]) -> bool:
    """True when the service carries a named volume — it keeps data on the one
    machine it runs on, so it has no failover and cannot be spread (D1)."""
    return any(
        isinstance(mount, dict) and str(mount.get("Type", "")) == "volume"
        for mount in _container_mounts(inspect)
    )


def _pinned_node(inspect: Dict[str, Any]) -> str:
    task = (inspect.get("Spec", {}) or {}).get("TaskTemplate", {}) or {}
    placement = task.get("Placement", {}) or {}
    for constraint in placement.get("Constraints", []) or []:
        match = _NODE_PIN.fullmatch(str(constraint).strip())
        if match:
            return match.group(1)
    return ""


def _row(state: str, reason: str) -> Dict[str, str]:
    return {"state": state, "reason": reason}


def normalize_node_runtime(nodes):
    """Lowercase the ``status`` and ``availability`` of each ``docker node ls``
    row in place. Docker emits them capitalized ("Ready"/"Down",
    "Active"/"Drain"/"Pause"); lowercasing once at the driver boundary makes a
    raw ``== "drain"`` hold for every consumer — the return-to-service render
    gate did not, so a drained node was stuck. Ids and hostnames keep Docker's
    casing. Returns the same list, so the driver can wrap the parse in one call.
    """
    for node in nodes or []:
        if isinstance(node, dict):
            for field in ("status", "availability"):
                value = node.get(field)
                if isinstance(value, str):
                    node[field] = value.lower()
    return nodes


def unschedulable_node_ids(enrolled_nodes, runtime_nodes) -> set:
    """The vaelor node ids of enrolled machines Swarm cannot schedule onto now.

    A machine is unschedulable when its live ``docker node ls`` row reads
    ``Status: Down`` (unreachable) or an ``Availability`` of drain/pause. Pure
    over the two reads the summary already holds: the enrolled records (each
    carrying its vaelor ``id`` and the ``swarm_node_id`` label that ties it to a
    live row) and the ``docker node ls`` rows. A node that is Ready+Active, or
    whose live row is ABSENT (removed, or a read that did not return it), is not
    included — `derive_app_state` prefers the non-alarming ``deploying`` over a
    false data-loss assertion whenever it cannot PROVE the pinned node is gone.
    """
    by_swarm: Dict[str, Dict[str, Any]] = {}
    for item in runtime_nodes or []:
        if isinstance(item, dict):
            by_swarm[str(item.get("id", ""))] = item
    down = set()
    for node in enrolled_nodes or []:
        if not isinstance(node, dict):
            continue
        node_id = str(node.get("id", ""))
        swarm_id = str((node.get("labels", {}) or {}).get("swarm_node_id", ""))
        live = by_swarm.get(swarm_id)
        if not node_id or not swarm_id or not isinstance(live, dict):
            continue
        status = str(live.get("status", "")).strip().lower()
        availability = str(live.get("availability", "")).strip().lower()
        if status == "down" or availability in {"drain", "pause"}:
            down.add(node_id)
    return down


def derive_app_state(
    inspect: Dict[str, Any], replicas: Any, unschedulable_nodes=frozenset()
) -> Dict[str, str]:
    """The honest ``{state, reason}`` for one app service.

    Reads Swarm's ACTUAL update and placement facts, never the replica string
    alone. A rolling update that failed and was rolled back is ``rolled-back``
    (not the "Running" its N/N replicas would read); a paused-after-failure
    update short of its replicas is ``failed``; a stateful app pinned to a
    machine that is PROVABLY unschedulable (its id in ``unschedulable_nodes``)
    is ``unavailable`` with the honest restore-from-backup reason. A stateful
    app short of its replicas on a HEALTHY pinned node is merely ``deploying``
    (its container is restarting) — under-replication ALONE is never read as a
    lost data node, because a restore is itself a data-loss risk.
    """
    if not isinstance(inspect, dict):
        return _row(UNAVAILABLE, "Vaelor could not read this service's state.")
    running, desired = _running_desired(replicas, inspect)
    update = inspect.get("UpdateStatus") or {}
    update_state = str(update.get("State", "")).strip().lower()
    message = str(update.get("Message", "")).strip()

    if update_state in _ROLLED_BACK_STATES:
        return _row(
            ROLLED_BACK,
            message
            or "An update failed and Swarm rolled it back to the previous "
            "version; the running copies are the old specification.",
        )
    if update_state == "updating":
        return _row(UPDATING, message or "A rolling update is in progress.")
    if update_state == "paused":
        if running < desired:
            return _row(
                FAILED,
                message
                or "An update failed and was paused before every replica came "
                "back; the service is short of its desired copies.",
            )
        return _row(UPDATING, message or "A rolling update is paused.")

    # No active update: the honest read is the replica count, split by whether a
    # stateful pin means a short count can never converge on its own.
    if desired and running >= desired:
        return _row(HEALTHY, "")
    pinned = _pinned_node(inspect)
    if inspect_is_stateful(inspect) and pinned and pinned in unschedulable_nodes:
        return _row(
            UNAVAILABLE,
            "Its data is on the machine it is pinned to, which cannot run it "
            "now. A lost data node is restored from a backup, not migrated — "
            "this app is not moved to another machine.",
        )
    return _row(
        DEPLOYING,
        "Waiting for {}/{} replicas to come up.".format(running, desired or 1),
    )


def annotate_service_states(
    services: Iterable[Dict[str, Any]],
    inspected: Iterable[Dict[str, Any]],
    unschedulable_nodes=frozenset(),
) -> list:
    """Attach ``state`` + ``reason`` to each Vaelor APP row from the bulk inspect.

    ``services`` are the ``docker service ls`` rows the status list already
    carries (name + replicas); ``inspected`` is the ONE bulk
    ``docker service inspect`` over them. ``unschedulable_nodes`` is the set of
    vaelor node ids Swarm cannot schedule onto now (see `unschedulable_node_ids`),
    so a stateful pin reads ``unavailable`` only on positive evidence its node is
    gone. Only ``vaelor-app-*`` rows are annotated — the model/pooled rows are a
    separate path the frontend renders from its own record. A row whose inspect
    is missing (removed between the two reads) is left without a state, and the
    frontend shows an honest unknown rather than a guessed one.
    """
    by_name: Dict[str, Dict[str, Any]] = {}
    for item in inspected or []:
        if isinstance(item, dict):
            name = str((item.get("Spec", {}) or {}).get("Name", ""))
            if name:
                by_name[name] = item
    annotated = []
    for service in services or []:
        row = dict(service) if isinstance(service, dict) else service
        name = str(row.get("name", "")) if isinstance(row, dict) else ""
        if name.startswith("vaelor-app-") and name in by_name:
            row.update(derive_app_state(
                by_name[name], row.get("replicas"), unschedulable_nodes
            ))
        annotated.append(row)
    return annotated


#: The state an app row folds to when a member's inspect is MISSING or unreadable
#: (B8): it ranks WORST, so an app whose members cannot all be read NEVER folds to
#: healthy. Frontend maps it to the same honest "unknown" tone a stateless row
#: without a derivation already shows.
UNKNOWN = "unknown"

#: Worst-to-best ordering the app-level aggregation folds over (B8). The design's
#: named order is unavailable > failed > rolled-back > updating > deploying >
#: healthy; UNKNOWN sits ABOVE unavailable so a member Vaelor could not read makes
#: the whole app read unknown rather than borrowing a healthier member's tone.
_STATE_RANK = {
    HEALTHY: 0,
    DEPLOYING: 1,
    UPDATING: 2,
    ROLLED_BACK: 3,
    FAILED: 4,
    UNAVAILABLE: 5,
    UNKNOWN: 6,
}


def _service_labels(inspect: Any) -> Dict[str, Any]:
    """The ``vaelor.*`` label map an app service carries in ``Spec.Labels``."""
    if not isinstance(inspect, dict):
        return {}
    labels = (inspect.get("Spec", {}) or {}).get("Labels", {}) or {}
    return labels if isinstance(labels, dict) else {}


def inspect_app_group(inspect: Any) -> str:
    """The ``vaelor.app-group`` label value of an inspect, or ``""`` when absent.

    The grouping key for the app-level aggregation and the app-level remove — read
    from the bulk inspect's ``Spec.Labels`` the summary already fetches, NEVER
    parsed out of the service name (a catalog ``vaelor-app-<name>`` carries no such
    label and so is never folded into a group)."""
    return str(_service_labels(inspect).get("vaelor.app-group", ""))


def inspect_app_service(inspect: Any) -> str:
    """The ``vaelor.app-service`` label — the manifest service key — of an inspect."""
    return str(_service_labels(inspect).get("vaelor.app-service", ""))


def app_group_members(inspected: Iterable[Dict[str, Any]], app_group: str) -> list:
    """The live service NAMES carrying ``vaelor.app-group == app_group``.

    Pure over the bulk inspect. The app-level remove reads its teardown set from
    this rather than parsing names, so a member renamed by the truncate-and-hash
    producer is still found by its label."""
    target = str(app_group)
    members = []
    for item in inspected or []:
        if not isinstance(item, dict):
            continue
        if inspect_app_group(item) == target and target:
            name = str((item.get("Spec", {}) or {}).get("Name", ""))
            if name:
                members.append(name)
    return members


def _member_state(
    inspect: Any, replicas: Any, unschedulable_nodes
) -> Dict[str, str]:
    """One app-group member's honest ``{state, reason}``.

    A member whose inspect is MISSING or unreadable — no ``Spec.TaskTemplate`` to
    derive from — is ``unknown`` (B8), never allowed to read as a healthy replica
    count off a stub. Everything readable goes through the same `derive_app_state`
    the per-service rows use, so a member's app-level and row-level state agree."""
    if not isinstance(inspect, dict) or not (
        inspect.get("Spec", {}) or {}
    ).get("TaskTemplate"):
        return _row(
            UNKNOWN, "Vaelor could not read this service's current state."
        )
    return derive_app_state(inspect, replicas, unschedulable_nodes)


def aggregate_app_groups(
    services: Iterable[Dict[str, Any]],
    inspected: Iterable[Dict[str, Any]],
    unschedulable_nodes=frozenset(),
) -> list:
    """Fold each researched app's per-service rows into ONE app-level summary (B8).

    PURE and ADDITIVE — this does not touch the per-service ``services`` list the
    frontend already renders; it produces a SEPARATE ``app_groups`` view keyed on
    the ``vaelor.app-group`` label read from the bulk inspect (never a name parse),
    so a catalog service with no such label is untouched and stays its own row, and
    a researched single-service app is simply a group of one that reads exactly
    like a normal row.

    For each group the app ``state`` is the WORST member state
    (unknown > unavailable > failed > rolled-back > updating > deploying > healthy)
    with a one-line ``reason`` naming the worst service, plus a per-service
    ``services`` breakdown. A member whose inspect is MISSING or unreadable makes
    the whole app ``unknown`` — an app NEVER reads healthy while a member is down
    or unreadable.
    """
    replicas_by_name: Dict[str, Any] = {}
    for service in services or []:
        if isinstance(service, dict):
            name = str(service.get("name", ""))
            if name:
                replicas_by_name[name] = service.get("replicas")

    groups: Dict[str, list] = {}
    order: list = []
    for inspect in inspected or []:
        group = inspect_app_group(inspect)
        if not group:
            continue
        if group not in groups:
            groups[group] = []
            order.append(group)
        groups[group].append(inspect)

    summaries = []
    for group in order:
        members = groups[group]
        breakdown = []
        worst_rank = -1
        worst = {"state": UNKNOWN, "reason": "", "service": ""}
        for inspect in members:
            name = str((inspect.get("Spec", {}) or {}).get("Name", ""))
            service_key = inspect_app_service(inspect) or name
            row = _member_state(
                inspect, replicas_by_name.get(name), unschedulable_nodes
            )
            entry = {
                "name": name,
                "service": service_key,
                "state": row["state"],
                "reason": row["reason"],
                # The live "running/desired" replica string for the member, read
                # from the same `docker service ls` rows the summary already
                # carries (never invented) so the D4d Manage breakdown can show
                # each service's replica count beside its state.
                "replicas": replicas_by_name.get(name),
            }
            breakdown.append(entry)
            rank = _STATE_RANK.get(row["state"], _STATE_RANK[UNKNOWN])
            if rank > worst_rank:
                worst_rank = rank
                worst = entry
        summaries.append({
            "app_group": group,
            "state": worst["state"],
            "reason": _group_reason(worst),
            "services": breakdown,
        })
    return summaries


def _group_reason(worst: Dict[str, str]) -> str:
    """The app row's one-line reason, naming the worst service.

    A healthy app carries no reason (like a healthy per-service row). Anything
    else leads with the worst service's manifest key so the operator knows WHICH
    part of a multi-service app is holding the whole app off healthy."""
    if worst["state"] == HEALTHY:
        return ""
    service = worst.get("service") or "a service"
    reason = worst.get("reason") or "Vaelor could not read its state."
    return "{}: {}".format(service, reason)


def details_is_stateful(details: Any) -> bool:
    """True when a ``service_details`` view carries a named volume — the app
    keeps data on the one machine it runs on, so a remove or drain of it is a
    data loss. Reads the NORMALIZED ``mounts`` shape the driver returns (lower-
    case ``type``), the counterpart of `inspect_is_stateful` over raw inspect."""
    return isinstance(details, dict) and any(
        isinstance(mount, dict) and mount.get("type") == "volume"
        for mount in details.get("mounts", []) or []
    )


def live_service_names(status: Any) -> list:
    """Every service name in a ``docker service ls`` status payload.

    The drain gate's candidate list — every managed app on the node it is asked
    about — starts from the whole live set, which `services_needing_backup` then
    narrows to the stateful ``vaelor-app-*`` rows with no backup. Pure over the
    already-fetched status object, so the gate glue in `cluster_operations` is a
    single call rather than an inline list comprehension against the driver.
    """
    services = status.get("services", []) if isinstance(status, dict) else []
    return [
        str(item.get("name", ""))
        for item in (services if isinstance(services, list) else [])
        if isinstance(item, dict)
    ]


def services_needing_backup(
    candidate_names, *, details_for, has_backup, pinned_to: str = ""
) -> list:
    """The stateful APP services among ``candidate_names`` that have no recovery
    backup — the services a remove or drain would lose data for (item 2 gate).

    Pure over injected I/O: ``details_for(name)`` returns that service's
    ``service_details`` (or ``None`` when it will not inspect) and
    ``has_backup(name)`` says whether a recovery archive exists, so the same
    decision runs in a unit test and in `cluster_operations` without this module
    importing the driver stack. ``pinned_to`` restricts the check to services
    pinned to that node — the drain gate asks only about the machine drained.
    """
    constraint = (
        f"node.labels.vaelor.node_id=={pinned_to}" if pinned_to else ""
    )
    at_risk = []
    for name in candidate_names:
        if not str(name).startswith("vaelor-app-"):
            continue
        details = details_for(name)
        if not details_is_stateful(details):
            continue
        constraints = [str(value) for value in (details.get("constraints") or [])]
        if constraint and constraint not in constraints:
            continue
        if not has_backup(name):
            at_risk.append(name)
    return at_risk


def stored_update_flags(details: Any) -> list:
    """The ``--update-*`` flags that RE-APPLY a service's stored update policy on
    a refresh/restart, instead of hardcoding one (D2 follow-up b).

    `refresh_service`/`restart_service` used to force ``--update-parallelism 1
    --update-failure-action rollback``, silently resetting an operator's
    configured rolling-update policy. This reads the policy back from
    ``service_details`` and re-applies it, COERCING a stored parallelism of 0 → 1
    (an unconfigured service reports 0, which Swarm reads as "all at once") and
    defaulting an unset failure action to ``rollback`` so the safety the hardcode
    provided is preserved for a service that never configured one.
    """
    policy = (details.get("update_policy", {}) if isinstance(details, dict) else {}) or {}
    try:
        parallelism = int(policy.get("parallelism", 0) or 0)
    except (TypeError, ValueError):
        parallelism = 0
    action = str(policy.get("failure_action", "")).strip().lower() or "rollback"
    flags = [
        "--update-parallelism", str(max(1, parallelism)),
        "--update-failure-action", action,
    ]
    order = str(policy.get("order", "")).strip().lower()
    if order in {"start-first", "stop-first"}:
        flags.extend(["--update-order", order])
    return flags


#: What a removal without a backup leaves (W4d-D20): `service rm` keeps the
#: node-local volume, and the console lists it as retained data.
REMOVAL_WITHOUT_BACKUP = (
    "Removing it keeps that data only in its volume on that machine, listed "
    "under Retained data, with no backup: deleting that volume, or losing the "
    "machine, loses the data."
)


def require_data_loss_ack(
    *, expected_ack: str, provided_ack: Any, subject: str,
    consequence: str = "This will irreversibly lose that data.",
) -> None:
    """Refuse a stateful data-loss action lacking both a backup and the typed ack.

    The GATE for item 2. A stateful service has no failover; removing it or
    draining the machine it is pinned to destroys its only copy of the data.
    When no recovery backup exists this proceeds ONLY if the operator typed
    ``expected_ack`` exactly — the same typed-string discipline the cluster job
    confirmations use, but a second acknowledgement specifically naming the
    irreversible loss, so the base confirm cannot stand in for it.
    """
    if str(provided_ack or "").strip() != expected_ack:
        raise ValueError(
            "{} keeps its data on the machine it runs on, and no recovery "
            "backup exists. {} Create a backup first, or type \"{}\" to "
            "confirm.".format(subject, consequence, expected_ack)
        )


def normalize_service_details(
    service: Dict[str, Any], tasks: list, name: str
) -> Dict[str, Any]:
    """Shape one ``docker service inspect`` object into the operator-safe view
    the Manage/reconfigure paths read — no environment secrets. Pure over the
    already-fetched inspect object + the ``docker service ps`` task rows, so the
    driver method stays a thin fetch and this shaping is unit-testable.
    """
    spec = service.get("Spec", {}) or {}
    task = spec.get("TaskTemplate", {}) or {}
    container = task.get("ContainerSpec", {}) or {}
    endpoint = spec.get("EndpointSpec", {}) or {}
    mode = spec.get("Mode", {}) or {}
    resources = task.get("Resources", {}) or {}
    update = spec.get("UpdateConfig", {}) or {}
    rollback = spec.get("RollbackConfig", {}) or {}
    labels = {
        str(key): str(value)
        for key, value in (spec.get("Labels", {}) or {}).items()
        if str(key).startswith("vaelor.")
    }
    mounts = []
    for mount in container.get("Mounts", []) or []:
        if isinstance(mount, dict):
            mounts.append({
                "type": str(mount.get("Type", "")),
                "source": str(mount.get("Source", ""))[:160],
                "target": str(mount.get("Target", ""))[:160],
                "read_only": bool(mount.get("ReadOnly")),
            })
    ports = []
    for port in endpoint.get("Ports", []) or []:
        if isinstance(port, dict):
            ports.append({
                "published": int(port.get("PublishedPort", 0) or 0),
                "target": int(port.get("TargetPort", 0) or 0),
                "protocol": str(port.get("Protocol", "tcp")),
                "mode": str(port.get("PublishMode", "ingress")),
            })
    return {
        "id": str(service.get("ID", "")),
        "name": str(spec.get("Name", name)),
        "image": str(container.get("Image", "")).split("@", 1)[0],
        "labels": labels,
        "constraints": [
            str(value)[:240]
            for value in (task.get("Placement", {}) or {}).get("Constraints", [])
        ],
        "mounts": mounts,
        "ports": ports,
        "desired_replicas": int(
            (mode.get("Replicated", {}) or {}).get("Replicas", 1) or 1
        ),
        "resources": {
            "limits": resources.get("Limits", {}),
            "reservations": resources.get("Reservations", {}),
        },
        "update_policy": {
            "parallelism": int(update.get("Parallelism", 0) or 0),
            "failure_action": str(update.get("FailureAction", "")),
            "order": str(update.get("Order", "")),
        },
        "rollback_policy": {
            "parallelism": int(rollback.get("Parallelism", 0) or 0),
            "failure_action": str(rollback.get("FailureAction", "")),
            "order": str(rollback.get("Order", "")),
        },
        "updated_at": str(service.get("UpdatedAt", "")),
        "tasks": tasks[:64],
    }
