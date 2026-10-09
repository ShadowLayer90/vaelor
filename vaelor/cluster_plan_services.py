"""Approval plans for the lifecycle of Vaelor-managed cluster services."""

from __future__ import annotations

from typing import Any, Dict

from .app_catalog import APP_TEMPLATES
from .application_deployments import ApplicationDeploymentError
from .app_port_claims import model_port_holders, refuse_model_port
from .cluster_app_manifest import ClusterAppManifestError, render_app_services
from .cluster_app_multideploy import (
    INVALID_APP_GROUP_MESSAGE,
    ClusterAppMultideployError,
    _refuse_id_collision,
    is_valid_app_group,
    missing_app_group_message,
)
from .cluster_app_placement import (
    PlacementError,
    ServiceFit,
    app_capacity_ledger,
    describe_placement,
    eligible_app_nodes,
    fit_app_services,
    reconcile,
    required_memory_bytes,
)
from .cluster_driver import ClusterDriverError
from .runtime_paths import data_path
from .cluster_plan_contract import (
    ClusterPlanContext,
    ClusterPlanError,
    managed_service_name,
)
from .cluster_service_reconfigure import (
    build_label_constraints,
    cpu_reservation_cores,
    reconcile_service_change,
    validate_cpu_limit,
)
from .cluster_service_state import (
    REMOVAL_WITHOUT_BACKUP,
    STATEFUL_REMOVE_ACK,
    app_group_members,
    live_service_names,
)

RESERVED_APP_PORTS = {34001, 34002}

SERVICE_UPDATE_STEPS = {
    "restart": [
        "Restart one task at a time without changing its image.",
        "Automatically roll back if Docker reports an update failure.",
        "Wait for every configured replica before reporting success.",
    ],
    "refresh": [
        "Resolve the current reviewed image tag to its available digest.",
        "Replace one task at a time with automatic failure rollback.",
        "Wait for every configured replica before reporting success.",
    ],
    "rollback": [
        "Ask Docker Swarm to restore the previous service specification.",
        "Replace one task at a time using the stored rollback policy.",
        "Wait for every configured replica before reporting success.",
    ],
}

SERVICE_UPDATE_IMPACTS = {
    "restart": (
        "A single-replica service may be briefly unavailable while "
        "its replacement starts."
    ),
    "refresh": (
        "A mutable image tag may contain application changes. "
        "Docker will retain the previous specification for rollback."
    ),
    "rollback": (
        "The service returns to Docker's immediately previous "
        "specification; node-local data is not changed."
    ),
}


def plan_deploy_app(
    action: str, body: Dict[str, Any], context: ClusterPlanContext
) -> Dict[str, Any]:
    template = APP_TEMPLATES.get(str(body.get("template_id", "")))
    if template is None:
        raise ClusterPlanError(
            "cluster_template_required",
            "Choose a reviewed application from the catalog.",
        )
    try:
        port = int(body.get("port", template["default_port"]))
    except (TypeError, ValueError):
        port = 0
    if (
        not 1024 <= port <= 65535
        or port in RESERVED_APP_PORTS
        or 8100 <= port <= 8199
    ):
        raise ClusterPlanError(
            "cluster_app_port", "Choose an available application port."
        )
    # The deploy's own refusal, previewed (review residual of a0edf29): Swarm
    # ingress binds the port on the controller, where a stored model claims it.
    try:
        refuse_model_port(port, model_port_holders(getattr(context.manager, "broker", None)))
    except ValueError as error:
        raise ClusterPlanError("cluster_app_port_model", str(error)) from error
    # The SAME placement reconcile the deploy runs, over live capacity, so the
    # preview reads back exactly where the app will land (VD-B3b-1). The env is
    # deliberately NOT generated here — the plan states it WILL be set and
    # carries no secret; the deploy job body mints it once (B2).
    node_id = str(body.get("node_id", "") or "").strip()
    intent = str(body.get("intent", "") or "").strip().lower()
    if not intent:
        intent = "pin" if node_id else "run-once"
    try:
        replicas = int(body.get("replicas", 1))
    except (TypeError, ValueError):
        replicas = 0
    manager = context.manager
    ledger_nodes = app_capacity_ledger(
        manager.store, manager.driver, manager.inventory_probe
    )
    required = required_memory_bytes(template["memory_mib"])
    eligible, excluded = eligible_app_nodes(ledger_nodes, required)
    try:
        decision = reconcile(
            intent, replicas, eligible,
            is_stateful=bool(template.get("volume")),
            node_id=node_id,
            excluded=excluded,
            required_bytes=required,
        )
    except PlacementError as error:
        raise ClusterPlanError(error.code, error.message) from error
    stateful = bool(template.get("volume"))
    env_step = None
    if template.get("secret_env"):
        env_step = (
            "Generate and set a unique password for this app and its reviewed "
            "environment (shown after it starts)."
        )
    elif template.get("env"):
        env_step = "Apply the app's reviewed environment settings."
    steps = [
        describe_placement(decision),
        f"Pull the reviewed image {template['image']} on the chosen machine.",
        (
            f"Reserve {template['memory']} per replica and preserve at least "
            "1 GB for each machine's operating system."
        ),
        (
            f"Publish port {port} through Swarm ingress and wait for "
            f"{decision.replicas} running replica"
            f"{'' if decision.replicas == 1 else 's'}."
        ),
    ]
    if env_step:
        steps.append(env_step)
    steps.append(
        "Keep persistent data on the machine it runs on."
        if stateful
        else "Run without persistent application storage."
    )
    if stateful and decision.replicas > 1:
        # W4d-D19: a pin lands every copy on one machine, and they share one
        # data volume there. Most data apps keep one database file and allow
        # one writer; say so before approval.
        steps.append(
            f"All {decision.replicas} copies share one data volume on this "
            "machine. An app that keeps a single database file, as most do, "
            "can corrupt it with more than one writer; use 1 replica unless "
            "this app supports several."
        )
    return {
        "title": f"Deploy {template['name']} — {describe_placement(decision)}",
        "steps": steps,
        "impact": (
            (
                "Persistent data is machine-local; drain or removal requires a "
                "backup first. "
                if stateful
                else "The stateless service can be removed and recreated from "
                "its template. "
            )
            + "The app's environment is stored in its Swarm service spec, "
            "readable by the machine's docker administrators — the same "
            "exposure as a single-node install."
        ),
        "approval_required": True,
    }


def _describe_service_fit(fit: ServiceFit) -> str:
    """The plain-language placement line for one researched service in the plan.

    Reads the SAME `ServiceFit` the deploy threads, so the preview names the same
    machines, the same replica count, and the same memory the deploy will use
    (preview==deploy). When the memory came from the manifest's 512 MB default —
    research could not determine a footprint — it says so plainly rather than
    presenting a guessed figure as a measured one.
    """
    memory = "{} MiB".format(fit.memory_mib)
    if fit.memory_defaulted:
        memory += " (the 512 MB default — set a limit if it needs more or less)"
    machines = ", ".join(fit.node_names) or "an eligible machine"
    if fit.stateful or fit.intent == "pin":
        where = "Pin {} to {}, reserving {}.".format(
            fit.service_key, machines, memory
        )
    elif fit.intent == "spread":
        where = (
            "Spread {} across {} machine{} ({}), one replica each, reserving {} "
            "per replica.".format(
                fit.service_key, fit.replicas,
                "" if fit.replicas == 1 else "s", machines, memory,
            )
        )
    else:
        where = "Run {} once on {}, reserving {}.".format(
            fit.service_key, machines, memory
        )
    return where


def plan_deploy_researched_app(
    action: str, body: Dict[str, Any], context: ClusterPlanContext
) -> Dict[str, Any]:
    """Preview deploying an APPROVED researched app across the cluster (D4b).

    Resolves the approved draft the SAME way the deploy does
    (`resolve_import` — state must be approved, digests intact), renders it with
    the SAME pure `render_app_services`, and runs the SAME `fit_app_services`
    against the live reservation-aware ledger, so this preview reads back exactly
    the per-service placement, replica counts and memory the deploy will use — and
    refuses for exactly the reason the deploy would (an unfittable service, a
    published port already live on the fleet). No secret is resolved and no side
    effect runs; the deploy job body mints the env-files and creates the services.
    """
    store = context.callbacks.get("application_deployment_store")
    if store is None:
        raise ClusterPlanError(
            "cluster_app_draft_unavailable",
            "The approved application draft store is unavailable, so this "
            "deployment cannot be previewed.",
        )
    draft_id = str(body.get("draft_id", "")).strip()
    manifest_digest = str(body.get("manifest_digest", "")).strip()
    try:
        draft = store.resolve_import(draft_id, manifest_digest, context.actor)
    except ApplicationDeploymentError as error:
        raise ClusterPlanError("cluster_app_draft", str(error), status=404) from error
    placements = body.get("placements")
    placements = placements if isinstance(placements, dict) else {}
    operations = context.callbacks.get("cluster_operations")
    workloads_root = getattr(operations, "workloads_root", None) or data_path(
        "workloads"
    )
    try:
        rendered = render_app_services(
            draft.get("manifest") or {},
            draft.get("compose") or {},
            placements,
            workloads_root=workloads_root,
        )
    except ClusterAppManifestError as error:
        raise ClusterPlanError("cluster_app_render", str(error)) from error
    manager = context.manager
    try:
        plan = fit_app_services(
            rendered.services, manager.store, manager.driver,
            manager.inventory_probe,
        )
    except PlacementError as error:
        raise ClusterPlanError(error.code, error.message) from error

    # The SAME live-service-name collision check the deploy runs (M1): a rendered
    # service name already live would make the deploy refuse, so the preview must
    # refuse for the same reason rather than promising a deploy that then fails.
    # Reads live names via `driver.status()` only — no broker credential is
    # resolved here; that stays deploy-only, so the preview leases no token.
    try:
        _refuse_id_collision(manager, rendered)
    except ClusterAppMultideployError as error:
        raise ClusterPlanError("cluster_app_id_collision", str(error)) from error

    steps = [_describe_service_fit(fit) for fit in plan.services]
    steps.append(
        "Attach every service to the overlay network {} so they reach each "
        "other by their compose names.".format(rendered.network_name)
    )
    steps.append(
        "Create each service's environment on a per-service private file; "
        "wait for every service to become ready or roll the whole app back."
    )
    stateful = [fit for fit in plan.services if fit.stateful]
    impact = (
        "The app deploys as {} service{} on one overlay network. ".format(
            len(plan.services), "" if len(plan.services) == 1 else "s"
        )
        + (
            "Each stateful service keeps its data on the machine it is pinned "
            "to; draining or removing it requires a backup first. "
            if stateful
            else "No service keeps persistent data. "
        )
        + "Each service's environment is stored in its Swarm service spec, "
        "readable by that machine's docker administrators."
    )
    return {
        "title": "Deploy {} across the cluster as {} service{}".format(
            rendered.app_group, len(plan.services),
            "" if len(plan.services) == 1 else "s",
        ),
        "steps": steps,
        "impact": impact,
        "approval_required": True,
    }


def plan_remove_researched_app(
    action: str, body: Dict[str, Any], context: ClusterPlanContext
) -> Dict[str, Any]:
    """Preview removing a researched multi-service app as ONE unit (D4d).

    The remove operation (`cluster_app_multideploy.remove_researched_app`) finds
    the app by its ``vaelor.app-group`` label off the LIVE services, tears down
    every member and then the overlay network, and RETAINS node-local volumes.
    This preview reads the SAME live members through the same
    `app_group_members` helper and runs the SAME whole-group data-loss check the
    operation enforces (`_stateful_names_needing_backup` over every member), so
    the plan surfaces the typed-acknowledgement token when — and only when — the
    deploy will require it. It is read-only: it inspects services and the backup
    store and mutates nothing.
    """
    app_group = str(body.get("app_group", "")).strip()
    if not is_valid_app_group(app_group):
        raise ClusterPlanError("cluster_app_group", INVALID_APP_GROUP_MESSAGE)
    driver = context.manager.driver
    try:
        inspected = driver.inspect_services(live_service_names(driver.status()))
    except ClusterDriverError as error:
        raise ClusterPlanError("cluster_app_group", str(error)) from error
    members = app_group_members(inspected, app_group)
    if not members:
        raise ClusterPlanError(
            "cluster_app_group", missing_app_group_message(app_group), status=404,
        )
    operations = context.callbacks.get("cluster_operations")
    # The whole app-group's at-risk stateful members, read exactly as the
    # operation reads them (a named volume with no backup under its OWN name).
    # Best-effort in the preview: if the operations facade is unavailable the
    # operation still makes the final call and refuses without the token.
    at_risk = (
        operations._stateful_names_needing_backup(members) if operations else []
    )
    plural = "" if len(members) == 1 else "s"
    plan = {
        "title": "Remove {} ({} service{})".format(app_group, len(members), plural),
        "steps": [
            "Stop and remove every service of this application.",
            "Remove the application's overlay network once Swarm detaches it.",
            RETAINED_STEP,
        ],
        "impact": (
            "Every service of this application stops immediately after approval. "
            + RETAINED_IMPACT
        ),
        "approval_required": True,
    }
    if at_risk:
        # Stated plainly (items 2/3): at least one stateful member has no backup,
        # so removal is an irreversible data loss. The token the operation
        # requires rides on the plan so the confirm step collects it.
        plan["impact"] = (
            "At least one stateful service of this application keeps its data on "
            "the machine it runs on and has no recovery backup. "
            + REMOVAL_WITHOUT_BACKUP
            + BACKUP_OR_ACKNOWLEDGE
        )
        plan["data_loss_ack"] = STATEFUL_REMOVE_ACK
    return plan


#: W4d-D20: what a removal does with a data volume, said the same way by the
#: single-service and the whole-app plan (the removal keeps it; the console
#: lists it; the owner reuses or deletes it).
RETAINED_STEP = (
    "Keep any data volume on its machine and list it under Retained data, where "
    "it can be deleted."
)
BACKUP_OR_ACKNOWLEDGE = " Create a backup first, or acknowledge that to proceed."
RETAINED_IMPACT = (
    "A kept data volume comes back if you deploy the same app under the same "
    "name on that machine; until then it still uses that machine's disk."
)


def _stateful_without_backup(
    service_name: str, context: ClusterPlanContext
) -> bool:
    """True when the service keeps a named volume and has no recovery backup —
    the condition the item-2 data-loss gate refuses a remove/drain on. Read from
    the same facts the operation checks (its mounts, the backup store), so the
    plan states plainly what the deploy will enforce. Best-effort: a service that
    will not inspect, or a missing backup store, is treated as not at risk here
    and the operation makes the final call."""
    try:
        details = context.manager.driver.service_details(service_name)
    except (ClusterDriverError, ValueError):
        return False
    stateful = any(
        item.get("type") == "volume" for item in details.get("mounts", [])
    )
    if not stateful:
        return False
    store = context.callbacks.get("cluster_backups")
    try:
        return not store.list(service_name=service_name, limit=1)
    except (AttributeError, OSError, ValueError):
        return False


def plan_remove_service(
    action: str, body: Dict[str, Any], context: ClusterPlanContext
) -> Dict[str, Any]:
    service_name = managed_service_name(body, context)
    at_risk = _stateful_without_backup(service_name, context)
    plan = {
        "title": f"Remove {service_name}",
        "steps": [
            "Stop the service and remove its Swarm definition.",
            "Release its published port and scheduling reservation.",
            RETAINED_STEP,
        ],
        "impact": "The service will stop immediately after approval. " + RETAINED_IMPACT,
        "approval_required": True,
    }
    if at_risk:
        # Data loss stated plainly (item 2): a stateful app with no backup loses
        # its only copy of the data on removal. The typed acknowledgement the
        # operation requires rides on the plan so the confirm step can collect it.
        plan["impact"] = (
            "This app keeps its data on the machine it runs on and has no "
            "recovery backup. " + REMOVAL_WITHOUT_BACKUP
            + BACKUP_OR_ACKNOWLEDGE
        )
        plan["data_loss_ack"] = STATEFUL_REMOVE_ACK
    return plan


def plan_service_update(
    action: str, body: Dict[str, Any], context: ClusterPlanContext
) -> Dict[str, Any]:
    service_name = managed_service_name(body, context)
    verb = action.removesuffix("-service")
    titles = {
        "restart": f"Restart {service_name}",
        "refresh": f"Refresh {service_name} from its current image tag",
        "rollback": f"Roll back {service_name}",
    }
    return {
        "title": titles[verb],
        "steps": SERVICE_UPDATE_STEPS[verb],
        "impact": SERVICE_UPDATE_IMPACTS[verb],
        "approval_required": True,
    }


def plan_configure_service(
    action: str, body: Dict[str, Any], context: ClusterPlanContext
) -> Dict[str, Any]:
    service_name = managed_service_name(body, context)
    try:
        replicas = int(body.get("replicas", 0))
        memory_limit = int(body.get("memory_limit_mib", 0))
        memory_reservation = int(body.get("memory_reservation_mib", 0))
        parallelism = int(body.get("update_parallelism", 0))
    except (TypeError, ValueError) as error:
        raise ClusterPlanError(
            "cluster_service_configuration",
            "Replica, memory, and rolling-update settings must be numbers.",
        ) from error
    order = str(body.get("update_order", "")).strip().lower()
    if (
        not 1 <= replicas <= 32
        or not 16 <= memory_limit <= 131072
        or not 16 <= memory_reservation <= memory_limit
        or not 1 <= parallelism <= min(replicas, 8)
        or order not in {"start-first", "stop-first"}
    ):
        raise ClusterPlanError(
            "cluster_service_configuration",
            (
                # The floor matches the client and the driver exactly (D2
                # follow-up a): the preview must not refuse a 64 MiB app
                # (it-tools) the reconfigure would accept, or preview != deploy.
                "Choose 1–32 replicas, a 16 MiB–128 GiB memory "
                "limit, a reservation no larger than that limit, "
                "and a valid rolling-update policy."
            ),
        )
    order_label = (
        "Start replacements before stopping old tasks"
        if order == "start-first"
        else "Stop old tasks before starting replacements"
    )
    # D2: validate the CPU limit + label constraints and re-run placement
    # eligibility for the NEW replicas/reservation through the SAME shared
    # reconcile the apply runs, so this preview refuses for exactly the reason
    # the reconfigure would (preview==deploy). A refusal raises here.
    manager = context.manager
    try:
        cpu_limit = validate_cpu_limit(body.get("cpu_limit"))
        label_constraints = build_label_constraints(body.get("label_constraints"))
        decision = reconcile_service_change(
            service_name,
            replicas=replicas,
            memory_reservation_mib=memory_reservation,
            store=manager.store,
            driver=manager.driver,
            inventory_probe=manager.inventory_probe,
        )
    except PlacementError as error:
        raise ClusterPlanError(error.code, error.message) from error
    cpu_reservation = cpu_reservation_cores(cpu_limit)
    steps = [
        f"Confirm placement fits: {decision.summary}",
        f"Set the service to {replicas} replica"
        f"{'' if replicas == 1 else 's'}.",
        (
            f"Reserve {memory_reservation} MiB and enforce a "
            f"{memory_limit} MiB memory limit per replica."
        ),
        (
            f"Limit each replica to {cpu_limit} CPU core"
            f"{'' if cpu_limit == 1 else 's'} and reserve "
            f"{cpu_reservation} for scheduling."
        ),
        (
            f"Update {parallelism} task"
            f"{'' if parallelism == 1 else 's'} at a time. "
            f"{order_label}."
        ),
        "Roll back automatically if Docker reports an update failure.",
        "Wait for every configured replica before reporting success.",
    ]
    if label_constraints:
        new_constraints = [
            constraint for constraint in label_constraints
            if constraint not in decision.existing_constraints
        ]
        if new_constraints:
            steps.append(
                "Constrain placement to nodes matching "
                + ", ".join(new_constraints)
                + "."
            )
    return {
        "title": f"Change settings for {service_name}",
        "steps": steps,
        "impact": (
            "Increasing replicas or memory can consume worker capacity. "
            "Start-first updates temporarily need room for both old and "
            "replacement tasks."
        ),
        "approval_required": True,
    }


def plan_backup_service(
    action: str, body: Dict[str, Any], context: ClusterPlanContext
) -> Dict[str, Any]:
    service_name = str(body.get("service_name", "")).strip()
    if service_name not in context.live_service_names():
        raise ClusterPlanError(
            "cluster_service_not_found",
            "Choose a live managed cluster service.",
            status=404,
        )
    try:
        details = context.manager.driver.service_details(service_name)
    except (ClusterDriverError, ValueError) as error:
        raise ClusterPlanError(
            "cluster_service_unavailable", str(error)
        ) from error
    volumes = [
        item for item in details.get("mounts", [])
        if item.get("type") == "volume"
    ]
    if len(volumes) != 1:
        raise ClusterPlanError(
            "cluster_backup_volume",
            "Backup currently requires exactly one managed named volume.",
        )
    return {
        "title": f"Back up {service_name}",
        "steps": [
            "Re-check the pinned worker identity and service placement.",
            f"Archive named volume {volumes[0]['source']} without changing it.",
            "Copy the archive to the controller and verify its SHA-256 checksum.",
            "Keep the SSH credential and plaintext password out of backup metadata.",
        ],
        "impact": (
            "The service remains online. A busy database may require its "
            "application-native export for transactional consistency."
        ),
        "approval_required": True,
    }


def plan_restore_service(
    action: str, body: Dict[str, Any], context: ClusterPlanContext
) -> Dict[str, Any]:
    service_name = str(body.get("service_name", "")).strip()
    backup_id = str(body.get("backup_id", "")).strip()
    store = context.callbacks.get("cluster_backups")
    try:
        backup = store.get(backup_id, verify=True)
    except (AttributeError, OSError, ValueError) as error:
        raise ClusterPlanError(
            "cluster_backup_unavailable", str(error), status=404
        ) from error
    if backup.get("service_name") != service_name:
        raise ClusterPlanError(
            "cluster_backup_service_mismatch",
            "Choose a backup created for this service.",
        )
    return {
        "title": f"Restore {service_name}",
        "steps": [
            "Verify the selected archive checksum and current worker placement.",
            "Create and verify a new safety backup of the current volume.",
            "Stop every service replica before replacing the volume contents.",
            "Restore the selected archive and wait for every replica to recover.",
            "If the restore fails, put the safety backup back and scale the "
            "service back up either way; the job says what could not be done.",
        ],
        "impact": (
            "The service is unavailable during restore. Data written after "
            "the selected backup is replaced, but retained in the safety backup."
        ),
        "approval_required": True,
        "backup": {
            key: backup[key]
            for key in (
                "id", "service_name", "volume", "size_bytes",
                "sha256", "created_at",
            )
        },
    }
