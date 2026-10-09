"""Impure orchestration: an approved researched app -> N Swarm services, atomic.

This is the D4a-deploy body, the mutation half of the researched-app cluster
deploy whose pure renderer is `cluster_app_manifest.render_app_services`. It
mirrors how `cluster_app_deploy.deploy_catalog_app` is written — a function
taking the `ClusterOperations` instance and reaching back through it for the
draft store, the mutation driver, the credential broker, and the controller-
labeling helpers — rather than a class of its own.

What it owns that the single-service catalog deploy did not:

* **Approved-draft-only intake (Q8).** The draft is consumed ONLY through
  ``ApplicationDeploymentStore.resolve_import`` (state must be ``approved``, the
  manifest digest is compared constant-time, and both the manifest and compose
  are re-hashed against their stored digests). There is no ``template_id``
  shortcut and no raw-manifest field on the job; a browser- or model-supplied
  Compose can never reach a ``service create`` here. The renderer then re-runs
  ``compose_policy.validate_normalized`` before any argv is built.

* **Overlay network lifecycle.** One attachable overlay network per app
  (``vaelor-app-<app>-net``), created before the services and removed with them
  on any failure. Every service attaches with a network ALIAS equal to its
  manifest service key, so a sibling reaches it by the compose name the app
  config expects (e.g. ``db``).

* **Per-service secret env-files (B4 — new security-sensitive glue).** For each
  service, every ``${VAELOR_CREDENTIAL_*}`` placeholder is resolved through the
  broker to the leased token and written as a literal ``KEY=VALUE`` line to THAT
  service's own root-owned 0600 ``--env-file`` (`tempfile.mkstemp`), alongside
  its plain env. The token never rides an argv, a log line, or an error string,
  and every env-file is unlinked in a ``finally``. This is not a reuse: the
  single-node import maps ``${}`` into a child-process env
  (`application_executor.application_secret_environment`), and the catalog path
  writes a flat non-secret dict (`cluster_app_deploy._write_env_file`); neither
  resolves a per-service placeholder to a 0600 file.

* **Atomic rollback (Q5).** The whole app comes up or none of it does. On ANY
  failure — the network, a ``service create``, or a readiness wait — every
  service created so far is removed, the network is removed, every env-file is
  unlinked, and any controller node label this deploy added is stripped when
  nothing else still pins it. A partial app is impossible, preserving the
  synchronous/atomic property D3's honest-state work relies on.
"""

from __future__ import annotations

import os
import re
import tempfile
import time
from typing import Any, Dict, List, Optional

from . import cluster_retained_volumes
from .app_port_claims import model_port_holders, refuse_model_port
from .application_deployments import ApplicationDeploymentError
from .cluster_app_deploy import remove_controller_label_if_unused
from .cluster_app_manifest import (
    ClusterAppManifestError,
    RenderedApp,
    ServiceSpec,
    app_network_name_for_group,
    render_app_services,
)
from .cluster_app_placement import (
    AppFitPlan,
    MAX_SPREAD_REPLICAS,
    PlacementError,
    SPREAD,
    fit_app_services,
    pin_node_required_message,
)
from .cluster_placement import CONTROLLER_PLACEMENT_ID
from .cluster_service_reconfigure import cpu_reservation_cores
from .cluster_service_state import (
    REMOVAL_WITHOUT_BACKUP, STATEFUL_REMOVE_ACK,
    app_group_members,
    inspect_app_service,
    live_service_names,
    require_data_loss_ack,
    stored_update_flags,
)
from .credential_broker import CredentialError
from .credential_use import note_credential_use

#: The managed constraint that pins a service to a data node. Shared spelling
#: with `cluster_app_deploy`, so the label the controller path adds here matches
#: the one its removal helper looks for.
_NODE_PIN = "node.labels.vaelor.node_id=="

#: Swarm rejects a reservation larger than the limit, so the reservation floor
#: matches the catalog deploy's (16 MiB) rather than a higher one that would
#: emit reserve > limit for a sub-64-MiB service and fail where a smaller floor
#: succeeds.
_RESERVATION_FLOOR_MIB = 16


class ClusterAppMultideployError(ValueError):
    """A plain-language refusal to deploy a researched app to the cluster.

    Every raise carries an operator-facing sentence — an app id that collides
    with a live service, a stateful service with no chosen pin node, a draft
    that was never approved — and NEVER the value of a leased secret.
    """


def _resolve_draft(operations, payload: Dict[str, Any], actor: str) -> Dict[str, Any]:
    """The approved draft, through ``resolve_import`` and nothing else (Q8).

    ``resolve_import`` is the one gate that enforces state ``approved`` plus the
    constant-time digest and the manifest/compose integrity re-hash; a draft
    that fails any of those is refused with the store's own words. There is no
    other way in — no ``template_id``, no raw manifest — so a Compose that
    skipped validate/approve cannot be deployed.
    """
    store = getattr(operations, "application_deployments", None)
    if store is None:
        raise ClusterAppMultideployError(
            "Researched application deployment is unavailable on this node."
        )
    draft_id = str(payload.get("draft_id", ""))
    manifest_digest = str(payload.get("manifest_digest", ""))
    try:
        return store.resolve_import(draft_id, manifest_digest, actor)
    except ApplicationDeploymentError as error:
        raise ClusterAppMultideployError(str(error)) from error


def _refuse_id_collision(operations, rendered: RenderedApp) -> None:
    """Refuse when a rendered service name is already a LIVE service (impure B).

    The renderer mints deterministic ``vaelor-app-<app>[-<svc>]`` names but
    cannot know what is already running; a name equal to a live catalog or
    researched service would otherwise make ``service create`` fail (or, worse,
    read as the same app). Read the live set once and refuse by name, honestly.
    """
    live = set(live_service_names(operations.driver.status()))
    for spec in rendered.services:
        if spec.service_name in live:
            raise ClusterAppMultideployError(
                "A cluster service named {} is already running; remove it or "
                "choose a different application id before deploying.".format(
                    spec.service_name
                )
            )


def _fit_services(operations, rendered: RenderedApp) -> AppFitPlan:
    """Place every rendered service against ONE running ledger tally (D4b, B7).

    Runs `fit_app_services` — which fetches the reservation-aware ledger once and
    decrements a running tally as each service is placed, so two co-deployed
    services are never each told the same node fits when together they do not —
    then threads the fit's computed replica count back onto each spec's
    placement. A spread service's replica count is the eligible-machine count the
    fit found (the renderer emits none; `_replicas` reads it here), so the dead
    spread count is finally correct. A `PlacementError` (a service that will not
    fit, or a published port already live on the fleet) is surfaced as the deploy
    refusal BEFORE any Swarm object exists, with the fit's per-service arithmetic.
    """
    try:
        plan = fit_app_services(
            rendered.services,
            operations.store,
            operations.driver,
            getattr(operations, "inventory_probe", None),
        )
    except PlacementError as error:
        raise ClusterAppMultideployError(error.message) from error
    by_key = {fit.service_key: fit for fit in plan.services}
    for spec in rendered.services:
        fit = by_key.get(spec.service_key)
        if fit is not None and fit.intent == SPREAD:
            spec.placement["replicas"] = fit.replicas
    return plan


def _service_env_lines(
    spec: ServiceSpec, secret_references: Dict[str, Any], broker,
    leases: Optional[List[Any]] = None,
) -> List[str]:
    """The ``KEY=VALUE`` lines for one service's env-file: secrets then plain.

    Each secret key names a ``${VAELOR_CREDENTIAL_*}`` placeholder whose backing
    credential id is in ``secret_references``; it is leased through the broker to
    its literal token here, and the token appears only in the returned line (it
    is written to a 0600 file by the caller and never elsewhere). A broker
    failure is surfaced without the credential id or any token text.
    """
    lines: List[str] = []
    for key in spec.secret_env_vars:
        credential_id = secret_references.get(key)
        if not isinstance(credential_id, str) or not credential_id:
            raise ClusterAppMultideployError(
                "Service {} needs the managed credential for {}, which the "
                "approved draft does not carry.".format(spec.service_key, key)
            )
        try:
            lease = broker.resolve(credential_id, "application-deploy")
        except CredentialError as error:
            raise ClusterAppMultideployError(
                "A managed credential for {} could not be leased.".format(key)
            ) from error
        token = str((lease or {}).get("token", ""))
        if not token:
            raise ClusterAppMultideployError(
                "The managed credential for {} is empty.".format(key)
            )
        lines.append("{}={}".format(key, token))
        if leases is not None:
            leases.append(lease)
    for key, value in spec.plain_env.items():
        lines.append("{}={}".format(key, value))
    return lines


def _write_env_file(lines: List[str]) -> Optional[str]:
    """A root-owned 0600 file of ``KEY=VALUE`` lines, or ``None`` for no env.

    `tempfile.mkstemp` creates the file 0600 by owner; the explicit chmod states
    the guarantee the secret depends on. Mirrors
    `cluster_app_deploy._write_env_file`, kept local so this security-sensitive
    write is read in one place with its resolution.
    """
    if not lines:
        return None
    descriptor, path = tempfile.mkstemp(prefix="vaelor-app-env-", suffix=".env")
    try:
        os.chmod(path, 0o600)
        with os.fdopen(descriptor, "w", encoding="utf-8") as handle:
            for line in lines:
                handle.write(line + "\n")
    except Exception:
        os.unlink(path)
        raise
    return path


def _replicas(spec: ServiceSpec) -> int:
    """The replica count for one service from its placement intent.

    A pinned (stateful) or run-once service is a single replica; a spread
    service runs the requested count, one per machine (the ``--replicas-max-per-
    node 1`` flag rides separately). Per-service fit is D4b's; this only reads
    the count the placement already carries, bounded to Swarm's sane range.
    """
    if spec.placement.get("intent") == "spread":
        try:
            requested = int(spec.placement.get("replicas", 1))
        except (TypeError, ValueError):
            requested = 1
        return max(1, min(requested, MAX_SPREAD_REPLICAS))
    return 1


def _placement_arguments(spec: ServiceSpec) -> List[str]:
    """The ``--constraint``/``--replicas-max-per-node`` flags for one service.

    Stateful (pin) constrains the chosen data node — refused honestly when no
    pin node was chosen, since a pinned service cannot deploy without one.
    Spread caps one copy per machine and pins nothing. Any operator label
    constraints are appended as additional constraints.
    """
    arguments: List[str] = []
    intent = str(spec.placement.get("intent", ""))
    if spec.stateful or intent == "pin":
        pin_node = str(spec.placement.get("pin_node") or "").strip()
        if not pin_node:
            raise ClusterAppMultideployError(
                pin_node_required_message(spec.service_key)
            )
        arguments.extend(["--constraint", "{}{}".format(_NODE_PIN, pin_node)])
    elif intent == "spread":
        arguments.extend(["--replicas-max-per-node", "1"])
    for constraint in spec.placement.get("label_constraints", []) or []:
        arguments.extend(["--constraint", str(constraint)])
    return arguments


def _service_create_arguments(
    spec: ServiceSpec, network_name: str, env_file: Optional[str]
) -> List[str]:
    """The full ``docker service create`` argv for one rendered service.

    Built here (not in the driver, which is at its line ceiling) and handed to
    the driver's docker runner. Order: name, overlay attach with the service-key
    alias, replicas, memory limit + a reservation no larger than the limit,
    optional CPU limit + reservation, restart policy + the stored rolling-update
    defaults, every rendered label, ingress and host-mode publishings, named
    volumes namespaced by the service, the placement, the env-file, then the
    digest-pinned image last.
    """
    memory_mib = int(spec.memory_mib)
    reservation = min(
        memory_mib, max(_RESERVATION_FLOOR_MIB, int(memory_mib * 0.75))
    )
    arguments: List[str] = [
        "service", "create",
        # Detached: return once Swarm accepts the spec, never blocking on
        # convergence (a crash-looping service never converges and the CLI would
        # hang to the 600s driver timeout). The deploy's `wait_service` loop does
        # the honest, bounded readiness wait.
        "--detach",
        "--name", spec.service_name,
        "--network", "name={},alias={}".format(network_name, spec.service_key),
        "--replicas", str(_replicas(spec)),
        "--reserve-memory", "{}M".format(reservation),
        "--limit-memory", "{}M".format(memory_mib),
        "--restart-condition", "on-failure",
        "--restart-max-attempts", "5",
        *stored_update_flags(None),
    ]
    if spec.cpu:
        arguments.extend([
            "--limit-cpu", "{}".format(float(spec.cpu)),
            "--reserve-cpu", "{}".format(cpu_reservation_cores(float(spec.cpu))),
        ])
    for key, value in spec.labels.items():
        arguments.extend(["--label", "{}={}".format(key, value)])
    for port in spec.published_ports:
        arguments.extend(["--publish", _publish_value(port, "ingress")])
    for port in spec.host_ports:
        arguments.extend(["--publish", _publish_value(port, "host")])
    for volume in spec.volumes:
        source = "{}-{}".format(spec.service_name, volume.get("name", ""))
        mount = "type=volume,source={},target={}".format(
            source, volume.get("target", "")
        )
        if volume.get("mode") == "ro":
            mount += ",readonly"
        arguments.extend(["--mount", mount])
    arguments.extend(_placement_arguments(spec))
    if env_file:
        arguments.extend(["--env-file", str(env_file)])
    arguments.append(str(spec.image))
    return arguments


def _publish_value(port: Dict[str, Any], mode: str) -> str:
    """One ``--publish`` value for an ingress or host-mode port mapping."""
    protocol = str(port.get("protocol", "tcp")).strip().lower() or "tcp"
    target = port.get("target")
    published = port.get("published")
    parts: List[str] = []
    if published:
        parts.append("published={}".format(int(published)))
    elif mode == "host" and target:
        parts.append("published={}".format(int(target)))
    if target:
        parts.append("target={}".format(int(target)))
    parts.append("protocol={}".format(protocol))
    parts.append("mode={}".format(mode))
    return ",".join(parts)


def _label_controller_pins(operations, rendered: RenderedApp) -> bool:
    """Label the controller ``vaelor.node_id==controller`` when a service pins it.

    Only workers carry ``vaelor.node_id``; a service constrained to the
    controller would otherwise sit Pending forever. When any stateful service
    pins the controller this labels it on demand (the `cluster_app_deploy`
    precedent) and returns whether the label was added, so the rollback knows to
    try to remove it. `remove_controller_label_if_unused` strips it again unless
    another managed service still pins the controller.
    """
    pins_controller = any(
        str(spec.placement.get("pin_node") or "") == CONTROLLER_PLACEMENT_ID
        for spec in rendered.services
    )
    if not pins_controller:
        return False
    node = operations._placement_node(CONTROLLER_PLACEMENT_ID)
    swarm_node_id = str(node.get("labels", {}).get("swarm_node_id", ""))
    if not swarm_node_id:
        raise ClusterAppMultideployError(
            "The controller could not be labeled for application placement."
        )
    operations.driver.label_node(
        swarm_node_id, {"vaelor.node_id": CONTROLLER_PLACEMENT_ID}
    )
    return True


def _create_network(operations, network_name: str) -> None:
    """Create the attachable overlay network, reusing an orphan of the same name.

    A prior run that failed after the network but before its rollback finished
    can leave this app's network behind; the name is app-specific and, because
    the id-collision check has already proved no service of this app is live,
    reusing it is safe. Any other create failure is surfaced.
    """
    try:
        operations.driver._docker(
            "network", "create", "--driver", "overlay", "--attachable",
            network_name,
        )
    except Exception as error:
        if "already exists" in str(error).lower():
            return
        raise ClusterAppMultideployError(
            "The application overlay network could not be created: {}".format(
                str(error)[:200]
            )
        ) from error


#: How long the resilient teardown waits for Swarm to detach a service's
#: endpoints from the overlay before removing the network (LIVE SMOKE FINDING).
_NETWORK_TEARDOWN_TIMEOUT = 30.0


def _network_absent(error: Exception) -> bool:
    """True when a docker error means the network is already gone (a done state).

    Docker reports a missing overlay as "Error: No such network: <name>" for both
    ``network inspect`` and ``network rm``, so that one phrase is the whole test."""
    return "no such network" in str(error).lower()


def _network_endpoint_count(operations, network_name: str) -> Optional[int]:
    """Attached-endpoint count for the overlay, or ``None`` when it no longer exists.

    ``None`` is the already-removed done state, not an error. A network that is
    present but whose inspect will not parse is reported as ``0`` so the caller
    proceeds to the bounded ``network rm`` retry rather than looping on the read.
    """
    try:
        out = operations.driver._docker(
            "network", "inspect", network_name,
            "--format", "{{len .Containers}}",
        )
    except Exception as error:
        if _network_absent(error):
            return None
        return 0
    try:
        return int(str(out).strip())
    except (TypeError, ValueError):
        return 0


def resilient_network_teardown(
    operations,
    network_name: str,
    *,
    timeout: float = _NETWORK_TEARDOWN_TIMEOUT,
    sleep=time.sleep,
    clock=time.monotonic,
) -> Dict[str, Any]:
    """Remove an app overlay network only AFTER Swarm detaches its endpoints.

    LIVE SMOKE FINDING: Swarm detaches service endpoints from an overlay
    ASYNCHRONOUSLY, so a ``network rm`` issued straight after ``service rm`` fails
    with "network has active endpoints" or leaves the net lingering, and the next
    deploy of the same app then collides on ``network create``. So this polls the
    attached-endpoint count to zero (bounded) BEFORE removing, and retries the
    remove with a short backoff, tolerating a network that is already gone. Never
    raises — teardown is best-effort — returning what it did for the caller's log.
    """
    deadline = clock() + max(1.0, float(timeout))
    while True:
        count = _network_endpoint_count(operations, network_name)
        if count is None:
            return {
                "removed": True, "network": network_name, "already_gone": True,
            }
        if count <= 0 or clock() >= deadline:
            break
        sleep(min(2.0, max(0.0, deadline - clock())))
    delay = 0.5
    last = ""
    while True:
        try:
            operations.driver._docker("network", "rm", network_name)
            return {"removed": True, "network": network_name}
        except Exception as error:
            if _network_absent(error):
                return {
                    "removed": True, "network": network_name,
                    "already_gone": True,
                }
            last = str(error)[:200]
            if clock() >= deadline:
                return {
                    "removed": False, "network": network_name, "reason": last,
                }
            sleep(min(delay, max(0.0, deadline - clock())))
            delay = min(delay * 2, 4.0)


def _rollback(
    operations,
    *,
    network_name: str,
    created_services: List[str],
    labeled_controller: bool,
) -> None:
    """Remove every created service and the network — best effort, never raises.

    The atomic-failure path (Q5): a half-built app leaves nothing behind. Each
    service remove is guarded so one cleanup failure never masks the original
    error or stops the rest; the overlay comes down through the resilient
    teardown (D4c) — services first, then the network once Swarm has detached its
    endpoints — so a failed deploy never orphans a network for the next deploy of
    the same app to collide on. The controller label is stripped last, only when
    no other managed service still pins it.
    """
    for name in reversed(created_services):
        try:
            operations.driver._docker("service", "rm", name)
        except Exception:
            pass
    resilient_network_teardown(operations, network_name)
    if labeled_controller:
        try:
            remove_controller_label_if_unused(operations)
        except Exception:
            pass


def refuse_model_ports(operations, services) -> None:
    """Refuse a service publishing a port a stored model credential names.

    Swarm ingress binds a published port on every node, the controller
    included, and a host-mode port with no published value publishes its target
    (`_publish_value`). Review residual of a0edf29 (LESSONS 6, 7).
    """
    holders = model_port_holders(getattr(operations, "broker", None))
    if not holders:
        return
    for spec in services:
        for entry in list(spec.published_ports) + list(spec.host_ports):
            port = entry.get("published") or (
                entry.get("target") if entry in spec.host_ports else None)
            if port:
                refuse_model_port(int(port), holders)


def deploy_researched_app(
    operations, payload: Dict[str, Any], actor: str, progress=None
) -> Dict[str, Any]:
    """Deploy an approved researched app across the cluster as N services.

    ``operations`` is the `ClusterOperations` whose ``deploy_researched_app``
    checked the ``deploy-researched-app`` confirm token before delegating here.
    The draft is resolved through ``resolve_import`` only; the renderer re-runs
    the compose policy and refuses binds/duplicate ports; then the network, the
    per-service 0600 env-files, the N ``service create`` calls and the readiness
    waits run under one atomic try/finally, rolling the whole app back on any
    failure.
    """
    report = progress or (lambda _percent, _message: None)
    draft = _resolve_draft(operations, payload, actor)
    compose = draft.get("compose") or {}
    secret_references = compose.get("x-vaelor-secret-references", {})
    if not isinstance(secret_references, dict):
        raise ClusterAppMultideployError(
            "The approved application draft has invalid credential references."
        )
    placements = payload.get("placements")
    placements = placements if isinstance(placements, dict) else {}

    try:
        rendered = render_app_services(
            draft.get("manifest") or {},
            compose,
            placements,
            workloads_root=operations.workloads_root,
        )
    except ClusterAppManifestError as error:
        raise ClusterAppMultideployError(str(error)) from error

    # Per-service fit against a RUNNING intra-deploy ledger tally (D4b): refuse an
    # unfittable app — or a published port already live on the fleet — with honest
    # arithmetic BEFORE any side effect, and thread the computed spread replica
    # count onto each spec so `_replicas` deploys the right number.
    _fit_services(operations, rendered)
    _refuse_id_collision(operations, rendered)
    refuse_model_ports(operations, rendered.services)
    # Preflight every service's placement so a stateful service with no chosen
    # pin node is refused BEFORE any env-file, network, or service is created —
    # fail-closed rather than leaning on the rollback for a knowable-early miss.
    for spec in rendered.services:
        _placement_arguments(spec)

    overall_timeout = max(
        30, min(int(payload.get("startup_timeout_seconds", 180) or 180), 600)
    )
    env_files: List[Optional[str]] = []
    leases: List[Any] = []
    created_services: List[str] = []
    labeled_controller = False
    try:
        # Every env-file first, so a broker failure refuses the app before any
        # Swarm object exists (nothing to roll back yet).
        for spec in rendered.services:
            env_files.append(_write_env_file(
                _service_env_lines(spec, secret_references, operations.broker, leases)
            ))
        labeled_controller = _label_controller_pins(operations, rendered)
        report(20, "Creating the application overlay network")
        _create_network(operations, rendered.network_name)
        for index, spec in enumerate(rendered.services):
            report(
                30 + int(40 * index / max(1, len(rendered.services))),
                "Creating cluster service {}".format(spec.service_key),
            )
            operations.driver._docker(*_service_create_arguments(
                spec, rendered.network_name, env_files[index]
            ))
            created_services.append(spec.service_name)
        report(75, "Waiting for every application service to become ready")
        deadline = time.monotonic() + overall_timeout
        for index, spec in enumerate(rendered.services):
            remaining = int(deadline - time.monotonic())
            operations.driver.wait_service(
                spec.service_name,
                timeout=max(10, remaining),
                expected_replicas=_replicas(spec),
            )
    except Exception as error:
        _rollback(
            operations,
            network_name=rendered.network_name,
            created_services=created_services,
            labeled_controller=labeled_controller,
        )
        if isinstance(error, ClusterAppMultideployError):
            raise
        raise ClusterAppMultideployError(
            "The application could not be deployed and was rolled back: "
            "{}".format(str(error)[:300])
        ) from error
    finally:
        for env_file in env_files:
            if env_file:
                try:
                    os.unlink(env_file)
                except OSError:
                    pass
    # Every service is up: the secrets they were given were used (ACC-107).
    for lease in leases:
        note_credential_use(lease, broker=operations.broker)

    return {
        "deployed": True,
        "app_group": rendered.app_group,
        "network": rendered.network_name,
        "services": [
            {
                "name": spec.service_name,
                "service": spec.service_key,
                "stateful": spec.stateful,
                "replicas": _replicas(spec),
            }
            for spec in rendered.services
        ],
        "draft_id": draft.get("id", ""),
    }


#: An app-group label is the ``[a-z0-9-]`` slug the naming producer minted, ≤63
#: chars — the exact charset/length the widened ops regexes and Swarm admit.
_APP_GROUP = re.compile(r"[a-z0-9][a-z0-9-]{0,62}")

#: The two operator-facing refusals of the researched-app remove, given ONE home
#: here (the module that owns the teardown) so the remove OPERATION and its plan
#: PREVIEW (`cluster_plan_services.plan_remove_researched_app`) cannot drift —
#: the plan imports both rather than repeating the sentence (VD-090 item 3).
INVALID_APP_GROUP_MESSAGE = "Choose a Vaelor-managed application to remove."


def is_valid_app_group(app_group: str) -> bool:
    """True when ``app_group`` is a well-formed managed app-group id.

    The single check both the remove OPERATION and its plan PREVIEW use, so the
    pattern itself has one home here rather than a second compiled copy (or a
    second module-level binding of it) in the plan module."""
    return bool(_APP_GROUP.fullmatch(str(app_group)))


def missing_app_group_message(app_group: str) -> str:
    """The refusal shown when no live researched app matches ``app_group``."""
    return "No running researched application '{}' was found.".format(app_group)


def _volumes_left_behind(operations, inspected, members) -> list:
    """``(service, volumes, holders)`` for each member that keeps a volume."""
    wanted = set(members)
    entries = []
    for inspect in inspected or []:
        if not isinstance(inspect, dict):
            continue
        name, volumes, pinned = cluster_retained_volumes.inspect_volume_mounts(inspect)
        if name not in wanted or not volumes:
            continue
        details: Dict[str, Any] = {}
        if not pinned:
            try:
                details = operations.driver.service_details(name)
            except Exception:  # unread placement: the record says unknown
                details = {}
        entries.append(
            (name, volumes, cluster_retained_volumes.holders_of(operations, details, pinned))
        )
    return entries


def _member_volumes(inspected, members) -> List[Dict[str, str]]:
    """The node-local named volumes of the app's members, ``{service, volume}``.

    Read from the bulk inspect so the remove result can name — honestly — the
    data that is RETAINED (never deleted) for a restore-from-backup, service by
    service."""
    wanted = set(members)
    volumes: List[Dict[str, str]] = []
    for inspect in inspected or []:
        if not isinstance(inspect, dict):
            continue
        name = str((inspect.get("Spec", {}) or {}).get("Name", ""))
        if name not in wanted:
            continue
        task = (inspect.get("Spec", {}) or {}).get("TaskTemplate", {}) or {}
        container = task.get("ContainerSpec", {}) or {}
        for mount in container.get("Mounts", []) or []:
            if isinstance(mount, dict) and str(mount.get("Type", "")) == "volume":
                source = str(mount.get("Source", ""))
                if source:
                    volumes.append({"service": name, "volume": source})
    return volumes


def _group_data_loss_subject(at_risk, key_by_name) -> str:
    """A subject that names the app's at-risk data services for the typed-ack gate.

    Reads naturally before the shared ack message's "keeps its data on the
    machine it runs on…", and names WHICH services would lose data so the
    operator can back exactly those up first."""
    keys = sorted(str(key_by_name.get(name, name)) for name in at_risk)
    plural = "" if len(keys) == 1 else "s"
    return "This app (its {} data service{}: {})".format(
        len(keys), plural, ", ".join(keys)
    )


def remove_researched_app(
    operations, payload: Dict[str, Any]
) -> Dict[str, Any]:
    """Remove a researched multi-service app: every service, then its network.

    ``operations`` is the `ClusterOperations` whose ``remove_researched_app``
    checked the ``remove-researched-app`` confirm token before delegating here.
    The app is found by its ``vaelor.app-group`` label read off the LIVE services
    (never by parsing names), so a member the truncate-and-hash producer renamed
    is still torn down. The multi-service data-loss GATE runs first: every at-risk
    stateful member (a named volume with no backup under its OWN name) must be
    backed up, or the operator must type the exact acknowledgement, before ANY
    service is removed. Teardown order is services → overlay network (through the
    resilient teardown, after Swarm detaches endpoints); node-local volumes are
    RETAINED for a restore-from-backup, never deleted. Fail-closed: a service that
    will not remove leaves the network in place (safe — the next deploy reuses it)
    and is reported, rather than a half-removed app claiming success.
    """
    app_group = str(payload.get("app_group", "")).strip()
    if not is_valid_app_group(app_group):
        raise ClusterAppMultideployError(INVALID_APP_GROUP_MESSAGE)
    inspected = operations.driver.inspect_services(
        live_service_names(operations.driver.status())
    )
    members = app_group_members(inspected, app_group)
    if not members:
        raise ClusterAppMultideployError(missing_app_group_message(app_group))

    # GATE (items 2/3): the whole app-group's at-risk stateful members, each
    # checked for a backup under its OWN name by the same pure predicate the
    # single-service remove uses. The typed ack is required if ANY is at risk.
    at_risk = operations._stateful_names_needing_backup(members)
    if at_risk:
        key_by_name = {
            str((inspect.get("Spec", {}) or {}).get("Name", "")):
                inspect_app_service(inspect)
            for inspect in inspected if isinstance(inspect, dict)
        }
        require_data_loss_ack(
            expected_ack=STATEFUL_REMOVE_ACK,
            provided_ack=payload.get("data_loss_ack"),
            subject=_group_data_loss_subject(at_risk, key_by_name),
            consequence=REMOVAL_WITHOUT_BACKUP,
        )

    retained_volumes = _member_volumes(inspected, members)
    # W4d-D20 / F3: where each member's volumes are, read BEFORE removal - an
    # unpinned member's from its task rows, which go with the service.
    left_behind = _volumes_left_behind(operations, inspected, members)
    failed: List[str] = []
    removed: List[str] = []
    for name in members:
        try:
            operations.driver._docker("service", "rm", name)
            removed.append(name)
        except Exception:
            failed.append(name)
    if failed:
        # Fail-closed: leave the overlay in place (the next deploy reuses it) and
        # report honestly rather than tearing the network down over a half-removed
        # app or claiming a removal that did not fully happen.
        raise ClusterAppMultideployError(
            "Removed {} of {} services, but could not remove: {}. The "
            "application was left in place; retry the removal.".format(
                len(removed), len(members), ", ".join(sorted(failed))
            )
        )

    network = app_network_name_for_group(app_group)
    teardown = resilient_network_teardown(operations, network)
    # Strip any controller node label these services' placement added, unless
    # another managed service still pins the controller (B1).
    try:
        remove_controller_label_if_unused(operations)
    except Exception:
        pass

    # W4d-D20: list what the removal kept, per machine, so the console can show
    # it and the owner can reuse or delete it.
    recorded = cluster_retained_volumes.record_retained(operations, [
        entry for entry in left_behind if entry[0] in set(removed)
    ])
    return {
        "removed": True,
        "app_group": app_group,
        "network": network,
        "network_removed": bool(teardown.get("removed")),
        "services": removed,
        "retained_volumes": retained_volumes,
        "volumes_retained": bool(retained_volumes),
        **({"retained_record_error": recorded["retained_record_error"]}
           if "retained_record_error" in recorded else {}),
    }
