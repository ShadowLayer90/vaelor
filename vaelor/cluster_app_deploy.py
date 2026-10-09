"""The catalog-app cluster deploy body: placement, env, and controller labeling.

The validation + placement + service-creation + readiness-wait + rollback body
of `cluster_operations.ClusterOperations.deploy_app`, housed here so that module
stays under the 1,000-line ceiling — the same extraction `cluster_llm_deploy`
made for the LLM path, and written the same way, as a function taking the
`ClusterOperations` instance rather than a class of its own, because it is a
continuation of one method reaching back through the operations facade for the
store, driver and node-resolution helpers.

Three things this body owns that the single-node deploy did not (D1):

* **Placement.** The intent + replica count are reconciled through the shared
  `cluster_app_placement.reconcile`, so the deploy lands exactly where the plan
  previewed (VD-B3b-1). Run-once picks the reservation-aware best fit; spread
  emits Swarm's one-per-machine cap; pin constrains to the chosen machine.
* **The controller as a placement target (B1).** Only workers carry a
  ``vaelor.node_id`` label; the controller's Swarm node has none, so a
  constraint naming it would match nothing and the task would hang Pending.
  When a decision constrains the controller, this body labels it on demand
  (precedent `cluster_llm_deploy`) and removes that label on rollback — and
  `remove_service` removes it on teardown — unless another managed service
  still pins the controller, so a co-tenant's constraint is never broken.
* **Env / secrets (B2/S1/S2).** A catalog app's install env — plain settings
  plus any generated password — is minted ONCE here (never the plan, which
  would mint a fresh secret every preview) and delivered through
  ``--env-file`` from a root-owned 0600 file, unlinked after ``service create``
  (VD-129: never ``--env`` on an argv /proc exposes, never a 0644 file). Swarm
  then preserves ``ContainerSpec.Env`` across every later update/scale/rollback,
  so the env is set once and never re-minted — no new store, no per-app record.
"""

from __future__ import annotations

import os
import re
import tempfile
from typing import Any, Callable, Dict, Optional

from .app_catalog import APP_TEMPLATES, build_install_env
from .app_port_claims import model_port_holders, refuse_model_port
from .cluster_app_placement import (
    PlacementDecision,
    app_capacity_ledger,
    describe_placement,
    eligible_app_nodes,
    reconcile,
    required_memory_bytes,
)
from .cluster_placement import CONTROLLER_PLACEMENT_ID
from .cluster_service_reconfigure import (
    build_label_constraints,
    cpu_reservation_cores,
    validate_cpu_limit,
)

#: The managed constraint that pins a service to the controller. Read back when
#: deciding whether removing a service should also strip the controller's
#: on-demand ``vaelor.node_id`` label (only when nothing else still pins it).
_CONTROLLER_CONSTRAINT = f"node.labels.vaelor.node_id=={CONTROLLER_PLACEMENT_ID}"


def _intent_and_replicas(payload: Dict[str, Any]) -> tuple[str, int, str]:
    """The requested intent, replica count and pin target, defaulted for compat.

    An older client that sends only ``node_id`` and no ``intent`` gets the pin
    behaviour it always had (its one node, one replica); a client that sends
    neither gets run-once. New clients send the intent explicitly.
    """
    node_id = str(payload.get("node_id", "") or "").strip()
    intent = str(payload.get("intent", "") or "").strip().lower()
    if not intent:
        intent = "pin" if node_id else "run-once"
    try:
        replicas = int(payload.get("replicas", 1))
    except (TypeError, ValueError):
        replicas = 0
    return intent, replicas, node_id


def _install_env(template_id: str, template: Dict[str, Any]) -> Dict[str, str]:
    """The app's install env: reviewed plain settings plus any fresh secret.

    Mirrors `app_catalog.render_compose`'s env handling exactly (plain ``env``
    merged under generated ``secret_env``) so the cluster service is configured
    the way the single-node compose delivers it. Minted here in the deploy body,
    never the plan (B2).
    """
    env = dict(template.get("env") or {})
    secrets = build_install_env(template_id)
    if secrets:
        env.update(secrets)
    return env


def _write_env_file(env: Dict[str, str]) -> Optional[str]:
    """A root-owned 0600 file of ``KEY=VALUE`` lines for ``docker --env-file``.

    `tempfile.mkstemp` creates the file 0600 by owner already (VD-129: the
    secret is on no argv and in no 0644 file); the caller unlinks it the moment
    ``service create`` has read it. ``None`` when the app declares no env, so no
    file is written for an app that needs none.
    """
    if not env:
        return None
    descriptor, path = tempfile.mkstemp(prefix="vaelor-app-env-", suffix=".env")
    try:
        # mkstemp already creates the file 0600 on POSIX; the explicit chmod
        # states the guarantee the secret depends on rather than relying on the
        # default, and is a harmless no-op where it already holds.
        os.chmod(path, 0o600)
        with os.fdopen(descriptor, "w", encoding="utf-8") as handle:
            for key, value in env.items():
                handle.write(f"{key}={value}\n")
    except Exception:
        os.unlink(path)
        raise
    return path


def _controller_pins(operations, exclude_service: str = "") -> int:
    """How many managed services still constrain the controller, bar one.

    Read before stripping the controller's on-demand ``vaelor.node_id`` label:
    the label is shared, so a co-tenant service (another app, or an LLM deploy
    that labelled the controller and left it) must keep it alive. Best-effort —
    a service that will not inspect is not counted, matching how the capacity
    ledger treats one.
    """
    pins = 0
    for service in operations.driver.status().get("services", []):
        name = str(service.get("name", ""))
        if name == exclude_service or not name.startswith("vaelor-"):
            continue
        try:
            details = operations.driver.service_details(name)
        except Exception:
            continue
        if _CONTROLLER_CONSTRAINT in [
            str(value) for value in details.get("constraints", [])
        ]:
            pins += 1
    return pins


def remove_controller_label_if_unused(
    operations, removed_service: str = ""
) -> None:
    """Strip the controller's on-demand node label when nothing else pins it.

    Called from the deploy's rollback and from `remove_service`, so a controller
    label this deploy added never lingers once no managed service still names it.
    """
    # The read-then-strip is a TOCTOU only if a deploy and a remove race; every
    # cluster mutation is serialized through the executor, so no two run at once.
    if _controller_pins(operations, exclude_service=removed_service):
        return
    try:
        node = operations._placement_node(CONTROLLER_PLACEMENT_ID)
    except ValueError:
        return
    swarm_node_id = str(node.get("labels", {}).get("swarm_node_id", ""))
    if not swarm_node_id:
        return
    try:
        operations.driver.remove_node_label(swarm_node_id, "vaelor.node_id")
    except Exception:
        # A label already gone is the state we wanted; never let cleanup raise
        # over the operation it is tidying up after.
        pass


def _label_controller_if_constrained(
    operations, decision: PlacementDecision
) -> bool:
    """Label the controller ``vaelor.node_id==controller`` when a constraint names it.

    Returns whether this deploy added the label, so the rollback knows to try to
    remove it. The controller is resolved through the placement-node resolver
    (not `_joined_node`, which raises for the controller), matching how the LLM
    deploy reaches it.
    """
    if CONTROLLER_PLACEMENT_ID not in decision.node_ids:
        return False
    node = operations._placement_node(CONTROLLER_PLACEMENT_ID)
    swarm_node_id = str(node.get("labels", {}).get("swarm_node_id", ""))
    if not swarm_node_id:
        raise ValueError("The controller could not be labeled for app placement.")
    operations.driver.label_node(
        swarm_node_id, {"vaelor.node_id": CONTROLLER_PLACEMENT_ID}
    )
    return True


def _validate_request(payload: Dict[str, Any]) -> tuple[Dict[str, Any], str, str, int]:
    """The template/name/port checks the single-node deploy already enforced."""
    template_id = str(payload.get("template_id", "")).strip()
    template = APP_TEMPLATES.get(template_id)
    if template is None:
        raise ValueError("Choose a reviewed application from the catalog.")
    name = str(payload.get("name", "")).strip()
    if not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9-]{0,39}", name):
        raise ValueError("Use a short letters-and-numbers deployment name.")
    try:
        port = int(payload.get("port", template["default_port"]))
    except (TypeError, ValueError) as error:
        raise ValueError("The published application port must be a number.") from error
    if (
        not 1024 <= port <= 65535
        or port in {34001, 34002}
        or 8100 <= port <= 8199
    ):
        raise ValueError("Choose an available app port from 1024 to 65535.")
    return template, template_id, name, port


def deploy_catalog_app(
    operations, payload: Dict[str, Any], progress: Optional[Callable] = None
) -> Dict[str, Any]:
    """Validate, place, configure and health-check a catalog app on the cluster.

    ``operations`` is the `ClusterOperations` whose ``deploy_app`` delegated here
    after checking the ``deploy-cluster-app`` gate.
    """
    template, template_id, name, port = _validate_request(payload)
    # Swarm ingress binds this port on the controller too: never one a stored
    # model credential names (review residual of a0edf29, LESSONS 6).
    refuse_model_port(port, model_port_holders(getattr(operations, "broker", None)))
    intent, replicas, node_id = _intent_and_replicas(payload)
    # D2: an operator CPU limit + label constraints, validated the SAME way the
    # plan previewed (`cluster_service_reconfigure`), so a bad value is refused
    # here exactly as the preview refused it. CPU never gates placement.
    cpu_limit = validate_cpu_limit(payload.get("cpu_limit"))
    label_constraints = build_label_constraints(payload.get("label_constraints"))

    ledger_nodes = app_capacity_ledger(
        operations.store, operations.driver, operations.inventory_probe
    )
    required = required_memory_bytes(template["memory_mib"])
    eligible, excluded = eligible_app_nodes(ledger_nodes, required)
    # A pinned worker that has been drained or paused is returned to service
    # first — the one guard the memory ledger does not carry.
    if intent == "pin":
        pinned = operations.store.get_node(node_id)
        if pinned is not None and pinned.get("state") in {"drain", "pause"}:
            raise ValueError(
                "Return this worker to service before deploying an app."
            )

    decision = reconcile(
        intent, replicas, eligible,
        is_stateful=bool(template.get("volume")),
        node_id=node_id,
        excluded=excluded,
        required_bytes=required,
    )

    report = progress or (lambda _percent, _message: None)
    env_file = _write_env_file(_install_env(template_id, template))
    labeled_controller = False
    result = None
    try:
        labeled_controller = _label_controller_if_constrained(
            operations, decision
        )
        report(25, "Creating the resource-bounded application service")
        result = operations.driver.deploy_app(
            name=name,
            image=template["image"],
            container_port=int(template["container_port"]),
            published_port=port,
            memory_limit_mib=int(template["memory_mib"]),
            template_id=template_id,
            constraints=decision.constraints,
            placement_flags=decision.flags,
            replicas=decision.replicas,
            env_file=env_file,
            placement_intent=decision.intent,
            requested_replicas=decision.replicas,
            volume=template.get("volume"),
            extra_ports=template.get("extra_ports"),
            cpu_limit=cpu_limit,
            cpu_reservation=cpu_reservation_cores(cpu_limit),
            label_constraints=label_constraints,
        )
        report(70, "Waiting for the application replicas to become ready")
        health = operations.driver.wait_service(
            result["name"],
            timeout=max(
                30, min(int(payload.get("startup_timeout_seconds", 180)), 600)
            ),
            expected_replicas=decision.replicas,
        )
    except Exception:
        if result:
            operations.driver.remove_service(result["name"])
        if labeled_controller:
            # Exclude the service we just removed by NAME rather than leaning on
            # `docker service rm` having landed first — the pin count must not
            # see this failed deploy's own service and keep the label alive.
            remove_controller_label_if_unused(
                operations, removed_service=result["name"] if result else ""
            )
        raise
    finally:
        if env_file:
            try:
                os.unlink(env_file)
            except OSError:
                pass
    return {
        **result,
        **health,
        "placement_intent": decision.intent,
        "replicas": decision.replicas,
        "placement": describe_placement(decision),
    }
