"""Re-check where a running app's replicas fit when an operator reconfigures it,
and validate the two argv extras D2 adds (CPU limit, label constraints).

D2 closes the boundary D1 left open. There is ONE apply site
(`cluster_operations.ClusterOperations.configure_service`) and ONE preview site
(`cluster_plan_services.plan_configure_service`) — scaling IS the configure
form's replicas field, so there is no separate scale action to route. The driver
issues ``service update --replicas/--limit-memory`` WITHOUT ``--constraint``, so
Swarm PRESERVES the managed pin and the spread's ``--replicas-max-per-node 1``:
the only defect is the missing eligibility RE-CHECK. Scaling a spread past its
eligible-machine count, or a pin past the room on its one machine, left tasks
Pending until ``wait_service`` timed out.

`reconcile_service_change` is that re-check, called by BOTH the preview and the
apply so the plan refuses for exactly the reason the reconfigure would
(preview==deploy). It reads the service's intent from the ``vaelor.placement-
intent`` label D1 wrote and the pinned node from the pin constraint, nets the
service's OWN reservation out of the ledger (B1), and re-runs the shared
eligibility over the NEW replica count and memory reservation.

The module also houses the CPU and label-constraint VALIDATIONS so the capped
`cluster_driver` and `cluster_operations` carry only thin argv/emission lines.
CPU is a container limit/reservation only — it never gates eligibility, because
the capacity ledger tracks memory, not CPU. Label constraints only ADD to the
managed pin/intent: a key in the reserved ``vaelor.*``/``pironman.*`` namespace
is refused so a user constraint can never hijack the managed placement labels.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Any, Callable, Dict, List, Optional

from .cluster_app_placement import (
    PIN,
    SPREAD,
    VALID_INTENTS,
    PlacementError,
    _gib,
    app_capacity_ledger,
    eligible_app_nodes,
    pin_capacity_bytes,
    required_memory_bytes,
)

#: The managed pin constraint Swarm holds for a run-once/pin service; the node it
#: names is where every replica lands, so a reconfigure sizes the change there.
_NODE_ID_CONSTRAINT = re.compile(r"node\.labels\.vaelor\.node_id==(.+)")

#: The default per-replica CPU limit, mirroring the single-node compose's literal
#: ``cpus: "2.0"`` (app_catalog.render_compose) so the cluster app is bounded the
#: way the standalone install is, and editable by the operator.
CPU_LIMIT_DEFAULT = 2.0
CPU_LIMIT_MIN = 0.25
CPU_LIMIT_MAX = 64.0
#: The scheduling reservation is a conservative fixed quarter-core, never more
#: than the limit itself. A CPU reservation is what Swarm schedules against, so a
#: small one keeps CPU from dominating placement — the ledger gates on memory,
#: and CPU stays a limit/reservation, not a new fit gate.
CPU_RESERVATION_CORES = 0.25

#: An operator label constraint is ``key op value`` with a bounded safe charset.
#: Keys in the reserved namespaces are refused (see `build_label_constraints`).
_LABEL_TOKEN = re.compile(r"[A-Za-z0-9._-]{1,63}")
_RESERVED_LABEL_PREFIXES = ("vaelor.", "pironman.")
LABEL_CONSTRAINT_OPS = ("==", "!=")
MAX_LABEL_CONSTRAINTS = 8


@dataclass(frozen=True)
class ReconcileDecision:
    """The reconfigure re-check outcome the preview reads back and the apply uses.

    ``existing_constraints`` are the constraints the service already carries, so
    the apply can add only the label constraints not already present (a
    ``service update --constraint-add`` of an existing constraint would be
    redundant) — D2 adds constraints; removing one is deferred with label-setting
    to phase G.
    """

    intent: str
    replicas: int
    node_id: str
    eligible_count: int
    summary: str
    existing_constraints: tuple = ()


def reconcile_service_change(
    service_name: str,
    *,
    replicas: int,
    memory_reservation_mib: int,
    store: Any,
    driver: Any,
    inventory_probe: Optional[Callable[[], Dict[str, Any]]],
) -> ReconcileDecision:
    """Re-run placement eligibility for a reconfigure, refusing with the arithmetic.

    Reads the running service's placement intent (the ``vaelor.placement-intent``
    label) and its pinned node (the pin constraint), then re-checks the NEW
    replica count against live eligibility with the service's OWN reservation
    netted out (B1). A spread scaled past the eligible-machine count is refused
    with the count arithmetic; a pin (or run-once, which also holds a node
    constraint) scaled past the room on its one machine is refused with the byte
    arithmetic. Raises `PlacementError`, which the preview maps to a plan error
    and the apply surfaces as a refusal — the same decision on both sides.
    """
    details = driver.service_details(service_name)
    labels = details.get("labels", {}) or {}
    existing_constraints = tuple(
        str(value) for value in (details.get("constraints", []) or [])
    )
    intent = str(labels.get("vaelor.placement-intent", "")).strip().lower()
    node_id = ""
    for constraint in existing_constraints:
        match = _NODE_ID_CONSTRAINT.fullmatch(constraint.strip())
        if match:
            node_id = match.group(1)
            break
    # An app deployed before D1 wrote the intent label: derive its shape from the
    # constraint Swarm actually holds. A node pin means every replica lands there
    # (run-once/pin); no pin means Swarm spreads them.
    if intent not in VALID_INTENTS:
        intent = PIN if node_id else SPREAD

    try:
        wanted = int(replicas)
    except (TypeError, ValueError):
        wanted = 0
    required = required_memory_bytes(memory_reservation_mib)
    ledger_nodes = app_capacity_ledger(
        store, driver, inventory_probe, exempt_service=service_name
    )
    eligible, excluded = eligible_app_nodes(ledger_nodes, required)
    count = len(eligible)

    if intent == SPREAD:
        if wanted > count:
            machines = "machine" if count == 1 else "machines"
            raise PlacementError(
                "cluster_app_reconfigure_capacity",
                f"Spreading {wanted} copies needs {wanted} eligible machines "
                f"but only {count} {machines} can host a replica now; reduce "
                "the replicas or add a machine.",
            )
        return ReconcileDecision(
            SPREAD,
            wanted,
            "",
            count,
            f"Spreads {wanted} copies across {count} candidate machines.",
            existing_constraints,
        )

    # run-once / pin: every replica lands on the one constrained machine.
    match = next((node for node in eligible if node.node_id == node_id), None)
    if match is None:
        blocked = next(
            (
                item for item in excluded
                if str(item.get("node_id", "")) == node_id
            ),
            None,
        )
        raise PlacementError(
            "cluster_app_reconfigure_pin_capacity",
            (blocked or {}).get("reason")
            or "The machine this app runs on can no longer host it.",
        )
    needed = pin_capacity_bytes(wanted, required)
    if match.free_memory_bytes < needed:
        raise PlacementError(
            "cluster_app_reconfigure_pin_capacity",
            f"Running {wanted} copies on {match.name} needs about "
            f"{_gib(needed)}; it has {_gib(match.free_memory_bytes)} free once "
            "this app's own reservation is returned.",
        )
    return ReconcileDecision(
        intent,
        wanted,
        node_id,
        count,
        f"{wanted} replica{'' if wanted == 1 else 's'} on {match.name}.",
        existing_constraints,
    )


def validate_cpu_limit(value: Any) -> float:
    """The operator's per-replica CPU limit in cores, defaulted and bounded.

    An empty/absent value is the 2.0-core default (the single-node compose's
    literal). CPU never gates eligibility — this only bounds the container — so
    the check is a plain range, not a capacity question.
    """
    if value is None or value == "":
        return CPU_LIMIT_DEFAULT
    try:
        cpu = float(value)
    except (TypeError, ValueError):
        cpu = None
    if cpu is None or not CPU_LIMIT_MIN <= cpu <= CPU_LIMIT_MAX:
        raise PlacementError(
            "cluster_app_cpu", "Choose a CPU limit between 0.25 and 64 cores."
        )
    return round(cpu, 2)


def cpu_reservation_cores(cpu_limit: float) -> float:
    """The conservative scheduling reservation for a CPU limit.

    A fixed quarter-core, never larger than the limit itself, so an app's CPU
    reservation never dominates placement (Swarm schedules against reservations).
    """
    return round(min(CPU_RESERVATION_CORES, float(cpu_limit)), 2)


def build_label_constraints(raw: Any) -> List[str]:
    """Validate operator label constraints and render them as Swarm constraints.

    ``raw`` is a list of ``{key, op, value}`` items. Each is checked to a safe
    shape (``op`` in ``==``/``!=``; key/value the bounded ``[A-Za-z0-9._-]``
    charset), and a key in the reserved ``vaelor.*``/``pironman.*`` namespace is
    REFUSED (S2) so a user constraint can only ADD to the managed pin/intent,
    never redefine or hijack a managed placement label. Returns full
    ``node.labels.<key><op><value>`` expressions the deploy/configure emit.
    """
    items = raw or []
    if not isinstance(items, (list, tuple)):
        raise PlacementError(
            "cluster_app_label", "Label constraints must be a list."
        )
    if len(items) > MAX_LABEL_CONSTRAINTS:
        raise PlacementError(
            "cluster_app_label",
            f"Use at most {MAX_LABEL_CONSTRAINTS} label constraints.",
        )
    constraints: List[str] = []
    for item in items:
        if not isinstance(item, dict):
            raise PlacementError(
                "cluster_app_label",
                "Each label constraint needs a key, an operator and a value.",
            )
        key = str(item.get("key", "")).strip()
        op = str(item.get("op", "")).strip()
        value = str(item.get("value", "")).strip()
        if op not in LABEL_CONSTRAINT_OPS:
            raise PlacementError(
                "cluster_app_label_op",
                "A label constraint operator must be == or !=.",
            )
        if not _LABEL_TOKEN.fullmatch(key) or not _LABEL_TOKEN.fullmatch(value):
            raise PlacementError(
                "cluster_app_label_charset",
                "Label keys and values use letters, numbers, dot, dash and "
                "underscore, up to 63 characters.",
            )
        # A key typed as the whole `node.labels.<x>` expression would emit
        # `node.labels.node.labels.<x>` and constrain a label no node carries -
        # a task that hangs Pending on a typo. Enter the bare key; Vaelor adds
        # the prefix. (This also catches `node.labels.vaelor.node_id`, which the
        # reserved-namespace check below would otherwise miss.)
        if key.lower().startswith(("node.labels.", "node.")):
            raise PlacementError(
                "cluster_app_label_key_shape",
                "Enter just the label key (e.g. 'disktype'), not the whole "
                "'node.labels.<key>' expression - Vaelor adds that prefix.",
            )
        if key.lower().startswith(_RESERVED_LABEL_PREFIXES):
            raise PlacementError(
                "cluster_app_label_reserved",
                "The vaelor.* and pironman.* labels are managed by Vaelor; "
                "constrain against your own node labels instead.",
            )
        constraints.append(f"node.labels.{key}{op}{value}")
    return constraints
