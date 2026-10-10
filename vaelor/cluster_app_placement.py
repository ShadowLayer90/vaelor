"""Where a catalog app's replicas land — the one decision the plan and the
deploy share (D1).

PLACEMENT ONLY. This module answers *which machines run how many copies of an
app, and under what Swarm constraints* — nothing about the app's environment or
secrets, which the deploy job body mints once and never the plan (B2). It is
imported by both `cluster_plan_services.plan_deploy_app` (the preview) and
`cluster_app_deploy.deploy_catalog_app` (the act), so the same `reconcile`
decides both and a preview can never promise a placement the deploy then refuses
(VD-B3b-1).

The core — `eligible_app_nodes` and `reconcile` — is a pure function of
already-fetched capacity facts, unit-tested without SSH or Docker. The single
I/O helper, `app_capacity_ledger`, fetches those facts the same way for both
sides, so eligibility is read off one shape however the caller reached it.

Three intents, plain-language in the UI:

* **run-once** — one replica on the single best-fit eligible machine, chosen by
  the most reservation-aware free system memory the capacity ledger reports
  (`cluster_capacity` `free.memory_bytes`, NOT raw inventory: a machine already
  hosting services has less room than its nameplate).
* **spread** — N replicas, at most one per machine (`--replicas-max-per-node 1`).
  Refused when N exceeds the eligible-machine count K, and refused for a
  STATEFUL template (one with a data volume): spreading replicas over per-task
  local volumes splits the app's data. Only volumeless apps may spread.
* **pin** — N replicas, all constrained to the operator-chosen machine.

Eligibility is deliberately simple (VD-031 keeps one architecture per fleet, so
per-node arch filtering is redundant — the image supports the fleet arch or the
whole deploy is refused): a machine is eligible when it reported its memory and
has enough free for the app plus 1 GB of OS headroom. A machine whose memory is
unreadable is EXCLUDED with a named reason rather than silently credited or
refused (honest degradation). The controller counts as an ordinary placement
target — apps are ordinary Swarm workloads, unlike the GPU pooled path.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional, Tuple

from .cluster_capacity import (
    compute_capacity_ledger,
    controller_node_facts,
    reservation_from_service_details,
)
from .cluster_placement import CONTROLLER_PLACEMENT_ID
from .cluster_store import controller_recorded

#: The three placement intents an operator chooses between, and the label each
#: carries as a managed service label so `service_details` can read the decision
#: back (mirrors `vaelor.template=`).
RUN_ONCE = "run-once"
SPREAD = "spread"
PIN = "pin"
VALID_INTENTS = (RUN_ONCE, SPREAD, PIN)

#: Swarm's sane ceiling on replica count, and the one the researched deploy caps
#: a spread at (`cluster_app_multideploy._replicas`). The fit references it so a
#: spread over a >32-node fleet reserves for — and reports — the 32 replicas the
#: deploy will actually run, rather than over-reserving for every eligible node.
MAX_SPREAD_REPLICAS = 32

#: The OS headroom every app placement preserves on a machine, matching the
#: single-node deploy's own "preserve 1 GB for the worker operating system"
#: rule. Folded into the memory an eligibility check demands to be free.
OS_HEADROOM_BYTES = 1024 ** 3

_GIB = 1024 ** 3
_MIB = 1024 ** 2


class PlacementError(ValueError):
    """A placement refusal the plan renders as an API error and the deploy raises.

    Carries a stable ``code`` so the plan builder can surface the same
    machine-readable reason the rest of the cluster plan layer uses, while the
    message stays plain-language for the operator.
    """

    def __init__(self, code: str, message: str):
        super().__init__(message)
        self.code = code
        self.message = message


@dataclass(frozen=True)
class EligibleNode:
    """One machine that can run the app, with the free memory that ranked it."""

    node_id: str
    name: str
    free_memory_bytes: int


@dataclass(frozen=True)
class PlacementDecision:
    """The reconciled placement: how many replicas, and the Swarm shape.

    ``constraints`` are full ``node.labels.vaelor.node_id==<id>`` expressions
    (one for run-once/pin, none for spread — Swarm's own spread does the
    scheduling). ``flags`` are extra ``service create`` arguments
    (``--replicas-max-per-node 1`` for spread). ``node_ids`` names the machines
    a constraint pins to, so the deploy knows which to label the controller for;
    ``chosen`` carries the machines for the plain-language read-back.
    """

    intent: str
    replicas: int
    constraints: List[str]
    flags: List[str]
    node_ids: List[str]
    chosen: List[EligibleNode]
    eligible_count: int
    stateful: bool = False

    def as_tuple(self) -> Tuple[int, List[str], List[str]]:
        """The ``(replicas, constraints, flags)`` the deploy hands the driver."""
        return self.replicas, list(self.constraints), list(self.flags)


def node_constraint(node_id: str) -> str:
    """The Swarm placement constraint pinning a service to one enrolled machine."""
    return f"node.labels.vaelor.node_id=={node_id}"


def _gib(value: Any) -> str:
    try:
        return f"{int(value) / _GIB:.1f} GiB"
    except (TypeError, ValueError):
        return "an unknown amount"


def required_memory_bytes(memory_mib: Any) -> int:
    """Free bytes a machine must have to run an app of ``memory_mib`` — plus OS."""
    try:
        app_bytes = int(memory_mib) * _MIB
    except (TypeError, ValueError):
        app_bytes = 0
    return max(0, app_bytes) + OS_HEADROOM_BYTES


def pin_capacity_bytes(replicas: Any, required_bytes: int) -> int:
    """Bytes a pin of ``replicas`` copies needs on its one machine.

    A pin lands ALL replicas on a single machine, so it needs one per-replica
    reservation (``required_bytes`` less the single OS headroom) for each copy,
    plus that one OS headroom. The deploy's pin check and the reconfigure
    re-check size against this one function, so they cannot disagree.
    """
    try:
        wanted = max(0, int(replicas))
    except (TypeError, ValueError):
        wanted = 0
    per_replica = max(0, int(required_bytes) - OS_HEADROOM_BYTES)
    return wanted * per_replica + OS_HEADROOM_BYTES


def eligible_app_nodes(
    ledger_nodes: List[Dict[str, Any]], required_bytes: int
) -> Tuple[List[EligibleNode], List[Dict[str, str]]]:
    """Split the ledger's machines into those that can run the app and those that
    cannot, each exclusion carrying its named reason and a machine-readable
    ``reason_code``.

    Pure over `compute_capacity_ledger` node rows. A machine is excluded when it
    has not JOINED the swarm (an enrolled worker whose join failed or is pending
    has real free memory but no ``vaelor.node_id`` label for a constraint to
    match, so a run-once/pin decision on it would emit a constraint matching
    nothing and hang the task Pending — VD-B3b-1: the preview must refuse it, not
    promise it), when it never reported its memory (``capacity.memory_bytes`` <= 0
    — unreadable, a refresh is the remedy, never a silent zero), or when its
    reservation-aware ``free.memory_bytes`` cannot cover the app plus 1 GB of OS
    headroom. Join-state rides on each row as ``joined`` (set by
    `app_capacity_ledger` from the store's ``labels.swarm_node_id``; the
    controller is always joined because it is labelled on demand). A row without
    the key — a hand-built ledger row in a pure test — is treated as joined.
    """
    eligible: List[EligibleNode] = []
    excluded: List[Dict[str, str]] = []
    for node in ledger_nodes:
        node_id = str(node.get("node_id", ""))
        name = str(node.get("name", "")) or node_id or "A machine"
        if not node.get("joined", True):
            excluded.append({
                "node_id": node_id,
                "name": name,
                "reason_code": "not_joined",
                "reason": (
                    f"{name} hasn't joined this cluster; join it before "
                    "deploying an app to it."
                ),
            })
            continue
        capacity = (node.get("capacity") or {}).get("memory_bytes", 0)
        free = (node.get("free") or {}).get("memory_bytes", 0)
        try:
            capacity_bytes = int(capacity or 0)
            free_bytes = int(free or 0)
        except (TypeError, ValueError):
            capacity_bytes = free_bytes = 0
        if capacity_bytes <= 0:
            excluded.append({
                "node_id": node_id,
                "name": name,
                "reason_code": "memory_unreadable",
                "reason": (
                    f"{name} did not report how much memory it has; refresh it "
                    "to capture its capacity."
                ),
            })
            continue
        if free_bytes < required_bytes:
            excluded.append({
                "node_id": node_id,
                "name": name,
                "reason_code": "insufficient_memory",
                "reason": (
                    f"{name} has {_gib(free_bytes)} free; this app needs "
                    f"{_gib(required_bytes)} including 1 GB of OS headroom."
                ),
            })
            continue
        eligible.append(EligibleNode(node_id, name, free_bytes))
    return eligible, excluded


def _best_fit(eligible: List[EligibleNode]) -> EligibleNode:
    # Most free memory wins; the node id breaks a tie so the choice is
    # deterministic (a preview and the deploy pick the same machine when two are
    # level, rather than depending on ledger order).
    return sorted(
        eligible, key=lambda node: (-node.free_memory_bytes, node.node_id)
    )[0]


def reconcile(
    intent: str,
    replicas: int,
    eligible_nodes: List[EligibleNode],
    is_stateful: bool,
    node_id: Optional[str] = None,
    excluded: Optional[List[Dict[str, str]]] = None,
    required_bytes: int = 0,
) -> PlacementDecision:
    """Reconcile a placement intent + replica count against the eligible machines.

    The one decision the plan and the deploy share. Raises `PlacementError`
    with the arithmetic when a request cannot be honoured, so a preview refuses
    for exactly the reason the deploy would.

    ``excluded`` is the companion list from `eligible_app_nodes`: when a pin
    names a machine that was excluded, its ``reason_code`` decides the refusal —
    a non-joined target is refused as "join it first" rather than the generic
    "not eligible", restoring the guard the single-node deploy carried.
    ``required_bytes`` is the single-replica requirement (app + 1 GB OS
    headroom); a pin of N copies is refused unless the target has room for N
    per-replica reservations plus that one headroom, because pin lands ALL N on
    the one machine and eligibility only proved room for one.
    """
    chosen_intent = str(intent or "").strip().lower() or RUN_ONCE
    if chosen_intent not in VALID_INTENTS:
        raise PlacementError(
            "cluster_app_intent",
            "Choose run-once, spread, or pinned placement.",
        )
    try:
        wanted = int(replicas)
    except (TypeError, ValueError):
        wanted = 0
    count = len(eligible_nodes)
    # A pin naming a machine that was EXCLUDED is refused for that machine's own
    # reason, ahead of the generic no-capacity refusal — so pinning to the one
    # enrolled-but-unjoined node reads "join it first", not "no machine can run
    # this app". Run-once/spread never name a machine, so this is pin-only.
    if chosen_intent == PIN:
        target = str(node_id or "")
        if not any(node.node_id == target for node in eligible_nodes):
            blocked = next(
                (
                    item for item in (excluded or [])
                    if str(item.get("node_id", "")) == target
                ),
                None,
            )
            if blocked is not None and blocked.get("reason_code") == "not_joined":
                raise PlacementError(
                    "cluster_worker_required",
                    "Choose a worker that has joined this cluster.",
                )
            if blocked is not None:
                raise PlacementError("cluster_app_pin_target", blocked["reason"])
    if count == 0:
        raise PlacementError(
            "cluster_app_no_capacity",
            "No enrolled machine can run this app right now; free memory on a "
            "machine or add one, then try again.",
        )

    if chosen_intent == RUN_ONCE:
        best = _best_fit(eligible_nodes)
        return PlacementDecision(
            intent=RUN_ONCE,
            replicas=1,
            constraints=[node_constraint(best.node_id)],
            flags=[],
            node_ids=[best.node_id],
            chosen=[best],
            eligible_count=count,
        )

    if chosen_intent == SPREAD:
        if is_stateful:
            raise PlacementError(
                "cluster_app_stateful_spread",
                "This app keeps its data on the machine it runs on, so its "
                "replicas cannot be spread across several without splitting that "
                "data. Deploy it run-once or pinned instead.",
            )
        if not 2 <= wanted <= 32:
            raise PlacementError(
                "cluster_app_replicas",
                "Choose between 2 and 32 replicas to spread across machines.",
            )
        if wanted > count:
            machines = "machine" if count == 1 else "machines"
            raise PlacementError(
                "cluster_app_spread_capacity",
                f"Spread wants {wanted} copies but only {count} eligible "
                f"{machines}; reduce the replicas or add a machine.",
            )
        return PlacementDecision(
            intent=SPREAD,
            replicas=wanted,
            constraints=[],
            flags=["--replicas-max-per-node", "1"],
            node_ids=[],
            chosen=list(eligible_nodes),
            eligible_count=count,
        )

    # PIN. A target that was excluded raised above with its own reason (join it
    # first, unreadable memory, too full); reaching here with no match means the
    # named machine is not in the fleet at all.
    match = next(
        (node for node in eligible_nodes if node.node_id == str(node_id or "")),
        None,
    )
    if match is None:
        raise PlacementError(
            "cluster_app_pin_target",
            "Choose a machine that can run this app; the one selected is not "
            "eligible.",
        )
    if not 1 <= wanted <= 32:
        raise PlacementError(
            "cluster_app_replicas",
            "Choose 1 to 32 replicas for this pinned app.",
        )
    # Pin lands ALL N replicas on the one machine; eligibility only proved room
    # for a single replica, so N copies of a per-replica reservation plus one OS
    # headroom must actually fit or the extra replicas hang Pending (VD-B3b-1).
    if required_bytes:
        needed = pin_capacity_bytes(wanted, required_bytes)
        if match.free_memory_bytes < needed:
            raise PlacementError(
                "cluster_app_pin_capacity",
                f"Pinning {wanted} copies of this app needs about "
                f"{_gib(needed)}; {match.name} has "
                f"{_gib(match.free_memory_bytes)} free.",
            )
    return PlacementDecision(
        intent=PIN,
        replicas=wanted,
        constraints=[node_constraint(match.node_id)],
        flags=[],
        node_ids=[match.node_id],
        chosen=[match],
        eligible_count=count,
    )


def describe_placement(decision: PlacementDecision) -> str:
    """The plain-language read-back the plan renders and the modal echoes."""
    if decision.intent == RUN_ONCE and decision.chosen:
        best = decision.chosen[0]
        return (
            f"Best fit: {best.name}, {_gib(best.free_memory_bytes)} free — "
            "one replica."
        )
    if decision.intent == SPREAD:
        return (
            f"{decision.replicas} replicas, one per machine across "
            f"{decision.replicas} of {decision.eligible_count} eligible machines."
        )
    if decision.intent == PIN and decision.chosen:
        pinned = decision.chosen[0]
        plural = "" if decision.replicas == 1 else "s"
        return (
            f"Pinned to {pinned.name}: {decision.replicas} replica{plural}."
        )
    return "Placement decided at deploy time."


def app_capacity_ledger(
    store: Any,
    driver: Any,
    inventory_probe: Optional[Callable[[], Dict[str, Any]]],
    exempt_service: str = "",
) -> List[Dict[str, Any]]:
    """The capacity-ledger machines for the app path, fetched once for both sides.

    The single I/O function of this module, called with the plan's manager and
    the deploy's operations facade alike (both carry ``store``, ``driver`` and
    ``inventory_probe``), so eligibility reads one shape however it was reached —
    the preview==deploy guarantee for the app path (VD-B3b-1).

    It mirrors `cluster_manager.capacity_ledger`'s fetch but leaves out the GPU
    reclaimable question, which only the accelerator fields answer and an app
    (a CPU/memory workload sized on ``free.memory_bytes``) never reads. The
    arithmetic is `compute_capacity_ledger`, the same pure function the capacity
    view uses, so the two cannot disagree about a machine's free memory.

    The controller is a placement target only when it is an active,
    control-available head; otherwise the ledger is the enrolled workers alone,
    exactly as the capacity view scopes it.

    ``exempt_service`` names a managed service whose reservation is left OUT of
    the ledger sum — the reconfigure path passes the service being changed so it
    does not count against its own eligibility (B1).
    """
    status = driver.status()
    # Only a machine that JOINED the swarm carries the `vaelor.node_id` label a
    # placement constraint matches (applied at join, `cluster_operations`), so an
    # enrolled-but-unjoined worker is not a valid placement target however much
    # free memory it has. The controller is always joinable — it is labelled on
    # demand — so it seeds the set.
    joined_ids = {CONTROLLER_PLACEMENT_ID}
    node_facts: List[Dict[str, Any]] = []
    for node in store.list_nodes():
        node_facts.append({
            "node_id": node["id"],
            "name": node["name"],
            "role": node.get("role", "worker"),
            "inventory": node.get("inventory", {}),
        })
        if node.get("labels", {}).get("swarm_node_id"):
            joined_ids.add(str(node["id"]))
    controller = store.controller()
    if controller_recorded(controller) and status.get("control_available"):
        try:
            hardware = (inventory_probe() if inventory_probe else {}) or {}
        except (AttributeError, OSError, TypeError, ValueError):
            hardware = {}
        node_facts.append(
            controller_node_facts(controller, status, hardware)
        )
    reservations: List[Dict[str, Any]] = []
    for service in status.get("services", []):
        name = str(service.get("name", ""))
        if not name.startswith(("vaelor-app-", "vaelor-llm-")):
            continue
        # B1: a reconfigure nets out the TARGET service's own reservation, or the
        # running service self-excludes and a no-op change (same replicas) is
        # wrongly refused. Skipping it here reads the ledger as if the service
        # were absent, so the re-check asks the same question the first deploy
        # did — this is the ONLY correct way to size a change to a live service.
        if exempt_service and name == exempt_service:
            continue
        try:
            details = driver.service_details(name)
        except Exception:
            # A service that will not inspect is skipped rather than failing the
            # whole ledger; its reservation is simply not yet counted, exactly as
            # the capacity view treats it.
            continue
        reservations.append(reservation_from_service_details(details))
    nodes = compute_capacity_ledger(node_facts, reservations)["nodes"]
    # Ride join-state onto each row so `eligible_app_nodes` — a pure function of
    # ledger rows — can drop an unjoined machine with a named reason.
    for row in nodes:
        row["joined"] = str(row.get("node_id", "")) in joined_ids
    return nodes


# --- Multi-service fit (D4b) -------------------------------------------------
#
# A researched app is N services placed TOGETHER, so the single-service
# `reconcile` above is not enough on its own: it reads a static ledger, and two
# co-deployed services would each be told the same node fits when together they
# do not (B7). `fit_app_services` places the services against a RUNNING tally —
# it fetches the reservation-aware ledger ONCE, then decrements a node's free
# memory as each service is placed, so the second service sees the first's
# reservation even though neither is created yet. It reuses `eligible_app_nodes`
# and `reconcile` per service (so a fit refuses for exactly the reason the
# single-service deploy would) and adds the honest arithmetic the app path needs:
# a stateful pin sized against the running tally, a stateless spread whose
# replica count is the eligible-machine count, and a cross-fleet published-port
# collision the renderer's within-app check cannot see.


def pin_node_required_message(service_key: str) -> str:
    """The one refusal for a stateful service with no chosen pin node.

    Shared by the fit (which refuses it in the preflight, before any side effect)
    and `cluster_app_multideploy._placement_arguments` (the deploy's own guard),
    so the two cannot drift into two spellings of one refusal.
    """
    return (
        "Service {} keeps data and must be pinned to a machine; choose a node "
        "for it before deploying.".format(service_key)
    )


def _parse_live_ports(ports_text: Any) -> List[Tuple[int, str]]:
    """The (published, protocol) pairs in a ``docker service ls`` ``Ports`` cell.

    The cell is human text — ``*:8080->80/tcp, 10.0.0.1:9090->90/udp`` — parsed
    without a regex (so no new alternation rule joins the reachability table): the
    number before ``->`` is the ingress published port, the token after the final
    ``/`` its protocol.
    """
    pairs: List[Tuple[int, str]] = []
    for chunk in str(ports_text or "").split(","):
        chunk = chunk.strip()
        if "->" not in chunk:
            continue
        left, right = chunk.split("->", 1)
        published = left.rsplit(":", 1)[-1].strip()
        protocol = (
            right.split("/")[-1].strip().lower() if "/" in right else "tcp"
        )
        if published.isdigit():
            pairs.append((int(published), protocol or "tcp"))
    return pairs


@dataclass(frozen=True)
class ServiceFit:
    """One service's confirmed placement, ready for the deploy to thread.

    ``replicas`` is 1 for a pin/run-once and the eligible-machine count for a
    spread; ``node_ids``/``node_names`` name the machines it lands on (one for
    run-once/pin, one per replica for spread). ``memory_defaulted`` carries the
    renderer's unknown-footprint flag so the plan surfaces it.
    """

    service_key: str
    service_name: str
    stateful: bool
    intent: str
    replicas: int
    memory_mib: int
    memory_defaulted: bool
    node_ids: List[str] = field(default_factory=list)
    node_names: List[str] = field(default_factory=list)


@dataclass(frozen=True)
class AppFitPlan:
    """Every service of one researched app placed against one running tally."""

    services: List[ServiceFit] = field(default_factory=list)


def _service_intent(spec: Any) -> str:
    """The placement intent for one rendered service: stateful always pins."""
    if getattr(spec, "stateful", False):
        return PIN
    placement = getattr(spec, "placement", {}) or {}
    return str(placement.get("intent") or "").strip().lower() or RUN_ONCE


def _fit_rank(spec: Any) -> Tuple[int, str]:
    # Deterministic placement order for the running tally: fixed-node pins first
    # (they have no choice of machine, so their demand is reserved before the
    # flexible services compete for what's left), then run-once (a single
    # best-fit), then spread (which fills every remaining eligible machine). The
    # service key breaks ties so the order — and therefore which service a
    # capacity shortfall is charged to — is stable between a preview and a deploy.
    intent = _service_intent(spec)
    rank = {PIN: 0, RUN_ONCE: 1, SPREAD: 2}.get(intent, 1)
    return rank, str(getattr(spec, "service_key", ""))


def _decrement(tally: List[Dict[str, Any]], node_id: str, app_bytes: int) -> None:
    """Net a placed service's per-replica reservation out of one node's free
    memory in the running tally (B7). The OS headroom is a threshold every
    service re-checks, not a per-service consumption, so only the app bytes are
    subtracted — matching `pin_capacity_bytes`, whose per-replica figure is
    exactly ``required_bytes`` less the single headroom."""
    for row in tally:
        if str(row.get("node_id", "")) == node_id:
            free = row.get("free") or {}
            free["memory_bytes"] = max(0, int(free.get("memory_bytes", 0)) - app_bytes)
            row["free"] = free
            return


def _refuse_live_port_collisions(specs: List[Any], status: Dict[str, Any]) -> None:
    """Refuse an ingress published port already live on the fleet (B5, impure).

    The renderer already rejects two services of THIS app sharing a port; this
    is the cross-fleet half it cannot see — a port another managed
    (``vaelor-app-*``/``vaelor-llm-*``) service is already publishing through the
    Swarm routing mesh, where a second publisher of the same ingress port cannot
    come up. Read from live ``docker service ls`` state so the refusal is honest
    about what is actually running, before any deploy side effect.
    """
    live: Dict[Tuple[int, str], str] = {}
    for service in status.get("services", []) or []:
        if not isinstance(service, dict):
            continue
        name = str(service.get("name", ""))
        if not name.startswith(("vaelor-app-", "vaelor-llm-")):
            continue
        for published, protocol in _parse_live_ports(service.get("ports", "")):
            live[(published, protocol)] = name
    for spec in specs:
        for port in getattr(spec, "published_ports", []) or []:
            try:
                published = int(port.get("published") or 0)
            except (TypeError, ValueError):
                published = 0
            if not published:
                continue
            protocol = str(port.get("protocol", "tcp")).strip().lower() or "tcp"
            holder = live.get((published, protocol))
            if holder:
                raise PlacementError(
                    "cluster_app_port_live",
                    "Service {} publishes port {}/{}, which {} is already using "
                    "on this cluster; free that port or change the app's port "
                    "before deploying.".format(
                        getattr(spec, "service_key", ""), published, protocol,
                        holder,
                    ),
                )


def _place_stateful(
    spec: Any, tally: List[Dict[str, Any]], required: int
) -> Tuple[ServiceFit, int]:
    placement = getattr(spec, "placement", {}) or {}
    pin_node = str(placement.get("pin_node") or "").strip()
    if not pin_node:
        raise PlacementError(
            "cluster_app_pin_target",
            pin_node_required_message(getattr(spec, "service_key", "")),
        )
    eligible, excluded = eligible_app_nodes(tally, required)
    try:
        decision = reconcile(
            PIN, 1, eligible, is_stateful=True, node_id=pin_node,
            excluded=excluded, required_bytes=required,
        )
    except PlacementError as error:
        raise PlacementError(
            error.code,
            "Pinning {}: {}".format(getattr(spec, "service_key", ""), error.message),
        ) from error
    return _fit_from_nodes(spec, decision.chosen, required)


def _place_run_once(
    spec: Any, tally: List[Dict[str, Any]], required: int
) -> Tuple[ServiceFit, int]:
    eligible, excluded = eligible_app_nodes(tally, required)
    try:
        decision = reconcile(
            RUN_ONCE, 1, eligible, is_stateful=False,
            excluded=excluded, required_bytes=required,
        )
    except PlacementError as error:
        raise PlacementError(
            error.code,
            "Placing {}: {}".format(getattr(spec, "service_key", ""), error.message),
        ) from error
    return _fit_from_nodes(spec, decision.chosen, required)


def _place_spread(
    spec: Any, tally: List[Dict[str, Any]], required: int
) -> Tuple[ServiceFit, int]:
    # A researched spread runs ONE replica per eligible machine — the count is
    # decided by the fleet, not requested — so it is refused only when no machine
    # can host it, with the same per-node arithmetic `eligible_app_nodes` gives.
    eligible, excluded = eligible_app_nodes(tally, required)
    if not eligible:
        detail = "; ".join(item["reason"] for item in excluded) or (
            "no enrolled machine reported enough free memory."
        )
        raise PlacementError(
            "cluster_app_spread_capacity",
            "Spreading {} needs at least one machine with room, but none "
            "qualifies: {}".format(getattr(spec, "service_key", ""), detail),
        )
    # Swarm caps the replica count at MAX_SPREAD_REPLICAS, so a spread over a
    # larger fleet lands on only that many machines. Cap the fit at the same
    # ceiling the deploy uses so the fit reserves for — and reports — the replica
    # count the deploy will actually run, not one per eligible node (M2).
    return _fit_from_nodes(spec, eligible[:MAX_SPREAD_REPLICAS], required)


def _fit_from_nodes(
    spec: Any, chosen: List[EligibleNode], required: int
) -> Tuple[ServiceFit, int]:
    """Build the resolution for a placed service and — the load-bearing step —
    net each chosen machine's per-replica reservation out of the running tally so
    a later sibling sees this service's demand (B7)."""
    app_bytes = max(0, required - OS_HEADROOM_BYTES)
    return ServiceFit(
        service_key=str(getattr(spec, "service_key", "")),
        service_name=str(getattr(spec, "service_name", "")),
        stateful=bool(getattr(spec, "stateful", False)),
        intent=_service_intent(spec),
        replicas=len(chosen),
        memory_mib=int(getattr(spec, "memory_mib", 0) or 0),
        memory_defaulted=bool(getattr(spec, "memory_defaulted", False)),
        node_ids=[node.node_id for node in chosen],
        node_names=[node.name for node in chosen],
    ), app_bytes


def fit_app_services(
    specs: List[Any],
    store: Any,
    driver: Any,
    inventory_probe: Optional[Callable[[], Dict[str, Any]]],
) -> AppFitPlan:
    """Place every service of a researched app against ONE running ledger tally.

    The load-bearing correctness fix (B7). It fetches the reservation-aware
    ledger once (`app_capacity_ledger`), then places the services in a
    deterministic order — pins, then run-once, then spread — DECREMENTING a
    node's free memory as each service lands, so two co-deployed services are
    never each told the same node fits when together they do not. Each service is
    sized with the same `eligible_app_nodes`/`reconcile` the single-service deploy
    uses (so a memory-unreadable or unjoined node is EXCLUDED with its named
    reason, never credited zero), and the cross-fleet published-port collision is
    refused up front. Raises `PlacementError` with per-service arithmetic on any
    refusal; the plan and the deploy call this identically, so a preview can never
    promise a placement the deploy then refuses (preview==deploy).
    """
    _refuse_live_port_collisions(specs, driver.status())
    # A private, mutable copy of the ledger rows — the free dict is re-created so
    # decrementing the tally never mutates the fetched ledger a caller may reuse.
    tally: List[Dict[str, Any]] = [
        {**row, "free": dict(row.get("free") or {})}
        for row in app_capacity_ledger(store, driver, inventory_probe)
    ]
    fits: List[ServiceFit] = []
    for spec in sorted(specs, key=_fit_rank):
        required = required_memory_bytes(getattr(spec, "memory_mib", 0))
        intent = _service_intent(spec)
        if intent == PIN:
            fit, app_bytes = _place_stateful(spec, tally, required)
        elif intent == SPREAD:
            fit, app_bytes = _place_spread(spec, tally, required)
        else:
            fit, app_bytes = _place_run_once(spec, tally, required)
        for node_id in fit.node_ids:
            _decrement(tally, node_id, app_bytes)
        fits.append(fit)
    # Return in manifest order so the plan reads services the way the app defines
    # them, not in the internal placement order.
    order = {str(getattr(spec, "service_key", "")): index
             for index, spec in enumerate(specs)}
    fits.sort(key=lambda item: order.get(item.service_key, 0))
    return AppFitPlan(services=fits)
