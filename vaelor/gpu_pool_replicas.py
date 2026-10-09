"""The throughput intent: one replica of the model per machine, behind a balancer.

VD-129, designed from the measurement VD-127 recorded: on the pair, N
independent single-node vLLM servers behind the product's own nginx gave 959
tokens per second aggregate at 128 concurrent against 886 for Ray
data-parallel, held goodput at 100 percent at 64 concurrent where data-parallel
held 72, and lost only the three in-flight streams when a node died. So the
primitive is a replica per machine and a balancer, not a collective - no Ray,
no RCCL, no cluster interface - and this module is that primitive's home,
housed out of `gpu_pool_operations`, `gpu_pool_runtime` and `gpu_cluster_mode`
because each of those sits within a few lines of the ceiling (VD-127 cleanup
item 26). The units' names and the record's derivation stay `gpu_pool_units`'s;
the commands stay the runtime's; the fit verdict stays the engine's.

**Two processes run this file, and it matters which** (LESSONS pattern 14):

* The WORKLOAD EXECUTOR runs :class:`ReplicatedServing` (the deploy job's
  replicated path after the shared prefix in `GpuPoolOperations.deploy`),
  :class:`ReplicaStartupWaiter` (the deploy's wait), :class:`BalancerController`
  (the executor-side seam the deploy, `stop_units_for` and the mode switch's
  reconcile drive), and :class:`ReplicaHealth` (the 30 s mode watch's reading
  of a healthy row's replicas).
* The ROOT HARDWARE BRIDGE runs :class:`BalancerProcess` and
  :func:`render_balancer_config` behind its ``balancer_start``/``stop``/``status``
  verbs, shaped exactly like the LLM Server proxy's: the config is rendered
  root-side under ``/run/vaelor`` at ``0600``, mounted read-only, and the key
  is on no argv. Both, with the controller, live in `gpu_pool_balancer` and
  are re-exported here.

**The endpoint (the VD-129 amendment, after the first live probe).** EVERY
replica binds loopback and carries no key. vLLM's own ``--api-key`` was probed
on the laptop before anything shipped: every ``/v1`` path answered 401 unkeyed,
but ``POST /invocations`` (the SageMaker route) returned a full chat completion
with no key, and ``/tokenize`` and ``/metrics`` answered too - a keyed worker
replica bound to the LAN was an unkeyed inference endpoint through a side
door. So on each WORKER the runtime starts a gate beside the replica
(`GpuPoolRuntime.start_gate`): the product's nginx image, listening on the
worker's cluster address on the replica port, requiring the cluster key on
``/v1/``, forwarding ``/health`` unkeyed, and answering 404 to everything else,
its config a root ``0600`` file on that worker with the key in it and nowhere
else. The balancer listens on ``127.0.0.1:<port>``, pools the controller's
loopback replica and every worker's GATE (``<address>:<replica port>`` - the
address the record wrote), sends a request that names its conversation in
``X-Session-Id`` to the same replica every time and spreads the rest (VD-157,
`session_affinity`), gives up on a replica that fails before any reply and
sends that request once to the other (VD-149), and adds ``Authorization: Bearer <key>``
to every ``/v1/`` request it forwards, so the one key the deploy mints and
records on the cluster credential opens every gate. **The balancer answers the
SAME path map as a worker's gate** (the final VD-129 review's S2 finding,
fixed): ``/health`` unkeyed, ``/v1/`` with the key stamped on, and 404 to
everything else, both rendered from `llm_server_proxy.replica_locations` - so
a keyed client asking ``/tokenize``, ``/invocations`` or ``/metrics`` gets the
one answer regardless of which replica the balancer happens to pick, rather
than 200 through the keyless controller replica and 404 through a worker's
gate (LESSONS pattern 6). The LLM Server proxy fronts the balancer's loopback
port as one ``model_port`` and is otherwise untouched.

**Health.** ``/health`` is probed unauthenticated - the controller's replica
directly, a worker's THROUGH its gate, which forwards that one path unkeyed
exactly as the LLM Server proxy does, so the probe sees the gate and the
replica at once - by the startup wait (round-robin over the replicas, each
unit read through its own transport, a worker's gate unit beside its server
unit) and by the mode watch (over HTTP, no transport).
``Restart=on-failure`` with ``StartLimitIntervalSec=0`` handles a
crashed process; the watch writes ``units.replicas[].alive`` and
``units.replicas[].reachable`` - a replica that answered its check unhealthy
is alive-false and reachable-true; one at which nothing answered at all (a
node that left the cluster, was unplugged, or is rebooting) is both false -
and, at 0 of N for four consecutive passes, marks the row failed naming every
replica's last state, which the reconcile's ``no-healthy-deployment`` leave
then tears down. No automatic eviction, no re-placement.
"""

from __future__ import annotations

import json
import logging
import time
from dataclasses import dataclass
from typing import Any, Callable, Dict, List, Mapping, Optional, Sequence, Tuple

from .cluster_capacity import gpu_fit_node
from .cluster_gpu_sizing import VERDICT_REPLICATED
from .cluster_placement import CONTROLLER_PLACEMENT_ID
from .gpu_pool_pull import PULL_TRANSPORT_FAILURES
from .gpu_pool_startup import (
    API_POLL_SECONDS, NODE_UNREACHABLE, STARTUP_CHECKPOINT_PERCENT,
    STARTUP_REPORT_SECONDS, UNREACHABLE, server_died, startup_cap_seconds,
    unit_alive,
)
# The balancer lives in its own module; its names are re-exported here so
# every caller of this module - and its tests - is unchanged.
from .gpu_pool_balancer import (  # noqa: F401 - re-exported
    BALANCER_CONNECT_TIMEOUT, BALANCER_NEXT_UPSTREAM, BALANCER_OTHER_RELEASE,
    BALANCER_OTHER_RELEASE_NOTE, BALANCER_RESEND_TRIES,
    BALANCER_RESEND_WINDOW, BALANCER_UPSTREAM,
    BALANCER_ZONE_SIZE, BalancerController, BalancerProcess, _validate_upstream,
    balancer_config_digest, balancer_config_path, default_balancer,
    render_balancer_config, replica_entries, replica_upstreams, stop_balancer,
)
from .gpu_pool_units import VLLM_ENGINE, deployment_name, is_replicated
from .gpu_serving_target import (
    CLUSTER_INFERENCE_PURPOSE, SERVER_LOOPBACK_HOST, serving_port,
)
from .llm_server_state import generate_key
from .model_thinking import THINKING_FIELD
from .vllm_serve_options import OPTIONS_FIELD
from .ssh_transport import SshTransportError

LOGGER = logging.getLogger(__name__)

#: The unauthenticated readiness path every replica answers once loaded - on
#: loopback for the controller's, and through its gate for a worker's, which
#: forwards exactly this path without the key.
REPLICA_HEALTH_PATH = "/health"

#: How many consecutive watch passes may see NO replica alive before the row
#: is failed and the cluster torn down (VD-129). Four at the 30 s cadence is
#: about two minutes: long enough for every replica to be mid-restart at once
#: after a power blip, short enough that AI Chat is not left on a dead
#: endpoint for long. Below that the record stays healthy and the row reads
#: "k of N serving" from what the watch wrote.
REPLICA_DOWN_PASSES = 4

#: The progress line while replicas are alive and their APIs have not answered
#: - "N replicas compiling" on a cold cache, said with the nodes.
REPLICAS_PROGRESS = (
    "{count} vLLM replica(s) alive on {nodes}, waiting for their APIs "
    "({minutes} min; first start compiles kernels)"
)

#: Failure: the cap, not a replica, ended the wait. Same shape as the single
#: server's cap sentence, plural, naming the replicas still waiting.
REPLICAS_CAP_REACHED = (
    "The vLLM replica(s) on {nodes} were still alive after {minutes} minutes "
    "but their APIs had not answered; the {cap}-minute startup cap ended the "
    "wait, not the servers. A first start compiles kernels; raise "
    "startup_timeout_seconds to wait longer."
)

#: The row's ``units["failure"]`` when the watch saw no replica alive for
#: :data:`REPLICA_DOWN_PASSES` passes: every replica's last state, by unit and
#: node, so the note says what was found rather than that something was.
REPLICAS_DOWN = (
    "No vLLM replica answered its health check for {passes} consecutive "
    "checks: {states}."
)

#: The three words a replica's last state is written in.
REPLICA_ALIVE = "answering"
REPLICA_DOWN = "not answering"
REPLICA_UNREACHABLE = "unreachable"


@dataclass(frozen=True)
class ReplicaPlacement:
    """One replica as the deploy started it: where, how, and through what.

    ``gate`` is the gate unit on a WORKER (the VD-129 amendment) and empty
    for the controller's replica, which has none; ``address``/``port`` is
    where the replica is REACHED - the gate on a worker, loopback here.
    """

    node: Dict[str, Any]
    transport: Any
    unit: str
    address: str
    port: int
    gate: str = ""

    @property
    def node_id(self) -> str:
        return str(self.node.get("id", ""))

    @property
    def name(self) -> str:
        return str(self.node.get("name", self.node.get("id", "")))

    @property
    def units(self) -> Tuple[str, ...]:
        """The units that must be alive for this replica to answer: its server, and its gate if it has one."""
        return tuple(unit for unit in (self.unit, self.gate) if unit)

    def record(self, *, alive: bool, reachable: bool = True) -> Dict[str, Any]:
        """The ``units.replicas[]`` entry: ``{node_id, address, port, unit, alive, reachable}``, plus ``gate`` on a worker."""
        entry = {
            "node_id": self.node_id, "address": self.address,
            "port": int(self.port), "unit": self.unit, "alive": bool(alive),
            "reachable": bool(reachable),
        }
        if self.gate:
            entry["gate"] = self.gate
        return entry


def replica_health_url(address: str, port: int) -> str:
    """Where a replica's unauthenticated readiness is probed."""
    return "http://{}:{}{}".format(address, int(port), REPLICA_HEALTH_PATH)


def replica_address(node: Mapping[str, Any], advertise_address: Callable[[Any], str]) -> str:
    """Where a replica on ``node`` is reached: loopback here, its advertised address elsewhere.

    The ONE derivation of a replica's address, read by the deploy (which
    records it and bounds the balancer's pool to it) and by the refusal that
    runs before the deploying row is written, so the address that is checked
    is the address that is served.
    """
    if str(node.get("id", "")) == CONTROLLER_PLACEMENT_ID:
        return SERVER_LOOPBACK_HOST
    return advertise_address(node["host"])


def mint_replica_key() -> str:
    """The cluster's one key: minted by the deploy, recorded on the credential.

    The LLM Server's own generator, so a replica key has the strength and the
    shape (`llm_server_state.generate_key`) the proxy's key rule already
    admits; it dies on every node with the replica's unit.
    """
    return generate_key()


class ReplicaStartupWaiter:
    """Block until every replica's API answers, while every replica is alive.

    Round-robin over the replicas, no threads: each cycle probes the
    unauthenticated ``/health`` of every replica not yet answering (a
    worker's through its gate, so a gate that is not up reads as a replica
    not yet answering), reads the units of each one still silent through ITS
    OWN transport (the bridge for the controller's, SSH for a worker's; the
    worker's gate unit beside its server unit), fails the deploy at once on a
    dead unit naming it, bounds a dropped transport per node the way the pull
    pollers do, and reports the replicas still waiting. The cap is the single
    server's (`gpu_pool_startup.startup_cap_seconds`) and only ever ends a
    wait on replicas that were just seen alive.
    """

    def __init__(
        self, runtime: Any, *, probe: Callable[[str], bool],
        sleep: Callable[[float], None] = time.sleep,
        monotonic: Callable[[], float] = time.monotonic,
    ):
        self.runtime = runtime
        self._probe = probe
        self._sleep = sleep
        self._monotonic = monotonic

    def wait(
        self, replicas: Sequence[ReplicaPlacement], payload: Mapping[str, Any],
        report: Callable[[int, str], None],
    ) -> None:
        cap = startup_cap_seconds(payload.get("startup_timeout_seconds"))
        started = self._monotonic()
        pending = list(replicas)
        unreachable: Dict[str, int] = {}
        reported_minute = -1
        while True:
            still: List[ReplicaPlacement] = []
            for replica in pending:
                if self._probe(replica_health_url(replica.address, replica.port)):
                    continue
                self._check_alive(replica, unreachable)
                still.append(replica)
            pending = still
            if not pending:
                return
            elapsed = self._monotonic() - started
            nodes = ", ".join(replica.name for replica in pending)
            if elapsed >= cap:
                raise RuntimeError(REPLICAS_CAP_REACHED.format(
                    nodes=nodes, minutes=int(elapsed // 60), cap=cap // 60,
                ))
            minute = int(elapsed // STARTUP_REPORT_SECONDS)
            if minute > reported_minute:
                reported_minute = minute
                report(STARTUP_CHECKPOINT_PERCENT, REPLICAS_PROGRESS.format(
                    count=len(pending), nodes=nodes, minutes=minute,
                ))
            self._sleep(API_POLL_SECONDS)

    def _check_alive(
        self, replica: ReplicaPlacement, unreachable: Dict[str, int]
    ) -> None:
        """One liveness read of a silent replica's units, through its own transport.

        Its server unit and, on a worker, its gate unit: the probe goes
        through the gate, so a dead gate is as silent as a dead server and
        the deploy must name whichever it was rather than ride to the cap.
        """
        for unit in replica.units:
            try:
                state = self.runtime.unit_state(replica.transport, unit)
            except SshTransportError as error:
                unreachable[replica.node_id] = unreachable.get(replica.node_id, 0) + 1
                if unreachable[replica.node_id] >= PULL_TRANSPORT_FAILURES:
                    raise RuntimeError(
                        NODE_UNREACHABLE.format(node=replica.name, error=error)
                    ) from error
                return
            unreachable[replica.node_id] = 0
            if not unit_alive(state):
                # With vLLM's own reason from the unit's journal, when it has one.
                raise RuntimeError(server_died(
                    self.runtime, replica.transport, unit, replica.name, state,
                ))


class ReplicaHealth:
    """The mode watch's reading of a healthy replicated row's replicas.

    Probes each replica's ``/health`` over HTTP - no transport, because a
    node that left the cluster cannot be reached for one and the answer is
    the same as for a replica that is down - and answers the row's new
    ``units.replicas`` with ``alive`` written, the count, and, when no replica
    has answered for :data:`REPLICA_DOWN_PASSES` consecutive passes, the
    failure sentence naming every replica's last state. The pass counter is
    this process's: a watch that restarts counts afresh, which delays a
    teardown by at most the passes it lost and never invents one.
    """

    def __init__(self, probe: Callable[[str], bool], *, passes: int = REPLICA_DOWN_PASSES):
        self._probe = probe
        self._passes = max(1, int(passes))
        self._silent: Dict[str, int] = {}

    def observe(self, record: Mapping[str, Any]) -> Optional[Dict[str, Any]]:
        """``{replicas, alive, total, failure}`` for a healthy replicated row, else ``None``."""
        if str(record.get("state", "")) != "healthy" or not is_replicated(record):
            return None
        entries = replica_entries(record)
        if not entries:
            return None
        name = str(record.get("name", ""))
        for entry in entries:
            answer = self._probe(
                replica_health_url(entry.get("address", ""), int(entry.get("port", 0) or 0))
            )
            # Two facts from one probe: whether anything answered at the
            # address at all, and whether what answered was healthy. A node
            # that left the cluster is the first false; a replica mid-restart
            # on a node that is there is only the second.
            entry["reachable"] = answer is not UNREACHABLE
            entry["alive"] = entry["reachable"] and bool(answer)
        alive = sum(1 for entry in entries if entry["alive"])
        failure = ""
        if alive:
            self._silent.pop(name, None)
        else:
            self._silent[name] = self._silent.get(name, 0) + 1
            if self._silent[name] >= self._passes:
                self._silent.pop(name, None)
                failure = REPLICAS_DOWN.format(
                    passes=self._passes, states=describe_replicas(entries),
                )
        return {
            "replicas": entries, "alive": alive, "total": len(entries),
            "failure": failure,
        }


def replica_state_word(entry: Mapping[str, Any]) -> str:
    """One of the three words for a replica's last state, from its two flags.

    A row written before ``reachable`` existed carries only ``alive``, and is
    read as reachable: "not answering" is what its probe established.
    """
    if entry.get("alive"):
        return REPLICA_ALIVE
    if entry.get("reachable", True) is False:
        return REPLICA_UNREACHABLE
    return REPLICA_DOWN


def describe_replicas(entries: Sequence[Mapping[str, Any]]) -> str:
    """``unit on node (answering|not answering|unreachable); ...`` - every replica's last state."""
    return "; ".join(
        "{} on {} ({})".format(
            entry.get("unit", ""), entry.get("node_id", ""), replica_state_word(entry),
        )
        for entry in entries
    )


class ReplicatedServing:
    """The deploy's replicated path, after `GpuPoolOperations.deploy`'s shared prefix.

    Handed the participants (controller first, as `_lead_first` orders them),
    their transports and GIDs, the pulled repo and the launch settings; starts
    one `start_single_server` per node with the per-node compile cache - on
    loopback, always - and on each worker a `start_gate` beside it carrying
    the cluster key (the VD-129 amendment), waits for every replica, starts
    the balancer, and answers the healthy record's ``units``, the endpoint
    and the key the credential is registered with. Every unit it starts is
    appended to ``started`` as it goes, gates included, so the caller's
    rollback stops exactly what exists.
    """

    def __init__(
        self, runtime: Any, balancer: BalancerController, *,
        advertise_address: Callable[[Any], str], waiter: ReplicaStartupWaiter,
    ):
        self.runtime = runtime
        self.balancer = balancer
        self._advertise_address = advertise_address
        self._waiter = waiter

    def refuse_unroutable(
        self, participants: Sequence[Mapping[str, Any]], replica_port: int,
    ) -> None:
        """Refuse, by node, any replica address the balancer could not pool.

        The balancer's pool admits only loopback and private IPv4
        (`_validate_upstream`, applied again root-side when the config is
        rendered); a worker whose cluster address is outside that - a
        carrier-grade NAT range, a public address, a hostname - would
        otherwise be refused only at ``balancer.start``, after both replicas
        had pulled and loaded. Asked BEFORE the deploying row is written and
        before `enter` stops Mode A, with the same sentence, prefixed by the
        machine it is about; the late check stays as the boundary's own.
        """
        for node in participants:
            address = replica_address(node, self._advertise_address)
            try:
                _validate_upstream("{}:{}".format(address, int(replica_port)))
            except ValueError as error:
                label = node.get("name") or node.get("id")
                raise ValueError(f"{label}: {error}") from error

    def serve(
        self, *, name: str, participants: Sequence[Dict[str, Any]],
        transports: Mapping[str, Any], group_ids: Mapping[str, List[int]],
        repo: str, revision: Optional[str], port: int, replica_port: int,
        gpu_memory_utilization: float, max_model_len: int,
        payload: Mapping[str, Any], report: Callable[[int, str], None],
        started: List[Any], thinking_default: bool = False,
        vllm_options: Any = None,
    ) -> Tuple[Dict[str, Any], str, str]:
        api_key = mint_replica_key()
        replicas: List[ReplicaPlacement] = []
        for index, node in enumerate(participants):
            transport = transports[node["id"]]
            address = replica_address(node, self._advertise_address)
            report(
                60 + int(20 * index / len(participants)),
                f"Starting the vLLM replica on {node['name']}",
            )
            # Loopback and keyless on every node: the runtime offers no other
            # bind. What makes a worker's replica reachable - and reachable
            # ONLY with the key - is the gate started beside it.
            unit = self.runtime.start_single_server(
                transport, name=name, model=repo, port=replica_port,
                tensor_parallel_size=gpu_fit_node(node)["device_count"],
                gpu_memory_utilization=gpu_memory_utilization,
                max_model_len=max_model_len, group_ids=group_ids[node["id"]],
                revision=revision, thinking_default=thinking_default,
                vllm_options=vllm_options,
            )
            started.append((transport, node, unit))
            gate = ""
            if address != SERVER_LOOPBACK_HOST:
                gate = self.runtime.start_gate(
                    transport, name=name, listen_host=address,
                    port=replica_port, api_key=api_key,
                )
                started.append((transport, node, gate))
            replicas.append(ReplicaPlacement(
                node=node, transport=transport, unit=unit, address=address,
                port=replica_port, gate=gate,
            ))
        self._waiter.wait(replicas, payload, report)
        units = {
            "engine": VLLM_ENGINE,
            "mode": VERDICT_REPLICATED,
            "lan_exposed": False,
            "api_host": SERVER_LOOPBACK_HOST,
            "replicas": [replica.record(alive=True) for replica in replicas],
            "port": int(port),
            "replica_port": int(replica_port),
            "gpu_memory_utilization": float(gpu_memory_utilization),
            "max_model_len": int(max_model_len),
            "repo": repo,
            "revision": revision,
            # What the replicas were launched with, so a Load renders the same.
            THINKING_FIELD: bool(thinking_default),
            OPTIONS_FIELD: vllm_options,
        }
        report(85, "Starting the replica balancer on this controller")
        self.balancer.start({"name": name, "units": units}, api_key)
        endpoint = "http://{}:{}/v1".format(SERVER_LOOPBACK_HOST, int(port))
        return units, endpoint, api_key


def replica_ports(port: Any) -> Tuple[int, int]:
    """``(balancer port, replica port)`` from the requested port, both in the band.

    The balancer takes the requested port (default 8000) and every replica
    listens on the next one, and both go through the one band rule
    (`gpu_serving_target.serving_port`) BEFORE anything is written - so 8079
    is refused, because its replicas would land on Mode A's allocator band.
    """
    balancer = serving_port(port)
    return balancer, serving_port(balancer + 1)


#: The Endpoints surface's id and label for the internal cluster serving key
#: (design section 5). The key is DISPLAYED and ROTATED IN PLACE here - it gates
#: the internal balancer->worker hop and is NEVER presented by a client, so the
#: surface marks it internal and never reveals it.
CLUSTER_SERVING_ENDPOINT_ID = "cluster-serving"
CLUSTER_SERVING_LABEL = "Cluster serving"

#: The confirm token the rotate route stamps and the operation checks, in one
#: home so the two cannot drift (the pattern `cluster_job_confirmations` uses).
ROTATE_CLUSTER_KEY_CONFIRMATION = "rotate-cluster-key"

#: What a rotate is refused with when the deployment is distributed/single-head:
#: its loopback head carries no cluster key, so there is nothing to rotate. The
#: surface shows that state honestly rather than offering a rotate that no-ops.
ROTATE_NOT_REPLICATED = (
    "A distributed (single-head) deployment has no internal cluster key to "
    "rotate; only a replicated deployment gates its worker replicas with one."
)


#: Which cluster row the serving card stands for when several exist: a
#: serving one first (replicated before distributed, the keyed endpoint being
#: the one the surface can rotate), then one loading, one paused, one failed.
_CARD_PRECEDENCE = ("healthy", "deploying", "unloaded", "failed")


def _active_cluster_deployment(
    deployments: Sequence[Mapping[str, Any]]
) -> Optional[Mapping[str, Any]]:
    """The one vLLM deployment the cluster-serving endpoint stands for.

    ACC-055: not only a healthy one. A row that stopped serving used to take
    the card with it - the endpoint simply vanished - so the owner was never
    told; the card now stays and says what the row's state and health are
    (:func:`vaelor.gpu_pool_serving_health.serving_reading`). ``None`` only when
    there is no vLLM deployment at all.
    """
    best: Optional[Mapping[str, Any]] = None
    best_rank = None
    for record in deployments or []:
        units = record.get("units") or {}
        if units.get("engine") != VLLM_ENGINE:
            continue
        state = str(record.get("state", ""))
        if state not in _CARD_PRECEDENCE:
            continue
        rank = (_CARD_PRECEDENCE.index(state), 0 if is_replicated(record) else 1)
        if best_rank is None or rank < best_rank:
            best, best_rank = record, rank
    return best


def cluster_serving_view(
    deployments: Sequence[Mapping[str, Any]],
    credentials: Sequence[Mapping[str, Any]],
    unload_cause_of: Optional[Callable[[str], str]] = None,
) -> Optional[Dict[str, Any]]:
    """The read model for the internal cluster serving key (design section 5, B2).

    Built from the deployment rows and the broker's fingerprint-only ``list()``,
    both control-plane readable. For a REPLICATED (Mode-B) deployment it reports
    ``key_present`` with the cluster-inference credential's FINGERPRINT and marks
    the endpoint ``internal`` (a client never presents this key - it gates the
    balancer->worker hop). For a DISTRIBUTED/single-head deployment it reports
    ``key_present`` false honestly - never a fabricated key. ``state`` is the
    row's own and ``serving`` the watch's measured reading of it (ACC-055);
    ``None`` only when there is no cluster deployment at all.

    The base URL is the deployment's own loopback endpoint, which is honest for
    an internal key: the balancer answers there and the LLM Server proxy is the
    only LAN door in front of it. ``unload_cause_of(name)`` answers an unloaded
    row's cause (`gpu_serving_target.deployment_unload_cause`), so the paused
    sentence says who unloaded it (review S5); asked only for such a row.
    """
    from .gpu_pool_serving_health import serving_reading

    active = _active_cluster_deployment(deployments)
    if active is None:
        return None
    cause = ""
    if unload_cause_of is not None and str(active.get("state", "")) == "unloaded":
        cause = str(unload_cause_of(str(active.get("name", "") or "")) or "")
    replicated = is_replicated(active)
    credential_id = str(active.get("credential_id", "") or "")
    fingerprint = ""
    if replicated and credential_id:
        for credential in credentials or []:
            if str(credential.get("id", "")) == credential_id:
                fingerprint = str(credential.get("fingerprint", "") or "")
                break
    units = active.get("units") or {}
    return {
        "id": CLUSTER_SERVING_ENDPOINT_ID,
        "label": CLUSTER_SERVING_LABEL,
        "name": str(active.get("name", "") or ""),
        "model": str(units.get("repo", "") or active.get("model_id", "") or ""),
        "base_url": str(active.get("endpoint", "") or ""),
        "internal": True,
        "replicated": replicated,
        "key_present": bool(replicated),
        "key_fingerprint": fingerprint,
        "state": str(active.get("state", "") or ""),
        "serving": serving_reading(active, cause),
        # The release skew the watch wrote (S1), said where the balancer is.
        "balancer_note": (
            BALANCER_OTHER_RELEASE_NOTE
            if replicated and units.get(BALANCER_OTHER_RELEASE) else ""
        ),
    }


def _reput_cluster_key(
    broker: Any, credential_id: str, lease: Mapping[str, Any], new_key: str
) -> Dict[str, Any]:
    """Re-encrypt the SAME cluster-inference credential with a new ``api_key``.

    The key is rotated IN PLACE (design B2): the credential keeps its id and its
    ``cluster-inference`` assignment, so `repoint`/`_converge_balancer` read the
    new key back through the unchanged ``resolve(cred_id, CLUSTER_INFERENCE_PURPOSE)``
    path. Only the profile's ``api_key`` changes; the base URL and model are
    preserved from the lease. A ``put`` with the existing id bumps the version
    and re-encrypts, exactly as an outbound-credential rotation does.
    """
    profile = json.dumps(
        {
            "base_url": str(lease.get("base_url", "") or ""),
            "model": str(lease.get("model", "") or ""),
            "api_key": str(new_key),
        },
        separators=(",", ":"),
    )
    return broker.put(
        "openai-compatible", str(lease.get("label", "") or ""), profile, credential_id
    )


def _rekey_worker_gates(
    runtime: Any, joined_node: Callable[..., Mapping[str, Any]],
    transport: Callable[[Mapping[str, Any]], Any], name: str, replica_port: int,
    gates: Sequence[Mapping[str, Any]], api_keys: Sequence[str],
) -> None:
    """Re-render every worker gate to accept exactly ``api_keys`` and restart it.

    Each worker's gate is the LAN-facing door in front of its loopback replica
    (VD-129); re-rendering it with a key SET is the byte-exact multi-key gate
    (F3b-i). The controller's own replica has no gate (it sits behind the
    balancer on loopback), so only worker entries are re-keyed.
    """
    for entry in gates:
        node = joined_node(str(entry.get("node_id", "")), include_credential=True)
        runtime.rekey_gate(
            transport(node), name=name, listen_host=str(entry.get("address", "")),
            port=int(entry.get("port", replica_port) or replica_port),
            api_keys=list(api_keys),
        )


def rotate_cluster_serving_key(
    payload: Mapping[str, Any], *, store: Any, broker: Any, runtime: Any,
    balancer: BalancerController, joined_node: Callable[..., Mapping[str, Any]],
    transport: Callable[[Mapping[str, Any]], Any],
    serving_lock: Callable[[], Any],
    progress: Optional[Callable[[int, str], None]] = None,
) -> Dict[str, Any]:
    """Rotate the internal cluster key with an OVERLAP-TOLERANT ORDERED re-key.

    Cross-host re-keying is not atomic (design S1), so the order is load-bearing
    and each step leaves the deployment still serving:

    a. mint a new ``vsk_`` key;
    b. re-render EVERY worker gate to accept BOTH ``{old, new}`` - now the
       balancer may stamp either key and no gate rejects it;
    c. under the serving lock (serialised with the mode reconcile's balancer
       convergence), re-key the cluster-inference credential to ``new`` and
       restart the balancer so it STAMPS ``new``;
    d. re-render every worker gate to ``{new}`` only, dropping ``old``.

    At no point may the balancer stamp a key the gates reject: the gates accept
    ``{old, new}`` before the balancer is restarted (b before c), and the
    balancer stamps ``new`` before any gate drops ``old`` (c before d). Any
    failure mid-rotation propagates with the gates still accepting ``{old, new}``
    (steps c and d never run out of order), which fails toward still-serving
    rather than leaving a half-keyed deployment. No plaintext is returned: this
    key is internal and is never revealed to a client.

    A failure WITHIN step c (the credential re-put succeeded but the balancer
    restart did not, with the old balancer surviving) leaves the credential's
    fingerprint ahead of the key the balancer still stamps. Serving continues
    (the gates accept ``{old, new}``) and the job surfaces the error, but the
    reconcile's ``converged`` check does not inspect the key, so it will not
    self-heal this drift - re-running the rotate reconciles it. The Mode-B live
    test exercises the kill-balancer-mid-rotate recovery.
    """
    if (payload or {}).get("confirm") != ROTATE_CLUSTER_KEY_CONFIRMATION:
        raise ValueError("Confirm the cluster serving key rotation.")
    name = deployment_name(str((payload or {}).get("name", "")))
    deployment = store.get_pooled_deployment(name)
    if deployment is None:
        raise ValueError("No GPU deployment by that name was found to rotate.")
    units = deployment.get("units") or {}
    if units.get("engine") != VLLM_ENGINE:
        raise ValueError(
            "That deployment is not a vLLM GPU serving deployment, so it has no "
            "cluster key."
        )
    if str(deployment.get("state", "")) != "healthy":
        raise ValueError("Only a healthy deployment's cluster key can be rotated.")
    if not is_replicated(deployment):
        raise ValueError(ROTATE_NOT_REPLICATED)
    credential_id = str(deployment.get("credential_id", "") or "")
    if not credential_id:
        raise ValueError("This deployment has no cluster credential to rotate.")
    report = progress or (lambda _percent, _message: None)
    lease = broker.resolve(credential_id, CLUSTER_INFERENCE_PURPOSE)
    old_key = str(lease.get("api_key", "") or "")
    if not old_key:
        raise ValueError(
            "This deployment's cluster credential carries no key to rotate."
        )
    replica_port = int(units.get("replica_port", 0) or 0)
    gates = [
        entry
        for entry in replica_entries(deployment)
        if str(entry.get("gate", "") or "")
        and str(entry.get("address", "")) != SERVER_LOOPBACK_HOST
    ]
    new_key = mint_replica_key()
    report(20, "Re-keying every worker gate to accept both the old and new key")
    _rekey_worker_gates(
        runtime, joined_node, transport, name, replica_port, gates,
        [old_key, new_key],
    )
    report(60, "Restarting the balancer on the new cluster key")
    with serving_lock():
        rekeyed = _reput_cluster_key(broker, credential_id, lease, new_key)
        balancer.start(deployment, new_key)
    report(85, "Retiring the old key from every worker gate")
    _rekey_worker_gates(
        runtime, joined_node, transport, name, replica_port, gates, [new_key]
    )
    report(100, "The cluster serving key was rotated")
    return {
        "name": name,
        "rotated": True,
        "endpoint_id": CLUSTER_SERVING_ENDPOINT_ID,
        "credential_id": credential_id,
        "key_fingerprint": str(rekeyed.get("fingerprint", "") or ""),
        "gates_rekeyed": len(gates),
    }
