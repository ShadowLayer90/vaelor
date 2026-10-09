"""Secure head-controller orchestration for enrolled cluster workers.

This line used to say "enrolled Raspberry Pi workers", which was the whole of
the arm64 assumption stated outright. Workers are admitted on what the
controller observes about them, and the one thing they must share with it is
its processor architecture (VD-031) — not its board and not its OS.
"""

from __future__ import annotations

import ipaddress
import json
import logging
import socket
import time
from typing import Any, Dict, Optional

from .cluster_architecture import (
    controller_architecture,
    fleet_architecture,
    join_compatibility,
    node_architecture_fact,
)
from .cluster_capacity import (
    compute_capacity_ledger,
    controller_node_facts,
    reservation_from_service_details,
    worker_placement_state,
)
from .cluster_driver import ClusterDriverError, DockerSwarmDriver
from .cluster_service_state import (
    aggregate_app_groups,
    annotate_service_states,
    unschedulable_node_ids,
)
from .cluster_node_removal import (
    forget_ingest_status,
    failure_words,
    refuse_if_in_use,
    take_profile_off_safely,
    take_telemetry_off,
)
from .cluster_store import NODE_NOT_FOUND, ClusterStore
from .cluster_worker_profile import WorkerProfileMixin
from .cluster_worker_telemetry import (
    WorkerTelemetryMixin,
    _database_last_sample_time,
    _default_ingest_status,
    reconcile_sentence,
    repair_note,
)
from .credential_broker import CredentialBrokerClient
from .credential_broker_client import CredentialError
from .gpu_node_facts import local_cluster_interface
from .gpu_render_ledger import NEXT_STEP, refresh_note
from .gpu_pool_recover import load_offered
from .gpu_pool_serving_health import split_checking_note
from .model_thinking import has_thinking_switch, thinking_on_record
from .vllm_serve_options import image_on_record
from .ssh_transport import SshTransport, SshTransportError, host_key_fingerprint
from .workload_broker import WorkloadBrokerClient


LOGGER = logging.getLogger(__name__)


#: How fresh a selected worker's stored inventory must be for a serve-form
#: fit to trust it without re-probing. The fit is re-issued on debounced
#: form changes, so a window this size (seconds) keeps a burst of edits from
#: re-probing the same node on every keystroke while still catching a GPU
#: freed since the join-time snapshot the ledger otherwise reads (Defect C).
REFIT_FRESHNESS = 25


def split_fence_notes(units: Dict[str, Any], state: str = "") -> Dict[str, str]:
    """A split's two owner-visible fence notes (ACC-187 reviews 3-4), ``""`` when none.

    ``cleanup_note``: an unload stopped every unit but could not clear a
    machine's firewall and slice; Remove retries it, Load fences again. Said
    only on an ``unloaded`` row - the state it describes.
    ``fence_note``: a split deployed before its units re-checked the firewall
    at every start, said only on a serving or unloaded row, with the step that
    applies the check from that state.
    """
    left = [
        entry for entry in units.get("plane_cleanup") or []
        if isinstance(entry, dict) and state == "unloaded"
    ]
    cleanup = ""
    if left:
        cleanup = (
            "The model is unloaded, but the split's firewall and slice could not be "
            "cleared on {}. Load it again, or Remove it to retry the clean-up."
        ).format(", ".join(str(entry.get("node") or entry.get("node_id") or "a machine")
                           for entry in left))
    fence = ""
    if units.get("mode") == "distributed" and not units.get("slice_check"):
        step = NEXT_STEP.get(state, "")
        if step:
            fence = ("This split was deployed before Vaelor checked its firewall at every "
                     "start. {} to apply that check.".format(step))
    return {"cleanup_note": cleanup, "fence_note": fence}


def projected_pooled_deployments(
    deployments: list[Dict[str, Any]], unload_cause: Any = None,
) -> list[Dict[str, Any]]:
    """Surface each deployment's serving ``engine`` in the fleet summary row.

    The GPU (vLLM) and CPU (distributed-llama) tiers share the
    ``pooled_deployments`` store table, discriminated by an ``engine`` field
    inside each row's ``units`` blob — but the fleet summary a client reads only
    carried ``{name, state, model_id, node_ids, endpoint}``, so the console could
    not tell a vLLM deployment (removed through `cluster.gpu.remove`) from a CPU
    pooled one (removed through `cluster.pooled.remove`) and pick the right path.
    A row is ``vllm`` only when its stored units say so; every other row is the
    CPU ``distributed-llama`` engine, which is exactly the discriminator
    `pooled_operations.remove` and `gpu_pool_operations.remove` already use.

    An ``unloaded`` vLLM row also carries ``unload_cause`` - ``unloaded-idle``
    (the next request wakes it), ``unloaded-manual`` (only a Load does), or
    ``""`` (not known) - from ``unload_cause(name)``, which the summary binds to
    :func:`vaelor.gpu_serving_target.deployment_unload_cause`. It is asked only
    for such a row, so an ordinary summary makes no broker read.
    """
    projected = []
    for deployment in deployments:
        engine = deployment.get("units", {}).get("engine")
        row = {
            **deployment,
            "engine": "vllm" if engine == "vllm" else "distributed-llama",
            # G3b: the scale-to-zero window (seconds; 0 = off), surfaced from
            # the units blob so the console can label an auto-unloading model.
            "idle_timeout": int(deployment.get("units", {}).get("idle_timeout", 0) or 0),
            # Whether the model thinks before answering (owner decision
            # 2026-09-29): the setting the record carries, or None for a record
            # from before it existed - that deployment still thinks until its
            # next Load renders the new default. `thinking_switch` says whether
            # the setting reaches this model at all (`model_thinking`).
            "thinking": thinking_on_record(deployment.get("units", {})),
            "thinking_switch": has_thinking_switch(
                deployment.get("units", {}).get("repo") or deployment.get("model_id")
            ),
            # Which vLLM a cluster deployment runs ("" for the CPU engine):
            # the image its record names, or the one every deployment ran
            # before an image was recorded (`vllm_images`).
            "vllm_version": (
                image_on_record(deployment.get("units", {})).name
                if engine == "vllm" else ""
            ),
            # A machine this deployment lost to a forced removal, in plain words
            # ("" when none): the console shows the row degraded (B1).
            "degraded_reason": str(
                deployment.get("units", {}).get("degraded_reason", "") or ""
            ),
            **split_fence_notes(deployment.get("units", {}) or {},
                                str(deployment.get("state", "") or "")),
            # A serving split failing its checks, said from the first failed
            # one with the watch's window ("" otherwise, W4-D6).
            "checking_note": split_checking_note(
                deployment.get("units", {}) or {}, str(deployment.get("state", "") or "")),
            # Whether Load serves this row: paused, or failed after serving
            # (W4-D7) - the console offers Load from this, never on its own.
            "load_offered": load_offered(deployment),
            # What a post-upgrade refresh could not do, worded for the row's
            # state ("" when none, W4-D1).
            "refresh_note": refresh_note(deployment.get("units", {}) or {},
                                         str(deployment.get("state", "") or "")),
        }
        if engine == "vllm" and deployment.get("state") == "unloaded" and unload_cause:
            row["unload_cause"] = str(unload_cause(str(deployment.get("name", ""))) or "")
        projected.append(row)
    return projected


def enrollment_readiness(
    runtime: Dict[str, Any], architecture: str = ""
) -> Dict[str, Any]:
    """Whether a worker can be enrolled right now, and why not.

    The Add-worker form was enabled — and asked for a sudo password — on a page
    that said in its own text that the operation could not succeed. Collecting
    an administrator's sudo credential for an action known in advance to fail
    is the worst-shaped thing in this area: the credential is the most
    sensitive input the product asks for anywhere, and it was being taken for
    nothing.

    So readiness is published as a fact the form can be disabled on, rather
    than left for prose to describe and the button to ignore.

    ``architecture`` is this controller's *discovered* architecture class. A
    controller that cannot read its own silicon cannot tell whether a worker
    matches it, and under VD-031 that is a refusal rather than a guess — same
    rule as the rest: nothing that cannot succeed asks for the password.
    """
    facts = dict(runtime or {})
    if not facts.get("available"):
        # #141: this used to say "the container runtime is not reachable on
        # this appliance" for every way the status read could fail — a claim
        # about the machine, made while five Vaelor-managed containers ran on
        # that same runtime. All this branch actually knows is what the
        # engine facts say: the binary is absent, or one read did not answer.
        return {
            "available": False,
            "reason": (
                "Docker is not installed on this appliance, so it cannot "
                "enrol a worker."
                if str(facts.get("engine", "")) == "absent" else
                "Vaelor could not read this appliance's cluster state, so it "
                "cannot enrol a worker."
            ),
            "requires_sudo_password": False,
        }
    if not facts.get("initialized"):
        return {
            "available": False,
            "reason": (
                "This appliance is not a cluster controller yet. Initialise "
                "the cluster here before enrolling a worker into it."
            ),
            "requires_sudo_password": False,
        }
    if not facts.get("control_available"):
        return {
            "available": False,
            "reason": (
                "This node is part of a cluster but is not its controller, so "
                "it cannot enrol workers. Run this from the controller."
            ),
            "requires_sudo_password": False,
        }
    if not str(architecture or "").strip():
        return {
            "available": False,
            "reason": (
                "This controller could not determine its own processor "
                "architecture, so it cannot confirm that a worker matches "
                "it. A Vaelor cluster keeps one architecture."
            ),
            "requires_sudo_password": False,
        }
    return {
        "available": True,
        "reason": "",
        # Stated rather than implied by a form field appearing. Enrolling runs
        # privileged commands on the *worker*, which is why it is asked for.
        "requires_sudo_password": True,
    }


class ClusterManager(WorkerTelemetryMixin, WorkerProfileMixin):
    def __init__(
        self,
        store: Optional[ClusterStore] = None,
        broker: Optional[CredentialBrokerClient] = None,
        driver: Optional[DockerSwarmDriver] = None,
        transport_factory=SshTransport,
        inventory_probe=None,
        gpu_chat_reclaimable=None,
        last_sample_time=None,
        ingest_status=None,
    ):
        self.store = store or ClusterStore()
        self.broker = broker or CredentialBrokerClient(timeout_seconds=30)
        # Status reads go through the workload broker (#141): this manager
        # lives in the control plane, which deliberately holds no docker
        # group. Querying the socket directly from here answered "permission
        # denied", and the screen translated that into "the container runtime
        # is not reachable on this appliance" while five brokered containers
        # ran beside it. The executor's ClusterOperations keeps a direct
        # driver — that process holds the privilege itself.
        self.driver = driver or DockerSwarmDriver(
            runner=WorkloadBrokerClient().run
        )
        self.transport_factory = transport_factory
        if inventory_probe is None:
            from .copilot_setup import hardware_inventory

            inventory_probe = hardware_inventory
        self.inventory_probe = inventory_probe
        # VD-125 (D1): whether the controller's held GPU memory is the AI-Chat
        # model the cluster mode switch stops, so the `/cluster/fit` preview
        # sizes against the memory the deploy WILL free. Injectable, and the
        # default is the same call `cluster_operations` makes for the deploy -
        # one engine, so the two cannot report different room.
        if gpu_chat_reclaimable is None:
            from .gpu_serving_target import controller_gpu_chat_reclaimable

            gpu_chat_reclaimable = controller_gpu_chat_reclaimable
        self.gpu_chat_reclaimable = gpu_chat_reclaimable
        # The controller-side liveness accessor reconcile uses to tell an
        # active-but-not-reporting agent from a healthy one. Injectable for
        # tests; the default reads this process's telemetry retention source.
        self.last_sample_time = last_sample_time or _database_last_sample_time
        # ACC-126: what the ingest route last did with this node's posts
        # (accepted, or refused for its clock), read by the reconcile so a
        # skewed clock is reported instead of reinstalled every 15 minutes.
        self.ingest_status = ingest_status or _default_ingest_status
        # Defect C: monotonic seconds of the last pre-fit re-probe ATTEMPT
        # per node (success or failure), so a persistently unreachable
        # selected worker is probed at most once per freshness window rather
        # than on every debounced serve-form fit. In-memory; resets on
        # control-plane restart.
        self._fit_probe_attempts: Dict[str, float] = {}

    def controller_architecture(self) -> str:
        """This appliance's architecture class, read from its own hardware.

        Probed rather than declared, and deliberately not derived from the
        machine class: the workstation driver is chosen by SMBIOS presence,
        which says nothing about silicon. A probe that fails answers ``""``,
        and every caller treats that as "cannot confirm", never as "matches".
        """
        try:
            return controller_architecture(self.inventory_probe())
        except (AttributeError, OSError, TypeError, ValueError):
            return ""

    @staticmethod
    def _private_host(host: str, port: int) -> str:
        clean = str(host).strip()
        if not clean or len(clean) > 253 or not 1 <= int(port) <= 65535:
            raise ValueError("Enter a valid private-LAN SSH address.")
        try:
            addresses = {
                ipaddress.ip_address(item[4][0])
                for item in socket.getaddrinfo(clean, int(port), type=socket.SOCK_STREAM)
            }
        except (OSError, ValueError) as error:
            raise ValueError("The SSH address could not be resolved.") from error
        if not addresses or any(
            not address.is_private
            or address.is_loopback
            or address.is_link_local
            or address.is_multicast
            or address.is_unspecified
            for address in addresses
        ):
            raise ValueError("Cluster workers must use a private, non-loopback LAN address.")
        return clean

    @staticmethod
    def address_hint(controller_address: str) -> str:
        """A worker-address hint derived from this controller's own address.

        Never a literal. The Add-worker form shipped pre-populated with
        a literal private address - a real machine from someone else's
        network, on an appliance whose own subnet was different - so the one
        concrete thing the form suggested was wrong, and it looked like a
        remembered value rather than an example.

        Returns "" when this controller has no usable address of its own,
        because a hint that is not derived from anything is the defect being
        fixed.
        """
        import ipaddress

        try:
            address = ipaddress.ip_address(str(controller_address or "").strip())
        except ValueError:
            return ""
        if address.version != 4 or address.is_loopback:
            return ""
        octets = str(address).split(".")
        return "{}.{}.{}.".format(*octets[:3])

    def _unload_cause(self, deployment_name: str) -> str:
        """Why a paused vLLM deployment is paused (see `deployment_unload_cause`)."""
        from .gpu_cluster_mode_state import ClusterModeStore
        from .gpu_serving_target import deployment_unload_cause

        try:
            mode_state = ClusterModeStore().read()
        except OSError as error:
            LOGGER.warning("The cluster mode record could not be read: %s", error)
            return ""
        return deployment_unload_cause(self.broker, mode_state, deployment_name)

    def summary(self, controller_address: str = "") -> Dict[str, Any]:
        runtime = self.driver.status()
        # D3: give each app row an HONEST backend state derived from Swarm's real
        # update + placement facts, so the Deployments view renders it through
        # the shared status tones instead of a replica-string regex that called a
        # silently rolled-back or data-unavailable app "Running". ONE bulk
        # `service inspect` over the app services the status list already named
        # (not one call per row); the derivation is the pure `cluster_service_
        # state`. Best-effort — a failed inspect leaves rows without a state and
        # the frontend shows an honest unknown rather than a guess.
        services = runtime.get("services", [])
        if isinstance(services, list):
            app_names = [
                str(item.get("name", ""))
                for item in services
                if isinstance(item, dict)
                and str(item.get("name", "")).startswith("vaelor-app-")
            ]
            if app_names:
                # A stateful pin reads `unavailable` only on POSITIVE evidence
                # its machine is unschedulable (Down or drained) — under-
                # replication alone is a restarting container, not a lost data
                # node. The proof comes from the enrolled records tied to the
                # live `docker node ls` rows this summary already read.
                unschedulable = unschedulable_node_ids(
                    self.store.list_nodes_tolerant(), runtime.get("nodes", [])
                )
                inspected = self.driver.inspect_services(app_names)
                runtime["services"] = annotate_service_states(
                    services, inspected, unschedulable
                )
                # D4c (B8): ADD an app-level, label-grouped view of the researched
                # multi-service apps alongside — never in place of — the per-service
                # rows above. The existing `services` list is untouched, so the
                # current frontend keeps rendering; D4d renders this grouped view.
                # Each app row folds to its WORST member and never reads healthy
                # while a member is down or its inspect is unreadable.
                app_groups = aggregate_app_groups(
                    services, inspected, unschedulable
                )
                if app_groups:
                    runtime["app_groups"] = app_groups
        controller = self.store.controller()
        if runtime.get("initialized"):
            controller = self.store.set_controller({
                "initialized": True,
                "cluster_id": runtime.get("node_id", ""),
                "advertise_address": controller.get("advertise_address", ""),
            })
        # After `set_controller`, which returns a fresh record: assigning this
        # earlier meant an initialised cluster silently lost it.
        if controller_address:
            controller["candidate_address"] = controller_address
        controller.update(self.advertise_address_health(controller))
        # VD-125: the controller's OWN cluster link, read live off this machine
        # the same way `cluster_operations._controller_cluster_interface` reads
        # it for a deploy - through the one shared helper, so the display path
        # and the deploy path cannot disagree about this machine's NIC. Landed
        # in `inventory.cluster_interface`, the exact shape/location the frontend
        # reads a worker's from, so the serve form and fleet card render it
        # identically. GUARD: a not-yet-initialised controller (no advertise or
        # candidate address) is left WITHOUT a link rather than fed a guess - the
        # frontend already shows an honest "unknown" for that.
        address = str(
            controller.get("advertise_address")
            or controller.get("candidate_address")
            or controller_address
            or ""
        ).strip()
        if address:
            inventory = dict(controller.get("inventory") or {})
            inventory["cluster_interface"] = local_cluster_interface(address)
            controller["inventory"] = inventory
        # PH-R1: one corrupt row degrades that machine only, never the Fleet view.
        enrolled = self.store.list_nodes_tolerant()
        architecture = self.controller_architecture()
        runtime_nodes = {
            str(item.get("id", "")): item for item in runtime.get("nodes", [])
        }
        # Whether each node has been provisioned with a telemetry ingest key
        # (E2b). A node has a stored hash only between install (which mints one)
        # and remove (which clears it), so this is the definitive install-state
        # the Fleet card reads to offer Install vs Remove telemetry — distinct
        # from whether the agent is currently REPORTING, which the metrics panel
        # shows from sample freshness. Only the boolean is published; the hash
        # never leaves the store.
        telemetry_provisioned = {
            node_id for node_id, _hash in self.store.ingest_key_hashes()
        }
        drift_memory = self._drift_guard()
        profiles = self.store.node_profiles()
        for node in enrolled:
            # VD-194 P1: the worker's software against its profile, as words.
            node["worker_software"] = self.worker_software_safe(node, profiles.get(node["id"], {}))
            node["telemetry_provisioned"] = node["id"] in telemetry_provisioned
            # An agent update the repair pass could not apply (review S-5).
            node["telemetry_repair_note"] = repair_note(
                drift_memory.get(node["id"]) or (node.get("inventory") or {}).get("telemetry_repair")
            )
            # Published for every node, matching or not. A node enrolled before
            # VD-031 existed is not fixed by refusing new joins, and showing it
            # as ordinary would be the same lie arriving by another route.
            node["architecture"] = node_architecture_fact(architecture, node)
            swarm_id = str(node.get("labels", {}).get("swarm_node_id", ""))
            live = runtime_nodes.get(swarm_id)
            node["runtime"] = live
            if swarm_id and live is None and runtime.get("control_available"):
                node["runtime_state"] = "missing"
            elif swarm_id and live is None:
                # ACC-091: a joined node whose cluster state could not be read
                # is UNKNOWN - it used to fall through to "enrolled", which the
                # card then drew as an idle, ready machine.
                node["runtime_state"] = "unreadable"
            elif live:
                node["runtime_state"] = (
                    "ready"
                    if str(live.get("status", "")).lower() == "ready"
                    else str(live.get("status", "unknown")).lower()
                )
            else:
                node["runtime_state"] = "enrolled"
        return {
            "controller": controller,
            "runtime": runtime,
            "enrolled_nodes": enrolled,
            "pooled_deployments": projected_pooled_deployments(
                self.store.list_pooled_deployments(), self._unload_cause,
            ),
            # `worker_os: ["Ubuntu", "Debian", "Raspberry Pi OS (64-bit)"]`
            # used to sit under `requirements` — an OS whitelist standing in
            # for what is really an architecture constraint plus a runtime
            # capability, and one the dashboard never rendered. Both
            # replacements are discovered, and the architecture is published
            # once: it also has to carry the nodes that do not match it.
            "architecture": {
                **fleet_architecture(architecture, enrolled),
                # VD-033. A node this appliance drained and removed leaves no
                # row in the fleet inventory, so a view that simply stopped
                # listing it would be silent about the most destructive thing
                # Vaelor does to its own state. What it removed is published
                # beside what it kept, with the mismatch that caused it.
                "removals": self.store.list_evictions(),
            },
            "requirements": {
                "ports": ["2377/tcp", "7946/tcp+udp", "4789/udp"],
                "network": "trusted LAN or VPN",
                "container_runtime": (
                    "Docker, or a Debian-family operating system so Vaelor "
                    "can install it during the join."
                ),
            },
            "enrollment": {
                **enrollment_readiness(runtime, architecture),
                # Derived from this appliance's own address, so the form can
                # suggest something true instead of the hard-coded
                # literal address it shipped with - a real host from a network
                # this machine has never been on.
                "address_hint": self.address_hint(controller_address),
            },
        }

    def capacity_ledger(
        self, refresh_node_ids=None, freshness_seconds=REFIT_FRESHNESS
    ) -> Dict[str, Any]:
        """The cluster capacity ledger: what each node has and what is committed.

        The read-only foundation two later features consume. Fetching lives
        here (store, driver status, per-service inspects); the arithmetic is
        `cluster_capacity.compute_capacity_ledger`, a pure function this hands
        already-fetched facts so it can be unit-tested without SSH or Docker.

        The controller is included as a placement target (it can host
        workloads) only when it is an active, control-available cluster head,
        so a not-yet-initialised appliance reports an empty ledger rather than
        a lone controller row with no fleet behind it.

        ``refresh_node_ids`` scopes a best-effort live re-probe of the
        selected worker nodes before the ledger is built (Defect C): a
        worker's GPU facts come from its join-time snapshot, refreshed only by
        the manual per-node recheck, so a node whose GPU was freed previewed a
        stale won't-fit. `refresh_nodes_for_fit` re-probes only those ids, and
        only when their snapshot is older than ``freshness_seconds``; the
        controller row keeps its own live probe below. Passing nothing leaves
        every caller that selects no nodes (the plain capacity read) as it was.
        """
        self.refresh_nodes_for_fit(refresh_node_ids, freshness_seconds)
        status = self.driver.status()
        # ACC-091: every worker row carries whether work can be placed on it
        # now, read from its membership record and the live node list, so the
        # fleet's free memory counts only machines that can take work.
        node_facts: list[Dict[str, Any]] = [
            {
                "node_id": node["id"],
                "name": node["name"],
                "role": node.get("role", "worker"),
                "inventory": node.get("inventory", {}),
                **worker_placement_state(node, status),
            }
            for node in self.store.list_nodes()
        ]
        controller = self.store.controller()
        if controller.get("initialized") and status.get("control_available"):
            try:
                hardware = self.inventory_probe() or {}
            except (AttributeError, OSError, TypeError, ValueError):
                hardware = {}
            reclaimable, reclaim_reason = self.gpu_chat_reclaimable()
            node_facts.append(
                controller_node_facts(
                    controller, status, hardware,
                    mode_a_reclaimable=bool(reclaimable),
                    mode_a_reclaimable_reason=reclaim_reason,
                )
            )
        reservations: list[Dict[str, Any]] = []
        for service in status.get("services", []):
            name = str(service.get("name", ""))
            if not name.startswith(("vaelor-app-", "vaelor-llm-")):
                continue
            try:
                details = self.driver.service_details(name)
            except (ClusterDriverError, ValueError):
                # A service that will not inspect is skipped rather than failing
                # the whole ledger; its reservation is simply not yet counted.
                continue
            reservations.append(reservation_from_service_details(details))
        return compute_capacity_ledger(node_facts, reservations)

    def inspect_host(self, host: str, port: int = 22) -> Dict[str, str]:
        clean_host = self._private_host(host, int(port))
        result = host_key_fingerprint(clean_host, int(port))
        return {"host": clean_host, "port": int(port), **result}

    @staticmethod
    def _is_local_target(host: str, port: int) -> bool:
        """Return true when a target address belongs to this controller."""
        try:
            targets = {
                ipaddress.ip_address(item[4][0])
                for item in socket.getaddrinfo(
                    host, int(port), type=socket.SOCK_STREAM
                )
            }
        except (OSError, ValueError):
            return False
        for target in targets:
            family = socket.AF_INET6 if target.version == 6 else socket.AF_INET
            probe = socket.socket(family, socket.SOCK_DGRAM)
            try:
                probe.connect((str(target), int(port)))
                local = ipaddress.ip_address(probe.getsockname()[0])
                if local == target:
                    return True
            except (OSError, ValueError):
                continue
            finally:
                probe.close()
        return False

    #: The port a swarm manager advertises for control-plane traffic. Used to
    #: ask "is this address mine", not to connect to anything.
    ADVERTISE_PORT = 2377

    def advertise_address_health(
        self, controller: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Whether the address this controller advertises is one it still has.

        **Recorded once at cluster init and never re-derived**, so a DHCP
        renewal or a router swap silently invalidates it while every reading
        keeps reporting a healthy cluster. Measured on 2026-08-10: a router swap
        moved the Pi to a new subnet, and its swarm manager went on advertising
        its old address - one the machine no longer had. Docker kept trying
        to hold raft leadership and bind ingress on it, which is why *both*
        boxes had to be hard powered down to pick up their new addresses. The
        owner's only symptom was a machine that would not shut down.

        Clustering genuinely requires a stable address - that part is not a
        defect and a reservation is the right answer. What was missing is that
        nothing said the address had gone. This says it.

        Reported rather than repaired. Rewriting the address would not move
        Docker's own binding, and an appliance that quietly re-points its
        cluster identity is worse than one that says the identity is wrong.
        """
        address = str(controller.get("advertise_address", "") or "")
        if not controller.get("initialized") or not address:
            # Not clustered: there is no address to be wrong about, and a
            # "healthy" verdict here would be a claim about nothing.
            return {"advertise_address_held": None, "advertise_address_reason": ""}
        if self._is_local_target(address, self.ADVERTISE_PORT):
            return {
                "advertise_address_held": True,
                "advertise_address_reason": "",
            }
        return {
            "advertise_address_held": False,
            "advertise_address_reason": (
                "This controller advertises {} to the cluster, and that is not "
                "an address this machine currently holds. Clustering needs a "
                "fixed address: give this machine a static lease or a DHCP "
                "reservation, then re-initialise the cluster. Until then a "
                "worker cannot join, and Docker may hang on shutdown trying to "
                "reach it."
            ).format(address),
        }

    def enroll(self, request: Dict[str, Any]) -> Dict[str, Any]:
        host = self._private_host(request.get("host", ""), int(request.get("port", 22)))
        if self._is_local_target(host, int(request.get("port", 22))):
            raise ValueError(
                "The head controller cannot be enrolled as its own worker."
            )
        name = str(request.get("name", "")).strip()[:80] or host
        username = str(request.get("username", "")).strip()
        password = str(request.get("password", ""))
        fingerprint = str(request.get("host_key_fingerprint", "")).strip()
        observed = host_key_fingerprint(host, int(request.get("port", 22)))
        if observed["fingerprint"] != fingerprint:
            raise ValueError("The confirmed SSH fingerprint does not match this node.")
        secret = json.dumps({
            "host": host,
            "port": int(request.get("port", 22)),
            "username": username,
            "password": password,
            "host_key_fingerprint": fingerprint,
            "sudo_uses_login_password": bool(
                request.get("sudo_uses_login_password", True)
            ),
        })
        credential = self.broker.put("ssh", f"Cluster node: {name}", secret)
        try:
            profile = self.broker.resolve(credential["id"], "cluster-node")
            inventory = self.transport_factory(profile).probe()
            # VD-031, enforced on the first thing the controller learns about
            # the machine rather than at deploy or at container start. The
            # `except` below deletes the credential, so a refused cross-
            # architecture host leaves no stored SSH secret behind.
            verdict = join_compatibility(
                self.controller_architecture(),
                inventory.get("architecture"),
                where=host,
            )
            if not verdict["compatible"]:
                raise ValueError(verdict["reason"])
            node = self.store.add_node(
                name=name,
                host=host,
                port=int(request.get("port", 22)),
                credential_id=credential["id"],
                fingerprint=fingerprint,
                inventory=inventory,
            )
        except Exception:
            self.broker.delete(credential["id"])
            raise
        return node

    def refresh_node(self, node_id: str) -> Dict[str, Any]:
        """Recheck: re-probe a worker, then check its telemetry agent.

        Returns the refreshed record with a ``telemetry`` outcome
        (``{"action", "message"}``) the Recheck route audits and the console
        shows. The reconcile is best-effort - a transport, telemetry-store or
        missing-unit failure never fails the capacity half - but its outcome is
        no longer thrown away (ACC-121): a re-key, a reinstall or a failure is
        returned here and recorded by the route, like every other change.
        """
        try:
            refreshed = self._reprobe_node(node_id)
        except Exception as error:
            self.note_worker_profile_unreached(node_id, error)
            raise
        try:
            outcome = dict(self.reconcile_worker_telemetry(node_id, recheck=True) or {})
            outcome["message"] = reconcile_sentence(outcome)
        except Exception as error:  # noqa: BLE001 - reported, never raised
            LOGGER.warning(
                "telemetry check after Recheck failed for node %s: %s", node_id, error
            )
            outcome = {
                "action": "failed",
                "message": (
                    "Vaelor could not check this machine's telemetry agent: "
                    "{}".format(str(error)[:200])
                ),
            }
        refreshed["telemetry"] = outcome
        # VD-194 P1: Recheck also reads the worker's software (read-only).
        refreshed["worker_software"] = self.recheck_worker_profile_contained(node_id)
        return refreshed

    def _reprobe_node(self, node_id: str) -> Dict[str, Any]:
        """Re-read a worker's hardware over SSH and store what was found.

        **Only the inventory changes (ACC-118).** This used to write
        ``state="enrolled"`` on every Recheck and every serve-form fit re-probe,
        erasing ``drain``/``pause``/``joined``: the card still said Drained
        while the deploy guards, which read ``state``, no longer saw it. The
        membership state is owned by join, drain and remove; a Recheck changes
        it only in the one case it has proof for - the swarm no longer lists the
        node - and then the node really is awaiting a join again.

        **A failed check is recorded, not only raised (ACC-094).** The probe
        only ever wrote ``reachable: true``, so the awaiting-join pill was green
        for a machine that could not be reached. A failure now stores
        ``reachable: false`` with the reason and when it was checked, over the
        last good snapshot, and then raises as before.
        """
        node = self.store.get_node(node_id, include_credential=True)
        if node is None:
            raise ValueError(NODE_NOT_FOUND)
        checked_at = int(time.time())
        try:
            profile = self.broker.resolve(node["credential_id"], "cluster-node")
            inventory = self.transport_factory(profile).probe()
        except Exception as error:
            self._record_unreachable(node, error, checked_at)
            raise
        changes: Dict[str, Any] = {
            "inventory": {**inventory, "checked_at": checked_at},
            "last_seen": checked_at,
        }
        bound = dict(node.get("labels") or {})
        placement = str(bound.get("swarm_node_id", ""))
        if placement and self._swarm_binding_orphaned(placement):
            # Recheck doubles as a recovery path. A label that names a swarm
            # node the controller can no longer see is stale, and while it
            # lingers the Fleet card keeps the rejoin offer hidden. Drop that
            # one label, keep the vaelor.*/pironman.* labels, and the record
            # re-enters the awaiting-join state the affordance keys on - the
            # one case where the membership state is rewritten.
            bound.pop("swarm_node_id", None)
            changes["labels"] = bound
            changes["state"] = "enrolled"
        return self.store.update_node(node_id, **changes)

    def _record_unreachable(
        self, node: Dict[str, Any], error: BaseException, checked_at: int
    ) -> None:
        """Keep the last good snapshot, marked unreachable with the reason."""
        inventory = dict(node.get("inventory") or {})
        inventory.update({
            "reachable": False,
            "checked_at": checked_at,
            # What actually failed - a stored sign-in the broker could not read
            # is not "could not reach over SSH" (review nit).
            "unreachable_reason": "{} at its last check.".format(
                failure_words(error)
            ),
        })
        try:
            self.store.update_node(node["id"], inventory=inventory)
        except Exception:  # noqa: BLE001 - the probe's own error is the one raised
            LOGGER.warning("could not record node %s as unreachable", node["id"])

    def _swarm_binding_orphaned(self, placement: str) -> bool:
        """True only when a bound placement is provably gone from the swarm.

        The recovery clear above acts on proof, never a guess. A live row
        that reads Down is an offline worker due back, so its binding stays;
        a swarm query that could not be read is unknown rather than empty,
        so its binding stays too. Only a readable, control-bearing node list
        that omits the id altogether earns the eviction.
        """
        report = self.driver.status()
        if not report.get("available") or not report.get("control_available"):
            return False
        rows = report.get("nodes")
        live = {
            str(row.get("id", ""))
            for row in (rows if isinstance(rows, list) else [])
            if isinstance(row, dict) and row.get("id")
        }
        # An empty or unparsable list is unreadable, not proof: a manager
        # always lists at least itself (review nit).
        if not live:
            return False
        return placement not in live

    def refresh_nodes_for_fit(
        self, node_ids=None, freshness_seconds=REFIT_FRESHNESS
    ) -> None:
        """Live re-probe the selected worker nodes before a serve-form fit.

        Best-effort and scoped (Defect C). For each requested worker whose
        stored inventory is older than ``freshness_seconds`` (or was never
        stamped), this re-probes over SSH as `refresh_node`'s capacity half does and
        writes the fresh inventory back, so the ledger built next reflects a
        GPU freed since the join-time snapshot rather than the stale verdict
        it otherwise reports.

        Fail-soft per node, without exception: a worker that cannot be
        reached, or whose stored credential no longer resolves, keeps its
        stored snapshot, so no per-node failure ever turns a fit preview into
        a 400 or a 500. Expected transient/config failures fall back quietly;
        anything else is logged at WARNING and still falls back, so a real
        bug surfaces without breaking the fit. Only ids that name an enrolled
        worker are re-probed; the controller placement id names no store row
        and keeps its own live probe in `capacity_ledger`.

        Two freshness gates keep a burst of debounced fits - and a down node
        - off the wire: a node whose stored ``last_seen`` is within the
        window is already fresh, and a node whose last probe ATTEMPT is
        within the window is not retried even if that attempt failed (only
        success stamps ``last_seen``). So a persistently unreachable node is
        re-probed at most once per window, not once per keystroke.
        """
        if not node_ids:
            return
        wanted = {str(node_id) for node_id in node_ids}
        clock = __import__("time").monotonic
        now = __import__("time").time_ns() // 1_000_000_000
        for node in self.store.list_nodes():
            node_id = str(node.get("id", ""))
            if node_id not in wanted:
                continue
            last_seen = node.get("last_seen")
            if (
                isinstance(last_seen, (int, float))
                and now - last_seen <= freshness_seconds
            ):
                continue
            last_attempt = self._fit_probe_attempts.get(node_id)
            if (
                last_attempt is not None
                and clock() - last_attempt <= freshness_seconds
            ):
                # A recent attempt (success OR failure) already covered this
                # window, so a down node is not re-SSH'd on every fit.
                continue
            self._fit_probe_attempts[node_id] = clock()
            try:
                # The capacity half of Recheck only: a fit PREVIEW must not
                # restart, re-key or reinstall a telemetry agent as a side
                # effect of opening the serve form (ACC-121).
                self._reprobe_node(node_id)
            except (
                SshTransportError, OSError, ClusterDriverError, CredentialError,
            ):
                # Expected transient/config failure (unreachable host, or a
                # stale or removed credential): keep this node's stored
                # snapshot and let the fit compute. Never raised out here.
                continue
            except Exception:
                # Backstop: an unexpected failure (e.g. a locked store) must
                # not break the fit either, but it is a real bug, so surface
                # it at WARNING rather than swallow it silently.
                LOGGER.warning("using stored snapshot for node %s", node_id)
                continue

    def remove_node(self, node_id: str) -> bool:
        """Remove an UNJOINED enrolment (the Setup tab's direct route).

        ACC-117, the same rule as the approved-plan job: refused while a model
        deployment still names the node, and Vaelor's telemetry agent is taken
        off the machine - while the credential this deletes can still reach it
        - before anything is deleted.
        """
        try:
            node = self.store.get_node(node_id, include_credential=True)
        except ValueError:
            # PH-R2: a record that will not decode is removed as far as it reads.
            node = self.store.node_core(node_id)
        # Review S8: a record whose labels will not decode still says, in its
        # state column, that the machine is joined - drained through a plan only.
        if node and (node.get("labels", {}).get("swarm_node_id") or node.get("state") == "joined"):
            raise ValueError(
                "Drain and remove this joined worker through an approved cluster plan."
            )
        if node is not None:
            refuse_if_in_use(self.store, node)
            take_telemetry_off(
                self.store, node, lambda: self._worker_transport(node), force=False,
            )
            take_profile_off_safely(self.store, node, lambda: self._worker_transport(node))
        credential_id = self.store.delete_node(node_id)
        if credential_id is None:
            return False
        self.broker.delete(credential_id)
        # This process holds the ingest route's per-node record (review nit).
        forget_ingest_status(node_id)
        # Its chart colour is kept until its history has aged out (VD-147 S4).
        try:
            from .performance_dashboard_slots import ChartSlots

            (getattr(self, "chart_slots", None) or ChartSlots()).release(node_id)
        except Exception as error:  # noqa: BLE001 - the removal stands
            LOGGER.warning("could not release the chart slot of node %s: %s", node_id, error)
        return True
