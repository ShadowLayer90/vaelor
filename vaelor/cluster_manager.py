"""Secure head-controller orchestration for enrolled cluster workers.

This line used to say "enrolled Raspberry Pi workers", which was the whole of
the arm64 assumption stated outright. Workers are admitted on what the
controller observes about them, and the one thing they must share with it is
its processor architecture (VD-031) — not its board and not its OS.
"""

from __future__ import annotations

import ipaddress
import json
import socket
from typing import Any, Dict, Optional

from .cluster_architecture import (
    controller_architecture,
    fleet_architecture,
    join_compatibility,
    node_architecture_fact,
)
from .cluster_driver import DockerSwarmDriver
from .cluster_store import ClusterStore
from .credential_broker import CredentialBrokerClient
from .ssh_transport import SshTransport, host_key_fingerprint
from .workload_broker import WorkloadBrokerClient


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


class ClusterManager:
    def __init__(
        self,
        store: Optional[ClusterStore] = None,
        broker: Optional[CredentialBrokerClient] = None,
        driver: Optional[DockerSwarmDriver] = None,
        transport_factory=SshTransport,
        inventory_probe=None,
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
        ``192.168.0.181`` — a real machine from someone else's network, on an
        appliance whose own subnet was 192.168.4.x — so the one concrete thing
        the form suggested was wrong, and it looked like a remembered value
        rather than an example.

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

    def summary(self, controller_address: str = "") -> Dict[str, Any]:
        runtime = self.driver.status()
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
        enrolled = self.store.list_nodes()
        architecture = self.controller_architecture()
        runtime_nodes = {
            str(item.get("id", "")): item for item in runtime.get("nodes", [])
        }
        for node in enrolled:
            # Published for every node, matching or not. A node enrolled before
            # VD-031 existed is not fixed by refusing new joins, and showing it
            # as ordinary would be the same lie arriving by another route.
            node["architecture"] = node_architecture_fact(architecture, node)
            swarm_id = str(node.get("labels", {}).get("swarm_node_id", ""))
            live = runtime_nodes.get(swarm_id)
            node["runtime"] = live
            if swarm_id and live is None and runtime.get("control_available"):
                node["runtime_state"] = "missing"
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
            "pooled_deployments": self.store.list_pooled_deployments(),
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
                # 192.168.0.181 it shipped with - a real host from a network
                # this machine has never been on.
                "address_hint": self.address_hint(controller_address),
            },
        }

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
        192.168.0.173 - an address the machine no longer had. Docker kept trying
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
        node = self.store.get_node(node_id, include_credential=True)
        if node is None:
            raise ValueError("Cluster node was not found.")
        profile = self.broker.resolve(node["credential_id"], "cluster-node")
        inventory = self.transport_factory(profile).probe()
        return self.store.update_node(
            node_id, inventory=inventory, state="enrolled",
            last_seen=__import__("time").time_ns() // 1_000_000_000,
        )

    def remove_node(self, node_id: str) -> bool:
        node = self.store.get_node(node_id)
        if node and node.get("labels", {}).get("swarm_node_id"):
            raise ValueError(
                "Drain and remove this joined worker through an approved cluster plan."
            )
        credential_id = self.store.delete_node(node_id)
        if credential_id is None:
            return False
        self.broker.delete(credential_id)
        return True
