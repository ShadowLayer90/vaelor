"""Swappable cluster drivers; Docker Swarm is the first supported engine."""

from __future__ import annotations

import json
import re
import shutil
import socket
import subprocess
import time
from typing import Any, Dict, Optional

from .cluster_service_state import (
    normalize_node_runtime,
    normalize_service_details,
    stored_update_flags,
)
from .model_service_compose import CPU_IMAGE, prompt_cache_mib
from .platform_drivers import AptPackageManager
from .ssh_transport import SshTransport
from .workload_broker import (
    SERVICE_TASKS_FORMAT,
    SWARM_INFO_COMMAND,
    SWARM_NODES_COMMAND,
    SWARM_SERVICES_COMMAND,
)


class ClusterDriverError(RuntimeError):
    pass


#: The context window every swarm LLM service runs (#138); the `cluster_operations`
#: memory limit derives from a footprint measured at this window, one slot (VD-076/VD-080).
SWARM_CONTEXT_TOKENS = 4096


class DockerSwarmDriver:
    """Head-controller operations for a Docker Swarm fleet."""

    name = "docker-swarm"

    def __init__(self, timeout: int = 120, package_manager=None, runner=None):
        self.timeout = max(10, min(int(timeout), 600))
        self.package_manager = package_manager or AptPackageManager(
            finder=lambda name: name
        )
        #: How Docker is reached. ``None`` runs ``docker`` directly, correct only for
        #: a process holding the privilege itself (the workload executor). The control
        #: plane does not (#141): its ClusterManager passes the broker's ``run`` here.
        self.runner = runner

    def _docker(self, *arguments: str) -> str:
        command = ["docker", *arguments]
        try:
            if self.runner is not None:
                # Tighter than the direct cap on purpose: the broker serves one
                # connection at a time, so a read unanswered in ten seconds is a hung
                # daemon; the full window would starve every other brokered read.
                result = self.runner(command, min(self.timeout, 10))
            else:
                result = subprocess.run(
                    command,
                    check=False,
                    capture_output=True,
                    text=True,
                    timeout=self.timeout,
                )
        except (OSError, subprocess.TimeoutExpired) as error:
            raise ClusterDriverError("Docker is unavailable on the head controller.") from error
        if result.returncode != 0:
            raise ClusterDriverError(
                (result.stderr or result.stdout).strip()[:500]
                or "The Docker cluster command failed."
            )
        return result.stdout.strip()

    @staticmethod
    def _managed_service_name(service_name: str) -> str:
        value = str(service_name).strip()
        if (
            not value.startswith((
                "vaelor-app-", "vaelor-llm-",
            ))
            or len(value) > 63
            or not all(
                character.isalnum() or character in "._-"
                for character in value
            )
        ):
            raise ValueError("Choose a Vaelor-managed cluster service.")
        return value

    def _swarm_unreadable(self, error: Exception) -> Dict[str, Any]:
        """Task #75. Why one failed Swarm query is not a verdict about Docker.

        ``status()`` had a single failure branch: any way ``docker info`` could
        fail — binary missing, daemon socket refused to an unprivileged control
        plane, timeout, template error, unparseable answer — became
        ``available: False``, which the console renders as *"Check that Docker
        is installed and running."*

        Measured on an HP Z2 Mini G1a and independently on a Pironman
        appliance: Apps and AI reported ``DOCKER READY`` and a container had
        been deployed through Vaelor minutes earlier, while Cluster told the
        owner Docker was missing. The two paths do not measure the same thing.
        The workload path asks the *broker* (:mod:`vaelor.workload_broker`,
        which exists precisely because this control plane is unprivileged);
        this driver shells out to ``docker`` itself, and a socket the service
        user may not open is not an absent engine.

        So the engine fact is reported separately from the Swarm fact, and it
        is reported from evidence: ``absent`` only when the binary genuinely is
        not on this machine, ``unreadable`` — with the real failure text —
        otherwise. Nothing here guesses a remedy.

        #141 adds the one failure whose subject is not the engine at all:
        ``permission denied`` on the socket means *this process* may not ask,
        and says nothing about Docker, which on the appliance that reported it
        was running five Vaelor containers at that moment. The reason names
        the caller, because "the runtime is not reachable on this appliance"
        was the defect.
        """
        present = shutil.which("docker") is not None
        detail = str(error).strip()[:300]
        # Which process actually asked Docker: through a runner the query ran inside
        # the broker's daemon, so "this Vaelor process" would name the wrong caller.
        asker = (
            "Vaelor's workload broker" if self.runner is not None
            else "this Vaelor service"
        )
        if not present:
            reason = "Docker is not installed on this machine."
        elif "permission denied" in detail.lower():
            reason = (
                "Docker refused {}'s query for lack of permission; that is a "
                "fact about the asking process, not the engine, which may be "
                "running normally: {}".format(asker, detail)
            )
        else:
            reason = (
                "Docker is installed here, but Vaelor could not read its "
                "cluster state: {}".format(detail or "the query failed.")
            )
        return {
            "available": False,
            "initialized": False,
            "driver": self.name,
            "control_available": False,
            "engine": "absent" if not present else "unreadable",
            "engine_reason": reason,
        }

    def status(self) -> Dict[str, Any]:
        # The three reads are issued verbatim from the shared constants, so the broker
        # allowlist admits exactly what this method sends. All sit in one handler: a
        # node/service read failing after a good info read now reports status unreadable.
        try:
            raw = self._docker(*SWARM_INFO_COMMAND[1:])
            swarm = json.loads(raw)
            if not isinstance(swarm, dict):
                # `docker info` answered but had no Swarm section to report; that used
                # to reach `swarm.get(...)` and raise, turning a readable engine into a 500.
                return self._swarm_unreadable(
                    ClusterDriverError("Docker reported no cluster section.")
                )
            initialized = (
                str(swarm.get("LocalNodeState", "")).lower() == "active"
            )
            control = bool(swarm.get("ControlAvailable"))
            nodes = []
            services = []
            if control:
                # `docker node ls` emits Status/Availability capitalized;
                # `normalize_node_runtime` lowercases them so a raw `== "drain"` holds (ids/hostnames keep case).
                nodes = normalize_node_runtime(
                    self._parse_json_lines(self._docker(*SWARM_NODES_COMMAND[1:]))
                )
                services = self._parse_json_lines(
                    self._docker(*SWARM_SERVICES_COMMAND[1:])
                )
        except (ClusterDriverError, json.JSONDecodeError) as error:
            return self._swarm_unreadable(error)
        return {
            "available": True,
            "initialized": initialized,
            "driver": self.name,
            "control_available": control,
            # Stated on the success path too, so a reader of this payload never
            # has to infer the engine's condition from the absence of a key.
            "engine": "ready",
            "engine_reason": "",
            "node_id": swarm.get("NodeID", ""),
            "nodes": nodes,
            "services": services,
        }

    @staticmethod
    def _parse_json_lines(raw: str) -> list[Dict[str, Any]]:
        result = []
        for line in raw.splitlines():
            try:
                item = json.loads(line)
            except json.JSONDecodeError:
                continue
            if isinstance(item, dict):
                result.append(item)
        return result

    def initialize(self, advertise_address: str) -> Dict[str, Any]:
        self._docker("swarm", "init", "--advertise-addr", advertise_address)
        return self.status()

    def join_worker(
        self,
        transport: SshTransport,
        advertise_address: str,
        *,
        install_docker: bool,
    ) -> Dict[str, Any]:
        cleared_stale_swarm = False
        if install_docker:
            transport.run(
                self.package_manager.update_command(),
                sudo=True,
                timeout=300,
            )
            transport.run(
                self.package_manager.install_command(["docker.io"]),
                sudo=True,
                timeout=600,
            )
            transport.run(["systemctl", "enable", "--now", "docker"], sudo=True)
        # A worker rebuilt against a fresh controller still holds its old swarm
        # membership, and Docker refuses `swarm join` on a node already in one. Probe
        # (sudo: not in the docker group); if not clean, force-leave. Unreadable: untouched.
        try:
            probe = transport.run(
                ["docker", "info", "--format", "{{.Swarm.LocalNodeState}}"],
                sudo=True,
            )
            seen = [line for line in str(probe).splitlines() if line.strip()]
            prior_swarm_state = seen[-1].strip().lower() if seen else ""
        except Exception:
            prior_swarm_state = ""
        if prior_swarm_state and prior_swarm_state != "inactive":
            try:
                transport.run(
                    ["docker", "swarm", "leave", "--force"],
                    sudo=True,
                    timeout=180,
                )
            except Exception as error:
                if not self._already_left_swarm(str(error)):
                    raise ClusterDriverError(
                        "The worker holds a stale cluster membership that could "
                        "not be cleared automatically. Clear it and retry."
                    ) from error
            cleared_stale_swarm = True
        hostname = transport.run(["uname", "-n"]).strip()
        # A force-left node lingers (Down) under this hostname, so record
        # the prior ids now to tell the freshly joined node from the orphan.
        before_ids = {
            str(n.get("id", "")) for n in self.status().get("nodes", [])
            if n.get("hostname") == hostname
        }
        token = self._docker("swarm", "join-token", "-q", "worker")
        try:
            output = transport.run(
                [
                    "docker", "swarm", "join",
                    "--token", token,
                    f"{advertise_address}:2377",
                ],
                sudo=True,
                timeout=180,
            )
        finally:
            # A join token is a short-lived bootstrap secret in Pironman's flow.
            self._docker("swarm", "join-token", "--rotate", "worker")
        # The new node shares the hostname but its id is absent from
        # before_ids; prefer a Ready one, else fall back to first match.
        fresh = [
            n for n in self.status().get("nodes", [])
            if n.get("hostname") == hostname
            and str(n.get("id", "")) not in before_ids
        ]
        ready = [n for n in fresh if str(n.get("status", "")).lower() == "ready"]
        picks = ready or fresh
        if picks:
            swarm_node_id = str(picks[0].get("id", ""))
        else:
            swarm_node_id = self.node_id_by_hostname(hostname)
        if cleared_stale_swarm:
            # Reap this hostname's prior node(s) only; a normal join never does.
            for stale_id in before_ids:
                try:
                    self._docker("node", "rm", "--force", stale_id)
                except ClusterDriverError:
                    pass
        result = {
            "joined": True,
            "message": output[-300:],
            "hostname": hostname[:253],
            "swarm_node_id": swarm_node_id,
            "cleared_stale_swarm": cleared_stale_swarm,
        }
        if cleared_stale_swarm:
            result["prior_swarm_state"] = prior_swarm_state
        return result

    def node_id_by_hostname(self, hostname: str) -> str:
        for node in self.status().get("nodes", []):
            if node.get("hostname") == hostname:
                return str(node.get("id", ""))
        raise ClusterDriverError("The worker joined, but its Swarm node record was not found.")

    def label_node(self, swarm_node_id: str, labels: Dict[str, str]) -> None:
        arguments = ["node", "update"]
        for key, value in sorted(labels.items()):
            arguments.extend(["--label-add", f"{key}={value}"])
        arguments.append(swarm_node_id)
        self._docker(*arguments)

    def remove_node_label(self, swarm_node_id: str, label: str) -> None:
        if not label or not all(
            character.isalnum() or character in "._-"
            for character in str(label)
        ):
            raise ValueError("Choose a valid managed node label.")
        self._docker(
            "node", "update", "--label-rm", str(label), str(swarm_node_id)
        )

    def set_node_availability(self, swarm_node_id: str, availability: str) -> Dict[str, Any]:
        normalized = str(availability).strip().lower()
        if normalized not in {"active", "pause", "drain"}:
            raise ValueError("Choose active, pause, or drain for this worker.")
        self._docker(
            "node", "update", "--availability", normalized, str(swarm_node_id)
        )
        return {"swarm_node_id": str(swarm_node_id), "availability": normalized}

    @staticmethod
    def _already_left_swarm(message: str) -> bool:
        # `docker swarm leave` exits non-zero on a node already out of the swarm
        # ("This node is not part of a swarm"). For a removal that is the state we
        # want, not a failure: a retried removal must not abort the cleanup.
        return "not part of a swarm" in message.lower()

    @staticmethod
    def _node_already_removed(message: str) -> bool:
        text = message.lower()
        return "not found" in text or "no such node" in text

    def _await_node_down(
        self, swarm_node_id: str, *, timeout: float = 30.0, interval: float = 2.0
    ) -> None:
        # A manager rejects `docker node rm` until it has observed a departed worker
        # go Down ("node ... is not down and can't be removed"), and that observation
        # lags the worker's own `swarm leave` by a heartbeat. Poll so a clean removal
        # does not lose that race: return the moment it is Down or gone, and after the
        # timeout regardless so a slow detection still attempts the removal, not hang.
        deadline = time.monotonic() + timeout
        while True:
            try:
                state = self._docker(
                    "node", "inspect", str(swarm_node_id),
                    "--format", "{{ .Status.State }}",
                ).lower()
            except ClusterDriverError:
                return
            if "down" in state or time.monotonic() >= deadline:
                return
            time.sleep(interval)

    def remove_worker(
        self,
        transport: SshTransport,
        swarm_node_id: str,
        *,
        force: bool = False,
    ) -> Dict[str, Any]:
        try:
            self.set_node_availability(swarm_node_id, "drain")
        except ClusterDriverError:
            # Force is a deliberate override that cleans up regardless of state.
            # The clean path must NOT proceed on a drain that cannot find the
            # node: reporting a removal it did not do strands a live worker.
            if not force:
                raise
        leave_error = ""
        try:
            arguments = ["docker", "swarm", "leave"]
            if force:
                arguments.append("--force")
            transport.run(arguments, sudo=True, timeout=180)
        except Exception as error:
            leave_error = str(error)
            # A worker already out of the swarm is success, not a failure to
            # leave; only a genuinely stuck or unreachable worker aborts.
            if not force and not self._already_left_swarm(leave_error):
                raise ClusterDriverError(
                    "The worker could not leave cleanly. Reconnect it or review a forced removal."
                ) from error
        # Wait for the manager to see the worker go Down before removing it, so
        # the clean path does not race the heartbeat; a forced removal skips the
        # wait and removes regardless of the node's observed state.
        if not force:
            self._await_node_down(swarm_node_id)
        arguments = ["node", "rm"]
        if force:
            arguments.append("--force")
        arguments.append(str(swarm_node_id))
        try:
            self._docker(*arguments)
        except ClusterDriverError as error:
            # Idempotent: a retried removal whose node rm had already succeeded
            # finds the node gone, which is the outcome we wanted.
            if not self._node_already_removed(str(error)):
                raise
        return {
            "removed": True,
            "swarm_node_id": str(swarm_node_id),
            "forced": bool(force),
            "worker_leave_warning": leave_error[:300],
        }

    def deploy_llm(
        self,
        *,
        name: str,
        node_label: Optional[str],
        model_repo: str,
        model_file: str,
        memory_limit_mib: int,
        port: Optional[int] = None,
        pool_label: Optional[str] = None,
        replicas: int = 1,
    ) -> Dict[str, Any]:
        replica_count = max(1, min(int(replicas), 8))
        if bool(node_label) == bool(pool_label):
            raise ValueError("Choose one managed LLM placement policy.")
        service_name = "vaelor-llm-{}".format(
            "".join(character for character in name.lower() if character.isalnum() or character == "-")[:40]
        )
        placement = (
            f"node.labels.{pool_label}==true"
            if pool_label
            else f"node.labels.vaelor.node_id=={node_label}"
        )
        command = [
            "service", "create",
            # Accept the spec without blocking on convergence: the engine default
            # waits, and a crash-looping app never converges. wait_service waits.
            "--detach",
            "--name", service_name,
            "--constraint", placement,
            "--replicas", str(replica_count),
            "--replicas-max-per-node", "1",
            "--reserve-memory", f"{max(768, int(memory_limit_mib * 0.8))}M",
            "--limit-memory", f"{int(memory_limit_mib)}M",
            "--mount", f"type=volume,source={service_name}-cache,target=/root/.cache",
            "--restart-condition", "on-failure",
            "--restart-max-attempts", "5",
            "--label", "vaelor.managed=true",
            "--label", "vaelor.workload=llm",
            "--label", (
                "vaelor.inference.mode=replicated"
                if replica_count > 1 else
                "vaelor.inference.mode=single"
            ),
        ]
        if pool_label:
            command.extend(["--label", f"vaelor.pool-label={pool_label}"])
        if port is not None:
            command.extend([
                "--publish", f"published={int(port)},target=8080,mode=ingress",
            ])
        command.extend([
            # The same pinned engine the single-node deploy uses. A swarm service
            # resolving `:server` independently would put a different build on each
            # worker as the tag moved: the #130 reproducibility defect, pool-sized.
            CPU_IMAGE,
            "-hf", f"{model_repo}:{model_file}",
            "--host", "0.0.0.0",
            "--port", "8080",
            # Stated, not inherited (#138): the memory limit this service runs under
            # is derived from a footprint measured at this window, so the window must
            # be this one by declaration, not by the engine default happening to match.
            "--ctx-size", str(SWARM_CONTEXT_TOKENS),
            # **The image was inherited from the single-node deploy; the flags that
            # make it survivable were not.** Pinning the engine then running it with
            # its own defaults under a hard `--limit-memory` is the worse of both.
            #
            # `--parallel` first, because it is the one that took the appliance
            # down: unstated, llama.cpp allocates the window once per slot and
            # defaults to four, so a limit computed for one context is enforced
            # against four copies of it (#109). This is a served, LAN-reachable
            # surface, which is exactly the case that measurement came from.
            "--parallel", "1",
            # And the prompt cache, from the same bound the compose path uses. The
            # engine default is 8192 MiB - larger than a Pi worker's entire RAM - and
            # unbounded it grew ~13 MB per prompt with no plateau in 40 (VD-080).
            # `tests/test_prompt_cache_bound.py` covered only one of two renderers.
            "--cache-ram", str(prompt_cache_mib(int(memory_limit_mib))),
        ])
        service_id = self._docker(*command)
        return {
            "service_id": service_id,
            "name": service_name,
            "compatibility": "OpenAI-compatible",
            "api_paths": ["/v1/models", "/v1/chat/completions", "/v1/completions"],
            "exposure": "lan" if port is not None else "cluster-private",
            "port": port,
            "memory_limit_mib": int(memory_limit_mib),
            "replicas": replica_count,
            "deployment_mode": "replicated" if replica_count > 1 else "single",
            "pool_label": pool_label or "",
        }

    def deploy_app(
        self,
        *,
        name: str,
        image: str,
        container_port: int,
        published_port: int,
        memory_limit_mib: int,
        template_id: str,
        constraints: Optional[list[str]] = None,
        placement_flags: Optional[list[str]] = None,
        replicas: int = 1,
        env_file: Optional[str] = None,
        placement_intent: str = "",
        requested_replicas: int = 0,
        volume: Optional[tuple[str, str]] = None,
        extra_ports: Optional[list[tuple[str, str, int]]] = None,
        cpu_limit: float = 0.0,
        cpu_reservation: float = 0.0,
        label_constraints: Optional[list[str]] = None,
    ) -> Dict[str, Any]:
        safe_name = "".join(
            character
            for character in str(name).lower()
            if character.isalnum() or character == "-"
        )[:40]
        service_name = f"vaelor-app-{safe_name}"
        replica_count = max(1, min(int(replicas), 32))
        command = [
            "service", "create",
            # Accept the spec without blocking on convergence: the engine default
            # waits, and a crash-looping app never converges. wait_service waits.
            "--detach",
            "--name", service_name,
            "--replicas", str(replica_count),
            # Floor 16 MiB, matching the reconfigure reservation floor: a higher
            # deploy floor emits reserve > limit for a sub-64-MiB template (Swarm
            # rejects it), so deploy would fail where reconfigure succeeds.
            "--reserve-memory", f"{max(16, int(memory_limit_mib * 0.75))}M",
            "--limit-memory", f"{int(memory_limit_mib)}M",
            "--restart-condition", "on-failure",
            "--restart-max-attempts", "5",
            "--publish",
            (
                f"published={int(published_port)},target={int(container_port)},"
                "mode=ingress"
            ),
            "--label", "vaelor.managed=true",
            "--label", "vaelor.workload=app",
            "--label", f"vaelor.template={template_id}",
        ]
        # The placement decision the shared reconcile made: a run-once/pin
        # constraint pins one machine; spread's `--replicas-max-per-node 1`
        # caps one copy per machine and pins none (Swarm spreads them).
        for constraint in constraints or []:
            command.extend(["--constraint", str(constraint)])
        command.extend(str(flag) for flag in placement_flags or [])
        # D2: an operator CPU limit + a conservative reservation (validated in
        # `cluster_service_reconfigure`), and any operator label constraints,
        # which ADD to the managed pin. CPU never gates placement (memory does).
        if cpu_limit:
            command.extend([
                "--limit-cpu", f"{float(cpu_limit)}",
                "--reserve-cpu", f"{float(cpu_reservation)}",
            ])
        for constraint in label_constraints or []:
            command.extend(["--constraint", str(constraint)])
        # The placement intent + requested replica count ride as managed labels,
        # recovered through `service_details`, so the status view reads the
        # decision back without a per-app record (mirrors vaelor.template).
        if placement_intent:
            command.extend(
                ["--label", f"vaelor.placement-intent={placement_intent}"]
            )
        if requested_replicas:
            command.extend([
                "--label", f"vaelor.placement-replicas={int(requested_replicas)}",
            ])
        # The app's install env (plain settings + any generated password) is
        # delivered from a root-owned 0600 file the caller unlinks after this
        # returns — never on an argv /proc exposes, never a 0644 file (VD-129).
        if env_file:
            command.extend(["--env-file", str(env_file)])
        for _extra_name, protocol, extra_port in extra_ports or []:
            command.extend([
                "--publish",
                (
                    f"published={int(extra_port)},target={int(extra_port)},"
                    f"protocol={protocol},mode=host"
                ),
            ])
        if volume:
            volume_name, target = volume
            command.extend([
                "--mount",
                (
                    f"type=volume,source={service_name}-{volume_name},"
                    f"target={target}"
                ),
            ])
        command.append(str(image))
        service_id = self._docker(*command)
        pinned = ""
        for constraint in constraints or []:
            match = re.fullmatch(
                r"node\.labels\.vaelor\.node_id==(.+)", str(constraint)
            )
            if match:
                pinned = match.group(1)
                break
        return {
            "service_id": service_id,
            "name": service_name,
            "template_id": template_id,
            "port": int(published_port),
            "memory_limit_mib": int(memory_limit_mib),
            "node_id": pinned,
            "replicas": replica_count,
            "placement_intent": placement_intent,
            "persistent": bool(volume),
        }

    def wait_service(
        self,
        service_name: str,
        timeout: int = 180,
        expected_replicas: int = 1,
    ) -> Dict[str, Any]:
        expected = max(1, min(int(expected_replicas), 32))
        deadline = time.monotonic() + max(10, min(int(timeout), 600))
        last = ""
        while time.monotonic() < deadline:
            # `--filter name=` is a SUBSTRING match, so a service that is a prefix of
            # another (vaelor-llm-a / vaelor-llm-abc) returns several rows. Emit the
            # name alongside the replicas and select the EXACT match, so a healthy
            # service is never failed for a sibling's line (#Recovery-8).
            rows = self._docker(
                "service", "ls", "--filter", f"name={service_name}",
                "--format", "{{.Name}} {{.Replicas}}",
            ).splitlines()
            last = ""
            for row in rows:
                row_name, _, replicas = row.strip().partition(" ")
                if row_name == service_name:
                    last = replicas.strip()
                    break
            # {{.Replicas}} trails a placement annotation for a service created with
            # --replicas-max-per-node ("1/1 (max 1 per node)") - and the appliance's
            # managed LLM services are. The ready count is the leading whitespace token;
            # compare on that, not the whole field, so a healthy such service is not
            # polled to the deadline and failed (#Recovery-8b). Full string stays in `last`.
            count = last.split()[0] if last.split() else ""
            if count == f"{expected}/{expected}":
                return {"ready": True, "replicas": last}
            if count.endswith(f"/{expected}"):
                time.sleep(2)
                continue
            time.sleep(2)
        # An honest, ACTIONABLE timeout: the replica count alone ("0/1") does not say
        # WHY. Append the most recent task's real state so the operator tells a crash
        # ("task: non-zero exit (1)") from a download ("Preparing"/"Pulling"), best-effort.
        detail = self._last_task_detail(service_name)
        raise ClusterDriverError(
            "The service did not reach {} healthy replica{} (last state: {}){}.".format(
                expected,
                "" if expected == 1 else "s",
                last or "missing",
                "; " + detail if detail else "",
            )
        )

    def _last_task_detail(self, service_name: str) -> str:
        """The most recent task's current state and error, for a readiness
        timeout message. Best-effort: a failure to read it returns "" so the
        message still renders."""
        try:
            rows = self._docker(
                "service", "ps", service_name, "--no-trunc",
                "--format", "{{.CurrentState}}\t{{.Error}}",
            ).splitlines()
        except (ClusterDriverError, OSError):
            return ""
        for row in rows:
            state, _, error = row.partition("\t")
            state = state.strip()
            error = error.strip()
            if state:
                return "last task: {}{}".format(
                    state, " — " + error if error else ""
                )
        return ""

    def service_details(self, service_name: str) -> Dict[str, Any]:
        """Return an operator-safe view without exposing environment secrets."""
        name = self._managed_service_name(service_name)
        try:
            raw = json.loads(self._docker("service", "inspect", name))
        except json.JSONDecodeError as error:
            raise ClusterDriverError(
                "Docker returned invalid cluster service details."
            ) from error
        if not isinstance(raw, list) or not raw or not isinstance(raw[0], dict):
            raise ClusterDriverError("The managed cluster service was not found.")
        tasks = self._parse_json_lines(self._docker(
            "service", "ps", name, "--no-trunc", "--format",
            SERVICE_TASKS_FORMAT,
        ))
        return normalize_service_details(raw[0], tasks, name)

    def inspect_services(self, names) -> list:
        """Bulk ``docker service inspect`` over managed services for the D3 state
        read: ONE inspect for the whole app list (names re-validated as Vaelor-
        managed here, matching the broker allowlist). Best-effort — a failed read
        yields an empty list so the summary degrades to an unknown state."""
        managed = []
        for name in names or []:
            try:
                managed.append(self._managed_service_name(name))
            except ValueError:
                continue
        if not managed:
            return []
        try:
            data = json.loads(self._docker("service", "inspect", *managed))
        except (ClusterDriverError, json.JSONDecodeError):
            return []
        return [item for item in data if isinstance(item, dict)] if isinstance(
            data, list
        ) else []

    def service_logs(
        self, service_name: str, *, lines: int = 200
    ) -> Dict[str, Any]:
        name = self._managed_service_name(service_name)
        tail = max(20, min(int(lines), 500))
        command = [
            "docker", "service", "logs", "--raw", "--timestamps",
            "--tail", str(tail), name,
        ]
        # Through the runner when one is set (#141 review): this method used
        # to shell out unconditionally, so a brokered driver still ran docker
        # directly — the exact defect the brokering exists to remove, one
        # method away from the fix.
        try:
            if self.runner is not None:
                result = self.runner(command, min(self.timeout, 15))
            else:
                result = subprocess.run(
                    command,
                    check=False,
                    capture_output=True,
                    text=True,
                    timeout=min(self.timeout, 30),
                )
        except (OSError, subprocess.TimeoutExpired) as error:
            raise ClusterDriverError(
                "Cluster service logs are unavailable."
            ) from error
        output = "\n".join(
            part.strip() for part in (result.stdout, result.stderr)
            if part.strip()
        )
        if result.returncode != 0:
            raise ClusterDriverError(
                output[-500:] or "Cluster service logs are unavailable."
            )
        return {
            "service": name,
            "lines": tail,
            "output": output[-65536:],
        }

    def refresh_service(self, service_name: str) -> Dict[str, Any]:
        name = self._managed_service_name(service_name)
        details = self.service_details(name)
        image = str(details.get("image", ""))
        if not image:
            raise ClusterDriverError("The service image could not be resolved.")
        # Re-apply the STORED update policy, never a hardcoded one (D2 follow-up b): a
        # refresh must not reset an operator's configured parallelism/order to 1/rollback.
        self._docker(
            "service", "update",
            # Detached like create: a non-converging update would block to 600s.
            "--detach",
            "--image", image,
            "--force",
            *stored_update_flags(details),
            "--rollback-parallelism", "1",
            name,
        )
        readiness = self.wait_service(
            name,
            timeout=min(self.timeout, 600),
            expected_replicas=details["desired_replicas"],
        )
        return {
            "name": name,
            "action": "refresh",
            "image": image,
            **readiness,
        }

    def restart_service(self, service_name: str) -> Dict[str, Any]:
        name = self._managed_service_name(service_name)
        details = self.service_details(name)
        # Re-apply the STORED update policy, not a hardcoded 1/rollback (D2
        # follow-up b) — a restart preserves the operator's configured policy.
        self._docker(
            "service", "update",
            # Detached like create: a non-converging update would block to 600s.
            "--detach",
            "--force",
            *stored_update_flags(details),
            name,
        )
        readiness = self.wait_service(
            name,
            timeout=min(self.timeout, 600),
            expected_replicas=details["desired_replicas"],
        )
        return {"name": name, "action": "restart", **readiness}

    def rollback_service(self, service_name: str) -> Dict[str, Any]:
        name = self._managed_service_name(service_name)
        details = self.service_details(name)
        self._docker("service", "rollback", name)
        readiness = self.wait_service(
            name,
            timeout=min(self.timeout, 600),
            expected_replicas=details["desired_replicas"],
        )
        return {"name": name, "action": "rollback", **readiness}

    def scale_service(
        self, service_name: str, replicas: int
    ) -> Dict[str, Any]:
        name = self._managed_service_name(service_name)
        desired = max(0, min(int(replicas), 32))
        self._docker("service", "scale", f"{name}={desired}")
        if desired == 0:
            deadline = time.monotonic() + min(self.timeout, 180)
            last = ""
            while time.monotonic() < deadline:
                last = self._docker(
                    "service", "ls", "--filter", f"name={name}",
                    "--format", "{{.Replicas}}",
                ).strip()
                if last == "0/0":
                    return {
                        "name": name,
                        "replicas": last,
                        "ready": True,
                    }
                time.sleep(1)
            raise ClusterDriverError(
                f"The service did not stop every replica (last state: {last})."
            )
        return {
            "name": name,
            **self.wait_service(
                name,
                timeout=min(self.timeout, 600),
                expected_replicas=desired,
            ),
        }

    def configure_service(
        self,
        service_name: str,
        *,
        replicas: int,
        memory_limit_mib: int,
        memory_reservation_mib: int,
        update_parallelism: int,
        update_order: str,
        cpu_limit: float = 0.0,
        cpu_reservation: float = 0.0,
        label_constraints_add: Optional[list[str]] = None,
    ) -> Dict[str, Any]:
        """Apply the bounded service settings exposed by the control plane."""
        name = self._managed_service_name(service_name)
        desired = int(replicas)
        limit = int(memory_limit_mib)
        reservation = int(memory_reservation_mib)
        parallelism = int(update_parallelism)
        order = str(update_order).strip().lower()
        if not 1 <= desired <= 32:
            raise ClusterDriverError("Choose between 1 and 32 replicas.")
        if not 16 <= limit <= 131072:
            raise ClusterDriverError(
                "Choose a memory limit from 16 MiB to 128 GiB."
            )
        if not 16 <= reservation <= limit:
            raise ClusterDriverError(
                "Memory reservation must be at least 16 MiB and no larger "
                "than the limit."
            )
        if not 1 <= parallelism <= min(desired, 8):
            raise ClusterDriverError(
                "Update parallelism must be between 1 and the replica count."
            )
        if order not in {"start-first", "stop-first"}:
            raise ClusterDriverError(
                "Choose start-first or stop-first rolling updates."
            )
        update = [
            "service", "update",
            # Detached like create: a non-converging update would block to 600s.
            "--detach",
            "--replicas", str(desired),
            "--limit-memory", f"{limit}M",
            "--reserve-memory", f"{reservation}M",
            "--update-parallelism", str(parallelism),
            "--update-order", order,
            "--update-failure-action", "rollback",
            "--rollback-parallelism", "1",
        ]
        # D2: a CPU limit/reservation (validated in cluster_service_reconfigure); and
        # operator label constraints, ADDED (the managed pin is preserved by NOT passing
        # --constraint). Removing a label constraint is deferred with label-setting to G.
        if cpu_limit:
            update.extend([
                "--limit-cpu", f"{float(cpu_limit)}",
                "--reserve-cpu", f"{float(cpu_reservation)}",
            ])
        for constraint in label_constraints_add or []:
            update.extend(["--constraint-add", str(constraint)])
        update.append(name)
        self._docker(*update)
        readiness = self.wait_service(
            name,
            timeout=min(self.timeout, 600),
            expected_replicas=desired,
        )
        return {
            "name": name,
            "action": "configure",
            "configuration": {
                "replicas": desired,
                "memory_limit_mib": limit,
                "memory_reservation_mib": reservation,
                "update_parallelism": parallelism,
                "update_order": order,
                "cpu_limit": float(cpu_limit) if cpu_limit else None,
            },
            **readiness,
        }

    def remove_service(self, service_name: str) -> None:
        service_name = self._managed_service_name(service_name)
        pool_label = ""
        try:
            pool_label = self._docker(
                "service", "inspect", "--format",
                '{{index .Spec.Labels "vaelor.pool-label"}}',
                str(service_name),
            ).strip()
        except ClusterDriverError:
            pass
        self._docker("service", "rm", str(service_name))
        if pool_label and all(
            character.isalnum() or character in "._-"
            for character in pool_label
        ):
            try:
                node_ids = self._docker("node", "ls", "-q").splitlines()
            except ClusterDriverError:
                node_ids = []
            for node_id in node_ids:
                try:
                    self.remove_node_label(node_id, pool_label)
                except ClusterDriverError:
                    continue

    @staticmethod
    def controller_hostname() -> str:
        return socket.gethostname()
