"""Swappable cluster drivers; Docker Swarm is the first supported engine."""

from __future__ import annotations

import json
import shutil
import socket
import subprocess
import time
from typing import Any, Dict, Optional

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


#: The context window every swarm LLM service runs (#138). One value, shared
#: with the sizing in `cluster_operations`, because the memory limit is
#: derived from a footprint measured at exactly this window with one slot —
#: the shipped single-node configuration (VD-076/VD-080).
SWARM_CONTEXT_TOKENS = 4096


class DockerSwarmDriver:
    """Head-controller operations for a Docker Swarm fleet."""

    name = "docker-swarm"

    def __init__(self, timeout: int = 120, package_manager=None, runner=None):
        self.timeout = max(10, min(int(timeout), 600))
        self.package_manager = package_manager or AptPackageManager(
            finder=lambda name: name
        )
        #: How Docker is reached. ``None`` runs ``docker`` directly, which is
        #: correct only for a process that holds the privilege itself — the
        #: workload executor. The control plane deliberately does not (#141):
        #: its ClusterManager passes the workload broker's ``run`` here, the
        #: same door every other Docker read in the product uses.
        self.runner = runner

    def _docker(self, *arguments: str) -> str:
        command = ["docker", *arguments]
        try:
            if self.runner is not None:
                # Tighter than the direct cap on purpose: the broker serves
                # one connection at a time, and a read that has not answered
                # in ten seconds is a hung daemon, not a slow one. Waiting the
                # full window would starve every other brokered read behind it.
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
        # Which process actually asked Docker: through a runner the query ran
        # inside the workload broker's daemon, so "this Vaelor process" would
        # name the wrong caller — the control plane's request was accepted.
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
        # The three reads below are issued verbatim from the shared constants,
        # so the broker's allowlist admits exactly what this method sends. All
        # three sit inside the same handler: a node or service read that fails
        # after a successful info read — broker timeout, broker restart,
        # allowlist drift — used to raise out of this method and reach the
        # /cluster route as an unhandled 500, which is the untruthful-failure
        # shape this whole area exists to eliminate. One failed read makes the
        # status unreadable, reported as such with the real failure text.
        try:
            raw = self._docker(*SWARM_INFO_COMMAND[1:])
            swarm = json.loads(raw)
            if not isinstance(swarm, dict):
                # `docker info` answered but had no Swarm section to report.
                # That used to reach `swarm.get(...)` and raise, turning a
                # readable engine into a 500.
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
                nodes = self._parse_json_lines(
                    self._docker(*SWARM_NODES_COMMAND[1:])
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
        hostname = transport.run(["uname", "-n"]).strip()
        return {
            "joined": True,
            "message": output[-300:],
            "hostname": hostname[:253],
            "swarm_node_id": self.node_id_by_hostname(hostname),
        }

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

    def remove_worker(
        self,
        transport: SshTransport,
        swarm_node_id: str,
        *,
        force: bool = False,
    ) -> Dict[str, Any]:
        self.set_node_availability(swarm_node_id, "drain")
        leave_error = ""
        try:
            arguments = ["docker", "swarm", "leave"]
            if force:
                arguments.append("--force")
            transport.run(arguments, sudo=True, timeout=180)
        except Exception as error:
            leave_error = str(error)
            if not force:
                raise ClusterDriverError(
                    "The worker could not leave cleanly. Reconnect it or review a forced removal."
                ) from error
        arguments = ["node", "rm"]
        if force:
            arguments.append("--force")
        arguments.append(str(swarm_node_id))
        self._docker(*arguments)
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
            # The same pinned engine the single-node deploy uses. A swarm
            # service resolving `:server` independently would put a different
            # build on each worker as the tag moved, which is the reproducibility
            # defect of #130 multiplied by the size of the pool.
            CPU_IMAGE,
            "-hf", f"{model_repo}:{model_file}",
            "--host", "0.0.0.0",
            "--port", "8080",
            # Stated, not inherited (#138): the memory limit this service runs
            # under is derived from a footprint measured at this window, so the
            # window must be this one by declaration rather than by the engine
            # default happening to match it.
            "--ctx-size", str(SWARM_CONTEXT_TOKENS),
            # **The image was inherited from the single-node deploy; the flags
            # that make it survivable were not.** Pinning the engine and then
            # running it with the engine's own defaults under a hard
            # `--limit-memory` is the worse half of both worlds.
            #
            # `--parallel` first, because it is the one that took the appliance
            # down: unstated, llama.cpp allocates the window once per slot and
            # defaults to four, so a limit computed for one context is enforced
            # against four copies of it (#109). This is a served, LAN-reachable
            # surface, which is exactly the case that measurement came from.
            "--parallel", "1",
            # And the prompt cache, from the same bound the compose path uses.
            # The engine default is 8192 MiB - larger than a Pi worker's entire
            # RAM - and unbounded it grew ~13 MB per prompt with no plateau in
            # 40 (VD-080). `tests/test_prompt_cache_bound.py` said "every
            # compose carries the bound" while covering one of two renderers.
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
        node_label: str,
        image: str,
        container_port: int,
        published_port: int,
        memory_limit_mib: int,
        template_id: str,
        volume: Optional[tuple[str, str]] = None,
        extra_ports: Optional[list[tuple[str, str, int]]] = None,
    ) -> Dict[str, Any]:
        safe_name = "".join(
            character
            for character in str(name).lower()
            if character.isalnum() or character == "-"
        )[:40]
        service_name = f"vaelor-app-{safe_name}"
        command = [
            "service", "create",
            "--name", service_name,
            "--constraint", f"node.labels.vaelor.node_id=={node_label}",
            "--reserve-memory", f"{max(64, int(memory_limit_mib * 0.75))}M",
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
        return {
            "service_id": service_id,
            "name": service_name,
            "template_id": template_id,
            "port": int(published_port),
            "memory_limit_mib": int(memory_limit_mib),
            "node_id": node_label,
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
            # `--filter name=` is a SUBSTRING match, so a service that is a
            # prefix of another (vaelor-llm-a / vaelor-llm-abc) returns several
            # rows. Emit the name alongside the replicas and select the EXACT
            # match, so a healthy service is never failed for a sibling's line
            # never matching the equality check (#Recovery-8).
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
            # {{.Replicas}} trails a placement annotation for a service created
            # with --replicas-max-per-node ("1/1 (max 1 per node)") - and the
            # appliance's own managed LLM services are. The ready count is the
            # leading whitespace token; compare on that, not the whole field,
            # so a healthy such service is not polled to the deadline and
            # reported failed (#Recovery-8b). The full string stays in `last`
            # for the error message.
            count = last.split()[0] if last.split() else ""
            if count == f"{expected}/{expected}":
                return {"ready": True, "replicas": last}
            if count.endswith(f"/{expected}"):
                time.sleep(2)
                continue
            time.sleep(2)
        raise ClusterDriverError(
            "The service did not reach {} healthy replica{} (last state: {}).".format(
                expected,
                "" if expected == 1 else "s",
                last or "missing",
            )
        )

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
        service = raw[0]
        spec = service.get("Spec", {})
        task = spec.get("TaskTemplate", {})
        container = task.get("ContainerSpec", {})
        endpoint = spec.get("EndpointSpec", {})
        mode = spec.get("Mode", {})
        resources = task.get("Resources", {})
        update = spec.get("UpdateConfig", {})
        rollback = spec.get("RollbackConfig", {})
        labels = {
            str(key): str(value)
            for key, value in (spec.get("Labels", {}) or {}).items()
            if str(key).startswith("vaelor.")
        }
        mounts = []
        for mount in container.get("Mounts", []) or []:
            if not isinstance(mount, dict):
                continue
            mounts.append({
                "type": str(mount.get("Type", "")),
                "source": str(mount.get("Source", ""))[:160],
                "target": str(mount.get("Target", ""))[:160],
                "read_only": bool(mount.get("ReadOnly")),
            })
        ports = []
        for port in endpoint.get("Ports", []) or []:
            if not isinstance(port, dict):
                continue
            ports.append({
                "published": int(port.get("PublishedPort", 0) or 0),
                "target": int(port.get("TargetPort", 0) or 0),
                "protocol": str(port.get("Protocol", "tcp")),
                "mode": str(port.get("PublishMode", "ingress")),
            })
        desired_replicas = int(
            (mode.get("Replicated", {}) or {}).get("Replicas", 1) or 1
        )
        tasks = self._parse_json_lines(self._docker(
            "service", "ps", name, "--no-trunc", "--format",
            SERVICE_TASKS_FORMAT,
        ))
        return {
            "id": str(service.get("ID", "")),
            "name": str(spec.get("Name", name)),
            "image": str(container.get("Image", "")).split("@", 1)[0],
            "labels": labels,
            "constraints": [
                str(value)[:240]
                for value in (task.get("Placement", {}) or {}).get(
                    "Constraints", []
                )
            ],
            "mounts": mounts,
            "ports": ports,
            "desired_replicas": desired_replicas,
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
        self._docker(
            "service", "update",
            "--image", image,
            "--force",
            "--update-parallelism", "1",
            "--update-failure-action", "rollback",
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
        self._docker(
            "service", "update",
            "--force",
            "--update-parallelism", "1",
            "--update-failure-action", "rollback",
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
        if not 128 <= limit <= 131072:
            raise ClusterDriverError(
                "Choose a memory limit from 128 MiB to 128 GiB."
            )
        if not 64 <= reservation <= limit:
            raise ClusterDriverError(
                "Memory reservation must be at least 64 MiB and no larger "
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
        self._docker(
            "service", "update",
            "--replicas", str(desired),
            "--limit-memory", f"{limit}M",
            "--reserve-memory", f"{reservation}M",
            "--update-parallelism", str(parallelism),
            "--update-order", order,
            "--update-failure-action", "rollback",
            "--rollback-parallelism", "1",
            name,
        )
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
