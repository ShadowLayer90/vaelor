"""Dependency composition for the authenticated control plane."""

from __future__ import annotations

from typing import Any, Callable, Dict, Optional

from .agent_api import AgentApiTokenStore, create_agent_api_blueprint
from .agent_task_runner import AgentTaskRunner
from .agent_tasks import AgentTaskStore
from .api_v2 import create_api_v2_blueprint
from .appliance_recovery import (
    FactoryResetPlans, PortableImportPlans, UninstallPlans,
)
from .application_deployments import ApplicationDeploymentStore
from .application_features import application_features
from .application_research_capability import application_research_capability
from .application_intent_refinement import ApplicationIntentRefiner
from .application_learning import ApplicationLearningStore
from .application_research_server import ApplicationResearchClient
from .application_research_intelligence import ApplicationResearchIntelligence
from .application_search import SearxSearchClient
from .application_validation import validate_application_compose
from .assistant_memory import AssistantMemoryStore
from .assistant_memory_reconciler import MemoryReconciler
from .assistant_reconciler_scheduler import (
    MemoryReconcileScheduler,
    any_tier_loaded,
    registry_probe_reader,
)
from .assistant_slot_cache import AssistantSlotCache
from .alert_channels import AlertChannelStore, build_delivery_callback
from .assistant_skills import AssistantSkillStore
from .assistant_tools import AssistantToolRegistry
from .automations import AutomationRunner, AutomationStore
from .backup_schedule import (
    BackupScheduleStore,
    BackupScheduler,
    launch_backup_autostart,
)
from .checkpoints import CheckpointInventory
from .chat_inference import ChatInference
from .cluster_driver import DockerSwarmDriver
from .cluster_manager import ClusterManager
from .cluster_backups import ClusterBackupStore
from .cluster_operations import ClusterOperations
from .copilot_setup import hardware_inventory
from .credential_broker import CredentialBrokerClient
from .custom_agents import CustomAgentStore
from .custom_connector_runtime import ConnectorRuntime
from .integration_runtime import IntegrationRuntime
from .deployment_agent import DeploymentAgent
from .inference_client import remote_inference_budget
from .model_connection import assistant_model_configured
from .docker_health import container_runtime_healthy
from .fan_control import CpuFanController
from .web_research import SEARCH_URL, WebResearchManager
from .host_desktop import HostDesktopClient
from .inference_gateway import (
    create_inference_gateway_blueprint,
    inference_gateway_status,
)
from .inference_metrics import InferenceGatewayMetrics
from .jobs import JobStore, workload_capabilities
from .kvm import KvmCapabilityProbe, KvmControlStore
from .mcp_client import ExternalMcpTools, configured_servers
from .platform_drivers import default_platform_drivers
from .rag_chat import RagChatStore
from .release_source import default_release_source
from .security import SecurityStore
from .subagents import SubagentCoordinator
from .system_inventory import SystemInventory
from .vnc_gateway import HostRemoteDesktopProbe, VncSessionStore
from .workload_act_grants import WorkloadActGrantStore
from .workload_files import AppFileBrowser
from .workload_inventory import WorkloadInventory
from .workload_dependencies import WorkloadDependencyService
from .runtime_paths import data_path


def _numeric(value: Any) -> Optional[float]:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        return None
    return float(value)


def _fan_rpm(data: Dict[str, Any]) -> Optional[float]:
    """The highest live fan speed in RPM, or None when no tachometer reads.

    The fan-speed key is platform-specific and there is no single one to read:
    the Pironman Pi enclosure publishes a scalar ``pwm_fan_speed``, while an x86
    workstation publishes one or more labelled fans under ``wmi_fans`` (see
    wmi_sensors) and has no ``pwm_fan_speed`` at all. Reading only the Pi key
    made a healthy workstation - three fans spinning - look like a stopped fan,
    a false ``fan_failure`` on any hot idle (#Recovery-10, confirmed live: the
    Z2 box reports wmi_fans and no pwm_fan_speed). A missing sensor is not a
    reading of zero, so no reading at all returns None rather than 0.
    """
    readings = []
    direct = _numeric(data.get("pwm_fan_speed"))
    if direct is not None:
        readings.append(direct)
    fans = data.get("wmi_fans")
    if isinstance(fans, list):
        for fan in fans:
            if isinstance(fan, dict):
                rpm = _numeric(fan.get("rpm"))
                if rpm is not None:
                    readings.append(rpm)
    if not readings:
        return None
    return max(readings)


def _fan_faulted(data: Dict[str, Any]) -> bool:
    """Whether any labelled fan explicitly reports a hardware fault.

    ``wmi_fans`` entries carry a ``fault`` flag the producer states is real
    evidence of a stopped fan (wmi_sensors: "only a fan reporting a fault, or
    one that never reads at all, is evidence"). ``_fan_rpm`` collapses to the
    highest reading, which hides a single faulted fan behind its healthy
    siblings on a multi-fan box - so read the flag directly. Only an explicit
    ``fault is True`` counts; a bare zero or an absent reading does NOT, that
    being the idle case #Recovery-10 deliberately stopped alarming on.
    """
    fans = data.get("wmi_fans")
    if not isinstance(fans, list):
        return False
    return any(isinstance(fan, dict) and fan.get("fault") is True for fan in fans)


class ControlPlaneRuntime:
    """Own long-lived services and register their HTTP adapters."""

    def __init__(self, app, callbacks: Dict[str, Callable[..., Any]]):
        self.platform_drivers = default_platform_drivers()
        self.current_data = lambda: self.platform_drivers[
            "telemetry_provider"
        ].snapshot(callbacks["current_data"]())
        # Both seams were added to this controller and then never connected:
        # constructing it bare left `absent_message` as the generic default and
        # `temperature_reader` as None, so the fan logic kept reading
        # `thermal_zone0` - the ~35 °C acpitz zone that `linux_sensors` was
        # written to replace - on the very machines that replacement was for.
        self.cpu_fan = self._build_cpu_fan()
        try:
            system_config = (callbacks["read_config"]() or {}).get("system", {})
            if system_config.get("cpu_fan_mode") == "custom":
                self.cpu_fan.set_mode("custom", curve=system_config.get("cpu_fan_curve"))
        except (AttributeError, OSError, PermissionError, RuntimeError, ValueError):
            # Hardware discovery must never prevent the dashboard from starting.
            pass
        self.jobs = JobStore()
        self.credential_broker = CredentialBrokerClient()
        self.deployment_agent = DeploymentAgent(
            timeout_seconds=remote_inference_budget(),
            credential_broker=CredentialBrokerClient(timeout_seconds=3)
        )
        self.application_intent_refiner = ApplicationIntentRefiner(
            self.deployment_agent.connection, timeout_seconds=45,
        )
        self.memory = AssistantMemoryStore()
        # The Assistant model's saved KV prefix, keyed by conversation
        # (VD-076). Sharing the memory store's database is what lets a deleted
        # conversation cascade its slot mapping away.
        self.slot_cache = AssistantSlotCache(self.memory)
        self.workloads = WorkloadInventory(credential_broker=self.credential_broker)
        self.vnc_sessions = VncSessionStore()
        self.host_remote_desktop = HostRemoteDesktopProbe(
            remote_access_provider=self.platform_drivers[
                "remote_access_provider"
            ]
        )
        self.custom_agents = CustomAgentStore(credential_broker=self.credential_broker)
        self.connector_runtime = ConnectorRuntime(self.credential_broker)
        self.integrations = IntegrationRuntime(
            self.workloads, self.custom_agents, self.credential_broker
        )
        try:
            self.integrations.reconcile()
        except (OSError, RuntimeError, TypeError, ValueError):
            # Integration authorization refreshes again and fails closed before use.
            pass
        self.workload_dependencies = WorkloadDependencyService(
            self.workloads, self.credential_broker, self.custom_agents
        )
        self.rag_chat = RagChatStore()
        self.chat_inference = ChatInference(self.credential_broker)
        self.cluster = ClusterManager(broker=self.credential_broker)
        self.cluster_backups = ClusterBackupStore()
        # NOT `self.cluster.driver` (#141 review). The manager's driver reads
        # through the workload broker, whose allowlist admits reads only;
        # sharing it here would send this ClusterOperations' mutations —
        # eviction's `node update`/`node rm` among them — to a socket that
        # refuses them by design, in every deployment. These operations keep
        # the direct driver they always had; on the appliance their privilege
        # gap is a separate, pre-existing matter (task #153).
        self.cluster_operations = ClusterOperations(
            store=self.cluster.store,
            broker=self.credential_broker,
            driver=DockerSwarmDriver(timeout=600),
            backup_store=self.cluster_backups,
        )
        # Default-inert acting grants (VD-100 #96): empty until an administrator
        # grants workloads:act. It gates both the operator's Assistant chat and
        # the pinned envelope an installed agent may propose within.
        self.workload_act_grants = WorkloadActGrantStore()
        self.tasks = AgentTaskStore(
            profile_store=self.custom_agents,
            app_grant_context=self.integrations,
            workload_act_grants=self.workload_act_grants,
        )
        self.skills = AssistantSkillStore()
        self.automations = AutomationStore(profile_store=self.custom_agents)
        self._automation_security = None
        self.system = SystemInventory(
            storage_provider=self.platform_drivers["storage_provider"],
            service_catalog=self.platform_drivers[
                "operating_system"
            ].managed_services(),
            package_manager=self.platform_drivers["package_manager"],
        )
        self.kvm_capabilities = KvmCapabilityProbe()
        self.kvm_control = KvmControlStore()
        self.checkpoints = CheckpointInventory()
        self.agent_api_tokens = AgentApiTokenStore()
        self.inference_metrics = InferenceGatewayMetrics()
        self.factory_reset_plans = FactoryResetPlans()
        self.uninstall_plans = UninstallPlans()
        self.portable_import_plans = PortableImportPlans()
        self.application_deployments = ApplicationDeploymentStore()
        self.application_learning = ApplicationLearningStore()
        self.application_research = ApplicationResearchClient()
        self.application_research_intelligence = ApplicationResearchIntelligence(
            self.application_research,
            self.deployment_agent.connection,
            search_client=SearxSearchClient(SEARCH_URL),
            # #247r: auto-provision guarded web research on demand for the
            # custom-agent search tool (below) and any synchronous research call.
            web_research=WebResearchManager(docker_healthy=container_runtime_healthy),
            timeout_seconds=60,
        )
        self.host_desktop_broker = HostDesktopClient(timeout=45)
        self.tools = AssistantToolRegistry(
            {
                "device_info": callbacks["device_info"],
                "current_data": self.current_data,
                "read_config": callbacks["read_config"],
                "cpu_fan_status": self.cpu_fan.snapshot,
                # The Assistant reads the machine through the same selected
                # driver every other surface uses, so it cannot disagree with
                # /api/v2/system/machine about what hardware is fitted.
                "platform_drivers": self.platform_drivers,
                "hardware_inventory": hardware_inventory,
                "deployment_agent": self.deployment_agent,
                "chat_inference": self.chat_inference,
                "telemetry_history": callbacks.get("telemetry_history"),
                "telemetry_history_range": callbacks.get("telemetry_history_range"),
                "workload_capabilities": lambda: workload_capabilities(
                    self.platform_drivers
                ),
                "workload_inventory": self.workloads,
                "job_store": self.jobs,
                "checkpoints": self.checkpoints,
                "assistant_memory": self.memory,
                "system_inventory": self.system,
                "cluster_summary": self.cluster.summary,
                # #247r: guarded_search auto-provisions the owned SearXNG backend
                # on demand (or surfaces a deliberate disable) before searching,
                # so an internet-parsing custom agent no longer dead-ends when
                # web research was never manually enabled.
                "public_search": lambda query: (
                    self.application_research_intelligence.guarded_search([query])
                ),
                "public_fetch": self.application_research.fetch_evidence,
            }
        )
        # External MCP servers extend the tool surface without extending
        # appliance authority: they carry their own scope, and the approval
        # gate is resolved live against the account's current role rather than
        # against whatever it held when the server was enrolled.
        self.external_mcp = ExternalMcpTools(
            configured_servers(),
            approval=lambda _server, _tool, actor: self._actor_is_administrator(actor),
        )
        self.subagents = SubagentCoordinator(
            self.tasks,
            self.tools,
            self.deployment_agent,
            self.skills,
            self.memory,
            self.rag_chat,
        )
        self.task_runner = AgentTaskRunner(
            self.tasks,
            self.tools,
            self.deployment_agent,
            self.skills,
            self.memory,
            knowledge_store=self.rag_chat,
            connector_runtime=self.connector_runtime,
            app_capability_broker=self.integrations.broker,
            administrator_resolver=self._actor_is_administrator,
            # Ground custom-agent runs in the actor-scoped, sanitized operational
            # lessons the deployment learning store records (which fed no prompt
            # before), read-only. Curated appliance memory is deliberately NOT
            # wired: it is stored global, so feeding it to agents needs an
            # explicit per-agent memory permission that does not exist yet.
            learning_store=self.application_learning,
        )
        self.task_runner.start()
        # Alert delivery: a fired trigger reaches a human out-of-band (email /
        # webhook). Secrets live only in the broker, resolved per delivery by
        # purpose; the store holds channel config and last-delivery outcomes.
        self.alert_channels = AlertChannelStore()

        def _resolve_alert_secret(purpose):
            lease = self.credential_broker.resolve_active(purpose)
            return lease.get("token") if isinstance(lease, dict) else None

        self.automation_runner = AutomationRunner(
            self.automations,
            self.tasks,
            context_provider=self._automation_signals,
            owner_authorized=self._automation_owner_authorized,
            deliver=build_delivery_callback(
                self.alert_channels, _resolve_alert_secret
            ),
        )
        self.automation_runner.start()
        # Scheduled + off-site backups, built on the portable-state archive
        # primitive. The scheduler resolves the archive passphrase from the
        # broker per run and supervises itself (restart-on-failure, not just
        # boot); off-site delivery is best-effort and never deletes the local
        # archive on failure.
        self.backup_schedule_store = BackupScheduleStore()
        self.backup_scheduler = BackupScheduler(
            self.backup_schedule_store,
            self.portable_import_plans.portable_state,
            export_root=self.portable_import_plans.export_root,
            secret_resolver=lambda purpose: self.credential_broker.resolve_active(
                purpose
            )["token"],
        )
        launch_backup_autostart(self.backup_scheduler)
        # The down-cycle memory reconciler (VD-101 #97). Built and committed but
        # unscheduled until now - the same "no production caller" class VD-100
        # names. Its probe is the read tool registry; it sweeps only when no
        # inference tier is loaded, and it flags, never deletes (LESSONS 8).
        self.memory_reconciler = MemoryReconciler(
            self.memory, registry_probe_reader(self.tools)
        )
        self.reconciler_scheduler = MemoryReconcileScheduler(
            self.memory_reconciler, tier_loaded=self._inference_tier_loaded,
        )
        self.reconciler_scheduler.start()
        app.register_blueprint(create_api_v2_blueprint(self._api_callbacks(callbacks)))
        app.register_blueprint(
            create_agent_api_blueprint(
                self.agent_api_tokens,
                self.deployment_agent,
                self._external_agent_context,
            )
        )
        app.register_blueprint(
            create_inference_gateway_blueprint(
                self.agent_api_tokens, self.credential_broker,
                self.inference_metrics,
            )
        )

    def _build_cpu_fan(self) -> CpuFanController:
        """Construct the fan controller with both of its seams connected.

        A separate method because ``__init__`` is otherwise untestable without
        standing up every service on the runtime, and "the seams are wired" is
        exactly the thing that regressed: they were added to the controller and
        then it was constructed bare, so the generic absent message and the
        acpitz temperature were what shipped.
        """
        return CpuFanController(
            absent_message=self._cpu_fan_absent_message(),
            temperature_reader=self._labelled_cpu_temperature,
        )

    def _cpu_fan_absent_message(self) -> str:
        """The platform driver's own reason, not a sentence about a Pi.

        Surfaced verbatim by the API, so on a machine with no controllable fan
        it must name *that* machine's reason. The driver already writes one.
        """
        from .fan_control import DEFAULT_ABSENT_MESSAGE

        try:
            driver = self.platform_drivers["hardware"]
            record = (driver.capabilities() or {}).get("cpu_fan") or {}
            return str(record.get("reason") or "") or DEFAULT_ABSENT_MESSAGE
        except (AttributeError, KeyError, OSError, RuntimeError, TypeError, ValueError):
            return DEFAULT_ABSENT_MESSAGE

    def _labelled_cpu_temperature(self):
        """The labelled CPU sensor, falling back to nothing rather than acpitz.

        `thermal_zone0` on an AMD workstation is the acpitz zone, which reads
        about 35 °C while the package is at 60 — driving a fan curve from it
        means the curve never engages. `linux_sensors.cpu_temperature` finds the
        labelled k10temp/coretemp reading instead, and reports its provenance.
        """
        from .linux_sensors import cpu_temperature

        try:
            reading = cpu_temperature()
        except (OSError, ValueError):
            return None
        celsius = reading.get("celsius")
        return float(celsius) if isinstance(celsius, (int, float)) else None

    def _actor_is_administrator(self, actor: str) -> bool:
        """Resolve an account's role now, against the live user table.

        Gating an endpoint stops new work; it does nothing about work already in
        a queue, or about the account that asked for it being demoted
        afterwards. This is read when the unattended run happens, against the
        same user table the request path uses, so revoking access revokes the
        unattended runs too. Its own store instance reads the same SQLite file
        the API writes; it never creates sessions. An unreadable account table
        is not an administrator - it is no answer, and the caller gets the
        closed one.
        """
        try:
            if self._automation_security is None:
                self._automation_security = SecurityStore()
            return any(
                item["username"] == actor
                and item["enabled"]
                and item["role"] == "administrator"
                for item in self._automation_security.list_users()
            )
        except Exception:
            self._automation_security = None
            return False

    def _automation_owner_authorized(self, actor: str) -> bool:
        """A schedule or alert rule runs only while its owner is still an
        administrator."""
        return self._actor_is_administrator(actor)

    def _inference_tier_loaded(self) -> bool:
        """True while any inference engine holds a model - the down-cycle signal.

        Reuses `inference.status`'s own per-engine truth (LESSONS 6: one
        definition of "loaded", not a second one here). A probe that raises is
        the scheduler's problem, not this method's; it returns a plain bool and
        the scheduler treats a raising signal as "loaded" to stay off a busy box.
        """
        return any_tier_loaded(self.tools.run("inference.status")["result"])

    def _automation_signals(self) -> Dict[str, float]:
        data = self.tools.callbacks["current_data"]()
        disk_values = [
            float(value)
            for key, value in data.items()
            if str(key).startswith("disk_")
            and str(key).endswith("_percent")
            and isinstance(value, (int, float))
        ]
        services = self.system.services()
        temperature = float(data.get("cpu_temperature") or 0)
        rpm = _fan_rpm(data)
        faulted = _fan_faulted(data)
        return {
            "cpu_temperature": temperature,
            "memory_percent": float(data.get("memory_percent") or 0),
            "storage_percent": max(disk_values, default=0),
            "service_failures": float(
                sum(
                    1
                    for item in services
                    if item.get("available") and item.get("active") != "active"
                )
            ),
            # Only a hot box with a fan we can READ reading zero, or one that
            # explicitly reports a hardware fault, is a failure. An ABSENT
            # reading is not a stopped fan (a missing sensor is not a reading of
            # zero), so an unknown rpm never fires the alarm; the fault flag
            # catches a single stopped fan even when a sibling still spins and
            # the max reading looks healthy.
            "fan_failure": float(
                temperature >= 65
                and ((rpm is not None and rpm <= 0) or faulted)
            ),
        }

    def _api_callbacks(
        self, callbacks: Dict[str, Callable[..., Any]]
    ) -> Dict[str, Any]:
        return {
            **callbacks,
            "current_data": self.current_data,
            "cpu_fan_status": self.cpu_fan.snapshot,
            "cpu_fan_update": self.cpu_fan.set_mode,
            "job_store": self.jobs,
            "workload_capabilities": lambda: workload_capabilities(
                self.platform_drivers
            ),
            "deployment_agent": self.deployment_agent,
            "assistant_memory": self.memory,
            "assistant_slot_cache": self.slot_cache,
            "assistant_tools": self.tools,
            "external_mcp_tools": self.external_mcp,
            "agent_tasks": self.tasks,
            "workload_act_grants": self.workload_act_grants,
            "custom_agents": self.custom_agents,
            "connector_runtime": self.connector_runtime,
            "app_capability_registry": self.integrations.registry,
            "integration_connections": self.integrations.connections,
            "agent_app_grants": self.integrations.grants,
            "app_capability_broker": self.integrations.broker,
            "integration_connection_test": self.integrations.test_connection,
            "integration_reconcile": self.integrations.reconcile,
            "rag_chat": self.rag_chat,
            "chat_inference": self.chat_inference,
            "subagents": self.subagents,
            "assistant_skills": self.skills,
            "automations": self.automations,
            "alert_channels": self.alert_channels,
            "backup_scheduler": self.backup_scheduler,
            "system_inventory": self.system,
            "hardware_inventory": hardware_inventory,
            "credential_broker": self.credential_broker,
            "workload_inventory": self.workloads,
            "app_file_browser": AppFileBrowser(self.workloads),
            "workload_dependencies": self.workload_dependencies,
            "vnc_sessions": self.vnc_sessions,
            "host_remote_desktop": self.host_remote_desktop,
            "host_vnc": self.host_remote_desktop.vnc,
            "kvm_capabilities": self.kvm_capabilities,
            "kvm_control": self.kvm_control,
            "checkpoints": self.checkpoints,
            "agent_api_tokens": self.agent_api_tokens,
            "inference_gateway_status": lambda: inference_gateway_status(
                self.credential_broker, self.inference_metrics
            ),
            "factory_reset_plans": self.factory_reset_plans,
            "uninstall_plans": self.uninstall_plans,
            "portable_import_plans": self.portable_import_plans,
            "application_deployment_store": self.application_deployments,
            "application_learning_store": self.application_learning,
            "application_intent_refiner": self.application_intent_refiner.refine,
            "application_features": lambda: application_features(
                assistant_model_configured(self.credential_broker)
            ),
            "application_research_manifest": (
                self.application_research_intelligence.research_manifest
            ),
            "application_research_capability": lambda request, **details: (
                application_research_capability(
                    request, self.deployment_agent.connection(), **details
                )
            ),
            "application_compose_validator": lambda compose: (
                validate_application_compose(
                    compose,
                    data_path("workloads"),
                    hardware_inventory(),
                )
            ),
            "cluster_manager": self.cluster,
            "cluster_backups": self.cluster_backups,
            "cluster_operations": self.cluster_operations,
            "host_desktop_broker": self.host_desktop_broker,
            "platform_drivers": self.platform_drivers,
            # The appliance-upgrade routes read the GitHub release source here;
            # without it they fall back to the offline StubReleaseSource.
            "release_source": default_release_source(),
        }

    def _external_agent_context(self) -> Dict[str, Any]:
        return {
            "hardware": hardware_inventory(),
            "workloads": self.workloads.list_all(),
            "system": self.system.snapshot(),
            "policy": {
                "external_api": True,
                "direct_mutations": False,
                "approval_required": True,
            },
        }
