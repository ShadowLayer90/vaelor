"""Dependency composition for the authenticated control plane."""

from __future__ import annotations

import logging
from typing import Any, Callable, Dict, List, Optional, Tuple

from .agent_api import AgentApiTokenStore
from .agent_deployments import AgentDeploymentStore
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
from .app_port_claims import model_port_holders
from .application_validation import validate_application_compose
from .assistant_memory import AssistantMemoryStore
from .assistant_memory_reconciler import MemoryReconciler
from .assistant_reconciler_scheduler import (
    MemoryReconcileScheduler,
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
from .chat_inference import AI_CHAT_TRACE_SERVICE_NAME, ChatInference
from .cluster_driver import DockerSwarmDriver
from .cluster_manager import ClusterManager
from .gpu_cluster_mode import ClusterModeStore
from .gpu_idle_watch import (
    deployment_idle_targets,
    enqueue_cluster_load,
    read_target_body,
    wake_unloaded_cluster,
)
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
    gateway_status_reader,
)
from .inference_metrics import InferenceGatewayMetrics
from .llm_server_wake import start_control_plane_wake_door
from .model_residency import appliance_residency
from .usage_meter import UsageMeter
from .usage_collection import UsageCollection
from .inference_tracing import TraceEmitter
from .hardware_bridge_client import HardwareBridgeClient
from .performance_snapshot_source import assistant_performance_snapshot, performance_callbacks
from .phoenix_state import PhoenixStore, trace_endpoint
from .jobs import JobStore, workload_capabilities
from .kvm import KvmCapabilityProbe, KvmControlStore
from .mcp_client import ExternalMcpTools, configured_servers
from .mcp_catalog import AgentMcpGrantStore, McpCatalogStore
from .skills_library import SkillAttachmentStore, SkillsLibraryStore
from .platform_drivers import default_platform_drivers
from .rag_chat import RagChatStore
from .release_source import default_release_source
from .security import SecurityStore
from .cluster_worker_telemetry import AUTOMATIC_REPAIR_ACTOR, audit_reconcile
from .serving_metrics_poller import ServingMetricsPoller, VllmServingPoller
from .subagents import SubagentCoordinator
from .system_inventory import SystemInventory
from .telemetry_ingest_status import worker_is_reporting
from .telemetry_store import (
    MAX_HISTORY_BUCKETS,
    REPORTING_WINDOW_SECONDS,
    TelemetryStoreError,
)
from .vnc_gateway import HostRemoteDesktopProbe, VncSessionStore
from .workload_act_grants import WorkloadActGrantStore
from .workload_files import AppFileBrowser
from .workload_inventory import WorkloadInventory
from .workload_dependencies import WorkloadDependencyService
from .cluster_worker_profile import start_profile_sweep
from .worker_telemetry_reconcile_scheduler import (
    WorkerTelemetryReconcileScheduler,
)
from .runtime_paths import data_path

LOGGER = logging.getLogger(__name__)


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


#: How recently a worker must have reported for its telemetry to arm an alert
#: rule: the one reporting window every surface shares (ACC-128), not a number
#: of its own. Past it the last reading is stale, and firing a threshold on a
#: value the machine has since moved away from is a false alarm on data we no
#: longer have. A worker older than this is OMITTED from the per-node signals so
#: its rules skip - exactly when the Fleet card says "not reporting".
WORKER_SIGNAL_MAX_AGE_SECONDS = REPORTING_WINDOW_SECONDS

#: The window read from a worker's telemetry history to recover its latest
#: reading. Wide enough to contain a fresh sample comfortably above the sampling
#: interval; the freshness gate above, not this span, is what decides staleness.
WORKER_SIGNAL_WINDOW_SECONDS = 900

#: The alert signals a worker's telemetry can raise (TRIGGER_SOURCES marked
#: ``worker``). A worker also reports disk, network and fan readings (VD-205
#: item 6), but no worker alert is built on them yet, so storage, service
#: failures and the fan signal stay controller-only alerts and create_trigger
#: refuses them for a worker.
WORKER_SIGNAL_FIELDS = ("cpu_temperature", "memory_percent")


def _measured(value: Any) -> Optional[float]:
    """A reading as a float, or None when nothing was measured."""
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        return None
    return float(value)


def _controller_signals(data: Dict[str, Any], services) -> Dict[str, float]:
    """The controller's own readings for the alert signals it actually measured.

    Split out from the per-node assembly so the controller's signal derivation —
    the fan logic #Recovery-10 hardened in particular — has one home and can be
    tested directly, and so a worker (which carries only two of the five) shares
    exactly the cpu/memory derivation rather than a second copy of it.

    A signal with no reading is LEFT OUT, never written as 0 (ACC-129): the
    engine skips a rule whose signal is absent, while a 0 let an "at or below"
    rule fire on a sensor that does not exist, the same way a worker that is
    not reporting is omitted rather than zeroed.
    """
    disk_values = [
        float(value)
        for key, value in data.items()
        if str(key).startswith("disk_")
        and str(key).endswith("_percent")
        and isinstance(value, (int, float))
        and not isinstance(value, bool)
    ]
    signals: Dict[str, float] = {
        "service_failures": float(
            sum(
                1
                for item in services
                if item.get("available") and item.get("active") != "active"
            )
        ),
    }
    temperature = _measured(data.get("cpu_temperature"))
    memory = _measured(data.get("memory_percent"))
    if temperature is not None:
        signals["cpu_temperature"] = temperature
    if memory is not None:
        signals["memory_percent"] = memory
    if disk_values:
        signals["storage_percent"] = max(disk_values)
    if temperature is not None:
        rpm = _fan_rpm(data)
        faulted = _fan_faulted(data)
        # Only a hot box with a fan we can READ reading zero, or one that
        # explicitly reports a hardware fault, is a failure. An ABSENT
        # reading is not a stopped fan (a missing sensor is not a reading of
        # zero), so an unknown rpm never fires the alarm; the fault flag
        # catches a single stopped fan even when a sibling still spins and
        # the max reading looks healthy. With no temperature the "hot box"
        # half cannot be judged, so the signal is left out.
        signals["fan_failure"] = float(
            temperature >= 65
            and ((rpm is not None and rpm <= 0) or faulted)
        )
    return signals


def _capable_lease_active(deployment_agent: Any) -> bool:
    """True when the capable GPU (AI-Chat) model has a live lease.

    Fails closed: any broker trouble resolving the ``capable`` connection reads as
    unavailable, so the manual research-escalation route degrades honestly instead
    of promising a model that is not there.
    """
    try:
        return deployment_agent.connection("capable") is not None
    except Exception:  # noqa: BLE001 - availability probe must fail closed
        return False


class ControlPlaneRuntime:
    """Own long-lived services and register their HTTP adapters."""

    def __init__(self, app, callbacks: Dict[str, Callable[..., Any]]):
        self._host_callbacks = callbacks
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
        # ACC-106: the NPU models are listed too, with what the NPU is serving
        # read from the hardware bridge's status verb.
        self.workloads = WorkloadInventory(
            credential_broker=self.credential_broker,
            npu_status=lambda: HardwareBridgeClient().flm_status(),
        )
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
        # The agent-facing MCP catalog and its per-agent, version-pinned tool
        # grants (F1). Both share one SQLite file so grant creation reads the
        # server's live approved_tools in the same transaction. These feed only a
        # future agent runtime's OWN outbound client; they are never handed to the
        # inbound VaelorMcpServer(external=...) surface.
        self.mcp_catalog = McpCatalogStore()
        self.agent_mcp_grants = AgentMcpGrantStore()
        # The agent-facing skills/plugins library and its per-target,
        # version-pinned attach records (F2). Both share one SQLite file so an
        # attachment reads the skill's live manifest in the same transaction. An
        # attachment is advisory capability the F4 agent runtime still gates; it
        # never widens a target's own scope set.
        self.skills_library = SkillsLibraryStore()
        self.skill_attachments = SkillAttachmentStore()
        # The cluster agent deployment rows the executor writes. The control
        # plane reads them to turn an agent's memory bearer into its identity
        # (ACC-132): the memory routes had no store to resolve against, so every
        # agent memory call was a 401.
        self.agent_deployments = AgentDeploymentStore()
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
        # The never-pruned per-key usage meter of the inference gateway (Phase G,
        # VD-128), joined onto the API-key rows. How much each deployment's MODEL
        # did, and per-key LLM Server use, are the usage collection's
        # (`vaelor.usage_collection`, ACC-043/044), fed below.
        self.usage_meter = UsageMeter()
        self.usage = UsageCollection(
            self.credential_broker, lambda: self.cluster.store,
            bridge=lambda: self.hardware_bridge_client,
        )
        # The Phoenix trace-collector flag and the gateway's OTLP emitter (VD-128).
        # The emitter reads the flag afresh per request through `trace_endpoint`,
        # so enabling/disabling Phoenix takes effect with no gateway restart, and
        # a disabled or absent Phoenix makes the emitter a silent no-op. The bridge
        # client is the same loopback root socket the executor uses, here only for
        # the best-effort live "running" read on the Phoenix surface.
        self.phoenix_store = PhoenixStore()
        self.hardware_bridge_client = HardwareBridgeClient()
        self.inference_tracer = TraceEmitter(
            lambda: trace_endpoint(self.phoenix_store.read())
        )
        # AI Chat POSTs the cluster balancer directly and never reaches the
        # /inference/v1 gateway, so the gateway's collectors above never saw its
        # traffic (Observability Unit 2, items b + h). The RED store and usage
        # meter are injected into ChatInference unchanged, and a chat-scoped
        # trace emitter reuses the same Phoenix flag but stamps a distinct
        # service.name so AI-Chat spans file apart from the gateway's. Recording
        # is best-effort and only for the Mode B cluster-inference case.
        self.ai_chat_tracer = TraceEmitter(
            lambda: trace_endpoint(self.phoenix_store.read()),
            service_name=AI_CHAT_TRACE_SERVICE_NAME,
        )
        self.chat_inference = ChatInference(
            self.credential_broker,
            cluster_wake=lambda: self._cluster_wake("ai-chat"),
            tracer=self.ai_chat_tracer,
            inference_metrics=self.inference_metrics,
            usage_meter=self.usage_meter,
        )
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
                # The enrolled workers, whose GPU rows gpu.status reads (ACC-201).
                "cluster_manager": self.cluster,
                # The LLM Server's proxy status, read for the Assistant's answer.
                "hardware_bridge_client": self.hardware_bridge_client,
                "performance_snapshot": self.performance_snapshot,
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
            context_provider=self._automation_signals_by_node,
            owner_authorized=self._automation_owner_authorized,
            machine_names=self._alert_machine_names,
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
        # The serving-metrics scrapes (Phase E', Observability Unit 1): while
        # this machine's llama.cpp model serves on a loopback port, and while a
        # Mode B vLLM deployment serves (each replica read over the operations'
        # own transport, off any serving/bridge lock), /metrics is read every
        # ten seconds for the Performance card. Discovery and the store write
        # fail closed to "nothing", so a Pi or a box serving neither writes
        # nothing. The same bodies feed the usage ledger through `usage_sink`:
        # the model's own counters, so every door is counted (ACC-044).
        self.serving_metrics_poller = ServingMetricsPoller(
            endpoint_source=self._serving_metrics_endpoint,
            record=callbacks.get("serving_metrics_record"),
            usage_sink=self.usage.local_sink,
        )
        self.serving_metrics_poller.start()
        self.vllm_serving_metrics_poller = VllmServingPoller(
            targets_source=self._vllm_serving_targets,
            scrape_body=lambda node_id, port: read_target_body(
                self.cluster_operations.gpu_operations, node_id, port
            ),
            record=callbacks.get("serving_metrics_record"),
            usage_sink=self.usage.cluster_sink,
        )
        self.vllm_serving_metrics_poller.start()
        # Per-key LLM Server use, drained from the gate's access log (ACC-043).
        self.usage.start()
        # The worker-telemetry self-heal cadence (B-scope). The reconcile
        # that revives a stale or down worker agent had only the operator
        # Recheck route to call it, so a drifted worker never recovered
        # unattended - the VD-100 inert-mechanism class. This slow daemon
        # loop walks the telemetry-provisioned joined workers and drives the
        # cluster manager's reconcile for each, per node isolated so one
        # unreachable box cannot stall the rest; the reconcile is idempotent
        # (it no-ops a healthy agent), so repeating the walk cannot thrash.
        self.worker_telemetry_reconciler = WorkerTelemetryReconcileScheduler(
            list_workers=lambda: [
                node_id for node_id, _hash in self.cluster.store.ingest_key_hashes()
            ],
            reconcile=self._audited_telemetry_repair,
        )
        self.worker_telemetry_reconciler.start()
        # VD-194 P2: Recheck and this 15-minute check queue the worker profile update.
        self.worker_profile_sweep = start_profile_sweep(self.cluster, self.jobs)
        app.register_blueprint(create_api_v2_blueprint(self._api_callbacks(callbacks)))
        app.register_blueprint(
            create_inference_gateway_blueprint(
                self.agent_api_tokens, self.credential_broker,
                self.inference_metrics, self.inference_tracer,
                cluster_wake=lambda: self._cluster_wake("cluster-inference"),
                meter=self.usage_meter,
                # The same check without the load (ACC-057): a model listing
                # answers from the lease and never wakes the model.
                cluster_resting=lambda: self._cluster_wake("cluster-inference", wake=False),
            )
        )
        # ACC-058: the LLM Server door's upstream while the cluster model is
        # idle-unloaded; started only where the gate can run.
        self.llm_server_wake = start_control_plane_wake_door(self)

    def _audited_telemetry_repair(self, node_id: str) -> Dict[str, Any]:
        """One worker's 15-minute telemetry self-heal, recorded in Activity.

        ACC-121: the sweep re-keyed and reinstalled agents with no audit entry
        while the Activity page promises every audited change. A change is
        recorded under `AUTOMATIC_REPAIR_ACTOR`; a failure only when the node
        starts failing, so an unplugged worker is one entry, not one every 15
        minutes. The same `audit_reconcile` rule the Recheck route applies.
        """
        failing = getattr(self, "_repair_failing", None)
        if failing is None:
            failing = self._repair_failing = set()
        # An audit store that cannot be opened must not stop the repair
        # itself; it is logged and the repair runs unrecorded (review nit).
        audit = None
        try:
            if getattr(self, "_repair_security", None) is None:
                self._repair_security = SecurityStore()
            audit = self._repair_security.audit
        except Exception:  # noqa: BLE001 - the repair still runs
            LOGGER.warning("the telemetry repair cannot reach the audit trail")
        try:
            result = self.cluster.reconcile_worker_telemetry(node_id)
        except Exception as error:
            if node_id not in failing:
                failing.add(node_id)
                audit_reconcile(audit, AUTOMATIC_REPAIR_ACTOR, node_id, error=error)
            raise
        failing.discard(node_id)
        audit_reconcile(audit, AUTOMATIC_REPAIR_ACTOR, node_id, result=result)
        return result

    def _cluster_wake(self, purpose: str, wake: bool = True) -> str:
        """G3b (BL-1/CN-4): wake a scale-to-zero cluster model a lease names.

        The seam AI Chat (``ai-chat``) and the external inference gateway
        (``cluster-inference``) call at lease resolution: when the active lease
        for ``purpose`` points at a cluster deployment in state ``unloaded``, it
        enqueues one warm ``cluster.gpu.load`` and returns the name so the
        caller answers an honest 'waking' rather than a dead endpoint.
        """
        return wake_unloaded_cluster(
            purpose=purpose, broker=self.credential_broker,
            mode_store=ClusterModeStore(), cluster_store=self.cluster.store,
            enqueue_load=lambda name: enqueue_cluster_load(self.jobs, name),
            wake=wake,
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

    def _serving_metrics_endpoint(self) -> Optional[str]:
        """The GPU llama.cpp engine's ``/metrics`` URL right now, or ``None``.

        Endpoint discovery for the E′ serving scrape: it reads the SERVING STATE
        (``resolve_gpu_serving_target``) rather than hardcoding a port, and
        returns a URL only when the target is the managed-local llama.cpp engine
        on a loopback port — the one engine whose ``/metrics`` carries the
        ``llamacpp:`` family the parser expects, exposed on the same host:port the
        model serves. A cluster (vLLM) target, a hosted model, a single-model
        box, or nothing serving all return ``None``, so the poller writes nothing
        rather than scraping an endpoint it cannot read (the worker vLLM scrape is
        a separate, out-of-scope path). Every dependency is imported inside the
        call. A read that fails raises (pass-5 review B1): the poller then
        takes its discovery-failed path and records no stop for a serving engine.
        """
        from .gpu_serving_target import KIND_MANAGED_LOCAL, REASON_BROKER_UNAVAILABLE, resolve_gpu_serving_target
        from .serving_metrics_poller import DiscoveryUnknown, read_mode_strictly

        target = resolve_gpu_serving_target(self.credential_broker, read_mode_strictly(ClusterModeStore()))
        if target.reason == REASON_BROKER_UNAVAILABLE:
            # The broker could not be asked: not known, never "not serving" (pass-6 B1).
            raise DiscoveryUnknown("the credential broker could not be read")
        if not target.available or target.kind != KIND_MANAGED_LOCAL:
            return None
        endpoint = "http://127.0.0.1:{}/metrics".format(target.port)
        # The usage sink names the model from this same resolution (S6).
        self.usage.note_local_target(endpoint, target)
        return endpoint

    def _vllm_serving_targets(self) -> List[Tuple[str, int]]:
        """The vLLM replicas whose ``/metrics`` feed the serving card, or ``[]``.

        Endpoint discovery for the cluster half of the serving scrape
        (Observability Unit 1). It returns the per-replica ``(node_id, port)``
        servers of the current deployment ONLY while the mode file reads Mode B,
        the named record is a vLLM engine, and it is neither unloaded (scaled to
        zero) nor still deploying; every other case is an empty list, so the
        poller writes nothing rather than SSHing a server that is not up. The
        targets themselves come from :func:`~vaelor.gpu_idle_watch.deployment_idle_targets`,
        the same reader the idle watch trusts, so the serving scrape and the idle
        scrape cannot disagree about which nodes host the servers. An empty list
        means the record SAYS nothing serves (single-machine mode, unloaded,
        deploying, another engine); a read that fails raises instead (pass-5
        review B1), so the poller keeps its targets and records no stop.
        """
        from .gpu_cluster_mode import DEPLOYING_STATE, MODE_CLUSTER, UNLOADED_STATE
        from .gpu_pool_units import VLLM_ENGINE
        from .serving_metrics_poller import read_mode_strictly

        state = read_mode_strictly(ClusterModeStore())
        if str(getattr(state, "mode", "")) != MODE_CLUSTER:
            return []
        name = str(getattr(state, "deployment_name", "") or "")
        if not name:
            return []
        deployment = self.cluster.store.get_pooled_deployment(name)
        units = (deployment or {}).get("units") or {}
        if not deployment or units.get("engine") != VLLM_ENGINE:
            return []
        if str(deployment.get("state", "")) in (UNLOADED_STATE, DEPLOYING_STATE):
            return []
        return deployment_idle_targets(deployment)

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
        """True while any engine holds a model, or might - the down-cycle signal.

        VD-134: the MEASURED reading (`vaelor.model_residency`, the one owner
        of "is a model resident now"), busy for resident or unknown - not
        which engines are configured, which never went idle on a box with a
        local engine. A probe that raises is the scheduler's problem; it
        treats a raising signal as "loaded" to stay off a busy box.
        """
        return appliance_residency(
            bridge=self.hardware_bridge_client, broker=self.credential_broker,
            mode_store=ClusterModeStore(), cluster_store=self.cluster.store,
        ).busy

    def _automation_signals_by_node(self) -> Dict[str, Dict[str, float]]:
        """Per-machine alert readings, keyed by node id (``""`` is the controller).

        The controller keeps all five signals; each enrolled worker that has
        reported RECENTLY contributes the two its telemetry carries. A worker
        that is stale or has never reported is omitted, not zeroed, so the alert
        engine skips its rules rather than firing on data we do not have — the
        one honest-degradation rule this whole feature turns on.
        """
        data = self.tools.callbacks["current_data"]()
        signals: Dict[str, Dict[str, float]] = {
            "": _controller_signals(data, self.system.services())
        }
        for node_id in self._enrolled_worker_ids():
            readings = self._worker_signals(node_id)
            if readings is not None:
                signals[node_id] = readings
        return signals

    def _alert_machine_names(self) -> Dict[str, str]:
        """Each enrolled worker's name by node id, for a fired alert to name it.

        Read from the same cluster store as `_enrolled_worker_ids`, so the
        machine an alert names is the machine the Fleet screen shows. Store
        trouble raises into the runner, which logs it and labels the machine
        generically rather than printing its id.
        """
        manager = getattr(self, "cluster", None)
        if manager is None:
            return {}
        return {
            str(record.get("id", "")): str(record.get("name") or "").strip()
            for record in manager.store.list_nodes()
            if str(record.get("id", ""))
        }

    def _enrolled_worker_ids(self) -> list:
        """Every enrolled worker node id, or empty when the fleet is unwired.

        Read from the same cluster store the Fleet views and the ingest route
        use, so the set of nodes an alert may target cannot disagree with who is
        actually in the fleet. Any store trouble degrades to no workers rather
        than raising into the runner's evaluation pass.
        """
        manager = getattr(self, "cluster", None)
        if manager is None:
            return []
        try:
            records = manager.store.list_nodes()
        except (AttributeError, OSError, TypeError, ValueError):
            return []
        return [str(record.get("id", "")) for record in records if str(record.get("id", ""))]

    def _worker_signals(self, node_id: str) -> Optional[Dict[str, float]]:
        """A worker's latest cpu/memory readings, or None when not reporting.

        Reuses the ``telemetry_history_range`` callback the Performance tab reads
        through — the node's newest raw row, not a bucket mean, is its current
        reading — rather than opening a second path to the store. Returns None
        (so the caller omits the node) when the store is unwired or unreadable,
        when ``worker_is_reporting`` (the one owner, on the controller's receive
        clock first) says it is not reporting, or when the row carried no usable
        cpu/memory value. Never invents a zero for an absent reading.
        """
        ranged = self.tools.callbacks.get("telemetry_history_range")
        if ranged is None:
            return None
        try:
            result = ranged(WORKER_SIGNAL_WINDOW_SECONDS, MAX_HISTORY_BUCKETS, node_id)
        except (TelemetryStoreError, OSError, ValueError):
            return None
        if not worker_is_reporting(node_id, result.get("last_sample_age_seconds")):
            return None
        row = result.get("latest") if isinstance(result.get("latest"), dict) else {}
        readings: Dict[str, float] = {}
        for field in WORKER_SIGNAL_FIELDS:
            value = row.get(field)
            if isinstance(value, (int, float)) and not isinstance(value, bool):
                readings[field] = float(value)
        return readings or None

    def performance_snapshot(self) -> Dict[str, Any]:
        """``system.performance``: the tab's snapshot, off the one list of sources."""
        return assistant_performance_snapshot(performance_callbacks(self, self._host_callbacks))

    def _api_callbacks(
        self, callbacks: Dict[str, Callable[..., Any]]
    ) -> Dict[str, Any]:
        return {
            **callbacks,
            # Everything the Performance snapshot reads, shared with the
            # Assistant's system.performance tool (performance_callbacks).
            **performance_callbacks(self, callbacks),
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
            "mcp_catalog": self.mcp_catalog,
            "agent_mcp_grants": self.agent_mcp_grants,
            "skills_library": self.skills_library,
            "skill_attachments": self.skill_attachments,
            "agent_deployments": self.agent_deployments,
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
            "inference_gateway_status": gateway_status_reader(self),
            # The gateway's per-key usage meter (Phase G), joined onto the API-key
            # rows. The request stores, the model's usage ledger, the LLM Server
            # gate's record, the root bridge client and the trace status come in
            # with performance_callbacks above.
            "usage_meter": self.usage_meter,
            "phoenix_store": self.phoenix_store,
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
            # A live AI-Chat (capable GPU) lease gates the manual "re-run on the
            # larger model" route. Fails closed: any broker trouble reads as
            # unavailable so the route degrades honestly, never blind-escalates.
            "application_capable_available": lambda: _capable_lease_active(
                self.deployment_agent
            ),
            "application_compose_validator": lambda compose: (
                validate_application_compose(
                    compose,
                    data_path("workloads"),
                    hardware_inventory(),
                    # W7-2: the ports stored models come back on.
                    model_ports=model_port_holders(self.credential_broker),
                )
            ),
            "cluster_backups": self.cluster_backups,
            "cluster_operations": self.cluster_operations,
            "host_desktop_broker": self.host_desktop_broker,
            # The appliance-upgrade routes read the GitHub release source here;
            # without it they fall back to the offline StubReleaseSource.
            "release_source": default_release_source(),
        }
