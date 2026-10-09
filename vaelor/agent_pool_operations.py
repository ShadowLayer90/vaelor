"""Deploy and remove one cluster Agent, record-first with strand-nothing rollback.

The F4b-ii orchestration tier (design F4b-ii-C, item C1). It is the agent-tier
sibling of :class:`vaelor.gpu_pool_operations.GpuPoolOperations`: it turns a
requested agent into a live, LAN-reachable, OpenAI-compatible runtime backed by a
cluster model, tool boundary enforced, following the very same spine that module
established -

* **Record first.** A ``deploying`` row is written before any node is touched
  (:meth:`vaelor.agent_deployments.AgentDeploymentStore.create`), so a deploy that
  dies with its executor leaves a removable record rather than an orphaned unit.
* **Health is a probe, never a claim.** The row is promoted to ``healthy`` only
  after the inbound gate answers ``/health`` - a written unit is not health, the
  rule :mod:`vaelor.gpu_pool_operations` records for the GPU tier applied here.
* **Strand nothing.** Any failure after the row exists runs a best-effort
  rollback in a fixed order - stop the unit and gate, revoke the minted inbound
  key, then mark the row ``failed`` - so no partial deploy leaves a running unit,
  an unrevoked door key, or a row that claims to be live.

Everything security-critical is REUSED, never re-implemented here: the read-only
refusal and the effective-tool intersection live in
:mod:`vaelor.agent_runtime_gate`; the inbound-key mint/revoke live in
:mod:`vaelor.served_endpoint_keys` (reached through the broker); the pinned
custom-agent definition comes from :mod:`vaelor.custom_agents`; the fixed-ExecStart
launch and its inbound gate live behind the root bridge's ``agent_start`` /
``agent_stop`` verbs in :mod:`vaelor.agent_service`. This module only sequences
them and owns the rollback.

**One secret rule (design D6).** The plaintext model key and the ``vsk_`` inbound
gate keys travel ONLY inside the 0600 ``config_content`` and the ``gate`` payload
over the bridge socket. Neither is ever written to the deployment row, returned in
a fingerprint, or logged. The deploy's first key is revealed ONCE by the control
plane that minted it (``agent_deploy_keys.mint_first_agent_key``, in the POST that
queues the deploy), exactly as the F3 mint response does; the executor only
resolves it by id.

**The gate follows the CURRENT key set (ACC-070).** The gate admits every active
key of the agent's endpoint, and :meth:`AgentPoolOperations.rekey` replaces it
when that set changes; the reconcile (``agent_reconcile``) reads what the running
gate carries off the bridge and re-keys on drift. The runtime likewise carries a
``surface_digest`` of what it was started with, and is relaunched when its tools,
skills, instructions or backing model change (ACC-074).

Every collaborator is injected so the whole path is unit-testable with fakes and
reaches no real bridge, broker or socket. The class is a control-plane object; it
never runs the agent itself, only asks the root bridge to.
"""

from __future__ import annotations

import logging
import time
from typing import Any, Callable, Dict, List, Mapping, Optional

from .agent_backing import backing_identity, surface_digest
from .agent_memory import memory_client_config, mint_memory_key
from .agent_runtime_gate import AgentRuntimeGateError, assert_read_only, build_mcp_grant_gate
from .agent_wake import wake_endpoint
from .agent_skill_surface import (
    MAX_SKILL_BLOCK_CHARS as _MAX_SKILL_BLOCK_CHARS,  # noqa: F401 - re-exported
    SKILL_BLOCK_HEADER as _SKILL_BLOCK_HEADER,  # noqa: F401 - re-exported
    SkillRefused, append_skill_block, skill_surface,
)
from .custom_agents import ALLOWED_SCOPES
from .gpu_pool_units import deployment_name
from .gpu_serving_target import is_loopback_host
from .served_endpoint_keys import SERVED_ENDPOINT_PROVIDER, SERVED_ENDPOINT_PURPOSE_PREFIX
from .agent_pool_seams import (  # noqa: F401 - AgentPoolError is re-exported
    AGENT_LIFECYCLE_LOCK, RUNTIME_LOOPBACK_HOST, AgentPoolError, _BACKING_UNAVAILABLE, _default_allocate_port,
    _default_bridge, _default_probe, _default_validate_runtime, _function_spec,
    _resolve_recorded, _split_endpoint,
)

LOGGER = logging.getLogger(__name__)

#: The confirm tokens this operation re-checks (design D2, SF-E). The route that
#: turns a request into a job checks the same string first; re-checking it here is
#: the second, non-skippable layer, so a job dispatched by any path still cannot
#: mutate without the reviewed word. Named individually rather than as one map so
#: this vocabulary is not mistaken for a shared cross-module token set.
DEPLOY_CONFIRM = "deploy-cluster-agent"
REMOVE_CONFIRM = "remove-cluster-agent"

#: The endpoint-id family a deployed agent's inbound key is minted under. The
#: broker derives the resolvable purpose ``served-endpoint:agent:<id>`` from this
#: id (:func:`vaelor.served_endpoint_keys.is_served_endpoint_purpose`), which is
#: what the reboot reconcile (SF-A, built in C2) resolves the key back through.
AGENT_ENDPOINT_PREFIX = "agent:"


#: How long, and how often, the deploy waits for the inbound gate to answer a
#: health probe before it calls the launch failed and rolls back. Bounded so a
#: unit that never comes up fails honestly rather than hanging the job.
HEALTH_WAIT_SECONDS = 45.0
HEALTH_POLL_SECONDS = 1.5

#: Sentences written at one call site each, kept here so a later edit cannot let
#: two copies drift, and each phrased to differ from every other backend module's
#: refusals rather than share a literal with one.
_CONFIRM_DEPLOY = "Confirm the reviewed cluster agent deployment."
_CONFIRM_REMOVE = "Confirm the reviewed cluster agent removal."
_AGENT_NOT_FOUND = "That cluster agent deployment was not found."
_DEFINITION_MISSING = (
    "The pinned custom agent definition (id {!r} version {}) could not be loaded."
)
_MUTATING_REFUSED = (
    "A deployed cluster agent offers read-only tools only, so it cannot be "
    "backed by an agent that holds an acting permission: {}"
)
_NOT_INFERENCE_REFUSED = (
    "Only an inference-surface agent may deploy to the cluster; this "
    "definition was authored for the assistant surface."
)
_HEALTH_FAILED = (
    "The cluster agent runtime did not answer a health check after it was launched."
)
#: The runtime is validated in process with the SAME function its unit runs
#: (``agent_server.build_runtime``) before anything is launched, so a config the
#: runtime would refuse fails the deploy with the runtime's own reason instead
#: of a unit that crash-loops behind a generic health failure (ACC-076).
_RUNTIME_REFUSED = "The agent could not be started: {}"
#: A grant on the built-in server: it has no network address, so the agent's
#: outbound MCP client can never call it (ACC-076).
_BUILTIN_SERVER_REFUSED = (
    "The built-in Vaelor tool server cannot be given to a deployed cluster "
    "agent: it has no network address the agent could call. Vaelor's own "
    "read-only tools reach a deployed agent through the scopes its definition "
    "grants; remove the built-in server from this agent's tools."
)


class AgentPoolOperations:
    """Deploy and remove one cluster agent, mirroring the GPU pool operations.

    All collaborators are injected so the path is driven with fakes and reaches no
    real bridge, broker or socket:

    * ``store`` - the :class:`vaelor.agent_deployments.AgentDeploymentStore`.
    * ``broker`` - the credential broker client (reads the backing lease).
    * ``catalog`` - the :class:`vaelor.mcp_catalog.McpCatalogStore` whose ``get``
      is the ``catalog_lookup`` for :func:`~vaelor.agent_runtime_gate.build_mcp_grant_gate`.
    * ``custom_agents`` - the :class:`vaelor.custom_agents.CustomAgentStore` the
      pinned definition is read from.
    * ``skills_library`` - the :class:`vaelor.skills_library.SkillsLibraryStore`
      the attached skills are resolved against by id, read-only-safely.
    * ``keys`` - the served-endpoint key funcs (``mint`` / ``revoke``); defaults to
      the broker, which exposes exactly those.
    * ``bridge`` - the ``agent_start`` / ``agent_stop`` root-bridge client.
    * ``resolve_backing`` / ``allocate_port`` / ``probe`` - the seams a test drives
      without a live model, a free socket, or an HTTP server.
    * ``cluster_store`` - where the backing deployment is recorded (the default
      backing resolve and the surface digest read it). The production wiring
      always passes the cluster's own store; without one the digest names the
      deployment only.
    * ``validate_runtime`` - the pre-launch check, by default the runtime's own
      ``build_runtime`` over the rendered config.
    """

    def __init__(
        self, *, store, broker, catalog, custom_agents,
        skills_library: Optional[Any] = None,
        keys: Optional[Any] = None,
        bridge: Optional[Any] = None,
        resolve_backing: Optional[Callable[[Any, str], Dict[str, Any]]] = None,
        allocate_port: Optional[Callable[..., int]] = None,
        probe: Optional[Callable[[str], bool]] = None,
        progress: Optional[Callable[[int, str], None]] = None,
        sleep: Optional[Callable[[float], None]] = None,
        monotonic: Optional[Callable[[], float]] = None,
        cluster_store: Optional[Any] = None,
        validate_runtime: Optional[Callable[[str], None]] = None,
    ):
        self.store = store
        self.broker = broker
        self.catalog = catalog
        self.custom_agents = custom_agents
        self.skills_library = skills_library
        self._cluster_store = cluster_store
        self._validate_runtime = validate_runtime or _default_validate_runtime
        # The mint/revoke pair is on the broker client, so the broker is the
        # default; kept a separate seam so a test can spy on the key path alone.
        self.keys = keys if keys is not None else broker
        self._bridge = bridge
        self._resolve_backing = resolve_backing or (
            lambda broker_, name_: _resolve_recorded(broker_, self.cluster_store, name_)
        )
        self._allocate_port = allocate_port or _default_allocate_port
        self._probe = probe or _default_probe
        self._progress = progress
        import time

        self._sleep = sleep or time.sleep
        self._monotonic = monotonic or time.monotonic

    @property
    def bridge(self) -> Any:
        """The launch bridge, built on first use so import stays socket-free."""
        if self._bridge is None:
            self._bridge = _default_bridge()
        return self._bridge

    @property
    def cluster_store(self) -> Any:
        """The injected cluster store, or ``None`` (never a default database)."""
        return self._cluster_store

    # -- deploy ---------------------------------------------------------------

    def deploy(self, payload: Dict[str, Any], progress=None) -> Dict[str, Any]:
        """Deploy one cluster agent under the agent lifecycle lock (B1)."""
        with AGENT_LIFECYCLE_LOCK:
            return self._deploy(payload, progress)

    def _deploy(self, payload: Dict[str, Any], progress=None) -> Dict[str, Any]:
        """Deploy one cluster agent: record-first, launch, health-probe, promote.

        The steps and their order are the design's (F4b-ii-C, C1): validate the
        confirm token and the inputs, load and read-only-check the pinned
        definition, write the ``deploying`` row and pin its grants/skills, resolve
        the backing cluster model, compute the effective MCP surface, mint the
        inbound key (storing only its id), allocate a loopback port, build the
        0600 config, launch through the bridge, health-probe, set the endpoint and
        promote to ``healthy``. Any failure after the row exists rolls back in a
        fixed order and re-raises.
        """
        report = progress or self._progress or (lambda _percent, _message: None)
        try:
            if payload.get("confirm") != DEPLOY_CONFIRM:
                raise AgentPoolError(_CONFIRM_DEPLOY)
            request = self._validate_inputs(payload)
            definition = self._pinned_definition(request, require_inference_surface=True)
            # Resolve the attached skills and REFUSE the whole deploy (fail
            # closed) before the record-first write - adjacent to the read-only
            # refusal, so an unsafe skill leaves no deploying row behind.
            skills = self._skill_surface(request["skills"], strict=True)
            self._refuse_builtin_grants(request["mcp_grants"])

            report(5, "Recording the cluster agent deployment")
            name = request["name"]
            create_fields = dict(
                name=name,
                custom_agent_id=request["custom_agent_id"],
                custom_agent_version=request["custom_agent_version"],
                model_deployment_name=request["model_deployment_name"],
                node_ids=request["node_ids"],
                memory_policy=request.get("memory_policy"),
            )
            if request["deployment_id"]:
                # The control plane chose the id so it could mint (and reveal)
                # the first key under ``agent:<id>`` before queueing this deploy.
                create_fields["deployment_id"] = request["deployment_id"]
            # Inside this try (S2): a refused create - a name already taken -
            # must revoke the first key too, or it stays live with no agent.
            row = self.store.create(**create_fields)
        except Exception:
            # No row of ours exists, so the rollback cannot find the first key
            # the control plane minted for this deploy: revoke it here.
            self._revoke_quietly(
                str(payload.get("name", "")), str(payload.get("first_key_id") or ""),
                endpoint_id=AGENT_ENDPOINT_PREFIX + str(payload.get("deployment_id") or ""),
            )
            raise

        key_id = ""
        try:
            self.store.attach_grants(
                name, request["custom_agent_version"], request["mcp_grants"]
            )
            self.store.attach_skills(
                name, request["custom_agent_version"], request["skills"]
            )
            report(20, "Resolving the backing cluster model")
            backing = self._backing_profile(request["model_deployment_name"])

            report(35, "Computing the approved tool surface")
            surface = self._mcp_surface(request["mcp_grants"])
            for entry in surface["dropped"]:
                LOGGER.info(
                    "Cluster agent %s: grant for %s/%s is not offered (%s).",
                    name, entry.get("server_id"), entry.get("tool"),
                    entry.get("reason"),
                )

            endpoint_id = AGENT_ENDPOINT_PREFIX + str(row["id"])
            keys_read_at = time.time()
            if request["api_key_id"]:
                report(50, "Reading the inbound endpoint key minted for this deploy")
                key_id = request["api_key_id"]
                minted = {"credential_id": key_id}
                key_value = self._resolve_gate_key(key_id, endpoint_id)
            else:
                report(50, "Minting the inbound endpoint key")
                minted = self.keys.mint(endpoint_id, endpoint_id)
                key_id = str(minted.get("credential_id", ""))
                key_value = str(minted.get("key", ""))
            # The id, and only the id, is persisted (invariant 1). The value stays
            # in this frame and reaches the store, the payload and no log.
            self.store.set_api_key_id(name, key_id)

            report(65, "Allocating a loopback runtime port")
            port = self._allocate_port()

            memory = mint_memory_key()
            self.store.set_memory_key_hash(name, memory["hash"])
            config_content = self._build_config(
                definition, request, backing, surface, port,
                memory_token=memory["token"], skill_surface=skills,
            )
            self._check_runtime(config_content)
            gate = {
                "listen_host": request["advertise_address"],
                "listen_port": port,
                "api_keys": [key_value],
                "keys_read_at": keys_read_at,
            }

            report(80, "Launching the agent through the root bridge")
            self.bridge.agent_start(
                name=name, port=port,
                config_content=config_content, gate=gate,
            )

            endpoint = "http://{}:{}/v1".format(request["advertise_address"], port)
            health_url = "http://{}:{}/health".format(
                request["advertise_address"], port
            )
            report(90, "Waiting for the agent to answer a health check")
            if not self._await_health(health_url):
                raise AgentPoolError(_HEALTH_FAILED)

            self.store.set_endpoint(name, endpoint)
            self.store.update_state(name, "healthy")
        except Exception as error:
            self._rollback(name, key_id, error)
            raise

        report(100, "The cluster agent is healthy behind its inbound gate")
        return {
            "name": name,
            "deployment_id": str(row["id"]),
            "deployment_mode": "agent",
            "state": "healthy",
            "endpoint": endpoint,
            "backed_by": request["model_deployment_name"],
            "backing_model": backing.get("model", ""),
            "backing_loopback": backing.get("loopback", False),
            "node_ids": request["node_ids"],
            # The key VALUE is returned for a single reveal to the caller and is
            # never persisted or logged; the fingerprint/last4 are the durable,
            # safe references (invariant 1).
            "key": key_value,
            "key_fingerprint": str(minted.get("key_fingerprint", "")),
            "key_last4": str(minted.get("last4", "")),
            "api_key_id": key_id,
            "effective_pairs": [list(pair) for pair in surface["pairs"]],
            "dropped_grants": surface["dropped"],
            # What each attached skill sent the agent and granted it (ACC-138).
            "skills_report": skills["guidance"],
        }

    # -- remove ---------------------------------------------------------------

    def remove(self, payload: Dict[str, Any], progress=None) -> Dict[str, Any]:
        """Remove one cluster agent under the agent lifecycle lock (B1)."""
        with AGENT_LIFECYCLE_LOCK:
            return self._remove(payload, progress)

    def _remove(self, payload: Dict[str, Any], progress=None) -> Dict[str, Any]:
        """Tear one cluster agent down: revoke, stop the unit and gate, delete.

        Idempotent and best-effort against a partially-torn deployment: the row
        is moved to ``removing``, EVERY active key of its endpoint is revoked
        FIRST (B1: whatever still runs for a moment after refuses at once, and
        nothing that comes back could admit a key), then the bridge stops the
        unit and gate and unlinks the 0600 configs, and the record is dropped. A
        stop or revoke that fails is logged, not fatal - refusing to finish a
        removal would leave the operator no way to clear it.
        """
        report = progress or self._progress or (lambda _percent, _message: None)
        if payload.get("confirm") != REMOVE_CONFIRM:
            raise AgentPoolError(_CONFIRM_REMOVE)
        name = deployment_name(str(payload.get("name", "")))
        deployment = self.store.get(name)
        if deployment is None:
            raise AgentPoolError(_AGENT_NOT_FOUND)

        report(10, "Marking the cluster agent for removal")
        try:
            self.store.update_state(name, "removing")
        except Exception as error:  # noqa: BLE001 - a terminal row is still removable
            LOGGER.warning("Could not mark %s removing: %s", name, error)

        report(40, "Revoking the inbound endpoint keys")
        key_id = str(deployment.get("api_key_id") or "")
        self._revoke_quietly(name, key_id)
        self._revoke_endpoint_keys_quietly(deployment)
        self._clear_memory_quietly(name)

        report(70, "Stopping the agent unit and its inbound gate")
        self._stop_quietly(name)

        report(90, "Dropping the deployment record")
        self.store.delete(name)
        report(100, "The cluster agent has been removed")
        return {"name": name, "removed": True}

    # -- relaunch (SF-A reboot survival) --------------------------------------

    def relaunch(
        self, row: Mapping[str, Any], *, gate_key: Optional[str] = None,
        gate_keys: Optional[List[str]] = None, keys_read_at: Optional[float] = None,
    ) -> Dict[str, Any]:
        """Re-render a row's runtime and inbound gate from its CURRENT surface.

        The reconcile (``agent_reconcile``) calls this after a reboot or a crash
        cleared the tmpfs config, and when the agent's tools, skills,
        instructions or backing changed since it was started (ACC-074). It
        rebuilds the 0600 config from the persisted row, re-resolves the RECORDED
        backing deployment, and re-launches through the bridge with the
        endpoint's current key set (``gate_keys``; ``gate_key`` is the one-key
        form older callers pass). An empty set runs the runtime with NO gate. It
        never mints a key, never allocates a new port, and never writes the
        deployment record. A backing or definition that cannot be resolved
        raises, and the reconcile turns that into a surfaced, in-place row.
        """
        name = deployment_name(str(row.get("name", "")))
        endpoint = str(row.get("endpoint") or "")
        advertise, port = _split_endpoint(endpoint)
        definition, request, surface, skills = self._row_surface(row, strict=False)
        backing = self._backing_profile(request["model_deployment_name"])
        for dropped in skills["refusals"] + surface["dropped"]:
            LOGGER.info(
                "Cluster agent %s: %s dropped on relaunch (%s).", name,
                dropped.get("skill_id") or dropped.get("server_id"),
                dropped.get("reason"),
            )
        memory = mint_memory_key()
        self.store.set_memory_key_hash(name, memory["hash"])
        config_content = self._build_config(
            definition, request, backing, surface, port,
            memory_token=memory["token"], skill_surface=skills,
        )
        self._check_runtime(config_content)
        keys = list(gate_keys) if gate_keys is not None else (
            [gate_key] if gate_key else []
        )
        gate = {
            "listen_host": advertise, "listen_port": port, "api_keys": keys,
            "keys_read_at": time.time() if keys_read_at is None else keys_read_at,
        }
        started = self.bridge.agent_start(
            name=name, port=port, config_content=config_content, gate=gate,
        )
        change = started.get("gate_change") if isinstance(started, Mapping) else None
        return {"name": name, "endpoint": endpoint, "relaunched": True, "gate_change": change}

    def rekey(
        self, row: Mapping[str, Any], gate_keys: List[str], *,
        keys_read_at: Optional[float] = None,
    ) -> Dict[str, Any]:
        """Replace only the row's gate with ``gate_keys``; the runtime keeps serving.

        The key-change half of ACC-070: a mint, rotate or revoke reaches the
        running gate here. An empty set takes the gate down (no keyless door).
        ``keys_read_at`` is when the set was read; a gate already running a set
        read later is kept (S4).
        """
        name = deployment_name(str(row.get("name", "")))
        advertise, port = _split_endpoint(str(row.get("endpoint") or ""))
        gate = {
            "listen_host": advertise, "listen_port": port, "api_keys": list(gate_keys),
            "keys_read_at": time.time() if keys_read_at is None else keys_read_at,
        }
        return self.bridge.agent_rekey(name=name, port=port, gate=gate)

    def desired_digest(
        self, row: Mapping[str, Any], credential_rows: List[Mapping[str, Any]],
    ) -> str:
        """The surface digest a runtime started NOW for ``row`` would carry.

        Built from the same pieces a relaunch renders - the pinned definition,
        the catalog-intersected tools, the attached skills' guidance and scopes,
        and the recorded backing's identity - with no secret in it, so the
        reconcile and the console can tell a runtime started with an older
        surface from a current one without reading its config.
        """
        definition, request, surface, skills = self._row_surface(row, strict=False)
        name = request["model_deployment_name"]
        identity = (
            backing_identity(self.cluster_store, name, credential_rows)
            if self.cluster_store is not None else {"deployment": name}
        )
        return self._digest(definition, surface, skills, identity)

    def _row_surface(self, row: Mapping[str, Any], *, strict: bool):
        """``(definition, request, mcp surface, skill surface)`` for a stored row."""
        advertise, _port = _split_endpoint(str(row.get("endpoint") or ""))
        request = {
            "name": deployment_name(str(row.get("name", ""))),
            "custom_agent_id": str(row.get("custom_agent_id", "")),
            "custom_agent_version": row.get("custom_agent_version"),
            "actor": str(row.get("actor", "") or ""),
            "advertise_address": advertise,
            "mcp_grants": row.get("mcp_grants") or [],
            "skills": row.get("skills") or [],
            "model_deployment_name": str(
                (row.get("backing") or {}).get("model_deployment_name", "")
            ),
        }
        definition = self._pinned_definition(request, require_inference_surface=False)
        # B1/C3: the pinned skills are re-applied on every relaunch (else they
        # go inert on each crash/restart), but a now-unsafe or missing skill is
        # DROPPED rather than aborting - a reboot must never strand a healthy
        # agent over one skill that changed.
        return (
            definition, request, self._mcp_surface(request["mcp_grants"]),
            self._skill_surface(request["skills"], strict=strict),
        )

    # -- helpers --------------------------------------------------------------

    def _validate_inputs(self, payload: Mapping[str, Any]) -> Dict[str, Any]:
        """Resolve and shape the deploy inputs, refusing a malformed request.

        The name goes through the shared cluster slug rule; the custom-agent pin,
        the backing deployment name and the LAN address are required; the grants
        and skills are carried through to the store's own validators unchanged.
        """
        name = deployment_name(str(payload.get("name", "")))
        custom_agent_id = str(payload.get("custom_agent_id", "") or "").strip()
        if not custom_agent_id:
            raise AgentPoolError("A cluster agent needs a custom agent id.")
        try:
            version = int(payload.get("custom_agent_version"))
            if version < 1:
                raise ValueError
        except (TypeError, ValueError) as error:
            raise AgentPoolError(
                "A cluster agent needs a positive custom agent version."
            ) from error
        model_deployment_name = str(
            payload.get("model_deployment_name", "") or ""
        ).strip()
        if not model_deployment_name:
            raise AgentPoolError("A cluster agent needs a backing model deployment.")
        advertise = self._lan_host(payload.get("advertise_address"))
        node_ids = payload.get("node_ids") or []
        if not isinstance(node_ids, list):
            raise AgentPoolError("The node placement must be a list of node ids.")
        actor = str(payload.get("actor", "") or "").strip()
        return {
            "deployment_id": str(payload.get("deployment_id", "") or "").strip(),
            "api_key_id": str(payload.get("first_key_id", "") or "").strip(),
            "name": name,
            "custom_agent_id": custom_agent_id,
            "custom_agent_version": version,
            "model_deployment_name": model_deployment_name,
            "advertise_address": advertise,
            "node_ids": [str(node) for node in node_ids],
            "mcp_grants": payload.get("mcp_grants") or [],
            "skills": payload.get("skills") or [],
            "memory_policy": payload.get("memory_policy"),
            "actor": actor,
        }

    @staticmethod
    def _lan_host(value: Any) -> str:
        """A non-loopback, host-shaped LAN address the inbound gate can bind.

        The gate must front the agent on the LAN, so a loopback address is refused
        here (it would publish the runtime to nobody); the syntactic check keeps
        an unusable value out of the endpoint URL a caller is later handed.
        """
        import re

        text = str(value or "").strip()
        if not text or not re.fullmatch(r"[A-Za-z0-9_.:\-]{1,253}", text):
            raise AgentPoolError(
                "A cluster agent needs a LAN address for its inbound gate."
            )
        if is_loopback_host(text):
            raise AgentPoolError(
                "The inbound gate LAN address must not be a loopback address."
            )
        return text

    def _pinned_definition(
        self, request: Mapping[str, Any], *, require_inference_surface: bool = True
    ) -> Dict[str, Any]:
        """Load the pinned definition and refuse a mutating agent (design C6/B3).

        ``assert_read_only`` refuses ANY acting permission, so a first-cut deployed
        agent holds none. The refusal happens BEFORE the record-first write, so a
        mutating agent leaves no row behind at all.

        ``require_inference_surface`` gates the B1 surface refusal to the INITIAL
        deploy-from-gallery path only. A relaunch reconciles a deployment that
        already exists in the record store - a cluster agent by fact - so it must
        relaunch regardless of the surface its pinned (possibly legacy) definition
        reads back, or a pre-B1 deployment would stay down after a reboot. The
        read-only belt still applies on both paths.
        """
        definition = self.custom_agents.get_version(
            request["custom_agent_id"], request["actor"],
            request["custom_agent_version"],
        )
        if not isinstance(definition, Mapping):
            raise AgentPoolError(
                _DEFINITION_MISSING.format(
                    request["custom_agent_id"], request["custom_agent_version"]
                )
            )
        if require_inference_surface and str(
            definition.get("surface", "assistant")
        ) != "inference":
            raise AgentPoolError(_NOT_INFERENCE_REFUSED)
        permissions = list(definition.get("permissions") or [])
        try:
            assert_read_only(permissions)
        except AgentRuntimeGateError as error:
            raise AgentPoolError(
                _MUTATING_REFUSED.format(", ".join(str(item) for item in permissions))
            ) from error
        return dict(definition)

    def _backing_profile(self, model_deployment_name: str) -> Dict[str, Any]:
        """The backing model's ``{base_url, model, api_key}``, LAN base_url preferred.

        Design C1: agent_server's ``chat_completion`` serialises on the local
        inference slot for any loopback base_url, so the cluster's non-loopback LAN
        endpoint is used when the lease carries one; a loopback-only lease is
        recorded (``loopback``) and used anyway - the runtime's own 429 fallback
        covers the throttle rather than the deploy refusing.
        """
        profile = self._resolve_backing(self.broker, model_deployment_name)
        base_url = str(profile.get("base_url", "") or "")
        if not base_url:
            raise AgentPoolError(_BACKING_UNAVAILABLE.format(model_deployment_name))
        import urllib.parse

        host = urllib.parse.urlsplit(base_url).hostname
        return {
            "base_url": base_url,
            "model": str(profile.get("model", "") or ""),
            "api_key": str(profile.get("api_key", "") or ""),
            "loopback": is_loopback_host(host),
        }

    def _mcp_surface(self, mcp_grants: Any) -> Dict[str, Any]:
        """Compute the frozen effective tool set, the specs and the servers.

        The gate is built from the grants INTERSECTED with the catalog's current
        approvals (:func:`~vaelor.agent_runtime_gate.build_mcp_grant_gate`); only a
        pair in the effective set gets an OpenAI function spec, and each granted
        server's ``{name, endpoint}`` comes from the catalog. Dropped grants are
        surfaced for the log, never widened.
        """
        builtin = self._builtin_grants(mcp_grants)
        gate = build_mcp_grant_gate(
            [grant for grant in mcp_grants or [] if grant not in builtin],
            self.catalog.get,
        )
        pairs = sorted(gate.effective_pairs)
        endpoints: Dict[str, str] = {}
        for grant in mcp_grants or []:
            if not isinstance(grant, Mapping):
                continue
            record = self.catalog.get(str(grant.get("server_id") or "").strip())
            if isinstance(record, Mapping):
                server_name = str(record.get("name") or "").strip()
                if server_name:
                    endpoints[server_name] = str(record.get("endpoint") or "")
        specs: List[Dict[str, Any]] = []
        servers: List[Dict[str, Any]] = []
        seen_servers: set = set()
        for server_name, tool in pairs:
            specs.append(_function_spec(server_name, tool))
            if server_name not in seen_servers:
                seen_servers.add(server_name)
                servers.append({
                    "name": server_name,
                    "endpoint": endpoints.get(server_name, ""),
                })
        dropped = list(gate.dropped_grants) + [
            {"server_id": str(grant.get("server_id") or ""), "tool": "",
             "reason": _BUILTIN_SERVER_REFUSED}
            for grant in builtin
        ]
        return {
            "pairs": pairs, "specs": specs, "servers": servers, "dropped": dropped,
        }

    def _builtin_grants(self, mcp_grants: Any) -> List[Mapping[str, Any]]:
        """The grants that name a built-in (address-less) catalog server."""
        found: List[Mapping[str, Any]] = []
        for grant in mcp_grants or []:
            if not isinstance(grant, Mapping):
                continue
            record = self.catalog.get(str(grant.get("server_id") or "").strip())
            if isinstance(record, Mapping) and str(record.get("kind") or "") == "builtin":
                found.append(grant)
        return found

    def _refuse_builtin_grants(self, mcp_grants: Any) -> None:
        """Refuse a deploy that grants the built-in server, before any row exists."""
        if self._builtin_grants(mcp_grants):
            raise AgentPoolError(_BUILTIN_SERVER_REFUSED)

    def _skill_surface(self, skills: Any, *, strict: bool) -> Dict[str, Any]:
        """The attached skills' scopes, guidance block and report (one derivation).

        :func:`vaelor.agent_skill_surface.skill_surface`, shared with the review
        preview; a strict (deploy) refusal becomes this module's error.
        """
        try:
            return skill_surface(self.skills_library, skills, strict=strict)
        except SkillRefused as error:
            raise AgentPoolError(str(error)) from error

    def _resolve_gate_key(self, key_id: str, endpoint_id: str) -> str:
        """The plaintext of the key the control plane minted for this deploy.

        Resolved by id under ``served-endpoint:agent:<id>`` over the broker socket
        - the same lease the reconcile uses - so the value exists only in this
        frame and the bridge payload.
        """
        lease = self.broker.resolve(key_id, SERVED_ENDPOINT_PURPOSE_PREFIX + endpoint_id)
        value = str((lease or {}).get("token") or "")
        if not value:
            raise AgentPoolError("The inbound key minted for this deploy could not be read.")
        return value

    def _check_runtime(self, config_content: str) -> None:
        """Refuse a config the runtime itself would refuse, with its reason."""
        try:
            self._validate_runtime(config_content)
        except AgentPoolError:
            raise
        except Exception as error:  # noqa: BLE001 - the runtime's own refusal, named
            raise AgentPoolError(_RUNTIME_REFUSED.format(error)) from error

    @staticmethod
    def _digest(definition, surface, skills, identity) -> str:
        return surface_digest({
            "instructions": append_skill_block(
                str(definition.get("instructions") or ""), skills.get("instructions"),
            ),
            "scopes": sorted(set(definition.get("scopes") or []) | set(skills.get("scopes") or [])),
            "web_access": definition.get("web_access"),
            "pairs": [list(pair) for pair in surface["pairs"]],
            "servers": surface["servers"],
            "backing": identity,
        })

    def _build_config(
        self, definition: Mapping[str, Any], request: Mapping[str, Any],
        backing: Mapping[str, Any], surface: Mapping[str, Any], port: int,
        memory_token: str = "", skill_surface: Optional[Mapping[str, Any]] = None,
    ) -> str:
        """Render the 0600 JSON config the agent runtime reads (design D6).

        The model key and the injected MCP data live in this body and travel only
        over the bridge socket. Scopes are defensively intersected with
        :data:`~vaelor.custom_agents.ALLOWED_SCOPES`, permissions are forced empty
        (a deployed agent is read-only), and the listen host is fixed to loopback.
        """
        import json

        skills = dict(skill_surface or {})
        # UNION the accepted skills' read scopes in, then re-intersect the
        # whole with ALLOWED_SCOPES so the result stays read-only by
        # construction even if a skill (or the definition) named a scope
        # outside the read vocabulary.
        scopes = sorted({
            scope
            for scope in list(definition.get("scopes") or [])
            + list(skills.get("scopes") or [])
            if scope in ALLOWED_SCOPES
        })
        web_access = definition.get("web_access")
        config = {
            "custom_agent": {
                "id": request["custom_agent_id"],
                "version": request["custom_agent_version"],
                "instructions": append_skill_block(
                    str(definition.get("instructions") or ""),
                    str(skills.get("instructions") or ""),
                ),
                "scopes": scopes,
                "permissions": [],
                "web_access": web_access if isinstance(web_access, Mapping) else None,
            },
            "model": {
                "base_url": backing.get("base_url", ""),
                "model": backing.get("model", ""),
                "api_key": backing.get("api_key", ""),
            },
            "mcp": {
                "specs": list(surface["specs"]),
                "effective_pairs": [list(pair) for pair in surface["pairs"]],
                "servers": list(surface["servers"]),
            },
            "listen": {"host": RUNTIME_LOOPBACK_HOST, "port": int(port)},
            # What this runtime is started with, secret-free: the bridge reads
            # it back so the reconcile can tell a stale runtime (ACC-074).
            "surface_digest": self._digest(
                definition, surface, skills,
                self._identity(request.get("model_deployment_name")),
            ),
        }
        if memory_token:
            config["memory"] = memory_client_config(
                request["advertise_address"], memory_token
            )
            # The wake block (idle scale-to-zero, 2026-09-28) is the memory
            # block - same bearer, same TLS pin - aimed at the wake route.
            config["wake"] = dict(
                config["memory"], endpoint=wake_endpoint(config["memory"]["endpoint"]),
            )
        return json.dumps(config, separators=(",", ":"))

    def _identity(self, model_deployment_name: Any) -> Dict[str, Any]:
        """The backing identity for the digest; unreadable parts are left blank."""
        if self.cluster_store is None:
            return {"deployment": str(model_deployment_name or "")}
        try:
            rows = list(self.broker.list() or [])
        except Exception:  # noqa: BLE001 - the digest then carries no version
            rows = []
        try:
            return backing_identity(self.cluster_store, str(model_deployment_name or ""), rows)
        except Exception:  # noqa: BLE001 - an unreadable store leaves the identity blank
            return {"deployment": str(model_deployment_name or "")}

    def _await_health(self, health_url: str) -> bool:
        """Poll the inbound gate's ``/health`` for a real signal, bounded (design D4)."""
        deadline = self._monotonic() + HEALTH_WAIT_SECONDS
        while self._monotonic() < deadline:
            if self._probe(health_url):
                return True
            self._sleep(HEALTH_POLL_SECONDS)
        return bool(self._probe(health_url))

    def _rollback(self, name: str, key_id: str, error: Exception) -> None:
        """Strand nothing after a mid-deploy failure (design SF-E, mirrors GPU).

        The order is the one ``remove`` keeps: STOP the unit and gate first (so a
        launched runtime is not left holding its port and its key file), then
        REVOKE the minted inbound key, then mark the row ``failed`` carrying the
        reason. Each step is best-effort so the original error is never lost.
        """
        LOGGER.warning("Rolling back cluster agent %s: %s", name, error)
        self._stop_quietly(name)
        self._revoke_quietly(name, key_id)
        try:
            row = self.store.get(name)
        except Exception:  # noqa: BLE001 - a rollback never re-raises
            row = None
        if isinstance(row, Mapping):
            self._revoke_endpoint_keys_quietly(row)
        self._clear_memory_quietly(name)
        try:
            self.store.update_state(name, "failed")
        except Exception as mark_error:  # noqa: BLE001 - a rollback never re-raises
            LOGGER.warning("Could not mark %s failed: %s", name, mark_error)

    def _stop_quietly(self, name: str) -> None:
        """Ask the bridge to stop the unit and gate, swallowing any trouble."""
        try:
            self.bridge.agent_stop(name)
        except Exception as error:  # noqa: BLE001 - best-effort teardown
            LOGGER.warning("Could not stop cluster agent %s: %s", name, error)

    def _revoke_quietly(self, name: str, key_id: str, endpoint_id: str = "") -> None:
        """Revoke the minted inbound key by its stored id, swallowing any trouble."""
        if not key_id:
            return
        endpoint_id = endpoint_id or AGENT_ENDPOINT_PREFIX + str(self._endpoint_suffix(name, key_id))
        try:
            self.keys.revoke(key_id, endpoint_id)
        except Exception as error:  # noqa: BLE001 - best-effort teardown
            LOGGER.warning("Could not revoke the key for %s: %s", name, error)

    def _revoke_endpoint_keys_quietly(self, row: Mapping[str, Any]) -> None:
        """Revoke every ACTIVE key of the row's endpoint, not just the first.

        Keys minted from the console after the deploy are bound to the same
        ``agent:<id>`` endpoint; a removed agent must leave none of them live.
        Read from the broker's fingerprint-only listing; best-effort.
        """
        endpoint_id = AGENT_ENDPOINT_PREFIX + str(row.get("id") or "")
        try:
            listing = list(self.broker.list() or [])
        except Exception as error:  # noqa: BLE001 - best-effort teardown
            LOGGER.warning("Could not list the keys of %s: %s", row.get("name"), error)
            return
        for item in listing:
            if (item.get("provider") == SERVED_ENDPOINT_PROVIDER
                    and str(item.get("endpoint_id") or "") == endpoint_id
                    and not item.get("revoked")):
                try:
                    self.keys.revoke(str(item.get("id")), endpoint_id)
                except Exception as error:  # noqa: BLE001 - best-effort teardown
                    LOGGER.warning("Could not revoke a key of %s: %s", row.get("name"), error)

    def _clear_memory_quietly(self, name: str) -> None:
        """Clear this agent's memory-token hash, swallowing any trouble.

        A torn-down or failed agent's memory bearer must die with it, so the
        stored hash is emptied alongside the inbound-key revoke; the row holds
        only that hash, never the token, so nothing secret is touched here.
        """
        try:
            self.store.clear_memory_key(name)
        except Exception as error:  # noqa: BLE001 - best-effort teardown
            LOGGER.warning("Could not clear the memory key for %s: %s", name, error)

    def _endpoint_suffix(self, name: str, key_id: str) -> str:
        """The row id the key's endpoint is bound to, read back for the revoke.

        The endpoint id is ``agent:<row id>``; the revoke needs that binding to
        pass the served-endpoint boundary check. The row is re-read so a rollback
        that runs after the id is stored uses the durable value, and falls back to
        the name only if the row is already gone.
        """
        try:
            row = self.store.get(name)
        except Exception:  # noqa: BLE001 - a missing row is not fatal to revoke
            row = None
        if isinstance(row, Mapping) and row.get("id"):
            return str(row["id"])
        return name


