"""Lifecycle orchestration for vLLM GPU model serving across the GPU nodes.

VD-125: those nodes are the enrolled SSH workers **and the head controller
itself**. The controller cannot be its own SSH worker, so its commands travel
through the root hardware bridge (`vaelor.bridge_transport`) while a worker's go
over pinned SSH; `_transport` is the one place that choice is made, and both
objects present the same surface, so nothing below it branches on the answer.
The controller is otherwise an ordinary participant - the fit engine orders it
against the workers on measured memory and it can be the lead or a worker.


The GPU-tier sibling of `pooled_operations`. It follows the same shape - resolve
the chosen nodes, decide the placement, make the runtime available, obtain the
model, start the workers and the API, wait for a real health signal on the OpenAI
endpoint, register a credential-broker ``openai-compatible`` profile, persist the
deployment record, and roll everything back on any failure - but the engine is
**vLLM** (single-GPU, or vLLM + Ray + RCCL across GPU nodes) rather than the CPU
``distributed-llama`` mesh, and the placement decision is delegated to the
committed `cluster_gpu_sizing.plan_gpu_fit` fit engine.

Since MB1 the vLLM runtime is a CONTAINER, so "make the runtime available" is
``ensure_image`` plus a per-node resolution of the numeric render/video GIDs the
container needs to open the accelerator devices. Those GIDs differ per node and
are read from the node itself, never assumed. The cluster NIC each node's
collectives bind is per node for the same reason, but is NOT read here by
default: VD-125 makes it a node fact captured at cluster formation, so
`gpu_node_facts.cluster_interfaces` reads the enrolled record - below both
operator pins - and only then falls back to a live read. As always, this module
assembles no command line: `gpu_pool_runtime`
remains the one place that knows the docker, vLLM and Ray invocations.

Two rules from DECISIONS.md are load-bearing here and are why this mirrors the
pooled path rather than inventing a parallel one:

* **Health is a probe, never a claim (VD-040 / VD-054).** The record is marked
  ``healthy`` only after the endpoint answers ``/v1/models``; a written unit is
  not health.
* **Engine by capability (VD-056 / VD-057).** vLLM serves GPU nodes; CPU-only
  nodes stay on the pooled llama.cpp path. `select_serving_engine` is the one
  gate, and a mixed selection is refused rather than half-served.

The deployment record reuses the ``pooled_deployments`` store table, discriminated
by an ``engine`` field inside its ``units`` blob, so GPU and CPU pooled serving
share one persistence surface without a parallel table.

**What a record runs is DERIVED from it, never listed on it.** The units a
deployment owns are a function of its name and its node list
(`gpu_pool_units.serving_units`: the server on the lead, a Ray worker on each
other node), and `stop_units_for` stops exactly those - for `remove`, and for
the mode watch's pass over a deploy that died with the executor. A ``started``
list once stood in for this and was written only with the healthy or the failed
record, so the ``deploying`` row a dead deploy left behind derived nothing: the
watch restored Mode A over units still loading, and `remove` deleted the row
that named them. `gpu_pool_units`'s docstring carries the full account.

**Mode B is a switch, not just a deploy (VD-125).** When this controller is one
of the participants, the deploy is also the moment the appliance stops serving AI
Chat on llama.cpp and starts serving it on the cluster: `ClusterModeSwitch.enter`
clears the GPU before any node is touched, `repoint` moves AI Chat and the LLM
Server onto the healthy cluster endpoint, and `leave_if_owned` gives them back
on removal or on any rollback - through one name gate, so neither path can
tear down a cluster that a DIFFERENT deployment is serving on. The controller
is then always the LEAD and binds its OpenAI API on loopback, so the cluster
endpoint is reachable only through Vaelor's own auth proxy; a worker-led
cluster keeps the NIC binding and SAYS SO, because the proxy cannot front an
upstream on another machine.

**Two placements, one prefix.** The fit engine refuses a model that fits one
machine (VD-127, D6: that is AI Chat's, on llama.cpp), so the verdicts this
module makes a placement from are ``distributed`` (the capacity intent: one
server sharded over Ray) and ``replicated`` (the throughput intent, VD-129:
one whole copy per machine behind a balancer on this controller), and every
other verdict is raised as the engine's own sentence. Both share everything
up to the units - the nodes, the name, the fit, the port band, the mode
switch's `enter`, the image, the GIDs and the weights pull - and diverge only
at what is started: the distributed body is here, the replicated body is
`gpu_pool_replicas.ReplicatedServing`, and both come back to the same
credential, the same `repoint` and the same rollback.
"""

from __future__ import annotations

import logging
import time
from contextlib import nullcontext
from typing import Any, Callable, Dict, List, Mapping, Optional

from . import cluster_link
from . import gpu_render_ledger as render_ledger
from .bridge_transport import default_bridge_transport
from .cluster_capacity import gpu_fit_node
from .cluster_gpu_sizing import (
    INTENT_CAPACITY, VERDICT_DISTRIBUTED, VERDICT_REPLICATED, build_model_spec,
    plan_gpu_fit, validate_gpu_memory_utilization,
)
from .cluster_placement import CONTROLLER_PLACEMENT_ID
from .gpu_cluster_mode import UNLOADED_STATE
from .gpu_model_catalog import GPU_MODEL_CATALOG, catalog_spec_body, repo_of_source
from .gpu_node_facts import interface_name
from .gpu_pool_pull import ModelPullWaiter
from .gpu_pool_removal import (  # noqa: F401 - UNITS_NOT_STOPPED is re-exported
    UNITS_NOT_STOPPED, remove_gpu_deployment, stop_derived_units, stop_placements,
)
from .gpu_pool_replicas import (
    BalancerController, ReplicaHealth, ReplicaStartupWaiter, ReplicatedServing,
    default_balancer, mint_replica_key, replica_health_url, replica_ports,
    rotate_cluster_serving_key, stop_balancer,
)
from .gpu_pool_startup import ServerStartupWaiter, probe_health
from .gpu_pool_units import (
    LLAMACPP_ENGINE, RAY_WORKER_ROLE, SERVER_ROLE, VLLM_ENGINE, deployment_name, lead_first,
)
from .gpu_serving_target import (
    SERVER_LOOPBACK_HOST, register_cluster_credential,
    serving_port, switch_gpu_chat_reclaimable,
)
from .gpu_idle_watch import idle_timeout_from_payload
from .hf_model_source import resolve_model_source
from .model_thinking import THINKING_FIELD, thinking_for_deploy
#: How THIS model is served - parsers, text-only, MTP, tuned MoE tables - is
#: decided once, from its profile and the payload (`vllm_serve_options`).
from .hf_cached_config import with_cached_config
from .gpu_pool_refit import WEIGHT_FIELD, refit_after_fetch
from .gpu_ray_plane import confirm_unit_in_slice, prepare_split
from .gpu_pool_refit import fit_nodes as refit_fit_nodes
from .vllm_model_profile import kernel_note, profile_for
from .vllm_serve_options import (
    CONFIG_READ_FIELD, IMAGE_FIELD, MTP_FIELD, OPTIONS_FIELD, cache_note,
    options_for,
)
from .vllm_images import download_line, image_profile
from .cluster_gpu_sizing import with_launch_cache
from .cluster_split_mode import (
    SPLIT_MODE_FIELD, SPLIT_TENSOR, degrees, parse_split_mode, split_devices, tensor_refusal,
)
from .cluster_link_recommendation import SPLIT_LINK_FIELD, parse_split_link, split_link_for
from .ssh_transport import SshTransport

LOGGER = logging.getLogger(__name__)

#: What a GPU deploy is refused with when it is asked to make the cluster serve
#: the on-device Assistant (VD-125, D4). The Assistant keeps its own model on its
#: own accelerator - on a Z2 the NPU - and the whole point of the mode switch is
#: that GPU clustering moves AI Chat and the LLM Server, and nothing else. Left
#: as an option it would silently retire the NPU tier the moment the cluster came
#: down, which no teardown path restores. VD-202 item 1 widened it to every
#: cluster model (CPU single, replicated and pooled too), so the one refusal sits
#: in `ClusterOperations.deploy_llm`; this path keeps its own as well.
ASSISTANT_NOT_CLUSTERED = (
    "The Assistant keeps its own installed model; a cluster model serves AI "
    "Chat and the LLM Server."
)


def select_serving_engine(nodes: List[Dict[str, Any]]) -> str:
    """Return the serving engine for a set of chosen nodes, by capability.

    GPU-capable nodes serve on **vLLM** (this backend); CPU-only nodes serve on
    the existing pooled **llama.cpp** path. A selection that mixes the two has no
    single engine and is refused with the reason rather than half-served on one -
    the honest-degradation rule applied to placement. The GPU signal is the
    capacity ledger's discovered ``inventory.gpu.present``, never an assumption
    from the model or the node name.
    """
    if not nodes:
        raise ValueError("Choose at least one node to serve the model on.")
    gpu = [node for node in nodes if _gpu_present(node)]
    cpu = [node for node in nodes if not _gpu_present(node)]
    if gpu and cpu:
        raise ValueError(
            "Choose either GPU nodes (served with vLLM) or CPU-only nodes "
            "(served with pooled llama.cpp), not a mix of the two."
        )
    return VLLM_ENGINE if gpu else LLAMACPP_ENGINE


def _gpu_present(node: Dict[str, Any]) -> bool:
    return bool(node.get("inventory", {}).get("gpu", {}).get("present"))


class GpuPoolOperations:
    """Deploy and remove vLLM GPU model serving, mirroring the pooled operations."""

    def __init__(
        self, *, store, broker, runtime,
        joined_node: Callable[..., Dict[str, Any]],
        advertise_address: Callable[[Any], str],
        probe: Optional[Callable[[str], bool]] = None,
        sleep: Callable[[float], None] = time.sleep,
        monotonic: Callable[[], float] = time.monotonic,
        bridge_transport_factory: Optional[
            Callable[[Dict[str, Any]], Any]
        ] = None,
        mode_switch: Optional[Any] = None,
        balancer: Optional[BalancerController] = None,
        split_link: Optional[Callable[[], Any]] = None,
    ):
        self.store = store
        self.broker = broker
        self.runtime = runtime
        self.joined_node = joined_node
        self.advertise_address = advertise_address
        # VD-162: the owner's chosen cluster link as it is NOW, read when a
        # split is deployed. `ClusterOperations` wires the stored choice;
        # an instance built without one deploys as before the setting.
        self.split_link = split_link or (lambda: None)
        # The GPU serving mode switch (VD-125), OPTIONAL by design (D12). Only
        # the EXECUTOR's instance gets a real one: it runs in the process that
        # holds the GPU failure-watch and can drive the root bridge, so it is the
        # only place a switch can stop llama.cpp and be sure it stayed stopped.
        # The control plane's instance has none, and every switch call below is
        # gated on that - a preview-side object must never move AI Chat.
        self.mode_switch = mode_switch
        # How the CONTROLLER's commands leave this process (VD-125). An
        # enrolled worker is reached over SSH with a broker credential; the
        # controller has none - it is the Swarm manager, not a member - so its
        # commands go through the root hardware bridge instead. Injectable so a
        # test can prove which node got which transport.
        self._bridge_transport_factory = (
            bridge_transport_factory or default_bridge_transport
        )
        # The endpoint readiness probe is injectable so the deploy path can be
        # unit-tested without a live HTTP server, while defaulting to the real
        # GET (`gpu_pool_startup.probe_health`, which also tells "answered
        # unhealthy" from "nothing answered" for the mode watch's row). The
        # poll wait and the clock are injectable for the same reason and in
        # the same shape `GpuModelLibrary` uses: a test can drive a multi-poll
        # pull, or an hour-long compile, without waiting for either. ONE
        # clock, handed to both waiters.
        self._probe = probe or probe_health
        self._sleep = sleep
        self._monotonic = monotonic
        self._pull = ModelPullWaiter(runtime, sleep=sleep, monotonic=monotonic)
        # The probe is handed over as a thunk so ``_probe`` stays a live seam:
        # a test that swaps it after construction is still what the wait asks.
        self._startup = ServerStartupWaiter(
            runtime, probe=lambda url: self._probe(url),
            sleep=sleep, monotonic=monotonic,
        )
        # The throughput intent's endpoint (VD-129): the balancer this
        # controller runs, driven over the root bridge like the LLM Server
        # proxy. Held here because three things stop or start it - the
        # deploy, `stop_units_for`, and the mode switch's reconcile - and the
        # last is handed the SAME controller by `executor_mode_switch`.
        # Injectable so a test can put a recording fake behind it.
        self.balancer = balancer or default_balancer()
        # The mode watch's reading of a replicated row's replicas, over the
        # same probe the deploy's own health checks use.
        self.replica_health = ReplicaHealth(lambda url: self._probe(url))
        self._replicas = ReplicatedServing(
            runtime, self.balancer, advertise_address=advertise_address,
            waiter=ReplicaStartupWaiter(
                runtime, probe=lambda url: self._probe(url),
                sleep=sleep, monotonic=monotonic,
            ),
        )

    def _transport(self, node: Dict[str, Any]):
        """How this participant's commands are carried, by which node it is.

        The controller runs them through the root hardware bridge and every
        enrolled worker over pinned SSH. Both objects present the same ``run``
        and ``profile`` surface, so nothing downstream - `GpuPoolRuntime`,
        `gpu_node_facts`, the rollback - branches on the answer.
        """
        if str(node.get("id", "")) == CONTROLLER_PLACEMENT_ID:
            return self._bridge_transport_factory(node)
        return SshTransport(
            self.broker.resolve(node["credential_id"], "cluster-node"),
            timeout=120,
        )

    def _stop_each(self, placements: List[Any]) -> List[Dict[str, str]]:
        """Stop placements, last started first (`gpu_pool_removal.stop_placements`)."""
        return stop_placements(self, placements)

    def stop_units_for(
        self, deployment: Mapping[str, Any]
    ) -> List[Dict[str, str]]:
        """Stop what a record derives, each on its node (`gpu_pool_removal.stop_derived_units`)."""
        return stop_derived_units(self, deployment)

    def deploy(self, payload: Dict[str, Any], progress=None) -> Dict[str, Any]:
        node_ids = self._node_ids(payload)
        nodes = [
            self.joined_node(node_id, include_credential=True)
            for node_id in node_ids
        ]
        engine = select_serving_engine(nodes)
        if engine != VLLM_ENGINE:
            raise ValueError(
                "This backend serves GPU nodes with vLLM; CPU-only nodes use "
                "pooled llama.cpp inference."
            )

        # The name rule lives in one place (gpu_pool_units.deployment_name) so
        # this entry point and the runtime cannot drift on it.
        name = deployment_name(str(payload.get("name", "")))
        existing = self.store.get_pooled_deployment(name)
        if existing and existing.get("state") in {"deploying", "healthy"}:
            raise ValueError("A deployment already uses that name.")
        if existing and existing.get("state") == UNLOADED_STATE:
            raise ValueError(
                "'{}' is unloaded, not gone. Load it to serve it warm, or remove "
                "it first.".format(name)
            )

        if payload.get("use_for_assistant"):
            # D4, refused up front so nothing is started for a request the
            # teardown could never undo.
            raise ValueError(ASSISTANT_NOT_CLUSTERED)

        repo, revision, spec = self._model_and_spec(payload)
        # Whether the model thinks before answering unless a request asks
        # (owner decision 2026-09-29): off unless the payload says True, and
        # refused here - before anything is written - when it is not a boolean
        # or is turned on for a model that has no thinking switch.
        thinking = thinking_for_deploy(repo, payload)
        by_id = {node["id"]: node for node in nodes}
        # The controller's fit node counts the memory the mode switch will free
        # (D1), so this deploy and the `/cluster/fit` preview that led to it size
        # against the same number (`gpu_pool_refit.fit_nodes`, which a Load's
        # fit reads too).
        fit_nodes = refit_fit_nodes(self.broker, self.mode_switch, nodes)
        intent = str(payload.get("intent", "") or INTENT_CAPACITY)
        # The launch fraction is validated and defaulted by the fit engine
        # (VD-129) and handed to it, so the verdict below is decided on the
        # number vLLM will actually be launched with - preview equals deploy
        # on the figure that decides a replica's fit.
        gpu_memory_utilization = validate_gpu_memory_utilization(
            payload.get("gpu_memory_utilization"), intent=intent,
        )
        # Sized for the cache this launch keeps - multi-token prediction's
        # draft layer and state, and a hybrid model's bf16 blocks on these
        # machines' GPUs - by the call the fit preview makes; refused here if
        # the prediction choice cannot be had.
        gpu_targets = [node["gfx_target_version"] for node in fit_nodes]
        spec = with_launch_cache(
            spec, payload.get(MTP_FIELD), vllm_image=payload.get(IMAGE_FIELD),
            gpu_targets=gpu_targets,
        )
        plan = plan_gpu_fit(
            spec, fit_nodes, link=str(payload.get("link", "cross-node")),
            intent=intent, gpu_memory_utilization=gpu_memory_utilization,
            max_model_len=payload.get("max_model_len"),
        )
        placement = self._placement(plan, by_id, payload, spec)
        # The context vLLM is launched at is the one the fit sized the KV
        # cache at - the engine validated the payload's and echoes it - so a
        # typed context cannot preview at the model's full window and launch
        # at a fraction of it (VD-129, one derivation).
        max_model_len = int(plan["max_model_len"])
        replicated = placement["mode"] == VERDICT_REPLICATED
        # The model's serving options, from what its config.json (or name)
        # says and the owner's two choices - refused HERE, with nothing
        # written, when a choice cannot be honoured (MTP on a model without
        # the head, or across a model split between machines).
        vllm_options = options_for(
            profile_for(repo, spec.facts), payload, replicated=replicated,
            gpu_targets=gpu_targets,
        ).as_record()

        # D11: the requested port goes through the runtime's own band rule
        # BEFORE it is used, so an out-of-band port is a refusal with nothing
        # written rather than a unit file that collides with Mode A's allocator.
        # A replicated deployment takes two: the balancer's, and the next one
        # for every replica (VD-129), both in the band - so 8079 is refused.
        replica_port = None
        if replicated:
            port, replica_port = replica_ports(payload.get("port", 8000))
        else:
            port = serving_port(payload.get("port", 8000))
        # OPTIONAL pins, not defaults, and validated here - before the deploying
        # record is written and long before anything is started - so a typo is a
        # refusal with nothing to roll back. Left empty, which is the normal
        # case, each node's cluster NIC comes from the fact enrolment captured;
        # see `gpu_node_facts.cluster_interfaces`.
        pinned_interface = str(payload.get("interface", "") or "").strip()
        node_pins: Dict[str, str] = {}
        raw_pins = payload.get("interfaces") or {}
        # A written refusal, not whatever `dict()` says about a str or a list.
        # ``dict("eth0")`` raises about a "dictionary update sequence element",
        # which names nothing an operator typed and reads as a crash.
        if not isinstance(raw_pins, dict):
            raise ValueError(
                "The per-node interface pins must be a mapping of node id to "
                "interface name."
            )
        for node_id, value in raw_pins.items():
            # A per-node pin is refused by the NODE it was typed for: the rule's
            # own sentence says what is wrong with the name but not which
            # machine's entry to go and correct, and this map carries one pin
            # per participant. Prefix it, never respell it.
            node_id = str(node_id)
            try:
                node_pins[node_id] = interface_name(value)
            except ValueError as error:
                label = (by_id.get(node_id) or {}).get("name") or node_id
                raise ValueError(f"{label}: {error}") from error
        if pinned_interface:
            pinned_interface = interface_name(pinned_interface)
        # VD-162: read and checked HERE, before the deploying record and
        # before Mode A is stopped - a chosen link that is gone, or a
        # deploy that also pins interfaces, is a refusal with nothing to
        # roll back. Replicas run no collective and never read the link.
        # The owner's link choice (the form's `split_link`): Ethernet never
        # reads the chosen link; nothing sent keeps the Setup choice.
        # Pins beside the cluster link are refused as the preview refuses them.
        choice = parse_split_link(payload.get(SPLIT_LINK_FIELD))
        link = None if replicated else split_link_for(
            choice, self.split_link, node_pins, pinned_interface)
        # No link chosen: the split rides each machine's Ethernet, fenced by
        # socket owner (owner 2026-10-01, ACC-187; `gpu_ray_plane`).
        # D2: when the controller participates it is ALWAYS the lead - the lead
        # is the node whose merged Ray-head-and-server container publishes the
        # OpenAI API, and only on this machine can that API bind loopback behind
        # Vaelor's own auth proxy. The fit engine still decides WHICH nodes; this
        # only decides which of them publishes.
        participants = lead_first(placement["nodes"])
        participant_ids = [node["id"] for node in participants]
        # A pin is an explicit operator instruction about ONE machine, so a pin
        # for a machine this deployment will not serve on cannot be quietly
        # dropped: the operator would be left believing a NIC was pinned while
        # the machine they meant took a different answer entirely. Typing an id
        # that is not a participant is a mistake either way - an id that does
        # not exist, or one the fit engine did not place - and both are told
        # here, before the deploying record is written and before any node is
        # touched, naming the id that has nowhere to land.
        for node_id in node_pins:
            if node_id not in participant_ids:
                raise ValueError(
                    "The cluster interface pinned for '{}' cannot be used: no "
                    "such machine is serving this deployment ({}).".format(
                        node_id, ", ".join(participant_ids),
                    )
                )
        # A replica the balancer could not pool (VD-129) is refused here,
        # beside the pin refusal, for the same reason: before the row, before
        # `enter` stops Mode A, before a pull - rather than at the balancer's
        # start after both replicas have loaded.
        if replicated:
            self._replicas.refuse_unroutable(participants, replica_port)
        controller_leads = participant_ids[0] == CONTROLLER_PLACEMENT_ID

        report = progress or (lambda _percent, _message: None)
        # ``(transport, node, unit)`` per unit actually started, in start order,
        # for the rollback in THIS process: it holds the transports and knows
        # exactly what it started. It is not written to the record - the record
        # derives its units (`stop_units_for`), and a list that was only ever
        # written with the final record was absent from the one row a deploy
        # that died with its executor leaves behind.
        started: List[Any] = []
        # Pull oneshots currently in flight, by node id. `_pull_model` clears an
        # entry as soon as it stops one, so what survives here is what a failure
        # left behind - the record carries it, and `remove` finishes the job.
        pulls: Dict[str, str] = {}
        credential_id = ""
        # The mode is on the DEPLOYING row (VD-129): the units a dead deploy's
        # row derives depend on it, and a row that said only the engine would
        # derive the union of both shapes - correct, but a read per unit the
        # record could have spared.
        # ACC-191: the serving options - the chosen image above all, the
        # default resolved in - ride on the deploying row too, so a deploy
        # that fails is listed on the image the owner chose.
        row_units = {"engine": VLLM_ENGINE, "mode": placement["mode"],
                     OPTIONS_FIELD: vllm_options}
        if not replicated:  # its Ray units carry the start check (ACC-187)
            row_units["slice_check"] = True
        if replicated:
            row_units["replica_port"] = replica_port
        self.store.put_pooled_deployment(
            name=name, state="deploying", model_id=repo,
            node_ids=participant_ids, units=dict(row_units),
        )
        # Mode A comes down BEFORE the first node is touched, and `enter` does
        # not return until the GPU gave the memory back. It runs INSIDE the try:
        # `enter` writes the Mode B record before it stops anything, so a refusal
        # (the model would not leave) rolls back through `leave_if_owned` to a
        # removable `failed` record; a refusal because ANOTHER deployment holds
        # the switch wrote nothing, and the same gate then leaves nothing (D9).
        entered = self._switch_owns_deploy(controller_leads)
        try:
            if entered:
                self.mode_switch.enter(name, lambda note: report(8, note))
            transports = {
                node["id"]: self._transport(node) for node in participants
            }
            group_ids: Dict[str, List[int]] = {}
            # Every participant's NIC is settled FIRST, before the image, the
            # weights or any unit: a node that cannot name one is a refusal
            # with nothing started and nothing to roll back, which is the whole
            # point of making this a fact rather than a guess. What each node
            # was actually told to bind, and where that came from, go on the
            # record so it answers both rather than implying one fleet-wide
            # name. Replicas run no collective, so no interface is resolved
            # for them and no RCCL environment is set (VD-129).
            interfaces: Dict[str, str] = {}
            interface_sources: Dict[str, str] = {}
            addresses: Dict[str, str] = {}
            if not replicated:
                addresses, interfaces, interface_sources = cluster_link.split_bindings(
                    participants, transports, node_pins, pinned_interface,
                    advertise_address=self.advertise_address,
                    resolve=self.runtime.resolve_interface,
                    link=link, read_tables=self.runtime.read_link_tables,
                )
            for index, node in enumerate(participants):
                report(
                    10 + int(25 * index / len(participants)),
                    f"Preparing the vLLM container image on {node['name']}",
                )
                self.runtime.ensure_image(
                    transports[node["id"]], image=vllm_options[IMAGE_FIELD],
                    on_download=lambda profile, node=node: report(
                        12, download_line(profile, node["name"]),
                    ),
                )
                # Read on the node, never assumed: the render/video GIDs the
                # container needs to open /dev/kfd and /dev/dri differ per box.
                group_ids[node["id"]] = self.runtime.resolve_group_ids(
                    transports[node["id"]]
                )
            for index, node in enumerate(participants):
                report(
                    35 + int(20 * index / len(participants)),
                    f"Fetching {repo} on {node['name']}",
                )
                self._pull.wait(
                    transports[node["id"]], node, repo, revision, report, pulls,
                    image=vllm_options[IMAGE_FIELD],
                )
            # A model whose config.json was not in the library when this
            # deploy began has just been fetched: the fit is asked again with
            # the model's own layout before anything starts, and the options
            # are decided from it (`gpu_pool_refit`).
            spec, served_facts, refitted = refit_after_fetch(
                self, payload, repo, revision, spec=spec, nodes=fit_nodes,
                by_id=by_id, participants=participants, intent=intent,
                fraction=gpu_memory_utilization, gpu_targets=gpu_targets,
                replicated=replicated,
            )
            vllm_options = refitted or vllm_options

            api_key = ""
            with render_ledger.recording() as renders:
                if replicated:
                    units, endpoint, api_key = self._replicas.serve(
                        name=name, participants=participants, transports=transports,
                        group_ids=group_ids, repo=repo, revision=revision,
                        port=port, replica_port=replica_port,
                        gpu_memory_utilization=gpu_memory_utilization,
                        max_model_len=max_model_len, payload=payload,
                        report=report, started=started, thinking_default=thinking,
                        vllm_options=vllm_options,
                    )
                else:
                    units, endpoint, api_key = self._serve_distributed(
                        name=name, participants=participants, transports=transports,
                        group_ids=group_ids, interfaces=interfaces,
                        interface_sources=interface_sources,
                        addresses=addresses, repo=repo,
                        revision=revision, port=port,
                        gpu_memory_utilization=gpu_memory_utilization,
                        max_model_len=max_model_len, placement=placement,
                        payload=payload, report=report, started=started,
                        thinking_default=thinking, vllm_options=vllm_options,
                    )
            units[render_ledger.RENDERS_FIELD] = render_ledger.entries(renders, transports)

            units["idle_timeout"] = idle_timeout_from_payload(payload)
            credential_id = register_cluster_credential(
                self.broker, name, endpoint, repo, api_key, payload, report,
            )
            if entered:
                # The cluster is healthy and tested: AI Chat and, if it was on
                # before, the LLM Server move onto it now - never earlier, so a
                # credential is only ever activated for an endpoint that answers.
                # The name goes with it: `repoint` refuses a mode file that does
                # not record THIS deployment as the one holding the switch.
                self.mode_switch.repoint(credential_id, port, name)
            # Whether the options were decided with the model's config read:
            # a Load decides them again when they were not (`options_for_load`).
            units[CONFIG_READ_FIELD] = served_facts is not None
            # The weight size the fit was made on, which a Load's fit needs.
            units[WEIGHT_FIELD] = int(spec.weight_bytes)
            self.store.put_pooled_deployment(
                name=name, state="healthy", model_id=repo,
                node_ids=participant_ids, units=units, endpoint=endpoint,
                credential_id=credential_id,
            )
        except Exception as error:
            # D9, the order `remove` keeps: stop, leave, THEN delete. `delete`
            # cascades a credential's purpose assignments, so deleting first
            # would clear ``ai-chat`` out from under the restore `leave` makes.
            # Stopping first is what makes that restore true, not optimistic.
            not_stopped = self._stop_each(started)
            # The balancer goes with the replicas (VD-129), before the leave,
            # which stops it again under the lock - an idempotent stop, so
            # a rollback that reached the balancer and one that did not read
            # the same way.
            if replicated:
                not_stopped += stop_balancer(self.balancer, name)
            # Through the ONE name gate `remove` uses: only if the file names
            # THIS deployment. A deploy `enter` refused because another
            # deployment holds the switch must not leave that deployment's Mode
            # B on its way out. Non-raising, so a rollback never loses the
            # original error.
            if entered:
                self.mode_switch.leave_if_owned(name)
            if credential_id:
                try:
                    self.broker.delete(credential_id)
                except Exception:
                    pass
            # The FAILED record carries the GPU marker `remove` gates on, the
            # node list its units derive from, WHY it failed (the exception's
            # own sentence, under the ``failure`` key the console reads and the
            # watch's abandoned pass writes), what the rollback could NOT stop
            # (by node and unit), and the pull oneshots nobody stopped - a fact
            # only this deploy knows, a pull being named per repo not record.
            self.store.put_pooled_deployment(
                name=name, state="failed", model_id=repo,
                node_ids=participant_ids,
                units={
                    **row_units,
                    "failure": str(error),
                    "unstopped": not_stopped,
                    "pulls": [
                        {"node_id": node_id, "unit": unit}
                        for node_id, unit in pulls.items()
                    ],
                    "repo": repo,
                    "revision": revision,
                    "port": port,
                },
            )
            raise

        report(100, "vLLM serving is healthy behind the managed gateway")
        return {
            "name": name,
            "deployment_mode": "gpu",
            "engine": VLLM_ENGINE,
            "mode": placement["mode"],
            "parallelism": placement["parallelism"],
            "tensor_parallel_size": placement["tensor_parallel_size"],
            "pipeline_parallel_size": placement["pipeline_parallel_size"],
            "node_ids": participant_ids,
            "lead_node_id": participants[0]["id"],
            "worker_node_ids": participant_ids[1:],
            "replica_count": len(participants) if replicated else 0,
            "units": units,
            "endpoint": endpoint,
            "credential_id": credential_id,
            "model": {"repo": repo, "revision": revision},
            "fit": plan["verdict"],
            # Never LAN-open now: a worker lead is behind its keyed gate
            # (ACC-162). Kept on the result for the readers of the word.
            "lan_exposed": False,
            "exposure": "",
            "mode_switched": entered,
            # How fast this model's 4-bit weights will run in the image it was
            # deployed on, in plain words ("" when there is nothing to say).
            "kernel_note": kernel_note(
                profile_for(repo, served_facts).weight_format,
                image_profile(vllm_options[IMAGE_FIELD]).fast_w4a16,
            ),
            # Why a hybrid model got no cache settings, when a fact they are
            # decided from could not be read ("" otherwise).
            "cache_note": cache_note(
                profile_for(repo, served_facts),
                image_profile(vllm_options[IMAGE_FIELD]), fit_nodes,
            ),
        }

    def _serve_distributed(
        self, *, name, participants, transports, group_ids, interfaces,
        interface_sources, repo, revision, port, gpu_memory_utilization,
        max_model_len, placement, payload, report,
        started, thinking_default=False, vllm_options=None, addresses=None,
    ):
        """The capacity intent's body: one sharded server over Ray; ``(units, endpoint, key)``.

        The lead's Ray head and its vLLM server are ONE container, so the
        lead starts FIRST and the workers join it: under the Ray backend
        `vllm serve` waits for its placement group's GPUs, so a worker
        arriving after the server is exactly the expected order. Answers the
        healthy record's ``units`` and the endpoint; the replicated twin is
        `gpu_pool_replicas.ReplicatedServing.serve`.

        **The API binds loopback on every lead** (VD-156 amended, ACC-162).
        A WORKER lead gets the keyed gate a worker replica has, on its cluster
        address and the serving port, with a minted key the credential then
        carries - the only way the controller reaches it; readiness is the
        gate's unkeyed ``/health``. This controller's lead needs no gate.
        """
        lead = participants[0]
        controller_leads = lead["id"] == CONTROLLER_PLACEMENT_ID
        # The address each node binds: the one the deploy settled (the
        # cluster link's, VD-162), else its enrolled address - which is
        # also what a record written before `addresses` re-serves on.
        bound = {
            node["id"]: (addresses or {}).get(node["id"])
            or self.advertise_address(node["host"])
            for node in participants
        }
        lead_ip = bound[lead["id"]]
        lead_transport = transports[lead["id"]]
        # Every machine is guarded BEFORE any Ray process starts (ACC-163).
        prepare_split(self.runtime, name, participants, transports,
                      self.advertise_address, started, bound, interfaces)
        report(60, f"Starting the distributed vLLM server on {lead['name']}")
        server_unit = self.runtime.start_distributed_server(
            lead_transport, name=name,
            model=repo,
            port=port,
            tensor_parallel_size=placement["tensor_parallel_size"],
            pipeline_parallel_size=placement["pipeline_parallel_size"],
            gpu_memory_utilization=gpu_memory_utilization,
            max_model_len=max_model_len,
            bind_ip=lead_ip, interface=interfaces[lead["id"]],
            device_count=gpu_fit_node(lead)["device_count"],
            group_ids=group_ids[lead["id"]], revision=revision,
            api_host=SERVER_LOOPBACK_HOST, thinking_default=thinking_default,
            vllm_options=vllm_options,
        )
        started.append((lead_transport, lead, server_unit))
        # ACC-187 review 1 (SC1): the fence covers only the slice, so each Ray
        # container must be seen running in it - or the rollback stops it.
        self._confirm_in_slice(lead_transport, name, SERVER_ROLE, server_unit)
        api_key, gate, probe_url = "", "", None
        endpoint = "http://{}:{}/v1".format(SERVER_LOOPBACK_HOST, port)
        if not controller_leads:
            api_key = mint_replica_key()
            # On the address this controller reaches the lead on - never a
            # dedicated link, which only the split's machines are on.
            reach = self.advertise_address(lead["host"])
            gate = self.runtime.start_gate(
                lead_transport, name=name, listen_host=reach, port=port,
                api_key=api_key,
            )
            started.append((lead_transport, lead, gate))
            endpoint = "http://{}:{}/v1".format(reach, port)
            probe_url = replica_health_url(reach, port)
        worker_units = []
        for node in participants[1:]:
            worker_ip = bound[node["id"]]
            report(70, f"Joining {node['name']} to the Ray cluster")
            worker_unit = self.runtime.start_ray_worker(
                transports[node["id"]], name=name, head_ip=lead_ip,
                bind_ip=worker_ip, interface=interfaces[node["id"]],
                group_ids=group_ids[node["id"]],
                device_count=gpu_fit_node(node)["device_count"],
                vllm_options=vllm_options,
            )
            worker_units.append(worker_unit)
            started.append((transports[node["id"]], node, worker_unit))
            self._confirm_in_slice(transports[node["id"]], name, RAY_WORKER_ROLE, worker_unit)
        # Waited for while the server is ALIVE, read through the lead's own
        # transport (the bridge when this controller leads), and no longer:
        # a clock alone killed a live server mid-compile (VD-127).
        self._wait_for_api(
            endpoint, payload, report, lead_transport, lead, server_unit,
            probe_url,
        )
        units = {
            "engine": VLLM_ENGINE,
            "mode": placement["mode"],
            # Its Ray units check the driver and the slice at every start
            # (ACC-187 review 2); a split recorded without this predates it.
            "slice_check": True,
            "lan_exposed": False,
            "api_host": SERVER_LOOPBACK_HOST,
            "parallelism": placement["parallelism"],
            "tensor_parallel_size": placement["tensor_parallel_size"],
            "pipeline_parallel_size": placement["pipeline_parallel_size"],
            "gpu_memory_utilization": gpu_memory_utilization,
            "max_model_len": max_model_len,
            "server": server_unit,
            # A worker lead's keyed gate ("" when this controller leads).
            "gate": gate,
            # No separate head unit exists to record: it lives inside the
            # server container above.
            "ray": {"head": "", "workers": worker_units},
            "interfaces": interfaces,
            "interface_sources": interface_sources,
            # What each node was told to bind, so a Load renders the same.
            "addresses": bound,
            "repo": repo,
            "revision": revision,
            "port": port,
            THINKING_FIELD: bool(thinking_default),
            # What the units were rendered with, so a Load renders the same.
            OPTIONS_FIELD: vllm_options,
        }
        return units, endpoint, api_key

    def _serving_lock(self):
        """The GPU serving mutex, when this instance runs beside the mode watch.

        The switch's own ``watch_lock`` (:data:`~vaelor.gpu_cluster_mode.GPU_SERVING_LOCK`
        in production) - the one lock the watch, the reconcile, `enter` and
        `leave` hold - so the record write `remove` makes is serialised with
        the watch's. The control plane's instance has no switch and no watch
        thread beside it (D12), and holds nothing.
        """
        if self.mode_switch is None:
            return nullcontext()
        return self.mode_switch.watch_lock

    def _switch_owns_deploy(self, controller_leads: bool) -> bool:
        """Whether the mode switch owns this deploy, asked BEFORE it is used.

        Only when this controller is the lead: the switch's whole job is to free
        THIS machine's GPU and move THIS machine's AI Chat, and a cluster that
        does not include the controller leaves both alone. Combined with
        `_lead_first`, "the controller leads" and "the controller participates"
        are the same condition, so this is the one gate.

        Separate from the `enter` call so the answer exists BEFORE `enter` runs.
        `enter` records Mode B and then refuses if the model will not leave the
        GPU; a flag set from its return value would have been false in exactly
        that case, so the rollback would skip the leave and strand the box in
        Mode B with a `failed` record nothing could remove. And it says "would
        use the switch", never "took it": whether THIS deployment holds the
        switch is the file's to answer, through `leave_if_owned`.
        """
        return self.mode_switch is not None and bool(controller_leads)

    def _node_ids(self, payload: Dict[str, Any]) -> List[str]:
        raw_ids = payload.get("node_ids", [])
        if not isinstance(raw_ids, list):
            raise ValueError("Choose the GPU nodes to serve the model on.")
        node_ids = list(dict.fromkeys(
            str(node_id).strip() for node_id in raw_ids
            if str(node_id).strip()
        ))
        if not node_ids:
            raise ValueError("Choose at least one GPU node.")
        # The controller is an ordinary participant here (VD-125). It has no
        # SSH credential - it cannot be enrolled as its own worker - but its
        # commands travel through the root bridge instead (`_transport`), and
        # its GPU is half of the only two-node cluster the product ships into.
        # The fit engine orders it against the workers on measured memory like
        # any other node, so it can be the lead or a worker; nothing here
        # special-cases which. The CPU pooled path still refuses it, because it
        # has no such transport.
        return node_ids

    def _model_and_spec(self, payload: Dict[str, Any]):
        """Resolve ``model_source`` to ``(repo, revision, spec)``.

        A catalog id contributes both the weights repo and the sizing geometry;
        a pasted Hugging Face link contributes the repo and revision, and the
        geometry must come from ``payload["model_spec"]`` because the fit engine
        cannot size a model whose shape it has not been told (and B1 does not
        fetch ``config.json``). Either way the spec is validated by
        `build_model_spec`, so a bad geometry is a clear ``ValueError`` here.
        """
        source = resolve_model_source(
            payload.get("model_source", ""), GPU_MODEL_CATALOG.keys()
        )
        repo, revision = repo_of_source(source)
        # Either way the repo's own config.json, when this controller's model
        # library holds it, joins the body (`hf_cached_config`): it is what
        # says which layers hold a KV cache and what the model carries.
        if source["kind"] == "catalog":
            return repo, revision, build_model_spec(
                with_cached_config(catalog_spec_body(source["id"]), repo, revision),
            )
        spec_body = payload.get("model_spec")
        if not isinstance(spec_body, dict):
            raise ValueError(
                "Paste a Hugging Face repo with its model geometry "
                "('model_spec'), or choose a catalog model."
            )
        return repo, revision, build_model_spec(
            with_cached_config(spec_body, repo, revision),
        )

    def _confirm_in_slice(self, transport: Any, name: str, role: str, unit: str) -> None:
        """Refuse the split unless this Ray container runs in its fenced slice."""
        confirm_unit_in_slice(self.runtime, transport, name, role, unit,
                              sleep=self._sleep, monotonic=self._monotonic)

    def _placement(
        self, plan: Dict[str, Any], by_id: Dict[str, Dict[str, Any]],
        payload: Any = None, spec: Any = None,
    ) -> Dict[str, Any]:
        """Turn a fit verdict into the nodes to serve on and the parallel degrees.

        **Every refusal is the engine's own sentence, verbatim.** The verdicts
        a placement is made from are ``distributed`` and ``replicated``; every
        other one - ``wont_fit`` (the deficit and the ways out),
        ``single_refused`` (D6: it fits one machine, so serve it there as AI
        Chat) and ``unsupported_intent`` (VD-129: replication without this
        controller) - carries the summary `cluster_gpu_sizing` wrote for it,
        and that summary is raised as it stands, so the `/cluster/fit` preview
        and this deploy tell the operator the same thing. A second refusal
        written here is how a preview and a deploy come to disagree, and a
        branch for a verdict the engine never produces is how one silently
        serves what it should refuse: VD-127 makes a one-machine model AI
        Chat's, the engine answers ``single_refused`` wherever it once
        answered ``single``, and the single branch that used to stand here was
        unreachable.

        What is left is the mapping this method owns: ``distributed`` serves
        on the engine's chosen subset as a pipeline, one stage per machine,
        unless the owner EXPLICITLY chose a tensor-parallel split
        (``split_mode``, VD-167; refused when the heads do not divide);
        ``replicated`` serves one whole copy on every selected node.
        """
        if plan["verdict"] not in {VERDICT_DISTRIBUTED, VERDICT_REPLICATED}:
            raise ValueError(plan["summary"])
        placement = plan["placement"]
        chosen = [by_id[view["node_id"]] for view in placement["nodes"]]
        count = int(placement["node_count"])
        if plan["verdict"] == VERDICT_REPLICATED:
            return {
                "mode": VERDICT_REPLICATED,
                "parallelism": "",
                "tensor_parallel_size": gpu_fit_node(chosen[0])["device_count"],
                "pipeline_parallel_size": 1,
                "nodes": chosen,
            }
        # The degree across machines comes from the owner's choice only,
        # never off the plan's word: a pipeline unless tensor was asked for.
        split_mode = parse_split_mode((payload or {}).get(SPLIT_MODE_FIELD))
        devices = split_devices(
            [(node.get("name") or node["id"], gpu_fit_node(node)["device_count"]) for node in chosen])
        refusal = tensor_refusal(spec, count, devices) if split_mode == SPLIT_TENSOR else None
        if refusal is not None:
            raise ValueError(refusal)
        tensor, pipeline = degrees(split_mode, count, devices)
        return {
            "mode": VERDICT_DISTRIBUTED, "parallelism": split_mode,
            "tensor_parallel_size": tensor, "pipeline_parallel_size": pipeline,
            "nodes": chosen,
        }

    def _wait_for_api(
        self, endpoint, payload, report, transport, node, unit, probe_url=None,
    ) -> None:
        """Wait for the lead's API while its server unit is alive (VD-127).

        `gpu_pool_startup.ServerStartupWaiter` owns the rule: probe, then read
        the unit through ``transport`` - the lead's own, so the bridge when
        this controller leads - and fail at once if it has stopped; wait up to
        the documented compile cap while it lives; say which of the two ended
        the wait. Kept as a method so a test about the rollback's ORDER can
        substitute the wait without driving a clock.
        """
        self._startup.wait(
            transport, node, unit, endpoint, payload, report, probe_url=probe_url,
        )

    def remove(self, payload: Dict[str, Any]) -> Dict[str, Any]:
        """Stop a GPU deployment's units and drop its record.

        The order, the refusals and the departed-machine exemption live in
        `gpu_pool_removal.remove_gpu_deployment` (extracted for review R1).
        """
        return remove_gpu_deployment(self, payload)

    def rotate_cluster_serving_key(
        self, payload: Dict[str, Any], progress=None
    ) -> Dict[str, Any]:
        """Rotate the internal cluster key in place (design F3e).

        A thin wiring of `gpu_pool_replicas.rotate_cluster_serving_key` with this
        operation's collaborators - the store, broker, runtime (whose `rekey_gate`
        re-renders a worker gate), balancer and node transports - so the ordered
        overlap rotation lives in the replicated-serving module while these stay
        here; the serving lock serialises the balancer restart with the reconcile.
        """
        return rotate_cluster_serving_key(
            payload, store=self.store, broker=self.broker, runtime=self.runtime,
            balancer=self.balancer, joined_node=self.joined_node,
            transport=self._transport, serving_lock=self._serving_lock,
            progress=progress,
        )
