"""The one gate: what is serving AI Chat's GPU model, and may it be exposed.

Three call sites asked this question separately and could answer it three ways
(LESSONS pattern 6): :meth:`vaelor.executor_gpu_deploy.ExecutorGpuDeployMixin.ensure_gpu_chat_served`
(the 30 s GPU failure-watch), :meth:`~vaelor.executor_gpu_deploy.ExecutorGpuDeployMixin.apply_llm_server`
(the LAN auth proxy) and ``api_llm_server_routes._resolve_gpu_chat`` (the
surface). Each re-spelled the VD-085 independence test, each spelled its own
refusal reason - two of them differently for the same condition - and none of
them could see Mode B at all. This module is now their one home; the three are
adapters over :func:`resolve_gpu_serving_target`.

**What "the GPU serving target" means, and the two kinds it has (VD-125).**
Vaelor serves one GPU engine at a time:

* ``managed-local`` - **Mode A**, the default. llama.cpp (the ROCmFPX container)
  serves the AI-Chat model on a loopback port of this controller, and the
  ``ai-chat`` lease is the credential Vaelor minted for it. This is the only kind
  the failure-watch relaunches, because it is the only kind it supervises.
* ``cluster`` - **Mode B**, GPU clustering. vLLM serves the model across two or
  more GPU nodes and the lead binds its OpenAI API on this controller's loopback;
  the ``ai-chat`` lease is the credential the cluster deploy minted. The
  failure-watch must NOT relaunch llama.cpp under it (that would put two engines
  on one GPU), but the LLM Server proxy still fronts it - the LAN exposure is a
  property of the port, not of the engine behind it. Since VD-210 the owner may
  move AI Chat to another connection while the cluster serves; the target is
  then still the cluster, read through its ``cluster-inference`` lease, because
  the LLM Server fronts the cluster whatever AI Chat chose.
* ``none`` - anything else, with the reason. A hosted ai-chat, a single-model
  box, or a box with nothing deployed.

**A user-entered endpoint is never a target.** ``cluster`` is granted only to the
credential id the mode file itself recorded, on a loopback host, inside the GPU
serving port band - never to any other ``openai-compatible`` credential. Without
all three, somebody's LM Studio on the LAN could be proxied to the world under
Vaelor's own API key, which is the opposite of what the LLM Server promises.

**Loopback is checked as an address, not as a parsable port.**
The executor's own ``_loopback_port`` only asserted that a port number parsed,
so ``http://192.0.2.11:8000/v1`` read as a loopback model. That is the LAN, and
fronting it with the auth proxy (which reaches its upstream on ``127.0.0.1``)
would have proxied to nothing. :func:`loopback_port` asks
:mod:`ipaddress` whether the host really is loopback, and is now the one
reading - the NPU Assistant's boot reconcile reads its pinned port through it
too, so the two encodings could not drift apart again (LESSONS pattern 6).
"""

from __future__ import annotations

import ipaddress
import json
import logging
import time
import urllib.parse
from dataclasses import dataclass
from typing import Any, Callable, Mapping, Optional, Tuple

from .credential_broker import NO_ACTIVE_CREDENTIAL, CredentialError
from .gpu_pool_startup import (
    API_POLL_SECONDS,
    STARTUP_CHECKPOINT_PERCENT,
    startup_cap_seconds,
)
from .managed_local_credentials import PREFIX, _endpoint_of


LOGGER = logging.getLogger(__name__)


#: The purposes this gate reads. ``ai-chat`` is the lease the GPU tier holds;
#: ``deployment-agent`` is the Assistant's, and is read only to tell an
#: independent GPU model from a single-model box that shares one endpoint. The
#: Assistant's lease is never written here (VD-042/VD-085).
AI_CHAT_PURPOSE = "ai-chat"
ASSISTANT_PURPOSE = "deployment-agent"
#: The purpose the cluster deploy activates its minted credential under, and
#: the one the mode switch reads that credential's key back through when it
#: has to restart the replica balancer (VD-129). Owned beside the other two
#: because the switch and the deploy are the only writers and readers.
CLUSTER_INFERENCE_PURPOSE = "cluster-inference"


#: The two GPU serving modes, as the mode file spells them.
#: :mod:`vaelor.gpu_cluster_mode` persists and switches them; they are named here
#: because this is the module that READS them to decide a target's kind, and one
#: home for the words keeps the switch and the gate from drifting.
MODE_SINGLE = "A"
MODE_CLUSTER = "B"

#: What AI Chat's picker shows beside this machine's own GPU model, and its
#: activate route answers, while the box is in Mode B. VD-210: AI Chat's model
#: is the owner's choice while the cluster serves, so only the managed-local
#: connection is refused - its llama.cpp is stopped, and a managed-local lease
#: under Mode B is what the failure-watch would have read as "relaunch
#: llama.cpp" into the aperture vLLM holds. One sentence, owned here beside the
#: mode it describes, shown BEFORE the click and returned if one lands anyway.
AI_CHAT_HELD_BY_CLUSTER = (
    "GPU clustering is using this machine's GPU, so AI Chat cannot use this "
    "machine's own GPU model until the cluster is removed. AI Chat can use any "
    "other connection."
)

#: What the ``model.deploy`` job answers when a deploy that is NOT AI Chat's
#: would still put llama.cpp on the GPU while the mode file reads Mode B: an
#: Assistant-surface GGUF whose accelerator plan settled on vulkan or rocm (the
#: console's Assistant deploy, `vaelor-refresh-models`, the upgrade's refresh).
#: The sentence above is about AI Chat's connection and would be a lie here -
#: the connection is not what is refused, the GPU is. The deploy is refused
#: rather than quietly re-planned onto the CPU, because a plan the operator did
#: not ask for is a false record; the operator picks the CPU, or ends the cluster.
GPU_HELD_BY_CLUSTER = (
    "The GPU is serving the cluster model; deploy this model on the CPU or "
    "remove the cluster deployment first."
)


#: What is serving. ``none`` is not a failure - it is the honest answer on a Pi,
#: on a fresh appliance, or on a box whose AI Chat points at a hosted provider.
KIND_NONE = "none"
KIND_MANAGED_LOCAL = "managed-local"
KIND_CLUSTER = "cluster"


#: Why a target is unavailable. These are machine tokens the LLM Server surface
#: turns into human copy (``frontend/src/components/LlmServerPanel.tsx`` keys its
#: ``UNAVAILABLE_COPY`` table on exactly these), so they are a wire vocabulary
#: this module owns and every reader imports rather than re-spells. Before this
#: module the executor said ``no-ai-chat``/``shared-endpoint`` for the two
#: conditions the route called ``no-gpu-model``/``single-model`` - the same
#: question, two vocabularies, and the surface only understood one of them.
REASON_BROKER_UNAVAILABLE = "broker-unavailable"
REASON_NO_GPU_MODEL = "no-gpu-model"
REASON_NOT_MANAGED_LOCAL = "not-managed-local"
REASON_NO_LOOPBACK_PORT = "no-loopback-port"
REASON_SINGLE_MODEL = "single-model"

#: Why a reclaimable read answered nothing. Not a target reason - it is carried
#: as a NODE fact onto the capacity ledger, so the fit verdict can say it was
#: sized without the AI Chat model's memory instead of quietly refusing a model
#: the mode switch would have made room for.
REASON_LIVENESS_UNREADABLE = "gpu-liveness-unreadable"


#: The port band a cluster deploy may serve on, and the band's one home. It stops
#: at 8079 because the single-node GPU tier picks the lowest free port from 8080
#: to 8099 (:mod:`vaelor.executor_network`) on the very machine VD-125 made a
#: cluster serving node: overlapping bands would let a cluster deploy claim the
#: port Mode A's chat is about to take. :meth:`vaelor.gpu_pool_runtime.GpuPoolRuntime._port`
#: validates against it, `gpu_pool_operations.deploy` routes the requested port
#: through that validator before any use, and this module refuses to call a
#: credential outside it a cluster target.
GPU_SERVING_PORT_MIN = 8000
GPU_SERVING_PORT_MAX = 8079

#: The unprivileged port range a managed-local model may be served on. Wider than
#: the cluster band because Mode A allocates from 8080-8099 and an older
#: appliance's stored credential may name anything the allocator once chose.
MODEL_PORT_MIN = 1024
MODEL_PORT_MAX = 65535


#: Where a served GPU model's OpenAI API may bind: loopback, and nothing else
#: (VD-156, amended by ACC-162). The vLLM server carries no key, so an API on
#: any other address is a model anyone on that network can call. Every lead -
#: this controller or a worker, split or replica - binds here; a worker's is
#: reached from the controller only through its keyed gate (`gpu_pool_gate`),
#: and the controller's through the balancer or the LLM Server proxy. A
#: worker-led split used to bind ``0.0.0.0`` with no key. Ray's and RCCL's
#: own sockets are a separate plane on the chosen cluster link and are not
#: this value. One literal rather than a free-form host: whatever reaches here
#: goes into a unit file's ``ExecStart``.
SERVER_LOOPBACK_HOST = "127.0.0.1"
API_BIND_HOSTS = frozenset({SERVER_LOOPBACK_HOST})


@dataclass(frozen=True)
class ServingTarget:
    """What the GPU serving surfaces should act on, and whether there is one.

    ``available`` is redundant with ``kind != "none"`` and is carried anyway,
    because the LLM Server surface publishes it as its own field and a reader
    should never have to know the kind vocabulary to answer "is there anything
    to expose". ``label`` is the ``ai-chat`` credential's label, which is the one
    durable identifier a generic Mode A lease keeps across a reboot
    (:func:`vaelor.managed_local_credentials.managed_model_by_label` reverses
    it); it rides along so the failure-watch resolves the lease once
    rather than reading the broker a second time for it.
    """

    kind: str = KIND_NONE
    credential_id: str = ""
    port: int = 0
    model: str = ""
    label: str = ""
    reason: str = REASON_NO_GPU_MODEL
    available: bool = False


def is_loopback_host(host: Any) -> bool:
    """Whether ``host`` really is an address on this machine's loopback.

    ``localhost`` is accepted by name because that is what it resolves to and a
    credential may legitimately spell it; everything else must parse as a
    loopback IP address. A name this cannot resolve is refused rather than
    looked up: a DNS round trip inside a 30 s reconcile is a stall, and a
    managed-local credential is written by Vaelor as a literal address.
    """
    text = str(host or "").strip().lower()
    if not text:
        return False
    if text == "localhost":
        return True
    try:
        return ipaddress.ip_address(text).is_loopback
    except ValueError:
        return False


def loopback_port(base_url: str) -> Optional[int]:
    """The loopback port a credential's ``base_url`` names, or ``None``.

    ``None`` for a URL that names no port, a port outside the unprivileged range,
    or - the check that was missing - a host that is not actually loopback. Every
    caller reads ``None`` as "nothing local to act on", which is the fail-safe
    direction: a LAN endpoint is left alone rather than relaunched, proxied or
    counted as this appliance's own model.
    """
    parts = urllib.parse.urlsplit(str(base_url or "").strip())
    try:
        port = parts.port
    except ValueError:
        return None
    if port is None or not MODEL_PORT_MIN <= port <= MODEL_PORT_MAX:
        return None
    return port if is_loopback_host(parts.hostname) else None


def serving_port(value: Any) -> int:
    """A cluster serving port, validated against :data:`GPU_SERVING_PORT_MIN`/``MAX``.

    Raises ``ValueError`` with the band in it, so a typed port is refused before
    a deploy writes a record or starts a unit rather than after.
    """
    try:
        port = int(value)
    except (TypeError, ValueError):
        raise ValueError("The serving port must be a number.")
    if not GPU_SERVING_PORT_MIN <= port <= GPU_SERVING_PORT_MAX:
        raise ValueError(
            "GPU serving ports must be between {} and {}.".format(
                GPU_SERVING_PORT_MIN, GPU_SERVING_PORT_MAX
            )
        )
    return port


def api_bind_host(value: Any) -> str:
    """One of :data:`API_BIND_HOSTS`, or a refusal. Never a free-form host."""
    host = str(value or SERVER_LOOPBACK_HOST).strip()
    if host not in API_BIND_HOSTS:
        raise ValueError(
            "The model API binds loopback only; a machine's model is reached "
            "from the controller through its keyed gate."
        )
    return host


def _shares_the_assistants_endpoint(broker: Any, chat: Mapping[str, Any]) -> bool:
    """Whether ``ai-chat`` and the Assistant are the same managed endpoint.

    The VD-085 independence key. On a single-model box (a Pi) one managed model
    holds both leases, and its server belongs to the NPU/llama.cpp reconcile -
    relaunching a GPU server on that port would evict the Assistant, and exposing
    it on the LAN would expose the Assistant. No ``deployment-agent`` lease at
    all means ai-chat is the only managed tier, so it is not shared.
    """
    try:
        agent = broker.resolve_active(ASSISTANT_PURPOSE)
    except CredentialError:
        return False
    return bool(_endpoint_of(agent)) and _endpoint_of(agent) == _endpoint_of(chat)


def no_active_credential(error: BaseException) -> bool:
    """Whether a broker error is its "nothing is assigned to that purpose" answer.

    The broker's refusal crosses its socket as a message, so the message is the
    type: this is the one place that compares it (VD-136, review SC1). Any
    other error - the broker down, a timeout, a malformed answer - is a lease
    that could not be read.
    """
    return isinstance(error, CredentialError) and str(error) == NO_ACTIVE_CREDENTIAL


def gpu_cluster_mode_active(mode_state: Any = None) -> bool:
    """Whether the mode file says Mode B - the one reading of its word.

    ``mode_state`` is a :class:`vaelor.gpu_cluster_mode.ClusterModeState` a
    caller already holds (the executor's cached store); ``None`` reads the file
    itself, which is what the control plane's chat runtime has. The file is the
    fact (VD-127): it is written before the switch stops or moves anything and
    reset only after the leave restores Mode A, so a reader that trusts it
    cannot start an engine inside the window a cluster deploy owns.

    Four readers act on it directly, and only four; a fifth only describes
    it (below). The GPU failure-watch
    returns without relaunching llama.cpp while it is true, whatever the
    ``ai-chat`` lease says (a lease that drifted is the mode reconcile's to
    re-point, not the watch's to serve); `chat_inference.activate` refuses to
    move the lease onto this machine's own GPU model (VD-210: any other
    connection may be picked), with :data:`AI_CHAT_HELD_BY_CLUSTER`; and the
    executor's ``model.deploy`` job refuses, through one helper
    (`executor_gpu_deploy.ExecutorGpuDeployMixin._refuse_deploy_under_cluster`),
    an ``ai-chat`` deploy with that same sentence before it stops or starts
    any engine, and any other deploy whose accelerator plan would put
    llama.cpp on the GPU with :data:`GPU_HELD_BY_CLUSTER`; and
    `chat_inference.ChatInference.answer` skips the one-at-a-time local slot
    for the cluster's loopback balancer, because vLLM batches (ACC-105). The
    one reader that acts on nothing is the ``/inference/status`` route, which
    asks `ChatInference.cluster_mode_active` so System > Graphics never names
    llama.cpp on a clustered GPU (ACC-114). The store is imported inside the
    call because it imports this module.
    """
    if mode_state is None:
        from .gpu_cluster_mode import ClusterModeStore

        mode_state = ClusterModeStore().read()
    return (
        str(getattr(mode_state, "mode", MODE_SINGLE) or MODE_SINGLE)
        == MODE_CLUSTER
    )


def resolve_gpu_serving_target(broker: Any, mode_state: Any = None) -> ServingTarget:
    """The one answer to "what is serving AI Chat's GPU model right now".

    ``mode_state`` is a :class:`vaelor.gpu_cluster_mode.ClusterModeState` (or
    anything carrying ``mode`` and ``cluster_credential_id``); ``None`` reads as
    Mode A, which is what every caller that has no mode file sees.

    The two kinds are mutually exclusive by construction - a managed-local
    credential id carries :data:`~vaelor.managed_local_credentials.PREFIX` and a
    broker-minted cluster credential does not - so the order below is a reading
    order, not a precedence rule.
    """
    if broker is None:
        return ServingTarget(reason=REASON_BROKER_UNAVAILABLE)
    try:
        chat = broker.resolve_active(AI_CHAT_PURPOSE)
    except CredentialError as error:
        # "No credential is active" is a real answer - nothing is assigned to
        # AI Chat (a manual unload with no Assistant connection clears it on
        # purpose). Any other broker error means the lease could not be read.
        if no_active_credential(error):
            return ServingTarget(reason=REASON_NO_GPU_MODEL)
        return ServingTarget(reason=REASON_BROKER_UNAVAILABLE)
    credential_id = str(chat.get("credential_id", ""))
    model = str(chat.get("model") or "")
    label = str(chat.get("label") or "")
    base_url = str(chat.get("base_url") or "")
    facts = {"credential_id": credential_id, "model": model, "label": label}

    if credential_id.startswith(PREFIX):
        port = loopback_port(base_url)
        if port is None:
            return ServingTarget(reason=REASON_NO_LOOPBACK_PORT, **facts)
        if _shares_the_assistants_endpoint(broker, chat):
            return ServingTarget(
                port=port, reason=REASON_SINGLE_MODEL, **facts
            )
        return ServingTarget(
            kind=KIND_MANAGED_LOCAL, port=port, reason="", available=True,
            **facts,
        )

    recorded = str(getattr(mode_state, "cluster_credential_id", "") or "")
    # `None` reads as Mode A here (see the docstring), so the predicate is
    # asked only of a state a caller actually holds.
    in_cluster_mode = mode_state is not None and gpu_cluster_mode_active(mode_state)
    if in_cluster_mode and recorded and credential_id == recorded:
        return _cluster_target(base_url, facts)
    if in_cluster_mode and recorded:
        # VD-210: AI Chat is on a connection the owner chose. A managed-local
        # lease (above) keeps the reading it always had: the switch refuses
        # that pick in Mode B, and an old park on the Assistant is VD-136's
        # "manual" witness for a box unloaded before VD-210.
        return _cluster_beside_the_owners_choice(broker, mode_state, recorded, facts)
    return ServingTarget(reason=REASON_NOT_MANAGED_LOCAL, **facts)


def _cluster_target(base_url: str, facts: Mapping[str, str]) -> ServingTarget:
    """The recorded cluster credential as a target, if its address may be fronted."""
    port = loopback_port(base_url)
    if port is None or not (GPU_SERVING_PORT_MIN <= port <= GPU_SERVING_PORT_MAX):
        # A recorded cluster credential that is not on a loopback port in the
        # serving band is not something this appliance may front. Refusing is
        # the fail-safe: `reconcile` reads the same "no target" answer and
        # returns the box to Mode A rather than proxying an unknown endpoint.
        return ServingTarget(reason=REASON_NO_LOOPBACK_PORT, **facts)
    return ServingTarget(kind=KIND_CLUSTER, port=port, reason="", available=True, **facts)


def _cluster_beside_the_owners_choice(
    broker: Any, mode_state: Any, recorded: str, facts: Mapping[str, str],
) -> ServingTarget:
    """Mode B with AI Chat on a connection the owner chose: still the cluster (VD-210).

    The ``ai-chat`` lease no longer names the cluster, so it is read through
    the ``cluster-inference`` lease its deploy activated, and granted only if
    that is the very credential the mode file recorded - the S1 rule is the
    same as for the ``ai-chat`` path. A manual unload is not a target, which is
    what the lease's park said before VD-210 (`unload_cause`): the recorded
    cause now says it instead, since the owner's connection is never parked.
    """
    not_fronted = ServingTarget(reason=REASON_NOT_MANAGED_LOCAL, **facts)
    if str(getattr(mode_state, "unload_cause", "") or "") == UNLOAD_CAUSE_MANUAL:
        return not_fronted
    try:
        lease = broker.resolve_active(CLUSTER_INFERENCE_PURPOSE)
    except CredentialError as error:
        return not_fronted if no_active_credential(error) else ServingTarget(
            reason=REASON_BROKER_UNAVAILABLE)
    if str(lease.get("credential_id", "")) != recorded:
        return not_fronted
    return _cluster_target(str(lease.get("base_url") or ""), {
        "credential_id": recorded, "model": str(lease.get("model") or ""),
        "label": str(lease.get("label") or ""),
    })


#: Who unloaded a paused Mode B cluster model, because that decides whether a
#: request wakes it. An idle (scale-to-zero) unload holds the ``ai-chat`` lease
#: on the cluster credential, so the serving target still resolves to the
#: cluster and the next request loads the model again; a manual unload parks
#: ``ai-chat`` elsewhere, and only a Load from Cluster > Deployments brings it
#: back. :func:`unload_cause` is the ONE rule: the cluster-agent workshop reads
#: it through the fleet summary (:func:`deployment_unload_cause`), and the LLM
#: Server's paused sentence (``api_llm_server_routes.llm_server_runtime``) asks
#: it of the target that surface already resolved.
UNLOAD_CAUSE_IDLE = "unloaded-idle"
UNLOAD_CAUSE_MANUAL = "unloaded-manual"
#: The lease could not be read, so who unloaded the model is not known.
UNLOAD_CAUSE_UNKNOWN = ""


def unload_cause(target: "ServingTarget") -> str:
    """Idle when the resolved target is still the cluster; unknown if unreadable.

    Asked only about the Mode B deployment while its record is ``unloaded``.
    A target the gate built without being able to read the ``ai-chat`` lease -
    no broker, or a broker error other than "no credential is active"
    (:data:`REASON_BROKER_UNAVAILABLE`) - gives :data:`UNLOAD_CAUSE_UNKNOWN`:
    never guessed awake, never guessed manual. Every other target is manual:
    an available model that is not the cluster (the lease parked elsewhere),
    and NO active lease at all (:data:`REASON_NO_GPU_MODEL`), which is the
    manual unload's own state when there is no Assistant connection to park
    AI Chat on (review SC1).
    """
    if target.reason == REASON_BROKER_UNAVAILABLE:
        return UNLOAD_CAUSE_UNKNOWN
    if target.available and target.kind == KIND_CLUSTER:
        return UNLOAD_CAUSE_IDLE
    return UNLOAD_CAUSE_MANUAL


def deployment_unload_cause(broker: Any, mode_state: Any, deployment_name: str) -> str:
    """The unload cause of one ``unloaded`` vLLM deployment, or ``""`` if unknown.

    Only the Mode B deployment holds the ``ai-chat`` lease that tells an idle
    unload from a manual one, so any other unloaded record is ``manual``: no
    lease of the appliance's points at it and nothing is known to wake it. A
    resolve that raises gives :data:`UNLOAD_CAUSE_UNKNOWN`; every other case is
    :func:`unload_cause`'s, including its unreadable-lease answer.
    """
    if not (
        mode_state is not None
        and gpu_cluster_mode_active(mode_state)
        and str(getattr(mode_state, "deployment_name", "") or "") == str(deployment_name)
    ):
        return UNLOAD_CAUSE_MANUAL
    try:
        target = resolve_gpu_serving_target(broker, mode_state)
    except Exception as error:  # noqa: BLE001 - an unreadable lease is "unknown"
        LOGGER.warning("Could not read the AI Chat lease for %s: %s", deployment_name, error)
        return UNLOAD_CAUSE_UNKNOWN
    return unload_cause(target)


def gpu_chat_reclaimable(
    broker: Any, mode_state: Any, gpu_status: Optional[Mapping[str, Any]]
) -> bool:
    """Whether stopping Mode A would give this machine's GPU memory back (D1).

    True only when the target is ``managed-local`` AND the llama.cpp server is
    actually running: a Mode A model that is not resident is holding nothing to
    reclaim, and a cluster or hosted target is not ours to stop. The fit engine
    adds that node's used bytes to its free memory so the ``/cluster/fit``
    preview and the deploy size against the memory the switch WILL free rather
    than the memory that happens to be free while the 27B is loaded.
    """
    if not bool((gpu_status or {}).get("running")):
        return False
    target = resolve_gpu_serving_target(broker, mode_state)
    return target.kind == KIND_MANAGED_LOCAL


def reclaimable_or_unreadable(read: Callable[[], bool]) -> Tuple[bool, str]:
    """``(reclaimable, reason)`` from one liveness read, non-fatally.

    **"No" and "could not ask" are different answers with the same
    arithmetic.** A bridge timeout or a permission error used to be swallowed
    as a plain ``False``, and the fit then sized against the memory that is
    free WHILE the 27B is loaded - which on the product's own hardware is the
    difference between "fits" and "will not fit", reported as a finding.
    :data:`REASON_LIVENESS_UNREADABLE` travels with the zero so the verdict can
    say it was sized without that memory rather than silently refusing the
    deploy the switch exists to make possible.

    ``read`` is the one call that answers :func:`gpu_chat_reclaimable` with
    whatever collaborators the caller holds - the control plane's fresh
    clients, or the executor's switch - so the non-fatal shape and the reason
    are written once for both sides of the same question (VD-127).
    """
    try:
        return bool(read()), ""
    except Exception as error:  # noqa: BLE001 - a capacity read may never raise
        LOGGER.warning(
            "The AI Chat model's GPU state could not be read, so the GPU fit "
            "was sized without the memory stopping it would free: %s", error,
        )
        return False, REASON_LIVENESS_UNREADABLE


def controller_gpu_chat_reclaimable() -> Tuple[bool, str]:
    """The control plane's wiring of :func:`gpu_chat_reclaimable`, non-fatally.

    Both sides of the same question read one engine: the control plane's
    capacity ledger (which feeds ``POST /cluster/fit``) and
    `cluster_operations`' own controller GPU facts (which feed the deploy) call
    this, and the executor's `GpuPoolOperations` reads the same
    :func:`gpu_chat_reclaimable` through the switch it already holds, inside
    the same :func:`reclaimable_or_unreadable` shape - so a preview and a
    deploy cannot disagree about how much memory the switch would free
    (VD-B3b-1's rule applied to reclaimable memory).

    **The control plane can read the model's liveness.** The root hardware
    bridge admits the control plane (``vaelor``) and the workload executor
    (``vaelor-workloads``) alike (VD-143, `bridge_peers`), so ``gpu_status`` -
    a read verb - is reachable from both accounts; no second probe exists and
    none is invented here.

    Every dependency is imported inside the call: the mode store imports this
    module, so a top-level import would be a cycle, and a box with no bridge must
    not fail a capacity read.
    """
    def _read() -> bool:
        from .credential_broker import CredentialBrokerClient
        from .gpu_cluster_mode import ClusterModeStore
        from .hardware_bridge import HardwareBridgeClient

        return gpu_chat_reclaimable(
            CredentialBrokerClient(),
            ClusterModeStore().read(),
            HardwareBridgeClient().gpu_status(),
        )

    return reclaimable_or_unreadable(_read)


def switch_gpu_chat_reclaimable(broker: Any, mode_switch: Any) -> Tuple[bool, str]:
    """The executor's wiring of :func:`gpu_chat_reclaimable`, through its switch.

    Answers ``(reclaimable, reason)``. ``(False, "")`` without a switch: a
    control-plane instance of the GPU operations has no business reading the
    GPU, and sizing against memory that is free right now is the conservative
    direction and not a failed reading. With one, the same reading the
    ``/cluster/fit`` preview makes (:func:`controller_gpu_chat_reclaimable`),
    through the switch's own collaborators rather than fresh clients, in the
    one non-fatal shape - so an unreadable answer carries the same reason on
    both sides (VD-127).
    """
    if mode_switch is None:
        return False, ""
    return reclaimable_or_unreadable(lambda: gpu_chat_reclaimable(
        broker, mode_switch.state(), mode_switch.gpu_supervisor.status(),
    ))


#: The substring the broker's openai-compatible test returns when the endpoint
#: did not answer in time - a refused connection or a read timeout, as against
#: an HTTP status or a rejected key. The cluster credential test retries only
#: on this: a cold box's FIRST inference through the just-started balancer can
#: exceed the broker's 45 s chat-completions budget, and that slow first token
#: is a warming server, not a dead one. Every other ok=False (a rejected key, a
#: model that is not loaded) is a real misconfiguration and fails at once.
CLUSTER_TEST_UNREACHABLE = "could not be reached"

#: The reason carried when the broker returns ok=False with no message of its own.
CLUSTER_TEST_FAILED = "The vLLM model API test failed."

#: The cold-start progress line the cluster credential test emits while it waits
#: for the just-started balancer to answer, kept honest about why a first start
#: is slow the way the replica wait's own progress line is (VD-129).
CLUSTER_WARMUP_PROGRESS = (
    "Waiting for the cluster endpoint to serve its first token "
    "(a first start compiles kernels)"
)


def register_cluster_credential(
    broker: Any, name: str, endpoint: str, model: str, api_key: str = "",
    payload: Optional[Mapping[str, Any]] = None,
    report: Optional[Callable[[int, str], None]] = None,
    *,
    sleep: Callable[[float], None] = time.sleep,
    monotonic: Callable[[], float] = time.monotonic,
) -> str:
    """Mint the cluster's ``openai-compatible`` credential, model included.

    The one credential this gate will ever call a ``cluster`` target: minted
    by the GPU deploy once its endpoint answers, tested by the broker itself,
    and activated under :data:`CLUSTER_INFERENCE_PURPOSE`. ``ai-chat`` is NOT
    claimed here - the mode switch's `repoint` moves it, because moving it is
    half of a switch that has a matching `leave`; and ``deployment-agent`` is
    never claimed at all (the deploy refuses ``use_for_assistant``).

    The model is the weights repo the cluster is actually serving, written
    because every surface that reads this credential shows it: the LLM Server
    panel names what is behind the LAN gate, and AI Chat's connection details
    name what is answering. An empty ``model`` left both showing a cluster
    with no model on it, which reads as broken rather than as unnamed - and
    vLLM's own ``/v1/models`` answers under this id.

    ``api_key`` is the replicated deployment's one key (VD-129): recorded on
    the credential so the broker's test goes through the balancer with it, AI
    Chat sends it, and the switch can read it back to restart the balancer.
    Empty for a distributed deployment, whose loopback head has none.

    **The test is cold-start-tolerant (defect A).** On a cold box the FIRST
    inference through the just-started balancer compiles kernels and can
    exceed the broker's one-shot 45 s chat budget, so an unchanged test timed
    out seconds before the endpoint would have answered and the deploy
    reverted Mode B to A. ``payload`` and ``report`` are threaded in from the
    deploy so this reuses the SAME cold-start budget the replica wait uses:
    :func:`~vaelor.gpu_pool_startup.startup_cap_seconds` of the operator's
    ``startup_timeout_seconds`` bounds :meth:`broker.test` in a bounded retry
    at :data:`~vaelor.gpu_pool_startup.API_POLL_SECONDS` cadence that loops
    ONLY on the reachability timeout (:data:`CLUSTER_TEST_UNREACHABLE`). Both
    'the balancer is not up yet' (its ``/v1/models`` fails fast) and the cold
    first inference (its ``/chat/completions`` times out) surface as that
    timeout, so the one retry covers both; any real misconfiguration fails
    fast. When the budget is spent the last failure is raised, so the deploy's
    own rollback still runs for a genuinely dead endpoint.
    ``sleep``/``monotonic`` are injected by the tests.

    The credential is deleted again if the test or the activation fails, so a
    broker profile is never orphaned by a mid-registration error - the deploy's
    own rollback only sees a credential id once this returns.
    """
    report = report or (lambda _percent, _message: None)
    cap = startup_cap_seconds((payload or {}).get("startup_timeout_seconds"))
    profile = json.dumps(
        {"base_url": endpoint, "model": str(model or ""), "api_key": str(api_key or "")},
        separators=(",", ":"),
    )
    credential = broker.put("openai-compatible", f"GPU LLM · {name}", profile)
    credential_id = credential["id"]
    try:
        deadline = monotonic() + cap
        test = _test_cluster_credential(
            broker, credential_id, deadline, report, sleep, monotonic
        )
        if not test.get("ok"):
            raise RuntimeError(str(test.get("message", CLUSTER_TEST_FAILED)))
        broker.activate(credential_id, CLUSTER_INFERENCE_PURPOSE)
    except Exception:
        try:
            broker.delete(credential_id)
        except Exception:
            pass
        raise
    return credential_id


def _test_cluster_credential(
    broker: Any,
    credential_id: str,
    deadline: float,
    report: Callable[[int, str], None],
    sleep: Callable[[float], None],
    monotonic: Callable[[], float],
) -> Mapping[str, Any]:
    """``broker.test``, retried only while it fails for reachability, in budget.

    The one failure this loops on is the slow FIRST inference: a cold
    balancer's first ``/chat/completions`` can exceed the broker's own 45 s
    budget and return :data:`CLUSTER_TEST_UNREACHABLE`, which is a warming
    server rather than a dead one. Every other ok=False - a rejected key, a
    model that is not loaded - is a real misconfiguration and is returned at
    once, so the deploy rolls back without a wait. When the budget is spent
    the last failure is returned, so a genuinely dead endpoint still rolls
    back.
    """
    reported = False
    while True:
        test = broker.test(credential_id)
        if test.get("ok"):
            return test
        # Retrying keys on the client-side reachability timeout, which assumes
        # the balancer's read timeout stays LONGER than the broker's 45 s chat
        # client timeout (nginx default 60 s today): the slow first inference
        # surfaces as a client timeout, not a gateway 5xx. If a warmup path ever
        # returned the cold first inference as an HTTP 5xx, it would fail fast
        # here and would need handling.
        if CLUSTER_TEST_UNREACHABLE not in str(test.get("message", "")):
            return test
        if monotonic() >= deadline:
            return test
        if not reported:
            reported = True
            report(STARTUP_CHECKPOINT_PERCENT, CLUSTER_WARMUP_PROGRESS)
        sleep(API_POLL_SECONDS)
