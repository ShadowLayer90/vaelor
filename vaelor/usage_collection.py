"""Who used what, and how much: the control plane's usage collection, wired once.

Three sources answer three different questions, and each has one owner:

* **How much did the model do?** The serving engine's own counters
  (:mod:`vaelor.model_usage`), fed by the serving pollers' ``usage_sink`` - every
  door is counted there, the LLM Server, AI Chat, agents and the gateway alike.
* **Which LLM Server key was used, how often, and when?** The gate's access
  log (:mod:`vaelor.llm_gate_usage`).
* **Which inference-gateway key was used?** The gateway's own request log and
  token store, as before (:mod:`vaelor.inference_metrics`,
  :mod:`vaelor.agent_api`).

This module holds the first two for the control plane
(:class:`UsageCollection`: the ledger, the gate store, the gate drain and the two
sinks that name the deployment a reading belongs to) and composes the
"External API access" summary from the doors (:func:`external_usage`), so the
runtime wires them with a few lines and the screens read one shape.

**What is counted, and what is not** (the card says so too): this machine's
llama.cpp model when it serves on its own, and a cluster deployment this
controller leads. Not a model shared with the Assistant on a single-model box,
not a cluster led by a worker, and not the NPU Assistant - no poller reads
those engines' counters.
"""

from __future__ import annotations

import logging
import threading
import time
from typing import Any, Callable, Dict, List, Mapping, Optional

from .llm_gate_usage import (
    USAGE_COUNTING,
    USAGE_UNREADABLE,
    GateUsageCollector,
    GateUsageStore,
    cached_proxy_status,
    gate_logging_state,
)
from .cluster_placement import CONTROLLER_PLACEMENT_ID, CONTROLLER_PLACEMENT_NAME
from .gpu_pool_units import record_mode
from .model_usage import (
    CLUSTER_PREFIX,
    ENGINE_LLAMACPP,
    ENGINE_VLLM,
    LOCAL_PREFIX,
    ModelUsageLedger,
    counters_from_llamacpp,
    counters_from_vllm,
)
from .model_usage_replica import PLACEMENT_SINGLE
from .usage_rollup import BUCKET_SECONDS, SOURCE_AI_CHAT, SOURCE_GATEWAY

LOGGER = logging.getLogger(__name__)

#: The window Settings > External API access reports.
EXTERNAL_WINDOW_SECONDS = 24 * 3600


def _read_mode() -> Any:
    from .gpu_cluster_mode import ClusterModeStore

    return ClusterModeStore().read()


def local_identity(target: Any) -> Optional[Dict[str, str]]:
    """The deployment this machine's llama.cpp model is, or None when it is not serving.

    ``target`` is the :class:`~vaelor.gpu_serving_target.ServingTarget` the
    llama.cpp poller's endpoint discovery already resolved this tick (S6: no
    second lease of the AI Chat credential). Keyed by the model it serves,
    which a relaunch or a redeploy of the same model keeps; the managed-local
    credential's label carries the model's file stem when the lease's own
    ``model`` field is empty (a generic fork lease).
    """
    from .gpu_serving_target import KIND_MANAGED_LOCAL
    from .managed_local_credentials import MANAGED_LOCAL_CREDENTIAL_LABEL

    if target is None or not getattr(target, "available", False):
        return None
    if getattr(target, "kind", "") != KIND_MANAGED_LOCAL:
        return None
    prefix = MANAGED_LOCAL_CREDENTIAL_LABEL.split("{}", 1)[0]
    label = str(getattr(target, "label", "") or "")
    stem = label[len(prefix):] if label.startswith(prefix) else label
    name = str(getattr(target, "model", "") or stem or "").strip()
    if not name:
        return None
    return {
        "identity": LOCAL_PREFIX + name, "name": name, "model": name,
        "engine": ENGINE_LLAMACPP, "placement": PLACEMENT_SINGLE,
    }


def cluster_identity(cluster_store: Any, state: Any) -> Optional[Dict[str, str]]:
    """The cluster deployment Mode B serves, named as the owner deployed it."""
    from .gpu_cluster_mode import MODE_CLUSTER

    name = str(getattr(state, "deployment_name", "") or "")
    if str(getattr(state, "mode", "")) != MODE_CLUSTER or not name:
        return None
    record = cluster_store.get_pooled_deployment(name) if cluster_store else None
    return {
        "identity": CLUSTER_PREFIX + name, "name": name,
        "model": str((record or {}).get("model_id") or ""), "engine": ENGINE_VLLM,
        # "replicated" (one full copy per machine) or "distributed" (one model
        # spanning machines, read on its lead): the record's own word.
        "placement": record_mode(record or {}),
    }


class ReplicaSink:
    """What a serving poller is handed for one engine: a read, or a missed read.

    Called with ``(replica, counters, performance)`` for a replica that was
    read; ``miss(replica, seconds)`` for a tick in which it was a target and
    could not be read. Both reach the ledger, so its per-replica buckets know
    how much of each interval was really observed (VD-147).
    """

    def __init__(self, observe: Callable[..., None], miss: Callable[..., None],
                 transition: Optional[Callable[..., None]] = None,
                 last: Optional[Callable[[], Dict[str, str]]] = None,
                 flush: Optional[Callable[[], None]] = None):
        self._observe = observe
        self._miss = miss
        self._transition = transition
        self._last = last
        self._flush = flush

    def __call__(self, replica: str, counters: Mapping[str, float],
                 performance: Optional[Mapping[str, Any]] = None) -> None:
        self._observe(replica, counters, performance)

    def transition(self, replica: str, kind: str) -> None:
        """The poller's target set gained or lost this replica (pass-4 review B1)."""
        if self._transition is not None:
            self._transition(replica, kind)

    def last_transitions(self) -> Dict[str, str]:
        """Each replica's last recorded start or stop, for a poller's first tick (pass-5)."""
        return self._last() if self._last is not None else {}

    def flush_transitions(self) -> None:
        """Write any start or stop kept from a locked ledger (pass-6: every tick)."""
        if self._flush is not None:
            self._flush()

    def miss(self, replica: str, seconds: Optional[float] = None) -> None:
        self._miss(replica, seconds)


class UsageCollection:
    """The ledger, the gate store and the gate drain, for the control plane.

    ``cluster_store`` is a zero-argument callable returning the cluster store,
    because the runtime builds that after this; ``mode_reader`` reads the GPU
    serving mode record (the shipping store's file unless a test injects one);
    ``bridge`` returns the root hardware bridge client, whose proxy status says
    whether the LLM Server gate is recording use; ``llm_server_enabled`` reads
    the LLM Server flag. A sink's failure is caught by the poller that calls
    it: usage must never stop the Performance card's scrape.
    """

    def __init__(
        self, broker: Any, cluster_store: Callable[[], Any], *,
        ledger: Optional[ModelUsageLedger] = None,
        gate_store: Optional[GateUsageStore] = None,
        gate_directory: Optional[str] = None,
        mode_reader: Optional[Callable[[], Any]] = None,
        bridge: Optional[Callable[[], Any]] = None,
        llm_server_enabled: Optional[Callable[[], bool]] = None,
    ):
        self._broker = broker
        self._cluster_store = cluster_store
        self._mode_reader = mode_reader or _read_mode
        self._bridge = bridge
        self._llm_server_enabled = llm_server_enabled or _llm_server_enabled
        self.ledger = ledger or ModelUsageLedger()
        self.gate_store = gate_store or GateUsageStore()
        options = {"directory": gate_directory} if gate_directory else {}
        self.gate = GateUsageCollector(self.gate_store, broker=broker, **options)
        self._local_targets: Dict[str, Any] = {}
        self._lock = threading.Lock()
        #: The two sinks the serving pollers are wired with.
        self.local_sink = ReplicaSink(self._local_read, self._local_miss, self._local_transition,
                                      self._last_transitions, self._flush_transitions)
        self.cluster_sink = ReplicaSink(self._cluster_read, self._cluster_miss, self._cluster_transition,
                                        self._last_transitions, self._flush_transitions)
        #: Starts and stops not yet written (a locked ledger): kept in order and
        #: written with the next one, so a lock never loses a marker.
        self._pending_transitions: List[Dict[str, Any]] = []

    def note_local_target(self, endpoint: Optional[str], target: Any) -> None:
        """Remember the serving target the llama.cpp endpoint discovery resolved.

        The poller hands the sink the same endpoint URL it scraped, so the sink
        names the model from this resolution rather than leasing the AI Chat
        credential a second time every ten seconds.
        """
        with self._lock:
            self._local_targets = {endpoint: target} if endpoint else {}

    def _node_name(self, node_id: str) -> str:
        """What a machine is called now, captured with each row it writes.

        The controller by its placement name, a worker by its enrolled name. An
        unreadable store gives ``""`` - the row keeps whatever name an earlier
        tick recorded, and the id is never shown in its place.
        """
        if not node_id or node_id == CONTROLLER_PLACEMENT_ID:
            return CONTROLLER_PLACEMENT_NAME
        try:
            record = self._cluster_store().get_node(node_id)
        except Exception:  # noqa: BLE001 - absence-ok: the name is optional, the reading is not
            return ""
        return str((record or {}).get("name") or "")

    def _observe(self, identity: Optional[Mapping[str, str]], replica: str,
                 counters: Mapping[str, float],
                 performance: Optional[Mapping[str, Any]] = None) -> None:
        if not identity or not counters:
            return
        node_id = str((performance or {}).get("node_id") or "")
        self.ledger.observe(
            identity=identity["identity"], name=identity["name"],
            model=identity["model"], engine=identity["engine"],
            replica=replica, counters=counters,
            # No performance reading (an older caller): usage only, as before.
            performance=performance or None,
            node_id=node_id, node_name=self._node_name(node_id) if performance else "",
            placement=str(identity.get("placement") or ""),
        )

    def _missed(self, identity: Optional[Mapping[str, str]], replica: str,
                seconds: Optional[float], node_id: str) -> None:
        if not identity:
            return
        self.ledger.observe_miss(
            identity=identity["identity"], replica=replica, seconds=seconds,
            model=identity["model"], engine=identity["engine"], node_id=node_id,
            node_name=self._node_name(node_id), placement=str(identity.get("placement") or ""),
        )

    def _local_read(self, replica: str, counters: Mapping[str, float],
                    performance: Optional[Mapping[str, Any]] = None) -> None:
        """The llama.cpp poller's read (Mode A, this machine's model)."""
        with self._lock:
            target = self._local_targets.get(replica)
        self._observe(
            local_identity(target), replica, counters_from_llamacpp(counters), performance,
        )

    def _local_miss(self, replica: str, seconds: Optional[float] = None) -> None:
        with self._lock:
            target = self._local_targets.get(replica)
        self._missed(local_identity(target), replica, seconds, CONTROLLER_PLACEMENT_ID)

    def _cluster_read(self, replica: str, aggregate: Mapping[str, float],
                      performance: Optional[Mapping[str, Any]] = None) -> None:
        """The vLLM poller's read (Mode B, one call per replica)."""
        self._observe(
            cluster_identity(self._cluster_store(), self._mode_reader()), replica,
            counters_from_vllm(aggregate), performance,
        )

    def _cluster_miss(self, replica: str, seconds: Optional[float] = None) -> None:
        # A vLLM replica is named "<node id>:<port>" by its poller.
        node_id = str(replica).rpartition(":")[0]
        self._missed(
            cluster_identity(self._cluster_store(), self._mode_reader()), replica, seconds, node_id,
        )

    def _cluster_transition(self, replica: str, kind: str, now: Optional[float] = None) -> None:
        """Record that a vLLM replica joined or left the targets, even when the record is unreadable.

        The deployment's identity is optional here: a start or stop is a fact
        about the replica, and the dashboard reads it by node. A write that
        fails is kept and retried with the next transition.
        """
        node_id = str(replica).rpartition(":")[0]
        try:
            identity = cluster_identity(self._cluster_store(), self._mode_reader()) or {}
        except Exception:  # noqa: BLE001 - absence-ok: the marker does not need the record
            identity = {}
        self._write_transition({
            "replica": replica, "kind": kind, "at": time.time() if now is None else float(now),
            "identity": str(identity.get("identity") or ""), "node_id": node_id,
            "node_name": self._node_name(node_id), "placement": str(identity.get("placement") or "")})

    def _local_transition(self, replica: str, kind: str, now: Optional[float] = None) -> None:
        """Mode A's start or stop: the llama.cpp engine on this machine (pass-5 review)."""
        with self._lock:
            target = self._local_targets.get(replica)
        identity = local_identity(target) or {}
        self._write_transition({
            "replica": replica, "kind": kind, "at": time.time() if now is None else float(now),
            "identity": str(identity.get("identity") or ""), "node_id": CONTROLLER_PLACEMENT_ID,
            "node_name": CONTROLLER_PLACEMENT_NAME, "placement": PLACEMENT_SINGLE})

    def _last_transitions(self) -> Dict[str, str]:
        return self.ledger.last_transitions()

    def _flush_transitions(self) -> None:
        self._write_transition(None)

    def _write_transition(self, entry: Optional[Dict[str, Any]]) -> None:
        """Write a start or stop (or none), and any kept from a locked ledger, in order."""
        with self._lock:
            if entry is not None:
                self._pending_transitions.append(entry)
            pending, self._pending_transitions = self._pending_transitions, []
        if not pending:
            return
        written = 0
        try:
            for item in pending:
                self.ledger.record_transition(**item)
                written += 1
        except Exception as error:  # noqa: BLE001 - kept for the next attempt
            LOGGER.debug("A serving start or stop was not recorded yet: %s", error)
            with self._lock:
                self._pending_transitions = pending[written:] + self._pending_transitions

    def llm_server_usage(self, now: Optional[float] = None) -> Dict[str, Any]:
        """This collection's reading of :func:`llm_server_usage_state`."""
        return llm_server_usage_state(
            self.gate_store, self._bridge() if self._bridge is not None else None,
            enabled_reader=self._llm_server_enabled, now=now,
        )

    def start(self) -> None:
        self.gate.start()


def _llm_server_enabled() -> bool:
    from .llm_server_state import LlmServerStore

    return LlmServerStore().read().enabled


def llm_server_usage_state(
    gate_store: GateUsageStore, bridge: Any, *,
    enabled_reader: Optional[Callable[[], bool]] = None, now: Optional[float] = None,
) -> Dict[str, Any]:
    """Whether the LLM Server's requests are being counted: the one reading.

    Settings' External API card and the Performance card both ask this, so the
    two cannot disagree about whether a zero from the LLM Server is a measured
    zero. ``state`` is one of :data:`~vaelor.llm_gate_usage.USAGE_STATES`: an
    enabled server whose running gate is not recording, or whose bridge could
    not be asked, is not logging (:func:`gate_logging_state`); otherwise it is
    the drain's own state. ``enabled`` is whether the LLM Server is switched on
    (an unreadable flag counts as on, so a gate is never assumed idle). A store
    that cannot be read raises, for the caller to name.
    """
    try:
        enabled = bool((enabled_reader or _llm_server_enabled)())
    except Exception:  # noqa: BLE001 - an unreadable flag is "not known"
        enabled = True
    logging_state = gate_logging_state(True, cached_proxy_status(bridge)) if enabled else None
    state = dict(logging_state or gate_store.reader_state(now))
    state["enabled"] = enabled
    return state


def _model_tokens(metrics: Any, collection: UsageCollection, window_seconds: int,
                  clock: float) -> Optional[Dict[str, Any]]:
    """The model's tokens over the window, less AI Chat's where they can be told apart.

    AI Chat records its cluster answers (Mode B) in the gateway's store under
    its own source, so those tokens are subtracted. It records nothing for this
    machine's llama.cpp model (Mode A), so where that engine served in the
    window its AI Chat tokens stay in, and ``excludes_ai_chat`` says so. The
    two counts land in buckets up to one poll apart, so the difference is held
    at zero rather than going negative.
    """
    window = collection.ledger.window_totals(window_seconds, now=clock)
    if window.get("counting_since") is None:
        return None
    ai_chat = metrics.window_totals(window_seconds, sources=(SOURCE_AI_CHAT,), now=clock)
    return {
        "prompt_tokens": max(0, window["prompt_tokens"] - ai_chat["prompt_tokens"]),
        "completion_tokens": max(0, window["completion_tokens"] - ai_chat["completion_tokens"]),
        "counting_since": window["counting_since"],
        "since": window["since"],
        "excludes_ai_chat": ENGINE_LLAMACPP not in window.get("engines", ()),
    }


def external_usage(
    metrics: Any, collection: Optional[UsageCollection], *,
    window_seconds: int = EXTERNAL_WINDOW_SECONDS, now: Optional[float] = None,
) -> Dict[str, Any]:
    """What reached the model from outside Vaelor over a window (ACC-046/048).

    Requests, failures and latency are counted at the two doors an outside
    client can use - the inference gateway (only its own requests, never AI
    Chat's) and the LLM Server gate - and summed, each door with the start of
    the span it really covers. Tokens are the MODEL's count for the window
    less AI Chat's (:func:`_model_tokens`); the LLM Server cannot see tokens
    per request. A door that is not being read, or a gate that is not
    recording, is ``None`` with a sentence saying why - never a zero.
    """
    clock = time.time() if now is None else float(now)
    gateway = metrics.window_totals(window_seconds, sources=(SOURCE_GATEWAY,), now=clock)
    llm_server: Optional[Dict[str, Any]] = None
    llm_state: Dict[str, Any] = {"state": "", "detail": "The LLM Server's usage is not collected here."}
    model_tokens: Optional[Dict[str, Any]] = None
    if collection is not None:
        try:
            llm_state = collection.llm_server_usage(clock)
            if llm_state["state"] == USAGE_COUNTING:
                llm_server = collection.gate_store.window_totals(window_seconds, now=clock)
        except Exception as error:  # noqa: BLE001 - one unreadable door must not hide the other
            llm_state = {"state": USAGE_UNREADABLE, "detail": "The LLM Server's usage could not be read ({}).".format(error)}
        try:
            model_tokens = _model_tokens(metrics, collection, window_seconds, clock)
        except Exception as error:  # noqa: BLE001 - tokens are reported apart
            LOGGER.debug("Model usage could not be read: %s", error)
    requests = gateway["requests"] + (llm_server["requests"] if llm_server else 0)
    duration = gateway["duration_ms_sum"] + (llm_server["duration_ms_sum"] if llm_server else 0.0)
    since = gateway["since"]
    return {
        "window_seconds": int(window_seconds),
        # One start when the doors cover the same span (to within a bucket),
        # otherwise each door's own below.
        "since": since if not llm_server or abs(llm_server["since"] - since) <= BUCKET_SECONDS
        else max(since, llm_server["since"]),
        "requests": requests,
        "failures": gateway["failures"] + (llm_server["failures"] if llm_server else 0),
        "client_errors": gateway["client_errors"] + (llm_server["client_errors"] if llm_server else 0),
        "average_latency_ms": round(duration / requests, 1) if requests else None,
        "gateway": {"requests": gateway["requests"], "failures": gateway["failures"], "since": since},
        "llm_server": (
            {
                "requests": llm_server["requests"], "failures": llm_server["failures"],
                "refused": llm_server["refused"], "since": llm_server["since"],
            }
            if llm_server else None
        ),
        "llm_server_state": llm_state.get("state", ""),
        "llm_server_detail": llm_state.get("detail", ""),
        "model_tokens": model_tokens,
    }
