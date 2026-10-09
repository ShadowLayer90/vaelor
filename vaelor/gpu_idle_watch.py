"""Scale-to-zero for a served vLLM cluster deployment (G3b).

G3a gave an operator a manual Unload (reclaim the GPU, keep the record warm) and
a manual Load (re-serve it warm from the compile cache). G3b adds the automatic
half: a deployment carrying an ``idle_timeout`` inside its ``units`` blob is
unloaded on its own once it has served nothing for that long, and is woken on
the next request that names it.

**Why this is its own module.** The three serving-core modules
(`gpu_cluster_mode`, `gpu_pool_operations`, `gpu_pool_runtime`) each sit at the
1,000-line ceiling `CLAUDE.md` sets, so the idle detector - the metrics scrape,
the ``vllm:`` text parser, the per-mode aggregation, the consecutive-idle
counter, the pin-warm exclusion and the idle decision - lives here, and the mode
reconcile calls :func:`observe_idle` in one line.

**Where each half runs (BL-1, BL-2, BL-3).**

* *Detect (BL-3).* :func:`observe_idle` runs on the executor's mode-reconcile
  thread, AFTER the switch pass has released the GPU serving lock. It scrapes
  each vLLM server's ``/metrics`` over the transport the operations already hold
  (a loopback GET on this controller, a ``python3`` one-liner over the worker's
  SSH channel elsewhere - the bridge argv policy admits no ``curl``, and vLLM
  binds ``/metrics`` unkeyed on every node's loopback). A ``replicated`` record
  is idle only when EVERY replica reads zero; a ``distributed`` one is read on
  its lead node.

* *Unload (BL-2).* When the consecutive-idle count crosses the threshold the
  reconcile does NOT unload inline - the serving lock is a re-entrant lock the
  reconcile already holds through the switch pass, and an inline unload would
  hold it across the multi-second unit stops G3a deliberately runs outside it.
  It ENQUEUES a ``cluster.gpu.unload`` job carrying ``reason: "idle"``; the
  executor's job loop then runs it with the same lock discipline the manual
  path uses.

* *Wake (BL-1).* An idle unload does NOT repoint AI Chat onto the on-device
  Assistant the way a manual unload does (`gpu_pool_reload`): it leaves the
  ``ai-chat`` lease on the unloaded cluster credential, so the next send resolves
  a cluster deployment in state ``unloaded``. :func:`wake_unloaded_cluster` is the
  narrow seam AI Chat and the external gateway call at lease-resolution time: it
  reads the mode record, confirms the active lease is the cluster's and the
  record is ``unloaded``, enqueues one warm ``cluster.gpu.load`` and returns the
  name so the caller can answer an honest "the model is waking" rather than a
  hang or a wrong endpoint.

The load enqueue is a CANONICAL ``{name, confirm}`` payload so the job store's
exact-payload dedup collapses a burst of requests into one load, and
:func:`load_pending` mirrors ``executor_service.assistant_deploy_pending`` to
cover the window after a load leaves the deduped states but is still running.
"""

from __future__ import annotations

import logging
import math
import urllib.request
from dataclasses import dataclass
from typing import Any, Callable, Dict, List, Mapping, Optional, Tuple

from .cluster_gpu_sizing import VERDICT_REPLICATED
from .cluster_job_confirmations import confirmation_for
from .cluster_placement import CONTROLLER_PLACEMENT_ID
from .gpu_cluster_mode import DEPLOYING_STATE, MODE_CLUSTER, UNLOADED_STATE
from .gpu_pool_replicas import replica_entries
from .agent_deployments import STATE_DEPLOYING, STATE_HEALTHY
from .gpu_pool_units import VLLM_ENGINE, deployment_name, record_mode
from .job_vocabulary import (
    ACTIVE_JOB_STATES, CLUSTER_GPU_LOAD_JOB, CLUSTER_GPU_REFRESH_JOB, CLUSTER_GPU_UNLOAD_JOB,
)

LOGGER = logging.getLogger(__name__)

#: The wake seam's word for the LLM Server's wake door, which is not a broker
#: purpose. That door woke through AI Chat's lease until VD-210 let the owner
#: move AI Chat off the cluster for good; it now wakes whatever the LLM Server
#: fronts (`gpu_serving_target.resolve_gpu_serving_target`), so an owner's
#: choice in AI Chat does not stop LAN clients waking an idle model, and a
#: manual unload is still never woken.
LLM_SERVER_WAKE = "llm-server"

#: The Prometheus gauge names a running vLLM server publishes for its in-flight
#: work, and the counter for finished requests. Kept as module constants because
#: the exact spellings are a live-verify item (a vLLM upgrade may re-prefix
#: them): change them here and both the periodic detector and the under-lock
#: re-check follow. The gauge pair is the primary idle signal; the counter delta
#: is the robust cross-check that catches a request that both began and ended
#: between two polls, which would read the gauges as zero on either side.
METRIC_RUNNING = "vllm:num_requests_running"
METRIC_WAITING = "vllm:num_requests_waiting"
METRIC_SUCCESS = "vllm:request_success_total"

#: How often the mode reconcile calls :func:`observe_idle` - the same cadence
#: `executor_service.GPU_CHAT_SUPERVISE_INTERVAL_SECONDS` runs that thread on.
#: Spelled here rather than imported to keep this module free of the executor
#: wiring that drives it; the required-poll arithmetic reads it and a test
#: overrides it directly.
IDLE_POLL_INTERVAL_SECONDS = 30.0

#: The floor an ``idle_timeout`` is clamped up to before it decides anything. A
#: window shorter than this would let a model that answered one request flap
#: straight back to unloaded before a follow-up landed; two minutes bounds the
#: reload-then-unload churn while still reclaiming a genuinely idle GPU quickly.
MINIMUM_IDLE_TIMEOUT_SECONDS = 120

#: How long a single ``/metrics`` read may take. Short, because the periodic
#: read is off the lock and the under-lock re-check must not stall serving: a
#: server that cannot answer in this budget is treated as busy (never unloaded),
#: not waited on.
SCRAPE_TIMEOUT_SECONDS = 4

#: The reader that pulls a node's loopback ``/metrics`` when this controller is
#: that node - a direct GET, because the executor runs here and the bridge argv
#: policy admits no HTTP client. A worker's is read through its SSH channel with
#: this one-liner, whose only argv token is ``python3`` (which that channel's
#: allowlist admits and the bridge's does not).
_METRICS_FETCH_PROGRAM = (
    "import sys,urllib.request;"
    "sys.stdout.write(urllib.request.urlopen("
    "'http://127.0.0.1:{port}/metrics',timeout={timeout}).read()"
    ".decode('utf-8','replace'))"
)


@dataclass(frozen=True)
class MetricsReading:
    """One vLLM server's idle-relevant numbers, any absent as ``None``.

    ``running`` and ``waiting`` are the in-flight gauges; a reading with either
    missing is not usable for an idle verdict (something answered but not the
    shape expected). ``success`` is the finished-request counter, used only as a
    between-poll delta and tolerated absent.
    """

    running: Optional[float]
    waiting: Optional[float]
    success: Optional[float]

    @property
    def usable(self) -> bool:
        return self.running is not None and self.waiting is not None

    @property
    def in_flight(self) -> float:
        return float(self.running or 0.0) + float(self.waiting or 0.0)


def parse_vllm_metrics(text: str) -> MetricsReading:
    """Read the three idle-relevant series out of a Prometheus text body.

    No regular expression: each non-comment line is ``name{labels} value`` (the
    labels optional), so the metric name is the text before the first brace or
    space and the value is the last whitespace-separated field. A name may
    appear on several labelled series (one per model, or per finish reason for
    the counter), so matching series are SUMMED - the aggregate is what "is
    anything in flight" and "did any request finish" ask. A name that never
    appears stays ``None`` so a missing gauge is told from a gauge reading zero.
    """
    running: Optional[float] = None
    waiting: Optional[float] = None
    success: Optional[float] = None
    for raw in str(text or "").splitlines():
        line = raw.strip()
        if not line or line.startswith("#"):
            continue
        fields = line.split()
        if len(fields) < 2:
            continue
        series = fields[0]
        name = series.split("{", 1)[0]
        try:
            value = float(fields[-1])
        except (TypeError, ValueError):
            continue
        if name == METRIC_RUNNING:
            running = (running or 0.0) + value
        elif name == METRIC_WAITING:
            waiting = (waiting or 0.0) + value
        elif name == METRIC_SUCCESS:
            success = (success or 0.0) + value
    return MetricsReading(running=running, waiting=waiting, success=success)


def idle_timeout_from_payload(payload: Mapping[str, Any]) -> int:
    """The non-negative ``idle_timeout`` seconds a serve payload asked for.

    Zero (or absent, or unparseable) means scale-to-zero is off for the
    deployment - the normal case. The value is stored raw and clamped to
    :data:`MINIMUM_IDLE_TIMEOUT_SECONDS` only when it is used to decide, so a
    later change to the floor needs no re-write of a stored record.
    """
    try:
        seconds = int((payload or {}).get("idle_timeout", 0) or 0)
    except (TypeError, ValueError):
        return 0
    return max(0, seconds)


def read_idle_timeout(deployment: Mapping[str, Any]) -> int:
    """The ``idle_timeout`` a stored record carries inside its ``units`` blob."""
    units = deployment.get("units") if isinstance(deployment, Mapping) else None
    return idle_timeout_from_payload(units if isinstance(units, Mapping) else {})


def required_idle_polls(
    idle_timeout: int, interval: float = IDLE_POLL_INTERVAL_SECONDS
) -> int:
    """Consecutive idle polls that stand for an idle window of ``idle_timeout``.

    The window is clamped up to :data:`MINIMUM_IDLE_TIMEOUT_SECONDS` first, then
    divided by the poll cadence and rounded up, so a whole window of quiet is
    always observed before an unload and at least one poll is always required.
    """
    window = max(int(idle_timeout), MINIMUM_IDLE_TIMEOUT_SECONDS)
    step = interval if interval and interval > 0 else IDLE_POLL_INTERVAL_SECONDS
    return max(1, math.ceil(window / step))


def deployment_idle_targets(
    deployment: Mapping[str, Any]
) -> List[Tuple[str, int]]:
    """The ``(node_id, port)`` servers whose ``/metrics`` decide this record's idle.

    A ``replicated`` record has one independent vLLM server per replica behind
    the balancer, so every replica's loopback port is a target and idle needs
    ALL of them. A ``distributed`` record has one server, on its lead node
    (``node_ids[0]``, which the record wrote lead-first), read on that node's
    own loopback port - never a hardcoded controller loopback, which would miss
    a worker-led deployment entirely.
    """
    units = deployment.get("units") or {}
    if record_mode(deployment) == VERDICT_REPLICATED:
        return [
            (str(entry.get("node_id", "")), int(entry.get("port", 0) or 0))
            for entry in replica_entries(deployment)
            if int(entry.get("port", 0) or 0)
        ]
    node_ids = [str(node) for node in (deployment.get("node_ids") or []) if str(node)]
    port = int(units.get("port", 0) or 0)
    return [(node_ids[0], port)] if node_ids and port else []


def _default_http_get(url: str) -> str:
    with urllib.request.urlopen(url, timeout=SCRAPE_TIMEOUT_SECONDS) as response:
        return response.read(256 * 1024).decode("utf-8", "replace")


def read_target_body(
    ops: Any, node_id: str, port: int,
    http_get: Optional[Callable[[str], str]] = None,
) -> Optional[str]:
    """The raw ``/metrics`` text of one server, or ``None`` when it did not answer.

    The transport half of :func:`scrape_target`, split out so a second reader -
    the serving-card vLLM scrape - can parse the SAME body for a wider set of
    families without re-implementing, or re-securing, the controller-loopback vs
    worker-``python3`` transport this module already proved. This controller's own
    server is a direct loopback GET (the executor is co-located and the bridge
    admits no HTTP client); a worker's is read on its own loopback through the SSH
    channel the operations resolve, whose one argv token is ``python3``. Any
    failure - an unreachable node, a timeout, an empty port - is ``None``, never a
    raise into the caller's loop.
    """
    if not port:
        return None
    getter = http_get or _default_http_get
    try:
        if str(node_id) == CONTROLLER_PLACEMENT_ID:
            return getter("http://127.0.0.1:{}/metrics".format(int(port)))
        node = ops.joined_node(node_id, include_credential=True)
        return ops._transport(node).run(
            [
                "python3", "-c",
                _METRICS_FETCH_PROGRAM.format(
                    port=int(port), timeout=SCRAPE_TIMEOUT_SECONDS
                ),
            ],
            timeout=SCRAPE_TIMEOUT_SECONDS,
        )
    except Exception as error:  # noqa: BLE001 - an unreadable server is not idle
        LOGGER.debug("Idle scrape of %s:%s did not answer: %s", node_id, port, error)
        return None


def scrape_target(
    ops: Any, node_id: str, port: int, http_get: Callable[[str], str]
) -> Optional[MetricsReading]:
    """Read one server's ``/metrics`` into the idle reading, or ``None``.

    A thin wrapper over :func:`read_target_body`: it takes that body over the
    same controller-loopback-or-worker transport and turns it into the three
    idle-relevant series. A body that could not be read is ``None``, which the
    aggregation reads as "cannot confirm idle" and so never unloads on.
    """
    body = read_target_body(ops, node_id, port, http_get)
    return parse_vllm_metrics(body) if body is not None else None


def aggregate_idle(
    readings: List[Optional[MetricsReading]], previous_success: Optional[float]
) -> Tuple[bool, Optional[float]]:
    """Is every server idle this poll, and the finished-request total to remember.

    Idle requires that every target was readable, that no request is running or
    waiting on any of them, AND that the finished-request counter has not moved
    since the last poll (the between-poll cross-check). A single unreadable
    target makes the whole verdict "not idle" - a deployment is only scaled to
    zero on a complete, confident reading. The returned total is the new
    baseline for the next poll, or ``None`` when no server reported the counter.
    """
    if not readings or any(reading is None or not reading.usable for reading in readings):
        return False, previous_success
    in_flight = sum(reading.in_flight for reading in readings)  # type: ignore[union-attr]
    totals = [
        reading.success for reading in readings  # type: ignore[union-attr]
        if reading.success is not None
    ]
    current_success = sum(totals) if totals else None
    served_since = (
        previous_success is not None
        and current_success is not None
        and current_success > previous_success
    )
    idle = in_flight == 0 and not served_since
    return idle, current_success


def is_deployment_idle_now(
    ops: Any, deployment: Mapping[str, Any],
    *, http_get: Optional[Callable[[str], str]] = None,
) -> bool:
    """A single fresh idle reading, for the under-lock re-check (CN-1).

    The idle unload op calls this the instant before it stops the units, so the
    read-then-act gap the enqueue opened cannot unload a model a request reached
    in the meantime. One scrape, no counter and no delta (there is no earlier
    poll to compare against under the lock): every target must be readable and
    report nothing in flight, or the answer is False and the unload no-ops.
    """
    getter = http_get or _default_http_get
    targets = deployment_idle_targets(deployment)
    if not targets:
        return False
    readings = [scrape_target(ops, node_id, port, getter) for node_id, port in targets]
    if any(reading is None or not reading.usable for reading in readings):
        return False
    return sum(reading.in_flight for reading in readings) == 0  # type: ignore[union-attr]


def pinned_warm(agent_store: Any, name: str) -> bool:
    """Whether a live cluster agent pins this deployment warm.

    A deployment backing an agent whose own state is ``deploying`` or ``healthy``
    must not be scaled to zero underneath it: the agent would meet a dead
    endpoint on its next call. The match is on the immutable
    ``backing.model_deployment_name`` key; a ``removing`` or ``failed`` agent is
    not a live consumer and does not pin. An unreadable store is treated as no
    pin, so a query error never wedges scale-to-zero off.
    """
    try:
        rows = agent_store.list()
    except Exception:  # noqa: BLE001 - an unreadable agent store pins nothing
        return False
    for row in rows:
        backing = row.get("backing") or {}
        if str(backing.get("model_deployment_name", "")) != str(name):
            continue
        if str(row.get("state", "")) in {STATE_DEPLOYING, STATE_HEALTHY}:
            return True
    return False


def load_pending(store: Any, name: str) -> bool:
    """Whether a warm load for ``name`` is already queued or running.

    Mirrors ``executor_service.assistant_deploy_pending``: the job store's
    exact-payload dedup collapses a burst only while a job sits in its deduped
    states, so a load that has moved on to ``running`` would otherwise admit a
    second. This reads any ``cluster.gpu.load`` for this deployment in
    ``queued`` or an active state as "already loading". An unreadable store
    reads as nothing pending, with the store's own dedup the backstop.
    """
    canonical = deployment_name(name)
    try:
        records = store.list(limit=200)
    except (OSError, ValueError):
        return False
    for record in records:
        # A post-upgrade refresh is an Unload and a Load in one job (W4-D1):
        # its Load is the one a wake would ask for, so a wake in its gap is
        # merged into it rather than queued to meet a row it is reloading.
        if record.get("type") not in (CLUSTER_GPU_LOAD_JOB, CLUSTER_GPU_REFRESH_JOB):
            continue
        if deployment_name((record.get("payload") or {}).get("name", "")) != canonical:
            continue
        state = str(record.get("state") or "")
        if state == "queued" or state in ACTIVE_JOB_STATES:
            return True
    return False


def enqueue_cluster_load(store: Any, name: str) -> Optional[Dict[str, Any]]:
    """Enqueue ONE warm ``cluster.gpu.load``, or no-op when one is pending.

    The canonical ``{name, confirm}`` payload (actor ``system``) is byte-identical
    across every consumer, so the store's dedup collapses a concurrent burst; the
    :func:`load_pending` guard covers the running window past the deduped states.
    Returns the created (or deduped) record, or ``None`` when a load was already
    pending.
    """
    canonical = deployment_name(name)
    if not canonical or load_pending(store, canonical):
        return None
    return store.create(
        CLUSTER_GPU_LOAD_JOB, "system",
        {"name": canonical, "confirm": confirmation_for(CLUSTER_GPU_LOAD_JOB)},
    )


def enqueue_idle_unload(store: Any, name: str) -> Optional[Dict[str, Any]]:
    """Enqueue a ``cluster.gpu.unload`` carrying ``reason: "idle"`` for ``name``.

    The reason flag is what makes the shared unload op do its fresh under-lock
    idle re-check (CN-1) rather than the manual path's unconditional stop. The
    store's exact-payload dedup collapses repeats while one is queued or active.
    """
    canonical = deployment_name(name)
    if not canonical:
        return None
    return store.create(
        CLUSTER_GPU_UNLOAD_JOB, "system",
        {
            "name": canonical,
            "confirm": confirmation_for(CLUSTER_GPU_UNLOAD_JOB),
            "reason": "idle",
        },
    )


class GpuIdleWatch:
    """The per-deployment consecutive-idle state, held across reconcile passes.

    Kept in memory beside the mode switch (which lives for the executor's life),
    never persisted: a restart counting afresh delays an unload by at most the
    window it lost and can never invent one - the same choice
    `gpu_pool_replicas.ReplicaHealth` makes for its down-pass counter.
    """

    def __init__(self) -> None:
        self._idle_polls: Dict[str, int] = {}
        self._last_success: Dict[str, Optional[float]] = {}
        self._healthy: set = set()
        #: Deployments a pinned-warm skip has already logged once, so an
        #: operator sees the skip at the start of a pin episode and not on
        #: every 30 s poll for as long as the agent lives (DEFECT 3).
        self._pin_logged: set = set()

    def forget(self, name: str) -> None:
        """Drop a deployment's idle progress when it is no longer healthy-serving.

        Called whenever the pass is not a healthy Mode-B verdict for this name -
        an unload landed, a load is in flight, the cluster left - so the next
        return to healthy starts the idle window fresh from that transition.
        """
        self._healthy.discard(name)
        self._idle_polls.pop(name, None)
        self._last_success.pop(name, None)
        self._pin_logged.discard(name)

    def observe(
        self, *, name: str, deployment: Mapping[str, Any], ops: Any,
        job_store: Any, agent_store: Any, http_get: Callable[[str], str],
        interval: float = IDLE_POLL_INTERVAL_SECONDS,
    ) -> Dict[str, Any]:
        """One idle poll of a healthy-serving deployment; enqueue an unload if due.

        The order matters: a just-healthy deployment resets the counter and reads
        nothing this pass (CN-2 warm-up, so it cannot unload on the zero requests
        a fresh model shows); an ``idle_timeout`` of zero is off; a pinned-warm
        deployment is excluded; otherwise the servers are scraped and the
        consecutive-idle count advances, and at the threshold a ``cluster.gpu.unload``
        job is enqueued and the counter reset.
        """
        if name not in self._healthy:
            self._healthy.add(name)
            self._idle_polls[name] = 0
            self._last_success[name] = None
            return {"idle": False, "reason": "warming"}
        idle_timeout = read_idle_timeout(deployment)
        if idle_timeout <= 0:
            self._idle_polls[name] = 0
            return {"idle": False, "reason": "disabled"}
        if pinned_warm(agent_store, name):
            self._idle_polls[name] = 0
            if name not in self._pin_logged:
                self._pin_logged.add(name)
                LOGGER.warning(
                    "Holding GPU deployment '%s' warm: a live agent depends "
                    "on it, so scale-to-zero is skipped until that agent is "
                    "removed.", name,
                )
            return {"idle": False, "reason": "pinned-warm"}
        self._pin_logged.discard(name)
        targets = deployment_idle_targets(deployment)
        readings = [scrape_target(ops, node_id, port, http_get) for node_id, port in targets]
        idle_now, current_success = aggregate_idle(
            readings, self._last_success.get(name)
        )
        if current_success is not None:
            self._last_success[name] = current_success
        if not idle_now:
            self._idle_polls[name] = 0
            return {"idle": False, "reason": "busy"}
        polls = self._idle_polls.get(name, 0) + 1
        self._idle_polls[name] = polls
        if polls < required_idle_polls(idle_timeout, interval):
            return {"idle": True, "reason": "counting", "polls": polls}
        self._idle_polls[name] = 0
        enqueue_idle_unload(job_store, name)
        LOGGER.warning(
            "Scaling the idle GPU deployment '%s' to zero after %d idle checks "
            "(idle_timeout %ds); a request will wake it.",
            name, polls, idle_timeout,
        )
        return {"idle": True, "reason": "unload-enqueued", "polls": polls}


def _watch_for(switch: Any) -> GpuIdleWatch:
    """The idle state for this switch, created once and kept on it.

    The switch is the one collaborator the reconcile pass holds that outlives a
    single pass, so the counter rides on it rather than in a module global that
    a test could not isolate. A test drives :meth:`GpuIdleWatch.observe` directly
    with its own instance instead.
    """
    watch = getattr(switch, "_idle_watch", None)
    if not isinstance(watch, GpuIdleWatch):
        watch = GpuIdleWatch()
        try:
            switch._idle_watch = watch
        except (AttributeError, TypeError):
            pass
    return watch


def _healthy_serving(result: Mapping[str, Any]) -> bool:
    """Whether a reconcile result is a healthy, converged Mode-B verdict.

    That is the one verdict scale-to-zero acts on: the tail of the switch's
    healthy pass returns cluster mode, reconciled true and an empty reason,
    which every other verdict (a leave, a deploy in flight, an unloaded row)
    fails on.
    """
    return (
        bool(result)
        and str(result.get("mode", "")) == MODE_CLUSTER
        and bool(result.get("reconciled"))
        and not str(result.get("reason", ""))
    )


def observe_idle(
    switch: Any, ops: Any, cluster_store: Any, result: Mapping[str, Any],
    job_store: Any, *, agent_store: Any = None,
    http_get: Optional[Callable[[str], str]] = None,
    interval: float = IDLE_POLL_INTERVAL_SECONDS,
) -> Dict[str, Any]:
    """The mode reconcile's one-line idle hook, run after the switch pass.

    A no-op without the executor's collaborators (a control-plane reconcile has
    no job store and no operations) or when the pass was not a healthy Mode-B
    verdict, in which case the deployment's idle progress is forgotten so a later
    return to healthy starts fresh. Otherwise it reads the record the mode file
    names and hands one idle poll to the per-switch :class:`GpuIdleWatch`. Never
    raises: it runs on the same daemon thread as the switch pass, and one bad
    poll must not stop the next.
    """
    watch = _watch_for(switch)
    try:
        name = switch.state().deployment_name
    except Exception:  # noqa: BLE001 - an unreadable mode file is not serving
        return {"idle": False, "reason": "no-state"}
    if not name or job_store is None or ops is None or not _healthy_serving(result):
        if name:
            watch.forget(name)
        return {"idle": False, "reason": "not-serving"}
    try:
        deployment = cluster_store.get_pooled_deployment(name)
    except Exception:  # noqa: BLE001 - a locked store decides nothing this pass
        return {"idle": False, "reason": "store-unavailable"}
    units = (deployment or {}).get("units") or {}
    if deployment is None or units.get("engine") != VLLM_ENGINE:
        watch.forget(name)
        return {"idle": False, "reason": "not-vllm"}
    if agent_store is None:
        from .agent_deployments import AgentDeploymentStore

        agent_store = AgentDeploymentStore()
    try:
        return watch.observe(
            name=name, deployment=deployment, ops=ops, job_store=job_store,
            agent_store=agent_store, http_get=http_get or _default_http_get,
            interval=interval,
        )
    except Exception as error:  # noqa: BLE001 - a bad poll must not kill the watch
        LOGGER.warning("The GPU idle-watch pass did not finish: %s", error)
        return {"idle": False, "reason": "error"}


def wake_unloaded_cluster(
    *, purpose: str, broker: Any, mode_store: Any, cluster_store: Any,
    enqueue_load: Callable[[str], Any], wake: bool = True,
) -> str:
    """Wake an idle-unloaded cluster model the active ``purpose`` lease points at.

    The lease-resolution seam AI Chat (``ai-chat``) and the external gateway
    (``cluster-inference``) call before they forward. It confirms the mode file
    is clustering, that the active lease for ``purpose`` is the cluster's own
    credential (so an operator who repointed the lease elsewhere is untouched),
    and that the named deployment is in state ``unloaded`` (the scaled-to-zero
    rest state) or ``deploying`` (a load already in flight); a crashed or
    removed one in any other state it leaves alone. For ``unloaded`` it
    enqueues one warm load; for ``deploying`` it enqueues nothing, the load
    being already under way. Either returns the deployment name so the caller
    answers an honest "waking / starting, retry shortly"; otherwise ``""``.

    ``wake=False`` asks the same question and enqueues nothing: the gateway's
    model listing uses it to answer from the lease, because a listing is not
    inference and waking on it made a polling client cycle the model through
    load and idle-unload for ever (ACC-057).
    """
    try:
        state = mode_store.read()
    except Exception:  # noqa: BLE001 - an unreadable mode file wakes nothing
        return ""
    name = getattr(state, "deployment_name", "")
    if getattr(state, "mode", "") != MODE_CLUSTER or not name:
        return ""
    if not _wakes_for(purpose, broker, state):
        return ""
    try:
        deployment = cluster_store.get_pooled_deployment(name)
    except Exception:  # noqa: BLE001 - a locked store wakes nothing this call
        return ""
    record_state = str((deployment or {}).get("state", ""))
    # ``unloaded`` is the scaled-to-zero rest state that enqueues one warm
    # load; ``deploying`` means a load or deploy is ALREADY in flight, so it
    # enqueues nothing yet still returns the name, which is what lets the
    # caller answer the honest 'starting, retry shortly' for the whole load
    # window instead of the dead-endpoint error. Any other state (a crashed
    # or removed deployment) is left alone.
    if deployment is None or record_state not in (UNLOADED_STATE, DEPLOYING_STATE):
        return ""
    if record_state == UNLOADED_STATE and wake:
        try:
            enqueue_load(name)
        except Exception as error:  # noqa: BLE001 - the honest 503 still stands
            LOGGER.warning("Could not enqueue a wake load for '%s': %s", name, error)
    return str(name)


def _wakes_for(purpose: str, broker: Any, state: Any) -> bool:
    """Whether ``purpose``'s caller may wake the Mode B cluster at all.

    A broker purpose wakes only while its own active lease is the cluster's
    credential, so a caller whose lease the owner pointed elsewhere is
    untouched. :data:`LLM_SERVER_WAKE` asks the LLM Server's own question
    instead: is the cluster the target its door fronts.
    """
    recorded = getattr(state, "cluster_credential_id", "")
    if purpose == LLM_SERVER_WAKE:
        from .gpu_serving_target import KIND_CLUSTER, resolve_gpu_serving_target

        try:
            target = resolve_gpu_serving_target(broker, state)
        except Exception:  # noqa: BLE001 - an unreadable target wakes nothing
            return False
        return target.kind == KIND_CLUSTER and target.credential_id == recorded
    try:
        lease = broker.resolve_active(purpose)
    except Exception:  # noqa: BLE001 - no active lease is nothing to wake
        return False
    return str(lease.get("credential_id", "")) == recorded
