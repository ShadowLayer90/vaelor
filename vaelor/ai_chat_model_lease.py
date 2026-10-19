"""The one answer to "what is AI Chat's model right now?" (VD-220 B1).

**One resolver, three readers** (LESSONS 6). AI Chat itself
(`ChatInference._connection`), the research escalation
(`model_connection.resolve_model_connection(mode="capable")`) and custom agents
(``mode="agent"``, and `AgentTaskRunner` before a run) all ask
:func:`resolve_ai_chat_model`. Before VD-220 the first two each read the
``ai-chat`` lease their own way, and only AI Chat knew the cluster sentence.

**Custom agents run on AI Chat's model, wherever it points** (VD-220, owner):
this machine's GPU model, the GPU cluster, a server on the owner's network or
a hosted service (VD-206). Never the Assistant's NPU model, and never a
fallback to it: when AI Chat has no model the run fails at once with
:attr:`AiChatModel.reason`. A cluster scaled to zero is woken and the run is
reported unavailable; it does not wait.

**How a custom agent reaches a hosted service.** Only through
`hosted_transport`, as AI Chat does (`chat_hosted.answer_hosted`):
:func:`custom_agent_model` stamps :data:`CUSTOM_AGENT_ON_AI_CHAT` on the
connection, and `inference_client.chat_completion` sends a hosted connection
only when it carries that stamp (:func:`hosted_agent_completion`). Nothing else
sets it, so the Assistant, deploy planning and research stay refused a hosted
model exactly as before (LESSONS 18). OpenAI is reached as AI Chat reaches it.

Processes: AI Chat and the agent runner both live in the control plane, which
holds the cluster wake seam; the research escalation runs in the executor and
never wakes anything.
"""

from __future__ import annotations

import time
from contextlib import ExitStack, contextmanager
from contextvars import ContextVar
from dataclasses import dataclass
from typing import Any, Callable, Dict, Iterator, Mapping, Optional

from .credential_broker import CredentialError
from .local_inference_gate import LocalModelBusy, cluster_inference_slot, local_inference_slot
from .hosted_providers import HOSTED_ANSWER_TOKENS, HOSTED_KINDS, HOSTED_PROVIDERS
from .model_credential_roles import (
    AI_CHAT_NEVER_ON_THE_ASSISTANT_NPU, ASSISTANT_PURPOSE, ai_chat_may_use_lease,
)
from .provider_runtime import managed_local_connection

OPENAI_BASE_URL = "https://api.openai.com/v1"
OPENAI_DEFAULT_MODEL = "gpt-5.6-terra"

#: The mode a custom agent's model calls are resolved under (VD-220).
AGENT_MODE = "agent"

#: Stamped on a connection only by :func:`custom_agent_model`: the one licence
#: the shared inference path accepts to send to a hosted service (VD-220).
CUSTOM_AGENT_ON_AI_CHAT = "custom_agent_on_ai_chat"
#: Stamped beside it in Mode B: the request takes AI Chat's cluster permits.
AI_CHAT_CLUSTER_SLOT = "ai_chat_cluster_slot"

#: What AI Chat answers when nothing it may use holds its lease.
CHOOSE_AN_AI_CHAT_CONNECTION = "Choose an AI Chat connection before sending a message."
#: What AI Chat answers in Mode B while the cluster holds no lease for it: the
#: cluster is loading, or its model was unloaded by hand (W4d-D26). AI Chat is
#: never parked on the NPU Assistant meanwhile, so this is the honest answer
#: rather than "choose a connection", which the cluster forbids in Mode B.
AI_CHAT_CLUSTER_NOT_SERVING = (
    "GPU clustering serves AI Chat, and its model is not serving right now: it "
    "is still loading, or it was unloaded in Cluster > Deployments. Load it "
    "there, or retry once it has finished loading."
)
AI_CHAT_CONNECTION_INCOMPLETE = "The AI Chat connection is incomplete."
#: LESSONS 8: a broker that is down or answered nothing is "unknown", never
#: "no model chosen" - the two have opposite remedies (kb R195).
AI_CHAT_MODEL_UNREADABLE = (
    "Vaelor could not read which model AI Chat is set to ({detail}). Retry in a moment."
)

# The states one resolution can end in. READY carries a connection; every
# other state carries the sentence that says why there is none.
READY = "ready"
NO_MODEL = "none"
ON_THE_NPU = "npu"
CLUSTER_NOT_SERVING = "cluster-not-serving"
INCOMPLETE = "incomplete"
UNREADABLE = "unreadable"
MODEL_NOT_CHOSEN = "model-not-chosen"
WAKING = "waking"
WAKE_TIMED_OUT = "wake-timed-out"
BUSY = "busy"
#: A wake wait that ended because the run was cancelled, or the runner stopped.
CANCELLED = "cancelled"
STOPPED = "stopped"

#: VD-220 (owner, on review R196): a scheduled or alert-triggered run waits
#: this long for the wake it started, then runs; a hand-started run does not.
WAKE_WAIT_SECONDS = 300
WAKE_POLL_SECONDS = 10
#: VD-220 (owner, on review R194): how long one custom-agent model call waits
#: for its turn on AI Chat's model before it fails as busy. AI Chat itself still
#: refuses at once (it holds a request worker; the agent runner does not).
AGENT_TURN_WAIT_SECONDS = 120

_AGENT = "Custom agents run on AI Chat's model, and "
#: The runner's error code in front of an AGENT_UNAVAILABLE sentence. The
#: console strips it and keys "Go to AI Chat" on `_AGENT`; both are pinned
#: against `frontend/src/components/customAgentModel.tsx` (LESSONS 6).
AI_CHAT_UNAVAILABLE_CODE = "ai_chat_model_unavailable"
#: What a custom agent's run says, per state (VD-220: plain, names why, no wait).
AGENT_UNAVAILABLE = {
    NO_MODEL: _AGENT + (
        "AI Chat has no model chosen. Choose one in AI Chat, then run this agent again."
    ),
    ON_THE_NPU: _AGENT + (
        "AI Chat is still set to the Assistant's on-device (NPU) model, which "
        "it may not use. Choose a model in AI Chat, then run this agent again."
    ),
    CLUSTER_NOT_SERVING: _AGENT + (
        "that is the GPU cluster's model, which is not serving right now: it is "
        "still loading, or it was unloaded in Cluster > Deployments. Load it "
        "there, or run this agent again once it has finished loading."
    ),
    INCOMPLETE: _AGENT + (
        "AI Chat's connection is incomplete. Choose it again in AI Chat, then "
        "run this agent again."
    ),
    MODEL_NOT_CHOSEN: _AGENT + (
        "no model is chosen for {label} in AI Chat. Choose one there, then run "
        "this agent again."
    ),
    WAKING: _AGENT + (
        "that is the GPU cluster's model, which was scaled to zero while idle. "
        "Vaelor has started waking it, which can take a minute; run this agent "
        "again shortly."
    ),
    UNREADABLE: _AGENT + (
        "Vaelor could not read which model AI Chat is set to just now ({label}). "
        "Run this agent again in a moment."
    ),
    WAKE_TIMED_OUT: _AGENT + (
        "that is the GPU cluster's model, which was scaled to zero while idle. "
        "Vaelor woke it and waited {label} seconds, and it has not finished "
        "loading, so this run stopped. Check Cluster > Deployments."
    ),
    BUSY: _AGENT + (
        "AI Chat's model stayed busy with other requests for {label} seconds, "
        "so this run stopped. Run this agent again in a moment."
    ),
}


#: Why a client may not send a task key a schedule or an alert rule writes.
RESERVED_TASK_KEY = (
    "Task keys starting with automation: or trigger: are reserved for "
    "schedules and alert rules."
)


class AgentRunCancelled(RuntimeError):
    """The run was cancelled while it waited; nothing is sent to the model.

    A RuntimeError so no agent path's ValueError fallback catches it.
    """


#: Whether the current run is still wanted - the runner binds its task's
#: ``store.heartbeat``, which answers False once the task was cancelled.
_RUN_STILL_WANTED: ContextVar[Optional[Callable[[], Any]]] = ContextVar(
    "ai_chat_run_still_wanted", default=None)


def bind_run(still_wanted: Optional[Callable[[], Any]]) -> None:
    """Bind this thread's current run; the runner calls it once per run."""
    _RUN_STILL_WANTED.set(still_wanted)


class AiChatModelBusy(LocalModelBusy):
    """A custom agent's call that waited its turn on AI Chat's model and got none.

    A ``LocalModelBusy`` (a RuntimeError), deliberately not a ValueError: the
    tool loop falls back to the one-shot path on a ValueError, which would
    wait the whole bound a second time. The runner shows its sentence.
    """

#: Anthropic's native connector takes no OpenAI tool calls; the runner's
#: one-shot path answers instead (a ValueError is the tool loop's fallback).
ANTHROPIC_TAKES_NO_TOOL_CALLS = (
    "Anthropic's connector does not take tool calls on this path."
)
#: `hosted_transport.refusal_for`'s code for a 400 it cannot name otherwise.
_REQUEST_REFUSED = "hosted_request_refused"


@dataclass(frozen=True)
class AiChatModel:
    """One resolution: ``connection`` when ``state`` is READY, else ``reason``.

    ``capable`` is true when AI Chat's model is not also the Assistant's own
    (a Pi shares one CPU model between them; the Z2's GPU, the cluster and any
    connection the owner added are larger), and picks the agent loop budget.
    """

    state: str
    connection: Optional[Dict[str, Any]] = None
    reason: str = ""
    capable: bool = False
    #: The broker's own words when ``state`` is UNREADABLE.
    detail: str = ""


def _cluster_mode_active() -> bool:
    from .gpu_serving_target import gpu_cluster_mode_active

    try:
        return bool(gpu_cluster_mode_active())
    except Exception:  # noqa: BLE001 - absence-ok: a mode read never fails a caller
        return False


def _shaped(lease: Mapping[str, Any], model: str) -> Dict[str, Any]:
    if lease.get("provider") == "openai":
        return {
            **lease, "base_url": OPENAI_BASE_URL, "api_key": lease.get("token", ""),
            "model": model or lease.get("model") or OPENAI_DEFAULT_MODEL,
        }
    return {**lease, "model": model or lease.get("model", "")}


def _unreadable(error: BaseException) -> AiChatModel:
    detail = " ".join(str(error).split())[:160].rstrip(".") or type(error).__name__
    return AiChatModel(UNREADABLE, reason=AI_CHAT_MODEL_UNREADABLE.format(detail=detail),
                       detail=detail)


def resolve_ai_chat_model(
    broker: Any, *, model: str = "",
    cluster_mode_active: Optional[Callable[[], bool]] = None,
) -> AiChatModel:
    """AI Chat's model, shaped as an inference connection, or why there is none.

    ``model`` overrides the lease's own (AI Chat's per-turn pick). A lease left
    on the Assistant's NPU model reads as no model, with the VD-210 sentence;
    the row is left for the owner's next pick, never rewritten here.
    """
    if broker is None:
        return AiChatModel(NO_MODEL, reason=CHOOSE_AN_AI_CHAT_CONNECTION)
    try:
        lease = broker.resolve_active("ai-chat")
    except CredentialError as error:
        from .gpu_serving_target import no_active_credential

        if not no_active_credential(error):
            # The broker down, silent, or a lease it could not open: unknown.
            return _unreadable(error)
        if (cluster_mode_active or _cluster_mode_active)():
            return AiChatModel(CLUSTER_NOT_SERVING, reason=AI_CHAT_CLUSTER_NOT_SERVING)
        return AiChatModel(NO_MODEL, reason=CHOOSE_AN_AI_CHAT_CONNECTION)
    try:
        listed = broker.list()
    except CredentialError as error:
        return _unreadable(error)
    if not ai_chat_may_use_lease(lease, listed):
        return AiChatModel(
            ON_THE_NPU,
            reason=AI_CHAT_NEVER_ON_THE_ASSISTANT_NPU + " " + CHOOSE_AN_AI_CHAT_CONNECTION,
        )
    connection = _shaped(lease, model)
    if not connection.get("base_url"):
        return AiChatModel(INCOMPLETE, reason=AI_CHAT_CONNECTION_INCOMPLETE)
    row = next((item for item in listed or () if isinstance(item, Mapping)
                and item.get("id") == lease.get("credential_id")), None)
    shared = row is None or ASSISTANT_PURPOSE in (row.get("active_for") or [])
    return AiChatModel(READY, connection=connection, capable=not shared)


def _agent_unavailable(state: str, label: str = "") -> AiChatModel:
    return AiChatModel(state, reason=AGENT_UNAVAILABLE[state].format(label=label))


def custom_agent_model(
    broker: Any, *, cluster_mode_active: Optional[Callable[[], bool]] = None,
    cluster_wake: Optional[Callable[[], Any]] = None,
) -> AiChatModel:
    """AI Chat's model for one custom agent's run, or the sentence saying why not.

    ``cluster_wake`` is AI Chat's own seam (`ControlPlaneRuntime._cluster_wake`):
    when the lease names a cluster model scaled to zero it enqueues the warm
    load and returns its name, and the run is reported unavailable rather than
    held. Model calls inside a run resolve with no wake (``mode="agent"``).
    """
    cluster_mode_active = cluster_mode_active or _cluster_mode_active
    try:
        resolved = resolve_ai_chat_model(broker, cluster_mode_active=cluster_mode_active)
    except Exception as error:  # noqa: BLE001 - LESSONS 8: unread is "unknown", never "none"
        return _agent_unavailable(UNREADABLE, type(error).__name__)
    if resolved.state == UNREADABLE:
        return _agent_unavailable(UNREADABLE, resolved.detail)
    if resolved.connection is None:
        return _agent_unavailable(resolved.state)
    connection = resolved.connection
    label = str(connection.get("label") or "AI Chat's connection")
    if connection.get("provider") in HOSTED_KINDS and not connection.get("model"):
        # VD-206: never "the first model listed" for a hosted service.
        return _agent_unavailable(MODEL_NOT_CHOSEN, label)
    if cluster_wake is not None:
        try:
            waking = cluster_wake()
        except Exception as error:  # noqa: BLE001 - LESSONS 8: unread is "unknown"
            return _agent_unavailable(UNREADABLE, type(error).__name__)
        if waking:
            return _agent_unavailable(WAKING)
    stamped = {
        **connection, CUSTOM_AGENT_ON_AI_CHAT: True,
        AI_CHAT_CLUSTER_SLOT: bool(
            managed_local_connection(connection) and cluster_mode_active()
        ),
    }
    return AiChatModel(READY, connection=stamped, capable=resolved.capable)


#: The states a custom agent's run fails in until the owner chooses a model in
#: AI Chat. Only these hold the agent screens' New agent, Run and schedules
#: back: a cluster that is loading or was unloaded, or a read that failed, is
#: said in the runner's own sentence and blocks nothing (VD-220, kb R212).
NEEDS_AN_AI_CHAT_CHOICE = frozenset({NO_MODEL, ON_THE_NPU, INCOMPLETE, MODEL_NOT_CHOSEN})


def custom_agent_readiness(
    broker: Any, *, cluster_mode_active: Optional[Callable[[], bool]] = None,
) -> Dict[str, Any]:
    """The runner's own verdict on AI Chat's model, for the agent screens.

    The same `custom_agent_model` a run resolves, with no wake (a read must not
    start one), so the console never infers a state from the lease's shape.
    """
    chosen = custom_agent_model(broker, cluster_mode_active=cluster_mode_active)
    connection = chosen.connection or {}
    label = " · ".join(part for part in (
        str(connection.get("model") or ""), str(connection.get("label") or "")) if part)
    return {
        "state": chosen.state,
        "blocks": chosen.state in NEEDS_AN_AI_CHAT_CHOICE,
        "sentence": "" if chosen.state == READY else chosen.reason,
        "label": label,
    }


def unattended_run(task: Mapping[str, Any]) -> bool:
    """Whether a schedule or an alert rule started this run.

    Read off the task's ``idempotency_key``, whose prefix `automations` writes
    for exactly those two (a retry is keyed ``retry:`` and is hand-started).
    """
    from .automations import ALERT_TASK_KEY, SCHEDULE_TASK_KEY

    return str(task.get("idempotency_key") or "").startswith((SCHEDULE_TASK_KEY, ALERT_TASK_KEY))


def wait_for_wake(
    resolve: Callable[[], AiChatModel], *, wait: Callable[[float], Any] = time.sleep,
    heartbeat: Callable[[], Any] = lambda: None, clock: Callable[[], float] = time.monotonic,
) -> AiChatModel:
    """Re-ask ``resolve`` until the woken cluster model serves, bounded.

    For an unattended run only (VD-220, owner): each ask re-confirms the wake,
    which enqueues nothing while the load is already in flight
    (`gpu_idle_watch.wake_unloaded_cluster`). ``heartbeat`` answering False
    means the task was cancelled (CANCELLED: never sent). ``wait`` returning
    truthy means the runner is stopping (STOPPED: the task is left running
    for `recover_stale`, never reported as a timed-out wake).
    """
    deadline = clock() + WAKE_WAIT_SECONDS
    while True:
        if heartbeat() is False:
            return AiChatModel(CANCELLED)
        if clock() >= deadline:
            return _agent_unavailable(WAKE_TIMED_OUT, str(WAKE_WAIT_SECONDS))
        if wait(WAKE_POLL_SECONDS):
            return AiChatModel(STOPPED)
        resolved = resolve()
        if resolved.state != WAKING:
            return resolved


@contextmanager
def ai_chat_turn(connection: Mapping[str, Any]) -> Iterator[None]:
    """One custom agent's call holding AI Chat's own gate, waiting its turn.

    Mode B takes AI Chat's cluster permits, Mode A the endpoint's one slot, so
    an agent step and an AI Chat answer never run on the model at once. AI
    Chat refuses at once when the gate is held; this waits, bounded by
    :data:`AGENT_TURN_WAIT_SECONDS`, then raises :class:`AiChatModelBusy`. A
    run cancelled while it waited raises :class:`AgentRunCancelled` once the
    turn is won, so nothing is sent (a hosted call takes no gate but is
    checked the same way).
    """
    seconds = AGENT_TURN_WAIT_SECONDS
    gate = (cluster_inference_slot(wait_seconds=seconds) if connection.get(AI_CHAT_CLUSTER_SLOT)
            else local_inference_slot(dict(connection), wait_seconds=seconds))
    with ExitStack() as stack:
        try:
            stack.enter_context(gate)
        except LocalModelBusy:
            raise AiChatModelBusy(AGENT_UNAVAILABLE[BUSY].format(label=seconds)) from None
        still_wanted = _RUN_STILL_WANTED.get()
        if still_wanted is not None and still_wanted() is False:
            raise AgentRunCancelled("The run was cancelled; nothing was sent to AI Chat's model.")
        yield


def off_machine_agent_body(connection: Mapping[str, Any], body: Dict[str, Any]) -> Dict[str, Any]:
    """A custom agent's request as an off-machine model takes it.

    The agent paths size answers for a small local model; a hosted model often
    reasons inside the same limit, so it gets AI Chat's hosted ceiling (a
    ceiling, not a spend). OpenAI takes ``max_completion_tokens`` and no
    temperature, as AI Chat sends it. The local-only template switch is dropped.
    """
    shaped = {key: value for key, value in body.items() if key != "chat_template_kwargs"}
    ceiling = max(int(shaped.pop("max_tokens", 0) or 0), HOSTED_ANSWER_TOKENS)
    if connection.get("provider") == "openai":
        shaped.pop("temperature", None)
        shaped["max_completion_tokens"] = ceiling
    else:
        shaped["max_tokens"] = ceiling
    return shaped


def hosted_agent_completion(
    connection: Mapping[str, Any], request_body: Mapping[str, Any], timeout: float,
    send: Any = None,
) -> Dict[str, Any]:
    """One custom agent's chat completion against a hosted AI Chat model.

    Sent by `chat_hosted.answer_hosted`, so it goes out through
    `hosted_transport` (public addresses only, pinned, no redirects) exactly as
    an AI Chat answer does. The body comes back OpenAI-shaped. Any refusal is a
    ``ValueError`` whose sentence names the service, which every agent caller
    already turns into its own degraded or failed result.
    """
    from .chat_hosted import HostedChatError, answer_hosted

    if not connection.get(CUSTOM_AGENT_ON_AI_CHAT):
        raise ValueError("A hosted connection is sent only a custom agent's run.")
    turns = [dict(item) for item in request_body.get("messages") or () if isinstance(item, Mapping)]
    system = "\n\n".join(str(item.get("content") or "") for item in turns if item.get("role") == "system")
    turns = [item for item in turns if item.get("role") != "system"]
    engine = HOSTED_PROVIDERS[str(connection.get("provider"))]["engine"]
    if engine == "anthropic-messages" and (
        request_body.get("tools") or any(item.get("role") == "tool" for item in turns)
    ):
        raise ValueError(ANTHROPIC_TAKES_NO_TOOL_CALLS)
    question = str(turns[-1].get("content") or "") if turns else ""
    payload = off_machine_agent_body(connection, dict(request_body))

    def ask(body: Dict[str, Any]) -> Dict[str, Any]:
        return answer_hosted(
            connection, payload=body, system=system, history=turns[:-1],
            question=question, sources=(), max_tokens=body["max_tokens"],
            timeout=timeout, send=send,
        )[1]

    try:
        try:
            return ask(payload)
        except HostedChatError as error:
            if error.code != _REQUEST_REFUSED or "response_format" not in payload:
                raise
            # The service refused the structured-output contract; the prompt
            # still asks for JSON, which the agent readers parse either way.
            return ask({key: value for key, value in payload.items() if key != "response_format"})
    except HostedChatError as error:
        raise ValueError(str(error)) from None
