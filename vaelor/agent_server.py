"""The loopback OpenAI-compatible runtime for one deployed cluster Agent.

F4b-ii-A, the security-critical core: this turns a stored agent record (already
resolved and injected as a 0600 config file) into a live, loopback-bound,
OpenAI-shaped endpoint whose every tool call passes the :mod:`agent_runtime_gate`
boundary. It reads ONLY its config file - it opens no broker socket, holds no
database, and reaches no node - so the whole trust surface is the config and the
:class:`agent_runtime_gate.AgentToolset` it constructs.

Three startup gates refuse an unsafe agent before it ever serves:

* :func:`agent_runtime_gate.assert_read_only` refuses to start a mutating agent
  (this first cut holds NO acting permission at all, design C6).
* :func:`inference_client.allowed_inference_endpoint` validates the injected
  model ``base_url`` and REFUSES to start if it fails - the anti-relay gate
  (design C3), because the bridge socket is group-vaelor and an in-group caller
  could otherwise inject an arbitrary endpoint and make the agent a
  LAN-to-anywhere relay.
* the toolset is built from the injected effective ``(server, tool)`` pairs and
  the granted read-only scopes, so the model can only ever see what was granted.

The request handling is a pure function of its input and the injected
collaborators (:meth:`AgentRuntime.chat_completions` / :meth:`AgentRuntime.health`),
so a test drives the whole surface without binding a socket. :func:`main` is a
thin bind around it.

Degrade is honest and never hangs: a model that cannot be reached answers HTTP
503 with a reason, a busy single-flight local slot answers HTTP 429, and neither
ever puts the model API key, the config, or a raw body into the response or a
log line. Only unauthenticated MCP servers are supported in this cut (design
C5): :class:`mcp_client.ExternalMcpTools` sends no Authorization header, so a
``servers[].endpoint`` is dialled as it stands.
"""

from __future__ import annotations

import json
import logging
import os
import ssl
import threading
import time
import urllib.error
import urllib.parse
import urllib.request
import uuid
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional, Tuple

from .agent_wake import ModelProbe, WakeClient, build_wake_client, wake_answer
from .agent_runtime_gate import (
    AgentRuntimeGateError,
    AgentToolRegistry,
    AgentToolset,
    McpGrantGate,
    assert_read_only,
)
from .assistant_tools import AssistantToolError, AssistantToolRegistry
from .inference_client import (
    allowed_inference_endpoint,
    chat_completion,
    inference_timeout,
)
from .local_inference_gate import LocalModelBusy
from .mcp_client import ExternalMcpTools, McpClientError
from .session_affinity import (
    SESSION_HEADER, new_session_key, only_header_value, opening_session_key,
    session_headers, valid_session_key,
)

LOGGER = logging.getLogger(__name__)

#: The environment variable naming the 0600 config file the launch unit writes.
CONFIG_ENV = "VAELOR_AGENT_CONFIG"

#: Bounds the agent's own chat loop, independent of what the model asks for: at
#: most this many model round trips, this many tool calls honoured per round, and
#: this much of each tool result kept for the next turn. Distinct from the
#: research loop's caps because this is a general chat agent, not a synthesiser.
MAX_TURNS = 6
MAX_CALLS_PER_TURN = 5
MAX_TOOL_RESPONSE_CHARS = 6000
MAX_REQUEST_MESSAGES = 40
MAX_BODY_BYTES = 256 * 1024
DEFAULT_TIMEOUT_SECONDS = 240
#: A slow or dribbled request body must not tie a handler thread up for good
#: (ThreadingHTTPServer caps no threads). This per-connection socket read
#: deadline bounds it; the model call runs on a separate outbound socket, so a
#: long completion is unaffected.
REQUEST_SOCKET_TIMEOUT_SECONDS = 30

#: The two honest-degrade reasons a caller reads back. One home each so the words
#: the test asserts and the words the route emits cannot drift apart.
MODEL_UNREACHABLE_REASON = "The backing model is unreachable right now."
#: A model that is up but answered 503: busy, not gone.
MODEL_BUSY_REASON = "The agent's model is busy with other requests; retry shortly."
MODEL_BUSY_RETRY_SECONDS = 5
AGENT_BUSY_REASON = "The agent is busy with another request; retry shortly."
#: When the model was reached and replied with an HTTP error status it
#: REJECTED the request; kept distinct from unreachable so the two never blur.
MODEL_REJECTED_REASON = (
    "The backing model was reached but rejected the request (HTTP {})."
)

#: Fixed answers for the routing edges, each written once and reused.
_NO_SUCH_ROUTE = "This agent server serves no such route."
_BAD_JSON = "The request body could not be read as JSON."
_BAD_MESSAGES = "A chat request needs a non-empty list of messages."
_TOO_MANY_MESSAGES = "The chat request carries more messages than are accepted."
_BAD_BODY_SIZE = "The request body size is outside the accepted bound."
_MODEL_BAD_BODY = "The backing model returned a response that could not be used."

#: Agent memory (F5b). The recent-turn recollection injected before the model
#: sees a request, and the bounded summary written after it answers. Every value
#: is small and every call is best-effort, so a memory subsystem that is slow or
#: down changes an answer's latency negligibly and its content not at all.
MEMORY_RECALL_LIMIT = 8
MEMORY_RECALL_TIMEOUT_SECONDS = 4
MEMORY_RECORD_TIMEOUT_SECONDS = 6
MEMORY_SUMMARY_FIELD_CHARS = 1024
MEMORY_RECALL_LINE_CHARS = 240
MEMORY_BLOCK_CHARS = 2000
MAX_MEMORY_RESPONSE_BYTES = 256 * 1024

#: Written once each so the recollection heading, the one disabled-on-bad-block
#: note and the incomplete-call debug line cannot drift between their sites.
_MEMORY_BLOCK_HEADING = "Recent memory from earlier turns (most recent first):"
_MEMORY_DISABLED_ON_BAD_BLOCK = "agent memory config was unusable; memory left off"
_MEMORY_CALL_FAILED = "agent memory %s call failed: %s"


def _memory_failure_reason(error: BaseException) -> str:
    """A plain, secret-free reason for a failed memory call.

    Names what went wrong - the control plane's HTTP answer, a certificate that
    did not match the pinned one, or an unreachable host - and never the URL,
    the bearer token or a body. This is what reaches the journal and ``/health``
    so a memory that never works is visible, not swallowed at DEBUG (ACC-132).
    """
    if isinstance(error, urllib.error.HTTPError):
        return "the control plane answered HTTP {}".format(error.code)
    reason = getattr(error, "reason", None)
    if isinstance(error, ssl.SSLError) or isinstance(reason, ssl.SSLError):
        return "the control plane's certificate did not match the pinned one"
    if isinstance(error, urllib.error.URLError):
        return "the control plane could not be reached ({})".format(
            type(reason).__name__ if isinstance(reason, BaseException) else "no route"
        )
    if isinstance(error, ValueError):
        return "the control plane's reply could not be read"
    return "the call raised {}".format(type(error).__name__)


def pinned_certificate_context(ca_pem: str) -> ssl.SSLContext:
    """A TLS client context that trusts exactly one certificate: the control plane's.

    The deploy hands the agent the controller's own serving certificate (leaf
    only). Trusting only that certificate - no system roots - is an exact pin:
    no other key can complete the handshake, so a hostname comparison adds
    nothing and is switched off. That keeps memory working when the advertise
    address is not one of the names the certificate was minted for.
    ``VERIFY_X509_PARTIAL_CHAIN`` lets a CA-issued leaf be the trust anchor too.
    """
    context = ssl.create_default_context(cadata=ca_pem)
    context.check_hostname = False
    context.verify_mode = ssl.CERT_REQUIRED
    context.verify_flags |= getattr(ssl, "VERIFY_X509_PARTIAL_CHAIN", 0)
    return context


class AgentServerConfigError(ValueError):
    """A safe, presentable error raised when the injected agent config is unusable."""


def _error_body(code: str, reason: str) -> Dict[str, Any]:
    """One OpenAI-style error envelope, carrying a code and a reason only."""
    return {"error": {"type": str(code), "message": str(reason)}}


def _require_mapping(value: Any, field: str) -> Dict[str, Any]:
    if not isinstance(value, Mapping):
        raise AgentServerConfigError(
            "The agent config section named {} must be an object.".format(field)
        )
    return dict(value)


def _require_text(value: Any, field: str) -> str:
    text = str(value or "").strip()
    if not text:
        raise AgentServerConfigError(
            "The agent config value named {} is required.".format(field)
        )
    return text


def _is_loopback(host: str) -> bool:
    """Whether a listen host stays on this machine's own loopback."""
    name = str(host or "").strip().lower().strip("[]")
    return name == "localhost" or name == "127.0.0.1" or name == "::1"


def parse_tool_calls_unclamped(message: Any) -> List[Dict[str, Any]]:
    """Read ``message.tool_calls`` WITHOUT the 64-character name clamp (design C2).

    A qualified ``mcp.<server>.<tool>`` name runs well past sixty-four
    characters, and the research loop's parser truncates there - which would
    leave a granted MCP tool silently uncallable. This parser keeps the full
    name; everything else mirrors the tolerant reading (arguments as a JSON
    string, an object, or absent; an empty name dropped).
    """
    if not isinstance(message, Mapping):
        return []
    raw = message.get("tool_calls")
    if not isinstance(raw, list):
        return []
    calls: List[Dict[str, Any]] = []
    for item in raw:
        if not isinstance(item, Mapping):
            continue
        function = item.get("function")
        function = function if isinstance(function, Mapping) else {}
        name = str(function.get("name") or "").strip()
        if not name:
            continue
        arguments = function.get("arguments")
        if isinstance(arguments, Mapping):
            parsed: Any = dict(arguments)
        elif isinstance(arguments, str) and arguments.strip():
            try:
                parsed = json.loads(arguments)
            except (ValueError, TypeError):
                parsed = {}
        else:
            parsed = {}
        calls.append({
            "id": str(item.get("id") or "")[:128] or "call_{}".format(len(calls)),
            "name": name,
            "arguments": parsed if isinstance(parsed, dict) else {},
        })
    return calls


def _final_message(body: Any) -> Dict[str, Any]:
    """The assistant message from a completion body, or a refusal if unusable."""
    if not isinstance(body, Mapping):
        raise ValueError(_MODEL_BAD_BODY)
    choices = body.get("choices")
    first = choices[0] if isinstance(choices, list) and choices else {}
    message = first.get("message") if isinstance(first, Mapping) else {}
    return dict(message) if isinstance(message, Mapping) else {}


def _spawn_record_thread(target: Any) -> None:
    """Run a best-effort memory record off the request thread so it never delays it."""
    threading.Thread(target=target, daemon=True).start()


def _memory_endpoint_ok(endpoint: str) -> bool:
    """Whether a memory endpoint is a usable http(s) URL carrying a host.

    The 0600 config is minted by the control plane, yet a partial or malformed
    block must never crash the runtime - it disables memory instead. This is the
    lightweight guard the design permits in place of the model endpoint's
    private-range check: the endpoint is the controller's own LAN address,
    reached across the same trust boundary as the model base_url, so a
    scheme-and-host check is the bound that matters here.
    """
    parts = urllib.parse.urlsplit(endpoint)
    return parts.scheme in ("http", "https") and bool(parts.netloc)


def _last_user_message(request_messages: List[Any]) -> str:
    """The most recent user turn's content as text, or an empty string."""
    for item in reversed(list(request_messages)):
        if isinstance(item, Mapping) and item.get("role") == "user":
            content = item.get("content")
            return content if isinstance(content, str) else str(content or "")
    return ""


def _format_memory_block(records: List[Dict[str, Any]]) -> str:
    """A compact, bounded recollection for the system message, or an empty string.

    Only the ``request``/``answer`` fields a record write stores are surfaced,
    each trimmed to one short line and the whole block capped, so a large or
    hostile row can neither bloat the prompt nor break its shape.
    """
    lines: List[str] = []
    used = len(_MEMORY_BLOCK_HEADING)
    for record in records:
        if not isinstance(record, Mapping):
            continue
        asked = str(record.get("request") or "").strip()[:MEMORY_RECALL_LINE_CHARS]
        answered = str(record.get("answer") or "").strip()[:MEMORY_RECALL_LINE_CHARS]
        if not asked and not answered:
            continue
        line = "- asked {!r}, answered {!r}".format(asked, answered)
        if used + len(line) + 1 > MEMORY_BLOCK_CHARS:
            break
        lines.append(line)
        used += len(line) + 1
    if not lines:
        return ""
    return _MEMORY_BLOCK_HEADING + "\n" + "\n".join(lines)


class AgentMemoryClient:
    """A bounded, best-effort bearer client for this agent's own memory endpoint.

    It POSTs JSON to the control-plane recall/record routes with the per-agent
    ``vak_`` token as a bearer, on a short timeout with at most one retry, never
    lets a failure raise into the chat, and never writes the token, the URL or a
    body to a log line. It is the ONE new capability a deployed agent gains
    (design D1): the agent still opens no broker socket and holds no database.

    A failed call is logged at WARNING with a plain reason and remembered, so
    :meth:`status` (surfaced on ``/health``) says whether memory is working.
    """

    def __init__(
        self, endpoint: str, token: str, *,
        opener: Any = None,
        ssl_context: Optional[ssl.SSLContext] = None,
        recall_timeout: int = MEMORY_RECALL_TIMEOUT_SECONDS,
        record_timeout: int = MEMORY_RECORD_TIMEOUT_SECONDS,
    ):
        base = endpoint.rstrip("/")
        self._recall_url = base + "/recall"
        self._record_url = base + "/record"
        self._token = token
        self._opener = opener or urllib.request.urlopen
        self._ssl_context = ssl_context
        self._recall_timeout = recall_timeout
        self._record_timeout = record_timeout
        self._outcome: Dict[str, Any] = {"state": "not_used_yet"}

    def status(self) -> Dict[str, Any]:
        """The last call's outcome: ``not_used_yet``, ``working`` or ``failing``."""
        return dict(self._outcome)

    def recall(self, limit: int) -> List[Dict[str, Any]]:
        """The recent memory rows for this agent, or an empty list on any trouble."""
        body = self._post(
            "recall", self._recall_url, {"limit": int(limit)},
            timeout=self._recall_timeout, retries=0,
        )
        # The control plane answers in its API envelope, ``{"ok": true, "data":
        # {"agent": ..., "records": [...]}}``. Reading ``records`` off the top
        # level (as this did) found nothing on every call, even a working one.
        data = body.get("data") if isinstance(body, Mapping) else None
        rows = data.get("records") if isinstance(data, Mapping) else None
        if not isinstance(rows, list):
            return []
        return [row for row in rows if isinstance(row, Mapping)]

    def record(self, request_text: str, answer_text: str) -> None:
        """Append one bounded interaction summary, swallowing every failure."""
        summary = {"request": request_text, "answer": answer_text}
        self._post(
            "record", self._record_url, {"records": [summary]},
            timeout=self._record_timeout, retries=1,
        )

    def _post(
        self, operation: str, url: str, payload: Dict[str, Any], *,
        timeout: int, retries: int,
    ) -> Optional[Dict[str, Any]]:
        data = json.dumps(payload).encode("utf-8")
        headers = {
            "Content-Type": "application/json",
            "Accept": "application/json",
            "Authorization": "Bearer {}".format(self._token),
        }
        extra = {} if self._ssl_context is None else {"context": self._ssl_context}
        for attempt in range(retries + 1):
            try:
                request = urllib.request.Request(
                    url, data=data, headers=headers, method="POST",
                )
                with self._opener(request, timeout=timeout, **extra) as response:
                    content = response.read(MAX_MEMORY_RESPONSE_BYTES + 1)
                if len(content) > MAX_MEMORY_RESPONSE_BYTES:
                    raise ValueError("oversized memory reply")
                body = json.loads(content.decode("utf-8"))
                self._outcome = {"state": "working", "at": time.time()}
                return body
            except Exception as error:  # noqa: BLE001 - memory is best-effort, never raise into a chat
                if attempt >= retries:
                    reason = _memory_failure_reason(error)
                    self._outcome = {
                        "state": "failing", "operation": operation,
                        "reason": reason, "at": time.time(),
                    }
                    LOGGER.warning(_MEMORY_CALL_FAILED, operation, reason)
                    return None
        return None


def build_memory_client(section: Any) -> Optional[AgentMemoryClient]:
    """The memory client an optional ``config['memory']`` block asks for, or None.

    A missing block leaves memory off with no fuss; a present block that cannot
    be used disables memory and logs ONCE, and never with the bearer token.
    """
    if not isinstance(section, Mapping) or not section:
        return None
    endpoint = str(section.get("endpoint") or "").strip()
    token = str(section.get("token") or "").strip()
    if not endpoint or not token or not _memory_endpoint_ok(endpoint):
        LOGGER.warning(_MEMORY_DISABLED_ON_BAD_BLOCK)
        return None
    context = None
    ca_pem = str(section.get("ca_pem") or "").strip()
    if ca_pem and urllib.parse.urlsplit(endpoint).scheme == "https":
        try:
            context = pinned_certificate_context(ca_pem)
        except (ssl.SSLError, ValueError):
            LOGGER.warning(_MEMORY_DISABLED_ON_BAD_BLOCK)
            return None
    return AgentMemoryClient(endpoint, token, ssl_context=context)


class AgentRuntime:
    """The in-process request handler for exactly one deployed agent.

    Built from the validated config's collaborators (or from injected fakes in a
    test). It never binds a socket - :func:`main` does that around it - and its
    two request methods depend only on their argument and the injected
    collaborators, so the whole OpenAI surface is testable in process.
    """

    def __init__(
        self,
        *,
        instructions: str,
        connection: Dict[str, str],
        toolset: AgentToolset,
        catalog: Any,
        web_access: Optional[Dict[str, Any]],
        actor: Optional[str],
        chat_fn: Any = chat_completion,
        max_turns: int = MAX_TURNS,
        memory: Optional[AgentMemoryClient] = None,
        record_runner: Any = None,
        model_probe: Optional[ModelProbe] = None,
        wake: Optional[WakeClient] = None,
    ):
        # ACC-072 / the wake decision: /health says whether the model answers,
        # and an unreachable model asks the control plane to wake it.
        self._model_probe = model_probe
        self._wake = wake
        self._instructions = str(instructions or "")
        self._connection = dict(connection)
        self._toolset = toolset
        self._catalog = catalog
        self._web_access = web_access
        self._actor = actor
        self._chat_fn = chat_fn
        self._max_turns = max(1, int(max_turns))
        self._model = str(self._connection.get("model") or "")
        self._memory = memory
        self._record_runner = record_runner or _spawn_record_thread

    # -- request surface --------------------------------------------------
    def health(self) -> Tuple[int, Dict[str, Any]]:
        """The unauthenticated liveness probe the front gate calls.

        Liveness stays ``ok`` (the runtime is up and the reconcile must not
        restart it for a model or memory outage). ``model`` says whether the
        backing model answered its last probe (ACC-072), so the console never
        reads Serving over a model that is gone; the memory outcome rides along
        (``off`` when the config carries no memory block) so a memory that
        fails is visible from outside (ACC-132).
        """
        model = self._model_probe.status() if self._model_probe is not None else None
        memory = {"state": "off"} if self._memory is None else self._memory.status()
        return 200, {"status": "ok", "model": model, "memory": memory}

    def models(self) -> Tuple[int, Dict[str, Any]]:
        """``GET /v1/models``: the one model this agent answers as (ACC-073).

        OpenAI-style clients list models before they chat; the agent answers as
        its backing model's id, the same ``model`` its completions carry.
        """
        return 200, {
            "object": "list",
            "data": [{"id": self._model, "object": "model", "owned_by": "vaelor-agent"}],
        }

    def chat_completions(
        self, request_body: Any, *, session_key: str = "",
    ) -> Tuple[int, Dict[str, Any]]:
        """Handle one ``POST /v1/chat/completions`` as ``(status, json body)``.

        Never raises for an expected condition: a model reached but replying
        with an HTTP error status is a 502, a truly unreachable model a 503, a
        busy local slot a 429, and no branch ever writes the model API key or a
        raw provider body into the answer.

        ``session_key`` is the client's ``X-Session-Id`` (VD-157). Valid, it
        names the conversation on every model call this request makes. Absent
        or invalid, the key is derived from this agent and the conversation's
        first user message (VD-158), so the tool loop of this request and the
        later requests of the same chat all find their prompt on the replica
        that already holds it; with no user message either, one fresh key
        still holds this request's loop together.
        """
        if not isinstance(request_body, Mapping):
            return 400, _error_body("invalid_request", _BAD_JSON)
        request_messages = request_body.get("messages")
        if not isinstance(request_messages, list) or not request_messages:
            return 400, _error_body("invalid_request", _BAD_MESSAGES)
        if len(request_messages) > MAX_REQUEST_MESSAGES:
            return 400, _error_body("invalid_request", _TOO_MANY_MESSAGES)
        try:
            message = self._run_chat(
                request_messages,
                valid_session_key(session_key)
                or opening_session_key(self._actor, request_messages)
                or new_session_key(),
            )
        except LocalModelBusy:
            return 429, _error_body("agent_busy", AGENT_BUSY_REASON)
        except urllib.error.HTTPError as error:
            # HTTPError is a URLError SUBCLASS, so it must be caught first. The
            # model was reached and answered with an error status: it rejected
            # the request, it was NOT unreachable. Log the exception type and
            # the numeric status only - the provider body may quote the request
            # or the key, so it never reaches a log line or the answer.
            status = int(getattr(error, "code", 0) or 0)
            LOGGER.warning(
                "agent completion rejected by the model: %s HTTP %s",
                type(error).__name__, status,
            )
            if status in (502, 503, 504):
                return self._unreachable(status)
            return 502, _error_body("model_rejected", MODEL_REJECTED_REASON.format(status))
        except (OSError, urllib.error.URLError):
            return self._unreachable()
        except Exception as error:  # noqa: BLE001 - never hang, never leak the key
            # Log the exception TYPE only: a message or traceback could quote a
            # request fragment, and the key must never reach a log line.
            LOGGER.warning("agent completion failed with %s", type(error).__name__)
            return 502, _error_body("model_error", _MODEL_BAD_BODY)
        return 200, self._envelope(message)

    def _unreachable(self, http_status: int = 0) -> Tuple[int, Dict[str, Any]]:
        """503 for a model that did not answer, asking for a wake first.

        A model scaled to zero after sitting idle is woken by the control plane
        and the client is told to retry; one unloaded by hand is named as such.
        A model that is up and answered 503 is busy, and says so; anything else
        keeps the plain unreachable reason.
        """
        answer = wake_answer(self._wake.ask() if self._wake is not None else None)
        if answer is not None:
            return 503, answer
        if http_status == 503:
            body = _error_body("model_busy", MODEL_BUSY_REASON)
            body["error"]["retry_after"] = MODEL_BUSY_RETRY_SECONDS
            return 503, body
        return 503, _error_body("model_unreachable", MODEL_UNREACHABLE_REASON)

    # -- the bounded chat loop -------------------------------------------
    def _run_chat(
        self, request_messages: List[Any], session_key: str = "",
    ) -> Dict[str, Any]:
        messages: List[Dict[str, Any]] = [
            {"role": "system", "content": self._system_content()}
        ]
        messages.extend(dict(item) for item in request_messages if isinstance(item, Mapping))
        timeout = inference_timeout(self._connection, DEFAULT_TIMEOUT_SECONDS)
        headers = {**self._headers(), **session_headers(session_key)}
        last_message: Dict[str, Any] = {}
        final_message: Dict[str, Any] = {}
        for _turn in range(self._max_turns):
            specs = self._toolset.specifications(self._catalog, self._web_access)
            request: Dict[str, Any] = {"model": self._model, "messages": messages}
            if specs:
                request["tools"] = specs
                request["tool_choice"] = "auto"
            body = self._chat_fn(self._connection, request, headers, timeout)
            message = _final_message(body)
            last_message = message
            calls = parse_tool_calls_unclamped(message)
            if not calls:
                final_message = message
                break
            honoured = calls[:MAX_CALLS_PER_TURN]
            messages.append({
                "role": "assistant",
                "content": message.get("content") or "",
                "tool_calls": [
                    {"id": call["id"], "type": "function",
                     "function": {"name": call["name"],
                                  "arguments": json.dumps(call["arguments"])}}
                    for call in honoured
                ],
            })
            for call in honoured:
                messages.append({
                    "role": "tool",
                    "tool_call_id": call["id"],
                    "name": call["name"],
                    "content": self._dispatch(call),
                })
        else:
            final_message = last_message
        self._record_turn(request_messages, final_message)
        return final_message

    def _system_content(self) -> str:
        """The system instructions, with a recalled-memory block appended if any."""
        if self._memory is None:
            return self._instructions
        block = _format_memory_block(self._recall())
        if not block:
            return self._instructions
        return "{}\n\n{}".format(self._instructions.rstrip(), block)

    def _recall(self) -> List[Dict[str, Any]]:
        """Best-effort recall; on any trouble return nothing so the chat proceeds."""
        try:
            return self._memory.recall(MEMORY_RECALL_LIMIT)
        except Exception:  # noqa: BLE001 - recall never blocks a chat from answering
            return []

    def _record_turn(
        self, request_messages: List[Any], final_message: Mapping[str, Any],
    ) -> None:
        """Fire a bounded record of this turn off-thread; never affect the answer."""
        if self._memory is None:
            return
        request_text = _last_user_message(request_messages)[:MEMORY_SUMMARY_FIELD_CHARS]
        answer_text = ""
        if isinstance(final_message, Mapping):
            answer_text = str(final_message.get("content") or "")[:MEMORY_SUMMARY_FIELD_CHARS]
        if not request_text and not answer_text:
            return

        def _write() -> None:
            try:
                self._memory.record(request_text, answer_text)
            except Exception as error:  # noqa: BLE001 - a record never affects the served answer
                LOGGER.warning(_MEMORY_CALL_FAILED, "record", _memory_failure_reason(error))

        self._record_runner(_write)

    def _dispatch(self, call: Dict[str, Any]) -> str:
        """Run one gated tool call, returning a bounded ``tool`` message body.

        A denial (guard, approval, or an ungranted/mutating Vaelor tool) is
        returned to the model as an error string rather than crashing the loop -
        the tool did NOT run, which is the security guarantee, and the model may
        recover on its next turn.
        """
        try:
            outcome = self._toolset.dispatch(call["name"], call["arguments"], self._actor)
        except (AgentRuntimeGateError, McpClientError, AssistantToolError) as error:
            return json.dumps({"error": str(error)[:300]})
        result = outcome.get("result") if isinstance(outcome, Mapping) else outcome
        text = json.dumps(result, default=str, separators=(",", ":"))
        if len(text) > MAX_TOOL_RESPONSE_CHARS:
            return text[:MAX_TOOL_RESPONSE_CHARS] + "...[truncated]"
        return text

    def _envelope(self, message: Mapping[str, Any]) -> Dict[str, Any]:
        """A clean OpenAI ``chat.completion`` from the final message.

        Built fresh rather than passed through, so the ``performance`` timing key
        ``chat_completion`` injects onto the provider body (design C4) is never
        echoed to the caller.
        """
        content = message.get("content") if isinstance(message, Mapping) else ""
        return {
            "id": "chatcmpl-{}".format(uuid.uuid4().hex),
            "object": "chat.completion",
            "created": int(time.time()),
            "model": self._model,
            "choices": [{
                "index": 0,
                "message": {"role": "assistant", "content": content or ""},
                "finish_reason": "stop",
            }],
        }

    def _headers(self) -> Dict[str, str]:
        headers = {"Content-Type": "application/json", "Accept": "application/json"}
        key = self._connection.get("api_key")
        if key:
            headers["Authorization"] = "Bearer {}".format(key)
        return headers


def _effective_pairs(value: Any) -> List[Tuple[str, str]]:
    pairs: List[Tuple[str, str]] = []
    for item in value or []:
        if isinstance(item, (list, tuple)) and len(item) == 2:
            pairs.append((str(item[0]), str(item[1])))
    return pairs


def _mcp_servers(value: Any, effective_pairs: List[Tuple[str, str]]) -> List[Dict[str, Any]]:
    """Shape the injected servers for the outbound client.

    The gate is the authority on what may be called; the client's own
    ``enabled``/``approved_tools`` are belt-and-suspenders, so they are seeded
    from the effective pairs. Only unauthenticated servers are supported here
    (design C5), so no credential is attached - the endpoint is dialled as-is.
    """
    approved: Dict[str, List[str]] = {}
    for server_name, tool in effective_pairs:
        approved.setdefault(server_name, []).append(tool)
    servers: List[Dict[str, Any]] = []
    for item in value or []:
        if not isinstance(item, Mapping):
            continue
        name = str(item.get("name") or "")
        servers.append({
            "name": name,
            "endpoint": item.get("endpoint"),
            "enabled": True,
            "approved_tools": approved.get(name, []),
        })
    return servers


def build_runtime(config: Mapping[str, Any], *, chat_fn: Any = chat_completion) -> AgentRuntime:
    """Validate the config and assemble the runtime, applying the startup gates.

    Raises :class:`AgentServerConfigError` for a mutating agent (design C6) or a
    ``base_url`` the inference client would not allow (design C3), before any
    collaborator is built.
    """
    agent = _require_mapping(config.get("custom_agent"), "custom_agent")
    model = _require_mapping(config.get("model"), "model")
    mcp = _require_mapping(config.get("mcp"), "mcp") if config.get("mcp") is not None else {}

    # Startup gate one: a deployed agent offers only read-only tools.
    try:
        assert_read_only(agent.get("permissions") or [])
    except AgentRuntimeGateError as error:
        raise AgentServerConfigError(str(error)) from error

    connection = {
        "base_url": _require_text(model.get("base_url"), "model.base_url"),
        "model": _require_text(model.get("model"), "model.model"),
        "api_key": str(model.get("api_key") or ""),
        "provider": "openai-compatible",
    }
    # Startup gate two: the anti-relay endpoint check.
    if not allowed_inference_endpoint(connection):
        raise AgentServerConfigError(
            "The injected model base_url is not an allowed inference endpoint."
        )

    scopes = [str(item) for item in (agent.get("scopes") or [])]
    raw_web = agent.get("web_access")
    web_access = dict(raw_web) if isinstance(raw_web, Mapping) else None

    inner = AssistantToolRegistry()
    vaelor_registry = AgentToolRegistry(inner, scopes)
    effective_pairs = _effective_pairs(mcp.get("effective_pairs"))
    gate = McpGrantGate.from_pairs(effective_pairs)
    mcp_client = ExternalMcpTools(
        servers=_mcp_servers(mcp.get("servers"), effective_pairs),
        approval=gate.approval,
    )
    specs = mcp.get("specs") if isinstance(mcp.get("specs"), list) else []
    toolset = AgentToolset(vaelor_registry, gate, mcp_client, specs, web_access=web_access)

    return AgentRuntime(
        instructions=str(agent.get("instructions") or ""),
        connection=connection,
        toolset=toolset,
        catalog=inner.catalog(),
        web_access=web_access,
        actor=(str(agent.get("id") or "") or None),
        chat_fn=chat_fn,
        memory=build_memory_client(config.get("memory")),
        model_probe=ModelProbe(connection["base_url"], connection["api_key"]),
        wake=build_wake_client(config.get("wake")),
    )


def load_config(path: Optional[str] = None) -> Dict[str, Any]:
    """Read and JSON-parse the 0600 config file named by :data:`CONFIG_ENV`."""
    location = path or os.environ.get(CONFIG_ENV, "")
    if not location:
        raise AgentServerConfigError("No agent config file path was supplied.")
    try:
        raw = json.loads(Path(location).read_text(encoding="utf-8"))
    except (OSError, ValueError) as error:
        raise AgentServerConfigError("The agent config file could not be read.") from error
    if not isinstance(raw, Mapping):
        raise AgentServerConfigError("The agent config file must hold a JSON object.")
    return dict(raw)


class _Handler(BaseHTTPRequestHandler):
    """The thin stdlib HTTP surface; every decision lives in :class:`AgentRuntime`."""

    server_version = "vaelor-agent/1"
    #: Bound each connection's socket reads so a slow-body client frees its thread.
    timeout = REQUEST_SOCKET_TIMEOUT_SECONDS

    def log_message(self, *_args: Any) -> None:
        """Silence the default logger; :meth:`_write` logs method+path+status."""

    def _write(self, status: int, body: Dict[str, Any]) -> None:
        payload = json.dumps(body).encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", "application/json")
        error = body.get("error") if isinstance(body, dict) else None
        if isinstance(error, dict) and error.get("retry_after"):
            self.send_header("Retry-After", str(int(error["retry_after"])))
        self.send_header("Content-Length", str(len(payload)))
        self.end_headers()
        self.wfile.write(payload)
        LOGGER.info("agent %s %s %s", self.command, self.path.split("?")[0], status)

    def do_GET(self) -> None:  # noqa: N802 - stdlib handler name
        if self.path.split("?")[0] == "/health":
            self._write(*self.server.runtime.health())  # type: ignore[attr-defined]
            return
        if self.path.split("?")[0] == "/v1/models":
            self._write(*self.server.runtime.models())  # type: ignore[attr-defined]
            return
        self._write(404, _error_body("not_found", _NO_SUCH_ROUTE))

    def do_POST(self) -> None:  # noqa: N802 - stdlib handler name
        if self.path.split("?")[0] != "/v1/chat/completions":
            self._write(404, _error_body("not_found", _NO_SUCH_ROUTE))
            return
        try:
            length = int(self.headers.get("Content-Length") or 0)
        except (TypeError, ValueError):
            length = 0
        if length <= 0 or length > MAX_BODY_BYTES:
            self._write(400, _error_body("invalid_request", _BAD_BODY_SIZE))
            return
        try:
            request_body = json.loads(self.rfile.read(length).decode("utf-8"))
        except (ValueError, UnicodeDecodeError):
            self._write(400, _error_body("invalid_request", _BAD_JSON))
            return
        self._write(*self.server.runtime.chat_completions(  # type: ignore[attr-defined]
            request_body,
            session_key=only_header_value(self.headers.get_all(SESSION_HEADER)),
        ))


def main() -> None:
    logging.basicConfig(level=logging.INFO)
    config = load_config()
    runtime = build_runtime(config)
    listen = config.get("listen") if isinstance(config.get("listen"), Mapping) else {}
    host = str(listen.get("host") or "127.0.0.1")
    if not _is_loopback(host):
        raise AgentServerConfigError("The agent server binds a loopback host only.")
    port = int(listen.get("port") or 0)
    server = ThreadingHTTPServer((host, port), _Handler)
    server.runtime = runtime  # type: ignore[attr-defined]
    try:
        server.serve_forever()
    finally:
        server.server_close()


if __name__ == "__main__":
    main()
