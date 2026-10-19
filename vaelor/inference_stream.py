"""A streamed chat completion from a managed local model, stoppable mid-answer.

VD-218. The NPU Assistant writes at about 14 tokens a second, so a long answer
took 20-40 s to arrive in one piece while its first words existed ~3 s in. This
reads the OpenAI-style server-sent-event stream FastFlowLM sends for
``"stream": true`` and hands each piece of text to a callback as it lands.

**Measured on the Z2 (FLM 1.0.2, qwen3.5:4b, 2026-10-16).** The stream is
``data: {...}`` lines; text arrives in ``choices[0].delta.content``, the
closing chunk carries ``finish_reason`` and - with
``stream_options.include_usage`` - the same ``usage`` object the unstreamed
reply has (``prefill_duration_ttft``, ``decoding_speed_tps``, ...), then
``data: [DONE]``. A tool call is NOT streamed: FLM holds the whole generation
and sends one ``delta.tool_calls`` chunk at the end, so a nameless tool call is
~15 s of silence followed by no text.

**Who runs what.** :func:`stream_chat_completion` runs on the worker thread that
answers the question. :meth:`StopSwitch.stop` is called from ANOTHER thread -
the request that carries the owner's Stop (``POST /assistant/chat/stop``).

**A FastFlowLM generation is never ended by closing its connection (R136).**
Measured on the Z2 2026-10-17: FastFlowLM 1.0.2 keeps one of its ten
connection places for every request whose client closes mid-generation, and
after ten refuses every connection until it is restarted; a request read to
its end costs nothing (36 Stops, R143). So a FastFlowLM generation
(:func:`~vaelor.inference_client.flm_connection`, the product's NPU marker) is
named - a ``request_id`` in its body - and every way of ending one early (the
owner's Stop, the budget, a reply over the size cap, a malformed chunk, a
failing reader) asks the server to cancel it (``POST /api/cancel``) and then
reads the connection to its end, showing nothing more. FastFlowLM 1.0.2
accepts that cancel and ignores it (R143), so the slot frees when the
generation ends - up to ``max_tokens`` later - and only the owner's answer is
immediate. Only when even that outlasts
:func:`~vaelor.inference_client.drain_seconds` on the wall clock is the
connection closed, and that is logged. Every other server - llama-server, the
vLLM balancer - has no cancel request and ends cleanly when its connection
closes, so for those an early end closes the connection at once, as before.
A page that merely closed does NOT stop the answer: it finishes and is saved
(owner, VD-218 2026-10-16; VD-110 #247i).

A stream counts as complete only when it says so - a ``finish_reason`` or
``[DONE]``. A server that dies mid-answer raises (R121), as the unstreamed
call does, rather than handing back the words it got as if they were all.

The returned body has the unstreamed reply's shape (``choices[0].message``,
``usage``, :data:`~vaelor.inference_client.PERFORMANCE_KEY`), so everything
that judges a reply reads a streamed one the same way.
"""

from __future__ import annotations

import http.client
import json
import logging
import threading
import time
from typing import Any, Callable, Dict, Optional
from urllib.parse import urlsplit

from .inference_client import (
    LAST_RESORT,
    OVER_BUDGET,
    PERFORMANCE_KEY,
    TOO_MUCH_DATA,
    CancelAtDeadline,
    cancel_request,
    drain_seconds,
    flm_connection,
    hang_up,
    new_request_id,
    normalize_performance,
)
from .flm_service import FLM_PENALTIES
from .local_inference_gate import local_inference_slot

#: The most text one streamed reply may carry, matching the unstreamed cap.
MAX_STREAMED_TEXT_CHARS = 2 * 1024 * 1024

#: The longest single SSE line read; FLM's chunks are a few hundred bytes.
MAX_EVENT_LINE_BYTES = 256 * 1024

_LOGGER = logging.getLogger(__name__)


#: Why a streamed answer ended early: the owner pressed Stop.
STOPPED = "Stop was pressed, so the answer was stopped."


class AnswerStopped(RuntimeError):
    """The owner pressed Stop, so the answer was stopped before it finished.

    A ``RuntimeError`` on purpose, like ``LocalModelBusy``: the Assistant's
    answer guard degrades ``OSError``/``ValueError`` into a built-in reply, and a
    stop is not a model failure to paper over - the owner asked for nothing.
    """


class StopSwitch:
    """Ends one streamed answer from another thread.

    ``cancel()`` ends the generation early (VD-219 at a repeat, or a budget);
    ``stop()`` is the owner's Stop: it cancels too, and the answer then raises
    :class:`AnswerStopped`. How the generation ends is the stream's to say
    (:func:`stream_chat_completion` attaches it): FastFlowLM is asked by name
    and read to its end; any other server's connection is closed. Either is
    safe at any time, any number of times - before that is attached (it runs
    when it is), while a read is waiting, or after the answer ended - and the
    ending runs at most once.
    """

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._stopped = threading.Event()
        self._cancelled = threading.Event()
        self._sender: Optional[Callable[[], Any]] = None
        self._sent = False

    @property
    def stopped(self) -> bool:
        return self._stopped.is_set()

    @property
    def cancelled(self) -> bool:
        return self._cancelled.is_set()

    def stop(self) -> None:
        self._stopped.set()
        self.cancel()

    def cancel(self) -> None:
        with self._lock:
            self._cancelled.set()
            sender = self._take_sender()
        if sender is not None:
            sender()

    def attach(self, sender: Callable[[], Any]) -> None:
        """Say how this generation is ended early; a cancel already asked for runs now."""
        with self._lock:
            self._sender = sender
            pending = self._take_sender() if self._cancelled.is_set() else None
        if pending is not None:
            pending()

    def detach(self) -> None:
        with self._lock:
            self._sender = None

    def _take_sender(self) -> Optional[Callable[[], Any]]:
        if self._sent or self._sender is None:
            return None
        self._sent = True
        return self._sender


def _connection_for(base_url: str, timeout: Optional[float]) -> http.client.HTTPConnection:
    parts = urlsplit(base_url)
    if parts.scheme != "http" or not parts.hostname:
        # managed_local_connection admits only http://127.0.0.1 and http://[::1].
        raise ValueError("A streamed answer is only read from this appliance's own model server.")
    return http.client.HTTPConnection(parts.hostname, parts.port or 80, timeout=timeout)


#: The stream's own end marker, after the closing chunk.
STREAM_DONE = "[DONE]"

#: VD-218 / R121: the model server closed the stream before saying it was done.
CUT_SHORT = (
    "The model server closed its answer before finishing it, so there is no "
    "complete answer to show."
)


def _events(response: http.client.HTTPResponse) -> Any:
    """Each decoded ``data:`` payload, then ``STREAM_DONE`` if the stream said so.

    A stream that simply ends - the server died between chunks - yields no
    ``STREAM_DONE``, and the caller must not read what it got as complete.
    The budget is kept by :class:`~vaelor.inference_client.CancelAtDeadline`,
    which cancels the generation; this keeps reading until the server ends it.
    """
    while True:
        line = response.readline(MAX_EVENT_LINE_BYTES)
        if not line:
            return
        text = line.decode("utf-8", "replace").strip()
        if not text.startswith("data:"):
            continue
        payload = text[5:].strip()
        if payload == STREAM_DONE:
            yield STREAM_DONE
            return
        try:
            event = json.loads(payload)
        except ValueError:
            continue
        if isinstance(event, dict):
            yield event


def stream_chat_completion(
    connection: Dict[str, str],
    request_body: Dict[str, Any],
    headers: Dict[str, str],
    timeout: int,
    *,
    on_text: Callable[[str], None],
    on_started: Callable[[], None] = lambda: None,
    stop: Optional[StopSwitch] = None,
) -> Dict[str, Any]:
    """One streamed completion: ``on_text`` gets the whole text so far, per piece.

    ``on_started`` fires once the endpoint's slot is held - the moment a busy
    refusal (``LocalModelBusy``, raised before it) is no longer possible.
    Raises :class:`AnswerStopped` when ``stop`` fired, ``OSError`` for a lost
    or timed-out model connection, and ``ValueError`` for a refusal the server
    answered with an error status - the same types the unstreamed call raises.
    """
    stop = stop or StopSwitch()
    parts = urlsplit(connection["base_url"])
    path = parts.path.rstrip("/") + "/chat/completions"
    drain = flm_connection(connection)
    request_id = new_request_id() if drain else ""
    body = dict(request_body, stream=True, stream_options={"include_usage": True})
    if request_id:
        body.update(FLM_PENALTIES, request_id=request_id)
    start = time.monotonic()
    with local_inference_slot(connection):
        on_started()
        if stop.stopped:
            raise AnswerStopped(STOPPED)
        # FastFlowLM: no per-read timeout - its budget and drain are wall-clock
        # timers below, so the socket never gives up on a working generation.
        client = _connection_for(connection["base_url"], None if drain else timeout)
        # R145 item 4: the whole FastFlowLM request is bounded on the wall
        # clock - its budget plus the most it could still write - so a silent
        # or wedged server cannot hold it between events. Past that, the
        # connection is closed as the logged last resort.
        held: Dict[str, Any] = {}  # the socket; http.client drops its own reference
        give_up = threading.Timer(float(timeout) + drain_seconds(request_body),
                                  lambda: hang_up(held.get("sock"), LAST_RESORT))
        give_up.daemon = True

        def end_early() -> None:
            if drain:
                cancel_request(connection, request_id)  # 1.0.2 accepts and ignores it (R143)
            else:
                hang_up(held.get("sock"))  # these servers end cleanly on a close

        # VD-218 R145: the budget runs from the request, not from the headers.
        budget = CancelAtDeadline(timeout, stop.cancel)
        failure: Optional[BaseException] = None
        try:
            client.connect()
            held["sock"] = client.sock
            if drain:
                give_up.start()
            stop.attach(end_early)
            client.request("POST", path, body=json.dumps(body).encode("utf-8"),
                           headers={**headers, "Accept": "text/event-stream"})
            response = client.getresponse()
            if response.status >= 400:
                detail = response.read(4096).decode("utf-8", "replace")
                raise ValueError("The model server refused the request (HTTP {}): {}".format(
                    response.status, detail[:300]))
            text, tool_calls, finish, usage, timings, done = "", [], None, {}, {}, False
            for event in _events(response):
                if event == STREAM_DONE:
                    done = True
                    continue
                if stop.cancelled and not drain:
                    break
                try:
                    if isinstance(event.get("usage"), dict):
                        usage = event["usage"]
                    # VD-218 / R123: llama-server reports timings on the closing chunk.
                    if isinstance(event.get("timings"), dict):
                        timings = event["timings"]
                    choices = event.get("choices") or []
                    choice = choices[0] if choices and isinstance(choices[0], dict) else {}
                    delta = choice.get("delta") if isinstance(choice.get("delta"), dict) else {}
                    finish = choice.get("finish_reason") or finish
                    if stop.cancelled:
                        continue  # draining: read to the end, show nothing more
                    tool_calls.extend(item for item in delta.get("tool_calls") or [] if isinstance(item, dict))
                    piece = delta.get("content")
                    if isinstance(piece, str) and piece:
                        text += piece
                        if len(text) > MAX_STREAMED_TEXT_CHARS:
                            raise ValueError(TOO_MUCH_DATA)
                        on_text(text)
                except Exception as error:  # noqa: BLE001 - raised once the generation ended
                    # A malformed chunk, the size cap or a failing reader: end the
                    # generation the server's way, then say why (R145 item 5).
                    failure = failure or error
                    stop.cancel()
        except (OSError, http.client.HTTPException) as error:
            if stop.stopped:
                raise AnswerStopped(STOPPED) from None
            if failure is not None:
                raise failure from None
            if budget.fired:
                raise TimeoutError(OVER_BUDGET) from error
            if isinstance(error, OSError):
                raise
            # R121: a close inside a chunk is IncompleteRead, an HTTPException no
            # answer guard catches; it is a lost connection, said as one.
            raise ConnectionError(CUT_SHORT) from error
        finally:
            budget.cancel()
            give_up.cancel()
            stop.detach()
            client.close()
        if stop.stopped:
            raise AnswerStopped(STOPPED)
        if failure is not None:
            raise failure
        if budget.fired:
            raise TimeoutError(OVER_BUDGET)
        if not done and finish is None:
            # R121 (LESSONS 1, 8): partial text is not an answer.
            raise ValueError(CUT_SHORT)
    message: Dict[str, Any] = {"role": "assistant", "content": text}
    if tool_calls:
        message["tool_calls"] = tool_calls
    reply: Dict[str, Any] = {"choices": [{"index": 0, "message": message, "finish_reason": finish}],
                             "usage": usage}
    if timings:
        reply["timings"] = timings
    reply[PERFORMANCE_KEY] = normalize_performance(reply, time.monotonic() - start)
    return reply

