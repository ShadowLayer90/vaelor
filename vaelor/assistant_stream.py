"""The Assistant's answer, streamed to the page while the model writes it.

VD-218. ``POST /assistant/chat`` was one blocking request: the page showed
nothing until the last token, ~20 s for a long NPU answer whose first words
existed ~3 s in. A request that sends ``Accept: application/x-ndjson`` now gets
the answer as it is written - one JSON object per line:

* ``{"event": "started"}`` once the model's slot is held;
* ``{"event": "draft", "append": "..."}`` or ``{"event": "draft", "text": "..."}``
  - the provisional text grew, or was replaced (``""`` takes it back before a
  retry);
* ``{"event": "wait"}`` each second nothing else was sent;
* ``{"event": "answer", "status": 200, "body": {...}}`` - the same envelope and
  status the unstreamed request returns, and the text that is stored. It
  replaces the draft.

**Nothing changes until the model starts writing.** The question runs on a
worker thread, and the route waits for its first event: a turn that finishes
first - a built-in answer, a dedupe replay, the busy gate's 503, a bad request -
is returned exactly as the unstreamed route returns it, status included. Only a
model that holds its slot opens the stream.

**The draft is provisional, never the answer.** :func:`provisional_text` shows
only what the final answer would also show. It shows whole sentences only, so
each is judged complete, and then runs the same judges the final answer runs:
the echo guard, the scope guard's misrouted-refusal check (a declining sentence
is left out, as :func:`~vaelor.assistant_scope_guard.guarded_answer` leaves it
out), the internal-names judge and the JSON flattening. A reply that writes a
prompt-envelope key anywhere shows nothing at all. The final ``answer`` event
always replaces the draft.

**Two threads, and who may stop the answer.** The worker thread
(:func:`streamed_answer`'s ``work``) runs the route's own turn, model call
included; the response thread (:func:`_stream`) writes lines to the browser.
When the browser goes - a closed tab, a reload, a dropped link - the response
thread only detaches: the answer finishes and is saved, and a reloaded page's
resume-poll shows it (owner, VD-218 2026-10-16; VD-110 #247i). Only the owner's
explicit Stop - ``POST /assistant/chat/stop`` with the turn's idempotency key,
from the same account (:func:`stop_running`) - stops the model and frees its
slot; the turn then records the stop in the transcript (:func:`record_stopped`).
"""

from __future__ import annotations

import json
import logging
import queue
import threading
import time
from typing import Any, Callable, Dict, Iterator, Optional, Tuple

from flask import Response, copy_current_request_context, g, request

from .assistant_contract_reader import contract_end, contract_text, without_hand_off
from .assistant_local_answer import (
    humanize_answer,
    internal_names_only,
    is_degenerate,
    truncated_json,
)
from .assistant_scope_guard import echoes_prompt_envelope, kept_sentences, misrouted_refusal
from .inference_client import MAX_INFERENCE_SECONDS, PERFORMANCE_KEY
from .inference_stream import STOPPED, AnswerStopped, StopSwitch, stream_chat_completion

_LOGGER = logging.getLogger(__name__)

#: What a streaming client sends in ``Accept``.
STREAM_MEDIA_TYPE = "application/x-ndjson"

#: Seconds between ``wait`` lines while nothing else is sent. A write is how a
#: closed browser is noticed, and the page treats a long silence as a lost link.
HEARTBEAT_SECONDS = 1.0

#: The longest the route waits for the turn's first event: the model's own
#: ceiling plus the time to gather readings. The turn's model call is bounded
#: by its budget; this only stops a wedged worker holding the request forever.
FIRST_EVENT_SECONDS = MAX_INFERENCE_SECONDS + 120

#: VD-219 / R136, R138: end a generation early with the server's own cancel
#: request at the first repeated key, and when a new ask finds that tail still
#: running - then read the tail to its end. The connection is never closed on
#: a running generation either way (R136: FastFlowLM 1.0.2 keeps a connection
#: place for every one, and refuses everyone after ten). Off, the tail simply
#: runs to its end, which is also leak-free, only slower. The owner's Stop
#: always cancels (VD-218).
FLM_API_CANCEL = True

#: The transcript's record of an answer the owner stopped.
STOPPED_ANSWER = (
    "This answer was stopped before it finished, because Stop was pressed. The "
    "appliance stopped writing it and freed the on-device model, and nothing on "
    "this appliance was changed. Ask again for an answer."
)

#: The prompt envelope's own keys. A reply that writes one of them as an object
#: key is reflecting its input, and the draft shows nothing of it (R122): the
#: echo guard counts four before it fires, and a draft cannot wait for four.
_ENVELOPE_KEYS = ("question", "context", "facts", "memories", "conversation")

#: Where a sentence is known to be complete: its end mark, then a space.
_SENTENCE_ENDS = (". ", "! ", "? ", ".\n", "!\n", "?\n")


def wants_stream(accept: str) -> bool:
    """Whether the request's ``Accept`` header asks for the streamed answer."""
    return STREAM_MEDIA_TYPE in str(accept or "")


def _complete_sentences(text: str) -> str:
    """``text`` up to its last finished sentence or line; ``""`` when none is finished."""
    ends = [text.rfind(mark) + 1 for mark in _SENTENCE_ENDS] + [text.rfind("\n")]
    return text[:max(ends)].rstrip() if max(ends) > 0 else ""


def _writes_envelope_key(text: str) -> bool:
    """Whether ``text`` uses a prompt-envelope word as an object key, in either quote."""
    for key in _ENVELOPE_KEYS:
        for quote in ('"', "'"):
            marker = quote + key + quote
            at = text.find(marker)
            while at >= 0:
                if text[at + len(marker):].lstrip()[:1] in (":", ""):
                    return True
                at = text.find(marker, at + 1)
    return False


def _without_dangling_labels(text: str) -> str:
    """``text`` without trailing label lines (``**Sources:**``) or bare list markers."""
    lines = text.split("\n")
    while lines and (not lines[-1].strip(" *_-") or lines[-1].strip(" *_").endswith(":")):
        lines.pop()
    # A list still growing ends on its separator; judged with it, the empty
    # last entry made a list of internal names read as prose.
    return "\n".join(lines).strip().rstrip(",;").rstrip()


def provisional_text(raw: Any, question: Any = "") -> str:
    """What the page may show of a reply still being written; ``""`` to show nothing.

    VD-218, R122 (LESSONS 1, 6): the draft runs the judges the final answer
    runs, on whole sentences, so it shows nothing the final answer would hide -
    not a code fence's marker, a JSON reply's keys or braces, a label over
    internal names, an echoed prompt or any part of one, nor a sentence the
    scope guard withholds as a misrouted refusal.
    """
    text = str(raw or "").lstrip()
    if _writes_envelope_key(text) or echoes_prompt_envelope(text):
        return ""
    if text.startswith("```"):
        newline = text.find("\n")
        if newline < 0:
            return ""
        text = text[newline + 1:]
        closing = text.find("```")
        text = (text if closing < 0 else text[:closing] + "\n").lstrip()
    shown, complete = text, False
    if text.startswith("{"):
        # VD-219: a JSON reply - the trained contract - is read the way the
        # final answer reads it: the summary, then the details, finished
        # sentences only, never a key, a brace or the sources. One with no
        # answer key yet shows nothing.
        shown, complete = contract_text(text, settled=False), True
    elif text.startswith("["):
        if truncated_json(text):
            return ""
        rendered = humanize_answer(text)
        # A complete JSON value renders whole; prose that merely starts
        # with a bracket ("[WARN] ...") is judged as prose below.
        shown, complete = (rendered, True) if rendered != text.strip() else (text, False)
    shown = _without_dangling_labels(shown if complete else _complete_sentences(shown))
    if not shown or internal_names_only(shown) or is_degenerate(shown) or echoes_prompt_envelope(shown):
        return ""
    # VD-220 B4 (R208, R209): the final answer never shows the model's hand-off
    # line - it is the AI Chat redirect, the guard's notice, or dropped beside
    # an answer - so the draft never shows it either.
    shown = without_hand_off(shown)
    if not shown:
        return ""
    if misrouted_refusal(question, shown, "connected-model"):
        shown = kept_sentences(shown)
    return shown


class AnswerDraft:
    """The page's view of one answer while the model writes it.

    Made on the request thread, handed to the worker: ``completion`` is the
    streamed model call (the ``call`` :func:`local_answer` takes) and
    ``restart`` takes the draft back before a retry. Both run on the worker.
    ``detach`` runs on the response thread when the browser goes; ``stop`` on
    the request that carries the owner's Stop.
    """

    def __init__(self, question: Any = "") -> None:
        self.events: "queue.Queue[Tuple[str, Any]]" = queue.Queue()
        self.stop_switch = StopSwitch()
        self._owner_stopped = False
        self.question = question
        self._shown = ""
        self._started = False
        self._reader = True

    def completion(self, connection, request_body, headers, timeout) -> Dict[str, Any]:
        """One streamed model call; answered early, whole, when the reply starts repeating itself.

        VD-219 (LESSONS 26): the NPU fine-tune often never stops after its
        answer - it writes its last keys again and again to the token limit
        (12 of 44 answers, ~25 s each). :func:`contract_end` sees the first
        repeated key, and the text before it is returned at once as the
        finished reply. A FastFlowLM call is never cut off (R136: closing a
        running generation costs FastFlowLM 1.0.2 one of its ten connection
        places): with :data:`FLM_API_CANCEL` on it is asked to cancel by name -
        which 1.0.2 accepts and ignores (R143) - and is read to its end on a
        worker; the slot frees when the generation ends, up to its token limit
        later. The next call to the same model - this answer's one re-ask, or
        the owner's next question - first waits, bounded, for that tail (R138),
        so it is answered after it instead of being refused as "busy". The
        owner's Stop answers at once and leaves the same tail behind it.
        """
        self._await_tail(connection, timeout)
        switch = self.stop_switch = StopSwitch()
        if self._owner_stopped:
            switch.stop()
        seen = {"text": "", "end": -1}
        ended, finished = threading.Event(), threading.Event()
        outcome: Dict[str, Any] = {}
        started = time.monotonic()

        def on_text(raw: str) -> None:
            if seen["end"] >= 0:
                return
            end = contract_end(raw) if raw.lstrip().startswith("{") else -1
            seen["text"], seen["end"] = (raw[:end], end) if end >= 0 else (raw, -1)
            self._show(provisional_text(seen["text"], self.question))
            if end >= 0:
                ended.set()
                if FLM_API_CANCEL:
                    switch.cancel()

        def call() -> None:
            try:
                outcome["reply"] = stream_chat_completion(
                    connection, request_body, headers, timeout,
                    on_text=on_text, on_started=self._start, stop=switch)
            except BaseException as error:  # noqa: BLE001 - handed to the asking thread
                outcome["error"] = error
            finally:
                finished.set()

        threading.Thread(target=call, name="vaelor-assistant-call", daemon=True).start()
        while not finished.wait(0.05):
            if self._owner_stopped:
                # VD-218 / R136: the Stop answers the owner now; the cancelled
                # generation is read to its end behind it, and the next ask on
                # this model waits for that tail.
                with _TAILS_LOCK:
                    _TAILS[str(connection.get("base_url") or "")] = (finished, switch)
                raise AnswerStopped(STOPPED)
            if ended.is_set():
                with _TAILS_LOCK:
                    _TAILS[str(connection.get("base_url") or "")] = (finished, switch)
                return {"choices": [{"index": 0, "finish_reason": "stop",
                                     "message": {"role": "assistant", "content": seen["text"]}}],
                        "usage": {}, "ended_at_repeat": True,
                        PERFORMANCE_KEY: {"total_seconds": round(time.monotonic() - started, 3)}}
        if "error" in outcome:
            raise outcome["error"]
        if self._owner_stopped:
            raise AnswerStopped(STOPPED)
        reply = outcome["reply"]
        if seen["end"] >= 0:
            reply["choices"][0]["message"]["content"] = seen["text"]
        return reply

    def _await_tail(self, connection: Dict[str, str], timeout: Any) -> None:
        """Wait, bounded, for the tail of an answer handed on at a repeat on this model (R138).

        Measured on the Z2 (LESSONS 16): asked at once, 28 of 44 questions met
        the previous answer's tail as "busy". Only a tail waits here; a model
        busy with another real answer still refuses fast, as the gate intends.
        """
        with _TAILS_LOCK:
            held = _TAILS.get(str(connection.get("base_url") or ""))
        tail = held[0] if held is not None else None
        if tail is None or tail.is_set():
            return
        # Every tail was already asked to end - at its repeat, or by the Stop
        # that left it (StopSwitch sends that once) - so there is nothing more
        # to ask here; it is waited for (R145).
        deadline = time.monotonic() + max(1.0, float(timeout or 0))
        while not tail.wait(0.1):
            if self._owner_stopped:
                raise AnswerStopped(STOPPED)
            if time.monotonic() > deadline:
                return  # the slot gate then says busy, as it always has

    def restart(self) -> None:
        self._show("")

    def stop(self) -> None:
        """The owner's Stop: cancel the generation by name and read it to its end - never hang up (R136)."""
        self._owner_stopped = True
        self.stop_switch.stop()

    def detach(self) -> None:
        """Nobody reads the draft any more: stop queueing it, keep answering."""
        self._reader = False

    def _start(self) -> None:
        if not self._started:
            self._started = True
            self.events.put(("started", None))

    def _show(self, text: str) -> None:
        if text == self._shown:
            return
        if self._shown and text.startswith(self._shown):
            event = {"event": "draft", "append": text[len(self._shown):]}
        else:
            event = {"event": "draft", "text": text}
        self._shown = text
        if self._reader:
            # Detached, the queue would grow with every piece until the turn
            # ended, holding text no page will read (R124).
            self.events.put(("line", event))


#: The call of an answer handed on at a repeat, still running to its limit, by
#: model endpoint (VD-219 / R138): the next call to that model waits for it.
_TAILS: Dict[str, Tuple[threading.Event, StopSwitch]] = {}
_TAILS_LOCK = threading.Lock()


#: Answers being written now, by (account, idempotency key), for Stop.
_RUNNING: Dict[Tuple[str, str], AnswerDraft] = {}
_RUNNING_LOCK = threading.Lock()


def stop_running(actor: str, key: str) -> bool:
    """Stop the answer ``actor`` is waiting on under ``key``; False when there is none.

    Keyed by the account as well as the key, so an answer can only be stopped
    by the account that asked for it.
    """
    with _RUNNING_LOCK:
        draft = _RUNNING.get((str(actor), str(key)))
    if draft is None:
        return False
    draft.stop()
    return True


def record_stopped(store: Any, conversation_id: str, actor: str) -> None:
    """Close the transcript on a stopped answer, so a reload is not left waiting.

    The question was written before the model ran; with no reply under it, the
    page's resume-poll would wait out its window for an answer that is never
    coming (VD-110 #247i).
    """
    store.append_message(conversation_id, actor, "assistant", STOPPED_ANSWER,
                         metadata={"source": "stopped", "stopped": True, "evidence": []})


def _line(event: Dict[str, Any]) -> bytes:
    return (json.dumps(event, separators=(",", ":"), default=str) + "\n").encode("utf-8")


def _final(outcome: Any) -> Tuple[int, Any]:
    response, status = outcome if isinstance(outcome, tuple) else (outcome, None)
    return int(status or response.status_code), response.get_json(silent=True)


def streamed_answer(turn: Callable[[AnswerDraft], Any], *, actor: str = "",
                    key: str = "", question: Any = "") -> Any:
    """Run ``turn(draft)`` on a worker; return its result, or stream it.

    ``turn`` is the route's whole answer, returning what the route returns.
    While it runs, ``(actor, key)`` names it for :func:`stop_running`.
    """
    draft = AnswerDraft(question)
    saved = {name: getattr(g, name) for name in g}
    # Read here, so the worker's copy of the request never reads a body this
    # request has already finished with (Flask caches the parsed JSON).
    request.get_json(silent=True)
    name: Optional[Tuple[str, str]] = (str(actor), str(key)) if actor and key else None
    if name is not None:
        with _RUNNING_LOCK:
            _RUNNING[name] = draft

    @copy_current_request_context
    def work() -> None:
        # A copied request context gets a fresh application context, and with
        # it an empty `g`: the signed-in session the route reads is put back.
        for attribute, value in saved.items():
            setattr(g, attribute, value)
        try:
            draft.events.put(("done", turn(draft)))
        except BaseException as error:  # noqa: BLE001 - handed to the response thread
            draft.events.put(("crash", error))
        finally:
            if name is not None:
                with _RUNNING_LOCK:
                    if _RUNNING.get(name) is draft:
                        del _RUNNING[name]

    threading.Thread(target=work, name="vaelor-assistant-answer", daemon=True).start()
    try:
        kind, value = draft.events.get(timeout=FIRST_EVENT_SECONDS)
    except queue.Empty:
        draft.stop()
        raise TimeoutError("The Assistant's answer did not start within its time budget.")
    if kind == "done":
        return value
    if kind == "crash":
        raise value
    return Response(_stream(draft), mimetype=STREAM_MEDIA_TYPE, headers={
        "Cache-Control": "no-store", "X-Accel-Buffering": "no"})


def _stream(draft: AnswerDraft) -> Iterator[bytes]:
    try:
        yield _line({"event": "started"})
        while True:
            try:
                kind, value = draft.events.get(timeout=HEARTBEAT_SECONDS)
            except queue.Empty:
                yield _line({"event": "wait"})
                continue
            if kind == "line":
                yield _line(value)
            elif kind == "done":
                status, body = _final(value)
                yield _line({"event": "answer", "status": status, "body": body})
                return
            elif kind == "crash":
                _LOGGER.error("Assistant answer failed after streaming began",
                              exc_info=(type(value), value, value.__traceback__))
                yield _line({"event": "answer", "status": 500, "body": {"ok": False, "error": {
                    "code": "assistant_failed",
                    "message": "The Assistant failed while writing this answer. Ask again.",
                }}})
                return
    finally:
        # The browser went (waitress closed this generator), or the answer was
        # delivered. Either way nobody reads the draft now; the answer itself
        # runs to the end and is saved (owner, VD-218 2026-10-16).
        draft.detach()
