"""AI Chat's native connector to Anthropic's Messages API (VD-206 item 1).

Not Anthropic's OpenAI-compatibility endpoint, which Anthropic marks as
limited: this speaks ``POST /v1/messages`` and ``GET /v1/models`` with the
``x-api-key`` and ``anthropic-version`` headers, as documented at
platform.claude.com (read 2026-10-07; recorded in the change's design note).

What it handles, and how:

- **Streaming.** Every answer is requested with ``"stream": true`` and read as
  server-sent events. AI Chat shows an answer when it is complete, so the
  stream is accumulated here; it is streamed because Anthropic advises against
  a large ``max_tokens`` on a non-streaming request (idle connections drop on
  long generations), and because an error can arrive mid-stream after a 200.
- **The system prompt** goes in the top-level ``system`` field; there is no
  system role in ``messages``.
- **Multi-turn history** is sent as alternating ``user`` / ``assistant``
  turns. Consecutive turns of one role are joined and a leading assistant
  turn is dropped, so the conversation always starts and ends on the user.
- **Long outputs**: ``max_tokens`` is the caller's; a ``max_tokens`` stop is
  reported as ``finish_reason: "length"``, the same truncation signal every
  other AI Chat connection gives.
- **Files**: AI Chat sends what it has read from the owner's uploaded
  documents as retrieved passages. Each passage goes to Anthropic as a
  ``document`` content block with a plain-text source, titled with its
  ``[S#]`` marker, so it is file input in Anthropic's own shape rather than
  text pasted into the question.
- **Thinking** (VD-209): the fields `chat_thinking.plan` chose for the
  owner's step are merged into the body as given (``thinking``,
  ``output_config``); with none, each model uses its default. Thinking is
  never part of the answer: ``thinking_delta`` text is gathered apart, as the
  summary AI Chat shows collapsed, and the time from the first thinking block
  opening to the last one closing is measured off the stream. A
  ``redacted_thinking`` block carries no text and adds none. History goes as
  plain text with no thinking blocks, which the docs allow outside tool use.
  No ``temperature`` is sent (deprecated on newer models).
- **Errors**: a 401, a rate limit (429) and "overloaded" (529, or an
  ``overloaded_error`` event inside a 200 stream) each answer their own
  sentence (`hosted_transport.refusal_for`); no sentence carries the key.
"""

from __future__ import annotations

import json
import urllib.error
import time
from typing import Any, Callable, Dict, Iterable, List, Mapping, Optional, Sequence

from .credential_broker_client import CredentialError
from .hosted_transport import (
    HostedResponse,
    USER_AGENT,
    error_type_of,
    redact,
    refusal_for,
    request,
)

ANTHROPIC_BASE_URL = "https://api.anthropic.com/v1"
ANTHROPIC_VERSION = "2023-06-01"
ANTHROPIC_COMPANY = "Anthropic"
#: ``GET /v1/models`` returns 20 by default; 1,000 is its documented maximum.
MODEL_LIST_LIMIT = 1000
#: The stop reasons that mean the answer was cut short, not finished. The
#: second is the context window filling up mid-answer.
LENGTH_STOPS = frozenset({"max_tokens", "model_context_window_exceeded"})
#: Thinking text kept while a stream is read: one character past the 12,000
#: `chat_thinking.bounded_summary` keeps, so it can still tell a cut summary.
THINKING_CAP = 12001

Send = Callable[..., HostedResponse]


class AnthropicError(CredentialError):
    """A refusal from Anthropic, in the owner's words, with a stable code."""

    def __init__(self, message: str, *, code: str, status: int = 0, detail: str = ""):
        super().__init__(message)
        self.code = code
        self.status = status
        #: Anthropic's own reason, redacted: read to tell a refused thinking
        #: setting from any other 400 (`chat_thinking.refused_setting`).
        self.detail = detail


def anthropic_headers(api_key: str) -> Dict[str, str]:
    """The documented headers: the key in ``x-api-key``, never a bearer token."""
    return {
        "x-api-key": api_key,
        "anthropic-version": ANTHROPIC_VERSION,
        "content-type": "application/json",
        "accept": "application/json",
        "user-agent": USER_AGENT,
    }


def _turns(history: Iterable[Mapping[str, Any]]) -> List[Dict[str, Any]]:
    """History as Anthropic accepts it: user first, roles alternating."""
    turns: List[Dict[str, Any]] = []
    for item in history or []:
        role = item.get("role")
        text = str(item.get("content") or "").strip()
        if role not in {"user", "assistant"} or not text:
            continue
        if not turns and role == "assistant":
            continue
        if turns and turns[-1]["role"] == role:
            turns[-1]["content"] += "\n\n" + text
        else:
            turns.append({"role": role, "content": text})
    return turns


def document_blocks(sources: Sequence[Mapping[str, str]]) -> List[Dict[str, Any]]:
    """Retrieved passages as plain-text ``document`` blocks, one per ``[S#]``."""
    blocks = []
    for source in sources:
        text = str(source.get("text") or "").strip()
        if not text:
            continue
        blocks.append({
            "type": "document",
            "source": {"type": "text", "media_type": "text/plain", "data": text},
            "title": str(source.get("title") or "Retrieved source")[:500],
        })
    return blocks


def build_request(
    *, model: str, system: str, history: Iterable[Mapping[str, Any]],
    question: str, sources: Sequence[Mapping[str, str]], max_tokens: int,
    thinking: Optional[Mapping[str, Any]] = None,
) -> Dict[str, Any]:
    """The ``/v1/messages`` body for one AI Chat turn."""
    messages = _turns(history)
    final: List[Dict[str, Any]] = [*document_blocks(sources), {"type": "text", "text": question}]
    if messages and messages[-1]["role"] == "user":
        # The previous question never got an answer; ask both in one turn.
        earlier = messages.pop()["content"]
        final = [{"type": "text", "text": earlier}, *final]
    messages.append({"role": "user", "content": final})
    body: Dict[str, Any] = {
        "model": model,
        "max_tokens": int(max_tokens),
        "messages": messages,
        "stream": True,
    }
    if system.strip():
        body["system"] = system
    for name, value in (thinking or {}).items():
        body[name] = value
    return body


def _events(lines: Iterable[str]) -> Iterable[Dict[str, Any]]:
    """Server-sent events as parsed JSON payloads (event name in ``type``)."""
    data: List[str] = []
    for line in lines:
        if line == "":
            if data:
                try:
                    payload = json.loads("\n".join(data))
                except json.JSONDecodeError:
                    payload = None
                data = []
                if isinstance(payload, dict):
                    yield payload
            continue
        if line.startswith(":"):
            continue
        if line.startswith("data:"):
            data.append(line[5:].lstrip())
    if data:
        try:
            payload = json.loads("\n".join(data))
        except json.JSONDecodeError:
            return
        if isinstance(payload, dict):
            yield payload


def read_stream(lines: Iterable[str], *, model: str = "",
                clock: Callable[[], float] = time.monotonic) -> Dict[str, Any]:
    """Accumulate one streamed answer: its text, thinking, stop reason and usage.

    ``thinking`` is the summarized thinking text, kept apart from the answer
    and capped at `THINKING_CAP` characters while it is read, and
    ``thinking_seconds`` the measured span from the first thinking block
    opening to the last one closing (None when no thinking block arrived).

    Raises `AnthropicError` for an ``error`` event (Anthropic sends one inside
    a 200 when, for example, it becomes overloaded mid-answer) and for a
    stream that ends before ``message_stop``.
    """
    text: List[str] = []
    thought: List[str] = []
    thought_size = 0
    opened: Optional[float] = None
    closed: Optional[float] = None
    thinking_blocks = set()
    stop_reason = ""
    usage = {"input_tokens": 0, "output_tokens": 0}
    finished = False
    for event in _events(lines):
        kind = event.get("type")
        if kind == "message_start":
            started = (event.get("message") or {}).get("usage") or {}
            usage["input_tokens"] = int(started.get("input_tokens") or 0)
            usage["output_tokens"] = int(started.get("output_tokens") or 0)
        elif kind == "content_block_start":
            block = event.get("content_block") or {}
            if block.get("type") in ("thinking", "redacted_thinking"):
                thinking_blocks.add(event.get("index"))
                opened = clock() if opened is None else opened
            if block.get("type") == "text" and block.get("text"):
                text.append(str(block["text"]))
        elif kind == "content_block_delta":
            delta = event.get("delta") or {}
            if delta.get("type") == "text_delta":
                text.append(str(delta.get("text") or ""))
            elif delta.get("type") == "thinking_delta" and thought_size < THINKING_CAP:
                piece = str(delta.get("thinking") or "")[:THINKING_CAP - thought_size]
                thought.append(piece)
                thought_size += len(piece)
        elif kind == "content_block_stop":
            if event.get("index") in thinking_blocks:
                closed = clock()
        elif kind == "message_delta":
            stop_reason = str((event.get("delta") or {}).get("stop_reason") or stop_reason)
            counted = event.get("usage") or {}
            if "output_tokens" in counted:
                usage["output_tokens"] = int(counted.get("output_tokens") or 0)
        elif kind == "message_stop":
            finished = True
            break
        elif kind == "error":
            detail = event.get("error") or {}
            code, sentence = refusal_for(
                ANTHROPIC_COMPANY, 0, str(detail.get("type") or ""), model=model,
            )
            raise AnthropicError(sentence, code=code)
    if not finished:
        raise AnthropicError(
            "Anthropic's answer stream ended before the answer was complete. "
            "Retry the question.",
            code="hosted_stream_incomplete",
        )
    seconds = None
    if opened is not None:
        seconds = max(0.0, (closed if closed is not None else clock()) - opened)
    return {
        "text": "".join(text), "stop_reason": stop_reason, "usage": usage,
        "thinking": "".join(thought), "thinking_seconds": seconds,
    }


def _refusal(error: urllib.error.HTTPError, api_key: str, model: str = "") -> AnthropicError:
    kind, detail = error_type_of(error)
    clean = redact(detail, api_key)
    code, sentence = refusal_for(ANTHROPIC_COMPANY, error.code, kind, clean, model)
    return AnthropicError(sentence, code=code, status=error.code, detail=clean)


def complete(
    api_key: str,
    *,
    model: str,
    system: str,
    history: Iterable[Mapping[str, Any]],
    question: str,
    sources: Sequence[Mapping[str, str]] = (),
    max_tokens: int,
    timeout: float,
    send: Optional[Send] = None,
    base_url: str = ANTHROPIC_BASE_URL,
    thinking: Optional[Mapping[str, Any]] = None,
) -> Dict[str, Any]:
    """One AI Chat answer, returned in the OpenAI chat-completions shape.

    ``choices[0].message.content`` is the accumulated text and
    ``finish_reason`` is ``"length"`` for a ``max_tokens`` stop, so the caller
    treats it exactly as any other connection's answer.
    """
    body = build_request(
        model=model, system=system, history=history, question=question,
        sources=sources, max_tokens=max_tokens, thinking=thinking,
    )
    try:
        with (send or request)(
            "POST", base_url.rstrip("/") + "/messages",
            headers=anthropic_headers(api_key),
            body=json.dumps(body).encode("utf-8"), timeout=timeout,
        ) as response:
            result = read_stream(response.lines(), model=model)
    except urllib.error.HTTPError as error:
        raise _refusal(error, api_key, model) from None
    if result["stop_reason"] == "refusal" and not result["text"].strip():
        raise AnthropicError(
            "{} declined to answer this request (stop reason: refusal).".format(model or "The model"),
            code="hosted_declined",
        )
    return {
        "model": model,
        "choices": [{
            # The thinking summary rides where OpenAI-shaped bodies carry
            # reasoning, so AI Chat reads every provider's the same way.
            "message": {"role": "assistant", "content": result["text"],
                        "reasoning": result["thinking"]},
            "finish_reason": "length" if result["stop_reason"] in LENGTH_STOPS else "stop",
        }],
        "usage": {
            "prompt_tokens": result["usage"]["input_tokens"],
            "completion_tokens": result["usage"]["output_tokens"],
        },
        "thinking_seconds": result["thinking_seconds"],
    }


def list_models(api_key: str, *, send: Optional[Send] = None,
                base_url: str = ANTHROPIC_BASE_URL, timeout: float = 20) -> List[str]:
    """The model ids this key may use, newest first as Anthropic lists them."""
    try:
        with (send or request)(
            "GET", "{}/models?limit={}".format(base_url.rstrip("/"), MODEL_LIST_LIMIT),
            headers=anthropic_headers(api_key), timeout=timeout,
        ) as response:
            payload = response.json()
    except urllib.error.HTTPError as error:
        raise _refusal(error, api_key) from None
    items = payload.get("data") if isinstance(payload, dict) else None
    if not isinstance(items, list):
        raise AnthropicError(
            "Anthropic's model list was not in the shape Vaelor reads.",
            code="hosted_invalid_response",
        )
    models: List[str] = []
    for item in items:
        model_id = str((item or {}).get("id") or "").strip() if isinstance(item, dict) else ""
        if model_id and model_id not in models:
            models.append(model_id)
    return models


def probe(api_key: str, *, send: Optional[Send] = None,
          base_url: str = ANTHROPIC_BASE_URL) -> Dict[str, Any]:
    """The connection test: list the models, which needs a valid key."""
    try:
        models = list_models(api_key, send=send, base_url=base_url, timeout=12)
    except CredentialError as error:
        return {"ok": False, "message": str(error)}
    except (urllib.error.URLError, OSError):
        return {
            "ok": False,
            "message": "Anthropic could not be reached from this appliance. "
                       "Check its internet connection.",
        }
    if not models:
        return {"ok": False, "message": "Anthropic accepted the key but listed no models."}
    return {
        "ok": True,
        "message": "Anthropic accepted the API key and lists {} model{}.".format(
            len(models), "" if len(models) == 1 else "s"),
    }
