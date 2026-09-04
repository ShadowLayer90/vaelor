"""Managed-local Assistant answers: natural-language in, plain text out.

**Measured live on the NPU (qwen3.5:4b via FastFlowLM), 2026-08-25.** The
managed-local answer path used to hand the model a JSON prompt envelope -
``{"question": ..., "context": {"facts": {...}, ...}}`` - as the user turn. On
that input the model ECHOED the envelope verbatim instead of answering: 10 of 10
questions across the battery came back as the input reflected, broad diagnostic
and plain reading alike. The specific-reading questions only look answered in the
product because a built-in reading answers them before the model is ever called
(:mod:`vaelor.deployment_agent`); every question that actually reached the model
echoed. The broad diagnostic questions - "are there any issues?", "is everything
healthy?" - have no built-in reading to fall back to, so the echo was what the
owner saw, either leaked raw or replaced by the degrade guard.

Presented the identical question and readings as *natural language* - the
question, then the same facts as readable lines - the model ANSWERED 9 of 9 of
the same battery. So the envelope's JSON *shape* was the trigger, not the
question's breadth: a small instruct model reads a lone JSON object as "reflect
this" far more readily than a sentence followed by evidence. This module builds
that natural-language user turn.

The 4B still tends to *reply* in JSON even when asked for prose (adding "no JSON"
to the prompt was measured to make some answers worse, per the repo's
small-model-steering lesson), so :func:`humanize_answer` coerces a JSON-shaped
reply back to plain text rather than steering the model harder. And because a
single small model is never perfectly reliable, :func:`is_degenerate` /
:func:`with_retry_nudge` let the caller retry once when the first reply still
echoes or comes back empty, before the scope guard degrades it.
"""

from __future__ import annotations

import ast
import json
from typing import Any, Dict

from .assistant_scope_guard import echoes_prompt_envelope
from .provider_runtime import assistant_context

#: One line added to the *retry* only. Kept to a single sentence on purpose:
#: the fix is the natural-language format, not the instruction, and piling
#: steering onto this tier regresses it (small-model-steering lesson). It never
#: rides the first request, so a question the format already answers is never
#: paying for it.
_RETRY_NUDGE = (
    "Answer the question above in plain sentences using those readings. "
    "Do not repeat the input and do not reply with JSON."
)

#: A managed-local user turn always carried this suffix (``provider_user_content``
#: appends it for the loopback connection), so the natural-language turn keeps it
#: rather than changing two things at once.
_NO_THINK = "\n/no_think"

#: Keys a JSON-shaped reply uses to carry the actual answer sentence, richest
#: first. The 4B most often leads a real answer with ``summary``; the others are
#: the shapes seen across the live battery.
_LEAD_KEYS = ("answer", "summary", "response", "reply", "text", "message", "result")

#: List-valued keys whose entries expand on the lead sentence.
_DETAIL_KEYS = (
    "details", "findings", "notes", "issues", "recommendations", "checks",
    "next_actions", "actions", "observations",
)


def local_user_content(
    message: str, context: Dict[str, Any], connection: Dict[str, str]
) -> str:
    """The user turn for a managed-local answer, as natural language.

    The facts are the same ones the JSON envelope carried - selected and
    budget-trimmed by :func:`vaelor.provider_runtime.assistant_context`, so the
    context window policy is unchanged - only their *presentation* differs: a
    question, then the readings as ``- name: value`` lines, instead of one JSON
    object. Everything measured used exactly this shape.
    """
    trimmed = assistant_context(message, context, connection)
    facts = trimmed.get("facts", {}) if isinstance(trimmed, dict) else {}
    parts = [str(message).strip()]
    if isinstance(facts, dict) and facts:
        lines = "\n".join(
            "- {}: {}".format(key, json.dumps(value, separators=(",", ":"), default=str))
            for key, value in facts.items()
        )
        parts.append("Current readings from this machine:\n" + lines)
    return "\n\n".join(parts) + _NO_THINK


def first_choice_content(body: Any) -> str:
    """The first choice's message content, or a caught error when absent.

    An OpenAI-compatible endpoint can return HTTP 200 with an empty ``choices``
    list - a refusal, a content-filter block, or a truncated shape. Indexing
    ``choices[0]`` on that raises ``IndexError``, which the Assistant answer
    guard does not catch, so it used to escape as an HTTP 500 and leak the
    in-flight dedupe claim. It is raised here as a ``ValueError`` instead - a
    type the answer guard already degrades - matching the treatment
    :mod:`vaelor.chat_inference` gives this same wire shape.
    """
    choices = body.get("choices") if isinstance(body, dict) else None
    if not isinstance(choices, list) or not choices:
        raise ValueError(
            "The model server returned no choices, so its reply carries no "
            "answer to read."
        )
    first = choices[0]
    message = first.get("message") if isinstance(first, dict) else None
    if not isinstance(message, dict) or "content" not in message:
        raise ValueError(
            "The model server's reply had no message content to read."
        )
    return message["content"]


def with_retry_nudge(user_content: str) -> str:
    """The same user turn with the one-line answer nudge, kept before ``/no_think``."""
    if user_content.endswith(_NO_THINK):
        return user_content[: -len(_NO_THINK)] + "\n" + _RETRY_NUDGE + _NO_THINK
    return user_content + "\n" + _RETRY_NUDGE


def is_degenerate(content: Any) -> bool:
    """Whether a managed-local reply is an echo or empty rather than an answer.

    Empty counts: a model that returned nothing usable has not answered, and the
    caller should retry before the guard degrades it. Echo detection reuses the
    scope guard's own predicate and adds the truncated-envelope opener the guard
    misses on its own - a reply that begins ``{"question":`` is the envelope
    reflected even when the model ran out of output budget before reproducing
    four of its keys (the live "are there any issues?" leak).
    """
    text = str(content or "").strip()
    if not text:
        return True
    if echoes_prompt_envelope(text):
        return True
    stripped = text.lstrip()
    return stripped.startswith('{"question"') or stripped.startswith("{'question'")


def _strip_code_fence(text: str) -> str:
    stripped = text.strip()
    if stripped.startswith("```"):
        stripped = stripped.split("\n", 1)[-1] if "\n" in stripped else ""
        if stripped.rstrip().endswith("```"):
            stripped = stripped.rstrip()[:-3]
    return stripped.strip()


def _parse_object(text: str) -> Any:
    """The object ``text`` is, or the one embedded in it, or ``None``.

    Both quote styles are tried (``json`` then ``ast.literal_eval``) so a
    single-quoted Python ``repr`` humanizes too, matching how the scope guard
    parses a leak.
    """
    candidates = [text]
    start, end = text.find("{"), text.rfind("}")
    if 0 <= start < end:
        candidates.append(text[start : end + 1])
    for candidate in candidates:
        for loader in (json.loads, ast.literal_eval):
            try:
                return loader(candidate)
            except (ValueError, SyntaxError, TypeError, MemoryError, RecursionError):
                continue
    return None


def _render_value(value: Any) -> str:
    if isinstance(value, str):
        return value.strip()
    if isinstance(value, bool):
        return "yes" if value else "no"
    if isinstance(value, (int, float)):
        return str(value)
    if isinstance(value, list):
        return ", ".join(_render_value(item) for item in value if _render_value(item))
    if isinstance(value, dict):
        return "; ".join(
            "{}: {}".format(key, _render_value(item))
            for key, item in value.items()
            if _render_value(item)
        )
    return str(value).strip()


def _render_dict(parsed: Dict[str, Any]) -> str:
    """A readable sentence or two from a JSON-shaped reply.

    A lead key carries the answer sentence; the detail lists expand it. When the
    reply has neither - a bare ``{"cpu_temperature": 40.8, ...}`` reading - the
    fields are rendered as ``key: value`` prose rather than dropped, so nothing
    the model actually said is lost.
    """
    pieces = []
    for key in _LEAD_KEYS:
        rendered = _render_value(parsed.get(key)) if key in parsed else ""
        if rendered:
            pieces.append(rendered if rendered.endswith((".", "!", "?")) else rendered + ".")
            break
    for key in _DETAIL_KEYS:
        if key in parsed:
            rendered = _render_value(parsed.get(key))
            if rendered:
                pieces.append(rendered if rendered.endswith((".", "!", "?")) else rendered + ".")
    if pieces:
        return " ".join(pieces)
    # No named answer field: render the whole object as readable pairs.
    pairs = [
        "{}: {}".format(str(key).replace("_", " "), _render_value(value))
        for key, value in parsed.items()
        if _render_value(value)
    ]
    return "; ".join(pairs)


def local_answer(request_body, headers, timeout, connection, call) -> Dict[str, Any]:
    """One managed-local answer: call, retry once if degenerate, humanize.

    ``call`` is :func:`vaelor.inference_client.chat_completion`, passed in rather
    than imported so this module stays free of the client's import graph. The
    retry mutates the user turn in place - it is the same logical generation, one
    more try - and never fires when the first reply already answered, which is
    the measured common case.
    """
    body = call(connection, request_body, headers, timeout)
    content = first_choice_content(body)
    if is_degenerate(content):
        message = request_body["messages"][-1]
        message["content"] = with_retry_nudge(message["content"])
        body = call(connection, request_body, headers, timeout)
        content = first_choice_content(body)
    # The raw body carries the wall-clock and usage timings the client attached
    # (inference_client.PERFORMANCE_KEY); pass it on so the Assistant can show
    # the same compact performance line AI Chat does. Empty when unreported.
    performance = body.get("performance", {}) if isinstance(body, dict) else {}
    if is_degenerate(content):
        # Still echoed (or empty) after the retry. Hand the raw text on so the
        # scope guard recognises the envelope and degrades it - humanizing an
        # echo would flatten its keys into prose and slip it past that guard.
        return {"answer": str(content or "").strip(), "performance": performance}
    return {"answer": humanize_answer(content), "performance": performance}


def _salvage_truncated(text: str) -> str:
    """The lead answer sentence from an unterminated JSON reply, or ``""``.

    The 4B's output budget is finite and it favours a verbose JSON reply, so a
    real answer is routinely cut off before its closing brace - unparseable, but
    the ``"summary": "..."`` sentence at the front is intact and is the answer.
    Read with string scanning rather than a regex so this module adds no
    module-level matching pattern (test_vocabulary_reachability). Both quote
    styles; the value runs to its closing quote, or to end-of-text when the
    model stopped mid-sentence.
    """
    for key in _LEAD_KEYS:
        for quote in ('"', "'"):
            marker = quote + key + quote
            head = text.find(marker)
            if head < 0:
                continue
            after = text[head + len(marker):].lstrip()
            if not after.startswith(":"):
                continue
            after = after[1:].lstrip()
            if not after or after[0] not in "\"'":
                continue
            value_quote = after[0]
            body = after[1:]
            end = body.find(value_quote)
            value = (body if end < 0 else body[:end]).replace('\\"', '"').strip()
            if value:
                return value if value.endswith((".", "!", "?")) else value + "."
    return ""


def humanize_answer(content: Any) -> str:
    """A managed-local reply as plain text.

    A JSON-shaped reply (which the 4B favours even when asked for prose) is
    flattened to sentences. When the model ran out of budget mid-object the JSON
    will not parse, so the lead answer sentence is salvaged from the raw text
    rather than shown with its braces. Anything that is not JSON at all is
    returned as written, minus a code fence. The scope guard still runs on the
    result, so an echo the model wrapped in prose is caught downstream.
    """
    text = _strip_code_fence(str(content or "").strip())
    parsed = _parse_object(text)
    if isinstance(parsed, dict):
        rendered = _render_dict(parsed)
        if rendered:
            return rendered
    if text.lstrip().startswith(("{", "[")):
        salvaged = _salvage_truncated(text)
        if salvaged:
            return salvaged
    return text
