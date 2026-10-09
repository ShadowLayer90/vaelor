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
import functools
import json
from typing import Any, Dict, List, Tuple

from .assistant_memory_framing import MEMORY_INSTRUCTION_POLICY
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
    return local_turn(message, context, connection)[0]


def local_turn(
    message: str, context: Dict[str, Any], connection: Dict[str, str]
) -> Tuple[str, List[str]]:
    """The managed-local user turn, and the slugs of the guidance it carries.

    **Review B5.** This turn used to carry the question and the facts only.
    The administrator's saved notes, the earlier turns of the conversation
    and the matched skill guidance were all assembled by the route and then
    dropped here, so the on-device model never saw a pinned note and could not
    read the previous turn behind "and the NPU?" - while every answer claimed
    "Applied reviewed guidance". Each now travels in plain lines, inside the
    same budget `assistant_context` applies; notes keep their untrusted
    framing, and the slugs returned are the only guidance an answer may claim.
    """
    trimmed = assistant_context(message, context, connection)
    trimmed = trimmed if isinstance(trimmed, dict) else {}
    facts = trimmed.get("facts", {})
    parts = [str(message).strip()]
    if isinstance(facts, dict) and facts:
        lines = "\n".join(
            "- {}: {}".format(key, json.dumps(value, separators=(",", ":"), default=str))
            for key, value in facts.items()
        )
        parts.append("Current readings from this machine:\n" + lines)
    earlier = [item for item in trimmed.get("conversation") or [] if isinstance(item, dict)]
    if earlier and str(earlier[-1].get("content", "")).strip() == str(message).strip():
        earlier = earlier[:-1]
    if earlier:
        parts.append("Earlier in this conversation (oldest first):\n" + "\n".join(
            "- {}: {}".format("Owner" if item.get("role") == "user" else "Assistant",
                              " ".join(str(item.get("content", "")).split()))
            for item in earlier[-4:]))
    notes = [item for item in trimmed.get("memories") or [] if isinstance(item, dict) and item.get("content")]
    if notes:
        parts.append(
            "Notes saved on this appliance. {}\n".format(MEMORY_INSTRUCTION_POLICY)
            + "\n".join("- {}".format(" ".join(str(item["content"]).split())) for item in notes))
    guidance = [item for item in trimmed.get("guidance") or [] if isinstance(item, dict)]
    for item in guidance:
        parts.append("Reviewed guidance for this kind of question ({}, version {}):\n{}".format(
            item.get("name") or item.get("slug"), item.get("version"), str(item.get("guidance", "")).strip()))
    return "\n\n".join(parts) + _NO_THINK, [str(item.get("slug")) for item in guidance if item.get("slug")]


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


#: What a managed-local reply with no text is told as, once its retry also
#: came back without any (W5-D1). Raised as a ValueError so the Assistant's
#: answer guard states the model failure, as it does for every other reply it
#: cannot read, instead of showing an empty bubble as an answer.
NO_USABLE_REPLY = (
    "The on-device model gave no text that could be shown, twice (an empty "
    "reply, one cut off before it said anything, or only a list of internal "
    "names), so there is no answer to show. Ask again."
)


def reply_text(body: Any) -> Any:
    """The first choice's content, with a content-less reply read as no text.

    FastFlowLM can answer a plain question with a tool-call finish reason
    and one nameless tool call - and no ``content`` at all (W5-D1, recorded on
    the Z2). The Assistant offers no tools and reads no tool-call field (the
    native-tool-call rule, ``NativeToolCallingTests``): judged by its content
    alone, that is a reply with nothing in it, the same as an empty one, so it
    is retried, never shown. A body with no choices still raises, through
    :func:`first_choice_content`.
    """
    try:
        return first_choice_content(body)
    except ValueError:
        message = body["choices"][0].get("message") if isinstance(body, dict) and body.get("choices") else None
        if isinstance(message, dict):
            return ""
        raise


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
    content = reply_text(body)
    if is_degenerate(content) or not shown_text(content):
        message = request_body["messages"][-1]
        message["content"] = with_retry_nudge(message["content"])
        body = call(connection, request_body, headers, timeout)
        content = reply_text(body)
    if not str(content or "").strip() or (
        not is_degenerate(content) and not shown_text(content)
    ):
        # W5-D1, W6-2 (LESSONS 1, 8): nothing to show, twice - judged on the
        # text the owner would see, so a bare code fence or an unreadable
        # object opener is no answer either. Shown, it was an empty bubble.
        raise ValueError(NO_USABLE_REPLY)
    # The raw body carries the wall-clock and usage timings the client attached
    # (inference_client.PERFORMANCE_KEY); pass it on so the Assistant can show
    # the same compact performance line AI Chat does. Empty when unreported.
    performance = body.get("performance", {}) if isinstance(body, dict) else {}
    if is_degenerate(content):
        # Still echoed after the retry. Hand the raw text on so the
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


def shown_text(content: Any) -> str:
    """The text the owner would be shown for ``content``, or ``""`` if unusable.

    W6-2 (LESSONS 1, 8, 5). The line, after the fence is stripped and any JSON
    is rendered or salvaged (:func:`humanize_answer`): a reply is unusable when
    nothing is left, or when what is left is a JSON object or list CUT OFF AT
    THE END (:func:`truncated_json`), or when it is only short labels and the
    appliance's internal names outside a fenced excerpt
    (:func:`internal_names_only`, VD-205 L4). Everything else is an answer -
    prose that merely starts with ``[`` or ``{`` (a markdown link, ``[WARN]``,
    ``[1]``, ``{name}``) included; the first cut refused every one of them.
    """
    text = humanize_answer(content).strip()
    if truncated_json(text) or (internal_names_only(text) and not _is_excerpt(content)):
        return ""
    return text


@functools.lru_cache(maxsize=1)
def _internal_names() -> Tuple[frozenset, Tuple[str, ...]]:
    """The names the appliance gives its own tools, facts and engines, and their prefixes.

    VD-205 L4 (LESSONS 6): read from the owners - both tool registries,
    ``answer_evidence.EVIDENCE_SOURCES`` and its families, the managed serving
    and pull name prefixes, and the on-device engine id - so a new tool is
    recognised without a second list here. Imported when first needed: the
    registries import far more than this module does.

    The control-plane systemd units are deliberately NOT here. The model reads
    them in ``services.status``, so "Failed units: vaelor-workload-executor"
    can be the true answer to the owner's question, not an echo of a tool id.
    """
    from .answer_evidence import EVIDENCE_SOURCE_FAMILIES, EVIDENCE_SOURCES
    from .assistant_acting_tools import AssistantActingToolRegistry
    from .assistant_machine_tools import LOCAL_ENGINE_ID
    from .assistant_tools import AssistantToolRegistry
    from .gpu_pool_units import MANAGED_NAME_PREFIXES

    exact = {*EVIDENCE_SOURCES, *AssistantToolRegistry().names(),
             *AssistantActingToolRegistry().names(), LOCAL_ENGINE_ID}
    return (frozenset(name.lower() for name in exact),
            tuple(EVIDENCE_SOURCE_FAMILIES) + tuple(MANAGED_NAME_PREFIXES))


#: Characters an internal name is spelled with; anything else is prose.
_NAME_CHARACTERS = frozenset("abcdefghijklmnopqrstuvwxyz0123456789.-_/")

#: What joins the parts of an id - ``metrics.history``, ``vaelor-phoenix``,
#: ``memory_status``. A token without one is a word, not an id.
_ID_JOINERS = ".-_"


def _bare_token(token: str) -> str:
    """``token`` without the quotes, brackets, backticks and emphasis around it."""
    return token.strip().strip("`'\"[](){}*_").rstrip(".").strip("`'\"*_")


def _is_internal_name(token: str) -> bool:
    name = _bare_token(token).lower()
    if not name or not set(name) <= _NAME_CHARACTERS:
        return False
    exact, prefixes = _internal_names()
    return name in exact or any(
        name.startswith(prefix) and len(name) > len(prefix) for prefix in prefixes)


def _is_id_shaped(token: str) -> bool:
    """Lowercase, no spaces, and joined by a dot, hyphen or underscore: ``gpu.status``.

    ``jellyfin`` and ``Docker`` are words; ``system.thermal`` and
    ``vaelor-phoenix`` are shaped like ids whether or not anything registers them.
    """
    name = _bare_token(token)
    return bool(name) and set(name) <= _NAME_CHARACTERS and any(
        joiner in name for joiner in _ID_JOINERS)


def _is_label(text: str) -> bool:
    """A short field name - ``sources``, ``engines``, ``evidence used`` - not a clause."""
    words = text.replace("_", " ").split()
    return 0 < len(words) <= 3 and len(text) <= 32 and all(word.isalnum() for word in words)


#: Labels that only cite where an answer came from. Under one of these, a list
#: of id-shaped entries is a citation of tools or facts whether or not each id
#: is registered - the 4B invents them (``system.thermal.status``).
_CITATION_LABELS = frozenset((
    "source", "sources", "sources used", "evidence", "evidence used",
    "tool", "tools", "tools used",
))


def _is_registered_name(token: str) -> bool:
    """An exact registered tool, evidence or engine id - not a managed-name prefix match."""
    return _bare_token(token).lower() in _internal_names()[0]


def internal_names_only(text: str) -> bool:
    """Whether ``text`` is nothing but short labels over lists of the appliance's ids.

    VD-205 L4 (LESSONS 1, 8). Asked "Did Docker crash last night?", the 4B
    replied ``sources: workloads.inventory, metrics.history``; earlier it
    replied ``engines: assistant-local, vaelor-vllm-...`` (L1). Each is the ids
    of what it was shown, not an answer, and each passed the empty-or-cut-off
    judge. The Z2 measurement then showed the same shape with ids beside the
    known ones - ``engines: assistant-local, vaelor-llm-proxy, vaelor-phoenix,
    ...`` and the invented ``sources: system.thermal, gpu.status``.

    The reply is read as parts: a labelled line with the unlabelled lines
    under it, and any lines before the first label as one more; a label with
    nothing under it is dropped. True when every remaining part is a comma list
    of entries that are each an internal name (:func:`_internal_names`) or
    id-shaped (:func:`_is_id_shaped`), AND the part is anchored as the
    appliance's own: every entry is an internal name, or one entry is an EXACT
    registered id (:func:`_is_registered_name`), or its label only cites
    (:data:`_CITATION_LABELS`). A managed serving prefix alone is no anchor, so
    the owner's app list ``Running apps: system-web-research-search-1,
    vaelor-llm-proxy, ..., vaelor-vllm-gpu-model-server`` is an answer, and so
    is "Failed units: vaelor-workload-executor" or "Failed units:
    docker.service" - nothing in them is the appliance's own id. Markdown a 4B
    adds is read through: a bullet or ``1.`` marker, a ``**bold**`` label, and
    a label on a line of its own above its list. One word of prose anywhere -
    "Docker did not crash: no restart was recorded." - and it is an answer.
    """
    parts: List[Tuple[str, List[str]]] = []
    for line in text.replace(";", "\n").splitlines():
        line = _without_list_marker(line.strip().lstrip("-*").strip())
        if not line:
            continue
        label, colon, rest = line.partition(":")
        label, rest = label.strip().strip("*_").strip(), rest.strip().strip("*_").strip()
        labelled = bool(colon) and _is_label(label)
        if labelled or not parts:
            parts.append((" ".join(label.replace("_", " ").lower().split()) if labelled else "", []))
        if labelled and not rest:
            continue  # "sources:" alone, its list on the lines below
        parts[-1][1].extend((rest if labelled else line).split(","))
    parts = [(label, entries) for label, entries in parts if entries]
    if not parts:
        return False
    for label, entries in parts:
        if not all(_is_internal_name(entry) or _is_id_shaped(entry) for entry in entries):
            return False
        if not (label in _CITATION_LABELS
                or all(_is_internal_name(entry) for entry in entries)
                or any(_is_registered_name(entry) for entry in entries)):
            return False
    return True


def _without_list_marker(line: str) -> str:
    """``line`` without a leading ``1.`` or ``1)`` numbered-list marker."""
    head, space, tail = line.partition(" ")
    if space and len(head) > 1 and head[:-1].isdigit() and head[-1] in ".)":
        return tail.strip()
    return line


def _is_excerpt(content: Any) -> bool:
    """Whether the raw reply fences a block that is not JSON - code or a log.

    An excerpt the owner asked for may be nothing but names; a fence tagged
    ``json``, or holding a JSON object or list, is the 4B's usual reply shape
    and is judged like any other.
    """
    raw = str(content or "")
    fence = raw.find("```")
    if fence < 0:
        return False
    tag = raw[fence + 3:].split("\n", 1)[0].strip().lower()
    return tag != "json" and not isinstance(_parse_object(_strip_code_fence(raw)), (dict, list))


def truncated_json(text: str) -> bool:
    """Whether ``text`` is a JSON object or list the reply stopped inside.

    True only when a JSON reader, starting at the opening ``{`` or ``[``, runs
    off the END of the text, or into a string that never closes - the shape a
    budget-cut ``{"summary": "``, ``[{"a":`` or ``{"`` has. A value that closes
    (``[1] Docker is running.``) or text a JSON reader rejects before the end
    (``[WARN] ...``, ``{name} ...``, ``[Cooling](#/...)``) is not.
    """
    if not text.startswith(("{", "[")):
        return False
    try:
        json.JSONDecoder().raw_decode(text)
    except json.JSONDecodeError as error:
        return error.msg.startswith("Unterminated string") or error.pos >= len(text)
    return False


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
