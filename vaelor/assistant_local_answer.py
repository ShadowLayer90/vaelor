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

**A tool call is never re-sent (VD-218, kb R120).** Measured on the Z2
2026-10-16 with the product's own request: "Hi" came back as one nameless tool
call - ~15 s of hidden generation, no text - in 6 of 7 asks, and a question with
no readings in 3 of 3. The full re-send recovered 2 of 4 at a cost of 11-25 s
more, and a retry with a much smaller prompt (no machine brief) recovered 0 of
5 - all ten of those calls were tool calls too. So a reply that is only a tool
call ends the model's part at once: :class:`UnusableReply` says so and the
Assistant answers from the readings it already gathered. Every call an ask
makes is counted in its timing line (:func:`combined_performance`).

**The NPU fine-tune is asked in the contract it was trained on (VD-219,
LESSONS 26).** ff-4b-h learned one reply shape - a JSON object with summary,
findings, recommendations, next_actions, and sources last - and never saw the
Assistant's "plain text" prompt. Asked for plain text it wrote a half-remembered
version of that shape: of 44 real questions on the Z2, 14 cut off, 10
malformed, 3 a copied reading and 2 a sources list with no answer. The model is
now asked in its contract (:data:`CONTRACT_MODELS`), and :func:`contract_text`
reads the reply - whole, cut off or malformed - into plain lines: the summary,
then the details, never a key, a brace or the sources (those become evidence,
:func:`cited_evidence`). A reply that only cites or only copies a reading is
asked once more with its answer opened for it (:data:`SUMMARY_PRIMER`).
"""

from __future__ import annotations

import ast
import functools
import json
import logging
from typing import Any, Dict, List, Tuple

from .assistant_contract_reader import (
    CITATION_KEYS,
    DETAIL_KEYS,
    LEAD_KEYS,
    cited_evidence,
    cites_only,
    contract_text,
    contract_values,
    copies_reading,
    is_lead_key,
)
from .assistant_memory_framing import MEMORY_INSTRUCTION_POLICY
from .assistant_scope_guard import echoes_prompt_envelope
from .provider_runtime import assistant_context

_LOGGER = logging.getLogger(__name__)

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

#: The keys a reply answers, details and cites with live with the reader.
_LEAD_KEYS, _DETAIL_KEYS, _CITATION_KEYS = LEAD_KEYS, DETAIL_KEYS, CITATION_KEYS

#: VD-219 / R126: the opening of the one re-ask's reply. The first key the
#: fine-tune writes decides the answer: primed with this, 5 of 5 sources-only
#: asks answered; primed with a sources list, a question that normally answers
#: wrote only sources, 2 of 2. Sent as the assistant turn, with no nudge.
SUMMARY_PRIMER = '{"summary":"'

#: VD-219: the room a contract answer is given. Not a cap (the owner rejected
#: capping answers, D176): 14 of 44 real answers ran into the old 320.
#: PROVISIONAL until measured on the Z2 against 512 and 1024.
CONTRACT_ANSWER_TOKENS = 768

def asks_in_contract(connection: Dict[str, str], model: str = "") -> bool:
    """Whether this connection is the NPU fine-tune, asked in its trained JSON contract.

    VD-219 / R139 item 5 (LESSONS 5): keyed on what the model IS, not on its
    tag - FastFlowLM's stock model and other runtimes also serve ``qwen3.5:4b``.
    It is the Assistant's on-device NPU credential (the id marker the vault
    writes) serving the catalog's NPU release (``model_catalog.NPU_RELEASE_MODELS``,
    whose release tag follows a rename). The marker with another model is said
    in the log, not silently asked in the plain prompt.
    """
    from .model_catalog import NPU_RELEASE_MODELS
    from .model_credential_roles import ASSISTANT_NPU_PREFIX

    if not str(connection.get("credential_id") or "").startswith(ASSISTANT_NPU_PREFIX):
        return False
    tags = {str(item.get("release_tag") or "").lower() for item in NPU_RELEASE_MODELS
            if item.get("surface") == "assistant"}
    served = str(model or connection.get("model", "")).strip().lower()
    if served not in tags:
        _LOGGER.warning("The Assistant's NPU model %r is not the catalog's fine-tune %s; "
                        "it is asked in the plain-text prompt (VD-219).", served, sorted(tags))
        return False
    return True


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
            "- {}: {}".format(key, value if isinstance(value, str)
                              else json.dumps(value, separators=(",", ":"), default=str, ensure_ascii=False))
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


class UnusableReply(ValueError):
    """No text the owner could be shown, with the timing of every call that tried.

    A ``ValueError`` so the Assistant's answer guard degrades it as it does
    every other unreadable reply; ``performance`` lets that degraded answer still
    say how long the model spent (VD-218: the timing line counts every call).
    """

    def __init__(self, message: str, performance: Dict[str, Any]) -> None:
        super().__init__(message)
        self.performance = performance


#: VD-218 / R120: the reply was a tool call with no name and no text. The
#: Assistant offers no tools, so this is the model not answering - said once,
#: never re-sent (the measured re-send recovered half, at double the wait).
EMPTY_TOOL_CALL_REPLY = (
    "The on-device model started a tool call instead of answering, and the "
    "Assistant gives it no tools, so it wrote no text to show."
)

#: VD-218: the reply named readings to fetch instead of answering. Every
#: reading this question selected was already gathered and sent, so the answer
#: comes from those, not from a second ask.
NAMED_TOOL_CALL_REPLY = (
    "The on-device model asked to read {} instead of answering from the "
    "readings it was sent, so it wrote no text to show."
)


def tool_call_names(body: Any) -> List[str]:
    """The first choice's tool-call function names; ``[""]`` for a nameless one.

    FastFlowLM reports a call it parsed from the model's own output as
    ``finish_reason: tool_calls`` - sometimes with the call list, and its name
    empty (R120). Either signal counts; an ordinary reply yields ``[]``.
    """
    choices = body.get("choices") if isinstance(body, dict) else None
    choice = choices[0] if isinstance(choices, list) and choices and isinstance(choices[0], dict) else {}
    message = choice.get("message") if isinstance(choice.get("message"), dict) else {}
    calls = message.get("tool_calls") or []
    names = [str((call.get("function") or {}).get("name") or "").strip()
             for call in calls if isinstance(call, dict)]
    if not names and choice.get("finish_reason") == "tool_calls":
        return [""]
    return names


def combined_performance(timings: List[Dict[str, Any]]) -> Dict[str, Any]:
    """One timing line for an ask that called the model ``len(timings)`` times.

    VD-218 / R120: the line used to carry the LAST call only, so a re-sent ask
    that took 26 s said 11 s. ``total_seconds`` is now every call's wall clock
    summed and ``model_calls`` says how many there were; the rates and the
    time to first token are the last call's, the one whose text is shown.
    """
    if len(timings) <= 1:
        return dict(timings[0]) if timings else {}
    last = next((item for item in reversed(timings) if item), {})
    combined: Dict[str, Any] = {
        key: last[key] for key in ("ttft_seconds", "prefill_tps", "decode_tps") if key in last
    }
    combined["total_seconds"] = round(sum(
        float(item.get("total_seconds") or 0) for item in timings), 3)
    combined["model_calls"] = len(timings)
    return combined


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
    """A readable sentence or two from a JSON-shaped reply with no answer the reader found.

    The citation keys are never shown (R139). A value that is a sentence, or a
    list of them, is shown as written, without its key; a bare reading - a
    ``{"cpu_temperature": 40.8}`` number or word - keeps its field name, said
    as words, because the value means nothing without it.
    """
    pieces = []
    for key, value in parsed.items():
        rendered = _render_value(value)
        if not rendered or str(key).strip().lower() in _CITATION_KEYS:
            continue
        sentences = isinstance(value, list) or (isinstance(value, str) and len(value.split()) > 2)
        pieces.append(rendered if sentences else "{}: {}".format(
            str(key).replace("_", " ").strip(), rendered))
    return "; ".join(pieces)


def _continues_primer(content: Any) -> bool:
    """Whether a primed reply is the rest of the opened summary, not a reply of its own.

    A server that continues the assistant turn returns ``Docker is idle.",...``:
    no opener of its own, and the summary's closing quote inside.
    """
    text = str(content or "").lstrip()
    return bool(text) and text[0] not in '{["`' and ('",' in text or text.rstrip().endswith('"}'))


def needs_primer(content: Any, sent: str) -> bool:
    """Whether a reply only cites (R126) or only copies a reading (R130) - asked again, primed."""
    return cites_only(content) or copies_reading(content, sent)


def local_answer(request_body, headers, timeout, connection, call, *, restart=None,
                 contract=False) -> Dict[str, Any]:
    """One managed-local answer: call, retry once if degenerate, humanize.

    ``call`` is :func:`vaelor.inference_client.chat_completion`, or a streamed
    call with the same signature (:mod:`vaelor.assistant_stream`), passed in
    rather than imported so this module stays free of the client's import graph.
    The retry is the same logical generation, one more try, and never fires when
    the first reply already answered, which is the measured common case, nor when
    the reply was a tool call (VD-218). ``restart`` runs before a retry, so text
    a streamed first try already put on the page is taken back.

    VD-219: the retry opens the model's reply for it (:data:`SUMMARY_PRIMER`,
    no nudge) when the model was asked in its trained ``contract``, or when the
    reply only cited or only copied a reading; otherwise it carries the
    plain-sentences nudge as before. The cited readings come back as evidence.
    """
    timings: List[Dict[str, Any]] = []
    sent = str(request_body["messages"][-1].get("content") or "")

    def ask() -> Any:
        body = call(connection, request_body, headers, timeout)
        timings.append(body.get("performance") or {} if isinstance(body, dict) else {})
        content = reply_text(body)
        names = tool_call_names(body)
        if names and not str(content or "").strip():
            # VD-218 / R120 (LESSONS 1, 16): never re-sent. Named or not, the
            # readings this question selected were already sent; a second ask
            # recovered half the time at double the wait.
            named = [name for name in names if name]
            raise UnusableReply(
                NAMED_TOOL_CALL_REPLY.format(", ".join(named)) if named else EMPTY_TOOL_CALL_REPLY,
                combined_performance(timings))
        return content

    content = ask()
    primed = needs_primer(content, sent)
    if primed or is_degenerate(content) or not shown_text(content):
        if restart is not None:
            restart()
        if contract or primed:
            # VD-219 / R126: the first key decides the answer; open it.
            request_body["messages"].append({"role": "assistant", "content": SUMMARY_PRIMER})
            content = ask()
            if _continues_primer(content):
                content = SUMMARY_PRIMER + str(content)
        else:
            message = request_body["messages"][-1]
            message["content"] = with_retry_nudge(message["content"])
            content = ask()
    # VD-218: the line counts every call the ask made, not the last one.
    performance = combined_performance(timings)
    if not str(content or "").strip() or needs_primer(content, sent) or (
        not is_degenerate(content) and not shown_text(content)
    ):
        # W5-D1, W6-2, VD-219 (LESSONS 1, 8): nothing to show, twice - judged
        # on the text the owner would see, so a bare code fence, an unreadable
        # object opener, a sources list or a copied reading is no answer either.
        raise UnusableReply(NO_USABLE_REPLY, performance)
    if is_degenerate(content):
        # Still echoed after the retry. Hand the raw text on so the
        # scope guard recognises the envelope and degrades it - humanizing an
        # echo would flatten its keys into prose and slip it past that guard.
        return {"answer": str(content or "").strip(), "performance": performance}
    return {"answer": humanize_answer(content), "performance": performance,
            "evidence": cited_evidence(content, sent)}


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
    if truncated_json(text) or cites_only(content) or (
            internal_names_only(text) and not _is_excerpt(content)):
        # VD-219 / R126: a sources list is no answer, whatever its entries say.
        return ""
    if text.startswith(("{", "[")) and (contract_values(text) or '":' in text or "':" in text):
        # VD-219 / R139: still an object with keys after every reading - never shown raw.
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
    read with :func:`contract_text` - whole, cut off or malformed - into its
    summary and detail lines, finished sentences only (VD-219). An object with
    no answer key, such as a bare reading, is rendered as readable pairs.
    Anything that is not JSON at all is returned as written, minus a code fence;
    prose before an object is kept, and the object is read only when it is the
    reply's own tail - one that runs on into more prose is an example the
    answer quotes (R139). The scope guard still runs on the result, so an echo
    the model wrapped in prose is caught downstream.
    """
    text = _strip_code_fence(str(content or "").strip())
    start = text.find("[") if text.startswith("[") else text.find("{")
    if start < 0:
        return text
    prose, tail = text[:start].strip(), text[start:]
    if prose:
        try:
            _value, end = json.JSONDecoder().raw_decode(tail)
        except ValueError:
            return text
        if tail[end:].strip():
            return text
    rendered = contract_text(tail[1:].lstrip() if tail[1:].lstrip().startswith("{") else tail)
    if not rendered:
        parsed = _parse_object(tail)
        rendered = _render_dict(parsed) if isinstance(parsed, dict) else ""
    if not rendered:
        return text
    return prose + "\n" + rendered if prose else rendered
