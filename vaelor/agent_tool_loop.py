"""A native OpenAI function-calling loop for user-defined agents.

The owner's directive: "Agents are actual agents with tool calls etc. The model
should be allowed to call tools if required as part of the agents." Until now the
runner PRE-FETCHED research and appliance facts on the model's behalf, because
the managed-local 4B was believed to have no usable native tool calling. A live
probe of the FLM (fastflowlm) endpoint corrected that: fed tools as free text in
the system prompt the 4B returned an EMPTY call (``name:""``), but fed the
tools the OpenAI way - the native ``tools=[{"type":"function",...}]`` request
field - it returned a CORRECT call (``web_search`` with a real query). The
endpoint has its own tool template, so this loop feeds tools natively and reads
``message.tool_calls`` back, rather than parsing Hermes ``<tool_call>`` text.

What this module owns:

* ``build_tool_specifications`` renders the agent's GRANTED, READ-ONLY tools as
  OpenAI function specs. A mutating tool, an approval-gated tool, a tool whose
  scope the agent was not granted, and (with web off) a ``research:*`` tool are
  all excluded. This is the ONLY set the model may call.
* ``parse_tool_calls`` reads ``message.tool_calls`` from a response. A call with
  an empty name - the exact failure the text-prompt form produced - yields
  nothing, and malformed argument JSON degrades to an empty object rather than a
  new repair pass.
* ``run_agent_tool_loop`` is the bounded loop: the model calls tools, each call
  is RE-VALIDATED against the granted read-only set before it runs (the model is
  never trusted), executed through the guarded ``tools.run`` path, and its
  result is returned as a bounded ``tool`` message. When the model stops calling
  tools it answers, and that answer is parsed and put through the same #247w
  honest-degradation policy the one-shot pass uses.
* ``run_custom_agent_synthesis`` is the runner-facing seam that PREFERS the loop
  and FALLS BACK to the existing pre-fetch + a83 one-shot path if the endpoint
  yields no usable tool call or the loop errors, so an agent is never worse than
  a83. The path taken is recorded for the capability audit.

The security envelope is unchanged: every executed call is re-checked against the
agent's grants, mutating tools are never in the spec, guarded ``research.search``
still enforces the allowlist and ``research.fetch`` still enforces provenance
(this loop only ever lets the model fetch a URL its OWN guarded search returned).
"""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass, replace
from datetime import datetime, timezone
from typing import Any, Dict, List, Mapping, Optional

from .agent_result_shape import validate_specialist
from .agent_search_planner import PUBLIC_RESEARCH_SOURCE_TITLE, _resolve_model
from .assistant_tools import AssistantToolError, _safe_value
from .custom_agent_model import (
    AGENT_VISIBLE_TOKENS,
    apply_grounding_policy,
    build_agent_system_prompt,
)
from .inference_client import allowed_inference_endpoint, chat_completion, inference_timeout
from .model_profiles import reasoning_headroom_tokens
from .research_json import extract_json_document
from .provider_runtime import (
    agent_context_chars,
    generation_parameters,
    provider_context,
    provider_system_prompt,
    provider_user_content,
)

LOGGER = logging.getLogger(__name__)

# Tight on purpose: the appliance's NPU serves one generation at a time, so an
# unbounded chain of model calls would monopolise it. Five turns is enough for a
# search, a fetch, and an answer, and the loop refuses to run past it.
MAX_ITERATIONS = 5

# Per-turn ceilings that bound the growing message list independently of the
# model's behaviour: at most this many calls are honoured per turn, each tool
# result is truncated to this many characters, and the whole conversation is
# capped at this many messages (whereupon the loop forces a final synthesis).
MAX_CALLS_PER_ITERATION = 3
MAX_TOOL_RESPONSE_CHARS = 2400
MAX_MESSAGES = 32

# The capable (GPU) tier reasons and fetches far more thoroughly than the NPU -
# on the live MLB run the 27B fetched three separate days from the MLB Stats API
# and ran out of the NPU-tuned budget WHILE STILL GATHERING. It is not
# single-slot-constrained the way the NPU is, so it gets a larger budget: more
# turns, more calls per turn, more of each large API response kept for synthesis,
# and more room to write the final answer. The NPU caps above are untouched.
CAPABLE_MAX_ITERATIONS = 8
CAPABLE_MAX_CALLS_PER_ITERATION = 5
CAPABLE_MAX_TOOL_RESPONSE_CHARS = 6000
CAPABLE_MAX_MESSAGES = 64
CAPABLE_VISIBLE_TOKENS = 1200


@dataclass(frozen=True)
class _LoopBudget:
    """The bounds one tool-loop run holds itself to, chosen by model tier."""

    iterations: int
    calls_per_turn: int
    tool_response_chars: int
    max_messages: int
    visible_tokens: int


def _loop_budget(capable: bool, max_iterations: Optional[int]) -> _LoopBudget:
    """The NPU caps by default, the larger capable-tier caps on the GPU 27B.

    ``max_iterations``, when a caller passes it explicitly (a test, or a future
    override), still wins for the iteration count so boundedness stays testable.
    """
    budget = (
        _LoopBudget(CAPABLE_MAX_ITERATIONS, CAPABLE_MAX_CALLS_PER_ITERATION,
                    CAPABLE_MAX_TOOL_RESPONSE_CHARS, CAPABLE_MAX_MESSAGES,
                    CAPABLE_VISIBLE_TOKENS)
        if capable else
        _LoopBudget(MAX_ITERATIONS, MAX_CALLS_PER_ITERATION,
                    MAX_TOOL_RESPONSE_CHARS, MAX_MESSAGES, AGENT_VISIBLE_TOKENS)
    )
    if max_iterations is not None:
        budget = replace(budget, iterations=max(1, int(max_iterations)))
    return budget

# The tool-use contract, appended to the persona/output-contract system prompt.
# Steering, not just permission: the 4B was observed answering a research task
# from its own memory and never calling a tool, because the persona prompt is
# dominated by "return JSON with keys ...". This makes tool-use the DEFAULT for
# anything it cannot already verify, while still allowing an honest "the tools
# could not find it" after a genuine attempt (#247w preserved). It is generic -
# it names no topic - so it holds for any task, not only web research. The final
# answer is still distinguished by carrying NO tool call.
TOOL_LOOP_INSTRUCTIONS = (
    "You have tools for this run. For anything factual, current, or "
    "time-sensitive that you cannot already verify with certainty, you MUST call "
    "a tool to gather evidence BEFORE you answer - do not answer from your own "
    "memory or training when a tool could check it. Call a tool whenever you need "
    "information you were not already given; you will receive its result and may "
    "then call another tool or answer. A web search returns only a list of links "
    "and titles - that is not the answer: when a search names a source that "
    "should hold what you need, OPEN it with a fetch tool and read its content "
    "before you conclude. Do not answer from search titles alone. Produce the final JSON answer object "
    "described above ONLY after your tools have given you what you need, or after "
    "you have genuinely tried and the tools could not find it. A final answer is "
    "a message with NO tool call; a tool call is how you gather evidence first."
)

# The one-time nudge (see the loop): sent as a user turn when the model answered
# on turn one without calling a tool. It fires at most once, so the loop stays
# bounded; if the model still will not call a tool, the run falls back.
TOOL_LOOP_NUDGE = (
    "Use your tools to gather evidence before answering. Call a tool now to check "
    "the facts this task needs; do not answer from memory. If a tool genuinely "
    "cannot find what is needed, then say so honestly in the final answer."
)

# The one-time reprompt when the model's final answer would not parse as JSON.
# The 4B intermittently wraps the object in prose/a fence or truncates it; the
# tolerant extractor recovers most of those, and this re-ask (with no tools, so
# it must answer) recovers the rest without discarding the tool work already done.
FINAL_JSON_RETRY = (
    "Return ONLY the final JSON result object described above - no prose, no code "
    "fence, nothing before or after it."
)

# The forced final synthesis, sent (with tools removed) when the loop hit its
# turn/message budget while the model was STILL gathering and never wrote an
# answer. A thorough model that spent every tool turn fetching (the GPU 27B
# fetched three days of MLB data) must still get to write up what it found,
# rather than the run ending on an empty answer. This is the key fix for
# "reached the data but returned no formatted answer".
FORCED_SYNTHESIS = (
    "You have gathered enough evidence. Now produce the final JSON result object "
    "from what your tools have returned. Do NOT call any more tools; reply with "
    "ONLY the JSON object described above."
)


class AgentToolLoopFallback(Exception):
    """The native loop could not drive the model; use the pre-fetch fallback.

    Raised when the model never emits a usable tool call (it cannot follow the
    native tool template), when the loop exhausts its iteration or message
    budget, or when the endpoint returns nothing usable. The runner catches it -
    together with any transport or shape error - and runs the existing one-shot
    path instead, so the loop is strictly an improvement or a no-op.

    ``nudged`` records whether the one-time steering nudge fired before the run
    gave up, so the capability audit can show it even on the fallback path.
    """

    def __init__(self, message: str, *, nudged: bool = False) -> None:
        super().__init__(message)
        self.nudged = nudged


def current_date_utc() -> str:
    """Today's date as ``YYYY-MM-DD`` in UTC, for grounding the agent prompt.

    The 4B was observed treating a past date in the task as a future one ("...
    October 2023 or November 2024? (Need to clarify current date)"). The runner
    passes this real value into the agent context so the prompt can state it; it
    is never hardcoded.
    """
    return datetime.now(timezone.utc).strftime("%Y-%m-%d")


def build_tool_specifications(
    catalog: Any, granted_scopes: Any, web_access: Any,
) -> List[Dict[str, Any]]:
    """Render the agent's granted, read-only tools as OpenAI function specs.

    The exclusions are the security boundary of the whole loop: a tool that
    mutates or needs approval is never offered, a tool whose scope the agent was
    not granted is never offered, and a ``research:*`` tool is offered only when
    this run has web access. What survives is exactly what the model is allowed
    to call, and the loop re-checks each actual call against this same set.
    """
    granted = {str(scope) for scope in (granted_scopes or ())}
    web_on = bool((web_access or {}).get("enabled"))
    specs: List[Dict[str, Any]] = []
    for tool in catalog or []:
        if not isinstance(tool, Mapping):
            continue
        if tool.get("mutation") or tool.get("approval_required"):
            continue
        scope = str(tool.get("scope", ""))
        if scope not in granted:
            continue
        if scope == "research:read" and not web_on:
            continue
        name = str(tool.get("name", "")).strip()
        if not name:
            continue
        parameters = tool.get("parameters")
        if not isinstance(parameters, Mapping):
            parameters = {"type": "object", "properties": {}}
        specs.append({
            "type": "function",
            "function": {
                "name": name,
                "description": str(tool.get("description", ""))[:1024],
                "parameters": dict(parameters),
            },
        })
    return specs


def parse_tool_calls(message: Any) -> List[Dict[str, Any]]:
    """Extract ``{"id","name","arguments"}`` from a response's ``tool_calls``.

    Tolerates arguments given as a JSON string (the OpenAI wire form), as an
    object, or absent; a malformed argument string degrades to ``{}`` rather than
    invoking a new repair pass. A call with an empty name - the failure the
    text-prompt form produced - is dropped, so a model that cannot follow the
    tool template yields an empty list and the loop falls back.
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
            "name": name[:64],
            "arguments": parsed if isinstance(parsed, dict) else {},
        })
    return calls


def _bounded_content(outcome: Any, limit: int) -> str:
    """One tool result as a bounded JSON string for a ``tool`` message.

    ``limit`` is the tier's tool-response ceiling: larger on the capable tier so
    enough of a big API payload (a full day's MLB schedule) survives for the
    model to synthesise the scores from it.
    """
    result = outcome.get("result") if isinstance(outcome, Mapping) else outcome
    text = json.dumps(result, default=str, separators=(",", ":"))
    if len(text) > limit:
        return text[:limit] + "...[truncated]"
    return text


def _collect_vetted(outcome: Any, vetted: set) -> None:
    """Record the URLs a guarded search returned so a later fetch may open them.

    ``research.fetch`` refuses any URL that is neither on the agent's domain
    allowlist nor one this run's own guarded search returned. The loop threads
    those exact URLs into the fetch call's web-access policy, so the model can
    read what it searched without ever being trusted with an invented address.
    """
    result = outcome.get("result") if isinstance(outcome, Mapping) else {}
    results = result.get("results") if isinstance(result, Mapping) else None
    for item in results or []:
        if isinstance(item, Mapping):
            url = str(item.get("url") or "").strip()
            if url:
                vetted.add(url)


def _loop_context(connection: Mapping[str, str], context: Mapping[str, Any], generation_tokens: int) -> Any:
    """The granted context sent alongside the task, bounded to the window.

    The loop lets the model gather its own evidence, but the agent definition,
    policy, and any supplied notes still ground it. This is the same compaction
    the one-shot pass uses, so neither the prompt nor the model's own generation
    can overrun the context window.
    """
    return provider_context(
        dict(context),
        max_chars=agent_context_chars(dict(connection), generation_tokens=generation_tokens),
    )


def _message_from_body(body: Any) -> Mapping[str, Any]:
    """The assistant message from a completion body, or ``{}`` when unusable."""
    if not isinstance(body, Mapping):
        return {}
    choices = body.get("choices")
    first = choices[0] if isinstance(choices, list) and choices else {}
    return (first.get("message") if isinstance(first, Mapping) else {}) or {}


def _content_text(content: Any) -> str:
    """A model message's content as text (a dict/list is JSON-encoded)."""
    if isinstance(content, str):
        return content
    if isinstance(content, (dict, list)):
        return json.dumps(content, default=str)
    return str(content or "")


def _parse_or_none(content: Any) -> Optional[Dict[str, Any]]:
    """Tolerantly extract a JSON object, or ``None`` when nothing parses.

    Uses the codebase's fence-strip + balanced-scan extractor, so an intermittent
    4B slip (prose or a code fence around the object) is recovered rather than
    thrown. This is extraction, not content-repair.
    """
    try:
        parsed = extract_json_document(content)
    except ValueError:
        return None
    return parsed if isinstance(parsed, dict) else None


def _honest_result_from_tool_work(sources: List[Dict[str, Any]]) -> Dict[str, Any]:
    """An honest result when the final JSON would not parse even after a retry.

    The tools DID run; their sources are preserved and shown. There are no
    fabricated findings - the run says plainly it gathered evidence but could not
    format an answer, so no view presents an invented result while the real
    evidence stands. This keeps the run on the native path with its tool work.
    """
    return {
        "summary": "Gathered evidence with tools, but the model's final answer "
                   "could not be read as a result. The sources it retrieved are listed.",
        "findings": [], "recommendations": [], "next_actions": [],
        "sources": list(sources or []),
        "warnings": ["The agent's final answer was not valid JSON after a retry; "
                     "the tools ran and their sources are preserved below."],
    }


def run_agent_tool_loop(
    connection: Mapping[str, str], task: str, definition: Mapping[str, Any],
    context: Mapping[str, Any], timeout_seconds: int, *,
    tools: Any, actor: Optional[str], administrator: bool,
    granted_scopes: Any, web_access: Any, capable: bool = False,
    max_iterations: Optional[int] = None,
) -> tuple[Dict[str, Any], List[Dict[str, Any]], List[Dict[str, Any]], bool]:
    """Drive the model through a bounded native tool-calling loop.

    Returns ``(result, invocations, sources, nudged)``: the parsed, honest-
    degradation-checked answer; the bounded audit of every call the model made
    (name, redacted arguments, and whether it was executed, refused as
    out-of-grant, or errored); the pages the model actually fetched, as
    ``{source, summary}`` entries, so the runner surfaces the LOOP's evidence
    rather than a pre-fetch's; and whether the one-time steering nudge fired.

    ``capable`` selects the larger GPU-tier budget (more turns, calls, kept API
    text, and answer room) - the NPU keeps the tight single-slot caps. If the
    model answers turn one WITHOUT a tool it is nudged ONCE. If it spends the
    whole budget still gathering and never writes an answer, ONE tools-free
    synthesis turn is forced so a thorough model always gets to write up what it
    found. Raises ``AgentToolLoopFallback`` only when no tool call ever happened
    (even after the nudge) or the endpoint body is unusable, so the caller can
    run the a83 one-shot path.
    """
    connection = dict(connection)
    budget = _loop_budget(capable, max_iterations)
    specs = build_tool_specifications(tools.catalog(), granted_scopes, web_access)
    if not specs:
        raise AgentToolLoopFallback("the agent has no callable read-only tools")
    allowed = {spec["function"]["name"] for spec in specs}

    headers = {"Content-Type": "application/json", "Accept": "application/json"}
    if connection.get("api_key"):
        headers["Authorization"] = "Bearer {}".format(connection["api_key"])
    timeout = inference_timeout(connection, timeout_seconds)
    model = _resolve_model(connection, headers, timeout)
    generation_tokens = reasoning_headroom_tokens(connection, budget.visible_tokens)

    system = build_agent_system_prompt(connection, definition, context)
    system = system + "\n" + TOOL_LOOP_INSTRUCTIONS
    messages: List[Dict[str, Any]] = [
        {"role": "system", "content": provider_system_prompt(system, connection)},
        {"role": "user", "content": provider_user_content({
            "task": str(task)[:4000],
            "granted_context": _loop_context(connection, context, generation_tokens),
        }, connection)},
    ]

    invocations: List[Dict[str, Any]] = []
    sources: List[Dict[str, Any]] = []
    vetted_urls: set = set()
    tool_calls_seen = 0
    nudged = False
    for _ in range(budget.iterations):
        body = chat_completion(connection, {
            "model": model,
            "messages": messages,
            "tools": specs,
            "tool_choice": "auto",
            **generation_parameters(connection, max_tokens=generation_tokens, temperature=0.1),
        }, headers, timeout)
        # A non-object body (a JSON array or scalar from a broken endpoint) would
        # make `.get` raise; treat any unusable body as a fall-back signal.
        if not isinstance(body, Mapping):
            raise AgentToolLoopFallback("the endpoint returned a non-object body")
        choices = body.get("choices")
        first = choices[0] if isinstance(choices, list) and choices else {}
        message = (first.get("message") if isinstance(first, Mapping) else {}) or {}
        calls = parse_tool_calls(message)
        if not calls:
            if tool_calls_seen == 0 and not nudged:
                # The model answered from memory without checking. Steer it ONCE
                # to gather evidence with a tool, then give it one more turn. The
                # persona prompt's "return JSON" pull is strong on the 4B, so a
                # single firm nudge is what turns tool-use from optional to the
                # default for a research agent - bounded to exactly one retry.
                nudged = True
                messages.append({"role": "assistant", "content": message.get("content") or ""})
                messages.append({"role": "user", "content": TOOL_LOOP_NUDGE})
                continue
            if tool_calls_seen == 0:
                # Even after the nudge the model would not call a tool - the
                # pre-fetch path (which gathers evidence for it) is strictly
                # better, so fall back rather than ship a memory-only answer.
                raise AgentToolLoopFallback(
                    "the model emitted no tool call even after a nudge", nudged=nudged)
            # A final answer, and the loop HAS done tool work (tool_calls_seen>0).
            # Parse it robustly and, on a JSON slip, retry once - never discard the
            # search/fetch the model already did over a final-parse hiccup.
            answer = _finalize_loop_answer(
                message, task, definition, context, sources=sources,
                connection=connection, model=model, headers=headers, timeout=timeout,
                generation_tokens=generation_tokens, messages=messages,
            )
            return answer, invocations, sources, nudged
        # Execute at most the per-turn cap, and echo an assistant message whose
        # tool_calls are EXACTLY those executed, re-serialised with the SAME ids
        # used on the tool responses. Echoing the full raw tool_calls while
        # answering only a subset makes a strict endpoint reject the next request
        # ("missing tool response for tool_call_id ..."); rebuilding it here keeps
        # the assistant calls and the tool responses in 1:1 correspondence.
        executed = calls[:budget.calls_per_turn]
        messages.append({
            "role": "assistant",
            "content": message.get("content") or "",
            "tool_calls": [
                {"id": call["id"], "type": "function",
                 "function": {"name": call["name"],
                              "arguments": json.dumps(call["arguments"])}}
                for call in executed
            ],
        })
        for call in executed:
            tool_calls_seen += 1
            content = _execute_tool_call(
                call, allowed=allowed, tools=tools, actor=actor,
                administrator=administrator, granted_scopes=granted_scopes,
                web_access=web_access, vetted_urls=vetted_urls,
                invocations=invocations, sources=sources,
                response_limit=budget.tool_response_chars,
            )
            messages.append({
                "role": "tool", "tool_call_id": call["id"],
                "name": call["name"], "content": content,
            })
        if len(messages) > budget.max_messages:
            break  # budget spent mid-gather -> force one synthesis turn below
    # The loop spent its whole turn/message budget while still gathering and never
    # wrote a final answer. If it never called a tool at all, fall back; otherwise
    # FORCE one tools-free synthesis turn so a thorough model that fetched
    # everything still gets to write up what it found (the key a89 fix).
    if tool_calls_seen == 0:
        raise AgentToolLoopFallback(
            "the tool loop spent its budget with no tool call", nudged=nudged)
    answer = _forced_synthesis(
        task, definition, context, sources=sources, connection=connection,
        model=model, headers=headers, timeout=timeout, specs=specs,
        generation_tokens=generation_tokens, messages=messages,
    )
    return answer, invocations, sources, nudged


def _forced_synthesis(
    task: str, definition: Mapping[str, Any], context: Mapping[str, Any], *,
    sources: List[Dict[str, Any]], connection: Dict[str, Any], model: str,
    headers: Dict[str, str], timeout: int, specs: List[Dict[str, Any]],
    generation_tokens: int, messages: List[Dict[str, Any]],
) -> Dict[str, Any]:
    """One tools-free turn to synthesise the final answer from gathered evidence.

    ``tool_choice="none"`` forbids further tool calls (the schema stays for
    context) so the model MUST write the JSON answer from what it already
    fetched. The result runs through the same robust parse/retry/honest-result
    path, so a slip here still never discards the tool work.
    """
    messages = list(messages) + [{"role": "user", "content": FORCED_SYNTHESIS}]
    body = chat_completion(connection, {
        "model": model, "messages": messages, "tools": specs, "tool_choice": "none",
        **generation_parameters(connection, max_tokens=generation_tokens, temperature=0.0),
    }, headers, timeout)
    return _finalize_loop_answer(
        _message_from_body(body), task, definition, context, sources=sources,
        connection=connection, model=model, headers=headers, timeout=timeout,
        generation_tokens=generation_tokens, messages=messages,
    )


def _finalize_loop_answer(
    message: Mapping[str, Any], task: str, definition: Mapping[str, Any],
    context: Mapping[str, Any], *, sources: List[Dict[str, Any]],
    connection: Dict[str, Any], model: str, headers: Dict[str, str], timeout: int,
    generation_tokens: int, messages: List[Dict[str, Any]],
) -> Dict[str, Any]:
    """Parse the model's final answer robustly; retry once on a JSON slip.

    The loop has already searched/fetched by the time this runs, so a fragile-4B
    JSON slip must NOT discard that work. A tolerant extraction recovers most
    slips; if it still will not parse, the model is re-asked ONCE for JSON only
    (no tools, so it must answer); if THAT also fails, an honest result is built
    from the evidence the loop gathered - never a fabrication and never a silent
    discard. The #247w grounding policy then runs on whatever parsed, with the
    loop's own fetched sources counted as citable (they are real retrieved pages).
    """
    content = message.get("content") if isinstance(message, Mapping) else None
    parsed = _parse_or_none(content)
    if parsed is None:
        LOGGER.warning(
            "tool loop final answer did not parse (%d chars); re-asking once for "
            "JSON. tail=%r", len(_content_text(content)), _content_text(content)[-80:],
        )
        retry_messages = list(messages) + [
            {"role": "assistant", "content": _content_text(content)},
            {"role": "user", "content": FINAL_JSON_RETRY},
        ]
        body = chat_completion(connection, {
            "model": model, "messages": retry_messages,
            **generation_parameters(connection, max_tokens=generation_tokens, temperature=0.0),
        }, headers, timeout)
        parsed = _parse_or_none(_message_from_body(body).get("content"))
    if parsed is None:
        return _honest_result_from_tool_work(sources)
    if sources and not parsed.get("sources"):
        parsed["sources"] = list(sources)
    return apply_grounding_policy(parsed, task, definition, context)


def _collect_fetched_source(outcome: Any, sources: List[Dict[str, Any]]) -> None:
    """Record a page the model actually fetched, as a citable source.

    This is what the runner surfaces as the run's "sources it used" on the loop
    path: the real page the model read, with a redirect recorded on it so a
    citation never hides that the evidence came from a different URL than the
    link. It is the loop's own evidence, not a pre-fetch's.
    """
    result = outcome.get("result") if isinstance(outcome, Mapping) else {}
    if not isinstance(result, Mapping):
        return
    url = str(result.get("url") or "").strip()
    if not url:
        return
    evidence = result.get("evidence") if isinstance(result.get("evidence"), Mapping) else {}
    metadata = evidence.get("metadata") if isinstance(evidence.get("metadata"), Mapping) else {}
    title = str(metadata.get("title") or PUBLIC_RESEARCH_SOURCE_TITLE)[:400]
    source = {"source": url[:2000], "summary": title}
    final = str(result.get("final_url") or "").strip()
    if result.get("redirected") and final and final != url:
        source["retrieved_from"] = final[:2000]
    sources.append(source)


def _execute_tool_call(
    call: Dict[str, Any], *, allowed: set, tools: Any, actor: Optional[str],
    administrator: bool, granted_scopes: Any, web_access: Any, vetted_urls: set,
    invocations: List[Dict[str, Any]], sources: List[Dict[str, Any]],
    response_limit: int,
) -> str:
    """Re-validate one model tool call, run it if granted, and audit it.

    Security-critical: a call to a tool outside the granted read-only set is
    REFUSED here - never handed to ``tools.run`` - and returned to the model as a
    tool error, so the model cannot reach a tool it was not granted or one that
    mutates. ``tools.run`` re-checks the scope a second time as defence in depth.
    """
    name = call["name"]
    arguments = call["arguments"]
    record: Dict[str, Any] = {"name": name, "arguments": _safe_value(arguments)}
    if name not in allowed:
        record["status"] = "refused"
        invocations.append(record)
        return json.dumps({
            "error": "tool_not_granted: {} is not in this agent's read-only tools.".format(name)
        })
    call_web_access = web_access
    if name == "research.fetch":
        # Let the model open only what its own guarded search returned.
        call_web_access = {**(web_access or {}), "search_result_urls": sorted(vetted_urls)}
    try:
        outcome = tools.run(
            name, arguments, actor=actor, administrator=administrator,
            granted_scopes=granted_scopes, web_access=call_web_access,
        )
    except AssistantToolError as error:
        record["status"] = "error"
        invocations.append(record)
        return json.dumps({"error": str(error)[:300]})
    if name == "research.search":
        _collect_vetted(outcome, vetted_urls)
    elif name == "research.fetch":
        _collect_fetched_source(outcome, sources)
    record["status"] = "executed"
    invocations.append(record)
    return _bounded_content(outcome, response_limit)


def _tool_loop_eligible(granted_scopes: Any, web_access: Any) -> bool:
    """Whether this run should attempt the native loop.

    Gated to web research agents deliberately. That is the case the pre-fetch
    path serves least well (the model, not the runner, should decide what to
    search and fetch) and the one the live probe validated. A purely appliance
    agent keeps the pre-fetch path unchanged, which bounds the extra
    single-slot-NPU calls to the runs that actually benefit. Widening this to any
    tool-bearing scope is a one-line change when the NPU can afford it.
    """
    return bool((web_access or {}).get("enabled")) and "research:read" in {
        str(scope) for scope in (granted_scopes or ())
    }


def loop_eligible(definition: Mapping[str, Any], agent: Any, context_policy: str,
                  granted_scopes: Any, web_access: Any) -> bool:
    """Whether the runner should drive this run through the native loop first.

    A custom agent, under the ``granted`` context policy, with web research and a
    runtime that implements the loop. Anything else keeps the a83 pre-fetch path
    unchanged (specialist runs, narrowed-context runs, appliance-only agents), so
    the loop only ever changes the runs it was built for.
    """
    return bool(
        definition.get("custom")
        and context_policy == "granted"
        and getattr(agent, "supports_tool_loop", False)
        and _tool_loop_eligible(granted_scopes, web_access)
    )


def drive_custom_agent_loop(
    agent: Any, tools: Any, task: str, definition: Mapping[str, Any],
    context: Mapping[str, Any], mode: str, *, actor: Optional[str],
    administrator: bool, granted_scopes: Any, web_access: Any, capable: bool = False,
) -> tuple[Dict[str, Any], List[Dict[str, Any]], List[Dict[str, Any]], bool]:
    """Resolve the connection, run the loop, validate, and thread its evidence.

    Returns ``(validated_result, invocations, sources, nudged)``. The same envelope
    ``custom_agent`` enforces - a safe, allowed endpoint and a bounded task - but
    the MODEL chooses the granted read-only tools. ``capable`` (the GPU 27B tier)
    selects the larger loop budget. A missing or disallowed endpoint raises
    ``AgentToolLoopFallback`` so the runner runs the a83 one-shot path; any
    transport or shape error propagates for the runner to catch and fall back on
    too. ``validate_specialist`` is the agent's own result gate; the model's own
    clarifications survive it so #247w honest degradation is kept.
    """
    connection = agent._connection(mode) if hasattr(agent, "_connection") else None
    if not connection or not allowed_inference_endpoint(connection):
        raise AgentToolLoopFallback("no safe selected endpoint for the tool loop")
    clean_task = str(task).strip()
    if not clean_task or len(clean_task) > 4000:
        raise AgentToolLoopFallback("task outside the 1-4,000 character bound")
    result, invocations, sources, nudged = run_agent_tool_loop(
        connection, clean_task, definition, context or {},
        int(getattr(agent, "timeout_seconds", 20)),
        tools=tools, actor=actor, administrator=administrator,
        granted_scopes=granted_scopes, web_access=web_access, capable=capable,
    )
    validated = validate_specialist(
        result, str(definition.get("name", "Custom agent")), "custom-agent-model",
    )
    clarifications = result.get("clarifications") if isinstance(result, Mapping) else None
    if clarifications:
        validated["clarifications"] = [
            str(item)[:400] for item in list(clarifications)[:6] if str(item).strip()
        ]
    return validated, invocations, sources, nudged
