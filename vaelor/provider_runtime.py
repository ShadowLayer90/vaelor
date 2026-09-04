"""Shared response-budget policy for connected AI providers."""

from __future__ import annotations

import json
import re
from typing import Any, Dict, NamedTuple, Tuple

from .assistant_vocabulary import BASE_TOOLS, literal_phrases
from .phrase_match import mentions
from .telemetry_trend import SUBJECT_NOT_RETAINED
from .token_estimate import context_chars as _context_chars


def managed_local_connection(connection: Dict[str, str]) -> bool:
    return connection.get("base_url", "").startswith(
        ("http://127.0.0.1:", "http://[::1]:")
    )


def generation_parameters(
    connection: Dict[str, str], *, max_tokens: int, temperature: float
) -> Dict[str, Any]:
    """Shape one generation request, holding managed-local work to the Pi.

    The managed-local ceiling used to be an unexplained 768 applied silently, so
    a caller that asked for more got less without being told and without anyone
    recording why 768. It is now one documented number derived from what the Pi
    can actually generate inside an inference budget, and it lives with the rest
    of the measurement policy.

    The import is deferred because `model_calibration` reads
    `managed_local_connection` from this module; taking it at module scope would
    close the loop.
    """
    if connection.get("provider") == "openai":
        return {
            "max_completion_tokens": max_tokens,
            "reasoning_effort": "low",
        }
    if managed_local_connection(connection):
        from .model_profiles import managed_local_token_ceiling

        max_tokens = min(max_tokens, managed_local_token_ceiling())
    return {"max_tokens": max_tokens, "temperature": temperature}


def _model_billions(connection: Dict[str, str]) -> float:
    matches = re.findall(r"(?<!\d)(\d+(?:\.\d+)?)\s*[bB](?![A-Za-z])", connection.get("model", ""))
    return float(matches[-1]) if matches else 0.0


def model_capability(connection: Dict[str, str] | None) -> Dict[str, Any]:
    """Describe the selected engine honestly for the UI and support tooling."""
    if not connection:
        return {
            "tier": "rules",
            "label": "Built-in appliance intelligence",
            "description": "Reliable for known Vaelor node facts and guarded actions; not a general-purpose model.",
            "limitations": ["Broader questions require a connected model."],
        }
    if connection.get("provider") == "openai":
        return {
            "tier": "frontier",
            "label": "Frontier intelligence",
            "description": "Best suited to nuanced questions and multi-step reasoning.",
            "limitations": ["Answers still pass Vaelor's approval and deployment policy."],
        }
    if not managed_local_connection(connection):
        return {
            "tier": "connected",
            "label": "Connected model",
            "description": "Capability depends on the model exposed by this endpoint.",
            "limitations": ["Vaelor cannot infer quality from an endpoint name alone."],
        }
    size = _model_billions(connection)
    if not size or size <= 2:
        return {
            "tier": "basic-local",
            "label": "Basic local intelligence",
            "description": "Private and lightweight, optimized for short appliance questions.",
            "limitations": [
                "Limited multi-step reasoning and follow-up interpretation.",
                "May miss nuance; deployment claims are verified by built-in policy.",
            ],
        }
    if size <= 8:
        return {
            "tier": "capable-local",
            "label": "Capable local intelligence",
            "description": "Better context handling and reasoning, with higher memory use.",
            "limitations": ["Complex plans still require Vaelor's approval checks."],
        }
    return {
        "tier": "advanced-local",
        "label": "Advanced local intelligence",
        "description": "Higher-capacity local reasoning with substantial hardware requirements.",
        "limitations": ["Performance depends on available RAM, CPU, and runtime settings."],
    }


def request_complexity(request: str) -> str:
    """Classify prompt budget needs without delegating safety policy to a model."""
    lower = str(request).lower()
    complex_terms = (
        "deploy", "install", "configure", "troubleshoot", "diagnose", "research",
        "compare", "migrate", "restore", "cluster", "compose", "dependency",
        "why does", "not working", "keeps failing", "step by step",
    )
    return "complex" if len(request) > 500 or any(term in lower for term in complex_terms) else "simple"


def _declared_context_tokens(connection: Dict[str, str]) -> int:
    for key in ("context_window", "context_length", "max_context_tokens"):
        try:
            value = int(connection.get(key, "") or 0)
        except (TypeError, ValueError):
            continue
        if 512 <= value <= 2_000_000:
            return value
    return 0


# What a connected endpoint is assumed to hold when it does not declare a
# window. Measured on this appliance: LM Studio serves `qwen/qwen3.6-27b` with
# `n_ctx = 8192` even though the model itself supports 262,144, and a prompt one
# token over that window is refused with HTTP 400 before any generation starts.
# Assuming the model's advertised maximum would therefore turn a working agent
# into an instant rejection, so the assumption is the smaller, safer number and
# an operator who knows better raises it with `context_window`.
DEFAULT_CONTEXT_TOKENS = 8192

# Both context budgets below turn a window in tokens into a budget in
# characters. That conversion used to carry its own local constant, 3,
# "deliberately below the usual four so the estimate reserves more room than
# the tokenizer needs" - true when counting tokens, and exactly backwards in
# this direction. Turning a window into characters, three does not reserve
# room, it discards it: every request was sized to about 72% of a window the
# appliance had already paid for. The factor and its measurement now live in
# `token_estimate`, which is the only place that number appears.

# The prompt is more than the granted context: a system prompt with the agent's
# own instructions, the task text, and the JSON scaffolding around both.
PROMPT_OVERHEAD_TOKENS = 1200

# Never hand an agent less than this, even on a tiny declared window: below it
# there is no room for evidence at all and the run can only answer from the
# task text.
MIN_AGENT_CONTEXT_CHARS = 2000

# Managed-local is the appliance's own model on the appliance's own Pi, and the
# limit there is throughput, not the window: the 1.7B measured on this hardware
# generates at roughly 2.4 tokens a second, so every extra thousand characters
# of prompt is paid for twice - once evaluating it, once reasoning about it.
# That model keeps the budget it was tuned for; the room found below is for
# connected endpoints, which is where agent research actually runs.
MANAGED_LOCAL_CONTEXT_CHARS = 700


def declared_context_tokens(connection: Dict[str, str]) -> int:
    """The context window this connection states, or 0 when it states none."""
    return _declared_context_tokens(connection or {})


def agent_context_chars(
    connection: Dict[str, str], *, generation_tokens: int
) -> int:
    """How much granted context one agent request may carry.

    Sized against the window rather than against a constant. The constant was
    700 characters - chosen for a 1.7B appliance model and then applied to every
    endpoint, including a 27B on a LAN - and it silently discarded the retrieved
    pages an agent had just been at pains to fetch.

    The generation has to fit beside the prompt, but a model that reasons
    verbosely must not be allowed to squeeze the evidence out of its own
    context: the reserve honours what the request asks for, capped at a third of
    the window.
    """
    if managed_local_connection(connection or {}):
        return MANAGED_LOCAL_CONTEXT_CHARS
    window = declared_context_tokens(connection) or DEFAULT_CONTEXT_TOKENS
    reserve = min(max(0, int(generation_tokens)), window // 3)
    room = window - reserve - PROMPT_OVERHEAD_TOKENS
    return max(MIN_AGENT_CONTEXT_CHARS, _context_chars(room))


def assistant_budget(connection: Dict[str, str], request: str = "") -> Dict[str, int]:
    """Scale relevant context and output to model capability and task complexity."""
    complex_request = request_complexity(request) == "complex"
    if connection.get("provider") == "openai":
        budget = {"context_chars": 64000 if complex_request else 32000, "max_tokens": 1600 if complex_request else 900}
    elif not managed_local_connection(connection):
        budget = {"context_chars": 32000 if complex_request else 16000, "max_tokens": 1024 if complex_request else 512}
    else:
        size = _model_billions(connection)
        if not size or size <= 2:
            budget = {"context_chars": 12000 if complex_request else 6000, "max_tokens": 320 if complex_request else 192}
        elif size <= 4:
            budget = {"context_chars": 24000 if complex_request else 12000, "max_tokens": 512 if complex_request else 320}
        elif size <= 8:
            budget = {"context_chars": 48000 if complex_request else 24000, "max_tokens": 768 if complex_request else 512}
        else:
            budget = {"context_chars": 64000 if complex_request else 32000, "max_tokens": 1024 if complex_request else 640}
    declared = _declared_context_tokens(connection)
    if declared:
        # Three things share a declared window: this request's generation, the
        # standing prompt (the system prompt and the machine brief, capped at
        # 1000 tokens on its own), and only what is left over is evidence.
        # `declared * 3` budgeted the *whole* window as evidence and relied on
        # a wrong characters-per-token factor to leave accidental room for the
        # other two. Correcting the factor without naming the reserve would
        # have turned a 28% under-fill into an overflow, so the reserve is
        # subtracted where it can be read.
        room = declared - budget["max_tokens"] - PROMPT_OVERHEAD_TOKENS
        budget["context_chars"] = min(
            budget["context_chars"],
            max(MIN_AGENT_CONTEXT_CHARS, _context_chars(room)),
        )
    return budget


def structured_task_budget(connection: Dict[str, str], task: str) -> Dict[str, int]:
    """Bound short JSON tasks separately from conversational answers.

    Classification and routing should not consume a chat-sized generation on a
    small appliance model.  The limits still scale upward for models that can
    reliably produce richer structured output.
    """
    tier = model_capability(connection)["tier"]
    if task == "application-intent":
        policies = {
            "basic-local": {"max_tokens": 72, "timeout_seconds": 15},
            "capable-local": {"max_tokens": 144, "timeout_seconds": 24},
            "advanced-local": {"max_tokens": 192, "timeout_seconds": 30},
            "connected": {"max_tokens": 192, "timeout_seconds": 25},
            "frontier": {"max_tokens": 256, "timeout_seconds": 30},
        }
        return policies.get(tier, {"max_tokens": 96, "timeout_seconds": 18})
    return {"max_tokens": 192, "timeout_seconds": 25}


#: How long a managed-local Assistant answer may take before Vaelor gives up.
#:
#: **Measured on the appliance 2026-08-11, and the old ceiling was the defect.**
#: This was `min(max(configured, 15), 20)` - 15 to 20 seconds - on hardware
#: where the Assistant's turns take 45 to 84 seconds. llama-server's log shows
#: Vaelor hanging up at 10.0 s, 11.0 s and 15.1 s (`srv stop: cancel task`)
#: on generations that then completed at ~55 s, ~84 s and ~17 s. Every single
#: model answer was cancelled, so the Assistant fell back to built-in readings
#: and answered a question the owner had not asked.
#:
#: **The control experiment was already in the product.** AI Chat, on the same
#: machine and the same model minutes later, allows
#: `MANAGED_LOCAL_CHAT_SECONDS = 120`, waited 25.25 s, and returned a correct
#: answer. One surface waited and worked; the other cancelled and misled. The
#: hardware was never the problem.
#:
#: 120 matches that working reference rather than inventing a third number.
#: The Assistant's prompt is larger than a plain chat turn - standing brief,
#: tool map, facts, conversation - so it needs at least the room AI Chat has,
#: not less. A wake costs ~11 s of model reload before a token exists, which a
#: 15 s ceiling could never survive.
#:
#: **Bounded is still the intent**, and it is still bounded: a genuinely dead
#: endpoint fails at 120 s rather than hanging, and slower work still belongs
#: in specialist tasks. What changed is that the bound no longer sits below
#: what the machine can do.
#:
#: VD-086 records why this number is never taken from the remote budget.
MANAGED_LOCAL_ANSWER_SECONDS = 120


def answer_timeout(connection: Dict[str, str], configured: int) -> int:
    """How long to wait for one interactive answer, bounded but not below the
    hardware's measured behaviour. See :data:`MANAGED_LOCAL_ANSWER_SECONDS`."""
    if not managed_local_connection(connection):
        return configured
    # **The local budget is not the remote budget.** `configured` is
    # `DeploymentAgent.timeout_seconds`, which the control plane builds from
    # `remote_inference_budget()` - 240 by default and up to 900 by
    # environment. Honouring it upward handed a wedged local llama-server a
    # fifteen-minute hold on a Flask worker, which is the exact outcome
    # `chat_inference.MANAGED_LOCAL_CHAT_SECONDS` exists to prevent and says
    # so in its own comment. A number meant for a cloud endpoint must not
    # reach the model running on this box.
    #
    # **But a flat constant was the wrong correction, and review caught it
    # inverted.** AI Chat reaches the same 120 through `inference_timeout`,
    # which ends `min(max(budget, profile_timeout_floor(connection)), 900)` -
    # so AI Chat's 120 is a *floor that measurement raises* and this was a
    # *ceiling that discards it*. Fed the latencies this file's own docstring
    # quotes, the two surfaces disagreed on the identical model: measured at
    # 84 s the profile floor is 269 s, AI Chat waited 269 s and the Assistant
    # would have cut off at 120 s. Claiming parity while being the stricter
    # surface is worse than the 15 s bug, because the number reads correct.
    #
    # `profile_timeout_floor` is documented "only ever used to *raise* a
    # timeout": it is this machine's own measurement of this model, not a
    # setting someone typed, and the whole reason VD-075 and #117 exist is
    # that measurements get taken here and then not read. Honour it, and keep
    # the same 900 s ceiling AI Chat has so a runaway profile cannot undo the
    # bound this function exists for.
    #
    # Deferred for the same reason `generation_parameters` defers its import:
    # `model_profiles` reaches `model_calibration`, which reads
    # `managed_local_connection` from this module, so a module-scope import
    # closes the loop.
    from .inference_client import MAX_INFERENCE_SECONDS
    from .model_profiles import profile_timeout_floor

    return min(
        max(MANAGED_LOCAL_ANSWER_SECONDS, profile_timeout_floor(connection)),
        MAX_INFERENCE_SECONDS,
    )


def _compact_context(value: Any, item_limit: int, string_limit: int) -> Any:
    """One pass of `provider_context` at a fixed pair of limits."""

    def compact(item: Any, depth: int = 0) -> Any:
        if item is None or isinstance(item, (bool, int, float)):
            return item
        if isinstance(item, str):
            return item[:string_limit]
        if depth >= 7:
            return str(item)[:string_limit]
        if isinstance(item, list):
            return [compact(entry, depth + 1) for entry in item[:item_limit]]
        if isinstance(item, dict):
            return {
                str(key)[:100]: compact(entry, depth + 1)
                for key, entry in list(item.items())[:item_limit]
            }
        return str(item)[:string_limit]

    return compact(value)


# The per-string ceiling the search below works within. 8,000 characters is what
# `assistant_tools` already lets one tool result carry, so a retrieved page can
# arrive whole when the budget has room for it. The old fixed ceiling of 2,000
# was below the amount of navigation chrome a modern sports page carries before
# its first score, which meant the model received the site's menu and none of
# its data.
MAX_STRING_LIMIT = 8000
MIN_STRING_LIMIT = 60


def provider_context(value: Any, *, max_chars: int = 700) -> Any:
    """Bound untrusted context so a model retains output space.

    Overflow loses *detail*, not *structure*. The old behaviour replaced the
    whole context with the first few hundred characters of its own JSON the
    moment it did not fit, and that is what an agent asked to research the web
    actually received: a run whose facts were 21,449 characters arrived as a
    520-character snapshot holding two search-result URLs and nothing else. The
    model then said, correctly, that it had been given links rather than pages.
    So the limits are searched for rather than fixed: the longest strings that
    still fit are used, and only when no string length fits are fewer items
    tried. That keeps the keys the caller assembled - `research.fetch` stays
    present - and spends whatever room is left on the evidence itself. The
    snapshot remains as the last resort for a context that cannot be made to
    fit at all.
    """
    items = max(3, min(24, max_chars // 800))
    while True:
        low, high = MIN_STRING_LIMIT, min(max_chars, MAX_STRING_LIMIT)
        best = None
        while low <= high:
            middle = (low + high) // 2
            bounded = _compact_context(value, items, middle)
            encoded = json.dumps(bounded, separators=(",", ":"), default=str)
            if len(encoded) <= max_chars:
                best = bounded
                low = middle + 1
            else:
                high = middle - 1
        if best is not None:
            return best
        if items <= 3:
            break
        items = max(3, items // 2)
    encoded = json.dumps(
        _compact_context(value, 3, MIN_STRING_LIMIT),
        separators=(",", ":"), default=str,
    )
    return _snapshot(encoded, max_chars)


def _snapshot(encoded: str, max_chars: int) -> Dict[str, Any]:
    """The last-resort form, trimmed until the wrapper itself fits the budget.

    Slicing to a fixed allowance overshot: escaping the quotes in a JSON
    fragment inflates it again once it is nested inside this object, so the cut
    is checked rather than assumed.
    """
    note = "Additional read-only context was omitted to fit this local model."
    cut = max(200, max_chars - 180)
    while True:
        result = {"truncated_snapshot": encoded[:cut], "note": note}
        if cut <= 200 or len(
            json.dumps(result, separators=(",", ":"))
        ) <= max_chars:
            return result
        cut = max(200, int(cut * 0.8))


class Topic(NamedTuple):
    """One question vocabulary and the readings that answer it.

    Named because the pair was a bare two-tuple and the table below is long
    enough that ``row[1]`` stops reading as "the tools this question needs".

    ``terms`` is no longer written by hand - see `_TOPIC_KEYS`. ``extra``
    keeps the row's own contribution separately rather than only folding it
    into ``terms``, because the two are different claims: ``terms`` is what
    this row matches, ``extra`` is what this row asserted the shared
    vocabulary does not cover. A guard that could only read ``terms`` could
    not tell a word that was inherited from a word that was restated, and
    `test_shared_assistant_vocabulary.py` proved exactly that about itself -
    the assertion passed against a row with ``fan`` written into it by hand.
    """

    terms: Tuple[str, ...]
    tools: Tuple[str, ...]
    extra: Tuple[str, ...] = ()


def _topic(extra: Tuple[str, ...], tools: Tuple[str, ...]) -> Topic:
    """One filter row, its vocabulary inherited from the gather table.

    The row admits the readings in ``tools`` whenever the question named any
    of them, and "named" is decided by
    `vaelor.assistant_vocabulary.TOOL_PHRASES` - the same table that decided
    to fetch them. ``extra`` is this filter's own, and every entry in it is
    there to select a **base** reading: `system.identity` and
    `system.telemetry` are gathered for every question, so no word selects
    them and no word for them exists in the shared table.
    """
    terms: list = []
    for tool in tools:
        if tool in BASE_TOOLS:
            continue
        for phrase in literal_phrases(tool):
            if phrase not in terms:
                terms.append(phrase)
    for phrase in extra:
        if phrase not in terms:
            terms.append(phrase)
    return Topic(tuple(terms), tools, tuple(extra))


#: Question vocabulary, mapped to the readings that answer it.
#:
#: **Matched on whole words, and it was substrings.** ``ip`` sits inside
#: "description", "equipment", "multiple" and "script"; ``app`` sits inside
#: "appliance", "apply" and "happens", so *every* question naming this
#: appliance selected the workload tools; ``ram`` sits inside "program" and
#: "diagram". Over-matching here is not a harmless extra fetch - only the
#: first four keys survive the cap below, so a wrong match **evicts a right
#: one**.
#:
#: **The boot row is why this table is worth reading twice.**
#: ``system.telemetry`` carries ``boot_time`` and is a base tool, so it is
#: gathered for every question ever asked. No term here mentioned booting, so
#: the filter discarded it before the model saw it, and the appliance replied
#: that it "holds no record of the last reboot time" - a claim about the
#: machine, when the truth was about this table (LESSONS pattern 11).
#:
#: Word boundaries are a property of :func:`vaelor.phrase_match.mentions`, the
#: builder every table in this product matches through, rather than of any
#: entry below. That is what makes this a fix for the *class*: a term added
#: here later cannot reintroduce the substring trap, because no term here ever
#: decides how it is matched.
#:
#: **The vocabulary below is no longer written here.** It was, and it was the
#: same vocabulary as `assistant_intents._INTENT_PHRASES` spelled a second
#: time - a gather stage and a filter stage in one pipeline, each with its own
#: hand-maintained word list. They drifted on every edit, and the drift is not
#: symmetric: a word the gatherer knows and this filter does not means the
#: reading is fetched off the hardware and **discarded before the model sees
#: it**. Measured on the commit that changed this: 8 of 21 ordinary questions
#: lost a reading that way. "What is my ping?" fetched `network.status` and
#: showed the model identity alone. That is `FOUNDATION_EVIDENCE` section 3
#: instance 12, and #188 (VD-091). The battery is
#: `tests/test_shared_assistant_vocabulary.ORDINARY_QUESTIONS` and re-derives
#: the figure; it was quoted in five places and lived in none until
#: 2026-08-13.
#:
#: Each row now inherits the shared vocabulary of the fact tools it admits, so
#: this filter cannot be narrower than the gather that feeds it. What stays
#: local to this table is the part that is genuinely this table's own:
#:
#: * **the order**, which is a priority - only the first four readings present
#:   in `facts` survive the cap in `assistant_context`, so a wrong match here
#:   still evicts a right one;
#: * **the supporting readings** a row admits alongside its own, which are
#:   gathered for their own reasons and are pulled in because the question
#:   needs them read together;
#: * **`extra`**, and every entry in it selects a *base* reading.
#:   `system.identity` and `system.telemetry` are gathered for every question
#:   ever asked, so nothing selects them and nothing for them can live in a
#:   table about what to fetch. `test_shared_assistant_vocabulary.py` fails if
#:   an `extra` entry is one the shared table already knows, which is the
#:   shape of somebody re-adding a fact-tool word here by hand.
#:
#: **The boot row is why this table is worth reading twice.**
#: ``system.telemetry`` carries ``boot_time`` and is a base tool, so it is
#: gathered for every question ever asked. No term here mentioned booting, so
#: the filter discarded it before the model saw it, and the appliance replied
#: that it "holds no record of the last reboot time" - a claim about the
#: machine, when the truth was about this table (LESSONS pattern 11). It is
#: the clearest case of an `extra`: nothing gathers telemetry, so no shared
#: word could ever have carried these.
_TOPIC_KEYS: Tuple[Topic, ...] = tuple(
    _topic(extra, tools) for extra, tools in (
        # ``cpu``/``cpus`` are here for the telemetry, not the fan: a bare
        # "what is my cpu doing" gathers no cooling reading, so admitting
        # `cooling.status` for them costs nothing and admitting
        # `system.telemetry` is the whole point.
        (("cpu", "cpus"), ("cooling.status", "system.telemetry")),
        (("ram", "memory", "swap"), ("system.telemetry", "system.identity")),
        ((), ("storage.status",)),
        # ``ip address`` and ``hostname`` are the two things anyone asks the
        # network for first, and neither was here: an owner asking for the
        # address of their own appliance was answered from identity alone.
        ((), ("network.status", "system.identity")),
        ((), ("workloads.capabilities", "workloads.inventory")),
        ((), ("workloads.inventory", "system.identity", "system.telemetry")),
        ((), ("updates.status", "jobs.recent")),
        ((), ("display.status",)),
        ((), ("lighting.status",)),
        # ``status`` is this filter's own and deliberately not shared: it is
        # the widest word an owner uses, it selects only the telemetry here,
        # and putting it in the gather table would make "what is the status of
        # the Voyager probe" a question about this appliance (#190).
        (("status",),
         ("services.status", "health.status", "system.telemetry")),
        # The fault verdict every other surface renders. `select_tools`
        # gathers `health.status` for these words and this filter had no term
        # for any of them, so the one reading that answers "is anything
        # wrong" was fetched and then thrown away before the model saw it.
        ((), ("health.status", "services.status", "system.telemetry")),
        # How long this machine has been up - see the note above.
        (("boot", "boots", "booted", "boot time", "reboot", "reboots",
          "rebooted", "restart", "restarts", "restarted", "uptime",
          "up since", "been up", "been running", "powered on", "power on",
          "how long"),
         ("system.telemetry", "system.identity", "services.status")),
        # Retained telemetry samples, which `past_time_answer` reads and
        # nothing was gathering. Without this row a trend question reached the
        # model with a single instant and no way to tell a spike from a load.
        ((), ("metrics.history", "system.telemetry")),
        # "Slow" on an accelerated appliance is almost never a CPU question.
        # Selecting only CPU telemetry for it is how an answer described a
        # processor at 4% while the accelerator was saturated.
        ((), ("gpu.status", "system.telemetry", "services.status")),
        ((), ("npu.status", "gpu.status")),
        ((), ("inference.status", "workloads.inventory")),
        ((), ("jobs.recent",)),
        ((), ("recovery.checkpoints", "workloads.inventory")),
    )
)


def assistant_context(question: str, value: Any, connection: Dict[str, str]) -> Any:
    """Select question-relevant facts before applying the small-model budget."""
    if not isinstance(value, dict):
        return provider_context(value)
    appliance = value.get("appliance", {})
    facts = appliance.get("facts", {}) if isinstance(appliance, dict) else {}
    lower = str(question).lower()
    selected_keys = []
    for topic in _TOPIC_KEYS:
        if mentions(lower, topic.terms):
            selected_keys.extend(topic.tools)
    selected_keys.extend(("system.identity",))
    selected_facts = {}
    for key in selected_keys:
        if key in facts and key not in selected_facts:
            selected_facts[key] = facts[key]
        if len(selected_facts) >= 4:
            break
    compact_value = {
        # The standing brief is deliberately *not* here. It used to lead this
        # object, safe from the topic filter but paying for the privilege: as a
        # JSON string value every newline in it was escaped and the whole thing
        # quoted, which measured 785 prompt tokens for a 717-token brief. FLM
        # caches no prefix, so that 68-token envelope was re-prefilled on every
        # request forever.
        #
        # It now rides in the system message (``brief_system_prompt``), where
        # it costs its own length and nothing more, and where the topic filter
        # cannot reach it at all - a stronger guarantee than the one this
        # comment used to claim.
        "facts": selected_facts,
        "conversation": value.get("conversation", [])[-4:],
        "memories": value.get("memories", [])[:2],
    }
    return provider_context(
        compact_value, max_chars=assistant_budget(connection, question)["context_chars"]
    )


#: The sentence the built-in path writes when nothing on this machine answered.
#:
#: **One marker, read by both predicates below.** They are siblings deciding
#: the same thing from opposite sides - "is this reply usable as it stands" -
#: and only one of them knew what a refusal looked like. So
#: `is_complete_builtin_answer` correctly declined to treat the refusal as an
#: answer while `is_grounded_live_answer`, five lines above it, classified the
#: identical text as a **reading-backed live answer** and returned it at
#: `deployment_agent.answer` without ever asking the model. Asked "What is my
#: IP address?", an appliance whose local model was connected and answering
#: replied *"Connect a local model"* in 0.0 s with `answered=True`, while
#: `network.status` held the address.
#:
#: Two lists that must agree are one list (LESSONS pattern 6), so this is the
#: list and :func:`declines_for_lack_of_a_reading` is the only reader of it.
#: The second marker is #203's, and it is a defect fix rather than a tidy-up.
#: `assistant_answer_scope.past_time_answer` declines a retrospective question
#: whose subject the retained samples do not cover - "has the fan speed been
#: rising" - and that reply carries `metrics.history` evidence, so
#: :func:`is_grounded_live_answer` classified a refusal as a reading and
#: returned it in place of asking the model. The model had `cooling.status` in
#: front of it and could have said what the fan was doing. A refusal that
#: outranks the model removes working capability, which is LESSONS pattern 4
#: and is the reason this list exists at all.
#:
#: Imported rather than spelled again, on this list's own rule.
BUILTIN_REFUSAL_MARKERS = (
    "enough built-in knowledge",
    SUBJECT_NOT_RETAINED,
)


def declines_for_lack_of_a_reading(answer: Dict[str, Any]) -> bool:
    """Whether this reply says it could not answer, rather than answering."""
    text = str((answer or {}).get("answer", "")).lower()
    return any(marker in text for marker in BUILTIN_REFUSAL_MARKERS)


#: Vocabulary that means the question asked for a reading this appliance takes.
#:
#: **Whole words, and these were substrings.** Every short entry here is a
#: fragment of ordinary English, and the failures are not hypothetical:
#: ``fan`` sits inside "Gra*fan*a", so *"How do I get to Grafana from my
#: laptop?"* was classified as a fan question; ``ip`` sits inside "mult*ip*le",
#: "descr*ip*tion", "equ*ip*ment" and "scr*ip*t"; ``ram`` sits inside
#: "prog*ram*", "diag*ram*" and "pa*ram*eter"; ``space`` sits inside
#: "name*space*"; ``cool`` sits inside "*cool*ant". Firing here returns the
#: built-in reply and never asks the model, so a false match is an answer the
#: owner does not get.
#:
#: Inflections are listed rather than derived, the same way
#: `assistant_intents._INTENT_PHRASES` lists them, and matching is
#: :func:`vaelor.phrase_match.mentions`' business rather than any entry's.
#: A ``\\w*`` suffix instead of an explicit inflection would have re-admitted
#: "fancy", "random" and "iPhone" through the same door.
_LIVE_TERMS = (
    "fan", "fans", "cool", "cooling", "temperature", "temperatures",
    "thermal", "thermals", "cpu", "cpus", "ram", "memory",
    "storage", "disk", "disks", "nvme", "microsd", "space",
    "network", "networks", "internet", "wifi", "wi-fi", "ethernet", "dns",
    # ``ip address`` was here and could never change the answer: `mentions`
    # matches whole words, ``ip`` is two entries left, and every string
    # holding "ip address" holds "ip". It was decoration, and decoration in a
    # word list is not free - it was one of the three modules that made
    # `'ip address'` a duplicated sentence (#188).
    "ip", "ips", "oled", "display", "displays",
    "screen", "screens", "rgb", "lighting",
    "update", "updates", "upgrade", "upgrades", "package", "packages",
    "service", "services", "health", "healthy", "status",
    "job", "jobs", "task", "tasks", "progress", "installed",
    "what is wrong", "anything wrong",
    "backup", "backups", "checkpoint", "checkpoints", "restore", "restored",
    "recovery",
)


def asks_for_a_live_reading(message: str) -> bool:
    """Whether the question names something this appliance measures."""
    return mentions(str(message or ""), _LIVE_TERMS)


def is_grounded_live_answer(message: str, answer: Dict[str, Any]) -> bool:
    """Whether this built-in reply is a reading, and so outranks the model."""
    if not answer.get("evidence"):
        return False
    # A refusal carrying `system.identity` as evidence is still a refusal.
    # Evidence means "something was read", never "the question was answered".
    if declines_for_lack_of_a_reading(answer):
        return False
    return asks_for_a_live_reading(message)


def is_complete_builtin_answer(answer: Dict[str, Any]) -> bool:
    text = str(answer.get("answer", ""))
    return (
        bool(text)
        and not answer.get("evidence")
        and not declines_for_lack_of_a_reading(answer)
    )


def provider_system_prompt(prompt: str, connection: Dict[str, str]) -> str:
    if managed_local_connection(connection):
        return "/no_think\n" + prompt
    return prompt


def provider_user_content(value: Any, connection: Dict[str, str]) -> str:
    content = json.dumps(value, separators=(",", ":"), default=str)
    if managed_local_connection(connection):
        return content + "\n/no_think"
    return content
