"""AI Chat's thinking control (VD-209): which models can think, and what each step sends.

The owner picks one of four steps - Off, Low, Medium, High - beside the model
picker. This module is the **one owner** (LESSONS 6) of three answers:

1. **Can this model think, and which steps does it take?** Decided from the
   provider's own data where it publishes any, never guessed from the name:

   - Anthropic's Models API carries ``capabilities.thinking.types`` and
     ``capabilities.effort`` per model (``GET /v1/models/{id}``);
   - OpenRouter's model list carries a ``reasoning`` object per model
     (``supported_efforts``, ``mandatory``).

   OpenAI's and Gemini's model lists carry no such field, so for them it is
   the explicit family tables below, each row from the provider's model page
   (scratchpad ``thinking/API_NOTES.md`` records which page said what). The
   owner's own servers (vLLM, llama.cpp, any OpenAI-compatible server) report
   nothing Vaelor can read, so the control is hidden for them (VD-209 item 3).

2. **What does each step send?** `plan` turns the chosen step into the request
   fields and the answer ceiling. A step the model does not take is never sent:
   a model that cannot turn thinking off is not offered Off (LESSONS 10 - the
   control never claims an effect that does not happen).

3. **What came back?** A thinking summary is model output: it is bounded here
   (`bounded_summary`), stored as plain text and rendered escaped.

A provider refusing the setting is not silent: `refusal_sentence` names the
step and the provider's reason, and the answer fails rather than falling back.
"""

from __future__ import annotations

import time
import urllib.parse
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, Mapping, Optional, Tuple

from .anthropic_messages import anthropic_headers
from .hosted_transport import UNNAMED_SERVICE, bearer_headers, clipped_reason

#: The four steps, in the order the control draws them. The word is the wire
#: value the console sends and stores; `STEP_LABELS` is what it shows.
STEP_OFF = "off"
STEP_LOW = "low"
STEP_MEDIUM = "medium"
STEP_HIGH = "high"
THINKING_STEPS: Tuple[str, ...] = (STEP_OFF, STEP_LOW, STEP_MEDIUM, STEP_HIGH)
#: The steps that ask for thinking. Each provider's own word for them is the
#: same word (Anthropic ``effort``, OpenAI/Gemini ``reasoning_effort``,
#: OpenRouter ``reasoning.effort``), which is why the step is the wire value.
REASONING_STEPS: Tuple[str, ...] = THINKING_STEPS[1:]
STEP_LABELS = {STEP_OFF: "Off", STEP_LOW: "Low", STEP_MEDIUM: "Medium", STEP_HIGH: "High"}

#: The step a connection starts on before the owner picks one (owner,
#: 2026-10-08): Off wherever the model can turn thinking off, and Low for a
#: model that cannot. A model with neither starts on the first step it takes.
DEFAULT_STEPS: Tuple[str, ...] = (STEP_OFF, STEP_LOW)

#: How each kind of model is driven.
ENGINE_ANTHROPIC_ADAPTIVE = "anthropic-adaptive"   # thinking.type adaptive + output_config.effort
ENGINE_ANTHROPIC_BUDGET = "anthropic-budget"       # thinking.type enabled + budget_tokens
ENGINE_REASONING_EFFORT = "reasoning-effort"       # top-level reasoning_effort (OpenAI, Gemini)
ENGINE_OPENROUTER = "openrouter-reasoning"         # reasoning: {effort}

#: Where a capability came from: the provider's listing, or the table here.
SOURCE_LISTING = "listing"
SOURCE_TABLE = "table"

#: Anthropic manual mode (4.5-era models): the thinking budget per step. The
#: API's minimum is 1,024 and the budget must stay below ``max_tokens``, since
#: thinking counts toward it; `plan` adds the answer's own room on top.
ANTHROPIC_MIN_BUDGET = 1024
ANTHROPIC_BUDGETS = {STEP_LOW: 2048, STEP_MEDIUM: 8192, STEP_HIGH: 16384}
#: The least room an answer keeps when a model's own ceiling forces the budget
#: down: below this a budget would starve the reply it exists to improve.
MIN_ANSWER_ROOM = 1024

#: The answer ceiling per step for an effort-driven model. Thinking and the
#: reply share ``max_tokens`` on every provider here, so a deeper step needs a
#: larger ceiling or the reply arrives empty with a length stop. It is a
#: ceiling, not a spend. Off and Low keep the ceiling AI Chat already gives.
STEP_ANSWER_TOKENS = {STEP_OFF: 0, STEP_LOW: 0, STEP_MEDIUM: 16384, STEP_HIGH: 32768}

#: A thinking summary is model output: bounded like the answer is (12,000
#: characters) so one turn cannot grow the store or the page without limit.
THINKING_SUMMARY_CHARS = 12000

#: How long a capability read is trusted. A provider changes what a model
#: accepts rarely; an hour keeps one extra request per connection and model
#: per hour, and a new key (a new credential version) starts fresh.
CAPABILITY_SECONDS = 3600
#: A failed read is retried sooner, so a brief outage does not hide the
#: control for an hour.
FAILED_READ_SECONDS = 60

_EVERY_STEP = THINKING_STEPS
_THINKING_ONLY = REASONING_STEPS

#: OpenAI: no capability field in ``GET /v1/models``, so each family is a row
#: here, from its model page (developers.openai.com/api/docs/models/<id>,
#: read 2026-10-08). A row matches the family name exactly or followed by
#: ``-`` (a snapshot or variant); anything else is not offered the control and
#: is sent no ``reasoning_effort`` - gpt-4.1, a "non-reasoning model", rejects it.
OPENAI_REASONING_FAMILIES: Tuple[Tuple[str, Tuple[str, ...]], ...] = (
    ("gpt-6.1-sol", _THINKING_ONLY),   # "none and minimal ... are not supported"
    ("gpt-6-astra", _THINKING_ONLY),   # "does not support none"
    ("gpt-6-luna", _EVERY_STEP),
    ("gpt-6-sol", _EVERY_STEP),
    ("gpt-5.6", _EVERY_STEP),          # none, low, medium (default), high, xhigh, max
    ("gpt-5.5", _EVERY_STEP),          # none, low, medium (default), high, xhigh
    ("gpt-5.4", _EVERY_STEP),          # none (default), low, medium, high, xhigh
    ("gpt-5.3", _THINKING_ONLY),       # UNCONFIRMED: no page read; the common subset
    ("gpt-5.2", _EVERY_STEP),          # none (default), low, medium, high, xhigh
    ("gpt-5.1", _EVERY_STEP),          # none (default), low, medium, high
    ("gpt-5", _THINKING_ONLY),         # minimal, low, medium, high - no none
    ("o4-mini", _THINKING_ONLY),       # UNCONFIRMED (o-series)
    ("o3", _THINKING_ONLY),            # UNCONFIRMED (o-series)
    ("o1", _THINKING_ONLY),            # UNCONFIRMED (o-series)
)

#: OpenAI models a family row would match but that take no reasoning setting
#: on Chat Completions (review R18, model pages read 2026-10-08): the
#: ``-chat-latest`` aliases, the ``-pro`` models (Chat Completions not
#: supported), and o1-mini / o1-preview. They get no control and no
#: ``reasoning_effort``. A model whose id carries one of these segments or
#: names is excluded before any row is read.
OPENAI_EXCLUDED_NAMES: Tuple[str, ...] = ("o1-mini", "o1-preview")
OPENAI_EXCLUDED_SUFFIX = "-chat-latest"
OPENAI_EXCLUDED_SEGMENT = "pro"


def _openai_excluded(model: str) -> bool:
    name = str(model or "").strip().lower()
    if any(name == item or name.startswith(item + "-") for item in OPENAI_EXCLUDED_NAMES):
        return True
    return name.endswith(OPENAI_EXCLUDED_SUFFIX) or OPENAI_EXCLUDED_SEGMENT in name.split("-")

#: Gemini through Google's OpenAI-compatible endpoint
#: (ai.google.dev/gemini-api/docs/openai, read 2026-10-08): ``reasoning_effort``
#: maps to ``thinking_level`` (3.x) or ``thinking_budget`` (2.5). "Set
#: reasoning_effort to none for 2.5 models" to turn it off; "reasoning cannot
#: be turned off for Gemini 2.5 Pro or 3 models". The compat model list
#: carries ids only. Matched on the id with any ``models/`` prefix removed.
GEMINI_REASONING_FAMILIES: Tuple[Tuple[str, Tuple[str, ...]], ...] = (
    ("gemini-2.5-pro", _THINKING_ONLY),
    ("gemini-2.5-flash", _EVERY_STEP),     # includes -flash-lite
    ("gemini-3", _THINKING_ONLY),          # 3 Pro, 3 Flash, ...
    ("gemini-3.", _THINKING_ONLY),         # 3.1, 3.8, ... Pro, Flash, Flash-Lite
)

#: The words a provider's own 400 reason uses when it is the thinking setting
#: it refused (each from a documented error text; see API_NOTES).
REFUSAL_WORDS: Tuple[str, ...] = ("thinking", "reasoning", "effort", "budget_tokens", "output_config")


class ThinkingSettingError(ValueError):
    """A step that cannot be sent to this model as asked, in the owner's words."""

    code = "chat_thinking_refused"


@dataclass(frozen=True)
class ThinkingCapability:
    """What one model takes: its steps, how they are sent, what comes back."""

    engine: str
    steps: Tuple[str, ...]
    decided_by: str
    #: Whether the provider returns readable thinking text on AI Chat's path.
    summary: bool
    #: The model's own ``max_tokens`` ceiling, 0 when the provider gives none.
    max_tokens: int = 0
    #: What Off sends as Anthropic's ``thinking`` value: ``disabled``, or the
    #: model's own off switch from `ANTHROPIC_OFF_SWITCHES`.
    off_type: str = "disabled"

    def view(self) -> Dict[str, Any]:
        return {
            "engine": self.engine, "steps": list(self.steps),
            "decided_by": self.decided_by, "summary": self.summary,
        }


@dataclass(frozen=True)
class ThinkingPlan:
    """One answer's thinking: the step sent, the fields and the ceiling."""

    step: str = ""
    capability: Optional[ThinkingCapability] = None
    #: Fields merged into the request body; empty sends nothing.
    body: Dict[str, Any] = field(default_factory=dict)
    max_tokens: int = 0

    @property
    def sent(self) -> bool:
        return bool(self.body)


def _supported(value: Any) -> bool:
    return isinstance(value, Mapping) and value.get("supported") is True


def _family_steps(table, model: str) -> Tuple[str, ...]:
    name = str(model or "").strip().lower()
    if name.startswith("models/"):
        name = name[len("models/"):]
    for family, steps in table:
        # A row ending in "." names a whole version line ("gemini-3." is 3.1,
        # 3.8 ...); any other row matches itself or a "-" snapshot or variant.
        if name == family or name.startswith(family if family.endswith(".") else family + "-"):
            return steps
    return ()


#: Anthropic models whose off switch is not ``thinking.type: "disabled"``. The
#: Models API's ``thinking.types`` has no field for it, so it is a row here,
#: from the docs (thinking-troubleshooting, read 2026-10-08): "Only Claude
#: Sonnet 5.5 accepts "between_tools", and it takes that value in place of
#: "disabled"" - it turns off up-front thinking, at effort high or below, and
#: takes no other field (no ``display``). Matched like the family tables.
ANTHROPIC_OFF_SWITCHES: Tuple[Tuple[str, str], ...] = (
    ("claude-sonnet-5-5", "between_tools"),
)


def _anthropic_off_switch(model: str) -> str:
    name = str(model or "").strip().lower()
    for family, off_type in ANTHROPIC_OFF_SWITCHES:
        if name == family or name.startswith(family + "-"):
            return off_type
    return ""


def anthropic_capability(info: Any) -> Optional[ThinkingCapability]:
    """A model's thinking from Anthropic's ``ModelInfo``, or None if it has none.

    Adaptive thinking with effort is preferred where the model takes it; the
    manual budget is used only where adaptive is absent (the 4.5-era models,
    which reject ``adaptive`` with a 400). Off is offered only where
    ``thinking.types.disabled`` is supported: Opus 5.5 and the other
    always-on models reject ``disabled`` with a 400, so they get no Off.
    """
    if not isinstance(info, Mapping):
        return None
    capabilities = info.get("capabilities")
    if not isinstance(capabilities, Mapping):
        return None
    thinking = capabilities.get("thinking")
    if not _supported(thinking):
        return None
    types = thinking.get("types") if isinstance(thinking.get("types"), Mapping) else {}
    effort = capabilities.get("effort")
    levels: Tuple[str, ...] = ()
    if _supported(types.get("adaptive")) and _supported(effort):
        engine = ENGINE_ANTHROPIC_ADAPTIVE
        levels = tuple(step for step in REASONING_STEPS if _supported(effort.get(step)))
    if not levels and _supported(types.get("enabled")):
        engine = ENGINE_ANTHROPIC_BUDGET
        levels = REASONING_STEPS
    if not levels:
        return None
    switch = _anthropic_off_switch(str(info.get("id") or ""))
    off_type = switch or "disabled"
    off = (STEP_OFF,) if switch or _supported(types.get("disabled")) else ()
    try:
        ceiling = max(0, int(info.get("max_tokens") or 0))
    except (TypeError, ValueError):
        ceiling = 0
    return ThinkingCapability(engine, off + levels, SOURCE_LISTING, True, ceiling, off_type)


def openrouter_capability(item: Any) -> Optional[ThinkingCapability]:
    """A model's thinking from OpenRouter's model list, or None.

    ``supported_efforts`` absent means "the model does not expose effort
    selection"; ``mandatory`` means the model rejects ``effort: "none"``, so it
    gets no Off. OpenRouter returns the reasoning in ``message.reasoning``.
    """
    reasoning = item.get("reasoning") if isinstance(item, Mapping) else None
    if not isinstance(reasoning, Mapping):
        return None
    efforts = reasoning.get("supported_efforts")
    if not isinstance(efforts, list):
        return None
    levels = tuple(step for step in REASONING_STEPS if step in efforts)
    if not levels:
        return None
    off = () if reasoning.get("mandatory") is True else (STEP_OFF,)
    return ThinkingCapability(ENGINE_OPENROUTER, off + levels, SOURCE_LISTING, True)


def table_capability(provider: str, model: str) -> Optional[ThinkingCapability]:
    """OpenAI's and Gemini's thinking from the family tables, or None.

    Neither returns a thinking summary on the endpoint AI Chat uses (OpenAI's
    summaries are Responses-API only; Gemini's compat response shape is not
    documented), so neither shows a thinking line (VD-209 item 4).
    """
    if provider == "openai" and _openai_excluded(model):
        return None
    table = {"openai": OPENAI_REASONING_FAMILIES, "gemini": GEMINI_REASONING_FAMILIES}.get(provider)
    steps = _family_steps(table, model) if table else ()
    if not steps:
        return None
    return ThinkingCapability(ENGINE_REASONING_EFFORT, steps, SOURCE_TABLE, False)


def effective_step(capability: Optional[ThinkingCapability], chosen: str) -> str:
    """The step this model is sent: the owner's, if the model takes it."""
    if capability is None:
        return ""
    if chosen in capability.steps:
        return chosen
    return default_step(capability)


def default_step(capability: Optional[ThinkingCapability]) -> str:
    """The step a connection starts on: Off if the model takes it, else Low."""
    if capability is None:
        return ""
    for step in DEFAULT_STEPS:
        if step in capability.steps:
            return step
    return capability.steps[0]


def _anthropic_budget(step: str, answer_tokens: int, ceiling: int, has_off: bool) -> Tuple[int, int]:
    """``(budget_tokens, max_tokens)`` with the budget below ``max_tokens``."""
    budget = ANTHROPIC_BUDGETS[step]
    total = budget + answer_tokens
    if ceiling and total > ceiling:
        total = ceiling
        budget = min(budget, total - MIN_ANSWER_ROOM)
    if budget < ANTHROPIC_MIN_BUDGET or budget >= total:
        raise ThinkingSettingError(
            "This model's output limit ({} tokens) leaves no room for a {} "
            "thinking budget. Choose {}.".format(
                ceiling, STEP_LABELS[step], "Off or a lower step" if has_off else "a lower step")
        )
    return budget, total


def plan(capability: Optional[ThinkingCapability], chosen: str, answer_tokens: int) -> ThinkingPlan:
    """The request fields and answer ceiling for one answer at ``chosen``.

    No capability sends nothing at all: a model that cannot think, or whose
    capability could not be read, is asked exactly as before VD-209.
    """
    step = effective_step(capability, chosen)
    if capability is None or not step:
        return ThinkingPlan(max_tokens=answer_tokens)
    ceiling = capability.max_tokens
    deeper = max(answer_tokens, STEP_ANSWER_TOKENS[step])
    if ceiling:
        deeper = min(deeper, ceiling)
    if capability.engine in (ENGINE_ANTHROPIC_ADAPTIVE, ENGINE_ANTHROPIC_BUDGET):
        if step == STEP_OFF:
            # No effort is sent with Off: each model's default is high or
            # below, where both "disabled" and "between_tools" are accepted.
            return ThinkingPlan(step, capability, {"thinking": {"type": capability.off_type}}, answer_tokens)
        if capability.engine == ENGINE_ANTHROPIC_BUDGET:
            budget, total = _anthropic_budget(step, answer_tokens, ceiling, STEP_OFF in capability.steps)
            return ThinkingPlan(step, capability, {
                "thinking": {"type": "enabled", "budget_tokens": budget},
            }, total)
        return ThinkingPlan(step, capability, {
            # Opus 4.7 and later default to "omitted" (empty thinking text);
            # the summary VD-209 item 4 shows has to be asked for.
            "thinking": {"type": "adaptive", "display": "summarized"},
            "output_config": {"effort": step},
        }, deeper)
    wire = "none" if step == STEP_OFF else step
    if capability.engine == ENGINE_OPENROUTER:
        return ThinkingPlan(step, capability, {"reasoning": {"effort": wire}}, deeper)
    return ThinkingPlan(step, capability, {"reasoning_effort": wire}, deeper)


def thinking_view(connection: Mapping[str, Any], model: str,
                  capability: Optional[ThinkingCapability], reason: str) -> Dict[str, Any]:
    """What the console is told about the control for this connection and model.

    ``available`` false hides the control; ``reason`` says why when the answer
    is "could not read" rather than "this model does not think".
    """
    view: Dict[str, Any] = {
        "credential_id": str(connection.get("credential_id") or ""),
        "model": model, "available": capability is not None,
        "steps": [], "summary": False, "reason": reason,
        "default_step": default_step(capability),
    }
    if capability is not None:
        view.update(capability.view())
    return view


def bounded_summary(text: Any) -> Tuple[str, bool]:
    """A thinking summary held to `THINKING_SUMMARY_CHARS`, and whether it was cut."""
    clean = str(text or "").strip()
    if len(clean) <= THINKING_SUMMARY_CHARS:
        return clean, False
    return clean[:THINKING_SUMMARY_CHARS].rstrip(), True


def thinking_result(plan_: ThinkingPlan, summary: Any, seconds: Optional[float]) -> Dict[str, Any]:
    """What an answer reports about its thinking; empty when none came back.

    ``seconds`` is measured from the stream (the first thinking block opening
    to the last one closing) and is None where the provider answered in one
    piece, so nothing can be measured: the line then shows no duration rather
    than a guessed one.
    """
    text, truncated = bounded_summary(summary)
    if not text or plan_.capability is None or not plan_.capability.summary:
        return {}
    result: Dict[str, Any] = {"summary": text, "step": plan_.step}
    if seconds is not None and seconds >= 0:
        result["seconds"] = round(float(seconds), 1)
    if truncated:
        result["truncated"] = True
    return result


def refused_setting(plan_: ThinkingPlan, status: int, detail: str) -> bool:
    """Whether a provider's 400 is a refusal of the thinking setting sent.

    Only when a setting was sent, and only when the provider's own reason
    names it: a 400 about something else must keep its own sentence rather
    than be blamed on the thinking step.
    """
    reason = (detail or "").lower()
    return plan_.sent and status == 400 and any(word in reason for word in REFUSAL_WORDS)


def refusal_sentence(company: str, model: str, plan_: ThinkingPlan, detail: str) -> str:
    """The chat error for a refused step: what was refused and what to do."""
    clean = clipped_reason(detail)  # detail arrives redacted; clip after
    reason = ": {}".format(clean) if clean else "."
    return (
        '{} refused the thinking setting "{}" for {}{} Choose another thinking '
        "step and retry; Vaelor did not send the question without it.".format(
            company or UNNAMED_SERVICE, STEP_LABELS.get(plan_.step, plan_.step),
            model or "this model", reason)
    )


class ThinkingCapabilities:
    """Capability reads per connection and model, cached for an hour.

    ``read(connection, model, send)`` answers ``(capability, reason)``. A
    failed read answers ``(None, sentence)`` and is retried after a minute:
    the control is hidden rather than shown with steps that may not exist.
    The key includes the credential version, so a replaced key reads again.
    """

    def __init__(self, *, clock: Callable[[], float] = time.monotonic):
        self._clock = clock
        self._cache: Dict[Tuple[str, str, str], Tuple[float, Optional[ThinkingCapability], str]] = {}

    def read(self, connection: Mapping[str, Any], model: str, send: Any) -> Tuple[Optional[ThinkingCapability], str]:
        provider = str(connection.get("provider") or "")
        if provider in ("openai", "gemini"):
            return table_capability(provider, model), ""
        if provider not in ("anthropic", "openrouter") or not model:
            return None, ""
        key = (
            str(connection.get("credential_id") or ""),
            str(connection.get("credential_version") or ""), model,
        )
        now = self._clock()
        cached = self._cache.get(key)
        if cached is not None and cached[0] > now:
            return cached[1], cached[2]
        try:
            capability = self._fetch(provider, connection, model, send)
            entry = (now + CAPABILITY_SECONDS, capability, "")
        except Exception:  # noqa: BLE001 - any failed read hides the control, and says so
            entry = (now + FAILED_READ_SECONDS, None,
                     "Vaelor could not read whether {} can think, so the thinking "
                     "control is hidden for now.".format(model))
        self._cache[key] = entry
        return entry[1], entry[2]

    @staticmethod
    def _fetch(provider: str, connection: Mapping[str, Any], model: str, send: Any) -> Optional[ThinkingCapability]:
        base_url = str(connection.get("base_url") or "").rstrip("/")
        api_key = str(connection.get("api_key") or "")
        if provider == "anthropic":
            url = "{}/models/{}".format(base_url, urllib.parse.quote(model, safe=""))
            with send("GET", url, headers=anthropic_headers(api_key), timeout=12) as response:
                return anthropic_capability(response.json())
        with send("GET", base_url + "/models", headers=bearer_headers(api_key), timeout=20) as response:
            payload = response.json()
        items = payload.get("data") if isinstance(payload, Mapping) else None
        for item in items if isinstance(items, list) else []:
            if isinstance(item, Mapping) and item.get("id") == model:
                return openrouter_capability(item)
        return None


#: The one cache AI Chat reads through, so the setup route and an answer agree.
CAPABILITIES = ThinkingCapabilities()
