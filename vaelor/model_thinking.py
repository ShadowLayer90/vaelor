"""Whether a served cluster model thinks before it answers: the one owner of the switch.

Qwen3 thinks by default: before its answer it writes a ``<think>`` block that
can run to thousands of tokens. On the two-node cluster (Qwen/Qwen3-8B in BF16,
about 13 tokens a second per stream) that made a short LAN request take
29 seconds and a longer one run past the 600 seconds an app waited for it. So a
cluster deployment serves with thinking OFF unless its owner turns it on, and a
client can still ask for it on one request (owner decision, 2026-09-29).

**The mechanism is vLLM's own, read at the pinned version (LESSONS 15).** vLLM
0.22.1's ``--default-chat-template-kwargs`` (``FrontendArgs`` in
``vllm/entrypoints/openai/cli_args.py``, parsed with ``json.loads``) gives the
chat template default keyword arguments, and the chat server merges them under
a request's own ``chat_template_kwargs`` with the request's values winning
(``ChatParams.with_defaults`` calling ``merge_kwargs(defaults, overrides)`` in
``vllm/renderers/params.py``). It is spelled here in vLLM's dotted form,
``--default-chat-template-kwargs.enable_thinking=false``, which
``FlexibleArgumentParser.parse_args`` turns into the JSON object itself
(``json.loads("false")`` is the boolean) - so no brace or quote ever reaches a
systemd ``ExecStart`` or the lead's ``bash -c`` script, the same form the unit
already uses for ``--profiler-config.profiler=torch``. What vLLM does NOT do:
know which templates read the key. A template that never mentions
``enable_thinking`` ignores it, so the flag would be harmless there, but the
setting would be a switch that does nothing and the console would be claiming a
behaviour it cannot deliver.

**This module owns the setting; the family table owns the switch.** Whether a
model's template takes ``enable_thinking`` is `vllm_model_profile`'s answer
(``has_thinking_switch``), the one table that also says which tool-call and
reasoning parsers a family speaks - so the thinking flag and the parser that
reads ``<think>`` can never disagree about a model. Only the Qwen3 chat
releases whose template was read get the flag; the 2507 split releases, Next,
Omni and the coder, vision, base, embedding, reranker and guard variants carry
no switch and are left alone, as is every other family.
"""

from __future__ import annotations

from typing import Any, Dict, List, Mapping, Optional

from . import vllm_model_profile

#: What a new deployment, and an existing one whose record predates the setting,
#: serves with: no thinking. An existing deployment picks this up the next time
#: its units are rendered (a Load, or a fresh deploy).
THINKING_DEFAULT = False

#: The payload and record key the setting travels under.
THINKING_FIELD = "thinking"

#: The chat-template keyword Qwen3's template reads, and the vLLM flag that
#: gives it a server-side default a request can override.
TEMPLATE_KEYWORD = "enable_thinking"
DEFAULT_KWARGS_FLAG = "--default-chat-template-kwargs"

#: The one sentence a setting that is not a real boolean is refused with.
THINKING_NOT_BOOLEAN = "The thinking setting must be true or false."


def require_boolean(value: Any) -> bool:
    """``value`` when it is ``True`` or ``False``; refused otherwise, never read for truthiness."""
    if not isinstance(value, bool):
        raise ValueError(THINKING_NOT_BOOLEAN)
    return value


#: Plain words for the deploy form, from this module so the console never
#: guesses which models the setting reaches.
SWITCH_DETAIL = (
    "Off by default: the model answers straight away. An app can still ask for "
    "thinking on one request."
)
NO_SWITCH_DETAIL = (
    "This model has no thinking switch, so it answers the way its makers "
    "trained it; this setting does not apply."
)


#: What a deploy that turns thinking ON for a model without the switch is
#: refused with. :data:`NO_SWITCH_DETAIL` describes the model to someone
#: looking at a form; this answers someone whose request was turned down, so
#: it says that it was, why, and what to send instead.
THINKING_REFUSED = (
    "The deploy was refused: it asked for thinking to be turned on, and this "
    "model has no thinking switch, so it would have recorded a setting the "
    "model cannot follow. Send the deploy again without the thinking setting."
)


def has_thinking_switch(repo: Any) -> bool:
    """Whether ``org/name``'s chat template reads ``enable_thinking`` (the family table's answer)."""
    return vllm_model_profile.has_thinking_switch(repo)


def thinking_from_payload(payload: Mapping[str, Any]) -> bool:
    """The thinking setting a deploy asked for: a real boolean, or the default.

    Absent (or ``None``) is the default, off. Anything but ``True`` or
    ``False`` is refused rather than read for truthiness: the string
    ``"false"`` is truthy, and turning thinking ON because a client sent the
    word "false" is the defect this refuses.
    """
    value = (payload or {}).get(THINKING_FIELD)
    if value is None:
        return THINKING_DEFAULT
    return require_boolean(value)


def thinking_for_deploy(repo: Any, payload: Mapping[str, Any]) -> bool:
    """The thinking setting a deploy of ``repo`` serves with, refused where it cannot apply.

    `thinking_from_payload`, and then one more refusal: thinking turned ON for
    a model whose template has no switch. Such a deploy used to be accepted
    and recorded as "thinking on" while nothing was rendered for it - a
    setting the record claimed and the model never had (the 0.27 benchmark's
    Instruct-2507 finding, 2026-09-30). Off, or absent, is what every model
    without the switch already does, so it is accepted and recorded as off.
    """
    thinking = thinking_from_payload(payload)
    if thinking and not has_thinking_switch(repo):
        raise ValueError(THINKING_REFUSED)
    return thinking


def thinking_from_record(units: Mapping[str, Any]) -> bool:
    """The setting a stored deployment serves with; a record without one is off.

    A record written before the setting existed carries no key and serves with
    the default from its next Load on - it never inherits the old
    think-by-default behaviour. A stored value that is not a boolean reads as
    the default too: the record is Vaelor's own, and off is the safe answer.
    """
    value = (units or {}).get(THINKING_FIELD)
    return value if isinstance(value, bool) else THINKING_DEFAULT


def unit_thinking_value(repo: Any, thinking: bool) -> Dict[str, bool]:
    """The ``thinking_default`` a managed unit is sent, or nothing for a model without the switch.

    Only a model that renders the flag carries the value, so a Load of any
    other model sends exactly the values an older root bridge already takes.
    """
    require_boolean(thinking)
    return {"thinking_default": thinking} if has_thinking_switch(repo) else {}


def thinking_on_record(units: Mapping[str, Any]) -> Optional[bool]:
    """What a stored deployment's units say, ``None`` when they predate the setting.

    For the console only: a record without the key is still serving with
    whatever its old unit was rendered with (a Qwen3 template's own default is
    to think) until a Load re-renders it, so the row must not claim "off".
    """
    value = (units or {}).get(THINKING_FIELD)
    return value if isinstance(value, bool) else None


def serve_thinking_arguments(repo: Any, thinking: bool) -> List[str]:
    """The ``vllm serve`` word that sets the default, or none for a family without the switch.

    Spelled explicitly both ways, so the unit says what the deployment was set
    to rather than leaning on whatever a template's own default is.
    """
    require_boolean(thinking)
    if not has_thinking_switch(repo):
        return []
    return [
        "{}.{}={}".format(
            DEFAULT_KWARGS_FLAG, TEMPLATE_KEYWORD, "true" if thinking else "false",
        )
    ]


def thinking_switch(repo: Any) -> Dict[str, Any]:
    """What the deploy form shows about the setting for one model."""
    supported = has_thinking_switch(repo)
    return {
        "switch": supported,
        "default": THINKING_DEFAULT,
        "detail": SWITCH_DETAIL if supported else NO_SWITCH_DETAIL,
    }
