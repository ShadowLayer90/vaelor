"""What part of a model's reply AI Chat shows, and how saved memory is quoted.

Split from :mod:`vaelor.chat_inference` (VD-206) to keep that module under the
1,000-line production ceiling; `chat_inference` re-exports every name here, so
importers keep spelling them there. Pure text handling: no I/O, no model call.
"""

from __future__ import annotations

import re


# Reasoning models narrate to themselves before answering. Whether that
# narration reaches the user depends entirely on the serving stack, and when it
# does it arrives glued to the reply - "The user asks ... Must explain how to
# check.I don't have access to your sensors" - which reads as the assistant
# talking about the user in the third person. These are the shapes that can be
# separated deterministically; see visible_answer for the shape that cannot.
_CLOSING_REASONING_TAG = re.compile(
    r"<\s*/\s*(?:think|thinking|reason|reasoning|analysis|scratchpad)\s*>",
    re.IGNORECASE,
)
_REASONING_TAG = re.compile(
    r"<\s*/?\s*(?:think|thinking|reason|reasoning|analysis|scratchpad)\s*>",
    re.IGNORECASE,
)
# The harmony format used by gpt-oss models labels each segment with a channel.
# Only the final channel is meant to be read.
_HARMONY_FINAL = re.compile(
    r"<\|channel\|>\s*final\s*<\|message\|>(.*?)(?:<\|end\|>|<\|return\|>|\Z)",
    re.IGNORECASE | re.DOTALL,
)
_HARMONY_HIDDEN = re.compile(
    r"<\|channel\|>\s*(?:analysis|commentary)\s*<\|message\|>"
    r".*?(?:<\|end\|>|<\|return\|>|(?=<\|channel\|>)|\Z)",
    re.IGNORECASE | re.DOTALL,
)
_HARMONY_MARKER = re.compile(r"<\|[a-z_]+\|>", re.IGNORECASE)
# A complete narration block, matched by its own tag name so `<think>...
# </thinking>` is not treated as a pair.
_REASONING_BLOCK = re.compile(
    r"<\s*(think|thinking|reason|reasoning|analysis|scratchpad)\s*>"
    r".*?<\s*/\s*\1\s*>",
    re.IGNORECASE | re.DOTALL,
)
# Fenced and inline code, where a reasoning tag is content the user asked
# about rather than narration the serving stack emitted.
_CODE_REGION = re.compile(r"```.*?(?:```|\Z)|`[^`\n]*`", re.DOTALL)


def _code_spans(text: str):
    return [(match.start(), match.end()) for match in _CODE_REGION.finditer(text)]


def _replace_outside_code(pattern, replacement: str, text: str) -> str:
    """Apply a substitution everywhere except inside code."""
    spans = _code_spans(text)
    if not spans:
        return pattern.sub(replacement, text)
    result = []
    cursor = 0
    for start, end in spans:
        result.append(pattern.sub(replacement, text[cursor:start]))
        result.append(text[start:end])
        cursor = end
    result.append(pattern.sub(replacement, text[cursor:]))
    return "".join(result)


def _first_close_outside_code(text: str):
    spans = _code_spans(text)
    for match in _CLOSING_REASONING_TAG.finditer(text):
        if not any(start <= match.start() < end for start, end in spans):
            return match
    return None


def visible_answer(content: str, reasoning: str = "") -> str:
    """Return only the part of a completion the user is meant to read.

    Three separable cases are handled: a harmony `final` channel, a closed
    reasoning tag, and a server that returns the scratchpad in its own field
    and *also* prepends it to the content.

    Every rule here is bounded so it can only ever remove narration. Complete
    blocks are removed in place, so a model that narrates twice keeps both
    pieces of real answer between them. A dangling close - the shape a serving
    stack leaves when it swallows the opening tag - cuts at the *first* one,
    because everything after the last one is only the tail of a multi-block
    reply. A close with nothing after it is not a cut at all, or
    "Answer here.</think>" would return nothing. And a tag inside a code fence
    or inline code is text the user asked about: `</think>` in a snippet about
    reasoning tags stays exactly where it is.

    A model that emits untagged reasoning into a plain content string is not
    separable here - there is no marker to cut on, and guessing at sentence
    boundaries would truncate real answers. That case needs the serving stack
    to label its output.
    """
    text = str(content or "")
    final = _HARMONY_FINAL.search(text)
    if final:
        text = final.group(1)
    else:
        text = _HARMONY_HIDDEN.sub("", text)
    text = _HARMONY_MARKER.sub("", text)
    text = _replace_outside_code(_REASONING_BLOCK, "", text)
    closing = _first_close_outside_code(text)
    if closing and text[closing.end():].strip():
        text = text[closing.end():]
    elif closing:
        text = text[:closing.start()]
    text = _replace_outside_code(_REASONING_TAG, "", text)
    narration = str(reasoning or "").strip()
    if narration:
        candidate = text.lstrip()
        if candidate.startswith(narration):
            text = candidate[len(narration):]
    return text.strip()


# Curated memory is reviewed content, but it is still text that was typed into
# this appliance, and it must not be able to write the structure of the prompt
# it lands in. Two shapes matter. An [S#] marker claims to be a citation into a
# retrieved passage the user can open in the citation list - and a forged one
# is not in that list, so the user is shown a reference they cannot check. A
# block heading claims the text after it is a different kind of input
# altogether, which is how one memory could append its own "Retrieved sources:"
# section containing whatever it liked. Both are removed, and each memory is
# flattened to a single line so it cannot introduce structure at all.
_CITATION_MARKER = re.compile(r"\[\s*S\s*\d+\s*\]", re.IGNORECASE)
_PROMPT_HEADING = re.compile(
    r"(?:retrieved\s+sources|saved\s+appliance\s+memory[^:\n]*)\s*:", re.IGNORECASE
)


def neutralized_memory(content: str) -> str:
    """One memory, reduced to something that can only be read as a memory."""
    text = _CITATION_MARKER.sub("[citation removed]", str(content or ""))
    text = _PROMPT_HEADING.sub("(heading removed)", text)
    return " ".join(text.split())
