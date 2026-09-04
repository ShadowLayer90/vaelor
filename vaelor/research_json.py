"""Robust JSON extraction from a possibly messy small-model reply.

Small models used for guarded application research fail structured-output tasks
intermittently (#247v): they wrap the answer in prose, a Markdown code fence, or
leading/trailing chatter. These helpers recover the intended JSON document
before the research module gives up and retries or falls back. They are generic
and provider-agnostic - no research or application knowledge lives here - and
deliberately separate from the shared ``parse_model_object`` so other callers
keep their exact contract while research gets the extra robustness locally.
"""

from __future__ import annotations

import json
import re
from typing import Any

# The balanced scan is O(n^2) in the worst case (a run of unbalanced openers,
# each scanned to end of string), and chat_completion reads up to 2 MB regardless
# of the requested max_tokens - a non-compliant or hostile endpoint could hand
# back a megabytes-long blob of "{" and burn minutes of CPU while holding the
# inference slot. A real research JSON document is a few hundred bytes, so cap
# the input well above any legitimate reply and refuse anything larger with the
# same content-flake ValueError the empty/no-JSON cases raise - the caller then
# retries or falls back deterministically rather than stalling.
_MAX_EXTRACTION_CHARS = 256 * 1024


def extract_json_document(content: Any) -> Any:
    """Pull a JSON object/array out of a possibly messy model reply.

    Strips a leading/closing Markdown code fence, tries a straight parse, then
    scans for the outermost balanced ``{...}`` or ``[...]`` document and returns
    the first that parses. Raises ``ValueError`` when nothing parseable is
    present - including an over-large input, refused before the balanced scan -
    so the caller can retry or fall back.
    """
    if isinstance(content, (dict, list)):
        return content
    text = str(content or "").strip()
    if not text:
        raise ValueError("The connected AI endpoint returned an empty response.")
    if len(text) > _MAX_EXTRACTION_CHARS:
        raise ValueError("The connected AI endpoint returned a response too large to parse as JSON.")
    if text.startswith("```"):
        text = re.sub(r"^```[^\n`]*\n?", "", text, count=1)
        text = re.sub(r"\n?```\s*$", "", text, count=1).strip()
    try:
        parsed = json.loads(text)
    except json.JSONDecodeError:
        parsed = first_balanced_json(text)
    if not isinstance(parsed, (dict, list)):
        raise ValueError("The connected AI endpoint returned no JSON document.")
    return parsed


def first_balanced_json(text: str) -> Any:
    """Return the first balanced ``{...}``/``[...]`` in ``text`` that parses.

    A character scan that tracks string literals and escapes, so braces inside
    quoted strings do not confuse the depth count. Returns ``None`` when no
    balanced, parseable JSON document is found.
    """
    openers = {"{": "}", "[": "]"}
    for start, opener in enumerate(text):
        closer = openers.get(opener)
        if closer is None:
            continue
        depth = 0
        in_string = False
        escaped = False
        for index in range(start, len(text)):
            char = text[index]
            if in_string:
                if escaped:
                    escaped = False
                elif char == "\\":
                    escaped = True
                elif char == '"':
                    in_string = False
                continue
            if char == '"':
                in_string = True
            elif char == opener:
                depth += 1
            elif char == closer:
                depth -= 1
                if depth == 0:
                    try:
                        return json.loads(text[start:index + 1])
                    except json.JSONDecodeError:
                        break  # unbalanced/invalid here; try the next opener
    return None
