"""Reading the NPU fine-tune's JSON contract reply into plain lines (VD-219).

The fine-tune (ff-4b-h) answers in the shape it was trained on - one JSON
object with summary, findings, recommendations, next_actions and the sources
last - and its replies are often not JSON: cut off, a key without its colon, a
second object, unescaped quotes inside a value, markdown bullets inside the
braces. So the reply is read by scanning, not parsing, and only what the owner
can use is shown: the summary, then the details, finished sentences only,
never a key, a brace or the sources (those become evidence), with every
appliance id said in words. :mod:`vaelor.assistant_local_answer` owns the
calls and the judges; this module only reads.
"""

from __future__ import annotations

import json
import re
from typing import Any, Dict, List, Tuple

from .agent_prompts import ASSISTANT_HAND_OFF, LOCAL_ASSISTANT_PROMPT

#: Keys a JSON-shaped reply uses to carry the actual answer sentence, richest
#: first. The 4B most often leads a real answer with ``summary``.
LEAD_KEYS = ("answer", "summary", "response", "reply", "text", "message", "result")

#: Key spellings that are a lead key with a suffix (``summary_text``).
_LEAD_STEMS = ("summary", "answer")

#: List-valued keys whose entries expand on the lead sentence.
DETAIL_KEYS = (
    "details", "findings", "notes", "issues", "recommendations", "checks",
    "next_actions", "actions", "observations",
)

#: Keys that only say where an answer came from. A reply with nothing else is
#: no answer (R126); beside an answer, they are its evidence.
CITATION_KEYS = frozenset((
    "source", "sources", "sources_used", "evidence", "evidence_used",
    "tools", "tools_used", "source_list", "citations", "references",
))

#: What a reading the answer cites says in the evidence line.
CITED_READING = "The answer cites this reading."

#: Where a cut-off value may still be shown up to: a finished sentence.
_SENTENCE_STOPS = (". ", "! ", "? ", ".\n", "!\n", "?\n")

#: A word before a full stop that does not end a sentence there.
_NOT_SENTENCE_ENDS = frozenset(("etc", "vs", "approx", "ca", "no", "fig", "incl", "eg", "ie"))

#: Words a reading's id is not spelled with in a sentence.
_SHOUTED = {"gpu": "GPU", "npu": "NPU", "cpu": "CPU", "ram": "RAM", "ssd": "SSD", "apu": "APU"}

#: What may follow a string's closing quote; anything else makes it an inner quote.
_AFTER_A_STRING = ",:}]"


def _local():
    """The judges this reader shares with :mod:`vaelor.assistant_local_answer` (imported late)."""
    from . import assistant_local_answer
    return assistant_local_answer


def is_lead_key(key: str) -> bool:
    """``summary``, ``answer`` ... or one of them with a suffix (``summary_text``)."""
    key = key.strip().lower()
    return key in LEAD_KEYS or key.startswith(tuple(stem + "_" for stem in _LEAD_STEMS))


def _read_string(text: str, start: int, quote: str = '"', settled: bool = True) -> Tuple[str, int, bool]:
    """The string opening at ``text[start]``, the index after it, and whether it closed.

    VD-219 / R137 (LESSONS 1): the fine-tune writes unescaped quotes inside a
    value (``likelihood "unlikely" but``) in 7 of 44 replies. A quote closes the
    string only when what follows it is ``, : } ]``, another string, a line
    break or the end of the reply; otherwise it is part of the value. While the
    reply is still being written (``settled`` false) a quote at the very end
    has not shown which it is, so the string is not finished yet.
    """
    out: List[str] = []
    index = start + 1
    escapes = {"n": "\n", "t": "\t", "r": "\r", "b": "", "f": "", "/": "/", '"': '"', "'": "'", "\\": "\\"}
    while index < len(text):
        char = text[index]
        if char == quote:
            after = index + 1
            while after < len(text) and text[after] in " \t":
                after += 1
            following = text[after] if after < len(text) else ""
            if following == "" and not settled:
                return "".join(out), len(text), False
            if following in ("", "\r", "\n", quote) or following in _AFTER_A_STRING:
                return "".join(out), index + 1, True
            out.append(char)
            index += 1
            continue
        if char == "\\" and index + 1 < len(text):
            code = text[index + 1]
            if code == "u" and index + 5 < len(text):
                try:
                    out.append(chr(int(text[index + 2:index + 6], 16)))
                    index += 6
                    continue
                except ValueError:
                    pass
            out.append(escapes.get(code, code))
            index += 2
            continue
        out.append(char)
        index += 1
    return "".join(out), index, False


def _next_mark(text: str, index: int) -> Tuple[str, int]:
    """The next non-space character at or after ``index``, and where it is."""
    while index < len(text) and text[index] in " \t\r\n":
        index += 1
    return (text[index], index) if index < len(text) else ("", index)


def _looks_like_key(value: str) -> bool:
    """A short label - ``recommendation``, ``what-to-check-next`` - not a sentence."""
    words = value.replace("_", " ").replace("-", " ").split()
    return 0 < len(words) <= 4 and len(value) <= 40 and all(
        word.replace(".", "").isalnum() for word in words) and not value.endswith(".")


def _quote_of(text: str) -> str:
    """``'`` for a Python-literal reply (``{'answer': ...``), else ``"``."""
    head = text.lstrip()
    return "'" if head.startswith("{") and head[1:].lstrip().startswith("'") else '"'


def contract_values(text: str, settled: bool = True) -> List[Tuple[str, str, bool]]:
    """``(top-level key, string value, finished)`` for each string a JSON-shaped reply answers with.

    VD-219 (LESSONS 1, 26). Every string that is the value of a top-level key,
    or an item of a list that is, is kept with that key; so is a string under
    a lead key one object down (``{"result":{"summary":...}}``). Anything nested
    deeper - a reading the model copied into its answer - is not. A key written
    without its colon (``"recommendation","No action needed"``) is still read as
    a key; a markdown ``- `` bullet line outside any string is kept under ``""``.
    """
    values: List[Tuple[str, str, bool]] = []
    stack: List[str] = []
    quote = _quote_of(text)
    top, inner, after_colon, index = "", "", False, 0
    while index < len(text):
        char = text[index]
        if char == quote:
            value, index, finished = _read_string(text, index, quote, settled)
            mark, at = _next_mark(text, index)
            if mark == ":":
                if len(stack) == 1:
                    top = value
                elif stack == ["{", "{"]:
                    inner = value
                index, after_colon = at + 1, len(stack) == 1
                continue
            if stack == ["{"] and not after_colon and (mark == "," and _looks_like_key(value) or mark == ""):
                if mark:
                    top = value  # a key written without its colon
                # at the end of the text: a key whose colon is not written yet
            elif stack == ["{"] or stack == ["{", "["]:
                values.append((top, value, finished))
            elif stack == ["{", "{"] and is_lead_key(inner):
                values.append((inner, value, finished))
            after_colon = False
            continue
        if char == "{" and stack == ["{"] and not after_colon:
            # VD-219: a second object opened inside an unclosed first one
            # ('{"summary":"..."\n{"findings":[...]}') is no key's value - it
            # carries the same top-level keys. Read nested, its findings were
            # lost and "explain every reading" showed its summary alone.
            after_colon = False
        elif char in "{[":
            stack.append(char)
            after_colon = False
        elif char in "}]":
            if stack:
                stack.pop()
            after_colon = False
        elif char == "\n" and len(stack) <= 1:
            line = text[index + 1:].split("\n", 1)[0]
            bare = line.strip()
            if bare.startswith(("- ", "* ")) and quote not in bare and len(bare.split()) >= 4:
                values.append(("", bare[2:].strip(), True))
                index += len(line)
        elif char == ",":
            after_colon = False
        index += 1
    return values


def contract_end(text: str) -> int:
    """Where a contract reply starts repeating itself, or ``-1`` while it has not.

    VD-219 (LESSONS 26): measured on the Z2, 12 of 44 contract answers never
    stopped - after the sources the fine-tune writes ``clarifications`` and
    ``sources`` again and again until the token limit. Its answer is over when
    a top-level key it already wrote starts again - in the same object or a
    second one; the text before that point is the whole answer. A closing
    brace is not the end: one answer closed its object early and went on with
    new keys in a second one.
    """
    stack: List[str] = []
    seen: set = set()
    index = 0
    while index < len(text):
        char = text[index]
        if char == '"':
            value, after, _finished = _read_string(text, index, '"', settled=False)
            mark, _at = _next_mark(text, after)
            if mark == ":" and stack == ["{"]:
                key = value.strip().lower()
                if key in seen:
                    comma = text.rfind(",", 0, index)
                    return comma if comma > 0 else index
                seen.add(key)
            index = after
            continue
        if char in "{[":
            stack.append(char)
        elif char in "}]" and stack:
            stack.pop()
        index += 1
    return -1


def ends_sentence(text: str) -> bool:
    """Whether ``text`` ends a sentence: ``!``, ``?``, or a full stop that is not "0." or "e.g.".

    VD-219 / R139: a value cut after "at 0." (the reply was going on to "0.5%")
    or after "e.g." ends on a full stop and is still half a sentence.
    """
    text = text.rstrip()
    if text.endswith(("!", "?")):
        return True
    if not text.endswith(".") or not text.split():
        return False
    core = text.split()[-1].rstrip(".")
    return bool(core) and not core[-1].isdigit() and "." not in core and not (
        len(core) == 1 and core.islower()) and core.lower().strip("(") not in _NOT_SENTENCE_ENDS


def finished_part(value: str, finished: bool, settled: bool = True) -> str:
    """``value`` if it closed; else only its finished sentences (never half of one).

    ``settled`` - the reply has ended - lets a cut-off value end on its own full
    stop; while it is still being written, "at 0." may yet become "at 0.5%".
    """
    value = value.strip()
    if finished:
        return value
    if settled and ends_sentence(value):
        return value
    cuts = [index for index in range(len(value) - 1) if value[index:index + 2] in _SENTENCE_STOPS]
    for cut in reversed(cuts):
        if ends_sentence(value[:cut + 1]):
            return value[:cut + 1].strip()
    return ""


def _without_tool_label(line: str) -> str:
    """``line`` without a leading ``gpu.status:`` - the owner has no such surface."""
    label, colon, rest = line.partition(":")
    if colon and rest.strip() and " " not in label.strip() and _local()._is_internal_name(label):
        return rest.strip()
    return line


def _word_parts(core: str, joiner: str) -> List[str]:
    """``core`` split on ``joiner`` when every part is a word-led id part, else ``[]``."""
    parts = core.split(joiner)
    if len(parts) < 2 or any(not part or not part[0].isalpha() or not part.replace("_", "").isalnum()
                             or len(part) < 2 for part in parts):
        return []
    return parts


def _plain_name(token: str) -> str:
    """``token`` with an id in it said as words: ``services.status`` -> "the services reading".

    VD-219 / R118, R139 (LESSONS 1): the Assistant's rule is never to name a
    tool - the owner has no such surface. A registered id is said as the reading
    it is; an id the model invented (``inference.uptime``) or a field name
    (``reboot_required``) is said as its words. Versions, numbers, paths,
    addresses and hyphenated unit names are left as written.
    """
    core = token.lstrip("([{\"'`*").rstrip(",;:?!.)]}\"'`*=")
    # VD-219 / kb R156: "cooling.status's" and "system/telemetry" leaked: a
    # possessive and a slash joint are still the id.
    possessive = core[-2:].lower() in ("'s", "’s") and len(core) > 2
    if possessive:
        core = core[:-2]
    if "/" in core and core.replace("/", ".").lower() in _local()._internal_names()[0]:
        return token.replace(core, _plain_name(core.replace("/", ".")))
    if not core or ":" in core or "/" in core:
        return token
    name = core.lower()
    exact = _local()._internal_names()[0]
    owner = next((name[:cut] for cut in range(len(name), 0, -1)
                  if name[:cut] in exact and (cut == len(name) or name[cut] in "._")), "")
    if owner and "." in owner:
        words = [_SHOUTED.get(word, word) for word in owner.replace("-", " ").replace("_", " ").split(".")
                 if word != "status"]
        rest = core[len(owner):].lstrip("._").replace("_", " ").replace(".", " ")
        return token.replace(core, " ".join(filter(None, ["the {} reading".format(" ".join(words)), rest])))
    parts = _word_parts(core, ".") or _word_parts(core, "_")
    if not parts:
        return token
    said = " ".join(_SHOUTED.get(word.lower(), word) for part in parts for word in part.split("_"))
    return token.replace(core, said)


#: An id the model spelled with a space or a stray mark in its joint:
#: ``cooling. status``, ``metrics . history``, ``inference .status``, and
#: ``Cooling. Status`` with both halves capitalised (kb R156). A sentence that
#: ends on lowercase "cooling." and starts the next with "Status" is left alone.
_SPLIT_ID = re.compile(r"\b([A-Za-z][a-z_]*)\s*\.\s*([a-z][a-z_]*)\b")

#: Both halves capitalised, joined only where a sentence starts: "on Storage.
#: Status of the NPU" is two sentences (kb R159). Not preceded by a word and a
#: space, nor by a word character - so the start of a line, or after a stop.
_SPLIT_CAPITAL = re.compile(r"(?<![\w,)\]] )(?<!\w)([A-Z][a-z_]*)\.\s+([A-Z][a-z_]*)\b")


#: An article the model wrote before an id that is now said as "the ... reading".
_THE_THE = re.compile(r"\b([Tt]he) the (?=\S+ (?:\S+ )?reading\b)")

#: "the system.telemetry reading" said as "the system telemetry reading reading" (kb R156).
_READING_READING = re.compile(r"\breading reading\b")


def _joined_ids(line: str) -> str:
    """``line`` with a registered id spelled apart written back as the id (VD-219).

    The grade of 2026-10-17 (kb N143) found ``cooling. status`` and ``metrics.
    history`` on the page: split on the space, neither half is an id, so the
    reader said neither in words. Only a pair that joins into a registered name
    is joined - "the disk is fine. status lights" stays as written.
    """
    exact = _local()._internal_names()[0]

    def join(match: "re.Match[str]") -> str:
        first, second = match.group(1), match.group(2)
        name = "{}.{}".format(first, second)
        return name.lower() if name.lower() in exact else match.group(0)

    return _SPLIT_CAPITAL.sub(join, _SPLIT_ID.sub(join, line))


def said_plainly(line: str) -> str:
    """``line`` on one line, with every id said as words; ``""`` when no word is left."""
    said = " ".join(_plain_name(token) for token in _joined_ids(line).split())
    # "Do the updates.status figures" said as "the updates reading" (VD-219).
    said = _READING_READING.sub("reading", _THE_THE.sub(r"\1 ", said))
    line = said[:1].upper() + said[1:] if said[:4] == "the " and not line.startswith("the ") else said
    return line if any(len(word.strip(".,;:!?()'\"")) > 1 and word.strip(".,;:!?()'\"").isalpha()
                       for word in line.replace("-", " ").split()) else ""


#: How many words in a row a detail line may share with the Assistant's own
#: rules before it is the rules read back rather than an answer.
_ECHO_RUN = 6


def _words(text: str) -> List[str]:
    return re.findall(r"[a-z0-9]+", text.lower())


def _runs(words: List[str]) -> set:
    return {tuple(words[index:index + _ECHO_RUN]) for index in range(len(words) - _ECHO_RUN + 1)}


#: What the rules tell the model to *say*: a line using it is following the
#: rules, not reading them back (kb R156: "the store is empty or unavailable").
#: (Each is at least six words, the echo run, or excusing it would excuse nothing.)
_PRESCRIBED = ("the store is empty or unavailable", ASSISTANT_HAND_OFF)

_RULE_RUNS = frozenset(_runs(_words(LOCAL_ASSISTANT_PROMPT))
                       - {run for phrase in _PRESCRIBED for run in _runs(_words(phrase))})


def echoes_rules(line: str) -> bool:
    """Whether ``line`` reads the Assistant's own rules back (VD-219).

    Graded 2026-10-17 (kb N143): asked to describe its readings at length, the
    4B filled the page with "the metrics history reading will downsample samples
    over the window when ... the store is off, unreadable, or empty" - its
    system prompt, not this machine. Six words in a row from the rules is that
    copy; an answer says what the readings say, in its own words.
    """
    return bool(_runs(_words(line)) & _RULE_RUNS)


#: The hand-off sentence wherever it stands in a line, in any case or punctuation
#: (R217: the prompt asks for it last, so it often ends a line of answer).
_HAND_OFF_SAID = re.compile(
    r"(?<![a-z0-9])" + r"[^a-z0-9\n]*".join(re.escape(word) for word in _words(ASSISTANT_HAND_OFF))
    + r"(?![a-z0-9])[.!?]*", re.IGNORECASE)


def is_hand_off(text: Any) -> bool:
    """Whether a reply's whole answer is the Assistant's hand-off to AI Chat (VD-220 B4).

    The answer as the owner would read it - after :func:`contract_text` - when
    it says :data:`~vaelor.agent_prompts.ASSISTANT_HAND_OFF`, in any case or
    punctuation, and nothing else. A reply that says anything else as well
    answered part of the question (R209, R217): :func:`without_hand_off` keeps it.
    """
    text = str(text or "")
    return bool(_HAND_OFF_SAID.search(text)) and not _words(_HAND_OFF_SAID.sub(" ", text))


def without_hand_off(text: Any) -> str:
    """``text`` without the hand-off sentence, wherever it stands; unchanged when it has none (R209, R217)."""
    text = str(text or "")
    if not _HAND_OFF_SAID.search(text):
        return text
    lines = (" ".join(_HAND_OFF_SAID.sub(" ", line).split()) for line in text.split("\n"))
    return "\n".join(line for line in lines if line).strip()


def contract_text(text: Any, *, settled: bool = True) -> str:
    """A JSON-shaped reply's answer as plain lines: the summary, then the details.

    VD-219. ``""`` when the reply carries no answer key (:func:`is_lead_key`) -
    a bare reading or a copy is judged by the caller. The sources are never
    shown (:func:`cited_evidence` turns them into evidence); a value cut off
    mid-sentence shows only its finished sentences, and no full stop is added
    to it; a line that is only internal names, or a citation sentence, is
    dropped; a leading tool name is taken off; repeats are shown once.
    ``settled=False`` is the streamed draft's reading of a reply still being
    written.
    """
    lead, details = "", []
    local = _local()
    for key, value, finished in contract_values(str(text or ""), settled):
        key = key.strip().lower()
        line = _without_tool_label(finished_part(value, finished, settled))
        if (not line or key in CITATION_KEYS or line.lower().startswith(("source:", "sources:"))
                or local.internal_names_only(line)):
            continue
        line = said_plainly(line)
        if not line:
            continue
        if not lead and is_lead_key(key):
            # A cut value is already its finished sentences; only a closed one may lack its mark.
            lead = line if line.endswith((".", "!", "?")) else line + "."
        elif line not in details and not echoes_rules(line):
            details.append(line)
    if not lead:
        return ""
    return "\n".join([lead] + [line for line in details if line != lead])


def cites_only(content: Any) -> bool:
    """Whether a reply is nothing but a sources list - no answer at all (R126)."""
    values = contract_values(_local()._strip_code_fence(str(content or "")))
    keys = {key.strip().lower() for key, _value, _finished in values}
    return bool(values) and keys <= CITATION_KEYS


def _scalars(value: Any) -> List[Any]:
    if isinstance(value, dict):
        return [item for inner in value.values() for item in _scalars(inner)]
    if isinstance(value, list):
        return [item for inner in value for item in _scalars(inner)]
    return [] if value in (None, True, False, "") else [value]


def copies_reading(content: Any, sent: str) -> bool:
    """Whether a reply with no answer key is a reading the model was sent, copied (R130).

    True when it parses to an object with no lead key and at least four in five
    of its values appear, as written, in the turn ``sent`` - "Hi" answered with
    the machine's identity object, word for word.
    """
    local = _local()
    parsed = local._parse_object(local._strip_code_fence(str(content or "").strip()))
    if not isinstance(parsed, dict) or any(is_lead_key(str(key)) for key in parsed):
        return False
    values = _scalars(parsed)
    found = sum(1 for value in values
                if json.dumps(value, ensure_ascii=False) in sent or str(value) in sent)
    return len(values) >= 2 and found * 5 >= len(values) * 4


def cited_evidence(content: Any, sent: str) -> List[Dict[str, str]]:
    """The readings a reply's sources list names that were really sent, as evidence entries.

    VD-219: the sources the fine-tune writes last are where the answer came
    from, so they belong in the evidence line - never in the answer. Kept only
    when the name is an appliance evidence source AND was one of this turn's
    reading lines, so a model cannot cite a reading it never saw.
    """
    from .answer_evidence import EVIDENCE_SOURCES

    cited: List[Dict[str, str]] = []
    for key, value, finished in contract_values(_local()._strip_code_fence(str(content or ""))):
        name = value.strip().split(":", 1)[0].strip().lower()
        if (finished and key.strip().lower() in CITATION_KEYS and name in EVIDENCE_SOURCES
                and "- {}:".format(name) in sent and name not in [item["source"] for item in cited]):
            cited.append({"source": name, "summary": CITED_READING})
    return cited
