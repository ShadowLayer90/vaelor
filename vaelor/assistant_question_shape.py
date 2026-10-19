"""The shape of an owner's sentence, read for what to gather (VD-219).

Each predicate here decides which readings a question is shown - never whether
the question is this appliance's business, except
:func:`asks_about_this_processor`, which only *keeps* a question here (the
Assistant's own redirect) and is never read by AI Chat. They were in
`assistant_intents`, which they would have taken past the 1,000-line ceiling.
"""

from __future__ import annotations

import re
from typing import Any, Dict

from .assistant_vocabulary import REGEX_PREFIX, TOOL_PHRASES

__all__ = [
    "asks_about_this_processor", "asks_for_a_past_window", "asks_for_overview", "asks_to_compose",
    "history_arguments", "names_an_app",
]


#: VD-219 (LESSONS 11): an overview asked in other words. "Give me a summary of
#: this machine's state", "should I be worried about anything on this machine",
#: "what should I watch on this machine" and "explain every reading you have"
#: named no sweep phrase, so the model was shown the identity reading alone and
#: invented the rest (6 of 44 graded asks). Read for **gathering only**, never
#: for scope (`matched_intents`): "summary" and "watch" are ordinary English, so
#: a state word counts only beside this machine named as a machine, and the
#: reading forms count only where they say "readings".
_READING_SWEEP = re.compile(r"\b(?:every|each|all)\b(?:\W+\w+){0,2}?\W+readings?\b", re.IGNORECASE)
_STATE_WORDS = re.compile(
    r"\b(?:summary|summari[sz]e|rundown|state|worried|worry|concern|concerned|watch|monitor)\b",
    re.IGNORECASE,
)
_THIS_BOX = re.compile(
    # "this machine learning paper" names no machine (kb R159).
    r"\b(?:this|my|the)\s+(?:machine|appliance|box|computer|pc|node|server)\b(?!\s+learning\b)",
    re.IGNORECASE,
)
#: Asking for something to be written is a composition request, never an overview.
_COMPOSE = re.compile(r"\b(?:write|draft|compose|translate|rewrite)\b", re.IGNORECASE)


#: A name written mid-sentence with a capital: "Can I run Plex here?",
#: "Would Paperless-ngx run on this box?", "Can this run Pi-hole?". A word
#: after a sentence or clause mark is that clause's first word, not a name
#: ("Hello Vaelor, How is the GPU?", kb R154).
_PROPER_NAME = re.compile(r"(?<![.!?,;:]\s)(?<![.!?,;:])(?<!^)\b([A-Z][\w]*(?:-[\w]+)*)\b")

#: This product's own names and the hardware it names, which are not apps.
_OWN_NAMES = frozenset((
    "vaelor", "pironman", "ubuntu", "linux", "amd", "hp", "intel", "radeon", "ryzen", "strix",
    "halo", "xdna", "assistant", "chat", "server", "zbook", "raspberry", "pi", "mini", "max", "pro",
))


def names_an_app(message: str) -> bool:
    """Whether ``message`` names something by a proper name that is not this product's own (VD-219).

    App questions ("Can I run Plex here?", "Is Heimdall running?") gathered no
    app inventory, so the answer could not say whether the app is already
    installed or which ports are taken - 450 of 2,640 routed training prompts.
    Read from the sentence's shape - a capitalised word that is not its first
    or a clause's first, not one of the readings' own words and not this
    product's name - so an app nobody listed is still an app. A hyphenated
    name is read whole ("Pi-hole" is not "Pi", kb R150). Gather-only.
    """
    from .assistant_intents import _CONTROL_PLANE_WORDS

    own = _OWN_NAMES | _CONTROL_PLANE_WORDS | {
        word for phrases in TOOL_PHRASES.values() for phrase in phrases
        if not phrase.startswith(REGEX_PREFIX) for word in phrase.split()}
    text = " ".join(str(message or "").split())
    if not _RUNS_SOFTWARE.search(text):
        return False
    return any(found.group(1).lower() not in own and not found.group(1).isupper()
               for found in _PROPER_NAME.finditer(text))


#: The sentence asks about running software, not about a person or a place:
#: "Should I buy Bitcoin?" and "I love Taylor Swift" name no app to look up
#: (kb R155: they must not gather readings).
_RUNS_SOFTWARE = re.compile(
    r"\b(?:run|runs|running|host|install|installed|deploy|deployed|support|supported|work|works|serve|"
    r"up|crashed|port|app)\b", re.IGNORECASE)


def history_arguments(message: str) -> Dict[str, Any]:
    """The `metrics.history` arguments for ``message``: its past window, or none (the newest rows)."""
    from .assistant_answer_scope import window_for_question

    return {"window": window_for_question(message)} if asks_for_a_past_window(message) else {}


def asks_for_a_past_window(message: str) -> bool:
    """Whether ``message`` asks about a span of time this appliance retains (VD-219, kb R154).

    The history answer's own window parser decides the span; it must lie in the
    past ("in the last 15 minutes", not "in the next hour") and inside the
    retained week ("will the GPU last 5 years" names no window to read).
    """
    from .assistant_answer_scope import asks_about_a_past_time, window_for_question
    from .assistant_history_tools import _parse_window
    from .telemetry_store import MAX_HISTORY_WINDOW_SECONDS

    window = window_for_question(message)
    seconds = _parse_window(window) if window else None
    if not seconds or seconds > MAX_HISTORY_WINDOW_SECONDS:
        return False
    from .assistant_intents import matched_intents

    return asks_about_a_past_time(message) or "metrics.history" in matched_intents(message)


def asks_to_compose(message: str) -> bool:
    """Whether ``message`` asks for something to be written - a composition request, never an overview (kb R159)."""
    return bool(_COMPOSE.search(str(message or "")))


def asks_for_overview(message: str) -> bool:
    """Whether ``message`` asks about the whole machine without a sweep phrase (VD-219)."""
    text = str(message or "")
    return bool(_READING_SWEEP.search(text)) or bool(
        _STATE_WORDS.search(text) and _THIS_BOX.search(text) and not _COMPOSE.search(text))


#: VD-219 / kb R144, R153: the processor asked about by its everyday name.
#: "How hot is the processor?" went to AI Chat - "cpu" marks a question as this
#: machine's and "processor" and "chip" do not, because they mean any device's.
#: Read only here, for the Assistant's own redirect, never in
#: `is_appliance_question` (which AI Chat reads to decline - kb R153): a bare
#: "the processor" or "the chip", beside a word for one of its readings, with
#: no other device named ("the chip in the iPhone 15", "the processor in my
#: laptop") and no knowledge request ("explain how the processor schedules").
_THE_PROCESSOR = re.compile(
    r"\b(?:the|my|this|your)\s+(?:graphics\s+)?(?:processor|chip|cpu)s?\b", re.IGNORECASE)
#: VD-220 R218: "the processor's clock speed", "how loaded is the processor" and
#: "is the processor sitting idle" were sent to AI Chat: this list knew
#: temperature and load and no other word for a reading.
_PROCESSOR_READING = re.compile(
    r"\b(?:hot|hotter|warm|temperature|temp|thermal|busy|load|loaded|usage|speed|clock|capacity|idle|"
    r"occupied|working|doing|state|status|stressed|up\s+to)\b", re.IGNORECASE)
#: VD-220 R218, R223: the processor's state right now, read even inside a
#: knowledge request ("explain in detail how the graphics processor is doing",
#: graded ask 19; "summarize the graphics chip's state"). A word alone is a
#: lesson there ("explain the processor state during an interrupt", kb R153).
_PROCESSOR_NOW = re.compile(
    r"\b(?:is|are)\s+(?:\w+\s+){0,2}?(?:doing|putting|working)\b|'s\s+(?:state|status|load)\b"
    r"|\b(?:right\s+now|currently|at\s+the\s+moment)\b", re.IGNORECASE)
#: Another device named after the processor ("the chip in the iPhone 15"); a
#: determiner may stand between, and this machine named that way is not another.
_ANOTHER_DEVICE = re.compile(
    r"\b(?:processor|chip|cpu)s?\s+(?:in|of|on|inside|from)\s+"
    r"(?!(?:\w+\s+)?(?:here|machine|box|appliance)\b)\w+", re.IGNORECASE)
#: VD-220 R223, R225: a situation in general, read only inside a knowledge request
#: ("explain the processor state during an interrupt"). Never a veto on a reading:
#: "how hot is the processor during a game" is this machine's (VD-042).
_IN_GENERAL = re.compile(r"\b(?:during|whenever|when\s+it\s+is)\b", re.IGNORECASE)


#: VD-220 R207: the processor placed in this machine by name - "what processor
#: is in the box", "which chip is inside this machine".
_IN_THIS_MACHINE = re.compile(
    r"\b(?:processor|chip|cpu)s?\s+(?:\w+\s+){0,2}(?:in|inside|on)\s+(?:the|this|my)\s+"
    r"(?:appliance|machine|box|unit|node|device|server|host)\b", re.IGNORECASE)


def asks_about_this_processor(message: str) -> bool:
    """Whether ``message`` asks for a reading of this machine's processor by its everyday name.

    The whole sentence, or any one of its clauses (VD-220 R219, R222): in "how
    hard is the processor working, and can you write me a haiku" the second
    clause's request is not the first's; in "the processor, how busy is it" the
    comma splits the processor from its reading, and the whole sentence holds it.
    """
    from .assistant_intents import _COMPOUND_COORDINATOR

    text = str(message or "")
    return _asks_about_this_processor(text) or any(
        _asks_about_this_processor(part) for part in _COMPOUND_COORDINATOR.split(text) if part.strip())


def _asks_about_this_processor(text: str) -> bool:
    from .assistant_intents import _KNOWLEDGE_REQUEST

    if names_an_app(text):
        return False
    if _THE_PROCESSOR.search(text) and not _ANOTHER_DEVICE.search(text):
        if _KNOWLEDGE_REQUEST.search(text):
            return _PROCESSOR_NOW.search(text) is not None and not _IN_GENERAL.search(text)
        if _PROCESSOR_READING.search(text):
            return True
    return bool(_IN_THIS_MACHINE.search(text)) and not _KNOWLEDGE_REQUEST.search(text)
