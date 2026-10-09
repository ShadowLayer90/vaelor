"""Which machine a question is about (VD-205 item 2).

A cluster question names a machine the way the owner names it on the Fleet
screen - "Worker A", "the ZBook", "the controller", "the worker" - and the
answer has to read that machine's own readings and history. Answering a
question about a worker from the controller's readings is the misattribution
the VD-205 baseline found: ``metrics.history`` read the controller whatever
machine the question named.

The rule: a name or role that picks out exactly one machine resolves to it; a
role that fits several ("the worker" with two workers) or a name that matches
none is not guessed - the answer asks which machine was meant.
"""

from __future__ import annotations

import re
from typing import Any, Dict, List, Mapping, NamedTuple, Optional

from .cluster_placement import CONTROLLER_PLACEMENT_ID, CONTROLLER_PLACEMENT_NAME
from .assistant_console_places import CONSOLE_PHRASES, PAGE_NAMES
from .assistant_vocabulary import MACHINE_SET_PHRASES
from .phrase_match import mentions

#: Words that name the controller by its role.
CONTROLLER_WORDS = ("controller", "head node", "head controller")
#: Words that name a worker by its role, without saying which.
WORKER_WORDS = ("worker", "workers", "the worker", "a worker")
#: Words that name every machine at once.
EVERY_MACHINE_WORDS = MACHINE_SET_PHRASES + ("all the machines", "both machines")

#: "on the zbook", "the zbook's" - a definite reference to a machine by a name,
#: used to notice a name that matches no enrolled machine.
#: "On the Z9", "was the Z9's": a determiner before a capitalised word is how
#: an owner names a machine. Two characters are enough ("the Z9", review round
#: 2 S4), and the leading word may open the sentence ("Was the Z9 hot").
#: **The determiner is required (review round 3, N-B1)**: without it "Did
#: Docker crash", "Was SSH used" and "on Ubuntu" read as unknown machines.
_NAMED_REFERENCE = re.compile(
    r"\b(?i:on|from|of|for|is|was|did)\s+the\s+([A-Za-z][\w-]{1,})(?:'s)?\b")


#: A bare capitalised name after a preposition, "on Z9" (review round 4,
#: R3-B1). Read only by the history path, where a missing name must not fall
#: back to this controller's rows; the preposition must be lowercase, so the
#: name cannot be a sentence's first word.
_BARE_REFERENCE = re.compile(r"\b(?:on|for|of|from)\s+([A-Z][\w-]{1,})\b")
#: A capitalised possessive, "Z9's", "Zeus's" (review round 4, R3-B1).
_POSSESSIVE_NAME = re.compile(r"\b([A-Z][\w-]{1,})'s\b")


class Machine(NamedTuple):
    """One machine in the cluster, as the Fleet screen names it."""

    node: str
    name: str
    role: str


class Resolution(NamedTuple):
    """What a question names: the machines, or why it could not be told."""

    machines: List[Machine]
    every: bool = False
    ambiguous: str = ""


def machine_directory(callbacks: Mapping[str, Any]) -> List[Machine]:
    """The controller and every enrolled worker, by the names the Fleet screen uses."""
    from .performance_snapshot_source import worker_nodes

    machines = [Machine(CONTROLLER_PLACEMENT_ID, CONTROLLER_PLACEMENT_NAME, "controller")]
    for worker in worker_nodes(dict(callbacks)):
        machines.append(Machine(str(worker["id"]), str(worker.get("name") or ""), "worker"))
    return machines


def _name_tokens(name: str) -> List[str]:
    return [token for token in re.split(r"[\s_-]+", name.lower()) if len(token) >= 3]


#: Words a machine name may carry that are also how an owner says something
#: else: a role ("controller", "worker"), a part of any machine ("gpu", "box"),
#: or a word an ordinary question uses ("main", "server"). A name made of one
#: of these, or a token of a name that is one, never resolves on the word
#: alone (adversarial review B-6: a worker called "GPU Box" took every GPU
#: question, "Main" took "is the main model loaded").
_ROLE_WORDS = frozenset({"controller", "worker", "workers", "head", "node", "nodes",
                         "machine", "machines", "appliance", "cluster", "fleet"})
_ORDINARY_WORDS = frozenset({
    "main", "box", "server", "host", "computer", "desktop", "laptop", "home", "office",
    "this", "that", "the", "my", "new", "old", "big", "small", "first", "second", "test",
    "gpu", "gpus", "cpu", "cpus", "npu", "model", "models", "pc", "station", "studio",
    "ultra", "pro", "mini", "max",
})


def _assistant_words() -> frozenset:
    """Every single word the Assistant's gather table hears, computed once."""
    global _VOCABULARY
    if _VOCABULARY is None:
        from .assistant_vocabulary import TOOL_PHRASES, literal_phrases

        _VOCABULARY = frozenset(
            word for tool in TOOL_PHRASES for phrase in literal_phrases(tool)
            for word in phrase.split())
    return _VOCABULARY


_VOCABULARY: Optional[frozenset] = None


def _ordinary(word: str) -> bool:
    lowered = word.lower()
    return lowered in _ROLE_WORDS or lowered in _ORDINARY_WORDS or lowered in _assistant_words()


#: Words that are never a machine name on their own, however an owner named
#: one: a worker called "the" matched every question (review B-6, round 2).
_STOPWORDS = frozenset({
    "the", "and", "for", "with", "from", "into", "what", "which", "how", "why", "when",
    "where", "who", "does", "did", "was", "are", "is", "it", "its", "this", "that", "these",
    "those", "my", "our", "your", "any", "all", "not", "but", "has", "have", "had", "can",
    "will", "use", "using", "on", "in", "at", "to", "of", "by", "or", "an", "a",
})
_CONSOLE_PHRASE = re.compile(
    r"\b(?:" + "|".join(re.escape(phrase) for phrase in CONSOLE_PHRASES) + r")(?!\w)", re.IGNORECASE)
_PAGE_REFERENCE = re.compile(
    r"\b(?:" + "|".join(re.escape(page) for page in PAGE_NAMES)
    + r")\s*(?:page|screen|tab|view|card|>)", re.IGNORECASE)


def without_console_phrases(text: str) -> str:
    """``text`` with every console feature, place and page reference blanked.

    "Is AI Chat working?" is about AI Chat, never about a worker named "AI";
    "the LLM Server" is the feature, not a worker named "Server".
    """
    return _PAGE_REFERENCE.sub(" ", _CONSOLE_PHRASE.sub(" ", str(text or "")))


def _single_word_allowed(name: str) -> bool:
    """A one-word name stands alone only when it is 3+ letters and not a stopword."""
    return len(name) >= 3 and name.lower() not in _STOPWORDS


def _names_it(text: str, lower: str, name: str) -> bool:
    """Whether the question names a machine by its whole name.

    A one-word name must be written exactly as the owner wrote it,
    capitalised as the name is: "how much power is the GPU drawing" is never
    about a worker named "Power" (adversarial review B-6, round 3). An
    ordinary word ("Main", "Controller") must also not follow "the": "the
    controller" is the controller's role, never a worker named "Controller".
    """
    tokens = name.lower().split()
    if len(tokens) == 1:
        if not _single_word_allowed(name):
            return False
        text = without_console_phrases(text)
        lower = text.lower()
    if not mentions(lower, (name.lower(),)):
        return False
    if len(tokens) > 1:
        return True
    if not _ordinary(tokens[0]):
        return re.search(r"\b" + re.escape(name) + r"\b", text) is not None
    return re.search(r"(?<!\bthe )(?<!\bThe )\b" + re.escape(name) + r"\b", text) is not None


def resolve_machines(message: str, machines: List[Machine]) -> Resolution:
    """The machines ``message`` names, or a reason to ask which one was meant.

    Whole names first, and a shorter name contained in a longer matched one is
    dropped ("Worker 2" is not also "Worker"). A single token of a name stands
    for it only when it is distinctive - not a role word, not an ordinary or
    Assistant vocabulary word - and when several workers share it the answer
    asks which was meant (adversarial review B-6).
    """
    text = str(message or "")
    lower = text.lower()
    workers = [machine for machine in machines
               if machine.role == "worker" and machine.name.strip() and machine.name != machine.node]
    whole = [machine for machine in workers if _names_it(text, lower, machine.name.strip())]
    whole = [machine for machine in whole if not any(
        other is not machine and machine.name.lower() in other.name.lower()
        and len(other.name) > len(machine.name) for other in whole)]
    found: List[Machine] = list(whole)
    if not found:
        by_token: Dict[str, List[Machine]] = {}
        bare = without_console_phrases(text).lower()
        # A one-word name was already looked for, case-sensitively, above;
        # only a name of several words is found by one distinctive word of it.
        for machine in workers:
            if len(machine.name.split()) < 2:
                continue
            for token in _name_tokens(machine.name):
                if _single_word_allowed(token) and not _ordinary(token) and mentions(bare, (token,)):
                    by_token.setdefault(token, []).append(machine)
        for token, holders in by_token.items():
            if len(holders) > 1:
                return Resolution([], ambiguous=which_machine(holders))
            found.extend(holders)
    if mentions(lower, EVERY_MACHINE_WORDS):
        return Resolution(list(machines), every=True)
    if mentions(lower, CONTROLLER_WORDS) and not any(
            "controller" in _name_tokens(machine.name) for machine in whole):
        found.insert(0, machines[0])
    if not found and mentions(lower, WORKER_WORDS):
        if len(workers) == 1:
            found.append(workers[0])
        elif len(workers) > 1 and not mentions(lower, ("workers",)):
            return Resolution([], ambiguous=which_machine(workers))
        elif workers:
            return Resolution(list(workers), every=True)
    return Resolution(list(dict.fromkeys(found)))


def which_machine(candidates: List[Machine]) -> str:
    """The question back to the owner when a name or role fits several machines."""
    names = [machine.name for machine in candidates if machine.name]
    return "Which machine do you mean - {}?".format(
        " or ".join(names) if len(names) <= 2 else ", ".join(names[:-1]) + " or " + names[-1])


def no_machine_named(name: str, machines: List[Machine]) -> str:
    """The answer to a question naming a machine this cluster does not have."""
    return "No machine in this cluster is named {}. {}".format(name, which_machine(machines))


def inventory_names(facts: Mapping[str, Any], brief_text: str = "") -> frozenset:
    """Every app, model and managed-service name this turn read, lowercased.

    ``brief_text`` adds the words of the standing brief's operating-system
    line, so "on Ubuntu" is never a missing machine (review round 4).

    Review round 3 (N-B1, LESSONS 6): "the Jellyfin" is an app, never an
    unknown machine. The names come from this turn's own inventory readings
    (``workloads.inventory`` and ``services.status``), not a list kept here.
    """
    found = set()
    workloads = facts.get("workloads.inventory") if isinstance(facts.get("workloads.inventory"), Mapping) else {}
    for key in ("apps", "models"):
        for item in workloads.get(key) or []:
            if isinstance(item, Mapping):
                found.update(str(item.get(field) or "").lower() for field in ("name", "id", "project"))
    services = facts.get("services.status") if isinstance(facts.get("services.status"), list) else []
    found.update(str(item.get("id") or "").lower() for item in services if isinstance(item, Mapping))
    for line in str(brief_text or "").splitlines():
        if line.startswith("Operating system:"):
            found.update(word.strip(".,()").lower() for word in line.split(":", 1)[1].split())
    return frozenset(found - {""})


def unknown_machine(message: str, machines: List[Machine], not_machines: frozenset = frozenset()) -> str:
    """A machine-shaped name in the question that matches no machine, or ``""``.

    Only consulted for a question that already reads as about the cluster, so
    "on the zbook" with no machine of that name is asked about rather than
    answered from whichever machine happens to be in hand. ``not_machines``
    is :func:`inventory_names`: an app or service is never a missing machine.
    """
    known = {token for machine in machines for token in _name_tokens(machine.name)}
    known.update(not_machines)
    known.update(machine.name.lower() for machine in machines)
    known.update(word.split()[-1] for word in CONTROLLER_WORDS + WORKER_WORDS + EVERY_MACHINE_WORDS)
    for match in _NAMED_REFERENCE.finditer(str(message or "")):
        word = match.group(1)
        lowered = word.lower()
        if (word[:1].isupper() and lowered not in known and lowered not in _COMMON_WORDS
                and not any(name.startswith(lowered) for name in known)):
            return word
    return ""


#: Capitalised words that open ordinary clauses rather than naming a machine.
_COMMON_WORDS = frozenset({"gpu", "cpu", "npu", "the", "this", "that", "it", "model", "cluster",
                           "vaelor", "fleet", "ai", "chat", "assistant", "llm", "os", "ip", "ram"})


#: Calendar words that open a time, never a machine: "on Monday", "of March".
_CALENDAR_WORDS = frozenset({
    "monday", "tuesday", "wednesday", "thursday", "friday", "saturday", "sunday", "january", "february",
    "march", "april", "may", "june", "july", "august", "september", "october", "november", "december",
})


def unknown_bare_name(message: str, machines: List[Machine], not_machines: frozenset = frozenset()) -> str:
    """A bare or possessive capitalised name that matches no machine, or ``""`` (review round 4).

    Only the history path reads this: "What happened on Z9 last night?" was
    answered from this controller's rows. Known machines, this turn's app and
    service names, calendar words and console page names are never missing.
    """
    known = {token for machine in machines for token in _name_tokens(machine.name)}
    known.update(machine.name.lower() for machine in machines)
    known.update(not_machines, _COMMON_WORDS, _CALENDAR_WORDS, (page.lower() for page in PAGE_NAMES))
    text = str(message or "")
    for word in [match.group(1) for pattern in (_BARE_REFERENCE, _POSSESSIVE_NAME)
                 for match in pattern.finditer(text)]:
        lowered = word.lower()
        if lowered not in known and not any(name.startswith(lowered) for name in known):
            return word
    return ""


def as_records(machines: List[Machine]) -> List[Dict[str, str]]:
    """The machines as plain records, for a fact or a test."""
    return [machine._asdict() for machine in machines]


def single(resolution: Resolution) -> Optional[Machine]:
    """The one machine a resolution names, or ``None``."""
    return resolution.machines[0] if len(resolution.machines) == 1 and not resolution.every else None


def machines_named(message: str, machines: List[Machine]) -> bool:
    """Whether the question names a worker by its name (route gather gate)."""
    resolution = resolve_machines(message, machines)
    return any(machine.role == "worker" for machine in resolution.machines) or bool(resolution.ambiguous)


def node_for_name(callbacks: Mapping[str, Any], node: Any) -> Optional[str]:
    """The node a machine name (or id) names: ``None`` for the controller, ``""`` for none."""
    if not isinstance(node, str) or not node.strip():
        return None
    wanted = node.strip().lower()
    for machine in machine_directory(callbacks):
        if wanted in (machine.name.lower(), machine.node.lower()):
            return None if machine.role == "controller" else machine.node
    return ""
