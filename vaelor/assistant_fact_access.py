"""A gathered fact, or ``None`` when the tool behind it gave no usable reading.

**A failed tool used to read as a healthy, empty one (review B1).** The chat
route stores a tool that raised as ``{"unavailable": reason}``. That dict is
truthy, so it passed every ``and facts`` gate in the built-in answer path, and
the summaries then read it as a real reading with nothing in it: an inventory
probe that failed answered "this node manages 0 apps", a failed storage read
"Vaelor detected 0 storage devices", a failed network read "no network
interface reports itself as up", and ``services.status`` - a dict where a list
was expected - crashed the route with an HTTP 500 because iterating a dict
yields its keys. Each of those carried evidence, so each outranked a working
model (LESSONS 1 and 8: the observer's failure reported as the machine's
state).

One accessor decides, for every reader: a fact that is missing, that carries
``unavailable``, or that is not the shape its reader needs is not a reading.
:func:`unread` then says so in plain words; the tool's raw error stays in the
fact for the log and never reaches the sentence (LESSONS 24).
"""

from __future__ import annotations

from typing import Any, Mapping, Optional

from .answer_evidence import describe_missing

#: The key the chat route writes when a fact tool raised. Read here only, so
#: the shape of a failure is decided in one place.
UNAVAILABLE_KEY = "unavailable"


def usable_fact(facts: Any, key: str, kind: Any = dict) -> Optional[Any]:
    """The fact ``key`` when it is a usable reading of ``kind``, else ``None``.

    ``kind`` is the shape the caller is about to read: a ``dict`` for most
    tools, a ``list`` for ``services.status`` and ``jobs.recent``. A list whose
    entries are not mappings is not a usable list of records either - that is
    the same crash one level down.
    """
    if not isinstance(facts, Mapping):
        return None
    value = facts.get(key)
    if value is None:
        return None
    if isinstance(value, Mapping) and UNAVAILABLE_KEY in value:
        return None
    if not isinstance(value, kind):
        return None
    if isinstance(value, list) and any(not isinstance(item, Mapping) for item in value):
        return None
    return value


def gathered_but_unread(facts: Any, key: str, kind: Any = dict) -> bool:
    """Whether ``key`` was gathered for this question and gave no usable reading.

    Distinct from "not gathered": a question that never asked for storage must
    not be told storage could not be read.
    """
    return (
        isinstance(facts, Mapping) and key in facts
        and usable_fact(facts, key, kind) is None
    )


def unread(subject: str) -> str:
    """The sentence for a subject whose reading failed this time.

    No raw error and no guess: the reading was attempted and did not arrive,
    which says nothing about the machine itself.
    """
    return describe_missing(
        subject,
        "The reading was attempted for this question and did not arrive, so "
        "this says nothing either way about the machine itself.",
    )
