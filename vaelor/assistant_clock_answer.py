"""The time, the date and the time zone, read from this machine's own clock (VD-220).

Owner ruling (VD-220, 2026-10-18): "what time is it" and "what's the date" are
the Assistant's, answered from the box's own clock - time, date and time zone -
as a built-in reading. They were sent to AI Chat by the bare question opener,
and AI Chat cannot read this machine's clock either.

**Only the clock of this machine.** A question that names somewhere else ("what
time is it in Tokyo") is world knowledge and is not matched here; it routes as
it did before. The detector is a whole-sentence shape, so a longer sentence
that merely contains "what time" is not claimed.

Runs in the control plane, on the appliance, so ``datetime.now()`` is the box's
clock and its local zone is the box's zone.
"""

from __future__ import annotations

import os
import re
from datetime import datetime
from typing import Any, Callable, Dict, Optional

CLOCK_SOURCE = "built-in-clock"

_OPENER = r"^\s*(?:so|ok|okay|hey|hi|hello|please)?[\s,]*"
_NOW = r"(?:\s+(?:is\s+it|today|now|right\s+now|here))?"
_MACHINE = r"(?:box|machine|appliance|server|node|device|unit)"

#: "what time is it", "what's the date", "what day is it today", "what's today's date".
_TIME_OR_DATE = re.compile(
    _OPENER + r"(?:"
    r"what(?:'s|\s+is)\s+(?:the\s+)?(?:time|date|day)" + _NOW +
    r"|what\s+(?:time|day|date)\s+is\s+it" + r"(?:\s+(?:now|today|right\s+now|here))?"
    r"|what(?:'s|\s+is)\s+today'?s\s+date"
    r")\s*[?.!]*\s*$", re.IGNORECASE)

#: "what time zone is this box in", "what's the timezone here", "which time zone am I in".
_TIME_ZONE = re.compile(
    _OPENER + r"(?:"
    r"(?:what|which)(?:'s|\s+is)?\s+(?:the\s+|this\s+" + _MACHINE + r"'?s\s+)?time\s*zone"
    r"(?:\s+(?:here|on\s+(?:this|the)\s+" + _MACHINE + r"|set))?"
    r"|(?:what|which)\s+time\s*zone\s+(?:is|am|are)\s+(?:(?:this|the|my)\s+" + _MACHINE + r"|i|we)"
    r"(?:\s+(?:in|on|set\s+to|using))?"
    r")\s*[?.!]*\s*$", re.IGNORECASE)


def asks_the_clock(message: Any) -> bool:
    """Whether ``message`` asks this machine's time, date, day or time zone."""
    text = str(message or "")
    return bool(_TIME_OR_DATE.match(text) or _TIME_ZONE.match(text))


def _zone_name() -> str:
    """The IANA zone the box is set to ("Europe/Zagreb"), or ``""`` when it does not say."""
    try:
        with open("/etc/timezone", encoding="utf-8") as handle:
            name = handle.read().strip()
        if name:
            return name
    except OSError:  # absence-ok: not every system writes /etc/timezone; /etc/localtime is read next
        pass
    target = os.path.realpath("/etc/localtime")
    return target.split("zoneinfo/", 1)[1] if "zoneinfo/" in target else ""


def clock_answer(message: Any, now: Optional[Callable[[], datetime]] = None,
                 zone: Optional[Callable[[], str]] = None) -> Optional[Dict[str, Any]]:
    """The built-in answer from this machine's clock, or ``None`` when ``message`` does not ask it."""
    if not asks_the_clock(message):
        return None
    moment = (now or datetime.now)()
    if moment.tzinfo is None:
        moment = moment.astimezone()  # the box's local zone
    offset = moment.strftime("%z")
    utc = "UTC{}{}:{}".format(offset[:1], offset[1:3], offset[3:]) if offset else "UTC"
    named = (zone or _zone_name)()
    where = "{} ({}, {})".format(named, moment.tzname(), utc) if named else "{} ({})".format(moment.tzname(), utc)
    reading = "{} on {}".format(moment.strftime("%H:%M"), moment.strftime("%A %d %B %Y").replace(" 0", " "))
    if _TIME_ZONE.match(str(message or "")):
        answer = "This machine's clock is set to the {} time zone. It reads {}.".format(where, reading)
    else:
        answer = "By this machine's clock it is {}, time zone {}.".format(reading, where)
    return {
        "answer": answer,
        "evidence": [{"source": "assistant.clock", "summary": "Read from this machine's own clock and time zone setting."}],
        "suggested_actions": [],
        "proposed_job": None,
    }
