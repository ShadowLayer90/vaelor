"""Named past windows, resolved on this machine's local clock (review S1).

"Last night", "yesterday" and "overnight" used to mean a rolling 48 hours and
"this morning" a rolling 24, with no clock or time zone behind either, and the
answer never said which window it read. Asked "What was the highest CPU
temperature yesterday?" with a 75 °C spike in hand, the reply led with "CPU
temperature held steady, around 45 °C". And a clock time or a weekday was not
treated as the past at all, so "What was the CPU temperature at 3am?" got the
current reading, recorded as answered (#159's original defect).

Each named window resolves here to a local start and end, so the history read
has an end bound and the answer can say the window in words. The clock is this
machine's: the appliance's own local time is the one its owner lives in.
"""

from __future__ import annotations

import re
import time
from datetime import datetime, timedelta
from typing import Any, NamedTuple, Optional, Tuple

#: The local hours the day-part words cover.
NIGHT_STARTS = 20
MORNING_STARTS = 6
AFTERNOON_STARTS = 12
EVENING_STARTS = 18
#: Half the window read around a named clock time.
AROUND_A_TIME = timedelta(minutes=30)
#: How far back "before the reboot" reads.
BEFORE_REBOOT = timedelta(hours=1)

_WEEKDAYS = ("monday", "tuesday", "wednesday", "thursday", "friday", "saturday", "sunday")

#: A past-tense verb, which a clock time or weekday needs beside it to be about
#: the past: "is 3am a good time to reboot" is a question about a plan.
_PAST_VERB = re.compile(r"\b(?:was|were|did|had|happened|got|went|rose|fell|peaked|spiked|ran)\b")
_CLOCK_TIME = re.compile(r"\b(?:at|around|about)\s+(\d{1,2})(?::(\d{2}))?\s*(am|pm|a\.m\.|p\.m\.)?(?![\w:])")
#: "Between 11pm and 2am": a span of two clock times (review round 2, S4). It
#: was answered with the named machine's current line.
_CLOCK_SPAN = re.compile(
    r"\bbetween\s+(\d{1,2})(?::(\d{2}))?\s*(am|pm)?\s+and\s+(\d{1,2})(?::(\d{2}))?\s*(am|pm)?(?![\w:])")
_WEEKDAY = re.compile(r"\bon\s+(" + "|".join(_WEEKDAYS) + r")\b")
_BEFORE_REBOOT = re.compile(r"\bbefore\s+(?:the|its|my|this\s+machine's|the\s+last)\s+(?:reboot|restart|boot)\b")
_NAMED = (
    (re.compile(r"\blast night\b|\bovernight\b"), "night"),
    (re.compile(r"\byesterday\b"), "yesterday"),
    (re.compile(r"\bthis morning\b"), "morning"),
    (re.compile(r"\bthis afternoon\b"), "afternoon"),
    (re.compile(r"\bthis evening\b|\btonight\b"), "evening"),
    (re.compile(r"\btoday\b|\bso far today\b"), "today"),
)
#: A question about a peak, which the answer leads with.
PEAK_QUESTION = re.compile(
    r"\b(?:highest|peak|peaked|maximum|max|hottest|warmest|lowest|coolest|minimum|"
    r"how hot did|how high did|how warm did)\b")


def asks_for_peak(message: str) -> bool:
    """Whether the question asks for a peak, which the answer then leads with."""
    return PEAK_QUESTION.search(str(message or "").lower()) is not None


class Window(NamedTuple):
    """A resolved window: epoch seconds, and how the answer names it."""

    start: float
    end: float
    words: str


def _local(now: Optional[float]) -> datetime:
    return datetime.fromtimestamp(time.time() if now is None else float(now)).astimezone()


def _clock(moment: datetime) -> str:
    return moment.strftime("%H:%M")


def _hour(text: str, half: Optional[str]) -> int:
    hour = int(text)
    if half == "pm" and hour < 12:
        return hour + 12
    return 0 if half == "am" and hour == 12 else hour


def _clock_span(lower: str) -> Optional["re.Match[str]"]:
    """A span of two clock times; "between 10 and 20 GB" is two numbers, not times."""
    found = _CLOCK_SPAN.search(lower)
    return found if found and any(found.group(index) for index in (2, 3, 5, 6)) else None


def _span_window(found: "re.Match[str]", local: datetime) -> Optional[Window]:
    """The most recent past stretch between two clock times, crossing midnight if it must."""
    first, last = _hour(found.group(1), found.group(3)), _hour(found.group(4), found.group(6))
    minutes = (int(found.group(2) or 0), int(found.group(5) or 0))
    if first > 23 or last > 23 or max(minutes) > 59:
        return None
    end = local.replace(hour=last, minute=minutes[1], second=0, microsecond=0)
    if end > local:
        end -= timedelta(days=1)
    start = end.replace(hour=first, minute=minutes[0])
    if start >= end:
        start -= timedelta(days=1)
    return Window(start.timestamp(), end.timestamp(), "between {} on {} and {} on {}".format(
        _clock(start), start.strftime("%Y-%m-%d"), _clock(end), end.strftime("%Y-%m-%d")))


def _day_part(now: datetime, first: int, last: Optional[int]) -> Optional[Window]:
    start = now.replace(hour=first, minute=0, second=0, microsecond=0)
    if start > now:
        return None
    end = now if last is None else min(now, now.replace(hour=last, minute=0, second=0, microsecond=0))
    return Window(start.timestamp(), end.timestamp(), "between {} and {} today".format(_clock(start), _clock(end)))


def named_window(message: str, now: Optional[float] = None, boot_time: Any = None) -> Optional[Window]:
    """The local window ``message`` names, or ``None`` when it names none."""
    lower = " ".join(str(message or "").lower().split())
    local = _local(now)
    midnight = local.replace(hour=0, minute=0, second=0, microsecond=0)
    for pattern, kind in _NAMED:
        if not pattern.search(lower):
            continue
        if kind == "night":
            start = (midnight - timedelta(days=1)).replace(hour=NIGHT_STARTS)
            end = min(local, midnight.replace(hour=MORNING_STARTS))
            return Window(start.timestamp(), end.timestamp(),
                          "between {} yesterday and {} today".format(_clock(start), _clock(end)))
        if kind == "yesterday":
            start = midnight - timedelta(days=1)
            return Window(start.timestamp(), midnight.timestamp(), "yesterday, from 00:00 to 24:00")
        bounds = {"morning": (MORNING_STARTS, AFTERNOON_STARTS), "afternoon": (AFTERNOON_STARTS, EVENING_STARTS),
                  "evening": (EVENING_STARTS, None), "today": (0, None)}[kind]
        return _day_part(local, *bounds)
    if _PAST_VERB.search(lower):
        found = _BEFORE_REBOOT.search(lower)
        if found and isinstance(boot_time, (int, float)) and not isinstance(boot_time, bool) and boot_time > 0:
            boot = datetime.fromtimestamp(float(boot_time)).astimezone()
            return Window((boot - BEFORE_REBOOT).timestamp(), boot.timestamp(),
                          "in the hour before the last start, at {}".format(boot.strftime("%Y-%m-%d %H:%M")))
        found = _clock_span(lower)
        if found:
            return _span_window(found, local)
        found = _CLOCK_TIME.search(lower)
        if found:
            hour, minute, half = int(found.group(1)), int(found.group(2) or 0), (found.group(3) or "")
            if half.startswith("p") and hour < 12:
                hour += 12
            if half.startswith("a") and hour == 12:
                hour = 0
            if hour < 24 and minute < 60:
                at = local.replace(hour=hour, minute=minute, second=0, microsecond=0)
                if at > local:
                    at -= timedelta(days=1)
                return Window((at - AROUND_A_TIME).timestamp(), min(local, at + AROUND_A_TIME).timestamp(),
                              "around {} on {}".format(_clock(at), at.strftime("%Y-%m-%d")))
        found = _WEEKDAY.search(lower)
        if found:
            back = (local.weekday() - _WEEKDAYS.index(found.group(1))) % 7 or 7
            start = midnight - timedelta(days=back)
            return Window(start.timestamp(), (start + timedelta(days=1)).timestamp(),
                          "on {} {}".format(found.group(1).capitalize(), start.strftime("%Y-%m-%d")))
    return None


def asks_about_a_named_past(message: str, boot_time: Any = None) -> bool:
    """Whether a past-tense question names a clock time, a weekday or "before the reboot"."""
    lower = " ".join(str(message or "").lower().split())
    if not _PAST_VERB.search(lower):
        return False
    return bool(_clock_span(lower) or _CLOCK_TIME.search(lower) or _WEEKDAY.search(lower)
                or _BEFORE_REBOOT.search(lower))


#: How a day part is named when it has not begun yet.
_DAY_PART_NAMES = {"morning": "This morning", "afternoon": "This afternoon", "evening": "This evening"}


#: Whose start time a refusal names when the caller names no machine.
MACHINE_OWNER = "The appliance"


def window_or_reason(
    message: str, now: Optional[float] = None, boot_time: Any = None, boot_owner: str = MACHINE_OWNER,
) -> Tuple[Optional[Window], str]:
    """The named window, or why a window the question names has no readings.

    **Adversarial review B-3.** "This morning" asked at 05:00, "this afternoon"
    before noon, "this evening" before 18:00 and "before the reboot" with no
    start time read each resolved to *no window*, and the answer then fell
    back to the most recent rows and answered the past with the present. A
    window that is named and cannot be read is refused with its reason; only a
    question that names no window at all keeps the caller's recent read.
    """
    window = named_window(message, now, boot_time)
    if window is not None:
        return window, ""
    lower = " ".join(str(message or "").lower().split())
    local = _local(now)
    for pattern, kind in _NAMED:
        if pattern.search(lower) and kind in _DAY_PART_NAMES:
            return None, (
                "{} has not started yet on this machine's clock (it is {}), so there "
                "are no readings from it to report.".format(_DAY_PART_NAMES[kind], _clock(local)))
    if _PAST_VERB.search(lower) and _BEFORE_REBOOT.search(lower):
        return None, (
            "{}'s last start time was not read, so the hour before it cannot be "
            "found and I will not answer it from other readings.".format(boot_owner))
    return None, ""
