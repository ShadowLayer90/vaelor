"""Read a direction out of the retained samples, and the words that ask for it.

**The appliance advertised this capability for a day without having it.**
Alpha 50 made telemetry retention real - a `vaelor` store, a 7-day policy, rows
at a one-second interval, samples reaching the Assistant (VD-095). Asked *"show
me the temperature trend over the last hour"* it answered, badged
``OUT OF SCOPE``:

    There are 30 retained samples, ordered oldest to newest. **Ask for a trend
    - whether something rose, fell, or held steady - and I can answer from
    those.**

Asked the question it had just invited - *"Did the CPU temperature rise, fall,
or hold steady?"* - it returned the current instant and no trend, and
``metrics.history`` was not fetched at all. Nothing in the tree computed a
trend; `grep` for "rose", "fell" and "held steady" across `vaelor/` found that
invitation and some prose. The offer and the capability were two separate
things, and only one of them existed.

**So this module holds both, and that is the whole design.** The direction
words an owner reads back are :data:`DIRECTIONS`, and the only way to obtain
one is :func:`trends`, which returns nothing when nothing can be compared.
There is no invitation string anywhere: a reply either states a direction it
computed or says what it actually holds. An offer cannot drift away from a
capability it is built out of.

**And the same words are the vocabulary.** :data:`TREND_PHRASES` is what an
owner says to ask which way a reading moved; :data:`TREND_PATTERN` is the same
tuple rendered as the one regular expression the gather table carries, built by
:func:`_alternation` from that tuple rather than typed a second time. So the
words that reach `assistant_answer_scope.past_time_answer` and the words that
make `assistant_intents.select_tools` fetch `metrics.history` are one object in
two renderings. #188 is one idea in two hand-maintained tables, three instances
deep (VD-091, VD-094); this is the shape that does not admit a fourth.

**Why one regex entry rather than forty literals, which is the part to read
before widening it.** `assistant_intents.matched_phrases` counts *distinct
vocabulary phrases* and `_rests_on_one_borrowed_word` treats two of them as a
sentence corroborating itself. Forty literals were measured on 2026-08-13
against a battery of world questions using these words in their ordinary
English sense: "when did the Roman Empire rise and fall" counts ``rise`` and
``fall`` as two independent witnesses and stops being handed to AI Chat, and so
do "which stock markets fell last week" and "how many people climbed Everest
last month" - 4 of 12 lost, which is #190 arriving for the second time by the
same door. As **one** entry the whole family counts once, a person hears it
once, and the same battery is unmoved.

**The guard for that is `test_a_world_question_borrowing_one_of_these_words
_still_redirects`, and this paragraph named the wrong one for a round.** It
cited `tests/test_assistant_scope_consequence` as "the pin that would have
reported it". It would not: review spelled this family out as literals and
re-ran that pin's sixteen questions, and **0 moved** - not one of them uses a
direction word, so it is structurally blind to this particular widening. It
remains the right pin for the vocabulary as a whole. A guard cited from memory
is `LESSONS` pattern 10 in a docstring.

**A direction is only ever stated about a reading the question asked for.**
:func:`trends` takes ``only`` and the caller passes the fields the sentence
named. Without it this module trended whatever it could: with five rising
``cpu_temperature`` rows in hand, *"how many jobs failed earlier today"*,
*"has the backup job been dropping tasks"* and *"did my package delivery
arrive yesterday"* were each answered "CPU temperature rose 4 °C", reported
``answered=True``, and returned to the owner - an off-subject refusal turned
into an off-subject assertion, which is worse than the invitation it replaced
and is #159's founding defect arriving from the other side.

**The span is stated because thirty samples at a one-second interval are thirty
seconds.** The question that exposed all this asked for "the last hour" and the
store held half a minute. An answer that adopts the window in the question is
the same defect as the invitation it replaces - a claim about this machine that
nothing measured - so :func:`describe_span` reports what the sample times
actually cover, says so when it cannot read them, and every reply says that is
all this appliance retains.
"""

from __future__ import annotations

import re
from datetime import datetime
from typing import Any, Dict, List, NamedTuple, Optional, Sequence, Tuple

from .answer_evidence import FIELD_NAMES
from .phrase_match import mentions

#: The clause that marks a reply as declining because the samples are about
#: something else.
#:
#: Read by `provider_runtime.BUILTIN_REFUSAL_MARKERS`, which decides whether a
#: built-in reply outranks the model - and a refusal must not. It is a
#: constant rather than a second spelling for the reason that list gives about
#: itself: two lists that must agree are one list.
#:
#: **All lower case, and that is load-bearing.**
#: `declines_for_lack_of_a_reading` lower-cases the answer text and compares
#: the markers raw, so a marker carrying a capital can never match. The first
#: version of this was "so I cannot answer it from them", the sentence's own
#: mid-clause capital `I` made it unmatchable, and the refusal went on
#: outranking the model with every test green.
SUBJECT_NOT_RETAINED = "samples do not record"

#: What this module is able to conclude, and therefore the only direction words
#: any reply can contain. They are read back to an owner verbatim, and
#: :data:`TREND_PHRASES` below hears every one of them - the appliance must be
#: able to take back the question it answers in its own words.
ROSE = "rose"
FELL = "fell"
HELD_STEADY = "held steady"
DIRECTIONS: Tuple[str, ...] = (ROSE, FELL, HELD_STEADY)


class Reading(NamedTuple):
    """One value in a retained sample that a direction can be read out of.

    ``steady_within`` is the movement below which this reading is reported as
    holding steady rather than as rising. It is per reading because the units
    are not comparable: half a degree of CPU temperature is noise, and half a
    percent of memory is not much either, but the numbers mean different
    things and one shared threshold would be a constant two callers believe
    for different reasons - which is where this project's most expensive
    defects live.

    The label comes from `answer_evidence.FIELD_NAMES`, which already names
    these three fields for the evidence panel. Restating them here would be a
    second place for a reading's name to go stale.

    ``asked_by`` is how an owner names **this reading**, and it sits on the
    reading rather than anywhere else in the tree. Its first two entries are
    *derived* - the reading's own label and its field, spelled as words - so
    the name this appliance prints is always a name it hears back, and
    ``also_asked_by`` carries only the synonyms that cannot be derived.
    Typing them instead put ``'memory use'`` in this module and in
    `answer_evidence.FIELD_NAMES`, which `tests/test_duplicate_literals` found
    on the first run: one idea in two places, inside the repair for one idea
    in two places.

    **Three rounds of review said the subject cannot be inferred from another
    table, and each round moved the boundary rather than removing it.** The
    answer-topic row was tried: four of its rows match a shape and carry no
    words, so "has the backup been restored yesterday" named nothing and got a
    CPU trend. The fact tool was tried: ``cooling.status`` carries fan speed,
    RPM, duty cycle, mode and cooling policy *and* the one temperature this
    store keeps, so "has the fan speed been rising" was answered "CPU
    temperature rose 4 °C". **A tool is not a subject.** One trended reading
    out of a tool's several is exactly the granularity that has no
    representation anywhere else, so it is written here, beside the reading it
    describes, and the guard in `tests/test_assistant_trend_answer.py`
    requires it to contain the reading's own label and field words - rename
    the field and the guard fires rather than the vocabulary quietly
    describing a reading that no longer exists.
    """

    field: str
    unit: str
    steady_within: float
    also_asked_by: Tuple[str, ...]

    @property
    def label(self) -> str:
        return FIELD_NAMES.get(self.field, str(self.field).replace("_", " "))

    @property
    def asked_by(self) -> Tuple[str, ...]:
        """Every way an owner names this reading, its own names first."""
        names = [self.label.lower(), self.field.replace("_", " ")]
        for phrase in self.also_asked_by:
            if phrase not in names:
                names.append(phrase)
        return tuple(names)


#: The sample fields this module knows how to name, unit and threshold.
#:
#: Deliberately a declared list rather than "every numeric field in the row".
#: `DataLogger.get_data` writes whatever the hardware bridge hands it, so a
#: generic walk would report "raid_fan_1 rose 3" in the machine's own voice
#: with no unit and no idea what the number means. A field absent from here is
#: simply not trended, which is a silence rather than a wrong sentence.
#:
#: Ordered as an owner asks: the live question was about temperature.
TRENDED_READINGS: Tuple[Reading, ...] = (
    Reading(
        "cpu_temperature", "°C", 0.5,
        # ``cooling`` is claimed and ``fan``, ``rpm``, ``airflow`` and
        # ``cool down`` deliberately are not: the temperature is the one
        # cooling reading this store keeps, and the rest of that subsystem is
        # what it does not. A sentence naming both - "is the cooling fan ok" -
        # still refuses, because one unclaimed subject is enough.
        ("cpu temp", "temperature", "temperatures",
         "temp", "temps", "thermal", "thermals", "cooling",
         "hot", "hotter", "hottest", "warm", "warmer", "warmest",
         "cool", "cooler", "coolest", "cold", "colder", "coldest",
         "heat", "degrees", "celsius", "overheat", "overheating",
         "cpu", "cpus", "processor", "processors"),
    ),
    Reading(
        "cpu_percent", "%", 2.0,
        ("cpu usage", "cpu load", "cpu", "cpus",
         "processor", "processors", "load", "loads", "usage",
         "busy", "idle"),
    ),
    Reading(
        "memory_percent", "%", 1.0,
        ("memory usage", "memory", "ram", "swap",
         "free memory", "memory pressure"),
    ),
)


class Trend(NamedTuple):
    """Which way one reading moved across the samples in hand."""

    reading: Reading
    direction: str
    first: float
    last: float
    low: float
    high: float
    count: int

    @property
    def change(self) -> float:
        return abs(self.last - self.first)


class Span(NamedTuple):
    """What period the samples actually cover, and how many there are.

    ``seconds`` is ``None`` when the rows carried no readable time. That is a
    third state and it stays one: reporting an unread interval as zero, or
    borrowing the window from the question, is how "the last hour" gets
    attached to thirty seconds of data.
    """

    count: int
    seconds: Optional[float]


# ---------------------------------------------------------------------------
# The vocabulary, and its one regular expression
# ---------------------------------------------------------------------------

#: Single words for a reading that moved.
#:
#: :data:`ROSE` and :data:`FELL` are spliced in rather than typed again, and
#: that is not tidiness. The appliance must be able to hear back the answer it
#: gives: it replies "CPU temperature rose 1.5 °C", and the owner's follow-up
#: "did it rise or fall?" has to reach this reading. Written twice, the pair
#: could part company - which is the whole of #203 in miniature, one file down.
_DIRECTION_WORDS: Tuple[str, ...] = (
    "rise", "rises", "rising", ROSE, "risen",
    "fall", "falls", "falling", FELL, "fallen",
    "climb", "climbs", "climbing", "climbed",
    "drop", "drops", "dropping", "dropped",
    "increase", "increases", "increasing", "increased",
    "decrease", "decreases", "decreasing", "decreased",
    "trending", "trended",
)

#: The same idea said in two or three words. ``steady`` never appears alone:
#: "a steady job" and "steady on" are not questions about this machine, and a
#: bare entry would make them so. :data:`HELD_STEADY` is spliced in for the
#: reason above.
_DIRECTION_PHRASES: Tuple[str, ...] = (
    "going up", "going down", "gone up", "gone down",
    "went up", "went down", "trending up", "trending down",
    "hold steady", "holds steady", "holding steady", HELD_STEADY,
    "stayed the same", "staying the same", "stayed flat",
)

#: Recent windows the retained samples can speak for.
#:
#: ``last hour`` and ``past hour`` are **not** here: they are already literal
#: entries in `assistant_vocabulary.TOOL_PHRASES`, and a sentence matching both
#: a literal and this pattern would count as two corroborating phrases - the
#: exact arithmetic the single-entry design above exists to avoid. Nothing in
#: this tuple may duplicate one of those literals, and
#: `tests/test_assistant_trend_answer.py` asserts it.
_WINDOW_PHRASES: Tuple[str, ...] = (
    "last few minutes", "past few minutes", "last few hours", "past few hours",
    "last minute", "past minute", "last day", "past day",
    "last week", "past week", "last month", "past month",
    "last year", "past year",
)

#: Everything an owner says to ask which way a reading moved, in one tuple.
#:
#: Read as whole words by `phrase_match.mentions` (through the answer topic
#: table) and as one compiled pattern by `assistant_intents` (through
#: :data:`TREND_PATTERN`). Both renderings come from this tuple.
TREND_PHRASES: Tuple[str, ...] = (
    _DIRECTION_WORDS + _DIRECTION_PHRASES + _WINDOW_PHRASES
)


def _alternation(phrases: Sequence[str]) -> str:
    """Render ``phrases`` as one whole-word alternation.

    Deliberately identical in construction to `phrase_match._pattern`: escaped
    words, joined by ``\\s+`` inside a phrase, longest first, both ends
    anchored. That is what makes :data:`TREND_PATTERN` and
    ``mentions(text, TREND_PHRASES)`` accept the same sentences rather than
    two nearly-equal sets - which is the failure `throttl` was (VD-091), where
    one entry meant the whole family to one reader and nothing to the other.
    """
    return r"\b(?:" + "|".join(
        r"\s+".join(re.escape(word) for word in phrase.split())
        for phrase in sorted(phrases, key=len, reverse=True)
        if phrase.strip()
    ) + r")\b"


#: :data:`TREND_PHRASES` as the single gather-table entry, with the ``re:``
#: prefix `assistant_vocabulary.REGEX_PREFIX` names. It is not written out
#: here; it is built from the tuple above on import.
TREND_PATTERN: str = "re:" + _alternation(TREND_PHRASES)


# ---------------------------------------------------------------------------
# The computation
# ---------------------------------------------------------------------------


def _number(value: Any) -> Optional[float]:
    """``value`` as a float, or ``None`` for anything that is not a reading.

    ``bool`` is excluded on purpose: it is an ``int`` to Python and a state to
    an owner, and "power state rose 1" is not a sentence this appliance may
    write.
    """
    if isinstance(value, bool) or value is None:
        return None
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    return number if number == number else None  # NaN reads as a hole


def _moment(sample: Any) -> Optional[datetime]:
    """The instant a sample was recorded, or ``None`` if it cannot be read.

    InfluxDB returns RFC 3339 with a ``Z`` and up to nanosecond precision;
    `datetime.fromisoformat` takes at most microseconds, so the fraction is
    trimmed rather than allowed to fail the whole span.
    """
    if not isinstance(sample, dict):
        return None
    text = str(sample.get("time") or "").strip()
    if not text:
        return None
    text = text[:-1] + "+00:00" if text.endswith(("Z", "z")) else text
    found = re.search(r"\.(\d+)", text)
    if found and len(found.group(1)) > 6:
        text = text.replace("." + found.group(1), "." + found.group(1)[:6], 1)
    try:
        return datetime.fromisoformat(text)
    except ValueError:
        return None


def _series(samples: Sequence[Any], field: str) -> List[float]:
    return [
        number for number in (
            _number(sample.get(field)) if isinstance(sample, dict) else None
            for sample in samples
        )
        if number is not None
    ]


def span(samples: Sequence[Any]) -> Span:
    """How many samples there are and what period they cover."""
    rows = list(samples or ())
    moments = [moment for moment in map(_moment, rows) if moment is not None]
    if len(moments) < 2:
        return Span(len(rows), None)
    try:
        seconds = abs((max(moments) - min(moments)).total_seconds())
    except TypeError:
        # Mixed aware and naive timestamps. A store that produced them is not
        # a store whose interval can be stated.
        return Span(len(rows), None)
    return Span(len(rows), seconds)


def readings_named(message: str) -> Tuple[str, ...]:
    """The trended readings this sentence asks about, by name.

    Per reading, not per tool and not per topic. That is the whole of round
    three: `cooling.status` carries fan speed and one temperature, so any test
    at tool granularity answers a fan question with a temperature.
    """
    text = str(message or "")
    return tuple(
        reading.field for reading in TRENDED_READINGS
        if mentions(text, reading.asked_by)
    )


def claimed_by_a_reading(phrase: str) -> bool:
    """Whether this vocabulary phrase names one of the retained readings.

    Read against the phrase itself rather than the sentence, so a caller can
    ask of each subject a sentence named: is this one of mine? Anything a
    reading does not claim is a subject this store does not carry.
    """
    text = str(phrase or "")
    return any(mentions(text, reading.asked_by) for reading in TRENDED_READINGS)


def wanted(only: Optional[Sequence[str]]) -> Tuple[Reading, ...]:
    """The readings a caller asked about, or every one this module knows.

    **Empty means every reading, and that is not the same claim it looked
    like.** This was written to give ``None`` and ``()`` different meanings -
    "no reading named" against "readings named, none of them mine" - and a
    mutation proved the distinction unreachable: the caller collapses one into
    the other, because a question naming somebody else's subject is refused
    one branch earlier and never arrives here. A rule production cannot
    execute is `LESSONS` pattern 10 written as code, so it is gone rather than
    documented.

    What survives is the rule that matters: a question naming no reading gets
    every reading, because there is no subject there to be wrong about, and
    the reply names each one it covers.
    """
    if not only:
        return TRENDED_READINGS
    asked = set(only)
    return tuple(item for item in TRENDED_READINGS if item.field in asked)


def trends(
    samples: Sequence[Any], only: Optional[Sequence[str]] = None
) -> Tuple[Trend, ...]:
    """Which way each asked-for reading moved, or nothing at all.

    Empty means no direction can be honestly stated - one sample, no numbers,
    no field this module knows, or none of the fields the question was about.
    It is the *only* gate on saying "rose", "fell" or "held steady" anywhere in
    this product, which is what keeps the invitation and the capability from
    parting company again.
    """
    rows = list(samples or ())
    found: List[Trend] = []
    for reading in wanted(only):
        series = _series(rows, reading.field)
        if len(series) < 2:
            continue
        first, last = series[0], series[-1]
        if abs(last - first) < reading.steady_within:
            direction = HELD_STEADY
        else:
            direction = ROSE if last > first else FELL
        found.append(Trend(
            reading, direction, first, last,
            min(series), max(series), len(series),
        ))
    return tuple(found)


def _figure(value: float, unit: str) -> str:
    rounded = round(float(value), 1)
    shown = "{:.0f}".format(rounded) if rounded == int(rounded) else "{:.1f}".format(rounded)
    return "{}{}".format(shown, unit)


def _plural(count: float, noun: str) -> str:
    shown = int(count) if float(count) == int(count) else round(float(count), 1)
    return "{} {}{}".format(shown, noun, "" if shown == 1 else "s")


#: How a sample is named to a reader. One constant because three sentences
#: below say it, and `tests/test_duplicate_literals` counts a phrase written
#: twice in one module as exactly the shape this project keeps repeating.
_SAMPLE = "retained sample"


def _samples(count: int) -> str:
    return _plural(count, _SAMPLE)


def describe_span(measured: Span, more_retained: bool = False) -> str:
    """What the samples cover, in the units they actually cover it in.

    **This is the sentence the whole task turns on.** Thirty rows at a
    one-second interval are thirty seconds, and the question that produced
    them asked about an hour. The window is never taken from the question.

    **`more_retained` stops the second lie this sentence could tell (VD-097 /
    #224).** VD-097 chose "that is everything this appliance has retained" on
    the premise that the store had just been created and thirty rows really was
    all of it. That premise expired: the store keeps seven days, and the read
    is capped at a recent window (`metrics.history` returns at most its
    requested count, default thirty). Measured live, the reply said "29 seconds
    is everything this appliance has retained" while InfluxDB held 3.4 days and
    297,247 rows. When the caller knows the fetch was capped - it got as many
    rows as it asked for, so older ones exist - it passes `more_retained=True`
    and the reply says what it read is a recent window, not the whole store.
    """
    if measured.seconds is None:
        if more_retained:
            return (
                "Those {} are the most recent samples I read; this appliance has "
                "retained more, but their timestamps could not be read, so I "
                "cannot say what period they cover.".format(_samples(measured.count))
            )
        return (
            "Those {} are everything this appliance has retained. Their "
            "timestamps could not be read, so I cannot say what period they "
            "cover.".format(_samples(measured.count))
        )
    # **Each band ends at 1.5x its unit, and rounding the quotient to a whole
    # number overstated every one of those boundaries.** `round(1.5)` is 2, so
    # 90 seconds was reported as "2 minutes", 5,400 as "2 hours" and 129,600 as
    # "2 days" - a duration nothing measured, which is the one thing this
    # module promises never to write, and reachable on the appliance at a
    # one-second interval with a 91-sample read. One decimal place is kept
    # instead: 90 seconds reads as "1.5 minutes", which is the same instant.
    # `_plural` already renders a non-integer with its decimal and an integer
    # without one, so nothing else has to change.
    seconds = measured.seconds
    if seconds < 90:
        window = _plural(round(seconds), "second")
    elif seconds < 5400:
        window = _plural(round(seconds / 60, 1), "minute")
    elif seconds < 129600:
        window = _plural(round(seconds / 3600, 1), "hour")
    else:
        window = _plural(round(seconds / 86400, 1), "day")
    if more_retained:
        # No time-unit example ("the last hour") in this sentence: naming a
        # window here reads as echoing the one in the question, which is the
        # #203 defect a sibling guard forbids. The measured window above is the
        # only duration this reply may state.
        return (
            "Those {} cover {} - the most recent samples I read. This appliance "
            "has retained more than that; ask for a specific window and I can "
            "summarise it.".format(_samples(measured.count), window)
        )
    return (
        "Those {} cover {} - that is everything this appliance has retained, "
        "so I cannot speak for a longer window than "
        "that.".format(_samples(measured.count), window)
    )


def _opening(label: str) -> str:
    """``label`` starting a sentence, without flattening the rest of it.

    `str.capitalize` lower-cases everything after the first character, which
    turned "CPU temperature" into "Cpu temperature" - an acronym rewritten by
    a formatting call, in a reply whose whole purpose is to be read as a
    machine's own account of itself.
    """
    return label[:1].upper() + label[1:]


def describe(trend: Trend) -> str:
    """One reading's movement, in the direction words an owner asked with."""
    unit = trend.reading.unit
    if trend.direction == HELD_STEADY:
        line = "{} held steady, around {}.".format(
            _opening(trend.reading.label), _figure(trend.last, unit))
    else:
        line = "{} {} {}, from {} to {}.".format(
            _opening(trend.reading.label), trend.direction,
            _figure(trend.change, unit),
            _figure(trend.first, unit), _figure(trend.last, unit))
    # A net move hides a spike, and a single figure pair cannot show one. Say
    # the range whenever the samples went further than the endpoints did.
    if (trend.high - trend.low) - trend.change > trend.reading.steady_within:
        line += " It ranged between {} and {} across those samples.".format(
            _figure(trend.low, unit), _figure(trend.high, unit))
    return line


def trend_lines(
    samples: Sequence[Any],
    only: Optional[Sequence[str]] = None,
    more_retained: bool = False,
) -> Tuple[str, ...]:
    """Every sentence a reply may say about which way the samples moved.

    Empty when :func:`trends` is empty, so there is no sentence to emit and no
    way to write an invitation that outlives the computation behind it.

    ``more_retained`` is passed straight to :func:`describe_span`: True when the
    caller read a capped recent window of a larger store, so the closing
    sentence must not call it the whole retention (VD-097 / #224).
    """
    found = trends(samples, only)
    if not found:
        return ()
    return tuple(describe(trend) for trend in found) + (
        describe_span(span(samples), more_retained),
    )


def _listed(names: Sequence[str]) -> str:
    """``a, b and c`` - the one place this module joins a list of readings."""
    items = list(names)
    if len(items) < 2:
        return "".join(items)
    return "{} and {}".format(", ".join(items[:-1]), items[-1])


def nothing_to_compare(samples: Sequence[Any]) -> str:
    """What is held, when a direction cannot be read out of it.

    Reached when the samples arrived but nothing in them can be compared for
    what was asked. It states the holding and stops - inviting a trend here is
    the defect this module was written to remove.
    """
    measured = span(samples)
    if measured.count == 1:
        return (
            "This appliance is holding {}, so there is nothing to compare it "
            "against yet. Retention records from the moment it "
            "starts.".format(_samples(measured.count))
        )
    return (
        "This appliance is holding {}, and none of them carry two comparable "
        "values for that, so I cannot say which way it "
        "moved.".format(_samples(measured.count))
    )


def off_subject(samples: Sequence[Any]) -> str:
    """What the retained samples are about, when the question is not.

    **The sentence that had to exist before the gate could refuse.** Without
    it the off-subject branch would have to borrow either "no record covering
    that time" - false, the appliance holds samples covering it - or
    :func:`nothing_to_compare`, which claims the rows carry nothing
    comparable when they carry a rising temperature. Naming what is retained
    is the only one of the three that is true.

    The names come from the readings actually found in these rows, falling
    back to what this module can trend at all when there are none, so the
    sentence never lists a reading this store did not record.
    """
    found = trends(samples)
    labels = [trend.reading.label for trend in found] or [
        reading.label for reading in TRENDED_READINGS
    ]
    return (
        "The telemetry this appliance retains is {}. That question is about "
        "something these {}, so I cannot answer it from them.".format(
            _listed(labels), SUBJECT_NOT_RETAINED)
    )


def evidence(samples: Sequence[Any]) -> Dict[str, str]:
    """The evidence entry for a reply built out of these samples."""
    measured = span(samples)
    return {
        "source": "metrics.history",
        "summary": "Read {} retained telemetry samples from this "
                   "appliance.".format(measured.count),
    }
