"""What the built-in answer path can answer, declared once.

`DeploymentAgent._fallback_answer` decided what a question was about with
twelve hand-written keyword tuples, each matched by ``word in lower``. Two
things were wrong with that and only one of them was the matching.

**The matching.** ``ram`` sits inside "prog*ram*", so *"How do I stop a program
from starting at boot?"* was answered *"Memory use is currently 31.0%"*.
Measured on this tree, the same way: ``app`` inside "*app*liance" and "*app*ly",
``light`` inside "high*light*" and "f*light*", ``load`` inside "down*load*",
"up*load*" and "re*load*", ``space`` inside "name*space*", ``cool`` inside
"*cool*ant". `vaelor.phrase_match.mentions` already existed, and
`provider_runtime._TOPIC_KEYS`, `_LIVE_TERMS` and `assistant_intents`'
tables already matched through it. The answer path was the copy that did not.

**The count.** Twelve places were free to invent their own matching, which is
why correcting twelve lists is not the repair - it leaves the thirteenth free.
One table, one predicate, and `tests/test_assistant_builtin_answers.py` fails
on any word vocabulary in the answer path matched any other way.

**Inflections are listed, not derived**, following the reasoning already
written into `provider_runtime._LIVE_TERMS`: a ``\\w*`` suffix re-admits
"fancy" for ``fan``, "random" for ``ram`` and "iPhone" for ``ip``. Listing
them is also the half of this repair that could have gone wrong quietly -
matching ``fan`` as a whole word alone loses "fans", "displays", "lights",
"updates", "services", "jobs", "disks" and "packages", and every one of those
falls through to a refusal. Trading a wrong answer for a wrong refusal is not
a fix; LESSONS pattern 4 is eight instances of a machine denying what it could
read.

**A topic must not claim a reading the gather path cannot supply**, and this
table shipped claiming nine. `assistant_intents.select_tools` decides what is
fetched; the branches here decide what is said about what arrived. They are two
stages of one pipeline, so a word that only ever appears *here* names a branch
that can never run: "what is installed on this machine", "is the system
updating right now" and "is anything still installing" each reached this path
with nothing fetched, fell past every branch, and got the capability refusal
while the appliance held the answer. That is #188's defect - one idea in two
hand-maintained tables - between two modules written in one week rather than
across a release, which VD-091 fixed once already.

**It is not fixed by making this table inherit the other one**, and that was
measured rather than assumed. `provider_runtime`'s filter rows do inherit,
because a filter narrower than its gather always loses a reading. This table is
not a filter. Inheriting `literal_phrases` per row would *delete* ``running``
from `services` and `workloads` - it is the one regex entry, and a whole-word
reader cannot see it - and would hand `updates` the word ``reboot``, so "when
did it last reboot" would be answered "0 operating-system updates are
available". Gathering errs generous on purpose (VD-093: a missed word costs a
reading nobody fetched); saying errs precise (a wrong branch costs a wrong
answer). Merging tables with opposite error costs is what #190 was.

So the relation is stated instead, and guarded in the one direction where
divergence is a defect: :attr:`Topic.reads` names what the branch needs in
hand, `tests/test_assistant_builtin_answers.py` drives `select_tools` with
every word and fails on any that cannot fetch it, and
:data:`ANSWER_ONLY_WORDS` is where a word that genuinely cannot gather says so
and why. The converse - a gather word no branch can spend - is left open and
recorded there, because closing it widens what this path answers. VD-094 is
the decision and carries the measurements.

**And the subject column, which is the other defect this table closes.** The
refusal written when nothing answered ended *"I can still answer appliance
questions ... using live cooling, lighting, display, storage, network,
services, workloads, and updates"* - eight subjects, typed by hand, against
eighteen this path actually answers. An owner reading that is told this
appliance cannot tell them whether anything is wrong, what its version is,
whether it has a GPU, or how much memory is fitted, all of which it can.
The sentence is now derived from the rows below, so the two cannot drift.
"""

from __future__ import annotations

from typing import Dict, NamedTuple, Optional, Tuple

from .assistant_vocabulary import literal_phrases
from .phrase_match import mentions
from .telemetry_trend import TREND_PHRASES

#: Every word that fetches the retained samples, as the branch that spends
#: them hears it.
#:
#: **This row inherits and the others do not, and the difference is measured
#: rather than stylistic.** VD-094 rejected inheritance for this table because
#: `literal_phrases` *deletes* ``running`` from `services` and `workloads` -
#: the one regex entry, invisible to a whole-word reader - and hands `updates`
#: the word ``reboot``, so "when did it last reboot" would be answered with a
#: package count. Neither applies here: this row spends exactly one reading,
#: `metrics.history`'s gather list is entirely words about retained samples,
#: and its regex entry is `TREND_PATTERN`, which is added back from the tuple
#: it was built from. So the vocabulary that fetches the samples and the
#: vocabulary that spends them are the same set by construction.
#:
#: Without the inherited half, "show me the temperature trend over the last
#: hour" - the sentence that opened #203 - fetched the samples and was heard
#: by no word in this table.
_HISTORY_WORDS: Tuple[str, ...] = TREND_PHRASES + tuple(
    phrase for phrase in literal_phrases("metrics.history")
    if phrase not in TREND_PHRASES
)


class Topic(NamedTuple):
    """One subject the built-in answer path can speak about.

    ``words`` is empty where the branch that answers this subject does its own
    matching on a *shape* rather than a vocabulary - a past-tense construction,
    a fault determiner - and cannot be reduced to a word list. Those rows are
    still declared, because the capability sentence has to name them and
    because ``answered_by`` is checked against the tree.

    ``reads`` names the fact keys the branch must find in ``facts`` before it
    can write a line, and it is what relates this table to the gather table.
    Every word in ``words`` must make `assistant_intents.select_tools` fetch
    one of them, or the branch is asking about a reading nobody went for -
    see :data:`ANSWER_ONLY_WORDS` for the one declared exception and
    `tests/test_assistant_builtin_answers.py` for the guard.

    ``extra`` is this table's own contribution, kept separate from ``words``
    rather than only folded into it, for the reason
    `provider_runtime.Topic.extra` gives: a guard that could read only
    ``words`` cannot tell a word that gathers from a word that was declared
    unable to, and would pass against either. It is **not written in the row**
    - it is read out of :data:`ANSWER_ONLY_WORDS` by :func:`_topic`, so the
    word and the reason it cannot gather are one sentence in one place. Writing
    it in both is the restatement `provider_runtime`'s own guard forbids, and
    it put three sentences into this module twice on the first attempt.
    """

    name: str
    words: Tuple[str, ...]
    subject: str
    answered_by: str
    reads: Tuple[str, ...] = ()
    extra: Tuple[str, ...] = ()


_FALLBACK = "vaelor.deployment_agent:DeploymentAgent._fallback_answer"


#: Topics whose branch spends no *reading*, so the gather relation cannot
#: apply to them. Declared rather than inferred from an empty ``reads``: an
#: empty field is what a new row has before anyone thinks about it, and a
#: guard that reads silence as consent is the one that does nothing.
TOPICS_THAT_SPEND_NO_READING: Tuple[str, ...] = ("specialist",)


#: Words the answer path matches that **cannot** fetch the reading they spend,
#: and why each is not simply a word missing from the gather table.
#:
#: Keyed ``(topic, word)``, and this is the **only** place these words are
#: written: :func:`_topic` appends them to the row's vocabulary, so deleting a
#: reason deletes the word. Restating them in the row as well is what
#: `provider_runtime`'s own guard forbids, and it wrote three sentences into
#: this module twice before the count in `test_duplicate_literals` said so.
#:
#: **A two-part key rather than ``"topic:word"``, and the reason is a guard.**
#: Spliced into one string, the word stops being a literal anybody can find:
#: `'what is wrong'` is written in `provider_runtime` too, and folding it into
#: a compound key silently removed it from `KNOWN_DUPLICATES` - repairing a
#: recorded duplication by hiding one half of it rather than by merging the
#: two. The tuple keeps the word greppable and keeps that backlog honest.
#:
#: Every entry is charged a written reason, for the same finding that charges
#: :data:`DECLARED_SUBSTRING_VOCABULARIES` forty characters: a backlog that
#: costs nothing gets declared away, and this is exactly the table a future
#: hand-correction would try to grow rather than fix the vocabulary. The guard
#: fails on an entry whose word *does* gather - that is somebody widening this
#: list instead of noticing the widening made it unnecessary.
ANSWER_ONLY_WORDS: Dict[Tuple[str, str], str] = {
    ("services", "status"): (
        "The widest word an owner uses, and VD-091 measured what putting it "
        "in the gather table costs: the union of the gather and filter "
        "vocabularies took #190's scope battery to 0 of 12, with `status` one "
        "of the four terms that newly suppressed the redirect. \"What is the "
        "status of the Voyager probe\" would become a question about this "
        "appliance. It stays here because it can still spend a services "
        "reading another word fetched - \"is the daemon status ok\" gathers "
        "on `daemon`, which no answer word covers - and `provider_runtime`'s "
        "own `status` row is declared local for the same reason."
    ),
    ("health-verdict", "working correctly"): (
        "Fetches no reading at all, so the verdict line runs only when "
        "another word in the same sentence gathered the cooling reading it "
        "is gated on. Not repaired here: the fix is to widen "
        "`_fallback_answer`'s gate from `cooling` to the health reading, "
        "which changes the most-read sentence this product writes and is the "
        "owner's call - VD-090's reconciliation records that verdict "
        "disagreeing across surfaces once already."
    ),
    ("health-verdict", "healthy"): (
        "Gathers `services.status` and not `cooling.status`, so the verdict "
        "is gated out while a services reading sits in hand unspent - which "
        "is LESSONS pattern 4, live. The cheap repair is to give the "
        "`services` topic the words its own gather list already owns "
        "(`healthy`, `broken`, `failing`, `degraded`, `daemon`), and that is "
        "a widening of what the built-in path answers, filed for the owner "
        "rather than taken in a pre-hardware correction batch."
    ),
    ("health-verdict", "what is wrong"): (
        "Gathers `services.status` through `wrong`, not `cooling.status`. "
        "The `services` topic already carries `wrong`, so this phrase is "
        "redundant with a row that does gather, and the sentence is answered "
        "- by the services branch rather than the verdict branch. Kept so "
        "the two rows are not silently reordered into disagreement."
    ),
    ("health-verdict", "needs attention"): (
        "Gathers `health.status` through `attention`, which no topic in this "
        "table reads: the verdict branch is gated on `cooling.status` and "
        "`faults` matches a shape rather than a vocabulary. The reading is "
        "fetched and no worded branch can spend it, which is the same "
        "LESSONS pattern 4 shape as `healthy` above and is filed with it."
    ),
}


def _topic(
    name: str,
    words: Tuple[str, ...],
    subject: str,
    answered_by: str,
    reads: Tuple[str, ...] = (),
) -> Topic:
    """One row, with its declared answer-only words folded in.

    ``words`` lists what an owner says that also *fetches* the reading this
    branch spends. Anything that cannot fetch it lives in
    :data:`ANSWER_ONLY_WORDS` with the reason, and arrives here - one word,
    one place, and a reason that is load-bearing rather than commentary.
    """
    extra = tuple(
        word for topic_name, word in ANSWER_ONLY_WORDS if topic_name == name
    )
    return Topic(
        name, tuple(words) + extra, subject, answered_by, tuple(reads), extra)


#: Ordered as `_fallback_answer` asks, because the order is load-bearing where
#: vocabularies overlap: `services` and `workloads` both hold "running".
ANSWER_TOPICS: Tuple[Topic, ...] = (
    _topic(
        # `_fallback_answer` gates this line on a cooling reading being in
        # hand. Four more words reach this row from `ANSWER_ONLY_WORDS`, each
        # of which fetches no cooling reading and says there why.
        "health-verdict",
        ("is everything", "overall", "anything wrong"),
        "the overall health verdict",
        _FALLBACK,
        reads=("cooling.status",),
    ),
    # **This row is not the whole vocabulary for temperature, and a second
    # reader had to find that out.** It carries no `hot`, `warm` or `hotter`,
    # while `assistant_vocabulary`'s `cooling.status` list does - so "is the
    # cpu hot compared to earlier" reaches this branch and does not reach this
    # row. `telemetry_trend.TRENDED_READINGS[0].asked_by` therefore carries
    # its own temperature words rather than reading this one; it also has to
    # separate the temperature from the fan, which this row deliberately does
    # not. A third reader must not assume these words are the subject
    # vocabulary for a CPU temperature question.
    _topic(
        "cooling",
        ("fan", "fans", "cool", "cools", "cooled", "cooling",
         "temperature", "temperatures", "thermal", "thermals"),
        "cooling and temperature",
        _FALLBACK,
        reads=("cooling.status",),
    ),
    _topic(
        "display",
        ("oled", "display", "displays", "screen", "screens"),
        "the front display",
        _FALLBACK,
        reads=("display.status",),
    ),
    _topic(
        "lighting",
        ("rgb", "light", "lights", "lighting"),
        "case lighting",
        _FALLBACK,
        reads=("lighting.status",),
    ),
    _topic(
        # "patch" and "patches" are not a widening: `select_tools` already
        # gathers `updates.status` for them, so the fact arrived and this
        # branch had no word to spend it on - which is how "can I apply a
        # patch" was refused over a reading in hand.
        "updates",
        ("update", "updates", "updated", "updating", "upgrade", "upgrades",
         "upgraded", "upgrading", "package", "packages", "patch", "patches"),
        "operating-system updates",
        _FALLBACK,
        reads=("updates.status",),
    ),
    _topic(
        # ``status`` reaches this row from `ANSWER_ONLY_WORDS`: it must not
        # become a gather word, and that measurement is written there.
        "services",
        ("service", "services", "health", "wrong", "problem",
         "problems", "running"),
        "managed services",
        _FALLBACK,
        reads=("services.status",),
    ),
    # ``cpu`` and ``memory`` spend `system.telemetry`, which is a base tool:
    # it is gathered for every question ever asked, so no word selects it and
    # no word for it belongs in the gather table. That is the same boundary
    # `assistant_vocabulary` states and `provider_runtime`'s rows rely on.
    _topic(
        "cpu",
        ("cpu", "cpus", "processor", "processors", "load", "loads", "usage"),
        "processor load",
        _FALLBACK,
        reads=("system.telemetry",),
    ),
    _topic(
        "memory",
        ("ram", "memory", "swap"),
        "memory",
        _FALLBACK,
        reads=("system.telemetry",),
    ),
    _topic(
        # "capacity" comes from `provider_runtime._TOPIC_KEYS`, which already
        # routes it to `storage.status`; the two encodings disagreed.
        "storage",
        ("storage", "disk", "disks", "nvme", "microsd", "sd card", "space",
         "capacity"),
        "storage capacity",
        _FALLBACK,
        reads=("storage.status",),
    ),
    _topic(
        "network",
        ("network", "networks", "ip", "ips", "ip address", "ethernet",
         "wifi", "wi-fi", "dns", "internet", "hostname"),
        "the network",
        _FALLBACK,
        reads=("network.status",),
    ),
    _topic(
        "workloads",
        ("app", "apps", "application", "applications", "container",
         "containers", "model", "models", "installed", "running"),
        "installed apps and models",
        _FALLBACK,
        reads=("workloads.inventory",),
    ),
    _topic(
        "jobs",
        ("job", "jobs", "task", "tasks", "progress", "installing",
         "downloading"),
        "running jobs",
        _FALLBACK,
        reads=("jobs.recent",),
    ),
    # The only worded row that spends no *reading*: its branch reads
    # `specialist_results` off the conversation context, which no fact tool
    # gathers. Declared in `TOPICS_THAT_SPEND_NO_READING` so a new row cannot
    # opt out of the gather relation by leaving `reads` empty.
    _topic(
        "specialist",
        ("specialist", "specialists", "agent result", "agent output"),
        "specialist review results",
        _FALLBACK,
    ),
    # Below: subjects this path answers through a matcher that is a shape
    # rather than a vocabulary. They carry no words and are never asked about
    # here; they are declared so the capability sentence tells the truth.
    _topic(
        "faults", (), "active warnings and alerts",
        "vaelor.assistant_fault_answers:health_alert_answer",
    ),
    _topic(
        "accelerator", (), "the GPU and neural accelerator",
        "vaelor.assistant_fault_answers:accelerator_presence_answer",
    ),
    _topic(
        "identity", (), "this appliance's software version",
        "vaelor.assistant_answer_scope:identity_answer",
    ),
    _topic(
        "recovery", (), "backups and restore points",
        "vaelor.assistant_recovery:recovery_answer",
    ),
    # **The one row above that stopped being shape-only.** It carried no words
    # while `past_time_answer` was gated on tense alone, so "did the cpu
    # temperature rise, fall, or hold steady" - the question the appliance
    # itself invited - reached no branch here and was answered with the
    # current instant. The vocabulary is `_HISTORY_WORDS` above: the trend
    # module's own tuple plus everything else that fetches this reading, so
    # the words that ask for a trend, the words that fetch the samples and the
    # words the answer reads back cannot become three sets.
    #
    # `reads` makes that relation enforced rather than described:
    # `TheAnswerPathClaimsOnlyWhatTheGatherPathFetchesTests` fails on any word
    # here that `select_tools` will not fetch `metrics.history` for.
    _topic(
        "history", _HISTORY_WORDS, "recent telemetry samples",
        "vaelor.assistant_answer_scope:past_time_answer",
        reads=("metrics.history",),
    ),
)

_BY_NAME: Dict[str, Topic] = {topic.name: topic for topic in ANSWER_TOPICS}


#: Sites that decide a topic from an owner's words by substring, deliberately.
#:
#: Keyed ``<path>:<function>`` rather than by line, so an edit above does not
#: turn a live declaration into a stale one. The reason is charged forty
#: characters and printed by every failure: `test_wire_vocabularies` shipped
#: with free backlog tables and a real finding was silenced by a pasted line
#: carrying an empty note.
DECLARED_SUBSTRING_VOCABULARIES: Dict[str, str] = {
    "vaelor/assistant_answer_presentation.py:is_general_knowledge_question": (
        "Its entries are positional, not lexical: ' my ', ' our ' and "
        "' this ' carry hand-written spaces that mean 'not at the start of "
        "the sentence', which is a boundary rule `mentions` does not express "
        "and would silently discard. 'current' is also there to reach "
        "'currently'. Converting it changes what the phrases mean."
    ),
    "vaelor/deployment_plans.py:fallback_plan": (
        "Reached only behind `assistant_policy.is_deployment_request`, which "
        "already splits the message into words with `re.findall` and has "
        "established that a deployment of some workload was asked for. These "
        "lists choose which *kind* of plan among candidates already agreed to "
        "be workload words, so a fragment inside a longer word cannot pull a "
        "non-deployment question in here."
    ),
}


def asks_about(topic: str, message: str) -> bool:
    """Whether ``message`` asks about ``topic``, as whole words.

    Raises for a topic nobody declared, rather than returning ``False``: a
    silent ``False`` is a branch that can never run, which is the shape of
    LESSONS pattern 14 and reads exactly like a question nobody asks.
    """
    row = _BY_NAME.get(str(topic))
    if row is None or not row.words:
        raise KeyError(
            "No answer topic named {!r} carries a vocabulary. Declared: "
            "{}".format(topic, sorted(
                name for name, item in _BY_NAME.items() if item.words
            ))
        )
    return mentions(str(message or ""), row.words)


def subjects_answered() -> Tuple[str, ...]:
    """Every subject this path can answer, for the sentence that lists them."""
    return tuple(topic.subject for topic in ANSWER_TOPICS)


def capability_sentence(machine: Optional[str] = None) -> str:
    """The one place that says what this appliance can still answer.

    Derived from :data:`ANSWER_TOPICS` so it cannot understate the path the
    way the hand-typed list did.
    """
    subjects = subjects_answered()
    listed = "{}, and {}".format(", ".join(subjects[:-1]), subjects[-1])
    return (
        "I can still answer appliance questions about {} from live readings: "
        "{}.".format(str(machine or "this Vaelor node"), listed)
    )
