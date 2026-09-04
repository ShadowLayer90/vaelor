"""Map a plain-language question onto the live fact tools it needs, and say
when none of them can.

The previous selector table matched bare substrings, which failed twice over:

* Ordinary thermal phrasing was invisible to it. "is my pi running hot" shares
  no substring with ``fan/cool/temperature/thermal/rpm``, so no cooling facts
  were gathered.
* Short tokens matched greedily across topics. That same question *did* contain
  "running", so it selected the service and workload tools and was answered
  with a list of managed apps instead of a temperature.

Matching here is word-boundary anchored and grouped by what a person actually
says, so "running hot", "too warm", "overheating" and "what temperature" all
reach the same cooling facts, while "what is running" still reaches services.

The same table answers a second question. The Assistant runs a small model and
will be asked about history, science, and the news, which it would answer
confidently and wrongly. Nothing here asks that model to judge its own
competence - a small model is unreliable at that. What is asked instead is
whether *this machine* holds any evidence for the question: the tools above
cover the appliance and nothing covers the rest, so a question none of them
touches has no source behind it and is redirected to AI Chat rather than
guessed at. See :func:`is_out_of_scope_question`, which is deliberately hard to
trigger - refusing a question the appliance could have answered removes working
capability, which is worse than the guess it was meant to prevent.

**Answering both questions from one table is what broke the second (#190).** A
word earns its place here by being how an owner asks for a reading, and several
of the best ones are ordinary English: `address`, `history`, `trend`, `ago`.
Read as "which readings would help", they are right. Read as "is this appliance
business at all", they made "who delivered the Gettysburg address" a network
question. The two readings have opposite error costs - a missed word costs a
reading nobody fetched, a false in-scope costs a fabricated answer in the
machine's own voice - so the scope reading now asks two further things of the
sentence that gathering never has to: whether it points at this machine, and
whether more than one phrase is doing the work. See
:func:`_rests_on_one_borrowed_word`.
"""

from __future__ import annotations

import re
from typing import Any, Dict, Iterable, Mapping, Optional, Sequence, Set, Tuple

from .answer_evidence import describe_missing, first_sentence, sentence
from .assistant_answer_presentation import (
    asks_about_host_facts, asks_about_identity, is_appliance_question,
)
from .assistant_vocabulary import BASE_TOOLS, REGEX_PREFIX, TOOL_PHRASES

__all__ = [
    "BASE_TOOLS", "matched_intents", "matched_phrases", "named_subjects",
    "select_tools", "order_facts_by_relevance",
    "is_out_of_scope_question", "knowledge_redirect", "chat_destination",
    "world_followup",
    "CHAT_READY", "CHAT_ABSENT", "CHAT_UNKNOWN",
]

#: The vocabulary that decides which readings to gather, read from its one
#: home. It was written out a second time in `provider_runtime._TOPIC_KEYS`,
#: which filters the gathered facts down to what the small model sees, and the
#: two drifted on every hand edit: "what is my ping?" fetched `network.status`
#: through this table and had it discarded by that one. See
#: `vaelor/assistant_vocabulary.py` for the measurement and VD-091.
_INTENT_PHRASES: Dict[str, Tuple[str, ...]] = TOOL_PHRASES


# Questions that plainly ask for a whole-appliance sweep.
#
# **The open diagnostic question is a sweep, and it had no entry (VD-116 #2).**
# An owner who asks "what should I check?", "what should I look at?" or "anything
# I should keep an eye on?" is asking for exactly what "status report" and
# "overview" ask for - the whole appliance, read and summarised - but in the
# register of inspection rather than the noun "status". Those phrasings share no
# vocabulary with any fact tool, named no sweep phrase, and so reached
# `is_out_of_scope_question` with nothing matched and an interrogative opener,
# which redirected them to AI Chat - a surface that cannot read this machine -
# for a question this machine is the only source for. That is VD-035's defect
# verbatim, one register further on from the fault vocabulary.
#
# The inspection verbs sit here rather than in `_CONTROL_PLANE_WORDS` on purpose:
# a sweep phrase both *suppresses* the redirect (through `matched_intents`, the
# same way "everything" does) **and** gathers the whole-appliance readings, so
# the answer is a health summary rather than merely a non-redirect. A word in the
# suppression list would do the first and not the second, which is #191 - a
# redirect stopped, no evidence gathered, a small model left to refuse from
# identity and telemetry.
#
# Anchored to the self-directed inspection register ("should I check", "keep an
# eye", "watch out for") rather than the bare verb: "check" alone would make
# "how do I check my blood pressure" a sweep. Each phrase carries an owner
# sentence somebody wrote in `tests/test_assistant_intents.OPEN_DIAGNOSTIC_
# QUESTIONS`, and `GENERAL_KNOWLEDGE_AFTER_WIDENING` in the same file re-proves
# the general set still redirects (measured live on the Z2, before/after).
_BROAD_PHRASES: Tuple[str, ...] = (
    "everything", "overall", "diagnose", "full check", "health check",
    "anything i should", "anything wrong", "how is my", "how's my",
    "status report", "overview",
    "should i check", "should i look", "keep an eye", "keeping an eye",
    "look out for", "watch out for", "keep tabs",
)

_BROAD_TOOLS: Tuple[str, ...] = (
    "cooling.status", "lighting.status", "display.status", "updates.status",
    "services.status", "storage.status", "network.status",
    # The verdict every other surface shows. A whole-appliance sweep that does
    # not include it leaves the Assistant reasoning towards a health answer the
    # sidebar and Home have already computed, against thresholds it would have
    # to guess at.
    "health.status",
    # A whole-appliance sweep that skips the accelerator is not a whole-
    # appliance sweep on a machine whose accelerator does the work. Both
    # degrade to a stated absence on hardware without them.
    "gpu.status", "npu.status",
)


def _compile(phrases: Iterable[str]) -> Tuple[re.Pattern[str], ...]:
    compiled = []
    for phrase in phrases:
        if phrase.startswith(REGEX_PREFIX):
            compiled.append(
                re.compile(phrase[len(REGEX_PREFIX):], re.IGNORECASE))
            continue
        # Allow up to three filler words inside a multi-word phrase so
        # "running really hot" still matches "running hot".
        parts = [re.escape(part) for part in phrase.split()]
        joined = r"(?:\W+\w+){0,3}\W+".join(parts) if len(parts) > 1 else parts[0]
        # **Both ends anchored, and one entry used to be exempt.** ``throttl``
        # was compiled as a prefix so it could stand for the whole family. Once
        # the table moved to `assistant_vocabulary` and a second reader
        # appeared, that shorthand became an entry meaning two different things
        # in the two places it was read - `phrase_match.mentions` anchors both
        # ends and matched none of the family. The inflections are written out
        # there instead, so no entry now depends on which matcher reads it.
        compiled.append(re.compile(r"\b" + joined + r"\b", re.IGNORECASE))
    return tuple(compiled)


_COMPILED_INTENTS: Dict[str, Tuple[re.Pattern[str], ...]] = {
    tool: _compile(phrases) for tool, phrases in _INTENT_PHRASES.items()
}
_COMPILED_BROAD: Tuple[re.Pattern[str], ...] = _compile(_BROAD_PHRASES)


def matched_intents(message: str) -> Set[str]:
    """Return the fact tools whose vocabulary appears in ``message``."""
    text = str(message or "")
    if not text.strip():
        return set()
    matched = {
        tool
        for tool, patterns in _COMPILED_INTENTS.items()
        if any(pattern.search(text) for pattern in patterns)
    }
    if any(pattern.search(text) for pattern in _COMPILED_BROAD):
        matched.update(_BROAD_TOOLS)
    return matched


def matched_phrases(message: str) -> Set[str]:
    """Return the distinct vocabulary phrases appearing in ``message``.

    :func:`matched_intents` answers "which readings would help", which is what
    gathering needs. This answers "how much of the sentence actually asked for
    a reading", which is what *scope* needs, and the two are not the same
    count: "what is my ip address", "who delivered the Gettysburg address" and
    "what is the mac address of the adapter" all select exactly one tool, while
    they say one, one and two vocabulary phrases respectively.

    **Returning a set of phrase text is the de-duplication**, not a detail.
    ``history`` is listed under both `jobs.recent` and `metrics.history`, and a
    sentence that says it once has said one word; counting it twice would
    corroborate "who wrote the history of the Peloponnesian War" with itself
    and put it straight back in scope.

    **`_BROAD_PHRASES` is read here too, and omitting it was a defect** caught
    by `test_deployment_agent` driving the real route: "why is everything so
    sluggish today" selects all twelve sweep tools through ``everything`` and
    named no phrase at all, so the strongest appliance signal this product has
    counted as zero evidence and the question was redirected. A phrase that
    selects a set of tools is still a phrase; only its right-hand side differs.

    This walks the two compiled tables rather than building a third keyed by
    phrase. A derived collection cannot be traced back to a line of source, so
    `test_vocabulary_reachability` could not place its entries or carry a
    waiver on one - the pairing below is the same correspondence `_compile`
    already guarantees, made explicit.

    **Both tables are looped, not indexed**, which is also that guard's rule:
    `_COMPILED_INTENTS[tool]` would subscript an instrumented table the
    reachability walk fans out across on the assumption its callers iterate
    it, and one indexed entry makes the whole table read as evaluated against
    subjects it never saw. `_COMPILED_INTENTS` is built by a comprehension
    over `_INTENT_PHRASES`, so the two share key order by construction and
    zipping their values pairs each phrase with its own pattern.
    """
    text = str(message or "")
    if not text.strip():
        return set()
    found = {
        phrase
        for phrases, patterns in zip(
            _INTENT_PHRASES.values(), _COMPILED_INTENTS.values())
        for phrase, pattern in zip(phrases, patterns)
        if pattern.search(text)
    }
    found.update(
        phrase
        for phrase, pattern in zip(_BROAD_PHRASES, _COMPILED_BROAD)
        if pattern.search(text)
    )
    return found


def named_subjects(message: str) -> Set[str]:
    """The vocabulary phrases that name a **subject**, without the sweep words.

    :func:`matched_phrases` includes `_BROAD_PHRASES` deliberately and says
    why: "A phrase that selects a set of tools is still a phrase; only its
    right-hand side differs." That is right for *scope*, which counts how much
    of a sentence asked for a reading.

    It is wrong for a caller asking "what is this question about", and the
    difference cost an answer. ``how is my`` selects all twelve sweep tools
    and names no subject at all, so "how is my cpu doing over the last few
    minutes" was read as naming eleven subjects the retained samples do not
    carry, and was refused with a sentence that named CPU temperature as what
    the appliance retains - about a question asking for CPU temperature. A
    sweep asks for everything; it does not ask about something else.
    """
    return matched_phrases(message) - set(_BROAD_PHRASES)


def select_tools(message: str, *, base: Sequence[str] = BASE_TOOLS) -> Set[str]:
    """Return every fact tool needed to answer ``message``."""
    from .assistant_acting_wiring import mentions_workload_action

    selected: Set[str] = set(base)
    selected.update(matched_intents(message))
    # A checkpoint question is meaningless without knowing what it protects.
    if "recovery.checkpoints" in selected:
        selected.add("workloads.inventory")
    # VD-100 evidence gate: an acting request ("restart nextcloud") must read
    # live inventory this turn, so the proposal binds to a managed app that
    # reading named - not to a name the model supplied.
    if mentions_workload_action(message):
        selected.add("workloads.inventory")
    return selected


def order_facts_by_relevance(facts: Dict[str, Any], message: str) -> Dict[str, Any]:
    """Return ``facts`` with the readings this message asks about first.

    A custom-agent run assembles every granted reading, but the granted-context
    budget keeps only the *leading* entries once it is compacted for the model
    (``provider_runtime.provider_context``), and the tool registry hands facts
    back in alphabetical order. So a granted reading late in the alphabet -
    ``storage.status`` for "how much NVMe space is free" - was gathered off the
    hardware and then truncated away before the model saw it, which is exactly
    the "granted context does not include disk data" the model then reported.
    Moving the readings this question selects to the front makes a granted
    capability's *data* survive the budget, not just its name. Nothing is
    dropped; only the order is a priority. ``select_tools`` is the same governed
    vocabulary the Assistant gathers by, so the two cannot disagree about which
    reading answers a question.
    """
    if not isinstance(facts, dict) or not facts:
        return facts
    relevant = select_tools(message)
    ordered = {key: value for key, value in facts.items() if key in relevant}
    for key, value in facts.items():
        if key not in ordered:
            ordered[key] = value
    return ordered


# ---------------------------------------------------------------------------
# Scope: questions this appliance holds no evidence for
# ---------------------------------------------------------------------------

# Product vocabulary that names this appliance or the control plane around it
# without having a fact tool of its own. "who is the current administrator" and
# "where do i change the api endpoint" are machine questions even though no
# `*.status` tool answers them, and a scope gate that knew only the hardware
# words would have sent both to AI Chat. Removing working capability is the
# worse failure, so this list exists to suppress the redirect, never to trigger
# one.
_CONTROL_PLANE_WORDS = frozenset({
    "system", "machine", "hardware", "firmware", "bios", "kernel", "os",
    "hostname", "enclosure", "chassis", "board", "sensor", "sensors",
    "telemetry", "metric", "metrics", "reading", "readings", "dashboard",
    "account", "accounts", "user", "users", "username", "login", "signin",
    "role", "roles", "admin", "administrator", "operator", "viewer",
    "permission", "permissions", "audit", "session", "sessions", "auth",
    "sidebar", "page", "tab", "screen", "setting", "settings", "preference",
    "api", "endpoint", "route", "webhook", "certificate", "firewall",
    "credential", "credentials", "connection", "connections", "provider",
    "assistant", "chat", "agent", "agents", "skill", "skills",
    "memory", "memories", "automation", "automations", "connector", "mcp",
    "checkpoint", "swarm", "fleet", "worker", "workers", "tool", "tools",
    "vaelor", "pironman", "appliance", "enclosures",
    # Things this appliance can be asked to run or research. "What is a
    # Palworld server?" was tripping the redirect: it names a workload the
    # deployment planner researches, so it is appliance business even though
    # no `*.status` tool answers it. Found by sweeping every message-shaped
    # string in the test suite through this gate.
    "server", "servers", "image", "images", "repo", "repository", "port",
    "ports", "volume", "volumes", "runtime", "runtimes", "gguf", "daemon",
    "systemd", "process", "processes", "registry", "webui",
    # Inventory fields a person reads off the hardware page.
    "asset", "serial", "sku", "vendor", "manufacturer", "revision", "tag",
    # **Fault vocabulary.** This list had no word for something being wrong,
    # which is the question an appliance is asked most. "What does the error
    # mean", "what is causing the warning", "what should I do about the alert"
    # and "how do I fix the error I am seeing" were all redirected to AI Chat -
    # a surface that cannot read this machine - because they open with an
    # interrogative and name nothing in the intent tables. Adding "this" to any
    # of them rescued it, which is not a phrasing a person owes the product.
    # VD-035 is explicit that a redirect on a diagnostic question is a defect
    # in the tool surface rather than correct behaviour.
    # Several of these already reach a fact tool through `_INTENT_PHRASES`
    # ("problem", "job", "backup") and are listed anyway, so the vocabulary a
    # fault question can use is legible in one place rather than split across
    # two tables by accident of which tool happened to name it.
    "error", "errors", "warning", "warnings", "alert", "alerts", "fault",
    "faults", "log", "logs", "issue", "issues", "problem", "problems",
    "failure", "failures", "diagnostic", "diagnostics", "crash", "exception",
    "timeout", "stacktrace", "traceback", "symptom", "symptoms",
    # **Vaelor's own operational vocabulary.** Each of these is a term the
    # product puts on a screen or in a confirmation, so a question using one is
    # a question about this machine. "What does bootstrap required mean" is the
    # first-run state; "what is the difference between drain and remove" is the
    # cluster page's own pair of verbs.
    #
    # Kept to words this product owns. "plan", "profile", "policy", "console",
    # "desktop", "remote" and "draft" were considered and left out: their
    # general sense dominates, and "draft" in particular would make "draft an
    # email to my landlord" appliance business again - the exact reason the
    # bare possessive "my" was removed from `_SELF_REFERENCE` above.
    # Inflections are listed rather than derived: `_names_control_plane` only
    # strips a trailing "s", so "commissioning" is a different word to it.
    "bootstrap", "bootstrapping", "commission", "commissioning",
    "commissioned", "onboarding", "setup", "reset", "migration",
    "migrate", "drain", "drained", "evict", "eviction", "enroll", "enrollment",
    "node", "nodes", "cluster", "clusters", "controller", "replica",
    "replicas", "quorum", "backup", "restore", "rollback", "snapshot",
    "checkpoints", "job", "jobs", "approval", "approve", "template",
    "blueprint", "lease", "broker", "mfa", "passkey", "quota", "vnc", "kvm",
    "kiosk",
})

#: Generic computing concepts that live in `_CONTROL_PLANE_WORDS` only so a
#: *diagnostic* question can be read on-box ("what is wrong with my system",
#: "which os is installed here"), but whose bare **definition** is world
#: knowledge, not a reading of this machine. "What is an operating system?" and
#: "what is a kernel?" share the word `system`/`os`/`kernel` with that
#: diagnostic vocabulary, yet a textbook definition is AI Chat's to give - the
#: same VD-121 boundary that already sends "what is a CPU?" there. This set is
#: what a bare definition is allowed to rest on and still redirect; the moment
#: it also names a word this product *owns* (`replica`, `cluster`, `server`,
#: `bootstrap` …) the question is this control plane's own and is answered.
#: Kept to the generic OS/hardware nouns a person asks as a definition; the
#: owned operational vocabulary is deliberately excluded.
_GENERIC_COMPUTING_SUBJECT = frozenset({
    "system", "systems", "os", "machine", "machines", "hardware",
    "firmware", "bios", "kernel", "kernels", "software",
})

# Anything addressed to this Assistant, or to the machine it speaks for, is in
# scope by construction: "what can you do", "how do I use this", "is it set up
# here". The bare possessive "my" is deliberately absent - it was already found
# to make "Draft an email to my landlord" appliance business - but every other
# deictic stays, because suppressing the redirect is the safe direction.
_DEICTIC_WORDS = (
    r"you|you'?re|your|yours|yourself|we|our|us|"
    r"this|these|those|here|locally|on\s+box"
)

#: The register an owner uses about the machine in front of them.
#:
#: "my pi", "my ip", "am i on", "did it reboot" - first person and the bare
#: pronoun, which the deictics above omit. These are read **only** where a fact
#: tool has already matched (see :func:`_rests_on_one_borrowed_word`), never as
#: a general suppressor, and that boundary is the whole reason they are a
#: separate string: "my" as a general suppressor makes "Draft an email to my
#: landlord" appliance business, which is why it was taken out of the deictics
#: in the first place. Attached to a reading word it means the opposite.
#:
#: ``i`` is the weakest entry and was measured rather than assumed. It costs
#: "how do I address a letter to a judge", which stays here; it saves "which
#: subnet am i on", which is a real diagnostic question. VD-035 prices the
#: second higher than the first, so it is in.
_OWNER_REGISTER_WORDS = r"my|mine|i|ours|it|its|itself"

_SELF_REFERENCE = re.compile(
    r"\b(?:" + _DEICTIC_WORDS + r")\b", re.IGNORECASE)

#: Every way a sentence can point at this machine. Composed from the deictics
#: rather than written out again: two hand-maintained spellings of one idea is
#: LESSONS 6, and these two would have been read side by side in one function.
_THIS_MACHINE = re.compile(
    r"\b(?:" + _DEICTIC_WORDS + r"|" + _OWNER_REGISTER_WORDS + r")\b",
    re.IGNORECASE,
)

#: How many distinct vocabulary phrases make a reading question, rather than a
#: sentence that happened to borrow one ordinary English word.
#:
#: **Counting tools does not separate these and was measured not to.** "who
#: delivered the Gettysburg address", "what is the mac address of the adapter"
#: and "what is my ip address" each select exactly one fact tool. Counting
#: distinct *phrases* does separate them: one, two ("address" and "mac
#: address") and three. "what was the trend overnight" carries "trend" and
#: "overnight"; "what is the trend in global sea level" carries only "trend".
#:
#: This is read as a **suppressor and never as a trigger**, so raising it can
#: never redirect a question the appliance answers today - it can only cost
#: world-knowledge recall. That asymmetry is what makes it safe under VD-035,
#: and it is why the threshold sits here rather than on `matched_intents`,
#: where a count of two would have redirected real diagnostic questions.
_CORROBORATING_PHRASES = 2

# A question aimed at the world rather than at this box. Two forms count: an
# interrogative opening ("who won the world series"), and an explicit request
# for knowledge or composition ("explain photosynthesis", "write me a poem").
_KNOWLEDGE_OPENER = re.compile(
    r"^\s*(?:so|ok|okay|hey|hi|hello|and|but|also|please)?[\s,]*"
    r"(?:what|what'?s|who|who'?s|when|where|why|which|how|whose|whom)\b",
    re.IGNORECASE,
)
_KNOWLEDGE_REQUEST = re.compile(
    r"\b(?:tell\s+me\s+about|explain|define|definition\s+of|meaning\s+of|"
    r"summari[sz]e|translate|paraphrase|rewrite|"
    r"write\s+(?:me\s+)?(?:a|an|the)|draft\s+(?:me\s+)?(?:a|an)|"
    r"give\s+me\s+(?:a|an|some)|who\s+won|what\s+year|capital\s+of|"
    r"recipe\s+for)\b",
    re.IGNORECASE,
)

#: A request for a DEFINITION, told apart from a diagnostic or reading question
#: that opens the same way. VD-121: "what is a CPU?" is AI Chat's even though it
#: names appliance hardware, but "what is USING the cpu" (diagnostic) and "what
#: is THE cpu temperature" (reading) are this machine's to answer. The
#: discriminator is the **indefinite article**: a definition asks what *a* thing
#: is ("what is a cpu", "explain how an ssd works"), while the diagnostic form
#: puts a verb where the article would be ("what is using/throttling/filling…")
#: and the reading form puts a definite or possessive article ("what is the/my
#: cpu …"). `define X` and "what does a X mean" are the same request in other
#: words. This is deliberately NARROWER than `is_general_knowledge_question`,
#: which a review proved redirects real on-box diagnostics ("what is using the
#: gpu") because its escape hatch requires the article *immediately* after the
#: opener - a gerund defeats it. Matching the article positively cannot.
_DEFINITION_REQUEST = re.compile(
    r"^\s*(?:so|ok|okay|hey|hi|hello|and|but|also|please)?[\s,]*"
    r"(?:"
    r"(?:what(?:'?s| is| are)|explain\s+(?:how|what)|tell\s+me\s+what)"
    r"\s+an?\s+\w"
    r"|define\s+\w"
    r"|what\s+does\s+an?\s+.+\bmean\b"
    r")",
    re.IGNORECASE,
)

# A fragment is a follow-up, not a fresh general question. "why?", "and the
# other one?" and "what about tomorrow" carry no subject of their own, and
# redirecting them would break every ordinary conversation.
_MINIMUM_WORDS = 4
_WORD = re.compile(r"[\w']+")


def _control_plane_words_in(message: str) -> frozenset[str]:
    words = {word.lower() for word in _WORD.findall(str(message or ""))}
    singulars = {word[:-1] for word in words if word.endswith("s") and len(word) > 2}
    return frozenset((words | singulars) & _CONTROL_PLANE_WORDS)


def _names_control_plane(message: str) -> bool:
    return bool(_control_plane_words_in(message))


def _is_bare_generic_definition(message: str) -> bool:
    """A textbook definition whose only control-plane words are generic.

    "What is an operating system?" reaches the control-plane gate because
    ``system``/``os`` sit in `_CONTROL_PLANE_WORDS` for diagnostics, and that
    gate would answer it on a small model with no live reading behind it - the
    exact VD-121 leak that "what is a CPU?" was fixed for, reappearing for any
    definition subject that happens to be a control-plane word. It is world
    knowledge, and AI Chat's, **only** when every control-plane word it names is
    a generic computing concept (`_GENERIC_COMPUTING_SUBJECT`); the moment it
    also names a word this product owns - "what is a replica in a cluster",
    "what is a Palworld server" - the question is this control plane's own and
    stays in scope, so the definition form alone never over-redirects.

    **The deictic is the real discriminator, checked first.** "What is a kernel
    panic on my machine" opens like a definition and rests only on generic words
    (`kernel`, `machine`), but the "my machine" makes it a live diagnostic about
    THIS box - answered on-box, never sent to AI Chat which cannot read the
    machine (VD-035). Only a definition that points at the box *nowhere* ("what
    is a kernel") is world knowledge. Without this the generic fault nouns would
    redirect exactly the on-box diagnostics the control-plane gate exists to
    keep.
    """
    text = str(message or "")
    if not _DEFINITION_REQUEST.match(text):
        return False
    if _THIS_MACHINE.search(text):
        return False
    present = _control_plane_words_in(text)
    return bool(present) and present <= _GENERIC_COMPUTING_SUBJECT


def _asks_for_world_knowledge(text: str) -> bool:
    """Whether the sentence is shaped like a question about the world."""
    return bool(_KNOWLEDGE_REQUEST.search(text)) or bool(
        _KNOWLEDGE_OPENER.match(text.strip()))


def _rests_on_one_borrowed_word(text: str) -> bool:
    """Whether a fact tool matched only because one ordinary word coincided.

    This is #190 in one function. `matched_intents` does two jobs with opposite
    error costs: as a *gather* signal it should err generous, because a missed
    word means a reading is never fetched; as a *scope* signal it should err
    precise, because a false in-scope means a 1.7B model answers a history
    question confidently and wrongly. Every widening of the first narrowed the
    second, and alpha 48's `address`, `ago`, `history` and `trend` narrowed it
    until "who delivered the Gettysburg address" was a network question.

    Three conditions, all required, and each one is a property of the sentence
    rather than a list somebody maintains:

    1. it is shaped like a request for world knowledge;
    2. it points at this machine **nowhere** - no "my", no "this", no "it";
    3. one single vocabulary phrase is doing all the work.

    Ownership is what decides the ambiguous pairs. "what is **my** ip address"
    and "what is the address of **the White House**" share the word `address`
    and differ in whose thing they are about, which is the difference a person
    hears immediately and the gather table cannot express.
    """
    if not _asks_for_world_knowledge(text):
        return False
    if _THIS_MACHINE.search(text):
        return False
    return len(matched_phrases(text)) < _CORROBORATING_PHRASES


def is_out_of_scope_question(message: str) -> bool:
    """Whether nothing on this machine can evidence ``message``.

    Every gate below returns ``False`` - "answer it normally" - so the redirect
    is reached only by a message that is a recognisable request for world
    knowledge *and* shares no vocabulary with any fact tool, any hardware term,
    any control-plane term, and asks nothing of this Assistant. A general
    question phrased outside these forms still reaches the model, which is the
    behaviour that already existed; this narrows nothing that used to work.

    **The tool-match gate is the only conditional one, deliberately.**
    `is_appliance_question` below is read by `chat_appliance_scope` to *decline*
    a question on AI Chat, where a false positive blocks the one thing that
    surface exists for - so qualifying it damages a second surface. Measured:
    applying this same test to it recovers six more world questions and costs
    thirteen appliance ones. `AMBIGUOUS_TOWARD_THE_APPLIANCE` records that
    ambiguity resolves toward the appliance; this narrows what counts as
    ambiguous and does not overturn that.
    """
    text = str(message or "")
    if len(_WORD.findall(text)) < _MINIMUM_WORDS:
        return False
    if matched_intents(text) and not _rests_on_one_borrowed_word(text):
        return False
    # An appliance identity/version question ("what version am I on?") is
    # answered on-box by `assistant_answer_scope.identity_answer`, which
    # consults this same `asks_about_identity`. Both must agree: LESSONS 6 /
    # #205 - the identity path answered it while this gate redirected it to AI
    # Chat, because the version markers lived only in `assistant_answer_scope`.
    # The single detector now lives in `assistant_answer_presentation`; do not
    # copy the markers back into this module.
    if asks_about_identity(text):
        return False
    # OS-version and CPU-core-count questions are answered on-box by
    # `assistant_answer_scope.host_fact_answer` from the standing brief. Both
    # this gate and that answer consult one predicate so they cannot disagree
    # about scope (LESSONS 6 / #205, #144): the identity gate above had exactly
    # this shape when the version markers lived only in one reader.
    if asks_about_host_facts(text):
        return False
    # A control-plane word normally claims the question for this machine - but
    # not when the whole message is a bare definition resting only on a generic
    # computing word ("what is an operating system", "what is a kernel"). That
    # is world knowledge sharing `system`/`os`/`kernel` with the diagnostic
    # vocabulary, so it is left to fall through to the definition redirect
    # below (VD-121). A definition that also names an owned term is not generic
    # and is still answered here.
    if _names_control_plane(text) and not _is_bare_generic_definition(text):
        return False
    if _SELF_REFERENCE.search(text):
        return False
    # **A definition request is AI Chat's, even when it names appliance hardware
    # (VD-121).** VD-035 sends general-knowledge questions to AI Chat, but the
    # shared `is_appliance_question` predicate held "what is a CPU?" in scope
    # because ``cpu`` is a hardware word - so a small NPU model answered a
    # definition it has no live reading behind, the exact "recorded cost"
    # `AMBIGUOUS_TOWARD_THE_APPLIANCE` documented. `_DEFINITION_REQUEST` matches
    # the indefinite-article definition form ("what is A cpu", "explain how AN
    # ssd works", "define X") and NOTHING else, so a diagnostic ("what is using
    # the gpu") or a reading ("what is the fan speed") - both this machine's to
    # answer - is left in scope. It is deliberately narrower than
    # `is_general_knowledge_question`, which a review proved over-redirects the
    # verb-in-the-middle diagnostic form.
    #
    # **Order is load-bearing.** It runs AFTER the identity, host-fact,
    # control-plane and self-reference gates above, because each of those names a
    # question this machine truly owns that can also open with an article: "what
    # is a Palworld server" and "what is a replica in a cluster" are this control
    # plane's own vocabulary (`_names_control_plane`). They are claimed first and
    # never reach this check.
    if _DEFINITION_REQUEST.match(text):
        return True
    # This also covers a request to *change* something here: a message that
    # `detect_action_requests` recognises is an appliance question by
    # construction. A separate guard for it was added and then removed - no
    # mutation could make it fail, because nothing reached it. It sits below the
    # definitional check so a hardware word does not pull a bare definition back
    # into scope, but still catches every reading and action request, which the
    # form test above deliberately leaves alone.
    if is_appliance_question(text):
        return False
    if _KNOWLEDGE_REQUEST.search(text):
        return True
    return _KNOWLEDGE_OPENER.match(text.strip()) is not None


#: A coordinator joining two questions, and a *world* sub-question beside an
#: appliance one.
#:
#: **Live on the Pi #205 deep pass.** "What's the CPU temp AND the capital of
#: France?" answered the temperature and dropped the world half in silence. The
#: appliance part is answered by the reading path; this adds the honest pointer
#: for the part this appliance holds no evidence for, rather than pretending the
#: question had one half.
#:
#: **The marker match alone was not enough (#205 M3).** `tell me about`, `when
#: was`, `who is` and `how old` match ordinary appliance phrasings - "tell me
#: about my cpu and disk usage", "how is my disk and when was the last backup",
#: "who is running the most memory", "how old is this data" - and a spurious
#: "ask AI Chat" note sent the owner elsewhere for something the appliance owns.
#: So the marker must fall inside a *coordinated clause that is itself
#: world-scoped*: no appliance signal (`is_appliance_question`) and no pointer
#: at this machine (`_THIS_MACHINE`) in that clause.
_COMPOUND_COORDINATOR = re.compile(r"\b(?:and|also|plus)\b|[;,]", re.IGNORECASE)
_WORLD_SUBQUESTION = re.compile(
    r"\bcapital\s+of\b|\bcapital\s+city\b|\bwho\s+won\b|\bwhat\s+year\b|"
    r"\brecipe\s+for\b|\btell\s+me\s+about\b|\bpopulation\s+of\b|\btranslate\b|"
    r"\bwho\s+(?:is|was|were|are)\b|\bwhen\s+(?:was|were|did)\b|"
    r"\bhow\s+(?:tall|far|old|big|many\s+people)\b",
    re.IGNORECASE,
)


def world_followup(message: str) -> Optional[Dict[str, str]]:
    """A pointer to AI Chat for the world half of a compound question.

    Returns ``None`` unless one coordinated clause is *genuinely* a world
    question - carries a world marker and names neither an appliance thing nor
    this machine. A caller may consult it on every answered appliance reply and
    only ever adds the note where a world part was really present (#205 item 3,
    tightened for M3).
    """
    text = str(message or "")
    if not _COMPOUND_COORDINATOR.search(text):
        return None
    for clause in _COMPOUND_COORDINATOR.split(text):
        clause = clause.strip()
        if not _WORLD_SUBQUESTION.search(clause):
            continue
        # An appliance clause that merely happens to contain a marker word
        # ("when was the last backup", "how old is this data") is not a world
        # question - the appliance owns it. Only a clause with no appliance or
        # self reference earns the pointer.
        if is_appliance_question(clause) or _THIS_MACHINE.search(clause):
            continue
        # **Worded not to read as a subject refusal.** `assistant_scope_guard`
        # withholds any sentence that declines an appliance question ("outside
        # what I cover", "AI Chat is the place to ask"); this note rides beside
        # an answered appliance reading, so a refusal phrasing would make the
        # guard withhold it and mark the whole reply unanswered. It is a forward
        # pointer, like `_PARTLY_WITHHELD_NOTE` in that module.
        return {
            "note": (
                "The other part of your question is a general-knowledge one, "
                "which AI Chat can answer - it is this appliance's "
                "general-question surface."
            ),
            "action": "Open AI Chat for the general-knowledge part.",
        }
    return None


#: Whether AI Chat can actually take the question. ``UNKNOWN`` is a third
#: state and stays one: reporting an unread connection list as "no connection"
#: would assert a machine fact from a default, which is the defect
#: :mod:`vaelor.answer_evidence` exists to prevent.
CHAT_READY = "ready"
CHAT_ABSENT = "absent"
CHAT_UNKNOWN = "unknown"


def chat_destination(
    connections: Optional[Sequence[Mapping[str, Any]]],
) -> Dict[str, str]:
    """Summarise AI Chat's readiness from what was actually read.

    ``None`` means the list could not be read at all, which is not the same
    claim as an empty list and is not rendered as one.
    """
    if connections is None:
        return {"state": CHAT_UNKNOWN, "label": ""}
    active = [
        item for item in connections
        if isinstance(item, Mapping) and "ai-chat" in (item.get("active_for") or ())
    ]
    if not active:
        return {"state": CHAT_ABSENT, "label": ""}
    return {"state": CHAT_READY, "label": str(active[0].get("label") or "")}


_SCOPE_SENTENCE = (
    "I answer questions about this appliance - its hardware, telemetry, "
    "services, configuration, and logs - and nothing I can read on this machine "
    "covers that question, so any answer from me would be a guess."
)

# Every branch names "AI Chat", which `navigation_steps` turns into the real
# #/ai-chat route, so the destination is a link rather than a sentence. The
# absent branch says what setting it up involves, because sending someone to a
# surface with no connection is the same dead end in a different place.
_CHAT_SENTENCES = {
    CHAT_READY: (
        "AI Chat is where to ask it: it runs the model you chose and is this "
        "appliance's general-question surface."
    ),
    CHAT_ABSENT: (
        "AI Chat is where that question belongs, but no AI Chat connection is "
        "active on this appliance yet. Setting one up means adding a model "
        "connection on AI Chat - either install a local model on this machine "
        "or add an external provider key - and then selecting which model to "
        "use."
    ),
    CHAT_UNKNOWN: (
        "AI Chat is where that question belongs. I could not read whether an "
        "AI Chat connection is active on this appliance, so open AI Chat to "
        "check, and add a local model or an external provider there if it has "
        "none."
    ),
}

_CHAT_ACTIONS = {
    CHAT_READY: "Open AI Chat and ask it there.",
    CHAT_ABSENT: (
        "Open AI Chat to add a local model or an external provider connection."
    ),
    CHAT_UNKNOWN: "Open AI Chat to check whether a model connection is set up.",
}


def knowledge_redirect(
    message: str, ai_chat: Optional[Mapping[str, Any]] = None
) -> Optional[Dict[str, Any]]:
    """An honest redirect for a question this machine holds no evidence for.

    Returns ``None`` for everything else, so a caller that consults it
    unconditionally still answers every appliance question it always did.
    """
    if not is_out_of_scope_question(message):
        return None
    destination = dict(ai_chat or {})
    state = str(destination.get("state") or CHAT_UNKNOWN)
    if state not in _CHAT_SENTENCES:
        state = CHAT_UNKNOWN
    if state == CHAT_READY:
        summary = first_sentence(
            sentence(
                "Read from this machine: the active AI Chat connection is {label}.",
                label=destination.get("label", ""),
            ),
            "Read from this machine: an AI Chat connection is active.",
        )
    elif state == CHAT_ABSENT:
        summary = "Read from this machine: no AI Chat connection is active."
    else:
        summary = describe_missing("the AI Chat connection list")
    return {
        "answer": "{} {}".format(_SCOPE_SENTENCE, _CHAT_SENTENCES[state]),
        "evidence": [
            {
                "source": "assistant.scope",
                "summary": (
                    "No appliance fact tool covers this question, so Vaelor "
                    "answered nothing about this machine."
                ),
            },
            {"source": "ai-chat.connections", "summary": summary},
        ],
        "suggested_actions": [_CHAT_ACTIONS[state]],
        "proposed_job": None,
    }
