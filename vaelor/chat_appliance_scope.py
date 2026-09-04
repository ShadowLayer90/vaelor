"""AI Chat declines a question about this machine and names the Assistant.

VD-042's boundary runs both ways, and this is the direction nothing enforced.
The Assistant is stopped from pushing appliance questions outward; nothing
stopped AI Chat from answering one inward. A connected frontier provider asked
*"how hot is my CPU"* will answer it - from nothing - and the better the model
the more convincing the fabrication. Only AI Chat's page descriptor said it
could not see this machine, and a descriptor is not a check.

That makes this the more dangerous half of the boundary. #72's failure produced
a refusal, which is annoying and visible. This one produces a confident number
with no reading behind it, which is indistinguishable from a real answer.

The threshold, and what it rests on
-----------------------------------

The trigger is deliberately *not* :func:`is_appliance_question` alone. That
predicate exists to **suppress** the Assistant's outward redirect, where a
false positive costs nothing - it just answers a question it would have
answered anyway. Here a false positive **blocks a legitimate general question**,
which is the whole purpose of the surface. The two jobs need different
thresholds and reusing one for the other would have been the whole defect.

So the rule was measured rather than guessed, against the phrase corpora the
test suite already carries - collected from live-test transcripts across
:mod:`tests.test_assistant_intents` and :mod:`tests.test_deployment_agent`, not
written for this gate. As measured over 152 recorded phrases:

* **62 of 62** recorded general-knowledge, RAG and conversational phrases pass
  through untouched. No false decline, including the ones that name appliance
  hardware in a general sense - *"what is a docker container"*, *"how do i
  install docker on ubuntu"*, *"what is the temperature on Mars"*, and AI
  Chat's own canonical retrieval question *"what is the fan threshold?"*.
* **34 of 90** recorded appliance phrasings are declined, including **every**
  phrasing recorded from a live test.

The recall gap is real and is accepted deliberately: the misses are bare
definite-article forms - *"are the lights blue?"*, *"are there any alerts"* -
which on a general-chat surface are genuinely ambiguous, and which AI Chat's
page descriptor still covers. A variant that caught them (a state-predicate
branch) reached 48 of 90 but started declining *"how do i install docker on
ubuntu"*, and breaking general chat to catch an ambiguous phrasing is the
trade the #72 refusal already showed to be wrong. **Precision is the property
being defended here; recall is not.** Do not widen the trigger without
re-running that measurement - the tests below pin both directions.
"""

from __future__ import annotations

import re
from typing import Any, Dict, Optional

from .assistant_action_requests import detect_action_requests
from .assistant_answer_presentation import (
    is_appliance_question,
    is_general_knowledge_question,
)

#: Which surface answered. Not a model, and it must not be reported as one.
APPLIANCE_SCOPE_PROVIDER = "appliance-scope"

#: Where the question actually belongs. The frontend renders the route.
ASSISTANT_ROUTE = "#/assistant"

#: A phrase binding the question to *this* machine rather than to the subject
#: in general.
#:
#: This is the whole difference between "how hot is my CPU" and "how hot does a
#: CPU run". A bare possessive is safe here in a way it is not on the
#: Assistant's own gate - "draft an email to my landlord" is not an appliance
#: question in the first place, so it never reaches this test. The deictic is
#: also what makes a model most likely to answer as though it had access: a
#: question phrased as though the reader can see the box invites a reading, and
#: a capable model supplies one.
_THIS_MACHINE = re.compile(
    r"\b(?:my|our)\b"
    r"|\bthis\s+(?:appliance|machine|box|thing|unit|node|device|server|system|pi|host)\b"
    # "the current appliance state", "the spare node" - up to two words
    # between the article and the noun, because people qualify the noun.
    r"|\bthe\s+(?:\w+\s+){0,2}(?:appliance|box|unit|node)\b"
    r"|\bon\s+(?:this|here)\b"
    r"|\b(?:right\s+now|currently|at\s+the\s+moment|in\s+here)\b"
    r"|\bvaelor\b|\bpironman\b",
    re.IGNORECASE,
)

#: The product's own names. Nothing in the world but this appliance is called
#: either, so a question naming one has no general reading to be answered from
#: - and a provider asked to explain a term absent from its training data does
#: not decline, it invents. That makes a *definitional* question about Vaelor
#: the purest form of the fabrication this gate exists to prevent, which is why
#: it is admitted before the definitional test below rather than after it.
_PRODUCT_NAME = re.compile(r"\b(?:vaelor|pironman)\b", re.IGNORECASE)

_DECLINE_ANSWER = (
    "That is a question about this appliance, and AI Chat cannot read this "
    "machine - it has no sensors, no service state, and no job history in "
    "front of it, so any figure it gave you would be invented rather than "
    "measured. The Assistant is the surface that reads this machine: ask it "
    "there and it will answer from live readings. AI Chat is here for your own "
    "documents and for general questions."
)


#: Words that carry no topic, so a retrieved passage sharing only these with
#: the question is not evidence the documents answer it. Retrieval matches on
#: any indexed term - the FTS index keeps function words - so "what is the CPU
#: temperature" matches a note about the mascot on "the"/"is" alone. Counting
#: that as relevant would wave a telemetry question past the gate to a provider
#: that then invents a reading, which is the #247o adversarial finding. This is
#: a stop-list, not a score threshold: the store keeps no usable relevance
#: cutoff (bm25 is degenerate for a single-chunk collection), so relevance is
#: decided by content-word overlap instead.
#:
#: The final line is the appliance-category deictic nouns, added after re-review.
#: A telemetry question and an ops runbook almost always share one of these, so
#: "what is the appliance CPU temperature right now" overlapped a mascot note on
#: "appliance" alone and was deemed relevant - letting the model invent a
#: reading from a passage with no sensor data, the very VD-042 fabrication this
#: gate exists to stop. A category noun locates the machine but says nothing
#: about the sensor asked for, so sharing only it is not grounding. Sensor and
#: topic nouns (cpu, fan, disk, temperature, memory, gpu...) are deliberately
#: NOT here: a runbook that really mentions CPU temperature must still ground a
#: CPU-temperature question.
_RELEVANCE_STOPWORDS = frozenset({
    "the", "is", "are", "was", "were", "be", "been", "being", "am", "an", "of",
    "to", "in", "on", "at", "for", "and", "or", "not", "no", "my", "our", "us",
    "this", "that", "these", "those", "it", "its", "what", "which", "who",
    "whom", "how", "when", "where", "why", "do", "does", "did", "done", "can",
    "could", "would", "should", "will", "shall", "may", "might", "must", "if",
    "so", "as", "by", "with", "from", "into", "about", "i", "you", "we", "they",
    "he", "she", "him", "her", "them", "here", "there", "right", "now",
    "currently", "please", "tell", "show", "give", "say", "said", "have", "has",
    "had", "get", "got", "any", "all", "some", "state", "status", "value",
    "appliance", "machine", "box", "device", "unit", "node", "server",
    "system", "host", "pi", "thing",
})


def _content_tokens(text: Any) -> set:
    """The topic-bearing words of ``text``, plural-normalised and lowercased.

    Mirrors the store's ``[A-Za-z0-9]{2,}`` split so the two sides compare on
    the same units, drops the stop-list above, and strips a trailing plural
    ``s`` so "fans" and "fan" meet - a light stand-in for the index's porter
    stemmer, enough that a document written in the singular still answers a
    question asked in the plural.
    """
    tokens = re.findall(r"[a-z0-9]{2,}", str(text or "").lower())
    return {
        token[:-1] if token.endswith("s") and len(token) > 3 else token
        for token in tokens
        if token not in _RELEVANCE_STOPWORDS
    }


def retrieval_answers_question(message: Any, retrieved: Any) -> bool:
    """Whether any retrieved passage shares a topic word with ``message``.

    The retrieve-first pre-emption (#247o) turns on this: when a knowledge
    collection is active the question runs through retrieval first, and the
    appliance-scope gate declines only what the documents could not answer. A
    non-empty result is not that evidence on its own, because retrieval matches
    on function words too; relevance means a real content-word overlap. A
    telemetry question the documents do not cover has none, so it falls back to
    the deflection rather than reaching a provider that would fabricate.
    """
    wanted = _content_tokens(message)
    if not wanted:
        return False
    for passage in retrieved or []:
        content = passage.get("content", "") if isinstance(passage, dict) else ""
        if wanted & _content_tokens(content):
            return True
    return False


def asks_about_this_machine(message: Any) -> bool:
    """Whether ``message`` asks AI Chat for the state of the machine it runs on.

    Three conditions, all required:

    1. it names something this appliance has;
    2. it is not a definitional or general-knowledge question about that thing;
    3. it points at *this* machine, or instructs it to change.

    Condition 3 is what a topic-word match alone cannot supply, and dropping it
    would decline half of ordinary general chat. Naming the product short-
    circuits condition 2, because a definitional question about Vaelor has no
    general answer for AI Chat to give.
    """
    text = str(message or "")
    if not is_appliance_question(text):
        return False
    if _PRODUCT_NAME.search(text):
        return True
    if is_general_knowledge_question(text):
        return False
    if detect_action_requests(text):
        # An instruction to change this machine is unambiguous, and a model
        # that replies "done, the lights are purple" has fabricated an action
        # rather than a reading. Worse, not better.
        return True
    return _THIS_MACHINE.search(text) is not None


def appliance_scope_decline(message: Any) -> Optional[Dict[str, Any]]:
    """The reply AI Chat gives instead of answering an appliance question.

    ``None`` means answer normally, so a caller that consults this on every
    request still answers everything AI Chat has always answered.
    """
    if not asks_about_this_machine(message):
        return None
    return {
        "answer": _DECLINE_ANSWER,
        "metadata": {
            "source": APPLIANCE_SCOPE_PROVIDER,
            "declined": True,
            # Named so the client can offer a link rather than asking the
            # reader to go and find the surface. "Ask the Assistant" with no
            # route is the dead end the Assistant's own redirect was fixed for;
            # reproducing it here would be the same defect facing the other way.
            "route": ASSISTANT_ROUTE,
            "reason": "appliance_question",
        },
    }


def record_decline(
    store: Any, actor: str, decline: Dict[str, Any], message: str, *,
    model: str = "", temporary: bool = False,
    conversation: Optional[Dict[str, Any]] = None,
    collections: Optional[list] = None,
) -> Dict[str, Any]:
    """Persist the declined turn and return the route's response body.

    A decline is a turn that happened, so on a saved conversation both halves
    are written: the question the user asked and the answer they were given.
    Dropping the exchange would leave a chat whose history skips the moment the
    boundary was applied, and the reader would see their own message vanish.

    ``model`` is reported unchanged - it is the model the user has selected,
    and this turn does not change that - while ``provider`` says plainly that
    no model produced this reply.
    """
    if temporary:
        return {
            "conversation_id": "",
            "message": {
                "role": "assistant", "content": decline["answer"],
                "citations": [], "metadata": decline["metadata"],
            },
            "model": model,
            "provider": APPLIANCE_SCOPE_PROVIDER,
            "temporary": True,
            "declined": True,
        }
    if conversation is None:
        conversation = store.ensure_conversation(
            actor, title=str(message)[:100], model=model,
            collections=list(collections or []),
        )
    recorded = store.add_exchange(
        actor, conversation["id"], message, decline["answer"], [], None,
        APPLIANCE_SCOPE_PROVIDER,
    )
    recorded["metadata"] = decline["metadata"]
    return {
        "conversation_id": conversation["id"],
        "message": recorded,
        "model": model,
        "provider": APPLIANCE_SCOPE_PROVIDER,
        "declined": True,
    }
