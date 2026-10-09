"""Which model AI Chat, the LLM Server and the Assistant are using, answered from readings.

**VD-205 live check L1.** On the Z2 in Mode B, "Which model does AI Chat use?"
and "Is the LLM Server running?" reached the on-device model, which answered
with a bare list of engine and connection ids. Neither answer needs a model:
the serving-mode record, the cluster's deployment record, the AI Chat lease and
the LLM Server's own reading say it, so the answer is built here and the model
is never asked (a wrong answer repeated by two models is a code path, not a
model gap).

Three parts:

* :func:`asks_about_serving` - whether a question asks *which model* one of
  the three uses, or whether it is up. It is deliberately narrow (review round
  2, B2): a reading, a model choice, the model inventory, a change request, a
  how-to or a definition that merely names AI Chat or the LLM Server keeps the
  answer that owns it.
* :func:`serving_reading` - the ``serving.status`` fact, gathered only for
  such a question (review S2: the LLM Server's gate probe and the model checks
  cost seconds when something hangs). The LLM Server comes from
  `api_llm_server_routes.llm_server_reading`, the one reader its console card
  uses (LESSONS 6). No key, fingerprint or key count is carried.
* :func:`serving_answer` - the sentences, each subject only when asked, with
  the sources read as evidence. A state is stated, never put behind a "Yes" or
  "No" that answers the state rather than the question (review B1). A reading
  that did not arrive is said as not read, never as "off" (LESSONS 8).

FOLLOWUP VD205-b is folded in: "Is the Assistant running on the NPU?" is
answered from the Assistant's own lease, identified the way the NPU supervisor
identifies it (an FLM model tag), and "served" only once that model answered.
"""

from __future__ import annotations

import logging
import re
from typing import Any, Dict, List, Mapping, Optional

from .assistant_builtin_answer import ASSISTANT_BASIC_MODE, chat_connection_words
from .assistant_cluster_answers import SPEED_WORDS
from .assistant_console_places import AI_CHAT, CONNECTIONS_PLACE, ENDPOINTS_PLACE, LLM_SERVER
from .assistant_fact_access import usable_fact
from .phrase_match import mentions

LOGGER = logging.getLogger(__name__)

#: The source this answer is recorded under; reading-backed, so it outranks the model.
SERVING_SOURCE = "built-in-serving"
#: The fact this answer reads, gathered only for a question it answers.
SERVING_TOOL = "serving.status"

# --- Which questions are about serving ------------------------------------------------

#: The three subjects, by the names the console gives them.
_AI_CHAT = re.compile(r"\bai[\s-]?chat\b")
_LLM_SERVER = re.compile(r"\b(?:llm[\s-]?server|llm\s+endpoint|(?:port\s+)?11434)\b")
_ASSISTANT = "Assistant"
#: "Which model does AI Chat use", "what LLM powers it", "what model are you".
_WHICH_MODEL = re.compile(r"\b(?:which|what)\s+(?:model|llm)\b")
#: "Is AI Chat up", "does it work": a yes/no question about whether the thing
#: is up. **The state word ends the question** (review round 3, N-S2): "Does
#: AI Chat work offline?" and "Is AI Chat open source?" ask about a capability.
_STATE_ASK = re.compile(
    r"^(?:is|are|does|can)\b[^?]*\b(?:up|down|on|off|running|working|work|serving|enabled|"
    r"disabled|open|closed|reachable|listening)(?:\s+(?:right\s+)?now|\s+from\s+my\s+network)?\s*\??$")
#: "Can my LAN apps reach the LLM Server": whether it can be reached.
_REACH = re.compile(r"^can\b[^?]*\breach\b")
#: "What does AI Chat use", "what's AI Chat running", "what is it running on".
#: The subject comes before the verb (review round 3, N-S2): "What uses the
#: LLM Server?" asks which clients use it.
_WHAT_RUNS = re.compile(r"\bwhat(?:'s|\s+is|\s+does)\b[^?]*\b(?:runs?|running|uses?|serves?)\b")
#: "What runs AI Chat?": the feature as the object (review round 4).
_WHAT_SERVES = re.compile(r"^what\s+(?:runs|serves|powers)\b")
#: The Assistant itself as the thing running: "is the Assistant running on the
#: NPU", "are you on the NPU". A device named as the object ("Assistant, is
#: the GPU working") is a reading question and does not match.
_ASSISTANT_STATE = re.compile(r"\b(?:assistant|you)\s+(?:running|on|served|serving)\b")
_YOU = re.compile(r"\bare\s+you\b")

#: Questions that name these features but ask something another answer owns
#: (review round 2, B2). Each family is pinned as a negative case in
#: `tests/test_assistant_vd205_live.py`.
_NOT_SERVING_WORDS = (
    # a reading of the hardware the model runs on
    "temperature", "temp", "hot", "warm", "heat", "memory", "ram", "vram", "gtt", "utilisation",
    "utilization", "busy", "usage", "load", "power", "watts", "fan", "fans", "disk", "storage",
    # choosing a model, which the model-selection guidance owns
    "should", "best", "recommend", "recommended", "better",
    # the model inventory
    "installed", "download", "downloads", "list",
    # a change request, which the decline path answers with "nothing was changed"
    "turn", "switch", "change", "enable", "disable", "start", "stop", "restart", "set", "install",
    "create", "connect", "load", "unload",
    # how to do something, and what something is
    "how do i", "how can i", "how to", "where do i", "difference", "explain", "why",
    # the keys and the exposure, which are not a running state
    "key", "keys", "fingerprint", "token", "tokens", "secure", "internet",
    # speed, which the cluster and accelerator answers own
    "slow", "slower", "sluggish", "lag", "laggy", "bottleneck",
    # a capability or a version, not whether it is up (review round 3, N-S2)
    "with", "offline", "source", "for", "need", "needs", "version", "update", "latest",
) + SPEED_WORDS


def _subjects(message: str) -> List[str]:
    from .assistant_intents import is_definition_request

    text = " ".join(str(message or "").split())
    lower = text.lower()
    if is_definition_request(text) or mentions(lower, _NOT_SERVING_WORDS):
        return []
    which, state, runs = (bool(pattern.search(lower)) for pattern in (_WHICH_MODEL, _STATE_ASK, _WHAT_RUNS))
    state = state or bool(_REACH.search(lower))
    runs = runs or bool(_WHAT_SERVES.search(lower))
    subjects = []
    if _AI_CHAT.search(lower) and (which or state or runs):
        subjects.append("ai-chat")
    if _LLM_SERVER.search(lower) and (which or state or runs or mentions(lower, ("port",))):
        subjects.append("llm-server")
    assistant = mentions(lower, (_ASSISTANT.lower(),)) or bool(_YOU.search(lower))
    # "Are you serving AI Chat too?" names a feature: it is not about the Assistant's own model.
    feature = bool(_AI_CHAT.search(lower) or _LLM_SERVER.search(lower))
    if assistant and not feature and (which or _ASSISTANT_STATE.search(lower)):
        subjects.append("assistant")
    return subjects


def asks_about_serving(message: str) -> bool:
    """Whether the question asks which model AI Chat, the LLM Server or the Assistant uses, or if it is up."""
    return bool(_subjects(message))


# --- The reading -----------------------------------------------------------------------

_SHAPES = {"replicated": "one copy on each of {} machines", "distributed": "split across {} machines"}


def _cluster_record(callbacks: Mapping[str, Any], state: Any) -> Dict[str, Any]:
    """The Mode B deployment that serves AI Chat, by the names the Fleet screen uses."""
    from .cluster_placement import CONTROLLER_PLACEMENT_ID, CONTROLLER_PLACEMENT_NAME
    from .performance_snapshot_source import worker_nodes

    manager = callbacks.get("cluster_manager")
    name = str(getattr(state, "deployment_name", "") or "")
    if manager is None or not name:
        return {"read": False}
    try:
        record = next((row for row in manager.store.list_pooled_deployments()
                       if isinstance(row, Mapping) and row.get("name") == name), None)
        names = {CONTROLLER_PLACEMENT_ID: CONTROLLER_PLACEMENT_NAME,
                 **{str(worker["id"]): str(worker.get("name") or "") for worker in worker_nodes(dict(callbacks))}}
    except Exception:  # noqa: BLE001 - an unread record is said as not read
        LOGGER.exception("the Assistant could not read the cluster deployment serving AI Chat")
        return {"read": False}
    if record is None:
        return {"read": False}
    units = record.get("units") if isinstance(record.get("units"), Mapping) else {}
    # Review S1: a forced removal or the split health watch marks a lost
    # machine here while the record's state stays as it was.
    lost = {str(entry.get("node_id")) for entry in units.get("lost_nodes") or [] if isinstance(entry, Mapping)}
    return {
        "read": True, "deployment": name, "model": str(record.get("model_id") or ""),
        "state": str(record.get("state") or ""), "shape": str(units.get("mode") or ""),
        "machines": [names.get(str(node), str(node)) for node in record.get("node_ids") or []
                     if str(node) not in lost],
        "lost": [names.get(node, node) for node in sorted(lost)],
        "degraded": str(units.get("degraded_reason") or ""),
    }


def _answers(connection: Optional[Mapping[str, Any]], what: str) -> Optional[bool]:
    """Whether a model connection answered a check; ``None`` when none was made."""
    from .model_reachability import probe_connection

    if not connection:
        return None
    try:
        return bool(probe_connection(dict(connection)).get("reachable"))
    except Exception:  # noqa: BLE001 - an unanswered check is "not checked", never "down"
        LOGGER.exception("the Assistant could not check %s", what)
        return None


def _ai_chat_answering(callbacks: Mapping[str, Any]) -> Optional[bool]:
    chat = callbacks.get("chat_inference")
    try:
        connection = chat.active_local_connection() if chat is not None else None
    except Exception:  # noqa: BLE001 - said as not checked
        LOGGER.exception("the Assistant could not resolve AI Chat's model")
        return None
    return _answers(connection, "AI Chat's model")


def _assistant(callbacks: Mapping[str, Any]) -> Dict[str, Any]:
    """The Assistant's own lease: its pinned model, and whether it answered."""
    agent = callbacks.get("deployment_agent")
    try:
        connection = agent.connection("local") if agent is not None else None
    except Exception:  # noqa: BLE001 - said as not read
        LOGGER.exception("the Assistant could not read its own model lease")
        return {"read": False}
    if not connection:
        return {"read": True, "model": "", "configured": False, "answering": None}
    from .deployment_agent import AUTO_DETECT_MODEL

    # Review round 3 NIT: the auto-detect placeholder is no pinned model.
    model = str(connection.get("model") or "")
    return {"read": True, "model": "" if model == AUTO_DETECT_MODEL else model, "configured": True,
            "answering": _answers(connection, "its own model lease")}


def _llm_server(callbacks: Mapping[str, Any]) -> Dict[str, Any]:
    """The LLM Server's reading, from the card's own reader; keys never leave it."""
    from .api_llm_server_routes import llm_server_reading
    from .llm_server_state import LlmServerStore

    try:
        reading = llm_server_reading(
            store=callbacks.get("llm_server_store") or LlmServerStore(),
            broker=getattr(callbacks.get("chat_inference"), "broker", None),
            manager=callbacks.get("cluster_manager"),
            bridge=callbacks.get("hardware_bridge_client"),
        )
    except Exception:  # noqa: BLE001 - logged with its traceback, said as not read
        LOGGER.exception("the Assistant could not read the LLM Server")
        return {"read": False}
    return {
        "read": True, "enabled": bool(reading["settings"].enabled),
        "state": str(reading["runtime"].get("state") or ""),
        "detail": str(reading["runtime"].get("detail") or ""),
        "model": str(reading["model"] or ""), "kind": str(reading["target"].kind or ""),
    }


def serving_reading(callbacks: Mapping[str, Any]) -> Dict[str, Any]:
    """The ``serving.status`` fact: mode, cluster record, model checks, LLM Server."""
    from .assistant_cluster_digest import serving_mode_state
    from .gpu_serving_target import gpu_cluster_mode_active

    state = serving_mode_state(callbacks)
    mode = None if state is None else ("B" if gpu_cluster_mode_active(state) else "A")
    return {
        "mode": mode,
        "cluster": _cluster_record(callbacks, state) if mode == "B" else None,
        "ai_chat_answering": _ai_chat_answering(callbacks) if mode != "B" else None,
        "assistant": _assistant(callbacks),
        "llm_server": _llm_server(callbacks),
    }


#: What the tool says it reads, for the tool catalogue.
SERVING_DESCRIPTION = (
    "Read what serves AI Chat (the serving mode and, in cluster mode, the deployment record), "
    "whether AI Chat's and the Assistant's own models answer a check, and the LLM Server's "
    "state as its console card reads it."
)

# --- The sentences ---------------------------------------------------------------------


def _listed(names: List[str]) -> str:
    from .assistant_cluster_digest import name_in_sentence

    names = [name_in_sentence(name) for name in names]
    return names[0] if len(names) == 1 else "{} and {}".format(", ".join(names[:-1]), names[-1])


def _cluster_line(cluster: Mapping[str, Any]) -> str:
    if not cluster.get("read"):
        return ("AI Chat is served by the cluster, but its deployment record could not be "
                "read, so which model it serves is not known.")
    machines = list(cluster.get("machines") or [])
    if cluster.get("degraded") or cluster.get("lost"):
        where = "now on {}".format(_listed(machines)) if machines else "on no machine still in service"
    else:
        spread = _SHAPES.get(cluster.get("shape"), "on {} machines").format(len(machines))
        where = "{} ({})".format(spread, _listed(machines)) if machines else "on no recorded machine"
    model = cluster.get("model") or "a model the record does not name"
    state = cluster.get("state")
    if state == "healthy":
        lead = "AI Chat is served by the cluster model {}, {}.".format(model, where)
    elif state == "unloaded":
        lead = "AI Chat is set to the cluster model {}, {}, which is unloaded right now.".format(model, where)
    elif state == "deploying":
        lead = "AI Chat is set to the cluster model {}, {}, which is still starting.".format(model, where)
    else:
        lead = "AI Chat is set to the cluster model {}, {}; its deployment is recorded as {}.".format(
            model, where, state or "in no known state")
    if cluster.get("degraded"):
        lead += " The deployment is degraded: {}".format(cluster["degraded"])
    elif cluster.get("lost"):
        lead += " It has lost {}.".format(_listed(list(cluster["lost"])))
    return lead + " This controller's own AI Chat model is stopped while the cluster serves AI Chat."


def _single_line(inference: Mapping[str, Any], answering: Optional[bool]) -> str:
    chat = inference.get("chat") if isinstance(inference.get("chat"), Mapping) else {}
    if not chat or chat.get("reason"):
        return "Which connection AI Chat uses could not be read."
    active = [item for item in chat.get("connections") or [] if isinstance(item, Mapping) and item.get("active")]
    if not active:
        return "AI Chat has no model connected. Choose one in {}, or add one under {}.".format(
            AI_CHAT, CONNECTIONS_PLACE)
    check = {True: "; it answered a check just now", False: "; it did not answer a check just now"}.get(
        answering, "; whether it is answering was not checked")
    return "AI Chat uses {}{}.".format(chat_connection_words(active[0]), check)


def _ai_chat_line(inference: Mapping[str, Any], serving: Mapping[str, Any]) -> str:
    if serving.get("mode") == "B":
        return _cluster_line(serving.get("cluster") or {})
    lead = "" if serving.get("mode") == "A" else "Whether the cluster serves AI Chat was not read. "
    return lead + _single_line(inference, serving.get("ai_chat_answering"))


#: The LLM Server's state, stated (review B1: never a "Yes"/"No" that answers
#: the state instead of the question). ``{port}`` is the server's LAN port.
_LLM_SERVER_STATES = {
    "serving": "The LLM Server is running on port {port}.",
    "off": "The LLM Server is turned off. Turn it on with Enable in {place}.",
    "no-keys": ("The LLM Server is turned on but has no API key, so its port ({port}) is closed. "
                "Create a key in {place} to open it."),
    "still-open": ("The LLM Server is turned off, but its port ({port}) still answers: the change "
                   "has not reached its proxy yet."),
    "paused": "The LLM Server is paused.",
    "starting": "The LLM Server is starting.",
    "applying-keys": "The LLM Server is running on port {port}, but it is still applying a change to its API keys.",
    "unknown": "Whether the LLM Server is running is not known.",
}
#: Runtime sentences written for the console card, which an answer does not repeat.
_CARD_ONLY_STATES = ("serving", "off", "no-keys", "still-open")


def _llm_server_line(server: Mapping[str, Any]) -> str:
    from .llm_server_proxy import LLM_SERVER_PROXY_PORT

    if not server.get("read"):
        return "Whether the LLM Server is running was not read."
    state, detail = server.get("state"), str(server.get("detail") or "")
    lead = _LLM_SERVER_STATES.get(
        state, "The LLM Server is turned on but is not serving.").format(
            port=LLM_SERVER_PROXY_PORT, place=ENDPOINTS_PLACE)
    if state == "serving":
        lead += " Clients on your LAN that hold an LLM Server key reach {}{} through it.".format(
            server.get("model") or "the model it fronts",
            ", by way of the cluster's load balancer," if server.get("kind") == "cluster" else "")
    return lead if state in _CARD_ONLY_STATES or not detail else "{} {}".format(lead, detail)


def _assistant_line(assistant: Mapping[str, Any]) -> str:
    from .managed_local_credentials import pins_an_flm_tag

    if not assistant.get("read"):
        return "Which model the Assistant runs was not read."
    model = str(assistant.get("model") or "")
    if not assistant.get("configured"):
        return ASSISTANT_BASIC_MODE
    if not model:
        return ("The Assistant has Vaelor's own model connection set up, but no model is pinned on it, "
                "so which model it runs is not known.")
    answering = assistant.get("answering")
    if pins_an_flm_tag(model):
        if answering:
            return "The Assistant runs on Vaelor's own model, {}, served by FastFlowLM on the neural processor.".format(
                model)
        return ("The Assistant is configured to run Vaelor's own model, {}, on the neural processor "
                "(FastFlowLM), but {}.").format(
                    model, "it did not answer a check just now" if answering is False
                    else "whether it is answering was not checked")
    return ("The Assistant runs on Vaelor's own model, {}. It is not a neural-processor (FastFlowLM) "
            "model, so the neural processor is not serving the Assistant.").format(model)


def serving_answer(message: str, facts: Mapping[str, Any]) -> Optional[Dict[str, Any]]:
    """The deterministic answer to an AI Chat, LLM Server or Assistant model question, or ``None``."""
    subjects = _subjects(message)
    if not subjects:
        return None
    inference = usable_fact(facts, "inference.status")
    serving = usable_fact(facts, SERVING_TOOL)
    if inference is None or serving is None:
        named = [{"ai-chat": AI_CHAT, "llm-server": LLM_SERVER, "assistant": _ASSISTANT}[subject]
                 for subject in subjects]
        return {"answer": "What {} is running was not read, so I cannot say.".format(" or ".join(named)),
                "evidence": [], "suggested_actions": [], "proposed_job": None, "answered": False}
    lines: List[str] = []
    read: List[bool] = []
    evidence = [{"source": "inference.status",
                 "summary": "Read which model the Assistant runs and which connection AI Chat uses."}]
    if "ai-chat" in subjects:
        lines.append(_ai_chat_line(inference, serving))
        chat = inference.get("chat") if isinstance(inference.get("chat"), Mapping) else {}
        read.append(bool((serving.get("cluster") or {}).get("read")) if serving.get("mode") == "B"
                    else bool(chat) and not chat.get("reason"))
        evidence.append({"source": "cluster.serving-mode",
                         "summary": "Read the serving-mode record and, in cluster mode, the deployment "
                                    "record of the model that serves AI Chat."})
    if "llm-server" in subjects:
        server = serving.get("llm_server") or {}
        lines.append(_llm_server_line(server))
        read.append(bool(server.get("read")) and server.get("state") != "unknown")
        evidence.append({"source": "llm-server.status",
                         "summary": "Read the LLM Server's setting, whether it holds an API key (no key "
                                    "is shown) and its proxy, as its console card does."})
    if "assistant" in subjects:
        assistant = serving.get("assistant") or {}
        lines.append(_assistant_line(assistant))
        read.append(bool(assistant.get("read")))
    return {"answer": " ".join(lines), "evidence": evidence, "suggested_actions": [],
            "proposed_job": None, "answered": any(read)}
