"""AI Chat's Web search: the control's route, and one turn's search (VD-221).

- ``GET /ai-chat/web-search`` says whether the toggle can be offered, read
  without searching (`chat_web_search.web_search_availability`), and whether
  this account may turn it on.
- :func:`web_search_turn` is what `api_chat_routes` asks on every send and
  regenerate: either a reply that stands instead of a guess, or the web pages to
  add to the answer's sources.

**The toggle is the request's own ``web_search: true`` and nothing else.** The
conversation remembers the choice so the screen can show it again, but a send
that does not carry it searches nothing, so no stored flag, default or retry
can send a question out (the owner's rule: nothing leaves the appliance for a
search unless the toggle is on).

**Only an administrator may turn it on (owner, VD-221 item 14).** The flag on a
send and the conversation patch are both refused for any other role here, on
the server: the console's disabled toggle is a courtesy, not the rule.
"""

from __future__ import annotations

import sqlite3
from typing import Any, Dict, List, Mapping, Optional

from flask import g

from .api_common import ApiContext, payload as _payload
from .chat_appliance_scope import LIVE_FACTS_PROVIDER, record_decline, retrieval_answers_question
from .chat_connections import describe_connections
from .chat_web_search import (
    WEB_SEARCH_FAILED,
    WEB_SEARCH_FOUND,
    WEB_SEARCH_INSTRUCTION,
    WEB_SEARCH_NO_RESULTS,
    WEB_SEARCH_OFF,
    WEB_SEARCH_UNAVAILABLE,
    LIVE_FACT_CHECK_INSTRUCTION,
    asks_for_live_fact,
    live_fact_reply,
    locality,
    appliance_reading_reply,
    may_ask_for_live_fact,
    no_answer_found_reply,
    read_weather_for,
    search_the_web,
    weather_answer,
    weather_question,
    web_search_availability,
    web_search_metadata,
)
from .docker_health import container_runtime_healthy
from .weather_source import (
    OPEN_METEO, OPEN_METEO_HOSTS, WEATHER_ANSWERED, WEATHER_ASKED, WEATHER_FAILED, WEATHER_WHERE,
)
from .web_research import WebResearchManager

#: Sources an AI Chat answer reads at most (`ChatInference.answer` keeps six).
ANSWER_SOURCES = 6

#: Where a weather answer's requests went, as the audit names it (VD-222 item 4).
WEATHER_DESTINATION = "{} ({})".format(OPEN_METEO, ", ".join(OPEN_METEO_HOSTS))

#: The role that may turn Web search on (owner ruling, VD-221 item 14).
WEB_SEARCH_ROLE = "administrator"
WEB_SEARCH_ADMIN_ONLY = (
    "Only an administrator can turn on Web search, because it sends questions "
    "off this appliance to public search engines."
)


def research_manager(callbacks: Mapping[str, Any]) -> Any:
    """The guarded search's lifecycle owner, as the setup routes build it."""
    configured = callbacks.get("web_research_manager")
    return configured if configured is not None else WebResearchManager(
        docker_healthy=container_runtime_healthy,
    )


def may_search(role: str) -> bool:
    return role == WEB_SEARCH_ROLE


def web_search_refusal(body: Mapping[str, Any], role: str):
    """The 403 for a non-administrator asking to turn Web search on, or None."""
    if body.get("web_search") is True and not may_search(role):
        return _payload(error={"code": "web_search_admin_only", "message": WEB_SEARCH_ADMIN_ONLY}, status=403)
    return None


class WebTurn:
    """What Web search decided for one turn.

    ``decline`` is a reply that stands instead of an answer; otherwise
    ``passages`` and ``citations`` go first among the answer's sources and
    ``metadata`` is stored on the answer.
    """

    def __init__(self, *, decline: Optional[Dict[str, Any]] = None,
                 searched: Optional[Mapping[str, Any]] = None, sent_to: str = "",
                 check_live: bool = False):
        self.decline = decline
        self.searched = dict(searched or {})
        self.passages: List[Dict[str, Any]] = list(self.searched.get("passages") or [])
        self.web_citations: List[Dict[str, Any]] = list(self.searched.get("citations") or [])
        self.metadata = (
            {"web_search": web_search_metadata(self.searched, sent_to=sent_to)}
            if self.searched else {}
        )
        # Passed to `answer` only when there is something to say: a stand-in
        # provider with the older signature still answers a turn with neither.
        # With pages, their own instruction covers live facts; without, a
        # question that may want live data is told to check (review round 2).
        notes = (WEB_SEARCH_INSTRUCTION,) if self.passages else (
            (LIVE_FACT_CHECK_INSTRUCTION,) if check_live else ())
        self.options = {"instructions": notes} if notes else {}

    def knowledge(self, retrieved: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """The owner's passages that still fit beside the web pages."""
        return list(retrieved or [])[:max(0, ANSWER_SOURCES - len(self.passages))]

    def sources(self, retrieved: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        return [*self.passages, *self.knowledge(retrieved)]

    def citations(self, knowledge_citations: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """One citation per source, in the order the model numbers them."""
        return [*self.web_citations, *knowledge_citations]


def _connection(inference: Any, model: str) -> Optional[Dict[str, Any]]:
    """AI Chat's resolved connection, or None when it cannot be read here."""
    try:
        return inference._connection(model)  # the one resolver (VD-220)
    except Exception:  # noqa: BLE001 - absence-ok: no planner, nothing named as the destination
        return None


def _destination(connection: Optional[Mapping[str, Any]], cluster_credential_id: str = "") -> str:
    """Where the pages go with the question when AI Chat's model is not on this
    appliance - decided by `connection_locality`, the rule the picker reads.
    Vaelor's own GPU cluster is the owner's machines and is never named here."""
    if locality(connection, cluster_credential_id).get("local") is not False:
        return ""
    described = describe_connections([dict(connection or {})])
    item = described[0] if described else {}
    return " ".join(str(item.get("provider_label") or item.get("label") or "another machine").split())


def web_search_turn(
    callbacks: Mapping[str, Any], inference: Any, *, actor: str, role: str,
    message: str, model: str, enabled: bool, retrieved: Any, history: Any = (),
    security: Any = None, remote_addr: str = "", cluster_credential_id: str = "",
) -> WebTurn:
    """Decide Web search for one AI Chat turn.

    With the toggle off, nothing is searched; a question about the world as it
    is now gets the honest reply instead of a guess, unless the owner's own
    documents answer it. With it on, the guarded search runs; a live question it
    cannot answer is answered honestly (and asked about), and any other question
    goes on to the model with whatever was found. Every search that leaves the
    appliance, and every refusal, is audited through ``security`` (the actions
    are written here as literals, so tools/audit_actions.py names them on screen).
    """
    # Two altitudes (review round 2): only the tight shape skips the model; a
    # loose one reaches it with an instruction to check, which costs nothing
    # when it was a false match.
    answered_by_knowledge = retrieval_answers_question(message, retrieved)
    live = asks_for_live_fact(message, history) and not answered_by_knowledge
    # The owner's own documents are a source; with them, no check is asked for.
    check_live = not live and not answered_by_knowledge and may_ask_for_live_fact(message)

    def decline(reply: Dict[str, Any]) -> WebTurn:
        if security is not None:
            security.audit(actor, "ai_chat.live_fact.decline", "success",
                           remote_addr=remote_addr, details={
                               "web_search": reply["metadata"]["web_search"]})
        return WebTurn(decline=reply)

    # "cpu temperature" is this machine's reading, whatever the toggle: the
    # Assistant reads it (VD-220), and no part of it is a place to look up
    # (VD-222 item 4 review, HIGH 1).
    if live and (weather_question(message, history) or {}).get("component"):
        if security is not None:
            security.audit(actor, "ai_chat.live_fact.decline", "success", remote_addr=remote_addr,
                           details={"reason": "appliance_question"})
        return WebTurn(decline=appliance_reading_reply())

    enabled = enabled and may_search(role)
    if not enabled:
        if not live:
            return WebTurn(check_live=check_live)
        available = web_search_availability(research_manager(callbacks))
        if available["available"]:
            return decline(live_fact_reply(WEB_SEARCH_OFF))
        return decline(live_fact_reply(WEB_SEARCH_UNAVAILABLE, available["reason"]))
    available = web_search_availability(research_manager(callbacks))
    tools = callbacks.get("assistant_tools")
    if not available["available"] or tools is None:
        reason = available["reason"] if not available["available"] else (
            "the guarded search tools are not wired on this appliance")
        if live:
            return decline(live_fact_reply(WEB_SEARCH_UNAVAILABLE, reason))
        return WebTurn(searched={"outcome": WEB_SEARCH_UNAVAILABLE, "reason": reason}, check_live=check_live)
    # VD-222 item 4: a weather question naming a place - or a reply naming one
    # of the places asked about - is read from Open-Meteo, not searched. Only
    # the place leaves the appliance. A place Open-Meteo does not know, or a
    # failed read, goes on to the guarded search below as before.
    weather = read_weather_for(tools, message, history, live=live, actor=actor,
                               administrator=may_search(role))
    if weather is not None:
        if security is not None and weather.get("requests"):
            security.audit(
                actor, "ai_chat.web_search", "failure" if weather["outcome"] == WEATHER_FAILED else "success",
                target=str(weather.get("place") or "")[:200], remote_addr=remote_addr, details={
                    "query": str(weather.get("place") or "")[:200], "outcome": weather["outcome"],
                    "destination": WEATHER_DESTINATION, "requests": list(weather.get("requests") or []),
                    "reason": str(weather.get("reason") or "")[:300], "sent_to": "",
                })
        if weather["outcome"] in {WEATHER_ANSWERED, WEATHER_ASKED, WEATHER_WHERE}:
            return WebTurn(decline=weather_answer(weather))
    connection = _connection(inference, model)
    searched = search_the_web(
        tools, message, actor=actor, administrator=may_search(role),
        connection=connection, history=history, cluster_credential_id=cluster_credential_id,
    )
    sent_to = _destination(connection, cluster_credential_id) if searched["outcome"] == WEB_SEARCH_FOUND else ""
    # The query left the appliance for the public engines through the guarded
    # search; the pages may next leave it for a hosted model. Both are said.
    if security is not None:
        security.audit(
            actor, "ai_chat.web_search",
            "failure" if searched["outcome"] == WEB_SEARCH_FAILED else "success",
            target=searched["query"][:200], remote_addr=remote_addr, details={
                "query": searched["query"], "outcome": searched["outcome"],
                "results": searched["results"], "pages": len(searched["passages"]),
                "destination": "guarded web research (public search engines)",
                "sent_to": sent_to,
            })
    if searched["outcome"] == WEB_SEARCH_FAILED and live:
        return decline(live_fact_reply(WEB_SEARCH_FAILED, searched["reason"]))
    if searched["outcome"] == WEB_SEARCH_NO_RESULTS and live:
        return decline(no_answer_found_reply(message, searched["query"], searched["results"]))
    return WebTurn(searched=searched, sent_to=sent_to, check_live=check_live)


def web_search_choice(body: Mapping[str, Any], *, new_conversation: bool) -> Dict[str, Any]:
    """The conversation patch remembering the toggle, for a chat this send creates.

    An existing chat's choice is written by its own PATCH when the owner clicks;
    writing it again at the end of a send would undo a click made while the
    search ran (review round 1, item 7).
    """
    chosen = body.get("web_search")
    return {"web_search": chosen} if new_conversation and isinstance(chosen, bool) else {}


def record_live_reply(
    store: Any, actor: str, turn: WebTurn, message: str, body: Mapping[str, Any], *,
    model: str, temporary: bool, conversation: Optional[Dict[str, Any]],
    collections: Optional[list],
) -> Dict[str, Any]:
    """Store the honest reply as the turn it was, with its metadata, so a reload
    still says which state Web search was in (review round 1, item 10)."""
    if temporary:
        held = record_decline(
            store, actor, turn.decline, message, model=model, temporary=True,
            provider=LIVE_FACTS_PROVIDER,
        )
        # An Open-Meteo answer is not a refusal and cites its source (VD-222 item 4).
        if turn.decline["metadata"].get("declined") is False:
            held["message"]["citations"] = list(turn.decline.get("citations") or [])
            held["declined"] = False
        return held
    created = conversation is None
    if created:
        conversation = store.ensure_conversation(
            actor, title=str(message)[:100], collections=list(collections or []),
        )
    choice = web_search_choice(body, new_conversation=created)
    if choice:
        try:
            store.update_conversation(actor, conversation["id"], choice)
        except (AttributeError, ValueError, sqlite3.Error):  # absence-ok: the reply stands
            pass
    store.add_message(actor, conversation["id"], "user", message)
    recorded = store.add_message(
        actor, conversation["id"], "assistant", turn.decline["answer"],
        list(turn.decline.get("citations") or []), None,
        LIVE_FACTS_PROVIDER, turn.decline["metadata"],
    )
    return {
        "conversation_id": conversation["id"], "message": recorded, "model": model,
        "provider": LIVE_FACTS_PROVIDER, "declined": turn.decline["metadata"].get("declined") is not False,
    }


def register_chat_web_search_routes(context: ApiContext) -> None:
    blueprint = context.blueprint
    callbacks = context.callbacks
    require_auth = context.require_auth

    @blueprint.get("/ai-chat/web-search")
    @require_auth("operator")
    def chat_web_search_status():
        """Whether AI Chat may offer Web search; never runs a search."""
        view = web_search_availability(research_manager(callbacks))
        role = getattr(g.auth_session, "role", "")
        view["administrator"] = role == "administrator"
        view["may_turn_on"] = may_search(role)
        view["role_reason"] = "" if may_search(role) else WEB_SEARCH_ADMIN_ONLY
        return _payload(view)
