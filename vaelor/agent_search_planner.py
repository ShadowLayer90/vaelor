"""Model-formulated, self-correcting web-search planning for agent runs.

#247y, proven live on the Z2: a custom research agent asked to "list the final
scores of MLB games Aug 11-17 2026" got dictionary definitions of the word
"final" and a FIFA link, not baseball. The runner did ONE deterministic keyword
search whose query LED with "final" (a common word treated as a distinctive
subject); a general engine returned dictionary "final" pages, and relevance
KEPT them because "final" was a query token. The model - which understands the
task's intent - never formulated the query, because the managed-local 4B has no
native tool-calling and the runner searches on its behalf.

This module lets the model write and, when the first pass finds nothing, rewrite
the actual web-search query, orchestrated by the runner. There are no per-topic,
per-subject, or blocklist vocabularies here: the model reads the task and names
the subject generically, the same way a person phrasing a search would. Every
query the runner actually runs is recorded in the capability audit (LESSONS
5/9), and when the model call fails or returns nothing the runner falls back to
the deterministic `shaped_query`, so this path is never worse than before.

The three helpers are deliberately thin so `agent_task_runner` stays under the
1,000-line ceiling by delegating search orchestration here rather than inlining
it:

* ``plan_search_query`` - one bounded model call returning ``(query, reason)``.
* ``run_web_search`` - one guarded ``research.search``, returning its facts,
  candidates, and any failure, so the reformulation pass can re-run the search
  without duplicating the acquisition-error handling.
* ``rank_search`` - the open-web relevance split (or the allowlist deferral),
  returning the on-topic set and the search facts with off-topic hits accounted
  for, so both the first pass and the reformulation rank identically.
"""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass, field
from typing import Any, Dict, List, Mapping, Optional

from .agent_research_relevance import rank_candidates, shaped_query
from .assistant_tools import AssistantToolError
from .inference_client import chat_completion, inference_timeout
from .model_profiles import reasoning_headroom_tokens
from .provider_runtime import (
    generation_parameters,
    provider_system_prompt,
    provider_user_content,
)

LOGGER = logging.getLogger(__name__)

# The fallback title for a fetched page that carried none, used by both the a83
# pre-fetch pipeline here and the tool loop's own source threading. One home so
# the two research-sourcing paths cannot drift, and the duplicate-literal scan
# does not read the same sentence in two modules as an accident.
PUBLIC_RESEARCH_SOURCE_TITLE = "Public research source"

# A search query is a handful of keywords, not prose. Kept small so a reasoning
# model cannot spend the whole budget thinking and return nothing; the headroom
# helper adds this model's own thinking margin on top.
QUERY_VISIBLE_TOKENS = 48

# The guarded search tool accepts 2..300 characters; a planned query is held
# well inside that so a verbose model answer cannot smuggle a sentence through.
MAX_QUERY_CHARS = 200

# The prompt is intentionally generic: it tells the model HOW to phrase a search
# (lead with the subject, drop filler, keep it short) without ever naming a
# topic, so it works for any task the user writes. No subject or blocklist
# vocabulary lives here or anywhere this module reaches.
SEARCH_PLANNER_PROMPT = (
    "You turn a research task into ONE web search query for a general search "
    "engine.\n"
    "Rules:\n"
    "- Lead with the concrete subject of the task: the specific names, places, "
    "dates, teams, products, or things it is actually about.\n"
    "- Drop filler: question words, instructions like 'list' or 'find' or 'give "
    "me', and articles.\n"
    "- Keep it short - a few keywords a search engine would use, not a "
    "sentence. No quotes, boolean operators, or explanation.\n"
    "- If a previous attempt is shown, its results were the WRONG subject. Write "
    "a DIFFERENT query that targets the specific page which would answer the "
    "task (for example the actual results or record, not a generic definition "
    "or an unrelated topic that merely shares a word).\n"
    "Return only the query text on a single line. You may instead return "
    '{"query": "..."}.'
)


def _resolve_model(connection: Mapping[str, str], headers: Dict[str, str], timeout: int) -> str:
    """The configured model id, or the endpoint's first advertised model."""
    model = str(connection.get("model", ""))
    if model:
        return model
    import urllib.request

    request = urllib.request.Request(
        "{}/models".format(connection["base_url"]), headers=headers
    )
    with urllib.request.urlopen(request, timeout=timeout) as response:
        body = json.loads(response.read(1024 * 1024).decode("utf-8"))
    return str(body["data"][0]["id"])


def _extract_query(content: Any) -> str:
    """One bounded query line from the model's reply.

    Accepts a bare line or a tiny ``{"query": "..."}`` object, tolerating a code
    fence, without adding a JSON-repair pass: if the text is not that object it
    is simply used as the query. Whitespace is collapsed and the result bounded.
    """
    text = str(content or "").strip()
    if text.startswith("```"):
        text = text.strip("`")
        if text[:4].lower() == "json":
            text = text[4:]
        text = text.strip()
    candidate = text
    if text.startswith("{"):
        try:
            parsed = json.loads(text)
        except json.JSONDecodeError:
            parsed = None
        if isinstance(parsed, dict) and str(parsed.get("query", "")).strip():
            candidate = str(parsed["query"])
    # A model that ignored "single line" gets its first non-empty line taken.
    for line in candidate.splitlines():
        collapsed = " ".join(line.split())
        if collapsed:
            return collapsed[:MAX_QUERY_CHARS]
    return ""


def plan_search_query(
    connection: Mapping[str, str], task_text: Any, timeout_seconds: int,
    *, prior: Optional[Mapping[str, Any]] = None,
) -> tuple[str, str]:
    """One bounded model call turning a task into a web-search query.

    Returns ``(query, reason)``. On any failure or an empty/short reply the query
    is ``""`` and the reason explains the fall-back, so the caller uses the
    deterministic ``shaped_query`` and the run never regresses below today.

    ``prior`` carries ``{"query": <previous>, "off_topic_titles": [...]}`` so the
    model can REFORMULATE after a first pass that read only off-topic results.
    """
    task = " ".join(str(task_text or "").split())[:2000]
    if not task:
        return "", "empty-task: no text to plan a search query from"
    headers = {"Content-Type": "application/json", "Accept": "application/json"}
    if connection.get("api_key"):
        headers["Authorization"] = "Bearer {}".format(connection["api_key"])
    payload: Dict[str, Any] = {"task": task}
    reformulated = False
    if isinstance(prior, Mapping):
        previous = " ".join(str(prior.get("query", "")).split())[:MAX_QUERY_CHARS]
        titles = [
            " ".join(str(item).split())[:160]
            for item in (prior.get("off_topic_titles") or [])[:5]
            if str(item).strip()
        ]
        if previous:
            payload["previous_attempt"] = {"query": previous, "off_topic_results": titles}
            reformulated = True
    try:
        timeout = inference_timeout(dict(connection), timeout_seconds)
        model = _resolve_model(connection, headers, timeout)
        body = chat_completion(
            dict(connection),
            {
                "model": model,
                "messages": [
                    {
                        "role": "system",
                        "content": provider_system_prompt(
                            SEARCH_PLANNER_PROMPT, dict(connection)
                        ),
                    },
                    {
                        "role": "user",
                        "content": provider_user_content(payload, dict(connection)),
                    },
                ],
                **generation_parameters(
                    dict(connection),
                    max_tokens=reasoning_headroom_tokens(
                        connection, QUERY_VISIBLE_TOKENS
                    ),
                    temperature=0,
                ),
            },
            headers,
            timeout,
        )
        content = (body["choices"][0].get("message") or {}).get("content") or ""
    except Exception:
        # Any endpoint, transport, or shape failure degrades to the deterministic
        # query; the caller owns the fallback so this path stays best-effort. But
        # a broken planner would otherwise fall back on EVERY run, forever,
        # indistinguishable from "endpoint down" - the silent-degradation trap
        # (LESSONS 1). Log it (a programming error carries its traceback here)
        # so an operator can tell "the model is unreachable" from "the planner
        # code is broken". The reason string still records the fallback.
        LOGGER.warning("search query planning failed; using the deterministic query", exc_info=True)
        return "", "planner-unavailable: model query planning failed; used the deterministic query"
    query = _extract_query(content)
    if len(query) < 2:
        return "", "planner-empty: model returned no usable query; used the deterministic query"
    if reformulated:
        return query, (
            "model-reformulated: the model rewrote the query after the first pass "
            "read only off-topic results ('{}')".format(query)
        )
    return query, (
        "model-formulated: the model wrote the web-search query for this task "
        "('{}')".format(query)
    )


def run_web_search(
    tools, query: str, *, actor: str, administrator: bool, scopes, web_access,
) -> tuple[Any, list, Optional[dict]]:
    """Run one guarded ``research.search``; return ``(facts, candidates, failure)``.

    On success ``facts`` is the search result dict and ``candidates`` its
    results. On an acquisition error ``facts`` is ``{"unavailable": failure}`` -
    the same shape the runner records elsewhere - ``candidates`` is empty, and
    ``failure`` is the recorded entry so the caller can append it once.
    """
    try:
        result = tools.run(
            "research.search", {"query": query}, actor=actor,
            administrator=administrator, granted_scopes=scopes,
            web_access=web_access,
        )["result"]
    except AssistantToolError as error:
        failure = {
            "stage": "acquisition", "tool": "research.search",
            "error": str(error)[:300],
        }
        return {"unavailable": failure}, [], failure
    candidates = result.get("results", []) if isinstance(result, dict) else []
    return result, candidates, None


def rank_search(
    search: Any, candidates: list, task_text: Any, query: str, domains: list,
) -> tuple[list, list, str, Any]:
    """Split candidates into on-topic and off-topic, and fold the drop count in.

    Returns ``(relevant, off_topic, reason, search_facts)``. Under a domain
    allowlist relevance is deferred to that allowlist (a stronger, user-chosen
    boundary the host already enforces), so nothing is dropped here. Otherwise
    `rank_candidates` drops results that share no distinctive topic word with the
    task before any is fetched or cited (#247w, LESSONS 8). ``search_facts`` is
    ``search`` with its results narrowed to the on-topic set and the excluded
    count updated, so the model can neither read nor cite a dropped result.
    """
    if domains:
        relevant, off_topic = list(candidates), []
        reason = "allowlisted: relevance deferred to the domain allowlist"
    else:
        relevant, off_topic, reason = rank_candidates(candidates, task_text, query)
    facts = search
    if isinstance(search, dict) and off_topic:
        facts = {
            **search,
            "results": relevant,
            "excluded_results": int(search.get("excluded_results") or 0) + len(off_topic),
            "off_topic_excluded": len(off_topic),
        }
    return relevant, off_topic, reason, facts


def fetch_relevant_pages(
    tools, heartbeat, *, relevant: list, search: Any, domains: list, web_access,
    actor: str, administrator: bool, scopes,
) -> tuple[list, list, Optional[list], Optional[dict]]:
    """Read up to three on-topic pages and build the citable source index.

    Returns ``(research_sources, fetch_failures, fetched, search_index)``.

    ``research_sources`` is the run's real, relevance-filtered candidate set as
    ``{source, summary}`` entries - never a URL the model invented. The URL is a
    key, not prose: it is checked against provenance and handed to the fetch
    tool, so it is bounded at 2,000 (what ``_public_https_url`` enforces) rather
    than truncated shorter, which once made any longer result silently
    unfetchable.

    An empty domain allowlist means "read what this run's own guarded search
    returned": those exact URLs are vetted and passed down so the tool checks
    provenance rather than trusting any address. When pages are in hand,
    ``search_index`` drops the bulky ``results`` bodies to a bare url+title index
    (the fetched text says it better, measured at 7,000 of 9,900 characters), and
    a redirect is recorded on the source so a citation never hides that the
    evidence came from a different page than the link. ``fetched`` and
    ``search_index`` are ``None`` when nothing was read.
    """
    research_sources = [
        {
            "source": str(item.get("url", ""))[:2000],
            "summary": str(item.get("title", PUBLIC_RESEARCH_SOURCE_TITLE))[:400],
        }
        for item in relevant[:20]
        if isinstance(item, dict) and str(item.get("url", "")).strip()
    ]
    vetted_urls = [item["source"] for item in research_sources]
    fetch_failures: list = []
    if not (relevant and (domains or vetted_urls)):
        return research_sources, fetch_failures, None, None
    fetch_access = {**web_access, "search_result_urls": vetted_urls}
    fetched: list = []
    for candidate in relevant[:3]:
        try:
            fetched.append(tools.run(
                "research.fetch", {"url": candidate.get("url", "")},
                actor=actor, administrator=administrator,
                granted_scopes=scopes, web_access=fetch_access,
            )["result"])
        except AssistantToolError as error:
            fetch_failures.append({
                "stage": "acquisition", "tool": "research.fetch",
                "url": str(candidate.get("url", ""))[:300],
                "error": str(error)[:300],
            })
        heartbeat()
    if not fetched:
        return research_sources, fetch_failures, None, None
    search_index = None
    if isinstance(search, dict):
        search_index = {
            **{key: value for key, value in search.items() if key != "results"},
            "results": [
                {
                    "url": str(item.get("url", ""))[:2000],
                    "title": str(item.get("title", ""))[:240],
                }
                for item in relevant[:20]
                if isinstance(item, dict)
            ],
        }
    landed = {
        str(item.get("url", "")): str(item.get("final_url", ""))
        for item in fetched
        if isinstance(item, dict) and item.get("redirected")
    }
    for source in research_sources:
        final = landed.get(source["source"], "")
        if final and final != source["source"]:
            source["retrieved_from"] = final[:2000]
    return research_sources, fetch_failures, fetched, search_index


@dataclass
class ResearchAcquisition:
    """The a83 pre-fetch pipeline's outputs, gathered in one return value.

    This is the runner's FALLBACK acquisition, extracted here so the runner stays
    under its line ceiling and so the loop-first path can skip it entirely. It
    reproduces exactly what the runner used to inline: model-formulated (and, on
    junk, model-reformulated) search, relevance ranking, and reading the on-topic
    pages into a citable source index.
    """

    facts: Dict[str, Any] = field(default_factory=dict)
    research_sources: list = field(default_factory=list)
    tool_failures: list = field(default_factory=list)
    search_attempts: list = field(default_factory=list)
    query_shaping: str = ""
    relevance: str = ""
    candidates: list = field(default_factory=list)
    search: Any = None
    relevant: list = field(default_factory=list)


def acquire_web_research(
    tools, heartbeat, *, agent, mode: str, task_description: str, scopes,
    administrator: bool, actor: str, web_access: Mapping[str, Any], domains: list,
) -> ResearchAcquisition:
    """Run the a83 research pre-fetch and return everything the runner surfaces.

    Called ONLY when the native tool loop did not drive the run (the agent is not
    loop-eligible, or the model could not follow the tool template). The model
    formulates the query when the endpoint is reachable, the deterministic
    ``shaped_query`` stands in otherwise, and every query actually run is recorded
    in ``search_attempts`` - so this path is byte-for-byte the behaviour the
    runner had before the loop existed.
    """
    facts: Dict[str, Any] = {}
    tool_failures: List[Dict[str, Any]] = []
    search_query, query_shaping = shaped_query(task_description)
    planner_connection = None
    if web_access.get("enabled") is True and hasattr(agent, "_connection"):
        try:
            planner_connection = agent._connection(mode)
        except Exception:
            planner_connection = None
    planner_timeout = getattr(agent, "timeout_seconds", 20)
    if planner_connection:
        planned_query, planned_reason = plan_search_query(
            planner_connection, task_description, planner_timeout,
        )
        if planned_query:
            search_query, query_shaping = planned_query, planned_reason
    search_attempts = [{"query": search_query, "reason": query_shaping}]
    search, candidates, failure = run_web_search(
        tools, search_query, actor=actor, administrator=administrator,
        scopes=scopes, web_access=web_access,
    )
    if failure:
        tool_failures.append(failure)
    facts["research.search"] = search
    relevant, off_topic, relevance, search = rank_search(
        search, candidates, task_description, search_query, domains,
    )
    if off_topic:
        facts["research.search"] = search
    if (
        planner_connection is not None and not domains and not relevant
        and isinstance(search, dict) and "unavailable" not in search
        and web_access.get("enabled") is True
    ):
        prior = {
            "query": search_query,
            "off_topic_titles": [
                str(item.get("title", "")) for item in (off_topic or candidates)
                if isinstance(item, dict)
            ],
        }
        next_query, next_reason = plan_search_query(
            planner_connection, task_description, planner_timeout, prior=prior,
        )
        if next_query and next_query != search_query:
            search_query, query_shaping = next_query, next_reason
            search_attempts.append({"query": next_query, "reason": next_reason})
            retry, candidates, retry_failure = run_web_search(
                tools, next_query, actor=actor, administrator=administrator,
                scopes=scopes, web_access=web_access,
            )
            if retry_failure:
                tool_failures.append(retry_failure)
            relevant, off_topic, relevance, search = rank_search(
                retry, candidates, task_description, next_query, domains,
            )
            facts["research.search"] = search
    research_sources, fetch_failures, fetched, search_index = fetch_relevant_pages(
        tools, heartbeat, relevant=relevant, search=search, domains=domains,
        web_access=web_access, actor=actor, administrator=administrator, scopes=scopes,
    )
    tool_failures.extend(fetch_failures)
    if fetched is not None:
        facts["research.fetch"] = fetched
        if search_index is not None:
            facts["research.search"] = search_index
    return ResearchAcquisition(
        facts=facts, research_sources=research_sources, tool_failures=tool_failures,
        search_attempts=search_attempts, query_shaping=query_shaping,
        relevance=relevance, candidates=candidates, search=search, relevant=relevant,
    )
