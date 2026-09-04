"""Query shaping, topical relevance, and honest uncertainty for research agents.

#247w, confirmed live on the a79 SearXNG (bing+mojeek): a granted web-research
agent asked "What is the latest stable release version of the Caddy web server?"
got a candidate list of generic news homepages (foxnews.com, cnn.com, apnews.com
...), fetched them blind, and the model then answered "v2.8.4, on caddyserver.com"
- a site that was never among the candidates. The answer being right was luck.
Three independent gaps produced that failure, and this module closes the first
two and feeds the third:

* **The query was the whole question, verbatim.** On the live box, a query that
  leads with generic words ("latest stable release version ... caddy") made the
  engines return a trending-news default set, while the same words led by the
  distinctive subject ("caddy ... latest stable") returned caddyserver.com as the
  top hit. Reordering to put subject words first is a real improvement, but not a
  guarantee - "nginx web server latest version" still returned reddit.com on the
  same box - so it is best-effort, and relevance is the actual guarantee.

* **Nothing scored the candidates.** Whatever the engine returned was surfaced as
  "sources used", fetched, and left for the model to reason past. A general news
  homepage shares no topic word with "Caddy web server version"; scoring on that
  overlap drops it before it can be read or cited (LESSONS 8).

* **A run that read nothing still answered.** When no candidate is relevant the
  run has no public evidence, and presenting a training-knowledge answer as cited
  research is the LESSONS-11 failure. `clarifying_questions` turns that dead end
  into concrete, grounded questions the user can answer by re-running with more
  detail - an async agent's version of "ask when unsure".

No per-app or per-site lists live here: the subject nouns that identify what is
being asked about (caddy, nginx, grafana, kubernetes...) are exactly the words
NOT in the generic vocabulary below, so they carry the relevance signal
generically for any app the user names.
"""

from __future__ import annotations

import re
from typing import Any
from urllib.parse import urlsplit

# Topic-neutral words a technical task and an unrelated news homepage can share.
# Keeping them out of the "distinctive" set stops "Latest" on cnn.com counting
# as evidence for a "latest Caddy version" query. This is a generic vocabulary,
# not a subject list.
_GENERIC_TERMS = frozenset({
    "latest", "new", "news", "update", "updated", "current", "recent", "today",
    "now", "version", "release", "released", "stable", "download", "install",
    "installed", "official", "website", "site", "page", "home", "homepage",
    "online", "free", "best", "top", "guide", "tutorial", "docs", "documentation",
    "doc", "info", "information", "about", "overview", "review", "compare",
    "comparison", "world", "video", "watch", "live", "breaking", "story",
    "report", "blog", "post", "article", "software", "app", "application",
    "tool", "service", "internet", "program", "platform", "number", "headline",
    # Infrastructure-generic words a subject noun is never carried by alone: a
    # "Best Web Hosting Servers" listicle shares only these with a "Caddy web
    # server" task, so treating them as distinctive kept it (#247w finding 4).
    "web", "server", "cloud", "data", "hosting", "host",
})

# Question and command scaffolding: dropped from the search query and never a
# relevance signal. Mirrors chat_appliance_scope._RELEVANCE_STOPWORDS in spirit
# (#247v), tuned for a research task rather than an appliance-scope question.
_QUERY_STOPWORDS = frozenset({
    "the", "a", "an", "is", "are", "was", "were", "be", "been", "being", "am",
    "of", "to", "in", "on", "at", "for", "and", "or", "not", "no", "my", "our",
    "us", "this", "that", "these", "those", "it", "its", "what", "whats",
    "which", "who", "whom", "how", "when", "where", "why", "do", "does", "did",
    "done", "can", "could", "would", "should", "will", "shall", "may", "might",
    "must", "if", "so", "as", "by", "with", "from", "into", "i", "you", "we",
    "they", "he", "she", "him", "her", "them", "here", "there", "please",
    "tell", "show", "give", "find", "list", "fetch", "look", "search", "check",
    "want", "need", "me", "your", "their", "has", "have", "had", "get",
})


def _tokens(text: Any) -> list[str]:
    """Lowercase ``[a-z0-9]{2,}`` word tokens, matching the store's own split."""
    return re.findall(r"[a-z0-9]{2,}", str(text or "").lower())


def _normalize(token: str) -> str:
    """Strip a trailing plural ``s`` so "servers" meets "server".

    A light stand-in for a stemmer, applied identically to both sides of every
    comparison so "kubernetes" folds to the same "kubernete" on each - crude but
    consistent, which is all relevance overlap needs.
    """
    return token[:-1] if token.endswith("s") and len(token) > 3 else token


def query_terms(text: Any) -> list[str]:
    """Topic words of ``text`` in written order, de-duplicated by normalized form.

    Question scaffolding is dropped; the surviving words keep their original
    spelling and order, so a caller can tell which subject word came first.
    """
    seen: set[str] = set()
    terms: list[str] = []
    for token in _tokens(text):
        if token in _QUERY_STOPWORDS:
            continue
        norm = _normalize(token)
        if norm in _QUERY_STOPWORDS or norm in seen:
            continue
        seen.add(norm)
        terms.append(token)
    return terms


def shaped_query(task_text: Any) -> tuple[str, str]:
    """A search query with distinctive topic words first, and why it was shaped.

    Returns ``(query, reason)``. The reason is recorded in the run's capability
    audit so a reshaped query is never a silent edit (LESSONS 5/9). Falls back to
    the bounded raw text when there is nothing topical to lead with.
    """
    raw = " ".join(str(task_text or "").split())
    terms = query_terms(task_text)
    if not terms:
        return raw[:300], "no-topic-words: sent the task text unchanged"
    distinctive = [term for term in terms if _normalize(term) not in _GENERIC_TERMS]
    generic = [term for term in terms if _normalize(term) in _GENERIC_TERMS]
    if not distinctive:
        return (
            " ".join(terms)[:300],
            "generic-only: no distinctive subject word to lead the query with",
        )
    ordered = distinctive + generic
    reason = (
        "subject-first: led the query with the task's distinctive terms "
        "('{}') so the engine does not read leading filler as a request for "
        "trending news".format(" ".join(distinctive[:3]))
    )
    return " ".join(ordered)[:300], reason


def _candidate_haystack(item: dict) -> tuple[set[str], str]:
    """A candidate's normalized title+URL tokens, and its lowered raw text."""
    haystack = "{} {}".format(item.get("title", ""), item.get("url", "")).lower()
    return {_normalize(token) for token in _tokens(haystack)}, haystack


def _relevance(item: dict, distinctive: set[str], generic: set[str]) -> tuple[int, int]:
    """(distinctive_hits, generic_hits) for one candidate against the task terms.

    A distinctive term scores two for an exact token match and one for a
    substring of a longer token, so a "caddy" query matches the host
    "caddyserver.com" without matching every page that merely says "server".
    Generic shared words ("latest", "version") only break ties; they never make
    an off-topic page relevant on their own.
    """
    tokens, haystack = _candidate_haystack(item)
    distinct_hits = 0
    for term in distinctive:
        if term in tokens:
            distinct_hits += 2
        elif len(term) >= 4 and term in haystack:
            distinct_hits += 1
    generic_hits = sum(1 for term in generic if term in tokens)
    return distinct_hits, generic_hits


def rank_candidates(
    candidates: Any, task_text: Any, query: Any = "",
) -> tuple[list[dict], list[dict], str]:
    """Split search candidates into on-topic (best first) and off-topic.

    Returns ``(relevant, dropped, reason)``. A result that shares no distinctive
    topic word with the task - a general news homepage for a "Caddy web server
    version" task - is dropped before it can be fetched or cited. When the task
    itself carries no distinctive subject word, relevance falls back to any
    topical overlap rather than dropping everything.
    """
    items = [item for item in candidates or [] if isinstance(item, dict)]
    terms = query_terms("{} {}".format(task_text, query))
    distinctive = {_normalize(term) for term in terms if _normalize(term) not in _GENERIC_TERMS}
    generic = {_normalize(term) for term in terms if _normalize(term) in _GENERIC_TERMS}
    if not items:
        return [], [], "no-candidates: the search returned nothing"
    if not distinctive and not generic:
        return list(items), [], "no-topic-words: relevance not applied"
    use_distinctive = bool(distinctive)
    scored: list[tuple[int, int, dict]] = []
    dropped: list[dict] = []
    for item in items:
        distinct_hits, generic_hits = _relevance(item, distinctive, generic)
        keep = distinct_hits >= 1 if use_distinctive else generic_hits >= 1
        if keep:
            scored.append((distinct_hits, generic_hits, item))
        else:
            dropped.append(item)
    # Stable sort keeps equal-scored candidates in their original engine order.
    scored.sort(key=lambda row: (row[0], row[1]), reverse=True)
    relevant = [row[2] for row in scored]
    if not relevant:
        reason = "no-relevant-candidates: none of {} results shared a topic word with the task".format(
            len(items)
        )
    elif dropped:
        reason = "relevance-filtered: kept {} of {} results that share a topic word with the task".format(
            len(relevant), len(items)
        )
    else:
        reason = "all-relevant: every result shares a topic word with the task"
    return relevant, dropped, reason


def _subject_phrase(task_text: Any) -> str:
    """A short, human-readable name for what the task is about, for questions."""
    terms = query_terms(task_text)
    distinctive = [term for term in terms if _normalize(term) not in _GENERIC_TERMS]
    chosen = distinctive[:4] or terms[:4]
    phrase = " ".join(chosen)
    return phrase or " ".join(str(task_text or "").split())[:60] or "this task"


def _distinct_hosts(candidates: Any) -> list[str]:
    """Registrable-looking hostnames of candidate URLs, in first-seen order."""
    hosts: list[str] = []
    for item in candidates or []:
        if not isinstance(item, dict):
            continue
        try:
            host = (urlsplit(str(item.get("url", ""))).hostname or "").lower()
        except ValueError:
            host = ""
        host = host[4:] if host.startswith("www.") else host
        if host and host not in hosts:
            hosts.append(host)
    return hosts


def clarifying_questions(
    task_text: Any, *, candidates: Any, relevant: Any,
) -> list[str]:
    """Concrete questions to put back to the user when nothing could be verified.

    An async agent run cannot take an interactive turn, so "ask when unsure"
    means the result carries specific, answerable questions grounded in what the
    search actually returned - never a fabricated confident answer, never a bare
    "could not verify". Called only when a granted search ran and no page was
    read, so every branch here already knows the run has no evidence to reason
    over (owner requirement, 2026-08-21; LESSONS 8/11).
    """
    subject = _subject_phrase(task_text)
    candidate_list = [item for item in candidates or [] if isinstance(item, dict)]
    questions: list[str] = []
    hosts = _distinct_hosts(relevant)
    if len(hosts) >= 2:
        questions.append(
            "I found more than one candidate and could not tell which you meant: "
            "{}. Which one is right?".format(", ".join(hosts[:4]))
        )
    if not candidate_list:
        questions.append(
            "I could not find any public results for {}. Do you have an official "
            "documentation or release URL I can read? Re-run this task with that "
            "link included.".format(subject)
        )
    else:
        questions.append(
            "The public search returned {} result(s) but none clearly about {}, "
            "so I could not verify an answer. Do you have the official docs or "
            "release URL for it? Re-run this task with that link.".format(
                len(candidate_list), subject
            )
        )
    questions.append(
        "If I searched for the wrong thing, give me the exact project name or "
        "homepage for {} and I will look again.".format(subject)
    )
    # Preserve order, drop duplicates, bound the list and each question.
    seen: set[str] = set()
    result: list[str] = []
    for question in questions:
        text = " ".join(str(question).split())[:400]
        if text and text.lower() not in seen:
            seen.add(text.lower())
            result.append(text)
    return result[:4]
