"""Grounding inputs shared by every AI Chat answer path.

AI Chat grounds an answer on two independent things, and they must never be
conflated. Knowledge collections are this surface's retrieval corpus: they are
searched per request and cited back as [S#]. Saved assistant memory is
cross-surface background the appliance recorded earlier; it is never a citation
and never a search result. Both selections are resolved here so the message and
regenerate routes cannot drift apart on either one.
"""

from __future__ import annotations

import logging
import re
import sqlite3
from typing import Any, Dict, List, Mapping, Optional, Sequence, Tuple


LOGGER = logging.getLogger(__name__)

MAX_CHAT_MEMORIES = 4
# One request may search at most this many collections, which is the ceiling
# the store already applies. Bounding the list here means a client that sends
# five thousand ids is rejected as a selection rather than reaching SQL.
MAX_CHAT_COLLECTIONS = 10

# Curated memory is administrator-owned content: every verb on
# /api/v2/assistant/memories is administrator-only. Grounding an answer on a
# memory is a read of that content - the model is handed the text, and on a
# connected provider it leaves this appliance - so it is gated on the role that
# may read it. Widening this is a product decision about where curated content
# is allowed to travel, not an implementation detail, and it needs its own test.
MEMORY_GROUNDING_ROLE = "administrator"


def memory_grounding_allowed(role: str) -> bool:
    """Whether an answer for this role may be grounded on curated memory."""
    return str(role or "").strip().lower() == MEMORY_GROUNDING_ROLE


def _bounded_ids(items: Any) -> List[str]:
    """A clean, deduplicated, bounded list of collection identifiers."""
    selected: List[str] = []
    for item in items or []:
        if not isinstance(item, str):
            continue
        value = item.strip()
        if value and value not in selected:
            selected.append(value)
        if len(selected) >= MAX_CHAT_COLLECTIONS:
            break
    return selected


def resolve_collections(
    body: Mapping[str, Any],
    preference: Mapping[str, Any],
    conversation: Optional[Mapping[str, Any]] = None,
) -> Tuple[List[str], bool]:
    """Resolve which collections to search, and whether the client chose them.

    An empty list and an absent key used to mean the same thing, and that is
    what silently turned retrieval off for good. The client sends
    `collection_ids` on every request, so a conversation created before any
    collection existed replayed its stored `[]` as "search nothing" - and the
    same request then wrote that `[]` back over the conversation, so the
    setting could never recover on its own.

    An explicit list is now always honoured, empty included, because
    deselecting everything is a real choice. An absent or non-list value means
    "not specified": it falls back to the conversation's own selection, then to
    the actor's saved preference, and the caller leaves the stored selection
    alone rather than overwriting it.

    Whatever the source, the result is bounded and deduplicated. An
    unbounded client list reached the query builder verbatim and turned a
    stale browser tab into a 500.
    """
    requested = body.get("collection_ids")
    if isinstance(requested, (list, tuple)):
        return _bounded_ids(requested), True
    stored = _bounded_ids((conversation or {}).get("collections"))
    if stored:
        return stored, False
    return _bounded_ids(preference.get("collection_ids")), False


def searchable_collections(
    store: Any, actor: str, collection_ids: Sequence[str]
) -> List[str]:
    """The subset of a selection this actor can actually read.

    A collection the actor does not own is not a search target, and counting
    it as one is how a request could report "searched 1 collection" for
    something it never looked at. Resolving ownership once, here, also keeps
    an id that no longer exists - an administrator deleted the collection
    while an operator's tab was open - from ever reaching the query builder.
    """
    requested = _bounded_ids(collection_ids)
    if not requested or store is None:
        return []
    resolver = getattr(store, "owned_collections", None)
    if resolver is None:
        return requested
    try:
        return [str(item) for item in resolver(actor, requested)]
    except (AttributeError, TypeError, ValueError, sqlite3.Error):
        LOGGER.warning(
            "AI Chat could not resolve collection ownership for %r; "
            "answering without retrieval",
            actor, exc_info=True,
        )
        return []


def retrieval_summary(
    collection_ids: Sequence[str], retrieved: Optional[Sequence[Any]]
) -> Dict[str, Any]:
    """Say plainly whether a search ran over the user's knowledge, and what it found.

    A search that matched nothing produced exactly the same answer shape as no
    search at all: clean, confident and uncited. The user could not tell the
    two apart, which is the single most common reason retrieval is reported as
    "not working" when it ran correctly and the documents simply did not match.

    The count is of collections that were actually searched, so this must be
    given the ownership-filtered selection rather than whatever the client
    asked for.
    """
    return {
        "searched": bool(collection_ids),
        "collections": len(collection_ids),
        "passages": len(retrieved or []),
    }


def curated_memories(store: Any, query: str, limit: int = MAX_CHAT_MEMORIES):
    """Bounded curated memory for the AI Chat prompt.

    The memory store is appliance-wide rather than per-surface, so AI Chat
    reads the same curated facts the Assistant does. Only the content and its
    type travel into the prompt, and the caller labels them as background
    context so the model cannot present one as a retrieved source.

    Two things are reported, not one. A store that could not be read is not an
    appliance that remembers nothing, and answering as though memory had been
    consulted when the read failed is a quiet untruth - so the caller is told
    which of the two happened and says so in the reply.
    """
    if store is None:
        return {"memories": [], "available": True}
    if not re.findall(r"[A-Za-z0-9]{2,}", str(query)):
        # No searchable term means an empty full-text expression, which
        # matches every row and orders by pinned. A blank message used to
        # return the whole curated store in a single request. No question, no
        # grounding.
        return {"memories": [], "available": True}
    try:
        rows = store.list_memories(
            query=str(query)[:200], limit=max(1, min(int(limit), 8))
        )
    except (AttributeError, TypeError, ValueError, sqlite3.Error):
        LOGGER.warning(
            "AI Chat could not read curated memory; answering without it",
            exc_info=True,
        )
        return {"memories": [], "available": False}
    selected = []
    for item in rows:
        if not isinstance(item, Mapping):
            continue
        content = str(item.get("content", "")).strip()
        if not content:
            continue
        selected.append({
            "type": str(item.get("memory_type", ""))[:40],
            "content": content[:700],
        })
        if len(selected) >= limit:
            break
    return {"memories": selected, "available": True}
