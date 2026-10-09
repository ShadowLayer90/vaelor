"""Parse and validate a pasted model source: a catalog id or a Hugging Face repo.

The Serve-a-model flow (§3 "Models") lets the operator either pick a curated
catalog model or **paste a Hugging Face link** and let Vaelor pull it. Both
arrive as one free-text ``model_source`` field, so the ambiguity has to be
resolved once, in a pure function, before any SSH or download happens:

* a value that matches a catalog id is a **catalog** source (its weights repo
  and sizing geometry come from the reviewed catalog entry);
* anything else is treated as a **Hugging Face** repo reference and is parsed
  into ``{repo, revision}`` and *validated* - a non-Hugging-Face host, a path
  that is not ``org/name``, or junk characters are refused with a reason rather
  than handed to the runtime to fail opaquely later.

Keeping this pure (no network, no filesystem) is deliberate: whether a pasted
string is a well-formed repo reference is decided by its shape, and the honest
answer to "is this a Hugging Face repo?" is a syntactic one. Whether the repo
*exists* is the downloader's job (a later phase); B1 only needs the parsed,
validated source threaded through to the runtime, which is what pulls it.
"""

from __future__ import annotations

from typing import Any, Dict, Iterable, Optional
from urllib.parse import urlsplit

#: The hosts a pasted link may point at. A Hugging Face repo reference that
#: names any other host is refused rather than silently rewritten, because a
#: link to some other site is not a repo this platform knows how to pull, and
#: guessing would be exactly the "configured is not working" lie the project
#: removes. ``hf.co`` is Hugging Face's own short domain and redirects to
#: ``huggingface.co``, so it is accepted as the same source.
HUGGINGFACE_HOSTS = frozenset({"huggingface.co", "www.huggingface.co", "hf.co"})

#: One path segment of an ``org/name`` repo id. Hugging Face allows letters,
#: digits, and ``-`` ``_`` ``.`` and caps segment length; a segment must start
#: with an alphanumeric so a leading dot (a hidden/relative path) can never be
#: read as a repo owner.
_SEGMENT = "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789"
_SEGMENT_BODY = _SEGMENT + "-_."
_MAX_SEGMENT = 96

#: The path prefixes a Hugging Face repo URL uses before a revision. A blob or
#: tree link (``/org/name/tree/<rev>``) carries the branch/tag/commit, which is
#: the revision to pin; anything else after ``org/name`` is an unexpected shape
#: we refuse rather than guess at.
_REVISION_MARKERS = frozenset({"tree", "blob", "resolve"})

#: A revision (branch, tag, or commit sha). Branch names may contain ``/`` and
#: ``.`` (e.g. ``refs/pr/3``), so this is permissive but bounded and forbids
#: whitespace and control characters, which no valid git ref contains.
_REVISION_BODY = _SEGMENT + "-_./"
_MAX_REVISION = 128


def _valid_segment(value: str) -> bool:
    return (
        1 <= len(value) <= _MAX_SEGMENT
        and value[0] in _SEGMENT
        and all(character in _SEGMENT_BODY for character in value)
    )


def _valid_revision(value: str) -> bool:
    return (
        1 <= len(value) <= _MAX_REVISION
        and value[0] in _SEGMENT
        and all(character in _REVISION_BODY for character in value)
    )


def _repo_from_segments(owner: str, name: str) -> str:
    if not _valid_segment(owner) or not _valid_segment(name):
        raise ValueError(
            "A Hugging Face repo is 'org/name' with letters, digits, and "
            "'-' '_' '.' only."
        )
    return f"{owner}/{name}"


def parse_hf_repo(value: Any) -> Dict[str, Optional[str]]:
    """Normalise a Hugging Face reference to ``{repo, revision}`` or raise.

    Accepts the shapes an operator actually pastes:

    * ``org/name`` and ``org/name@revision`` (the bare id git and the HF CLI
      both take);
    * a full ``https://huggingface.co/org/name`` URL, optionally with a
      ``/tree/<revision>`` (or ``/blob/`` / ``/resolve/``) suffix or a
      ``?revision=`` query.

    A URL on any host outside `HUGGINGFACE_HOSTS`, a path that is not exactly a
    repo id, or a segment with illegal characters raises ``ValueError`` with the
    reason - the honest-degradation rule: an unusable source is refused up front,
    not passed to the runtime to fail during a pull.
    """
    text = str(value or "").strip()
    if not text:
        raise ValueError("Paste a Hugging Face repo link or 'org/name' id.")

    # A scheme (or a leading '//') means this is a URL: the host must be Hugging
    # Face and the path must be a repo id, optionally followed by a revision
    # marker. A bare 'org/name' has no scheme and is handled below.
    if "://" in text or text.startswith("//"):
        split = urlsplit(text if "://" in text else "https:" + text)
        host = split.hostname or ""
        if host.lower() not in HUGGINGFACE_HOSTS:
            raise ValueError(
                "Only huggingface.co links are accepted; '{}' is not a Hugging "
                "Face host.".format(host or "(no host)")
            )
        segments = [segment for segment in split.path.split("/") if segment]
        if len(segments) < 2:
            raise ValueError(
                "The link must point at a repo, as huggingface.co/org/name."
            )
        repo = _repo_from_segments(segments[0], segments[1])
        revision: Optional[str] = None
        rest = segments[2:]
        if rest:
            if rest[0] not in _REVISION_MARKERS or len(rest) < 2:
                raise ValueError(
                    "The link has an unexpected path after 'org/name'; paste "
                    "the repo home or a '/tree/<revision>' link."
                )
            revision = "/".join(rest[1:])
        query_revision = _query_revision(split.query)
        revision = query_revision or revision
        if revision is not None and not _valid_revision(revision):
            raise ValueError("The revision in the link is not a valid git ref.")
        return {"repo": repo, "revision": revision}

    # Bare 'org/name' or 'org/name@revision'.
    reference, _, revision = text.partition("@")
    owner, slash, name = reference.partition("/")
    if not slash:
        raise ValueError(
            "A Hugging Face repo id is 'org/name'; '{}' has no owner.".format(
                reference
            )
        )
    repo = _repo_from_segments(owner, name)
    revision = revision.strip() or None
    if revision is not None and not _valid_revision(revision):
        raise ValueError("The revision after '@' is not a valid git ref.")
    return {"repo": repo, "revision": revision}


def _query_revision(query: str) -> Optional[str]:
    for pair in query.split("&"):
        key, _, value = pair.partition("=")
        if key == "revision" and value:
            return value
    return None


def resolve_model_source(
    value: Any, catalog_ids: Iterable[str]
) -> Dict[str, Optional[str]]:
    """Classify ``value`` as a catalog id or a Hugging Face repo reference.

    Returns a discriminated dict: ``{"kind": "catalog", "id": ...}`` when the
    value names a reviewed catalog model, else ``{"kind": "huggingface", "repo":
    ..., "revision": ...}`` from `parse_hf_repo`. The two are distinguishable
    without a network call because a catalog id is a flat token while a Hugging
    Face reference always carries an ``org/name`` slash (or a URL), so a value
    that is neither a known catalog id nor slash-shaped is refused with a reason
    rather than guessed at.
    """
    text = str(value or "").strip()
    if not text:
        raise ValueError("Choose a catalog model or paste a Hugging Face link.")
    if text in set(catalog_ids):
        return {"kind": "catalog", "id": text, "repo": None, "revision": None}
    if "/" not in text and "://" not in text:
        raise ValueError(
            "'{}' is not a catalog model, and a Hugging Face source needs an "
            "'org/name' repo or a huggingface.co link.".format(text)
        )
    parsed = parse_hf_repo(text)
    return {"kind": "huggingface", "id": None, **parsed}
