"""Fetch and parse ``SKILL.md`` files from a GitHub repository into skills.

An admin points at a public GitHub repository laid out as
``skills/<name>/SKILL.md`` (for example ``github.com/amd/skills``) and this
module bulk-imports every ``SKILL.md`` it finds as a candidate cluster skill.
It is the fetch-and-parse SERVICE only: a separate unit adds the skill
``instructions`` field, the HTTP route, and the operator surface, so nothing
here reaches into ``api_skills_routes.py`` or ``skills_library.py``.

Security posture (this is the one seam here that reaches the public internet
from the control plane, so every choice fails closed):

* The HOST is ALWAYS exactly ``api.github.com`` or ``raw.githubusercontent.com``.
  Both are hardcoded constants and are NEVER assembled from caller input; only
  the URL PATH carries sanitized owner / repo / ref / tree-path components.
* HTTPS only, with the certificate verified through
  ``ssl.create_default_context()``, a per-request timeout, and cross-host
  redirects refused rather than followed.
* Every response read is bounded (at most cap+1 bytes, rejected when over) and
  the whole import shares one total-bytes ceiling.
* An optional ``token`` is sent as an ``Authorization: Bearer`` header ONLY to
  those two GitHub hosts, is NEVER written to a log line or echoed into a
  warning or error, and is not persisted anywhere by this module.

Tolerance: one malformed ``SKILL.md`` (absent frontmatter, a missing name or
description, an oversize body, or a non-200 reply) is SKIPPED with a warning
and never aborts the run. Only a rejected owner / repo / ref, an unreachable
listing, or a 404 / 403 on the repository raises a typed :class:`ValueError`.

Completeness: GitHub's recursive tree listing stops at a size limit and says so
with ``truncated: true``. That flag is read, never ignored: a truncated listing
is followed by a bounded, non-recursive walk of the sub-trees, and when even the
walk cannot finish (its request or time budget, a rate limit, or a sub-tree
that is itself truncated) the result says ``listing_complete: false`` and a
warning tells the administrator the repository was too large to list in full.
"Nothing found" is only ever claimed about a listing that was complete.

Each skill's ``warnings`` are filled from what was actually read: frontmatter
keys Vaelor does not keep, and files in the skill's folder that the import does
not bring along. Whether the library would REFUSE a skill is not decided here;
the route asks the library's own register rules (``skills_library``).
"""

from __future__ import annotations

import json
import re
import ssl
import time
import urllib.error
import urllib.parse
import urllib.request
from typing import Any, Callable, Dict, List, Optional, Tuple

try:  # PyYAML is declared in requirements-release.txt / pyproject.toml.
    import yaml
except ImportError:  # pragma: no cover - exercised via the honest degrade below.
    yaml = None  # type: ignore[assignment]


# --- Fixed endpoints (never built from caller input) -------------------------

#: The API host used to resolve a default branch and to walk the git tree.
API_HOST = "api.github.com"
#: The host that serves raw file contents for a resolved ref.
RAW_HOST = "raw.githubusercontent.com"
#: Nothing outside this pair may ever be contacted, including via a redirect.
ALLOWED_HOSTS = frozenset({API_HOST, RAW_HOST})

#: GitHub rejects requests without one, so every call carries this identifier.
USER_AGENT = "vaelor-skills-import"


# --- Bounds (all fail closed) ------------------------------------------------

#: Ceiling on a single JSON reply from the API host.
MAX_API_BYTES = 2 * 1024 * 1024
#: Ceiling on one fetched ``SKILL.md`` body.
MAX_SKILL_BYTES = 256 * 1024
#: Ceiling on the whole import, summed across every fetch it performs.
MAX_TOTAL_BYTES = 32 * 1024 * 1024
#: Seconds a single request may take before it is abandoned.
REQUEST_TIMEOUT_SECONDS = 10
#: Default cap on how many ``SKILL.md`` files one import will pull.
DEFAULT_MAX_FILES = 50
#: Sub-tree listings a walk of a truncated repository may request. Without a
#: token GitHub allows 60 API requests an hour for the whole appliance, so an
#: anonymous walk takes at most a quarter of that and leaves room for the
#: owner's next import; a token (5,000 an hour) allows a deeper walk.
MAX_TREE_WALK_REQUESTS_ANONYMOUS = 15
MAX_TREE_WALK_REQUESTS_WITH_TOKEN = 40
#: Seconds a walk of a truncated repository may spend before it stops honestly.
MAX_TREE_WALK_SECONDS = 30.0
#: The warning a listing that could not be completed carries.
INCOMPLETE_LISTING_WARNING = (
    "This repository is too large to list completely, so some SKILL.md files "
    "may be missing from this list."
)
#: What an import says when GitHub refuses it for its request limit (a 403 or
#: 429 carrying the rate-limit headers or message), rather than "status 403".
RATE_LIMIT_MESSAGE = (
    "GitHub's limit on requests from this appliance has been reached (60 an "
    "hour without a GitHub token). Wait up to an hour, or add a token, and "
    "try again."
)
RATE_LIMIT_MESSAGE_WITH_TOKEN = (
    "GitHub's request limit for this token has been reached. Wait for it to "
    "reset, which can take up to an hour, and try again."
)


def _rate_limit_message(token: Optional[str]) -> str:
    """The rate-limit sentence that fits: with a token, advice to add one is wrong."""
    return RATE_LIMIT_MESSAGE_WITH_TOKEN if token else RATE_LIMIT_MESSAGE
RATE_LIMIT_WARNING = (
    "GitHub's request limit was reached part-way through, so the listing "
    "stopped early. Add a GitHub token or try again in an hour."
)

#: Frontmatter keys the import keeps; any other key is reported, not applied.
_KEPT_FRONTMATTER_KEYS = frozenset({"name", "description"})

#: Owner and repository segments accept only these characters.
_OWNER_REPO_PATTERN = re.compile(r"^[A-Za-z0-9._-]{1,100}$")
#: A branch, tag, or sha may also carry slashes.
_REF_PATTERN = re.compile(r"^[A-Za-z0-9._/-]{1,100}$")
#: A git object id is lowercase hex, 40 (SHA-1) or 64 (SHA-256) characters.
_HEX_DIGITS = frozenset("0123456789abcdef")
_TREE_SHA_LENGTHS = (40, 64)

#: The ``fetch`` seam contract: a URL and headers in, a status and bytes out,
#: optionally followed by the response headers that matter to the import
#: (``x-ratelimit-remaining``, lower-cased). A two-item reply is still read.
FetchResult = Tuple[Any, ...]
Fetch = Callable[[str, Dict[str, str]], FetchResult]

#: Response headers the real fetch passes back through the seam.
_KEPT_RESPONSE_HEADERS = ("x-ratelimit-remaining", "retry-after")


class _SkillParseError(Exception):
    """A single ``SKILL.md`` body could not be turned into a skill."""


class _FetchError(Exception):
    """A transport-level failure that carries no token or response detail."""


class _BudgetError(_FetchError):
    """The shared total-bytes ceiling for the import was reached."""


class _RateLimited(_FetchError):
    """GitHub refused the request for its request limit."""


def _is_rate_limited(status: int, body: bytes, headers: Dict[str, str]) -> bool:
    """Whether a 403/429 is GitHub's request limit rather than a real refusal.

    A 429 always is. A 403 is when the remaining-requests header reads 0, or the
    body says so (GitHub's JSON ``message``: "API rate limit exceeded ..."); a
    403 for a private repository carries neither and stays a refusal.
    """
    if status == 429:
        return True
    if status != 403:
        return False
    if str(headers.get("x-ratelimit-remaining", "")).strip() == "0":
        return True
    return b"rate limit" in bytes(body[:2048]).lower()


class _ByteBudget:
    """Track bytes pulled so the whole run stays under one hard ceiling."""

    def __init__(self, limit: int) -> None:
        self.limit = limit
        self.used = 0

    def charge(self, count: int) -> None:
        self.used += int(count)
        if self.used > self.limit:
            raise _BudgetError("total import budget reached")


class _NoFollowRedirect(urllib.request.HTTPRedirectHandler):
    """Refuse every redirect so a 3xx can never reach an unexpected host."""

    def redirect_request(self, request, file_pointer, code, message, headers, new_url):
        return None


# --- Input sanitizing --------------------------------------------------------


def _sanitized_segment(value: Any, label: str, pattern: "re.Pattern[str]") -> str:
    text = str(value or "").strip()
    if not pattern.match(text):
        raise ValueError("The GitHub {} contains characters that are not allowed.".format(label))
    return text


def _safe_tree_path(path: Any) -> Optional[str]:
    text = str(path or "")
    if not text or text.startswith("/") or ".." in text:
        return None
    return text


# --- The real fetch (small, TLS-verified, bounded) ---------------------------


def _default_fetch(url: str, headers: Dict[str, str]) -> FetchResult:
    context = ssl.create_default_context()
    opener = urllib.request.build_opener(
        _NoFollowRedirect,
        urllib.request.HTTPSHandler(context=context),
    )
    request = urllib.request.Request(url, headers=dict(headers))
    try:
        with opener.open(request, timeout=REQUEST_TIMEOUT_SECONDS) as response:
            status = int(getattr(response, "status", 200) or 200)
            body = response.read(MAX_API_BYTES + 1)
            kept = _kept_headers(response.headers)
    except urllib.error.HTTPError as error:
        try:
            body = error.read(MAX_API_BYTES + 1)
        except OSError:
            body = b""
        return int(error.code), bytes(body or b""), _kept_headers(error.headers)
    return status, bytes(body), kept


def _kept_headers(headers: Any) -> Dict[str, str]:
    """The few response headers the import reads, lower-cased."""
    kept: Dict[str, str] = {}
    for name in _KEPT_RESPONSE_HEADERS:
        try:
            value = headers.get(name) if headers is not None else None
        except AttributeError:
            value = None
        if value is not None:
            kept[name] = str(value)
    return kept


# --- Request assembly (host guarded, token scoped) ---------------------------


def _request_headers(host: str, token: Optional[str], *, api: bool) -> Dict[str, str]:
    headers = {"User-Agent": USER_AGENT}
    if api:
        headers["Accept"] = "application/vnd.github+json"
    if token and host in ALLOWED_HOSTS:
        headers["Authorization"] = "Bearer " + str(token)
    return headers


def _checked_fetch(
    fetcher: Fetch, url: str, token: Optional[str], *, api: bool, budget: _ByteBudget,
) -> FetchResult:
    if not url.startswith("https://"):
        raise ValueError("Only HTTPS GitHub URLs may be fetched.")
    host = urllib.parse.urlsplit(url).hostname or ""
    if host not in ALLOWED_HOSTS:
        raise ValueError("Refusing to contact a host outside GitHub.")
    headers = _request_headers(host, token, api=api)
    try:
        reply = fetcher(url, headers)
    except _BudgetError:
        raise
    except Exception as error:  # noqa: BLE001 - normalized so no token can leak.
        raise _FetchError("the GitHub request failed") from _scrub(error)
    status, body = int(reply[0]), bytes(reply[1] or b"")
    reply_headers = {
        str(key).lower(): str(value)
        for key, value in (dict(reply[2]) if len(reply) > 2 and reply[2] else {}).items()
    }
    budget.charge(len(body))
    if _is_rate_limited(status, body, reply_headers):
        raise _RateLimited("GitHub's request limit was reached")
    return status, body


def _scrub(error: BaseException) -> Optional[BaseException]:
    """Keep the exception type for the chain but drop any address detail."""
    return type(error)() if isinstance(error, (urllib.error.URLError, OSError)) else None


def _quote_path(path: str) -> str:
    return "/".join(urllib.parse.quote(segment, safe="._-") for segment in path.split("/"))


def _quote_ref(ref: str) -> str:
    return urllib.parse.quote(ref, safe="._/-")


# --- JSON helpers ------------------------------------------------------------


def _load_api_json(body: bytes) -> Dict[str, Any]:
    if len(body) > MAX_API_BYTES:
        raise ValueError("The GitHub reply exceeded the response size limit.")
    try:
        data = json.loads(body.decode("utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError) as error:
        raise ValueError("The GitHub reply was not readable JSON.") from error
    if not isinstance(data, dict):
        raise ValueError("The GitHub reply was not a JSON object.")
    return data


def _raise_for_status(status: int, owner: str, repo: str) -> None:
    if status in (403, 404):
        raise ValueError("GitHub returned {} for {}/{}.".format(status, owner, repo))
    if status != 200:
        raise ValueError("GitHub answered {}/{} with status {}.".format(owner, repo, status))


# --- Frontmatter parsing -----------------------------------------------------

_FRONTMATTER = re.compile(r"^---[ \t]*\r?\n(.*?)\r?\n---[ \t]*(?:\r?\n(.*)|\Z)", re.DOTALL)


def _split_frontmatter(text: str) -> Optional[Tuple[str, str]]:
    stripped = text.lstrip("﻿")
    match = _FRONTMATTER.match(stripped)
    if not match:
        return None
    return match.group(1), match.group(2) or ""


def parse_skill_document(text: str) -> Tuple[str, str, str]:
    """Return ``(name, description, body)`` or raise :class:`_SkillParseError`."""
    name, description, body, _extra = _parse_skill(text)
    return name, description, body


def _parse_skill(text: str) -> Tuple[str, str, str, List[str]]:
    """``(name, description, body, ignored_keys)``; the keys are sorted text."""
    if yaml is None:  # pragma: no cover - covered by the module-level degrade.
        raise _SkillParseError("PyYAML is not installed on this host")
    parts = _split_frontmatter(text)
    if parts is None:
        raise _SkillParseError("no YAML frontmatter block was found")
    front, body = parts
    try:
        data = yaml.safe_load(front)
    except yaml.YAMLError:
        raise _SkillParseError("the frontmatter block was not valid YAML")
    if not isinstance(data, dict):
        raise _SkillParseError("the frontmatter block was not a mapping")
    name = data.get("name")
    description = data.get("description")
    if not isinstance(name, str) or not name.strip():
        raise _SkillParseError("the frontmatter is missing a text name")
    if not isinstance(description, str) or not description.strip():
        raise _SkillParseError("the frontmatter is missing a text description")
    ignored = sorted(str(key) for key in data if str(key) not in _KEPT_FRONTMATTER_KEYS)
    return name.strip(), description.strip(), body, ignored


# --- Tree selection ----------------------------------------------------------


def _blob_paths(tree: List[Any]) -> List[str]:
    """Every safe file path in a listing, in listing order, without repeats."""
    paths: List[str] = []
    seen = set()
    for entry in tree:
        if not isinstance(entry, dict) or entry.get("type") != "blob":
            continue
        safe = _safe_tree_path(entry.get("path"))
        if safe is None or safe in seen:
            continue
        seen.add(safe)
        paths.append(safe)
    return paths


def _skill_folder(path: str) -> str:
    return path.rsplit("/", 1)[0] if "/" in path else ""


def _unimported_files(skill_paths: List[str], blobs: List[str]) -> Dict[str, List[str]]:
    """Map each SKILL.md to the other files its folder holds.

    A file belongs to the deepest skill folder that contains it, so a skill
    nested inside another skill's folder is not counted against the outer one.
    """
    by_folder = {_skill_folder(path): path for path in skill_paths}
    owned: Dict[str, List[str]] = {path: [] for path in skill_paths}
    for blob in blobs:
        if blob in owned:
            continue
        folder = _skill_folder(blob)
        while True:
            owner = by_folder.get(folder)
            if owner is not None:
                owned[owner].append(blob)
                break
            if not folder:
                break
            folder = _skill_folder(folder)
    return owned


def _file_warnings(ignored_keys: List[str], extra_files: List[str], complete: bool) -> List[str]:
    warnings: List[str] = []
    if ignored_keys:
        warnings.append(
            "Its frontmatter also sets {}; only the name, description and "
            "instructions are imported, so these are not applied.".format(
                ", ".join(ignored_keys)[:200],
            )
        )
    if extra_files:
        count = len(extra_files)
        warnings.append(
            "Its folder holds {}{} other {} (for example {}) that {} not imported; "
            "only the SKILL.md text is.".format(
                "" if complete else "at least ",
                count,
                "file" if count == 1 else "files",
                extra_files[0],
                "is" if count == 1 else "are",
            )
        )
    return warnings


def _select_skill_paths(tree: List[Any]) -> List[str]:
    return [path for path in _blob_paths(tree) if path.rsplit("/", 1)[-1].lower() == "skill.md"]


# --- Per-stage fetches -------------------------------------------------------


def _resolve_default_branch(
    fetcher: Fetch, owner: str, repo: str, token: Optional[str], budget: _ByteBudget,
) -> str:
    url = "https://{}/repos/{}/{}".format(API_HOST, owner, repo)
    try:
        status, body = _checked_fetch(fetcher, url, token, api=True, budget=budget)
    except _RateLimited as error:
        raise ValueError(_rate_limit_message(token)) from error
    except _FetchError as error:
        raise ValueError("The repository metadata could not be reached.") from error
    _raise_for_status(status, owner, repo)
    branch = _load_api_json(body).get("default_branch")
    if not isinstance(branch, str) or not _REF_PATTERN.match(branch):
        raise ValueError("The repository default branch could not be determined.")
    return branch


def _list_tree(
    fetcher: Fetch, owner: str, repo: str, ref: str, token: Optional[str], budget: _ByteBudget,
) -> Tuple[List[Any], bool]:
    """The recursive listing and whether GitHub truncated it."""
    url = "https://{}/repos/{}/{}/git/trees/{}?recursive=1".format(
        API_HOST, owner, repo, _quote_ref(ref),
    )
    try:
        status, body = _checked_fetch(fetcher, url, token, api=True, budget=budget)
    except _RateLimited as error:
        raise ValueError(_rate_limit_message(token)) from error
    except _FetchError as error:
        raise ValueError("The repository file listing could not be reached.") from error
    _raise_for_status(status, owner, repo)
    data = _load_api_json(body)
    tree = data.get("tree")
    if not isinstance(tree, list):
        raise ValueError("The repository file listing was missing its tree.")
    return tree, data.get("truncated") is True


def _walk_tree(
    fetcher: Fetch, owner: str, repo: str, ref: str, token: Optional[str],
    budget: _ByteBudget, clock: Callable[[], float],
) -> Tuple[List[Dict[str, Any]], bool, bool]:
    """List a too-large repository one folder at a time, breadth first.

    Returns ``(entries, complete, rate_limited)``; ``entries`` carry full
    paths. ``complete`` is true only when every folder was listed and none was
    itself truncated - any stop (request or time budget, a non-200, a failed or
    over-budget fetch, an unreadable reply) leaves it false. ``rate_limited``
    says the stop was GitHub's request limit, so the owner is told that
    rather than a generic "incomplete". The request cap depends on whether a
    token was given (see :data:`MAX_TREE_WALK_REQUESTS_ANONYMOUS`).
    """
    cap = MAX_TREE_WALK_REQUESTS_WITH_TOKEN if token else MAX_TREE_WALK_REQUESTS_ANONYMOUS
    deadline = clock() + MAX_TREE_WALK_SECONDS
    pending: List[Tuple[str, str]] = [("", ref)]
    entries: List[Dict[str, Any]] = []
    requests = 0
    while pending:
        if requests >= cap or clock() > deadline:
            return entries, False, False
        prefix, tree_ref = pending.pop(0)
        url = "https://{}/repos/{}/{}/git/trees/{}".format(
            API_HOST, owner, repo, _quote_ref(tree_ref),
        )
        requests += 1
        try:
            status, body = _checked_fetch(fetcher, url, token, api=True, budget=budget)
        except _RateLimited:
            return entries, False, True
        except _FetchError:
            return entries, False, False
        if status != 200:
            return entries, False, False
        try:
            data = _load_api_json(body)
        except ValueError:
            return entries, False, False
        tree = data.get("tree")
        if not isinstance(tree, list) or data.get("truncated") is True:
            return entries, False, False
        for entry in tree:
            if not isinstance(entry, dict):
                continue
            name = _safe_tree_path(entry.get("path"))
            if name is None:
                continue
            path = prefix + name
            if entry.get("type") == "blob":
                entries.append({"type": "blob", "path": path})
            elif entry.get("type") == "tree":
                sha = str(entry.get("sha") or "")
                if len(sha) not in _TREE_SHA_LENGTHS or not set(sha) <= _HEX_DIGITS:
                    return entries, False, False
                pending.append((path + "/", sha))
    return entries, True, False


def _import_one(
    fetcher: Fetch, owner: str, repo: str, ref: str, path: str,
    token: Optional[str], budget: _ByteBudget,
    extra_files: Tuple[List[str], bool] = ([], True),
) -> Tuple[Optional[Dict[str, Any]], Optional[str]]:
    url = "https://{}/{}/{}/{}/{}".format(
        RAW_HOST, owner, repo, _quote_ref(ref), _quote_path(path),
    )
    status, body = _checked_fetch(fetcher, url, token, api=False, budget=budget)
    if status != 200:
        return None, "Skipped {}: GitHub returned status {}.".format(path, status)
    if len(body) > MAX_SKILL_BYTES:
        return None, "Skipped {}: the file was over the per-file size limit.".format(path)
    try:
        text = body.decode("utf-8")
    except UnicodeDecodeError:
        return None, "Skipped {}: the file was not valid UTF-8 text.".format(path)
    try:
        name, description, instructions, ignored = _parse_skill(text)
    except _SkillParseError as error:
        return None, "Skipped {}: {}.".format(path, error)
    files, complete = extra_files
    skill = {
        "path": path,
        "name": name,
        "description": description,
        "instructions": instructions,
        "warnings": _file_warnings(ignored, files, complete),
    }
    return skill, None


# --- Public entry ------------------------------------------------------------


def import_repo_skills(
    owner: str,
    repo: str,
    ref: Optional[str] = None,
    token: Optional[str] = None,
    *,
    fetch: Optional[Fetch] = None,
    max_files: int = DEFAULT_MAX_FILES,
    clock: Callable[[], float] = time.monotonic,
) -> Dict[str, Any]:
    """Fetch and parse a repository's ``SKILL.md`` files into skill records.

    Returns ``{"skills": [...], "warnings": [...], "ref": str,
    "listing_complete": bool}`` where each skill is ``{"path", "name",
    "description", "instructions", "warnings"}``. ``listing_complete`` is false
    when the repository could not be listed in full (see the module docstring).
    Raises :class:`ValueError` on a rejected owner / repo / ref, an unreachable
    listing, or a 404 / 403; a single bad file is skipped with a warning.
    """
    owner = _sanitized_segment(owner, "owner", _OWNER_REPO_PATTERN)
    repo = _sanitized_segment(repo, "repository name", _OWNER_REPO_PATTERN)
    if ref is not None:
        ref = _sanitized_segment(ref, "ref", _REF_PATTERN)
    limit = max(1, int(max_files))
    fetcher: Fetch = fetch or _default_fetch
    budget = _ByteBudget(MAX_TOTAL_BYTES)
    warnings: List[str] = []

    if ref is None:
        ref = _resolve_default_branch(fetcher, owner, repo, token, budget)

    tree, truncated = _list_tree(fetcher, owner, repo, ref, token, budget)
    complete = not truncated
    if truncated:
        # The partial recursive listing is kept: a walk cut short may still
        # have missed files the partial listing already named.
        walked, complete, rate_limited = _walk_tree(
            fetcher, owner, repo, ref, token, budget, clock,
        )
        tree = list(tree) + walked
        if rate_limited:
            warnings.append(RATE_LIMIT_WARNING)
    if not complete:
        warnings.append(INCOMPLETE_LISTING_WARNING)
    skill_paths = _select_skill_paths(tree)
    if len(skill_paths) > limit:
        warnings.append(
            "Found {} SKILL.md files; only the first {} were imported.".format(
                len(skill_paths), limit,
            )
        )
        skill_paths = skill_paths[:limit]
    extra = _unimported_files(skill_paths, _blob_paths(tree))

    if yaml is None:
        warnings.append("PyYAML is unavailable, so no frontmatter could be parsed.")
        return {"skills": [], "warnings": warnings, "ref": ref, "listing_complete": complete}

    skills: List[Dict[str, Any]] = []
    for path in skill_paths:
        try:
            skill, warning = _import_one(
                fetcher, owner, repo, ref, path, token, budget, (extra[path], complete),
            )
        except _BudgetError:
            warnings.append("Stopped early: the import download budget was reached.")
            break
        except _RateLimited:
            warnings.append(RATE_LIMIT_WARNING)
            break
        except _FetchError:
            warnings.append("Skipped {}: the file download did not complete.".format(path))
            continue
        if warning:
            warnings.append(warning)
        if skill is not None:
            skills.append(skill)

    return {"skills": skills, "warnings": warnings, "ref": ref, "listing_complete": complete}
