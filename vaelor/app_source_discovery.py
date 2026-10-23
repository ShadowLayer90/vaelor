"""Find a custom application from its own sources first (VD-222 item 1).

Before any keyless web search, application research asks the two places an
open-source self-hosted application actually lives: its GitHub repository and
its Docker Hub image. Both are searched by the owner's own words for the
application - there is no table of applications here and there must never be
one (the owner's no-hardcoded-app-data rule). What is found is ranked by
evidence anyone can check: the repository's stars, the image's pull count, and
whether the repository's README names the image.

Outside destinations (VD-222 says each is named to the owner):

* ``api.github.com`` - repository search, README and latest release, read
  anonymously. GitHub allows 10 searches a minute and 60 other reads an hour
  per box.
* ``hub.docker.com`` - repository search and tag metadata, read anonymously
  (180 reads per window, measured 2026-10-19).

Every request goes through the caller's ``fetch``, which in production is the
application-research broker: the same public-HTTPS-only, DNS-pinned,
size-bounded fetch every other research source takes. Nothing here opens a
socket, and the URLs are built here from those two hosts only.

Answers are cached (bounded, with expiry) so repeated research does not spend
the box's GitHub allowance, and a reached limit is remembered and said plainly
rather than retried or guessed around. Several plausible matches are put to
the owner as a question; this module never picks one silently.
"""

from __future__ import annotations

import base64
import binascii
import difflib
import math
import re
import threading
import time
from collections import OrderedDict
from datetime import datetime, timezone
from typing import Any, Callable, Dict, Iterable, List, Mapping, Tuple
from urllib.parse import quote

from .application_research import ApplicationResearchError
from .application_research_workflow import ResearchWorkflowError, emit
from .container_registry import (
    canonical_reference, parse_image_reference, registry_candidate_urls,
)


GITHUB_API = "https://api.github.com"
DOCKER_HUB_API = "https://hub.docker.com/v2"
DIRECT_SOURCE_HOSTS = frozenset({"api.github.com", "hub.docker.com"})

#: How long one answer is reused. Search moves faster than a README.
SEARCH_CACHE_SECONDS = 10 * 60
DETAIL_CACHE_SECONDS = 30 * 60
CACHE_ENTRIES = 256
#: Wall-clock budget for one discovery. The broker serves one request at a
#: time, so this also bounds how long other research waits behind it; one
#: fetch already under way may still run to the broker's 8-second timeout.
DISCOVERY_BUDGET_SECONDS = 10.0
#: A second candidate this close to the leader is a real choice for the owner.
AMBIGUOUS_SHARE = 0.25
#: A misspelled name is offered back only when it is this close and has users.
NEAR_NAME_RATIO = 0.8
NEAR_MIN_PULLS = 10000
#: Standing a repository needs before it can prove anything (VD-222 owner
#: ruling: "not new and starless"). Below either bar it is not a match.
MIN_STANDING_STARS = 50
MIN_STANDING_DAYS = 30

GITHUB, DOCKER_HUB = "GitHub", "Docker Hub"
_LIMIT_NAMES = {"github": GITHUB, "github-search": GITHUB, "dockerhub": DOCKER_HUB}
_PREFIXED_IMAGE = re.compile(
    r"(?i)(?:ghcr\.io|docker\.io|registry\.hub\.docker\.com|hub\.docker\.com/r)"
    r"/([a-z0-9][a-z0-9._-]*)/([a-z0-9][a-z0-9._-]*)(?::([A-Za-z0-9_][A-Za-z0-9._-]{0,127}))?"
)
_BARE_IMAGE = re.compile(
    r"(?<![\w./:-])([a-z0-9][a-z0-9_-]*)/([a-z0-9][a-z0-9._-]*)"
    r"(?::([A-Za-z0-9_][A-Za-z0-9._-]{0,127}))?(?![\w/.-])"
)
# A GitHub package page, github.com/OWNER/REPO/pkgs/container/NAME, is the
# image ghcr.io/OWNER/NAME.
_GITHUB_PACKAGE = re.compile(
    r"(?i)github\.com/([a-z0-9][a-z0-9-]*)/[a-z0-9._-]+/pkgs/container/([a-z0-9][a-z0-9._-]*)"
)
_IMAGE_LINE = re.compile(r"(?i)docker\s+(?:run|pull|create)|^\s*image\s*:")


class SourceLimitReached(Exception):
    """A direct source refused this box until its limit resets."""

    def __init__(self, source: str, retry_seconds: int | None) -> None:
        self.source = source
        self.retry_seconds = retry_seconds
        super().__init__(limit_message(source, retry_seconds))


def limit_message(source: str, retry_seconds: int | None) -> str:
    """The one plain sentence for a reached limit (research-asks-when-unsure)."""
    name = _LIMIT_NAMES.get(source, source)
    if retry_seconds is None:
        return "{}'s limit for this box is reached; try again later.".format(name)
    minutes = max(1, int(math.ceil(retry_seconds / 60.0)))
    return "{}'s limit for this box is reached; try again in {} minute{}.".format(
        name, minutes, "" if minutes == 1 else "s",
    )


def _compact(value: Any) -> str:
    return re.sub(r"[^a-z0-9]+", "", str(value or "").lower())


def _words(query: str) -> List[str]:
    return re.findall(r"[a-z0-9]+", str(query or "").lower())


def _source_of(url: str) -> str:
    """The limit a request counts against: GitHub's search and its other reads
    are separate allowances (10 a minute, 60 an hour)."""
    if url.startswith(GITHUB_API + "/search/"):
        return "github-search"
    return "github" if url.startswith(GITHUB_API) else "dockerhub"


class DirectSourceCache:
    """Bounded expiring answers plus the remembered limit of each source."""

    def __init__(self, clock: Callable[[], float] = time.time, entries: int = CACHE_ENTRIES) -> None:
        self._clock = clock
        self._entries = max(1, int(entries))
        self._answers: "OrderedDict[str, Tuple[float, Any]]" = OrderedDict()
        self._limited_until: Dict[str, float] = {}
        self._lock = threading.Lock()

    def get(self, url: str) -> Tuple[bool, Any]:
        with self._lock:
            entry = self._answers.get(url)
            if entry is None:
                return False, None
            if entry[0] <= self._clock():
                del self._answers[url]
                return False, None
            self._answers.move_to_end(url)
            return True, entry[1]

    def put(self, url: str, value: Any, seconds: float) -> None:
        with self._lock:
            self._answers[url] = (self._clock() + seconds, value)
            self._answers.move_to_end(url)
            while len(self._answers) > self._entries:
                self._answers.popitem(last=False)

    def limited(self, source: str) -> int | bool:
        """Seconds until ``source`` may be asked again, or ``False`` if it may now."""
        with self._lock:
            until = self._limited_until.get(source)
            if until is None:
                return False
            remaining = until - self._clock()
            if remaining <= 0:
                del self._limited_until[source]
                return False
            return int(math.ceil(remaining))

    def mark_limited(self, source: str, retry_seconds: int | None) -> None:
        """Remember a refusal; one that names no reset holds off for a minute."""
        with self._lock:
            wait = 60 if retry_seconds is None else max(1, int(retry_seconds))
            self._limited_until[source] = self._clock() + wait


def _retry_seconds(headers: Mapping[str, str], now: float) -> int | None:
    retry_after = str(headers.get("retry-after", "")).strip()
    if retry_after.isdigit():
        return int(retry_after)
    reset = str(headers.get("x-ratelimit-reset", "")).strip()
    if reset.isdigit():
        return max(0, int(reset) - int(now))
    return None


def _is_limit(status: int, headers: Mapping[str, str]) -> bool:
    if status == 429:
        return True
    return status == 403 and (
        str(headers.get("x-ratelimit-remaining", "")).strip() == "0"
        or "retry-after" in headers
    )


class AppSourceDiscovery:
    """Search GitHub and Docker Hub for one application, by its name only."""

    def __init__(
        self,
        fetch: Callable[[str], Any],
        *,
        target_architecture: str = "amd64",
        cache: DirectSourceCache | None = None,
        clock: Callable[[], float] = time.time,
        monotonic: Callable[[], float] = time.monotonic,
    ) -> None:
        self._fetch = fetch
        self._target = "arm" if target_architecture == "arm/v7" else target_architecture
        self._clock = clock
        self._monotonic = monotonic
        self.cache = cache or DirectSourceCache(clock)

    # --- one guarded, cached read -------------------------------------------
    def _get(self, url: str, ttl: float, deadline: float) -> Any:
        """Read one JSON document, or ``None`` when the source has no such page."""
        found, value = self.cache.get(url)
        if found:
            return value
        source = _source_of(url)
        wait = self.cache.limited(source)
        if wait is not False:
            raise SourceLimitReached(source, wait)
        if self._monotonic() > deadline:
            raise TimeoutError("direct source discovery ran out of time")
        try:
            document = self._fetch(url)
        except ApplicationResearchError as error:
            status = int(getattr(error, "status", 0) or 0)
            headers = dict(getattr(error, "rate_headers", {}) or {})
            if _is_limit(status, headers):
                retry = _retry_seconds(headers, self._clock())
                self.cache.mark_limited(source, retry)
                raise SourceLimitReached(source, retry) from error
            if status == 404:
                self.cache.put(url, None, ttl)
                return None
            raise
        self.cache.put(url, document, ttl)
        return document

    # --- the discovery ------------------------------------------------------
    def discover(self, query: str) -> Dict[str, Any]:
        """Return what the application's own sources say about it.

        ``status`` is ``found`` (sources and at most one image), ``ambiguous``
        (``candidates`` for the owner to choose from), ``limited`` (a source's
        limit stopped the search; ``notes`` says so plainly) or ``none``.
        """
        words = _words(query)
        name = _compact(query)
        result: Dict[str, Any] = {
            "status": "none", "query": " ".join(words), "repository": None,
            "image": None, "candidates": [], "notes": [], "source_urls": [],
            "image_reference": [],
        }
        if len(name) < 2:
            return result
        deadline = self._monotonic() + DISCOVERY_BUDGET_SECONDS
        try:
            return self._discover(words, name, result, deadline)
        except SourceLimitReached as limit:
            result["status"] = "limited"
            result["notes"].append(str(limit))
            return result
        except (ApplicationResearchError, TimeoutError, KeyError, TypeError, ValueError) as error:
            result["notes"].append(
                "GitHub and Docker Hub could not be read for this request ({}), so "
                "Vaelor searched the web instead.".format(" ".join(str(error).split())[:120])
            )
            return result

    def _discover(
        self, words: List[str], name: str, result: Dict[str, Any], deadline: float,
    ) -> Dict[str, Any]:
        repos = self._github_search(words, deadline)
        images = self._docker_search(words, deadline)
        prefixes = _name_prefixes(words)
        repo_matches = self._standing_matches(repos, prefixes)
        shorter = _matched_words(words, prefixes, repo_matches or _rank(
            [repo for repo in repos if _identity(prefixes, repo["owner"], repo["name"])],
            prefixes, "stars"))
        if shorter:
            # Only a leading run of the words names a repository. GitHub ranks
            # the whole phrase differently - measured: a name plus one
            # descriptive word surfaced a copy of the project, not the project -
            # so search for that run alone (one more search, inside 10 a minute).
            repos = self._github_search(shorter, deadline)
            repo_matches = self._standing_matches(repos, prefixes)
        image_matches = _rank(
            [image for image in images if _identity(prefixes, image["namespace"], image["name"])],
            prefixes, "pulls",
        )
        if not repo_matches:
            # No repository with standing names the app, so nothing here can
            # prove an image: what Docker Hub has is the owner's question, never
            # a pick by pull count (VD-222 owner ruling).
            result["candidates"] = (
                [_candidate(match, "pulls") for match in image_matches[:4]]
                or _near_candidates(name, [r for r in repos if self._has_standing(r)], images)
            )
            if result["candidates"]:
                result["status"] = "ambiguous"
            return result
        rivals = _rivals(repo_matches, "owner", "stars")
        if rivals:
            result["status"] = "ambiguous"
            result["candidates"] = rivals
            return result
        repository = repo_matches[0]
        linked = self._repository_detail(repository, prefixes, deadline, result["notes"])
        image = _proven_image(repository, linked, image_matches)
        result["repository"] = repository
        if image is None:
            # The project is known but names no image it publishes: its pages
            # become evidence and the web search looks further (VD-222 item 1).
            result["status"] = "repository"
            result["notes"].append(
                "{} on GitHub does not say which container image it publishes, so "
                "Vaelor searched the web as well.".format(repository["full_name"]))
            result["source_urls"], _references = _evidence(repository, None)
            return result
        image = self._confirm_tag(image, repository, deadline, result["notes"])
        result.update(status="found", image=image)
        result["source_urls"], result["image_reference"] = _evidence(repository, image)
        return result

    def _has_standing(self, repo: Mapping[str, Any]) -> bool:
        """A repository counts only with real standing: starred and not new.

        Anyone can create an empty repository named after an application; one
        that does not clear both bars proves nothing (VD-222 owner ruling).
        """
        try:
            created = datetime.strptime(str(repo.get("created_at", "")), "%Y-%m-%dT%H:%M:%SZ")
        except ValueError:
            return False
        age_days = (self._clock() - created.replace(tzinfo=timezone.utc).timestamp()) / 86400
        return repo.get("stars", 0) >= MIN_STANDING_STARS and age_days >= MIN_STANDING_DAYS

    def _standing_matches(
        self, repos: List[Dict[str, Any]], prefixes: List[str],
    ) -> List[Dict[str, Any]]:
        return _rank(
            [repo for repo in repos
             if self._has_standing(repo) and _identity(prefixes, repo["owner"], repo["name"])],
            prefixes, "stars",
        )

    def _github_search(self, words: List[str], deadline: float) -> List[Dict[str, Any]]:
        url = "{}/search/repositories?q={}&per_page=10".format(
            GITHUB_API, quote(" ".join(words)))
        document = self._get(url, SEARCH_CACHE_SECONDS, deadline) or {}
        repos = []
        for item in document.get("items", [])[:10]:
            if not isinstance(item, dict) or item.get("fork") or item.get("archived"):
                continue
            full_name = str(item.get("full_name", ""))
            owner, _slash, repo = full_name.partition("/")
            if not owner or not repo:
                continue
            repos.append({
                "where": GITHUB, "full_name": full_name, "owner": owner, "name": repo,
                "stars": int(item.get("stargazers_count") or 0),
                "html_url": "https://github.com/{}".format(full_name),
                "description": str(item.get("description") or "")[:200],
                "homepage": str(item.get("homepage") or "")[:200],
                "created_at": str(item.get("created_at") or "")[:40],
            })
        return repos

    def _docker_search(self, words: List[str], deadline: float) -> List[Dict[str, Any]]:
        url = "{}/search/repositories/?query={}&page_size=25".format(
            DOCKER_HUB_API, quote("-".join(words)))
        document = self._get(url, SEARCH_CACHE_SECONDS, deadline) or {}
        images = []
        for item in document.get("results", [])[:25]:
            if not isinstance(item, dict):
                continue
            namespace, _slash, repo = str(item.get("repo_name", "")).partition("/")
            if not repo:
                namespace, repo = "library", namespace
            if parse_image_reference("{}/{}".format(namespace, repo)) is None:
                continue
            images.append({
                "where": DOCKER_HUB, "full_name": "{}/{}".format(namespace, repo),
                "namespace": namespace, "name": repo,
                "pulls": int(item.get("pull_count") or 0),
                "official": bool(item.get("is_official")),
            })
        return images

    def _repository_detail(
        self, repository: Dict[str, Any], prefixes: List[str], deadline: float,
        notes: List[str],
    ) -> List[Any]:
        """Fetch the README and latest release; return the images the README names."""
        base = "{}/repos/{}".format(GITHUB_API, repository["full_name"])
        linked: List[Any] = []
        try:
            readme = self._get(base + "/readme", DETAIL_CACHE_SECONDS, deadline)
            if isinstance(readme, dict):
                download = str(readme.get("download_url") or "")
                if download.startswith("https://raw.githubusercontent.com/"):
                    repository["readme_url"] = download
                linked = readme_images(_readme_text(readme), repository, prefixes)
            release = self._get(base + "/releases/latest", DETAIL_CACHE_SECONDS, deadline)
            if isinstance(release, dict) and release.get("tag_name"):
                repository["release"] = {
                    "tag": str(release["tag_name"])[:128],
                    "url": str(release.get("html_url") or "")[:300],
                    "published_at": str(release.get("published_at") or "")[:40],
                }
        except SourceLimitReached as limit:
            notes.append(str(limit) + " Vaelor used the repository without its README or release.")
        except ApplicationResearchError:
            notes.append("The README or latest release of {} could not be read.".format(
                repository["full_name"]))
        return linked

    def _confirm_tag(
        self, image: Dict[str, Any], repository: Dict[str, Any] | None,
        deadline: float, notes: List[str],
    ) -> Dict[str, Any]:
        """Settle on a Docker Hub tag that exists and runs on this machine."""
        reference = image["reference"]
        if reference.registry != "docker.io":
            return image
        release = (repository or {}).get("release", {}).get("tag", "")
        wanted = list(dict.fromkeys(tag for tag in (
            reference.tag, "latest", release, release[1:] if release[:1] in "vV" else "",
        ) if tag))
        namespace, repo = reference.path.split("/", 1)
        base = "{}/namespaces/{}/repositories/{}/tags/".format(DOCKER_HUB_API, namespace, repo)
        try:
            for tag in wanted:
                document = self._get(base + quote(tag, safe=""), DETAIL_CACHE_SECONDS, deadline)
                architectures = _architectures(document)
                if self._target in architectures:
                    return {**image, "reference": parse_image_reference(
                        "{}:{}".format(reference.path, tag)), "architectures": architectures,
                        "architecture_confirmed": True}
        except SourceLimitReached as limit:
            notes.append(str(limit))
            return image
        except ApplicationResearchError:
            return image
        notes.append("Docker Hub shows no {} build of {} under the tags {}.".format(
            self._target, reference.path, ", ".join(wanted)))
        return image


def _proven_image(
    repository: Mapping[str, Any], linked: List[Any], matches: List[Dict[str, Any]],
) -> Dict[str, Any] | None:
    """The image the project itself proves, or ``None``.

    Proof is the README naming the image, or a Docker Hub image under the
    repository owner's own name. Pull count only orders images the README
    names; it never chooses one on its own (VD-222 owner ruling).
    """
    pulls = {"docker.io/" + match["full_name"]: match["pulls"] for match in matches}
    owner = _compact(repository["owner"])
    if linked:
        reference = sorted(linked, key=lambda ref: (
            _compact(ref.namespace) != owner, ref.registry != "docker.io",
            -pulls.get(ref.registry + "/" + ref.path, 0)))[0]
        return {"reference": reference, "proof": "readme",
                "pulls": pulls.get(reference.registry + "/" + reference.path)}
    own = [match for match in matches if _compact(match["namespace"]) == owner]
    if own:
        return {"reference": parse_image_reference(own[0]["full_name"]), "proof": "owner",
                "pulls": own[0]["pulls"]}
    return None


def _name_prefixes(words: List[str]) -> List[str]:
    """The whole name first, then shorter leading runs of two words or more.

    "Acme PDF editor" still finds a repository called Acme-PDF; the whole name
    always ranks above a shorter run. A single leading word is never enough:
    measured live, a misspelled two-word name otherwise matched an unrelated
    image named after its first word alone.
    """
    prefixes = []
    for length in range(len(words), 0, -1):
        if length < min(2, len(words)):
            break
        prefix = "".join(words[:length])
        if len(prefix) >= 3 and prefix not in prefixes:
            prefixes.append(prefix)
    return prefixes


def _matched_words(
    words: List[str], prefixes: List[str], matches: List[Dict[str, Any]],
) -> List[str]:
    """The leading words that named the best match, when not all of them did."""
    if not matches:
        return []
    level = _identity(prefixes, matches[0]["owner"], matches[0]["name"])
    if level >= len(prefixes):
        return []
    matched = prefixes[len(prefixes) - level]
    for length in range(1, len(words)):
        if "".join(words[:length]) == matched:
            return words[:length]
    return []


def _identity(prefixes: List[str], owner: str, name: str) -> int:
    """How well ``owner/name`` is named by the request: 0 none, higher better."""
    forms = {_compact(name), _compact(owner), _compact(owner + name)}
    for rank, prefix in enumerate(prefixes):
        if prefix in forms:
            return len(prefixes) - rank
    return 0


def _rank(items: List[Dict[str, Any]], prefixes: List[str], weight: str) -> List[Dict[str, Any]]:
    """Best-named first; only the best naming level is kept; then by popularity."""
    if not items:
        return []
    owner_key = "owner" if weight == "stars" else "namespace"
    scored = [(_identity(prefixes, item[owner_key], item["name"]), item) for item in items]
    best = max(score for score, _item in scored)
    kept = [item for score, item in scored if score == best]
    return sorted(kept, key=lambda item: -item[weight])


def _rivals(matches: List[Dict[str, Any]], owner_key: str, weight: str) -> List[Dict[str, Any]]:
    """Candidates under different owners close enough to the leader to ask about."""
    if len(matches) < 2:
        return []
    leader = matches[0]
    rivals = [
        match for match in matches[1:]
        if _compact(match[owner_key]) != _compact(leader[owner_key])
        and match[weight] >= AMBIGUOUS_SHARE * max(1, leader[weight])
    ]
    return [_candidate(leader, weight)] + [_candidate(m, weight) for m in rivals[:3]] if rivals else []


def _candidate(item: Mapping[str, Any], weight: str) -> Dict[str, Any]:
    return {"where": item["where"], "name": item["full_name"], weight: item[weight]}


def _near_candidates(
    name: str, repos: List[Dict[str, Any]], images: List[Dict[str, Any]],
) -> List[Dict[str, Any]]:
    """Well-used names close to a possibly misspelled request, most-used first."""
    def close(item: Mapping[str, Any]) -> bool:
        return difflib.SequenceMatcher(None, name, _compact(item["name"])).ratio() >= NEAR_NAME_RATIO

    near = [_candidate(r, "stars") for r in repos if close(r)]
    near += [_candidate(i, "pulls") for i in images if i["pulls"] >= NEAR_MIN_PULLS and close(i)]
    near.sort(key=lambda item: -(item.get("pulls") or item.get("stars") or 0))
    return near[:4]


def _readme_text(readme: Mapping[str, Any]) -> str:
    if str(readme.get("encoding", "")).lower() != "base64":
        return ""
    try:
        return base64.b64decode(str(readme.get("content", "")), validate=False).decode(
            "utf-8", "replace")[:200_000]
    except (binascii.Error, ValueError):
        return ""


def _prefixed_images(text: str) -> List[str]:
    """Registry-qualified image names (Docker Hub and GitHub package links
    included) in ``text``."""
    found = ["ghcr.io/{}/{}".format(match.group(1), match.group(2))
             for match in _GITHUB_PACKAGE.finditer(text)]
    for match in _PREFIXED_IMAGE.finditer(text):
        registry = "ghcr.io/" if match.group(0).lower().startswith("ghcr.io") else ""
        found.append("{}{}/{}{}".format(
            registry, match.group(1), match.group(2),
            ":" + match.group(3) if match.group(3) else ""))
    return found


def readme_images(text: str, repository: Mapping[str, Any], prefixes: List[str]) -> List[Any]:
    """Image references a README names for its own project, in README order.

    A README also names images it merely depends on (a database, a proxy), so a
    reference counts only when its owner is the repository's owner or its name
    is the application's name.
    """
    found: List[Any] = []
    candidates: List[str] = []
    # The commands a README tells you to run come first, so the tag it
    # documents (a major version, say) wins over a badge link to the same image.
    for line in text.splitlines():
        if _IMAGE_LINE.search(line):
            candidates.extend(_prefixed_images(line))
            candidates.extend(match.group(0) for match in _BARE_IMAGE.finditer(line))
    candidates.extend(_prefixed_images(text))
    owner = _compact(repository.get("owner", ""))
    for value in candidates:
        reference = parse_image_reference(value)
        if reference is None or "." in reference.namespace:
            continue
        own = _compact(reference.namespace) == owner
        if not (own or _identity(prefixes, reference.namespace, reference.repository)):
            continue
        if all(existing.path != reference.path or existing.registry != reference.registry
               for existing in found):
            found.append(reference)
    return found


def _architectures(document: Any) -> List[str]:
    if not isinstance(document, dict):
        return []
    images = document.get("images", [])
    return sorted({
        str(item.get("architecture")) for item in images
        if isinstance(item, dict) and item.get("os") == "linux" and item.get("architecture")
    })


def _evidence(
    repository: Mapping[str, Any] | None, image: Mapping[str, Any] | None,
) -> Tuple[List[str], List[str]]:
    """The pages research will cite, and the one image reference to prove."""
    urls: List[str] = []
    if repository:
        urls.append(repository["html_url"])
        if repository.get("readme_url"):
            urls.append(repository["readme_url"])
        release_url = str((repository.get("release") or {}).get("url", ""))
        if release_url.startswith("https://github.com/"):
            urls.append(release_url)
    references: List[str] = []
    if image and image.get("reference") is not None:
        reference = image["reference"]
        references.append(canonical_reference(reference))
        # Only the proven image's own registry URL: no guessed alternatives,
        # and a GHCR image stays a GHCR image (adversarial review M1).
        urls.extend(registry_candidate_urls([reference]))
    return list(dict.fromkeys(urls))[:6], references


def describe_candidates(candidates: Iterable[Mapping[str, Any]]) -> str:
    """"owner/name (GitHub, 1,234 stars), ..." - exactly what was found."""
    parts = []
    for item in candidates:
        if "pulls" in item:
            measure = "{:,} pulls".format(int(item["pulls"]))
        else:
            measure = "{:,} stars".format(int(item.get("stars", 0)))
        parts.append("{} ({}, {})".format(item["name"], item["where"], measure))
    return ", ".join(parts)


def direct_source_discovery(
    broker_client: Any, draft: Mapping[str, Any], progress: Any = None,
) -> Tuple[Dict[str, Any] | None, Dict[str, List[str]]]:
    """Control-plane side: ask the broker, then shape a research discovery.

    Returns ``(discovery, carry)``. ``discovery`` is set only when the project
    proved its own image; then research may run without a model (VD-222 owner
    ruling). Otherwise it is ``None`` and the web search runs, with ``carry``
    holding the plain sentences it must keep (a reached limit, say) and the
    project's own pages as extra sources. Several plausible matches stop
    research with the owner's question instead.
    """
    carry: Dict[str, List[str]] = {"notes": [], "sources": []}
    lookup = getattr(broker_client, "discover_sources", None)
    query = " ".join(str((draft.get("intent") or {}).get("application_query", "")).split())[:160]
    # A reviewed application researches from its pinned sources alone: no
    # discovery, no model-proposed or searched pages, no guessed registry URLs.
    # The service rejects any other source, so letting the model or the search
    # add one turned every reviewed request into a refusal (VD-222 live check;
    # the same refusal existed before VD-222 whenever a model was selected).
    from .application_research_service import ApplicationResearchService
    if ApplicationResearchService._reviewed(query):
        return {
            "application_name": query, "image_reference": [], "source_urls": [],
            "search_queries": [], "confidence": "high", "reviewed_sources": True,
            "limitations": ["Vaelor used its reviewed list of official sources for this application."],
        }, carry
    if not callable(lookup) or not query:
        return None, carry
    emit(progress, "searching", 30, "Looking for the application on GitHub and Docker Hub")
    try:
        found = lookup(query)
    except (OSError, ValueError) as error:
        carry["notes"].append("GitHub and Docker Hub could not be asked ({}).".format(
            " ".join(str(error).split())[:160]))
        return None, carry
    if not isinstance(found, dict):
        return None, carry
    carry["notes"] = [str(note) for note in found.get("notes", []) if str(note).strip()][:3]
    if found.get("status") == "ambiguous" and found.get("clarification"):
        raise ResearchWorkflowError(str(found["clarification"]), "discovery")
    sources = [str(url) for url in found.get("source_urls", [])][:6]
    if found.get("status") == "repository":
        carry["sources"] = sources
    image = found.get("image") or {}
    if found.get("status") != "found" or not sources or not found.get("image_reference"):
        return None, carry
    repository = found.get("repository") or {}
    return {
        "application_name": str(repository.get("name") or query),
        "image_reference": [str(value) for value in found["image_reference"]][:1],
        "source_urls": sources,
        "search_queries": [],
        # High only for an image the project proves AND that has a build for
        # this machine; a GHCR image's architectures are proven later.
        "confidence": "high" if image.get("architecture_confirmed") else "medium",
        "limitations": carry["notes"],
        "direct_sources": True,
    }, carry


def image_provenance(
    found: Mapping[str, Any] | None, reviewed: bool, images: Iterable[Mapping[str, Any]],
) -> Dict[str, Any]:
    """Where the proposed image came from, and every warning, for the review
    screen (VD-222 owner ruling: it always shows both).

    ``found`` is this box's direct-source answer for the same request, if any;
    ``images`` are the images the manifest actually proved.
    """
    if reviewed:
        return {"image_source": "From Vaelor's reviewed list of official sources for "
                                "this application.", "warnings": []}
    found = found or {}
    warnings = [str(note)[:300] for note in found.get("notes", []) if str(note).strip()][:4]
    repository = found.get("repository") or {}
    image = found.get("image") or {}
    proven = {str(item.get("repository", "")) for item in images}
    reference = parse_image_reference(image.get("reference", ""))
    named = reference is not None and (
        reference.path in proven or "{}/{}".format(reference.registry, reference.path) in proven)
    if found.get("status") == "found" and named and repository:
        where = "github.com/{} ({:,} stars)".format(
            repository.get("full_name", ""), int(repository.get("stars") or 0))
        source = (
            "Named in the README of {}.".format(where) if image.get("proof") == "readme"
            else "Published under the project owner's own name, by {}.".format(where)
        )
        return {"image_source": source, "warnings": warnings}
    return {
        "image_source": "Chosen by research from the sources listed below; the "
                        "project's own GitHub and Docker Hub pages did not prove it.",
        "warnings": warnings,
    }
