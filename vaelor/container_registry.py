"""Generic container-image reference resolution and registry digest proofs.

This module is deliberately application-agnostic. Every function here operates
on an arbitrary image reference or homepage; it holds NO per-app table and no
app literals (the OWNER DIRECTIVE on no hardcoded app data, #247t). Its job is
to turn "the official image is probably X" into the public registry API URL
whose digest-attested document the research service already knows how to
verify, so an *unreviewed* application can prove a digest-pinned image exactly
the way a reviewed one does.

No network, filesystem, shell, credential, or Docker authority lives here: the
functions are pure string and JSON transforms. The research service performs
the bounded, SSRF-hardened HTTPS fetches; this module only says which public
URLs to fetch and how to read the resulting registry document.

Two registries are supported, mirroring the digest-proof model already in
``application_research_service``:

* Docker Hub - :func:`docker_hub_tag_url` builds the tag-API URL that
  ``_DOCKER_TAG_API`` + ``_images()`` fetch and parse for a per-arch digest.
* GHCR - an OCI registry needing an anonymous pull token. :func:`ghcr_token_url`
  and :func:`ghcr_manifest_url` name the two hops, and
  :func:`proven_image_from_manifest_list` reads the manifest LIST the second hop
  returns, selecting the linux/<target-architecture> entry's ``sha256:`` digest.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Any, Iterable, List, Mapping
from urllib.parse import urlsplit


DOCKER_HUB_REGISTRY = "docker.io"
GHCR_REGISTRY = "ghcr.io"

# The Accept the GHCR manifest hop must send to receive a multi-arch manifest
# LIST (whose entries carry per-platform digests) rather than a single manifest.
GHCR_MANIFEST_ACCEPT = (
    "application/vnd.oci.image.index.v1+json, "
    "application/vnd.docker.distribution.manifest.list.v2+json"
)

# A single lowercase path component of an image repository (OCI distribution
# grammar): a run of alphanumerics, optionally joined by single separators.
_COMPONENT = r"[a-z0-9]+(?:[._-][a-z0-9]+)*"
_PATH = re.compile(r"^{c}(?:/{c})*$".format(c=_COMPONENT))
_TAG = re.compile(r"^[A-Za-z0-9_][A-Za-z0-9._-]{0,127}$")
_DIGEST = re.compile(r"^sha256:[0-9a-f]{64}$")

# The GHCR manifest URL the service resolves through the anonymous OCI
# token+manifest flow. The full repository path (owner plus image, possibly
# multi-segment) is captured so the token scope and repository name can be
# rebuilt from the URL alone on the service side.
GHCR_MANIFEST_API = re.compile(
    r"^https://ghcr\.io/v2/(?P<path>[a-z0-9]+(?:[._/-][a-z0-9]+)*)"
    r"/manifests/(?P<tag>[A-Za-z0-9._-]+)$"
)

# GitHub project homepage -> owner/repo, the generic fallback used when the
# model did not name an image reference. A GitHub repo commonly publishes to
# ghcr.io/<owner>/<repo> and/or docker.io/<owner>/<repo>.
_GITHUB_REPO = re.compile(
    r"^/(?P<owner>[a-z0-9][a-z0-9-]*)/(?P<repo>[a-z0-9][a-z0-9._-]*?)(?:\.git)?/?$"
)

# Docker Hub host aliases that all mean the canonical public registry.
_DOCKER_HUB_ALIASES = {
    "docker.io", "registry-1.docker.io", "index.docker.io", "registry.hub.docker.com",
}


@dataclass(frozen=True)
class ImageReference:
    """A normalized, registry-qualified image reference.

    ``path`` is the full repository path (Docker Hub is always exactly
    ``namespace/repository``; GHCR may be deeper, e.g. ``owner/image``).
    """

    registry: str
    path: str
    tag: str

    @property
    def namespace(self) -> str:
        return self.path.split("/", 1)[0]

    @property
    def repository(self) -> str:
        parts = self.path.split("/", 1)
        return parts[1] if len(parts) > 1 else parts[0]


def parse_image_reference(value: Any) -> ImageReference | None:
    """Normalize an arbitrary image reference, or return ``None`` if unusable.

    Handles a bare name (``nginx`` -> ``docker.io/library/nginx:latest``), a
    Docker Hub ``namespace/repository``, a registry-qualified reference
    (``ghcr.io/owner/image``), an explicit ``:tag`` (default ``latest``), and a
    trailing ``@sha256:...`` digest (dropped; a directed lookup re-proves it).
    Never invents a registry it was not given, and never special-cases an app.
    """
    text = str(value or "").strip().lower()
    if not text or len(text) > 400 or " " in text:
        return None
    text = text.split("@", 1)[0]  # a caller-supplied digest is re-proven, not trusted
    registry = DOCKER_HUB_REGISTRY
    first, slash, rest = text.partition("/")
    if slash and ("." in first or ":" in first or first == "localhost"):
        registry, remainder = first, rest
    else:
        remainder = text
    if registry in _DOCKER_HUB_ALIASES:
        registry = DOCKER_HUB_REGISTRY
    prefix, _slash, last = remainder.rpartition("/")
    tag = "latest"
    if ":" in last:
        name, _colon, candidate = last.rpartition(":")
        if not candidate or not _TAG.fullmatch(candidate):
            return None
        last, tag = name, candidate
    path = "{}/{}".format(prefix, last) if prefix else last
    if registry == DOCKER_HUB_REGISTRY and "/" not in path:
        path = "library/{}".format(path)
    if not path or not _PATH.fullmatch(path):
        return None
    # A Docker Hub repository is always exactly namespace/repository; a deeper
    # path is not a real Docker Hub image and must not be fabricated into one.
    if registry == DOCKER_HUB_REGISTRY and path.count("/") != 1:
        return None
    return ImageReference(registry=registry, path=path, tag=tag)


def canonical_reference(reference: ImageReference) -> str:
    """A stable round-trippable string form for carrying a reference by value."""
    return "{}/{}:{}".format(reference.registry, reference.path, reference.tag)


def docker_hub_tag_url(reference: ImageReference) -> str | None:
    """Build the Docker Hub tag-API URL, or ``None`` for a non-Docker-Hub ref.

    The form is the ``namespaces/.../repositories/...`` shape that the service's
    ``_DOCKER_TAG_API`` matches and ``_images()`` parses; a ``library`` image
    uses namespace ``library``.
    """
    if reference.registry != DOCKER_HUB_REGISTRY or reference.path.count("/") != 1:
        return None
    namespace, repository = reference.path.split("/", 1)
    return (
        "https://hub.docker.com/v2/namespaces/{}/repositories/{}/tags/{}".format(
            namespace, repository, reference.tag
        )
    )


def ghcr_manifest_url(reference: ImageReference) -> str | None:
    """Build the GHCR manifest URL, or ``None`` for a non-GHCR reference."""
    if reference.registry != GHCR_REGISTRY:
        return None
    return "https://ghcr.io/v2/{}/manifests/{}".format(reference.path, reference.tag)


def ghcr_token_url(repository_path: str) -> str:
    """Build the anonymous GHCR pull-token URL for a repository path."""
    return "https://ghcr.io/token?scope=repository:{}:pull".format(repository_path)


def registry_candidate_urls(references: Iterable[ImageReference]) -> List[str]:
    """Map image references to the public registry URLs that prove their digest.

    Docker Hub references become a tag-API URL (fetched and parsed by the
    service's existing ``_images()``); GHCR references become a manifest URL the
    service resolves via the OCI token+manifest flow. Order is preserved and
    duplicates dropped; unusable references are skipped.
    """
    urls: List[str] = []
    for reference in references:
        url = docker_hub_tag_url(reference) or ghcr_manifest_url(reference)
        if url and url not in urls:
            urls.append(url)
    return urls


def homepage_image_candidates(homepage: Any) -> List[ImageReference]:
    """Derive candidate official image refs from a GitHub project homepage.

    ``github.com/<owner>/<repo>`` yields ``ghcr.io/<owner>/<repo>``,
    ``docker.io/<owner>/<repo>`` and ``docker.io/library/<repo>`` - the places a
    project of that name most often publishes. These are only candidates: a
    reference that names no real image simply fails to fetch and degrades to an
    honest "unverified", never to a fabricated digest.
    """
    parsed = urlsplit(str(homepage or "").strip().lower())
    if parsed.hostname not in ("github.com", "www.github.com"):
        return []
    match = _GITHUB_REPO.match(parsed.path or "")
    if not match:
        return []
    owner, repo = match.group("owner"), match.group("repo")
    candidates: List[ImageReference] = []
    for raw in (
        "{}/{}/{}".format(GHCR_REGISTRY, owner, repo),
        "{}/{}".format(owner, repo),
        "library/{}".format(repo),
    ):
        reference = parse_image_reference(raw)
        if reference is not None and reference not in candidates:
            candidates.append(reference)
    return candidates


def proven_image_from_manifest_list(
    manifests: Any, target_architecture: str, repository: str, source_url: str,
) -> dict | None:
    """Select the linux/<target-architecture> entry of a manifest LIST.

    Returns the same normalized "proven image" shape ``_images()`` produces -
    ``service``, ``repository`` (``ghcr.io/<owner>/<image>``), ``digest``, the
    matched ``architectures``, and the ``source_url`` - or ``None`` when no
    entry pins that architecture. Only a real ``sha256:`` digest carried by the
    registry is accepted; nothing is fabricated.
    """
    if not isinstance(manifests, list):
        return None
    for entry in manifests:
        if not isinstance(entry, Mapping):
            continue
        platform = entry.get("platform")
        if not isinstance(platform, Mapping) or platform.get("os") != "linux":
            continue
        architecture = str(platform.get("architecture", ""))
        variant = str(platform.get("variant", ""))
        normalized = "arm/v7" if architecture == "arm" and variant == "v7" else architecture
        digest = str(entry.get("digest", ""))
        if normalized == target_architecture and _DIGEST.fullmatch(digest):
            return {
                "service": "app",
                "repository": repository,
                "digest": digest,
                "architectures": [normalized],
                "source_url": source_url,
            }
    return None
