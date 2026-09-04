"""Server-owned source selection and evidence-to-manifest normalization."""

from __future__ import annotations

import re
import platform
from typing import Any, Callable, Dict, Iterable, List, Mapping
from urllib.parse import urlsplit

from .application_research import ApplicationResearchBroker, ApplicationResearchError
from .application_research_prompt import interpret_research_evidence
from .container_registry import (
    GHCR_MANIFEST_ACCEPT,
    GHCR_MANIFEST_API,
    canonical_reference,
    ghcr_token_url,
    parse_image_reference,
    proven_image_from_manifest_list,
)
from .research_provenance import FetchProvenance


_DIGEST = re.compile(r"^sha256:[0-9a-f]{64}$")
_DOCKER_TAG_API = re.compile(
    r"^https://hub\.docker\.com/v2/(?:namespaces/)?"
    r"(?P<namespace>[a-z0-9._-]+)/repositories/"
    r"(?P<repository>[a-z0-9._-]+)/tags/(?P<tag>[A-Za-z0-9._-]+)$"
)

_REVIEWED = {
    "uptime kuma": {
        "id": "uptime-kuma",
        "name": "Uptime Kuma",
        "summary": "A self-hosted service and endpoint monitor.",
        "homepage": "https://github.com/louislam/uptime-kuma",
        "license": "MIT",
        "sources": [
            "https://github.com/louislam/uptime-kuma",
            "https://hub.docker.com/v2/namespaces/louislam/repositories/uptime-kuma/tags/2",
        ],
        "publishers": ["github.com", "hub.docker.com"],
        "repositories": ["louislam/uptime-kuma"],
        "ports": [("web", "tcp", 3001, "Web interface")],
        "volumes": [("uptime-kuma-data", "/app/data")],
        "memory_bytes": 536870912,
        "storage_bytes": 2147483648,
    },
    "plex": {
        "id": "plex-media-server",
        "name": "Plex Media Server",
        "summary": "A private media library and streaming server.",
        "homepage": "https://github.com/plexinc/pms-docker",
        "license": "Proprietary application; container files GPL-2.0",
        "sources": [
            "https://github.com/plexinc/pms-docker",
            "https://hub.docker.com/v2/namespaces/plexinc/repositories/pms-docker/tags/latest",
        ],
        "publishers": ["github.com", "hub.docker.com"],
        "repositories": ["plexinc/pms-docker"],
        "ports": [("web", "tcp", 32400, "Plex web and API")],
        "volumes": [("plex-config", "/config")],
        "memory_bytes": 1073741824,
        "storage_bytes": 5368709120,
    },
    "palworld": {
        "id": "palworld-dedicated-server",
        "name": "Palworld Dedicated Server",
        "summary": "The official dedicated multiplayer server.",
        "homepage": "https://docs.palworldgame.com/getting-started/requirements/",
        "license": "Proprietary",
        "sources": [
            "https://docs.palworldgame.com/getting-started/requirements/",
            "https://raw.githubusercontent.com/pocketpairjp/palworld-dedicated-server-docker/main/compose/compose.yaml",
        ],
        "publishers": ["docs.palworldgame.com", "raw.githubusercontent.com"],
        "repositories": [],
        "ports": [("game", "udp", 8211, "Game traffic")],
        "volumes": [],
        "memory_bytes": 17179869184,
        "storage_bytes": 42949672960,
        "unsupported": (
            "Official sources do not identify a digest-pinned container image. "
            "Vaelor will not substitute an unreviewed community image."
        ),
    },
}


# --- #247x: grounded "ask the user when unsure" research state --------------
# When the research pipeline cannot proceed confidently (an image it cannot
# verify, several plausible official-image candidates it cannot disambiguate, or
# a discovery too thin to act on), it surfaces CONCRETE clarifying questions and
# gates progress on the answer - like a careful assistant - instead of only
# dead-ending in a recovery. The frontend keys its "Vaelor needs a bit more to
# proceed" block off CLARIFYING_QUESTIONS_MARKER and splits the questions on the
# same separator, so the marker and separator live here as the one source both
# ends share. Every question must be GROUNDED in what research actually found;
# this helper only formats them and never invents a candidate (LESSONS 8/11).
CLARIFYING_QUESTIONS_MARKER = "Vaelor needs a bit more to proceed"
_CLARIFY_SEP = " ||| "
# Stand-in when the request text is empty, shared by both clarifying builders so
# the phrase is written once (LESSONS 6).
_UNNAMED_APP = "this application"


def clarifying_research_message(context: str, questions: List[str]) -> str:
    """Format a grounded clarifying-questions research message (#247x).

    ``context`` is one grounded sentence naming what was found; ``questions`` are
    concrete, answerable questions built only from that. The marker leads so the
    frontend can detect the state, and the separator joins each question so the
    frontend can render them as a list.
    """
    lead = " ".join("{} {}".format(CLARIFYING_QUESTIONS_MARKER, context).split())
    grounded = [" ".join(str(question).split()) for question in questions if str(question).strip()]
    return _CLARIFY_SEP.join([lead, *grounded])


def ambiguous_image_clarification(
    query: str, image_references: Iterable[Any]
) -> str | None:
    """Clarifying message when official-image candidates conflict (#247x).

    Considers ONLY model-declared references (homepage-derived variants are one
    project, never an ambiguity). Two or more references under different owners
    is a genuine "which is official" question and returns a grounded message that
    names exactly the references found; otherwise returns ``None``. Nothing is
    invented.
    """
    references = []
    for value in image_references:
        reference = parse_image_reference(value)
        if reference is not None:
            references.append(reference)
    owners = list(dict.fromkeys(reference.namespace for reference in references))
    if len(owners) < 2:
        return None
    app = " ".join(str(query).split()) or _UNNAMED_APP
    listed = ", ".join(canonical_reference(reference) for reference in references[:4])
    return clarifying_research_message(
        "Research found more than one image that could be {}: {}. Vaelor will "
        "not guess which one is the official image.".format(app, listed),
        [
            "Which of these is the official image: {}?".format(listed),
            "If none is correct, what is the official image reference or the "
            "official documentation URL for {}?".format(app),
        ],
    )


def unresolved_source_clarification(query: str) -> str:
    """Clarifying message when research found nothing authoritative (#247x).

    Grounded in the request text only - it asks the user for the official image
    reference/documentation and the desired edition, and invents no candidate.
    """
    app = " ".join(str(query).split()) or _UNNAMED_APP
    return clarifying_research_message(
        "Research could not confirm an official image or authoritative source "
        "for {} on its own.".format(app),
        [
            "Do you have the official image reference (for example a Docker Hub "
            "repository or a ghcr.io image) or the official documentation URL "
            "for {}?".format(app),
            "Which edition or tag do you want (for example a stable release, "
            "latest, or a specific version)?",
        ],
    )


class ApplicationResearchService:
    """Resolve reviewed sources, fetch them safely, and emit one manifest."""

    def __init__(
        self,
        broker: ApplicationResearchBroker | None = None,
        target_architecture: str | None = None,
        model_interpreter: Callable[[Dict[str, Any]], Any] | None = None,
    ):
        self.broker = broker or ApplicationResearchBroker()
        raw_architecture = (target_architecture or platform.machine()).lower()
        self.target_architecture = {
            "aarch64": "arm64", "x86_64": "amd64", "armv7l": "arm/v7",
            "amd64": "amd64", "arm64": "arm64",
        }.get(raw_architecture, raw_architecture)
        # Optional and deliberately dependency-injected: this service never
        # grants an LLM network, shell, filesystem, or container authority.
        self.model_interpreter = model_interpreter

    def research_manifest(
        self, draft: Mapping[str, Any], source_urls: Iterable[str]
    ) -> Dict[str, Any]:
        query = str((draft.get("intent") or {}).get("application_query", "")).strip()
        spec = self._reviewed(query)
        requested = list(source_urls)
        if spec and requested:
            # A reviewed app is pinned to its own baked sources: any caller-
            # supplied source (a directed registry candidate included) that is
            # not one of them is rejected, never silently substituted. Reviewed
            # apps verify offline through this path with empty sources; the
            # generic directed lookup (#247t) targets UNREVIEWED apps only.
            allowed_sources = set(spec["sources"])
            if any(url not in allowed_sources for url in requested):
                raise ValueError(
                    "Reviewed applications can use only their approved official sources."
                )
        if not requested and spec:
            requested = list(spec["sources"])
        if not requested:
            raise ValueError(
                "This application is not in the reviewed source directory. "
                "Add official documentation and Docker Hub tag URLs."
            )
        results = []
        failures = []
        for url in requested:
            try:
                ghcr = GHCR_MANIFEST_API.fullmatch(url)
                if ghcr:
                    results.append(self._resolve_ghcr(url, ghcr.group("path")))
                else:
                    results.append(self.broker.research(url).to_dict())
            except ApplicationResearchError as error:
                failures.append({
                    "url": url,
                    "layer": "acquisition",
                    "error": " ".join(str(error).split())[:300],
                })
        if not results:
            detail = failures[0]["error"] if failures else "No source returned evidence."
            raise ValueError(
                "Vaelor selected {} research source{}, but none could be acquired: {} "
                "Retry later, use a stronger model to find alternate authoritative sources, "
                "or provide official sources in Advanced recovery."
                .format(len(requested), "" if len(requested) == 1 else "s", detail)
            )
        if spec:
            self._validate_reviewed_results(spec, results)
        synthesis = interpret_research_evidence(
            dict(draft.get("intent") or {}), results,
            target_architecture=self.target_architecture,
            model_call=self.model_interpreter,
        )
        return {
            "broker_results": results,
            "broker_failures": failures,
            "manifest": self._manifest(query, spec, results, synthesis),
        }

    def fetch_evidence(
        self, url: str, provenance: FetchProvenance | Mapping[str, Any] | None = None
    ) -> Dict[str, Any]:
        """Fetch one public source through the same SSRF-safe broker policy.

        `provenance` names the pages this run was authorized to read and is
        enforced on every redirect hop, so a vetted page cannot bounce the
        fetch to an unrelated host.
        """
        if provenance is not None and not isinstance(provenance, FetchProvenance):
            provenance = FetchProvenance.from_dict(provenance)
        return self.broker.research(str(url), provenance).to_dict()

    @staticmethod
    def _reviewed(query: str) -> Dict[str, Any] | None:
        lower = re.sub(r"[^a-z0-9]+", " ", query.lower()).strip()
        for key, spec in _REVIEWED.items():
            if key in lower:
                return dict(spec)
        return None

    @staticmethod
    def _validate_reviewed_results(
        spec: Mapping[str, Any], results: List[Dict[str, Any]]
    ) -> None:
        allowed_sources = set(spec.get("sources", []))
        allowed_publishers = set(spec.get("publishers", []))
        for result in results:
            source = result.get("source", {})
            requested_url = str(source.get("requested_url", ""))
            final_url = str(source.get("final_url", ""))
            hostname = (urlsplit(final_url).hostname or "").lower()
            if requested_url not in allowed_sources or hostname not in allowed_publishers:
                raise ValueError(
                    "A reviewed source changed publisher identity and was rejected."
                )

    def _manifest(
        self,
        query: str,
        spec: Dict[str, Any] | None,
        results: List[Dict[str, Any]],
        synthesis: Mapping[str, Any] | None = None,
    ) -> Dict[str, Any]:
        spec = spec or self._generic_spec(query, results, synthesis)
        allowed_repositories = spec.get("repositories") if spec else None
        # A directed registry lookup proves at most one image for the app: prefer
        # the Docker Hub tags-API proof, else the GHCR manifest-list proof (#247t).
        images = self._images(results, allowed_repositories) or self._ghcr_images(results)
        images = images[:1]
        architectures = sorted({
            architecture
            for image in images
            for architecture in image["architectures"]
        })
        unsupported = str(spec.get("unsupported", ""))
        if unsupported:
            status, reason = "unsupported", unsupported
            architectures = architectures or ["amd64"]
            images = []
        elif images and self.target_architecture in architectures:
            status = "verified"
            reason = (
                "A digest-pinned Linux {} image was verified in registry metadata."
                .format(self.target_architecture)
            )
        else:
            status = "unknown"
            # The architecture is the real host target, not a hardcoded "ARM64"
            # (#247p): on the Z2 this reads "amd64", on a Pi "arm64", exactly as
            # the verified branch above interpolates it. The cause is that no
            # registry source proved a digest-pinned image, which the phrase
            # "could be verified from the available sources" names verbatim so
            # the frontend can offer web-research/source recovery only for this
            # recoverable case and not for a genuine incompatibility (#247q).
            reason = (
                "No digest-pinned Linux {} image could be verified from the "
                "available sources.".format(self.target_architecture)
            )
            architectures = architectures or [self.target_architecture]
            images = []
        service = images[0]["service"] if images else "app"
        return {
            "application": {
                "id": spec["id"],
                "name": spec["name"],
                "summary": spec["summary"],
                "homepage": spec["homepage"],
                "license": spec["license"],
            },
            "compatibility": {
                "status": status,
                "architectures": architectures,
                "reason": reason,
            },
            "images": images,
            "ports": [
                {
                    "service": service,
                    "name": name,
                    "protocol": protocol,
                    "target": port,
                    "published": port,
                    "required": True,
                }
                for name, protocol, port, _purpose in spec.get("ports", [])
            ] if images else [],
            "volumes": [
                {
                    "service": service,
                    "name": name,
                    "mount_path": target,
                    "mode": "rw",
                    "required": True,
                }
                for name, target in spec.get("volumes", [])
            ] if images else [],
            "variables": [],
            "resources": {
                "memory_bytes": int(spec.get("memory_bytes", 536870912)),
                "cpu_cores": 1,
                "storage_bytes": int(spec.get("storage_bytes", 1073741824)),
            },
            "sources": [self._source(result) for result in results],
        }

    @staticmethod
    def _generic_spec(
        query: str, results: List[Dict[str, Any]],
        synthesis: Mapping[str, Any] | None = None,
    ) -> Dict[str, Any]:
        name = query[:120] or "Unidentified application"
        # Truncating to 64 chars can land the slice on a hyphen, and a slug
        # that ends in "-" fails `application_deployments._SAFE_ID` - which is
        # how a verbose research request surfaced as a bare "Choose an
        # application id using lowercase letters, numbers, and hyphens." with no
        # id field on screen. Strip again *after* the slice so the derived id is
        # always valid, and never make validation reject what we generated.
        slug = re.sub(r"[^a-z0-9]+", "-", name.lower()).strip("-")[:64].strip("-")
        first = results[0]["source"]["final_url"]
        model_summary = ""
        if isinstance(synthesis, Mapping) and synthesis.get("source") == "selected-model":
            model_summary = str(synthesis.get("summary", "")).strip()[:800]
        return {
            "id": slug or "researched-application",
            "name": name,
            "summary": model_summary or "An application described by the supplied public sources.",
            "homepage": first,
            "license": "Unknown",
            "ports": [],
            "volumes": [],
        }

    @staticmethod
    def _source(result: Dict[str, Any]) -> Dict[str, Any]:
        source = result["source"]
        metadata = result["metadata"]
        return {
            "url": source["final_url"],
            "title": str(metadata.get("title") or source["final_url"])[:200],
            "kind": result["kind"],
            "sha256": "sha256:" + source["sha256"],
            "verified": True,
        }

    def _images(
        self,
        results: List[Dict[str, Any]],
        allowed_repositories: Iterable[str] | None = None,
    ) -> List[Dict[str, Any]]:
        repository_allowlist = (
            set(allowed_repositories) if allowed_repositories is not None else None
        )
        for result in results:
            source_url = result["source"]["final_url"]
            match = _DOCKER_TAG_API.fullmatch(source_url)
            document = result.get("metadata", {}).get("document")
            if not match or not isinstance(document, dict):
                continue
            candidates = document.get("images", [])
            if isinstance(candidates, dict):
                candidates = [candidates]
            if not isinstance(candidates, list):
                continue
            linux = [
                item for item in candidates
                if isinstance(item, dict)
                and item.get("os") == "linux"
                and item.get("architecture") in {"amd64", "arm64", "arm"}
                and _DIGEST.fullmatch(str(item.get("digest", "")))
            ]
            target_name = "arm" if self.target_architecture == "arm/v7" else self.target_architecture
            selected_target = next(
                (item for item in linux if item.get("architecture") == target_name),
                None,
            )
            selected = selected_target
            if selected is None:
                continue
            architecture = str(selected["architecture"])
            if architecture == "arm":
                architecture = "arm/v7"
            repository = f"{match.group('namespace')}/{match.group('repository')}"
            if repository_allowlist is not None and repository not in repository_allowlist:
                raise ValueError(
                    "Registry metadata did not match the reviewed application repository."
                )
            return [{
                "service": "app",
                "repository": repository,
                "digest": selected["digest"],
                "architectures": [architecture],
                "source_url": source_url,
            }]
        return []

    def _resolve_ghcr(self, manifest_url: str, repository_path: str) -> Dict[str, Any]:
        """Prove a GHCR image via the anonymous OCI token+manifest flow (#247t).

        GHCR is an OCI registry: an anonymous pull token is minted first, then
        the manifest LIST is fetched under a Bearer header and a manifest-list
        Accept. The bearer is scoped to this repository, fetched per request, and
        never a user credential; the transport sends it only to ghcr.io. The
        returned manifest-list document is normalized exactly like any other
        registry result so it also becomes an attributable, digest-bearing
        source. A registry error surfaces as an unverified source, not a crash.
        """
        token_result = self.broker.research(ghcr_token_url(repository_path)).to_dict()
        metadata = token_result.get("metadata", {})
        document = metadata.get("document") if isinstance(metadata, dict) else None
        token = ""
        if isinstance(document, dict):
            token = str(document.get("token") or document.get("access_token") or "")
        if not token:
            raise ApplicationResearchError(
                "The container registry did not issue an anonymous pull token."
            )
        return self.broker.research(
            manifest_url,
            request_headers={
                "Authorization": "Bearer {}".format(token),
                "Accept": GHCR_MANIFEST_ACCEPT,
            },
        ).to_dict()

    def _ghcr_images(self, results: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Extract proven images from any GHCR manifest-list result (#247t)."""
        images: List[Dict[str, Any]] = []
        for result in results:
            source_url = result["source"]["final_url"]
            match = GHCR_MANIFEST_API.fullmatch(source_url)
            metadata = result.get("metadata", {})
            if not match or not isinstance(metadata, dict):
                continue
            proven = proven_image_from_manifest_list(
                metadata.get("manifests"),
                self.target_architecture,
                "ghcr.io/{}".format(match.group("path")),
                source_url,
            )
            if proven is not None:
                images.append(proven)
        return images
