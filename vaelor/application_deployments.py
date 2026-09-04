"""Server-owned application deployment intents, manifests, and drafts.

Models may help identify an application, but they never provide executable
Compose.  This module is the trust boundary: research results are normalized,
Compose is generated from that schema, and only a digest-bound validated draft
can become a ``compose.import`` proposal.
"""

from __future__ import annotations

import hashlib
import json
import os
import re
import secrets
import sqlite3
import threading
import time
import unicodedata
from contextlib import closing
from pathlib import Path
from typing import Any, Dict, Optional
from urllib.parse import urlsplit, urlunsplit

try:
    import grp
except ImportError:  # Windows development and test hosts have no Unix groups.
    grp = None

from .compose_policy import PRIVILEGED_HOST_MOUNTS
from .runtime_paths import env_value, jobs_group_id, state_path


SCHEMA_VERSION = 1
DRAFT_STATES = {
    "research", "configuration", "validated", "approved", "imported", "failed",
}
COMPATIBILITY_STATES = {"verified", "unsupported", "unknown"}
ARCHITECTURES = {"amd64", "arm64", "arm/v7"}
PROTOCOLS = {"tcp", "udp"}
VOLUME_MODES = {"rw", "ro"}
_SAFE_ID = re.compile(r"^[a-z0-9](?:[a-z0-9-]{0,62}[a-z0-9])?$")
# The executor's ``_project_directory`` gate. A researched manifest id is only
# checked against ``_SAFE_ID`` (1-64 chars), which is *looser* at both ends than
# this: it admits a single character and up to 64, while the executor demands
# 2-48. ``compose_project_name`` is the single producer that closes that gap so
# a valid-for-research id can never be rejected at deploy as "invalid".
_PROJECT_NAME = re.compile(r"[a-z0-9][a-z0-9_-]{1,47}")
_SAFE_SERVICE = re.compile(r"^[a-z][a-z0-9_-]{0,62}$")
_SAFE_VARIABLE = re.compile(r"^[A-Z][A-Z0-9_]{0,79}$")
_IMAGE_REPOSITORY = re.compile(
    r"^(?:[a-z0-9.-]+(?::[0-9]{1,5})?/)?[a-z0-9]+"
    r"(?:[._/-][a-z0-9]+)*$"
)
_DIGEST = re.compile(r"^sha256:[0-9a-f]{64}$")
_CREDENTIAL_ID = re.compile(r"^cred_[A-Za-z0-9_-]{6,120}$")
_SECRET_TEXT = re.compile(
    r"(?:-----BEGIN [A-Z ]*PRIVATE KEY-----|\b(?:password|token|secret|api[_-]?key)\s*[:=])",
    re.IGNORECASE,
)

#: Refusal shown only when a discard would target the saved record of a
#: genuinely-live application. Owned here and surfaced verbatim by the discard
#: route (pairs-with the KNOWN_DUPLICATES entry of the same sentence) so the two
#: layers can never disagree about why a discard was blocked.
LIVE_DEPLOYMENT_DRAFT_DISCARD_REFUSED = (
    "A running application cannot have its saved research discarded; remove the "
    "application first."
)


# Single source of truth for application-change cues (LESSONS 6 / #247d /
# #247f). The deterministic minting gate (classify_application_intent) and the
# model-refinement gate (application_intent_refinement.application_change_signal)
# both read these, so "run" cannot be an action cue in one and absent from the
# other, and the two entry points cannot silently disagree about what a change
# request looks like.
APPLICATION_ACTION_CUES = frozenset({
    "deploy", "deployment", "host", "hosting", "install", "installation",
    "launch", "create", "run", "spin", "setup", "stream",
})
APPLICATION_NOUN_CUES = frozenset({
    "app", "application", "compose", "container", "docker", "game", "server",
    "service", "web", "website", "dashboard",
})
# Words that describe the mechanics or audience of the request rather than the
# application itself, dropped when forming a human-readable request summary
# (#247e / LESSONS 15). The action verbs, articles, and generic nouns are here,
# plus the connectives and social words ("and", "with", "for my friends") whose
# leftovers turned a name into the mangled "Minecraft friends." this replaces.
_QUERY_STOPWORDS = frozenset({
    "a", "an", "and", "also", "app", "application", "can", "compose",
    "container", "could", "create", "deploy", "deployment", "docker", "family",
    "for", "friend", "friends", "get", "getting", "going", "host", "hosting",
    "i", "install", "installation", "just", "kid", "kids", "launch", "let",
    "lets", "like", "me", "mine", "my", "need", "needs", "on", "or", "our",
    "play", "please", "plus", "run", "server", "service", "set", "setup", "so",
    "spin", "stream", "the", "them", "then", "their", "to", "together", "up",
    "us", "using", "via", "want", "we", "will", "with", "would",
})
# Interrogative openers. A message that opens with one (or ends with "?") and
# carries no direct imperative deploy verb is a question, not a change request:
# "Can I host a media server?" must decline cleanly rather than mint the
# stopword bag "Can media" as a deployment (#247c / LESSONS 1). The model, which
# is told that "whether it can run" questions are not deployments, decides such
# cases; the deterministic gate does not pre-empt it with a bogus intent.
_INTERROGATIVE_OPENERS = frozenset({
    "can", "could", "would", "should", "shall", "will", "do", "does", "did",
    "is", "are", "was", "were", "has", "have", "had", "may", "might",
    "how", "what", "why", "when", "where", "which", "who", "whom", "whose",
})
# An imperative deploy verb at or near the start of the message. Shared by the
# deterministic gate and by looks_like_pure_question so both agree on what a
# direct instruction looks like: "Deploy X?" ending in a question mark is still
# a request, while "Can I host X?" opening with an interrogative is not.
_DIRECT_ACTION = re.compile(
    r"^(?:(?:i|we)\s+(?:want|need|would like)\s+to\s+)?"
    r"(?:deploy|host|install|launch|run|set\s+up|setup)\b",
)


def looks_like_pure_question(message: str) -> bool:
    """True when the message reads as a plain question with no deploy verb.

    The intake uses this to decide what NOT to spend a model call on (#247y):
    an obvious question ("what is X?", "can this run X?") is the common
    non-deploy, and asking the intermittently-flaky local model to judge it is
    the mis-mint risk the intake guards against. It mirrors the deterministic
    gate's own interrogative rule exactly - opens with an interrogative or ends
    with "?", and carries no direct imperative deploy verb - so the fast decline
    and the deterministic decline never disagree about the same message.
    """
    lower = " ".join(str(message or "").lower().split())
    if not lower or _DIRECT_ACTION.match(lower):
        return False
    opener = lower.split(" ", 1)[0]
    return lower.endswith("?") or opener in _INTERROGATIVE_OPENERS


class ApplicationDeploymentError(ValueError):
    """Safe validation failure for application deployment state."""


def _canonical(value: Any) -> str:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False)


def content_digest(value: Any) -> str:
    return "sha256:" + hashlib.sha256(_canonical(value).encode("utf-8")).hexdigest()


def _text(value: Any, name: str, limit: int, *, required: bool = True) -> str:
    text = unicodedata.normalize("NFKC", str(value or "")).strip()
    if required and not text:
        raise ApplicationDeploymentError(f"Enter {name}.")
    if len(text) > limit:
        raise ApplicationDeploymentError(f"{name.capitalize()} is too long.")
    if any(unicodedata.category(char) in {"Cc", "Cf", "Cs"} for char in text):
        raise ApplicationDeploymentError(f"{name.capitalize()} contains unsafe characters.")
    if _SECRET_TEXT.search(text):
        raise ApplicationDeploymentError("Secrets do not belong in application research data.")
    return text


def _slug(value: Any, name: str = "an application id") -> str:
    slug = str(value or "").strip().lower()
    if not _SAFE_ID.fullmatch(slug):
        raise ApplicationDeploymentError(f"Choose {name} using lowercase letters, numbers, and hyphens.")
    return slug


def _project_slug(value: Any) -> str:
    """Slug one identity string to the executor's project-name shape, or "".

    Lowercases, transliterates away accents, folds every run of characters the
    executor forbids into a single hyphen, and trims separators. The result is
    truncated to the executor's 48-character ceiling and padded up from a lone
    surviving character to clear its 2-character floor, so any non-empty return
    already satisfies ``_PROJECT_NAME``. A source with no ASCII-alphanumeric
    content (e.g. a name written entirely in non-Latin script) yields "" and
    lets the caller fall through to the next identity source.
    """
    ascii_value = (
        unicodedata.normalize("NFKD", str(value or ""))
        .encode("ascii", "ignore").decode("ascii")
    )
    slug = re.sub(r"[^a-z0-9]+", "-", ascii_value.lower()).strip("-_")
    slug = slug[:48].rstrip("-_")
    if not slug:
        return ""
    if len(slug) < 2:
        slug = (slug + "-app")[:48]
    return slug


def compose_project_name(manifest: Any) -> str:
    """Derive an executor-valid, non-empty Compose project name from identity.

    This is the single producer of the ``compose.import`` project name for a
    researched draft. The manifest id passed research only against ``_SAFE_ID``,
    which is looser than the executor's ``_PROJECT_NAME`` gate at both length
    bounds, so passing the raw id straight through let a legitimate app fail at
    the final deploy step with "The Compose project name is invalid" and create
    no container (LESSONS: draft-config to deploy-job propagation). Preferring
    the app id, then its display name, this slugifies to guarantee the gate is
    satisfied; a deterministic digest-based fallback covers identities with no
    usable ASCII content so the return is always non-empty and valid.
    """
    application = manifest.get("application", {}) if isinstance(manifest, dict) else {}
    if not isinstance(application, dict):
        application = {}
    for source in (application.get("id"), application.get("name")):
        slug = _project_slug(source)
        if slug:
            return slug
    fallback = "app-" + content_digest(application).split(":", 1)[1]
    return fallback[:48]


def _service(value: Any) -> str:
    service = str(value or "").strip().lower()
    if not _SAFE_SERVICE.fullmatch(service):
        raise ApplicationDeploymentError("Choose a safe service name.")
    return service


def _integer(value: Any, name: str, minimum: int, maximum: int) -> int:
    if isinstance(value, bool):
        raise ApplicationDeploymentError(f"{name.capitalize()} must be a number.")
    try:
        result = int(value)
    except (TypeError, ValueError) as error:
        raise ApplicationDeploymentError(f"{name.capitalize()} must be a number.") from error
    if result < minimum or result > maximum:
        raise ApplicationDeploymentError(
            f"{name.capitalize()} must be between {minimum} and {maximum}."
        )
    return result


def _https_url(value: Any) -> str:
    url = _text(value, "a source URL", 2000)
    parsed = urlsplit(url)
    if parsed.scheme.lower() != "https" or not parsed.hostname:
        raise ApplicationDeploymentError("Application sources must use public HTTPS URLs.")
    if parsed.username or parsed.password:
        raise ApplicationDeploymentError("Credentials are not allowed in source URLs.")
    if parsed.fragment:
        parsed = parsed._replace(fragment="")
    return urlunsplit(parsed)


def _application_query(text: str) -> str:
    """Form a human-readable request summary from a change request.

    A bag of leftover stopwords with a stray period is not a summary: "run a
    Minecraft server for my friends" must yield "Minecraft", never the mangled
    "Minecraft friends." this replaces (#247e / LESSONS 15). Each token is
    stripped of surrounding punctuation, and the words that describe the
    request's mechanics or audience rather than the application are dropped;
    the meaningful remainder (the name, and any distinguishing descriptors such
    as "multi-node" or "OAuth") is kept in order. An empty result is handled by
    the caller's clearly-worded fallback rather than surfacing mangled text.
    """
    tokens = []
    for token in re.findall(r"[A-Za-z0-9][A-Za-z0-9_.+-]{0,79}", text):
        cleaned = token.strip("._+-")
        if cleaned and cleaned.lower() not in _QUERY_STOPWORDS:
            tokens.append(cleaned)
    return " ".join(tokens)[:160]


def classify_application_intent(message: str) -> Optional[Dict[str, Any]]:
    """Classify an explicit unknown-application change request.

    This deterministic gate intentionally refuses to classify ordinary
    questions. A model can enrich wording later, but cannot widen the gate.
    """
    text = _text(message, "what you want to deploy", 4000)
    lower = " ".join(text.lower().split())
    words = set(re.findall(r"[a-z0-9]+", lower))
    change = bool(words & APPLICATION_ACTION_CUES) or "set up" in lower
    application_words = APPLICATION_NOUN_CUES
    direct_action = _DIRECT_ACTION.match(lower) is not None
    if looks_like_pure_question(lower):
        return None
    non_application_targets = {
        "ai", "assistant", "driver", "firmware", "model", "models", "package",
        "packages", "update", "updates",
    }
    candidate_words = words - application_words - non_application_targets
    application = bool(words & application_words) or bool(
        direct_action and candidate_words - {
            "a", "an", "for", "i", "install", "my", "our", "please", "the",
            "to", "we", "want", "would",
        }
    )
    if not (change and application):
        return None
    query = _application_query(text) or "unknown application"
    delivery = "container" if words & {"compose", "container", "docker"} else "auto"
    return {
        "schema_version": SCHEMA_VERSION,
        "type": "application.deploy",
        "application_query": query,
        "delivery_method": delivery,
        "confidence": "high" if query != "unknown application" else "low",
        "research_required": True,
        "missing_inputs": [
            "verified container image and digest",
            "required ports and persistent storage",
            "resource limits and application settings",
        ],
    }


def normalize_manifest(raw: Any) -> Dict[str, Any]:
    """Return a bounded, source-attributed application manifest."""
    if not isinstance(raw, dict):
        raise ApplicationDeploymentError("Application research returned an invalid manifest.")
    app = raw.get("application")
    compatibility = raw.get("compatibility")
    if not isinstance(app, dict) or not isinstance(compatibility, dict):
        raise ApplicationDeploymentError("Application and compatibility details are required.")
    status = str(compatibility.get("status", "")).strip().lower()
    if status not in COMPATIBILITY_STATES:
        raise ApplicationDeploymentError("Choose a valid compatibility result.")
    architectures = sorted({str(item).strip().lower() for item in compatibility.get("architectures", [])})
    if not architectures or any(item not in ARCHITECTURES for item in architectures):
        raise ApplicationDeploymentError("List the verified supported architectures.")

    sources = []
    for item in raw.get("sources", []):
        if not isinstance(item, dict) or len(sources) >= 24:
            raise ApplicationDeploymentError("Application sources are invalid or too numerous.")
        digest = str(item.get("sha256", "")).strip().lower()
        if not _DIGEST.fullmatch(digest):
            raise ApplicationDeploymentError("Each source requires its captured SHA-256 digest.")
        sources.append({
            "url": _https_url(item.get("url")),
            "title": _text(item.get("title"), "a source title", 200),
            "kind": _text(item.get("kind"), "a source kind", 40).lower(),
            "sha256": digest,
            "verified": bool(item.get("verified", False)),
        })
    if not sources:
        raise ApplicationDeploymentError("At least one attributable research source is required.")

    images = []
    seen_services = set()
    for item in raw.get("images", []):
        if not isinstance(item, dict) or len(images) >= 12:
            raise ApplicationDeploymentError("Container image details are invalid or too numerous.")
        service = _service(item.get("service"))
        if service in seen_services:
            raise ApplicationDeploymentError("Each service may have only one pinned image.")
        repository = str(item.get("repository", "")).strip().lower()
        digest = str(item.get("digest", "")).strip().lower()
        if not _IMAGE_REPOSITORY.fullmatch(repository) or not _DIGEST.fullmatch(digest):
            raise ApplicationDeploymentError("Every image must use a safe repository and pinned SHA-256 digest.")
        image_arches = sorted({str(value).strip().lower() for value in item.get("architectures", [])})
        if not image_arches or any(value not in ARCHITECTURES for value in image_arches):
            raise ApplicationDeploymentError("Every image requires verified architecture metadata.")
        source_url = _https_url(item.get("source_url"))
        images.append({
            "service": service,
            "repository": repository,
            "digest": digest,
            "reference": f"{repository}@{digest}",
            "architectures": image_arches,
            "source_url": source_url,
        })
        seen_services.add(service)
    if status == "verified" and not images:
        raise ApplicationDeploymentError("A verified application requires at least one pinned image.")

    ports = []
    for item in raw.get("ports", []):
        if not isinstance(item, dict) or len(ports) >= 32:
            raise ApplicationDeploymentError("Application port details are invalid or too numerous.")
        protocol = str(item.get("protocol", "tcp")).strip().lower()
        if protocol not in PROTOCOLS:
            raise ApplicationDeploymentError("Only TCP and UDP application ports are supported.")
        ports.append({
            "service": _service(item.get("service")),
            "name": _text(item.get("name"), "a port name", 80),
            "protocol": protocol,
            "target": _integer(item.get("target"), "target port", 1, 65535),
            "published": _integer(item.get("published", item.get("target")), "published port", 1, 65535),
            "required": bool(item.get("required", True)),
        })

    volumes = []
    for item in raw.get("volumes", []):
        if not isinstance(item, dict) or len(volumes) >= 32:
            raise ApplicationDeploymentError("Application volume details are invalid or too numerous.")
        mount_path = _text(item.get("mount_path"), "a container mount path", 240)
        if not mount_path.startswith("/") or ".." in mount_path.split("/"):
            raise ApplicationDeploymentError("Container mount paths must be absolute and traversal-free.")
        mode = str(item.get("mode", "rw")).strip().lower()
        if mode not in VOLUME_MODES:
            raise ApplicationDeploymentError("Choose a supported volume mode.")
        volumes.append({
            "service": _service(item.get("service")),
            "name": _slug(item.get("name"), "a volume name"),
            "mount_path": mount_path,
            "mode": mode,
            "required": bool(item.get("required", True)),
        })

    variables = []
    for item in raw.get("variables", []):
        if not isinstance(item, dict) or len(variables) >= 64:
            raise ApplicationDeploymentError("Application variable details are invalid or too numerous.")
        name = str(item.get("name", "")).strip().upper()
        if not _SAFE_VARIABLE.fullmatch(name):
            raise ApplicationDeploymentError("Application variable names must use uppercase letters and underscores.")
        secret = bool(item.get("secret", False))
        normalized = {
            "service": _service(item.get("service")),
            "name": name,
            "required": bool(item.get("required", False)),
            "secret": secret,
            "description": _text(item.get("description", "Setting required by the application."), "a variable description", 300),
        }
        if not secret and item.get("default") is not None:
            normalized["default"] = _text(item.get("default"), "a variable default", 500, required=False)
        variables.append(normalized)

    resources = raw.get("resources", {})
    if not isinstance(resources, dict):
        raise ApplicationDeploymentError("Application resource guidance is invalid.")
    normalized = {
        "schema_version": SCHEMA_VERSION,
        "application": {
            "id": _slug(app.get("id")),
            "name": _text(app.get("name"), "an application name", 120),
            "summary": _text(app.get("summary"), "an application summary", 800),
            "homepage": _https_url(app.get("homepage")),
            "license": _text(app.get("license", "Unknown"), "a license", 120),
        },
        "compatibility": {
            "status": status,
            "architectures": architectures,
            "reason": _text(compatibility.get("reason"), "a compatibility explanation", 800),
        },
        "images": images,
        "ports": ports,
        "volumes": volumes,
        "variables": variables,
        "resources": {
            "memory_bytes": _integer(resources.get("memory_bytes", 536870912), "memory guidance", 67108864, 274877906944),
            "cpu_cores": _integer(resources.get("cpu_cores", 1), "CPU guidance", 1, 256),
            "storage_bytes": _integer(resources.get("storage_bytes", 1073741824), "storage guidance", 0, 10995116277760),
        },
        "sources": sources,
    }
    services = {image["service"] for image in images}
    for item in ports + volumes + variables:
        if item["service"] not in services:
            raise ApplicationDeploymentError("Manifest settings must refer to a pinned-image service.")
    return normalized


def _add_operator_ports(
    add_ports: Any, services: Dict[str, Dict[str, Any]]
) -> None:
    """Publish operator-chosen host port mappings the manifest never declared.

    Guided configuration can already re-map a port the research *declared*, but
    an application whose manifest declares no ports would deploy with nothing
    published and be unreachable. This lets the operator add a host->container
    mapping for such an app. It only pins a host port onto the already-pinned
    image - the digest-trusted image is untouched - so it does not weaken the
    trust model the way an image or command change would.
    """
    if not isinstance(add_ports, list):
        raise ApplicationDeploymentError("Published ports must be a list.")
    if len(add_ports) > 32:
        raise ApplicationDeploymentError("Too many published ports were supplied.")
    for entry in add_ports:
        if not isinstance(entry, dict):
            raise ApplicationDeploymentError("Each published port must be an object.")
        service = _service(entry.get("service"))
        if service not in services:
            raise ApplicationDeploymentError("A published port names an unknown service.")
        protocol = str(entry.get("protocol", "tcp")).strip().lower()
        if protocol not in PROTOCOLS:
            raise ApplicationDeploymentError("Only TCP and UDP published ports are supported.")
        target = _integer(entry.get("target"), "container port", 1, 65535)
        published = _integer(
            entry.get("published", target), "published port", 1, 65535
        )
        services[service].setdefault("ports", []).append(
            {"target": target, "published": published, "protocol": protocol}
        )


def _add_operator_host_mounts(
    host_mounts: Any, services: Dict[str, Dict[str, Any]]
) -> None:
    """Grant an allowlisted host mount, and only behind explicit consent.

    Some applications genuinely need a host path - a monitor needs the Docker
    socket, for instance - and guided configuration otherwise has no way to
    provide one, so the container crash-loops and is rolled back. This grants
    such a mount, but never silently: the source must be one the shared
    ``PRIVILEGED_HOST_MOUNTS`` allowlist blesses, and the operator must set an
    explicit per-mount ``consent`` flag. A missing or false flag is refused, so
    a privileged mount is never mounted by default or as a side effect. The
    allowlist's read-only constraint is applied by us, not chosen by the
    operator, and the executor's ``validate_normalized`` re-checks both the
    allowlist membership and the read-only constraint independently.
    """
    if not isinstance(host_mounts, list):
        raise ApplicationDeploymentError("Host mounts must be a list.")
    if len(host_mounts) > 8:
        raise ApplicationDeploymentError("Too many host mounts were supplied.")
    for entry in host_mounts:
        if not isinstance(entry, dict):
            raise ApplicationDeploymentError("Each host mount must be an object.")
        service = _service(entry.get("service"))
        if service not in services:
            raise ApplicationDeploymentError("A host mount names an unknown service.")
        source = str(entry.get("source", "")).strip()
        allowed = PRIVILEGED_HOST_MOUNTS.get(source)
        if allowed is None:
            raise ApplicationDeploymentError(
                "That host path is not permitted as a mount."
            )
        if entry.get("consent") is not True:
            raise ApplicationDeploymentError(
                "Mounting {} is privileged and needs explicit operator consent."
                .format(source)
            )
        target = str(entry.get("target", source)).strip() or source
        if not target.startswith("/") or ".." in target.split("/"):
            raise ApplicationDeploymentError(
                "Host mount targets must be absolute and traversal-free."
            )
        services[service].setdefault("volumes", []).append({
            "type": "bind",
            "source": source,
            "target": target,
            "read_only": bool(allowed["read_only"]),
        })


def build_compose(manifest: Dict[str, Any], configuration: Any) -> Dict[str, Any]:
    """Generate bounded Compose data from a normalized verified manifest."""
    normalized = normalize_manifest(manifest)
    if normalized["compatibility"]["status"] != "verified":
        raise ApplicationDeploymentError("Only a verified application manifest can produce Compose.")
    if not isinstance(configuration, dict):
        raise ApplicationDeploymentError("Application configuration must be an object.")
    unknown = set(configuration) - {
        "ports", "add_ports", "host_mounts", "variables", "resources",
        "replace_existing",
    }
    if unknown:
        raise ApplicationDeploymentError("Unsupported application configuration fields were supplied.")
    port_values = configuration.get("ports", {})
    variable_values = configuration.get("variables", {})
    resource_values = configuration.get("resources", {})
    if not isinstance(port_values, dict) or not isinstance(variable_values, dict) or not isinstance(resource_values, dict):
        raise ApplicationDeploymentError("Ports and variables must be configuration objects.")
    minimum_memory = normalized["resources"]["memory_bytes"]
    minimum_storage = normalized["resources"]["storage_bytes"]
    memory_bytes = _integer(
        resource_values.get("memory_bytes", minimum_memory),
        "memory limit", minimum_memory, 274877906944,
    )
    storage_bytes = _integer(
        resource_values.get("storage_bytes", minimum_storage),
        "reserved storage", minimum_storage, 10995116277760,
    )

    services: Dict[str, Dict[str, Any]] = {}
    for image in normalized["images"]:
        services[image["service"]] = {
            "image": image["reference"],
            "restart": "unless-stopped",
            "deploy": {"resources": {"limits": {
                "memory": str(memory_bytes),
                "cpus": str(normalized["resources"]["cpu_cores"]),
            }}},
        }
    for port in normalized["ports"]:
        published = _integer(
            port_values.get(port["name"], port["published"]), "published port", 1, 65535
        )
        services[port["service"]].setdefault("ports", []).append(
            {"target": port["target"], "published": published, "protocol": port["protocol"]}
        )
    _add_operator_ports(configuration.get("add_ports", []), services)
    _add_operator_host_mounts(configuration.get("host_mounts", []), services)
    compose_volumes = {}
    for volume in normalized["volumes"]:
        compose_volumes[volume["name"]] = {}
        services[volume["service"]].setdefault("volumes", []).append(
            {
                "type": "volume",
                "source": volume["name"],
                "target": volume["mount_path"],
                "read_only": volume["mode"] == "ro",
            }
        )
    secret_refs = {}
    known_variables = {item["name"] for item in normalized["variables"]}
    if set(variable_values) - known_variables:
        raise ApplicationDeploymentError("Unknown application variables were supplied.")
    for variable in normalized["variables"]:
        supplied = variable_values.get(variable["name"])
        if variable["secret"]:
            if not isinstance(supplied, dict) or not _CREDENTIAL_ID.fullmatch(str(supplied.get("credential_id", ""))):
                if variable["required"]:
                    raise ApplicationDeploymentError(f"Choose a managed credential for {variable['name']}.")
                continue
            credential_id = str(supplied["credential_id"])
            secret_refs[variable["name"]] = credential_id
            value = "${VAELOR_CREDENTIAL_" + variable["name"] + ":?managed credential required}"
        else:
            value = supplied if supplied is not None else variable.get("default")
            if value is None:
                if variable["required"]:
                    raise ApplicationDeploymentError(f"Enter {variable['name']}.")
                continue
            value = _text(value, variable["name"], 500, required=False)
        services[variable["service"]].setdefault("environment", {})[variable["name"]] = value
    compose: Dict[str, Any] = {
        "services": services,
        "x-vaelor-resources": {"storage_bytes": storage_bytes},
        "x-vaelor-replace-existing": bool(configuration.get("replace_existing", False)),
    }
    if compose_volumes:
        compose["volumes"] = compose_volumes
    if secret_refs:
        compose["x-vaelor-secret-references"] = secret_refs
    return compose


class ApplicationDeploymentStore:
    """SQLite-backed immutable deployment drafts and state transitions."""

    def __init__(self, database_path: Optional[str] = None):
        self.database_path = database_path or env_value(
            "VAELOR_APPLICATION_DB", "PM_APPLICATION_DB",
            state_path("applications/applications.sqlite3"),
        )
        self._schema_ready = False
        self._schema_lock = threading.Lock()
        self._connect().close()

    def _connect(self) -> sqlite3.Connection:
        path = Path(self.database_path)
        path.parent.mkdir(parents=True, exist_ok=True)
        try:
            descriptor = os.open(str(path), os.O_CREAT | os.O_EXCL | os.O_RDWR, 0o660)
        except FileExistsError:
            pass
        else:
            os.close(descriptor)
        try:
            os.chmod(path, 0o660)
        except PermissionError:
            # The executor may win the first-start race and own this shared
            # group-writable database. Its setgid parent still guarantees the
            # vaelor-jobs group; a non-owner must not fail merely reasserting
            # an already-correct mode.
            if path.stat().st_mode & 0o777 != 0o660:
                raise
        group_id = jobs_group_id(grp)
        if group_id is not None:
            try:
                os.chown(path, -1, group_id)
            except PermissionError:
                pass
        connection = sqlite3.connect(str(path), timeout=10)
        connection.row_factory = sqlite3.Row
        self._ensure_schema(connection)
        return connection

    def _ensure_schema(self, connection: sqlite3.Connection) -> None:
        if self._schema_ready:
            return
        with self._schema_lock:
            if self._schema_ready:
                return
            connection.execute("PRAGMA journal_mode = WAL")  # pairs-with: sqlite-journal-mode-spaced
            connection.executescript("""
                CREATE TABLE IF NOT EXISTS application_drafts (
                    id TEXT PRIMARY KEY,
                    actor TEXT NOT NULL,
                    state TEXT NOT NULL,
                    intent_json TEXT NOT NULL,
                    manifest_json TEXT,
                    manifest_digest TEXT,
                    configuration_json TEXT,
                    compose_json TEXT,
                    compose_digest TEXT,
                    validation_json TEXT,
                    failure TEXT NOT NULL DEFAULT '',
                    created_at INTEGER NOT NULL,
                    updated_at INTEGER NOT NULL
                );
                CREATE INDEX IF NOT EXISTS idx_application_drafts_actor
                    ON application_drafts(actor, updated_at DESC);
            """)
            connection.commit()
            self._schema_ready = True

    @staticmethod
    def _record(row: sqlite3.Row) -> Dict[str, Any]:
        result = {key: row[key] for key in ("id", "actor", "state", "manifest_digest", "compose_digest", "failure", "created_at", "updated_at")}
        for column, key in (
            ("intent_json", "intent"), ("manifest_json", "manifest"),
            ("configuration_json", "configuration"), ("compose_json", "compose"),
            ("validation_json", "validation"),
        ):
            result[key] = json.loads(row[column]) if row[column] else None
        return result

    def create(self, actor: str, intent: Dict[str, Any]) -> Dict[str, Any]:
        actor_name = _text(actor, "an actor", 120)
        if not isinstance(intent, dict) or intent.get("type") != "application.deploy":
            raise ApplicationDeploymentError("Choose a structured application deployment intent.")
        query = _text(intent.get("application_query"), "an application", 160)
        normalized_intent = {
            "schema_version": SCHEMA_VERSION, "type": "application.deploy",
            "application_query": query,
            "delivery_method": str(intent.get("delivery_method", "auto")) if intent.get("delivery_method") in {"auto", "container"} else "auto",
            "confidence": str(intent.get("confidence", "low")) if intent.get("confidence") in {"low", "medium", "high"} else "low",
            "research_required": True,
            "missing_inputs": [
                _text(item, "a missing input", 200)
                for item in list(intent.get("missing_inputs", []))[:12]
            ],
        }
        refinement = intent.get("refinement")
        if isinstance(refinement, dict):
            normalized_intent["refinement"] = {
                "source": _text(refinement.get("source", "built-in-parser"), "a refinement source", 80),
                "interpretation": _text(refinement.get("interpretation", ""), "an interpretation", 400, required=False),
                "model_used": bool(refinement.get("model_used", False)),
                "follow_up_questions": [
                    _text(item, "a follow-up question", 240)
                    for item in list(refinement.get("follow_up_questions", []))[:3]
                ],
            }
        now = int(time.time())
        draft_id = "appdraft_" + secrets.token_hex(12)
        with closing(self._connect()) as connection:
            connection.execute(
                "INSERT INTO application_drafts(id,actor,state,intent_json,created_at,updated_at) VALUES(?,?,?,?,?,?)",
                (draft_id, actor_name, "research", _canonical(normalized_intent), now, now),
            )
            connection.commit()
        return self.get(draft_id, actor_name)

    def get(self, draft_id: str, actor: Optional[str] = None) -> Optional[Dict[str, Any]]:
        query = "SELECT * FROM application_drafts WHERE id=?"
        params: tuple[Any, ...] = (str(draft_id),)
        if actor is not None:
            query += " AND actor=?"
            params += (_text(actor, "an actor", 120),)
        with closing(self._connect()) as connection:
            row = connection.execute(query, params).fetchone()
        return self._record(row) if row else None

    def discard(
        self, draft_id: str, actor: str,
        *, live_projects: frozenset[str] = frozenset(),
    ) -> Dict[str, Any]:
        """Remove one actor-owned saved draft and return an explicit cleanup result.

        A saved draft is deployment *intent*, not the deployment itself: deleting
        the row never stops or removes a container, so it is refused only when the
        draft backs a genuinely-live application -- its Compose project appears in
        ``live_projects``.  A wedged record (approved-but-failed, or
        imported-but-no-longer-running) is therefore always clearable, which the
        blanket ``approved``/``imported`` refusal made impossible.  Liveness is
        supplied by the caller because this store deliberately does not reach the
        container runtime.  The delete is scoped by both ID and actor and does not
        touch the JobStore or any active operation ledger.
        """
        current = self._require(draft_id, actor)
        manifest = current.get("manifest")
        project = compose_project_name(manifest) if isinstance(manifest, dict) else ""
        if project and project in live_projects:
            raise ApplicationDeploymentError(LIVE_DEPLOYMENT_DRAFT_DISCARD_REFUSED)
        actor_name = _text(actor, "an actor", 120)
        with closing(self._connect()) as connection:
            connection.execute("BEGIN IMMEDIATE")  # pairs-with: sqlite-begin-immediate
            cursor = connection.execute(
                "DELETE FROM application_drafts WHERE id=? AND actor=?",
                (str(draft_id), actor_name),
            )
            if cursor.rowcount != 1:
                connection.rollback()
                raise ApplicationDeploymentError(
                    "The application draft changed state; refresh before discarding."
                )
            connection.commit()
        return {
            "draft_id": str(draft_id),
            "actor": actor_name,
            "discarded": True,
            "cleanup": {
                "state": "completed",
                "saved_draft_removed": True,
                "resume_key_removed": True,
                "temporary_artifacts_removed": [],
                "errors": [],
            },
        }

    def attach_manifest(self, draft_id: str, actor: str, manifest: Dict[str, Any]) -> Dict[str, Any]:
        normalized = normalize_manifest(manifest)
        digest = content_digest(normalized)
        # An Advanced-recovery retry (or a duplicate/late research job finishing)
        # re-runs research on a draft the *first* pass already advanced to
        # ``configuration``. Only allowing ``research`` here dead-ended that retry
        # with "the application draft changed state; refresh before continuing." -
        # a stale-read the operator can neither see nor act on. Re-attaching the
        # fresh manifest is safe until a Compose draft has been generated from it,
        # so re-research is accepted from ``research`` OR an un-configured
        # ``configuration`` and is idempotent (last research wins). Once a compose
        # exists the manifest is load-bearing and must not silently change, so that
        # is refused with an explicit, actionable message instead of the race guard.
        current = self._require(draft_id, actor)
        if current["compose"] is not None:
            raise ApplicationDeploymentError(
                "This application has already been configured; discard it to research it again."
            )
        return self._transition(
            draft_id, actor, {"research", "configuration"}, "configuration",
            manifest_json=_canonical(normalized), manifest_digest=digest,
        )

    def configure(self, draft_id: str, actor: str, configuration: Dict[str, Any]) -> Dict[str, Any]:
        current = self._require(draft_id, actor)
        if current["state"] != "configuration" or current["compose"] is not None:
            raise ApplicationDeploymentError("This application draft can no longer be configured.")
        compose = build_compose(current["manifest"], configuration)
        return self._transition(
            draft_id, actor, {"configuration"}, "configuration",
            configuration_json=_canonical(configuration), compose_json=_canonical(compose),
            compose_digest=content_digest(compose),
        )

    def mark_validated(self, draft_id: str, actor: str, validation: Dict[str, Any]) -> Dict[str, Any]:
        current = self._require(draft_id, actor)
        if current["state"] != "configuration" or current["compose"] is None:
            raise ApplicationDeploymentError("Generate the server-owned Compose draft before validation.")
        if not isinstance(validation, dict) or validation.get("policy") != "validated":
            raise ApplicationDeploymentError("Deterministic Compose policy validation must pass first.")
        return self._transition(
            draft_id, actor, {"configuration"}, "validated",
            validation_json=_canonical(validation),
        )

    def approve(self, draft_id: str, actor: str, manifest_digest: str) -> Dict[str, Any]:
        current = self._require_digest(draft_id, actor, manifest_digest)
        if current["state"] != "validated":
            raise ApplicationDeploymentError("Only a validated server draft can be approved.")
        approved = self._transition(draft_id, actor, {"validated"}, "approved")
        return {
            "draft": approved,
            "proposed_job": {
                "type": "compose.import",
                "payload": {
                    "draft_id": draft_id,
                    "manifest_digest": manifest_digest,
                    "project": compose_project_name(approved["manifest"]),
                },
            },
        }

    def resolve_import(self, draft_id: str, manifest_digest: str, actor: str) -> Dict[str, Any]:
        current = self.get(draft_id, actor)
        if current is None:
            raise ApplicationDeploymentError("The application draft was not found.")
        if current["state"] != "approved":
            raise ApplicationDeploymentError("The application draft has not been approved.")
        if not secrets.compare_digest(str(current["manifest_digest"]), str(manifest_digest)):
            raise ApplicationDeploymentError("The application manifest digest no longer matches.")
        if content_digest(current["manifest"]) != current["manifest_digest"] or content_digest(current["compose"]) != current["compose_digest"]:
            raise ApplicationDeploymentError("The stored application draft failed its integrity check.")
        return current

    def mark_imported(self, draft_id: str, manifest_digest: str, actor: str) -> Dict[str, Any]:
        current = self.resolve_import(draft_id, manifest_digest, actor)
        return self._transition(draft_id, current["actor"], {"approved"}, "imported")

    def fail(self, draft_id: str, actor: str, message: str) -> Dict[str, Any]:
        current = self._require(draft_id, actor)
        if current["state"] in {"imported", "failed"}:
            raise ApplicationDeploymentError("This application draft is already final.")
        return self._transition(
            draft_id, actor, DRAFT_STATES - {"imported", "failed"}, "failed",
            failure=_text(message, "a failure reason", 800),
        )

    def _require(self, draft_id: str, actor: str) -> Dict[str, Any]:
        current = self.get(draft_id, actor)
        if current is None:
            raise ApplicationDeploymentError("The application draft was not found.")
        return current

    def _require_digest(self, draft_id: str, actor: str, digest: str) -> Dict[str, Any]:
        current = self._require(draft_id, actor)
        if not secrets.compare_digest(str(current["manifest_digest"] or ""), str(digest or "")):
            raise ApplicationDeploymentError("The application manifest digest no longer matches.")
        return current

    def _transition(
        self, draft_id: str, actor: str, allowed: set[str], state: str, **fields: Any
    ) -> Dict[str, Any]:
        actor_name = _text(actor, "an actor", 120)
        assignments = ["state=?", "updated_at=?"]
        values: list[Any] = [state, int(time.time())]
        allowed_fields = {
            "manifest_json", "manifest_digest", "configuration_json", "compose_json",
            "compose_digest", "validation_json", "failure",
        }
        for key, value in fields.items():
            if key not in allowed_fields:
                raise ApplicationDeploymentError("Unsupported draft state update.")
            assignments.append(f"{key}=?")
            values.append(value)
        placeholders = ",".join("?" for _ in allowed)
        values.extend([draft_id, actor_name, *sorted(allowed)])
        with closing(self._connect()) as connection:
            cursor = connection.execute(
                f"UPDATE application_drafts SET {','.join(assignments)} WHERE id=? AND actor=? AND state IN ({placeholders})",
                values,
            )
            connection.commit()
        if cursor.rowcount != 1:
            raise ApplicationDeploymentError("The application draft changed state; refresh before continuing.")
        return self._require(draft_id, actor_name)
