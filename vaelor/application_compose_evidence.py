"""Deterministic, security-hardened parser for a fetched ``docker-compose.yml``.

The single input is UNTRUSTED web evidence (a compose file the research broker
fetched). This module turns it into a bounded :class:`ComposeSpec` (per-service
image reference, ports, named volumes, and classified variables) or raises a
typed :class:`ComposeEvidenceError` so the caller degrades honestly to the
single-image path. It NEVER crashes on adversarial input and NEVER fabricates.

Discipline (mirrors ``container_registry.py``): PURE and self-contained. NO
network, NO subprocess, NO filesystem, NO Docker, and NO per-application data.
Every refusal is deterministic and driven only by the compose text. All
untrusted-YAML handling lives here so the surface is reviewable in one file.

The parser NEVER trusts the compose for image identity: it extracts only the
image *reference string* (any ``@sha256`` is left for the registry proof to
drop and re-prove) and NEVER carries a web literal that could be a credential
into a variable default. Image digest pinning, N-image proof, and manifest
assembly are the caller's job (B2), not this module's.
"""

from __future__ import annotations

import math
import re
from collections import Counter
from collections.abc import Hashable
from dataclasses import dataclass, field
from typing import Any

try:  # PyYAML is declared in requirements-release.txt / pyproject.toml.
    import yaml
except ImportError:  # pragma: no cover - exercised via the typed refusal below.
    yaml = None  # type: ignore[assignment]


# --- Bounds (all fail-closed) ------------------------------------------------

#: Hard byte ceiling on the compose text handed to the loader. Well under the
#: broker's 256 KB ``raw_text`` cap, so an oversized document is refused before
#: any YAML machinery runs.
MAX_COMPOSE_BYTES = 256 * 1024
#: Matches the manifest schema (``normalize_manifest`` accepts <=12 images).
MAX_SERVICES = 12
#: Per-service caps, mirroring the manifest's <=32 ports and <=64 variables.
MAX_PORTS_PER_SERVICE = 32
MAX_ENV_PER_SERVICE = 64
MAX_VOLUMES_PER_SERVICE = 32
#: Bounds nesting cheaply: no legitimate compose indents this far, and refusing
#: past it caps the structural depth the loader can be asked to build.
MAX_INDENT_COLUMNS = 64
#: Longest image reference string carried forward (``parse_image_reference``
#: itself refuses anything longer, so this only trims obviously-hostile input).
MAX_IMAGE_REF = 400
#: A carried non-secret default may be at most this long; longer values are
#: dropped to "operator supplies" rather than embedded from the web.
MAX_DEFAULT_LEN = 64

PROTOCOLS = frozenset({"tcp", "udp"})
VOLUME_MODES = frozenset({"rw", "ro"})

# Honest-degrade note templates, defined once so a message cannot drift between
# the short-form and long-form volume paths that both emit it.
_NOTE_BIND_DROPPED = "service {}: a host bind mount was dropped"
_NOTE_INVALID_VOLUME = "service {}: an invalid volume was dropped"

# Variable-name shape the manifest enforces (``_SAFE_VARIABLE`` in
# ``application_deployments``): uppercase, digits, underscore, <=80 chars. A
# name that cannot be reshaped to this is skipped, never passed on malformed.
_SAFE_VARIABLE = re.compile(r"^[A-Z][A-Z0-9_]{0,79}$")

# Secret classification by key NAME (case-insensitive substring). Deliberately
# WIDE: a false negative would carry a web-literal credential as a default,
# which is the exact defect this closes. URL/URI/DSN/CONN/AUTH are included
# because ``DATABASE_URL``/``*_DSN``/``SENTRY_DSN`` routinely embed passwords.
_SECRET_NAME_TOKENS = (
    "PASSWORD", "PASS", "PW", "SECRET", "TOKEN", "KEY", "CREDENTIAL",
    "AUTH", "URL", "URI", "DSN", "CONN", "CONNECTION",
    "ACCESS", "WEBHOOK", "SIGN", "SALT", "CERT", "PRIVATE",
)

# Backstop marker for a value that names a credential inline (a PEM header or a
# ``password:``/``token=`` style pair). A value matching this is never carried.
_SECRET_MARKER = re.compile(
    r"(?i)(?:-----BEGIN|(?:password|passwd|token|secret|api[_-]?key)\s*[:=])"
)

# A JWT: three base64url segments whose header begins ``eyJ`` (base64 of ``{"``).
# Anchored on the header prefix so a dotted version string ("1.2.3") is not
# mistaken for one.
_JWT = re.compile(r"^eyJ[A-Za-z0-9_-]{6,}\.[A-Za-z0-9_-]{6,}\.[A-Za-z0-9_-]*$")

# Service keys that make a compose undeployable on the managed cluster. Their
# mere PRESENCE refuses the whole compose (belt-and-suspenders to
# ``validate_normalized`` and the cluster renderer's blanket bind refusal).
_REFUSE_SERVICE_KEYS = frozenset({
    "privileged", "network_mode", "pid", "ipc", "cap_add", "devices",
    "build", "env_file", "extends", "include", "configs", "secrets",
})

# Textual sigils refused before the loader runs: YAML anchor/alias/merge. Real
# composes rarely use these, and refusing them up front kills the billion-laughs
# / merge vectors cheaply. Compose interpolation (``${VAR}``) is refused
# separately below, so the ``$$`` literal escape can be honoured.
_ANCHOR_ALIAS_MERGE = ("&", "*", "<<")
# One message covers both the anchor/alias/merge sigils and interpolation - the
# two guards that reject a compose we cannot safely read - so it is named once.
_INTERP_OR_ANCHOR_REFUSAL = "compose uses anchors, aliases, or interpolation"
# A YAML tag indicator (``!!python/object``, ``!Ref``) at a node position - line
# start, or after a ``:``/``-``/flow indicator and whitespace - so an ordinary
# ``!`` inside a scalar value ("Hello!World") does not trip it. The SafeLoader
# already refuses unknown tags via ConstructorError; this is a cheap early
# backstop, not the sole guard.
_TAG_INDICATOR = re.compile(r"(?m)(?:^[ \t]*|[:\-\[{,][ \t]+)!(?:!|[A-Za-z<])")


class ComposeEvidenceError(ValueError):
    """A fetched compose could not be parsed into a safe spec.

    ``reason`` is a short, non-leaking phrase the caller can log; it never
    echoes untrusted content. The caller degrades to the single-image path.
    """

    def __init__(self, reason: str) -> None:
        super().__init__(reason)
        self.reason = reason


@dataclass(frozen=True)
class ComposeService:
    name: str
    image_ref: str
    ports: list[dict[str, Any]] = field(default_factory=list)
    volumes: list[dict[str, Any]] = field(default_factory=list)
    variables: list[dict[str, Any]] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "image_ref": self.image_ref,
            "ports": [dict(item) for item in self.ports],
            "volumes": [dict(item) for item in self.volumes],
            "variables": [dict(item) for item in self.variables],
        }


@dataclass(frozen=True)
class ComposeSpec:
    services: list[ComposeService] = field(default_factory=list)
    #: Ordered unique image reference strings, so the caller fetches each once.
    image_refs: list[str] = field(default_factory=list)
    #: Honest-degrade breadcrumbs (dropped binds, dropped ordering, uncarried
    #: values). Never load-bearing; purely informational for the operator.
    notes: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return {
            "services": [service.to_dict() for service in self.services],
            "image_refs": list(self.image_refs),
            "notes": list(self.notes),
        }


class _StrictSafeLoader(yaml.SafeLoader if yaml is not None else object):  # type: ignore[misc]
    """SafeLoader that RAISES on a duplicate mapping key.

    Stock ``yaml.safe_load`` silently keeps the LAST duplicate key, so a
    reviewer could read the first ``image:``/service while the deployer pins the
    last - an image-smuggling vector a textual scan cannot catch. Constructing
    deeply here (the text is already size- and depth-bounded) makes the refusal
    immediate rather than deferred behind PyYAML's two-pass generators.
    """

    def construct_mapping(self, node: Any, deep: bool = True) -> dict[Any, Any]:
        if yaml is not None and not isinstance(node, yaml.MappingNode):
            raise ComposeEvidenceError("compose structure is invalid")
        mapping: dict[Any, Any] = {}
        for key_node, value_node in node.value:
            key = self.construct_object(key_node, deep=True)
            if not isinstance(key, Hashable):
                raise ComposeEvidenceError("compose uses an unhashable key")
            if key in mapping:
                raise ComposeEvidenceError("compose defines a duplicate key")
            mapping[key] = self.construct_object(value_node, deep=True)
        return mapping


def parse_compose(raw_text: str) -> ComposeSpec:
    """Parse untrusted compose text into a :class:`ComposeSpec`, fail-closed.

    Order of operations (each stage refuses rather than repairs):
    dependency guard -> byte cap -> textual sigil refusal -> indent-depth cap
    -> duplicate-key DoS-safe load -> structural refusals -> per-service extract.
    """
    if yaml is None:
        raise ComposeEvidenceError("compose parsing is unavailable")
    if not isinstance(raw_text, str):
        raise ComposeEvidenceError("compose text was not a string")
    if len(raw_text.encode("utf-8", "replace")) > MAX_COMPOSE_BYTES:
        raise ComposeEvidenceError("compose is too large")

    _refuse_textual_sigils(raw_text)
    _refuse_deep_indent(raw_text)
    document = _safe_load(raw_text)
    _refuse_interpolation(document)
    return _build_spec(document)


def _refuse_textual_sigils(raw_text: str) -> None:
    for sigil in _ANCHOR_ALIAS_MERGE:
        if sigil in raw_text:
            raise ComposeEvidenceError(_INTERP_OR_ANCHOR_REFUSAL)
    if _TAG_INDICATOR.search(raw_text):
        raise ComposeEvidenceError("compose uses YAML tags")


def _refuse_interpolation(node: Any) -> None:
    """Refuse Compose-time interpolation, checked on the DECODED scalars.

    Compose interpolates ``${VAR}``; ``$$`` is the escape for a literal ``$``
    (Compose emits ``${VAR}`` verbatim for the container's own runtime - NOT a
    compose-time substitution - and it is common in healthcheck/command
    strings). The escape reasoning is only exact on a single contiguous ``$``
    run, so it MUST run on the value the loader produced, not the raw text: a
    YAML line continuation (``\\`` + newline) or a ``\\u0024`` escape decodes to
    a different ``$`` run than the source shows, and a raw-text ``$$`` collapse
    would mis-pair the dollars across the break and let a genuine ``${VAR}``
    through. Post-load, ``$$`` runs are contiguous, so collapsing them and
    refusing a surviving ``${`` is exact. Keys are walked too - Compose
    interpolates them as well.
    """
    if isinstance(node, str):
        if "${" in node.replace("$$", ""):
            raise ComposeEvidenceError(_INTERP_OR_ANCHOR_REFUSAL)
    elif isinstance(node, dict):
        for key, value in node.items():
            _refuse_interpolation(key)
            _refuse_interpolation(value)
    elif isinstance(node, (list, tuple)):
        for item in node:
            _refuse_interpolation(item)


def _refuse_deep_indent(raw_text: str) -> None:
    for line in raw_text.splitlines():
        stripped = line.lstrip(" ")
        if stripped and (len(line) - len(stripped)) > MAX_INDENT_COLUMNS:
            raise ComposeEvidenceError("compose is nested too deeply")


def _safe_load(raw_text: str) -> Any:
    try:
        document = yaml.load(raw_text, Loader=_StrictSafeLoader)
    except ComposeEvidenceError:
        raise
    except (yaml.YAMLError, RecursionError) as error:
        raise ComposeEvidenceError("compose is not valid YAML") from error
    if not isinstance(document, dict):
        raise ComposeEvidenceError("compose must be a mapping")
    return document


def _build_spec(document: dict[Any, Any]) -> ComposeSpec:
    for key in ("configs", "secrets", "include"):
        if document.get(key):
            raise ComposeEvidenceError("compose uses top-level {}".format(key))

    services = document.get("services")
    if not isinstance(services, dict) or not services:
        raise ComposeEvidenceError("compose defines no services")
    if len(services) > MAX_SERVICES:
        raise ComposeEvidenceError("compose defines too many services")

    _refuse_depends_on_cycle(services)

    notes: list[str] = []
    built: list[ComposeService] = []
    image_refs: list[str] = []
    for raw_name, body in services.items():
        service = _build_service(raw_name, body, notes)
        built.append(service)
        if service.image_ref not in image_refs:
            image_refs.append(service.image_ref)
    if any("depends_on" in body for body in services.values() if isinstance(body, dict)):
        notes.append("startup ordering not enforced (Swarm has no depends_on)")
    return ComposeSpec(services=built, image_refs=image_refs, notes=notes)


def _build_service(raw_name: Any, body: Any, notes: list[str]) -> ComposeService:
    name = str(raw_name).strip().lower()
    if not re.fullmatch(r"[a-z][a-z0-9_-]{0,62}", name):
        raise ComposeEvidenceError("compose has an unsafe service name")
    if not isinstance(body, dict):
        raise ComposeEvidenceError("compose service {} is invalid".format(name))
    for key in _REFUSE_SERVICE_KEYS:
        if key in body:
            raise ComposeEvidenceError(
                "compose service {} uses a blocked setting".format(name)
            )
    image_ref = _image_ref(body.get("image"), name)
    ports = _ports(body.get("ports"), name, notes)
    volumes = _volumes(body.get("volumes"), name, notes)
    variables = _variables(body.get("environment"), name, notes)
    return ComposeService(
        name=name,
        image_ref=image_ref,
        ports=ports,
        volumes=volumes,
        variables=variables,
    )


def _image_ref(value: Any, service: str) -> str:
    if not isinstance(value, str):
        raise ComposeEvidenceError(
            "compose service {} has no pinnable image".format(service)
        )
    ref = value.strip()
    if not ref or len(ref) > MAX_IMAGE_REF or any(ch.isspace() for ch in ref):
        raise ComposeEvidenceError(
            "compose service {} has an invalid image".format(service)
        )
    return ref


def _refuse_depends_on_cycle(services: dict[Any, Any]) -> None:
    graph: dict[str, list[str]] = {}
    for name, body in services.items():
        node = str(name).strip().lower()
        graph[node] = _depends_on_targets(body)

    visiting: set[str] = set()
    done: set[str] = set()

    def visit(node: str) -> None:
        if node in done:
            return
        if node in visiting:
            raise ComposeEvidenceError("compose has a dependency cycle")
        visiting.add(node)
        for target in graph.get(node, ()):
            if target in graph:
                visit(target)
        visiting.discard(node)
        done.add(node)

    for node in graph:
        visit(node)


def _depends_on_targets(body: Any) -> list[str]:
    if not isinstance(body, dict):
        return []
    depends = body.get("depends_on")
    if isinstance(depends, dict):
        return [str(key).strip().lower() for key in depends]
    if isinstance(depends, list):
        return [str(item).strip().lower() for item in depends]
    return []


def _ports(value: Any, service: str, notes: list[str]) -> list[dict[str, Any]]:
    if value is None:
        return []
    if not isinstance(value, list):
        raise ComposeEvidenceError("compose service {} has invalid ports".format(service))
    if len(value) > MAX_PORTS_PER_SERVICE:
        raise ComposeEvidenceError("compose service {} declares too many ports".format(service))
    result: list[dict[str, Any]] = []
    for entry in value:
        mapping = _port_entry(entry)
        if mapping is None:
            notes.append("service {}: a non-ingress port was dropped".format(service))
            continue
        result.append(mapping)
    return result


def _port_entry(entry: Any) -> dict[str, Any] | None:
    if isinstance(entry, dict):
        return _long_port(entry)
    if isinstance(entry, int) and not isinstance(entry, bool):
        return None  # a bare container port has no deterministic host publish
    if not isinstance(entry, str):
        return None
    text = entry.strip()
    protocol = "tcp"
    if "/" in text:
        text, _sep, proto = text.rpartition("/")
        protocol = proto.strip().lower()
    parts = text.split(":")
    if protocol not in PROTOCOLS or len(parts) != 2:
        # A single port (ephemeral publish) or a host-IP-bound form
        # (ip:published:target, len 3) is not a deterministic ingress mapping.
        return None
    published = _port_number(parts[0])
    target = _port_number(parts[1])
    if published is None or target is None:
        return None
    return {"published": published, "target": target, "protocol": protocol}


def _long_port(entry: dict[str, Any]) -> dict[str, Any] | None:
    if entry.get("host_ip"):
        return None  # host-IP-bound is not ingress
    mode = str(entry.get("mode", "ingress")).strip().lower()
    if mode and mode != "ingress":
        return None
    published = _port_number(entry.get("published"))
    target = _port_number(entry.get("target"))
    if published is None or target is None:
        return None
    protocol = str(entry.get("protocol", "tcp")).strip().lower()
    if protocol not in PROTOCOLS:
        return None
    return {"published": published, "target": target, "protocol": protocol}


def _port_number(value: Any) -> int | None:
    if isinstance(value, bool):
        return None
    try:
        number = int(str(value).strip())
    except (TypeError, ValueError):
        return None
    return number if 1 <= number <= 65535 else None


def _volumes(value: Any, service: str, notes: list[str]) -> list[dict[str, Any]]:
    if value is None:
        return []
    if not isinstance(value, list):
        raise ComposeEvidenceError("compose service {} has invalid volumes".format(service))
    if len(value) > MAX_VOLUMES_PER_SERVICE:
        raise ComposeEvidenceError("compose service {} declares too many volumes".format(service))
    result: list[dict[str, Any]] = []
    for entry in value:
        mapping = _volume_entry(entry, service, notes)
        if mapping is not None:
            result.append(mapping)
    return result


def _volume_entry(
    entry: Any, service: str, notes: list[str]
) -> dict[str, Any] | None:
    if isinstance(entry, dict):
        return _long_volume(entry, service, notes)
    if not isinstance(entry, str):
        notes.append("service {}: an unreadable volume was dropped".format(service))
        return None
    parts = entry.split(":")
    if len(parts) < 2:
        notes.append("service {}: an anonymous volume was dropped".format(service))
        return None
    source, target = parts[0].strip(), parts[1].strip()
    mode = "ro" if len(parts) >= 3 and parts[2].strip().lower() == "ro" else "rw"
    if _is_host_path(source):
        notes.append(_NOTE_BIND_DROPPED.format(service))
        return None
    if not _is_named_source(source) or not _is_absolute_target(target):
        notes.append(_NOTE_INVALID_VOLUME.format(service))
        return None
    return {"name": source.lower(), "target": target, "mode": mode}


def _long_volume(
    entry: dict[str, Any], service: str, notes: list[str]
) -> dict[str, Any] | None:
    kind = str(entry.get("type", "volume")).strip().lower()
    source = str(entry.get("source", "")).strip()
    target = str(entry.get("target", "")).strip()
    if kind == "bind" or _is_host_path(source):
        notes.append(_NOTE_BIND_DROPPED.format(service))
        return None
    if kind != "volume" or not _is_named_source(source) or not _is_absolute_target(target):
        notes.append(_NOTE_INVALID_VOLUME.format(service))
        return None
    mode = "ro" if entry.get("read_only") else "rw"
    return {"name": source.lower(), "target": target, "mode": mode}


def _is_host_path(source: str) -> bool:
    return source.startswith(("/", ".", "~"))


def _is_named_source(source: str) -> bool:
    return bool(re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_.-]{0,62}", source))


def _is_absolute_target(target: str) -> bool:
    return target.startswith("/") and ".." not in target.split("/")


def _variables(value: Any, service: str, notes: list[str]) -> list[dict[str, Any]]:
    pairs = _environment_pairs(value, service)
    result: list[dict[str, Any]] = []
    seen: set[str] = set()
    for raw_key, raw_value in pairs:
        name = str(raw_key).strip().upper()
        if not _SAFE_VARIABLE.fullmatch(name):
            notes.append("service {}: an unusable variable name was skipped".format(service))
            continue
        if name in seen:
            notes.append("service {}: a duplicate variable was skipped".format(service))
            continue
        seen.add(name)
        result.append(_classify_variable(name, raw_value, service, notes))
    return result


def _environment_pairs(value: Any, service: str) -> list[tuple[str, Any]]:
    if value is None:
        return []
    if isinstance(value, dict):
        pairs = list(value.items())
    elif isinstance(value, list):
        pairs = []
        for item in value:
            if not isinstance(item, str):
                continue
            key, sep, val = item.partition("=")
            pairs.append((key, val if sep else None))
    else:
        raise ComposeEvidenceError(
            "compose service {} has invalid environment".format(service)
        )
    if len(pairs) > MAX_ENV_PER_SERVICE:
        raise ComposeEvidenceError(
            "compose service {} declares too many variables".format(service)
        )
    return pairs


def _classify_variable(
    name: str, raw_value: Any, service: str, notes: list[str]
) -> dict[str, Any]:
    if _is_secret_name(name):
        # A managed credential: never carry the web literal, operator supplies.
        return {"name": name, "secret": True, "required": True}
    if raw_value is None:
        return {"name": name, "secret": False, "required": False}
    default = _safe_default(raw_value)
    if default is None:
        notes.append(
            "service {}: value for {} not carried; operator supplies".format(service, name)
        )
        return {"name": name, "secret": False, "required": True}
    return {"name": name, "secret": False, "required": False, "default": default}


def _is_secret_name(name: str) -> bool:
    return any(token in name for token in _SECRET_NAME_TOKENS)


def _safe_default(value: Any) -> str | None:
    """Return a compose value ONLY if it is a plain, non-credential scalar.

    Fails closed: a URI/DSN (``://``), an ``@``-bearing value, a ``k=v`` pair,
    a JWT, a PEM/``token:`` marker, an over-long value, or a high-entropy token
    yields ``None`` (the caller emits the variable with no default). This is
    what stops ``DATABASE_URL=postgres://u:p@h/db``, a ``SENTRY_DSN``, a bare
    JWT, or a random API key from ever becoming a web-sourced default.
    """
    if isinstance(value, bool):
        return "true" if value else "false"
    if isinstance(value, (int, float)):
        return str(value)
    if not isinstance(value, str):
        return None
    text = value
    if not text or len(text) > MAX_DEFAULT_LEN:
        return None
    # POSITIVE whitelist: carry a value only when it is a plain scalar of safe
    # characters. This one check rejects ``:`` (host:port / user:pass), ``://``,
    # ``@``, ``=`` (k=v), whitespace, control chars, and ``#``/``~``/``%`` — so a
    # credential can never slip through on a value whose NAME the list missed.
    # (The prior blocklist let a bare ``:`` through, and its entropy gate was
    # inverted for out-of-charset values — the security review's Finding 1.)
    if not re.fullmatch(r"[A-Za-z0-9._/\-]+", text):
        return None
    if _SECRET_MARKER.search(text) or _JWT.fullmatch(text):
        return None
    if _is_high_entropy(text):
        return None
    return text


def _is_high_entropy(text: str) -> bool:
    if len(text) < 20 or not re.fullmatch(r"[A-Za-z0-9+/_=.\-]+", text):
        return False
    counts = Counter(text)
    length = len(text)
    entropy = -sum((n / length) * math.log2(n / length) for n in counts.values())
    return entropy > 3.5
