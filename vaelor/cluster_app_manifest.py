"""Pure renderer: an approved researched manifest -> per-service Swarm specs.

This is the D4a-render foundation. A researched application is a MULTI-service
manifest (`application_deployments.normalize_manifest`) whose `build_compose`
emits a single-node `docker compose` project; the cluster deploy needs, instead,
one `docker service create` per service on a shared overlay network. This module
turns the approved draft's ``manifest`` + ``compose`` into an ordered list of
per-service create-specs plus the overlay-network name and the app-group id, and
nothing else: it computes over the passed dicts and calls the pure
`compose_policy.validate_normalized`. It runs NO docker, NO subprocess, NO
network, writes NO file, and never reaches the draft store or the credential
broker — every impure step (network create, per-service 0600 env-files, the N
`service create` calls, atomic rollback) is D4a-deploy's, which consumes the
`ServiceSpec` list this produces.

Two things it owns that later slices lean on:

* **The naming producer (B1/B2).** `app_service_name` is the ONE deterministic
  function that mints a Swarm service name for an ``(app_id, service_key)`` pair.
  Charset is ``[a-z0-9-]`` only (Swarm's own name rule and the widened ops
  regexes), ≤63 chars (Swarm's hard limit), and content-hash-suffixed when a
  truncation is forced so two distinct logical names never collapse to one — the
  same truncate-plus-digest discipline `application_deployments.compose_project_name`
  uses. `app_group_id` and `app_network_name` share it. What it CANNOT guarantee
  by construction is that a researched ``vaelor-app-<app>`` never equals a live
  catalog ``vaelor-app-<name>`` or a second researched app; the impure caller
  (D4a-deploy) must reject an app id that collides with a live service name. Here
  the job is only determinism and exposing the names.

* **The security foundation (B5/B6/Q8).** Before any spec is built the renderer
  re-runs ``validate_normalized`` (Q8 belt-and-suspenders — the same backstop the
  single-node import runs, propagated verbatim), then ACTIVELY refuses every bind
  mount including the allowlisted docker.sock (B6 — a host path cannot follow a
  Swarm task across nodes, and ``validate_normalized`` PERMITS that bind so it
  would not catch it), then rejects duplicate ingress published ports within the
  app (B5). It fails closed with a plain-language reason and never emits a spec
  for anything unsafe.
"""

from __future__ import annotations

import hashlib
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional

from .compose_policy import (
    MAX_MEMORY_LIMIT_BYTES,
    MIN_MEMORY_LIMIT_BYTES,
    validate_normalized,
)
from .cluster_service_reconfigure import build_label_constraints

#: Every Vaelor-managed cluster service and the app overlay network start here.
#: Eleven characters, so the variable part that follows has 52 to spend before
#: the Swarm ceiling — the exact figure the widened ops regexes admit.
_SERVICE_PREFIX = "vaelor-app-"

#: Swarm's hard limit on a service (and network) name. A name that would exceed
#: it is truncated and hashed rather than rejected, so a long app/service pair
#: still deploys under a unique, valid name.
_MAX_NAME = 63

#: Hex characters of the content digest appended when a name must be truncated.
#: Eight keeps the collision space at 2**32 while leaving room for a readable
#: prefix; it mirrors the digest-suffix `compose_project_name` falls back to.
_HASH_LEN = 8

#: The app-level memory figure, in MiB, that the compose policy will accept.
#: These are ``MIN_MEMORY_LIMIT_BYTES``/``MAX_MEMORY_LIMIT_BYTES`` expressed in
#: MiB, so a per-service override is validated against the SAME 64 MiB – 256 GiB
#: bound the executor backstop enforces, not a second copy of the numbers.
_MIB = 1024 ** 2
_MIN_MEMORY_MIB = MIN_MEMORY_LIMIT_BYTES // _MIB
_MAX_MEMORY_MIB = MAX_MEMORY_LIMIT_BYTES // _MIB

#: The memory figure `application_deployments.normalize_manifest` writes when
#: research did NOT determine a footprint — 512 MiB, its `_integer` default for
#: ``resources.memory_bytes``. A rendered service whose limit came from this
#: (with no per-service override) is flagged ``memory_defaulted`` so the fit and
#: the plan can say "using the 512 MB default — set a limit" rather than present
#: a guessed footprint as a measured one. It is a heuristic (research CAN pick
#: 512 MiB deliberately), but an honest, bounded one the design accepted.
MANIFEST_DEFAULT_MEMORY_BYTES = 536870912

#: The literal placeholder `build_compose` writes for a managed-credential
#: variable — ``${VAELOR_CREDENTIAL_<NAME>:?managed credential required}``. Its
#: presence in an environment VALUE marks that key as a secret D4a-deploy must
#: resolve through the broker to a 0600 ``--env-file`` (B4); the literal token
#: never rides an argv.
_CREDENTIAL_PLACEHOLDER = re.compile(r"\$\{VAELOR_CREDENTIAL_[A-Z][A-Z0-9_]*")


class ClusterAppManifestError(ValueError):
    """A plain-language refusal to render a researched app for the cluster.

    Every raise carries a message fit to show an operator: a bind mount that
    cannot follow a Swarm task, two services fighting for one published port, or
    the compose-policy backstop's own verbatim reason.
    """


def _slug_segment(value: Any) -> str:
    """Fold one identity string to the ``[a-z0-9-]`` charset.

    Lowercases, replaces every run of characters outside ``[a-z0-9]`` (an
    underscore, a dot, a space) with a single hyphen — folding ``_``->``-`` and
    collapsing repeats in one pass — and strips leading/trailing hyphens. A
    source with no alphanumeric content returns ``""`` and lets the caller fall
    back to a digest so a name is never empty.
    """
    text = re.sub(r"[^a-z0-9]+", "-", str(value or "").lower())
    return text.strip("-")


def _bounded_name(prefix: str, variable: str, suffix: str = "") -> str:
    """A ``[a-z0-9-]`` name of ``prefix + variable + suffix`` capped at 63 chars.

    When the assembled name fits it is returned as-is (the D3-compatible plain
    shape). When it would exceed the Swarm limit the variable part is truncated
    and a content digest of the WHOLE untruncated name is appended, so two names
    that share a truncated prefix stay distinct and the result is still unique
    and valid. Deterministic: the same inputs always yield the same name.
    """
    logical = f"{prefix}{variable}{suffix}"
    if len(logical) <= _MAX_NAME:
        return logical
    digest = hashlib.sha256(logical.encode("utf-8")).hexdigest()[:_HASH_LEN]
    budget = _MAX_NAME - len(prefix) - len(suffix) - 1 - _HASH_LEN
    truncated = variable[: max(budget, 0)].rstrip("-")
    return f"{prefix}{truncated}-{digest}{suffix}"


def app_group_id(app_id: Any) -> str:
    """The stable ``vaelor.app-group`` label value for an application id.

    A ``[a-z0-9-]`` slug of the manifest ``application.id`` (already slug-checked
    upstream), bounded to the service-name ceiling through the SAME
    truncate-plus-digest discipline `_bounded_name` uses: a slug that fits is
    returned as-is (the plain shape), and a slug that would exceed the cap is
    truncated with a content digest of the WHOLE untruncated slug appended, so
    two ids that differ anywhere — even only in a tail past the truncation point —
    produce DIFFERENT group ids. This matters because the Deployments view and
    the app-level remove group services by this label; a collision here would let
    removing one app tear down another's services. Falls back to a digest for the
    degenerate case of an id with no alphanumeric content.
    """
    slug = _slug_segment(app_id)
    if not slug:
        slug = "app-" + hashlib.sha256(
            str(app_id or "").encode("utf-8")
        ).hexdigest()[:_HASH_LEN]
    return _bounded_name("", slug)


def app_service_name(
    app_id: Any, service_key: Any, *, single_service: bool
) -> str:
    """The Swarm service name for one manifest service (B1/B2).

    Single-service apps keep the D3/catalog-compatible ``vaelor-app-<app>`` shape
    (when it fits); a multi-service app names each service
    ``vaelor-app-<app>-<svc>``. The whole name is ``[a-z0-9-]`` only and ≤63
    chars, truncated-and-hashed when the app/service pair is too long. Distinct
    ``(app, service_key)`` pairs that both fit produce distinct names; the
    truncation path stays collision-safe through the content digest. The
    single-service plain name cannot be proven distinct from a live catalog
    ``vaelor-app-<name>`` here — that check is the impure caller's (D4a-deploy).
    """
    app = _slug_segment(app_id)
    if single_service:
        variable = app
    else:
        service = _slug_segment(service_key)
        variable = re.sub(r"-+", "-", f"{app}-{service}").strip("-")
    return _bounded_name(_SERVICE_PREFIX, variable)


def app_network_name(app_id: Any) -> str:
    """The attachable overlay network name for an app: ``vaelor-app-<app>-net``.

    Derived from the SAME `app_group_id` the app-level remove computes the network
    name from (`app_network_name_for_group`), NOT from the raw slug: at a 63/64-char
    id the raw slug and the bounded group id differ, so deriving from the slug here
    would let deploy create one network while remove looked for another — orphaning
    the real network and falsely reporting ``network_removed: true``. Deriving both
    from the group id makes the deploy name and the remove name IDENTICAL. Same
    producer discipline as the service names, keeping the ``-net`` suffix intact
    even under truncation, so the network name is always ``[a-z0-9-]`` and ≤63 chars.
    """
    return _bounded_name(_SERVICE_PREFIX, app_group_id(app_id), suffix="-net")


def app_network_name_for_group(app_group: Any) -> str:
    """The overlay network name for an app-group label, for the app-level teardown.

    The app-level remove knows the app only by its ``vaelor.app-group`` label read
    off the live services (it never has the raw manifest id). For an ordinary app
    the group IS ``_slug_segment(application.id)``, so this reproduces
    `app_network_name` exactly; the label is already ``[a-z0-9-]`` so it is not
    re-slugged. Deterministic and ≤63 chars through the same producer."""
    return _bounded_name(_SERVICE_PREFIX, str(app_group), suffix="-net")


@dataclass
class ServiceSpec:
    """Exactly what an eventual ``docker service create`` needs for one service.

    D4a-deploy consumes this; nothing here runs docker. ``service_key`` is the
    manifest service name and doubles as the overlay-network ALIAS, so a sibling
    reaches this service by the compose name the app expects (e.g. ``db``).
    ``image`` is the digest-pinned reference. ``published_ports`` are ingress
    publishings; ``host_ports`` (host-mode) are kept separate and never surfaced
    as the app's URL. ``secret_env_vars`` names the environment keys whose value
    is a ``${VAELOR_CREDENTIAL_*}`` placeholder for D4a-deploy to resolve to a
    0600 env-file; ``plain_env`` carries the non-secret literals. ``stateful`` is
    true when the service keeps a named volume — it pins, it never spreads.
    ``placement`` records the pin node (stateful) or app intent (stateless) taken
    from ``placements``. ``memory_defaulted`` is true when ``memory_mib`` came
    from the manifest's 512 MB default (no researched footprint, no per-service
    override), so the fit and plan surface that honestly rather than as a figure.
    """

    service_name: str
    service_key: str
    image: str
    published_ports: List[Dict[str, Any]] = field(default_factory=list)
    host_ports: List[Dict[str, Any]] = field(default_factory=list)
    volumes: List[Dict[str, Any]] = field(default_factory=list)
    memory_mib: int = 0
    memory_defaulted: bool = False
    cpu: float = 0.0
    labels: Dict[str, str] = field(default_factory=dict)
    secret_env_vars: List[str] = field(default_factory=list)
    plain_env: Dict[str, str] = field(default_factory=dict)
    stateful: bool = False
    placement: Dict[str, Any] = field(default_factory=dict)


@dataclass
class RenderedApp:
    """The whole researched app rendered for the cluster: the per-service specs
    (in manifest order), the one overlay network they share, and the app-group
    label value that groups them in the Deployments view."""

    app_group: str
    network_name: str
    services: List[ServiceSpec] = field(default_factory=list)


def _default_memory_mib(manifest: Dict[str, Any]) -> int:
    """The app-level memory figure in MiB, from ``resources.memory_bytes``.

    The manifest resources are app-level (one figure applied to every service by
    `build_compose`); D4 defaults each service to it and lets the operator adjust
    per service. The stored figure already passed `normalize_manifest`'s
    64 MiB – 256 GiB bound, so the conversion cannot fall outside range.
    """
    resources = manifest.get("resources") if isinstance(manifest, dict) else None
    resources = resources if isinstance(resources, dict) else {}
    try:
        memory_bytes = int(resources.get("memory_bytes", MIN_MEMORY_LIMIT_BYTES))
    except (TypeError, ValueError):
        memory_bytes = MIN_MEMORY_LIMIT_BYTES
    return max(memory_bytes // _MIB, _MIN_MEMORY_MIB)


def _default_cpu(manifest: Dict[str, Any]) -> float:
    resources = manifest.get("resources") if isinstance(manifest, dict) else None
    resources = resources if isinstance(resources, dict) else {}
    try:
        return float(resources.get("cpu_cores", 1) or 1)
    except (TypeError, ValueError):
        return 1.0


def _service_memory_mib(default_mib: int, entry: Dict[str, Any]) -> int:
    """The per-service memory limit: the app default unless the operator set a
    per-service override, validated to the compose policy's 64 MiB – 256 GiB."""
    override = entry.get("memory_mib")
    if override is None:
        return default_mib
    try:
        value = int(override)
    except (TypeError, ValueError) as error:
        raise ClusterAppManifestError(
            "A per-service memory limit must be a whole number of MiB."
        ) from error
    if not _MIN_MEMORY_MIB <= value <= _MAX_MEMORY_MIB:
        raise ClusterAppManifestError(
            "A per-service memory limit must be between {} and {} MiB.".format(
                _MIN_MEMORY_MIB, _MAX_MEMORY_MIB
            )
        )
    return value


def _app_memory_is_default(manifest: Dict[str, Any]) -> bool:
    """True when the app-level memory is the manifest's un-researched 512 MB
    default — the signal that research could not determine a footprint."""
    resources = manifest.get("resources") if isinstance(manifest, dict) else None
    resources = resources if isinstance(resources, dict) else {}
    try:
        return int(resources.get(
            "memory_bytes", MANIFEST_DEFAULT_MEMORY_BYTES
        )) == MANIFEST_DEFAULT_MEMORY_BYTES
    except (TypeError, ValueError):
        return True


def _service_cpu(default_cpu: float, entry: Dict[str, Any]) -> float:
    override = entry.get("cpu")
    if override is None:
        return default_cpu
    try:
        value = float(override)
    except (TypeError, ValueError) as error:
        raise ClusterAppManifestError(
            "A per-service CPU limit must be a number."
        ) from error
    if not 0 < value <= 256:
        raise ClusterAppManifestError(
            "A per-service CPU limit must be between 0 and 256 cores."
        )
    return value


def _refuse_bind_mounts(services: Dict[str, Any]) -> None:
    """Refuse ANY bind mount, the allowlisted docker.sock included (B6).

    ``validate_normalized`` PERMITS the allowlisted ``/var/run/docker.sock`` bind
    and confines other binds to the workload root — correct for a single machine,
    wrong for a cluster, where a task can land on any node and a host path there
    is a different (or absent) directory. So the renderer refuses every bind
    outright with a plain reason, before it looks at anything else.
    """
    for name, service in services.items():
        if not isinstance(service, dict):
            continue
        for volume in service.get("volumes", []) or []:
            if not isinstance(volume, dict):
                continue
            source = str(volume.get("source", ""))
            is_bind = volume.get("type") == "bind" or source.startswith("/")
            if is_bind:
                raise ClusterAppManifestError(
                    "Service {} uses the host path {}, which cannot follow the "
                    "service across cluster machines. Deploy this app on a "
                    "single machine instead.".format(name, source or "(unset)")
                )


def _port_mode(port: Dict[str, Any]) -> str:
    return str(port.get("mode", "ingress")).strip().lower() or "ingress"


def _refuse_duplicate_published_ports(services: Dict[str, Any]) -> None:
    """Reject two services in this app asking for the same ingress port (B5).

    Swarm's routing mesh publishes an ingress port fleet-wide, so two services
    of one app both binding published ``8080/tcp`` cannot both come up. The
    cross-fleet live-port collision (against other running apps) is impure and
    belongs to the deploy slice; this is the within-app check, keyed on the
    published port and protocol so a tcp and a udp service may share a number.
    """
    seen: Dict[tuple, str] = {}
    for name, service in services.items():
        if not isinstance(service, dict):
            continue
        for port in service.get("ports", []) or []:
            if not isinstance(port, dict) or _port_mode(port) == "host":
                continue
            try:
                published = int(port.get("published") or 0)
            except (TypeError, ValueError):
                published = 0
            if not published:
                continue
            protocol = str(port.get("protocol", "tcp")).strip().lower() or "tcp"
            key = (published, protocol)
            if key in seen:
                raise ClusterAppManifestError(
                    "Services {} and {} both publish port {}/{}; each ingress "
                    "port can be published by only one service.".format(
                        seen[key], name, published, protocol
                    )
                )
            seen[key] = name


def _split_ports(service: Dict[str, Any]) -> tuple:
    """Partition a service's compose ports into ingress and host-mode lists."""
    published: List[Dict[str, Any]] = []
    host: List[Dict[str, Any]] = []
    for port in service.get("ports", []) or []:
        if not isinstance(port, dict):
            continue
        entry = {
            "published": port.get("published"),
            "target": port.get("target"),
            "protocol": str(port.get("protocol", "tcp")).strip().lower() or "tcp",
        }
        (host if _port_mode(port) == "host" else published).append(entry)
    return published, host


def _named_volumes(service: Dict[str, Any]) -> List[Dict[str, Any]]:
    """The service's node-local named volumes as ``{name, target, mode}``.

    Only ``type == "volume"`` entries; bind mounts have already been refused, so
    a survivor here is a managed named volume that pins the service to one node.
    """
    volumes: List[Dict[str, Any]] = []
    for volume in service.get("volumes", []) or []:
        if not isinstance(volume, dict) or volume.get("type") != "volume":
            continue
        volumes.append({
            "name": str(volume.get("source", "")),
            "target": str(volume.get("target", "")),
            "mode": "ro" if volume.get("read_only") else "rw",
        })
    return volumes


def _partition_env(service: Dict[str, Any]) -> tuple:
    """Split a service's environment into secret placeholder KEYS and plain env.

    A value carrying the ``${VAELOR_CREDENTIAL_*}`` placeholder marks a secret
    key D4a-deploy resolves through the broker; everything else is a literal that
    is safe to keep. The secret's literal token is never copied into ``plain_env``.
    """
    environment = service.get("environment")
    environment = environment if isinstance(environment, dict) else {}
    secret_keys: List[str] = []
    plain: Dict[str, str] = {}
    for key, value in environment.items():
        if isinstance(value, str) and _CREDENTIAL_PLACEHOLDER.search(value):
            secret_keys.append(str(key))
        else:
            plain[str(key)] = "" if value is None else str(value)
    return sorted(secret_keys), plain


def _placement_for(
    service_key: str, stateful: bool, placements: Dict[str, Any]
) -> Dict[str, Any]:
    """The placement intent for one service, defaulted honestly.

    A stateful service (a named volume) pins to a data node — the operator picks
    the node later, so ``pin_node`` may be ``None`` here. A stateless service
    follows the app-level intent, defaulting to run-once. Any per-service label
    constraints the operator supplied ride along untouched.
    """
    entry = placements.get(service_key)
    entry = entry if isinstance(entry, dict) else {}
    if stateful:
        placement: Dict[str, Any] = {
            "intent": "pin",
            "pin_node": entry.get("pin_node"),
        }
    else:
        intent = str(entry.get("intent", "") or "").strip().lower() or "run-once"
        placement = {"intent": intent}
    constraints = entry.get("label_constraints")
    if constraints:
        # Validate operator label constraints the SAME way the catalog configure
        # path does, so a malformed one ({key, op, value} with a bad op, a
        # reserved vaelor.*/pironman.* key, or a wrong charset) is an honest
        # refusal here — before any Swarm object — rather than a confusing Swarm
        # error at `service create`. Returns the rendered node.labels.<k><op><v>
        # expressions the deploy emits.
        try:
            placement["label_constraints"] = build_label_constraints(constraints)
        except ValueError as error:
            raise ClusterAppManifestError(
                getattr(error, "message", None) or str(error)
            ) from error
    return placement


def render_app_services(
    manifest: Dict[str, Any],
    compose: Dict[str, Any],
    placements: Optional[Dict[str, Any]] = None,
    *,
    workloads_root: Path,
) -> RenderedApp:
    """Render an approved researched draft into per-service Swarm create-specs.

    PURE. ``manifest`` and ``compose`` are the resolved approved draft's fields
    (the caller resolves them via ``resolve_import`` and checks integrity — this
    module does not touch the store). ``placements`` maps a manifest service key
    to ``{intent | pin_node, memory_mib?, cpu?, label_constraints?}``; absent
    entries default stateful->pin and stateless->run-once. ``workloads_root`` is
    only handed to ``validate_normalized``; nothing is read from or written to it.

    Order of operations, fail-closed with honest messages:

    a. re-run ``validate_normalized`` and propagate its reason verbatim (Q8);
    b. refuse every bind mount, docker.sock included (B6);
    c. reject duplicate ingress published ports within the app (B5);
    d. build one ``ServiceSpec`` per service, alias = the manifest service key.
    """
    if not isinstance(compose, dict):
        raise ClusterAppManifestError("The application compose is missing.")
    services = compose.get("services")
    if not isinstance(services, dict) or not services:
        raise ClusterAppManifestError(
            "The application defines no services to deploy."
        )
    placements = placements if isinstance(placements, dict) else {}
    manifest = manifest if isinstance(manifest, dict) else {}

    # (a) The executor-side backstop, re-run here so a draft that reached the
    # renderer without it is refused with the policy's own words, not silently
    # trusted. ValueError from the policy is the honest reason to surface.
    try:
        # No model claims here (W7-2): a cluster app is published through Swarm
        # ingress, which this pure renderer cannot read the controller's stored
        # model ports for. The deploy's own port checks stand; residual noted.
        validate_normalized(dict(compose), Path(workloads_root), model_ports={})
    except ValueError as error:
        raise ClusterAppManifestError(str(error)) from error

    # (b) and (c): both are whole-app refusals that must fire before any spec is
    # built, so an unsafe app yields no partial render.
    _refuse_bind_mounts(services)
    _refuse_duplicate_published_ports(services)

    app_id = (manifest.get("application", {}) or {}).get("id", "")
    group = app_group_id(app_id)
    single_service = len(services) == 1
    default_memory = _default_memory_mib(manifest)
    default_cpu = _default_cpu(manifest)
    app_memory_defaulted = _app_memory_is_default(manifest)

    specs: List[ServiceSpec] = []
    used_names: Dict[str, str] = {}
    for service_key, service in services.items():
        if not isinstance(service, dict):
            raise ClusterAppManifestError(
                "Service {} is not a valid definition.".format(service_key)
            )
        entry = placements.get(service_key)
        entry = entry if isinstance(entry, dict) else {}
        volumes = _named_volumes(service)
        stateful = bool(volumes)
        published_ports, host_ports = _split_ports(service)
        secret_env_vars, plain_env = _partition_env(service)
        placement = _placement_for(service_key, stateful, placements)

        name = app_service_name(
            app_id, service_key, single_service=single_service
        )
        # Two manifest service keys whose slugs collapse to one name (an edge of
        # the lossy fold, e.g. ``a-b`` vs ``a_b``) would deploy as one service and
        # silently drop the other; refuse rather than render an ambiguous app.
        if name in used_names:
            raise ClusterAppManifestError(
                "Services {} and {} resolve to the same cluster service name "
                "{}; rename one so each service is addressable.".format(
                    used_names[name], service_key, name
                )
            )
        used_names[name] = service_key

        specs.append(ServiceSpec(
            service_name=name,
            service_key=str(service_key),
            image=str(service.get("image", "")),
            published_ports=published_ports,
            host_ports=host_ports,
            volumes=volumes,
            memory_mib=_service_memory_mib(default_memory, entry),
            memory_defaulted=(
                app_memory_defaulted and entry.get("memory_mib") is None
            ),
            cpu=_service_cpu(default_cpu, entry),
            labels={
                "vaelor.app-group": group,
                "vaelor.app-service": str(service_key),
                "vaelor.app-kind": "researched",
                "vaelor.placement-intent": str(placement["intent"]),
                # Consistency with the catalog renderer's managed-label set, so a
                # researched service is discoverable as Vaelor-managed the same way
                # (and future tooling that filters on these finds it too).
                "vaelor.managed": "true",
                "vaelor.workload": "app",
            },
            secret_env_vars=secret_env_vars,
            plain_env=plain_env,
            stateful=stateful,
            placement=placement,
        ))

    if not any(spec.image for spec in specs):
        raise ClusterAppManifestError(
            "No service carries a pinned image to deploy."
        )
    return RenderedApp(
        app_group=group,
        network_name=app_network_name(app_id),
        services=specs,
    )
