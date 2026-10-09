"""Ports an app install must not take because a Vaelor model comes back on them.

W6-D2: an app install checked its host port only by binding it. A stopped
AI Chat model keeps its credential and returns on the port that names
(`gpu_chat_relaunch`, VD-179), so in Mode B that port binds as free - and the
NGINX blueprint's default, 8080, is exactly the port the Z2's stored 27B model
names. The claims are read through S1's `npu_port_claims` (F6), never
re-derived here (LESSONS 6).
"""

from __future__ import annotations

import logging
import re
import socket
from pathlib import Path
from typing import Any, Callable, Dict, FrozenSet, Iterable, List, NamedTuple, Optional

from .flm_supervisor import npu_port_claims

LOGGER = logging.getLogger(__name__)

#: Ports the control plane itself serves on.
_CONTROL_PLANE_PORTS = frozenset({34001, 34002})

logger = logging.getLogger(__name__)


def _stored_model_names(broker: Any) -> Dict[int, str]:
    """``{port: model name}`` for the stored GPU-tier models (W7-D2).

    Names only - the claims themselves stay `npu_port_claims`'s. An unreadable
    listing names nobody, and the refusal falls back to "a stored model".
    """
    from .gpu_serving_target import loopback_port
    from .managed_local_credentials import gpu_tier_models

    try:
        models = gpu_tier_models(broker)
    except Exception as error:  # noqa: BLE001 - a name is not worth a failed install
        logger.warning("The stored model names could not be read: %s", error)
        return {}
    return {
        port: name for endpoint, name in models
        if name and (port := loopback_port(endpoint))
    }


def model_port_holders(broker: Any) -> Dict[int, str]:
    """``{port: what holds it}`` for every port a stored local model claims."""
    if broker is None:
        return {}
    own, reserved = npu_port_claims(broker)
    names = _stored_model_names(broker)
    holders = {
        port: (
            f"the stored AI Chat model {names[port]}, which comes back on this port"
            if names.get(port)
            else "a stored model for AI Chat, which comes back on this port"
        )
        for port in reserved
    }
    if own:
        holders[own] = "the on-device Assistant model"
    return holders


def _binds(port: int, socket_factory) -> bool:
    try:
        with socket_factory(socket.AF_INET, socket.SOCK_STREAM) as probe:
            probe.bind(("0.0.0.0", port))
    except OSError:
        return False
    return True


def suggest_port(
    start: int, taken: Iterable[int], socket_factory=socket.socket,
) -> Optional[int]:
    """The first port above ``start`` that binds and nothing claims."""
    claimed = set(taken) | _CONTROL_PLANE_PORTS
    for candidate in range(start + 1, 65536):
        if candidate not in claimed and _binds(candidate, socket_factory):
            return candidate
    return None


def refuse_model_port(
    port: int, holders: Dict[int, str], socket_factory=socket.socket,
) -> None:
    """Refuse ``port`` when a stored model claims it, naming the holder."""
    if port not in holders:
        return
    suggestion = suggest_port(port, holders, socket_factory)
    advice = f" Use port {suggestion} instead." if suggestion else " Choose another port."
    raise ValueError(f"Port {port} is held by {holders[port]}.{advice}")


def catalog_port_offers(
    templates: List[Dict[str, Any]], holders: Dict[int, str],
    socket_factory=socket.socket,
) -> List[Dict[str, Any]]:
    """Each catalog template with the port an install would be offered (W7-D2).

    ``offered_port`` is the default unless a stored model claims it, then the
    port `refuse_model_port` would suggest; ``default_port_holder`` says what
    holds the default. The same claims and suggestion as the install's own
    refusal, so the card and the refusal cannot disagree (LESSONS 6).
    """
    offered = []
    for template in templates:
        port = template.get("default_port")
        card = dict(template, offered_port=port)
        if isinstance(port, int) and port in holders:
            card["default_port_holder"] = holders[port]
            card["offered_port"] = suggest_port(port, holders, socket_factory)
        offered.append(card)
    return offered


#: Vaelor's own model-server projects: models, claimed through their
#: credentials (`model_port_holders`), never counted as installed apps.
_MODEL_PROJECTS = frozenset({"model-assistant", "model-chat"})


class AppPortsUnknown(ValueError):
    """An installed app's ports cannot be read, so no port is known to be free.

    W7-6 (LESSONS 8): an unreadable compose, an interpolated port with no
    value, or a host-network app that declares nothing used to read as "no
    ports", and a model deploy could take whatever the app binds. A model
    deploy is refused instead, naming the app, until its ports can be read.
    """


#: Compose interpolation: ``$$`` (an escaped, literal ``$``), ``${VAR}``,
#: ``${VAR-default}`` / ``${VAR:-default}``, and bare ``$VAR``. The escape is
#: matched first so the ``$`` after it never starts a variable.
_INTERPOLATION = re.compile(r"\$\$|\$\{(\w+)(?::?-([^}]*))?\}|\$(\w+)")


def _resolve(text: str, env: Dict[str, str]) -> str:
    """``text`` with compose interpolation resolved, or `AppPortsUnknown`."""
    def value(match):
        if match.group(0) == "$$":
            return "$"
        name = match.group(1) or match.group(3)
        if name in env:
            return env[name]
        if match.group(2) is not None:
            return match.group(2)
        raise AppPortsUnknown(f"its port uses ${{{name}}}, which has no value")
    return _INTERPOLATION.sub(value, text)


def _host_range(text: str) -> List[int]:
    """The ports a host part names: one port, or an inclusive range."""
    first, _, last = text.strip().partition("-")
    if not first.isdigit() or (last and not last.isdigit()):
        return []
    low, high = int(first), int(last or first)
    if not 0 < low <= high < 65536:
        return []
    return list(range(low, high + 1))


def published_ports(binding: Any, env: Dict[str, str]) -> List[int]:
    """Every host port one compose ``ports`` entry publishes (W7-6).

    Short syntax ``[address:]host:container[/protocol]`` - the address may be
    IPv6 in brackets, so the split is from the right (W7-5) - or long syntax
    with ``published``. Ranges expand; ``${VAR}`` and ``${VAR:-default}``
    resolve from the project's ``.env``; a bare container port publishes
    nothing fixed.
    """
    if isinstance(binding, dict):
        value = binding.get("published")
        if value is None or value == "":
            return []
        return _host_range(_resolve(str(value), env))
    text = _resolve(str(binding).strip().strip('"'), env).split("/", 1)[0]
    parts = text.rsplit(":", 2)
    return _host_range(parts[-2]) if len(parts) >= 2 else []


def _project_env(project: Path) -> Dict[str, str]:
    """The ``.env`` values docker compose interpolates for ``project``."""
    env: Dict[str, str] = {}
    try:
        lines = (project / ".env").read_text(encoding="utf-8").splitlines()
    except FileNotFoundError:
        return env
    for line in lines:
        key, sep, value = line.strip().partition("=")
        if sep and key and not key.startswith("#"):
            env[key.strip()] = value.strip().strip('"').strip("'")
    return env


def _service_ports(service: Dict[str, Any], env: Dict[str, str]) -> List[int]:
    if str(service.get("network_mode") or "") == "host":
        # Host networking publishes nothing; the app binds its own ports. The
        # ones it declares are claims; declaring none is unknown, not "none".
        declared: List[int] = []
        for entry in list(service.get("ports") or ()) + list(service.get("expose") or ()):
            if isinstance(entry, dict) or ":" in str(entry):
                declared.extend(published_ports(entry, env))
            else:
                declared.extend(_host_range(_resolve(str(entry), env).split("/", 1)[0]))
        if not declared:
            raise AppPortsUnknown("it uses the host's network and declares no port")
        return declared
    return [
        port for entry in service.get("ports") or () for port in published_ports(entry, env)
    ]


def _unknown_ports(project: Path, compose: Path, why: str) -> str:
    return (
        f'The ports of the installed app "{project.name}" cannot be known: {why}. '
        f"Vaelor will not pick a model port it might take; fix or remove {compose} first."
    )


class AppPorts(NamedTuple):
    """What the installed apps' compose files say about their ports.

    ``holders`` is every port a readable app publishes; ``unknown`` names, one
    sentence per app, each app whose ports cannot be known. Both are always
    complete: an unknown app never hides the ports the others publish
    (verification of 4b5b723, LESSONS 6).
    """

    holders: Dict[int, str]
    unknown: List[str]


def read_app_ports(workloads_root: Any) -> AppPorts:
    """Every installed app's published ports, and the apps whose ports are unknown.

    W6-D2 reverse (LESSONS 6): read from each app's compose file, so a STOPPED
    app - whose port binds as free - is counted exactly like a running one.
    An app is unknown when its file is unreadable, an interpolation has no
    value, or it uses the host network and declares nothing (W7-6).
    """
    holders: Dict[int, str] = {}
    unknown: List[str] = []
    if not workloads_root:
        return AppPorts(holders, unknown)
    root = Path(str(workloads_root))
    try:
        projects = sorted(child for child in root.iterdir() if child.is_dir())
    except OSError:
        return AppPorts(holders, unknown)
    for project in projects:
        compose = project / "compose.yaml"
        if project.name in _MODEL_PROJECTS or not compose.is_file():
            continue
        found: Dict[int, str] = {}
        try:
            import yaml

            document = yaml.safe_load(compose.read_text(encoding="utf-8")) or {}
            services = (document.get("services") or {}) if isinstance(document, dict) else None
            if not isinstance(services, dict):
                raise AppPortsUnknown("its services could not be read")
            env = _project_env(project)
            for service in services.values():
                for port in _service_ports(service if isinstance(service, dict) else {}, env):
                    found.setdefault(port, f'the installed app "{project.name}"')
        except AppPortsUnknown as error:
            unknown.append(_unknown_ports(project, compose, str(error)))
            continue
        except Exception as error:  # noqa: BLE001 - unreadable YAML, permissions
            unknown.append(_unknown_ports(
                project, compose, f"the file cannot be read ({error.__class__.__name__})",
            ))
            continue
        for port, holder in found.items():
            holders.setdefault(port, holder)
    return AppPorts(holders, unknown)


def app_port_holders(workloads_root: Any) -> Dict[int, str]:
    """``{port: what holds it}`` for every port an installed app publishes.

    Raises `AppPortsUnknown` naming the first app whose ports cannot be known.
    Refusing the model deploy is the choice, rather than claiming the whole
    model band with a warning: a band claim refuses the same deploys while
    hiding which file to fix.
    """
    ports = read_app_ports(workloads_root)
    if ports.unknown:
        raise AppPortsUnknown(ports.unknown[0])
    return ports.holders


_LOGGED_UNKNOWN: set = set()


def app_port_holders_keeping(
    workloads_root: Any, own_port: Optional[int], reserved: Iterable[int] = (),
) -> FrozenSet[int]:
    """`app_port_holders` for a model that keeps the port its lease names.

    Review of a47d4a4 (LESSONS 22): with ``own_port`` set, an app whose ports
    are unknown does not refuse - the model held that port first, and refusing
    would let an unrelated app keep it down after a crash or a redeploy. Only
    the unknown apps are waived (logged once each); every port a readable app
    publishes is still claimed (verification of 4b5b723). With no usable
    ``own_port`` (none, or one another model's credential ``reserved`` names)
    the model must choose a new port, and `AppPortsUnknown` stands.
    """
    ports = read_app_ports(workloads_root)
    if ports.unknown:
        if not own_port or own_port in set(reserved):
            raise AppPortsUnknown(ports.unknown[0])
        for message in ports.unknown:
            if message not in _LOGGED_UNKNOWN:
                _LOGGED_UNKNOWN.add(message)
                logger.warning("Keeping model port %s despite: %s", own_port, message)
    return frozenset(ports.holders)


def refuse_claimed_ports(
    services: Any, holders: Dict[int, str], env: Optional[Dict[str, str]] = None,
    socket_factory=socket.socket,
) -> None:
    """Refuse an app whose compose publishes a port a stored model claims (W7-2).

    The one refusal for every app install path - blueprint, import, researched
    app, configuration edit, checkpoint restore - read with the same
    `published_ports` the reverse direction uses (LESSONS 6).
    """
    if not holders or not isinstance(services, dict):
        return
    for service in services.values():
        if isinstance(service, dict):
            for port in _service_ports_or_none(service, env or {}):
                refuse_model_port(port, holders, socket_factory)


def refuse_claimed_compose(compose_file: Path, holders: Dict[int, str]) -> None:
    """`refuse_claimed_ports` for a compose file on disk (a checkpoint restore).

    The file is read with its project's ``.env``, as docker compose reads it.
    """
    if not holders:
        return
    import yaml

    document = yaml.safe_load(Path(compose_file).read_text(encoding="utf-8")) or {}
    services = document.get("services") if isinstance(document, dict) else None
    refuse_claimed_ports(services, holders, _project_env(Path(compose_file).parent))


def _service_ports_or_none(service: Dict[str, Any], env: Dict[str, str]) -> List[int]:
    """`_service_ports`, where a host-network app declaring nothing claims none."""
    try:
        return _service_ports(service, env)
    except AppPortsUnknown:
        return []


def pick_model_port(
    requested: int, workloads_root: Any, allocate: Callable[..., int],
    reserved: Iterable[int] = (),
) -> int:
    """The port a model deploy serves on, never one an installed app publishes.

    An explicit ``requested`` port an app publishes is refused naming the app;
    otherwise ``allocate(exclude=...)`` picks past every app's port and
    ``reserved`` (the stored models' claims, where the caller keeps them).
    """
    apps = app_port_holders(workloads_root)
    if requested:
        if int(requested) in apps:
            raise ValueError(
                f"Port {int(requested)} is published by {apps[int(requested)]}, which "
                "starts on it again. Choose another port for the model."
            )
        return int(requested)
    claimed = frozenset(apps) | frozenset(reserved)
    # With nothing claimed the allocator is called bare, so a seam that takes no
    # ``exclude`` (a supervisor's injected picker) still works unchanged.
    return int(allocate(exclude=claimed) if claimed else allocate())
