"""Shared safety policy for imported and subsequently edited Compose projects."""

from __future__ import annotations

import re
from pathlib import Path
from typing import Any, Dict


FORBIDDEN_KEYS = {
    "cap_add", "cgroup", "cgroup_parent", "configs", "credential_spec",
    "devices", "device_cgroup_rules", "env_file", "extends", "include",
    "ipc", "pid", "privileged", "runtime", "secrets", "security_opt",
    "sysctls", "userns_mode", "uts", "volumes_from",
}

#: Namespace keys a custom service may not override, each with the name it is
#: read back by.
#:
#: The name is written beside the key rather than derived from it.
#: ``key.replace("_", " ").capitalize()`` was here, and `str.capitalize`
#: lower-cases everything after the first character - so an owner's refusal
#: read *"Ipc namespace overrides are blocked."* and *"Pid namespace overrides
#: are blocked."*, two initialisms flattened by a formatting call in the
#: sentence that tells somebody why their import was rejected.
#: `agent_naming.draft_agent_name` already keeps an acronym set for the same
#: reason; three keys do not need one, and pairing the label with the key here
#: means there is no second list to drift.
BLOCKED_NAMESPACES = (
    ("network_mode", "Network mode"),
    ("pid", "PID"),
    ("ipc", "IPC"),
)

#: Sane absolute bounds for a single custom service's resource limits.
#:
#: These are the *same* bounds the researched-application path already enforces
#: in ``application_deployments`` (``normalize_manifest``/``build_compose`` pass
#: exactly ``67108864``/``274877906944`` to ``_integer`` for memory and ``256``
#: for CPU cores). The raw ``compose.import`` path skips ``build_compose`` and
#: reached ``validate_normalized`` with only a truthiness check, so a service
#: could ask for ``mem_limit: 999999999999g`` and deploy "healthy" (#205
#: finding 2). Duplicating the numbers in two mechanisms is exactly LESSONS 6 /
#: #98 - two lists of one fact that drift - so ``tests/test_compose_policy.py``
#: asserts these equal the ceiling ``build_compose`` behaviourally enforces.
#:
#: This is a sanity bound, not a fit check: whether a limit fits *this* machine
#: is decided later against measured hardware (``application_validation`` and the
#: model sizing). The job here is only to refuse the physically absurd before
#: Docker is handed it.
MIN_MEMORY_LIMIT_BYTES = 67108864  # 64 MiB
MAX_MEMORY_LIMIT_BYTES = 274877906944  # 256 GiB
MAX_CPU_LIMIT = 256

#: Host paths a custom application may bind-mount despite the blanket refusal of
#: host binds (a bind must otherwise stay inside the managed workload root, and
#: anything matching ``docker.sock`` is refused outright). Each allowlisted entry
#: carries the constraints that make that specific mount defensible.
#:
#: Mounting any of these hands the container privileged reach into the host - the
#: Docker socket in particular is root-equivalent host control even when the bind
#: is read-only, because a socket carries a bidirectional API and the read-only
#: flag governs the inode, not the conversation. So this is deliberately a *small
#: closed allowlist of legitimately-needed mounts*, never an open door: an
#: operator can grant one of these, and nothing else, and only read-only where
#: the entry says so.
#:
#: This is the single source of truth for "which host mounts are ever
#: legitimate". Both the guided-config assembler
#: (``application_deployments.build_compose``, which additionally refuses to emit
#: any of these without an explicit per-mount operator consent flag) and this
#: executor backstop read it, so the surface an operator can request and the
#: surface the privileged importer will accept cannot drift apart (LESSONS 6 /
#: #98). ``tests/test_compose_policy.py`` pins that they agree.
PRIVILEGED_HOST_MOUNTS = {
    "/var/run/docker.sock": {"read_only": True},
}

_MEMORY_UNITS = {
    "": 1, "b": 1,
    "k": 1024, "kb": 1024, "kib": 1024,
    "m": 1024 ** 2, "mb": 1024 ** 2, "mib": 1024 ** 2,
    "g": 1024 ** 3, "gb": 1024 ** 3, "gib": 1024 ** 3,
    "t": 1024 ** 4, "tb": 1024 ** 4, "tib": 1024 ** 4,
}


def _memory_limit_bytes(value: Any) -> int | None:
    """Parse a Compose memory limit (plain bytes or a k/m/g suffix) to bytes.

    ``docker compose config`` normalises to a plain byte count, but a hand-built
    dict or the deprecated ``mem_limit`` shorthand can still carry ``256m`` /
    ``2g``, so both are accepted. ``None`` means "not a size", which the caller
    reports rather than silently treating as unbounded.
    """
    if isinstance(value, bool):
        return None
    if isinstance(value, (int, float)):
        return int(value) if value >= 0 else None
    if not isinstance(value, str):
        return None
    match = re.fullmatch(r"\s*(\d+(?:\.\d+)?)\s*([a-zA-Z]*)\s*", value)
    if not match:
        return None
    unit = match.group(2).lower()
    if unit not in _MEMORY_UNITS:
        return None
    return int(float(match.group(1)) * _MEMORY_UNITS[unit])


def _cpu_limit_value(value: Any) -> float | None:
    """Parse a Compose CPU limit (int, float, or numeric string) to a float."""
    if isinstance(value, bool):
        return None
    if isinstance(value, (int, float)):
        return float(value)
    if isinstance(value, str):
        try:
            return float(value.strip())
        except ValueError:
            return None
    return None


def reject_secret_lines(content: str) -> None:
    if re.search(
        r"(?im)^\s*(?:password|passwd|token|secret|api[_-]?key)\s*:\s*\S+",
        content,
    ):
        raise ValueError("Store credentials in the credential broker, not Compose.")


def reject_unsafe_keys(content: str) -> None:
    match = re.search(
        r"(?im)^\s*({})\s*:".format("|".join(sorted(FORBIDDEN_KEYS))),
        content,
    )
    if match:
        raise ValueError(
            "Custom Compose imports cannot use the '{}' setting.".format(
                match.group(1)
            )
        )


def validate_normalized(
    normalized: Dict[str, Any], workloads_root: Path
) -> None:
    workloads_root = workloads_root.resolve()
    for key in ("configs", "secrets", "include"):
        if normalized.get(key):
            raise ValueError("Custom Compose imports cannot use {}.".format(key))
    services = normalized.get("services")
    if not isinstance(services, dict) or not services:
        raise ValueError("Compose must define at least one service.")
    for name, service in services.items():
        if not isinstance(service, dict):
            raise ValueError("Service {} is invalid.".format(name))
        if service.get("privileged"):
            raise ValueError("Privileged containers are blocked.")
        for key in (
            "cap_add", "cgroup", "cgroup_parent", "configs",
            "credential_spec", "device_cgroup_rules", "env_file", "extends",
            "runtime", "secrets", "security_opt", "sysctls", "userns_mode",
            "uts", "volumes_from",
        ):
            if service.get(key):
                raise ValueError(
                    "Custom services cannot use {}.".format(key.replace("_", " "))
                )
        for key, named in BLOCKED_NAMESPACES:
            if service.get(key):
                raise ValueError(
                    "{} namespace overrides are blocked.".format(named)
                )
        if service.get("devices"):
            raise ValueError("Arbitrary device access is blocked.")
        if not service.get("image") or service.get("build"):
            raise ValueError("Custom imports must use a published container image.")
        for volume in service.get("volumes", []) or []:
            if not isinstance(volume, dict):
                raise ValueError(
                    "Volume definitions must be normalized and explicit."
                )
            source = str(volume.get("source", ""))
            allowed = PRIVILEGED_HOST_MOUNTS.get(source)
            if allowed is not None:
                # An allowlisted privileged host mount (e.g. the Docker socket).
                # It escapes the workload-root and docker.sock refusals below,
                # but only on the allowlist's own terms: it must be an explicit
                # bind and must honour the read-only constraint the entry sets,
                # so an operator cannot turn a read-only grant into a writable
                # one. Consent that this app may have it at all is proven earlier,
                # where guided config assembles the mount.
                if volume.get("type") != "bind":
                    raise ValueError(
                        "A privileged host mount must be an explicit bind mount."
                    )
                if allowed["read_only"] and not volume.get("read_only"):
                    raise ValueError(
                        "{} may only be mounted read-only.".format(source)
                    )
                continue
            if "docker.sock" in source:
                raise ValueError("Docker socket access is blocked.")
            if volume.get("type") == "bind":
                resolved = Path(source).resolve()
                if workloads_root not in resolved.parents:
                    raise ValueError(
                        "Bind mounts must stay inside the managed workload root."
                    )
        for port in service.get("ports", []) or []:
            if not isinstance(port, dict):
                continue
            try:
                published = int(port.get("published") or 0)
            except (TypeError, ValueError):
                published = 0
            if published in {34001, 34002}:
                raise ValueError("A requested port is reserved for Vaelor.")
        limits = (
            ((service.get("deploy") or {}).get("resources") or {}).get("limits")
            or {}
        )
        memory_limit = limits.get("memory") or service.get("mem_limit")
        cpu_limit = limits.get("cpus") or service.get("cpus")
        if not memory_limit or not cpu_limit:
            raise ValueError("Every custom service needs CPU and memory limits.")
        memory_bytes = _memory_limit_bytes(memory_limit)
        if memory_bytes is None:
            raise ValueError(
                "The memory limit for service {} is not a valid size.".format(name)
            )
        if not MIN_MEMORY_LIMIT_BYTES <= memory_bytes <= MAX_MEMORY_LIMIT_BYTES:
            raise ValueError(
                "The memory limit for service {} must be between {} and {} bytes.".format(
                    name, MIN_MEMORY_LIMIT_BYTES, MAX_MEMORY_LIMIT_BYTES
                )
            )
        cpus = _cpu_limit_value(cpu_limit)
        if cpus is None or not 0 < cpus <= MAX_CPU_LIMIT:
            raise ValueError(
                "The CPU limit for service {} must be a number between 0 and {}.".format(
                    name, MAX_CPU_LIMIT
                )
            )
