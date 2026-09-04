"""Deterministic validation for server-generated application Compose."""

from __future__ import annotations

import re
import shutil
import socket
from pathlib import Path
from typing import Any, Dict, Mapping

from .compose_policy import validate_normalized


_PINNED_IMAGE = re.compile(r"^[^\s@]+@sha256:[0-9a-f]{64}$")


def validate_application_compose(
    compose: Mapping[str, Any],
    workloads_root: str,
    hardware: Mapping[str, Any],
) -> Dict[str, Any]:
    """Check policy, digest pins, host resources, ports, and storage."""
    if not isinstance(compose, dict):
        raise ValueError("The server-owned Compose draft is invalid.")
    root = Path(workloads_root).resolve()
    validate_normalized(dict(compose), root)
    services = compose.get("services", {})
    checks = ["Compose policy passed"]
    memory_required = 0
    ports = set()
    for service in services.values():
        image = str(service.get("image", ""))
        if not _PINNED_IMAGE.fullmatch(image):
            raise ValueError("Every application image must be digest-pinned.")
        limits = (((service.get("deploy") or {}).get("resources") or {}).get("limits") or {})
        memory_required += int(limits.get("memory") or 0)
        for item in service.get("ports", []) or []:
            port = int(item.get("published") or 0)
            protocol = str(item.get("protocol") or "tcp").lower()
            binding = (port, protocol)
            if binding in ports:
                raise ValueError("Published application ports must be unique.")
            ports.add(binding)
    available = int(hardware.get("memory_available_bytes") or 0)
    total = int(hardware.get("memory_total_bytes") or 0)
    budget = available or total
    if budget and memory_required > int(budget * 0.85):
        raise ValueError("This application exceeds the safe available-memory budget.")
    checks.append("Memory limit fits the current node budget")
    for port, protocol in ports:
        try:
            kind = socket.SOCK_DGRAM if protocol == "udp" else socket.SOCK_STREAM
            with socket.socket(socket.AF_INET, kind) as probe:
                probe.bind(("0.0.0.0", port))
        except OSError as error:
            raise ValueError(f"{protocol.upper()} port {port} is already in use.") from error
    checks.append("Published ports are currently available")
    existing = root
    while not existing.exists() and existing.parent != existing:
        existing = existing.parent
    try:
        free = shutil.disk_usage(existing).free
    except OSError:
        free = 0
    storage_required = int(
        ((compose.get("x-vaelor-resources") or {}).get("storage_bytes") or 0)
    )
    if free and free < max(1024 * 1024 * 1024, storage_required):
        raise ValueError("This application exceeds the safe available-storage budget.")
    checks.append("Managed workload storage is available")
    return {
        "policy": "validated",
        "checks": checks,
        "services": sorted(services),
        "memory_limit_bytes": memory_required,
        "published_ports": [
            {"port": port, "protocol": protocol}
            for port, protocol in sorted(ports)
        ],
        "storage_required_bytes": storage_required,
    }
