"""Host-port and docker-stats helpers shared by workload execution paths."""

from __future__ import annotations

import socket


#: Docker's `{{.MemUsage}}` units. Binary, and written with the `iB` suffix,
#: but the decimal spellings are accepted too rather than silently returning a
#: number that is 7.4% wrong if a future Docker changes its mind (VD-047).
_MEMORY_UNITS = {
    "b": 1,
    "kib": 1024, "mib": 1024 ** 2, "gib": 1024 ** 3, "tib": 1024 ** 4,
    "kb": 1000, "mb": 1000 ** 2, "gb": 1000 ** 3, "tb": 1000 ** 4,
}


def parse_memory_usage(text: str) -> int:
    """Bytes from a `docker stats` memory reading such as ``1.23GiB / 6.5GiB``.

    Only the first term is the usage; the second is the container's limit and
    is deliberately ignored - see ``_outgoing_model_bytes`` for why the limit is
    the wrong number here.

    Returns 0 for anything it cannot read. A deploy must not fail because a
    statistics format changed, and 0 leaves the sizing exactly as conservative
    as it was before this existed.
    """
    first = str(text or "").strip().split("/")[0].strip()
    digits = ""
    for character in first:
        if character.isdigit() or character == ".":
            digits += character
        else:
            break
    unit = first[len(digits):].strip().lower()
    if not digits or unit not in _MEMORY_UNITS:
        return 0
    try:
        return int(float(digits) * _MEMORY_UNITS[unit])
    except ValueError:
        return 0


def available_model_port(socket_factory=socket.socket) -> int:
    for port in range(8080, 8100):
        with socket_factory(socket.AF_INET, socket.SOCK_STREAM) as probe:
            try:
                probe.bind(("127.0.0.1", port))
            except OSError:
                continue
            return port
    raise ValueError(
        "No local AI port is available from 8080 to 8099. "
        "Stop an unused service and retry."
    )


def ensure_host_port_available(port: int, socket_factory=socket.socket) -> None:
    try:
        with socket_factory(socket.AF_INET, socket.SOCK_STREAM) as probe:
            probe.bind(("0.0.0.0", port))
    except OSError as error:
        raise ValueError(
            f"Port {port} is already in use. Choose another web address port."
        ) from error


def deployment_copilot_result(payload):
    if payload.get("profile") != "deployment-copilot":
        raise ValueError("Choose the supported deployment-copilot profile.")
    return {
        "profile": "deployment-copilot",
        "planner": "built-in",
        "local_model": "pending-model-selection",
        "approval_required": True,
    }
