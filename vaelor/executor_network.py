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


#: The system DNS resolver (systemd-resolved on this appliance) binds port 53,
#: so an app that publishes 53 collides with it more often than not.
_SYSTEM_DNS_PORT = 53


#: Kernel socket tables to consult per transport, IPv4 and IPv6 both. A UDP
#: occupant (Syncthing on 22000, a DNS resolver on 53) does not appear in the
#: TCP tables, so consulting the wrong transport would miss the collision.
_PROTO_TABLES = {"tcp": ("tcp", "tcp6"), "udp": ("udp", "udp6")}


#: `/proc/net/tcp` state column for a listening socket. Only listeners occupy a
#: TCP host port; an outbound ESTABLISHED connection from :random does not.
_TCP_LISTEN_STATE = "0A"


def _read_proc_net_table(name: str) -> str:
    """Return the text of ``/proc/net/<name>``, or ``""`` if it cannot be read.

    Fail-open: an unreadable table yields no occupants, so a readable-only edge
    never blocks a deploy. The raw docker bind error at container start remains
    the backstop.
    """
    try:
        with open(f"/proc/net/{name}", "r", encoding="ascii", errors="replace") as handle:
            return handle.read()
    except OSError:
        return ""


def _ports_in_use(proto: str, table_reader) -> set:
    """Host ports occupied on ``proto`` per the kernel socket tables.

    Reading the tables works for privileged and unprivileged ports alike from a
    non-root process, unlike a bind probe: ``vaelor-workloads`` cannot bind a
    port below 1024, so a bind probe raises ``EACCES`` and reports every
    privileged port as taken even when it is free (VD-062). TCP counts only
    ``LISTEN`` sockets; UDP has no listen state, so any bound datagram socket on
    the port counts.
    """
    proto = str(proto).lower()
    require_listen = proto == "tcp"
    ports: set = set()
    for table in _PROTO_TABLES.get(proto, _PROTO_TABLES["tcp"]):
        for line in table_reader(table).splitlines()[1:]:
            fields = line.split()
            if len(fields) < 4:
                continue
            if require_listen and fields[3] != _TCP_LISTEN_STATE:
                continue
            try:
                ports.add(int(fields[1].rsplit(":", 1)[-1], 16))
            except ValueError:
                continue
    return ports


def _host_port_is_free(port: int, proto: str = "tcp", table_reader=None) -> bool:
    # Resolve the reader at call time (not as a default value) so a test can
    # patch ``_read_proc_net_table`` on the module and reach the production path.
    return int(port) not in _ports_in_use(proto, table_reader or _read_proc_net_table)


def ensure_extra_host_ports_available(
    app_name: str, extra_ports, socket_factory=None, table_reader=None
) -> None:
    """Fail with a clear message if a template's fixed extra port is taken.

    ``extra_ports`` is the catalog shape ``[(name, proto, port), ...]``. Each
    entry is checked on its own transport against the kernel socket tables, so a
    UDP-only occupant is detected and a privileged port is judged truthfully
    even though the executor runs unprivileged (see ``_ports_in_use``). A raw
    docker bind failure at container start is opaque; this turns it into an
    up-front, app-named error, and calls out the system DNS resolver for the
    port-53 case operators hit most.

    ``socket_factory`` is retained for the existing positional caller in
    ``executor.py`` and is unused now that occupancy is read rather than probed.
    """
    for _name, proto, extra_port in extra_ports:
        port = int(extra_port)
        if _host_port_is_free(port, proto, table_reader):
            continue
        detail = " It is often the system DNS resolver." if port == _SYSTEM_DNS_PORT else ""
        raise ValueError(
            f"{app_name} needs host port {port}, which is already in use on this "
            f"appliance.{detail} Free it and try again."
        )


def deployment_copilot_result(payload):
    if payload.get("profile") != "deployment-copilot":
        raise ValueError("Choose the supported deployment-copilot profile.")
    return {
        "profile": "deployment-copilot",
        "planner": "built-in",
        "local_model": "pending-model-selection",
        "approval_required": True,
    }
