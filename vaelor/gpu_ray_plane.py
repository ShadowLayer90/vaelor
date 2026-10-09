"""The Ray plane of a split: its own token, and a host firewall that admits only its machines.

A pipeline split runs Ray across its machines (`gpu_pool_runtime`: the head
inside the lead's server container, one ``ray start --address`` per worker).
Ray 2.58, the version in both pinned images (`vllm_images`), starts with no
authentication by default. Its GCS, raylets and object managers listen on
every interface whatever ``--node-ip-address`` says, and its client server
takes remote drivers. The containers run as root with ``--network host``, the
GPU devices, ``seccomp=unconfined`` and a writable model store. So anyone who
could reach those ports could join a node or submit work, and run code as root
on every machine of the split (ACC-163). The collective under vLLM (RCCL for
the tensors, Gloo for its control messages, which carry pickled objects) has
no authentication at all.

**Two layers, set on every machine BEFORE any Ray process starts**
(:class:`RayPlaneMixin.prepare_ray_plane`; VD-156, amended for ACC-163):

* **Ray's token authentication.** Every Ray container carries
  ``RAY_AUTH_MODE=token`` and the PATH of the token, and mounts the machine's
  token file read-only (:data:`RAY_AUTH_ENVIRONMENT`, :func:`token_mount`) -
  always, so no Ray unit can be rendered without them. The token is minted per
  deployment and again at every Load (:func:`mint_ray_token`). It is written
  root-only, and is never on an argv, in a unit file, in the container's
  environment or on the record.
* **A host firewall** (:func:`render_ruleset`): one Vaelor-owned nftables table
  per deployment, named for it. Every socket the split's containers hold -
  Ray's, and the ephemeral ones RCCL, Gloo and the TCPStore take, which no
  setting narrows - answers only the split's other machines and loopback,
  matched by OWNER: each Ray container runs in the deployment's own systemd
  slice (:func:`slice_name`, ``--cgroup-parent``) and the rule names that
  slice's cgroup (ACC-187). So a split may ride either link (owner,
  2026-10-01): the machines' Ethernet, whose card keeps serving everything
  else, or a dedicated cluster link, which is also closed whole to everyone
  but the split. Ray's pinned band (:data:`RAY_PORT_PINS`) is dropped for
  everyone else besides. That is defence in depth for the doors whose token
  check was never probed from outside, and the only guard the collective has.
  nftables resolves the slice's cgroup when the rules LOAD, and refuses a
  path that does not exist, so the slice is started first and a Ray unit is
  bound to it (``BindsTo=``, which carries a stop across, never a restart);
  each start re-checks the driver and the slice (`ray_slice_check`).

A worker's files are staged over SSH on stdin (``install`` then ``tee``, as a
gate's key is) and loaded with ``nft -f``; the controller's go to the root
bridge as typed values - a name, the token, and validated IPv4 addresses - and
the bridge renders the same ruleset itself (`bridge_managed_units`). A machine
where the firewall cannot be set refuses the split: it never starts unguarded.
Both layers are cleared with the deployment, including on every rollback
(:class:`RayPlane` marks what a deploy prepared), best effort, and an older
bridge that does not know the clear is noted once and does not fail a stop.

**What Ray 2.58 cannot do, measured in the image** (DECISIONS VD-156): the
client server cannot be turned off from ``ray start`` (so it is token-guarded
and firewalled), and the GCS and raylet bind every interface with no flag to
narrow them (so the firewall names their ports on every interface). Pinning
every Ray port together made a token-holding driver hang, so only the raylet's
two ports and the worker range are pinned, at the worker range's own defaults.
"""

from __future__ import annotations

import ipaddress
import logging
import re
import secrets
from dataclasses import dataclass
from typing import Any, Callable, List, Mapping, Optional, Sequence, Tuple

from .bridge_transport import root_renders_units
from . import cluster_link, gpu_node_facts
from .cluster_link_shared_card import rides_shared_card
from .gpu_pool_units import (  # noqa: F401 - the start limit, re-exported beside the fence
    RAY_START_LIMIT_BURST, RAY_START_LIMIT_SECONDS, RAY_WORKER_ROLE, SERVER_ROLE,
    container_name, deployment_name,
)
from .ssh_transport import SshTransportError

LOGGER = logging.getLogger(__name__)

#: Where a split's token and ruleset live on each machine: root's, ``0700``.
#: Spelled again by `ssh_transport`, which may not import this module; a test
#: ties the spellings together.
RAY_TOKEN_ROOT = "/etc/vaelor-ray"
#: Its mode, asserted by the worker profile too (VD-194, LESSONS 6).
RAY_TOKEN_ROOT_MODE = "0700"

#: Where the token appears inside a Ray container, read-only.
CONTAINER_TOKEN_PATH = "/run/vaelor/ray-auth-token"

#: The two variables every Ray container carries: the mode, and the PATH of
#: the token. Neither is secret.
RAY_AUTH_ENVIRONMENT = ("RAY_AUTH_MODE=token", "RAY_AUTH_TOKEN_PATH=" + CONTAINER_TOKEN_PATH)

#: The head's own switches: no dashboard (and so no job API), no autoscaler
#: monitor. Two words each, so the head's shell line stays plain.
HEAD_HARDENING = ("--include-dashboard", "false", "--no-monitor")

#: Ray's own listeners, which Ray binds on EVERY interface (`::`) whatever
#: ``--node-ip-address`` says, pinned into one narrow band well clear of every
#: port Vaelor listens on (the console's 34001-34002, the LLM Server's 11434,
#: serving and agents in 8000-8399, Phoenix's 6006, InfluxDB's 8086, ...;
#: `tests/test_ray_plane.py` checks the list). The GCS and the client server
#: are Ray's fixed ports; the raylet's two are pinned; the core workers get a
#: range of 200, and vLLM's own distributed ports (``VLLM_PORT``, the TCPStore
#: and the like) the hundred after it.
GCS_PORT = 6379
NODE_MANAGER_PORT = 6380
OBJECT_MANAGER_PORT = 6381
CLIENT_SERVER_PORT = 10001
WORKER_PORTS = (26000, 26199)
VLLM_BASE_PORT = 26200
VLLM_PORTS = (VLLM_BASE_PORT, 26299)
#: Every port the fence drops for anyone but the split's machines, on every
#: interface other than the link (the link itself answers only them at all).
DROPPED_PORT_RANGES = (
    (GCS_PORT, OBJECT_MANAGER_PORT), (CLIENT_SERVER_PORT, CLIENT_SERVER_PORT),
    (WORKER_PORTS[0], VLLM_PORTS[1]),
)

#: What both ``ray start`` commands add, so Ray's listeners stay in the band.
RAY_PORT_PINS = (
    "--node-manager-port", str(NODE_MANAGER_PORT),
    "--object-manager-port", str(OBJECT_MANAGER_PORT),
    "--min-worker-port", str(WORKER_PORTS[0]), "--max-worker-port", str(WORKER_PORTS[1]),
)

#: Everything a Ray container of a split carries for its plane: the auth mode,
#: the token's PATH, and the base of vLLM's own distributed ports. None secret.
RAY_CONTAINER_ENVIRONMENT = (*RAY_AUTH_ENVIRONMENT, "VLLM_PORT={}".format(VLLM_BASE_PORT))

#: How each machine is asked which cgroup driver Docker places containers with.
#: Only ``systemd`` puts a container under its ``--cgroup-parent`` slice's own
#: cgroup; ``cgroupfs`` makes a top-level ``/<slice>/<id>`` that the fence's
#: level-3 rule never matches (review 1, SC1).
CGROUP_DRIVER_ARGV = ("docker", "info", "--format", "{{.CgroupDriver}}")
CGROUP_DRIVER_REFUSED = (
    "Docker on this machine places containers with the '{driver}' cgroup "
    "driver, so the firewall could not tell the split's containers apart from "
    "the rest. Set Docker to the systemd cgroup driver and deploy again."
)
#: How long a deploy waits for a Ray container to show where it runs.
SLICE_CHECK_SECONDS = 120
NOT_IN_SLICE = (
    "The split's {role} container on this machine is not running inside the "
    "split's own slice ({where}), so its firewall would not cover it; it was "
    "stopped."
)
NOT_CONFIRMED_IN_SLICE = (
    "Vaelor could not confirm that the split's {role} container on this "
    "machine runs inside the split's own slice, so it was stopped."
)
#: The two Ray container roles a split runs, and the refusal for any other.
RAY_ROLES = (SERVER_ROLE, RAY_WORKER_ROLE)
NOT_A_RAY_CONTAINER = "That is not a Ray container of a split."
#: What systemd says when the slice is already gone - a stop that is done.
SLICE_ALREADY_GONE = "not loaded"
#: What nft says when the table is already gone - a removal that is done.
TABLE_ALREADY_GONE = "No such file"
#: Where a machine's routing table is read, to find its default-route card.
ROUTE_TABLE = "/proc/net/route"
DEFAULT_ROUTE_REFUSED = (
    "{link} carries this machine's default route, so closing it to everyone "
    "but the split would cut this machine off its network. It can only carry a "
    "split fenced as a shared network card."
)
#: The fail-closed refusals of a DEDICATED fence (review 2, SC1): the IPv4 main
#: routing table (``/proc/net/route``) must prove the link is not the way this
#: machine is reached, or no whole-interface fence goes on it.
ROUTE_UNREAD = (
    "Vaelor could not read this machine's routing table, so it cannot show "
    "that {link} is not the way this machine is reached; a split there is "
    "fenced only as a shared network card."
)
NO_DEFAULT_ROUTE = (
    "This machine has no IPv4 default route in its main routing table, so "
    "Vaelor cannot show that {link} is not the way it is reached; a split "
    "there is fenced only as a shared network card."
)
LINKS_UNREAD = (
    "Vaelor could not read this machine's addresses, so it cannot show that "
    "{link} is not the way this machine is reached; a split there is fenced "
    "only as a shared network card."
)
LINK_NOT_HELD = (
    "Vaelor found no IPv4 address that this machine reaches through {link} "
    "alone (it may be a second card on the same network as another), so it "
    "cannot show {link} is not the way this machine is reached; a split there "
    "is fenced only as a shared network card."
)
GATEWAY_ROUTE_ON_LINK = (
    "{link} carries a route through a gateway (a VPN or a second way out), so "
    "closing it to everyone but the split could cut this machine off; it can "
    "only carry a split fenced as a shared network card."
)
SAME_NETWORK_AS_DEFAULT = (
    "{link} is on the same network as this machine's default route "
    "({network}), so closing it to everyone but the split could cut this "
    "machine off its network; it can only carry a split fenced as a shared "
    "network card."
)

#: The slice check every Ray unit runs at each start (review 2, SC2;
#: `ray_slice_check`), staged root-only beside the split's tokens.
SLICE_CHECK_PATH = "/etc/vaelor-ray/vaelor-slice-check.py"
#: Isolated mode (``-I``): no user site, no environment variables, no script
#: folder on the path - the check runs as root and must not import a stray file.
SLICE_CHECK_PYTHON = "/usr/bin/python3 -I"
#: Said when a Ray container dies as it starts, instead of waiting it out.
DIED_AT_START = (
    "The split's {role} container stopped as it started ({state}){detail}."
)

#: Where a split's slice unit is written on a worker (`gpu_pool_runtime`'s own).
SLICE_UNIT_DIRECTORY = "/etc/systemd/system"

#: The system ``nft`` every Ray unit loads its fence with before it starts.
NFT_BINARY = "/usr/sbin/nft"

#: 32 random bytes, as lowercase hex: the only shape a token may take.
_TOKEN_RULE = r"[0-9a-f]{64}"

#: Every split's Ray containers run in a systemd slice of its own under this
#: parent (ACC-187), and the fence matches sockets by that slice: nftables'
#: ``socket cgroupv2 level 3`` names the slice's own cgroup, three levels down.
SLICE_PARENT = "vaelor.slice/vaelor-ray.slice"
SLICE_LEVEL = 3

#: What is refused when the token handed to a prepare is not well formed.
TOKEN_REQUIRED = (
    "A split's Ray processes start only with the deployment's own Ray token, "
    "and none that could be one was given."
)
#: The one refusal a split meets when a machine's firewall cannot be set.
RAY_PLANE_REFUSED = (
    "The split was not started: this machine's firewall could not be set to "
    "admit Ray only from the other machines of the split. {reason}"
)
#: Said instead when the machine is the controller and its bridge predates this.
BRIDGE_TOO_OLD_FOR_RAY = (
    "This controller's hardware bridge is older than Vaelor and cannot guard "
    "a split's Ray ports. Restart the hardware bridge and deploy again."
)
_OLD_BRIDGE = ("not a controller unit verb", "Unsupported hardware action")

#: At most this many other machines in one split's firewall.
_MAX_PEERS = 31


def mint_ray_token() -> str:
    """A new token for one deployment's Ray plane: 32 random bytes, hex."""
    return secrets.token_hex(32)


def require_ray_token(value: Any) -> str:
    """``value``, or :data:`TOKEN_REQUIRED`. Never echoes what it was given."""
    if not isinstance(value, str) or not re.fullmatch(_TOKEN_RULE, value):
        raise ValueError(TOKEN_REQUIRED)
    return value


def token_file(deployment: str) -> str:
    """Where ``deployment``'s token is written on a machine."""
    return "{}/{}.token".format(RAY_TOKEN_ROOT, deployment_name(deployment))


def ruleset_file(deployment: str) -> str:
    """Where ``deployment``'s firewall ruleset is staged on a machine."""
    return "{}/{}.nft".format(RAY_TOKEN_ROOT, deployment_name(deployment))


def table_name(deployment: str) -> str:
    """The nftables table a deployment owns: derived from its validated name."""
    return "vaelor_ray_" + deployment_name(deployment).replace("-", "_")


def slice_name(deployment: str) -> str:
    """The systemd slice a deployment's Ray containers run in.

    systemd reads every dash of a slice name as one more level of nesting, so
    the deployment name's dashes become underscores: the slice always sits at
    :data:`SLICE_LEVEL`, which is the level the fence names.
    """
    return "vaelor-ray-{}.slice".format(deployment_name(deployment).replace("-", "_"))


def slice_path(deployment: str) -> str:
    """The slice's cgroup path below the cgroup root, as nftables names it."""
    return "{}/{}".format(SLICE_PARENT, slice_name(deployment))


def slice_unit_text(deployment: str) -> str:
    """The static slice unit: fixed text, no settings, only a name to hold the cgroup."""
    return "\n".join([
        "[Unit]",
        "Description=Vaelor split {}: the containers its firewall admits only "
        "the split's machines to".format(deployment_name(deployment)),
        "",
        "[Slice]",
        "",
    ])


def cgroup_parent(deployment: str) -> List[str]:
    """The ``docker run`` option that puts a Ray container in its split's slice."""
    return ["--cgroup-parent", slice_name(deployment)]


def cgroup_driver_refusal(text: Any) -> Optional[str]:
    """Why this Docker cannot run a fenced split, or ``None`` on the systemd driver."""
    driver = str(text or "").strip()
    if driver == "systemd":
        return None
    return CGROUP_DRIVER_REFUSED.format(driver=driver[:40] or "unknown")


def default_route_interfaces(route_text: Any) -> List[str]:
    """The interfaces carrying a default route, from ``/proc/net/route`` text."""
    found = []
    for line in str(route_text or "").splitlines()[1:]:
        fields = line.split()
        if len(fields) >= 8 and fields[1] == "00000000" and fields[7] == "00000000":
            found.append(fields[0])
    return found


def _flags(hex_text: str) -> int:
    """A ``/proc/net/route`` flags field; unreadable reads as a gateway route."""
    try:
        return int(hex_text, 16)
    except ValueError:
        return 0x2


def _gateway(hex_text: str) -> Optional[ipaddress.IPv4Address]:
    """A ``/proc/net/route`` gateway field (little-endian hex) as an address."""
    try:
        return ipaddress.IPv4Address(bytes(reversed(bytes.fromhex(hex_text))))
    except ValueError:
        return None


def dedicated_on_default_route(
    link: Any, shared: bool, route_text: Any, links: Any = None,
) -> Optional[str]:
    """Why a DEDICATED fence may not go on ``link``, else ``None``. Fails closed.

    Defence in depth (review 1 SC2, review 2 SC1): whatever decided
    ``shared``, a whole-interface fence goes only where the IPv4 main routing
    table PROVES it is not this machine's way out. A table that cannot be read
    or parsed, a table with no IPv4 default route, the default-route card
    itself, and a card holding an address on the default route's network or
    its gateway's (``links``: this machine's ``{name, network}`` pairs) are
    all refused. Scope, stated: IPv4 and the main table only - a policy
    routing rule or an IPv6-only way in is not seen here.
    """
    if shared:
        return None
    lines = str(route_text or "").splitlines()
    if not lines or not lines[0].split()[:1] == ["Iface"]:
        return ROUTE_UNREAD.format(link=link)
    defaults = [
        (fields[0], _gateway(fields[2])) for fields in (line.split() for line in lines[1:])
        if len(fields) >= 8 and fields[1] == "00000000" and fields[7] == "00000000"
    ]
    if not defaults:
        return NO_DEFAULT_ROUTE.format(link=link)
    # The link must hold an address of its own, read from this machine's
    # tables (review 3, SC3). A second card on the same LAN holds none - the
    # kernel's connected route gives its address to the first card - and an
    # unread table proves nothing: both are refused, never assumed safe.
    if links is None:
        return LINKS_UNREAD.format(link=link)
    if not any(isinstance(item, Mapping) and item.get("name") == str(link) for item in links):
        return LINK_NOT_HELD.format(link=link)
    if str(link) in {name for name, _gateway_address in defaults}:
        return DEFAULT_ROUTE_REFUSED.format(link=link)
    # Any route through a gateway on the link (flag 0x2) - an OpenVPN "def1"
    # pair 0.0.0.0/1 and 128.0.0.0/1 on tun0, say - is a way out (review 4).
    for fields in (line.split() for line in lines[1:]):
        if len(fields) >= 4 and fields[0] == str(link) and _flags(fields[3]) & 0x2:
            return GATEWAY_ROUTE_ON_LINK.format(link=link)
    held = [item for item in (links or []) if isinstance(item, Mapping)]
    default_networks = [
        str(item.get("network")) for item in held
        if item.get("name") in {name for name, _g in defaults}
    ]
    for item in held:
        if item.get("name") != str(link):
            continue
        try:
            network = ipaddress.ip_network(str(item.get("network")), strict=False)
        except ValueError:
            continue
        for other in default_networks:
            try:
                if network.overlaps(ipaddress.ip_network(other, strict=False)):
                    return SAME_NETWORK_AS_DEFAULT.format(link=link, network=other)
            except ValueError:
                continue
        for _name, gateway in defaults:
            if gateway is not None and not gateway.is_unspecified and gateway in network:
                return SAME_NETWORK_AS_DEFAULT.format(link=link, network=str(network))
    return None


def cgroup_of(proc_cgroup_text: Any) -> str:
    """A process's cgroup v2 path from its ``/proc/<pid>/cgroup`` text, or ``""``."""
    for line in str(proc_cgroup_text or "").splitlines():
        if line.startswith("0::"):
            return line[3:].strip()
    return ""


def in_slice(path: Any, deployment: str) -> bool:
    """Whether a cgroup path lies inside the deployment's own slice."""
    return str(path or "").startswith("/{}/".format(slice_path(deployment)))


def slice_check_pre() -> str:
    """The ``ExecStartPre`` every Ray unit runs: Docker must use the systemd driver."""
    return "{} {} driver".format(SLICE_CHECK_PYTHON, SLICE_CHECK_PATH)


def slice_check_post(deployment: str, role: str) -> str:
    """The ``ExecStartPost`` every Ray unit runs: its container must be in its slice."""
    return "{} {} container {} {}".format(
        SLICE_CHECK_PYTHON, SLICE_CHECK_PATH, container_name(deployment, role),
        slice_path(deployment),
    )


def slice_check_program() -> str:
    """The check program's text, as this package ships it (`ray_slice_check`)."""
    from pathlib import Path

    return (Path(__file__).parent / "ray_slice_check.py").read_text(encoding="utf-8")


def describe_dead_unit(properties: Mapping[str, str], journal: Any = "") -> Optional[str]:
    """What a Ray unit that died at start says about it, ``None`` while it lives.

    The unit's own systemd state and exit evidence (`gpu_pool_startup`) and the
    last line its run wrote, so a crash is reported as a crash (review 2).
    """
    from .gpu_pool_startup import describe_unit_state, unit_alive

    if unit_alive(properties):
        return None
    last = [line.strip() for line in str(journal or "").splitlines() if line.strip()]
    return "{}{}".format(describe_unit_state(properties), (": " + last[-1][:200]) if last else "")


def confirm_unit_in_slice(
    runtime: Any, transport: Any, name: str, role: str, unit: str, *,
    sleep: Callable[[float], Any], monotonic: Callable[[], float],
) -> None:
    """`confirm_in_slice` for one started Ray unit, reporting a crash as a crash."""
    def unit_failed() -> Optional[str]:
        try:
            state = runtime.unit_state(transport, unit)
        except (SshTransportError, RuntimeError, OSError, ValueError):
            return None
        if not isinstance(state, Mapping):
            return None
        try:
            journal = runtime.unit_journal(transport, unit)
        except (SshTransportError, RuntimeError, OSError, ValueError):
            journal = ""
        return describe_dead_unit(state, journal if isinstance(journal, str) else "")

    confirm_in_slice(
        lambda: runtime.container_cgroup(transport, name, role), name, role=role,
        sleep=sleep, monotonic=monotonic, unit_failed=unit_failed,
    )


def confirm_in_slice(
    read: Callable[[], str], deployment: str, *, role: str = "Ray",
    sleep: Callable[[float], Any], monotonic: Callable[[], float],
    deadline: float = SLICE_CHECK_SECONDS,
    unit_failed: Optional[Callable[[], Optional[str]]] = None,
) -> None:
    """Wait for a Ray container to show its cgroup; refuse unless it is the slice's.

    ``read`` answers the container's cgroup path, ``""`` while it is not
    running yet. A path outside the slice, or none within ``deadline``, is
    the split's refusal: the caller's rollback stops the unit (fail closed).
    ``unit_failed`` says why the container's unit died, if it did: that is
    reported at once, as what it is.
    """
    end = monotonic() + deadline
    while True:
        try:
            path = read()
        except (SshTransportError, RuntimeError, OSError):
            path = ""
        if path:
            if in_slice(path, deployment):
                return
            raise RuntimeError(RAY_PLANE_REFUSED.format(
                reason=NOT_IN_SLICE.format(role=role, where=str(path)[:120])))
        dead = unit_failed() if unit_failed is not None else None
        if dead:
            raise RuntimeError(DIED_AT_START.format(role=role, state=dead, detail=""))
        if monotonic() >= end:
            raise RuntimeError(RAY_PLANE_REFUSED.format(
                reason=NOT_CONFIRMED_IN_SLICE.format(role=role)))
        sleep(2)


def boot_start_pre(deployment: str) -> str:
    """The ``ExecStartPre`` every Ray unit runs: load its fence, or do not start."""
    return "{} -f {}".format(NFT_BINARY, ruleset_file(deployment))


def token_mount(deployment: str) -> List[str]:
    """The read-only ``-v`` that shows the machine's token file to a Ray container."""
    return ["-v", "{}:{}:ro".format(token_file(deployment), CONTAINER_TOKEN_PATH)]


_NOT_IPV4 = "The {} is not an IPv4 address."


def _cluster_address(value: Any, what: str) -> str:
    """A private, unicast IPv4 address in canonical form, or a refusal."""
    if not isinstance(value, str) or value != value.strip():
        raise ValueError(_NOT_IPV4.format(what))
    try:
        address = ipaddress.IPv4Address(value)
    except ValueError:
        raise ValueError(_NOT_IPV4.format(what)) from None
    if (not address.is_private or address.is_loopback or address.is_link_local
            or address.is_multicast or address.is_unspecified or str(address) != value):
        raise ValueError("The {} must be a private cluster address.".format(what))
    return value


def require_peers(address: Any, peers: Any) -> Tuple[str, List[str]]:
    """``(address, peers)`` checked: this machine's link address and the others'."""
    own = _cluster_address(address, "machine's cluster address")
    if isinstance(peers, (str, bytes)) or not isinstance(peers, Sequence):
        raise ValueError("The split's other machines must be a list of addresses.")
    checked = [_cluster_address(peer, "address of another machine") for peer in peers]
    if not checked or len(checked) > _MAX_PEERS or len(set(checked)) != len(checked) or own in checked:
        raise ValueError("The split's other machines must be 1 to {} distinct addresses, "
                         "not this machine's own.".format(_MAX_PEERS))
    return own, sorted(checked, key=ipaddress.IPv4Address)


def require_link(value: Any) -> str:
    """The cluster-link interface name, checked by the one interface-name rule."""
    name = gpu_node_facts.interface_name(value) if isinstance(value, str) else None
    if not name or name == "lo":
        raise ValueError("The cluster link must be one of this machine's own network links.")
    return name


def render_ruleset(
    deployment: str, link: Any, address: Any, peers: Any, *, shared: bool = False,
) -> str:
    """The one ruleset a deployment's fence is: fixed text, validated values.

    * **Every socket the split's containers hold** answers only the split's
      machines and loopback, on every interface (ACC-187): a NEW connection
      to a socket owned by the deployment's slice is dropped from any other
      IPv4 address and from all IPv6. RCCL, Gloo and the TCPStore listen on
      ports no setting narrows; this fences them whatever port they took. A
      slice whose cgroup does not exist makes nftables refuse the whole load.
    * On a DEDICATED link (``shared`` false) the link also answers only the
      split's machines, whatever the protocol, and nothing addressed to this
      machine's link address is taken from another interface. On the shared
      network card (``shared``) neither applies: the card carries everything
      else this machine does.
    * Ray's band, which Ray binds on every interface, is dropped for everyone
      but the split's machines, IPv4 and IPv6, TCP and UDP.

    Loaded with ``nft -f`` in one transaction: the table is created if absent
    and deleted, then written whole, so a second load replaces the first.
    """
    table = "table inet " + table_name(deployment)
    interface = require_link(link)
    own, others = require_peers(address, peers)
    on_link = '\t\tiifname "{}" '.format(interface)
    owned = '\t\tct state new socket cgroupv2 level {} "{}" '.format(
        SLICE_LEVEL, slice_path(deployment),
    )
    band = ", ".join(
        str(low) if low == high else "{}-{}".format(low, high)
        for low, high in DROPPED_PORT_RANGES
    )
    # The same two drops guard the slice's sockets and, on a dedicated
    # link, the whole interface.
    no_v6, strangers = "meta nfproto ipv6 drop", "ip saddr != @peers drop"
    dedicated = [] if shared else [
        on_link + no_v6,
        on_link + strangers,
        on_link + "accept",
        "\t\tip daddr {} drop".format(own),
    ]
    return "\n".join([
        table,
        "delete " + table,
        table + " {",
        "\tset peers {",
        "\t\ttype ipv4_addr",
        "\t\telements = { " + ", ".join(others) + " }",
        "\t}",
        "\tchain input {",
        "\t\ttype filter hook input priority -10; policy accept;",
        '\t\tiif "lo" accept',
        owned + no_v6,
        owned + strangers,
        *dedicated,
        "\t\tct state established,related accept",
        # On a dedicated link the split's machines are admitted there only.
        ("\t\tip saddr @peers accept" if shared else on_link + "ip saddr @peers accept"),
        "\t\tmeta l4proto { tcp, udp } th dport { " + band + " } drop",
        "\t}",
        "}",
        "",
    ])


def prepare_split(
    runtime: Any, name: str, participants: List[Any], transports: Any,
    advertise_address: Any, started: List[Any], bound: Any, interfaces: Any,
) -> str:
    """Guard every machine of a split before any Ray process starts; the token.

    One token for this run of the split - minted again at every Load, which
    comes through here too - and on each machine a firewall admitting the
    others' cluster-link addresses. Each machine is marked in ``started``
    (:class:`RayPlane`) BEFORE its prepare, so a rollback clears even one
    prepared half-way; a machine that cannot be guarded raises, refusing the
    split before anything of Ray runs.
    """
    token = mint_ray_token()
    for node in participants:
        # Either link carries a split (owner, 2026-10-01). A machine whose
        # split address is the one Vaelor reaches it on, or whose split
        # interface is the one it enrolled on, rides its SHARED network card:
        # its fence then guards the split's own sockets only (ACC-187).
        # One rule, owned by cluster_link_shared_card, so the cluster-link
        # confirmation says "shared" of exactly the card fenced as shared (B8).
        enrolled = ((node.get("inventory") or {}).get("cluster_interface") or {}).get("name")
        shared = rides_shared_card(
            link_name=interfaces.get(node["id"]), link_address=bound[node["id"]],
            advertise_address=advertise_address(node["host"]), enrolled_interface=enrolled,
        )
        started.append((transports[node["id"]], node, RayPlane(name)))
        runtime.prepare_ray_plane(
            transports[node["id"]], name=name, token=token,
            link=interfaces[node["id"]], address=bound[node["id"]],
            peers=[bound[other["id"]] for other in participants if other is not node],
            shared=shared,
        )
    return token


@dataclass(frozen=True)
class RayPlane:
    """What a deploy prepared on one machine, for its rollback and its stop."""

    deployment: str

    def __str__(self) -> str:
        return "the Ray token and firewall of {}".format(self.deployment)


_noted_old_bridge = []


def _reset_old_bridge_note() -> None:
    """For tests: forget that an old bridge was already noted."""
    _noted_old_bridge.clear()


def _is_old_bridge(error: Exception) -> bool:
    return any(marker in str(error) for marker in _OLD_BRIDGE)


class RayPlaneMixin:
    """`prepare_ray_plane` and `clear_ray_plane`, inherited by `GpuPoolRuntime`."""

    def prepare_ray_plane(
        self, transport, *, name: str, token: str, link: str, address: str,
        peers: List[str], shared: bool = False,
    ) -> None:
        """Put the slice, the token and the fence on one machine, or refuse the split.

        The slice is written and STARTED before the fence loads: the fence
        names the slice's cgroup, and nftables refuses a path that does not
        exist (ACC-187).
        """
        deployment = deployment_name(name)
        token = require_ray_token(token)
        ruleset = render_ruleset(deployment, link, address, peers, shared=bool(shared))
        try:
            if root_renders_units(transport):
                transport.prepare_ray_plane(
                    deployment, token, link, address, list(peers), bool(shared),
                )
                return
            refusal = cgroup_driver_refusal(transport.run(list(CGROUP_DRIVER_ARGV), sudo=True))
            if refusal is None and not shared:
                route = transport.run(["cat", ROUTE_TABLE], sudo=True)
                links = cluster_link.node_links_if_read(
                    route, transport.run(["cat", cluster_link.FIB_TRIE_PATH], sudo=True))
                refusal = dedicated_on_default_route(link, False, route, links)
            if refusal is not None:
                raise RuntimeError(refusal)
            self._stage_root_file(transport, SLICE_CHECK_PATH, slice_check_program())
            transport.run(["tee", "{}/{}".format(SLICE_UNIT_DIRECTORY, slice_name(deployment))],
                          sudo=True, stdin_text=slice_unit_text(deployment))
            transport.run(["systemctl", "daemon-reload"], sudo=True)
            transport.run(["systemctl", "start", slice_name(deployment)], sudo=True)
            self._stage_root_file(transport, token_file(deployment), token)
            self._stage_root_file(transport, ruleset_file(deployment), ruleset)
            transport.run(["nft", "-f", ruleset_file(deployment)], sudo=True)
        except (SshTransportError, RuntimeError, OSError) as error:
            if _is_old_bridge(error):
                raise RuntimeError(BRIDGE_TOO_OLD_FOR_RAY) from error
            raise RuntimeError(RAY_PLANE_REFUSED.format(reason=str(error)[:300])) from error

    @staticmethod
    def _stage_root_file(transport, path: str, text: str) -> None:
        """A root ``0600`` file in the ``0700`` folder, its text on stdin only."""
        transport.run(["install", "-d", "-m", RAY_TOKEN_ROOT_MODE, RAY_TOKEN_ROOT], sudo=True)
        transport.run(["install", "-m", "0600", "/dev/null", path], sudo=True)
        transport.run(["tee", path], sudo=True, stdin_text=text)

    def container_cgroup(self, transport, name: str, role: str) -> str:
        """Where one of a split's Ray containers runs: its cgroup path, ``""`` if not yet.

        On a worker, Docker names the container's process and the kernel its
        cgroup; on the controller the root bridge reads both (a typed verb).
        """
        deployment = deployment_name(name)
        if role not in RAY_ROLES:
            raise ValueError(NOT_A_RAY_CONTAINER)
        if root_renders_units(transport):
            return str(transport.ray_container_cgroup(deployment, role) or "")
        pid = str(transport.run(
            ["docker", "inspect", "--format", "{{.State.Pid}}", container_name(deployment, role)],
            sudo=True,
        ) or "").strip()
        if not pid.isdigit() or pid == "0":
            return ""
        return cgroup_of(transport.run(["cat", "/proc/{}/cgroup".format(pid)], sudo=True))

    def clear_ray_plane(self, transport, name: str) -> Optional[str]:
        """Stop the split's slice, then remove its firewall, ruleset and token; never raises.

        Returns ``None`` when the machine is clear, else the failure in plain
        words, which the stop puts on the record so `remove` keeps the row
        naming what was left (review 2, SC5). An older controller bridge that
        has no clear verb has no plane to clear: it is logged once and the stop
        goes on.

        **The containers go before their firewall** (review 1, SC3): stopping
        the slice stops every process in it. A slice that will not stop keeps
        the firewall in place - a running container is never left unfenced -
        and the failure is reported; one already gone is no obstacle.
        """
        deployment = deployment_name(name)
        if root_renders_units(transport):
            try:
                transport.clear_ray_plane(deployment)
            except (SshTransportError, RuntimeError, OSError) as error:
                if _is_old_bridge(error):
                    if not _noted_old_bridge:
                        _noted_old_bridge.append(True)
                        LOGGER.warning(
                            "The hardware bridge predates the Ray firewall, so %s could not "
                            "be cleared on this controller; restart the hardware bridge.",
                            RayPlane(deployment),
                        )
                    return None
                LOGGER.warning("Could not clear %s: %s", RayPlane(deployment), error)
                return str(error)[:300]
            return None
        try:
            transport.run(["systemctl", "stop", slice_name(deployment)], sudo=True)
        except (SshTransportError, RuntimeError, OSError) as error:
            if SLICE_ALREADY_GONE not in str(error):
                LOGGER.warning(
                    "Kept %s: the split's slice would not stop, so its containers may "
                    "still run: %s", RayPlane(deployment), error,
                )
                return "the split's containers would not stop, so its firewall was kept: {}".format(
                    str(error)[:200])
        failure = None
        for argv in (
            ["nft", "delete", "table", "inet", table_name(deployment)],
            ["rm", "-f", "{}/{}".format(SLICE_UNIT_DIRECTORY, slice_name(deployment))],
            ["systemctl", "daemon-reload"],
            ["rm", "-f", ruleset_file(deployment)],
            ["rm", "-f", token_file(deployment)],
        ):
            try:
                transport.run(argv, sudo=True)
            except (SshTransportError, RuntimeError, OSError) as error:
                LOGGER.info("Clearing %s: %s: %s", RayPlane(deployment), argv[0], error)
                # A table already gone is clear; any other nft failure is not.
                if argv[0] == "nft" and TABLE_ALREADY_GONE not in str(error):
                    failure = "its firewall could not be removed: {}".format(str(error)[:200])
        return failure
