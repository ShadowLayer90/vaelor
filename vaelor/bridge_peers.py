"""Who may talk to the root hardware bridge at all, decided per connection.

VD-143. The bridge's socket was ``0660 root:vaelor``, and group ``vaelor`` is
wider than the bridge's clients: it is the PRIMARY group of
``vaelor-research`` - the application-research service, which fetches and
reads untrusted web content, and the account deployed cluster agents run as -
and a supplementary group of ``vaelor-secrets`` and ``vaelor-vnc``. None of
those services calls the bridge, and every one of them could open it (LESSONS
pattern 18: ask who can reach this boundary before asking what it runs).

Two layers now decide, and either one alone refuses a stranger:

* **The kernel.** The socket is ``0660 root:vaelor-bridge``, a group whose
  only members are the two accounts whose services are the bridge's clients:
  ``vaelor`` (the control plane) and ``vaelor-workloads`` (the workload
  executor and broker). A process outside it cannot ``connect`` at all.
* **The bridge.** Every accepted connection's peer credentials
  (``SO_PEERCRED``, filled in by the kernel at ``connect`` and not forgeable by
  the client) are checked against the same two accounts plus root, BEFORE a
  byte of the request is read. This holds when the group layer does not: a box
  upgraded by a wheel alone, whose installer never created the group, gets a
  socket owned by the old group (so the appliance keeps working) and the peer
  check still turns the research account away.

A refused peer is answered with one JSON error line and the connection is
closed; its uid is logged, so a legitimate client this list forgot shows up in
the bridge's journal by number the first time it is turned away.
"""

from __future__ import annotations

import json
import socket
import struct
import sys
from typing import Any, Callable, FrozenSet, Optional, Tuple

#: The group the socket is owned by, and the accounts whose services call the
#: bridge. `deploy/install-vaelor.sh` creates the group and adds exactly these
#: accounts to it; `tests/test_bridge_peers.py` ties the two spellings.
BRIDGE_SOCKET_GROUP = "vaelor-bridge"
BRIDGE_CLIENT_ACCOUNTS = ("vaelor", "vaelor-workloads")

#: The group a box without :data:`BRIDGE_SOCKET_GROUP` falls back to - the one
#: the socket always had - so an appliance upgraded without its installer still
#: serves its own control plane while the peer check does the refusing.
FALLBACK_SOCKET_GROUP = "vaelor"

#: ``struct ucred``: pid, uid, gid, each a C int.
_UCRED = struct.Struct("3i")

#: What a refused peer is told. Deliberately says nothing about which accounts
#: would have been admitted.
REFUSAL = "This account may not use the hardware bridge."


def admitted_uids(lookup: Optional[Callable[[str], Any]] = None) -> FrozenSet[int]:
    """Root, and the uid of each client account that exists on this machine.

    An account that does not exist admits nobody rather than failing the
    bridge's start: a box without the executor still serves power and
    telemetry to its control plane.
    """
    if lookup is None:
        import pwd

        lookup = pwd.getpwnam
    uids = {0}
    for account in BRIDGE_CLIENT_ACCOUNTS:
        try:
            uids.add(int(lookup(account).pw_uid))
        except KeyError:
            continue
    return frozenset(uids)


def socket_group(lookup: Optional[Callable[[str], Any]] = None) -> Tuple[int, str]:
    """``(gid, name)`` the socket is owned by: the bridge group, else the fallback."""
    if lookup is None:
        import grp

        lookup = grp.getgrnam
    try:
        return int(lookup(BRIDGE_SOCKET_GROUP).gr_gid), BRIDGE_SOCKET_GROUP
    except KeyError:
        return int(lookup(FALLBACK_SOCKET_GROUP).gr_gid), FALLBACK_SOCKET_GROUP


def peer_uid(connection: Any) -> Optional[int]:
    """The connecting process's uid as the kernel recorded it, or ``None``.

    ``None`` when the platform has no ``SO_PEERCRED`` or the read fails - and
    ``None`` is never admitted, so a machine that cannot say who is calling
    serves nobody but its own tests.
    """
    option = getattr(socket, "SO_PEERCRED", None)
    if option is None:
        return None
    try:
        raw = connection.getsockopt(socket.SOL_SOCKET, option, _UCRED.size)
        _pid, uid, _gid = _UCRED.unpack(raw)
    except (OSError, struct.error, TypeError, ValueError):
        return None
    return int(uid)


def admit(connection: Any, admitted: FrozenSet[int], uid: Optional[int] = None) -> bool:
    """Whether to serve ``connection``; a refusal is answered and logged here.

    ``uid`` is the peer's, read from the connection when not given (a test
    gives it; the server never does).
    """
    peer = peer_uid(connection) if uid is None else uid
    if peer is not None and peer in admitted:
        return True
    line = json.dumps({"ok": False, "error": REFUSAL}, separators=(",", ":"))
    try:
        connection.sendall((line + "\n").encode("utf-8"))
    except OSError:
        pass
    print(
        "vaelor-hardware-bridge: refused a connection from uid {}".format(
            "unknown" if peer is None else peer
        ),
        file=sys.stderr, flush=True,
    )
    return False
