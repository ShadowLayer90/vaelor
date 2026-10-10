"""What the console may show about the household authority (VD-212).

Read by the control plane, never root: it opens only files its account can
read - the public root copy, the served certificate, the worker trust bundle,
the pending certificate (0750 root:vaelor folder) and the fleet trust record
it writes itself - and never the authority folder (0700 root) or a key.

The migration state is derived with the authority's own verdict
(:func:`vaelor.tls_fleet_gate.promotion_verdict`) over the same files, so the
panel and the authority cannot disagree about why the switch waits (LESSONS 6).
Absent and unreadable are always told apart (LESSONS 1).
"""

from __future__ import annotations

import datetime
import hashlib
import os
import socket
from typing import Any, Dict, List, Optional

from . import tls_authority_pki as pki
from . import tls_fleet_gate, tls_identity
from .tls_authority import LEGACY_RETENTION_SECONDS, LEGACY_SUFFIX, Layout, _load_cert, _read

MIGRATION_STATES = ("none", "waiting", "blocked", "done")


def _migration(layout: Layout, kind: str, root, now: datetime.datetime) -> Dict[str, Any]:
    if kind == pki.KIND_HOUSEHOLD:
        try:
            kept = os.stat(layout.leaf_cert + LEGACY_SUFFIX).st_mtime
        except OSError:
            return {"state": "none", "reason": None}
        until = datetime.datetime.fromtimestamp(kept + LEGACY_RETENTION_SECONDS,
                                                datetime.timezone.utc)
        return {"state": "done", "reason": "serving the household certificate; the "
                "replaced one is kept until {}".format(until.date().isoformat())}
    if kind not in pki.MIGRATING_KINDS:
        return {"state": "none", "reason": None}
    pending, unreadable = _load_cert(layout.pending_cert)
    if pending is None or root is None:
        return {"state": "waiting", "reason": unreadable or "the household certificate "
                "is issued on the authority's next pass"}
    try:
        bundle = _read(layout.worker_bundle) or b""
    except OSError:
        bundle = b""
    names, addresses = pki.leaf_identities(pending)
    verdict = tls_fleet_gate.promotion_verdict(
        tls_fleet_gate.read_record(layout.fleet_state),
        hashlib.sha256(bundle).hexdigest(), pki.issued_at(pending), now.timestamp(),
        names, addresses)
    if verdict.blocked:
        return {"state": "blocked", "reason": verdict.reason}
    if verdict.promote:
        return {"state": "waiting", "reason": "ready ({}); served on the authority's "
                "next pass, within five minutes".format(verdict.reason)}
    return {"state": "waiting", "reason": verdict.reason}


def names_left_out(root, hostname: str) -> List[Dict[str, str]]:
    """This machine's bare hostname when the root does not carry it, and why.

    Read from the public root and the hostname with the shipped TLD list; no
    key. A hostname changed since the root was made is reported too.
    """
    label = tls_identity.bare_hostname(hostname)
    if root is None or not label:
        return []
    dns, _ = pki.permitted_by_root(root)
    if label in dns:
        return []
    _, refusal = pki.root_hostname_label(label)
    return [{"name": label, "reason": refusal or "the root was made when this "
             "machine had another name"}]


def authority_status(layout: Layout, now: Optional[datetime.datetime] = None,
                     hostname: Optional[str] = None) -> Dict[str, Any]:
    """The panel's view. Shape:

    ``authority``: ``{"state": "not-set-up"}`` | ``{"state": "unreadable", "reason"}``
    | ``{"state": "ready", "fingerprint_sha256", "fingerprint_sha1", "created_at",
    "expires_at"}``; ``kind`` and ``kind_reason``; ``leaf``: ``{"expires_at",
    "names"}`` or None; ``migration``: ``{"state": none|waiting|blocked|done,
    "reason": str|None}``; ``names_left_out``: ``[{"name", "reason"}]``.
    """
    now = now or datetime.datetime.now(datetime.timezone.utc)
    root, root_reason = _load_cert(layout.public_root)
    served, leaf_reason = _load_cert(layout.leaf_cert)
    status: Dict[str, Any] = {"kind": pki.KIND_UNREADABLE, "kind_reason": leaf_reason,
                              "authority": {"state": "not-set-up"}, "leaf": None,
                              "migration": {"state": "none", "reason": None}}
    if root_reason:
        status["authority"] = {"state": "unreadable", "reason": root_reason}
        if served is None and not leaf_reason:
            status["kind"] = pki.KIND_NONE
        elif not leaf_reason:
            # Without the root, how the served certificate relates to it is
            # unknown; guessing would be a label not measured (LESSONS 5).
            status["kind_reason"] = "the household root could not be read"
    elif not leaf_reason:
        status["kind"] = pki.classify(served, root, (socket.gethostname(),))
    if root is not None:
        status["authority"] = {
            "state": "ready",
            "fingerprint_sha256": pki.fingerprint(root, "sha256"),
            "fingerprint_sha1": pki.fingerprint(root, "sha1"),
            "created_at": root.not_valid_before_utc.isoformat(),
            "expires_at": root.not_valid_after_utc.isoformat(),
        }
    if served is not None:
        names, addresses = pki.leaf_identities(served)
        status["leaf"] = {"expires_at": served.not_valid_after_utc.isoformat(),
                          "names": list(names) + list(addresses)}
    status["migration"] = _migration(layout, status["kind"], root, now)
    status["names_left_out"] = names_left_out(
        root, socket.gethostname() if hostname is None else hostname)
    return status
