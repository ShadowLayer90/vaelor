"""How a browser trusts this console: transport status, the household root, and the
commands and QR code that install it (VD-212).

One route here answers without a session, by the owner's decision: the root's
public certificate at :data:`vaelor.tls_paths.ROOT_ROUTE`, so a first-time
owner can trust the console before typing a password. It serves only that
certificate, re-encoded from its parse, and the fingerprint check in every
command is what makes an unauthenticated download safe. Everything else needs
a viewer's session.
"""

from __future__ import annotations

import hashlib
import io
from pathlib import Path
from typing import Any, Dict

from cryptography import x509
from cryptography.hazmat.primitives.serialization import Encoding
from flask import Response, request

from . import tls_authority
from . import tls_trust_commands as trust
from .api_common import ApiContext, payload as _payload
from .tls_paths import ROOT_ROUTE, leaf_cert

_PREFIX = "/api/v2"
if not ROOT_ROUTE.startswith(_PREFIX + "/"):  # the blueprint's own prefix
    raise RuntimeError("ROOT_ROUTE must sit under " + _PREFIX)
#: The public route, relative to the blueprint.
ROOT_RULE = ROOT_ROUTE[len(_PREFIX):]

#: Both certificate downloads (the public root route and the old
#: /security/certificate path, kept for compatibility) send this name.
ROOT_FILENAME = "vaelor-household-root"


def _not_ready(unavailable: trust.TrustUnavailable):
    status = 404 if unavailable.state == "not-set-up" else 503
    return _payload(
        error={"code": "authority_" + unavailable.state.replace("-", "_"),
               "message": str(unavailable)},
        status=status,
    )


def _status() -> Dict[str, Any]:
    """Stream A's one reading of the root and the served certificate (LESSONS 6)."""
    return tls_authority.authority_status()


def _root_or_refusal():
    """``(root, None)`` when the authority is ready, else ``(None, refusal)``.

    Whether there is a root, and whether it could be read, is
    :func:`vaelor.tls_authority.authority_status`'s answer, not this module's;
    the bytes served are then the renderer's parse of that same public copy.
    """
    authority = _status().get("authority") or {}
    state = authority.get("state")
    if state != "ready":
        if state == "not-set-up":
            return None, _not_ready(trust.TrustUnavailable("not-set-up", trust.NOT_SET_UP))
        reason = authority.get("reason")
        return None, _not_ready(trust.TrustUnavailable(
            "unreadable", "The household authority's certificate could not be read{}.".format(
                ": " + reason if reason else "")))
    try:
        return trust.read_root(trust.public_root_sources()), None
    except trust.TrustUnavailable as unavailable:  # changed between the two reads
        return None, _not_ready(unavailable)


def _root_response(attachment: bool):
    """The root's public certificate, PEM by default, DER with ``?form=der``."""
    root, refusal = _root_or_refusal()
    if refusal is not None:
        return refusal
    der = request.args.get("form", "pem").lower() == "der"
    filename = ROOT_FILENAME + (".cer" if der else ".crt")
    disposition = "attachment" if attachment else "inline"
    return Response(
        root.der if der else root.pem,
        mimetype="application/x-x509-ca-cert",
        headers={
            "Content-Disposition": '{}; filename="{}"'.format(disposition, filename),
            "Cache-Control": "no-store",
            "X-Content-Type-Options": "nosniff",
        },
    )


def _leaf_fingerprint() -> str:
    """The pre-VD-212 ``certificate_fingerprint`` field, in its old form
    (lower-case hex, no colons); empty when there is no leaf it can read."""
    try:
        leaf = x509.load_pem_x509_certificate(Path(leaf_cert()).read_bytes())
    except (OSError, ValueError):
        return ""
    return hashlib.sha256(leaf.public_bytes(Encoding.DER)).hexdigest()


def _transport_authority(status: Dict[str, Any]) -> Dict[str, Any]:
    """A's ``authority`` and ``kind`` passed through, in the shape the panel reads:
    ``reason`` is surfaced as ``message``; nothing is re-derived here."""
    facts = status.get("authority") or {"state": "not-set-up"}
    return {
        "state": facts.get("state", "not-set-up"),
        "message": facts.get("reason") or "",
        "kind": status.get("kind") or "none",
        "kind_reason": status.get("kind_reason"),
        "fingerprint_sha256": facts.get("fingerprint_sha256", ""),
        "fingerprint_sha1": facts.get("fingerprint_sha1", ""),
        "created_at": facts.get("created_at"),
        "expires_at": facts.get("expires_at"),
    }


def _console_address():
    host = request.host
    return host if trust.valid_address(host) else None


def _bad_address():
    return _payload(
        error={"code": "invalid_console_address",
               "message": "This console was reached by an address Vaelor will "
                          "not put into a command. Open it by its IP address or "
                          "host name and try again."},
        status=400,
    )


def register_trust_routes(context: ApiContext) -> None:
    blueprint = context.blueprint
    require_auth = context.require_auth

    @blueprint.get(ROOT_RULE)
    def household_root_certificate():
        # Public on purpose (VD-212 build answers): no session, root cert only.
        return _root_response(attachment=True)

    @blueprint.get("/security/certificate")
    @require_auth("viewer")
    def transport_certificate():
        # The old leaf download; it now hands out the household root, which is
        # what a device should trust (VD-212). The path stays for compatibility.
        return _root_response(attachment=True)

    @blueprint.get("/security/transport")
    @require_auth("viewer")
    def transport_security():
        fingerprint = _leaf_fingerprint()
        status = _status()
        authority = _transport_authority(status)
        leaf_facts = status.get("leaf")
        return _payload({
            "secure": request.is_secure,
            "scheme": "https" if request.is_secure else "http",
            "certificate_managed": bool(fingerprint),
            "certificate_fingerprint": fingerprint,
            "vnc_secure": bool(request.is_secure and fingerprint),
            "remote_ready": bool(request.is_secure and fingerprint),
            "authority": authority,
            "leaf": leaf_facts,
            # Passed through as stream A states them (LESSONS 6 / 10): whether
            # the switch to the household certificate waits, is held, or is
            # done, and why; and each name the root could not cover. None when
            # the authority did not report it - never an invented "none".
            "migration": status.get("migration"),
            "names_left_out": status.get("names_left_out"),
        })

    @blueprint.get("/security/trust/commands")
    @require_auth("viewer")
    def trust_commands():
        address = _console_address()
        if address is None:
            return _bad_address()
        root, refusal = _root_or_refusal()
        if refusal is not None:
            return refusal
        return _payload(trust.render(root, address))

    @blueprint.get("/security/trust/qr.svg")
    @require_auth("viewer")
    def trust_qr():
        address = _console_address()
        if address is None:
            return _bad_address()
        _root, refusal = _root_or_refusal()
        if refusal is not None:
            return refusal
        try:
            import segno  # declared in pyproject.toml; checked here so a missing one is said
        except ImportError:
            return _payload(
                error={"code": "qr_unavailable",
                       "message": "The QR code library (segno) is not installed "
                                  "on this Vaelor."},
                status=503,
            )
        buffer = io.BytesIO()
        # Dark on white whatever the console's theme: a phone camera reads that.
        segno.make(trust.root_url(address), error="m").save(
            buffer, kind="svg", scale=4, border=4, dark="#000000", light="#ffffff",
            xmldecl=False, title="Trust this Vaelor: " + trust.root_url(address))
        return Response(buffer.getvalue(), mimetype="image/svg+xml",
                        headers={"Cache-Control": "no-store",
                                 "X-Content-Type-Options": "nosniff"})
