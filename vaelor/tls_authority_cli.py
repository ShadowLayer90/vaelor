"""``python -m vaelor.tls_authority``: the household authority's commands (VD-212).

    ensure                    one pass: create/publish/trust/issue/migrate (installer, idempotent)
    run                       the service loop: ``ensure`` every five minutes
    status                    what the console may show, as JSON (no key material)
    promote                   serve the pending certificate now (owner override of the fleet gate)
    export --output FILE      the root, encrypted with a passphrase the owner chooses
    import FILE [--replace]   restore an exported root
    recreate --replace        a new root, when the old key can no longer be decrypted

Exit codes: 0 done, 1 failed (the reason is on stderr), 2 usage, 3 refused
(an import over a different root without ``--replace``).

The export is opt-in, CLI only, and never part of a scheduled backup. Its
envelope (``vaelor-authority/1``) is scrypt + AES-256-GCM through
:func:`vaelor.portable_state_core.derive_passphrase_key`, the same passphrase
rule as the portable-state archive. The passphrase comes from
``VAELOR_AUTHORITY_PASSPHRASE`` or an interactive prompt, never argv.
"""

from __future__ import annotations

import argparse
import base64
import getpass
import json
import os
import signal
import sys
import threading
from typing import Any, Dict, Optional, Sequence

from cryptography.exceptions import InvalidTag
from cryptography.hazmat.primitives import serialization
from cryptography.hazmat.primitives.ciphers.aead import AESGCM

from . import tls_authority_pki as pki
from . import tls_public_names
from .portable_state_core import SCRYPT_PARAMETERS, PortableStateError, derive_passphrase_key
from .tls_identity import PRIVATE_DNS_SUFFIXES, PRIVATE_LAN_NETWORKS
from .tls_authority import TICK_SECONDS, Authority, AuthorityError, TickReport

EXPORT_SCHEMA = "vaelor-authority/1"
PASSPHRASE_ENV = "VAELOR_AUTHORITY_PASSPHRASE"
EXIT_OK, EXIT_FAILED, EXIT_REFUSED = 0, 1, 3
_MAX_EXPORT_BYTES = 1 << 16
_LEFT_OUT_OF_ROOT = "tls-authority: left out of the root: "
_NOT_AN_EXPORT = "This is not a supported household authority export."


def _require_root() -> None:
    if hasattr(os, "geteuid") and os.geteuid() != 0:
        raise AuthorityError("The household authority runs as root; use sudo.")


def _passphrase(confirm: bool) -> str:
    value = os.environ.get(PASSPHRASE_ENV, "")
    if value:
        return value
    first = getpass.getpass("Passphrase for the household authority export: ")
    if confirm and getpass.getpass("Repeat the passphrase: ") != first:
        raise AuthorityError("The two passphrases differ; nothing was written.")
    return first


def _header_bytes(header: Dict[str, Any]) -> bytes:
    return json.dumps(header, separators=(",", ":"), sort_keys=True).encode("utf-8")


def seal_export(cert, key, passphrase: str) -> bytes:
    salt, nonce = os.urandom(16), os.urandom(12)
    header = dict(SCRYPT_PARAMETERS, kdf="scrypt", schema=EXPORT_SCHEMA,
                  salt=base64.b64encode(salt).decode("ascii"),
                  nonce=base64.b64encode(nonce).decode("ascii"))
    plaintext = json.dumps({
        "certificate": pki.cert_pem(cert).decode("ascii"),
        "key": pki.key_pem(key).decode("ascii"),
    }).encode("utf-8")
    try:
        sealed = AESGCM(derive_passphrase_key(passphrase, salt)).encrypt(
            nonce, plaintext, _header_bytes(header))
    except PortableStateError as error:
        raise AuthorityError(str(error)) from None
    return json.dumps(dict(header, ciphertext=base64.b64encode(sealed).decode(
        "ascii")), sort_keys=True).encode("utf-8") + b"\n"


def open_export(data: bytes, passphrase: str):
    """(certificate, key) from an export; refuses anything but a household root."""
    try:
        document = json.loads(data.decode("utf-8"))
        ciphertext = base64.b64decode(document.pop("ciphertext"), validate=True)
        if document.get("schema") != EXPORT_SCHEMA or document.get("kdf") != "scrypt":
            raise AuthorityError(_NOT_AN_EXPORT)
        # The cost the export was written with (its header, bound to the
        # ciphertext as associated data), bounded by derive_passphrase_key.
        cost = {k: document.get(k) for k in SCRYPT_PARAMETERS}
        salt = base64.b64decode(document["salt"], validate=True)
        nonce = base64.b64decode(document["nonce"], validate=True)
    except AuthorityError:
        raise
    except (UnicodeDecodeError, ValueError, KeyError, TypeError, AttributeError):
        raise AuthorityError(_NOT_AN_EXPORT) from None
    try:
        plain = AESGCM(derive_passphrase_key(passphrase, salt, **cost)).decrypt(
            nonce, ciphertext, _header_bytes(document))
    except PortableStateError as error:
        raise AuthorityError(str(error)) from None
    except (InvalidTag, ValueError):
        raise AuthorityError("The passphrase is incorrect or the export was "
                             "changed.") from None
    try:
        content = json.loads(plain.decode("utf-8"))
        cert = pki.load_pem_certificates(content["certificate"].encode("ascii"))[0]
        key = serialization.load_pem_private_key(
            content["key"].encode("ascii"), password=None)
    except (UnicodeDecodeError, ValueError, KeyError, TypeError, AttributeError):
        raise AuthorityError("The export decrypted but holds no usable root.") from None
    finally:
        del plain
    if not pki.key_matches(cert, key):
        raise AuthorityError("The exported key does not match its certificate.")
    if not (pki.is_household_root(cert) and pki.is_certificate_authority(cert)):
        raise AuthorityError("The export is not a household root.")
    dns, networks = pki.permitted_by_root(cert)
    problems = tls_public_names.constraint_violations(
        dns, networks, pki.name_constraints_critical(cert),
        PRIVATE_DNS_SUFFIXES, PRIVATE_LAN_NETWORKS)
    if problems:
        raise AuthorityError("The exported root may vouch for more than the private "
                             "LAN, so it is not imported: {}.".format("; ".join(problems)))
    return cert, key


def _print_report(report: TickReport, quiet_findings: bool = False) -> None:
    """Actions and errors always; notes and dropped names unless ``quiet_findings``
    (the loop says those once, when they change, not every five minutes)."""
    out = sys.stdout
    for line in report.actions:
        print("tls-authority: " + line, file=out)
    for line in [] if quiet_findings else report.notes:
        print("tls-authority: note: " + line, file=out)
    for line in [] if quiet_findings else report.dropped:
        print("tls-authority: left out of the console certificate: " + line, file=out)
    migration = report.migration
    if migration.get("state") in ("waiting", "blocked") and not quiet_findings:
        reason = migration.get("reason") or ""
        if "promote`" not in reason:
            reason += "; override: `python -m vaelor.tls_authority promote`"
        print("tls-authority: migration {}: {}".format(migration["state"], reason),
              file=out)
    for line in report.root_dropped:
        print(_LEFT_OUT_OF_ROOT + line, file=out)
    for line in report.errors:
        print("tls-authority: ERROR: " + line, file=sys.stderr)
    out.flush()


def _ensure(authority: Authority, force_promote: bool = False) -> int:
    report = authority.tick(force_promote=force_promote)
    _print_report(report)
    return EXIT_FAILED if report.errors else EXIT_OK


def _run(authority: Authority, interval: float = TICK_SECONDS,
         stop: Optional[threading.Event] = None) -> int:
    """Tick until SIGTERM. A failed tick is logged and retried; it never stops
    the loop, so one bad pass (say, update-ca-certificates) cannot crash-loop
    the unit. ``stop`` is the test seam; the service passes none."""
    if stop is None:
        stop = threading.Event()
        signal.signal(signal.SIGTERM, lambda *_: stop.set())
    said = None
    while not stop.is_set():
        report = authority.tick()
        findings = (tuple(report.notes), tuple(report.dropped),
                    tuple(sorted(report.migration.items())))
        _print_report(report, quiet_findings=findings == said)
        said = findings
        stop.wait(interval)
    return EXIT_OK


def _export(authority: Authority, output: str) -> int:
    cert, key = authority.load_root()
    if cert is None:
        raise AuthorityError("There is no household root to export yet.")
    data = seal_export(cert, key, _passphrase(confirm=True))
    descriptor = os.open(output, os.O_CREAT | os.O_EXCL | os.O_WRONLY, 0o600)
    with os.fdopen(descriptor, "wb") as stream:
        stream.write(data)
    print("tls-authority: exported the household root to {} (SHA-256 {}). Keep the "
          "file and its passphrase apart.".format(output, pki.fingerprint(cert)))
    return EXIT_OK


def _import(authority: Authority, source: str, replace: bool) -> int:
    with open(source, "rb") as stream:
        data = stream.read(_MAX_EXPORT_BYTES + 1)
    if len(data) > _MAX_EXPORT_BYTES:
        raise AuthorityError(_NOT_AN_EXPORT)
    cert, key = open_export(data, _passphrase(confirm=False))
    try:
        existing, _ = authority.load_root()
    except AuthorityError:
        if not replace:
            raise
        existing = None
    if existing is not None and pki.fingerprint(existing) == pki.fingerprint(cert):
        print("tls-authority: this machine already holds that household root.")
        return _ensure(authority)
    if existing is not None and not replace:
        print("tls-authority: REFUSED: this machine already has a different household "
              "root ({}). Re-run with --replace to use the imported one; every device "
              "that trusts the current root must then trust the new one.".format(
                  pki.fingerprint(existing)), file=sys.stderr)
        return EXIT_REFUSED
    with authority.locked():
        authority.store_root(cert, key)
    print("tls-authority: imported the household root (SHA-256 {}).".format(
        pki.fingerprint(cert)))
    if existing is not None:
        print("tls-authority: every device that trusted the previous root must now "
              "trust this one.")
    return _ensure(authority)


def _recreate(authority: Authority) -> int:
    with authority.locked():
        authority.create_root()
    for line in authority.root_dropped:
        print(_LEFT_OUT_OF_ROOT + line)
    print("tls-authority: made a new household root; every device must trust it again.")
    return _ensure(authority)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="python -m vaelor.tls_authority",
                                     description="Vaelor's household authority (VD-212).")
    sub = parser.add_subparsers(dest="command", required=True)
    sub.add_parser("ensure", help="one idempotent pass (installer, service start)")
    sub.add_parser("run", help="the service loop")
    sub.add_parser("status", help="what the console shows, as JSON")
    sub.add_parser("promote", help="serve the pending certificate now")
    export = sub.add_parser("export", help="write the root, passphrase-encrypted")
    export.add_argument("--output", required=True)
    imported = sub.add_parser("import", help="restore an exported root")
    imported.add_argument("file")
    imported.add_argument("--replace", action="store_true")
    recreate = sub.add_parser("recreate", help="make a new root")
    recreate.add_argument("--replace", action="store_true", required=True)
    return parser


def main(argv: Optional[Sequence[str]] = None, authority: Optional[Authority] = None) -> int:
    args = build_parser().parse_args(argv)
    if args.command == "status":
        from .tls_authority import authority_status

        print(json.dumps(authority_status(), indent=2, sort_keys=True))
        return EXIT_OK
    try:
        _require_root()
        authority = authority or Authority()
        if args.command == "ensure":
            return _ensure(authority)
        if args.command == "run":
            return _run(authority)
        if args.command == "promote":
            return _ensure(authority, force_promote=True)
        if args.command == "export":
            return _export(authority, args.output)
        if args.command == "import":
            return _import(authority, args.file, args.replace)
        return _recreate(authority)
    except AuthorityError as error:
        print("tls-authority: ERROR: {}".format(error), file=sys.stderr)
        return EXIT_FAILED
    except OSError as error:
        print("tls-authority: ERROR: {} ({})".format(
            error.strerror or type(error).__name__, error.filename or ""), file=sys.stderr)
        return EXIT_FAILED
