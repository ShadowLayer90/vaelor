"""The household authority: a root unique to this install that signs the console's certificate (VD-212).

Runs as root, in ``vaelor-tls-authority.service`` (``python -m
vaelor.tls_authority run``) and once from the installer (``ensure``). One pass
(:meth:`Authority.tick`) does everything, idempotently:

1. Creates the root if there is none (cryptography, P-256, ten years,
   name-constrained to the private LAN). Its key is sealed with
   ``systemd-creds --with-key=host`` and only ever decrypted into memory.
2. Publishes the root's public certificate, and installs it in this
   controller's own OS trust store.
3. Keeps the served certificate right: issues one when there is none,
   re-issues before expiry or when the machine's names change, migrates an
   install that serves the pre-VD-212 self-signed certificate (phases 0-2,
   gated by :mod:`vaelor.tls_fleet_gate`), and never touches a ``custom``
   certificate the owner installed.

Fail loud (VD-212): a missing or unreadable authority is an error the caller
sees, never a fallback to a self-signed certificate.

Private key hygiene (VD-124, LESSONS 24): no key material in a log line, an
exception, argv, /tmp, or any file other than the sealed root and the served
leaf key (0640 root:vaelor, which the console needs to serve TLS).
"""

from __future__ import annotations

import collections
import contextlib
import datetime
import glob
import hashlib
import os
import re
import secrets
import socket
import subprocess
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional, Tuple

from cryptography.hazmat.primitives import serialization

from . import tls_authority_pki as pki
from . import tls_fleet_gate, tls_identity, tls_paths
from .tls_authority_host import (  # re-exported: callers import them from here
    CREDENTIAL_NAME, SERVICE_GROUP, AuthorityError, Runner, SystemdCreds,
    _COMMAND_TIMEOUT_SECONDS, _scrubbed, default_chown)

#: The renewal loop's period.
TICK_SECONDS = 300
#: Keep the replaced self-signed pair this long after promotion (phase 2).
LEGACY_RETENTION_SECONDS = 7 * 24 * 3600
#: A change of names must be read this many times running before it re-issues.
IDENTITY_CONFIRMATIONS = 2
#: ...and re-issues for a name change are capped per day (anti-flap).
IDENTITY_REISSUES_PER_DAY = 4
LEGACY_SUFFIX = ".legacy"
_TRUST_FILE = re.compile(r"vaelor-household-[0-9a-f]{16}\.crt")


@dataclass(frozen=True)
class Layout:
    authority_dir: str
    root_cert: str
    root_key_cred: str
    public_root: str
    worker_bundle: str
    pending_dir: str
    leaf_cert: str
    leaf_key: str
    fleet_state: str
    controller_trust_template: str

    @classmethod
    def default(cls) -> "Layout":
        return cls(
            authority_dir=tls_paths.AUTHORITY_DIR,
            root_cert=tls_paths.ROOT_CERT,
            root_key_cred=tls_paths.ROOT_KEY_CRED,
            public_root=tls_paths.PUBLIC_ROOT,
            worker_bundle=tls_paths.WORKER_TRUST_BUNDLE,
            pending_dir=tls_paths.PENDING_DIR,
            leaf_cert=tls_paths.leaf_cert(),
            leaf_key=tls_paths.leaf_key(),
            fleet_state=tls_paths.FLEET_TRUST_STATE,
            controller_trust_template=tls_paths.CONTROLLER_OS_TRUST_TEMPLATE,
        )

    @property
    def pending_cert(self) -> str:
        return os.path.join(self.pending_dir, os.path.basename(self.leaf_cert))

    @property
    def pending_key(self) -> str:
        return os.path.join(self.pending_dir, os.path.basename(self.leaf_key))


def _read(path: str) -> Optional[bytes]:
    try:
        with open(path, "rb") as stream:
            return stream.read()
    except FileNotFoundError:
        return None


def _key_at(path: str):
    """The private key in ``path``, or None when absent or unusable."""
    data = _read(path)
    if not data:
        return None
    try:
        return serialization.load_pem_private_key(data, password=None)
    except (ValueError, TypeError):
        return None


def pair_matches(cert, key_path: str) -> bool:
    """True when the key file holds ``cert``'s private half (F2: a crash
    between the two replaces must never leave a pair that is served or kept)."""
    return cert is not None and pki.key_matches(cert, _key_at(key_path))


def _first_cert(path: str):
    """The first certificate in ``path``, or None when the file is absent."""
    cert, _ = _load_cert(path)
    return cert


def _load_cert(path: str):
    """(certificate, None), (None, None) when absent, or (None, reason) when the
    file exists but cannot be read or parsed (LESSONS 1: never shown as absent).
    The reason names the file and the failure, never its content."""
    try:
        data = _read(path)
    except OSError as error:
        return None, "{} could not be read ({})".format(
            os.path.basename(path), error.strerror or type(error).__name__)
    if data is None:
        return None, None
    try:
        return pki.load_pem_certificates(data)[0], None
    except (ValueError, IndexError):
        return None, "{} is not a PEM certificate".format(os.path.basename(path))


@contextlib.contextmanager
def _directory_lock(path: str):
    """Serialise the service loop and a CLI run (flock on the directory)."""
    try:
        import fcntl
    except ImportError:
        yield
        return
    descriptor = os.open(path, os.O_RDONLY)
    try:
        fcntl.flock(descriptor, fcntl.LOCK_EX)
        yield
    finally:
        os.close(descriptor)


@dataclass
class TickReport:
    kind: str = pki.KIND_NONE
    phase: str = ""
    actions: List[str] = field(default_factory=list)
    dropped: Tuple[str, ...] = ()
    root_dropped: Tuple[str, ...] = ()
    #: {"state": "none"|"waiting"|"blocked"|"done", "reason": str|None}
    migration: Dict[str, Any] = field(
        default_factory=lambda: {"state": "none", "reason": None})
    notes: List[str] = field(default_factory=list)
    errors: List[str] = field(default_factory=list)

    def as_dict(self) -> Dict[str, Any]:
        return {"kind": self.kind, "phase": self.phase, "actions": self.actions,
                "dropped": list(self.dropped), "root_dropped": list(self.root_dropped),
                "migration": dict(self.migration),
                "notes": self.notes,
                "errors": self.errors}


class Authority:
    def __init__(
        self,
        layout: Optional[Layout] = None,
        sealer: Any = None,
        *,
        run: Runner = subprocess.run,
        chown: Callable[[str, bool], None] = default_chown,
        now: Callable[[], datetime.datetime] = lambda: datetime.datetime.now(
            datetime.timezone.utc),
        read_identity: Callable[[], tls_identity.IdentityReading] = tls_identity.read_identity,
        hostname: Callable[[], str] = socket.gethostname,
        root_key_factory: Callable[[], Any] = pki.new_root_key,
        leaf_key_factory: Callable[[], Any] = pki.new_leaf_key,
    ):
        self.layout = layout or Layout.default()
        self.sealer = sealer or SystemdCreds(run)
        self._run = run
        self._chown = chown
        self._now = now
        self._read_identity = read_identity
        self._hostname = hostname
        self._root_key_factory = root_key_factory
        self._leaf_key_factory = leaf_key_factory
        self._streak: Tuple[Any, int] = (None, 0)
        self._identity_reissues: collections.deque = collections.deque()
        self._os_trust_done = False
        #: What the last created root left out of its constraints, and why.
        self.root_dropped: Tuple[str, ...] = ()

    # -- files ------------------------------------------------------------
    def _write(self, path: str, data: bytes, mode: int, service_group: bool = False) -> bool:
        """Stage beside ``path``, set mode and owner, fsync, then os.replace.

        Returns False (and writes nothing) when the content is already there.
        """
        if _read(path) == data and (os.name != "posix" or (
                os.stat(path).st_mode & 0o7777) == mode):
            return False
        directory = os.path.dirname(path) or "."
        stage = os.path.join(directory, ".{}.{}.tmp".format(
            os.path.basename(path), secrets.token_hex(6)))
        descriptor = os.open(stage, os.O_CREAT | os.O_EXCL | os.O_WRONLY, 0o600)
        try:
            with os.fdopen(descriptor, "wb") as stream:
                stream.write(data)
                stream.flush()
                os.fsync(stream.fileno())
            os.chmod(stage, mode)
            self._chown(stage, service_group)
            os.replace(stage, path)
        except BaseException:
            with contextlib.suppress(OSError):
                os.unlink(stage)
            raise
        return True

    def _directory(self, path: str, mode: int, service_group: bool = False,
                   service_user: bool = False) -> None:
        if not os.path.isdir(path):
            os.makedirs(path, mode)
            os.chmod(path, mode)
            self._chown(path, service_group, service_user)

    def _fleet_folder(self, report: TickReport) -> None:
        """The control plane's folder for the fleet trust record, made if missing.

        A console upgrade never runs the installer, so an upgraded controller
        may lack it and the control plane (no write on the state root) could
        never record its sweep. Made vaelor:vaelor 0750; the record inside is
        only ever written by the control plane, never here.
        """
        folder = os.path.dirname(self.layout.fleet_state)
        try:
            self._directory(folder, 0o750, service_group=True, service_user=True)
        except OSError as error:
            report.errors.append("{} could not be created ({}).".format(
                folder, error.strerror or type(error).__name__))

    # -- the root ---------------------------------------------------------
    def load_root(self):
        """(certificate, key) of the root, or (None, None) when there is none."""
        cert_bytes = _read(self.layout.root_cert)
        sealed = _read(self.layout.root_key_cred)
        if cert_bytes is None and sealed is None:
            return None, None
        if cert_bytes is None or sealed is None:
            raise AuthorityError(
                "The household authority is incomplete: {} is missing. Restore it "
                "with `python -m vaelor.tls_authority import FILE --replace`, or make "
                "a new root with `python -m vaelor.tls_authority recreate --replace` "
                "(every device must then trust the new one).".format(
                    self.layout.root_cert if cert_bytes is None
                    else self.layout.root_key_cred))
        try:
            cert = pki.load_pem_certificates(cert_bytes)[0]
        except ValueError:
            raise AuthorityError("{} is not a certificate.".format(
                self.layout.root_cert)) from None
        plain = self.sealer.unseal(sealed)
        try:
            key = serialization.load_pem_private_key(plain, password=None)
        except (ValueError, TypeError):
            raise AuthorityError("The household root key did not decrypt to a "
                                 "usable key.") from None
        finally:
            del plain
        if not pki.key_matches(cert, key):
            raise AuthorityError("The household root key does not match {}.".format(
                self.layout.root_cert))
        return cert, key

    def store_root(self, cert, key) -> None:
        """Seal and write a root (creation, import, recreate). Key first."""
        self._directory(self.layout.authority_dir, 0o700)
        os.chmod(self.layout.authority_dir, 0o700)
        self._write(self.layout.root_key_cred, self.sealer.seal(pki.key_pem(key)), 0o600)
        self._write(self.layout.root_cert, pki.cert_pem(cert), 0o644)

    def create_root(self):
        label = tls_identity.bare_hostname(self._hostname())
        _, refusal = pki.root_hostname_label(label)
        self.root_dropped = ("{} ({})".format(label, refusal),) if refusal else ()
        cert, key = pki.new_root(label, self._now(), self._root_key_factory)
        self.store_root(cert, key)
        return cert, key

    def install_controller_trust(self, root, report: TickReport) -> None:
        """The root in this controller's OS store, on every authority start."""
        template = self.layout.controller_trust_template
        target = template.format(pki.root_id(root))
        directory = os.path.dirname(target)
        changed = False
        if not os.path.isdir(directory):
            report.errors.append("The OS trust folder {} does not exist; install "
                                 "the ca-certificates package.".format(directory))
            return
        changed |= self._write(target, pki.cert_pem(root), 0o644)
        for stale in glob.glob(os.path.join(directory, "vaelor-household-*.crt")):
            if _TRUST_FILE.fullmatch(os.path.basename(stale)) and stale != target:
                os.unlink(stale)
                changed = True
        if not changed and self._os_trust_done:
            return
        try:
            result = self._run(["update-ca-certificates"], capture_output=True,
                               timeout=_COMMAND_TIMEOUT_SECONDS, check=False)
        except (OSError, subprocess.SubprocessError) as error:
            report.errors.append("update-ca-certificates could not run ({}).".format(
                type(error).__name__))
            return
        if result.returncode != 0:
            report.errors.append("update-ca-certificates failed (exit {}): {}".format(
                result.returncode, _scrubbed(result.stderr)))
            return
        self._os_trust_done = True
        if changed:
            report.actions.append("installed the household root in this machine's "
                                  "trust store")

    # -- leaves -----------------------------------------------------------
    def _wanted_identity(self, root, report: TickReport):
        reading = self._read_identity()
        dns, networks = pki.permitted_by_root(root)
        names, addresses, dropped = tls_identity.filter_identity(reading, dns, networks)
        report.dropped = dropped
        report.notes.extend(reading.notes)
        return (names, addresses), reading.complete

    def _issue(self, root, root_key, identity, cert_path: str, key_path: str) -> None:
        names, addresses = identity
        cert, key = pki.issue_leaf(root, root_key, names, addresses, self._now(),
                                   self._leaf_key_factory)
        # Key first: the console reloads on the certificate's mtime, so by the
        # time the certificate changes its key is already in place.
        self._write(key_path, pki.key_pem(key), 0o640, service_group=True)
        self._write(cert_path, pki.cert_pem(cert), 0o644, service_group=True)

    def _renewal_reason(self, leaf, wanted, complete: bool) -> str:
        now = self._now()
        if pki.expires_within(leaf, now):
            return "it expires within {} days".format(pki.RENEW_BEFORE_DAYS)
        if pki.leaf_identities(leaf) == wanted or not complete:
            self._streak = (None, 0)
            return ""
        seen, count = self._streak
        count = count + 1 if seen == wanted else 1
        self._streak = (wanted, count)
        if count < IDENTITY_CONFIRMATIONS:
            return ""
        stamp = now.timestamp()
        while self._identity_reissues and stamp - self._identity_reissues[0] > 86400:
            self._identity_reissues.popleft()
        if len(self._identity_reissues) >= IDENTITY_REISSUES_PER_DAY:
            return ""
        self._identity_reissues.append(stamp)
        self._streak = (None, 0)
        return "this machine's names or addresses changed"

    def _bundle(self, root, served_bytes: Optional[bytes]) -> bytes:
        data = pki.cert_pem(root)
        if served_bytes:
            data += served_bytes if served_bytes.endswith(b"\n") else served_bytes + b"\n"
        return data

    def _write_bundle(self, data: bytes, report: TickReport) -> None:
        if self._write(self.layout.worker_bundle, data, 0o644, service_group=True):
            report.actions.append("updated the worker trust bundle")

    def promote(self, report: TickReport, reason: str) -> None:
        """Serve the pending certificate; keep the replaced pair 7 days.

        Only a served pair whose key matches its certificate is backed up, so
        an earlier interrupted pass can never overwrite a good backup with a
        mismatched key.
        """
        layout = self.layout
        if pair_matches(_first_cert(layout.leaf_cert), layout.leaf_key):
            for live, mode in ((layout.leaf_key, 0o640), (layout.leaf_cert, 0o644)):
                self._write(live + LEGACY_SUFFIX, _read(live), mode, service_group=True)
                os.utime(live + LEGACY_SUFFIX, None)
        else:
            report.notes.append("the served certificate and key did not match, so "
                                "they were not kept as the backup")
        os.replace(layout.pending_key, layout.leaf_key)
        os.replace(layout.pending_cert, layout.leaf_cert)
        report.actions.append("now serving the household certificate ({})".format(reason))

    def _pending_ok(self, root, wanted, complete: bool) -> bool:
        pending = _first_cert(self.layout.pending_cert)
        if not pair_matches(pending, self.layout.pending_key):
            return False
        if not pki.directly_issued_by(pending, root):
            return False
        return not self._renewal_reason(pending, wanted, complete)

    def _retire_legacy(self, report: TickReport) -> None:
        legacy = self.layout.leaf_cert + LEGACY_SUFFIX
        try:
            age = self._now().timestamp() - os.stat(legacy).st_mtime
        except FileNotFoundError:
            return
        if age < LEGACY_RETENTION_SECONDS:
            report.phase = "2"
            return
        for path in (legacy, self.layout.leaf_key + LEGACY_SUFFIX):
            with contextlib.suppress(FileNotFoundError):
                os.unlink(path)
        report.actions.append("removed the replaced self-signed certificate")

    def _clear_pending(self) -> None:
        for path in (self.layout.pending_cert, self.layout.pending_key):
            with contextlib.suppress(FileNotFoundError):
                os.unlink(path)

    # -- one pass ---------------------------------------------------------
    def locked(self):
        """Hold the authority folder's lock (a CLI import against the loop).

        Never nested: a second flock from this process on a new descriptor
        would wait on the first forever.
        """
        self._directory(self.layout.authority_dir, 0o700)
        return _directory_lock(self.layout.authority_dir)

    def tick(self, *, force_promote: bool = False) -> TickReport:
        report = TickReport()
        with self.locked():
            try:
                self._tick(report, force_promote)
            except (AuthorityError, ValueError) as error:
                report.errors.append(str(error))
            except OSError as error:
                report.errors.append("{} on {}".format(
                    error.strerror or type(error).__name__, error.filename or "a file"))
        return report

    def _tick(self, report: TickReport, force_promote: bool) -> None:
        layout = self.layout
        root, root_key = self.load_root()
        if root is None:
            root, root_key = self.create_root()
            report.actions.append("created the household root")
            report.root_dropped = self.root_dropped
        self._directory(os.path.dirname(layout.public_root), 0o750, service_group=True)
        if self._write(layout.public_root, pki.cert_pem(root), 0o644):
            report.actions.append("published the household root certificate")
        self.install_controller_trust(root, report)

        self._fleet_folder(report)
        served_bytes = _read(layout.leaf_cert)
        served, unreadable = _load_cert(layout.leaf_cert)
        if unreadable:
            report.kind = pki.KIND_UNREADABLE
            report.errors.append("The served certificate {}; it is left in place. "
                                 "Replace or remove it, then run `python -m "
                                 "vaelor.tls_authority ensure`.".format(unreadable))
            return
        report.kind = pki.classify(served, root, (self._hostname(),))
        wanted, complete = self._wanted_identity(root, report)

        if report.kind == pki.KIND_CUSTOM:
            report.notes.append("a certificate this authority did not issue is "
                                "served; it is the owner's and is left in place")
            self._write_bundle(self._bundle(root, served_bytes), report)
            return
        if report.kind == pki.KIND_NONE:
            self._issue(root, root_key, wanted, layout.leaf_cert, layout.leaf_key)
            report.actions.append("issued the console certificate")
            report.kind = pki.KIND_HOUSEHOLD
            self._write_bundle(self._bundle(root, None), report)
            return
        if report.kind == pki.KIND_HOUSEHOLD:
            self._clear_pending()
            if pair_matches(served, layout.leaf_key):
                reason = self._renewal_reason(served, wanted, complete)
            else:
                reason = "its key did not match it"
            if reason:
                self._issue(root, root_key, wanted, layout.leaf_cert, layout.leaf_key)
                report.actions.append("re-issued the console certificate: " + reason)
            self._write_bundle(self._bundle(root, None), report)
            self._retire_legacy(report)
            return
        # Migrating: legacy self-signed, or a previous household root's leaf.
        self._migrate(root, root_key, served_bytes, wanted, complete, report,
                      force_promote)

    def _migrate(self, root, root_key, served_bytes, wanted, complete,
                 report: TickReport, force_promote: bool) -> None:
        layout = self.layout
        pending = _first_cert(layout.pending_cert)
        if (pending is not None and pki.directly_issued_by(pending, root)
                and not os.path.exists(layout.pending_key)
                and pair_matches(pending, layout.leaf_key)):
            # A promotion stopped between its two replaces: the new key is
            # served, its certificate still pending. Finish it.
            os.replace(layout.pending_cert, layout.leaf_cert)
            report.actions.append("finished an interrupted switch to the household "
                                  "certificate")
            report.kind, report.phase = pki.KIND_HOUSEHOLD, "2"
            self._write_bundle(self._bundle(root, None), report)
            return
        bundle = self._bundle(root, served_bytes)
        self._write_bundle(bundle, report)
        # 0750 root:vaelor: the console reads the pending certificate to show
        # why the switch waits; the pending key is as readable as the served one.
        self._directory(layout.pending_dir, 0o750, service_group=True)
        if not self._pending_ok(root, wanted, complete):
            self._issue(root, root_key, wanted, layout.pending_cert, layout.pending_key)
            report.actions.append("issued the household certificate, not served yet")
        report.phase = "0"
        pending = _first_cert(layout.pending_cert)
        issued_at = pki.issued_at(pending)
        if force_promote:
            verdict = tls_fleet_gate.PromotionVerdict(True, "promoted by the owner")
        else:
            names, addresses = pki.leaf_identities(pending)
            verdict = tls_fleet_gate.promotion_verdict(
                tls_fleet_gate.read_record(layout.fleet_state),
                hashlib.sha256(bundle).hexdigest(), issued_at,
                self._now().timestamp(), names, addresses)
        if not verdict.promote:
            # A waiting state, never an error: the installer stops on any failed
            # `ensure`, and nothing here is broken (review F3 correction).
            report.phase = "1"
            report.migration = {"state": "blocked" if verdict.blocked else "waiting",
                                "reason": verdict.reason}
            return
        report.migration = {"state": "done", "reason": verdict.reason}
        self.promote(report, verdict.reason)
        report.kind = pki.KIND_HOUSEHOLD
        report.phase = "2"
        self._write_bundle(self._bundle(root, None), report)


def authority_status(layout: Optional[Layout] = None,
                     now: Optional[datetime.datetime] = None,
                     hostname: Optional[str] = None) -> Dict[str, Any]:
    """What the console may show; see :mod:`vaelor.tls_authority_status`."""
    from .tls_authority_status import authority_status as read_status

    return read_status(layout or Layout.default(), now, hostname)


if __name__ == "__main__":  # pragma: no cover - python -m vaelor.tls_authority
    from .tls_authority_cli import main

    raise SystemExit(main())
