"""The TLS identity GNOME Remote Desktop presents to an RDP client.

**Every function here runs inside the privileged host-desktop broker, as
root.** gnome-remote-desktop's certificate directory is mode 0700 owned by
that account, and `grdctl --system` is a system-level call; the control plane
reaches all of it only through the broker's unix socket. Saying which process
runs what is the cheap habit LESSONS pattern 14 asks for, and it is the reason
`rdp_certificate_fingerprint` exists as a broker action rather than something the
Flask process reads for itself.

Split out of `host_desktop.py` so the certificate has room to explain itself
without pushing that module towards the 1,000-line ceiling.

**VD-088 records what was decided here and why**, including the one that a
future reader will be tempted to undo: the client's "Continue With Insecure
Connection?" dialog is a *symptom*, and it is removed by naming the machine in
the certificate rather than by lowering `authentication level` in the `.rdp`
profile, which would remove the check instead of the cause.

**The `rdp_` prefix is load-bearing, not decoration.** This appliance holds
two TLS certificates for two different services, and they are not derived from
one another: this one, which GNOME Remote Desktop presents on 3389, and the
control plane's own HTTPS identity on 34001, whose fingerprint
`api_auth_routes.py` publishes as `certificate_fingerprint`. A function here
called `certificate_fingerprint` would have made `grep certificate_fingerprint`
return both with nothing to tell them apart - one name, two meanings, believed
by different modules, which is VD-077's shape exactly. Renamed while it was
still four references old.

On the wire they were never ambiguous and still are not: the HTTPS one is a
flat `certificate_fingerprint`, this one is nested under `rdp.certificate`.
That field name stays as it is - `/api/v2` is a public interface and
`Administration.tsx` reads it.
"""

from __future__ import annotations

import ipaddress
import os
import re
import socket
import subprocess
import tempfile
from pathlib import Path
from typing import Callable, Dict, List, Tuple

try:
    import pwd
except ImportError:  # pragma: no cover - deployment target is Linux
    pwd = None


RDP_TLS_DIRECTORY = Path(
    "/var/lib/gnome-remote-desktop/.local/share/gnome-remote-desktop/certificates"
)
RDP_TLS_CERT = RDP_TLS_DIRECTORY / "vaelor-rdp.crt"
RDP_TLS_KEY = RDP_TLS_DIRECTORY / "vaelor-rdp.key"
RDP_TLS_DAYS = 825
RESOLVER_CONFIGURATION = Path("/etc/resolv.conf")
#: Long enough for grdctl to answer from its own D-Bus service, short enough
#: that a wedged one cannot dominate the status route the console polls.
FINGERPRINT_TIMEOUT_SECONDS = 3

_DNS_NAME = re.compile(
    r"[A-Za-z0-9](?:[A-Za-z0-9-]{0,61}[A-Za-z0-9])?"
    r"(?:\.[A-Za-z0-9](?:[A-Za-z0-9-]{0,61}[A-Za-z0-9])?)*"
)
#: `TLS fingerprint: AA:BB:...` as grdctl prints it. The label is captured
#: loosely because the tool's own wording is the only thing pinning it.
_FINGERPRINT = re.compile(r"TLS fingerprint:\s*([0-9A-Fa-f](?:[0-9A-Fa-f:]{15,}))")
#: A certificate's Common Name field cannot exceed 64 characters.
_COMMON_NAME_LIMIT = 64
_FALLBACK_COMMON_NAME = "Vaelor Remote Desktop"


def _routed_address() -> str:
    """The address this machine uses to reach its own network.

    A UDP socket is `connect`ed and never written to, so nothing leaves the
    host: the kernel picks a source address from its routing table and
    `getsockname` reports it. `ip addr` is the obvious tool and the one to
    avoid — it enumerates over `AF_NETLINK`, which a Vaelor unit's
    `RestrictAddressFamilies` need not list, and the empty stdout that
    produces is indistinguishable from a machine with no addresses (#84,
    LESSONS pattern 8).
    """
    probe = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    try:
        probe.connect(("192.0.2.1", 9))  # TEST-NET-1: documented as unrouted.
        return str(probe.getsockname()[0])
    except OSError:
        return ""
    finally:
        probe.close()


def _search_domains() -> List[str]:
    """The domains this host's resolver appends to a short name.

    **`myhost.lan` is a name the appliance answers to that `gethostname`
    never mentions.** It arrives with the DHCP lease as `search lan` in the
    resolver configuration, and it is a likely thing for an owner to type -
    so a certificate that covers `myhost` and not `myhost.lan` still warns.
    `getfqdn` finds it only when reverse DNS cooperates; this does not depend
    on that.
    """
    try:
        text = RESOLVER_CONFIGURATION.read_text(encoding="utf-8", errors="replace")
    except OSError:
        # A host with no resolver configuration has no search domains, which
        # is the same answer. Nothing downstream depends on this alone.
        return []
    domains: List[str] = []
    for line in text.splitlines():
        fields = line.split()
        if fields and fields[0] in {"search", "domain"}:
            domains.extend(fields[1:])
    return domains


def certificate_identities() -> Tuple[List[str], List[str]]:
    """Every name and address an owner could type to reach this machine.

    **A certificate with no `subjectAltName` cannot match anything.** Measured
    on the appliance 2026-08-11: `subject=CN=Vaelor Remote Desktop`, issuer
    identical, and no SAN extension present at all — on a machine answering to
    `myhost`, `myhost.lan` and `192.168.0.50`. Every client reported a name
    mismatch whatever the owner typed, because a bare Common Name is not
    consulted for identity: there was nothing to compare against, so the
    warning was guaranteed rather than incidental.

    Discovered here rather than written down, because these are *readings*
    and not facts (LESSONS pattern 9). A hostname change, a new DHCP lease or
    a second interface moves every one of them, and a value typed into this
    file would already be wrong on the next machine that ran it.

    Returns (dns names, ip addresses), each ordered most-specific first and
    de-duplicated. Either may be empty; the caller decides what that means.
    """
    names: List[str] = []
    addresses: List[str] = []

    def add_name(value: str) -> None:
        cleaned = str(value or "").strip().rstrip(".")
        if cleaned and _DNS_NAME.fullmatch(cleaned) and cleaned not in names:
            names.append(cleaned)

    def add_address(value: str) -> None:
        try:
            parsed = ipaddress.ip_address(str(value or "").split("%", 1)[0])
        except ValueError:
            return
        if str(parsed) not in addresses:
            addresses.append(str(parsed))

    hostname = socket.gethostname()
    qualified = socket.getfqdn(hostname) if hostname else socket.getfqdn()
    # Qualified first: it is what an owner on a network with DNS types, and
    # the Common Name is taken from the head of this list.
    add_name(qualified)
    for domain in _search_domains():
        add_name("{}.{}".format(hostname, domain.strip(".")))
    add_name(hostname)
    try:
        for info in socket.getaddrinfo(
            hostname or None, None, flags=socket.AI_CANONNAME
        ):
            add_name(info[3])
            add_address(info[4][0])
    except (OSError, UnicodeError, ValueError):
        # A machine whose own name does not resolve is ordinary, and the
        # routed address below does not depend on the resolver. The absence
        # is not reported because it is not evidence of anything — but it is
        # also not allowed to be the whole answer: `install_rdp_tls` refuses
        # to build a certificate that names nothing.
        pass
    add_address(_routed_address())
    # Loopback last, and included deliberately: Vaelor's own readiness probe
    # connects to 127.0.0.1:3389, and an owner reaching the appliance through
    # an SSH tunnel types localhost.
    add_name("localhost")
    add_address("127.0.0.1")
    return names, addresses


def _common_name(names: List[str], addresses: List[str]) -> str:
    for candidate in [*names, *addresses]:
        if len(candidate) <= _COMMON_NAME_LIMIT:
            return candidate
    return _FALLBACK_COMMON_NAME


def _subject_alt_name(names: List[str], addresses: List[str]) -> str:
    return ",".join(
        [*("DNS:{}".format(name) for name in names),
         *("IP:{}".format(address) for address in addresses)]
    )


def install_rdp_tls(run: Callable[..., str]) -> None:
    """Generate and publish the certificate GNOME Remote Desktop serves.

    **Three defects the `openssl req -x509` defaults produced, all measured on
    the appliance 2026-08-11.** No `subjectAltName`, so no client could ever
    match the name it was given. `basicConstraints critical CA:TRUE`, so a
    server certificate claimed to be a certificate authority. And no
    `keyUsage` or `extendedKeyUsage` at all, so nothing said the key was for
    serving TLS. The certificate remains self-signed, which is not fixable
    without an authority the appliance does not have and is not what produced
    the owner's warning; the missing SAN is.

    **This regenerates on every `configure_rdp`, deliberately.** An appliance
    commissioned before this change has a certificate that cannot match its
    own name, and rotating credentials is the one action an owner already
    takes when remote login is wrong — gating regeneration on some "is the old
    one good enough" test would be a second mechanism deciding the same
    question (LESSONS pattern 6), and the cheap five-second rebuild does not
    need one.

    **Nothing is published until it exists.** openssl writes into a private
    temporary directory and the pair is moved into place only after it has
    been generated, so a failure at any point before that — an openssl build
    without `-addext`, a full disk — leaves the working certificate exactly
    where it was, and `_stopped_for_reconfiguration` puts the service back up
    on it. Two files cannot be replaced in one atomic step, so the previous
    pair is held in memory across the only window that remains and restored if
    the second move fails; a certificate from one generation beside a key from
    another is a listener that starts and can never complete a handshake.

    **"A host that names itself nothing" used to be listed there and is not a
    case this function has** (#192). :func:`certificate_identities` appends
    ``localhost`` and ``127.0.0.1`` unconditionally, so the refusal below
    cannot fire on any machine; the only test of it patched that function out
    and returned ``([], [])``, which is a state production never creates.
    Written as current behaviour, it was design intent — LESSONS 10.
    """
    if pwd is None:
        raise RuntimeError("GNOME Remote Desktop setup requires Linux.")
    account = pwd.getpwnam("gnome-remote-desktop")
    names, addresses = certificate_identities()
    # Unreachable while `certificate_identities` guarantees the loopback pair,
    # and kept as a cheap check on that invariant rather than removed - a
    # certificate naming nothing is unusable and this is the last point that
    # could say so. It is deliberately NOT tightened to fire on a
    # loopback-only result: the readiness probe connects to 127.0.0.1:3389 and
    # an owner on an SSH tunnel types localhost, so refusing there would remove
    # working capability (LESSONS 4). `test_the_nameless_machine_refusal_cannot
    # _be_reached_from_production` pins the invariant instead of this branch.
    if not names and not addresses:
        raise RuntimeError(
            "This machine reported no host name and no network address, so no "
            "certificate could name it. Remote login was left unchanged."
        )
    common_name = _common_name(names, addresses)
    RDP_TLS_DIRECTORY.mkdir(mode=0o700, parents=True, exist_ok=True)
    os.chown(RDP_TLS_DIRECTORY, account.pw_uid, account.pw_gid)
    os.chmod(RDP_TLS_DIRECTORY, 0o700)
    temporary = Path(tempfile.mkdtemp(prefix=".rdp-tls-", dir=RDP_TLS_DIRECTORY))
    certificate = temporary / "certificate.pem"
    private_key = temporary / "private-key.pem"
    try:
        run([
            "/usr/bin/openssl", "req", "-x509", "-nodes", "-newkey", "rsa:3072",
            "-days", str(RDP_TLS_DAYS), "-subj", "/CN={}".format(common_name),
            "-addext", "subjectAltName={}".format(
                _subject_alt_name(names, addresses)
            ),
            "-addext", "basicConstraints=critical,CA:FALSE",
            "-addext", "keyUsage=critical,digitalSignature,keyEncipherment",
            "-addext", "extendedKeyUsage=serverAuth",
            "-keyout", str(private_key), "-out", str(certificate),
        ], timeout=30)
        os.chown(certificate, account.pw_uid, account.pw_gid)
        os.chown(private_key, account.pw_uid, account.pw_gid)
        os.chmod(certificate, 0o600)
        os.chmod(private_key, 0o600)
        previous = {
            path: path.read_bytes()
            for path in (RDP_TLS_CERT, RDP_TLS_KEY) if path.exists()
        }
        try:
            os.replace(certificate, RDP_TLS_CERT)
            os.replace(private_key, RDP_TLS_KEY)
        except OSError:
            for path, content in previous.items():
                path.write_bytes(content)
                os.chown(path, account.pw_uid, account.pw_gid)
                os.chmod(path, 0o600)
            raise
    finally:
        for item in temporary.iterdir():
            item.unlink(missing_ok=True)
        temporary.rmdir()


def rdp_certificate_fingerprint() -> Dict[str, str]:
    """What the owner should compare against before trusting the certificate.

    The appliance signs its own RDP certificate, so every client asks the
    owner to accept something they have no way to check. They do have a way:
    the machine can state the fingerprint of what it is serving, and the
    client's warning dialog shows the fingerprint of what it received. That
    turns "click Yes" into an actual comparison — which is the whole value,
    and the reason this is worth four lines of screen.

    **The digest name is derived from the digest, not assumed.** grdctl prints
    the value without saying which hash produced it, and a label asserting
    SHA-256 over a SHA-1 value is exactly LESSONS pattern 5. Sixty-four hex
    characters is SHA-256 and forty is SHA-1; anything else is reported with
    no algorithm rather than a guessed one.

    **`--show-credentials` is not passed and raw output never leaves here.**
    That flag prints the owner's RDP password, and `_verify_rdp_configuration`
    exists because output from this same command reached an HTTP error body
    once already. Only the matched fingerprint and grdctl's stderr are
    returned.
    """
    try:
        result = subprocess.run(
            ["/usr/bin/grdctl", "--system", "status"],
            capture_output=True,
            text=True,
            check=False,
            timeout=FINGERPRINT_TIMEOUT_SECONDS,
        )
    except (OSError, subprocess.SubprocessError) as error:
        return _no_fingerprint(
            "The certificate fingerprint could not be read: {}".format(
                str(error)[:160]
            )
        )
    if result.returncode != 0:
        # stderr, never stdout: the tool's diagnostics live there and the
        # credentials do not. Discarding it would make "grdctl could not
        # reach D-Bus" and "this machine has no certificate" arrive looking
        # the same (LESSONS pattern 8).
        detail = " ".join((result.stderr or "").split())[:200]
        return _no_fingerprint(
            "grdctl exited {} and did not report a fingerprint.{}".format(
                result.returncode, " " + detail if detail else ""
            )
        )
    match = _FINGERPRINT.search(result.stdout or "")
    if not match:
        return _no_fingerprint(
            "GNOME Remote Desktop reported no TLS fingerprint for this "
            "certificate."
        )
    value = match.group(1).strip().upper()
    digits = value.replace(":", "")
    return {
        "fingerprint": value,
        "algorithm": {64: "SHA-256", 40: "SHA-1"}.get(len(digits), ""),
        "detail": "",
    }


def _no_fingerprint(detail: str) -> Dict[str, str]:
    return {"fingerprint": "", "algorithm": "", "detail": detail}
