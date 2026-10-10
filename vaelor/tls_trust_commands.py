"""How the owner trusts this Vaelor on their own devices: the one renderer (VD-212).

The installer prints this at the end of an install (``python -m
vaelor.tls_trust_commands --address A``) and the console's Trust this Vaelor
panel shows the same text from ``/security/trust/commands``. One function draws
both, so the fingerprint a terminal prints and the one the panel shows cannot
disagree (LESSONS 6: three readers once derived the controller's certificate
three ways).

Each command downloads the household root from the console's public route,
compares the SHA-256 of what arrived with the fingerprint embedded in the
command, and installs it only when they match: the download itself is not
trusted (the console's certificate is exactly what is not trusted yet), the
fingerprint is. A mismatch installs nothing and says so.

What the served certificate IS relative to the root (household, legacy,
custom...) is not decided here: :func:`vaelor.tls_authority_pki.classify` is the
one classifier, and the fingerprints come from its :func:`fingerprint`.
"""

from __future__ import annotations

import argparse
import re
import socket
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple

from cryptography import x509
from cryptography.hazmat.primitives.serialization import Encoding

from . import tls_paths
from .runtime_paths import env_value
from .tls_authority_pki import fingerprint
from .tls_paths import ROOT_ROUTE


def root_sources() -> Tuple[str, ...]:
    """Where the root's public certificate is read from, in order: the public
    copy the console serves (the same file ``authority_status`` reads), then the
    authority's own (readable by root only, so the installer can print before
    the public copy exists). Nothing else is read."""
    return (tls_paths.PUBLIC_ROOT, tls_paths.ROOT_CERT)


def public_root_sources() -> Tuple[str, ...]:
    """What the console reads and serves: the public copy only. The control
    plane cannot open the authority's 0700 folder, so trying it there would
    turn "not set up yet" into "could not be read"."""
    return (tls_paths.PUBLIC_ROOT,)

#: The name a trusted root carries in the stores that show one.
AUTHORITY_LABEL = "Vaelor household authority"

#: A console address: a host name, an IPv4 address or a bracketed IPv6
#: address, with an optional port. It is pasted into shell and PowerShell
#: commands, so nothing outside this shape is ever rendered (a Host header is
#: the caller's to choose).
_ADDRESS = re.compile(
    r"^(?:\[[0-9A-Fa-f:.]{2,45}\]|[A-Za-z0-9](?:[A-Za-z0-9.-]{0,251}[A-Za-z0-9])?)"
    r"(?::[0-9]{1,5})?$"
)


#: What every surface says when there is no root yet.
NOT_SET_UP = (
    "This Vaelor's household authority has not been set up yet. It is "
    "created by the vaelor-tls-authority service (python -m "
    "vaelor.tls_authority ensure).")


class TrustUnavailable(Exception):
    """The root cannot be shown; ``state`` says why (``not-set-up``/``unreadable``)."""

    def __init__(self, state: str, message: str):
        super().__init__(message)
        self.state = state


@dataclass(frozen=True)
class RootFacts:
    """The household root's public facts. Never carries a key."""

    certificate: x509.Certificate
    source: str

    @property
    def der(self) -> bytes:
        return self.certificate.public_bytes(Encoding.DER)

    @property
    def pem(self) -> bytes:
        return self.certificate.public_bytes(Encoding.PEM)

    @property
    def sha256(self) -> str:
        """``AB:12:...``, as every certificate dialog and phone shows it."""
        return fingerprint(self.certificate, "sha256")

    @property
    def sha1(self) -> str:
        return fingerprint(self.certificate, "sha1")

    @property
    def sha256_plain(self) -> str:
        """The form the commands compare against: what the tools print."""
        return plain_hex(self.sha256)

    @property
    def short_id(self) -> str:
        """The file name suffix on the owner's devices: the fingerprint's first 16 hex."""
        return self.sha256_plain[:16]


def plain_hex(colon_fingerprint: str) -> str:
    """The ONE normaliser from the displayed form to what each OS command
    produces: ``sha256sum``/``shasum`` print lower case without colons, and
    PowerShell's ``-eq`` compares without case, so lower case serves all three."""
    return colon_fingerprint.replace(":", "").lower()


def read_root(sources: Optional[Tuple[str, ...]] = None) -> RootFacts:
    """The root from the first source that exists, re-encoded from its parse.

    Serving the parse rather than the file bytes means nothing else a file
    might hold (a second block, a stray key) can ever leave this function.
    A file that exists but cannot be read or parsed is ``unreadable``, never
    ``not-set-up``: "Vaelor could not look" is not "there is nothing"
    (LESSONS 8).
    """
    for source in sources if sources is not None else root_sources():
        path = Path(source)
        try:
            if not path.is_file():
                continue
            data = path.read_bytes()
        except OSError as error:
            raise TrustUnavailable(
                "unreadable",
                "The household authority's certificate at {} could not be read ({}).".format(
                    source, error.strerror or type(error).__name__),
            ) from None
        try:
            certificate = (x509.load_der_x509_certificate(data)
                           if not data.lstrip().startswith(b"-----")
                           else x509.load_pem_x509_certificate(data))
        except ValueError:
            raise TrustUnavailable(
                "unreadable",
                "The file at {} is not a certificate Vaelor can read.".format(source),
            ) from None
        return RootFacts(certificate=certificate, source=source)
    raise TrustUnavailable("not-set-up", NOT_SET_UP)


def valid_address(address: str) -> bool:
    # fullmatch, not match: `$` also matches before a trailing newline, and this
    # is the boundary in front of a shell and a PowerShell command.
    return bool(address) and len(address) <= 260 and bool(_ADDRESS.fullmatch(address))


def root_url(address: str, form: str = "der") -> str:
    return "https://{}{}?form={}".format(address, ROOT_ROUTE, form)


#: Windows PowerShell 5.1 has no -SkipCertificateCheck, and a script-block
#: callback fails there ("no Runspace available": HttpWebRequest calls it on
#: another thread, found by running this line), so 5.1 compiles a one-method
#: accept-all delegate, sets it for this one download, and restores the
#: previous callback in finally. The fingerprint check is what decides trust,
#: and like the macOS and Linux lines it hashes the downloaded FILE's bytes -
#: one rule for all three (LESSONS 6), so the file checked is the file imported.
_WINDOWS = (
    "$u='{url}';$f='{sha}';$p=Join-Path $env:TEMP 'vaelor-household-root.cer';"
    "try{{if($PSVersionTable.PSVersion.Major -ge 6){{"
    "Invoke-WebRequest -Uri $u -OutFile $p -UseBasicParsing -SkipCertificateCheck}}"
    "else{{$sp=[Net.ServicePointManager]::SecurityProtocol;"
    "$cb=[Net.ServicePointManager]::ServerCertificateValidationCallback;"
    "[Net.ServicePointManager]::SecurityProtocol=$sp -bor 3072;"
    "if(-not('VaelorAcceptOnce' -as [type])){{Add-Type 'public static class VaelorAcceptOnce{{"
    "public static bool Accept(object s,System.Security.Cryptography.X509Certificates.X509Certificate c,"
    "System.Security.Cryptography.X509Certificates.X509Chain h,System.Net.Security.SslPolicyErrors e)"
    "{{return true;}}}}'}};"
    "[Net.ServicePointManager]::ServerCertificateValidationCallback=[Delegate]::CreateDelegate("
    "[Net.Security.RemoteCertificateValidationCallback],[VaelorAcceptOnce].GetMethod('Accept'));"
    "try{{Invoke-WebRequest -Uri $u -OutFile $p -UseBasicParsing}}"
    "finally{{[Net.ServicePointManager]::ServerCertificateValidationCallback=$cb;"
    "[Net.ServicePointManager]::SecurityProtocol=$sp}}}};"
    "$h=-join([Security.Cryptography.SHA256]::Create().ComputeHash([IO.File]::ReadAllBytes($p))|ForEach-Object{{$_.ToString('x2')}});"
    "if($h -eq $f){{Import-Certificate -FilePath $p -CertStoreLocation Cert:\\CurrentUser\\Root|Out-Null;"
    "'Trusted: {label}'}}"
    "else{{\"Fingerprint mismatch: nothing was installed. Expected $f, received $h.\"}}}}"
    "finally{{Remove-Item -LiteralPath $p -ErrorAction SilentlyContinue}}"
)

_FETCH = (
    "f=\"$(mktemp)\"; if curl -fsSk '{url}' -o \"$f\"; then "
    "if [ \"$({hash} \"$f\" | cut -d' ' -f1)\" = '{sha}' ]; then {install}; "
    "else echo 'Fingerprint mismatch: nothing was installed.'; fi; "
    "else echo 'Could not download the certificate from {address}.'; fi; rm -f \"$f\""
)

_MACOS_INSTALL = (
    "security add-trusted-cert -r trustRoot -k \"$HOME/Library/Keychains/login.keychain-db\" \"$f\""
    " && echo 'Trusted: {label}'"
)

_LINUX_INSTALL = (
    "n='vaelor-household-{short}.crt'; "
    "if command -v update-ca-certificates >/dev/null 2>&1; then "
    "d=/usr/local/share/ca-certificates; u=update-ca-certificates; "
    "else d=/etc/pki/ca-trust/source/anchors; u=update-ca-trust; fi; "
    "{{ echo '-----BEGIN CERTIFICATE-----'; base64 -w 64 \"$f\"; echo '-----END CERTIFICATE-----'; }}"
    " | sudo tee \"$d/$n\" >/dev/null && sudo \"$u\" && echo 'Trusted system-wide: {label}'; "
    "if command -v certutil >/dev/null 2>&1 && [ -d \"$HOME/.pki/nssdb\" ]; then "
    "certutil -d \"sql:$HOME/.pki/nssdb\" -A -t 'C,,' -n '{label} {short}' -i \"$f\""
    " && echo 'Trusted in Chrome'; fi"
)


def render(root: RootFacts, address: str) -> Dict[str, object]:
    """Every command and fact for ``address``; the API returns it, the CLI prints it."""
    if not valid_address(address):
        raise ValueError("not a console address: {!r}".format(address[:80]))
    url = root_url(address)
    values = {"url": url, "sha": root.sha256_plain, "label": AUTHORITY_LABEL,
              "short": root.short_id, "address": address}
    commands: List[Dict[str, str]] = [
        {"os": "windows", "label": "Windows",
         "where": "PowerShell (Windows PowerShell 5.1 or PowerShell 7); adds it to your own account's trusted roots",
         "command": _WINDOWS.format(**values)},
        {"os": "macos", "label": "macOS",
         "where": "Terminal; adds it to your login keychain (macOS asks for your password)",
         "command": _FETCH.format(hash="shasum -a 256", install=_MACOS_INSTALL.format(**values), **values)},
        {"os": "linux", "label": "Linux",
         "where": "Terminal; the system trust store (asks for sudo), and Chrome's store when certutil is installed",
         "command": _FETCH.format(hash="sha256sum", install=_LINUX_INSTALL.format(**values), **values)},
    ]
    return {
        "address": address,
        "root_url": url,
        "fingerprint_sha256": root.sha256,
        "fingerprint_sha1": root.sha1,
        "commands": commands,
        "firefox": (
            "Firefox: if it still warns after the command above, open Settings, "
            "Privacy & Security, View Certificates, Authorities, Import, choose the "
            "certificate from {} and tick \"Trust this CA to identify websites\".".format(url)
        ),
    }


def render_text(rendered: Dict[str, object]) -> str:
    """The terminal form the installer prints."""
    lines = [
        "Trust this Vaelor on your own devices",
        "",
        "Household authority fingerprint (check it matches before trusting):",
        "  SHA-256  " + str(rendered["fingerprint_sha256"]),
        "  SHA-1    " + str(rendered["fingerprint_sha1"]),
        "",
        "Each command downloads the authority from {} and installs it only if".format(rendered["root_url"]),
        "its SHA-256 fingerprint matches the one above. A mismatch installs nothing.",
    ]
    for entry in rendered["commands"]:  # type: ignore[union-attr]
        lines += ["", "{} - {}:".format(entry["label"], entry["where"]), "  " + entry["command"]]
    lines += [
        "", str(rendered["firefox"]), "",
        "Phones: open Settings > Connections > Trust this Vaelor in the console and scan the code.",
    ]
    return "\n".join(lines) + "\n"


def _default_address() -> str:
    port = env_value("VAELOR_PORT", "PM_DASHBOARD_PORT", "34001")
    return "{}:{}".format(socket.gethostname(), port)


def main(argv: Optional[List[str]] = None) -> int:
    parser = argparse.ArgumentParser(
        prog="python -m vaelor.tls_trust_commands",
        description="Print the household authority's fingerprint and one trust command per OS.",
    )
    parser.add_argument("--address", default="",
                        help="the console address owners reach, e.g. 192.0.2.10:34001 "
                             "(default: this host's name and the console port)")
    arguments = parser.parse_args(argv)
    address = arguments.address.strip() or _default_address()
    if not valid_address(address):
        print("Not a console address: {!r}. Give a host name or IP address, "
              "optionally with :port.".format(address), file=sys.stderr)
        return 2
    try:
        root = read_root()
    except TrustUnavailable as unavailable:
        print(str(unavailable), file=sys.stderr)
        return 1
    sys.stdout.write(render_text(render(root, address)))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
