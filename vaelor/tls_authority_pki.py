"""Certificates for the household authority: building, reading, classifying (VD-212).

Pure functions over the ``cryptography`` library: nothing here touches a file,
a process or the clock except through its arguments. The root is created with
the private-LAN name constraints from :mod:`vaelor.tls_identity`, and every
leaf is filtered against the constraints the root *actually carries* (read back
from the certificate), so an imported root is held to its own limits rather
than to this module's defaults.
"""

from __future__ import annotations

import datetime
import hashlib
import ipaddress
import secrets
from typing import Callable, List, Sequence, Tuple

from cryptography import x509
from cryptography.exceptions import InvalidSignature
from cryptography.hazmat.primitives import hashes, serialization
from cryptography.hazmat.primitives.asymmetric import ec, rsa
from cryptography.x509.oid import ExtendedKeyUsageOID, NameOID

from . import tls_public_names
from .tls_identity import PRIVATE_DNS_SUFFIXES, PRIVATE_LAN_NETWORKS

ROOT_VALIDITY_DAYS = 3650
LEAF_VALIDITY_DAYS = 365
#: A leaf with less than this left is re-issued.
RENEW_BEFORE_DAYS = 30
ROOT_NAME_PREFIX = "Vaelor household authority "
LEAF_COMMON_NAME = "Vaelor console"
_BACKDATE = datetime.timedelta(hours=1)

#: How the served certificate relates to the root. ``previous-household`` is a
#: leaf from a household root this machine no longer holds (after an import
#: with ``--replace``); it migrates exactly like the legacy self-signed one.
KIND_HOUSEHOLD = "household"
KIND_PREVIOUS_HOUSEHOLD = "previous-household"
KIND_LEGACY = "legacy-self-signed"
KIND_CUSTOM = "custom"
KIND_NONE = "none"
#: The served file exists but cannot be read or parsed. Reported, never
#: replaced: it may be the owner's own certificate in a form not read here.
KIND_UNREADABLE = "unreadable"
MIGRATING_KINDS = (KIND_LEGACY, KIND_PREVIOUS_HOUSEHOLD)

KeyFactory = Callable[[], object]


def new_root_key():
    return ec.generate_private_key(ec.SECP256R1())


def new_leaf_key():
    return rsa.generate_private_key(public_exponent=65537, key_size=3072)


def root_hostname_label(hostname_label: str) -> Tuple[str, str]:
    """(label the root may carry, why it was left out). A public TLD, or any
    label while the TLD list is unavailable, is left out (VD-212 review F1)."""
    if not hostname_label or hostname_label in PRIVATE_DNS_SUFFIXES:
        return "", ""
    refusal = tls_public_names.bare_label_refusal(hostname_label)
    return ("", refusal) if refusal else (hostname_label, "")


def _name_constraints(hostname_label: str) -> x509.NameConstraints:
    dns = list(PRIVATE_DNS_SUFFIXES)
    label, _ = root_hostname_label(hostname_label)
    if label:
        dns.append(label)
    permitted: List[x509.GeneralName] = [x509.DNSName(name) for name in dns]
    permitted += [x509.IPAddress(network) for network in PRIVATE_LAN_NETWORKS]
    return x509.NameConstraints(permitted_subtrees=permitted, excluded_subtrees=None)


def new_root(hostname_label: str, now: datetime.datetime,
             key_factory: KeyFactory = new_root_key):
    """A ten-year root, constrained to the private LAN and this machine's label."""
    key = key_factory()
    name = x509.Name([
        x509.NameAttribute(NameOID.COMMON_NAME,
                           ROOT_NAME_PREFIX + secrets.token_hex(4)),
    ])
    public = key.public_key()
    cert = (
        x509.CertificateBuilder()
        .subject_name(name).issuer_name(name)
        .public_key(public)
        .serial_number(x509.random_serial_number())
        .not_valid_before(now - _BACKDATE)
        .not_valid_after(now + datetime.timedelta(days=ROOT_VALIDITY_DAYS))
        .add_extension(x509.BasicConstraints(ca=True, path_length=0), critical=True)
        .add_extension(x509.KeyUsage(
            digital_signature=False, content_commitment=False,
            key_encipherment=False, data_encipherment=False, key_agreement=False,
            key_cert_sign=True, crl_sign=True, encipher_only=False,
            decipher_only=False), critical=True)
        .add_extension(x509.SubjectKeyIdentifier.from_public_key(public),
                       critical=False)
        .add_extension(_name_constraints(hostname_label), critical=True)
        .sign(key, hashes.SHA256())
    )
    return cert, key


def name_constraints_critical(root: x509.Certificate) -> bool:
    try:
        return root.extensions.get_extension_for_class(x509.NameConstraints).critical
    except x509.ExtensionNotFound:
        return False


def permitted_by_root(root: x509.Certificate) -> Tuple[List[str], list]:
    """The DNS subtrees and IP networks the root's own name constraints permit."""
    try:
        constraints = root.extensions.get_extension_for_class(x509.NameConstraints).value
    except x509.ExtensionNotFound:
        return [], []
    dns, networks = [], []
    for general in constraints.permitted_subtrees or []:
        if isinstance(general, x509.DNSName):
            dns.append(general.value)
        elif isinstance(general, x509.IPAddress):
            networks.append(general.value)
    return dns, networks


def issue_leaf(root: x509.Certificate, root_key, names: Sequence[str],
               addresses: Sequence[str], now: datetime.datetime,
               key_factory: KeyFactory = new_leaf_key):
    """The console's serving certificate, signed by the root (365 days)."""
    if not names and not addresses:
        raise ValueError("a console certificate must name at least one name or address")
    key = key_factory()
    public = key.public_key()
    alt = [x509.DNSName(n) for n in names]
    alt += [x509.IPAddress(ipaddress.ip_address(a)) for a in addresses]
    try:
        authority_key_id = x509.AuthorityKeyIdentifier.from_issuer_subject_key_identifier(
            root.extensions.get_extension_for_class(x509.SubjectKeyIdentifier).value)
    except x509.ExtensionNotFound:  # an imported root may carry no SKI
        authority_key_id = x509.AuthorityKeyIdentifier.from_issuer_public_key(
            root.public_key())
    cert = (
        x509.CertificateBuilder()
        .subject_name(x509.Name([
            x509.NameAttribute(NameOID.COMMON_NAME, LEAF_COMMON_NAME)]))
        .issuer_name(root.subject)
        .public_key(public)
        .serial_number(x509.random_serial_number())
        .not_valid_before(now - _BACKDATE)
        .not_valid_after(now + datetime.timedelta(days=LEAF_VALIDITY_DAYS))
        .add_extension(x509.BasicConstraints(ca=False, path_length=None), critical=True)
        .add_extension(x509.KeyUsage(
            digital_signature=True, content_commitment=False,
            key_encipherment=True, data_encipherment=False, key_agreement=False,
            key_cert_sign=False, crl_sign=False, encipher_only=False,
            decipher_only=False), critical=True)
        .add_extension(x509.ExtendedKeyUsage([ExtendedKeyUsageOID.SERVER_AUTH]),
                       critical=False)
        .add_extension(x509.SubjectAlternativeName(alt), critical=False)
        .add_extension(x509.SubjectKeyIdentifier.from_public_key(public),
                       critical=False)
        .add_extension(authority_key_id, critical=False)
        .sign(root_key, hashes.SHA256())
    )
    return cert, key


def cert_pem(cert: x509.Certificate) -> bytes:
    return cert.public_bytes(serialization.Encoding.PEM)


def key_pem(key) -> bytes:
    """Unencrypted PKCS#8: only for the served leaf key and the sealed root."""
    return key.private_bytes(
        serialization.Encoding.PEM, serialization.PrivateFormat.PKCS8,
        serialization.NoEncryption())


def load_pem_certificates(data: bytes) -> List[x509.Certificate]:
    return x509.load_pem_x509_certificates(data)


def key_matches(cert: x509.Certificate, key) -> bool:
    """True when ``key`` is the private half of ``cert``'s public key."""
    spki = serialization.PublicFormat.SubjectPublicKeyInfo
    der = serialization.Encoding.DER
    try:
        return key.public_key().public_bytes(der, spki) ==             cert.public_key().public_bytes(der, spki)
    except (AttributeError, ValueError, TypeError):
        return False


def fingerprint(cert: x509.Certificate, algorithm: str = "sha256") -> str:
    """Colon-separated upper-case hex of the DER, as every OS dialog shows it."""
    digest = hashlib.new(algorithm, cert.public_bytes(serialization.Encoding.DER))
    return ":".join("{:02X}".format(b) for b in digest.digest())


def root_id(cert: x509.Certificate) -> str:
    """A short stable id for file names: the first 16 hex of the SHA-256."""
    return hashlib.sha256(
        cert.public_bytes(serialization.Encoding.DER)).hexdigest()[:16]


def directly_issued_by(cert: x509.Certificate, issuer: x509.Certificate) -> bool:
    try:
        cert.verify_directly_issued_by(issuer)
    except (ValueError, TypeError, InvalidSignature):
        return False
    return True


def is_household_root(cert: x509.Certificate) -> bool:
    common = cert.subject.get_attributes_for_oid(NameOID.COMMON_NAME)
    return bool(common) and str(common[0].value).startswith(ROOT_NAME_PREFIX)


def is_certificate_authority(cert: x509.Certificate) -> bool:
    try:
        return bool(cert.extensions.get_extension_for_class(
            x509.BasicConstraints).value.ca)
    except x509.ExtensionNotFound:
        return False


def leaf_identities(cert: x509.Certificate) -> Tuple[Tuple[str, ...], Tuple[str, ...]]:
    """(dns names, ip addresses) the certificate's subjectAltName carries, sorted."""
    try:
        alt = cert.extensions.get_extension_for_class(x509.SubjectAlternativeName).value
    except x509.ExtensionNotFound:
        return (), ()
    names = tuple(sorted(n.lower() for n in alt.get_values_for_type(x509.DNSName)))
    addresses = tuple(str(a) for a in sorted(
        alt.get_values_for_type(x509.IPAddress), key=lambda a: (a.version, a)))
    return names, addresses


def classify(served: x509.Certificate | None, root: x509.Certificate | None,
             hostnames: Sequence[str] = ()) -> str:
    """What kind of certificate the console serves, relative to the root.

    ``legacy-self-signed`` is the shape the pre-VD-212 installer wrote:
    self-signed and naming ``localhost`` and ``127.0.0.1`` (or, before that
    installer learned SANs, self-signed with no SAN and the hostname as CN).
    Anything else this authority did not issue is the owner's own ``custom``
    certificate and is never replaced.
    """
    if served is None:
        return KIND_NONE
    if root is not None and directly_issued_by(served, root):
        return KIND_HOUSEHOLD
    if served.issuer == served.subject and directly_issued_by(served, served):
        names, addresses = leaf_identities(served)
        if "localhost" in names and "127.0.0.1" in addresses:
            return KIND_LEGACY
        common = served.subject.get_attributes_for_oid(NameOID.COMMON_NAME)
        if not names and not addresses and common and str(
                common[0].value).lower() in {h.lower() for h in hostnames if h}:
            return KIND_LEGACY
        return KIND_CUSTOM
    common = served.issuer.get_attributes_for_oid(NameOID.COMMON_NAME)
    if common and str(common[0].value).startswith(ROOT_NAME_PREFIX):
        return KIND_PREVIOUS_HOUSEHOLD
    return KIND_CUSTOM


def issued_at(cert: x509.Certificate) -> float:
    """When the authority issued ``cert`` (its notBefore is backdated an hour)."""
    return cert.not_valid_before_utc.timestamp() + _BACKDATE.total_seconds()


def expires_within(cert: x509.Certificate, now: datetime.datetime,
                   days: int = RENEW_BEFORE_DAYS) -> bool:
    return cert.not_valid_after_utc - now < datetime.timedelta(days=days)
