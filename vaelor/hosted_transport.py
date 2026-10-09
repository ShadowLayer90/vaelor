"""The one way Vaelor reaches a hosted AI service: checked, pinned HTTPS.

VD-201 / VD-206: AI Chat may connect to a hosted service at a public address,
over HTTPS only. That makes this module an SSRF boundary (LESSONS 18: a reused
allowlist is not a reused boundary), so it has its own rule rather than the
private-network rule `credential_provider_probe.validate_compatible_profile`
applies to "a server on your network" - and that rule, which every agent path
also uses, is deliberately left as it was.

**The rule** (`address_refusal`, `checked_address`):

1. The scheme is ``https``. Plain HTTP never leaves for a hosted service.
2. Every address the name resolves to is a public (global) address: loopback,
   private, shared (100.64/10), link-local, multicast, reserved, unspecified and
   documentation ranges are refused, and the cloud metadata addresses are
   refused by name. An IPv4 address carried inside IPv6 (mapped, 6to4, NAT64)
   is checked as the IPv4 address it carries; a Teredo tunnel address is
   refused outright.
3. If ANY address the name resolves to is refused, the name is refused, so a
   split-horizon name cannot choose its own path.
4. The connection is made to the address that was checked - resolved once,
   never again for that request - so a name that re-resolves between the check
   and the connect (DNS rebinding) reaches nothing new. TLS is still verified
   against the host name (SNI and the certificate), so pinning the address
   never weakens the certificate check.
5. A redirect is never followed: a 3xx is refused with its own sentence, so a
   public service cannot hand Vaelor on to a private address.

Nothing here logs a header, a body or a key. Every message is owner-readable
and names the host, never the key.

Runs in two processes: the credential broker (connection tests and model lists,
on a key it has just decrypted) and the control plane (AI Chat answers, on a
key leased from the broker). Both call `request`; neither has another path out.
"""

from __future__ import annotations

import base64
import http.client
import io
import ipaddress
import json
import re
import socket
import ssl
import time
import urllib.error
import urllib.parse
from typing import Any, Callable, Dict, Iterator, List, Mapping, Optional, Tuple

from .credential_broker_client import CredentialError

#: Cloud instance-metadata addresses, refused by name as well as by range, so
#: a future change to what Python calls "global" cannot let one through.
METADATA_ADDRESSES = frozenset(
    ipaddress.ip_address(item) for item in (
        "169.254.169.254",  # AWS, GCP, Azure, OpenStack, DigitalOcean
        "169.254.170.2",    # AWS ECS task metadata
        "fd00:ec2::254",    # AWS IMDS over IPv6
        "100.100.100.200",  # Alibaba Cloud
    )
)
#: Carrier-grade NAT space: not private in Python's sense, not public either.
_SHARED_ADDRESS_SPACE = ipaddress.ip_network("100.64.0.0/10")

#: The largest JSON body and the largest stream Vaelor reads from a service.
MAX_JSON_BYTES = 4 * 1024 * 1024
MAX_STREAM_BYTES = 16 * 1024 * 1024
#: How much of an error body is kept, to read the provider's error type.
MAX_ERROR_BYTES = 64 * 1024

USER_AGENT = "Vaelor-AI-Chat/1"
#: What a refusal calls a service it cannot name.
UNNAMED_SERVICE = "The service"

# The reasons an address is refused; each has one sentence in `_REFUSALS`.
REFUSED_METADATA = "metadata"
REFUSED_LINK_LOCAL = "link-local"
REFUSED_PRIVATE = "private"
REFUSED_NOT_PUBLIC = "not-public"

_REFUSALS = {
    REFUSED_METADATA: (
        "{host} is a cloud metadata address, which Vaelor never connects to."
    ),
    REFUSED_LINK_LOCAL: (
        "{host} resolves to a link-local address, which Vaelor never connects to."
    ),
    REFUSED_PRIVATE: (
        "{host} is on a private network or this machine. Add it as a server on "
        "your network instead; a hosted service must be on the internet."
    ),
    REFUSED_NOT_PUBLIC: (
        "{host} does not resolve to a public internet address, so Vaelor will "
        "not send a key to it."
    ),
}


class HostedAddressError(CredentialError):
    """A hosted address Vaelor refuses to connect to, in the owner's words."""


#: NAT64's well-known prefix (RFC 6052): the low 32 bits are an IPv4 address
#: the translator reaches. Python before the CVE-2024-4032 fix calls the whole
#: prefix global, so it is unwrapped here by hand, like the other carriers.
_NAT64_PREFIX = ipaddress.ip_network("64:ff9b::/96")


def _carried_ipv4(address: Any) -> Any:
    """The IPv4 address an IPv6 address carries, or the address itself."""
    if address.version == 6:
        for carried in (address.ipv4_mapped, address.sixtofour):
            if carried is not None:
                return carried
        # Teredo (2001::/32) is deliberately NOT unwrapped: Python counts the
        # whole tunnel prefix as private, so every Teredo address is refused,
        # where unwrapping would admit one carrying a public client address.
        if address in _NAT64_PREFIX:
            return ipaddress.IPv4Address(int(address) & 0xFFFFFFFF)
    return address


def address_refusal(address: Any) -> str:
    """Why ``address`` may not be reached for a hosted service, or ``""``.

    The order matters only for the sentence: a metadata or link-local address
    is named as such before the broader private and not-public reasons.
    """
    candidate = _carried_ipv4(ipaddress.ip_address(str(address)))
    if candidate in METADATA_ADDRESSES or ipaddress.ip_address(str(address)) in METADATA_ADDRESSES:
        return REFUSED_METADATA
    if candidate.is_link_local:
        return REFUSED_LINK_LOCAL
    if (
        candidate.is_loopback or candidate.is_private
        or (candidate.version == 4 and candidate in _SHARED_ADDRESS_SPACE)
    ):
        return REFUSED_PRIVATE
    if (
        candidate.is_multicast or candidate.is_unspecified
        or candidate.is_reserved or not candidate.is_global
    ):
        return REFUSED_NOT_PUBLIC
    return ""


def parse_hosted_url(url: str, *, allow_query: bool = False) -> urllib.parse.SplitResult:
    """Split a hosted URL, refusing anything but a plain ``https://host/path``.

    ``allow_query`` is for a request URL Vaelor built itself (``/models?limit=``)
    and nothing else: the address an owner types never carries a query.
    """
    text = str(url or "").strip()
    if len(text) > 2000 or (not allow_query and len(text) > 500):
        raise HostedAddressError("The service address is too long.")
    parsed = urllib.parse.urlsplit(text)
    if parsed.scheme != "https":
        raise HostedAddressError(
            "A hosted service must use HTTPS. Plain HTTP is accepted only for "
            "a server on your private network."
        )
    if not parsed.hostname or parsed.username or parsed.password:
        raise HostedAddressError(
            "Enter the service address as https://host/path, without a user name or password."
        )
    if (parsed.query and not allow_query) or parsed.fragment:
        raise HostedAddressError(
            "Enter the service address without query text or a # fragment."
        )
    try:
        port = parsed.port
    except ValueError:
        port = 0
    if port is not None and not 1 <= port <= 65535:
        raise HostedAddressError("The service address has an invalid port.")
    return parsed


Resolver = Callable[..., List[Tuple[Any, ...]]]


def checked_addresses(host: str, port: int, resolver: Optional[Resolver] = None) -> List[str]:
    """Resolve ``host`` once and return every address, refusing the name
    unless every one of them is public.

    ``resolver`` is `socket.getaddrinfo` in production; a test passes its own.
    """
    lookup = resolver or socket.getaddrinfo
    try:
        records = lookup(host, port, type=socket.SOCK_STREAM)
    except (OSError, UnicodeError) as error:
        raise HostedAddressError(
            "{} could not be resolved from this appliance. Check the address "
            "and this appliance's internet connection.".format(host)
        ) from error
    addresses: List[str] = []
    for record in records:
        value = str(record[4][0]).split("%", 1)[0]
        if value not in addresses:
            addresses.append(value)
    if not addresses:
        raise HostedAddressError("{} did not resolve to any address.".format(host))
    for value in addresses:
        try:
            reason = address_refusal(value)
        except ValueError:
            reason = REFUSED_NOT_PUBLIC
        if reason:
            raise HostedAddressError(_REFUSALS[reason].format(host=host))
    return addresses


def checked_address(host: str, port: int, resolver: Optional[Resolver] = None) -> str:
    """The first of `checked_addresses` (a profile check needs only the verdict)."""
    return checked_addresses(host, port, resolver)[0]


def tls_context() -> ssl.SSLContext:
    """The TLS settings every hosted connection uses: certificates verified,
    the host name checked. Named so a test can hold it to that."""
    return ssl.create_default_context()


class _PinnedHTTPSConnection(http.client.HTTPSConnection):
    """HTTPS to an address already checked, verified against the host name."""

    def __init__(self, host: str, port: int, address: str, timeout: float,
                 context: ssl.SSLContext):
        super().__init__(host, port, timeout=timeout, context=context)
        self._pinned_address = address

    def connect(self) -> None:
        raw = socket.create_connection((self._pinned_address, self.port), self.timeout)
        self.sock = self._context.wrap_socket(raw, server_hostname=self.host)


class HostedResponse:
    """One successful (2xx) response, read as JSON or as server-sent lines.

    Every read is held to one overall deadline, not only a per-read timeout: a
    service that drips a byte at a time would otherwise satisfy each read and
    hold a control-plane worker for as long as it liked.
    """

    def __init__(self, response: http.client.HTTPResponse, connection: Any,
                 deadline: float, sock: Any = None):
        self.status = response.status
        self.headers = response.headers
        self._response = response
        self._connection = connection
        self._deadline = deadline
        self._sock = sock

    def _chunks(self) -> Iterator[bytes]:
        while True:
            remaining = self._deadline - time.monotonic()
            if remaining <= 0:
                raise TimeoutError("The service did not finish its answer in time.")
            if self._sock is not None:
                try:
                    self._sock.settimeout(remaining)
                except OSError:
                    pass
            chunk = self._response.read1(65536)
            if not chunk:
                return
            yield chunk

    def json(self) -> Any:
        raw = b""
        for chunk in self._chunks():
            raw += chunk
            if len(raw) > MAX_JSON_BYTES:
                raise CredentialError("The service's response was too large.")
        try:
            return json.loads(raw.decode("utf-8"))
        except (UnicodeDecodeError, json.JSONDecodeError) as error:
            raise CredentialError("The service did not return valid JSON.") from error

    def lines(self) -> Iterator[str]:
        """Each line of the body, decoded, without its line ending."""
        total = 0
        pending = b""
        for chunk in self._chunks():
            total += len(chunk)
            if total > MAX_STREAM_BYTES:
                raise CredentialError("The service's answer stream was too large.")
            pending += chunk
            *complete, pending = pending.split(b"\n")
            for line in complete:
                yield line.decode("utf-8", errors="replace").rstrip("\r")
        if pending:
            yield pending.decode("utf-8", errors="replace").rstrip("\r")

    def close(self) -> None:
        try:
            self._response.close()
        finally:
            self._connection.close()

    def __enter__(self) -> "HostedResponse":
        return self

    def __exit__(self, *_exc: Any) -> None:
        self.close()


def exchange(
    connection: Any,
    method: str,
    url: str,
    *,
    headers: Mapping[str, str],
    body: Optional[bytes] = None,
    timeout: Optional[float] = None,
) -> HostedResponse:
    """Send one request over an open ``connection`` and classify the answer.

    The part of `request` that does not depend on how the socket was made, so
    a provider adapter can be driven against a local fake server through the
    same status handling production uses. A 3xx is refused (rule 5); any other
    non-2xx raises `urllib.error.HTTPError` carrying the start of the body, so
    a caller can read the provider's error type. ``timeout`` (default: the
    connection's own) is the whole exchange's deadline, reading included.
    """
    budget = timeout if timeout is not None else (getattr(connection, "timeout", None) or 30)
    deadline = time.monotonic() + float(budget)
    parsed = urllib.parse.urlsplit(url)
    target = parsed.path or "/"
    if parsed.query:
        target += "?" + parsed.query
    try:
        connection.request(method, target, body=body, headers=dict(headers))
        sock = getattr(connection, "sock", None)
        response = connection.getresponse()
    except BaseException:
        connection.close()
        raise
    if 300 <= response.status < 400:
        response.close()
        connection.close()
        raise HostedAddressError(
            "{} answered with a redirect (HTTP {}). Vaelor does not follow "
            "redirects from a hosted service; enter the address it redirects "
            "to if you trust it.".format(parsed.hostname, response.status)
        )
    if not 200 <= response.status < 300:
        try:
            detail = response.read(MAX_ERROR_BYTES)
        finally:
            response.close()
            connection.close()
        raise urllib.error.HTTPError(
            url, response.status, response.reason or "", response.headers,
            io.BytesIO(detail),
        )
    return HostedResponse(response, connection, deadline, sock)


def request(
    method: str,
    url: str,
    *,
    headers: Mapping[str, str],
    body: Optional[bytes] = None,
    timeout: float = 30,
    resolver: Optional[Resolver] = None,
) -> HostedResponse:
    """One request to a hosted service under the full rule above.

    The URL is one Vaelor built (a checked base address plus a path, and a
    query only for its own parameters). Every resolved address was checked, so
    each is tried in order until one connects; none is resolved twice.
    """
    parsed = parse_hosted_url(url, allow_query=True)
    port = parsed.port or 443
    failure: Optional[BaseException] = None
    for address in checked_addresses(parsed.hostname, port, resolver):
        connection = _PinnedHTTPSConnection(parsed.hostname, port, address, timeout, tls_context())
        try:
            connection.connect()
        except OSError as error:
            connection.close()
            failure = error
            continue
        return exchange(connection, method, url, headers=headers, body=body, timeout=timeout)
    raise failure if failure is not None else OSError("No address could be reached.")


def error_type_of(error: urllib.error.HTTPError) -> Tuple[str, str]:
    """The provider's own error type and message from an error body, if any.

    Anthropic answers ``{"type": "error", "error": {"type", "message"}}``; an
    OpenAI-compatible service ``{"error": {"type"|"code", "message"}}``; Google's
    OpenAI-compatible endpoint wraps that object in a one-item list. A body
    that is none of these answers ``("", "")``.
    """
    try:
        payload = json.loads(error.read(MAX_ERROR_BYTES).decode("utf-8"))
    except (OSError, ValueError, AttributeError):
        return "", ""
    if isinstance(payload, list) and payload and isinstance(payload[0], dict):
        payload = payload[0]
    detail = payload.get("error") if isinstance(payload, dict) else None
    if not isinstance(detail, dict):
        return "", ""
    kind = str(detail.get("type") or detail.get("status") or detail.get("code") or "")
    # The message is returned whole (the body read is already bounded by
    # MAX_ERROR_BYTES): it is redacted by the caller and only then clipped
    # (`clipped_reason`). Clipping first cut a key at the edge into a fragment
    # too short for `redact` to recognise (LESSONS 24, review R20).
    return kind[:60], str(detail.get("message") or "")


#: A run long enough to be a key or token, in any of the alphabets keys and
#: their encodings use. A provider's error text is redacted of every such run.
_TOKEN_RUN = r"[A-Za-z0-9_\-+/=.%~]{24,}"


def redact(text: str, secret: str) -> str:
    """``text`` with ``secret`` removed in every form a provider echoes it in.

    The key as given, URL-encoded, base64 (standard and URL-safe, with or
    without padding), a masked form that keeps its start or end, and any long
    token-like run at all - an error detail is never worth a leaked key.
    """
    clean = str(text or "")
    if secret and len(secret) >= 8:
        encoded = secret.encode("utf-8")
        forms = {
            secret,
            urllib.parse.quote(secret, safe=""),
            urllib.parse.quote_plus(secret),
            base64.b64encode(encoded).decode("ascii"),
            base64.urlsafe_b64encode(encoded).decode("ascii"),
        }
        forms |= {form.rstrip("=") for form in forms}
        for form in sorted(forms, key=len, reverse=True):
            if form:
                clean = clean.replace(form, "[redacted]")
        if len(secret) >= 16:
            for piece in (secret[:6], secret[-4:]):
                clean = clean.replace(piece, "[redacted]")
    return re.sub(_TOKEN_RUN, "[redacted]", clean)


# The owner's sentence for each kind of refusal a hosted service answers with.
# Keyed by status, with the provider's error type where it says more.
HOSTED_KEY_REJECTED = "hosted_key_rejected"
HOSTED_RATE_LIMITED = "hosted_rate_limited"
HOSTED_OVERLOADED = "hosted_overloaded"


#: A provider's own 429 reason that is about credits or billing, not a rate.
CREDIT_WORDS = ("credit", "billing", "prepay", "prepaid", "payment")

#: How much of a provider's own refusal reason is quoted after Vaelor's sentence.
PROVIDER_REASON_CHARS = 200


def clipped_reason(detail: str) -> str:
    """An ALREADY-REDACTED provider reason on one line, clipped to
    `PROVIDER_REASON_CHARS`. Redact first, clip second - never the reverse."""
    clean = " ".join(str(detail or "").split())
    if len(clean) > PROVIDER_REASON_CHARS:
        clean = clean[:PROVIDER_REASON_CHARS - 1].rstrip() + "…"
    return clean


def _provider_says(who: str, detail: str) -> str:
    """`` "<who> says: <reason>"`` from an already-redacted reason, or ``""``."""
    clean = clipped_reason(detail)
    return " {} says: {}".format(who, clean) if clean else ""


def refusal_for(company: str, status: int, kind: str = "", detail: str = "",
                model: str = "") -> Tuple[str, str]:
    """``(code, sentence)`` for a hosted service's error answer.

    ``detail`` is the provider's own message; it is included only for a 400
    (where it is the only useful fact) and must already be redacted.
    """
    who = company or UNNAMED_SERVICE
    # LESSONS 8: a billing, permission or not-found refusal is answered with
    # the provider's own reason after Vaelor's sentence - "billing reasons"
    # alone hid Google's actual one (a model not on the free tier) from the
    # owner. ``detail`` arrives redacted, and empty for a custom service.
    says = _provider_says(who, detail)
    if status == 401 or kind == "authentication_error" or (
        status == 400 and "api key" in detail.lower()
    ):
        return HOSTED_KEY_REJECTED, (
            "{} rejected the API key. Check the key in your {} account, then "
            "add the connection again.".format(who, company or "provider")
        )
    if status == 429 or kind == "rate_limit_error":
        # A 429 is a limit, never "billing" (LESSONS 8): Google answers a
        # quota or rate limit with 429 RESOURCE_EXHAUSTED, and depleted
        # Prepay credit with 402. Whether the limit is about credits is the
        # provider's own words to say, never the status's.
        if any(word in (detail or "").lower() for word in CREDIT_WORDS):
            return HOSTED_RATE_LIMITED, (
                "{} is limiting this API key on quota or prepaid credits (HTTP "
                "429). Check the account's plan and credits, then retry.{}".format(who, says)
            )
        return HOSTED_RATE_LIMITED, (
            "{} is rate limiting this API key (HTTP 429). Wait a minute, then "
            "retry.{}".format(who, says)
        )
    if status == 402 or kind == "billing_error":
        return "hosted_billing", (
            "{} refused the request for billing reasons. Check the account's "
            "credit or payment details.{}".format(who, says)
        )
    if status == 403 or kind == "permission_error":
        return "hosted_permission", (
            "{} says this API key does not have permission for that model or "
            "request.{}".format(who, says)
        )
    if status == 404 or kind == "not_found_error":
        return "hosted_not_found", (
            "{} does not offer {} to this key. Choose another model.{}".format(
                who, model or "that model", says)
        )
    if status == 413 or kind == "request_too_large":
        return "hosted_too_large", (
            "{} refused the request as too large. Ask with fewer attached "
            "sources.".format(who)
        )
    if status == 529 or kind == "overloaded_error":
        return HOSTED_OVERLOADED, (
            "{} is overloaded right now. Retry in a moment.".format(who)
        )
    if status >= 500:
        return "hosted_server_error", (
            "{} had a server error (HTTP {}). Retry in a moment.".format(who, status)
        )
    if status == 0:
        # An error event inside a stream that had already answered 200.
        return "hosted_stream_error", (
            "{} stopped the answer with an error ({}). Retry in a moment.".format(
                who, kind or "the service named no error type")
        )
    reason = clipped_reason(detail)
    suffix = ": {}".format(reason) if reason else "."
    return "hosted_request_refused", (
        "{} refused the request (HTTP {}){}".format(who, status, suffix)
    )


def bearer_headers(api_key: str, *, json_body: bool = False) -> Dict[str, str]:
    """The headers an OpenAI-compatible hosted service is sent."""
    headers = {"Accept": "application/json", "User-Agent": USER_AGENT}
    if api_key:
        headers["Authorization"] = "Bearer {}".format(api_key)
    if json_body:
        headers["Content-Type"] = "application/json"
    return headers
