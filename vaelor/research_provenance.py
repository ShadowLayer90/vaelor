"""Per-hop provenance policy for guarded public research fetches.

Reachability ("is this address public?") and provenance ("was this page
authorized for this run?") are different questions, and the research broker
only ever answered the first one on a redirect.  A vetted search result could
therefore bounce the fetch to any public host, and the audit trail still named
the vetted URL as the source.

This module owns the second question.  It is deliberately transport-free and
side-effect-free: it decides, it never connects, so the same rule can be
enforced before a hop is dialled *and* re-checked against the redirect chain a
broker reports afterwards.
"""

from __future__ import annotations

import inspect
import ipaddress
from dataclasses import dataclass
from typing import Any, Callable, Iterable, Mapping
from urllib.parse import urlsplit, urlunsplit


MAX_FETCH_HOPS = 3
"""Authorized destinations per fetch, counting the requested URL."""

# Not the Public Suffix List.  Vaelor ships no offline PSL copy and refuses to
# fetch one at runtime, so `registrable_domain` uses "last two labels" plus the
# explicit exceptions below.  Two consequences, stated honestly:
#   * A multi-label suffix that is missing here (say "co.zz") makes two
#     unrelated sites under it look like one site, so a redirect between them
#     would be allowed.  Adding entries tightens the rule; it never loosens it.
#   * Nothing here is load-bearing for reachability.  Every hop is still proved
#     public, HTTPS, and non-rebinding by the broker regardless of this table.
_PUBLIC_SUFFIXES = frozenset({
    # Country-code registries that sell at the third level.
    "ac.uk", "co.uk", "gov.uk", "ltd.uk", "me.uk", "net.uk", "org.uk", "plc.uk",
    "co.jp", "ne.jp", "or.jp", "ac.jp", "go.jp", "com.au", "net.au", "org.au",
    "edu.au", "gov.au", "co.nz", "net.nz", "org.nz", "govt.nz", "co.za",
    "org.za", "com.br", "net.br", "org.br", "gov.br", "com.cn", "net.cn",
    "org.cn", "gov.cn", "com.hk", "org.hk", "com.sg", "com.tw", "com.tr",
    "com.mx", "com.ar", "com.co", "co.in", "net.in", "org.in", "co.kr",
    "or.kr", "co.il", "com.pl", "com.ua", "com.vn", "co.th", "com.my",
    # Shared hosting namespaces where each tenant is an independent site.
    "github.io", "gitlab.io", "githubusercontent.com", "blogspot.com",
    "wordpress.com", "herokuapp.com", "azurewebsites.net", "appspot.com",
    "firebaseapp.com", "web.app", "pages.dev", "workers.dev", "vercel.app",
    "netlify.app", "netlify.com", "surge.sh", "glitch.me", "readthedocs.io",
    "cloudfront.net", "s3.amazonaws.com", "amazonaws.com", "notion.site",
    "translate.goog", "sourceforge.io", "gitbook.io", "fly.dev", "onrender.com",
})


def canonical_url(value: Any) -> str:
    """Return one comparable HTTPS URL, or "" when the URL is not usable.

    Two URLs are treated as the same page only when this returns the same
    string for both, so the comparison stays exact - never a prefix or
    substring test.  Normalization is limited to the parts that carry no
    meaning: scheme case, host case, IDNA form, one trailing host dot, and an
    empty path.  Credentials, fragments, non-HTTPS schemes and off-443 ports
    are refused outright rather than normalized away.
    """
    text = str(value or "").strip()
    if not text or len(text) > 4096:
        return ""
    if any(ord(character) < 32 or ord(character) == 127 for character in text):
        return ""
    try:
        parsed = urlsplit(text)
        port = parsed.port
        host = (parsed.hostname or "").rstrip(".")
    except (UnicodeError, ValueError):
        return ""
    if parsed.scheme.lower() != "https" or not host:
        return ""
    if parsed.username is not None or parsed.password is not None or parsed.fragment:
        return ""
    if port not in (None, 443):
        return ""
    try:
        host = host.encode("idna").decode("ascii").lower()
    except UnicodeError:
        host = host.lower()
        if any(ord(character) > 127 for character in host):
            return ""
    netloc = "[{}]".format(host) if ":" in host else host
    return urlunsplit(("https", netloc, parsed.path or "/", parsed.query, ""))


def url_host(value: Any) -> str:
    """Return the canonical host of a URL, or "" when there is not one."""
    canonical = canonical_url(value)
    if not canonical:
        return ""
    return (urlsplit(canonical).hostname or "").lower()


def registrable_domain(host: str) -> str:
    """Best-effort eTLD+1 for a host name, or "" when there is not one.

    See `_PUBLIC_SUFFIXES` for the accuracy limits.  Address literals and bare
    suffixes return "", which callers must treat as "never the same site".
    """
    name = str(host or "").strip(".").lower()
    if not name:
        return ""
    try:
        ipaddress.ip_address(name.strip("[]"))
        return ""
    except ValueError:
        pass
    labels = name.split(".")
    if len(labels) < 2 or not all(labels):
        return ""
    for size in (3, 2):
        suffix = ".".join(labels[-size:])
        if suffix in _PUBLIC_SUFFIXES:
            return ".".join(labels[-(size + 1):]) if len(labels) > size else ""
    return ".".join(labels[-2:])


def same_site(host: str, other: str) -> bool:
    """True when two hosts are the same host or share a registrable domain."""
    left, right = str(host or "").lower(), str(other or "").lower()
    if not left or not right:
        return False
    if left == right:
        return True
    domain = registrable_domain(left)
    return bool(domain) and domain == registrable_domain(right)


def _describe(value: str) -> str:
    """Quote a remote-controlled host safely inside a failure message."""
    safe = "".join(
        character if character.isalnum() or character in ".-:[]_" else "?"
        for character in str(value or "")[:100]
    )
    return safe or "an unnamed host"


@dataclass(frozen=True)
class FetchProvenance:
    """Which pages this run may read, independent of whether they are public."""

    allowed_domains: tuple[str, ...] = ()
    search_result_urls: tuple[str, ...] = ()
    max_hops: int = MAX_FETCH_HOPS

    @classmethod
    def build(
        cls,
        allowed_domains: Iterable[Any] = (),
        search_result_urls: Iterable[Any] = (),
        max_hops: int = MAX_FETCH_HOPS,
    ) -> "FetchProvenance":
        domains = tuple(
            str(item).strip().lower().rstrip(".")
            for item in list(allowed_domains)[:20]
            if str(item).strip()
        )
        vetted = tuple(
            dict.fromkeys(
                canonical
                for canonical in (
                    canonical_url(item) for item in list(search_result_urls)[:20]
                )
                if canonical
            )
        )
        return cls(domains, vetted, max(1, min(int(max_hops), MAX_FETCH_HOPS)))

    @classmethod
    def from_dict(cls, value: Mapping[str, Any] | None) -> "FetchProvenance | None":
        """Rebuild a policy that crossed the research broker's socket."""
        if value is None:
            return None
        if not isinstance(value, Mapping) or set(value) - {
            "allowed_domains", "search_result_urls", "max_hops"
        }:
            raise ValueError("The research provenance policy is invalid.")
        domains = value.get("allowed_domains", [])
        vetted = value.get("search_result_urls", [])
        hops = value.get("max_hops", MAX_FETCH_HOPS)
        if not isinstance(domains, list) or not isinstance(vetted, list):
            raise ValueError("The research provenance policy is invalid.")
        if isinstance(hops, bool) or not isinstance(hops, int):
            raise ValueError("The research provenance policy is invalid.")
        if not all(isinstance(item, str) and len(item) <= 2000 for item in domains + vetted):
            raise ValueError("The research provenance policy is invalid.")
        return cls.build(domains, vetted, hops)

    def to_dict(self) -> dict[str, Any]:
        return {
            "allowed_domains": list(self.allowed_domains),
            "search_result_urls": list(self.search_result_urls),
            "max_hops": self.max_hops,
        }

    def allows(self, url: Any, *, origin: str = "") -> bool:
        """True when `url` is authorized, given the hop that redirected to it."""
        host = url_host(url)
        if not host:
            return False
        if any(
            host == domain or host.endswith("." + domain)
            for domain in self.allowed_domains
        ):
            return True
        if canonical_url(url) in self.search_result_urls:
            return True
        return bool(origin) and same_site(host, url_host(origin))

    def refusal(self, url: Any, *, origin: str = "") -> str:
        """Return "" when the hop is authorized, else a user-facing reason."""
        canonical = canonical_url(url)
        if not canonical:
            return (
                "redirect_denied: the destination is not a public HTTPS URL without "
                "credentials, fragments, or a non-standard port."
            )
        if self.allows(canonical, origin=origin):
            return ""
        if not origin:
            return (
                "url_denied: {} is not an allowlisted domain and was not returned "
                "by this run's guarded search.".format(_describe(url_host(canonical)))
            )
        return (
            "redirect_denied: {} redirected to {}, which is neither an allowlisted "
            "domain, a page this run's guarded search returned, nor the same site."
            .format(_describe(url_host(origin)), _describe(url_host(canonical)))
        )


class ProvenanceTrail:
    """The authorized hops of one fetch, in order.

    A trail is single-use and stateful on purpose: the rule that authorized a
    hop is the hop before it, so the decision cannot be made from the URL alone.
    """

    def __init__(self, policy: FetchProvenance):
        self.policy = policy
        self.hops: list[str] = []

    def authorize(self, url: Any) -> str:
        """Record and authorize one destination; return "" or a refusal."""
        canonical = canonical_url(url)
        origin = self.hops[-1] if self.hops else ""
        reason = self.policy.refusal(url, origin=origin)
        if reason:
            return reason
        if canonical in self.hops:
            return (
                "redirect_denied: {} redirected back to a page this fetch had "
                "already loaded.".format(_describe(url_host(canonical)))
            )
        if len(self.hops) >= self.policy.max_hops:
            return (
                "redirect_denied: the fetch exceeded {} authorized hops."
                .format(self.policy.max_hops)
            )
        self.hops.append(canonical)
        return ""


def verify_chain(
    policy: FetchProvenance, requested: Any, final: Any, chain: Iterable[Any]
) -> str:
    """Re-check a completed fetch against the policy that authorized it.

    The pre-flight check is the guard; this is the audit.  It proves the
    evidence a broker returned actually came from where the policy allowed,
    even if that broker enforced nothing.
    """
    trail = ProvenanceTrail(policy)
    hops = [requested] + [item for item in chain]
    if final and canonical_url(final) not in {canonical_url(item) for item in hops}:
        hops.append(final)
    for hop in hops:
        reason = trail.authorize(hop)
        if reason:
            return reason
    return ""


def supports_provenance(fetch: Callable[..., Any]) -> bool:
    """True when a fetch callback can receive a per-hop provenance policy.

    Callers fail closed on False.  A broker that silently ignores the argument
    would follow redirects on reachability alone, which is the defect this
    module exists to close.
    """
    try:
        signature = inspect.signature(fetch)
    except (TypeError, ValueError):
        return False
    parameters = signature.parameters
    if any(
        parameter.kind == inspect.Parameter.VAR_KEYWORD
        for parameter in parameters.values()
    ):
        return True
    parameter = parameters.get("provenance")
    return parameter is not None and parameter.kind in {
        inspect.Parameter.POSITIONAL_OR_KEYWORD,
        inspect.Parameter.KEYWORD_ONLY,
    }
