"""Names the household root must never vouch for: the public top-level domains (VD-212).

The root's DNS constraints include this machine's bare hostname, and an X.509
DNS subtree permits every name *below* it. A box named ``dev``, ``app`` or
``box`` - each a real top-level domain - would get a root able to vouch for
``accounts.google.dev``. Owner (VD-212, adversarial review 1): keep the bare
hostname for ordinary names, but never one that is a real top-level domain,
checked against IANA's official root-zone list shipped with Vaelor.

**Fail closed.** When the list is missing, unreadable or implausibly short, no
bare hostname is trusted at all, and the reason says so (LESSONS 1): a
missing list must not read as "no name is public".
"""

from __future__ import annotations

import functools
import ipaddress
from pathlib import Path
from typing import FrozenSet, List, Optional, Sequence

#: IANA's root-zone list (https://data.iana.org/TLD/tlds-alpha-by-domain.txt),
#: shipped as a data file; its header records the source and the version date.
TLD_LIST = Path(__file__).resolve().parent / "data" / "iana-root-zone-tlds.txt"
#: The root zone has held well over a thousand TLDs since 2016; fewer means a
#: truncated or wrong file, which is treated as no list.
MINIMUM_TLDS = 1000
PUBLIC_TLD = "it is a public top-level domain"
NO_LIST = ("the public top-level domain list is not available, so no bare "
           "hostname is trusted")


@functools.lru_cache(maxsize=None)
def public_tlds(path: Path = TLD_LIST) -> Optional[FrozenSet[str]]:
    """Every delegated TLD, lower-case; None when the list cannot be trusted."""
    return read_tld_list(path)


def read_tld_list(path: Path) -> Optional[FrozenSet[str]]:
    """Parse IANA's list at `path` (uncached; `public_tlds` caches it)."""
    try:
        text = path.read_text(encoding="ascii")
    except (OSError, UnicodeDecodeError):
        return None
    found = frozenset(line.strip().lower() for line in text.splitlines()
                      if line.strip() and not line.lstrip().startswith("#"))
    return found if len(found) >= MINIMUM_TLDS else None


def bare_label_refusal(label: str, tlds: Optional[FrozenSet[str]] = None) -> str:
    """Why a single-label name may not be vouched for, or "" when it may."""
    known = public_tlds() if tlds is None else tlds
    if not known:
        return NO_LIST
    return PUBLIC_TLD if label.lower().strip(".") in known else ""


def constraint_violations(dns: Sequence[str], networks: Sequence, critical: bool,
                          private_dns: Sequence[str], private_networks: Sequence,
                          tlds: Optional[FrozenSet[str]] = None) -> List[str]:
    """What makes a root's name constraints wider than the private LAN.

    For an imported root: every permitted network must sit inside a private
    one, every DNS subtree must be one of the local suffixes (or below one) or
    a single label that is not a public TLD, and the extension must be critical.
    """
    problems = [] if critical else ["its name constraints are not critical"]
    # RFC 5280: a name type with no permitted subtree is unconstrained, so a
    # root with only IP subtrees vouches for every DNS name (review R22).
    if not dns:
        problems.append("it has no DNS name constraint, so it may vouch for every DNS name")
    if not networks:
        problems.append("it has no IP address constraint, so it may vouch for every "
                        "IP address")
    for network in networks:
        net = ipaddress.ip_network(network)
        if not any(net.version == p.version and net.subnet_of(p)
                   for p in private_networks):
            problems.append("{} is not a private LAN range".format(net))
    for name in dns:
        base = name.lower().strip(".")
        local = any(base == s or base.endswith("." + s) for s in private_dns)
        if local:
            continue
        if "." in base or not base:
            problems.append("{} is not a local name".format(name))
            continue
        refusal = bare_label_refusal(base, tlds)
        if refusal:
            problems.append("{} ({})".format(name, refusal))
    return problems
