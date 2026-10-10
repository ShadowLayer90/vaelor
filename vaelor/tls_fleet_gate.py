"""When an existing install may start serving its household certificate (VD-212 migration).

An install from before VD-212 serves a self-signed certificate its workers pin.
The authority ships them the root *and* that certificate first, and swaps the
served certificate only once the workers hold the new bundle. Which workers
hold it is known to the control plane (its profile sweep reads each worker's
copy, its telemetry intake sees each worker report); it records that in
:data:`vaelor.tls_paths.FLEET_TRUST_STATE`, and this module is both the
writer's format and the reader's verdict, so the two sides cannot disagree
about the schema (LESSONS 6).

**The record is advisory.** It is written by the control plane's account, so
the root authority lets it decide only *when* to promote - never which names or
keys are signed. Every field can only delay promotion; none adds a name.

The rule (plan, VD-212, adversarial review 1):

* A worker is **online** when the sweep reached it, or its telemetry arrived,
  in the last 24 hours. An online worker without the current bundle blocks.
  A worker offline for 24 hours does not (LESSONS 22).
* The controller's **advertise address** - what workers dial - must be in the
  pending certificate, or promotion is refused loudly: the new certificate is
  private-LAN filtered and the old one may have named an address it drops.
* **No record at all** (missing, unreadable, wrong schema): promote 24 hours
  after the pending certificate was issued. Absence is not an empty fleet
  (LESSONS 8), but it must not be a state nothing leaves either (LESSONS 22).
* **A record that exists but is stale or names another bundle**: the same 24 h
  fallback, *unless* it shows a worker whose telemetry was recent - then the
  authority keeps waiting, because a worker that is demonstrably online may
  pin only the old certificate, and promoting would cut it off. The owner's
  exit is always ``python -m vaelor.tls_authority promote``.
"""

from __future__ import annotations

import ipaddress
import json
import time
from dataclasses import dataclass
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence

FLEET_TRUST_SCHEMA = "vaelor-fleet-trust/1"
#: A worker neither reached nor heard from for this long no longer blocks.
REACHED_WINDOW_SECONDS = 24 * 3600
#: With no usable record, promote this long after the pending certificate.
UNOBSERVED_PROMOTION_SECONDS = 24 * 3600
#: A record older than this is not a current observation (the sweep runs
#: every 15 minutes; several missed sweeps mean the observer stopped).
RECORD_STALE_SECONDS = 2 * 3600
_MAX_RECORD_BYTES = 1 << 20
_OVERRIDE = "run `python -m vaelor.tls_authority promote` to serve it anyway"


def _epoch(value: Any) -> Optional[float]:
    return None if value is None else float(value)


def fleet_trust_record(bundle_sha256: str, workers: Iterable[Mapping[str, Any]],
                       now: Optional[float] = None,
                       advertise_address: Optional[str] = None) -> Dict[str, Any]:
    """The record the control plane writes after each profile sweep.

    ``workers``: one mapping per joined worker with ``id`` (str), ``matches``
    (its controller-trust file's SHA-256 equals ``bundle_sha256``),
    ``last_reached`` (epoch seconds the sweep last read it, or None) and
    ``last_telemetry`` (epoch seconds its telemetry last arrived, or None).
    ``advertise_address``: the name or address workers dial for this
    controller, or None.
    """
    return {
        "schema": FLEET_TRUST_SCHEMA,
        "written_at": float(time.time() if now is None else now),
        "bundle_sha256": str(bundle_sha256),
        "advertise_address": None if advertise_address is None else str(advertise_address),
        "workers": [
            {"id": str(w.get("id", "")), "matches": bool(w.get("matches")),
             "last_reached": _epoch(w.get("last_reached")),
             "last_telemetry": _epoch(w.get("last_telemetry"))}
            for w in workers
        ],
    }


@dataclass(frozen=True)
class PromotionVerdict:
    promote: bool
    reason: str
    #: True when promotion is refused for a reason the owner must act on
    #: (the advertise address the new certificate would not cover).
    blocked: bool = False


def read_record(path: str) -> Dict[str, Any] | None:
    """The parsed record, or None when it is absent, oversized or not JSON."""
    try:
        with open(path, "rb") as stream:
            raw = stream.read(_MAX_RECORD_BYTES + 1)
    except OSError:
        return None
    if len(raw) > _MAX_RECORD_BYTES:
        return None
    try:
        parsed = json.loads(raw.decode("utf-8"))
    except (UnicodeDecodeError, ValueError):
        return None
    return parsed if isinstance(parsed, dict) else None


def _number(value: Any) -> float | None:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        return None
    return float(value)


def _covered(advertise: str, names: Sequence[str], addresses: Sequence[str]) -> bool:
    try:
        return str(ipaddress.ip_address(advertise.strip())) in addresses
    except ValueError:
        return advertise.strip().rstrip(".").lower() in {n.lower() for n in names}


def _recent(worker: Mapping[str, Any], key: str, now: float) -> bool:
    stamp = _number(worker.get(key))
    return stamp is not None and now - stamp <= REACHED_WINDOW_SECONDS


def promotion_verdict(record: Mapping[str, Any] | None, bundle_sha256: str,
                      pending_issued_at: float, now: float,
                      pending_names: Sequence[str] = (),
                      pending_addresses: Sequence[str] = ()) -> PromotionVerdict:
    """Whether the pending certificate may be served now, and why."""
    fallback = now - pending_issued_at >= UNOBSERVED_PROMOTION_SECONDS

    def unobserved(why: str) -> PromotionVerdict:
        if fallback:
            return PromotionVerdict(True, why + "; the pending certificate is over "
                                    "24 hours old, so it is promoted on schedule")
        return PromotionVerdict(False, why + "; waiting up to 24 hours for the "
                                "control plane to observe the workers")

    if not isinstance(record, Mapping) or record.get("schema") != FLEET_TRUST_SCHEMA:
        return unobserved("no readable fleet trust record")
    advertise = record.get("advertise_address")
    if isinstance(advertise, str) and advertise.strip() and not _covered(
            advertise, pending_names, pending_addresses):
        return PromotionVerdict(
            False, "workers reach this controller at {}, which the household "
            "certificate cannot name (it covers private LAN names and addresses "
            "only). Advertise a private address for the cluster, or {}".format(
                advertise.strip()[:255], _OVERRIDE), blocked=True)
    workers = record.get("workers")
    entries: List[Mapping[str, Any]] = [
        w for w in workers if isinstance(w, Mapping)] if isinstance(workers, list) else []
    heard = sorted(str(w.get("id", "?"))[:64] for w in entries
                   if _recent(w, "last_telemetry", now))

    def stale(why: str) -> PromotionVerdict:
        if heard:
            return PromotionVerdict(False, "{}, and {} still sent telemetry in the "
                                    "last 24 hours, so it is not promoted on "
                                    "schedule; {}".format(why, ", ".join(heard),
                                                          _OVERRIDE))
        return unobserved(why)

    written = _number(record.get("written_at"))
    if written is None or now - written > RECORD_STALE_SECONDS or written > now + 300:
        return stale("the fleet trust record is not current")
    if record.get("bundle_sha256") != bundle_sha256:
        return stale("the fleet trust record describes a different bundle")
    if not isinstance(workers, list) or len(entries) != len(workers):
        return stale("the fleet trust record's worker list is malformed")
    online = [w for w in entries
              if _recent(w, "last_reached", now) or _recent(w, "last_telemetry", now)]
    blocking = sorted(str(w.get("id", "?"))[:64] for w in online
                      if w.get("matches") is not True)
    if blocking:
        return PromotionVerdict(False, "waiting for {} to receive the household "
                                "certificate bundle".format(", ".join(blocking)))
    if not entries:
        return PromotionVerdict(True, "no workers are joined")
    if not online:
        return PromotionVerdict(True, "none of the {} joined workers was reached or "
                                "sent telemetry in the last 24 hours, so none can "
                                "hold this back".format(len(entries)))
    return PromotionVerdict(True, "{} of {} joined workers were online in the last "
                            "24 hours and each holds the household certificate "
                            "bundle".format(len(online), len(entries)))
