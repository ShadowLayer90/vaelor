"""Package power from RAPL energy counters, sampled over a known interval.

RAPL publishes a monotonic *energy* counter, not a power reading, so a single
read says nothing at all — power is the difference between two reads divided by
the time between them. Reporting one sample as watts would be inventing a
denominator.

The counters are root-only on a stock install, which is why this module exists
on the privileged side: the hardware bridge already runs as root and that is
precisely what it is for. The control plane used to report package power as
simply unavailable on this class of machine, which was true of the account it
was asking as and not true of the machine.

Cross-checked on the target hardware: 6.63 W over a 2 s window against
``amd-smi``'s SOCKET_POWER of 8 W and APU_AVERAGE_SOCKET_POWER of 7.7 W. Two
independent sources in agreement, which is the only reason this is reported as
a measurement rather than as a number a file contained.

Re-measured on the same machine on 2026-08-07, this time with both readings
taken across one shared window: **RAPL package-0 8.86 W against
``apu_average_socket_power`` 8.85 W**. The bridge answered the unprivileged
control plane with 10.29 W on a live call in the same session, so the card that
said *"Energy counters on this machine are root-only, so Vaelor cannot read
them"* was describing a route Vaelor had already built and was not using.

Two ways to take the reading live here, and they are not interchangeable.
:func:`package_power` sleeps its own interval, which is right for a one-shot
report and wrong on a poll loop — a telemetry snapshot that blocks for two
seconds stalls every reader for two seconds. :class:`EnergySampler` measures
across the gap between successive polls instead, exactly as the CPU and network
counters in ``linux_telemetry`` already do, and reports nothing at all until it
has two samples far enough apart to divide by.
"""

from __future__ import annotations

import time
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional

from .platforms.graphics_software import PACKAGE_SOCKET_POWER_FIELD


#: Where the kernel exposes RAPL domains.
RAPL_ROOT = "/sys/class/powercap"

#: The directory prefix the kernel currently uses. Kept for documentation, and
#: deliberately *not* used as the filter: an AMD part reports through
#: ``intel-rapl`` today, a future kernel may not, and a domain is identified by
#: publishing an energy counter rather than by what its directory is called.
#: Matching on the counter also means this reads correctly on any tree layout.
PACKAGE_PREFIX = "intel-rapl:"

#: The counter wraps at ``max_energy_range_uj``. A wrap looks like a huge
#: negative delta, which would otherwise render as an absurd power figure.
DEFAULT_WRAP_UJ = 2 ** 32

#: Shortest interval that yields a stable figure. Below this the quantisation
#: of the counter dominates and the answer is noise wearing a decimal point.
MINIMUM_INTERVAL_SECONDS = 0.5


def _read(path: Path) -> str:
    try:
        return path.read_text(encoding="utf-8", errors="replace").strip()
    except OSError:
        return ""


def _integer(path: Path) -> Optional[int]:
    raw = _read(path)
    try:
        return int(raw)
    except (TypeError, ValueError):
        return None


def domains(rapl_root: str = RAPL_ROOT) -> List[Dict[str, Any]]:
    """RAPL domains this host exposes, with their names and counters."""
    found: List[Dict[str, Any]] = []
    try:
        entries = sorted((Path(rapl_root)).iterdir())
    except OSError:
        return found
    for entry in entries:
        if not entry.is_dir():
            continue
        energy = _integer(entry / "energy_uj")
        if energy is None:
            # Not a RAPL domain: `powercap` also holds constraint and
            # enable nodes that publish no counter.
            continue
        found.append({
            "domain": entry.name,
            "name": _read(entry / "name") or entry.name,
            "energy_uj": energy,
            "wrap_uj": _integer(entry / "max_energy_range_uj") or DEFAULT_WRAP_UJ,
        })
    return found


def _delta(first: Dict[str, Any], second: Dict[str, Any]) -> int:
    consumed = second["energy_uj"] - first["energy_uj"]
    if consumed < 0:
        # The counter wrapped between samples rather than the processor
        # producing energy.
        consumed += int(first.get("wrap_uj") or DEFAULT_WRAP_UJ)
    return consumed


def package_power(
    rapl_root: str = RAPL_ROOT,
    *,
    interval_seconds: float = 2.0,
    sleep: Callable[[float], None] = time.sleep,
    clock: Callable[[], float] = time.monotonic,
) -> Dict[str, Any]:
    """Watts per RAPL domain, measured across two samples.

    Returns ``available: False`` with a reason whenever a real measurement
    cannot be made — no counters, one sample only, or an interval too short to
    mean anything. None of those is reported as zero watts.
    """
    first = {entry["domain"]: entry for entry in domains(rapl_root)}
    if not first:
        return {
            "available": False,
            "watts": None,
            "domains": [],
            "reason": (
                "This host exposes no RAPL energy counters under {}. They are "
                "also root-only, so an unprivileged reader sees none even where "
                "they exist."
            ).format(rapl_root),
        }
    wanted = max(MINIMUM_INTERVAL_SECONDS, float(interval_seconds or 0))
    started = clock()
    sleep(wanted)
    elapsed = clock() - started
    second = {entry["domain"]: entry for entry in domains(rapl_root)}
    if elapsed < MINIMUM_INTERVAL_SECONDS:
        return {
            "available": False,
            "watts": None,
            "domains": [],
            "reason": (
                "Only {:.2f} s elapsed between energy samples, which is below "
                "the {:.2f} s needed for the reading to mean anything."
            ).format(elapsed, MINIMUM_INTERVAL_SECONDS),
        }
    return _measure(first, second, elapsed)


def _measure(
    first: Dict[str, Dict[str, Any]],
    second: Dict[str, Dict[str, Any]],
    elapsed: float,
) -> Dict[str, Any]:
    """Watts per domain from two counter reads and the gap between them."""
    measured: List[Dict[str, Any]] = []
    for domain, before in first.items():
        after = second.get(domain)
        if after is None:
            # The domain went away between samples; a delta across a gap is
            # not a measurement.
            continue
        watts = _delta(before, after) / 1_000_000.0 / elapsed
        measured.append({
            "domain": domain,
            "name": before["name"],
            "watts": round(watts, 2),
        })
    if not measured:
        return {
            "available": False,
            "watts": None,
            "domains": [],
            "reason": (
                "No RAPL domain survived both samples, so no interval could be "
                "measured across one."
            ),
        }
    # The package domain is the total; sub-domains are slices of it and must
    # not be summed into it.
    package = next(
        (entry for entry in measured if entry["name"].startswith("package")),
        measured[0],
    )
    return {
        "available": True,
        "watts": package["watts"],
        "source": "rapl/{}".format(package["name"]),
        "interval_seconds": round(elapsed, 3),
        "domains": measured,
        "reason": "",
    }


#: Why no reading is available, by cause. Kept apart from each other on
#: purpose. "Nothing privileged is running", "the privileged side could not
#: answer" and "this processor publishes no counters" are three different facts
#: about three different situations, and only the last is a statement about the
#: hardware. Collapsing them is how an x86 workstation came to be told its own
#: hardware could not be read — and a reader that answered the first two with
#: the third would be making the same mistake in a new place, because a wedged
#: or out-of-date bridge looks exactly like an empty ``/sys/class/powercap``
#: from here.
NO_PRIVILEGED_READER = (
    "Package energy counters are root-only and the privileged hardware bridge "
    "is not running, so nothing on this host can read them."
)
BRIDGE_REFUSED = (
    "Package energy counters are root-only and the privileged hardware bridge "
    "did not answer: {}. That is a fault in the reader, not a statement about "
    "this processor."
)
NO_COUNTERS = (
    "This processor publishes no RAPL energy counters under {}."
)
FIRST_SAMPLE = (
    "Energy counters are cumulative, so the first sample only sets a baseline. "
    "Watts appear on the next reading."
)


class EnergySampler:
    """Watts across successive polls, for a caller that must not sleep.

    Holds the previous counter read and divides by the interval the caller's
    own polling produced. Until there are two samples far enough apart there is
    no reading, and the absence says which of the two reasons it is — a
    baseline still being established, or a machine with no counters at all.
    """

    def __init__(
        self,
        *,
        minimum_interval: float = MINIMUM_INTERVAL_SECONDS,
        clock: Callable[[], float] = time.monotonic,
    ) -> None:
        self.minimum_interval = float(minimum_interval)
        self.clock = clock
        self._previous: Optional[Dict[str, Dict[str, Any]]] = None
        self._previous_at: Optional[float] = None
        self._last: Optional[Dict[str, Any]] = None
        self._last_at: Optional[float] = None

    def observe(self, entries: Optional[List[Dict[str, Any]]]) -> Dict[str, Any]:
        """Take one counter read and report what can be measured from it."""
        samples = {entry["domain"]: entry for entry in (entries or [])}
        if not samples:
            # Deliberately before the clock is read. A host with no counters
            # must cost nothing at all, including a timestamp, so that a caller
            # sharing a clock with something else is not disturbed by a reading
            # that was never going to happen.
            # A machine that publishes nothing must not inherit the last
            # reading from a machine-shaped memory of one.
            self._previous = self._previous_at = None
            self._last = self._last_at = None
            return {
                "available": False, "watts": None, "domains": [],
                "reason": NO_COUNTERS.format(RAPL_ROOT),
            }
        now = self.clock()
        previous, previous_at = self._previous, self._previous_at
        if previous is None or previous_at is None:
            self._previous, self._previous_at = samples, now
            return {
                "available": False, "watts": None, "domains": [],
                "reason": FIRST_SAMPLE,
            }
        elapsed = now - previous_at
        if elapsed < self.minimum_interval:
            # Too soon to divide by. The baseline is deliberately *not*
            # replaced, or a fast poller would reset the window on every call
            # and never accumulate one.
            if self._last is not None and self._last_at is not None:
                return {**self._last, "age_seconds": round(now - self._last_at, 3)}
            return {
                "available": False, "watts": None, "domains": [],
                "reason": FIRST_SAMPLE,
            }
        result = _measure(previous, samples, elapsed)
        self._previous, self._previous_at = samples, now
        if result.get("available"):
            self._last, self._last_at = result, now
            return {**result, "age_seconds": 0.0}
        return result


#: Telemetry keys. ``basis`` is required by VD-040: a watt figure from a
#: two-sample delta and a watt figure a sensor published are different claims,
#: and the difference is stated rather than left to be discovered.
TELEMETRY_KEYS = {
    "watts": "cpu_package_power_watts",
    "source": "cpu_package_power_source",
    "interval": "cpu_package_power_interval_seconds",
    "age": "cpu_package_power_age_seconds",
    "reason": "cpu_package_power_reason",
    "basis": "cpu_package_power_basis",
    "crosscheck": "cpu_package_power_crosscheck",
}

#: Source label when the reading comes from ``amd-smi`` rather than RAPL. The
#: field name is imported, not re-spelled, so this and the reader cannot drift.
AMD_SMI_PACKAGE_SOURCE = "amd-smi {}".format(PACKAGE_SOCKET_POWER_FIELD)

#: How close RAPL and amd-smi must be for the cross-check to call them agreed:
#: within the larger of an absolute floor and a fraction of the reported figure.
#: The two are separate instruments and a small gap is expected; a large one is
#: a real signal and must read as one. A cross-check that says "agree" no matter
#: how far apart the sources are is the guarded-but-mute pattern (CLAUDE.md) — a
#: line that cannot fail to reassure, which is worse than no line at all.
CROSSCHECK_ABSOLUTE_WATTS = 0.5
CROSSCHECK_RELATIVE = 0.05


def _crosscheck_line(rapl_source: str, rapl_watts: float, socket_watts: float) -> str:
    """Compare the RAPL reading against the reported amd-smi figure, honestly.

    Agreement is claimed only when the gap is within
    ``max(CROSSCHECK_ABSOLUTE_WATTS, CROSSCHECK_RELATIVE * socket_watts)``; a
    wider gap is stated as a disagreement, so the line distinguishes concord
    from conflict rather than reassuring in both.
    """
    delta = abs(rapl_watts - socket_watts)
    tolerance = max(CROSSCHECK_ABSOLUTE_WATTS, CROSSCHECK_RELATIVE * abs(socket_watts))
    if delta <= tolerance:
        return "{} reads {:.2f} W; the two agree within {:.2f} W.".format(
            rapl_source, rapl_watts, delta
        )
    return (
        "{} reads {:.2f} W, disagreeing with the amd-smi figure by {:.2f} W — "
        "beyond the {:.2f} W cross-check tolerance."
    ).format(rapl_source, rapl_watts, delta, tolerance)


def read_domains(rapl_root: str = RAPL_ROOT, client: Any = None) -> Dict[str, Any]:
    """The counters, read directly if that works and through the bridge if not.

    Direct first because it costs nothing and succeeds wherever the control
    plane happens to hold the privilege; the bridge second because on a stock
    install it is the only account that can open these files.

    Returns the domains **and, when there are none, why** — from the one call
    that knows. A bridge that is missing, wedged, or too old to know this action
    is indistinguishable from an empty ``/sys/class/powercap`` by the time an
    empty list reaches the caller, and answering any of those with "this
    processor publishes no counters" would be a fresh instance of the mistake
    this whole module exists to correct.
    """
    direct = domains(rapl_root)
    if direct:
        return {"domains": direct, "reason": ""}
    from .hardware_bridge import HardwareBridgeClient, HardwareBridgeError

    reader = client if client is not None else HardwareBridgeClient()
    if not getattr(reader, "available", False):
        return {"domains": [], "reason": NO_PRIVILEGED_READER}
    try:
        payload = reader.rapl_energy()
    except (HardwareBridgeError, OSError, RuntimeError, TypeError, ValueError) as error:
        return {"domains": [], "reason": BRIDGE_REFUSED.format(error)}
    found = (payload or {}).get("domains")
    return {
        "domains": list(found) if isinstance(found, list) else [],
        "reason": "",
    }


def package_power_telemetry(
    sampler: EnergySampler,
    reader: Callable[[], Dict[str, Any]] = read_domains,
    *,
    socket_watts: Optional[float] = None,
) -> Dict[str, Any]:
    """Telemetry keys for the poll loop, with a reason whenever there is no watt.

    The reason is published beside the gap rather than instead of it: a card
    reading "?" with "the counters are root-only and the privileged bridge is
    not running" tells an owner something actionable, and a card reading "?"
    with "Vaelor cannot read them" tells them something false.

    ``socket_watts`` is the whole-socket power from the shared ``amd-smi``
    snapshot the GPU and NPU cards also read. When present it is the reported
    figure, so all three power cards come from one instrument at one instant;
    RAPL then serves only as an honest cross-check. When it is ``None`` — every
    host without amd-smi, the Pi included — the RAPL energy-delta is the reading
    exactly as before, and that hardware is unchanged.
    """
    reading = reader()
    entries = reading.get("domains") or []
    result = sampler.observe(entries)
    if socket_watts is not None:
        telemetry = {
            TELEMETRY_KEYS["watts"]: socket_watts,
            TELEMETRY_KEYS["source"]: AMD_SMI_PACKAGE_SOURCE,
            TELEMETRY_KEYS["basis"]: "socket-power",
        }
        if result.get("available"):
            telemetry[TELEMETRY_KEYS["crosscheck"]] = _crosscheck_line(
                result["source"], result["watts"], socket_watts
            )
        return telemetry
    if not result.get("available"):
        # The reader's own account of the gap wins where it has one: it knows
        # whether it was refused, and the sampler only ever sees an empty list.
        reason = str(reading.get("reason") or "") or str(result.get("reason") or "")
        return {TELEMETRY_KEYS["reason"]: reason} if reason else {}
    telemetry = {
        TELEMETRY_KEYS["watts"]: result["watts"],
        TELEMETRY_KEYS["source"]: result["source"],
        TELEMETRY_KEYS["interval"]: result["interval_seconds"],
        TELEMETRY_KEYS["basis"]: "energy-delta",
    }
    age = result.get("age_seconds")
    if age:
        telemetry[TELEMETRY_KEYS["age"]] = age
    return telemetry
