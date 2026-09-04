"""Render the live telemetry snapshot as Prometheus text exposition (v0.0.4).

The same readings the System page shows over auth-gated JSON
(`/api/v2/telemetry/current`) are already collected every second; this module
turns one such snapshot — plus the service, application and job state the rest
of the API serves — into the Prometheus text exposition format so a scraper can
read them at `/api/v2/metrics`. Nothing here fetches or samples: it is a pure
projection of a snapshot dict a caller already holds, so `/metrics` and the UI
read the same numbers at the same instant rather than two clocks.

Three rules this module exists to hold, each a defect it would otherwise invite:

**A missing reading is omitted, never zeroed.** The snapshot omits a key
entirely when the sensor is absent (a Pi has no GPU; a box without amd-smi
publishes no NPU activity). Emitting `vaelor_gpu_temperature_celsius 0` there
would report a running card at 0 °C, indistinguishable from a real reading — the
exact "a missing sensor is not a reading of zero" rule the telemetry providers
already keep. So :func:`_gauge` drops a metric whose value is absent or not a
finite number, and a family with no samples emits no `# HELP`/`# TYPE` at all.

**Labels are low-cardinality by construction.** Every label value comes from a
fixed physical or catalogue set — a board fan channel, a real mount point, a
managed service id, a container name, a job-state word — never a per-request or
unbounded string. A scraper's series count is therefore bounded by the hardware
and the managed inventory, not by traffic. The one family whose cardinality
grows with what the operator runs is `vaelor_app_running`, bounded by the
container count (the inventory itself caps enumeration at 100); it is the number
a future scaler should watch.

**A ratio is a fraction, a temperature is degrees, a byte count is bytes.** The
snapshot carries load and activity as 0–100 percents; they are published here as
0–1 ratios under `_ratio` names, because that is the Prometheus convention and
mixing the two under one suffix is how a dashboard multiplies by a hundred
twice. Scaling is the only transform; nothing here invents a reading.
"""

from __future__ import annotations

import math
from typing import Any, Dict, List, Mapping, NamedTuple, Optional, Sequence, Tuple


#: The Content-Type a Prometheus scrape expects for the text exposition format.
#: Version 0.0.4 is the stable text format; the charset is stated because the
#: HELP lines are UTF-8 and a scraper is entitled to know it.
PROMETHEUS_CONTENT_TYPE = "text/plain; version=0.0.4; charset=utf-8"

#: Percent-to-fraction scale for the readings the snapshot carries as 0–100 but
#: Prometheus convention publishes as 0–1. One constant so every ``*_ratio``
#: family divides by the same number.
_PERCENT_TO_RATIO = 0.01

#: Decimal places a scaled value is rounded to, so ``22.5 * 0.01`` reads as
#: ``0.225`` rather than the float's ``0.22500000000000003``. Six places is finer
#: than any sensor's real precision and coarse enough to hide the binary
#: representation.
_SCALE_PRECISION = 6


class _Sample(NamedTuple):
    """One value of a metric family, with its (already low-cardinality) labels."""

    labels: Tuple[Tuple[str, str], ...]
    value: float


class _Family(NamedTuple):
    """A metric name, its type, its help line, and every sample under it.

    Emitted only when it has at least one sample, so an absent subsystem
    contributes no ``# HELP``/``# TYPE`` header describing metrics that are not
    there.
    """

    name: str
    kind: str
    help: str
    samples: Tuple[_Sample, ...]


#: The scalar gauges, each read from one snapshot key. ``scale`` is applied and
#: the result rounded only when it is not 1, so a temperature stays exact and a
#: percent becomes a ratio. A key absent from the snapshot, or carrying a
#: non-finite value, contributes nothing — the family is dropped entirely.
#:
#: Ordered subsystem by subsystem the way an operator reads a machine: processor,
#: memory, graphics, neural accelerator, network.
_SCALAR_GAUGES: Tuple[Tuple[str, str, float, str], ...] = (
    ("vaelor_cpu_utilization_ratio", "cpu_percent", _PERCENT_TO_RATIO,
     "Processor load as a fraction from 0 to 1."),
    ("vaelor_cpu_temperature_celsius", "cpu_temperature", 1.0,
     "Processor temperature in degrees Celsius."),
    ("vaelor_cpu_frequency_mhz", "cpu_freq", 1.0,
     "Processor clock frequency in megahertz."),
    ("vaelor_cpu_cores", "cpu_cores", 1.0,
     "Number of physical processor cores."),
    ("vaelor_cpu_package_power_watts", "cpu_package_power_watts", 1.0,
     "Processor package power draw in watts."),
    ("vaelor_boot_time_seconds", "boot_time", 1.0,
     "Unix time of the last boot, in seconds."),
    ("vaelor_memory_total_bytes", "memory_total", 1.0,
     "Total physical memory in bytes."),
    ("vaelor_memory_used_bytes", "memory_used", 1.0,
     "Physical memory in use in bytes."),
    ("vaelor_memory_available_bytes", "memory_available", 1.0,
     "Physical memory available to allocate in bytes."),
    ("vaelor_gpu_utilization_ratio", "gpu_busy_percent", _PERCENT_TO_RATIO,
     "Graphics processor load as a fraction from 0 to 1."),
    ("vaelor_gpu_temperature_celsius", "gpu_temperature_c", 1.0,
     "Graphics processor temperature in degrees Celsius."),
    ("vaelor_gpu_vram_used_bytes", "gpu_vram_used_bytes", 1.0,
     "Graphics memory in use in bytes."),
    ("vaelor_gpu_vram_total_bytes", "gpu_vram_total_bytes", 1.0,
     "Total graphics memory in bytes."),
    ("vaelor_gpu_power_watts", "gpu_power_watts", 1.0,
     "Graphics processor power draw in watts."),
    ("vaelor_gpu_clock_mhz", "gpu_clock_mhz", 1.0,
     "Graphics processor clock frequency in megahertz."),
    ("vaelor_npu_activity_ratio", "npu_activity_percent", _PERCENT_TO_RATIO,
     "Neural processor activity as a fraction from 0 to 1."),
    ("vaelor_npu_power_watts", "npu_power_watts", 1.0,
     "Neural processor power draw in watts."),
    ("vaelor_npu_clock_mhz", "npu_clock_mhz", 1.0,
     "Neural processor clock frequency in megahertz."),
    ("vaelor_network_receive_bytes_per_second", "network_download_speed", 1.0,
     "Network receive throughput in bytes per second."),
    ("vaelor_network_transmit_bytes_per_second", "network_upload_speed", 1.0,
     "Network transmit throughput in bytes per second."),
)

#: Job-ledger counts published as one gauge family labelled by state. The states
#: are the fixed keys `JobStore.ledger()` returns, so the ``state`` label is
#: bounded to this set rather than growing with the number of jobs.
_JOB_STATES: Tuple[str, ...] = (
    "total", "active", "attention", "retryable", "resolved_failures",
)


def _number(value: Any) -> Optional[float]:
    """``value`` as a finite float, or ``None`` for anything that is not one.

    ``bool`` is rejected before ``int``: it is an ``int`` to Python and a state
    to a machine, and ``True`` is not a reading of one watt. ``NaN`` and the
    infinities are holes, not numbers, and a metric with one is omitted rather
    than emitted as an unparseable value. An integer too large to fit a float
    (``10**400``) raises ``OverflowError`` on conversion; it is a hole too, not
    a reading, and is omitted the same way.
    """
    if isinstance(value, bool) or value is None:
        return None
    try:
        number = float(value)
    except (TypeError, ValueError, OverflowError):
        return None
    return number if math.isfinite(number) else None


def _render_number(value: float) -> str:
    """A finite float as Prometheus expects it: an integer without a point.

    ``repr`` gives the shortest string that round-trips a float, so a genuine
    fraction keeps its precision; an integer-valued float is printed without the
    trailing ``.0`` a scraper would otherwise carry through every byte count.
    """
    if value == int(value):
        return str(int(value))
    return repr(value)


def _escape_label(value: str) -> str:
    """Escape a label value for the exposition format.

    Backslash, double-quote and newline are the three characters the format
    reserves inside a quoted label value; everything else, spaces included, is
    literal.
    """
    return (
        str(value)
        .replace("\\", "\\\\")
        .replace("\"", "\\\"")
        .replace("\n", "\\n")
    )


def _sample(value: Any, *, scale: float = 1.0,
            labels: Sequence[Tuple[str, str]] = ()) -> Optional[_Sample]:
    """One sample, or ``None`` when the reading is absent or not finite.

    The scale is applied and the result rounded only when it is not 1, so a
    reading that needs no transform is never perturbed by float arithmetic.
    """
    number = _number(value)
    if number is None:
        return None
    if scale != 1.0:
        number = round(number * scale, _SCALE_PRECISION)
    return _Sample(tuple(labels), number)


def _gauge(name: str, help_text: str, value: Any, *, scale: float = 1.0
           ) -> Optional[_Family]:
    """A single-sample gauge family, or ``None`` when the reading is missing."""
    sample = _sample(value, scale=scale)
    if sample is None:
        return None
    return _Family(name, "gauge", help_text, (sample,))


def _scalar_families(snapshot: Mapping[str, Any]) -> List[_Family]:
    families = []
    for name, key, scale, help_text in _SCALAR_GAUGES:
        family = _gauge(name, help_text, snapshot.get(key), scale=scale)
        if family is not None:
            families.append(family)
    return families


def _dedup_by_labels(samples: Sequence[Optional[_Sample]]) -> Tuple[_Sample, ...]:
    """Drop ``None`` and keep the first sample for each distinct label set.

    Prometheus rejects a scrape that carries two samples with the same metric
    name and label set ("duplicate sample for timestamp") and can fail the whole
    target, so no family may emit two lines that share a name and labels. This is
    the single point that guarantees it: every labelled family passes its samples
    through here, so a collision — a compose-scaled service's replicas sharing
    one ``display_identity``, a synthetic mount colliding with a real one, a
    repeated service id — becomes one series rather than a broken exposition.
    """
    seen: set = set()
    kept: List[_Sample] = []
    for sample in samples:
        if sample is None or sample.labels in seen:
            continue
        seen.add(sample.labels)
        kept.append(sample)
    return tuple(kept)


def _labelled_gauge(
    name: str, help_text: str, samples: Sequence[Optional[_Sample]]
) -> Optional[_Family]:
    """A gauge family with labels, or ``None`` when nothing was readable."""
    kept = _dedup_by_labels(samples)
    if not kept:
        return None
    return _Family(name, "gauge", help_text, kept)


def _fan_families(snapshot: Mapping[str, Any]) -> List[_Family]:
    """``vaelor_fan_rpm{fan="..."}`` from the WMI fan channels.

    The ``fan`` label is the board's own channel name — a fixed physical set of
    three on the Z2 — so the cardinality is the machine's fan count. A channel
    whose tachometer does not read a number is omitted, not reported as a
    stopped fan.
    """
    fans = snapshot.get("wmi_fans")
    if not isinstance(fans, list):
        return []
    samples = []
    for entry in fans:
        if not isinstance(entry, Mapping):
            continue
        label = str(entry.get("label") or "").strip()
        if not label:
            continue
        sample = _sample(entry.get("rpm"), labels=(("fan", label),))
        if sample is not None:
            samples.append(sample)
    family = _labelled_gauge(
        "vaelor_fan_rpm", "Fan speed in revolutions per minute.", samples
    )
    return [family] if family is not None else []


def _storage_volumes(snapshot: Mapping[str, Any]) -> List[Tuple[str, int, int]]:
    """``(mount, used, total)`` for each real filesystem in the snapshot.

    The provider publishes ``disk_<volume>_{total,used,mount}`` for every real
    filesystem plus a synthetic ``disk_primary_*`` copy of one of them; the
    primary is skipped here so a mount is not counted twice. The label is the
    actual mount point, a fixed set of real filesystems.
    """
    volumes = []
    for key in snapshot:
        if not (isinstance(key, str) and key.startswith("disk_")
                and key.endswith("_total")):
            continue
        name = key[len("disk_"):-len("_total")]
        if name == "primary":
            continue
        total = _number(snapshot.get("disk_{}_total".format(name)))
        used = _number(snapshot.get("disk_{}_used".format(name)))
        if total is None or used is None:
            continue
        mount = snapshot.get("disk_{}_mount".format(name))
        mount = str(mount) if mount else "/" + name
        volumes.append((mount, int(used), int(total)))
    return sorted(volumes)


def _storage_families(snapshot: Mapping[str, Any]) -> List[_Family]:
    volumes = _storage_volumes(snapshot)
    if not volumes:
        return []
    total_samples = []
    used_samples = []
    free_samples = []
    for mount, used, total in volumes:
        labels = (("mount", mount),)
        total_samples.append(_Sample(labels, float(total)))
        used_samples.append(_Sample(labels, float(used)))
        free_samples.append(_Sample(labels, float(max(0, total - used))))
    families = [
        _labelled_gauge("vaelor_storage_total_bytes",
                        "Filesystem capacity in bytes.", total_samples),
        _labelled_gauge("vaelor_storage_used_bytes",
                        "Filesystem space in use in bytes.", used_samples),
        _labelled_gauge("vaelor_storage_free_bytes",
                        "Filesystem space free in bytes.", free_samples),
    ]
    return [family for family in families if family is not None]


def _service_families(services: Optional[Sequence[Mapping[str, Any]]]
                      ) -> List[_Family]:
    """``vaelor_service_up`` and ``vaelor_service_restarts_total`` per service.

    The ``service`` label is a managed-service catalogue id, a fixed low set.
    A service whose state could not be read — ``available`` false, meaning
    systemctl is absent or the unit is not loaded — is omitted rather than
    reported down, because "not measured" and "measured and stopped" are
    different answers.
    """
    if not services:
        return []
    up_samples = []
    restart_samples = []
    for entry in services:
        if not isinstance(entry, Mapping) or not entry.get("available"):
            continue
        service_id = str(entry.get("id") or "").strip()
        if not service_id:
            continue
        labels = (("service", service_id),)
        up_samples.append(
            _Sample(labels, 1.0 if entry.get("active") == "active" else 0.0)
        )
        restarts = _sample(entry.get("restarts"), labels=labels)
        if restarts is not None:
            restart_samples.append(restarts)
    families = [
        _labelled_gauge(
            "vaelor_service_up",
            "1 if the managed service is active, 0 if it is not.",
            up_samples,
        ),
    ]
    counter = _counter(
        "vaelor_service_restarts_total",
        "Times the managed service has been restarted since it last started.",
        restart_samples,
    )
    if counter is not None:
        families.append(counter)
    return [family for family in families if family is not None]


def _counter(name: str, help_text: str, samples: Sequence[Optional[_Sample]]
             ) -> Optional[_Family]:
    kept = _dedup_by_labels(samples)
    if not kept:
        return None
    return _Family(name, "counter", help_text, kept)


def _app_families(apps: Optional[Sequence[Mapping[str, Any]]]) -> List[_Family]:
    """``vaelor_app_running`` and ``vaelor_app_replicas`` per app identity.

    The ``app`` label is the container's display identity — a stable name, not a
    per-request value — but this is the one family whose cardinality tracks what
    the operator runs, bounded by the container count (the inventory caps its
    own enumeration at 100). A future scaler watching series growth should watch
    this one.

    Several containers can share one identity: a compose-scaled service
    (``--scale web=3``) gives three containers whose ``display_identity`` is the
    identical ``project/service``. Emitting one ``vaelor_app_running{app=...}``
    line per container would put three lines with the same name and label set in
    the scrape, which Prometheus rejects. So containers are grouped by identity
    into one series each: ``vaelor_app_running`` is 1 when *any* replica of that
    identity is running (the app is reachable), and the multiplicity that
    grouping would otherwise hide is published honestly as
    ``vaelor_app_replicas`` — the container count for the identity, the number a
    scaler actually watches.
    """
    if not apps:
        return []
    grouped: Dict[str, Dict[str, Any]] = {}
    for entry in apps:
        if not isinstance(entry, Mapping):
            continue
        name = str(
            entry.get("display_identity") or entry.get("name") or ""
        ).strip()
        if not name:
            continue
        running = bool(entry.get("running"))
        state = grouped.get(name)
        if state is None:
            grouped[name] = {"running": running, "replicas": 1}
        else:
            state["running"] = state["running"] or running
            state["replicas"] += 1
    running_samples = [
        _Sample((("app", name),), 1.0 if state["running"] else 0.0)
        for name, state in grouped.items()
    ]
    replica_samples = [
        _Sample((("app", name),), float(state["replicas"]))
        for name, state in grouped.items()
    ]
    families = [
        _labelled_gauge(
            "vaelor_app_running",
            "1 if any container for this application identity is running, else 0.",
            running_samples,
        ),
        _labelled_gauge(
            "vaelor_app_replicas",
            "Number of containers sharing this application identity.",
            replica_samples,
        ),
    ]
    return [family for family in families if family is not None]


def _job_families(job_counts: Optional[Mapping[str, Any]]) -> List[_Family]:
    """``vaelor_jobs_count{state="..."}`` from the job-ledger counters.

    The ``state`` label is bounded to :data:`_JOB_STATES`, the fixed keys the
    ledger publishes, so this family's cardinality does not grow with the number
    of jobs.
    """
    if not job_counts:
        return []
    samples = []
    for state in _JOB_STATES:
        sample = _sample(job_counts.get(state), labels=(("state", state),))
        if sample is not None:
            samples.append(sample)
    family = _labelled_gauge(
        "vaelor_jobs_count",
        "Operations in the job ledger, by state.",
        samples,
    )
    return [family] if family is not None else []


def _render_family(family: _Family) -> str:
    lines = [
        "# HELP {} {}".format(family.name, family.help),
        "# TYPE {} {}".format(family.name, family.kind),
    ]
    for sample in family.samples:
        if sample.labels:
            rendered = ",".join(
                '{}="{}"'.format(key, _escape_label(value))
                for key, value in sample.labels
            )
            series = "{}{{{}}}".format(family.name, rendered)
        else:
            series = family.name
        lines.append("{} {}".format(series, _render_number(sample.value)))
    return "\n".join(lines)


def render_metrics(
    snapshot: Optional[Mapping[str, Any]] = None,
    *,
    services: Optional[Sequence[Mapping[str, Any]]] = None,
    apps: Optional[Sequence[Mapping[str, Any]]] = None,
    job_counts: Optional[Mapping[str, Any]] = None,
) -> str:
    """The Prometheus text exposition for one telemetry snapshot and state.

    Pure: it reads the arguments and returns text, sampling nothing. Pass the
    very dict `/api/v2/telemetry/current` serves as ``snapshot`` and `/metrics`
    reports the numbers the System page shows, at that instant. Every family
    with no readable sample is absent from the output rather than emitted as a
    zero, so the exposition describes only what was actually measured.

    A trailing newline terminates the last line, which the format requires.
    """
    readings = dict(snapshot or {})
    families: List[_Family] = []
    families.extend(_scalar_families(readings))
    families.extend(_fan_families(readings))
    families.extend(_storage_families(readings))
    families.extend(_service_families(services))
    families.extend(_app_families(apps))
    families.extend(_job_families(job_counts))
    if not families:
        return ""
    return "\n".join(_render_family(family) for family in families) + "\n"


def _services_from(callbacks: Mapping[str, Any]
                   ) -> Optional[List[Mapping[str, Any]]]:
    inventory = callbacks.get("system_inventory")
    if inventory is None:
        return None
    try:
        services = inventory.services()
    except Exception:
        return None
    return services if isinstance(services, list) else None


def _apps_from(callbacks: Mapping[str, Any]
               ) -> Optional[List[Mapping[str, Any]]]:
    inventory = callbacks.get("workload_inventory")
    if inventory is None:
        return None
    try:
        listed = inventory.list_all()
        apps = listed.get("apps") if isinstance(listed, Mapping) else None
    except Exception:
        return None
    return apps if isinstance(apps, list) else None


def _job_counts_from(callbacks: Mapping[str, Any]
                     ) -> Optional[Mapping[str, Any]]:
    store = callbacks.get("job_store")
    if store is None:
        return None
    try:
        ledger = store.ledger()
        counts = ledger.get("counts") if isinstance(ledger, Mapping) else None
    except Exception:
        return None
    return counts if isinstance(counts, Mapping) else None


def metrics_from_callbacks(callbacks: Mapping[str, Any]) -> str:
    """Gather the snapshot and state from the API callbacks, then render.

    The snapshot is ``current_data`` — the identical source
    `/api/v2/telemetry/current` reads — so the two endpoints cannot disagree.
    Service, application and job state are each read defensively: a source that
    is not wired, or that raises, simply contributes no metrics rather than
    failing the scrape, because a scraper losing CPU temperature over a Docker
    hiccup is the wrong trade.
    """
    current = callbacks.get("current_data")
    try:
        snapshot = (current() if callable(current) else None) or {}
    except Exception:
        snapshot = {}
    return render_metrics(
        snapshot,
        services=_services_from(callbacks),
        apps=_apps_from(callbacks),
        job_counts=_job_counts_from(callbacks),
    )
