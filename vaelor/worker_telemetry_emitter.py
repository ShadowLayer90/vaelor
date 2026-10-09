"""The lean per-worker telemetry emitter the controller installs (Phase E2b).

Telegraf runs this once per interval as an ``inputs.exec`` command on a lean
cluster worker, and it prints **one** InfluxDB line-protocol line::

    history,node=<node_id> cpu_percent=3.4,memory_percent=41.2,...

There is exactly one tag, ``node=<node_id>`` — the ingest route 400s any other
tag (`telemetry_ingest._parse_line`), so the emitter never writes one — and only
fields from :data:`telemetry_ingest.ACCEPTED_FIELDS` appear.

**No drift.** The host fields are read through Vaelor's own canonical reader
(`LinuxTelemetryProvider` and `platforms.accelerators`) and flattened through the
same `telemetry_flatten` the controller's store writer uses, so a worker's row is
computed byte-for-byte the way the controller computes its own. **No fabrication.**
A field whose source is absent — no GPU, no readable temperature — is *omitted*,
never emitted as a zero (a zero for an unreadable GPU temperature is how a
dashboard ends up claiming a running GPU is at 0 °C).

**No third party, no venv.** Nothing in this module's import closure reaches
outside the standard library, so the bundle (`worker_telemetry_bundle`) runs on
the worker's system ``python3`` with no dependencies installed. In particular it
does **not** import `data_logger` (which pulls ``influxdb``); the flatten it
shares with the store lives in the dependency-free `telemetry_flatten`.

**No amd-smi, and no subprocess at all.** The GPU utilisation, GTT bytes and
edge temperature come from sysfs. The graphics engine's own power and
temperature come only from ``amd-smi``, which this process never runs: a
separate non-root service (`worker_gpu_sampler`) runs it and leaves its reading
in a small file, and the emitter reads that file through the checks in
`gpu_vendor_sample` (VD-147). Every row carries ``gpu_vendor_status``, the code
saying whether that reading was taken and, if not, why: a missing or stale file
is "the sampler has stopped or is missing", and a machine with no AMD
integrated GPU is "no graphics readings apply", decided here from the same
discovery the controller uses.

**Fresh process each interval.** Telegraf starts this anew every second, so the
CPU-load delta cannot be carried across runs the way the long-lived controller
provider carries it. The provider is instead sampled twice within this one run,
around a short sleep, which is exactly the two-read delta `_cpu` computes; the
sleep is injectable so a test can advance a fake ``/proc/stat`` between the reads.

**Disk, network and fans (VD-205 item 6).** The Assistant answers about a
worker's storage, traffic and cooling from these, so each follows the same
absent-never-zero rule:

* ``disk_root_*_bytes`` is the filesystem at ``/``; ``disk_data_*_bytes`` the
  one holding ``/var/lib`` (where Docker and Vaelor keep their data), written
  only when ``/var/lib`` sits on a different filesystem (its ``st_dev``
  differs), so one disk is never reported twice. Free is ``f_bavail``, what an
  unprivileged writer may still use - the controller's own definition
  (`linux_telemetry.filesystem_bytes`).
* ``net_{rx,tx}_bytes_per_second`` sums the physical NICs only
  (`linux_telemetry.is_physical_interface`). A rate needs two counter reads, and
  this process lives for one interval, so the counters are read on either side
  of the same short sleep the CPU delta already takes and divided by the
  monotonic time between them. No state file: nothing outlives the run, so a
  stale baseline cannot be divided by a wrong interval. A wrapped or reset
  counter, or an interface that came or went mid-sample, yields no rate.
* ``fan_rpm`` is the fastest readable fan and ``fan_count`` how many fans gave
  a reading (`linux_sensors.readable_fan_speeds`); both are absent when none
  did. A readable 0 RPM is a stopped fan and is kept.
"""

from __future__ import annotations

import argparse
import os
import shutil
import sys
import time
from typing import Any, Callable, Dict, Optional

from .gpu_vendor_sample import read_sample_file, vendor_fields
from .gpu_vendor_status import NOT_APPLICABLE, STATUS_FIELD
from .linux_sensors import readable_fan_speeds
from .linux_telemetry import (
    LinuxTelemetryProvider,
    filesystem_bytes,
    network_rates,
    physical_interface_counters,
)
from .platforms.accelerators import accelerator_telemetry, discover_accelerators
from .telemetry_bounds import DISCARDED_COUNT_FIELD
from .telemetry_flatten import flatten_storable
from .telemetry_ingest import ACCEPTED_FIELDS, INGEST_MEASUREMENT, NODE_TAG_KEY, valid_node_id

#: Seconds between the two ``/proc/stat`` reads the CPU-load delta needs. Long
#: enough to span several scheduler ticks (jiffies advance at USER_HZ, typically
#: 100 Hz) so the second read sees a real delta, short enough that a once-a-second
#: Telegraf exec is not delayed noticeably.
DEFAULT_SAMPLE_SECONDS = 0.4

#: A finder and a runner that guarantee the accelerator reader spawns no
#: ``amd-smi`` on the poll path. GPU utilisation, GTT bytes and the edge
#: temperature are sysfs reads that need no vendor tool; the vendor readings
#: come from the sampler's file, handed to the reader as its ``vendor`` source,
#: so the reader never reaches for the tool. The finder resolves nothing on
#: PATH and the runner refuses outright, so even a code path that did reach for
#: it could not start a process once a second.
#: The filesystem every Linux machine has, reported as ``disk_root_*``.
ROOT_FILESYSTEM_PATH = "/"

#: Where Docker (``/var/lib/docker``) and Vaelor keep their data, reported as
#: ``disk_data_*`` only when it is a separate filesystem from the root.
DATA_FILESYSTEM_PATH = "/var/lib"


def _device_of(path: str) -> int:
    return os.stat(path).st_dev


def _no_executable(_name: str) -> Optional[str]:
    return None


def _no_subprocess(*_args, **_kwargs):
    raise OSError("The worker telemetry emitter never spawns a subprocess.")


def read_cluster_fields(
    *,
    proc_root: str = "/proc",
    sys_root: str = "/sys",
    sample_seconds: float = DEFAULT_SAMPLE_SECONDS,
    sleeper: Callable[[float], Any] = time.sleep,
    provider_factory: Callable[..., LinuxTelemetryProvider] = LinuxTelemetryProvider,
    sample_reader: Callable[..., Dict[str, Any]] = read_sample_file,
    clock: Callable[[], float] = time.time,
    disk_usage: Callable[[str], Any] = shutil.disk_usage,
    device_of: Callable[[str], int] = _device_of,
    monotonic: Callable[[], float] = time.monotonic,
) -> Dict[str, float]:
    """The subset of :data:`ACCEPTED_FIELDS` this host can actually read.

    Every value is produced by the canonical reader and run through the store's
    own flatten (which applies the one bounds table), then filtered to the
    accepted vocabulary. A field with no source is simply absent from the
    result — the caller emits only what is here. ``sample_reader`` reads the
    GPU sampler's file; it is called only on a machine with an AMD integrated
    GPU.
    """
    provider = provider_factory(proc_root=proc_root, sys_root=sys_root)
    # Prime the CPU baseline, let the kernel's jiffies advance, then read the
    # delta — the exact computation `_cpu` performs across two ~1 Hz polls on the
    # controller, done inside this one short-lived process.
    # The network counters are read on either side of the same sleep, timed by
    # the monotonic clock, so their rate is a delta over a measured interval.
    provider._cpu()
    counters_before = physical_interface_counters(proc_root, sys_root)
    started = monotonic()
    sleeper(sample_seconds)
    cpu_percent, _frequency = provider._cpu()
    counters_after = physical_interface_counters(proc_root, sys_root)
    elapsed = monotonic() - started

    reading: Dict[str, Any] = {"cpu_percent": cpu_percent}
    reading.update(network_rates(counters_before, counters_after, elapsed))
    reading.update(_disk_fields(disk_usage, device_of))
    reading.update(_fan_fields(sys_root))
    reading.update(provider._memory())
    reading["cpu_temperature"] = provider._temperature()["celsius"]
    # sysfs-only accelerator read. On an AMD integrated GPU the vendor readings
    # (and their status code) come from the sampler's file; on any other GPU
    # the reader says no graphics readings apply; with no GPU at all the
    # emitter says the same, so every row carries the code.
    records = discover_accelerators(sys_root)
    reading.update(
        accelerator_telemetry(
            accelerators=records,
            sys_root=sys_root,
            finder=_no_executable,
            runner=_no_subprocess,
            vendor=lambda: _vendor_or_missing(sample_reader, clock),
        )
    )
    if not records:
        reading[STATUS_FIELD] = NOT_APPLICABLE
    dropped: list = []
    flat = flatten_storable(reading, dropped=dropped)
    fields = {key: flat[key] for key in ACCEPTED_FIELDS if key in flat}
    if dropped:
        # Told to the controller, which counts it for this machine's card
        # (pass-3 review): a discard here is otherwise a silent gap there.
        fields[DISCARDED_COUNT_FIELD] = float(len(dropped))
    return fields


def _disk_fields(
    disk_usage: Callable[[str], Any], device_of: Callable[[str], int],
) -> Dict[str, int]:
    """``disk_root_*_bytes``, plus ``disk_data_*_bytes`` for a separate data disk.

    An unreadable filesystem contributes nothing. The data filesystem is
    reported only when both paths' devices were read and differ: an unknown
    device cannot be shown to be a second disk.
    """
    fields: Dict[str, int] = {}
    root = filesystem_bytes(ROOT_FILESYSTEM_PATH, disk_usage=disk_usage)
    for key, value in (root or {}).items():
        fields["disk_root_{}".format(key)] = value
    try:
        separate = device_of(DATA_FILESYSTEM_PATH) != device_of(ROOT_FILESYSTEM_PATH)
    except OSError:
        separate = False
    if separate:
        data = filesystem_bytes(DATA_FILESYSTEM_PATH, disk_usage=disk_usage)
        for key, value in (data or {}).items():
            fields["disk_data_{}".format(key)] = value
    return fields


def _fan_fields(sys_root: str) -> Dict[str, int]:
    """``fan_rpm`` and ``fan_count``, or nothing when no fan gave a reading."""
    speeds = readable_fan_speeds(sys_root)
    if not speeds:
        return {}
    return {"fan_rpm": max(speeds), "fan_count": len(speeds)}


def _vendor_or_missing(
    sample_reader: Callable[..., Dict[str, Any]], clock: Callable[[], float],
) -> Dict[str, float]:
    """The sampler's fields, or the missing-sampler code when reading them failed.

    A vendor read that raises must never cost the host its CPU, memory and GPU
    fields (review B2): the whole row would be lost every second.
    """
    try:
        return vendor_fields(sample_reader(now=clock()))
    except Exception:  # noqa: BLE001 - the host's own fields must survive any vendor failure
        return vendor_fields(None)


def _format_field(value: float) -> str:
    """One field value as a line-protocol float token.

    ``repr`` gives the shortest round-tripping decimal, and every value here is
    already a finite float (`telemetry_flatten.storable` dropped non-finites and
    coerced ints to float), so the token always matches the ingest reader's float
    grammar and re-parses to the same number. Fields are written as floats
    because the controller's own store writes them as floats; a worker emitting
    an integer for the same field would be an InfluxDB per-measurement type
    conflict that drops the row.
    """
    return repr(float(value))


def render_line(node_id: str, fields: Dict[str, float]) -> str:
    """The one InfluxDB line for ``node_id`` and ``fields``, or ``""`` if empty.

    The only tag is ``node=<node_id>``. Fields are emitted in a fixed order so
    the output is deterministic. An empty field set yields an empty string — a
    line with no fields is not valid line protocol, and emitting nothing is the
    honest way to say "this interval had nothing readable" rather than inventing
    a row. No timestamp is written: Telegraf stamps the metric at collection
    time, and the ingest route stamps a timeless line with the server clock, so
    the worker never dates a row itself.
    """
    if not valid_node_id(node_id):
        raise ValueError("The node id is not a well-formed cluster node id.")
    present = {key: fields[key] for key in sorted(ACCEPTED_FIELDS | {DISCARDED_COUNT_FIELD}) if key in fields}
    if not present:
        return ""
    body = ",".join("{}={}".format(key, _format_field(value)) for key, value in present.items())
    return "{},{}={} {}".format(INGEST_MEASUREMENT, NODE_TAG_KEY, node_id, body)


def emit(
    node_id: str,
    *,
    proc_root: str = "/proc",
    sys_root: str = "/sys",
    sample_seconds: float = DEFAULT_SAMPLE_SECONDS,
    sleeper: Callable[[float], Any] = time.sleep,
    provider_factory: Callable[..., LinuxTelemetryProvider] = LinuxTelemetryProvider,
    disk_usage: Callable[[str], Any] = shutil.disk_usage,
    device_of: Callable[[str], int] = _device_of,
) -> str:
    """Read this host and render its one line for ``node_id`` (possibly empty)."""
    fields = read_cluster_fields(
        proc_root=proc_root, sys_root=sys_root, sample_seconds=sample_seconds,
        sleeper=sleeper, provider_factory=provider_factory,
        disk_usage=disk_usage, device_of=device_of,
    )
    return render_line(node_id, fields)


def main(argv: Optional[list] = None) -> int:
    """Print one line for the ``--node`` id. Empty output is a normal interval."""
    parser = argparse.ArgumentParser(description="Vaelor worker telemetry emitter.")
    parser.add_argument("--node", required=True, help="This node's cluster id.")
    parser.add_argument("--proc-root", default="/proc")
    parser.add_argument("--sys-root", default="/sys")
    parser.add_argument("--interval", type=float, default=DEFAULT_SAMPLE_SECONDS)
    options = parser.parse_args(argv)
    line = emit(
        options.node, proc_root=options.proc_root, sys_root=options.sys_root,
        sample_seconds=options.interval,
    )
    if line:
        sys.stdout.write(line + "\n")
    return 0


if __name__ == "__main__":
    sys.exit(main())
