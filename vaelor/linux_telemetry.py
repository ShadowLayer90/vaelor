"""Generic Linux telemetry adapter used when hardware drivers omit metrics."""

from __future__ import annotations

import shutil
import threading
import time
from pathlib import Path
from typing import Any, Callable, Optional

from .linux_sensors import (
    core_temperatures,
    cpu_temperature,
    peripheral_temperatures,
)
from .memory_ecc import memory_ecc_telemetry
from .package_power import (
    EnergySampler,
    package_power_telemetry,
    read_domains,
)
from .platforms.accelerators import (
    accelerator_telemetry,
    cached_vendor_tool_metrics,
    neural_accelerator_telemetry,
)
from .platforms.base import cpu_topology
from .platforms.graphics_software import socket_power_watts
from .wmi_sensors import wmi_sensor_telemetry


#: Filesystems that represent real storage an operator can fill up. Everything
#: else — snap loopbacks, tmpfs, overlays — is noise in a storage panel.
REAL_FILESYSTEMS = {
    "ext2", "ext3", "ext4", "xfs", "btrfs", "f2fs", "jfs", "reiserfs",
    "vfat", "exfat", "ntfs", "ntfs3", "zfs", "bcachefs",
}

#: Mount points excluded from ``disk_*`` telemetry: they are small by design
#: and a nearly-full 1 GiB EFI partition must not read as "storage critical".
EXCLUDED_MOUNT_PREFIXES = ("/boot", "/snap", "/var/snap", "/var/lib/docker")

MINIMUM_REPORTED_VOLUME_BYTES = 4 * 1024 ** 3


class LinuxTelemetryProvider:
    def __init__(
        self,
        *,
        proc_root: str = "/proc",
        sys_root: str = "/sys",
        clock: Callable[[], float] = time.monotonic,
        disk_usage: Callable[[str], Any] = shutil.disk_usage,
    ):
        self.proc_root = Path(proc_root)
        self.sys_root = Path(sys_root)
        self.clock = clock
        self.disk_usage = disk_usage
        self._lock = threading.Lock()
        self._previous_cpu: Optional[tuple[int, int]] = None
        # The last CPU-load percent computed from an advancing jiffy delta. Held
        # so a read that lands in the same scheduler tick as the previous one
        # reports the most recent real figure rather than a hole — see `_cpu`.
        self._last_cpu_percent: Optional[float] = None
        self._previous_network: Optional[tuple[float, int, int]] = None
        # Package watts are a rate, measured the same way CPU and network
        # already are: remember the last counter read and divide by the gap the
        # poll loop produced. Sleeping for an interval here would stall every
        # reader of `/telemetry/current` for the length of it.
        # Its own clock on purpose: this measures the gap between energy reads,
        # which is not the same quantity as the network rate window, and a
        # caller injecting a finite clock for one must not have it drained by
        # the other.
        self._energy = EnergySampler()
        # SMBIOS does not change without a reboot, so the ECC answer is read
        # once and held. `None` means "not asked yet", which is why the sentinel
        # is not simply an empty dict.
        self._memory_ecc: Optional[dict[str, Any]] = None

    def capabilities(self) -> dict[str, Any]:
        return {
            "id": "linux-proc-sys",
            "cpu": (self.proc_root / "stat").is_file(),
            "memory": (self.proc_root / "meminfo").is_file(),
            "thermal": (self.sys_root / "class/thermal").is_dir(),
            "labelled_sensors": (self.sys_root / "class/hwmon").is_dir(),
            "network": (self.proc_root / "net/dev").is_file(),
            "storage": (self.proc_root / "self/mounts").is_file(),
            "uptime": (self.proc_root / "stat").is_file(),
            "accelerator": bool(
                accelerator_telemetry(sys_root=str(self.sys_root))
            ),
        }

    @staticmethod
    def _read(path: Path) -> str:
        try:
            return path.read_text(encoding="utf-8", errors="replace")
        except OSError:
            return ""

    def _cpu(self) -> tuple[Optional[float], Optional[float]]:
        fields = self._read(self.proc_root / "stat").splitlines()
        cpu_line = next((line for line in fields if line.startswith("cpu ")), "")
        values: list[int] = []
        try:
            values = [int(value) for value in cpu_line.split()[1:]]
        except ValueError:
            pass
        percent = None
        if values:
            idle = sum(values[3:5])
            total = sum(values)
            # `/proc/stat` jiffies only move across a scheduler tick, so a load
            # is a delta between two reads taken far enough apart to span one.
            # This single baseline is shared by two ~1 Hz callers — the retention
            # writer (`_telemetry_write_source`) and the live-UI reader
            # (`/telemetry/current`, `/stream/telemetry`) — and when their polls
            # land in the same tick the second sees `total == previous[0]`. The
            # baseline is advanced only on a real move, so a coincident read does
            # not shrink the window the next genuine read measures over, and the
            # last real load is held rather than reported as `None`. Without this
            # the second caller blanked the Processor-load card every tick while
            # the first recorded a real figure — the field populated in retained
            # history yet read `—` live, the whole of this defect.
            with self._lock:
                previous = self._previous_cpu
                if previous is None or total > previous[0]:
                    self._previous_cpu = (total, idle)
                if previous and total > previous[0]:
                    total_delta = total - previous[0]
                    idle_delta = max(0, idle - previous[1])
                    percent = round(
                        max(0, min(100, (1 - idle_delta / total_delta) * 100)),
                        1,
                    )
                    self._last_cpu_percent = percent
                elif previous is not None:
                    # No tick since the last read: the most recent real load is
                    # still true. `None` survives only until the first delta.
                    percent = self._last_cpu_percent
        frequency = None
        raw_frequency = self._read(
            self.sys_root
            / "devices/system/cpu/cpu0/cpufreq/scaling_cur_freq"
        ).strip()
        try:
            frequency = round(float(raw_frequency) / 1000, 1)
        except ValueError:
            pass
        return percent, frequency

    def _memory(self) -> dict[str, int | float]:
        values: dict[str, int] = {}
        for line in self._read(self.proc_root / "meminfo").splitlines():
            if ":" not in line:
                continue
            name, raw = line.split(":", 1)
            try:
                values[name] = int(raw.strip().split()[0]) * 1024
            except (IndexError, ValueError):
                continue
        total = values.get("MemTotal", 0)
        available = values.get("MemAvailable", 0)
        used = max(0, total - available)
        return {
            "memory_total": total,
            "memory_used": used,
            "memory_available": available,
            "memory_percent": (
                round(used / total * 100, 1) if total else 0
            ),
        }

    def _temperature(self) -> dict[str, Any]:
        """Prefer a labelled CPU sensor over the hottest thermal zone."""
        return cpu_temperature(str(self.sys_root))

    def _boot_time(self) -> Optional[int]:
        """Epoch seconds of the last boot, for uptime.

        Nothing emitted this, so the Overview uptime field read ``—`` forever.
        """
        for line in self._read(self.proc_root / "stat").splitlines():
            if line.startswith("btime "):
                try:
                    return int(line.split()[1])
                except (IndexError, ValueError):
                    break
        raw = self._read(self.proc_root / "uptime").split()
        if raw:
            try:
                return int(time.time() - float(raw[0]))
            except ValueError:
                pass
        return None

    @staticmethod
    def _volume_key(mount_point: str) -> str:
        if mount_point == "/":
            return "root"
        cleaned = mount_point.strip("/").replace("/", "_").replace("-", "_")
        cleaned = "".join(
            character if character.isalnum() or character == "_" else "_"
            for character in cleaned
        )
        return cleaned.lower() or "root"

    def _mount_points(self) -> list[str]:
        mounts = []
        for line in self._read(self.proc_root / "self/mounts").splitlines():
            fields = line.split()
            if len(fields) < 3 or fields[2] not in REAL_FILESYSTEMS:
                continue
            mount_point = fields[1].replace("\\040", " ")
            if mount_point.startswith(EXCLUDED_MOUNT_PREFIXES):
                continue
            if mount_point not in mounts:
                mounts.append(mount_point)
        return mounts

    def _storage(self) -> dict[str, Any]:
        """``disk_<volume>_{percent,used,total}`` for each real filesystem."""
        metrics: dict[str, Any] = {}
        for mount_point in self._mount_points():
            try:
                usage = self.disk_usage(mount_point)
            except OSError:
                continue
            total = int(getattr(usage, "total", 0) or 0)
            if total < MINIMUM_REPORTED_VOLUME_BYTES:
                continue
            used = int(getattr(usage, "used", 0) or 0)
            key = self._volume_key(mount_point)
            metrics["disk_{}_total".format(key)] = total
            metrics["disk_{}_used".format(key)] = used
            metrics["disk_{}_percent".format(key)] = round(used / total * 100, 1)
            metrics["disk_{}_mount".format(key)] = mount_point
        return self._with_primary_volume(metrics)

    @staticmethod
    def _with_primary_volume(metrics: dict[str, Any]) -> dict[str, Any]:
        """Add one canonical volume whose three numbers cannot disagree.

        The sidebar showed 8% beside "30.7 GB of 251.7 GB", which is 12%, and
        the Z2 showed 20% beside a ratio of 25%. Two surfaces were reading two
        sources for one figure. This publishes a single record - mount, used,
        total, and a percentage *derived from the used and total in the same
        record* - so a reader that takes all three from here cannot produce a
        contradiction, and one that takes them from anywhere else is visibly
        doing something else.

        Root is the primary volume when present, because that is the one the
        summary surfaces mean; otherwise the largest by capacity.
        """
        volumes = []
        for key, value in metrics.items():
            if not (str(key).startswith("disk_") and str(key).endswith("_total")):
                continue
            name = str(key)[len("disk_"):-len("_total")]
            total = int(metrics.get("disk_{}_total".format(name)) or 0)
            used = int(metrics.get("disk_{}_used".format(name)) or 0)
            if total <= 0:
                continue
            volumes.append((name, used, total))
        if not volumes:
            return metrics
        primary = next(
            (item for item in volumes if item[0] == "root"),
            max(volumes, key=lambda item: item[2]),
        )
        name, used, total = primary
        metrics["disk_primary_volume"] = name
        metrics["disk_primary_mount"] = metrics.get("disk_{}_mount".format(name))
        metrics["disk_primary_used"] = used
        metrics["disk_primary_total"] = total
        metrics["disk_primary_percent"] = round(used / total * 100, 1)
        return metrics

    def _network(self) -> tuple[float, float]:
        received = 0
        sent = 0
        for line in self._read(self.proc_root / "net/dev").splitlines()[2:]:
            if ":" not in line:
                continue
            name, raw = line.split(":", 1)
            if name.strip() == "lo":
                continue
            fields = raw.split()
            try:
                received += int(fields[0])
                sent += int(fields[8])
            except (IndexError, ValueError):
                continue
        now = self.clock()
        download = 0.0
        upload = 0.0
        with self._lock:
            previous = self._previous_network
            self._previous_network = (now, received, sent)
        if previous and now > previous[0]:
            elapsed = now - previous[0]
            download = max(0, received - previous[1]) / elapsed
            upload = max(0, sent - previous[2]) / elapsed
        return round(download, 1), round(upload, 1)

    def _ecc(self) -> dict[str, Any]:
        """The ECC answer, read once and held for the life of the process.

        Tried unprivileged first so a root-run development host and the test
        fixtures answer without a bridge at all. The SMBIOS memory records are
        mode 0400 root on a stock install, so on the appliance itself that
        attempt returns "unreported" and the privileged side is asked — which
        is the whole point. What must never happen is the unprivileged failure
        being published as "this memory has no ECC".
        """
        from .memory_ecc import memory_ecc

        if self._memory_ecc is not None:
            return self._memory_ecc
        reading = memory_ecc(str(self.sys_root))
        if reading.get("state") == "unreported":
            reading = self._privileged_ecc() or reading
        self._memory_ecc = reading
        return reading

    @staticmethod
    def _socket_power() -> Optional[float]:
        """Whole-socket power from the shared ``amd-smi`` snapshot, or ``None``.

        The GPU and NPU cards read this same cached snapshot, so taking the
        Processor card's package figure from it too makes all three power cards
        agree at one instant instead of comparing amd-smi against RAPL sampled
        on a different clock. ``None`` on any host without amd-smi — the Pi
        included — which leaves the RAPL fallback, and that hardware, unchanged.
        """
        metrics = cached_vendor_tool_metrics()
        if not metrics.get("available"):
            return None
        return socket_power_watts(metrics.get("metrics"))

    @staticmethod
    def _privileged_ecc() -> Optional[dict[str, Any]]:
        from .hardware_bridge import HardwareBridgeClient, HardwareBridgeError

        client = HardwareBridgeClient()
        if not client.available:
            return None
        try:
            answer = client.memory_ecc()
        except (HardwareBridgeError, OSError, RuntimeError, ValueError):
            return None
        return answer if answer.get("sources") else None

    def snapshot(
        self, metrics: Optional[dict[str, Any]] = None
    ) -> dict[str, Any]:
        merged = dict(metrics or {})
        cpu_percent, frequency = self._cpu()
        memory = self._memory()
        temperature = self._temperature()
        download, upload = self._network()
        peripherals = peripheral_temperatures(str(self.sys_root))
        cores = core_temperatures(str(self.sys_root))
        fallbacks = {
            "cpu_percent": cpu_percent,
            "cpu_freq": frequency,
            "cpu_temperature": temperature["celsius"],
            "cpu_temperature_source": temperature["source"],
            "cpu_temperature_labelled": (
                temperature["labelled"] if temperature["celsius"] is not None else None
            ),
            "cpu_core_temperatures": cores or None,
            "storage_temperatures": peripherals["storage"] or None,
            "network_temperatures": peripherals["network"] or None,
            "boot_time": self._boot_time(),
            "network_download_speed": download,
            "network_upload_speed": upload,
            **self._storage(),
            # Accelerator keys are omitted entirely when nothing is present.
            # A missing sensor is not a reading of zero.
            **accelerator_telemetry(sys_root=str(self.sys_root)),
            # The NPU is the one reading on this strip that needs a vendor
            # tool: the kernel publishes no utilisation for it. Absent when
            # amd-smi is missing, fails, or prints N/A - never zeroed, because
            # a zero here reads as an idle accelerator and cannot be told apart
            # from one.
            **neural_accelerator_telemetry(sys_root=str(self.sys_root)),
            # Fans and board temperatures from `hp_wmi_sensors`, which ships
            # with the distribution and was never loaded. Omitted entirely when
            # the module is absent - the reason travels separately so a surface
            # can explain the gap rather than showing nothing.
            # Package watts. The counters are root-only, which is a reason to
            # ask the privileged side rather than to report the machine as
            # unreadable — the card that said Vaelor could not read them was
            # describing a route Vaelor had already built.
            **package_power_telemetry(
                self._energy,
                reader=lambda: read_domains(
                    str(self.sys_root / "class" / "powercap")
                ),
                socket_watts=self._socket_power(),
            ),
            # Whether the memory corrects errors, and on what evidence. Static,
            # so it is read once; published every poll so a surface never has
            # to hard-code the answer.
            **memory_ecc_telemetry(self._ecc()),
            **wmi_sensor_telemetry(
                sys_root=str(self.sys_root),
                # The on-die reading, so the firmware CPU channel can be
                # cross-checked against it rather than either being picked as
                # the winner. They tracked to within 1.4 °C across a 29 °C
                # swing; a later divergence is itself a signal.
                die_celsius=temperature["celsius"],
            ),
            **memory,
        }
        for key, value in fallbacks.items():
            if key not in merged and value is not None:
                merged[key] = value
        # `os.cpu_count()` counts **logical** CPUs, so on an SMT part this key
        # published threads under the name `cpu_cores` — 32 on a 16-core Z2,
        # printed to the Assistant as "32 cores" in every answer about the
        # machine. It went unnoticed for as long as it did because the Pi has
        # four cores and no SMT, where the wrong field holds the right number.
        topology = cpu_topology(
            str(self.proc_root / "cpuinfo"), str(self.sys_root)
        )
        merged.setdefault("cpu_cores", topology["cores"] or topology["logical"])
        merged.setdefault("cpu_threads", topology["logical"])
        merged.setdefault("cpu_cores_are_threads", topology["cores"] is None)
        if topology["threads_per_core"] is not None:
            merged.setdefault("cpu_threads_per_core", topology["threads_per_core"])
        merged.setdefault("cpu_topology_source", topology["source"])
        return merged
