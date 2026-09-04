"""Linux block-storage adapter with bounded, media-aware inventory."""

from __future__ import annotations

import json
import shutil
import subprocess
from pathlib import Path
from typing import Any, Callable


MEDIA_KINDS = ("microsd", "nvme", "usb", "sata", "other")


class LinuxStorageProvider:
    def __init__(
        self,
        runner: Callable[..., Any] = subprocess.run,
        finder: Callable[[str], str | None] | None = None,
        disk_usage: Callable[[str], Any] | None = None,
    ):
        self.runner = runner
        self.finder = finder or shutil.which
        self.disk_usage = disk_usage or shutil.disk_usage

    def capabilities(self) -> dict[str, Any]:
        return {
            "id": "linux-block",
            "media": ["microSD", "NVMe", "USB", "SATA"],
            "inventory": self.finder("lsblk") is not None,
            "health": self.finder("smartctl") is not None,
        }

    def _command(self, command: list[str], timeout: int = 5):
        try:
            return self.runner(
                command,
                capture_output=True,
                text=True,
                check=False,
                timeout=timeout,
            )
        except (OSError, subprocess.SubprocessError):
            return type(
                "Result", (), {"returncode": 1, "stdout": "", "stderr": ""}
            )()

    @staticmethod
    def _media_kind(device: dict[str, Any], inherited: str = "other") -> str:
        name = str(device.get("name", "")).lower()
        path = str(device.get("path", "")).lower()
        transport = str(device.get("tran", "")).lower()
        if transport == "nvme" or name.startswith("nvme"):
            return "nvme"
        if transport == "usb":
            return "usb"
        if name.startswith("mmcblk") or path.startswith("/dev/mmcblk"):
            return "microsd"
        if transport in {"sata", "ata"}:
            return "sata"
        return inherited

    def inventory(self) -> dict[str, Any]:
        lsblk = self.finder("lsblk")
        if not lsblk:
            return {
                "devices": [],
                "volumes": [],
                "media_counts": {},
                "media_presence": {
                    kind: {"present": False, "count": 0}
                    for kind in MEDIA_KINDS
                },
                "nvme_temperatures": {},
                "smartctl_available": False,
            }
        result = self._command([
            lsblk,
            "-J",
            "-b",
            "-o",
            (
                "NAME,PATH,TYPE,SIZE,FSTYPE,MOUNTPOINTS,MODEL,SERIAL,"
                "ROTA,TRAN,RM,HOTPLUG"
            ),
        ])
        try:
            devices = json.loads(result.stdout).get("blockdevices", [])
        except (json.JSONDecodeError, AttributeError):
            devices = []
        temperatures: dict[str, float] = {}
        for sensor in Path("/sys/class/nvme").glob(
            "nvme*/device/hwmon/hwmon*/temp1_input"
        ):
            try:
                temperatures[sensor.parts[-4]] = round(
                    int(sensor.read_text().strip()) / 1000, 1
                )
            except (OSError, ValueError):
                pass
        volumes: list[dict[str, Any]] = []
        seen_mounts: set[str] = set()

        def collect(
            items,
            inherited_kind: str = "other",
            inherited_model: str = "",
            root_device: str = "",
        ) -> None:
            for device in items or []:
                kind = self._media_kind(device, inherited_kind)
                model = str(
                    device.get("model")
                    or inherited_model
                    or device.get("name")
                    or ""
                ).strip()
                device_type = str(device.get("type") or "").lower()
                physical_device = root_device or str(
                    device.get("path") or device.get("name") or ""
                )
                raw_mountpoints = device.get("mountpoints") or []
                if isinstance(raw_mountpoints, str):
                    raw_mountpoints = [raw_mountpoints]
                filesystem = str(device.get("fstype") or "").lower()
                # One volume per real filesystem, not one per mountpoint. The
                # same partition mounted at several points - the Pi's microSD
                # root, bind-mounted for containers - is one real disk, and the
                # live box listed `/dev/mmcblk0p2` FOUR times, each at the full
                # 251 GB. Any consumer that summed volumes (the Assistant among
                # them) then over-counted a single card four-fold (#205, VD-089
                # neighbourhood; verified against `df`, which reports one
                # filesystem). Dedupe at the source so every consumer sees one.
                valid_mountpoints: list[str] = []
                for raw_mountpoint in raw_mountpoints:
                    mountpoint = str(raw_mountpoint or "").strip()
                    if (
                        not mountpoint
                        or mountpoint in seen_mounts
                        or mountpoint == "[SWAP]"
                        or mountpoint.startswith("/snap/")
                        or device_type in {"loop", "rom", "zram"}
                        or filesystem
                        in {"squashfs", "overlay", "tmpfs", "devtmpfs"}
                    ):
                        continue
                    valid_mountpoints.append(mountpoint)
                if valid_mountpoints:
                    # Root is the mountpoint a reader means by "the disk"; failing
                    # that the shortest path, which is the mount the others hang
                    # off. The full set travels in `mountpoints` so nothing the
                    # box actually mounts is hidden by the dedupe.
                    primary = next(
                        (m for m in valid_mountpoints if m == "/"),
                        min(valid_mountpoints, key=len),
                    )
                    for mountpoint in valid_mountpoints:
                        seen_mounts.add(mountpoint)
                    try:
                        usage = self.disk_usage(primary)
                    except OSError:
                        usage = None
                    if usage is not None:
                        percent = (
                            round((usage.used / usage.total) * 100, 1)
                            if usage.total
                            else 0
                        )
                        volumes.append({
                            "id": (
                                f"{device.get('path') or device.get('name')}:"
                                f"{primary}"
                            ),
                            "device_id": physical_device,
                            "name": str(device.get("name") or ""),
                            "path": str(device.get("path") or ""),
                            "model": model,
                            "kind": kind,
                            "transport": str(device.get("tran") or ""),
                            "filesystem": filesystem,
                            "mountpoint": primary,
                            # Every place this one filesystem is mounted, so the
                            # dedupe hides nothing: a reader can still see the
                            # bind mounts, they are just not four separate disks.
                            "mountpoints": valid_mountpoints,
                            "total_bytes": usage.total,
                            "used_bytes": usage.used,
                            "free_bytes": usage.free,
                            # `free` and `total - used` are different quantities
                            # and both are correct: `statvfs` reports `f_bavail`,
                            # what an unprivileged process may still write, while
                            # `total - used` is `f_bfree`, which also counts the
                            # filesystem's reserved blocks. A live tester read
                            # 220.7 GB free off one surface and 231.3 GB off
                            # another for the same volume in the same second, and
                            # nothing in the payload said the two were answering
                            # different questions. The difference is now a field,
                            # so `used + free + reserved == total` holds and any
                            # surface can say which "free" it means.
                            "reserved_bytes": max(
                                0,
                                int(usage.total)
                                - int(usage.used)
                                - int(usage.free),
                            ),
                            "used_percent": percent,
                            "removable": bool(device.get("rm")),
                            "hotplug": bool(device.get("hotplug")),
                        })
                collect(
                    device.get("children") or [],
                    kind,
                    model,
                    physical_device,
                )

        collect(devices)
        media_counts: dict[str, int] = {}
        for device_id, kind in {
            (item["device_id"], item["kind"]) for item in volumes
        }:
            del device_id
            media_counts[kind] = media_counts.get(kind, 0) + 1
        return {
            "devices": devices[:32],
            "volumes": sorted(
                volumes, key=lambda item: (item["kind"], item["mountpoint"])
            )[:64],
            "media_counts": media_counts,
            "media_presence": {
                kind: {
                    "present": media_counts.get(kind, 0) > 0,
                    "count": media_counts.get(kind, 0),
                }
                for kind in MEDIA_KINDS
            },
            "nvme_temperatures": temperatures,
            "smartctl_available": self.finder("smartctl") is not None,
        }
