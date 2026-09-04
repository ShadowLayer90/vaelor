"""Remote-console hardware discovery and exclusive control leases."""

from __future__ import annotations

import sqlite3
import time
from contextlib import closing
from pathlib import Path
from typing import Optional

from .runtime_paths import env_value, state_path


#: Video devices that belong to the host's own graphics or ISP hardware rather
#: than to an HDMI capture card. Without the AMD and Intel entries an x86 host
#: offers its integrated video encoder as a KVM capture source.
INTERNAL_VIDEO_MARKERS = (
    "pisp", "rpivid", "rpi-hevc", "bcm2835", "codec", "image signal processor",
    "amdgpu", "radeon", "vcn", "vaapi", "i915", "intel", "nvidia", "nvenc",
    "video decoder", "video encoder",
)


def readiness_for(video: dict, hid: dict, console_ready: bool) -> dict:
    """Return one truthful readiness record for the physical KVM path."""
    video_ready = bool(video.get("stream_ready", False))
    input_ready = bool(hid.get("input_ready", False))
    ready = bool(console_ready and video_ready and input_ready)
    if ready:
        state = "ready"
        reason = "Protected video and isolated keyboard/mouse input are ready."
    elif video.get("capture_detected") or hid.get("controller_available"):
        state = "not_configured"
        reason = "Hardware was detected, but protected video and isolated input are not both commissioned."
    else:
        state = "unavailable"
        reason = "No supported capture adapter or USB device controller was detected."
    return {
        "state": state,
        "ready": ready,
        "video_ready": video_ready,
        "input_ready": input_ready,
        "reason": reason,
    }


class KvmCapabilityProbe:
    def __init__(
        self,
        video_root: str = "/sys/class/video4linux",
        udc_root: str = "/sys/class/udc",
        gadget_root: str = "/sys/kernel/config/usb_gadget",
        stream_url: Optional[str] = None,
        hid_input_path: Optional[str] = None,
    ):
        self.video_root = Path(video_root)
        self.udc_root = Path(udc_root)
        self.gadget_root = Path(gadget_root)
        self.stream_url = (
            stream_url
            if stream_url is not None
            else env_value("VAELOR_KVM_STREAM_URL", "PM_KVM_STREAM_URL", "")
        )
        self.hid_input_path = (
            hid_input_path
            if hid_input_path is not None
            else env_value(
                "VAELOR_KVM_HID_INPUT_PATH", "PM_KVM_HID_INPUT_PATH", ""
            )
        )

    @staticmethod
    def _names(root: Path, pattern: str):
        try:
            return sorted(path.name for path in root.glob(pattern))
        except OSError:
            return []

    def snapshot(self):
        video = []
        try:
            nodes = sorted(self.video_root.glob("video*"))[:64]
        except OSError:
            nodes = []
        for node in nodes:
            try:
                name = (node / "name").read_text().strip()[:160]
            except OSError:
                name = node.name
            internal = any(marker in name.lower() for marker in INTERNAL_VIDEO_MARKERS)
            video.append({"device": "/dev/{}".format(node.name), "name": name, "internal": internal})
        capture = [item for item in video if not item["internal"]]
        udcs = self._names(self.udc_root, "*")
        hid_functions = []
        try:
            hid_functions = [
                str(path) for path in self.gadget_root.glob("*/functions/hid.*")
            ][:16]
        except OSError:
            pass
        stream_ready = bool(capture and self.stream_url)
        hid_input_ready = bool(
            hid_functions
            and self.hid_input_path
            and Path(self.hid_input_path).exists()
        )
        atx_path = env_value(
            "VAELOR_KVM_ATX_POWER_PATH", "PM_KVM_ATX_POWER_PATH", ""
        )
        atx_detected = bool(atx_path and Path(atx_path).exists())
        video_result = {
            "capture_detected": bool(capture),
            "capture_devices": capture,
            "internal_devices": video,
            "stream_ready": stream_ready,
            "stream_configured": bool(self.stream_url),
        }
        hid_result = {
            "controller_available": bool(udcs),
            "controllers": udcs,
            "configured": bool(hid_functions),
            "functions": hid_functions,
            "input_ready": hid_input_ready,
        }
        console_ready = bool(stream_ready and hid_input_ready)
        return {
            "video": video_result,
            "hid": hid_result,
            "atx": {
                "detected": atx_detected,
                "actions": ["power", "reset", "long_press"] if atx_detected else [],
            },
            "virtual_media": {
                "available": False,
                "reason": "USB mass-storage isolation has not been commissioned.",
            },
            "console_ready": console_ready,
            "readiness": readiness_for(video_result, hid_result, console_ready),
            "commissioning": [
                {
                    "id": "capture",
                    "complete": stream_ready,
                    "title": "Connect a supported USB HDMI capture adapter",
                    "detail": (
                        "The capture adapter and protected video stream are ready."
                        if stream_ready
                        else (
                            "A video device is present; commission its protected "
                            "stream before control can unlock."
                            if capture
                            else "Vaelor will detect external video devices automatically."
                        )
                    ),
                },
                {
                    "id": "hid",
                    "complete": hid_input_ready,
                    "title": "Commission isolated keyboard and mouse emulation",
                    "detail": (
                        "The isolated HID gadget and guarded input bridge are ready."
                        if hid_input_ready
                        else (
                            "The HID gadget exists; commission its guarded input bridge."
                            if hid_functions
                            else (
                                "A USB device controller is available."
                                if udcs
                                else "No USB device controller is available."
                            )
                        )
                    ),
                },
                {
                    "id": "atx",
                    "complete": atx_detected,
                    "title": "Connect isolated ATX power leads",
                    "detail": "Optional; enables audited target power and reset controls.",
                },
            ],
        }


class KvmControlStore:
    def __init__(self, database_path: Optional[str] = None, lease_seconds: int = 300):
        self.database_path = database_path or env_value(
            "VAELOR_KVM_CONTROL_DB", "PM_KVM_CONTROL_DB",
            state_path("kvm/control.sqlite3"),
        )
        self.lease_seconds = max(30, min(int(lease_seconds), 900))
        Path(self.database_path).parent.mkdir(parents=True, exist_ok=True, mode=0o700)
        with closing(sqlite3.connect(self.database_path)) as connection:
            connection.execute(
                """
                CREATE TABLE IF NOT EXISTS control_lease (
                    singleton INTEGER PRIMARY KEY CHECK(singleton=1),
                    owner TEXT NOT NULL, acquired_at REAL NOT NULL,
                    expires_at REAL NOT NULL
                )
                """
            )
            connection.commit()

    def status(self):
        now = time.time()
        with closing(sqlite3.connect(self.database_path)) as connection:
            connection.row_factory = sqlite3.Row
            connection.execute("DELETE FROM control_lease WHERE expires_at<=?", (now,))
            row = connection.execute("SELECT * FROM control_lease WHERE singleton=1").fetchone()
            connection.commit()
        return dict(row) if row else {"owner": None, "acquired_at": None, "expires_at": None}

    def acquire(self, owner: str):
        now = time.time()
        with closing(sqlite3.connect(self.database_path)) as connection:
            connection.row_factory = sqlite3.Row
            connection.execute("BEGIN IMMEDIATE")  # pairs-with: sqlite-begin-immediate
            connection.execute("DELETE FROM control_lease WHERE expires_at<=?", (now,))
            row = connection.execute("SELECT owner FROM control_lease WHERE singleton=1").fetchone()
            if row and row["owner"] != owner:
                raise ValueError("Another operator currently owns keyboard and mouse control.")
            connection.execute(
                """
                INSERT INTO control_lease(singleton,owner,acquired_at,expires_at)
                VALUES(1,?,?,?)
                ON CONFLICT(singleton) DO UPDATE SET
                    owner=excluded.owner, expires_at=excluded.expires_at
                """,
                (owner, now, now + self.lease_seconds),
            )
            connection.commit()
        return self.status()

    def release(self, owner: str, force: bool = False):
        with closing(sqlite3.connect(self.database_path)) as connection:
            if force:
                cursor = connection.execute("DELETE FROM control_lease WHERE singleton=1")
            else:
                cursor = connection.execute(
                    "DELETE FROM control_lease WHERE singleton=1 AND owner=?", (owner,)
                )
            connection.commit()
        if not cursor.rowcount:
            raise ValueError("You do not own the active control session.")
        return self.status()
