"""NVMe SMART health, read through the kernel's admin-passthrough ioctl.

No ``nvme-cli`` dependency: the ioctl is the same interface that tool uses, and
requiring a package to be installed before an appliance can tell its owner the
drive is wearing out is a dependency on a fact about the host rather than about
the hardware.

The value that prompted this: **65 unsafe shutdowns against 145 power cycles**
on the target machine. Nearly half of this appliance's power-offs did not flush
the drive, which is exactly the kind of thing a product that lives on a shelf
should be saying out loud rather than leaving in a log nobody reads.

Requires ``CAP_SYS_ADMIN``, so it runs on the privileged side — the hardware
bridge — and the control plane asks it there. Reporting "SMART unavailable"
from an unprivileged reader would describe the reader, not the drive.
"""

from __future__ import annotations

import ctypes
import glob
import os
from typing import Any, Callable, Dict, List, Optional

try:  # Linux only. Absent on a developer workstation, and on a Pi it is there.
    import fcntl
except ImportError:  # pragma: no cover - exercised by not being Linux
    fcntl = None  # type: ignore[assignment]


#: ``NVME_IOCTL_ADMIN_CMD``: ``_IOWR('N', 0x41, struct nvme_admin_cmd)``, and
#: the struct is 72 bytes.
NVME_ADMIN_COMMAND = 0xC0484E41

#: Admin opcode 0x02 is Get Log Page; log id 0x02 is the SMART / Health
#: Information log, which is 512 bytes.
GET_LOG_PAGE = 0x02
SMART_LOG_ID = 0x02
SMART_LOG_BYTES = 512

#: ``nsid`` 0xFFFFFFFF asks for controller-wide statistics rather than one
#: namespace's.
WHOLE_CONTROLLER = 0xFFFFFFFF

#: Critical-warning bits from the NVMe specification, in the order they appear.
CRITICAL_WARNINGS = (
    (0x01, "spare capacity is below its threshold"),
    (0x02, "temperature is outside its operating range"),
    (0x04, "the drive's reliability is degraded"),
    (0x08, "the media is read-only"),
    (0x10, "the volatile memory backup device has failed"),
    (0x20, "persistent memory has become read-only"),
)

#: Above this share of power-offs being unsafe, an owner should be told. Not a
#: drive fault - it is usually the machine losing power rather than shutting
#: down - but it is the thing that corrupts filesystems, and 45% of power-offs
#: on the measured machine were unsafe.
UNSAFE_SHUTDOWN_WARNING_RATIO = 0.10


class _AdminCommand(ctypes.Structure):
    """``struct nvme_admin_cmd`` from ``include/uapi/linux/nvme_ioctl.h``."""

    _pack_ = 1
    _fields_ = [
        ("opcode", ctypes.c_uint8),
        ("flags", ctypes.c_uint8),
        ("rsvd1", ctypes.c_uint16),
        ("nsid", ctypes.c_uint32),
        ("cdw2", ctypes.c_uint32),
        ("cdw3", ctypes.c_uint32),
        ("metadata", ctypes.c_uint64),
        ("addr", ctypes.c_uint64),
        ("metadata_len", ctypes.c_uint32),
        ("data_len", ctypes.c_uint32),
        ("cdw10", ctypes.c_uint32),
        ("cdw11", ctypes.c_uint32),
        ("cdw12", ctypes.c_uint32),
        ("cdw13", ctypes.c_uint32),
        ("cdw14", ctypes.c_uint32),
        ("cdw15", ctypes.c_uint32),
        ("timeout_ms", ctypes.c_uint32),
        ("result", ctypes.c_uint32),
    ]


def controllers(dev_root: str = "/dev") -> List[str]:
    """NVMe controller character devices on this host."""
    return sorted(glob.glob(os.path.join(dev_root, "nvme[0-9]*")))[:8]


def _integer(buffer: bytes, offset: int, length: int) -> int:
    """A little-endian unsigned integer from the SMART log."""
    return int.from_bytes(buffer[offset:offset + length], "little")


def parse_smart_log(buffer: bytes) -> Dict[str, Any]:
    """Decode the fields worth reporting from a 512-byte SMART log.

    Offsets are from the NVMe base specification, "SMART / Health Information
    Log Page". The 128-bit counters are read whole rather than truncated to
    their low 64 bits, because a drive that has outlived a 64-bit counter is
    precisely the one whose numbers must not silently wrap.
    """
    if len(buffer) < SMART_LOG_BYTES:
        raise ValueError("The SMART log was shorter than the specified 512 bytes.")
    warning_bits = buffer[0]
    kelvin = _integer(buffer, 1, 2)
    return {
        "critical_warnings": [
            text for bit, text in CRITICAL_WARNINGS if warning_bits & bit
        ],
        "critical_warning_bits": warning_bits,
        # The composite temperature is reported in kelvin. 0 means the drive
        # did not supply one, which is not -273 °C.
        "composite_temperature_c": (
            round(kelvin - 273.15, 1) if kelvin else None
        ),
        "available_spare_percent": buffer[3],
        "available_spare_threshold_percent": buffer[4],
        # "Percentage used" is the wear estimate: 0 is a new drive and it may
        # exceed 100 on one past its rated endurance.
        "wear_percent": buffer[5],
        "data_units_read": _integer(buffer, 32, 16),
        "data_units_written": _integer(buffer, 48, 16),
        "power_cycles": _integer(buffer, 112, 16),
        "power_on_hours": _integer(buffer, 128, 16),
        "unsafe_shutdowns": _integer(buffer, 144, 16),
        "media_errors": _integer(buffer, 160, 16),
        "error_log_entries": _integer(buffer, 176, 16),
    }


def read_smart_log(
    device: str,
    opener: Callable[..., int] = os.open,
    control: Optional[Callable[..., int]] = None,
    closer: Callable[[int], None] = os.close,
) -> bytes:
    """Fetch the raw SMART log from one controller.

    The seams are injected so this is testable without a drive and without
    root: the ioctl is the only part that cannot be exercised in a unit test,
    and it is the part least likely to be wrong.
    """
    ioctl = control or (fcntl.ioctl if fcntl is not None else None)
    if ioctl is None:
        raise OSError("NVMe admin passthrough needs a Linux ioctl interface.")
    buffer = ctypes.create_string_buffer(SMART_LOG_BYTES)
    command = _AdminCommand(
        opcode=GET_LOG_PAGE,
        nsid=WHOLE_CONTROLLER,
        addr=ctypes.addressof(buffer),
        data_len=SMART_LOG_BYTES,
        # cdw10 packs the log id in the low byte and the number of dwords to
        # return, minus one, in the high half.
        cdw10=SMART_LOG_ID | (((SMART_LOG_BYTES // 4) - 1) << 16),
    )
    descriptor = opener(device, os.O_RDONLY)
    try:
        ioctl(descriptor, NVME_ADMIN_COMMAND, command)
    finally:
        closer(descriptor)
    return buffer.raw


def _health(smart: Dict[str, Any]) -> Dict[str, Any]:
    """Turn the counters into something an owner can act on."""
    concerns = list(smart.get("critical_warnings", []))
    wear = int(smart.get("wear_percent") or 0)
    spare = int(smart.get("available_spare_percent") or 0)
    threshold = int(smart.get("available_spare_threshold_percent") or 0)
    cycles = int(smart.get("power_cycles") or 0)
    unsafe = int(smart.get("unsafe_shutdowns") or 0)
    if wear >= 90:
        concerns.append(
            "the drive has used {}% of its rated write endurance".format(wear)
        )
    if threshold and spare and spare <= threshold:
        concerns.append("spare capacity has reached its threshold")
    if int(smart.get("media_errors") or 0):
        concerns.append(
            "{} media errors have been recorded".format(smart["media_errors"])
        )
    ratio = (unsafe / cycles) if cycles else 0.0
    if cycles and ratio >= UNSAFE_SHUTDOWN_WARNING_RATIO:
        concerns.append(
            "{} of {} power-offs did not shut the drive down cleanly "
            "({:.0%}), which risks filesystem damage".format(
                unsafe, cycles, ratio
            )
        )
    return {
        "healthy": not concerns,
        "concerns": concerns,
        "unsafe_shutdown_ratio": round(ratio, 3) if cycles else None,
    }


def drive_health(
    dev_root: str = "/dev",
    reader: Optional[Callable[[str], bytes]] = None,
) -> Dict[str, Any]:
    """SMART health for every NVMe controller, or a stated reason for none."""
    fetch = reader or read_smart_log
    devices = controllers(dev_root)
    if not devices:
        return {
            "available": False,
            "drives": [],
            "reason": "No NVMe controller was found on this host.",
        }
    drives = []
    failures = []
    for device in devices:
        try:
            smart = parse_smart_log(fetch(device))
        except (OSError, ValueError) as error:
            # A drive that refuses the command is reported as a drive whose
            # SMART could not be read, not as a drive with no problems.
            failures.append("{}: {}".format(
                os.path.basename(device), type(error).__name__
            ))
            continue
        drives.append({
            "device": os.path.basename(device),
            **smart,
            "health": _health(smart),
        })
    if not drives:
        return {
            "available": False,
            "drives": [],
            "reason": (
                "SMART could not be read from any NVMe controller ({}). The "
                "admin passthrough needs CAP_SYS_ADMIN."
            ).format("; ".join(failures) or "no reason reported"),
        }
    return {
        "available": True,
        "drives": drives,
        "unreadable": failures,
        "reason": "",
    }
