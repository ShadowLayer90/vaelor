"""Whether this machine's memory corrects errors, established rather than assumed.

The card said *"ECC: not reported by this machine"*, which reads like "there is
no ECC" and claims something different: that the machine was asked and stayed
silent. It was never asked. Those two statements are not interchangeable and
this module exists so that whichever one is published is the one that is true.

**Three independent sources, all read live on the target hardware.**

* ``/sys/devices/system/edac/mc`` holds no memory controller, and
  ``modprobe amd64_edac`` fails with ``No such device`` — the driver declines
  to bind, so there are no correctable or uncorrectable counts to read.
* SMBIOS type 16 (Physical Memory Array) reports Memory Error Correction ``3``,
  which the specification defines as ``None``.
* All eight SMBIOS type 17 records report a total width of 32 bits equal to
  their data width of 32 bits. ECC memory carries check bits *beyond* the data
  width, so equal widths mean none are wired.

The three agree, so the honest answer on this machine is **no host-visible
ECC** — not "unreported".

**The caveat that must travel with that answer.** LPDDR5X commonly carries
on-die ECC, which corrects within the DRAM die and is never reported over
SMBIOS or EDAC by design. Reading "no ECC" here as "errors are never corrected
anywhere" would be a claim this evidence does not support, so the summary says
*host-visible* every time.

SMBIOS lives in ``/sys/firmware/dmi/entries``, which is mode 0400 root, so this
runs on the privileged side of the bridge for the same reason the RAPL counters
do. The records are parsed from their raw bytes rather than by shelling out to
``dmidecode``: an appliance must not need a package installed by hand before it
can answer a question about its own memory.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, List, Optional


EDAC_ROOT = "devices/system/edac/mc"
DMI_ENTRIES = "firmware/dmi/entries"

#: SMBIOS 3.4.0, table 7.16.3 — Physical Memory Array, Memory Error Correction.
#: Byte 0x06 of a type 16 record. ``other`` and ``unknown`` are the firmware
#: declining to answer, and are treated as exactly that rather than as "none".
ERROR_CORRECTION = {
    1: ("unreported", "an error-correction type of 'other'"),
    2: ("unreported", "an error-correction type of 'unknown'"),
    3: ("absent", "no error correction"),
    4: ("present", "parity"),
    5: ("present", "single-bit ECC"),
    6: ("present", "multi-bit ECC"),
    7: ("present", "CRC"),
}

#: Offsets into the raw records. Both are inside the fixed-length header of
#: their SMBIOS version, so a record shorter than this is truncated and is
#: skipped rather than read past.
TYPE16_CORRECTION_OFFSET = 0x06
TYPE17_TOTAL_WIDTH = 0x08
TYPE17_DATA_WIDTH = 0x0A

#: A type 17 record reports width ``0xFFFF`` for "unknown". Comparing that
#: against a real data width would manufacture a difference and read as ECC.
UNKNOWN_WIDTH = 0xFFFF

ON_DIE_CAVEAT = (
    "On-die correction inside the DRAM itself is never reported to the host, "
    "so this describes what reaches the memory controller."
)


def _entries(sys_root: str, record_type: int) -> List[bytes]:
    root = Path(sys_root) / DMI_ENTRIES
    found: List[bytes] = []
    try:
        listing = sorted(root.iterdir())
    except OSError:
        return found
    for entry in listing:
        if not entry.name.startswith("{}-".format(record_type)):
            continue
        try:
            found.append((entry / "raw").read_bytes())
        except OSError:
            continue
    return found


def _edac_controllers(sys_root: str) -> List[str]:
    """Memory controllers EDAC has bound to, which only happens for ECC."""
    try:
        listing = sorted((Path(sys_root) / EDAC_ROOT).iterdir())
    except OSError:
        return []
    return [entry.name for entry in listing if entry.name.startswith("mc")]


def _array_correction(sys_root: str) -> Optional[Dict[str, Any]]:
    for raw in _entries(sys_root, 16):
        if len(raw) <= TYPE16_CORRECTION_OFFSET:
            continue
        code = raw[TYPE16_CORRECTION_OFFSET]
        state, described = ERROR_CORRECTION.get(
            code, ("unreported", "an error-correction code this reader does "
                   "not recognise ({})".format(code))
        )
        return {
            "source": "smbios/type-16",
            "state": state,
            "detail": described,
        }
    return None


def _device_widths(sys_root: str) -> Optional[Dict[str, Any]]:
    """Whether any memory device carries check bits beyond its data width."""
    wider = 0
    counted = 0
    for raw in _entries(sys_root, 17):
        if len(raw) < TYPE17_DATA_WIDTH + 2:
            continue
        total = int.from_bytes(raw[TYPE17_TOTAL_WIDTH:TYPE17_TOTAL_WIDTH + 2], "little")
        data = int.from_bytes(raw[TYPE17_DATA_WIDTH:TYPE17_DATA_WIDTH + 2], "little")
        if UNKNOWN_WIDTH in (total, data) or not data:
            continue
        counted += 1
        if total > data:
            wider += 1
    if not counted:
        return None
    return {
        "source": "smbios/type-17",
        "state": "present" if wider else "absent",
        "detail": (
            "{} of {} memory devices carry check bits beyond their data width"
            .format(wider, counted) if wider else
            "all {} memory devices report a total width equal to their data "
            "width, so no check bits are wired".format(counted)
        ),
    }


def memory_ecc(sys_root: str = "/sys") -> Dict[str, Any]:
    """What this machine says about memory error correction, and how it said it.

    ``state`` is one of ``present``, ``absent`` or ``unreported``, and the third
    is a real answer rather than a placeholder: a machine whose firmware
    publishes ``unknown`` and whose kernel binds no EDAC device has genuinely
    not told anyone, and saying so is different from saying there is no ECC.
    """
    sources: List[Dict[str, Any]] = []
    controllers = _edac_controllers(sys_root)
    if controllers:
        sources.append({
            "source": "edac",
            "state": "present",
            "detail": (
                "the kernel registered {} memory controller{} under "
                "/sys/devices/system/edac/mc, which it only does for memory "
                "that reports errors"
            ).format(len(controllers), "" if len(controllers) == 1 else "s"),
        })
    else:
        sources.append({
            "source": "edac",
            "state": "unreported",
            "detail": (
                "no memory controller is registered under "
                "/sys/devices/system/edac/mc"
            ),
        })
    for reading in (_array_correction(sys_root), _device_widths(sys_root)):
        if reading is not None:
            sources.append(reading)
    firmware = [entry for entry in sources if entry["source"].startswith("smbios")]
    states = {entry["state"] for entry in sources}
    if "present" in states:
        state = "present"
    elif firmware and all(entry["state"] == "absent" for entry in firmware):
        state = "absent"
    else:
        state = "unreported"
    return {
        "state": state,
        "available": state != "unreported",
        "value": _value(state, sources),
        "summary": _summary(state, sources, bool(firmware)),
        "basis": ", ".join(entry["source"] for entry in sources),
        "sources": sources,
    }


def _value(state: str, sources: List[Dict[str, Any]]) -> Optional[str]:
    if state == "absent":
        return "None"
    if state != "present":
        return None
    kinds = [
        entry["detail"] for entry in sources
        if entry["state"] == "present" and entry["source"] == "smbios/type-16"
    ]
    return "Enabled ({})".format(kinds[0]) if kinds else "Enabled"


def _summary(state: str, sources: List[Dict[str, Any]], firmware: bool) -> str:
    """One sentence an owner can act on, naming what was read."""
    read = "; ".join(
        "{} reports {}".format(entry["source"], entry["detail"])
        for entry in sources
    )
    if state == "present":
        return "This machine reports ECC memory. {}.".format(read)
    if state == "absent":
        return (
            "This memory has no ECC visible to the host. {}. {}"
        ).format(read, ON_DIE_CAVEAT)
    if not firmware:
        return (
            "This machine does not report whether its memory has ECC: the "
            "SMBIOS memory records are root-only and were not readable, and "
            "{}. That is a gap in what Vaelor could read, not a finding that "
            "there is no ECC."
        ).format(sources[0]["detail"])
    return (
        "This machine does not report whether its memory has ECC. {}. That is "
        "the firmware declining to answer, not a finding that there is no ECC."
    ).format(read)


#: Telemetry keys. Static for the life of a boot — SMBIOS does not change
#: without one — so a caller is expected to read this once and hold it rather
#: than parsing the DMI tree on every poll.
TELEMETRY_KEYS = {
    "state": "memory_ecc_state",
    "value": "memory_ecc_value",
    "summary": "memory_ecc_summary",
    "basis": "memory_ecc_basis",
}


def memory_ecc_telemetry(reading: Optional[Dict[str, Any]]) -> Dict[str, Any]:
    """Telemetry keys from one :func:`memory_ecc` result, or nothing at all."""
    if not reading or not reading.get("sources"):
        return {}
    telemetry = {
        TELEMETRY_KEYS["state"]: reading["state"],
        TELEMETRY_KEYS["summary"]: reading["summary"],
        TELEMETRY_KEYS["basis"]: reading["basis"],
    }
    if reading.get("value"):
        telemetry[TELEMETRY_KEYS["value"]] = reading["value"]
    return telemetry
