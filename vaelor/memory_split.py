"""How this machine's RAM is divided, and who decided the division.

``MemTotal`` is not how much memory is fitted. On a unified-memory part the
firmware reserves a slice for graphics *before* Linux boots, and what the OS
reports is what is left over: 45 GiB visible on a Z2 Mini with a 16 GiB
carve-out. Reporting the OS figure as "the memory in this machine" is wrong by
the size of the carve-out, and reporting the carve-out as a property of the
part is wrong in a worse way — **it is a BIOS setting the owner can change.**

VD-062 is the rule this module exists to keep: *can an owner change this —
through BIOS, configuration, the interface, or by installing something? If yes,
it is a reading, and it must be derived rather than recalled.* The split is
exactly that, so what is published here is the relationship (carve-out plus
OS-visible is what is fitted) with both numbers read fresh, and a statement of
whether the setting itself could be read on this host.

The derivation chain is the invariant; the megabytes are not:

    firmware carve-out → OS-visible RAM → GTT aperture

Where the carve-out cannot be read, this says so rather than presenting the OS
view as the whole truth. An unreadable reservation is an unanswered question,
not a reservation of zero — the same distinction the accelerator and context
checks draw between ``unknown`` and ``false``.
"""

from __future__ import annotations

from typing import Any, Dict, Mapping, Optional, Sequence


GIB = 1024 ** 3


def _accelerator_carve_out(
    accelerators: Optional[Sequence[Mapping[str, Any]]],
) -> Optional[int]:
    """The carve-out as the driver reports it, on a unified part only.

    A discrete card's ``vram_total`` is memory on the card and is not carved
    out of anything the OS could otherwise use, so counting it here would
    invent RAM that is not fitted.
    """
    for accelerator in accelerators or []:
        if not accelerator.get("unified_memory"):
            continue
        total = int(accelerator.get("vram_total_bytes") or 0)
        if total > 0:
            return total
    return None


def graphics_memory_split(
    os_visible_bytes: Any,
    accelerators: Optional[Sequence[Mapping[str, Any]]] = None,
    firmware: Optional[Mapping[str, Any]] = None,
) -> Dict[str, Any]:
    """Physical total, current split, and whether the split is configurable.

    ``firmware`` is the reading from
    :func:`~vaelor.platforms.accelerators.configured_graphics_memory` — the
    BIOS attribute itself. It is passed in rather than read here so this module
    stays free of sysfs paths and so the "attribute not present" arm is
    reachable from a test on any host.

    Three states, and they are deliberately different answers:

    * **The firmware attribute is readable.** The size is the setting, and the
      split is *established* as configurable, with the attribute named.
    * **Only the driver reports it.** The size is real and the total can be
      derived, but nothing on this host proved the value is settable, so
      ``configurable`` is ``None`` rather than ``True``.
    * **Neither.** ``graphics_reserved_bytes`` is ``None``, the total is
      unknown, and the reason says the OS view is not the whole truth rather
      than implying it is.
    """
    visible = max(0, int(os_visible_bytes or 0))
    reading = dict(firmware or {})
    firmware_bytes = reading.get("bytes") if reading.get("available") else None
    driver_bytes = _accelerator_carve_out(accelerators)
    reserved = firmware_bytes or driver_bytes
    record: Dict[str, Any] = {
        "os_visible_bytes": visible,
        "graphics_reserved_bytes": reserved,
        # OS-visible plus the reservation. Named "accounted" and not "fitted"
        # on purpose: it is the sum of two readings, and firmware reserves
        # further small regions that neither of them counts, so it is a floor
        # on what is installed rather than the figure on the memory module.
        "accounted_total_bytes": (visible + reserved) if reserved else None,
        "reserved_source": (
            (reading.get("source") or "hp-bioscfg") if firmware_bytes
            else "amdgpu vram_total" if driver_bytes
            else ""
        ),
        "reserved_value": str(reading.get("value") or "") if firmware_bytes else "",
        "configurable": True if firmware_bytes else None,
        "reason": "",
        "note": (
            "OS-visible RAM excludes the graphics carve-out, so what is fitted "
            "is the carve-out plus what the OS can see. The carve-out is a "
            "firmware setting rather than a property of the processor."
        ),
    }
    if firmware_bytes:
        record["reason"] = (
            "The graphics reservation is a firmware setting on this machine "
            "and reads {} at {}. An owner can change it in the BIOS, which "
            "moves both this figure and the memory the operating system sees."
        ).format(reading.get("value") or "the value shown", record["reserved_source"])
        return record
    if driver_bytes:
        record["reason"] = (
            "The graphics reservation is reported by the display driver, not "
            "by a firmware setting this host publishes, so its size is known "
            "but whether it can be changed here was not established."
        ) + " " + str(reading.get("reason") or "")
        record["reason"] = record["reason"].strip()
        return record
    record["reason"] = (
        "No graphics reservation could be read on this machine, so the {:.0f} "
        "GB the operating system reports cannot be turned into a total for the "
        "hardware. That is an unanswered question rather than a reservation of "
        "zero: firmware may hold memory back before Linux ever sees it."
    ).format(visible / (1000 ** 3))
    return record
