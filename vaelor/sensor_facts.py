"""What this hardware does not expose, and what must never be touched.

This codebase has been wrong six times in the direction of declaring a sensor
absent that was in fact readable. The correction for that is to probe rather
than assert — which is what the rest of the sensor code now does. But the
opposite error is also available, and the fix for it is different: where
research has *established* that something is not there, that finding needs
recording with its evidence, or the next person re-derives it from scratch or,
worse, ships a zero.

Two kinds of entry live here.

**Prohibited nodes.** Things that are writable, or that look like readings, and
must not be used. These are not gaps to be filled later; they are hazards.

**Proven absent.** Sources established to be unavailable, each with the
evidence that settled it, so a future reviewer can check the reasoning instead
of re-opening the question.
"""

from __future__ import annotations

from typing import Any, Dict, List, Mapping, Optional


#: hwmon attributes Vaelor must never read as telemetry or write as control.
#:
#: ``pwm1_enable`` on ``hp-wmi`` is the dangerous one, for three separate
#: reasons established from ``drivers/platform/x86/hp/hp-wmi.c``:
#:
#: * **Its value is not a reading.** ``hp_wmi_hwmon_read()`` returns
#:   ``priv->mode``, a driver-side cache initialised to ``PWM_MODE_AUTO`` at
#:   probe that never queries the embedded controller. The ``2`` observed on
#:   the live machine is that default, not the machine's state. Publishing it
#:   as fan-mode telemetry would be inventing a sensor.
#: * **Writing 0 pins both fans to maximum** and latches it with a 90-second
#:   keep-alive, so a single exploratory write changes the cooling of the
#:   owner's machine until something keeps renewing it or the timer lapses.
#: * **Writing 1 returns -EOPNOTSUPP**: manual mode is DMI-gated to HP Victus
#:   laptops, so the "control" this node appears to offer does not exist here.
#:
#: ``pwm1`` itself is declared by the driver but hidden behind the same gate;
#: ``hwmon7`` exposes only ``pwm1_enable``. There is no fan speed control on
#: this hardware, and ``cpu_fan`` being unavailable is now proven rather than
#: assumed.
PROHIBITED_HWMON_ATTRIBUTES = {
    "pwm1_enable": (
        "On hp-wmi this is a driver-side cached default, not a reading, and "
        "writing it either fails or pins the fans to maximum for 90 seconds. "
        "It is neither telemetry nor control."
    ),
    "pwm2_enable": (
        "Same driver, same cached-default behaviour as pwm1_enable."
    ),
}

#: Safe alternatives, recorded so nobody returns to ``pwm1_enable`` looking for
#: fan behaviour. Both live in ``hp_bioscfg``. Deliberately **not** wired up:
#: they are firmware settings, changing them is a real change to the owner's
#: machine, and nobody has asked for it.
FAN_BEHAVIOUR_ALTERNATIVES = (
    {
        "setting": "Increase Idle Fan Speed(%)",
        "source": "hp_bioscfg",
        "range": "0-100",
        "observed": 0,
    },
    {
        "setting": "Performance Control",
        "source": "hp_bioscfg",
        "range": "",
        "observed": None,
    },
)


def prohibited_reason(attribute: str) -> str:
    """Why this hwmon attribute must not be used, or ``""`` if it is fine."""
    return PROHIBITED_HWMON_ATTRIBUTES.get(str(attribute or "").strip(), "")


def is_prohibited(attribute: str) -> bool:
    return bool(prohibited_reason(attribute))


#: Sources established to be unavailable on this hardware, with the evidence.
#:
#: ``reportable`` says whether the absence itself is worth surfacing. Most are:
#: "this machine has no ECC counters because its memory is not ECC" is a useful
#: thing for an owner to be told once. ``vddgfx``/``vddnb`` are the exception —
#: see below.
PROVEN_ABSENT = {
    # Confirmed live on 2026-08-07 from three independent sources, and now
    # *discovered* rather than recorded: `vaelor/memory_ecc.py` reads EDAC and
    # the SMBIOS memory records on the machine in front of it. This row stays
    # as the finding's history; the probe is what a surface should ask, because
    # it can also tell "this memory has no ECC" from "this machine does not
    # report ECC state" — which this row, on its own, cannot.
    "ecc_counters": {
        "name": "Memory error counters",
        "evidence": (
            "The memory is non-ECC LPDDR5X and amd64_edac returns ENODEV, so "
            "there are no correctable or uncorrectable error counts to read. "
            "Re-established live: SMBIOS type 16 reports error correction "
            "'None', all eight type 17 records report a total width equal to "
            "their data width, and no controller registers under "
            "/sys/devices/system/edac/mc."
        ),
        "reportable": True,
    },
    "dimm_temperature": {
        "name": "Memory module temperature",
        "evidence": (
            "The eight spd5118 i2c clients are DMI-derived phantoms: all eight "
            "were read over SMBus and every register returns a constant 0x09. "
            "There are no physical DIMMs behind them - the memory is soldered."
        ),
        "reportable": True,
    },
    "gpu_voltage_rails": {
        "name": "GPU voltage rails",
        "evidence": (
            "vddgfx and vddnb are hardcoded `*value = 0` stubs in "
            "smu_v14_0_0_ppt.c. The zero is the driver's placeholder, not a "
            "measurement of zero volts."
        ),
        # The one entry that must not be surfaced even as an absence. A panel
        # saying "GPU voltage: unavailable" invites someone to go looking for
        # it, and what they would find is a file containing 0 that reads like a
        # working sensor. Silence is the honest answer here.
        "reportable": False,
    },
    "ppt_limit": {
        "name": "Package power limit",
        "evidence": (
            "Firmware carries StapmOpnLimit but the driver never copies it "
            "into gpu_metrics, so it is not reachable from userspace."
        ),
        "reportable": True,
    },
    "per_ccd_temperature": {
        "name": "Per-CCD processor temperature",
        "evidence": (
            "A driver gap rather than missing silicon: k10temp only enumerates "
            "CCDs for family 0x1a models 0x40-0x4f and this processor is "
            "0x1a/0x70. Establishing whether the silicon reports it needs a "
            "patched module and MOK enrollment under Secure Boot."
        ),
        "reportable": True,
    },
    "firmware_thermal_alarms": {
        "name": "Firmware temperature and fan threshold alarms",
        "evidence": (
            "hp_wmi claims the WMI notify handler before hp_wmi_sensors can "
            "subscribe, so the sensor driver reports has_events = false. "
            "Vaelor's own polling covers the same ground without depending on "
            "firmware-side thresholds."
        ),
        "reportable": True,
    },
}


def absent_sources(include_unreportable: bool = False) -> List[Dict[str, Any]]:
    """Established absences worth telling an owner about, with their evidence."""
    return [
        {"id": key, **{k: v for k, v in facts.items() if k != "reportable"}}
        for key, facts in sorted(PROVEN_ABSENT.items())
        if include_unreportable or facts.get("reportable")
    ]


def absence_evidence(source: str) -> str:
    """Why this source is unavailable, or ``""`` if nothing is recorded."""
    facts = PROVEN_ABSENT.get(str(source or "").strip())
    return str(facts.get("evidence", "")) if facts else ""


def filter_readings(readings: Optional[Mapping[str, Any]]) -> Dict[str, Any]:
    """Drop values from sources proven to be placeholders.

    A driver stub returning 0 is indistinguishable from a real reading of zero
    to everything downstream, so it is removed here rather than annotated.
    """
    result = {}
    for key, value in dict(readings or {}).items():
        name = str(key).lower()
        if name in {"vddgfx", "vddnb"} or name.endswith(("_vddgfx", "_vddnb")):
            continue
        result[key] = value
    return result
