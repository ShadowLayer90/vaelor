"""Generic x86-64 and workstation-class driver.

An x86 host has no enclosure bridge, no PMIC and no GPIO. It usually does have
an accelerator and an EC that owns the fans. This driver reports exactly that,
with reasons, rather than inheriting Raspberry Pi defaults.
"""

from __future__ import annotations

from typing import Any, Dict, Optional

from .accelerators import (
    discover_accelerators,
    discover_npus,
    graphics_inventory,
)
from .base import (
    GENERIC,
    GENERIC_PRODUCT,
    WORKSTATION,
    HardwarePlatformBase,
    Snapshot,
    battery_supplies,
    capability,
    controllable_fans,
    dmi_identity,
    dmi_model_name,
)
from ..wmi_sensors import read_sensors
from .raspberry_pi import detect_product, enclosure_detected


#: HP, Dell and Lenovo business machines keep the fan curve in the embedded
#: controller. The in-tree ``hp-wmi-sensors`` interface is documented read-only
#: and exposes no ``pwm*`` or ``fan*`` attribute at all on the machines checked,
#: so there is nothing to drive even when the hwmon node exists.
EC_FAN_VENDORS = ("hp", "hewlett", "dell", "lenovo")


def _fan_reason(vendor: str, readable: int = 0) -> str:
    """Why fan *control* is unavailable, without denying the fans exist.

    These two were conflated, and the conflation was wrong in the worst
    direction: the appliance told an HP workstation with three readable fans
    that it had none. Reading a fan and setting its speed are separate
    capabilities, and only the second is in doubt here.
    """
    seen = (
        " Vaelor can read {} fan{} on this machine.".format(
            readable, "" if readable == 1 else "s"
        )
        if readable else ""
    )
    if vendor and vendor.lower().startswith(EC_FAN_VENDORS):
        return (
            "{} keeps fan control in the embedded controller, so Vaelor has "
            "not established that their speed can be set from Linux.{}"
        ).format(vendor, seen)
    return "No controllable chassis fan was found on this machine." + seen


class WorkstationPlatform(HardwarePlatformBase):
    """Desktop, workstation, or server-class host with no enclosure bridge."""

    machine_class = WORKSTATION

    def __init__(
        self,
        *,
        variant_path: str = "/opt/pironman5/.variant",
        **kwargs: Any,
    ):
        # `sys_root` is the base's now: the CPU topology is read from it on
        # every platform, so a driver that consumed the argument on its way
        # past left the base reading the real /sys under a fixture.
        super().__init__(**kwargs)
        self._variant_path = variant_path
        self._accelerators: Optional[list[Dict[str, Any]]] = None
        self._npus: Optional[list[Dict[str, Any]]] = None

    def accelerators(self) -> list[Dict[str, Any]]:
        if self._accelerators is None:
            self._accelerators = discover_accelerators(self._sys_root)
        return self._accelerators

    def neural_accelerators(self) -> list[Dict[str, Any]]:
        if self._npus is None:
            self._npus = discover_npus(self._sys_root)
        return self._npus

    def graphics(self) -> Dict[str, Any]:
        return graphics_inventory(self._sys_root)

    def product(
        self, raw: Optional[Snapshot] = None, board: Optional[Snapshot] = None
    ) -> Dict[str, Any]:
        """Identify the machine from SMBIOS without claiming an enclosure.

        The product id stays ``generic`` because that is the contract the UI
        already branches on; only the name becomes truthful. Discovery still
        wins over machine class: if an enclosure bridge really is reporting on
        this host, its product is reported rather than overruled.
        """
        discovered = detect_product(
            raw,
            variant_path=self._variant_path,
            enclosure_present=enclosure_detected(raw),
        )
        if discovered.get("id") != "generic":
            return discovered
        product = dict(GENERIC_PRODUCT)
        dmi = dmi_identity(self._dmi_root)
        name = dmi_model_name(dmi)
        if name:
            product.update({
                "name": name,
                "detected_from": "smbios",
                "vendor": dmi["vendor"],
                "chassis": dmi["chassis"],
            })
        if self.accelerators():
            product["capabilities"] = ["gpu"]
        return product

    def power(self, metrics: Optional[Snapshot], board: Snapshot) -> Dict[str, Any]:
        """No PMIC, no PiPower, and no unprivileged RAPL on a stock install.

        Reporting ``source: "Raspberry Pi PMIC"`` here is what made an AMD
        workstation claim Broadcom power telemetry. ``None`` is the honest
        answer, and the accelerator supplies the only real power reading this
        class of machine exposes without root.
        """
        accelerator = (self.accelerators() or [{}])[0]
        watts = accelerator.get("power_watts")
        return {
            "source": None,
            "input_voltage": None,
            "input_current": None,
            "output_voltage": None,
            "output_current": None,
            "output_watts": None,
            "watts_available": False,
            "throttled_flags": None,
            "undervoltage_now": False,
            # An x86 host loses power the same way an appliance does, and it
            # has no Broadcom flag word to mislead anyone with — so the only
            # reading of the event is the one below.
            "previous_shutdown": self.previous_shutdown(),
            "accelerator_watts": watts,
            "reason": (
                "This machine exposes no unprivileged power telemetry. Package "
                "energy counters are root-only and there is no PMIC or UPS "
                "accessory."
            ),
            "battery": self._battery(),
        }

    def _battery(self) -> Dict[str, Any]:
        """Whatever ``/sys/class/power_supply`` reports, or a stated absence.

        Hardcoding ``available: False`` here was right for a mini PC on mains
        and wrong for a laptop, and it was asserted without ever opening the
        file the kernel publishes for exactly this.
        """
        supplies = battery_supplies(self._sys_root)
        if not supplies:
            return {
                "available": False,
                "percentage": None,
                "voltage": None,
                "current": None,
                "charging": None,
                "power_source": None,
                "reason": (
                    "The kernel reports no battery under "
                    "/sys/class/power_supply on this machine."
                ),
            }
        primary = supplies[0]
        status = (primary.get("status") or "").lower()
        return {
            "available": True,
            "percentage": primary.get("percentage"),
            "voltage": primary.get("voltage"),
            "current": primary.get("current"),
            "charging": status == "charging" if status else None,
            "power_source": primary.get("name"),
            "reason": "",
        }

    def power_telemetry(self) -> bool:
        # Accelerator hwmon is the only power reading this class of machine
        # exposes without root; RAPL energy counters are root-only.
        return any(
            accelerator.get("power_watts") is not None
            for accelerator in self.accelerators()
        )

    def capabilities(
        self,
        raw: Optional[Snapshot] = None,
        metrics: Optional[Snapshot] = None,
        inventory: Optional[Snapshot] = None,
    ) -> Dict[str, Dict[str, Any]]:
        board = self.board()
        vendor = str(board.get("vendor") or "")
        chassis = board.get("chassis") or "machine"
        # An enclosure bridge could still be present on an exotic host; trust
        # discovery over the machine class rather than hard-coding absence.
        detected = enclosure_detected(raw)
        no_enclosure = (
            "No enclosure controller is fitted to this {}.".format(chassis)
        )
        readable_fans = len(read_sensors(self._sys_root).get("fans", []))
        return {
            "case_fan": capability(
                detected,
                "No enclosure fan controller was detected on this {}. {}".format(
                    chassis, _fan_reason(vendor, readable_fans)
                ),
            ),
            # Reading a fan and setting its speed are different capabilities,
            # and only the second is in doubt. Separating them is what stops
            # "we cannot control the fans" from rendering as "there are no
            # fans" on a machine with three of them.
            "fan_readings": capability(
                bool(readable_fans),
                "No fan tachometer is readable on this machine.",
            ),
            "case_lighting": capability(detected, no_enclosure),
            "oled": capability(
                detected,
                "No I²C display is attached to this {}.".format(chassis),
            ),
            # Probed, not asserted. A vendor that keeps fan control in its
            # embedded controller genuinely offers nothing here, but that is
            # something to discover: a board exposing `pwm1` was being told it
            # had no controllable fan without anyone looking for one.
            # Still gated on a probe, and deliberately still False on this
            # hardware. `hp-wmi` exposes a writable `pwm1_enable` reading 2,
            # but what its values mean on this driver is not established, and
            # the only way to find out is to write to a stranger's cooling.
            # Until that is answered this stays unavailable with a reason that
            # no longer denies the fans exist.
            "cpu_fan": capability(
                bool(controllable_fans(self._sys_root)),
                _fan_reason(vendor, readable_fans),
            ),
            "battery": capability(
                bool(battery_supplies(self._sys_root)),
                "The kernel reports no battery under /sys/class/power_supply "
                "on this machine.",
            ),
            "gpu": capability(
                bool(self.accelerators()),
                "No inference accelerator was found on the PCI bus.",
            ),
            # Discovered so a workload that supports it can be granted the
            # device. Vaelor's own inference backends cannot target an NPU.
            "npu": capability(
                bool(self.neural_accelerators()),
                "No neural processing unit was found on the PCI bus.",
            ),
        }


class GenericPlatform(WorkstationPlatform):
    """Fallback for a host that is neither a Pi nor SMBIOS-identifiable."""

    machine_class = GENERIC
