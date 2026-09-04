"""Raspberry Pi and Pironman enclosure driver.

This is one implementation of the hardware platform contract, not the base
case. Everything Broadcom-, device-tree- or SunFounder-specific lives here.
"""

from __future__ import annotations

import re
import shutil
import subprocess
from pathlib import Path
from typing import Any, Callable, Dict, Optional

from ..boot_forensics import THROTTLED_HISTORY_NOTE
from .base import (
    GENERIC_PRODUCT,
    PI_APPLIANCE,
    HardwarePlatformBase,
    Snapshot,
    capability,
    text,
)


PRODUCTS: Dict[str, Dict[str, Any]] = {
    "pironman5": {
        "name": "Pironman 5",
        "icon": "pironman5",
        "fan_count": 2,
        "nvme_slots": 1,
        "capabilities": ["oled", "rgb", "cpu_fan", "case_fan", "power_button"],
    },
    "pironman5-max": {
        "name": "Pironman 5 Max",
        "icon": "pironman5-max",
        "fan_count": 2,
        "nvme_slots": 2,
        "capabilities": [
            "oled", "rgb", "cpu_fan", "case_fan", "case_fan_led",
            "power_button",
        ],
    },
    "pironman5-mini": {
        "name": "Pironman 5 Mini",
        "icon": "pironman5-mini",
        "fan_count": 1,
        "nvme_slots": 1,
        "capabilities": [
            "rgb", "cpu_fan", "case_fan", "case_fan_led", "power_button",
        ],
    },
    "pironman5-pro-max": {
        "name": "Pironman 5 Pro Max",
        "icon": "pironman5-pro-max",
        "fan_count": 3,
        "nvme_slots": 2,
        "capabilities": [
            "oled", "rgb", "cpu_fan", "case_fan", "case_fan_led",
            "power_button", "touchscreen", "camera", "speakers",
        ],
    },
}

ALIASES = {
    "base": "pironman5",
    "max": "pironman5-max",
    "mini": "pironman5-mini",
    "pro_max": "pironman5-pro-max",
    "pironman-5": "pironman5",
    "pironman 5": "pironman5",
    "pironman 5 max": "pironman5-max",
    "pironman 5 mini": "pironman5-mini",
    "pironman 5 pro max": "pironman5-pro-max",
    "pironman5_max": "pironman5-max",
    "pironman5_mini": "pironman5-mini",
    "pironman5_pro_max": "pironman5-pro-max",
}

#: (product_id, product_ver) from the HAT ID EEPROM -> Vaelor product id.
#:
#: SunFounder ships NO product_id table. Their own software
#: (pironman5 `variants/__init__.py`, v1.3.17) reads only `/opt/pironman5/.variant`
#: and defaults to base, and no `.eep` image or id map exists in the pironman5,
#: pm_auto or sf_rpi_status repos. So this table is built from *measured hardware*,
#: not transcribed from a vendor source. Only the Max is confirmed so far, read
#: off the live board on 2026-08-14 (product "Pironman 5", vendor "SunFounder",
#: product_id 0x0132, product_ver 0x000b). The other three variants are unknown
#: until their boards are read - and an unrecognised id reports "unknown", never
#: base (#211, and #59/VD-005: never claim a model discovery cannot confirm).
HAT_EEPROM_VARIANTS: Dict[tuple, str] = {
    (0x0132, 0x000B): "pironman5-max",
}

#: An enclosure whose HAT EEPROM is present but whose id is not in the table.
#: fan/nvme/capabilities are zeroed on purpose: a case is fitted and we know
#: nothing about which, so we claim nothing (VD-005).
UNKNOWN_ENCLOSURE: Dict[str, Any] = {
    "id": "pironman-unknown",
    "name": "Pironman enclosure (unrecognised model)",
    "icon": "pironman5",
    "fan_count": 0,
    "nvme_slots": 0,
    "capabilities": [],
    "detected_from": "hat-eeprom-unknown",
    "confident": False,
}


def _parse_hat_hex(raw: str) -> Optional[int]:
    """Parse a HAT EEPROM id string such as ``0x0132`` to an int, or None."""
    cleaned = raw.replace("\x00", "").strip()
    if not cleaned:
        return None
    try:
        return int(cleaned, 16)
    except ValueError:
        return None


def _is_sunfounder_hat(hat: Dict[str, Any]) -> bool:
    """Is this HAT EEPROM a SunFounder Pironman, and not some other HAT?

    Gating on the vendor keeps a foreign HAT from being reported as an
    unrecognised *Pironman* enclosure; a non-SunFounder HAT falls through to the
    ``.variant`` file / no-enclosure path instead.
    """
    vendor = str(hat.get("vendor", "")).lower()
    product = str(hat.get("product", "")).lower()
    return "sunfounder" in vendor or "pironman" in product


def read_hat_eeprom(hat_root: str = "/proc/device-tree/hat") -> Optional[Dict[str, Any]]:
    """The Raspberry Pi HAT ID EEPROM, or None when no HAT node is present.

    The Pi firmware exposes a fitted HAT+'s ID EEPROM under
    ``/proc/device-tree/hat/`` as NUL-terminated strings: ``product_id`` and
    ``product_ver`` are hex like ``0x0132`` / ``0x000b``. This is the enclosure
    stating its own identity in hardware, which is why detection prefers it over
    the SunFounder-written ``.variant`` file (#211).
    """
    root = Path(hat_root)
    product_id = _parse_hat_hex(text(root / "product_id"))
    product_ver = _parse_hat_hex(text(root / "product_ver"))
    if product_id is None or product_ver is None:
        return None
    return {
        "product_id": product_id,
        "product_ver": product_ver,
        "vendor": text(root / "vendor"),
        "product": text(root / "product"),
    }


#: Peripherals reported by the enclosure bridge. Their presence is the only
#: honest evidence that an enclosure is fitted.
ENCLOSURE_PERIPHERALS = (
    "gpio_fan_state", "gpio_fan", "gpio_fan_mode", "gpio_fan_led",
    "pwm_fan_speed", "pwm_fan", "ws2812", "oled",
)


def enclosure_detected(raw: Optional[Dict[str, Any]]) -> bool:
    """Did anything actually report an enclosure on this host?"""
    peripherals = {
        str(item).lower() for item in (raw or {}).get("peripherals", []) or []
    }
    if any(item.startswith("oled") for item in peripherals):
        return True
    return bool(peripherals.intersection(ENCLOSURE_PERIPHERALS))


#: Enclosure capabilities the bridge can actually observe, and the peripheral
#: names that are evidence for each. Every other capability in :data:`PRODUCTS`
#: — ``power_button``, ``touchscreen``, ``camera``, ``speakers`` — is invisible
#: to discovery, so it can neither confirm nor rule out an enclosure.
DISCOVERABLE_CAPABILITIES: Dict[str, frozenset] = {
    "rgb": frozenset({"ws2812"}),
    "case_fan": frozenset({"gpio_fan", "gpio_fan_state", "gpio_fan_mode"}),
    "case_fan_led": frozenset({"gpio_fan_led"}),
    "cpu_fan": frozenset({"pwm_fan", "pwm_fan_speed"}),
}


def observed_capabilities(raw: Optional[Dict[str, Any]] = None) -> set:
    """Enclosure capabilities the bridge positively reported on this host.

    Matching is exact rather than by prefix, because ``gpio_fan_led`` starts
    with ``gpio_fan`` and a prefix rule would read a case-fan LED as a case fan
    and make every enclosure look alike. The OLED is the one exception: it is
    reported as ``oled`` or as ``oled_<page>``, which is how
    :meth:`RaspberryPiPlatform.capabilities` already reads it.
    """
    peripherals = {
        str(item).lower() for item in (raw or {}).get("peripherals", []) or []
    }
    observed = {
        name
        for name, evidence in DISCOVERABLE_CAPABILITIES.items()
        if peripherals & evidence
    }
    if any(item == "oled" or item.startswith("oled_") for item in peripherals):
        observed.add("oled")
    return observed


def plausible_products(
    raw: Optional[Dict[str, Any]] = None,
    *,
    variant_path: str = "/opt/pironman5/.variant",
    products: Optional[Dict[str, Dict[str, Any]]] = None,
) -> list:
    """Enclosures this machine's own evidence is consistent with.

    ``PATCH /api/v2/device/model`` used to accept any key of :data:`PRODUCTS`
    on an appliance. Its own comment called the choice *"a disambiguation
    between enclosures that discovery found plausible"*, which was true on a
    workstation — the route refuses outright there — and false on the Pi, where
    a Pironman 5 Mini could be told it was a Pro Max and then reported an OLED,
    a third fan and a touchscreen that are not in the case. VD-005 says
    availability comes from discovery; this is the set discovery leaves open.

    A product is plausible when the capabilities it declares *and discovery can
    see* are exactly the ones discovery saw. So a Max and a Pro Max stay
    plausible together — the bridge reports nothing that separates them, and
    choosing between those two is what disambiguation means — while a Mini and
    a Pro Max never do, because one has an OLED and the other does not.

    Returns ``[]`` when nothing reported an enclosure at all. An empty list is
    not "choose freely"; it is "there is nothing here to name".
    """
    catalog = PRODUCTS if products is None else products
    named = detect_product(raw, variant_path=variant_path, enclosure_present=True)
    if named.get("confident") and named.get("id") in catalog:
        # The variant file, or a bridge that named itself, outranks inference
        # from peripherals: it is the enclosure stating its own model.
        return [str(named["id"])]
    if not enclosure_detected(raw):
        return []
    observed = observed_capabilities(raw)
    return [
        product_id
        for product_id, product in catalog.items()
        if {
            capability_name
            for capability_name in product.get("capabilities", ())
            if capability_name in DISCOVERABLE_CAPABILITIES or capability_name == "oled"
        } == observed
    ]


def detect_product(
    raw: Optional[Dict[str, Any]] = None,
    variant_path: str = "/opt/pironman5/.variant",
    enclosure_present: bool = False,
    hat_root: str = "/proc/device-tree/hat",
) -> Dict[str, Any]:
    """Identify the enclosure, or report a generic host.

    ``enclosure_present`` defaults to ``False`` so that omitting it fails safe.
    It used to default to ``True``, which meant every call site that forgot the
    argument silently reported a Pironman 5 with two case fans on hardware that
    had none.

    The HAT ID EEPROM is read first (#211): it is the enclosure stating its own
    identity in hardware, so it outranks the SunFounder-written ``.variant`` file
    and the bridge id. A HAT that is fitted but whose id is not in
    :data:`HAT_EEPROM_VARIANTS` is reported as an unrecognised enclosure, never
    guessed as base (#59/VD-005). ``.variant`` is the fallback only when no HAT
    EEPROM node is present.
    """
    raw = raw or {}
    hat = read_hat_eeprom(hat_root)
    if hat is not None and _is_sunfounder_hat(hat):
        variant_id = HAT_EEPROM_VARIANTS.get((hat["product_id"], hat["product_ver"]))
        if variant_id in PRODUCTS:
            product = dict(PRODUCTS[variant_id])
            product.update({
                "id": variant_id,
                "detected_from": "hat-eeprom",
                "confident": True,
                "hat": hat,
            })
            return product
        product = dict(UNKNOWN_ENCLOSURE)
        product["hat"] = hat
        return product
    candidates = [
        text(Path(variant_path)),
        str(raw.get("id", "")),
        str(raw.get("name", "")),
    ]
    product_id = ""
    detected_from = "fallback"
    for index, candidate in enumerate(candidates):
        normalized = candidate.strip().lower()
        product_id = ALIASES.get(normalized, normalized)
        if product_id in PRODUCTS:
            detected_from = ("variant", "device-id", "device-name")[index]
            break
        product_id = ""
    if not product_id:
        if not enclosure_present:
            return dict(GENERIC_PRODUCT)
        # PiPower 5 is an accessory, not an enclosure model. Battery presence
        # must enrich the power snapshot without replacing the detected case.
        product_id = "pironman5"
    product = dict(PRODUCTS[product_id])
    product.update({
        "id": product_id,
        "detected_from": detected_from,
        "confident": detected_from != "fallback",
    })
    return product


def _number(metrics: Dict[str, Any], key: str) -> Optional[float]:
    value = metrics.get(key)
    return float(value) if isinstance(value, (int, float)) else None


def _volts(value: Optional[float]) -> Optional[float]:
    if value is None:
        return None
    return round(value / 1000 if abs(value) > 100 else value, 3)


def _amps(value: Optional[float]) -> Optional[float]:
    if value is None:
        return None
    return round(value / 1000 if abs(value) > 100 else value, 3)


def power_snapshot(
    metrics: Optional[Dict[str, Any]] = None,
    runner: Callable[..., Any] = subprocess.run,
) -> Dict[str, Any]:
    metrics = metrics or {}
    input_voltage = _volts(_number(metrics, "input_voltage"))
    input_current = _amps(_number(metrics, "input_current"))
    output_voltage = _volts(_number(metrics, "output_voltage"))
    output_current = _amps(_number(metrics, "output_current"))
    watts = _number(metrics, "output_power")
    if watts is None and output_voltage is not None and output_current is not None:
        watts = round(output_voltage * output_current, 2)
    source = "PiPower 5" if input_voltage is not None or output_voltage is not None else "Raspberry Pi PMIC"
    if input_voltage is None:
        try:
            result = runner(
                ["vcgencmd", "pmic_read_adc", "EXT5V_V"],
                capture_output=True, text=True, check=False, timeout=3,
            )
            match = re.search(r"EXT5V_V[^=]*=\s*([\d.]+)V", result.stdout or "")
            input_voltage = float(match.group(1)) if match else None
        except (OSError, subprocess.SubprocessError):
            pass
    throttled = None
    try:
        result = runner(
            ["vcgencmd", "get_throttled"],
            capture_output=True, text=True, check=False, timeout=3,
        )
        match = re.search(r"0x([0-9a-fA-F]+)", result.stdout or "")
        throttled = int(match.group(1), 16) if match else None
    except (OSError, subprocess.SubprocessError):
        pass
    battery_percentage = _number(metrics, "battery_percentage")
    return {
        "source": source,
        "input_voltage": input_voltage,
        "input_current": input_current,
        "output_voltage": output_voltage,
        "output_current": output_current,
        "output_watts": round(watts, 2) if watts is not None else None,
        "watts_available": watts is not None,
        "throttled_flags": throttled,
        "undervoltage_now": bool(throttled is not None and throttled & 0x1),
        # What the flag word covers, carried with it. `get_throttled` reports
        # the *current* boot and a power loss clears it, so `0x0` after a
        # cutout describes the boot that has just started and proves nothing
        # about the one that ended. Reading it as "no undervoltage ever
        # occurred" clears the actual cause, and that wrong turn was taken for
        # real in this incident before the journal contradicted it.
        "throttled_history_note": THROTTLED_HISTORY_NOTE,
        "throttled_covers": "this boot only",
        "battery": {
            "available": battery_percentage is not None,
            "percentage": battery_percentage,
            "voltage": _volts(_number(metrics, "battery_voltage")),
            "current": _amps(_number(metrics, "battery_current")),
            "charging": metrics.get("is_charging"),
            "power_source": metrics.get("power_source"),
        },
    }


class RaspberryPiPlatform(HardwarePlatformBase):
    """Raspberry Pi board with an optional Pironman enclosure."""

    machine_class = PI_APPLIANCE

    def __init__(
        self,
        *,
        variant_path: str = "/opt/pironman5/.variant",
        hat_root: str = "/proc/device-tree/hat",
        discharge_observer: Any = None,
        **kwargs: Any,
    ):
        # `runner` reaches the base now and is stored there: boot forensics run
        # on every platform, and a driver that swallowed the argument left the
        # base spawning real subprocesses under a test's injected one.
        super().__init__(**kwargs)
        self._variant_path = variant_path
        # The HAT ID EEPROM node; injectable so detection can be driven from a
        # test without a real Pi (#211).
        self._hat_root = hat_root
        # Built lazily so importing the driver never touches the state
        # directory, and injectable so the observation can be driven from a
        # test without a real clock or a real file.
        self._discharge_observer = discharge_observer

    def _observer(self):
        if self._discharge_observer is None:
            from ..battery_runtime import DischargeObserver

            self._discharge_observer = DischargeObserver()
        return self._discharge_observer

    def product(
        self, raw: Optional[Snapshot] = None, board: Optional[Snapshot] = None
    ) -> Dict[str, Any]:
        board = self.board() if board is None else board
        return detect_product(
            raw,
            variant_path=self._variant_path,
            enclosure_present=(
                bool(board.get("is_raspberry_pi")) or enclosure_detected(raw)
            ),
            hat_root=self._hat_root,
        )

    def power(self, metrics: Optional[Snapshot], board: Snapshot) -> Dict[str, Any]:
        """Power, plus a UPS runtime once a discharge has actually been seen.

        Recording the sample here is deliberate: this is the only place that
        sees the battery percentage at the poll cadence, and a runtime figure
        is worth nothing unless something watched the pack fall. Until it has,
        ``runtime_minutes`` is absent and ``discharge_observed`` is false, so
        the panel states the runtime is unknown rather than estimating it.
        """
        from ..battery_runtime import battery_runtime

        snapshot = power_snapshot(metrics, runner=self._runner)
        # How the last boot ended, beside the flag word that cannot answer it.
        # The appliance could not say it had lost power at all: the owner found
        # out by finding the machine dark. Attached here rather than inside
        # `power_snapshot` because it is a journal read, not a PMIC read, and
        # the two have no reason to share a subprocess call.
        snapshot["previous_shutdown"] = self.previous_shutdown()
        battery = snapshot.get("battery") or {}
        if not battery.get("available"):
            return snapshot
        observer = self._observer()
        observer.record(
            battery.get("percentage"), charging=battery.get("charging")
        )
        snapshot["battery"] = {**battery, **battery_runtime(battery, observer)}
        return snapshot

    def power_telemetry(self) -> bool:
        return shutil.which("vcgencmd") is not None

    def power_actions(self) -> Dict[str, Dict[str, Any]]:
        """Reboot, power off and restart the control plane through systemd-logind.

        A Pironman gets the same host-power path as a bare Pi. It once did not:
        this override installed ``sf_rpi_status.reboot``/``shutdown`` to "prefer
        SunFounder's sequenced shutdown". Two facts, both verified on the
        appliance, made that override broken *and* pointless (#208):

        * ``sf_rpi_status.reboot()`` is literally
          ``os.system('sudo systemctl reboot -i')`` (``.shutdown()`` the
          ``poweroff`` equivalent). The hardware bridge unit sets
          ``NoNewPrivileges=yes``, which makes ``sudo`` fail outright — *"the
          'no new privileges' flag is set, which prevents sudo from running as
          root"*, exit 1 — so the machine never rebooted while the action was
          audited as success. The bridge already runs as root and
          ``systemctl reboot``/``poweroff`` work from it directly, so the
          ``sudo`` wrapper added only a guaranteed failure.
        * There is no sequence left to preserve. These functions are not the
          OLED-off power-button sequence; that lives in ``pironman5.service``,
          which VD-098 has ``mask``ed so Vaelor owns the only control plane. A
          masked service runs nothing, so nothing is sequenced whichever helper
          we call. ``sf_rpi_status`` exposes no non-``sudo`` reboot/shutdown
          entry point to route around this.

        So the base systemd-logind path is the whole story, and this platform
        deliberately does not re-introduce a ``sudo`` wrapper that cannot run
        under ``NoNewPrivileges``.
        """
        return super().power_actions()

    def capabilities(
        self,
        raw: Optional[Snapshot] = None,
        metrics: Optional[Snapshot] = None,
        inventory: Optional[Snapshot] = None,
    ) -> Dict[str, Dict[str, Any]]:
        raw = raw or {}
        metrics = metrics or {}
        peripherals = {
            str(item).lower() for item in raw.get("peripherals", []) or []
        }
        case_fan = bool(
            peripherals.intersection({"gpio_fan_state", "gpio_fan"})
            or "gpio_fan_state" in metrics
        )
        lighting = "ws2812" in peripherals
        oled = any(item == "oled" or item.startswith("oled_") for item in peripherals)
        cpu_fan = bool(
            peripherals.intersection({"pwm_fan_speed", "pwm_fan"})
            or "pwm_fan_speed" in metrics
        )
        battery = metrics.get("battery_percentage") is not None
        missing = "No Pironman enclosure was detected on this Raspberry Pi."
        return {
            "case_fan": capability(case_fan, missing),
            "case_lighting": capability(lighting, missing),
            "oled": capability(
                oled,
                "No Pironman OLED display was detected on the I²C bus.",
            ),
            "cpu_fan": capability(
                cpu_fan,
                "No Raspberry Pi PWM cooling fan was detected.",
            ),
            # On a Pironman the enclosure reports fan speed through the same
            # peripherals that offer control, so the two travel together here.
            # They are still answered separately, because that is a property of
            # this enclosure rather than of fans in general.
            "fan_readings": capability(
                case_fan or cpu_fan,
                "No fan tachometer is reporting on this Raspberry Pi.",
            ),
            "battery": capability(
                battery,
                "No PiPower battery accessory is reporting on this appliance.",
            ),
            "gpu": capability(
                False,
                "The Raspberry Pi has no discrete or unified inference accelerator.",
            ),
            "npu": capability(
                False,
                "The Raspberry Pi has no neural processing unit.",
            ),
        }
