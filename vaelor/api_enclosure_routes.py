"""Enclosure control routes: case fans, CPU fan, case lighting, OLED.

Every mutation in this module writes to hardware that only exists on some
machines. Each one is gated on the platform driver's capability answer and
fails closed with that driver's reason. Before the gate existed these routes
called an enclosure bridge that raised on any host without one, and Flask
turned the exception into an HTTP 500 with an empty body.
"""

from __future__ import annotations

from typing import Any, Dict

from flask import g, request

from .api_common import ApiContext, payload as _payload
from .api_machine import capability_gate, platform_driver
from .device_identity import read_override, selected_product
from .platforms import PRODUCTS


FAN_PROFILES = [
    {"id": 0, "name": "Continuous", "description": "Keep enclosure airflow running at every CPU fan level."},
    {"id": 1, "name": "Early start", "description": "Start enclosure airflow with CPU cooling level 1 (about 50°C)."},
    {"id": 2, "name": "Normal start", "description": "Start enclosure airflow with CPU cooling level 2 (about 60°C)."},
    {"id": 3, "name": "Late start", "description": "Start enclosure airflow with CPU cooling level 3 (about 67.5°C)."},
    {"id": 4, "name": "Emergency only", "description": "Start enclosure airflow only at maximum CPU cooling (about 75°C)."},
]
CPU_FAN_CURVE = [
    {"temperature": 50, "percent": 30, "state": 1},
    {"temperature": 60, "percent": 50, "state": 2},
    {"temperature": 67.5, "percent": 70, "state": 3},
    {"temperature": 75, "percent": 100, "state": 4},
]
RGB_STYLES = [
    {"id": "solid", "name": "Solid"},
    {"id": "breathing", "name": "Breathing"},
    {"id": "flow", "name": "Flow"},
    {"id": "flow_reverse", "name": "Reverse flow"},
    {"id": "rainbow", "name": "Rainbow"},
    {"id": "rainbow_reverse", "name": "Reverse rainbow"},
    {"id": "hue_cycle", "name": "Hue cycle"},
]

BRIDGE_UNAVAILABLE = (
    "The enclosure hardware service is not running on this machine, so the "
    "setting was not applied."
)


def register_enclosure_routes(context: ApiContext) -> None:
    blueprint = context.blueprint
    callbacks = context.callbacks
    security = context.security
    require_auth = context.require_auth

    def _apply_config(patch: Dict[str, Any]):
        """Write enclosure configuration, converting an absent bridge to 503.

        ``update_config`` is always supplied by the control plane, so the
        ``is None`` guards below never fire on a host whose bridge is missing.
        The callback raises there instead, and an uncaught ``RuntimeError``
        inside a Flask view is a 500. A control that did nothing must report
        that it did nothing.
        """
        update_config = callbacks.get("update_config")
        if update_config is None:
            return False, BRIDGE_UNAVAILABLE
        try:
            update_config({"system": patch})
        except (OSError, RuntimeError) as error:
            return False, str(error) or BRIDGE_UNAVAILABLE
        return True, ""

    def fan_state():
        device = callbacks["device_info"]() or {}
        metrics = callbacks["current_data"]() or {}
        read_config = callbacks.get("read_config")
        config = read_config() if read_config else {}
        system_config = config.get("system", {}) if isinstance(config, dict) else {}
        selected_device = dict(device)
        override = read_override() or system_config.get("device_variant_override")
        if override in PRODUCTS:
            selected_device["id"] = override
        # The product must come from the same discovery `/device` uses. When
        # this route detected the product on its own it omitted the enclosure
        # fact and every generic host was told it had two Pironman case fans.
        product = selected_product(override, PRODUCTS) or platform_driver(
            callbacks
        ).product(selected_device)
        peripherals = set(device.get("peripherals", []))
        pwm_detected = (
            "pwm_fan_speed" in peripherals
            or "pwm_fan" in peripherals
            or "pwm_fan_speed" in metrics
        )
        gpio_detected = (
            "gpio_fan_state" in peripherals
            or "gpio_fan" in peripherals
            or "gpio_fan_state" in metrics
        )
        mode = system_config.get("gpio_fan_mode", 0)
        if not isinstance(mode, int) or mode not in range(len(FAN_PROFILES)):
            mode = 0
        cpu_status_callback = callbacks.get("cpu_fan_status")
        cpu_status: Dict[str, Any] = {}
        if cpu_status_callback is not None:
            try:
                reported_status = cpu_status_callback()
                if isinstance(reported_status, dict):
                    cpu_status = reported_status
            except (OSError, RuntimeError, ValueError):
                cpu_status = {}
        cpu_fan = {
            "id": "cpu-pwm",
            "name": "CPU PWM fan",
            "kind": "pwm",
            "detected": pwm_detected,
            "control": "system-managed",
            "rpm": metrics.get("pwm_fan_speed"),
            "mode": "automatic",
            "curve": CPU_FAN_CURVE,
            "safety_limit": 80,
        }
        cpu_fan.update(cpu_status)
        if cpu_fan.get("rpm") is None:
            cpu_fan["rpm"] = metrics.get("pwm_fan_speed")
        reported_case_running = (
            bool(metrics.get("gpio_fan_state"))
            if metrics.get("gpio_fan_state") is not None
            else None
        )
        cpu_state = cpu_fan.get("current_state")
        commanded_case_running = (
            bool(cpu_state >= mode)
            if gpio_detected and isinstance(cpu_state, int)
            else True if gpio_detected and mode == 0
            else None
        )
        return {
            "profiles": FAN_PROFILES,
            "fans": [
                cpu_fan,
                {
                    "id": "case-gpio",
                    "name": "Enclosure airflow",
                    "kind": "gpio",
                    "detected": gpio_detected,
                    "control": "profile",
                    "running": (
                        commanded_case_running
                        if commanded_case_running is not None
                        else reported_case_running
                    ),
                    "commanded_running": commanded_case_running,
                    "reported_running": reported_case_running,
                    "state_source": (
                        "profile-policy"
                        if commanded_case_running is not None
                        else "telemetry"
                    ),
                    "fan_count": product["fan_count"],
                    "shared_control": True,
                    "rpm_available": False,
                    "profile": mode,
                    "led": system_config.get("gpio_fan_led", "follow"),
                    "pin": system_config.get("gpio_fan_pin"),
                },
            ],
        }

    @blueprint.get("/fans")
    @require_auth("viewer")
    def fans():
        return _payload(fan_state())

    @blueprint.patch("/fans/case")
    @require_auth("operator", csrf=True)
    def update_case_fan():
        body = request.get_json(silent=True) or {}
        patch: Dict[str, Any] = {}
        if "profile" in body:
            profile = body["profile"]
            if isinstance(profile, str):
                profile = next(
                    (
                        item["id"]
                        for item in FAN_PROFILES
                        if item["name"].lower() == profile.strip().lower()
                    ),
                    None,
                )
            if not isinstance(profile, int) or profile not in range(len(FAN_PROFILES)):
                return _payload(
                    error={
                        "code": "invalid_fan_profile",
                        "message": "Choose a supported case-fan profile.",
                    },
                    status=400,
                )
            patch["gpio_fan_mode"] = profile
        if "led" in body:
            led = str(body["led"]).strip().lower()
            if led not in {"follow", "on", "off"}:
                return _payload(
                    error={
                        "code": "invalid_fan_led_mode",
                        "message": "Choose follow, on, or off for the fan LED.",
                    },
                    status=400,
                )
            patch["gpio_fan_led"] = led
        if not patch:
            return _payload(
                error={
                    "code": "empty_fan_update",
                    "message": "Choose a fan profile or LED mode to update.",
                },
                status=400,
            )
        unavailable = capability_gate(
            callbacks,
            "case_fan",
            code="fan_control_unavailable",
            fallback="Fan control is not available on this appliance.",
        )
        if unavailable is not None:
            return _payload(error=unavailable, status=503)
        applied, reason = _apply_config(patch)
        if not applied:
            return _payload(
                error={"code": "fan_control_unavailable", "message": reason},
                status=503,
            )
        security.audit(
            g.auth_session.username,
            "fan.case.update",
            "success",
            target="case-gpio",
            remote_addr=request.remote_addr or "",
            details=patch,
        )
        return _payload({"accepted": True, "updated": patch, **fan_state()})

    @blueprint.patch("/fans/cpu")
    @require_auth("operator", csrf=True)
    def update_cpu_fan():
        body = request.get_json(silent=True) or {}
        mode = str(body.get("mode", "")).strip().lower()
        if mode not in {"automatic", "boost", "custom"}:
            return _payload(
                error={
                    "code": "invalid_cpu_fan_mode",
                    "message": "Choose automatic, custom curve, or boost CPU fan mode.",
                },
                status=400,
            )

        level = body.get("level")
        curve = body.get("curve")
        duration_minutes = body.get("duration_minutes", 15)
        if mode == "boost":
            current = fan_state()["fans"][0]
            maximum = current.get("max_state")
            if (
                not isinstance(level, int)
                or not isinstance(maximum, int)
                or level < 1
                or level > maximum
            ):
                return _payload(
                    error={
                        "code": "invalid_cpu_fan_level",
                        "message": "Choose a supported CPU fan boost level.",
                    },
                    status=400,
                )
            if (
                not isinstance(duration_minutes, int)
                or duration_minutes < 1
                or duration_minutes > 60
            ):
                return _payload(
                    error={
                        "code": "invalid_cpu_fan_duration",
                        "message": "Choose a boost duration between 1 and 60 minutes.",
                    },
                    status=400,
                )

        update_fan = callbacks.get("cpu_fan_update")
        if update_fan is None:
            return _payload(
                error={
                    "code": "cpu_fan_control_unavailable",
                    "message": "CPU fan control is not available on this appliance.",
                },
                status=503,
            )
        try:
            arguments: Dict[str, Any] = {
                "level": level if mode == "boost" else None,
                "duration_minutes": duration_minutes,
            }
            if mode == "custom":
                arguments["curve"] = curve
            update_fan(mode, **arguments)
        except ValueError as error:
            return _payload(
                error={"code": "invalid_cpu_fan_request", "message": str(error)},
                status=400,
            )
        except PermissionError:
            return _payload(
                error={
                    "code": "cpu_fan_permission_denied",
                    "message": "The control-plane service cannot write the CPU fan state.",
                },
                status=503,
            )
        except (OSError, RuntimeError) as error:
            return _payload(
                error={"code": "cpu_fan_unavailable", "message": str(error)},
                status=503,
            )
        persisted: Dict[str, Any] = {"cpu_fan_mode": mode}
        if mode == "custom":
            persisted["cpu_fan_curve"] = curve
        # The fan itself was already commanded successfully; failing to persist
        # the preference must not be reported as a failed cooling change.
        _apply_config(persisted)

        details = {"mode": mode}
        if mode == "boost":
            details.update({"level": level, "duration_minutes": duration_minutes})
        elif mode == "custom":
            details["curve"] = curve
        security.audit(
            g.auth_session.username,
            "fan.cpu.update",
            "success",
            target="cpu-pwm",
            remote_addr=request.remote_addr or "",
            details=details,
        )
        return _payload({"accepted": True, **fan_state()})

    def lighting_state():
        device = callbacks["device_info"]() or {}
        read_config = callbacks.get("read_config")
        config = read_config() if read_config else {}
        system_config = config.get("system", {}) if isinstance(config, dict) else {}
        peripherals = set(device.get("peripherals", []))
        # Discovery only. A stored `rgb_*` key is a preference somebody once
        # saved, or a default a package shipped, or a leftover from an
        # enclosure that has since been removed - it is not evidence that a LED
        # strip is wired to this machine. Treating it as evidence returned
        # `hardware: "WS2812 SPI strip", led_count: 4` on a mini PC whose own
        # /system/machine says case lighting is unavailable, and rendered a
        # live console whose Save could only ever 503.
        detected = "ws2812" in peripherals
        # The platform driver is the authority on whether this machine can have
        # case lighting at all, and PATCH already refuses on its verdict. GET
        # must agree, or the interface offers a console whose Save can only 503.
        gate = capability_gate(
            callbacks,
            "case_lighting",
            code="lighting_control_unavailable",
            fallback="Case lighting control is not available.",
        )
        if gate is not None:
            detected = False
        if not detected:
            # Returning WS2812 defaults for hardware that is not present is how
            # a machine with no lighting rendered a four-LED case preview.
            return {
                "detected": False,
                "hardware": None,
                "led_count": None,
                "enabled": None,
                "color": None,
                "brightness": None,
                "speed": None,
                "style": None,
                "styles": RGB_STYLES,
                # The same sentence PATCH would refuse with, so the interface
                # can explain the absence instead of rendering a dead control.
                "reason": (
                    gate.get("message", "")
                    if gate is not None
                    else "No addressable lighting strip was discovered on this machine."
                ),
            }
        return {
            "detected": True,
            "hardware": "WS2812 SPI strip",
            "led_count": system_config.get("rgb_led_count", 4),
            "enabled": system_config.get("rgb_enable", True),
            "color": system_config.get("rgb_color", "#00ffff"),
            "brightness": system_config.get("rgb_brightness", 100),
            "speed": system_config.get("rgb_speed", 50),
            "style": system_config.get("rgb_style", "breathing"),
            "styles": RGB_STYLES,
        }

    @blueprint.get("/lighting")
    @require_auth("viewer")
    def lighting():
        return _payload(lighting_state())

    @blueprint.patch("/lighting")
    @require_auth("operator", csrf=True)
    def update_lighting():
        body = request.get_json(silent=True) or {}
        patch: Dict[str, Any] = {}
        if "enabled" in body:
            if not isinstance(body["enabled"], bool):
                return _payload(
                    error={
                        "code": "invalid_rgb_power",
                        "message": "Choose whether the case lighting is on or off.",
                    },
                    status=400,
                )
            patch["rgb_enable"] = body["enabled"]
        if "color" in body:
            color = str(body["color"]).strip().lower()
            if not (
                len(color) == 7
                and color.startswith("#")
                and all(character in "0123456789abcdef" for character in color[1:])
            ):
                return _payload(
                    error={
                        "code": "invalid_rgb_color",
                        "message": "Choose a valid RGB color.",
                    },
                    status=400,
                )
            patch["rgb_color"] = color
        if "style" in body:
            style = str(body["style"]).strip().lower()
            if style not in {item["id"] for item in RGB_STYLES}:
                return _payload(
                    error={
                        "code": "invalid_rgb_style",
                        "message": "Choose a supported lighting effect.",
                    },
                    status=400,
                )
            patch["rgb_style"] = style
        for request_key, config_key, label in (
            ("brightness", "rgb_brightness", "brightness"),
            ("speed", "rgb_speed", "animation speed"),
        ):
            if request_key in body:
                value = body[request_key]
                if not isinstance(value, int) or isinstance(value, bool) or not 0 <= value <= 100:
                    return _payload(
                        error={
                            "code": "invalid_rgb_{}".format(request_key),
                            "message": "Choose {} between 0 and 100.".format(label),
                        },
                        status=400,
                    )
                patch[config_key] = value
        if not patch:
            return _payload(
                error={
                    "code": "empty_lighting_update",
                    "message": "Choose a lighting setting to update.",
                },
                status=400,
            )
        unavailable = capability_gate(
            callbacks,
            "case_lighting",
            code="lighting_control_unavailable",
            fallback="Case lighting control is not available.",
        )
        if unavailable is not None:
            return _payload(error=unavailable, status=503)
        applied, reason = _apply_config(patch)
        if not applied:
            return _payload(
                error={"code": "lighting_control_unavailable", "message": reason},
                status=503,
            )
        security.audit(
            g.auth_session.username,
            "lighting.update",
            "success",
            target="ws2812",
            remote_addr=request.remote_addr or "",
            details=patch,
        )
        return _payload({"accepted": True, "updated": patch, **lighting_state()})

    def _display_payload():
        read_config = callbacks.get("read_config")
        config = ((read_config() if read_config else {}) or {}).get("system", {})
        device = callbacks.get("device_info", lambda: {})() or {}
        peripherals = set(device.get("peripherals", []))
        detected = any(
            peripheral == "oled" or peripheral.startswith("oled_")
            for peripheral in peripherals
        )
        return {
            "detected": detected,
            "hardware": "SSD1306 128×64 OLED" if detected else None,
            "bus": "I²C 0x3C" if detected else None,
            "enabled": bool(config.get("oled_enable", False)),
            "rotation": int(config.get("oled_rotation", 0)),
            "sleep_timeout": int(config.get("oled_sleep_timeout", 0)),
            "pages": list(config.get("oled_pages", [])),
            "disk": config.get("oled_disk"),
            "network_interface": config.get("oled_network_interface"),
        }

    @blueprint.get("/system/display")
    @require_auth("viewer")
    def system_display():
        return _payload(_display_payload())

    @blueprint.patch("/system/display")
    @require_auth("operator", csrf=True)
    def system_display_update():
        body = request.get_json(silent=True) or {}
        patch = {}
        if "enabled" in body:
            if not isinstance(body["enabled"], bool):
                return _payload(error={"code": "invalid_display_setting", "message": "Display power must be on or off."}, status=400)
            patch["oled_enable"] = body["enabled"]
        if "rotation" in body:
            if body["rotation"] not in (0, 180):
                return _payload(error={"code": "invalid_display_setting", "message": "Choose 0° or 180° rotation."}, status=400)
            patch["oled_rotation"] = body["rotation"]
        if "sleep_timeout" in body:
            try:
                timeout = int(body["sleep_timeout"])
            except (TypeError, ValueError):
                timeout = -1
            if timeout not in (0, 10, 30, 60, 300, 600, 1800, 3600):
                return _payload(error={"code": "invalid_display_setting", "message": "Choose a supported display sleep time."}, status=400)
            patch["oled_sleep_timeout"] = timeout
        if not patch:
            return _payload(error={"code": "invalid_display_setting", "message": "Choose a display setting to change."}, status=400)
        unavailable = capability_gate(
            callbacks,
            "oled",
            code="display_control_unavailable",
            fallback="This machine has no display panel to configure.",
        )
        if unavailable is not None:
            return _payload(error=unavailable, status=503)
        applied, reason = _apply_config(patch)
        if not applied:
            return _payload(
                error={"code": "display_control_unavailable", "message": reason},
                status=503,
            )
        security.audit(
            g.auth_session.username, "system.display.update", "success",
            target="oled", remote_addr=request.remote_addr or "",
            details={"fields": sorted(patch)},
        )
        return _payload(_display_payload())
