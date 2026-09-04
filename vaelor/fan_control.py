"""Safe control of a thermal cooling fan exposed by the Linux thermal core."""

from __future__ import annotations

import glob
import os
import threading
import time
from pathlib import Path
from typing import Any, Callable, Dict, Optional

DEFAULT_CURVE = [
    {"temperature": 50.0, "percent": 30, "state": 1},
    {"temperature": 60.0, "percent": 50, "state": 2},
    {"temperature": 67.5, "percent": 70, "state": 3},
    {"temperature": 75.0, "percent": 100, "state": 4},
]

#: Cooling-device types that are emphatically **not** fans, even if a loose
#: substring match would accept them. An x86 machine typically exposes one
#: ``Processor`` cooling device per thread — thirty-two of them on the first
#: workstation this ran on — each a passive throttle state with
#: ``max_state == 3``. Treating those as a fan produces a control that looks
#: like a working four-level fan boost and actually throttles the CPU.
NON_FAN_COOLING_TYPES = ("processor", "link_speed", "intel_powerclamp", "idle_inject")

DEFAULT_ABSENT_MESSAGE = (
    "No controllable cooling fan was detected on this machine."
)


class CpuFanController:
    """Maintain an optional, time-limited minimum CPU fan cooling state.

    The Linux thermal governor remains authoritative. A boost only raises
    ``cur_state`` when it is below the requested minimum and never lowers a
    state selected by the governor.
    """

    def __init__(
        self,
        thermal_root: str = "/sys/class/thermal",
        hwmon_root: str = "/sys/devices/platform/cooling_fan/hwmon",
        clock: Callable[[], float] = time.time,
        absent_message: str = DEFAULT_ABSENT_MESSAGE,
        temperature_reader: Optional[Callable[[], Optional[float]]] = None,
    ):
        self.thermal_root = Path(thermal_root)
        self.hwmon_root = Path(hwmon_root)
        self.clock = clock
        # Surfaced verbatim by the API, so it must describe the machine the
        # operator is actually looking at rather than naming a Raspberry Pi.
        self.absent_message = absent_message
        self._temperature_reader = temperature_reader
        self._lock = threading.Lock()
        self._io_lock = threading.Lock()
        self._boost_level: Optional[int] = None
        self._boost_expires_at = 0.0
        self._custom_curve: Optional[list[Dict[str, Any]]] = None
        self._worker: Optional[threading.Thread] = None

    def _read_int(self, path: Path) -> Optional[int]:
        with self._io_lock:
            try:
                return int(path.read_text(encoding="utf-8").strip())
            except (OSError, TypeError, ValueError):
                return None

    def _write_state(self, path: Path, value: int) -> None:
        with self._io_lock:
            path.write_text(str(value), encoding="utf-8")

    def _cooling_device(self) -> Optional[Path]:
        candidates = sorted(self.thermal_root.glob("cooling_device*"))
        for candidate in candidates:
            device_type = ""
            try:
                device_type = (candidate / "type").read_text(encoding="utf-8").strip()
            except OSError:
                pass
            normalized = device_type.lower()
            if any(marker in normalized for marker in NON_FAN_COOLING_TYPES):
                continue
            if (
                "fan" in normalized
                and (candidate / "cur_state").exists()
                and (candidate / "max_state").exists()
            ):
                return candidate
        return None

    def _rpm(self) -> Optional[int]:
        for match in glob.glob(str(self.hwmon_root / "*" / "fan1_input")):
            value = self._read_int(Path(match))
            if value is not None:
                return value
        return None

    def _temperature(self) -> Optional[float]:
        # ``thermal_zone0`` is the CPU on a Raspberry Pi and frequently the
        # ACPI board sensor on x86, so a platform may supply its own labelled
        # reader instead.
        if self._temperature_reader is not None:
            return self._temperature_reader()
        value = self._read_int(self.thermal_root / "thermal_zone0" / "temp")
        return value / 1000 if value is not None else None

    def _automatic_target(self, maximum: int) -> Optional[int]:
        """Reconstruct the kernel fan-map target from the active trip points.

        Writing a temporary boost to ``cur_state`` bypasses the thermal
        governor.  When the boost ends, sysfs does not immediately lower that
        value again, so calculate the state the firmware fan map currently
        calls for instead of leaving the fan pinned at the boost speed.
        """
        zone = self.thermal_root / "thermal_zone0"
        temperature = self._read_int(zone / "temp")
        if temperature is None:
            return None
        thresholds = []
        for type_path in zone.glob("trip_point_*_type"):
            try:
                if type_path.read_text(encoding="utf-8").strip().lower() != "active":
                    continue
            except OSError:
                continue
            prefix = type_path.name[: -len("_type")]
            trip = self._read_int(zone / f"{prefix}_temp")
            hysteresis = self._read_int(zone / f"{prefix}_hyst") or 0
            if trip is not None:
                thresholds.append(trip - hysteresis)
        if not thresholds:
            return None
        target = sum(temperature >= threshold for threshold in sorted(thresholds))
        return max(0, min(maximum, target))

    def _restore_automatic_state(self, device: Optional[Path] = None) -> None:
        device = device or self._cooling_device()
        if device is None:
            return
        maximum = self._read_int(device / "max_state")
        if maximum is None:
            return
        target = self._automatic_target(maximum)
        if target is not None:
            self._write_state(device / "cur_state", target)

    def snapshot(self) -> Dict[str, Any]:
        device = self._cooling_device()
        now = self.clock()
        with self._lock:
            if self._boost_level is not None and now >= self._boost_expires_at:
                self._boost_level = None
                self._boost_expires_at = 0
            boost_level = self._boost_level
            expires_at = self._boost_expires_at
            custom_curve = self._custom_curve
        if device is None:
            return {
                "detected": False,
                "control": "unavailable",
                "mode": "automatic",
                "rpm": self._rpm(),
            }
        current_state = self._read_int(device / "cur_state")
        max_state = self._read_int(device / "max_state")
        return {
            "detected": True,
            "control": "firmware-with-overrides",
            "mode": "boost" if boost_level is not None else "custom" if custom_curve else "automatic",
            "boost_supported": True,
            "custom_curve_supported": True,
            "rpm": self._rpm(),
            "current_state": current_state,
            "max_state": max_state,
            "boost_level": boost_level,
            "boost_expires_at": int(expires_at * 1000) if boost_level is not None else None,
            "temperature": self._temperature(),
            "writable": os.access(device / "cur_state", os.W_OK),
            "curve": custom_curve or DEFAULT_CURVE,
        }

    def set_mode(
        self,
        mode: str,
        *,
        level: Optional[int] = None,
        duration_minutes: int = 15,
        curve: Optional[list[Dict[str, Any]]] = None,
    ) -> Dict[str, Any]:
        normalized = mode.strip().lower()
        if normalized not in {"automatic", "boost", "custom"}:
            raise ValueError("Choose automatic, custom curve, or boost CPU fan mode.")
        # Every mode, including "automatic", needs a device to act on. Letting
        # "automatic" return a snapshot regardless meant "Apply cooling policy"
        # returned 200, wrote `success` to the audit log and told the operator
        # the policy had been applied on a machine with no controllable fan.
        device = self._cooling_device()
        if device is None:
            raise RuntimeError(self.absent_message)
        if normalized == "automatic":
            with self._lock:
                self._boost_level = None
                self._boost_expires_at = 0
                self._custom_curve = None
            self._restore_automatic_state(device)
            return self.snapshot()
        max_state = self._read_int(device / "max_state")
        if max_state is None or max_state < 1:
            raise RuntimeError("The CPU fan cooling states could not be read.")
        if not os.access(device / "cur_state", os.W_OK):
            raise PermissionError("The control-plane service cannot write the CPU fan state.")
        if normalized == "custom":
            validated = self._validate_curve(curve, max_state)
            with self._lock:
                self._boost_level = None
                self._boost_expires_at = 0
                self._custom_curve = validated
            self._apply_custom()
            self._ensure_worker()
            return self.snapshot()
        if not isinstance(level, int) or level < 1 or level > max_state:
            raise ValueError("Choose a CPU fan level between 1 and {}.".format(max_state))
        if (
            not isinstance(duration_minutes, int)
            or duration_minutes < 1
            or duration_minutes > 60
        ):
            raise ValueError("Choose a boost duration between 1 and 60 minutes.")
        with self._lock:
            self._custom_curve = None
            self._boost_level = level
            self._boost_expires_at = self.clock() + (duration_minutes * 60)
        self._apply_boost()
        self._ensure_worker()
        return self.snapshot()

    @staticmethod
    def _validate_curve(
        curve: Optional[list[Dict[str, Any]]], maximum: int
    ) -> list[Dict[str, Any]]:
        if not isinstance(curve, list) or len(curve) != maximum:
            raise ValueError("Provide one temperature threshold for every CPU fan level.")
        validated = []
        previous = 34.0
        for state, point in enumerate(curve, 1):
            temperature = point.get("temperature") if isinstance(point, dict) else None
            if not isinstance(temperature, (int, float)) or not previous < temperature < 80:
                raise ValueError("Fan temperatures must increase and stay between 35°C and 79°C.")
            percent = point.get("percent", round(state / maximum * 100))
            if not isinstance(percent, (int, float)) or not 1 <= percent <= 100:
                raise ValueError("Fan percentages must be between 1 and 100.")
            validated.append({
                "temperature": round(float(temperature), 1),
                "percent": round(float(percent)),
                "state": state,
            })
            previous = float(temperature)
        return validated

    def _apply_custom(self) -> None:
        device = self._cooling_device()
        temperature = self._temperature()
        maximum = self._read_int(device / "max_state") if device else None
        with self._lock:
            curve_token = self._custom_curve
            curve = list(curve_token or [])
        if device is None or temperature is None or maximum is None or not curve:
            return
        target = sum(temperature >= float(point["temperature"]) for point in curve)
        if temperature >= 80:
            target = maximum
        with self._lock:
            if self._custom_curve is not curve_token:
                return
        self._write_state(device / "cur_state", min(maximum, target))

    def _apply_boost(self) -> None:
        device = self._cooling_device()
        if device is None:
            return
        with self._lock:
            target = self._boost_level
            expires_at = self._boost_expires_at
        if target is None or self.clock() >= expires_at:
            with self._lock:
                self._boost_level = None
                self._boost_expires_at = 0
            self._restore_automatic_state(device)
            return
        current = self._read_int(device / "cur_state")
        maximum = self._read_int(device / "max_state")
        temperature = self._temperature()
        if maximum is None:
            return
        safe_target = maximum if temperature is not None and temperature >= 80 else target
        with self._lock:
            if (
                self._boost_level != target
                or self._boost_expires_at != expires_at
            ):
                return
        if current is None or current < safe_target:
            self._write_state(device / "cur_state", safe_target)

    def _ensure_worker(self) -> None:
        if self._worker is not None and self._worker.is_alive():
            return
        self._worker = threading.Thread(
            target=self._run,
            name="pm-cpu-fan-boost",
            daemon=True,
        )
        self._worker.start()

    def _run(self) -> None:
        while True:
            # set_mode applies the requested state synchronously. Delay the
            # first maintenance write to avoid an immediate redundant sysfs
            # write racing the response reader.
            time.sleep(1)
            with self._lock:
                active = self._boost_level is not None or self._custom_curve is not None
            if not active:
                return
            try:
                if self._custom_curve is not None:
                    self._apply_custom()
                else:
                    self._apply_boost()
            except OSError:
                with self._lock:
                    self._boost_level = None
                    self._boost_expires_at = 0
                return
