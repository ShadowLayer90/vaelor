"""Battery runtime from an observed discharge, or nothing at all.

A UPS runtime figure computed from a capacity rating and a current draw is a
number nobody measured. It looks authoritative on a panel, it is wrong by a
factor that depends on the pack's age and temperature, and an operator plans a
shutdown around it. So Vaelor does not compute one.

What it does instead is watch. While the appliance is running on battery, each
percentage reading is recorded. Once the run is long enough and the drop is
large enough to give a rate that means something, the rate becomes a real
observation and a runtime can be stated — with ``discharge_observed`` true,
because the caller is entitled to know which of the two it is looking at.
Before that, both keys are absent and the panel says the runtime is unknown.

The store is deliberately small and disposable. It describes this pack on this
appliance; losing it costs one future observation, and inventing a runtime to
avoid that trade would be the exact defect this module exists to prevent.
"""

from __future__ import annotations

import json
import threading
import time
from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional

from .runtime_paths import state_path


#: A discharge is only a measurement once it has run this long. Below this the
#: rate is dominated by the sampling interval and by whatever the appliance
#: happened to be doing in that minute.
MINIMUM_OBSERVED_SECONDS = 300.0

#: ...and once the charge has actually fallen this far. A 1% drop is one
#: quantisation step on most gauges, so a rate derived from it is noise.
MINIMUM_OBSERVED_DROP_PERCENT = 3.0

#: ...across at least this many samples, so one bad reading cannot define a
#: rate on its own.
MINIMUM_OBSERVED_SAMPLES = 4

#: Samples kept for the current discharge run. Bounded so an appliance left on
#: battery cannot grow the file without limit; the rate is computed from the
#: ends of the run, so dropping the middle costs nothing.
MAX_SAMPLES = 240

#: A gap longer than this means the appliance was off, asleep, or the poller
#: stopped, so the samples either side do not describe one continuous
#: discharge and the run restarts.
MAX_SAMPLE_GAP_SECONDS = 900.0


def store_path() -> str:
    """Where the discharge observation lives between restarts."""
    import os

    return os.environ.get("VAELOR_BATTERY_OBSERVATION_PATH", "").strip() or state_path(
        "battery-discharge.json"
    )


class DischargeObserver:
    """Record battery samples and report a rate only once one is measured."""

    def __init__(self, path: Optional[str] = None):
        self._path = path
        self._lock = threading.Lock()
        self._samples: Optional[List[Dict[str, float]]] = None

    # -- persistence ------------------------------------------------------

    def _file(self) -> Path:
        return Path(self._path or store_path())

    def _load(self) -> List[Dict[str, float]]:
        if self._samples is not None:
            return self._samples
        samples: List[Dict[str, float]] = []
        try:
            raw = json.loads(self._file().read_text(encoding="utf-8"))
            for entry in raw.get("samples", []):
                at = float(entry["at"])
                percentage = float(entry["percentage"])
                if at == at and percentage == percentage:
                    samples.append({"at": at, "percentage": percentage})
        except (AttributeError, KeyError, OSError, TypeError, ValueError):
            # An unreadable file means "nothing observed yet", which is a state
            # this module already handles correctly. It must never raise into a
            # telemetry poll.
            samples = []
        self._samples = samples
        return samples

    def _persist(self) -> None:
        path = self._file()
        try:
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text(
                json.dumps({"samples": self._samples or []}, separators=(",", ":")),
                encoding="utf-8",
            )
        except OSError:
            # An appliance with an unwritable state directory still benefits
            # from the in-memory observation for the life of the process.
            pass

    def reset(self) -> None:
        with self._lock:
            self._samples = []
            self._persist()

    # -- observation ------------------------------------------------------

    def record(
        self,
        percentage: Any,
        *,
        charging: Any = None,
        now: Optional[float] = None,
    ) -> None:
        """Note one battery reading.

        Charging, or an unreadable percentage, ends the current discharge run:
        a rate measured across a charge is not a discharge rate.
        """
        moment = time.time() if now is None else float(now)
        value = (
            float(percentage)
            if isinstance(percentage, (int, float)) and not isinstance(percentage, bool)
            else None
        )
        with self._lock:
            samples = self._load()
            if value is None or charging is True:
                if samples:
                    self._samples = []
                    self._persist()
                return
            if samples:
                last = samples[-1]
                # A gap, or a rise in charge, means this is not a continuation
                # of the same discharge.
                if (
                    moment - last["at"] > MAX_SAMPLE_GAP_SECONDS
                    or value > last["percentage"]
                    or moment < last["at"]
                ):
                    samples = []
            samples.append({"at": moment, "percentage": value})
            self._samples = samples[-MAX_SAMPLES:]
            self._persist()

    def observation(self, *, now: Optional[float] = None) -> Dict[str, Any]:
        """The measured discharge rate, or a statement that there isn't one."""
        with self._lock:
            samples = list(self._load())
        if len(samples) < MINIMUM_OBSERVED_SAMPLES:
            return _unobserved(
                "Vaelor has not yet watched this appliance run on battery long "
                "enough to measure how fast it discharges."
            )
        first, last = samples[0], samples[-1]
        elapsed = last["at"] - first["at"]
        dropped = first["percentage"] - last["percentage"]
        if elapsed < MINIMUM_OBSERVED_SECONDS or dropped < MINIMUM_OBSERVED_DROP_PERCENT:
            return _unobserved((
                "This discharge has run {:.0f} of the {:.0f} seconds and used "
                "{:.1f} of the {:.1f} percent needed before the rate means "
                "anything."
            ).format(
                max(0.0, elapsed), MINIMUM_OBSERVED_SECONDS,
                max(0.0, dropped), MINIMUM_OBSERVED_DROP_PERCENT,
            ))
        percent_per_minute = dropped / (elapsed / 60.0)
        if percent_per_minute <= 0:
            return _unobserved("The observed discharge rate is not usable.")
        return {
            "discharge_observed": True,
            "percent_per_minute": round(percent_per_minute, 4),
            "observed_seconds": round(elapsed, 1),
            "observed_drop_percent": round(dropped, 2),
            "samples": len(samples),
            "observed_at": last["at"],
            "reason": (
                "Measured: this appliance lost {:.1f}% of its charge over "
                "{:.0f} seconds on battery, across {} readings."
            ).format(dropped, elapsed, len(samples)),
        }

    def runtime_minutes(
        self, percentage: Any, *, now: Optional[float] = None
    ) -> Optional[int]:
        """Minutes left at the observed rate, or ``None`` if none was observed."""
        observation = self.observation(now=now)
        if not observation.get("discharge_observed"):
            return None
        if isinstance(percentage, bool) or not isinstance(percentage, (int, float)):
            return None
        remaining = max(0.0, float(percentage))
        rate = float(observation["percent_per_minute"])
        if rate <= 0:
            return None
        return int(remaining / rate)


def _unobserved(reason: str) -> Dict[str, Any]:
    return {
        "discharge_observed": False,
        "percent_per_minute": None,
        "observed_seconds": 0.0,
        "observed_drop_percent": 0.0,
        "samples": 0,
        "reason": reason,
    }


def battery_runtime(
    battery: Mapping[str, Any], observer: "DischargeObserver",
    *, now: Optional[float] = None,
) -> Dict[str, Any]:
    """The runtime keys to merge into a battery payload.

    Emits ``runtime_minutes`` **only** alongside ``discharge_observed: True``.
    The two travel together on purpose: a runtime without the flag is exactly
    the computed-from-nothing estimate this refuses to produce.
    """
    observation = observer.observation(now=now)
    result: Dict[str, Any] = {
        "discharge_observed": bool(observation.get("discharge_observed")),
        "runtime_reason": observation.get("reason", ""),
    }
    if not result["discharge_observed"]:
        return result
    minutes = observer.runtime_minutes(battery.get("percentage"), now=now)
    if minutes is None:
        result["discharge_observed"] = False
        result["runtime_reason"] = (
            "A discharge rate was measured, but this battery is not reporting a "
            "charge percentage to apply it to."
        )
        return result
    result["runtime_minutes"] = minutes
    result["discharge_percent_per_minute"] = observation["percent_per_minute"]
    return result
