"""The dependency-free flatten used by both the store writer and the worker emitter.

`data_logger` writes the controller's own rows and imports the third-party
``influxdb`` client at module top for its writer. The E2b worker telemetry
emitter has to flatten a machine reading into the *same* flat scalar row the
controller stores — schema-identical, so a worker's ``history`` rows and the
controller's cannot drift — but it must drag in **no** ``influxdb`` and no other
third party, because it runs on a lean worker whose only pre-installed
interpreter is the system ``python3``.

So the flatten lives here, on its own, importing nothing outside the standard
library. `data_logger` imports it (keeping one copy, so a change to how a
reading is stored reaches both the controller and every worker at once), and the
emitter's zipapp bundles it without ever touching ``data_logger``.

Both public names keep an underscore-prefixed alias, because that is how
`data_logger` and its tests have always spelled them.
"""

from __future__ import annotations

import math
from typing import Any, Dict, List, Optional

from .telemetry_bounds import POWER_CAP_KEY, REFUSED_FIELDS_KEY, apply_bounds


def storable(value: Any) -> Optional[Any]:
    """One scalar InfluxDB can hold as a field, or ``None`` to drop it.

    ``bool`` is checked before ``int`` because it is a subclass of it, and is
    written as 0/1 so a field that reads as a number on one platform and a flag
    on another keeps a single stored type. Numbers and strings pass through;
    everything else (a list, a ``None``, a nested object already handled by the
    caller) returns ``None`` so the writer drops it rather than handing InfluxDB
    a value it cannot store.
    """
    if isinstance(value, bool):
        return int(value)
    if isinstance(value, int):
        # One stored type per numeric field. A telemetry field can read `int 0`
        # at exact idle (`round(max(0, 0.0), 1)` collapses to int) and `float`
        # otherwise; InfluxDB rejects a write whose field type differs from the
        # stored one and drops that interval's whole row, so numbers are stored
        # as float uniformly.
        return float(value)
    if isinstance(value, float):
        # NaN and inf are not valid InfluxDB line-protocol field values; the
        # server rejects the write. A non-finite value in the FIRST sample would
        # fail retention startup for the whole box (VD-095 residual), so drop it
        # here rather than hand it to the store.
        return value if math.isfinite(value) else None
    if isinstance(value, str):
        return value
    return None


def flatten_storable(
    data: Dict[str, Any], prefix: str = "", dropped: Optional[List[str]] = None,
) -> Dict[str, Any]:
    """Flatten a machine reading into the flat scalar row InfluxDB stores.

    The writer used to keep only top-level scalars and drop every ``list`` and
    ``dict``. On a Raspberry Pi the HAT reads back flat scalars, so that kept
    everything that mattered; on an x86 workstation the same reading arrives
    either empty or with its numbers nested one level down - a HAT-less host's
    telemetry is assembled from ``/proc`` and ``/sys``, and a provider can group
    it as ``{"cpu": {"percent": 3, "temperature": 33}}``. Dropping every dict
    threw those numbers away, so ``get_data`` came back empty and the store
    recorded nothing every interval (the "nothing to record yet" warning on the
    x86 box).

    Flattening is platform-agnostic and is why this is not an ``if workstation``
    branch: a flat Pi reading passes through unchanged - there are no nested
    dicts to descend into - and a nested reading yields ``cpu_percent``,
    ``cpu_temperature`` and the like. Lists and any value that is not a number,
    bool or string are dropped rather than crashing the writer: InfluxDB cannot
    store them as a field, and a first sample that raised would read to the
    caller as "cannot record" rather than the empty reading it actually is.

    **A physically implausible value is dropped here** (VD-147): the finished
    row goes through the one bounds table (`telemetry_bounds`), for the
    controller's writer and the worker emitter alike, since both build their
    row with this function. Nothing is clamped; the dropped field names are
    appended to ``dropped`` when the caller passes a list, so it can count them.
    """
    flat = _flatten(data, prefix)
    kept, removed = apply_bounds(flat, flat.get(POWER_CAP_KEY))
    if dropped is not None:
        # A reading refused before the row was built (a power figure past its
        # own cap) is counted like one dropped here (pass-3 review).
        refused = data.get(REFUSED_FIELDS_KEY) if isinstance(data, dict) else None
        dropped.extend(str(name) for name in (refused or []) if isinstance(name, str))
        dropped.extend(removed)
    return kept


def _flatten(data: Dict[str, Any], prefix: str) -> Dict[str, Any]:
    """The flat scalar row, before the bounds are applied to its final names."""
    flat: Dict[str, Any] = {}
    for key, value in data.items():
        name = "{}{}".format(prefix, key)
        if isinstance(value, dict):
            flat.update(_flatten(value, "{}_".format(name)))
            continue
        stored = storable(value)
        if stored is not None:
            flat[name] = stored
    return flat


#: The names `data_logger` and its tests have always imported.
_storable = storable
_flatten_storable = flatten_storable
