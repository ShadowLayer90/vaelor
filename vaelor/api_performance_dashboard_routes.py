"""`/api/v2/cluster/performance/dashboard` - the Performance dashboard's two layers (VD-147 S4, §3.1).

* ``GET …/dashboard/now`` - the now layer, polled every ten seconds: the engine,
  each machine's freshness, the hottest-GPU and busiest-CPU tiles, every
  panel's freshness.
* ``GET …/dashboard?range=1h`` - the range layer, polled every
  ``max(20 s, step)``: the step, the panels, tiles T1/T2/T3/T5, the events.
  ``range`` must be one of the offered ranges (the step table owns them); any
  other value is a 400 with a sentence.

Both use the same read authorisation as ``/cluster/performance``. One response
per ``(layer, range)`` is kept for ``min(step / 2, 5 s)`` (:data:`CACHE_SECONDS`),
so several open tabs cannot multiply the load on the stores. A key is built by
one request at a time (pass-4 review): the others wait for that build rather
than start their own, and the now layer's reads are gathered once and shared
by the now route and every range route inside the same window.
"""

from __future__ import annotations

import threading
import time
from typing import Any, Callable, Dict, Tuple

from flask import request

from .api_common import ApiContext, payload as _payload
from .performance_dashboard import DEFAULT_RANGE, build_now, offered_ranges, range_refused
from .performance_dashboard_panels import build_range
from .performance_dashboard_source import gather_now, gather_range
from .serving_metrics import SERVING_POLL_SECONDS

#: The longest a response is reused, in seconds.
CACHE_SECONDS = 5.0

#: How long a request waits for another request's build of the same key before
#: building it itself.
BUILD_WAIT_SECONDS = 15.0

_CACHE: Dict[Tuple[str, str], Tuple[float, Any]] = {}
_CACHE_LOCK = threading.Lock()
_BUILDING: Dict[Tuple[str, str], threading.Event] = {}


def cache_seconds(step: float) -> float:
    return min(float(step) / 2.0, CACHE_SECONDS)


def cached(key: Tuple[str, str], step: float, build: Callable[[], Any],
           clock: Callable[[], float] = time.monotonic) -> Any:
    """One built response per key, reused for :func:`cache_seconds` of ``step``; one build at a time."""
    waited = False
    while True:
        moment = clock()
        with _CACHE_LOCK:
            hit = _CACHE.get(key)
            if hit is not None and hit[0] > moment:
                return hit[1]
            building = _BUILDING.get(key)
            if building is None or waited:
                mine = threading.Event()
                _BUILDING[key] = mine
                break
        # Another request is building this key: take its answer when it lands.
        waited = not building.wait(BUILD_WAIT_SECONDS)
    try:
        built = build()
        # Stamped when the build finished (pass-5 review): a build slower than
        # the window must not hand its waiters an answer that is already stale.
        with _CACHE_LOCK:
            _CACHE[key] = (clock() + cache_seconds(step), built)
        return built
    finally:
        with _CACHE_LOCK:
            if _BUILDING.get(key) is mine:
                del _BUILDING[key]
        mine.set()


def clear_cache() -> None:
    with _CACHE_LOCK:
        _CACHE.clear()


def now_inputs(callbacks: Any) -> Dict[str, Any]:
    """The now layer's reads, gathered once per window and shared by both routes."""
    return cached(("now-inputs", ""), SERVING_POLL_SECONDS, lambda: gather_now(callbacks))


def register_performance_dashboard_routes(context: ApiContext) -> None:
    blueprint = context.blueprint
    callbacks = context.callbacks
    require_auth = context.require_auth

    @blueprint.get("/cluster/performance/dashboard/now")
    @require_auth("viewer")
    def performance_dashboard_now():
        return _payload(cached(("now", ""), SERVING_POLL_SECONDS,
                               lambda: build_now(now_inputs(callbacks))))

    @blueprint.get("/cluster/performance/dashboard")
    @require_auth("viewer")
    def performance_dashboard_range():
        chosen = (request.args.get("range", DEFAULT_RANGE) or DEFAULT_RANGE).strip().lower()
        ranges = offered_ranges()
        if chosen not in ranges:
            return _payload(error={"code": "range_not_offered", "message": range_refused()}, status=400)
        _seconds, step = ranges[chosen]
        return _payload(cached(("range", chosen), step,
                               lambda: build_range(gather_range(callbacks, chosen,
                                                                now_inputs=now_inputs(callbacks)))))
