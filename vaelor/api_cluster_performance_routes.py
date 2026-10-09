"""`/api/v2/cluster/performance` — the §6b "why is this slow?" snapshot and the
on-demand serving profile (VD-128).

``GET /cluster/performance`` is pure plumbing: it reads the caller's ``?window``,
hands the gathering to ``performance_snapshot_source.build_performance_snapshot``
— the ONE derivation the Assistant/MCP ``system.performance`` tool also reads, so
the tab and the agent can never quote different numbers — and stamps the window
the caller asked for onto the result. The RED/USE/health assembly and the
one-line diagnosis live one layer down in ``performance_snapshot.build_snapshot``;
the input gathering lives in the shared source module. Neither invents telemetry.

Two windows are read from every door's request record inside the builder (the
inference gateway, the LLM Server and AI Chat): the *recent* one the caller
asks for (default 15 minutes) and the *baseline* window of equal length
immediately before it. Diffing the two is what lets the snapshot name
what changed rather than only stating a level.

``POST /cluster/performance/profile`` triggers ONE bounded on-demand profile of
the GPU serving process through the root hardware bridge (the executor sandbox is
non-root and cannot profile). Profiling perturbs the very box it measures, so the
route is **administrator + CSRF** and **rate-limited**: at most one profile in
flight, and one per short interval, process-wide — a global gate, not per user,
because the cost is the box's, not the caller's. It is audited. The bridge runs
the captures the box supports and returns an honest reason for every capture that
cannot run; this route never fabricates a profile.
"""

from __future__ import annotations

import threading
import time

from flask import g, request

from .api_common import ApiContext, payload as _payload
from .assistant_machine_tools import _parse_window
from .hardware_bridge_client import HardwareBridgeError
from .performance_snapshot_source import (
    DEFAULT_WINDOW,
    DEFAULT_WINDOW_SECONDS,
    build_performance_snapshot,
)

#: The shortest gap between accepted profiles, process-wide. A profile runs
#: ``perf`` + ``amd-smi`` against the serving process for a few seconds and steals
#: real cycles from it; spacing them stops the action from being used to peg the
#: box. Global (not per-user) because the load lands on one machine regardless of
#: who asked.
PROFILE_MIN_INTERVAL_SECONDS = 20

_PROFILE_LOCK = threading.Lock()
#: ``in_flight`` refuses a concurrent profile; ``last_finished`` enforces the
#: interval. In-memory and process-wide, matching the ingest-cap style.
_PROFILE_STATE = {"in_flight": False, "last_finished": 0.0}


def register_cluster_performance_routes(context: ApiContext) -> None:
    blueprint = context.blueprint
    callbacks = context.callbacks
    security = context.security
    require_auth = context.require_auth

    @blueprint.get("/cluster/performance")
    @require_auth("viewer")
    def cluster_performance():
        requested = request.args.get("window", DEFAULT_WINDOW) or DEFAULT_WINDOW
        seconds = _parse_window(requested) or DEFAULT_WINDOW_SECONDS
        snapshot = build_performance_snapshot(callbacks, window_seconds=seconds)
        snapshot["requested_window"] = requested
        return _payload(snapshot)

    @blueprint.post("/cluster/performance/profile")
    @require_auth("administrator", csrf=True)
    def cluster_performance_profile():
        client = callbacks.get("hardware_bridge_client")
        if client is None or not getattr(client, "available", False):
            # No fabricated profile: the privileged bridge that runs perf/amd-smi
            # is the only account that can, so its absence is an honest 503.
            return _payload(
                error={
                    "code": "profiler_unavailable",
                    "message": (
                        "The privileged hardware bridge that runs the profiler "
                        "is not available on this host."
                    ),
                },
                status=503,
            )
        now = time.time()
        with _PROFILE_LOCK:
            if _PROFILE_STATE["in_flight"]:
                return _rate_limited("A profile is already running; wait for it to finish.")
            if now - _PROFILE_STATE["last_finished"] < PROFILE_MIN_INTERVAL_SECONDS:
                return _rate_limited(
                    "A profile ran moments ago; profiling is throttled so it "
                    "cannot peg the box."
                )
            _PROFILE_STATE["in_flight"] = True
        seconds = _requested_seconds()
        try:
            result = client.run_serving_profile(seconds)
        except HardwareBridgeError as error:
            security.audit(
                g.auth_session.username, "cluster.performance.profile", "failure",
                target="gpu-serving", remote_addr=request.remote_addr or "",
            )
            # The bridge's own message, not a 500: a wedged or too-old bridge is a
            # fault in the reader, reported as a bad-gateway with its reason.
            return _payload(
                error={"code": "profiler_failed", "message": str(error)}, status=502
            )
        finally:
            with _PROFILE_LOCK:
                _PROFILE_STATE["in_flight"] = False
                _PROFILE_STATE["last_finished"] = time.time()
        security.audit(
            g.auth_session.username, "cluster.performance.profile", "success",
            target="gpu-serving", remote_addr=request.remote_addr or "",
        )
        return _payload(result)


def _rate_limited(message: str):
    """A 429 that names why the profile was refused, in plain language."""
    return _payload(
        error={"code": "rate_limited", "message": message}, status=429
    )


def _requested_seconds():
    """The caller's requested capture window, or ``None`` for the default.

    The bridge clamps it to the bounded range regardless, so a hostile value is
    harmless; a missing or non-object body just uses the default.
    """
    body = request.get_json(silent=True)
    if isinstance(body, dict) and body.get("seconds") is not None:
        try:
            return int(body["seconds"])
        except (TypeError, ValueError):
            return None
    return None
