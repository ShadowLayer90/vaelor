"""Restart the control plane, reboot, or shut down: the power routes.

Moved out of ``api_workload_routes`` (which sat at 971 lines) when W4d-D29
changed how a control-plane restart is carried out.

**W4d-D29: "Restart service" left no audit entry and restarted twice.** The
route ran the restart inline and audited afterwards, but the process the
restart stops is the one that would have written the audit row, and it was
stopped before the response left. The browser then saw its POST reset within a
second, which its transport reads as "never reached the server" and re-sends
once (``frontend/src/lib/api.ts``), so the control plane restarted a second
time. Both are fixed here, at the side that causes them: the restart is audited
as ``accepted`` before anything stops, the response is sent, and only then -
after :data:`RESTART_DEFERRAL_SECONDS` - is the restart started. A restart that
fails (the bridge refused it) is audited ``failure`` by the still-running
process. Reboot and shut down keep their inline path: the host takes longer
than the transport's retry window to go down, and their failures must still be
returned to the operator (#208).
"""

from __future__ import annotations

import logging
import threading
import time
from typing import Any, Callable

from flask import g, request

from .api_common import ApiContext, payload as _payload
from .host_power import POWER_ACTION_REFUSAL

_LOGGER = logging.getLogger(__name__)

#: How long the control plane waits, after answering, before it restarts
#: itself. Long enough for the 202 to reach the browser so nothing re-sends the
#: request; short enough that the owner still sees the restart follow at once.
RESTART_DEFERRAL_SECONDS = 1.5

#: How long a control-plane restart takes at most once it starts: systemd's
#: default stop timeout (90 s) is the longest the old process may linger, and
#: a start is seconds on top of that.
RESTART_TAKES_AT_MOST_SECONDS = 120

#: How long an accepted restart counts as pending (R2-8). It is the timer's
#: delay plus the longest a restart takes, so a second request while one is
#: really under way is answered "already pending" - and past it, a restart
#: that never happened (systemctl failing after ``_spawn``'s grace window had
#: already reported success) no longer holds every later request off for ever.
RESTART_PENDING_SECONDS = RESTART_DEFERRAL_SECONDS + RESTART_TAKES_AT_MOST_SECONDS

#: The clock the pending window is read on; a test moves it.
_monotonic = time.monotonic


def _defer(seconds: float, action: Callable[[], Any]) -> None:
    """Run ``action`` once, ``seconds`` from now, off the request thread."""
    timer = threading.Timer(seconds, action)
    timer.daemon = True
    timer.start()


def register_power_routes(context: ApiContext) -> None:
    blueprint = context.blueprint
    callbacks = context.callbacks
    security = context.security
    require_auth = context.require_auth
    # F10: one pending control-plane restart at a time. Two accepted POSTs
    # inside the deferral used to start two timers - two restarts, the defect
    # W4d-D29 fixed for the browser's re-send, back for a double click or a
    # second tab. R2-8: the latch is a deadline, not a lock. A lock released
    # only when power_action raised stuck for good when systemctl failed after
    # the grace window, and every later Restart answered "already pending".
    # Cleared at once when the restart fails, because then the process is
    # still up and the owner must be able to retry.
    restart_lock = threading.Lock()
    restart_pending = {"until": 0.0}

    def claim_restart() -> bool:
        with restart_lock:
            now = _monotonic()
            if now < restart_pending["until"]:
                return False
            restart_pending["until"] = now + RESTART_PENDING_SECONDS
            return True

    def release_restart() -> None:
        with restart_lock:
            restart_pending["until"] = 0.0

    @blueprint.get("/power/capabilities")
    @require_auth("viewer")
    def power_capabilities():
        provider = callbacks.get("power_capabilities")
        if provider is None:
            actions = {
                name: {"available": True, "reason": ""}
                for name in ("restart_service", "reboot", "shutdown")
            }
            return _payload({"id": "legacy-callback", "actions": actions})
        return _payload(provider())

    @blueprint.post("/power/actions")
    @require_auth("operator", csrf=True)
    def power_action():
        body = request.get_json(silent=True) or {}
        action = str(body.get("action", "")).strip().lower()
        confirmations = {
            "restart_service": "restart-service",
            "reboot": "reboot-device",
            "shutdown": "shutdown-device",
        }
        if action not in confirmations:
            return _payload(
                error={
                    "code": "invalid_power_action",
                    "message": POWER_ACTION_REFUSAL,
                },
                status=400,
            )
        if body.get("confirmation") != confirmations[action]:
            return _payload(
                error={
                    "code": "power_confirmation_required",
                    "message": "Review and confirm this power action before continuing.",
                },
                status=400,
            )
        provider = callbacks.get("power_capabilities")
        capabilities = provider() if provider is not None else {
            "actions": {
                name: {"available": True, "reason": ""}
                for name in confirmations
            }
        }
        action_capability = capabilities.get("actions", {}).get(action, {})
        if not action_capability.get("available"):
            return _payload(
                error={
                    "code": "power_action_unavailable",
                    "message": (
                        action_capability.get("reason")
                        or "This power action is unavailable on the current platform."
                    ),
                },
                status=409,
            )
        actor = g.auth_session.username
        remote_addr = request.remote_addr or ""

        def audit(outcome: str, **details: Any) -> None:
            security.audit(
                actor, "power.{}".format(action), outcome,
                target="local-host", remote_addr=remote_addr, details=details,
            )

        if action == "restart_service":
            if not claim_restart():
                return _payload(
                    {
                        "accepted": True,
                        "action": action,
                        "already_pending": True,
                        "message": "A restart of the Vaelor service is already "
                        "pending; it was not scheduled a second time.",
                    },
                    status=202,
                )
            # Audited before the process that writes the audit is stopped, and
            # started after the answer is on its way (W4d-D29).
            audit("accepted", deferred_seconds=RESTART_DEFERRAL_SECONDS)

            def restart() -> None:
                try:
                    callbacks["power_action"](action)
                except Exception as error:  # noqa: BLE001 - recorded, the process is still up
                    _LOGGER.warning("The control-plane restart failed: %s", error)
                    audit("failure", reason=str(error)[:300])
                    release_restart()

            (callbacks.get("power_defer") or _defer)(RESTART_DEFERRAL_SECONDS, restart)
            return _payload(
                {
                    "accepted": True,
                    "action": action,
                    "starts_in_seconds": RESTART_DEFERRAL_SECONDS,
                },
                status=202,
            )
        # #208 / LESSONS #191: audit what actually happened. The power path now
        # raises when a command fails immediately (e.g. sudo blocked by
        # NoNewPrivileges), so a failure is recorded as `failure` and its reason
        # returned to the operator — never audited as success while the box
        # stays up.
        try:
            callbacks["power_action"](action)
        except Exception as error:
            audit("failure")
            return _payload(
                error={
                    "code": "power_action_failed",
                    "message": str(error) or "The power action failed before it could take effect.",
                },
                status=502,
            )
        audit("success")
        return _payload({"accepted": True, "action": action}, status=202)
