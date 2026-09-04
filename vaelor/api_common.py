"""Shared API composition, authentication, and response contracts."""

from __future__ import annotations

import ipaddress
import re
import socket
from functools import wraps
from typing import Any, Dict, Optional

from flask import Blueprint, g, jsonify, request

from .model_reachability import probe_connection
from .security import LoginLimiter, SecurityStore
from .runtime_paths import env_value


SESSION_COOKIE = "vaelor_session"
LEGACY_SESSION_COOKIE = "pm_session"

#: The one wording for "the credential broker did not answer", shared by every
#: route that resolves a managed secret (credentials CRUD, alert channels,
#: backups). One home so the message cannot drift between routes and is not a
#: cross-module duplicate the governance walk has to record.
CREDENTIAL_BROKER_UNAVAILABLE = "Secure credential storage is unavailable."

#: Interface name prefixes whose addresses are private and routable only on
#: this host. Matched as a prefix rather than exactly, because Docker names
#: user-defined bridges `br-<network id>` and there are seven of them on the
#: appliance this was found on. See `ApiContext.appliance_address`.
_VIRTUAL_INTERFACE_PREFIXES = (
    "docker", "br-", "veth", "virbr", "vmnet", "tun", "tap", "cni", "flannel",
)


def payload(
    data: Any = None,
    *,
    error: Optional[Dict[str, Any]] = None,
    status: int = 200,
):
    body: Dict[str, Any] = {"ok": error is None}
    body["data" if error is None else "error"] = data if error is None else error
    return jsonify(body), status


def assistant_model_status(callbacks: Dict[str, Any], actor: str) -> Dict[str, Any]:
    """Resolve whether the appliance has a usable model-backed intelligence.

    Readiness reflects the model the *appliance* has connected, not a per-user
    preference. `intelligence_choice` is stored per (actor, key), so it is only
    set for whoever ran setup or the deploy; on the Z2 that was actor `system`,
    leaving the admin's choice `""` and every other user with no choice at all.
    Gating on `local`/`provider` therefore marked a working, connected model
    "unavailable" for everyone who had not personally picked it.

    An unset choice uses whatever model the appliance has connected -- the
    "installed model just works" rule (VD-109's principle). This mirrors
    `resolve_model_connection`, which already returns the active deployment
    lease for every mode except the literal `"basic"`. Only an explicit
    `"basic"` is a deliberate no-model opt-out. A fresh appliance with no model
    still yields `configured=False` below, so it stays not-ready.
    """
    agent = callbacks.get("deployment_agent")
    memory = callbacks.get("assistant_memory")
    if agent is None or memory is None:
        return {"ready": False, "mode": "", "model": "", "provider": ""}
    mode = memory.get_preference(actor, "intelligence_choice", "")
    if mode == "basic":
        return {"ready": False, "mode": mode, "model": "", "provider": ""}
    try:
        status = agent.status(mode=mode)
    except (OSError, TypeError, ValueError):
        status = {}
    configured = bool(status.get("configured"))
    # `configured` only says a model was chosen. Whether it answers is a
    # separate question, and reporting the first as the second is what let the
    # UI show verified health while every request through it failed.
    probe = {"reachable": False, "detail": "", "endpoint": ""}
    if configured:
        try:
            probe = probe_connection(agent._connection(mode))
        except (AttributeError, OSError, TypeError, ValueError):
            probe = {"reachable": True, "detail": "", "endpoint": ""}
    return {
        "ready": configured,
        "reachable": bool(probe.get("reachable")),
        "unreachable_reason": "" if probe.get("reachable") else str(probe.get("detail", "")),
        # Reachable and offering nothing is its own state. Reporting it as
        # unreachable is what put "MODEL UNREACHABLE" on the Assistant at the
        # same moment AI Chat correctly said the server was reachable but
        # offering no model - one endpoint, two verdicts, and the other one was
        # right. `None` here means the question was never reached.
        "offering_models": probe.get("offering_models"),
        "model_availability_reason": str(
            probe.get("model_availability_reason", "")
        ),
        "endpoint": str(probe.get("endpoint", "")),
        "mode": mode,
        "model": status.get("model") or "",
        "provider": status.get("provider") or "",
    }


class ApiContext:
    """Dependencies and cross-cutting policies shared by route modules."""

    def __init__(
        self,
        callbacks: Dict[str, Any],
        store: Optional[SecurityStore] = None,
    ):
        self.callbacks = callbacks
        self.blueprint = Blueprint("api_v2", __name__, url_prefix="/api/v2")
        self.security = store or SecurityStore(session_hours=int(env_value(
            "VAELOR_SESSION_HOURS", "PM_SESSION_HOURS", "12"
        )))
        self.limiter = LoginLimiter()

    def appliance_address(self) -> str:
        configured = env_value(
            "VAELOR_APPLIANCE_HOST", "PM_APPLIANCE_HOST", ""
        )
        if configured and re.fullmatch(r"[A-Za-z0-9.\-]{1,253}", configured):
            return configured
        inventory = self.callbacks.get("system_inventory")
        try:
            interfaces = inventory.network().get("interfaces", [])
        except (AttributeError, OSError, TypeError, ValueError):
            interfaces = []
        # **This address is handed to a human to type into a client**, so it
        # has to be one reachable from where that human is sitting. Taking the
        # first private IPv4 on the first `up` interface does not guarantee
        # that: container and VM bridges are `up`, are private, and are
        # enumerated alongside the real one. On 2026-08-11 the remote console
        # offered `172.20.0.1:3389` - a Docker bridge - behind a green "remote
        # access available" badge, while eth0 held 192.168.0.50. This Pi has
        # seven such bridges.
        #
        # Worse than being wrong, it was *unstably* wrong: bridges come and go
        # as containers start, so enumeration order shifts and the same code
        # answers differently on different days. Ranking makes the answer
        # deterministic. A bridge is still returned if it is genuinely all
        # this machine has, because a private address somebody might reach
        # beats falling through to a `.local` name that needs mDNS.
        candidates = []
        for interface in interfaces:
            if interface.get("state") != "up":
                continue
            name = str(interface.get("name", ""))
            virtual = name.startswith(_VIRTUAL_INTERFACE_PREFIXES)
            for address in interface.get("addresses", []):
                value = str(address.get("address", ""))
                try:
                    parsed = ipaddress.ip_address(value)
                except ValueError:
                    continue
                if parsed.version != 4 or not parsed.is_private:
                    continue
                # Link-local is an autoconfiguration fallback, not an address
                # a client should be told to use; loopback reaches only here.
                if parsed.is_loopback or parsed.is_link_local:
                    continue
                candidates.append((1 if virtual else 0, len(candidates), value))
        if candidates:
            return min(candidates)[2]
        hostname = re.sub(r"[^A-Za-z0-9.\-]", "", socket.gethostname())[:240]
        return f"{hostname}.local" if hostname else "vaelor.local"

    def session_from_request(self):
        token = (
            request.cookies.get(SESSION_COOKIE, "")
            or request.cookies.get(LEGACY_SESSION_COOKIE, "")
        )
        return token, self.security.get_session(token)

    def require_auth(self, *roles: str, csrf: bool = False):
        def decorator(function):
            @wraps(function)
            def wrapped(*args, **kwargs):
                token, session = self.session_from_request()
                if session is None:
                    return payload(
                        error={
                            "code": "authentication_required",
                            "message": "Sign in to continue.",
                        },
                        status=401,
                    )
                if roles and not self.security.role_allows(session.role, roles):
                    return payload(
                        error={
                            "code": "insufficient_permission",
                            "message": "Your role cannot perform this action.",
                        },
                        status=403,
                    )
                if csrf and not self.security.csrf_matches(
                    session,
                    request.headers.get("X-CSRF-Token", ""),
                    token,
                ):
                    return payload(
                        error={
                            "code": "invalid_csrf_token",
                            "message": "Refresh the page and try again.",
                        },
                        status=403,
                    )
                g.auth_token = token
                g.auth_session = session
                return function(*args, **kwargs)

            return wrapped

        return decorator
