"""Managed workloads, remote desktops, jobs, and power routes."""

from __future__ import annotations

import json
import re
import socket
import time
from typing import Any, Dict
from urllib.parse import quote, urlsplit

from flask import Response, g, request, stream_with_context

from .job_vocabulary import (
    RECOVERY_ROUTE_REQUIRED, REPLAN_REQUIRED_JOB_TYPES, administrator_only,
)
from .app_catalog import public_catalog
from .api_common import ApiContext, payload as _payload
from .api_power_routes import register_power_routes
from .api_workload_app_config_routes import register_workload_app_config_routes
from .agent_deploy_keys import (
    AGENT_DEPLOY_RETRY_REFUSED, FIRST_KEY_MINT_FAILED, agent_deploy_retry_refused,
    mint_first_agent_key, revoke_cancelled_deploy_key, revoke_unqueued_key, stamp_agent_deploy,
)
from .application_job_admission import application_job_refusal
from .cluster_job_confirmations import CLUSTER_JOB_CONFIRMATIONS, confirmation_accepted
from .gpu_memory_pool_nodes import (
    CONFIRMATION_REQUIRED, HOST_GPU_MEMORY_JOB, controller_job_confirmed,
)
from .app_port_claims import model_port_holders
from .credential_broker_client import CredentialError
from .workload_dependencies import DependencyError


def _browser_desktop_host(fallback: str) -> str:
    """Use the address that successfully reached this appliance in the browser."""
    hostname = urlsplit(request.host_url).hostname or fallback
    return f"[{hostname}]" if ":" in hostname else hostname


def _bind_checkpoint_restore(inventory: Any, payload: Any) -> Dict[str, Any]:
    """Return a server-owned restore payload after verifying the archive.

    The recovery UI intentionally submits only the checkpoint identity,
    project, and typed confirmation.  When a caller supplies a SHA-256, it is
    treated as an assertion and must match the archive currently on disk.
    Otherwise the current inventory record supplies the digest before the
    binding is performed.  ``bind_restore`` performs a second on-disk
    verification and supplies the manifest digest that is persisted with the
    job.
    """

    if not isinstance(payload, dict):
        raise ValueError("Checkpoint restore payload must be an object.")
    checkpoint = str(payload.get("checkpoint", "")).strip()
    project = str(payload.get("project", "")).strip()
    confirmation = str(payload.get("confirm", ""))
    if confirmation != project:
        raise ValueError("Type the project name to confirm restoration.")

    digest = str(payload.get("sha256", "")).strip().lower()
    if not digest:
        verifier = getattr(inventory, "verify", None) or getattr(inventory, "checksum", None)
        if verifier is None:
            raise AttributeError("Checkpoint verification is unavailable.")
        current = verifier(checkpoint)
        digest = str(current.get("sha256", "")).strip().lower()
    binding = inventory.bind_restore(checkpoint, project, digest)
    if not isinstance(binding, dict):
        raise ValueError("Checkpoint binding is incomplete.")
    if (
        binding.get("checkpoint") != checkpoint
        or binding.get("project") != project
        or str(binding.get("sha256", "")).lower() != digest
        or not re.fullmatch(r"[0-9a-f]{64}", str(binding.get("manifest_digest", "")).lower())
        or binding.get("verified") is not True
        or binding.get("restorable") is not True
    ):
        raise ValueError("Checkpoint binding is incomplete.")
    return {**binding, "confirm": confirmation}


def _catalog_with_ports(broker: Any) -> list:
    """The catalog, each card carrying the port an install would get (W7-D2)."""
    from . import app_port_claims

    try:
        holders = app_port_claims.model_port_holders(broker)
    except Exception:  # noqa: BLE001 - the install re-checks and refuses itself
        holders = {}
    return app_port_claims.catalog_port_offers(public_catalog(), holders)


def _port_preflight(inventory: Any, port: int, broker: Any = None) -> Dict[str, Any]:
    """Check a requested host port against inventory and the live socket table."""
    if isinstance(port, bool) or not isinstance(port, int):
        raise ValueError("Choose a whole-number application port.")
    if not 1024 <= port <= 65535 or port in {34001, 34002}:
        raise ValueError("Choose an available port from 1024 to 65535.")

    owners: list[str] = []
    occupied_ports: set[int] = set()
    snapshot = inventory.list_all() if inventory is not None else {}
    for app in snapshot.get("apps", []) if isinstance(snapshot, dict) else []:
        if not isinstance(app, dict):
            continue
        identity = str(app.get("display_identity") or app.get("name") or "managed app")
        # Docker retains published bindings for stopped containers, while some
        # inventory adapters expose the normalized ``published_ports`` shape.
        # Treat both as reservations: a stopped managed workload still owns
        # its configured host port and a suggestion must not collide with it.
        bindings = [
            *(app.get("ports", []) if isinstance(app.get("ports"), list) else []),
            *(app.get("published_ports", []) if isinstance(app.get("published_ports"), list) else []),
            *(app.get("reserved_ports", []) if isinstance(app.get("reserved_ports"), list) else []),
        ]
        for binding in bindings:
            raw_host = binding.get("host") if isinstance(binding, dict) else binding
            if isinstance(binding, dict):
                raw_host = binding.get(
                    "host",
                    binding.get("host_port", binding.get("published", binding.get("port"))),
                )
            try:
                reserved = int(raw_host)
            except (TypeError, ValueError):
                continue
            if 1024 <= reserved <= 65535:
                occupied_ports.add(reserved)
                if reserved == port:
                    owners.append(identity)

    # W6-D2: the same claims the install refuses on - a stored model's port.
    try:
        model_holders = model_port_holders(broker)
    except Exception:  # noqa: BLE001 - the install re-checks and refuses itself
        model_holders = {}
    occupied_ports.update(model_holders)
    if port in model_holders:
        owners.append(model_holders[port])

    socket_busy = False
    probe = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    try:
        probe.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        probe.bind(("127.0.0.1", port))
    except OSError:
        socket_busy = True
    finally:
        probe.close()

    conflict = bool(owners or socket_busy or port in occupied_ports)
    suggestion = None
    if conflict:
        for candidate in range(port + 1, 65536):
            if candidate in {34001, 34002} or candidate in occupied_ports:
                continue
            candidate_probe = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            try:
                candidate_probe.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
                candidate_probe.bind(("127.0.0.1", candidate))
            except OSError:
                continue
            finally:
                candidate_probe.close()
            suggestion = candidate
            break

    reason = ""
    if conflict:
        owner_text = ", ".join(sorted(set(owners))) if owners else "another service on this node"
        reason = f"Port {port} is already used by {owner_text}. Choose another port before approval."
    return {
        "requested_port": port,
        "available": not conflict,
        "conflict": conflict,
        "suggested_port": suggestion,
        "reason": reason,
    }

def register_workload_routes(context: ApiContext) -> None:
    blueprint = context.blueprint
    callbacks = context.callbacks
    security = context.security
    limiter = context.limiter
    require_auth = context.require_auth
    appliance_address = context.appliance_address

    # Administrator-only secret reveal and config-file editing, kept in a
    # focused module so this one stays under the line ceiling.
    register_workload_app_config_routes(context)
    # Restart, reboot and shut down, in their own module (W4d-D29).
    register_power_routes(context)

    @blueprint.get("/managed")
    @require_auth("viewer")
    def managed_inventory():
        inventory = callbacks.get("workload_inventory")
        if inventory is None:
            return _payload({"apps": [], "models": [], "capabilities": {}})
        return _payload(inventory.list_all())

    @blueprint.get("/workloads/port-preflight")
    @require_auth("operator")
    def workload_port_preflight():
        inventory = callbacks.get("workload_inventory")
        try:
            port = int(request.args.get("port", ""))
            return _payload(_port_preflight(
                inventory, port, callbacks.get("credential_broker"),
            ))
        except (TypeError, ValueError) as error:
            return _payload(error={"code": "port_preflight_invalid", "message": str(error)}, status=400)

    @blueprint.get("/managed/removal-plan")
    @require_auth("administrator")
    def managed_removal_plan():
        dependencies = callbacks.get("workload_dependencies")
        if dependencies is None:
            return _payload(
                error={
                    "code": "dependency_service_unavailable",
                    "message": "Dependency-aware removal is unavailable.",
                },
                status=503,
            )
        try:
            report = dependencies.report(
                request.args.get("kind", ""),
                request.args.get("id", ""),
                g.auth_session.username,
            )
        except DependencyError as error:
            return _payload(
                error={"code": "removal_plan_unavailable", "message": str(error)},
                status=400,
            )
        return _payload(report)

    @blueprint.get("/apps/catalog")
    @require_auth("viewer")
    def app_catalog():
        return _payload(_catalog_with_ports(callbacks.get("credential_broker")))

    @blueprint.get("/managed/apps/<app_id>/logs")
    @require_auth("operator")
    def managed_app_logs(app_id):
        inventory = callbacks.get("workload_inventory")
        try:
            return _payload(inventory.logs(app_id, request.args.get("tail", "200")))
        except (AttributeError, ValueError) as error:
            return _payload(error={"code": "app_logs_unavailable", "message": str(error)}, status=400)

    @blueprint.post("/managed/apps/<app_id>/diagnostics")
    @require_auth("operator", csrf=True)
    def managed_app_diagnostics(app_id):
        inventory = callbacks.get("workload_inventory")
        tool = str((request.get_json(silent=True) or {}).get("tool", "stats"))
        try:
            result = inventory.diagnostics(app_id, tool)
        except (AttributeError, ValueError) as error:
            return _payload(error={"code": "diagnostic_failed", "message": str(error)}, status=400)
        security.audit(
            g.auth_session.username, "app.diagnostic", "success", target=app_id,
            remote_addr=request.remote_addr or "", details={"tool": tool},
        )
        return _payload(result)

    @blueprint.post("/managed/apps/<app_id>/remote-desktop")
    @require_auth("operator", csrf=True)
    def managed_app_remote_desktop(app_id):
        inventory = callbacks.get("workload_inventory")
        sessions = callbacks.get("vnc_sessions")
        if inventory is None or sessions is None:
            return _payload(
                error={
                    "code": "remote_desktop_unavailable",
                    "message": "Remote Desktop is not commissioned.",
                },
                status=503,
            )
        try:
            target_port = inventory.vnc_target(app_id)
            session = sessions.create(g.auth_session.username, app_id, target_port)
        except (OSError, ValueError) as error:
            return _payload(
                error={"code": "remote_desktop_unavailable", "message": str(error)},
                status=400,
            )
        hostname = _browser_desktop_host(appliance_address())
        scheme = "https" if request.is_secure else "http"
        websocket_path = quote("websockify?token=" + session["token"], safe="")
        session_url = (
            f"{scheme}://{hostname}:34002/vnc.html"
            "?autoconnect=1&resize=scale&reconnect=0"
            f"&path={websocket_path}"
        )
        security.audit(
            g.auth_session.username,
            "app.remote_desktop.session",
            "success",
            target=app_id,
            remote_addr=request.remote_addr or "",
            details={"target_port": target_port, "expires_at": session["expires_at"]},
        )
        return _payload({"url": session_url, "session_id": session["session_id"], "expires_at": session["expires_at"]})

    @blueprint.get("/host/remote-desktop")
    @require_auth("viewer")
    def host_remote_desktop_status():
        probe = callbacks.get("host_remote_desktop")
        hostname = appliance_address()
        if probe is None:
            return _payload({
                "available": False,
                "port": 3389,
                "name": "Host Remote Login",
                "kind": "host-rdp",
                "detail": "Host remote desktop discovery is unavailable.",
                "address": hostname,
                "rdp": {"available": False, "port": 3389},
                "browser_vnc": {"available": False, "port": 5901},
            })
        status = probe.status()
        status["address"] = hostname
        return _payload(status)

    @blueprint.post("/host/remote-desktop/rdp")
    @require_auth("administrator", csrf=True)
    def host_remote_desktop_rdp_enable():
        broker = callbacks.get("host_desktop_broker")
        probe = callbacks.get("host_remote_desktop")
        if broker is None:
            return _payload(
                error={
                    "code": "host_remote_desktop_unavailable",
                    "message": "Host Remote Login management is unavailable.",
                },
                status=503,
            )
        status = probe.status() if probe is not None else {}
        if not (status.get("desktop") or {}).get("available", True):
            return _payload(
                error={
                    "code": "host_desktop_unavailable",
                    "message": (
                        "The graphical desktop is not healthy. Vaelor has kept "
                        "console access available instead of enabling an unusable RDP service."
                    ),
                },
                status=409,
            )
        body = request.get_json(silent=True) or {}
        try:
            result = broker.configure_rdp(
                str(body.get("username", "")),
                str(body.get("password", "")),
            )
        except (OSError, RuntimeError, ValueError) as error:
            security.audit(
                g.auth_session.username, "host.rdp.enable", "failure",
                target="host-remote-login", remote_addr=request.remote_addr or "",
            )
            return _payload(
                error={"code": "host_rdp_setup_failed", "message": str(error)},
                status=400,
            )
        security.audit(
            g.auth_session.username, "host.rdp.enable", "success",
            target="host-remote-login", remote_addr=request.remote_addr or "",
        )
        return _payload(result)

    @blueprint.delete("/host/remote-desktop/rdp")
    @require_auth("administrator", csrf=True)
    def host_remote_desktop_rdp_disable():
        broker = callbacks.get("host_desktop_broker")
        if broker is None:
            return _payload(
                error={
                    "code": "host_remote_desktop_unavailable",
                    "message": "Host Remote Login management is unavailable.",
                },
                status=503,
            )
        try:
            result = broker.disable_rdp()
        except (OSError, RuntimeError, ValueError) as error:
            return _payload(
                error={"code": "host_rdp_disable_failed", "message": str(error)},
                status=400,
            )
        security.audit(
            g.auth_session.username, "host.rdp.disable", "success",
            target="host-remote-login", remote_addr=request.remote_addr or "",
        )
        return _payload(result)

    @blueprint.get("/host/remote-desktop/profile")
    @require_auth("viewer")
    def host_remote_desktop_profile():
        probe = callbacks.get("host_remote_desktop")
        if probe is None or not probe.status().get("available"):
            return _payload(
                error={
                    "code": "host_rdp_unavailable",
                    "message": "Enable Host Remote Login before downloading a connection profile.",
                },
                status=409,
            )
        hostname = appliance_address()
        mode = str(request.args.get("mode", "responsive")).strip().lower()
        if mode not in {"responsive", "quality"}:
            mode = "responsive"
        profile_lines = [
            f"full address:s:{hostname}:3389",
            "prompt for credentials:i:1",
            # **Left at 2 — "warn me" — deliberately.** Level 0 silences the
            # dialog by not checking, trading the owner's only defence for a
            # quieter screen; `host_desktop_tls` fixes the cause instead.
            "authentication level:i:2",
            # Windows keys its accepted-certificate cache to the name it was
            # given, so a host name, its FQDN and its IP were three separate
            # trust decisions. Pinned to the `full address` above.
            "use redirection server name:i:1",
            "enablecredsspsupport:i:1",
            "redirectclipboard:i:1",
            "smart sizing:i:1",
            "networkautodetect:i:1",
            "bandwidthautodetect:i:1",
        ]
        if mode == "responsive":
            profile_lines.extend([
                "desktopwidth:i:1600",
                "desktopheight:i:900",
                "session bpp:i:24",
                "dynamic resolution:i:0",
                "audiomode:i:2",
                "disable wallpaper:i:1",
                "allow font smoothing:i:0",
                "allow desktop composition:i:0",
            ])
        else:
            profile_lines.extend([
                "dynamic resolution:i:1",
                "audiomode:i:0",
            ])
        profile = "\r\n".join([*profile_lines, ""])
        return Response(
            profile,
            mimetype="application/x-rdp",
            headers={
                "Content-Disposition": "attachment; filename=vaelor-remote-login.rdp",
                "Cache-Control": "no-store",
            },
        )

    @blueprint.post("/host/remote-desktop/browser-session")
    @blueprint.post("/host/remote-desktop/session")
    @require_auth("operator", csrf=True)
    def host_remote_desktop_session():
        probe = callbacks.get("host_vnc")
        sessions = callbacks.get("vnc_sessions")
        if probe is None or sessions is None:
            return _payload(
                error={
                    "code": "host_remote_desktop_unavailable",
                    "message": "The host browser desktop is not commissioned.",
                },
                status=503,
            )
        try:
            target_port = probe.target_port()
            session = sessions.create(
                g.auth_session.username,
                "host-browser-desktop",
                target_port,
            )
        except (OSError, ValueError) as error:
            return _payload(
                error={
                    "code": "host_remote_desktop_unavailable",
                    "message": str(error),
                },
                status=400,
            )
        hostname = _browser_desktop_host(appliance_address())
        scheme = "https" if request.is_secure else "http"
        websocket_path = quote("websockify?token=" + session["token"], safe="")
        session_url = (
            f"{scheme}://{hostname}:34002/vnc.html"
            "?autoconnect=1&resize=scale&reconnect=0"
            f"&path={websocket_path}"
        )
        security.audit(
            g.auth_session.username,
            "host.remote_desktop.session",
            "success",
            target="host-browser-desktop",
            remote_addr=request.remote_addr or "",
            details={"target_port": target_port, "expires_at": session["expires_at"]},
        )
        return _payload({"url": session_url, "session_id": session["session_id"], "expires_at": session["expires_at"]})

    @blueprint.get("/remote-desktop/browser-sessions/<session_id>")
    @require_auth("operator")
    def remote_desktop_browser_session_status(session_id):
        sessions = callbacks.get("vnc_sessions")
        if sessions is None:
            return _payload(error={"code": "remote_session_unavailable", "message": "Browser desktop sessions are unavailable."}, status=503)
        result = sessions.status(session_id, g.auth_session.username)
        return _payload(result)

    @blueprint.post("/remote-desktop/browser-sessions/end")
    @require_auth("operator", csrf=True)
    def remote_desktop_end_browser_session():
        """End the desktop running on the appliance, not just this view.

        **The owner had no way out of a broken desktop.** "Close session" in
        the console cleared the browser's own state and nothing else, so a
        session that had locked itself — GNOME demanding a password for an
        account that has none — could only be recovered over SSH. The desktop
        is a stated one-use session holding nothing worth keeping, so ending
        it is the recovery action, and the next open starts clean.
        """
        # `host_desktop_broker` is the key `control_plane_runtime` registers.
        # The first version asked for `host_desktop`, got None, and returned a
        # 503 the owner never saw because the notice renders behind the session
        # overlay — so the button read as doing nothing at all.
        broker = callbacks.get("host_desktop_broker")
        if broker is None:
            return _payload(
                error={
                    "code": "host_remote_desktop_unavailable",
                    "message": "The host desktop service is not configured.",
                },
                status=503,
            )
        try:
            result = broker.end_desktop_session()
        except (OSError, RuntimeError, ValueError) as error:
            return _payload(
                error={
                    "code": "host_remote_desktop_unavailable",
                    "message": str(error),
                },
                status=502,
            )
        # Revoking the viewing tokens as well, so a session ended on the
        # appliance cannot still be reached through a URL someone kept. No
        # `hasattr` guard: the method exists, and a guard would turn a rename
        # into silence rather than a failure.
        sessions = callbacks.get("vnc_sessions")
        revoked = sessions.revoke_all(g.auth_session.username) if sessions else 0
        security.audit(
            g.auth_session.username,
            "host.remote_desktop.session_end",
            "success",
            target="host-browser-desktop",
            remote_addr=request.remote_addr or "",
            details={
                "account": str(result.get("account", "")),
                "tokens_revoked": revoked,
                # Whether there was a session to end at all is a separate
                # question from whether the request succeeded.
                "ended": bool(result.get("ended")),
            },
        )
        return _payload(dict(result, tokens_revoked=revoked))

    @blueprint.get("/managed/apps/<app_id>/config")
    @require_auth("operator")
    def managed_app_config(app_id):
        inventory = callbacks.get("workload_inventory")
        try:
            return _payload(inventory.read_config(app_id))
        except (AttributeError, OSError, ValueError) as error:
            return _payload(error={"code": "config_unavailable", "message": str(error)}, status=400)

    @blueprint.put("/managed/apps/<app_id>/config")
    @require_auth("administrator", csrf=True)
    def managed_app_config_save(app_id):
        inventory = callbacks.get("workload_inventory")
        content = (request.get_json(silent=True) or {}).get("content", "")
        try:
            result = inventory.save_config(app_id, content)
        except (AttributeError, OSError, ValueError) as error:
            return _payload(error={"code": "config_invalid", "message": str(error)}, status=400)
        security.audit(
            g.auth_session.username, "app.config.update", "success", target=app_id,
            remote_addr=request.remote_addr or "", details={"backup": result.get("backup")},
        )
        return _payload(result)

    @blueprint.get("/jobs")
    @require_auth("operator")
    def jobs():
        job_store = callbacks.get("job_store")
        if job_store is None:
            return _payload(
                error={
                    "code": "job_service_unavailable",
                    "message": "The privileged job service is not configured.",
                },
                status=503,
            )
        try:
            limit = int(request.args.get("limit", "50"))
        except ValueError:
            limit = 50
        actor = None if g.auth_session.role == "administrator" else g.auth_session.username
        if request.args.get("summary", "").lower() in {"1", "true", "yes"}:
            return _payload(job_store.ledger(limit, actor=actor))
        return _payload(job_store.list(limit, actor=actor))

    def _administrator_required():
        """The one refusal for a job only an administrator may create or retry."""
        return _payload(
            error={
                "code": "administrator_required",
                "message": "Administrator access is required for destructive recovery actions.",
            },
            status=403,
        )

    @blueprint.post("/jobs")
    @require_auth("operator", csrf=True)
    def create_job():
        job_store = callbacks.get("job_store")
        if job_store is None:
            return _payload(
                error={
                    "code": "job_service_unavailable",
                    "message": "The privileged job service is not configured.",
                },
                status=503,
            )
        body = request.get_json(silent=True) or {}
        job_type = str(body.get("type", "")).strip().lower()
        job_payload = body.get("payload", {})
        if job_type in {"compose.remove", "model.remove"}:
            return _payload(
                error={
                    "code": "dependency_plan_required",
                    "message": (
                        "Load a dependency-aware removal plan and submit managed.remove."
                    ),
                },
                status=409,
            )
        if job_type == "managed.remove":
            dependencies = callbacks.get("workload_dependencies")
            if dependencies is None:
                return _payload(
                    error={
                        "code": "dependency_service_unavailable",
                        "message": "Dependency-aware removal is unavailable.",
                    },
                    status=503,
                )
            try:
                dependencies.validate_removal(
                    job_payload, g.auth_session.username
                )
            except DependencyError as error:
                return _payload(
                    error={"code": "invalid_removal_plan", "message": str(error)},
                    status=409,
                )
        _refusal = application_job_refusal(job_type, job_payload, callbacks)
        if _refusal is not None:
            return _payload(error={"code": _refusal[0], "message": _refusal[1]}, status=_refusal[2])
        if (
            job_type == "system.update.apply"
            and (
                not isinstance(job_payload, dict)
                or job_payload.get("confirm") != "apply-updates"
            )
        ):
            return _payload(
                error={
                    "code": "update_confirmation_required",
                    "message": "Confirm the staged operating-system update before installing it.",
                },
                status=400,
            )
        # The recovery jobs run a one-use plan: only their Settings > Recovery
        # routes create them, after staging/approving that plan, so this
        # generic door refuses them for every account (w2-creds review).
        if job_type in REPLAN_REQUIRED_JOB_TYPES:
            return _payload(
                error={"code": "recovery_route_required", "message": RECOVERY_ROUTE_REQUIRED},
                status=403,
            )
        if (
            job_type == "host.vnc.enable"
            and (
                not isinstance(job_payload, dict)
                or job_payload.get("confirm") != "enable-host-vnc"
            )
        ):
            return _payload(
                error={
                    "code": "host_vnc_confirmation_required",
                    "message": "Review and confirm host browser desktop setup.",
                },
                status=400,
            )
        # Every cluster job is administrator-gated (fleet initialize/join/remove,
        # app and model deploy, service lifecycle, and the model-cache pull and
        # remove), so the whole `cluster.*` namespace is matched by prefix rather
        # than re-listed here - a new cluster job type is administrator-only by
        # default, which is the safe direction.
        if administrator_only(job_type) and g.auth_session.role != "administrator":
            return _administrator_required()
        if job_type == HOST_GPU_MEMORY_JOB and not controller_job_confirmed(job_payload):
            return _payload(
                error={"code": "gpu_memory_confirmation_required", "message": CONFIRMATION_REQUIRED},
                status=400,
            )
        if job_type == "checkpoint.restore":
            inventory = callbacks.get("checkpoints")
            if inventory is None:
                return _payload(
                    error={
                        "code": "checkpoint_restore_unavailable",
                        "message": "Checkpoint restore is unavailable.",
                    },
                    status=503,
                )
            try:
                job_payload = _bind_checkpoint_restore(inventory, job_payload)
            except (AttributeError, OSError, ValueError) as error:
                return _payload(
                    error={
                        "code": "checkpoint_restore_rejected",
                        "message": str(error),
                    },
                    status=400,
                )
        cluster_confirmations = CLUSTER_JOB_CONFIRMATIONS
        if job_type == "cluster.node.availability":
            availability = (
                str(job_payload.get("availability", "")).strip().lower()
                if isinstance(job_payload, dict) else ""
            )
            expected = (
                f"set-worker-{availability}"
                if availability in {"active", "pause", "drain"} else None
            )
            if expected is None or job_payload.get("confirm") != expected:
                return _payload(
                    error={
                        "code": "cluster_confirmation_required",
                        "message": "Review and confirm the worker availability change first.",
                    },
                    status=400,
                )
        if job_type == "cluster.service.manage":
            service_action = (
                str(job_payload.get("action", "")).strip().lower()
                if isinstance(job_payload, dict) else ""
            )
            expected = {
                "restart": "restart-cluster-service",
                "refresh": "refresh-cluster-service",
                "rollback": "rollback-cluster-service",
            }.get(service_action)
            if expected is None or job_payload.get("confirm") != expected:
                return _payload(
                    error={
                        "code": "cluster_confirmation_required",
                        "message": "Review and confirm this cluster service change first.",
                    },
                    status=400,
                )
        _agent_deploy = job_type == "cluster.agent.deploy" and isinstance(job_payload, dict)
        if _agent_deploy:
            stamp_agent_deploy(job_payload, callbacks, g.auth_session.username)
        if job_type in cluster_confirmations and not confirmation_accepted(
            job_type, job_payload
        ):
            return _payload(
                error={
                    "code": "cluster_confirmation_required",
                    "message": "Review and confirm this cluster change first.",
                },
                status=400,
            )
        _key_reveal = None
        if _agent_deploy:
            # GG14: the agent's first key, minted here and shown once in this reply.
            try:
                _key_reveal = mint_first_agent_key(job_payload, callbacks.get("credential_broker"))
            except (AttributeError, CredentialError):
                return _payload(error={"code": "agent_key_mint_failed", "message": FIRST_KEY_MINT_FAILED}, status=503)
        try:
            job = job_store.create(
                job_type,
                g.auth_session.username,
                job_payload,
            )
        except Exception as error:
            # Whatever refused the queueing, the first key minted for it opens
            # nothing and must not stay live (S2).
            revoke_unqueued_key(callbacks.get("credential_broker"), _key_reveal)
            if not isinstance(error, ValueError):
                raise
            return _payload(
                error={"code": "invalid_job_request", "message": str(error)},
                status=400,
            )
        security.audit(
            g.auth_session.username,
            "job.create",
            "success",
            target=job["id"],
            remote_addr=request.remote_addr or "",
            details={
                "type": job_type,
                "deduplicated": bool(job.get("deduplicated")),
                "resource_id": job.get("resource_id"),
                "display_identity": job.get("display_identity"),
            },
        )
        return _payload({**job, "key_reveal": _key_reveal} if _key_reveal else job, status=202)

    @blueprint.get("/jobs/<job_id>")
    @require_auth("operator")
    def job_detail(job_id):
        job_store = callbacks.get("job_store")
        actor = None if g.auth_session.role == "administrator" else g.auth_session.username
        job = job_store.get(job_id, actor=actor) if job_store is not None else None
        if job is None:
            return _payload(
                error={"code": "job_not_found", "message": "Job was not found."},
                status=404,
            )
        job["events"] = job_store.events(job_id, actor=actor)
        return _payload(job)

    @blueprint.post("/jobs/<job_id>/cancel")
    @require_auth("operator", csrf=True)
    def cancel_job(job_id):
        job_store = callbacks.get("job_store")
        try:
            actor = None if g.auth_session.role == "administrator" else g.auth_session.username
            job = job_store.request_cancel(job_id, actor=actor)
            revoke_cancelled_deploy_key(callbacks.get("credential_broker"), job)
        except KeyError:
            return _payload(error={"code": "job_not_found", "message": "Job was not found."}, status=404)
        except (AttributeError, ValueError) as error:
            return _payload(error={"code": "job_cancel_rejected", "message": str(error)}, status=400)
        security.audit(
            g.auth_session.username, "job.cancel", "success", target=job_id,
            remote_addr=request.remote_addr or "",
        )
        return _payload(job)

    @blueprint.post("/jobs/<job_id>/retry")
    @require_auth("operator", csrf=True)
    def retry_job(job_id):
        job_store = callbacks.get("job_store")
        # Review B1: a retry creates a job, so it takes the role the create
        # route takes. The store's own actor filter is not that gate - it lets
        # anyone retry a job in their own name, whatever they may create now.
        original = job_store.get(job_id) if job_store is not None else None
        if (
            original is not None and administrator_only(original.get("type"))
            and g.auth_session.role != "administrator"
        ):
            return _administrator_required()
        if agent_deploy_retry_refused(job_store, job_id):
            return _payload(error={"code": "job_retry_rejected", "message": AGENT_DEPLOY_RETRY_REFUSED}, status=400)
        try:
            job = job_store.retry(
                job_id,
                g.auth_session.username,
                allow_all=g.auth_session.role == "administrator",
            )
        except KeyError:
            return _payload(error={"code": "job_not_found", "message": "Job was not found."}, status=404)
        except (AttributeError, ValueError) as error:
            return _payload(error={"code": "job_retry_rejected", "message": str(error)}, status=400)
        security.audit(
            g.auth_session.username, "job.retry", "success", target=job["id"],
            remote_addr=request.remote_addr or "",
            details={
                "retry_of": job_id,
                "resource_id": job.get("resource_id"),
                "display_identity": job.get("display_identity"),
            },
        )
        return _payload(job, status=202)

    @blueprint.get("/stream/jobs")
    @require_auth("operator")
    def job_stream():
        job_store = callbacks.get("job_store")
        @stream_with_context
        def generate():
            previous = None
            while True:
                actor = None if g.auth_session.role == "administrator" else g.auth_session.username
                snapshot = job_store.list(50, actor=actor) if job_store is not None else []
                encoded = json.dumps(snapshot, separators=(",", ":"), default=str)
                if encoded != previous:
                    yield "event: jobs\ndata: {}\n\n".format(encoded)
                    previous = encoded
                else:
                    yield ": keepalive\n\n"
                time.sleep(1)
        return Response(
            generate(), mimetype="text/event-stream",
            headers={"Cache-Control": "no-cache, no-transform", "X-Accel-Buffering": "no"},
        )
