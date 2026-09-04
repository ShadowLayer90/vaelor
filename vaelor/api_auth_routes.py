"""Authentication, security, and administration routes."""

from __future__ import annotations

import hashlib
import os
import re
import ssl
import time
import uuid
from pathlib import Path

from flask import Response, g, request, send_file

from .appliance_recovery import (
    CONFIRMATION as RESET_CONFIRMATION,
    IMPORT_CONFIRMATION,
)
from .api_common import (
    ApiContext,
    LEGACY_SESSION_COOKIE,
    SESSION_COOKIE,
    payload as _payload,
)
from .runtime_paths import env_value
from .portable_state import MAX_ARCHIVE_BYTES, PortableStateError
from .security import LastAdministratorError


def register_auth_routes(context: ApiContext) -> None:
    blueprint = context.blueprint
    callbacks = context.callbacks
    security = context.security
    limiter = context.limiter
    require_auth = context.require_auth
    appliance_address = context.appliance_address

    @blueprint.get("/auth/status")
    def auth_status():
        return _payload({"bootstrap_required": not security.has_users()})

    @blueprint.get("/security/transport")
    @require_auth("viewer")
    def transport_security():
        certificate_path = Path(env_value(
            "VAELOR_TLS_CERT", "PM_TLS_CERT", ""
        ))
        fingerprint = ""
        if certificate_path.is_file():
            try:
                pem = certificate_path.read_text(encoding="ascii")
                fingerprint = hashlib.sha256(
                    ssl.PEM_cert_to_DER_cert(pem)
                ).hexdigest()
            except (OSError, ValueError):
                fingerprint = ""
        return _payload({
            "secure": request.is_secure,
            "scheme": "https" if request.is_secure else "http",
            "certificate_managed": bool(fingerprint),
            "certificate_fingerprint": fingerprint,
            "vnc_secure": bool(request.is_secure and fingerprint),
            "remote_ready": bool(request.is_secure and fingerprint),
        })

    @blueprint.get("/security/certificate")
    @require_auth("administrator")
    def transport_certificate():
        certificate_path = Path(env_value(
            "VAELOR_TLS_CERT", "PM_TLS_CERT", ""
        ))
        try:
            content = certificate_path.read_bytes()
        except OSError:
            return _payload(
                error={
                    "code": "certificate_unavailable",
                    "message": "The local HTTPS certificate is not configured.",
                },
                status=404,
            )
        return Response(
            content,
            mimetype="application/x-pem-file",
            headers={
                "Content-Disposition": "attachment; filename=vaelor-local.crt",
                "Cache-Control": "no-store",
            },
        )

    @blueprint.post("/auth/bootstrap")
    def bootstrap():
        body = request.get_json(silent=True) or {}
        username = str(body.get("username", "")).strip().lower()
        password = str(body.get("password", ""))
        if not username or len(username) > 64:
            return _payload(
                error={"code": "invalid_username", "message": "Enter a valid username."},
                status=400,
            )
        if len(password) < 12:
            return _payload(
                error={
                    "code": "weak_password",
                    "message": "Use a passphrase with at least 12 characters.",
                },
                status=400,
            )
        if not security.bootstrap(username, password):
            return _payload(
                error={
                    "code": "already_configured",
                    "message": "The administrator account is already configured.",
                },
                status=409,
            )
        security.audit(
            username,
            "auth.bootstrap",
            "success",
            target=username,
            remote_addr=request.remote_addr or "",
        )
        return _payload({"username": username, "role": "administrator"}, status=201)

    @blueprint.post("/auth/login")
    def login():
        body = request.get_json(silent=True) or {}
        username = str(body.get("username", "")).strip().lower()
        password = str(body.get("password", ""))
        limiter_key = "{}:{}".format(request.remote_addr or "unknown", username)
        if not limiter.allowed(limiter_key):
            return _payload(
                error={
                    "code": "login_rate_limited",
                    "message": "Too many attempts. Try again in a few minutes.",
                },
                status=429,
            )
        user = security.authenticate(username, password)
        if user is None:
            limiter.failed(limiter_key)
            security.audit(
                username or "unknown",
                "auth.login",
                "failure",
                remote_addr=request.remote_addr or "",
            )
            return _payload(
                error={
                    "code": "invalid_credentials",
                    "message": "The username or password is incorrect.",
                },
                status=401,
            )
        if user.get("mfa_enabled") and not security.verify_totp(
            username, str(body.get("totp_code", ""))
        ):
            limiter.failed(limiter_key)
            security.audit(
                username, "auth.mfa", "failure",
                remote_addr=request.remote_addr or "",
            )
            return _payload(
                error={
                    "code": "totp_required",
                    "message": "Enter the six-digit code from your authenticator app.",
                },
                status=401,
            )
        limiter.succeeded(limiter_key)
        created = security.create_session(
            user["username"],
            request.remote_addr or "",
            request.headers.get("User-Agent", ""),
        )
        security.audit(
            user["username"],
            "auth.login",
            "success",
            remote_addr=request.remote_addr or "",
        )
        response, status = _payload(
            {
                "user": {"username": user["username"], "role": user["role"]},
                "csrf_token": created["csrf_token"],
                "expires_at": created["expires_at"],
            }
        )
        secure_cookie = request.is_secure or env_value(
            "VAELOR_SECURE_COOKIES", "PM_DASHBOARD_SECURE_COOKIES", "0"
        ) == "1"
        response.set_cookie(
            SESSION_COOKIE,
            created["token"],
            max_age=security.session_seconds,
            secure=secure_cookie,
            httponly=True,
            samesite="Strict",
            path="/",
        )
        return response, status

    @blueprint.get("/auth/session")
    @require_auth("viewer")
    def current_session():
        session = g.auth_session
        csrf_token = security.csrf_for(g.auth_token)
        return _payload(
            {
                "user": {"username": session.username, "role": session.role},
                "expires_at": session.expires_at,
                "csrf_token": csrf_token,
            }
        )

    @blueprint.post("/auth/logout")
    @require_auth("viewer", csrf=True)
    def logout():
        security.revoke_session(g.auth_token)
        security.audit(
            g.auth_session.username,
            "auth.logout",
            "success",
            remote_addr=request.remote_addr or "",
        )
        response, status = _payload({"signed_out": True})
        response.delete_cookie(SESSION_COOKIE, path="/")
        response.delete_cookie(LEGACY_SESSION_COOKIE, path="/")
        return response, status

    @blueprint.get("/auth/totp")
    @require_auth("viewer")
    def totp_status():
        user = next(
            item for item in security.list_users()
            if item["username"] == g.auth_session.username
        )
        return _payload({"enabled": user["mfa_enabled"]})

    @blueprint.post("/auth/totp/setup")
    @require_auth("viewer", csrf=True)
    def totp_setup():
        current = next(
            item for item in security.list_users()
            if item["username"] == g.auth_session.username
        )
        if current["mfa_enabled"]:
            return _payload(
                error={
                    "code": "totp_already_enabled",
                    "message": "Verify and disable the current authenticator before replacing it.",
                },
                status=409,
            )
        result = security.begin_totp(g.auth_session.username)
        security.audit(g.auth_session.username, "auth.totp.setup", "success", target=g.auth_session.username, remote_addr=request.remote_addr or "")
        return _payload(result, status=201)

    @blueprint.post("/auth/totp/confirm")
    @require_auth("viewer", csrf=True)
    def totp_confirm():
        body = request.get_json(silent=True) or {}
        try:
            result = security.confirm_totp(
                g.auth_session.username, body.get("code", "")
            )
        except ValueError as error:
            return _payload(error={"code": "invalid_totp", "message": str(error)}, status=400)
        security.audit(g.auth_session.username, "auth.totp.enable", "success", target=g.auth_session.username, remote_addr=request.remote_addr or "")
        return _payload(result)

    @blueprint.delete("/auth/totp")
    @require_auth("viewer", csrf=True)
    def totp_disable():
        body = request.get_json(silent=True) or {}
        try:
            result = security.disable_totp(
                g.auth_session.username, body.get("code", "")
            )
        except ValueError as error:
            return _payload(error={"code": "invalid_totp", "message": str(error)}, status=400)
        security.audit(g.auth_session.username, "auth.totp.disable", "success", target=g.auth_session.username, remote_addr=request.remote_addr or "")
        return _payload(result)

    @blueprint.get("/admin/users")
    @require_auth("administrator")
    def admin_users():
        return _payload(security.list_users())

    @blueprint.post("/admin/users")
    @require_auth("administrator", csrf=True)
    def admin_user_create():
        body = request.get_json(silent=True) or {}
        username = str(body.get("username", "")).strip().lower()
        password = str(body.get("password", ""))
        role = str(body.get("role", "viewer")).strip().lower()
        if not re.fullmatch(r"[a-z0-9][a-z0-9_.-]{1,63}", username):
            return _payload(error={"code": "invalid_username", "message": "Use 2-64 lowercase letters, numbers, dots, dashes, or underscores."}, status=400)
        if len(password) < 12 or len(password) > 256:
            return _payload(error={"code": "weak_password", "message": "Use a passphrase with 12-256 characters."}, status=400)
        try:
            user = security.create_user(username, password, role)
        except ValueError as error:
            return _payload(error={"code": "invalid_user", "message": str(error)}, status=400)
        security.audit(g.auth_session.username, "admin.user.create", "success", target=username, remote_addr=request.remote_addr or "", details={"role": role})
        return _payload(user, status=201)

    @blueprint.patch("/admin/users/<username>")
    @require_auth("administrator", csrf=True)
    def admin_user_update(username):
        body = request.get_json(silent=True) or {}
        role = body.get("role")
        enabled = body.get("enabled")
        password = body.get("password")
        if enabled is not None and not isinstance(enabled, bool):
            return _payload(error={"code": "invalid_user", "message": "Account access must be enabled or disabled."}, status=400)
        if password is not None and not 12 <= len(str(password)) <= 256:
            return _payload(error={"code": "weak_password", "message": "Use a passphrase with 12-256 characters."}, status=400)
        users = {item["username"]: item for item in security.list_users()}
        existing = users.get(username)
        if existing is None:
            return _payload(error={"code": "user_not_found", "message": "Account was not found."}, status=404)
        removes_admin = existing["role"] == "administrator" and (
            role not in (None, "administrator") or enabled is False
        )
        if removes_admin and security.administrator_count() <= 1:
            return _payload(error={"code": "last_administrator", "message": "Keep at least one enabled administrator account."}, status=409)
        if username == g.auth_session.username and enabled is False:
            return _payload(error={"code": "current_account", "message": "You cannot disable the account you are using."}, status=409)
        try:
            user = security.update_user(
                username,
                role=str(role).lower() if role is not None else None,
                enabled=enabled,
                password=str(password) if password is not None else None,
            )
        except LastAdministratorError as error:
            return _payload(error={"code": "last_administrator", "message": str(error)}, status=409)
        except ValueError as error:
            return _payload(error={"code": "invalid_user", "message": str(error)}, status=400)
        security.audit(
            g.auth_session.username,
            "admin.user.update",
            "success",
            target=username,
            remote_addr=request.remote_addr or "",
            details={"fields": sorted(body)},
        )
        return _payload(user)

    @blueprint.delete("/admin/users/<username>")
    @require_auth("administrator", csrf=True)
    def admin_user_delete(username):
        users = {item["username"]: item for item in security.list_users()}
        existing = users.get(username)
        if existing is None:
            return _payload(
                error={"code": "user_not_found", "message": "Account was not found."},
                status=404,
            )
        if (
            existing["role"] == "administrator"
            and existing["enabled"]
            and security.administrator_count() <= 1
        ):
            return _payload(
                error={
                    "code": "last_administrator",
                    "message": "Keep at least one enabled administrator account.",
                },
                status=409,
            )
        if username == g.auth_session.username:
            return _payload(
                error={
                    "code": "current_account",
                    "message": "You cannot delete the account you are using.",
                },
                status=409,
            )
        try:
            deleted = security.delete_user(username)
        except LastAdministratorError as error:
            return _payload(
                error={"code": "last_administrator", "message": str(error)},
                status=409,
            )
        if not deleted:
            return _payload(
                error={"code": "user_not_found", "message": "Account was not found."},
                status=404,
            )
        security.audit(
            g.auth_session.username,
            "admin.user.delete",
            "success",
            target=username,
            remote_addr=request.remote_addr or "",
        )
        return _payload({"deleted": True, "username": username})

    @blueprint.get("/admin/sessions")
    @require_auth("administrator")
    def admin_sessions():
        current_hash = security._hash_token(g.auth_token)
        sessions = security.list_sessions()
        for item in sessions:
            item["current"] = item["id"] == current_hash
        return _payload(sessions)

    @blueprint.delete("/admin/sessions/<session_id>")
    @require_auth("administrator", csrf=True)
    def admin_session_revoke(session_id):
        if not re.fullmatch(r"[a-f0-9]{64}", session_id):
            return _payload(error={"code": "invalid_session", "message": "Session identifier is invalid."}, status=400)
        if session_id == security._hash_token(g.auth_token):
            return _payload(error={"code": "current_session", "message": "Sign out to end your current session."}, status=409)
        if not security.revoke_session_id(session_id):
            return _payload(error={"code": "session_not_found", "message": "Session was not found."}, status=404)
        security.audit(g.auth_session.username, "admin.session.revoke", "success", target=session_id[:12], remote_addr=request.remote_addr or "")
        return _payload({"revoked": True})

    @blueprint.get("/admin/recovery/factory-reset")
    @require_auth("administrator")
    def factory_reset_status():
        plans = callbacks.get("factory_reset_plans")
        if plans is None:
            return _payload(
                error={
                    "code": "factory_reset_unavailable",
                    "message": "The appliance recovery service is unavailable.",
                },
                status=503,
            )
        status = plans.status()
        return _payload({
            "staged": False,
            "plan": None,
            "confirmation": status["confirmation"],
            "erases": status["erases"],
            "retains": status["retains"],
        })

    @blueprint.post("/admin/recovery/factory-reset")
    @require_auth("administrator", csrf=True)
    def factory_reset_execute():
        plans = callbacks.get("factory_reset_plans")
        jobs = callbacks.get("job_store")
        body = request.get_json(silent=True) or {}
        if plans is None or jobs is None:
            return _payload(
                error={
                    "code": "factory_reset_unavailable",
                    "message": "The appliance recovery service is unavailable.",
                },
                status=503,
            )
        confirmation = str(body.get("confirmation", ""))
        if confirmation != RESET_CONFIRMATION:
            return _payload(
                error={
                    "code": "factory_reset_confirmation",
                    "message": "Type the displayed confirmation exactly before resetting.",
                },
                status=400,
            )
        handoff = plans.stage(g.auth_session.username)["plan"]
        job = jobs.create(
            "appliance.factory-reset",
            g.auth_session.username,
            {"plan_id": handoff["id"], "confirmation": confirmation},
        )
        security.audit(
            g.auth_session.username,
            "appliance.factory_reset.approve",
            "success",
            target=job["id"],
            remote_addr=request.remote_addr or "",
        )
        return _payload(job, status=202)

    @blueprint.post("/admin/recovery/factory-reset/stage")
    @require_auth("administrator", csrf=True)
    def factory_reset_stage():
        plans = callbacks.get("factory_reset_plans")
        if plans is None:
            return _payload(
                error={
                    "code": "factory_reset_unavailable",
                    "message": "The appliance recovery service is unavailable.",
                },
                status=503,
            )
        result = plans.stage(g.auth_session.username)
        security.audit(
            g.auth_session.username,
            "appliance.factory_reset.stage",
            "success",
            target=result["plan"]["id"],
            remote_addr=request.remote_addr or "",
        )
        return _payload(result, status=201)

    @blueprint.delete("/admin/recovery/factory-reset/stage")
    @require_auth("administrator", csrf=True)
    def factory_reset_cancel():
        plans = callbacks.get("factory_reset_plans")
        if plans is None:
            return _payload(
                error={
                    "code": "factory_reset_unavailable",
                    "message": "The appliance recovery service is unavailable.",
                },
                status=503,
            )
        result = plans.cancel()
        security.audit(
            g.auth_session.username,
            "appliance.factory_reset.cancel",
            "success",
            remote_addr=request.remote_addr or "",
        )
        return _payload(result)

    @blueprint.get("/admin/portable-state")
    @require_auth("administrator")
    def portable_state_status():
        plans = callbacks.get("portable_import_plans")
        if plans is None:
            return _payload(
                error={
                    "code": "portable_state_unavailable",
                    "message": "Portable state transfer is unavailable.",
                },
                status=503,
            )
        return _payload(plans.status())

    @blueprint.post("/admin/portable-state/export")
    @require_auth("administrator", csrf=True)
    def portable_state_export():
        plans = callbacks.get("portable_import_plans")
        if plans is None:
            return _payload(
                error={
                    "code": "portable_state_unavailable",
                    "message": "Portable state transfer is unavailable.",
                },
                status=503,
            )
        body = request.get_json(silent=True) or {}
        passphrase = str(body.get("passphrase", ""))
        export_root = plans.export_root
        export_root.mkdir(parents=True, exist_ok=True)
        archive = export_root / f"{uuid.uuid4().hex}.vaelor"
        try:
            plans.portable_state.export(archive, passphrase)
        except (OSError, PortableStateError) as error:
            archive.unlink(missing_ok=True)
            return _payload(
                error={"code": "portable_export_failed", "message": str(error)},
                status=400,
            )
        security.audit(
            g.auth_session.username,
            "appliance.portable_state.export",
            "success",
            remote_addr=request.remote_addr or "",
        )

        response = send_file(
            archive,
            as_attachment=True,
            download_name=f"vaelor-state-{time.strftime('%Y%m%d')}.vaelor",
            mimetype="application/octet-stream",
            max_age=0,
        )
        response.call_on_close(lambda: archive.unlink(missing_ok=True))
        return response

    @blueprint.post("/admin/portable-state/import/stage")
    @require_auth("administrator", csrf=True)
    def portable_state_import_stage():
        plans = callbacks.get("portable_import_plans")
        transfer = request.files.get("archive")
        passphrase = str(request.form.get("passphrase", ""))
        if plans is None:
            return _payload(
                error={
                    "code": "portable_state_unavailable",
                    "message": "Portable state transfer is unavailable.",
                },
                status=503,
            )
        if transfer is None:
            return _payload(
                error={
                    "code": "portable_import_missing",
                    "message": "Choose an encrypted Vaelor transfer archive.",
                },
                status=400,
            )
        plans.import_root.mkdir(parents=True, exist_ok=True)
        archive = plans.import_root / f"{uuid.uuid4().hex}.vaelor"
        descriptor = os.open(
            archive, os.O_CREAT | os.O_EXCL | os.O_WRONLY, 0o600
        )
        total = 0
        try:
            with os.fdopen(descriptor, "wb") as output:
                while True:
                    chunk = transfer.stream.read(1024 * 1024)
                    if not chunk:
                        break
                    total += len(chunk)
                    if total > MAX_ARCHIVE_BYTES:
                        raise PortableStateError(
                            "The transfer archive exceeds the 512 MiB limit."
                        )
                    output.write(chunk)
            staged = plans.stage(g.auth_session.username, archive, passphrase)
        except (OSError, PortableStateError, ValueError) as error:
            archive.unlink(missing_ok=True)
            return _payload(
                error={"code": "portable_import_invalid", "message": str(error)},
                status=400,
            )
        security.audit(
            g.auth_session.username,
            "appliance.portable_state.stage",
            "success",
            target=staged["plan"]["id"],
            remote_addr=request.remote_addr or "",
        )
        return _payload(staged, status=201)

    @blueprint.delete("/admin/portable-state/import/stage")
    @require_auth("administrator", csrf=True)
    def portable_state_import_cancel():
        plans = callbacks.get("portable_import_plans")
        if plans is None:
            return _payload(
                error={
                    "code": "portable_state_unavailable",
                    "message": "Portable state transfer is unavailable.",
                },
                status=503,
            )
        try:
            result = plans.cancel()
        except ValueError as error:
            return _payload(
                error={"code": "portable_import_approved", "message": str(error)},
                status=409,
            )
        security.audit(
            g.auth_session.username,
            "appliance.portable_state.cancel",
            "success",
            remote_addr=request.remote_addr or "",
        )
        return _payload(result)

    @blueprint.post("/admin/portable-state/import")
    @require_auth("administrator", csrf=True)
    def portable_state_import_apply():
        plans = callbacks.get("portable_import_plans")
        jobs = callbacks.get("job_store")
        body = request.get_json(silent=True) or {}
        status = plans.status() if plans is not None else {"plan": None}
        confirmation = str(body.get("confirmation", ""))
        if confirmation != IMPORT_CONFIRMATION:
            return _payload(
                error={
                    "code": "portable_import_confirmation",
                    "message": "Type the displayed import confirmation exactly.",
                },
                status=400,
            )
        if jobs is None or not status.get("plan"):
            return _payload(
                error={
                    "code": "portable_import_not_staged",
                    "message": "Review an encrypted transfer archive first.",
                },
                status=409,
            )
        plan_id = status["plan"]["id"]
        try:
            plans.approve(plan_id)
            job = jobs.create(
                "appliance.portable-import",
                g.auth_session.username,
                {
                    "plan_id": plan_id,
                    "confirmation": confirmation,
                },
            )
        except ValueError as error:
            return _payload(
                error={"code": "portable_import_approved", "message": str(error)},
                status=409,
            )
        except Exception:
            plans.unapprove(plan_id)
            raise
        security.audit(
            g.auth_session.username,
            "appliance.portable_state.approve",
            "success",
            target=job["id"],
            remote_addr=request.remote_addr or "",
        )
        return _payload(job, status=202)

    @blueprint.get("/admin/agent-api-tokens")
    @require_auth("administrator")
    def admin_agent_api_tokens():
        tokens = callbacks.get("agent_api_tokens")
        if tokens is None:
            return _payload(error={"code": "agent_api_unavailable", "message": "Assistant API access is unavailable."}, status=503)
        return _payload(tokens.list())

    @blueprint.get("/admin/inference-gateway")
    @require_auth("administrator")
    def admin_inference_gateway():
        status = callbacks.get("inference_gateway_status")
        if status is None:
            return _payload(
                error={
                    "code": "inference_gateway_unavailable",
                    "message": "Inference gateway status is unavailable.",
                },
                status=503,
            )
        return _payload(status())

    @blueprint.post("/admin/agent-api-tokens")
    @require_auth("administrator", csrf=True)
    def admin_agent_api_token_create():
        tokens = callbacks.get("agent_api_tokens")
        body = request.get_json(silent=True) or {}
        try:
            created = tokens.create(body.get("label", ""), body.get("scopes"))
        except (AttributeError, ValueError) as error:
            return _payload(error={"code": "invalid_agent_api_token", "message": str(error)}, status=400)
        security.audit(g.auth_session.username, "admin.agent_api_token.create", "success", target=created["id"], remote_addr=request.remote_addr or "")
        return _payload(created, status=201)

    @blueprint.delete("/admin/agent-api-tokens/<token_id>")
    @require_auth("administrator", csrf=True)
    def admin_agent_api_token_revoke(token_id):
        tokens = callbacks.get("agent_api_tokens")
        try:
            revoked = tokens.revoke(token_id)
        except (AttributeError, KeyError):
            return _payload(error={"code": "agent_api_token_not_found", "message": "API connection was not found."}, status=404)
        security.audit(g.auth_session.username, "admin.agent_api_token.revoke", "success", target=token_id, remote_addr=request.remote_addr or "")
        return _payload(revoked)
