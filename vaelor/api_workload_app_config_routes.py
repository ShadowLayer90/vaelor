"""Admin routes to reveal Vaelor-generated secrets and edit app config files.

Split out of ``api_workload_routes`` so that module stays under the 1,000-line
ceiling. These endpoints are the administrator-only, audited surface behind two
live-tested defects: a generated password the Configuration tab redacts (so an
operator had no way to retrieve it), and file-configured apps such as Homepage
whose settings live in YAML files in the data volume rather than a web UI.

Every write is bounded and audited, and the disk/container access is delegated
to :class:`~vaelor.workload_inventory.WorkloadInventory`, which resolves the
template, enforces the declared-file allowlist, and copies through the
allowlisted ``docker cp`` broker shape.
"""

from __future__ import annotations

from flask import Response, g, request
from werkzeug.exceptions import RequestEntityTooLarge

from .api_common import ApiContext, payload as _payload


#: Per-file ceiling on a file-manager upload (bytes). The browser reads only up
#: to this many bytes from the multipart stream, so an over-limit upload is
#: refused without being buffered whole.
FS_UPLOAD_MAX_BYTES = 100 * 1024 * 1024

#: Hard ceiling Werkzeug enforces on the whole multipart body during parsing -
#: the per-file cap plus room for the multipart envelope (boundaries, the small
#: ``path`` field, per-part headers) - so a chunked body with no Content-Length
#: is stopped mid-parse rather than spooled to disk.
_UPLOAD_BODY_CEILING = FS_UPLOAD_MAX_BYTES + 1024 * 1024


def _disposition_name(filename: str) -> str:
    """A header-safe attachment name (no quotes or line breaks)."""
    return "".join(
        character for character in str(filename)
        if character not in '"\r\n'
    ) or "download"


def register_workload_app_config_routes(context: ApiContext) -> None:
    blueprint = context.blueprint
    callbacks = context.callbacks
    security = context.security
    require_auth = context.require_auth

    @blueprint.get("/managed/apps/<app_id>/credentials")
    @require_auth("administrator")
    def managed_app_credentials(app_id):
        inventory = callbacks.get("workload_inventory")
        try:
            secrets = inventory.read_secrets(app_id)
        except (AttributeError, OSError, ValueError) as error:
            return _payload(
                error={"code": "app_credentials_unavailable", "message": str(error)},
                status=400,
            )
        security.audit(
            g.auth_session.username, "app.credentials.read", "success", target=app_id,
            remote_addr=request.remote_addr or "", details={"keys": sorted(secrets)},
        )
        return _payload({"secrets": secrets})

    @blueprint.get("/managed/apps/<app_id>/files")
    @require_auth("administrator")
    def managed_app_files(app_id):
        inventory = callbacks.get("workload_inventory")
        try:
            return _payload({"files": inventory.list_app_files(app_id)})
        except (AttributeError, OSError, ValueError) as error:
            return _payload(
                error={"code": "app_files_unavailable", "message": str(error)},
                status=400,
            )

    @blueprint.get("/managed/apps/<app_id>/files/content")
    @require_auth("administrator")
    def managed_app_file_read(app_id):
        inventory = callbacks.get("workload_inventory")
        try:
            return _payload(inventory.read_app_file(app_id, request.args.get("path", "")))
        except (AttributeError, OSError, ValueError) as error:
            return _payload(
                error={"code": "app_file_unavailable", "message": str(error)},
                status=400,
            )

    @blueprint.put("/managed/apps/<app_id>/files/content")
    @require_auth("administrator", csrf=True)
    def managed_app_file_write(app_id):
        inventory = callbacks.get("workload_inventory")
        body = request.get_json(silent=True) or {}
        path = str(body.get("path", ""))
        try:
            result = inventory.write_app_file(app_id, path, body.get("content", ""))
        except (AttributeError, OSError, ValueError) as error:
            return _payload(
                error={"code": "app_file_invalid", "message": str(error)},
                status=400,
            )
        security.audit(
            g.auth_session.username, "app.file.update", "success", target=app_id,
            remote_addr=request.remote_addr or "", details={"path": path},
        )
        return _payload(result)

    @blueprint.post("/managed/apps/<app_id>/exec")
    @require_auth("administrator", csrf=True)
    def managed_app_exec(app_id):
        # Administrator console: run one shell command inside the app's own
        # container (reset a password, change a setting). Container-confined and
        # broker-allowlisted; every command is audited before its output returns.
        inventory = callbacks.get("workload_inventory")
        command = str((request.get_json(silent=True) or {}).get("command", ""))
        security.audit(
            g.auth_session.username, "app.console.exec", "attempt", target=app_id,
            remote_addr=request.remote_addr or "", details={"command": command[:512]},
        )
        try:
            result = inventory.exec_in_app(app_id, command)
        except (AttributeError, OSError, ValueError) as error:
            security.audit(
                g.auth_session.username, "app.console.exec", "failure", target=app_id,
                remote_addr=request.remote_addr or "",
                details={"command": command[:512], "error": str(error)[:256]},
            )
            return _payload(
                error={"code": "app_exec_failed", "message": str(error)},
                status=400,
            )
        # Record the outcome (exit code) so the trail shows not just that a command
        # was attempted but how it ran.
        security.audit(
            g.auth_session.username, "app.console.exec",
            "success" if result.get("exit_code") == 0 else "failure",
            target=app_id, remote_addr=request.remote_addr or "",
            details={"command": command[:512], "exit_code": result.get("exit_code")},
        )
        return _payload(result)

    # File manager over the app's data roots. Every path is confined at the
    # broker independently of these routes; the browser resolves the app to a
    # curated template and rechecks the path before acting. Mutations require
    # the administrator role plus CSRF and are audited.
    @blueprint.get("/managed/apps/<app_id>/fs/list")
    @require_auth("administrator")
    def managed_app_fs_list(app_id):
        browser = callbacks.get("app_file_browser")
        try:
            return _payload(browser.list_dir(app_id, request.args.get("path") or None))
        except (AttributeError, OSError, ValueError) as error:
            return _payload(
                error={"code": "app_fs_unavailable", "message": str(error)},
                status=400,
            )

    @blueprint.post("/managed/apps/<app_id>/fs/mkdir")
    @require_auth("administrator", csrf=True)
    def managed_app_fs_mkdir(app_id):
        browser = callbacks.get("app_file_browser")
        body = request.get_json(silent=True) or {}
        path = str(body.get("path", ""))
        name = str(body.get("name", ""))
        security.audit(
            g.auth_session.username, "app.fs.mkdir", "attempt", target=app_id,
            remote_addr=request.remote_addr or "", details={"path": path, "name": name},
        )
        try:
            result = browser.make_dir(app_id, path or None, name)
        except (AttributeError, OSError, ValueError) as error:
            security.audit(
                g.auth_session.username, "app.fs.mkdir", "failure", target=app_id,
                remote_addr=request.remote_addr or "",
                details={"path": path, "name": name, "error": str(error)[:256]},
            )
            return _payload(
                error={"code": "app_fs_invalid", "message": str(error)},
                status=400,
            )
        security.audit(
            g.auth_session.username, "app.fs.mkdir", "success", target=app_id,
            remote_addr=request.remote_addr or "",
            details={"path": path, "name": name},
        )
        return _payload(result)

    @blueprint.post("/managed/apps/<app_id>/fs/delete")
    @require_auth("administrator", csrf=True)
    def managed_app_fs_delete(app_id):
        browser = callbacks.get("app_file_browser")
        path = str((request.get_json(silent=True) or {}).get("path", ""))
        security.audit(
            g.auth_session.username, "app.fs.delete", "attempt", target=app_id,
            remote_addr=request.remote_addr or "", details={"path": path},
        )
        try:
            result = browser.delete_path(app_id, path)
        except (AttributeError, OSError, ValueError) as error:
            security.audit(
                g.auth_session.username, "app.fs.delete", "failure", target=app_id,
                remote_addr=request.remote_addr or "",
                details={"path": path, "error": str(error)[:256]},
            )
            return _payload(
                error={"code": "app_fs_invalid", "message": str(error)},
                status=400,
            )
        security.audit(
            g.auth_session.username, "app.fs.delete", "success", target=app_id,
            remote_addr=request.remote_addr or "", details={"path": path},
        )
        return _payload(result)

    @blueprint.post("/managed/apps/<app_id>/fs/upload")
    @require_auth("administrator", csrf=True)
    def managed_app_fs_upload(app_id):
        browser = callbacks.get("app_file_browser")
        # Bound the body BEFORE any parsing touches request.form/request.files:
        # Werkzeug enforces this ceiling while it reads (and would otherwise
        # spool) the multipart body, so a chunked or Content-Length-less request
        # that skips a header check cannot stream gigabytes to disk. The ceiling
        # is the per-file cap plus a small multipart-envelope allowance.
        request.max_content_length = _UPLOAD_BODY_CEILING
        try:
            path = str(request.form.get("path", ""))
            upload = request.files.get("file")
        except RequestEntityTooLarge:
            return _payload(
                error={"code": "app_fs_too_large", "message": "That upload is too large."},
                status=413,
            )
        if upload is None:
            return _payload(
                error={"code": "app_fs_invalid", "message": "Attach a file to upload."},
                status=400,
            )
        filename = upload.filename or ""
        security.audit(
            g.auth_session.username, "app.fs.upload", "attempt", target=app_id,
            remote_addr=request.remote_addr or "",
            details={"path": path, "filename": filename},
        )
        try:
            result = browser.upload(
                app_id, path or None, filename, upload.stream, FS_UPLOAD_MAX_BYTES
            )
        except (AttributeError, OSError, ValueError) as error:
            security.audit(
                g.auth_session.username, "app.fs.upload", "failure", target=app_id,
                remote_addr=request.remote_addr or "",
                details={"path": path, "filename": filename, "error": str(error)[:256]},
            )
            return _payload(
                error={"code": "app_fs_invalid", "message": str(error)},
                status=400,
            )
        security.audit(
            g.auth_session.username, "app.fs.upload", "success", target=app_id,
            remote_addr=request.remote_addr or "",
            details={"path": path, "filename": filename, "size": result.get("size")},
        )
        return _payload(result)

    @blueprint.get("/managed/apps/<app_id>/fs/download")
    @require_auth("administrator")
    def managed_app_fs_download(app_id):
        browser = callbacks.get("app_file_browser")
        path = request.args.get("path", "")
        try:
            with browser.download(app_id, path) as (host_path, filename):
                data = host_path.read_bytes()
        except (AttributeError, OSError, ValueError) as error:
            return _payload(
                error={"code": "app_fs_unavailable", "message": str(error)},
                status=400,
            )
        security.audit(
            g.auth_session.username, "app.fs.download", "success", target=app_id,
            remote_addr=request.remote_addr or "",
            details={"path": path, "size": len(data)},
        )
        return Response(
            data,
            mimetype="application/octet-stream",
            headers={
                "Content-Disposition": (
                    'attachment; filename="{}"'.format(_disposition_name(filename))
                ),
            },
        )
