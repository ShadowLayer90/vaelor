"""Browser entry routes kept separate from the legacy dashboard API."""

from __future__ import annotations

import os
import re

from flask import Response, abort, send_from_directory
from flask_cors import cross_origin

# A content-hashed build artifact under assets/ never changes for a given URL:
# a new build emits a new hash. Tell the browser it may keep it for a year and
# never revalidate.
IMMUTABLE_ASSET_CACHE = "public, max-age=31536000, immutable"
# index.html must be revalidated on every load. It is the one file whose URL is
# stable across deploys while its contents (the asset hashes it points at)
# change, so caching it immutably would strand users on an old build. `no-cache`
# still lets the browser store it - it just may not reuse it without asking.
REVALIDATE_HTML_CACHE = "no-cache"
# The recovery shim below is generated per-request and must never be stored.
NO_STORE_CACHE = "no-store"


def register_frontend_routes(app, www_v2_path: str) -> None:
    @app.route("/v2/")
    @app.route("/v2/<path:path>")
    def dashboard_v2(path=""):
        requested = path or "index.html"
        requested_path = os.path.join(www_v2_path, requested)
        if path and os.path.isfile(requested_path):
            response = send_from_directory(www_v2_path, requested)
            # Only the hashed assets/ files are immutable; other real files
            # served from here (favicon, manifest) keep the default handling.
            if requested.startswith("assets/"):
                response.headers["Cache-Control"] = IMMUTABLE_ASSET_CACHE
            return response
        if requested.startswith("assets/"):
            filename = os.path.basename(requested)
            match = re.fullmatch(
                r"(?P<component>[A-Za-z_$][A-Za-z0-9_$]*)-[A-Za-z0-9_-]+\.js",
                filename,
            )
            if match:
                component = match.group("component")
                script = (
                    "const url=new URL(window.location.href);"
                    "url.searchParams.set('_vaelor_release',Date.now().toString());"
                    "window.location.replace(url.pathname+url.search+url.hash);"
                    f"export function {component}(){{return null;}}"
                    f"export default {component};"
                )
                response = Response(script, mimetype="application/javascript")
                response.headers["X-Vaelor-Stale-Asset-Recovery"] = "1"
                response.headers["Cache-Control"] = NO_STORE_CACHE
                return response
            abort(404)
        response = send_from_directory(www_v2_path, "index.html")
        response.headers["Cache-Control"] = REVALIDATE_HTML_CACHE
        return response

    @app.route("/<path:path>")
    @cross_origin()
    def catch_all(path):
        with open(f"{app.static_folder}/index.html") as source:
            response = Response(source.read(), mimetype="text/html")
        response.headers["Cache-Control"] = REVALIDATE_HTML_CACHE
        return response
