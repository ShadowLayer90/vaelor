"""Least-privileged Unix-socket service for bounded public research."""

from __future__ import annotations

import json
import os
import socket
from pathlib import Path
from typing import Any, Dict, Iterable, Mapping

from .application_research_service import ApplicationResearchService
from .research_provenance import FetchProvenance
from .runtime_paths import env_value, run_path


SOCKET_PATH = Path(env_value(
    "VAELOR_APPLICATION_RESEARCH_SOCKET",
    "PM_APPLICATION_RESEARCH_SOCKET",
    run_path("application-research.sock"),
))
MAX_REQUEST = 64 * 1024
MAX_RESPONSE = 4 * 1024 * 1024


class ApplicationResearchClient:
    def __init__(self, socket_path: str | Path = SOCKET_PATH, timeout: int = 30):
        self.socket_path = str(socket_path)
        self.timeout = max(1, min(int(timeout), 60))

    def research_manifest(
        self, draft: Mapping[str, Any], source_urls: Iterable[str]
    ) -> Dict[str, Any]:
        request = json.dumps({
            "action": "research_manifest",
            "draft": {"intent": dict(draft.get("intent") or {})},
            "source_urls": list(source_urls),
        }, separators=(",", ":")).encode("utf-8") + b"\n"
        if len(request) > MAX_REQUEST:
            raise ValueError("The application research request is too large.")
        try:
            with socket.socket(socket.AF_UNIX, socket.SOCK_STREAM) as client:
                client.settimeout(self.timeout)
                client.connect(self.socket_path)
                client.sendall(request)
                response = _receive(client, MAX_RESPONSE)
        except OSError as error:
            raise ValueError("The application research broker is unavailable.") from error
        try:
            decoded = json.loads(response.decode("utf-8"))
        except (UnicodeDecodeError, json.JSONDecodeError) as error:
            raise ValueError("The application research broker returned invalid data.") from error
        if not isinstance(decoded, dict) or not decoded.get("ok"):
            message = decoded.get("error") if isinstance(decoded, dict) else ""
            raise ValueError(str(message or "Application research failed.")[:500])
        result = decoded.get("data")
        if not isinstance(result, dict):
            raise ValueError("The application research broker returned invalid data.")
        return result

    def fetch_evidence(
        self, url: str, provenance: FetchProvenance | Mapping[str, Any] | None = None,
    ) -> Dict[str, Any]:
        """Fetch one page, carrying the per-hop provenance policy across the socket.

        The policy has to travel with the request: the broker runs in a
        separate least-privileged process and is the only component that sees
        a redirect, so a policy left behind on this side would authorize the
        first hop and nothing after it.
        """
        payload: Dict[str, Any] = {"action": "fetch_evidence", "url": str(url)}
        if provenance is not None:
            payload["provenance"] = (
                provenance.to_dict()
                if isinstance(provenance, FetchProvenance)
                else dict(provenance)
            )
        request = json.dumps(payload, separators=(",", ":")).encode("utf-8") + b"\n"
        return self._request(request)

    def _request(self, request: bytes) -> Dict[str, Any]:
        if len(request) > MAX_REQUEST:
            raise ValueError("The application research request is too large.")
        try:
            with socket.socket(socket.AF_UNIX, socket.SOCK_STREAM) as client:
                client.settimeout(self.timeout)
                client.connect(self.socket_path)
                client.sendall(request)
                response = _receive(client, MAX_RESPONSE)
        except OSError as error:
            raise ValueError("The application research broker is unavailable.") from error
        try:
            decoded = json.loads(response.decode("utf-8"))
        except (UnicodeDecodeError, json.JSONDecodeError) as error:
            raise ValueError("The application research broker returned invalid data.") from error
        if not isinstance(decoded, dict) or not decoded.get("ok"):
            message = decoded.get("error") if isinstance(decoded, dict) else ""
            raise ValueError(str(message or "Application research failed.")[:500])
        result = decoded.get("data")
        if not isinstance(result, dict):
            raise ValueError("The application research broker returned invalid data.")
        return result


def _receive(connection: socket.socket, limit: int) -> bytes:
    value = b""
    while b"\n" not in value and len(value) <= limit:
        chunk = connection.recv(min(65536, limit + 1 - len(value)))
        if not chunk:
            break
        value += chunk
    if len(value) > limit or b"\n" not in value:
        raise ValueError("The application research message exceeded its limit.")
    return value.split(b"\n", 1)[0]


def _handle(request: Any, service: ApplicationResearchService) -> Dict[str, Any]:
    if not isinstance(request, dict):
        raise ValueError("The application research request is invalid.")
    if request.get("action") == "fetch_evidence":
        if set(request) - {"action", "url", "provenance"} or "url" not in request:
            raise ValueError("The application research fetch request is invalid.")
        url = request.get("url")
        if not isinstance(url, str) or not 9 <= len(url) <= 2000:
            raise ValueError("The application research URL is invalid.")
        provenance = FetchProvenance.from_dict(request.get("provenance"))
        return service.fetch_evidence(url, provenance)
    if set(request) != {"action", "draft", "source_urls"}:
        raise ValueError("The application research request is invalid.")
    if request.get("action") != "research_manifest":
        raise ValueError("The application research action is not allowed.")
    draft = request.get("draft")
    urls = request.get("source_urls")
    if not isinstance(draft, dict) or set(draft) != {"intent"}:
        raise ValueError("The application research draft is invalid.")
    if not isinstance(urls, list) or len(urls) > 8 or not all(
        isinstance(url, str) and len(url) <= 2000 for url in urls
    ):
        raise ValueError("The application research source list is invalid.")
    return service.research_manifest(draft, urls)


def main() -> None:
    SOCKET_PATH.parent.mkdir(parents=True, exist_ok=True)
    SOCKET_PATH.unlink(missing_ok=True)
    service = ApplicationResearchService()
    with socket.socket(socket.AF_UNIX, socket.SOCK_STREAM) as server:
        server.bind(str(SOCKET_PATH))
        os.chmod(SOCKET_PATH, 0o660)
        server.listen(8)
        while True:
            connection, _address = server.accept()
            with connection:
                try:
                    request = json.loads(_receive(connection, MAX_REQUEST).decode("utf-8"))
                    response = {"ok": True, "data": _handle(request, service)}
                except Exception as error:
                    response = {"ok": False, "error": str(error)[:500]}
                encoded = json.dumps(response, separators=(",", ":")).encode("utf-8")
                if len(encoded) > MAX_RESPONSE:
                    encoded = json.dumps({
                        "ok": False,
                        "error": "Application research produced too much evidence.",
                    }).encode("utf-8")
                connection.sendall(encoded + b"\n")


if __name__ == "__main__":
    main()
