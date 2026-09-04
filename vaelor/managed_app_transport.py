"""Server-owned HTTP transport for the curated Grafana capabilities.

This module is deliberately narrower than the generic capability broker.  It
accepts only a :class:`CapabilityTransportRequest`, resolves the managed app
and its current published port from the registry, and maps three built-in
operation IDs to fixed Grafana API requests.  No caller-controlled URL,
hostname, path, method, or header reaches the HTTP client.
"""

from __future__ import annotations

import json
import re
import urllib.error
import urllib.parse
import urllib.request
from collections.abc import Callable, Mapping
from typing import Any

from .app_capability_broker import CapabilityTransportError, CapabilityTransportRequest
from .app_capability_registry import AppCapabilityRegistry
from .credential_broker import CredentialBrokerClient
from .integration_connections import IntegrationConnectionStore
from .managed_app_capabilities import builtin_manifests


class ManagedAppTransportError(CapabilityTransportError):
    """Safe transport failure with no request, credential, or endpoint data."""


_GRAFANA_APP_ID = "grafana"
_GRAFANA_PURPOSE = "custom-agent-connector"
_GRAFANA_TARGET_PORT = 3000
_LOOPBACK_HOST = "127.0.0.1"
_MAX_TIMEOUT_SECONDS = 10
_MAX_RESPONSE_BYTES = 64 * 1024
_MAX_DASHBOARDS = 100
_MAX_TEXT = 1024
_MAX_TAGS = 16
_MAX_TAG_TEXT = 64
_MAX_UID = 64
_MAX_TIMESTAMP = 4_102_444_800_000  # 2100-01-01, milliseconds

_KNOWN_OPERATIONS = frozenset({"read_health", "read_dashboards", "write_annotation"})
_DASHBOARD_OUTPUT_KEYS = (
    "id", "uid", "title", "type", "folderUid", "folderTitle", "isStarred", "tags", "slug",
)
_HEALTH_OUTPUT_KEYS = ("database", "version", "commit", "message", "name", "type")
_ANNOTATION_OUTPUT_KEYS = ("id", "message", "startId", "endId", "time", "timeEnd", "tags")
_URL_TEXT = re.compile(r"https?://[^\s\"'<>]+", re.IGNORECASE)
_CREDENTIAL_REF_TEXT = re.compile(r"cred_[A-Za-z0-9_-]{8,120}")
_UID = re.compile(r"^[A-Za-z0-9_.-]{1,64}$")


class _NoRedirect(urllib.request.HTTPRedirectHandler):
    """Keep a Grafana response from turning the fixed URL into SSRF."""

    def redirect_request(self, request, file_pointer, code, message, headers, new_url):
        return None


def _default_http_opener() -> Callable[..., Any]:
    return urllib.request.build_opener(_NoRedirect())


def _safe_text(value: Any, maximum: int, *, field: str, allow_empty: bool = False) -> str:
    if not isinstance(value, str):
        raise ManagedAppTransportError(f"The Grafana {field} is invalid.")
    value = value.strip()
    if not value and not allow_empty:
        raise ManagedAppTransportError(f"The Grafana {field} is invalid.")
    if len(value) > maximum or any(ord(character) < 32 for character in value):
        raise ManagedAppTransportError(f"The Grafana {field} is invalid.")
    return value


def _safe_port(value: Any) -> int | None:
    if isinstance(value, bool):
        return None
    if isinstance(value, int):
        port = value
    elif isinstance(value, str) and value.strip().isdigit():
        port = int(value.strip())
    else:
        return None
    return port if 1 <= port <= 65535 else None


def _target_is_grafana(value: Any) -> bool:
    if value is None:
        return False
    normalized = str(value).strip().lower()
    return normalized in {"3000", "3000/tcp"}


def _port_from_entry(entry: Any, *, key_hint: Any = None) -> int | None:
    if not isinstance(entry, Mapping):
        return None
    target = next(
        (entry.get(key) for key in ("container", "container_port", "target", "target_port", "TargetPort", "ContainerPort") if key in entry),
        key_hint,
    )
    if target is not None and not _target_is_grafana(target):
        return None
    for key in ("host", "host_port", "published", "published_port", "HostPort", "PublishedPort"):
        if key in entry:
            return _safe_port(entry[key])
    return None


def _ports_from_value(value: Any, *, key_hint: Any = None) -> list[int]:
    if isinstance(value, Mapping):
        result: list[int] = []
        for key, child in value.items():
            if isinstance(child, list):
                result.extend(_ports_from_value(child, key_hint=key))
            else:
                port = _port_from_entry(child, key_hint=key)
                if port is not None:
                    result.append(port)
        return result
    if isinstance(value, list):
        result = []
        for child in value:
            if isinstance(child, Mapping):
                port = _port_from_entry(child, key_hint=key_hint)
                if port is not None:
                    result.append(port)
            elif key_hint in {"published_port", "published_host_port", "host_port"}:
                port = _safe_port(child)
                if port is not None:
                    result.append(port)
        return result
    if key_hint in {"published_port", "published_host_port", "host_port"}:
        port = _safe_port(value)
        return [] if port is None else [port]
    return []


def _published_grafana_port(health_evidence: Any) -> int:
    """Read only explicit reconciler port facts and fail closed when unclear."""
    if not isinstance(health_evidence, Mapping):
        raise ManagedAppTransportError("The managed app has no verified published port.")
    roots: list[Mapping[str, Any]] = [health_evidence]
    for key in ("runtime", "runtime_facts", "runtimeFacts"):
        child = health_evidence.get(key)
        if isinstance(child, Mapping):
            roots.append(child)

    candidates: list[int] = []
    for root in roots:
        for key in ("published_port", "published_host_port", "host_port"):
            if key in root:
                candidates.extend(_ports_from_value(root[key], key_hint=key))
        for key in ("published_ports", "ports", "port_facts"):
            if key in root:
                candidates.extend(_ports_from_value(root[key]))
    unique = sorted(set(candidates))
    if len(unique) != 1:
        raise ManagedAppTransportError("The managed app published port is not verified.")
    return unique[0]


def _canonical_json(value: Any) -> bytes:
    try:
        encoded = json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"), allow_nan=False).encode("utf-8")
    except (TypeError, ValueError) as error:
        raise ManagedAppTransportError("Grafana returned invalid JSON data.") from error
    if len(encoded) > _MAX_RESPONSE_BYTES:
        raise ManagedAppTransportError("Grafana returned too much data.")
    return encoded


def _redact_string(value: str, token: str) -> str:
    value = value.replace(token, "[redacted]") if token else value
    value = _CREDENTIAL_REF_TEXT.sub("[redacted]", value)
    return _URL_TEXT.sub("[redacted]", value)


def _public_value(value: Any, token: str, maximum: int = 512) -> Any:
    if isinstance(value, str):
        return _redact_string(value[:maximum], token)
    if isinstance(value, bool) or value is None:
        return value
    if isinstance(value, int) and not isinstance(value, bool):
        return value
    if isinstance(value, list):
        if len(value) > _MAX_TAGS:
            raise ManagedAppTransportError("Grafana returned too many values.")
        return [_public_value(item, token, _MAX_TAG_TEXT) for item in value]
    return None


def _project_object(payload: Mapping[str, Any], keys: tuple[str, ...], token: str) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key in keys:
        if key in payload:
            value = _public_value(payload[key], token)
            if value is not None or payload[key] is None:
                result[key] = value
    return result


class ManagedAppTransport:
    """Concrete, server-owned transport for the built-in Grafana manifest."""

    def __init__(
        self,
        registry: AppCapabilityRegistry,
        connections: IntegrationConnectionStore,
        credential_broker: CredentialBrokerClient,
        *,
        http_opener: Callable[..., Any] | None = None,
    ) -> None:
        self.registry = registry
        self.connections = connections
        self.credential_broker = credential_broker
        self.http_opener = http_opener or _default_http_opener()

    def __call__(self, request: CapabilityTransportRequest) -> dict[str, Any]:
        return self.invoke(request)

    def _authorize_target(self, request: CapabilityTransportRequest) -> tuple[int, str]:
        if not isinstance(request, CapabilityTransportRequest):
            raise ManagedAppTransportError("The capability transport request is invalid.")
        if request.operation_id not in _KNOWN_OPERATIONS:
            raise ManagedAppTransportError("The Grafana operation is not supported.")

        try:
            app = self.registry.get_app_instance(request.app_instance_id)
        except Exception as error:
            raise ManagedAppTransportError("The managed app is unavailable.") from error
        builtin = builtin_manifests()[_GRAFANA_APP_ID]
        if not isinstance(app, Mapping) or app.get("app_id") != _GRAFANA_APP_ID:
            raise ManagedAppTransportError("The managed app is not Grafana.")
        if request.manifest_digest != builtin.manifest_digest or app.get("manifest_digest") != builtin.manifest_digest:
            raise ManagedAppTransportError("The Grafana capability manifest is stale.")
        if app.get("state") != "active" or app.get("health") != "healthy" or app.get("compatibility") != "compatible":
            raise ManagedAppTransportError("The managed app is not healthy and active.")

        try:
            connection = self.connections.get(request.connection_id, request.actor)
        except Exception as error:
            raise ManagedAppTransportError("The Grafana connection is unavailable.") from error
        if not isinstance(connection, Mapping) or connection.get("actor") != request.actor:
            raise ManagedAppTransportError("The Grafana connection is unavailable.")
        if connection.get("provider") != _GRAFANA_APP_ID or connection.get("revoked") or connection.get("revoked_at") is not None:
            raise ManagedAppTransportError("The Grafana connection is unavailable.")
        if connection.get("test_status") != "healthy":
            raise ManagedAppTransportError("The Grafana connection is not healthy.")

        try:
            port = _published_grafana_port(app.get("health_evidence"))
        except ManagedAppTransportError:
            raise
        except Exception as error:
            raise ManagedAppTransportError("The managed app published port is not verified.") from error

        return port, self._connection_token(connection, request.actor)

    @staticmethod
    def _input(request: CapabilityTransportRequest) -> dict[str, Any]:
        if not isinstance(request.input, Mapping):
            raise ManagedAppTransportError("The Grafana operation input is invalid.")
        return dict(request.input)

    @staticmethod
    def _health_request(token: str, port: int) -> urllib.request.Request:
        return urllib.request.Request(
            f"http://{_LOOPBACK_HOST}:{port}/api/health",
            headers={
                "Accept": "application/json",
                "Authorization": f"Bearer {token}",
                "User-Agent": "Vaelor-Grafana-Capability/1",
            },
            method="GET",
        )

    def _connection_token(self, connection: Mapping[str, Any], actor: str) -> str:
        credential_ref = connection.get("credential_ref")
        if not isinstance(credential_ref, str) or not credential_ref:
            raise ManagedAppTransportError("The Grafana credential is unavailable.")
        try:
            lease = self.credential_broker.resolve(credential_ref, _GRAFANA_PURPOSE, actor)
        except Exception as error:
            raise ManagedAppTransportError("The Grafana credential is unavailable.") from error
        if (
            not isinstance(lease, Mapping)
            or lease.get("provider") != "application-secret"
            or lease.get("credential_id") != credential_ref
        ):
            raise ManagedAppTransportError("The Grafana credential is not a supported token lease.")
        token = lease.get("token")
        if (
            not isinstance(token, str)
            or not token
            or len(token) > 4096
            or any(character.isspace() or ord(character) < 32 for character in token)
        ):
            raise ManagedAppTransportError("The Grafana credential is not a supported token lease.")
        return token

    def _request_spec(self, request: CapabilityTransportRequest, token: str, port: int) -> urllib.request.Request:
        values = self._input(request)
        operation_id = request.operation_id
        if operation_id == "read_health":
            if values:
                raise ManagedAppTransportError("The Grafana health operation accepts no input.")
            return self._health_request(token, port)
        elif operation_id == "read_dashboards":
            if set(values) - {"limit"}:
                raise ManagedAppTransportError("The Grafana dashboard input contains unsupported fields.")
            limit = values.get("limit", _MAX_DASHBOARDS)
            if isinstance(limit, bool) or not isinstance(limit, int) or not 1 <= limit <= _MAX_DASHBOARDS:
                raise ManagedAppTransportError("The Grafana dashboard limit is invalid.")
            path = "/api/search"
            query = urllib.parse.urlencode((("type", "dash-db"), ("limit", str(limit))))
            body = None
            method = "GET"
        elif operation_id == "write_annotation":
            allowed = {"text", "tags", "time", "time_end", "dashboard_uid", "panel_id"}
            if set(values) - allowed or "text" not in values:
                raise ManagedAppTransportError("The Grafana annotation input is invalid.")
            body_value: dict[str, Any] = {"text": _safe_text(values["text"], _MAX_TEXT, field="annotation text")}
            if "tags" in values:
                tags = values["tags"]
                if not isinstance(tags, list) or len(tags) > _MAX_TAGS or any(not isinstance(tag, str) for tag in tags):
                    raise ManagedAppTransportError("The Grafana annotation tags are invalid.")
                body_value["tags"] = [_safe_text(tag, _MAX_TAG_TEXT, field="annotation tag") for tag in tags]
            if "dashboard_uid" in values:
                uid = _safe_text(values["dashboard_uid"], _MAX_UID, field="dashboard UID")
                if not _UID.fullmatch(uid):
                    raise ManagedAppTransportError("The Grafana dashboard UID is invalid.")
                body_value["dashboardUID"] = uid
            if "panel_id" in values:
                panel_id = values["panel_id"]
                if isinstance(panel_id, bool) or not isinstance(panel_id, int) or not 0 <= panel_id <= 2_147_483_647:
                    raise ManagedAppTransportError("The Grafana panel ID is invalid.")
                body_value["panelId"] = panel_id
            for source, target in (("time", "time"), ("time_end", "timeEnd")):
                if source in values:
                    timestamp = values[source]
                    if isinstance(timestamp, bool) or not isinstance(timestamp, int) or not 0 <= timestamp <= _MAX_TIMESTAMP:
                        raise ManagedAppTransportError("The Grafana annotation time is invalid.")
                    body_value[target] = timestamp
            if "time" in body_value and "timeEnd" in body_value and body_value["timeEnd"] < body_value["time"]:
                raise ManagedAppTransportError("The Grafana annotation time range is invalid.")
            path = "/api/annotations"
            query = ""
            body = json.dumps(body_value, ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode("utf-8")
            method = "POST"
        else:  # pragma: no cover - _authorize_target guards this set.
            raise ManagedAppTransportError("The Grafana operation is not supported.")

        url = f"http://{_LOOPBACK_HOST}:{port}{path}"
        if query:
            url += "?" + query
        headers = {
            "Accept": "application/json",
            "Authorization": f"Bearer {token}",
            "User-Agent": "Vaelor-Grafana-Capability/1",
        }
        if body is not None:
            headers["Content-Type"] = "application/json"
        return urllib.request.Request(url, data=body, headers=headers, method=method)

    def _open_json(self, http_request: urllib.request.Request, timeout_seconds: int) -> Any:
        timeout = max(1, min(int(timeout_seconds), _MAX_TIMEOUT_SECONDS))
        response = None
        try:
            response = self.http_opener(http_request, timeout=timeout)
            status = int(getattr(response, "status", getattr(response, "code", 200)))
            if status < 200 or status >= 300:
                raise ManagedAppTransportError("Grafana rejected the capability request.")
            raw = response.read(_MAX_RESPONSE_BYTES + 1)
            if not isinstance(raw, bytes) or len(raw) > _MAX_RESPONSE_BYTES:
                raise ManagedAppTransportError("Grafana returned too much data.")
            try:
                return json.loads(raw.decode("utf-8"))
            except (UnicodeDecodeError, json.JSONDecodeError) as error:
                raise ManagedAppTransportError("Grafana returned invalid JSON data.") from error
        except ManagedAppTransportError:
            raise
        except (urllib.error.URLError, TimeoutError, OSError, ValueError) as error:
            raise ManagedAppTransportError("The Grafana capability request failed.") from error
        except Exception as error:
            raise ManagedAppTransportError("The Grafana capability request failed.") from error
        finally:
            if response is not None:
                close = getattr(response, "close", None)
                if callable(close):
                    close()

    def _public_result(self, operation_id: str, payload: Any, token: str) -> Any:
        if operation_id == "read_dashboards":
            if not isinstance(payload, list) or len(payload) > _MAX_DASHBOARDS:
                raise ManagedAppTransportError("Grafana returned an invalid dashboard list.")
            result = []
            for item in payload:
                if not isinstance(item, Mapping):
                    raise ManagedAppTransportError("Grafana returned an invalid dashboard list.")
                result.append(_project_object(item, _DASHBOARD_OUTPUT_KEYS, token))
        elif operation_id == "read_health":
            if not isinstance(payload, Mapping):
                raise ManagedAppTransportError("Grafana returned an invalid health response.")
            result = _project_object(payload, _HEALTH_OUTPUT_KEYS, token)
        else:
            if not isinstance(payload, Mapping):
                raise ManagedAppTransportError("Grafana returned an invalid annotation response.")
            result = _project_object(payload, _ANNOTATION_OUTPUT_KEYS, token)
        _canonical_json(result)
        return result

    def test_connection(
        self,
        actor: str,
        app_instance_id: str,
        connection_id: str,
    ) -> dict[str, Any]:
        """Bootstrap one pending Grafana connection without weakening invoke."""
        try:
            app = self.registry.get_app_instance(app_instance_id)
        except Exception as error:
            raise ManagedAppTransportError("The managed app is unavailable.") from error
        builtin = builtin_manifests()[_GRAFANA_APP_ID]
        if not isinstance(app, Mapping) or app.get("app_id") != _GRAFANA_APP_ID:
            raise ManagedAppTransportError("The managed app is not Grafana.")
        if (
            app.get("manifest_digest") != builtin.manifest_digest
            or app.get("observed_manifest_digest", app.get("manifest_digest"))
            != builtin.manifest_digest
            or app.get("compatibility") != "compatible"
        ):
            raise ManagedAppTransportError("The managed app is incompatible.")

        evidence = app.get("health_evidence")
        runtime = evidence.get("runtime") if isinstance(evidence, Mapping) else None
        running = evidence.get("running") if isinstance(evidence, Mapping) else None
        if running is not True and isinstance(runtime, Mapping):
            running = runtime.get("running")
        if app.get("state") not in {"active", "degraded"} or running is not True:
            raise ManagedAppTransportError("The managed app is not running.")
        port = _published_grafana_port(evidence)

        try:
            connection = self.connections.get(connection_id, actor)
        except Exception as error:
            raise ManagedAppTransportError("The Grafana connection is unavailable.") from error
        if (
            not isinstance(connection, Mapping)
            or connection.get("actor") != actor
            or connection.get("provider") != _GRAFANA_APP_ID
            or connection.get("revoked")
            or connection.get("revoked_at") is not None
        ):
            raise ManagedAppTransportError("The Grafana connection is unavailable.")
        if connection.get("test_status") != "pending":
            raise ManagedAppTransportError("The Grafana connection is not pending.")
        token = self._connection_token(connection, actor)

        try:
            payload = self._open_json(
                self._health_request(token, port),
                _MAX_TIMEOUT_SECONDS,
            )
            if not isinstance(payload, Mapping):
                raise ManagedAppTransportError(
                    "Grafana returned an invalid health response."
                )
        except ManagedAppTransportError:
            return {
                "healthy": False,
                "detail": "Grafana health check failed.",
                "app_instance_id": app_instance_id,
            }
        healthy = str(payload.get("database", "")).strip().lower() == "ok"
        return {
            "healthy": healthy,
            "detail": (
                "Grafana health check passed."
                if healthy
                else "Grafana reported an unhealthy database."
            ),
            "app_instance_id": app_instance_id,
        }

    def invoke(self, request: CapabilityTransportRequest) -> dict[str, Any]:
        port, token = self._authorize_target(request)
        http_request = self._request_spec(request, token, port)
        payload = self._open_json(http_request, request.timeout_seconds)
        return self._public_result(request.operation_id, payload, token)


GrafanaManagedAppTransport = ManagedAppTransport


__all__ = ["GrafanaManagedAppTransport", "ManagedAppTransport", "ManagedAppTransportError"]
