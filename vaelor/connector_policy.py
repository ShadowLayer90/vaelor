"""Declarative policy validation for custom-agent REST connectors."""

from __future__ import annotations

import hashlib
import ipaddress
import json
import math
import re
from typing import Any, Dict
from urllib.parse import urlsplit


class ConnectorPolicyError(ValueError):
    pass


READ_METHODS = {"GET", "HEAD"}
WRITE_METHODS = {"POST", "PUT", "PATCH", "DELETE"}
ALLOWED_METHODS = READ_METHODS | WRITE_METHODS
AUTH_MODES = {"none", "bearer", "x-api-key"}
ID_PATTERN = re.compile(r"[a-z][a-z0-9_-]{1,47}")
PATH_PARAMETER = re.compile(r"\{([a-z][a-z0-9_]{0,47})\}")
SECRET_FIELD_MARKERS = ("password", "secret", "token", "api_key", "apikey", "authorization", "cookie")


def _public_origin(value: Any) -> str:
    text = str(value or "").strip().rstrip("/")
    try:
        parsed = urlsplit(text)
        port = parsed.port
        host = (parsed.hostname or "").encode("idna").decode("ascii").lower()
    except (UnicodeError, ValueError) as error:
        raise ConnectorPolicyError("Connector origin must be a valid public HTTPS origin.") from error
    if (
        parsed.scheme != "https" or not host or parsed.username or parsed.password
        or parsed.path not in ("", "/") or parsed.query or parsed.fragment
        or port not in (None, 443)
    ):
        raise ConnectorPolicyError("Connector origin must be a public HTTPS origin without a path.")
    try:
        address = ipaddress.ip_address(host)
    except ValueError:
        address = None
    if (
        host == "localhost" or host.endswith((".local", ".internal", ".lan"))
        or address is not None
    ):
        raise ConnectorPolicyError("Connector origins must use public DNS names, not local names or IPs.")
    return "https://" + host


def _schema(value: Any, *, response: bool = False, depth: int = 0) -> Dict[str, Any]:
    if depth > 3 or not isinstance(value, dict):
        raise ConnectorPolicyError("Connector schemas must be bounded JSON Schema objects.")
    allowed = {"type", "properties", "required", "additionalProperties", "items",
               "enum", "maxLength", "minimum", "maximum", "maxItems"}
    if set(value) - allowed:
        raise ConnectorPolicyError("Connector schemas use unsupported JSON Schema keywords.")
    kind = value.get("type")
    if kind not in {"object", "array", "string", "integer", "number", "boolean", "null"}:
        raise ConnectorPolicyError("Connector schemas require an explicit supported type.")
    result: Dict[str, Any] = {"type": kind}
    if kind == "object":
        properties = value.get("properties", {})
        if not isinstance(properties, dict) or len(properties) > 40:
            raise ConnectorPolicyError("Connector object schemas are limited to 40 properties.")
        clean_properties = {}
        for name, child in properties.items():
            if not re.fullmatch(r"[A-Za-z][A-Za-z0-9_.-]{0,63}", str(name)):
                raise ConnectorPolicyError("Connector schema property names are invalid.")
            if any(marker in str(name).lower() for marker in SECRET_FIELD_MARKERS):
                raise ConnectorPolicyError("Connector schemas cannot expose secret-shaped fields.")
            clean_properties[str(name)] = _schema(child, response=response, depth=depth + 1)
        required = value.get("required", [])
        if not isinstance(required, list) or any(item not in clean_properties for item in required):
            raise ConnectorPolicyError("Connector schema required fields must name declared properties.")
        result.update({
            "properties": clean_properties,
            "required": list(dict.fromkeys(map(str, required))),
            "additionalProperties": False,
        })
    elif kind == "array":
        result["items"] = _schema(value.get("items"), response=response, depth=depth + 1)
        result["maxItems"] = max(1, min(int(value.get("maxItems", 100)), 500 if response else 100))
    elif kind == "string":
        result["maxLength"] = max(1, min(int(value.get("maxLength", 1000)), 8000 if response else 2000))
        if "enum" in value:
            enum = value["enum"]
            if not isinstance(enum, list) or not 1 <= len(enum) <= 50:
                raise ConnectorPolicyError("Connector string enums are limited to 50 values.")
            result["enum"] = [str(item)[:result["maxLength"]] for item in enum]
    elif kind in {"integer", "number"}:
        bounds = {}
        for name in ("minimum", "maximum"):
            if name not in value:
                continue
            item = value[name]
            valid = (
                isinstance(item, int) and not isinstance(item, bool)
                if kind == "integer" else
                isinstance(item, (int, float)) and not isinstance(item, bool)
            )
            if not valid or not math.isfinite(float(item)):
                raise ConnectorPolicyError(
                    "Connector numeric schema bounds must be finite numbers of the declared type."
                )
            bounds[name] = item
        if (
            "minimum" in bounds and "maximum" in bounds
            and bounds["minimum"] > bounds["maximum"]
        ):
            raise ConnectorPolicyError(
                "Connector numeric schema minimum cannot exceed its maximum."
            )
        result.update(bounds)
    return result


def validate_connectors(value: Any) -> list[Dict[str, Any]]:
    if value is None:
        return []
    if not isinstance(value, list) or len(value) > 10:
        raise ConnectorPolicyError("An agent may have at most 10 REST connectors.")
    connectors = []
    ids = set()
    for raw in value:
        if not isinstance(raw, dict) or set(raw) - {
            "id", "name", "base_origin", "credential_ref", "auth", "operations"
        }:
            raise ConnectorPolicyError("A connector definition contains unsupported fields.")
        connector_id = str(raw.get("id", "")).strip().lower()
        if not ID_PATTERN.fullmatch(connector_id) or connector_id in ids:
            raise ConnectorPolicyError("Connector IDs must be unique safe identifiers.")
        ids.add(connector_id)
        name = str(raw.get("name", "")).strip()[:100]
        origin = _public_origin(raw.get("base_origin"))
        auth = str(raw.get("auth", "none")).strip().lower()
        credential_ref = str(raw.get("credential_ref", "")).strip()
        if auth not in AUTH_MODES:
            raise ConnectorPolicyError("Connector auth must be none, bearer, or x-api-key.")
        if auth == "none" and credential_ref:
            raise ConnectorPolicyError("Public connectors cannot retain a credential reference.")
        if auth != "none" and not re.fullmatch(r"cred_[a-f0-9]{24}", credential_ref):
            raise ConnectorPolicyError("Authenticated connectors require an encrypted credential reference.")
        operations = _operations(raw.get("operations", []))
        if not name or not operations:
            raise ConnectorPolicyError("Connector name and operations are required.")
        connectors.append({
            "id": connector_id, "name": name, "base_origin": origin,
            "credential_ref": credential_ref, "auth": auth,
            "operations": operations,
        })
    return connectors


def _operations(value: Any) -> list[Dict[str, Any]]:
    if not isinstance(value, list) or not 1 <= len(value) <= 20:
        raise ConnectorPolicyError("Each connector needs between 1 and 20 operations.")
    result = []
    ids = set()
    for raw in value:
        if not isinstance(raw, dict) or set(raw) - {
            "id", "description", "method", "path", "input_location",
            "request_schema", "response_schema", "timeout_seconds",
            "max_response_bytes", "rate_limit_per_minute", "approval"
        }:
            raise ConnectorPolicyError("A connector operation contains unsupported fields.")
        operation_id = str(raw.get("id", "")).strip().lower()
        method = str(raw.get("method", "")).strip().upper()
        path = str(raw.get("path", "")).strip()
        location = str(raw.get("input_location", "query" if method in READ_METHODS else "json"))
        if not ID_PATTERN.fullmatch(operation_id) or operation_id in ids:
            raise ConnectorPolicyError("Operation IDs must be unique safe identifiers.")
        ids.add(operation_id)
        if method not in ALLOWED_METHODS:
            raise ConnectorPolicyError("Connector methods are limited to GET, HEAD, POST, PUT, PATCH, and DELETE.")
        if (
            not path.startswith("/") or len(path) > 500 or "?" in path or "#" in path
            or ".." in path or "//" in path
        ):
            raise ConnectorPolicyError("Connector operation paths must be fixed relative paths.")
        if location not in {"query", "json"} or (method in READ_METHODS and location != "query"):
            raise ConnectorPolicyError("Read operations use query input; write operations may use JSON input.")
        request_schema = _schema(raw.get("request_schema", {
            "type": "object", "properties": {}, "required": [],
        }))
        missing = set(PATH_PARAMETER.findall(path)) - set(request_schema["properties"])
        if missing:
            raise ConnectorPolicyError("Every path placeholder must be declared in request_schema.")
        approval = str(raw.get("approval", "required" if method in WRITE_METHODS else "not_required"))
        if method in WRITE_METHODS and approval != "required":
            raise ConnectorPolicyError("State-changing connector operations currently require approval.")
        if method in READ_METHODS and approval != "not_required":
            raise ConnectorPolicyError("Read connector operations do not use mutation approval.")
        result.append({
            "id": operation_id,
            "description": str(raw.get("description", "")).strip()[:240],
            "method": method, "path": path, "input_location": location,
            "request_schema": request_schema,
            "response_schema": _schema(raw.get("response_schema", {"type": "object", "properties": {}}), response=True),
            "timeout_seconds": max(2, min(int(raw.get("timeout_seconds", 10)), 30)),
            "max_response_bytes": max(1024, min(int(raw.get("max_response_bytes", 65536)), 262144)),
            "rate_limit_per_minute": max(1, min(int(raw.get("rate_limit_per_minute", 10)), 60)),
            "approval": approval,
        })
    return result


def policy_fingerprint(connector: Dict[str, Any], operation: Dict[str, Any]) -> str:
    encoded = json.dumps(
        {"connector": connector, "operation": operation}, sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def validate_instance(value: Any, schema: Dict[str, Any], path: str = "input") -> Any:
    kind = schema["type"]
    if kind == "object":
        if not isinstance(value, dict):
            raise ConnectorPolicyError(f"{path} must be an object.")
        if set(value) - set(schema["properties"]):
            raise ConnectorPolicyError(f"{path} contains undeclared fields.")
        if any(name not in value for name in schema.get("required", [])):
            raise ConnectorPolicyError(f"{path} is missing required fields.")
        return {
            name: validate_instance(item, schema["properties"][name], f"{path}.{name}")
            for name, item in value.items()
        }
    if kind == "array":
        if not isinstance(value, list) or len(value) > schema["maxItems"]:
            raise ConnectorPolicyError(f"{path} must be a bounded array.")
        return [validate_instance(item, schema["items"], path) for item in value]
    if kind == "string":
        if not isinstance(value, str) or len(value) > schema["maxLength"]:
            raise ConnectorPolicyError(f"{path} must be a bounded string.")
        if "enum" in schema and value not in schema["enum"]:
            raise ConnectorPolicyError(f"{path} is outside the allowed values.")
        return value
    if kind == "integer" and (isinstance(value, bool) or not isinstance(value, int)):
        raise ConnectorPolicyError(f"{path} must be an integer.")
    if kind == "number" and (isinstance(value, bool) or not isinstance(value, (int, float))):
        raise ConnectorPolicyError(f"{path} must be a number.")
    if kind == "boolean" and not isinstance(value, bool):
        raise ConnectorPolicyError(f"{path} must be a boolean.")
    if kind == "null" and value is not None:
        raise ConnectorPolicyError(f"{path} must be null.")
    if kind in {"integer", "number"}:
        if "minimum" in schema and value < schema["minimum"]:
            raise ConnectorPolicyError(f"{path} is below the minimum.")
        if "maximum" in schema and value > schema["maximum"]:
            raise ConnectorPolicyError(f"{path} is above the maximum.")
    return value
