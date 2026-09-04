"""Typed, secret-free capability manifests for managed applications.

The manifest is the policy boundary shared by the app registry, grants, and
invocation code.  Its digest is computed from a canonical JSON projection;
display labels are deliberately not part of any authorization decision.
"""

from __future__ import annotations

import hashlib
import json
import math
import re
from dataclasses import dataclass, field
from types import MappingProxyType
from typing import Any, Mapping


class ManifestValidationError(ValueError):
    """Raised when an installed-app manifest is unsafe or structurally invalid."""


_ID_RE = re.compile(r"^[A-Za-z0-9][A-Za-z0-9_.-]{0,127}$")
_SCHEMA_TYPES = {"array", "boolean", "integer", "null", "number", "object", "string"}
_FORBIDDEN_KEY_PARTS = (
    "api_key", "access_key", "authorization", "credential", "endpoint",
    "password", "private_key", "secret", "token", "url", "uri",
)
_MAX_SCHEMA_DEPTH = 12
_MAX_JSON_BYTES = 512 * 1024


def _text(value: Any, field_name: str, *, maximum: int = 256) -> str:
    if not isinstance(value, str):
        raise ManifestValidationError(f"{field_name} must be text.")
    value = value.strip()
    if not value or len(value) > maximum:
        raise ManifestValidationError(f"{field_name} must be non-empty and bounded.")
    return value


def _identifier(value: Any, field_name: str) -> str:
    value = _text(value, field_name, maximum=128)
    if not _ID_RE.fullmatch(value):
        raise ManifestValidationError(f"{field_name} contains an invalid identifier.")
    return value


def _check_json(value: Any, path: str = "value", depth: int = 0) -> None:
    if depth > _MAX_SCHEMA_DEPTH:
        raise ManifestValidationError(f"{path} is nested too deeply.")
    if value is None or isinstance(value, (str, bool, int)):
        return
    if isinstance(value, float):
        if not math.isfinite(value):
            raise ManifestValidationError(f"{path} contains a non-finite number.")
        return
    if isinstance(value, Mapping):
        if len(value) > 128:
            raise ManifestValidationError(f"{path} has too many properties.")
        for key, child in value.items():
            if not isinstance(key, str) or len(key) > 128:
                raise ManifestValidationError(f"{path} has an invalid property name.")
            _check_json(child, f"{path}.{key}", depth + 1)
        return
    if isinstance(value, (list, tuple)):
        if len(value) > 256:
            raise ManifestValidationError(f"{path} has too many items.")
        for index, child in enumerate(value):
            _check_json(child, f"{path}[{index}]", depth + 1)
        return
    raise ManifestValidationError(f"{path} contains a value that is not JSON data.")


def _reject_sensitive_keys(value: Any, path: str = "manifest") -> None:
    if isinstance(value, Mapping):
        for key, child in value.items():
            normalized = re.sub(r"[^a-z0-9]", "_", str(key).lower()).strip("_")
            if any(part in normalized.split("_") for part in _FORBIDDEN_KEY_PARTS):
                raise ManifestValidationError(
                    f"{path}.{key} is not allowed in a capability manifest."
                )
            _reject_sensitive_keys(child, f"{path}.{key}")
    elif isinstance(value, (list, tuple)):
        for index, child in enumerate(value):
            _reject_sensitive_keys(child, f"{path}[{index}]")


def _schema(value: Any, field_name: str) -> dict[str, Any]:
    if not isinstance(value, Mapping):
        raise ManifestValidationError(f"{field_name} must be a JSON schema object.")
    result = dict(value)
    _check_json(result, field_name)
    schema_type = result.get("type")
    if schema_type is not None:
        allowed = set(schema_type) if isinstance(schema_type, list) else {schema_type}
        if not allowed or not allowed <= _SCHEMA_TYPES:
            raise ManifestValidationError(f"{field_name}.type contains an unsupported type.")
    properties = result.get("properties")
    if properties is not None and not isinstance(properties, Mapping):
        raise ManifestValidationError(f"{field_name}.properties must be an object.")
    return result


def _rate(value: Any, field_name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or not 1 <= value <= 100_000:
        raise ManifestValidationError(f"{field_name} must be an integer from 1 to 100000.")
    return value


def _approval(value: Any) -> str:
    value = _text(value, "approval_policy", maximum=32).lower()
    if value not in {"none", "operator", "always"}:
        raise ManifestValidationError("approval_policy must be none, operator, or always.")
    return value


@dataclass(frozen=True)
class AppOperation:
    """One explicitly grantable operation exposed by an installed app."""

    operation_id: str
    label: str
    mode: str = "read"
    input_schema: Mapping[str, Any] = field(default_factory=dict)
    output_schema: Mapping[str, Any] = field(default_factory=dict)
    risk: str = "low"
    approval_policy: str = "none"
    timeout_seconds: int = 30
    rate_limit_per_minute: int = 60
    requires_connection: bool = False

    def __post_init__(self) -> None:
        if isinstance(self, AppOperation):
            object.__setattr__(self, "operation_id", _identifier(self.operation_id, "operation_id"))
            object.__setattr__(self, "label", _text(self.label, "operation label"))
            mode = _text(self.mode, "operation mode", maximum=16).lower()
            if mode not in {"read", "write"}:
                raise ManifestValidationError("operation mode must be read or write.")
            object.__setattr__(self, "mode", mode)
            object.__setattr__(self, "input_schema", MappingProxyType(_schema(self.input_schema, "input_schema")))
            object.__setattr__(self, "output_schema", MappingProxyType(_schema(self.output_schema, "output_schema")))
            risk = _text(self.risk, "risk", maximum=32).lower()
            if risk not in {"low", "medium", "high", "critical"}:
                raise ManifestValidationError("risk must be low, medium, high, or critical.")
            object.__setattr__(self, "risk", risk)
            object.__setattr__(self, "approval_policy", _approval(self.approval_policy))
            if isinstance(self.timeout_seconds, bool) or not 1 <= self.timeout_seconds <= 3600:
                raise ManifestValidationError("timeout_seconds must be from 1 to 3600.")
            object.__setattr__(self, "rate_limit_per_minute", _rate(self.rate_limit_per_minute, "rate_limit_per_minute"))
            if not isinstance(self.requires_connection, bool):
                raise ManifestValidationError("requires_connection must be boolean.")

    @classmethod
    def from_dict(cls, value: Mapping[str, Any]) -> "AppOperation":
        if not isinstance(value, Mapping):
            raise ManifestValidationError("Each operation must be an object.")
        return cls(
            operation_id=value.get("operation_id", value.get("id")),
            label=value.get("label", value.get("name")),
            mode=value.get("mode", value.get("kind", "read")),
            input_schema=value.get("input_schema", value.get("input", {})),
            output_schema=value.get("output_schema", value.get("output", {})),
            risk=value.get("risk", "low"),
            approval_policy=value.get("approval_policy", value.get("approval", "none")),
            timeout_seconds=value.get("timeout_seconds", value.get("timeout", 30)),
            rate_limit_per_minute=value.get("rate_limit_per_minute", value.get("rate_limit", 60)),
            requires_connection=value.get("requires_connection", False),
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "operation_id": self.operation_id,
            "label": self.label,
            "mode": self.mode,
            "input_schema": dict(self.input_schema),
            "output_schema": dict(self.output_schema),
            "risk": self.risk,
            "approval_policy": self.approval_policy,
            "timeout_seconds": self.timeout_seconds,
            "rate_limit_per_minute": self.rate_limit_per_minute,
            "requires_connection": self.requires_connection,
        }


@dataclass(frozen=True)
class AppEvent:
    """One typed event that an installed app may emit."""

    event_id: str
    label: str
    payload_schema: Mapping[str, Any] = field(default_factory=dict)
    rate_limit_per_minute: int = 60

    def __post_init__(self) -> None:
        object.__setattr__(self, "event_id", _identifier(self.event_id, "event_id"))
        object.__setattr__(self, "label", _text(self.label, "event label"))
        object.__setattr__(self, "payload_schema", MappingProxyType(_schema(self.payload_schema, "payload_schema")))
        object.__setattr__(self, "rate_limit_per_minute", _rate(self.rate_limit_per_minute, "rate_limit_per_minute"))

    @classmethod
    def from_dict(cls, value: Mapping[str, Any]) -> "AppEvent":
        if not isinstance(value, Mapping):
            raise ManifestValidationError("Each event must be an object.")
        return cls(
            event_id=value.get("event_id", value.get("id")),
            label=value.get("label", value.get("name")),
            payload_schema=value.get("payload_schema", value.get("schema", {})),
            rate_limit_per_minute=value.get("rate_limit_per_minute", value.get("rate_limit", 60)),
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "event_id": self.event_id,
            "label": self.label,
            "payload_schema": dict(self.payload_schema),
            "rate_limit_per_minute": self.rate_limit_per_minute,
        }


@dataclass(frozen=True)
class AppCapabilityManifest:
    """Versioned declarative capabilities for one app identity."""

    app_id: str
    app_label: str
    app_version: str
    manifest_version: int = 1
    operations: tuple[AppOperation, ...] = field(default_factory=tuple)
    events: tuple[AppEvent, ...] = field(default_factory=tuple)
    health_probe: Mapping[str, Any] = field(default_factory=dict)
    requires_connection: bool = False
    metadata: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        object.__setattr__(self, "app_id", _identifier(self.app_id, "app_id"))
        object.__setattr__(self, "app_label", _text(self.app_label, "app_label"))
        object.__setattr__(self, "app_version", _text(self.app_version, "app_version", maximum=64))
        if isinstance(self.manifest_version, bool) or not isinstance(self.manifest_version, int) or self.manifest_version < 1:
            raise ManifestValidationError("manifest_version must be a positive integer.")
        operations = tuple(item if isinstance(item, AppOperation) else AppOperation.from_dict(item) for item in self.operations)
        events = tuple(item if isinstance(item, AppEvent) else AppEvent.from_dict(item) for item in self.events)
        if len({item.operation_id for item in operations}) != len(operations):
            raise ManifestValidationError("operation IDs must be unique within a manifest.")
        if len({item.event_id for item in events}) != len(events):
            raise ManifestValidationError("event IDs must be unique within a manifest.")
        health_probe = dict(self.health_probe)
        metadata = dict(self.metadata)
        _check_json(health_probe, "health_probe")
        _check_json(metadata, "metadata")
        _reject_sensitive_keys({"health_probe": health_probe, "metadata": metadata})
        if not isinstance(self.requires_connection, bool):
            raise ManifestValidationError("requires_connection must be boolean.")
        projection = {
            "app_id": self.app_id,
            "app_label": self.app_label,
            "app_version": self.app_version,
            "manifest_version": self.manifest_version,
            "operations": [item.to_dict() for item in operations],
            "events": [item.to_dict() for item in events],
            "health_probe": health_probe,
            "requires_connection": self.requires_connection,
            "metadata": metadata,
        }
        _reject_sensitive_keys(projection)
        encoded = _canonical_json(projection)
        if len(encoded) > _MAX_JSON_BYTES:
            raise ManifestValidationError("The capability manifest is too large.")
        object.__setattr__(self, "operations", operations)
        object.__setattr__(self, "events", events)
        object.__setattr__(self, "health_probe", MappingProxyType(health_probe))
        object.__setattr__(self, "metadata", MappingProxyType(metadata))

    @classmethod
    def from_dict(cls, value: Mapping[str, Any]) -> "AppCapabilityManifest":
        if not isinstance(value, Mapping):
            raise ManifestValidationError("The capability manifest must be an object.")
        _check_json(value, "manifest")
        _reject_sensitive_keys(value, "manifest")
        identity = value.get("identity", {})
        if not isinstance(identity, Mapping):
            raise ManifestValidationError("manifest.identity must be an object.")
        return cls(
            app_id=value.get("app_id", identity.get("id")),
            app_label=value.get("app_label", identity.get("label", identity.get("name"))),
            app_version=value.get("app_version", identity.get("version")),
            manifest_version=value.get("manifest_version", value.get("version", 1)),
            operations=tuple(value.get("operations", ())),
            events=tuple(value.get("events", ())),
            health_probe=value.get("health_probe", {}),
            requires_connection=value.get("requires_connection", False),
            metadata=value.get("metadata", {}),
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "app_id": self.app_id,
            "app_label": self.app_label,
            "app_version": self.app_version,
            "manifest_version": self.manifest_version,
            "operations": [item.to_dict() for item in self.operations],
            "events": [item.to_dict() for item in self.events],
            "health_probe": dict(self.health_probe),
            "requires_connection": self.requires_connection,
            "metadata": dict(self.metadata),
        }

    @property
    def manifest_digest(self) -> str:
        return hashlib.sha256(_canonical_json(self.to_dict())).hexdigest()

    def operation(self, operation_id: str) -> AppOperation | None:
        return next((item for item in self.operations if item.operation_id == operation_id), None)


def _canonical_json(value: Any) -> bytes:
    """Encode JSON deterministically for digesting and persistence."""
    return json.dumps(
        value, ensure_ascii=False, sort_keys=True, separators=(",", ":"),
        allow_nan=False,
    ).encode("utf-8")


def canonical_manifest_digest(value: AppCapabilityManifest | Mapping[str, Any]) -> str:
    """Return the SHA-256 digest of a validated canonical manifest."""
    return validate_manifest(value).manifest_digest

def validate_manifest(value: AppCapabilityManifest | Mapping[str, Any]) -> AppCapabilityManifest:
    """Normalize and validate an incoming manifest."""
    return value if isinstance(value, AppCapabilityManifest) else AppCapabilityManifest.from_dict(value)


__all__ = [
    "AppCapabilityManifest", "AppEvent", "AppOperation", "ManifestValidationError",
    "canonical_manifest_digest", "validate_manifest",
]
