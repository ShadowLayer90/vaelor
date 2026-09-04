"""Fail-closed invocation broker for installed-app capabilities.

The broker is the last authorization boundary before an app transport is
called.  It intentionally accepts only server-owned IDs and resolves the
current registry, connection, grant, and agent facts on every invocation.
Credential values are never part of the transport request; an injected
transport is responsible for resolving the opaque connection reference in
its private credential boundary.
"""

from __future__ import annotations

import hashlib
import json
import time
from dataclasses import dataclass
from typing import Any, Callable, Mapping, Optional

from .agent_app_grants import AgentAppGrantError, AgentAppGrantStore
from .app_capability_manifest import AppCapabilityManifest, AppOperation
from .app_capability_registry import AppCapabilityRegistry
from .integration_connections import IntegrationConnectionStore


class CapabilityBrokerError(RuntimeError):
    """Base error for a blocked or failed capability invocation."""


class CapabilityAuthorizationError(CapabilityBrokerError):
    """Raised when a server-side pin or lifecycle fact is not valid."""

    def __init__(self, message: str, decision: Mapping[str, Any]):
        super().__init__(message)
        self.decision = dict(decision)


class CapabilityApprovalError(CapabilityAuthorizationError):
    """Raised when a write lacks a matching, current exact preview approval."""


class CapabilityTransportError(CapabilityBrokerError):
    """Raised when the authorized transport fails or returns unsafe output."""


class CapabilityIdempotencyError(CapabilityBrokerError):
    """Raised when an idempotency key is reused for a different request."""


@dataclass(frozen=True)
class CapabilityTransportRequest:
    """Safe transport input; no endpoint or credential value is exposed."""

    actor: str
    task_id: str
    grant_id: str
    agent_id: str
    agent_version: int
    app_instance_id: str
    manifest_digest: str
    connection_id: str
    operation_id: str
    input: Mapping[str, Any]
    timeout_seconds: int
    idempotency_key: Optional[str]


@dataclass(frozen=True)
class _Authorization:
    grant: Mapping[str, Any]
    app: Mapping[str, Any]
    manifest: AppCapabilityManifest
    operation: AppOperation
    connection: Mapping[str, Any]
    agent: Mapping[str, Any]


_MAX_INPUT_BYTES = 64 * 1024
_MAX_OUTPUT_BYTES = 64 * 1024
_MAX_IDEMPOTENCY_KEY = 160
_SENSITIVE_PARTS = {
    "api",
    "authorization",
    "credential",
    "endpoint",
    "password",
    "private",
    "secret",
    "token",
    "uri",
    "url",
}


def _canonical(value: Any) -> bytes:
    return json.dumps(
        value,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    ).encode("utf-8")


def _bounded_json(value: Any, field: str, maximum: int) -> Any:
    try:
        encoded = _canonical(value)
    except (TypeError, ValueError) as error:
        raise CapabilityBrokerError(f"{field} must contain bounded JSON data.") from error
    if len(encoded) > maximum:
        raise CapabilityBrokerError(f"{field} is too large.")
    return json.loads(encoded.decode("utf-8"))


def _sensitive_key(key: Any) -> bool:
    normalized = "".join(character.lower() if character.isalnum() else "_" for character in str(key))
    parts = {part for part in normalized.split("_") if part}
    return bool(parts & _SENSITIVE_PARTS)


def _reject_sensitive_input(value: Any, path: str = "input") -> None:
    if isinstance(value, Mapping):
        for key, child in value.items():
            if _sensitive_key(key):
                raise CapabilityBrokerError(f"{path} contains a credential-shaped field.")
            _reject_sensitive_input(child, f"{path}.{key}")
    elif isinstance(value, (list, tuple)):
        for index, child in enumerate(value):
            _reject_sensitive_input(child, f"{path}[{index}]")


def _safe_public(value: Any) -> Any:
    """Project transport output without returning secret-shaped fields."""
    if isinstance(value, Mapping):
        return {
            str(key): "[redacted]" if _sensitive_key(key) else _safe_public(child)
            for key, child in value.items()
        }
    if isinstance(value, (list, tuple)):
        return [_safe_public(child) for child in value]
    if value is None or isinstance(value, (str, bool, int, float)):
        return value
    raise CapabilityTransportError("The app returned unsupported result data.")


def _text(value: Any, field: str, maximum: int = 256) -> str:
    result = str(value or "").strip()
    if not result or len(result) > maximum:
        raise CapabilityBrokerError(f"{field} is invalid.")
    return result


class AppCapabilityBroker:
    """Resolve, revalidate, audit, and execute a capability invocation."""

    def __init__(
        self,
        registry: AppCapabilityRegistry,
        connections: IntegrationConnectionStore,
        grants: AgentAppGrantStore,
        *,
        agent_facts: Mapping[Any, Any] | Callable[[str, str, int], Optional[Mapping[str, Any]]],
        transport: Callable[[CapabilityTransportRequest], Any],
        clock: Callable[[], float] = time.time,
        refresh_state: Optional[Callable[[], Any]] = None,
    ) -> None:
        self.registry = registry
        self.connections = connections
        self.grants = grants
        self.agent_facts = agent_facts
        self.transport = transport
        self.clock = clock
        self.refresh_state = refresh_state

    def _agent(self, actor: str, agent_id: str, agent_version: int) -> Mapping[str, Any]:
        """Resolve one exact immutable agent revision within its actor boundary."""
        try:
            if callable(self.agent_facts):
                facts = self.agent_facts(actor, agent_id, agent_version)
            else:
                facts = self.agent_facts.get((actor, agent_id, agent_version))
                if not isinstance(facts, Mapping):
                    facts = None
                    by_actor = self.agent_facts.get(actor)
                    if isinstance(by_actor, Mapping):
                        by_agent = by_actor.get(agent_id)
                        if isinstance(by_agent, Mapping):
                            candidate = by_agent.get(agent_version, by_agent.get(str(agent_version)))
                            if isinstance(candidate, Mapping):
                                facts = candidate
                    if facts is None:
                        by_agent = self.agent_facts.get((actor, agent_id))
                        if isinstance(by_agent, Mapping):
                            candidate = by_agent.get(agent_version, by_agent.get(str(agent_version)))
                            if isinstance(candidate, Mapping):
                                facts = candidate
        except Exception as error:
            raise CapabilityAuthorizationError(
                "The agent facts are unavailable.",
                {"status": "blocked", "reasons": ["agent_facts_unavailable"]},
            ) from error
        if not isinstance(facts, Mapping):
            raise CapabilityAuthorizationError(
                "The pinned agent revision is unavailable.",
                {"status": "blocked", "reasons": ["agent_version_unavailable"]},
            )
        declared_actor = facts.get("actor", facts.get("owner", facts.get("actor_id")))
        if declared_actor is not None and str(declared_actor) != actor:
            raise CapabilityAuthorizationError(
                "The agent facts belong to a different actor.",
                {"status": "blocked", "reasons": ["agent_actor_mismatch"]},
            )
        current_id = str(facts.get("agent_id", agent_id))
        current_version = facts.get("agent_version", facts.get("version"))
        if current_id != agent_id or current_version is None:
            raise CapabilityAuthorizationError(
                "The pinned agent revision is unavailable.",
                {"status": "incompatible", "reasons": ["agent_version_unavailable"]},
            )
        try:
            version = int(current_version)
        except (TypeError, ValueError) as error:
            raise CapabilityAuthorizationError(
                "The pinned agent version is invalid.",
                {"status": "incompatible", "reasons": ["agent_version_invalid"]},
            ) from error
        if version != agent_version:
            raise CapabilityAuthorizationError(
                "The resolved agent revision does not match the pinned version.",
                {"status": "incompatible", "reasons": ["agent_version_stale"]},
            )
        if facts.get("active", facts.get("enabled", True)) is False:
            raise CapabilityAuthorizationError(
                "The agent version is unavailable.",
                {"status": "blocked", "reasons": ["agent_unavailable"]},
            )
        result = dict(facts)
        result["agent_id"] = current_id
        result["agent_version"] = version
        return result
    def _authorize(self, actor: str, grant_id: str, operation_id: str) -> _Authorization:
        actor = _text(actor, "actor", 120)
        grant_id = _text(grant_id, "grant_id", 160)
        operation_id = _text(operation_id, "operation_id", 128)
        if self.refresh_state is not None:
            try:
                self.refresh_state()
            except Exception as error:
                raise CapabilityAuthorizationError(
                    "Current managed-app state could not be verified.",
                    {"status": "blocked", "reasons": ["app_state_refresh_failed"]},
                ) from error
        grant = self.grants.get(grant_id, actor)
        if grant is None:
            raise CapabilityAuthorizationError(
                "The capability grant was not found.",
                {"status": "blocked", "reasons": ["grant_unavailable"]},
            )

        def deny(status: str, reason: str, message: str) -> None:
            raise CapabilityAuthorizationError(message, {"status": status, "reasons": [reason]})

        if grant.get("revoked") or grant.get("revoked_at") is not None:
            deny("revoked", "grant_revoked", "The capability grant is revoked.")
        if operation_id not in set(grant.get("operation_ids", ())):
            deny("blocked", "operation_not_granted", "The operation is not granted to this agent version.")

        app = self.registry.get_app_instance(str(grant["app_instance_id"]))
        if app is None:
            deny("blocked", "app_removed", "The installed app instance is unavailable.")
        if app["state"] == "incompatible":
            deny("incompatible", "app_incompatible", "The installed app is incompatible.")
        if app["state"] in {"stopped", "removed"}:
            deny("blocked", f"app_{app['state']}", f"The installed app is {app['state']}.")
        if app["state"] != "active":
            deny("blocked", "app_not_active", "The installed app is not active.")
        if app.get("health") != "healthy":
            deny("blocked", "app_unhealthy", "The installed app health is not healthy.")
        if app.get("compatibility") != "compatible":
            deny("incompatible", "app_incompatible", "The installed app is incompatible.")
        if app.get("manifest_digest") != grant.get("manifest_digest"):
            deny("incompatible", "manifest_digest_stale", "The grant manifest pin is stale.")
        if app.get("observed_manifest_digest", app.get("manifest_digest")) != grant.get("manifest_digest"):
            deny("incompatible", "manifest_digest_stale", "The installed manifest has changed.")

        manifest = self.registry.get_manifest(str(grant["manifest_digest"]))
        if manifest is None or manifest.manifest_digest != grant["manifest_digest"]:
            deny("incompatible", "manifest_unavailable", "The pinned manifest is unavailable.")
        operation = manifest.operation(operation_id)
        if operation is None:
            deny("incompatible", "operation_unlisted", "The operation is not listed by the pinned manifest.")

        connection = self.connections.get(str(grant["connection_id"]), actor)
        if connection is None:
            deny("blocked", "connection_unavailable", "The integration connection is unavailable.")
        if connection.get("revoked") or connection.get("revoked_at") is not None:
            deny("blocked", "connection_revoked", "The integration connection is revoked.")
        expires_at = connection.get("expires_at")
        if expires_at is not None and float(expires_at) <= self.clock():
            deny("blocked", "connection_expired", "The integration connection has expired.")
        if connection.get("test_status") != "healthy":
            deny("blocked", "connection_unhealthy", "The integration connection is not healthy.")
        if int(connection.get("secret_version", 0)) != int(grant.get("connection_secret_version", 0)):
            deny("incompatible", "connection_secret_stale", "The integration connection secret version changed.")

        agent = self._agent(actor, str(grant["agent_id"]), int(grant["agent_version"]))
        return _Authorization(grant, app, manifest, operation, connection, agent)

    @staticmethod
    def _preview(auth: _Authorization, operation_id: str, parameters: Mapping[str, Any], task_id: str) -> dict[str, Any]:
        payload = {
            "agent_id": auth.grant["agent_id"],
            "agent_version": int(auth.grant["agent_version"]),
            "app_instance_id": auth.grant["app_instance_id"],
            "manifest_digest": auth.grant["manifest_digest"],
            "connection_id": auth.grant["connection_id"],
            "connection_secret_version": int(auth.grant["connection_secret_version"]),
            "grant_id": auth.grant["id"],
            "operation_id": operation_id,
            "input": parameters,
            "task_id": str(task_id or "")[:160],
        }
        digest = hashlib.sha256(_canonical(payload)).hexdigest()
        return {
            "preview_version": 1,
            "preview_digest": digest,
            "grant_id": auth.grant["id"],
            "agent_id": auth.grant["agent_id"],
            "agent_version": int(auth.grant["agent_version"]),
            "app_instance_id": auth.grant["app_instance_id"],
            "manifest_digest": auth.grant["manifest_digest"],
            "connection_id": auth.grant["connection_id"],
            "connection_secret_version": int(auth.grant["connection_secret_version"]),
            "operation_id": operation_id,
            "mode": auth.operation.mode,
            "risk": auth.operation.risk,
            "input": parameters,
            "task_id": str(task_id or "")[:160],
        }

    def preview(
        self,
        actor: str,
        grant_id: str,
        operation_id: str,
        input: Optional[Mapping[str, Any]] = None,
        *,
        task_id: str = "",
    ) -> dict[str, Any]:
        """Build an exact, server-pinned write preview without transporting it."""
        parameters = _bounded_json(dict(input or {}), "input", _MAX_INPUT_BYTES)
        if not isinstance(parameters, dict):
            raise CapabilityBrokerError("input must be an object.")
        _reject_sensitive_input(parameters)
        auth = self._authorize(actor, grant_id, operation_id)
        return self._preview(auth, operation_id, parameters, task_id)

    def _audit(
        self,
        actor: str,
        grant_id: str,
        operation_id: str,
        *,
        outcome: str,
        approval: str,
        detail: str,
        duration_ms: int,
        task_id: str,
        request_digest: str = "",
        idempotency_key: Optional[str] = None,
    ) -> Optional[dict[str, Any]]:
        try:
            return self.grants.record_invocation(
                actor,
                grant_id,
                task_id=task_id,
                operation_id=operation_id if operation_id else "",
                approval=approval,
                duration_ms=max(0, min(int(duration_ms), 86_400_000)),
                outcome=outcome,
                result_provenance={"request_digest": request_digest} if request_digest else {},
                detail=detail[:256],
                idempotency_key=idempotency_key,
            )
        except (AgentAppGrantError, OSError):
            return None

    def invoke(
        self,
        actor: str,
        grant_id: str,
        operation_id: str,
        input: Optional[Mapping[str, Any]] = None,
        *,
        task_id: str = "",
        approval: Optional[Mapping[str, Any]] = None,
        idempotency_key: Optional[str] = None,
    ) -> dict[str, Any]:
        """Invoke once after fresh authorization; writes require exact approval."""
        started = self.clock()
        parameters = _bounded_json(dict(input or {}), "input", _MAX_INPUT_BYTES)
        if not isinstance(parameters, dict):
            raise CapabilityBrokerError("input must be an object.")
        _reject_sensitive_input(parameters)
        actor = _text(actor, "actor", 120)
        grant_id = _text(grant_id, "grant_id", 160)
        operation_id = _text(operation_id, "operation_id", 128)
        key = None if idempotency_key is None else _text(idempotency_key, "idempotency_key", _MAX_IDEMPOTENCY_KEY)
        auth: Optional[_Authorization] = None
        request_digest = ""
        try:
            auth = self._authorize(actor, grant_id, operation_id)
            preview = self._preview(auth, operation_id, parameters, task_id)
            request_digest = preview["preview_digest"]
            if auth.operation.mode == "write":
                expected = {
                    "preview_version": 1,
                    "preview_digest": preview["preview_digest"],
                    "grant_id": preview["grant_id"],
                    "agent_id": preview["agent_id"],
                    "agent_version": preview["agent_version"],
                    "app_instance_id": preview["app_instance_id"],
                    "manifest_digest": preview["manifest_digest"],
                    "connection_id": preview["connection_id"],
                    "connection_secret_version": preview["connection_secret_version"],
                    "operation_id": preview["operation_id"],
                    "input": preview["input"],
                    "task_id": preview["task_id"],
                }
                if not isinstance(approval, Mapping) or any(approval.get(field) != value for field, value in expected.items()):
                    decision = {"status": "blocked", "reasons": ["approval_missing_or_stale"], "preview": preview}
                    self._audit(actor, grant_id, operation_id, outcome="blocked", approval="rejected", detail="Write approval did not match the current exact preview.", duration_ms=int((self.clock() - started) * 1000), task_id=task_id, request_digest=request_digest, idempotency_key=key)
                    raise CapabilityApprovalError("The write approval is missing or stale.", decision)

            if key:
                existing = next(
                    (item for item in self.grants.audit(actor, limit=200)
                     if item.get("idempotency_key") == key),
                    None,
                )
                if existing is not None:
                    prior_digest = existing.get("result_provenance", {}).get("request_digest")
                    if prior_digest and prior_digest != request_digest:
                        raise CapabilityIdempotencyError(
                            "The idempotency key was already used for a different request."
                        )
                    return {"status": "replayed", "result": None, "preview": preview, "audit": existing}

            request = CapabilityTransportRequest(
                actor=actor,
                task_id=str(task_id or "")[:160],
                grant_id=grant_id,
                agent_id=str(auth.grant["agent_id"]),
                agent_version=int(auth.grant["agent_version"]),
                app_instance_id=str(auth.grant["app_instance_id"]),
                manifest_digest=str(auth.grant["manifest_digest"]),
                connection_id=str(auth.grant["connection_id"]),
                operation_id=operation_id,
                input=parameters,
                timeout_seconds=int(auth.operation.timeout_seconds),
                idempotency_key=key,
            )
            raw_result = self.transport(request)
            public_result = _safe_public(raw_result)
            public_result = _bounded_json(public_result, "transport result", _MAX_OUTPUT_BYTES)
            audit = self._audit(actor, grant_id, operation_id, outcome="succeeded", approval="approved" if auth.operation.mode == "write" else "not_required", detail="Capability transport completed.", duration_ms=int((self.clock() - started) * 1000), task_id=task_id, request_digest=request_digest, idempotency_key=key)
            return {"status": "succeeded", "result": public_result, "preview": preview, "audit": audit}
        except CapabilityApprovalError:
            raise
        except CapabilityAuthorizationError as error:
            grant = self.grants.get(grant_id, actor)
            if grant is not None:
                audit_operation = operation_id if operation_id in set(grant.get("operation_ids", ())) else ""
                self._audit(actor, grant_id, audit_operation, outcome="blocked", approval="not_required", detail=";".join(error.decision.get("reasons", ())), duration_ms=int((self.clock() - started) * 1000), task_id=task_id, request_digest=request_digest, idempotency_key=key)
            raise
        except CapabilityTransportError:
            if auth is not None:
                self._audit(actor, grant_id, operation_id, outcome="failed", approval="approved" if auth.operation.mode == "write" else "not_required", detail="Capability transport returned unsafe output.", duration_ms=int((self.clock() - started) * 1000), task_id=task_id, request_digest=request_digest, idempotency_key=key)
            raise
        except CapabilityBrokerError:
            raise
        except Exception as error:
            if auth is not None:
                self._audit(actor, grant_id, operation_id, outcome="failed", approval="approved" if auth.operation.mode == "write" else "not_required", detail="Capability transport failed.", duration_ms=int((self.clock() - started) * 1000), task_id=task_id, request_digest=request_digest, idempotency_key=key)
            raise CapabilityTransportError("The capability transport failed.") from error


CapabilityBroker = AppCapabilityBroker


__all__ = [
    "AppCapabilityBroker",
    "CapabilityApprovalError",
    "CapabilityAuthorizationError",
    "CapabilityBroker",
    "CapabilityBrokerError",
    "CapabilityIdempotencyError",
    "CapabilityTransportError",
    "CapabilityTransportRequest",
]
