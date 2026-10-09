"""Persisted enable flag and endpoint derivation for the Phoenix trace collector.

Arize **Phoenix** is the OTLP trace collector the inference gateway ships spans to
(VD-128, Phase E'). It runs as one Vaelor-managed container bound to LOOPBACK on
the controller (:mod:`vaelor.phoenix_service`); nothing keyed is exposed on the
LAN, exactly as the model ports are not. This module is the single home of the
small persisted intent - ``{"enabled": bool}`` under the state root - plus the
pure derivation that turns it into the OTLP endpoint the emitter POSTs to.

Two processes touch the record, so it is hardened the same way the LLM Server's
is (:func:`vaelor.llm_server_state.harden_for_jobs_group`, reused rather than
re-spelled): the control plane writes it (the admin enable/disable route) and the
workload executor reads it (to start/stop the container and to bring it back
across a reboot via the ``phoenix.apply`` reconcile). Reads fail safe to disabled,
so a corrupt or absent record is "no tracing", never a half-on state.
"""

from __future__ import annotations

import json
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping

from .llm_server_state import harden_for_jobs_group
from .runtime_paths import state_path

#: The port Phoenix serves BOTH its UI and its OTLP/HTTP collector on (its default).
#: One constant, imported by the service and the surface, so the publish, the
#: endpoint and the UI hint cannot drift to different ports.
PHOENIX_PORT = 6006

#: Phoenix binds loopback on the controller: the appliance already fronts TLS and
#: auth, and an unkeyed trace UI must not be reachable on the LAN. The gateway
#: reaches the collector on loopback, and an operator reaches the UI through the
#: console, the same as every other internal port.
PHOENIX_LOOPBACK_HOST = "127.0.0.1"

#: Phoenix's OTLP/HTTP traces path. The emitter POSTs the OTLP JSON here.
OTLP_TRACES_PATH = "/v1/traces"

#: The persisted record's default location, under the Vaelor state root.
STATE_FILE = state_path("phoenix/state.json")


@dataclass(frozen=True)
class PhoenixSettings:
    """The trace collector's persisted intent: whether it is enabled.

    Immutable so a read cannot be mutated in place by a caller. No secret: Phoenix
    is loopback-only and unkeyed by design, so the record carries an enable flag
    and nothing else.
    """

    enabled: bool = False


def _coerce(raw: Any) -> PhoenixSettings:
    """A stored record turned into settings, failing safe to disabled."""
    if not isinstance(raw, Mapping):
        return PhoenixSettings()
    return PhoenixSettings(enabled=bool(raw.get("enabled")))


def trace_endpoint(settings: PhoenixSettings) -> str:
    """The OTLP endpoint the emitter should POST to for this state, or ``""``.

    A loopback URL only when enabled, so a disabled record makes the emitter a
    silent no-op (:meth:`vaelor.inference_tracing.TraceEmitter.endpoint`). The
    host is loopback because the gateway runs on the controller alongside Phoenix.
    """
    if not settings.enabled:
        return ""
    return "http://{}:{}{}".format(PHOENIX_LOOPBACK_HOST, PHOENIX_PORT, OTLP_TRACES_PATH)


class PhoenixStore:
    """Read and write the trace collector's persisted enable flag.

    Injectable ``path`` so tests drive it against a tmp file with no state root.
    Reads fail safe to disabled; writes are atomic (temp file then ``os.replace``)
    and hardened to the shared jobs group, exactly as the LLM Server store is, so
    the executor account can read what the control-plane account wrote.
    """

    def __init__(self, path: str = STATE_FILE):
        self._path = Path(path)

    def read(self) -> PhoenixSettings:
        try:
            raw = json.loads(self._path.read_text(encoding="utf-8"))
        except (OSError, ValueError):
            return PhoenixSettings()
        return _coerce(raw)

    def _write(self, settings: PhoenixSettings) -> PhoenixSettings:
        self._path.parent.mkdir(parents=True, exist_ok=True)
        payload = json.dumps({"enabled": settings.enabled}, separators=(",", ":"))
        tmp = self._path.with_name(self._path.name + ".tmp")
        tmp.write_text(payload + "\n", encoding="utf-8")
        harden_for_jobs_group(tmp)
        os.replace(tmp, self._path)
        harden_for_jobs_group(self._path)
        return settings

    def enable(self) -> PhoenixSettings:
        """Turn tracing on. Idempotent."""
        return self._write(PhoenixSettings(enabled=True))

    def disable(self) -> PhoenixSettings:
        """Turn tracing off. Idempotent."""
        return self._write(PhoenixSettings(enabled=False))
