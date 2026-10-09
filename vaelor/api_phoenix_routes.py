"""Routes for the Phoenix trace collector (VD-128, Phase E' observability).

Arize **Phoenix** captures the inference gateway's per-request OTLP traces. These
routes surface whether it is enabled/running and drive its lifecycle (enable,
disable). They mirror the LLM Server routes (:mod:`vaelor.api_llm_server_routes`)
and the same split:

**The control plane owns the persisted flag; the executor applies it.** A toggle
writes the desired ``{enabled}`` record (:mod:`vaelor.phoenix_state`) and enqueues
one ``phoenix.apply`` job. Only the workload executor can drive the root hardware
bridge that starts/stops the Phoenix container, so it is the account that brings
the collector up or down to match the state - and the 30 s reconcile
(:func:`vaelor.executor_service.launch_phoenix_autostart`) is the guarantee even
if the job service was down when the toggle was made.

Every verb is administrator-only: deploying a container and turning request
tracing on or off is a privileged decision. Enabling tracing points the gateway's
emitter at the loopback Phoenix endpoint; nothing keyed is exposed on the LAN.
"""

from __future__ import annotations

from flask import g, request

from .api_common import ApiContext, payload as _payload
from .phoenix_service import trace_collector_status
from .phoenix_state import PhoenixStore


def register_phoenix_routes(context: ApiContext) -> None:
    blueprint = context.blueprint
    callbacks = context.callbacks
    security = context.security
    require_auth = context.require_auth

    def _store() -> PhoenixStore:
        # A pre-wired store (a test's, over a tmp file) wins; production reads the
        # persisted record from the state root.
        existing = callbacks.get("phoenix_store")
        return existing if existing is not None else PhoenixStore()

    def _bridge():
        # The root bridge client, for the best-effort live "running" read. A test
        # injects a fake; production uses the shared client callback. Absent is
        # fine: the status builder only reads it when it is present.
        return callbacks.get("hardware_bridge_client")

    def _surface() -> dict:
        """The full Phoenix surface: persisted flag + best-effort live status."""
        return trace_collector_status(_store(), _bridge())

    def _enqueue_apply(action: str):
        """Enqueue one ``phoenix.apply`` job so the executor converges the container.

        Non-fatal if the job service is down: the flag is already persisted, so the
        30 s reconcile brings the collector to it. The response says whether a job
        was queued so the frontend can distinguish "applying now" from "will apply".
        """
        job_store = callbacks.get("job_store")
        if job_store is None:
            return None
        try:
            job = job_store.create(
                "phoenix.apply", g.auth_session.username, {"action": action}
            )
        except ValueError:
            return None
        return job.get("id")

    def _toggle_payload(job_id):
        """A toggle response that does not LIE about whether it took effect.

        The persisted flag is set here, but the container is only converged by the
        executor (the enqueued job, or the reconcile if the job service was down),
        so ``apply`` reports ``queued`` or ``pending`` rather than letting the bare
        ``enabled`` imply the container was already up/down.
        """
        return _payload({
            **_surface(),
            "job_id": job_id,
            "apply": "queued" if job_id else "pending",
        })

    @blueprint.get("/phoenix")
    @require_auth("administrator")
    def phoenix_status():
        return _payload(_surface())

    @blueprint.post("/phoenix/enable")
    @require_auth("administrator", csrf=True)
    def phoenix_enable():
        _store().enable()
        job_id = _enqueue_apply("enable")
        security.audit(
            g.auth_session.username, "phoenix.enable", "success",
            target="trace-collector", remote_addr=request.remote_addr or "",
            details={"job_id": job_id or ""},  # W5-D5: its job's evidence
        )
        return _toggle_payload(job_id)

    @blueprint.post("/phoenix/disable")
    @require_auth("administrator", csrf=True)
    def phoenix_disable():
        _store().disable()
        job_id = _enqueue_apply("disable")
        security.audit(
            g.auth_session.username, "phoenix.disable", "success",
            target="trace-collector", remote_addr=request.remote_addr or "",
            details={"job_id": job_id or ""},  # W5-D5: its job's evidence
        )
        return _toggle_payload(job_id)
