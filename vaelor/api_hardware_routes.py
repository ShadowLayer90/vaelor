"""Telemetry, identity, machine capability, system, and KVM routes.

Enclosure control lives in ``api_enclosure_routes.py``.
"""

from __future__ import annotations

import json
import time

from flask import Response, g, request, stream_with_context

from .console_capabilities import (
    console_ladder,
    firmware_out_of_band,
    unchecked_reachability,
)
from .copilot_setup import hardware_inventory
from .health_evaluation import evaluate_health
from .metrics_export import PROMETHEUS_CONTENT_TYPE, metrics_from_callbacks
from .api_common import ApiContext, payload as _payload
from .api_machine import machine_payload, platform_driver
from .platforms import PRODUCTS, observed_capabilities, plausible_products
from .device_identity import (
    appliance_identity, read_override, selected_product, write_override,
)


def register_hardware_routes(context: ApiContext) -> None:
    blueprint = context.blueprint
    callbacks = context.callbacks
    security = context.security
    limiter = context.limiter
    require_auth = context.require_auth
    appliance_address = context.appliance_address

    @blueprint.get("/device")
    @require_auth("viewer")
    def device():
        raw = callbacks["device_info"]() or {}
        read_config = callbacks.get("read_config")
        config = read_config() if read_config else {}
        system_config = config.get("system", {}) if isinstance(config, dict) else {}
        override = read_override() or system_config.get("device_variant_override")
        detection_raw = dict(raw)
        if override in PRODUCTS:
            detection_raw["id"] = override
            detection_raw["name"] = PRODUCTS[override]["name"]
        metrics = callbacks["current_data"]() or {}
        inventory_probe = callbacks.get("hardware_inventory")
        inventory = inventory_probe() if inventory_probe else hardware_inventory()
        platform_data = platform_driver(callbacks).snapshot(
            detection_raw, metrics, inventory
        )
        candidates = plausible_products(raw)
        manual_product = selected_product(override, PRODUCTS, candidates)
        if manual_product is not None:
            platform_data["product"] = manual_product
        # #185: this route used to write `__version__` here on its own, which
        # was right - and left `system.identity` reading the same callback's
        # raw `version`, the Pironman runtime's 1.3.18. One callback, two
        # readers, two answers. `appliance_identity` is the one decision.
        identity = appliance_identity(raw)
        safe = {
            "name": platform_data["product"]["name"] if override else raw.get("name") or platform_data["product"]["name"],
            "id": platform_data["product"]["id"],
            "version": identity["version"],
            "enclosure_runtime_version": identity["enclosure_runtime_version"],
            "peripherals": raw.get("peripherals", []),
            "platform": platform_data,
            # Only the enclosures this machine's own evidence is consistent
            # with. The list used to be every key of PRODUCTS, so the picker
            # offered a Pro Max on a Mini and the PATCH behind it accepted.
            "model_choices": [
                {
                    "id": product_id,
                    "name": PRODUCTS[product_id]["name"],
                    "icon": PRODUCTS[product_id]["icon"],
                }
                for product_id in candidates
                if product_id in PRODUCTS
            ],
        }
        return _payload(safe)

    @blueprint.patch("/device/model")
    @require_auth("operator", csrf=True)
    def choose_device_model():
        body = request.get_json(silent=True) or {}
        model = str(body.get("model", "")).strip()
        if model not in PRODUCTS:
            return _payload(
                error={"code": "invalid_device_model", "message": "Choose a supported Pironman model."},
                status=400,
            )
        # Validating only that the string is a known product let an operator
        # tell an x86 workstation it was a Pironman 5 Max, after which /device
        # reported OLED, RGB and case fans that do not exist. The choice is a
        # disambiguation between enclosures that discovery found plausible, not
        # a way to assert hardware into existence.
        #
        # That sentence was true on the workstation path, where the machine
        # class guard refused outright, and false on the appliance path, where
        # the guard passed and every key of PRODUCTS was accepted — so a
        # Pironman 5 Mini could be told it was a Pro Max. The appliance is what
        # ships, so the check is now the same on both: the enclosures this
        # machine's own evidence is consistent with, and nothing else.
        raw = callbacks["device_info"]() or {}
        candidates = plausible_products(raw)
        if not candidates:
            return _payload(
                error={
                    "code": "device_model_not_applicable",
                    "message": (
                        "No Pironman enclosure was detected on this machine, so "
                        "a Pironman model cannot be selected for it."
                    ),
                },
                status=409,
            )
        if model not in candidates:
            return _payload(
                error={
                    "code": "device_model_not_plausible",
                    "message": (
                        "Discovery on this machine is consistent with {}, not "
                        "with {}. It found {}, and Vaelor will not claim "
                        "hardware nothing reported.".format(
                            " or ".join(PRODUCTS[item]["name"] for item in candidates),
                            PRODUCTS[model]["name"],
                            ", ".join(sorted(observed_capabilities(raw))) or "no enclosure peripherals",
                        )
                    ),
                    "choices": list(candidates),
                },
                status=409,
            )
        try:
            write_override(model)
        except OSError:
            return _payload(
                error={"code": "device_selection_unavailable", "message": "Device selection is unavailable."},
                status=503,
            )
        security.audit(
            g.auth_session.username,
            "device.model.choose",
            "success",
            target=model,
            remote_addr=request.remote_addr or "",
        )
        return _payload({"accepted": True, "model": model})

    @blueprint.get("/system/machine")
    @require_auth("viewer")
    def system_machine():
        return _payload(machine_payload(callbacks))

    @blueprint.get("/live")
    def live():
        # Unauthenticated liveness for an anonymous load balancer or uptime
        # probe. `/health` stays auth-gated because it carries thermal readings
        # and a health band; this says only that the process is up and answering
        # and reveals nothing else — no version, telemetry, or identity — so it
        # is safe to expose without a session.
        return _payload({"status": "ok"})

    @blueprint.get("/health")
    @require_auth("viewer")
    def health():
        data = callbacks["current_data"]() or {}
        # 70/80 °C is a Raspberry Pi constant. A workstation processor boosts
        # into the mid-nineties by design, so reusing those numbers pinned it
        # in "Attention" permanently while idle.
        policy = platform_driver(callbacks).thermal_policy()
        return _payload({
            **evaluate_health(data, policy),
            "sampled_at": int(time.time() * 1000),
        })

    @blueprint.get("/telemetry/current")
    @require_auth("viewer")
    def telemetry_current():
        return _payload(
            {
                "sampled_at": int(time.time() * 1000),
                "metrics": callbacks["current_data"]() or {},
            }
        )

    @blueprint.get("/telemetry/retention")
    @require_auth("viewer")
    def telemetry_retention():
        # #186 / VD-089: the store applies 7 days while the configuration asks
        # 30, an override that was logged only at boot and shown nowhere. This
        # exposes the configured period beside the applied one, and the state,
        # so the owner can see which is in force.
        status = callbacks.get("telemetry_retention")
        if status is None:
            return _payload(
                error={
                    "code": "telemetry_retention_unavailable",
                    "message": "Telemetry retention status is unavailable.",
                },
                status=503,
            )
        return _payload(status())

    @blueprint.get("/stream/telemetry")
    @require_auth("viewer")
    def telemetry_stream():
        @stream_with_context
        def generate():
            while True:
                event = {
                    "sampled_at": int(time.time() * 1000),
                    "metrics": callbacks["current_data"]() or {},
                }
                yield "event: telemetry\ndata: {}\n\n".format(
                    json.dumps(event, separators=(",", ":"), default=str)
                )
                time.sleep(1)

        return Response(
            generate(),
            mimetype="text/event-stream",
            headers={
                "Cache-Control": "no-cache, no-transform",
                "X-Accel-Buffering": "no",
            },
        )

    # --- Prometheus/OpenMetrics export (P1.1) ---------------------------------
    # A scrape endpoint over the same `current_data` snapshot the System page and
    # `/telemetry/current` already read, rendered as Prometheus text exposition
    # (v0.0.4). Authenticated at "viewer" like the rest of the read API: a
    # scraper carries a session or token as any client does. (An UNauthenticated
    # liveness probe is a separate endpoint added elsewhere; this one stays
    # authenticated.) All formatting lives in `metrics_export`; this route only
    # wires the callbacks to it and sets the scrape Content-Type.
    @blueprint.get("/metrics")
    @require_auth("viewer")
    def prometheus_metrics():
        return Response(
            metrics_from_callbacks(callbacks),
            content_type=PROMETHEUS_CONTENT_TYPE,
        )

    @blueprint.get("/audit")
    @require_auth("operator")
    def audit():
        limit = request.args.get("limit", "50")
        try:
            parsed_limit = int(limit)
        except ValueError:
            parsed_limit = 50
        return _payload(security.list_audit(parsed_limit))

    @blueprint.get("/workloads/capabilities")
    @require_auth("viewer")
    def workload_capability_status():
        probe = callbacks.get("workload_capabilities")
        if probe is None:
            return _payload(
                {
                    "docker": {
                        "installed": False,
                        "compose": False,
                        "compose_version": None,
                    },
                    "job_types": [],
                }
            )
        return _payload(probe())

    @blueprint.get("/system/inventory")
    @require_auth("viewer")
    def system_inventory():
        inventory = callbacks.get("system_inventory")
        if inventory is None:
            return _payload(error={"code": "system_inventory_unavailable", "message": "System inventory is unavailable."}, status=503)
        return _payload(inventory.snapshot())

    @blueprint.get("/system/storage")
    @require_auth("viewer")
    def system_storage():
        inventory = callbacks.get("system_inventory")
        if inventory is None:
            return _payload(
                error={"code": "system_inventory_unavailable", "message": "Storage inventory is unavailable."},
                status=503,
            )
        return _payload(inventory.storage())

    @blueprint.get("/system/memory")
    @require_auth("viewer")
    def system_memory():
        inventory = callbacks.get("system_inventory")
        if inventory is None:
            return _payload(
                error={"code": "system_inventory_unavailable", "message": "Memory inventory is unavailable."},
                status=503,
            )
        return _payload(inventory.memory())

    @blueprint.get("/system/services/<service_id>/logs")
    @require_auth("operator")
    def system_service_logs(service_id):
        inventory = callbacks.get("system_inventory")
        try:
            return _payload(inventory.service_logs(service_id, request.args.get("lines", 200)))
        except (AttributeError, TypeError, ValueError) as error:
            return _payload(error={"code": "service_logs_unavailable", "message": str(error)}, status=400)

    @blueprint.post("/system/network/test")
    @require_auth("operator", csrf=True)
    def system_network_test():
        inventory = callbacks.get("system_inventory")
        result = inventory.connectivity()
        # Same defect class as the assistant turn recorded as SUCCESS: the
        # audit said "success" for the *act of probing*, so a probe that found
        # no DNS and no internet was indistinguishable in the trail from one
        # that found both. And `fixed-connectivity-probe` named nothing an
        # operator could act on - the row said an unexplained internal thing
        # had succeeded. Both halves of the reply now say what happened.
        reached = bool(result.get("dns")) and bool(result.get("internet"))
        security.audit(
            g.auth_session.username, "system.network.test",
            "success" if reached else "failure",
            target=", ".join(str(item) for item in result.get("targets", []))
            or "connectivity probe",
            remote_addr=request.remote_addr or "",
            details={
                "dns": result["dns"],
                "internet": result["internet"],
                "latency_ms": result.get("latency_ms"),
            },
        )
        return _payload(result)

    @blueprint.get("/kvm/capabilities")
    @require_auth("viewer")
    def kvm_capabilities():
        probe = callbacks.get("kvm_capabilities")
        control = callbacks.get("kvm_control")
        if probe is None or control is None:
            return _payload(error={"code": "kvm_unavailable", "message": "Remote console discovery is unavailable."}, status=503)
        result = probe.snapshot()
        control_status = control.status()
        result["control"] = control_status
        readiness = result.get("readiness")
        if not isinstance(readiness, dict):
            video = result.get("video") if isinstance(result.get("video"), dict) else {}
            hid = result.get("hid") if isinstance(result.get("hid"), dict) else {}
            ready = bool(result.get("console_ready"))
            readiness = {
                "state": "ready" if ready else "not_configured" if video.get("capture_detected") or hid.get("controller_available") else "unavailable",
                "ready": ready,
                "video_ready": bool(video.get("stream_ready", ready)),
                "input_ready": bool(hid.get("input_ready", ready)),
                "reason": "Protected video and isolated keyboard/mouse input are ready." if ready else "Physical KVM is not fully commissioned.",
            }
        result["readiness"] = readiness
        # Three states were collapsed into one, and they are not the same
        # state: advertised ≠ reachable ≠ provisioned. The ladder reports each
        # capability at the highest rung it has evidence for and names who can
        # move it up, which is what stops the interface implying Vaelor can
        # finish something only a person at the machine can start.
        firmware = callbacks.get("out_of_band_firmware")
        reachability = callbacks.get("out_of_band_reachability")
        result["ladder"] = list(
            console_ladder(
                result,
                platform_driver(callbacks).machine_class,
                firmware=firmware() if firmware else firmware_out_of_band(),
                # No off-box vantage exists yet, and this machine cannot
                # measure its own management engine, so the answer is unknown
                # rather than a confident negative.
                reachability=(
                    reachability() if reachability else unchecked_reachability()
                ),
            )
        )
        return _payload(result)

    @blueprint.get("/checkpoints")
    @require_auth("operator")
    def checkpoints():
        inventory = callbacks.get("checkpoints")
        if inventory is None:
            return _payload(error={"code": "checkpoints_unavailable", "message": "Recovery checkpoints are unavailable."}, status=503)
        return _payload(inventory.list(request.args.get("limit", 100)))

    @blueprint.post("/checkpoints/<checkpoint_id>/verify")
    @require_auth("operator", csrf=True)
    def checkpoint_verify(checkpoint_id):
        inventory = callbacks.get("checkpoints")
        if inventory is None:
            return _payload(
                error={
                    "code": "checkpoints_unavailable",
                    "message": "Recovery checkpoints are unavailable.",
                },
                status=503,
            )
        try:
            verifier = getattr(inventory, "verify", None) or getattr(inventory, "checksum", None)
            if verifier is None:
                raise AttributeError("Checkpoint verification is unavailable.")
            result = verifier(checkpoint_id)
        except (AttributeError, ValueError) as error:
            return _payload(error={"code": "checkpoint_verify_failed", "message": str(error)}, status=400)
        security.audit(g.auth_session.username, "checkpoint.verify", "success", target=checkpoint_id, remote_addr=request.remote_addr or "")
        return _payload(result)

    @blueprint.post("/checkpoints/<checkpoint_id>/restore")
    @require_auth("administrator", csrf=True)
    def checkpoint_restore(checkpoint_id):
        inventory = callbacks.get("checkpoints")
        job_store = callbacks.get("job_store")
        if inventory is None or job_store is None:
            return _payload(
                error={
                    "code": "checkpoint_restore_unavailable",
                    "message": "Checkpoint restore is unavailable.",
                },
                status=503,
            )
        body = request.get_json(silent=True) or {}
        project = str(body.get("project", "")).strip()
        digest = str(body.get("sha256", "")).strip().lower()
        confirmation = str(body.get("confirm", ""))
        try:
            if confirmation != project:
                raise ValueError("Type the project name to confirm restoration.")
            binding = inventory.bind_restore(checkpoint_id, project, digest)
            job = job_store.create(
                "checkpoint.restore",
                g.auth_session.username,
                {**binding, "confirm": confirmation},
            )
        except (AttributeError, ValueError) as error:
            return _payload(
                error={
                    "code": "checkpoint_restore_rejected",
                    "message": str(error),
                },
                status=400,
            )
        security.audit(
            g.auth_session.username,
            "checkpoint.restore.request",
            "success",
            target=checkpoint_id,
            remote_addr=request.remote_addr or "",
            details={
                "project": project,
                "sha256": digest,
                "manifest_digest": binding["manifest_digest"],
            },
        )
        return _payload(job, status=202)

    @blueprint.delete("/checkpoints/<checkpoint_id>")
    @require_auth("administrator", csrf=True)
    def checkpoint_delete(checkpoint_id):
        inventory = callbacks.get("checkpoints")
        body = request.get_json(silent=True) or {}
        try:
            result = inventory.delete(
                checkpoint_id, str(body.get("confirmation", ""))
            )
        except (AttributeError, OSError, ValueError) as error:
            return _payload(
                error={
                    "code": "checkpoint_delete_failed",
                    "message": str(error),
                },
                status=400,
            )
        security.audit(
            g.auth_session.username,
            "checkpoint.delete",
            "success",
            target=checkpoint_id,
            remote_addr=request.remote_addr or "",
        )
        return _payload(result)

    @blueprint.post("/kvm/control")
    @require_auth("operator", csrf=True)
    def kvm_control_acquire():
        control = callbacks.get("kvm_control")
        probe = callbacks.get("kvm_capabilities")
        # Gated on the same ladder the screen renders, so a row reporting
        # anything below `provisioned` and a grantable control cannot drift
        # apart. Only rung 4 may hold a session.
        blocked = [] if probe is None else [
            row for row in console_ladder(
                probe.snapshot(), platform_driver(callbacks).machine_class
            )
            if row["id"].startswith("console.") and not row["actionable"]
        ]
        if probe is None or blocked:
            return _payload(
                error={
                    "code": "kvm_not_commissioned",
                    "message": "; ".join(
                        "{}: {}".format(row["title"], row["detail"]) for row in blocked
                    ) or "Connect and commission HDMI capture and keyboard/mouse hardware first.",
                },
                status=409,
            )
        try:
            lease = control.acquire(g.auth_session.username)
        except (AttributeError, ValueError) as error:
            return _payload(error={"code": "kvm_control_busy", "message": str(error)}, status=409)
        security.audit(g.auth_session.username, "kvm.control.acquire", "success", target="keyboard-mouse", remote_addr=request.remote_addr or "")
        return _payload(lease)

    @blueprint.delete("/kvm/control")
    @require_auth("operator", csrf=True)
    def kvm_control_release():
        control = callbacks.get("kvm_control")
        try:
            lease = control.release(
                g.auth_session.username,
                force=g.auth_session.role == "administrator",
            )
        except (AttributeError, ValueError) as error:
            return _payload(error={"code": "kvm_control_release_rejected", "message": str(error)}, status=409)
        security.audit(g.auth_session.username, "kvm.control.release", "success", target="keyboard-mouse", remote_addr=request.remote_addr or "")
        return _payload(lease)
