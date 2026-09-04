"""Standalone Vaelor control-plane server for portable installations."""

from __future__ import annotations

import logging
import threading
from pathlib import Path

from .model_profiles import calibrate_if_unmeasured
from .runtime_paths import env_value
from .wsgi_server import serve

LOGGER = logging.getLogger(__name__)


def measure_selected_model(agent=None) -> bool:
    """Measure the model this appliance is already using, once, at startup.

    Calibration only ever ran when someone selected, tested, or explicitly
    re-measured a model. An appliance whose model was chosen before that code
    shipped therefore had no profile and could never acquire one - which is how
    the appliance this was found on came to guess its structured-output contract
    on every request and size every budget from a default.

    Startup is the right moment because it is exactly when new code arrives: the
    upgrade that adds the measurement also restarts the service. It runs on its
    own thread, so a 161-second probe never delays the port opening, and it is
    never allowed to fail the boot - an appliance whose model server is down
    must still come up and serve everything else.
    """
    if agent is None:
        from .control_plane import __control_plane__

        agent = getattr(__control_plane__, "deployment_agent", None)
    if agent is None:
        return False
    try:
        connection = agent.connection("")
    except Exception:  # noqa: BLE001 - a probe must never stop the server booting
        LOGGER.info("Model calibration skipped: no usable model connection yet.")
        return False
    try:
        return calibrate_if_unmeasured(connection)
    except Exception:  # noqa: BLE001 - same
        LOGGER.exception("Model calibration could not be started.")
        return False


def main() -> None:
    # Composing the application does not require a Pironman hardware service;
    # unavailable platform adapters fail closed.
    from .control_plane import __app__, start_telemetry_retention

    host = env_value("VAELOR_HOST", "PM_DASHBOARD_HOST", "0.0.0.0")
    port = int(env_value("VAELOR_PORT", "PM_DASHBOARD_PORT", "34001"))
    if not 1 <= port <= 65535:
        raise RuntimeError("VAELOR_PORT must be between 1 and 65535.")
    certificate = env_value("VAELOR_TLS_CERT", "PM_TLS_CERT", "")
    private_key = env_value("VAELOR_TLS_KEY", "PM_TLS_KEY", "")
    ssl_context = None
    if certificate or private_key:
        if not (
            certificate
            and private_key
            and Path(certificate).is_file()
            and Path(private_key).is_file()
        ):
            raise RuntimeError(
                "VAELOR_TLS_CERT and VAELOR_TLS_KEY must both be readable."
            )
        ssl_context = (certificate, private_key)
    threading.Thread(
        target=measure_selected_model, name="vaelor-startup-calibration", daemon=True,
    ).start()
    # Telemetry retention starts here because this is the only entrypoint the
    # appliance actually runs. `ControlPlaneRuntime.run()` creates the database,
    # applies the retention policy and starts the data logger, and it is the
    # legacy Pironman entrypoint that no Vaelor process calls - so none of that
    # had ever happened and two weeks of telemetry were lost (VD-095 / #187).
    #
    # On its own thread, like the calibration above and for the same reason:
    # `Database.start()` sleeps two seconds and then waits up to ten more for
    # InfluxDB, and neither the port opening nor the boot may depend on that.
    # `start_telemetry_retention` swallows its own failures for the same reason;
    # the read path reports them as a reason instead.
    #
    # The *what* to start lives on the control plane, not here, so that this
    # module stays a process entrypoint. The other ten Vaelor units have their
    # own `main()` and never reach this line, which is what keeps exactly one
    # data logger writing.
    threading.Thread(
        target=start_telemetry_retention,
        name="vaelor-telemetry-retention",
        daemon=True,
    ).start()
    # Production WSGI serving with TLS terminated in front of it. NOT
    # werkzeug.serving.make_server: that is a development server and it discloses
    # exact framework/interpreter versions in its Server header (#205). See
    # vaelor/wsgi_server.py for why TLS is terminated in front rather than inside
    # waitress (measured: waitress-over-TLS raised SSLWantWriteError under load).
    serve(__app__, host, port, ssl_context=ssl_context, log=LOGGER)
