"""Deprecated embedded-dashboard API retained for one migration window."""

from .control_plane import VaelorControlPlane


class PMDashboard(VaelorControlPlane):
    def __init__(
        self,
        device_info=None,
        database="pm_dashboard",
        config=None,
        log=None,
        get_logger=None,
    ):
        super().__init__(
            device_info=device_info,
            database=database,
            config=config,
            log=log,
            get_logger=get_logger,
        )


__all__ = ["PMDashboard"]
