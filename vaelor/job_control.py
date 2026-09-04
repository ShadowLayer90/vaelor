"""Shared workload-executor control signals."""


class JobCancelled(RuntimeError):
    """Raised when a queued workload operation is cancelled safely."""
