"""The one time grain every usage roll-up in Vaelor is kept at.

Three stores roll their counts up into time buckets so a window can be answered
without holding every request: the inference gateway's request log
(:mod:`vaelor.inference_metrics`), the LLM Server gate's per-key counts
(:mod:`vaelor.llm_gate_usage`) and the model's own token counters
(:mod:`vaelor.model_usage`). They share this grain so a screen that puts two of
them side by side - requests from the doors, tokens from the model - is adding
figures that cover the same span.

A bucket is named by its start, in UTC epoch seconds. A window "the last 24
hours" is answered from every bucket that STARTS at or after the bucket holding
``now - 24 h``, so it can reach back up to one bucket further than asked; the
answer therefore carries the real start (:func:`window_start`) and a screen
states it rather than implying an exact 24 hours (ACC-047).
"""

from __future__ import annotations

from .telemetry_store import DEFAULT_RETENTION_DAYS

#: Ten minutes: a 24-hour window is 144 buckets, and the largest overshoot a
#: window can carry is one bucket.
BUCKET_SECONDS = 600

#: How long bucket rows are kept. Lifetime totals are kept separately and never
#: pruned; the buckets only answer windows, and no screen asks for one longer
#: than this.
BUCKET_RETENTION_SECONDS = 90 * 86400


#: The second, finer grain: the Performance dashboard's per-replica serving
#: buckets (VD-147). Twenty seconds is two scrapes of the serving pollers, so a
#: bucket normally holds two ticks and one missed tick still leaves a reading.
#: ``BUCKET_SECONDS`` is a whole multiple of it, so a usage bucket and the
#: thirty performance buckets inside it cover exactly the same span.
PERFORMANCE_BUCKET_SECONDS = 20

#: How long the performance buckets are kept: the telemetry store's own period
#: (VD-089), from its one constant, so no chart can reach further back on the
#: serving panels than on the machine panels beside them.
PERFORMANCE_BUCKET_RETENTION_SECONDS = DEFAULT_RETENTION_DAYS * 86400

#: Above this many replicas a 7-day range reads too many twenty-second rows
#: (30,240 per replica); ranges of six hours and more then need a 60-second
#: roll-up of the replica buckets. The roll-up is not built yet: the dashboard
#: backend (slice S4) owns it and its query-time test at this many replicas.
REPLICAS_BEFORE_COARSER_GRAIN = 8


def performance_bucket_start(moment: float) -> int:
    """The start of the performance bucket holding ``moment`` (epoch seconds)."""
    return int(float(moment) // PERFORMANCE_BUCKET_SECONDS) * PERFORMANCE_BUCKET_SECONDS


def performance_prune_before(now: float) -> int:
    """The performance bucket start below which rows are past their retention."""
    return performance_bucket_start(float(now) - PERFORMANCE_BUCKET_RETENTION_SECONDS)


def bucket_start(moment: float) -> int:
    """The start of the bucket holding ``moment`` (epoch seconds)."""
    return int(float(moment) // BUCKET_SECONDS) * BUCKET_SECONDS


def window_start(now: float, window_seconds: float) -> int:
    """The start of the earliest bucket a window of ``window_seconds`` reads."""
    return bucket_start(float(now) - max(0.0, float(window_seconds)))


def prune_before(now: float) -> int:
    """The bucket start below which rows are older than the retention."""
    return bucket_start(float(now) - BUCKET_RETENTION_SECONDS)


#: Which door a recorded request came through, where one store keeps several:
#: the inference gateway's own requests, AI Chat's, and the LLM Server gate's.
#: Only the gateway's and the LLM Server's are traffic from outside Vaelor.
SOURCE_GATEWAY = "gateway"
SOURCE_AI_CHAT = "ai-chat"
SOURCE_LLM_SERVER = "llm-server"
#: A row recorded before its store kept a source: the door it came through is
#: not known, and it is never counted as any door's.
SOURCE_UNATTRIBUTED = "unattributed"


#: The lowest HTTP status counted as a FAILURE of the thing behind a door (a
#: server-side error, including the gateway's own "model unavailable"). The one
#: owner of the line ACC-056 draws: a request the model refused as malformed or
#: too long is the caller's error, not a failure of the model. The RED windows,
#: the request trace (`inference_metrics.is_failure`) and every door's roll-up
#: read it here.
FAILURE_STATUS = 500

#: The lowest status of a CLIENT error, counted apart from failures.
CLIENT_ERROR_STATUS = 400

#: The three classes :func:`status_class` answers with.
STATUS_FAILURE = "failure"
STATUS_CLIENT_ERROR = "client-error"
STATUS_OK = "ok"


def status_class(status: object) -> str:
    """:data:`STATUS_FAILURE`, :data:`STATUS_CLIENT_ERROR` or :data:`STATUS_OK`."""
    try:
        code = int(status or 0)
    except (TypeError, ValueError):
        code = 0
    if code >= FAILURE_STATUS:
        return STATUS_FAILURE
    if code >= CLIENT_ERROR_STATUS:
        return STATUS_CLIENT_ERROR
    return STATUS_OK
