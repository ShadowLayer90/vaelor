"""The cluster and history tool definitions of the Assistant's read registry.

Split out of `assistant_tools` when that module reached its line ceiling
(adversarial review should-fix 8). The definitions are unchanged in shape;
each reads the registry's callbacks when it runs.
"""

from __future__ import annotations

from typing import Any, Dict, Tuple

from .assistant_cluster_digest import CLUSTER_DIGEST_DESCRIPTION, cluster_digest
from .assistant_history_tools import metrics_history
from .assistant_serving_answers import SERVING_DESCRIPTION, SERVING_TOOL, serving_reading
from .performance_snapshot_source import assistant_performance_snapshot

_PERFORMANCE_DESCRIPTION = (
    "Read the cluster Performance snapshot: the speed verdict on the model's own time to "
    "first word and writing speed, request RED at every door - the inference gateway, the "
    "LLM Server and AI Chat, combined and per door (whole-request p95/p99 as information, "
    "error rate, traffic) - per-node USE, the live GPU serving gauges, the one-line "
    "why-it-is-slow, and the signals that are not collected."
)
_HISTORY_DESCRIPTION = (
    "Read retained telemetry so a trend can be distinguished from a single instant. Pass "
    "'limit' for the newest N samples by count (1-120), or 'window' (a duration like '24h' "
    "or '7d') for a downsampled trend over that span of time - use 'window' to answer 'over "
    "the last day/week'. Pass 'node' (a machine's name) to read that cluster machine's own "
    "history; with no window it is read over the last hour."
)


def cluster_and_history_tools(registry: Any, definition: Any, empty: Dict[str, Any]) -> Tuple[Any, ...]:
    """``cluster.summary``, ``cluster.digest``, ``system.performance``, ``serving.status`` and ``metrics.history``."""
    return (
        definition(
            "cluster.summary",
            "Read the head controller, enrolled workers, Swarm runtime, and pooled inference inventory.",
            "cluster:read", "read_only", 8,
            lambda _args: registry._call("cluster_summary"), empty,
        ),
        # VD-205 item 1: every machine in the cluster in plain words.
        definition(
            "cluster.digest", CLUSTER_DIGEST_DESCRIPTION, "cluster:read", "read_only", 12,
            lambda _args: cluster_digest(registry.callbacks), empty,
        ),
        # The "why is this slow?" snapshot from the ONE shared derivation the
        # Performance tab also reads, so the tab and the agent quote identical
        # numbers. A missing callback degrades to the honest not-collected state.
        definition(
            "system.performance", _PERFORMANCE_DESCRIPTION, "cluster:read", "read_only", 8,
            lambda _args: (registry.callbacks.get("performance_snapshot")
                           or (lambda: assistant_performance_snapshot(registry.callbacks)))(), empty,
        ),
        # VD-205 live check L1: what serves AI Chat and the LLM Server.
        definition(
            SERVING_TOOL, SERVING_DESCRIPTION, "workloads:read", "read_only", 12,
            lambda _args: serving_reading(registry.callbacks), empty,
        ),
        definition(
            "metrics.history", _HISTORY_DESCRIPTION, "system:read", "read_only", 8,
            lambda args: metrics_history(registry.callbacks, args),
            {
                "type": "object",
                "properties": {
                    "limit": {"type": "integer", "minimum": 1, "maximum": 120},
                    "window": {"type": "string", "minLength": 2, "maxLength": 16},
                    "node": {"type": "string", "minLength": 1, "maxLength": 64},
                },
                "additionalProperties": False,
            },
        ),
    )
