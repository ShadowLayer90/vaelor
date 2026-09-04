"""Measured memory sizing for swarm LLM services (#138).

One function, called by both the approval plan (`cluster_plan_inference`) and
the deploy (`cluster_operations`), so the number the owner approves is the
number the service gets.
"""

from __future__ import annotations

from .cluster_driver import SWARM_CONTEXT_TOKENS
from .model_footprint import MIB, identify, normalise_platform, reservation_bytes


def swarm_llm_memory_limit_mib(
    name: str, repository: str, model_file: str, nodes: list
) -> int:
    """The memory limit for a swarm LLM service, from a measured footprint.

    #138: the limit used to be `model_size * 1.35 + 512 MB`, which gave the
    shipped 4B a 3,570 MiB cap against a measured cgroup steady state of
    3,632 MiB - a container limited below what the process demonstrably
    uses, restarted five times, then given up on. This sizes exactly as the
    single-node deploy does: the measured reservation for this model on this
    platform at the service's stated window (`--ctx-size`,
    SWARM_CONTEXT_TOKENS) with one slot, with the same margin and the same
    256 MiB rounding - for the shipped 4B that derives the verified 4,352
    MiB fixed point. An unmeasured combination is refused with the
    measurement it needs, because a limit nobody measured is the OOM kill
    arriving later.
    """
    from .copilot_setup import MEASURED_LIMIT_MARGIN

    model_id = identify(repository, model_file)
    if model_id is None:
        # The refusal carries the owner's next step on both branches: the
        # unmeasured-platform branch below inherits its instructions from the
        # reservation reason, and this one must not be the odd one out with a
        # bare fact and nowhere to go.
        raise ValueError(
            "Only catalog models with a measured memory footprint can be "
            "served to the cluster. {}/{} is not in the catalog: choose a "
            "catalog model, or have it added and measured on this hardware "
            "before serving it to the cluster.".format(repository, model_file)
        )
    required_memory = 0
    for node in nodes:
        platform = normalise_platform(
            (node.get("inventory") or {}).get("architecture")
        )
        reservation = reservation_bytes(
            model_id, platform or "", SWARM_CONTEXT_TOKENS, 1
        )
        if reservation["bytes"] is None:
            raise ValueError(
                "A cluster service limit cannot be derived for {} on "
                "{}: {}".format(name, node["name"], reservation["reason"])
            )
        required_memory = max(
            required_memory,
            int(int(reservation["bytes"]) * MEASURED_LIMIT_MARGIN),
        )
    return max(1024, ((required_memory + 255 * MIB) // (256 * MIB)) * 256)
