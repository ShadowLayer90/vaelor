"""Drain and unenrol a cluster node whose architecture is not the controller's.

Per **VD-033** a mismatched node is not merely flagged. VD-031 refuses the
join and refuses deployments onto a node enrolled before that rule existed —
but the node stays enrolled and whatever was already scheduled on it keeps
running. Refusing new work while leaving old work running is the same lie
arriving by a different route, so the node itself goes.

The owner has confirmed no mixed cluster was ever built. That is why removal is
right rather than draining alone: there is no legitimate mixed cluster whose
workloads deserve preserving, and this is a safety net for a state that should
not exist.

**Two things this module must never do**, because it destroys state:

* Remove a node whose architecture could not be read. ``architecture_class()``
  answers ``""`` for anything unrecognised and ``""`` equals nothing, so an
  unreadable node can never fall into the mismatch branch — the classification
  in `cluster_architecture.eviction_disposition` names it separately and this
  module only ever acts on ``removable``.
* Remove a node because it did not answer. Reachability is not an input here.
  What decides is the architecture *discovered* over SSH and stored in the
  node's inventory; a node that is merely offline still carries the reading
  that was taken of it, and a node nothing was ever read from is unreadable,
  not mismatched.
"""

from __future__ import annotations

import time
from typing import Any, Callable, Dict, List

from .cluster_architecture import mismatched_nodes

#: Typed by the operator, not defaulted. This is the only Vaelor operation that
#: destroys an enrollment without the operator naming the node.
EVICTION_CONFIRMATION = "evict-mismatched-nodes"


class ArchitectureEviction:
    """Find, drain, unenrol, and report architecture-mismatched cluster nodes.

    Collaborators are passed in rather than constructed: ``remove_node`` is
    `ClusterOperations.remove_node`, which already drains the worker, asks it
    to leave Swarm, deletes its store record **and** deletes its stored SSH
    credential. Reusing it is deliberate — the enrollment refusal path deletes
    the credential of a host it turns away, and a host this appliance evicts
    must not leave a secret behind either.
    """

    def __init__(
        self,
        *,
        store,
        controller_architecture: Callable[[], str],
        remove_node: Callable[[Dict[str, Any]], Dict[str, Any]],
        clock: Callable[[], float] = time.time,
    ):
        self.store = store
        self._controller_architecture = controller_architecture
        self._remove_node = remove_node
        self._clock = clock

    def survey(self) -> Dict[str, Any]:
        """Which enrolled nodes are removable, changing nothing."""
        controller = self._controller_architecture()
        return {
            "controller_architecture": controller,
            "mismatched": mismatched_nodes(controller, self.store.list_nodes()),
        }

    def evict(self, payload: Dict[str, Any]) -> Dict[str, Any]:
        """Drain and unenrol every mismatched node, and report what was done."""
        if payload.get("confirm") != EVICTION_CONFIRMATION:
            raise ValueError(
                "Confirm draining and removing architecture-mismatched nodes."
            )
        controller = self._controller_architecture()
        enrolled = self.store.list_nodes()
        targets = mismatched_nodes(controller, enrolled)
        requested = str(payload.get("node_id", "")).strip()
        if requested:
            targets = [
                target for target in targets
                if target["node_id"] == requested
            ]
            if not targets:
                # Reached when the named node matches, when its architecture
                # could not be read, or when this controller could not read its
                # own. None of those is evidence, so none of them removes.
                raise ValueError(
                    "That node is not a confirmed architecture mismatch, so it "
                    "is not removed."
                )
        removed: List[Dict[str, Any]] = []
        failed: List[Dict[str, Any]] = []
        for target in targets:
            record = {
                "node_id": target["node_id"],
                "name": target["name"],
                "host": target["host"],
                "architecture": target["node_architecture"],
                "label": target["label"],
                "controller_architecture": target["controller_architecture"],
                "reason": target["reason"],
                "was_joined": target["joined"],
                "at": int(self._clock()),
            }
            try:
                outcome = self._remove_worker(target)
            except Exception as error:  # noqa: BLE001 - reported, not hidden
                failed.append({
                    **record,
                    "removed": False,
                    "drained": False,
                    "error": str(error)[:300],
                })
                continue
            removed.append({**record, **outcome})
        if removed:
            self.store.record_evictions(removed)
        return {
            "controller_architecture": controller,
            "checked": len(enrolled),
            "evicted": removed,
            "failed": failed,
        }

    def _remove_worker(self, target: Dict[str, Any]) -> Dict[str, Any]:
        """Remove one confirmed-mismatched node, cleanly if the node allows it.

        A worker that will not leave Swarm cleanly has already been drained by
        the time the failure surfaces, so stopping here would leave the node
        half-removed: no longer running work, still enrolled, still holding a
        credential. That is the half-truth VD-033 exists to end, so a confirmed
        mismatch that cannot leave cleanly is removed by force — and the
        escalation is reported rather than smoothed over.
        """
        forced_because = ""
        try:
            result = self._remove_node({
                "confirm": "remove-worker-node",
                "node_id": target["node_id"],
                "force": False,
            })
        except Exception as error:  # noqa: BLE001 - escalation is reported
            forced_because = str(error)[:300]
            result = self._remove_node({
                "confirm": "force-remove-worker-node",
                "node_id": target["node_id"],
                "force": True,
            })
        swarm_node_id = str(result.get("swarm_node_id", ""))
        return {
            "removed": bool(result.get("removed", False)),
            # An enrollment that never joined Swarm has no scheduling to stop.
            # Claiming a drain there would be a small lie in a report whose
            # whole purpose is not telling one.
            "drained": bool(swarm_node_id),
            "forced": bool(result.get("forced", False)) or bool(forced_because),
            "forced_because": forced_because,
            "swarm_node_id": swarm_node_id,
        }
