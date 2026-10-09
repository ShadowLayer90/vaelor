"""Pull, remove, and inventory model weights on selected cluster nodes.

The weights-on-the-box layer beneath GPU serving: a cluster operator pastes a
Hugging Face link (or picks a curated catalog id) and this pulls the weights onto
the chosen node(s)' Hugging Face cache, records them in the ``model_cache``
inventory with per-node disk accounting, and can remove them again. It is *not*
the serve/deploy flow - `gpu_pool_operations` consumes a cache this has already
populated - which is why it lives in its own focused module rather than bloating
that one.

Two design rules carry through, both mirroring the pooled/GPU deploy paths:

* **Truthful, granular progress.** The pull is a backgrounded systemd oneshot
  (`GpuPoolRuntime.start_model_pull`) that runs the vLLM runtime container's own
  Python on the runtime's pull script, which writes a progress JSON; this
  polls that file plus the unit's state and translates each poll into a real
  ``report(percent, message)`` checkpoint. A node whose launcher has not reported
  a percent yet is reported by its honest phase ("starting", "fetching",
  "verifying"), never a fabricated number.
* **Crash-tolerant and boundary-legal.** Every remote command runs through the
  runtime's start/poll/remove pair, whose commands clear `SshTransport.run`'s
  allowlist on a worker and `bridge_argv_policy`'s shapes on the controller. A
  dropped SSH channel mid-pull does not kill the fetch (systemd supervises it);
  the next poll reconnects and reads the file.

VD-125: the nodes include the **head controller**. Its GPU is half of the only
two-node cluster the product ships into, and a model it cannot cache is a model
it cannot serve, so the controller is an ordinary target here exactly as it is
in `gpu_pool_operations` - its commands travel through the root hardware bridge
rather than over SSH, and `_transport` is the one place that choice is made.
"""

from __future__ import annotations

import time
from typing import Any, Callable, Dict, List, Mapping, Optional

from .cluster_placement import (
    CONTROLLER_PLACEMENT_ID,
    CONTROLLER_PLACEMENT_NAME,
)
from .gpu_model_catalog import GPU_MODEL_CATALOG
from .bridge_transport import default_bridge_transport
from .gpu_pool_pull import (
    PULL_INCOMPLETE_MESSAGE, PULL_TRANSPORT_FAILURES, classify_pull,
)
from .gpu_pool_runtime import GpuPoolRuntime
from .hf_model_source import resolve_model_source
from .ssh_transport import SshTransport, SshTransportError
from .vllm_images import download_line

#: The phases the launcher may report, in the order they occur. Used here only to
#: give an honest message when no percent has been written yet; whether the pull
#: is OVER is `GpuPoolRuntime.pull_verdict`'s call, which asks systemd first and
#: only then lets the phase corroborate a clean exit.
_PULL_PHASES = ("starting", "fetching", "verifying", "done", "error")


class GpuModelLibrary:
    """Orchestrate cache pull/remove and read the per-node cache inventory."""

    def __init__(
        self, *, store, broker, runtime: Optional[GpuPoolRuntime] = None,
        joined_node: Callable[..., Dict[str, Any]],
        advertise_address: Callable[[Any], str],
        poll_interval_seconds: float = 5.0,
        pull_timeout_seconds: float = 7200.0,
        sleep: Callable[[float], None] = time.sleep,
        clock: Callable[[], float] = time.monotonic,
        bridge_transport_factory: Optional[
            Callable[[Dict[str, Any]], Any]
        ] = None,
    ):
        self.store = store
        self.broker = broker
        self.runtime = runtime or GpuPoolRuntime()
        self.joined_node = joined_node
        self.advertise_address = advertise_address
        # How the CONTROLLER's commands leave this process (VD-125), injectable
        # for the same reason and in the same shape `GpuPoolOperations` uses: a
        # test can prove the controller got the bridge and every worker SSH.
        self._bridge_transport_factory = (
            bridge_transport_factory or default_bridge_transport
        )
        # Injectable so the poll loop can be unit-tested with scripted progress
        # and no real waiting, while defaulting to real time in production.
        self.poll_interval_seconds = poll_interval_seconds
        self.pull_timeout_seconds = pull_timeout_seconds
        self._sleep = sleep
        self._clock = clock

    def _transport(self, node: Dict[str, Any]):
        """How this node's commands are carried, by which node it is.

        The controller runs them through the root hardware bridge and every
        enrolled worker over pinned SSH, the same choice `GpuPoolOperations`
        makes and for the same reason: the controller is the Swarm manager, so
        it has no credential to log in with and no login to log in as. Both
        objects present the same ``run``/``profile`` surface, so the runtime's
        start/poll/remove trio never learns which one it holds.
        """
        if str(node.get("id", "")) == CONTROLLER_PLACEMENT_ID:
            return self._bridge_transport_factory(node)
        return SshTransport(
            self.broker.resolve(node["credential_id"], "cluster-node"),
            timeout=120,
        )

    @staticmethod
    def _node_ids(payload: Dict[str, Any]) -> List[str]:
        raw_ids = payload.get("node_ids", [])
        if not isinstance(raw_ids, list):
            raise ValueError("Choose the nodes to cache the model on.")
        node_ids = list(dict.fromkeys(
            str(node_id).strip() for node_id in raw_ids if str(node_id).strip()
        ))
        if not node_ids:
            raise ValueError("Choose at least one node to cache the model on.")
        # The controller is an ordinary cache target (VD-125). It has no SSH
        # credential, but its commands go through the root bridge instead, and
        # refusing it here would leave the machine that serves half the cluster
        # unable to hold the weights it is about to serve.
        return node_ids

    @staticmethod
    def _resolve(payload: Dict[str, Any]):
        """``(repo, revision)`` from a pasted link or a catalog id.

        A catalog id contributes its reviewed weights repo; a Hugging Face link
        contributes the parsed repo and revision. Unlike the deploy path, no
        transformer geometry is needed here - pulling weights does not size a
        model - so a link needs nothing beyond a valid ``org/name``.
        """
        source = resolve_model_source(
            payload.get("model_source", ""), GPU_MODEL_CATALOG.keys()
        )
        if source["kind"] == "catalog":
            return GPU_MODEL_CATALOG[source["id"]]["repo"], None
        return source["repo"], source["revision"]

    def pull(self, payload: Dict[str, Any], progress=None) -> Dict[str, Any]:
        """Cache ``model_source``'s weights onto the selected nodes, with progress.

        Starts a supervised pull on every chosen node, records each as
        ``pulling``, then polls all of them to completion - recording ``ready``
        with the bytes cached, or ``error`` with the reason. A node that fails
        leaves an ``error`` row, never an orphaned ``pulling`` one.
        """
        if payload.get("confirm") != "pull-cluster-model":
            raise ValueError("Confirm the reviewed cluster model pull.")
        repo, revision = self._resolve(payload)
        node_ids = self._node_ids(payload)
        nodes = [
            self.joined_node(node_id, include_credential=True)
            for node_id in node_ids
        ]
        report = progress or (lambda _percent, _message: None)
        transports = {node["id"]: self._transport(node) for node in nodes}

        started: Dict[str, str] = {}
        revision_key = revision or ""
        try:
            for index, node in enumerate(nodes):
                report(
                    5 + int(10 * index / len(nodes)),
                    f"Preparing {node['name']} to cache {repo}",
                )
                # The pull runs the runtime's container image, so the image has
                # to be on the node before the unit starts - made present here,
                # by name, never fetched by the pull's own `docker run`. It is
                # one the machine already has where it has one; a machine with
                # none downloads the default, and the job says so.
                transport = transports[node["id"]]
                image = self.runtime.pull_image_key(transport)
                self.runtime.ensure_image(
                    transport, image=image,
                    on_download=lambda profile, node=node, index=index: report(
                        5 + int(10 * index / len(nodes)),
                        download_line(profile, node["name"]),
                    ),
                )
                info = self.runtime.start_model_pull(
                    transport, repo=repo, revision=revision, image=image,
                )
                self.store.put_model_cache(
                    node_id=node["id"], repo=repo, revision=revision_key,
                    state="pulling", unit=info["unit"], message="Pull starting",
                )
                started[node["id"]] = info["unit"]
        except Exception:
            # A failure starting a later node must not orphan the nodes already
            # started in ``pulling`` with a running unit forever: stop each
            # started unit and reconcile its row to ``error`` before propagating,
            # mirroring the B1 deploy rollback.
            self._rollback_started(transports, started, repo, revision_key)
            raise
        return self._poll(nodes, transports, started, repo, revision_key, report)

    def _rollback_started(self, transports, started, repo, revision_key) -> None:
        for node_id, unit in started.items():
            try:
                self.runtime.stop_pull_unit(transports[node_id], unit)
            except (SshTransportError, ValueError):
                pass
            self.store.put_model_cache(
                node_id=node_id, repo=repo, revision=revision_key,
                state="error", unit=unit,
                message="The pull was rolled back after another node failed to start.",
            )

    def _poll(
        self, nodes, transports, started, repo, revision_key, report,
    ) -> Dict[str, Any]:
        by_id = {node["id"]: node for node in nodes}
        remaining = dict(started)
        results: Dict[str, Dict[str, Any]] = {}
        percents: Dict[str, Optional[int]] = {nid: None for nid in started}
        # Consecutive transport failures PER NODE. Per node, because one
        # unreachable box must not end the pull on its healthy siblings.
        failures: Dict[str, int] = {nid: 0 for nid in started}
        deadline = self._clock() + self.pull_timeout_seconds
        while remaining and self._clock() < deadline:
            for node_id, unit in list(remaining.items()):
                node = by_id[node_id]
                transport = transports[node_id]
                try:
                    state, status, progress_data = self.runtime.pull_verdict(
                        transport, unit, repo,
                    )
                except SshTransportError as error:
                    # A transient SSH failure is not a pull failure - systemd
                    # keeps fetching across a dropped channel - so a blip is
                    # absorbed and the counter resets on the next answer. But
                    # coercing EVERY failure to "keep waiting" (which an empty
                    # property dict does, since it classifies as ``pending``)
                    # held an unreachable node's row in `pulling` for the whole
                    # two-hour deadline. Bounded per node, exactly as the deploy
                    # pre-fetch bounds it; the other nodes poll on.
                    failures[node_id] += 1
                    if failures[node_id] < PULL_TRANSPORT_FAILURES:
                        continue
                    del remaining[node_id]
                    results[node_id] = self._unreachable_node(
                        transport, node, unit, repo, revision_key, error,
                    )
                    continue
                failures[node_id] = 0
                percents[node_id] = _progress_percent(progress_data)
                phase = _progress_phase(progress_data)
                report(
                    _overall_percent(percents),
                    f"{node['name']}: {_phase_message(phase, percents[node_id])}",
                )
                if state in {"pending", "running"}:
                    self.store.put_model_cache(
                        node_id=node_id, repo=repo, revision=revision_key,
                        state="pulling", unit=unit,
                        bytes_on_disk=int(progress_data.get("downloaded_bytes") or 0),
                        message=_phase_message(phase, percents[node_id]),
                    )
                    continue
                del remaining[node_id]
                results[node_id] = self._finish_node(
                    transport, node_id, unit, repo, revision_key,
                    status, progress_data, succeeded=state == "ok",
                )
                # A finished node no longer reports a live percent; count a
                # success as 100 so the aggregate band stays monotonic and can
                # reach the top rather than being dragged down by a stale entry.
                if results[node_id]["state"] == "ready":
                    percents[node_id] = 100
            if remaining:
                self._sleep(self.poll_interval_seconds)

        for node_id, unit in remaining.items():
            reason = "The pull did not finish before the deadline."
            # Stop the still-running unit too, so a timed-out pull leaves no
            # dangling oneshot (and no half-written .part files under a live
            # download) - the same cleanup a finished node gets.
            try:
                self.runtime.stop_pull_unit(transports[node_id], unit)
            except (SshTransportError, ValueError):
                pass
            self.store.put_model_cache(
                node_id=node_id, repo=repo, revision=revision_key,
                state="error", unit=unit, message=reason,
            )
            results[node_id] = {"node_id": node_id, "state": "error", "message": reason}

        report(100, f"Model cache pull finished for {repo}")
        ready = [r for r in results.values() if r["state"] == "ready"]
        return {
            "repo": repo,
            "revision": revision_key or None,
            "nodes": [results[node["id"]] for node in nodes],
            "ready_node_count": len(ready),
            "failed_node_count": len(results) - len(ready),
        }

    def _unreachable_node(
        self, transport, node, unit, repo, revision_key, error,
    ) -> Dict[str, Any]:
        """Record a node that stopped answering, and clean up after it.

        The row names the node and the last transport message, so the operator
        reads which box went away rather than a bare timeout. Stopping the unit
        is best-effort for the obvious reason: the node is not answering.
        """
        reason = "{} stopped answering while caching {}: {}".format(
            node["name"], repo, error
        )
        try:
            self.runtime.stop_pull_unit(transport, unit)
        except (SshTransportError, ValueError):
            pass
        self.store.put_model_cache(
            node_id=node["id"], repo=repo, revision=revision_key,
            state="error", unit=unit, message=reason,
        )
        return {"node_id": node["id"], "state": "error", "message": reason}

    def _finish_node(
        self, transport, node_id, unit, repo, revision_key, status, progress_data,
        *, succeeded: bool,
    ) -> Dict[str, Any]:
        # ``succeeded`` is `gpu_pool_runtime.pull_outcome`'s reading of the unit
        # and the progress file together: a oneshot that ran and exited
        # ``success`` pulled the weights, even if its progress file is
        # momentarily unreadable at the finish poll (an absent file says
        # nothing, and must never turn a successful pull into an error), while a
        # file that positively says the program was still fetching contradicts
        # the clean exit and is a failure.
        # The transient unit is cleaned up either way; the weights it fetched
        # stay in the cache. A cleanup failure must not mask the pull outcome.
        try:
            self.runtime.stop_pull_unit(transport, unit)
        except SshTransportError:
            pass
        if succeeded:
            reported = int(
                progress_data.get("total_bytes")
                or progress_data.get("downloaded_bytes")
                or 0
            )
            # Fall back to the last bytes recorded during polling when the final
            # progress read came back empty, rather than resetting a real size.
            bytes_on_disk = reported or self._last_bytes(node_id, repo, revision_key)
            self.store.put_model_cache(
                node_id=node_id, repo=repo, revision=revision_key,
                state="ready", bytes_on_disk=bytes_on_disk,
                message="Weights cached",
            )
            return {
                "node_id": node_id, "state": "ready",
                "bytes_on_disk": bytes_on_disk,
            }
        # A unit that failed carries the puller's own reason; a unit that exited
        # cleanly without recording completion has none to carry, and its
        # progress file's last line is a "fetching" note that would read as
        # progress rather than as the contradiction it is.
        if classify_pull(status) == "ok":
            reason = PULL_INCOMPLETE_MESSAGE
        else:
            reason = str(
                progress_data.get("message")
                or f"The pull unit did not report success (result: {status.get('Result', 'failed')})."
            )
        self.store.put_model_cache(
            node_id=node_id, repo=repo, revision=revision_key,
            state="error", message=reason,
        )
        return {"node_id": node_id, "state": "error", "message": reason}

    def _last_bytes(self, node_id: str, repo: str, revision_key: str) -> int:
        row = self.store.get_model_cache(node_id, repo, revision_key)
        return int(row.get("bytes_on_disk") or 0) if row else 0

    def remove(self, payload: Dict[str, Any]) -> Dict[str, Any]:
        """Delete a repo's cached weights from the selected nodes and its rows.

        Removal is **repo-granular**, because the on-disk reality is: the Hugging
        Face hub holds a repo's weights in one ``models--org--name`` directory
        that contains *every* revision, so ``rm -rf`` on a node removes them all
        at once. Deleting only the one pasted revision's row would leave a sibling
        revision reading ``ready`` while its weights are gone, so this drops every
        ``model_cache`` row for ``(node_id, repo)`` on the node it clears. A node
        the pull never reached is simply skipped.

        The pull units are stopped *before* the ``rm -rf``: a still-running pull
        writes into the very directory being cleared, so removing first would
        race a live downloader and leave it re-creating what was just deleted.
        """
        if payload.get("confirm") != "remove-cluster-model":
            raise ValueError("Confirm the reviewed cached-model removal.")
        repo, _revision = self._resolve(payload)
        node_ids = self._node_ids(payload)
        # ACC-110: weights a deployment still names are not removed - a paused
        # (scaled-to-zero) one loads again FROM them, so its next wake would
        # fail. Checked before anything is stopped or deleted on any node.
        users = []
        deployments = self._read_deployments()
        for node_id in node_ids:
            using = _deployments_using(deployments, repo, node_id)
            if using is None:
                raise ValueError(DEPLOYMENTS_UNREADABLE.format(repo=repo))
            users += [(node_id, user) for user in using]
        if users:
            raise ValueError(IN_USE_REFUSAL.format(
                repo=repo,
                users="; ".join(
                    "'{}' ({}) on {}".format(user["name"], user["state_label"], node_id)
                    for node_id, user in users
                ),
            ))
        removed: List[Dict[str, Any]] = []
        for node_id in node_ids:
            node = self.joined_node(node_id, include_credential=True)
            transport = self._transport(node)
            for unit in self._units_for(node_id, repo):
                try:
                    self.runtime.stop_pull_unit(transport, unit)
                except (SshTransportError, ValueError):
                    pass
            path = self.runtime.remove_model_cache(transport, repo)
            rows_removed = self.store.delete_model_cache_repo(node_id, repo)
            removed.append(
                {"node_id": node_id, "path": path, "rows_removed": rows_removed}
            )
        return {"repo": repo, "removed": removed}

    def _read_deployments(self) -> Optional[List[Mapping[str, Any]]]:
        """Every deployment row, read ONCE per listing or removal; ``None`` if unreadable."""
        try:
            return list(self.store.list_pooled_deployments())
        except Exception:  # noqa: BLE001 - not known is never "unused"
            return None

    def _units_for(self, node_id: str, repo: str) -> List[str]:
        """Every recorded pull unit for a repo on a node, across revisions.

        A pull unit is per-repo (its name is a slug of the repo), so the rows for
        different revisions carry the same unit name; this de-duplicates them so a
        stale unit from any revision's row is cleaned up when the repo is removed.
        """
        units = {
            str(row.get("unit", ""))
            for row in self.store.list_model_cache()
            if row["node_id"] == node_id and row["repo"] == repo and row.get("unit")
        }
        return sorted(units)

    def inventory(self) -> Dict[str, Any]:
        """The cache inventory joined with per-node disk accounting.

        Every enrolled node is listed (even with nothing cached) with its
        ``root_free_bytes`` from discovery and the summed ``bytes_on_disk`` of its
        ready models; each cache row is attached to its node. A row whose node is
        no longer enrolled is still reported, under its node id, so a stale cache
        is visible rather than silently dropped.

        The CONTROLLER is listed too (VD-125). It is not in ``list_nodes`` - it
        is the Swarm manager, not one of its members - but it caches and serves
        weights like any other GPU node, so leaving it out showed an operator a
        library missing the machine they had just pulled onto.
        """
        rows = self.store.list_model_cache()
        nodes = self.store.list_nodes()
        deployments = self._read_deployments()
        by_node: Dict[str, Dict[str, Any]] = {}
        controller = self._controller_entry()
        if controller is not None:
            by_node[CONTROLLER_PLACEMENT_ID] = controller
        for node in nodes:
            by_node[node["id"]] = {
                "node_id": node["id"],
                "name": node.get("name", node["id"]),
                "root_free_bytes": int(
                    (node.get("inventory") or {}).get("root_free_bytes") or 0
                ),
                "cached_bytes": 0,
                "models": [],
            }
        for row in rows:
            entry = by_node.get(row["node_id"])
            if entry is None:
                entry = {
                    "node_id": row["node_id"], "name": row["node_id"],
                    "root_free_bytes": 0, "cached_bytes": 0, "models": [],
                }
                by_node[row["node_id"]] = entry
            if row["state"] == "ready":
                entry["cached_bytes"] += int(row.get("bytes_on_disk") or 0)
            # ACC-110: which deployments still need these weights here, so the
            # list can say "In use" and keep Remove from breaking a wake.
            using = _deployments_using(deployments, row["repo"], row["node_id"])
            row["in_use_known"] = using is not None
            row["in_use_by"] = using or []
            entry["models"].append(row)
        return {"models": rows, "nodes": list(by_node.values())}

    def _controller_entry(self) -> Optional[Dict[str, Any]]:
        """The controller's inventory row, or ``None`` when it cannot be resolved.

        Resolved through the same ``joined_node`` every other target goes
        through, so its name and free space come from the one resolver rather
        than a second reading of the controller record. An uninitialised
        controller raises there; that is not an error for a listing, so it is
        simply absent - and any cache row it left behind still surfaces through
        the unenrolled-row path below.
        """
        try:
            node = self.joined_node(CONTROLLER_PLACEMENT_ID)
        except Exception:  # noqa: BLE001 - a listing never fails on one node
            return None
        return {
            "node_id": CONTROLLER_PLACEMENT_ID,
            "name": str(node.get("name") or CONTROLLER_PLACEMENT_NAME),
            "root_free_bytes": int(
                (node.get("inventory") or {}).get("root_free_bytes") or 0
            ),
            "cached_bytes": 0,
            "models": [],
        }


#: The deployment states that will read a node's cached weights again, and
#: how each is named in a refusal (ACC-110).
_HOLDING_STATES = {
    "healthy": "serving",
    "deploying": "loading",
    "unloaded": "paused, and it loads again from these weights",
}

#: Why cached weights were not removed: the deployments could not be read.
DEPLOYMENTS_UNREADABLE = (
    "The weights for {repo} were not removed: Vaelor could not read the "
    "cluster's deployments, so it cannot tell whether one still needs them. "
    "Try again in a moment."
)

#: Why cached weights were not removed: a deployment still needs them.
IN_USE_REFUSAL = (
    "The weights for {repo} were not removed: they are in use by {users}. "
    "Remove that deployment first, then remove the weights."
)


def _progress_percent(progress_data: Dict[str, Any]) -> Optional[int]:
    """The launcher's reported percent, clamped to 0-100, or ``None`` if absent.

    ``None`` is the honest "no number yet" that keeps a fabricated percent out of
    a checkpoint; a present-but-bad value is treated as absent rather than shown.
    """
    if "percent" not in progress_data:
        return None
    try:
        percent = int(progress_data["percent"])
    except (TypeError, ValueError):
        return None
    return max(0, min(100, percent))


def _progress_phase(progress_data: Dict[str, Any]) -> str:
    phase = str(progress_data.get("phase", "") or "")
    return phase if phase in _PULL_PHASES else "starting"


def _phase_message(phase: str, percent: Optional[int]) -> str:
    if percent is None:
        messages = {
            "starting": "starting the pull",
            "fetching": "fetching weights",
            "verifying": "verifying weights",
            "done": "finishing",
            "error": "reporting an error",
        }
        return messages.get(phase, messages["starting"])
    return f"fetching weights ({percent}%)"


def _overall_percent(percents: Dict[str, Optional[int]]) -> int:
    """A truthful aggregate percent for the job checkpoint (a 15-95 band).

    Averages only the nodes that have actually reported a percent, and maps that
    into a 15-95 band so the checkpoint never claims 0 while work is running nor
    100 before the poll records completion. Nodes with no report yet count as 0
    of their own progress, so the band cannot run ahead of the slowest launcher.
    """
    if not percents:
        return 15
    known = [value for value in percents.values() if value is not None]
    if not known:
        return 15
    average = sum(known) / len(percents)
    return 15 + int(80 * max(0.0, min(100.0, average)) / 100)


def _deployments_using(
    deployments: Optional[List[Mapping[str, Any]]], repo: str, node_id: str,
) -> Optional[List[Dict[str, str]]]:
    """The cluster deployments whose serving needs ``repo`` on ``node_id``.

    Every vLLM deployment that names the repo (``units.repo``, else its
    ``model_id``) and places a part on the node, in a state that will read
    the weights again: serving, loading, or paused (a paused deployment is
    the scale-to-zero rest state, woken on the next request). A failed or
    removed one does not hold them. ``None`` when the deployments could not
    be read: not known, and never "unused".
    """
    if deployments is None:
        return None
    users = []
    for record in deployments:
        units = record.get("units") or {}
        if units.get("engine") != "vllm":
            continue
        state = str(record.get("state", ""))
        if state not in _HOLDING_STATES:
            continue
        named = str(units.get("repo") or record.get("model_id") or "")
        if named == repo and node_id in list(record.get("node_ids") or []):
            users.append({"name": str(record.get("name", "")), "state": state,
                          "state_label": _HOLDING_STATES[state]})
    return users
