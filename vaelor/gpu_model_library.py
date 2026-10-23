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
from .cluster_store import controller_recorded
from .gpu_model_catalog import GPU_MODEL_CATALOG
from .bridge_transport import default_bridge_transport
from .gpu_pool_pull import (
    DRAFT_PULL_LABEL, PULL_INCOMPLETE_MESSAGE, PULL_TRANSPORT_FAILURES, classify_pull,
)
from .gpu_pool_runtime import GpuPoolRuntime
from .hf_model_source import resolve_model_source
from .ssh_transport import SshTransport, SshTransportError
from .vllm_drafters import DRAFTERS, Drafter, drafters_for
from .vllm_images import download_line
from .vllm_serve_advanced import default_drafter

#: The phases the launcher may report, in the order they occur. Used here only to
#: give an honest message when no percent has been written yet; whether the pull
#: is OVER is `GpuPoolRuntime.pull_verdict`'s call, which asks systemd first and
#: only then lets the phase corroborate a clean exit.
_PULL_PHASES = ("starting", "fetching", "verifying", "done", "error")

#: What a cache row says between its pull unit starting and its first poll.
_PULL_STARTING = "Pull starting"
#: What a ready row says, whether the library's pull or a deploy's fetch wrote it.
_WEIGHTS_CACHED = "Weights cached"


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
        inventory_probe: Optional[Callable[[], Mapping[str, Any]]] = None,
    ):
        self.store = store
        # The controller's own hardware facts (its model store's free space),
        # for the listing only (:meth:`_controller_entry`); ``None`` reads them
        # with `copilot_setup.hardware_inventory`, as `ClusterOperations` does.
        self._inventory_probe = inventory_probe
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
        # VD-214: the draft model a default deploy of this model serves with,
        # fetched here too so a pre-downloaded model deploys without waiting
        # on it. ``None`` for a model with none: nothing else is fetched.
        drafter = default_drafter(repo)
        band = _bands(repo, drafter)
        images: Dict[str, str] = {}

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
                images[node["id"]] = image
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
                    state="pulling", unit=info["unit"], message=_PULL_STARTING,
                )
                started[node["id"]] = info["unit"]
        except Exception:
            # A failure starting a later node must not orphan the nodes already
            # started in ``pulling`` with a running unit forever: stop each
            # started unit and reconcile its row to ``error`` before propagating,
            # mirroring the B1 deploy rollback.
            self._rollback_started(transports, started, repo, revision_key)
            raise
        result = self._poll(
            nodes, transports, started, repo, revision_key, report, band[0],
        )
        result["drafter"] = None
        if drafter is not None:
            result["drafter"] = self._pull_drafter(
                nodes, transports, images, result, drafter, report, band[1],
            )
        report(100, f"Model cache pull finished for {repo}")
        return result

    def _pull_drafter(
        self, nodes, transports, images, result, drafter: Drafter, report, band,
    ) -> Dict[str, Any]:
        """Fetch ``drafter`` at its pinned commit onto the nodes that cached the model.

        The same runtime start/poll pair and the same poll loop as the model's
        own pull, in the image that pull ran in, so the drafter lands in the
        same hub cache the deploy's fetch (`gpu_pool_pull`) would put it in. A
        node whose model pull failed gets no drafter: it has nothing to draft
        for. Its rows are ordinary cache rows, so the library lists, sizes and
        removes it like any model.
        """
        ready = {entry["node_id"] for entry in result["nodes"] if entry["state"] == "ready"}
        targets = [node for node in nodes if node["id"] in ready]
        started: Dict[str, str] = {}
        try:
            for node in targets:
                info = self.runtime.start_model_pull(
                    transports[node["id"]], repo=drafter.repo,
                    revision=drafter.revision, image=images[node["id"]],
                )
                self.store.put_model_cache(
                    node_id=node["id"], repo=drafter.repo, revision=drafter.revision,
                    state="pulling", unit=info["unit"], message=_PULL_STARTING,
                )
                started[node["id"]] = info["unit"]
        except (SshTransportError, ValueError, RuntimeError) as error:
            # Adversarial review 1: a drafter that cannot START is the same
            # outcome as one whose download fails - the model is cached and
            # its draft model is not - so it is recorded the same way, as
            # error rows the job message reads, never as a failed job over
            # weights that are on disk.
            self._rollback_started(transports, started, drafter.repo, drafter.revision)
            reason = DRAFTER_START_FAILED.format(error=error)
            outcome = []
            for node in targets:
                if node["id"] not in started:
                    self.store.put_model_cache(
                        node_id=node["id"], repo=drafter.repo,
                        revision=drafter.revision, state="error", message=reason,
                    )
                outcome.append({"node_id": node["id"], "name": node["name"],
                                "state": "error", "message": reason})
            return dict(_drafter_summary(drafter), nodes=outcome,
                        ready_node_count=0, failed_node_count=len(outcome))
        label = DRAFT_PULL_LABEL.format(gb=round(drafter.weight_bytes / 1e9, 1))
        polled = self._poll(
            targets, transports, started, drafter.repo, drafter.revision, report,
            band, label=label,
        )
        polled.update(_drafter_summary(drafter))
        return polled

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
        self, nodes, transports, started, repo, revision_key, report, band,
        *, label: str = "",
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
                    _overall_percent(percents, band),
                    f"{node['name']}: {label}{_phase_message(phase, percents[node_id])}",
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

        for node in nodes:
            # The job message names machines, not ids (:func:`pull_job_message`).
            results[node["id"]]["name"] = node["name"]
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
                message=_WEIGHTS_CACHED,
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

        VD-214: each node's entry also says what became of the model's draft
        models (``drafters_removed``, ``drafters_kept`` with the reason) - see
        :meth:`_remove_drafters`.
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
            entry = {"node_id": node_id, "name": node.get("name", node_id),
                     "path": path, "rows_removed": rows_removed}
            entry.update(self._remove_drafters(transport, node_id, repo))
            removed.append(entry)
        return {"repo": repo, "removed": removed}

    def _remove_drafters(self, transport, node_id, repo) -> Dict[str, Any]:
        """Remove the draft models ``repo`` used on ``node_id`` that nothing there still needs.

        VD-214: the library's download fetched the drafter beside the model,
        and a deploy may have fetched it with no cache row at all, so it is
        removed by repo whether or not a row names it - a removal is ``rm -rf``
        of one hub directory, harmless when absent. It stays while another
        model on the machine it drafts for is cached, or while a deployment
        there serves with it (:func:`_drafter_kept_reason`).

        Adversarial review 1: the cache rows and the deployments are read
        HERE, after the model's own removal and just before the drafter's,
        not reused from :meth:`remove`'s check - a deploy of another model
        it drafts for may have started in between. Deployments that cannot
        be read keep the drafter: not known is never "unused".
        """
        done: Dict[str, List[Dict[str, Any]]] = {
            "drafters_removed": [], "drafters_kept": [],
        }
        drafters = drafters_for(repo)
        if not drafters:
            return done
        rows = list(self.store.list_model_cache())
        deployments = self._read_deployments()
        for drafter in drafters:
            users = _deployments_using(deployments, drafter.repo, node_id)
            reason = DRAFTER_KEPT_UNKNOWN if users is None else _drafter_kept_reason(
                drafter, repo, node_id, rows, users,
            )
            if reason:
                done["drafters_kept"].append({"repo": drafter.repo, "reason": reason})
                continue
            for unit in self._units_for(node_id, drafter.repo):
                try:
                    self.runtime.stop_pull_unit(transport, unit)
                except (SshTransportError, ValueError):
                    pass
            done["drafters_removed"].append({
                "repo": drafter.repo,
                "path": self.runtime.remove_model_cache(transport, drafter.repo),
                "rows_removed": self.store.delete_model_cache_repo(node_id, drafter.repo),
            })
        return done

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
        rows = list(self.store.list_model_cache())
        nodes = self.store.list_nodes()
        deployments = self._read_deployments()
        # VD-133 1.5.5 #9: weights a deployment serves from are on that machine
        # whether or not a library pull recorded them (a deploy before 1.5.5
        # recorded none, and a cold install keeps the weights but not the rows).
        rows += self._record_held(_held_by_deployments(deployments, rows))
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
                # Adversarial L3: 0 bytes is no measurement - a fetch that
                # reported none, or a row recorded from a deployment.
                if int(row.get("bytes_on_disk") or 0) <= 0:
                    row.update(size_known=False, message=HELD_NOT_RECORDED)
            if row.get("size_known") is False:
                # The node's total leaves these out, and says how many it left.
                entry["sizes_unrecorded"] = entry.get("sizes_unrecorded", 0) + 1
            # ACC-110: which deployments still need these weights here, so the
            # list can say "In use" and keep Remove from breaking a wake.
            using = _deployments_using(deployments, row["repo"], row["node_id"])
            row["in_use_known"] = using is not None
            row["in_use_by"] = using or []
            # VD-214: a draft model's row says which models it drafts for, so
            # the library does not list 1.4 GB of unexplained weights.
            drafts_for = [
                list(item.targets) for item in DRAFTERS.values() if item.repo == row["repo"]
            ]
            if drafts_for:
                row["draft_model_for"] = drafts_for[0]
            entry["models"].append(row)
        return {"models": rows, "nodes": list(by_node.values())}

    def _record_held(self, held: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Store each held-but-unrecorded row the first time it is found (adversarial M1).

        So they stay listed and removable after that deployment goes. A store
        that cannot be written lists them from the deployment, this once.
        """
        for row in held:
            try:
                self.store.put_model_cache(node_id=row["node_id"], repo=row["repo"],
                                           revision="", state="ready", bytes_on_disk=0,
                                           message=HELD_NOT_RECORDED)
            except Exception:  # noqa: BLE001 - listed from the deployment instead
                pass
        return held

    def _controller_entry(self) -> Optional[Dict[str, Any]]:
        """The controller's inventory row, or ``None`` when it is not set up as one.

        VD-133 1.5.5 #9 (LESSONS 6, 14): from the controller RECORD
        (`controller_recorded`) and the hardware inventory, never the deploy's
        resolver - it asks Docker, which the control plane serving this listing
        cannot (task #153), so the controller was never listed. Unreadable free
        space is 0, as an enrolled node's missing figure is.
        """
        try:
            if not controller_recorded(self.store.controller()):
                return None
        except Exception:  # noqa: BLE001 - a listing never fails on one node
            return None
        try:
            if self._inventory_probe is None:
                from .copilot_setup import hardware_inventory as probe
            else:
                probe = self._inventory_probe
            free = int((probe() or {}).get("storage_free_bytes") or 0)
        except Exception:  # noqa: BLE001 - the machine is listed without the figure
            free = 0
        return {
            "node_id": CONTROLLER_PLACEMENT_ID, "name": CONTROLLER_PLACEMENT_NAME,
            "root_free_bytes": free, "cached_bytes": 0, "models": [],
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


#: A drafter download that could not be started on a machine (VD-214).
DRAFTER_START_FAILED = "The draft model's download could not be started: {error}"


def _drafter_summary(drafter: Drafter) -> Dict[str, Any]:
    return {"repo": drafter.repo, "revision": drafter.revision,
            "weight_bytes": drafter.weight_bytes}


def _machines(entries: List[Mapping[str, Any]]) -> str:
    return ", ".join(str(entry.get("name") or entry["node_id"]) for entry in entries)


def pull_job_message(result: Mapping[str, Any]) -> str:
    """The pull job's last line: where the model and its draft model are cached, and where not.

    VD-214 (adversarial review 1): one verdict however the draft model's
    download failed, said in the operation dialog the library's job shows,
    with what it means - a deploy there fetches it then. A model cached
    nowhere is never called cached (LESSONS 1).
    """
    entries = list(result.get("nodes") or [])
    ready = [entry for entry in entries if entry["state"] == "ready"]
    failed = [entry for entry in entries if entry["state"] != "ready"]
    if not ready:
        return PULL_CACHED_NOWHERE.format(machines=_machines(failed))
    words = [PULL_CACHED_ON.format(machines=_machines(ready))]
    if failed:
        words.append(PULL_NOT_CACHED_ON.format(machines=_machines(failed)))
    drafter = result.get("drafter")
    if drafter:
        missing = [entry for entry in drafter["nodes"] if entry["state"] != "ready"]
        words.append(DRAFTER_NOT_CACHED_ON.format(
            machines=_machines(missing),
            gb=round(int(drafter.get("weight_bytes") or 0) / 1e9, 1),
        ) if missing else DRAFTER_CACHED_TOO)
    return " ".join(words)


def remove_job_message(result: Mapping[str, Any]) -> str:
    """The remove job's last line, saying what became of the draft model on each machine."""
    words = [REMOVED_WEIGHTS]
    for entry in result.get("removed") or []:
        name = entry.get("name") or entry["node_id"]
        if entry.get("drafters_removed"):
            words.append(DRAFTER_REMOVED_ON.format(machine=name))
        for kept in entry.get("drafters_kept") or []:
            words.append(DRAFTER_KEPT_ON.format(machine=name, reason=kept["reason"]))
    return " ".join(words)


PULL_CACHED_ON = "Model weights cached on {machines}."
PULL_NOT_CACHED_ON = "They were not cached on {machines}; the model library shows why."
PULL_CACHED_NOWHERE = (
    "The model weights were not cached on {machines}; the model library shows why."
)
DRAFTER_CACHED_TOO = "Its draft model was cached beside them."
DRAFTER_NOT_CACHED_ON = (
    "Its draft model was not cached on {machines}, so a deploy there fetches "
    "it then ({gb} GB); the model library shows why."
)
REMOVED_WEIGHTS = "Cached model weights removed."
DRAFTER_REMOVED_ON = "Its draft model was removed from {machine} too."
DRAFTER_KEPT_ON = "Its draft model stays on {machine}. {reason}"

#: Why a removed model's draft model stayed on the machine (VD-214).
DRAFTER_KEPT_UNKNOWN = (
    "Kept: Vaelor could not read the cluster's deployments, so it cannot tell "
    "whether one serves with it. Remove it from the library later."
)
DRAFTER_KEPT_FOR_MODEL = (
    "Kept: it also drafts for {models}, still cached on this machine."
)
DRAFTER_KEPT_FOR_DEPLOYMENT = (
    "Kept: {users} serves with it on this machine."
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


#: The job checkpoint's band for a pull: never 0 while work runs, never 100
#: before the poll records completion.
_PULL_BAND = (15, 95)


def _bands(repo: str, drafter: Optional[Drafter]):
    """``(model band, drafter band)``: the pull band shared by bytes, so progress never runs back.

    VD-214: with a draft model the library makes two downloads, one after the
    other. Each gets the share of the band its weights are of the whole, from
    the catalog's measured size - a model with a default drafter is always a
    catalog entry - so the checkpoint climbs once from 15 to 95.
    """
    if drafter is None:
        return _PULL_BAND, None
    low, high = _PULL_BAND
    model_bytes = next(
        (int(entry["weight_bytes"]) for entry in GPU_MODEL_CATALOG.values()
         if entry["repo"] == repo), 0,
    )
    total = model_bytes + drafter.weight_bytes
    split = low + int((high - low) * (model_bytes / total if total else 0.5))
    return (low, split), (split, high)


def _overall_percent(percents: Dict[str, Optional[int]], band=_PULL_BAND) -> int:
    """A truthful aggregate percent for the job checkpoint, inside ``band``.

    Averages only the nodes that have actually reported a percent, and maps that
    into the band (15-95 for the whole pull, :func:`_bands`) so the checkpoint
    never claims 0 while work is running nor 100 before the poll records
    completion. Nodes with no report yet count as 0 of their own progress, so
    the band cannot run ahead of the slowest launcher.
    """
    low, high = band
    if not percents:
        return low
    known = [value for value in percents.values() if value is not None]
    if not known:
        return low
    average = sum(known) / len(percents)
    return low + int((high - low) * max(0.0, min(100.0, average)) / 100)


def _deployments_using(
    deployments: Optional[List[Mapping[str, Any]]], repo: str, node_id: str,
) -> Optional[List[Dict[str, str]]]:
    """The cluster deployments whose serving needs ``repo`` on ``node_id``.

    Every vLLM deployment that names the repo (``units.repo``, else its
    ``model_id``) and places a part on the node, in a state that will read
    the weights again: serving, loading, or paused (a paused deployment is
    the scale-to-zero rest state, woken on the next request). A failed or
    removed one does not hold them. VD-214: a deployment that serves WITH a
    draft model reads the drafter's weights on that wake too, so it holds
    the drafter's repo as it holds its model's. ``None`` when the deployments could not
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
        named = {str(units.get("repo") or record.get("model_id") or ""),
                 _served_drafter_repo(units)}
        if repo in named and node_id in list(record.get("node_ids") or []):
            users.append({"name": str(record.get("name", "")), "state": state,
                          "state_label": _HOLDING_STATES[state]})
    return users


def _served_drafter_repo(units: Mapping[str, Any]) -> str:
    """The repo of the draft model a deployment's recorded options serve with, or ``""``."""
    options = units.get("vllm_options")
    chosen = options.get("drafter") if isinstance(options, Mapping) else None
    key = chosen.get("key") if isinstance(chosen, Mapping) else None
    drafter = DRAFTERS.get(key) if isinstance(key, str) else None
    return drafter.repo if drafter is not None else ""


#: The deployment states whose machines are known to hold its weights: it
#: served from them (or is paused, and loads again from them). A FIRST deploy's
#: ``deploying`` row is not one - its download may still be running (VD-133
#: 1.5.5 #9) - but a Load's is (adversarial L6): it keeps the served record,
#: ``units.repo`` included, which a first deploy's row does not carry yet.
_SERVED_STATES = ("healthy", "unloaded")

#: What a ready row with no measured size says (VD-133 1.5.5 #9, adversarial L4:
#: it claims no cause it cannot know; "Needed by" says which deployment).
HELD_NOT_RECORDED = "Vaelor has no record of the size of these weights."


def _held_by_deployments(
    deployments: Optional[List[Mapping[str, Any]]], rows: List[Mapping[str, Any]],
) -> List[Dict[str, Any]]:
    """A ``ready`` row for each model (and draft model) a served deployment holds with no row.

    VD-133 1.5.5 #9: a cold install keeps weights but not rows, and a deploy
    before 1.5.5 recorded none. The deployment is the evidence the weights are
    there; their size is never guessed (0 bytes, which the listing reads as
    ``size_known`` false, LESSONS 5).
    """
    known = {(str(row["node_id"]), str(row["repo"])) for row in rows}
    held: Dict[Any, Dict[str, Any]] = {}
    for record in deployments or ():
        units = record.get("units") or {}
        state = str(record.get("state", ""))
        if units.get("engine") != "vllm" or not (
                state in _SERVED_STATES or (state == "deploying" and units.get("repo"))):
            continue
        repos = (str(units.get("repo") or record.get("model_id") or ""),
                 _served_drafter_repo(units))
        for node_id in record.get("node_ids") or ():
            for repo in filter(None, repos):
                key = (str(node_id), repo)
                if key not in known and key not in held:
                    held[key] = {
                        "node_id": key[0], "repo": repo, "revision": "",
                        "state": "ready", "bytes_on_disk": 0,
                        "message": HELD_NOT_RECORDED, "unit": "",
                    }
    return list(held.values())


def deploy_cache_recorder(store: Any) -> Callable[[str, str, str, int], None]:
    """What a deploy's finished weights fetch records, so the library lists it (VD-133 1.5.5 #9).

    `gpu_pool_pull.ModelPullWaiter` calls it once a node's fetch ends ``ok``
    with the bytes the pull program measured on disk; a fetch that reported
    none keeps a size recorded before rather than writing 0 over it.
    """
    def record(node_id: str, repo: str, revision: str, bytes_on_disk: int) -> None:
        size = int(bytes_on_disk or 0) or int(
            (store.get_model_cache(node_id, repo, revision) or {}).get("bytes_on_disk") or 0)
        store.put_model_cache(node_id=node_id, repo=repo, revision=revision,
                              state="ready", bytes_on_disk=size, message=_WEIGHTS_CACHED)
    return record


def _drafter_kept_reason(
    drafter: Drafter, removed_repo: str, node_id: str,
    rows: List[Mapping[str, Any]], users: List[Dict[str, str]],
) -> str:
    """Why ``drafter`` stays on ``node_id`` once ``removed_repo`` goes, or ``""`` to remove it.

    VD-214: kept while another cached (or caching) model on the machine is one
    it drafts for - a deploy of that model would fetch it again - or while a
    deployment there serves with it (``users``, :func:`_deployments_using`).
    """
    others = sorted({
        str(row["repo"]) for row in rows
        if row["node_id"] == node_id and row["repo"] != removed_repo
        and row["repo"] in drafter.targets and row.get("state") in {"ready", "pulling"}
    })
    if others:
        return DRAFTER_KEPT_FOR_MODEL.format(models=", ".join(others))
    if users:
        return DRAFTER_KEPT_FOR_DEPLOYMENT.format(users="; ".join(
            "'{}' ({})".format(user["name"], user["state_label"]) for user in users
        ))
    return ""
