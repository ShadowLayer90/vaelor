"""Wait for one GPU node's weights fetch, and say truthfully how it went.

Housed out of `gpu_pool_operations` for the 1,000-line ceiling `CLAUDE.md` sets,
and it is a coherent job on its own: a deploy needs ONE node's pull finished
before it can serve, where `gpu_model_library` polls many nodes at once and
records cache rows. The runtime still owns every command; this owns the waiting,
the checkpoint wording and the two failure sentences.

The polling discipline is the one VD-125's adversarial reviews arrived at:
`GpuPoolRuntime.pull_verdict` reads the unit and then the progress file and
weighs the two together, so a QUEUED unit is never mistaken for a finished one
and deleted mid-fetch (``systemctl start --no-block`` returns on enqueue, and a
unit that has never run reports ``inactive``/``success`` - byte-for-byte what a
clean finish reports), and a unit that exited cleanly without the program
recording ``done`` is a failure rather than a served half-download. A dropped
transport is bounded per node rather than coerced to "still starting" for the
whole 7,200 s deadline.

The two pure readings that discipline rests on - :func:`classify_pull` (what a
pull unit's systemd properties mean) and :func:`pull_outcome` (the unit and the
progress file weighed together) - live here too, beside the waiter that acts on
them; `GpuPoolRuntime.pull_verdict` imports them and owns only the ORDER the two
remote reads are made in. They moved here from the runtime when the compile
cache and the serving-unit liveness read (VD-127, the bigger-than-one-box
proof) took the runtime to its line ceiling; nothing here imports the runtime,
so the dependency runs one way.
"""

from __future__ import annotations

import logging
import time
from typing import Any, Callable, Dict, Mapping, Optional

from .ssh_transport import SshTransportError
from .vllm_drafters import drafter_profile
from .vllm_serve_options import require_options

LOGGER = logging.getLogger(__name__)

#: How the deploy's weight pre-fetch is polled: the gap between reads of the
#: pull unit, and the ceiling on one node's fetch. Generous because the artifact
#: is tens of gigabytes over whatever link the node has.
PULL_POLL_SECONDS = 5.0
PULL_TIMEOUT_SECONDS = 7200.0

#: The percent a pull checkpoint reports. One number for the whole fetch: the
#: puller's own percent goes in the message, and inventing a deploy-level
#: percentage from it would be a second, disagreeing progress figure.
PULL_CHECKPOINT_PERCENT = 45

#: How many CONSECUTIVE transport failures a poll absorbs before it gives up on
#: the node. Six at the pollers' five-second cadence is about thirty seconds,
#: which rides out a reconnect without letting an unreachable node hold a pull
#: for the whole two-hour deadline. One home, because every poller - the deploy
#: pre-fetch, the model library's per-node loop, and the server startup wait's
#: liveness read (`gpu_pool_startup`) - bounds it the same.
PULL_TRANSPORT_FAILURES = 6

#: The ``ActiveState`` values that mean the pull unit is doing work. Read only
#: by :func:`classify_pull`, which is the one place any caller asks what a pull's
#: properties mean.
_PULL_RUNNING_STATES = frozenset(
    {"active", "activating", "reloading", "deactivating"}
)

#: The only phase the pull program writes once the weights are on disk; see the
#: progress-file contract in :mod:`vaelor.gpu_pull_program`.
_PULL_DONE_PHASE = "done"

#: What a poller reports when the unit exited cleanly but the progress file
#: never recorded completion. Written here rather than in each poller so the two
#: cannot describe the same contradiction two ways.
PULL_INCOMPLETE_MESSAGE = (
    "The pull unit finished without recording that the weights were cached."
)


#: What a checkpoint says while a node fetches a deployment's draft model.
DRAFT_PULL_LABEL = "draft model ({gb} GB) - "


def classify_pull(properties: Mapping[str, str]) -> str:
    """What a pull unit's systemd properties mean: pending/running/ok/failed.

    The one place that reading lives, because both the deploy pre-fetch
    (`gpu_pool_operations`) and the model library's poll ask it, and the answer
    is subtle: ``systemctl start --no-block`` returns once the job is ENQUEUED,
    and a unit that has never run reports ``ActiveState=inactive`` with
    ``Result=success`` - systemd's default for "no result yet". Reading that as
    a finished pull declared success before the fetch began and deleted the
    queued unit. ``ExecMainStartTimestampMonotonic`` is 0/absent until the main
    process starts, so it separates "queued" from "done"; ``pending`` keeps the
    caller polling under its own deadline.
    """
    active = str(properties.get("ActiveState", "") or "").strip()
    result = str(properties.get("Result", "") or "").strip()
    started = str(
        properties.get("ExecMainStartTimestampMonotonic", "") or ""
    ).strip()
    if (result and result != "success") or active == "failed":
        return "failed"
    if started in {"", "0"}:
        return "pending"
    if active in _PULL_RUNNING_STATES:
        return "running"
    if active == "inactive" and result == "success":
        return "ok"
    return "failed"


def pull_outcome(
    properties: Mapping[str, str], progress: Mapping[str, Any]
) -> str:
    """What a pull is doing, read from the unit AND the progress file together.

    The one place that judgement lives; neither source suffices. It is reached
    through `GpuPoolRuntime.pull_verdict`, which owns the ORDER the two are
    read in. **systemd decides every state but one**: a file cannot promote a
    running unit, and a failed unit stays failed whatever it says.

    **A ``pending`` unit whose file says ``done`` is over.** systemd garbage
    collects a finished, never-enabled oneshot between polls and then reports a
    FRESH unit - ``ExecMainStartTimestampMonotonic=0``, i.e. ``pending`` -
    forever, burning the whole deadline on a pull that already succeeded. The
    file is the only evidence left, and trustworthy because `start_model_pull`
    removes the per-repo file before every start: a ``done`` can only be this
    program's, and a never-started unit has none, so the never-ran false
    success cannot recur.

    **A clean exit is corroborated by that same file.** For the same reason it
    holds no leftovers, a ``fetching`` row beside a unit that exited ``success``
    means the oneshot ended before the program's ``done`` write, and serving
    that would load a repo half on disk. An absent or unreadable file is not a
    contradiction - it says nothing - and keeps the unit's own verdict.
    """
    state = classify_pull(properties)
    phase = str((progress or {}).get("phase", "") or "").strip()
    if state == "pending":
        return "ok" if phase == _PULL_DONE_PHASE else "pending"
    if state != "ok":
        return state
    if phase and phase != _PULL_DONE_PHASE:
        return "failed"
    return "ok"


def pull_note(progress: Dict[str, Any]) -> str:
    """A truthful one-line note about a running pull, from its progress object.

    The percent is shown only when the puller has written one; until then the
    note carries the phase alone, so a checkpoint never invents a number.
    """
    phase = str(progress.get("phase", "") or "starting")
    try:
        percent = int(progress["percent"])
    except (KeyError, TypeError, ValueError):
        return f"the weights pull is {phase}"
    return "the weights pull is {} ({}%)".format(
        phase, max(0, min(100, percent))
    )


def pull_failure_note(
    status: Dict[str, Any], progress: Dict[str, Any]
) -> str:
    """Why a finished pull did not succeed, in the deploy's own voice.

    Two different endings need two different sentences: a unit that FAILED
    carries the puller's own reason, while a unit that exited cleanly without
    the program writing ``done`` has no reason to report and needs the
    contradiction named instead - the progress file's last line there is a
    "fetching" note, which would read as progress, not as the failure it is.
    """
    if classify_pull(status) == "ok":
        return PULL_INCOMPLETE_MESSAGE
    return str(progress.get("message") or "the pull did not succeed.")


class ModelPullWaiter:
    """Start one node's weights fetch and block until it ends, either way.

    ``runtime`` is the :class:`vaelor.gpu_pool_runtime.GpuPoolRuntime` (or a
    test's mock of it) and ``sleep`` is the poll wait, injected in the same shape
    `GpuModelLibrary` uses so a multi-poll pull is driven in a unit test without
    waiting minutes for it.
    """

    def __init__(
        self,
        runtime: Any,
        *,
        sleep: Callable[[float], None] = time.sleep,
        monotonic: Callable[[], float] = time.monotonic,
        on_cached: Optional[Callable[[str, str, str, int], None]] = None,
    ):
        self.runtime = runtime
        self._sleep = sleep
        self._monotonic = monotonic
        # VD-133 1.5.5 #9: told ``(node id, repo, revision, bytes on disk)``
        # when a fetch ends with the weights there, so the model library
        # lists what a deploy downloaded (`gpu_model_library.deploy_cache_recorder`).
        self._on_cached = on_cached

    def wait(
        self,
        transport: Any,
        node: Dict[str, Any],
        repo: str,
        revision: Optional[str],
        report: Callable[[int, str], None],
        pulls: Optional[Dict[str, str]] = None,
        image: Optional[str] = None,
        vllm_options: Any = None,
    ) -> None:
        """Fetch ``repo``'s weights onto one node and wait for the pull to end.

        ``vllm_options`` is the serving options the deploy decided
        (`vllm_serve_options`): when they carry a draft model it is fetched
        too, and a failed fetch of it fails the deploy like the model's own -
        a unit is never started "with a drafter" it does not have. Without
        one, nothing else is fetched.

        The container image ships no weights, and each participating node needs
        its own copy, so the deploy pre-fetches before any server starts. It goes
        through the runtime's supervised start/poll pair rather than one blocking
        remote command: a dropped channel mid-multi-GB fetch would otherwise take
        the whole deploy down, and systemd keeps the pull alive across it.

        ``pulls`` is the deploy's in-flight map, if it passed one: this puts the
        unit in on start and takes it out the moment it is stopped, so what is
        left in it when the deploy fails is exactly the oneshot nobody stopped.
        """
        tracked = pulls if pulls is not None else {}
        self._fetch(transport, node, repo, revision, report, tracked, image, "")
        # VD-213: the draft model the decided options serve with is fetched
        # onto every machine that serves it, at its pinned commit and into
        # the same hub cache, so the container reads it as it reads the
        # model. Its size is said, because it is a second download.
        key = require_options(vllm_options).advanced[2] if vllm_options is not None else None
        if key is not None:
            drafter = drafter_profile(key)
            self._fetch(
                transport, node, drafter.repo, drafter.revision, report, tracked,
                image, DRAFT_PULL_LABEL.format(gb=round(drafter.weight_bytes / 1e9, 1)),
            )

    def fetch_drafter(
        self, transport: Any, node: Dict[str, Any], key: str,
        report: Callable[[int, str], None], image: Optional[str] = None,
    ) -> None:
        """Fetch one pinned draft model onto one node, as :meth:`wait` does for a deploy.

        VD-217: a Load that gives a record made before the draft model this
        release's one (`vllm_serve_options.with_release_defaults`) fetches it
        first - a Load pulls no weights, and a unit is never started "with a
        drafter" it does not have. Already there, the pull ends at once.
        """
        drafter = drafter_profile(key)
        self._fetch(
            transport, node, drafter.repo, drafter.revision, report, {}, image,
            DRAFT_PULL_LABEL.format(gb=round(drafter.weight_bytes / 1e9, 1)),
        )

    def _fetch(
        self, transport: Any, node: Dict[str, Any], repo: str,
        revision: Optional[str], report: Callable[[int, str], None],
        tracked: Dict[str, str], image: Optional[str], label: str,
    ) -> None:
        """One repo's supervised pull on one node (:meth:`wait`), ``label`` naming a second one."""
        # In the deployment's own image (``image``), which the deploy made
        # present on this node before calling here.
        info = self.runtime.start_model_pull(
            transport, repo=repo, revision=revision, image=image,
        )
        unit = info["unit"]
        node_id = str(node.get("id", ""))
        tracked[node_id] = unit
        deadline = self._monotonic() + PULL_TIMEOUT_SECONDS
        unreachable = 0
        while self._monotonic() < deadline:
            try:
                state, status, progress = self.runtime.pull_verdict(
                    transport, unit, repo,
                )
            except SshTransportError as error:
                # A transient transport failure is not a pull failure - systemd
                # is still running the fetch - but an unreachable node is, and
                # coercing every failure to "still working" would hold the deploy
                # for the whole pull deadline. Bounded, and reset by any answer.
                unreachable += 1
                if unreachable >= PULL_TRANSPORT_FAILURES:
                    # Leave no dangling oneshot behind, exactly as the deadline
                    # path below does. Best-effort: the node is by definition not
                    # answering, so a failed stop must not replace the reason the
                    # deploy is being abandoned - and the unit stays in
                    # ``tracked`` for the record, because a node that will not
                    # answer probably did not act on the stop either.
                    self.stop_quietly(transport, unit)
                    raise RuntimeError(
                        "{} stopped answering while fetching {}: {}".format(
                            node["name"], repo, error
                        )
                    ) from error
                report(
                    PULL_CHECKPOINT_PERCENT,
                    f"{node['name']}: reconnecting to follow the pull",
                )
                self._sleep(PULL_POLL_SECONDS)
                continue
            unreachable = 0
            if state in {"pending", "running"}:
                report(
                    PULL_CHECKPOINT_PERCENT,
                    f"{node['name']}: {label}{pull_note(progress)}",
                )
                self._sleep(PULL_POLL_SECONDS)
                continue
            self.runtime.stop_pull_unit(transport, unit)
            tracked.pop(node_id, None)
            if state == "ok":
                self._record(node_id, repo, revision, progress)
                return
            raise RuntimeError(
                "{} could not fetch {}: {}".format(
                    node["name"], repo, pull_failure_note(status, progress),
                )
            )
        self.runtime.stop_pull_unit(transport, unit)
        tracked.pop(node_id, None)
        raise RuntimeError(
            f"{node['name']} did not finish fetching {repo} in time."
        )

    def _record(self, node_id: str, repo: str, revision: Optional[str], progress: Any) -> None:
        """Tell ``on_cached`` the weights are on ``node_id``; a failed record never fails the serve."""
        if self._on_cached is None:
            return
        size = progress.get("total_bytes") if isinstance(progress, Mapping) else 0
        try:
            self._on_cached(node_id, repo, str(revision or ""), int(size or 0))
        except Exception as error:  # noqa: BLE001 - the weights are served either way
            LOGGER.warning("The model library could not record %s on %s: %s",
                           repo, node_id, error)

    def stop_quietly(self, transport: Any, unit: str) -> None:
        """Stop a pull unit on an abort path, tolerating a dead transport."""
        try:
            self.runtime.stop_pull_unit(transport, unit)
        except SshTransportError:
            pass
