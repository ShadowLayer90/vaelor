"""What an update is doing now, phase by phase, for the update panel (VD-217, R104).

**Why.** In the 2026-10-16 rehearsal the panel said "Update in progress ...
Reload it in a moment" for more than five minutes after the box ran 1.5.3,
and said nothing of the build steps, the guesser download or the 30B going
offline (kb R104). Every fact it needed already existed, in four places and
three processes:

* the ``appliance.upgrade`` job (the executor: download and check);
* the broker's result file (`appliance_upgrade.RESULT_PATH`: install, restart,
  readback, rollback) - written only when the apply ENDS, after the refresh;
* the refresh's own record, :data:`RECORD_NAME`, written here by
  ``vaelor-refresh-models --apply`` as root (`deployment_refresh`) as each build
  step ends, then with the reload jobs it queued;
* those jobs' events, each filed under a refresh phase
  (`gpu_pool_refresh.REFRESH_PHASES`).

:func:`progress` reads them, read only, and derives ONE list of phases
(:data:`PHASES`) with a state from :data:`PHASE_STATES` and a plain reason
each - the panel renders it and decides nothing (LESSONS 6: one derivation).
Only allowlisted fields leave (LESSONS 24): no wheel path, no digest, no job
payload.

**Who writes the record.** Root, from the new package, in the refresh the
broker (any release's) or the installer runs. It goes beside the result file
in ``/var/lib/vaelor/upgrade`` (``2770 vaelor-workloads:vaelor-jobs``,
setgid), so it is ``root:vaelor-jobs 0640`` like the result, readable by the
control plane. When that folder is missing nothing is written: creating it as
root would lock the executor out of its own staging folder.
"""

from __future__ import annotations

import re
import time
from pathlib import Path
from typing import Any, Callable, Dict, List, Mapping, Optional

from .upgrade_files import read_json, write_json

#: The refresh's record, beside the broker's result file.
RECORD_NAME = "refresh-record.json"

#: The phases, in order, by key, with what the panel calls them.
PHASES = (
    ("download", "Download and check the release"),
    ("install", "Install it"),
    ("restart", "Restart Vaelor"),
    ("build", "Build the speed-ups"),
    ("guesser", "Download the guesser"),
    ("reload", "Reload the AI model"),
)

#: A phase's state. The last four end it.
PHASE_STATES = ("pending", "running", "done", "skipped", "held-back", "failed")
ENDED = frozenset(PHASE_STATES[2:])
#: Which state stands for several jobs' together: the least advanced, by
#: position in :data:`PHASE_STATES` (failed first, then running, pending,
#: held back, done, skipped).
_PRECEDENCE = tuple(PHASE_STATES[index] for index in (5, 1, 0, 4, 2, 3))

#: What the reload phase says while it runs (measured on the Z2, kb N93: 1m40s).
MODEL_OFFLINE = "The AI model is offline for about a minute and a half while it reloads."

#: R104 / G88: the one sentence the release notes carry for an update FROM
#: 1.5.1, whose page is 1.5.1's own and cannot show any of this.
OLD_PAGE_NOTE = (
    "Updating from 1.5.1: the update page you start it from is 1.5.1's own and "
    "does not refresh by itself, so reload the page about a minute after you "
    "press Confirm update to follow the rest of the update."
)

#: Why the later phases did not run, when an earlier one ended the update.
NOT_APPLIED = "the update was not applied"
NOT_DOWNLOADED = "the release was not downloaded"
#: R107: what a phase says that can no longer end by itself - an apply cut off
#: (no result, the box still on the old version) or a reload no longer on record.
NOT_FINISHED = (
    "this update did not finish and nothing is still working on it. Check the "
    "version above and start the update again if it is not the one you wanted"
)
#: How long an apply can run by its own bounds (`appliance_upgrade`): the pip
#: reinstall and the unmet dependencies (each PIP_TIMEOUT_SECONDS), the health
#: gate and the refresh with its build steps. R114: the first cut-off (45 min)
#: was shorter than this sum and called a slow, live apply unfinished.
def _apply_bound() -> int:
    from .appliance_upgrade import (
        HEALTH_TIMEOUT_SECONDS, PIP_TIMEOUT_SECONDS, REFRESH_TIMEOUT_SECONDS,
    )

    return 2 * PIP_TIMEOUT_SECONDS + HEALTH_TIMEOUT_SECONDS + REFRESH_TIMEOUT_SECONDS


#: The margin over that bound for the restart, the units check and the queue.
UNFINISHED_MARGIN_SECONDS = 15 * 60
#: After this an update with nothing still active is called unfinished (R107):
#: 4,500 s of bounds plus the margin, 90 minutes.
UNFINISHED_AFTER_SECONDS = _apply_bound() + UNFINISHED_MARGIN_SECONDS
NO_RECORD_OF_RELOADS = "this update left no record of its model reloads"
RELOAD_GONE = "its reload is no longer on record"
NOTHING_TO_RELOAD = "no deployed model needed reloading for this release"
NOTHING_FOR_GUESSER = "nothing new was needed for the guesser on any machine"
NO_GUESSER_MODEL = "no cluster model needed the guesser"
NO_BUILD_RECORD = "no record of the build steps was written"
#: R111 (VD-204's rule): the only reasons a viewer reads - Vaelor's own fixed
#: sentences. Step outcomes, job messages and model notes can name machines,
#: paths and addresses, so a viewer gets the states without them.
VIEWER_REASONS = frozenset({
    MODEL_OFFLINE, NOT_APPLIED, NOT_DOWNLOADED, NOT_FINISHED, NO_RECORD_OF_RELOADS,
    RELOAD_GONE, NOTHING_TO_RELOAD, NOTHING_FOR_GUESSER, NO_GUESSER_MODEL, NO_BUILD_RECORD,
})

#: How a step label's ledger tag is cut before it reaches the panel. Every
#: post_install_steps label carries a VD- tag and nothing else (VD-124).
_TAG = re.compile(r"\s*\(VD-\d+\)$")


def record_path() -> Path:
    """The record's path: beside the broker's result file."""
    from .appliance_upgrade import RESULT_PATH

    return Path(RESULT_PATH).parent / RECORD_NAME


class RefreshRecord:
    """The refresh's record, rewritten whole at each change. Never raises."""

    def __init__(self, path: Optional[Path] = None, clock: Callable[[], float] = time.time):
        self.path = Path(path) if path is not None else record_path()
        self._clock = clock
        self.data: Dict[str, Any] = {"started_at": int(clock()), "steps": [],
                                     "steps_ended_at": None, "queued": [], "ended_at": None}
        self._write()

    def step(self, entry: Mapping[str, Any]) -> None:
        self.data["steps"].append({key: str(entry.get(key, ""))[:600]
                                   for key in ("label", "status", "outcome")})
        self._write()

    def steps_ended(self) -> None:
        self.data["steps_ended_at"] = int(self._clock())
        self._write()

    def queued(self, job: Mapping[str, Any], target: str) -> None:
        self.data["queued"].append({"id": str(job.get("id", "")), "type": str(job.get("type", "")),
                                    "target": str(target)[:200]})
        self._write()

    def ended(self) -> None:
        self.data["ended_at"] = int(self._clock())
        self._write()

    def _write(self) -> None:
        if not self.path.parent.is_dir():
            return
        try:
            # R106: no fixed temporary name in a folder the executor can write.
            write_json(self.path, self.data, 0o640)
        except (OSError, ValueError):  # absence-ok: the journal and the printed lines carry the same
            pass


def read_record(path: Optional[Path] = None) -> Optional[Dict[str, Any]]:
    """The refresh's record: root's own regular file, or ``None`` (R106)."""
    return read_json(Path(path) if path is not None else record_path())


# --- The derivation -------------------------------------------------------------

def _phase(key: str, state: str, reason: str = "", **extra: Any) -> Dict[str, Any]:
    label = dict(PHASES)[key]
    return {"key": key, "label": label, "state": state, "reason": str(reason)[:600], **extra}


def _latest_upgrade(job_store: Any) -> Optional[Dict[str, Any]]:
    try:
        jobs = job_store.list(limit=50)
    except Exception:  # noqa: BLE001 - no job list is no update to follow
        return None
    for job in jobs:
        if job.get("type") == "appliance.upgrade":
            return job
    return None


def _ended_unsuccessfully(state: str) -> bool:
    """A job state that ended without the job's work done (`job_vocabulary.TERMINAL_JOB_STATES`)."""
    from .job_vocabulary import TERMINAL_JOB_STATES

    return state in TERMINAL_JOB_STATES and state != "completed" and state != "healthy"         and state != "superseded"


def _same(left: Any, right: Any) -> bool:
    from .appliance_upgrade import _same_version

    return bool(left) and bool(right) and _same_version(str(left), str(right))


def progress(job_store: Any, running_version: str, last_result: Optional[Mapping[str, Any]],
             record: Optional[Mapping[str, Any]], *, now: Optional[float] = None,
             detail: bool = True) -> Optional[Dict[str, Any]]:
    """The latest update's phases, or ``None`` when no update was ever started here.

    ``detail`` false is a viewer's copy (:func:`for_viewer`).
    """
    from .job_vocabulary import ACTIVE_JOB_STATES

    job = _latest_upgrade(job_store) if job_store is not None else None
    if job is None:
        return None
    target = str((job.get("payload") or {}).get("to_version", ""))
    since = int(job.get("created_at") or 0) // 1000
    # Stale data never describes this update: a result or record older than
    # its job is an earlier update's; a record begun after this update's
    # result is a later installer run's (G89).
    result = last_result if last_result and int(last_result.get("completed_at") or 0) >= since else None
    record = record if record and int(record.get("started_at") or 0) >= since else None
    if record is not None and result is not None and \
            int(record.get("started_at") or 0) > int(result.get("completed_at") or 0):
        record = None
    phases: List[Dict[str, Any]] = []

    state = str(job.get("state", ""))
    if state == "completed":
        phases.append(_phase("download", "done"))
    elif _ended_unsuccessfully(state):
        phases.append(_phase("download", "failed", job.get("message", "")))
    else:
        phases.append(_phase("download", "running" if state in ACTIVE_JOB_STATES else "pending",
                             job.get("message", "")))
    downloaded = phases[0]["state"] == "done"
    # R107: reached from THIS apply's own progress, not the version alone - a
    # same-build re-apply runs the target version before anything happened.
    started_from = str((job.get("result") or {}).get("from_version", ""))
    reached = downloaded and _same(running_version, target) and bool(
        (result is not None and result.get("completed")) or record is not None
        or (started_from and not _same(started_from, target)))

    if result is not None and not result.get("completed"):
        why = (result.get("error") or "the update did not complete")
        if result.get("rolled_back"):
            why = "{}; Vaelor {} was put back".format(why, result.get("from_version", ""))
        phases.append(_phase("install", "failed", why))
        phases.append(_phase("restart", "skipped", NOT_APPLIED))
        phases.extend(_phase(key, "skipped", NOT_APPLIED)
                      for key in ("build", "guesser", "reload"))
        return _ending(job, target, reached, phases, None, detail)
    phases.append(_phase("install", "done" if reached else ("running" if downloaded else "pending")))
    phases.append(_phase("restart", "done" if reached else "pending"))
    if phases[0]["state"] == "failed":
        for entry in phases[1:]:
            entry.update(state="skipped", reason=NOT_DOWNLOADED)
        phases.extend(_phase(key, "skipped", NOT_DOWNLOADED)
                      for key in ("build", "guesser", "reload"))
        return _ending(job, target, reached, phases, None, detail)

    build = _build_phase(record, reached, result)
    phases.append(build)
    jobs = [_job(job_store, entry) for entry in (record or {}).get("queued") or ()]
    phases.extend(_model_phases(record, jobs, ended=result is not None))
    if build.pop("deferred", False):
        _build_after_download(phases[-2])
    age = (time.time() if now is None else now) - since
    # R114: still working = the download job, an apply that has written no
    # result yet (within its own bounds), or a reload job not ended.
    active = (not phases[0]["state"] in ENDED and state in ACTIVE_JOB_STATES | {"queued"}) or (
        downloaded and result is None and age <= UNFINISHED_AFTER_SECONDS) or any(
        entry["job"] is not None and str(entry["job"].get("state", "")) not in _terminal()
        for entry in jobs)
    if not active and age > UNFINISHED_AFTER_SECONDS:
        unended = [entry for entry in phases if entry["state"] not in ENDED]
        for index, entry in enumerate(unended):
            # The first phase that never ended failed; the ones after it never ran.
            entry.update(state="failed" if index == 0 else "skipped", reason=NOT_FINISHED)
    return _ending(job, target, reached, phases,
                   _model_result([entry for entry in jobs if entry["job"] is not None]), detail, active)


def _terminal() -> Any:
    from .job_vocabulary import TERMINAL_JOB_STATES

    return TERMINAL_JOB_STATES


def _build_phase(record: Optional[Mapping[str, Any]], reached: bool,
                 result: Optional[Mapping[str, Any]]) -> Dict[str, Any]:
    steps = [{"label": _TAG.sub("", str(step.get("label", ""))),
              "state": str(step.get("status", "")), "reason": _plain(step.get("outcome", ""))}
             for step in (record or {}).get("steps") or () if isinstance(step, Mapping)]
    # The first update from a release before the guesser: its build has no
    # drafter yet and runs after the download instead (N94) - the normal path,
    # not a decline, so it is shown there, never as held back here.
    kept = [step for step in steps if not _waits_for_drafter(step)]
    deferred = len(kept) != len(steps)
    steps = kept
    if record is None:
        if result is not None:
            return _phase("build", "skipped", NO_BUILD_RECORD, steps=[])
        return _phase("build", "running" if reached else "pending", steps=[])
    if not record.get("steps_ended_at"):
        return _phase("build", "running", steps=steps, deferred=deferred)
    held = [step for step in steps if step["state"] == "held-back"]
    if held:
        return _phase("build", "held-back", "; ".join(
            "{}: {}".format(step["label"], step["reason"]) for step in held), steps=steps,
            deferred=deferred)
    return _phase("build", "done", steps=steps, deferred=deferred)


#: The guesser phase's sub-step when the build waited for the download.
BUILD_AFTER_DOWNLOAD = "build the 4-bit copy"


def _waits_for_drafter(step: Mapping[str, Any]) -> bool:
    """The guesser build step that found no drafter yet (`vllm_guesser_build.INPUT_ABSENT`)."""
    from .vllm_guesser_build import INPUT_ABSENT

    return step.get("state") == "held-back" and INPUT_ABSENT in str(step.get("reason", ""))


def _build_after_download(guesser: Dict[str, Any]) -> None:
    """The 4-bit build under the guesser phase, in the phase's own state and words.

    Held back only when the refresh's pre-stop build declined it
    (`gpu_pool_refresh.prebuild_guesser`'s note): a timeout, a refusal, a fetch
    that failed.
    """
    state = guesser["state"]
    guesser["steps"] = [{"label": BUILD_AFTER_DOWNLOAD, "state": state,
                         "reason": guesser["reason"] if state == "held-back" else ""}]


#: What a build module prints before its outcome: ``<what> (<key>): ``.
_ITEM = re.compile(r"^[^():/]{1,80}\([^()]{1,80}\):\s*")


def _plain(outcome: Any) -> str:
    """A step's outcome without each item's ``<what> (<key>):`` prefix."""
    return " / ".join(_ITEM.sub("", part.strip()) for part in str(outcome).split(" / "))[:600]


def _job(job_store: Any, entry: Mapping[str, Any]) -> Dict[str, Any]:
    """A queued job with its phases; ``job`` is ``None`` when the store no longer has it."""
    try:
        job = job_store.get(str(entry.get("id", "")))
        events = job_store.events(str(entry.get("id", ""))) if job else []
    except Exception:  # noqa: BLE001 - an unread job reads as not on record
        job, events = None, []
    return {"job": job or None, "type": str(entry.get("type", "")),
            "phases": [str(event.get("phase") or "") for event in events],
            "target": str(entry.get("target", ""))}


def _model_phases(record: Optional[Mapping[str, Any]], jobs: List[Dict[str, Any]],
                  ended: bool = False) -> List[Dict[str, Any]]:
    """The guesser and reload phases. ``ended``: the apply's result is written, so
    a refresh that left no record (an older release's) will never write one."""
    from .gpu_pool_refresh import PHASE_BUILD, PHASE_FETCH, PHASE_RELOAD
    from .job_vocabulary import CLUSTER_GPU_REFRESH_JOB, TERMINAL_JOB_STATES

    if record is None and ended:
        return [_phase("guesser", "skipped", NO_RECORD_OF_RELOADS),
                _phase("reload", "skipped", NO_RECORD_OF_RELOADS)]
    if not jobs:
        if record is None or not record.get("ended_at"):
            return [_phase("guesser", "pending"), _phase("reload", "pending")]
        return [_phase("guesser", "skipped", NOTHING_TO_RELOAD),
                _phase("reload", "skipped", NOTHING_TO_RELOAD)]
    guesser, reload = [], []
    for entry in jobs:
        job, seen = entry["job"], entry["phases"]
        if job is None:
            # R107: a job the store no longer returns never ends by itself.
            if entry["type"] == CLUSTER_GPU_REFRESH_JOB:
                guesser.append(("skipped", RELOAD_GONE))
            reload.append(("skipped", RELOAD_GONE))
            continue
        ended = str(job.get("state", "")) in TERMINAL_JOB_STATES
        failed = _ended_unsuccessfully(str(job.get("state", "")))
        result = job.get("result") or {}
        fetching = PHASE_FETCH in seen or PHASE_BUILD in seen
        reloading = PHASE_RELOAD in seen
        if job.get("type") != CLUSTER_GPU_REFRESH_JOB:
            pass  # the Assistant's redeploy has no guesser
        elif ended and result.get("guesser_note"):
            # R109: a decline is said even when the fetch failed before its
            # first progress line, so no guesser event was ever filed.
            guesser.append(("held-back", str(result["guesser_note"])))
        elif fetching and not reloading and not ended:
            guesser.append(("running", str(job.get("message", ""))))
        elif fetching:
            guesser.append(("done", ""))
        elif ended or reloading:
            guesser.append(("skipped", NOTHING_FOR_GUESSER))
        else:
            guesser.append(("pending", ""))
        if failed:
            reload.append(("failed", str(job.get("message", ""))))
        elif ended and not result.get("refreshed", job.get("type") == "model.deploy"):
            reload.append(("skipped", str(result.get("reason") or job.get("message", ""))))
        elif ended:
            reload.append(("done", str(result.get("reason", ""))))
        elif reloading:
            reload.append(("running", MODEL_OFFLINE))
        else:
            reload.append(("pending", ""))
    return [_phase("guesser", *_combined(guesser or [("skipped", NO_GUESSER_MODEL)])),
            _phase("reload", *_combined(reload))]


def _combined(states: List[tuple]) -> tuple:
    """One state for several jobs: the least advanced, its reason first."""
    state = min((state for state, _ in states), key=_PRECEDENCE.index)
    reasons = [reason for each, reason in states if each == state and reason]
    return state, " ".join(reasons)


def _model_result(jobs: List[Dict[str, Any]]) -> Optional[Dict[str, Any]]:
    """What each reloaded model now serves with (`gpu_pool_refresh.serving_summary`)."""
    served = []
    for entry in jobs:
        result = entry["job"].get("result") or {}
        summary = result.get("serving")
        if isinstance(summary, Mapping):
            served.append({"name": str(result.get("name") or entry["target"]),
                           **{key: summary.get(key) for key in (
                               "read", "drafter", "guesser", "expert_kernels",
                               "step_merging", "notes")}})
    return {"models": served} if served else None


def _ending(job: Mapping[str, Any], target: str, reached: bool,
            phases: List[Dict[str, Any]], model: Optional[Dict[str, Any]],
            detail: bool = True, active: bool = False) -> Dict[str, Any]:
    found = {
        "job_id": str(job.get("id", "")),
        "from_version": str((job.get("result") or {}).get("from_version", "")),
        "to_version": target,
        "started_at": int(job.get("created_at") or 0) // 1000,
        "reached": reached,
        "phases": phases,
        "finished": all(phase["state"] in ENDED for phase in phases),
        "failed": any(phase["state"] == "failed" for phase in phases),
        "model": model,
        # R114: something is still working on it - the panel never offers
        # another update while this is true, however long it has followed.
        "active": active,
    }
    return found if detail else for_viewer(found)


def for_viewer(found: Dict[str, Any]) -> Dict[str, Any]:
    """A viewer's copy: states and Vaelor's fixed sentences only (R111, VD-204)."""
    def plain(reason: Any) -> str:
        return reason if reason in VIEWER_REASONS else ""

    phases = []
    for phase in found["phases"]:
        copy = dict(phase, reason=plain(phase["reason"]))
        if "steps" in phase:
            copy["steps"] = [dict(step, reason=plain(step.get("reason")))
                             for step in phase["steps"]]
        phases.append(copy)
    model = found.get("model")
    if model:
        model = {"models": [dict({key: entry.get(key) for key in (
            "read", "drafter", "guesser", "expert_kernels", "step_merging")},
            name="", notes=[]) for entry in model["models"]]}
    return dict(found, phases=phases, model=model, job_id="")


#: What a viewer reads instead of a failed apply's own words (pip or systemd
#: text, up to 500 characters each).
RESULT_NOT_SHOWN = "The update did not complete. An administrator can read why on this page."
#: The broker result fields a viewer may read: states and versions, no text.
#: Field names, not job states, so written as one string (the wire-vocabulary
#: scan reads a tuple of these words as a job-state set).
_VIEWER_RESULT_FIELDS = tuple(
    "ok completed rolled_back from_version to_version running_version completed_at".split())


def viewer_result(result: Optional[Mapping[str, Any]]) -> Optional[Dict[str, Any]]:
    """The broker's last result as a viewer reads it (R111 item 4, VD-204): no error text."""
    if not result:
        return None
    copy = {key: result[key] for key in _VIEWER_RESULT_FIELDS if key in result}
    if not result.get("completed"):
        copy["error"] = RESULT_NOT_SHOWN
    return copy
