"""Release a recovery plan whose job ended without running it (CR1 follow-up).

The three Settings > Recovery actions each run a one-use plan: the control
plane stages it (and, for a portable import, marks it approved), queues one job,
and the recovery broker consumes the plan when the job hands it over. A job that
fails or is cancelled BEFORE that hand-over leaves the plan behind. Those jobs
are never retried (`job_vocabulary.REPLAN_REQUIRED_JOB_TYPES`), so a plan left
behind is either dead weight or - for a portable import, whose plan stays marked
approved - a lock: the screen refuses a second approval until the plan expires.

Called by the control plane (which owns the plan files) whenever an owner looks
at, or acts on, a recovery plan: if the job that plan was queued for has ended
failed or cancelled, the plan is released - un-approved for a portable import,
so the staged archive can be approved again; withdrawn for a reset or a removal,
so the next approval stages a fresh one. A plan whose job is still queued or
running is never touched.
"""

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any

#: The job states that mean "ended without the recovery broker running it".
ENDED_WITHOUT_RUNNING = frozenset({"failed", "cancelled"})


def release_failed_plan(plans: Any, jobs: Any, job_type: str, *, unapprove: bool = False) -> bool:
    """Release ``plans``' current plan if its ``job_type`` job has ended; say whether."""
    if plans is None or jobs is None:
        return False
    try:
        plan = (plans.status() or {}).get("plan") or {}
        recent = jobs.list(200)
    except Exception:  # noqa: BLE001 - absence-ok: the plan is simply left as it is
        return False
    plan_id = str(plan.get("id") or "")
    if not plan_id:
        return False
    queued = [
        job for job in recent
        if job.get("type") == job_type
        and str((job.get("payload") or {}).get("plan_id") or "") == plan_id
    ]
    if not queued or any(job.get("state") not in ENDED_WITHOUT_RUNNING for job in queued):
        return False
    if unapprove:
        return unapprove_if_current(plans, plan_id)
    plans.cancel()
    return True


def unapprove_if_current(plans: Any, plan_id: str) -> bool:
    """Drop a portable-import plan's approval ONLY if the file still holds it.

    A plain read-then-write could resurrect a plan the recovery broker claimed
    in between (it moves the file aside to ``.claimed`` and deletes it): the
    write would put the consumed plan back. So the file is first moved aside
    atomically - if it is gone, the broker took it and nothing is written - then
    its id is compared, and only a match is written back without the approval.
    """
    path = Path(plans.path)
    holding = path.with_name(path.name + ".releasing")
    try:
        os.replace(path, holding)
    except FileNotFoundError:
        return False
    try:
        plan = json.loads(holding.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        os.replace(holding, path)
        return False
    if not isinstance(plan, dict) or plan.get("id") != plan_id:
        os.replace(holding, path)
        return False
    plan.pop("approved_at", None)
    staged = path.with_name(path.name + ".staged")
    staged.write_text(json.dumps(plan, separators=(",", ":"), sort_keys=True), encoding="utf-8")
    os.chmod(staged, 0o600)
    os.replace(staged, path)
    holding.unlink(missing_ok=True)
    return True
