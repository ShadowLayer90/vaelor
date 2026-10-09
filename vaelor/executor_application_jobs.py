"""Application-research queue orchestration owned outside the core executor."""

from __future__ import annotations

from typing import Any, Dict

from .application_executor import execute_application_job
from .application_research_workflow import ResearchWorkflowError
from .job_control import JobCancelled


class ExecutorApplicationJobsMixin:
    """Run durable application planning jobs through their feature owner."""

    def _run_application_job(self, job: Dict[str, Any]) -> Dict[str, Any]:
        try:
            state, message, result = execute_application_job(
                job["type"], job["payload"], job["actor"],
                self.application_deployments, self.workloads_root,
                lambda percent, detail, phase: self._checkpoint(
                    percent, detail, phase=phase,
                ),
                self.application_learning,
                self.application_research,
                broker=self.credential_broker,
                # The job queue, so a discovery-weak 4B pass can enqueue exactly
                # one capable follow-up job on the same draft (auto-escalation).
                job_store=self.store,
            )
            return self.store.finish(
                job["id"], state=state, message=message, result=result,
                phase=(
                    "ready_for_review"
                    if job["type"] == "application.research" else None
                ),
            )
        except JobCancelled:
            raise
        except ResearchWorkflowError as error:
            # A hard research failure still reports whether the capable graphics
            # model is available (and which tier just failed), so the failure
            # screen can offer or honestly disable "Try the larger model".
            # `execute_application_job` stamps these onto the error; absent (a
            # pre-model-probe failure) they simply do not appear.
            failure_result: Dict[str, Any] = {
                "code": "application_research_blocked",
                "blocker_layer": error.blocker_layer,
            }
            if isinstance(getattr(error, "capable_available", None), bool):
                failure_result["capable_available"] = error.capable_available
            if getattr(error, "capable_unavailable_reason", ""):
                failure_result["capable_unavailable_reason"] = error.capable_unavailable_reason
            if isinstance(getattr(error, "model_tier_used", None), str):
                failure_result["model_tier_used"] = error.model_tier_used
            return self.store.finish(
                job["id"], state="failed", message=str(error),
                result=failure_result,
                phase=error.phase,
                blocker_layer=error.blocker_layer,
            )
        except (OSError, RuntimeError, ValueError) as error:
            research = job["type"] == "application.research"
            return self.store.finish(
                job["id"], state="failed", message=str(error),
                result={
                    "code": "job_failed",
                    **({"blocker_layer": "execution"} if research else {}),
                },
                phase="needs_input" if research else None,
                blocker_layer="execution" if research else None,
            )
