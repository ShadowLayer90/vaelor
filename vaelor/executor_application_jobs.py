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
            return self.store.finish(
                job["id"], state="failed", message=str(error),
                result={
                    "code": "application_research_blocked",
                    "blocker_layer": error.blocker_layer,
                },
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
