import { useState } from "react";
import { canonicalOperationState } from "../lib/jobPresentation";
import type { WorkloadJob } from "./workloads-types";

export function confirmApplicationDeploymentClose(dirty: boolean) {
  return !dirty || window.confirm("Discard unsaved deployment changes?");
}

export function applicationResumeRequestSummary(job: Pick<WorkloadJob, "payload">) {
  return typeof job.payload?.application_query === "string" ? job.payload.application_query : undefined;
}

export function applicationResearchResumeEligible(job: WorkloadJob) {
  if (job.type !== "application.research" || typeof job.payload?.draft_id !== "string") return false;
  const operation = canonicalOperationState(job);
  if (["cancelled", "rejected", "superseded"].includes(operation)) return false;
  if (operation === "failed") return job.phase === "needs_input";
  if (operation === "completed" || operation === "healthy") return job.phase === "ready_for_review";
  return true;
}

function consumeApplicationResearchRequest() {
  const request = window.sessionStorage.getItem("vaelor.application-research-request") ?? "";
  if (request) window.sessionStorage.removeItem("vaelor.application-research-request");
  return request;
}

export function useApplicationResearchLifecycle(jobs: WorkloadJob[]) {
  const [applicationResearchRequest] = useState(consumeApplicationResearchRequest);
  const [researchedRequest, setResearchedRequest] = useState(applicationResearchRequest);
  const [applicationDeploymentDirty, setApplicationDeploymentDirty] = useState(false);
  const applicationResumeJob = jobs.find(applicationResearchResumeEligible);
  const applicationResume = applicationResumeJob
    ? {
        jobId: applicationResumeJob.id,
        draftId: String(applicationResumeJob.payload?.draft_id),
        requestSummary: applicationResumeRequestSummary(applicationResumeJob),
      }
    : undefined;

  return {
    applicationDeploymentDirty,
    applicationResearchRequest,
    applicationResume,
    researchedRequest,
    setApplicationDeploymentDirty,
    setResearchedRequest,
  };
}
