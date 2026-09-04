import type { ProposedJob } from "../components/ActionReviewDialog";

export function continueProposedWorkload(job: ProposedJob) {
  const raw = job.payload.query ?? job.payload.message ?? job.payload.project ?? "";
  const request = typeof raw === "string" && raw.trim()
    ? raw.trim()
    : job.type === "model.inspect"
      ? "Find and deploy a local AI model that fits this Vaelor node"
      : "Prepare this workload for deployment";
  window.sessionStorage.setItem("vaelor.workload-planner-request", request);
  window.dispatchEvent(new CustomEvent("pironman:navigate", { detail: "workloads" }));
}
