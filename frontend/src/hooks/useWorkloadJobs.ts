import { useCallback, useEffect, useRef, useState } from "react";
import { apiRequest } from "../lib/api";
import { jobIsTerminal } from "../lib/jobPresentation";
import type { WorkloadJob } from "../components/workloads-types";
import type { WorkloadActivityLedger } from "../lib/workloadActivity";
import type { Session } from "../types";

type ProjectedWorkloadJob = WorkloadJob & {
  operation_state?: string;
  attention?: boolean;
  retryable?: boolean;
  readiness?: string;
  liveness?: string;
  resolved_by_retry?: boolean;
  retry_depth?: number;
  retry_ancestry?: string[];
  retry_root_id?: string;
};

type WorkloadJobsResponse = WorkloadActivityLedger & { jobs: ProjectedWorkloadJob[] };

type WorkloadJobsPayload = WorkloadJobsResponse | ProjectedWorkloadJob[];

const jobsFromResponse = (response: WorkloadJobsPayload): ProjectedWorkloadJob[] => (
  Array.isArray(response) ? response : response.jobs
);

const jobSignature = (job: WorkloadJob) => {
  const projected = job as ProjectedWorkloadJob;
  return `${job.id}:${projected.operation_state ?? job.state}:${projected.attention ?? ""}:${projected.retryable ?? ""}:${projected.readiness ?? ""}:${projected.liveness ?? ""}:${projected.resolved_by_retry ?? ""}:${projected.retry_depth ?? ""}`;
};

export function useWorkloadJobs({
  onLifecycleChanged,
  role,
}: {
  onLifecycleChanged?: () => Promise<void> | void;
  role: Session["user"]["role"];
}) {
  const [jobs, setJobs] = useState<WorkloadJob[]>([]);
  const jobStateSignature = useRef("");

  const refreshJobs = useCallback(async () => {
    if (role === "viewer") return;
    const response = await apiRequest<WorkloadJobsPayload>("/jobs?limit=50&summary=true");
    const nextJobs = jobsFromResponse(response);
    jobStateSignature.current = nextJobs.map(jobSignature).join("|");
    setJobs(nextJobs);
  }, [role]);

  useEffect(() => {
    if (role === "viewer") return;
    let polling = false;
    const pollJobs = () => {
      if (!document.hidden && !polling) {
        polling = true;
        void apiRequest<WorkloadJobsPayload>("/jobs?limit=50&summary=true")
          .then((response) => {
            const nextJobs = jobsFromResponse(response);
            const nextSignature = nextJobs.map(jobSignature).join("|");
            const lifecycleChanged = nextSignature !== jobStateSignature.current;
            jobStateSignature.current = nextSignature;
            setJobs(nextJobs);
            if (
              lifecycleChanged
              && nextJobs.some((job) => (
                jobIsTerminal(job)
                && (
                  job.type.startsWith("compose.")
                  || job.type.startsWith("model.")
                  || job.type === "host.docker.install"
                )
              ))
            ) {
              void Promise.resolve(onLifecycleChanged?.()).catch(() => undefined);
            }
          })
          .catch(() => undefined)
          .finally(() => { polling = false; });
      }
    };
    const interval = window.setInterval(pollJobs, 3000);
    const visibilityChanged = () => {
      if (!document.hidden) pollJobs();
    };
    document.addEventListener("visibilitychange", visibilityChanged);
    return () => {
      window.clearInterval(interval);
      document.removeEventListener("visibilitychange", visibilityChanged);
    };
  }, [onLifecycleChanged, role]);

  return { jobs, refreshJobs, setJobs };
}
