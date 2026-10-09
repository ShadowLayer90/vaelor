import { useCallback, useEffect, useRef, useState } from "react";
import { apiRequest } from "../lib/api";
import { pollCadence, trackProgress, type ProgressMarks } from "../lib/jobPollCadence";
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
  const jobsRef = useRef<WorkloadJob[]>([]);
  jobsRef.current = jobs;
  const jobStateSignature = useRef("");
  // W6-D3: a job the page added from the POST's answer (queued) is re-read
  // within a second. FE-W7-7: only while an active job moves - a parked job
  // (needs approval, blocked) polls at the normal cadence, and one that stops
  // moving backs off and is named in `stalledJobIds` (`jobPollCadence`).
  const marks = useRef<ProgressMarks>(new Map());
  const nowSeconds = () => Date.now() / 1000;
  marks.current = trackProgress(marks.current, jobs, nowSeconds());
  const [stalledJobIds, setStalledJobIds] = useState<string[]>([]);
  // W8 race: jobs the page added itself (from a POST's answer), with when.
  // A /jobs read that STARTED before the add cannot know the job, so its answer
  // must not drop it; a read started after the add is the server's word.
  const localAdds = useRef(new Map<string, { job: WorkloadJob; at: number }>());
  const readSequence = useRef(0);
  const mergeRead = useCallback((nextJobs: WorkloadJob[], startedAt: number) => {
    const known = new Set(nextJobs.map((job) => job.id));
    const kept: WorkloadJob[] = [];
    for (const [id, entry] of localAdds.current) {
      if (known.has(id) || entry.at < startedAt) localAdds.current.delete(id);
      else kept.push(entry.job);
    }
    return kept.length ? [...kept, ...nextJobs] : nextJobs;
  }, []);
  const setJobsTracked = useCallback((update: WorkloadJob[] | ((current: WorkloadJob[]) => WorkloadJob[])) => {
    setJobs((current) => {
      const next = typeof update === "function" ? update(current) : update;
      const before = new Set(current.map((job) => job.id));
      for (const job of next) {
        if (!before.has(job.id)) localAdds.current.set(job.id, { job, at: ++readSequence.current });
      }
      return next;
    });
  }, []);

  const refreshJobs = useCallback(async () => {
    if (role === "viewer") return;
    const startedAt = ++readSequence.current;
    const response = await apiRequest<WorkloadJobsPayload>("/jobs?limit=50&summary=true");
    const nextJobs = mergeRead(jobsFromResponse(response), startedAt);
    jobStateSignature.current = nextJobs.map(jobSignature).join("|");
    setJobs(nextJobs);
  }, [mergeRead, role]);

  useEffect(() => {
    if (role === "viewer") return;
    // Clearing the interval stops the *next* poll; withdrawing the signal stops
    // the one already in flight from being re-sent by the transport's
    // dead-socket retry after this surface has gone.
    const lifetime = new AbortController();
    let polling = false;
    const pollJobs = () => {
      if (!document.hidden && !polling) {
        polling = true;
        const startedAt = ++readSequence.current;
        void apiRequest<WorkloadJobsPayload>("/jobs?limit=50&summary=true", { signal: lifetime.signal })
          .then((response) => {
            const nextJobs = mergeRead(jobsFromResponse(response), startedAt);
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
    // One-second ticks; a read when the cadence's interval has passed.
    let ticks = 0;
    const interval = window.setInterval(() => {
      ticks += 1;
      const current = pollCadence(jobsRef.current, marks.current, Date.now() / 1000);
      setStalledJobIds((previous) => (
        previous.join("|") === current.stalled.join("|") ? previous : current.stalled
      ));
      if (ticks >= current.seconds) {
        ticks = 0;
        pollJobs();
      }
    }, 1000);
    const visibilityChanged = () => {
      if (!document.hidden) pollJobs();
    };
    document.addEventListener("visibilitychange", visibilityChanged);
    return () => {
      lifetime.abort();
      window.clearInterval(interval);
      document.removeEventListener("visibilitychange", visibilityChanged);
    };
  }, [mergeRead, onLifecycleChanged, role]);

  return { jobs, refreshJobs, setJobs: setJobsTracked, stalledJobIds };
}
