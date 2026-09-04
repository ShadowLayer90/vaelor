import type { JobHistoryItem } from "./jobHistory";
import { jobIsRetryable, jobIsSuccessful } from "./jobPresentation";

export interface ActivityModel {
  file: string;
  path: string;
  in_use?: boolean;
}

export interface ActivityJob extends JobHistoryItem {
  attempt?: number;
  message?: string;
  progress?: number;
  retry_ancestry?: string[];
  retry_root_id?: string;
  result?: {
    candidates?: Array<{
      id: string;
      files?: Array<{ name: string }>;
    }>;
    path?: string;
    file?: string;
    endpoint?: string;
  };
}

export interface WorkloadActivityLedger {
  jobs: ActivityJob[];
  counts: {
    total: number;
    active: number;
    attention: number;
    retryable: number;
    resolved_failures: number;
  };
  resolved_failure_ids: string[];
  attention_ids: string[];
  retryable_ids: string[];
}

const normalizePath = (value: unknown) => String(value ?? "")
  .replaceAll("\\", "/")
  .replace(/\/+$/, "")
  .toLowerCase();

const sameFile = (left: unknown, right: unknown) => {
  const a = normalizePath(left);
  const b = normalizePath(right);
  return Boolean(a && b && (a === b || a.endsWith(`/${b}`) || b.endsWith(`/${a}`)));
};

export function modelForJob(job: ActivityJob, models: ActivityModel[]) {
  const path = job.result?.path ?? job.payload?.path;
  const file = job.result?.file ?? job.payload?.file;
  return models.find((model) => sameFile(model.path, path) || sameFile(model.file, file));
}

export function modelForCandidate(file: string, models: ActivityModel[]) {
  return models.find((model) => sameFile(model.file, file) || sameFile(model.path, file));
}

export const filesForCandidate = <TFile extends { name: string }>(
  candidate: { files?: TFile[] },
): TFile[] => (Array.isArray(candidate.files) ? candidate.files : []);

const inspectionHasInstalledCandidate = (job: ActivityJob, models: ActivityModel[]) =>
  Boolean(job.result?.candidates?.some((candidate) =>
    filesForCandidate(candidate).some((file) => modelForCandidate(file.name, models)),
  ));

/**
 * Turns the server-owned workload ledger projection into an operator-facing
 * summary. Retry ancestry and resolution are read from the projection; the UI
 * does not rebuild them from the bounded job list.
 */
export function summarizeWorkloadActivity<T extends ActivityJob>(jobs: T[], models: ActivityModel[]) {
  const projectedAncestors = new Set(
    jobs.flatMap((job) => Array.isArray(job.retry_ancestry) ? job.retry_ancestry : []),
  );
  const jobsById = new Map(jobs.map((job) => [job.id, job]));
  const earlierAttemptsByJobId = new Map<string, number>();
  const resolvedByJobId = new Map<string, number>();
  for (const job of jobs) {
    const retryDepth = typeof job.retry_depth === "number"
      ? job.retry_depth
      : Array.isArray(job.retry_ancestry) ? job.retry_ancestry.length : 0;
    if (retryDepth > 0) earlierAttemptsByJobId.set(job.id, retryDepth);
    const resolvedFailures = (job.retry_ancestry ?? [])
      .filter((ancestorId) => jobsById.get(ancestorId)?.resolved_by_retry === true)
      .length;
    if (resolvedFailures > 0) resolvedByJobId.set(job.id, resolvedFailures);
  }

  const visible = jobs
    .filter((job) => !job.resolved_by_retry && !projectedAncestors.has(job.id))
    .filter((job) => {
      if (!jobIsSuccessful(job)) return true;
      if (job.type === "model.download" && modelForJob(job, models)) return false;
      if (job.type === "model.inspect" && inspectionHasInstalledCandidate(job, models)) return false;
      return true;
    });
  return { visible, resolvedByJobId, earlierAttemptsByJobId };
}

/**
 * The one deploy entry allowed to claim the Assistant's current model.
 *
 * #145: `in_use` matches by artifact, so every historical deploy of the file
 * now serving matched, and three consecutive entries each wore "Active in
 * Assistant" with "This is the Assistant's current private model." Only one
 * can be. The newest successful deploy of the in-use artifact is the current
 * one; older rows are the same file's history.
 *
 * #179: this read `jobIsLive`, which was the console re-deriving a server
 * projection. Two things then decided whether a row could claim the Assistant -
 * a liveness word carried from a finished job, and `in_use` from the installed
 * model list - and only the second is a reading of the present.
 */
export function currentAssistantDeployJobId(
  jobs: ActivityJob[], models: ActivityModel[],
): string | null {
  let newest: ActivityJob | null = null;
  for (const job of jobs) {
    if (job.type !== "model.deploy" || !jobIsSuccessful(job)) continue;
    if (!modelForJob(job, models)?.in_use) continue;
    if (!newest || job.created_at > newest.created_at) newest = job;
  }
  return newest?.id ?? null;
}

export function activityStateLabel(
  job: ActivityJob, models: ActivityModel[], currentDeployId: string | null,
) {
  const model = modelForJob(job, models);
  if (job.type === "model.deploy" && jobIsSuccessful(job) && model?.in_use) {
    return job.id === currentDeployId ? "active in assistant" : "available";
  }
  if (job.type === "model.deploy" && jobIsSuccessful(job) && model) {
    return "available";
  }
  return null;
}

export function canRetryFromActivity(job: ActivityJob) {
  if (job.type === "application.research" || job.type === "compose.draft") return false;
  return jobIsRetryable(job);
}
