export interface JobHistoryItem {
  id: string;
  type: string;
  state: string;
  created_at: number;
  payload?: Record<string, unknown>;
  retry_of?: string | null;
  operation_state?: string;
  attention?: boolean;
  retryable?: boolean;
  readiness?: string;
  liveness?: string;
  resolved_by_retry?: boolean;
  retry_depth?: number;
}


import { jobIsRetryable, jobIsSuccessful } from "./jobPresentation";


const APPROVAL_FIELDS = new Set(["confirm", "confirmation"]);

const stableValue = (value: unknown, ignoredFields: Set<string>): unknown => {
  if (Array.isArray(value)) {
    return value.map((item) => stableValue(item, ignoredFields));
  }
  if (value && typeof value === "object") {
    return Object.fromEntries(
      Object.entries(value as Record<string, unknown>)
        .filter(([key]) => !ignoredFields.has(key))
        .sort(([left], [right]) => left.localeCompare(right))
        .map(([key, item]) => [key, stableValue(item, ignoredFields)]),
    );
  }
  return value;
};

const operationKey = (job: JobHistoryItem) => {
  const ignoredFields = new Set(APPROVAL_FIELDS);
  if (job.type === "model.deploy" || job.type === "compose.install") {
    ignoredFields.add("port");
  }
  if (job.type === "model.download") {
    ignoredFields.add("inspection_job_id");
  }
  if (job.type === "model.inspect") {
    ignoredFields.add("runtime");
  }
  if (job.type === "application.research") {
    // A retry may discover a different set of sources for the same durable
    // draft. Source URLs are evidence, not the identity of the operation.
    ignoredFields.add("source_urls");
  }
  return `${job.type}:${JSON.stringify(stableValue(job.payload ?? {}, ignoredFields))}`;
};

const successResolves = (success: JobHistoryItem, failure: JobHistoryItem) => {
  if (operationKey(success) === operationKey(failure)) return true;
  if (success.type === "host.web-research.manage" && failure.type === success.type) {
    const convergentActions = new Set(["install", "repair"]);
    return convergentActions.has(String(success.payload?.action ?? ""))
      && convergentActions.has(String(failure.payload?.action ?? ""));
  }
  if (success.type === "cluster.initialize" && failure.type === "cluster.initialize") {
    return true;
  }
  if (success.type !== "compose.install" || failure.type !== "compose.validate") {
    return false;
  }
  const template = String(success.payload?.template ?? "").toLowerCase();
  const request = String(failure.payload?.request ?? "").toLowerCase();
  return Boolean(template && request.includes(template.replaceAll("-", " ")));
};

export function summarizeJobHistory<T extends JobHistoryItem>(jobs: T[]) {
  const retriedJobIds = new Set(
    jobs.map((job) => job.retry_of).filter((id): id is string => Boolean(id)),
  );
  const jobsById = new Map(jobs.map((job) => [job.id, job]));
  const earlierAttemptsByJobId = new Map<string, number>();
  for (const job of jobs) {
    let ancestorId = job.retry_of;
    const visited = new Set<string>();
    while (ancestorId && !visited.has(ancestorId)) {
      visited.add(ancestorId);
      ancestorId = jobsById.get(ancestorId)?.retry_of;
    }
    if (visited.size) earlierAttemptsByJobId.set(job.id, visited.size);
  }

  const successfulByOperation = new Map<string, T[]>();
  for (const job of jobs) {
    if (!jobIsSuccessful(job)) continue;
    const key = operationKey(job);
    const successful = successfulByOperation.get(key) ?? [];
    successful.push(job);
    successfulByOperation.set(key, successful);
  }

  for (const successful of successfulByOperation.values()) {
    successful.sort((left, right) => right.created_at - left.created_at);
  }

  const resolvedByJobId = new Map<string, number>();
  const visible = jobs.filter((job) => {
    if (job.resolved_by_retry) return false;
    // Retry rows are an append-only audit chain. Keep that ledger intact, but
    // show only its newest descendant in the operator-facing activity list.
    if (retriedJobIds.has(job.id)) return false;
    const successful = successfulByOperation.get(operationKey(job)) ?? [];
    if (jobIsRetryable(job)) {
      return !jobs.some(
        (candidate) =>
          jobIsSuccessful(candidate)
          && candidate.created_at > job.created_at
          && successResolves(candidate, job),
      );
    }
    if (jobIsSuccessful(job) && successful[0]?.id === job.id) {
      const resolved = jobs.filter(
        (candidate) =>
          jobIsRetryable(candidate)
          && candidate.created_at < job.created_at
          && successResolves(job, candidate),
      ).length;
      if (resolved) resolvedByJobId.set(job.id, resolved);
    }
    if (jobIsSuccessful(job) && successful[0]?.id !== job.id) return false;
    return true;
  });

  return { visible, resolvedByJobId, earlierAttemptsByJobId };
}
