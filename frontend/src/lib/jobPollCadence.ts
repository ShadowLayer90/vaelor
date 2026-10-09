import { canonicalOperationState, type JobProjectionInput } from "./jobPresentation";

/**
 * How often the setup-activity list re-reads the jobs (FE-W7-7, LESSONS 16).
 *
 * W6-D3 polled every second while any shown job was non-terminal. A job parked
 * in approval or blocked is non-terminal too, so one of them kept
 * `/jobs?limit=50&summary=true` at 1 Hz indefinitely. Now only an active job
 * (queued or running) that has moved in the last half minute is polled fast;
 * one that stops moving backs off, and after five minutes without change it is
 * named as possibly stuck.
 */

export const FAST_SECONDS = 1;
export const NORMAL_SECONDS = 3;
export const SLOW_SECONDS = 10;
export const MOVING_WINDOW_SECONDS = 30;
export const STALLED_AFTER_SECONDS = 300;

type CadenceJob = JobProjectionInput & { id: string; progress?: number; message?: string };

/** When each job last changed what the owner sees: state, progress or message. */
export type ProgressMarks = ReadonlyMap<string, { key: string; since: number }>;

const ACTIVE = new Set(["queued", "running"]);

const progressKey = (job: CadenceJob) =>
  `${canonicalOperationState(job)}|${job.progress ?? ""}|${job.message ?? ""}`;

/** ``marks`` updated for ``jobs`` read at ``now`` (seconds). */
export function trackProgress(marks: ProgressMarks, jobs: readonly CadenceJob[], now: number): ProgressMarks {
  const next = new Map<string, { key: string; since: number }>();
  for (const job of jobs) {
    const key = progressKey(job);
    const previous = marks.get(job.id);
    next.set(job.id, previous && previous.key === key ? previous : { key, since: now });
  }
  return next;
}

/** The poll interval in seconds, and the active jobs that have not moved for five minutes. */
export function pollCadence(
  jobs: readonly CadenceJob[], marks: ProgressMarks, now: number,
): { seconds: number; stalled: string[] } {
  const active = jobs.filter((job) => ACTIVE.has(canonicalOperationState(job)));
  if (!active.length) return { seconds: NORMAL_SECONDS, stalled: [] };
  const idle = (job: CadenceJob) => now - (marks.get(job.id)?.since ?? now);
  const stalled = active.filter((job) => idle(job) >= STALLED_AFTER_SECONDS).map((job) => job.id);
  if (active.some((job) => idle(job) < MOVING_WINDOW_SECONDS)) return { seconds: FAST_SECONDS, stalled };
  if (stalled.length === active.length) return { seconds: SLOW_SECONDS, stalled };
  return { seconds: NORMAL_SECONDS, stalled };
}
