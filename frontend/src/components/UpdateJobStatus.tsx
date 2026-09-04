import { useMemo } from "react";
import {
  canonicalOperationState,
  jobIsReady,
  jobIsTerminal,
  jobLabel,
  jobNeedsAttention,
  jobStateLabel,
  jobSummary,
} from "../lib/jobPresentation";
import { StatusPill } from "./StatusPill";
import { Button } from "./ui";
import type { StatusTone } from "./ui/status";

/** One row of the server-side job event log (`GET /jobs/<id>` -> `events`). */
export interface UpdateEvent {
  id?: string | number;
  state: string;
  progress: number;
  message: string;
  created_at?: number;
  phase?: string;
}

export interface UpdateJob {
  id: string;
  type: string;
  state: string;
  operation_state?: string;
  attention?: boolean;
  retryable?: boolean;
  readiness?: string;
  liveness?: string;
  progress: number;
  message: string;
  /** Appended live as polling brings new steps; absent on the create response. */
  events?: UpdateEvent[];
}

/*
 * The staging/apply pipeline emits its own vocabulary (queued -> validating ->
 * downloading -> completed) that predates the canonical operation states, so it
 * is mapped to a tone here rather than routed through `statusTone`, which only
 * knows the canonical names and would grey out every "downloading" row.
 */
const EVENT_TONE: Record<string, StatusTone> = {
  queued: "info",
  validating: "info",
  downloading: "info",
  starting: "info",
  running: "info",
  completed: "success",
  healthy: "success",
  failed: "danger",
  rejected: "danger",
  cancelled: "neutral",
  superseded: "neutral",
};

const eventTone = (state: string): StatusTone => EVENT_TONE[(state ?? "").toLowerCase()] ?? "info";

const clampProgress = (value: number) => Math.max(0, Math.min(100, Math.round(value || 0)));

/**
 * Live status view of a running or finished software-update job.
 *
 * The old feedback was a single `{message} {progress}%` line that testers
 * reported as "does nothing" — easy to miss, and gone the moment the poll
 * loop stopped. This keeps the whole job visible: an overall progress bar, a
 * live timeline of every step, and a clearly distinguished terminal state so a
 * failure surfaces its message instead of vanishing.
 */
export function UpdateJobStatus({
  job,
  onDismiss,
}: {
  job: UpdateJob;
  onDismiss?: () => void;
}) {
  const terminal = jobIsTerminal(job);
  const failed = terminal && (jobNeedsAttention(job) || canonicalOperationState(job) === "rejected");
  const succeeded = terminal && jobIsReady(job) && !failed;
  const tone: StatusTone = failed ? "danger" : succeeded ? "success" : "info";
  const progress = clampProgress(succeeded ? 100 : job.progress);

  // Newest first, so the current step is visible without scrolling the log.
  const events = useMemo(() => [...(job.events ?? [])].reverse(), [job.events]);

  const title = jobLabel(job.type);

  return (
    <section
      aria-busy={!terminal || undefined}
      aria-live={failed ? undefined : "polite"}
      className={`update-status update-status--${tone}`}
      role={failed ? "alert" : "status"}
    >
      <div className="update-status__head">
        <div className="update-status__title">
          <strong>{title}</strong>
          <StatusPill label={jobStateLabel(job)} tone={tone} />
        </div>
        {terminal && onDismiss && (
          <Button onClick={onDismiss} variant="quiet">Dismiss</Button>
        )}
      </div>

      <div
        aria-label={`${title} progress`}
        aria-valuemax={100}
        aria-valuemin={0}
        aria-valuenow={progress}
        className="update-status__bar"
        role="progressbar"
      >
        <span className="update-status__bar-fill" style={{ width: `${progress}%` }} />
        <span className="update-status__bar-value">{progress}%</span>
      </div>

      {failed && <p className="update-status__error">{jobSummary(job)}</p>}

      {events.length > 0 && (
        <ol className="update-status__timeline">
          {events.map((event, index) => (
            <li className="update-status__event" key={event.id ?? `${event.state}-${index}`}>
              <span
                aria-hidden="true"
                className={`update-status__dot update-status__dot--${eventTone(event.state)}`}
              />
              <span className="update-status__event-message">
                {event.message || jobStateLabel({ state: event.state })}
              </span>
              <span className="update-status__event-progress">{clampProgress(event.progress)}%</span>
            </li>
          ))}
        </ol>
      )}
    </section>
  );
}
