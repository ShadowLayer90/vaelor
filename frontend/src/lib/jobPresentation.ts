import { toUserFacingError } from "./userFacingErrors";

/** States whose message may carry raw executor text. */
const TERMINAL_STATES = new Set(["failed", "rejected", "cancelled"]);

export interface PresentableJob {
  id: string;
  type: string;
  state: string;
  actor?: string;
  operation_state?: string;
  resource_liveness?: string;
  liveness?: string;
  health?: string;
  readiness?: string;
  attention?: boolean;
  retryable?: boolean;
  resolved_by_retry?: boolean;
  retry_depth?: number;
  message?: string;
  result?: Record<string, unknown>;
}

const TERMINAL_OPERATION_STATES = new Set(["completed", "healthy", "failed", "rejected", "cancelled", "superseded"]); // vocabulary: terminal-state
const SUCCESS_OPERATION_STATES = new Set(["completed", "healthy"]);

export type CanonicalJob = PresentableJob;

export type JobProjectionInput = Pick<PresentableJob, "state"> & Partial<Pick<
  PresentableJob,
  "operation_state" | "attention" | "retryable" | "readiness" | "liveness" | "message" | "result"
>>;

/** Use the server projection for operator decisions; the raw state fallback is
 * retained only for older API fixtures and pre-projection responses. */
export function canonicalOperationState(job: JobProjectionInput) {
  if (job.operation_state) return job.operation_state;
  if (job.state === "needs_input" || job.state === "needs-approval") return "needs_approval";
  if (["validating", "downloading", "starting", "running", "cancelling"].includes(job.state)) return "running"; // vocabulary: active-job-state
  if (job.state === "failed" && !jobNeedsAttention(job)) return "rejected";
  return job.state;
}

export function jobIsTerminal(job: JobProjectionInput) {
  return TERMINAL_OPERATION_STATES.has(canonicalOperationState(job));
}

export function jobIsSuccessful(job: JobProjectionInput) {
  return SUCCESS_OPERATION_STATES.has(canonicalOperationState(job)) || job.readiness === "ready";
}

export function jobIsRetryable(job: JobProjectionInput) {
  if (typeof job.retryable === "boolean") return job.retryable;
  return ["failed", "cancelled"].includes(canonicalOperationState(job));
}

export function jobCanCancel(job: JobProjectionInput) {
  const operation = canonicalOperationState(job);
  return !jobIsTerminal(job) && ["queued", "running", "waiting", "needs_approval", "paused"].includes(operation);
}

export function jobIsReady(job: JobProjectionInput) {
  return job.readiness === "ready" || jobIsSuccessful(job);
}

/*
 * `jobIsLive` stood here (#179). It read
 * `job.liveness === "live" || canonicalOperationState(job) === "healthy"` —
 * the console's own copy of a server rule, and the server rule is gone.
 * `healthy` is a terminal job state, the JobStore row is immutable once it is
 * reached and nothing re-reads the deployed resource, so "live" was a
 * present-tense claim derived from a fact about the past; `job_projection` now
 * derives `liveness` from the same partition the ledger counts `active` from,
 * and a finished job reports `stopped`.
 *
 * All four call sites were asking "did this deployment succeed", which
 * `jobIsSuccessful` answers from the state the server projected. Whether a
 * model is the one the Assistant is *using* is decided by `in_use` on the
 * installed-model list, which is read fresh rather than remembered.
 */


/**
 * One canonical name per job type.
 *
 * The same operation was shown under two different names one screen apart:
 * Activity called `compose.backup` "Create checkpoint" and `model.deploy`
 * "Switch AI model", while the Workloads activity list called them "Back up app
 * configuration" and "Start managed local AI". A beginner reading both screens
 * could not tell it was the same job. Where Activity already names a type, its
 * wording is the canonical one and is reproduced here exactly; every other
 * surface reads this table rather than inventing a second vocabulary.
 */
export const jobLabels: Record<string, string> = {
  "system.update.stage": "Download system updates",
  "system.update.apply": "Install system updates",
  "host.vnc.enable": "Set up host browser desktop",
  "host.docker.install": "Install Docker and Compose",
  "host.memory.optimize": "Apply memory policy",
  "host.web-research.manage": "Manage guarded web research",
  "model.inspect": "Check model compatibility",
  "model.download": "Download AI model",
  "model.deploy": "Switch AI model",
  "model.install_release": "Set up the on-device Assistant",
  "appliance.factory-reset": "Factory reset Vaelor",
  "appliance.portable-import": "Import saved appliance state",
  // The compose and cluster operations previously fell through to their raw
  // job-type slug, so one list mixed "Apply memory policy" with "compose
  // install" and "managed remove".
  "compose.install": "Install app",
  "compose.import": "Import Docker stack",
  "compose.validate": "Validate Docker stack",
  "compose.deploy": "Deploy application",
  "compose.pull": "Download app images",
  "compose.start": "Start app",
  "compose.stop": "Stop app",
  "compose.restart": "Restart app",
  "compose.remove": "Remove app",
  "compose.backup": "Create checkpoint",
  "compose.restore": "Restore app configuration",
  "compose.update": "Update application",
  "compose.draft": "Draft an app configuration",
  "checkpoint.restore": "Restore checkpoint",
  "managed.remove": "Remove managed resource",
  "application.research": "Research application",
  "agent.deploy": "Set up Deployment Copilot",
  "custom_agent.run": "Run custom agent",
  "agent.task": "Run custom agent",
  "cluster.initialize": "Set up cluster controller",
  "cluster.node.enroll": "Enrol cluster worker",
  "cluster.app.deploy": "Deploy app to cluster",
  "cluster.llm.deploy": "Deploy cluster model server",
};

const labels = jobLabels;

/**
 * Title-case a raw job type as a last resort, so slugs never reach a user.
 *
 * A job type nobody has named yet must still read the same on every screen, so
 * this reproduces the Activity fallback exactly: each word capitalised, not
 * just the first. Anything that acquires a real name belongs in `jobLabels`.
 */
export function humanizeJobType(type: string) {
  const words = String(type ?? "").replaceAll(".", " ").replaceAll("_", " ").trim();
  if (!words) return "Operation";
  return words.replace(/\b\w/g, (letter) => letter.toUpperCase());
}

export const jobLabel = (type: string) => labels[type] ?? humanizeJobType(type);

export function jobSummary(job: PresentableJob, fallback = "") {
  const operation = canonicalOperationState(job);
  if (operation === "failed" && job.type === "system.update.stage") {
    return "Update download failed before any packages were installed. Retry only if no newer download succeeded.";
  }
  if (operation === "failed" && job.type === "system.update.apply") {
    return "Update installation stopped safely. Review system health before retrying.";
  }
  const raw = job.message || fallback || "Setup operation is waiting for details.";
  const normalized = raw.replace(/\s+/g, " ").trim();
  if (/seteuid 42 failed|method gave invalid 400 uri/i.test(normalized)) {
    return "An earlier update download was blocked by package-service permissions.";
  }
  // Failures are rewritten for a reader; the original stays available through
  // jobTechnicalDetail for a "Technical details" disclosure.
  // Every terminal state can carry an executor exception, not just "failed":
  // a refused or cancelled operation reached the same message field.
  const summary = TERMINAL_STATES.has(operation)
    ? toUserFacingError(normalized).summary
    : normalized;
  return summary.length > 180 ? `${summary.slice(0, 177)}…` : summary;
}

/** The original failure text, when it was rewritten for display. */
export function jobTechnicalDetail(job: PresentableJob): string | undefined {
  if (!TERMINAL_STATES.has(canonicalOperationState(job))) return undefined;
  const raw = (job.message ?? "").replace(/\s+/g, " ").trim();
  if (!raw) return undefined;
  return toUserFacingError(raw).technical;
}

/** Cause and recovery guidance for a failed or refused operation, when known. */
export function jobRecoveryGuidance(job: PresentableJob) {
  const state = canonicalOperationState(job);
  if (state === "rejected" || (state === "failed" && !jobNeedsAttention(job))) {
    // A refusal is not a breakage: the request was understood and declined on
    // policy. Saying so is the difference between "try again" and "change
    // something first" — and Retry is deliberately not offered.
    return {
      cause: "Vaelor refused this request because it breaks a safety rule.",
      recovery: "Change the setting named above, then submit it again. Retrying it unchanged will be refused again.",
    };
  }
  if (state !== "failed") return undefined;
  const presented = toUserFacingError((job.message ?? "").replace(/\s+/g, " ").trim());
  if (!presented.cause && !presented.recovery) return undefined;
  return { cause: presented.cause, recovery: presented.recovery };
}

const expectedRejection =
  /^(Choose an available port|Choose a supported application|Choose a supported memory profile|Port \d+ is already in use|Type the project name to confirm|Confirm the reviewed|Custom Compose imports cannot|Custom services cannot|Every custom service needs|Compose must define|Docker rejected this Compose file|Enter a Compose configuration|The Compose project name is invalid)/i;

export function jobNeedsAttention(job: JobProjectionInput) {
  if (typeof job.attention === "boolean") return job.attention;
  if (job.operation_state) return job.operation_state === "failed";
  if (job.state !== "failed") return false;
  if (job.result?.code === "invalid_job_request") return false;
  return !expectedRejection.test((job.message ?? "").trim());
}

export function jobStateLabel(job: JobProjectionInput) {
  const state = canonicalOperationState(job);
  const operatorLabels: Record<string, string> = {
    queued: "queued",
    running: "in progress",
    waiting: "waiting",
    paused: "paused",
    needs_approval: "needs approval",
    completed: "completed",
    healthy: "completed",
    failed: "failed",
    rejected: "rejected",
    cancelled: "cancelled",
    superseded: "superseded",
  };
  return operatorLabels[state] ?? state.replaceAll("_", " ");
}
