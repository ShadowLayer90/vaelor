/**
 * What a restart would cut off (`GET /power/restart-impact`, VD-133 1.5.5 #5).
 *
 * In 1.5.4 a controller restart cut off a running worker software update,
 * which then showed as failed; the dialog had said nothing. The backend reads
 * the same set of jobs the restart fails (`ACTIVE_JOB_STATES`) and an update
 * still being applied; this module only types the answer and builds the
 * acknowledgement the power route checks.
 */

export interface RestartImpactItem {
  id: string;
  type: string;
  message: string;
  progress: number | null;
}

export interface RestartImpact {
  /** `false`: the job store could not be read, so nothing is known - never "nothing runs". */
  readable: boolean;
  reason: string;
  /** `interrupts`: this controller (jobs are failed). `waits`: a worker (its restart waits behind them). */
  effect: "interrupts" | "waits";
  items: RestartImpactItem[];
  /** Jobs of other accounts, counted and never described to an operator. */
  others: number;
  /** An opaque name for exactly those jobs, sent back so the route can tell the same jobs from as many new ones (review L1). */
  others_digest: string;
  wait_limit_seconds: number | null;
}

/** The machine the impact is asked about: the controller, or a worker's node id. */
export function restartImpactPath(machine: string): string {
  return `/power/restart-impact?machine=${encodeURIComponent(machine)}`;
}

export function readRestartImpact(value: unknown): RestartImpact | null {
  if (!value || typeof value !== "object") return null;
  const impact = value as Partial<RestartImpact>;
  if (typeof impact.readable !== "boolean" || !Array.isArray(impact.items)) return null;
  if (impact.effect !== "interrupts" && impact.effect !== "waits") return null;
  return {
    readable: impact.readable,
    reason: typeof impact.reason === "string" ? impact.reason : "",
    effect: impact.effect,
    items: impact.items.filter((item): item is RestartImpactItem => Boolean(item) && typeof item.id === "string"),
    others: typeof impact.others === "number" ? impact.others : 0,
    others_digest: typeof impact.others_digest === "string" ? impact.others_digest : "",
    wait_limit_seconds: typeof impact.wait_limit_seconds === "number" ? impact.wait_limit_seconds : null,
  };
}

/** Whether the owner must tick the acknowledgement before the restart can be sent. */
export function needsAcknowledgement(impact: RestartImpact | null): boolean {
  return impact !== null && impact.effect === "interrupts" && (impact.items.length > 0 || impact.others > 0);
}

/** What the power route is told the owner saw (`restart_interruptions.unacknowledged`). */
export function acknowledgement(impact: RestartImpact | null): { acknowledged: string[]; acknowledged_others: string } {
  return {
    acknowledged: impact?.items.map((item) => item.id) ?? [],
    acknowledged_others: impact?.others_digest ?? "",
  };
}

/** The refusal the power route sends when work started after the dialog read. */
export const RESTART_INTERRUPTS_CODE = "restart_interrupts_running_work";
