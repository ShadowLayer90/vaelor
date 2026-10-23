import { exactTime } from "./format";
import type { StatusTone } from "../components/ui/status";

/*
 * VD-133 1.5.5 #4: how a failed operation reads once the server says it no
 * longer needs the owner. In 1.5.4 a worker update cut off by a restart was
 * brought up to date by the automatic recheck four minutes later, and still
 * showed red: Activity read the server's verdict, Apps read the row's raw
 * state, and Apps ignored the owner's dismissal. Activity, Fleet's activity
 * and the Apps rows now all render these words from the same server fields
 * (`needs_attention`, `resolution`, `dismissal`, `interrupted_by_restart`) and
 * decide nothing themselves.
 */

/** What resolved a failure: a successful retry or a later run of the same operation. `at` is epoch ms. */
export interface OperationResolution {
  kind: string;
  at: number | null;
  /** The resolving run, as an operation id (Activity) or a job id (`/jobs`). */
  by_operation_id?: string;
  by?: string;
}

export interface OutcomeInput {
  needs_attention?: boolean;
  resolution?: OperationResolution | null;
  /** `dismissed_at` is epoch SECONDS, unlike a job's own millisecond times. */
  dismissal?: { dismissed_at: number; dismissed_by: string } | null;
  interrupted_by_restart?: boolean;
}

/** The word for a failure: work a restart cut short was interrupted, not broken. */
export function failureWord(input: OutcomeInput): string {
  return input.interrupted_by_restart === true ? "Interrupted by a restart" : "Failed";
}

/** "Resolved by a later successful run of the same operation at <time>." */
export function resolutionNote(resolution: OperationResolution): string {
  const when = typeof resolution.at === "number" ? ` at ${exactTime(resolution.at)}` : "";
  const by = resolution.kind === "retry" ? "a successful retry" : "a later successful run of the same operation";
  return `Resolved by ${by}${when}.`;
}

/**
 * A failure the server says is settled, as a pill and a sentence; `null`
 * while it still needs the owner (or the server has not said). The record
 * is kept and shown: resolved is history, not deletion.
 */
export function settledFailure(input: OutcomeInput): { label: string; tone: StatusTone; note: string } | null {
  if (input.needs_attention !== false) return null;
  if (input.resolution) return { label: "Resolved", tone: "success", note: resolutionNote(input.resolution) };
  const dismissal = input.dismissal;
  if (dismissal) {
    return {
      label: "Dismissed",
      tone: "neutral",
      note: `Dismissed by ${dismissal.dismissed_by || "an operator"} at ${exactTime(dismissal.dismissed_at * 1000)}.`,
    };
  }
  return null;
}
