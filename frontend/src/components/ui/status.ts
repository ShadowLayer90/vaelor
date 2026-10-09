import type { Health } from "../../types";

export const statusTones = ["neutral", "info", "success", "warning", "danger"] as const; // vocabulary: status-tone
export type StatusTone = (typeof statusTones)[number];
export type LegacyStatus = Health["status"] | "neutral" | "ready" | "active" | "available" | "completed" | "success" | "queued" | "running" | "pending" | "waiting" | "paused" | "failed" | "rejected" | "error" | "cancelled" | "superseded" | "unavailable";
export type OperationState = "idle" | "pending" | "success" | "warning" | "error";

/**
 * A model server that was asked and did not answer: read, and broken - the
 * States board's "Failed" - so it is red on every surface that says it. One
 * value, because four surfaces once painted the same two words grey, amber
 * and red.
 */
export const NOT_ANSWERING: { readonly label: string; readonly tone: StatusTone } = {
  label: "Not answering",
  tone: "danger",
};

const toneLabels: Record<StatusTone, string> = {
  neutral: "Not reported",
  info: "In progress",
  success: "Operational",
  warning: "Needs attention",
  danger: "Blocked",
};

const legacyStatusTones: Record<string, StatusTone> = {
  healthy: "success",
  degraded: "warning",
  critical: "danger",
  offline: "neutral",
  neutral: "neutral",
  ready: "success",
  active: "success",
  available: "success",
  completed: "success",
  success: "success",
  queued: "info",
  running: "info",
  pending: "info",
  waiting: "info",
  paused: "neutral",
  failed: "danger",
  rejected: "danger",
  error: "danger",
  cancelled: "neutral",
  superseded: "neutral",
  unavailable: "neutral",
};

export function statusTone(status: string | undefined): StatusTone {
  return status ? legacyStatusTones[status.toLowerCase()] ?? "neutral" : "neutral";
}

export function statusLabel(tone: StatusTone) {
  return toneLabels[tone];
}

export function operationTone(state: OperationState): StatusTone {
  if (state === "pending") return "info";
  if (state === "success") return "success";
  if (state === "warning") return "warning";
  if (state === "error") return "danger";
  return "neutral";
}
