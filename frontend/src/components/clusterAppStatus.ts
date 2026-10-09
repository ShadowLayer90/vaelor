/**
 * The one word + tone for each honest app `state` the backend derives from
 * Swarm's real update + placement facts (D3), shared by the per-service app rows
 * and the D4d grouped multi-service app row and its Manage breakdown so a state
 * reads identically wherever the cluster reports it.
 *
 * A researched app's aggregate `state` is the WORST member state
 * (unknown > unavailable > failed > rolled-back > updating > deploying >
 * healthy), so this map covers the same six named states, and anything it has no
 * word for — including the explicit `unknown` an unreadable member forces — reads
 * as an honest neutral "Unknown state", never a guessed Running/Deploying.
 */
export type StatusTone = "ok" | "warn" | "danger" | "neutral";

export interface RowStatus {
  label: string;
  tone: StatusTone;
}

const APP_STATUS: Partial<Record<string, RowStatus>> = {
  healthy: { label: "Running", tone: "ok" },
  deploying: { label: "Deploying", tone: "warn" },
  updating: { label: "Updating", tone: "warn" },
  "rolled-back": { label: "Rolled back", tone: "warn" },
  failed: { label: "Failed", tone: "danger" },
  unavailable: { label: "Unavailable", tone: "danger" },
};

const UNKNOWN_APP_STATUS: RowStatus = { label: "Unknown state", tone: "neutral" };

/** The row status for an honest backend app `state`, unknown for anything the
 *  map has no word for (an unenriched row, or an explicit worst-of `unknown`). */
export function statusForAppState(state?: string | null): RowStatus {
  return APP_STATUS[String(state ?? "")] ?? UNKNOWN_APP_STATUS;
}
