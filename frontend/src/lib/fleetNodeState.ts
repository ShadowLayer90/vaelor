import type { StatusTone } from "../components/ui/status";
import type { RowStatus } from "../components/clusterAppStatus";

/**
 * The words the Fleet screen uses for a machine's usable state (ACC-091).
 *
 * The capacity ledger derives one `state` per node (contract 3): the
 * controller, a joined Ready active worker ("ready"), and every way a worker
 * can be unable to take work - enrolled but never joined, drained, paused,
 * offline, missing from the cluster's own list, or unknown (its cluster state
 * could not be read). Each has its own
 * owner-readable word and tone here, and ANY state this map has no word for -
 * including a row too old to carry one - reads "Unknown", never Ready or Idle
 * (LESSONS pattern 2: a label derived from a boolean lies for every state the
 * boolean does not name).
 *
 * "ready" has no fixed word: a ready machine is then described by what it
 * holds (Low space / Idle / Ready), which the machine card decides.
 */
export interface NodeStateWord {
  label: string;
  tone: StatusTone;
  /** How the fleet headline names the state when it leaves the machine out. */
  phrase: string;
}

const NODE_STATE_WORDS: Partial<Record<string, NodeStateWord>> = {
  controller: { label: "Controller", tone: "info", phrase: "not taking work" },
  ready: { label: "Ready", tone: "success", phrase: "not taking work" },
  "not-joined": { label: "Not joined", tone: "warning", phrase: "not joined" },
  drained: { label: "Drained", tone: "warning", phrase: "drained" },
  paused: { label: "Paused", tone: "warning", phrase: "paused" },
  offline: { label: "Offline", tone: "danger", phrase: "offline" },
  // A joined machine the cluster's own list no longer carries.
  missing: { label: "Left cluster", tone: "danger", phrase: "no longer in the cluster" },
};

const UNKNOWN_NODE_STATE: NodeStateWord = {
  label: "Unknown",
  tone: "neutral",
  phrase: "in an unknown state",
};

/** The word, tone and headline phrase for a ledger node state. */
export function nodeStateWord(state: string | undefined | null): NodeStateWord {
  return NODE_STATE_WORDS[String(state ?? "")] ?? UNKNOWN_NODE_STATE;
}

/**
 * A deployment-row tone (ok/warn/danger/neutral) as the shared status-pill
 * tone, so a workload chip on a machine card reads the same state in the same
 * colour as its row on the Deployments tab.
 */
export function pillToneForRow(status: RowStatus): StatusTone {
  if (status.tone === "ok") return "success";
  if (status.tone === "warn") return "warning";
  if (status.tone === "danger") return "danger";
  return "neutral";
}
