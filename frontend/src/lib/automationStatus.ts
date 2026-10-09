import type { StatusTone } from "../components/ui/status";

/**
 * The status the server derives for a schedule or an alert rule
 * (`vaelor/automation_status.py`). The server is the one owner of the
 * question "what is this doing?"; this module only maps its stable `state`
 * word to a tone. The pill used to come from the `enabled` flag, so a schedule
 * whose runs all failed, a one-time schedule that had already run and a rule
 * on a machine that stopped reporting all read green (ACC-086/135/139).
 */
export interface AutomationItemStatus {
  state: string;
  label: string;
  detail: string;
}

const STATE_TONES: Record<string, StatusTone> = {
  scheduled: "success",
  watching: "success",
  due: "info",
  waiting: "info",
  paused: "neutral",
  finished: "neutral",
  not_reporting: "warning",
  delivery_failing: "warning",
  failing: "danger",
  failed_once: "danger",
};

/** A state that means the item will not act on its own. */
const IDLE_STATES = new Set(["paused", "finished", "failed_once"]);

/** A one-time schedule that has had its one run, succeeded or not: it cannot be enabled again. */
export function scheduleIsSpent(status: AutomationItemStatus | undefined): boolean {
  return status?.state === "finished" || status?.state === "failed_once";
}

/**
 * States that are about a reading not yet taken. The States board draws "not
 * read" grey with a hollow dot (VD-200), so the pill says it as unread rather
 * than in a tone that could be taken for a reading.
 */
const UNREAD_STATES = new Set(["waiting"]);

/** The pill for a schedule or rule; an item with no status says so, never green. */
export function automationPill(status: AutomationItemStatus | undefined): { tone: StatusTone; label: string; reading?: "unread" } {
  if (!status) return { tone: "neutral", label: "Status unknown" };
  return {
    tone: STATE_TONES[status.state] ?? "neutral",
    label: status.label || "Status unknown",
    ...(UNREAD_STATES.has(status.state) ? { reading: "unread" as const } : {}),
  };
}

/** Whether a schedule or rule will still act on its own (paused and finished do not). */
export function automationIsActive(item: { enabled: boolean; status?: AutomationItemStatus }): boolean {
  if (item.status) return !IDLE_STATES.has(item.status.state);
  return item.enabled;
}

/** Whether the next-run time means anything: only while the item is scheduled or due. */
export function showsNextRun(status: AutomationItemStatus | undefined): boolean {
  return status?.state === "scheduled" || status?.state === "due";
}

/**
 * A rule's signal in words. The server sends `signal_label` from its one
 * table (`TRIGGER_SOURCES`), so no second copy of the labels lives here; a
 * record without one reads as unrecognised, never as its raw key.
 */
export function signalLabel(rule: { signal_label?: string }): string {
  return rule.signal_label || "An unrecognised signal";
}

/** A rule's comparison in words: `>=` is "at or above", `<=` "at or below". */
export function operatorLabel(operator: string): string {
  if (operator === ">=") return "at or above";
  if (operator === "<=") return "at or below";
  return "compared with";
}

/** A schedule's kind in words. */
export function scheduleKindLabel(kind: string): string {
  if (kind === "interval") return "Repeats";
  if (kind === "once") return "One time";
  return "Schedule";
}
