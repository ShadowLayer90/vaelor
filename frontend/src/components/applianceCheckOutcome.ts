import type { NoticeSeverity } from "./ui";
import type { AgentTask } from "./agentTypes";

export interface ApplianceCheckOutcome {
  message: string;
  severity: NoticeSeverity;
}

/**
 * What the banner says, derived from the run's own state.
 *
 * The old copy was set purely on the POST resolving, so "The appliance check
 * finished. Any proposed repair still requires separate approval." sat above a
 * run that was `blocked` and had executed nothing at all. The server already
 * returns the truthful state on the create response; this reads it.
 *
 * Every message names History, because that is where the run is. "Below" was
 * written when Ask carried the archive; once the archive became its own tab,
 * a banner on Ask pointed at empty space, and a check the reader had just
 * created looked like nothing had happened. The caller now moves the reader to
 * History with the Checks filter before showing any of these, so "below" is
 * true again.
 */
export function applianceCheckOutcome(
  task: Pick<AgentTask, "state" | "error" | "result"> | null | undefined,
): ApplianceCheckOutcome {
  const state = String(task?.state ?? "");
  const detail = (task?.error || task?.result?.summary || "").trim();
  const because = detail ? ` ${detail}` : "";
  switch (state) {
    case "needs_approval":
      return {
        message: "Appliance check saved, and it needs your approval before it runs. "
          + "History is showing Checks — review and approve it below.",
        severity: "info",
      };
    case "completed":
    case "archived":
      return {
        message: "The appliance check finished. History is showing Checks — its evidence is "
          + "below. Any proposed repair still requires separate approval.",
        severity: "info",
      };
    case "blocked":
      return {
        // A blocked run executed nothing. Reading it as "finished" is the
        // difference between "my appliance is fine" and "nobody looked".
        message: `The appliance check was blocked before it could run, so nothing was checked and nothing was changed.${because}`,
        severity: "warning",
      };
    case "failed":
      return {
        message: `The appliance check failed and made no changes.${because} History is showing Checks — you can retry it below.`,
        severity: "warning",
      };
    case "cancelled":
      return { message: "The appliance check was cancelled. Nothing was changed.", severity: "info" };
    case "triage":
    case "ready":
    case "running":
      return {
        // Still going is not the same as timed out. The run keeps its own
        // lifecycle; the history below is where it lands.
        message: "The appliance check is still running. History is showing Checks — its findings appear below as soon as it finishes.",
        severity: "info",
      };
    default:
      return {
        message: state
          // Every underscore, not only the first.
          ? `The appliance check is ${state.replaceAll("_", " ")}. History is showing Checks — its result appears below.`
          : "The appliance check was accepted. History is showing Checks — its result appears below.",
        severity: "info",
      };
  }
}

/**
 * The reader stopped waiting, or the browser did. Neither means the run failed:
 * the appliance keeps going and the result lands in the history either way.
 *
 * These two keep the reader on Ask with their question restored, so unlike the
 * outcomes above they must not say "below": there is no history under the
 * question box. They name the tab instead.
 */
export function applianceCheckInterrupted(aborted: boolean): ApplianceCheckOutcome {
  return {
    message: aborted
      ? "Stopped waiting for this check. If the appliance already accepted it, it finishes in the background and appears under History, in Checks."
      : "This check is taking longer than expected, so Vaelor stopped waiting for it. It keeps running and appears under History, in Checks, when it finishes.",
    severity: "info",
  };
}
