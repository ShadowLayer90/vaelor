import type { Health } from "../types";
import type { ConnectionState } from "./connectionState";

/**
 * What the health strip is allowed to say it checked.
 *
 * Home rendered, in prose, "No thermal, memory, storage, or service alerts."
 * `GET /health` computes its status from `cpu_temperature` and `memory_percent`
 * and nothing else — no storage input, no service input, and no GPU input — so
 * two of the four named categories were never evaluated. On a workstation with
 * the graphics processor at 100 °C and 108 W mid-inference the landing page
 * stated there were no thermal alerts. That is not an omission; it is a false
 * claim, and it is the most consequential line on the page because the sidebar
 * badge, the status pill and the Assistant's own health answer all derive from
 * the same reading.
 *
 * The rule this module enforces is narrow and absolute: **enumerate only what
 * was checked.** Widening the claim is the server's job. `/health` now serves
 * `checked` (`vaelor/health_evaluation.py`) and evaluates graphics, so the
 * sentence grows and shrinks with the sample rather than with this file.
 *
 * The rule has a second half that the first version of this module got wrong,
 * and it is the harder one: **a served answer always wins, including an empty
 * one.** `checked: []` means the sample could judge nothing; substituting a
 * hardcoded pair there re-created the over-claim in the one case where the
 * appliance had been explicit about its own ignorance. See `healthCoverage`.
 */
export type HealthCategory =
  | "processor"
  | "graphics"
  | "neural"
  | "memory"
  | "storage"
  | "services"
  | "enclosure";

const categoryNouns: Record<HealthCategory, string> = {
  processor: "processor",
  graphics: "graphics",
  neural: "neural accelerator",
  memory: "memory",
  storage: "storage",
  services: "service",
  enclosure: "enclosure",
};

/**
 * What an appliance too old to serve `checked` is known to have evaluated: a
 * thermal band against `cpu_temperature` and a utilisation band against
 * `memory_percent`.
 *
 * This is the floor for a **missing** field only. `vaelor/health_evaluation.py`
 * serves `checked` today, so on a current appliance this constant is never
 * used, and it must never stand in for a served answer — see `healthCoverage`.
 */
export const evaluatedHealthCategories: HealthCategory[] = ["processor", "memory"];

function isCategory(value: unknown): value is HealthCategory {
  return typeof value === "string" && value in categoryNouns;
}

/**
 * What this payload is entitled to claim.
 *
 * The distinction this type exists to force: **an empty `checked` array is an
 * answer, not a silence.** `health_evaluation.evaluate_health` returns
 * `checked: []` with `status: "healthy"` when no reading on the sample was
 * measurable, and the first version of this module treated that identically to
 * a missing field — falling back to a hardcoded `["processor", "memory"]`, so a
 * machine that had checked *nothing* told the reader two categories were clear.
 * That also made the "nothing is claimed" sentence below unreachable whenever
 * the route answered at all: the one line written to prevent the over-claim
 * could never run.
 *
 * - `reported` — the route named categories this module understands.
 * - `none` — the route explicitly evaluated nothing.
 * - `unnamed` — the route evaluated something whose name this build does not
 *   know. Claiming the floor here would assert two checks that may not have
 *   happened, so nothing is enumerated.
 * - `unreported` — no `checked` field at all: an older appliance, and the only
 *   case where the floor is justified.
 */
export type HealthCoverage =
  | { kind: "reported"; categories: HealthCategory[] }
  | { kind: "none" }
  | { kind: "unnamed" }
  | { kind: "unreported"; categories: HealthCategory[] };

export function healthCoverage(health: Health): HealthCoverage {
  const served = (health as { checked?: unknown }).checked;
  if (!Array.isArray(served)) {
    return { kind: "unreported", categories: evaluatedHealthCategories };
  }
  if (served.length === 0) return { kind: "none" };
  // The server orders `checked` as the sentence should read it, so that order
  // is kept rather than re-sorted into this module's own preference.
  const recognised = served.filter(isCategory);
  return recognised.length
    ? { kind: "reported", categories: recognised }
    : { kind: "unnamed" };
}

/** Which categories this health payload may state were clear. */
export function healthCategories(health: Health): HealthCategory[] {
  const coverage = healthCoverage(health);
  return "categories" in coverage ? coverage.categories : [];
}

function joinNouns(nouns: string[]): string {
  if (nouns.length <= 1) return nouns[0] ?? "";
  if (nouns.length === 2) return `${nouns[0]} or ${nouns[1]}`;
  return `${nouns.slice(0, -1).join(", ")}, or ${nouns.at(-1)}`;
}

/**
 * The sentence under the health title.
 *
 * When something is wrong the server's own reasons are shown — they name the
 * sensor. When nothing is wrong the reassurance enumerates the categories that
 * were examined, and no others.
 */
export function healthReassurance(health: Health): string {
  if (health.reasons.length) return health.reasons.join(" · ");
  if (health.status === "offline") {
    return "No readings have arrived yet, so nothing has been checked.";
  }
  const coverage = healthCoverage(health);
  if (coverage.kind === "none") {
    // Reachable, and it must be: this is what the appliance says about itself
    // when no reading on the sample could be judged.
    return "No reading on this sample could be checked, so nothing is claimed.";
  }
  if (coverage.kind === "unnamed") {
    return "No alerts were raised, though this page cannot name what was checked.";
  }
  return `No ${joinNouns(coverage.categories.map((category) => categoryNouns[category]))} alerts.`;
}

/**
 * The health claim on Home, and the one place it is decided.
 *
 * Measured during a telemetry outage: the hero read **"All systems
 * operational"** with a **red dot beside that sentence**, and the pill in the
 * page heading still said `OPERATIONAL`. The dot came from the connection
 * state and the sentence came from the last `/health` answer, which is #52's
 * defect one more time — a mark and the words beside it derived from two
 * sources — except here the words are a safety claim.
 *
 * Losing contact with a machine is not evidence that all its systems are
 * operational. It is evidence of nothing, and "nothing" is a third answer,
 * not a quiet fourth reading of the second-most-recent one.
 *
 * `paused` deliberately keeps the served verdict. A reader who collapsed a tab
 * group has not lost contact; nothing is being retried because nothing is
 * being asked, and the top bar already carries the age of the last sample.
 * Blanking the health there would be the alarm fatigue this project keeps
 * removing. Only `error` — readings not arriving while Vaelor is still asking
 * — and `unknown` — never arrived at all — withdraw the claim.
 */
export interface HealthClaim {
  title: string;
  detail: string;
  /** What the page-heading pill may say, painted from this same value. */
  pillStatus: Health["status"];
  /** True when the claim is the server's verdict rather than a withdrawal. */
  reported: boolean;
}

export function healthClaim(
  health: Health,
  connection: ConnectionState,
  machineNoun: string,
): HealthClaim {
  if (connection === "error") {
    return {
      title: "Health not known right now",
      detail: `Readings are not arriving from this ${machineNoun}, so nothing about it can be checked.`,
      pillStatus: "offline",
      reported: false,
    };
  }
  if (connection === "unknown" || health.status === "offline") {
    return {
      title: `Waiting for this ${machineNoun}`,
      detail: "No readings have arrived yet, so nothing has been checked.",
      pillStatus: "offline",
      reported: false,
    };
  }
  return {
    title: health.status === "healthy"
      ? "All systems operational"
      : `${health.status.charAt(0).toUpperCase()}${health.status.slice(1)} state`,
    detail: healthReassurance(health),
    pillStatus: health.status,
    reported: true,
  };
}
