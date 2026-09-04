import type { StatusTone } from "../components/ui/status";

/**
 * The two readings `vaelor/accelerator_runtime.py` takes after a model server
 * answers, and what a screen is allowed to say about each.
 *
 * They exist because of one measurement: with no ROCm directory on
 * `LD_LIBRARY_PATH`, llama.cpp cannot resolve its HIP backend, **loads the CPU
 * backend and carries on** — no error, no warning, `/health` 200 throughout,
 * and 198 tokens per second of prefill instead of 1,835. The same shape
 * applies to the context window: a slot context above the model's training
 * context is capped rather than refused, and the overshoot is paid in resident
 * bytes no request can use.
 *
 * `grep -rn "acceleration" frontend/src` returned zero hits, so none of it had
 * ever reached a screen. The test class asserting the state is called
 * `SurfacedStateTests` — *"the state has to reach a screen"* — and it only
 * checked a Python dict.
 *
 * **Three states, and the two that are not verdicts are the ones to get
 * right.** `unknown` means the check could not be made and must render as
 * neither outcome. `unattributed` means the accelerator *is* holding a model's
 * worth of memory but a baseline read before this server started already
 * covered it — consistent with a model another engine loaded and with a
 * baseline taken too early, and the reading cannot tell those apart. It is an
 * unanswered question, **not a fault**, and painting it as one would invert
 * the module's purpose: asserting a 9x degradation on a healthy machine.
 */

export interface AccelerationReading {
  /** What was launched. */
  backend?: string;
  /** What was asked for, which is not always what was launched. */
  requested_backend?: string;
  gpu_layers_requested?: number;
  expected_accelerator?: boolean;
  compute_library?: string;
  gpu_memory_bytes?: number | null;
  baseline_bytes?: number | null;
  held_bytes?: number;
  /**
   * `differential` subtracted a baseline read while no managed model was
   * resident, so what is left is this deployment's own allocation.
   * `absolute` had no baseline: it can say the accelerator is holding a
   * model's worth of memory, but not whose.
   */
  basis?: string;
  /** `true`, `false`, or `null` for "the check could not be made". */
  in_use: boolean | null;
  state: string;
  detail: string;
  reference_measurement?: Record<string, unknown> | null;
}

export interface ContextReading {
  requested?: number;
  built?: number | null;
  trained?: number | null;
  total_slots?: number;
  /** `null` alongside `unknown: true`; never rendered as "capped". */
  capped: boolean | null;
  unknown?: boolean;
  state: string;
  detail: string;
  reference_measurement?: Record<string, unknown> | null;
}

export interface Verdict {
  /** The state the reading came back with, carried through unchanged. */
  state: string;
  tone: StatusTone;
  /** A few words a reader can scan. The `detail` says the rest. */
  headline: string;
  /**
   * True when this reading establishes a real degradation. Only these are
   * announced loudly; an unanswered question is not a finding, and a banner
   * for one is the same defect as a banner for a failure that is not there.
   */
  degraded: boolean;
}

const ACCELERATION_VERDICTS: Record<string, Omit<Verdict, "state">> = {
  accelerated: { tone: "success", headline: "Running on the accelerator", degraded: false },
  // Not a fault and not a degradation: this deployment asked for the CPU.
  cpu: { tone: "neutral", headline: "Running on the CPU by configuration", degraded: false },
  "cpu-fallback": { tone: "danger", headline: "Running on the CPU — the GPU library did not load", degraded: true },
  "not-offloaded": { tone: "danger", headline: "An accelerator was requested and not given", degraded: true },
  // The accelerator is in use; this engine cannot be shown to be what is using
  // it. An unanswered question, and it must not read as a fault.
  unattributed: { tone: "info", headline: "In use, but not attributable to this engine", degraded: false },
  unknown: { tone: "neutral", headline: "Not established", degraded: false },
};

const CONTEXT_VERDICTS: Record<string, Omit<Verdict, "state">> = {
  "as-requested": { tone: "success", headline: "Built the context window it asked for", degraded: false },
  // The server honoured the request and the request was too small. Warning
  // rather than success, because "built what it asked for" was the sentence
  // that made a window too small for the Assistant's own brief look healthy.
  "below-brief": { tone: "warning", headline: "The context window is too small for the Assistant", degraded: true },
  capped: { tone: "warning", headline: "The context window was capped", degraded: true },
  unknown: { tone: "neutral", headline: "Not established", degraded: false },
};

function verdictFor(
  table: Record<string, Omit<Verdict, "state">>,
  state: string | undefined,
): Verdict {
  const key = String(state || "unknown");
  // An unrecognised state is `unknown`, never a pass: a screen inventing a
  // verdict for a state it has not been taught is how "not established" turns
  // into "fine".
  return { state: key, ...(table[key] ?? table.unknown) };
}

export function accelerationVerdict(reading: AccelerationReading | null | undefined): Verdict | null {
  if (!reading) return null;
  return verdictFor(ACCELERATION_VERDICTS, reading.state);
}

export function contextVerdict(reading: ContextReading | null | undefined): Verdict | null {
  if (!reading) return null;
  return verdictFor(CONTEXT_VERDICTS, reading.state);
}

/**
 * How the accelerator reading was taken, in the reader's words.
 *
 * A differential and an absolute reading of the same fact answer different
 * questions, and a reading that does not say which one it answered invites two
 * callers to contradict each other over one fact. Only stated where it changes
 * what the reading is entitled to claim — a `cpu` engine has nothing to
 * attribute either way.
 */
export function basisNote(reading: AccelerationReading): string {
  if (!["accelerated", "unattributed"].includes(String(reading.state))) return "";
  return reading.basis === "differential"
    ? "Measured against a baseline taken before this server started, so what is left is this server's own allocation."
    : "Measured with no baseline, so it establishes what the accelerator is holding rather than which server put it there.";
}
