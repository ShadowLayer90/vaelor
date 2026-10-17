import { Icon } from "./Icon";

/** `upgrade_progress.PHASE_STATES`, copied: the Python one is right if they differ. */
export const PHASE_STATES = ["pending", "running", "done", "skipped", "held-back", "failed"] as const; // vocabulary: update-phase-state
export type PhaseState = (typeof PHASE_STATES)[number];

/** One build step inside the "Build the speed-ups" phase. */
export interface UpdateStep {
  label: string;
  state: PhaseState | string;
  reason: string;
}

/** One phase of an update, as `upgrade_progress.progress` derives it. */
export interface UpdatePhase {
  key: string;
  label: string;
  state: PhaseState | string;
  reason: string;
  steps?: UpdateStep[];
}

/** What one reloaded model now serves with (`gpu_pool_refresh.serving_summary`). */
export interface ServedModel {
  name: string;
  read?: boolean;
  drafter?: boolean;
  guesser?: "4-bit" | "full-size" | "none" | string;
  expert_kernels?: "vaelor" | "vllm" | string;
  step_merging?: boolean;
  notes?: string[];
}

/** `GET /upgrade`'s `progress`: the latest update, phase by phase. */
export interface UpgradeProgress {
  job_id: string;
  from_version: string;
  to_version: string;
  started_at: number;
  reached: boolean;
  phases: UpdatePhase[];
  finished: boolean;
  failed: boolean;
  model: { models: ServedModel[] } | null;
  /** Whether anything still works on this update (`upgrade_progress`, R114). */
  active?: boolean;
}

/** The states that end a phase: every one after "running" (`upgrade_progress.ENDED`). */
const ENDED: ReadonlySet<string> = new Set(PHASE_STATES.slice(2));

/** Whether a phase has ended (done, skipped, held back or failed). */
export function phaseEnded(state: string): boolean {
  return ENDED.has(state);
}

/** The word a state is read as; the icon beside it never carries it alone. */
const STATE_WORD: Record<string, string> = {
  pending: "Waiting",
  running: "In progress",
  done: "Done",
  skipped: "Skipped",
  "held-back": "Held back",
  failed: "Failed",
};

function stateWord(state: string): string {
  return STATE_WORD[state] ?? "Not known";
}

function Marker({ state }: { state: string }) {
  if (state === "done") return <Icon name="done" size={16} />;
  if (state === "held-back" || state === "failed") return <Icon name="alert" size={16} />;
  return <span aria-hidden="true" className="sys-run__dot" />;
}

/**
 * The update's phases, in order, each with its state and plain reason
 * (R104). `restarting` is the page's own knowledge that the appliance stopped
 * answering after the update was accepted: the restart phase is then shown
 * as in progress, since the server that would say so is the one restarting.
 */
export function UpdatePhases({ phases, restarting = false }: { phases: UpdatePhase[]; restarting?: boolean }) {
  // R111: a screen reader hears each phase change, not only the headline.
  const current = phases.find((phase) => !phaseEnded(phase.state));
  const announce = restarting
    ? "Restart Vaelor: In progress"
    : current ? `${current.label}: ${stateWord(current.state)}` : "Every step of the update has ended";
  return (
    <>
    <p aria-atomic="true" aria-live="polite" className="sr-only">{announce}</p>
    <ol className="sys-run__phases">
      {phases.map((phase) => {
        const state = restarting && phase.key === "restart" && !phaseEnded(phase.state)
          ? "running"
          : restarting && phase.key === "install" && phase.state === "running" ? "done" : phase.state;
        const reason = restarting && phase.key === "restart" && state === "running"
          ? "Vaelor is restarting. This page reconnects by itself."
          : phase.reason;
        return (
          <li className={`sys-run__phase sys-run__phase--${state}`} key={phase.key}>
            <Marker state={state} />
            <div className="sys-run__body">
              <div className="sys-run__head">
                <span className="sys-run__label">{phase.label}</span>
                <span className="sys-run__state">{stateWord(state)}</span>
              </div>
              {/* A phase with steps says its reasons on the steps, once. */}
              {reason && !(phase.steps && phase.steps.length > 0) && <p className="sys-run__reason">{reason}</p>}
              {phase.steps && phase.steps.length > 0 && (
                <ul className="sys-run__steps">
                  {phase.steps.map((step) => (
                    <li className={`sys-run__step sys-run__phase--${step.state}`} key={step.label}>
                      <span className="sys-run__label">{step.label}</span>
                      <span className="sys-run__state">{stateWord(step.state)}</span>
                      {step.state !== "done" && step.reason && <p className="sys-run__reason">{step.reason}</p>}
                    </li>
                  ))}
                </ul>
              )}
            </div>
          </li>
        );
      })}
    </ol>
    </>
  );
}

function guesserWords(model: ServedModel): string {
  if (model.guesser === "4-bit") return "The 4-bit guesser";
  if (model.guesser === "full-size") return "The full-size guesser";
  return "None";
}

/**
 * What each reloaded model now runs with, and what was held back and why.
 * `name` is the deployment's internal id, so it is never the heading: one
 * model is "The cluster model", several are told apart by position.
 */
export function ServedModels({ models }: { models: ServedModel[] }) {
  return (
    <div className="sys-run__models">
      {models.map((model, index) => (
        <section className="sys-run__model" key={model.name}>
          <h4>{models.length === 1 ? "The cluster model now runs with" : `Cluster model ${index + 1} now runs with`}</h4>
          {model.read === false ? (
            <p className="sys-run__reason">Its settings could not be read back after the reload.</p>
          ) : (
            <dl className="sys-run__served">
              <div><dt>Guesser</dt><dd>{guesserWords(model)}</dd></div>
              <div><dt>GPU routines</dt><dd>{model.expert_kernels === "vaelor" ? "Vaelor's expert kernels" : "vLLM's own expert kernels"}</dd></div>
              <div><dt>Step merging</dt><dd>{model.step_merging ? "On" : "Off"}</dd></div>
            </dl>
          )}
          {(model.notes ?? []).length > 0 && (
            <ul className="sys-run__notes">
              {(model.notes ?? []).map((note) => <li key={note}>{note}</li>)}
            </ul>
          )}
        </section>
      ))}
    </div>
  );
}
