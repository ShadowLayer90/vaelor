import { useState } from "react";
import "../styles/apps-manage.css";
import { AppsFacts } from "./appsKit";
import { Icon, ICON_SIZE } from "./Icon";
import { Button, Notice } from "./ui";
import { MODEL_SURFACE_NAMES, RUNTIME_MODE_NAMES, type RuntimeMode } from "../lib/modelModes";

export type { RuntimeMode };

/** What the switch panel needs from a downloaded-model row. */
export interface ModelSwitchTarget {
  name: string;
  /** "ai-chat" or "assistant": the tier the deploy will serve (the backend's rule). */
  surface?: string;
  /** The context window the GPU route runs whatever profile is chosen (W4d-D28). */
  serving_context?: number;
  /** Why this model cannot be switched to now, in the backend's own words. */
  switch_refusal?: string;
}

/** How a model's serving tier reads to a person. */
export function modelTierLabel(model: { surface?: string }): string {
  return model.surface === "ai-chat" ? MODEL_SURFACE_NAMES["ai-chat"] : MODEL_SURFACE_NAMES.assistant;
}

const MODES: ReadonlyArray<readonly [RuntimeMode, string, string]> = [
  ["efficient", RUNTIME_MODE_NAMES.efficient, "2K context · most RAM left for apps"],
  ["balanced", RUNTIME_MODE_NAMES.balanced, "4K context · recommended"],
  ["quality", RUNTIME_MODE_NAMES.quality, "Up to 8K context · automatically reduced if RAM is tight"],
];

/**
 * "Use model": choose how it runs, review, then switch.
 *
 * W4d-D28 (LESSONS 5): the panel offered 2K / 4K / 8K for a model the GPU
 * route runs at its own window (the 27B ran at 131,072 under "Balanced 4K"),
 * and "Review and switch model" switched with no review. A model with a
 * `serving_context` now states that window and offers no profile, and the
 * first press opens a review of exactly what will happen; only its second
 * button queues the switch. A `switch_refusal` (Mode B holds AI Chat) is
 * said before any press and the switch is not offered.
 */
export function ModelSwitchPanel({
  busy,
  model,
  onClose,
  onSwitch,
}: {
  busy: boolean;
  model: ModelSwitchTarget;
  onClose: () => void;
  /** `null` when the model's runtime decides its window and no profile applies. */
  onSwitch: (mode: RuntimeMode | null) => void;
}) {
  const [mode, setMode] = useState<RuntimeMode>("balanced");
  const [reviewing, setReviewing] = useState(false);
  const tier = modelTierLabel(model);
  const fixed = model.serving_context ? model.serving_context.toLocaleString("en-US") : "";
  const chosen = MODES.find(([id]) => id === mode);
  const refused = Boolean(model.switch_refusal);
  const refusedReason = refused ? "The switch is refused for the reason above." : undefined;
  return (
    <section className="ui-card manage-panel" aria-labelledby="model-switch-title">
      <header className="manage-panel__head">
        <div className="manage-panel__titles">
          <span className="apps-eyebrow">{tier} model</span>
          <h2 id="model-switch-title">Use {model.name}</h2>
          <p>{fixed
            ? `Runs on the graphics processor with a ${fixed}-token context window, the window its runtime uses whatever memory profile is chosen.`
            : "Vaelor will size the runtime from current RAM and preserve memory for the operating system."}</p>
        </div>
        <Button className="manage-panel__close" onClick={onClose} variant="quiet">Close</Button>
      </header>
      <div className="manage-panel__body">
        {model.switch_refusal ? <Notice severity="warning">{model.switch_refusal}</Notice> : null}
        {reviewing ? (
          <div aria-label="Review the switch" role="group">
            <AppsFacts rows={[
              { label: "Model", value: model.name },
              { label: "Serves", value: tier },
              { label: "Context window", value: fixed ? `${fixed} tokens` : `${chosen?.[1]} (${chosen?.[2]})` },
              { label: "What happens", value: `The current local ${tier} model stops while this one loads. If it does not pass its health check, Vaelor restores the previous one.` },
            ]} />
          </div>
        ) : fixed ? (
          <p className="manage-tool__quiet">No memory profile is offered: the GPU runtime uses its own window.</p>
        ) : (
          <div className="manage-mode-grid" role="group" aria-label="How it runs">
            {MODES.map(([id, name, description]) => (
              <Button aria-pressed={mode === id} className={mode === id ? "manage-mode is-on" : "manage-mode"} key={id} onClick={() => setMode(id)} variant="quiet">
                <strong>{name}</strong><span>{description}</span>
              </Button>
            ))}
          </div>
        )}
      </div>
      <footer className="manage-panel__foot">
        <span className="manage-panel__shield"><Icon name="shield" size={ICON_SIZE.inline} /> Switching briefly restarts local AI. If the replacement fails, Vaelor restores the previous managed service.</span>
        {reviewing ? (
          <div className="manage-panel__buttons">
            <Button disabled={busy} onClick={() => setReviewing(false)} variant="quiet">Back</Button>
            <Button busy={busy} disabledReason={refusedReason} onClick={() => onSwitch(fixed ? null : mode)} variant="primary">{busy ? "Switching…" : "Switch model"}</Button>
          </div>
        ) : (
          <Button disabled={busy} disabledReason={refusedReason} onClick={() => setReviewing(true)} variant="primary">Review switch</Button>
        )}
      </footer>
    </section>
  );
}
