import type { AiChatThinkingState } from "../hooks/useAiChatThinking";
import type { AiChatThinking, AiChatThinkingStep } from "./aiChatTypes";
import { SegmentedControl } from "./ui";

/**
 * VD-209: AI Chat's thinking control and the collapsed thinking line.
 *
 * The control sits beside the model picker. It is drawn only when the
 * appliance says the chosen model can think, and it offers only the steps that
 * model takes. A model that cannot stop thinking gets no Off, and says so in
 * words beside the steps rather than in a tooltip.
 */

const STEP_LABELS: Record<AiChatThinkingStep, string> = {
  off: "Off",
  low: "Low",
  medium: "Medium",
  high: "High",
};

export function AiChatThinkingControl({ thinking }: { thinking?: AiChatThinkingState }) {
  const view = thinking?.view;
  if (!thinking || !view?.available || !thinking.step) return null;
  const alwaysOn = !view.steps.includes("off");
  return (
    <div className="ai-chat-thinking-control">
      <span aria-hidden="true" className="ai-chat-thinking-control__key">Thinking</span>
      <SegmentedControl
        label={alwaysOn ? "Thinking (this model always thinks)" : "Thinking"}
        onChange={thinking.choose}
        options={view.steps.map((step) => ({ value: step, label: STEP_LABELS[step] }))}
        value={thinking.step}
      />
      {alwaysOn && <span aria-hidden="true" className="ai-chat-thinking-control__note">Always on</span>}
      {thinking.error && <span className="ai-chat-thinking-control__error" role="alert">{thinking.error}</span>}
    </div>
  );
}

/** "Thought for 3.5 s": one decimal under ten seconds, whole seconds above. */
export function thoughtFor(seconds?: number): string {
  if (typeof seconds !== "number" || !Number.isFinite(seconds) || seconds < 0) return "Thinking summary";
  const shown = seconds < 10 ? seconds.toFixed(1).replace(/\.0$/, "") : String(Math.round(seconds));
  return `Thought for ${shown} s`;
}

/**
 * The collapsed line above an answer, expanding to the summary the provider
 * returned. Nothing at all when none came back (VD-209 item 4). The summary is
 * model output: it is rendered as text, so React escapes it, and never as
 * Markdown or HTML.
 */
export function AiChatThought({ thinking }: { thinking?: AiChatThinking }) {
  const summary = typeof thinking?.summary === "string" ? thinking.summary.trim() : "";
  if (!summary) return null;
  return (
    <details className="ai-chat-thought">
      <summary>{thoughtFor(thinking?.seconds)}</summary>
      <p className="ai-chat-thought__text">{summary}</p>
      {thinking?.truncated && <p className="ai-chat-thought__note">The summary was longer than AI Chat keeps; the end is not shown.</p>}
    </details>
  );
}
