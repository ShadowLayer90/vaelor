import { ApiError } from "../lib/api";
import { modelDisplayName, modelIdentity } from "../lib/modelIdentity";
import type { AiChatConnection } from "./aiChatTypes";
import { Select } from "./ui";

/**
 * Choosing what you want to talk to.
 *
 * Extracted from `AiChat.tsx` without any change of behaviour, because that
 * file stood at **999 lines** and `styles/ai-chat.css` at **exactly 1,000** —
 * one line and none of headroom respectively, against the 1,000-line limit in
 * `CLAUDE.md`. Nothing further can be built here until the picker has its own
 * file, and this is that file. The rules that came with it are in
 * `styles/ai-chat-picker.css` for the same reason.
 *
 * Per VD-007 the user picks the AI Chat model — AI Chat is their tool, not
 * infrastructure — and AI Chat keeps **both** local model install and external
 * providers. Neither half may be simplified away here.
 */

/**
 * The model this failure belongs to, if the appliance blamed one.
 *
 * `chat_model_rejected`, `chat_model_timeout` and `chat_model_invalid_response`
 * are the appliance saying the provider took the request and this model could
 * not serve it; `request_timeout` is the browser giving up on the same wait.
 * `chat_connection_unreachable` and the rest are not the model's fault and must
 * not mark it, or a server that was briefly down would condemn every model on
 * it.
 */
export function blamedModel(error: unknown, model: string): string {
  if (!model || !(error instanceof ApiError)) return "";
  return error.code.startsWith("chat_model_") || error.code === "request_timeout"
    ? model
    : "";
}

/**
 * How a model is named in the picker, and how a failed one is marked.
 *
 * The name comes from `modelIdentity`, so a managed local server offering
 * `/models/Qwen3-1.7B-Q4_K_M.gguf` presents `Qwen3 1.7B · Q4_K_M` — a choice
 * rather than a file path. A provider's own model id is not a path and is
 * passed through unchanged.
 */
export function modelOptionLabel(model: string, failed: boolean): string {
  const name = modelDisplayName(model);
  return failed ? `${name} — last request failed` : name;
}

/**
 * Five states, and why none of them may be folded into another.
 *
 * The picker used to have two: enabled with a list, or disabled with the words
 * "Choose a model". Everything that was not a usable list landed in the second
 * one — so a machine that had not answered yet, a machine with no provider
 * connected at all, and a connected provider reporting an empty catalogue were
 * one indistinguishable grey box inviting a choice that could not be made. The
 * first of those is the `LightingControl` defect exactly: a control that states
 * a settled fact about data it has not read.
 *
 * - `reading`    — nothing has been asked yet, or the answer is being replaced.
 *                  Not knowing is its own answer and gets its own display.
 * - `unconnected`— asked and answered: no provider is connected. The fix is to
 *                  install a local model or add an external provider, and the
 *                  sentence names both because VD-007 keeps both.
 * - `empty`      — a provider is connected and reports no model. That is the
 *                  provider's problem, not the user's choice, and naming the
 *                  connection is what makes it actionable.
 * - `unchosen`   — models are available and none is selected. A real choice.
 * - `chosen`     — a model is selected. The only state where sending is
 *                  expected to work.
 */
export type ModelPickerState = "reading" | "unconnected" | "empty" | "unchosen" | "chosen";

export type ModelPickerInput = {
  /**
   * `undefined` means Vaelor has not finished asking; `null` means it asked and
   * there is no active connection. The distinction is carried in the type
   * because collapsing it into a boolean is what produced the two-state picker.
   */
  connection: AiChatConnection | null | undefined;
  models: string[];
  value: string;
};

export type ModelPickerPresentation = {
  state: ModelPickerState;
  /** Text of the leading option, which is never a choice the user can make. */
  placeholder: string;
  /** Sentence under the control. One state, one sentence. */
  message: string;
  /**
   * The exact identifier the server gave, when it is not what the sentence
   * says. Empty whenever the two are the same string, so the path is shown
   * once and only where it adds something: a reader debugging a model server
   * needs `/models/Qwen3-1.7B-Q4_K_M.gguf`, and nobody else does.
   */
  identifier: string;
  /** Whether the control accepts input in this state. */
  interactive: boolean;
};

/**
 * The model actually in use, reconciled against what the connection offers.
 *
 * A remembered model the live list no longer contains — `gpt-oss-sg:20b`,
 * saved when a different local server answered AI Chat, now that a GPU server
 * offers only Qwen — is a phantom. The dropdown cannot select an option that
 * is not there, so the browser already falls back to showing the first real
 * option while the caption, reading the raw value, still named the ghost: the
 * dropdown and the sentence beneath it disagreed. Both must name the same
 * model. An empty value is left empty — no choice has been made yet — and a
 * value the connection still offers is left exactly as chosen; only a
 * non-empty value absent from a non-empty list falls back to the first
 * available model, which is the one the dropdown already shows.
 *
 * This does not touch a genuinely historical selection: a stored conversation
 * that recorded a now-absent model still reaches here, but its recorded id is
 * reconciled to the served model for display just the same, and the failure
 * itself is carried by the `blamed` branch, which reads the same reconciled
 * value.
 */
export function effectiveModel(models: string[], value: string): string {
  return value && models.length && !models.includes(value) ? models[0] : value;
}

export function modelPickerState({ connection, models, value }: ModelPickerInput): ModelPickerState {
  if (connection === undefined) return "reading";
  if (connection === null) return "unconnected";
  if (!models.length) return "empty";
  return value ? "chosen" : "unchosen";
}

/**
 * `blamed` marks the selected model as one the appliance has already refused.
 * It changes the sentence and nothing else: the failure itself is reported by
 * the banner, and repeating that text here as a second `role="alert"` would
 * announce one failure twice to a screen reader.
 */
export function modelPickerPresentation(
  input: ModelPickerInput,
  blamed = false,
): ModelPickerPresentation {
  // The reconciled selection drives every branch below, so the state, the
  // sentence and the dropdown value can never name three different models.
  const selected = effectiveModel(input.models, input.value);
  const state = modelPickerState({ ...input, value: selected });
  const connectionLabel = input.connection?.label ?? "";
  switch (state) {
    case "reading":
      return {
        state,
        placeholder: "Reading the available models",
        message: "Vaelor is still asking this machine what it can run.",
        identifier: "",
        interactive: false,
      };
    case "unconnected":
      return {
        state,
        placeholder: "No provider connected",
        message: "Install a local model or add an external provider in Details, then pick one here.",
        identifier: "",
        interactive: false,
      };
    case "empty":
      return {
        state,
        placeholder: "No models offered",
        message: `${connectionLabel || "The connected provider"} is reachable but is not offering any model.`,
        identifier: "",
        interactive: false,
      };
    case "unchosen":
      return {
        state,
        placeholder: "Choose a model",
        message: `${input.models.length} model${input.models.length === 1 ? "" : "s"} available on ${connectionLabel || "this connection"}.`,
        identifier: "",
        interactive: true,
      };
    default: {
      const chosen = modelIdentity(selected);
      const name = modelDisplayName(selected);
      return {
        state,
        placeholder: "Choose a model",
        message: blamed
          ? `This chat uses ${name}, which could not run the last request.`
          : `This chat uses ${name}.`,
        identifier: chosen.renamed ? chosen.identifier : "",
        interactive: true,
      };
    }
  }
}

export function AiChatModelPicker({
  connection,
  models,
  modelFailures,
  onChoose,
  value,
}: {
  connection: AiChatConnection | null | undefined;
  models: string[];
  /** Models the appliance has already blamed for a failed request. */
  modelFailures: Record<string, string>;
  onChoose: (model: string) => void;
  value: string;
}) {
  /*
   * The model this picker actually points at. A remembered value the live list
   * no longer offers is reconciled to the first available model here — the same
   * one the dropdown falls back to showing — so the caption, the failure
   * mark and the dropdown all read one model rather than three.
   */
  const selected = effectiveModel(models, value);
  /*
   * One call, one state. The words under the control, the placeholder inside
   * it and whether it accepts input all come out of the same object, so the
   * picker cannot say "choose a model" while refusing to let one be chosen.
   */
  const picker = modelPickerPresentation({ connection, models, value }, selected in modelFailures);
  /*
   * The name is the answer; the path is the evidence. Both are in the hint so
   * they share one `aria-describedby` — a screen-reader user hears the model
   * it is using and then, quietly, the file it came from, rather than being
   * read a directory path as if it were a product name.
   */
  const hint = picker.identifier ? (
    <>
      {picker.message}{" "}
      <span className="ai-chat-model-picker__identifier">{picker.identifier}</span>
    </>
  ) : picker.message;
  return (
    <div className="ai-chat-model-picker" data-model-state={picker.state}>
      <Select
        disabled={!picker.interactive}
        disabledReason={picker.interactive ? undefined : picker.message}
        hint={picker.interactive ? hint : undefined}
        id="ai-chat-model"
        label="Model"
        onChange={(event) => onChoose(event.target.value)}
        value={selected}
      >
        {/*
          * The placeholder is a prompt, never a choice. It was selectable, so
          * on a machine with one local model an owner who opened the list to
          * *look* — the ordinary reason to open it — could mis-click the first
          * row and silently unset the only model the appliance has (#163). A
          * conversation dated four days earlier was already labelled "No
          * model", so the state was reachable and had been reached. Once a
          * model is chosen the prompt stays visible and inert; before one is
          * chosen it is still the honest empty value.
          */}
        <option disabled={Boolean(selected)} value="">{picker.placeholder}</option>
        {models.map((model) => (
          <option key={model} value={model}>
            {modelOptionLabel(model, model in modelFailures)}
          </option>
        ))}
      </Select>
    </div>
  );
}
