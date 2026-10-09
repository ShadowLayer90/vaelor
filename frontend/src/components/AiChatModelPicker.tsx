import {
  type KeyboardEvent as ReactKeyboardEvent,
  useCallback,
  useEffect,
  useId,
  useRef,
  useState,
} from "react";
import { ApiError } from "../lib/api";
import { modelDisplayName, modelIdentity } from "../lib/modelIdentity";
import { LOCAL_BY_MANAGED_PREFIX } from "../lib/gpuServingMode";
import type { AiChatClustering, AiChatConnection } from "./aiChatTypes";
import { Icon, ICON_SIZE } from "./Icon";
import { StatusPill } from "./StatusPill";
import { Button } from "./ui";

/**
 * Choosing what you want to talk to (the ChatModels board).
 *
 * A button in the chat's top row that opens a panel: a search box, the
 * backend's sentence when the connection cannot change, the models grouped by
 * where they run, and, for an administrator, Add a connection and Manage
 * connections.
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
  if (!(error instanceof ApiError)) return "";
  // The appliance names the model the request actually went to (ACC-095).
  // The picker's value is only a fallback for a failure that never reached
  // the appliance, such as the browser's own timeout.
  const sent = typeof error.details.model === "string" ? error.details.model : "";
  const blamed = sent || model;
  if (!blamed) return "";
  // `chat_model_busy` is Vaelor refusing the request itself (its own
  // concurrency bound); nothing reached the model, so the model is not blamed.
  if (error.code === "chat_model_busy") return "";
  return error.code.startsWith("chat_model_") || error.code === "request_timeout"
    ? blamed
    : "";
}

/**
 * The model a reopened chat puts in the picker (ACC-095).
 *
 * The model that last answered in it, when the connection still offers it;
 * otherwise the current choice, when that is offered; otherwise the first
 * model offered. A model the live list does not offer is never selected: the
 * picker could not show it, and in cluster mode vLLM answered it with a 404
 * that was then blamed on the model the picker did show.
 */
export function reopenedChatModel(answered: string, current: string, offered: string[]): string {
  if (!offered.length) return current;
  if (answered && offered.includes(answered)) return answered;
  return offered.includes(current) ? current : offered[0];
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
        // VD-200: this named an add form in Details that does not exist. The
        // form is in Apps and AI (`connectionFormHandoff`).
        message: "Install a local model, or connect an external provider for AI Chat under Apps and AI, then pick one here.",
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

/**
 * A panel that opens under a top-row button: closed by Escape, by a press
 * outside it, or by a choice. Escape is taken here, in the capture phase, so it
 * closes the panel and does not also leave focus view.
 */
export function useChatPopover() {
  const [open, setOpen] = useState(false);
  const rootRef = useRef<HTMLDivElement>(null);
  const buttonRef = useRef<HTMLButtonElement>(null);
  const close = useCallback((refocus = true) => {
    setOpen(false);
    if (refocus) buttonRef.current?.focus();
  }, []);
  useEffect(() => {
    if (!open) return;
    const pressOutside = (event: MouseEvent) => {
      if (!rootRef.current?.contains(event.target as Node)) setOpen(false);
    };
    const escape = (event: KeyboardEvent) => {
      if (event.key !== "Escape") return;
      event.stopPropagation();
      close();
    };
    document.addEventListener("mousedown", pressOutside);
    document.addEventListener("keydown", escape, true);
    return () => {
      document.removeEventListener("mousedown", pressOutside);
      document.removeEventListener("keydown", escape, true);
    };
  }, [close, open]);
  return { open, setOpen, close, rootRef, buttonRef };
}

/** Arrow keys, Home and End move between the enabled options of a list. */
export function moveOptionFocus(event: ReactKeyboardEvent<HTMLElement>) {
  const keys = ["ArrowDown", "ArrowUp", "Home", "End"];
  if (!keys.includes(event.key)) return;
  const options = [...event.currentTarget.querySelectorAll<HTMLButtonElement>('[role="option"]:not([aria-disabled="true"])')];
  if (!options.length) return;
  event.preventDefault();
  const at = options.indexOf(document.activeElement as HTMLButtonElement);
  const next = event.key === "Home" ? 0
    : event.key === "End" ? options.length - 1
      : event.key === "ArrowDown" ? Math.min(options.length - 1, at + 1)
        : Math.max(0, at - 1);
  options[next]?.focus();
}

/** The three groups the board draws, in its order. */
const GROUP_CLUSTER = "Cluster";
const GROUP_MACHINE = "This machine";
const GROUP_CONNECTIONS = "Your connections";

/**
 * Where a connection's models run, as the picker groups them. The cluster's
 * row is the one the mode file names (`cluster_credential_id`, VD-210), asked
 * of the backend rather than inferred from a refusal, since any other
 * connection may now be chosen while clustered. "This machine" is claimed
 * only on `vaelor-managed`, the one locality value that proves this
 * appliance serves the model itself.
 */
export function connectionGroup(connection: AiChatConnection, clusterId = ""): string {
  if (clusterId && connection.id === clusterId) return GROUP_CLUSTER;
  return connection.local_source === LOCAL_BY_MANAGED_PREFIX ? GROUP_MACHINE : GROUP_CONNECTIONS;
}

/**
 * The line under a connection's name: the service's own name (`provider_label`,
 * "Anthropic" rather than a wire id), and - for `remote`, the one locality known
 * to leave this machine - that prompts and files do (VD-206).
 */
export function connectionLine(connection: AiChatConnection): string {
  const name = connection.provider_label || connection.provider;
  return connection.local_source === "remote"
    ? `${name} · prompts and files leave this machine`
    : name;
}

type PickerRow = {
  key: string;
  /** The model id a model row chooses; absent on a connection row. */
  model?: string;
  name: string;
  label: string;
  line: string;
  selected: boolean;
  disabled: boolean;
  pill: { label: string; tone: "success" | "danger" | "neutral" } | null;
  choose: () => void;
};

export function AiChatModelPicker({
  administrator = false,
  busy = false,
  connection,
  connections = [],
  models,
  modelFailures,
  onActivateConnection,
  onChoose,
  onConnect,
  onManageConnections,
  clustering,
  value,
}: {
  /** Only an administrator can add or manage a connection (`POST /credentials`). */
  administrator?: boolean;
  /** A connection is being activated, so no other choice may start. */
  busy?: boolean;
  connection: AiChatConnection | null | undefined;
  /** Every connection AI Chat can be pointed at, the active one included. */
  connections?: AiChatConnection[];
  models: string[];
  /** Models the appliance has already blamed for a failed request. */
  modelFailures: Record<string, string>;
  onActivateConnection?: (connection: AiChatConnection) => void;
  onChoose: (model: string) => void;
  /** Open the add-a-connection form. */
  onConnect?: () => void;
  /** Open Settings, Connections. */
  onManageConnections?: () => void;
  /** VD-210: the cluster's row and the rows AI Chat cannot take, in the backend's words. */
  clustering?: AiChatClustering;
  value: string;
}) {
  const popover = useChatPopover();
  const [query, setQuery] = useState("");
  const searchRef = useRef<HTMLInputElement>(null);
  const ids = useId().replaceAll(":", "");
  /*
   * The model this picker actually points at. A remembered value the live list
   * no longer offers is reconciled to the first available model here, so the
   * button, the failure mark and the list all read one model rather than three.
   */
  const selected = effectiveModel(models, value);
  const picker = modelPickerPresentation({ connection, models, value }, selected in modelFailures);
  const buttonLabel = picker.state === "chosen" ? modelDisplayName(selected) : picker.placeholder;

  useEffect(() => {
    if (popover.open) searchRef.current?.focus();
    else setQuery("");
  }, [popover.open]);

  const groups = new Map<string, PickerRow[]>();
  const add = (group: string, row: PickerRow) => groups.set(group, [...(groups.get(group) ?? []), row]);
  if (connection) {
    const group = connectionGroup(connection, clustering?.clusterId);
    groups.set(group, []);
    for (const model of models) {
      const failed = model in modelFailures;
      add(group, {
        key: "model:" + model,
        model,
        name: modelDisplayName(model),
        label: modelOptionLabel(model, failed),
        line: `${connection.label} · ${connectionLine(connection)}`,
        selected: model === selected,
        disabled: false,
        pill: model === selected
          ? (failed ? { label: "Last request failed", tone: "danger" } : { label: "In use", tone: "success" })
          : null,
        choose: () => {
          onChoose(model);
          popover.close();
        },
      });
    }
  }
  for (const item of connections) {
    if (item.id === connection?.id) continue;
    // VD-210: only a row the backend refuses is held, and its reason is
    // its own line, beside it - every other connection can still be chosen.
    const held = clustering?.refusals[item.id] ?? "";
    add(connectionGroup(item, clustering?.clusterId), {
      key: "connection:" + item.id,
      name: item.label,
      label: item.label,
      line: held || connectionLine(item),
      selected: false,
      disabled: busy || Boolean(held) || !onActivateConnection,
      pill: { label: held ? "Paused for the cluster" : "Not active", tone: "neutral" },
      choose: () => {
        onActivateConnection?.(item);
        popover.close();
      },
    });
  }
  const needle = query.trim().toLowerCase();
  const shown = [...groups.entries()]
    .map(([group, rows]) => [group, rows.filter((row) => !needle || row.name.toLowerCase().includes(needle))] as const)
    .filter(([, rows]) => rows.length);

  return (
    <div className="ai-chat-popover-anchor ai-chat-model-picker" data-model-state={picker.state} ref={popover.rootRef}>
      <Button
        aria-controls={popover.open ? `ai-chat-model-panel-${ids}` : undefined}
        aria-expanded={popover.open}
        aria-haspopup="dialog"
        aria-label={`Model ${buttonLabel}`}
        className={popover.open ? "ai-chat-menu-button is-open" : "ai-chat-menu-button"}
        id="ai-chat-model"
        onClick={() => popover.setOpen((open) => !open)}
        ref={popover.buttonRef}
        type="button"
      >
        <span className="ai-chat-menu-button__key">Model</span>
        <span className="ai-chat-menu-button__value">{buttonLabel}</span>
        <Icon className="ai-chat-menu-button__chevron" name="chevron" size={ICON_SIZE.inline} />
      </Button>
      {popover.open && (
        <div
          aria-label="Choose a model"
          className="ai-chat-popover ai-chat-popover--models"
          id={`ai-chat-model-panel-${ids}`}
          role="dialog"
        >
          <label className="ai-chat-popover__search">
            <span className="sr-only">Search models</span>
            <input
              className="ai-chat-input"
              onChange={(event) => setQuery(event.target.value)}
              // ArrowDown from the search goes to the first choice it left (VD-200 assist review).
              onKeyDown={(event) => {
                if (event.key !== "ArrowDown") return;
                const first = event.currentTarget.closest(".ai-chat-popover")?.querySelector<HTMLElement>('[role="option"]:not([disabled])');
                if (first) { event.preventDefault(); first.focus(); }
              }}
              placeholder="Search models"
              ref={searchRef}
              type="search"
              value={query}
            />
          </label>
          <p className="ai-chat-popover__message" data-model-state={picker.state}>
            {picker.message}
            {picker.identifier && <span className="ai-chat-model-picker__identifier"> {picker.identifier}</span>}
          </p>
          {shown.map(([group, rows]) => {
            const headingId = `ai-chat-model-${ids}-${group.replaceAll(" ", "-")}`;
            return (
              <div className="ai-chat-popover__group" key={group}>
                <h3 className="ai-chat-popover__eyebrow" id={headingId}>{group}</h3>
                <div aria-labelledby={headingId} onKeyDown={moveOptionFocus} role="listbox">
                  {rows.map((row) => (
                    <Button
                      aria-label={row.label}
                      aria-selected={row.selected}
                      className={row.selected ? "ai-chat-option is-selected" : "ai-chat-option"}
                      data-model={row.model}
                      disabled={row.disabled}
                      key={row.key}
                      onClick={row.choose}
                      role="option"
                      type="button"
                      variant="quiet"
                    >
                      <span className="ai-chat-option__text">
                        <strong>{row.name}</strong>
                        <small>{row.line}</small>
                      </span>
                      {row.pill && <StatusPill label={row.pill.label} tone={row.pill.tone} />}
                    </Button>
                  ))}
                </div>
              </div>
            );
          })}
          {needle && !shown.length && <p className="ai-chat-popover__message">No model or connection matches that search.</p>}
          {administrator && (onConnect || onManageConnections) && (
            <div className="ai-chat-popover__footer">
              {onConnect && (
                <Button onClick={() => { popover.close(false); onConnect(); }} type="button" variant="primary">Add a connection</Button>
              )}
              {onManageConnections && (
                <Button className="ai-chat-ghost" onClick={() => { popover.close(false); onManageConnections(); }} type="button" variant="quiet">Manage connections</Button>
              )}
            </div>
          )}
        </div>
      )}
    </div>
  );
}
