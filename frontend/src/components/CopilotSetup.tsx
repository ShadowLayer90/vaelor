import { useEffect, useState } from "react";
import "../styles/apps-setup.css";
import { apiRequest } from "../lib/api";
import { openAiChatConnectionForm } from "../lib/connectionFormHandoff";
import { formatQuantity } from "../lib/format";
import { RUNTIME_MODE_NAMES, type RuntimeMode } from "../lib/modelModes";
import type { Session } from "../types";
import { AppsFacts, AppsIconTile } from "./appsKit";
import { AppsBanner } from "./appsSetupParts";
import { StatusPill } from "./StatusPill";
import { Button, Select } from "./ui";

/**
 * One model this appliance can actually install.
 *
 * The shape is `vaelor/model_catalog.py`'s, and the fields that matter are
 * `repo` and `file`: an entry earns its place there by having a repository and
 * a file a download can be started from, which is what stops a name being
 * recommended that nothing downstream could fetch. **No model name exists
 * outside that module**, so nothing here invents one.
 */
export interface LocalModelChoice {
  id: string;
  name: string;
  parameter_size?: string;
  quantization?: string;
  repo?: string;
  file?: string;
  download_bytes?: number;
  /** `measured` or `upper-bound`; the size sentence already carries which. */
  size_source?: string;
  size_note: string;
  /** The running memory cost, stated where the choice is made (#146). */
  memory_note?: string;
  search_query: string;
  experience: string;
  /** Set on the pick served on the GPU (the ROCmFP4 27B). The card badges it. */
  served_on_gpu?: boolean;
  /** The inference engine behind the pick, e.g. `rocmfpx` for the GPU model. */
  backend?: string;
  /**
   * The engine the entry declares. `flm-npu` marks a fine-tuned on-device model
   * delivered as a release: it installs through `/copilot/install-npu-model`
   * (the bridge unpacks model + runtime), not the Hugging Face download flow.
   */
  engine?: string;
  /** The flm tag a release-sourced NPU model serves as, e.g. `qwen3.5:4b`. */
  release_tag?: string;
}

/**
 * A model recommendation, shared shape for the two roles this appliance serves.
 *
 * The backend returns two of these because the appliance runs two local models:
 * the Assistant on the neural processor, and AI Chat on the GPU. `recommendation`
 * is the Assistant's pick (the NPU model, or the base GGUF on a Pi);
 * `catalog_recommendation` is the installable-catalog pick (the GPU 27B on a
 * gfx1151 box, or the base GGUF elsewhere). Merging the two into one field was
 * the bug that recommended the GPU 27B as the Assistant's brain.
 */
export interface ModelRecommendation {
  tier: string;
  /** The recommended model's **own** size, not a parameter class. */
  parameter_range: string;
  /** What the hardware could hold, kept apart from what is offered. */
  hardware_tier?: string;
  /** True when this machine could run more than the catalog stocks. */
  exceeds_catalog?: boolean;
  /** Says that about the *hardware*, and names no model. Empty otherwise. */
  catalog_note?: string;
  /** True when the recommended pick is served on this machine's GPU. */
  served_on_gpu?: boolean;
  /** True when the Assistant is served on this machine's neural processor. */
  served_on_npu?: boolean;
  /** Every installable model, served rather than written in the client. */
  catalog?: LocalModelChoice[];
  quantization: string;
  context_tokens: number;
  can_install: boolean;
  storage_ok: boolean;
  rationale: string;
  primary: LocalModelChoice;
  alternatives: LocalModelChoice[];
  runtime_modes?: Array<{
    id: RuntimeMode;
    name: string;
    description: string;
  }>;
}

export interface CopilotSetupData {
  hardware: {
    device: string;
    architecture: string;
    cpu_cores: number;
    memory_total_bytes: number;
    memory_available_bytes: number;
    storage_free_bytes: number;
  };
  /**
   * The **Assistant's** model: the NPU model on a Z2, or the base GGUF on a Pi.
   * The GPU 27B is never merged here — that was the bug. Read by CopilotSetup
   * and AgentAssistantPanel.
   */
  recommendation: ModelRecommendation;
  /**
   * The installable-**catalog** recommendation: the GPU 27B on a gfx1151 box,
   * or the base GGUF elsewhere. Read by ModelCatalog so the catalog can
   * recommend the 27B independently of the Assistant. Same shape as
   * `recommendation`.
   */
  catalog_recommendation: ModelRecommendation;
  providers: Array<{
    id: string;
    name: string;
    auth: string;
    available: boolean;
    description: string;
    note?: string;
  }>;
  credential_storage_ready: boolean;
  /**
   * Whose model the Assistant's lease names (`credential_listing.assistant_connection`).
   * `added` is a box set up before VD-201 item 3 whose Assistant still uses a
   * connection the owner added; `other` a model Vaelor runs for AI Chat or the
   * cluster (VD-202); `unknown` a listing that could not be read. `message` is
   * the server's sentence for each. Operators and administrators get `label`
   * and the sentence naming it; a viewer gets "" and a sentence without it
   * (VD-204, `credential_listing.sees_connection_names`).
   */
  assistant_connection?: AssistantConnection;
  /**
   * Whether AI Chat's lease names a model (`credential_listing.ai_chat_connection`).
   * `assistant-model` is AI Chat's lease left on the Assistant's on-device
   * model, which AI Chat refuses to send to (VD-210) and the next on-device
   * install removes; `switching` is cluster mode being
   * entered; `cluster` is cluster mode with nothing assigned; `none` is AI Chat
   * with no model. The sentence ends with a step the reader's role can take.
   * Reported only: nothing is moved for the owner.
   */
  ai_chat_connection?: AiChatConnection;
}

export interface AiChatConnection {
  state: "connected" | "assistant-model" | "switching" | "cluster" | "none" | "unknown";
  message: string;
}

export interface AssistantConnection {
  state: "vaelor" | "added" | "other" | "none" | "unknown";
  label: string;
  message: string;
}

/** Where saved connections are listed, tested and removed (an administrator's page). */
export const SAVED_CONNECTIONS_HASH = "#/admin/connections";

function formatCapacity(bytes: number) {
  return formatQuantity(bytes, "capacity");
}

export type CopilotSetupProps = {
  data: CopilotSetupData;
  busy: boolean;
  session: Session;
  onClose: () => void;
  onChooseLocal: (query: string, mode: RuntimeMode) => void;
  /** Install a release-sourced on-device NPU model by its flm tag (no HF flow). */
  onInstallNpuRelease?: (releaseTag: string) => void;
  /** Built-in basic mode: one of the two ways back for an Assistant on an added connection. */
  onChooseBasic?: () => void;
  /**
   * "Connect a model for AI Chat". Apps and AI opens its Add a connection
   * dialog; without it (the Assistant page) the handoff lands there instead.
   */
  onConnectModel?: () => void;
  /**
   * The eyebrow, title and Close drawn by this component. The dialog
   * (CopilotSetupDialog) and a drawer draw their own and pass false.
   */
  showHeader?: boolean;
  /**
   * Retired with the inline connection form (VD-200): the form is the Add a
   * connection dialog now, which Apps and AI opens from the handoff. Accepted
   * and ignored so a caller that still passes it keeps compiling.
   */
  openConnectionForm?: boolean;
};

/**
 * The Assistant setup's content (the AppsAssistantSetup board): the hardware
 * check, the recommended local model with its RAM profile, AI Chat's card and
 * the standing notices the server reports. Chrome-less by choice: the dialog,
 * the Assistant page and the Change-intelligence drawer each frame it.
 */
export function CopilotSetup({
  onInstallNpuRelease,
  data,
  busy,
  session,
  onClose,
  onChooseLocal,
  onChooseBasic,
  onConnectModel,
  showHeader = true,
}: CopilotSetupProps) {
  const { hardware, recommendation } = data;
  const localModels = [recommendation.primary, ...recommendation.alternatives];
  const runtimeModes = recommendation.runtime_modes ?? [
    { id: "efficient" as const, name: RUNTIME_MODE_NAMES.efficient, description: "Leaves the most RAM for apps." },
    { id: "balanced" as const, name: RUNTIME_MODE_NAMES.balanced, description: "Recommended for normal use." },
    { id: "quality" as const, name: RUNTIME_MODE_NAMES.quality, description: "Uses more RAM for longer prompts." },
  ];
  const [localModelId, setLocalModelId] = useState(recommendation.primary.id);
  const [runtimeMode, setRuntimeMode] = useState<RuntimeMode>("balanced");
  const localModel = localModels.find((item) => item.id === localModelId) ?? recommendation.primary;
  /*
   * #145: this card presented the model that was already installed and
   * serving as "Download about 2.4 GB / Review local installation", with no
   * sign it was on disk. The installed inventory says which files exist,
   * how large they are, and which one serves. Two honesty rules on top:
   *
   * - A name match alone is not an identity. VD-065 records two publishers
   *   shipping this exact filename 1,056 bytes and one digest apart, so
   *   "already installed" additionally requires the on-disk byte length to
   *   equal the catalog entry's recorded download size. A same-named file of
   *   a different length is a different artifact, and the catalog's verified
   *   one genuinely still needs downloading.
   * - A failed read is said, not papered over: the Download row keeps the
   *   size but adds that Vaelor could not check what is already installed,
   *   instead of silently reverting to the pre-#145 download framing.
   */
  const [installedFiles, setInstalledFiles] = useState<Record<string, { size: number; inUse: boolean }>>({});
  const [installedUnread, setInstalledUnread] = useState(false);
  useEffect(() => {
    let current = true;
    void (async () => {
      try {
        const inventory = await apiRequest<{ models?: Array<{ file?: string; size_bytes?: number; in_use?: boolean }> }>("/managed");
        const byBasename: Record<string, { size: number; inUse: boolean }> = {};
        for (const item of inventory.models ?? []) {
          const basename = String(item.file ?? "").split("/").pop();
          if (basename) byBasename[basename] = { size: Number(item.size_bytes ?? 0), inUse: Boolean(item.in_use) };
        }
        if (!current) return;
        setInstalledFiles(byBasename);
        setInstalledUnread(false);
      } catch {
        if (!current) return;
        setInstalledFiles({});
        setInstalledUnread(true);
      }
    })();
    return () => { current = false; };
  }, []);
  const installedEntry = installedFiles[String(localModel.file ?? "")];
  const localModelInstalled = installedEntry !== undefined
    && typeof localModel.download_bytes === "number"
    && installedEntry.size === localModel.download_bytes;
  const localModelServing = localModelInstalled && installedEntry.inUse;
  /*
   * VD-202 (LESSONS 6): the container being in use is not the Assistant using
   * it. When the server says the Assistant's lease names something else, the
   * card says the model runs and the Assistant is not on it, rather than a
   * second, contradicting answer under the warning.
   */
  const assistantElsewhere = data.assistant_connection?.state === "added" || data.assistant_connection?.state === "other";
  const isAdministrator = session.user.role === "administrator";
  // PATCH /assistant/preferences is operator-level; a viewer is told who can
  // choose basic mode by the server's sentence instead of a button that 403s.
  const canChooseBasic = Boolean(onChooseBasic) && session.user.role !== "viewer";
  const aiChat = data.ai_chat_connection;
  const connectReason = !isAdministrator
    ? "An administrator adds connections."
    : !data.credential_storage_ready
      ? "The secure credential store is not running."
      : undefined;
  const npuRelease = localModel.engine === "flm-npu" && Boolean(localModel.release_tag) && Boolean(onInstallNpuRelease);

  // #145: what is on the appliance is stated as such; only a genuinely absent
  // file is a download, and a failed inventory read is admitted rather than
  // presented as an absence.
  const downloadRow = localModelServing
    ? assistantElsewhere
      ? "Already installed and running; the Assistant is not using it"
      : "Already installed and serving the Assistant"
    : localModelInstalled
      ? "Already on this appliance - no download needed"
      : installedUnread
        ? `${localModel.size_note} - Vaelor could not check what is already installed`
        : localModel.size_note;

  return (
    <section aria-labelledby={showHeader ? "copilot-setup-title" : undefined} className="apps-assistant-setup">
      {showHeader && (
        <div className="apps-assistant-setup__header">
          <div>
            <span className="apps-eyebrow">Assistant setup</span>
            <h2 id="copilot-setup-title">Choose how Vaelor Assistant thinks</h2>
          </div>
          <Button onClick={onClose} variant="quiet">Close</Button>
        </div>
      )}
      <p className="apps-assistant-setup__intro">We checked this device and narrowed the choices down for you.</p>

      <div className="apps-hardware-strip">
        <AppsIconTile name="cpu" />
        <div className="apps-hardware-strip__text">
          <small>Detected automatically</small>
          <strong>{formatCapacity(hardware.memory_total_bytes)} {hardware.device}</strong>
          <span>{hardware.cpu_cores} CPU cores · {hardware.architecture} · {formatCapacity(hardware.storage_free_bytes)} free</span>
        </div>
        <StatusPill
          label={recommendation.can_install ? "Local AI fits" : "More space needed"}
          tone={recommendation.can_install ? "success" : "warning"}
        />
      </div>

      {/* VD-049 / VD-201 / VD-202: whose model the Assistant's lease names,
          as the server reports it. A box set up before the fix may still run
          the Assistant on an added connection or on a model Vaelor runs for
          something else; the server's sentence is shown as it is, with the
          ways back, and nothing is moved on its own. A standing state, so a
          status, never an alert. "unknown" is said, never left blank. */}
      {assistantElsewhere && (
        <AppsBanner
          action={(canChooseBasic || (data.assistant_connection?.state === "added" && isAdministrator)) && (
            <>
              {canChooseBasic && <Button disabled={busy} onClick={onChooseBasic}>Use built-in basic mode</Button>}
              {/* Saved connections are listed, tested and removed in
                  Settings > Connections, an administrator's page. */}
              {data.assistant_connection?.state === "added" && isAdministrator && (
                <Button onClick={() => { window.location.hash = SAVED_CONNECTIONS_HASH; }}>Show saved connections</Button>
              )}
            </>
          )}
          className="assistant-added-connection"
          tone="warning"
        >
          {data.assistant_connection?.message}
        </AppsBanner>
      )}
      {data.assistant_connection?.state === "unknown" && data.assistant_connection.message && (
        <AppsBanner tone="info">{data.assistant_connection.message}</AppsBanner>
      )}
      {/* AI Chat's model, as the server reports it: left on the Assistant's
          on-device model (lost on that model's next install), switching to the
          cluster model, or none at all, each with the step this role can take.
          Said, never fixed on the owner's behalf. A connected AI Chat comes
          with no sentence, so nothing is drawn for it. */}
      {aiChat?.message && (
        <AppsBanner className="ai-chat-model-report" tone={aiChat.state === "assistant-model" ? "warning" : "info"}>
          {aiChat.message}
        </AppsBanner>
      )}

      <div className="apps-assistant-grid">
        <article className="apps-choice apps-choice--accent">
          <div className="apps-choice__top">
            <span className="apps-flag apps-flag--accent">Recommended for this hardware</span>
            <span className="apps-label">Local and private</span>
          </div>
          <h3>{localModel.name}</h3>
          <p>{localModel.experience}</p>
          {localModels.length > 1 && (
            <Select id="copilot-local-model" label="Default local model" onChange={(event) => setLocalModelId(event.target.value)} value={localModelId}>
              {localModels.map((item) => <option key={item.id} value={item.id}>{item.name} - {item.size_note}</option>)}
            </Select>
          )}
          <AppsFacts
            className="apps-assistant-facts"
            rows={[
              { label: "Download", value: downloadRow },
              { label: "Model size", value: localModel.parameter_size ?? recommendation.parameter_range },
              { label: "Privacy", value: "Stays on device" },
            ]}
          />
          {localModel.memory_note && <p className="apps-fineprint">{localModel.memory_note}</p>}
          <fieldset className="apps-ram-profile">
            <legend>RAM profile</legend>
            <div className="apps-ram-profile__options">
              {runtimeModes.map((mode) => (
                <Button
                  aria-pressed={runtimeMode === mode.id}
                  className="apps-ram-profile__option"
                  key={mode.id}
                  onClick={() => setRuntimeMode(mode.id)}
                >
                  <strong>{mode.name}</strong><span>{mode.description}</span>
                </Button>
              ))}
            </div>
          </fieldset>
          <div className="apps-choice__action">
            <Button
              disabled={busy}
              disabledReason={recommendation.can_install ? undefined : "This machine needs more free storage or memory for the recommended model."}
              onClick={() =>
                npuRelease
                  ? onInstallNpuRelease?.(localModel.release_tag as string)
                  : onChooseLocal(localModel.search_query, runtimeMode)
              }
              variant="primary"
            >
              {busy
                ? "Checking model…"
                : localModelServing
                  ? "Review runtime settings"
                  : localModel.engine === "flm-npu"
                    ? "Set up the on-device Assistant"
                    : "Review local installation"}
            </Button>
          </div>
          <p className="apps-fineprint">Vaelor verifies the file and reserves system RAM. If the selected profile will not fit safely, it automatically steps down before starting.</p>
        </article>

        <article className="apps-choice">
          <div className="apps-choice__top">
            <AppsIconTile name="chat" />
            <span className="apps-label">For AI Chat</span>
          </div>
          <h3>Another model for AI Chat</h3>
          <p>Add a hosted provider or a model server on your network for AI Chat conversations. The Assistant keeps Vaelor's own model.</p>
          <AppsFacts
            className="apps-assistant-facts"
            rows={[
              { label: "Used by", value: "AI Chat only" },
              { label: "Assistant", value: "Unchanged" },
              { label: "Privacy", value: "Provider terms apply" },
            ]}
          />
          <div className="apps-choice__action">
            <Button disabledReason={connectReason} onClick={() => (onConnectModel ?? openAiChatConnectionForm)()}>
              Connect a model for AI Chat
            </Button>
          </div>
          <p className="apps-fineprint">Opens Add a connection. Saved connections are listed, tested and removed in Settings › Connections.</p>
          <p className="apps-fineprint">OpenAI uses an API key; ChatGPT and API billing are separate. OAuth will appear only for providers that officially support it.</p>
        </article>
      </div>

      <details className="apps-disclosure">
        <summary>Why this model?</summary>
        <p>{recommendation.rationale} Vaelor uses {recommendation.quantization}; the selected RAM profile chooses a safe context size when the model starts.</p>
      </details>
    </section>
  );
}
