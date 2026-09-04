import { useCallback, useEffect, useState } from "react";
import { apiRequest } from "../lib/api";
import { formatQuantity } from "../lib/format";
import type { Session } from "../types";
import { Icon } from "./Icon";
import { Button, Input, Notice, Select } from "./ui";

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
    id: "efficient" | "balanced" | "quality";
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
  recommendation: {
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
      id: "efficient" | "balanced" | "quality";
      name: string;
      description: string;
    }>;
  };
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
}

function formatCapacity(bytes: number) {
  return formatQuantity(bytes, "capacity");
}

const compatiblePresets = {
  lemonade: {
    name: "AMD Lemonade",
    placeholder: "http://192.168.0.50:13305/api/v1",
    help: "Lemonade normally uses port 13305 and the /api/v1 path.",
  },
  lmstudio: {
    name: "LM Studio",
    placeholder: "http://192.168.0.50:1234/v1",
    help: "Start the LM Studio server and enable network access if it runs on another computer.",
  },
  llamacpp: {
    name: "llama.cpp",
    placeholder: "http://192.168.0.50:8080/v1",
    help: "llama-server normally uses port 8080 and the /v1 path.",
  },
  custom: {
    name: "Other compatible server",
    placeholder: "http://192.168.0.50:8000/v1",
    help: "Enter the base URL that comes before /models and /chat/completions.",
  },
} as const;

export function CopilotSetup({
  onInstallNpuRelease,
  data,
  busy,
  session,
  onClose,
  onChooseLocal,
  onIntelligenceConnected,
}: {
  data: CopilotSetupData;
  busy: boolean;
  session: Session;
  onClose: () => void;
  onChooseLocal: (query: string, mode: "efficient" | "balanced" | "quality") => void;
  /** Install a release-sourced on-device NPU model by its flm tag (no HF flow). */
  onInstallNpuRelease?: (releaseTag: string) => void;
  onIntelligenceConnected?: () => void;
}) {
  const { hardware, recommendation } = data;
  const localModels = [recommendation.primary, ...recommendation.alternatives];
  const runtimeModes = recommendation.runtime_modes ?? [
    { id: "efficient" as const, name: "Memory saver", description: "Leaves the most RAM for apps." },
    { id: "balanced" as const, name: "Balanced", description: "Recommended for normal use." },
    { id: "quality" as const, name: "Long context", description: "Uses more RAM for longer prompts." },
  ];
  const [localModelId, setLocalModelId] = useState(recommendation.primary.id);
  const [runtimeMode, setRuntimeMode] = useState<"efficient" | "balanced" | "quality">("balanced");
  const localModel = localModels.find((item) => item.id === localModelId) ?? recommendation.primary;
  const [showProvider, setShowProvider] = useState(false);
  const [provider, setProvider] = useState("openai");
  const [compatiblePreset, setCompatiblePreset] = useState<keyof typeof compatiblePresets>("lmstudio");
  const [baseUrl, setBaseUrl] = useState("");
  const [model, setModel] = useState("");
  const [label, setLabel] = useState("");
  const [secret, setSecret] = useState("");
  const [credentials, setCredentials] = useState<Array<{
    id: string;
    provider: string;
    label: string;
    fingerprint: string;
    last_test_status?: string | null;
    active_for?: string[];
    selected_model?: string;
  }>>([]);
  const [credentialBusy, setCredentialBusy] = useState(false);
  const [credentialNotice, setCredentialNotice] = useState("");
  const [pendingDelete, setPendingDelete] = useState("");
  const [modelsByCredential, setModelsByCredential] = useState<Record<string, string[]>>({});
  const [selectedModels, setSelectedModels] = useState<Record<string, string>>({});
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
    void (async () => {
      try {
        const inventory = await apiRequest<{ models?: Array<{ file?: string; size_bytes?: number; in_use?: boolean }> }>("/managed");
        const byBasename: Record<string, { size: number; inUse: boolean }> = {};
        for (const item of inventory.models ?? []) {
          const basename = String(item.file ?? "").split("/").pop();
          if (basename) byBasename[basename] = { size: Number(item.size_bytes ?? 0), inUse: Boolean(item.in_use) };
        }
        setInstalledFiles(byBasename);
        setInstalledUnread(false);
      } catch {
        setInstalledFiles({});
        setInstalledUnread(true);
      }
    })();
  }, []);
  const installedEntry = installedFiles[String(localModel.file ?? "")];
  const localModelInstalled = installedEntry !== undefined
    && typeof localModel.download_bytes === "number"
    && installedEntry.size === localModel.download_bytes;
  const localModelServing = localModelInstalled && installedEntry.inUse;

  const loadCredentials = useCallback(async () => {
    if (!data.credential_storage_ready || session.user.role !== "administrator") return;
    try {
      const result = await apiRequest<{ credentials: typeof credentials }>("/credentials");
      setCredentials(result.credentials);
    } catch (error) {
      setCredentialNotice(error instanceof Error ? error.message : "Connections could not be loaded.");
    }
  }, [data.credential_storage_ready, session.user.role]);

  useEffect(() => {
    void loadCredentials();
  }, [loadCredentials]);

  const saveCredential = async () => {
    const compatible = provider === "openai-compatible";
    if ((!compatible && !secret.trim()) || (compatible && !baseUrl.trim())) return;
    setCredentialBusy(true);
    setCredentialNotice(compatible ? "Checking the models and chat connection…" : "");
    try {
      const result = await apiRequest<{
        id?: string;
        connection_test?: { message?: string };
        discovered_models?: string[];
        selection_required?: boolean;
      }>(
        "/credentials",
        {
          method: "POST",
          body: JSON.stringify({
            provider,
            label: label.trim() || (
              provider === "openai"
                ? "OpenAI account"
                : provider === "huggingface"
                  ? "Hugging Face account"
                  : compatiblePresets[compatiblePreset].name
            ),
            secret: compatible
              ? JSON.stringify({
                  base_url: baseUrl.trim(),
                  model: model.trim(),
                  api_key: secret.trim(),
                })
              : secret,
          }),
        },
        session.csrf_token,
      );
      setSecret("");
      setLabel("");
      if (compatible) {
        setBaseUrl("");
        setModel("");
      }
      if (
        compatible
        && result.id
        && result.selection_required
        && result.discovered_models?.length
      ) {
        setModelsByCredential((current) => ({
          ...current,
          [result.id!]: result.discovered_models!,
        }));
        setSelectedModels((current) => ({
          ...current,
          [result.id!]: result.discovered_models![0] ?? "",
        }));
      }
      setCredentialNotice(
        result.selection_required
          ? `${result.discovered_models?.length ?? 0} chat models found. Choose and test one before Vaelor uses this connection.`
          : result.connection_test?.message
          ? `${result.connection_test.message} Vaelor Assistant is now using it.`
          : "Provider credential encrypted and stored.",
      );
      if (provider === "openai" || provider === "openai-compatible") {
        onIntelligenceConnected?.();
      }
      await loadCredentials();
    } catch (error) {
      setCredentialNotice(error instanceof Error ? error.message : "Credential could not be stored.");
    } finally {
      setSecret("");
      setCredentialBusy(false);
    }
  };

  const testCredential = async (credentialId: string) => {
    setCredentialBusy(true);
    setCredentialNotice("Testing the provider connection…");
    try {
      const result = await apiRequest<{ ok: boolean; message: string }>(
        `/credentials/${credentialId}/test`,
        { method: "POST" },
        session.csrf_token,
      );
      setCredentialNotice(result.message);
      await loadCredentials();
    } catch (error) {
      setCredentialNotice(error instanceof Error ? error.message : "Connection test failed.");
    } finally {
      setCredentialBusy(false);
    }
  };

  const activateCredential = async (credentialId: string) => {
    setCredentialBusy(true);
    setCredentialNotice("Testing the connection before switching…");
    try {
      const result = await apiRequest<{ connection_test?: { message?: string } }>(
        `/credentials/${credentialId}/activate`,
        { method: "POST" },
        session.csrf_token,
      );
      setCredentialNotice(
        result.connection_test?.message
          ? `${result.connection_test.message} This is now the active Assistant.`
          : "This connection is now the active Assistant.",
      );
      onIntelligenceConnected?.();
      await loadCredentials();
    } catch (error) {
      setCredentialNotice(error instanceof Error ? error.message : "The Assistant connection could not be switched.");
    } finally {
      setCredentialBusy(false);
    }
  };

  const loadProviderModels = async (credentialId: string) => {
    setCredentialBusy(true);
    setCredentialNotice("Loading models available to this connection…");
    try {
      const result = await apiRequest<{ models: string[] }>(
        `/credentials/${credentialId}/models`,
      );
      setModelsByCredential((current) => ({ ...current, [credentialId]: result.models }));
      setSelectedModels((current) => ({
        ...current,
        [credentialId]: current[credentialId]
          || credentials.find((item) => item.id === credentialId)?.selected_model
          || result.models[0]
          || "",
      }));
      setCredentialNotice(`${result.models.length} available chat model${result.models.length === 1 ? "" : "s"} loaded.`);
    } catch (error) {
      setCredentialNotice(error instanceof Error ? error.message : "Available models could not be loaded.");
    } finally {
      setCredentialBusy(false);
    }
  };

  const selectProviderModel = async (credentialId: string) => {
    const selectedModel = selectedModels[credentialId];
    if (!selectedModel) return;
    setCredentialBusy(true);
    setCredentialNotice("Testing the selected model before switching…");
    try {
      await apiRequest(
        `/credentials/${credentialId}/model`,
        {
          method: "PATCH",
          body: JSON.stringify({ model: selectedModel }),
        },
        session.csrf_token,
      );
      setCredentialNotice(`${selectedModel} is now the active Vaelor Assistant model.`);
      onIntelligenceConnected?.();
      await loadCredentials();
    } catch (error) {
      setCredentialNotice(error instanceof Error ? error.message : "The selected model could not be activated.");
    } finally {
      setCredentialBusy(false);
    }
  };

  const deleteCredential = async (credentialId: string) => {
    if (pendingDelete !== credentialId) {
      setPendingDelete(credentialId);
      return;
    }
    setCredentialBusy(true);
    try {
      await apiRequest(
        `/credentials/${credentialId}`,
        { method: "DELETE" },
        session.csrf_token,
      );
      setPendingDelete("");
      setCredentialNotice("Provider disconnected from this Vaelor node.");
      await loadCredentials();
    } catch (error) {
      setCredentialNotice(error instanceof Error ? error.message : "Provider could not be disconnected.");
    } finally {
      setCredentialBusy(false);
    }
  };

  return (
    <section className="copilot-setup" aria-labelledby="copilot-setup-title">
      <div className="copilot-setup__header">
        <div>
          <span className="page-eyebrow">Assistant setup</span>
          <h2 id="copilot-setup-title">Choose how Vaelor Assistant thinks</h2>
          <p>We checked this device and narrowed the choices down for you.</p>
        </div>
        <Button variant="quiet" onClick={onClose}>Close</Button>
      </div>

      <div className="hardware-verdict">
        <span className="hardware-verdict__icon"><Icon name="cpu" size={28} /></span>
        <div>
          <small>Detected automatically</small>
          <strong>{formatCapacity(hardware.memory_total_bytes)} {hardware.device}</strong>
          <p>{hardware.cpu_cores} CPU cores · {hardware.architecture} · {formatCapacity(hardware.storage_free_bytes)} free</p>
        </div>
        <span className={`fit-badge ${recommendation.can_install ? "fit-badge--good" : "fit-badge--warning"}`}>
          {recommendation.can_install ? "Local AI fits" : "More space needed"}
        </span>
      </div>

      <div className="copilot-choice-grid">
        <article className="copilot-choice copilot-choice--recommended">
          <div className="copilot-choice__flag">Recommended for this hardware</div>
          <span className="copilot-choice__icon"><Icon name="shield" size={25} /></span>
          <small>LOCAL & PRIVATE</small>
          <h3>{localModel.name}</h3>
          <p>{localModel.experience}</p>
          {localModels.length > 1 && <div className="local-model-select"><Select id="copilot-local-model" label="Default local model" onChange={(event) => setLocalModelId(event.target.value)} value={localModelId}>{localModels.map((item) => <option key={item.id} value={item.id}>{item.name} - {item.size_note}</option>)}</Select></div>}
          <dl>
            {/* #145: this row said "Download about 2.4 GB" for a model that
                was already on disk and serving. What is on the appliance is
                stated as such; only a genuinely absent file is a download,
                and a failed inventory read is admitted rather than presented
                as an absence. */}
            <div><dt>Download</dt><dd>{localModelServing
              ? "Already installed and serving the Assistant"
              : localModelInstalled
                ? "Already on this appliance — no download needed"
                : installedUnread
                  ? `${localModel.size_note} — Vaelor could not check what is already installed`
                  : localModel.size_note}</dd></div>
            <div><dt>Model size</dt><dd>{localModel.parameter_size ?? recommendation.parameter_range}</dd></div>
            <div><dt>Privacy</dt><dd>Stays on device</dd></div>
          </dl>
          {localModel.memory_note && <p className="copilot-choice__fineprint">{localModel.memory_note}</p>}
          <fieldset className="runtime-mode-picker">
            <legend>RAM profile</legend>
            {runtimeModes.map((mode) => (
              <Button aria-pressed={runtimeMode === mode.id} key={mode.id} onClick={() => setRuntimeMode(mode.id)} type="button">
                <strong>{mode.name}</strong><span>{mode.description}</span>
              </Button>
            ))}
          </fieldset>
          <Button variant="primary"

            disabled={busy || !recommendation.can_install}
            onClick={() =>
              localModel.engine === "flm-npu" && localModel.release_tag && onInstallNpuRelease
                ? onInstallNpuRelease(localModel.release_tag)
                : onChooseLocal(localModel.search_query, runtimeMode)
            }
          >
            {busy
              ? "Checking model…"
              : localModelServing
                ? "Review runtime settings"
                : localModel.engine === "flm-npu"
                  ? "Set up the on-device Assistant"
                  : "Review local installation"}
          </Button>
          <p className="copilot-choice__fineprint">Vaelor verifies the file and reserves system RAM. If the selected profile will not fit safely, it automatically steps down before starting.</p>
        </article>

        <article className="copilot-choice">
          <span className="copilot-choice__icon"><Icon name="network" size={25} /></span>
          <small>HOSTED FRONTIER AI</small>
          <h3>Connect an AI provider</h3>
          <p>Use a more capable hosted model. Responses need internet access and may have usage charges.</p>
          <dl>
            <div><dt>Speed</dt><dd>Usually faster</dd></div>
            <div><dt>Intelligence</dt><dd>Frontier models</dd></div>
            <div><dt>Privacy</dt><dd>Provider terms apply</dd></div>
          </dl>
          <Button

            disabled={!data.credential_storage_ready || session.user.role !== "administrator"}
            onClick={() => setShowProvider((current) => !current)}
            title={!data.credential_storage_ready ? "Secure credential broker is not running" : undefined}
          >
            {showProvider ? "Hide provider setup" : "Connect provider"}
          </Button>
          <p className="copilot-choice__fineprint">OpenAI uses an API key; ChatGPT and API billing are separate. OAuth will appear only for providers that officially support it.</p>
        </article>
      </div>

      {showProvider && (
        <div className="provider-setup">
          <div className="provider-setup__heading">
            <div>
              <span className="page-eyebrow">Encrypted connection</span>
              <h3>Connect an AI service</h3>
              <p>Use OpenAI, Hugging Face, or a compatible model server already running on your network.</p>
            </div>
            <span className="vault-state"><Icon name="lock" size={16} /> Vault ready</span>
          </div>

          <div className={`provider-form ${provider === "openai-compatible" ? "provider-form--compatible" : ""}`}>
            <Select id="copilot-provider" label="Provider" onChange={(event) => setProvider(event.target.value)} value={provider}><option value="openai">OpenAI API</option><option value="huggingface">Hugging Face</option><option value="openai-compatible">OpenAI-compatible server</option></Select>
            {provider === "openai-compatible" && (
              <>
                <Select id="copilot-compatible-preset" label="Server app" onChange={(event) => setCompatiblePreset(event.target.value as keyof typeof compatiblePresets)} value={compatiblePreset}><option value="lemonade">AMD Lemonade</option><option value="lmstudio">LM Studio</option><option value="llamacpp">llama.cpp</option><option value="custom">Other compatible server</option></Select>
                <div className="provider-form__endpoint"><Input autoCapitalize="none" id="copilot-base-url" inputMode="url" label="Server base URL" maxLength={500} onChange={(event) => setBaseUrl(event.target.value)} placeholder={compatiblePresets[compatiblePreset].placeholder} spellCheck={false} type="url" value={baseUrl} /></div>
                <Input id="copilot-model-id" label={<>Model ID <span className="field-optional">optional</span></>} maxLength={200} onChange={(event) => setModel(event.target.value)} placeholder="Auto-detect first loaded model" spellCheck={false} value={model} />
              </>
            )}
            <Input id="copilot-connection-name" label="Connection name" maxLength={80} onChange={(event) => setLabel(event.target.value)} placeholder={provider === "openai-compatible" ? `Example: Office ${compatiblePresets[compatiblePreset].name}` : "Example: My OpenAI account"} value={label} />
            <div className="provider-form__secret"><Input autoComplete="new-password" id="copilot-secret" label={provider === "openai" ? "API key" : provider === "huggingface" ? "Access token" : <>API key <span className="field-optional">optional</span></>} maxLength={8192} onChange={(event) => setSecret(event.target.value)} placeholder={provider === "openai" ? "Paste your OpenAI API key" : provider === "huggingface" ? "Paste your Hugging Face token" : "Leave blank when authentication is disabled"} spellCheck={false} type="password" value={secret} /></div>
            <Button variant="primary"

              disabled={credentialBusy || (provider === "openai-compatible" ? !baseUrl.trim() : secret.trim().length < 8)}
              onClick={() => void saveCredential()}
            >
              {credentialBusy ? "Testing securely…" : provider === "openai-compatible" ? "Test and discover models" : "Encrypt and connect"}
            </Button>
          </div>
          {provider === "openai-compatible" && (
            <div className="compatible-guidance">
              <span><Icon name="network" size={17} /></span>
              <div>
                <strong>{compatiblePresets[compatiblePreset].help}</strong>
                <p>Use the computer’s LAN address—not localhost—unless the server runs on this Vaelor node. Only private network addresses are accepted.</p>
              </div>
            </div>
          )}

          {credentialNotice && <Notice severity="info"><Icon name="shield" />{credentialNotice}</Notice>}

          {credentials.length > 0 && (
            <div className="credential-list">
              <span className="control-label">Saved connections</span>
              {credentials.map((credential) => (
                <div className="credential-row" key={credential.id}>
                  <span className="credential-row__icon"><Icon name="lock" size={17} /></span>
                  <span>
                    <strong>{credential.label}</strong>
                    <small>
                      {credential.provider} · ••••{credential.fingerprint.slice(-4)}
                      {credential.active_for?.includes("deployment-agent") ? " · Assistant active" : ""}
                      {credential.selected_model ? ` · ${credential.selected_model}` : ""}
                    </small>
                  </span>
                  <span className={`credential-status credential-status--${credential.last_test_status ?? "untested"}`}>
                    {credential.last_test_status ?? "not tested"}
                  </span>
                  <Button disabled={credentialBusy} onClick={() => void testCredential(credential.id)}>Test</Button>
                  {credential.provider !== "huggingface" && (
                    <Button disabled={credentialBusy} onClick={() => void loadProviderModels(credential.id)}>Choose model</Button>
                  )}
                  {credential.provider !== "huggingface" && !credential.active_for?.includes("deployment-agent") && (
                    <Button variant="primary" disabled={credentialBusy} onClick={() => void activateCredential(credential.id)}>Use this</Button>
                  )}
                  <Button variant="danger" disabled={credentialBusy} onClick={() => void deleteCredential(credential.id)}>
                    {pendingDelete === credential.id ? "Confirm disconnect" : "Disconnect"}
                  </Button>
                  {modelsByCredential[credential.id]?.length ? (
                    <div className="credential-model-picker">
                      <Select id={`copilot-model-${credential.id}`} label="Available model" onChange={(event) => setSelectedModels((current) => ({ ...current, [credential.id]: event.target.value }))} value={selectedModels[credential.id] || ""}>{modelsByCredential[credential.id].map((availableModel) => <option key={availableModel} value={availableModel}>{availableModel}</option>)}</Select>
                      <Button variant="primary" disabled={credentialBusy || !selectedModels[credential.id]} onClick={() => void selectProviderModel(credential.id)}>
                        Use selected model
                      </Button>
                    </div>
                  ) : null}
                </div>
              ))}
            </div>
          )}
        </div>
      )}

      <details className="advanced-details copilot-setup__advanced">
        <summary>Why this model?</summary>
        <p>{recommendation.rationale} Vaelor uses {recommendation.quantization}; the selected RAM profile chooses a safe context size when the model starts.</p>
      </details>
    </section>
  );
}
