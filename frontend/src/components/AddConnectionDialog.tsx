import { useEffect, useId, useRef, useState } from "react";
import "../styles/apps-setup.css";
import { apiRequest } from "../lib/api";
import type { Session } from "../types";
import { AppsDialog } from "./appsKit";
import { AppsBanner } from "./appsSetupParts";
import { StatusPill } from "./StatusPill";
import { Button, Input, Select } from "./ui";

/*
 * Add a connection (the approved AddConnection board): the one form that adds
 * a model for AI Chat. VD-049 / VD-201 / VD-206: a connection is AI Chat's,
 * never the Assistant's, and the words say so.
 *
 * Six choices (VD-206): OpenAI, Anthropic, Google Gemini and OpenRouter by
 * name, a server on the owner's network, and any other OpenAI-compatible
 * service at an HTTPS address. Each shows only what it needs. Six is past what
 * a segmented control holds, so they are a radio group.
 *
 * The server saves and tests in one step (POST /credentials): a connection
 * that fails its test is not kept. So "Test connection" saves it; the tested
 * box then offers the model to use and "Use in AI Chat", which records that
 * model (PATCH .../model) or switches AI Chat to the connection (.../activate).
 * A hosted service is not given to AI Chat until a model is chosen: its model
 * list is a catalogue, and its first entry is not the owner's pick.
 */

type Kind = "openai" | "anthropic" | "gemini" | "openrouter" | "network" | "custom";

interface Choice {
  kind: Kind;
  label: string;
  /** The broker's provider kind this choice saves. */
  provider: string;
  /** Who receives prompts; "" for a server on the owner's own network. */
  company: string;
  /** What the choice is, in one line under its name. */
  line: string;
  defaultName: string;
}

/*
 * A literal array, drawn with one `.map`, so the UI inventory reads one radio
 * per choice by its label (tools/ui_inventory_expand.py; VD-132 sweep).
 */
const CHOICE_LIST: readonly Choice[] = [
  { kind: "openai", label: "OpenAI", provider: "openai", company: "OpenAI", line: "GPT models with an OpenAI API key", defaultName: "OpenAI account" },
  { kind: "anthropic", label: "Anthropic", provider: "anthropic", company: "Anthropic", line: "Claude models through Anthropic's own API", defaultName: "Anthropic" },
  { kind: "gemini", label: "Google Gemini", provider: "gemini", company: "Google", line: "Gemini models with a Google AI Studio key", defaultName: "Google Gemini" },
  { kind: "openrouter", label: "OpenRouter", provider: "openrouter", company: "OpenRouter", line: "Many providers' models with one key", defaultName: "OpenRouter" },
  { kind: "network", label: "A server on your network", provider: "openai-compatible", company: "", line: "LM Studio, Lemonade, llama.cpp or similar", defaultName: "" },
  { kind: "custom", label: "A custom hosted service (HTTPS)", provider: "hosted-compatible", company: "", line: "Any OpenAI-compatible service on the internet", defaultName: "Hosted service" },
];

const CHOICES = Object.fromEntries(CHOICE_LIST.map((item) => [item.kind, item])) as Record<Kind, Choice>;

const NETWORK_PRESETS = {
  lmstudio: {
    name: "LM Studio",
    placeholder: "http://192.0.2.50:1234/v1",
    help: "Start the LM Studio server and enable network access if it runs on another computer.",
  },
  lemonade: {
    name: "AMD Lemonade",
    placeholder: "http://192.0.2.50:13305/api/v1",
    help: "Lemonade normally uses port 13305 and the /api/v1 path.",
  },
  llamacpp: {
    name: "llama.cpp",
    placeholder: "http://192.0.2.50:8080/v1",
    help: "llama-server normally uses port 8080 and the /v1 path.",
  },
  custom: {
    name: "Other compatible server",
    placeholder: "http://192.0.2.50:8000/v1",
    help: "Enter the base URL that comes before /models and /chat/completions.",
  },
} as const;

type Preset = keyof typeof NETWORK_PRESETS;

/** OpenRouter routes each request on to another company (VD-206). */
const OPENROUTER_FORWARDS = "OpenRouter forwards each prompt and file to the provider of the model you choose, under that provider's terms.";

interface CreatedCredential {
  id?: string;
  label?: string;
  connection_test?: { ok?: boolean; message?: string };
  /** null: the server could not read the list; [] read and empty. */
  discovered_models?: string[] | null;
  selection_required?: boolean;
  selected_model?: string;
  active_for?: string[];
  /** Why AI Chat could not take it (the cluster holds AI Chat), or "". */
  ai_chat_refusal?: string;
}

interface Saved {
  id: string;
  label: string;
  /** null: the model list was not read; [] read and empty. */
  models: string[] | null;
  testOk: boolean;
  testedAt: number;
  /** A hosted catalogue: the owner picks the model; nothing is preselected. */
  chooseModel: boolean;
  usedByAiChat: boolean;
  aiChatRefusal: string;
}

function refusal(error: unknown, fallback: string) {
  return error instanceof Error && error.message ? error.message : fallback;
}

/** "Tested just now", then minutes: a test result ages in words (LESSONS 1). */
function testedAge(testedAt: number, now: number) {
  const minutes = Math.floor((now - testedAt) / 60000);
  return minutes < 1 ? "Tested just now" : `Tested ${minutes} min ago`;
}

/**
 * The amber notice: where prompts and files go, for every choice (VD-206).
 * "The Assistant never uses it" is true while VD-207's approval is unbuilt:
 * research escalation to an off-machine model fails closed
 * (`model_connection.resolve_model_connection`). Revisit when it is built.
 */
function destinationNotice(kind: Kind) {
  const company = CHOICES[kind].company;
  if (kind === "network") return "Prompts and files you send in AI Chat go to this server. The Assistant never uses it.";
  return company
    ? `Prompts and files you send in AI Chat leave this machine for ${company}. The Assistant never uses it.`
    : "Prompts and files you send in AI Chat leave this machine for this service. The Assistant never uses it.";
}

export function AddConnectionDialog({
  onClose,
  onNotice,
  session,
}: {
  session: Session;
  onClose: () => void;
  onNotice?: (message: string, refused?: boolean) => void;
}) {
  const [kind, setKind] = useState<Kind>("openai");
  const [preset, setPreset] = useState<Preset>("lmstudio");
  const [name, setName] = useState("");
  const [address, setAddress] = useState("");
  const [apiKey, setApiKey] = useState("");
  const [busy, setBusy] = useState<"" | "test" | "use" | "token">("");
  const [error, setError] = useState("");
  const [saved, setSaved] = useState<Saved | null>(null);
  const [model, setModel] = useState("");
  /** The model to ask for; blank lets the test list them (or a network server pick its loaded one). */
  const [modelId, setModelId] = useState("");
  const [now, setNow] = useState(() => Date.now());
  const [token, setToken] = useState("");
  const [tokenNote, setTokenNote] = useState<{ text: string; refused: boolean } | null>(null);
  const mounted = useRef(true);
  const groupId = "add-connection-kind-" + useId().replaceAll(":", "");
  useEffect(() => () => { mounted.current = false; }, []);
  useEffect(() => {
    if (!saved) return;
    const timer = window.setInterval(() => setNow(Date.now()), 30000);
    return () => window.clearInterval(timer);
  }, [saved]);

  const isAdministrator = session.user.role === "administrator";
  const choice = CHOICES[kind];
  const network = kind === "network";
  const custom = kind === "custom";
  const needsAddress = network || custom;
  const defaultLabel = network ? NETWORK_PRESETS[preset].name : choice.defaultName;
  const label = name.trim() || defaultLabel;
  const testReason = !isAdministrator
    ? "An administrator adds connections."
    : needsAddress && !address.trim()
      ? "Enter the server address first."
      : custom && !address.trim().toLowerCase().startsWith("https://")
        ? "A hosted service address starts with https://."
        : !network && apiKey.trim().length < 8
          ? "Paste the API key first."
          : undefined;

  const finish = (message: string, refused = false) => {
    onNotice?.(message, refused);
    onClose();
  };

  const cancel = () => {
    if (busy) return;
    if (!saved) { onClose(); return; }
    // The connection is kept: the server saved it when it passed its test.
    finish(saved.usedByAiChat
      ? `${saved.label} is saved and AI Chat uses it.`
      : saved.aiChatRefusal
        ? `${saved.label} is saved. ${saved.aiChatRefusal}`
        : `${saved.label} is saved. AI Chat keeps its current model until you switch to it in AI Chat.`);
  };

  const requestBody = () => {
    if (needsAddress) {
      return {
        provider: choice.provider,
        label,
        secret: JSON.stringify({ base_url: address.trim(), model: modelId.trim(), api_key: apiKey.trim() }),
      };
    }
    return { provider: choice.provider, label, secret: apiKey.trim(), ...(modelId.trim() ? { model: modelId.trim() } : {}) };
  };

  const testConnection = async () => {
    if (testReason || busy) return;
    setBusy("test");
    setError("");
    try {
      const created = await apiRequest<CreatedCredential>(
        "/credentials",
        { method: "POST", body: JSON.stringify(requestBody()) },
        session.csrf_token,
      );
      if (!created.id) throw new Error("The connection was not saved.");
      let models: string[] | null = created.discovered_models === undefined ? null : created.discovered_models;
      if (created.discovered_models === undefined) {
        try {
          models = (await apiRequest<{ models: string[] }>(`/credentials/${created.id}/models`)).models;
        } catch {
          models = null;
        }
      }
      if (!mounted.current) return;
      setSaved({
        id: created.id,
        label: created.label || label,
        models,
        testOk: true,
        testedAt: Date.now(),
        chooseModel: Boolean(created.selection_required) && !created.selected_model,
        usedByAiChat: Boolean(created.active_for?.includes("ai-chat")),
        aiChatRefusal: created.ai_chat_refusal ?? "",
      });
      setNow(Date.now());
      // A hosted service's list is a catalogue: its first entry is not the
      // owner's choice, so nothing is preselected there (VD-206).
      setModel(created.selected_model || (created.selection_required ? "" : models?.[0] || ""));
    } catch (caught) {
      if (mounted.current) setError(refusal(caught, "The connection could not be tested."));
    } finally {
      // The key is held by the appliance now, or was refused; it is not kept here.
      if (mounted.current) { setApiKey(""); setBusy(""); }
    }
  };

  const testAgain = async () => {
    if (!saved || busy) return;
    setBusy("test");
    setError("");
    try {
      const result = await apiRequest<{ ok: boolean; message: string }>(
        `/credentials/${saved.id}/test`, { method: "POST" }, session.csrf_token,
      );
      if (!mounted.current) return;
      setSaved({ ...saved, testOk: result.ok, testedAt: Date.now() });
      setNow(Date.now());
      if (!result.ok) setError(result.message || "The connection test failed.");
    } catch (caught) {
      if (mounted.current) setError(refusal(caught, "The connection test failed."));
    } finally {
      if (mounted.current) setBusy("");
    }
  };

  const useInAiChat = async () => {
    if (!saved || busy) return;
    setBusy("use");
    setError("");
    try {
      if (model && saved.models?.length) {
        await apiRequest(`/credentials/${saved.id}/model`, { method: "PATCH", body: JSON.stringify({ model }) }, session.csrf_token);
        if (mounted.current) finish(`AI Chat now uses ${model} on ${saved.label}.`);
      } else {
        await apiRequest(`/credentials/${saved.id}/activate`, { method: "POST" }, session.csrf_token);
        if (mounted.current) finish(`AI Chat now uses ${saved.label}.`);
      }
    } catch (caught) {
      if (mounted.current) setError(refusal(caught, "AI Chat could not be switched to this connection."));
    } finally {
      if (mounted.current) setBusy("");
    }
  };

  const storeToken = async () => {
    if (busy || token.trim().length < 8) return;
    setBusy("token");
    setTokenNote(null);
    try {
      await apiRequest(
        "/credentials",
        { method: "POST", body: JSON.stringify({ provider: "huggingface", label: "Hugging Face account", secret: token.trim() }) },
        session.csrf_token,
      );
      if (mounted.current) setTokenNote({ text: "The Hugging Face token is stored encrypted and used for model downloads.", refused: false });
    } catch (caught) {
      if (mounted.current) setTokenNote({ text: refusal(caught, "The token could not be stored."), refused: true });
    } finally {
      if (mounted.current) { setToken(""); setBusy(""); }
    }
  };

  const useReason = saved?.aiChatRefusal
    ? "Saved and listed; AI Chat can use it once the cluster is removed."
    : saved && !saved.testOk
      ? "Test the connection again first."
      : saved?.chooseModel && saved.models?.length && !model
        ? "Choose a model first."
        : undefined;
  const modelsRead = saved?.models ?? null;

  return (
    <AppsDialog
      busy={Boolean(busy)}
      error={error}
      eyebrow="For AI Chat"
      footer={saved ? (
        <>
          <Button disabled={Boolean(busy)} onClick={() => void testAgain()}>{busy === "test" ? "Testing…" : "Test again"}</Button>
          <Button disabled={Boolean(busy)} disabledReason={useReason} onClick={() => void useInAiChat()} variant="primary">
            {busy === "use" ? "Switching…" : "Use in AI Chat"}
          </Button>
        </>
      ) : (
        <Button disabled={Boolean(busy)} disabledReason={testReason} onClick={() => void testConnection()} variant="primary">
          {busy === "test" ? "Testing securely…" : "Test connection"}
        </Button>
      )}
      footerStart={<Button disabled={Boolean(busy)} onClick={cancel} variant="quiet">Cancel</Button>}
      onClose={cancel}
      showClose={false}
      className="apps-connection-dialog"
      size="standard"
      title="Add a connection"
      titleId="add-connection-title"
    >
      <fieldset className="apps-connection-where" disabled={Boolean(saved)}>
        <legend className="apps-connection-label" id={groupId}>Where the model runs</legend>
        <div className="apps-connection-choices">
          {CHOICE_LIST.map((option) => (
            <label className={option.kind === kind ? "apps-connection-choice is-on" : "apps-connection-choice"} key={option.kind}>
              <input
                checked={option.kind === kind}
                className="apps-connection-choice__radio"
                name={groupId}
                onChange={() => { if (!saved) { setKind(option.kind); setError(""); } }}
                type="radio"
                value={option.kind}
              />
              <span className="apps-connection-choice__text">
                <strong>{option.label}</strong>{" "}
                <small>{option.line}</small>
              </span>
            </label>
          ))}
        </div>
      </fieldset>

      {network && (
        <Select
          disabled={Boolean(saved)}
          hint={NETWORK_PRESETS[preset].help}
          id="add-connection-preset"
          label="Server app"
          onChange={(event) => setPreset(event.target.value as Preset)}
          value={preset}
        >
          {(Object.keys(NETWORK_PRESETS) as Preset[]).map((key) => <option key={key} value={key}>{NETWORK_PRESETS[key].name}</option>)}
        </Select>
      )}

      <div className={needsAddress ? "apps-connection-pair" : undefined}>
        <Input
          id="add-connection-name"
          label="Name"
          maxLength={80}
          onChange={(event) => setName(event.target.value)}
          placeholder={defaultLabel}
          readOnly={Boolean(saved)}
          value={saved ? saved.label : name}
        />
        {needsAddress && (
          <Input
            autoCapitalize="none"
            hint={custom
              ? "HTTPS only, at a public internet address. The base URL that comes before /models and /chat/completions."
              : "Use the computer's network address, not localhost, unless the server runs on this machine. Only private network addresses are accepted."}
            id="add-connection-address"
            inputMode="url"
            label="Address"
            maxLength={500}
            onChange={(event) => setAddress(event.target.value)}
            placeholder={custom ? "https://api.example.com/v1" : NETWORK_PRESETS[preset].placeholder}
            readOnly={Boolean(saved)}
            showCount={false}
            spellCheck={false}
            type="url"
            value={address}
          />
        )}
      </div>

      {!saved && (
        <Input
          autoComplete="new-password"
          hint={network
            ? "Optional: leave blank when the server has no sign-in. Stored encrypted on this machine and never shown again."
            : "Stored encrypted on this machine and never shown again."}
          id="add-connection-key"
          label="API key"
          maxLength={8192}
          onChange={(event) => setApiKey(event.target.value)}
          placeholder="Paste the key"
          spellCheck={false}
          type="password"
          value={apiKey}
        />
      )}

      <Input
        autoCapitalize="none"
        hint={network
          ? "Leave blank to use the first model the server has loaded."
          : "Leave blank to choose from the models the service lists when it is tested."}
        id="add-connection-model-id"
        label="Model ID (optional)"
        maxLength={200}
        onChange={(event) => setModelId(event.target.value)}
        placeholder={network ? "Auto-detect first loaded model" : "Choose after the test"}
        readOnly={Boolean(saved)}
        showCount={false}
        spellCheck={false}
        value={modelId}
      />

      <AppsBanner tone="warning">{destinationNotice(kind)}</AppsBanner>
      {kind === "openrouter" && <p className="apps-fineprint">{OPENROUTER_FORWARDS}</p>}

      {saved && (
        <div className="apps-connection-tested" aria-live="polite">
          <div className="apps-connection-tested__head">
            <StatusPill
              label={!saved.testOk
                ? "Test failed"
                : modelsRead
                  ? `Connected · ${modelsRead.length} model${modelsRead.length === 1 ? "" : "s"}`
                  : "Connected · models not read"}
              tone={saved.testOk ? "success" : "danger"}
            />
            <span className="apps-field-hint">{testedAge(saved.testedAt, now)}</span>
          </div>
          {modelsRead && modelsRead.length > 0 ? (
            <Select id="add-connection-model" label="Default model" onChange={(event) => setModel(event.target.value)} value={model}>
              {saved.chooseModel && <option disabled value="">Choose a model</option>}
              {modelsRead.map((item) => <option key={item} value={item}>{item}</option>)}
            </Select>
          ) : (
            <p className="apps-fineprint">{modelsRead
              ? "The service listed no chat models."
              : "Vaelor could not read which models this service offers."}</p>
          )}
          <p className="apps-fineprint">Saved on this machine. Cancel keeps the connection; remove it in Settings, Connections.</p>
          {saved.aiChatRefusal && <AppsBanner tone="info">{saved.aiChatRefusal}</AppsBanner>}
        </div>
      )}

      <details className="apps-disclosure">
        <summary>Hugging Face token for model downloads</summary>
        <p>Model downloads from Hugging Face use this token. It is not a connection for AI Chat.</p>
        <div className="apps-connection-token">
          <Input
            autoComplete="new-password"
            id="add-connection-hf-token"
            label="Access token"
            maxLength={8192}
            onChange={(event) => setToken(event.target.value)}
            placeholder="Paste your Hugging Face token"
            spellCheck={false}
            type="password"
            value={token}
          />
          <Button
            disabled={Boolean(busy)}
            disabledReason={!isAdministrator ? "An administrator stores tokens." : token.trim().length < 8 ? "Paste the token first." : undefined}
            onClick={() => void storeToken()}
          >
            {busy === "token" ? "Storing…" : "Store token"}
          </Button>
        </div>
        {tokenNote && <AppsBanner tone={tokenNote.refused ? "danger" : "info"}>{tokenNote.text}</AppsBanner>}
      </details>
    </AppsDialog>
  );
}
