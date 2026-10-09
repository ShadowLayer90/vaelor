import { useEffect, useState } from "react";
import "../styles/apps-manage.css";
import "../styles/apps-manage-tools.css";
import { apiRequest } from "../lib/api";
import { formatQuantity } from "../lib/format";
import { AppsInset } from "./appsKit";
import { Button, LoadingLines, Notice, Textarea, type OperationState } from "./ui";

/*
 * The app manager's two configuration tabs (VD-200, the ManageLogsConfig and
 * ManageFiles boards): the Compose configuration and the declared config
 * files. Each tab reads what it edits when it opens and says, in the tab, what
 * a save did - applied, saved but not running yet, or refused.
 */

export interface ManagedFile {
  path: string;
  size: number;
  exists: boolean;
}

const NOT_APPLIED_FALLBACK =
  "The running app keeps its current configuration until you restart, start or update it from Vaelor.";

/** What a tab says after its own action: the message and how it reads. */
export interface TabMessage { text: string; state: OperationState }

export const messageText = (error: unknown, fallback: string) => error instanceof Error && error.message ? error.message : fallback;

/**
 * Save an app's Compose configuration and say what the running app now runs
 * (W4d-D12). A save writes the file only; start, restart and update apply it.
 * `notApplied` is the sentence saying when it will apply, or null when the
 * backend reported the change in effect.
 */
export async function saveAppConfiguration(
  appId: string,
  content: string,
  csrfToken: string,
): Promise<{ message: string; state: OperationState; notApplied: string | null }> {
  const saved = await apiRequest<{ applied?: boolean; applies_when?: string }>(
    `/managed/apps/${appId}/config`,
    { method: "PUT", body: JSON.stringify({ content }) },
    csrfToken,
  );
  if (saved?.applied === false) {
    const when = saved.applies_when || NOT_APPLIED_FALLBACK;
    return { message: `Configuration validated and saved. A backup was created. ${when}`, state: "pending", notApplied: when };
  }
  return { message: "Configuration validated, saved and applied. A backup was created.", state: "success", notApplied: null };
}

/** A tab's own answer, in the board's colours: green done, red refused, blue under way. */
export function TabNotice({ message }: { message: TabMessage | null }) {
  if (!message?.text) return null;
  const severity = message.state === "error" ? "danger" : message.state === "success" ? "success" : message.state === "warning" ? "warning" : "info";
  return <Notice severity={severity}>{message.text}</Notice>;
}

export function WorkloadConfigurationTab({
  appId,
  csrfToken,
  isAdministrator,
  notApplied,
  running,
  onApply,
  onSaved,
}: {
  appId: string;
  csrfToken: string;
  isAdministrator: boolean;
  /** When the saved configuration applies, while the running app has not taken it. */
  notApplied: string | null;
  running: boolean;
  onApply: () => void;
  /** The save's answer: when it applies (null once the backend says it runs now). */
  onSaved: (notApplied: string | null) => void;
}) {
  const [config, setConfig] = useState("");
  const [load, setLoad] = useState<"loading" | "ok" | "error">("loading");
  const [busy, setBusy] = useState(false);
  const [message, setMessage] = useState<TabMessage | null>(null);

  useEffect(() => {
    let active = true;
    setLoad("loading");
    void apiRequest<{ content: string }>(`/managed/apps/${appId}/config`)
      .then((data) => { if (active) { setConfig(data.content); setLoad("ok"); } })
      .catch(() => { if (active) setLoad("error"); });
    return () => { active = false; };
  }, [appId]);

  const save = async () => {
    setBusy(true);
    setMessage(null);
    try {
      const saved = await saveAppConfiguration(appId, config, csrfToken);
      onSaved(saved.notApplied);
      // A save that is not running yet is said once, by the banner below.
      setMessage(saved.notApplied ? null : { text: saved.message, state: saved.state });
    } catch (error) {
      setMessage({ text: messageText(error, "Configuration could not be saved."), state: "error" });
    } finally {
      setBusy(false);
    }
  };

  if (load === "error") return <Notice severity="danger">This tool is unavailable: the configuration could not be read.</Notice>;
  return (
    <div className="manage-tool">
      <AppsInset detail="Vaelor validates changes and creates a backup before saving." icon="shield" title="Safe configuration editor" />
      {load === "loading" ? <LoadingLines label="Reading the configuration" lines={3} /> : (
        <>
          <Textarea className="manage-code-editor" disabled={busy || !isAdministrator} label="Configuration YAML" onChange={(event) => setConfig(event.target.value)} spellCheck={false} value={config} />
          {!isAdministrator && <p className="manage-tool__quiet">The editor is read-only.</p>}
          <div className="manage-tool__row">
            <Button busy={busy} disabledReason={isAdministrator ? undefined : "Administrator access is required to change an app configuration."} onClick={() => void save()} variant="primary">Validate and save</Button>
          </div>
        </>
      )}
      <TabNotice message={message} />
      {notApplied && isAdministrator && (
        <div className="manage-banner manage-banner--warning" role="status">
          <span><strong>Saved, not applied yet</strong> {notApplied}</span>
          <Button disabled={busy} onClick={onApply}>{running ? "Restart to apply" : "Start to apply"}</Button>
        </div>
      )}
    </div>
  );
}

export function WorkloadConfigFilesTab({
  appId,
  csrfToken,
  isAdministrator,
}: {
  appId: string;
  csrfToken: string;
  isAdministrator: boolean;
}) {
  const [files, setFiles] = useState<ManagedFile[]>([]);
  const [load, setLoad] = useState<"loading" | "ok" | "error">("loading");
  const [selectedFile, setSelectedFile] = useState<string | null>(null);
  const [fileContent, setFileContent] = useState("");
  const [busy, setBusy] = useState(false);
  const [message, setMessage] = useState<TabMessage | null>(null);

  useEffect(() => {
    let active = true;
    setLoad("loading");
    setSelectedFile(null);
    void apiRequest<{ files: ManagedFile[] }>(`/managed/apps/${appId}/files`)
      .then((data) => { if (active) { setFiles(data.files ?? []); setLoad("ok"); } })
      .catch(() => { if (active) setLoad("error"); });
    return () => { active = false; };
  }, [appId]);

  const openFile = async (path: string) => {
    setSelectedFile(path);
    setFileContent("");
    setBusy(true);
    setMessage(null);
    try {
      const data = await apiRequest<{ path: string; content: string }>(
        `/managed/apps/${appId}/files/content?path=${encodeURIComponent(path)}`,
      );
      setFileContent(data.content);
    } catch (error) {
      setMessage({ text: messageText(error, "That file could not be opened."), state: "error" });
    } finally {
      setBusy(false);
    }
  };

  const saveFile = async () => {
    if (!selectedFile) return;
    setBusy(true);
    setMessage(null);
    try {
      await apiRequest(
        `/managed/apps/${appId}/files/content`,
        { method: "PUT", body: JSON.stringify({ path: selectedFile, content: fileContent }) },
        csrfToken,
      );
      setMessage({ text: `${selectedFile} saved.`, state: "success" });
    } catch (error) {
      setMessage({ text: messageText(error, "The file could not be saved."), state: "error" });
    } finally {
      setBusy(false);
    }
  };

  return (
    <div className="manage-tool">
      <AppsInset detail="Edit the files this app reads. Vaelor writes only the files listed here." icon="shield" title="Configuration files" />
      {load === "error" ? <Notice severity="danger">This app&apos;s files could not be listed.</Notice>
        : load === "loading" ? <LoadingLines label="Listing the configuration files" />
          : files.length === 0 ? <p className="manage-tool__quiet">No editable files were found for this app.</p>
            : (
              <div className="manage-config-files">
                <ul aria-label="Configuration files" className="manage-config-files__list">
                  {files.map((file) => (
                    <li key={file.path}>
                      <Button
                        aria-pressed={selectedFile === file.path}
                        className={selectedFile === file.path ? "manage-file-choice is-chosen" : "manage-file-choice"}
                        disabled={busy}
                        onClick={() => void openFile(file.path)}
                        variant="quiet"
                      >
                        <strong>{file.path}</strong>
                        <span>{file.exists ? formatQuantity(file.size, "used") : "Not created yet"}</span>
                      </Button>
                    </li>
                  ))}
                </ul>
                <div className="manage-config-files__editor">
                  {selectedFile ? (
                    <>
                      <Textarea className="manage-code-editor manage-code-editor--short" disabled={busy || !isAdministrator} label={`Editing ${selectedFile}`} onChange={(event) => setFileContent(event.target.value)} spellCheck={false} value={fileContent} />
                      <div className="manage-tool__row">
                        <Button disabled={busy} disabledReason={isAdministrator ? undefined : "Administrator access is required to edit configuration files."} onClick={() => void saveFile()} variant="primary">Save file</Button>
                      </div>
                    </>
                  ) : <p className="manage-tool__quiet">Choose a file to edit it.</p>}
                  <TabNotice message={message} />
                </div>
              </div>
            )}
    </div>
  );
}
