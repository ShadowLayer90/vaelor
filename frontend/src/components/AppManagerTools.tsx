import { useState } from "react";
import "../styles/apps-manage.css";
import "../styles/apps-manage-tools.css";
import { apiRequest } from "../lib/api";
import { AppsInset } from "./appsKit";
import { Icon, ICON_SIZE } from "./Icon";
import { Button, Input, Notice } from "./ui";
import { messageText } from "./WorkloadConfigurationTabs";

/*
 * The app manager's Console and Remote desktop tabs (VD-200, the
 * ManageConsoleRestore board). Each says its own refusal in the tab.
 */

export function AppConsoleTab({
  appId,
  csrfToken,
  isAdministrator,
  diagnosticRefusal,
  onResourceUse,
}: {
  appId: string;
  csrfToken: string;
  isAdministrator: boolean;
  /**
   * Why this role cannot run the diagnostics (the server takes them from an
   * operator up), or undefined when it can. Said beside the buttons.
   */
  diagnosticRefusal?: string;
  /** The Resource use output, so the Overview's CPU and Memory can read it too. */
  onResourceUse?: (output: string) => void;
}) {
  const [command, setCommand] = useState("");
  const [output, setOutput] = useState("");
  const [busy, setBusy] = useState(false);
  const [error, setError] = useState("");
  const refusalId = diagnosticRefusal ? `manage-diagnostics-refusal-${appId}` : undefined;

  const runDiagnostic = async (diagnostic: "stats" | "processes") => {
    setBusy(true);
    setError("");
    try {
      const data = await apiRequest<{ output: string }>(
        `/managed/apps/${appId}/diagnostics`,
        { method: "POST", body: JSON.stringify({ tool: diagnostic }) },
        csrfToken,
      );
      setOutput(data.output);
      if (diagnostic === "stats") onResourceUse?.(data.output);
    } catch (caught) {
      setError(messageText(caught, "The diagnostic could not run."));
    } finally {
      setBusy(false);
    }
  };

  const runCommand = async () => {
    if (!command.trim()) return;
    setBusy(true);
    setError("");
    try {
      const data = await apiRequest<{ output: string; exit_code: number; truncated: boolean }>(
        `/managed/apps/${appId}/exec`,
        { method: "POST", body: JSON.stringify({ command }) },
        csrfToken,
      );
      const status = `\n[exit ${data.exit_code}${data.truncated ? ", output truncated" : ""}]`;
      setOutput((data.output || "") + status);
    } catch (caught) {
      setError(messageText(caught, "The command could not run."));
    } finally {
      setBusy(false);
    }
  };

  return (
    <div className="manage-tool">
      <AppsInset
        detail="Run a command inside this app's container - to reset a password or change a setting. Administrator only; each command is audited and confined to the container."
        icon="terminal"
        title="App console"
      />
      {isAdministrator ? (
        <div className="manage-console-run">
          <Input
            disabled={busy}
            label="Command"
            onChange={(event) => setCommand(event.target.value)}
            onKeyDown={(event) => { if (event.key === "Enter") void runCommand(); }}
            placeholder="e.g. filebrowser users update admin --password NEW"
            value={command}
          />
          <Button disabled={busy || !command.trim()} onClick={() => void runCommand()} variant="primary">Run</Button>
        </div>
      ) : (
        <AppsInset detail="Administrator access is required to run commands in an app." icon="lock">
          {!diagnosticRefusal && <p className="manage-tool__quiet">Resource use and Running processes stay available.</p>}
        </AppsInset>
      )}
      <div className="manage-tool__row">
        <Button aria-describedby={refusalId} disabled={busy || Boolean(diagnosticRefusal)} onClick={() => void runDiagnostic("stats")}>Resource use</Button>
        <Button aria-describedby={refusalId} disabled={busy || Boolean(diagnosticRefusal)} onClick={() => void runDiagnostic("processes")}>Running processes</Button>
      </div>
      {/* One reason for the pair, beside them (owner rule: not a tooltip, not twice). */}
      {diagnosticRefusal && <p className="ui-button__disabled-reason" id={refusalId}>{diagnosticRefusal}</p>}
      {error && <Notice severity="danger">{error}</Notice>}
      <pre className="manage-console-output">{output || (diagnosticRefusal ? "No diagnostic has run." : "Run a command, or choose a diagnostic below.")}</pre>
    </div>
  );
}

export function AppRemoteDesktopTab({
  appId,
  appName,
  csrfToken,
}: {
  appId: string;
  appName: string;
  csrfToken: string;
}) {
  const [remoteUrl, setRemoteUrl] = useState("");
  const [busy, setBusy] = useState(false);
  const [error, setError] = useState("");

  const start = async () => {
    setBusy(true);
    setError("");
    try {
      const data = await apiRequest<{ url: string }>(
        `/managed/apps/${appId}/remote-desktop`,
        { method: "POST", body: "{}" },
        csrfToken,
      );
      setRemoteUrl(data.url);
    } catch (caught) {
      setError(messageText(caught, "Remote Desktop could not start."));
    } finally {
      setBusy(false);
    }
  };

  if (remoteUrl) {
    return (
      <div className="manage-tool">
        <AppsInset detail="The access token is short-lived and valid for one connection." icon="display" title="Remote Desktop connected" />
        <iframe
          allow="clipboard-read; clipboard-write; fullscreen"
          className="manage-remote-frame"
          sandbox="allow-forms allow-same-origin allow-scripts"
          src={remoteUrl}
          title={`${appName} Remote Desktop`}
        />
        <div className="manage-tool__row"><Button onClick={() => setRemoteUrl("")}>End session</Button></div>
      </div>
    );
  }
  return (
    <div className="manage-tool">
      <div className="manage-remote-ready">
        <span aria-hidden="true" className="apps-ico apps-ico--accent"><Icon name="display" size={ICON_SIZE.nav} /></span>
        <h3>VNC desktop ready</h3>
        <p>Vaelor will create a one-use 90-second connection ticket, then open the desktop inside this page.</p>
        <Button busy={busy} onClick={() => void start()} variant="primary">{busy ? "Starting…" : "Open remote desktop"}</Button>
      </div>
      {error && <Notice severity="danger">{error}</Notice>}
    </div>
  );
}
