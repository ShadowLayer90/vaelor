import { useEffect, useState, type ReactNode } from "react";
import "../styles/apps-manage.css";
import { apiRequest } from "../lib/api";
import { appEndpoint } from "../lib/appPorts";
import type { AppSetup } from "./AppCatalog";
import { AppsInset, AppsKv } from "./appsKit";
import { lifecycleAvailable, primaryPort, readingWithAge, secretLabel, type AppReadings, type LifecycleAction, type ManagedApp, type Tool } from "./manageModel";
import { recoveryPointSummaryText, type RecoveryPointSummary } from "./RecoveryPointList";
import { StatusPill } from "./StatusPill";
import { Button, Notice } from "./ui";

/*
 * The app manager's Overview (VD-200, the approved Manage board): open the
 * app and its lifecycle actions, the readings, a saved configuration the app
 * has not taken yet, and the tiles that lead to Restore points, the console
 * and removal. The catalog's setup guidance for the app sits under the
 * readings; the board has no place of its own for it.
 */

const NOT_READ = "Not read";

/** An image reference wraps after its "/", ":" and "@", never inside a name or tag. */
function imageWithBreaks(image: string): ReactNode {
  return image.split(/(?<=[/:@])/).map((part, index) => <span key={index}>{index > 0 && <wbr />}{part}</span>);
}

export interface AppOverviewProps {
  app: ManagedApp;
  busy: boolean;
  csrfToken: string;
  isAdministrator: boolean;
  /** Why this account may not start, stop, restart or update the app; unset when it may. */
  lifecycleRefusal?: string;
  /** The catalog's setup guidance for this app, null when it names none. */
  setupGuide: AppSetup | null;
  /** The template declares a Vaelor-generated secret (code-server's password). */
  secretEnv: string[];
  readings: AppReadings;
  /** When a saved configuration applies, while the running app has not taken it. */
  configNotApplied: string | null;
  recovery: RecoveryPointSummary;
  /** The durable operation this app is running, with its messages. */
  operation?: ReactNode;
  onLifecycle: (action: LifecycleAction) => void;
  onOpenTool: (tool: Tool) => void;
  onReviewRemoval: () => void;
  onCopied: (message: string, failed?: boolean) => void;
}

export function AppOverviewTab({
  app,
  busy,
  configNotApplied,
  csrfToken,
  isAdministrator,
  lifecycleRefusal,
  operation,
  readings,
  recovery,
  secretEnv,
  setupGuide,
  onCopied,
  onLifecycle,
  onOpenTool,
  onReviewRemoval,
}: AppOverviewProps) {
  const endpoint = appEndpoint(app);
  const httpsRequired = Boolean(setupGuide?.https_required);
  const managed = lifecycleAvailable(app);
  const port = primaryPort(app);
  // One reason for the row, beside it, named by every button it holds back.
  const refusalId = lifecycleRefusal ? `manage-lifecycle-refusal-${app.id}` : undefined;
  const held = busy || Boolean(lifecycleRefusal);
  return (
    <div className="manage-overview">
      <div className="manage-overview__actions">
        {/* An app that needs HTTPS first is not led to its plain http:// address;
            the setup guidance below says what to do instead. */}
        {endpoint && !httpsRequired && <a className="ui-button ui-button--primary" href={endpoint} rel="noopener noreferrer" target="_blank">Open {app.name}</a>}
        {managed && !app.running && <Button aria-describedby={refusalId} disabled={held} onClick={() => onLifecycle("start")} variant={endpoint && !httpsRequired ? "secondary" : "primary"}>Start</Button>}
        {managed && app.running && <Button aria-describedby={refusalId} disabled={held} onClick={() => onLifecycle("restart")}>Restart</Button>}
        {managed && <Button aria-describedby={refusalId} disabled={held} onClick={() => onLifecycle("update")}>Review update</Button>}
        {managed && app.running && <Button aria-describedby={refusalId} disabled={held} onClick={() => onLifecycle("stop")} variant="danger">Stop</Button>}
      </div>
      {managed && lifecycleRefusal && <p className="ui-button__disabled-reason" id={refusalId}>{lifecycleRefusal}</p>}
      {!managed && <p className="manage-tool__quiet">Vaelor discovered this app and does not run it, so it offers no start, stop or update here.</p>}
      {operation}

      <AppsKv
        cells={[
          { label: "CPU", value: readingWithAge(readings.cpu, readings.readAt) ?? NOT_READ },
          { label: "Memory", value: readingWithAge(readings.memory, readings.readAt) ?? NOT_READ },
          { label: "Port", value: port ?? (app.host_network ? "Host network" : "None published") },
          { label: "Image", value: imageWithBreaks(app.image) },
        ]}
        label={`${app.name} readings`}
      />

      <SetupGuidance app={app} csrfToken={csrfToken} endpoint={endpoint} guide={setupGuide} secretEnv={secretEnv} onCopied={onCopied} />

      {configNotApplied && isAdministrator && (
        <div className="manage-overview__config">
          <div className="manage-overview__config-head">
            <h3>Configuration</h3>
            <StatusPill label="Saved, not applied yet" tone="warning" />
          </div>
          <div className="manage-banner manage-banner--warning" role="status">
            <span>Your saved settings apply the next time the app {app.running ? "restarts" : "starts"}.</span>
            <Button disabled={busy} onClick={() => onLifecycle(app.running ? "restart" : "start")}>{app.running ? "Restart now" : "Start now"}</Button>
          </div>
        </div>
      )}

      <div className="manage-overview__tiles">
        {managed && (
          <div className="manage-tile">
            <strong>Restore points</strong>
            <span>{recoveryPointSummaryText(recovery)}</span>
            <Button onClick={() => onOpenTool("restore")}>Open restore points</Button>
          </div>
        )}
        {app.capabilities.console && (
          <div className="manage-tile">
            <strong>App console</strong>
            <span>Run one command inside the app</span>
            <Button onClick={() => onOpenTool("console")}>Open console</Button>
          </div>
        )}
        {managed && isAdministrator && (
          <div className="manage-tile">
            <strong>Remove</strong>
            <span>Checks what depends on it first</span>
            <Button disabled={busy} onClick={onReviewRemoval} variant="danger">Review removal</Button>
          </div>
        )}
      </div>
    </div>
  );
}

/**
 * Post-deploy setup help: the catalog's steps and facts for a blueprint app,
 * and the Vaelor-generated secret when the template declares one (code-server's
 * step points the user here). A discovered app gets the generic line.
 */
function SetupGuidance({
  app,
  csrfToken,
  endpoint,
  guide,
  secretEnv,
  onCopied,
}: {
  app: ManagedApp;
  csrfToken: string;
  endpoint: string | null;
  guide: AppSetup | null;
  secretEnv: string[];
  onCopied: (message: string, failed?: boolean) => void;
}) {
  const [secrets, setSecrets] = useState<Record<string, string> | null>(null);

  /*
   * Only fetch when the template declares secret_env — a needless admin-only
   * request otherwise. A non-admin (403) or any error degrades to a muted "sign
   * in with the password set for this app" line; the secret is never logged or
   * cached beyond this state.
   */
  useEffect(() => {
    setSecrets(null);
    if (secretEnv.length === 0) return;
    let active = true;
    void apiRequest<{ secrets: Record<string, string> }>(`/managed/apps/${app.id}/credentials`)
      .then((data) => { if (active) setSecrets(data.secrets ?? null); })
      .catch(() => { if (active) setSecrets(null); });
    return () => { active = false; };
  }, [app.id, secretEnv, csrfToken]);

  const copy = async (value: string) => {
    try {
      await navigator.clipboard.writeText(value);
      onCopied("Copied to clipboard.");
    } catch {
      onCopied("Copy failed. Select the value and copy it manually.", true);
    }
  };

  const steps = guide?.steps ?? [];
  const hasGuidance = Boolean(guide && (steps.length || guide.login || guide.password_source || guide.https_required || guide.first_run_wizard || guide.note));
  const revealedSecret = secretEnv.length > 0 ? (
    secrets && Object.keys(secrets).length > 0 ? (
      <dl className="manage-setup__facts manage-setup__secrets">
        {Object.entries(secrets).map(([key, value]) => (
          <div key={key}>
            <dt>{secretLabel(key)}</dt>
            <dd className="manage-setup__secret-value">
              <code>{value}</code>
              <Button onClick={() => void copy(value)} variant="quiet">Copy</Button>
            </dd>
          </div>
        ))}
      </dl>
    ) : <p className="manage-tool__quiet">Sign in with the password set for this app.</p>
  ) : null;

  // With nothing to add, the Open button already says how to finish setup.
  if (!hasGuidance && !revealedSecret && endpoint) return null;
  return (
    <AppsInset className="manage-setup" title={hasGuidance || endpoint ? "Finish setup" : "No web interface"}>
      {hasGuidance && guide ? (
        <>
          {steps.length > 0 && <ol className="manage-setup__steps">{steps.map((step, index) => <li key={index}>{step}</li>)}</ol>}
          {guide.https_required && <Notice heading="HTTPS required" severity="warning" standing>This app needs a trusted HTTPS reverse proxy (TLS) before you can sign in or create an account. The plain <code>http://</code> address won&apos;t let you finish setup, so set up TLS and reach the app over HTTPS first.</Notice>}
          {guide.first_run_wizard && <AppsInset detail="This app walks you through the rest of setup on your first visit." icon="settings" title="Opens its own setup wizard" />}
          {(guide.login || guide.password_source || guide.note) && (
            <dl className="manage-setup__facts">
              {guide.login && <div><dt>Default sign-in</dt><dd>{guide.login}</dd></div>}
              {guide.password_source && <div><dt>Password</dt><dd>{guide.password_source}</dd></div>}
              {guide.note && <div><dt>Good to know</dt><dd>{guide.note}</dd></div>}
            </dl>
          )}
        </>
      ) : endpoint
        ? <p className="manage-tool__quiet">Open the app to finish its own setup.</p>
        : <p className="manage-tool__quiet">This app has no web interface to open.</p>}
      {revealedSecret}
    </AppsInset>
  );
}
