import { useEffect, useState, type ReactNode } from "react";
import "../styles/apps-manage.css";
import { apiRequest } from "../lib/api";
import type { Session } from "../types";
import type { AppSetup, AppTemplate } from "./AppCatalog";
import { AppFileManager } from "./AppFileManager";
import { AppLogViewer } from "./AppLogViewer";
import { AppConsoleTab, AppRemoteDesktopTab } from "./AppManagerTools";
import { AppOverviewTab } from "./AppOverviewTab";
import { AppsIconTile } from "./appsKit";
import { appHeaderPill, fileManagerAvailable, lifecycleAvailable, parseResourceUse, type AppReadings, type LifecycleAction, type ManagedApp, type Tool } from "./manageModel";
import { RecoveryPointList, recoveryPointSummary, recoveryReadRefusal, useRecoveryPoints } from "./RecoveryPointList";
import { StatusPill } from "./StatusPill";
import { Button, TabSet, type TabItem } from "./ui";
import { WorkloadConfigFilesTab, WorkloadConfigurationTab } from "./WorkloadConfigurationTabs";

/*
 * The app manager (VD-200, the Manage* boards): the panel on the right of the
 * Manage tab for the chosen app. Its tools are tabs under the header; a tab
 * the app does not offer stays in the strip, disabled with its reason, while
 * Config files and Files are left out when the app has none.
 */

interface CatalogFacts { setupGuide: AppSetup | null; secretEnv: string[]; configFiles: string[] }
const NO_CATALOG: CatalogFacts = { setupGuide: null, secretEnv: [], configFiles: [] };
const NO_READINGS: AppReadings = { cpu: null, memory: null };

/**
 * The catalog's facts for a blueprint app: its setup guidance, generated
 * secrets and editable config files. Only blueprint apps carry a template id,
 * so a discovered container makes no request. Any failure (older server,
 * offline catalog) reads as "the catalog names nothing" in this read-only
 * helper, never as an error.
 */
function useCatalogFacts(templateId: string | null | undefined): CatalogFacts {
  const [facts, setFacts] = useState<CatalogFacts>(NO_CATALOG);
  useEffect(() => {
    setFacts(NO_CATALOG);
    if (!templateId) return;
    let active = true;
    void apiRequest<AppTemplate[]>("/apps/catalog")
      .then((templates) => {
        if (!active) return;
        const template = templates.find((candidate) => candidate.id === templateId);
        setFacts({ setupGuide: template?.setup ?? null, secretEnv: template?.secret_env ?? [], configFiles: template?.config_files ?? [] });
      })
      .catch(() => { if (active) setFacts(NO_CATALOG); });
    return () => { active = false; };
  }, [templateId]);
  return facts;
}

/**
 * CPU and memory, read once when the manager opens a running app, through the
 * same audited Resource use diagnostic the Console runs. A viewer cannot run
 * it, a stopped app has no reading, and a failed read stays "Not read".
 */
function useAppReadings(app: ManagedApp, session: Session): [AppReadings, (output: string) => void] {
  const [readings, setReadings] = useState<AppReadings>(NO_READINGS);
  const canRead = app.running && session.user.role !== "viewer";
  useEffect(() => {
    setReadings(NO_READINGS);
    if (!canRead) return;
    let active = true;
    void apiRequest<{ output: string }>(
      `/managed/apps/${app.id}/diagnostics`,
      { method: "POST", body: JSON.stringify({ tool: "stats" }) },
      session.csrf_token,
    )
      .then((data) => { if (active) setReadings({ ...parseResourceUse(data.output || ""), readAt: Date.now() }); })
      .catch(() => { if (active) setReadings(NO_READINGS); });
    return () => { active = false; };
  }, [app.id, canRead, session.csrf_token]);
  return [readings, (output) => setReadings({ ...parseResourceUse(output), readAt: Date.now() })];
}

export function AppManagerPanel({
  app,
  busy,
  configNotApplied,
  notice,
  operation,
  recoveryRefresh,
  session,
  tool,
  onClose,
  onConfigSaved,
  onCopied,
  onLifecycle,
  onReviewRemoval,
  onSelectTool,
}: {
  app: ManagedApp;
  busy: boolean;
  configNotApplied: string | null;
  /** The manager's own messages (lifecycle, removal), shown above every tab. */
  notice?: ReactNode;
  /** The durable operation running for this app. */
  operation?: ReactNode;
  recoveryRefresh: number;
  session: Session;
  tool: Tool;
  onClose: () => void;
  onConfigSaved: (notApplied: string | null) => void;
  onCopied: (message: string, failed?: boolean) => void;
  onLifecycle: (action: LifecycleAction) => void;
  onReviewRemoval: () => void;
  onSelectTool: (tool: Tool) => void;
}) {
  const isAdministrator = session.user.role === "administrator";
  // Logs and the diagnostics are served from an operator up; a viewer is told
  // so beside the control instead of pressing it into a refusal.
  const operatorRefusal = (what: string) => (session.user.role === "viewer" ? `Operator access is required to ${what}.` : undefined);
  const catalog = useCatalogFacts(app.template_id);
  const [readings, setResourceUse] = useAppReadings(app, session);
  const managed = lifecycleAvailable(app);
  // One read feeds the Overview tile and the Restore points tab (VD-200 apps S3).
  // A viewer may not list them, so it is not asked (VD-200 apps verify).
  const recoveryRefusal = recoveryReadRefusal(session.user.role);
  const recoveryReading = useRecoveryPoints(recoveryRefresh, Boolean(managed && app.project) && !recoveryRefusal);
  const recovery = recoveryRefusal ? { state: "refused" as const, count: 0, newest: null } : recoveryPointSummary(recoveryReading, app.project);
  // Start, stop, restart, update and a checkpoint are each a POST /jobs, which
  // the server takes from an operator up (api_workload_routes.py).
  const lifecycleRefusal = operatorRefusal("start, stop, restart or update an app");
  const pill = appHeaderPill(app);
  const files = fileManagerAvailable(app);


  return (
    <section aria-labelledby="app-manager-title" className="ui-card manage-panel app-manager">
      <header className="manage-panel__head">
        <div className="manage-panel__identity">
          <AppsIconTile accent name="display" />
          <div className="manage-panel__titles">
            <span className="apps-eyebrow">App manager</span>
            <h2 id="app-manager-title">{app.name}</h2>
          </div>
        </div>
        <div className="manage-panel__state">
          <StatusPill description={pill.description} label={pill.label} tone={pill.tone} />
          <Button aria-label="Close the app manager" className="manage-panel__close" onClick={onClose} variant="quiet">Close</Button>
        </div>
      </header>
      <TabSet
        // Every tab is listed literally here (the UI inventory reads this table);
        // Config files and Files are then left out when the app has none.
        items={([
          { id: "overview", label: "Overview" },
          { id: "logs", label: "Logs", disabledReason: app.capabilities.logs ? undefined : "This app does not report logs to Vaelor." },
          { id: "configuration", label: "Configuration", disabledReason: app.capabilities.configuration ? undefined : "Only an app Vaelor manages has an editable configuration." },
          { id: "files", label: "Config files" },
          { id: "filemanager", label: "Files" },
          { id: "console", label: "Console", disabledReason: app.capabilities.console ? undefined : "This app offers no console." },
          { id: "remote", label: "Remote desktop", disabledReason: app.capabilities.remote_desktop ? undefined : "This app offers no remote desktop." },
          { id: "restore", label: "Restore points", disabledReason: managed ? undefined : "Only an app Vaelor manages has restore points." },
        ] as TabItem[]).filter((tab) => (tab.id !== "files" || catalog.configFiles.length > 0) && (tab.id !== "filemanager" || files))}
        label={`${app.name} tools`}
        listClassName="manage-panel__tabs"
        onSelect={(id) => onSelectTool(id as Tool)}
        panelClassName="manage-panel__body"
        selectedId={tool}
      >
        {notice}
        {tool === "overview" && (
          <AppOverviewTab
            app={app}
            busy={busy}
            configNotApplied={configNotApplied}
            csrfToken={session.csrf_token}
            isAdministrator={isAdministrator}
            lifecycleRefusal={lifecycleRefusal}
            operation={operation}
            readings={readings}
            recovery={recovery}
            secretEnv={catalog.secretEnv}
            setupGuide={catalog.setupGuide}
            onCopied={onCopied}
            onLifecycle={onLifecycle}
            onOpenTool={onSelectTool}
            onReviewRemoval={onReviewRemoval}
          />
        )}
        {tool === "logs" && <AppLogViewer appId={app.id} appName={app.name} refusal={operatorRefusal("read an app's logs")} />}
        {tool === "configuration" && (
          <WorkloadConfigurationTab
            appId={app.id}
            csrfToken={session.csrf_token}
            isAdministrator={isAdministrator}
            notApplied={managed ? configNotApplied : null}
            running={app.running}
            onApply={() => onLifecycle(app.running ? "restart" : "start")}
            onSaved={onConfigSaved}
          />
        )}
        {tool === "files" && <WorkloadConfigFilesTab appId={app.id} csrfToken={session.csrf_token} isAdministrator={isAdministrator} />}
        {tool === "filemanager" && files && <AppFileManager appId={app.id} csrfToken={session.csrf_token} role={session.user.role} />}
        {tool === "console" && <AppConsoleTab appId={app.id} csrfToken={session.csrf_token} diagnosticRefusal={operatorRefusal("run diagnostics")} isAdministrator={isAdministrator} onResourceUse={setResourceUse} />}
        {tool === "remote" && <AppRemoteDesktopTab appId={app.id} appName={app.name} csrfToken={session.csrf_token} />}
        {tool === "restore" && managed && app.project && (
          <>
            {operation}
            <RecoveryPointList
              headerAction={<Button disabled={busy} disabledReason={operatorRefusal("create a checkpoint")} onClick={() => onLifecycle("backup")} variant="primary">Review checkpoint</Button>}
              project={app.project}
              reading={recoveryReading}
              refreshSignal={recoveryRefresh}
              session={session}
            />
          </>
        )}
      </TabSet>
    </section>
  );
}
