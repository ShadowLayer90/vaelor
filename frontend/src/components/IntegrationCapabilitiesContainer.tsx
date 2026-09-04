import { useCallback, useEffect, useMemo, useState } from "react";
import { apiRequest } from "../lib/api";
import { IntegrationCapabilities } from "./IntegrationCapabilities";
import type { AgentProfile } from "./agentTypes";
import type {
  CompatibleAgentVersion,
  IntegrationAppStatus,
  IntegrationCapabilitiesData,
  IntegrationConnection,
  IntegrationConnectionStatus,
  IntegrationDependent,
  IntegrationGrant,
  IntegrationGrantPreview,
  IntegrationGrantSelection,
  IntegrationGrantStatus,
  IntegrationOperation,
  IntegrationOperationAvailability,
  IntegrationOperationKind,
  IntegrationOperationRisk,
} from "./integrationCapabilitiesTypes";

type JsonRecord = Record<string, unknown>;

type InstalledApp = {
  id: string;
  name: string;
  status: IntegrationAppStatus;
  operationCount: number;
};

type AppsResponse = { items?: unknown[] };

const APP_STATUSES: IntegrationAppStatus[] = ["active", "degraded", "stopped", "incompatible", "removed"];
const CONNECTION_STATUSES: IntegrationConnectionStatus[] = ["pending", "healthy", "degraded", "expired", "revoked"];
const GRANT_STATUSES: IntegrationGrantStatus[] = ["active", "blocked", "incompatible", "revoked"];
const OPERATION_KINDS: IntegrationOperationKind[] = ["read", "write"];
const OPERATION_RISKS: IntegrationOperationRisk[] = ["low", "medium", "high"];
const OPERATION_AVAILABILITY: IntegrationOperationAvailability[] = ["available", "incompatible", "removed"];

function record(value: unknown): JsonRecord {
  return value && typeof value === "object" && !Array.isArray(value) ? value as JsonRecord : {};
}

function text(value: unknown, fallback = ""): string {
  return typeof value === "string" && value.trim() ? value.trim() : fallback;
}

function list(value: unknown): unknown[] {
  return Array.isArray(value) ? value : [];
}

function strings(value: unknown): string[] {
  return list(value).filter((item): item is string => typeof item === "string" && Boolean(item.trim()));
}

function enumValue<T extends string>(value: unknown, allowed: readonly T[], fallback: T): T {
  return typeof value === "string" && allowed.includes(value as T) ? value as T : fallback;
}

function mapOperation(value: unknown): IntegrationOperation | null {
  const item = record(value);
  const id = text(item.id);
  if (!id) return null;
  const availability = item.availability == null
    ? undefined
    : enumValue(item.availability, OPERATION_AVAILABILITY, "incompatible");
  return {
    id,
    label: text(item.label, id),
    description: text(item.description, "Published operation from the installed app."),
    kind: enumValue(item.kind, OPERATION_KINDS, "read"),
    risk: enumValue(item.risk, OPERATION_RISKS, "medium"),
    ...(availability ? { availability } : {}),
    ...(text(item.unavailableReason) ? { unavailableReason: text(item.unavailableReason) } : {}),
  };
}

function mapConnection(value: unknown): IntegrationConnection | null {
  const item = record(value);
  const id = text(item.id);
  if (!id) return null;
  return {
    id,
    label: text(item.label, "Integration connection"),
    status: enumValue(item.status, CONNECTION_STATUSES, "pending"),
    scopes: strings(item.scopes),
    ...(text(item.expiresAt) ? { expiresAt: text(item.expiresAt) } : {}),
    ...(text(item.lastTestedAt) ? { lastTestedAt: text(item.lastTestedAt) } : {}),
    ...(text(item.issue) ? { issue: text(item.issue) } : {}),
  };
}

function mapAgentVersion(value: unknown, selectedAgent: AgentProfile): CompatibleAgentVersion | null {
  const item = record(value);
  if (text(item.agentId) !== selectedAgent.id) return null;
  const versionId = text(item.versionId);
  if (!versionId) return null;
  return {
    agentId: selectedAgent.id,
    agentName: text(item.agentName, selectedAgent.name),
    versionId,
    versionLabel: text(item.versionLabel, `v${selectedAgent.version ?? versionId}`),
    compatibleOperationIds: strings(item.compatibleOperationIds),
    status: item.status === "archived" || item.status === "draft" ? item.status : "active",
  };
}

function mapDependent(value: unknown): IntegrationDependent | null {
  const item = record(value);
  const id = text(item.id);
  if (!id) return null;
  const kind = item.kind === "task" || item.kind === "automation" ? item.kind : "agent";
  const status = item.status === "blocked" || item.status === "stopped" ? item.status : "active";
  return {
    id,
    label: text(item.label, id),
    kind,
    status,
    impact: text(item.impact, "This dependent uses the installed app."),
    recoveryAction: text(item.recoveryAction, "Review the grant and reconnect it if needed."),
  };
}

function mapGrant(value: unknown): IntegrationGrant | undefined {
  const item = record(value);
  const id = text(item.id);
  if (!id) return undefined;
  return {
    id,
    status: enumValue(item.status, GRANT_STATUSES, "blocked"),
    agentName: text(item.agentName, "Custom agent"),
    agentVersionLabel: text(item.agentVersionLabel, "Pinned version"),
    operationIds: strings(item.operationIds),
    ...(text(item.connectionId) ? { connectionId: text(item.connectionId) } : {}),
    ...(text(item.blockedReason) ? { blockedReason: text(item.blockedReason) } : {}),
  };
}

function mapDetail(value: unknown, selectedAgent: AgentProfile): IntegrationCapabilitiesData | null {
  const item = record(value);
  const appInstanceId = text(item.appInstanceId);
  if (!appInstanceId) return null;
  const operations = list(item.operations).map(mapOperation).filter((entry): entry is IntegrationOperation => Boolean(entry));
  const agentVersions = list(item.agentVersions).map((entry) => mapAgentVersion(entry, selectedAgent)).filter((entry): entry is CompatibleAgentVersion => Boolean(entry));
  const connections = list(item.connections).map(mapConnection).filter((entry): entry is IntegrationConnection => Boolean(entry));
  const dependents = list(item.dependents).map(mapDependent).filter((entry): entry is IntegrationDependent => Boolean(entry));
  return {
    appInstanceId,
    appName: text(item.appName, "Installed app"),
    appVersion: text(item.appVersion, "Unknown version"),
    manifestVersion: text(item.manifestVersion, "Unknown manifest"),
    status: enumValue(item.status, APP_STATUSES, "degraded"),
    healthSummary: text(item.healthSummary, "The app health state is not available."),
    ...(text(item.compatibilitySummary) ? { compatibilitySummary: text(item.compatibilitySummary) } : {}),
    connectionRequired: Boolean(item.connectionRequired),
    operations,
    connections,
    agentVersions,
    existingGrant: mapGrant(item.existingGrant),
    dependents,
    recoveryActions: strings(item.recoveryActions),
  };
}

function mapInstalledApp(value: unknown): InstalledApp | null {
  const item = record(value);
  const id = text(item.appInstanceId);
  const operationCount = list(item.operations).length;
  if (!id || operationCount === 0) return null;
  return {
    id,
    name: text(item.appName, "Installed app"),
    status: enumValue(item.status, APP_STATUSES, "degraded"),
    operationCount,
  };
}

function errorMessage(error: unknown, fallback: string): string {
  return error instanceof Error && error.message ? error.message : fallback;
}

export function IntegrationCapabilitiesContainer({
  agent,
  csrfToken,
  onChanged,
}: {
  agent: AgentProfile;
  csrfToken: string;
  onChanged: () => void;
}) {
  const [apps, setApps] = useState<InstalledApp[]>([]);
  const [selectedAppId, setSelectedAppId] = useState("");
  const [data, setData] = useState<IntegrationCapabilitiesData | null>(null);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);
  const [preview, setPreview] = useState<IntegrationGrantPreview | null>(null);
  const [previewing, setPreviewing] = useState(false);
  const [saving, setSaving] = useState(false);
  const [revoking, setRevoking] = useState(false);
  const [testingConnectionId, setTestingConnectionId] = useState<string | null>(null);
  const [message, setMessage] = useState<string | null>(null);
  const [actionError, setActionError] = useState<string | null>(null);

  const selectedApp = useMemo(() => apps.find((item) => item.id === selectedAppId), [apps, selectedAppId]);

  const loadDetail = useCallback(async (appId: string, cancelled?: () => boolean) => {
    if (!appId) {
      setData(null);
      setLoading(false);
      return;
    }
    setLoading(true);
    setError(null);
    try {
      const response = await apiRequest<unknown>(`/integrations/apps/${encodeURIComponent(appId)}`);
      if (cancelled?.()) return;
      const mapped = mapDetail(response, agent);
      if (!mapped) throw new Error("The installed app returned an incomplete capability description.");
      setData(mapped);
      setPreview(null);
    } catch (loadError) {
      if (!cancelled?.()) {
        setData(null);
        setError(errorMessage(loadError, "The installed app capabilities could not be loaded."));
      }
    } finally {
      if (!cancelled?.()) setLoading(false);
    }
  }, [agent]);

  const loadApps = useCallback(async (isCancelled?: () => boolean) => {
    setLoading(true);
    setError(null);
    try {
      const response = await apiRequest<AppsResponse>("/integrations/apps");
      const nextApps = list(response.items).map(mapInstalledApp).filter((entry): entry is InstalledApp => Boolean(entry));
      if (isCancelled?.()) return;
      setApps(nextApps);
      setSelectedAppId((current) => nextApps.some((item) => item.id === current) ? current : nextApps[0]?.id ?? "");
      if (nextApps.length === 0) {
        setData(null);
        setLoading(false);
      }
    } catch (loadError) {
      if (!isCancelled?.()) {
        setApps([]);
        setSelectedAppId("");
        setData(null);
        setError(errorMessage(loadError, "Installed app capabilities could not be listed."));
        setLoading(false);
      }
    }
  }, []);

  useEffect(() => {
    let cancelled = false;
    void loadApps(() => cancelled);
    return () => { cancelled = true; };
  }, [agent.id, loadApps]);

  useEffect(() => {
    let cancelled = false;
    if (selectedAppId) void loadDetail(selectedAppId, () => cancelled);
    return () => { cancelled = true; };
  }, [loadDetail, selectedAppId]);

  const refreshDetail = useCallback(async () => {
    if (selectedAppId) await loadDetail(selectedAppId);
  }, [loadDetail, selectedAppId]);

  const testConnection = async (connectionId: string) => {
    setTestingConnectionId(connectionId); setActionError(null); setMessage(null);
    try {
      await apiRequest(`/integrations/connections/${encodeURIComponent(connectionId)}/test`, { method: "POST" }, csrfToken);
      setMessage("Connection test completed. Capability status was refreshed.");
      await refreshDetail();
    } catch (requestError) {
      setActionError(errorMessage(requestError, "The connection test could not be completed."));
    } finally { setTestingConnectionId(null); }
  };

  const previewGrant = async (selection: IntegrationGrantSelection) => {
    setPreviewing(true); setActionError(null); setMessage(null);
    try {
      const result = await apiRequest<IntegrationGrantPreview>(
        `/assistant/custom-agents/${encodeURIComponent(agent.id)}/app-grants/preview`,
        { method: "POST", body: JSON.stringify(selection) },
        csrfToken,
      );
      setPreview(result);
    } catch (requestError) {
      setActionError(errorMessage(requestError, "The grant preview could not be prepared. Refresh and try again."));
    } finally { setPreviewing(false); }
  };

  const saveGrant = async (selection: IntegrationGrantSelection & { previewId: string }) => {
    setSaving(true); setActionError(null); setMessage(null);
    try {
      await apiRequest(`/assistant/custom-agents/${encodeURIComponent(agent.id)}/app-grants`, { method: "POST", body: JSON.stringify(selection) }, csrfToken);
      setMessage("Access saved as a new custom-agent version. The selected app was refreshed.");
      setPreview(null);
      await Promise.resolve(onChanged());
      await refreshDetail();
    } catch (requestError) {
      setActionError(errorMessage(requestError, "The access grant could not be saved. Refresh and review the current version."));
    } finally { setSaving(false); }
  };

  const revokeGrant = async (grantId: string) => {
    setRevoking(true); setActionError(null); setMessage(null);
    try {
      await apiRequest(`/assistant/custom-agents/${encodeURIComponent(agent.id)}/app-grants/${encodeURIComponent(grantId)}`, { method: "DELETE" }, csrfToken);
      setMessage("Access revoked. Dependents now fail closed until explicitly reconnected.");
      await refreshDetail();
    } catch (requestError) {
      setActionError(errorMessage(requestError, "The access grant could not be revoked."));
    } finally { setRevoking(false); }
  };

  const createConnectionNotice = () => {
    setActionError(null);
    setMessage("Create or select an opaque broker credential in the credential broker, then return here to test it. Plaintext secrets are never collected in this view.");
  };

  return (
    <section className="custom-agent-app-access" aria-labelledby="custom-agent-app-access-title">
      <div className="custom-agent-app-access__heading">
        <div>
          <span className="page-eyebrow">Capability control</span>
          <h3 id="custom-agent-app-access-title">App access</h3>
          <p>Pin installed-app operations to the exact <strong>{agent.name}</strong> version. Stopped or incompatible apps remain visible with recovery guidance.</p>
        </div>
        {apps.length > 0 && <label className="custom-agent-app-access__selector"><span>Installed capability-enabled app</span><select aria-label="Installed capability-enabled app" className="ui-control custom-agent-app-access__select" onChange={(event) => { setSelectedAppId(event.target.value); setMessage(null); setActionError(null); }} value={selectedAppId}>{apps.map((app) => <option key={app.id} value={app.id}>{app.name} · {app.status} · {app.operationCount} operation{app.operationCount === 1 ? "" : "s"}</option>)}</select></label>}
      </div>
      {(message || actionError) && <div className={`custom-agent-app-access__feedback ${actionError ? "custom-agent-app-access__feedback--error" : ""}`} role={actionError ? "alert" : "status"}>{actionError ?? message}</div>}
      {selectedApp && <p className="custom-agent-app-access__selected">Reviewing {selectedApp.name}. No credential reference, endpoint, or secret is displayed.</p>}
      <IntegrationCapabilities
        data={data}
        error={error}
        loading={loading}
        onCreateConnection={createConnectionNotice}
        onPreviewGrant={(selection) => void previewGrant(selection)}
        onRetry={() => { void loadApps().then(() => { if (selectedAppId) void loadDetail(selectedAppId); }); }}
        onRevokeGrant={(grantId) => void revokeGrant(grantId)}
        onSaveGrant={(selection) => void saveGrant(selection)}
        onTestConnection={(connectionId) => void testConnection(connectionId)}
        preview={preview}
        previewing={previewing}
        revoking={revoking}
        saving={saving}
        successMessage={message}
      />
      {testingConnectionId && <span className="custom-agent-app-access__sr-status" role="status">Testing connection…</span>}
    </section>
  );
}
