import { Button } from "./ui";
import { useEffect, useMemo, useState } from "react";
import { Icon } from "./Icon";
import type {
  CompatibleAgentVersion,
  IntegrationAppStatus,
  IntegrationCapabilitiesData,
  IntegrationCapabilitiesProps,
  IntegrationConnection,
  IntegrationConnectionStatus,
  IntegrationGrantSelection,
  IntegrationOperation,
} from "./integrationCapabilitiesTypes";
import "../styles/integration-capabilities.css";

const appStatusCopy: Record<IntegrationAppStatus, { label: string; tone: string; detail: string }> = {
  active: { label: "Active", tone: "positive", detail: "The installed app is available for reviewed operations." },
  degraded: { label: "Degraded", tone: "warning", detail: "Health checks are reporting a partial failure. Grants stay blocked until recovery." },
  stopped: { label: "Stopped", tone: "neutral", detail: "The installed app is not running. Start it and retry the health check." },
  incompatible: { label: "Incompatible", tone: "danger", detail: "The installed manifest no longer matches this app registration." },
  removed: { label: "Removed", tone: "danger", detail: "This app registration was removed. Existing access cannot be reused." },
};

const connectionStatusCopy: Record<IntegrationConnectionStatus, { label: string; tone: string }> = {
  pending: { label: "Testing", tone: "warning" },
  healthy: { label: "Healthy", tone: "positive" },
  degraded: { label: "Degraded", tone: "warning" },
  expired: { label: "Expired", tone: "danger" },
  revoked: { label: "Revoked", tone: "danger" },
};

const kindCopy = {
  read: "Read",
  write: "Write",
} as const;

const riskCopy = {
  low: "Low risk",
  medium: "Medium risk",
  high: "High risk",
} as const;

function defaultConnectionId(data: IntegrationCapabilitiesData | null) {
  return data?.connections.find((connection) => connection.status === "healthy")?.id ?? data?.connections[0]?.id ?? "";
}

function defaultAgentVersionId(data: IntegrationCapabilitiesData | null) {
  return data?.agentVersions[0]?.versionId ?? "";
}

function getSelectionKey(selection: IntegrationGrantSelection) {
  return [
    selection.appInstanceId,
    selection.agentId,
    selection.agentVersionId,
    selection.connectionId ?? "",
    selection.manifestVersion,
    [...selection.operationIds].sort().join(","),
  ].join("|");
}

function StatusBadge({ label, tone }: { label: string; tone: string }) {
  return (
    <span className={`integration-capabilities__status integration-capabilities__status--${tone}`}>
      <span aria-hidden="true" className="integration-capabilities__status-dot" />
      {label}
    </span>
  );
}

function ReadinessNotice({ data, selectedConnection }: { data: IntegrationCapabilitiesData; selectedConnection?: IntegrationConnection }) {
  const appStatus = appStatusCopy[data.status];
  const appReady = data.status === "active";
  const connectionReady = !data.connectionRequired || selectedConnection?.status === "healthy";

  if (appReady && connectionReady) {
    return (
      <div className="integration-capabilities__notice integration-capabilities__notice--positive" role="status">
        <Icon name="shield" size={18} />
        <div>
          <strong>Ready for a reviewed grant</strong>
          <span>Reads can run through the broker. Writes will require an exact preview before approval.</span>
        </div>
      </div>
    );
  }

  const reasons = [
    !appReady ? appStatus.detail : null,
    data.connectionRequired && !connectionReady
      ? selectedConnection?.status === "expired"
        ? "The selected connection has expired. Choose a healthy connection or create a replacement."
        : selectedConnection?.status === "revoked"
          ? "The selected connection was revoked. Choose a healthy connection or create a replacement."
          : "A healthy connection is required before access can be granted."
      : null,
  ].filter(Boolean);

  return (
    <div className="integration-capabilities__notice integration-capabilities__notice--warning" role="status">
      <Icon name="alert" size={18} />
      <div>
        <strong>Access is blocked until this is resolved</strong>
        {reasons.map((reason) => <span key={reason}>{reason}</span>)}
      </div>
    </div>
  );
}

function AppHealth({ data }: { data: IntegrationCapabilitiesData }) {
  const status = appStatusCopy[data.status];
  return (
    <section className="integration-capabilities__card integration-capabilities__health" aria-labelledby="integration-health-title">
      <div className="integration-capabilities__section-heading">
        <div>
          <span className="integration-capabilities__eyebrow">Installed app</span>
          <h2 id="integration-health-title">{data.appName}</h2>
          <p>{data.healthSummary}</p>
        </div>
        <StatusBadge label={status.label} tone={status.tone} />
      </div>
      <dl className="integration-capabilities__facts">
        <div><dt>App version</dt><dd>{data.appVersion}</dd></div>
        <div><dt>Manifest</dt><dd>{data.manifestVersion}</dd></div>
        <div><dt>Access model</dt><dd>{data.connectionRequired ? "Brokered connection required" : "No connection required"}</dd></div>
      </dl>
      {data.compatibilitySummary && (
        <details className="integration-capabilities__details">
          <summary>How compatibility is checked</summary>
          <p>{data.compatibilitySummary}</p>
        </details>
      )}
      {data.recoveryActions && data.recoveryActions.length > 0 && data.status !== "active" && (
        <div className="integration-capabilities__recovery">
          <strong>Recovery path</strong>
          <ul>{data.recoveryActions.map((action) => <li key={action}>{action}</li>)}</ul>
        </div>
      )}
    </section>
  );
}

function ConnectionCard({
  connection,
  selected,
  disabled,
  onSelect,
  onTest,
}: {
  connection: IntegrationConnection;
  selected: boolean;
  disabled: boolean;
  onSelect: () => void;
  onTest: () => void;
}) {
  const status = connectionStatusCopy[connection.status];
  const unusable = connection.status !== "healthy";
  return (
    <div className={`integration-capabilities__connection ${selected ? "integration-capabilities__connection--selected" : ""} ${unusable ? "integration-capabilities__connection--unusable" : ""}`}>
      <label className="integration-capabilities__connection-select">
        <input
          className="integration-capabilities__connection-input" checked={selected}
          disabled={disabled}
          name="integration-connection"
          onChange={onSelect}
          type="radio"
          value={connection.id}
        />
        <span className="integration-capabilities__connection-body">
          <span className="integration-capabilities__connection-heading">
            <strong>{connection.label}</strong>
            <StatusBadge label={status.label} tone={status.tone} />
          </span>
          {connection.issue && <span className="integration-capabilities__connection-issue">{connection.issue}</span>}
          <span className="integration-capabilities__scope-list">
            {connection.scopes.map((scope) => <span key={scope}>{scope}</span>)}
          </span>
          {connection.expiresAt && <small>Expires {connection.expiresAt}</small>}
        </span>
      </label>
      <Button
        className="integration-capabilities__text-button"
        disabled={connection.status === "revoked"}
        onClick={onTest}
        type="button"
      >
        Test connection
      </Button>
    </div>
  );
}
function ConnectionStep({
  data,
  selectedConnectionId,
  disabled,
  onSelect,
  onTest,
  onCreate,
}: {
  data: IntegrationCapabilitiesData;
  selectedConnectionId: string;
  disabled: boolean;
  onSelect: (connectionId: string) => void;
  onTest: (connectionId: string) => void;
  onCreate?: () => void;
}) {
  if (!data.connectionRequired) {
    return (
      <section className="integration-capabilities__card integration-capabilities__step" aria-labelledby="integration-connection-title">
        <div className="integration-capabilities__step-number">01</div>
        <div className="integration-capabilities__step-content">
          <div className="integration-capabilities__section-heading">
            <div><span className="integration-capabilities__eyebrow">Connection</span><h2 id="integration-connection-title">No connection required</h2><p>This app exposes the selected operations without a credential-backed connection.</p></div>
            <StatusBadge label="Not needed" tone="neutral" />
          </div>
        </div>
      </section>
    );
  }

  return (
    <section className="integration-capabilities__card integration-capabilities__step" aria-labelledby="integration-connection-title">
      <div className="integration-capabilities__step-number">01</div>
      <div className="integration-capabilities__step-content">
        <div className="integration-capabilities__section-heading">
          <div><span className="integration-capabilities__eyebrow">Step 1 · Connection</span><h2 id="integration-connection-title">Choose a tested connection</h2><p>Vaelor stores only the broker reference. Credentials and endpoints stay outside this view.</p></div>
          {onCreate && <Button disabled={disabled} onClick={onCreate} type="button" variant="quiet">Create connection</Button>}
        </div>
        {data.connections.length === 0 ? (
          <div className="integration-capabilities__empty integration-capabilities__empty--inline">
            <Icon name="lock" size={20} />
            <div><strong>No connections available</strong><span>Create and test a connection before granting access.</span></div>
            {onCreate && <Button disabled={disabled} onClick={onCreate} type="button" variant="quiet">Create connection</Button>}
          </div>
        ) : (
          <div className="integration-capabilities__connection-list">
            {data.connections.map((connection) => (
              <ConnectionCard
                connection={connection}
                disabled={disabled}
                key={connection.id}
                onSelect={() => onSelect(connection.id)}
                onTest={() => onTest(connection.id)}
                selected={selectedConnectionId === connection.id}
              />
            ))}
          </div>
        )}
      </div>
    </section>
  );
}

function AgentStep({
  agents,
  selectedAgentVersionId,
  disabled,
  onSelect,
}: {
  agents: CompatibleAgentVersion[];
  selectedAgentVersionId: string;
  disabled: boolean;
  onSelect: (versionId: string) => void;
}) {
  const selected = agents.find((agent) => agent.versionId === selectedAgentVersionId);
  return (
    <section className="integration-capabilities__card integration-capabilities__step" aria-labelledby="integration-agent-title">
      <div className="integration-capabilities__step-number">02</div>
      <div className="integration-capabilities__step-content">
        <div className="integration-capabilities__section-heading">
          <div><span className="integration-capabilities__eyebrow">Step 2 · Agent pin</span><h2 id="integration-agent-title">Choose the exact custom-agent version</h2><p>Access is pinned to this version. Vaelor will not silently move a grant to the latest definition.</p></div>
          {selected && <StatusBadge label={selected.status === "archived" ? "Archived" : "Version pinned"} tone={selected.status === "archived" ? "warning" : "positive"} />}
        </div>
        {agents.length === 0 ? (
          <div className="integration-capabilities__empty integration-capabilities__empty--inline">
            <Icon name="alert" size={20} />
            <div><strong>No compatible custom-agent version</strong><span>Create or update a custom agent with an app-compatible version before continuing.</span></div>
          </div>
        ) : (
          <label className="integration-capabilities__field">
            <span>Select exact version</span>
            <select className="ui-control integration-capabilities__agent-select" disabled={disabled} onChange={(event) => onSelect(event.target.value)} value={selectedAgentVersionId}>
              {agents.map((agent) => <option key={agent.versionId} value={agent.versionId}>{agent.agentName} · {agent.versionLabel}</option>)}
            </select>
          </label>
        )}
        {selected && (
          <div className="integration-capabilities__agent-summary">
            <span><strong>{selected.agentName}</strong> will receive access through <strong>{selected.versionLabel}</strong>.</span>
            <small>{selected.compatibleOperationIds.length} compatible operation{selected.compatibleOperationIds.length === 1 ? "" : "s"} detected from this version.</small>
          </div>
        )}
      </div>
    </section>
  );
}

function OperationCard({
  operation,
  selected,
  disabled,
  compatible,
  onToggle,
}: {
  operation: IntegrationOperation;
  selected: boolean;
  disabled: boolean;
  compatible: boolean;
  onToggle: () => void;
}) {
  const unavailable = !compatible || (operation.availability && operation.availability !== "available");
  return (
    <label className={`integration-capabilities__operation integration-capabilities__operation--${operation.kind} ${unavailable ? "integration-capabilities__operation--unavailable" : ""}`}>
      <input checked={selected} className="integration-capabilities__operation-input" disabled={disabled || Boolean(unavailable)} onChange={onToggle} type="checkbox" />
      <span className="integration-capabilities__operation-body">
        <span className="integration-capabilities__operation-heading">
          <strong>{operation.label}</strong>
          <span className="integration-capabilities__operation-tags">
            <span className="integration-capabilities__tag">{kindCopy[operation.kind]}</span>
            <span className={`integration-capabilities__tag integration-capabilities__tag--${operation.risk}`}>{riskCopy[operation.risk]}</span>
          </span>
        </span>
        <span>{operation.description}</span>
        {unavailable && <small>{!compatible ? "Not compatible with the selected agent version." : operation.unavailableReason ?? "This operation is not compatible with the selected app or agent version."}</small>}
      </span>
    </label>
  );
}

function OperationStep({
  data,
  selectedAgent,
  selectedOperationIds,
  disabled,
  onToggle,
}: {
  data: IntegrationCapabilitiesData;
  selectedAgent?: CompatibleAgentVersion;
  selectedOperationIds: string[];
  disabled: boolean;
  onToggle: (operationId: string) => void;
}) {
  const compatibleIds = new Set(selectedAgent?.compatibleOperationIds ?? []);
  const compatibleOperations = data.operations.filter((operation) => compatibleIds.has(operation.id));
  return (
    <section className="integration-capabilities__card integration-capabilities__step" aria-labelledby="integration-operation-title">
      <div className="integration-capabilities__step-number">03</div>
      <div className="integration-capabilities__step-content">
        <div className="integration-capabilities__section-heading">
          <div><span className="integration-capabilities__eyebrow">Step 3 · Grant scope</span><h2 id="integration-operation-title">Select compatible operations</h2><p>Only operations declared by the pinned agent version can be selected. Review risk before continuing.</p></div>
          <span className="integration-capabilities__selection-count">{selectedOperationIds.length} selected</span>
        </div>
        {data.operations.length === 0 ? (
          <div className="integration-capabilities__empty integration-capabilities__empty--inline"><Icon name="database" size={20} /><div><strong>No operations published</strong><span>This app manifest does not currently expose any grantable operations.</span></div></div>
        ) : compatibleOperations.length === 0 ? (
          <div className="integration-capabilities__empty integration-capabilities__empty--inline"><Icon name="alert" size={20} /><div><strong>No compatible operations for this version</strong><span>Choose another exact agent version or update the agent definition.</span></div></div>
        ) : (
          <div className="integration-capabilities__operation-list">
            {data.operations.map((operation) => (
              <OperationCard
                compatible={compatibleIds.has(operation.id)}
                disabled={disabled || !selectedAgent}
                key={operation.id}
                onToggle={() => onToggle(operation.id)}
                operation={operation}
                selected={selectedOperationIds.includes(operation.id)}
              />
            ))}
          </div>
        )}
      </div>
    </section>
  );
}

function GrantPreview({
  preview,
  onSave,
  saving,
}: {
  preview: NonNullable<IntegrationCapabilitiesProps["preview"]>;
  onSave: () => void;
  saving: boolean;
}) {
  return (
    <section className="integration-capabilities__preview" aria-labelledby="integration-preview-title">
      <div className="integration-capabilities__section-heading">
        <div><span className="integration-capabilities__eyebrow">Server-owned preview</span><h2 id="integration-preview-title">Review before saving</h2><p>{preview.summary}</p></div>
        <StatusBadge label="Preview ready" tone="positive" />
      </div>
      <ul className="integration-capabilities__preview-list">
        {preview.items.map((item) => <li key={`${item.kind}-${item.label}`}><span><strong>{item.label}</strong><small>{item.summary}</small></span><span className={`integration-capabilities__tag integration-capabilities__tag--${item.risk}`}>{kindCopy[item.kind]} · {riskCopy[item.risk]}</span></li>)}
      </ul>
      {preview.warnings.length > 0 && <div className="integration-capabilities__preview-warning" role="note"><Icon name="alert" size={18} /><div>{preview.warnings.map((warning) => <span key={warning}>{warning}</span>)}</div></div>}
      <div className="integration-capabilities__preview-footer"><small>Preview expires {preview.expiresAt}</small><Button disabled={saving} onClick={onSave} type="button" variant="primary">{saving ? "Saving grant…" : "Save grant"}</Button></div>
    </section>
  );
}

function DependentImpact({ data, showRevoke, onToggleRevoke, onRevoke, revoking }: { data: IntegrationCapabilitiesData; showRevoke: boolean; onToggleRevoke: () => void; onRevoke: () => void; revoking: boolean }) {
  const grant = data.existingGrant;
  if (!grant) return null;
  return (
    <section className="integration-capabilities__card integration-capabilities__dependents" aria-labelledby="integration-dependents-title">
      <div className="integration-capabilities__section-heading">
        <div><span className="integration-capabilities__eyebrow">Existing access</span><h2 id="integration-dependents-title">Dependent impact and recovery</h2><p>{grant.agentName} · {grant.agentVersionLabel} has a {grant.status} grant for {grant.operationIds.length} operation{grant.operationIds.length === 1 ? "" : "s"}.</p></div>
        <StatusBadge label={grant.status} tone={grant.status === "active" ? "positive" : grant.status === "revoked" ? "danger" : "warning"} />
      </div>
      {grant.blockedReason && <div className="integration-capabilities__notice integration-capabilities__notice--warning"><Icon name="alert" size={18} /><div><strong>Grant is blocked</strong><span>{grant.blockedReason}</span></div></div>}
      <details className="integration-capabilities__details" open={showRevoke} onToggle={(event) => { if (event.currentTarget.open !== showRevoke) onToggleRevoke(); }}>
        <summary>Review revoke impact</summary>
        {data.dependents.length === 0 ? (
          <p>No dependent agents, tasks, or automations are currently recorded.</p>
        ) : (
          <div className="integration-capabilities__dependent-list">
            {data.dependents.map((dependent) => <div className="integration-capabilities__dependent" key={dependent.id}><span><strong>{dependent.label}</strong><small>{dependent.impact}</small></span><span className={`integration-capabilities__tag integration-capabilities__tag--${dependent.status === "active" ? "low" : "high"}`}>{dependent.status}</span><small>Recovery: {dependent.recoveryAction}</small></div>)}
          </div>
        )}
        <div className="integration-capabilities__revoke-confirmation">
          <p>Revoking removes this grant immediately. Dependents will fail closed and must be reconnected or re-granted explicitly.</p>
          <Button disabled={revoking || grant.status === "revoked"} onClick={onRevoke} type="button" variant="danger">{revoking ? "Revoking…" : "Revoke grant"}</Button>
        </div>
      </details>
    </section>
  );
}

function LoadingState() {
  return <div className="integration-capabilities__state" role="status" aria-live="polite"><span className="integration-capabilities__spinner" aria-hidden="true" /><strong>Loading integration capabilities…</strong><span>Checking app health, manifest compatibility, and available grants.</span></div>;
}

function ErrorState({ error, onRetry }: { error: string; onRetry: () => void }) {
  return <div className="integration-capabilities__state integration-capabilities__state--error" role="alert"><Icon name="alert" size={24} /><strong>Capabilities could not be loaded</strong><span>{error}</span><Button onClick={onRetry} type="button" variant="quiet">Retry</Button></div>;
}

function EmptyState() {
  return <div className="integration-capabilities__state"><Icon name="grid" size={24} /><strong>No installed integration selected</strong><span>Choose an installed app to inspect its health, operations, connections, and dependent access.</span></div>;
}

export function IntegrationCapabilities({
  data,
  loading = false,
  error = null,
  preview = null,
  previewing = false,
  saving = false,
  revoking = false,
  successMessage = null,
  onRetry,
  onTestConnection,
  onCreateConnection,
  onPreviewGrant,
  onSaveGrant,
  onRevokeGrant,
}: IntegrationCapabilitiesProps) {
  const [selectedConnectionId, setSelectedConnectionId] = useState(defaultConnectionId(data));
  const [selectedAgentVersionId, setSelectedAgentVersionId] = useState(defaultAgentVersionId(data));
  const [selectedOperationIds, setSelectedOperationIds] = useState<string[]>([]);
  const [previewSelectionKey, setPreviewSelectionKey] = useState<string | null>(null);
  const [validationMessage, setValidationMessage] = useState<string | null>(null);
  const [showRevokeImpact, setShowRevokeImpact] = useState(false);

  useEffect(() => {
    setSelectedConnectionId(defaultConnectionId(data));
    setSelectedAgentVersionId(defaultAgentVersionId(data));
    setSelectedOperationIds([]);
    setPreviewSelectionKey(null);
    setValidationMessage(null);
    setShowRevokeImpact(false);
  }, [data?.appInstanceId]);

  const selectedAgent = data?.agentVersions.find((agent) => agent.versionId === selectedAgentVersionId);
  const selectedConnection = data?.connections.find((connection) => connection.id === selectedConnectionId);
  const compatibleIds = useMemo(() => new Set(selectedAgent?.compatibleOperationIds ?? []), [selectedAgent]);
  const hasWrite = data?.operations.some((operation) => selectedOperationIds.includes(operation.id) && operation.kind === "write") ?? false;
  const selection: IntegrationGrantSelection | null = data && selectedAgent && selectedOperationIds.length > 0
    ? {
      agentId: selectedAgent.agentId,
      agentVersionId: selectedAgent.versionId,
      appInstanceId: data.appInstanceId,
      connectionId: data.connectionRequired ? selectedConnectionId || undefined : undefined,
      manifestVersion: data.manifestVersion,
      operationIds: selectedOperationIds,
    }
    : null;
  const selectionKey = selection ? getSelectionKey(selection) : null;
  const currentPreview = preview && selectionKey && previewSelectionKey === selectionKey ? preview : null;
  const appBlocked = data?.status !== "active";
  const connectionBlocked = Boolean(data?.connectionRequired && selectedConnection?.status !== "healthy");
  const formDisabled = appBlocked || connectionBlocked || !data;

  if (loading) {
    return <section className="integration-capabilities" aria-labelledby="integration-capabilities-title"><div className="integration-capabilities__intro"><span className="integration-capabilities__eyebrow">Operator surface</span><h1 id="integration-capabilities-title">Integration capabilities</h1></div><LoadingState /></section>;
  }
  if (error) {
    return <section className="integration-capabilities" aria-labelledby="integration-capabilities-title"><div className="integration-capabilities__intro"><span className="integration-capabilities__eyebrow">Operator surface</span><h1 id="integration-capabilities-title">Integration capabilities</h1></div><ErrorState error={error} onRetry={onRetry} /></section>;
  }
  if (!data) {
    return <section className="integration-capabilities" aria-labelledby="integration-capabilities-title"><div className="integration-capabilities__intro"><span className="integration-capabilities__eyebrow">Operator surface</span><h1 id="integration-capabilities-title">Integration capabilities</h1></div><EmptyState /></section>;
  }

  const toggleOperation = (operationId: string) => {
    if (!compatibleIds.has(operationId)) return;
    setSelectedOperationIds((current) => current.includes(operationId) ? current.filter((id) => id !== operationId) : [...current, operationId]);
    setPreviewSelectionKey(null);
    setValidationMessage(null);
  };

  const validateSelection = () => {
    if (!selectedAgent) return "Choose an exact compatible custom-agent version.";
    if (data.connectionRequired && selectedConnection?.status !== "healthy") return "Choose a healthy connection before creating a grant.";
    if (selectedOperationIds.length === 0) return "Select at least one compatible operation.";
    if (selectedOperationIds.some((operationId) => !compatibleIds.has(operationId))) return "One or more selected operations are not compatible with this agent version.";
    if (data.status !== "active") return "The installed app must be active before access can be granted.";
    return null;
  };

  const previewGrant = () => {
    const problem = validateSelection();
    setValidationMessage(problem);
    if (problem || !selection || !selectionKey) return;
    setPreviewSelectionKey(selectionKey);
    onPreviewGrant(selection);
  };

  const saveGrant = () => {
    const problem = validateSelection();
    setValidationMessage(problem);
    if (problem || !selection || !currentPreview) return;
    onSaveGrant({ ...selection, previewId: currentPreview.previewId });
  };

  return (
    <section className="integration-capabilities" aria-labelledby="integration-capabilities-title">
      <header className="integration-capabilities__intro">
        <div>
          <span className="integration-capabilities__eyebrow">Operator surface · Access control</span>
          <h1 id="integration-capabilities-title">Integration capabilities</h1>
          <p>Inspect what an installed app can do, pin access to an exact agent version, and review the impact before anything is saved.</p>
        </div>
        <div className="integration-capabilities__privacy-note"><Icon name="lock" size={18} /><span>Credential-safe view<br /><small>No endpoints, refs, or secrets are shown.</small></span></div>
      </header>

      {successMessage && <div className="integration-capabilities__notice integration-capabilities__notice--positive" role="status" aria-live="polite"><Icon name="shield" size={18} /><div><strong>Success</strong><span>{successMessage}</span></div></div>}
      <AppHealth data={data} />
      <ReadinessNotice data={data} selectedConnection={selectedConnection} />

      <div className="integration-capabilities__workflow">
        <ConnectionStep data={data} disabled={formDisabled} onCreate={onCreateConnection} onSelect={(id) => { setSelectedConnectionId(id); setPreviewSelectionKey(null); setValidationMessage(null); }} onTest={onTestConnection} selectedConnectionId={selectedConnectionId} />
        <AgentStep agents={data.agentVersions} disabled={formDisabled} onSelect={(id) => { setSelectedAgentVersionId(id); setSelectedOperationIds([]); setPreviewSelectionKey(null); setValidationMessage(null); }} selectedAgentVersionId={selectedAgentVersionId} />
        <OperationStep data={data} disabled={formDisabled} onToggle={toggleOperation} selectedAgent={selectedAgent} selectedOperationIds={selectedOperationIds} />
      </div>

      <section className="integration-capabilities__card integration-capabilities__review" aria-labelledby="integration-review-title">
        <div className="integration-capabilities__section-heading">
          <div><span className="integration-capabilities__eyebrow">Step 4 · Review</span><h2 id="integration-review-title">Preview and save the grant</h2><p>Vaelor validates the pinned identities and selected operation IDs before saving a new grant.</p></div>
          {hasWrite && <StatusBadge label="Includes write access" tone="warning" />}
        </div>
        {validationMessage && <div className="integration-capabilities__validation" role="alert"><Icon name="alert" size={18} />{validationMessage}</div>}
        <div className="integration-capabilities__review-summary">
          <span><small>App</small><strong>{data.appName} · {data.appVersion}</strong></span>
          <span><small>Agent version</small><strong>{selectedAgent ? `${selectedAgent.agentName} · ${selectedAgent.versionLabel}` : "Not selected"}</strong></span>
          <span><small>Operations</small><strong>{selectedOperationIds.length ? `${selectedOperationIds.length} selected` : "None selected"}</strong></span>
        </div>
        {currentPreview ? <GrantPreview onSave={saveGrant} preview={currentPreview} saving={saving} /> : (
          <div className="integration-capabilities__review-actions"><span>{hasWrite ? "Write operations stop at an exact preview and need approval." : "Read operations will use the selected broker connection."}</span><Button disabled={formDisabled || previewing} onClick={previewGrant} type="button" variant="primary">{previewing ? "Preparing preview…" : "Preview grant"}</Button></div>
        )}
      </section>

      <DependentImpact data={data} onRevoke={() => { if (data.existingGrant) onRevokeGrant(data.existingGrant.id); }} onToggleRevoke={() => setShowRevokeImpact((current) => !current)} revoking={revoking} showRevoke={showRevokeImpact} />
    </section>
  );
}
