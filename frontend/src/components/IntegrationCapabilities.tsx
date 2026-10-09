import { type ReactNode, useEffect, useId, useMemo, useState } from "react";
import { ConfirmDialog } from "./ConfirmDialog";
import { Icon, ICON_SIZE } from "./Icon";
import { StatusPill } from "./StatusPill";
import { Button, EmptyState, Notice, type StatusTone } from "./ui";
import type {
  CompatibleAgentVersion,
  IntegrationAppStatus,
  IntegrationCapabilitiesData,
  IntegrationCapabilitiesProps,
  IntegrationConnection,
  IntegrationConnectionStatus,
  IntegrationDependent,
  IntegrationGrantSelection,
  IntegrationGrantStatus,
  IntegrationOperation,
} from "./integrationCapabilitiesTypes";

/*
 * App access for one custom agent (the "Capability control" card on the
 * AssistRoutines board): what an installed app can do, pinned to an exact
 * agent version, reviewed before anything is saved. Four numbered steps -
 * connection, agent pin, grant scope, review - then the access that already
 * exists and what revoking it would break. Nothing here shows an endpoint, a
 * credential reference or a secret.
 */

const appStatusCopy: Record<IntegrationAppStatus, { label: string; tone: StatusTone; detail: string }> = {
  active: { label: "Active", tone: "success", detail: "The installed app is available for reviewed operations." },
  degraded: { label: "Degraded", tone: "warning", detail: "Health checks are reporting a partial failure. Grants stay blocked until recovery." },
  stopped: { label: "Stopped", tone: "neutral", detail: "The installed app is not running. Start it and retry the health check." },
  incompatible: { label: "Incompatible", tone: "danger", detail: "The installed manifest no longer matches this app registration." },
  removed: { label: "Removed", tone: "danger", detail: "This app registration was removed. Existing access cannot be reused." },
};

const connectionStatusCopy: Record<IntegrationConnectionStatus, { label: string; tone: StatusTone }> = {
  pending: { label: "Testing", tone: "warning" },
  healthy: { label: "Healthy", tone: "success" },
  degraded: { label: "Degraded", tone: "warning" },
  expired: { label: "Expired", tone: "danger" },
  revoked: { label: "Revoked", tone: "danger" },
};

/** A grant's state in the operator's words: the pill never prints the wire's slug. */
const grantStatusCopy: Record<IntegrationGrantStatus, { label: string; tone: StatusTone }> = {
  active: { label: "Active", tone: "success" },
  blocked: { label: "Blocked", tone: "warning" },
  incompatible: { label: "Incompatible", tone: "warning" },
  revoked: { label: "Revoked", tone: "danger" },
};

/** What a dependent is doing now, in words. */
const dependentStatusCopy: Record<IntegrationDependent["status"], { label: string; tone: StatusTone }> = {
  active: { label: "Active", tone: "neutral" },
  blocked: { label: "Blocked", tone: "warning" },
  stopped: { label: "Stopped", tone: "warning" },
};

/** A state this page has no words for is said as unrecognised, never printed as its slug. */
const UNRECOGNISED = { label: "Unrecognised state", tone: "neutral" } as const satisfies { label: string; tone: StatusTone };
const grantStatus = (status: IntegrationGrantStatus) => grantStatusCopy[status] ?? UNRECOGNISED;
const dependentStatus = (status: IntegrationDependent["status"]) => dependentStatusCopy[status] ?? UNRECOGNISED;

const kindCopy = {
  read: "Read",
  write: "Write",
} as const;

const riskCopy = {
  low: "Low risk",
  medium: "Medium risk",
  high: "High risk",
} as const;

const riskTone: Record<keyof typeof riskCopy, StatusTone> = { low: "neutral", medium: "warning", high: "danger" };

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

/** A numbered step: the circle is green once the step is satisfied and orange while it is the one to do. */
function Step({ children, label, number, state, text, title }: {
  children: ReactNode;
  label: string;
  number: number;
  state?: "done" | "on";
  text: string;
  title: string;
}) {
  return (
    <section aria-label={title} className="ar-step">
      <span className={state ? `ar-step__num is-${state}` : "ar-step__num"}>{number}</span>
      <div className="ar-step__content">
        <div><span className="as-label">{label}</span><h3 className="ar-step__title">{title}</h3><span className="ar-step__hint">{text}</span></div>
        {children}
      </div>
    </section>
  );
}

function ReadinessNotice({ data, selectedConnection }: { data: IntegrationCapabilitiesData; selectedConnection?: IntegrationConnection }) {
  const appStatus = appStatusCopy[data.status];
  const appReady = data.status === "active";
  const connectionReady = !data.connectionRequired || selectedConnection?.status === "healthy";

  if (appReady && connectionReady) {
    return (
      <Notice heading="Ready for a reviewed grant" severity="success" standing>
        Reads can run through the broker. Writes will require an exact preview before approval.
      </Notice>
    );
  }

  const reasons = [
    !appReady ? `${appStatus.label}: ${appStatus.detail.charAt(0).toLowerCase()}${appStatus.detail.slice(1)}` : null,
    data.connectionRequired && !connectionReady
      ? selectedConnection?.status === "expired"
        ? "The selected connection has expired. Choose a healthy connection or create a replacement."
        : selectedConnection?.status === "revoked"
          ? "The selected connection was revoked. Choose a healthy connection or create a replacement."
          : "A healthy connection is required before access can be granted."
      : null,
  ].filter((reason): reason is string => Boolean(reason));

  return (
    <Notice heading="Access is blocked until this is resolved" severity="warning" standing>
      {reasons.map((reason) => <span className="ar-block-line" key={reason}>{reason}</span>)}
    </Notice>
  );
}

function AppFacts({ data }: { data: IntegrationCapabilitiesData }) {
  const status = appStatusCopy[data.status];
  return (
    <div className="ar-app-facts">
      <dl className="as-kv ar-kv2">
        <div><dt>Installed app</dt><dd>{data.appName}</dd></div>
        <div><dt>App version</dt><dd className="as-mono">{data.appVersion}</dd></div>
        <div><dt>Manifest</dt><dd className="as-mono">{data.manifestVersion}</dd></div>
        <div><dt>Access model</dt><dd>{data.connectionRequired ? "Brokered connection required" : "No connection required"}</dd></div>
      </dl>
      <span className="ar-inline"><StatusPill label={status.label} tone={status.tone} /><span className="ar-step__hint">{data.healthSummary}</span></span>
      {data.compatibilitySummary && (
        <details className="ar-output">
          <summary>How compatibility is checked</summary>
          <p>{data.compatibilitySummary}</p>
        </details>
      )}
      {data.recoveryActions && data.recoveryActions.length > 0 && data.status !== "active" && (
        <div className="as-box">
          <strong className="ar-step__title">Recovery path</strong>
          <ul className="ar-list">{data.recoveryActions.map((action) => <li key={action}>{action}</li>)}</ul>
        </div>
      )}
    </div>
  );
}

function ConnectionRow({ connection, disabled, onSelect, onTest, selected }: {
  connection: IntegrationConnection;
  disabled: boolean;
  onSelect: () => void;
  onTest: () => void;
  selected: boolean;
}) {
  const id = useId().replaceAll(":", "");
  const status = connectionStatusCopy[connection.status];
  return (
    <div className={connection.status === "healthy" ? "ar-check ar-check--row" : "ar-check ar-check--row is-unusable"}>
      <input checked={selected} className="ar-check__input" disabled={disabled} id={`${id}-radio`} name="integration-connection" onChange={onSelect} type="radio" value={connection.id} />
      <span className="ar-check__text">
        <label htmlFor={`${id}-radio`} id={`${id}-label`}>{connection.label}</label>
        {connection.issue && <span className="ar-step__hint">{connection.issue}</span>}
        {connection.scopes.length > 0 && <span className="ar-step__hint as-mono">{connection.scopes.join(" · ")}</span>}
        {connection.expiresAt && <span className="ar-step__hint">Expires {connection.expiresAt}</span>}
      </span>
      <StatusPill label={status.label} tone={status.tone} />
      <Button aria-describedby={`${id}-label`} aria-label="Test connection" disabled={connection.status === "revoked"} onClick={onTest}>Test</Button>
    </div>
  );
}

function OperationRow({ compatible, disabled, onToggle, operation, selected }: {
  compatible: boolean;
  disabled: boolean;
  onToggle: () => void;
  operation: IntegrationOperation;
  selected: boolean;
}) {
  const id = useId().replaceAll(":", "");
  const unavailable = !compatible || (operation.availability && operation.availability !== "available");
  return (
    <div className={unavailable ? "ar-check ar-check--row is-unusable" : "ar-check ar-check--row"}>
      <input aria-describedby={`${id}-detail`} aria-labelledby={`${id}-label`} checked={selected} className="ar-check__input" disabled={disabled || Boolean(unavailable)} id={`${id}-box`} onChange={onToggle} type="checkbox" />
      <span className="ar-check__text">
        <label htmlFor={`${id}-box`} id={`${id}-label`}>{operation.label}</label>
        <span className="ar-step__hint" id={`${id}-detail`}>
          {kindCopy[operation.kind].toLowerCase()} · {operation.description}
          {unavailable && <> · {!compatible ? "Not compatible with the selected agent version." : operation.unavailableReason ?? "This operation is not compatible with the selected app or agent version."}</>}
        </span>
      </span>
      <StatusPill label={riskCopy[operation.risk]} tone={riskTone[operation.risk]} />
    </div>
  );
}

function GrantPreview({ onSave, preview, saving }: {
  onSave: () => void;
  preview: NonNullable<IntegrationCapabilitiesProps["preview"]>;
  saving: boolean;
}) {
  return (
    <section aria-labelledby="integration-preview-title" className="as-box ar-preview">
      <div className="ar-split-head">
        <div><span className="as-label">Server-owned preview</span><h4 className="ar-step__title" id="integration-preview-title">Review before saving</h4><span className="ar-step__hint">{preview.summary}</span></div>
        <StatusPill label="Preview ready" tone="success" />
      </div>
      <ul className="ar-plain-list">
        {preview.items.map((item) => (
          <li key={`${item.kind}-${item.label}`}>
            <span><strong>{item.label}</strong><span className="ar-step__hint">{item.summary}</span></span>
            <StatusPill label={`${kindCopy[item.kind]} · ${riskCopy[item.risk]}`} tone={riskTone[item.risk]} />
          </li>
        ))}
      </ul>
      {preview.warnings.length > 0 && (
        <Notice severity="warning" standing>{preview.warnings.map((warning) => <span className="ar-block-line" key={warning}>{warning}</span>)}</Notice>
      )}
      <div className="ar-actions ar-actions--spread">
        <span className="ar-step__hint">Preview expires {preview.expiresAt}</span>
        <Button disabled={saving} onClick={onSave} variant="primary">{saving ? "Saving grant…" : "Save grant"}</Button>
      </div>
    </section>
  );
}

/**
 * What the revoke confirmation says will break: every recorded dependent by
 * name with its recovery, or that none is recorded. The owner reads this in
 * the dialog itself, so a one-click revoke never hides who fails closed.
 */
function revokeDependentsNote(dependents: IntegrationDependent[]) {
  if (dependents.length === 0) return "No dependent agents, tasks, or automations are currently recorded.";
  const named = dependents.map((dependent) => `${dependent.label} (recovery: ${dependent.recoveryAction})`);
  return `${dependents.length === 1 ? "This dependent fails" : `These ${dependents.length} dependents fail`} closed until reconnected or re-granted: ${named.join("; ")}.`;
}

function ExistingAccess({ data, onRevoke, onToggleRevoke, revoking, showRevoke }: {
  data: IntegrationCapabilitiesData;
  onRevoke: () => void;
  onToggleRevoke: () => void;
  revoking: boolean;
  showRevoke: boolean;
}) {
  const [confirming, setConfirming] = useState(false);
  const grant = data.existingGrant;
  if (!grant) return null;
  const alreadyRevoked = grant.status === "revoked";
  return (
    <section aria-labelledby="integration-dependents-title" className="ar-existing">
      <div><span className="as-label">Existing access</span><h3 className="ar-step__title" id="integration-dependents-title">Dependent impact and recovery</h3></div>
      <div className="as-box">
        <span className="ar-split-head">
          <strong className="ar-step__title">{grant.agentName} · {grant.agentVersionLabel}</strong>
          <StatusPill label={grantStatus(grant.status).label} tone={grantStatus(grant.status).tone} />
        </span>
        <span className="ar-step__hint">A {grantStatus(grant.status).label.toLowerCase()} grant for {grant.operationIds.length} operation{grant.operationIds.length === 1 ? "" : "s"}.</span>
        {grant.blockedReason && <Notice heading="Grant is blocked" severity="warning" standing>{grant.blockedReason}</Notice>}
        {showRevoke && (
          data.dependents.length === 0 ? (
            <span className="ar-step__hint">No dependent agents, tasks, or automations are currently recorded.</span>
          ) : (
            <ul className="ar-plain-list">
              {data.dependents.map((dependent) => (
                <li key={dependent.id}>
                  <span><strong>{dependent.label}</strong><span className="ar-step__hint">{dependent.impact}</span><span className="ar-step__hint">Recovery: {dependent.recoveryAction}</span></span>
                  <StatusPill label={dependentStatus(dependent.status).label} tone={dependentStatus(dependent.status).tone} />
                </li>
              ))}
            </ul>
          )
        )}
        <span className="ar-step__hint">Revoking removes this grant immediately. Dependents will fail closed and must be reconnected or re-granted explicitly.</span>
        <span className="ar-actions">
          <Button aria-expanded={showRevoke} onClick={onToggleRevoke}>Review revoke impact</Button>
          <Button className="as-btn-danger" disabled={revoking} disabledReason={alreadyRevoked ? "Already revoked" : undefined} onClick={() => setConfirming(true)}>{revoking ? "Revoking…" : "Revoke grant"}</Button>
        </span>
      </div>
      <ConfirmDialog
        busy={revoking}
        confirmLabel="Revoke access"
        description={`Revoke ${grant.agentName} · ${grant.agentVersionLabel} access to ${data.appName}? The grant is removed immediately and cannot be restored; access needs a new reviewed grant.`}
        irreversible
        note={revokeDependentsNote(data.dependents)}
        onCancel={() => setConfirming(false)}
        onConfirm={() => { setConfirming(false); onRevoke(); }}
        open={confirming && !alreadyRevoked}
        title="Revoke this access grant?"
      />
    </section>
  );
}

/** The surface frame: named for assistive technology, with the credential-safe promise on top. */
function Surface({ children }: { children: ReactNode }) {
  return (
    <section aria-labelledby="integration-capabilities-title" className="ar-capabilities">
      <h2 className="sr-only" id="integration-capabilities-title">Integration capabilities</h2>
      <span className="ar-inline ar-step__hint"><Icon name="shield" size={ICON_SIZE.inline} />Credential-safe view · No endpoints, refs, or secrets are shown.</span>
      {children}
    </section>
  );
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
    return (
      <Surface>
        <div aria-live="polite" className="as-box" role="status">
          <span className="ar-step__hint">Loading integration capabilities… Checking app health, manifest compatibility, and available grants.</span>
        </div>
      </Surface>
    );
  }
  if (error) {
    return (
      <Surface>
        <Notice severity="danger">
          <span className="ar-notice-row"><span>Capabilities could not be loaded. {error}</span><Button onClick={onRetry}>Retry</Button></span>
        </Notice>
      </Surface>
    );
  }
  if (!data) {
    return (
      <Surface>
        <EmptyState icon={<Icon name="grid" size={18} />} text="Choose an installed app to inspect its health, operations, connections, and dependent access." title="No installed integration selected" />
      </Surface>
    );
  }

  const resetPreview = () => { setPreviewSelectionKey(null); setValidationMessage(null); };
  const toggleOperation = (operationId: string) => {
    if (!compatibleIds.has(operationId)) return;
    setSelectedOperationIds((current) => current.includes(operationId) ? current.filter((id) => id !== operationId) : [...current, operationId]);
    resetPreview();
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

  const compatibleOperations = data.operations.filter((operation) => compatibleIds.has(operation.id));
  const connectionDone = !data.connectionRequired || selectedConnection?.status === "healthy";
  const selectedLabels = data.operations.filter((operation) => selectedOperationIds.includes(operation.id)).map((operation) => operation.label);

  return (
    <Surface>
      {successMessage && <Notice heading="Success" severity="success">{successMessage}</Notice>}
      <ReadinessNotice data={data} selectedConnection={selectedConnection} />
      <AppFacts data={data} />

      <Step label={data.connectionRequired ? "Step 1 · Connection" : "Connection"} number={1} state={connectionDone ? "done" : "on"}
        text={data.connectionRequired ? "Vaelor stores only the broker reference. Credentials and endpoints stay outside this view." : "This app exposes the selected operations without a credential-backed connection."}
        title={data.connectionRequired ? "Choose a tested connection" : "No connection required"}>
        {data.connectionRequired && (
          data.connections.length === 0 ? (
            <div className="as-box ar-empty-row">
              <span aria-hidden="true" className="ar-icon"><Icon name="lock" size={16} /></span>
              <div><strong className="ar-step__title">No connections available</strong><span className="ar-step__hint">Create and test a connection before granting access.</span></div>
            </div>
          ) : data.connections.map((connection) => (
            <ConnectionRow
              connection={connection}
              disabled={appBlocked}
              key={connection.id}
              onSelect={() => { setSelectedConnectionId(connection.id); resetPreview(); }}
              onTest={() => onTestConnection(connection.id)}
              selected={selectedConnectionId === connection.id}
            />
          ))
        )}
        {data.connectionRequired && onCreateConnection && (
          <span><Button className="as-btn-ghost" disabled={appBlocked} onClick={onCreateConnection} variant="quiet"><Icon className="ar-btn-icon" name="add" size={ICON_SIZE.inline} />Create connection</Button></span>
        )}
      </Step>

      <Step label="Step 2 · Agent pin" number={2} state={selectedAgent ? "done" : connectionDone ? "on" : undefined}
        text="Access is pinned to this version. Vaelor will not silently move a grant to the latest definition."
        title="Choose the exact custom-agent version">
        {data.agentVersions.length === 0 ? (
          <div className="as-box ar-empty-row">
            <span aria-hidden="true" className="ar-icon"><Icon name="alert" size={16} /></span>
            <div><strong className="ar-step__title">No compatible custom-agent version</strong><span className="ar-step__hint">Create or update a custom agent with an app-compatible version before continuing.</span></div>
          </div>
        ) : (
          <label className="ar-field">
            <span className="ar-field__label">Select exact version</span>
            <select className="ui-control" disabled={formDisabled} onChange={(event) => { setSelectedAgentVersionId(event.target.value); setSelectedOperationIds([]); resetPreview(); }} value={selectedAgentVersionId}>
              {data.agentVersions.map((agent) => <option key={agent.versionId} value={agent.versionId}>{agent.agentName} · {agent.versionLabel}{agent.status === "archived" ? " · Archived" : " · Version pinned"}</option>)}
            </select>
          </label>
        )}
        {selectedAgent?.status === "archived" && (
          // The archived warning the earlier screen carried beside the version (VD-200 assist review).
          <span><StatusPill label="Archived version" tone="warning" /></span>
        )}
        {selectedAgent && <AgentSummary agent={selectedAgent} />}
      </Step>

      <Step label="Step 3 · Grant scope" number={3} state={selectedOperationIds.length ? "done" : selectedAgent && connectionDone ? "on" : undefined}
        text="Only operations declared by the pinned agent version can be selected. Review risk before continuing."
        title="Select compatible operations">
        {data.operations.length === 0 ? (
          <div className="as-box ar-empty-row"><span aria-hidden="true" className="ar-icon"><Icon name="database" size={16} /></span><div><strong className="ar-step__title">No operations published</strong><span className="ar-step__hint">This app manifest does not currently expose any grantable operations.</span></div></div>
        ) : compatibleOperations.length === 0 ? (
          <div className="as-box ar-empty-row"><span aria-hidden="true" className="ar-icon"><Icon name="alert" size={16} /></span><div><strong className="ar-step__title">No compatible operations for this version</strong><span className="ar-step__hint">Choose another exact agent version or update the agent definition.</span></div></div>
        ) : data.operations.map((operation) => (
          <OperationRow
            compatible={compatibleIds.has(operation.id)}
            disabled={formDisabled || !selectedAgent}
            key={operation.id}
            onToggle={() => toggleOperation(operation.id)}
            operation={operation}
            selected={selectedOperationIds.includes(operation.id)}
          />
        ))}
      </Step>

      <Step label="Step 4 · Review" number={4} state={currentPreview ? "on" : undefined}
        text="Vaelor validates the pinned identities and selected operation IDs before saving a new grant."
        title="Preview and save the grant">
        {hasWrite && <span><StatusPill label="Includes write access" tone="warning" /></span>}
        {validationMessage && <Notice severity="danger">{validationMessage}</Notice>}
        <dl className="as-kv ar-kv1">
          <div><dt>App</dt><dd>{data.appName} · {data.appVersion}</dd></div>
          <div><dt>Agent version</dt><dd>{selectedAgent ? `${selectedAgent.agentName} · ${selectedAgent.versionLabel}` : "Not selected"}</dd></div>
          <div><dt>Operations</dt><dd>{selectedLabels.length ? selectedLabels.join(" · ") : "None selected"}</dd></div>
        </dl>
        {currentPreview ? <GrantPreview onSave={saveGrant} preview={currentPreview} saving={saving} /> : (
          <>
            <span className="ar-step__hint">{hasWrite ? "Write operations stop at an exact preview and need approval." : "Read operations will use the selected broker connection."}</span>
            <span className="ar-actions">
              <Button disabled={formDisabled || previewing} onClick={previewGrant} variant="primary">{previewing ? "Preparing preview…" : "Preview grant"}</Button>
              <Button disabledReason="Preview first">Save grant</Button>
            </span>
          </>
        )}
      </Step>

      <ExistingAccess data={data} onRevoke={() => { if (data.existingGrant) onRevokeGrant(data.existingGrant.id); }} onToggleRevoke={() => setShowRevokeImpact((current) => !current)} revoking={revoking} showRevoke={showRevokeImpact} />
    </Surface>
  );
}

function AgentSummary({ agent }: { agent: CompatibleAgentVersion }) {
  return (
    <span className="ar-step__hint">
      <span><strong>{agent.agentName}</strong> will receive access through <strong>{agent.versionLabel}</strong>.</span>{" "}
      <span>{agent.compatibleOperationIds.length} compatible operation{agent.compatibleOperationIds.length === 1 ? "" : "s"} detected from this version.</span>
    </span>
  );
}
