export type IntegrationAppStatus =
  | "active"
  | "degraded"
  | "stopped"
  | "incompatible"
  | "removed";

export type IntegrationConnectionStatus =
  | "pending"
  | "healthy"
  | "degraded"
  | "expired"
  | "revoked";

export type IntegrationGrantStatus =
  | "active"
  | "blocked"
  | "incompatible"
  | "revoked";

export type IntegrationOperationKind = "read" | "write";
export type IntegrationOperationRisk = "low" | "medium" | "high";
export type IntegrationOperationAvailability = "available" | "incompatible" | "removed";

export interface IntegrationOperation {
  id: string;
  label: string;
  description: string;
  kind: IntegrationOperationKind;
  risk: IntegrationOperationRisk;
  availability?: IntegrationOperationAvailability;
  unavailableReason?: string;
}

export interface IntegrationConnection {
  id: string;
  label: string;
  status: IntegrationConnectionStatus;
  scopes: string[];
  expiresAt?: string;
  lastTestedAt?: string;
  issue?: string;
}

export interface CompatibleAgentVersion {
  agentId: string;
  agentName: string;
  versionId: string;
  versionLabel: string;
  compatibleOperationIds: string[];
  status?: "active" | "draft" | "archived";
}

export interface IntegrationDependent {
  id: string;
  label: string;
  kind: "agent" | "task" | "automation";
  status: "active" | "blocked" | "stopped";
  impact: string;
  recoveryAction: string;
}

export interface IntegrationGrant {
  id: string;
  status: IntegrationGrantStatus;
  agentName: string;
  agentVersionLabel: string;
  operationIds: string[];
  connectionId?: string;
  blockedReason?: string;
}

export interface IntegrationCapabilitiesData {
  appInstanceId: string;
  appName: string;
  appVersion: string;
  manifestVersion: string;
  status: IntegrationAppStatus;
  healthSummary: string;
  compatibilitySummary?: string;
  connectionRequired: boolean;
  operations: IntegrationOperation[];
  connections: IntegrationConnection[];
  agentVersions: CompatibleAgentVersion[];
  existingGrant?: IntegrationGrant;
  dependents: IntegrationDependent[];
  recoveryActions?: string[];
}

export interface IntegrationGrantSelection {
  appInstanceId: string;
  agentId: string;
  agentVersionId: string;
  connectionId?: string;
  operationIds: string[];
  manifestVersion: string;
}

export interface IntegrationGrantPreviewItem {
  label: string;
  kind: IntegrationOperationKind;
  risk: IntegrationOperationRisk;
  summary: string;
}

export interface IntegrationGrantPreview {
  previewId: string;
  expiresAt: string;
  summary: string;
  items: IntegrationGrantPreviewItem[];
  warnings: string[];
}

export interface IntegrationCapabilitiesProps {
  data: IntegrationCapabilitiesData | null;
  loading?: boolean;
  error?: string | null;
  preview?: IntegrationGrantPreview | null;
  previewing?: boolean;
  saving?: boolean;
  revoking?: boolean;
  successMessage?: string | null;
  onRetry: () => void;
  onTestConnection: (connectionId: string) => void;
  onCreateConnection?: () => void;
  onPreviewGrant: (selection: IntegrationGrantSelection) => void;
  onSaveGrant: (selection: IntegrationGrantSelection & { previewId: string }) => void;
  onRevokeGrant: (grantId: string) => void;
}
