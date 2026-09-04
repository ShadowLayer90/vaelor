export const OPERATION_SCHEMA = "vaelor.operation.v1" as const;
export const OPERATION_RESUME_STORAGE_KEY = "vaelor.operation-owner.resume";

export const OPERATION_STATES = [ // vocabulary: operation-state
  "draft",
  "needs_approval",
  "ready",
  "queued",
  "running",
  "waiting",
  "paused",
  "completed",
  "healthy",
  "failed",
  "rejected",
  "cancelled",
  "superseded",
] as const;

export type OperationState = (typeof OPERATION_STATES)[number];
export type OperationRevision = number | string;
export type OperationAction = "cancel" | "retry" | "resume" | "discard" | "done";
export type OperationActionMethod = "GET" | "POST" | "PUT" | "PATCH" | "DELETE";
export type OperationConnection = "idle" | "hydrating" | "connected" | "reconnecting";

export const TERMINAL_OPERATION_STATES: ReadonlySet<OperationState> = new Set([ // vocabulary: terminal-state
  "completed",
  "healthy",
  "failed",
  "rejected",
  "cancelled",
  "superseded",
]);

/**
 * The states in which an operation has not finished and something is expected
 * to move it. This gates the poll loop in `useOperationOwner`.
 *
 * **`vaelor/operation_projection.py` `ACTIVE_OPERATION_STATES` owns these six
 * words and this is a hand copy, because no import crosses Python to
 * TypeScript.** Until #201 they were not related to anything: this list held
 * six, `vaelor/jobs.py` counted the ledger's active rows from five, and
 * `jobPresentation.ts` wrote the same five again. Both sets were *right* —
 * `ready` is unreachable for a job row and reachable for an agent task — which
 * is exactly why they would have drifted, since a fourteenth operation state
 * would have been added to one side only.
 *
 * `tests/test_operation_activity.py` reads this array out of the tree and
 * requires it to hold exactly the Python owner's words. If the two ever
 * disagree, decide which is right and move the other to follow; do not edit
 * whichever one is red. The comment on `WAITING_OPERATION_STATES` in
 * `OperationOwner.tsx` that ends "the Python one is right" is about that other
 * list and is not authority here.
 */
export const ACTIVE_OPERATION_STATES: ReadonlySet<OperationState> = new Set([
  "needs_approval",
  "ready",
  "queued",
  "running",
  "waiting",
  "paused",
]);

export interface OperationOwnerTarget {
  route: string;
  resource: string | null;
}

export interface OperationRecoverableError {
  code: string;
  message: string;
  recoverable: boolean;
  details?: unknown;
}

export interface OperationCorrection {
  available: boolean;
  fields: Record<string, unknown>;
  field?: string;
  message?: string;
  values?: Record<string, unknown>;
  [key: string]: unknown;
}

export interface OperationResult {
  summary?: string;
  open_url?: string;
  resource_url?: string;
  [key: string]: unknown;
}

export interface OperationArtifact {
  id?: string;
  name?: string;
  url?: string;
  kind?: string;
  [key: string]: unknown;
}

export interface OperationRetryLineage {
  parent_operation_id: string | null;
  root_operation_id: string;
  attempt: number;
  ancestry: string[];
  depth: number;
  [key: string]: unknown;
}

export interface OperationProgress {
  value: number | null;
  percent: number | null;
  determinate: boolean;
  label: string;
}

export interface OperationResultSummary {
  available: boolean;
  data: unknown;
}

export interface OperationTimestamps {
  created_at: string | number | null;
  started_at: string | number | null;
  updated_at: string | number | null;
  completed_at: string | number | null;
  [key: string]: unknown;
}

export interface OperationCleanup {
  state: string;
  message: string;
  available?: boolean;
  [key: string]: unknown;
}

/**
 * #180: the server's own verdict on whether a reporting operation has gone
 * quiet. Optional so pre-projection fixtures without it fall through to the
 * client computation in `operationHasStalled`; when present it is rendered
 * rather than recomputed, so the threshold and the reporting-state set live in
 * one place (`vaelor/operation_projection.py`) instead of two.
 */
export interface OperationStaleness {
  reporting: boolean;
  stale: boolean;
  silent_seconds: number | null;
  stale_after_seconds: number;
}

export interface OperationActionDescriptor {
  href?: string;
  url?: string;
  path?: string;
  method?: OperationActionMethod;
  body?: unknown;
  label?: string;
}

export type OperationActionReference = string | OperationActionDescriptor;
export type OperationActionMap = Partial<Record<OperationAction, OperationActionReference>>;
export type OperationPermissionMap = Partial<Record<OperationAction, boolean>>;
export type OperationEndpointMap = Partial<Record<"self" | "cancel" | "retry" | "resume" | "discard" | "audit", string | null>>;

/** The single server-owned wire projection shared by all operation owners. */
export interface OperationProjection {
  schema: typeof OPERATION_SCHEMA;
  operation_id: string;
  operation_key: string;
  ledger: string;
  source_id: string;
  owner: OperationOwnerTarget;
  owner_route: string;
  owner_resource: string | null;
  type: string;
  state: OperationState;
  canonical_state: OperationState;
  source_state: string;
  phase: string;
  progress: OperationProgress;
  message: string;
  error: OperationRecoverableError;
  recoverable_error: OperationRecoverableError;
  correction: OperationCorrection;
  permissions: OperationPermissionMap;
  result_summary: OperationResultSummary;
  result: OperationResult;
  artifacts: OperationArtifact[];
  endpoint: string;
  audit_link: string;
  retry_lineage: OperationRetryLineage;
  timestamps: OperationTimestamps;
  staleness?: OperationStaleness;
  cleanup: OperationCleanup;
  revision: OperationRevision;
  action_endpoints: OperationEndpointMap;
  endpoints: OperationEndpointMap;
}

export interface OperationResumeKey {
  operation_key: string;
}

export interface ResolvedOperationAction {
  path: string;
  method: OperationActionMethod;
  body?: unknown;
  label: string;
}

export const OPERATION_ACTION_LABELS: Readonly<Record<OperationAction, string>> = {
  cancel: "Cancel active operation",
  retry: "Retry operation",
  resume: "Resume saved operation",
  discard: "Discard saved draft",
  done: "Done",
};

export const OPERATION_STORAGE_LABELS: Readonly<Record<OperationAction, string>> = {
  cancel: "Active operation cancellation",
  retry: "Saved operation resume key",
  resume: "Saved operation resume key",
  discard: "Saved-draft discard",
  done: "Saved operation resume key",
};

const MAX_OPERATION_KEY_LENGTH = 512;

function isRecord(value: unknown): value is Record<string, unknown> {
  return typeof value === "object" && value !== null;
}

function normalizeKey(value: unknown): string | null {
  if (typeof value !== "string") return null;
  const key = value.trim();
  return key && key.length <= MAX_OPERATION_KEY_LENGTH ? key : null;
}

function canonicalOperationKey(value: unknown): string | null {
  const key = normalizeKey(value);
  return key && /^(jobs|agent_tasks)\/[^:]+$/.test(key) ? key : null;
}

function isOperationState(value: unknown): value is OperationState {
  return typeof value === "string" && (OPERATION_STATES as readonly string[]).includes(value);
}

function isRevision(value: unknown): value is OperationRevision {
  return (typeof value === "number" && Number.isFinite(value))
    || (typeof value === "string" && value.trim().length > 0);
}

function isProgress(value: unknown): value is OperationProgress {
  if (!isRecord(value)) return false;
  const validNumber = (candidate: unknown) => candidate === null
    || (typeof candidate === "number" && Number.isFinite(candidate) && candidate >= 0 && candidate <= 100);
  return validNumber(value.value)
    && validNumber(value.percent)
    && value.value === value.percent
    && typeof value.determinate === "boolean"
    && value.determinate === (value.value !== null)
    && typeof value.label === "string";
}

function isStringOrNull(value: unknown): value is string | null {
  return value === null || typeof value === "string";
}

function isErrorProjection(value: unknown): value is OperationRecoverableError {
  return isRecord(value)
    && typeof value.code === "string"
    && typeof value.message === "string"
    && typeof value.recoverable === "boolean";
}

function isEndpointMap(value: unknown): value is OperationEndpointMap {
  if (!isRecord(value) || typeof value.self !== "string" || typeof value.audit !== "string") return false;
  return ["cancel", "retry", "resume", "discard"].every((key) => !(key in value) || isStringOrNull(value[key]));
}

function isTimestamps(value: unknown): value is OperationTimestamps {
  if (!isRecord(value)) return false;
  return ["created_at", "started_at", "updated_at", "completed_at"].every((key) => {
    const candidate = value[key];
    return candidate === null || typeof candidate === "string" || (typeof candidate === "number" && Number.isFinite(candidate));
  });
}

function isRetryLineage(value: unknown): value is OperationRetryLineage {
  return isRecord(value)
    && isStringOrNull(value.parent_operation_id)
    && typeof value.root_operation_id === "string"
    && Number.isInteger(value.depth) && Number(value.depth) >= 0
    && Number.isInteger(value.attempt) && Number(value.attempt) >= 1
    && Array.isArray(value.ancestry) && value.ancestry.every((item) => typeof item === "string");
}

export function isOperationProjection(value: unknown): value is OperationProjection {
  if (!isRecord(value)) return false;
  const ledger = value.ledger;
  const sourceId = value.source_id;
  return value.schema === OPERATION_SCHEMA
    && (ledger === "jobs" || ledger === "agent_tasks")
    && typeof sourceId === "string" && sourceId.length > 0 && !sourceId.includes(":")
    && value.operation_key === `${ledger}/${sourceId}`
    && value.operation_id === `${ledger}:${sourceId}`
    && isRecord(value.owner) && typeof value.owner.route === "string" && isStringOrNull(value.owner.resource)
    && value.owner_route === value.owner.route
    && value.owner_resource === value.owner.resource
    && typeof value.type === "string"
    && isOperationState(value.state)
    && value.canonical_state === value.state
    && typeof value.source_state === "string"
    && typeof value.phase === "string"
    && isProgress(value.progress)
    && typeof value.message === "string"
    && isErrorProjection(value.error)
    && isErrorProjection(value.recoverable_error)
    && isRecord(value.correction) && typeof value.correction.available === "boolean" && typeof value.correction.message === "string" && isRecord(value.correction.fields)
    && isRecord(value.permissions) && ["cancel", "retry", "resume", "discard"].every((key) => typeof (value.permissions as Record<string, unknown>)[key] === "boolean")
    && isRecord(value.result_summary) && typeof value.result_summary.available === "boolean" && "data" in value.result_summary
    && isRecord(value.result)
    && Array.isArray(value.artifacts)
    && typeof value.endpoint === "string"
    && typeof value.audit_link === "string"
    && isRetryLineage(value.retry_lineage)
    && isTimestamps(value.timestamps)
    && isRecord(value.cleanup) && typeof value.cleanup.state === "string" && typeof value.cleanup.message === "string"
    && isEndpointMap(value.action_endpoints)
    && isEndpointMap(value.endpoints)
    && value.endpoint === value.action_endpoints.self
    && value.audit_link === value.action_endpoints.audit
    && JSON.stringify(value.endpoints) === JSON.stringify(value.action_endpoints)
    && isRevision(value.revision);
}

export function createOperationResumeKey(operationKey: unknown): OperationResumeKey | null {
  const normalized = canonicalOperationKey(operationKey);
  return normalized ? { operation_key: normalized } : null;
}

export function serializeOperationResumeKey(operationKey: unknown): string | null {
  const resumeKey = createOperationResumeKey(operationKey);
  return resumeKey ? JSON.stringify(resumeKey) : null;
}

export function parseOperationResumeKey(raw: string | null | undefined): OperationResumeKey | null {
  if (!raw) return null;
  try {
    const parsed: unknown = JSON.parse(raw);
    if (!isRecord(parsed) || Object.keys(parsed).length !== 1) return null;
    return createOperationResumeKey(parsed.operation_key);
  } catch {
    return null;
  }
}

export function readOperationResumeKey(storage: Storage | undefined, storageKey = OPERATION_RESUME_STORAGE_KEY): OperationResumeKey | null {
  if (!storage) return null;
  try {
    return parseOperationResumeKey(storage.getItem(storageKey));
  } catch {
    return null;
  }
}

export function writeOperationResumeKey(
  storage: Storage | undefined,
  operationKey: unknown,
  storageKey = OPERATION_RESUME_STORAGE_KEY,
): OperationResumeKey | null {
  const resumeKey = createOperationResumeKey(operationKey);
  if (!storage || !resumeKey) return null;
  try {
    storage.setItem(storageKey, JSON.stringify(resumeKey));
  } catch {
    return null;
  }
  return resumeKey;
}

export function clearOperationResumeKey(
  storage: Storage | undefined,
  storageKey = OPERATION_RESUME_STORAGE_KEY,
): void {
  try {
    storage?.removeItem(storageKey);
  } catch {
    // Storage can be unavailable in private browsing or an embedded document.
  }
}

export function operationIsTerminal(state: OperationState): boolean {
  return TERMINAL_OPERATION_STATES.has(state);
}

export function operationIsActive(state: OperationState): boolean {
  return ACTIVE_OPERATION_STATES.has(state);
}

function numericRevision(value: OperationRevision): number | null {
  const numberValue = typeof value === "number" ? value : Number(value);
  return Number.isFinite(numberValue) ? numberValue : null;
}

export function compareOperationRevision(left: OperationRevision, right: OperationRevision): number {
  const leftNumber = numericRevision(left);
  const rightNumber = numericRevision(right);
  if (leftNumber !== null && rightNumber !== null) return leftNumber - rightNumber;
  return String(left).localeCompare(String(right));
}

export function operationRevisionKey(operation: OperationProjection): string {
  return `${operation.operation_id}:${String(operation.revision)}`;
}

export function operationActionLabel(action: OperationAction): string {
  return OPERATION_ACTION_LABELS[action];
}

function actionReference(operation: OperationProjection, action: OperationAction): OperationActionReference | undefined {
  if (action === "done") return undefined;
  return operation.action_endpoints?.[action] ?? operation.endpoints?.[action] ?? undefined;
}

function actionPath(reference: OperationActionReference): string | null {
  const raw = typeof reference === "string"
    ? reference
    : reference.href ?? reference.url ?? reference.path;
  if (!raw || !raw.trim()) return null;
  const candidate = raw.trim();
  if (/^https?:\/\//i.test(candidate)) {
    try {
      const url = new URL(candidate);
      if (typeof window !== "undefined" && url.origin !== window.location.origin) return null;
      return `${url.pathname}${url.search}`;
    } catch {
      return null;
    }
  }
  const path = candidate.startsWith("/") ? candidate : `/${candidate}`;
  return path.startsWith("/api/v2/") ? path.slice("/api/v2".length) : path;
}

export function resolveOperationAction(
  operation: OperationProjection,
  action: OperationAction,
): ResolvedOperationAction | null {
  if (operation.permissions?.[action] === false) return null;
  const reference = actionReference(operation, action);
  if (!reference) return null;
  const path = actionPath(reference);
  if (!path) return null;
  const descriptor = typeof reference === "string" ? undefined : reference;
  return {
    path,
    method: descriptor?.method ?? "POST",
    ...(descriptor && "body" in descriptor ? { body: descriptor.body } : {}),
    label: descriptor?.label ?? operationActionLabel(action),
  };
}

export function operationIdFromKey(operationKey: string): string | null {
  const normalized = normalizeKey(operationKey);
  if (!normalized) return null;
  if (/^(jobs|agent_tasks):[^:]+$/.test(normalized)) return normalized;
  const match = /^(jobs|agent_tasks)\/(.+)$/.exec(normalized);
  return match ? `${match[1]}:${match[2]}` : null;
}

export function operationDetailPath(operationId: string): string {
  return `/operations/${encodeURIComponent(operationId)}`;
}

/** Extracts a v1 projection from either a direct response or a collection envelope. */
export function findOperationProjection(payload: unknown, operationKey?: string): OperationProjection | null {
  const expectedKey = normalizeKey(operationKey);
  const queue: unknown[] = [payload];
  const visited = new Set<object>();
  let fallback: OperationProjection | null = null;
  let inspected = 0;
  while (queue.length && inspected < 128) {
    const candidate = queue.shift();
    inspected += 1;
    if (isOperationProjection(candidate)) {
      if (!expectedKey || candidate.operation_key === expectedKey || candidate.operation_id === expectedKey) return candidate;
      fallback ??= candidate;
      continue;
    }
    if (Array.isArray(candidate)) {
      queue.push(...candidate);
      continue;
    }
    if (!isRecord(candidate) || visited.has(candidate)) continue;
    visited.add(candidate);
    for (const key of ["operation", "projection", "operations", "items", "data"]) {
      if (key in candidate) queue.push(candidate[key]);
    }
  }
  return expectedKey ? null : fallback;
}

export type OperationRequest = <T>(path: string, options?: RequestInit, csrfToken?: string) => Promise<T>;
