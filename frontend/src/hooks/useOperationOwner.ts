import { useCallback, useEffect, useRef, useState } from "react";
import { apiRequest } from "../lib/api";
import {
  clearOperationResumeKey,
  compareOperationRevision,
  createOperationResumeKey,
  findOperationProjection,
  operationDetailPath,
  operationIdFromKey,
  operationIsActive,
  operationIsTerminal,
  operationRevisionKey,
  readOperationResumeKey,
  resolveOperationAction,
  writeOperationResumeKey,
  OPERATION_RESUME_STORAGE_KEY,
  type OperationAction,
  type OperationConnection,
  type OperationProjection,
  type OperationRequest,
  type OperationResumeKey,
} from "../lib/operationOwner";

export interface UseOperationOwnerOptions {
  operationKey?: string | null;
  storage?: Storage | null;
  resumeStorageKey?: string;
  pollIntervalMs?: number;
  enabled?: boolean;
  csrfToken?: string;
  request?: OperationRequest;
  onResourceRefresh?: (operation: OperationProjection) => void | Promise<void>;
}

export interface UseOperationOwnerResult {
  operation: OperationProjection | null;
  lastServerTruth: OperationProjection | null;
  terminalResult: OperationProjection["result"] | null;
  resumeKey: OperationResumeKey | null;
  hasSavedState: boolean;
  connection: OperationConnection;
  error: string | null;
  actionError: string | null;
  resourceRefreshError: string | null;
  actionPending: OperationAction | null;
  isActive: boolean;
  isTerminal: boolean;
  refresh: () => Promise<OperationProjection | null>;
  cancel: () => Promise<OperationProjection | null>;
  retry: () => Promise<OperationProjection | null>;
  resume: () => Promise<OperationProjection | null>;
  discardSavedState: () => Promise<OperationProjection | null>;
  done: () => Promise<OperationProjection | null>;
}

const DEFAULT_POLL_INTERVAL_MS = 3000;

function browserStorage(): Storage | undefined {
  if (typeof window === "undefined") return undefined;
  try {
    return window.localStorage;
  } catch {
    return undefined;
  }
}

function errorMessage(error: unknown, fallback: string): string {
  return error instanceof Error && error.message ? error.message : fallback;
}

function boundedPollInterval(interval: number | undefined): number {
  if (!Number.isFinite(interval)) return DEFAULT_POLL_INTERVAL_MS;
  return Math.max(250, Math.floor(interval ?? DEFAULT_POLL_INTERVAL_MS));
}

export function useOperationOwner(options: UseOperationOwnerOptions = {}): UseOperationOwnerResult {
  const storage = options.storage === undefined ? browserStorage() : options.storage ?? undefined;
  const resumeStorageKey = options.resumeStorageKey ?? OPERATION_RESUME_STORAGE_KEY;
  const directKey = createOperationResumeKey(options.operationKey)?.operation_key ?? null;
  const initialResumeKey = directKey
    ? createOperationResumeKey(directKey)
    : readOperationResumeKey(storage, resumeStorageKey);
  const enabled = options.enabled !== false;
  const requestRef = useRef<OperationRequest>(options.request ?? apiRequest);
  const csrfTokenRef = useRef(options.csrfToken);
  const resourceRefreshRef = useRef(options.onResourceRefresh);
  const mountedRef = useRef(false);
  const generationRef = useRef(0);
  const operationRef = useRef<OperationProjection | null>(null);
  const operationKeyRef = useRef<string | null>(initialResumeKey?.operation_key ?? null);
  const inFlightRef = useRef<Promise<OperationProjection | null> | null>(null);
  const actionPromiseRef = useRef<Promise<OperationProjection | null> | null>(null);
  const actionPendingRef = useRef<OperationAction | null>(null);
  const terminalRefreshesRef = useRef(new Set<string>());

  requestRef.current = options.request ?? apiRequest;
  csrfTokenRef.current = options.csrfToken;
  resourceRefreshRef.current = options.onResourceRefresh;

  const [resolvedOperationKey, setResolvedOperationKey] = useState<string | null>(initialResumeKey?.operation_key ?? null);
  const [resumeKey, setResumeKey] = useState<OperationResumeKey | null>(initialResumeKey);
  const [operation, setOperation] = useState<OperationProjection | null>(null);
  const [lastServerTruth, setLastServerTruth] = useState<OperationProjection | null>(null);
  const [connection, setConnection] = useState<OperationConnection>(enabled && initialResumeKey ? "hydrating" : "idle");
  const [error, setError] = useState<string | null>(null);
  const [actionError, setActionError] = useState<string | null>(null);
  const [resourceRefreshError, setResourceRefreshError] = useState<string | null>(null);
  const [actionPending, setActionPending] = useState<OperationAction | null>(null);

  const applyProjection = useCallback((next: OperationProjection): OperationProjection => {
    const current = operationRef.current;
    if (
      current
      && current.operation_key === next.operation_key
      && compareOperationRevision(next.revision, current.revision) < 0
    ) {
      return current;
    }

    operationRef.current = next;
    operationKeyRef.current = next.operation_key;
    setResolvedOperationKey(next.operation_key);
    const safeResumeKey = createOperationResumeKey(next.operation_key);
    writeOperationResumeKey(storage, next.operation_key, resumeStorageKey);
    setResumeKey(safeResumeKey);
    setOperation(next);
    setLastServerTruth(next);
    setConnection("connected");
    setError(null);

    if (operationIsTerminal(next.state)) {
      const revision = operationRevisionKey(next);
      if (!terminalRefreshesRef.current.has(revision)) {
        terminalRefreshesRef.current.add(revision);
        const refresh = resourceRefreshRef.current;
        if (refresh) {
          void Promise.resolve(refresh(next)).then(
            () => {
              if (mountedRef.current) setResourceRefreshError(null);
            },
            (refreshError: unknown) => {
              if (mountedRef.current) setResourceRefreshError(errorMessage(refreshError, "The owned resource could not be refreshed."));
            },
          );
        }
      }
    }
    return next;
  }, [resumeStorageKey, storage]);

  const refreshForKey = useCallback(async (operationKey: string): Promise<OperationProjection | null> => {
    if (inFlightRef.current) return inFlightRef.current;
    const generation = generationRef.current;
    const task = (async () => {
      if (!operationRef.current && mountedRef.current) setConnection("hydrating");
      try {
        const current = operationRef.current;
        const operationId = current && current.operation_key === operationKey
          ? current.operation_id
          : operationIdFromKey(operationKey);
        if (!operationId) throw new Error("The saved operation key is invalid.");
        const path = operationDetailPath(operationId);
        const response = await requestRef.current<unknown>(path, undefined, csrfTokenRef.current);
        const next = findOperationProjection(response, operationKey);
        if (!next) throw new Error("The saved operation could not be found.");
        if (generation !== generationRef.current || !mountedRef.current) return operationRef.current;
        return applyProjection(next);
      } catch (refreshError: unknown) {
        if (generation === generationRef.current && mountedRef.current) {
          setConnection("reconnecting");
          setError(errorMessage(refreshError, "The operation could not be refreshed."));
        }
        throw refreshError;
      }
    })();
    inFlightRef.current = task;
    task.then(
      () => {
        if (inFlightRef.current === task) inFlightRef.current = null;
      },
      () => {
        if (inFlightRef.current === task) inFlightRef.current = null;
      },
    );
    return task;
  }, [applyProjection]);

  const refresh = useCallback(() => {
    const key = operationKeyRef.current;
    if (!key) return Promise.resolve(null);
    return refreshForKey(key);
  }, [refreshForKey]);

  const runAction = useCallback((action: OperationAction): Promise<OperationProjection | null> => {
    if (actionPromiseRef.current) return actionPromiseRef.current;
    const current = operationRef.current;
    if (!current) {
      const missing = Promise.reject(new Error("There is no saved operation to update."));
      setActionError("There is no saved operation to update.");
      return missing;
    }
    const resolved = resolveOperationAction(current, action);
    if (!resolved) {
      const unavailable = Promise.reject(new Error(`The ${action.replaceAll("_", " ")} action is not available.`));
      setActionError(`The ${action.replaceAll("_", " ")} action is not available.`);
      return unavailable;
    }

    actionPendingRef.current = action;
    setActionPending(action);
    setActionError(null);
    const operationKey = current.operation_key;
    const requestOptions: RequestInit = { method: resolved.method };
    if (resolved.body !== undefined) requestOptions.body = JSON.stringify(resolved.body);

    const task = (async () => {
      try {
        const response = await requestRef.current<unknown>(resolved.path, requestOptions, csrfTokenRef.current);
        if (action === "discard") {
          clearOperationResumeKey(storage, resumeStorageKey);
          operationRef.current = null;
          operationKeyRef.current = null;
          setResolvedOperationKey(null);
          setResumeKey(null);
          setOperation(null);
          setLastServerTruth(null);
          setConnection("idle");
          setError(null);
          return null;
        }

        const responseProjection = findOperationProjection(response);
        let next: OperationProjection | null = null;
        if (responseProjection) {
          const belongsToOwner = action === "retry"
            ? responseProjection.retry_lineage.parent_operation_id === current.operation_id
            : responseProjection.operation_id === current.operation_id;
          if (!belongsToOwner) throw new Error("The operation service returned a result for a different operation.");
          next = applyProjection(responseProjection);
        } else {
          next = await refreshForKey(operationKey);
        }
        return next;
      } catch (actionFailure: unknown) {
        if (mountedRef.current) setActionError(errorMessage(actionFailure, "The operation action could not be completed."));
        throw actionFailure;
      } finally {
        if (mountedRef.current) {
          actionPendingRef.current = null;
          setActionPending(null);
        }
        actionPromiseRef.current = null;
      }
    })();
    actionPromiseRef.current = task;
    return task;
  }, [applyProjection, refreshForKey, resumeStorageKey, storage]);

  useEffect(() => {
    requestRef.current = options.request ?? apiRequest;
    csrfTokenRef.current = options.csrfToken;
    resourceRefreshRef.current = options.onResourceRefresh;
  }, [options.csrfToken, options.onResourceRefresh, options.request]);

  useEffect(() => {
    mountedRef.current = true;
    generationRef.current += 1;
    const generation = generationRef.current;
    const nextKey = directKey ?? readOperationResumeKey(storage, resumeStorageKey)?.operation_key ?? null;
    operationKeyRef.current = nextKey;
    setResolvedOperationKey(nextKey);
    setResumeKey(nextKey ? createOperationResumeKey(nextKey) : null);
    operationRef.current = null;
    setOperation(null);
    setLastServerTruth(null);
    setError(null);
    setActionError(null);
    setResourceRefreshError(null);
    terminalRefreshesRef.current.clear();

    if (!enabled || !nextKey) {
      setConnection("idle");
      return () => {
        mountedRef.current = false;
        generationRef.current += 1;
      };
    }
    setConnection("hydrating");
    if (directKey) {
      writeOperationResumeKey(storage, directKey, resumeStorageKey);
      setResumeKey(createOperationResumeKey(directKey));
    }
    void refreshForKey(nextKey).catch(() => undefined);
    return () => {
      if (generation === generationRef.current) generationRef.current += 1;
      mountedRef.current = false;
    };
  }, [directKey, enabled, refreshForKey, resumeStorageKey, storage]);

  useEffect(() => {
    if (!enabled || !resolvedOperationKey) return;
    const intervalMs = boundedPollInterval(options.pollIntervalMs);
    const poll = () => {
      const current = operationRef.current;
      if (!current || operationIsActive(current.state)) {
        void refreshForKey(operationKeyRef.current ?? resolvedOperationKey).catch(() => undefined);
      }
    };
    const interval = window.setInterval(poll, intervalMs);
    const reconnect = () => poll();
    window.addEventListener("online", reconnect);
    return () => {
      window.clearInterval(interval);
      window.removeEventListener("online", reconnect);
    };
  }, [enabled, options.pollIntervalMs, refreshForKey, resolvedOperationKey]);

  const cancel = useCallback(() => runAction("cancel"), [runAction]);
  const retry = useCallback(() => runAction("retry"), [runAction]);
  const resume = useCallback(() => {
    if (actionPromiseRef.current) return actionPromiseRef.current;
    actionPendingRef.current = "resume";
    setActionPending("resume");
    setActionError(null);
    const task = refresh().catch((resumeFailure: unknown) => {
      if (mountedRef.current) setActionError(errorMessage(resumeFailure, "The saved operation could not be resumed."));
      throw resumeFailure;
    }).finally(() => {
      if (mountedRef.current) {
        actionPendingRef.current = null;
        setActionPending(null);
      }
      actionPromiseRef.current = null;
    });
    actionPromiseRef.current = task;
    return task;
  }, [refresh]);
  const discardSavedState = useCallback(() => runAction("discard"), [runAction]);
  const done = useCallback(() => {
    clearOperationResumeKey(storage, resumeStorageKey);
    setResumeKey(null);
    return Promise.resolve(operationRef.current);
  }, [resumeStorageKey, storage]);
  const currentOperation = operation ?? lastServerTruth;

  return {
    operation: currentOperation,
    lastServerTruth,
    terminalResult: currentOperation?.result ?? null,
    resumeKey,
    hasSavedState: resumeKey !== null,
    connection,
    error,
    actionError,
    resourceRefreshError,
    actionPending,
    isActive: currentOperation ? operationIsActive(currentOperation.state) : false,
    isTerminal: currentOperation ? operationIsTerminal(currentOperation.state) : false,
    refresh,
    cancel,
    retry,
    resume,
    discardSavedState,
    done,
  };
}
