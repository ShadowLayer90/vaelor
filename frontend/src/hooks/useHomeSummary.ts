import { useEffect, useState } from "react";
import { apiRequest } from "../lib/api";
import type { AgentStatus } from "../components/agentTypes";
import type { AiChatSetup } from "../components/aiChatTypes";
import type { FleetSummary } from "../components/fleetTypes";
import type { ClusterCapacityLedger } from "../components/fleetTypes";
import type { LlmServerSurface } from "../lib/endpoints";
import type { OperationProjection } from "../lib/operationOwner";
import type { TelemetryHistory } from "./useMachineMetrics";
import type { Role } from "../types";

/**
 * Every read Home's redesigned panels draw from (VD-200: "match the mockups,
 * wired to the data that already exists"). Each read is independent: one that
 * fails or that this role may not make leaves its own panel saying so, never
 * the others. A read is `{state}` first and data second, so "not read" can
 * never be mistaken for an empty answer (LESSONS 1, 8).
 */
export type Read<T> =
  | { state: "loading" }
  | { state: "ok"; data: T }
  | { state: "forbidden" }
  | { state: "failed" };

export interface OperationsAttention {
  items: OperationProjection[];
  summary?: { attention?: number };
}

export interface HomeSummary {
  history: Read<TelemetryHistory>;
  agent: Read<AgentStatus>;
  intelligenceChoice: Read<{ intelligence_choice?: string }>;
  aiChat: Read<AiChatSetup>;
  llmServer: Read<LlmServerSurface>;
  cluster: Read<FleetSummary>;
  capacity: Read<ClusterCapacityLedger>;
  attention: Read<OperationsAttention>;
  /** Each enrolled worker's own short history, keyed by its fleet id. */
  workers: Record<string, Read<TelemetryHistory>>;
}

const LOADING = { state: "loading" } as const;

/** How often Home re-reads its panels. The live tiles have their own 2.5 s poll. */
export const SUMMARY_POLL_MS = 30_000;

/** One route read as a `Read`: a 401 or 403 is "forbidden", anything else that fails is "failed". */
export async function read<T>(path: string, signal: AbortSignal): Promise<Read<T>> {
  try {
    return { state: "ok", data: await apiRequest<T>(path, { signal }) };
  } catch (error) {
    const status = (error as { status?: unknown } | null)?.status;
    if (status === 401 || status === 403) return { state: "forbidden" };
    return { state: "failed" };
  }
}

/** Which reads a role may make (the routes' own minimum roles). */
function allowed(role: Role) {
  return {
    operator: role === "operator" || role === "administrator",
    administrator: role === "administrator",
  };
}

type OwnReads = Omit<HomeSummary, "cluster" | "workers">;

/**
 * `cluster` is the shell's one `/cluster` read (useClusterSummary), shared
 * rather than read a second time here.
 */
export function useHomeSummary(role: Role, enabled: boolean, cluster: Read<FleetSummary>): HomeSummary {
  const [own, setOwn] = useState<OwnReads>({
    history: LOADING, agent: LOADING, intelligenceChoice: LOADING, aiChat: LOADING, llmServer: LOADING,
    capacity: LOADING, attention: LOADING,
  });
  const [workers, setWorkers] = useState<Record<string, Read<TelemetryHistory>>>({});
  const nodeIds = cluster.state === "ok" && Array.isArray(cluster.data?.enrolled_nodes)
    ? cluster.data.enrolled_nodes.map((node) => node.id)
    : [];
  // A stable key, so the worker reads restart when the machines change and not on every re-read.
  const nodeKey = JSON.stringify(nodeIds);

  useEffect(() => {
    if (!enabled) return;
    const controller = new AbortController();
    const { signal } = controller;
    const can = allowed(role);
    const forbidden = { state: "forbidden" } as const;
    const refresh = async () => {
      const [history, agent, intelligenceChoice, aiChat, llmServer, capacity, attention] = await Promise.all([
        read<TelemetryHistory>("/telemetry/history?window=1h", signal),
        read<AgentStatus>("/agent/status", signal),
        can.administrator ? read<{ intelligence_choice?: string }>("/assistant/preferences", signal) : forbidden,
        can.operator ? read<AiChatSetup>("/ai-chat/setup", signal) : forbidden,
        can.administrator ? read<LlmServerSurface>("/llm-server", signal) : forbidden,
        can.operator ? read<ClusterCapacityLedger>("/cluster/capacity", signal) : forbidden,
        read<OperationsAttention>("/operations?limit=5&bucket=attention", signal),
      ]);
      if (signal.aborted) return;
      setOwn({ history, agent, intelligenceChoice, aiChat, llmServer, capacity, attention });
    };
    void refresh();
    const timer = window.setInterval(() => { if (!document.hidden) void refresh(); }, SUMMARY_POLL_MS);
    return () => {
      window.clearInterval(timer);
      controller.abort();
    };
  }, [enabled, role]);

  useEffect(() => {
    if (!enabled) return;
    const controller = new AbortController();
    const { signal } = controller;
    const ids = JSON.parse(nodeKey) as string[];
    const refresh = async () => {
      const reads = await Promise.all(ids.map(async (id) => [
        id,
        await read<TelemetryHistory>(`/telemetry/history?node=${encodeURIComponent(id)}&window=15m`, signal),
      ] as const));
      if (!signal.aborted) setWorkers(Object.fromEntries(reads));
    };
    void refresh();
    const timer = window.setInterval(() => { if (!document.hidden) void refresh(); }, SUMMARY_POLL_MS);
    return () => {
      window.clearInterval(timer);
      controller.abort();
    };
  }, [enabled, nodeKey]);

  return { ...own, cluster, workers };
}
