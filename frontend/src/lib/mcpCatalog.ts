import { useCallback, useEffect, useState } from "react";
import { apiRequest } from "./api";

/**
 * MCP server catalog (F6a): the typed data behind the "Agents & tools" fleet
 * section. It lists the Model Context Protocol servers an agent may call, and -
 * for an administrator - registers, edits, enables, re-probes and removes them.
 *
 * The honesty rule the backend enforces is carried here: a server row never
 * holds a secret. `credential_id` is the id of a stored credential, never its
 * value, and `health` is whatever the last probe recorded - read defensively,
 * never assumed.
 */

/** The last recorded health probe. Its shape is the backend's; read defensively. */
export type McpServerHealth = Record<string, unknown>;

/** One registered MCP server, exactly as the catalog returns it. Never a secret. */
export interface McpServer {
  id: string;
  name: string;
  kind: string;
  endpoint: string;
  credential_id: string;
  enabled: boolean;
  approved_tools: string[];
  health: McpServerHealth;
  /** When the health reading was taken (unix seconds), or null if never. */
  checked_at: number | null;
  /** How many tools the last check found, or null when it did not say. */
  tools_count: number | null;
  /**
   * The controller started a background re-check because the reading was older
   * than five minutes; the list should be read again shortly (ACC-077).
   */
  health_refreshing: boolean;
  created_at: number | null;
  updated_at: number | null;
}

/**
 * The catalog surface: whether an MCP catalog is configured on this appliance at
 * all, the registered servers, and an optional note the backend uses to explain
 * an unconfigured or degraded catalog in its own words.
 */
export interface McpCatalog {
  configured: boolean;
  servers: McpServer[];
  note: string;
}

/** The fields an administrator supplies to register or edit a server. */
export interface McpServerInput {
  name: string;
  kind: string;
  endpoint: string;
  credential_id: string;
  enabled: boolean;
  approved_tools: string[];
}

/** The wire shape of `GET /mcp/catalog/servers`. */
interface McpCatalogSurface {
  configured?: boolean;
  servers?: unknown;
  note?: string;
}

function asString(value: unknown): string {
  return typeof value === "string" ? value : "";
}

function asNumberOrNull(value: unknown): number | null {
  return typeof value === "number" && Number.isFinite(value) ? value : null;
}

function asToolNames(value: unknown): string[] {
  return Array.isArray(value) ? value.filter((item): item is string => typeof item === "string") : [];
}

/**
 * Coerce one server object from any of the routes into the {@link McpServer} the
 * surface renders. The catalog, create, update and health routes all return a
 * server in the same shape; a missing or wrong-typed field degrades to an empty
 * default rather than throwing, so a slightly older backend never blanks the
 * whole surface.
 */
function asCountOrNull(value: unknown): number | null {
  return typeof value === "number" && Number.isFinite(value) && value >= 0 ? Math.floor(value) : null;
}

function normalizeServer(value: unknown): McpServer {
  const record = (value ?? {}) as Record<string, unknown>;
  const health = record.health;
  const healthRecord = health && typeof health === "object" ? (health as Record<string, unknown>) : {};
  return {
    id: asString(record.id),
    name: asString(record.name),
    kind: asString(record.kind),
    endpoint: asString(record.endpoint),
    credential_id: asString(record.credential_id),
    enabled: record.enabled === true,
    approved_tools: asToolNames(record.approved_tools),
    health: healthRecord,
    checked_at: asNumberOrNull(healthRecord.checked_at),
    tools_count: asCountOrNull(healthRecord.tools_count),
    health_refreshing: record.health_refreshing === true,
    created_at: asNumberOrNull(record.created_at),
    updated_at: asNumberOrNull(record.updated_at),
  };
}

/** Read the catalog. Reports `configured: false` with no servers when unset. */
export async function listMcpServers(signal?: AbortSignal): Promise<McpCatalog> {
  const surface = await apiRequest<McpCatalogSurface>("/mcp/catalog/servers", { cache: "no-store", signal });
  return {
    configured: surface.configured === true,
    servers: Array.isArray(surface.servers) ? surface.servers.map(normalizeServer) : [],
    note: typeof surface.note === "string" ? surface.note : "",
  };
}

/** Register a new server (administrator). Returns the created server. */
export async function registerMcpServer(input: McpServerInput, csrfToken: string): Promise<McpServer> {
  const server = await apiRequest<unknown>(
    "/mcp/catalog/servers",
    { method: "POST", body: JSON.stringify(input), cache: "no-store" },
    csrfToken,
  );
  return normalizeServer(server);
}

/**
 * Apply a partial update to a server (administrator) - a single `{ enabled }`
 * for the row toggle, or a set of edited fields. Returns the updated server.
 */
export async function updateMcpServer(
  id: string,
  patch: Partial<McpServerInput>,
  csrfToken: string,
): Promise<McpServer> {
  const server = await apiRequest<unknown>(
    `/mcp/catalog/servers/${encodeURIComponent(id)}`,
    { method: "POST", body: JSON.stringify(patch), cache: "no-store" },
    csrfToken,
  );
  return normalizeServer(server);
}

/** Remove a server from the catalog (administrator). */
export async function deleteMcpServer(id: string, csrfToken: string): Promise<void> {
  await apiRequest<unknown>(
    `/mcp/catalog/servers/${encodeURIComponent(id)}`,
    { method: "DELETE", cache: "no-store" },
    csrfToken,
  );
}

/**
 * Re-probe a server's health (administrator). The health route returns only
 * `{id, health}` (not a full server view), so this returns just that pair; the
 * caller should reload the catalog to refresh the rest of the row.
 */
export async function probeMcpServerHealth(
  id: string,
  csrfToken: string,
): Promise<{ id: string; health: McpServerHealth }> {
  const record = await apiRequest<Record<string, unknown>>(
    `/mcp/catalog/servers/${encodeURIComponent(id)}/health`,
    { method: "POST", body: "{}", cache: "no-store" },
    csrfToken,
  );
  const health = record.health;
  return {
    id: asString(record.id) || id,
    health: health && typeof health === "object" ? (health as McpServerHealth) : {},
  };
}

/**
 * Split a free-text tool-name entry ("read_file, list_dir search") into a
 * de-duplicated list, tolerating commas, spaces and newlines as separators so
 * the operator is not forced into one delimiter.
 */
export function parseApprovedTools(value: string): string[] {
  const seen = new Set<string>();
  for (const token of value.split(/[\s,]+/)) {
    const name = token.trim();
    if (name) seen.add(name);
  }
  return [...seen];
}

/** A StatusPill tone the panel can render for a server's last health probe. */
export type HealthTone = "success" | "warning" | "danger" | "info" | "neutral";

/** A health probe rendered as a pill: a label, a tone, and any probe detail. */
export interface HealthSummary {
  label: string;
  tone: HealthTone;
  detail: string;
}

const HEALTHY = new Set(["healthy", "ok", "up", "ready", "reachable", "online", "pass", "passing"]);
const DEGRADED = new Set(["degraded", "warning", "warn", "slow"]);
const FAILING = new Set(["error", "unreachable", "down", "failed", "failing", "offline", "unhealthy"]);

/**
 * Read a probe's recorded health without assuming its exact shape. It honours a
 * string `status` (mapped to a tone, but shown in the backend's own word), falls
 * back to a boolean `ok`, and otherwise reports "Not checked" rather than
 * inventing a verdict. Any `detail`, `error` or `message` string is surfaced.
 */
export function summarizeHealth(health: McpServerHealth): HealthSummary {
  const record = health ?? {};
  const detail = asString(record.detail) || asString(record.error) || asString(record.message);
  const rawStatus = asString(record.status).trim();
  if (!rawStatus) {
    if (typeof record.ok === "boolean") {
      return record.ok
        ? { label: "Healthy", tone: "success", detail }
        : { label: "Unreachable", tone: "danger", detail };
    }
    return { label: "Not checked", tone: "neutral", detail };
  }
  const key = rawStatus.toLowerCase();
  const tone: HealthTone = HEALTHY.has(key)
    ? "success"
    : DEGRADED.has(key)
      ? "warning"
      : FAILING.has(key)
        ? "danger"
        : "info";
  return { label: rawStatus.charAt(0).toUpperCase() + rawStatus.slice(1), tone, detail };
}

export interface McpCatalogState {
  catalog: McpCatalog;
  loading: boolean;
  error: string;
  reload: () => void;
}

/** While a server's health is being re-checked, the list is re-read this often. */
export const MCP_REFRESHING_POLL_MS = 3_000;
/** Otherwise a mounted catalog panel re-reads the list this often. */
export const MCP_IDLE_POLL_MS = 60_000;

/**
 * Read the MCP catalog, exposing the same three states the sibling admin
 * surfaces use: a first load, a read catalog, and a failure carrying the
 * appliance's own message. The request is abortable so leaving the tab never
 * lands a stale answer, and `reload` re-reads after a register, edit or removal.
 *
 * With `poll`, the list is also re-read in the background - every
 * {@link MCP_REFRESHING_POLL_MS} while any server's health is being
 * re-checked, else every {@link MCP_IDLE_POLL_MS} - so a server that went down
 * stops reading healthy without anyone pressing Re-check (ACC-077). One
 * background read at a time: the next is scheduled only after the last ends.
 */
export function useMcpCatalog({ poll = false }: { poll?: boolean } = {}): McpCatalogState {
  const [catalog, setCatalog] = useState<McpCatalog>({ configured: false, servers: [], note: "" });
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState("");
  const [reloadKey, setReloadKey] = useState(0);
  const [pollTick, setPollTick] = useState(0);
  const reload = useCallback(() => setReloadKey((value) => value + 1), []);
  const refreshing = catalog.servers.some((server) => server.health_refreshing);

  useEffect(() => {
    if (!poll) return;
    const controller = new AbortController();
    const timer = window.setTimeout(() => {
      listMcpServers(controller.signal)
        .then((result) => {
          if (controller.signal.aborted) return;
          setCatalog(result);
          setError("");
        })
        .catch((reason: unknown) => {
          if (controller.signal.aborted) return;
          setError(reason instanceof Error ? reason.message : "The MCP catalog could not be read.");
        })
        .finally(() => {
          if (!controller.signal.aborted) setPollTick((value) => value + 1);
        });
    }, refreshing ? MCP_REFRESHING_POLL_MS : MCP_IDLE_POLL_MS);
    return () => {
      window.clearTimeout(timer);
      controller.abort();
    };
  }, [poll, pollTick, refreshing, reloadKey]);

  useEffect(() => {
    const controller = new AbortController();
    setLoading(true);
    listMcpServers(controller.signal)
      .then((result) => {
        if (controller.signal.aborted) return;
        setCatalog(result);
        setError("");
      })
      .catch((reason: unknown) => {
        if (controller.signal.aborted) return;
        setError(reason instanceof Error ? reason.message : "The MCP catalog could not be read.");
      })
      .finally(() => {
        if (!controller.signal.aborted) setLoading(false);
      });
    return () => controller.abort();
  }, [reloadKey]);

  return { catalog, loading, error, reload };
}
