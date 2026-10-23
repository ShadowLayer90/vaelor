import { bytesIn, formatQuantity } from "./format";
import type { StatusTone } from "../components/ui";

/**
 * The machine settings surface (`GET /host-settings`): each machine's GPU
 * memory pool and the cluster link. Every sentence a reader sees about a pool
 * that cannot be changed comes from the backend (`reason`, `blocked_reason`,
 * `restart_reason`); this module only turns the measured figures into labels.
 */

export interface PoolStatus {
  supported: boolean;
  reason: string;
  can_change: boolean;
  blocked_reason: string;
  ram_bytes?: number;
  current_bytes?: number;
  default_bytes?: number;
  system_left_bytes?: number;
  override?: { state: "absent" | "vaelor" | "unrecognised"; bytes: number | null };
  min_gib?: number;
  max_gib?: number;
  /** `null` is "cannot be told", which is not the same as "no restart needed". */
  restart_pending?: boolean | null;
  restart_reason?: string;
  /**
   * What the next restart does to the running pool, and whether the owner
   * chose it (VD-133 1.5.5 #3). A restart is offered only when `chosen`:
   * a pool whose saved size an uninstall removed shrinks at the next
   * restart, and that restart is not the fix.
   */
  next_restart?: { change: "none" | "grow" | "shrink" | "unknown"; chosen: boolean; bytes: number | null };
  /** The whole-GiB size that saves the pool running now, when it is not saved; else `null`. */
  keep_gib?: number | null;
}

export interface PoolMachine {
  node_id: string;
  name: string;
  role: "controller" | "worker";
  /** When a worker's pool was last read (epoch seconds); `null` for the
   *  controller, which is read live, and for a worker never read. */
  checked_at: number | null;
  /** Non-empty when the backend calls a worker's reading too old to review a
   *  change against; the card then asks for a Recheck first. */
  stale_reason: string;
  pool: PoolStatus;
}

export type LinkKind = "wired" | "wireless" | "thunderbolt" | "unknown";

export interface NetworkLink {
  name: string;
  kind: LinkKind;
  state: string;
  speed_mbps: number | null;
  mtu: number | null;
  address: string;
  network: string;
  usable: boolean;
  reason: string;
  /**
   * B8: the backend's sentence when this link is the controller's shared LAN
   * card (how a split is fenced there, and that the LAN stays reachable), or
   * "" / absent for a link of its own. Shown in the confirmation as sent.
   */
  shared_note?: string;
}

export interface ChosenLink extends Partial<NetworkLink> {
  name: string;
  valid: boolean;
  reason: string;
}

export interface LinkMachine {
  node_id: string;
  name: string;
  /** `null`: not known yet (never read, or no link chosen). */
  on_link: boolean | null;
  address: string;
  interface: string;
  reason: string;
}

export interface HostSettings {
  gpu_memory_pool: { machines: PoolMachine[] };
  cluster_link: {
    available: boolean;
    reason: string;
    links: NetworkLink[];
    chosen: ChosenLink | null;
    machines: LinkMachine[];
    notes: string[];
  };
}

/**
 * The surface as the backend sends it, or `null` when the answer is not
 * that shape. The panel then says the settings could not be read; it must
 * never draw a half-understood answer as if every figure had been measured,
 * and it must never take the page around it down.
 */
export function readHostSettings(value: unknown): HostSettings | null {
  if (!value || typeof value !== "object") return null;
  const surface = value as Partial<HostSettings>;
  const link = surface.cluster_link;
  if (
    !Array.isArray(surface.gpu_memory_pool?.machines)
    || !link
    || !Array.isArray(link.links)
    || !Array.isArray(link.machines)
    || !Array.isArray(link.notes)
  ) {
    return null;
  }
  return surface as HostSettings;
}


/** The confirm tokens the backend checks; one spelling each, used by the panel. */
export const CONFIRM = {
  setPool: "set-gpu-memory-pool",
  revertPool: "revert-gpu-memory-pool",
  workerPool: "change-worker-gpu-memory",
  workerRestart: "reboot-worker-node",
  controllerRestart: "reboot-device",
  clusterLink: "set-cluster-link",
} as const;

/**
 * A pool or memory figure, read as the Fleet card reads it (W7-D5: one pool
 * was "47.2 GB" on Fleet and "44.0 GiB" here). GiB stays only the unit a new
 * size is typed in - formatTypedGib shows it beside its GB figure.
 */
export function poolBytes(value: number | null | undefined): string {
  return formatQuantity(value ?? undefined, "capacity");
}

/**
 * A pool SIZE in the two units the card uses together: "30.4 GiB (32.7 GB)".
 * VD-133 1.5.5 #3 review L7: one card read "About 32.7 GB", "about 30.4 GiB"
 * and "From 31 GiB (33.3 GB)" for sizes of the same pool. GiB is the unit a
 * size is typed and saved in; the GB beside it is the Fleet card's figure
 * (W7-D5). The backend's sentences use the same form (gpu_memory_pool._gib_text).
 */
export function poolSize(value: number | null | undefined): string {
  if (typeof value !== "number" || !Number.isFinite(value) || value < 0) return poolBytes(value);
  return `${gibFigure(value)} GiB (${poolBytes(value)})`;
}

/**
 * Bytes in GiB, whole only when whole: never "44.0" for a size just off 44
 * (review r3), the rule `gpu_memory_pool.gib_figure` follows.
 */
export function gibFigure(value: number): string {
  const gib = bytesIn(1, "GiB");
  if (value % gib === 0) return String(value / gib);
  for (const places of [1, 2, 3]) {
    const text = (value / gib).toFixed(places);
    if (!text.endsWith("0")) return text;
  }
  return `${value % gib < gib / 2 ? "just over" : "just under"} ${Math.round(value / gib)}`;
}

/**
 * A machine's name inside a sentence. The controller is listed as "This
 * controller", a name only at the start of a sentence; mid-sentence it is
 * "this controller" (W7-D5).
 */
export function machineInSentence(name: string): string {
  return /^This controller\b/.test(name) ? "t" + name.slice(1) : name;
}

/**
 * Whether the card's restart line is information, not a warning: a size is
 * saved and the machine runs another one. VD-161 amended (owner, 2026-10-20):
 * an install or a join saves the recommended size on a fresh machine, so this
 * is the first thing its card says, and it is no error ("Saved 45 GiB; it
 * counts from the next restart"). Only an unsaved pool warns.
 */
export function poolSavedForRestart(pool: PoolStatus): boolean {
  return pool.supported && pool.restart_pending === true && pool.override?.state === "vaelor";
}

/** The pill for one machine's pool: what is true now, in one or two words. */
export function poolPill(pool: PoolStatus): { tone: StatusTone; label: string } {
  if (!pool.supported) return { tone: "neutral", label: "Not available" };
  if (poolSavedForRestart(pool)) {
    // Still the owner's size, so information - but a smaller one says so.
    return { tone: "info", label: pool.next_restart?.change === "shrink" ? "Smaller at restart" : "Counts at restart" };
  }
  if (pool.restart_pending === true) {
    // VD-133 1.5.5 #3: "Restart needed" only when the restart moves the pool
    // to a size the owner chose; an unsaved pool says what a restart does.
    if (pool.next_restart?.chosen === true) return { tone: "warning", label: "Restart needed" };
    if (pool.next_restart?.change === "shrink") return { tone: "warning", label: "Shrinks at restart" };
    return { tone: "warning", label: "Not saved" };
  }
  if (pool.restart_pending === null || pool.restart_pending === undefined) {
    return { tone: "warning", label: "Check needed" };
  }
  if (pool.override?.state === "vaelor") return { tone: "success", label: "Set" };
  return { tone: "neutral", label: "Kernel default" };
}

/**
 * The size a reader typed, as a whole number of GiB inside the machine's
 * range, or the sentence saying why it is not one. The backend re-checks; this
 * only keeps an impossible value from being reviewed at all.
 */
export function parsePoolSize(
  text: string, pool: PoolStatus,
): { size: number | null; error: string } {
  const min = pool.min_gib ?? 0;
  const max = pool.max_gib ?? 0;
  const trimmed = text.trim();
  if (!trimmed) return { size: null, error: "" };
  if (!/^\d+$/.test(trimmed)) {
    return { size: null, error: `Enter a whole number from ${min} to ${max}.` };
  }
  const size = Number(trimmed);
  if (size < min || size > max) {
    return { size: null, error: `Enter a whole number from ${min} to ${max}.` };
  }
  return { size, error: "" };
}

const KIND_LABELS: Record<LinkKind, string> = {
  wired: "Wired",
  wireless: "Wi-Fi",
  thunderbolt: "Thunderbolt / USB4",
  unknown: "Type not reported",
};

export function linkKindLabel(kind: LinkKind): string {
  return KIND_LABELS[kind] ?? KIND_LABELS.unknown;
}

/** A link's negotiated speed, or that it was not reported - never a guess. */
export function linkSpeedLabel(speed: number | null): string {
  if (typeof speed !== "number" || !Number.isFinite(speed) || speed <= 0) {
    return "Speed not reported";
  }
  if (speed >= 1000) {
    const gigabits = speed / 1000;
    return `${Number.isInteger(gigabits) ? gigabits : gigabits.toFixed(1)} Gb/s`;
  }
  return `${speed} Mb/s`;
}

/**
 * What a split's "Link between machines" says when a cluster link is chosen
 * (ACC-208): that link, and whether each selected worker is on it. A split's
 * traffic goes over the chosen link, not the LAN each machine enrolled on, so
 * the enrolled connection is the wrong thing to show. `null` when no valid
 * link is chosen (or the settings could not be read): the caller then shows
 * the enrolled connections, which is what a split uses in that case.
 */
export function splitLinkLine(
  link: HostSettings["cluster_link"] | null | undefined,
  selected: ReadonlyArray<{ id: string; name: string }>,
): string | null {
  const chosen = link?.available ? link.chosen : null;
  if (!link || !chosen?.valid) return null;
  const head = `${chosen.name} \u00b7 ${linkKindLabel(chosen.kind ?? "unknown")} \u00b7 ${linkSpeedLabel(chosen.speed_mbps ?? null)}, the chosen cluster link.`;
  const workers = selected.flatMap((node) => {
    const machine = link.machines.find((entry) => entry.node_id === node.id);
    if (!machine) return [];
    return [machine.on_link === true
      ? `${machine.name} is on it as ${machine.address}.`
      : machine.reason || `Whether ${machine.name} is on it is not known yet.`];
  });
  return [head, ...workers].join(" ");
}

/** One line describing a link in the picker and the list. */
export function linkSummary(link: Pick<NetworkLink, "name" | "kind" | "speed_mbps">): string {
  return [link.name, linkKindLabel(link.kind), linkSpeedLabel(link.speed_mbps)].join(" · ");
}
