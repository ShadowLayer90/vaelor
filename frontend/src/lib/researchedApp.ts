/**
 * Pure helpers that turn an approved researched draft into the per-service facts
 * the cluster placement step needs (D4d), mirroring the backend renderer
 * (`vaelor/cluster_app_manifest.render_app_services`) exactly so the modal
 * offers the same choice the deploy will make.
 *
 * A service is STATEFUL when its normalized compose entry keeps a named volume
 * (`type: "volume"`) — the same signal `_named_volumes` reads — so it must pin to
 * one data node and can never spread. A stateless service follows the app-level
 * intent. The per-service memory defaults to the manifest's app-level figure; the
 * backend flags that figure as the 512 MB default when research determined no
 * footprint, and this mirrors the same constant so the modal can surface the
 * honest "using the default — set a limit" note rather than a guessed number.
 */

import { typedFromBytes } from "./format";

/**
 * The bytes `application_deployments.normalize_manifest` writes for
 * ``resources.memory_bytes`` when research found no footprint — 512 MiB. Mirrored
 * from `cluster_app_manifest.MANIFEST_DEFAULT_MEMORY_BYTES`; a test on each side
 * would fail if the two drift.
 */
export const MANIFEST_DEFAULT_MEMORY_BYTES = 536870912;


/** One manifest service as the placement step reasons about it. */
export interface PlacementService {
  /** The compose service name (and overlay-network alias), e.g. `web` or `db`. */
  key: string;
  /** Keeps a named volume, so it pins to one node and cannot spread. */
  stateful: boolean;
  /** The app-level memory figure in MiB every service defaults to. */
  defaultMemoryMib: number;
  /**
   * True when that figure is the 512 MB manifest default (no researched
   * footprint), so the modal surfaces the honest "set a limit" note.
   */
  memoryDefaulted: boolean;
}

/** The per-service placement input the deploy plan/job consumes. */
export interface PlacementInput {
  intent?: "run-once" | "spread";
  pin_node?: string;
  memory_mib?: number;
  cpu?: number;
  label_constraints?: Array<{ key: string; op: "==" | "!="; value: string }>;
}

function isRecord(value: unknown): value is Record<string, unknown> {
  return Boolean(value) && typeof value === "object" && !Array.isArray(value);
}

/** True when a normalized compose service keeps at least one named volume. */
function serviceIsStateful(service: unknown): boolean {
  if (!isRecord(service) || !Array.isArray(service.volumes)) return false;
  return service.volumes.some(
    (volume) => isRecord(volume) && volume.type === "volume",
  );
}

/**
 * The manifest services in compose order, each with its stateful flag and the
 * defaulted per-service memory figure. Returns an empty list for a draft whose
 * compose is missing or has no services, so a caller renders "nothing to place"
 * honestly rather than throwing.
 */
export function researchedPlacementServices(
  compose: Record<string, unknown> | null | undefined,
  memoryBytes: number | null | undefined,
): PlacementService[] {
  const services = isRecord(compose) && isRecord(compose.services) ? compose.services : null;
  if (!services) return [];
  const bytes = typeof memoryBytes === "number" && memoryBytes > 0
    ? memoryBytes
    : MANIFEST_DEFAULT_MEMORY_BYTES;
  const defaultMemoryMib = Math.max(1, typedFromBytes(bytes, "MiB"));
  const memoryDefaulted = bytes === MANIFEST_DEFAULT_MEMORY_BYTES;
  return Object.keys(services).map((key) => ({
    key,
    stateful: serviceIsStateful(services[key]),
    defaultMemoryMib,
    memoryDefaulted,
  }));
}
