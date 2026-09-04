/**
 * Telemetry history retention, made visible to the owner (#186 / VD-089).
 *
 * The store applies 7 days while the appliance configuration asks 30. That
 * override was logged only at boot and appeared on no surface, so an owner
 * reading their own 30 had no way to learn 7 was in force. `/api/v2/telemetry/
 * retention` now serves the configured period beside the applied one and the
 * retention state; this decides when the owner needs to be told and reuses the
 * server's own sentence rather than writing a second one that could drift from
 * it (LESSONS 6).
 */
export interface RetentionStatus {
  /** One of "starting" | "off" | "failed" | "running". */
  state: string;
  applied_retention_days: number;
  /** `null` until the configuration has been read at least once. */
  configured_retention_days: number | null;
  override_active: boolean;
  reason: string;
  failure?: string | null;
}

/**
 * A one-line honest note about retention, or "" when there is nothing the owner
 * needs to act on. The wording is the server's, so the two cannot disagree.
 *
 * - `running` with no override: the healthy case, say nothing.
 * - `starting`: transient every restart, not worth a line; ask again shortly.
 * - `running` with an override, `off`, or `failed`: surface the reason.
 */
export function retentionNote(status: RetentionStatus | null | undefined): string {
  if (!status) return "";
  if (status.state === "starting") return "";
  if (status.state === "running" && !status.override_active) return "";
  return status.reason ?? "";
}
