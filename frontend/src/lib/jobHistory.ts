/**
 * The fields of a job-ledger row that the workload activity list reads
 * (`workloadActivity.ts`). The list's own folding of retries lives there and
 * reads the server's projection; this file keeps only the shared row shape.
 */
export interface JobHistoryItem {
  id: string;
  type: string;
  state: string;
  created_at: number;
  payload?: Record<string, unknown>;
  retry_of?: string | null;
  operation_state?: string;
  attention?: boolean;
  retryable?: boolean;
  readiness?: string;
  liveness?: string;
  resolved_by_retry?: boolean;
  retry_depth?: number;
}
