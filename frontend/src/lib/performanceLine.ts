/**
 * The compact per-answer timing an assistant message can carry.
 *
 * The backend folds two serving stacks (FastFlowLM's `usage`, llama.cpp's
 * `timings`) plus its own wall-clock into this one shape, and omits any field
 * it could not measure. A hosted provider that reports no timings sends nothing
 * here at all, so the reader sees no line rather than an empty one.
 */
export interface PerformanceSummary {
  total_seconds?: number;
  ttft_seconds?: number;
  prefill_tps?: number;
  decode_tps?: number;
}

function seconds(value: number): string {
  // Sub-second times are the interesting ones for TTFT, so keep two digits
  // below a second, one below ten, and none above - "0.78s", "2.7s", "42s".
  const digits = value < 1 ? 2 : value < 10 ? 1 : 0;
  return `${value.toFixed(digits)}s`;
}

/**
 * A one-line performance summary, or "" when there is nothing to show.
 *
 * Each present field becomes one middot-separated clause, in the fixed order
 * total, TTFT, prefill, decode - e.g.
 * `total 2.7s · TTFT 0.78s · prefill 49 tok/s · decode 14 tok/s`.
 */
export function performanceLine(performance?: PerformanceSummary | null): string {
  if (!performance) return "";
  const parts: string[] = [];
  if (typeof performance.total_seconds === "number") {
    parts.push(`total ${seconds(performance.total_seconds)}`);
  }
  if (typeof performance.ttft_seconds === "number") {
    parts.push(`TTFT ${seconds(performance.ttft_seconds)}`);
  }
  if (typeof performance.prefill_tps === "number") {
    parts.push(`prefill ${Math.round(performance.prefill_tps)} tok/s`);
  }
  if (typeof performance.decode_tps === "number") {
    parts.push(`decode ${Math.round(performance.decode_tps)} tok/s`);
  }
  return parts.join(" · ");
}
