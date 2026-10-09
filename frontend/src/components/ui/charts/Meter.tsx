import { joinClassNames } from "../field";
import { DEFAULT_CHART_LABELS, type ChartThreshold } from "./types";

export interface MeterProps {
  /** The accessible name: what is being measured. */
  label: string;
  value: number;
  /** The value as the caller formats it, for example "71 C". */
  valueText: string;
  min?: number;
  max: number;
  /** Marks the backend sent, such as this machine's warning and critical limits. */
  marks?: ChartThreshold[];
  /** Where the marks come from, in the backend's words. */
  marksSource?: string;
  /**
   * "below" prints the marks and their source under the bar. "hidden" keeps
   * that sentence for assistive technology and the bar's `title` only, for a
   * caller (the stat tile) whose own caption has no line to spare.
   */
  caption?: "below" | "hidden";
  marksLabel?: string;
  className?: string;
}

/** Where `value` sits on the bar, 0..100, never outside it. */
function percentAlong(value: number, min: number, max: number): number {
  if (!(max > min) || !Number.isFinite(value)) return 0;
  return Math.min(100, Math.max(0, ((value - min) / (max - min)) * 100));
}

/**
 * A 0..max meter. The fill is one neutral colour: the primitive holds no
 * threshold, so it does not decide that a reading is a warning. The marks
 * are where the backend's limits sit, and their words are always available
 * as text, never as colour alone.
 */
export function Meter({
  label,
  value,
  valueText,
  min = 0,
  max,
  marks = [],
  marksSource,
  caption = "below",
  marksLabel = DEFAULT_CHART_LABELS.marks,
  className,
}: MeterProps) {
  const marksText = marks.length
    ? marksLabel + ": " + marks.map((mark) => mark.label).join(", ") + (marksSource ? " (" + marksSource + ")" : "")
    : "";
  const clamped = Math.min(max, Math.max(min, Number.isFinite(value) ? value : min));
  return (
    <div className={joinClassNames("chart-meter", className)}>
      <div
        aria-label={label}
        aria-valuemax={max}
        aria-valuemin={min}
        aria-valuenow={clamped}
        aria-valuetext={marksText ? valueText + "; " + marksText : valueText}
        className="chart-meter__bar"
        role="meter"
        title={marksText || undefined}
      >
        <svg aria-hidden="true" className="chart-meter__svg" focusable="false">
          <rect className="chart-meter__track" height="100%" rx="4" width="100%" />
          <rect className="chart-meter__fill" height="100%" rx="4" width={percentAlong(value, min, max) + "%"} />
          {marks.filter((mark) => mark.value >= min && mark.value <= max).map((mark) => {
            const x = percentAlong(mark.value, min, max) + "%";
            return (
              <g data-mark={mark.label} key={mark.label}>
                <line className="chart-meter__mark-ring" x1={x} x2={x} y1="-3" y2="11" />
                <line className="chart-meter__mark" x1={x} x2={x} y1="-3" y2="11" />
              </g>
            );
          })}
        </svg>
      </div>
      {marksText && (
        <p className={caption === "below" ? "chart-meter__caption" : "sr-only"}>{marksText}</p>
      )}
    </div>
  );
}
