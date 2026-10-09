import { statusLabel, statusTone, type LegacyStatus, type StatusTone } from "./ui/status";

export function StatusPill({
  status,
  tone,
  label,
  className,
  description,
  reading,
}: {
  status?: LegacyStatus;
  tone?: StatusTone;
  label?: string;
  className?: string;
  /**
   * Why the pill says what it says, in a sentence. Shown on hover and read by
   * a screen reader after the label; a `title` alone reaches neither a
   * keyboard nor a touch user nor, reliably, assistive technology.
   */
  description?: string;
  /**
   * A value that was not read now: `unread` (nothing was measured) or `stale`
   * (the last good reading, old). Either one is grey whatever tone was asked
   * for, because green means only "read just now and fine" - the label must
   * say which in words ("Not read", "22 min old").
   */
  reading?: "unread" | "stale";
}) {
  const canonicalTone = reading ? "neutral" : tone ?? statusTone(status);
  const text = label ?? statusLabel(canonicalTone);
  return (
    <span
      className={["status-pill", "status-pill--" + canonicalTone, reading && "status-pill--" + reading, className]
        .filter(Boolean).join(" ")}
      data-status-tone={canonicalTone}
      title={description}
    >
      <span className="status-pill__dot" aria-hidden="true" />
      {text}
      {description && <span className="sr-only">{`. ${description}`}</span>}
    </span>
  );
}
