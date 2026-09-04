import { statusLabel, statusTone, type LegacyStatus, type StatusTone } from "./ui/status";

export function StatusPill({
  status,
  tone,
  label,
  className,
}: {
  status?: LegacyStatus;
  tone?: StatusTone;
  label?: string;
  className?: string;
}) {
  const canonicalTone = tone ?? statusTone(status);
  const text = label ?? statusLabel(canonicalTone);
  return (
    <span
      className={["status-pill", "status-pill--" + canonicalTone, className].filter(Boolean).join(" ")}
      data-status-tone={canonicalTone}
    >
      <span className="status-pill__dot" aria-hidden="true" />
      {text}
    </span>
  );
}
