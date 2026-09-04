import type { ReactNode } from "react";
import { Button, UnavailableValue } from "./ui";
import type { IconName } from "./Icon";
import { Icon } from "./Icon";
import { Sparkline } from "./Sparkline";

export function MetricCard({
  icon,
  label,
  value,
  detail,
  values,
  tone,
  unavailableReason,
  actionLabel,
  actionDescription,
  onAction,
}: {
  icon: IconName;
  label: string;
  value: string;
  /**
   * A node, so a detail line whose components are partly unmeasured can carry
   * an `UnavailableValue` for the missing part instead of a substituted zero.
   */
  detail: ReactNode;
  values: Array<number | null>;
  tone: "blue" | "green" | "amber" | "pink";
  /**
   * Why this machine cannot produce the reading. When present the card shows
   * `UnavailableValue` and its explanation instead of a number, because a card
   * that renders "0%" for a sensor that does not exist is a measurement claim.
   */
  unavailableReason?: string | null;
  actionLabel?: string;
  /**
   * What the action will do, before it does it. A bare verb sitting on a chart
   * gives the reader no way to predict the outcome, so the description is
   * carried into the accessible name and the tooltip.
   */
  actionDescription?: string;
  onAction?: () => void;
}) {
  const measuredCount = values.filter((entry) => entry !== null).length;
  return (
    <article className={`metric-card metric-card--${tone}`}>
      <header className="metric-card__header">
        <span className="metric-card__index" aria-hidden="true">
          <Icon name={icon} size={16} />
        </span>
        <p>{label}</p>
        <span className="metric-card__sample-count">
          <span aria-hidden="true" />
          {unavailableReason
            ? "No samples"
            : `${Math.max(measuredCount, 1).toString().padStart(2, "0")} samples`}
        </span>
      </header>
      <div className="metric-card__reading">
        {unavailableReason ? (
          <strong className="metric-card__live-value">
            <UnavailableValue label={`${label} unavailable`} reason={unavailableReason} />
          </strong>
        ) : (
          <strong className="metric-card__live-value" key={value}>{value}</strong>
        )}
        <span>{unavailableReason ?? detail}</span>
      </div>
      <Sparkline values={unavailableReason ? [] : values} tone={tone} />
      {actionLabel && onAction && (
        <Button
          aria-label={actionDescription ? `${actionLabel} — ${actionDescription}` : actionLabel}
          className="metric-card__action"
          onClick={onAction}
          title={actionDescription ?? actionLabel}
          type="button"
        >
          {actionLabel}
        </Button>
      )}
    </article>
  );
}
