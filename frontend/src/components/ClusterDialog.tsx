import { useId, type ReactNode, type RefObject } from "react";
import { ModalShell } from "./ModalShell";
import { Button } from "./ui/Button";
import { ModalError } from "./ui/ModalError";
import { joinClassNames } from "./ui/field";
import "../styles/cluster.css";

export type DialogTone = "default" | "accent" | "danger" | "warning";

/**
 * The boards' dialog: an eyebrow over a title, an optional Close in the
 * header, a body that scrolls inside the dialog, the refusal of the dialog's
 * own action at the foot of the body (never on the page beneath), and the
 * actions in a fixed footer. `standard` is 440 px, `wide` 720 px.
 */
export function ClusterDialog({
  aside,
  children,
  className,
  closeLabel,
  describedBy,
  error,
  eyebrow,
  footer,
  initialFocusRef,
  onClose,
  role,
  size = "standard",
  subtitle,
  title,
  titleId,
  tone = "default",
}: {
  /** What sits at the header's right end (a status pill). */
  aside?: ReactNode;
  children?: ReactNode;
  className?: string;
  /** A text Close in the header (the agent and run dialogs); omitted when the footer closes it. */
  closeLabel?: string;
  describedBy?: string;
  error?: ReactNode;
  eyebrow?: ReactNode;
  footer?: ReactNode;
  initialFocusRef?: RefObject<HTMLElement | null>;
  onClose: () => void;
  /** `alertdialog` when the dialog is a confirmation (a restart, a link change). */
  role?: "dialog" | "alertdialog";
  size?: "standard" | "wide";
  /** A quiet line under the title (an image name). */
  subtitle?: ReactNode;
  title: ReactNode;
  titleId?: string;
  tone?: DialogTone;
}) {
  const generated = "cl-dialog-" + useId().replaceAll(":", "") + "-title";
  const headingId = titleId ?? generated;
  return (
    <ModalShell
      className={joinClassNames("cl-dialog", "cl-dialog--" + size, className)}
      describedBy={describedBy}
      initialFocusRef={initialFocusRef}
      labelledBy={headingId}
      onClose={onClose}
      role={role}
    >
      <header className="cl-dialog__header">
        <div className="cl-dialog__titles">
          {eyebrow && <span className={joinClassNames("cl-eyebrow", tone !== "default" && "cl-eyebrow--" + tone)}>{eyebrow}</span>}
          <h2 id={headingId}>{title}</h2>
          {subtitle && <p className="cl-dialog__subtitle">{subtitle}</p>}
        </div>
        {aside && <div className="cl-dialog__aside">{aside}</div>}
        {closeLabel && (
          <Button className="cl-dialog__close" onClick={onClose} variant="quiet">{closeLabel}</Button>
        )}
      </header>
      <div className="cl-dialog__body">
        {children}
        <ModalError error={error} />
      </div>
      {footer && <footer className="cl-dialog__footer">{footer}</footer>}
    </ModalShell>
  );
}

