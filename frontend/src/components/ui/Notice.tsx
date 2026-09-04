import type { HTMLAttributes, ReactNode } from "react";
import { joinClassNames } from "./field";

export type NoticeSeverity = "info" | "success" | "warning" | "danger";

export type NoticeProps = {
  severity: NoticeSeverity;
  children: ReactNode;
  heading?: ReactNode;
  className?: string;
} & Omit<HTMLAttributes<HTMLDivElement>, "role" | "aria-live" | "children">;

export function Notice({
  severity,
  children,
  heading,
  className,
  ...props
}: NoticeProps) {
  const urgent = severity === "warning" || severity === "danger";
  return (
    <div
      {...props}
      aria-atomic="true"
      aria-live={urgent ? "assertive" : "polite"}
      className={joinClassNames("notice", "ui-notice", "ui-notice--" + severity, className)}
      data-severity={severity}
      role={urgent ? "alert" : "status"}
    >
      <div className="ui-notice__content">
        {heading && <strong className="ui-notice__heading">{heading}</strong>}
        <span>{children}</span>
      </div>
    </div>
  );
}
