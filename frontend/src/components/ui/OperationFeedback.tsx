import type { HTMLAttributes, ReactNode } from "react";
import { operationTone, type OperationState } from "./status";
import { joinClassNames } from "./field";

export type { OperationState };

export type OperationFeedbackProps = {
  state: OperationState;
  message?: ReactNode;
  className?: string;
} & Omit<HTMLAttributes<HTMLDivElement>, "role" | "aria-live" | "aria-busy" | "children">;

export function OperationFeedback({
  state,
  message,
  className,
  ...props
}: OperationFeedbackProps) {
  const urgent = state === "warning" || state === "error";
  const empty = state === "idle" && !message;
  return (
    <div
      {...props}
      aria-atomic="true"
      aria-busy={state === "pending" || undefined}
      aria-hidden={empty || undefined}
      aria-live={urgent ? "assertive" : "polite"}
      className={joinClassNames(
        "operation-feedback",
        "ui-operation-feedback",
        "ui-operation-feedback--" + operationTone(state),
        className,
      )}
      data-state={state}
      role={urgent ? "alert" : "status"}
    >
      {message && <span className="ui-operation-feedback__message">{message}</span>}
    </div>
  );
}
