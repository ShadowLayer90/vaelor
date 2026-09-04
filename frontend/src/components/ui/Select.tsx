import type { ReactNode, SelectHTMLAttributes } from "react";
import {
  FieldLabel,
  FieldMessages,
  joinClassNames,
  useFieldSupport,
  type FieldAriaProps,
  type FieldProps,
} from "./field";

export type SelectProps = FieldProps & Omit<
  SelectHTMLAttributes<HTMLSelectElement>,
  keyof FieldProps | "children" | "aria-describedby" | "aria-invalid"
> & FieldAriaProps & {
  children: ReactNode;
  className?: string;
};

export function Select({
  id,
  label,
  hint,
  error,
  disabledReason,
  className,
  disabled = false,
  children,
  "aria-describedby": ariaDescribedBy,
  "aria-invalid": ariaInvalid,
  ...props
}: SelectProps) {
  const field = useFieldSupport(id, { hint, error, disabledReason, describedBy: ariaDescribedBy });
  return (
    <div className="ui-field ui-field--select">
      <FieldLabel htmlFor={field.controlId}>{label}</FieldLabel>
      <select
        {...props}
        aria-describedby={field.describedBy || undefined}
        aria-invalid={error ? true : ariaInvalid}
        className={joinClassNames("ui-control", "ui-control--select", className)}
        disabled={disabled}
        id={field.controlId}
      >
        {children}
      </select>
      <FieldMessages
        disabled={disabled}
        disabledReason={disabledReason}
        disabledReasonId={field.disabledReasonId}
        error={error}
        errorId={field.errorId}
        hint={hint}
        hintId={field.hintId}
      />
    </div>
  );
}
