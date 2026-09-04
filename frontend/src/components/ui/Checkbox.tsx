import type { InputHTMLAttributes } from "react";
import {
  FieldMessages,
  joinClassNames,
  useFieldSupport,
  type FieldAriaProps,
  type FieldProps,
} from "./field";

export type CheckboxProps = FieldProps & Omit<
  InputHTMLAttributes<HTMLInputElement>,
  keyof FieldProps | "type" | "aria-describedby" | "aria-invalid"
> & FieldAriaProps & {
  className?: string;
};

export function Checkbox({
  id,
  label,
  hint,
  error,
  disabledReason,
  className,
  disabled = false,
  "aria-describedby": ariaDescribedBy,
  "aria-invalid": ariaInvalid,
  ...props
}: CheckboxProps) {
  const field = useFieldSupport(id, { hint, error, disabledReason, describedBy: ariaDescribedBy });
  return (
    <div className="ui-field ui-field--checkbox">
      <label className="ui-checkbox" htmlFor={field.controlId}>
        <input
          {...props}
          aria-describedby={field.describedBy || undefined}
          aria-invalid={error ? true : ariaInvalid}
          className={joinClassNames("ui-control", "ui-control--checkbox", className)}
          disabled={disabled}
          id={field.controlId}
          type="checkbox"
        />
        <span className="ui-checkbox__label">{label}</span>
      </label>
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
