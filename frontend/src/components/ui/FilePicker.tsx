import type { InputHTMLAttributes } from "react";
import {
  FieldLabel,
  FieldMessages,
  joinClassNames,
  useFieldSupport,
  type FieldAriaProps,
  type FieldProps,
} from "./field";

export type FilePickerProps = FieldProps & Omit<
  InputHTMLAttributes<HTMLInputElement>,
  keyof FieldProps | "type" | "value" | "defaultValue" | "aria-describedby" | "aria-invalid"
> & FieldAriaProps & {
  className?: string;
};

export function FilePicker({
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
}: FilePickerProps) {
  const field = useFieldSupport(id, { hint, error, disabledReason, describedBy: ariaDescribedBy });
  return (
    <div className="ui-field ui-field--file-picker">
      <FieldLabel htmlFor={field.controlId}>{label}</FieldLabel>
      <input
        {...props}
        aria-describedby={field.describedBy || undefined}
        aria-invalid={error ? true : ariaInvalid}
        className={joinClassNames("ui-control", "ui-control--file-picker", className)}
        disabled={disabled}
        id={field.controlId}
        type="file"
      />
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
