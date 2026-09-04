import type { TextareaHTMLAttributes } from "react";
import {
  FieldCounter,
  FieldLabel,
  FieldMessages,
  joinClassNames,
  shouldCount,
  useFieldSupport,
  type FieldAriaProps,
  type FieldProps,
} from "./field";

export type TextareaProps = FieldProps & Omit<
  TextareaHTMLAttributes<HTMLTextAreaElement>,
  keyof FieldProps | "aria-describedby" | "aria-invalid"
> & FieldAriaProps & {
  className?: string;
};

export function Textarea({
  id,
  label,
  hint,
  error,
  disabledReason,
  className,
  disabled = false,
  showCount,
  "aria-describedby": ariaDescribedBy,
  "aria-invalid": ariaInvalid,
  ...props
}: TextareaProps) {
  const counting = shouldCount(showCount, props.maxLength);
  const field = useFieldSupport(id, {
    hint, error, disabledReason, describedBy: ariaDescribedBy, counter: counting,
  });
  return (
    <div className="ui-field ui-field--textarea">
      <FieldLabel htmlFor={field.controlId}>{label}</FieldLabel>
      <textarea
        {...props}
        aria-describedby={field.describedBy || undefined}
        aria-invalid={error ? true : ariaInvalid}
        className={joinClassNames("ui-control", "ui-control--textarea", className)}
        disabled={disabled}
        id={field.controlId}
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
      {counting && (
        <FieldCounter
          id={field.counterId}
          limit={Number(props.maxLength)}
          used={String(props.value ?? "").length}
        />
      )}
    </div>
  );
}
