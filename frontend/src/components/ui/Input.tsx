import { forwardRef, type InputHTMLAttributes } from "react";
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

export type InputProps = FieldProps & Omit<
  InputHTMLAttributes<HTMLInputElement>,
  keyof FieldProps | "aria-describedby" | "aria-invalid"
> & FieldAriaProps & {
  className?: string;
};

export const Input = forwardRef<HTMLInputElement, InputProps>(function Input({
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
}: InputProps, ref) {
  // Never count a masked value by default: a live length readout under a
  // password or token discloses exactly how long the secret is.
  const counting = props.type === "password"
    ? showCount === true
    : shouldCount(showCount, props.maxLength);
  const field = useFieldSupport(id, {
    hint, error, disabledReason, describedBy: ariaDescribedBy, counter: counting,
  });
  return (
    <div className="ui-field ui-field--input">
      <FieldLabel htmlFor={field.controlId}>{label}</FieldLabel>
      <input
        {...props}
        ref={ref}
        aria-describedby={field.describedBy || undefined}
        aria-invalid={error ? true : ariaInvalid}
        className={joinClassNames("ui-control", "ui-control--input", className)}
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
});
