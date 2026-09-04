import { forwardRef, useId, type ButtonHTMLAttributes, type ReactNode } from "react";
import { joinClassNames, joinDescribedBy, type FieldAriaProps } from "./field";

export type ButtonVariant = "primary" | "secondary" | "quiet" | "danger";

export type ButtonProps = Omit<
  ButtonHTMLAttributes<HTMLButtonElement>,
  "className" | "disabled" | "aria-describedby" | "children"
> & Pick<FieldAriaProps, "aria-describedby"> & {
  children: ReactNode;
  variant?: ButtonVariant;
  busy?: boolean;
  disabledReason?: ReactNode;
  disabled?: boolean;
  className?: string;
};

export const Button = forwardRef<HTMLButtonElement, ButtonProps>(function Button({
  children,
  variant = "secondary",
  busy = false,
  disabled = false,
  disabledReason,
  className,
  type = "button",
  "aria-describedby": ariaDescribedBy,
  ...props
}: ButtonProps, ref) {
  const generatedId = useId().replaceAll(":", "");
  const isDisabled = disabled || busy || Boolean(disabledReason);
  const reasonId = disabledReason ? "ui-button-" + generatedId + "-disabled-reason" : undefined;
  return (
    <span className="ui-button-wrap">
      <button
        {...props}
        ref={ref}
        aria-busy={busy || undefined}
        aria-describedby={joinDescribedBy(ariaDescribedBy, reasonId)}
        aria-disabled={isDisabled || undefined}
        className={joinClassNames(
          "button",
          "ui-button",
          "ui-button--" + variant,
          className,
        )}
        disabled={isDisabled}
        type={type}
      >
        {busy && <span aria-live="polite" className="sr-only">Working?</span>}
        {children}
      </button>
      {disabledReason && isDisabled && (
        <span className="ui-button__disabled-reason" id={reasonId}>{disabledReason}</span>
      )}
    </span>
  );
});
