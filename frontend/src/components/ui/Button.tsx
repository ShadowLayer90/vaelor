import { forwardRef, useId, type ButtonHTMLAttributes, type ReactNode } from "react";
import { joinClassNames, joinDescribedBy, type FieldAriaProps } from "./field";

export type ButtonVariant = "primary" | "secondary" | "quiet" | "danger";

/**
 * Where a disabled button's reason is drawn. The reason is always visible text
 * (owner rule): this only decides whether it takes room in the button's row.
 *
 * - `stacked` (default): under the button, in flow. Right in a form or a card,
 *   where the content below can move down to make room.
 * - `detached`: hung under the button, out of flow, so the button lines up with
 *   its neighbours and the row is no wider than without it. For a toolbar or
 *   a top bar, where an in-flow reason lifted the button off its row and
 *   pushed the bar past the window (Power, viewer top bar, 768 px: 84 px).
 */
export type DisabledReasonLayout = "stacked" | "detached";

export type ButtonProps = Omit<
  ButtonHTMLAttributes<HTMLButtonElement>,
  "className" | "disabled" | "aria-describedby" | "children"
> & Pick<FieldAriaProps, "aria-describedby"> & {
  children: ReactNode;
  variant?: ButtonVariant;
  busy?: boolean;
  disabledReason?: ReactNode;
  disabledReasonLayout?: DisabledReasonLayout;
  disabled?: boolean;
  className?: string;
};

export const Button = forwardRef<HTMLButtonElement, ButtonProps>(function Button({
  children,
  variant = "secondary",
  busy = false,
  disabled = false,
  disabledReason,
  disabledReasonLayout = "stacked",
  className,
  type = "button",
  "aria-describedby": ariaDescribedBy,
  ...props
}: ButtonProps, ref) {
  const generatedId = useId().replaceAll(":", "");
  const isDisabled = disabled || busy || Boolean(disabledReason);
  const reasonId = disabledReason ? "ui-button-" + generatedId + "-disabled-reason" : undefined;
  const showReason = Boolean(disabledReason) && isDisabled;
  return (
    <span
      className={joinClassNames(
        "ui-button-wrap",
        showReason && disabledReasonLayout === "detached" && "ui-button-wrap--reason-detached",
      )}
    >
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
        {busy && <span aria-live="polite" className="sr-only">Working…</span>}
        {children}
      </button>
      {showReason && (
        <span className="ui-button__disabled-reason" id={reasonId}>{disabledReason}</span>
      )}
    </span>
  );
});
