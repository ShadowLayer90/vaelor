import { useId, type AriaAttributes, type ReactNode } from "react";

export type FieldAriaProps = Pick<AriaAttributes, "aria-describedby" | "aria-invalid">;

export type FieldProps = {
  id?: string;
  label: ReactNode;
  hint?: ReactNode;
  error?: ReactNode;
  disabledReason?: ReactNode;
  /**
   * Show a live "used / limit" counter. Defaults on for any field with a
   * `maxLength` at or above COUNTER_THRESHOLD, because the browser truncates a
   * paste at that limit silently and the user is otherwise never told that
   * text was dropped.
   */
  showCount?: boolean;
};

/** Fields able to hold prose need a visible limit; short codes do not. */
export const COUNTER_THRESHOLD = 200;

type FieldSupportOptions = Pick<FieldProps, "hint" | "error" | "disabledReason"> & {
  describedBy?: string;
  counter?: boolean;
};

export function useFieldSupport(id: string | undefined, options: FieldSupportOptions) {
  const generatedId = useId().replaceAll(":", "");
  const controlId = id ?? "ui-field-" + generatedId;
  const hintId = options.hint ? controlId + "-hint" : undefined;
  const errorId = options.error ? controlId + "-error" : undefined;
  const disabledReasonId = options.disabledReason ? controlId + "-disabled-reason" : undefined;
  const counterId = options.counter ? controlId + "-counter" : undefined;
  const describedBy = [options.describedBy, hintId, errorId, disabledReasonId, counterId]
    .filter(Boolean)
    .join(" ");

  return { controlId, hintId, errorId, disabledReasonId, counterId, describedBy };
}

export function shouldCount(
  showCount: boolean | undefined,
  maxLength: number | string | undefined,
) {
  const limit = Number(maxLength);
  if (!Number.isFinite(limit) || limit <= 0) return false;
  return showCount ?? limit >= COUNTER_THRESHOLD;
}

export function FieldCounter({
  id,
  used,
  limit,
}: {
  id?: string;
  used: number;
  limit: number;
}) {
  const remaining = limit - used;
  // Only nag near the limit; a counter that shouts from character one is noise.
  const nearLimit = remaining <= Math.max(20, Math.round(limit * 0.1));
  return (
    <p
      className={nearLimit ? "ui-field__counter is-near-limit" : "ui-field__counter"}
      id={id}
    >
      {used.toLocaleString()} / {limit.toLocaleString()} characters
      {remaining <= 0 ? " · limit reached, further typing is ignored" : ""}
    </p>
  );
}

export function FieldLabel({ htmlFor, children }: { htmlFor: string; children: ReactNode }) {
  return <label className="ui-field__label" htmlFor={htmlFor}>{children}</label>;
}

export function FieldMessages({
  hint,
  error,
  disabled,
  disabledReason,
  hintId,
  errorId,
  disabledReasonId,
}: Pick<FieldProps, "hint" | "error" | "disabledReason"> & {
  disabled: boolean;
  hintId?: string;
  errorId?: string;
  disabledReasonId?: string;
}) {
  return (
    <>
      {hint && <p className="ui-field__hint" id={hintId}>{hint}</p>}
      {error && <p className="ui-field__error" id={errorId} role="alert">{error}</p>}
      {disabled && disabledReason && (
        <p className="ui-field__disabled-reason" id={disabledReasonId}>{disabledReason}</p>
      )}
    </>
  );
}

export function joinDescribedBy(...values: Array<string | undefined>) {
  return values.filter(Boolean).join(" ") || undefined;
}

export function joinClassNames(...values: Array<string | undefined | false>) {
  return values.filter(Boolean).join(" ");
}
