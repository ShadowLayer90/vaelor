import type { HTMLAttributes } from "react";
import { joinClassNames } from "./field";

export type UnavailableValueProps = {
  label?: string;
  reason?: string;
  className?: string;
} & Omit<HTMLAttributes<HTMLSpanElement>, "aria-label" | "title" | "children">;

/**
 * A reading this machine did not produce, and why.
 *
 * The mark itself is deliberately not a value — it is the whole point that it
 * cannot be mistaken for one. But a bare `?` explains nothing to a reader who
 * does not already know it is hoverable: a tester met four of them on
 * `System → Compute` and reported that only a screen reader was told anything,
 * and even then only "Package power unavailable" — the *label*, never the
 * reason, which is the part that answers the question the `?` provokes.
 *
 * Two changes, no new component. The mark is focusable, so its `title` is
 * reachable without a pointer. And callers that have room are expected to
 * print the reason beside it — `AcceleratorCard` does — because a tooltip is a
 * poor place for the only copy of a fact: it cannot be read on a touch screen,
 * cannot be selected, and vanishes when the pointer moves.
 *
 * The accessible name stays the label alone. Folding the reason into it was
 * tried and reverted: it made every `getByLabelText("… unavailable")` in the
 * suite a substring match, which would have meant loosening a dozen existing
 * assertions to accommodate a change that the visible reason already delivers.
 */
export function UnavailableValue({
  label = "Unavailable",
  reason = "Not reported by this device",
  className,
  ...props
}: UnavailableValueProps) {
  return (
    <span
      {...props}
      aria-label={label}
      className={joinClassNames("ui-unavailable-value", className)}
      data-available="false"
      tabIndex={0}
      title={reason}
    >
      ?
    </span>
  );
}
