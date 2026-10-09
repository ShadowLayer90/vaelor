import { createElement, type SVGProps } from "react";
import { hugeicons, type HugeiconName } from "./hugeicons";
import "../styles/icon.css";

/*
 * Every icon in the app goes through this one component (VD-200): the
 * approved Hugeicons set (hugeicons.ts), colour from currentColor, a 1.5 px
 * stroke. Each glyph carries its own line caps and joins, as the set draws
 * them, so the svg sets none.
 *
 * The stroke is 1.5 CSS px at every size, not 1.5 viewBox units: each shape
 * carries vector-effect="non-scaling-stroke". In viewBox units a 16 px icon
 * drew a 1.0 px line and an 18 px one 1.125 px, so at DPR 1 nine inked pixels
 * in ten were only partly covered and every inline icon looked soft (polish
 * audit, 2026-10-07). The set's own guidance is the same: scale the canvas,
 * not the stroke. At 24 px nothing changes.
 *
 * The svg carries the `ui-icon` class (styles/icon.css), which keeps it from
 * shrinking in a flex row: without it a long callout squeezed its shield to
 * 4 px on a phone.
 */

/** Older names for an approved glyph: the same meaning, drawn by the set. */
const aliases = { nvme: "drive", terminal: "console" } as const satisfies Record<string, HugeiconName>;

export type IconName = HugeiconName | keyof typeof aliases;

/** The approved sizes (VD-200): inline text, the rail and icon tiles, buttons, standalone. */
export const ICON_SIZE = { inline: 16, nav: 18, button: 20, standalone: 24 } as const;

/** A size on the approved scale; anything else is a type error, not a fuzzy icon. */
export type IconSize = (typeof ICON_SIZE)[keyof typeof ICON_SIZE];

/** The stroke every glyph draws, in CSS px at any size. */
export const ICON_STROKE_PX = 1.5;

export function Icon({
  name,
  size = ICON_SIZE.button,
  className,
  ...props
}: { name: IconName; size?: IconSize } & Omit<SVGProps<SVGSVGElement>, "width" | "height">) {
  const glyph = hugeicons[name in aliases ? aliases[name as keyof typeof aliases] : name as HugeiconName];
  return (
    <svg
      aria-hidden="true"
      fill="none"
      height={size}
      viewBox="0 0 24 24"
      width={size}
      stroke="currentColor"
      strokeWidth={ICON_STROKE_PX}
      {...props}
      className={className ? `ui-icon ${className}` : "ui-icon"}
    >
      {glyph.map(([tag, attrs], index) => createElement(tag, { key: index, vectorEffect: "non-scaling-stroke", ...attrs }))}
    </svg>
  );
}
