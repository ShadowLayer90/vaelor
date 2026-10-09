import type { SVGProps } from "react";

/** The size the small mark is drawn for: its 18-unit grid maps 1:1 onto CSS px. */
export const PRODUCT_MARK_SMALL_PX = 18;

/**
 * Vaelor's clustered-signal mark.
 *
 * The six outer nodes represent heterogeneous compute joined through one
 * control plane, around one amplified core.
 *
 * Two drawings, one mark. `small` (the default, and every mount today: the
 * rail, the top bar, sign-in and the error screen) is drawn on an 18-unit grid
 * shown at exactly 18 px, so each 1.5-unit stroke is 1.5 CSS px - 3 device px
 * at DPR 2 - and nothing is translucent. It keeps the large mark's identity:
 * the hexagon of nodes, three spokes into a solid core, and the base bar, a
 * filled 2 px rect on whole pixels. A hexagon and ring alone read as the
 * Settings gear at 18 px (owner's pick "Framed Y", VD-200, 2026-10-07).
 * The 32-unit `large` drawing, with its inner ring, spokes and opacity
 * layers, is kept for sizes of 32 px and up:
 * shrunk to 18 px its 1.2 to 1.7 strokes fell to 0.7 to 1 px and its .52
 * layers turned to mush (polish audit, 2026-10-07).
 */
export function ProductMark({ variant = "small", ...props }: { variant?: "small" | "large" } & SVGProps<SVGSVGElement>) {
  if (variant === "small") {
    return (
      <svg
        aria-hidden="true"
        fill="none"
        height={PRODUCT_MARK_SMALL_PX}
        viewBox="0 0 18 18"
        width={PRODUCT_MARK_SMALL_PX}
        stroke="currentColor"
        strokeLinecap="round"
        strokeLinejoin="round"
        strokeWidth="1.5"
        {...props}
      >
        <path d="M9 2 14 5v5l-5 3-5-3V5Z" />
        <path data-mark-part="spokes" d="M9 13 9 9M14 5l-3.66 1.83M4 5l3.66 1.83" />
        <g fill="currentColor" stroke="none">
          <circle data-mark-part="core" cx="9" cy="7.5" r="2" />
          <circle cx="9" cy="2" r="1.5" />
          <circle cx="14" cy="5" r="1.5" />
          <circle cx="14" cy="10" r="1.5" />
          <circle cx="9" cy="13" r="1.5" />
          <circle cx="4" cy="10" r="1.5" />
          <circle cx="4" cy="5" r="1.5" />
        </g>
        <rect data-mark-part="base" x="5" y="16" width="8" height="2" rx=".75" fill="currentColor" stroke="none" />
      </svg>
    );
  }
  return (
    <svg
      aria-hidden="true"
      fill="none"
      viewBox="0 0 32 32"
      stroke="currentColor"
      strokeLinecap="round"
      strokeLinejoin="round"
      {...props}
    >
      <path d="M7 8.5 16 3l9 5.5v11L16 25l-9-5.5v-11Z" strokeWidth="1.6" opacity=".82" />
      <path d="m9.8 10.2 6.2-3.8 6.2 3.8v7.6L16 21.6l-6.2-3.8v-7.6Z" strokeWidth="1.2" opacity=".52" />
      <path d="M16 6.4v5.1m-6.2-1.3 4.2 2.5m8.2-2.5-4.2 2.5M9.8 17.8l4.2-2.5m8.2 2.5L18 15.3M16 16.5v5.1" strokeWidth="1.35" />
      <circle cx="16" cy="14" r="3.2" strokeWidth="1.7" />
      <circle cx="16" cy="14" r="1.15" fill="currentColor" stroke="none" />
      <circle cx="16" cy="3" r="1.25" fill="currentColor" stroke="none" />
      <circle cx="25" cy="8.5" r="1.25" fill="currentColor" stroke="none" />
      <circle cx="25" cy="19.5" r="1.25" fill="currentColor" stroke="none" />
      <circle cx="16" cy="25" r="1.25" fill="currentColor" stroke="none" />
      <circle cx="7" cy="19.5" r="1.25" fill="currentColor" stroke="none" />
      <circle cx="7" cy="8.5" r="1.25" fill="currentColor" stroke="none" />
      <path d="M11 28.5h10" strokeWidth="1.4" opacity=".72" />
    </svg>
  );
}
