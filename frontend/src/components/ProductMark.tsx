import type { SVGProps } from "react";

/**
 * Vaelor's clustered-signal mark.
 *
 * The six outer nodes represent heterogeneous compute joined through one
 * control plane. The amplified core remains legible at navigation-icon sizes.
 */
export function ProductMark(props: SVGProps<SVGSVGElement>) {
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
