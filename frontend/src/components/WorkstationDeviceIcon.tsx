import { useId } from "react";

/**
 * An original technical illustration of a compact x86 workstation.
 *
 * Drawn the way the Pironman illustrations are drawn: isometric primitives in
 * Vaelor's graphite, cyan and orange language, never traced or derived from
 * product photography. See ASSET_PROVENANCE.md.
 *
 * Every feature it encodes is one Vaelor measured on the machine rather than
 * one read off a datasheet: a low square chassis with the chamfered top edge
 * this class of mini workstation has, a single NVMe drive, a rear port bank,
 * side ventilation, and three compute engines - processor, graphics and neural
 * accelerator. It deliberately draws no lighting, no front display and no
 * controllable case fan, because this machine has none of those and the
 * drawing is not permitted to promise hardware the product then cannot offer.
 */

/**
 * The frame, measured from the drawing rather than chosen.
 *
 * The chassis spans x 48..192 and the shadow ellipse x 42..198, so the drawing
 * is already dead centre horizontally: midpoint 120 of a 240-wide box. It
 * spans y 68..178 with the shadow reaching 196, which put its vertical
 * midpoint at 132 against a box midpoint of 110 — the drawing sat 22 units, a
 * tenth of its own height, below the middle of the panel it is centred in. The
 * origin moves so that the two agree; **no coordinate in the drawing changes**,
 * and the shadow's lowest point (196) stays well inside the frame's lower
 * bound (242).
 */
export const VIEWBOX_ORIGIN_Y = 22;
export const VIEWBOX_HEIGHT = 220;
export const VIEWBOX_WIDTH = 240;

interface Props {
  label?: string;
  size?: number;
  /** Draw the neural engine mark. Omitted when no NPU was discovered. */
  neural?: boolean;
}

export function WorkstationDeviceIcon({ label, neural = true, size = 92 }: Props) {
  const titleId = useId();
  const defs = `${titleId}-workstation`;
  return (
    <svg
      aria-hidden={label ? undefined : "true"}
      aria-labelledby={label ? titleId : undefined}
      className="workstation-device-icon"
      data-enclosure="workstation"
      data-neural={neural ? "true" : "false"}
      height={size}
      role={label ? "img" : undefined}
      viewBox={`0 ${VIEWBOX_ORIGIN_Y} ${VIEWBOX_WIDTH} ${VIEWBOX_HEIGHT}`}
      width={size}
    >
      {label && <title id={titleId}>{label}</title>}
      <defs>
        <linearGradient id={`${defs}-lid`} x1="0" x2="1" y1="0" y2="1">
          <stop offset="0" stopColor="#3b474f" />
          <stop offset=".46" stopColor="#222c33" />
          <stop offset="1" stopColor="#131a1f" />
        </linearGradient>
        <linearGradient id={`${defs}-left`} x1="0" x2="0" y1="0" y2="1">
          <stop offset="0" stopColor="#1a2228" />
          <stop offset="1" stopColor="#080c0f" />
        </linearGradient>
        <linearGradient id={`${defs}-right`} x1="0" x2="0" y1="0" y2="1">
          <stop offset="0" stopColor="#121a20" />
          <stop offset="1" stopColor="#05080a" />
        </linearGradient>
        <linearGradient id={`${defs}-chamfer`} x1="0" x2="1">
          <stop offset="0" stopColor="#8c99a2" />
          <stop offset="1" stopColor="#44515a" />
        </linearGradient>
        <filter id={`${defs}-glow`} x="-70%" y="-70%" width="240%" height="240%">
          <feGaussianBlur stdDeviation="2.6" />
        </filter>
      </defs>

      <ellipse cx="120" cy="186" rx="78" ry="10" fill="#000" opacity=".55" />

      {/* Body: a low square chassis seen isometrically. */}
      <path d="M48 104 120 68l72 36-72 36Z" fill={`url(#${defs}-lid)`} stroke="#9aa6ae" strokeWidth="2" />
      <path d="M48 104v38l72 36v-38Z" fill={`url(#${defs}-left)`} stroke="#7d8a93" strokeWidth="1.6" />
      <path d="M192 104v38l-72 36v-38Z" fill={`url(#${defs}-right)`} stroke="#5d6a73" strokeWidth="1.6" />

      {/* The chamfered top edge this chassis class is shaped by. */}
      <path d="M120 68 192 104l-14 7-58-29Z" fill={`url(#${defs}-chamfer)`} opacity=".85" />
      <path d="M120 68 48 104l14 7 58-29Z" fill={`url(#${defs}-chamfer)`} opacity=".5" />

      {/* Ventilation on the right flank — this chassis cools itself and exposes
          no fan Vaelor can command, so it is drawn as louvres, not a fan.

          The flank's upper edge runs (120,140)->(192,104), so it drops 4.5 for
          every 9 of x. The louvres step with it and start 6.5 below it, which
          keeps all six on the flank; stepping 5 instead started them 5.5-8
          above that edge, so they began inside the lid and crossed the chamfer
          — slots cut through the corner of a solid. */}
      <g stroke="#4a5960" strokeWidth="1.4" opacity=".9">
        <path d="M133 140v26m9-30.5v26m9-30.5v26m9-30.5v26m9-30.5v26m9-30.5v26" />
      </g>

      {/* Rear port bank on the left flank. */}
      <g>
        <rect x="58" y="120" width="20" height="8" rx="1.2" fill="#04070a" stroke="#67757e" strokeWidth="1.1" />
        <rect x="58" y="132" width="20" height="8" rx="1.2" fill="#04070a" stroke="#67757e" strokeWidth="1.1" />
        <circle cx="88" cy="139" r="3.4" fill="#04070a" stroke="#67757e" strokeWidth="1.1" />
        <circle cx="98" cy="144" r="3.4" fill="#04070a" stroke="#67757e" strokeWidth="1.1" />
      </g>

      {/* One NVMe drive — the single fitted disk, drawn once. */}
      <g transform="translate(96 86)">
        <path d="M0 0h34l8 4-34 0Z" fill="#0d151a" stroke="#00aee8" strokeOpacity=".75" strokeWidth="1.2" />
        <path d="M6 2h20" stroke="#29c9f2" strokeWidth="1.1" opacity=".8" />
      </g>

      {/* The three engines. This machine is defined by them, so they are the
          instrument marks on the lid rather than decoration. */}
      <g>
        <circle cx="96" cy="106" r="8.5" fill="#00aee8" filter={`url(#${defs}-glow)`} opacity=".2" />
        <circle cx="96" cy="106" r="6" fill="#061218" stroke="#29c9f2" strokeWidth="1.5" />
        <path d="M93 106h6M96 103v6" stroke="#66e6ff" strokeWidth="1.2" />

        <circle cx="120" cy="118" r="8.5" fill="#ff6a00" filter={`url(#${defs}-glow)`} opacity=".22" />
        <circle cx="120" cy="118" r="6" fill="#160b04" stroke="#ff8a33" strokeWidth="1.5" />
        <path d="M117 118h6M120 115v6" stroke="#ffb066" strokeWidth="1.2" />

        {neural && (
          <>
            <circle cx="144" cy="106" r="8.5" fill="#24d875" filter={`url(#${defs}-glow)`} opacity=".2" />
            <circle cx="144" cy="106" r="6" fill="#04140b" stroke="#40e885" strokeWidth="1.5" />
            <path d="M141 106h6M144 103v6" stroke="#8bf5b6" strokeWidth="1.2" />
          </>
        )}
      </g>

      {/* Power indicator on the front chamfer. */}
      <circle cx="120" cy="150" r="3.2" fill="#24d875" />
    </svg>
  );
}
