import { useId } from "react";

/**
 * An original technical illustration of a mobile workstation — a laptop.
 *
 * Drawn the way the Pironman and workstation illustrations are drawn: isometric
 * primitives in Vaelor's graphite, cyan and orange language, never traced or
 * derived from product photography or CAD. See ASSET_PROVENANCE.md.
 *
 * A laptop is a different form factor from the compact desktop, so it draws
 * different hardware. Every feature it encodes is one this class of machine
 * actually has: an open lid carrying an internal display (this machine has an
 * eDP panel — the desktop has none), a keyboard deck with a trackpad, a battery
 * (the one thing a laptop has that the desktop does not, so it is honest to show
 * it), side ventilation, and the three compute engines — processor, graphics
 * and neural accelerator. The three engine marks sit across the display, the
 * machine's clean principal face, the same way the workstation's marks sit on
 * its lid — spaced, not crowded — so the keyboard deck stays a plain keyboard
 * and reads unmistakably as a laptop. It deliberately draws no controllable
 * case fan and no lighting: this laptop's fans are embedded-controller-only,
 * neither readable nor writable from software, so the drawing shows louvres and
 * never a fan the product would then be unable to command.
 */

/**
 * The frame, measured from the drawing rather than chosen.
 *
 * The deck's front-left corner sits at x 44 and the lid and base right edge at
 * x 196, so the drawing is dead centre horizontally: (44 + 196) / 2 = 120, the
 * midpoint of a 240-wide box, with a matching 44-unit margin on each side. It
 * spans y 12 (the top of the raised screen) to 235 (the base of the shadow
 * ellipse). The origin and height are chosen so the 8-unit clearance above the
 * screen equals the 8-unit clearance below the shadow: 12 − 4 = 4 + 239 − 235.
 * **No coordinate in the drawing changes** to achieve it.
 */
export const VIEWBOX_ORIGIN_Y = 4;
export const VIEWBOX_HEIGHT = 239;
export const VIEWBOX_WIDTH = 240;

interface Props {
  label?: string;
  size?: number;
  /** Draw the neural engine mark. Omitted when no NPU was discovered. */
  neural?: boolean;
}

export function LaptopDeviceIcon({ label, neural = true, size = 92 }: Props) {
  const titleId = useId();
  const defs = `${titleId}-laptop`;
  return (
    <svg
      aria-hidden={label ? undefined : "true"}
      aria-labelledby={label ? titleId : undefined}
      className="laptop-device-icon"
      data-enclosure="laptop"
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
        <linearGradient id={`${defs}-screen`} x1="0" x2="1" y1="0" y2="1">
          <stop offset="0" stopColor="#123138" />
          <stop offset="1" stopColor="#06161c" />
        </linearGradient>
        <linearGradient id={`${defs}-deck`} x1="0" x2="0" y1="0" y2="1">
          <stop offset="0" stopColor="#2c363d" />
          <stop offset="1" stopColor="#161d22" />
        </linearGradient>
        <linearGradient id={`${defs}-side`} x1="0" x2="0" y1="0" y2="1">
          <stop offset="0" stopColor="#1a2228" />
          <stop offset="1" stopColor="#080c0f" />
        </linearGradient>
        <filter id={`${defs}-glow`} x="-70%" y="-70%" width="240%" height="240%">
          <feGaussianBlur stdDeviation="2.6" />
        </filter>
      </defs>

      <ellipse cx="120" cy="224" rx="86" ry="11" fill="#000" opacity=".55" />

      {/* The open lid, raised from the deck's rear edge. Drawn first so the deck
          tucks its hinge in front of it. */}
      <path d="M74 150 196 128 196 12 74 34Z" fill={`url(#${defs}-lid)`} stroke="#9aa6ae" strokeWidth="2" />

      {/* The internal display. This machine has one — an eDP panel — so it is
          drawn, unlike the desktop, which has no front display and draws none. */}
      <path d="M82.5 40.6 187.5 21.7 187.5 121.4 82.5 140.3Z" fill={`url(#${defs}-screen)`} stroke="#2a3b44" strokeWidth="1.2" />

      {/* The three engines — processor, graphics and neural accelerator. They
          are the instrument marks this class of machine is defined by, spaced
          across the centre of the display (the machine's clean principal face)
          and following its tilt, exactly as the workstation lays them along its
          lid. The display is the laptop's counterpart to that flat top. */}
      <g>
        <circle cx="111" cy="85" r="8.5" fill="#00aee8" filter={`url(#${defs}-glow)`} opacity=".2" />
        <circle cx="111" cy="85" r="6" fill="#061218" stroke="#29c9f2" strokeWidth="1.5" />
        <path d="M108 85h6M111 82v6" stroke="#66e6ff" strokeWidth="1.2" />

        <circle cx="135" cy="81" r="8.5" fill="#ff6a00" filter={`url(#${defs}-glow)`} opacity=".22" />
        <circle cx="135" cy="81" r="6" fill="#160b04" stroke="#ff8a33" strokeWidth="1.5" />
        <path d="M132 81h6M135 78v6" stroke="#ffb066" strokeWidth="1.2" />

        {neural && (
          <>
            <circle cx="159" cy="77" r="8.5" fill="#24d875" filter={`url(#${defs}-glow)`} opacity=".2" />
            <circle cx="159" cy="77" r="6" fill="#04140b" stroke="#40e885" strokeWidth="1.5" />
            <path d="M156 77h6M159 74v6" stroke="#8bf5b6" strokeWidth="1.2" />
          </>
        )}
      </g>

      {/* Base side faces — the deck's 12-unit thickness. */}
      <path d="M166 188 196 128 196 140 166 200Z" fill={`url(#${defs}-side)`} stroke="#5d6a73" strokeWidth="1.4" />
      <path d="M44 210 166 188 166 200 44 222Z" fill={`url(#${defs}-side)`} stroke="#7d8a93" strokeWidth="1.4" />

      {/* Ventilation on the right flank — this laptop cools itself through an
          embedded controller and exposes no fan software can command, so it is
          drawn as louvres, not a fan. */}
      <g stroke="#4a5960" strokeWidth="1.2" opacity=".85">
        <path d="M172 189 178 177m-1 15 6-12m5 9 6-12" />
      </g>

      {/* The keyboard deck. */}
      <path d="M74 150 196 128 166 188 44 210Z" fill={`url(#${defs}-deck)`} stroke="#8c99a2" strokeWidth="1.8" />

      {/* A suggestion of keys — three rows stepping into the deck. */}
      <g stroke="#3d474e" strokeWidth="1.3" opacity=".85">
        <path d="M76.3 158.6 183.7 139.2" />
        <path d="M71.6 168.1 179 148.7" />
        <path d="M66.8 177.6 174.2 158.2" />
      </g>

      {/* The trackpad, front-centre of the deck. */}
      <path d="M98 186.9 130 181.1 122 197.1 90 202.9Z" fill="#10171c" stroke="#4a5960" strokeWidth="1.2" />

      {/* A battery cue — the honest one thing this machine has that the desktop
          does not. A thin gauge on the deck, filled to indicate charge. */}
      <g>
        <path d="M63 190.5 85 186.5 81 195.5 59 199.5Z" fill="#0b1216" stroke="#67757e" strokeWidth="1.1" />
        <path d="M63 190.5 76.2 188.1 72.2 197.1 59 199.5Z" fill="#24d875" opacity=".85" />
        <path d="M85 186.9 89 186.2 87.5 189.6 83.5 190.3Z" fill="#67757e" />
      </g>
    </svg>
  );
}
