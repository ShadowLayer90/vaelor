/**
 * Conversion and validation for the case-lighting colour, shared by the hex
 * field, the three channel fields, the presets and the native picker so every
 * representation stays in step.
 */

export interface Rgb {
  r: number;
  g: number;
  b: number;
}

export const CHANNEL_MIN = 0;
export const CHANNEL_MAX = 255;

const HEX_PATTERN = /^#?([0-9a-f]{6}|[0-9a-f]{3})$/i;

/** Expand `#abc` to `#aabbcc` and normalise to lower-case with a leading `#`. */
export function normalizeHex(value: string): string | null {
  const match = HEX_PATTERN.exec(String(value ?? "").trim());
  if (!match) return null;
  const digits = match[1].toLowerCase();
  const full = digits.length === 3
    ? digits.split("").map((digit) => digit + digit).join("")
    : digits;
  return "#" + full;
}

export function hexToRgb(value: string): Rgb | null {
  const hex = normalizeHex(value);
  if (!hex) return null;
  return {
    r: Number.parseInt(hex.slice(1, 3), 16),
    g: Number.parseInt(hex.slice(3, 5), 16),
    b: Number.parseInt(hex.slice(5, 7), 16),
  };
}

export function rgbToHex({ r, g, b }: Rgb): string {
  const channel = (value: number) =>
    clampChannel(value).toString(16).padStart(2, "0");
  return "#" + channel(r) + channel(g) + channel(b);
}

export function clampChannel(value: number): number {
  if (!Number.isFinite(value)) return CHANNEL_MIN;
  return Math.min(CHANNEL_MAX, Math.max(CHANNEL_MIN, Math.round(value)));
}

/**
 * Validate a typed channel value. Out-of-range and non-numeric entries are
 * reported rather than silently clamped, so the user is told what was wrong.
 */
export function channelError(raw: string): string {
  const text = String(raw ?? "").trim();
  if (!text) return "Enter a number from 0 to 255.";
  if (!/^-?\d+$/.test(text)) return "Use whole numbers only.";
  const value = Number(text);
  if (value < CHANNEL_MIN || value > CHANNEL_MAX) return "Use a value from 0 to 255.";
  return "";
}

export function hexError(raw: string): string {
  const text = String(raw ?? "").trim();
  if (!text) return "Enter a colour such as #FF6A00.";
  if (!normalizeHex(text)) return "Use a 6-digit hex colour such as #FF6A00.";
  return "";
}

/** Effects that render their own colour cycle and ignore the chosen colour. */
export const MULTICOLOUR_STYLES = ["rainbow", "rainbow_reverse", "hue_cycle"] as const;

export function isMulticolourStyle(style: string): boolean {
  return (MULTICOLOUR_STYLES as readonly string[]).includes(style);
}
