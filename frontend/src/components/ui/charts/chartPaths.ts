/**
 * The SVG path strings of the time-series chart, and the stacking of layers
 * they are drawn from. Pure arithmetic, like `timeScale`: a `null` is "no
 * sample" and is never turned into a number here.
 */
import type { Segment } from "./timeScale";

const isSample = (value: number | null | undefined): value is number =>
  typeof value === "number" && Number.isFinite(value);

export interface StackedLayer { lower: (number | null)[]; upper: (number | null)[] }

/**
 * Stack layers bottom-up. A bucket where ANY layer has no sample is a gap in
 * every layer: a stack with a missing layer would draw a total that is not one.
 */
export function stackLayers(layers: readonly (readonly (number | null)[])[]): StackedLayer[] {
  const count = Math.max(0, ...layers.map((layer) => layer.length));
  const running: (number | null)[] = Array.from({ length: count }, (_, index) =>
    layers.every((layer) => isSample(layer[index])) ? 0 : null);
  return layers.map((layer) => {
    const lower = [...running];
    const upper = running.map((base, index) => (base === null ? null : base + (layer[index] as number)));
    upper.forEach((value, index) => { running[index] = value; });
    return { lower, upper };
  });
}

const point = (x: number, y: number) => x.toFixed(1) + " " + y.toFixed(1);

/** The `d` of one segment's line. */
export function linePath(segment: Segment, xAt: (index: number) => number, yAt: (index: number) => number): string {
  const parts: string[] = [];
  for (let index = segment.from; index <= segment.to; index += 1) {
    parts.push((index === segment.from ? "M" : "L") + point(xAt(index), yAt(index)));
  }
  return parts.join(" ");
}

/** The `d` of one segment's filled band between two curves. */
export function areaPath(
  segment: Segment,
  xAt: (index: number) => number,
  upperAt: (index: number) => number,
  lowerAt: (index: number) => number,
): string {
  const back: string[] = [];
  for (let index = segment.to; index >= segment.from; index -= 1) back.push("L" + point(xAt(index), lowerAt(index)));
  return linePath(segment, xAt, upperAt) + " " + back.join(" ") + " Z";
}
