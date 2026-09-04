/**
 * A trend line for readings that exist.
 *
 * Missing samples arrive as `null` and are simply not plotted. They used to
 * arrive as `0`, which drew a filled trace along the axis under the label
 * "Recent metric trend" — a chart of a sensor this machine does not have,
 * shaped exactly like a chart of a sensor pinned at zero.
 */
export function Sparkline({
  values,
  tone = "blue",
}: {
  values: Array<number | null>;
  tone?: "blue" | "green" | "amber" | "pink";
}) {
  const span = Math.max(values.length - 1, 1);
  const measured = values
    .map((value, index) => ({ value, index }))
    .filter((entry): entry is { value: number; index: number } => entry.value !== null);

  if (!measured.length) {
    return (
      <svg
        className={`sparkline sparkline--${tone} sparkline--unmeasured`}
        viewBox="0 0 100 32"
        preserveAspectRatio="none"
        role="img"
        aria-label="No trend to show: this reading is not available on this machine"
      >
        <g className="sparkline__grid" aria-hidden="true">
          <line x1="0" x2="100" y1="8" y2="8" />
          <line x1="0" x2="100" y1="19" y2="19" />
          <line x1="0" x2="100" y1="30" y2="30" />
        </g>
      </svg>
    );
  }

  const numbers = measured.map((entry) => entry.value);
  const min = Math.min(...numbers);
  const max = Math.max(...numbers);
  const range = max - min || 1;
  const coordinates = measured.map(({ value, index }) => ({
    x: (index / span) * 100,
    y: 30 - ((value - min) / range) * 24,
  }));
  const points = coordinates.map(({ x, y }) => `${x},${y}`).join(" ");
  const latest = coordinates.at(-1) ?? { x: 100, y: 16 };
  const first = coordinates[0];
  const areaPoints = `${first.x},32 ${points} ${latest.x},32`;
  return (
    <svg
      className={`sparkline sparkline--${tone}`}
      viewBox="0 0 100 32"
      preserveAspectRatio="none"
      role="img"
      aria-label={`Recent metric trend, ${measured.length} of ${values.length} samples measured`}
    >
      <g className="sparkline__grid" aria-hidden="true">
        <line x1="0" x2="100" y1="8" y2="8" />
        <line x1="0" x2="100" y1="19" y2="19" />
        <line x1="0" x2="100" y1="30" y2="30" />
      </g>
      <polygon className="sparkline__area" points={areaPoints} />
      <polyline className="sparkline__trace" points={points} />
      <circle
        className="sparkline__endpoint-glow"
        cx={latest.x}
        cy={latest.y}
        r="2.4"
      />
      <circle
        className="sparkline__endpoint"
        cx={latest.x}
        cy={latest.y}
        r="1.15"
      />
    </svg>
  );
}
