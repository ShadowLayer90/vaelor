export function AmbientBackground() {
  return (
    <div className="ambient-background" aria-hidden="true">
      <span className="ambient-background__thermal ambient-background__thermal--orange" />
      <span className="ambient-background__thermal ambient-background__thermal--green" />
      <span className="ambient-background__grid" />
      <span className="ambient-background__nodes" />
      <span className="ambient-background__radar" />
      <span className="ambient-background__scan-band" />
      <span className="ambient-background__trace ambient-background__trace--one" />
      <span className="ambient-background__trace ambient-background__trace--two" />
      <span className="ambient-background__trace ambient-background__trace--three" />
    </div>
  );
}
