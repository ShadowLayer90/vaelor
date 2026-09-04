/**
 * How this harness states a failure.
 *
 * `assert` is the only way anything in the responsive acceptance harness fails,
 * and `variantLabel` is the phrase every one of those failures uses to name the
 * viewport and zoom level it failed at. The two live together because a report
 * that cannot say *where* it looked is not usable: most of the defects this
 * harness has caught were true at one or two of eighteen variants, and without
 * the label there is no way to tell which.
 *
 * They are separate from the detectors so the dependency runs one way. The
 * viewport matrix needs both to describe itself, and the detectors need both to
 * describe what they found; neither should have to import the other.
 */

export function assert(condition, message) {
  if (!condition) throw new Error(message);
}

export function variantLabel(viewport, zoom = false) {
  const window = zoom ? ` of a ${Math.round(viewport.width * zoom)}x${Math.round(viewport.height * zoom)} window` : '';
  return `${viewport.width}x${viewport.height} CSS px at ${zoom ? Math.round(zoom * 100) : 100}% browser zoom${window}`;
}
