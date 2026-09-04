/**
 * Detector self-test.
 *
 * VLR-074 is why this exists. The composer was unreachable at 150% zoom behind
 * a `overflow: hidden` panel whose height clamp had inverted, and this harness
 * reported green through the whole of it - twice, across two releases - because
 * the only question it knew how to ask was "is the document wider than the
 * window". Reachability and occlusion checks were then added. But a detector
 * that never fires is indistinguishable from a page with no defects, and there
 * was nothing in the harness that could tell the two apart: a typo'd selector,
 * a `continue` guard that swallows every case, or a rename in the app would all
 * leave the harness confidently passing and blind.
 *
 * Each probe in `detector-probes.mjs` is a page built to trip exactly one
 * detector, run through the very same `qualityReport` / `layoutReport` /
 * `assertKeyControlsReachable` used against the product. A clean page then has
 * to trip none of them, so an over-eager detector is caught as well as a dead
 * one.
 *
 * This is a separate responsibility from running the product acceptance: it
 * proves the instruments read before anything is claimed about the appliance,
 * and `node scripts/responsive-acceptance.mjs --self-test` runs it alone, with
 * no appliance target and no Python environment.
 */

import {
  cleanProbe, detectorProbes, unreachableProbe, wideProbe,
} from './detector-probes.mjs';
import {
  assertKeyControlsReachable, layoutReport, qualityReport,
} from './responsive-detectors.mjs';
import { assert } from './responsive-report.mjs';

export async function assertDetectorsFire(browser) {
  const context = await browser.newContext({ viewport: { width: 900, height: 600 } });
  const page = await context.newPage();
  const fired = [];
  try {
    for (const probe of detectorProbes) {
      await page.setContent(probe.html);
      const report = await qualityReport(page);
      const hits = report[probe.field];
      assert(
        Array.isArray(hits) && hits.length > 0,
        `Detector ${probe.name} did not fire on a page built to trip it. `
        + `It cannot be relied on against the product. Report: ${JSON.stringify(report[probe.field])}`,
      );
      fired.push({ detector: probe.name, hits: hits.length });
    }

    // Horizontal escape, from `layoutReport` rather than `qualityReport`.
    await page.setContent(wideProbe);
    const wide = await layoutReport(page);
    assert(wide.offenders.length > 0, 'Detector offenders did not fire on an element painted past the viewport edge.');
    fired.push({ detector: 'offenders', hits: wide.offenders.length });

    /*
     * `assertKeyControlsReachable` throws rather than returning a list, so the
     * only way to prove it is live is to require the throw.
     */
    await page.setContent(unreachableProbe);
    let threw = false;
    try {
      await assertKeyControlsReachable(page, 'probe', { viewport: { width: 900, height: 600 }, zoom: false }, ['.probe']);
    } catch {
      threw = true;
    }
    assert(threw, 'assertKeyControlsReachable accepted a control that hit-tests to the element covering it.');
    fired.push({ detector: 'assertKeyControlsReachable', hits: 1 });

    // And nothing fires on a page with none of these problems, so a detector
    // that reports everything is caught the same way a dead one is.
    await page.setContent(cleanProbe);
    const clean = await qualityReport(page);
    const cleanLayout = await layoutReport(page);
    const noisy = Object.entries(clean)
      .filter(([field, value]) => Array.isArray(value) && value.length > 0)
      .map(([field, value]) => `${field}: ${JSON.stringify(value)}`);
    assert(noisy.length === 0, `Detectors reported problems on a clean page: ${noisy.join('; ')}`);
    assert(cleanLayout.offenders.length === 0, `Layout offenders reported on a clean page: ${JSON.stringify(cleanLayout.offenders)}`);
  } finally {
    await context.close();
  }
  return fired;
}
