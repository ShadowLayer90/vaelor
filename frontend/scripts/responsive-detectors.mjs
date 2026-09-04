/**
 * The instruments. Everything this harness is able to notice is in this file.
 *
 * `layoutReport` and `qualityReport` are pure measurements: they run in the
 * page, return findings, and assert nothing. `assertPageQuality` turns those
 * findings into failures, and `assertKeyControlsReachable` asks the one
 * question a report cannot - can a named control actually be reached.
 *
 * They are separated from the surfaces they are pointed at because they are
 * also pointed at synthetic pages: `responsive-self-test.mjs` runs each
 * detector against a page built to trip it, using these exact functions. A
 * detector that only ever runs against the product cannot be told apart from a
 * product with no defects, which is how VLR-074 shipped twice behind a green
 * run.
 */

import { assert, variantLabel } from './responsive-report.mjs';

export async function layoutReport(page, selector = 'main *') {
  return page.locator(selector).evaluateAll((elements) => {
    const main = document.querySelector('main');
    const mainRect = main?.getBoundingClientRect();
    const sidebar = document.querySelector('.sidebar');
    const sidebarRect = sidebar?.getBoundingClientRect();
    return {
    viewport: { width: window.innerWidth, height: window.innerHeight },
    documentWidth: document.documentElement.scrollWidth,
    /*
     * What scrollWidth has to be compared against is clientWidth, not
     * innerWidth: innerWidth includes the vertical scrollbar gutter, so with
     * classic scrollbars the two differ by about 15px and a real horizontal
     * overflow up to the gutter width passed this gate unseen.
     */
    documentClientWidth: document.documentElement.clientWidth,
    documentHeight: document.documentElement.scrollHeight,
    /*
     * Taller-than-the-viewport is only a defect when the overflow cannot be
     * reached. A rail that scrolls keeps every destination clickable at 150%
     * zoom; one that does not silently drops "Settings" off the bottom.
     */
    shellContainmentIssues: window.innerWidth > 720 && sidebar && sidebarRect && (sidebarRect.top < -1 || sidebarRect.bottom > window.innerHeight + 1 || (sidebar.scrollHeight > sidebar.clientHeight + 1 && !['auto', 'scroll'].includes(window.getComputedStyle(sidebar).overflowY)))
      ? [{ className: 'sidebar', top: Math.round(sidebarRect.top), bottom: Math.round(sidebarRect.bottom), scrollHeight: sidebar.scrollHeight, clientHeight: sidebar.clientHeight }]
      : [],
    topLevelContainmentIssues: main && mainRect ? [...main.children].flatMap((element) => {
      const rect = element.getBoundingClientRect();
      const style = window.getComputedStyle(element);
      if (style.display === 'none' || style.visibility === 'hidden' || ['absolute', 'fixed'].includes(style.position) || rect.width === 0 || rect.height === 0) return [];
      const outside = rect.left < mainRect.left - 1 || rect.right > mainRect.right + 1 || rect.top < mainRect.top - 1 || rect.bottom > mainRect.bottom + 1;
      return outside ? [{ tag: element.tagName.toLowerCase(), className: String(element.className), child: { left: Math.round(rect.left), right: Math.round(rect.right), top: Math.round(rect.top), bottom: Math.round(rect.bottom) }, parent: { left: Math.round(mainRect.left), right: Math.round(mainRect.right), top: Math.round(mainRect.top), bottom: Math.round(mainRect.bottom) } }] : [];
    }) : [],
    offenders: elements.flatMap((element) => {
      const rect = element.getBoundingClientRect();
      const style = window.getComputedStyle(element);
      if (style.display === 'none' || style.visibility === 'hidden' || style.opacity === '0' || rect.width === 0 || rect.height === 0 || rect.bottom < 0 || element.closest(`[aria-hidden='true']`) || /(?:ambient|glow|orb|sweep)/.test(String(element.className))) return [];
      const left = Math.round(rect.left);
      const right = Math.round(rect.right);
      return left < -1 || right > window.innerWidth + 1 ? [{ tag: element.tagName.toLowerCase(), className: String(element.className), text: (element.textContent ?? '').trim().replace(/\s+/g, ' ').slice(0, 100), left, right }] : [];
    }),
  };
  });
}

export async function qualityReport(page) {
  return page.locator('main').evaluate((main) => {
    const visible = (element) => {
      const rect = element.getBoundingClientRect();
      const style = window.getComputedStyle(element);
      const closedDetails = element.closest('details:not([open])');
      const hiddenByDisclosure = Boolean(closedDetails && !element.closest('summary'));
      return !hiddenByDisclosure && style.display !== 'none' && style.visibility !== 'hidden' && style.opacity !== '0' && rect.width > 0 && rect.height > 0;
    };
    const controls = [...main.querySelectorAll('button, input, select, textarea, a[href], summary, [role=tab]')].filter(visible);
    const unlabeledControls = controls.flatMap((control) => {
      const labelledBy = control.getAttribute('aria-labelledby');
      const referenced = labelledBy ? labelledBy.split(/\s+/).some((id) => document.getElementById(id)?.textContent?.trim()) : false;
      const labels = 'labels' in control && control.labels ? control.labels.length : 0;
      const named = control.getAttribute('aria-label') || control.getAttribute('title') || control.textContent?.trim() || control.getAttribute('placeholder') || control.closest('label')?.textContent?.trim() || labels || referenced;
      return named ? [] : [{ tag: control.tagName.toLowerCase(), className: String(control.className), id: control.id, type: control.getAttribute('type'), html: control.outerHTML.slice(0, 240) }];
    });
    const validInline = (control) => {
      const style = window.getComputedStyle(control);
      if (control.matches(`[data-inline-control='true']`)) return true;
      if (control.tagName === 'A' && style.display === 'inline' && !control.closest('nav, form, [role=tablist], [role=dialog], .button')) return true;
      if (control.matches(`input[type='checkbox'], input[type='radio']`)) {
        const label = control.closest('label');
        if (label) {
          const rect = label.getBoundingClientRect();
          return rect.width >= 44 && rect.height >= 44;
        }
      }
      return false;
    };
    const undersizedControls = controls.flatMap((control) => {
      const rect = control.getBoundingClientRect();
      return rect.width < 44 || rect.height < 44 ? validInline(control) ? [] : [{ tag: control.tagName.toLowerCase(), label: control.getAttribute('aria-label') || control.textContent?.trim().slice(0, 60), width: Math.round(rect.width), height: Math.round(rect.height) }] : [];
    });
    const describe = (control) => ({
      tag: control.tagName.toLowerCase(),
      className: String(control.className),
      label: (control.getAttribute('aria-label') || control.textContent?.trim() || control.id || '').slice(0, 60),
    });
    /*
     * Reachability. A control clipped by an ancestor that hides its overflow
     * cannot be scrolled to, and nothing about the document's own size reports
     * it: the Ask Vaelor composer was cut off inside a height-capped panel
     * while the document stayed exactly as tall as the viewport.
     */
    const clippedControls = controls.flatMap((control) => {
      if (['absolute', 'fixed'].includes(window.getComputedStyle(control).position)) return [];
      const rect = control.getBoundingClientRect();
      for (let parent = control.parentElement; parent && parent !== document.documentElement; parent = parent.parentElement) {
        /*
         * `overflow` on <body> propagates to the viewport, so body reports
         * "hidden" while the page itself scrolls normally. Its box is the whole
         * content height and it clips nothing - treating it as a clipper
         * reported every below-the-fold control on a phone as cut off.
         */
        if (parent === document.body) continue;
        const parentStyle = window.getComputedStyle(parent);
        const clipsX = ['hidden', 'clip'].includes(parentStyle.overflowX);
        const clipsY = ['hidden', 'clip'].includes(parentStyle.overflowY);
        if (!clipsX && !clipsY) continue;
        // Nothing is actually out of reach if the box holds all of its content.
        if (parent.scrollHeight <= parent.clientHeight
          && parent.scrollWidth <= parent.clientWidth) continue;
        const box = parent.getBoundingClientRect();
        const hidden = (clipsY && (rect.bottom > box.bottom + 2 || rect.top < box.top - 2))
          || (clipsX && (rect.right > box.right + 2 || rect.left < box.left - 2));
        if (hidden) {
          return [{
            ...describe(control),
            control: { top: Math.round(rect.top), bottom: Math.round(rect.bottom), left: Math.round(rect.left), right: Math.round(rect.right) },
            clipper: {
              tag: parent.tagName, id: parent.id, className: String(parent.className),
              overflowX: parentStyle.overflowX, overflowY: parentStyle.overflowY,
              scrollHeight: parent.scrollHeight, clientHeight: parent.clientHeight,
              top: Math.round(box.top), bottom: Math.round(box.bottom),
              left: Math.round(box.left), right: Math.round(box.right),
            },
          }];
        }
      }
      return [];
    });
    /*
     * Occlusion. `.color-presets` paints after the colour grid and captured the
     * pointer for two thirds of the Blue field between 830px and 868px wide,
     * without ever widening the document. Chrome that is meant to sit on top —
     * fixed or sticky shells, and an open dialog over the page behind it — is
     * not a collision, so those are excluded rather than reported.
     */
    const layered = (element) => {
      for (let node = element; node && node !== document.documentElement; node = node.parentElement) {
        if (['fixed', 'sticky'].includes(window.getComputedStyle(node).position)) return true;
      }
      return false;
    };
    const openDialog = document.querySelector('[aria-modal="true"], [role=dialog]');
    const samplePoints = [[0.5, 0.5], [0.25, 0.25], [0.75, 0.25], [0.25, 0.75], [0.75, 0.75]];
    /*
     * A control scrolled out of its own scroller is not covered by whatever
     * paints at that coordinate - it is simply not on screen yet, and the user
     * reaches it by scrolling. Without this, everything below the fold of any
     * scroll region reported as occluded by the chrome above it.
     */
    const scrolledOutOfView = (element, x, y) => {
      for (let node = element.parentElement; node && node !== document.body; node = node.parentElement) {
        const style = window.getComputedStyle(node);
        const scrolls = ['auto', 'scroll'].includes(style.overflowY)
          || ['auto', 'scroll'].includes(style.overflowX);
        if (!scrolls) continue;
        const box = node.getBoundingClientRect();
        if (y < box.top || y > box.bottom || x < box.left || x > box.right) return true;
      }
      return false;
    };
    const coveredControls = controls.flatMap((control) => {
      if (['absolute', 'fixed'].includes(window.getComputedStyle(control).position)) return [];
      if (openDialog && !openDialog.contains(control)) return [];
      const rect = control.getBoundingClientRect();
      for (const [alongX, alongY] of samplePoints) {
        const x = rect.left + rect.width * alongX;
        const y = rect.top + rect.height * alongY;
        if (x < 1 || y < 1 || x > window.innerWidth - 1 || y > window.innerHeight - 1) continue;
        if (scrolledOutOfView(control, x, y)) continue;
        const hit = document.elementFromPoint(x, y);
        if (!hit || hit === control || control.contains(hit) || hit.contains(control) || layered(hit)) continue;
        if (hit.tagName === 'LABEL' && hit.htmlFor && document.getElementById(hit.htmlFor) === control) continue;
        return [{ ...describe(control), point: [Math.round(x), Math.round(y)], covering: { tag: hit.tagName.toLowerCase(), className: String(hit.className) } }];
      }
      return [];
    });
    const shreddedTextControls = controls.flatMap((control) => {
      const text = control.textContent?.trim().replace(/\s+/g, ' ') ?? '';
      const rect = control.getBoundingClientRect();
      return text.length >= 4 && rect.height > rect.width * 3
        ? [{ tag: control.tagName.toLowerCase(), label: text.slice(0, 60), width: Math.round(rect.width), height: Math.round(rect.height) }]
        : [];
    });
    const textProblems = [...main.querySelectorAll('h1, h2, h3, h4, p, li, dt, dd, label, small, code, th, td, button, summary')].flatMap((element) => {
      if (!visible(element) || !element.textContent?.trim()) return [];
      const style = window.getComputedStyle(element);
      const fontSize = Number.parseFloat(style.fontSize);
      const lineHeight = style.lineHeight === 'normal' ? fontSize * 1.2 : Number.parseFloat(style.lineHeight);
      return fontSize < 10 || lineHeight < fontSize ? [{ tag: element.tagName.toLowerCase(), text: element.textContent.trim().replace(/\s+/g, ' ').slice(0, 80), fontSize, lineHeight }] : [];
    });
    const ids = [...main.querySelectorAll('[id]')].map((element) => element.id).filter(Boolean);
    const duplicateIds = [...new Set(ids.filter((id, index) => ids.indexOf(id) !== index))];
    const verticalSpillIssues = [...main.querySelectorAll('*')].flatMap((element) => {
      const parent = element.parentElement;
      if (!parent || parent === main || !visible(element) || !visible(parent)) return [];
      if (parent.classList.contains('ui-button-wrap')) return [];
      const style = window.getComputedStyle(element);
      const parentStyle = window.getComputedStyle(parent);
      if (['absolute', 'fixed'].includes(style.position) || ['auto', 'scroll', 'hidden', 'clip'].includes(parentStyle.overflowY)) return [];
      if (!['block', 'flex', 'grid', 'inline-block', 'inline-flex', 'inline-grid'].includes(style.display)) return [];
      const rect = element.getBoundingClientRect();
      const parentRect = parent.getBoundingClientRect();
      if (rect.bottom <= parentRect.bottom + 8) return [];
      return [{
        tag: element.tagName.toLowerCase(),
        className: String(element.className),
        parentTag: parent.tagName.toLowerCase(),
        parentClassName: String(parent.className),
        spill: Math.round(rect.bottom - parentRect.bottom),
        text: (element.textContent ?? '').trim().replace(/\s+/g, ' ').slice(0, 100),
      }];
    });
    /*
     * Collision between two pieces of text that are meant to sit side by side.
     *
     * Reported at 150% and 175% desktop zoom: the Home hero's
     * UPTIME / VERSION / ACCESS / OPERATING SYSTEM row overlapped —
     * `2.1.0a19` clipped, `Administrator` over its neighbour, `Ubuntu 2…`
     * colliding — while this harness passed. Nothing here could see it.
     * `offenders` and `documentWidth` measure escape from the *viewport*, and
     * a grid item that overruns its own track and paints over the item beside
     * it never leaves the page. `coveredControls` measures occlusion, but only
     * of controls; a `<dl>` of `<dt>`/`<dd>` has no control in it.
     *
     * A harness that passes while 150% zoom is visibly broken is the more
     * serious half of that finding, so the instrument is what gets fixed here
     * and the stylesheet follows it.
     *
     * What is measured is the child's own content against the child's own box.
     * Comparing the *boxes* to each other does not work and is worth writing
     * down: a grid item sized by `minmax(0, 1fr)` keeps its track's width to
     * the pixel while the words inside it run past the edge, so the rectangles
     * never intersect and the reader still sees two labels on top of each
     * other. `scrollWidth > clientWidth` with nothing clipping or scrolling it
     * is what "this text is painted over its neighbour" actually looks like.
     *
     * Restricted to grid and flex parents with more than one child, where
     * something *is* sitting beside this. A lone child overflowing its parent
     * is a different finding, and `offenders` already answers it.
     */
    const textOf = (element) => (element.textContent ?? '').trim().replace(/\s+/g, ' ');
    const textCollisionIssues = [...main.querySelectorAll('*')].flatMap((parent) => {
      const parentStyle = window.getComputedStyle(parent);
      if (!['grid', 'flex', 'inline-grid', 'inline-flex'].includes(parentStyle.display)) return [];
      const children = [...parent.children].filter((child) => {
        const style = window.getComputedStyle(child);
        return visible(child)
          && !['absolute', 'fixed'].includes(style.position)
          && !child.closest(`[aria-hidden='true']`)
          /*
           * Text, not controls. A control has an intrinsic minimum — a 44px hit
           * target, an unbreakable button label — that legitimately exceeds a
           * narrow column, and whether it then covers anything is a hit-test
           * question `coveredControls` already asks, with `undersizedControls`
           * and `offenders` covering the rest. Arithmetic on `scrollWidth`
           * cannot tell those apart and reports every squeezed button.
           *
           * What is left is what this detector is for: a run of words wider
           * than the box holding it, painted over the words beside it, with no
           * control involved anywhere and nothing else in the harness able to
           * see it.
           */
          && !child.querySelector('button, input, select, textarea, a[href], summary, [role=tab]')
          && !child.matches('button, input, select, textarea, a[href], summary, [role=tab], .ui-button-wrap');
      });
      // One child is a column of one: it can overflow the parent, which
      // `offenders` and `documentWidth` already answer for.
      if (children.length < 2) return [];
      return children.flatMap((child) => {
        const style = window.getComputedStyle(child);
        if (style.overflowX !== 'visible') return [];
        const spill = child.scrollWidth - child.clientWidth;
        if (spill <= 3 || !textOf(child)) return [];
        return [{
          parent: `${parent.tagName.toLowerCase()}.${String(parent.className)}`,
          tag: child.tagName.toLowerCase(),
          className: String(child.className),
          text: textOf(child).slice(0, 80),
          spill: Math.round(spill),
          scrollWidth: child.scrollWidth,
          clientWidth: child.clientWidth,
        }];
      });
    });
    const feedbackIssues = [...new Set([...main.querySelectorAll('.notice, [role=alert], [role=status], .field-error')])].filter(visible).flatMap((element) => {
      const rect = element.getBoundingClientRect();
      const role = element.getAttribute('role');
      const style = window.getComputedStyle(element);
      return rect.left < -1 || rect.right > window.innerWidth + 1 || element.scrollWidth > element.clientWidth + 1 || !['alert', 'status'].includes(role) || (style.position === 'fixed' && rect.top < 0) ? [{ text: element.textContent?.trim().replace(/\s+/g, ' ').slice(0, 100), role, left: Math.round(rect.left), right: Math.round(rect.right), scrollWidth: element.scrollWidth, clientWidth: element.clientWidth }] : [];
    });
    const longValueSelectors = ['code[title]', 'code', 'pre', '.managed-card p', '.managed-card small', '.managed-model-list small', '.memory-card p', '.memory-card strong', '.memory-card small', '.app-drawer p', '[data-long-value]'];
    const longValueCandidates = [...new Set(longValueSelectors.flatMap((selector) => [...main.querySelectorAll(selector)]))].filter((element) => visible(element) && ((element.textContent ?? '').trim().length >= 48 || (element.getAttribute('title') ?? '').length >= 48));
    const overflowIssues = longValueCandidates.flatMap((element) => {
      const rect = element.getBoundingClientRect();
      const style = window.getComputedStyle(element);
      const bounded = style.textOverflow === 'ellipsis' || style.overflowWrap !== 'normal' || style.wordBreak !== 'normal' || ['auto', 'scroll'].includes(style.overflowX);
      return rect.right > window.innerWidth + 1 || (element.scrollWidth > element.clientWidth + 1 && !bounded) ? [{ tag: element.tagName.toLowerCase(), className: String(element.className), text: (element.textContent ?? '').trim().replace(/\s+/g, ' ').slice(0, 120), right: Math.round(rect.right), scrollWidth: element.scrollWidth, clientWidth: element.clientWidth }] : [];
    });
    return { clippedControls, controls: controls.length, coveredControls, duplicateIds, feedbackIssues, longValueCandidates: longValueCandidates.length, overflowIssues, shreddedTextControls, textCollisionIssues, textProblems, undersizedControls, unlabeledControls, verticalSpillIssues };
  });
}

export async function assertPageQuality(page, label, variant) {
  const layout = await layoutReport(page);
  const quality = await qualityReport(page);
  const suffix = ` at ${variantLabel(variant.viewport, variant.zoom)}`;
  assert(layout.documentWidth <= layout.documentClientWidth + 1, `${label} creates horizontal scrolling${suffix}: ${layout.documentWidth}px in a ${layout.documentClientWidth}px content box; offenders: ${JSON.stringify(layout.offenders)}.`);
  assert(layout.offenders.length === 0, `${label} has elements outside the viewport${suffix}: ${JSON.stringify(layout.offenders)}`);
  assert(layout.topLevelContainmentIssues.length === 0, `${label} has a top-level surface painted outside main${suffix}: ${JSON.stringify(layout.topLevelContainmentIssues)}`);
  assert(layout.shellContainmentIssues.length === 0, `${label} has shell content clipped outside the viewport${suffix}: ${JSON.stringify(layout.shellContainmentIssues)}`);
  assert(quality.undersizedControls.length === 0, `${label} has targets below 44px${suffix}: ${JSON.stringify(quality.undersizedControls)}`);
  assert(quality.shreddedTextControls.length === 0, `${label} has vertically shredded control text${suffix}: ${JSON.stringify(quality.shreddedTextControls)}`);
  assert(quality.unlabeledControls.length === 0, `${label} has unlabeled controls${suffix}: ${JSON.stringify(quality.unlabeledControls)}`);
  assert(quality.duplicateIds.length === 0, `${label} has duplicate IDs${suffix}: ${JSON.stringify(quality.duplicateIds)}`);
  assert(quality.textProblems.length === 0, `${label} has unreadable text geometry${suffix}: ${JSON.stringify(quality.textProblems)}`);
  assert(quality.feedbackIssues.length === 0, `${label} has invalid feedback-banner geometry or semantics${suffix}: ${JSON.stringify(quality.feedbackIssues)}`);
  assert(quality.overflowIssues.length === 0, `${label} has unhandled long-value overflow${suffix}: ${JSON.stringify(quality.overflowIssues)}`);
  assert(quality.verticalSpillIssues.length === 0, `${label} has children spilling below non-scrolling parents${suffix}: ${JSON.stringify(quality.verticalSpillIssues)}`);
  assert(quality.clippedControls.length === 0, `${label} has controls cut off by an ancestor that hides its overflow, with no scrollbar to recover them${suffix}: ${JSON.stringify(quality.clippedControls)}`);
  assert(quality.coveredControls.length === 0, `${label} has controls painted over by other page content${suffix}: ${JSON.stringify(quality.coveredControls)}`);
  assert(quality.textCollisionIssues.length === 0, `${label} has text colliding with the text beside it${suffix}: ${JSON.stringify(quality.textCollisionIssues)}`);
  return { layout, quality };
}

/**
 * Reachability for the controls a screen exists to offer: scroll each one into
 * view, then require that it lands inside the viewport and hit-tests to itself.
 * Containment checks alone pass happily while a control sits outside the box
 * that is supposed to hold it.
 */
export async function assertKeyControlsReachable(page, label, variant, selectors) {
  const report = await page.evaluate((list) => list.map((selector) => {
    const element = document.querySelector(selector);
    if (!element) return { selector, present: false };
    element.scrollIntoView({ block: 'center', inline: 'nearest' });
    const rect = element.getBoundingClientRect();
    const hit = document.elementFromPoint(rect.left + rect.width / 2, rect.top + rect.height / 2);
    return {
      selector,
      present: true,
      inside: rect.top >= -1 && rect.bottom <= window.innerHeight + 1 && rect.left >= -1 && rect.right <= window.innerWidth + 1,
      reachable: Boolean(hit) && (hit === element || element.contains(hit) || hit.contains(element)),
      covering: hit && hit !== element ? `${hit.tagName.toLowerCase()}.${String(hit.className)}` : null,
      rect: { top: Math.round(rect.top), bottom: Math.round(rect.bottom), left: Math.round(rect.left), right: Math.round(rect.right) },
      viewport: { width: window.innerWidth, height: window.innerHeight },
    };
  }), selectors);
  const unreachable = report.filter((item) => !item.present || !item.inside || !item.reachable);
  assert(unreachable.length === 0, `${label} has key controls that cannot be reached at ${variantLabel(variant.viewport, variant.zoom)}: ${JSON.stringify(unreachable)}`);
  return report;
}

export async function assertRequiredSurface(page, label, selectors) {
  for (const selector of selectors) assert(await page.locator(selector).count() > 0, `${label} surface is missing: ${selector}.`);
}
