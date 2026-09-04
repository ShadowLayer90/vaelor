/**
 * The product screens, and what each of them has to still be true.
 *
 * Everything here knows about Vaelor: destination names, tab labels, the
 * selectors a screen is made of, and the specific defect each assertion was
 * added for. The detectors in `responsive-detectors.mjs` know none of that -
 * they measure a page. This file is where the two meet, once per screen per
 * viewport, and it is the file that changes when the app changes.
 */

import {
  assertKeyControlsReachable, assertPageQuality, assertRequiredSurface,
} from './responsive-detectors.mjs';
import {
  MOBILE_RAIL_MAX_WIDTH, destinationName, routes, setViewport,
} from './responsive-matrix.mjs';
import { assert, variantLabel } from './responsive-report.mjs';
import { capture } from './responsive-config.mjs';

/*
 * The preview's stderr belongs to the runner, but the one place it is worth
 * reading is here: a custom-agent Run trigger that stays disabled almost always
 * means the preview raised on an assistant endpoint, and the traceback is the
 * only useful thing to print. The runner installs the accessor rather than this
 * file importing the runner, so the dependency stays one way.
 */
let previewDiagnostic = () => '';
export function setPreviewDiagnostic(accessor) { previewDiagnostic = accessor; }

async function waitForUsable(page) {
  await page.locator('#main-content').waitFor({ state: 'visible' });
  const loading = page.locator('.page-loading');
  if (await loading.count()) await loading.waitFor({ state: 'hidden', timeout: 10_000 });
  await page.waitForTimeout(180);
}

/*
 * Scoped to the navigation rail on purpose. Home now offers quick actions
 * carrying the same canonical destination names, so "exactly one button called
 * Cluster on the page" stopped being true - correctly, since both are meant to
 * take you to the same place.
 */
async function visibleNavButton(page, name) {
  /*
   * The rail renders both a desktop and a mobile list, one of which is
   * display:none. Target the rendered one directly: filtering a locator that
   * spans both lists resolved to nothing at some viewports.
   */
  const rail = page.locator('.sidebar__nav--desktop, .sidebar__nav--mobile')
    .locator('visible=true');
  const locator = rail.getByRole('button', { name, exact: true }).first();
  const count = await rail.getByRole('button', { name, exact: true }).count();
  if (count < 1) {
    const pageWide = await page.getByRole('button', { name, exact: true })
      .filter({ visible: true }).count();
    const present = await page.locator('.sidebar button').evaluateAll(
      (nodes) => nodes.map((node) => `${node.getAttribute('aria-label') ?? node.textContent?.trim()}${node.offsetParent ? '' : ' [hidden]'}`),
    );
    assert(false, `Expected one visible ${name} button in the navigation rail, `
      + `found ${count} there and ${pageWide} on the page. Rail contains: ${JSON.stringify(present)}`);
  }
  return locator;
}

async function waitForEnabled(locator, timeoutMs = 10_000) {
  const deadline = Date.now() + timeoutMs;
  while (Date.now() < deadline) {
    if (await locator.isEnabled()) return true;
    await new Promise((resolve) => setTimeout(resolve, 100));
  }
  return false;
}

/*
 * The composer's problem-area refinement is populated from `/assistant/profiles`
 * after first paint, and the composer is deliberately usable on Automatic while
 * that is in flight. Asserting on the first render therefore raced the fetch and
 * failed on timing rather than on layout.
 */
async function waitForOptions(locator, minimum = 2, timeoutMs = 10_000) {
  const deadline = Date.now() + timeoutMs;
  while (Date.now() < deadline) {
    if (await locator.locator('option').count() >= minimum) return true;
    await new Promise((resolve) => setTimeout(resolve, 100));
  }
  return false;
}

async function assistantReadinessDiagnostic(page) {
  return page.evaluate(async () => {
    const endpoints = ['/api/v2/agent/status', '/api/v2/assistant/preferences', '/api/v2/assistant/profiles', '/api/v2/assistant/tasks', '/api/v2/assistant/handoff-targets', '/api/v2/copilot/setup'];
    return Promise.all(endpoints.map(async (endpoint) => {
      const response = await fetch(endpoint);
      return { endpoint, status: response.status, body: (await response.text()).slice(0, 300) };
    }));
  });
}

async function navigate(page, name, viewportWidth) {
  const started = performance.now();
  // The mobile rail shows the first four destinations and folds the rest into
  // "More"; deriving that from the same table keeps it correct when the
  // destination names change.
  const overflowNames = routes.slice(4).map(([, label]) => label);
  if (viewportWidth <= MOBILE_RAIL_MAX_WIDTH && overflowNames.includes(name)) {
    const more = page.getByLabel('More navigation', { exact: true }).filter({ visible: true });
    assert(await more.count() === 1, 'Mobile More navigation is unavailable.');
    await more.click();
  }
  await (await visibleNavButton(page, name)).click();
  await waitForUsable(page);
  return Math.round(performance.now() - started);
}

/*
 * Troubleshoot stopped being a tab and a modal. It is the same question box as
 * Ask, with two optional refinements below it, so what this checks is that the
 * refinements are reachable and default to letting Vaelor decide - not that a
 * second screen still opens.
 *
 * They sit under the composer inside a disclosure now: asking somebody to
 * categorise a problem before they have stated it is the wrong order, and both
 * controls default to the do-nothing value. The disclosure has to be opened
 * before either can be reached, which is the point.
 */
async function exerciseAskComposer(page, variant) {
  await page.getByRole('tab', { name: 'Ask', exact: true }).click();
  await page.getByRole('heading', { name: 'What do you want to know?' }).waitFor();
  await assertRequiredSurface(page, 'Ask composer', [
    '.assistant-refinement-disclosure', '.assistant-chat__composer', '.capability-strip',
    '#assistant-problem-area', '#assistant-durable',
  ]);
  const disclosure = page.locator('.assistant-refinement-disclosure');
  assert(
    await disclosure.evaluate((element) => !element.open),
    'The refinement disclosure must start closed so the question comes first.',
  );
  const composerBox = await page.locator('.assistant-chat__composer').boundingBox();
  const disclosureBox = await disclosure.boundingBox();
  assert(
    disclosureBox.y >= composerBox.y,
    `The refinement must sit below the question box, got composer y=${Math.round(composerBox.y)} and refinement y=${Math.round(disclosureBox.y)}.`,
  );
  await disclosure.locator('summary').click();
  const problemArea = page.locator('#assistant-problem-area');
  if (!(await waitForOptions(problemArea))) {
    const diagnostic = await assistantReadinessDiagnostic(page);
    assert(false, `The problem-area refinement never offered an appliance area: ${JSON.stringify(await problemArea.locator('option').allTextContents())} ${JSON.stringify(diagnostic)}`);
  }
  const defaultArea = await problemArea.inputValue();
  if (defaultArea !== '') {
    const diagnostic = await assistantReadinessDiagnostic(page);
    assert(false, `The problem-area refinement must default to Automatic, got "${defaultArea}": ${JSON.stringify(diagnostic)}`);
  }
  const options = await problemArea.locator('option').allTextContents();
  assert(options[0] === 'Automatic', `The problem-area refinement must lead with Automatic: ${JSON.stringify(options)}.`);
  assert(await page.locator('#assistant-durable:checked').count() === 0, 'The re-runnable check option must start unchecked.');
  const geometry = await page.locator('.assistant-chat__composer').evaluate((element) => {
    const textarea = element.querySelector('textarea');
    const bounds = textarea?.getBoundingClientRect();
    return {
      textareaHeight: Math.round(bounds?.height ?? 0),
      textareaWidth: Math.round(bounds?.width ?? 0),
      viewportWidth: window.innerWidth,
    };
  });
  /*
   * Both halves of the old dialog's field-size guard, re-baselined for the
   * inline composer. Only the width half survived the move, and a floor that is
   * measured but not asserted is not a floor: the height was free to fall to
   * zero with nothing but a reported number to show for it.
   *
   * 92px is `min-height` on `.assistant-chat__composer textarea` in
   * `src/styles/assistant-chat.css`, and 92 is what this measures at every
   * viewport in the matrix. 88 leaves the sub-pixel room a zoomed viewport can
   * cost without leaving room for the field to shrink meaningfully.
   */
  assert(geometry.textareaHeight >= 88, `Ask composer request field is unusably short: ${JSON.stringify(geometry)}.`);
  assert(geometry.textareaWidth >= Math.min(280, variant.viewport.width - 64), `Ask composer request field is unusably narrow: ${JSON.stringify(geometry)}.`);
  await assertKeyControlsReachable(page, 'Ask composer', variant, [
    '.assistant-chat__composer textarea',
    '.assistant-chat__composer button',
    '#assistant-problem-area',
    '#assistant-durable',
  ]);
  await assertPageQuality(page, 'Ask composer', variant);
  /*
   * The transcript is not allowed to be the one box on this screen with no
   * room. It measured 0px of visible height at 1280x720 and at 1024x768 inside
   * a height-capped panel wrapped around a second scroller; the page is the
   * only scroller now, so the stream is as tall as its own content.
   */
  const transcript = await page.locator('.assistant-chat__stream').evaluate((element) => ({
    visible: Math.round(element.getBoundingClientRect().height),
    content: element.scrollHeight,
    panelOverflow: window.getComputedStyle(element.parentElement).overflowY,
    streamOverflow: window.getComputedStyle(element).overflowY,
  }));
  assert(
    transcript.visible >= transcript.content - 2,
    `The Assistant transcript is clipped by a nested scroller: ${JSON.stringify(transcript)}.`,
  );
  assert(
    transcript.panelOverflow === 'visible' && transcript.streamOverflow === 'visible',
    `The Assistant transcript must not own a scroller: ${JSON.stringify(transcript)}.`,
  );
  // The archive is its own tab now: Ask carries the question and the answer.
  assert(
    await page.locator('.assistant-run-history').count() === 0,
    'The run history must not be rendered on Ask.',
  );
  return { surface: 'ask-composer', geometry, options, transcript };
}

/*
 * History: what has already run. One heading pair, one control band, and the
 * same four filters that used to sit at the bottom of Ask.
 */
async function exerciseRunHistory(page, variant) {
  await page.getByRole('tab', { name: 'History', exact: true }).click();
  await page.locator('.assistant-run-history').waitFor();
  await assertRequiredSurface(page, 'Run history', [
    '.assistant-run-history__filters', '.agent-task-board .section-heading',
  ]);
  const bands = await page.locator('.assistant-run-history .section-heading').count();
  assert(bands === 1, `The run history must carry one heading pair, found ${bands}.`);
  assert(
    await page.getByText('What Vaelor has run here').count() === 0,
    'The duplicate run-history heading is still rendered.',
  );
  for (const filter of ['All', 'Checks', 'Agent runs', 'Automatic']) {
    const control = page.locator('.assistant-run-history__filters button')
      .filter({ hasText: new RegExp(`^${filter} \\(\\d+\\)$`) });
    assert(await control.count() === 1, `Run-history filter "${filter}" is missing.`);
    await control.click();
    await assertPageQuality(page, `Run history · ${filter}`, variant);
  }
  await page.locator('.assistant-run-history__filters button').first().click();
  return { surface: 'run-history', bands };
}

async function exerciseTestRunDialog(page, variant) {
  await page.getByRole('tab', { name: 'Routines', exact: true }).click();
  await page.getByRole('heading', { name: 'Build and manage custom agents' }).waitFor();
  /*
   * Two actions stay on the card and the other five moved into an overflow
   * disclosure, so the primary one is now "Run" rather than "Run agent".
   */
  const card = page.locator('.custom-agent-card').first();
  const openActions = await card.locator('.custom-agent-card__actions > .ui-button-wrap button').count();
  assert(openActions <= 2, `A custom-agent card must expose at most two actions in the open, found ${openActions}.`);
  assert(
    await card.locator('.custom-agent-card__menu-items button').count() >= 4,
    'The custom-agent overflow menu lost the actions it was meant to keep.',
  );
  const trigger = card.getByRole('button', { name: 'Run', exact: true }).first();
  if (!(await waitForEnabled(trigger))) {
    const diagnostic = await assistantReadinessDiagnostic(page);
    const serverError = previewDiagnostic();
    const traceStart = serverError.lastIndexOf('Traceback');
    const trace = traceStart >= 0 ? serverError.slice(traceStart, traceStart + 8_000) : '';
    assert(false, `Custom-agent Run trigger remained disabled after Assistant data loaded: ${JSON.stringify(diagnostic)}${trace ? `\nPreview diagnostic:\n${trace}` : ''}`);
  }
  /*
   * Nobody asked for the credential wizard. It was appended permanently for
   * anybody who owned a single agent - about 1,400px and twenty controls.
   */
  assert(
    await page.locator('.custom-agent-app-access-panel').count() === 0,
    'The app-access wizard is rendered without being asked for.',
  );
  await trigger.focus();
  await trigger.click();
  const dialog = page.getByRole('dialog', { name: /^Run / });
  await dialog.waitFor();
  await assertRequiredSurface(page, 'Test-run dialog', ['.custom-agent-test-dialog', `button[aria-label='Close test run']`, 'textarea']);
  await assertPageQuality(page, 'Test-run dialog', variant);
  assert(await dialog.getByRole('button', { name: 'Run now', exact: true }).count() === 1, 'Agent run dialog is missing Run now.');
  await page.getByLabel('Close test run', { exact: true }).click();
  await dialog.waitFor({ state: 'detached' });
  return { surface: 'test-run-dialog' };
}

async function exerciseCustomApplicationRequest(page, variant) {
  const trigger = page.getByRole('button', { name: 'Describe a custom application', exact: true });
  await trigger.click();
  await page.getByRole('heading', { name: 'What should Vaelor deploy?' }).waitFor();
  const textarea = page.getByLabel('Describe the application and how you expect to use it');
  const submit = page.getByRole('button', { name: 'Research and prepare plan', exact: true });
  const geometry = await page.locator('.application-deployment__request-form').evaluate((element) => {
    const field = element.querySelector('textarea');
    const action = element.querySelector('button');
    const fieldBounds = field?.getBoundingClientRect();
    const actionBounds = action?.getBoundingClientRect();
    return {
      buttonTop: actionBounds?.top ?? 0,
      textareaBottom: fieldBounds?.bottom ?? 0,
      textareaHeight: fieldBounds?.height ?? 0,
      textareaWidth: fieldBounds?.width ?? 0,
    };
  });
  assert(await textarea.count() === 1 && await submit.count() === 1, 'Custom application request controls are missing.');
  assert(geometry.textareaHeight >= 180, `Custom application request field is too short: ${JSON.stringify(geometry)}.`);
  assert(geometry.textareaWidth >= Math.min(480, variant.viewport.width - 64), `Custom application request field is too narrow: ${JSON.stringify(geometry)}.`);
  assert(geometry.buttonTop >= geometry.textareaBottom, `Custom application submit action overlaps or sits beside the request field: ${JSON.stringify(geometry)}.`);
  await assertPageQuality(page, 'Custom application request', variant);
  await page.getByRole('button', { name: 'Back to Workloads', exact: true }).click();
  await page.locator('.workload-product-grid').waitFor();
  return { surface: 'custom-application-request', geometry };
}

/*
 * Memory left the Assistant. The same store answers both AI surfaces, so it is
 * its own `#/memory` route reached from the composer chip rather than a tab
 * that implied the Assistant owned it.
 */
async function exerciseMemoryCards(page, variant) {
  await page.getByRole('tab', { name: 'Ask', exact: true }).click();
  const chip = page.locator('.capability-strip .capability-chip--link');
  await chip.waitFor({ state: 'visible', timeout: 10_000 });
  const target = await chip.getAttribute('href');
  assert(target === '#/memory', `The memory chip must open #/memory, got ${JSON.stringify(target)}.`);
  await chip.click();
  await waitForUsable(page);
  await page.getByRole('heading', { level: 1, name: /^What Vaelor remembers/ }).waitFor();
  const heading = await page.getByRole('heading', { level: 1 }).first().textContent();
  assert(
    /used by both/.test(heading ?? '') && /AI Chat/.test(heading ?? ''),
    `The memory page must say both AI surfaces use it: ${JSON.stringify(heading)}.`,
  );
  const cards = page.locator('.memory-card');
  await cards.first().waitFor({ state: 'visible', timeout: 10_000 });
  assert(await cards.count() >= 2, 'Memory card fixtures are missing after navigation.');
  await assertRequiredSurface(page, 'Memory cards', ['.memory-card__meta', '.memory-card__actions']);
  const result = await assertPageQuality(page, 'Memory cards', variant);
  assert(result.quality.longValueCandidates > 0, 'Memory card long-content fixture is missing.');
  const count = await cards.count();
  await page.getByRole('link', { name: 'Back to Assistant', exact: true }).click();
  await waitForUsable(page);
  await page.getByRole('tab', { name: 'Ask', exact: true }).waitFor();
  return { surface: 'memory-cards', cards: count };
}

async function exerciseManagedCards(page, variant) {
  await page.getByRole('tab', { name: /^Manage/ }).click();
  await page.getByRole('heading', { name: 'Your services' }).waitFor();
  const cards = page.locator('.managed-card');
  await cards.first().waitFor({ state: 'visible', timeout: 10_000 });
  assert(await cards.count() > 0, 'Managed workload card fixtures are missing after navigation.');
  await assertRequiredSurface(page, 'Managed workload cards', ['.managed-card__head', '.managed-card__meta', '.managed-card button']);
  const result = await assertPageQuality(page, 'Managed workload cards', variant);
  assert(result.quality.longValueCandidates > 0, 'Managed workload long-path/image fixture is missing.');
  return { surface: 'managed-workload-cards', cards: await cards.count() };
}

async function exerciseAssistantAndChat(page, variant) {
  await page.getByRole('tab', { name: 'Ask', exact: true }).click();
  await page.getByRole('heading', { name: 'What do you want to know?' }).waitFor();
  await assertRequiredSurface(page, 'Assistant', ['.assistant-tabs', '.assistant-chat']);
  const assistant = await assertPageQuality(page, 'Assistant', variant);
  const composer = await assertKeyControlsReachable(page, 'Assistant composer', variant, [
    '.assistant-chat__composer textarea',
    '.assistant-chat__composer button',
  ]);
  await navigate(page, destinationName('ai-chat'), variant.viewport.width);
  await page.getByRole('heading', { name: 'AI Chat' }).waitFor();
  await assertRequiredSurface(page, 'AI Chat', ['.ai-chat-page', '.ai-chat-main']);
  const chat = await assertPageQuality(page, 'AI Chat', variant);
  // At real browser zoom the page is expected to scroll, so "the chat fits the
  // viewport exactly" is a 100%-only contract. Reachability under zoom is
  // covered by the clipping and occlusion checks, which run at every variant.
  if (variant.viewport.width > 720 && !variant.zoom) {
    const workspace = await page.locator('.ai-chat-workspace').evaluate((element) => {
      const bounds = element.getBoundingClientRect();
      return {
        bottom: Math.round(bounds.bottom),
        height: Math.round(bounds.height),
        top: Math.round(bounds.top),
        viewportHeight: window.innerHeight,
      };
    });
    assert(workspace.bottom <= workspace.viewportHeight + 1, `AI Chat workspace is clipped below the viewport at ${variantLabel(variant.viewport, variant.zoom)}: ${JSON.stringify(workspace)}.`);
    assert(chat.layout.documentHeight <= variant.viewport.height + 1, `AI Chat creates an unnecessary document scrollbar at ${variantLabel(variant.viewport, variant.zoom)}: ${chat.layout.documentHeight}px document for ${variant.viewport.height}px viewport.`);
  }
  return { surface: 'assistant-ai-chat', assistant, chat, composer };
}

async function exerciseOperationEvidenceModal(page, variant) {
  const buttons = page.getByRole('button', { name: 'View Activity evidence', exact: true });
  const buttonCount = await buttons.count();
  assert(buttonCount > 0, 'Activity fixtures do not expose operation evidence.');
  await buttons.first().click();
  const dialog = page.getByRole('dialog', { name: 'Activity evidence' });
  await dialog.waitFor({ state: 'visible' });
  const geometry = await dialog.evaluate((element) => {
    const bounds = element.getBoundingClientRect();
    return {
      bottom: Math.round(bounds.bottom),
      left: Math.round(bounds.left),
      right: Math.round(bounds.right),
      top: Math.round(bounds.top),
      viewportHeight: window.innerHeight,
      viewportWidth: window.innerWidth,
    };
  });
  assert(
    geometry.left >= -1 && geometry.right <= geometry.viewportWidth + 1
      && geometry.top >= -1 && geometry.bottom <= geometry.viewportHeight + 1,
    `Activity evidence modal escapes the viewport at ${variantLabel(variant.viewport, variant.zoom)}: ${JSON.stringify(geometry)}.`,
  );
  const recordedAction = dialog.getByText('Recorded action', { exact: true });
  await recordedAction.waitFor({ state: 'visible', timeout: 10_000 });
  assert(await recordedAction.count() > 0, 'Activity evidence modal does not render readable event cards.');
  assert(await dialog.locator('pre').count() === 0, 'Activity evidence modal exposes raw JSON before technical details are requested.');
  await dialog.getByRole('button', { name: 'Done', exact: true }).click();
  await dialog.waitFor({ state: 'hidden' });
  return { surface: 'operation-evidence-modal', geometry };
}

async function exerciseCaseLighting(page, variant) {
  await page.getByRole('tab', { name: 'Case lighting', exact: true }).click();
  const preview = page.locator('.lighting-preview');
  await preview.waitFor();
  const dynamicColor = await preview.evaluate((element) => window.getComputedStyle(element).getPropertyValue('--lighting-color').trim());
  const presetColor = await page.locator('.color-preset span').first().evaluate((element) => window.getComputedStyle(element).backgroundColor);
  assert(dynamicColor.length > 0, `Production CSP blocked the lighting preview CSSOM color at ${variantLabel(variant.viewport, variant.zoom)}.`);
  assert(presetColor !== 'rgba(0, 0, 0, 0)' && presetColor !== 'transparent', `Production CSP blocked the lighting preset CSSOM color at ${variantLabel(variant.viewport, variant.zoom)}.`);
  // The last channel field and the last preset sit on opposite sides of the
  // wrap point, which is where the colour grid used to slide underneath the
  // presets and lose two thirds of the Blue field to them.
  await assertKeyControlsReachable(page, 'Case lighting colour', variant, [
    '.color-field__hex',
    '.color-field__entry > .ui-field:last-child input',
    '.color-presets .ui-button-wrap:last-child button',
  ]);
  const colourRoom = await page.locator('.color-field__entry').evaluate((entry) => [...entry.querySelectorAll('input.ui-control')].map((input) => {
    const style = window.getComputedStyle(input);
    const width = input.getBoundingClientRect().width
      - Number.parseFloat(style.paddingLeft) - Number.parseFloat(style.paddingRight)
      - Number.parseFloat(style.borderLeftWidth) - Number.parseFloat(style.borderRightWidth);
    return { id: input.id, className: String(input.className), room: Math.round(width) };
  }));
  // A hex value needs about 60px; three digits need about 25px.
  const cramped = colourRoom.filter((field) => field.room < (field.className.includes('__hex') ? 60 : 25));
  assert(cramped.length === 0, `Case lighting colour fields have no room for their values at ${variantLabel(variant.viewport, variant.zoom)}: ${JSON.stringify(colourRoom)}`);
}

async function assertChatWorkspaceContainment(page, label, variant) {
  const workspaceContainment = await page.locator('.ai-chat-workspace').evaluate((workspace) => {
    const workspaceRect = workspace.getBoundingClientRect();
    const mainRect = workspace.querySelector('.ai-chat-main')?.getBoundingClientRect();
    const composerRect = workspace.querySelector('.ai-chat-composer')?.getBoundingClientRect();
    return {
      workspaceBottom: Math.round(workspaceRect.bottom),
      mainBottom: Math.round(mainRect?.bottom ?? 0),
      composerBottom: Math.round(composerRect?.bottom ?? 0),
    };
  });
  assert(
    workspaceContainment.mainBottom <= workspaceContainment.workspaceBottom + 1
      && workspaceContainment.composerBottom <= workspaceContainment.workspaceBottom + 1,
    `${label} lets the chat history rail push the conversation or composer outside the workspace at ${variantLabel(variant.viewport, variant.zoom)}: ${JSON.stringify(workspaceContainment)}`,
  );
}

async function walkEveryDestination(page, variant) {
  const routeMatrix = [];
  for (const [route, label] of routes) {
    const navigationMs = await navigate(page, label, variant.viewport.width);
    if (route === 'workloads') {
      const installTab = page.getByRole('tab', { name: 'Install', exact: true });
      if ((await installTab.getAttribute('aria-selected')) !== 'true') {
        await installTab.click();
        await page.locator('.workload-product-grid').waitFor();
      }
    }
    if (route === 'system') await exerciseCaseLighting(page, variant);
    const required = route === 'assistant' ? ['.assistant-tabs', '.assistant-chat'] : route === 'ai-chat' ? ['.ai-chat-page', '.ai-chat-main'] : route === 'workloads' ? ['.workloads-page', '.workload-product-grid'] : ['main'];
    await assertRequiredSurface(page, label, required);
    const result = await assertPageQuality(page, label, variant);
    if (route === 'ai-chat' && variant.viewport.width > 720) {
      await assertChatWorkspaceContainment(page, label, variant);
    }
    assert(navigationMs < 3_000, `${label} took ${navigationMs}ms to become usable at ${variantLabel(variant.viewport, variant.zoom)}.`);
    routeMatrix.push({ route, viewport: variantLabel(variant.viewport, variant.zoom), documentWidth: result.layout.documentWidth, controls: result.quality.controls, navigationMs });
    await capture(page, `${variant.zoom ? `zoom${Math.round(variant.zoom * 100)}-` : ''}${route}-${variant.viewport.width}x${variant.viewport.height}`);
  }
  return routeMatrix;
}

export async function runVariant(page, variant, prerequisites) {
  await setViewport(page, variant);
  const routeMatrix = await walkEveryDestination(page, variant);
  await navigate(page, destinationName('assistant'), variant.viewport.width);
  const assistant = await exerciseAssistantAndChat(page, variant);
  await navigate(page, destinationName('activity'), variant.viewport.width);
  const operationEvidence = await exerciseOperationEvidenceModal(page, variant);
  await navigate(page, destinationName('assistant'), variant.viewport.width);
  const specialist = await exerciseAskComposer(page, variant);
  const runHistory = await exerciseRunHistory(page, variant);
  const testRun = await exerciseTestRunDialog(page, variant);
  const memory = await exerciseMemoryCards(page, variant);
  await navigate(page, destinationName('workloads'), variant.viewport.width);
  const customApplication = await exerciseCustomApplicationRequest(page, variant);
  const managed = await exerciseManagedCards(page, variant);
  await capture(page, `${variant.zoom ? `zoom${Math.round(variant.zoom * 100)}-` : ''}target-surfaces-${variant.viewport.width}x${variant.viewport.height}`);
  return { routeMatrix, assistant, operationEvidence, specialist, runHistory, testRun, memory, customApplication, managed, prerequisites };
}
