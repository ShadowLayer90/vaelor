/**
 * Responsive acceptance: the run itself.
 *
 * This file owns the appliance target and nothing else - starting a disposable
 * preview, authenticating, proving the fixtures the run depends on are actually
 * present, watching for console and network failures, and printing the report.
 * What gets looked at lives in `responsive-surfaces.mjs`, what can be noticed
 * lives in `responsive-detectors.mjs`, and whether the instruments still read
 * lives in `responsive-self-test.mjs`.
 *
 *   npm run acceptance:responsive          full run against a preview
 *   npm run acceptance:self-test           the detectors alone
 *
 * The self-test needs a browser and nothing else: no appliance, no Python, no
 * fixtures. That is why it is the first thing both entry points do, and why
 * `npm run check` runs it too - it answers the one question a green run cannot
 * answer about itself, at the cost of twelve synthetic pages.
 *
 * A product pass reported by a harness that has gone blind is worse than no
 * report, because it will be believed. The zoom regression that shipped was
 * invisible for two independent reasons, and only one of them was in the app.
 */

import { spawn } from 'node:child_process';
import { readFile } from 'node:fs/promises';
import path from 'node:path';
import process from 'node:process';
import { chromium } from 'playwright-core';
import {
  baseUrl, capture, captureEvidence, chrome, evidenceRoot, password,
  prepareEvidenceDirectory, previewPort, python, repositoryRoot, selfTestOnly,
  startPreview, username,
} from './responsive-config.mjs';
import { assertPageQuality } from './responsive-detectors.mjs';
import { setViewport, viewports, zoomVariants } from './responsive-matrix.mjs';
import { assert, variantLabel } from './responsive-report.mjs';
import { assertDetectorsFire } from './responsive-self-test.mjs';
import { runVariant, setPreviewDiagnostic } from './responsive-surfaces.mjs';

const operationEvidenceFixture = JSON.parse(await readFile(
  new URL('../src/test/fixtures/operation-projection-v1.json', import.meta.url),
  'utf8',
));
const operationAuditFixture = {
  events: [{
    id: 417,
    action: 'job.create',
    actor: 'responsive-admin',
    created_at: 1785711509,
    details: {
      deduplicated: false,
      display_identity: 'nginx-welcome',
      resource_id: 'nginx-welcome',
      type: 'compose.backup',
    },
    remote_addr: '192.168.0.72',
    result: 'success',
    target: 'job-42',
  }],
  operation_id: operationEvidenceFixture.operation_id,
  schema: 'vaelor.operation.v1',
};

let server;
let browser;
let zoomContext;
let serverError = '';
setPreviewDiagnostic(() => serverError);

async function stopPreview() {
  if (!server) return;
  const exited = new Promise((resolve) => server.once('exit', resolve));
  if (server.exitCode === null) server.kill();
  await Promise.race([exited, new Promise((resolve) => setTimeout(resolve, 2_000))]);
  server.stdout?.destroy();
  server.stderr?.destroy();
  server.unref();
}

async function installOperationEvidenceFixture(context) {
  await context.route(/\/api\/v2\/operations\?limit=50$/, (route) => route.fulfill({
    contentType: 'application/json',
    body: JSON.stringify({ ok: true, data: { operations: [operationEvidenceFixture] } }),
  }));
  await context.route(/\/api\/v2\/operations\/jobs:job-42\/audit$/, (route) => route.fulfill({
    contentType: 'application/json',
    body: JSON.stringify({ ok: true, data: operationAuditFixture }),
  }));
}

await prepareEvidenceDirectory();
if (startPreview) {
  server = spawn(python, ['-m', 'examples.preview_v2'], {
    cwd: repositoryRoot,
    env: {
      ...process.env,
      VAELOR_ACCEPTANCE_FIXTURES: '1',
      VAELOR_ACCEPTANCE_USERNAME: username,
      VAELOR_PREVIEW_PORT: String(previewPort),
      SystemRoot: process.env.SystemRoot ?? 'C:\\Windows',
      WINDIR: process.env.WINDIR ?? 'C:\\Windows',
      PATH: `${path.dirname(python)};${process.env.SystemRoot ?? 'C:\\Windows'}\\System32`,
    },
    stdio: ['ignore', 'pipe', 'pipe'],
    windowsHide: true,
  });
  server.stderr.on('data', (chunk) => { serverError += chunk.toString(); });
}

async function waitForTarget() {
  const deadline = Date.now() + 15_000;
  while (Date.now() < deadline) {
    try {
      if ((await fetch(`${baseUrl}/api/v2/auth/status`)).ok) return;
    } catch {
      // The target is still starting.
    }
    await new Promise((resolve) => setTimeout(resolve, 150));
  }
  throw new Error(`Acceptance target did not become ready at ${baseUrl}.${serverError ? `\n${serverError}` : ''}`);
}

function extractData(body) {
  return body && typeof body === 'object' && 'data' in body ? body.data : undefined;
}

async function apiGet(context, endpoint) {
  const response = await context.request.get(`${baseUrl}/api/v2${endpoint}`);
  const body = await response.json().catch(() => ({}));
  return { response, body, data: extractData(body) };
}

async function authenticate(context) {
  const status = await apiGet(context, '/auth/status');
  assert(status.response.ok(), `Auth status failed with HTTP ${status.response.status()}.`);
  if (status.data?.bootstrap_required) {
    const bootstrap = await context.request.post(`${baseUrl}/api/v2/auth/bootstrap`, { data: { username, password } });
    assert(bootstrap.ok(), `Bootstrap failed with HTTP ${bootstrap.status()}.`);
  }
  const login = await context.request.post(`${baseUrl}/api/v2/auth/login`, { data: { username, password } });
  const body = await login.json().catch(() => ({}));
  assert(login.ok(), `Login failed with HTTP ${login.status()}.`);
  const csrfToken = body?.data?.csrf_token ?? body?.data?.session?.csrf_token;
  assert(typeof csrfToken === 'string' && csrfToken.length > 0, 'Login did not return a CSRF token.');
  return csrfToken;
}

async function acceptancePrerequisites(context) {
  const [profiles, memories, inventory, agentStatus] = await Promise.all([apiGet(context, '/assistant/profiles'), apiGet(context, '/assistant/memories'), apiGet(context, '/managed'), apiGet(context, '/agent/status')]);
  const missing = [];
  if (!profiles.response.ok() || !Array.isArray(profiles.data)) missing.push('assistant task/profile store and administrator access');
  if (!memories.response.ok() || !Array.isArray(memories.data) || memories.data.length < 2) missing.push(`at least two disposable Memory card fixtures (HTTP ${memories.response.status()}, count ${Array.isArray(memories.data) ? memories.data.length : 'invalid'})`);
  if (!inventory.response.ok() || !inventory.data?.apps?.some((item) => item.managed)) missing.push('at least one managed workload card fixture');
  if (!inventory.response.ok() || !inventory.data?.models?.length) missing.push('at least one downloaded model/path fixture');
  const managedLongValues = [...(inventory.data?.apps ?? []), ...(inventory.data?.models ?? [])].flatMap((item) => [item.name, item.image, item.file, item.path, item.digest, item.manifest_digest]).filter((value) => String(value ?? '').length >= 48);
  if (!managedLongValues.length) missing.push('long managed path or digest fixture');
  if (!agentStatus.response.ok() || !agentStatus.data?.configured) missing.push('a configured Assistant model runtime');
  const operationalSpecialist = profiles.data?.find((item) => !item.custom && item.operational);
  const customAgent = profiles.data?.find((item) => item.custom && item.enabled);
  if (!operationalSpecialist) missing.push('one operational built-in specialist profile');
  if (!customAgent) missing.push('one enabled custom-agent fixture for Test run');
  assert(missing.length === 0, `Responsive acceptance prerequisites are missing: ${missing.join(', ')}. Use a disposable authenticated target with the named PRE-02 fixtures; the bundled preview lacks assistant stores and a model runtime.`);
  return { customAgent, inventory: inventory.data, memories: memories.data, operationalSpecialist };
}

function attachRuntimeGuards(page, state) {
  page.on('console', (message) => {
    if (message.type() === 'error' && !message.text().startsWith('Failed to load resource:')) state.consoleErrors.push(message.text());
  });
  page.on('requestfailed', (request) => { state.failedRequests.push(`${request.method()} ${request.url()} ${request.failure()?.errorText ?? ''}`); });
  page.on('response', (response) => {
    if (response.status() === 503) state.expectedUnavailable.push(response.url());
    else if (response.status() >= 500) void response.text().then((body) => state.serverResponses.push(`${response.status()} ${response.url()} ${body.slice(0, 500)}`));
    else if (response.status() >= 400) state.clientResponses.push(`${response.status()} ${response.url()}`);
  });
}

async function launchBrowser() {
  try {
    return await chromium.launch({ executablePath: chrome, headless: true, args: ['--disable-gpu', '--no-first-run'] });
  } catch (error) {
    throw new Error(
      `The acceptance browser could not be started at ${chrome}. Set `
      + `VAELOR_ACCEPTANCE_CHROME to a Chrome or Chromium executable. `
      + `Underlying error: ${error instanceof Error ? error.message : String(error)}`,
    );
  }
}

try {
  /*
   * The instruments are proven first, unconditionally, and before the target
   * is so much as contacted.
   *
   * This used to sit after `waitForTarget()`, which put a product-side
   * prerequisite ahead of the question "can this harness still see anything".
   * A preview that fails to start then ends the run with "target did not
   * become ready" and no word about the detectors - so the one condition that
   * invalidates every later claim is the one the run never gets round to
   * checking. It is twelve synthetic pages and no appliance; there is nothing
   * it needs to wait for.
   */
  browser = await launchBrowser();
  const detectors = await assertDetectorsFire(browser);
  if (selfTestOnly) {
    console.log(JSON.stringify({ selfTest: true, passed: true, detectors }, null, 2));
  } else {
  await waitForTarget();
  const context = await browser.newContext({ viewport: { width: 375, height: 812 } });
  const page = await context.newPage();
  const runtime = { clientResponses: [], consoleErrors: [], expectedUnavailable: [], failedRequests: [], serverResponses: [] };
  attachRuntimeGuards(page, runtime);
  await page.goto(`${baseUrl}/v2/`, { waitUntil: 'networkidle' });
  await page.getByRole('heading', { name: 'Commission this node' }).waitFor();
  const authMatrix = [];
  for (const viewport of viewports) {
    const variant = { viewport, zoom: false };
    await setViewport(page, variant);
    const result = await assertPageQuality(page, 'Entry screen', variant);
    authMatrix.push({ viewport: variantLabel(viewport), documentWidth: result.layout.documentWidth, controls: result.quality.controls });
    await capture(page, `entry-${viewport.width}x${viewport.height}`);
  }
  const csrfToken = await authenticate(context);
  const prerequisites = await acceptancePrerequisites(context);
  await installOperationEvidenceFixture(context);
  await page.goto(`${baseUrl}/v2/`, { waitUntil: 'networkidle' });
  await page.getByRole('heading', { level: 1, name: 'Home' }).waitFor();
  const normalResults = [];
  for (const viewport of viewports) normalResults.push(await runVariant(page, { viewport, zoom: false }, prerequisites));
  const storageState = await context.storageState();
  await context.close();
  zoomContext = await browser.newContext({ deviceScaleFactor: 2, storageState, viewport: { width: zoomVariants[0].viewport.width, height: zoomVariants[0].viewport.height } });
  await installOperationEvidenceFixture(zoomContext);
  const zoomPage = await zoomContext.newPage();
  await zoomPage.goto(`${baseUrl}/v2/`, { waitUntil: 'networkidle' });
  await zoomPage.getByRole('heading', { level: 1, name: 'Home' }).waitFor();
  attachRuntimeGuards(zoomPage, runtime);
  const zoomResults = [];
  for (const variant of zoomVariants) zoomResults.push(await runVariant(zoomPage, variant, prerequisites));
  await new Promise((resolve) => setTimeout(resolve, 250));
  const tracebackStart = serverError.lastIndexOf('Traceback');
  const traceback = tracebackStart >= 0 ? serverError.slice(tracebackStart, tracebackStart + 8_000) : serverError.slice(-8_000);
  assert(runtime.serverResponses.length === 0, `Unexpected server errors: ${JSON.stringify(runtime.serverResponses)}${traceback ? `\nPreview diagnostic:\n${traceback}` : ''}`);
  assert(runtime.clientResponses.every((item) => item.endsWith('/favicon.ico')), `Unexpected 4xx responses: ${JSON.stringify(runtime.clientResponses)}`);
  assert(runtime.failedRequests.length === 0, `Failed browser requests: ${JSON.stringify(runtime.failedRequests)}`);
  assert(runtime.consoleErrors.length === 0, `Frontend console errors: ${JSON.stringify(runtime.consoleErrors)}`);
  console.log(JSON.stringify({ passed: true, baseUrl, detectors, evidenceRoot: captureEvidence ? evidenceRoot : null, authMatrix, normalResults, zoomResults, browserZoom: zoomVariants.map((variant) => variantLabel(variant.viewport, variant.zoom)), clientResponses: runtime.clientResponses, consoleErrors: runtime.consoleErrors, expectedUnavailableResponses: [...new Set(runtime.expectedUnavailable)], failedRequests: runtime.failedRequests, serverResponses: runtime.serverResponses, csrfTokenPresent: Boolean(csrfToken) }, null, 2));
  }
} finally {
  await zoomContext?.close();
  await browser?.close();
  await stopPreview();
}
