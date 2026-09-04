/**
 * Where the harness points, what it runs against, and what it records.
 *
 * Everything here is decided once, from the environment, before any browser
 * opens: the target URL, the disposable preview port, the interpreter that can
 * start a preview, the credentials, and the evidence directory. Keeping it
 * apart from the run itself means `--self-test` can resolve exactly the two
 * things it needs (a browser and nowhere to write) without dying on an
 * interpreter it was never going to use.
 */

import { existsSync } from 'node:fs';
import { mkdir } from 'node:fs/promises';
import { createServer } from 'node:net';
import os from 'node:os';
import path from 'node:path';
import process from 'node:process';
import { assert } from './responsive-report.mjs';

const frontendRoot = process.cwd();
export const repositoryRoot = path.resolve(frontendRoot, '..');

/*
 * `--self-test` exercises the detectors against synthetic pages and nothing
 * else: no appliance target, no Python, no fixtures. It exists because a
 * harness that reports green is making two claims - "the app is fine" and "I
 * looked" - and only the first of them was ever checked. See
 * `responsive-self-test.mjs`.
 */
export const selfTestOnly = process.argv.includes('--self-test');
export const startPreview = !process.env.VAELOR_ACCEPTANCE_BASE_URL && !selfTestOnly;

async function availableLoopbackPort() {
  const probe = createServer();
  await new Promise((resolve, reject) => {
    probe.once('error', reject);
    probe.listen(0, '127.0.0.1', resolve);
  });
  const address = probe.address();
  const port = typeof address === 'object' && address ? address.port : 0;
  await new Promise((resolve, reject) => probe.close((error) => error ? reject(error) : resolve()));
  assert(port > 0, 'The operating system did not provide a disposable preview port.');
  return port;
}

export const previewPort = Number(process.env.VAELOR_ACCEPTANCE_PREVIEW_PORT ?? (startPreview ? await availableLoopbackPort() : 4173));
export const baseUrl = (process.env.VAELOR_ACCEPTANCE_BASE_URL ?? `http://127.0.0.1:${previewPort}`).replace(/\/$/, '');
export const evidenceRoot = path.resolve(process.env.VAELOR_ACCEPTANCE_EVIDENCE_DIR ?? path.join(os.tmpdir(), 'vaelor-responsive-acceptance'));
export const captureEvidence = process.env.VAELOR_ACCEPTANCE_CAPTURE_EVIDENCE === '1';

/*
 * A git worktree has no `.venv` of its own, so this resolved to a path that
 * does not exist and the harness died with ENOENT before testing anything -
 * which reads exactly like "the harness is broken" rather than "point me at an
 * interpreter". Walk up to the checkout that owns the environment.
 */
function resolvePython() {
  if (process.env.VAELOR_ACCEPTANCE_PYTHON) return process.env.VAELOR_ACCEPTANCE_PYTHON;
  const candidates = [];
  for (let directory = repositoryRoot, depth = 0; depth < 6; depth += 1) {
    candidates.push(path.join(directory, '.venv', 'Scripts', 'python.exe'));
    candidates.push(path.join(directory, '.venv', 'bin', 'python'));
    const parent = path.dirname(directory);
    if (parent === directory) break;
    directory = parent;
  }
  const found = candidates.find((candidate) => existsSync(candidate));
  if (!found) {
    throw new Error(
      'No Python environment found. Set VAELOR_ACCEPTANCE_PYTHON to an '
      + `interpreter that can import vaelor. Looked in:\n  ${candidates.join('\n  ')}`,
    );
  }
  return found;
}

/*
 * Resolved only when a preview is actually going to be started. `--self-test`
 * needs a browser and nothing else, and dying on a missing interpreter it was
 * never going to use would make the self-test unrunnable exactly where it is
 * most useful - a checkout with no Python environment.
 */
export const python = startPreview ? resolvePython() : null;
export const chrome = process.env.VAELOR_ACCEPTANCE_CHROME ?? 'C:\\Program Files\\Google\\Chrome\\Application\\chrome.exe';
export const username = process.env.VAELOR_ACCEPTANCE_USERNAME ?? 'responsive-admin';
export const password = process.env.VAELOR_ACCEPTANCE_PASSWORD ?? 'responsive acceptance passphrase';

export async function prepareEvidenceDirectory() {
  if (captureEvidence) await mkdir(evidenceRoot, { recursive: true });
}

export async function capture(page, name) {
  if (!captureEvidence) return;
  await page.screenshot({ path: path.join(evidenceRoot, `${name}.png`), fullPage: true });
}
