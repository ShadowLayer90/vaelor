# Contributing to Vaelor

Vaelor welcomes bug reports, documentation improvements, platform adapters,
tests, and focused feature changes. By contributing, you agree that your work
may be distributed under GPL-2.0-only.

## Development setup

Use Python 3.10 or newer and Node.js 20 or newer:

```text
python -m venv .venv
.\.venv\Scripts\python -m pip install -e .
cd frontend
npm ci
```

On Linux, replace the Windows virtual-environment path with
`.venv/bin/python`.

## Required checks

The public repository is a curated snapshot of Vaelor's source and installer:
`vaelor/`, `frontend/`, `deploy/`, the compatibility aliases, and the documents
at its root. The Python and frontend test suites and the project's internal
design records are kept in the development tree and are not part of the
snapshot, so `python -m pytest` and `npm test` can be run only there;
`npm run check` and `npm run build` run from the snapshot. A change proposed
against the snapshot is run through the full set of checks below before it is
merged.

Before proposing a change:

```text
python -m pytest -q
cd frontend
npm ci
npm test -- --run
npm run check
npm run build
```

`npm run check` ends with the Performance dashboard layout contract
(`npm run dashboard:contract`, `frontend/scripts/dashboard-layout-contract.mjs`).
It renders every dashboard state in a headless Chrome at 1920 x 1080, in Easy
and in Advanced mode, and fails when a held clause misses. Easy: equal tiles
and columns, tiles and the first chart row above the fold, every plot at least
124 px tall with gridlines at least 20 px apart, no horizontal overflow, no cut
legend or chip name, and no cut tile caption. A machine name on a tile is cut
only where long names would otherwise push the first chart row below the fold
(the `four_long_names` state), and a cut name keeps its full text as its title
and for screen readers. Advanced: the three Advanced panels (package
power, GTT used, GPU edge temperature) keep the same plot-height and gridline
clauses, with no horizontal overflow and no shortened name; the fold does not
apply.

The same contract renders the real Cluster page from
`frontend/scripts/dashboard-contract/fleet.html` (fixture words, not the
backend's) and holds three more surfaces:

- **Fleet machine tiles**, one worker card open in Advanced, at 1920, 1440,
  1100, 900, 768 and 375 px, for each power sensor word ("graphics engine",
  "board", none), and with a long machine name at 1440, 900 and 375 px: every
  tile's round icon badge is a full 38 x 38 px, no tile title runs past two
  lines, no reading runs under its trend line, and the page has no horizontal
  overflow.
- **Deployments row**, a split with both fence notes and a long LAN endpoint,
  at eleven widths from 1920 down to 375 px: the endpoint is one line and
  carries its full text as its title, the type badge and the status are one
  line each, the name keeps at least 160 px, both notes show uncut, a list 900
  px or wider keeps the row on one line, and the page has no horizontal
  overflow (also at 1220 and 1200 px with a long IPv6 endpoint). A live-like
  tab (the row serving, the LLM Server card with two keys, the cluster serving
  card) is held at 1440, 1024, 768, 600, 506, 480 and 375 px, and there the
  contract also looks inside scroll boxes, which the overflow clause skips: no
  scroll box hides content sideways (a one-line `code` value beside its own
  Copy button is the one exception), no API key row is taller than 220 px, and
  no button breaks a word across lines.
- **Cluster tab strip**, at 768, 600 and 375 px, opened on "Activity" and on
  "Agents & tools": the selected tab is wholly inside the strip, and each end
  of the strip that hides tabs carries its cue (`data-more-start` /
  `data-more-end`, drawn as a fade), and only such an end.
- **Confirmation dialog**, the Setup cluster-link confirmation with a note as
  long as the backend's, at 1440 x 900, 375 x 667 and 320 x 667 px: the dialog
  stays inside the viewport and both its buttons are on screen and at least
  44 px tall; a long body scrolls inside the dialog instead of pushing the
  buttons off the screen.

A failure prints the clause, what was measured, and for the key and tab-strip
clauses the rule it enforces. Fix the layout, not the threshold.

Set `VAELOR_ACCEPTANCE_CHROME` to a Chrome or Chromium executable when it is not
in a standard install path, for example
`VAELOR_ACCEPTANCE_CHROME=/usr/bin/chromium npm run dashboard:contract`. If that
variable names a file that does not exist, the contract fails with that
message. If no browser is found, the contract fails (exit 1) unless
`VAELOR_SKIP_DASHBOARD_CONTRACT=1` is set; with it set, the contract prints a
loud SKIPPED notice and exits 0. The variable is for a machine with no
browser, such as a Linux build host; set it there only. Nothing sets it for
you. The appliance path, `npm run build`, does not run the contract.

Production modules must remain at or below 1,000 physical lines. That is not a
Python-only rule: the module-boundary test covers `.py` under `vaelor`,
`pm_dashboard`, `tools`, and `examples`, and `.ts`, `.tsx`, `.css`, and `.mjs`
under the whole of `frontend`, and fails on any source file no root or
exclusion accounts for, so a new top-level package is in scope by default.

A soft warning reports every production module over 850 lines. It never fails a
run and must not be turned into one. **The ceiling does not apply to test
files, wherever they live** — including the `.test.tsx` files that sit beside
the components they cover.

Put shared behavior in a focused module rather than copying it between API,
hardware, workload, or inference implementations.

## Design and safety rules

- Read [ARCHITECTURE.md](ARCHITECTURE.md) and
  [SUPPORTED_PLATFORMS.md](SUPPORTED_PLATFORMS.md) before changing platform
  behavior. For a larger change, open an issue describing it first: the
  maintainers keep a record of past design decisions and will say whether one
  already covers it.
- Every interface change uses the console's shared design tokens and
  primitives (`frontend/src/styles` and `frontend/src/components/ui`). Workflow
  completion, responsive layout, and accessibility are merge requirements.
- Do not add a one-off control or status vocabulary when the design system
  already owns that behavior.
- Keep hardware and OS behavior behind capability discovery. Generic code must
  not assume Raspberry Pi, Pironman, Ubuntu, Docker, a desktop, or a GPU.
- Read-only inspection may run immediately. Host, workload, credential,
  network, storage, or hardware mutations require a reviewable plan and an
  explicit approval.
- Never log passwords, API keys, SSH private keys, bearer tokens, model
  credentials, or decrypted broker values.
- Preserve GPL notices and record any new bundled third-party code or assets in
  `THIRD_PARTY_NOTICES.md`, and any new runtime component Vaelor downloads.
- Never put real IP addresses, host names, user names, e-mail addresses, or
  hardware identifiers in code, tests, logs, or documents. Use placeholders
  such as `192.0.2.10`, `10.20.30.40`, `mini-pc`, and `ubuntu`.
- Add failure-path tests, not only happy-path tests. Beginner mistakes and
  interrupted operations are supported product scenarios.

## Compatibility changes

The `vaelor` package and `VAELOR_*` settings are the public interfaces.
`pm_dashboard`, `PM_*`, and Pironman-era services and paths are temporary
compatibility aliases. Do not introduce new consumers of a legacy name.
