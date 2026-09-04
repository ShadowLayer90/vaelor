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

Before proposing a change:

```text
python -m pytest -q
cd frontend
npm ci
npm test -- --run
npm run check
npm run build
```

Production modules must remain at or below 1,000 physical lines. That is not a
Python-only rule: `tests/test_module_boundaries.py` derives the collection from
`PRODUCTION_ROOTS`, which covers `.py` under `vaelor`, `pm_dashboard`, `tools`,
and `examples`, and `.ts`, `.tsx`, `.css`, and `.mjs` under the whole of
`frontend`. A coverage walk fails on any source file no root or exclusion
accounts for, so a new top-level package is in scope by default.

A soft warning reports every production module over 850 lines. It never fails a
run and must not be turned into one (VD-030). **The ceiling does not apply to
test files, wherever they live** (VD-032) — including the `.test.tsx` files that
sit beside the components they cover.

Put shared behavior in a focused module rather than copying it between API,
hardware, workload, or inference implementations.

## Design and safety rules

- When you propose an architecture change, explain the reasoning in the pull
  request rather than asserting that something was "previously decided"; give a
  citation a reviewer can check.
- Reuse the shared design-system tokens and primitives for every interface
  change. Shared tokens, workflow completion, responsive viewports, and
  accessibility are merge requirements.
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
  `THIRD_PARTY_NOTICES.md`.
- Add failure-path tests, not only happy-path tests. Beginner mistakes and
  interrupted operations are supported product scenarios.

## Compatibility changes

The `vaelor` package and `VAELOR_*` settings are the public interfaces.
`pm_dashboard`, `PM_*`, and Pironman-era services and paths are temporary
compatibility aliases. Do not introduce new consumers of a legacy name.
