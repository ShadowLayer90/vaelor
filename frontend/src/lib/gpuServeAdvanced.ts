/**
 * The cluster deploy form's Advanced serving settings (VD-213): how many
 * requests one machine takes at once, how many prompt tokens it works through
 * per step, the draft model and its lookahead, and vLLM's newer model runner.
 *
 * The fit's `serving.advanced` object says what a model offers and what it
 * opens on; this module turns the owner's edits into the deploy fields. The
 * rule that makes the model's catalog default reach the box is ABSENCE: a
 * setting the owner did not change is never sent, so the server applies the
 * model's own defaults. Only a value that differs from the default is sent.
 *
 * The defaults shown are the ones the server will apply, which is not always
 * `advanced`'s: `vaelor/vllm_serve_advanced.advanced_for` applies the catalog
 * defaults only to one copy per machine, drops the default draft model beside
 * multi-token prediction, and stretches the default lookahead to a changed
 * cap. `effectiveDefaults` mirrors those three rules, so the form never shows
 * a draft model the deploy would not run (LESSONS 10).
 */

/** One draft model the server pins for this model and image. */
export interface GpuServeDrafter {
  key: string;
  label: string;
  repo: string;
  /** Bytes every machine that serves the model downloads for the drafter. */
  weight_bytes: number;
  /** The largest lookahead this drafter accepts. */
  max_tokens: number;
}

/** `[lo, hi, k]`: with lo..hi requests running, draft k tokens ahead. */
export type LookaheadStep = [number, number, number];

export interface GpuServeBound {
  default: number | null;
  min: number;
  max: number;
}

/** `serving.advanced` as `vllm_serve_advanced.advanced_choices` sends it. */
export interface GpuServingAdvanced {
  max_num_seqs: GpuServeBound;
  max_num_batched_tokens: GpuServeBound;
  drafters: GpuServeDrafter[];
  drafter_default: { key: string; schedule: LookaheadStep[] } | null;
  new_runner: { offered: boolean; default: boolean; note: string };
  note: string;
}

/**
 * The server's ranges, once, in TypeScript (LESSONS 6 / VD-213: two lists that
 * must agree). The rule lives in Python - `MAX_NUM_SEQS_RANGE` and
 * `BATCHED_TOKENS_RANGE` in `vaelor/vllm_serve_advanced.py`, `MAX_DRAFT_TOKENS`
 * in `vaelor/vllm_drafters.py`. The server also sends its bounds in `advanced`
 * (and each drafter's `max_tokens`); those win, and these only fill a bound an
 * older answer left out. Change the Python first, then this.
 */
export const ADVANCED_LIMITS = {
  max_num_seqs: { min: 1, max: 64 },
  max_num_batched_tokens: { min: 1024, max: 65536 },
  lookahead: { min: 1, max: 15 },
} as const;

/**
 * A lookahead to start from when the owner turns on a draft model the model
 * has no default schedule for. A starting point only, not a measured value;
 * the same number alone and busy, so it never forces the newer runner.
 */
const FALLBACK_LOOKAHEAD = 4;

/** Why the draft model and the runner are not offered on a split; the form's MTP sentence. */
export const SPLIT_ONLY_REASON = "Offered only when each machine serves its own copy.";

/** What the owner sees in each field; every number is the text in its field. */
export interface AdvancedDraft {
  maxNumSeqs: string;
  maxBatchedTokens: string;
  /** "" is Off. */
  drafterKey: string;
  alone: string;
  busy: string;
  newRunner: boolean;
}

/** Only the fields the owner changed; empty or null is untouched. */
export type AdvancedEdits = Partial<AdvancedDraft>;

export type AdvancedErrors = Partial<Record<keyof AdvancedDraft, string>>;

export interface AdvancedContext {
  /** One copy per machine (the throughput intent): the only layout with defaults. */
  replicated: boolean;
  /** The multi-token prediction the form would send. */
  mtpTokens: number;
}

export interface AdvancedResult {
  /** What each field shows: the server's defaults with the owner's edits on top. */
  shown: AdvancedDraft;
  /** What each field shows untouched: the defaults the server will apply here. */
  defaults: AdvancedDraft;
  /** The deploy (and fit) fields: only what the owner changed from the defaults. */
  fields: Record<string, unknown>;
  errors: AdvancedErrors;
  /** The newer runner is required: the lookahead differs alone and busy. */
  runnerForced: boolean;
  /** The busy lookahead applies: more than one request may run. */
  busyShown: boolean;
  /** Set when the draft model and the runner are not offered for this layout. */
  splitReason: string;
}

function bound(value: GpuServeBound | undefined, fallback: { min: number; max: number }) {
  return {
    min: typeof value?.min === "number" ? value.min : fallback.min,
    max: typeof value?.max === "number" ? value.max : fallback.max,
  };
}

/** An integer typed into a field, or null when it is not one. */
function whole(value: string): number | null {
  const text = value.trim();
  return /^\d+$/.test(text) ? Number(text) : null;
}

/** The lookahead for one request and for two or more, read off a schedule. */
function lookaheads(schedule: LookaheadStep[] | undefined): { alone: number; busy: number } | null {
  if (!schedule?.length) return null;
  const first = schedule.find(([lo]) => lo === 1) ?? schedule[0];
  const rest = schedule.find(([lo, hi]) => lo <= 2 && hi >= 2) ?? schedule[schedule.length - 1];
  return { alone: first[2], busy: rest[2] };
}

function count(value: unknown): number | null {
  return typeof value === "number" && Number.isInteger(value) ? value : null;
}

function boundOf(value: unknown, fallback: { min: number; max: number }): GpuServeBound {
  const raw = (value ?? {}) as Partial<GpuServeBound>;
  return { default: count(raw.default), min: count(raw.min) ?? fallback.min, max: count(raw.max) ?? fallback.max };
}

/**
 * `advanced` with every list and field present. An older or partial answer
 * (a field left out, a list missing) reads as "nothing offered" or as the
 * TS fallback range, never as a crash in the form. The server's own bounds
 * and each drafter's `max_tokens` win wherever they are given.
 */
export function normaliseAdvanced(raw: Partial<GpuServingAdvanced> | null | undefined): GpuServingAdvanced {
  const value = raw ?? {};
  const drafters = (Array.isArray(value.drafters) ? value.drafters : [])
    .filter((entry): entry is GpuServeDrafter => typeof entry?.key === "string" && entry.key !== "")
    .map((entry) => ({
      key: entry.key,
      label: typeof entry.label === "string" && entry.label ? entry.label : entry.key,
      repo: typeof entry.repo === "string" ? entry.repo : "",
      weight_bytes: count(entry.weight_bytes) ?? 0,
      max_tokens: count(entry.max_tokens) ?? ADVANCED_LIMITS.lookahead.max,
    }));
  const chosen = value.drafter_default;
  const runner = value.new_runner;
  const maxNumSeqs = boundOf(value.max_num_seqs, ADVANCED_LIMITS.max_num_seqs);
  // A default draft model is kept only when the form can show it as the
  // server will run it: every schedule row three whole numbers, a drafter the
  // list offers, and a default cap to cover. Anything else is dropped, so an
  // untouched form never shows an error the owner did not cause.
  const schedule = Array.isArray(chosen?.schedule) ? chosen.schedule : [];
  const wellFormed = schedule.length > 0 && schedule.every((row) =>
    Array.isArray(row) && row.length === 3 && row.every((part) => count(part) !== null));
  const keepDefault = Boolean(chosen) && typeof chosen?.key === "string" && wellFormed
    && drafters.some((entry) => entry.key === chosen.key) && maxNumSeqs.default !== null;
  return {
    max_num_seqs: maxNumSeqs,
    max_num_batched_tokens: boundOf(value.max_num_batched_tokens, ADVANCED_LIMITS.max_num_batched_tokens),
    drafters,
    drafter_default: keepDefault && chosen ? { key: chosen.key, schedule } : null,
    new_runner: {
      offered: runner?.offered === true,
      default: runner?.default === true,
      note: typeof runner?.note === "string" ? runner.note : "",
    },
    note: typeof value.note === "string" ? value.note : "",
  };
}

/**
 * The defaults the server will apply for this layout (`advanced_for`): none
 * on a split, and no default draft model beside multi-token prediction.
 */
export function effectiveDefaults(offered: GpuServingAdvanced, context: AdvancedContext): GpuServingAdvanced {
  const advanced = normaliseAdvanced(offered);
  if (!context.replicated) {
    return {
      ...advanced,
      max_num_seqs: { ...advanced.max_num_seqs, default: null },
      max_num_batched_tokens: { ...advanced.max_num_batched_tokens, default: null },
      drafters: [],
      drafter_default: null,
      new_runner: { ...advanced.new_runner, offered: false, default: false },
    };
  }
  return context.mtpTokens > 0 ? { ...advanced, drafter_default: null } : advanced;
}

/** What the fields show untouched, and return to on "Reset to recommended". */
export function recommendedDraft(offered: GpuServingAdvanced): AdvancedDraft {
  const advanced = normaliseAdvanced(offered);
  const chosen = advanced.drafter_default;
  const steps = lookaheads(chosen?.schedule) ?? { alone: FALLBACK_LOOKAHEAD, busy: FALLBACK_LOOKAHEAD };
  const seqs = advanced.max_num_seqs?.default;
  const batched = advanced.max_num_batched_tokens?.default;
  return {
    maxNumSeqs: seqs == null ? "" : String(seqs),
    maxBatchedTokens: batched == null ? "" : String(batched),
    drafterKey: chosen?.key ?? "",
    alone: String(steps.alone),
    busy: String(steps.busy),
    newRunner: Boolean(advanced.new_runner?.offered && advanced.new_runner.default),
  };
}

/**
 * The lookahead schedule for a request cap: k alone, and k for 2..cap busy
 * (VD-213: `[[1,1,4],[2,8,7]]`). One request at most has only the first step.
 */
export function buildSchedule(cap: number, alone: number, busy: number): LookaheadStep[] {
  return cap <= 1 ? [[1, 1, alone]] : [[1, 1, alone], [2, cap, busy]];
}

/**
 * Validate the owner's edits and build the fields to send. Null or empty
 * edits are untouched: nothing is sent and nothing can be wrong.
 */
export function advancedServeFields(
  offered: GpuServingAdvanced,
  rawEdits: AdvancedEdits | null,
  context: AdvancedContext,
): AdvancedResult {
  const advanced = effectiveDefaults(offered, context);
  const splitReason = context.replicated ? "" : SPLIT_ONLY_REASON;
  // On a split the draft model and the runner are not offered, so an edit
  // made to them under the other layout is set aside, never sent.
  const edits: AdvancedEdits = Object.fromEntries(Object.entries(rawEdits ?? {})
    .filter(([key]) => context.replicated || key === "maxNumSeqs" || key === "maxBatchedTokens"));
  const defaults = recommendedDraft(advanced);
  const shown: AdvancedDraft = { ...defaults, ...edits };
  const errors: AdvancedErrors = {};
  const seqBound = bound(advanced.max_num_seqs, ADVANCED_LIMITS.max_num_seqs);
  const tokenBound = bound(advanced.max_num_batched_tokens, ADVANCED_LIMITS.max_num_batched_tokens);

  // A blank field is "not given" to the server, so it means the default; with
  // a default that the field showed, blank is a mistake rather than a choice.
  const seqs = whole(shown.maxNumSeqs);
  const seqsBlank = shown.maxNumSeqs.trim() === "";
  if (seqsBlank ? defaults.maxNumSeqs !== "" : seqs === null || seqs < seqBound.min || seqs > seqBound.max) {
    errors.maxNumSeqs = `Choose a whole number from ${seqBound.min} to ${seqBound.max}.`;
  }
  const tokens = whole(shown.maxBatchedTokens);
  const tokensBlank = shown.maxBatchedTokens.trim() === "";
  if (tokensBlank ? defaults.maxBatchedTokens !== "" : tokens === null || tokens < tokenBound.min || tokens > tokenBound.max) {
    errors.maxBatchedTokens = `Choose a whole number from ${tokenBound.min.toLocaleString()} to ${tokenBound.max.toLocaleString()}.`;
  }

  const drafter = advanced.drafters.find((entry) => entry.key === shown.drafterKey) ?? null;
  if (shown.drafterKey && !drafter) {
    errors.drafterKey = "That draft model is not offered for this model and vLLM version.";
  }
  if (drafter && context.mtpTokens > 0) {
    errors.drafterKey = "Turn off multi-token prediction to use a draft model; vLLM runs one or the other.";
  }
  if (drafter && seqsBlank) {
    errors.maxNumSeqs = "Set how many requests a machine takes to use a draft model.";
  }
  const busyShown = seqs !== 1;
  const kMin = ADVANCED_LIMITS.lookahead.min;
  // The drafter's own limit is the server's (`vllm_drafters`); the TS one only fills a gap.
  const kMax = drafter?.max_tokens ?? ADVANCED_LIMITS.lookahead.max;
  const lookaheadError = `Choose a whole number from ${kMin} to ${kMax}.`;
  const alone = whole(shown.alone);
  const busy = busyShown ? whole(shown.busy) : alone;
  if (drafter && (alone === null || alone < kMin || alone > kMax)) errors.alone = lookaheadError;
  if (drafter && busyShown && (busy === null || busy < kMin || busy > kMax)) errors.busy = lookaheadError;

  const runnerOffered = Boolean(advanced.new_runner?.offered);
  const runnerForced = Boolean(drafter) && busyShown && alone !== null && busy !== null && alone !== busy;
  if (runnerForced && !runnerOffered) {
    errors.busy = "This vLLM version has only the older runner, so use the same lookahead alone and busy.";
  }
  const newRunner = runnerOffered && (runnerForced || shown.newRunner);

  const fields: Record<string, unknown> = {};
  if ("maxNumSeqs" in edits && seqs !== null && seqs !== advanced.max_num_seqs?.default) {
    fields.max_num_seqs = seqs;
  }
  if ("maxBatchedTokens" in edits && tokens !== null && tokens !== advanced.max_num_batched_tokens?.default) {
    fields.max_num_batched_tokens = tokens;
  }
  // An untouched default draft model is the server's to apply, stretched to
  // a changed cap there (`vllm_drafters.fit_schedule`); only a changed one is sent.
  const drafterEdited = shown.drafterKey !== defaults.drafterKey
    || (Boolean(drafter) && (shown.alone !== defaults.alone || (busyShown && shown.busy !== defaults.busy)));
  if (drafterEdited) {
    if (!drafter) {
      if (advanced.drafter_default) fields.drafter = null;
    } else if (seqs !== null && alone !== null && busy !== null) {
      fields.drafter = { key: drafter.key, schedule: buildSchedule(seqs, alone, busy) };
    }
  }
  if (runnerOffered && newRunner !== Boolean(advanced.new_runner.default)) fields.new_runner = newRunner;
  return { shown, defaults, fields, errors, runnerForced, busyShown, splitReason };
}

/** The first problem in the Advanced section, for the Serve button's reason. */
export function firstAdvancedError(result: AdvancedResult): string {
  return Object.values(result.errors)[0] ?? "";
}
