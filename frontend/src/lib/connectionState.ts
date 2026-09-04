import { timeAgo } from "./format";

/**
 * Task #52. The top bar used to hold two independent opinions about the same
 * fact: the sentence was computed from the age of the newest sample, and the
 * dot was computed from a `isLive` boolean written by the fetch handlers. The
 * two disagreed for six minutes at a time — the label read
 * `Paused · last updated 6 min ago` beside a pulsing green dot — because a
 * telemetry request that was already in flight when the tab was hidden
 * resolved afterwards and set `isLive` back to true. Nothing was left to poll,
 * so nothing ever set it back.
 *
 * Two lessons are built into this module.
 *
 * 1. **One value.** `connectionStatus()` returns the state, the sentence and
 *    the indicator class together. A caller cannot render the dot from one
 *    input and the words from another, because there is only one input.
 * 2. **Four states, not two.** `paused` is not a degraded `live` and it is not
 *    an error. Folding it into either forces a true fact into a bucket that
 *    does not fit, which is how the green dot survived review: green was the
 *    only value the code could produce for "the last request succeeded".
 */
export type ConnectionState = "live" | "paused" | "unknown" | "error";

/** The outcome of the most recent telemetry attempt. */
export type TelemetryOutcome = "pending" | "ok" | "failed";

export type ConnectionStatus = {
  state: ConnectionState;
  /** The sentence beside the indicator. */
  label: string;
  /** One word, for the narrow rail. */
  short: string;
  /**
   * Full class attribute for the indicator element. Derived here so the mark
   * and the words cannot be sourced separately again.
   */
  indicatorClassName: string;
  /** Whether this state is one a reader should act on. */
  attention: boolean;
};

/**
 * A sample older than this while polling is running means readings have
 * stopped arriving even though nobody suspended anything.
 */
export const STALE_AFTER_MS = 60_000;

/** Telemetry timestamps arrive in seconds from some providers and in
 * milliseconds from others. */
export function sampleTimeMs(lastSample: number): number {
  return lastSample > 0 && lastSample < 1_000_000_000_000 ? lastSample * 1000 : lastSample;
}

export type ConnectionInput = {
  /** False while the document is hidden and polling is deliberately suspended. */
  polling: boolean;
  outcome: TelemetryOutcome;
  /** Sample timestamp of the newest reading, or 0 when none has arrived. */
  lastSample: number;
  now?: number;
};

/**
 * Order of precedence, and why it is this order.
 *
 * `unknown` first: before the first answer nothing else is known, and "not
 * knowing is its own answer" — it must not borrow the reassuring display or
 * the alarming one.
 *
 * `paused` second, ahead of `failed`: while polling is suspended no request is
 * being retried, so raising an error about an attempt nobody is repeating asks
 * the reader to fix something that may not be broken. Returning to the tab
 * polls immediately, and a real failure re-reports itself within one round
 * trip. The sentence still carries the age, so staleness is never concealed.
 *
 * `error` covers both a failed request and a succeeded request whose newest
 * sample has stopped advancing. To a reader those are the same event —
 * readings are not arriving — and the old code called the second one "Paused",
 * which was simply untrue while the poller was running.
 */
export function connectionState({ polling, outcome, lastSample, now = Date.now() }: ConnectionInput): ConnectionState {
  if (outcome === "pending" && !lastSample) return "unknown";
  if (!polling) return "paused";
  if (outcome === "failed") return "error";
  if (!lastSample) return "unknown";
  return now - sampleTimeMs(lastSample) > STALE_AFTER_MS ? "error" : "live";
}

function sentence(state: ConnectionState, lastSample: number, now: number): string {
  if (state === "unknown") return "Waiting for telemetry";
  const age = timeAgo(sampleTimeMs(lastSample), now);
  if (state === "paused") {
    return lastSample ? `Paused · last updated ${age}` : "Paused · no readings yet";
  }
  if (state === "error") {
    return lastSample ? `Not updating · last updated ${age}` : "Cannot reach this machine";
  }
  const seconds = Math.max(0, Math.floor((now - sampleTimeMs(lastSample)) / 1000));
  /*
   * Named, because this strip is on every screen and the age it reports is its
   * own. Off Home the shell polls `/telemetry/current` every 30 seconds while
   * a panel with its own reading — the cooling page refreshes `/fans` every
   * three — updates the number under the reader's eyes. "Updated 25s ago"
   * beside a temperature that had visibly moved twice in that time is a true
   * statement about the wrong clock, and an unqualified one invites the reader
   * to apply it to whatever is nearest. Saying which reading it ages is the
   * difference between two clocks and two *labelled* clocks.
   */
  return seconds < 5 ? "Telemetry live · updated just now" : `Telemetry updated ${seconds}s ago`;
}

const shortWords: Record<ConnectionState, string> = {
  live: "Live",
  paused: "Paused",
  unknown: "Connecting",
  error: "Not updating",
};

/**
 * Whether this state is one a reader should act on.
 *
 * `paused` and `unknown` are deliberately not: a reader who collapsed a tab
 * group has not broken anything, and a shell that has not finished its first
 * request has nothing to report yet. Raising the rail on either is the alarm
 * fatigue that makes the real one ignorable.
 */
export function connectionNeedsAttention(state: ConnectionState): boolean {
  return state === "error";
}

/** One word for a state, for surfaces with no room for the sentence. */
export function connectionWord(state: ConnectionState): string {
  return shortWords[state];
}

/**
 * Task #74. The rail's health chip has a fifth thing to say that the top bar
 * does not: the connection is live and the *machine* wants attention. It is a
 * state of the chip, not a flag layered over one, which is why it belongs in
 * this union — a boolean beside the state is exactly the second source that
 * #52 removed.
 */
export type SignalState = ConnectionState | "attention";

/**
 * `indicator` is the base class of the mark being drawn — `live-dot`,
 * `signal-orb`, `telemetry-bus__pulse`, `hardware-led`, `hardware-state`.
 * Every one of them gets the same state suffix, so one state value paints
 * every mark *and every word* in the shell, and the base class on its own is
 * never a colour.
 */
export function connectionIndicator(state: SignalState, indicator: string): string {
  return `${indicator} ${indicator}--${state}`;
}

const signalWords: Record<SignalState, string> = {
  ...shortWords,
  live: "Healthy",
  attention: "Attention",
};

/** A mark and the word beside it, both painted from one value. */
export type SignalPresentation = {
  state: SignalState;
  word: string;
  /** Class attribute for the light. */
  markClassName: string;
  /** Class attribute for the label beside it. */
  labelClassName: string;
};

/**
 * Task #74, and it is #52 again.
 *
 * The sidebar rail took the four state modifiers on its LED and none on the
 * label beside it, so `.hardware-state` fell through to a hard-coded
 * `color: var(--success)` and the rail printed **PAUSED in success green next
 * to a grey light**. The dot and the word were once more deriving from
 * different sources — the dot from the connection state, the colour from a
 * stylesheet default — which is the precise failure the "one value" rule was
 * written to make unrepresentable.
 *
 * It is unrepresentable now because there is nothing here to disagree with:
 * one `state` produces the word, the mark's class and the label's class. A
 * caller cannot pass the label a different input, because the label is not an
 * input.
 */
export function hardwareSignal(connection: ConnectionState, attention: boolean): SignalPresentation {
  const state: SignalState = connection === "live" && attention ? "attention" : connection;
  return {
    state,
    word: signalWords[state],
    markClassName: connectionIndicator(state, "hardware-led"),
    labelClassName: connectionIndicator(state, "hardware-state"),
  };
}

export function connectionStatus(input: ConnectionInput, indicator = "live-dot"): ConnectionStatus {
  const now = input.now ?? Date.now();
  const state = connectionState({ ...input, now });
  return {
    state,
    label: sentence(state, input.lastSample, now),
    short: connectionWord(state),
    indicatorClassName: connectionIndicator(state, indicator),
    attention: connectionNeedsAttention(state),
  };
}
