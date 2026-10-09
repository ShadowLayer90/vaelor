import { useCallback, useEffect, useRef, useState } from "react";
import { Button, Notice } from "./ui";

/** What one worker Recheck did: when it read the machine, or why it could not. */
export type RecheckResult =
  // `telemetryFailed`: the machine was read but its telemetry check failed -
  // the backend still answers 200, with telemetry.action "failed" (W6-3).
  // `softwareNote` (VD-194 round 3 R3-2): the machine was read, but its worker
  // software could not be - the backend's own sentence for why.
  | { ok: true; checkedAt: number | null; message: string; telemetryFailed?: boolean; softwareNote?: string }
  | { ok: false; message: string };

/** What `POST /cluster/nodes/<id>/refresh` answers, the part a Recheck reports. */
export type RecheckResponse = {
  inventory?: { checked_at?: number };
  telemetry?: { action?: string; message?: string };
  worker_software?: { state?: string; sentence?: string; attempt_note?: string };
} | null | undefined;

/** Software states a Recheck reports as a refusal: nothing was read, or the update failed. */
const SOFTWARE_REFUSED_STATES = new Set(["unknown", "update-failed"]);

/**
 * The one reading of a Recheck's answer (W6-D4, LESSONS 6): the Fleet card and
 * the GPU pool card each read it their own way, and only one said "Rechecked at".
 */
export function recheckResultFrom(response: RecheckResponse): RecheckResult {
  const checkedAt = response?.inventory?.checked_at;
  return {
    ok: true,
    checkedAt: typeof checkedAt === "number" ? checkedAt : null,
    message: response?.telemetry?.message ?? "",
    telemetryFailed: response?.telemetry?.action === "failed",
    softwareNote: softwareNoteOf(response?.worker_software),
  };
}

/**
 * The software half's outcome, in the backend's words (VD-194): its sentence
 * when nothing could be read or the update failed, and otherwise the note of a
 * check that failed over a kept reading (round 4 N1) - which would else read
 * as "nothing needed changing".
 */
function softwareNoteOf(software: NonNullable<RecheckResponse>["worker_software"]): string | undefined {
  if (!software) return undefined;
  if (SOFTWARE_REFUSED_STATES.has(software.state ?? "")) return software.sentence || undefined;
  return software.attempt_note || undefined;
}

/** A Recheck that could not reach the machine, in the request's own words. */
export function recheckFailureFrom(error: unknown): RecheckResult {
  return { ok: false, message: error instanceof Error && error.message ? error.message : "The worker could not be reached." };
}

/**
 * A worker Recheck's own feedback (W5-D2). A Recheck that changed nothing said
 * nothing: the button never changed and no line moved, so the owner could not
 * tell it had run. The button is busy while it runs, and the line beside it
 * then says when the machine was read - or, as an alert (VD-189), why not.
 */
export function useRecheck(recheck: () => Promise<RecheckResult>) {
  const [busy, setBusy] = useState(false);
  const [result, setResult] = useState<RecheckResult | null>(null);
  const mounted = useRef(true);
  useEffect(() => () => { mounted.current = false; }, []);
  const run = useCallback(async () => {
    setBusy(true);
    setResult(null);
    const outcome = await recheck();
    if (!mounted.current) return;
    setResult(outcome);
    setBusy(false);
  }, [recheck]);
  return { busy, result, run };
}

/**
 * The Recheck button and its outcome line, for an action row that wraps: the
 * line takes a row of its own at the row's end (`.recheck-outcome`).
 */
export function RecheckButton({ recheck, machineName }: {
  recheck: () => Promise<RecheckResult>;
  /** Names the machine to a screen reader; the visible label stays one word (W6-2). */
  machineName?: string;
}) {
  const { busy, result, run } = useRecheck(recheck);
  return (
    <>
      <RecheckTrigger busy={busy} machineName={machineName} run={run} />
      <RecheckOutcome result={result} />
    </>
  );
}

/**
 * The button alone, for a card that shows the outcome line somewhere else
 * (the GPU pool card puts it below its aligned rows, FE-W7-4). Its words and
 * the outcome's stay those of RecheckButton (W6-D4, one wording).
 */
export function RecheckTrigger({ busy, run, machineName }: { busy: boolean; run: () => Promise<void>; machineName?: string }) {
  return (
    <Button aria-label={machineName ? `Recheck ${machineName}` : undefined} busy={busy} variant="quiet"
      onClick={() => void run()}>Recheck</Button>
  );
}

export function RecheckOutcome({ result }: { result: RecheckResult | null }) {
  if (!result) return null;
  if (!result.ok) {
    return <Notice className="recheck-outcome" severity="danger">{`Recheck failed: ${result.message}`}</Notice>;
  }
  const when = result.checkedAt ? `at ${new Date(result.checkedAt * 1000).toLocaleTimeString()}` : "just now";
  if (result.telemetryFailed) {
    // Read, but the telemetry check failed: that half is a refusal, said as
    // an alert (VD-189, W6-3), never as the status line below.
    // Round 4 N4: both halves' reasons when both failed.
    const software = result.softwareNote ? ` Worker software: ${result.softwareNote}` : "";
    return <Notice className="recheck-outcome" severity="danger">{`Rechecked ${when}, but ${result.message}${software}`}</Notice>;
  }
  if (result.softwareNote) {
    // VD-194 round 3 R3-2 / LESSONS 8: the software half could not be read or
    // recorded; said as an alert, with the backend's reason, never as success.
    return <Notice className="recheck-outcome" severity="warning">{`Rechecked ${when}. Worker software: ${result.softwareNote}`}</Notice>;
  }
  return (
    <p className="recheck-outcome" role="status">
      {`Rechecked ${when}. ${result.message || "The machine was read again; nothing needed changing."}`}
    </p>
  );
}
