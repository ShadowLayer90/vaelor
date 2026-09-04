import { useEffect, useRef, useState } from "react";

/**
 * Waiting out loud for an answer this browser is no longer waiting on.
 *
 * Both chat surfaces write the question to the appliance before inference
 * starts, so a reload mid-answer restores a transcript that ends on the
 * question. Nothing then said an answer was still coming: the question sat
 * alone with no reply, no spinner and no error for over ninety seconds, the
 * answer landed server-side, and the only way to see it was to navigate away
 * and back. In the meantime the reader re-asked, and the conversation ended up
 * with the same question twice and two answers.
 *
 * The caller arms this from the path that restored the transcript — never from
 * a request that failed in this browser, which is a different thing to say —
 * and supplies a `poll` that re-reads the stored conversation and resolves true
 * once the reply is there.
 */
const RESUME_POLL_MS = 4_000;
const RESUME_WINDOW_MS = 300_000;

/**
 * The `code`s `apiRequest` raises when the connection itself dropped mid-answer,
 * as opposed to the appliance returning a structured rejection after deciding
 * not to queue the work.
 *
 * The distinction is the whole point of arming from a live failure. Both chat
 * surfaces write the question to the appliance (`ensure_conversation` →
 * `append_message`) before inference runs, so a connection dropped mid-answer
 * leaves the appliance finishing a reply it will persist — exactly the state a
 * reload recovers from. A clean structured rejection (`assistant_model_busy`,
 * `invalid_assistant_request`, `model_starting`, `chat_model_*`) queued nothing,
 * so polling for a reply that will never be written is only spurious traffic and
 * a waiting message that never resolves.
 *
 * These are the codes `api.ts` mints when the connection itself failed before
 * the appliance could answer: a rejected fetch — a reset, a nav-away, a proxy
 * closing the idle POST (`service_unavailable`) — and a proxy answering the
 * held-open POST with a non-JSON error page (`invalid_response`).
 * `request_timeout` is deliberately excluded: the client's own budget outlasts
 * the appliance's inference ceiling, so by the time it fires the appliance has
 * itself given up and persisted nothing to wait for, and its "try a smaller
 * model" guidance is the truthful outcome.
 */
const DROPPED_CONNECTION_CODES = new Set([
  "service_unavailable",
  "invalid_response",
]);

/**
 * True when a failed request dropped at the transport, so the appliance may
 * still be writing the answer and the resume-poll should be armed to surface it
 * live. False for a user Stop (caller detects that first) and for any structured
 * rejection, which queued no work to wait for.
 */
export function isDroppedConnection(error: unknown): boolean {
  const code = error instanceof Error && "code" in error
    ? String((error as { code?: unknown }).code ?? "")
    : "";
  return DROPPED_CONNECTION_CODES.has(code);
}

/** Shown on both surfaces while the resume-poll waits out a dropped send. */
export const DROPPED_CONNECTION_NOTICE =
  "The connection to the appliance dropped, but it may still be answering. "
  + "The reply appears here as soon as it lands — you do not need to ask again.";

/**
 * The conversation id to poll after a live send failed, or "" to leave the
 * failure alone. `skip` short-circuits cases with nothing persisted to wait for
 * (a temporary chat). A known `currentId` is returned as is; when it is empty —
 * a new chat whose id is only learned from the POST that just dropped — the
 * appliance created the conversation at `ensure_conversation` before inference,
 * so `newestId` re-lists and adopts it. Any failure to recover an id yields ""
 * and the caller keeps its ordinary failure handling.
 */
export async function droppedConnectionSubject(
  error: unknown,
  skip: boolean,
  currentId: string,
  newestId: () => Promise<string>,
): Promise<string> {
  if (skip || !isDroppedConnection(error)) return "";
  if (currentId) return currentId;
  try {
    return await newestId();
  } catch {
    return "";
  }
}

export function useResumedAnswer({ armed, subject, poll }: {
  /** True when a restored transcript ended on an unanswered question. */
  armed: boolean;
  /** Identifies what is being waited on; a change starts the wait over. */
  subject: string;
  /** Re-read the conversation. Resolve true once the answer has landed. */
  poll: () => Promise<boolean>;
}): { awaiting: boolean; lost: boolean } {
  const [lost, setLost] = useState(false);
  const [settled, setSettled] = useState(false);
  // `poll` is a fresh closure every render; holding it in a ref keeps the
  // interval from being torn down and restarted before it can ever fire.
  const pollRef = useRef(poll);
  useEffect(() => { pollRef.current = poll; });
  useEffect(() => {
    setLost(false);
    setSettled(false);
  }, [armed, subject]);
  const awaiting = armed && !lost && !settled;
  useEffect(() => {
    if (!awaiting) return;
    const deadline = Date.now() + RESUME_WINDOW_MS;
    const interval = window.setInterval(() => {
      if (Date.now() > deadline) {
        setLost(true);
        return;
      }
      void pollRef.current().then((done) => {
        if (done) setSettled(true);
      }).catch(() => undefined);
    }, RESUME_POLL_MS);
    return () => window.clearInterval(interval);
  }, [awaiting]);
  return { awaiting, lost };
}
