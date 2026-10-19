import type { ApiEnvelope } from "../types";

/**
 * The Assistant's answer, read as the on-device model writes it (VD-218).
 *
 * `POST /assistant/chat` with `Accept: application/x-ndjson` answers in one of
 * two ways. A turn that finishes before the model starts writing — a built-in
 * answer, a replayed duplicate, the busy model's 503 — comes back as the usual
 * JSON envelope, status and all. A turn the model is writing comes back as one
 * JSON object per line: `started`, then `draft` lines (`append` grows the
 * provisional text, `text` replaces it, `""` takes it back), `wait` heartbeats,
 * and finally `answer`, carrying the same envelope and status the unstreamed
 * request returns. The draft is never the answer; `answer` replaces it.
 *
 * `apiRequest` drives this when a caller passes `answerStream` handlers, so
 * the request keeps the transport's retry, CSRF and error vocabulary.
 */
export const ASSISTANT_STREAM_MEDIA_TYPE = "application/x-ndjson";

/**
 * The appliance writes a line at least every second while it answers, so a
 * stream silent this long has lost its connection without saying so.
 */
export const STREAM_SILENCE_MS = 30_000;

export interface AnswerStreamHandlers {
  /** The model holds its slot and is writing; drafts follow. */
  onStarted?: () => void;
  /** The whole provisional text so far. */
  onDraft?: (text: string) => void;
}

interface StreamEvent {
  event: "started" | "draft" | "wait" | "answer";
  append?: string;
  text?: string;
  status?: number;
  body?: ApiEnvelope<unknown>;
}

/** How a stream ended: its `answer` line, or the connection going first. */
export type AnswerStreamOutcome =
  | { status: number; body: ApiEnvelope<unknown> | undefined }
  | { dropped: true };

export function isAnswerStream(response: Response): boolean {
  return Boolean(response.body)
    && (response.headers?.get("Content-Type") ?? "").includes(ASSISTANT_STREAM_MEDIA_TYPE);
}

/** The next chunk, or a rejection once the stream has been silent too long. */
function readWithin(reader: ReadableStreamDefaultReader<Uint8Array>, ms: number) {
  let timer = 0;
  const silent = new Promise<never>((_, reject) => {
    timer = window.setTimeout(() => reject(new Error("silent")), ms);
  });
  return Promise.race([reader.read(), silent]).finally(() => window.clearTimeout(timer));
}

export async function readAnswerStream(
  response: Response,
  handlers: AnswerStreamHandlers,
  silenceMs = STREAM_SILENCE_MS,
): Promise<AnswerStreamOutcome> {
  const reader = response.body!.getReader();
  const decoder = new TextDecoder();
  let buffered = "";
  let draft = "";
  for (;;) {
    let chunk: ReadableStreamReadResult<Uint8Array>;
    try {
      chunk = await readWithin(reader, silenceMs);
    } catch {
      void reader.cancel().catch(() => undefined);
      return { dropped: true };
    }
    if (chunk.done) return { dropped: true };
    buffered += decoder.decode(chunk.value, { stream: true });
    let newline = buffered.indexOf("\n");
    while (newline >= 0) {
      const line = buffered.slice(0, newline).trim();
      buffered = buffered.slice(newline + 1);
      newline = buffered.indexOf("\n");
      if (!line) continue;
      let event: StreamEvent;
      try {
        event = JSON.parse(line) as StreamEvent;
      } catch {
        continue;
      }
      if (event.event === "started") handlers.onStarted?.();
      if (event.event === "draft") {
        draft = typeof event.append === "string" ? draft + event.append : event.text ?? "";
        handlers.onDraft?.(draft);
      }
      if (event.event === "answer") {
        void reader.cancel().catch(() => undefined);
        return { status: event.status ?? 500, body: event.body };
      }
    }
  }
}
