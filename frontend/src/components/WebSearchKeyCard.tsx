import { useCallback, useEffect, useRef, useState } from "react";
import { apiRequest } from "../lib/api";
import type { Session } from "../types";
import { RecordConfirm } from "./RecordKit";
import { StatusPill } from "./StatusPill";
import { Button, Card, Input, Notice, type NoticeSeverity } from "./ui";
import "../styles/settings.css";

/*
 * Settings › Connections › Web search key (VD-222 item 2). One optional key
 * the owner adds for Brave's search API. It is off until a key is added, never
 * added by Vaelor itself, and never shown back: the page reads only whether it
 * is set, its fingerprint tail and when it was added. Saving or removing it
 * queues the search service to rewrite its settings and restart.
 */

/** What `GET /web-search/key` says; the key itself never comes back. */
export interface WebSearchKeyStatus {
  configured: boolean;
  provider: string;
  fingerprint_tail?: string;
  /** Seconds since the epoch. */
  added_at?: number | null;
  added_by?: string;
}

/** The queued settings job a save or removal returns, as `/jobs/<id>` reads it. */
interface KeyJob { id: string; state: string; message?: string }
type KeyAnswer = WebSearchKeyStatus & { job?: KeyJob };

/** How long the card follows the job: the service's own bring-up allows 90 s. */
const FOLLOW_FOR_MS = 120_000;
const FOLLOW_EVERY_MS = 2_000;
const FINISHED = new Set(["completed", "failed", "cancelled"]);

/** The pattern the server accepts (`web_research.SEARCH_KEY_SHAPE`), checked here only to say so early. */
const KEY_SHAPE = /^[A-Za-z0-9_-]{16,200}$/;

export function WebSearchKeyCard({ session }: { session: Session }) {
  const [status, setStatus] = useState<WebSearchKeyStatus | null>(null);
  const [readError, setReadError] = useState("");
  const [key, setKey] = useState("");
  const [busy, setBusy] = useState(false);
  const [removing, setRemoving] = useState(false);
  const [outcome, setOutcome] = useState<{ text: string; severity: NoticeSeverity } | null>(null);
  const mounted = useRef(true);
  useEffect(() => () => { mounted.current = false; }, []);

  const read = useCallback(async () => {
    try {
      setStatus(await apiRequest<WebSearchKeyStatus>("/web-search/key", { cache: "no-store" }));
      setReadError("");
    } catch (error) {
      setReadError(error instanceof Error && error.message ? error.message : "The web search key could not be read.");
    }
  }, []);

  useEffect(() => { void read(); }, [read]);

  /*
   * VD-222 adversarial F7: the key is stored at once, but the search service
   * takes it only when its queued job runs. Say "applying" until the job
   * finishes, then what it did; a failed job is shown, never a success.
   */
  const follow = async (job: KeyJob | undefined, success: string) => {
    if (!job) { setOutcome({ text: success, severity: "success" }); return; }
    setOutcome({ text: "Saved in secure storage. Applying it to the search service…", severity: "info" });
    const until = Date.now() + FOLLOW_FOR_MS;
    let current = job;
    while (!FINISHED.has(current.state) && Date.now() < until && mounted.current) {
      await new Promise((resolve) => setTimeout(resolve, FOLLOW_EVERY_MS));
      try { current = await apiRequest<KeyJob>(`/jobs/${job.id}`, { cache: "no-store" }); } catch { /* keep following */ }
    }
    if (!mounted.current) return;
    if (current.state === "completed") setOutcome({ text: success, severity: "success" });
    else if (FINISHED.has(current.state)) setOutcome({ text: current.message || "The search service did not take the change.", severity: "danger" });
    else setOutcome({ text: "Still applying. Operations shows the search service's job.", severity: "warning" });
  };

  const run = async (action: () => Promise<KeyAnswer>, success: string, failure: string) => {
    setBusy(true);
    setOutcome(null);
    try {
      const answer = await action();
      setStatus(answer);
      setKey("");
      setRemoving(false);
      await follow(answer.job, success);
      return true;
    } catch (error) {
      setOutcome({ text: error instanceof Error && error.message ? error.message : failure, severity: "danger" });
      return false;
    } finally {
      setBusy(false);
    }
  };

  const save = () => run(
    () => apiRequest<KeyAnswer>("/web-search/key", { method: "PUT", body: JSON.stringify({ key: key.trim() }) }, session.csrf_token),
    "Key applied. Web searches now also go to Brave.",
    "The key was not saved.",
  );
  const remove = () => run(
    () => apiRequest<KeyAnswer>("/web-search/key", { method: "DELETE" }, session.csrf_token),
    "Key removed. Web searches no longer go to Brave.",
    "The key was not removed.",
  );

  const trimmed = key.trim();
  const shapeError = trimmed && !KEY_SHAPE.test(trimmed)
    ? "Paste the key exactly as Brave shows it: letters, digits, '-' and '_', 16 to 200 characters."
    : undefined;
  const pill = status === null
    ? { label: readError ? "Not read" : "Reading", tone: readError ? "warning" as const : "neutral" as const }
    : status.configured ? { label: "On", tone: "success" as const } : { label: "Off", tone: "neutral" as const };
  const added = status?.added_at ? new Date(status.added_at * 1000).toLocaleDateString() : null;

  return (
    <Card
      actions={<StatusPill label={pill.label} tone={pill.tone} />}
      as="section"
      description="Optional. Off until you add a key."
      flush
      heading="Web search key"
    >
      <div className="stg-row">
        <div className="stg-row__text">
          <strong>Brave Search API</strong>
          <small>
            Web search on this appliance uses free public search engines. A Brave Search API key adds Brave as one
            more engine. With a key, every web search here (AI Chat, custom agents and app research) also goes to
            Brave under your account. Vaelor never adds a key by itself, and a saved key cannot be shown again.
          </small>
        </div>
      </div>
      {readError && <div className="stg-row"><Notice severity="warning">{readError}</Notice></div>}
      {status?.configured && (
        <div className="stg-row stg-row--key">
          <div className="stg-row__text">
            <strong>Key saved</strong>
            <small>{`fingerprint …${status.fingerprint_tail ?? ""}${added ? ` · added ${added}` : ""}${status.added_by ? ` by ${status.added_by}` : ""}`}</small>
          </div>
          <div className="stg-row__controls record-card-button">
            <Button className="record-ghost" disabled={busy} onClick={() => setRemoving(true)} variant="quiet">Remove key</Button>
          </div>
        </div>
      )}
      <form className="stg-api__create" onSubmit={(event) => { event.preventDefault(); if (trimmed && !shapeError && !busy) void save(); }}>
        <Input
          autoComplete="off"
          error={shapeError}
          id="web-search-key"
          label={status?.configured ? "Replace the key" : "Brave Search API key"}
          onChange={(event) => setKey(event.target.value)}
          spellCheck={false}
          type="password"
          value={key}
        />
        <span className="record-card-button">
          <Button
            disabled={busy || !trimmed || Boolean(shapeError) || status === null}
            disabledReason={!trimmed ? "Paste a key to save it." : status === null ? "Waiting to read the current key." : undefined}
            type="submit"
            variant="primary"
          >
            Save key
          </Button>
        </span>
      </form>
      {outcome && <div className="stg-row"><Notice severity={outcome.severity}>{outcome.text}</Notice></div>}
      {removing && (
        <RecordConfirm
          busy={busy}
          confirmLabel="Remove key"
          eyebrow="Brave stops being searched"
          eyebrowTone="plain"
          onCancel={() => { if (!busy) setRemoving(false); }}
          onConfirm={() => void remove()}
          title="Remove the web search key?"
        >
          <p>Vaelor deletes the stored key and restarts the search service without Brave. Web search goes on with the free engines.</p>
        </RecordConfirm>
      )}
    </Card>
  );
}
