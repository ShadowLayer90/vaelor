import { useEffect, useState } from "react";

/**
 * Does the console's certificate change soon, or did it just change under this
 * page? (VD-133 1.5.5 #1)
 *
 * The household authority re-issues the console's certificate before it
 * expires, or when the machine gains an address the certificate does not
 * cover. A browser that trusts the household authority accepts the new one
 * without a word. A browser that only accepted the old certificate's warning
 * refuses it: every request from an open page fails, and before this the page
 * simply stopped updating with nothing to say why.
 *
 * So the authority announces a re-issue before it happens
 * (`tls_authority.REISSUE_NOTICE_SECONDS`), the transport route carries the
 * notice, and every open page polls it. Before the switch the page says what
 * is about to happen; after it, if this page can no longer reach Vaelor at
 * all, it says that is the likely reason and offers a reload.
 *
 * **What the page remembers, and why** (adversarial review F1). After the
 * switch the authority deletes its notice, and the console closes the
 * connections set up under the old certificate; a read in between can still
 * succeed and say "no re-issue planned". So the page also remembers the
 * certificate fingerprint it first read. Once a notice was seen or that
 * fingerprint changed, the certificate is known to have changed (or to be about
 * to) under this page, and a later run of failed reads is explained by it - a
 * read that says "none" does not make the page forget. A notice withdrawn while
 * the fingerprint never changed is forgotten: nothing changed.
 *
 * A failed read alone proves little - a restarting console refuses connections
 * too - so the lost-connection message needs two failed reads in a row (the
 * second a quick retry, review F7) AND a certificate change known to this page.
 *
 * **Round 2.** The memory is cleared by a read that succeeds well after the
 * change (F2): the console closes an old-certificate connection once it is idle,
 * or at the latest `tls_reload.CLOSE_CAP_SECONDS` (15 min) after the swap, so a
 * read that succeeds later than that plus two watch ticks came over a connection
 * made under the new certificate - this browser accepted it, and a later outage
 * is not the certificate's doing. The page keeps reading while it is hidden
 * (F3: a tab in the background through the whole announcement still learns of
 * it; browsers slow hidden timers to about once a minute, the same pace). And
 * every time is judged on the controller's clock, which each answer carries as
 * `now`, never the browser's (F10).
 */

/** How often an open page reads the notice while it is visible. */
export const RENEWAL_CHECK_MS = 60_000;
/** The quick second read after a failed one, before anything is said. */
export const RENEWAL_RETRY_MS = 5_000;
/**
 * A notice this long past its time is stale and never shown as planned: the
 * authority acts within one pass, and treats a notice three passes late as
 * stale itself (`tls_authority.NOTICE_STALE_AFTER_SECONDS`).
 */
export const NOTICE_STALE_AFTER_MS = 15 * 60_000;
/**
 * A read that succeeds this long after the change was seen came over a new
 * connection: `tls_reload.CLOSE_CAP_SECONDS` (15 min) plus two watch ticks.
 */
export const SETTLED_AFTER_MS = 15 * 60_000 + 2 * 5_000;

const TRANSPORT_PATH = "/api/v2/security/transport";

/** `reissue` in `/security/transport`, as `tls_authority_status` states it. */
export type ReissueNotice =
  | { state: "none" }
  | { state: "planned"; reason: string; announced_at: number; reissue_after: number }
  | { state: "stale"; reason: string }
  | { state: "unreadable"; reason: string };

export type PlannedReissue = Extract<ReissueNotice, { state: "planned" }>;

export type RenewalView =
  | { kind: "announced"; at: number; reason: string }
  | { kind: "lost"; at: number };

/** What the page has learnt from its reads. */
export type RenewalMemory = {
  /** The last planned notice read, kept once the certificate changed under the page. */
  notice: PlannedReissue | null;
  /** The certificate fingerprint of the first read that carried one. */
  firstFingerprint: string | null;
  /** When (ms) a read first showed a different fingerprint, or null. */
  changedAt: number | null;
  /** Failed reads in a row; 0 after any read that reached Vaelor. */
  failures: number;
  /** The controller's clock minus this browser's (ms), from the last answer's `now`. */
  skew: number;
};

export const FRESH_MEMORY: RenewalMemory = {
  notice: null, firstFingerprint: null, changedAt: null, failures: 0, skew: 0,
};

/** The planned re-issue in a transport answer, or null for any other state. */
export function plannedReissue(value: unknown): PlannedReissue | null {
  const notice = value as Partial<PlannedReissue> | null | undefined;
  if (notice?.state !== "planned") return null;
  if (typeof notice.reissue_after !== "number" || typeof notice.reason !== "string") return null;
  return {
    state: "planned",
    reason: notice.reason,
    announced_at: Number(notice.announced_at) || 0,
    reissue_after: notice.reissue_after,
  };
}

/** The memory after a read that reached Vaelor and answered `data` (`clientNow`: this browser's ms). */
export function afterRead(
  memory: RenewalMemory,
  data: { reissue?: unknown; certificate_fingerprint?: unknown; now?: unknown },
  clientNow: number,
): RenewalMemory {
  const skew = typeof data.now === "number" ? data.now * 1000 - clientNow : memory.skew;
  const now = clientNow + skew;
  const fingerprint = typeof data.certificate_fingerprint === "string" && data.certificate_fingerprint
    ? data.certificate_fingerprint : null;
  // Settled (F2): a read this long after the change came over a new connection.
  const base = memory.changedAt !== null && now - memory.changedAt > SETTLED_AFTER_MS
    ? { ...FRESH_MEMORY, firstFingerprint: fingerprint } : memory;
  const firstFingerprint = base.firstFingerprint ?? fingerprint;
  const changedAt = base.changedAt
    ?? (fingerprint && firstFingerprint && fingerprint !== firstFingerprint ? now : null);
  const planned = plannedReissue(data.reissue);
  const fresh = planned && now <= planned.reissue_after * 1000 + NOTICE_STALE_AFTER_MS ? planned : null;
  // Kept when the certificate changed under the page; forgotten when nothing changed.
  const notice = fresh ?? (changedAt !== null ? base.notice : null);
  return { notice, firstFingerprint, changedAt, failures: 0, skew };
}

/** The memory after a read that did not reach Vaelor at all. */
export function afterFailedRead(memory: RenewalMemory): RenewalMemory {
  return { ...memory, failures: memory.failures + 1 };
}

/** What the page says, from what it remembers. */
export function renewalView(memory: RenewalMemory, clientNow: number = Date.now()): RenewalView | null {
  const { notice, changedAt, failures } = memory;
  const now = clientNow + memory.skew;
  const noticeAt = notice ? notice.reissue_after * 1000 : null;
  if (failures >= 2) {
    // A notice whose time has come, or a fingerprint seen to change.
    // Times shown are the controller's, moved onto this browser's clock.
    if (noticeAt !== null && now >= noticeAt) return { kind: "lost", at: noticeAt - memory.skew };
    if (changedAt !== null) return { kind: "lost", at: (noticeAt ?? changedAt) - memory.skew };
    return null;
  }
  if (failures === 0 && notice && changedAt === null && noticeAt !== null && now <= noticeAt + NOTICE_STALE_AFTER_MS) {
    return { kind: "announced", at: noticeAt - memory.skew, reason: notice.reason };
  }
  return null;
}

/** Clock time the owner reads, e.g. "14:05". */
export function clockTime(ms: number): string {
  return new Date(ms).toLocaleTimeString([], { hour: "2-digit", minute: "2-digit" });
}

/**
 * Poll the notice, hidden or not (F3), retrying once quickly after a failed
 * read, and say what the memory means.
 */
export function useCertificateRenewal(): RenewalView | null {
  const [memory, setMemory] = useState<RenewalMemory>(FRESH_MEMORY);
  const [now, setNow] = useState(() => Date.now());

  useEffect(() => {
    let cancelled = false;
    let failed = 0;
    let retry: ReturnType<typeof setTimeout> | undefined;
    const check = () => {
      fetch(TRANSPORT_PATH, { cache: "no-store", credentials: "same-origin" })
        .then(async (response) => {
          if (cancelled) return;
          failed = 0;
          setNow(Date.now());
          // Signed out, a restart, an older controller: Vaelor answered, but
          // with nothing to read - it still reached Vaelor.
          const body = response.ok
            ? ((await response.json().catch(() => null)) as { data?: Record<string, unknown> } | null)
            : null;
          if (cancelled) return;
          setMemory((current) => (body?.data && "reissue" in body.data
            ? afterRead(current, body.data, Date.now())
            : { ...current, failures: 0 }));
        })
        .catch(() => {
          if (cancelled) return;
          failed += 1;
          setNow(Date.now());
          setMemory(afterFailedRead);
          if (failed === 1) {
            globalThis.clearTimeout(retry);
            retry = globalThis.setTimeout(check, RENEWAL_RETRY_MS);
          }
        });
    };
    const poll = () => check();
    check();
    const timer = globalThis.setInterval(poll, RENEWAL_CHECK_MS);
    document.addEventListener("visibilitychange", poll);
    return () => {
      cancelled = true;
      globalThis.clearInterval(timer);
      globalThis.clearTimeout(retry);
      document.removeEventListener("visibilitychange", poll);
    };
  }, []);

  return renewalView(memory, now);
}
