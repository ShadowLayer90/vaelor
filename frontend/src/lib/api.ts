import type { ApiEnvelope } from "../types";

export const SESSION_EXPIRED_EVENT = "vaelor:session-expired";
/** Dispatched with the re-read `Session` as `detail` when the signed-in
 *  account's own access changed (CR2); the App adopts it and re-routes. */
export const SESSION_CHANGED_EVENT = "vaelor:session-changed";

const AUTH_ENTRY_PATHS = new Set([
  "/auth/bootstrap",
  "/auth/login",
  "/auth/session",
  "/auth/status",
]);

function signalSessionExpired(path: string, status: number): void {
  if (status === 401 && !AUTH_ENTRY_PATHS.has(path)) {
    window.dispatchEvent(new CustomEvent(SESSION_EXPIRED_EVENT));
  }
}

export class ApiError extends Error {
  status: number;
  code: string;
  details: Record<string, unknown>;

  constructor(message: string, status: number, code = "request_failed", details: Record<string, unknown> = {}) {
    super(message);
    this.name = "ApiError";
    this.status = status;
    this.code = code;
    this.details = details;
  }
}

interface ApiRequestOptions extends RequestInit {
  timeoutMs?: number;
}

/**
 * The caller withdrew the request before it completed — the surface that asked
 * unmounted, or the reader pressed Stop. It is not `request_timeout`: nothing
 * took too long, nobody is waiting, and no retry is owed.
 */
function cancelledRequest(): ApiError {
  return new ApiError("The request was cancelled before it completed.", 499, "request_cancelled");
}

/** Why an attempt stopped: the caller's own withdrawal, or the client's budget. */
function interrupted(callerSignal: RequestInit["signal"]): ApiError {
  return callerSignal?.aborted
    ? cancelledRequest()
    : new ApiError("The appliance took too long to respond. Try again.", 408, "request_timeout");
}

/** Resolves after `ms`, or as soon as `signal` aborts — whichever comes first. */
function delayUnlessAborted(ms: number, signal: AbortSignal): Promise<void> {
  return new Promise((resolve) => {
    if (signal.aborted) { resolve(); return; }
    let timer = 0;
    const abandon = () => { window.clearTimeout(timer); resolve(); };
    timer = window.setTimeout(() => { signal.removeEventListener("abort", abandon); resolve(); }, ms);
    signal.addEventListener("abort", abandon, { once: true });
  });
}

/**
 * Who is still waiting on one shared GET.
 *
 * Concurrent GETs of the same path share one wire request, so a caller's
 * AbortSignal cannot abort that request outright — another surface may be
 * waiting on the same answer. Each signal is a *withdrawal*: the request is
 * abandoned, its retry wait and re-send included, only once every caller has
 * gone. A caller with no signal keeps it alive until it settles.
 */
class SharedInterest {
  private readonly controller = new AbortController();
  private waiting = 0;
  private unconditional = false;

  get signal(): AbortSignal { return this.controller.signal; }

  join(signal: RequestInit["signal"]): void {
    if (!signal) { this.unconditional = true; return; }
    this.waiting += 1;
    signal.addEventListener("abort", () => {
      this.waiting -= 1;
      if (this.waiting === 0 && !this.unconditional) this.controller.abort();
    }, { once: true });
  }
}

/** The shared answer — unless this caller withdraws first, which settles it now. */
function untilWithdrawn<T>(shared: Promise<T>, signal: RequestInit["signal"]): Promise<T> {
  if (!signal) return shared;
  return new Promise<T>((resolve, reject) => {
    const withdraw = () => reject(cancelledRequest());
    signal.addEventListener("abort", withdraw, { once: true });
    shared.then(resolve, reject).finally(() => signal.removeEventListener("abort", withdraw));
  });
}

// A dead pooled keep-alive socket on same-origin (loopback or LAN) rejects in
// single-digit milliseconds (one round-trip: the RST arrives before the request
// is processed),
// whereas a front proxy's idle/read timeout resets a held-open request only
// after tens of seconds. 1000ms sits comfortably above the former and far
// below the latter, so it cleanly separates "never ran" from "reached the
// appliance and may be mid-processing". Only non-idempotent retries are gated
// by it; idempotent methods retry regardless.
const SOCKET_RESET_RETRY_WINDOW_MS = 1000;
const GET_CACHE_TTL_MS = 1500;
const responseCache = new Map<string, { expiresAt: number; value: unknown }>();
const inFlightGets = new Map<string, { promise: Promise<unknown>; interest: SharedInterest }>();
let cacheFetchIdentity: typeof fetch | null = null;
let cacheGeneration = 0;

export function clearApiRequestCache(): void {
  responseCache.clear();
  inFlightGets.clear();
  cacheGeneration += 1;
}

async function apiRequestUncached<T>(
  path: string,
  options: ApiRequestOptions = {},
  csrfToken?: string,
): Promise<T> {
  if (cacheFetchIdentity !== fetch) {
    clearApiRequestCache();
    cacheFetchIdentity = fetch;
  }
  const method = (options.method ?? "GET").toUpperCase();
  const { timeoutMs = ["GET", "HEAD"].includes(method) ? 8000 : 60000, ...fetchOptions } = options;
  const headers = new Headers(options.headers);
  headers.set("Accept", "application/json");
  if (options.body && !(options.body instanceof FormData)) {
    headers.set("Content-Type", "application/json");
  }
  if (!["GET", "HEAD", "OPTIONS"].includes(method) && csrfToken) {
    headers.set("X-CSRF-Token", csrfToken);
  }

  const controller = new AbortController();
  const timeout = window.setTimeout(() => controller.abort(), timeoutMs);
  if (options.signal) {
    if (options.signal.aborted) controller.abort();
    else options.signal.addEventListener("abort", () => controller.abort(), { once: true });
  }
  const idempotent = ["GET", "HEAD", "OPTIONS"].includes(method);
  let response: Response;
  try {
    // A network-level fetch failure (no response received) after a control-plane
    // restart is almost always a reused-but-dead keep-alive socket: the browser
    // sends the first request on a connection the server already closed, the
    // socket resets before the request is processed, and only the NEXT request
    // opens a fresh connection. That surfaced as "The local control service is
    // unavailable" on the first action after every deploy, cleared by a manual
    // retry. Retry once on a fresh connection instead.
    //
    // Invariant: retry only when the request provably never reached the
    // appliance, so a re-send cannot duplicate a mutation. That holds when
    // EITHER the method is idempotent (GET/HEAD/OPTIONS re-send is harmless by
    // definition) OR the rejection arrived within SOCKET_RESET_RETRY_WINDOW_MS.
    // On same-origin access (loopback or over the LAN) a dead pooled socket
    // resets in single-digit milliseconds — one round-trip, well inside the
    // window — because the request never got processed. A non-idempotent request (POST/PUT/PATCH/DELETE with no
    // idempotency key) that stays open PAST that window has reached the
    // appliance and may be mid-processing: a held-open chat POST that a front
    // proxy resets on its idle/read timeout after tens of seconds looks like a
    // network failure but the turn was already accepted and is being answered.
    // Re-sending it would write a second user turn and start a second
    // inference, so it is surfaced as unavailable and left to the caller's
    // live-drop recovery. Aborts (timeout or caller cancel) are never retried,
    // and a cancel that lands during the retry wait ends the wait: a re-send
    // then would put a request on the wire whose answer nobody will read.
    let retriedFreshConnection = false;
    for (;;) {
      const attemptStartedAt = Date.now();
      try {
        response = await fetch(`/api/v2${path}`, {
          ...fetchOptions,
          method,
          headers,
          credentials: "same-origin",
          signal: controller.signal,
        });
        break;
      } catch {
        if (controller.signal.aborted) throw interrupted(options.signal);
        const neverReachedServer = idempotent || Date.now() - attemptStartedAt < SOCKET_RESET_RETRY_WINDOW_MS;
        if (!retriedFreshConnection && neverReachedServer) {
          retriedFreshConnection = true;
          await delayUnlessAborted(200, controller.signal);
          if (controller.signal.aborted) throw interrupted(options.signal);
          continue;
        }
        // The same code is minted for a genuinely unreachable node and for a
        // request the appliance accepted but did not finish in time (a handler
        // that was slow or errored, a proxy closing a held-open POST). Asserting
        // the node is unreachable is only true for the first, so the copy stays
        // honest about both: the request did not complete, and a retry is the
        // next step. It does not blame the connection it cannot confirm is down.
        throw new ApiError(
          "The appliance could not complete that request. It may be busy or briefly unreachable — try again.",
          503,
          "service_unavailable",
        );
      }
    }
  } finally {
    window.clearTimeout(timeout);
  }
  signalSessionExpired(path, response.status);
  let envelope: ApiEnvelope<T>;
  try {
    envelope = (await response.json()) as ApiEnvelope<T>;
  } catch {
    throw new ApiError(
      "The local control service returned an invalid response. Try again.",
      response.status || 502,
      "invalid_response",
    );
  }
  if (!response.ok || !envelope.ok || envelope.data === undefined) {
    throw new ApiError(
      envelope.error?.message ?? "The request could not be completed.",
      response.status,
      envelope.error?.code,
      envelope.error ?? {},
    );
  }
  return envelope.data;
}

export function apiRequest<T>(
  path: string,
  options: ApiRequestOptions = {},
  csrfToken?: string,
): Promise<T> {
  const method = (options.method ?? "GET").toUpperCase();
  // Withdrawn before it started: nothing goes on the wire and no cache is warmed.
  if (options.signal?.aborted) return Promise.reject(cancelledRequest());
  if (!["GET", "HEAD", "OPTIONS"].includes(method)) clearApiRequestCache();
  // A caller's signal used to opt a GET out of the cache, because one shared
  // request could not honour two callers' aborts. SharedInterest can, so the
  // cache is keyed on the path alone: a surface that can cancel its discovery
  // still shares the answer with every other surface asking the same question.
  if (method !== "GET" || options.cache === "no-store") return apiRequestUncached<T>(path, options, csrfToken);
  const cached = responseCache.get(path);
  if (cached && cached.expiresAt > Date.now()) return Promise.resolve(cached.value as T);
  if (cached) responseCache.delete(path);
  const active = inFlightGets.get(path);
  if (active && !active.interest.signal.aborted) {
    active.interest.join(options.signal);
    return untilWithdrawn(active.promise as Promise<T>, options.signal);
  }
  const interest = new SharedInterest();
  interest.join(options.signal);
  const request = apiRequestUncached<T>(path, { ...options, signal: interest.signal }, csrfToken);
  // Read after the request starts: a fetch identity change clears the cache
  // (and bumps the generation) on the way in, and this answer belongs to the
  // generation that survives that.
  const generation = cacheGeneration;
  const tracked = request
    .then((value) => {
      if (generation === cacheGeneration) {
        responseCache.set(path, { expiresAt: Date.now() + GET_CACHE_TTL_MS, value });
      }
      return value;
    })
    .finally(() => {
      if (inFlightGets.get(path)?.promise === tracked) inFlightGets.delete(path);
    });
  inFlightGets.set(path, { promise: tracked, interest });
  return untilWithdrawn(tracked, options.signal);
}

export async function downloadApiRequest(
  path: string,
  fallbackName: string,
  options: RequestInit = {},
  csrfToken?: string,
): Promise<void> {
  const method = (options.method ?? "GET").toUpperCase();
  const headers = new Headers(options.headers);
  headers.set("Accept", "application/octet-stream");
  if (options.body && !(options.body instanceof FormData)) {
    headers.set("Content-Type", "application/json");
  }
  if (!["GET", "HEAD", "OPTIONS"].includes(method) && csrfToken) {
    headers.set("X-CSRF-Token", csrfToken);
  }
  const response = await fetch(`/api/v2${path}`, {
    ...options,
    method,
    credentials: "same-origin",
    headers,
  });
  signalSessionExpired(path, response.status);
  if (!response.ok) {
    let message = "The file could not be downloaded.";
    try {
      const envelope = (await response.json()) as ApiEnvelope<never>;
      message = envelope.error?.message ?? message;
    } catch {
      // Keep the safe fallback for non-JSON proxy errors.
    }
    throw new ApiError(message, response.status, "download_failed");
  }
  const blob = await response.blob();
  const disposition = response.headers.get("Content-Disposition") ?? "";
  const headerName = disposition.match(/filename="?([^";]+)"?/i)?.[1];
  const objectUrl = URL.createObjectURL(blob);
  const link = document.createElement("a");
  link.href = objectUrl;
  link.download = headerName || fallbackName;
  document.body.appendChild(link);
  link.click();
  link.remove();
  URL.revokeObjectURL(objectUrl);
}

export async function downloadApiFile(
  path: string,
  fallbackName: string,
): Promise<void> {
  return downloadApiRequest(path, fallbackName);
}
