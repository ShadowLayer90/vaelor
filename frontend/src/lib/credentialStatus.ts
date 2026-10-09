import type { StatusTone } from "../components/ui/status";
import { timeAgo } from "./format";

/**
 * One stored credential as `GET /credentials` lists it
 * (`vaelor.credential_listing.connection_listing`). Inbound endpoint keys are
 * not in this list; they are counted in `endpoint_keys` instead.
 */
export interface StoredCredential {
  id: string;
  provider: string;
  label: string;
  fingerprint: string;
  created_at?: number;
  updated_at?: number;
  last_test_status?: string | null;
  last_tested_at?: number | null;
  /** When a real use was recorded (a Test also stamps it, at its own time). */
  last_used_at?: number | null;
  active_for?: string[];
  selected_model?: string;
  kind_label?: string;
  ai_connection?: boolean;
  /** VD-206: a hosted connection; prompts and files sent to it leave this machine. */
  sends_off_machine?: boolean;
  /** Whether the broker has a real connection test for this kind. */
  testable?: boolean;
  managed_by?: string | null;
  manage_note?: string;
  /** Plain names of what is assigned to use it (`credential_listing.used_by`). */
  used_by?: string[];
  can_disconnect?: boolean;
}

export interface CredentialListing {
  credentials: StoredCredential[];
  endpoint_keys?: { active: number; revoked: number };
}

export interface CredentialStatus {
  label: string;
  tone: StatusTone;
  detail: string;
}

/**
 * Whether the Assistant's saved connections should offer this credential. The
 * server decides (`credential_listing.AI_CONNECTION_KINDS`); a row that does
 * not say is not offered, rather than guessed from a second copy of the list.
 */
export function isAiConnection(item: StoredCredential) {
  return item.ai_connection === true;
}

function ago(seconds: number, now: number) {
  return timeAgo(seconds * 1000, now);
}

function testSentence(item: StoredCredential, now: number) {
  if (item.last_test_status !== "success" && item.last_test_status !== "failed") return "Never tested";
  const verb = item.last_test_status === "success" ? "passed" : "failed";
  return item.last_tested_at ? `Last test ${verb} ${ago(item.last_tested_at, now)}` : `Last test ${verb} · date not recorded`;
}

/**
 * The one status an owner reads for a stored credential (ACC-068/104/107).
 *
 * Every credential used to read "Not tested" - including ones Vaelor uses on
 * every request. Three recorded facts are read: a real use (`last_used_at`,
 * recorded where the credential is actually used - a lease, a listing or a
 * page view is NOT a use), a test (its result and time), and a rotation
 * (`updated_at`). The pill shows whichever happened MOST RECENTLY and the
 * detail line keeps the other fact, so an old failure is not hidden behind a
 * newer use and a replaced key does not keep the old key's pass. With no use
 * on record it says so ("No use recorded yet"), never "Not tested".
 *
 * A use is a record, not proof the credential worked, so it is never shown in
 * the success colour; only a passing test is.
 */
export function credentialStatus(item: StoredCredential, now = Date.now()): CredentialStatus {
  const tested = item.last_test_status === "success" || item.last_test_status === "failed";
  const testAt = tested ? item.last_tested_at ?? 0 : 0;
  // `last_used_at` has one writer, the broker's record_use (a Test writes
  // only its own result and time), so every recorded value is a real use.
  const usedAt = item.last_used_at ?? 0;
  const changedAt = item.updated_at && item.created_at && item.updated_at > item.created_at ? item.updated_at : 0;

  if (changedAt && changedAt > testAt && changedAt > usedAt) {
    return {
      label: `Key replaced ${ago(changedAt, now)}`,
      tone: "neutral",
      detail: "No use or test recorded since it was replaced",
    };
  }
  if (usedAt && usedAt >= testAt) {
    return {
      label: `Used by Vaelor ${ago(usedAt, now)}`,
      tone: "neutral",
      detail: testSentence(item, now),
    };
  }
  if (tested) {
    const passed = item.last_test_status === "success";
    const when = item.last_tested_at ? ago(item.last_tested_at, now) : "· date not recorded";
    return {
      label: `${passed ? "Test passed" : "Test failed"} ${when}`,
      tone: passed ? "success" : "warning",
      detail: usedAt ? `Last used by Vaelor ${ago(usedAt, now)}` : "No use recorded yet",
    };
  }
  return { label: "No use recorded yet", tone: "neutral", detail: "Never tested" };
}

function joinNames(names: string[]) {
  if (names.length <= 1) return names.join("");
  return `${names.slice(0, -1).join(", ")} and ${names[names.length - 1]}`;
}

/**
 * What deleting this credential stops, in plain words (S-A). Only a credential
 * nothing is assigned to - and that the server offered for removal because
 * nothing references it - is called unused.
 */
export function removalConsequence(item: StoredCredential) {
  const names = item.used_by ?? [];
  if (!names.length) return "Nothing on this appliance uses it now.";
  return `Deleting it stops ${joinNames(names)} from using it until you connect another.`;
}
