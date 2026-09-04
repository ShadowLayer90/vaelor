/*
 * Drafts that survive navigation, with two guards the first version lacked:
 *
 * - Keys are namespaced by account. The repository's other persisted state is
 *   already keyed this way, and an unkeyed draft leaks: administrator A
 *   half-types a question naming a sensitive incident and signs out;
 *   administrator B signs in on the same tab and finds A's text presented as
 *   B's own draft.
 * - Every access is wrapped. With browser storage blocked, the
 *   `sessionStorage` getter itself throws — a draft must degrade to "does not
 *   survive navigation", never to a white screen.
 *
 * Credentials are never stored here; the callers hold that rule.
 */

export function readDraft(key: string, username: string): string {
  try {
    return sessionStorage.getItem(`${key}:${username}`) ?? "";
  } catch {
    return "";
  }
}

export function writeDraft(key: string, username: string, value: string): void {
  try {
    sessionStorage.setItem(`${key}:${username}`, value);
  } catch {
    // Storage unavailable: the draft simply does not survive navigation.
  }
}
