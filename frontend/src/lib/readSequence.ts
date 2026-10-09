/**
 * Newest-issued-wins for reads that more than one caller issues (SC6).
 *
 * Fleet re-reads the cluster summary and the capacity ledger from two places:
 * the background poll and the reload that follows an operator's action. If a
 * poll was already in flight when an action's reload landed, the poll's OLDER
 * answer could arrive last and overwrite the post-action state. Every read
 * takes a ticket when it is ISSUED; its answer is applied only if no newer read
 * of the same resource has been issued since. A superseded answer is dropped,
 * whatever order the answers arrive in.
 */
export interface ReadSequencer<K extends string> {
  /** Take a ticket for a read of `kind` that is about to be issued. */
  issue(kind: K): number;
  /** Whether `ticket` is still the newest read issued for `kind`. */
  isLatest(kind: K, ticket: number): boolean;
}

export function createReadSequencer<K extends string>(): ReadSequencer<K> {
  const latest = new Map<K, number>();
  let counter = 0;
  return {
    issue(kind) {
      counter += 1;
      latest.set(kind, counter);
      return counter;
    },
    isLatest(kind, ticket) {
      return latest.get(kind) === ticket;
    },
  };
}
