import { ModalShell } from "./ModalShell";
import type { AgentWriteProposal } from "./agentTypes";
import { Button } from "./ui";

/**
 * The exact knowledge document an agent run proposes, shown before anything is
 * written (the "Review knowledge document" dialog on the History board).
 *
 * Nothing is stored until "Approve knowledge write"; "Keep pending" leaves the
 * proposal on the run untouched. A refusal of the approval is this dialog's
 * own error (VD-189), so it is passed to ModalShell and said inside the
 * dialog, never on the inert page underneath.
 */
export function AgentWriteReviewDialog({
  proposal,
  busy,
  error,
  onApprove,
  onClose,
}: {
  proposal: AgentWriteProposal | null;
  busy: boolean;
  /** Why the write was refused (VD-189): shown inside this dialog. */
  error?: string;
  onApprove: () => void;
  onClose: () => void;
}) {
  if (!proposal) return null;
  return (
    <ModalShell
      className="as-dialog ah-write-review"
      describedBy="agent-write-review-description"
      error={error || undefined}
      labelledBy="agent-write-review-title"
      onClose={onClose}
    >
      <header className="as-dialog__head">
        <div>
          <span className="as-label">Approval required · nothing written</span>
          <h2 id="agent-write-review-title">Review knowledge document</h2>
          <p id="agent-write-review-description">The agent cannot alter this content after you approve it. Vaelor will store this exact document in the selected collection and record the action.</p>
        </div>
        <Button aria-label="Close review" className="as-btn-ghost" disabled={busy} onClick={onClose} type="button" variant="quiet">Close</Button>
      </header>
      <div className="as-dialog__body">
        <dl className="as-kv ah-write-review__facts">
          <div><dt>Destination</dt><dd>{proposal.collection_name || proposal.collection_id}</dd></div>
          <div><dt>Document name</dt><dd>{proposal.name}</dd></div>
          <div><dt>Format</dt><dd className="as-mono">{proposal.media_type}</dd></div>
        </dl>
        <section aria-labelledby="agent-write-review-content" className="ah-write-review__content">
          <h3 className="as-label" id="agent-write-review-content">Exact proposed content</h3>
          <pre>{proposal.content}</pre>
        </section>
      </div>
      <footer className="as-dialog__foot">
        <Button disabled={busy} onClick={onClose} type="button">Keep pending</Button>
        <Button variant="primary" disabled={busy} onClick={onApprove} type="button">{busy ? "Writing…" : "Approve knowledge write"}</Button>
      </footer>
    </ModalShell>
  );
}
