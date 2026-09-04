import { ModalShell } from "./ModalShell";
import type { AgentWriteProposal } from "./agentTypes";
import { Button } from "./ui";

export function AgentWriteReviewDialog({
  proposal,
  busy,
  onApprove,
  onClose,
}: {
  proposal: AgentWriteProposal | null;
  busy: boolean;
  onApprove: () => void;
  onClose: () => void;
}) {
  if (!proposal) return null;
  return (
    <ModalShell labelledBy="agent-write-review-title" onClose={onClose}>
      <header className="dialog__header">
        <div>
          <span className="section-eyebrow">Approval required · nothing written</span>
          <h2 id="agent-write-review-title">Review knowledge document</h2>
          <p>The agent cannot alter this content after you approve it. Vaelor will store this exact document in the selected collection and record the action.</p>
        </div>
        <Button aria-label="Close review" className="icon-button" disabled={busy} onClick={onClose} type="button">×</Button>
      </header>
      <div className="agent-write-review__facts">
        <div><small>Destination</small><strong>{proposal.collection_name || proposal.collection_id}</strong></div>
        <div><small>Document name</small><strong>{proposal.name}</strong></div>
        <div><small>Format</small><strong>{proposal.media_type}</strong></div>
      </div>
      <section className="agent-write-review__content">
        <h3>Exact proposed content</h3>
        <pre>{proposal.content}</pre>
      </section>
      <footer className="dialog__actions">
        <Button variant="quiet" disabled={busy} onClick={onClose} type="button">Keep pending</Button>
        <Button variant="primary" disabled={busy} onClick={onApprove} type="button">{busy ? "Writing…" : "Approve knowledge write"}</Button>
      </footer>
    </ModalShell>
  );
}
