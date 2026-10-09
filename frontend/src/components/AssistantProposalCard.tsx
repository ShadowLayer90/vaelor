import { Icon } from "./Icon";
import type { ProposedJob } from "./ActionReviewDialog";
import { Button } from "./ui";
import { destinations } from "../lib/destinations";
import { jobLabel } from "../lib/jobPresentation";

/**
 * What an answer proposes, as one row (VD-200, the AssistAnswer board): an
 * action waiting for review, or a deployment that continues on Apps and AI.
 * The deployment card named the page by its pre-rename name, Workloads; the
 * owner's fix on the approved board is the page's own name, read from the
 * destination table so it cannot drift again.
 */
export function AssistantProposalCard({
  job, busy, onContinue, onReview,
}: {
  job: ProposedJob;
  busy: boolean;
  onContinue: () => void;
  onReview: () => void;
}) {
  const workload = job.type === "model.inspect" || job.type === "compose.install";
  const place = destinations.workloads.name;
  return (
    <div className={workload ? "as-extra" : "as-extra as-extra--review"}>
      <span aria-hidden="true" className={workload ? "ui-row__icon" : "ui-row__icon ui-row__icon--accent"}>
        <Icon name={workload ? "package" : "shield"} size={16} />
      </span>
      <span className="as-extra__text">
        <strong>{workload ? `Continue in ${place}` : "Action ready for review"}</strong>
        <span className="as-small as-muted">{workload ? "Research, approval, deployment, and verification stay together" : `${jobLabel(job.type)} · nothing has run`}</span>
      </span>
      <Button disabled={busy} onClick={workload ? onContinue : onReview} type="button" variant={workload ? "secondary" : "primary"}>
        {workload ? `Open ${place}` : "Review action"}
      </Button>
    </div>
  );
}
