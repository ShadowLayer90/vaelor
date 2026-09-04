import { Icon } from "./Icon";
import type { ProposedJob } from "./ActionReviewDialog";
import { Button } from "./ui";

export function AssistantProposalCard({
  job, busy, onContinue, onReview,
}: {
  job: ProposedJob;
  busy: boolean;
  onContinue: () => void;
  onReview: () => void;
}) {
  const workload = job.type === "model.inspect" || job.type === "compose.install";
  return (
    <div className="assistant-proposal">
      <Icon name="shield" />
      <span>
        <strong>{workload ? "Continue in Workloads" : "Action ready for review"}</strong>
        <small>{workload ? "Research, approval, deployment, and verification stay together" : `${job.type.replaceAll(".", " ")} · nothing has run`}</small>
      </span>
      <Button disabled={busy} onClick={workload ? onContinue : onReview} type="button" variant="quiet">
        {workload ? "Open Workloads" : "Review action"}
      </Button>
    </div>
  );
}
