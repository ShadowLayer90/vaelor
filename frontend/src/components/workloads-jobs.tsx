import { formatQuantity } from "../lib/format";
import { activityStateLabel, canRetryFromActivity, filesForCandidate, currentAssistantDeployJobId,
  modelForJob, summarizeWorkloadActivity } from "../lib/workloadActivity";
import { jobCanCancel, jobIsReady, jobIsRetryable, jobLabel, jobNeedsAttention, jobIsSuccessful, jobRecoveryGuidance, jobStateLabel, jobSummary, jobTechnicalDetail } from "../lib/jobPresentation";
import { exactTime, timeAgo } from "../lib/format";
import { useOperationOwner } from "../hooks/useOperationOwner";
import { AccelerationVerdict } from "./AccelerationVerdict";
import { GpuChatTier, isGpuChatTier } from "./GpuChatTier";
import { catalogFailureNeedsPortChange } from "./workloads-catalog";
import type { ManagedInventory } from "./WorkloadManager";
import { Button } from "./ui";
import { Notice } from "./ui";
import { Icon } from "./Icon";
import { OperationOwner } from "./OperationOwner";
import { PaginatedItems } from "./PaginatedItems";
import type { WorkloadJob } from "./workloads-types";

export interface ModelDownloadSelection {
  inspectionJobId: string;
  repo: string;
  file: string;
  sizeBytes: number;
}

type JobAction = (job: WorkloadJob, action: "cancel" | "retry") => void | Promise<void>;

export function WorkloadOperation({
  activePlanJob,
  csrfToken,
  managedModels,
  onChangeCatalogPort,
  onDone,
  onManageModels,
  onResourceRefresh,
  onReviewModelDeploy,
  onReviewModelDownload,
}: {
  activePlanJob: WorkloadJob | null;
  csrfToken: string;
  managedModels: ManagedInventory["models"];
  onChangeCatalogPort: (job: WorkloadJob) => void;
  onDone: () => void;
  onManageModels: () => void;
  onResourceRefresh: () => void | Promise<void>;
  onReviewModelDeploy: (job: WorkloadJob) => void;
  onReviewModelDownload: (selection: ModelDownloadSelection) => void;
}) {
  const controller = useOperationOwner({
    csrfToken,
    enabled: Boolean(activePlanJob),
    operationKey: activePlanJob ? `jobs/${activePlanJob.id}` : null,
    onResourceRefresh: () => onResourceRefresh(),
    resumeStorageKey: "vaelor.workloads.operation",
  });
  if (!activePlanJob) return null;
  if (!controller.operation) {
    return controller.error
      ? <Notice heading="Current setup operation could not reconnect" severity="warning">{controller.error}</Notice>
      : <div className="agent-plan agent-plan--operation" role="status"><strong>Current setup operation</strong><span>Connecting to the durable operation record…</span></div>;
  }
  return (
    <OperationOwner className="agent-plan--operation" controller={controller} description={jobSummary(activePlanJob)} onDone={onDone} operation={controller.operation} title="Current setup operation">
      <div className="agent-plan__actions">
        {catalogFailureNeedsPortChange(activePlanJob) ? (
          <Button variant="primary" onClick={() => onChangeCatalogPort(activePlanJob)} type="button">Change port</Button>
        ) : null}
        {activePlanJob.type === "model.download" && jobIsReady(activePlanJob) && activePlanJob.result?.path && <Button variant="primary" onClick={() => onReviewModelDeploy(activePlanJob)}>Review and deploy local AI</Button>}
        {activePlanJob.type === "model.deploy" && jobIsSuccessful(activePlanJob) && <Button variant="primary" onClick={onManageModels}>Manage installed models</Button>}
      </div>
      {activePlanJob.type === "model.inspect" && jobIsReady(activePlanJob) && activePlanJob.result?.candidates?.length ? (
        <div className="job-model-choices">
          <p>Choose the exact verified file Vaelor should download:</p>
          <PaginatedItems
            items={activePlanJob.result.candidates.flatMap((candidate) => filesForCandidate(candidate).map((file) => ({ candidate, file })))}
            label="Verified model choices in setup assistant"
            pageSize={6}
            render={({ candidate, file }) => <span key={`${candidate.id}/${file.name}`}><Button variant="quiet" disabled={!file.size_bytes || file.fits_hardware !== true} onClick={() => onReviewModelDownload({ inspectionJobId: activePlanJob.id, repo: candidate.id, file: file.name, sizeBytes: file.size_bytes })}>Review download · {file.name} · {file.size_bytes ? formatQuantity(file.size_bytes, "model") : "size unavailable"}</Button><small>{file.fit_reason ?? "Vaelor could not prove this file fits the current node."}</small></span>}
          />
        </div>
      ) : null}
    </OperationOwner>
  );
}

export function WorkloadJobActivity({
  applicationResumeJobId,
  jobs,
  managedModels,
  onJobAction,
  onQueueModelDeploy,
  onQueueModelDownload,
  onResumeApplicationResearch,
  requestedDownloads = [],
}: {
  applicationResumeJobId?: string;
  jobs: WorkloadJob[];
  managedModels: ManagedInventory["models"];
  onJobAction: JobAction;
  onQueueModelDeploy: (job: WorkloadJob) => void | Promise<void>;
  onQueueModelDownload: (jobId: string, repo: string, file: string, sizeBytes: number) => void | Promise<void>;
  onResumeApplicationResearch: () => void;
  /** `repo/file` keys already accepted, so approval cannot be repeated. */
  requestedDownloads?: readonly string[];
}) {
  const jobHistory = summarizeWorkloadActivity(jobs, managedModels);
  // #145: only one entry may claim the Assistant's current model.
  const currentDeployId = currentAssistantDeployJobId(jobHistory.visible, managedModels);
  return (
    <section className="data-panel workload-jobs" aria-labelledby="workload-jobs-title">
      <div className="panel-heading">
        <div><h2 id="workload-jobs-title">Recent setup activity</h2><p>Current outcomes are shown here. Completed prerequisite steps and resolved failures are condensed automatically.</p></div>
        <span className="job-foundation-state"><span />Setup service ready</span>
      </div>
      {jobHistory.visible.length ? (
        <div className="job-list">
          <PaginatedItems items={jobHistory.visible} label="Setup activity" pageSize={8} render={(job) => (
            <div className="job-row" key={job.id}>
              <Icon name="activity" />
              <span className="job-row__summary">
                <strong>{jobLabel(job.type)}</strong>
                <small>
                {/* Relative age, with the exact local time on hover. Without a
                    time a user cannot tell a failure from two minutes ago from
                    one two weeks old.

                    **`job.created_at` is already milliseconds.** `jobs.py`
                    writes `int(time.time() * 1000)`; it is the only store that
                    does. `assistant_memory`, `rag_chat`, `security` and
                    `agent_tasks` all write seconds, so every other screen
                    multiplies by 1000 and is right to. This one did too, and
                    dated every row in the year 58,578 while the relative label
                    read "just now" for all of them - a future date is not
                    "ago", so the only visible symptom was that history
                    collapsed into a single instant. Found 2026-08-11. */}
                <time dateTime={new Date(job.created_at).toISOString()} title={exactTime(job.created_at)}>
                  {timeAgo(job.created_at)}
                </time>
                {" · "}{jobSummary(job)} · attempt {job.attempt}
                  {jobHistory.resolvedByJobId.has(job.id)
                  ? ` · resolved ${jobHistory.resolvedByJobId.get(job.id)} earlier ${jobHistory.resolvedByJobId.get(job.id) === 1 ? "failure" : "failures"}`
                    : jobHistory.earlierAttemptsByJobId.has(job.id)
                    ? ` · replaces ${jobHistory.earlierAttemptsByJobId.get(job.id)} earlier ${jobHistory.earlierAttemptsByJobId.get(job.id) === 1 ? "attempt" : "attempts"}`
                      : ""}
                </small>
                {/* aria-label on a role-less div is not exposed to screen
                    readers; progress needs real progressbar semantics. */}
                <span
                  aria-label={`${job.progress}% complete`}
                  aria-valuemax={100}
                  aria-valuemin={0}
                  aria-valuenow={job.progress}
                  className="job-progress"
                  role="progressbar"
                ><span style={{ width: `${job.progress}%` }} /></span>
                {jobRecoveryGuidance(job) && (
                  <span className="job-row__guidance">
                    {jobRecoveryGuidance(job)?.cause} {jobRecoveryGuidance(job)?.recovery}
                  </span>
                )}
                {jobTechnicalDetail(job) && (
                  <details className="job-row__technical">
                    <summary>Technical details</summary>
                    <code>{jobTechnicalDetail(job)}</code>
                  </details>
                )}
              </span>
              <span className="job-row__controls">
                <span className={`event-state event-state--${jobNeedsAttention(job) ? "failure" : jobIsReady(job) ? "success" : "progress"}`}>{activityStateLabel(job, managedModels, currentDeployId) ?? jobStateLabel(job)}</span>
                {applicationResumeJobId === job.id ? <Button variant="quiet" onClick={onResumeApplicationResearch}>Resume application research</Button> : jobCanCancel(job) ? (
                  <Button variant="quiet" onClick={() => void onJobAction(job, "cancel")}>Cancel</Button>
                ) : canRetryFromActivity(job) && jobIsRetryable(job) ? (
                  <Button variant="quiet" onClick={() => void onJobAction(job, "retry")}>Retry</Button>
                ) : null}
              </span>
              {job.type === "model.inspect" && jobIsReady(job) && job.result?.candidates?.length ? (
                <div className="job-model-choices">
                  <PaginatedItems
                    items={job.result.candidates.flatMap((candidate) => filesForCandidate(candidate).map((file) => ({ candidate, file })))}
                    label="Verified model choices"
                    pageSize={6}
                    render={({ candidate, file }) => {
                      // Approval is a one-shot control: once this artefact has
                      // been accepted the button cannot start a second
                      // multi-gigabyte transfer for it.
                      const alreadyRequested = requestedDownloads.includes(`${candidate.id}/${file.name}`);
                      return (
                        <span key={`${candidate.id}/${file.name}`}>
                          <Button variant="quiet" disabled={alreadyRequested || !file.size_bytes || file.fits_hardware !== true} onClick={() => void onQueueModelDownload(job.id, candidate.id, file.name, file.size_bytes)}>
                    {alreadyRequested ? "Download approved" : "Approve download"} · {file.name} · {file.size_bytes ? formatQuantity(file.size_bytes, "model") : "size unavailable"}
                          </Button>
                          <small>{alreadyRequested ? "Already approved in this session. Progress is in the setup activity above." : file.fit_reason ?? (candidate.gated ? "Hugging Face access is checked through the credential broker." : "Compatibility data is stale. Run the model check again.")}</small>
                        </span>
                      );
                    }}
                  />
                </div>
              ) : null}
              {job.type === "model.inspect" && jobIsReady(job) && job.result?.matching_files === 0 ? (
                <div className="job-model-choices"><small>No exact verified GGUF file matched. Nothing was downloaded; refine the repository or file name and check again.</small></div>
              ) : null}
              {job.type === "model.download" && jobIsReady(job) && job.result?.path && !modelForJob(job, managedModels) ? (
                <div className="job-model-choices">
                  <Button variant="primary" onClick={() => void onQueueModelDeploy(job)}>Deploy local AI server</Button>
                      <small>{job.result.file} · {job.result.size_bytes ? formatQuantity(job.result.size_bytes, "model") : "downloaded"}</small>
                </div>
              ) : null}
              {job.type === "model.deploy" && jobIsSuccessful(job) && job.result?.endpoint ? (
                <div className="job-model-active" role="status">
                  <span><Icon name="shield" /></span>
                  <div>
                    <strong>{job.id === currentDeployId ? "Active in Assistant" : modelForJob(job, managedModels) ? "Local model available" : "Deployment completed"}</strong>
                    <small>{job.id === currentDeployId ? "This is the Assistant's current private model." : modelForJob(job, managedModels) ? "This model is installed. Switch models from Manage when you want to use it." : "Open Manage to see the current installed-model state."}</small>
                  </div>
                </div>
              ) : null}
              {/*
                * A deployment that succeeded can still have got less than it
                * asked for, and both ways of doing so are silent: the CPU
                * fallback answers `/health` with 200, and a capped context
                * window starts normally. The deploy's own readings stay with
                * the deploy — this is the only place the context verdict
                * exists at all, and the accelerator reading here is the
                * differential one, taken against a baseline read while nothing
                * of ours was resident.
                */}
              {job.type === "model.deploy" && (job.result?.acceleration || job.result?.context_built) ? (
                <AccelerationVerdict
                  acceleration={job.result.acceleration}
                  context={job.result.context_built}
                  /* Named per deploy (#150): eight identical headings gave a
                     screen-reader outline no way to tell one server's verdict
                     from another's. The model file stem is the same
                     vocabulary this screen already uses for downloads. */
                  title={`What the ${String(job.result?.path ?? "").split("/").pop()?.replace(/\.gguf$/i, "") || "model"} server got`}
                />
              ) : null}
              {/*
                * The GPU AI-Chat tier tells the owner, in the backend's own
                * words, whether this is the recommended optimized build or a
                * standard model, and whether it landed fully on the GPU, split
                * with the CPU, or on the CPU. Present only when the deploy went
                * down the GPU fork, so a plain Assistant deploy shows nothing.
                */}
              {job.type === "model.deploy" && jobIsSuccessful(job) && job.result && isGpuChatTier(job.result) ? (
                <GpuChatTier result={job.result} />
              ) : null}
            </div>
          )} />
        </div>
      ) : (
        <div className="empty-state"><Icon name="shield" /><strong>Nothing is being installed</strong><span>Approved setups and their progress will appear here.</span></div>
      )}
    </section>
  );
}
