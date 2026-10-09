import type { ReactNode } from "react";
import { formatQuantity } from "../lib/format";
import { activityStateLabel, canRetryFromActivity, filesForCandidate, currentAssistantDeployJobId,
  modelForJob, summarizeWorkloadActivity } from "../lib/workloadActivity";
import { canonicalOperationState, jobCanCancel, jobIsReady, jobIsRetryable, jobLabel, jobNeedsAttention, jobIsSuccessful, jobRecoveryGuidance, jobStateLabel, jobSummary, jobTechnicalDetail } from "../lib/jobPresentation";
import { exactTime, timeAgo } from "../lib/format";
import { useOperationOwner } from "../hooks/useOperationOwner";
import { AccelerationVerdict } from "./AccelerationVerdict";
import { AppsIconTile, AppsInset, AppsProgress } from "./appsKit";
import { GpuChatTier, isGpuChatTier } from "./GpuChatTier";
import { Icon, type IconName } from "./Icon";
import { catalogFailureNeedsPortChange } from "./workloads-catalog";
import type { ManagedInventory } from "./WorkloadManager";
import { Button, Card, EmptyState, LoadingLines, Notice } from "./ui";
import type { StatusTone } from "./ui/status";
import { OperationOwner } from "./OperationOwner";
import { usePagination } from "./PaginatedItems";
import { StatusPill } from "./StatusPill";
import type { WorkloadJob } from "./workloads-types";
import "../styles/apps-install.css";

/*
 * The setup operation and the recent setup activity (VD-200, the
 * AppsSetupActivity board): one card per job, and the details a job carries
 * (model file choices, the deploy's readings, the GPU chat tier) under its row.
 */

export interface ModelDownloadSelection {
  inspectionJobId: string;
  repo: string;
  file: string;
  sizeBytes: number;
}

type JobAction = (job: WorkloadJob, action: "cancel" | "retry") => void | Promise<void>;
type Candidate = NonNullable<NonNullable<WorkloadJob["result"]>["candidates"]>[number];
type CandidateFile = Candidate["files"][number];

const FILES_PER_PAGE = 6;
const JOBS_PER_PAGE = 8;

/** Previous / Next under a list, as the boards draw it: "Showing 1-6 of 9 files". */
function ListPager({ label, noun, page, pageSize, setPage, total, totalPages }: {
  label: string; noun: string; page: number; pageSize: number;
  setPage: (next: number | ((current: number) => number)) => void; total: number; totalPages: number;
}) {
  if (totalPages <= 1) return null;
  const first = (page - 1) * pageSize + 1;
  const last = Math.min(total, page * pageSize);
  return (
    <nav aria-label={`${label} pages`} className="apps-pager">
      <span>{`Showing ${first}-${last} of ${total}${noun ? ` ${noun}` : ""}`}</span>
      <span className="apps-pager__buttons">
        <Button disabled={page === 1} onClick={() => setPage((current) => Math.max(1, current - 1))} variant="quiet">Previous</Button>
        <Button disabled={page === totalPages} onClick={() => setPage((current) => Math.min(totalPages, current + 1))}>Next</Button>
      </span>
    </nav>
  );
}

/** The verified files a compatibility check found, each with its own approval. */
function ModelFileChoices({ choices, label, renderAction }: {
  choices: Array<{ candidate: Candidate; file: CandidateFile }>;
  label: string;
  renderAction: (candidate: Candidate, file: CandidateFile) => { button: ReactNode; detail: string };
}) {
  const { page, setPage, totalPages, visible } = usePagination(choices, FILES_PER_PAGE);
  return (
    <div className="apps-job__choices">
      <ul aria-label={label} className="apps-job__files">
        {visible.map(({ candidate, file }) => {
          const action = renderAction(candidate, file);
          return (
            <li className="apps-job__file" key={`${candidate.id}/${file.name}`}>
              <div className="apps-job__file-text">
                <strong className="apps-mono">{file.name}</strong>
                <span>{file.size_bytes ? formatQuantity(file.size_bytes, "model") : "size unavailable"} · {action.detail}</span>
              </div>
              {action.button}
            </li>
          );
        })}
      </ul>
      <ListPager label={label} noun="files" page={page} pageSize={FILES_PER_PAGE} setPage={setPage} total={choices.length} totalPages={totalPages} />
    </div>
  );
}

const fileChoices = (job: WorkloadJob) => (job.result?.candidates ?? [])
  .flatMap((candidate) => filesForCandidate(candidate).map((file) => ({ candidate, file })));

export function WorkloadOperation({
  activePlanJob,
  csrfToken,
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
      ? <Notice className="apps-operation-lost" heading="Current setup operation could not reconnect" severity="warning">{controller.error}</Notice>
      : (
        <Card as="section" className="apps-operation-connecting" heading="Current setup operation">
          <span>Connecting to the durable operation record…</span>
          <LoadingLines label="Connecting to the durable operation record" lines={1} />
        </Card>
      );
  }
  const choices = activePlanJob.type === "model.inspect" && jobIsReady(activePlanJob) ? fileChoices(activePlanJob) : [];
  return (
    <OperationOwner
      actions={<>
        {catalogFailureNeedsPortChange(activePlanJob) ? (
          <Button variant="primary" onClick={() => onChangeCatalogPort(activePlanJob)} type="button">Change port</Button>
        ) : null}
        {activePlanJob.type === "model.download" && jobIsReady(activePlanJob) && activePlanJob.result?.path && <Button variant="primary" onClick={() => onReviewModelDeploy(activePlanJob)}>Review and deploy local AI</Button>}
        {activePlanJob.type === "model.deploy" && jobIsSuccessful(activePlanJob) && <Button variant="primary" onClick={onManageModels}>Manage installed models</Button>}
      </>}
      className="apps-operation"
      controller={controller}
      description={jobSummary(activePlanJob)}
      onDone={onDone}
      operation={controller.operation}
      title="Current setup operation"
    >
      {choices.length ? (
        <>
          <p className="apps-operation__choices-lead">Choose the exact verified file Vaelor should download:</p>
          <ModelFileChoices
            choices={choices}
            label="Verified model choices in setup assistant"
            renderAction={(candidate, file) => ({
              detail: file.fit_reason ?? "Vaelor could not prove this file fits the current node.",
              button: (
                <Button disabled={!file.size_bytes || file.fits_hardware !== true} onClick={() => onReviewModelDownload({ inspectionJobId: activePlanJob.id, repo: candidate.id, file: file.name, sizeBytes: file.size_bytes })}>
                  Review download<span className="sr-only">{` · ${file.name}`}</span>
                </Button>
              ),
            })}
          />
        </>
      ) : null}
    </OperationOwner>
  );
}

function jobIcon(type: string): IconName {
  if (type === "model.download") return "download";
  if (type === "model.inspect") return "search";
  if (type.startsWith("model.")) return "server";
  if (type.startsWith("compose.")) return "package";
  if (type.startsWith("application.")) return "file";
  if (type.startsWith("host.docker")) return "database";
  return "activity";
}

/** A job's pill: red only for attention, green only for a finished good outcome, blue while it works. */
export function jobTone(job: WorkloadJob): StatusTone {
  if (jobNeedsAttention(job)) return "danger";
  if (jobIsReady(job)) return "success";
  const state = canonicalOperationState(job);
  return state === "running" || state === "queued" ? "info" : "neutral";
}

const sentenceCase = (value: string) => value.charAt(0).toUpperCase() + value.slice(1);

export function WorkloadJobActivity({
  applicationResumeJobId,
  jobs,
  managedModels,
  onJobAction,
  onQueueModelDeploy,
  onQueueModelDownload,
  onResumeApplicationResearch,
  requestedDownloads = [],
  stalledJobIds = [],
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
  /** Active jobs with no change for five minutes (FE-W7-7, `jobPollCadence`). */
  stalledJobIds?: readonly string[];
}) {
  const jobHistory = summarizeWorkloadActivity(jobs, managedModels);
  // #145: only one entry may claim the Assistant's current model.
  const currentDeployId = currentAssistantDeployJobId(jobHistory.visible, managedModels);
  const { page, setPage, totalPages, visible } = usePagination(jobHistory.visible, JOBS_PER_PAGE);
  const stalled = jobs.filter((job) => stalledJobIds.includes(job.id));
  return (
    <Card
      actions={<StatusPill label="Setup service ready" tone="neutral" />}
      as="section"
      className="apps-activity"
      description="Current outcomes are shown here. Completed prerequisite steps and resolved failures are condensed automatically."
      heading="Recent setup activity"
    >
      {stalled.map((job) => (
        <Notice key={`stalled-${job.id}`} severity="warning">
          {`${jobLabel(job.type)} has shown no progress for over five minutes. It may be stuck: check its details below, or cancel it if it can be cancelled.`}
        </Notice>
      ))}
      {jobHistory.visible.length ? (
        <>
          <ul aria-label="Setup activity" className="apps-job-list">
            {visible.map((job) => <JobRow
              applicationResumeJobId={applicationResumeJobId}
              currentDeployId={currentDeployId}
              earlierAttempts={jobHistory.earlierAttemptsByJobId.get(job.id)}
              job={job}
              key={job.id}
              managedModels={managedModels}
              onJobAction={onJobAction}
              onQueueModelDeploy={onQueueModelDeploy}
              onQueueModelDownload={onQueueModelDownload}
              onResumeApplicationResearch={onResumeApplicationResearch}
              requestedDownloads={requestedDownloads}
              resolved={jobHistory.resolvedByJobId.get(job.id)}
            />)}
          </ul>
          <ListPager label="Setup activity" noun="" page={page} pageSize={JOBS_PER_PAGE} setPage={setPage} total={jobHistory.visible.length} totalPages={totalPages} />
        </>
      ) : (
        <EmptyState icon={<Icon name="shield" />} text="Approved setups and their progress will appear here." title="Nothing is being installed" />
      )}
    </Card>
  );
}

function JobRow({
  applicationResumeJobId, currentDeployId, earlierAttempts, job, managedModels, onJobAction,
  onQueueModelDeploy, onQueueModelDownload, onResumeApplicationResearch, requestedDownloads, resolved,
}: {
  applicationResumeJobId?: string;
  currentDeployId: string | null;
  earlierAttempts?: number;
  job: WorkloadJob;
  managedModels: ManagedInventory["models"];
  onJobAction: JobAction;
  onQueueModelDeploy: (job: WorkloadJob) => void | Promise<void>;
  onQueueModelDownload: (jobId: string, repo: string, file: string, sizeBytes: number) => void | Promise<void>;
  onResumeApplicationResearch: () => void;
  requestedDownloads: readonly string[];
  resolved?: number;
}) {
  const guidance = jobRecoveryGuidance(job);
  const technical = jobTechnicalDetail(job);
  const choices = job.type === "model.inspect" && jobIsReady(job) ? fileChoices(job) : [];
  const history = resolved
    ? ` · resolved ${resolved} earlier ${resolved === 1 ? "failure" : "failures"}`
    : earlierAttempts
      ? ` · replaces ${earlierAttempts} earlier ${earlierAttempts === 1 ? "attempt" : "attempts"}`
      : "";
  const noMatch = job.type === "model.inspect" && jobIsReady(job) && job.result?.matching_files === 0;
  const deployable = job.type === "model.download" && jobIsReady(job) && Boolean(job.result?.path) && !modelForJob(job, managedModels);
  const deployed = job.type === "model.deploy" && jobIsSuccessful(job) && Boolean(job.result?.endpoint);
  const readings = job.type === "model.deploy" && Boolean(job.result?.acceleration || job.result?.context_built);
  const gpuTier = job.type === "model.deploy" && jobIsSuccessful(job) && Boolean(job.result) && isGpuChatTier(job.result);
  const hasBody = Boolean(guidance || technical || choices.length || noMatch || deployable || deployed || readings || gpuTier);
  return (
    <li className="apps-job">
      <div className="apps-job__head">
        <AppsIconTile name={jobIcon(job.type)} />
        <div className="apps-job__summary">
          <strong>{jobLabel(job.type)}</strong>
          <span className="apps-job__meta">
            {/* `job.created_at` is already milliseconds (jobs.py writes
                time.time() * 1000), unlike every other store. */}
            <time dateTime={new Date(job.created_at).toISOString()} title={exactTime(job.created_at)}>{timeAgo(job.created_at)}</time>
            {" · "}{jobSummary(job)} · attempt {job.attempt}{history}
          </span>
          {/* The bar is drawn only while the job is running; a waiting or ended job has no progress to report. */}
          {canonicalOperationState(job) === "running" && <AppsProgress fraction={job.progress / 100} label={`${job.progress}% complete`} />}
        </div>
        <div className="apps-job__controls">
          <StatusPill label={sentenceCase(activityStateLabel(job, managedModels, currentDeployId) ?? jobStateLabel(job))} tone={jobTone(job)} />
          {applicationResumeJobId === job.id ? <Button variant="quiet" onClick={onResumeApplicationResearch}>Resume application research</Button> : jobCanCancel(job) ? (
            <Button variant="quiet" onClick={() => void onJobAction(job, "cancel")}>Cancel</Button>
          ) : canRetryFromActivity(job) && jobIsRetryable(job) ? (
            <Button variant="quiet" onClick={() => void onJobAction(job, "retry")}>Retry</Button>
          ) : null}
        </div>
      </div>
      {hasBody && (
        <div className="apps-job__body">
          {guidance && <p className="apps-job__guidance">{guidance.cause} {guidance.recovery}</p>}
          {technical && (
            <details className="apps-disclosure">
              <summary>Technical details</summary>
              <code>{technical}</code>
            </details>
          )}
          {choices.length ? (
            <ModelFileChoices
              choices={choices}
              label="Verified model choices"
              renderAction={(candidate, file) => {
                // Approval is one-shot: an accepted artefact cannot start a second multi-gigabyte transfer.
                const alreadyRequested = requestedDownloads.includes(`${candidate.id}/${file.name}`);
                const fits = Boolean(file.size_bytes) && file.fits_hardware === true;
                return {
                  detail: alreadyRequested
                    ? "Already approved in this session. Progress is in the setup activity above."
                    : file.fit_reason ?? (candidate.gated ? "Hugging Face access is checked through the credential broker." : "Compatibility data is stale. Run the model check again."),
                  button: (
                    <Button disabled={alreadyRequested || !fits} onClick={() => void onQueueModelDownload(job.id, candidate.id, file.name, file.size_bytes)} variant={!alreadyRequested && fits ? "primary" : "secondary"}>
                      {alreadyRequested ? "Download approved" : "Approve download"}<span className="sr-only">{` · ${file.name}`}</span>
                    </Button>
                  ),
                };
              }}
            />
          ) : null}
          {noMatch && <p className="apps-job__note">No exact verified GGUF file matched. Nothing was downloaded; refine the repository or file name and check again.</p>}
          {deployable && (
            <div className="apps-job__deploy">
              <Button variant="primary" onClick={() => void onQueueModelDeploy(job)}>Deploy local AI server</Button>
              <span>{job.result?.file} · {job.result?.size_bytes ? formatQuantity(job.result.size_bytes, "model") : "downloaded"}</span>
            </div>
          )}
          {deployed && (
            <AppsInset
              detail={job.id === currentDeployId ? "This is the Assistant's current private model." : modelForJob(job, managedModels) ? "This model is installed. Switch models from Manage when you want to use it." : "Open Manage to see the current installed-model state."}
              icon="shield"
              title={job.id === currentDeployId ? "Active in Assistant" : modelForJob(job, managedModels) ? "Local model available" : "Deployment completed"}
            />
          )}
          {/*
            * A deployment that succeeded can still have got less than it asked
            * for, silently: the CPU fallback answers `/health`, and a capped
            * context window starts normally. The deploy's own readings stay
            * with the deploy, beside the GPU AI-Chat tier's verdict when the
            * deploy went down the GPU fork.
            */}
          {(readings || gpuTier) && (
            <div className="apps-job__readings">
              {readings && (
                <AccelerationVerdict
                  acceleration={job.result?.acceleration}
                  context={job.result?.context_built}
                  /* Named per deploy (#150) so a screen-reader outline can tell one server's verdict from another's. */
                  title={`What the ${String(job.result?.path ?? "").split("/").pop()?.replace(/\.gguf$/i, "") || "model"} server got`}
                />
              )}
              {gpuTier && job.result && <GpuChatTier result={job.result} />}
            </div>
          )}
        </div>
      )}
    </li>
  );
}
