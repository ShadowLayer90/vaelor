import { useCallback, useEffect, useState } from "react";
import { apiRequest } from "../lib/api";
import { canonicalOperationState, jobCanCancel, jobIsReady, jobIsRetryable, jobIsTerminal, jobNeedsAttention, jobStateLabel } from "../lib/jobPresentation";
import type { Session } from "../types";
import "../styles/apps-setup.css";
import { ActionReviewDialog } from "./ActionReviewDialog";
import { AppsDialog, AppsIconTile, AppsProgress } from "./appsKit";
import { AppsBanner } from "./appsSetupParts";
import { StatusPill } from "./StatusPill";
import { Button } from "./ui";
import { joinClassNames } from "./ui/field";

type ResearchState = "ready" | "not_installed" | "degraded" | "blocked";

interface ResearchStatus {
  state: ResearchState;
  reason: string;
  installed: boolean;
  ready: boolean;
  managed: boolean;
  digest_pinned: boolean;
  endpoint: string;
  network_scope: string;
  image: string;
  actions: Array<"install" | "repair" | "remove">;
}

interface ResearchPlan {
  action: "install" | "repair" | "remove";
  confirmation: string;
  title: string;
  changes: string[];
  image: string;
  endpoint: string;
  data_path: string;
  approval_required: boolean;
  recovery: string;
}

interface ResearchJob {
  id: string;
  type: string;
  state: string;
  operation_state?: string;
  attention?: boolean;
  retryable?: boolean;
  readiness?: string;
  liveness?: string;
  progress: number;
  message: string;
}



export function WebResearchSetup({
  session,
  onResearch,
}: {
  session: Session;
  onResearch: () => void;
}) {
  const resumeKey = `vaelor.web-research.operation.${session.user.username}`;
  const [status, setStatus] = useState<ResearchStatus | null>(null);
  const [plan, setPlan] = useState<ResearchPlan | null>(null);
  const [job, setJob] = useState<ResearchJob | null>(null);
  const [open, setOpen] = useState(false);
  const [busy, setBusy] = useState(false);
  const [error, setError] = useState("");
  const [reviewError, setReviewError] = useState(""); // VD-189: the review's refusal, shown in the review

  const refreshStatus = useCallback(async () => {
    try {
      setStatus(await apiRequest<ResearchStatus>("/applications/research-service"));
      setError("");
    } catch (reason) {
      setError(reason instanceof Error ? reason.message : "Research readiness could not be checked.");
    }
  }, []);

  useEffect(() => {
    void refreshStatus();
    const savedJob = window.localStorage.getItem(resumeKey);
    if (!savedJob) return;
    setOpen(true);
    void apiRequest<ResearchJob>(`/jobs/${savedJob}`)
      .then(setJob)
      .catch(() => window.localStorage.removeItem(resumeKey));
  }, [refreshStatus, resumeKey]);

  useEffect(() => {
    if (!job || jobIsTerminal(job)) return;
    const timer = window.setInterval(() => {
      void apiRequest<ResearchJob>(`/jobs/${job.id}`)
        .then((next) => {
          setJob(next);
          if (jobIsTerminal(next)) void refreshStatus();
        })
        .catch(() => setError("Progress could not be refreshed. The operation is still saved and can be reloaded."));
    }, 2000);
    return () => window.clearInterval(timer);
  }, [job, refreshStatus]);

  async function prepare(action: ResearchPlan["action"]) {
    setBusy(true);
    setError("");
    try {
      setPlan(await apiRequest<ResearchPlan>(
        "/applications/research-service/plan",
        { method: "POST", body: JSON.stringify({ action }) },
        session.csrf_token,
      ));
      setOpen(true);
    } catch (reason) {
      setError(reason instanceof Error ? reason.message : "The research setup plan could not be prepared.");
      setOpen(true);
    } finally {
      setBusy(false);
    }
  }

  async function executePlan() {
    if (!plan) return;
    setBusy(true);
    setError(""); setReviewError("");
    try {
      const result = await apiRequest<{ plan: ResearchPlan; job: ResearchJob }>(
        "/applications/research-service/actions",
        {
          method: "POST",
          body: JSON.stringify({ action: plan.action, confirmation: plan.confirmation, purge: false }),
        },
        session.csrf_token,
      );
      setPlan(null);
      setJob(result.job);
      window.localStorage.setItem(resumeKey, result.job.id);
    } catch (reason) {
      setReviewError(reason instanceof Error ? reason.message : "The reviewed operation could not be queued.");
    } finally {
      setBusy(false);
    }
  }

  async function jobAction(action: "cancel" | "retry") {
    if (!job) return;
    setBusy(true);
    setError("");
    try {
      const next = await apiRequest<ResearchJob>(
        `/jobs/${job.id}/${action}`,
        { method: "POST", body: "{}" },
        session.csrf_token,
      );
      setJob(next);
      window.localStorage.setItem(resumeKey, next.id);
    } catch (reason) {
      setError(reason instanceof Error ? reason.message : `The operation could not ${action}.`);
    } finally {
      setBusy(false);
    }
  }

  const primaryAction = status?.state === "not_installed" ? "install" : "repair";
  const canManage = session.user.role === "administrator";
  const operationInProgress = Boolean(job && jobCanCancel(job));
  const recoveredAfterTimeout = Boolean(job && canonicalOperationState(job) === "failed" && status?.ready);
  const jobFailed = Boolean(job && jobNeedsAttention(job) && !recoveredAfterTimeout);
  const jobSettled = Boolean(job && (jobFailed || recoveredAfterTimeout || jobIsReady(job)));
  // The service's state in words: the server's own state name, sentence-cased.
  const stateTitle = operationInProgress
    ? "Setup in progress"
    : status?.ready
      ? "Ready for application research"
      : (status?.state ?? "").replaceAll("_", " ").replace(/^./, (letter) => letter.toUpperCase());
  const blockReason = !canManage
    ? "An administrator sets this up."
    : !status
      ? "Readiness is still being checked."
      : undefined;

  return (
    <>
      {/* The AppsWebResearch board: one row where research needs it - the
          capability, its state in a sentence, and the one next step. */}
      <div className="apps-research-row">
        <AppsIconTile accent={Boolean(status?.ready)} name="search" />
        <div className="apps-research-row__text">
          <strong>Guarded web research</strong>
          <span>{status?.ready ? "Guarded web research is ready on this node." : status?.reason ?? "Checking guarded web research readiness…"}</span>
        </div>
        <div className="apps-research-row__actions">
          {status?.ready ? (
            <>
              <Button onClick={onResearch} variant="primary">Research a public application</Button>
              {canManage && <Button onClick={() => setOpen(true)} variant="quiet">Manage research</Button>}
            </>
          ) : operationInProgress ? (
            <Button onClick={() => setOpen(true)}>Setup in progress</Button>
          ) : (
            <Button
              disabled={busy}
              disabledReason={blockReason}
              onClick={() => status?.state === "blocked" ? setOpen(true) : void prepare(primaryAction)}
              variant="primary"
            >
              {status?.state === "degraded" ? "Review repair" : status?.state === "blocked" ? "Port conflict needs attention" : "Set up web research"}
            </Button>
          )}
        </div>
      </div>
      {error && !open && <AppsBanner tone="danger">{error}</AppsBanner>}

      {open && (
        <AppsDialog
          eyebrow="Private research capability"
          footer={<Button onClick={() => setOpen(false)}>Close</Button>}
          onClose={() => setOpen(false)}
          title="Guarded web research"
          titleId="web-research-title"
        >
          <p>Vaelor uses this private service to find public evidence. Models never receive direct network, shell, Docker, or credential access.</p>
          {status && (
            <div className="apps-research-status">
              <AppsIconTile name={status.ready ? "shield" : "activity"} />
              <div className="apps-research-row__text">
                <strong>{stateTitle}</strong>
                <span>{operationInProgress ? "Vaelor is starting and verifying the private service. The status will update when verification finishes." : status.reason}</span>
              </div>
            </div>
          )}
          {job && (
            <section
              aria-live="polite"
              className={joinClassNames("apps-research-job", jobFailed && "apps-research-job--failed")}
            >
              {jobSettled ? (
                <div className="apps-research-job__head">
                  <strong>{recoveredAfterTimeout ? "Ready after delayed startup" : jobStateLabel(job)}</strong>
                  <StatusPill label={jobFailed ? "Failed" : "Ready"} tone={jobFailed ? "danger" : "success"} />
                </div>
              ) : (
                <div className="apps-research-job__head">
                  <div className="apps-research-job__title">
                    <AppsIconTile name="activity" />
                    <div className="apps-research-row__text">
                      <small>Current research-service operation</small>
                      <strong>{jobStateLabel(job)}</strong>
                      <span>{job.message || "Vaelor is preparing the guarded research service."}</span>
                    </div>
                  </div>
                  <span className="apps-research-job__percent">{job.progress}%</span>
                </div>
              )}
              {jobSettled && (
                <p className={jobFailed ? "apps-research-job__error" : "apps-fineprint"}>
                  {recoveredAfterTimeout ? "The service became ready after the original health check. No retry is needed." : job.message || "Vaelor is preparing the guarded research service."}
                </p>
              )}
              {!jobIsTerminal(job) && <AppsProgress fraction={job.progress / 100} label="Research-service operation progress" />}
              <div className="apps-research-job__actions">
                {jobCanCancel(job) && <Button disabled={busy} onClick={() => void jobAction("cancel")}>Cancel</Button>}
                {jobIsRetryable(job) && !recoveredAfterTimeout && <Button disabled={busy} onClick={() => void jobAction("retry")} variant="primary">Retry safely</Button>}
                {jobIsTerminal(job) && (
                  <Button
                    onClick={() => { setJob(null); window.localStorage.removeItem(resumeKey); void refreshStatus(); }}
                    variant={jobIsRetryable(job) && !recoveredAfterTimeout ? "secondary" : "primary"}
                  >
                    Done
                  </Button>
                )}
              </div>
            </section>
          )}
          {!job && status?.ready && canManage && (
            <div className="apps-research-job__actions">
              <Button disabled={busy} onClick={() => void prepare("repair")}>Review repair</Button>
              <Button disabled={busy} onClick={() => void prepare("remove")} variant="danger">Review removal</Button>
            </div>
          )}
          {!job && !status?.ready && status?.state !== "blocked" && canManage && (
            <div className="apps-research-job__actions">
              <Button disabled={busy || !status} onClick={() => void prepare(primaryAction)} variant="primary">
                {status?.state === "degraded" ? "Review repair" : "Review setup"}
              </Button>
            </div>
          )}
          {status?.state === "blocked" && (
            <AppsBanner tone="warning">Port 8888 belongs to another process. Vaelor changed nothing. Stop or reconfigure that service, then refresh this check.</AppsBanner>
          )}
          {error && <AppsBanner tone="danger">{error}</AppsBanner>}
        </AppsDialog>
      )}

      <ActionReviewDialog
        busy={busy}
        error={reviewError}
        evidence={plan ? [
          { source: "web-research.image", summary: plan.image },
          { source: "web-research.endpoint", summary: `${plan.endpoint} (${status?.network_scope ?? "loopback-only"})` },
        ] : []}
        job={plan ? { type: "host.web-research.manage", payload: { action: plan.action, endpoint: plan.endpoint } } : null}
        onApprove={() => void executePlan()}
        onCancel={() => { setReviewError(""); setPlan(null); }}
        summary={plan ? `${plan.changes.join(" ")} Recovery: ${plan.recovery}` : ""}
        suggestedActions={["Keep progress in this window or close it and resume here later.", "Vaelor verifies the private endpoint before marking research ready."]}
      />
    </>
  );
}
