import { useCallback, useEffect, useState } from "react";
import { apiRequest } from "../lib/api";
import { canonicalOperationState, jobCanCancel, jobIsReady, jobIsRetryable, jobIsTerminal, jobNeedsAttention, jobStateLabel } from "../lib/jobPresentation";
import type { Session } from "../types";
import { ActionReviewDialog } from "./ActionReviewDialog";
import { Icon } from "./Icon";
import { ModalShell } from "./ModalShell";
import { Button, Notice } from "./ui";

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
    setError("");
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
      setError(reason instanceof Error ? reason.message : "The reviewed operation could not be queued.");
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

  return (
    <>
          <p>{status?.ready ? "Guarded web research is ready on this node." : status?.reason ?? "Checking guarded web research readiness…"}</p>
      <div className="panel-heading__actions">
        {status?.ready ? (
          <>
            <Button onClick={onResearch} variant="primary">Research a public application</Button>
            {canManage && <Button onClick={() => setOpen(true)} variant="quiet">Manage research</Button>}
          </>
        ) : (
          <Button
            disabled={!canManage || busy || !status || operationInProgress}
            onClick={() => status?.state === "blocked" ? setOpen(true) : void prepare(primaryAction)}
            variant="primary"
          >
            {operationInProgress ? "Setup in progress" : status?.state === "degraded" ? "Review repair" : status?.state === "blocked" ? "Port conflict needs attention" : "Set up web research"}
          </Button>
        )}
      </div>
      {error && !open && <p className="agent-plan__warning" role="alert">{error}</p>}

      {open && (
        <ModalShell labelledBy="web-research-title" onClose={() => setOpen(false)}>
          <section className="custom-compose" aria-labelledby="web-research-title">
            <div className="panel-heading">
              <div>
                <span className="page-eyebrow">Private research capability</span>
                <h2 id="web-research-title">Guarded web research</h2>
                <p>Vaelor uses this private service to find public evidence. Models never receive direct network, shell, Docker, or credential access.</p>
              </div>
              <Button onClick={() => setOpen(false)} variant="quiet">Close</Button>
            </div>
            {status && (
              <div className="tool-explainer">
                <Icon name={status.ready ? "shield" : "activity"} />
                <span>
                  <strong>{operationInProgress ? "Setup in progress" : status.ready ? "Ready for application research" : status.state.replaceAll("_", " ")}</strong>
                  <small>{operationInProgress ? "Vaelor is starting and verifying the private service. The status will update when verification finishes." : status.reason}</small>
                </span>
              </div>
            )}
            {job && (
              <div className="agent-plan agent-plan--operation" aria-live="polite">
                <div className="agent-plan__summary">
                  <span><Icon name="activity" /></span>
                  <div><small>Current research-service operation</small><strong>{recoveredAfterTimeout ? "Ready after delayed startup" : jobStateLabel(job)}</strong><p>{recoveredAfterTimeout ? "The service became ready after the original health check. No retry is needed." : job.message || "Vaelor is preparing the guarded research service."}</p></div>
                  <span className={`event-state${jobNeedsAttention(job) && !recoveredAfterTimeout ? " event-state--failure" : jobIsReady(job) || recoveredAfterTimeout ? " event-state--success" : jobIsTerminal(job) ? "" : " event-state--progress"}`}>{recoveredAfterTimeout ? "READY" : `${job.progress}%`}</span>
                </div>
                <div className="agent-plan__actions">
                  {jobCanCancel(job) && <Button disabled={busy} onClick={() => void jobAction("cancel")}>Cancel</Button>}
                  {jobIsRetryable(job) && !recoveredAfterTimeout && <Button disabled={busy} onClick={() => void jobAction("retry")} variant="primary">Retry safely</Button>}
                  {jobIsTerminal(job) && <Button onClick={() => { setJob(null); window.localStorage.removeItem(resumeKey); void refreshStatus(); }} variant="primary">Done</Button>}
                </div>
              </div>
            )}
            {!job && status?.ready && canManage && (
              <div className="agent-plan__actions">
                <Button disabled={busy} onClick={() => void prepare("repair")}>Review repair</Button>
                <Button disabled={busy} onClick={() => void prepare("remove")} variant="danger">Review removal</Button>
              </div>
            )}
            {!job && !status?.ready && status?.state !== "blocked" && canManage && (
              <Button disabled={busy || !status} onClick={() => void prepare(primaryAction)} variant="primary">
                {status?.state === "degraded" ? "Review repair" : "Review setup"}
              </Button>
            )}
            {status?.state === "blocked" && <Notice severity="warning"><Icon name="activity" />Port 8888 belongs to another process. Vaelor changed nothing. Stop or reconfigure that service, then refresh this check.</Notice>}
            {error && <Notice severity="danger"><Icon name="activity" />{error}</Notice>}
          </section>
        </ModalShell>
      )}

      <ActionReviewDialog
        busy={busy}
        evidence={plan ? [
          { source: "pinned image", summary: plan.image },
          { source: "private endpoint", summary: `${plan.endpoint} (${status?.network_scope ?? "loopback-only"})` },
        ] : []}
        job={plan ? { type: "host.web-research.manage", payload: { action: plan.action, endpoint: plan.endpoint } } : null}
        onApprove={() => void executePlan()}
        onCancel={() => setPlan(null)}
        summary={plan ? `${plan.changes.join(" ")} Recovery: ${plan.recovery}` : ""}
        suggestedActions={["Keep progress in this window or close it and resume here later.", "Vaelor verifies the private endpoint before marking research ready."]}
      />
    </>
  );
}
