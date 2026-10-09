import type { ReactNode } from "react";
import { jobLabel } from "../lib/jobPresentation";
import { AppsDialog, AppsIconTile, AppsInset } from "./appsKit";
import { StatusPill } from "./StatusPill";
import { Button, Notice, Textarea } from "./ui";
import type { AgentPlan, AgentStatus } from "./workloads-types";
import "../styles/apps-install.css";

/*
 * The Vaelor deployment assistant (VD-200, the AppsPlanner board). The dialog's
 * own order is the sequence - describe, see the plan, review the exact action -
 * so it carries no separate ASK · CHECK · APPROVE rail. Nothing runs until the
 * shared action review is approved.
 */

const QUICK_PROMPTS = ["Install Grafana for dashboards", "Find a small private AI model", "Set up the local assistant"];

/**
 * Whether the planner's model is usable now. `configured` only says a model
 * was chosen; the route says separately whether it answered. Green needs both.
 */
export function plannerModelReady(status: AgentStatus | null): boolean {
  return Boolean(status?.configured && status.reachable === true);
}

export function WorkloadPlannerDialog({
  agentStatus,
  busy,
  canPlan,
  message,
  notice,
  noticeRefused,
  plan,
  setupOperation,
  onAsk,
  onBack,
  onClose,
  onMessageChange,
  onResearch,
  onReview,
}: {
  agentStatus: AgentStatus | null;
  busy: boolean;
  /** False for a viewer: they can read the dialog but not plan from it. */
  canPlan: boolean;
  message: string;
  notice: string;
  noticeRefused: boolean;
  plan: AgentPlan | null;
  /** The current setup operation, shown at the foot while one runs. */
  setupOperation: ReactNode;
  onAsk: () => void;
  onBack: () => void;
  onClose: () => void;
  onMessageChange: (value: string) => void;
  onResearch: () => void;
  onReview: () => void;
}) {
  const modelReady = plannerModelReady(agentStatus);
  const pill = modelReady
    ? <StatusPill label={agentStatus?.model || "Model enhanced"} tone="success" />
    : <StatusPill label="Code intelligence" tone="neutral" />;
  const footer = plan ? (
    <>
      <Button disabled={busy} onClick={onBack}>Go back</Button>
      {plan.application_intent && <Button disabled={busy} onClick={onResearch} variant="primary">Research and deploy</Button>}
      {plan.proposed_job && (
        <Button disabled={busy} onClick={onReview} variant="primary">
          {plan.proposed_job.type === "model.inspect" ? "Review compatibility check" : "Review exact action"}
        </Button>
      )}
    </>
  ) : (
    <Button
      busy={busy}
      disabledReason={!canPlan ? "Operator access is required to plan a setup." : !message.trim() ? "Describe what you want first." : undefined}
      onClick={onAsk}
      variant="primary"
    >
      {busy ? "Checking…" : "Show me the setup plan"}
    </Button>
  );
  return (
    <AppsDialog
      className="apps-planner"
      eyebrow="Setup assistant"
      footer={footer}
      headerActions={pill}
      onClose={onClose}
      size="wide"
      title="Vaelor deployment assistant"
      titleId="deployment-copilot-title"
    >
      {!plan && (
        <>
          <p>Describe what you want in everyday language. Vaelor can plan reviewed catalog apps, research other public apps, ask follow-up questions, and prepare an exact deployment plan for approval.</p>
          <div aria-label="Setup assistant capabilities" className="apps-planner__capabilities" role="group">
            <AppsInset detail="Vaelor checks reviewed blueprints first, then routes an unknown application into guarded public research without asking you to find another screen." title={<><small>Built-in planning</small>One request, matched automatically</>} />
            <AppsInset
              detail="A model improves ambiguous or complex requests, but it is not a blanket requirement for application setup."
              title={<><small>{modelReady ? "Model enhanced" : "Code intelligence"}</small>{modelReady ? "Better interpretation and follow-up questions" : "Reviewed apps and straightforward research are available"}</>}
            />
          </div>
          <div aria-label="Setup examples" className="apps-planner__prompts" role="group">
            {QUICK_PROMPTS.map((prompt) => (
              <Button disabled={busy || !canPlan} key={prompt} onClick={() => onMessageChange(prompt)}>{prompt}</Button>
            ))}
          </div>
        </>
      )}
      <Textarea
        className="apps-planner__request"
        disabled={busy || !canPlan}
        id="agent-request"
        label="What do you want to set up?"
        maxLength={4000}
        onChange={(event) => onMessageChange(event.target.value)}
        placeholder="Example: I want a private dashboard that shows the temperature of my home."
        rows={3}
        value={message}
      />
      {notice && <Notice severity={noticeRefused ? "danger" : "info"}>{notice}</Notice>}
      {plan && (plan.proposed_job ? (
        <div aria-live="polite" className="apps-planner__plan">
          <div className="apps-planner__plan-head">
            <AppsIconTile accent name="shield" />
            <div>
              <span className="apps-eyebrow">{plan.source.replaceAll("-", " ")}</span>
              <strong>{plan.summary}</strong>
              <p>{plan.rationale}</p>
            </div>
          </div>
          <div className="apps-planner__plan-grid">
            <AppsInset title="What Vaelor will check">
              <ol>{plan.checklist.map((item) => <li key={item}>{item}</li>)}</ol>
            </AppsInset>
            <AppsInset title="Before you continue">
              <p>Nothing changes until you continue to the reviewed deployment.</p>
              {plan.warnings.map((warning) => <p className="apps-planner__warning" key={warning}>{warning}</p>)}
              <details className="apps-disclosure">
                <summary>Advanced operation</summary>
                <p>{jobLabel(plan.proposed_job.type)} <small><code>{plan.proposed_job.type}</code></small></p>
              </details>
            </AppsInset>
          </div>
        </div>
      ) : (
        <div aria-live="polite">
          <AppsInset title={plan.summary}>
            <p>{plan.rationale}</p>
            {plan.checklist.length > 0 && <ol>{plan.checklist.map((item) => <li key={item}>{item}</li>)}</ol>}
            {plan.warnings.map((warning) => <p className="apps-planner__warning" key={warning}>{warning}</p>)}
            <p>{plan.application_intent ? "Nothing changes until you continue to the reviewed deployment." : "Give the assistant a little more information so it can prepare a safe plan."}</p>
          </AppsInset>
        </div>
      ))}
      {setupOperation}
    </AppsDialog>
  );
}
