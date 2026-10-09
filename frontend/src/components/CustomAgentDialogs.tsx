import type { FormEvent } from "react";
import { AgentRunWorkspace } from "./AgentRunWorkspace";
import { Icon } from "./Icon";
import { ModalShell } from "./ModalShell";
import type { AgentProfile, AgentTask } from "./agentTypes";
import { Button, Input, Select, SegmentedControl, Textarea } from "./ui";

/*
 * The agent workshop's smaller dialogs, drawn to the AssistRunDialogs board:
 * Test & activate, the approval-gated run (which turns into the run's own
 * surface once it starts), and Add automation. Their state and their API calls
 * stay in CustomAgentManager; these only draw them. Each takes its own refusal
 * as `error`, shown inside it - the page under an open dialog is inert (VD-189).
 */

function DialogHead({ eyebrow, id, title, text, closeLabel, onClose, busy = false }: {
  eyebrow: string;
  id: string;
  title: string;
  text?: string;
  closeLabel: string;
  onClose: () => void;
  busy?: boolean;
}) {
  return (
    <div className="as-dialog__head">
      <div>
        <span className="as-label">{eyebrow}</span>
        <h2 id={id}>{title}</h2>
        {text && <p>{text}</p>}
      </div>
      <Button aria-label={closeLabel} className="as-btn-ghost" disabled={busy} onClick={onClose} variant="quiet">Close</Button>
    </div>
  );
}

/** A separate dialog after the form has been saved, so it carries no step number from the form's own count. */
export function ActivationDialog({ agent, busy, canTest, error, onActivate, onClose, onTest }: {
  agent: AgentProfile;
  busy: boolean;
  /** Test before activation is offered only where the agent runs in the Assistant. */
  canTest: boolean | null;
  error: string;
  onActivate: () => void;
  onClose: () => void;
  onTest: () => void;
}) {
  return (
    <ModalShell className="as-dialog ar-dialog-sm" error={error} labelledBy="custom-agent-activation-title" onClose={() => !busy && onClose()}>
      <DialogHead busy={busy} closeLabel="Close activation" eyebrow="Test & activate" id="custom-agent-activation-title" onClose={onClose}
        text={`The definition is saved at version ${agent.version}. Prepare a reviewable run, then activate this version for future work.`}
        title={`${agent.name} is ready for a safe test`} />
      <div className="as-dialog__body">
        <ol className="ar-checklist">
          <li><span aria-hidden="true" className="ar-checklist__done"><Icon name="done" size={16} /></span><span><strong>Intent saved</strong><span className="ar-step__hint">{agent.description}</span></span></li>
          <li><span aria-hidden="true" className="ar-checklist__done"><Icon name="done" size={16} /></span><span><strong>Access pinned</strong><span className="ar-step__hint as-mono">{agent.scopes.length ? agent.scopes.join(" · ") : "No appliance access"}</span></span></li>
          <li><span aria-hidden="true" className="ar-checklist__open"><Icon name="shield" size={16} /></span><span><strong>Approval required</strong><span className="ar-step__hint">Tests and state-changing connector calls wait for operator review.</span></span></li>
        </ol>
      </div>
      <div className="as-dialog__foot">
        {canTest !== null && <Button disabled={busy || !canTest} onClick={onTest}>Test before activation</Button>}
        <Button busy={busy} disabled={busy} onClick={onActivate} variant="primary">Activate agent</Button>
      </div>
    </ModalShell>
  );
}

/**
 * Run, from a row's Run or from Test before activation. Once the run is
 * created this dialog becomes the run's own surface and follows it through
 * approval, execution and result, so nobody goes hunting for the outcome.
 */
export function RunDialog({ activeRun, agent, busy, error, onApprove, onCancelRun, onClose, onEscalate, onRequestChange, onRetry, onRunAgain, onSubmit, request, requestError }: {
  activeRun: AgentTask | null;
  agent: AgentProfile;
  busy: boolean;
  error: string;
  onApprove: () => void;
  onCancelRun: () => void;
  onClose: () => void;
  onEscalate: () => void;
  onRequestChange: (value: string) => void;
  onRetry: () => void;
  onRunAgain: () => void;
  onSubmit: (event: FormEvent) => void;
  request: string;
  requestError: string;
}) {
  return (
    <ModalShell className="as-dialog ar-dialog-sm" error={error} labelledBy="custom-agent-test-heading" onClose={onClose}>
      {activeRun ? (
        <div className="ar-run-dialog">
          <DialogHead closeLabel="Close agent run" eyebrow="Agent run" id="custom-agent-test-heading" onClose={onClose} title={agent.name} />
          <AgentRunWorkspace
            busy={busy}
            onApprove={onApprove}
            onCancel={onCancelRun}
            onClose={onClose}
            onEscalate={onEscalate}
            onRetry={onRetry}
            onRunAgain={onRunAgain}
            task={activeRun}
          />
        </div>
      ) : (
        <form className="ar-dialog-form" onSubmit={onSubmit}>
          <DialogHead closeLabel="Close test run" eyebrow="Approval-gated run" id="custom-agent-test-heading" onClose={onClose}
            text="Describe one realistic task. Vaelor prepares a reviewable run and shows the result here; nothing executes until you approve it."
            title={`Run ${agent.name}`} />
          <div className="as-dialog__body">
            <Textarea autoFocus error={requestError || undefined} label="What should it do?" maxLength={4000} onChange={(event) => onRequestChange(event.target.value)} placeholder="Example: Give me yesterday's score for the Yankees and tell me whether they won." rows={4} value={request} />
          </div>
          <div className="as-dialog__foot">
            <Button onClick={onClose}>Cancel</Button>
            <Button busy={busy} disabled={busy} type="submit" variant="primary">Run now</Button>
          </div>
        </form>
      )}
    </ModalShell>
  );
}

export type AutomationForm = {
  kind: "schedule" | "trigger";
  name: string;
  prompt: string;
  schedule: string;
  source: string;
  threshold: string;
};

/** Add automation: a time schedule or a Vaelor hardware signal, pinned to this agent version. */
export function AutomationDialog({ agent, busy, error, form, onChange, onClose, onSubmit }: {
  agent: AgentProfile;
  busy: boolean;
  error: string;
  form: AutomationForm;
  onChange: (patch: Partial<AutomationForm>) => void;
  onClose: () => void;
  onSubmit: (event: FormEvent) => void;
}) {
  const incomplete = !form.name.trim() || !form.prompt.trim() || (form.kind === "schedule" ? !form.schedule.trim() : !form.threshold);
  return (
    <ModalShell className="as-dialog ar-dialog-sm" error={error} labelledBy="custom-agent-automation-heading" onClose={onClose}>
      <form className="ar-dialog-form" onSubmit={onSubmit}>
        <DialogHead closeLabel="Close automation" eyebrow="Version-pinned automation" id="custom-agent-automation-heading" onClose={onClose}
          text="Choose a time schedule or a Vaelor hardware signal. Future edits create a new agent version and do not silently change this automation."
          title={`Automate ${agent.name}`} />
        <div className="as-dialog__body">
          <div className="ar-field">
            <span className="ar-field__label">Automation type</span>
            <SegmentedControl label="Automation type" onChange={(kind) => onChange({ kind })} options={[{ value: "schedule", label: "Time schedule" }, { value: "trigger", label: "Vaelor hardware trigger" }]} value={form.kind} />
          </div>
          <Input label="Name" maxLength={100} onChange={(event) => onChange({ name: event.target.value })} value={form.name} />
          <Textarea label="Task instructions" maxLength={4000} onChange={(event) => onChange({ prompt: event.target.value })} rows={3} value={form.prompt} />
          {form.kind === "schedule" ? (
            <Input hint="Examples: “in 30 minutes”, “every 6 hours”, or an ISO date and time." label="When" maxLength={120} onChange={(event) => onChange({ schedule: event.target.value })} value={form.schedule} />
          ) : (
            <div className="as-box ar-trigger-box">
              <span className="as-label">As a hardware trigger</span>
              <div className="as-grid2">
                <Select label="Signal" onChange={(event) => onChange({ source: event.target.value })} value={form.source}>
                  <option value="cpu_temperature">CPU temperature</option>
                  <option value="memory_percent">Memory use</option>
                  <option value="storage_percent">Storage use</option>
                  <option value="service_failures">Failed Vaelor services</option>
                  <option value="fan_failure">Fan failure signal</option>
                </Select>
                <Input label="Run when at or above" min="1" onChange={(event) => onChange({ threshold: event.target.value })} type="number" value={form.threshold} />
              </div>
            </div>
          )}
        </div>
        <div className="as-dialog__foot">
          <Button onClick={onClose}>Cancel</Button>
          <Button disabled={busy || incomplete} type="submit" variant="primary">Create automation</Button>
        </div>
      </form>
    </ModalShell>
  );
}
