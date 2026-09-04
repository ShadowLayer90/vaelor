import type { FormEvent, ReactNode } from "react";
import { timeAgo, timeUntil } from "../lib/format";
import type { AgentProfile, Automation, CapabilityDisclosure, Trigger } from "./agentTypes";
import { triggerLimits } from "./automationValidation";
import { ConfirmDialog } from "./ConfirmDialog";
import { Icon } from "./Icon";
import { StatusPill } from "./StatusPill";
import { Button, Input, Notice, Select, Textarea } from "./ui";

interface AutomationDelete {
  kind: "schedule" | "trigger";
  id: string;
  name: string;
}

interface AgentCenterAutomationsPanelProps {
  /** The alert-delivery channel configuration, composed in by the container. */
  alertChannelsPanel?: ReactNode;
  automationDelete: AutomationDelete | null;
  automationName: string;
  automationPrompt: string;
  automationSchedule: string;
  automationScheduleValid: boolean;
  automations: Automation[];
  busy: boolean;
  modelReady: boolean;
  notice: string;
  profile: string;
  profiles: AgentProfile[];
  triggerName: string;
  triggerSource: string;
  triggerThreshold: number;
  triggerThresholdValid: boolean;
  triggers: Trigger[];
  onCreateAutomation: (event: FormEvent) => void;
  onCreateTrigger: (event: FormEvent) => void;
  onDeleteAutomationItem: () => void;
  onSetAutomationDelete: (value: AutomationDelete | null) => void;
  onSetAutomationName: (value: string) => void;
  onSetAutomationPrompt: (value: string) => void;
  onSetAutomationSchedule: (value: string) => void;
  onSetProfile: (value: string) => void;
  onSetTriggerName: (value: string) => void;
  onSelectTriggerSource: (value: string) => void;
  onSetTriggerThreshold: (value: number) => void;
  onToggleAutomation: (automation: Automation) => void;
  onToggleTrigger: (trigger: Trigger) => void;
}

/**
 * What this rule's runs are allowed to do, on the rule.
 *
 * Nothing stops an unattended run for a second approval, so creating the rule
 * is the approval — and an approval you cannot read the terms of is not one.
 */
function UnattendedGrants({ disclosure }: { disclosure?: CapabilityDisclosure }) {
  if (!disclosure) return null;
  if (!disclosure.pinned_definition_available) {
    return (
      <p className="automation-grants automation-grants--unavailable">
        The pinned version of this agent is no longer readable, so what its runs
        may do cannot be shown. Delete this rule and create it again.
      </p>
    );
  }
  return (
    <dl className="automation-grants">
      <div><dt>Runs as</dt><dd>{disclosure.agent} version {disclosure.definition_version}</dd></div>
      <div><dt>Reads</dt><dd>{disclosure.reads.length ? disclosure.reads.join(" · ") : "Nothing beyond the task text"}</dd></div>
      <div><dt>Public research</dt><dd>{disclosure.web_access}</dd></div>
      <div><dt>Integrations</dt><dd>{disclosure.integrations.length ? disclosure.integrations.join(" · ") : "None"}</dd></div>
      <div><dt>Changes</dt><dd>{disclosure.writes}</dd></div>
    </dl>
  );
}

export function AgentCenterAutomationsPanel({
  alertChannelsPanel,
  automationDelete,
  automationName,
  automationPrompt,
  automationSchedule,
  automationScheduleValid,
  automations,
  busy,
  modelReady,
  notice,
  profile,
  profiles,
  triggerName,
  triggerSource,
  triggerThreshold,
  triggerThresholdValid,
  triggers,
  onCreateAutomation,
  onCreateTrigger,
  onDeleteAutomationItem,
  onSetAutomationDelete,
  onSetAutomationName,
  onSetAutomationPrompt,
  onSetAutomationSchedule,
  onSetProfile,
  onSetTriggerName,
  onSelectTriggerSource,
  onSetTriggerThreshold,
  onToggleAutomation,
  onToggleTrigger,
}: AgentCenterAutomationsPanelProps) {
  return (
    /*
     * Automations live with the agents they run, not on a tab of their own. A
     * schedule is a property of the thing being scheduled; separating them made
     * "why did this run?" a question you had to answer on another screen.
     */
    <section aria-labelledby="agent-automations-title" className="automation-center" id="agent-automations">
      <div className="section-heading"><div><span className="page-eyebrow">Unattended work</span><h2 id="agent-automations-title">Run this without being asked</h2><p>Schedule an appliance check or one of your agents. Creating the schedule is the approval: its runs start on their own, without stopping for another one. Each run reads only, stays in its own workspace, cannot create more schedules, and turns anything that would change something into a proposal that needs its own approval.</p></div><StatusPill status={modelReady ? "healthy" : "degraded"} label={modelReady ? "Read-only runs" : "Model required"} /></div>
      {!modelReady && <Notice severity="warning"><Icon name="cpu" />Connect or select a model before scheduling appliance checks or agent work.</Notice>}
      <form className="skill-proposal" onSubmit={onCreateAutomation}>
        <Input id="automation-name" label="Schedule name" maxLength={100} onChange={(event) => onSetAutomationName(event.target.value)} value={automationName} />
<Select id="automation-profile" label="Appliance check or agent" onChange={(event) => onSetProfile(event.target.value)} value={profile}>{profiles.map((item) => <option disabled={!item.operational} key={item.id} value={item.id}>{item.name}{item.operational ? "" : " unavailable"}</option>)}</Select>
<Textarea className="skill-proposal__content" id="automation-prompt" label="What should it check?" maxLength={4000} onChange={(event) => onSetAutomationPrompt(event.target.value)} rows={3} value={automationPrompt} />
        <div className="automation-field">
          <Input
            aria-describedby="schedule-help"
            id="automation-schedule"
            label="When"
            maxLength={120}
            onChange={(event) => onSetAutomationSchedule(event.target.value)}
            value={automationSchedule}
          />
          <small id="schedule-help">Schedule expressions are checked before they are saved.</small>
        </div>
        {!automationScheduleValid && <small role="alert">Use a future date, "in 30 minutes", or "every 6 hours" (5 minutes to 30 days).</small>}
        <Button disabled={!modelReady || busy || !automationName.trim() || !automationPrompt.trim() || !automationScheduleValid} type="submit" variant="primary">Create schedule</Button>
      </form>
      {notice && <Notice severity="info">{notice}</Notice>}
      <div className="skill-list">
        {automations.map((automation) => <article className="skill-card" key={automation.id}><div className="agent-task-card__header"><div><small>{automation.profile} · {automation.kind}</small><h2>{automation.name}</h2></div><StatusPill status={automation.enabled ? "healthy" : "neutral"} label={automation.enabled ? "enabled" : "paused"} /></div><p>{automation.prompt}</p><small>{automation.schedule_text}{automation.next_run_at ? ` · next ${timeUntil(automation.next_run_at * 1000)}` : ""}</small><UnattendedGrants disclosure={automation.capability_disclosure} /><div className="agent-task-card__actions"><Button aria-pressed={automation.enabled} disabled={busy} onClick={() => onToggleAutomation(automation)} type="button" variant="quiet">{automation.enabled ? "Pause" : "Enable"}</Button><Button className="danger-text" disabled={busy} onClick={() => onSetAutomationDelete({ kind: "schedule", id: automation.id, name: automation.name })} type="button" variant="danger">Delete</Button></div></article>)}
        {automations.length === 0 && <div className="empty-state"><Icon name="activity" /><h3>No schedules yet</h3><p>Create a safe health check or recurring review above.</p></div>}
      </div>
      <div className="automation-divider"><span className="page-eyebrow">Event-driven help</span><h2>Alert rules</h2><p>Launch a read-only diagnostic when a live hardware signal crosses a safe threshold. Enabling the rule is the approval for every diagnostic it launches; none of them stops for another one. A 30-minute cooldown prevents alert storms.</p></div>
      <form className="skill-proposal trigger-form" onSubmit={onCreateTrigger}>
        <Input id="trigger-name" label="Alert name" maxLength={100} onChange={(event) => onSetTriggerName(event.target.value)} placeholder="Example: CPU running hot" value={triggerName} />
        <Select id="trigger-source" label="Watch signal" onChange={(event) => onSelectTriggerSource(event.target.value)} value={triggerSource}>
          <option value="cpu_temperature">CPU temperature</option>
          <option value="memory_percent">Memory use</option>
          <option value="storage_percent">Storage use</option>
          <option value="service_failures">Failed Vaelor services</option>
          <option value="fan_failure">Fan failure signal</option>
        </Select>
        <Input
          aria-invalid={!triggerThresholdValid}
          id="trigger-threshold"
          label="Alert at or above"
          max={triggerLimits[triggerSource]?.[1]}
          min={triggerLimits[triggerSource]?.[0]}
          onChange={(event) => onSetTriggerThreshold(Number(event.target.value))}
          type="number"
          value={triggerThreshold}
        />
        <Button disabled={busy || !triggerName.trim() || !triggerThresholdValid} type="submit" variant="primary">Enable alert rule</Button>
      </form>
      <div className="skill-list">
        {triggers.map((trigger) => <article className="skill-card" key={trigger.id}><div className="agent-task-card__header"><div><small>{trigger.source.replaceAll("_", " ")} · {trigger.operator} {trigger.threshold}</small><h2>{trigger.name}</h2></div><StatusPill status={trigger.enabled ? "healthy" : "neutral"} label={trigger.enabled ? "watching" : "paused"} /></div><p>{trigger.prompt}</p><small>{trigger.last_value == null ? "Waiting for first sample" : `Latest value ${trigger.last_value}`}{trigger.last_triggered_at ? ` · triggered ${timeAgo(trigger.last_triggered_at * 1000)}` : ""}</small><UnattendedGrants disclosure={trigger.capability_disclosure} /><div className="agent-task-card__actions"><Button disabled={busy} onClick={() => onToggleTrigger(trigger)} type="button" variant="quiet">{trigger.enabled ? "Pause" : "Enable"}</Button><Button className="danger-text" disabled={busy} onClick={() => onSetAutomationDelete({ kind: "trigger", id: trigger.id, name: trigger.name })} type="button" variant="danger">Delete</Button></div></article>)}
        {triggers.length === 0 && <div className="empty-state"><Icon name="alert" /><h3>No alert rules yet</h3><p>Add a temperature, storage, memory, service, or fan rule above.</p></div>}
      </div>
      {alertChannelsPanel}
      <ConfirmDialog
        busy={busy}
        confirmLabel={automationDelete?.kind === "schedule" ? "Delete schedule" : "Delete alert"}
        description={automationDelete ? `Delete "${automationDelete.name}"? Its previous run records will also be removed.` : ""}
        onCancel={() => onSetAutomationDelete(null)}
        onConfirm={onDeleteAutomationItem}
        open={Boolean(automationDelete)}
        title={automationDelete?.kind === "schedule" ? "Delete schedule?" : "Delete alert rule?"}
      />
    </section>
  );
}
