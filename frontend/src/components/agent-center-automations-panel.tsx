import { type FormEvent, type ReactNode, useEffect, useState } from "react";
import {
  automationPill,
  operatorLabel,
  scheduleIsSpent,
  scheduleKindLabel,
  showsNextRun,
  signalLabel,
  type AutomationItemStatus,
} from "../lib/automationStatus";
import { timeAgo, timeUntil } from "../lib/format";
import type { AgentProfile, Automation, CapabilityDisclosure, Trigger } from "./agentTypes";
import { AssistantBarActions } from "./assistantBar";
import { isWorkerCapableSource, triggerLimits } from "./automationValidation";
import { ConfirmDialog } from "./ConfirmDialog";
import { Icon, ICON_SIZE } from "./Icon";
import { ModalShell } from "./ModalShell";
import { StatusPill } from "./StatusPill";
import { Button, EmptyState, Input, Notice, Select, Textarea } from "./ui";

interface AutomationDelete {
  kind: "schedule" | "trigger";
  id: string;
  name: string;
  /** Why the delete was refused (VD-189), shown inside the confirmation. */
  error?: string;
}

/** A machine an alert rule can be aimed at (an enrolled worker). */
interface AlertTarget {
  id: string;
  name: string;
}

/** The Schedules and alerts view's heading and its sentence: the terms of an unattended run. */
export const AUTOMATIONS_TITLE = "Run this without being asked";
export const AUTOMATIONS_LEAD = "Schedule an appliance check or one of your agents. Creating the schedule is the approval: its runs start on their own, without stopping for another one. Each run reads only, stays in its own workspace, cannot create more schedules, and turns anything that would change something into a proposal that needs its own approval.";

/** A create handler may say whether it succeeded; a dialog closes on `true`. */
type CreateHandler = (event: FormEvent) => void | Promise<unknown>;

interface AgentCenterAutomationsPanelProps {
  /** The alert-delivery channel configuration, composed in by the container. */
  alertChannelsPanel?: ReactNode;
  /**
   * The workers a rule may target. The Assistant passes none today, so the
   * form stays controller-only and every signal stays available; Cluster >
   * Activity draws its own rule form (ClusterAlerts). Unset also means the
   * fleet was not read, so a worker rule is named "A worker", never "no longer
   * in the fleet" (VD-200 assist review).
   */
  alertTargets?: AlertTarget[];
  automationDelete: AutomationDelete | null;
  automationName: string;
  automationPrompt: string;
  automationSchedule: string;
  automationScheduleValid: boolean;
  automations: Automation[];
  busy: boolean;
  /** A model is configured at all; false says "not set up" rather than "not answering". */
  modelConfigured?: boolean;
  /** The configured model is answering (the shared readiness projection). */
  modelReady: boolean;
  notice: string;
  /** The notice is a refusal: shown as an alert, not a quiet status (VD-189). */
  noticeRefused?: boolean;
  profile: string;
  profiles: AgentProfile[];
  triggerName: string;
  /** The selected target: "" is the controller, else a worker node id. */
  triggerNode?: string;
  triggerSource: string;
  triggerThreshold: number;
  triggerThresholdValid: boolean;
  triggers: Trigger[];
  onCreateAutomation: CreateHandler;
  onCreateTrigger: CreateHandler;
  onDeleteAutomationItem: () => void;
  onSetAutomationDelete: (value: AutomationDelete | null) => void;
  onSetAutomationName: (value: string) => void;
  onSetAutomationPrompt: (value: string) => void;
  onSetAutomationSchedule: (value: string) => void;
  onSetProfile: (value: string) => void;
  onSetTriggerName: (value: string) => void;
  onSelectTriggerNode?: (value: string) => void;
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
      <p className="ar-meta">
        The pinned version of this agent is no longer readable, so what its runs
        may do cannot be shown. Delete this rule and create it again.
      </p>
    );
  }
  return (
    <dl className="as-kv ar-kv2">
      <div><dt>Runs as</dt><dd>{disclosure.agent} version {disclosure.definition_version}</dd></div>
      <div><dt>Reads</dt><dd>{disclosure.reads.length ? disclosure.reads.join(" · ") : "Nothing beyond the task text"}</dd></div>
      <div><dt>Public research</dt><dd>{disclosure.web_access}</dd></div>
      <div><dt>Integrations</dt><dd>{disclosure.integrations.length ? disclosure.integrations.join(" · ") : "None"}</dd></div>
      <div className="ar-kv-wide"><dt>Changes</dt><dd>{disclosure.writes}</dd></div>
    </dl>
  );
}

/** The grants one click deeper, where the row already carries a problem to read first. */
function GrantsDisclosure({ disclosure }: { disclosure?: CapabilityDisclosure }) {
  if (!disclosure) return null;
  return (
    <details className="ar-grants">
      <summary className="ar-disc ar-disc--compact"><span>What its runs may do</span><Icon className="ar-disc__chevron" name="chevron" size={ICON_SIZE.inline} /></summary>
      <UnattendedGrants disclosure={disclosure} />
    </details>
  );
}

/** A problem status gets a warning notice; a routine one a quiet line. */
const PROBLEM_STATES = new Set(["failing", "failed_once", "not_reporting", "delivery_failing"]);

function StatusDetail({ status }: { status?: AutomationItemStatus }) {
  if (!status?.detail) return null;
  if (PROBLEM_STATES.has(status.state)) {
    return <Notice severity="warning">{status.detail}</Notice>;
  }
  return <span className="ar-meta">{status.detail}</span>;
}

/**
 * What deleting a schedule or alert rule removes (W4d-D14). The rule and its
 * own trigger history go; the operations it already started are audit records
 * and stay in Activity. The dialog used to say its "run records" were removed,
 * and the owner then found the run still listed in Operations.
 */
export function automationDeleteDescription(name: string): string {
  return `Delete "${name}"? It stops running and its own trigger history is removed. Operations it already started stay in Activity as the record of what ran.`;
}

/** One row: what it is on the left, its pill and its actions on the right. */
function RuleRow({ actions, children, pill }: { actions: ReactNode; children: ReactNode; pill: ReactNode }) {
  return (
    <article className="ar-rule">
      <div className="ar-rule__body">{children}</div>
      <div className="ar-rule__side">{pill}<span className="ar-actions">{actions}</span></div>
    </article>
  );
}

export function AgentCenterAutomationsPanel({
  alertChannelsPanel,
  alertTargets,
  automationDelete,
  automationName,
  automationPrompt,
  automationSchedule,
  automationScheduleValid,
  automations,
  busy,
  modelConfigured = true,
  modelReady,
  notice,
  noticeRefused = false,
  profile,
  profiles,
  triggerName,
  triggerNode = "",
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
  onSelectTriggerNode,
  onSelectTriggerSource,
  onSetTriggerThreshold,
  onToggleAutomation,
  onToggleTrigger,
}: AgentCenterAutomationsPanelProps) {
  // Which create dialog is open, and which one was submitted and is waiting
  // on its answer: a refusal from that submission is shown inside it (the page
  // under a dialog is inert, VD-189), and success closes it.
  const [open, setOpen] = useState<"" | "schedule" | "trigger">("");
  const [pending, setPending] = useState<"" | "schedule" | "trigger">("");
  const [answered, setAnswered] = useState(false);
  // A refusal shown in a dialog must be the answer to ITS submission, not a
  // notice left on the page by an earlier action.
  const [submittedHere, setSubmittedHere] = useState(false);
  // No targets passed: the selector is hidden and the form stays controller-only.
  const workers = alertTargets ?? [];
  const targetingWorker = triggerNode !== "";
  // A worker reports only cpu_temperature and memory_percent, so limit the
  // signal list to those when a worker is the target; the controller keeps all.
  const sourceOptions = ([
    ["cpu_temperature", "CPU temperature"],
    ["memory_percent", "Memory use"],
    ["storage_percent", "Storage use"],
    ["service_failures", "Failed Vaelor services"],
    ["fan_failure", "Fan failure signal"],
  ] as const).filter(([value]) => !targetingWorker || isWorkerCapableSource(value));
  // The name to show for a rule's target: the controller, the worker's name, or
  // a plain sentence when the worker is no longer in the passed fleet - never
  // its raw node id (ACC-140).
  const targetLabel = (node?: string) => {
    if (!node) return "Controller";
    return workers.find((worker) => worker.id === node)?.name ?? (alertTargets ? "A worker no longer in the fleet" : "A worker");
  };
  // The agent a schedule runs, by name: the pinned definition's own name, then
  // the profile list, never the profile id.
  const agentName = (automation: Automation) =>
    automation.capability_disclosure?.agent
    || profiles.find((item) => item.id === automation.profile)?.name
    || "An agent that is no longer available";

  /*
   * A dialog closes when its create succeeded. The container's handler may
   * return the hook's own answer (true on success); the Assistant's passes it
   * through. Where it returns nothing, success is read the way the hook marks
   * it: it clears the name field it just saved, and a refusal leaves it.
   */
  const [sawBusy, setSawBusy] = useState(false);
  useEffect(() => {
    if (!pending || answered) return;
    // Wait for the submission's own round trip: busy on, then busy off.
    if (busy) { setSawBusy(true); return; }
    if (!sawBusy) return;
    const cleared = pending === "schedule" ? automationName === "" : triggerName === "";
    if (cleared) setOpen("");
    setPending("");
    setSawBusy(false);
  }, [answered, automationName, busy, pending, sawBusy, triggerName]);
  const submit = (kind: "schedule" | "trigger", handler: CreateHandler) => (event: FormEvent) => {
    const result = handler(event);
    setPending(kind);
    setSubmittedHere(true);
    if (result instanceof Promise) {
      setAnswered(true);
      void result.then((created) => {
        if (created === true) setOpen("");
      }).finally(() => { setAnswered(false); setPending(""); });
    }
  };
  const openDialog = (kind: "schedule" | "trigger") => { setPending(""); setSawBusy(false); setSubmittedHere(false); setOpen(kind); };
  const dialogError = open && submittedHere && noticeRefused && notice ? notice : "";

  const modelPill = <StatusPill label={modelReady ? "Read-only runs" : modelConfigured ? "Model not answering" : "Model not set up"} status={modelReady ? "healthy" : "degraded"} />;
  const modelReason = modelConfigured ? "Waiting for the Assistant model to answer" : "No Assistant model is set up";
  // `modelReady` is "the model is answering" (ACC-136); a model that was
  // never set up is a different sentence from one that is down.
  const modelNotice = !modelReady && (
    <Notice severity="warning">{modelConfigured
      ? "The Assistant model is not answering, so new schedules cannot be created until it does."
      : "No Assistant model is set up yet. Connect or select one before scheduling appliance checks or agent work."}</Notice>
  );

  const scheduleFields = (
    <>
      <Input id="automation-name" label="Schedule name" maxLength={100} onChange={(event) => onSetAutomationName(event.target.value)} value={automationName} />
      <Select id="automation-profile" label="Appliance check or agent" onChange={(event) => onSetProfile(event.target.value)} value={profile}>{profiles.map((item) => <option disabled={!item.operational} key={item.id} value={item.id}>{item.name}{item.operational ? "" : " unavailable"}</option>)}</Select>
      <Textarea id="automation-prompt" label="What should it check?" maxLength={4000} onChange={(event) => onSetAutomationPrompt(event.target.value)} rows={3} value={automationPrompt} />
      <Input
        error={automationScheduleValid ? undefined : "Use a future date, \"in 30 minutes\", or \"every 6 hours\" (5 minutes to 30 days)."}
        hint="Schedule expressions are checked before they are saved."
        id="automation-schedule"
        label="When"
        maxLength={120}
        onChange={(event) => onSetAutomationSchedule(event.target.value)}
        value={automationSchedule}
      />
    </>
  );
  const scheduleSubmit = (
    <Button disabled={busy || !automationName.trim() || !automationPrompt.trim() || !automationScheduleValid} disabledReason={modelReady ? undefined : modelReason} type="submit" variant="primary">Create schedule</Button>
  );
  const triggerFields = (
    <>
      <Input id="trigger-name" label="Alert name" maxLength={100} onChange={(event) => onSetTriggerName(event.target.value)} placeholder="Example: CPU running hot" value={triggerName} />
      {workers.length > 0 && (
        <Select hint="A worker offers only the signals it reports." id="trigger-node" label="Watch machine" onChange={(event) => onSelectTriggerNode?.(event.target.value)} value={triggerNode}>
          <option value="">Controller</option>
          {workers.map((worker) => <option key={worker.id} value={worker.id}>{worker.name}</option>)}
        </Select>
      )}
      <Select id="trigger-source" label="Watch signal" onChange={(event) => onSelectTriggerSource(event.target.value)} value={triggerSource}>
        {sourceOptions.map(([value, label]) => <option key={value} value={value}>{label}</option>)}
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
    </>
  );
  const triggerSubmit = (
    <Button disabled={busy || !triggerThresholdValid} disabledReason={triggerName.trim() ? undefined : "Name the rule first"} type="submit" variant="primary">Enable alert rule</Button>
  );

  const scheduleRows = automations.map((automation) => {
    const pill = automationPill(automation.status);
    // A one-time schedule that already ran cannot run again, so it offers
    // no Enable (the server refuses it too) - only Delete.
    const finished = scheduleIsSpent(automation.status);
    const problem = Boolean(automation.status && PROBLEM_STATES.has(automation.status.state));
    return (
      <RuleRow
        actions={(
          <>
            {!finished && <Button aria-pressed={automation.enabled} disabled={busy} onClick={() => onToggleAutomation(automation)}>{automation.enabled ? "Pause" : "Enable"}</Button>}
            <Button className="as-btn-danger" disabled={busy} onClick={() => onSetAutomationDelete({ kind: "schedule", id: automation.id, name: automation.name })}>Delete</Button>
          </>
        )}
        key={automation.id}
        pill={<StatusPill label={pill.label} reading={pill.reading} tone={pill.tone} />}
      >
        <span className="ar-meta">{agentName(automation)} · {scheduleKindLabel(automation.kind)}</span>
        <h3 className="ar-rule__name">{automation.name}</h3>
        {!finished && <p className="ar-agent__purpose">{automation.prompt}</p>}
        <span className="ar-meta">
          {automation.schedule_text}{showsNextRun(automation.status) && automation.next_run_at ? ` · next ${timeUntil(automation.next_run_at * 1000)}` : ""}
          {automation.last_run ? ` · Last run ${automation.last_run.state_label}${automation.last_run.at ? ` ${timeAgo(automation.last_run.at * 1000)}` : ""}` : ""}
        </span>
        <StatusDetail status={automation.status} />
        {problem || finished ? <GrantsDisclosure disclosure={automation.capability_disclosure} /> : <UnattendedGrants disclosure={automation.capability_disclosure} />}
      </RuleRow>
    );
  });

  const triggerRows = triggers.map((trigger) => {
    const pill = automationPill(trigger.status);
    return (
      <RuleRow
        actions={(
          <>
            <Button disabled={busy} onClick={() => onToggleTrigger(trigger)}>{trigger.enabled ? "Pause" : "Enable"}</Button>
            <Button className="as-btn-danger" disabled={busy} onClick={() => onSetAutomationDelete({ kind: "trigger", id: trigger.id, name: trigger.name })}>Delete</Button>
          </>
        )}
        key={trigger.id}
        pill={<StatusPill label={pill.label} reading={pill.reading} tone={pill.tone} />}
      >
        <span className="ar-meta">{targetLabel(trigger.node)} · {signalLabel(trigger)} {operatorLabel(trigger.operator)} {trigger.threshold}</span>
        <h3 className="ar-rule__name">{trigger.name}</h3>
        <p className="ar-agent__purpose">{trigger.prompt}</p>
        <span className="ar-meta">
          {trigger.last_value == null
            ? "No reading yet"
            : `Latest value ${trigger.last_value}${trigger.last_value_at ? `, read ${timeAgo(trigger.last_value_at * 1000)}` : ""}`}
          {trigger.last_triggered_at ? ` · fired ${timeAgo(trigger.last_triggered_at * 1000)}` : ""}
        </span>
        <StatusDetail status={trigger.status} />
        <GrantsDisclosure disclosure={trigger.capability_disclosure} />
      </RuleRow>
    );
  });

  const newScheduleButton = (variant: "primary" | "secondary") => (
    <Button disabled={!modelReady} disabledReason={modelReady || variant === "primary" ? undefined : modelReason} onClick={() => openDialog("schedule")} variant={variant}><Icon className="ar-btn-icon" name="add" size={ICON_SIZE.inline} />New schedule</Button>
  );
  const newRuleButton = <Button onClick={() => openDialog("trigger")}><Icon className="ar-btn-icon" name="add" size={ICON_SIZE.inline} />New alert rule</Button>;
  const schedulesEmpty = (
    <EmptyState action={<Button disabled={!modelReady} onClick={() => openDialog("schedule")}>New schedule</Button>}
      icon={<Icon name="activity" size={18} />} text="Create a safe health check or recurring review." title="No schedules yet" />
  );
  const rulesEmpty = (
    <EmptyState action={<Button onClick={() => openDialog("trigger")}>New alert rule</Button>}
      icon={<Icon name="alert" size={18} />} text="Add a temperature, storage, memory, service, or fan rule." title="No alert rules yet" />
  );
  const rulesLead = "Launch a read-only diagnostic when a live hardware signal crosses a safe threshold. Enabling the rule is the approval for every diagnostic it launches; none of them stops for another one. A 30-minute cooldown prevents alert storms.";
  const pageNotice = notice && !open ? <Notice severity={noticeRefused ? "danger" : "info"}>{notice}</Notice> : null;

  const deleteConfirm = (
    <ConfirmDialog
      busy={busy}
      confirmLabel={automationDelete?.kind === "schedule" ? "Delete schedule" : "Delete alert"}
      description={automationDelete ? automationDeleteDescription(automationDelete.name) : ""}
      error={automationDelete?.error}
      onCancel={() => onSetAutomationDelete(null)}
      onConfirm={onDeleteAutomationItem}
      open={Boolean(automationDelete)}
      title={automationDelete?.kind === "schedule" ? "Delete schedule?" : "Delete alert rule?"}
    />
  );

  return (
    <section aria-label="Schedules and alerts" className="ar-automations" id="agent-automations">
      <AssistantBarActions>
        {modelPill}
        {newScheduleButton("primary")}
      </AssistantBarActions>
      {modelNotice}
      {pageNotice}
      <div className={alertChannelsPanel ? "as-split ar-split" : "ar-single"}>
        <div className="ar-column">
          <section aria-labelledby="agent-schedules-title" className="card ui-card ar-card">
            <header className="ar-card__head">
              <div><h2 className="ar-card__title" id="agent-schedules-title">Schedules</h2><span className="ar-meta">Unattended work, on a clock</span></div>
              {newScheduleButton("secondary")}
            </header>
            {automations.length ? <div className="ar-rows">{scheduleRows}</div> : schedulesEmpty}
          </section>
          <section aria-labelledby="agent-alert-rules-title" className="card ui-card ar-card">
            <header className="ar-card__head ar-card__head--start">
              <div>
                <span className="as-label">Event-driven help</span>
                <h2 className="ar-card__title" id="agent-alert-rules-title">Alert rules</h2>
                <p className="ar-step__hint">{rulesLead}</p>
              </div>
              {newRuleButton}
            </header>
            {triggers.length ? <div className="ar-rows">{triggerRows}</div> : rulesEmpty}
          </section>
        </div>
        {alertChannelsPanel}
      </div>
      {open === "schedule" && (
        <ModalShell className="as-dialog ar-dialog-sm" error={dialogError} labelledBy="new-schedule-title" onClose={() => setOpen("")}>
          <form className="ar-dialog-form" onSubmit={submit("schedule", onCreateAutomation)}>
            <div className="as-dialog__head">
              <div><span className="as-label">Unattended work</span><h2 id="new-schedule-title">New schedule</h2></div>
              <Button aria-label="Close new schedule" className="as-btn-ghost" onClick={() => setOpen("")} variant="quiet">Close</Button>
            </div>
            <div className="as-dialog__body">{scheduleFields}</div>
            <div className="as-dialog__foot"><Button onClick={() => setOpen("")}>Cancel</Button>{scheduleSubmit}</div>
          </form>
        </ModalShell>
      )}
      {open === "trigger" && (
        <ModalShell className="as-dialog ar-dialog-sm" error={dialogError} labelledBy="new-alert-rule-title" onClose={() => setOpen("")}>
          <form className="ar-dialog-form" onSubmit={submit("trigger", onCreateTrigger)}>
            <div className="as-dialog__head">
              <div><span className="as-label">Event-driven help</span><h2 id="new-alert-rule-title">New alert rule</h2></div>
              <Button aria-label="Close new alert rule" className="as-btn-ghost" onClick={() => setOpen("")} variant="quiet">Close</Button>
            </div>
            <div className="as-dialog__body">{triggerFields}</div>
            <div className="as-dialog__foot"><Button onClick={() => setOpen("")}>Cancel</Button>{triggerSubmit}</div>
          </form>
        </ModalShell>
      )}
      {deleteConfirm}
    </section>
  );
}
