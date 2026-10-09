import { useCallback, useEffect, useRef, useState } from "react";
import { ApiError, apiRequest } from "../lib/api";
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
import { useAssistantAutomations } from "../hooks/useAssistantAutomations";
import type { AgentProfile, AgentStatus, Automation, CapabilityDisclosure, Trigger } from "./agentTypes";
import type { FleetSummary } from "./fleetTypes";
import type { Session } from "../types";
import { automationDeleteDescription } from "./agent-center-automations-panel";
import { AlertChannelsPanel } from "./AlertChannelsPanel";
import { isWorkerCapableSource, triggerLimits } from "./automationValidation";
import { ClusterCard } from "./ClusterPrimitives";
import { Icon } from "./Icon";
import { ClusterConfirm } from "./ClusterConfirm";
import { StatusPill } from "./StatusPill";
import { Button, EmptyState, Input, Notice, Select, Textarea } from "./ui";
import "../styles/cluster-activity.css";

/**
 * Fleet alert management on the Cluster page's Activity tab (VD-200, the
 * ClusterActivityAlerts board): three cards - Alert rules, Where should alerts
 * go?, and Run this without being asked.
 *
 * It drives the SAME records the Assistant's Agents tab manages, through the
 * same `useAssistantAutomations` hook (same routes, same create, toggle and
 * delete calls, same delete confirmation), so a rule or schedule made here is
 * the one seen there. The delivery channels are the Assistant's own
 * `AlertChannelsPanel`, placed here by import only. This module draws the rule
 * and schedule cards in the Cluster page's layout; it does not change what any
 * of them sends.
 */
/** How often a rule still waiting for its first reading is re-read (W5-D9). */
export const WAITING_REREAD_MS = 10_000;
/** The most re-reads one waiting spell gets (2 min): bounded, so it settles (VD-173). */
export const WAITING_REREAD_CAP = 12;

/** A problem status gets a warning notice; a routine one a quiet line. */
const PROBLEM_STATES = new Set(["failing", "failed_once", "not_reporting", "delivery_failing"]);

/** The signals a rule can watch; a worker reports only the first two (`isWorkerCapableSource`). */
const SIGNAL_OPTIONS: ReadonlyArray<readonly [string, string]> = [
  ["cpu_temperature", "CPU temperature"],
  ["memory_percent", "Memory use"],
  ["storage_percent", "Storage use"],
  ["service_failures", "Failed Vaelor services"],
  ["fan_failure", "Fan failure signal"],
];

function StatusDetail({ status }: { status?: AutomationItemStatus }) {
  if (!status?.detail) return null;
  if (PROBLEM_STATES.has(status.state)) return <Notice severity="warning">{status.detail}</Notice>;
  return <p className="cl-meta">{status.detail}</p>;
}

/**
 * What this item's unattended runs may do, on the item. Nothing stops an
 * unattended run for a second approval, so creating it is the approval - and
 * an approval whose terms cannot be read is not one.
 */
function UnattendedGrants({ disclosure }: { disclosure?: CapabilityDisclosure }) {
  if (!disclosure) return null;
  if (!disclosure.pinned_definition_available) {
    return (
      <p className="cl-warn-text cl-alerts__grants">
        The pinned version of this agent is no longer readable, so what its runs may do cannot be
        shown. Delete this rule and create it again.
      </p>
    );
  }
  const parts = [
    `Runs as ${disclosure.agent} version ${disclosure.definition_version}`,
    `Reads: ${disclosure.reads.length ? disclosure.reads.join(", ") : "nothing beyond the task text"}`,
    `Public research: ${disclosure.web_access}`,
    `Integrations: ${disclosure.integrations.length ? disclosure.integrations.join(", ") : "none"}`,
    `Changes: ${disclosure.writes}`,
  ];
  return <p className="cl-meta cl-alerts__grants">{parts.join(" · ")}</p>;
}

/**
 * `reloadKey` is the Cluster page's Reload counter: a change re-reads the rules.
 * W5-D9: the panel read once at mount, so a new rule's card kept "Waiting for a
 * reading" after the server reported Watching, and Reload did not reach it.
 */
export function ClusterAlerts({ session, reloadKey }: { session: Session; reloadKey?: number }) {
  const isAdministrator = session.user.role === "administrator";
  const [profiles, setProfiles] = useState<AgentProfile[]>([]);
  const [agentStatus, setAgentStatus] = useState<AgentStatus | null>(null);
  const [intelligenceChoice, setIntelligenceChoice] = useState<"" | "basic" | "local" | "provider" | null>(null);
  const [profile, setProfile] = useState("system");
  const [busy, setBusy] = useState(false);
  const [notice, setNoticeText] = useState("");
  // A refusal is announced as an alert, a confirmation as a status (VD-189).
  const [noticeRefused, setNoticeRefused] = useState(false);
  const setNotice = useCallback((message: string, refused = false) => { setNoticeText(message); setNoticeRefused(refused); }, []);
  // The enrolled workers a rule may target, read from the same /cluster summary
  // the rest of the fleet uses so the selector cannot list a node the create
  // route would then reject. Empty until it loads (or on a controller-only box).
  const [alertTargets, setAlertTargets] = useState<{ id: string; name: string }[]>([]);

  const refreshRef = useRef<() => Promise<unknown>>(() => Promise.resolve());
  const automation = useAssistantAutomations({
    csrfToken: session.csrf_token,
    profile,
    refreshRef,
    setBusy,
    setNotice,
  });
  const loadAutomations = automation.load;

  const refresh = useCallback(async () => {
    const [nextProfiles, nextStatus, nextPreferences] = await Promise.all([
      apiRequest<AgentProfile[]>("/assistant/profiles"),
      apiRequest<AgentStatus>("/agent/status"),
      apiRequest<{ intelligence_choice: "" | "basic" | "local" | "provider" }>("/assistant/preferences"),
    ]);
    setProfiles(nextProfiles);
    setAgentStatus(nextStatus);
    setIntelligenceChoice(nextPreferences.intelligence_choice);
    // The fleet read is best-effort: a controller-only box, or a summary that
    // cannot be read, simply leaves the target list empty and the form stays
    // controller-only rather than failing the whole alerts screen.
    try {
      const fleet = await apiRequest<FleetSummary>("/cluster");
      setAlertTargets((fleet.enrolled_nodes ?? []).map(
        (node) => ({ id: node.id, name: node.name || node.id }),
      ));
    } catch {
      setAlertTargets([]);
    }
    await loadAutomations();
  }, [loadAutomations]);
  useEffect(() => { refreshRef.current = refresh; }, [refresh]);
  // LESSONS 24: the server's sentence is shown; anything else is logged.
  const readFailed = useCallback((error: unknown) => {
    if (!(error instanceof ApiError)) console.error("Alert rules could not be loaded.", error);
    setNotice(error instanceof ApiError && error.message ? error.message : "Alert rules could not be loaded.", true);
  }, [setNotice]);
  useEffect(() => {
    if (!isAdministrator) return;
    void refresh().catch(readFailed);
  }, [isAdministrator, readFailed, refresh]);

  // W5-D9: the page's Reload re-reads the rules (the first render already did).
  const seenReloadKey = useRef(reloadKey);
  useEffect(() => {
    if (!isAdministrator || seenReloadKey.current === reloadKey) return;
    seenReloadKey.current = reloadKey;
    void loadAutomations().catch(readFailed);
  }, [isAdministrator, loadAutomations, readFailed, reloadKey]);

  // W5-D9: a rule's state is derived by the server when it is read, and a new
  // rule says "waiting" until the evaluator first samples its signal. While any
  // rule waits, re-read on a bounded cadence: at most WAITING_REREAD_CAP times,
  // never while the page is hidden, and never after this panel unmounts.
  // W6-4: the budget belongs to the SET of waiting rules, so a rule that starts
  // waiting later gets its own re-reads; and once a budget is spent the panel
  // says it stopped checking instead of leaving "Waiting" looking live.
  const waitingKey = automation.triggers
    .filter((trigger) => trigger.status?.state === "waiting")
    .map((trigger) => trigger.id)
    .sort()
    .join(",");
  const [spentFor, setSpentFor] = useState("");
  const [checkRound, setCheckRound] = useState(0);
  useEffect(() => {
    if (!isAdministrator || !waitingKey) return;
    let rereads = 0;
    const timer = window.setInterval(() => {
      if (document.visibilityState === "hidden") return;
      rereads += 1;
      if (rereads > WAITING_REREAD_CAP) {
        window.clearInterval(timer);
        setSpentFor(waitingKey);
        return;
      }
      void loadAutomations().catch(() => undefined);
    }, WAITING_REREAD_MS);
    return () => window.clearInterval(timer);
  }, [checkRound, isAdministrator, loadAutomations, waitingKey]);
  const stoppedChecking = Boolean(waitingKey) && spentFor === waitingKey;
  const checkAgain = () => {
    setSpentFor("");
    setCheckRound((round) => round + 1);
    void loadAutomations().catch(readFailed);
  };

  // A configured model that has not opted out of local intelligence; the same
  // readiness the Assistant uses to gate schedule creation. Alert rules and
  // delivery channels do not need a model; schedules do.
  const modelConfigured = Boolean(agentStatus?.configured);
  const modelReady = Boolean(agentStatus?.configured && intelligenceChoice !== "basic");

  if (!isAdministrator) {
    return (
      <section aria-labelledby="cluster-alerts-title" className="cl-stack">
        <h2 className="cl-alerts__title" id="cluster-alerts-title">Alerts</h2>
        <Notice severity="info">Alert rules and delivery channels are managed by an administrator.</Notice>
      </section>
    );
  }

  const workers = alertTargets;
  const targetingWorker = automation.triggerNode !== "";
  const sourceOptions = SIGNAL_OPTIONS.filter(([value]) => !targetingWorker || isWorkerCapableSource(value));
  // A rule's target by name: the controller, the worker's name, or a sentence
  // when the worker left the fleet - never its raw node id (ACC-140).
  const targetLabel = (node?: string) => {
    if (!node) return "Controller";
    return workers.find((worker) => worker.id === node)?.name ?? "A worker no longer in the fleet";
  };
  // The agent a schedule runs, by name - never the profile id.
  const agentName = (item: Automation) =>
    item.capability_disclosure?.agent
    || profiles.find((candidate) => candidate.id === item.profile)?.name
    || "An agent that is no longer available";
  const pendingDelete = automation.automationDelete;

  return (
    <section aria-label="Alerts and schedules" className="cl-stack cl-alerts">
      {/*
        Honest scope: the controller covers all five signals; a worker covers the
        two its telemetry carries, and the notice says exactly which.
      */}
      <Notice severity="info">
        The controller is watched on all five signals: CPU temperature, memory, storage, Vaelor service
        failures, and the fan signal. Each enrolled worker is watched on the two its telemetry reports:
        CPU temperature and memory. A rule fires only when its own machine crosses; a worker that is not
        reporting is skipped rather than raising a false alarm.
      </Notice>
      {stoppedChecking && (
        <Notice severity="warning">
          <span>
            Vaelor stopped checking for a first reading after {Math.round((WAITING_REREAD_MS * WAITING_REREAD_CAP) / 60_000)} minutes,
            so a rule that still says "Waiting for a reading" may have one by now.
          </span>
          <Button onClick={checkAgain} variant="quiet">Check again</Button>
        </Notice>
      )}
      {notice && <Notice severity={noticeRefused ? "danger" : "info"}>{notice}</Notice>}

      <div className="cl-alerts__grid">
        <ClusterCard description="Event-driven help" icon="alert" title="Alert rules">
          <p className="cl-meta">
            Launch a read-only diagnostic when a live hardware signal crosses a safe threshold.
            Enabling the rule is the approval for every diagnostic it launches. A 30-minute cooldown
            prevents alert storms.
          </p>
          <form className="cl-alerts__form" onSubmit={(event) => void automation.createTrigger(event)}>
            <Input id="trigger-name" label="Alert name" maxLength={100} onChange={(event) => automation.setTriggerName(event.target.value)} placeholder="Example: CPU running hot" value={automation.triggerName} />
            <div className={workers.length > 0 ? "cl-alerts__pair" : undefined}>
              {workers.length > 0 && (
                <Select id="trigger-node" label="Watch machine" onChange={(event) => automation.selectTriggerNode(event.target.value)} value={automation.triggerNode}>
                  <option value="">Controller</option>
                  {workers.map((worker) => <option key={worker.id} value={worker.id}>{worker.name}</option>)}
                </Select>
              )}
              <Select id="trigger-source" label="Watch signal" onChange={(event) => automation.selectTriggerSource(event.target.value)} value={automation.triggerSource}>
                {sourceOptions.map(([value, label]) => <option key={value} value={value}>{label}</option>)}
              </Select>
            </div>
            <Input
              aria-invalid={!automation.triggerThresholdValid}
              id="trigger-threshold"
              label="Alert at or above"
              max={triggerLimits[automation.triggerSource]?.[1]}
              min={triggerLimits[automation.triggerSource]?.[0]}
              onChange={(event) => automation.setTriggerThreshold(Number(event.target.value))}
              type="number"
              value={automation.triggerThreshold}
            />
            <div className="cl-actions cl-actions--end">
              <Button disabled={busy || !automation.triggerName.trim() || !automation.triggerThresholdValid} type="submit" variant="primary">Enable alert rule</Button>
            </div>
          </form>
          <div className="cl-alerts__items">
            {automation.triggers.map((trigger: Trigger) => {
              const pill = automationPill(trigger.status);
              return (
                <article aria-label={trigger.name} className="cl-alerts__item" key={trigger.id}>
                  <div className="cl-alerts__item-top">
                    <div>
                      <small>{targetLabel(trigger.node)} · {signalLabel(trigger)} {operatorLabel(trigger.operator)} {trigger.threshold}</small>
                      <h3>{trigger.name}</h3>
                    </div>
                    <StatusPill label={pill.label} reading={pill.reading} tone={pill.tone} />
                  </div>
                  <p className="cl-alerts__reading">
                    {trigger.last_value == null
                      ? "No reading yet"
                      : `Latest value ${trigger.last_value}${trigger.last_value_at ? `, read ${timeAgo(trigger.last_value_at * 1000)}` : ""}`}
                    {trigger.last_triggered_at ? ` · fired ${timeAgo(trigger.last_triggered_at * 1000)}` : ""}
                  </p>
                  {trigger.prompt && <p className="cl-meta">{trigger.prompt}</p>}
                  <StatusDetail status={trigger.status} />
                  <UnattendedGrants disclosure={trigger.capability_disclosure} />
                  <div className="cl-actions">
                    <Button disabled={busy} onClick={() => void automation.toggleTrigger(trigger)} variant="quiet">{trigger.enabled ? "Pause" : "Enable"}</Button>
                    <Button className="cl-danger-outline" disabled={busy} onClick={() => automation.setAutomationDelete({ kind: "trigger", id: trigger.id, name: trigger.name })}>Delete</Button>
                  </div>
                </article>
              );
            })}
            {automation.triggers.length === 0 && (
              <EmptyState icon={<Icon name="alert" />} text="Add a temperature, storage, memory, service, or fan rule above." title="No alert rules yet" />
            )}
          </div>
        </ClusterCard>

        {/* The Assistant's channels panel, headless, in this page's own card (the ClusterActivityAlerts board). */}
        <ClusterCard className="cl-alerts__channels" description="Delivery channels" icon="shield" title="Where should alerts go?">
          <AlertChannelsPanel canManage={isAdministrator} csrfToken={session.csrf_token} headless />
        </ClusterCard>

        <ClusterCard description="Unattended work · creating the schedule is the approval" icon="activity" title="Run this without being asked">
          <div className="cl-alerts__lead">
            <StatusPill label={modelReady ? "Read-only runs" : modelConfigured ? "Model not answering" : "Model not set up"} tone={modelReady ? "success" : "warning"} />
            <p className="cl-meta">
              Schedule an appliance check or one of your agents. Its runs start on their own. Each run
              reads only, stays in its own workspace, cannot create more schedules, and turns any change
              into a proposal that needs its own approval.
            </p>
          </div>
          {/* `modelReady` is "the model is answering" (ACC-136); a model that
              was never set up is a different sentence from one that is down. */}
          {!modelReady && (
            <Notice severity="warning">{modelConfigured
              ? "The Assistant model is not answering, so new schedules cannot be created until it does."
              : "No Assistant model is set up yet. Connect or select one before scheduling appliance checks or agent work."}</Notice>
          )}
          <form className="cl-alerts__form" onSubmit={(event) => void automation.createAutomation(event)}>
            <Input id="automation-name" label="Schedule name" maxLength={100} onChange={(event) => automation.setAutomationName(event.target.value)} placeholder="Nightly health check" value={automation.automationName} />
            <Select id="automation-profile" label="Appliance check or agent" onChange={(event) => setProfile(event.target.value)} value={profile}>
              {profiles.map((item) => <option disabled={!item.operational} key={item.id} value={item.id}>{item.name}{item.operational ? "" : " unavailable"}</option>)}
            </Select>
            <Textarea id="automation-prompt" label="What should it check?" maxLength={4000} onChange={(event) => automation.setAutomationPrompt(event.target.value)} placeholder="Check temperatures, storage and failed services; tell me what needs attention." rows={2} value={automation.automationPrompt} />
            <Input
              hint="Schedule expressions are checked before they are saved."
              id="automation-schedule"
              label="When"
              maxLength={120}
              onChange={(event) => automation.setAutomationSchedule(event.target.value)}
              placeholder="every 6 hours"
              value={automation.automationSchedule}
            />
            {!automation.automationScheduleValid && <p className="cl-warn-text cl-meta" role="alert">Use a future date, "in 30 minutes", or "every 6 hours" (5 minutes to 30 days).</p>}
            <div className="cl-actions cl-actions--end">
              <Button
                disabled={!modelReady || busy || !automation.automationName.trim() || !automation.automationPrompt.trim() || !automation.automationScheduleValid}
                type="submit"
                variant="primary"
              >
                Create schedule
              </Button>
            </div>
          </form>
          <div className="cl-alerts__items">
            {automation.automations.map((item) => {
              const pill = automationPill(item.status);
              // A one-time schedule that already ran cannot run again, so it
              // offers no Enable (the server refuses it too) - only Delete.
              const finished = scheduleIsSpent(item.status);
              const when = [
                item.schedule_text,
                showsNextRun(item.status) && item.next_run_at ? `next ${timeUntil(item.next_run_at * 1000)}` : "",
                item.last_run ? `Last run ${item.last_run.state_label}${item.last_run.at ? ` ${timeAgo(item.last_run.at * 1000)}` : ""}` : "",
              ].filter(Boolean).join(" · ");
              return (
                <article aria-label={item.name} className="cl-alerts__item" key={item.id}>
                  <div className="cl-alerts__item-top">
                    <div>
                      <small>{agentName(item)} · {scheduleKindLabel(item.kind)}</small>
                      <h3>{item.name}</h3>
                    </div>
                    <StatusPill label={pill.label} reading={pill.reading} tone={pill.tone} />
                  </div>
                  <p className="cl-alerts__reading">{item.prompt}</p>
                  <p className="cl-meta">{when}</p>
                  <StatusDetail status={item.status} />
                  <UnattendedGrants disclosure={item.capability_disclosure} />
                  <div className="cl-actions">
                    {!finished && <Button aria-pressed={item.enabled} disabled={busy} onClick={() => void automation.toggleAutomation(item)} variant="quiet">{item.enabled ? "Pause" : "Enable"}</Button>}
                    <Button className="cl-danger-outline" disabled={busy} onClick={() => automation.setAutomationDelete({ kind: "schedule", id: item.id, name: item.name })}>Delete</Button>
                  </div>
                </article>
              );
            })}
            {automation.automations.length === 0 && (
              <EmptyState icon={<Icon name="activity" />} text="Create a safe health check or recurring review above." title="No schedules yet" />
            )}
          </div>
        </ClusterCard>
      </div>

      <ClusterConfirm
        busy={busy}
        busyLabel="Deleting…"
        confirmLabel={pendingDelete?.kind === "schedule" ? "Delete schedule" : "Delete alert"}
        description={pendingDelete ? automationDeleteDescription(pendingDelete.name) : ""}
        error={pendingDelete?.error}
        eyebrow="Can't be undone"
        onCancel={() => automation.setAutomationDelete(null)}
        onConfirm={() => void automation.deleteAutomationItem()}
        open={Boolean(pendingDelete)}
        title={pendingDelete?.kind === "schedule" ? "Delete schedule?" : "Delete alert rule?"}
      />
    </section>
  );
}
