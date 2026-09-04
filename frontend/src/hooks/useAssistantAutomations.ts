import { useCallback, useState, type FormEvent, type RefObject } from "react";
import { apiRequest } from "../lib/api";
import type { Automation, Trigger } from "../components/agentTypes";
import { isValidAutomationSchedule, isValidTriggerThreshold } from "../components/automationValidation";

/** Scheduled runs the server recorded, used to label a run nobody typed. */
interface AutomationRun {
  id: string;
  automation_id: string;
  task_id: string | null;
}

/** Alert-rule runs the server recorded, the other kind of run nobody typed. */
interface TriggerRun {
  id: string;
  trigger_id: string;
  task_id: string | null;
}

export interface AutomationDelete {
  kind: "schedule" | "trigger";
  id: string;
  name: string;
}

/**
 * Everything about work that runs without being asked.
 *
 * Pulled out of `AgentCenter` so the container shrinks as the Routines tab
 * grows rather than the other way round: schedules, alert rules, their forms,
 * their validation and the server's own record of which tasks they produced are
 * one subject, and none of it is needed to ask a question.
 */
export function useAssistantAutomations({
  csrfToken,
  profile,
  refreshRef,
  setBusy,
  setNotice,
}: {
  csrfToken: string;
  /** The appliance check or agent a new schedule is pinned to. */
  profile: string;
  /** The container's own refresh, read at call time to avoid a definition cycle. */
  refreshRef: RefObject<() => Promise<unknown>>;
  setBusy: (busy: boolean) => void;
  setNotice: (message: string) => void;
}) {
  const [automations, setAutomations] = useState<Automation[]>([]);
  const [triggers, setTriggers] = useState<Trigger[]>([]);
  const [automaticTaskIds, setAutomaticTaskIds] = useState<ReadonlySet<string>>(new Set());
  const [automationName, setAutomationName] = useState("");
  const [automationPrompt, setAutomationPrompt] = useState("");
  const [automationSchedule, setAutomationSchedule] = useState("every 24 hours");
  const [triggerName, setTriggerName] = useState("");
  const [triggerSource, setTriggerSource] = useState("cpu_temperature");
  const [triggerThreshold, setTriggerThreshold] = useState(80);
  const [automationDelete, setAutomationDelete] = useState<AutomationDelete | null>(null);

  const load = useCallback(async () => {
    const data = await apiRequest<{
      schedules: Automation[]; runs?: AutomationRun[];
      trigger_runs?: TriggerRun[]; triggers: Trigger[];
    }>("/assistant/automations");
    setAutomations(data.schedules);
    setTriggers(data.triggers ?? []);
    /*
     * The server records every run it started — scheduled runs against the
     * schedule, alert-rule runs against the trigger — so both lists together
     * are the only trustworthy way to say a run happened on its own rather than
     * guessing from a title the user could have written themselves. Missing the
     * trigger runs classified fired-alert runs under Checks beside typed ones.
     */
    setAutomaticTaskIds(new Set(
      [...(data.runs ?? []), ...(data.trigger_runs ?? [])]
        .map((run) => run.task_id)
        .filter((id): id is string => Boolean(id)),
    ));
  }, []);

  const createAutomation = async (event: FormEvent) => {
    event.preventDefault(); setBusy(true); setNotice("");
    try {
      await apiRequest("/assistant/automations", { method: "POST", body: JSON.stringify({
        name: automationName, prompt: automationPrompt, profile, schedule: automationSchedule,
      }) }, csrfToken);
      setAutomationName(""); setAutomationPrompt("");
      setNotice("Schedule created. Its runs start on their own, limited to the selected appliance check or agent, and read only.");
      await refreshRef.current?.();
      return true;
    } catch (error) {
      setNotice(error instanceof Error ? error.message : "The schedule could not be created.");
    } finally { setBusy(false); }
  };

  const toggleAutomation = async (automation: Automation) => {
    setBusy(true);
    try {
      await apiRequest(`/assistant/automations/${automation.id}`, {
        method: "PATCH", body: JSON.stringify({ enabled: !automation.enabled }),
      }, csrfToken);
      await refreshRef.current?.();
      return true;
    } catch (error) {
      setNotice(error instanceof Error ? error.message : "The schedule could not be updated.");
    } finally { setBusy(false); }
  };

  const createTrigger = async (event: FormEvent) => {
    event.preventDefault(); setBusy(true); setNotice("");
    try {
      await apiRequest("/assistant/triggers", { method: "POST", body: JSON.stringify({
        name: triggerName,
        prompt: `Inspect the appliance and explain why ${triggerSource.replaceAll("_", " ")} crossed its alert threshold. Recommend safe next steps.`,
        profile: "system",
        source: triggerSource, operator: ">=", threshold: triggerThreshold,
        cooldown_seconds: 1800,
      }) }, csrfToken);
      setTriggerName("");
      setNotice("Alert rule enabled. It launches a read-only diagnostic on its own, without a further approval.");
      await refreshRef.current?.();
      return true;
    } catch (error) {
      setNotice(error instanceof Error ? error.message : "The alert rule could not be created.");
    } finally { setBusy(false); }
  };

  const toggleTrigger = async (trigger: Trigger) => {
    setBusy(true);
    try {
      await apiRequest(`/assistant/triggers/${trigger.id}`, {
        method: "PATCH", body: JSON.stringify({ enabled: !trigger.enabled }),
      }, csrfToken);
      await refreshRef.current?.();
      return true;
    } catch (error) {
      setNotice(error instanceof Error ? error.message : "The alert rule could not be updated.");
    } finally { setBusy(false); }
  };

  const deleteAutomationItem = async () => {
    if (!automationDelete) return;
    setBusy(true); setNotice("");
    try {
      const collection = automationDelete.kind === "schedule" ? "automations" : "triggers";
      await apiRequest(
        `/assistant/${collection}/${automationDelete.id}`,
        { method: "DELETE" },
        csrfToken,
      );
      setNotice(automationDelete.kind === "schedule" ? "Schedule deleted." : "Alert rule deleted.");
      setAutomationDelete(null);
      await refreshRef.current?.();
      return true;
    } catch (error) {
      setNotice(error instanceof Error ? error.message : "The automation could not be deleted.");
    } finally { setBusy(false); }
  };

  const selectTriggerSource = (source: string) => {
    setTriggerSource(source);
    setTriggerThreshold(
      source === "cpu_temperature" ? 80
        : source === "service_failures" || source === "fan_failure" ? 1
          : 90,
    );
  };

  return {
    automaticTaskIds,
    automationDelete,
    automationName,
    automationPrompt,
    automationSchedule,
    automationScheduleValid: isValidAutomationSchedule(automationSchedule),
    automations,
    createAutomation,
    createTrigger,
    deleteAutomationItem,
    load,
    selectTriggerSource,
    setAutomationDelete,
    setAutomationName,
    setAutomationPrompt,
    setAutomationSchedule,
    setTriggerName,
    setTriggerThreshold,
    toggleAutomation,
    toggleTrigger,
    triggerName,
    triggerSource,
    triggerThreshold,
    triggerThresholdValid: isValidTriggerThreshold(triggerSource, triggerThreshold),
    triggers,
  };
}
