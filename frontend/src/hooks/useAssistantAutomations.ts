import { useCallback, useState, type FormEvent, type RefObject } from "react";
import { apiRequest } from "../lib/api";
import type { Automation, Trigger } from "../components/agentTypes";
import { isValidAutomationSchedule, isValidTriggerThreshold, isWorkerCapableSource } from "../components/automationValidation";

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
  /**
   * Why the delete was refused (VD-189). Carried on the pending delete itself so
   * the confirmation shows it inside the dialog for every caller of the panel,
   * and a fresh opening starts without it.
   */
  error?: string;
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
  /** `refused` marks a refusal, which the caller shows as an alert rather than a status (VD-189). */
  setNotice: (message: string, refused?: boolean) => void;
}) {
  const [automations, setAutomations] = useState<Automation[]>([]);
  const [triggers, setTriggers] = useState<Trigger[]>([]);
  const [automaticTaskIds, setAutomaticTaskIds] = useState<ReadonlySet<string>>(new Set());
  const [automationName, setAutomationName] = useState("");
  const [automationPrompt, setAutomationPrompt] = useState("");
  const [automationSchedule, setAutomationSchedule] = useState("every 24 hours");
  const [triggerName, setTriggerName] = useState("");
  const [triggerNode, setTriggerNode] = useState("");
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
      setNotice(error instanceof Error ? error.message : "The schedule could not be created.", true);
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
      setNotice(error instanceof Error ? error.message : "The schedule could not be updated.", true);
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
        // "" is the controller; a node id targets that enrolled worker (VD-128).
        node: triggerNode,
      }) }, csrfToken);
      setTriggerName("");
      setNotice("Alert rule enabled. It launches a read-only diagnostic on its own, without a further approval.");
      await refreshRef.current?.();
      return true;
    } catch (error) {
      setNotice(error instanceof Error ? error.message : "The alert rule could not be created.", true);
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
      setNotice(error instanceof Error ? error.message : "The alert rule could not be updated.", true);
    } finally { setBusy(false); }
  };

  const deleteAutomationItem = async () => {
    if (!automationDelete) return;
    setBusy(true); setNotice("");
    setAutomationDelete({ ...automationDelete, error: undefined });
    try {
      const collection = automationDelete.kind === "schedule" ? "automations" : "triggers";
      await apiRequest(
        `/assistant/${collection}/${automationDelete.id}`,
        { method: "DELETE" },
        csrfToken,
      );
    } catch (error) {
      // The confirmation stays open, so the refusal belongs in it (VD-189).
      setAutomationDelete({
        ...automationDelete,
        error: error instanceof Error && error.message ? error.message : "The automation could not be deleted.",
      });
      setBusy(false);
      return false;
    }
    // Deleted: the dialog closes. A reload that fails after it is not a refused
    // delete, so it is said on the page and never reopens the confirmation.
    setNotice(automationDelete.kind === "schedule" ? "Schedule deleted." : "Alert rule deleted.");
    setAutomationDelete(null);
    try {
      await refreshRef.current?.();
    } catch (error) {
      setNotice(`${automationDelete.kind === "schedule" ? "Schedule deleted" : "Alert rule deleted"}, but the list could not be reloaded: ${
        error instanceof Error && error.message ? error.message : "no reason was given"}`, true);
    } finally { setBusy(false); }
    return true;
  };

  const selectTriggerSource = (source: string) => {
    setTriggerSource(source);
    setTriggerThreshold(
      source === "cpu_temperature" ? 80
        : source === "service_failures" || source === "fan_failure" ? 1
          : 90,
    );
  };

  const selectTriggerNode = (node: string) => {
    setTriggerNode(node);
    // A worker cannot report a controller-only signal, so switching to one must
    // drop a now-invalid source back to a worker-capable default rather than
    // leave the form holding a source the server would refuse for that node.
    if (node && !isWorkerCapableSource(triggerSource)) {
      selectTriggerSource("cpu_temperature");
    }
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
    selectTriggerNode,
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
    triggerNode,
    triggerSource,
    triggerThreshold,
    triggerThresholdValid: isValidTriggerThreshold(triggerSource, triggerThreshold),
    triggers,
  };
}
