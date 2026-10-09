import { type FormEvent, type ReactNode, useEffect, useState } from "react";
import { apiRequest } from "../lib/api";
import { automationIsActive } from "../lib/automationStatus";
import { AssistantBarActions, useInAssistantBar } from "./assistantBar";
import { ConfirmDialog } from "./ConfirmDialog";
import { CustomAgentCard, UnscheduledAgentsNotice, type AgentRevision } from "./CustomAgentCard";
import { ActivationDialog, AutomationDialog, RunDialog, type AutomationForm } from "./CustomAgentDialogs";
import { CustomAgentEditor } from "./CustomAgentEditor";
import { EMPTY_DRAFT, type Draft, type KnowledgeCollection } from "./customAgentDraft";
import { Icon } from "./Icon";
import { IntegrationCapabilitiesContainer } from "./IntegrationCapabilitiesContainer";
import type { AgentConnector, AgentProfile, AgentTask, Automation, Trigger } from "./agentTypes";
import { Button, EmptyState, Notice, Textarea } from "./ui";

/*
 * An agent described as recurring ("every morning...") does not get a
 * schedule from being created, and nothing said so - the one thing the user
 * asked for silently never happened. It is stated once for the list rather
 * than repeated verbatim on every row.
 */
const RECURRING = /\b(every|each|daily|hourly|weekly|nightly)\b/i;

const EMPTY_AUTOMATION: AutomationForm = {
  kind: "schedule", name: "", prompt: "", schedule: "every 6 hours", source: "cpu_temperature", threshold: "80",
};

/**
 * The agent workshop: the Agents view of Routines (the AssistRoutines board),
 * and, with `surface="inference"`, the inference-agent workshop on Cluster >
 * Agents & tools.
 *
 * Its top-bar items (the model pill and New agent) go through
 * AssistantBarActions: inside the Assistant they sit in the console's top bar;
 * on Cluster they render in place.
 */
export function CustomAgentManager({
  automations = [],
  barStatus,
  clusterModelNote = "",
  clusterModelServing = false,
  csrfToken,
  modelLabel,
  modelNotAnsweringReason = "",
  modelReady,
  modelStatusResolved = true,
  onChanged,
  onViewRun,
  profiles,
  profilesRead = true,
  profilesReadError = "",
  surface = "assistant",
  tasks,
  triggers = [],
}: {
  automations?: Automation[];
  /** The page's model pill, drawn before New agent in the Assistant's top bar. */
  barStatus?: ReactNode;
  /** A cluster inference model is serving. The inference workshop authors on
      it; the Assistant workshop only points to where cluster agents are built. */
  clusterModelServing?: boolean;
  /** The inference workshop's sentence about the cluster model its agents
      answer with: a paused one (asleep and waking on request, or unloaded by
      hand), or which model serves, or why none does (ACC-079). */
  clusterModelNote?: string;
  csrfToken: string;
  modelLabel: string;
  /** Non-empty when the configured Assistant model is not answering (from the
      shared `assistantModelReadiness` projection). Run stays off meanwhile. */
  modelNotAnsweringReason?: string;
  modelReady: boolean;
  /** False until the Assistant's readiness has been read: no "connect a model" claim before then (VD-200). */
  modelStatusResolved?: boolean;
  onChanged: () => void;
  onViewRun?: (task: AgentTask) => void;
  profiles: AgentProfile[];
  /** False until the list was read once; the empty state waits for a real read. */
  profilesRead?: boolean;
  profilesReadError?: string;
  /** Which agent surface this workshop authors. An "assistant" agent runs in the
      Assistant runtime; an "inference" agent is a read-only definition deployed
      on the cluster, so its form offers no acting permissions and none of the
      Assistant-runtime affordances. */
  surface?: "assistant" | "inference";
  tasks: AgentTask[];
  triggers?: Trigger[];
}) {
  const inAssistantBar = useInAssistantBar();
  const [draft, setDraft] = useState<Draft | null>(null);
  const [activationAgent, setActivationAgent] = useState<AgentProfile | null>(null);
  const [busy, setBusy] = useState(false);
  const [notice, setNoticeText] = useState("");
  // A refusal is an alert; a confirmation is not (VD-189: an info refusal was never announced).
  const [noticeRefused, setNoticeRefused] = useState(false);
  const setNotice = (message: string, refused = false) => { setNoticeText(message); setNoticeRefused(refused); };
  // The open dialog's own refusal, shown inside it - never on the inert page beneath (VD-189).
  const [dialogError, setDialogError] = useState("");
  const [builderPrompt, setBuilderPrompt] = useState("");
  const [deleting, setDeleting] = useState<AgentProfile | null>(null);
  const [collections, setCollections] = useState<KnowledgeCollection[]>([]);
  const [revisions, setRevisions] = useState<Record<string, AgentRevision[]>>({});
  const [testAgent, setTestAgent] = useState<AgentProfile | null>(null);
  const [testRequest, setTestRequest] = useState("");
  // Inline validation for the run dialog: an empty task used to make Run now a
  // silent no-op, so the operator saw nothing happen and nothing said why.
  const [testError, setTestError] = useState("");
  // Which agents' run histories are expanded. Preparing a run opens the one it
  // landed in: the confirmation told people to approve it "below in this
  // agent's run history" while that disclosure sat closed and off-screen.
  const [openRuns, setOpenRuns] = useState<Record<string, boolean>>({});
  // The run currently being watched inside the run dialog.
  const [activeRunId, setActiveRunId] = useState("");
  const [automationAgent, setAutomationAgent] = useState<AgentProfile | null>(null);
  const [automationForm, setAutomationForm] = useState<AutomationForm>(EMPTY_AUTOMATION);
  const [connectorAudit, setConnectorAudit] = useState<Record<string, Array<Record<string, unknown>>>>({});
  // The topmost open dialog, in render order; a refusal goes to it, or to the page when none is open.
  const topDialog = deleting ? "delete" : automationAgent ? "automation" : testAgent ? "run" : activationAgent ? "activation" : draft ? "editor" : "";
  useEffect(() => setDialogError(""), [topDialog]);
  // A dialog's own result (a passed connection test, a prepared draft) is said
  // inside that dialog, never on the inert page under it (VD-189), and only
  // while that dialog is the one open.
  const [dialogStatus, setDialogStatus] = useState<{ dialog: string; text: string } | null>(null);
  useEffect(() => setDialogStatus((current) => (current?.dialog === topDialog ? current : null)), [topDialog]);
  const refuse = (error: unknown, fallback: string) => {
    const message = error instanceof Error && error.message ? error.message : fallback;
    if (topDialog) setDialogError(message);
    else setNotice(message, true);
  };
  // An inference agent is a read-only cluster definition: the form drops every
  // acting permission and the Assistant-runtime affordances (test runs,
  // automations, app access) that do not apply to a deployed cluster agent.
  const isInference = surface === "inference";
  // Authoring needs no model server-side (custom_agents.create and .draft are
  // model-free); the gate exists so a definition has a model that can back it.
  // Each workshop gates on ITS OWN backing model: a cluster agent on the
  // serving cluster model, an Assistant agent on the Assistant model. The
  // Assistant workshop (Routines) once opened on the cluster model and saved
  // an "assistant" agent that could neither run nor deploy (ACC-137); cluster
  // agents are authored in Cluster > Agents & tools, where they deploy from.
  const authoringReady = isInference ? clusterModelServing : modelReady;
  // Anything that sends work to the Assistant model (Run, test runs,
  // automations) needs it answering, not merely configured (ACC-136).
  const modelAnswering = modelReady && !modelNotAnsweringReason;
  // Why Run is off, said beside it: a configured model that is down is a
  // different sentence from one that was never connected.
  const runBlockedReason = modelAnswering ? undefined
    : modelReady ? "The Assistant model is not answering, so runs stay off until it does."
      : "Runs need a connected Assistant model.";
  const custom = profiles.filter((item) => item.custom);
  /*
   * App access is opened deliberately, never by default.
   *
   * `custom.find(...) ?? custom[0]` made this truthy for anybody who owned a
   * single agent, so a four-step credential wizard — about 1,400px and twenty
   * controls — was permanently appended to this tab and counted as first paint
   * for a job nobody had started. An empty id means the panel is closed; open,
   * it takes the right-hand column beside the agents.
   */
  const [appAccessAgentId, setAppAccessAgentId] = useState("");
  const appAccessAgent = appAccessAgentId
    ? custom.find((item) => item.id === appAccessAgentId) ?? null
    : null;

  useEffect(() => {
    // Only forget a selection whose agent is gone; never invent one.
    setAppAccessAgentId((current) => custom.some((item) => item.id === current) ? current : "");
  }, [profiles]);

  useEffect(() => {
    void apiRequest<{ collections: KnowledgeCollection[] }>("/ai-chat/setup")
      .then((result) => setCollections(result.collections))
      .catch(() => setCollections([]));
  }, []);

  const openCreate = () => {
    setDraft({ ...EMPTY_DRAFT, scopes: [...EMPTY_DRAFT.scopes] });
    setActivationAgent(null);
    setNotice("");
  };
  const openEdit = (profile: AgentProfile) => {
    setDraft({
      id: profile.id,
      name: profile.name,
      description: profile.description,
      instructions: profile.instructions ?? "",
      scopes: [...profile.scopes],
      permissions: [...(profile.permissions ?? [])],
      read_collection_ids: [...(profile.read_collection_ids ?? [])],
      write_collection_id: profile.write_collection_id ?? "",
      web_access: profile.web_access ?? { enabled: false, allowed_domains: [] },
      connectors: structuredClone(profile.connectors ?? []),
    });
    setActivationAgent(null);
  };

  const draftWithAssistant = async () => {
    setBusy(true); setNotice(""); setDialogError("");
    try {
      const next = await apiRequest<Draft>(
        "/assistant/custom-agents/draft",
        { method: "POST", body: JSON.stringify({ request: builderPrompt }) },
        csrfToken,
      );
      setDraft({ ...EMPTY_DRAFT, ...next, name: next.name.replace(/\s+specialist$/i, " agent"), web_access: next.web_access ?? { enabled: false, allowed_domains: [] }, connectors: next.connectors ?? [] });
      setDialogStatus({ dialog: "editor", text: "Draft prepared. Review every permission before creating the agent." });
    } catch (error) {
      refuse(error, "The assistant could not draft this agent.");
    } finally { setBusy(false); }
  };

  const save = async (event: FormEvent) => {
    event.preventDefault();
    if (!draft) return;
    setBusy(true); setNotice(""); setDialogError("");
    try {
      const saved = await apiRequest<AgentProfile>(
        draft.id ? `/assistant/custom-agents/${draft.id}` : "/assistant/custom-agents",
        { method: draft.id ? "PATCH" : "POST", body: JSON.stringify({ ...draft, surface }) },
        csrfToken,
      );
      setDraft(null);
      setActivationAgent(saved?.id ? saved : null);
      setNotice(draft.id ? "New agent version saved." : "Custom agent created.");
      onChanged();
    } catch (error) {
      refuse(error, "The agent could not be saved.");
    } finally { setBusy(false); }
  };

  const activateAgent = async () => {
    if (!activationAgent) return;
    setBusy(true); setNotice(""); setDialogError("");
    try {
      await apiRequest(
        `/assistant/custom-agents/${activationAgent.id}`,
        { method: "PATCH", body: JSON.stringify({ enabled: true }) },
        csrfToken,
      );
      setActivationAgent(null);
      setNotice(`${activationAgent.name} is active for reviewed runs.`);
      onChanged();
    } catch (error) {
      refuse(error, "The agent could not be activated.");
    } finally { setBusy(false); }
  };

  const setEnabled = async (profile: AgentProfile, enabled: boolean) => {
    setBusy(true); setNotice(""); setDialogError("");
    try {
      await apiRequest(
        `/assistant/custom-agents/${profile.id}`,
        { method: "PATCH", body: JSON.stringify({ enabled }) },
        csrfToken,
      );
      setNotice(enabled ? "Agent restored for new runs." : "Agent archived. Existing run history was retained.");
      onChanged();
    } catch (error) {
      refuse(error, "The agent could not be updated.");
    } finally { setBusy(false); }
  };

  const remove = async () => {
    if (!deleting) return;
    setBusy(true); setNotice(""); setDialogError("");
    try {
      await apiRequest(`/assistant/custom-agents/${deleting.id}`, { method: "DELETE" }, csrfToken);
      setDeleting(null);
      setNotice("Custom agent deleted. Existing task history was retained.");
      onChanged();
    } catch (error) {
      refuse(error, "The agent could not be deleted.");
    } finally { setBusy(false); }
  };

  const loadVersions = async (profile: AgentProfile) => {
    if (revisions[profile.id]) {
      setRevisions((current) => {
        const next = { ...current };
        delete next[profile.id];
        return next;
      });
      return;
    }
    setBusy(true); setNotice(""); setDialogError("");
    try {
      const result = await apiRequest<AgentRevision[]>(`/assistant/custom-agents/${profile.id}/revisions`);
      setRevisions((current) => ({ ...current, [profile.id]: result }));
    } catch (error) {
      refuse(error, "Agent versions could not be loaded.");
    } finally { setBusy(false); }
  };

  const transitionRun = async (task: AgentTask, state: "ready" | "cancelled") => {
    setBusy(true); setNotice(""); setDialogError("");
    try {
      await apiRequest(
        `/assistant/tasks/${task.id}`,
        { method: "PATCH", body: JSON.stringify({ state }) },
        csrfToken,
      );
      setNotice(state === "ready" ? "Custom-agent run approved." : "Custom-agent run cancelled.");
      onChanged();
    } catch (error) {
      refuse(error, "The custom-agent run could not be updated.");
    } finally { setBusy(false); }
  };

  const queueTestRun = async (event: FormEvent) => {
    event.preventDefault();
    if (!testAgent) return;
    if (!testRequest.trim()) { setTestError("Enter a task to run."); return; }
    setTestError("");
    setBusy(true); setNotice(""); setDialogError("");
    try {
      const created = await apiRequest<AgentTask>(
        "/assistant/tasks",
        {
          method: "POST",
          body: JSON.stringify({
            title: `${testAgent.name} test run`,
            description: testRequest.trim(),
            profile: testAgent.id,
            profile_version: testAgent.version,
            approval_required: true,
            idempotency_key: crypto.randomUUID(),
          }),
        },
        csrfToken,
      );
      // Keep this dialog open and turn it into the run's own surface. Closing
      // it here is what sent people hunting through disclosures for an outcome.
      if (created?.id) {
        setActiveRunId(created.id);
        /*
         * Start it. The operator typed this request one second ago and pressed
         * a button that says Run now - asking them to then approve their own
         * sentence is a checkpoint with nobody on the other side of it, and it
         * was the step everyone got stuck on. The task-level approval gate is
         * for a run the operator did not just type: a request matched from
         * chat. Schedules and triggers do not use it and never have - their
         * runs are created ready. Creating the rule is the approval for them,
         * which is why creating one is administrator-only, and why an
         * unattended run can execute reads but turns every write into a
         * proposal that needs its own approval.
         */
        await transitionRun(created, "ready");
      }
      setOpenRuns((current) => ({ ...current, [testAgent.id]: true }));
      setTestRequest("");
      setNotice("");
      onChanged();
    } catch (error) {
      refuse(error, "The run could not be started.");
    } finally { setBusy(false); }
  };

  const retryRun = async (task: AgentTask) => {
    setBusy(true); setNotice(""); setDialogError("");
    try {
      const retried = await apiRequest<AgentTask>(`/assistant/tasks/${task.id}/retry`, { method: "POST" }, csrfToken);
      // Follow the replacement run in place rather than sending the reader off
      // to find it.
      if (retried?.id && activeRunId) {
        setActiveRunId(retried.id);
        /*
         * "Try again" has to try again. The retry lands in needs_approval, so
         * the button reset the run to "Waiting for your approval" and sat
         * there - people waited for a run that was never going to start. This
         * is the same description and the same agent version the operator
         * approved seconds ago, so their intent already covers it; the
         * approval gate still stands for every run they have not seen.
         */
        await transitionRun(retried, "ready");
      } else {
        setNotice("A fresh custom-agent run is ready for review in this agent's run history.");
      }
      onChanged();
    } catch (error) {
      refuse(error, "The custom-agent run could not be retried.");
    } finally { setBusy(false); }
  };
  const escalateRun = async (task: AgentTask) => {
    setBusy(true); setNotice(""); setDialogError("");
    try {
      // Escalation is a FRESH run of the SAME task on the more capable GPU model,
      // not a retry: the common escalation case is a delivered-but-thin NPU run
      // in state "completed", which retry() rejects ("Only failed, cancelled, or
      // blocked tasks can be retried"). Creating a new approval-gated task with
      // the original description and use_capable_model works for every state the
      // action is offered on. Same read-only envelope and role as any create.
      const created = await apiRequest<AgentTask>(
        "/assistant/tasks",
        {
          method: "POST",
          body: JSON.stringify({
            title: task.title,
            description: task.description,
            profile: task.profile,
            ...(typeof task.profile_version === "number" && task.profile_version > 0
              ? { profile_version: task.profile_version } : {}),
            approval_required: true,
            use_capable_model: true,
            idempotency_key: crypto.randomUUID(),
          }),
        },
        csrfToken,
      );
      // Follow the capable run in place; the operator just chose to run it, so
      // start it - the approval gate still stands for every run they did not.
      if (created?.id && activeRunId) {
        setActiveRunId(created.id);
        await transitionRun(created, "ready");
      } else {
        setNotice("A capable-model run of this task is ready for review in this agent's run history.");
      }
      setOpenRuns((current) => ({ ...current, [task.profile]: true }));
      onChanged();
    } catch (error) {
      refuse(error, "The capable re-run could not be started.");
    } finally { setBusy(false); }
  };
  const createAutomation = async (event: FormEvent) => {
    event.preventDefault();
    const form = automationForm;
    if (!automationAgent || !form.name.trim() || !form.prompt.trim()) return;
    setBusy(true); setNotice(""); setDialogError("");
    try {
      if (form.kind === "schedule") {
        await apiRequest("/assistant/automations", { method: "POST", body: JSON.stringify({ name: form.name.trim(), prompt: form.prompt.trim(), profile: automationAgent.id, schedule: form.schedule.trim() }) }, csrfToken);
      } else {
        await apiRequest("/assistant/triggers", { method: "POST", body: JSON.stringify({ name: form.name.trim(), prompt: form.prompt.trim(), profile: automationAgent.id, source: form.source, operator: ">=", threshold: Number(form.threshold), cooldown_seconds: 1800 }) }, csrfToken);
      }
      setAutomationAgent(null);
      setNotice(form.kind === "schedule" ? "Custom-agent schedule created with this agent version pinned." : "Custom-agent hardware trigger created with this agent version pinned.");
      onChanged();
    } catch (error) {
      refuse(error, "The automation could not be created.");
    } finally { setBusy(false); }
  };

  const connectorReadyForTest = (connector: AgentConnector) => {
    const saved = custom.find((profile) => profile.id === draft?.id)?.connectors
      ?.find((item) => item.id === connector.id);
    return Boolean(saved && JSON.stringify(saved) === JSON.stringify(connector));
  };
  const testConnector = async (connector: AgentConnector) => {
    if (!draft?.id || !connectorReadyForTest(connector)) {
      refuse(null, "Save this agent version before testing new or changed integration settings.");
      return;
    }
    setBusy(true); setNotice(""); setDialogError(""); setDialogStatus(null);
    try {
      await apiRequest(`/assistant/custom-agents/${draft.id}/connectors/${connector.id}/test`, { method: "POST", body: JSON.stringify({ operation_id: connector.operations[0]?.id ?? "", arguments: {} }) }, csrfToken);
      setDialogStatus({ dialog: "editor", text: `${connector.name || "Connector"} connection test passed.` });
    } catch (error) {
      refuse(error, "The connector test failed.");
    } finally { setBusy(false); }
  };
  const loadConnectorAudit = async (agentId: string) => {
    setBusy(true); setNotice(""); setDialogError("");
    try {
      const result = await apiRequest<{ events?: Array<Record<string, unknown>> } | Array<Record<string, unknown>>>(`/assistant/custom-agents/${agentId}/connector-audit?limit=100`);
      setConnectorAudit((current) => ({ ...current, [agentId]: Array.isArray(result) ? result : result.events ?? [] }));
    } catch (error) {
      refuse(error, "Connector audit could not be loaded.");
    } finally { setBusy(false); }
  };

  // Only a run whose agent still exists. Deleting an agent retains its task
  // history, so without this the workshop kept showing a "<name> test run:
  // completed — …" banner for an agent that is no longer on the page.
  const latestTerminalRun = tasks.find((task) =>
    task.profile.startsWith("custom_")
    && ["completed", "failed", "cancelled", "blocked"].includes(task.state)
    && custom.some((agent) => agent.id === task.profile),
  );
  // Resolved from the live task list, so the dialog follows the run through
  // approval, execution and result without the user leaving it.
  const activeRun = activeRunId ? tasks.find((task) => task.id === activeRunId) ?? null : null;
  const closeRun = () => { setTestAgent(null); setActiveRunId(""); setTestRequest(""); setTestError(""); };
  const openRun = (agent: AgentProfile) => { setTestAgent(agent); setTestRequest(agent.description ?? ""); setActiveRunId(""); };
  const unscheduledRecurring = custom.filter((item) =>
    RECURRING.test(item.description ?? "")
    && !automations.some((automation) => automation.profile === item.id && automationIsActive(automation)),
  );
  const archivedCount = custom.filter((item) => !item.enabled).length;
  // Outside the Assistant bar (Cluster) New agent sits on the page, so it says
  // why it is off right beside it; in the bar, the model banner below says so.
  const newAgentBlocked = authoringReady || inAssistantBar ? undefined
    : isInference ? "A serving cluster model is needed first" : "An Assistant model is needed first";
  const modelUnread = !isInference && !modelStatusResolved;
  const draftBlocked = modelUnread ? (profilesReadError ? "The Assistant model was not read" : "Checking the Assistant model")
    : !authoringReady ? "An Assistant model is needed first"
    : !builderPrompt.trim() ? "Describe the agent first" : undefined;

  const barItems = (
    <AssistantBarActions>
      {inAssistantBar && barStatus}
      <Button disabled={!authoringReady} disabledReason={newAgentBlocked} onClick={openCreate} variant="primary"><Icon className="ar-btn-icon" name="add" size={16} />New agent</Button>
    </AssistantBarActions>
  );

  const agentList = custom.length > 0 ? (
    <section aria-labelledby="custom-agent-list-title" className="card ui-card ar-card ar-agents">
      <header className="ar-card__head">
        <h2 id="custom-agent-list-title">{isInference ? "Inference agents" : "Your agents"}</h2>
        <span className="ar-meta">{custom.length} agent{custom.length === 1 ? "" : "s"}{archivedCount ? ` · ${archivedCount} archived` : ""}</span>
      </header>
      <div className="ar-rows">
        {custom.map((item) => (
          <CustomAgentCard
            agent={item}
            appAccessOpen={appAccessAgentId === item.id}
            busy={busy}
            key={item.id}
            modelReady={modelAnswering}
            onApprove={(task) => void transitionRun(task, "ready")}
            onAutomate={() => {
              setAutomationAgent(item);
              setAutomationForm({ ...EMPTY_AUTOMATION, name: `${item.name} schedule`, prompt: item.description });
            }}
            onCancelRun={(task) => void transitionRun(task, "cancelled")}
            onDelete={() => setDeleting(item)}
            onEdit={() => openEdit(item)}
            onRetry={(task) => void retryRun(task)}
            onRun={() => openRun(item)}
            onToggleAppAccess={() => setAppAccessAgentId((current) => current === item.id ? "" : item.id)}
            onToggleEnabled={() => void setEnabled(item, !item.enabled)}
            onToggleRuns={(open) => setOpenRuns((current) => ({ ...current, [item.id]: open }))}
            onToggleVersions={() => void loadVersions(item)}
            revisions={revisions[item.id]}
            runBlockedReason={runBlockedReason}
            runs={tasks.filter((task) => task.profile === item.id)}
            runsOpen={openRuns[item.id] ?? false}
            runtimeManaged={!isInference}
            scheduleCount={automations.filter((automation) => automation.profile === item.id && automationIsActive(automation)).length}
            triggerCount={triggers.filter((trigger) => trigger.profile === item.id && automationIsActive(trigger)).length}
          />
        ))}
      </div>
    </section>
  ) : !profilesRead ? (
    // Not read is not none: "No custom agents yet" before the list arrived was a false empty (VD-200 review).
    <div className="card ui-card ar-card">
      <EmptyState
        icon={<Icon name="assistant" size={18} />}
        text={profilesReadError || "Reading the agents on this appliance…"}
        title={profilesReadError ? "Agents not read" : "Reading agents"}
      />
    </div>
  ) : (
    <div className="card ui-card ar-card">
      <EmptyState
        icon={<Icon name="assistant" size={18} />}
        text={isInference
          ? "Start from New agent. Nothing it may read is granted until you choose it."
          : "Describe the agent you need above, or start from New agent. Nothing it may do is granted until you choose it."}
        title={isInference ? "No inference agents yet" : "No custom agents yet"}
      />
    </div>
  );

  return (
    <section className="custom-agent-manager ar-manager">
      {inAssistantBar ? barItems : (
        // No Assistant strip (Cluster > Agents & tools): the workshop carries
        // its own heading, and New agent sits beside it.
        <div className="ar-manager__head">
          {isInference ? (
            <div>
              <span className="as-label">Cluster inference agents</span>
              <h2>Build a read-only inference agent</h2>
              <p>Define an agent that a cluster model backs, then deploy it from Deployments. It runs on the fleet, not the Assistant, so it takes no acting permissions and no schedules.</p>
            </div>
          ) : <span />}
          {barItems}
        </div>
      )}
      {/* Before the readiness read lands, or when it failed, nothing is claimed about the model (VD-200). */}
      {modelUnread ? (
        <Notice severity={profilesReadError ? "warning" : "info"}>
          {profilesReadError
            ? "The Assistant model could not be read, so New agent and runs stay off until it is."
            : "Checking the Assistant model…"}
        </Notice>
      ) : (
      <Notice severity={authoringReady && (isInference || modelAnswering) ? "success" : "warning"}>
        {isInference ? (clusterModelNote || (clusterModelServing ? "A cluster model is serving. Author a cluster agent here and deploy it from Cluster > Deployments." : "Serve a cluster model before creating cluster agents; a cluster agent runs on it."))
          : modelAnswering ? `Selected model: ${modelLabel}. Custom agents use this active Assistant model.`
            : modelReady ? `Selected model: ${modelLabel} is not answering. ${modelNotAnsweringReason} You can still edit agents, but runs stay off until it answers.`
              : clusterModelServing ? "No Assistant model is connected. A cluster model is serving, but agents that run on it are built in Cluster > Agents & tools > Inference agents, not here."
                : "Connect an Assistant model before creating or running custom agents."}
      </Notice>
      )}
      {notice && <Notice severity={noticeRefused ? "danger" : "info"}>{notice}</Notice>}
      <div className={appAccessAgent ? "as-split ar-split" : "ar-single"}>
        <div className="ar-column">
          {!isInference && (
            <section aria-labelledby="custom-agent-workshop-title" className="card ui-card ar-card ar-workshop">
              <div className="ar-card__pad">
                <div>
                  <span className="as-label">Agent workshop</span>
                  <h2 className="ar-card__title" id="custom-agent-workshop-title">Build and manage custom agents</h2>
                  <p className="ar-step__hint">Define a general-purpose agent for your own domain. Choose its model context and exact data or action grants; every edit creates a new version.</p>
                </div>
                <Textarea label="Describe the agent you need" maxLength={2000} onChange={(event) => setBuilderPrompt(event.target.value)} placeholder="Example: Compare my selected research notes and propose a cited briefing for approval." rows={2} value={builderPrompt} />
                <div className="ar-actions">
                  <Button disabled={busy} disabledReason={busy ? undefined : draftBlocked} onClick={() => void draftWithAssistant()}>Create a starting draft</Button>
                </div>
              </div>
            </section>
          )}
          <UnscheduledAgentsNotice agents={unscheduledRecurring} />
          {latestTerminalRun && !topDialog && (
            <Notice severity={latestTerminalRun.state === "failed" || latestTerminalRun.result.outcome === "needs_input" ? "warning" : "info"}>
              <span className="ar-notice-row">
                <span><strong>{latestTerminalRun.title}: {latestTerminalRun.result.outcome === "needs_input" ? "needs input" : latestTerminalRun.state}</strong> — {latestTerminalRun.result.summary || latestTerminalRun.error || "Run finished."}</span>
                {onViewRun && <Button onClick={() => onViewRun(latestTerminalRun)}>View agent run history</Button>}
              </span>
            </Notice>
          )}
          {agentList}
        </div>
        {appAccessAgent && (
          <aside aria-label={`App access for ${appAccessAgent.name}`} className="card ui-card ar-card ar-app-access">
            <IntegrationCapabilitiesContainer
              agent={appAccessAgent}
              csrfToken={csrfToken}
              onChanged={onChanged}
              onClose={() => setAppAccessAgentId("")}
            />
          </aside>
        )}
      </div>
      {draft && (
        <CustomAgentEditor
          authoringReady={authoringReady}
          busy={busy}
          collections={collections}
          connectorAudit={draft.id ? connectorAudit[draft.id] : undefined}
          connectorReadyForTest={connectorReadyForTest}
          draft={draft}
          error={dialogError}
          isInference={isInference}
          onClose={() => setDraft(null)}
          onLoadAudit={() => { if (draft.id) void loadConnectorAudit(draft.id); }}
          onSave={(event) => void save(event)}
          onTestConnector={(connector) => void testConnector(connector)}
          setDraft={setDraft}
          status={dialogStatus?.dialog === "editor" ? dialogStatus.text : ""}
        />
      )}
      {activationAgent && (
        <ActivationDialog
          agent={activationAgent}
          busy={busy}
          canTest={isInference ? null : modelAnswering}
          error={dialogError}
          onActivate={() => void activateAgent()}
          onClose={() => setActivationAgent(null)}
          onTest={() => openRun(activationAgent)}
        />
      )}
      {testAgent && (
        <RunDialog
          activeRun={activeRun}
          agent={testAgent}
          busy={busy}
          error={dialogError}
          onApprove={() => { if (activeRun) void transitionRun(activeRun, "ready"); }}
          onCancelRun={() => { if (activeRun) void transitionRun(activeRun, "cancelled"); }}
          onClose={closeRun}
          onEscalate={() => { if (activeRun) void escalateRun(activeRun); }}
          onRequestChange={(value) => { setTestRequest(value); if (testError) setTestError(""); }}
          onRetry={() => { if (activeRun) void retryRun(activeRun); }}
          onRunAgain={() => { setActiveRunId(""); setTestRequest(""); }}
          onSubmit={(event) => void queueTestRun(event)}
          request={testRequest}
          requestError={testError}
        />
      )}
      {automationAgent && (
        <AutomationDialog
          agent={automationAgent}
          busy={busy}
          error={dialogError}
          form={automationForm}
          onChange={(patch) => setAutomationForm((current) => ({ ...current, ...patch }))}
          onClose={() => setAutomationAgent(null)}
          onSubmit={(event) => void createAutomation(event)}
        />
      )}
      <ConfirmDialog irreversible open={Boolean(deleting)} title="Delete custom agent?" description="The definition and its revisions will be deleted. Completed task results remain in the ledger." confirmLabel="Delete agent" busy={busy} error={dialogError} onCancel={() => setDeleting(null)} onConfirm={() => void remove()} />
    </section>
  );
}
