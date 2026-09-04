import { type FormEvent, useEffect, useState } from "react";
import { apiRequest } from "../lib/api";
import { ConfirmDialog } from "./ConfirmDialog";
import { CustomAgentCard, UnscheduledAgentsNotice, type AgentRevision } from "./CustomAgentCard";
import { Icon } from "./Icon";
import { IntegrationCapabilitiesContainer } from "./IntegrationCapabilitiesContainer";
import { ModalShell } from "./ModalShell";
import type { AgentConnector, AgentProfile, AgentTask, Automation, Trigger } from "./agentTypes";
import { AgentRunWorkspace } from "./AgentRunWorkspace";
import { Button, Input, Notice, Textarea } from "./ui";

/*
 * An agent described as recurring ("every morning...") does not get a
 * schedule from being created, and nothing said so - the one thing the user
 * asked for silently never happened. It is stated once for the list rather
 * than repeated verbatim on every card.
 */
const RECURRING = /\b(every|each|daily|hourly|weekly|nightly)\b/i;

const CAPABILITIES = [
  ["system:read", "Hardware and system facts", "Telemetry, storage, network, services, and OS facts"],
  ["cooling:read", "Cooling", "Fan and thermal state, where this machine reports it"],
  ["workloads:read", "Workloads", "Docker apps, local models, and capabilities"],
  ["jobs:read", "Jobs", "Deployment state and audited history"],
  ["assistant:read", "Assistant", "Reviewed memory and assistant status"],
  ["cluster:read", "Fleet and cluster facts", "Controller, worker, service, and placement health"],
] as const;

type Draft = {
  id?: string;
  name: string;
  description: string;
  instructions: string;
  scopes: string[];
  permissions: string[];
  read_collection_ids: string[];
  write_collection_id: string;
  web_access: { enabled: boolean; allowed_domains: string[] };
  connectors: AgentConnector[];
};

type KnowledgeCollection = {
  id: string;
  name: string;
  description: string;
  document_count: number;
};

const EMPTY: Draft = {
  name: "",
  description: "",
  instructions: "",
  scopes: [],
  permissions: [],
  read_collection_ids: [],
  write_collection_id: "",
  web_access: { enabled: false, allowed_domains: [] },
  connectors: [],
};

export function CustomAgentManager({
  automations = [],
  csrfToken,
  modelLabel,
  modelReady,
  onChanged,
  onViewRun,
  profiles,
  tasks,
  triggers = [],
}: {
  automations?: Automation[];
  csrfToken: string;
  modelLabel: string;
  modelReady: boolean;
  onChanged: () => void;
  onViewRun?: (task: AgentTask) => void;
  profiles: AgentProfile[];
  tasks: AgentTask[];
  triggers?: Trigger[];
}) {
  const [draft, setDraft] = useState<Draft | null>(null);
  const [activationAgent, setActivationAgent] = useState<AgentProfile | null>(null);
  const [busy, setBusy] = useState(false);
  const [notice, setNotice] = useState("");
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
  // The run currently being watched inside the test dialog.
  const [activeRunId, setActiveRunId] = useState("");
  const [automationAgent, setAutomationAgent] = useState<AgentProfile | null>(null);
  const [automationKind, setAutomationKind] = useState<"schedule" | "trigger">("schedule");
  const [automationName, setAutomationName] = useState("");
  const [automationPrompt, setAutomationPrompt] = useState("");
  const [scheduleText, setScheduleText] = useState("every 6 hours");
  const [triggerSource, setTriggerSource] = useState("cpu_temperature");
  const [triggerThreshold, setTriggerThreshold] = useState("80");
  const [connectorAudit, setConnectorAudit] = useState<Record<string, Array<Record<string, unknown>>>>({});
  const custom = profiles.filter((item) => item.custom);
  /*
   * App access is opened deliberately, never by default.
   *
   * `custom.find(...) ?? custom[0]` made this truthy for anybody who owned a
   * single agent, so a four-step credential wizard — about 1,400px and twenty
   * controls — was permanently appended to this tab and counted as first paint
   * for a job nobody had started. An empty id means the panel is closed.
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
    setDraft({ ...EMPTY, scopes: [...EMPTY.scopes] });
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
  const toggleScope = (scope: string) => setDraft((current) => current ? {
    ...current,
    scopes: current.scopes.includes(scope)
      ? current.scopes.filter((item) => item !== scope)
      : [...current.scopes, scope],
  } : current);
  const togglePermission = (permission: string) => setDraft((current) => current ? {
    ...current,
    permissions: current.permissions.includes(permission)
      ? current.permissions.filter((item) => item !== permission)
      : [...current.permissions, permission],
    ...(permission === "knowledge:read" && current.permissions.includes(permission)
      ? { read_collection_ids: [] } : {}),
    ...(permission === "knowledge:write" && current.permissions.includes(permission)
      ? { write_collection_id: "" } : {}),
  } : current);
  const toggleReadCollection = (collectionId: string) => setDraft((current) => current ? {
    ...current,
    read_collection_ids: current.read_collection_ids.includes(collectionId)
      ? current.read_collection_ids.filter((item) => item !== collectionId)
      : [...current.read_collection_ids, collectionId],
  } : current);

  const draftWithAssistant = async () => {
    setBusy(true); setNotice("");
    try {
      const next = await apiRequest<Draft>(
        "/assistant/custom-agents/draft",
        { method: "POST", body: JSON.stringify({ request: builderPrompt }) },
        csrfToken,
      );
      setDraft({ ...EMPTY, ...next, name: next.name.replace(/\s+specialist$/i, " agent"), web_access: next.web_access ?? { enabled: false, allowed_domains: [] }, connectors: next.connectors ?? [] });
      setNotice("Draft prepared. Review every permission before creating the agent.");
    } catch (error) {
      setNotice(error instanceof Error ? error.message : "The assistant could not draft this agent.");
    } finally { setBusy(false); }
  };

  const save = async (event: FormEvent) => {
    event.preventDefault();
    if (!draft) return;
    setBusy(true); setNotice("");
    try {
      const saved = await apiRequest<AgentProfile>(
        draft.id ? `/assistant/custom-agents/${draft.id}` : "/assistant/custom-agents",
        { method: draft.id ? "PATCH" : "POST", body: JSON.stringify(draft) },
        csrfToken,
      );
      setDraft(null);
      setActivationAgent(saved?.id ? saved : null);
      setNotice(draft.id ? "New agent version saved." : "Custom agent created.");
      onChanged();
    } catch (error) {
      setNotice(error instanceof Error ? error.message : "The agent could not be saved.");
    } finally { setBusy(false); }
  };

  const activateAgent = async () => {
    if (!activationAgent) return;
    setBusy(true); setNotice("");
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
      setNotice(error instanceof Error ? error.message : "The agent could not be activated.");
    } finally { setBusy(false); }
  };

  const setEnabled = async (profile: AgentProfile, enabled: boolean) => {
    setBusy(true); setNotice("");
    try {
      await apiRequest(
        `/assistant/custom-agents/${profile.id}`,
        { method: "PATCH", body: JSON.stringify({ enabled }) },
        csrfToken,
      );
      setNotice(enabled ? "Agent restored for new runs." : "Agent archived. Existing run history was retained.");
      onChanged();
    } catch (error) {
      setNotice(error instanceof Error ? error.message : "The agent could not be updated.");
    } finally { setBusy(false); }
  };

  const remove = async () => {
    if (!deleting) return;
    setBusy(true); setNotice("");
    try {
      await apiRequest(`/assistant/custom-agents/${deleting.id}`, { method: "DELETE" }, csrfToken);
      setDeleting(null);
      setNotice("Custom agent deleted. Existing task history was retained.");
      onChanged();
    } catch (error) {
      setNotice(error instanceof Error ? error.message : "The agent could not be deleted.");
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
    setBusy(true); setNotice("");
    try {
      const result = await apiRequest<AgentRevision[]>(`/assistant/custom-agents/${profile.id}/revisions`);
      setRevisions((current) => ({ ...current, [profile.id]: result }));
    } catch (error) {
      setNotice(error instanceof Error ? error.message : "Agent versions could not be loaded.");
    } finally { setBusy(false); }
  };

  const queueTestRun = async (event: FormEvent) => {
    event.preventDefault();
    if (!testAgent) return;
    if (!testRequest.trim()) { setTestError("Enter a task to run."); return; }
    setTestError("");
    setBusy(true); setNotice("");
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
      setNotice(error instanceof Error ? error.message : "The run could not be started.");
    } finally { setBusy(false); }
  };

  const transitionRun = async (task: AgentTask, state: "ready" | "cancelled") => {
    setBusy(true); setNotice("");
    try {
      await apiRequest(
        `/assistant/tasks/${task.id}`,
        { method: "PATCH", body: JSON.stringify({ state }) },
        csrfToken,
      );
      setNotice(state === "ready" ? "Custom-agent run approved." : "Custom-agent run cancelled.");
      onChanged();
    } catch (error) {
      setNotice(error instanceof Error ? error.message : "The custom-agent run could not be updated.");
    } finally { setBusy(false); }
  };

  const retryRun = async (task: AgentTask) => {
    setBusy(true); setNotice("");
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
        setNotice("A fresh custom-agent run is ready for review below.");
      }
      onChanged();
    } catch (error) {
      setNotice(error instanceof Error ? error.message : "The custom-agent run could not be retried.");
    } finally { setBusy(false); }
  };
  const escalateRun = async (task: AgentTask) => {
    setBusy(true); setNotice("");
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
        setNotice("A capable-model run of this task is ready for review below.");
      }
      setOpenRuns((current) => ({ ...current, [task.profile]: true }));
      onChanged();
    } catch (error) {
      setNotice(error instanceof Error ? error.message : "The capable re-run could not be started.");
    } finally { setBusy(false); }
  };
  const createAutomation = async (event: FormEvent) => {
    event.preventDefault();
    if (!automationAgent || !automationName.trim() || !automationPrompt.trim()) return;
    setBusy(true); setNotice("");
    try {
      if (automationKind === "schedule") {
        await apiRequest("/assistant/automations", { method: "POST", body: JSON.stringify({ name: automationName.trim(), prompt: automationPrompt.trim(), profile: automationAgent.id, schedule: scheduleText.trim() }) }, csrfToken);
      } else {
        await apiRequest("/assistant/triggers", { method: "POST", body: JSON.stringify({ name: automationName.trim(), prompt: automationPrompt.trim(), profile: automationAgent.id, source: triggerSource, operator: ">=", threshold: Number(triggerThreshold), cooldown_seconds: 1800 }) }, csrfToken);
      }
      setAutomationAgent(null);
      setNotice(automationKind === "schedule" ? "Custom-agent schedule created with this agent version pinned." : "Custom-agent hardware trigger created with this agent version pinned.");
      onChanged();
    } catch (error) {
      setNotice(error instanceof Error ? error.message : "The automation could not be created.");
    } finally { setBusy(false); }
  };

  const addConnector = () => setDraft((current) => current ? {
    ...current,
    connectors: [...current.connectors, {
      id: `connector_${crypto.randomUUID().replaceAll("-", "")}`,
      name: "",
      base_origin: "https://",
      credential_ref: "",
      auth: "none",
      operations: [],
    }],
  } : current);
  const updateConnector = (index: number, patch: Partial<AgentConnector>) => setDraft((current) => current ? {
    ...current,
    connectors: current.connectors.map((connector, itemIndex) => itemIndex === index ? { ...connector, ...patch } : connector),
  } : current);
  const addOperation = (connectorIndex: number) => setDraft((current) => current ? {
    ...current,
    connectors: current.connectors.map((connector, itemIndex) => itemIndex === connectorIndex ? {
      ...connector,
      operations: [...connector.operations, {
        id: `operation_${crypto.randomUUID().replaceAll("-", "")}`,
        description: "",
        method: "GET",
        path: "/",
        input_location: "query",
        request_schema: { type: "object", properties: {}, additionalProperties: false },
        response_schema: { type: "object" },
        timeout_seconds: 15,
        max_response_bytes: 262144,
        rate_limit_per_minute: 30,
        approval: "not_required",
      }],
    } : connector),
  } : current);
  const updateOperation = (connectorIndex: number, operationIndex: number, patch: Partial<AgentConnector["operations"][number]>) => setDraft((current) => current ? {
    ...current,
    connectors: current.connectors.map((connector, itemIndex) => itemIndex === connectorIndex ? {
      ...connector,
      operations: connector.operations.map((operation, opIndex) => opIndex === operationIndex ? { ...operation, ...patch } : operation),
    } : connector),
  } : current);
  const testConnector = async (connector: AgentConnector) => {
    if (!draft?.id || !connectorReadyForTest(connector)) {
      setNotice("Save this agent version before testing new or changed integration settings.");
      return;
    }
    setBusy(true); setNotice("");
    try {
      await apiRequest(`/assistant/custom-agents/${draft.id}/connectors/${connector.id}/test`, { method: "POST", body: JSON.stringify({ operation_id: connector.operations[0]?.id ?? "", arguments: {} }) }, csrfToken);
      setNotice(`${connector.name || "Connector"} connection test passed.`);
    } catch (error) {
      setNotice(error instanceof Error ? error.message : "The connector test failed.");
    } finally { setBusy(false); }
  };
  const loadConnectorAudit = async (profile: AgentProfile) => {
    setBusy(true); setNotice("");
    try {
      const result = await apiRequest<{ events?: Array<Record<string, unknown>> } | Array<Record<string, unknown>>>(`/assistant/custom-agents/${profile.id}/connector-audit?limit=100`);
      setConnectorAudit((current) => ({ ...current, [profile.id]: Array.isArray(result) ? result : result.events ?? [] }));
    } catch (error) {
      setNotice(error instanceof Error ? error.message : "Connector audit could not be loaded.");
    } finally { setBusy(false); }
  };

  const invalidWrite = Boolean(
    draft?.permissions.includes("knowledge:write") && !draft.write_collection_id,
  );
  const invalidConnector = Boolean(draft?.connectors.some((connector) =>
    !connector.name.trim()
    || !connector.base_origin.startsWith("https://")
    || (connector.auth !== "none" && !/^cred_[a-f0-9]{24}$/.test(connector.credential_ref))
    || connector.operations.length === 0
    || connector.operations.some((operation) => !operation.description.trim() || !operation.path.startsWith("/")),
  ));
  const connectorReadyForTest = (connector: AgentConnector) => {
    const saved = custom.find((profile) => profile.id === draft?.id)?.connectors
      ?.find((item) => item.id === connector.id);
    return Boolean(saved && JSON.stringify(saved) === JSON.stringify(connector));
  };
  // Only a run whose agent still exists. Deleting an agent retains its task
  // history, so without this the workshop kept showing a "<name> test run:
  // completed — …" banner for an agent that is no longer on the page.
  const latestTerminalRun = tasks.find((task) =>
    task.profile.startsWith("custom_")
    && ["completed", "failed", "cancelled", "blocked"].includes(task.state)
    && custom.some((agent) => agent.id === task.profile),
  );
  // The draft carries research:read whenever the described job needs outside
  // information, so the wizard can state that this step is required instead of
  // leaving the user to discover it from a failed run.
  const researchRequired = Boolean(draft?.scopes.includes("research:read"));
  // Resolved from the live task list, so the dialog follows the run through
  // approval, execution and result without the user leaving it.
  const activeRun = activeRunId ? tasks.find((task) => task.id === activeRunId) ?? null : null;
  const closeRun = () => { setTestAgent(null); setActiveRunId(""); setTestRequest(""); setTestError(""); };
  const unscheduledRecurring = custom.filter((item) =>
    RECURRING.test(item.description ?? "")
    && !automations.some((automation) => automation.profile === item.id),
  );
  return (
    <section className="custom-agent-manager">
      <header>
        <div><span className="page-eyebrow">Agent workshop</span><h2>Build and manage custom agents</h2><p>Define a general-purpose agent for your own domain. Choose its model context and exact data or action grants; every edit creates a new version.</p></div>
        <Button variant="primary" disabled={!modelReady} onClick={openCreate}><Icon name="cpu" />New agent</Button>
      </header>
      <Notice severity={modelReady ? "success" : "warning"}><Icon name="cpu" />{modelReady ? `Selected model: ${modelLabel}. Custom agents use this active Assistant model.` : "Select a local or connected model before creating or running custom agents."}</Notice>
      {notice && <Notice severity="info">{notice}</Notice>}
      <div className="custom-agent-builder">
        <Textarea label="Describe the agent you need" maxLength={2000} onChange={(event) => setBuilderPrompt(event.target.value)} placeholder="Example: Compare my selected research notes and propose a cited briefing for approval." rows={4} value={builderPrompt} />
        <Button variant="quiet" disabled={!modelReady || busy || !builderPrompt.trim()} onClick={() => void draftWithAssistant()}>Create a starting draft</Button>
      </div>
      {latestTerminalRun && <Notice severity={latestTerminalRun.state === "failed" || latestTerminalRun.result.outcome === "needs_input" ? "warning" : "info"}><span><strong>{latestTerminalRun.title}: {latestTerminalRun.result.outcome === "needs_input" ? "needs input" : latestTerminalRun.state}</strong> — {latestTerminalRun.result.summary || latestTerminalRun.error || "Run finished."}</span>{onViewRun && <Button onClick={() => onViewRun(latestTerminalRun)} type="button" variant="quiet">View agent run history</Button>}</Notice>}
      {draft && (
        <ModalShell labelledBy="custom-agent-editor-title" onClose={() => !busy && setDraft(null)}>
        <form className="custom-agent-form" onSubmit={(event) => void save(event)}>
         <div className="panel-heading"><div><span className="page-eyebrow">Guided agent setup</span><h3 id="custom-agent-editor-title">{draft.id ? "Edit custom agent" : "Create custom agent"}</h3><p>Define its job, then grant only the information and actions it needs. Nothing can run until you save this version.</p></div><Button variant="quiet" aria-label="Close agent editor" disabled={busy} onClick={() => setDraft(null)}>Close</Button></div>
          {/*
            * One numbering scheme, and it is the one the body actually follows.
            *
            * A four-step strip across the top numbered the same form 1-4 while
            * the body sections numbered themselves 1-4 independently, and a
            * "Step 3 · Review" panel then appeared after body section 4. The
            * strip only ever set its own highlight - it never filtered or
            * advanced anything - so removing it removes a second, contradictory
            * count rather than removing a way through the form.
            */}
          <div className="custom-agent-step custom-agent-step--primary" data-route="/assistant/custom-agents/setup"><span>1</span><div><strong>Define the job</strong><small>Name the outcome this agent owns and how it should work.</small></div></div>
          <Input label="Name" maxLength={100} onChange={(event) => setDraft({ ...draft, name: event.target.value })} value={draft.name} />
          <Input label="Purpose" maxLength={400} onChange={(event) => setDraft({ ...draft, description: event.target.value })} value={draft.description} />
          <Textarea label="Operating instructions" maxLength={6000} onChange={(event) => setDraft({ ...draft, instructions: event.target.value })} rows={5} value={draft.instructions} />
          <details className="custom-agent-step-group">
          <summary><span>2</span><div><strong>Choose information access</strong><small>Optional Vaelor facts, saved knowledge, and deployment proposals.</small></div></summary>
          <fieldset><legend>Optional Vaelor context</legend><p>Leave all of these off for a general research or documentation agent. Grant only the appliance facts this agent actually needs.</p><div className="custom-agent-scopes">{CAPABILITIES.map(([id, name, description]) => <label key={id}><input className="custom-agent-control" checked={draft.scopes.includes(id)} onChange={() => toggleScope(id)} type="checkbox" /><span><strong>{name}</strong><small>{description}</small></span></label>)}</div></fieldset>
          <fieldset className="custom-agent-permissions">
            <legend>Knowledge and action permissions</legend>
            <p>No shell, raw credentials, arbitrary filesystem access, or unrestricted URLs are exposed.</p>
            <label><input className="custom-agent-control" checked={draft.permissions.includes("knowledge:read")} onChange={() => togglePermission("knowledge:read")} type="checkbox" /><span><strong>Read selected knowledge</strong><small>Retrieve cited chunks only from named collections.</small></span></label>
            {draft.permissions.includes("knowledge:read") && <div className="custom-agent-collections" role="group" aria-label="Readable knowledge collections">{collections.length ? collections.map((collection) => <label key={collection.id}><input className="custom-agent-control" checked={draft.read_collection_ids.includes(collection.id)} onChange={() => toggleReadCollection(collection.id)} type="checkbox" /><span><strong>{collection.name}</strong><small>{collection.document_count} document{collection.document_count === 1 ? "" : "s"}</small></span></label>) : <small>Create a collection in AI Chat before granting knowledge access.</small>}</div>}
            <label><input className="custom-agent-control" checked={draft.permissions.includes("knowledge:write")} onChange={() => togglePermission("knowledge:write")} type="checkbox" /><span><strong>Propose a knowledge document</strong><small>The exact content is held for operator approval before it is stored.</small></span></label>
            {draft.permissions.includes("knowledge:write") && <label className="custom-agent-write-target">Approved destination<select className="custom-agent-control" onChange={(event) => setDraft({ ...draft, write_collection_id: event.target.value })} value={draft.write_collection_id}><option value="">Choose a collection</option>{collections.map((collection) => <option key={collection.id} value={collection.id}>{collection.name}</option>)}</select></label>}
            <label><input className="custom-agent-control" checked={draft.permissions.includes("workloads:propose")} onChange={() => togglePermission("workloads:propose")} type="checkbox" /><span><strong>Propose workload changes</strong><small>May prepare a bounded deployment plan but cannot execute it during an agent run.</small></span></label>
          </fieldset>
          </details>
          {/*
            * Labelling this "Optional" was how a beginner built an agent that
            * could never work: an agent asked for scores or news has no way to
            * answer without it, and the failure only showed up after creation.
            */}
          <details className="custom-agent-step-group" open={researchRequired}>
          <summary><span>3</span><div><strong>Allow public research</strong><small>{researchRequired ? "Required — this agent needs information from the internet." : "Optional, guarded web access with domain limits."}</small></div></summary>
          <fieldset className="custom-agent-permissions">
            <legend>Internet permission</legend>
            <p>Internet access always goes through Vaelor's guarded research broker. The model never receives a raw network connection.</p>
            <label><input className="custom-agent-control" checked={draft.web_access.enabled} onChange={(event) => setDraft({ ...draft, web_access: { ...draft.web_access, enabled: event.target.checked }, scopes: event.target.checked ? [...new Set([...draft.scopes, "research:read"])] : draft.scopes.filter((scope) => scope !== "research:read") })} type="checkbox" /><span><strong>Allow guarded public research</strong><small>With no domains listed, the agent may search but cannot open arbitrary results.</small></span></label>
            {draft.web_access.enabled && <label>Allowed HTTPS domains (optional)<textarea className="custom-agent-control" maxLength={2000} onChange={(event) => setDraft({ ...draft, web_access: { enabled: true, allowed_domains: event.target.value.split(/[\s,]+/).map((item) => item.trim().toLowerCase()).filter(Boolean) } })} placeholder="example.com&#10;docs.example.org" rows={3} value={draft.web_access.allowed_domains.join("\n")} /><small>List only the documentation sites this agent may fetch. Search remains available when this is empty.</small></label>}
          </fieldset>
          </details>
          <details className="custom-agent-step-group">
          <summary><span>4</span><div><strong>Connect external services</strong><small>Optional, fixed API actions using brokered credentials.</small></div></summary>
          <fieldset className="custom-agent-integrations">
            <div className="panel-heading"><div><legend>Integrations and API grants</legend><p>Grant named operations only. Vaelor stores a credential reference—not the secret—and never gives the model an unrestricted URL.</p></div><div>{draft.id && draft.connectors.length > 0 && <Button variant="quiet" disabled={busy} onClick={() => void loadConnectorAudit({ id: draft.id!, name: draft.name, description: draft.description, scopes: draft.scopes, custom: true })} type="button">View API audit</Button>}<Button variant="quiet" onClick={addConnector} type="button">Add integration</Button></div></div>
            {draft.id && connectorAudit[draft.id] && <div className="custom-agent-connector-audit"><strong>Recent integration activity</strong>{connectorAudit[draft.id].length ? <ul>{connectorAudit[draft.id].slice(0, 10).map((event, index) => <li key={index}>{String(event.operation_id ?? event.action ?? "connector event")} · {String(event.status ?? event.result ?? "recorded")}</li>)}</ul> : <p>No integration calls recorded.</p>}</div>}
            {draft.connectors.length === 0 && <div className="empty-state"><Icon name="shield" /><h4>No third-party API access</h4><p>This agent cannot call an external service unless you add and save a bounded integration.</p></div>}
            {draft.connectors.map((connector, connectorIndex) => <article className="custom-agent-connector" key={connector.id}><div className="panel-heading"><div><small>Connector {connectorIndex + 1}</small><h4>{connector.name || "Unnamed integration"}</h4></div><div>{draft.id && <Button variant="quiet" disabled={busy || !connectorReadyForTest(connector)} onClick={() => void testConnector(connector)} type="button">{connectorReadyForTest(connector) ? "Test connection" : "Save before testing"}</Button>}<Button variant="quiet" className="danger-text" onClick={() => setDraft({ ...draft, connectors: draft.connectors.filter((_, index) => index !== connectorIndex) })} type="button">{draft.id ? "Revoke" : "Remove"}</Button></div></div><div className="custom-agent-connector-grid"><label>Connector name<input className="custom-agent-control" maxLength={100} onChange={(event) => updateConnector(connectorIndex, { name: event.target.value })} placeholder="Market data API" value={connector.name} /></label><label>HTTPS base origin<input className="custom-agent-control" maxLength={300} onChange={(event) => updateConnector(connectorIndex, { base_origin: event.target.value })} placeholder="https://api.example.com" value={connector.base_origin} /></label><label>Authentication<select className="custom-agent-control" onChange={(event) => updateConnector(connectorIndex, { auth: event.target.value as AgentConnector["auth"] })} value={connector.auth}><option value="none">No credential</option><option value="bearer">Bearer token</option><option value="x-api-key">X-API-Key header</option></select></label><label>Credential broker reference<input className="custom-agent-control" disabled={connector.auth === "none"} maxLength={29} onChange={(event) => updateConnector(connectorIndex, { credential_ref: event.target.value })} placeholder="cred_… (reference only)" value={connector.credential_ref} /><small>The secret is never displayed or copied into this agent.</small></label></div><div className="custom-agent-operation-heading"><div><strong>Allowed operations</strong><small>Reads may run without approval. POST, PUT, PATCH, and DELETE always require an exact preview and approval.</small></div><Button variant="quiet" onClick={() => addOperation(connectorIndex)} type="button">Add operation</Button></div>{connector.operations.map((operation, operationIndex) => { const write = !["GET", "HEAD"].includes(operation.method); return <div className="custom-agent-operation" key={operation.id}><label>Description<input className="custom-agent-control" maxLength={200} onChange={(event) => updateOperation(connectorIndex, operationIndex, { description: event.target.value })} placeholder="Read a stock quote" value={operation.description} /></label><label>Method<select className="custom-agent-control" onChange={(event) => { const method = event.target.value as AgentConnector["operations"][number]["method"]; const isWrite = !["GET", "HEAD"].includes(method); updateOperation(connectorIndex, operationIndex, { method, input_location: isWrite ? "json" : "query", approval: isWrite ? "required" : "not_required" }); }} value={operation.method}><option>GET</option><option>HEAD</option><option>POST</option><option>PUT</option><option>PATCH</option><option>DELETE</option></select></label><label>Fixed route template<input className="custom-agent-control" maxLength={300} onChange={(event) => updateOperation(connectorIndex, operationIndex, { path: event.target.value })} placeholder="/v1/quotes/{symbol}" value={operation.path} /></label><div className={write ? "operation-risk operation-risk--write" : "operation-risk"}><strong>{write ? "State-changing" : "Read only"}</strong><small>{write ? "Exact preview and operator approval required" : "No approval; rate and response limits still apply"}</small></div><Button variant="quiet" aria-label={`Remove ${operation.description || `operation ${operationIndex + 1}`}`} className="danger-text" onClick={() => updateConnector(connectorIndex, { operations: connector.operations.filter((_, index) => index !== operationIndex) })} type="button">Remove operation</Button></div>; })}</article>)}
          </fieldset>
          </details>
          <section className="custom-agent-review" aria-labelledby="custom-agent-review-heading">
            <div><span className="page-eyebrow">Step 5 · Review</span><h4 id="custom-agent-review-heading">Confirm intent and access before saving</h4><p>{draft.name || "This agent"} will be created as a versioned definition. The runner receives only the grants listed below.</p></div>
            <dl><div><dt>Purpose</dt><dd>{draft.description || "Not supplied"}</dd></div><div><dt>Vaelor access</dt><dd>{draft.scopes.length ? draft.scopes.join(" · ") : "None"}</dd></div><div><dt>Knowledge and actions</dt><dd>{draft.permissions.length ? draft.permissions.join(" · ") : "None"}</dd></div><div><dt>Public research</dt><dd>{draft.web_access.enabled ? (draft.web_access.allowed_domains.length ? draft.web_access.allowed_domains.join(" · ") : "Guarded search") : "Off"}</dd></div><div><dt>Integrations</dt><dd>{draft.connectors.length ? draft.connectors.map((connector) => connector.name || "Unnamed integration").join(" · ") : "None"}</dd></div></dl>
          </section>
          <section className="custom-agent-policy" aria-labelledby="custom-agent-policy-heading"><h4 id="custom-agent-policy-heading">Approval and automation policy</h4><dl><div><dt>Test and manual runs</dt><dd>Start when you press Run now; pressing it is the review</dd></div><div><dt>Knowledge writes</dt><dd>Exact destination and content require separate approval</dd></div><div><dt>Workload changes</dt><dd>Proposal only; deployment has its own approval</dd></div><div><dt>Schedules and triggers</dt><dd>Administrator-only, pinned to the saved version. Their runs start without a further approval, read only, and turn any write into a proposal that needs one</dd></div></dl></section>
          {invalidConnector && <Notice severity="warning">Complete each integration's HTTPS origin, broker reference, and at least one named route before saving.</Notice>}
          <div className="dialog__actions"><Button variant="quiet" onClick={() => setDraft(null)} type="button">Cancel</Button><Button variant="primary" disabled={!modelReady || busy || !draft.name.trim() || !draft.description.trim() || !draft.instructions.trim() || invalidWrite || invalidConnector} type="submit">{busy ? "Saving…" : draft.id ? "Save new version" : "Create agent"}</Button></div>
        </form>
        </ModalShell>
      )}
      {activationAgent && (
        <ModalShell labelledBy="custom-agent-activation-title" onClose={() => !busy && setActivationAgent(null)}>
          <div className="custom-agent-activation">
            {/* A separate dialog after the form has been saved, so it carries no
                step number from the form's own count. */}
            <div className="panel-heading"><div><span className="page-eyebrow">Test & activate</span><h3 id="custom-agent-activation-title">{activationAgent.name} is ready for a safe test</h3><p>The definition is saved at version {activationAgent.version}. Prepare a reviewable run, then activate this version for future work.</p></div><Button variant="quiet" aria-label="Close activation" disabled={busy} onClick={() => setActivationAgent(null)} type="button">Close</Button></div>
            <ol className="custom-agent-activation__checks"><li><strong>Intent saved</strong><span>{activationAgent.description}</span></li><li><strong>Access pinned</strong><span>{activationAgent.scopes.length ? activationAgent.scopes.join(" · ") : "No appliance access"}</span></li><li><strong>Approval required</strong><span>Tests and state-changing connector calls wait for operator review.</span></li></ol>
            <div className="custom-agent-activation__actions"><Button disabled={busy || !modelReady} onClick={() => { setTestAgent(activationAgent); setTestRequest(activationAgent.description ?? ""); setActiveRunId(""); }} type="button" variant="quiet">Test before activation</Button><Button busy={busy} disabled={busy} onClick={() => void activateAgent()} type="button" variant="primary">Activate agent</Button></div>
          </div>
        </ModalShell>
      )}
      {custom.length > 0 && (
        <>
          <UnscheduledAgentsNotice agents={unscheduledRecurring} />
          <div className="custom-agent-list">
            {custom.map((item) => (
              <CustomAgentCard
                agent={item}
                appAccessOpen={appAccessAgentId === item.id}
                busy={busy}
                key={item.id}
                modelReady={modelReady}
                onApprove={(task) => void transitionRun(task, "ready")}
                onAutomate={() => {
                  setAutomationAgent(item);
                  setAutomationName(`${item.name} schedule`);
                  setAutomationPrompt(item.description);
                }}
                onCancelRun={(task) => void transitionRun(task, "cancelled")}
                onDelete={() => setDeleting(item)}
                onEdit={() => openEdit(item)}
                onRetry={(task) => void retryRun(task)}
                onRun={() => { setTestAgent(item); setTestRequest(item.description ?? ""); setActiveRunId(""); }}
                onToggleAppAccess={() => setAppAccessAgentId((current) => current === item.id ? "" : item.id)}
                onToggleEnabled={() => void setEnabled(item, !item.enabled)}
                onToggleRuns={(open) => setOpenRuns((current) => ({ ...current, [item.id]: open }))}
                onToggleVersions={() => void loadVersions(item)}
                revisions={revisions[item.id]}
                runs={tasks.filter((task) => task.profile === item.id)}
                runsOpen={openRuns[item.id] ?? false}
                scheduleCount={automations.filter((automation) => automation.profile === item.id).length}
                triggerCount={triggers.filter((trigger) => trigger.profile === item.id).length}
              />
            ))}
          </div>
        </>
      )}
      {appAccessAgent && (
        <section aria-label={`App access for ${appAccessAgent.name}`} className="custom-agent-app-access-panel">
          <div className="custom-agent-app-access-panel__dismiss">
            <Button onClick={() => setAppAccessAgentId("")} type="button" variant="quiet">Close app access</Button>
          </div>
          <IntegrationCapabilitiesContainer agent={appAccessAgent} csrfToken={csrfToken} onChanged={onChanged} />
        </section>
      )}
      {testAgent && (
        <ModalShell labelledBy="custom-agent-test-heading" onClose={closeRun}>
          {activeRun ? (
            <div className="custom-agent-run-dialog">
              <div className="panel-heading">
                <div>
                  <span className="page-eyebrow">Agent run</span>
                  <h3 id="custom-agent-test-heading">{testAgent.name}</h3>
                </div>
                <Button variant="quiet" aria-label="Close agent run" onClick={closeRun} type="button">Close</Button>
              </div>
              <AgentRunWorkspace
                busy={busy}
                onApprove={() => void transitionRun(activeRun, "ready")}
                onCancel={() => void transitionRun(activeRun, "cancelled")}
                onClose={closeRun}
                onEscalate={() => void escalateRun(activeRun)}
                onRetry={() => void retryRun(activeRun)}
                onRunAgain={() => { setActiveRunId(""); setTestRequest(""); }}
                task={activeRun}
              />
            </div>
          ) : (
            <form className="custom-agent-run-dialog" onSubmit={(event) => void queueTestRun(event)}>
              <div className="panel-heading"><div><span className="page-eyebrow">Approval-gated run</span><h3 id="custom-agent-test-heading">Run {testAgent.name}</h3><p>Describe one realistic task. Vaelor prepares a reviewable run and shows the result here; nothing executes until you approve it.</p></div><Button variant="quiet" aria-label="Close test run" onClick={closeRun} type="button">Close</Button></div>
              <Textarea autoFocus error={testError || undefined} label="What should it do?" maxLength={4000} onChange={(event) => { setTestRequest(event.target.value); if (testError) setTestError(""); }} placeholder="Example: Give me yesterday's score for the Yankees and tell me whether they won." rows={5} value={testRequest} />
              <div className="dialog__actions"><Button variant="quiet" onClick={closeRun} type="button">Cancel</Button><Button busy={busy} variant="primary" disabled={busy} type="submit">Run now</Button></div>
            </form>
          )}
        </ModalShell>
      )}
      {automationAgent && <ModalShell labelledBy="custom-agent-automation-heading" onClose={() => setAutomationAgent(null)}><form className="custom-agent-run-dialog" onSubmit={(event) => void createAutomation(event)}><div className="panel-heading"><div><span className="page-eyebrow">Version-pinned automation</span><h3 id="custom-agent-automation-heading">Automate {automationAgent.name}</h3><p>Choose a time schedule or a Vaelor hardware signal. Future edits create a new agent version and do not silently change this automation.</p></div><Button variant="quiet" aria-label="Close automation" onClick={() => setAutomationAgent(null)} type="button">Close</Button></div><label>Automation type<select className="custom-agent-control" onChange={(event) => setAutomationKind(event.target.value as "schedule" | "trigger")} value={automationKind}><option value="schedule">Time schedule</option><option value="trigger">Vaelor hardware trigger</option></select></label><label>Name<input className="custom-agent-control" maxLength={100} onChange={(event) => setAutomationName(event.target.value)} value={automationName} /></label><label>Task instructions<textarea className="custom-agent-control" maxLength={4000} onChange={(event) => setAutomationPrompt(event.target.value)} rows={4} value={automationPrompt} /></label>{automationKind === "schedule" ? <label>When<input className="custom-agent-control" maxLength={120} onChange={(event) => setScheduleText(event.target.value)} value={scheduleText} /><small>Examples: “in 30 minutes”, “every 6 hours”, or an ISO date and time.</small></label> : <div className="custom-agent-automation-grid"><label>Signal<select className="custom-agent-control" onChange={(event) => setTriggerSource(event.target.value)} value={triggerSource}><option value="cpu_temperature">CPU temperature</option><option value="memory_percent">Memory use</option><option value="storage_percent">Storage use</option><option value="service_failures">Failed Vaelor services</option><option value="fan_failure">Fan failure signal</option></select></label><label>Run when at or above<input className="custom-agent-control" min="1" onChange={(event) => setTriggerThreshold(event.target.value)} type="number" value={triggerThreshold} /></label></div>}<div className="dialog__actions"><Button variant="quiet" onClick={() => setAutomationAgent(null)} type="button">Cancel</Button><Button variant="primary" disabled={busy || !automationName.trim() || !automationPrompt.trim() || (automationKind === "schedule" ? !scheduleText.trim() : !triggerThreshold)} type="submit">Create automation</Button></div></form></ModalShell>}
      <ConfirmDialog open={Boolean(deleting)} title="Delete custom agent?" description="The definition and its revisions will be deleted. Completed task results remain in the ledger." confirmLabel="Delete agent" busy={busy} onCancel={() => setDeleting(null)} onConfirm={() => void remove()} />
    </section>
  );
}
