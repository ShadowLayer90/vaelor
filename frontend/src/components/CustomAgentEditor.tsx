import { type FormEvent, type ReactNode, useId } from "react";
import { connectorEventText } from "../lib/connectorActivity";
import { Icon } from "./Icon";
import { ModalShell } from "./ModalShell";
import type { AgentConnector } from "./agentTypes";
import { CAPABILITIES, type Draft, type KnowledgeCollection } from "./customAgentDraft";
import { Button, Input, Notice, Select, Textarea } from "./ui";

type Operation = AgentConnector["operations"][number];

/**
 * The agent editor: five steps in one dialog (the AssistAgentEditor board).
 *
 * One numbering scheme, and it is the one the body actually follows. A
 * four-step strip across the top once numbered the same form 1-4 while the
 * body sections numbered themselves 1-4 independently, and a "Step 3 · Review"
 * panel then appeared after body section 4. The strip only ever set its own
 * highlight - it never filtered or advanced anything - so it is gone, and the
 * numbered circles down the left are the only count.
 *
 * Steps 2 to 4 start closed and say what they hold; the review and the
 * approval policy are always open above the save button. An inference agent
 * (Cluster > Agents & tools) is a read-only cluster definition, so it gets
 * only the job and the read-only context, then the review.
 */
export function CustomAgentEditor({
  authoringReady,
  busy,
  collections,
  connectorAudit,
  connectorReadyForTest,
  draft,
  error,
  isInference,
  onClose,
  onLoadAudit,
  onSave,
  onTestConnector,
  setDraft,
  status,
}: {
  authoringReady: boolean;
  busy: boolean;
  collections: KnowledgeCollection[];
  connectorAudit?: Array<Record<string, unknown>>;
  connectorReadyForTest: (connector: AgentConnector) => boolean;
  draft: Draft;
  /** This dialog's own refusal (VD-189): shown inside it, never on the inert page. */
  error: string;
  isInference: boolean;
  onClose: () => void;
  onLoadAudit: () => void;
  onSave: (event: FormEvent) => void;
  onTestConnector: (connector: AgentConnector) => void;
  setDraft: (update: (current: Draft | null) => Draft | null) => void;
  /** This dialog's own result (a passed connection test, a prepared draft). */
  status: string;
}) {
  const edit = (patch: Partial<Draft>) => setDraft((current) => current ? { ...current, ...patch } : current);
  const toggle = (key: "scopes" | "permissions" | "read_collection_ids", value: string) => setDraft((current) => {
    if (!current) return current;
    const had = current[key].includes(value);
    return {
      ...current,
      [key]: had ? current[key].filter((item) => item !== value) : [...current[key], value],
      // Withdrawing a knowledge grant withdraws what it was granted on.
      ...(key === "permissions" && had && value === "knowledge:read" ? { read_collection_ids: [] } : {}),
      ...(key === "permissions" && had && value === "knowledge:write" ? { write_collection_id: "" } : {}),
    };
  });
  const updateConnector = (index: number, patch: Partial<AgentConnector>) => setDraft((current) => current ? {
    ...current,
    connectors: current.connectors.map((connector, itemIndex) => itemIndex === index ? { ...connector, ...patch } : connector),
  } : current);
  const updateOperation = (connectorIndex: number, operationIndex: number, patch: Partial<Operation>) => setDraft((current) => current ? {
    ...current,
    connectors: current.connectors.map((connector, itemIndex) => itemIndex === connectorIndex ? {
      ...connector,
      operations: connector.operations.map((operation, opIndex) => opIndex === operationIndex ? { ...operation, ...patch } : operation),
    } : connector),
  } : current);
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

  const invalidWrite = draft.permissions.includes("knowledge:write") && !draft.write_collection_id;
  const invalidConnector = draft.connectors.some((connector) =>
    !connector.name.trim()
    || !connector.base_origin.startsWith("https://")
    || (connector.auth !== "none" && !/^cred_[a-f0-9]{24}$/.test(connector.credential_ref))
    || connector.operations.length === 0
    || connector.operations.some((operation) => !operation.description.trim() || !operation.path.startsWith("/")),
  );
  // The draft carries research:read whenever the described job needs outside
  // information, so the editor can state that this step is required instead of
  // leaving the user to discover it from a failed run.
  const researchRequired = draft.scopes.includes("research:read");
  const missingJob = !draft.name.trim() || !draft.description.trim() || !draft.instructions.trim();
  const saveBlocked = !authoringReady
    ? (isInference ? "A serving cluster model is needed first" : "An Assistant model is needed first")
    : missingJob ? "Name, purpose and instructions are needed first"
      : invalidWrite ? "Choose the approved destination first"
        : invalidConnector ? "Complete each integration first"
          : undefined;
  const reviewStep = isInference ? 3 : 5;

  return (
    <ModalShell className="as-dialog ar-editor" error={error} labelledBy="custom-agent-editor-title" onClose={() => !busy && onClose()} size="wide">
      <form className="ar-dialog-form" onSubmit={onSave}>
        <div className="as-dialog__head">
          <div>
            <span className="as-label">Guided agent setup</span>
            <h2 id="custom-agent-editor-title">{draft.id ? "Edit custom agent" : "Create custom agent"}</h2>
            <p>Define its job, then grant only the information and actions it needs. Nothing can run until you save this version.</p>
          </div>
          <Button aria-label="Close agent editor" className="as-btn-ghost" disabled={busy} onClick={onClose} variant="quiet">Close</Button>
        </div>
        <div className="as-dialog__body ar-editor__body">
          {status && <Notice severity="success">{status}</Notice>}
          <div className="ar-step" data-route="/assistant/custom-agents/setup">
            <span className="ar-step__num is-on">1</span>
            <div className="ar-step__content">
              <div><strong className="ar-step__title">Define the job</strong><span className="ar-step__hint">Name the outcome this agent owns and how it should work.</span></div>
              <div className="as-grid2">
                <Input label="Name" maxLength={100} onChange={(event) => edit({ name: event.target.value })} value={draft.name} />
                <Input label="Purpose" maxLength={400} onChange={(event) => edit({ description: event.target.value })} value={draft.description} />
              </div>
              <Textarea label="Operating instructions" maxLength={6000} onChange={(event) => edit({ instructions: event.target.value })} rows={4} value={draft.instructions} />
            </div>
          </div>
          <EditorStep number={2} hint="Optional Vaelor facts, saved knowledge, and deployment proposals." title="Choose information access">
            <span className="as-label">Optional Vaelor context</span>
            <span className="ar-step__hint">Leave all of these off for a general research or documentation agent. Grant only the appliance facts this agent actually needs.</span>
            <div className="ar-checks">{CAPABILITIES.map(([id, name, description]) => <AccessOption checked={draft.scopes.includes(id)} description={description} key={id} onChange={() => toggle("scopes", id)} title={name} />)}</div>
            {!isInference && (
              <>
                <span className="as-label">Knowledge and action permissions</span>
                <span className="ar-step__hint">No shell, raw credentials, arbitrary filesystem access, or unrestricted URLs are exposed.</span>
                <AccessOption checked={draft.permissions.includes("knowledge:read")} description="Retrieve cited chunks only from named collections." onChange={() => toggle("permissions", "knowledge:read")} title="Read selected knowledge">
                  {draft.permissions.includes("knowledge:read") && (
                    collections.length ? (
                      <span aria-label="Readable knowledge collections" className="ar-chips" role="group">
                        {collections.map((collection) => (
                          <Button aria-pressed={draft.read_collection_ids.includes(collection.id)} className={draft.read_collection_ids.includes(collection.id) ? "ar-chip as-btn-on" : "ar-chip"} key={collection.id} onClick={() => toggle("read_collection_ids", collection.id)} variant="quiet">
                            {collection.name} · {collection.document_count} document{collection.document_count === 1 ? "" : "s"}
                          </Button>
                        ))}
                      </span>
                    ) : <span className="ar-step__hint">Create a collection in AI Chat before granting knowledge access.</span>
                  )}
                </AccessOption>
                <AccessOption checked={draft.permissions.includes("knowledge:write")} description="The exact content is held for operator approval before it is stored." onChange={() => toggle("permissions", "knowledge:write")} title="Propose a knowledge document">
                  {draft.permissions.includes("knowledge:write") && (
                    <Select label="Approved destination" onChange={(event) => edit({ write_collection_id: event.target.value })} value={draft.write_collection_id}>
                      <option value="">Choose a collection</option>
                      {collections.map((collection) => <option key={collection.id} value={collection.id}>{collection.name}</option>)}
                    </Select>
                  )}
                </AccessOption>
                <AccessOption checked={draft.permissions.includes("workloads:propose")} description="May prepare a bounded deployment plan but cannot execute it during an agent run." onChange={() => toggle("permissions", "workloads:propose")} title="Propose workload changes" />
              </>
            )}
          </EditorStep>
          {/*
            * Labelling this "Optional" was how a beginner built an agent that
            * could never work: an agent asked for scores or news has no way to
            * answer without it, and the failure only showed up after creation.
            */}
          {!isInference && (
            <EditorStep number={3} open={researchRequired} required={researchRequired} hint={researchRequired ? "Required — this agent needs information from the internet." : "Optional, guarded web access with domain limits."} title="Allow public research">
              <span className="as-label">Internet permission</span>
              <span className="ar-step__hint">Internet access always goes through Vaelor's guarded research broker. The model never receives a raw network connection.</span>
              <AccessOption checked={draft.web_access.enabled} description="With no domains listed, the agent may search but cannot open arbitrary results." onChange={(checked) => setDraft((current) => current ? { ...current, web_access: { ...current.web_access, enabled: checked }, scopes: checked ? [...new Set([...current.scopes, "research:read"])] : current.scopes.filter((scope) => scope !== "research:read") } : current)} title="Allow guarded public research" />
              {draft.web_access.enabled && (
                <Textarea hint="List only the documentation sites this agent may fetch. Search remains available when this is empty." label="Allowed HTTPS domains (optional)" maxLength={2000} onChange={(event) => edit({ web_access: { enabled: true, allowed_domains: event.target.value.split(/[\s,]+/).map((item) => item.trim().toLowerCase()).filter(Boolean) } })} placeholder={"example.com\ndocs.example.org"} rows={3} value={draft.web_access.allowed_domains.join("\n")} />
              )}
            </EditorStep>
          )}
          {!isInference && (
            <EditorStep number={4} hint="Optional, fixed API actions using brokered credentials." title="Connect external services">
              <div className="ar-split-head">
                <div><strong className="ar-step__title">Integrations and API grants</strong><span className="ar-step__hint">Grant named operations only. Vaelor stores a credential reference—not the secret—and never gives the model an unrestricted URL.</span></div>
                <div className="ar-actions">
                  {draft.id && draft.connectors.length > 0 && <Button className="as-btn-ghost" disabled={busy} onClick={onLoadAudit} variant="quiet">View API audit</Button>}
                  <Button onClick={addConnector}>Add integration</Button>
                </div>
              </div>
              {connectorAudit && (
                <div className="as-box">
                  <strong className="ar-step__title">Recent integration activity</strong>
                  {connectorAudit.length
                    ? <ul className="ar-list">{connectorAudit.slice(0, 10).map((event, index) => <li key={index}>{connectorEventText(event, draft.connectors)}</li>)}</ul>
                    : <span className="ar-step__hint">No integration calls recorded.</span>}
                </div>
              )}
              {draft.connectors.length === 0 && (
                <div className="as-box ar-empty-row">
                  <span aria-hidden="true" className="ar-icon"><Icon name="shield" size={16} /></span>
                  <div><strong className="ar-step__title">No third-party API access</strong><span className="ar-step__hint">This agent cannot call an external service unless you add and save a bounded integration.</span></div>
                </div>
              )}
              {draft.connectors.map((connector, connectorIndex) => (
                <ConnectorBox
                  busy={busy}
                  connector={connector}
                  connectorIndex={connectorIndex}
                  key={connector.id}
                  onAddOperation={() => addOperation(connectorIndex)}
                  onRemove={() => setDraft((current) => current ? { ...current, connectors: current.connectors.filter((_, index) => index !== connectorIndex) } : current)}
                  onTest={() => onTestConnector(connector)}
                  onUpdate={(patch) => updateConnector(connectorIndex, patch)}
                  onUpdateOperation={(operationIndex, patch) => updateOperation(connectorIndex, operationIndex, patch)}
                  readyForTest={connectorReadyForTest(connector)}
                  saved={Boolean(draft.id)}
                />
              ))}
            </EditorStep>
          )}
          <div className="ar-step">
            <span className="ar-step__num">{reviewStep}</span>
            <section aria-labelledby="custom-agent-review-heading" className="ar-step__content ar-review">
              <div>
                <span className="as-label">{isInference ? "Review" : "Step 5 · Review"}</span>
                <h3 className="ar-step__title" id="custom-agent-review-heading">Confirm intent and access before saving</h3>
                <span className="ar-step__hint">{draft.name || "This agent"} will be created as a versioned definition. The runner receives only the grants listed below.</span>
              </div>
              <dl className="as-kv ar-kv2">
                <div><dt>Purpose</dt><dd>{draft.description || "Not supplied"}</dd></div>
                <div><dt>Vaelor access</dt><dd className="as-mono">{draft.scopes.length ? draft.scopes.join(" · ") : "None"}</dd></div>
                {!isInference && (
                  <>
                    <div><dt>Knowledge and actions</dt><dd className="as-mono">{draft.permissions.length ? draft.permissions.join(" · ") : "None"}</dd></div>
                    <div><dt>Public research</dt><dd>{draft.web_access.enabled ? (draft.web_access.allowed_domains.length ? draft.web_access.allowed_domains.join(" · ") : "Guarded search") : "Off"}</dd></div>
                    <div className="ar-kv-wide"><dt>Integrations</dt><dd>{draft.connectors.length ? draft.connectors.map((connector) => connector.name || "Unnamed integration").join(" · ") : "None"}</dd></div>
                  </>
                )}
              </dl>
              <span className="as-label" id="custom-agent-policy-heading">Approval and automation policy</span>
              <dl aria-labelledby="custom-agent-policy-heading" className="as-kv ar-kv2">
                <div><dt>Test and manual runs</dt><dd>Start when you press Run now; pressing it is the review</dd></div>
                <div><dt>Knowledge writes</dt><dd>Exact destination and content require separate approval</dd></div>
                <div><dt>Workload changes</dt><dd>Proposal only; deployment has its own approval</dd></div>
                <div><dt>Schedules and triggers</dt><dd>Administrator-only, pinned to the saved version. Their runs start without a further approval, read only, and turn any write into a proposal that needs one</dd></div>
              </dl>
            </section>
          </div>
          {invalidConnector && <Notice severity="warning">Complete each integration's HTTPS origin, broker reference, and at least one named route before saving.</Notice>}
        </div>
        <div className="as-dialog__foot">
          <Button onClick={onClose}>Cancel</Button>
          <Button disabled={busy} disabledReason={saveBlocked} type="submit" variant="primary">{busy ? "Saving…" : draft.id ? "Save new version" : "Create agent"}</Button>
        </div>
      </form>
    </ModalShell>
  );
}

/** A closed-by-default step: a numbered circle and a disclosure that says what it holds. */
function EditorStep({ children, hint, number, open = false, required = false, title }: {
  children: ReactNode;
  hint: string;
  number: number;
  open?: boolean;
  required?: boolean;
  title: string;
}) {
  return (
    <div className="ar-step">
      <span className="ar-step__num">{number}</span>
      <details className="ar-step__group" open={open}>
        <summary className="ar-disc">
          <span><strong className="ar-step__title">{title}</strong><span className={required ? "ar-step__hint ar-step__hint--required" : "ar-step__hint"}>{hint}</span></span>
          <Icon className="ar-disc__chevron" name="chevron" size={16} />
        </summary>
        <div className="ar-step__content">{children}</div>
      </details>
    </div>
  );
}

function ConnectorBox({ busy, connector, connectorIndex, onAddOperation, onRemove, onTest, onUpdate, onUpdateOperation, readyForTest, saved }: {
  busy: boolean;
  connector: AgentConnector;
  connectorIndex: number;
  onAddOperation: () => void;
  onRemove: () => void;
  onTest: () => void;
  onUpdate: (patch: Partial<AgentConnector>) => void;
  onUpdateOperation: (operationIndex: number, patch: Partial<Operation>) => void;
  readyForTest: boolean;
  saved: boolean;
}) {
  return (
    <article className="as-box ar-connector">
      <div className="ar-split-head">
        <div><span className="ar-step__hint">Connector {connectorIndex + 1}</span><strong className="ar-step__title">{connector.name || "Unnamed integration"}</strong></div>
        <div className="ar-actions">
          {saved && <Button disabled={busy || !readyForTest} onClick={onTest}>{readyForTest ? "Test connection" : "Save before testing"}</Button>}
          <Button className="as-btn-danger" onClick={onRemove}>{saved ? "Revoke" : "Remove"}</Button>
        </div>
      </div>
      <Input label="Connector name" maxLength={100} onChange={(event) => onUpdate({ name: event.target.value })} placeholder="Market data API" value={connector.name} />
      <Input label="HTTPS base origin" maxLength={300} onChange={(event) => onUpdate({ base_origin: event.target.value })} placeholder="https://api.example.com" value={connector.base_origin} />
      <Select label="Authentication" onChange={(event) => onUpdate({ auth: event.target.value as AgentConnector["auth"] })} value={connector.auth}>
        <option value="none">No credential</option>
        <option value="bearer">Bearer token</option>
        <option value="x-api-key">X-API-Key header</option>
      </Select>
      <Input disabled={connector.auth === "none"} hint="The secret is never displayed or copied into this agent." label="Credential broker reference" maxLength={29} onChange={(event) => onUpdate({ credential_ref: event.target.value })} placeholder="cred_… (reference only)" value={connector.credential_ref} />
      <div className="ar-split-head">
        <div><strong className="ar-step__title">Allowed operations</strong><span className="ar-step__hint">Reads may run without approval. POST, PUT, PATCH, and DELETE always require an exact preview and approval.</span></div>
        <Button className="as-btn-ghost" onClick={onAddOperation} variant="quiet">Add operation</Button>
      </div>
      {connector.operations.map((operation, operationIndex) => {
        const write = !["GET", "HEAD"].includes(operation.method);
        return (
          <div className="ar-operation" key={operation.id}>
            <div className="ar-operation__fields">
              <Input label="Description" maxLength={200} onChange={(event) => onUpdateOperation(operationIndex, { description: event.target.value })} placeholder="Read a stock quote" value={operation.description} />
              <Select label="Method" onChange={(event) => {
                const method = event.target.value as Operation["method"];
                const isWrite = !["GET", "HEAD"].includes(method);
                onUpdateOperation(operationIndex, { method, input_location: isWrite ? "json" : "query", approval: isWrite ? "required" : "not_required" });
              }} value={operation.method}>
                <option>GET</option><option>HEAD</option><option>POST</option><option>PUT</option><option>PATCH</option><option>DELETE</option>
              </Select>
              <Input label="Fixed route template" maxLength={300} onChange={(event) => onUpdateOperation(operationIndex, { path: event.target.value })} placeholder="/v1/quotes/{symbol}" value={operation.path} />
            </div>
            <div className="ar-operation__foot">
              <span className="ar-operation__risk">
                <span className={write ? "status-pill status-pill--warning" : "status-pill status-pill--neutral"}><span aria-hidden="true" className="status-pill__dot" />{write ? "State-changing" : "Read only"}</span>
                <span className="ar-step__hint">{write ? "Exact preview and operator approval required" : "No approval; rate and response limits still apply"}</span>
              </span>
              <Button aria-label={`Remove ${operation.description || `operation ${operationIndex + 1}`}`} className="as-btn-ghost ar-danger-word" onClick={() => onUpdate({ operations: connector.operations.filter((_, index) => index !== operationIndex) })} variant="quiet">Remove operation</Button>
            </div>
          </div>
        );
      })}
    </article>
  );
}

/**
 * One access checkbox in the agent editor: named by its title alone and
 * described by its help text (UX-A6). Wrapping both in the label made the
 * accessible name the title and the whole description run together. What it
 * unlocks (a collection, a destination) sits inside its box, under its words.
 */
function AccessOption({ checked, children, description, onChange, title }: {
  checked: boolean;
  children?: ReactNode;
  description: ReactNode;
  onChange: (checked: boolean) => void;
  title: ReactNode;
}) {
  const id = useId().replaceAll(":", "");
  return (
    <div className="ar-check">
      <input aria-describedby={`${id}-description`} aria-labelledby={`${id}-title`} checked={checked}
        className="ar-check__input" id={`${id}-input`} onChange={(event) => onChange(event.target.checked)} type="checkbox" />
      <span className="ar-check__text">
        <label htmlFor={`${id}-input`} id={`${id}-title`}>{title}</label>
        <span className="ar-step__hint" id={`${id}-description`}>{description}</span>
        {children}
      </span>
    </div>
  );
}
