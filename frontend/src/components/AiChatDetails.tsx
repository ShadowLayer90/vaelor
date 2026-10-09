import { useId, useRef, useState, type FormEvent } from "react";
import { DOCUMENT_ACCEPT } from "../lib/aiChatDocument";
import { formatQuantity } from "../lib/format";
import { useGpuServingMode } from "../hooks/useGpuServingMode";
import {
  CLUSTER_LAN_OPEN_CLAUSE,
  LOCAL_BY_MANAGED_PREFIX,
  SERVING_KIND_CLUSTER,
  clusterServedBy,
} from "../lib/gpuServingMode";
import type {
  AiChatCollection,
  AiChatConnection,
  AiChatDocument,
  AiChatSetup,
} from "./aiChatTypes";
import { connectionLine } from "./AiChatModelPicker";
import { Icon, ICON_SIZE } from "./Icon";
import { LlmServerSummary } from "./LlmServerSummary";
import { Button } from "./ui";
import type { Session } from "../types";

/**
 * The Details panel (the ChatArchive board): the active connection and the
 * engine behind it, Change connection, the LLM Server summary for an
 * administrator, and the knowledge collections with their files.
 */
export function AiChatDetails({
  setup,
  session,
  activeCollection,
  documents,
  selectedCollections,
  collectionName,
  collectionDescription,
  busy,
  onClose,
  onActivateConnection,
  onActiveCollection,
  onToggleCollection,
  onCollectionName,
  onCollectionDescription,
  onCreateCollection,
  onFile,
  onDeleteDocument,
  onDeleteCollection,
}: {
  setup: AiChatSetup | null;
  session?: Session;
  activeCollection: string;
  documents: AiChatDocument[];
  selectedCollections: string[];
  collectionName: string;
  collectionDescription: string;
  busy: string;
  onClose: () => void;
  onActivateConnection: (connection: AiChatConnection) => void;
  onActiveCollection: (id: string) => void;
  onToggleCollection: (id: string) => void;
  onCollectionName: (value: string) => void;
  onCollectionDescription: (value: string) => void;
  onCreateCollection: (event: FormEvent) => void;
  onFile: (file: File) => void;
  onDeleteDocument: (id: string) => void;
  onDeleteCollection: (collection: AiChatCollection) => void;
}) {
  const active = setup?.active_connection;
  const selected = setup?.collections.find((item) => item.id === activeCollection);
  const [changing, setChanging] = useState(false);
  const fileRef = useRef<HTMLInputElement>(null);
  /*
   * Which engine is behind this connection (VD-127). The connection card names
   * the credential; it could not say whether the model on the other end is this
   * controller's own llama.cpp server or a GPU cluster that has taken AI Chat
   * over — and in Mode B the AI Chat model the owner deployed is stopped, which
   * is exactly the fact they come to this panel to check. Rendered only when
   * the mode is actually known: an unreadable answer says nothing rather than
   * asserting "managed local model".
   *
   * Resolved ONCE, here, from both inputs: the LLM Server surface's own
   * `target_kind` (reported up by the panel below as it loads it) and the
   * fleet's deployments. The one answer feeds this card and the LLM Server
   * panel alike, so the two cannot disagree inside the same aside.
   */
  const [targetKind, setTargetKind] = useState<string | undefined>(undefined);
  const serving = useGpuServingMode(session?.user.role, targetKind);
  const clustered = serving.known && serving.kind === SERVING_KIND_CLUSTER;
  // "Managed local" is claimed only on the backend's own evidence that Vaelor
  // minted this credential for its own deploy — never merely because no cluster
  // was found, which would print "prompts never leave the machine" over a
  // hosted provider.
  const managedLocal = !clustered && active?.local_source === LOCAL_BY_MANAGED_PREFIX;
  /*
   * VD-210: AI Chat's model is the owner's choice while the cluster serves.
   * Only the connections the backend refuses (this machine's own GPU model,
   * whose llama.cpp is stopped while clustered) are held, each with the
   * backend's own sentence under it, so it is said BEFORE the click. A
   * whole-picker refusal (`assignment_refusal`) still holds every row if an
   * older appliance sends one. Keyed on the sentences, never on `clustered`:
   * the mode file the backend reads is the fact.
   */
  const refusals = setup?.connection_refusals ?? {};
  const heldAll = setup?.assignment_refusal || "";
  const ids = useId().replaceAll(":", "");
  const refusalId = "ai-chat-connection-refusal-" + ids;
  // The cluster card describes the cluster answering - only true when AI
  // Chat's connection is the cluster's own (an older surface names none).
  const onCluster = clustered && (!setup?.cluster_credential_id || active?.id === setup.cluster_credential_id);
  const connectionsId = "ai-chat-connections-" + ids;
  return (
    <aside className="ai-chat-details" aria-label="AI Chat settings">
      <header className="ai-chat-details__header">
        <div><p className="ai-chat-eyebrow">Chat configuration</p><h2>Details</h2></div>
        <Button aria-label="Close details" className="ai-chat-ghost" onClick={onClose} type="button" variant="quiet">Close</Button>
      </header>
      <section className="ai-chat-details__card ai-chat-details__connection" aria-label="Active connection">
        <div className="ai-chat-details__heading">
          <span className="ai-chat-icon-tile"><Icon name="server" size={ICON_SIZE.nav} /></span>
          <div>
            <small>Active connection</small>
            <strong>{active?.label ?? "No AI connection"}</strong>
            <small>{active ? connectionLine(active) : "Connect a model before chatting"}</small>
          </div>
        </div>
        {onCluster && (
          <p className="ai-chat-details__mode">
            <strong>
              {/* The LLM Server panel's Served-by spelling, then the model:
                  "GPU cluster · <name> (N replicas)" for a replicated
                  cluster (VD-129). */}
              {clusterServedBy(serving)}
              {serving.model ? ` · ${serving.model}` : ""}
            </strong>
            <span>
              GPU clustering is on, so this controller&rsquo;s own AI Chat model is
              stopped and answers come from the cluster
              {/* VD-127 D2: a worker-led cluster answers on the LAN with no key,
                  and the owner is told beside the fact that they are on it. */}
              {serving.lanExposed ? `, ${CLUSTER_LAN_OPEN_CLAUSE}` : ""}
              . Removing the cluster brings that model back.
            </span>
          </p>
        )}
        {managedLocal && (
          <p className="ai-chat-details__mode">
            <strong>Managed local model</strong>
            <span>
              This appliance serves the model itself, so prompts never leave the
              machine.
            </span>
          </p>
        )}
        <Button
          aria-controls={connectionsId}
          aria-expanded={changing}
          className="ai-chat-disclosure"
          onClick={() => setChanging((open) => !open)}
          type="button"
          variant="quiet"
        >
          <span>Change connection</span>
          <Icon name="chevron" size={ICON_SIZE.inline} />
        </Button>
        {changing && (
          <div className="ai-chat-details__connections" id={connectionsId}>
            {/* One sentence for the whole list, described-by from every
                choice, rather than a reason under each of N buttons. */}
            {heldAll && <p className="ai-chat-details__refusal" id={refusalId}>{heldAll}</p>}
            {setup?.connections.map((item) => {
              const held = heldAll || refusals[item.id] || "";
              const reasonId = heldAll ? refusalId : `${refusalId}-${item.id}`;
              return (
                <div className="ai-chat-details__choice" key={item.id}>
                  <Button
                    aria-describedby={held ? reasonId : undefined}
                    aria-pressed={item.id === active?.id}
                    className={item.id === active?.id ? "ai-chat-choice is-active" : "ai-chat-choice"}
                    disabled={busy === "connection" || Boolean(held)}
                    onClick={() => onActivateConnection(item)}
                    type="button"
                    variant="quiet"
                  >
                    <span><strong>{item.label}</strong><small>{connectionLine(item)}</small></span>
                    {item.id === active?.id && <Icon name="done" size={ICON_SIZE.inline} />}
                  </Button>
                  {held && !heldAll && <p className="ai-chat-details__refusal" id={reasonId}>{held}</p>}
                </div>
              );
            })}
            {!setup?.connections.length && <p className="ai-chat-details__empty">No connections are set up yet.</p>}
          </div>
        )}
      </section>
      {/* `/llm-server` is administrator-only: any other role's read is a 403,
          which the summary would show as an LLM Server fault. */}
      {session?.user.role === "administrator" ? (
        <LlmServerSummary onTargetKind={setTargetKind} />
      ) : null}
      <section className="ai-chat-details__knowledge" aria-labelledby={`ai-chat-knowledge-${ids}`}>
        <div className="ai-chat-details__section-title">
          <div><h3 id={`ai-chat-knowledge-${ids}`}>Knowledge</h3><p>{setup?.retrieval ?? "Local cited retrieval"}</p></div>
          <span>{selectedCollections.length} active</span>
        </div>
        {/*
          * Checking a collection narrows the answer to it, and nothing said so:
          * the same model that wrote an essay with it off replied "I don't have
          * any reliable information on that" with it on. The sentence names the
          * constraint beside the control that removes it.
          */}
        {selectedCollections.length > 0 && (
          <p className="ai-chat-details__note">
            Answers are grounded in the checked collections. Uncheck one to let Vaelor
            answer from what it already knows as well.
          </p>
        )}
        {setup?.collections.map((item) => {
          const checked = selectedCollections.includes(item.id);
          const open = item.id === activeCollection;
          return (
            <div className={checked ? "ai-chat-collection is-checked" : "ai-chat-collection"} key={item.id}>
              <input
                aria-label={`Use ${item.name}`}
                checked={checked}
                className="ai-chat-collection__check"
                id={"ai-chat-collection-" + item.id}
                onChange={() => onToggleCollection(item.id)}
                type="checkbox"
              />
              <label htmlFor={"ai-chat-collection-" + item.id}>
                <strong>{item.name}</strong>
                <small>{item.document_count} files · {formatQuantity(item.size_bytes, "capacity")}</small>
              </label>
              <Button
                aria-expanded={open}
                aria-label={"Manage " + item.name}
                className={open ? "ai-chat-collection__manage is-open" : "ai-chat-collection__manage"}
                onClick={() => onActiveCollection(open ? "" : item.id)}
                type="button"
                variant="quiet"
              >
                <Icon name="chevron" size={ICON_SIZE.inline} />
              </Button>
            </div>
          );
        })}
        {!setup?.collections.length && <p className="ai-chat-details__empty">No knowledge collections yet.</p>}
        {selected && (
          <div className="ai-chat-details__documents" aria-label={`${selected.name} files`} role="group">
            <div className="ai-chat-details__collection-head">
              <div><strong>{selected.name}</strong><small>{selected.description || "Collection files"}</small></div>
              <Button className="ai-chat-danger-outline" onClick={() => onDeleteCollection(selected)} type="button" variant="quiet">Delete</Button>
            </div>
            <input
              accept={DOCUMENT_ACCEPT}
              aria-label={`Add a text file to ${selected.name}`}
              className="ui-control ui-control--file-picker"
              disabled={busy === "document"}
              hidden
              onChange={(event) => {
                const file = event.target.files?.[0];
                if (file) onFile(file);
                event.target.value = "";
              }}
              ref={fileRef}
              type="file"
            />
            <Button className="ai-chat-details__upload" disabled={busy === "document"} onClick={() => fileRef.current?.click()} type="button">
              <Icon name="file" size={16} /> Add a text file
            </Button>
            <small>Text, Markdown, JSON, CSV, YAML, or log · 2 MiB max</small>
            {documents.map((item) => (
              <div className="ai-chat-document" key={item.id}>
                <span><strong>{item.name}</strong><small>{item.chunk_count} chunks · {formatQuantity(item.size_bytes, "capacity")}</small></span>
                <Button aria-label={`Delete ${item.name}`} className="ai-chat-ghost ai-chat-ghost--danger" disabled={busy === "document"} onClick={() => onDeleteDocument(item.id)} type="button" variant="quiet">Delete</Button>
              </div>
            ))}
            {!documents.length && <p className="ai-chat-details__empty">No indexed files in this collection.</p>}
          </div>
        )}
        <form className="ai-chat-details__create" onSubmit={onCreateCollection}>
          <h4>Create knowledge collection</h4>
          <label className="ai-chat-field">
            Name
            <input className="ai-chat-input" id="ai-chat-collection-name" maxLength={100} onChange={(event) => onCollectionName(event.target.value)} value={collectionName} />
          </label>
          <label className="ai-chat-field">
            Description
            <input className="ai-chat-input" id="ai-chat-collection-description" maxLength={500} onChange={(event) => onCollectionDescription(event.target.value)} placeholder="Optional" value={collectionDescription} />
          </label>
          <Button disabled={!collectionName.trim() || busy === "collection"} type="submit">Create</Button>
        </form>
      </section>
      <footer className="ai-chat-details__footer">
        <Icon name="shield" size={16} />
        <span>Retrieved text is treated as untrusted reference material. Sources remain local.</span>
      </footer>
    </aside>
  );
}
