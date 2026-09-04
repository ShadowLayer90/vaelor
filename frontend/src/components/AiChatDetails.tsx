import { type FormEvent } from "react";
import { formatQuantity } from "../lib/format";
import type {
  AiChatCollection,
  AiChatConnection,
  AiChatDocument,
  AiChatSetup,
} from "./aiChatTypes";
import { Icon } from "./Icon";
import { Button, Checkbox, Input } from "./ui";

export function AiChatDetails({
  setup,
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
  return (
    <aside className="ai-chat-details" aria-label="AI Chat settings">
      <header>
        <div><span className="page-eyebrow">Chat configuration</span><h2>Details</h2></div>
        <Button aria-label="Close details" className="ai-chat-details__close" onClick={onClose} type="button" variant="quiet">×</Button>
      </header>
      <section className="ai-chat-details__connection">
        <div className="ai-chat-details__heading">
          <span><Icon name="network" size={17} /></span>
          <div><strong>{active?.label ?? "No AI connection"}</strong><small>{active?.provider ?? "Connect a model before chatting"}</small></div>
        </div>
        <details>
          <summary>Change connection</summary>
          <div className="ai-chat-details__connections">
            {setup?.connections.map((item) => (
              <Button
                className={item.id === active?.id ? "is-active" : undefined}
                disabled={busy === "connection"}
                key={item.id}
                onClick={() => onActivateConnection(item)}
                type="button"
                variant="quiet"
              >
                <strong>{item.label}</strong><small>{item.provider}</small>
              </Button>
            ))}
          </div>
        </details>
      </section>
      <section className="ai-chat-details__knowledge">
        <div className="ai-chat-details__section-title">
          <div><h3>Knowledge</h3><p>{setup?.retrieval ?? "Local cited retrieval"}</p></div>
          <span>{selectedCollections.length} active</span>
        </div>
        <div className="ai-chat-collections">
          {setup?.collections.map((item) => (
            <div
              className={item.id === activeCollection ? "ai-chat-collection is-open" : "ai-chat-collection"}
              key={item.id}
            >
              <Checkbox
                checked={selectedCollections.includes(item.id)}
                className="ai-chat-collection__checkbox"
                id={"ai-chat-collection-" + item.id}
              label={<span><strong>{item.name}</strong><small>{item.document_count} files · {formatQuantity(item.size_bytes, "capacity")}</small></span>}
                onChange={() => onToggleCollection(item.id)}
              />
              <Button aria-label={"Manage " + item.name} className="ai-chat-collection__manage" onClick={() => onActiveCollection(item.id)} type="button" variant="quiet">
                <Icon name="chevron" size={15} />
              </Button>
            </div>
          ))}
          {!setup?.collections.length && <p className="empty-copy">No knowledge collections yet.</p>}
        </div>
        <details className="ai-chat-details__create" open={!setup?.collections.length}>
          <summary className="ui-button ui-button--quiet">Create knowledge collection</summary>
          <form onSubmit={onCreateCollection}>
            <Input id="ai-chat-collection-name" label="Name" maxLength={100} onChange={(event) => onCollectionName(event.target.value)} value={collectionName} />
            <Input id="ai-chat-collection-description" label="Description" maxLength={500} onChange={(event) => onCollectionDescription(event.target.value)} value={collectionDescription} />
            <Button disabled={!collectionName.trim() || busy === "collection"} type="submit" variant="quiet">Create</Button>
          </form>
        </details>
      </section>
      {selected && (
        <section className="ai-chat-details__documents">
          <div className="ai-chat-details__section-title">
            <div><h3>{selected.name}</h3><p>{selected.description || "Collection files"}</p></div>
            <Button className="danger-link" onClick={() => onDeleteCollection(selected)} type="button" variant="danger">Delete</Button>
          </div>
          <label className="ai-chat-upload">
            <Icon name="database" size={18} />
            <span><strong>Add a text file</strong><small>Text, Markdown, JSON, CSV, YAML, or log · 2 MiB max</small></span>
            <input
              accept=".txt,.md,.markdown,.json,.csv,.yaml,.yml,.log,text/*"
              className="ui-control ui-control--file-picker"
              disabled={busy === "document"}
              onChange={(event) => {
                const file = event.target.files?.[0];
                if (file) onFile(file);
                event.target.value = "";
              }}
              type="file"
            />
          </label>
          <div className="ai-chat-documents">
            {documents.map((item) => (
              <article key={item.id}>
                <div><strong>{item.name}</strong><small>{item.chunk_count} chunks · {formatQuantity(item.size_bytes, "capacity")}</small></div>
                <Button disabled={busy === "document"} onClick={() => onDeleteDocument(item.id)} type="button" variant="danger">Delete</Button>
              </article>
            ))}
            {!documents.length && <p className="empty-copy">No indexed files in this collection.</p>}
          </div>
        </section>
      )}
      <footer>
        <Icon name="shield" size={16} />
        <span>Retrieved text is treated as untrusted reference material. Sources remain local.</span>
      </footer>
    </aside>
  );
}
