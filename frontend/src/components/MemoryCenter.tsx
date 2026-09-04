import { useCallback, useEffect, useState } from "react";
import type { FormEvent } from "react";
import { apiRequest } from "../lib/api";
import { timeAgo } from "../lib/format";
import type { Session } from "../types";
import { Icon, type IconName } from "./Icon";
import { PaginatedItems } from "./PaginatedItems";
import { StatusPill } from "./StatusPill";
import { Button, Input, Notice, Select, Textarea } from "./ui";
import { destinations } from "../lib/destinations";

interface MemoryStats {
  memories: number;
  pinned: number;
  conversations: number;
  messages: number;
  search: string;
}

interface AssistantMemory {
  id: string;
  memory_type: string;
  scope: string;
  content: string;
  source: string;
  confidence: number;
  pinned: boolean;
  review_state: string;
  created_by: string;
  created_at: number;
  updated_at: number;
  last_used_at?: number | null;
  expires_at?: number | null;
}

interface AssistantConversation {
  id: string;
  title: string;
  summary: string;
  created_at: number;
  updated_at: number;
  archived: number;
  message_count: number;
}

const memoryTypes = [
  ["preference", "Preference"],
  ["device_fact", "Device fact"],
  ["project_fact", "Project fact"],
  ["decision", "Decision"],
  ["procedure", "Procedure"],
  ["incident", "Incident"],
  ["correction", "Correction"],
] as const;

const memoryScopes = [
  ["global", "Everywhere"],
  ["device", "This Vaelor node"],
  ["workload", "An installed app"],
  ["model", "AI models"],
  ["conversation", "One conversation"],
] as const;

export function MemoryCenter({ session }: { session: Session }) {
  const [stats, setStats] = useState<MemoryStats | null>(null);
  const [memories, setMemories] = useState<AssistantMemory[]>([]);
  const [conversations, setConversations] = useState<AssistantConversation[]>([]);
  const [query, setQuery] = useState("");
  const [typeFilter, setTypeFilter] = useState("");
  const [content, setContent] = useState("");
  const [memoryType, setMemoryType] = useState("preference");
  const [scope, setScope] = useState("global");
  const [source, setSource] = useState("Manual administrator entry");
  const [pinned, setPinned] = useState(true);
  const [editingId, setEditingId] = useState("");
  const [editingContent, setEditingContent] = useState("");
  const [pendingDelete, setPendingDelete] = useState("");
  const [busy, setBusy] = useState(false);
  const [notice, setNotice] = useState("");

  const load = useCallback(async (search = query, filter = typeFilter) => {
    const parameters = new URLSearchParams();
    if (search.trim()) parameters.set("query", search.trim());
    if (filter) parameters.set("type", filter);
    const suffix = parameters.size ? `?${parameters.toString()}` : "";
    const [nextStats, nextMemories, nextConversations] = await Promise.all([
      apiRequest<MemoryStats>("/assistant/status"),
      apiRequest<AssistantMemory[]>(`/assistant/memories${suffix}`),
      apiRequest<AssistantConversation[]>("/assistant/conversations?limit=8"),
    ]);
    setStats(nextStats);
    setMemories(nextMemories);
    setConversations(nextConversations);
  }, [query, typeFilter]);

  useEffect(() => {
    void load().catch((error) => {
      setNotice(error instanceof Error ? error.message : "Assistant memory could not be loaded.");
    });
  }, [load]);

  const createMemory = async (event: FormEvent) => {
    event.preventDefault();
    if (!content.trim()) return;
    setBusy(true);
    setNotice("");
    try {
      await apiRequest(
        "/assistant/memories",
        {
          method: "POST",
          body: JSON.stringify({
            content,
            memory_type: memoryType,
            scope,
            source,
            confidence: 1,
            pinned,
          }),
        },
        session.csrf_token,
      );
      setContent("");
      setNotice("Memory reviewed and saved. It is available to new assistant requests.");
      await load("", "");
      setQuery("");
      setTypeFilter("");
    } catch (error) {
      setNotice(error instanceof Error ? error.message : "Memory could not be saved.");
    } finally {
      setBusy(false);
    }
  };

  const updateMemory = async (memoryId: string, patch: Record<string, unknown>) => {
    setBusy(true);
    setNotice("");
    try {
      await apiRequest(
        `/assistant/memories/${memoryId}`,
        { method: "PATCH", body: JSON.stringify(patch) },
        session.csrf_token,
      );
      setEditingId("");
      setEditingContent("");
      setNotice("Memory updated.");
      await load();
    } catch (error) {
      setNotice(error instanceof Error ? error.message : "Memory could not be updated.");
    } finally {
      setBusy(false);
    }
  };

  const deleteMemory = async (memoryId: string) => {
    if (pendingDelete !== memoryId) {
      setPendingDelete(memoryId);
      return;
    }
    setBusy(true);
    try {
      await apiRequest(
        `/assistant/memories/${memoryId}`,
        { method: "DELETE" },
        session.csrf_token,
      );
      setPendingDelete("");
      setNotice("Memory permanently removed from active recall.");
      await load();
    } catch (error) {
      setNotice(error instanceof Error ? error.message : "Memory could not be removed.");
    } finally {
      setBusy(false);
    }
  };

  return (
    <div className="memory-page">
      {/*
        * Memory is appliance-wide, not the Assistant's. `_list_memories` has no
        * actor filter, so the same store answers both AI surfaces — and while
        * this lived as a tab inside the Assistant it read as the Assistant's
        * private notebook, which is a claim the server never made. The heading
        * says who uses it so nobody has to read the storage layer to find out.
        */}
      <div className="page-heading">
        <div>
          <h1>What Vaelor remembers — used by both {destinations.assistant.name} and {destinations["ai-chat"].name}</h1>
          <p>Review exactly what Vaelor can remember. Secrets and instruction overrides are blocked automatically.</p>
        </div>
        <StatusPill label="Local & private" status="healthy" />
      </div>
      <p className="memory-page__return">
        <a href="#/assistant">Back to {destinations.assistant.name}</a>
      </p>

      <section className="memory-stats" aria-label="Assistant memory status">
        {[
          ["memory", "Reviewed memories", stats?.memories ?? 0],
          ["shield", "Pinned context", stats?.pinned ?? 0],
          ["activity", "Conversations", stats?.conversations ?? 0],
          ["database", "Stored messages", stats?.messages ?? 0],
        ].map(([icon, label, value]) => (
          <div key={String(label)}>
            <span><Icon name={icon as IconName} /></span>
            <small>{label}</small>
            <strong>{value}</strong>
          </div>
        ))}
      </section>

      {notice && <Notice severity="info"><Icon name="shield" />{notice}</Notice>}

      <div className="memory-layout">
        <section className="data-panel memory-create" aria-labelledby="memory-create-title">
          <div className="panel-heading">
            <div>
              <h2 id="memory-create-title">Teach the assistant</h2>
              <p>Add one useful fact, preference, decision, or procedure.</p>
            </div>
            <Icon name="memory" />
          </div>
          <form onSubmit={createMemory}>
            <Textarea
              id="memory-content"
              label="What should Vaelor remember?"
              maxLength={1500}
              onChange={(event) => setContent(event.target.value)}
              placeholder="Example: Prefer beginner-friendly explanations and show the safe default first."
              required
              value={content}
            />
            <div className="memory-form-grid">
              <Select id="memory-kind" label="Kind of memory" value={memoryType} onChange={(event) => setMemoryType(event.target.value)}>
                {memoryTypes.map(([value, label]) => <option key={value} value={value}>{label}</option>)}
              </Select>
              <Select id="memory-scope" label="Where it applies" value={scope} onChange={(event) => setScope(event.target.value)}>
                {memoryScopes.map(([value, label]) => <option key={value} value={value}>{label}</option>)}
              </Select>
            </div>
            <Input id="memory-source" label="Where this came from" maxLength={160} onChange={(event) => setSource(event.target.value)} value={source} />
            <label className="memory-check">
              <input checked={pinned} className="memory-check-input" id="memory-pinned" onChange={(event) => setPinned(event.target.checked)} type="checkbox" />
              <span><strong>Keep in important context</strong><small>Pinned memories are considered on every request.</small></span>
            </label>
            <Button variant="primary" disabled={busy || !content.trim()} type="submit">
              {busy ? "Saving securely…" : "Review and remember"}
            </Button>
          </form>
        </section>

        <section className="data-panel memory-recent" aria-labelledby="memory-recent-title">
          <div className="panel-heading">
            <div>
              <h2 id="memory-recent-title">Recent conversations</h2>
              <p>Assistant requests are stored as named, searchable sessions.</p>
            </div>
            <span className="memory-engine">{stats?.search ?? "FTS5"}</span>
          </div>
          {conversations.length ? (
            <div className="memory-conversation-list">
              <PaginatedItems items={conversations} label="Recent conversations" pageSize={8} render={(conversation) => (
                <div key={conversation.id}>
                  <span><Icon name="activity" size={16} /></span>
                  <span><strong>{conversation.title}</strong><small>{conversation.message_count} messages · {timeAgo(conversation.updated_at * 1000)}</small></span>
                </div>
              )} />
            </div>
          ) : (
            <div className="empty-state"><Icon name="memory" /><strong>No conversations yet</strong><span>Ask the Setup Assistant something to begin one.</span></div>
          )}
        </section>
      </div>

      <section className="memory-library" aria-labelledby="memory-library-title">
        <div className="memory-library__heading">
          <div>
            <span className="page-eyebrow">Curated context</span>
            <h2 id="memory-library-title">Reviewed memories</h2>
          </div>
          <form
            className="memory-search"
            onSubmit={(event) => {
              event.preventDefault();
              void load();
            }}
            role="search"
          >
            <Input className="memory-search__input" id="memory-search" label={<span className="sr-only">Search memories</span>} onChange={(event) => setQuery(event.target.value)} placeholder="Search memories" value={query} />
            <Select className="memory-search__select" id="memory-type-filter" label={<span className="sr-only">Filter by type</span>} onChange={(event) => setTypeFilter(event.target.value)} value={typeFilter}>
              <option value="">All types</option>
              {memoryTypes.map(([value, label]) => <option key={value} value={value}>{label}</option>)}
            </Select>
            <Button type="submit">Search</Button>
          </form>
        </div>

        {memories.length ? (
          <div className="memory-list">
            <PaginatedItems items={memories} label="Assistant memories" pageSize={8} render={(memory) => (
              <article className={memory.pinned ? "memory-card memory-card--pinned" : "memory-card"} key={memory.id}>
                <div className="memory-card__meta">
                  <span>{memory.memory_type.replaceAll("_", " ")}</span>
                  <span>{memory.scope}</span>
                  {memory.pinned && <strong><Icon name="shield" size={12} />Pinned</strong>}
                </div>
                {editingId === memory.id ? (
                  <Textarea className="memory-edit-input"
                    id={`memory-edit-${memory.id}`}
                    label={<span className="sr-only">{`Edit ${memory.memory_type.replaceAll("_", " ")} memory`}</span>}
                    aria-label={`Edit ${memory.memory_type.replaceAll("_", " ")} memory`}
                    maxLength={1500}
                    onChange={(event) => setEditingContent(event.target.value)}
                    value={editingContent}
                  />
                ) : (
                  <p>{memory.content}</p>
                )}
                <footer>
                  <span><small>Source</small><strong>{memory.source}</strong></span>
                  <span><small>Updated</small><strong>{timeAgo(memory.updated_at * 1000)}</strong></span>
                  <div className="memory-card__actions">
                    {editingId === memory.id ? (
                      <>
                        <Button disabled={busy || !editingContent.trim()} onClick={() => void updateMemory(memory.id, { content: editingContent })}>Save</Button>
                        <Button disabled={busy} onClick={() => setEditingId("")} variant="quiet">Cancel</Button>
                      </>
                    ) : (
                      <>
                        <Button disabled={busy} onClick={() => void updateMemory(memory.id, { pinned: !memory.pinned })}>{memory.pinned ? "Unpin" : "Pin"}</Button>
                        <Button disabled={busy} onClick={() => { setEditingId(memory.id); setEditingContent(memory.content); }}>Edit</Button>
                        <Button variant="danger" disabled={busy} onClick={() => void deleteMemory(memory.id)}>
                          {pendingDelete === memory.id ? "Confirm remove" : "Remove"}
                        </Button>
                      </>
                    )}
                  </div>
                </footer>
              </article>
            )} />
          </div>
        ) : (
          <div className="empty-state memory-empty"><Icon name="memory" size={28} /><strong>No matching memories</strong><span>Add a reviewed memory or broaden the search.</span></div>
        )}
      </section>
    </div>
  );
}
