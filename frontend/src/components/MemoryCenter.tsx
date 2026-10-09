import { useCallback, useEffect, useState } from "react";
import type { FormEvent } from "react";
import { ApiError, apiRequest } from "../lib/api";
import { timeAgo } from "../lib/format";
import type { Session } from "../types";
import { useModalAction } from "../hooks/useModalAction";
import { ListPager } from "./assistantListPager";
import { Icon, ICON_SIZE } from "./Icon";
import { ModalShell } from "./ModalShell";
import { usePagination } from "./PaginatedItems";
import { Button, Checkbox, EmptyState, Input, LoadingLines, Notice, PageHeader, Select, StatTile, Textarea } from "./ui";
import { destinations } from "../lib/destinations";
import { hashForPage } from "../lib/navigation";
import { TopbarPageActions } from "../lib/topbarSlot";

/**
 * `/assistant/status`. Every figure is optional here because a reply that
 * leaves one out has not read it: the tile then says "Not read", never 0.
 */
interface MemoryStats {
  memories?: number;
  pinned?: number;
  conversations?: number;
  messages?: number;
  search?: string;
}

/** A tile's value while the first read is still on its way. */
const READING = "Reading…";

/** Memories per page of the Reviewed memories card. */
const MEMORIES_PER_PAGE = 8;

/** A count the server sent, in words; anything else was not read (null). */
function countWords(value: unknown): string | null {
  return typeof value === "number" && Number.isFinite(value) ? String(value) : null;
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
  /*
   * Each of the three reads is held apart, and `null` means "not read". The
   * three used to be one Promise.all into `?? 0`, so a status read that failed
   * drew four zeros and a list read that failed drew "No matching memories":
   * both claims about the store that nobody had measured (LESSONS 1).
   */
  const [stats, setStats] = useState<MemoryStats | null>(null);
  const [memories, setMemories] = useState<AssistantMemory[] | null>(null);
  const [conversations, setConversations] = useState<AssistantConversation[] | null>(null);
  const [loadFailure, setLoadFailure] = useState("");
  // False until the first read has answered: the tiles say "Reading…" until
  // then, never "Not read" for a read still on its way (VD-200 assist review).
  const [firstReadDone, setFirstReadDone] = useState(false);
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
  const [notice, setNoticeText] = useState("");
  // VD-189 (review round 2, N3): a refusal is an alert, never an info notice.
  const [noticeRefused, setNoticeRefused] = useState(false);
  const setNotice = useCallback((message: string, refused = false) => { setNoticeText(message); setNoticeRefused(refused); }, []);
  const memoryPages = usePagination(memories ?? [], MEMORIES_PER_PAGE);

  /**
   * Reads the store again; true when all three reads answered. The search and
   * filter are passed in, so typing in the search box sends nothing until
   * Search is pressed (it used to read the store on every keystroke).
   */
  const load = useCallback(async (search: string, filter: string): Promise<boolean> => {
    const parameters = new URLSearchParams();
    if (search.trim()) parameters.set("query", search.trim());
    if (filter) parameters.set("type", filter);
    const suffix = parameters.size ? `?${parameters.toString()}` : "";
    const [nextStats, nextMemories, nextConversations] = await Promise.allSettled([
      apiRequest<MemoryStats>("/assistant/status"),
      apiRequest<AssistantMemory[]>(`/assistant/memories${suffix}`),
      apiRequest<AssistantConversation[]>("/assistant/conversations?limit=8"),
    ]);
    setStats(nextStats.status === "fulfilled" ? nextStats.value : null);
    setMemories(nextMemories.status === "fulfilled" ? nextMemories.value : null);
    setConversations(nextConversations.status === "fulfilled" ? nextConversations.value : null);
    setFirstReadDone(true);
    const refused = [nextStats, nextMemories, nextConversations]
      .find((outcome): outcome is PromiseRejectedResult => outcome.status === "rejected");
    if (!refused) {
      setLoadFailure("");
      return true;
    }
    // The server's own refusal is worded for the reader; anything else (a
    // dropped connection, a programming error) goes to the log, not the page.
    if (refused.reason instanceof ApiError) {
      setLoadFailure(`Assistant memory could not be loaded. ${refused.reason.message}`);
    } else {
      console.error("Assistant memory could not be loaded", refused.reason);
      setLoadFailure("Assistant memory could not be loaded.");
    }
    return false;
  }, []);

  useEffect(() => {
    void load("", "");
  }, [load]);
  /** Re-reads with the search and filter now on screen. */
  const reload = () => load(query, typeFilter);

  // W5-D8 (LESSONS 5): "Review and remember" saved at once - the label
  // promised a step that did not exist. Submitting now opens the review, which
  // shows exactly what will be stored; only "Remember this" sends it, and a
  // refusal stays inside the review (VD-189) with what was typed kept.
  const [reviewing, setReviewing] = useState(false);
  const save = useModalAction();
  const openReview = (event: FormEvent) => {
    event.preventDefault();
    if (!content.trim()) return;
    save.clear();
    setReviewing(true);
  };
  const closeReview = () => {
    if (save.busy) return;
    save.clear();
    setReviewing(false);
  };

  const createMemory = async () => {
    setBusy(true);
    setNotice("");
    const saved = await save.run(() => apiRequest(
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
      ));
    setBusy(false);
    if (!saved) return;
    setReviewing(false);
    setContent("");
    setNotice("Memory saved. It is available to new assistant requests.");
    if (!await load("", "")) {
      setNotice("Memory saved, but the list could not be read again.", true);
    }
    setQuery("");
    setTypeFilter("");
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
      await reload();
    } catch (error) {
      setNotice(error instanceof Error ? error.message : "Memory could not be updated.", true);
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
      await reload();
    } catch (error) {
      setNotice(error instanceof Error ? error.message : "Memory could not be removed.", true);
    } finally {
      setBusy(false);
    }
  };

  const saveDisabledReason = content.trim() ? undefined : "Write what to remember first";

  return (
    <div className="memory-page mp-page">
      {/*
        * Memory is appliance-wide, not the Assistant's. `_list_memories` has no
        * actor filter, so the same store answers both AI surfaces — and while
        * this lived as a tab inside the Assistant it read as the Assistant's
        * private notebook, which is a claim the server never made. The heading
        * says who uses it so nobody has to read the storage layer to find out.
        */}
      {/* In the top bar (the MemoryPage board). */}
      <TopbarPageActions>
        {/* A fact about where the store lives, not a reading: outline and grey. */}
        <span className="mp-local"><Icon name="shield" size={ICON_SIZE.inline} />Local &amp; private</span>
        <a className="ui-button ui-button--quiet as-btn-ghost mp-back" href={hashForPage("assistant")}>Back to {destinations.assistant.name}</a>
      </TopbarPageActions>
      <PageHeader
        subtitle={<>Review exactly what Vaelor can remember. Secrets and instruction overrides are blocked automatically. Both the {destinations.assistant.name} and {destinations["ai-chat"].name} read it.</>}
        title="What Vaelor remembers"
      />

      <section className="mp-stats" aria-label="Assistant memory status">
        <StatTile label="Reviewed memories" value={firstReadDone ? countWords(stats?.memories) : READING} />
        <StatTile label="Pinned context" value={firstReadDone ? countWords(stats?.pinned) : READING} />
        <StatTile label="Conversations" value={firstReadDone ? countWords(stats?.conversations) : READING} />
        <StatTile label="Stored messages" value={firstReadDone ? countWords(stats?.messages) : READING} />
      </section>

      {loadFailure && <Notice severity="danger">{loadFailure}</Notice>}
      {notice && <Notice severity={noticeRefused ? "danger" : "success"}>{notice}</Notice>}

      <div className="mp-split">
        <section aria-labelledby="memory-create-title" className="card ui-card mp-teach">
          <header className="ui-card__header">
            <div className="ui-card__titles">
              <h2 id="memory-create-title">Teach the assistant</h2>
              <p>Add one useful fact, preference, decision, or procedure.</p>
            </div>
          </header>
          <form className="mp-teach__form" onSubmit={openReview}>
            <Textarea
              id="memory-content"
              label="What should Vaelor remember?"
              maxLength={1500}
              onChange={(event) => setContent(event.target.value)}
              placeholder="Example: Prefer beginner-friendly explanations and show the safe default first."
              required
              value={content}
            />
            <div className="as-grid2 mp-teach__grid">
              <Select id="memory-kind" label="Kind of memory" value={memoryType} onChange={(event) => setMemoryType(event.target.value)}>
                {memoryTypes.map(([value, label]) => <option key={value} value={value}>{label}</option>)}
              </Select>
              <Select id="memory-scope" label="Where it applies" value={scope} onChange={(event) => setScope(event.target.value)}>
                {memoryScopes.map(([value, label]) => <option key={value} value={value}>{label}</option>)}
              </Select>
            </div>
            <Input id="memory-source" label="Where this came from" maxLength={160} onChange={(event) => setSource(event.target.value)} value={source} />
            {/* Named by its title, described by its help: the two ran together
                as one name with no separator (W5-D8 info). */}
            <div className="mp-teach__check">
              <Checkbox
                checked={pinned}
                hint="Pinned memories are considered on every request."
                id="memory-pinned"
                label="Keep in important context"
                onChange={(event) => setPinned(event.target.checked)}
              />
            </div>
            <div className="mp-teach__submit">
              <Button disabled={busy} disabledReason={saveDisabledReason} type="submit" variant="primary">
                Review and remember
              </Button>
            </div>
          </form>
          {reviewing && (
            <ModalShell className="as-dialog mp-review" error={save.error} labelledBy="memory-review-title" onClose={closeReview}>
              <header className="as-dialog__head">
                <div>
                  <h2 id="memory-review-title">Remember this?</h2>
                  <p>Vaelor stores this on this appliance. Both the Assistant and AI Chat read it from new requests on.</p>
                </div>
              </header>
              <div className="as-dialog__body">
                <blockquote className="mp-review__content">{content}</blockquote>
                <dl className="as-kv">
                  <div><dt>Kind</dt><dd>{memoryTypes.find(([value]) => value === memoryType)?.[1] ?? memoryType}</dd></div>
                  <div><dt>Applies</dt><dd>{memoryScopes.find(([value]) => value === scope)?.[1] ?? scope}</dd></div>
                  <div><dt>Source</dt><dd>{source.trim() || "Not given"}</dd></div>
                  <div><dt>Context</dt><dd>{pinned ? "Pinned: considered on every request" : "Considered when a request is about it"}</dd></div>
                </dl>
              </div>
              <footer className="as-dialog__foot">
                <Button disabled={save.busy} onClick={closeReview} type="button">Back to edit</Button>
                <Button busy={save.busy} onClick={() => void createMemory()} type="button" variant="primary">Remember this</Button>
              </footer>
            </ModalShell>
          )}
        </section>

        <section aria-labelledby="memory-recent-title" className="card ui-card mp-recent">
          <header className="ui-card__header">
            <div className="ui-card__titles">
              <h2 id="memory-recent-title">Recent conversations</h2>
              <p>Assistant requests are stored as named, searchable sessions.</p>
            </div>
            {/* The search engine as the server reports it; no tag when the status was not read. */}
            {stats?.search && <span className="mp-tag">{stats.search}</span>}
          </header>
          {conversations === null ? (
            loadFailure
              ? <p className="as-small as-muted mp-unread">Not read</p>
              : <LoadingLines label="Loading recent conversations" />
          ) : conversations.length ? (
            <div className="mp-conversations">
              {conversations.map((conversation) => (
                <div className="mp-conversation" key={conversation.id}>
                  <span aria-hidden="true" className="mp-conversation__icon"><Icon name="chat" size={16} /></span>
                  <div>
                    <div className="mp-conversation__title">{conversation.title}</div>
                    <div className="as-small as-muted">{conversation.message_count} messages · {timeAgo(conversation.updated_at * 1000)}</div>
                  </div>
                </div>
              ))}
            </div>
          ) : (
            <EmptyState text="Ask the Setup Assistant something to begin one." title="No conversations yet" />
          )}
        </section>
      </div>

      <section aria-labelledby="memory-library-title" className="card ui-card mp-library">
        <header className="ui-card__header mp-library__header">
          <div className="ui-card__titles">
            <span className="as-label">Curated context</span>
            <h2 id="memory-library-title">Reviewed memories</h2>
          </div>
          <form
            className="mp-search"
            onSubmit={(event) => {
              event.preventDefault();
              void reload();
            }}
            role="search"
          >
            <Input className="mp-search__input" id="memory-search" label={<span className="sr-only">Search memories</span>} onChange={(event) => setQuery(event.target.value)} placeholder="Search memories" value={query} />
            <Select className="mp-search__select" id="memory-type-filter" label={<span className="sr-only">Filter by type</span>} onChange={(event) => { setTypeFilter(event.target.value); void load(query, event.target.value); }} value={typeFilter}>
              <option value="">All types</option>
              {memoryTypes.map(([value, label]) => <option key={value} value={value}>{label}</option>)}
            </Select>
            <Button className="mp-search__button" type="submit"><Icon name="search" size={16} />Search</Button>
          </form>
        </header>

        {memories === null ? (
          loadFailure
            ? <p className="as-small as-muted mp-unread">Reviewed memories · not read</p>
            : <LoadingLines label="Loading reviewed memories" />
        ) : memories.length ? (
          <div className="mp-memories">
            {memoryPages.visible.map((memory) => (
              <article className={memory.pinned ? "mp-memory mp-memory--pinned" : "mp-memory"} key={memory.id}>
                <div className="mp-memory__tags">
                  <span className="mp-tag">{memory.memory_type.replaceAll("_", " ")}</span>
                  <span className="mp-tag">{memory.scope}</span>
                  {memory.pinned && <span className="mp-pinned"><Icon name="shield" size={ICON_SIZE.inline} />Pinned</span>}
                </div>
                {editingId === memory.id ? (
                  <Textarea className="mp-memory__edit"
                    id={`memory-edit-${memory.id}`}
                    label={<span className="sr-only">{`Edit ${memory.memory_type.replaceAll("_", " ")} memory`}</span>}
                    aria-label={`Edit ${memory.memory_type.replaceAll("_", " ")} memory`}
                    maxLength={1500}
                    onChange={(event) => setEditingContent(event.target.value)}
                    value={editingContent}
                  />
                ) : (
                  <p className="mp-memory__text">{memory.content}</p>
                )}
                <div className="mp-memory__foot">
                  <span className="as-small"><span className="as-muted">Source</span> {memory.source}</span>
                  <span className="as-small"><span className="as-muted">Updated</span> {timeAgo(memory.updated_at * 1000)}</span>
                  <div className="mp-memory__actions">
                    {editingId === memory.id ? (
                      <>
                        <Button disabled={busy || !editingContent.trim()} onClick={() => void updateMemory(memory.id, { content: editingContent })} variant="primary">Save</Button>
                        <Button className="as-btn-ghost" disabled={busy} onClick={() => setEditingId("")} variant="quiet">Cancel</Button>
                      </>
                    ) : (
                      <>
                        <Button disabled={busy} onClick={() => void updateMemory(memory.id, { pinned: !memory.pinned })}>{memory.pinned ? "Unpin" : "Pin"}</Button>
                        <Button disabled={busy} onClick={() => { setEditingId(memory.id); setEditingContent(memory.content); }}>Edit</Button>
                        {/* The first press arms the remove; the second, now a solid
                            red "Confirm remove", deletes. */}
                        <Button
                          className={pendingDelete === memory.id ? undefined : "as-btn-danger"}
                          disabled={busy}
                          onClick={() => void deleteMemory(memory.id)}
                          variant={pendingDelete === memory.id ? "danger" : "secondary"}
                        >
                          {pendingDelete === memory.id ? "Confirm remove" : "Remove"}
                        </Button>
                      </>
                    )}
                  </div>
                </div>
              </article>
            ))}
            <ListPager
              label="Assistant memories"
              layout="ends"
              page={memoryPages.page}
              setPage={memoryPages.setPage}
              totalItems={memories.length}
              totalPages={memoryPages.totalPages}
            />
          </div>
        ) : (
          <EmptyState
            icon={<Icon name="memory" size={18} />}
            text="Add a reviewed memory or broaden the search."
            title="No matching memories"
          />
        )}
      </section>
    </div>
  );
}
