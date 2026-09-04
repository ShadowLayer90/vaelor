import { type FormEvent, useCallback, useEffect, useRef, useState } from "react";
import { apiRequest, ApiError } from "../lib/api";
import { modelDisplayName } from "../lib/modelIdentity";
import type { Session } from "../types";
import { AiChatComposer } from "./AiChatComposer";
import { AiChatDetails } from "./AiChatDetails";
import { blamedModel } from "./AiChatModelPicker";
import { AiChatRail } from "./AiChatRail";
import { AiChatToolbar } from "./AiChatToolbar";
import { AiChatThread, type AiChatRetrieval, type AiChatThreadMessage } from "./AiChatThread";
import { DROPPED_CONNECTION_NOTICE, droppedConnectionSubject, useResumedAnswer } from "../hooks/useResumedAnswer";
import type {
  AiChatCollection,
  AiChatConnection,
  AiChatConversation,
  AiChatDocument,
  AiChatAgent,
  AiChatAgentProposal,
  AiChatSetup,
} from "./aiChatTypes";
import { ConfirmDialog } from "./ConfirmDialog";
import { TextPromptDialog } from "./TextPromptDialog";
import { Button, Notice } from "./ui";
import { destinations } from "../lib/destinations";
import { conversationIdFromHash, useAiChatRoute } from "../hooks/useAiChatRoute";

type DeleteTarget = {
  type: "conversation" | "collection";
  id: string;
  name: string;
} | null;

/*
 * The browser must outlast the appliance, never the other way round. The
 * server's own budget is VAELOR_INFERENCE_TIMEOUT_SECONDS (240 by default,
 * 900 at most), so a client that gave up at 130 seconds reported a timeout
 * for a request the appliance was still working on - and because the failure
 * was invented in the browser, nothing was written to the conversation and
 * the answer vanished on reload. This sits above the server's ceiling so the
 * appliance always gets to state the outcome itself.
 */
const AI_CHAT_CLIENT_TIMEOUT_MS = 930_000;

/* Moved to `AiChatModelPicker.tsx`; re-exported so callers keep their import. */
export { blamedModel, modelOptionLabel } from "./AiChatModelPicker";
/* Routing moved to `useAiChatRoute`; re-exported so callers keep their import. */
export { conversationIdFromHash } from "../hooks/useAiChatRoute";

export function AiChat({ session }: { session: Session }) {
  const [setup, setSetup] = useState<AiChatSetup | null>(null);
  const [models, setModels] = useState<string[]>([]);
  const [selectedModel, setSelectedModel] = useState("");
  const [selectedCollections, setSelectedCollections] = useState<string[]>([]);
  const [conversations, setConversations] = useState<AiChatConversation[]>([]);
  const [conversationId, setConversationId] = useState(
    () => conversationIdFromHash(window.location.hash),
  );
  const [openRecord, setOpenRecord] = useState<AiChatConversation | null>(null);
  const [messages, setMessages] = useState<AiChatThreadMessage[]>([]);
  const [input, setInput] = useState("");
  const [search, setSearch] = useState("");
  const [showArchived, setShowArchived] = useState(false);
  const [temporary, setTemporary] = useState(false);
  const [detailsOpen, setDetailsOpen] = useState(false);
  const [focusMode, setFocusMode] = useState(false);
  const [activeCollection, setActiveCollection] = useState("");
  const [documents, setDocuments] = useState<AiChatDocument[]>([]);
  const [collectionName, setCollectionName] = useState("");
  const [collectionDescription, setCollectionDescription] = useState("");
  const [busy, setBusy] = useState("");
  const [notice, setNotice] = useState("");
  const [renameTitle, setRenameTitle] = useState<string | null>(null);
  const [deleteTarget, setDeleteTarget] = useState<DeleteTarget>(null);
  const [awaitingAnswer, setAwaitingAnswer] = useState(false);
  const [agentProposal, setAgentProposal] = useState<AiChatAgentProposal | null>(null);
  const [agentRunSubmitted, setAgentRunSubmitted] = useState(false);
  const [agents, setAgents] = useState<AiChatAgent[]>([]);
  /*
   * Whether `/assistant/profiles` has answered, however it answered. An empty
   * list before it does is not the same fact as an appliance with no agents,
   * and the Agent select is not allowed to state the second while holding the
   * first — the same rule the model picker's `reading` state exists for.
   */
  const [agentsResolved, setAgentsResolved] = useState(false);
  const [selectedAgentId, setSelectedAgentId] = useState("");
  /*
   * Models the server could not run, and what it said about each.
   *
   * The picker lists everything the provider reports and nothing distinguishes
   * a loaded model from an unloaded one - `/ai-chat/connections/<id>/models`
   * returns bare ids - so choosing one that is not loaded produced "LM Studio
   * rejected the chat request (HTTP 400)" with the picker still sitting
   * silently on it. What the appliance has told us cannot run is remembered
   * here and shown against the model itself until it answers.
   */
  const [modelFailures, setModelFailures] = useState<Record<string, string>>({});
  /*
   * Which thread the user is looking at right now. A reply belongs to the
   * conversation it was sent from: switching threads mid-request used to
   * append the in-flight answer to whatever was on screen when it landed, so
   * an answer appeared under a question that was never asked there. The
   * counter changes on every switch, and a reply whose ticket no longer
   * matches is dropped rather than shown in the wrong place.
   */
  const threadTicket = useRef(0);
  /*
   * The same ticket, in state, so the view can be rendered against it. The
   * "Generating with …" indicator was drawn from `busy` and the model picker -
   * both properties of whoever is looking - so a reply in flight in one chat
   * announced itself in whichever chat was opened next, under whichever model
   * name the picker had moved on to.
   */
  const [viewTicket, setViewTicket] = useState(0);
  const [pending, setPending] = useState<{ ticket: number; model: string } | null>(null);
  const openThread = useCallback(() => {
    threadTicket.current += 1;
    setViewTicket(threadTicket.current);
    return threadTicket.current;
  }, []);
  const abortSend = useRef<AbortController | null>(null);
  const stoppedByUser = useRef(false);

  const loadConversations = useCallback(async () => {
    const params = new URLSearchParams();
    if (showArchived) params.set("archived", "1");
    if (search.trim()) params.set("q", search.trim());
    const suffix = params.toString() ? `?${params}` : "";
    setConversations(await apiRequest<AiChatConversation[]>(`/ai-chat/conversations${suffix}`));
  }, [search, showArchived]);

  const loadSetup = useCallback(async (preserveSelection = false) => {
    const next = await apiRequest<AiChatSetup>("/ai-chat/setup");
    setSetup(next);
    if (!preserveSelection) {
      setSelectedModel(next.preference.model);
      setSelectedCollections(next.preference.collection_ids);
    }
    if (!activeCollection && next.collections.length) setActiveCollection(next.collections[0].id);
    if (next.active_connection) {
      const result = await apiRequest<{ models: string[] }>(
        `/ai-chat/connections/${next.active_connection.id}/models`,
      );
      const available = result.models ?? [];
      setModels(available);
      /*
       * Reconcile the remembered model against what this connection now offers.
       * The preference is a bare string saved from before; a model named there
       * when a different local server answered AI Chat ("gpt-oss-sg:20b") is a
       * phantom once a GPU server offering only Qwen is dedicated to the chat.
       * Sending against it draws a rejection, and the caption named a model the
       * dropdown could not even select. When the stored model is absent (or
       * unset), fall back to what the server reports active, else the first
       * available — the same model the picker now shows.
       */
      if (!preserveSelection && available.length && !available.includes(next.preference.model)) {
        const active = next.active_connection.selected_model ?? "";
        setSelectedModel(available.includes(active) ? active : available[0]);
      }
    } else {
      setModels([]);
    }
  }, [activeCollection]);

  useEffect(() => {
    void loadSetup().catch((error) => {
      setNotice(error instanceof Error ? error.message : "AI Chat could not be loaded.");
    });
    void apiRequest<AiChatAgent[]>("/assistant/profiles")
      .then((items) => setAgents(items.filter((item) => item.custom && item.enabled !== false)))
      .catch(() => setAgents([]))
      .finally(() => setAgentsResolved(true));
  }, [loadSetup]);

  useEffect(() => {
    const timer = window.setTimeout(() => {
      void loadConversations().catch((error) => {
        setNotice(error instanceof Error ? error.message : "Chat history could not be loaded.");
      });
    }, search ? 250 : 0);
    return () => window.clearTimeout(timer);
  }, [loadConversations, search]);

  useEffect(() => {
    if (!activeCollection) {
      setDocuments([]);
      return;
    }
    void apiRequest<AiChatDocument[]>(`/ai-chat/collections/${activeCollection}/documents`)
      .then(setDocuments)
      .catch(() => setDocuments([]));
  }, [activeCollection]);

  useEffect(() => {
    if (!focusMode) return;
    const leaveFocus = (event: KeyboardEvent) => {
      if (event.key === "Escape") setFocusMode(false);
    };
    document.addEventListener("keydown", leaveFocus);
    return () => document.removeEventListener("keydown", leaveFocus);
  }, [focusMode]);

  const saveContext = async (model: string, collections: string[]) => {
    await apiRequest(
      "/ai-chat/preferences",
      { method: "PATCH", body: JSON.stringify({ model, collection_ids: collections }) },
      session.csrf_token,
    );
    if (conversationId && !temporary) {
      await apiRequest(
        `/ai-chat/conversations/${conversationId}`,
        { method: "PATCH", body: JSON.stringify({ model, collections }) },
        session.csrf_token,
      );
    }
  };

  const activateConnection = async (connection: AiChatConnection) => {
    setBusy("connection"); setNotice("");
    try {
      await apiRequest(
        `/ai-chat/connections/${connection.id}/activate`,
        { method: "POST", body: "{}" },
        session.csrf_token,
      );
      setNotice(`${connection.label} is now dedicated to AI Chat.`);
      await loadSetup();
    } catch (error) {
      setNotice(error instanceof Error ? error.message : "Connection could not be activated.");
    } finally { setBusy(""); }
  };

  const chooseModel = async (model: string) => {
    setSelectedModel(model);
    try {
      await saveContext(model, selectedCollections);
      // The name the reader picked, not the file the server loads it from.
      setNotice(`This chat will use ${modelDisplayName(model)}.`);
    } catch (error) {
      setNotice(error instanceof Error ? error.message : "Model choice could not be saved.");
    }
  };

  const toggleCollection = async (id: string) => {
    const next = selectedCollections.includes(id)
      ? selectedCollections.filter((item) => item !== id)
      : [...selectedCollections, id];
    setSelectedCollections(next);
    try {
      await saveContext(selectedModel, next);
    } catch (error) {
      setNotice(error instanceof Error ? error.message : "Knowledge choice could not be saved.");
    }
  };

  const openConversation = async (item: AiChatConversation) => {
    openThread();
    setTemporary(false);
    setConversationId(item.id);
    setOpenRecord(item);
    setSelectedModel(item.model || setup?.preference.model || "");
    /*
     * Resuming a chat must never silently switch retrieval off. A conversation
     * started before any collection existed carries an empty list; adopting it
     * cleared the live selection, and the next request wrote that empty list
     * back over the conversation, so the chat could never search again and the
     * only clue was "General model knowledge" in small text. An empty stored
     * selection now falls back to the account preference instead.
     */
    const stored = item.collections ?? [];
    const preferred = setup?.preference.collection_ids ?? [];
    setSelectedCollections(
      stored.length ? stored : preferred.length ? preferred : selectedCollections,
    );
    const nextMessages = await apiRequest<AiChatThreadMessage[]>(
      `/ai-chat/conversations/${item.id}/messages`,
    );
    setMessages(nextMessages);
    /*
     * A stored transcript that ends on a question is a question the appliance
     * is still answering: it writes the turn before inference returns, so a
     * reload mid-answer restored the question with nothing after it and nothing
     * to say a reply was coming.
     */
    setAwaitingAnswer(nextMessages.at(-1)?.role === "user");
    const savedProposal = [...nextMessages].reverse().find(
      (message) => message.metadata?.proposed_agent_task,
    )?.metadata?.proposed_agent_task ?? null;
    setAgentProposal(savedProposal);
    setSelectedAgentId(savedProposal?.profile_id ?? "");
    setAgentRunSubmitted(false);
  };

  /*
   * Keep saying so until the answer lands. The appliance finishes the reply
   * server-side after a reload, and re-reading the conversation is the only
   * way this browser learns that; without it the reader re-asks.
   */
  const resumed = useResumedAnswer({
    armed: awaitingAnswer && Boolean(conversationId) && !temporary && busy !== "chat",
    subject: conversationId,
    poll: async () => {
      const ticket = threadTicket.current;
      const next = await apiRequest<AiChatThreadMessage[]>(
        `/ai-chat/conversations/${conversationId}/messages`,
      );
      if (ticket !== threadTicket.current) return true;
      if (next.at(-1)?.role !== "assistant") return false;
      setMessages(next);
      setAwaitingAnswer(false);
      return true;
    },
  });

  const newConversation = (isTemporary = false) => {
    openThread();
    setTemporary(isTemporary);
    setConversationId("");
    setOpenRecord(null);
    setAwaitingAnswer(false);
    setMessages([]);
    setInput("");
    setAgentProposal(null);
    setAgentRunSubmitted(false);
    setSelectedAgentId("");
    setNotice(isTemporary ? "Temporary chat ready. Nothing in this chat will be saved." : "");
  };

  // The address bar and the open conversation are kept in step here: the URL
  // names the open chat so a reload or shared link returns to it, a URL that
  // already names one reopens it, and a nav click or Back/Forward is honoured
  // rather than silently dropping the conversation (#164).
  useAiChatRoute({
    conversationId,
    conversations,
    hasMessages: messages.length > 0,
    onNotice: setNotice,
    openConversation,
    temporary,
  });

  /*
   * Archiving and restoring both empty the screen, so both have to say what
   * they did with it. They used to drop the reader into a blank "New chat"
   * with no message at all: the transcript they were reading vanished and
   * nothing said where it had gone or how to get it back.
   */
  const archiveConversation = async () => {
    if (!conversationId) return;
    const name = conversations.find((item) => item.id === conversationId)?.title
      ?? openRecord?.title ?? "That chat";
    const restoring = showArchived;
    await apiRequest(
      `/ai-chat/conversations/${conversationId}`,
      { method: "PATCH", body: JSON.stringify({ archived: !restoring }) },
      session.csrf_token,
    );
    newConversation();
    await loadConversations();
    setNotice(
      restoring
        ? `${name} restored. It is back in your chat list, and this is a new chat.`
        : `${name} archived. Open "Archive" in the chat list to read or restore it. This is a new chat.`,
    );
  };

  const renameConversation = async () => {
    if (!conversationId || !renameTitle?.trim()) return;
    setBusy("rename");
    try {
      await apiRequest(
        `/ai-chat/conversations/${conversationId}`,
        { method: "PATCH", body: JSON.stringify({ title: renameTitle.trim() }) },
        session.csrf_token,
      );
      setRenameTitle(null);
      await loadConversations();
    } finally { setBusy(""); }
  };

  const exportConversation = () => {
    const conversation = conversations.find((item) => item.id === conversationId);
    if (!conversation) return;
    const lines = [
      `# ${conversation.title}`, "",
      `Model: ${conversation.model || "Not recorded"}`, "",
      ...messages.flatMap((message) => [
        `## ${message.role === "user" ? "You" : "Vaelor AI"}`,
        "", message.content, "",
      ]),
    ];
    const link = document.createElement("a");
    link.href = URL.createObjectURL(new Blob([lines.join("\n")], { type: "text/markdown" }));
    link.download = `${conversation.title.replace(/[^a-z0-9]+/gi, "-").replace(/^-|-$/g, "").toLowerCase() || "ai-chat"}.md`;
    link.click();
    URL.revokeObjectURL(link.href);
  };

  const forkConversation = async (message: AiChatThreadMessage) => {
    if (!conversationId || !message.id) return;
    setBusy("chat");
    try {
      const branch = await apiRequest<AiChatConversation>(
        `/ai-chat/conversations/${conversationId}/fork`,
        { method: "POST", body: JSON.stringify({ through_message_id: message.id }) },
        session.csrf_token,
      );
      await loadConversations();
      await openConversation(branch);
      setNotice("Branch created. The original chat is unchanged.");
    } catch (error) {
      setNotice(error instanceof Error ? error.message : "The branch could not be created.");
    } finally { setBusy(""); }
  };

  /**
   * One failure, reported one way.
   *
   * Every failed question leaves the same three things behind: the failed
   * exchange in the transcript, the question back in the composer, and a
   * banner. What the banner could not do until now was say which model failed,
   * so a picker full of models the appliance cannot run gave the same sentence
   * whichever one produced it.
   */
  const reportFailure = (error: unknown, model: string, failure: string) => {
    const blamed = blamedModel(error, model);
    if (!blamed) return failure;
    setModelFailures((current) => ({ ...current, [blamed]: failure }));
    return `${blamed} could not run this request. ${failure}`;
  };

  const clearModelFailure = (model: string) => {
    if (!model) return;
    setModelFailures((current) => {
      if (!(model in current)) return current;
      const next = { ...current };
      delete next[model];
      return next;
    });
  };

  const regenerate = async (message: AiChatThreadMessage, content?: string) => {
    if (!conversationId || !message.id) return;
    const ticket = threadTicket.current;
    const requestModel = selectedModel;
    setBusy("chat"); setNotice("");
    setPending({ ticket, model: requestModel });
    try {
      const result = await apiRequest<{
        messages: AiChatThreadMessage[]; model: string; provider: string;
        retrieval?: AiChatRetrieval;
      }>(
        `/ai-chat/conversations/${conversationId}/regenerate`,
        {
          method: "POST",
          body: JSON.stringify({ message_id: message.id, content }),
          timeoutMs: AI_CHAT_CLIENT_TIMEOUT_MS,
        },
        session.csrf_token,
      );
      setMessages(result.messages);
      clearModelFailure(result.model || requestModel);
      setNotice(`${result.provider} · ${result.model}`);
      await loadConversations();
    } catch (error) {
      const failure = error instanceof Error
        ? error.message
        : "The answer could not be regenerated.";
      setNotice(reportFailure(error, requestModel, failure));
      /*
       * Re-asking a stored question is a send by another route, so it ends the
       * same way a send does: the question comes back to the composer. It used
       * to leave nothing but a banner, so a first failure and a failed retry of
       * the same question put the words in two different places.
       */
      if (message.role === "user") setInput((current) => current || (content ?? message.content));
    } finally {
      setBusy("");
      setPending(null);
    }
  };

  const createCollection = async (event: FormEvent) => {
    event.preventDefault(); setBusy("collection"); setNotice("");
    try {
      const created = await apiRequest<AiChatCollection>(
        "/ai-chat/collections",
        { method: "POST", body: JSON.stringify({ name: collectionName, description: collectionDescription }) },
        session.csrf_token,
      );
      setCollectionName(""); setCollectionDescription(""); setActiveCollection(created.id);
      const next = [...new Set([...selectedCollections, created.id])];
      setSelectedCollections(next);
      await saveContext(selectedModel, next);
      await loadSetup(true);
      setNotice("Knowledge collection created and enabled.");
    } catch (error) {
      setNotice(error instanceof Error ? error.message : "Collection could not be created.");
    } finally { setBusy(""); }
  };

  /*
   * Attaching a file has to work on a box that has never had one. Before this,
   * a first-time user pressing "+" was sent to a panel whose only create form
   * sat behind a collapsed disclosure, with no upload control rendered at all
   * because nothing was selected yet - so the button read as dead, and there
   * was no reachable way to give the assistant a document to read.
   */
  const ensureCollection = async () => {
    if (activeCollection) return activeCollection;
    const created = await apiRequest<AiChatCollection>(
      "/ai-chat/collections",
      {
        method: "POST",
        body: JSON.stringify({
          name: "My documents",
          description: "Files you add so Vaelor AI can answer from them.",
        }),
      },
      session.csrf_token,
    );
    setActiveCollection(created.id);
    const next = [...new Set([...selectedCollections, created.id])];
    setSelectedCollections(next);
    await saveContext(selectedModel, next);
    return created.id;
  };

  const ingestFile = async (file: File) => {
    setBusy("document"); setNotice("");
    try {
      if (file.size > (setup?.limits.document_bytes ?? 2 * 1024 * 1024)) {
        throw new Error("This file exceeds the 2 MiB document limit.");
      }
      if (!/\.(txt|md|markdown|json|csv|ya?ml|log)$/i.test(file.name)) {
        throw new Error("Use a text, Markdown, JSON, CSV, YAML, or log file.");
      }
      const collectionId = await ensureCollection();
      await apiRequest(
        `/ai-chat/collections/${collectionId}/documents`,
        {
          method: "POST",
          body: JSON.stringify({
            name: file.name,
            content: await file.text(),
            media_type: file.type || "text/plain",
          }),
          timeoutMs: 30000,
        },
        session.csrf_token,
      );
      setDocuments(await apiRequest<AiChatDocument[]>(
        `/ai-chat/collections/${collectionId}/documents`,
      ));
      await loadSetup(true);
      /*
       * Name the collection the file landed in. "+" adds to whichever
       * collection happens to be selected, which on a shared appliance is
       * whichever one the last person left selected - so a file could join
       * somebody else's collection with nothing on screen naming it.
       */
      const destination = setup?.collections.find((item) => item.id === collectionId)?.name
        ?? "My documents";
      setNotice(`${file.name} added to ${destination}. Vaelor can now cite it in answers.`);
    } catch (error) {
      setNotice(error instanceof Error ? error.message : "Document could not be indexed.");
    } finally { setBusy(""); }
  };

  const removeDocument = async (id: string) => {
    setBusy("document");
    try {
      await apiRequest(`/ai-chat/documents/${id}`, { method: "DELETE" }, session.csrf_token);
      setDocuments((current) => current.filter((item) => item.id !== id));
      await loadSetup(true);
      setNotice("Document deleted.");
    } finally { setBusy(""); }
  };

  const confirmDelete = async () => {
    if (!deleteTarget) return;
    setBusy("delete");
    try {
      const path = deleteTarget.type === "conversation"
        ? `/ai-chat/conversations/${deleteTarget.id}`
        : `/ai-chat/collections/${deleteTarget.id}`;
      await apiRequest(path, { method: "DELETE" }, session.csrf_token);
      if (deleteTarget.type === "conversation") newConversation();
      if (deleteTarget.type === "collection") {
        setSelectedCollections((current) => current.filter((id) => id !== deleteTarget.id));
        setActiveCollection(""); setDocuments([]);
      }
      setDeleteTarget(null);
      await Promise.all([loadConversations(), loadSetup(true)]);
      setNotice(`${deleteTarget.name} deleted.`);
    } catch (error) {
      setNotice(error instanceof Error ? error.message : "The item could not be deleted.");
    } finally { setBusy(""); }
  };

  const reviewAgentRun = async () => {
    if (!agentProposal || agentRunSubmitted) return;
    setBusy("agent-run");
    setNotice("");
    try {
      await apiRequest<{ id?: string }>(
        "/assistant/tasks",
        {
          method: "POST",
          body: JSON.stringify({
            title: `${agentProposal.profile_name} chat request`,
            description: agentProposal.task,
            profile: agentProposal.profile_id,
            profile_version: agentProposal.profile_version,
            approval_required: true,
            idempotency_key: globalThis.crypto?.randomUUID?.() ?? `ai-chat-${Date.now()}`,
          }),
        },
        session.csrf_token,
      );
      setAgentRunSubmitted(true);
      /*
       * Assistant › Agents is administrator-only. Naming it to anyone else sent
       * them looking for a tab that is not in their tablist; the run is in the
       * Assistant's own run history either way, under the Agent runs filter.
       * `task.id` used to pick between two identical strings.
       */
      setNotice(
        session.user.role === "administrator"
          ? "Saved. Open it under Assistant › Agents, in this agent's run history, to approve and run it."
          : "Saved. Open it under Assistant, in the Agent runs filter, to approve and run it.",
      );
    } catch (error) {
      setNotice(error instanceof Error ? error.message : "The agent run could not be prepared.");
    } finally {
      setBusy("");
    }
  };

  /*
   * This used to offer the "Assistant task ledger" and land on the Assistant's
   * chat tab. No ledger exists in this product, and the run the user was sent
   * to find was on a different tab behind a collapsed disclosure - so the one
   * button offered after a matched agent request led nowhere useful.
   */
  const openAgentRun = () => {
    window.dispatchEvent(new CustomEvent("pironman:navigate", { detail: "assistant" }));
    window.dispatchEvent(new CustomEvent("vaelor:assistant-tab", { detail: "customAgents" }));
  };

  const stopGeneration = useCallback(() => {
    stoppedByUser.current = true;
    abortSend.current?.abort();
  }, []);

  /**
   * Ask a question and place its answer after `priorMessages`.
   *
   * Retry re-enters here with the turns from before the question that failed,
   * so a second attempt replaces the failed exchange instead of stacking a
   * duplicate of the question under it.
   */
  const submitQuestion = async (question: string, priorMessages: AiChatThreadMessage[]) => {
    const ticket = threadTicket.current;
    const requestModel = selectedModel;
    const controller = new AbortController();
    abortSend.current = controller;
    stoppedByUser.current = false;
    setBusy("chat"); setNotice("");
    setAwaitingAnswer(false); setPending({ ticket, model: requestModel });
    setMessages([...priorMessages, { role: "user", content: question }]);
    try {
      const result = await apiRequest<{
        conversation_id: string;
        message: AiChatThreadMessage;
        model: string;
        provider: string;
        retrieval?: AiChatRetrieval;
        proposed_agent_task?: AiChatAgentProposal;
        approval_required?: boolean;
      }>(
        "/ai-chat/messages",
        {
          method: "POST",
          body: JSON.stringify({
            message: question,
            conversation_id: temporary ? undefined : conversationId || undefined,
            model: selectedModel,
            collection_ids: selectedCollections,
            temporary,
            history: temporary ? priorMessages : undefined,
            agent_id: selectedAgentId || undefined,
            idempotency_key: globalThis.crypto?.randomUUID?.() ?? `ai-chat-${Date.now()}`, // VD-112: one key per send, reused by the api.ts transport retry so the appliance dedupes the turn.
          }),
          timeoutMs: AI_CHAT_CLIENT_TIMEOUT_MS,
          signal: controller.signal,
        },
        session.csrf_token,
      );
      if (ticket !== threadTicket.current) {
        setNotice("That answer finished after you moved to another chat. It was saved where you asked it.");
        await loadConversations();
        return;
      }
      setConversationId(result.conversation_id);
      // It answered, so whatever it did last time is no longer true of it.
      clearModelFailure(result.model || requestModel);
      setMessages((current) => [
        ...current,
        { ...result.message, retrieval: result.message.retrieval ?? result.retrieval },
      ]);
      if (result.proposed_agent_task) {
        setAgentProposal(result.proposed_agent_task);
        setAgentRunSubmitted(false);
      } else {
        setAgentProposal(null);
        setAgentRunSubmitted(false);
      }
      setNotice(
        result.proposed_agent_task
          ? "Agent proposal ready. Nothing has run."
          : `${result.provider} · ${result.model}${temporary ? " · not saved" : ""}`,
      );
      if (!temporary) await loadConversations();
    } catch (error) {
      if (stoppedByUser.current) {
        stoppedByUser.current = false;
        if (ticket === threadTicket.current) {
          setMessages([...priorMessages, { role: "user", content: question }]);
          setNotice(
            "Stopped waiting. The appliance may still finish this answer — reopen this chat to check.",
          );
        }
        if (!temporary) await loadConversations();
        return;
      }
      if (ticket !== threadTicket.current) return;
      // The connection dropped mid-answer — the fault a reload used to be the
      // only cure for. The appliance finishes server-side, so arm the resume-poll
      // to surface the reply live (see `droppedConnectionSubject`). The question
      // stays put and the poll replaces it with the server's turns, so it is not
      // returned to the composer — re-asking is the duplicate that avoids.
      const droppedSubject = await droppedConnectionSubject(error, temporary, conversationId, async () => {
        const saved = await apiRequest<AiChatConversation[]>("/ai-chat/conversations");
        return ticket === threadTicket.current ? saved[0]?.id ?? "" : conversationId;
      });
      if (droppedSubject) {
        setConversationId(droppedSubject);
        setMessages([...priorMessages, { role: "user", content: question }]);
        setAwaitingAnswer(true);
        setNotice(DROPPED_CONNECTION_NOTICE);
        await loadConversations();
        return;
      }
      const failure = error instanceof ApiError && error.code === "request_timeout"
          ? "Vaelor stopped waiting before the appliance returned a model result. The model may still be loading or may be too large for its server. Confirm it is loaded, retry once, or choose a smaller model."
          : error instanceof Error ? error.message : "AI Chat could not answer.";
      const errorDetails = error instanceof ApiError ? error.details : {};
      const persistedConversationId = typeof errorDetails.conversation_id === "string"
        ? errorDetails.conversation_id
        : "";
      const persistedRecord = errorDetails.message_record;
      const failureMessage: AiChatThreadMessage = (
        persistedRecord
        && typeof persistedRecord === "object"
        && "role" in persistedRecord
        && persistedRecord.role === "assistant"
        && "content" in persistedRecord
        && typeof persistedRecord.content === "string"
      ) ? persistedRecord as AiChatThreadMessage : {
        role: "assistant",
        content: `This request failed: ${failure}`,
      };
      setMessages([
        ...priorMessages,
        { role: "user", content: question },
        { ...failureMessage, failed: true },
      ]);
      if (!temporary) {
        if (persistedConversationId) setConversationId(persistedConversationId);
        await loadConversations();
      }
      setNotice(reportFailure(error, requestModel, failure));
      /*
       * Losing the question as well as the answer is two losses for one
       * failure. The Assistant already hands it back; AI Chat emptied the
       * composer on submit and never refilled it, so recovering meant retyping
       * from memory. An answer the reader has since started typing wins.
       */
      setInput((current) => current || question);
    } finally {
      setBusy("");
      setPending(null);
    }
  };

  const send = async (event: FormEvent) => {
    event.preventDefault();
    const question = input.trim();
    if (!question) return;
    setInput("");
    await submitQuestion(question, messages);
  };

  /**
   * Ask the unanswered question again.
   *
   * A turn the server stored can be regenerated against its own id. A turn
   * left behind by Stop or by a failed first send was never written, so there
   * is nothing to regenerate and the repair is to send it once more, from the
   * conversation as it stood before it.
   */
  const retryAnswer = async (message: AiChatThreadMessage) => {
    // Take back only the copy the failure handler put in the composer. Text the
    // reader has since changed is theirs and stays where they left it.
    setInput((current) => (current === message.content ? "" : current));
    if (message.id && conversationId && !temporary) {
      await regenerate(message);
      return;
    }
    const index = messages.indexOf(message);
    await submitQuestion(message.content, index < 0 ? messages : messages.slice(0, index));
  };

  /*
   * The dialog focuses and selects its field in an effect keyed on its
   * handlers. Passing a fresh closure each render re-ran that effect after
   * every keystroke, so the field re-selected its own contents and the next
   * character replaced the last: typing "Cupboard box notes" saved as "s".
   * Stable handlers keep the selection to the moment the dialog opens.
   */
  const closeRename = useCallback(() => setRenameTitle(null), []);

  /*
   * Searching filters the conversation list, and the header read its title
   * out of that filtered list - so typing in Search chats renamed the open
   * conversation to "New chat" while its transcript was still on screen. The
   * open conversation is remembered independently of whatever the list is
   * currently showing.
   */
  const current = conversations.find((item) => item.id === conversationId)
    ?? (openRecord?.id === conversationId ? openRecord : undefined);
  // The composer uses a non-empty prop only to enable explicit agent routing;
  // ordinary chat still sends the empty selected model and receives its normal error.
  const composerModel = selectedModel || selectedAgentId || "agent-delegation";
  const noticeIsError = /could not|did not|failed|too long|exceeded|stopped waiting|rejected|lost the connection|error|unavailable/i.test(notice);
  return (
    <div className={focusMode ? "ai-chat-page ai-chat-page--focus" : "ai-chat-page"}>
      <header className="page-heading ai-chat-page__heading">
        <div><h1>{destinations["ai-chat"].name}</h1><p>{destinations["ai-chat"].descriptor}. The Assistant is a separate destination.</p></div>
      </header>
      {notice && <Notice className="ai-chat-notice" severity={noticeIsError ? "danger" : "info"}><span>{notice}</span><Button aria-label="Dismiss notification" className="ai-chat-notice__dismiss" onClick={() => setNotice("")} type="button" variant="quiet">×</Button></Notice>}
      <div className={detailsOpen ? "ai-chat-workspace has-details" : "ai-chat-workspace"}>
        <AiChatRail
          conversationId={conversationId}
          conversations={conversations}
          onArchive={() => void archiveConversation()}
          onDelete={() => current && setDeleteTarget({ type: "conversation", id: current.id, name: current.title })}
          onExport={exportConversation}
          onNew={() => newConversation(false)}
          onOpen={(item) => void openConversation(item)}
          onRename={() => setRenameTitle(current?.title ?? "")}
          onSearch={setSearch}
          onTemporary={() => newConversation(true)}
          onToggleArchive={() => { newConversation(); setShowArchived((value) => !value); }}
          search={search}
          showArchived={showArchived}
          temporary={temporary}
        />
        <section className="ai-chat-main">
          <AiChatToolbar
            agents={agents}
            agentsResolved={agentsResolved}
            busy={busy !== ""}
            // `undefined` until `/ai-chat/setup` answers and again while a
            // connection is being activated: at both moments Vaelor does not
            // know what this machine can run, and the picker says so.
            connection={setup && busy !== "connection" ? setup.active_connection : undefined}
            detailsOpen={detailsOpen}
            focusMode={focusMode}
            modelFailures={modelFailures}
            models={models}
            onChooseModel={(model) => void chooseModel(model)}
            onSelectAgent={(id) => {
              setSelectedAgentId(id);
              setAgentProposal(null);
              setAgentRunSubmitted(false);
            }}
            onToggleDetails={() => setDetailsOpen((value) => !value)}
            onToggleFocus={() => setFocusMode((value) => !value)}
            selectedAgentId={selectedAgentId}
            selectedModel={selectedModel}
            subtitle={temporary ? "No history will be saved" : selectedCollections.length ? "Cited knowledge enabled" : "General model knowledge"}
            title={temporary ? "Temporary chat" : current?.title || "New chat"}
          />
          <AiChatThread
            busy={busy === "chat"}
            canModify={!temporary && Boolean(conversationId)}
            generating={pending && pending.ticket === viewTicket ? { model: pending.model } : null}
            messages={messages}
            model={selectedModel || (agentProposal ? "Agent proposal" : "No model selected")}
            onFork={forkConversation}
            onNotice={setNotice}
            onRegenerate={regenerate}
            onRetry={retryAnswer}
            onStop={stopGeneration}
            onSuggestion={setInput}
            resumed={resumed}
          />
          {agentProposal && (
            <section className="assistant-proposal" aria-label="Custom agent run proposal">
              <span>
                <small>Custom agent · version {agentProposal.profile_version} · approval required</small>
                <strong>{agentProposal.profile_name}</strong>
                <p>{agentProposal.task}</p>
                <small>
                  {agentProposal.capabilities.length
                    ? `Capabilities: ${agentProposal.capabilities.join(", ")}`
                    : "Capabilities: none listed"}
                </small>
                <small>
                  {agentProposal.app_grants.length
                    ? agentProposal.app_grants.map((grant) => {
                      const operations = grant.operations.map((operation) => operation.name).join(", ");
                      return `${grant.app_name}: ${operations || "no operations listed"}`;
                    }).join(" · ")
                    : "App access: none granted"}
                </small>
                <small>
                  {agentProposal.integrations.length
                    ? `Integrations: ${agentProposal.integrations.join(", ")}`
                    : "Integrations: none"}
                </small>
              </span>
              <div>
                <Button disabled={busy !== "" || agentRunSubmitted} onClick={() => void reviewAgentRun()} type="button" variant="primary">{agentRunSubmitted ? "Saved for review" : "Review agent run"}</Button>
                {agentRunSubmitted && (
                  <Button onClick={openAgentRun} type="button" variant="quiet">Open this run</Button>
                )}
              </div>
            </section>
          )}
          <AiChatComposer
            busy={busy === "chat"}
            collections={setup?.collections ?? []}
            model={composerModel}
            onChange={setInput}
            onFile={(file) => void ingestFile(file)}
            onOpenDetails={() => setDetailsOpen(true)}
            onSubmit={(event) => void send(event)}
            onToggleCollection={(id) => void toggleCollection(id)}
            selectedCollections={selectedCollections}
            temporary={temporary}
            value={input}
          />
        </section>
        {detailsOpen && (
          <AiChatDetails
            activeCollection={activeCollection}
            busy={busy}
            collectionDescription={collectionDescription}
            collectionName={collectionName}
            documents={documents}
            onActivateConnection={(connection) => void activateConnection(connection)}
            onActiveCollection={setActiveCollection}
            onClose={() => setDetailsOpen(false)}
            onCollectionDescription={setCollectionDescription}
            onCollectionName={setCollectionName}
            onCreateCollection={(event) => void createCollection(event)}
            onDeleteCollection={(collection) => setDeleteTarget({ type: "collection", id: collection.id, name: collection.name })}
            onDeleteDocument={(id) => void removeDocument(id)}
            onFile={(file) => void ingestFile(file)}
            onToggleCollection={(id) => void toggleCollection(id)}
            selectedCollections={selectedCollections}
            setup={setup}
          />
        )}
      </div>
      <TextPromptDialog
        busy={busy === "rename"}
        description="Use a short name that will be easy to find later."
        label="Chat name"
        onCancel={closeRename}
        onChange={setRenameTitle}
        onSubmit={() => void renameConversation()}
        open={renameTitle !== null}
        title="Rename chat"
        value={renameTitle ?? ""}
      />
      <ConfirmDialog
        busy={busy === "delete"}
        confirmLabel="Delete permanently"
        description={deleteTarget?.type === "collection" ? "All indexed files and chunks in this collection will be permanently deleted. Saved chats remain." : "This conversation and every message in it will be permanently deleted."}
        onCancel={() => setDeleteTarget(null)}
        onConfirm={() => void confirmDelete()}
        open={Boolean(deleteTarget)}
        title={`Delete ${(deleteTarget?.name || "item").replace(/[.!?,;:]+$/, "")}?`}
      />
    </div>
  );
}
