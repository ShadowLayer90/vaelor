import { useCallback, useEffect, useRef, useState } from "react";
import { apiRequest } from "../lib/api";
import { readDraft, writeDraft } from "../lib/draftStorage";
import type { Session } from "../types";
import { MODEL_ANSWER_TIMEOUT_MS } from "../components/modelAnswerTimeout";
import {
  DROPPED_CONNECTION_NOTICE,
  droppedConnectionSubject,
  useResumedAnswer,
} from "./useResumedAnswer";
import type {
  AssistantAnswer,
  AssistantConversation,
  ChatMessage,
} from "../components/agentTypes";

/**
 * Durable conversation state and the chat handlers for the Assistant.
 *
 * Lifted out of `AgentCenter` so the page component stays inside the module
 * line limit while the Ask surface absorbs troubleshooting. The generation
 * counter and the abort controller move together on purpose: together they are
 * what keeps a late answer from an abandoned conversation abandoned, instead of
 * letting it appear in whatever chat the reader has moved on to.
 */
/**
 * What the transcript says where the answer would have been.
 *
 * Every clause is checkable. Nothing was changed because the Assistant
 * proposes changes and never makes them without a separate approval; the
 * appliance may still finish because aborting the request closes this
 * browser's connection and does not reach the run behind it; and it lands in
 * the saved history because `/assistant/chat` writes the turn server-side.
 */
export const STOPPED_RESPONSE =
  "You stopped this response. Your question above was not answered and nothing "
  + "on this appliance was changed. The appliance may still finish the answer — "
  + "if it does, it is saved with this chat.";

export interface AssistantChatController {
  /** True when a restored transcript ends on a question still being answered. */
  awaitingAnswer: boolean;
  /** True once the resumed wait gave up without an answer ever arriving. */
  awaitingAnswerLost: boolean;
  chatBusy: boolean;
  chatInput: string;
  chatMessages: ChatMessage[];
  chatNotice: string;
  chatRequestActive: boolean;
  /** Epoch ms the in-flight request started, or 0. Survives a tab change. */
  requestStartedAt: number;
  confirmChatDelete: boolean;
  conversationId: string;
  conversations: AssistantConversation[];
  conversationView: "active" | "archive";
  renameTitle: string | null;
  showChatHistory: boolean;
  archiveConversation: (archived: boolean) => Promise<void>;
  ask: () => Promise<void>;
  cancelRequest: () => void;
  deleteConversation: () => Promise<void>;
  exportConversation: () => Promise<void>;
  openConversation: (conversation: AssistantConversation) => Promise<void>;
  renameConversation: () => void;
  saveConversationTitle: () => Promise<void>;
  /** Start a fresh chat pre-filled from another surface, such as a run result. */
  seedChat: (draft: string, notice: string) => void;
  setChatInput: (value: string) => void;
  setChatNotice: (value: string) => void;
  setConfirmChatDelete: (open: boolean) => void;
  setConversationView: (view: "active" | "archive") => void;
  setRenameTitle: (title: string | null) => void;
  setShowChatHistory: (open: boolean) => void;
  startNewChat: () => void;
}

export function useAssistantChat(session: Session): AssistantChatController {
  const [conversationId, setConversationId] = useState("");
  const [conversations, setConversations] = useState<AssistantConversation[]>([]);
  const [showChatHistory, setShowChatHistory] = useState(false);
  const [renameTitle, setRenameTitle] = useState<string | null>(null);
  const [confirmChatDelete, setConfirmChatDelete] = useState(false);
  const [conversationView, setConversationView] = useState<"active" | "archive">("active");
  /*
   * #150: navigating away unmounted the Assistant and discarded a
   * part-written question without warning. The draft survives the round trip
   * — it is the reader's own text, not a credential — and clears itself the
   * moment the question is sent (setChatInput("")). It is keyed to the
   * signed-in account and storage-failure-safe (see draftStorage), which
   * also keeps the module loadable where no window storage exists at all,
   * which is how the production-asset contract test executes the bundle.
   */
  const [chatInput, setChatInputState] = useState(() =>
    readDraft("vaelor.assistant.draft", session.user.username),
  );
  const setChatInput = useCallback(
    (value: string | ((current: string) => string)) => {
      setChatInputState((current) => {
        const next = typeof value === "function" ? value(current) : value;
        writeDraft("vaelor.assistant.draft", session.user.username, next);
        return next;
      });
    },
    [session.user.username],
  );
  const [chatMessages, setChatMessages] = useState<ChatMessage[]>([]);
  const [chatBusy, setChatBusy] = useState(false);
  const [chatRequestActive, setChatRequestActive] = useState(false);
  const [chatNotice, setChatNotice] = useState("");
  const [requestStartedAt, setRequestStartedAt] = useState(0);
  const [awaitingAnswer, setAwaitingAnswer] = useState(false);
  const newChatGeneration = useRef(0);
  const chatAbort = useRef<AbortController | null>(null);

  useEffect(() => {
    void (async () => {
      const generation = newChatGeneration.current;
      try {
        const saved = await apiRequest<AssistantConversation[]>(
          "/assistant/conversations?limit=100&archived=1",
        );
        setConversations(saved);
        const latest = saved.find((conversation) => !conversation.archived);
        if (!latest) return;
        const messages = await apiRequest<ChatMessage[]>(
          `/assistant/conversations/${encodeURIComponent(latest.id)}/messages?limit=40`,
        );
        if (generation !== newChatGeneration.current) return;
        setConversationId(latest.id);
        setChatMessages(messages);
        setAwaitingAnswer(messages.at(-1)?.role === "user");
      } catch {
        // A new conversation can still be started when history is unavailable.
      }
    })();
  }, []);

  /*
   * Wait out loud for an answer the appliance is still writing, and re-read the
   * conversation until it lands.
   */
  const resumed = useResumedAnswer({
    armed: awaitingAnswer && Boolean(conversationId) && !chatRequestActive,
    subject: conversationId,
    poll: async () => {
      const generation = newChatGeneration.current;
      const messages = await apiRequest<ChatMessage[]>(
        `/assistant/conversations/${encodeURIComponent(conversationId)}/messages?limit=40`,
      );
      if (generation !== newChatGeneration.current) return true;
      if (messages.at(-1)?.role !== "assistant") return false;
      setChatMessages(messages);
      setAwaitingAnswer(false);
      return true;
    },
  });

  const refreshConversations = useCallback(async () => {
    const saved = await apiRequest<AssistantConversation[]>(
      "/assistant/conversations?limit=100&archived=1",
    );
    setConversations(saved);
  }, []);

  const startNewChat = useCallback(() => {
    chatAbort.current?.abort();
    chatAbort.current = null;
    newChatGeneration.current += 1;
    setConversationId("");
    setChatMessages([]);
    setChatInput("");
    setChatNotice("New chat ready. It will save automatically after your first message.");
    setShowChatHistory(false);
    setChatRequestActive(false);
    setChatBusy(false);
    setRequestStartedAt(0);
    setAwaitingAnswer(false);
  }, []);

  const seedChat = useCallback((draft: string, notice: string) => {
    chatAbort.current?.abort();
    chatAbort.current = null;
    newChatGeneration.current += 1;
    setConversationId("");
    setChatMessages([]);
    setShowChatHistory(false);
    setChatRequestActive(false);
    setChatBusy(false);
    setRequestStartedAt(0);
    setAwaitingAnswer(false);
    setChatInput(draft);
    setChatNotice(notice);
  }, []);

  const openConversation = useCallback(async (conversation: AssistantConversation) => {
    chatAbort.current?.abort();
    const generation = ++newChatGeneration.current;
    setChatBusy(true);
    setChatNotice("");
    try {
      const messages = await apiRequest<ChatMessage[]>(
        `/assistant/conversations/${encodeURIComponent(conversation.id)}/messages?limit=100`,
      );
      if (generation !== newChatGeneration.current) return;
      setConversationId(conversation.id);
      setChatMessages(messages);
      setAwaitingAnswer(messages.at(-1)?.role === "user");
      setShowChatHistory(false);
    } catch (error) {
      setChatNotice(error instanceof Error ? error.message : "The saved chat could not be opened.");
    } finally {
      if (generation === newChatGeneration.current) setChatBusy(false);
    }
  }, []);

  const renameConversation = useCallback(() => {
    if (!conversationId) return;
    const current = conversations.find((item) => item.id === conversationId);
    setRenameTitle(current?.title || "Vaelor chat");
  }, [conversationId, conversations]);

  const saveConversationTitle = useCallback(async () => {
    const title = renameTitle?.trim();
    if (!title) return;
    setChatBusy(true);
    await apiRequest(
      `/assistant/conversations/${encodeURIComponent(conversationId)}`,
      { method: "PATCH", body: JSON.stringify({ title }) },
      session.csrf_token,
    );
    setRenameTitle(null);
    setChatBusy(false);
    setChatNotice("Chat renamed. Changes are saved automatically.");
    await refreshConversations();
  }, [conversationId, refreshConversations, renameTitle, session.csrf_token]);

  const exportConversation = useCallback(async () => {
    if (!conversationId) return;
    const conversation = await apiRequest<AssistantConversation & { messages: ChatMessage[] }>(
      `/assistant/conversations/${encodeURIComponent(conversationId)}/export`,
    );
    const lines = [
      `# ${conversation.title}`,
      "",
      `Exported ${new Date().toLocaleString()}`,
      "",
      ...conversation.messages.flatMap((message) => [
        `## ${message.role === "user" ? "You" : "Vaelor"}`,
        "",
        message.content,
        "",
      ]),
    ];
    const link = document.createElement("a");
    link.href = URL.createObjectURL(new Blob([lines.join("\n")], { type: "text/markdown" }));
    link.download = `${conversation.title.replace(/[^a-z0-9]+/gi, "-").replace(/^-|-$/g, "").toLowerCase() || "vaelor-chat"}.md`;
    link.click();
    URL.revokeObjectURL(link.href);
    setChatNotice("Chat exported as a Markdown file.");
  }, [conversationId]);

  const deleteConversation = useCallback(async () => {
    if (!conversationId) return;
    setChatBusy(true);
    await apiRequest(
      `/assistant/conversations/${encodeURIComponent(conversationId)}`,
      { method: "DELETE" },
      session.csrf_token,
    );
    setConfirmChatDelete(false);
    setChatBusy(false);
    startNewChat();
    setChatNotice("Saved chat deleted.");
    await refreshConversations();
  }, [conversationId, refreshConversations, session.csrf_token, startNewChat]);

  const archiveConversation = useCallback(async (archived: boolean) => {
    if (!conversationId) return;
    await apiRequest(
      `/assistant/conversations/${encodeURIComponent(conversationId)}`,
      { method: "PATCH", body: JSON.stringify({ archived }) },
      session.csrf_token,
    );
    startNewChat();
    setChatNotice(archived ? "Chat moved to the archive." : "Chat restored to active history.");
    await refreshConversations();
  }, [conversationId, refreshConversations, session.csrf_token, startNewChat]);

  const ask = useCallback(async () => {
    const question = chatInput.trim();
    if (!question) return;
    setChatBusy(true);
    setChatRequestActive(true);
    setRequestStartedAt(Date.now());
    setAwaitingAnswer(false);
    setChatNotice("");
    setChatInput("");
    const controller = new AbortController();
    const generation = newChatGeneration.current;
    chatAbort.current = controller;
    // VD-112 follow-up. One key per ask of this turn, minted here — the one
    // user action — so the transport-level re-send in api.ts carries the SAME
    // key and the appliance dedupes it instead of writing a second user turn or
    // starting a second inference.
    const idempotencyKey = globalThis.crypto?.randomUUID?.() ?? `assistant-${Date.now()}`;
    setChatMessages((current) => [
      ...current,
      { role: "user", content: question, created_at: Math.floor(Date.now() / 1000) },
    ]);
    try {
      const answer = await apiRequest<AssistantAnswer>(
        "/assistant/chat",
        {
          method: "POST",
          signal: controller.signal,
          // Must outlast the server's own inference budget, or the browser
          // aborts a request the appliance is still happily answering and
          // the user is told it failed when it did not. 260s cleared the 240s
          // default but not the 900s ceiling the budget can be raised to, so
          // an operator who lengthened it broke this surface the same way the
          // 60s default broke the appliance check.
          timeoutMs: MODEL_ANSWER_TIMEOUT_MS,
          body: JSON.stringify({
            message: question,
            conversation_id: conversationId || undefined,
            idempotency_key: idempotencyKey,
          }),
        },
        session.csrf_token,
      );
      if (generation !== newChatGeneration.current) return;
      setConversationId(answer.conversation_id);
      setChatMessages((current) => [
        ...current,
        {
          role: "assistant",
          content: answer.answer,
          created_at: Math.floor(Date.now() / 1000),
          metadata: {
            source: answer.source,
            evidence: answer.evidence,
            suggested_actions: answer.suggested_actions,
            // The routes the answer's own destinations resolve to. Dropping
            // them here is what left "ask it in AI Chat" as unclickable grey
            // text with the real route sitting unused in the payload.
            next_steps: answer.next_steps,
            proposed_job: answer.proposed_job,
            application_intent: answer.application_intent,
            proposed_agent_task: answer.proposed_agent_task,
          },
        },
      ]);
      await refreshConversations();
    } catch (error) {
      if (generation !== newChatGeneration.current) return;
      if (controller.signal.aborted) {
        /*
         * A stopped response is a terminal state, and it had neither half of
         * one. The question stayed in the transcript with nothing under it,
         * and the same words were put back into the composer with no sign
         * that anything had been typed there: a live tester's next question
         * concatenated onto the restored one and was sent as
         * `what is the CPU temperaturewhy is the sky blue`.
         *
         * So nothing is restored — the question is already above, and a
         * second copy of it is what corrupted the input — and the stop says
         * so in the transcript, where the reader is looking.
         */
        setChatMessages((current) => [
          ...current,
          {
            role: "assistant",
            content: STOPPED_RESPONSE,
            created_at: Math.floor(Date.now() / 1000),
            metadata: { stopped: true },
          },
        ]);
        return;
      }
      /*
       * The connection dropped mid-answer — a proxy idle timeout closing the
       * held-open POST, a nav-away — but the appliance already wrote the
       * question and finishes the answer server-side. That is the state a
       * manual refresh recovers from, so recover it here without one: arm the
       * resume-poll so the reply lands live. Stop is handled above; a structured
       * rejection queued nothing and falls through to its notice, never to a
       * poll. The question already sits in the transcript and the poll replaces
       * it with the server's turns, so it is not put back in the composer —
       * re-asking is the duplicate avoided.
       */
      const droppedSubject = await droppedConnectionSubject(error, false, conversationId, async () => {
        const saved = await apiRequest<AssistantConversation[]>(
          "/assistant/conversations?limit=100&archived=1",
        );
        if (generation !== newChatGeneration.current) return "";
        setConversations(saved);
        return saved.find((conversation) => !conversation.archived)?.id ?? "";
      });
      if (droppedSubject) {
        setConversationId(droppedSubject);
        setAwaitingAnswer(true);
        setChatNotice(DROPPED_CONNECTION_NOTICE);
        return;
      }
      const code = error instanceof Error && "code" in error ? error.code : "";
      setChatNotice(
        code === "request_timeout"
          // Naming a duration the client no longer enforces was its own small
          // untruth: the wait is now the appliance's configured budget.
          ? "The selected model did not answer within this appliance's configured time budget. Try a smaller local model from Apps and AI, or connect a faster model in Settings."
          /*
           * VD-069. On a Pi the model is not running until asked, and asking
           * is what starts it. This is not a failure and must not read like
           * one - the question is still in the composer, and the next send
           * lands on a model that is loading or loaded.
           */
          : code === "model_starting"
          ? (error instanceof Error ? error.message : "")
            || "The local model is starting. Ask again in a moment."
          : error instanceof Error ? error.message : "The assistant could not answer.",
      );
      setChatInput((current) => current || question);
    } finally {
      if (generation === newChatGeneration.current) {
        chatAbort.current = null;
        setChatRequestActive(false);
        setChatBusy(false);
        setRequestStartedAt(0);
      }
    }
  }, [chatInput, conversationId, refreshConversations, session.csrf_token]);

  const cancelRequest = useCallback(() => chatAbort.current?.abort(), []);

  return {
    archiveConversation,
    ask,
    awaitingAnswer: resumed.awaiting,
    awaitingAnswerLost: resumed.lost,
    cancelRequest,
    chatBusy,
    chatInput,
    chatMessages,
    chatNotice,
    chatRequestActive,
    confirmChatDelete,
    conversationId,
    conversations,
    conversationView,
    deleteConversation,
    exportConversation,
    openConversation,
    renameConversation,
    renameTitle,
    requestStartedAt,
    saveConversationTitle,
    seedChat,
    setChatInput,
    setChatNotice,
    setConfirmChatDelete,
    setConversationView,
    setRenameTitle,
    setShowChatHistory,
    showChatHistory,
    startNewChat,
  };
}
