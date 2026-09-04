import { useEffect, useRef } from "react";
import { hashTargetsPage } from "../lib/navigation";
import type { AiChatConversation } from "../components/aiChatTypes";

const AI_CHAT_ROUTE = "#/ai-chat";

/** Read the conversation this address bar points at, if any. */
export function conversationIdFromHash(hash: string): string {
  const [page, id = ""] = hash.replace(/^#\/?/, "").split("?", 1)[0].split("/");
  return page === "ai-chat" ? id : "";
}

/**
 * Keep the address bar and the open conversation in step (#164).
 *
 * Three things have to agree: the conversation on screen, the id in the URL,
 * and a conversation named by a link the reader followed. This owns all of it,
 * so the component below is not carrying four routing effects among its chat
 * logic.
 *
 *  - The URL names the open conversation, so a reload, bookmark or shared link
 *    returns to it rather than a blank New chat.
 *  - A load whose URL already names a conversation reopens it once the list has
 *    arrived.
 *  - While mounted, the URL can still change under the view: the "AI Chat" nav
 *    item points at the id-less `#/ai-chat`, and Back or Forward can land on
 *    another conversation's link. A real id hydrates that conversation; the
 *    id-less route keeps naming the open one instead of dropping it; a hash
 *    that no longer targets AI Chat is a real departure and is left to the
 *    shell, which unmounts this view.
 */
export function useAiChatRoute({
  conversationId, temporary, conversations, hasMessages, openConversation, onNotice,
}: {
  conversationId: string;
  temporary: boolean;
  conversations: AiChatConversation[];
  hasMessages: boolean;
  openConversation: (item: AiChatConversation) => Promise<void>;
  onNotice: (message: string) => void;
}) {
  // `openConversation` and `onNotice` are fresh closures every render; holding
  // them in refs lets the effects call the latest without re-subscribing.
  const openRef = useRef(openConversation);
  const noticeRef = useRef(onNotice);
  useEffect(() => { openRef.current = openConversation; });
  useEffect(() => { noticeRef.current = onNotice; });
  // Only the first URL-named conversation is restored automatically; after that
  // the reader is driving.
  const restored = useRef(false);

  const open = (id: string) => {
    const item = conversations.find((entry) => entry.id === id);
    if (item) void openRef.current(item).catch(() => noticeRef.current("That conversation could not be opened."));
  };

  useEffect(() => {
    const next = !temporary && conversationId ? `${AI_CHAT_ROUTE}/${conversationId}` : AI_CHAT_ROUTE;
    if (window.location.hash !== next) window.history.replaceState(null, "", next);
  }, [conversationId, temporary]);

  useEffect(() => {
    if (restored.current || temporary || !conversationId || hasMessages) return;
    if (!conversations.some((entry) => entry.id === conversationId)) return;
    restored.current = true;
    open(conversationId);
    // `open` closes over the current list and callbacks; it is safe to call once.
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [conversations, conversationId, hasMessages, temporary]);

  useEffect(() => {
    const onHashChange = () => {
      if (!hashTargetsPage(window.location.hash, "ai-chat")) return;
      const target = conversationIdFromHash(window.location.hash);
      if (target === conversationId) return;
      if (target) open(target);
      else if (conversationId && !temporary) {
        window.history.replaceState(null, "", `${AI_CHAT_ROUTE}/${conversationId}`);
      }
    };
    window.addEventListener("hashchange", onHashChange);
    return () => window.removeEventListener("hashchange", onHashChange);
    // `open` reads the current list at call time via the refs above.
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [conversationId, conversations, temporary]);
}
