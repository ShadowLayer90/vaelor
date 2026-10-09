import { useEffect, useRef, type KeyboardEvent as ReactKeyboardEvent, type ReactNode, type RefObject } from "react";
import { timeAgo } from "../lib/format";
import type { useAssistantChat } from "../hooks/useAssistantChat";
import { Icon } from "./Icon";
import { Button, EmptyState } from "./ui";

type Chat = ReturnType<typeof useAssistantChat>;

/**
 * Close a popover on a press outside it or on Escape (focus goes back to the
 * button that opened it). One hook for both of this bar's popovers, so they
 * cannot disagree about how a popover leaves.
 */
function useDismiss(
  open: boolean,
  wrap: RefObject<HTMLElement | null>,
  trigger: RefObject<HTMLElement | null>,
  close: () => void,
) {
  useEffect(() => {
    if (!open) return undefined;
    const outside = (event: PointerEvent) => {
      if (!wrap.current?.contains(event.target as Node)) close();
    };
    const escape = (event: KeyboardEvent) => {
      if (event.key !== "Escape") return;
      close();
      trigger.current?.focus();
    };
    document.addEventListener("pointerdown", outside);
    document.addEventListener("keydown", escape);
    return () => {
      document.removeEventListener("pointerdown", outside);
      document.removeEventListener("keydown", escape);
    };
  }, [close, open, trigger, wrap]);
}

/**
 * The conversation's own row (VD-200, the Assistant and AssistChats boards):
 * its title and where it is kept, then New chat, Past chats with the count, and
 * More. Past chats opens the saved list under its button; More opens the
 * saved chat's menu.
 */
export function AssistantConversationBar({
  chat,
  moreOpen,
  setMoreOpen,
}: {
  chat: Chat;
  moreOpen: boolean;
  setMoreOpen: (open: boolean) => void;
}) {
  const currentConversation = chat.conversations.find((item) => item.id === chat.conversationId);
  const pastWrap = useRef<HTMLDivElement>(null);
  const pastButton = useRef<HTMLButtonElement>(null);
  const moreWrap = useRef<HTMLDivElement>(null);
  const moreButton = useRef<HTMLButtonElement>(null);
  useDismiss(chat.showChatHistory, pastWrap, pastButton, () => chat.setShowChatHistory(false));
  useDismiss(moreOpen, moreWrap, moreButton, () => setMoreOpen(false));
  const menuRef = useRef<HTMLDivElement>(null);
  // A role="menu" owes the menu keys (VD-200 assist review): focus lands on
  // the first item when it opens, and the arrows, Home and End move along it.
  useEffect(() => {
    if (moreOpen) menuRef.current?.querySelector<HTMLElement>("[role='menuitem']")?.focus();
  }, [moreOpen]);
  const menuKeys = (event: ReactKeyboardEvent<HTMLDivElement>) => {
    const items = [...event.currentTarget.querySelectorAll<HTMLElement>("[role='menuitem']:not([disabled])")];
    const at = items.indexOf(document.activeElement as HTMLElement);
    const next = { ArrowDown: at + 1, ArrowUp: at - 1, Home: 0, End: items.length - 1 }[event.key];
    if (next === undefined || items.length === 0) return;
    event.preventDefault();
    items[(next + items.length) % items.length].focus();
  };
  const archiveView = chat.conversationView === "archive";
  const visibleConversations = chat.conversations.filter(
    (conversation) => conversation.archived === archiveView,
  );
  const menuItem = (label: string, run: () => void, danger = false): ReactNode => (
    <Button
      className={danger ? "as-menu__danger" : undefined}
      onClick={() => { moreButton.current?.focus(); setMoreOpen(false); run(); }}
      role="menuitem"
      type="button"
      variant="quiet"
    >
      {label}
    </Button>
  );
  return (
    <div className="as-convo-bar">
      <div className="as-convo-bar__title">
        <strong id="assistant-chat-conversation">{currentConversation?.title || "New chat"}</strong>
        <span className="as-small as-muted">Saved automatically on this machine</span>
      </div>
      <div className="as-convo-bar__actions">
        <Button onClick={chat.startNewChat} type="button">New chat</Button>
        <div className="as-popover" ref={pastWrap}>
          <Button
            aria-expanded={chat.showChatHistory}
            className={chat.showChatHistory ? "as-btn-on" : undefined}
            onClick={() => { setMoreOpen(false); chat.setShowChatHistory(!chat.showChatHistory); }}
            ref={pastButton}
            type="button"
          >
            {/*
              * Not "History". The tablist above owns that word for the run
              * archive, and two controls with one name on one screen sent a
              * reader looking for a prepared agent run into their saved
              * conversations. The tab name is the structural one, so this is
              * the one that changes.
              */}
            Past chats
            {chat.conversations.length > 0 && <span className="as-mono as-convo-bar__count">{" "}{chat.conversations.length}</span>}
          </Button>
          {chat.showChatHistory && (
            <div aria-label={archiveView ? "Archived chats" : "Saved chats"} className="as-chats" role="region">
              <div className="as-chats__head">
                <div>
                  <strong>{archiveView ? "Archived chats" : "Saved chats"}</strong>
                  <span className="as-small as-muted">
                    {archiveView ? "Open, export, restore, or delete an archived chat." : "Select one to continue where you left off."}
                  </span>
                </div>
                <Button
                  className="as-btn-ghost"
                  onClick={() => chat.setConversationView(archiveView ? "active" : "archive")}
                  type="button"
                >
                  {archiveView ? "Back to saved chats" : "View archive"}
                </Button>
              </div>
              {visibleConversations.length ? visibleConversations.map((conversation) => (
                <Button
                  aria-current={conversation.id === chat.conversationId ? "true" : undefined}
                  className="as-chats__row"
                  key={conversation.id}
                  onClick={() => void chat.openConversation(conversation)}
                  type="button"
                  variant="quiet"
                >
                  <span>
                    <strong>{conversation.title}</strong>
                    <small>{conversation.message_count} messages · {timeAgo(conversation.updated_at * 1000)}</small>
                  </span>
                  <Icon name="chevron" size={16} />
                </Button>
              )) : archiveView ? (
                <EmptyState icon={<Icon name="database" size={16} />} title="No archived chats." />
              ) : (
                <EmptyState
                  icon={<Icon name="chat" size={16} />}
                  text="Your first message starts one automatically."
                  title="No saved chats yet"
                />
              )}
            </div>
          )}
        </div>
        {/* More appears only once the chat is saved: a new, empty chat has nothing to rename. */}
        {chat.conversationId && (
          <div className="as-popover" ref={moreWrap}>
            <Button
              aria-expanded={moreOpen}
              aria-haspopup="menu"
              className="as-btn-ghost"
              onClick={() => { chat.setShowChatHistory(false); setMoreOpen(!moreOpen); }}
              ref={moreButton}
              type="button"
            >
              More
            </Button>
            {moreOpen && (
              <div aria-label="This chat" className="as-menu" onKeyDown={menuKeys} ref={menuRef} role="menu">
                {menuItem("Rename", chat.renameConversation)}
                {menuItem("Export Markdown", () => void chat.exportConversation())}
                {menuItem(
                  currentConversation?.archived ? "Restore chat" : "Archive chat",
                  () => void chat.archiveConversation(!currentConversation?.archived),
                )}
                <hr />
                {menuItem("Delete chat", () => chat.setConfirmChatDelete(true), true)}
              </div>
            )}
          </div>
        )}
      </div>
    </div>
  );
}
