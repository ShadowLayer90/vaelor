import { timeAgo } from "../lib/format";
import { modelDisplayName } from "../lib/modelIdentity";
import type { AiChatConversation } from "./aiChatTypes";
import { Icon } from "./Icon";
import { Button } from "./ui";

/**
 * The conversation rail (the Chat, ChatTemporary and ChatArchive boards).
 *
 * New chat and Temp, the search box, the Recent or Archive list with the
 * switch between them, the open chat's actions (Rename, Export, Archive or
 * Restore, Delete), and one line at the foot saying where the chat is kept.
 */
export function AiChatRail({
  conversations,
  conversationId,
  search,
  showArchived,
  temporary,
  onSearch,
  onOpen,
  onNew,
  onTemporary,
  onToggleArchive,
  onRename,
  onArchive,
  onDelete,
  onExport,
}: {
  conversations: AiChatConversation[];
  conversationId: string;
  search: string;
  showArchived: boolean;
  temporary: boolean;
  onSearch: (value: string) => void;
  onOpen: (item: AiChatConversation) => void;
  onNew: () => void;
  onTemporary: () => void;
  onToggleArchive: () => void;
  onRename: () => void;
  onArchive: () => void;
  onDelete: () => void;
  onExport: () => void;
}) {
  return (
    <aside className="ai-chat-rail" aria-label="Chat history">
      <div className="ai-chat-rail__new">
        <Button className="ai-chat-rail__new-chat" onClick={onNew} type="button" variant="primary">
          New chat
        </Button>
        <Button
          aria-label="Start a temporary chat"
          aria-pressed={temporary}
          className={temporary ? "ai-chat-rail__temp is-selected" : "ai-chat-rail__temp"}
          onClick={onTemporary}
          title="Temporary chats are not saved"
          type="button"
        >
          Temp
        </Button>
      </div>
      <label className="ai-chat-search">
        <span className="sr-only">Search conversations</span>
        <input
          className="ai-chat-input"
          onChange={(event) => onSearch(event.target.value)}
          placeholder="Search chats"
          type="search"
          value={search}
        />
      </label>
      <div className="ai-chat-rail__heading">
        <h2 className={showArchived ? "is-archive" : undefined}>{showArchived ? "Archive" : "Recent"}</h2>
        <Button className="ai-chat-ghost" onClick={onToggleArchive} type="button" variant="quiet">
          {showArchived ? "Recent" : "Archive"}
        </Button>
      </div>
      {conversations.length > 0 && (
        <div className="ai-chat-rail__list">
          {conversations.map((item) => {
            // Only a model that answered in this chat is named (ACC-116); a
            // chat nothing has answered yet shows its age alone.
            const byline = [item.model ? modelDisplayName(item.model) : "", timeAgo(item.updated_at * 1000)]
              .filter(Boolean).join(" · ");
            return (
              <Button
                aria-current={item.id === conversationId ? "page" : undefined}
                aria-label={`${item.title} · ${byline}`}
                className="ai-chat-rail__item"
                key={item.id}
                onClick={() => onOpen(item)}
                title={item.title}
                type="button"
                variant="quiet"
              >
                <strong>{item.title}</strong>
                <small>{byline}</small>
              </Button>
            );
          })}
        </div>
      )}
      {conversationId && (
        <div className="ai-chat-rail__actions">
          <Button className="ai-chat-ghost" onClick={onRename} type="button" variant="quiet">Rename</Button>
          <Button className="ai-chat-ghost" onClick={onExport} type="button" variant="quiet">Export</Button>
          <Button className="ai-chat-ghost" onClick={onArchive} type="button" variant="quiet">{showArchived ? "Restore" : "Archive"}</Button>
          <Button className="ai-chat-ghost ai-chat-ghost--danger" onClick={onDelete} type="button" variant="quiet">Delete</Button>
        </div>
      )}
      {!conversations.length && (
        <p className="ai-chat-rail__empty">
          {search ? "No matching chats." : showArchived ? "No archived chats." : "No saved chats yet."}
        </p>
      )}
      {temporary ? (
        <p className="ai-chat-rail__privacy ai-chat-rail__privacy--temporary">
          <Icon name="shield" size={16} /> Temporary · not saved
        </p>
      ) : (
        <p className="ai-chat-rail__privacy">Saved on this Vaelor node</p>
      )}
    </aside>
  );
}
