import { timeAgo } from "../lib/format";
import type { AiChatConversation } from "./aiChatTypes";
import { Icon } from "./Icon";
import { Button, Input } from "./ui";

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
      <div className="ai-chat-rail__brand">
        <span className="ai-chat-rail__mark"><Icon name="memory" size={18} /></span>
        <div><strong>AI Chat</strong><small>Private workspace</small></div>
      </div>
      <div className="ai-chat-rail__new">
        <Button className="ai-chat-rail__new-chat" onClick={onNew} type="button" variant="primary">
          <span aria-hidden="true">+</span> New chat
        </Button>
        <Button
          aria-label="Start a temporary chat"
          className={temporary ? "is-active" : undefined}
          onClick={onTemporary}
          title="Temporary chats are not saved"
          type="button"
          variant="quiet"
        >
          <Icon name="lock" size={17} />
          <span>Temp</span>
        </Button>
      </div>
      <div className="ai-chat-search">
        <Input
          label={<span>Search conversations</span>}
          onChange={(event) => onSearch(event.target.value)}
          placeholder="Search chats"
          type="search"
          value={search}
        />
      </div>
      <div className="ai-chat-rail__section">
        <div className="ai-chat-rail__heading">
          <strong>{showArchived ? "Archive" : "Recent"}</strong>
          <Button onClick={onToggleArchive} type="button" variant="quiet">
            {showArchived ? "Recent" : "Archive"}
          </Button>

        </div>
        <div className="ai-chat-rail__list">
          {conversations.length ? conversations.map((item) => (
            <Button
              aria-current={item.id === conversationId ? "page" : undefined}
              aria-label={item.title + " · " + (item.model || "No model") + " · " + timeAgo(item.updated_at * 1000)}
              className="ai-chat-rail__item"
              key={item.id}
              onClick={() => onOpen(item)}
              title={item.title}
              type="button"
              variant="quiet"
            >
              <strong>{item.title}</strong>
              <small>{item.model || "No model"} · {timeAgo(item.updated_at * 1000)}</small>
            </Button>
          )) : (
            <p className="empty-copy">{search ? "No matching chats." : "No saved chats yet."}</p>
          )}
        </div>
      </div>
      {conversationId && (
        <div className="ai-chat-rail__actions">
          <Button onClick={onRename} type="button" variant="quiet">Rename</Button>
          <Button onClick={onExport} type="button" variant="quiet">Export</Button>
          <Button onClick={onArchive} type="button" variant="quiet">{showArchived ? "Restore" : "Archive"}</Button>
          <Button className="danger-link" onClick={onDelete} type="button" variant="danger">Delete</Button>
        </div>
      )}
      <div className="ai-chat-rail__privacy">
        <Icon name={temporary ? "lock" : "database"} size={16} />
        <span>{temporary ? "Temporary · not saved" : "Saved on this Vaelor node"}</span>
      </div>
    </aside>
  );
}
