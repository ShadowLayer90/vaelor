import { type FormEvent, type KeyboardEvent, useRef } from "react";
import type { AiChatCollection } from "./aiChatTypes";
import { Icon } from "./Icon";
import { Button, Textarea } from "./ui";

export function AiChatComposer({
  value,
  busy,
  model,
  temporary,
  collections,
  selectedCollections,
  onChange,
  onSubmit,
  onToggleCollection,
  onOpenDetails,
  onFile,
}: {
  value: string;
  busy: boolean;
  model: string;
  temporary: boolean;
  collections: AiChatCollection[];
  selectedCollections: string[];
  onChange: (value: string) => void;
  onSubmit: (event: FormEvent) => void;
  onToggleCollection: (id: string) => void;
  onOpenDetails: () => void;
  onFile: (file: File) => void;
}) {
  const fileRef = useRef<HTMLInputElement>(null);
  /*
   * One box, one name. The placeholder said "Message Vaelor AI" while the
   * accessible name said "Message AI Chat", so a screen-reader user and a
   * sighted user beside them were looking at two differently named controls.
   * "Vaelor AI" is what every answer in the transcript is signed, so that is
   * the name that stays.
   */
  const composerName = "Message Vaelor AI";
  const messageProblem = value.trim().length > 8000
    ? "Keep AI Chat messages to 8,000 characters or fewer."
    : "";
  const submitOnShortcut = (event: KeyboardEvent<HTMLTextAreaElement>) => {
    if (event.key === "Enter" && !event.shiftKey) {
      event.preventDefault();
      event.currentTarget.form?.requestSubmit();
    }
  };
  return (
    <form className="ai-chat-composer" onSubmit={onSubmit}>
      {selectedCollections.length > 0 && (
        <div className="ai-chat-composer__sources">
          {/*
            * Attaching a collection narrows the answer to it, and nothing said
            * so: the same model that wrote an essay with the chip off replied
            * "I don't have any reliable information on that" with it on, and a
            * chip that is selected by default made the assistant look broken
            * rather than constrained. The sentence names the constraint and
            * names the way out of it, beside the control that removes it.
            */}
          <small className="ai-chat-composer__sources-note">
            Answers are grounded in these documents. Remove one to let Vaelor answer
            from what it already knows as well.
          </small>
          {collections.filter((item) => selectedCollections.includes(item.id)).map((item) => (
            <Button
              aria-label={`Stop grounding answers in ${item.name}`}
              key={item.id}
              onClick={() => onToggleCollection(item.id)}
              title={`Stop grounding answers in ${item.name}`}
              type="button"
              variant="quiet"
            >
              <Icon name="database" size={13} /> {item.name} <span aria-hidden="true">×</span>
            </Button>
          ))}
        </div>
      )}
      <Textarea
        aria-describedby={messageProblem ? "ai-chat-message-error" : undefined}
        label={<span className="sr-only">{composerName}</span>}
        maxLength={8000}
        onChange={(event) => onChange(event.target.value)}
        onKeyDown={submitOnShortcut}
        placeholder={composerName}
        rows={3}
        value={value}
      />
      {messageProblem && <p className="field-error" id="ai-chat-message-error" role="alert">{messageProblem}</p>}
      <div className="ai-chat-composer__bar">
        <div>
          <input
            accept=".txt,.md,.markdown,.json,.csv,.yaml,.yml,.log,text/*"
            className="ui-control ui-control--file-picker"
            hidden
            onChange={(event) => {
              const file = event.target.files?.[0];
              if (file) onFile(file);
              event.target.value = "";
            }}
            ref={fileRef}
            type="file"
          />
          {/*
            * Always opens the file picker. It used to divert to the knowledge
            * panel whenever no collection existed yet - which is every new
            * appliance - and that panel had no visible way to make one, so the
            * button appeared to do nothing. A collection is created on demand
            * when the first file lands.
            */}
          <Button
            aria-label="Attach a file for Vaelor to read"
            onClick={() => fileRef.current?.click()}
            title="Attach a file for Vaelor to read"
            type="button"
            variant="quiet"
          >
            <span aria-hidden="true">+</span>
          </Button>
          <Button onClick={onOpenDetails} type="button" variant="quiet">
            <Icon name="settings" size={15} /> Knowledge
          </Button>
        </div>
        <small>{temporary ? "Temporary" : selectedCollections.length ? `${selectedCollections.length} source set` : "Model knowledge only"}</small>
        <Button
          aria-label="Send message"
          className="ai-chat-composer__send"
          disabled={busy || !value.trim() || Boolean(messageProblem) || !model}
          type="submit"
          variant="primary"
        >
          ↑
        </Button>
      </div>
    </form>
  );
}
