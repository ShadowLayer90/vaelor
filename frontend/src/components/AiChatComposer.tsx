import { type FormEvent, type KeyboardEvent, useId, useRef } from "react";
import type { AiChatCollection } from "./aiChatTypes";
import { DOCUMENT_ACCEPT } from "../lib/aiChatDocument";
import { Icon } from "./Icon";
import { Button } from "./ui";

/**
 * The message box under the transcript (the Chat boards): the question, then
 * one row with Attach, Knowledge, what the answer will draw on, the keyboard
 * hint and Send. A temporary chat draws the box with a dashed edge.
 */
export function AiChatComposer({
  value,
  busy,
  model,
  temporary,
  collections,
  selectedCollections,
  onChange,
  onSubmit,
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
  onOpenDetails: () => void;
  onFile: (file: File) => void;
}) {
  const fileRef = useRef<HTMLInputElement>(null);
  const ids = useId().replaceAll(":", "");
  /*
   * One box, one name. "Vaelor AI" is what every answer in the transcript is
   * signed with, so that is the name the field carries; the placeholder is an
   * example prompt, not a competing name for the field.
   */
  const composerName = "Message Vaelor AI";
  // Per-screen limit: AI Chat accepts longer prompts (8,000) than the Assistant
  // composer (4,000) because they post to different backends.
  const messageProblem = value.trim().length > 8000
    ? "Keep AI Chat messages to 8,000 characters or fewer."
    : "";
  // Mirrors the send button's disabled gate below: non-empty, not busy, within
  // length, and a model selected.
  const canSend = !busy && Boolean(value.trim()) && !messageProblem && Boolean(model);
  const submitOnShortcut = (event: KeyboardEvent<HTMLTextAreaElement>) => {
    if (event.key !== "Enter" || event.shiftKey || event.altKey) return;
    // Ignore the Enter that confirms an IME composition (keyCode 229 covers
    // Windows IMEs that clear isComposing on that keydown).
    if (event.nativeEvent.isComposing || event.nativeEvent.keyCode === 229) return;
    // Only claim the key when it will actually send; otherwise let it insert a
    // newline while a response streams or the message is empty/over length.
    if (!canSend) return;
    event.preventDefault();
    event.currentTarget.form?.requestSubmit();
  };
  /*
   * What the next answer will draw on, named. A selected collection narrows
   * the answer to it, so the line says which; Details is where it is switched
   * off (the Knowledge button opens it).
   */
  const chosen = collections.filter((item) => selectedCollections.includes(item.id));
  const sources = temporary
    ? "Temporary"
    : selectedCollections.length
      ? `${selectedCollections.length} source set`
      : "Model knowledge only";
  const hintId = `ai-chat-composer-hint-${ids}`;
  return (
    <form className={temporary ? "ai-chat-composer is-temporary" : "ai-chat-composer"} onSubmit={onSubmit}>
      <div className="ai-chat-composer__box">
        <label className="sr-only" htmlFor={`ai-chat-message-${ids}`}>{composerName}</label>
        <textarea
          aria-describedby={[hintId, messageProblem ? "ai-chat-message-error" : ""].filter(Boolean).join(" ")}
          aria-invalid={messageProblem ? true : undefined}
          className="ai-chat-composer__input"
          id={`ai-chat-message-${ids}`}
          maxLength={8000}
          onChange={(event) => onChange(event.target.value)}
          onKeyDown={submitOnShortcut}
          placeholder="Ask anything, or attach a document to ground the answer in it."
          rows={2}
          value={value}
        />
        {messageProblem && <p className="field-error" id="ai-chat-message-error" role="alert">{messageProblem}</p>}
        <div className="ai-chat-composer__bar">
          <div className="ai-chat-composer__tools">
            <input
              accept={DOCUMENT_ACCEPT}
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
              className="ai-chat-composer__attach"
              onClick={() => fileRef.current?.click()}
              title="Attach a file for Vaelor to read"
              type="button"
            >
              <Icon name="add" size={16} />
            </Button>
            <Button onClick={onOpenDetails} type="button">Knowledge</Button>
            <small title={chosen.map((item) => item.name).join(", ") || undefined}>{sources}</small>
          </div>
          <div className="ai-chat-composer__send-row">
            <small id={hintId}>Press Enter to send, Shift+Enter for a new line.</small>
            <Button
              className="ai-chat-composer__send"
              disabled={!canSend}
              type="submit"
              variant="primary"
            >
              {busy ? "Thinking…" : "Send"}
            </Button>
          </div>
        </div>
      </div>
    </form>
  );
}
