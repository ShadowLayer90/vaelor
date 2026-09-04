import { type FormEvent, type KeyboardEvent, useEffect, useRef, useState } from "react";
import { Button, Textarea } from "./ui";

/**
 * Shared "still working" affordance.
 *
 * Chat had elapsed time and a stop control while an appliance check of the
 * same length showed only a changed button label, so the slowest surface was
 * the one with no sign of life. `onCancel` is optional: a check that cannot be
 * interrupted still shows progress rather than pretending to be instant.
 */
export function AssistantResponseStatus({
  active,
  onCancel,
  label,
  startedAt = 0,
}: {
  active: boolean;
  onCancel?: () => void;
  label?: string;
  /**
   * When the request began, as epoch milliseconds.
   *
   * The counter used to start from the moment this component mounted, and this
   * panel is unmounted whenever the reader looks at another Assistant tab: a
   * request at "110s elapsed" came back reading "3s elapsed" after a round
   * trip to History, so the one number telling somebody how long the appliance
   * has really been working was reset by looking away from it. Zero falls back
   * to mount time for a caller that has no start to hand over.
   */
  startedAt?: number;
}) {
  const [elapsed, setElapsed] = useState(0);
  useEffect(() => {
    if (!active) return;
    const origin = startedAt || Date.now();
    const tick = () => setElapsed(Math.max(0, Math.floor((Date.now() - origin) / 1000)));
    tick();
    const interval = window.setInterval(tick, 1000);
    return () => window.clearInterval(interval);
  }, [active, startedAt]);
  if (!active) return null;
  return (
    <div className="assistant-thinking" role="status">
      <span className="assistant-thinking__dot" /><span className="assistant-thinking__dot" /><span className="assistant-thinking__dot" />
      <div className="assistant-thinking__copy">
        <strong>{label ?? (elapsed < 3 ? "Reading live appliance data" : "Preparing a concise answer")}</strong>
        <small>{elapsed}s elapsed</small>
      </div>
      {onCancel && <Button className="assistant-thinking__stop" onClick={onCancel} type="button" variant="quiet">Stop response</Button>}
    </div>
  );
}

export function AssistantChatComposer({
  blocked = false,
  busy,
  input,
  onChange,
  onSubmit,
  submitLabel = "Ask Vaelor",
}: {
  /**
   * The question as typed cannot be sent yet.
   *
   * Separate from `busy` on purpose. Validation used to be folded into it, so
   * an over-length question disabled the button *and* relabelled it "Thinking…"
   * with nothing in flight — the product claimed to be working on a request it
   * had refused to accept. A blocked composer keeps its own label, which is
   * what tells the reader the send has not happened.
   */
  blocked?: boolean;
  busy: boolean;
  input: string;
  onChange: (value: string) => void;
  onSubmit: (event: FormEvent) => void;
  /**
   * What this submission will actually do.
   *
   * With "Keep this as a check I can re-run" ticked the button posted a
   * reviewable appliance check and answered nothing, while still reading "Ask
   * Vaelor" — the label described the other branch.
   */
  submitLabel?: string;
}) {
  const formRef = useRef<HTMLFormElement>(null);
  const canSend = !busy && !blocked && Boolean(input.trim());

  // Enter sends, Shift+Enter (and IME composition) still insert a newline.
  const onKeyDown = (event: KeyboardEvent<HTMLTextAreaElement>) => {
    if (event.key !== "Enter" || event.shiftKey || event.altKey) return;
    // keyCode 229 covers Windows IMEs that clear isComposing on the
    // composition-confirming Enter.
    if (event.nativeEvent.isComposing || event.nativeEvent.keyCode === 229) return;
    // Only claim the key when it will actually send. Swallowing it while a
    // response streams would stop the user drafting a multi-line follow-up.
    if (!canSend || !formRef.current) return;
    event.preventDefault();
    formRef.current.requestSubmit();
  };

  return (
    <form className="assistant-chat__composer" onSubmit={onSubmit} ref={formRef}>
      <Textarea
        hint="Press Enter to send, Shift+Enter for a new line."
        id="assistant-question"
        label="Ask a question or describe what you want to do"
        maxLength={4000}
        onChange={(event) => onChange(event.target.value)}
        onKeyDown={onKeyDown}
        placeholder="Example: Check my cooling and tell me if anything needs attention."
        rows={3}
        value={input}
      />
      <div>
        <small>Live facts are read automatically. Changes always require a separate approval.</small>
        <Button disabled={!canSend} type="submit" variant="primary">
          {busy ? "Thinking…" : submitLabel}
        </Button>
      </div>
    </form>
  );
}
